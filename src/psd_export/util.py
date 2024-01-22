import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import psutil
import psd_tools.api.numpy_io as numpy_io
import psd_tools.constants as ptc
from psd_tools.api.layers import Layer

from . import rle, blendfuncs

file_writer_futures = []

worker_count = psutil.cpu_count(False)
pool = ThreadPoolExecutor(max_workers=worker_count)

def peval(func):
    return asyncio.get_running_loop().run_in_executor(pool, func)

# This is only meant to be used in filters.
def parallel_lerp(a, b, t, out=None):
    if out is None:
        out = np.empty_like(a)
    aa = np.array_split(a, worker_count)
    bb = np.array_split(b, worker_count)
    tt = np.array_split(t, worker_count)
    oo = np.array_split(out, worker_count)
    tasks = []
    for ai, bi, ti, oi in zip(aa, bb, tt, oo):
        tasks.append(pool.submit(blendfuncs.lerp, ai, bi, ti, out=oi))
    for task in tasks:
        task.result()
    return out

def swap(a):
    x, y = a
    return y, x

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

def full(shape, fill, dtype=None):
    if fill == 0:
        return np.zeros(shape, dtype=dtype)
    elif fill == 1:
        return np.ones(shape, dtype=dtype)
    else:
        return np.full(shape, fill_value=fill, dtype=dtype)

def is_grayscale(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    return (r == g).all() and (g == b).all()

def save_worker(file_name, image, imwrite_args=[]):
    image = np.dstack(image)
    image *= 255
    image = image.astype(np.uint8)
    if (image[:,:,3] == 255).all():
        if is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(file_name, image, imwrite_args)
    logging.info(f'exported: {file_name}')

def save_file(file_name, image, imwrite_args=[]):
    file_writer_futures.append(peval(lambda: save_worker(str(file_name), image, imwrite_args)))

async def save_workers_wait_all():
    await asyncio.gather(*file_writer_futures)
    file_writer_futures.clear()

async def layer_numpy(layer:Layer, channel=None):
    if channel == 'mask' and not layer.mask:
        return None

    depth = layer._psd.depth
    version = layer._psd.version

    def channel_matches(info):
        if channel == 'color':
            return info.id >= 0
        if channel == 'shape':
            return info.id == ptc.ChannelID.TRANSPARENCY_MASK
        if channel == 'mask':
            if not layer.mask:
                return False
            if layer.mask._has_real():
                return info.id == ptc.ChannelID.REAL_USER_LAYER_MASK
            else:
                return info.id == ptc.ChannelID.USER_LAYER_MASK
        else:
            raise ValueError(f'Unknown channel type: {channel}')

    channels = zip(layer._channels, layer._record.channel_info)
    channels = [channel for channel, info in channels if channel_matches(info)]

    if len(channels) == 0:
        return None

    # Use the psd-tools path if we are not decoding RLE
    if not all([channel.compression == ptc.Compression.RLE for channel in channels]):
        return await peval(lambda: layer.numpy(channel))

    if channel == 'mask':
        width, height = layer.mask.width, layer.mask.height
    else:
        width, height = layer.width, layer.height

    decoded = []
    for channel in channels:
        decoded.append(peval(lambda channel=channel: numpy_io._parse_array(rle.decode_rle(channel.data, width, height, depth, version), depth)))
    decoded = await asyncio.gather(*decoded)

    return await peval(lambda: np.stack(decoded, axis=1).reshape((height, width, -1)))
