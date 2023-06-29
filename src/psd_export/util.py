import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numba
import numpy as np
import psd_tools.api.numpy_io as numpy_io
import psd_tools.constants as ptc
import psutil
from psd_tools.api.layers import Layer

file_writer_futures = []

worker_count = psutil.cpu_count(False)
pool = ThreadPoolExecutor(max_workers=worker_count)

def peval(func):
    return asyncio.get_running_loop().run_in_executor(pool, func)

def swap(a):
    x, y = a
    return y, x

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

def clip(val, out=None):
    return np.clip(val, 0, 1, out=out)

def clip_in(val):
    return clip(val, val)

def safe_divide(a, b, out=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = np.finfo(a.dtype).eps
        return np.divide(a, b + eps, out=out)

def clip_divide(a, b, out=None):
    out = safe_divide(a, b, out=out)
    return clip_in(out)

def lerp(a, b, t, out=None):
    out = np.subtract(b, a, out=out)
    np.multiply(t, out, out=out)
    return np.add(a, out, out=out)

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

@numba.njit(cache=True, nogil=True)
def decode(counts, rows, result):
    row_offset = 0
    for i in range(counts.shape[0]):
        count = counts[i]
        row_view = rows[row_offset:row_offset + count]
        result_view = result[i]

        src = np.int32(0)
        dst = np.int32(0)
        data_size = row_view.shape[0]
        result_size = result_view.shape[0]
        while src < data_size:
            header = np.int32(row_view.view(np.byte)[src])
            src = src + 1
            if 0 <= header <= 127:
                length = header + 1
                src_next = src + length
                dst_next = dst + length
                if src_next <= data_size and dst_next <= result_size:
                    result_view[dst:dst_next] = row_view[src:src_next]
                    src = src_next
                    dst = dst_next
                else:
                    raise ValueError('Invalid RLE compression')
            elif header == -128:
                pass
            else:
                length = 1 - header
                src_next = src + 1
                dst_next = dst + length
                if src_next <= data_size and dst_next <= result_size:
                    result_view[dst:dst_next] = row_view[src]
                    src = src_next
                    dst = dst_next
                else:
                    raise ValueError('Invalid RLE compression')
        if dst < result_size:
            raise ValueError('dst < result_size', dst, result_size, src, data_size)

        row_offset += count

def decode_rle(data, width, height, depth, version):
    row_size = max(width * depth // 8, 1)
    dtype = (np.uint16, np.uint32)[version - 1]
    counts = np.frombuffer(data, dtype=dtype, count=height).copy()
    if sys.byteorder == 'little':
        counts.byteswap(inplace=True)
    rows = np.frombuffer(data, dtype=np.ubyte, offset=counts.nbytes)
    result = np.empty((height, row_size), dtype=np.ubyte)
    decode(counts, rows, result)
    return result

def layer_numpy(layer:Layer, channel=None, real_mask=True):
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
            if layer.mask._has_real() and real_mask:
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
        return layer.numpy(channel, real_mask)

    if channel == 'mask':
        width, height = layer.mask.width, layer.mask.height
    else:
        width, height = layer.width, layer.height

    decoded = []
    for channel in channels:
        data = decode_rle(channel.data, width, height, depth, version)
        data = numpy_io._parse_array(data, depth)
        decoded.append(data)

    return np.stack(decoded, axis=1).reshape((height, width, -1))
