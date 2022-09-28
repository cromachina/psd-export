import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import numpy as np
import time
import logging

from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import BlendMode

import pathlib

dtype = np.float64

def normal(dst, src, out):
    return src

def multiply(dst, src, out):
    np.multiply(dst, src, out)
    return out

def get_blend_func(blend_mode):
    if blend_mode == BlendMode.NORMAL:
        return normal
    if blend_mode == BlendMode.MULTIPLY:
        return multiply
    return normal

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

def blit(dst, src, offset):
    ox, oy = offset
    dx, dy = dst.shape[:2]
    sx, sy = src.shape[:2]
    d_min_x = clamp(0, dx, ox)
    d_min_y = clamp(0, dy, oy)
    d_max_x = clamp(0, dx, ox + sx)
    d_max_y = clamp(0, dy, oy + sy)
    s_min_x = clamp(0, sx, -ox)
    s_min_y = clamp(0, sy, -oy)
    s_max_x = clamp(0, sx, dx - ox)
    s_max_y = clamp(0, sy, dy - oy)
    dst[d_min_x:d_max_x, d_min_y:d_max_y] = src[s_min_x:s_max_x, s_min_y:s_max_y]

def swap(a):
    x, y = a
    return y, x

cache_lock = threading.Lock()

def get_cached_layer_data(layer, channel):
    attr_name = f'_cache_{channel}'
    with cache_lock:
        if hasattr(layer, attr_name):
            return getattr(layer, attr_name)
        else:
            data = layer.numpy(channel)
            setattr(layer, attr_name, data)
            return data

def padded_data(layer, channel, size, offset):
    data = get_cached_layer_data(layer, channel)
    if data is None:
        return None
    pad = np.zeros(size + data.shape[2:], dtype=data.dtype)
    blit(pad, data, np.array(swap(layer.offset)) - np.array(offset))
    return pad

def divide_and_clip(color, alpha):
    with np.errstate(divide='ignore', invalid='ignore'):
        color = color / alpha
        return np.clip(color, 0, 1)

def all_layers_normal(layer):
    if layer.kind == 'psdimage' or layer.blend_mode == BlendMode.NORMAL:
        for sublayer in layer:
            if sublayer.blend_mode != BlendMode.NORMAL:
                return False
        return True
    return False

def composite_layer(layer, size, offset, backdrop=None):
    if not layer.is_visible():
        return None
    if not layer.is_group():
        color_src = padded_data(layer, 'color', size, offset)
        alpha_src = padded_data(layer, 'shape', size, offset)
        if alpha_src is None:
            alpha_src = np.ones(color_src.shape[:2] + (1,), dtype=color_src.dtype)
        return color_src, alpha_src

    color_dst = np.zeros(size + (3,), dtype=dtype)
    alpha_dst = np.zeros(size + (1,), dtype=dtype)

    if backdrop:
        color_dst, alpha_dst = backdrop

    color_temp = np.empty(size + (3,), dtype=dtype)

    send_backdrop = all_layers_normal(layer)

    for sublayer in layer:
        next_backdrop = None
        if send_backdrop:
            next_backdrop = color_dst, alpha_dst
        data = composite_layer(sublayer, size, offset, next_backdrop)

        if data is None:
            continue
        color_src, alpha_src = data
        alpha_src *= sublayer.opacity / 255.0
        blend_func = get_blend_func(sublayer.blend_mode)

        if not sublayer.is_group():
            color_src *= alpha_src
        blend_src = blend_func(color_dst, color_src, color_temp)

        # A nice optimization, we can skip this if we have a backdrop.
        if not backdrop:
            blend_src = alpha_dst * blend_src + color_src * (1 - alpha_dst)

        color_dst = blend_src + color_dst * (1 - alpha_src)
        alpha_dst = alpha_dst + alpha_src - alpha_dst * alpha_src

    return color_dst, alpha_dst

def composite_mp_viewport(psd, size, offset, color, alpha):
    s = time.perf_counter()
    tile_color, tile_alpha = composite_layer(psd, size, offset)
    logging.info(f'{offset}, composite time {time.perf_counter() - s}')
    blit(color, tile_color, offset)
    blit(alpha, tile_alpha, offset)

def composite(file_name):
    with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as pool:
        psd = PSDImage.open(file_name)
        size = psd.height, psd.width

        color = np.ndarray(size + (3,), dtype=dtype)
        alpha = np.ndarray(size + (1,), dtype=dtype)

        tile_height, tile_width = 512, 512
        y = 0
        while y < psd.height:
            x = 0
            while x < psd.width:
                size_y = min(tile_height, psd.height - y)
                size_x = min(tile_width, psd.width - x)
                pool.submit(composite_mp_viewport, psd, (size_y, size_x), (y, x), color, alpha)
                x += tile_width
            y += tile_height

        pool.shutdown(wait=True)

        image = Image.fromarray(np.uint8(color * 255))

        return image

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('file_name')
    # args = parser.parse_args()
    # file_name = args.file_name
    file_name = 'H:/Art/temp/20220911.psd'
    start = time.perf_counter()
    image = composite(file_name)
    image.save(pathlib.Path(file_name).with_suffix('.png'))
    print(time.perf_counter() - start)
