import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
from PIL import Image
from psd_tools.constants import BlendMode, Clipping

import blendfuncs

dtype = np.float32

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

def padded_data(layer, channel, size, offset, data_offset, fill=0):
    data = get_cached_layer_data(layer, channel)
    if data is None:
        return None
    pad = np.full(size + data.shape[2:], fill_value=fill, dtype=dtype)
    blit(pad, data, np.array(swap(data_offset)) - np.array(offset))
    return pad

def get_pixel_layer_data(layer, size, offset):
    color_src = padded_data(layer, 'color', size, offset, layer.offset)
    alpha_src = padded_data(layer, 'shape', size, offset, layer.offset)
    if alpha_src is None:
        alpha_src = np.ones(color_src.shape[:2] + (1,), dtype=dtype)
    return color_src, alpha_src

def get_mask_data(layer, size, offset):
    mask = layer.mask
    if mask and not mask.disabled:
        return padded_data(layer, 'mask', size, offset, (mask.left, mask.top), 1)
    return None

def clip(color):
    np.clip(color, 0, 1, out=color)

def composite_layer(layer, size, offset, backdrop=None):
    if not layer.is_group():
        return get_pixel_layer_data(layer, size, offset)

    color_dst = None
    alpha_dst = None

    if backdrop:
        color_dst, alpha_dst = backdrop
    else:
        color_dst = np.zeros(size + (3,), dtype=dtype)
        alpha_dst = np.zeros(size + (1,), dtype=dtype)

    previous_non_clip_alpha_src = None

    for sublayer in layer:
        if not sublayer.visible:
            continue

        next_backdrop = None
        if sublayer.blend_mode == BlendMode.PASS_THROUGH:
            next_backdrop = (color_dst, alpha_dst)

        color_src, alpha_src = composite_layer(sublayer, size, offset, next_backdrop)
        mask_src = get_mask_data(sublayer, size, offset)
        alpha_src *= sublayer.opacity / 255.0
        if mask_src is not None:
            alpha_src *= mask_src

        if sublayer._record.clipping == Clipping.NON_BASE:
            if previous_non_clip_alpha_src is not None:
                alpha_src *= previous_non_clip_alpha_src
        else:
            previous_non_clip_alpha_src = alpha_src

        if sublayer.is_group():
            if mask_src is not None:
                color_src *= mask_src
        else:
            color_src *= alpha_src

        blend_func = blendfuncs.get_blend_func(sublayer.blend_mode)
        color_dst = blend_func(color_dst, color_src, alpha_dst, alpha_src)
        alpha_dst = blendfuncs.normal_alpha(alpha_dst, alpha_src)

        clip(color_dst)

    return color_dst, alpha_dst

def composite_tile(psd, size, offset, color, alpha):
    try:
        tile_color, tile_alpha = composite_layer(psd, size, offset)
        blit(color, tile_color, offset)
        blit(alpha, tile_alpha, offset)
    except Exception as e:
        logging.exception(e)

def composite(psd):
    with ThreadPoolExecutor(max_workers=psutil.cpu_count(False)) as pool:
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
                pool.submit(composite_tile, psd, (size_y, size_x), (y, x), color, alpha)
                x += tile_width
            y += tile_height

        pool.shutdown(wait=True)
        image = Image.fromarray(np.uint8(color * 255))
        return image
