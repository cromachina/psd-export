import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
from PIL import Image
from psd_tools.constants import BlendMode, Clipping, Tag

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
        alpha_src = np.ones(size + (1,), dtype=dtype)
    return color_src, alpha_src

def get_mask_data(layer, size, offset):
    mask = layer.mask
    if mask and not mask.disabled:
        return padded_data(layer, 'mask', size, offset, (mask.left, mask.top), 1)
    else:
        return np.ones(size + (1,), dtype=dtype)

def clip(color):
    np.clip(color, 0, 1, out=color)

def is_clipping(layer):
    return layer._record.clipping == Clipping.NON_BASE

def get_layer_and_clip_groupings(layers):
    grouped_layers = []
    clip_stack = []
    for layer in reversed(layers):
        if is_clipping(layer):
            clip_stack.append(layer)
        else:
            clip_stack.reverse()
            if layer.blend_mode == BlendMode.PASS_THROUGH:
                grouped_layers.append((layer, []))
                for sublayer in clip_stack:
                    grouped_layers.append((sublayer, []))
            else:
                grouped_layers.append((layer, clip_stack))
            clip_stack = []
    for sublayer in reversed(clip_stack):
        grouped_layers.append((sublayer, []))
    grouped_layers.reverse()
    return grouped_layers

def safe_divide(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(a, b, out=a)
        clip(a)

# From Systemax support: The 8 special blend modes use the following layer tag blocks:
# 'tsly' is set to 0
# 'iOpa' is set to the layer opacity
# Actual layer opacity is set to 255
def get_sai_special_mode_opacity(layer):
    blocks = layer._record.tagged_blocks
    tsly = blocks.get(Tag.TRANSPARENCY_SHAPES_LAYER, None)
    iOpa = blocks.get(Tag.BLEND_FILL_OPACITY, None)
    if tsly and iOpa and tsly.data == 0:
        return float(iOpa.data), True
    return layer.opacity, False

def composite_layers(layers, size, offset, backdrop=None, clip_mode=False):
    if backdrop:
        color_dst, alpha_dst = backdrop
    else:
        color_dst = np.zeros(size + (3,), dtype=dtype)
        alpha_dst = np.zeros(size + (1,), dtype=dtype)

    if clip_mode:
        layers = map(lambda l: (l, []), layers)
    else:
        layers = get_layer_and_clip_groupings(layers)

    for sublayer, clip_layers in layers:
        if not sublayer.visible:
            continue

        if sublayer.is_group():
            next_backdrop = None
            if sublayer.blend_mode == BlendMode.PASS_THROUGH:
                next_backdrop = (color_dst, alpha_dst)
            color_src, alpha_src = composite_layers(sublayer, size, offset, next_backdrop)
        else:
            color_src, alpha_src = get_pixel_layer_data(sublayer, size, offset)

        mask_src = get_mask_data(sublayer, size, offset)
        opacity, special_mode = get_sai_special_mode_opacity(sublayer)
        alpha_src *= opacity / 255.0
        alpha_src *= mask_src

        if sublayer.is_group():
            color_src *= mask_src

        if clip_layers:
            # Why does this work!? If I don't do this, then blend modes like multiply, darken, etc. will BRIGHTEN
            # areas at the edge of large transparency gradients. I was only able to figure it out after
            # randomly trying to multiply the alpha_src by itself, then sqrt, and obvserve it getting brighter
            # and darker respectively. I then tried successively larger roots until it looked sufficently like
            # SAI's output.
            corrected_alpha = np.float_power(alpha_src, 1/10000)
            color_src, _ = composite_layers(clip_layers, size, offset, (color_src, corrected_alpha), True)

        if not sublayer.is_group():
            color_src *= alpha_src

        blend_func = blendfuncs.get_blend_func(sublayer.blend_mode)
        color_dst = blend_func(color_dst, color_src, alpha_dst, alpha_src)
        alpha_dst = blendfuncs.normal_alpha(alpha_dst, alpha_src)

        clip(color_dst)

    return color_dst, alpha_dst

debug_path = ''

def debug_layer(name, offset, data):
    if data.shape[2] == 1:
        data = data.reshape(data.shape[:2])
    image = Image.fromarray((data * 255).astype(np.uint8))
    image.save(debug_path / f'{name}-{offset}.png')

def composite_tile(psd, size, offset, color, alpha):
    try:
        tile_color, tile_alpha = composite_layers(psd, size, offset)
        blit(color, tile_color, offset)
        blit(alpha, tile_alpha, offset)
    except Exception as e:
        logging.exception(e)

def composite(psd, tile_size=(256,256), worker_count=None):
    '''
    Composite the given PSD and return an PIL image.
    `tile_size` is arranged by (height, width)
    `worker_count` is for multi-threading. Set to None to use the number of physical processors
    '''
    if worker_count is None:
        worker_count = psutil.cpu_count(False)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        size = psd.height, psd.width
        color = np.ndarray(size + (3,), dtype=dtype)
        alpha = np.ndarray(size + (1,), dtype=dtype)
        tile_height, tile_width = tile_size

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
