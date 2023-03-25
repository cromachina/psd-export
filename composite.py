import logging
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
from PIL import Image
from psd_tools.constants import BlendMode, Clipping, Tag

import blendfuncs

# We use float64 precision to mitigate the effect of dividing group layers by their alpha
dtype = np.float64

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

def intersects(a, b):
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return not (inter[0] >= inter[2] or inter[1] >= inter[3])

def tile_has_data(layer, size, offset):
    offset = swap(offset)
    size = swap(size)
    bbox = offset + (size[0] + offset[0], size[1] + offset[1])
    return intersects(layer.bbox, bbox)

def get_pixel_layer_data(layer, size, offset):
    if not tile_has_data(layer, size, offset):
        return None, None
    color_src = padded_data(layer, 'color', size, offset, layer.offset)
    alpha_src = padded_data(layer, 'shape', size, offset, layer.offset)
    if alpha_src is None:
        alpha_src = np.ones(size + (1,), dtype=dtype)
    return color_src, alpha_src

def get_mask_data(layer, size, offset):
    mask = layer.mask
    if mask and not mask.disabled:
        return padded_data(layer, 'mask', size, offset, (mask.left, mask.top), mask.background_color / 255.0)
    else:
        return np.ones(size + (1,), dtype=dtype)

def clip(color):
    return np.clip(color, 0, 1)

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
    # Left over clip layers, caused by clipping onto no layer.
    for sublayer in clip_stack:
        grouped_layers.append((sublayer, []))
    grouped_layers.reverse()
    return grouped_layers

def safe_divide(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return clip(np.divide(a, b + np.finfo(dtype).eps))

# From Systemax support: The 8 special blend modes use the following layer tag blocks:
# 'tsly' is set to 0
# 'iOpa' is set to the layer opacity
# Actual layer opacity is set to 255
def get_sai_special_mode_opacity(layer):
    blocks = layer._record.tagged_blocks
    tsly = blocks.get(Tag.TRANSPARENCY_SHAPES_LAYER, None)
    iOpa = blocks.get(Tag.BLEND_FILL_OPACITY, None)
    if tsly and iOpa and tsly.data == 0:
        return float(iOpa.data) / 255.0, True
    return layer.opacity / 255.0, False

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

    tile_found = False

    for sublayer, clip_layers in layers:
        if not sublayer.visible:
            continue

        if sublayer.is_group():
            next_backdrop = None
            if sublayer.blend_mode == BlendMode.PASS_THROUGH:
                next_backdrop = (color_dst, alpha_dst)
                pass_color_src, pass_alpha_src = composite_layers(sublayer, size, offset, next_backdrop)
            color_src, alpha_src = composite_layers(sublayer, size, offset)
        else:
            color_src, alpha_src = get_pixel_layer_data(sublayer, size, offset)

        # Empty tile, can just ignore.
        if color_src is None:
            continue

        tile_found = True

        if sublayer.is_group():
            # Un-multiply group composites so that we can multiply group opacity correctly
            color_src = safe_divide(color_src, alpha_src)

        if clip_layers:
            # Composite the clip layers now. This basically overwrites just the color by blending onto it without
            # alpha blending it first. For whatever reason, applying a large root to the alpha source before passing
            # it to clip compositing fixes brightening that can occur with certain blend modes (like multiply).
            corrected_alpha = alpha_src ** 0.0001
            clip_src, _ = composite_layers(clip_layers, size, offset, (color_src, corrected_alpha), True)
            if clip_src is not None:
                color_src = clip_src

        # Opacity is actually FILL when special mode is true!
        opacity, special_mode = get_sai_special_mode_opacity(sublayer)

        # A pass-through layer has already been blended, so just lerp instead.
        if sublayer.blend_mode == BlendMode.PASS_THROUGH:
            mask_src = get_mask_data(sublayer, size, offset)
            mask_src = mask_src * opacity
            color_dst = blendfuncs.lerp(color_dst, pass_color_src, mask_src)
            alpha_dst = blendfuncs.lerp(alpha_dst, pass_alpha_src, mask_src)
            continue

        # Apply opacity (fill) before blending otherwise premultiplied blending of special modes will not work correctly.
        alpha_src = alpha_src * opacity

        # Now we can 'premultiply' the color_src for the main blend operation.
        color_src = color_src * alpha_src

        # Run the blend operation.
        blend_func = blendfuncs.get_blend_func(sublayer.blend_mode, special_mode)
        color_src = blend_func(color_dst, color_src, alpha_dst, alpha_src)

        # Premultiplied blending may cause out-of-range values, so it must be clipped.
        color_src = clip(color_src)

        # We apply the mask last and LERP the blended result onto the destination.
        # Why? Because this is how Photoshop and SAI do it. Applying the mask before blending
        # will yield a slightly different result from those programs.
        mask_src = get_mask_data(sublayer, size, offset)
        color_dst = blendfuncs.lerp(color_dst, color_src, mask_src)

        # Finally we can intersect the mask with the alpha_src and blend the alpha_dst together.
        alpha_src = alpha_src * mask_src
        alpha_dst = blendfuncs.normal_alpha(alpha_dst, alpha_src)

    if not tile_found:
        return None, None

    return color_dst, alpha_dst

debug_path = pathlib.Path('')

def debug_layer(name, offset, data):
    if data.shape[2] == 1:
        data = data.reshape(data.shape[:2])
    image = Image.fromarray((data * 255).astype(np.uint8))
    image.save(debug_path / f'{name}-{offset}.png')

def composite_tile(psd, size, offset, color, alpha):
    tile_color, tile_alpha = composite_layers(psd, size, offset)
    if tile_color is not None:
        blit(color, tile_color, offset)
        blit(alpha, tile_alpha, offset)

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

        tasks = []

        y = 0
        while y < psd.height:
            x = 0
            while x < psd.width:
                size_y = min(tile_height, psd.height - y)
                size_x = min(tile_width, psd.width - x)
                tasks.append(pool.submit(composite_tile, psd, (size_y, size_x), (y, x), color, alpha))
                x += tile_width
            y += tile_height

        # Invoke the result to bubble exceptions. Allows for debugging exceptions in worker threads.
        for task in tasks:
            task.result()

        pool.shutdown(wait=True)
        image = Image.fromarray(np.uint8(color * 255))
        return image

def union_mask_layer(psd, layers, size, offset):
    alpha_dst = np.zeros(size + (1,), dtype=dtype)
    for layer in layers:
        if layer.is_group():
            alpha_src = union_mask_tile(psd, layer, size, offset)
        else:
            _, alpha_src = get_pixel_layer_data(layer, size, offset)
        if alpha_src is not None:
            alpha_dst = blendfuncs.normal_alpha(alpha_dst, alpha_src)
    return alpha_dst

def union_mask_tile(psd, layers, size, offset, alpha):
    tile_alpha = union_mask_layer(psd, layers, size, offset)
    blit(alpha, tile_alpha, offset)

def union_mask(psd, layers, tile_size=(256, 256), worker_count=None):
    if worker_count is None:
        worker_count = psutil.cpu_count(False)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        size = psd.height, psd.width
        alpha = np.zeros(size + (1,), dtype=dtype)
        tile_height, tile_width = tile_size

        tasks = []

        y = 0
        while y < psd.height:
            x = 0
            while x < psd.width:
                size_y = min(tile_height, psd.height - y)
                size_x = min(tile_width, psd.width - x)
                tasks.append(pool.submit(union_mask_tile, psd, layers, (size_y, size_x), (y, x), alpha))
                x += tile_width
            y += tile_height

        for task in tasks:
            task.result()

        pool.shutdown(wait=True)
        image = Image.fromarray(np.uint8(alpha * 255).reshape(alpha.shape[:2]))
        return image
