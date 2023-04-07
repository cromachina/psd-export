import asyncio
import pathlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import psutil
from psd_tools.api.layers import Layer
from psd_tools.constants import BlendMode, Clipping, Tag

import blendfuncs
import util

worker_count = psutil.cpu_count(False)
pool = ThreadPoolExecutor(max_workers=worker_count)

def peval(func):
    return asyncio.get_running_loop().run_in_executor(pool, func)

async def barrier_skip(barrier):
    try:
        await asyncio.wait_for(asyncio.shield(barrier.wait()), timeout=0.0)
    except TimeoutError:
        pass

def mosaic_image(image, mosaic_factor):
    original_size = util.swap(image.shape[:2])
    min_dim = min(original_size) // mosaic_factor
    min_dim = max(4, min_dim)
    scale_dimension = (original_size[0] // min_dim, original_size[1] // min_dim)
    mosaic_image = cv2.resize(image, scale_dimension, interpolation=cv2.INTER_AREA)
    return cv2.resize(mosaic_image, original_size, interpolation=cv2.INTER_NEAREST)

def mosaic_op(color, alpha, mosaic_factor=100):
    return mosaic_image(color, mosaic_factor), alpha

def set_custom_operation(layer, func):
    layer._custom_op = func

def get_custom_operation(layer):
    return layer._custom_op if hasattr(layer, '_custom_op') else None

def get_overlap_tiles(dst, src, offset):
    ox, oy = offset
    dx, dy = dst.shape[:2]
    sx, sy = src.shape[:2]
    d_min_x = util.clamp(0, dx, ox)
    d_min_y = util.clamp(0, dy, oy)
    d_max_x = util.clamp(0, dx, ox + sx)
    d_max_y = util.clamp(0, dy, oy + sy)
    s_min_x = util.clamp(0, sx, -ox)
    s_min_y = util.clamp(0, sy, -oy)
    s_max_x = util.clamp(0, sx, dx - ox)
    s_max_y = util.clamp(0, sy, dy - oy)
    return dst[d_min_x:d_max_x, d_min_y:d_max_y], src[s_min_x:s_max_x, s_min_y:s_max_y]

def blit(dst, src, offset):
    dst, src = get_overlap_tiles(dst, src, offset)
    np.copyto(dst, src)

def get_visibility_all_children(layer:Layer, visited):
    if layer.visible:
        visited.add(id(layer))
    if layer.is_group():
        for sublayer in layer:
            get_visibility_all_children(sublayer, visited)
    return visited

def get_visibility_dependency_sub(layer:Layer, visited):
    if id(layer) in visited:
        return
    found = False
    for sublayer in reversed(layer.parent):
        if sublayer == layer:
            found = True
        if found:
            if sublayer.visible:
                get_visibility_all_children(sublayer, visited)
    if layer.parent.kind != 'psdimage' and layer.parent.blend_mode == BlendMode.PASS_THROUGH:
        get_visibility_dependency_sub(layer.parent, visited)

def get_visibility_dependency(layer:Layer, clip_layers:list[Layer]):
    visited = set()
    get_visibility_dependency_sub(layer, visited)
    for sublayer in clip_layers:
        get_visibility_all_children(sublayer, visited)
    return frozenset(visited)

def get_tile_cache(layer:Layer, offset):
    if not hasattr(layer, '_composite_cache'):
        layer._composite_cache = {}
    if offset not in layer._composite_cache:
        layer._composite_cache[offset] = {}
    return layer._composite_cache[offset]

def set_layer_extra_data(layer, tile_count, size):
    for sublayer in layer.descendants():
        sublayer._composite_cache_lock = asyncio.Lock()
        sublayer._data_cache_lock = asyncio.Lock()
        if get_custom_operation(sublayer) is not None:
            sublayer._custom_op_barrier = asyncio.Barrier(tile_count)
            sublayer._custom_op_condition = asyncio.Condition()
            sublayer._custom_op_finished = False
            sublayer._custom_op_color = np.zeros(size + (3,))
            sublayer._custom_op_alpha = np.zeros(size + (1,))

async def get_cached_composite(layer:Layer, offset, visibility_dependency):
    async with layer._composite_cache_lock:
        return get_tile_cache(layer, offset).get(visibility_dependency, None)

async def set_cached_composite(layer:Layer, offset, visibilty_dependency, tile_data):
    async with layer._composite_cache_lock:
        get_tile_cache(layer, offset)[visibilty_dependency] = tile_data

async def get_cached_layer_data(layer:Layer, channel):
    attr_name = f'_cache_{channel}'
    async with layer._data_cache_lock:
        if not hasattr(layer, attr_name):
            setattr(layer, attr_name, layer.numpy(channel))
        return getattr(layer, attr_name)

async def get_padded_data(layer, channel, size, offset, data_offset, fill=0.0):
    data = await get_cached_layer_data(layer, channel)
    if data is None:
        return None
    shape = size + data.shape[2:]
    pad = await peval(lambda: util.full(shape, fill))
    await peval(lambda: blit(pad, data, np.array(util.swap(data_offset)) - np.array(offset)))
    return pad

def intersection(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))

def is_intersecting(a, b):
    inter = intersection(a, b)
    return not (inter[0] >= inter[2] or inter[1] >= inter[3])

def make_bbox(size, offset):
    offset = util.swap(offset)
    size = util.swap(size)
    return offset + (size[0] + offset[0], size[1] + offset[1])

def has_tile_data(layer, size, offset):
    return is_intersecting(layer.bbox, make_bbox(size, offset))

async def is_zero_alpha(layer, size, offset):
    data = await get_cached_layer_data(layer, 'shape')
    if data is None:
        return False
    bbox = intersection(layer.bbox, make_bbox(size, offset))
    offset = layer.offset
    bbox = (bbox[0] - offset[0], bbox[1] - offset[1], bbox[2] - offset[0], bbox[3] - offset[1])
    return await peval(lambda: not data[bbox[1]:bbox[3], bbox[0]:bbox[2]].any())

async def get_pixel_layer_data(layer, size, offset):
    if not has_tile_data(layer, size, offset):
        return None, None
    if await is_zero_alpha(layer, size, offset):
        return None, None
    alpha_src = await get_padded_data(layer, 'shape', size, offset, layer.offset)
    if alpha_src is None:
        alpha_src = np.ones(size + (1,))
    color_src = await get_padded_data(layer, 'color', size, offset, layer.offset)
    return color_src, alpha_src

async def get_mask_data(layer, size, offset):
    mask = layer.mask
    if mask and not mask.disabled:
        return await get_padded_data(layer, 'mask', size, offset, (mask.left, mask.top), mask.background_color / 255.0)
    else:
        return None

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
                for sublayer in clip_stack:
                    grouped_layers.append((sublayer, []))
                grouped_layers.append((layer, []))
            else:
                grouped_layers.append((layer, clip_stack))
            clip_stack = []
    # Left over clip layers, caused by clipping onto no layer.
    for sublayer in clip_stack:
        grouped_layers.append((sublayer, []))
    grouped_layers.reverse()
    return grouped_layers

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

async def composite_layers(layers, size, offset, backdrop=None, clip_mode=False):
    if backdrop:
        color_dst, alpha_dst = backdrop
    else:
        color_dst = 0.0
        alpha_dst = 0.0

    if clip_mode:
        layers = map(lambda l: (l, []), layers)
    else:
        layers = get_layer_and_clip_groupings(layers)

    tile_found = False

    for sublayer, clip_layers in layers:
        custom_op = get_custom_operation(sublayer)

        if not sublayer.visible:
            if custom_op is not None:
                await barrier_skip(sublayer._custom_op_barrier)
            continue

        visibility_dependency = get_visibility_dependency(sublayer, clip_layers)
        cached_composite = await get_cached_composite(sublayer, offset, visibility_dependency)
        if cached_composite is not None:
            color_dst, alpha_dst = cached_composite
            tile_found = True
            if custom_op is not None:
                await barrier_skip(sublayer._custom_op_barrier)
            continue

        if sublayer.is_group():
            next_backdrop = None
            if sublayer.blend_mode == BlendMode.PASS_THROUGH:
                next_backdrop = (color_dst, alpha_dst)
            color_src, alpha_src = await composite_layers(sublayer, size, offset, next_backdrop)
        else:
            color_src, alpha_src = await get_pixel_layer_data(sublayer, size, offset)

        # Empty tile, can just ignore.
        if color_src is None:
            if custom_op is not None:
                await barrier_skip(sublayer._custom_op_barrier)
            continue

        tile_found = True

        # Perform custom filter over the current color_dst
        if custom_op is not None:
            if not np.isscalar(color_dst):
                await peval(lambda: blit(sublayer._custom_op_color, color_dst, offset))
                await peval(lambda: blit(sublayer._custom_op_alpha, alpha_src, offset))
            await sublayer._custom_op_barrier.wait()
            if not sublayer._custom_op_condition.locked():
                if not sublayer._custom_op_finished:
                    async with sublayer._custom_op_condition:
                        sublayer._custom_op_color, sublayer._custom_op_alpha = await peval(lambda: custom_op(sublayer._custom_op_color, sublayer._custom_op_alpha))
                        sublayer._custom_op_finished = True
                        sublayer._custom_op_condition.notify_all()
            else:
                async with sublayer._custom_op_condition:
                    await sublayer._custom_op_condition.wait_for(lambda: sublayer._custom_op_finished)
            neg_offset = -np.array(offset)
            await peval(lambda: blit(color_src, sublayer._custom_op_color, neg_offset))
            await peval(lambda: blit(alpha_src, sublayer._custom_op_alpha, neg_offset))

        # Opacity is actually FILL when special mode is true!
        opacity, special_mode = get_sai_special_mode_opacity(sublayer)

        # A pass-through layer has already been blended, so just lerp instead.
        # NOTE: Clipping layers do not apply to pass layers, as if clipping were simply disabled.
        if sublayer.blend_mode == BlendMode.PASS_THROUGH:
            mask_src = await get_mask_data(sublayer, size, offset)
            if mask_src is None:
                mask_src = opacity
            else:
                await peval(lambda: np.multiply(mask_src, opacity, out=mask_src))
            if np.isscalar(mask_src) and mask_src == 1.0:
                color_dst = color_src
                alpha_dst = alpha_src
            else:
                await peval(lambda: util.lerp(color_dst, color_src, mask_src, out=color_dst))
                await peval(lambda: util.lerp(alpha_dst, alpha_src, mask_src, out=alpha_dst))
            await set_cached_composite(sublayer, offset, visibility_dependency, (color_dst, alpha_dst))
            continue

        if sublayer.is_group():
            # Un-multiply group composites so that we can multiply group opacity correctly
            await peval(lambda: util.safe_divide(color_src, alpha_src, out=color_src))
            await peval(lambda: util.clip(color_src, out=color_src))

        if clip_layers:
            # Composite the clip layers now. This basically overwrites just the color by blending onto it without
            # alpha blending it first. For whatever reason, applying a large root to the alpha source before passing
            # it to clip compositing fixes brightening that can occur with certain blend modes (like multiply).
            corrected_alpha = await peval(lambda: alpha_src ** 0.0001)
            clip_src, _ = await composite_layers(clip_layers, size, offset, (color_src, corrected_alpha), True)
            if clip_src is not None:
                color_src = clip_src

        # Apply opacity (fill) before blending otherwise premultiplied blending of special modes will not work correctly.
        await peval(lambda: np.multiply(alpha_src, opacity, out=alpha_src))

        # Now we can 'premultiply' the color_src for the main blend operation.
        await peval(lambda: np.multiply(color_src, alpha_src, out=color_src))

        # Run the blend operation.
        blend_func = blendfuncs.get_blend_func(sublayer.blend_mode, special_mode)
        color_src = await peval(lambda: blend_func(color_dst, color_src, alpha_dst, alpha_src))

        # Premultiplied blending may cause out-of-range values, so it must be clipped.
        if sublayer.blend_mode != BlendMode.NORMAL:
            await peval(lambda: util.clip(color_src, out=color_src))

        # We apply the mask last and LERP the blended result onto the destination.
        # Why? Because this is how Photoshop and SAI do it. Applying the mask before blending
        # will yield a slightly different result from those programs.
        mask_src = await get_mask_data(sublayer, size, offset)
        if mask_src is not None:
            color_dst = await peval(lambda: util.lerp(color_dst, color_src, mask_src))
        else:
            color_dst = color_src

        # Finally we can intersect the mask with the alpha_src and blend the alpha_dst together.
        if mask_src is not None:
            await peval(lambda: np.multiply(alpha_src, mask_src, out=alpha_src))
        alpha_dst = await peval(lambda: blendfuncs.normal_alpha(alpha_dst, alpha_src))

        await set_cached_composite(sublayer, offset, visibility_dependency, (color_dst, alpha_dst))

    if not tile_found:
        return None, None

    return color_dst, alpha_dst

debug_path = pathlib.Path('')

def debug_layer(name, offset, data):
    if data.shape[2] == 1:
        data = data.reshape(data.shape[:2])
    data = (data * 255).astype(np.uint8)
    data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    path = debug_path / f'{name}-{offset}.png'
    cv2.imwrite(str(path), data)

async def composite_tile(psd, size, offset, color, alpha):
    tile_color, tile_alpha = await composite_layers(psd, size, offset)
    if tile_color is not None:
        await peval(lambda: blit(color, tile_color, offset))

def generate_tiles(size, tile_size):
    height, width = size
    tile_height, tile_width = tile_size
    y = 0
    while y < height:
        x = 0
        while x < width:
            size_y = min(tile_height, height - y)
            size_x = min(tile_width, width - x)
            yield ((size_y, size_x), (y, x))
            x += tile_width
        y += tile_height

def composite(psd, tile_size=(256,256)):
    '''
    Composite the given PSD and return a numpy array.
    `tile_size` is arranged by (height, width)
    '''
    size = psd.height, psd.width
    color = np.ndarray(size + (3,))
    alpha = None

    tasks = []

    for (tile_size, tile_offset) in generate_tiles(size, tile_size):
        tasks.append(composite_tile(psd, tile_size, tile_offset, color, alpha))

    set_layer_extra_data(psd, len(tasks), size)

    async def run():
        await asyncio.gather(*tasks)

    asyncio.run(run())

    return color
