from __future__ import annotations

import asyncio
import pathlib
import threading

import cv2
import numpy as np
from psd_tools import PSDImage
from psd_tools.api.layers import Layer
from psd_tools.constants import BlendMode, Clipping, Tag

import blendfuncs
import util
from util import peval

class WrappedLayer():
    def __init__(self, layer:Layer, clip_layers=[], parent:WrappedLayer=None):
        self.layer:Layer = layer
        self.parent:WrappedLayer = parent
        self.name = layer.name
        self.visible = True if layer.kind == 'psdimage' else layer.visible
        self.custom_op = None
        self.children:list[WrappedLayer] = []
        self.flat_children:list[WrappedLayer] = []
        self.clip_layers:list[WrappedLayer] = clip_layers
        self.composite_cache = {}
        self.data_cache_lock = threading.Lock()
        self.tags = []

        if self.layer.is_group():
            for sublayer, sub_clip_layers in get_layer_and_clip_groupings(self.layer):
                sub_clip_layers = [WrappedLayer(s, parent=self) for s in sub_clip_layers]
                sublayer = WrappedLayer(sublayer, sub_clip_layers, self)
                self.children.append(sublayer)
                self.flat_children.append(sublayer)
                self.flat_children.extend(sub_clip_layers)

    def __iter__(self):
        return iter(self.children)

    def __reversed__(self):
        return reversed(self.children)

    def descendants(self):
        for group in (self.children, self.clip_layers):
            for child in group:
                yield child
                for subchild in child.descendants():
                    yield subchild

def print_layers(layer:WrappedLayer, tab='', next_tab='  '):
    name = layer.name
    if is_clipping(layer.layer):
        name = '* ' + name
    print(tab, name)
    for child in reversed(layer.flat_children):
        print_layers(child, tab + next_tab)

def PSDOpen(fp, **kwargs):
    return WrappedLayer(PSDImage.open(fp, **kwargs))

async def barrier_skip(barrier:asyncio.Barrier):
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

mosaic_factor_default = 100

def mosaic_op(color, alpha, mosaic_factor=None, *_):
    if mosaic_factor is None:
        mosaic_factor = mosaic_factor_default
    mosaic_factor = int(mosaic_factor)
    return mosaic_image(color, mosaic_factor), alpha

def blur_op(color, alpha, size=50, *_):
    size = int(size)
    if size % 2 == 0:
        size += 1
    return cv2.GaussianBlur(color, (size, size), size), alpha

def chain_ops(ops):
    if not ops:
        return None
    def c(color, alpha):
        for op in ops:
            color, alpha = op(color, alpha)
        return color, alpha
    return c

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

def get_visibility_all_children(layer:WrappedLayer, visited):
    if layer.visible:
        visited.add(id(layer))
    if layer.layer.is_group():
        for sublayer in layer:
            get_visibility_all_children(sublayer, visited)
    return visited

def get_visibility_dependency_sub(layer:WrappedLayer, visited):
    if layer.parent == None:
        return
    if id(layer) in visited:
        return
    found = False
    for sublayer in reversed(layer.parent.flat_children):
        if sublayer == layer:
            found = True
        if found:
            if sublayer.visible:
                get_visibility_all_children(sublayer, visited)
    if layer.parent.parent != None and layer.parent.layer.blend_mode == BlendMode.PASS_THROUGH:
        get_visibility_dependency_sub(layer.parent, visited)

def get_visibility_dependency(layer:WrappedLayer):
    visited = set()
    get_visibility_dependency_sub(layer, visited)
    for sublayer in layer.clip_layers:
        get_visibility_all_children(sublayer, visited)
    return frozenset(visited)

def get_tile_cache(layer:WrappedLayer, offset):
    if offset not in layer.composite_cache:
        layer.composite_cache[offset] = {}
    return layer.composite_cache[offset]

def set_layer_extra_data(layer:WrappedLayer, tile_count, size):
    for sublayer in layer.descendants():
        if sublayer.visible:
            sublayer.visibility_dependency = get_visibility_dependency(sublayer)
        if sublayer.custom_op is not None:
            sublayer.custom_op_barrier = asyncio.Barrier(tile_count)
            sublayer.custom_op_condition = asyncio.Condition()
            sublayer.custom_op_finished = False
            sublayer.custom_op_color = np.zeros(size + (3,))
            sublayer.custom_op_alpha = np.zeros(size + (1,))

def get_cached_composite(layer:WrappedLayer, offset):
    return get_tile_cache(layer, offset).get(layer.visibility_dependency, None)

def set_cached_composite(layer:WrappedLayer, offset, tile_data):
    get_tile_cache(layer, offset)[layer.visibility_dependency] = tile_data

def get_cached_layer_data(layer:WrappedLayer, channel):
    attr_name = f'cache_{channel}'
    with layer.data_cache_lock:
        if not hasattr(layer, attr_name):
            data = layer.layer.numpy(channel)
            if data is not None:
                data = data.astype(np.float64)
            setattr(layer, attr_name, data)
        return getattr(layer, attr_name)

async def get_padded_data(layer:WrappedLayer, channel, size, offset, data_offset, fill=0.0):
    data = await peval(lambda: get_cached_layer_data(layer, channel))
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

def has_tile_data(layer:Layer, size, offset):
    return is_intersecting(layer.bbox, make_bbox(size, offset))

async def is_zero_alpha(layer:WrappedLayer, size, offset):
    data = await peval(lambda: get_cached_layer_data(layer, 'shape'))
    if data is None:
        return False
    bbox = intersection(layer.layer.bbox, make_bbox(size, offset))
    offset = layer.layer.offset
    bbox = (bbox[0] - offset[0], bbox[1] - offset[1], bbox[2] - offset[0], bbox[3] - offset[1])
    return await peval(lambda: not data[bbox[1]:bbox[3], bbox[0]:bbox[2]].any())

async def get_pixel_layer_data(layer:WrappedLayer, size, offset):
    if not has_tile_data(layer.layer, size, offset):
        return None, None
    if await is_zero_alpha(layer, size, offset):
        return None, None
    alpha_src = await get_padded_data(layer, 'shape', size, offset, layer.layer.offset)
    if alpha_src is None:
        alpha_src = np.ones(size + (1,))
    color_src = await get_padded_data(layer, 'color', size, offset, layer.layer.offset)
    return color_src, alpha_src

async def get_mask_data(layer:WrappedLayer, size, offset):
    mask = layer.layer.mask
    if mask and not mask.disabled:
        return await get_padded_data(layer, 'mask', size, offset, (mask.left, mask.top), mask.background_color / 255.0)
    else:
        return None

def is_clipping(layer:Layer):
    return layer.kind != 'psdimage' and layer._record.clipping == Clipping.NON_BASE

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
def get_sai_special_mode_opacity(layer:Layer):
    blocks = layer._record.tagged_blocks
    tsly = blocks.get(Tag.TRANSPARENCY_SHAPES_LAYER, None)
    iOpa = blocks.get(Tag.BLEND_FILL_OPACITY, None)
    if tsly and iOpa and tsly.data == 0:
        return float(iOpa.data) / 255.0, True
    return layer.opacity / 255.0, False

async def composite_group_layer(layer:WrappedLayer, size, offset, backdrop=None):
    if backdrop:
        color_dst, alpha_dst = backdrop
    else:
        color_dst = 0.0
        alpha_dst = 0.0

    tile_found = False

    sublayer:WrappedLayer
    for sublayer in layer:
        if not sublayer.visible:
            if sublayer.custom_op is not None:
                await barrier_skip(sublayer.custom_op_barrier)
            continue

        cached_composite = get_cached_composite(sublayer, offset)
        if cached_composite is not None:
            color_dst, alpha_dst = cached_composite
            tile_found = True
            if sublayer.custom_op is not None:
                await barrier_skip(sublayer.custom_op_barrier)
            continue

        blend_mode = sublayer.layer.blend_mode

        if sublayer.layer.is_group():
            next_backdrop = None
            if blend_mode == BlendMode.PASS_THROUGH:
                next_backdrop = (color_dst, alpha_dst)
            color_src, alpha_src = await composite_group_layer(sublayer, size, offset, next_backdrop)
        else:
            color_src, alpha_src = await get_pixel_layer_data(sublayer, size, offset)

        # Empty tile, can just ignore.
        if color_src is None:
            if sublayer.custom_op is not None:
                await barrier_skip(sublayer.custom_op_barrier)
            continue

        tile_found = True

        # Perform custom filter over the current color_dst
        if sublayer.custom_op is not None:
            if not np.isscalar(color_dst):
                await peval(lambda: blit(sublayer.custom_op_color, color_dst, offset))
                await peval(lambda: blit(sublayer.custom_op_alpha, alpha_src, offset))
            await sublayer.custom_op_barrier.wait()
            if not sublayer.custom_op_condition.locked():
                if not sublayer.custom_op_finished:
                    async with sublayer.custom_op_condition:
                        sublayer.custom_op_color, sublayer.custom_op_alpha = \
                            await peval(lambda: sublayer.custom_op(sublayer.custom_op_color, sublayer.custom_op_alpha))
                        sublayer.custom_op_finished = True
                        sublayer.custom_op_condition.notify_all()
            else:
                async with sublayer.custom_op_condition:
                    await sublayer.custom_op_condition.wait_for(lambda: sublayer.custom_op_finished)
            neg_offset = -np.array(offset)
            await peval(lambda: blit(color_src, sublayer.custom_op_color, neg_offset))
            await peval(lambda: blit(alpha_src, sublayer.custom_op_alpha, neg_offset))

        # Opacity is actually FILL when special mode is true!
        opacity, special_mode = get_sai_special_mode_opacity(sublayer.layer)

        # A pass-through layer has already been blended, so just lerp instead.
        # NOTE: Clipping layers do not apply to pass layers, as if clipping were simply disabled.
        if blend_mode == BlendMode.PASS_THROUGH:
            mask_src = await get_mask_data(sublayer.layer, size, offset)
            if mask_src is None:
                mask_src = opacity
            else:
                await peval(lambda: np.multiply(mask_src, opacity, out=mask_src))
            if np.isscalar(mask_src) and mask_src == 1.0:
                color_dst = color_src
                alpha_dst = alpha_src
            else:
                color_dst = await peval(lambda: util.lerp(color_dst, color_src, mask_src, out=color_src))
                alpha_dst = await peval(lambda: util.lerp(alpha_dst, alpha_src, mask_src, out=alpha_src))
            set_cached_composite(sublayer, offset, (color_dst, alpha_dst))
            continue

        if sublayer.layer.is_group():
            # Un-multiply group composites so that we can multiply group opacity correctly
            await peval(lambda: util.clip_divide(color_src, alpha_src, out=color_src))

        if sublayer.clip_layers:
            # Composite the clip layers now. This basically overwrites just the color by blending onto it without
            # alpha blending it first. For whatever reason, applying a large root to the alpha source before passing
            # it to clip compositing fixes brightening that can occur with certain blend modes (like multiply).
            corrected_alpha = await peval(lambda: alpha_src ** 0.0001)
            clip_src, _ = await composite_group_layer(sublayer.clip_layers, size, offset, (color_src, corrected_alpha))
            if clip_src is not None:
                color_src = clip_src

        # Apply opacity (fill) before blending otherwise premultiplied blending of special modes will not work correctly.
        await peval(lambda: np.multiply(alpha_src, opacity, out=alpha_src))

        # Now we can 'premultiply' the color_src for the main blend operation.
        await peval(lambda: np.multiply(color_src, alpha_src, out=color_src))

        # Run the blend operation.
        blend_func = blendfuncs.get_blend_func(blend_mode, special_mode)
        color_src = await peval(lambda: blend_func(color_dst, color_src, alpha_dst, alpha_src))

        # Premultiplied blending may cause out-of-range values, so it must be clipped.
        if blend_mode != BlendMode.NORMAL:
            await peval(lambda: util.clip(color_src, out=color_src))

        # We apply the mask last and LERP the blended result onto the destination.
        # Why? Because this is how Photoshop and SAI do it. Applying the mask before blending
        # will yield a slightly different result from those programs.
        mask_src = await get_mask_data(sublayer, size, offset)
        if mask_src is not None:
            color_dst = await peval(lambda: util.lerp(color_dst, color_src, mask_src, out=color_src))
        else:
            color_dst = color_src

        # Finally we can intersect the mask with the alpha_src and blend the alpha_dst together.
        if mask_src is not None:
            await peval(lambda: np.multiply(alpha_src, mask_src, out=alpha_src))
        alpha_dst = await peval(lambda: blendfuncs.normal_alpha(alpha_dst, alpha_src))

        set_cached_composite(sublayer, offset, (color_dst, alpha_dst))

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

async def composite_tile(psd:WrappedLayer, size, offset, color, alpha):
    tile_color, tile_alpha = await composite_group_layer(psd, size, offset)
    if tile_color is not None:
        await peval(lambda: blit(color, tile_color, offset))
        await peval(lambda: blit(alpha, tile_alpha, offset))

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

async def composite(psd:WrappedLayer, tile_size=(256,256)):
    '''
    Composite the given PSD and return color (RGB) and alpha arrays.
    `tile_size` is arranged by (height, width)
    '''
    size = psd.layer.height, psd.layer.width
    color = np.zeros(size + (3,))
    alpha = np.zeros(size + (1,))

    tasks = []

    for (tile_size, tile_offset) in generate_tiles(size, tile_size):
        tasks.append(composite_tile(psd, tile_size, tile_offset, color, alpha))

    set_layer_extra_data(psd, len(tasks), size)

    await asyncio.gather(*tasks)

    return color, alpha
