from __future__ import annotations

import asyncio
import pathlib
import threading

import cv2
import numpy as np
from psd_tools import PSDImage
from psd_tools.api.layers import Layer
from psd_tools.constants import BlendMode, Clipping, Tag

from . import blendfuncs
from . import util
from .util import peval

class WrappedLayer():
    def __init__(self, layer:Layer, clip_layers=[], parent:WrappedLayer=None):
        self.layer:Layer = layer
        self.parent:WrappedLayer = parent
        self.name = layer.name
        self.visible = True if layer.kind == 'psdimage' else layer.visible
        self.skip = False
        self.custom_op = None
        self.children:list[WrappedLayer] = []
        self.flat_children:list[WrappedLayer] = []
        self.clip_layers:list[WrappedLayer] = clip_layers
        self.composite_cache = {}
        self.data_cache = {}
        self.data_cache_lock = threading.Lock()
        self.tags = []
        self.visibility_dependency = None
        self.tag_dependency = None
        self.cache_hit = ''
        self.worker_counter = 0

        if self.layer.is_group():
            for sublayer, sub_clip_layers in get_layer_and_clip_groupings(self.layer):
                sub_clip_layers = [WrappedLayer(s, parent=self) for s in sub_clip_layers]
                sublayer = WrappedLayer(sublayer, sub_clip_layers, self)
                self.children.append(sublayer)
                self.flat_children.append(sublayer)
                self.flat_children.extend(sub_clip_layers)

    def layer_path(self):
        if self.parent:
            return self.parent.layer_path() + '/' + self.name
        else:
            return '/' + self.name

    def __repr__(self):
        return f'{self.__class__.__name__}({self.layer_path()})'

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

def print_layers(layer:WrappedLayer, tab='', next_tab='  ', more_fn=None):
    text = layer.name
    if is_clipping(layer.layer):
        text = f'* {text}'
    if more_fn:
        text = f'{text} {more_fn(layer)}'
    print(tab, text)
    for child in reversed(layer.flat_children):
        print_layers(child, tab + next_tab, next_tab, more_fn)

def PSDOpen(fp, **kwargs):
    return WrappedLayer(PSDImage.open(fp, **kwargs))

async def barrier_skip(barrier:asyncio.Barrier):
    try:
        await asyncio.wait_for(asyncio.shield(barrier.wait()), timeout=0.0)
    except TimeoutError:
        pass

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

def get_visibility_all_children(layer:WrappedLayer, visited, visit_all):
    if layer.visible or visit_all:
        tags = [not tag.ignore for tag in layer.tags]
        if any(tags):
            visited.add(layer)
        for sublayer in layer.flat_children:
            get_visibility_all_children(sublayer, visited, visit_all)
    return visited

def get_visibility_dependency_sub(layer:WrappedLayer, visited, visit_all):
    if layer.parent == None:
        return
    if layer in visited:
        return
    found = False
    for sublayer in reversed(layer.parent.flat_children):
        if sublayer == layer:
            found = True
        if found:
            get_visibility_all_children(sublayer, visited, visit_all)
    if layer.parent.parent != None and layer.parent.layer.blend_mode == BlendMode.PASS_THROUGH:
        get_visibility_dependency_sub(layer.parent, visited, visit_all)

def get_visibility_dependency(layer:WrappedLayer, visit_all=False):
    visited = set()
    get_visibility_dependency_sub(layer, visited, visit_all)
    return frozenset(visited)

def set_layer_extra_data(layer:WrappedLayer, tile_count, size):
    set_tag_dependency(layer)
    for sublayer in layer.descendants():
        sublayer.worker_counter = tile_count
        sublayer.cache_hit = ''
        sublayer.visibility_dependency = get_visibility_dependency(sublayer)
        if sublayer.custom_op is not None:
            sublayer.custom_op_barrier = asyncio.Barrier(tile_count)
            sublayer.custom_op_condition = asyncio.Condition()
            sublayer.custom_op_finished = False
            sublayer.custom_op_color = np.zeros(size + (3,))
            sublayer.custom_op_alpha = np.zeros(size + (1,))

def get_tile_cache(layer:WrappedLayer, offset):
    if offset not in layer.composite_cache:
        layer.composite_cache[offset] = {}
    return layer.composite_cache[offset]

def get_cached_composite(layer:WrappedLayer, offset):
    return get_tile_cache(layer, offset).get(layer.visibility_dependency, None)

def set_cached_composite(layer:WrappedLayer, offset, tile_data):
    for data in tile_data:
        if not np.isscalar(data):
            data.flags.writeable = False
    get_tile_cache(layer, offset)[layer.visibility_dependency] = tile_data

def clear_all_caches(layer:WrappedLayer):
    layer.composite_cache.clear()
    layer.data_cache.clear()
    for sublayer in layer.descendants():
        sublayer.composite_cache.clear()
        sublayer.data_cache.clear()

def clear_descendants_caches(layer:WrappedLayer):
    for sublayer in layer.descendants():
        sublayer.composite_cache.clear()
        sublayer.data_cache.clear()

def set_tag_dependency(layer:WrappedLayer):
    for sublayer in layer.descendants():
        if sublayer.tag_dependency is None:
            v = get_visibility_dependency(sublayer, True)
            sublayer.tag_dependency = False
            for v_layer in v:
                tags = [not tag.ignore for tag in v_layer.tags]
                if any(tags):
                    sublayer.tag_dependency = True
                    break

def set_skip_to_last_untagged(layer:WrappedLayer):
    skip = False
    for child in reversed(layer.children):
        if not child.tag_dependency and not skip:
            skip = True
            continue
        if skip and child.composite_cache:
            child.skip = True
            child.data_cache.clear()
            child.composite_cache.clear()

def set_skips(layer:WrappedLayer):
    for sublayer in layer.descendants():
        set_skip_to_last_untagged(sublayer)

def get_cached_layer_data(layer:WrappedLayer, channel):
    with layer.data_cache_lock:
        if channel not in layer.data_cache:
            data = layer.layer.numpy(channel)
            if data is not None:
                data = data.astype(np.float64)
            layer.data_cache[channel] = data
            return data
        return layer.data_cache[channel]

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

async def composite_group_layer(layer:WrappedLayer | list[WrappedLayer], size, offset, backdrop=None):
    if backdrop:
        color_dst, alpha_dst = backdrop
    else:
        color_dst = 0.0
        alpha_dst = 0.0

    sublayer:WrappedLayer
    for sublayer in layer:
        try:
            if not sublayer.visible or sublayer.skip:
                if sublayer.custom_op is not None:
                    await barrier_skip(sublayer.custom_op_barrier)
                continue

            cached_composite = get_cached_composite(sublayer, offset)
            if cached_composite is not None:
                color_dst, alpha_dst = cached_composite
                sublayer.cache_hit = True
                if sublayer.custom_op is not None:
                    await barrier_skip(sublayer.custom_op_barrier)
                continue
            else:
                sublayer.cache_hit = False

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
                set_cached_composite(sublayer, offset, (color_dst, alpha_dst))
                if sublayer.custom_op is not None:
                    await barrier_skip(sublayer.custom_op_barrier)
                # Need to enter the clip layers to make sure any barriers are hit.
                if sublayer.clip_layers:
                    await composite_group_layer(sublayer.clip_layers, size, offset)
                continue

            # Copy group output to prevent mutated cache tiles
            if sublayer.layer.is_group():
                color_src = color_src.copy()
                alpha_src = alpha_src.copy()

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
                mask_src = await get_mask_data(sublayer, size, offset)
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
            else:
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
        finally:
            sublayer.worker_counter -= 1
            if sublayer.worker_counter == 0 and not sublayer.tag_dependency:
                sublayer.data_cache.clear()
                clear_descendants_caches(sublayer)

    if np.isscalar(color_dst):
        return None, None

    return color_dst, alpha_dst

debug_path = pathlib.Path('')
debug_run = 0

def debug_layer(name, offset, data):
    data = (data * 255).astype(np.uint8)
    if data.shape[2] == 1:
        data = data.reshape(data.shape[:2])
    else:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    path = debug_path /  f'{name}-{offset}-{debug_run}.png'
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

    tiles = list(generate_tiles(size, tile_size))
    set_layer_extra_data(psd, len(tiles), size)

    async with asyncio.TaskGroup() as tg:
        for (tile_size, tile_offset) in tiles:
            tg.create_task(composite_tile(psd, tile_size, tile_offset, color, alpha))

    set_skips(psd)

    return color, alpha
