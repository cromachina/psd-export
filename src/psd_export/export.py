import argparse
import asyncio
import glob
import logging
import pathlib
import re
import time
from typing import NamedTuple

import cv2
from pyrsistent import pmap, pset, pvector

from . import composite
from . import filters
from . import util

class Tag(NamedTuple):
    ignore: bool
    name: str
    xor_group: str
    args: list[str]

tag_regex = re.compile('\[(.*?)\]')

def parse_tag(tag:str):
    args = tag.split()
    if len(args) == 0:
        return Tag(False, '', '', [])
    else:
        ignore = False
        arg0 = args[0]
        if arg0.startswith('#'):
            ignore = True
            arg0 = arg0.strip('#')
        name = arg0.split('@', 1)
        xor = None
        if len(name) == 2:
            name, xor = name
        else:
            name = name[0]
        return Tag(
            ignore,
            name,
            xor,
            args[1:]
        )

is_range = re.compile('(\\d+):(\\d+)')

def parse_tags(input):
    tags = []
    for tag_str in tag_regex.findall(input):
        tag = parse_tag(tag_str)
        tag_range = is_range.match(tag.name)
        if tag_range:
            for i in range(int(tag_range[1]), int(tag_range[2]) + 1):
                tags.append(tag._replace(name=str(i)))
        else:
            tags.append(tag)
    return tags

def fixed_primary_tag(tag):
    return tag if tag == '' else f'-{tag}'

def compute_file_name(base_file_name, config, enabled_tags):
    enabled_tags = [tag[0] for tag in enabled_tags]
    if config.primary_sub:
        primary_tag = ''
        group_name = '-'.join(enabled_tags)
    else:
        primary_tag = fixed_primary_tag(enabled_tags[0])
        group_name = '-'.join(enabled_tags[1:])
    if config.subfolders:
        next_file_name = base_file_name.with_stem(f'{base_file_name.stem}{primary_tag}')
        next_file_name = next_file_name.parent / group_name / next_file_name.name
    else:
        next_file_name = base_file_name.with_stem(f'{base_file_name.stem}{primary_tag}-{group_name}')
    return next_file_name

async def export_variant(psd, file_name, config, enabled_tags, count_mode):
    export_name = compute_file_name(file_name, config, enabled_tags)

    if config.dryrun and not count_mode:
        logging.info(f'would export: {export_name}')
        return

    # Configure tagged layers
    for layer in psd.descendants():
        custom_ops = []
        def add_op(tag):
            custom_op = filters.get_filter(tag.name)
            if custom_op:
                custom_ops.append(lambda *args: custom_op(*args, *tag.args))

        for tag in layer.tags:
            if not tag.ignore:
                layer.visible = False
        for tag in layer.tags:
            if tag.ignore:
                add_op(tag)
            elif (tag.name, tag.xor_group) in enabled_tags:
                layer.visible = True
                add_op(tag)
        layer.custom_op = filters.compose_ops(custom_ops)

    image = await composite.composite(psd, count_mode=count_mode)

    if count_mode:
        return

    export_name.parent.mkdir(parents=True, exist_ok=True)
    util.save_file(export_name, image, [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression, cv2.IMWRITE_JPEG_QUALITY, config.jpg_quality])

async def export_combinations(psd, file_name, config, secondary_tags, enabled_tags, count_mode):
    if not secondary_tags:
        return

    items = list(secondary_tags.items())
    items.sort()

    for xor_group, tags in items:
        secondary_tags = secondary_tags.remove(xor_group)
        for tag in tags:
            next_enabled = enabled_tags.append((tag, xor_group))
            await export_variant(psd, file_name, config, next_enabled, count_mode)
            await export_combinations(psd, file_name, config, secondary_tags, next_enabled, count_mode)

def get_least_tagged_layer(layer_map):
    lowest = None
    for pair in layer_map.items():
        if lowest is None or len(pair[1]) < len(lowest[1]):
            lowest = pair
    return lowest[1]

def remove_tag(layer_map, tag):
    for layer, tags in layer_map.items():
        tags = tags.discard(tag)
        if not tags:
            layer_map = layer_map.discard(layer)
        else:
            layer_map = layer_map.set(layer, tags)
    return layer_map

def is_reachable(layer:composite.WrappedLayer):
    while layer is not None:
        if not layer.tags and not layer.visible:
            return False
        layer = layer.parent
    return True

async def export_all_variants(file_name, config):
    psd = composite.PSDOpen(file_name)

    file_name = pathlib.Path(file_name).with_suffix(f'.{config.output_type}')

    primary_layers = pmap()
    secondary_tags = pmap()
    primary_filter = config.primary_filter.split()
    secondary_filter = config.secondary_filter.split()

    for layer in psd.descendants():
        tags = parse_tags(layer.name)
        layer.tags = tags

    for layer in psd.descendants():
        if not is_reachable(layer):
            continue
        for tag in layer.tags:
            if tag.ignore:
                continue
            if tag.xor_group is None:
                if not primary_filter or tag.name in primary_filter:
                    s = primary_layers.get(layer, pset()).add(tag.name)
                    primary_layers = primary_layers.set(layer, s)
            else:
                if not secondary_filter or tag.name in secondary_filter:
                    group = secondary_tags.get(tag.xor_group, pset()).add(tag.name)
                    secondary_tags = secondary_tags.set(tag.xor_group, group)

    if not primary_layers:
        primary_layers = primary_layers.set(psd, pset(['']))

    primary_tags = []
    while primary_layers:
        tags = get_least_tagged_layer(primary_layers)
        primary_tag = next(iter(tags))
        primary_tags.append(primary_tag)
        primary_layers = remove_tag(primary_layers, primary_tag)

    for count_mode in [True, False]:
        for primary_tag in primary_tags:
            enabled = pvector([(primary_tag, None)])
            if not config.only_secondary_tags:
                await export_variant(psd, file_name, config, enabled, count_mode)
            await export_combinations(psd, file_name, config, secondary_tags, enabled, count_mode)
        composite.clear_count_mode(psd)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--subfolders', action=argparse.BooleanOptionalAction, default=True,
    help='Export secondary tags to subfolders.')
arg_parser.add_argument('--primary-sub', action=argparse.BooleanOptionalAction, default=False,
    help='Make primary tags into subfolders.')
arg_parser.add_argument('--dryrun', action=argparse.BooleanOptionalAction, default=False,
    help='Show what files would have been exported, but do not actually export anything.')
arg_parser.add_argument('--only-secondary-tags', action=argparse.BooleanOptionalAction, default=False,
    help='Only export secondary tags. This is useful for when exporting a primary tag by itself does not produce a meaningful picture.')
arg_parser.add_argument('--primary-filter', default="", type=str,
    help='Only export primary tags matching the given names.')
arg_parser.add_argument('--secondary-filter', default="", type=str,
    help='Only export secondary tags matching the given names.')
arg_parser.add_argument('--mosaic-factor', default=filters.mosaic_factor_default, type=float,
    help='Set the mosaic proportion factor of censors, based on the minimum image dimension.')
arg_parser.add_argument('--png-compression', default=1, type=int,
    help='Set the compression level for PNG output (0 to 9).')
arg_parser.add_argument('--jpg-quality', default=95, type=int,
    help='Set the quality level for JPG output (0 to 100).')
arg_parser.add_argument('--output-type', default='png', type=str,
    help='Output type, whatever is supported by OpenCV, for example: png, jpg, webp, tiff.')
arg_parser.add_argument('--file-name',  default='*.psd', type=str,
    help='PSD files to process; can use a glob pattern.')

async def async_main():
    logging.basicConfig(level=logging.INFO)
    args = arg_parser.parse_args()
    filters.mosaic_factor_default = args.mosaic_factor
    start = time.perf_counter()
    for file_name in glob.iglob(args.file_name):
        await export_all_variants(file_name, args)
    await util.save_workers_wait_all()
    logging.info(f'export time: {time.perf_counter() - start}')

def main():
    asyncio.run(async_main())

if __name__ == '__main__':
    main()
