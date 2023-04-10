import argparse
import asyncio
import glob
import logging
import pathlib
import re
import time
from collections import namedtuple

import cv2
from psd_tools import PSDImage
from pyrsistent import pmap, pset, pvector

import composite
import util

Tag = namedtuple('Tag', ["ignore", "name", "xor_group", "args"])

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

filter_names = {
    'censor': composite.mosaic_op,
    'blur': composite.blur_op,
}

def parse_tags(input):
    return [parse_tag(tag) for tag in tag_regex.findall(input)]

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

async def export_variant(psd, file_name, config, enabled_tags):
    export_name = compute_file_name(file_name, config, enabled_tags)

    if config.dryrun:
        logging.info(f'would export: {export_name}')
        return

    # Configure tagged layers
    for layer in psd.descendants():
        custom_ops = []
        def add_op(tag):
            custom_op = filter_names.get(tag.name)
            if custom_op:
                custom_ops.append(lambda c, a: custom_op(c, a, *tag.args))

        for tag in layer._tags:
            if not tag.ignore:
                layer.visible = False
        for tag in layer._tags:
            if tag.ignore:
                add_op(tag)
            elif (tag.name, tag.xor_group) in enabled_tags:
                layer.visible = True
                add_op(tag)
        composite.set_custom_operation(layer, composite.chain_ops(custom_ops))

    image = await composite.composite(psd)

    export_name.parent.mkdir(parents=True, exist_ok=True)
    util.save_file(export_name, image, [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression, cv2.IMWRITE_JPEG_QUALITY, config.jpg_quality])

async def export_combinations(psd, file_name, config, secondary_tags, enabled_tags):
    if not secondary_tags:
        return

    items = list(secondary_tags.items())
    items.sort()

    for xor_group, tags in items:
        secondary_tags = secondary_tags.remove(xor_group)
        for tag in tags:
            next_enabled = enabled_tags.append((tag, xor_group))
            await export_variant(psd, file_name, config, next_enabled)
            await export_combinations(psd, file_name, config, secondary_tags, next_enabled)

async def export_all_variants(file_name, config):
    psd = PSDImage.open(file_name)

    file_name = pathlib.Path(file_name).with_suffix(f'.{config.output_type}')

    primary_tags = pset()
    secondary_tags = pmap()

    for layer in psd.descendants():
        tags = parse_tags(layer.name)
        layer._tags = tags
        for tag in tags:
            if tag.ignore:
                continue
            if tag.xor_group is None:
                primary_tags = primary_tags.add(tag.name)
            else:
                group = secondary_tags.get(tag.xor_group, pset()).add(tag.name)
                secondary_tags = secondary_tags.set(tag.xor_group, group)

    if not primary_tags:
        primary_tags = primary_tags.add('')

    primary_tags = list(primary_tags)
    primary_tags.sort()
    primary_tags = pvector(primary_tags)

    for primary_tag in primary_tags:
        enabled = pvector([(primary_tag, None)])
        if not config.only_secondary_tags:
            await export_variant(psd, file_name, config, enabled)
        await export_combinations(psd, file_name, config, secondary_tags, enabled)

async def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--subfolders', action=argparse.BooleanOptionalAction, default=True,
        help='Export secondary tags to subfolders.')
    parser.add_argument('--primary-sub', action=argparse.BooleanOptionalAction, default=False,
        help='Make primary tags into subfolders.')
    parser.add_argument('--dryrun', action=argparse.BooleanOptionalAction, default=False,
        help='Show what files would have been exported, but do not actually export anything.')
    parser.add_argument('--only-secondary-tags', action=argparse.BooleanOptionalAction, default=False,
        help='Only export secondary tags. This is useful for when exporting a primary tag by itself does not produce a meaningful picture.')
    parser.add_argument('--mosaic-factor', default=100, type=float,
        help='Set the mosaic proportion factor of censors, based on the minimum image dimension.')
    parser.add_argument('--png-compression', default=1, type=int,
        help='Set the compression level for PNG output (0 to 9).')
    parser.add_argument('--jpg-quality', default=95, type=int,
        help='Set the quality level for JPG output (0 to 100).')
    parser.add_argument('--output-type', type=str, default='png',
        help='Output type, whatever is supported by OpenCV, for example: png, jpg, webp, tiff.')
    parser.add_argument('--file-name', type=str, default='*.psd',
        help='PSD files to process; can use a glob pattern.')
    args = parser.parse_args()

    composite.mosaic_factor_default = args.mosaic_factor
    start = time.perf_counter()
    for file_name in glob.iglob(args.file_name):
        await export_all_variants(file_name, args)
    await util.save_workers_wait_all()
    logging.info(f'export time: {time.perf_counter() - start}')

if __name__ == '__main__':
    asyncio.run(main())