import argparse
import pathlib
import re
import time
import logging

from PIL import Image
from psd_tools import PSDImage
from pyrsistent import pmap, pset, pvector

import composite

tag_regex = re.compile('\[(.+)\]')
censor_regex = re.compile('\[censor\]|\[censor@.*\]')

def find_layers(layer, regex):
    return list(filter(lambda sublayer: regex.search(sublayer.name), layer.descendants()))

def find_layer(layer, exact_name):
    for sublayer in layer.descendants():
        if sublayer == exact_name:
            return sublayer

def apply_mosaic(image, mask):
    original_size = image.size
    min_dim = min(original_size) // 100
    min_dim = max(4, min_dim)
    scale_dimension = (original_size[0] // min_dim, original_size[1] // min_dim)
    mosaic_image = image.resize(scale_dimension).resize(original_size, Image.Resampling.NEAREST)
    return Image.composite(mosaic_image, image, mask)

def get_censor_composite_mask(layers, viewport):
    censor_composites = []
    for layer in layers:
        for censor_layer in find_layers(layer, censor_regex):
            censor_composites.append(censor_layer.composite(viewport=viewport))
    if len(censor_composites) > 0:
        composite = censor_composites[0]
        for other_composite in censor_composites[1:]:
            composite = Image.alpha_composite(composite, other_composite)
        return composite
    return None

def export_variant(psd, file_name, show_tags):
    # Disable all tags
    for layer in find_layers(psd, tag_regex):
        layer.visible = False

    has_mosaic_censor = False

    # Enable only active tags except censor
    show_layers = []
    for tag in show_tags:
        if censor_regex.search(f'[{re.escape(tag)}]'):
            has_mosaic_censor = True
        else:
            for layer in find_layers(psd, re.compile(f'\[{re.escape(tag)}\]')):
                layer.visible = True
                show_layers.append(layer)

    image = composite.composite(psd)

    if has_mosaic_censor:
        for layer in find_layers(psd, censor_regex):
            layer.visible = True
        mask = get_censor_composite_mask(show_layers, psd.viewbox)
        if mask:
            image = apply_mosaic(image, mask)

    file_name.parent.mkdir(parents=True, exist_ok=True)
    image.save(file_name)
    logging.info(f'exported: {file_name}')

def fixed_primary_tag(tag):
    return tag if tag == '' else f'-{tag}'

def export_combinations(psd, file_name, config, secondary_tags, enabled_tags, full_enabled_tags):
    if not secondary_tags:
        return

    for xor_group, tags in secondary_tags.items():
        for tag in tags:
            next_enabled = enabled_tags.append(tag)
            next_full_enabled = full_enabled_tags.append(f'{tag}@{xor_group}')
            next_secondary = secondary_tags.remove(xor_group)

            primary_tag = fixed_primary_tag(next_enabled[0])
            group_name = '-'.join(next_enabled[1:])
            if config.subfolders:
                next_file_name = file_name.with_stem(f'{file_name.stem}{primary_tag}')
                next_file_name = next_file_name.parent / group_name / next_file_name.name
            else:
                next_file_name = file_name.with_stem(f'{file_name.stem}{primary_tag}-{group_name}')

            if config.dryrun:
                logging.info(f'would export: {next_file_name}')
            else:
                export_variant(psd, next_file_name, next_full_enabled)
            export_combinations(psd, file_name, config, next_secondary, next_enabled, next_full_enabled)

        secondary_tags = secondary_tags.remove(xor_group)

def export_all_variants(file_name, config):
    start = time.perf_counter()
    psd = PSDImage.open(file_name)
    file_name = pathlib.Path(file_name).with_suffix('.png')
    tags = pset()

    tagged_layers = find_layers(psd, tag_regex)
    for layer in tagged_layers:
        for tag in tag_regex.findall(layer.name):
            tags = tags.add(tag)

    primary_tags = pset()
    secondary_tags = pmap()
    for tag in tags:
        if '@' in tag:
            sub_tag, xor_group = tag.split('@')
            group = secondary_tags.get(xor_group, pset()).add(sub_tag)
            secondary_tags = secondary_tags.set(xor_group, group)
        else:
            primary_tags = primary_tags.add(tag)

    if len(primary_tags) == 0:
        primary_tags = primary_tags.add('')

    for primary_tag in primary_tags:
        enabled = pvector([primary_tag])
        if not config.only_secondary_tags:
            primary_file_name =  file_name.with_stem(f'{file_name.stem}{fixed_primary_tag(primary_tag)}')
            if config.dryrun:
                logging.info(f'would export: {primary_file_name}')
            else:
                export_variant(psd, primary_file_name, enabled)

        export_combinations(psd, file_name, config, secondary_tags, enabled, enabled)

    logging.info(f'export time: {time.perf_counter() - start}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--subfolders', action=argparse.BooleanOptionalAction, default=True,
        help='Export secondary tags to subfolders.')
    parser.add_argument('--dryrun', action=argparse.BooleanOptionalAction, default=False,
        help='Show what files would have been exported, but do not actually export anything.')
    parser.add_argument('--only-secondary-tags', action=argparse.BooleanOptionalAction, default=False,
        help='Only export secondary tags. This is useful for when exporting a primary tag by itself does not produce a meaningful picture.')
    parser.add_argument('file_name', type=str)
    args = parser.parse_args()
    export_all_variants(args.file_name, args)
