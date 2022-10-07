import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
import pathlib
import re
import time
import logging

from PIL import Image
from psd_tools import PSDImage
from pyrsistent import pmap, pset, pvector
import numpy as np

import composite

tag_regex = re.compile('\[(.+?)\]')
censor_regex = re.compile('\[censor\]|\[censor@.*?\]')

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

def get_censor_composite_mask(psd, layers):
    censor_layers = set()
    for layer in layers:
        if layer.is_group():
            for censor_layer in find_layers(layer, censor_regex):
                censor_layers.add(censor_layer)
        elif censor_regex.search(layer.name):
            censor_layers.add(layer)
    if len(censor_layers) > 0:
        return composite.union_mask(psd, censor_layers)
    return None

def export_variant(psd, file_name, config, enabled_tags, full_enabled_tags):
    if config.dryrun:
        export_name = compute_file_name(file_name, config.subfolders, enabled_tags)
        logging.info(f'would export: {export_name}')
        return

    # Disable all tags
    for layer in find_layers(psd, tag_regex):
        layer.visible = False

    has_mosaic_censor = False

    # Enable only active tags
    show_layers = []
    for tag in full_enabled_tags:
        if censor_regex.search(f'[{re.escape(tag)}]'):
            has_mosaic_censor = True
        else:
            for layer in find_layers(psd, re.compile(f'\[{re.escape(tag)}\]')):
                layer.visible = True
                show_layers.append(layer)

    # Censor tags may also share a primary tag; disable them again.
    for layer in find_layers(psd, censor_regex):
        layer.visible = False

    image = None

    # Since we encountered a mosaic censor tag, the predecessor image might have been created already
    # and if so, we can use that and skip expensive compositing.
    if has_mosaic_censor:
        predecessor_file = compute_file_name(file_name, config.subfolders, enabled_tags.remove('censor'))
        image = config._file_cache[predecessor_file]

    if image is None:
        image = composite.composite(psd)

    if has_mosaic_censor:
        mask = get_censor_composite_mask(psd, show_layers)
        if mask:
            image = apply_mosaic(image, mask)

    export_name = compute_file_name(file_name, config.subfolders, enabled_tags)
    export_name.parent.mkdir(parents=True, exist_ok=True)
    save_file(config._save_pool, export_name, image)
    config._file_cache[export_name] = image

def fixed_primary_tag(tag):
    return tag if tag == '' else f'-{tag}'

def compute_file_name(base_file_name, use_subfolders, enabled_tags):
    primary_tag = fixed_primary_tag(enabled_tags[0])
    group_name = '-'.join(enabled_tags[1:])
    if use_subfolders:
        next_file_name = base_file_name.with_stem(f'{base_file_name.stem}{primary_tag}')
        next_file_name = next_file_name.parent / group_name / next_file_name.name
    else:
        next_file_name = base_file_name.with_stem(f'{base_file_name.stem}{primary_tag}-{group_name}')
    return next_file_name

def export_combinations(psd, file_name, config, secondary_tags, enabled_tags, full_enabled_tags):
    if not secondary_tags:
        return

    for xor_group, tags in secondary_tags.items():
        for tag in tags:
            next_enabled = enabled_tags.append(tag)
            next_full_enabled = full_enabled_tags.append(f'{tag}@{xor_group}')
            next_secondary = secondary_tags.remove(xor_group)

            if tag != 'censor':
                export_variant(psd, file_name, config, next_enabled, next_full_enabled)

            export_combinations(psd, file_name, config, next_secondary, next_enabled, next_full_enabled)

            if tag == 'censor':
                export_variant(psd, file_name, config, next_enabled, next_full_enabled)

        secondary_tags = secondary_tags.remove(xor_group)

def save_worker(file_name, shape, sm):
    logging.basicConfig(level=logging.INFO)
    try:
        array = np.ndarray(shape, dtype=np.uint8, buffer=sm.buf)
        image = Image.fromarray(array)
        image.save(file_name)
        logging.info(f'exported: {file_name}')
    except Exception as e:
        logging.exception(e)
    finally:
        sm.close()
        sm.unlink()

def save_file(pool, file_name, image):
    array = np.asarray(image)
    sm = SharedMemory(create=True, size=array.nbytes)
    a = np.ndarray(array.shape, dtype=array.dtype, buffer=sm.buf)
    np.copyto(a, array)
    pool.submit(save_worker, file_name, array.shape, sm)

def export_all_variants(file_name, config):
    start = time.perf_counter()
    config._save_pool = ProcessPoolExecutor(1)

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
            if sub_tag == '':
                continue
            group = secondary_tags.get(xor_group, pset()).add(sub_tag)
            secondary_tags = secondary_tags.set(xor_group, group)
        else:
            primary_tags = primary_tags.add(tag)

    if len(primary_tags) == 0:
        primary_tags = primary_tags.add('')

    for primary_tag in primary_tags:
        config._file_cache = {}
        enabled = pvector([primary_tag])
        if not config.only_secondary_tags:
            export_variant(psd, file_name, config, enabled, enabled)
        export_combinations(psd, file_name, config, secondary_tags, enabled, enabled)

    config._save_pool.shutdown(wait=True)
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
