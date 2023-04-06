import argparse
import glob
import logging
import pathlib
import re
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np
from psd_tools import PSDImage
from pyrsistent import pmap, pset, pvector

import composite

pool = ProcessPoolExecutor(1)

tag_regex = re.compile('\[(.+?)\]')
censor_regex = re.compile('\[censor\]|\[censor@.*?\]')

def find_layers(layer, regex):
    return list(filter(lambda sublayer: regex.search(sublayer.name), layer.descendants()))

def find_layer(layer, exact_name):
    for sublayer in layer.descendants():
        if sublayer.name == exact_name:
            return sublayer

def apply_mosaic(image, mask, mosaic_factor=100):
    original_size = composite.swap(image.shape[:2])
    min_dim = min(original_size) // mosaic_factor
    min_dim = max(4, min_dim)
    scale_dimension = (original_size[0] // min_dim, original_size[1] // min_dim)
    mosaic_image = cv2.resize(image, scale_dimension, interpolation=cv2.INTER_AREA)
    mosaic_image = cv2.resize(mosaic_image, original_size, interpolation=cv2.INTER_NEAREST)
    return composite.parallel_lerp(image, mosaic_image, mask)

def get_censor_composite_mask(psd, layers, censor_regex_set):
    if not layers:
        layers = psd
    censor_layers = set()
    for layer in layers:
        if layer.is_group():
            for regex in censor_regex_set:
                for censor_layer in find_layers(layer, regex):
                    censor_layers.add(censor_layer)
        elif censor_regex.search(layer.name):
            censor_layers.add(layer)
    if len(censor_layers) > 0:
        return composite.union_mask(psd, censor_layers)
    return None

def export_variant(psd, file_name, config, enabled_tags, full_enabled_tags):
    if config.dryrun:
        export_name = compute_file_name(file_name, config, enabled_tags)
        logging.info(f'would export: {export_name}')
        return

    # Disable all tags
    for layer in find_layers(psd, tag_regex):
        layer.visible = False

    censor_regex_set = set()

    # Enable only active tags
    show_layers = []
    for tag in full_enabled_tags:
        if censor_regex.search(f'[{re.escape(tag)}]'):
            censor_regex_set.add(re.compile(f'\[{re.escape(tag)}\]'))
        else:
            for layer in find_layers(psd, re.compile(f'\[{re.escape(tag)}\]')):
                layer.visible = True
                show_layers.append(layer)

    # Censor tags may also share a primary tag; disable them again.
    for layer in find_layers(psd, censor_regex):
        layer.visible = False

    image = composite.composite(psd)

    export_name = compute_file_name(file_name, config, enabled_tags)
    export_name.parent.mkdir(parents=True, exist_ok=True)
    save_file(export_name, image)

def fixed_primary_tag(tag):
    return tag if tag == '' else f'-{tag}'

def compute_file_name(base_file_name, config, enabled_tags):
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

def export_combinations(psd, file_name, config, secondary_tags, enabled_tags, full_enabled_tags):
    if not secondary_tags:
        return

    items = list(secondary_tags.items())
    items.sort()

    for xor_group, tags in items:
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

def is_grayscale(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    return (r == g).all() and (g == b).all()

def save_worker(file_name, image_sm):
    logging.basicConfig(level=logging.INFO)
    try:
        image = np.ndarray(image_sm[0], image_sm[1], image_sm[2].buf)
        image = np.multiply(image, 255).astype(np.uint8)
        if is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_name, image)
        logging.info(f'exported: {file_name}')
    except Exception as e:
        logging.exception(e)
    finally:
        delete_shared(image_sm)

def make_shared(image):
    sm = SharedMemory(create=True, size=image.nbytes)
    a = np.ndarray(image.shape, dtype=image.dtype, buffer=sm.buf)
    np.copyto(a, image)
    return (image.shape, image.dtype, sm)

def delete_shared(image_sm):
    image_sm[2].close()
    image_sm[2].unlink()

def save_file(file_name, image):
    pool.submit(save_worker, str(file_name), make_shared(image))

def export_all_variants(file_name, config):
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

    primary_tags = list(primary_tags)
    primary_tags.sort()
    primary_tags = pvector(primary_tags)

    for primary_tag in primary_tags:
        config._file_cache = {}
        enabled = pvector([primary_tag])
        if not config.only_secondary_tags:
            export_variant(psd, file_name, config, enabled, enabled)
        export_combinations(psd, file_name, config, secondary_tags, enabled, enabled)

if __name__ == '__main__':
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
    parser.add_argument('--mosaic_factor', default=100, type=float,
        help='Set the mosaic proportion factor of censors, based on the minimum image dimension.')
    parser.add_argument('--file_name', type=str, default='*.psd',
        help='PSD files to process; can use a glob pattern.')
    args = parser.parse_args()

    start = time.perf_counter()
    for file_name in glob.iglob(args.file_name):
        export_all_variants(file_name, args)
    pool.shutdown()
    logging.info(f'export time: {time.perf_counter() - start}')
