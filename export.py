import argparse
import pathlib
import re
import time
import logging
from enum import Enum, auto

from PIL import Image
from psd_tools import PSDImage

import composite

class CensorMode(Enum):
    NONE = auto()
    MOSAIC = auto()
    SOLID = auto()

tag_regex = re.compile('\[(.+)\]')
censor_regex = re.compile('censor')
censor_solid_regex = re.compile('censor solid')

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
            censor_layer.visible = True
            censor_composites.append(censor_layer.composite(viewport=viewport))
            censor_layer.visible = False
    if len(censor_composites) > 0:
        composite = censor_composites[0]
        for other_composite in censor_composites[1:]:
            composite = Image.alpha_composite(composite, other_composite)
        return composite
    return None

def export_variant(psd, file_name, show_tag, censor_mode):
    show_tag_regex = re.compile(f'\[({re.escape(show_tag)})\]')

    for regex in [tag_regex, censor_regex, censor_solid_regex]:
        for layer in find_layers(psd, regex):
            layer.visible = False

    show_layers = find_layers(psd, show_tag_regex)
    for layer in show_layers:
        layer.visible = True

    png_path = pathlib.Path(file_name).with_suffix('.png')
    image = None

    if censor_mode == CensorMode.MOSAIC:
        image = composite.composite(psd)
        mask = get_censor_composite_mask(show_layers, psd.viewbox)
        if mask:
            image = apply_mosaic(image, mask)
        png_path = png_path.parent / 'censor' / png_path.name
    elif censor_mode == CensorMode.SOLID:
        for layer in show_layers:
            for sublayer in find_layers(layer, censor_solid_regex):
                sublayer.visible = True
        image = composite.composite(psd)
        png_path = png_path.parent / 'censor-solid' / png_path.name
    else:
        image = composite.composite(psd)

    export_path = png_path
    if len(show_layers) > 0:
        tag = show_tag_regex.search(show_layers[0].name)[1]
        export_path = png_path.with_stem(f'{png_path.stem}-{tag}')

    export_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(export_path)
    logging.info(f'exported: {export_path}')

def export_all_variants(file_name):
    start = time.perf_counter()
    psd = PSDImage.open(file_name)
    tags = set()

    tagged_layers = find_layers(psd, tag_regex)
    for layer in tagged_layers:
        for tag in tag_regex.findall(layer.name):
            tags.add(tag)

    has_censors = len(find_layers(psd, censor_regex)) > 0
    has_censor_solids = len(find_layers(psd, censor_solid_regex)) > 0

    if len(tags) == 0:
        tags.add('')

    for tag in tags:
        export_variant(psd, file_name, tag, CensorMode.NONE)
        if has_censors:
            export_variant(psd, file_name, tag, CensorMode.MOSAIC)
        if has_censor_solids:
            export_variant(psd, file_name, tag, CensorMode.SOLID)

    logging.info(f'export time: {time.perf_counter() - start}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    args = parser.parse_args()
    export_all_variants(args.file_name)
