import argparse
import multiprocessing as mp
import pathlib
import re
import time
import logging
from enum import Enum, auto

from PIL import Image
from psd_tools import PSDImage

class CensorMode(Enum):
    NONE = auto()
    MOSAIC = auto()
    TWITTER = auto()

tag_regex = re.compile('\[(.+)\]')
censor_regex = re.compile('censor')
censor_tw_regex = re.compile('censor-tw')

def find_layers(layer, regex):
    return list(filter(lambda sublayer: regex.search(sublayer.name), layer.descendants()))

def find_layer(layer, exact_name):
    for sublayer in layer.descendants():
        if sublayer == exact_name:
            return sublayer

def apply_mosaic(image, mask):
    original_size = image.size
    min_dim = min(original_size) // 100
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

def export_variant(file_name, show_tag, censor_mode):
    logging.basicConfig(level=logging.INFO)
    try:
        psd = PSDImage.open(file_name)
        show_tag_regex = re.compile(f'\[({show_tag})\]')

        for regex in [tag_regex, censor_regex, censor_tw_regex]:
            for layer in find_layers(psd, regex):
                layer.visible = False

        show_layers = find_layers(psd, show_tag_regex)
        for layer in show_layers:
            layer.visible = True

        png_path = pathlib.Path(file_name).with_suffix('.png')
        image = None

        if censor_mode == CensorMode.MOSAIC:
            image = psd.composite(ignore_preview=True)
            mask = get_censor_composite_mask(show_layers, psd.viewbox)
            if mask:
                image = apply_mosaic(image, mask)
            png_path = png_path.parent / 'censor' / png_path.name
        elif censor_mode == CensorMode.TWITTER:
            for layer in show_layers:
                for sublayer in find_layers(layer, censor_tw_regex):
                    sublayer.visible = True
            image = psd.composite(ignore_preview=True)
            png_path = png_path.parent / 'censor-tw' / png_path.name
        else:
            image = psd.composite(ignore_preview=True)

        export_path = png_path
        if len(show_layers) > 0:
            tag = show_tag_regex.search(show_layers[0].name)[1]
            export_path = png_path.with_stem(f'{png_path.stem}-{tag}')

        export_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(export_path)
    except Exception as e:
        logging.exception(e)

def export_variants(file_name):
    with mp.Pool(processes=24) as pool:
        start = time.perf_counter()
        psd = PSDImage.open(file_name)
        tags = set()

        tagged_layers = find_layers(psd, tag_regex)
        for layer in tagged_layers:
            for tag in tag_regex.findall(layer.name):
                tags.add(tag)

        has_censors = len(find_layers(psd, censor_regex)) > 0
        has_tw_censors = len(find_layers(psd, censor_regex)) > 0

        for tag in tags:
            pool.apply_async(export_variant, (file_name, tag, CensorMode.NONE))
            if has_censors:
                pool.apply_async(export_variant, (file_name, tag, CensorMode.MOSAIC))
            if has_tw_censors:
                pool.apply_async(export_variant, (file_name, tag, CensorMode.TWITTER))
        pool.close()
        pool.join()
        print(time.perf_counter() - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    args = parser.parse_args()
    export_variants(args.file_name)
