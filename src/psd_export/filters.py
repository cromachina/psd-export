import cv2
import numpy as np

from . import util

filter_names = {}

def filter(name):
    def wrap(func):
        filter_names[name] = func
        return func
    return wrap

def get_filter(name):
    return filter_names.get(name, None)

def compose_ops(ops):
    if not ops:
        return None
    def c(color, alpha):
        for op in ops:
            color, alpha = op(color, alpha)
        return color, alpha
    return c

def mosaic(image, mosaic_factor):
    original_size = util.swap(image.shape[:2])
    min_dim = min(original_size) // mosaic_factor
    min_dim = max(4, min_dim)
    scale_dimension = (original_size[0] // min_dim, original_size[1] // min_dim)
    mosaic_image = cv2.resize(image, scale_dimension, interpolation=cv2.INTER_AREA)
    return cv2.resize(mosaic_image, original_size, interpolation=cv2.INTER_NEAREST).reshape(image.shape)

mosaic_factor_default = 100

@filter('censor')
def mosaic_op(color, alpha, mosaic_factor=None, apply_to_alpha=False, *_):
    if mosaic_factor is None:
        mosaic_factor = mosaic_factor_default
    mosaic_factor = int(mosaic_factor)
    color = mosaic(color, mosaic_factor)
    if apply_to_alpha:
        alpha = mosaic(alpha, mosaic_factor)
    return color, alpha

@filter('blur')
def blur_op(color, alpha, size=50, apply_to_alpha=False, *_):
    size = float(size)
    color = cv2.GaussianBlur(color, ksize=(0, 0), sigmaX=size, dst=color, borderType=cv2.BORDER_REPLICATE)
    if apply_to_alpha:
        alpha = cv2.GaussianBlur(alpha, ksize=(0, 0), sigmaX=size, dst=alpha, borderType=cv2.BORDER_REPLICATE)
    return color, alpha

def motion_blur(data, angle, size):
    kernel = np.zeros((size, size))
    kernel[(size - 1) // 2] = 1
    rotation = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    kernel *= (1.0 / np.sum(kernel))
    return cv2.filter2D(data, -1, kernel)

@filter('motion-blur')
def motion_blur_op(color, alpha, angle=0, size=50, apply_to_alpha=False, *_):
    angle = float(angle)
    size = int(size)
    color = motion_blur(color, angle, size)
    if apply_to_alpha:
        alpha = motion_blur(alpha, angle, size).reshape(alpha.shape)
    return color, alpha
