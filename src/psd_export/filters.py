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
    def c(color_dst, color_src, alpha_dst, alpha_src):
        for op in ops:
            color_dst, alpha_dst = op(color_dst, color_src, alpha_dst, alpha_src)
        return color_dst, alpha_dst
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
def mosaic_op(color_dst, color_src, alpha_dst, alpha_src, mosaic_factor=None, apply_to_alpha=False, *_):
    if mosaic_factor is None:
        mosaic_factor = mosaic_factor_default
    mosaic_factor = int(mosaic_factor)
    color = mosaic(color_dst, mosaic_factor)
    color = util.lerp(color_dst, color, alpha_src)
    alpha = alpha_dst
    if apply_to_alpha:
        alpha = mosaic(alpha_dst, mosaic_factor)
        alpha = util.lerp(alpha_dst, alpha, alpha_src)
    return color, alpha

@filter('blur')
def blur_op(color_dst, color_src, alpha_dst, alpha_src, size=50, *_):
    size = float(size)
    color = cv2.GaussianBlur(color_dst, ksize=(0, 0), sigmaX=size, borderType=cv2.BORDER_REPLICATE)
    alpha = cv2.GaussianBlur(alpha_dst, ksize=(0, 0), sigmaX=size, borderType=cv2.BORDER_REPLICATE).reshape(alpha_src.shape)
    color = util.lerp(color_dst, color, alpha_src)
    alpha = util.lerp(alpha_dst, alpha, alpha_src)
    return color, alpha

def motion_blur(data, angle, size):
    kernel = np.zeros((size, size))
    kernel[(size - 1) // 2] = 1
    rotation = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    kernel *= (1.0 / np.sum(kernel))
    return cv2.filter2D(data, -1, kernel)

@filter('motion-blur')
def motion_blur_op(color_dst, color_src, alpha_dst, alpha_src, angle=0, size=50, *_):
    angle = float(angle)
    size = int(size)
    color = motion_blur(color_dst, angle, size)
    alpha = motion_blur(alpha_dst, angle, size).reshape(alpha_src.shape)
    color = util.lerp(color_dst, color, alpha_src)
    alpha = util.lerp(alpha_dst, alpha, alpha_src)
    return color, alpha
