import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import psutil

file_writer_futures = []

worker_count = psutil.cpu_count(False)
pool = ThreadPoolExecutor(max_workers=worker_count)

def peval(func):
    return asyncio.get_running_loop().run_in_executor(pool, func)

def swap(a):
    x, y = a
    return y, x

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

def clip(val, out=None):
    return np.clip(val, 0, 1, out=out)

def clip_in(val):
    return clip(val, val)

def safe_divide(a, b, out=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = np.finfo(a.dtype).eps
        return np.divide(a, b + eps, out=out)

def clip_divide(a, b, out=None):
    out = safe_divide(a, b, out=out)
    return clip_in(out)

def lerp(a, b, t, out=None):
    out = np.subtract(b, a, out=out)
    np.multiply(t, out, out=out)
    return np.add(a, out, out=out)

def full(shape, fill, dtype=None):
    if fill == 0:
        return np.zeros(shape, dtype=dtype)
    elif fill == 1:
        return np.ones(shape, dtype=dtype)
    else:
        return np.full(shape, fill_value=fill, dtype=dtype)

def is_grayscale(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    return (r == g).all() and (g == b).all()

def save_worker(file_name, image, imwrite_args=[]):
    image = np.dstack(image)
    image *= 255
    image = image.astype(np.uint8)
    if (image[:,:,3] == 255).all():
        if is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(file_name, image, imwrite_args)
    logging.info(f'exported: {file_name}')

def save_file(file_name, image, imwrite_args=[]):
    file_writer_futures.append(peval(lambda: save_worker(str(file_name), image, imwrite_args)))

async def save_workers_wait_all():
    await asyncio.gather(*file_writer_futures)
    file_writer_futures.clear()
