import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

file_writer_pool = ProcessPoolExecutor(1)
file_writer_futures = []

def swap(a):
    x, y = a
    return y, x

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

def clip(val, out=None):
    return np.clip(val, 0, 1, out=out)

def safe_divide(a, b, out=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = np.finfo(np.float64).eps
        return np.divide(a, b + eps, out=out)

def clip_divide(a, b, out=None):
    out = safe_divide(a, b, out=out)
    return clip(out, out=out)

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

def make_shared(image):
    sm = SharedMemory(create=True, size=image.nbytes)
    a = np.ndarray(image.shape, dtype=image.dtype, buffer=sm.buf)
    np.copyto(a, image)
    return (image.shape, image.dtype, sm)

def delete_shared(image_sm):
    image_sm[2].close()
    image_sm[2].unlink()

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

def save_file(file_name, image):
    file_writer_futures.append(file_writer_pool.submit(save_worker, str(file_name), make_shared(image)))

def file_writer_wait_all():
    for f in file_writer_futures:
        f.result()
    file_writer_futures.clear()
