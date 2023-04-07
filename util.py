import numpy as np

def swap(a):
    x, y = a
    return y, x

def clamp(min_val, max_val, val):
    return max(min_val, min(max_val, val))

def safe_divide(a, b, out=None):
    with np.errstate(divide='ignore', invalid='ignore'):
        eps = np.finfo(np.float64).eps
        return np.divide(a, b + eps, out=out)

def lerp(a, b, t):
    return a + t * (b - a)

def clip(val, out=None):
    return np.clip(val, 0, 1, out=out)

def full(shape, fill, dtype=None):
    if fill == 0:
        return np.zeros(shape, dtype=dtype)
    elif fill == 1:
        return np.ones(shape, dtype=dtype)
    else:
        return np.full(shape, fill_value=fill, dtype=dtype)
