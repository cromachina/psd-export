import numpy as np
from psd_tools.constants import BlendMode

def comp(C, A):
    return C * (1 - A)

def normal(Cd, Cs, Ad, As):
    return Cs + comp(Cd, As)

def multiply(Cd, Cs, Ad, As):
    return Cs * Cd + comp(Cs, Ad) + comp(Cd, As)

def linear_dodge(Cd, Cs, Ad, As):
    return np.minimum(Cd + Cs, 1)

blend_modes = {
    BlendMode.NORMAL: normal,
    BlendMode.MULTIPLY: multiply,
    BlendMode.LINEAR_DODGE: linear_dodge,
}

def get_blend_func(blend_mode):
    return blend_modes.get(blend_mode, normal)
