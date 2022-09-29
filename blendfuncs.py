import numpy as np
from psd_tools.constants import BlendMode

# https://dev.w3.org/SVG/modules/compositing/master/

def comp(C, A):
    return C * (1 - A)

def normal(Cd, Cs, Ad, As):
    return Cs + comp(Cd, As)

def multiply(Cd, Cs, Ad, As):
    return Cs * Cd + comp(Cs, Ad) + comp(Cd, As)

def screen(Cd, Cs, Ad, As):
    return Cs + Cd - Cs * Cd

def overlay(Cd, Cs, Ad, As):
    return hard_light(Cs, Cd, As, Ad)

# SAI Shade
def linear_burn(Cd, Cs, Ad, As):
    return Cd + Cs - As

# SAI Shine
def linear_dodge(Cd, Cs, Ad, As):
    return Cd + Cs

# SAI Shade/Shine
def linear_light(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 > As
    c = comp(Cs, Ad) + comp(Cd, As)
    B = linear_burn(Cd, Cs2, Ad, As)
    B[index] = linear_dodge(Cd, Cs2 - 1, Ad, As)[index]
    return B

def hard_light(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 > As
    c = comp(Cs, Ad) + comp(Cd, As)
    B = Cs2 * Cd + c
    B[index] = (As * Ad - 2 * (Ad - Cd) * (As - Cs) + c)[index]
    return B

def darken(Cd, Cs, Ad, As):
    return np.minimum(Cs * Ad, Cd * As) + comp(Cs, Ad) + comp(Cd, As)

def lighten(Cd, Cs, Ad, As):
    return np.maximum(Cs * Ad, Cd * As) + comp(Cs, Ad) + comp(Cd, As)

def color_burn(Cd, Cs, Ad, As):
    pass

blend_modes = {
    BlendMode.NORMAL: normal,
    BlendMode.MULTIPLY: multiply,
    BlendMode.SCREEN: screen,
    BlendMode.OVERLAY: overlay,
    BlendMode.LINEAR_BURN: linear_burn,
    BlendMode.LINEAR_DODGE: linear_dodge,
    BlendMode.LINEAR_LIGHT: linear_light,
    # BlendMode.COLOR_BURN: color_burn,
    # BlendMode.COLOR_DODGE: color_dodge,
    # BlendMode.VIVID_LIGHT: vivid_light,
    BlendMode.HARD_LIGHT: hard_light,
    # BlendMode.SOFT_LIGHT: soft_light,
    # BlendMode.PIN_LIGHT: pin_light,
    # BlendMode.HARD_MIX: hard_mix,
    BlendMode.DARKEN: darken,
    BlendMode.LIGHTEN: lighten,
    # BlendMode.DIVIDE: divide,
    # BlendMode.DIFFERENCE: difference,
    # BlendMode.EXCLUSION: exclusion,
    # BlendMode.SUBTRACT: subtract,
    # BlendMode.HUE: hue,
    # BlendMode.SATURATION: saturation,
    # BlendMode.COLOR: color,
    # BlendMode.LUMINOSITY: luminosity,
    # BlendMode.DARKER_COLOR: darker_color,
    # BlendMode.LIGHTER_COLOR: lighter_color,
    # BlendMode.DISSOLVE: dissolve,
}

def normal_alpha(Ad, As):
    return Ad + As - Ad * As

def get_blend_func(blend_mode):
    return blend_modes.get(blend_mode, normal)
