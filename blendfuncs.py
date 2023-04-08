import numpy as np
from psd_tools.composite import blend
from psd_tools.constants import BlendMode

import util

# https://dev.w3.org/SVG/modules/compositing/master/
# http://ssp.impulsetrain.com/porterduff.html
# How to convert from non-premultiplied blend functions to premultiplied ones:
# Cd = Color Destination
# Cs = Color Source
# Ad = Alpha Destination
# As = Alpha Source
# - The form (C * (1 - A)) we will call 'comp(C, A)', for complementary alpha.
# - Any addition or subtraction of a constant N from the non-premultiplied forumla becomes
#   addition or subtraction of (N * As). So 1 -> As and 0 -> 0
# - Any constant multiplication or exponential is unchanged.
# - When a function is applied over a combination of Cd and Cs,
#   the comps are added to the result of the function:
# f(Cd) -> f(Cd * As) + comp(Cd, As)
# f(Cs) -> f(Cs * Ad) + comp(Cs, Ad)
# f(Cd, Cs) -> f(Cd * As, Cs * Ad) + comp(Cd, As) + comp(Cs, Ad)
# - The result can be optimized by expanding and simplifying.
# - Color burn is the odd one out. Its conversion is a bit confusing.

# Turn a non-premultiplied blend func into a premultiplied one.
# The result may sometimes look a little bit different from SAI.
# This is a hit to performance too because of the two divides.
def to_premul(non_premul_func):
    def fn(Cd, Cs, Ad, As):
        Cdp = util.clip_divide(Cd, Ad)
        Csp = util.clip_divide(Cs, As)
        B = non_premul_func(Cdp, Csp)
        Asrc = As * (1 - Ad)
        Adst = Ad * (1 - As)
        Aboth = As * Ad
        return Csp * Asrc + Cdp * Adst + Aboth * B
    return fn

def comp(C, A):
    return C * (1 - A)

def comp2(Cd, Cs, Ad, As):
    return comp(Cd, As) + comp(Cs, Ad)

def normal(Cd, Cs, Ad, As):
    return Cs + comp(Cd, As)

def multiply(Cd, Cs, Ad, As):
    return Cs * Cd + comp2(Cd, Cs, Ad, As)

def screen(Cd, Cs, Ad, As):
    return Cs + Cd - Cs * Cd

def overlay(Cd, Cs, Ad, As):
    # Really similar to hard light, 2 * Cd is the index.
    Cd2 = 2 * Cd
    Cs2 = 2 * Cs
    index = Cd2 > Ad
    B = multiply(Cd, Cs2, Ad, As)
    B[index] = screen(Cd, Cs2 - As, Ad, As)[index]
    return B

# SAI Shade
def linear_burn(Cd, Cs, Ad, As):
    return Cd + Cs - As * Ad

# SAI Shine
def linear_dodge(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    Csd = util.clip_divide(Cs, As)
    H = Cdd + Csd
    util.clip(H, H)
    return util.lerp(Cs, H, Ad)

# SAI Shade/Shine
def linear_light(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 > As
    B = linear_burn(Cd, Cs2, Ad, As)
    B[index] = linear_dodge(Cd, Cs2 - As, Ad, As)[index]
    return B

# TS Color Burn, SAI's is unknown, nonlinear
def ts_color_burn(Cd, Cs, Ad, As):
    index = Cs == 0
    index2 = index & np.isclose(Cd, Ad)
    index3 = Cs > 0
    c = comp(Cd, As)
    AsAd = As * Ad
    B = np.zeros_like(Cs)
    B[index3] = (AsAd * (1 - util.clip_divide(As * (Ad - Cd), Ad * Cs)) + comp(Cs, Ad) + c)[index3]
    B[index] = c[index]
    B[index2] = (AsAd + c)[index2]
    return B

# TS Color Dodge, SAI's is unknown, nonlinear
def ts_color_dodge(Cd, Cs, Ad, As):
    index = np.isclose(Cs, As)
    index2 = index & (Cd == 0)
    index3 = Cs < As
    c1 = comp(Cs, Ad)
    c2 = comp(Cd, As) + c1
    B = np.zeros_like(Cs)
    B[index3] = (As * Ad * util.clip_divide(Cd * As, Ad * (As - Cs)) + c2)[index3]
    B[index] = (As * Ad + c2)[index]
    B[index2] = c1[index2]
    return B

# TS Vivid Light, SAI's is unknown, nonlinear
# Technically correct Vivid Light? Seems like everyone else's
# vivid light is messed up at 100% opacity/fill with clipped/buggy pixels.
# Maybe this is related to hard light?
def ts_vivid_light(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 > As
    B = ts_color_burn(Cd, Cs2, Ad, As)
    B[index] = ts_color_dodge(Cd, Cs2 - As, Ad, As)[index]
    return B

# Slightly different from SAI
soft_light = to_premul(blend.soft_light)

# FIXME broken soft light. This function is so confusing.
# SAI: Seemingly linear.
def soft_light_broken(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 <= As
    ia = ~index
    ib = (4 * Cd) <= Ad
    index2 = ia & ib
    index3 = ia & (~ib)
    m = util.safe_divide(Cd, Ad)
    B = np.zeros_like(Cs)
    x = Cs2 - As
    Adx = Ad * x
    y = Cs - (Cs * Ad) + Cd
    B[index3] = (Adx * (np.sqrt(m) - m) + y)[index3]
    B[index2] = (Adx * (16 * (m ** 3) - 12 * (m ** 2) - 3 * m) + y)[index2]
    B[index] = (Cd * (As + x * (1 - m)) + comp2(Cd, Cs, Ad, As))[index]
    return B

def hard_light(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 > As
    B = multiply(Cd, Cs2, Ad, As)
    B[index] = screen(Cd, Cs2 - As, Ad, As)[index]
    return B

def pin_light(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 > As
    B = darken(Cd, Cs2, Ad, As)
    B[index] = lighten(Cd, Cs2 - As, Ad, As)[index]
    return B

def hard_mix(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    Csd = util.clip_divide(Cs, As)
    H = util.clip_divide(Cdd - As + As * Csd, 1 - As)
    return util.lerp(Cs, H, Ad)

def ts_hard_mix(Cd, Cs, Ad, As):
    index = Cd * As + Cs * Ad >= As
    H = np.zeros_like(Cs)
    H[index] = 1
    H *= As
    return H + comp2(Cd, Cs, Ad, As)

def darken(Cd, Cs, Ad, As):
    return np.minimum(Cs * Ad, Cd * As) + comp2(Cd, Cs, Ad, As)

def lighten(Cd, Cs, Ad, As):
    return np.maximum(Cs * Ad, Cd * As) + comp2(Cd, Cs, Ad, As)

darker_color = to_premul(blend.darker_color)

lighter_color = to_premul(blend.lighter_color)

# TS Difference; SAI: seemingly linear
def ts_difference(Cd, Cs, Ad, As):
    return Cs + Cd - 2 * np.minimum(Cd * As, Cs * Ad)

def exclusion(Cd, Cs, Ad, As):
    return (Cs * Ad + Cd * As - 2 * Cs * Cd) + comp2(Cd, Cs, Ad, As)

def subtract(Cd, Cs, Ad, As):
    return  np.maximum(0, Cd * As - Cs * Ad) + comp2(Cd, Cs, Ad, As)

# Seems like divide doesn't work properly unless converted to non-premul first.
divide = to_premul(blend.divide)

# FIXME Broken
hue = to_premul(blend.hue)

# FIXME Broken
saturation = to_premul(blend.saturation)

color = to_premul(blend.color)

luminosity = to_premul(blend.luminosity)

blend_modes = {
    BlendMode.NORMAL: normal,
    BlendMode.MULTIPLY: multiply,
    BlendMode.SCREEN: screen,
    BlendMode.OVERLAY: overlay,
    BlendMode.LINEAR_BURN: to_premul(blend.linear_burn),
    BlendMode.LINEAR_DODGE: to_premul(blend.linear_dodge),
    BlendMode.LINEAR_LIGHT: to_premul(blend.linear_light),
    BlendMode.COLOR_BURN: ts_color_burn,
    BlendMode.COLOR_DODGE: ts_color_dodge,
    BlendMode.VIVID_LIGHT: ts_vivid_light,
    BlendMode.HARD_LIGHT: hard_light,
    BlendMode.SOFT_LIGHT: soft_light,
    BlendMode.PIN_LIGHT: pin_light,
    BlendMode.HARD_MIX: ts_hard_mix,
    BlendMode.DARKEN: darken,
    BlendMode.LIGHTEN: lighten,
    BlendMode.DARKER_COLOR: darker_color,
    BlendMode.LIGHTER_COLOR: lighter_color,
    BlendMode.DIFFERENCE: ts_difference,
    BlendMode.EXCLUSION: exclusion,
    BlendMode.SUBTRACT: subtract,
    BlendMode.DIVIDE: divide,
    BlendMode.HUE: hue,
    BlendMode.SATURATION: saturation,
    BlendMode.COLOR: color,
    BlendMode.LUMINOSITY: luminosity,
}

special_blend_modes = {
    BlendMode.LINEAR_BURN: linear_burn,
    BlendMode.LINEAR_DODGE: linear_dodge,
    BlendMode.LINEAR_LIGHT: linear_light,
    BlendMode.COLOR_BURN: ts_color_burn,
    BlendMode.COLOR_DODGE: ts_color_dodge,
    BlendMode.VIVID_LIGHT: ts_vivid_light,
    BlendMode.HARD_MIX: hard_mix,
    BlendMode.DIFFERENCE: ts_difference,
}

def normal_alpha(Ad, As):
    return Ad + As - Ad * As

def get_blend_func(blend_mode, special_mode):
    if special_mode:
        return special_blend_modes.get(blend_mode, normal)
    else:
        return blend_modes.get(blend_mode, normal)
