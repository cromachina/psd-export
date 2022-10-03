import numpy as np
from psd_tools.constants import BlendMode
from psd_tools.composite import blend

# https://dev.w3.org/SVG/modules/compositing/master/
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

def clip(color):
    return np.clip(color, 0, 1)

def safe_divide(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return a / (b + np.finfo(a.dtype).eps)

# Turn a non-premultiplied blend func into a premultiplied one.
# The result may sometimes look a little bit different from SAI.
def to_premul(non_premul_func):
    def fn(Cd, Cs, Ad, As):
        Cdp = clip(safe_divide(Cd, Ad))
        Csp = clip(safe_divide(Cs, As))
        Csp = non_premul_func(Cdp, Csp)
        Csp *= As
        return normal(Cd, Csp, Ad, As)
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
    return Cd + Cs - As

# SAI Shine
def linear_dodge(Cd, Cs, Ad, As):
    return Cd + Cs

# SAI Shade/Shine
def linear_light(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 > As
    B = linear_burn(Cd, Cs2, Ad, As)
    B[index] = linear_dodge(Cd, Cs2 - As, Ad, As)[index]
    return B

# PS/CSP Color Burn, SAI's is unknown
def color_burn(Cd, Cs, Ad, As):
    index = Cs == 0
    index2 = index & np.isclose(Cd, Ad)
    index3 = Cs > 0
    c = comp(Cd, As)
    AsAd = As * Ad
    B = np.zeros_like(Cs)
    B[index3] = (AsAd * (1 - np.minimum(1, safe_divide(As * (Ad - Cd), Ad * Cs))) + comp(Cs, Ad) + c)[index3]
    B[index] = c[index]
    B[index2] = (AsAd + c)[index2]
    return B

# PS/CSP Color Dodge, SAI's is unknown
def color_dodge(Cd, Cs, Ad, As):
    index = np.isclose(Cs, As)
    index2 = index & (Cd == 0)
    index3 = Cs < As
    c1 = comp(Cs, Ad)
    c2 = comp(Cd, As) + c1
    B = np.zeros_like(Cs)
    B[index3] = (As * Ad * np.minimum(1, safe_divide(Cd * As, Ad * (As - Cs))) + c2)[index3]
    B[index] = (As * Ad + c2)[index]
    B[index2] = c1[index2]
    return B

# PS/CSP Vivid Light, SAI's is unknown
# Technically correct Vivid Light? Seems like everyone else's
# vivid light is messed up at 100% opacity/fill with clipped/buggy pixels.
# Maybe this is related to hard light?
# SAI: Nonlinear
def vivid_light(Cd, Cs, Ad, As):
    Cs2 = 2 * Cs
    index = Cs2 > As
    B = color_burn(Cd, Cs2, Ad, As)
    B[index] = color_dodge(Cd, Cs2 - As, Ad, As)[index]
    return B

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
    m = safe_divide(Cd, Ad)
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

def lerp(a, b, t):
    return a + t * (b - a)

hard_mix = to_premul(blend.hard_mix)

# FIXME broken hard mix
# SAI: Very nonlinear.. some sort of inverse logarithmic blend function for the alpha
# that I have yet reverse engineer. It seems to have a similar out nonlinear
# output as vivid light, but squashed into a range that causes clipping. That
# may be the trick to fixing this.
def hard_mix_broken(Cd, Cs, Ad, As):
    F = 4
    V = vivid_light(Cd, Cs * F, Ad, As * F)
    return normal(Cd, V * As, Ad, As)

    Cdd = clip(safe_divide(Cd, Ad))
    Csd = clip(safe_divide(Cs, As))
    As2 = np.maximum(As - 0.5, 0) * 2
    C = Cdd + Csd
    index = C > 1
    H = np.zeros_like(Cs)
    H[index] = 1
    H = lerp(Csd, H, clip(1 / (np.log2(1-As) * 2)))
    # H = As2 * H + (1 - As2) * Csd
    #H *= As
    # H = normal(Csd, H, 1, As2)
    #H = normal(Cd, H, 1, As)
    return H

def darken(Cd, Cs, Ad, As):
    return np.minimum(Cs * Ad, Cd * As) + comp2(Cd, Cs, Ad, As)

def lighten(Cd, Cs, Ad, As):
    return np.maximum(Cs * Ad, Cd * As) + comp2(Cd, Cs, Ad, As)

darker_color = to_premul(blend.darker_color)

lighter_color = to_premul(blend.lighter_color)

# SAI Difference
# SAI: Seemingly linear
def difference(Cd, Cs, Ad, As):
    return Cs + Cd - 2 * np.minimum(Cd * As, Cs * Ad)

def exclusion(Cd, Cs, Ad, As):
    return (Cs * Ad + Cd * As - 2 * Cs * Cd) + comp2(Cd, Cs, Ad, As)

def subtract(Cd, Cs, Ad, As):
    return  Cd + Cs - 2 * Cs * Ad

def divide(Cd, Cs, Ad, As):
    return safe_divide(Cd * As, Cs * Ad) + comp2(Cd, Cs, Ad, As)

hue = to_premul(blend.hue)

saturation = to_premul(blend.saturation)

# FIXME Broken color
color = to_premul(blend.color)

luminosity = to_premul(blend.luminosity)

blend_modes = {
    BlendMode.NORMAL: normal,
    BlendMode.MULTIPLY: multiply,
    BlendMode.SCREEN: screen,
    BlendMode.OVERLAY: overlay,
    BlendMode.LINEAR_BURN: linear_burn,
    BlendMode.LINEAR_DODGE: linear_dodge,
    BlendMode.LINEAR_LIGHT: linear_light,
    BlendMode.COLOR_BURN: color_burn,
    BlendMode.COLOR_DODGE: color_dodge,
    BlendMode.VIVID_LIGHT: vivid_light,
    BlendMode.HARD_LIGHT: hard_light,
    BlendMode.SOFT_LIGHT: soft_light,
    BlendMode.PIN_LIGHT: pin_light,
    BlendMode.HARD_MIX: hard_mix,
    BlendMode.DARKEN: darken,
    BlendMode.LIGHTEN: lighten,
    BlendMode.DARKER_COLOR: darker_color,
    BlendMode.LIGHTER_COLOR: lighter_color,
    BlendMode.DIFFERENCE: difference,
    BlendMode.EXCLUSION: exclusion,
    BlendMode.SUBTRACT: subtract,
    BlendMode.DIVIDE: divide,
    BlendMode.HUE: hue,
    BlendMode.SATURATION: saturation,
    BlendMode.COLOR: color,
    BlendMode.LUMINOSITY: luminosity,
    # BlendMode.DISSOLVE: dissolve,
}

def normal_alpha(Ad, As):
    return Ad + As - Ad * As

def get_blend_func(blend_mode):
    return blend_modes.get(blend_mode, normal)
