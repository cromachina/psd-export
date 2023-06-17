import numpy as np
from psd_tools.composite import blend
from psd_tools.constants import BlendMode

from . import util

# https://dev.w3.org/SVG/modules/compositing/master/
# http://ssp.impulsetrain.com/porterduff.html
# https://photoblogstop.com/photoshop/photoshop-blend-modes-explained

def comp(C, A):
    return C * (1 - A)

def comp2(Cd, Cs, Ad, As):
    return comp(Cd, As) + comp(Cs, Ad)

def premul(Cd, Cs, Ad, As, non_premul_func):
    Cdp = util.clip_divide(Cd, Ad)
    Csp = util.clip_divide(Cs, As)
    B = non_premul_func(Cdp, Csp)
    Asrc = comp(As, Ad)
    Adst = comp(Ad, As)
    Aboth = As * Ad
    return Csp * Asrc + Cdp * Adst + Aboth * B

def to_premul(non_premul_func):
    return lambda Cd, Cs, Ad, As: premul(Cd, Cs, Ad, As, non_premul_func)

def normal(Cd, Cs, Ad, As):
    return Cs + comp(Cd, As)

def multiply(Cd, Cs, Ad, As):
    return Cs * Cd + comp2(Cd, Cs, Ad, As)

def screen(Cd, Cs, Ad=None, As=None):
    return Cs + Cd - Cs * Cd

def overlay(Cd, Cs, Ad, As):
    return hard_light(Cs, Cd, As, Ad)

def sai_linear_burn(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    H = Cdd + Cs - As
    util.clip_in(H)
    return util.lerp(Cs, H, Ad, out=H)

def ts_linear_burn_non_premul(Cd, Cs):
    return util.clip_in(Cd + Cs - 1)

ts_linear_burn = to_premul(ts_linear_burn_non_premul)

def sai_linear_dodge(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    H = Cdd + Cs
    util.clip_in(H)
    return util.lerp(Cs, H, Ad, out=H)

def ts_linear_dodge_non_premul(Cd, Cs):
    return util.clip_in(Cd + Cs)

ts_linear_dodge = to_premul(ts_linear_dodge_non_premul)

def sai_linear_light(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    Cs2 = Cs * 2
    LB = Cdd + Cs2 - As
    util.clip_in(LB)
    return util.lerp(Cs, LB, Ad, out=LB)

@to_premul
def ts_linear_light(Cd, Cs):
    Cs2 = Cs * 2
    index = Cs > 0.5
    B = ts_linear_burn_non_premul(Cd, Cs2)
    D = ts_linear_dodge_non_premul(Cd, Cs2 - 1)
    B[index] = D[index]
    return B

def ts_color_burn_non_premul(Cd, Cs):
    return 1 - util.clip_divide(1 - Cd, Cs)

ts_color_burn = to_premul(ts_color_burn_non_premul)

# FIXME
def sai_color_burn(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    Csd = util.clip_divide(Cs, As)
    B = 1 - util.clip_divide(1 - Cdd, Cs) + comp(Cdd, As)
    return util.lerp(Cs, B, Ad, out=B)

def ts_color_dodge_non_premul(Cd, Cs):
    return util.clip_divide(Cd, 1 - Cs)

ts_color_dodge = to_premul(ts_color_dodge_non_premul)

def sai_color_dodge(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    H = util.clip_divide(Cdd, 1 - Cs)
    return util.lerp(Cs, H, Ad, out=H)

@to_premul
def ts_vivid_light(Cd, Cs):
    Cs2 = Cs * 2
    index = Cs > 0.5
    B = ts_color_burn_non_premul(Cd, Cs2)
    D = ts_color_dodge_non_premul(Cd, Cs2 - 1)
    B[index] = D[index]
    return B

# FIXME
def sai_vivid_light(Cd, Cs, Ad, As):
    return ts_vivid_light(Cd, Cs, Ad, As)
    Cs2 = 2 * Cs
    index = Cs2 > As
    B = ts_color_burn(Cd, Cs2, Ad, As)
    B[index] = ts_color_dodge(Cd, Cs2 - As, Ad, As)[index]
    return B

soft_light = to_premul(blend.soft_light)

@to_premul
def hard_light(Cd, Cs):
    Cs2 = Cs * 2
    index = Cs > 0.5
    M = Cd * Cs2
    S = screen(Cd, Cs2 - 1)
    M[index] = S[index]
    return M

@to_premul
def pin_light(Cd, Cs):
    Cs2 = Cs * 2
    index = Cs > 0.5
    D = np.minimum(Cs2, Cd)
    L = np.maximum(Cs2 - 1, Cd)
    D[index] = L[index]
    return D

def sai_hard_mix(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    Csd = util.clip_divide(Cs, As)
    H = util.clip_divide(Cdd - As + As * Csd, 1 - As)
    return util.lerp(Cs, H, Ad, out=H)

@to_premul
def ts_hard_mix(Cd, Cs):
    B = np.zeros_like(Cd)
    B[(Cd + Cs) > 1] = 1
    return B

def darken(Cd, Cs, Ad, As):
    return np.minimum(Cs * Ad, Cd * As) + comp2(Cd, Cs, Ad, As)

def lighten(Cd, Cs, Ad, As):
    return np.maximum(Cs * Ad, Cd * As) + comp2(Cd, Cs, Ad, As)

@to_premul
def darker_color(Cd, Cs):
    index = lum(Cs) < lum(Cd)
    B = Cd.copy()
    B[index] = Cs[index]
    return B

@to_premul
def lighter_color(Cd, Cs):
    index = lum(Cs) > lum(Cd)
    B = Cd.copy()
    B[index] = Cs[index]
    return B

def ts_difference(Cd, Cs, Ad, As):
    return Cs + Cd - 2 * np.minimum(Cd * As, Cs * Ad)

def sai_difference(Cd, Cs, Ad, As):
    Cdd = util.clip_divide(Cd, Ad)
    D = np.abs(Cdd - Cs)
    return util.lerp(Cs, D, Ad)

def exclusion(Cd, Cs, Ad, As):
    return (Cs * Ad + Cd * As - 2 * Cs * Cd) + comp2(Cd, Cs, Ad, As)

def subtract(Cd, Cs, Ad, As):
    return np.maximum(0, Cd * As - Cs * Ad) + comp2(Cd, Cs, Ad, As)

divide = to_premul(util.clip_divide)

def lum(C):
    return np.repeat(np.sum((0.3, 0.59, 0.11) * C, axis=2, keepdims=True), repeats=3, axis=2)

def clip_color(C):
    L = lum(C)
    C_min = np.repeat(np.min(C, axis=2, keepdims=True), repeats=3, axis=2)
    C_max = np.repeat(np.max(C, axis=2, keepdims=True), repeats=3, axis=2)
    i_min = C_min < 0.0
    i_max = C_max > 1.0
    L_min = L[i_min]
    L_max = L[i_max]
    C[i_min] = L_min + (C[i_min] - L_min) * L_min / (L_min - C_min[i_min])
    C[i_max] = L_max + (C[i_max] - L_max) * (1 - L_max) / (C_max[i_max] - L_max)
    return C

def set_lum(C, L):
    return clip_color(C + (L - lum(C)))

def sat(C):
    return np.repeat(np.max(C, axis=2, keepdims=True) - np.min(C, axis=2, keepdims=True), repeats=3, axis=2)

def set_sat(C, S):
    C = C.copy()
    C_max = np.repeat(np.max(C, axis=2, keepdims=True), repeats=3, axis=2)
    C_mid = np.repeat(np.median(C, axis=2, keepdims=True), repeats=3, axis=2)
    C_min = np.repeat(np.min(C, axis=2, keepdims=True), repeats=3, axis=2)
    i_diff = C_max > C_min
    i_mid = C == C_mid
    i_max = (C == C_max) & ~i_mid
    i_min = C == C_min
    C = np.zeros_like(C)
    C[i_diff & i_mid] = util.safe_divide((C_mid - C_min) * S, (C_max - C_min))[i_diff & i_mid]
    C[i_diff & i_max] = S[i_diff & i_max]
    C[~i_diff & i_mid] = 0
    C[~i_diff & i_max] = 0
    C[i_min] = 0
    return C

@to_premul
def hue(Cd, Cs):
    return set_lum(set_sat(Cs, sat(Cd)), lum(Cd))

@to_premul
def saturation(Cd, Cs):
    return set_lum(set_sat(Cd, sat(Cs)), lum(Cd))

@to_premul
def color(Cd, Cs):
    return set_lum(Cs, lum(Cd))

@to_premul
def luminosity(Cd, Cs):
    return set_lum(Cd, lum(Cs))

blend_modes = {
    BlendMode.NORMAL: normal,
    BlendMode.MULTIPLY: multiply,
    BlendMode.SCREEN: screen,
    BlendMode.OVERLAY: overlay,
    BlendMode.LINEAR_BURN: ts_linear_burn,
    BlendMode.LINEAR_DODGE: ts_linear_dodge,
    BlendMode.LINEAR_LIGHT: ts_linear_light,
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
    BlendMode.LINEAR_BURN: sai_linear_burn,
    BlendMode.LINEAR_DODGE: sai_linear_dodge,
    BlendMode.LINEAR_LIGHT: sai_linear_light,
    BlendMode.COLOR_BURN: sai_color_burn,
    BlendMode.COLOR_DODGE: sai_color_dodge,
    BlendMode.VIVID_LIGHT: sai_vivid_light,
    BlendMode.HARD_MIX: sai_hard_mix,
    BlendMode.DIFFERENCE: sai_difference,
}

def normal_alpha(Ad, As):
    return Ad + As - Ad * As

def get_blend_func(blend_mode, special_mode):
    if special_mode:
        return special_blend_modes.get(blend_mode, normal)
    else:
        return blend_modes.get(blend_mode, normal)
