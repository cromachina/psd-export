#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np
from psd_tools.constants import BlendMode

# https://dev.w3.org/SVG/modules/compositing/master/
# http://ssp.impulsetrain.com/porterduff.html
# https://photoblogstop.com/photoshop/photoshop-blend-modes-explained

cdef inline double _clip(double val) noexcept nogil:
    return max(min(val, 1), 0)

@cython.ufunc
cdef double clip(double val) noexcept nogil:
    return _clip(val)

cdef double eps = np.finfo(np.float64).eps

cdef inline double _safe_divide(double a, double b) noexcept nogil:
    return a / (b + eps)

@cython.ufunc
cdef double safe_divide(double a, double b) noexcept nogil:
    return _safe_divide(a, b)

cdef inline double _clip_divide(double a, double b) noexcept nogil:
    return _clip(_safe_divide(a, b))

@cython.ufunc
cdef double clip_divide(double a, double b) noexcept nogil:
    return _clip_divide(a, b)

cdef inline double _comp(double C, double A) noexcept nogil:
    return C * (1 - A)

@cython.ufunc
cdef double comp(double a, double b) noexcept nogil:
    return _comp(a, b)

cdef inline double _comp2(double Cd, double Cs, double Ad, double As) noexcept nogil:
    return _comp(Cd, As) + _comp(Cs, Ad)

@cython.ufunc
cdef double comp2(double Cd, double Cs, double Ad, double As)  noexcept nogil:
    return _comp2(Cd, Cs, Ad, As)

cdef inline double _lerp(double a, double b, double t) noexcept nogil:
    return (b - a) * t + a

@cython.ufunc
cdef double lerp(double a, double b, double t) noexcept nogil:
    return _lerp(a, b, t)

@cython.ufunc
cdef double normal(double Cd, double Cs, double Ad, double As) nogil:
    return Cs + _comp(Cd, As)

@cython.ufunc
cdef double normal_alpha(double Ad, double As) nogil:
    return Ad + As - Ad * As

cdef inline double _premul(double Cd, double Cs, double Ad, double As, double(*non_premul_func)(double, double) noexcept nogil) noexcept nogil:
    Cdp = _clip_divide(Cd, Ad)
    Csp = _clip_divide(Cs, As)
    B = non_premul_func(Cdp, Csp)
    Asrc = _comp(As, Ad)
    Adst = _comp(Ad, As)
    Aboth = As * Ad
    return Csp * Asrc + Cdp * Adst + Aboth * B

@cython.ufunc
cdef double multiply(double Cd, double Cs, double Ad, double As) nogil:
    return Cs * Cd + _comp2(Cd, Cs, Ad, As)

cdef inline double _screen(double Cd, double Cs) noexcept nogil:
    return Cs + Cd - Cs * Cd

@cython.ufunc
cdef double screen(double Cd, double Cs, double Ad, double As) nogil:
    return _screen(Cd, Cs)

def overlay(Cd, Cs, Ad, As):
    return hard_light(Cs, Cd, As, Ad)

@cython.ufunc
cdef double sai_linear_burn(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = Cdd + Cs - As
    H = _clip(H)
    return _lerp(Cs, H, Ad)

cdef inline double ts_linear_burn_non_premul(double Cd, double Cs) noexcept nogil:
    return _clip(Cd + Cs - 1)

@cython.ufunc
cdef double ts_linear_burn(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_burn_non_premul)

@cython.ufunc
cdef double sai_linear_dodge(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = Cdd + Cs
    H = _clip(H)
    return _lerp(Cs, H, Ad)

cdef inline double ts_linear_dodge_non_premul(double Cd, double Cs) noexcept nogil:
    return _clip(Cd + Cs)

@cython.ufunc
cdef double ts_linear_dodge(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_dodge_non_premul)

@cython.ufunc
cdef double sai_linear_light(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    Cs2 = Cs * 2
    LB = Cdd + Cs2 - As
    LB = _clip(LB)
    return _lerp(Cs, LB, Ad)

cdef inline double ts_linear_light_non_premul(double Cd, double Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return ts_linear_dodge_non_premul(Cd, Cs2 - 1)
    else:
        return ts_linear_burn_non_premul(Cd, Cs2)

@cython.ufunc
cdef double ts_linear_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_light_non_premul)

cdef inline double ts_color_burn_non_premul(double Cd, double Cs) noexcept nogil:
    return 1 - _clip_divide(1 - Cd, Cs)

@cython.ufunc
cdef double ts_color_burn(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_color_burn_non_premul)

@cython.ufunc
cdef double sai_color_burn(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    B = 1 - _clip_divide(1 - Cdd, 1 - As + Cs)
    return _lerp(Cs, B, Ad)

cdef inline double ts_color_dodge_non_premul(double Cd, double Cs) noexcept nogil:
    return _clip_divide(Cd, 1 - Cs)

@cython.ufunc
cdef double ts_color_dodge(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_color_dodge_non_premul)

@cython.ufunc
cdef double sai_color_dodge(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = _clip_divide(Cdd, 1 - Cs)
    return _lerp(Cs, H, Ad)

cdef inline double ts_vivid_light_non_premul(double Cd, double Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return ts_color_dodge_non_premul(Cd, Cs2 - 1)
    else:
        return ts_color_burn_non_premul(Cd, Cs2)

@cython.ufunc
cdef double ts_vivid_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_vivid_light_non_premul)

@cython.ufunc
cdef double sai_vivid_light(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    Csd = _clip_divide(Cs, As)
    Cs2 = As - Cs * 2
    CB = 1 - _clip_divide(1 - Cdd, 1 - Cs2)
    CD = _clip_divide(Cdd, 1 + Cs2)
    if Csd > 0.5:
        CB = CD
    if Cs == 1:
        CB = 1
    return _lerp(Cs, CB, Ad)

cdef extern from "math.h":
    double sqrt(double x) noexcept nogil

cdef inline double soft_light_non_premul(double Cd, double Cs) noexcept nogil:
    cdef double D, B
    if Cs <= 0.25:
        D = ((16 * Cd - 12) * Cd + 4) * Cd
    else:
        D = sqrt(Cd)
    if Cs <= 0.5:
        B = Cd - (1 - 2 * Cs) * Cd * (1 - Cd)
    else:
        B = Cd + (2 * Cs - 1) * (D - Cd)
    return B

@cython.ufunc
cdef double soft_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, soft_light_non_premul)

cdef inline double hard_light_non_premul(double Cd, double Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return _screen(Cd, Cs2 - 1)
    else:
        return Cd * Cs2

@cython.ufunc
cdef double hard_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, hard_light_non_premul)

cdef inline double pin_light_non_premul(double Cd, double Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return max(Cs2 - 1, Cd)
    else:
        return min(Cs2, Cd)

@cython.ufunc
cdef double pin_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, pin_light_non_premul)

@cython.ufunc
cdef double sai_hard_mix(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    Csd = _clip_divide(Cs, As)
    H = _clip_divide(Cdd - As + As * Csd, 1 - As)
    return _lerp(Cs, H, Ad)

cdef inline double ts_hard_mix_non_premul(double Cd, double Cs) noexcept nogil:
    if (Cd + Cs) > 1:
        return 1
    else:
        return 0

@cython.ufunc
cdef double ts_hard_mix(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_hard_mix_non_premul)

@cython.ufunc
cdef double darken(double Cd, double Cs, double Ad, double As) nogil:
    return min(Cs * Ad, Cd * As) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef double lighten(double Cd, double Cs, double Ad, double As) nogil:
    return max(Cs * Ad, Cd * As) + _comp2(Cd, Cs, Ad, As)

def ensure_array(D, S):
    return np.full_like(S, D) if np.isscalar(D) else D

# Premul for nonseperable blendfuncs.
def premul(non_premul_func):
    def to_premul(Cd, Cs, Ad, As):
        Cd = ensure_array(Cd, Cs)
        Ad = ensure_array(Ad, As)
        Cdp = clip_divide(Cd, Ad)
        Csp = clip_divide(Cs, As)
        B = non_premul_func(Cdp, Csp)
        Asrc = comp(As, Ad)
        Adst = comp(Ad, As)
        Aboth = As * Ad
        return Csp * Asrc + Cdp * Adst + Aboth * B
    return to_premul

@premul
def darker_color(Cd, Cs):
    index = lum(Cs) < lum(Cd)
    B = Cd.copy()
    B[index] = Cs[index]
    return B

@premul
def lighter_color(Cd, Cs):
    index = lum(Cs) > lum(Cd)
    B = Cd.copy()
    B[index] = Cs[index]
    return B

@cython.ufunc
cdef double ts_difference(double Cd, double Cs, double Ad, double As) nogil:
    return Cs + Cd - 2 * min(Cd * As, Cs * Ad)

@cython.ufunc
cdef double sai_difference(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    D = abs(Cdd - Cs)
    return _lerp(Cs, D, Ad)

@cython.ufunc
cdef double exclusion(double Cd, double Cs, double Ad, double As) nogil:
    return (Cs * Ad + Cd * As - 2 * Cs * Cd) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef double subtract(double Cd, double Cs, double Ad, double As) nogil:
    return max(0, Cd * As - Cs * Ad) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef double divide(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, _clip_divide)

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
    C_max = np.repeat(np.max(C, axis=2, keepdims=True), repeats=3, axis=2)
    C_mid = np.repeat(np.median(C, axis=2, keepdims=True), repeats=3, axis=2)
    C_min = np.repeat(np.min(C, axis=2, keepdims=True), repeats=3, axis=2)
    i_diff = C_max > C_min
    i_mid = C == C_mid
    i_max = (C == C_max) & ~i_mid
    i_min = C == C_min
    C = np.zeros_like(C)
    C[i_diff & i_mid] = safe_divide((C_mid - C_min) * S, (C_max - C_min))[i_diff & i_mid]
    C[i_diff & i_max] = S[i_diff & i_max]
    C[~i_diff & i_mid] = 0
    C[~i_diff & i_max] = 0
    C[i_min] = 0
    return C

# Small errors
@premul
def hue(Cd, Cs):
    return set_lum(set_sat(Cs, sat(Cd)), lum(Cd))

@premul
def saturation(Cd, Cs):
    return set_lum(set_sat(Cd, sat(Cs)), lum(Cd))

@premul
def color(Cd, Cs):
    return set_lum(Cs, lum(Cd))

@premul
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

def get_blend_func(blend_mode, special_mode):
    if special_mode:
        return special_blend_modes.get(blend_mode, normal)
    else:
        return blend_modes.get(blend_mode, normal)
