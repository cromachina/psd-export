#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np
from psd_tools.constants import BlendMode

# https://dev.w3.org/SVG/modules/compositing/master/
# http://ssp.impulsetrain.com/porterduff.html
# https://photoblogstop.com/photoshop/photoshop-blend-modes-explained

ctypedef (double, double, double) Color

cdef inline double _clip(double val) noexcept nogil:
    return max(min(val, 1), 0)

@cython.ufunc
cdef double clip(double val) noexcept nogil:
    return _clip(val)

cdef double eps = np.finfo(np.float64).eps

cdef inline double _safe_divide(double a, double b) noexcept nogil:
    return a / (b + eps)

@cython.ufunc
cdef double safe_divide_ufunc(double a, double b) noexcept nogil:
        return _safe_divide(a, b)

def safe_divide(a, b, /, **kwargs):
    with np.errstate(divide='ignore', invalid='ignore'):
        return safe_divide_ufunc(a, b, **kwargs)

cdef inline double _clip_divide(double a, double b) noexcept nogil:
    return _clip(_safe_divide(a, b))

@cython.ufunc
cdef double clip_divide_ufunc(double a, double b) noexcept nogil:
    return _clip_divide(a, b)

def clip_divide(a, b, /, **kwargs):
    with np.errstate(divide='ignore', invalid='ignore'):
        return clip_divide_ufunc(a, b, **kwargs)

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

cdef inline double _blend(double Csp, double Cdp, double Asrc, double Adst, double Aboth, double B) noexcept nogil:
    return Csp * Asrc + Cdp * Adst + Aboth * B

cdef inline double _premul(double Cd, double Cs, double Ad, double As, double(*straight_func)(double, double) noexcept nogil) noexcept nogil:
    Cdp = _clip_divide(Cd, Ad)
    Csp = _clip_divide(Cs, As)
    B = straight_func(Cdp, Csp)
    Asrc = _comp(As, Ad)
    Adst = _comp(Ad, As)
    Aboth = As * Ad
    return _blend(Csp, Cdp, Asrc, Adst, Aboth, B)

cdef inline Color _premul_nonseperable(Color Cd, Color Cs, double Ad, double As, (Color)(*straight_func)(Color, Color) noexcept nogil) noexcept nogil:
    Cdp_r = _clip_divide(Cd[0], Ad)
    Cdp_g = _clip_divide(Cd[1], Ad)
    Cdp_b = _clip_divide(Cd[2], Ad)
    Csp_r = _clip_divide(Cs[0], As)
    Csp_g = _clip_divide(Cs[1], As)
    Csp_b = _clip_divide(Cs[2], As)
    B_r, B_g, B_b = straight_func((Cdp_r, Cdp_g, Cdp_b), (Csp_r, Csp_g, Csp_b))
    Asrc = _comp(As, Ad)
    Adst = _comp(Ad, As)
    Aboth = As * Ad
    B_r = _blend(Csp_r, Cdp_r, Asrc, Adst, Aboth, B_r)
    B_g = _blend(Csp_g, Cdp_g, Asrc, Adst, Aboth, B_g)
    B_b = _blend(Csp_b, Cdp_b, Asrc, Adst, Aboth, B_b)
    return B_r, B_g, B_b

def ensure_array(D, S):
    return np.full_like(S, D) if np.isscalar(D) else D

def nonseperable(func):
    def wrap(Cd, Cs, Ad, As):
        Cd = ensure_array(Cd, Cs)
        Ad = ensure_array(Ad, As)
        return np.dstack(func(Cd[:,:,0], Cd[:,:,1], Cd[:,:,2], Cs[:,:,0], Cs[:,:,1], Cs[:,:,2], Ad.reshape(Ad.shape[:-1]), As.reshape(As.shape[:-1])))
    return wrap

@cython.ufunc
cdef double multiply(double Cd, double Cs, double Ad, double As) nogil:
    return Cs * Cd + _comp2(Cd, Cs, Ad, As)

cdef inline double screen_straight(double Cd, double Cs) noexcept nogil:
    return Cs + Cd - Cs * Cd

@cython.ufunc
cdef double screen(double Cd, double Cs, double Ad, double As) nogil:
    return screen_straight(Cd, Cs)

def overlay(Cd, Cs, Ad, As):
    return hard_light(Cs, Cd, As, Ad)

@cython.ufunc
cdef double sai_linear_burn(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = Cdd + Cs - As
    H = _clip(H)
    return _lerp(Cs, H, Ad)

cdef inline double ts_linear_burn_straight(double Cd, double Cs) noexcept nogil:
    return _clip(Cd + Cs - 1)

@cython.ufunc
cdef double ts_linear_burn(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_burn_straight)

@cython.ufunc
cdef double sai_linear_dodge(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = Cdd + Cs
    H = _clip(H)
    return _lerp(Cs, H, Ad)

cdef inline double ts_linear_dodge_straight(double Cd, double Cs) noexcept nogil:
    return _clip(Cd + Cs)

@cython.ufunc
cdef double ts_linear_dodge(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_dodge_straight)

@cython.ufunc
cdef double sai_linear_light(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    Cs2 = Cs * 2
    LB = Cdd + Cs2 - As
    LB = _clip(LB)
    return _lerp(Cs, LB, Ad)

cdef inline double ts_linear_light_straight(double Cd, double Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return ts_linear_dodge_straight(Cd, Cs2 - 1)
    else:
        return ts_linear_burn_straight(Cd, Cs2)

@cython.ufunc
cdef double ts_linear_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_light_straight)

cdef inline double ts_color_burn_straight(double Cd, double Cs) noexcept nogil:
    return 1 - _clip_divide(1 - Cd, Cs)

@cython.ufunc
cdef double ts_color_burn(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_color_burn_straight)

@cython.ufunc
cdef double sai_color_burn(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    B = 1 - _clip_divide(1 - Cdd, 1 - As + Cs)
    return _lerp(Cs, B, Ad)

cdef inline double ts_color_dodge_straight(double Cd, double Cs) noexcept nogil:
    return _clip_divide(Cd, 1 - Cs)

@cython.ufunc
cdef double ts_color_dodge(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_color_dodge_straight)

@cython.ufunc
cdef double sai_color_dodge(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = _clip_divide(Cdd, 1 - Cs)
    return _lerp(Cs, H, Ad)

cdef inline double ts_vivid_light_straight(double Cd, double Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return ts_color_dodge_straight(Cd, Cs2 - 1)
    else:
        return ts_color_burn_straight(Cd, Cs2)

@cython.ufunc
cdef double ts_vivid_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_vivid_light_straight)

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

cdef inline double soft_light_straight(double Cd, double Cs) noexcept nogil:
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
    return _premul(Cd, Cs, Ad, As, soft_light_straight)

cdef inline double hard_light_straight(double Cd, double Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return screen_straight(Cd, Cs2 - 1)
    else:
        return Cd * Cs2

@cython.ufunc
cdef double hard_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, hard_light_straight)

cdef inline double pin_light_straight(double Cd, double Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return max(Cs2 - 1, Cd)
    else:
        return min(Cs2, Cd)

@cython.ufunc
cdef double pin_light(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, pin_light_straight)

@cython.ufunc
cdef double sai_hard_mix(double Cd, double Cs, double Ad, double As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    Csd = _clip_divide(Cs, As)
    H = _clip_divide(Cdd - As + As * Csd, 1 - As)
    return _lerp(Cs, H, Ad)

cdef inline double ts_hard_mix_straight(double Cd, double Cs) noexcept nogil:
    if (Cd + Cs) > 1:
        return 1
    else:
        return 0

@cython.ufunc
cdef double ts_hard_mix(double Cd, double Cs, double Ad, double As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_hard_mix_straight)

@cython.ufunc
cdef double darken(double Cd, double Cs, double Ad, double As) nogil:
    return min(Cs * Ad, Cd * As) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef double lighten(double Cd, double Cs, double Ad, double As) nogil:
    return max(Cs * Ad, Cd * As) + _comp2(Cd, Cs, Ad, As)

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

cdef inline double _lum(Color C) noexcept nogil:
    return 0.3 * C[0] + 0.59 * C[1] + 0.11 * C[2]

cdef inline Color _clip_color(Color C) noexcept nogil:
    cdef double L = _lum(C)
    cdef double n = min(C[0], C[1], C[2])
    cdef double x = max(C[0], C[1], C[2])
    cdef double C_rL = C[0] - L
    cdef double C_gL = C[1] - L
    cdef double C_bL = C[2] - L
    if n < 0.0:
        Ln = L - n
        C[0] = L + _safe_divide(C_rL * L, Ln)
        C[1] = L + _safe_divide(C_gL * L, Ln)
        C[2] = L + _safe_divide(C_bL * L, Ln)
    if x > 1.0:
        L1 = 1.0 - L
        xL = x - L
        C[0] = L + _safe_divide(C_rL * L1, xL)
        C[1] = L + _safe_divide(C_gL * L1, xL)
        C[2] = L + _safe_divide(C_bL * L1, xL)
    return C

cdef inline Color _set_lum(Color C, double L) noexcept nogil:
    d = L - _lum(C)
    C[0] = C[0] + d
    C[1] = C[1] + d
    C[2] = C[2] + d
    return _clip_color(C)

cdef inline double _sat(Color C) noexcept nogil:
    return max(C[0], C[1], C[2]) - min(C[0], C[1], C[2])

cdef inline Color _set_sat(Color C, double S) noexcept nogil:
    cdef double amax = max(C[0], C[1], C[2])
    cdef double amin = min(C[0], C[1], C[2])
    cdef double amid
    cdef double* cmax
    cdef double* cmid
    cdef double* cmin
    if amax == C[0]:
        cmax = &C[0]
        if amin == C[1]:
            cmin = &C[1]
            cmid = &C[2]
        else:
            cmin = &C[2]
            cmid = &C[1]
    elif amax == C[1]:
        cmax = &C[1]
        if amin == C[0]:
            cmin = &C[0]
            cmid = &C[2]
        else:
            cmin = &C[2]
            cmid = &C[0]
    else:
        cmax = &C[2]
        if amin == C[0]:
            cmin = &C[0]
            cmid = &C[1]
        else:
            cmin = &C[1]
            cmid = &C[0]
    amid = cmid[0]
    if amax > amin:
        cmid[0] = _safe_divide((amid - amin) * S, amax - amin)
        cmax[0] = S
    else:
        cmid[0] = 0
        cmax[0] = 0
    cmin[0] = 0
    return (C[0], C[1], C[2])

cdef inline Color hue_straight(Color Cd, Color Cs) noexcept nogil:
    if Cs[0] == Cs[1] and Cs[0] == Cs[2]:
        Cs[0] = Cs[0] + eps
    return _set_lum(_set_sat(Cs, _sat(Cd)), _lum(Cd))

@cython.ufunc
cdef Color hue_nonseperable(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, hue_straight)

hue = nonseperable(hue_nonseperable)

cdef inline Color saturation_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(_set_sat(Cd, _sat(Cs)), _lum(Cd))

@cython.ufunc
cdef Color saturation_nonseperable(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, saturation_straight)

saturation = nonseperable(saturation_nonseperable)

cdef inline Color color_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(Cs, _lum(Cd))

@cython.ufunc
cdef Color color_nonseperable(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, color_straight)

color = nonseperable(color_nonseperable)

cdef inline Color luminosity_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(Cd, _lum(Cs))

@cython.ufunc
cdef Color luminosity_nonseperable(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, luminosity_straight)

luminosity = nonseperable(luminosity_nonseperable)

cdef inline Color darker_color_straight(Color Cd, Color Cs) noexcept nogil:
    if _lum(Cs) < _lum(Cd):
        return Cs
    else:
        return Cd

@cython.ufunc
cdef Color darker_color_nonseperable(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, darker_color_straight)

darker_color = nonseperable(darker_color_nonseperable)

cdef inline Color lighter_color_straight(Color Cd, Color Cs) noexcept nogil:
    if _lum(Cs) > _lum(Cd):
        return Cs
    else:
        return Cd

@cython.ufunc
cdef Color lighter_color_nonseperable(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, lighter_color_straight)

lighter_color = nonseperable(lighter_color_nonseperable)

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
