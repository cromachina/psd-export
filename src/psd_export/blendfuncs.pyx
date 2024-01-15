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

cdef inline double _blend(double Csp, double Cdp, double Asrc, double Adst, double Aboth, double B) noexcept nogil:
    return Csp * Asrc + Cdp * Adst + Aboth * B

cdef inline double _premul(double Cd, double Cs, double Ad, double As, double(*non_premul_func)(double, double) noexcept nogil) noexcept nogil:
    Cdp = _clip_divide(Cd, Ad)
    Csp = _clip_divide(Cs, As)
    B = non_premul_func(Cdp, Csp)
    Asrc = _comp(As, Ad)
    Adst = _comp(Ad, As)
    Aboth = As * Ad
    return _blend(Csp, Cdp, Asrc, Adst, Aboth, B)

cdef inline (double, double, double) _premul_nonsep(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As, (double, double, double)(*non_premul_func)(double, double, double, double, double, double) noexcept nogil) noexcept nogil:
    Cdp_r = _clip_divide(Cd_r, Ad)
    Cdp_g = _clip_divide(Cd_g, Ad)
    Cdp_b = _clip_divide(Cd_b, Ad)
    Csp_r = _clip_divide(Cs_r, As)
    Csp_g = _clip_divide(Cs_g, As)
    Csp_b = _clip_divide(Cs_b, As)
    B_r, B_g, B_b = non_premul_func(Cdp_r, Cdp_g, Cdp_b, Csp_r, Csp_g, Csp_b)
    Asrc = _comp(As, Ad)
    Adst = _comp(Ad, As)
    Aboth = As * Ad
    B_r = _blend(Csp_r, Cdp_r, Asrc, Adst, Aboth, B_r)
    B_g = _blend(Csp_g, Cdp_g, Asrc, Adst, Aboth, B_g)
    B_b = _blend(Csp_b, Cdp_b, Asrc, Adst, Aboth, B_b)
    return B_r, B_g, B_b

def ensure_array(D, S):
    return np.full_like(S, D) if np.isscalar(D) else D

def nonsep(func):
    def wrap(Cd, Cs, Ad, As):
        Cd = ensure_array(Cd, Cs)
        Ad = ensure_array(Ad, As)
        return np.dstack(func(Cd[:,:,0], Cd[:,:,1], Cd[:,:,2], Cs[:,:,0], Cs[:,:,1], Cs[:,:,2], Ad.reshape(Ad.shape[:-1]), As.reshape(As.shape[:-1])))
    return wrap

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

cdef inline double _lum(double C_r, double C_g, double C_b) noexcept nogil:
    return 0.3 * C_r + 0.59 * C_g + 0.11 * C_b

cdef inline (double, double, double) _clip_color(double C_r, double C_g, double C_b) noexcept nogil:
    cdef double L = _lum(C_r, C_g, C_b)
    cdef double n = min(C_r, C_g, C_b)
    cdef double x = max(C_r, C_g, C_b)
    cdef double C_rL = C_r - L
    cdef double C_gL = C_g - L
    cdef double C_bL = C_b - L
    if n < 0.0:
        Ln = L - n
        C_r = L + _safe_divide(C_rL * L, Ln)
        C_g = L + _safe_divide(C_gL * L, Ln)
        C_b = L + _safe_divide(C_bL * L, Ln)
    if x > 1.0:
        L1 = 1.0 - L
        xL = x - L
        C_r = L + _safe_divide(C_rL * L1, xL)
        C_g = L + _safe_divide(C_gL * L1, xL)
        C_b = L + _safe_divide(C_bL * L1, xL)
    return C_r, C_g, C_b

cdef inline (double, double, double) _set_lum(double C_r, double C_g, double C_b, double L) noexcept nogil:
    d = L - _lum(C_r, C_g, C_b)
    C_r = C_r + d
    C_b = C_b + d
    C_g = C_g + d
    return _clip_color(C_r, C_g, C_b)

cdef inline double _sat(double C_r, double C_g, double C_b) noexcept nogil:
    return max(C_r, C_g, C_b) - min(C_r, C_g, C_b)

cdef inline (double, double, double) _set_sat(double C_r, double C_g, double C_b, double S) noexcept nogil:
    cdef double amax = max(C_r, C_g, C_b)
    cdef double amin = min(C_r, C_g, C_b)
    cdef double amid
    cdef double* cmax
    cdef double* cmid
    cdef double* cmin
    if amax == C_r:
        cmax = &C_r
        if amin == C_g:
            cmin = &C_g
            cmid = &C_b
        else:
            cmin = &C_b
            cmid = &C_g
    elif amax == C_g:
        cmax = &C_g
        if amin == C_r:
            cmin = &C_r
            cmid = &C_b
        else:
            cmin = &C_b
            cmid = &C_r
    else:
        cmax = &C_b
        if amin == C_r:
            cmin = &C_r
            cmid = &C_g
        else:
            cmin = &C_g
            cmid = &C_r
    amid = cmid[0]
    if amax > amin:
        cmid[0] = _safe_divide((amid - amin) * S, amax - amin)
        cmax[0] = S
    else:
        cmid[0] = 0
        cmax[0] = 0
    cmin[0] = 0
    return (C_r, C_g, C_b)

cdef inline (double, double, double) _hue(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b) noexcept nogil:
    cdef double S = _sat(Cd_r, Cd_g, Cd_b)
    cdef double L = _lum(Cd_r, Cd_g, Cd_b)
    t_r, t_g, t_b = _set_sat(Cs_r, Cs_g, Cs_b, S)
    return _set_lum(t_r, t_g, t_b, L)

# TODO: Bug: 0 sat sources produce 0 sat results, but in SAI the hue result is red.
# However, technically it seems like this is correct according to PDF documentation.
@cython.ufunc
cdef (double, double, double) hue_nonsep(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonsep(Cd_r, Cd_g, Cd_b, Cs_r, Cs_g, Cs_b, Ad, As, _hue)

hue = nonsep(hue_nonsep)

cdef inline (double, double, double) _saturation(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b) noexcept nogil:
    cdef double S = _sat(Cs_r, Cs_g, Cs_b)
    cdef double L = _lum(Cd_r, Cd_g, Cd_b)
    t_r, t_g, t_b = _set_sat(Cd_r, Cd_g, Cd_b, S)
    return _set_lum(t_r, t_g, t_b, L)

@cython.ufunc
cdef (double, double, double) saturation_nonsep(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonsep(Cd_r, Cd_g, Cd_b, Cs_r, Cs_g, Cs_b, Ad, As, _saturation)

saturation = nonsep(saturation_nonsep)

cdef inline (double, double, double) _color(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b) noexcept nogil:
    cdef double L = _lum(Cd_r, Cd_g, Cd_b)
    return _set_lum(Cs_r, Cs_g, Cs_b, L)

@cython.ufunc
cdef (double, double, double) color_nonsep(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonsep(Cd_r, Cd_g, Cd_b, Cs_r, Cs_g, Cs_b, Ad, As, _color)

color = nonsep(color_nonsep)

cdef inline (double, double, double) _luminosity(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b) noexcept nogil:
    cdef double L = _lum(Cs_r, Cs_g, Cs_b)
    return _set_lum(Cd_r, Cd_g, Cd_b, L)

@cython.ufunc
cdef (double, double, double) luminosity_nonsep(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonsep(Cd_r, Cd_g, Cd_b, Cs_r, Cs_g, Cs_b, Ad, As, _luminosity)

luminosity = nonsep(luminosity_nonsep)

cdef inline (double, double, double) _darker_color(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b) noexcept nogil:
    cdef double Cd_L = _lum(Cd_r, Cd_g, Cd_b)
    cdef double Cs_L = _lum(Cs_r, Cs_g, Cs_b)
    if Cs_L < Cd_L:
        return (Cs_r, Cs_g, Cs_b)
    else:
        return (Cd_r, Cd_g, Cd_b)

@cython.ufunc
cdef (double, double, double) darker_color_nonsep(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonsep(Cd_r, Cd_g, Cd_b, Cs_r, Cs_g, Cs_b, Ad, As, _darker_color)

darker_color = nonsep(darker_color_nonsep)

cdef inline (double, double, double) _lighter_color(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b) noexcept nogil:
    cdef double Cd_L = _lum(Cd_r, Cd_g, Cd_b)
    cdef double Cs_L = _lum(Cs_r, Cs_g, Cs_b)
    if Cs_L > Cd_L:
        return (Cs_r, Cs_g, Cs_b)
    else:
        return (Cd_r, Cd_g, Cd_b)

@cython.ufunc
cdef (double, double, double) lighter_color_nonsep(double Cd_r, double Cd_g, double Cd_b, double Cs_r, double Cs_g, double Cs_b, double Ad, double As) nogil:
    return _premul_nonsep(Cd_r, Cd_g, Cd_b, Cs_r, Cs_g, Cs_b, Ad, As, _lighter_color)

lighter_color = nonsep(lighter_color_nonsep)

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
