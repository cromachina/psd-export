#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np
from psd_tools.constants import BlendMode

# https://dev.w3.org/SVG/modules/compositing/master/
# http://ssp.impulsetrain.com/porterduff.html
# https://photoblogstop.com/photoshop/photoshop-blend-modes-explained

ctypedef (float, float, float) Color

cdef inline float _clip(float val) noexcept nogil:
    return max(min(val, 1), 0)

@cython.ufunc
cdef float clip(float val) noexcept nogil:
    return _clip(val)

cdef float eps = np.finfo(np.float64).eps

cdef inline float _safe_divide(float a, float b) noexcept nogil:
    if b == 0:
        return 1.0
    else:
        return a / b

@cython.ufunc
cdef float safe_divide_ufunc(float a, float b) noexcept nogil:
        return _safe_divide(a, b)

def safe_divide(a, b, /, **kwargs):
    with np.errstate(divide='ignore', invalid='ignore'):
        return safe_divide_ufunc(a, b, **kwargs)

cdef inline float _clip_divide(float a, float b) noexcept nogil:
    return _clip(_safe_divide(a, b))

@cython.ufunc
cdef float clip_divide_ufunc(float a, float b) noexcept nogil:
    return _clip_divide(a, b)

def clip_divide(a, b, /, **kwargs):
    with np.errstate(divide='ignore', invalid='ignore'):
        return clip_divide_ufunc(a, b, **kwargs)

cdef inline float _comp(float C, float A) noexcept nogil:
    return C * (1 - A)

@cython.ufunc
cdef float comp(float a, float b) noexcept nogil:
    return _comp(a, b)

cdef inline float _comp2(float Cd, float Cs, float Ad, float As) noexcept nogil:
    return _comp(Cd, As) + _comp(Cs, Ad)

@cython.ufunc
cdef float comp2(float Cd, float Cs, float Ad, float As)  noexcept nogil:
    return _comp2(Cd, Cs, Ad, As)

cdef inline float _lerp(float a, float b, float t) noexcept nogil:
    return (b - a) * t + a

@cython.ufunc
cdef float lerp(float a, float b, float t) noexcept nogil:
    return _lerp(a, b, t)

@cython.ufunc
cdef float normal(float Cd, float Cs, float Ad, float As) nogil:
    return Cs + _comp(Cd, As)

@cython.ufunc
cdef float normal_alpha(float Ad, float As) nogil:
    return Ad + As - Ad * As

cdef inline float _blend(float Csp, float Cdp, float Asrc, float Adst, float Aboth, float B) noexcept nogil:
    return Csp * Asrc + Cdp * Adst + Aboth * B

cdef inline float _premul(float Cd, float Cs, float Ad, float As, float(*straight_func)(float, float) noexcept nogil) noexcept nogil:
    Cdp = _clip_divide(Cd, Ad)
    Csp = _clip_divide(Cs, As)
    B = straight_func(Cdp, Csp)
    Asrc = _comp(As, Ad)
    Adst = _comp(Ad, As)
    Aboth = As * Ad
    return _blend(Csp, Cdp, Asrc, Adst, Aboth, B)

cdef inline Color _premul_nonseperable(Color Cd, Color Cs, float Ad, float As, (Color)(*straight_func)(Color, Color) noexcept nogil) noexcept nogil:
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
cdef float multiply(float Cd, float Cs, float Ad, float As) nogil:
    return Cs * Cd + _comp2(Cd, Cs, Ad, As)

cdef inline float screen_straight(float Cd, float Cs) noexcept nogil:
    return Cs + Cd - Cs * Cd

@cython.ufunc
cdef float screen(float Cd, float Cs, float Ad, float As) nogil:
    return screen_straight(Cd, Cs)

def overlay(Cd, Cs, Ad, As):
    return hard_light(Cs, Cd, As, Ad)

@cython.ufunc
cdef float sai_linear_burn(float Cd, float Cs, float Ad, float As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = Cdd + Cs - As
    H = _clip(H)
    return _lerp(Cs, H, Ad)

cdef inline float ts_linear_burn_straight(float Cd, float Cs) noexcept nogil:
    return _clip(Cd + Cs - 1)

@cython.ufunc
cdef float ts_linear_burn(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_burn_straight)

@cython.ufunc
cdef float sai_linear_dodge(float Cd, float Cs, float Ad, float As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = Cdd + Cs
    H = _clip(H)
    return _lerp(Cs, H, Ad)

cdef inline float ts_linear_dodge_straight(float Cd, float Cs) noexcept nogil:
    return _clip(Cd + Cs)

@cython.ufunc
cdef float ts_linear_dodge(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_dodge_straight)

@cython.ufunc
cdef float sai_linear_light(float Cd, float Cs, float Ad, float As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    Cs2 = Cs * 2
    LB = Cdd + Cs2 - As
    LB = _clip(LB)
    return _lerp(Cs, LB, Ad)

cdef inline float ts_linear_light_straight(float Cd, float Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return ts_linear_dodge_straight(Cd, Cs2 - 1)
    else:
        return ts_linear_burn_straight(Cd, Cs2)

@cython.ufunc
cdef float ts_linear_light(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_light_straight)

cdef inline float ts_color_burn_straight(float Cd, float Cs) noexcept nogil:
    return 1 - _clip_divide(1 - Cd, Cs)

@cython.ufunc
cdef float ts_color_burn(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_color_burn_straight)

@cython.ufunc
cdef float sai_color_burn(float Cd, float Cs, float Ad, float As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    B = 1 - _clip_divide(1 - Cdd, 1 - As + Cs)
    return _lerp(Cs, B, Ad)

cdef inline float ts_color_dodge_straight(float Cd, float Cs) noexcept nogil:
    return _clip_divide(Cd, 1 - Cs)

@cython.ufunc
cdef float ts_color_dodge(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_color_dodge_straight)

@cython.ufunc
cdef float sai_color_dodge(float Cd, float Cs, float Ad, float As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    H = _clip_divide(Cdd, 1 - Cs)
    return _lerp(Cs, H, Ad)

cdef inline float ts_vivid_light_straight(float Cd, float Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return ts_color_dodge_straight(Cd, Cs2 - 1)
    else:
        return ts_color_burn_straight(Cd, Cs2)

@cython.ufunc
cdef float ts_vivid_light(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_vivid_light_straight)

@cython.ufunc
cdef float sai_vivid_light(float Cd, float Cs, float Ad, float As) nogil:
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
    float sqrt(float x) noexcept nogil

cdef inline float soft_light_straight(float Cd, float Cs) noexcept nogil:
    cdef float D, B
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
cdef float soft_light(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, soft_light_straight)

cdef inline float hard_light_straight(float Cd, float Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return screen_straight(Cd, Cs2 - 1)
    else:
        return Cd * Cs2

@cython.ufunc
cdef float hard_light(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, hard_light_straight)

cdef inline float pin_light_straight(float Cd, float Cs) noexcept nogil:
    Cs2 = Cs * 2
    if Cs > 0.5:
        return max(Cs2 - 1, Cd)
    else:
        return min(Cs2, Cd)

@cython.ufunc
cdef float pin_light(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, pin_light_straight)

@cython.ufunc
cdef float sai_hard_mix(float Cd, float Cs, float Ad, float As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    Csd = _clip_divide(Cs, As)
    H = _clip_divide(Cdd - As + As * Csd, 1 - As)
    return _lerp(Cs, H, Ad)

cdef inline float ts_hard_mix_straight(float Cd, float Cs) noexcept nogil:
    if (Cd + Cs) > 1:
        return 1
    else:
        return 0

@cython.ufunc
cdef float ts_hard_mix(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, ts_hard_mix_straight)

@cython.ufunc
cdef float darken(float Cd, float Cs, float Ad, float As) nogil:
    return min(Cs * Ad, Cd * As) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef float lighten(float Cd, float Cs, float Ad, float As) nogil:
    return max(Cs * Ad, Cd * As) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef float ts_difference(float Cd, float Cs, float Ad, float As) nogil:
    return Cs + Cd - 2 * min(Cd * As, Cs * Ad)

@cython.ufunc
cdef float sai_difference(float Cd, float Cs, float Ad, float As) nogil:
    Cdd = _clip_divide(Cd, Ad)
    D = abs(Cdd - Cs)
    return _lerp(Cs, D, Ad)

@cython.ufunc
cdef float exclusion(float Cd, float Cs, float Ad, float As) nogil:
    return (Cs * Ad + Cd * As - 2 * Cs * Cd) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef float subtract(float Cd, float Cs, float Ad, float As) nogil:
    return max(0, Cd * As - Cs * Ad) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef float divide(float Cd, float Cs, float Ad, float As) nogil:
    return _premul(Cd, Cs, Ad, As, _clip_divide)

cdef inline float _lum(Color C) noexcept nogil:
    return 0.3 * C[0] + 0.59 * C[1] + 0.11 * C[2]

cdef inline Color _clip_color(Color C) noexcept nogil:
    cdef float L = _lum(C)
    cdef float n = min(C[0], C[1], C[2])
    cdef float x = max(C[0], C[1], C[2])
    cdef float C_rL = C[0] - L
    cdef float C_gL = C[1] - L
    cdef float C_bL = C[2] - L
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

cdef inline Color _set_lum(Color C, float L) noexcept nogil:
    d = L - _lum(C)
    C[0] = C[0] + d
    C[1] = C[1] + d
    C[2] = C[2] + d
    return _clip_color(C)

cdef inline float _sat(Color C) noexcept nogil:
    return max(C[0], C[1], C[2]) - min(C[0], C[1], C[2])

cdef inline Color _set_sat(Color C, float S) noexcept nogil:
    cdef float amax = max(C[0], C[1], C[2])
    cdef float amin = min(C[0], C[1], C[2])
    cdef float amid
    cdef float* cmax
    cdef float* cmid
    cdef float* cmin
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
cdef Color hue_nonseperable(float Cd_r, float Cd_g, float Cd_b, float Cs_r, float Cs_g, float Cs_b, float Ad, float As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, hue_straight)

hue = nonseperable(hue_nonseperable)

cdef inline Color saturation_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(_set_sat(Cd, _sat(Cs)), _lum(Cd))

@cython.ufunc
cdef Color saturation_nonseperable(float Cd_r, float Cd_g, float Cd_b, float Cs_r, float Cs_g, float Cs_b, float Ad, float As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, saturation_straight)

saturation = nonseperable(saturation_nonseperable)

cdef inline Color color_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(Cs, _lum(Cd))

@cython.ufunc
cdef Color color_nonseperable(float Cd_r, float Cd_g, float Cd_b, float Cs_r, float Cs_g, float Cs_b, float Ad, float As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, color_straight)

color = nonseperable(color_nonseperable)

cdef inline Color luminosity_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(Cd, _lum(Cs))

@cython.ufunc
cdef Color luminosity_nonseperable(float Cd_r, float Cd_g, float Cd_b, float Cs_r, float Cs_g, float Cs_b, float Ad, float As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, luminosity_straight)

luminosity = nonseperable(luminosity_nonseperable)

cdef inline Color darker_color_straight(Color Cd, Color Cs) noexcept nogil:
    if _lum(Cs) < _lum(Cd):
        return Cs
    else:
        return Cd

@cython.ufunc
cdef Color darker_color_nonseperable(float Cd_r, float Cd_g, float Cd_b, float Cs_r, float Cs_g, float Cs_b, float Ad, float As) nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, darker_color_straight)

darker_color = nonseperable(darker_color_nonseperable)

cdef inline Color lighter_color_straight(Color Cd, Color Cs) noexcept nogil:
    if _lum(Cs) > _lum(Cd):
        return Cs
    else:
        return Cd

@cython.ufunc
cdef Color lighter_color_nonseperable(float Cd_r, float Cd_g, float Cd_b, float Cs_r, float Cs_g, float Cs_b, float Ad, float As) nogil:
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
