#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
cimport cython
import numpy as np
from psd_tools.constants import BlendMode
from libc.stdint cimport *
from cpython cimport array

# https://dev.w3.org/SVG/modules/compositing/master/
# http://ssp.impulsetrain.com/porterduff.html
# https://photoblogstop.com/photoshop/photoshop-blend-modes-explained

dtype = np.int16
ctypedef (short, short, short) Color
cdef short rangemax = 0x3fff
cdef short rangehalf = (rangemax // 2)
cdef short rangequarter = (rangemax // 4)
cdef short range2 = (rangemax * 2)
cdef short range4 = (rangemax * 4)
cdef short range12 = (rangemax * 12)
cdef short range16 = (rangemax * 16)
tile_size = (512, 512)

def get_max():
    return rangemax

@cython.ufunc
cdef inline short _from_bytes(short val):
    return (val << 6) | (val >> 2)

def from_bytes(val):
    val = dtype(val)
    return _from_bytes(val)

def to_bytes(data):
    return (data >> 6).astype(np.uint8)

def from_floats(val):
    return dtype(val * rangemax)

def parse_array(data, depth, lut: np.ndarray | None = None):
    if depth == 8:
        parsed = np.frombuffer(data, ">u1")
        if lut is not None:
            parsed = lut[parsed]
        return from_bytes(parsed)
    elif depth == 16:
        return np.frombuffer(data, ">u2").astype(dtype) // 4
    elif depth == 32:
        return (np.frombuffer(data, ">f4") * rangemax).astype(dtype)
    elif depth == 1:
        return from_bytes(np.unpackbits(np.frombuffer(data, np.uint8)))
    else:
        raise ValueError("Unsupported depth: %g" % depth)

@cython.ufunc
cdef threshold(short val):
    return 0 if val == 0 else rangemax

cdef inline short _clip(short val) noexcept nogil:
    return max(min(val, rangemax), 0)

@cython.ufunc
cdef short clip(short val) noexcept nogil:
    return _clip(val)

cdef inline short _safe_divide(short a, short b) noexcept nogil:
    if b == 0:
        return rangemax
    else:
        return <short> (rangemax * (<float> a / rangemax) / (<float> b / rangemax))

cdef inline short _clip_divide(short a, short b) noexcept nogil:
    return _clip(_safe_divide(a, b))

@cython.ufunc
cdef short clip_divide_ufunc(short a, short b) noexcept nogil:
    return _clip_divide(a, b)

def clip_divide(a, b, /, **kwargs):
    with np.errstate(divide='ignore', invalid='ignore'):
        return clip_divide_ufunc(a, b, **kwargs)

cdef inline short _mul(short a, short b) noexcept nogil:
    return (a * b) >> 14

@cython.ufunc
cdef short mul(short C, short A) noexcept nogil:
    return _mul(C, A)

cdef inline short _comp(short C, short A) noexcept nogil:
    return _mul(C, rangemax - A)

@cython.ufunc
cdef short comp(short a, short b) noexcept nogil:
    return _comp(a, b)

cdef inline short _comp2(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _comp(Cd, As) + _comp(Cs, Ad)

@cython.ufunc
cdef short comp2(short Cd, short Cs, short Ad, short As)  noexcept nogil:
    return _comp2(Cd, Cs, Ad, As)

cdef inline short _lerp(short a, short b, short t) noexcept nogil:
    return _mul(b - a, t) + a

@cython.ufunc
cdef short lerp(short a, short b, short t) noexcept nogil:
    return _lerp(a, b, t)

@cython.ufunc
cdef short normal(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return Cs + _comp(Cd, As)

@cython.ufunc
cdef short normal_alpha(short Ad, short As) noexcept nogil:
    return _clip(Ad + As - _mul(Ad, As))

cdef inline short _blend(short Csp, short Cdp, short Asrc, short Adst, short Aboth, short B) noexcept nogil:
    return _mul(Csp, Asrc) + _mul(Cdp, Adst) + _mul(Aboth, B)

cdef inline short _premul(short Cd, short Cs, short Ad, short As, short(*straight_func)(short, short) noexcept nogil) noexcept nogil:
    cdef short Cdp = _clip_divide(Cd, Ad)
    cdef short Csp = _clip_divide(Cs, As)
    cdef short B = straight_func(Cdp, Csp)
    cdef short Asrc = _comp(As, Ad)
    cdef short Adst = _comp(Ad, As)
    cdef short Aboth = _mul(As, Ad)
    return _blend(Csp, Cdp, Asrc, Adst, Aboth, B)

cdef inline Color _premul_nonseperable(Color Cd, Color Cs, short Ad, short As, (Color)(*straight_func)(Color, Color) noexcept nogil) noexcept nogil:
    cdef short Cdp_r = _clip_divide(Cd[0], Ad)
    cdef short Cdp_g = _clip_divide(Cd[1], Ad)
    cdef short Cdp_b = _clip_divide(Cd[2], Ad)
    cdef short Csp_r = _clip_divide(Cs[0], As)
    cdef short Csp_g = _clip_divide(Cs[1], As)
    cdef short Csp_b = _clip_divide(Cs[2], As)
    cdef short B_r, B_g, B_b
    B_r, B_g, B_b = straight_func((Cdp_r, Cdp_g, Cdp_b), (Csp_r, Csp_g, Csp_b))
    cdef short Asrc = _comp(As, Ad)
    cdef short Adst = _comp(Ad, As)
    cdef short Aboth = _mul(As, Ad)
    B_r = _blend(Csp_r, Cdp_r, Asrc, Adst, Aboth, B_r)
    B_g = _blend(Csp_g, Cdp_g, Asrc, Adst, Aboth, B_g)
    B_b = _blend(Csp_b, Cdp_b, Asrc, Adst, Aboth, B_b)
    return B_r, B_g, B_b

def ensure_array(D, S):
    return np.full_like(S, D) if np.isscalar(D) else D

def nonseperable(func):
    @cython.wraparound(True)
    def wrap(Cd, Cs, Ad, As):
        Cd = ensure_array(Cd, Cs)
        Ad = ensure_array(Ad, As)
        return np.dstack(func(Cd[:,:,0], Cd[:,:,1], Cd[:,:,2], Cs[:,:,0], Cs[:,:,1], Cs[:,:,2], Ad.reshape(Ad.shape[:-1]), As.reshape(As.shape[:-1])))
    return wrap

@cython.ufunc
cdef short multiply(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _mul(Cs, Cd) + _comp2(Cd, Cs, Ad, As)

cdef inline short screen_straight(short Cd, short Cs) noexcept nogil:
    return Cs + Cd - _mul(Cs, Cd)

@cython.ufunc
cdef short screen(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return screen_straight(Cd, Cs)

def overlay(Cd, Cs, Ad, As):
    return hard_light(Cs, Cd, As, Ad)

@cython.ufunc
cdef short sai_linear_burn(short Cd, short Cs, short Ad, short As) noexcept nogil:
    cdef short Cdd = _clip_divide(Cd, Ad)
    cdef short H = Cdd + Cs - As
    H = _clip(H)
    return _lerp(Cs, H, Ad)

cdef inline short ts_linear_burn_straight(short Cd, short Cs) noexcept nogil:
    return _clip(Cd + Cs - rangemax)

@cython.ufunc
cdef short ts_linear_burn(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_burn_straight)

@cython.ufunc
cdef short sai_linear_dodge(short Cd, short Cs, short Ad, short As) noexcept nogil:
    cdef short Cdd = _clip_divide(Cd, Ad)
    cdef short H = Cdd + Cs
    H = _clip(H)
    return _lerp(Cs, H, Ad)

cdef inline short ts_linear_dodge_straight(short Cd, short Cs) noexcept nogil:
    return _clip(Cd + Cs)

@cython.ufunc
cdef short ts_linear_dodge(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_dodge_straight)

@cython.ufunc
cdef short sai_linear_light(short Cd, short Cs, short Ad, short As) noexcept nogil:
    cdef short Cdd = _clip_divide(Cd, Ad)
    cdef short Cs2 = _mul(Cs, range2)
    cdef short LB = Cdd + Cs2 - As
    LB = _clip(LB)
    return _lerp(Cs, LB, Ad)

cdef inline short ts_linear_light_straight(short Cd, short Cs) noexcept nogil:
    cdef short Cs2 = _mul(Cs, range2)
    if Cs > rangehalf:
        return ts_linear_dodge_straight(Cd, Cs2 - rangemax)
    else:
        return ts_linear_burn_straight(Cd, Cs2)

@cython.ufunc
cdef short ts_linear_light(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, ts_linear_light_straight)

cdef inline short ts_color_burn_straight(short Cd, short Cs) noexcept nogil:
    return rangemax - _clip_divide(rangemax - Cd, Cs)

@cython.ufunc
cdef short ts_color_burn(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, ts_color_burn_straight)

@cython.ufunc
cdef short sai_color_burn(short Cd, short Cs, short Ad, short As) noexcept nogil:
    cdef short Cdd = _clip_divide(Cd, Ad)
    cdef short B = rangemax - _clip_divide(rangemax - Cdd, rangemax - As + Cs)
    return _lerp(Cs, B, Ad)

cdef inline short ts_color_dodge_straight(short Cd, short Cs) noexcept nogil:
    return _clip_divide(Cd, rangemax - Cs)

@cython.ufunc
cdef short ts_color_dodge(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, ts_color_dodge_straight)

@cython.ufunc
cdef short sai_color_dodge(short Cd, short Cs, short Ad, short As) noexcept nogil:
    cdef short Cdd = _clip_divide(Cd, Ad)
    cdef short H = _clip_divide(Cdd, rangemax - Cs)
    return _lerp(Cs, H, Ad)

cdef inline short ts_vivid_light_straight(short Cd, short Cs) noexcept nogil:
    cdef short Cs2 = _mul(Cs, range2)
    if Cs > rangehalf:
        return ts_color_dodge_straight(Cd, Cs2 - rangemax)
    else:
        return ts_color_burn_straight(Cd, Cs2)

@cython.ufunc
cdef short ts_vivid_light(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, ts_vivid_light_straight)

@cython.ufunc
cdef short sai_vivid_light(short Cd, short Cs, short Ad, short As) noexcept nogil:
    cdef short Cdd = _clip_divide(Cd, Ad)
    cdef short Csd = _clip_divide(Cs, As)
    cdef short Cs2 = As - _mul(Cs, range2)
    cdef short CB = rangemax - _clip_divide(rangemax - Cdd, rangemax - Cs2)
    cdef short CD = _clip_divide(Cdd, rangemax + Cs2)
    if Csd > rangehalf:
        CB = CD
    if Cs == rangemax:
        CB = rangemax
    return _lerp(Cs, CB, Ad)

cdef short[::1] sqrt_lut = (np.sqrt(np.arange(0, rangemax + 1, dtype=np.float32) / rangemax) * rangemax).astype(np.int16)

cdef inline short soft_light_straight(short Cd, short Cs) noexcept nogil:
    cdef short D, B
    if Cs <= rangequarter:
        D = _mul((_mul((_mul(range16, Cd) - range12), Cd) + range4), Cd)
    else:
        D = sqrt_lut[Cd]
    if Cs <= rangehalf:
        B = Cd - _mul(_mul((rangemax - _mul(range2, Cs)), Cd), (rangemax - Cd))
    else:
        B = Cd + _mul((_mul(range2, Cs) - rangemax), (D - Cd))
    return B

@cython.ufunc
cdef short soft_light(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, soft_light_straight)

cdef inline short hard_light_straight(short Cd, short Cs) noexcept nogil:
    cdef short Cs2 = _mul(Cs, range2)
    if Cs > rangehalf:
        return screen_straight(Cd, Cs2 - rangemax)
    else:
        return _mul(Cd, Cs2)

@cython.ufunc
cdef short hard_light(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, hard_light_straight)

cdef inline short pin_light_straight(short Cd, short Cs) noexcept nogil:
    cdef short Cs2 = _mul(Cs, range2)
    if Cs > rangehalf:
        return max(Cs2 - rangemax, Cd)
    else:
        return min(Cs2, Cd)

@cython.ufunc
cdef short pin_light(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, pin_light_straight)

@cython.ufunc
cdef short sai_hard_mix(short Cd, short Cs, short Ad, short As) noexcept nogil:
    cdef short Cdd = _clip_divide(Cd, Ad)
    cdef short Csd = _clip_divide(Cs, As)
    cdef short H = _clip_divide(Cdd - As + _mul(As, Csd), rangemax - As)
    return _lerp(Cs, H, Ad)

cdef inline short ts_hard_mix_straight(short Cd, short Cs) noexcept nogil:
    if (Cd + Cs) > rangemax:
        return rangemax
    else:
        return 0

@cython.ufunc
cdef short ts_hard_mix(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, ts_hard_mix_straight)

@cython.ufunc
cdef short darken(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return min(_mul(Cs, Ad), _mul(Cd, As)) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef short lighten(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return max(_mul(Cs, Ad), _mul(Cd, As)) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef short ts_difference(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return Cs + Cd - _mul(range2, min(_mul(Cs, Ad), _mul(Cd, As)))

@cython.ufunc
cdef short sai_difference(short Cd, short Cs, short Ad, short As) noexcept nogil:
    cdef short Cdd = _clip_divide(Cd, Ad)
    cdef short D = abs(Cdd - Cs)
    return _lerp(Cs, D, Ad)

@cython.ufunc
cdef short exclusion(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return (_mul(Cs, Ad) + _mul(Cd, As) - _mul(_mul(range2, Cs), Cd)) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef short subtract(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return max(0, _mul(Cd, As) - _mul(Cs, Ad)) + _comp2(Cd, Cs, Ad, As)

@cython.ufunc
cdef short divide(short Cd, short Cs, short Ad, short As) noexcept nogil:
    return _premul(Cd, Cs, Ad, As, _clip_divide)

cdef short gamma_r = 0x1332 # 0.30
cdef short gamma_g = 0x25c1 # 0.59
cdef short gamma_b = 0x070a # 0.11

cdef inline short _lum(Color C) noexcept nogil:
    return _mul(gamma_r, C[0]) + _mul(gamma_g, C[1]) + _mul(gamma_b, C[2])

cdef inline Color _clip_color(Color C) noexcept nogil:
    cdef short L = _lum(C)
    cdef short n = min(C[0], C[1], C[2])
    cdef short x = max(C[0], C[1], C[2])
    cdef short C_rL = C[0] - L
    cdef short C_gL = C[1] - L
    cdef short C_bL = C[2] - L
    cdef short Ln, L1, xL
    if n < 0:
        Ln = L - n
        C[0] = L + _safe_divide(_mul(C_rL, L), Ln)
        C[1] = L + _safe_divide(_mul(C_gL, L), Ln)
        C[2] = L + _safe_divide(_mul(C_bL, L), Ln)
    if x > rangemax:
        L1 = rangemax - L
        xL = x - L
        C[0] = L + _safe_divide(_mul(C_rL, L1), xL)
        C[1] = L + _safe_divide(_mul(C_gL, L1), xL)
        C[2] = L + _safe_divide(_mul(C_bL, L1), xL)
    return C

cdef inline Color _set_lum(Color C, short L) noexcept nogil:
    cdef short d = L - _lum(C)
    C[0] = C[0] + d
    C[1] = C[1] + d
    C[2] = C[2] + d
    return _clip_color(C)

cdef inline short _sat(Color C) noexcept nogil:
    return max(C[0], C[1], C[2]) - min(C[0], C[1], C[2])

cdef inline Color _set_sat(Color C, short S) noexcept nogil:
    cdef short amax = max(C[0], C[1], C[2])
    cdef short amin = min(C[0], C[1], C[2])
    cdef short amid
    cdef short* cmax
    cdef short* cmid
    cdef short* cmin
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
        cmid[0] = _safe_divide(_mul((amid - amin), S), amax - amin)
        cmax[0] = S
    else:
        cmid[0] = 0
        cmax[0] = 0
    cmin[0] = 0
    return (C[0], C[1], C[2])

cdef inline Color hue_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(_set_sat(Cs, _sat(Cd)), _lum(Cd))

@cython.ufunc
cdef Color hue_nonseperable(short Cd_r, short Cd_g, short Cd_b, short Cs_r, short Cs_g, short Cs_b, short Ad, short As) noexcept nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, hue_straight)

hue = nonseperable(hue_nonseperable)

cdef inline Color saturation_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(_set_sat(Cd, _sat(Cs)), _lum(Cd))

@cython.ufunc
cdef Color saturation_nonseperable(short Cd_r, short Cd_g, short Cd_b, short Cs_r, short Cs_g, short Cs_b, short Ad, short As) noexcept nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, saturation_straight)

saturation = nonseperable(saturation_nonseperable)

cdef inline Color color_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(Cs, _lum(Cd))

@cython.ufunc
cdef Color color_nonseperable(short Cd_r, short Cd_g, short Cd_b, short Cs_r, short Cs_g, short Cs_b, short Ad, short As) noexcept nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, color_straight)

color = nonseperable(color_nonseperable)

cdef inline Color luminosity_straight(Color Cd, Color Cs) noexcept nogil:
    return _set_lum(Cd, _lum(Cs))

@cython.ufunc
cdef Color luminosity_nonseperable(short Cd_r, short Cd_g, short Cd_b, short Cs_r, short Cs_g, short Cs_b, short Ad, short As) noexcept nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, luminosity_straight)

luminosity = nonseperable(luminosity_nonseperable)

cdef inline Color darker_color_straight(Color Cd, Color Cs) noexcept nogil:
    if _lum(Cs) < _lum(Cd):
        return Cs
    else:
        return Cd

@cython.ufunc
cdef Color darker_color_nonseperable(short Cd_r, short Cd_g, short Cd_b, short Cs_r, short Cs_g, short Cs_b, short Ad, short As) noexcept nogil:
    return _premul_nonseperable((Cd_r, Cd_g, Cd_b), (Cs_r, Cs_g, Cs_b), Ad, As, darker_color_straight)

darker_color = nonseperable(darker_color_nonseperable)

cdef inline Color lighter_color_straight(Color Cd, Color Cs) noexcept nogil:
    if _lum(Cs) > _lum(Cd):
        return Cs
    else:
        return Cd

@cython.ufunc
cdef Color lighter_color_nonseperable(short Cd_r, short Cd_g, short Cd_b, short Cs_r, short Cs_g, short Cs_b, short Ad, short As) noexcept nogil:
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
