"""Line images of tracers through the tube: shadowgraph preprocessing (L1) and a line detector (L2), the Python
prototype of the C++ port (plan: 2026-10-07-plan.md, Phase 4; report: 2026-10-07-lines-report.md).

The tube's astigmatism images each tracer as the lens aperture stretched by two different defocus blurs, across and
along the tube's axis: a short horizontal or vertical line at the two focal depths, an ellipse between and beyond
them.  Image coordinates as in OpenLPT: u = column (x), v = row (y), pixel centres at integers.

**L1, preprocessing (`preprocess`).**  The recording is a shadowgraph: dark particles on a bright background.
    signal = background - image            background: the per-pixel 90th percentile over the recording
    hp     = signal - median_31(signal)    removes the slowly varying dark clouds the background does not hold
    hp    -= beta                          the particles pull the median up (T1: by 0.5-0.75 noise sigma); beta is
                                           the level of the noise's negative tail, 2 Q(15.87 %) - Q(2.275 %)
    out    = clip(round(gain * hp), 0, 255)   8-bit, as OpenLPT reads it; negative noise is clipped to 0

**L2, the detector (`detect`).**  One object is modelled as an axis-aligned box of lengths (Lx, Ly) blurred by a
Gaussian, integrated over each pixel:
    m(x, y) = A * P(x - u; Lx, sx) * P(y - v; Ly, sy) + c,
    P(t; L, s) = the integral over the pixel [t - 1/2, t + 1/2] of S(tau) = (1/L) int_{-L/2}^{L/2} exp(-(tau - q)^2 / (2 s^2)) dq
(S(t; 0, s) = exp(-t^2 / (2 s^2))), so that the flux 2 pi A sx sy does not depend on the lengths: a particle's
shadow is spread as its image stretches.  A line is Ly = 0 (horizontal) or Lx = 0 (vertical), a round spot both
small.  Steps:
  1. matched filters: every template (Lx, Ly) of the bank DET["lengths_x"] x DET["lengths_y"] (blur sigma0), minus
     the mean of a box around it (so a smooth background or the clipping floor does not respond), each response
     divided by its own robust noise level over the frame;
  2. candidates: local maxima (3 x 3) of the best normalised response above k_detect;
  3. non-maximum suppression: by decreasing score, a candidate is dropped when its centre lies inside an accepted
     object's box (|du| <= Lx/2 + core_px, |dv| <= Ly/2 + core_px) or an accepted object's centre lies inside its
     own (a large template next to a bright object responds to that object);
  4. fit: Levenberg-Marquardt of m (8 parameters) on each object's window, all objects at once (batched), with the
     other objects' models held fixed in the window (deblending, Jacobi passes); with clipped = True the model is
     the expectation of the clipped image, E[max(0, m + noise)] = m Phi(m / s_n) + s_n phi(m / s_n); the centre
     stays within +- (0.25 L + 2.5 px) of the candidate and |c| <= 1.5 s_n (so that no object takes over a
     neighbour, and c cannot trade against A);
  5. second pass (optional): steps 1-4 on the residual image for objects hidden next to brighter ones (only the
     accepted objects' boxes, grown by 2 px, exclude a candidate there);
  6. validation: an object stays only if its own matched-filter score, on the image minus every other object's
     model, still reaches k_detect (drops the pieces of a long line that the suppression let through).
All parameters are in DET (one dict); they become config.json keys of `tracer.detector: "lines"` in the port (L5).
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
from scipy import ndimage, special

SQ2PI = math.sqrt(2.0 * math.pi)

# ------------------------------------------------------------------ parameters (one place)
DET = {
    "lengths_x": [0.0, 3.0, 5.0, 8.0, 12.0, 17.0, 24.0, 34.0],  # template box lengths along x (px)
    "lengths_y": [0.0, 3.0, 5.0, 8.0, 12.0, 17.0, 24.0],        # template box lengths along y (px)
    "sigma0": 1.0,            # Gaussian blur of the templates (px)
    "bg_guard": 4,            # local-mean box = template support + this many px on each side
    "noise_sub": 4,           # every n-th row and column for each template's robust noise level
    "k_detect": 6.0,          # detection threshold on the normalised score (pure clipped noise, 1280 x 800 px:
                              # 69 candidates per frame at 5, 20 at 5.5, 6.3 at 6, 1.7 at 6.5; l2_variants.py)
    "core_px": 2.0,           # suppression box half-size beyond L/2 (px)
    "fit_margin_sigma": 3.0,  # fit window: L/2 + this * sigma + 2 px each side
    "fit_max_half": 28,       # cap on a fit window's half-size (px)
    "fit_iters": 40,          # Levenberg-Marquardt iterations per pass (the centre settles in a few; the lengths
                              # and blurs of a short object, nearly degenerate, take up to ~100 to settle exactly)
    "deblend_passes": 2,      # Jacobi passes with the neighbours' models held fixed
    "second_pass": True,      # detection on the residual image
    "validate": True,         # step 6
    "L_min": 0.05, "L_max": 60.0, "sigma_min": 0.5, "sigma_max": 4.0,
    "centre_box_frac": 0.25,  # the fitted centre stays within the candidate's pixel +- (this * length + 2.5 px)
    "dynamic_bounds": True,   # 'length': the object's current fitted length, any length up to L_max; else (False)
    "length_growth": 1.5,     # the template's length, and the lengths stay below length_growth * template
    "length_margin": 8.0,     # length + length_margin (on T1 that pinned 55 % of the objects at 8 px: the texture
                              # of faint particles makes the long templates noisy, so objects are found with short
                              # ones)
    "nms_symmetric": True,    # step 3 also drops a candidate whose box holds an accepted object's centre
    "c_max_sigma": 1.5,       # the local constant c within +- this * noise (L1 centres the background on 0; with the
                              # clipped model a free c ran to -20 sigma, trading against A and the blur)
    "clipped": True,          # the image is clipped at 0 (8-bit preprocessed shadowgraph)
}


def det_params(**over) -> dict:
    p = dict(DET)
    p.update(over)
    return p


# ------------------------------------------------------------------ L1: preprocessing
def background(stack: np.ndarray, q: float = 90.0) -> np.ndarray:
    """Per-pixel q-th percentile over a stack of frames (n, H, W)."""
    return np.percentile(stack, q, axis=0).astype(np.float32)


def median_lowpass(sig: np.ndarray, size: int = 31, scale: float = 4.0, offset: float = 64.0,
                   smooth: int = 15) -> np.ndarray:
    """A size x size median of `sig` (float), computed by OpenCV's 8-bit median on round(scale * sig + offset)
    (resolution 1 / scale counts, range [-offset, (255 - offset)] / scale), then smoothed by a smooth x smooth box
    to remove the quantisation steps."""
    s8 = np.clip(np.round(sig * scale + offset), 0, 255).astype(np.uint8)
    med = (cv2.medianBlur(s8, size).astype(np.float32) - offset) / scale
    if smooth and smooth > 1:
        med = cv2.blur(med, (smooth, smooth), borderType=cv2.BORDER_REFLECT)
    return med


def highpass(img: np.ndarray, bg: np.ndarray, size: int = 31, recenter: bool = True) -> np.ndarray:
    """hp = (bg - img) - median_size(bg - img), float32 in counts; with recenter, minus the background level
    `noise_level` reads from the negative tail (the particles, all positive, pull the local median up)."""
    sig = bg.astype(np.float32) - img.astype(np.float32)
    hp = sig - median_lowpass(sig, size)
    if recenter:
        hp -= np.float32(noise_level(hp))
    return hp


def noise_level(hp: np.ndarray) -> float:
    """The level of a float image's noise from its negative tail: 2 Q(15.87 %) - Q(2.275 %) (the mean of Gaussian
    noise; objects, all positive, barely reach these quantiles; +0.15 sigma with 15 % of the pixels covered)."""
    q = _tail_quantiles(hp)
    return float(2 * q[1] - q[0])


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64).ravel()
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _tail_quantiles(hp: np.ndarray, sub: int = 7):
    """Q(2.275 %) and Q(15.87 %) of every sub-th value, each value spread by a uniform jitter of +-1/2 (seeded): the
    frames are whole counts, and where the background estimate is also whole (a flat synthetic background) the
    quantiles of hp would snap to whole numbers."""
    x = np.asarray(hp, np.float64).ravel()[::sub]
    x = x + np.random.default_rng(0).uniform(-0.5, 0.5, x.size)
    return np.percentile(x, [2.275, 15.865])


def noise_left(hp: np.ndarray) -> float:
    """Noise sigma of a float image whose objects are positive (hp): the spread of its negative tail,
    Q(15.87 %) - Q(2.275 %), which equals sigma for Gaussian noise and which the objects barely reach (+2 % with
    10 % of the pixels covered, where the MAD reads +10 %); the jitter's variance 1/12 is taken off."""
    q = _tail_quantiles(hp)
    return float(math.sqrt(max((q[1] - q[0]) ** 2 - 1.0 / 12.0, 1e-12)))


def to_8bit(hp: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(np.round(gain * hp), 0, 255).astype(np.uint8)


def preprocess(img: np.ndarray, bg: np.ndarray, gain: float, size: int = 31):
    """(hp in counts, the 8-bit output)."""
    hp = highpass(img, bg, size)
    return hp, to_8bit(hp, gain)


# ------------------------------------------------------------------ the profile and its derivatives
def _psi(z):
    return z * special.ndtr(z) + np.exp(-0.5 * z * z) / SQ2PI


def profile(t, L, s, deriv: bool = False):
    """P(t; L, s), the pixel integral of the box (length L) blurred by a Gaussian (sigma s), with S(0; 0, s) = 1.
    Broadcasts.  With deriv=True also returns dP/dt, dP/dL, dP/ds.  L must be >= 0.05 for the derivatives (the
    L = 0 limit is used below 1e-6)."""
    t = np.asarray(t, np.float64)
    L = np.asarray(L, np.float64)
    s = np.asarray(s, np.float64)
    small = L < 1e-6
    Ls = np.where(small, 1.0, L)
    z1 = (t + 0.5 + Ls / 2) / s
    z2 = (t + 0.5 - Ls / 2) / s
    z3 = (t - 0.5 + Ls / 2) / s
    z4 = (t - 0.5 - Ls / 2) / s
    K = s * s * SQ2PI / Ls
    D = _psi(z1) - _psi(z2) - _psi(z3) + _psi(z4)
    P = K * D
    # L -> 0: the pixel-integrated Gaussian
    za, zb = (t + 0.5) / s, (t - 0.5) / s
    P0 = s * SQ2PI * (special.ndtr(za) - special.ndtr(zb))
    P = np.where(small, P0, P)
    if not deriv:
        return P
    F1, F2, F3, F4 = special.ndtr(z1), special.ndtr(z2), special.ndtr(z3), special.ndtr(z4)
    dP_dt = K * (F1 - F2 - F3 + F4) / s
    dP_dL = -P / Ls + K * (F1 + F2 - F3 - F4) / (2.0 * s)
    dP_ds = 2.0 * P / s - K * (z1 * F1 - z2 * F2 - z3 * F3 + z4 * F4) / s
    g = lambda z: np.exp(-0.5 * z * z)
    dP0_dt = (g(za) - g(zb))
    dP0_ds = P0 / s - (za * g(za) - zb * g(zb))
    dP_dt = np.where(small, dP0_dt, dP_dt)
    dP_dL = np.where(small, 0.0, dP_dL)
    dP_ds = np.where(small, dP0_ds, dP_ds)
    return P, dP_dt, dP_dL, dP_ds


def kernel1d(L: float, s: float, half: int | None = None) -> np.ndarray:
    """The 1-D template P(n; L, s) on n = -half..half."""
    if half is None:
        half = int(math.ceil(L / 2 + 3 * s))
    n = np.arange(-half, half + 1, dtype=np.float64)
    return profile(n, L, s)


def clip_expect(m, sn):
    """E[max(0, m + noise)] for noise N(0, sn^2), and its derivative Phi(m / sn)."""
    z = m / sn
    Phi = special.ndtr(z)
    return m * Phi + sn * np.exp(-0.5 * z * z) / SQ2PI, Phi


# ------------------------------------------------------------------ synthetic objects
def render(shape_hw, objs, sigma_n: float = 0.0, rng=None, n_slices: int = 400):
    """Noiseless (and optionally noisy) image of objects, each a uniform ellipse (semi-axes a along x, b along y)
    blurred by a Gaussian sigma0 and integrated over the pixels: the geometric image of a circular aperture under
    astigmatism (a or b = 0 is a line whose brightness falls off as sqrt(1 - (x/a)^2) towards its ends).
    objs: array of rows (u, v, a, b, sigma0, flux).  The ellipse is cut into n_slices vertical chords at
    x = u + a xi (xi the midpoints of n_slices equal steps of [-1, 1]), each a uniform segment of length
    2 b sqrt(1 - xi^2) and weight sqrt(1 - xi^2): exact in y (profile()), a pixel-integrated Gaussian in x; this
    holds for lines (a or b = 0) as well.  Returns (image float64, peak value of each object alone)."""
    H, W = shape_hw
    img = np.zeros((H, W), np.float64)
    xi = -1.0 + (2.0 * np.arange(n_slices) + 1.0) / n_slices
    root = np.sqrt(1.0 - xi * xi)
    w = root / root.sum()
    peaks = np.zeros(len(objs))
    for k, (u, v, a, b, s0, flux) in enumerate(objs):
        hx = int(math.ceil(a + 4 * s0)) + 1
        hy = int(math.ceil(b + 4 * s0)) + 1
        c0, r0 = int(round(u)), int(round(v))
        cols = np.arange(c0 - hx, c0 + hx + 1)
        rows = np.arange(r0 - hy, r0 + hy + 1)
        px = u + a * xi
        gx = special.ndtr((cols[None, :] + 0.5 - px[:, None]) / s0) - special.ndtr((cols[None, :] - 0.5 - px[:, None]) / s0)
        gy = profile(rows[None, :] - v, (2.0 * b * root)[:, None], s0) / (s0 * SQ2PI)   # unit flux in y
        patch = flux * (gy * w[:, None]).T @ gx            # (rows, cols)
        peaks[k] = patch.max()
        c_lo, c_hi = max(cols[0], 0), min(cols[-1], W - 1)
        r_lo, r_hi = max(rows[0], 0), min(rows[-1], H - 1)
        if c_lo > c_hi or r_lo > r_hi:
            continue
        img[r_lo:r_hi + 1, c_lo:c_hi + 1] += patch[r_lo - rows[0]:r_hi - rows[0] + 1, c_lo - cols[0]:c_hi - cols[0] + 1]
    if sigma_n > 0:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0.0, sigma_n, img.shape)
    return img, peaks


# ------------------------------------------------------------------ L2: matched filters and candidates
def _box_mean(img, hx, hy):
    return cv2.blur(img, (2 * hx + 1, 2 * hy + 1), borderType=cv2.BORDER_REFLECT)


def matched_filter(img: np.ndarray, P: dict, noise_ref=None):
    """Best normalised response over the bank and its template index, per pixel.  Returns (score, index,
    templates) where templates[i] = (Lx, Ly, kx, ky, sum, norm2, sigma_resp).  noise_ref: an image to take each
    template's robust noise level from (default: img itself)."""
    s0 = float(P["sigma0"])
    g = int(P["bg_guard"])
    sub = int(P["noise_sub"])
    img = img.astype(np.float32)
    ref = img if noise_ref is None else noise_ref.astype(np.float32)
    best = np.full(img.shape, -np.inf, np.float32)
    idx = np.zeros(img.shape, np.int16)
    templates = []
    xconv = {}
    for Lx in P["lengths_x"]:
        kx = kernel1d(Lx, s0)
        xconv[Lx] = (kx, ndimage.correlate1d(img, kx.astype(np.float32), axis=1, mode="reflect"),
                     None if noise_ref is None else ndimage.correlate1d(ref, kx.astype(np.float32), axis=1, mode="reflect"))
    for Lx in P["lengths_x"]:
        kx, cx, cxr = xconv[Lx]
        hx = (len(kx) - 1) // 2
        for Ly in P["lengths_y"]:
            ky = kernel1d(Ly, s0)
            hy = (len(ky) - 1) // 2
            resp = ndimage.correlate1d(cx, ky.astype(np.float32), axis=0, mode="reflect")
            ssum = float(kx.sum() * ky.sum())
            bx, by = hx + g, hy + g
            nbox = (2 * bx + 1) * (2 * by + 1)
            resp = resp - ssum * _box_mean(img, bx, by)
            norm2 = float((kx ** 2).sum() * (ky ** 2).sum()) - ssum ** 2 / nbox
            if noise_ref is None:
                rs = resp[::sub, ::sub]
            else:
                rr = ndimage.correlate1d(cxr, ky.astype(np.float32), axis=0, mode="reflect")
                rs = (rr - ssum * _box_mean(ref, bx, by))[::sub, ::sub]
            sig = robust_sigma(rs)
            sig = max(sig, 1e-6)
            nr = resp / sig
            better = nr > best
            best[better] = nr[better]
            idx[better] = len(templates)
            templates.append((float(Lx), float(Ly), kx, ky, ssum, norm2, sig))
    return best, idx, templates


def local_maxima(score: np.ndarray, k: float):
    """(rows, cols) of 3 x 3 local maxima of score >= k (ties: the first in raster order)."""
    mx = ndimage.maximum_filter(score, size=3, mode="nearest")
    cand = (score >= k) & (score >= mx)
    r, c = np.nonzero(cand)
    # drop plateau duplicates: keep a pixel only if no earlier raster neighbour has the same value
    keep = np.ones(len(r), bool)
    s = score[r, c]
    order = np.lexsort((c, r))
    taken = set()
    for i in order:
        key = (r[i], c[i])
        dup = False
        for dr in (-1, 0):
            for dc in (-1, 0, 1):
                if dr == 0 and dc >= 0:
                    continue
                if (r[i] + dr, c[i] + dc) in taken and score[r[i] + dr, c[i] + dc] == s[i]:
                    dup = True
        if dup:
            keep[i] = False
        else:
            taken.add(key)
    return r[keep], c[keep]


def suppress(rows, cols, scores, Ls, P, existing=None, symmetric=True, core=None):
    """Greedy non-maximum suppression by decreasing score; Ls[i] = (Lx, Ly) of candidate i's template.  A candidate
    is dropped when its centre lies inside an accepted object's box (|du| <= Lx/2 + core, |dv| <= Ly/2 + core), and,
    with symmetric, also when an accepted object's centre lies inside the candidate's own box: a large template that
    covers a brighter object responds to that object (on the first pass every bright compact object had such
    off-centre companions).  existing: (u, v, Lx, Ly) already accepted.  Returns the indices kept."""
    core = float(P["core_px"]) if core is None else float(core)
    order = np.argsort(-scores, kind="stable")
    cell = 16
    boxes = {}      # cell -> accepted boxes overlapping it
    centres = {}    # cell -> accepted centres in it
    acc = []

    def add(u, v, Lx, Ly):
        k = len(acc)
        acc.append((u, v, Lx, Ly))
        rx, ry = Lx / 2 + core, Ly / 2 + core
        for gx in range(int((u - rx) // cell), int((u + rx) // cell) + 1):
            for gy in range(int((v - ry) // cell), int((v + ry) // cell) + 1):
                boxes.setdefault((gx, gy), []).append(k)
        centres.setdefault((int(u // cell), int(v // cell)), []).append(k)

    for (u, v, Lx, Ly) in (existing or []):
        add(u, v, Lx, Ly)
    kept = []
    for i in order:
        u, v = float(cols[i]), float(rows[i])
        Lx, Ly = Ls[i]
        hit = False
        for k in boxes.get((int(u // cell), int(v // cell)), ()):
            au, av, aLx, aLy = acc[k]
            if abs(u - au) <= aLx / 2 + core and abs(v - av) <= aLy / 2 + core:
                hit = True
                break
        if not hit and symmetric:
            rx, ry = Lx / 2 + core, Ly / 2 + core
            for gx in range(int((u - rx) // cell), int((u + rx) // cell) + 1):
                for gy in range(int((v - ry) // cell), int((v + ry) // cell) + 1):
                    for k in centres.get((gx, gy), ()):
                        au, av = acc[k][0], acc[k][1]
                        if abs(u - au) <= rx and abs(v - av) <= ry:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
        if hit:
            continue
        kept.append(i)
        add(u, v, Lx, Ly)
    return np.array(kept, int)


# ------------------------------------------------------------------ L2: the batched fit
NPAR = 8   # A, u, v, Lx, Ly, sx, sy, c


def _windows(params, P, shape):
    """Integer window half-sizes per object from its current lengths and blurs."""
    m = float(P["fit_margin_sigma"])
    cap = int(P["fit_max_half"])
    hx = np.minimum(np.ceil(params[:, 3] / 2 + m * params[:, 5] + 2), cap).astype(int)
    hy = np.minimum(np.ceil(params[:, 4] / 2 + m * params[:, 6] + 2), cap).astype(int)
    return np.maximum(hx, 3), np.maximum(hy, 3)


def _model_parts(params, xs, ys, deriv):
    A, u, v, Lx, Ly, sx, sy, c = [params[:, i][:, None] for i in range(NPAR)]
    Lxc = np.maximum(Lx, 1e-6)
    Lyc = np.maximum(Ly, 1e-6)
    if deriv:
        px, dpx_t, dpx_L, dpx_s = profile(xs - u, Lxc, sx, True)
        py, dpy_t, dpy_L, dpy_s = profile(ys - v, Lyc, sy, True)
        return px, py, dpx_t, dpx_L, dpx_s, dpy_t, dpy_L, dpy_s
    return profile(xs - u, Lxc, sx), profile(ys - v, Lyc, sy)


def render_models(shape, params, P, exclude_c: bool = True):
    """Sum of every object's model (without its constant c) over the image."""
    H, W = shape
    out = np.zeros((H, W), np.float64)
    if len(params) == 0:
        return out
    hx, hy = _windows(params, P, shape)
    for (HX, HY), sel in _buckets(hx, hy):
        pr = params[sel]
        u0 = np.round(pr[:, 1]).astype(int)
        v0 = np.round(pr[:, 2]).astype(int)
        xs = u0[:, None] + np.arange(-HX, HX + 1)[None, :]
        ys = v0[:, None] + np.arange(-HY, HY + 1)[None, :]
        px, py = _model_parts(pr, xs.astype(np.float64), ys.astype(np.float64), False)
        m = pr[:, 0][:, None, None] * py[:, :, None] * px[:, None, :]
        okx = (xs >= 0) & (xs < W)
        oky = (ys >= 0) & (ys < H)
        mask = oky[:, :, None] & okx[:, None, :]
        rr = np.broadcast_to(ys[:, :, None], m.shape)
        cc = np.broadcast_to(xs[:, None, :], m.shape)
        np.add.at(out, (rr[mask], cc[mask]), m[mask])
    return out


def _buckets(hx, hy, step: int = 4):
    """Group objects by window size, rounded up to a multiple of `step`."""
    bx = ((hx + step - 1) // step) * step
    by = ((hy + step - 1) // step) * step
    keys = {}
    for i, k in enumerate(zip(bx.tolist(), by.tolist())):
        keys.setdefault(k, []).append(i)
    return [((k[0], k[1]), np.array(v, int)) for k, v in sorted(keys.items())]


def _clamp(p, P, bounds=None, cmax=None):
    """Keep the parameters in their ranges; bounds (n, 6) = (u_lo, u_hi, v_lo, v_hi, Lx_hi, Ly_hi) per object;
    cmax: |c| <= cmax."""
    p[:, 0] = np.maximum(p[:, 0], 0.0)
    if cmax is not None:
        p[:, 7] = np.clip(p[:, 7], -cmax, cmax)
    p[:, 3] = np.clip(p[:, 3], P["L_min"], P["L_max"])
    p[:, 4] = np.clip(p[:, 4], P["L_min"], P["L_max"])
    p[:, 5] = np.clip(p[:, 5], P["sigma_min"], P["sigma_max"])
    p[:, 6] = np.clip(p[:, 6], P["sigma_min"], P["sigma_max"])
    if bounds is not None:
        if P.get("dynamic_bounds"):
            u0, v0 = 0.5 * (bounds[:, 0] + bounds[:, 1]), 0.5 * (bounds[:, 2] + bounds[:, 3])
            hu = float(P["centre_box_frac"]) * p[:, 3] + 2.5
            hv = float(P["centre_box_frac"]) * p[:, 4] + 2.5
            p[:, 1] = np.clip(p[:, 1], u0 - hu, u0 + hu)
            p[:, 2] = np.clip(p[:, 2], v0 - hv, v0 + hv)
        else:
            p[:, 1] = np.clip(p[:, 1], bounds[:, 0], bounds[:, 1])
            p[:, 2] = np.clip(p[:, 2], bounds[:, 2], bounds[:, 3])
            p[:, 3] = np.minimum(p[:, 3], bounds[:, 4])
            p[:, 4] = np.minimum(p[:, 4], bounds[:, 5])
    return p


def fit_objects(img, params, P, noise, offsets_img=None, fit_mask=None, bounds=None):
    """Levenberg-Marquardt of every object's 8 parameters on its own window, all at once (bucketed by window
    size; each iteration works only on the objects not yet converged).  offsets_img: the other objects' models to
    hold fixed (an image, from which each object's own current model is subtracted here).  fit_mask: fit only these
    objects (the others are returned unchanged).  bounds: per-object limits, see _clamp.  Returns (params, chi2 per
    object, pixels per object, converged)."""
    H, W = img.shape
    cmax = float(P["c_max_sigma"]) * float(noise)
    params = _clamp(np.array(params, np.float64), P, bounds, cmax)
    n = len(params)
    chi2 = np.full(n, np.nan)
    npx = np.zeros(n, int)
    conv = np.zeros(n, bool)
    if n == 0:
        return params, chi2, npx, conv
    clipped = bool(P["clipped"])
    sn = float(noise)
    hx_all, hy_all = _windows(params, P, img.shape)
    sel_all = np.arange(n) if fit_mask is None else np.flatnonzero(fit_mask)
    hx, hy = hx_all[sel_all], hy_all[sel_all]
    eye = np.eye(NPAR)[None]
    for (HX, HY), sub in _buckets(hx, hy, step=8):
        sel = sel_all[sub]
        p = params[sel].copy()
        bd = None if bounds is None else bounds[sel]
        u0 = np.round(p[:, 1]).astype(int)
        v0 = np.round(p[:, 2]).astype(int)
        xs = (u0[:, None] + np.arange(-HX, HX + 1)[None, :])
        ys = (v0[:, None] + np.arange(-HY, HY + 1)[None, :])
        okx = (xs >= 0) & (xs < W) & (np.abs(xs - u0[:, None]) <= hx[sub][:, None])
        oky = (ys >= 0) & (ys < H) & (np.abs(ys - v0[:, None]) <= hy[sub][:, None])
        mask = (oky[:, :, None] & okx[:, None, :]).astype(np.float64)
        xc = np.clip(xs, 0, W - 1)
        yc = np.clip(ys, 0, H - 1)
        D = img[yc[:, :, None], xc[:, None, :]].astype(np.float64)
        xs_f, ys_f = xs.astype(np.float64), ys.astype(np.float64)
        if offsets_img is not None:
            O = offsets_img[yc[:, :, None], xc[:, None, :]].astype(np.float64)
            px, py = _model_parts(p, xs_f, ys_f, False)
            O = O - p[:, 0][:, None, None] * py[:, :, None] * px[:, None, :]   # the object's own model out
        else:
            O = np.zeros_like(D)

        def evaluate(pp, idx, deriv):
            parts = _model_parts(pp, xs_f[idx], ys_f[idx], deriv)
            px, py = parts[0], parts[1]
            A = pp[:, 0][:, None, None]
            mlin = A * py[:, :, None] * px[:, None, :] + O[idx] + pp[:, 7][:, None, None]
            if clipped:
                m, dfac = clip_expect(mlin, sn)
            else:
                m, dfac = mlin, None
            r = (D[idx] - m) * mask[idx]
            cost = (r * r).sum(axis=(1, 2))
            if not deriv:
                return cost, r
            _, _, dpx_t, dpx_L, dpx_s, dpy_t, dpy_L, dpy_s = parts
            J = np.empty(r.shape + (NPAR,))
            J[..., 0] = py[:, :, None] * px[:, None, :]
            J[..., 1] = -A * py[:, :, None] * dpx_t[:, None, :]
            J[..., 2] = -A * dpy_t[:, :, None] * px[:, None, :]
            J[..., 3] = A * py[:, :, None] * dpx_L[:, None, :]
            J[..., 4] = A * dpy_L[:, :, None] * px[:, None, :]
            J[..., 5] = A * py[:, :, None] * dpx_s[:, None, :]
            J[..., 6] = A * dpy_s[:, :, None] * px[:, None, :]
            J[..., 7] = 1.0
            fac = mask[idx] if dfac is None else dfac * mask[idx]
            J *= fac[..., None]
            return cost, r, J

        nb = len(sel)
        lam = np.full(nb, 1e-2)
        done = np.zeros(nb, bool)
        act = np.arange(nb)
        need = True
        for it in range(int(P["fit_iters"])):
            if len(act) == 0:
                break
            pa = p[act]
            cost, r, J = evaluate(pa, act, True)
            Jf = J.reshape(len(act), -1, NPAR)
            JT = Jf.transpose(0, 2, 1)
            JTJ = np.matmul(JT, Jf)
            JTr = np.matmul(JT, r.reshape(len(act), -1, 1))[:, :, 0]
            dg = np.einsum("nii->ni", JTJ)
            Areg = JTJ + (lam[act][:, None] * np.maximum(dg, 1e-12))[:, :, None] * eye
            try:
                step = np.linalg.solve(Areg, JTr[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                step = np.zeros_like(JTr)
                for i in range(len(act)):
                    try:
                        step[i] = np.linalg.lstsq(Areg[i], JTr[i], rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            pn = _clamp(pa + step, P, None if bd is None else bd[act], cmax)
            cn, _ = evaluate(pn, act, False)
            better = cn < cost
            rel = np.where(better, (cost - cn) / np.maximum(cost, 1e-12), 0.0)
            p[act[better]] = pn[better]
            la = lam[act]
            la = np.where(better, np.maximum(la / 3.0, 1e-7), np.minimum(la * 4.0, 1e7))
            lam[act] = la
            # converged: the cost no longer falls and every parameter's step is small (the lengths and blurs of a
            # short object lie in a flat valley, so the centre settles long before them); or no step helps
            small_step = (np.all(np.abs(step[:, 1:7]) < 1e-5, axis=1)
                          & (np.abs(step[:, 0]) < 1e-6 * np.maximum(pa[:, 0], 1e-9)))
            fin = (better & (rel < 1e-10) & small_step) | (la >= 1e7)
            done[act[fin]] = True
            act = act[~fin]
        allidx = np.arange(nb)
        cost, _ = evaluate(p, allidx, False)
        params[sel] = p
        chi2[sel] = cost / (sn * sn)
        npx[sel] = mask.sum(axis=(1, 2)).astype(int)
        conv[sel] = done | (lam < 1e7)
    return params, chi2, npx, conv


# ------------------------------------------------------------------ L2: the detector
DET_DTYPE = np.dtype([("u", "f8"), ("v", "f8"), ("Lx", "f8"), ("Ly", "f8"), ("sx", "f8"), ("sy", "f8"),
                      ("wx", "f8"), ("wy", "f8"), ("A", "f8"), ("c", "f8"), ("flux", "f8"), ("peak", "f8"),
                      ("score", "f8"), ("chi2", "f8"), ("npx", "i4"), ("cls", "U7"), ("pass_", "i2")])
# wx, wy: the RMS widths sqrt(L^2 / 12 + s^2), well determined where L and s alone are not (short objects)


def classify_shape(Lx, Ly):
    Lx, Ly = np.asarray(Lx), np.asarray(Ly)
    out = np.full(Lx.shape, "ellipse", dtype="U7")
    out[(Lx >= 4) & (Lx >= 2 * Ly) & (Lx >= Ly + 4)] = "H"
    out[(Ly >= 4) & (Ly >= 2 * Lx) & (Ly >= Lx + 4)] = "V"
    out[(np.maximum(Lx, Ly) < 4)] = "round"
    return out


def _initial(img_bgsub, rows, cols, tidx, templates, P):
    """Initial parameters from the matched template (amplitude by least squares on the template), and the bounds
    of each object's fit (see _clamp and DET's centre_box_frac, length_growth)."""
    H, W = img_bgsub.shape
    p = np.zeros((len(rows), NPAR))
    bounds = np.zeros((len(rows), 6))
    fr, gr = float(P["centre_box_frac"]), float(P["length_growth"])
    for i, (r, c, t) in enumerate(zip(rows, cols, tidx)):
        Lx, Ly, kx, ky, ssum, norm2, sig = templates[t]
        hx, hy = (len(kx) - 1) // 2, (len(ky) - 1) // 2
        r0, r1 = max(r - hy, 0), min(r + hy, H - 1)
        c0, c1 = max(c - hx, 0), min(c + hx, W - 1)
        T = np.outer(ky[r0 - r + hy:r1 - r + hy + 1], kx[c0 - c + hx:c1 - c + hx + 1])
        patch = img_bgsub[r0:r1 + 1, c0:c1 + 1]
        A = float((T * patch).sum() / max((T * T).sum(), 1e-12))
        p[i] = (max(A, 1e-3), c, r, max(Lx, 0.5), max(Ly, 0.5), 1.0, 1.0, 0.0)
        bx, by = fr * Lx + 2.5, fr * Ly + 2.5
        lm = float(P["length_margin"])
        bounds[i] = (c - bx, c + bx, r - by, r + by, gr * Lx + lm, gr * Ly + lm)
    return p, bounds


def detect(img: np.ndarray, noise: float, P: dict | None = None, timing: dict | None = None):
    """Detect line, ellipse and round images in a preprocessed (8-bit or float, background ~ 0) image.
    noise: the image's noise sigma in its own counts (before clipping).  Returns a DET_DTYPE array."""
    P = P or DET
    t0 = time.perf_counter()
    I = img.astype(np.float64)
    H, W = I.shape
    score, tidx, templates = matched_filter(I, P)
    t1 = time.perf_counter()
    rows, cols = local_maxima(score, float(P["k_detect"]))
    Ls = [(templates[t][0], templates[t][1]) for t in tidx[rows, cols]]
    keep = suppress(rows, cols, score[rows, cols], Ls, P, symmetric=bool(P["nms_symmetric"]))
    rows, cols = rows[keep], cols[keep]
    sc = score[rows, cols].astype(np.float64)
    params, bounds = _initial(I, rows, cols, tidx[rows, cols], templates, P)
    passes = np.zeros(len(params), int)
    t2 = time.perf_counter()
    for _ in range(int(P["deblend_passes"])):
        allm = render_models(I.shape, params, P)
        params, chi2, npx, conv = fit_objects(I, params, P, noise, offsets_img=allm, bounds=bounds)
    t3 = time.perf_counter()
    if P["second_pass"] and len(params):
        allm = render_models(I.shape, params, P)
        resid = I - (clip_expect(allm, noise)[0] if P["clipped"] else allm)
        score2, tidx2, templates2 = matched_filter(resid, P, noise_ref=I)
        r2, c2 = local_maxima(score2, float(P["k_detect"]))
        Ls2 = [(templates2[t][0], templates2[t][1]) for t in tidx2[r2, c2]]
        existing = [(p[1], p[2], p[3], p[4]) for p in params]
        # on the residual the accepted objects are gone, so only their boxes (grown by 2 px) exclude a candidate
        keep2 = suppress(r2, c2, score2[r2, c2], Ls2, P, existing=existing, symmetric=False,
                         core=float(P["core_px"]) + 2.0)
        if len(keep2):
            r2, c2 = r2[keep2], c2[keep2]
            p2, b2 = _initial(resid, r2, c2, tidx2[r2, c2], templates2, P)
            params = np.vstack([params, p2])
            bounds = np.vstack([bounds, b2])
            sc = np.concatenate([sc, score2[r2, c2]])
            passes = np.concatenate([passes, np.ones(len(p2), int)])
            # refit the new objects and the old ones whose windows reach them
            new = np.zeros(len(params), bool)
            new[-len(p2):] = True
            hx, hy = _windows(params, P, I.shape)
            from scipy.spatial import cKDTree
            near = np.zeros(len(params), bool)
            tree = cKDTree(params[new][:, 1:3])
            reach = float(2 * P["fit_max_half"])
            for i, nb in enumerate(tree.query_ball_point(params[:, 1:3], r=reach)):
                for j in nb:
                    q = np.flatnonzero(new)[j]
                    if abs(params[i, 1] - params[q, 1]) <= hx[i] + hx[q] and abs(params[i, 2] - params[q, 2]) <= hy[i] + hy[q]:
                        near[i] = True
                        break
            fm = new | near
            for _ in range(int(P["deblend_passes"])):
                allm = render_models(I.shape, params, P)
                params, chi2, npx, conv = fit_objects(I, params, P, noise, offsets_img=allm, fit_mask=fm,
                                                      bounds=bounds)
    t4 = time.perf_counter()
    if P["validate"] and len(params):
        # each object's own score on the image minus every other object's model (all models subtracted, its own
        # added back in _own_scores); the weakest drop out, the rest are refitted, twice at most
        tb = {(t[0], t[1]): (t[2], t[3], t[4], t[6]) for t in templates}
        for _ in range(2):
            own = _own_scores(I - render_models(I.shape, params, P), params, P, tb)
            keep = own >= float(P["k_detect"])
            if keep.all():
                break
            params, sc, passes, bounds = params[keep], sc[keep], passes[keep], bounds[keep]
            params, chi2, npx, conv = fit_objects(I, params, P, noise, offsets_img=render_models(I.shape, params, P),
                                                  bounds=bounds)
    t5 = time.perf_counter()
    chi2, npx = np.zeros(len(params)), np.zeros(len(params), int)
    if len(params):
        params, chi2, npx, conv = fit_objects(I, params, P, noise, offsets_img=render_models(I.shape, params, P),
                                              bounds=bounds)
    out = np.zeros(len(params), DET_DTYPE)
    if len(params):
        out["A"], out["u"], out["v"] = params[:, 0], params[:, 1], params[:, 2]
        out["Lx"], out["Ly"], out["sx"], out["sy"], out["c"] = params[:, 3], params[:, 4], params[:, 5], params[:, 6], params[:, 7]
        out["wx"] = np.sqrt(params[:, 3] ** 2 / 12 + params[:, 5] ** 2)
        out["wy"] = np.sqrt(params[:, 4] ** 2 / 12 + params[:, 6] ** 2)
        out["flux"] = 2 * np.pi * params[:, 0] * params[:, 5] * params[:, 6]
        out["peak"] = params[:, 0] * profile(0.0, np.maximum(params[:, 3], 1e-6), params[:, 5]) * \
            profile(0.0, np.maximum(params[:, 4], 1e-6), params[:, 6])
        out["score"] = sc
        out["chi2"], out["npx"] = chi2, npx
        out["cls"] = classify_shape(params[:, 3], params[:, 4])
        out["pass_"] = passes
    t6 = time.perf_counter()
    if timing is not None:
        timing.update({"filters_s": t1 - t0, "candidates_s": t2 - t1, "fit_s": t3 - t2, "second_pass_s": t4 - t3,
                       "validate_s": t5 - t4, "final_fit_s": t6 - t5, "total_s": t6 - t0})
    return out


def _own_scores(resid, params, P, tb):
    """Each object's own normalised matched-filter score on resid + its own model (= the image minus every other
    object's model), with the bank template nearest to its fitted lengths."""
    H, W = resid.shape
    g = int(P["bg_guard"])
    out = np.zeros(len(params))
    for i, pr in enumerate(params):
        Lx = min(P["lengths_x"], key=lambda L: abs(L - pr[3]))
        Ly = min(P["lengths_y"], key=lambda L: abs(L - pr[4]))
        kx, ky, ssum, sig = tb[(Lx, Ly)]
        hx, hy = (len(kx) - 1) // 2, (len(ky) - 1) // 2
        bx, by = hx + g, hy + g
        r, c = int(round(pr[2])), int(round(pr[1]))
        if not (0 <= r < H and 0 <= c < W):
            continue
        r0, r1 = max(r - by, 0), min(r + by, H - 1)
        c0, c1 = max(c - bx, 0), min(c + bx, W - 1)
        xs = np.arange(c0, c1 + 1, dtype=np.float64)
        ys = np.arange(r0, r1 + 1, dtype=np.float64)
        own = pr[0] * np.outer(profile(ys - pr[2], max(pr[4], 1e-6), pr[6]), profile(xs - pr[1], max(pr[3], 1e-6), pr[5]))
        img = resid[r0:r1 + 1, c0:c1 + 1] + own
        mean = float(img.mean())
        rr0, rr1 = max(r - hy, 0), min(r + hy, H - 1)
        cc0, cc1 = max(c - hx, 0), min(c + hx, W - 1)
        T = np.outer(ky[rr0 - r + hy:rr1 - r + hy + 1], kx[cc0 - c + hx:cc1 - c + hx + 1])
        sub = img[rr0 - r0:rr1 - r0 + 1, cc0 - c0:cc1 - c0 + 1]
        out[i] = float((T * (sub - mean)).sum()) / sig
    return out


# ------------------------------------------------------------------ the noise of a clipped image
def estimate_noise_clipped(img8: np.ndarray, max_k: float = 1.0) -> float:
    """Noise sigma of an image whose background is ~0 and whose negative values were clipped to 0: a quadratic fit
    of log(histogram count) against the value on 1 .. ~max_k sigma (the half-normal of the background), iterated.
    A fallback: the objects' faint tails bias it up by a few % (+4 % with 15 % of the pixels covered); use the
    noise of the float image (noise_left) when it is known."""
    v = img8.ravel()
    h = np.bincount(v.astype(np.int64), minlength=256).astype(np.float64)
    sig = max(1.0, float(np.percentile(v[v > 0], 40)) / 0.5244) if (v > 0).any() else 1.0
    for _ in range(6):
        top = max(4, int(round(max_k * sig)))
        x = np.arange(1, top + 1)
        y = h[1:top + 1]
        ok = y > 0
        if ok.sum() < 3:
            break
        c = np.polyfit(x[ok], np.log(y[ok]), 2, w=np.sqrt(y[ok]))
        if c[0] >= 0:
            break
        new = math.sqrt(-1.0 / (2.0 * c[0]))
        if abs(new - sig) < 1e-3 * sig:
            sig = new
            break
        sig = new
    return float(sig)


# ------------------------------------------------------------------ scoring against a truth list
def match_truth(det_uv, truth, gate_pad: float = 2.0):
    """One-to-one matching of detections to true objects.  truth rows (u, v, a, b, ...): an object's footprint is
    the ellipse |du| <= a + pad, |dv| <= b + pad (normalised distance <= 1).  Greedy by normalised distance.
    Returns (match index per truth, -1 if none; per detection: matched truth or -1; per detection: the truth whose
    footprint holds it, or -1)."""
    det_uv = np.asarray(det_uv, np.float64).reshape(-1, 2)
    nt, nd = len(truth), len(det_uv)
    tm = np.full(nt, -1)
    dm = np.full(nd, -1)
    inside = np.full(nd, -1)
    if nt == 0 or nd == 0:
        return tm, dm, inside
    from scipy.spatial import cKDTree
    tree = cKDTree(det_uv)
    pairs = []
    for i, (u, v, a, b) in enumerate(truth[:, :4]):
        ra, rb = a + gate_pad, b + gate_pad
        for j in tree.query_ball_point((u, v), r=max(ra, rb)):
            du, dv = det_uv[j, 0] - u, det_uv[j, 1] - v
            dn = math.hypot(du / ra, dv / rb)
            if dn <= 1.0:
                pairs.append((dn, i, j))
    pairs.sort()
    best_inside = {}
    for dn, i, j in pairs:
        if j not in best_inside or dn < best_inside[j][0]:
            best_inside[j] = (dn, i)
        if tm[i] < 0 and dm[j] < 0:
            tm[i], dm[j] = j, i
    for j, (_, i) in best_inside.items():
        inside[j] = i
    return tm, dm, inside


def score_detections(det_uv, truth, peaks_snr, snr_bins=(0, 3, 5, 8, 12, 20, 1e9)):
    """Recall per SNR bin, duplicates (unmatched detections inside a matched object's footprint, per object),
    false positives (detections in no footprint), and centre errors across / along each line."""
    tm, dm, inside = match_truth(det_uv, truth)
    det_uv = np.asarray(det_uv, np.float64).reshape(-1, 2)
    out = {"n_true": int(len(truth)), "n_det": int(len(det_uv)), "recall_bins": []}
    for lo, hi in zip(snr_bins[:-1], snr_bins[1:]):
        s = (peaks_snr >= lo) & (peaks_snr < hi)
        if s.any():
            out["recall_bins"].append({"snr": [lo, hi if hi < 1e8 else None], "n": int(s.sum()),
                                       "recall": float((tm[s] >= 0).mean())})
    s5 = peaks_snr >= 5
    out["recall_snr_ge5"] = float((tm[s5] >= 0).mean()) if s5.any() else None
    out["recall_all"] = float((tm >= 0).mean())
    unmatched = dm < 0
    dup = unmatched & (inside >= 0)
    out["duplicates"] = int(dup.sum())
    out["duplicates_per_object"] = float(dup.sum() / max(len(truth), 1))
    out["objects_with_duplicate"] = float(len(set(inside[dup].tolist())) / max(len(truth), 1))
    out["false_positives"] = int((unmatched & (inside < 0)).sum())
    # centre errors
    a, b = truth[:, 2], truth[:, 3]
    m = tm >= 0
    du = np.full(len(truth), np.nan)
    dv = np.full(len(truth), np.nan)
    du[m] = det_uv[tm[m], 0] - truth[m, 0]
    dv[m] = det_uv[tm[m], 1] - truth[m, 1]
    cls = np.where((a >= 2) & (a >= 2 * b), "H", np.where((b >= 2) & (b >= 2 * a), "V", "other"))
    errs = {}
    for name in ("H", "V", "other"):
        s = m & (cls == name)
        if not s.any():
            continue
        if name == "H":
            along, across = du[s], dv[s]
        elif name == "V":
            along, across = dv[s], du[s]
        else:
            along, across = du[s], dv[s]
        hi = s & (peaks_snr >= 5)
        if name == "H":
            al5, ac5 = du[hi], dv[hi]
        elif name == "V":
            al5, ac5 = dv[hi], du[hi]
        else:
            al5, ac5 = du[hi], dv[hi]
        errs[name] = {"n": int(s.sum()),
                      "along_rms": float(np.sqrt(np.mean(along ** 2))), "along_bias": float(np.mean(along)),
                      "across_rms": float(np.sqrt(np.mean(across ** 2))), "across_bias": float(np.mean(across)),
                      "n_snr_ge5": int(hi.sum()),
                      "along_rms_snr_ge5": float(np.sqrt(np.mean(al5 ** 2))) if hi.any() else None,
                      "across_rms_snr_ge5": float(np.sqrt(np.mean(ac5 ** 2))) if hi.any() else None}
    out["centre_errors_px"] = errs
    return out, tm, dm, inside
