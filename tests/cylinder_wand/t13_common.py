"""T13 (wand balls seen through the tube): the machinery the T13 scripts share.  Plan: 2026-10-02-ball-centres-plan.md.

The wand: two opaque balls, A (1.0 mm) and B (1.5 mm), 10 mm apart centre to centre, joined by a 0.3 mm rod.

Rendering (`render_geometry`, `compose`), for one PINCYL camera and one wand frame:
  - Geometry, exact.  A (sub)pixel shows the object that its ray, refracted into the water, meets first: a ball (a
    sphere) or the rod (a cylinder between the balls' centres).  Pixels whose ray passes within BAND_MM of any outline
    are supersampled SS x SS; the others take their centre ray.
  - Lighting, in 8-bit grey levels:
      backlit    dark balls and rod (30) on a bright background (140): a silhouette;
      even       evenly lit matte balls (130) on a dark background (20).  The dark rod shows only where it hides a
                 ball;
      side       as even, but lit more from one side: 130 (1 + 0.15 n.l), with l a fixed world direction, so the lit
                 side is about 35 % brighter than the other;
      highlight  as even, plus a specular spot of up to +90 (Blinn-Phong, exponent 150) from a fixed light.
  - Blur: an anisotropic Gaussian per ball, sigma_u across the tube axis and sigma_v along it, from the ball's depth.
    It is fitted exactly through the three blur states the plate case measured on the real camera
    (cases/cyl_20260930_T0_cam1/tests/t2_synthetic_dots.py):
      sigma_u, sigma_v = 4.7, 1.5 / 2.8, 2.0 / 3.8, 4.6 px at 0, 22.86, 48.26 mm, the board moving towards the
      camera (the tube separates the horizontal and vertical foci).
    The wand volume's centre sits at `focus_z` of that traverse: 18 mm by default, between the two foci, where the
    image is sharpest overall.  The half of the rod next to a ball is blurred with that ball's blur.
  - Noise: Gaussian, 2 grey levels (as the plate case), added after the blur; the image is rounded to 8 bits.

Detectors (`detect`), each returning [(u, v, size), ...] smallest first (A, then B), with u = column, v = row and pixel
centres at integers:
  pyopenlpt_fast      pyOpenLPT's wand detector as the GUI's "fast" mode runs it (wand_calibrator.run_detection_task,
                      without the file reading): `detect_circles_robust` (Canny edges, a RANSAC circle, least squares),
                      the two best by metric sorted by radius; with fewer than two, the C++ CircleIdentifier.  The
                      functions are imported unchanged.  Radius range 10-60 px, sensitivity 0.85 (the GUI's default).
  pyopenlpt_reliable  the GUI's "reliable" mode: the C++ CircleIdentifier; the two best by metric, sorted by radius.
  blob                Otsu threshold, the two largest connected regions, their area centroids.
  blob_open           the same after an opening of OPEN_PX that removes the rod.
  ellipse             the plate case's estimator (t13_ellipse.detect_balls), after the same opening.
  ellipse_rodmask     the same, leaving out the edge points in the rod's path (a strip of ROD_MASK_PX either side of
                      the line joining the two balls; added after T13b's first look showed the rod pulling the fit).
  ellipse_sym         the same with the strip on the far side of each ball left out too (symmetric point set), and
                      the overlap guard of T13b2 (a pair whose ellipses come within GUARD_PX is dropped); added after
                      T13c.  Not in DETECTOR_NAMES, so T13b and T13c are unchanged.
  ellipse_sym2        ellipse_sym with 192 rays instead of 96 and fits from 25 % of them: the symmetric mask leaves
                      too few of 96 points on the small ball (T13b3).
The C++ CircleIdentifier takes ~6 s on a full 1280 x 800 frame, so it runs on a crop around the wand (`crop`).
"""
import t11_common as tc  # first: one BLAS thread per process

import math  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import rigs  # noqa: E402
from rigs import cm  # noqa: E402
import t13_ellipse as te  # noqa: E402

TUBE = rigs.CASE_TUBE
SENSOR = (1280, 800)                       # cam1's Phantom VEO710
L_WAND = 10.0
RHO = (0.5, 0.75)                          # ball radii, mm: A 1.0 mm, B 1.5 mm diameter
R_ROD = 0.15                               # rod radius, mm (0.3 mm diameter)
HALF = 8.0                                 # wand midpoints within +-8 mm: the common volume cam1's sensor sees
S_U, S_V = 50.5, 38.8                      # image scale near the axis, px per mm (t13_pre_sizing: a 1 mm ball's size)
SS = 8                                     # supersampling of the pixels on an outline
BAND_MM = 0.025                            # ... those whose centre ray passes within this of an outline
VIS_MARGIN_PX = 18.0                       # a ball counts as seen when its image +- this lies on the sensor
LIGHTINGS = ("backlit", "even", "side", "highlight")
POLARITY = {"backlit": "dark", "even": "bright", "side": "bright", "highlight": "bright"}
BG = {"backlit": 140.0, "even": 20.0, "side": 20.0, "highlight": 20.0}
DARK_OBJ = 30.0
BALL = 130.0
SIDE_K = 0.15
SPEC, SPEC_EXP = 90.0, 150.0
NOISE = 2.0
BLUR_STATES = ((0.0, 4.7, 1.5), (22.86, 2.8, 2.0), (48.26, 3.8, 4.6))
FOCUS_Z_MM = 18.0
DEFOCUS_Z_MM = 40.0
MIN_R, MAX_R, SENS = 10.0, 60.0, 0.85
OPEN_PX = 25                               # about 1.7 x the rod's width in the image (0.3 mm ~ 15 px across the axis)
ROD_MASK_PX = 18.0                         # the rod's half-width (~7.6 px) + 2 x its blur (up to ~4.7 px) + margin
GUARD_PX = 3.0                             # T13b2: a pair whose fitted ellipses come within this is dropped
DETECTOR_NAMES = ("pyopenlpt_fast", "pyopenlpt_reliable", "blob", "blob_open", "ellipse", "ellipse_rodmask")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


L_SIDE = _unit((0.6, 0.8, 0.0))
L_SPEC = _unit((0.3, 0.6, 0.75))


def cam1_rig(sensor=SENSOR):
    """Three replicas of cam1 (f 41,459 px, 1094 mm from the axis) at azimuths 0, 45, 90 deg."""
    return rigs.reference_rig(3, 45.0, sensor=sensor)


# ------------------------------------------------------------------ blur
def _quadratic_through(zs, sig):
    A = np.array([[1.0, z, z * z] for z in zs])
    return np.linalg.solve(A, np.asarray(sig, float) ** 2)


_CU = _quadratic_through([s[0] for s in BLUR_STATES], [s[1] for s in BLUR_STATES])
_CV = _quadratic_through([s[0] for s in BLUR_STATES], [s[2] for s in BLUR_STATES])


def blur_sigma(z):
    """(sigma_u, sigma_v) px at position z (mm) of the plate case's traverse: sigma^2 is the quadratic in z through
    its three measured states (a defocus blur growing linearly away from each focus, plus a floor)."""
    su = math.sqrt(max(_CU[0] + _CU[1] * z + _CU[2] * z * z, 0.25))
    sv = math.sqrt(max(_CV[0] + _CV[1] * z + _CV[2] * z * z, 0.25))
    return su, sv


def ball_blur(cam, X, focus_z=FOCUS_Z_MM):
    """The blur of a ball at X for camera `cam`: depth measured towards the camera from the volume's centre."""
    c = cam.C / np.linalg.norm(cam.C)
    return blur_sigma(focus_z + float(np.dot(X, c)))


# ------------------------------------------------------------------ frames
def wand_frames(cams, n, rng, half=HALF):
    """n wand frames (midpoints uniform in +-half, directions uniform) whose two balls are both seen whole, with
    margin, by at least two cameras.  Returns XA (n, 3), XB (n, 3) and uv (n, Nc, 2, 2), the exact images of the
    balls' centres (C++ PINCYL), NaN in a camera that does not see both balls whole."""
    W, H = cams[0].width, cams[0].height
    XA, XB, UV = [], [], []
    while len(XA) < n:
        wd = rigs.wand_frames(cams, TUBE, 2 * (n - len(XA)) + 16, L_WAND, rng, half=(half,) * 3, noise_px=0.0)
        for f in range(len(wd.XA)):
            uv = wd.uv[f].copy()
            for c in range(len(cams)):
                ok = True
                for e, rho in enumerate(RHO):
                    u, v = uv[c, e]
                    au, av = rho * S_U + VIS_MARGIN_PX, rho * S_V + VIS_MARGIN_PX
                    if not (np.isfinite(u) and np.isfinite(v)) or u - au < 0 or u + au > W - 1 or v - av < 0 \
                            or v + av > H - 1:
                        ok = False
                if not ok:
                    uv[c] = np.nan
            if int(np.sum(np.all(np.isfinite(uv[:, :, 0]), axis=1))) >= 2:
                XA.append(wd.XA[f])
                XB.append(wd.XB[f])
                UV.append(uv)
                if len(XA) == n:
                    break
    return np.array(XA), np.array(XB), np.array(UV)


def gap_px(uvA, uvB):
    """Approximate gap (px) between the two balls' images along the line joining their centres (negative: they
    overlap), with the image scale near the axis."""
    d = np.asarray(uvB, float) - np.asarray(uvA, float)
    dist = float(np.hypot(*d))
    phi = math.atan2(d[1], d[0])
    r = [1.0 / math.sqrt((math.cos(phi) / (rho * S_U)) ** 2 + (math.sin(phi) / (rho * S_V)) ** 2) for rho in RHO]
    return dist - r[0] - r[1]


# ------------------------------------------------------------------ rendering
def trace_pixels(cam, pu, pv):
    """Rays in the water of pixel positions (pu, pv): (o, d, ok)."""
    xy = np.column_stack([(pu - cam.cx) / cam.f, (pv - cam.cy) / cam.f])
    rw = cm.rows([cam], np.zeros(len(xy), int))
    o, d, ok, _, _ = cm.trace(rw, xy, TUBE)
    return o, d, ok


def _sphere(o, d, X, rho):
    """(t of the first hit or inf, signed distance of the ray from the sphere's outline, mm)."""
    w = X[None, :] - o
    s = np.einsum("ij,ij->i", w, d)
    q = np.maximum(np.einsum("ij,ij->i", w, w) - s * s, 0.0)
    disc = rho * rho - q
    t = s - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0.0) & (t > 0.0), t, np.inf), np.sqrt(q) - rho


def _rod(o, d, A, B, r):
    """(t of the first hit on the rod's side or inf, axial position of the hit from A, signed distance of the ray from
    the rod's outline, rod length).  The rod is the cylinder of radius r about the segment AB."""
    ab = B - A
    Lr = float(np.linalg.norm(ab))
    a = ab / Lr
    w0 = o - A[None, :]
    da = d @ a
    wa = w0 @ a
    dp = d - da[:, None] * a[None, :]
    wp = w0 - wa[:, None] * a[None, :]
    qa = np.einsum("ij,ij->i", dp, dp)
    qb = 2.0 * np.einsum("ij,ij->i", wp, dp)
    qc = np.einsum("ij,ij->i", wp, wp) - r * r
    disc = qb * qb - 4.0 * qa * qc
    good = (disc >= 0.0) & (qa > 1e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(good, (-qb - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * np.where(qa > 1e-15, qa, 1.0)), np.inf)
    ax = wa + t1 * da
    hit = good & (t1 > 0.0) & (ax >= 0.0) & (ax <= Lr)
    # distance between the ray and the segment
    dd = np.einsum("ij,ij->i", w0, d)
    den = np.maximum(1.0 - da * da, 1e-15)
    s_ = np.clip((wa - da * dd) / den, 0.0, Lr)
    t_ = s_ * da - dd
    sep = (o + t_[:, None] * d) - (A[None, :] + s_[:, None] * a[None, :])
    return np.where(hit, t1, np.inf), ax, np.linalg.norm(sep, axis=1) - r, Lr


def _hits(o, d, XA, XB, r_rod):
    """Labels of the first object hit (0 none, 1 ball A, 2 ball B, 3 rod near A, 4 rod near B), its t, and the signed
    distances from the three outlines."""
    tA, sA = _sphere(o, d, XA, RHO[0])
    tB, sB = _sphere(o, d, XB, RHO[1])
    if r_rod > 0:
        tR, ax, sR, Lr = _rod(o, d, XA, XB, r_rod)
    else:
        tR, ax, sR, Lr = np.full(len(o), np.inf), np.zeros(len(o)), np.full(len(o), np.inf), 1.0
    T = np.stack([tA, tB, tR])
    k = np.argmin(T, axis=0)
    t = T[k, np.arange(len(o))]
    lab = np.where(np.isfinite(t), k + 1, 0)
    lab = np.where((lab == 3) & (ax > 0.5 * Lr), 4, lab)
    return lab, t, (sA, sB, sR)


def render_geometry(cam, XA, XB, uvA, uvB, ss=SS, r_rod=R_ROD, band_mm=BAND_MM):
    """What each sample of one camera's view of one wand frame sees.  Returns a dict: window (u0, v0, w, h); for
    every sample that sees an object: pix (index in the window, row-major), wt (its share of the pixel), lab (as
    `_hits`), nrm (unit surface normal; zero on the rod), view (unit vector back along the ray)."""
    W, H = cam.width, cam.height
    m = VIS_MARGIN_PX + 4.0
    us = [uvA[0] - RHO[0] * S_U, uvA[0] + RHO[0] * S_U, uvB[0] - RHO[1] * S_U, uvB[0] + RHO[1] * S_U]
    vs = [uvA[1] - RHO[0] * S_V, uvA[1] + RHO[0] * S_V, uvB[1] - RHO[1] * S_V, uvB[1] + RHO[1] * S_V]
    u0, u1 = max(int(math.floor(min(us) - m)), 0), min(int(math.ceil(max(us) + m)), W - 1)
    v0, v1 = max(int(math.floor(min(vs) - m)), 0), min(int(math.ceil(max(vs) + m)), H - 1)
    w, h = u1 - u0 + 1, v1 - v0 + 1
    uu, vv = np.meshgrid(np.arange(u0, u1 + 1, dtype=float), np.arange(v0, v1 + 1, dtype=float))
    pu, pv = uu.ravel(), vv.ravel()
    o, d, ok = trace_pixels(cam, pu, pv)
    lab, t, (sA, sB, sR) = _hits(o, d, XA, XB, r_rod)
    lab = np.where(ok, lab, 0)
    band = ok & ((np.abs(sA) < band_mm) | (np.abs(sB) < band_mm) | (np.abs(sR) < band_mm))
    keep = (~band) & (lab > 0)
    parts = [(np.flatnonzero(keep), np.ones(int(keep.sum())), lab[keep], o[keep], d[keep], t[keep])]
    bi = np.flatnonzero(band)
    if len(bi):
        off = (np.arange(ss) + 0.5) / ss - 0.5
        ou, ov = np.meshgrid(off, off)
        su = (pu[bi][:, None] + ou.ravel()[None, :]).ravel()
        sv = (pv[bi][:, None] + ov.ravel()[None, :]).ravel()
        o2, d2, ok2 = trace_pixels(cam, su, sv)
        lab2, t2, _ = _hits(o2, d2, XA, XB, r_rod)
        lab2 = np.where(ok2, lab2, 0)
        k2 = lab2 > 0
        pix2 = np.repeat(bi, ss * ss)
        parts.append((pix2[k2], np.full(int(k2.sum()), 1.0 / (ss * ss)), lab2[k2], o2[k2], d2[k2], t2[k2]))
    pix = np.concatenate([p[0] for p in parts])
    wt = np.concatenate([p[1] for p in parts])
    lb = np.concatenate([p[2] for p in parts])
    oo = np.concatenate([p[3] for p in parts])
    dd = np.concatenate([p[4] for p in parts])
    tt = np.concatenate([p[5] for p in parts])
    P = oo + tt[:, None] * dd
    nrm = np.zeros_like(P)
    for k, (X, rho) in enumerate(((XA, RHO[0]), (XB, RHO[1])), start=1):
        sel = lb == k
        nrm[sel] = (P[sel] - X[None, :]) / rho
    return {"window": (u0, v0, w, h), "pix": pix, "wt": wt, "lab": lb, "nrm": nrm, "view": -dd}


def intensity_delta(geo, lighting):
    """Each sample's grey level minus the background's."""
    lab, n, v = geo["lab"], geo["nrm"], geo["view"]
    bg = BG[lighting]
    if lighting == "backlit":
        return np.full(len(lab), DARK_OBJ - bg)
    if lighting == "even":
        I = np.full(len(lab), BALL)
    elif lighting == "side":
        I = BALL * (1.0 + SIDE_K * (n @ L_SIDE))
    elif lighting == "highlight":
        hv = L_SPEC[None, :] + v
        hv /= np.linalg.norm(hv, axis=1, keepdims=True)
        I = BALL + SPEC * np.maximum(np.einsum("ij,ij->i", n, hv), 0.0) ** SPEC_EXP
    else:
        raise ValueError(lighting)
    return np.where(lab <= 2, I - bg, 0.0)          # the rod is as dark as the background


def compose(cam, geo, lighting, blurA, blurB, rng=None, noise=NOISE, as_float=False):
    """The camera's image: the background, the two blurred layers (ball A with the half of the rod next to it, ball
    B with the other half), and the noise (none when rng is None).  uint8, or float without rounding."""
    u0, v0, w, h = geo["window"]
    dI = intensity_delta(geo, lighting) * geo["wt"]
    layer = np.where((geo["lab"] == 1) | (geo["lab"] == 3), 0, 1)
    acc = np.bincount(layer * (w * h) + geo["pix"], weights=dI, minlength=2 * w * h).reshape(2, h, w)
    out = np.full((cam.height, cam.width), BG[lighting], np.float64)
    for k, (su, sv) in enumerate((blurA, blurB)):
        if su > 0 and sv > 0:
            acc[k] = cv2.GaussianBlur(acc[k], (0, 0), sigmaX=su, sigmaY=sv)
        out[v0:v0 + h, u0:u0 + w] += acc[k]
    if rng is not None and noise > 0:
        out += rng.normal(0.0, noise, out.shape)
    if as_float:
        return out
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def coverage_centroid(geo, label, as_area=False):
    """Area centroid (u, v) of one object's samples in a geometry (the rendered silhouette, before blur)."""
    u0, v0, w, h = geo["window"]
    sel = geo["lab"] == label
    pix, wt = geo["pix"][sel], geo["wt"][sel]
    u = u0 + (pix % w)
    v = v0 + (pix // w)
    A = float(wt.sum())
    c = np.array([float((wt * u).sum()) / A, float((wt * v).sum()) / A])
    return (c, A) if as_area else c


# ------------------------------------------------------------------ detectors
_WC = None


def _wand_calibrator():
    global _WC
    if _WC is None:
        from modules.camera_calibration.wand_calibration import wand_calibrator as wc
        _WC = wc
    return _WC


def _cpp_circles(g, crop=None):
    """The C++ CircleIdentifier on g, or on g[v0:v1, u0:u1] for crop = (u0, v0, u1, v1) (coordinates returned in
    the full image)."""
    import pyopenlpt
    u0 = v0 = 0
    if crop is not None:
        u0, v0, u1, v1 = crop
        g = np.ascontiguousarray(g[v0:v1, u0:u1])
    det = pyopenlpt.CircleIdentifier(pyopenlpt.Image.from_numpy(g))
    centers, radii, metrics = det.BubbleCenterAndSizeByCircle(MIN_R, MAX_R, SENS)
    pts = []
    for i in range(len(centers)):
        m = metrics[i] if i < len(metrics) else 0.0
        pts.append((float(centers[i][0]) + u0, float(centers[i][1]) + v0, float(radii[i]), float(m)))
    return pts


def crop_around(geo, cam, margin=100):
    """A crop (u0, v0, u1, v1) around a rendered wand's window, for the C++ CircleIdentifier."""
    u0, v0, w, h = geo["window"]
    return (max(u0 - margin, 0), max(v0 - margin, 0), min(u0 + w + margin, cam.width),
            min(v0 + h + margin, cam.height))


def detect(name, img, polarity, crop=None):
    """[(u, v, size), ...] smallest first, as the module docstring describes.  crop: the region the C++
    CircleIdentifier is run on (None: the whole image)."""
    if name in ("pyopenlpt_fast", "pyopenlpt_reliable"):
        g = img if polarity == "bright" else cv2.bitwise_not(img)
        if name == "pyopenlpt_fast":
            pts = _wand_calibrator().detect_circles_robust(g, MIN_R, MAX_R)
            if len(pts) >= 2:
                return sorted([(float(p[0]), float(p[1]), float(p[2])) for p in pts[:2]], key=lambda p: p[2])
        pts = sorted(_cpp_circles(g, crop), key=lambda p: p[3], reverse=True)[:2]
        return sorted([p[:3] for p in pts], key=lambda p: p[2])
    if name in ("blob", "blob_open"):
        g = img if polarity == "bright" else cv2.bitwise_not(img)
        _t, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if name == "blob_open":
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_PX, OPEN_PX)))
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(b, connectivity=8)
        order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:2]
        out = [(float(cents[i][0]), float(cents[i][1]), float(stats[i, cv2.CC_STAT_AREA])) for i in order
               if stats[i, cv2.CC_STAT_AREA] >= 50]
        return sorted(out, key=lambda p: p[2])
    if name == "ellipse":
        return [(r[0], r[1], r[2]) for r in te.detect_balls(img, polarity, OPEN_PX)]
    if name == "ellipse_rodmask":
        return [(r[0], r[1], r[2]) for r in te.detect_balls(img, polarity, OPEN_PX, rod_mask_px=ROD_MASK_PX)]
    if name in ("ellipse_sym", "ellipse_sym2"):
        if name == "ellipse_sym":
            r = te.detect_balls(img, polarity, OPEN_PX, rod_mask_px=ROD_MASK_PX, symmetric=True)
        else:
            r = te.detect_balls(img, polarity, OPEN_PX, det=dict(te.DET, rays=192), rod_mask_px=ROD_MASK_PX,
                                symmetric=True, min_keep_frac=0.25)
        if len(r) >= 2 and ellipse_gap(r[0], r[1]) < GUARD_PX:
            return []
        return [(x[0], x[1], x[2]) for x in r]
    raise ValueError(name)


def ellipse_gap(e1, e2):
    """Gap (px) between two fitted ellipses (u, v, area, a, b, theta, ...) along the line joining their centres."""
    du, dv = e2[0] - e1[0], e2[1] - e1[1]
    phi = math.atan2(dv, du)

    def rad(e, ang):
        return 1.0 / math.sqrt((math.cos(ang - e[5]) / e[3]) ** 2 + (math.sin(ang - e[5]) / e[4]) ** 2)
    return math.hypot(du, dv) - rad(e1, phi) - rad(e2, phi + math.pi)


def assign(dets, tA, tB, gate=5.0):
    """Compare detections (smallest first: A, B) with the true centres.  Returns (status, eA, eB): status "ok" (both
    within `gate` px with the right labels), "swapped", "wrong" or "missing"; eA, eB the errors (px) when ok."""
    nan2 = np.array([np.nan, np.nan])
    if len(dets) < 2:
        return "missing", nan2, nan2
    a, b = np.array(dets[0][:2]), np.array(dets[1][:2])
    eA, eB = a - tA, b - tB
    if np.hypot(*eA) < gate and np.hypot(*eB) < gate:
        return "ok", eA, eB
    if np.hypot(*(a - tB)) < gate and np.hypot(*(b - tA)) < gate:
        return "swapped", nan2, nan2
    return "wrong", nan2, nan2


__all__ = ["tc", "rigs", "cm"]
