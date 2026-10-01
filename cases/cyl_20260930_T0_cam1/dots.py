"""Sub-pixel detection of the dark dots of the calibration board.

The dots are engraved-looking: a darker rim, horizontal streaks inside, and edges blurred more horizontally than
vertically (the tube's astigmatism).  An intensity-weighted centroid would follow the rim and the streaks, so all
three estimators here work from the dot's *outline* instead:

* ``u, v``          (primary) an ellipse fitted to edge points found along rays from the centre, each at half the
                    *local* edge height.  A blurred step is crossed at half its height exactly at the step, whatever
                    its height, so a rim darker on one side than the other does not move these points much.
* ``u_iso, v_iso``  centroid of the region inside one iso-contour of the darkness (half the dot's core darkness).
                    Simple, but a rim darker on one side pushes the contour outwards on that side.
* ``u_sat, v_sat``  centroid of the darkness clipped at the core level (a soft area centroid).

Image coordinates: u = column, v = row, pixel centres at integers (as OpenCV and the pyOpenLPT GUI).

The darkness N = (B - I) / B is normalised by the local background B, which is estimated from the board alone
(dots masked out), so uneven lighting does not shift the edges.
"""
from __future__ import annotations

import math

import contourpy
import cv2
import numpy as np


# ------------------------------------------------------------------ background
def normalised_darkness(img: np.ndarray, det: dict):
    """Return (N, background, seg_level).  N is 0 on the board and ~0.3 inside a dot."""
    k = int(det["background_close_px"]) | 1
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg0 = cv2.GaussianBlur(closed, (0, 0), 15.0)
    n0 = (bg0 - img) / np.maximum(bg0, 1.0)
    core0 = float(np.median(n0[n0 > 0.12])) if np.any(n0 > 0.12) else 0.3
    # Mask everything dark (dots and specks), widened over the blurred edges, then interpolate the background
    # from the board pixels alone by normalised convolution.
    mask = (n0 > 0.25 * core0).astype(np.uint8)
    d = int(det["mask_dilate_px"])
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    w = (1 - mask).astype(np.float32)
    s = float(det["background_sigma_px"])
    num = cv2.GaussianBlur(img * w, (0, 0), s)
    den = cv2.GaussianBlur(w, (0, 0), s)
    bg = np.where(den > 0.05, num / np.maximum(den, 1e-6), bg0).astype(np.float32)
    N = (bg - img) / np.maximum(bg, 1.0)
    core = float(np.median(N[N > 0.12])) if np.any(N > 0.12) else core0
    return N.astype(np.float32), bg, float(det["seg_fraction"]) * core


# ------------------------------------------------------------------ geometry helpers
def fit_ellipse(x: np.ndarray, y: np.ndarray):
    """Direct least-squares ellipse (Halir & Flusser 1998).  Returns (xc, yc, a, b, theta) or None.

    a >= b are semi-axes; theta is the angle of the a-axis from +u, in radians.
    """
    if len(x) < 6:
        return None
    mx, my = float(np.mean(x)), float(np.mean(y))
    s = float(np.std(x) + np.std(y)) or 1.0
    xn, yn = (x - mx) / s, (y - my) / s
    D1 = np.column_stack([xn * xn, xn * yn, yn * yn])
    D2 = np.column_stack([xn, yn, np.ones_like(xn)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])
    vals, vecs = np.linalg.eig(M)
    vecs = np.real(vecs)
    cond = 4 * vecs[0] * vecs[2] - vecs[1] ** 2
    ok = np.where(cond > 0)[0]
    if len(ok) == 0:
        return None
    a1 = vecs[:, ok[0]]
    A, B, C = a1
    Dc, Ec, Fc = T @ a1
    den = B * B - 4 * A * C
    if den >= 0:
        return None
    xc = (2 * C * Dc - B * Ec) / den
    yc = (2 * A * Ec - B * Dc) / den
    F0 = A * xc * xc + B * xc * yc + C * yc * yc + Dc * xc + Ec * yc + Fc
    Q = np.array([[A, B / 2], [B / 2, C]])
    lam, vec = np.linalg.eigh(Q)
    with np.errstate(invalid="ignore"):
        ax = np.sqrt(-F0 / lam)
    if not np.all(np.isfinite(ax)):
        return None
    i_big = int(np.argmax(ax))
    a_ax, b_ax = float(ax[i_big]), float(ax[1 - i_big])
    th = math.atan2(vec[1, i_big], vec[0, i_big])
    return (xc * s + mx, yc * s + my, a_ax * s, b_ax * s, th)


def ellipse_radius(a: float, b: float, phi: np.ndarray) -> np.ndarray:
    """Radius of an ellipse (semi-axes a, b, axis-aligned in its own frame) in the direction phi."""
    return 1.0 / np.sqrt((np.cos(phi) / a) ** 2 + (np.sin(phi) / b) ** 2)


def polygon_centroid(xy: np.ndarray):
    """(area, cx, cy) of a closed polygon by the shoelace formula; area is positive."""
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = 0.5 * float(np.sum(cr))
    if abs(A) < 1e-12:
        return 0.0, float("nan"), float("nan")
    cx = float(np.sum((x + x1) * cr)) / (6 * A)
    cy = float(np.sum((y + y1) * cr)) / (6 * A)
    return abs(A), cx, cy


# ------------------------------------------------------------------ estimators
def radial_edges(Ns: np.ndarray, c, rx: float, ry: float, det: dict, extra: float = 14.0, mode: str = "halfmax"):
    """Edge points along rays from c, each where the darkness falls to edge_fraction of its local peak
    (mode "halfmax"), or where it falls most steeply (mode "gradient", a check that a dark rim affects less).

    Returns (points (K,2), ray angles (K,), complete) where complete is False if a ray left the image before its
    edge and 4 px past it were found (the dot is cut by the border).
    """
    rays = int(det["rays"])
    frac = float(det["edge_fraction"])
    th = np.arange(rays) * (2 * np.pi / rays)
    re = ellipse_radius(rx, ry, th)
    S = 240
    t = np.linspace(0.0, 1.0, S)
    r = (0.4 * re)[:, None] + (0.6 * re + extra)[:, None] * t[None, :]
    X = (c[0] + r * np.cos(th)[:, None]).astype(np.float32)
    Y = (c[1] + r * np.sin(th)[:, None]).astype(np.float32)
    H, W = Ns.shape
    inside = (X >= 0) & (Y >= 0) & (X <= W - 1) & (Y <= H - 1)
    P = cv2.remap(Ns, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-9.0)
    pts, angs = [], []
    complete = True
    for m in range(rays):
        # samples are usable up to the first one off the sensor
        n_in = int(np.argmin(inside[m])) if not inside[m].all() else S
        p = P[m, :n_in]
        if n_in < S // 3:
            complete = False
            continue
        reach = int(np.searchsorted(r[m], re[m] + 3.0))
        k_pk = int(np.argmax(p[:max(reach, 2)]))
        p_pk = float(p[k_pk])
        if p_pk <= 0.05:
            continue
        lev = frac * p_pk
        below = np.where(p[k_pk:] < lev)[0]
        n4 = int(math.ceil(4.0 / max(r[m, 1] - r[m, 0], 1e-6)))
        if len(below) == 0 or k_pk + int(below[0]) + n4 > n_in:
            if n_in < S:
                complete = False                         # the border cuts the ray before edge + 4 px
            continue
        k = k_pk + int(below[0])
        # the profile must stay low for 4 px past the edge (no speck or neighbour right behind it)
        tail = p[k:k + n4]
        if len(tail) and float(np.max(tail)) > 0.8 * p_pk:
            continue
        if mode == "gradient":
            # steepest descent of the darkness between the peak and 4 px past the half-height crossing
            g = np.gradient(p[:k + n4])
            lo = k_pk
            j = lo + int(np.argmin(g[lo:k + n4]))
            if j <= lo or j >= k + n4 - 1:
                continue
            g0, g1, g2 = float(g[j - 1]), float(g[j]), float(g[j + 1])
            den = g0 - 2 * g1 + g2
            off = 0.5 * (g0 - g2) / den if den > 0 else 0.0
            off = min(max(off, -0.5), 0.5)
            rr = r[m, j] + off * (r[m, 1] - r[m, 0])
        else:
            p0, p1 = float(p[k - 1]), float(p[k])
            f = (p0 - lev) / (p0 - p1) if p0 != p1 else 0.5
            rr = r[m, k - 1] + f * (r[m, k] - r[m, k - 1])
        pts.append((c[0] + rr * math.cos(th[m]), c[1] + rr * math.sin(th[m])))
        angs.append(th[m])
    return np.asarray(pts, float).reshape(-1, 2), np.asarray(angs, float), complete


def robust_ellipse(pts: np.ndarray, min_keep: int, rounds: int = 3, k_mad: float = 3.5):
    """Ellipse fit with points beyond k_mad robust sigmas (radial residual) dropped.  Returns (ell, rms, n, keep)."""
    keep = np.ones(len(pts), bool)
    ell = None
    for _ in range(rounds):
        if keep.sum() < min_keep:
            return None, float("nan"), int(keep.sum()), keep
        ell = fit_ellipse(pts[keep, 0], pts[keep, 1])
        if ell is None:
            return None, float("nan"), int(keep.sum()), keep
        res = ellipse_residual(ell, pts)
        sig = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))) + 1e-3
        new = np.abs(res) < k_mad * sig + 0.05
        if np.array_equal(new, keep):
            break
        keep = new
    res = ellipse_residual(ell, pts)
    return ell, float(np.sqrt(np.mean(res[keep] ** 2))), int(keep.sum()), keep


def ellipse_residual(ell, pts: np.ndarray) -> np.ndarray:
    """Radial distance of each point from the ellipse, measured from its centre (positive outside)."""
    xc, yc, a, b, th = ell
    dx, dy = pts[:, 0] - xc, pts[:, 1] - yc
    c, s = math.cos(th), math.sin(th)
    xe, ye = c * dx + s * dy, -s * dx + c * dy
    rho = np.hypot(xe, ye)
    return rho - ellipse_radius(a, b, np.arctan2(ye, xe))


def iso_centroid(N: np.ndarray, window, level: float, c):
    """Centroid of the region enclosed by the iso-contour N = level that contains c (largest such loop)."""
    x0, y0, x1, y1 = window
    sub = N[y0:y1, x0:x1].astype(np.float64)
    gen = contourpy.contour_generator(x=np.arange(x0, x1, dtype=float), y=np.arange(y0, y1, dtype=float),
                                      z=sub, line_type=contourpy.LineType.Separate)
    best = (0.0, float("nan"), float("nan"))
    for ln in gen.lines(level):
        if len(ln) < 8 or np.hypot(*(ln[0] - ln[-1])) > 1e-6:
            continue                                    # open line: cut by the window
        A, cx, cy = polygon_centroid(ln[:-1])
        inside = cv2.pointPolygonTest(ln.astype(np.float32).reshape(-1, 1, 2), (float(c[0]), float(c[1])), False)
        if A > best[0] and inside > 0:
            best = (A, cx, cy)
    return best


def saturated_centroid(N: np.ndarray, window, core: float, region: np.ndarray):
    """Centroid of clip(N / core, 0, 1) over `region` (a boolean mask of the window)."""
    x0, y0, x1, y1 = window
    w = np.clip(N[y0:y1, x0:x1] / core, 0.0, 1.0) * region
    sw = float(w.sum())
    if sw <= 0:
        return float("nan"), float("nan")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((w * xx).sum() / sw), float((w * yy).sum() / sw)


# ------------------------------------------------------------------ the detector
def detect(img: np.ndarray, det: dict) -> dict:
    """Detect every dot of one image.

    Returns {"dots": [dict, ...], "N": darkness, "seg_level": float, "a_ref": reference area}.  Each dot dict has
    kind in {"dot", "border", "irregular", "big"}; only kind == "dot" with ok True is meant for calibration.
    """
    H, W = img.shape
    N, bg, seg = normalised_darkness(img, det)
    Ns = cv2.GaussianBlur(N, (0, 0), float(det["smooth_sigma_px"]))
    m = (N > seg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    if len(areas) == 0:
        return {"dots": [], "N": N, "seg_level": seg, "a_ref": float("nan")}
    margin = int(det["border_margin_px"])

    def touches(i):
        x, y, w, h = stats[i, :4]
        return x <= margin or y <= margin or x + w >= W - margin or y + h >= H - margin

    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.2 * areas.max() and not touches(i)]
    a_ref = float(np.median([stats[i, cv2.CC_STAT_AREA] for i in big])) if big else float(areas.max())

    dots = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < float(det["min_area_frac"]) * a_ref * 0.3:
            continue                                     # speck: not reported
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hull_area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) if cnts else 0.0
        solidity = area / hull_area if hull_area > 0 else 0.0
        d = {"u_rough": float(cents[i][0]), "v_rough": float(cents[i][1]), "area": area,
             "bbox": [x, y, w, h], "solidity": solidity, "kind": "dot", "ok": False}
        if touches(i):
            d["kind"] = "border"
        elif area < float(det["min_area_frac"]) * a_ref:
            d["kind"] = "small"
        elif area > float(det["max_area_frac"]) * a_ref:
            d["kind"] = "big"
        elif solidity < float(det["min_solidity"]):
            d["kind"] = "irregular"
        for key in ("u", "v", "u_grad", "v_grad", "u_iso", "v_iso", "u_sat", "v_sat", "a", "b", "theta",
                    "edge_rms", "core"):
            d[key] = float("nan")
        d["n_edge"] = 0
        if d["kind"] != "dot":
            dots.append(d)
            continue

        # core darkness of this dot (its own level, not the image's)
        er = cv2.erode(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        vals = N[y:y + h, x:x + w][er > 0]
        core = float(np.median(vals)) if len(vals) > 20 else float(np.median(N[y:y + h, x:x + w][comp > 0]))
        d["core"] = core

        # primary: radial edges + ellipse, twice (re-centred)
        c = (float(cents[i][0]), float(cents[i][1]))
        rx, ry = 0.5 * w, 0.5 * h
        ell, rms, n_e, complete = None, float("nan"), 0, True
        for _ in range(2):
            pts, _angs, complete = radial_edges(Ns, c, rx, ry, det)
            ell, rms, n_e, _keep = robust_ellipse(pts, min_keep=int(0.6 * int(det["rays"])))
            if ell is None:
                break
            c = (ell[0], ell[1])
            # axis-aligned extents of the fitted ellipse, for the next round's search ranges
            ct, st = math.cos(ell[4]), math.sin(ell[4])
            rx = math.sqrt((ell[2] * ct) ** 2 + (ell[3] * st) ** 2)
            ry = math.sqrt((ell[2] * st) ** 2 + (ell[3] * ct) ** 2)
        if not complete:
            d["kind"] = "border"
        if ell is not None:
            d.update(u=ell[0], v=ell[1], a=ell[2], b=ell[3], theta=ell[4], edge_rms=rms)
            d["n_edge"] = n_e
            # check: the same, with each edge point at the steepest descent instead of half height
            pts_g, _a, _c = radial_edges(Ns, c, rx, ry, det, mode="gradient")
            ell_g, _r, _n, _k = robust_ellipse(pts_g, min_keep=int(0.6 * int(det["rays"])))
            if ell_g is not None:
                d["u_grad"], d["v_grad"] = ell_g[0], ell_g[1]

        # secondary estimators over a window around the dot
        pad = 18
        win = (max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H))
        A_iso, cu, cv_ = iso_centroid(N, win, 0.5 * core, c)
        d["u_iso"], d["v_iso"] = cu, cv_
        region = np.zeros((win[3] - win[1], win[2] - win[0]), np.uint8)
        region[y - win[1]:y - win[1] + h, x - win[0]:x - win[0] + w] = comp
        region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(np.float32)
        d["u_sat"], d["v_sat"] = saturated_centroid(N, win, core, region)
        d["ok"] = bool(d["kind"] == "dot" and ell is not None and np.isfinite(d["u"]))
        dots.append(d)
    return {"dots": dots, "N": N, "seg_level": seg, "a_ref": a_ref}
