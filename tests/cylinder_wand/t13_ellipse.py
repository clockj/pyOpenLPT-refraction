"""T13: the plate case's centre estimator, applied to wand balls.  Plan: 2026-10-02-ball-centres-plan.md.

The six functions between the two marker lines are copied verbatim from cases/cyl_20260930_T0_cam1/dots.py
(2026-10-02; the case folder is not edited).  That estimator found the centres of the board's dots through the same
tube: an ellipse fitted to edge points found along 96 rays from the centre, each where the darkness falls to half its
local peak, with outlying points dropped.  `DET` holds the case's settings (its config.json, "detect").

New here: `detect_balls`, the same estimator for the balls of a wand.  The blobs are separated first (an opening of
`open_px` removes the rod that joins them), and the largest `n_balls` are measured.  With `rod_mask_px` > 0 the edge
points in the rod's path are left out: those on the side facing the other ball and within rod_mask_px of the line
joining the two blobs' centres (the rod's blurred image pulls the half-height crossings there outwards).  With
`symmetric` the strip on the opposite side is left out too, so the points kept are symmetric about the centre: the
half-height outline of a blurred ellipse is shifted inwards by an amount that differs with the angle, and a lopsided
set of points turns that into a shift of the fitted centre.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DET = {"background_close_px": 101, "background_sigma_px": 25.0, "mask_dilate_px": 9, "seg_fraction": 0.5,
       "min_area_frac": 0.35, "max_area_frac": 2.0, "min_solidity": 0.85, "border_margin_px": 3, "rays": 96,
       "ray_step_px": 0.25, "edge_fraction": 0.5, "smooth_sigma_px": 1.0}


# ======================== copied verbatim from cases/cyl_20260930_T0_cam1/dots.py ========================
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

# ======================== end of the copy ========================


def detect_balls(img: np.ndarray, polarity: str, open_px: int, det: dict = None, n_balls: int = 2,
                 rod_mask_px: float = 0.0, symmetric: bool = False, min_keep_frac: float = None):
    """Centres of the wand's balls in one image, by the plate case's estimator.

    img: uint8 image.  polarity: "dark" (dark balls on a bright background, e.g. backlit) or "bright".
    open_px: diameter (px) of an opening that removes the rod before the blobs are separated (0: none).
    rod_mask_px: half-width (px) of the strip along the rod whose edge points are left out (0: none).
    symmetric: leave out the strip on the far side of the ball as well.
    min_keep_frac: the share of the rays a fit needs (default: the case's 0.6, or 0.4 with the rod mask).
    Returns [(u, v, area_px, a, b, theta, n_edge, edge_rms_px), ...] for the n_balls largest blobs whose outline was
    measured, smallest area first.  u = column, v = row, pixel centres at integers."""
    det = DET if det is None else det
    g = img.astype(np.float32)
    if polarity == "bright":
        g = 255.0 - g
    N, _bg, seg = normalised_darkness(g, det)
    Ns = cv2.GaussianBlur(N, (0, 0), float(det["smooth_sigma_px"]))
    m = (N > seg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if open_px:
        k = int(open_px) | 1
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:n_balls]
    out = []
    for i in order:
        _x, _y, w, h, _area = (int(v) for v in stats[i])
        c = (float(cents[i][0]), float(cents[i][1]))
        others = [np.array(cents[j], float) for j in order if j != i]
        rx, ry = 0.5 * w, 0.5 * h
        ell, rms, n_e, complete = None, float("nan"), 0, True
        for _ in range(2):                       # as the case: fit, re-centre, fit again
            pts, _angs, complete = radial_edges(Ns, c, rx, ry, det)
            if rod_mask_px > 0 and others and len(pts):
                q = np.asarray(c, float)
                for o in others:                 # drop the points in the strip towards each other ball
                    e = (o - q) / max(float(np.linalg.norm(o - q)), 1e-9)
                    rel = pts - q[None, :]
                    along = rel @ e
                    perp = np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0])
                    drop = (perp < rod_mask_px) & ((along > 0) | symmetric)
                    pts = pts[~drop]
            # the case's minimum (60 % of the rays); 40 % when the rod's strip is left out
            frac = min_keep_frac if min_keep_frac is not None else (0.4 if rod_mask_px > 0 else 0.6)
            ell, rms, n_e, _keep = robust_ellipse(pts, min_keep=int(frac * int(det["rays"])))
            if ell is None:
                break
            c = (ell[0], ell[1])
            ct, st = math.cos(ell[4]), math.sin(ell[4])
            rx = math.sqrt((ell[2] * ct) ** 2 + (ell[3] * st) ** 2)
            ry = math.sqrt((ell[2] * st) ** 2 + (ell[3] * ct) ** 2)
        if ell is None or not complete:
            continue
        out.append((ell[0], ell[1], math.pi * ell[2] * ell[3], ell[2], ell[3], ell[4], n_e, rms))
    return sorted(out, key=lambda r: r[2])
