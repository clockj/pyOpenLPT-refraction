"""The dot board of the Planes recording (run 20261007/Planes; plan: 2026-10-07-plan.md, P0e and Phase C).

The board of 30 Sep (3 mm pitch, 1.5 mm dark dots) at three stage positions `0`, `1`, `2` (README.txt: "Far from
cam1"; the user: about one full turn of the stage's screw, 0.125 in = 3.175 mm, between positions), one frame per
camera each.  cam1 sees the board face-on, cam2 at a steep angle.

The dots are found by the 30 Sep case's detector (cases/cyl_20260930_T0_cam1/dots.py, loaded from its file and used
unchanged): an ellipse fitted to the points where the darkness, normalised by the local background, falls to half
its local height along rays from the dot's centre.  Its settings are config.planes.detect.

Image coordinates: u = column, v = row, pixel centres at integers.
"""
from __future__ import annotations

import importlib.util
import math
import os

import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

import case

HERE = case.HERE
SEP30 = os.path.join(os.path.dirname(HERE), "cyl_20260930_T0_cam1")
_D = None
_G = None


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SEP30, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dots_module():
    """cases/cyl_20260930_T0_cam1/dots.py, loaded from its file (unchanged)."""
    global _D
    if _D is None:
        _D = _load("dots_sep30", "dots.py")
    return _D


def grid_module():
    """cases/cyl_20260930_T0_cam1/grid_index.py, loaded from its file (unchanged)."""
    global _G
    if _G is None:
        _G = _load("grid_index_sep30", "grid_index.py")
    return _G


def image_path(cfg: dict, pos: str, cam_id: int) -> str:
    folder = {int(c["id"]): c["folder"] for c in cfg["cameras"]}[int(cam_id)]
    return os.path.join(cfg["planes"]["raw_dir"], str(pos), folder, cfg["planes"]["file"])


def read(cfg: dict, pos: str, cam_id: int) -> np.ndarray:
    return case.read_image(image_path(cfg, pos, cam_id)).astype(np.float32)


def mask_border_blobs(img: np.ndarray, det: dict, factor: float = 4.0):
    """The image with every dark region that touches the image's border and is larger than `factor` times the
    median dark region filled with the background estimate, and the mask filled (bool).

    In cam2 the board's edge shows as a dark band down the left of the image.  dots.detect takes its reference dot
    area from the large regions that do not touch the border, falling back to the largest region; with the band
    in view no dot qualifies, the band becomes the reference and every dot is dropped as a speck."""
    D = dots_module()
    N, bg, seg = D.normalised_darkness(img, det)
    m = (N > seg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = img.shape
    if n <= 2:
        return img, np.zeros(img.shape, bool)
    areas = st[1:, cv2.CC_STAT_AREA].astype(float)
    med = float(np.median(areas[areas > 50])) if np.any(areas > 50) else float(np.median(areas))
    mask = np.zeros(img.shape, np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        touches = x <= 0 or y <= 0 or x + w >= W or y + h >= H
        if touches and a > factor * med:
            mask[lab == i] = 1
    if not mask.any():
        return img, mask.astype(bool)
    # widened over the region's blurred edge, and filled with the board's median brightness (not the background
    # estimate, which the region itself darkens next to it)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))).astype(bool)
    board = (~mask) & (N < 0.5 * seg)
    out = img.copy()
    out[mask] = float(np.median(img[board])) if board.any() else float(np.median(img))
    return out, mask


def detect(img: np.ndarray, det: dict) -> dict:
    """dots.detect with the given settings, after mask_border_blobs: {"dots": [...], "N": darkness, "seg_level",
    "a_ref", "masked": bool image}."""
    img2, mask = mask_border_blobs(img, det)
    r = dots_module().detect(img2, det)
    r["masked"] = mask
    return r


# ------------------------------------------------------------------ indexing
def _homography(ij, P):
    Hm, _ = cv2.findHomography(np.asarray(ij, np.float64), np.asarray(P, np.float64), 0)
    return Hm


def _h_apply(Hm, ij):
    q = np.column_stack([np.asarray(ij, float), np.ones(len(ij))]) @ Hm.T
    return q[:, :2] / q[:, 2:3]


def _h_inverse(Hm, P):
    q = np.column_stack([np.asarray(P, float), np.ones(len(P))]) @ np.linalg.inv(Hm).T
    return q[:, :2] / q[:, 2:3]


def index_dots(P: np.ndarray, seed_xy=None, iters: int = 12):
    """Lattice indices (i, jd) of dot centres P (N, 2) on one image: i grows along the rows (image right), jd down
    the columns (image down), (0, 0) at the dot nearest seed_xy (default: the image centre of the dots).

    Starts from grid_index's median nearest-neighbour steps and its affine indexing of the dots within 2.5 pitches
    of the seed, then grows outwards with homography refits (the oblique view of cam2 is far from affine) until the
    assignment no longer changes.  Returns (ij (N, 2) int, homography ij -> P, residual per dot (px))."""
    G = grid_module()
    P = np.asarray(P, float)
    a, b = G.lattice_vectors(P)
    if seed_xy is None:
        seed_xy = P.mean(axis=0)
    s = int(np.argmin(np.hypot(P[:, 0] - seed_xy[0], P[:, 1] - seed_xy[1])))
    A = np.column_stack([a, b])
    ij = np.rint(np.linalg.solve(A, (P - P[s]).T).T)
    pitch = min(np.linalg.norm(a), np.linalg.norm(b))
    use = np.hypot(*(P - P[s]).T) < 2.5 * pitch
    Hm = None
    for radius in (2.5, 4.0, 6.0, 9.0, 14.0, 99.0):
        use = np.hypot(*(P - P[s]).T) < radius * max(np.linalg.norm(a), np.linalg.norm(b))
        for _ in range(iters):
            if use.sum() < 6:
                break
            Hm = _homography(ij[use], P[use])
            new = np.rint(_h_inverse(Hm, P))
            if np.array_equal(new[use], ij[use]) and np.array_equal(new, ij):
                break
            ij = new
    Hm = _homography(ij, P)
    res = np.hypot(*(_h_apply(Hm, ij) - P).T)
    ij = ij.astype(int) - ij[s].astype(int)
    Hm = _homography(ij, P)
    keys = [tuple(v) for v in ij]
    if len(keys) != len(set(keys)):
        raise RuntimeError("index_dots: duplicate indices")
    return ij, Hm, res


DIHEDRAL = [np.array(m) for m in ([[1, 0], [0, 1]], [[-1, 0], [0, 1]], [[1, 0], [0, -1]], [[-1, 0], [0, -1]],
                                  [[0, 1], [1, 0]], [[0, -1], [1, 0]], [[0, 1], [-1, 0]], [[0, -1], [-1, 0]])]


def skew_distances(o1, d1, o2, d2) -> np.ndarray:
    """Distance between every line (o1[a], d1[a]) and every line (o2[b], d2[b]): (n1, n2)."""
    n = np.cross(d1[:, None, :], d2[None, :, :])
    nn = np.linalg.norm(n, axis=2)
    w = o2[None, :, :] - o1[:, None, :]
    return np.abs(np.einsum("abk,abk->ab", w, n)) / np.maximum(nn, 1e-300)


def correspondences(ij1, ij2, dist, min_pairs: int = 12, t_range: int = 40):
    """Score every lattice map ij2 = M ij1 + t (M one of the 8 symmetries of the square lattice, t an integer
    offset) by the median distance between the lines of sight of the pairs it implies (dist: skew_distances).
    Returns the candidates sorted by score, each {"M", "t", "pairs" [(a, b)], "median_mm", "n"}."""
    key2 = {tuple(v): b for b, v in enumerate(ij2)}
    out = []
    for M in DIHEDRAL:
        m1 = ij1 @ M.T
        for tx in range(-t_range, t_range + 1):
            for ty in range(-t_range, t_range + 1):
                pairs = [(a, key2[(int(p[0] + tx), int(p[1] + ty))]) for a, p in enumerate(m1)
                         if (int(p[0] + tx), int(p[1] + ty)) in key2]
                if len(pairs) < min_pairs:
                    continue
                d = np.array([dist[a, b] for a, b in pairs])
                out.append({"M": M.tolist(), "t": [tx, ty], "pairs": pairs, "median_mm": float(np.median(d)),
                            "n": len(pairs)})
    out.sort(key=lambda c: c["median_mm"])
    return out


# ------------------------------------------------------------------ edge blur
def _edge_model(r, B, A, r0, s):
    return B + A * 0.5 * erfc((r - r0) / (math.sqrt(2.0) * s))


def edge_sigma(N: np.ndarray, dot: dict, phi: float, half_fan_deg: float = 9.0, n_rays: int = 7):
    """The blur of the dot's outline in image direction phi (radians from +u, +v down): the median over a fan of
    rays of the sigma of an erf step fitted to the darkness profile across the outline.  Returns (sigma, n_fits)
    or (nan, 0).

    Each ray starts at the dot's fitted centre; the outline radius rho_e in that direction is the fitted
    half-height ellipse's.  The step model is B + A/2 erfc((r - r0) / (sqrt(2) sigma)), fitted on
    [rho_e - w_in, rho_e + w_out], the window widened after a first fit for wide edges."""
    D = dots_module()
    u, v, a, b, th = dot["u"], dot["v"], dot["a"], dot["b"], dot["theta"]
    H, W = N.shape
    out = []
    for dphi in np.radians(np.linspace(-half_fan_deg, half_fan_deg, n_rays)):
        ang = phi + dphi
        rho_e = float(D.ellipse_radius(a, b, np.array([ang - th]))[0])
        sig = 2.0
        ok = False
        for _pass in range(2):
            w_in = min(0.7 * rho_e, 3.0 * sig + 4.0)
            w_out = 3.0 * sig + 8.0
            r = np.arange(rho_e - w_in, rho_e + w_out, 0.25)
            X = (u + r * math.cos(ang)).astype(np.float32)
            Y = (v + r * math.sin(ang)).astype(np.float32)
            if X.min() < 0 or Y.min() < 0 or X.max() > W - 1 or Y.max() > H - 1:
                break
            p = cv2.remap(N, X.reshape(1, -1), Y.reshape(1, -1), cv2.INTER_LINEAR)[0].astype(float)
            A0 = max(float(np.max(p[: max(3, len(p) // 3)])) - float(np.median(p[-8:])), 1e-3)
            try:
                popt, _ = curve_fit(_edge_model, r, p, p0=[float(np.median(p[-8:])), A0, rho_e, sig],
                                    bounds=([-1.0, 0.0, rho_e - w_in, 0.15], [1.0, 2.0, rho_e + w_out, 25.0]),
                                    maxfev=400)
            except (RuntimeError, ValueError):
                break
            sig = float(popt[3])
            ok = popt[1] > 0.03
        if ok:
            out.append(sig)
    if not out:
        return float("nan"), 0
    return float(np.median(out)), len(out)


def edge_blur(N: np.ndarray, dot: dict) -> dict:
    """sigma_h (left and right edges: blur along the rows) and sigma_v (top and bottom: along the columns), px."""
    sl, nl = edge_sigma(N, dot, math.pi)
    sr, nr = edge_sigma(N, dot, 0.0)
    su, nu = edge_sigma(N, dot, -0.5 * math.pi)
    sd, nd = edge_sigma(N, dot, 0.5 * math.pi)
    h = [s for s in (sl, sr) if np.isfinite(s)]
    vv = [s for s in (su, sd) if np.isfinite(s)]
    return {"sigma_left": sl, "sigma_right": sr, "sigma_up": su, "sigma_down": sd,
            "sigma_h": float(np.mean(h)) if h else float("nan"), "sigma_v": float(np.mean(vv)) if vv else float("nan"),
            "n_fits": nl + nr + nu + nd}
