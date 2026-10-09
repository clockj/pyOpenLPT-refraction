"""Shared helpers of the self-calibration tests (plan: 2026-10-08-plan.md, approved 8 Oct with D1-D4 as recommended).

- paths: this folder's data/ and runs/ (git-ignored), T1's case (cases/cyl_20261007: its frames, cameras, runs), the
  study (tests/cylinder_stb_study: the synthetic cases, its scorer), build p4c (read only);
- tracks: the long tracks of an STB run with their stored projections in every camera;
- detections: the C++ line detector (LineFinder::detect), paired one to one with a track's projections;
- cameras: projections and their Jacobians d(u, v)/dX by central differences (hinted projections);
- the correction field (Field): per camera a smooth field c(u, v) = (du, dv), "where the camera's images sit relative to
  where the model projects them" (detection - projection).  A tensor-product cubic B-spline per component over the
  sensor, fitted by projected least squares: a track point seen by the cameras S has residuals r = d - P(X) (2|S|
  numbers) and Jacobian J = dP/dX (2|S| x 3); moving X changes r by -J dX, so only the part of r - c orthogonal to J's
  columns says anything about the cameras.  With N = an orthonormal basis of that complement (2|S| - 3 columns; after
  whitening by the detections' noise), the fit minimises
        sum_p  w_p |N_p^T (r_p - c_p(theta))|^2  +  lam  theta^T Omega theta  +  mu |theta|^2
  (Omega the bending energy, mu a small ridge for the directions no data sees), with Huber weights w_p.  With two
  cameras that complement is one number per point, the offset across the epipolar curve; the fit then puts the
  smallest field that explains it into the two cameras, and leaves the part along the curves (depth) as the model has
  it.  With three or four cameras it is 3 or 5 numbers, and only a smooth 3-D deformation of the world stays unseen;
- the dense field: the fit evaluated on every pixel; beyond the data (more than COVER_PX from a fitted point) it holds
  the value of the nearest covered pixel;
- the warp: an image whose content moves by -c (W(q) = I(q + c), cubic-spline interpolation), so that the unchanged
  cameras fit it.
Everything here runs on build p4c; nothing writes outside this folder.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # the case's and the study's modules are imported, never rewritten

import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs")
WORK = os.path.join(DATA, "work")
CASE = os.path.join(REPO, "cases", "cyl_20261007")
CASE_DATA = os.path.join(CASE, "data")
CASE_RUNS = os.path.join(CASE_DATA, "runs")
STUDY = os.path.join(REPO, "tests", "cylinder_stb_study")
CSTB = os.path.join(REPO, "tests", "cylinder_stb")
BUILD = "p4c"
PRE_M15 = os.path.join(CASE_DATA, "T1pre_s4_m15")      # T1's frames since 8 Oct: S1 with a 15 px median
RUN_M15 = os.path.join(CASE_RUNS, "s3_final_m15")      # T1's deliverable run on them
CAMS_T1 = [os.path.join(CASE_DATA, "cams_T0", f"cam{c}.txt") for c in (1, 2)]
W_IMG, H_IMG = 1280, 800
R_PAIR = 3.0          # px: a detection belongs to a track point's projection within this
COVER_PX = 48.0       # px: the field is fitted where data lie within this; beyond, it holds the nearest value
THREADS_MAX = 12      # the user's limit on OMP_NUM_THREADS
ROBUST = "cauchy"     # the field fit's robust loss (fit_with_noise's default): v2_diag.py, on the 4-camera set, found
S_FLOOR = 0.15        # Cauchy weights with s_f held at >= 0.15 px best (the fitted s_f fell to 0.01 px on clean images)
GAUGE = 0.0           # the weight of the correction's part that moves the points' triangulations (fit_field's default;
                      # v2_gauge.py: 0 = the smallest field (the prior alone), 1 = leave the reconstruction unmoved)
GAUGE_AFFINE = 0.0    # the weight of the affine part of those moves (fit_field's default; v2_gauge.py)

for _p in (STUDY, os.path.join(CASE, "tests"), CASE, CSTB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


# ------------------------------------------------------------------ the library and logs
_LPT = {}


def lib():
    """pyopenlpt of build p4c (and its build record)."""
    if "lpt" not in _LPT:
        import cstb
        _LPT["lpt"], _LPT["rec"] = cstb.use_build(BUILD)
    return _LPT["lpt"]


def build_record():
    lib()
    return _LPT["rec"]


class Log:
    """Prints, keeps and saves <name>_out.txt and <name>.json beside the scripts."""

    def __init__(self, name: str):
        self.stem = os.path.join(HERE, name)
        self.lines = []
        self.t0 = time.time()

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(str(msg))

    def save(self, data=None) -> None:
        with open(self.stem + "_out.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")
        if data is not None:
            save_json(self.stem + ".json", data)


def _json_default(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.bool_):
        return bool(v)
    raise TypeError(type(v))


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=_json_default)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def view_volume():
    return load_json(os.path.join(CASE_DATA, "t1_view_volume.json"))


def in_volume(X, vol):
    X = np.asarray(X, float)
    return ((X[..., 0] >= vol["x"][0]) & (X[..., 0] <= vol["x"][1]) & (X[..., 1] >= vol["y"][0])
            & (X[..., 1] <= vol["y"][1]) & (X[..., 2] >= vol["z"][0]) & (X[..., 2] <= vol["z"][1]))


def one_thread_env():
    """For worker processes that share the cores."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass


# ------------------------------------------------------------------ tracks
def read_tracks(run_dir, kinds=("LongTrackActive", "LongTrackInactive", "ExitTrack")):
    """{(kind, id): rows (frame, X, Y, Z, u_0, v_0, ..., u_{n-1}, v_{n-1})} of the run's last ConvergeTrack files
    (sorted by frame), and the last frame."""
    files = glob.glob(os.path.join(run_dir, "Results", "ConvergeTrack", "*.csv"))
    last = max(int(re.search(r"_(\d+)\.csv$", f).group(1)) for f in files)
    out = {}
    for kind in kinds:
        p = os.path.join(run_dir, "Results", "ConvergeTrack", f"{kind}_{last}.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        if not len(a):
            continue
        order = np.lexsort((a[:, 1], a[:, 0]))
        a = a[order]
        ids, start = np.unique(a[:, 0], return_index=True)
        ends = np.r_[start[1:], len(a)]
        for tid, s, e in zip(ids, start, ends):
            out[(kind, int(tid))] = a[s:e, 1:]
    return out, last


def track_points(tracks, frames, min_len=10):
    """The points of the tracks with at least min_len points at the given frames: arrays (track index, frame, X (3),
    stored projections (n_cam, 2)) and the track keys."""
    keys = [k for k, t in tracks.items() if len(t) >= min_len]
    fr = set(int(f) for f in frames)
    ti, ff, XX, UV = [], [], [], []
    for n, k in enumerate(keys):
        t = tracks[k]
        sel = np.array([int(f) in fr for f in t[:, 0]])
        if not sel.any():
            continue
        ti.append(np.full(int(sel.sum()), n))
        ff.append(t[sel, 0].astype(int))
        XX.append(t[sel, 1:4])
        UV.append(t[sel, 4:].reshape(int(sel.sum()), -1, 2))
    if not ti:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)), np.zeros((0, 0, 2)), keys
    return np.concatenate(ti), np.concatenate(ff), np.vstack(XX), np.concatenate(UV), keys


# ------------------------------------------------------------------ detections
DET_COLS = ["u", "v", "Lx", "Ly", "sx", "sy", "A", "c", "flux", "peak", "score", "chi2", "npx", "pass"]


def line_param(**over):
    lpt = lib()
    P = lpt.LineFinderParam()
    for k, v in over.items():
        setattr(P, k, v)
    return P


def detect(img8, noise, P=None):
    """LineFinder::detect: rows (u, v, Lx, Ly, sx, sy, A, c, flux, peak, score, chi2, npx, pass)."""
    lpt = lib()
    P = P or lpt.LineFinderParam()
    rows = lpt.detectLines(lpt.Image.from_numpy(np.ascontiguousarray(img8)), float(noise), P)
    return np.array(rows, float).reshape(-1, 14)


def read8(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return img


def pair_one_to_one(det_uv, proj_uv, r=R_PAIR):
    """Each projection's detection (index or -1): nearest pairs first within r px, each detection used once."""
    from scipy.spatial import cKDTree
    out = np.full(len(proj_uv), -1)
    ok = np.isfinite(proj_uv[:, 0])
    if not len(det_uv) or not ok.any():
        return out
    tree = cKDTree(det_uv)
    idx = np.flatnonzero(ok)
    lst = tree.query_ball_point(proj_uv[idx], r)
    cand = []
    for a, js in zip(idx, lst):
        for j in js:
            cand.append((float(np.hypot(*(det_uv[j] - proj_uv[a]))), int(a), int(j)))
    cand.sort()
    used = set()
    for _d, a, j in cand:
        if out[a] >= 0 or j in used:
            continue
        out[a] = j
        used.add(j)
    return out


# ------------------------------------------------------------------ cameras
_CAMS = {}


def camera(path):
    if path not in _CAMS:
        _CAMS[path] = lib().loadCameraModel(path)
    return _CAMS[path]


def project_jacobian(path, X, h=0.01):
    """Projections (n, 2) and Jacobians d(u, v)/dX (n, 2, 3) [px/mm] of points X (n, 3) in one camera, by central
    differences of step h mm started from each point's hint; NaN where a projection fails."""
    lpt = lib()
    cam = camera(path)
    X = np.atleast_2d(np.asarray(X, float))
    P = np.full((len(X), 2), np.nan)
    J = np.full((len(X), 2, 3), np.nan)
    for i, x in enumerate(X):
        ok, p, hint, _e = cam.projectWithHint(lpt.Pt3D(*map(float, x)), lpt.ProjectionHint())
        if not ok:
            continue
        P[i] = (p[0], p[1])
        good = True
        for a in range(3):
            xp, xm = x.copy(), x.copy()
            xp[a] += h
            xm[a] -= h
            ok1, p1, _h1, _e1 = cam.projectWithHint(lpt.Pt3D(*map(float, xp)), hint)
            ok2, p2, _h2, _e2 = cam.projectWithHint(lpt.Pt3D(*map(float, xm)), hint)
            if not (ok1 and ok2):
                good = False
                break
            J[i, 0, a] = (p1[0] - p2[0]) / (2 * h)
            J[i, 1, a] = (p1[1] - p2[1]) / (2 * h)
        if not good:
            J[i] = np.nan
    return P, J


def write_matcher_config(camfiles, work_dir, otf):
    """A minimal STB config of the cameras over T1's view volume, for StereoMatch's curved epipolar lines (read by
    load_matcher in any process)."""
    import s_runs
    os.makedirs(work_dir, exist_ok=True)
    lists = []
    for c in range(len(camfiles)):
        p = os.path.join(work_dir, f"dummy_cam{c + 1}.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(os.path.join(CASE_DATA, "T1pre_s4", "cam1", "frame000000.tif") + "\n")
        lists.append(p)
    cfg = os.path.join(work_dir, "config_matcher.json")
    s_runs.write_config(cfg, work_dir, "-", len(camfiles), (0, 0),
                        {"intensity_threshold": 30, "radius_px": 2.0, "otf_file": otf},
                        s_runs.ipr_for(len(camfiles), 1.0), image_lists=lists)
    c = load_json(cfg)
    c["cameras"] = [{"file": f, "max_intensity": 255} for f in camfiles]
    save_json(cfg, c)
    return cfg


def load_matcher(cfg):
    """(cameras, StereoMatch) of a matcher config."""
    lpt = lib()
    key = ("matcher", cfg)
    if key not in _CAMS:
        bs = lpt.BasicSetting()
        bs.readConfig(cfg)
        tc = lpt.TracerConfig()
        tc.readConfig(bs._object_config_paths[0], bs)
        cams = list(bs._cam_list)
        _CAMS[key] = (cams, lpt.StereoMatch(cams, [[] for _ in cams], tc), bs, tc)
    return _CAMS[key][:2]


def curve(cams, sm_, src, dst, uv):
    """The curved image in camera dst of camera src's line of sight through pixel uv: vertices (k, 2), or None."""
    lpt = lib()
    try:
        los = cams[src].lineOfSight(lpt.Pt2D(float(uv[0]), float(uv[1])))
    except Exception:
        return None, None
    ok, _ch, poly = sm_.makeCurve2DFromLOS3D(dst, los)
    if not ok:
        return None, los
    P = np.array([[p[0], p[1]] for p in poly])
    P = P[np.all(np.isfinite(P), axis=1)]
    return (P if len(P) >= 2 else None), los


def curve_v(P, u):
    """The curve's v at columns u (linear between vertices sorted by u), NaN outside it."""
    Q = P[np.argsort(P[:, 0])]
    return np.interp(u, Q[:, 0], Q[:, 1], left=np.nan, right=np.nan)


def poly_dist(P, Q):
    """Distance of points Q (m, 2) to the polyline P (k, 2), and whether the nearest point is inside it."""
    A, B = P[:-1], P[1:]
    E = B - A
    e2 = np.maximum((E ** 2).sum(axis=1), 1e-24)
    t = np.clip(((Q[:, None, :] - A[None]) * E[None]).sum(axis=2) / e2[None], 0.0, 1.0)
    D = np.hypot(*(Q[:, None, :] - (A[None] + t[..., None] * E[None])).transpose(2, 0, 1))
    k = np.argmin(D, axis=1)
    d = D[np.arange(len(Q)), k]
    tb = t[np.arange(len(Q)), k]
    inside = ~(((k == 0) & (tb <= 0.0)) | ((k == len(A) - 1) & (tb >= 1.0)))
    return d, inside


def midpoint(l1, l2):
    p1, d1 = np.array([l1.pt[i] for i in range(3)]), np.array([l1.unit_vector[i] for i in range(3)])
    p2, d2 = np.array([l2.pt[i] for i in range(3)]), np.array([l2.unit_vector[i] for i in range(3)])
    A = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]])
    b = np.array([(p2 - p1) @ d1, (p2 - p1) @ d2])
    try:
        s, t = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, np.inf
    q1, q2 = p1 + s * d1, p2 + t * d2
    return 0.5 * (q1 + q2), float(np.linalg.norm(q1 - q2))


def candidate_pairs(cams, sm_, d1, d2, vol, band=8.0, window=None):
    """S2's candidate pairs (cases/cyl_20261007/tests/s2_shape_fit.candidate_pairs, vectorised, plus the signed
    offset): for every cam1 detection i, its curved image line on cam2; every cam2 detection j whose distance d from it
    is at most band with its nearest point inside the curve, the two lines of sight within 1 mm, the midpoint X inside
    the view volume.  Rows (i, j, X, Y, Z, d, gap, signed offset v_j - curve_v(u_j)).  With window, also every cam2
    detection whose signed offset lies within +-window (T1-a's histogram; no volume or gap test): rows (i, j, offset)."""
    lpt = lib()
    Q2 = d2[:, :2]
    los2 = [None] * len(Q2)
    rows, wrows = [], []
    for i, uv in enumerate(d1[:, :2]):
        P, los1 = curve(cams, sm_, 0, 1, uv)
        if P is None:
            continue
        span = max(band, window or 0.0)
        lo, hi = P.min(axis=0) - span, P.max(axis=0) + span
        sel = np.flatnonzero(np.all((Q2 >= lo) & (Q2 <= hi), axis=1))
        if not len(sel):
            continue
        off = Q2[sel, 1] - curve_v(P, Q2[sel, 0])
        if window is not None:
            m = np.isfinite(off) & (np.abs(off) <= window)
            wrows.extend((i, int(j), float(o)) for j, o in zip(sel[m], off[m]))
        dist, inside = poly_dist(P, Q2[sel])
        keep = (dist <= band) & inside
        for j, dd, oo in zip(sel[keep], dist[keep], off[keep]):
            if los2[j] is None:
                try:
                    los2[j] = cams[1].lineOfSight(lpt.Pt2D(float(Q2[j, 0]), float(Q2[j, 1])))
                except Exception:
                    los2[j] = False
            if los2[j] is False:
                continue
            X, gap = midpoint(los1, los2[j])
            if X is None or gap > 1.0 or not in_volume(X, vol):
                continue
            rows.append((i, int(j), X[0], X[1], X[2], float(dd), gap, float(oo)))
    C = np.array(rows, float).reshape(-1, 8)
    Wn = np.array(wrows, float).reshape(-1, 3)
    return (C, Wn) if window is not None else C


# ------------------------------------------------------------------ the correction field
def bspline_basis(x, n_int, length):
    """Uniform cubic B-spline basis on [0, length] with n_int intervals: (first index (n,), values (n, 4)); the
    n_int + 3 functions, x clamped into the domain."""
    h = length / n_int
    s = np.clip(np.asarray(x, float) / h, 0.0, n_int - 1e-9)
    m = np.floor(s).astype(int)
    t = s - m
    t2, t3 = t * t, t * t * t
    vals = np.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], axis=-1)
    return m, vals


def _gram_1d(n_int, length, deriv_a, deriv_b, n_quad=8):
    """Integral over the domain of B_i^(deriv_a) B_j^(deriv_b) (cubic, uniform)."""
    h = length / n_int
    nb = n_int + 3
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    G = np.zeros((nb, nb))

    def local(t, d):
        if d == 0:
            return np.stack([(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
                             (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6], axis=-1)
        if d == 1:
            return np.stack([-(1 - t) ** 2 / 2, (3 * t ** 2 - 4 * t) / 2, (-3 * t ** 2 + 2 * t + 1) / 2,
                             t ** 2 / 2], axis=-1) / h
        if d == 2:
            return np.stack([(1 - t), (3 * t - 2), (-3 * t + 1), t], axis=-1) / h ** 2
        raise ValueError(d)

    t = 0.5 * (xg + 1.0)
    w = 0.5 * wg * h
    for m in range(n_int):
        A = local(t, deriv_a)
        B = local(t, deriv_b)
        G[m:m + 4, m:m + 4] += (A * w[:, None]).T @ B
    return G


class Field:
    """Per camera and component a tensor-product cubic B-spline over the sensor (n_u x n_v intervals)."""

    def __init__(self, n_cam, n_u=10, n_v=6, width=W_IMG, height=H_IMG):
        self.n_cam, self.n_u, self.n_v, self.width, self.height = n_cam, n_u, n_v, width, height
        self.nbu, self.nbv = n_u + 3, n_v + 3
        self.nb = self.nbu * self.nbv
        self.theta = np.zeros(n_cam * 2 * self.nb)
        self.cov = None
        self.meta = {}
        self.dense = None           # (n_cam, 2, H, W) float32, with the coverage fill
        self.covered = None         # (n_cam, H, W) bool

    def index(self, cam, comp):
        return (cam * 2 + comp) * self.nb

    def basis_rows(self, uv):
        """Columns (n, 16) within one component block and values (n, 16) at pixels uv (n, 2)."""
        mu, bu = bspline_basis(np.asarray(uv)[:, 0] + 0.5, self.n_u, self.width)
        mv, bv = bspline_basis(np.asarray(uv)[:, 1] + 0.5, self.n_v, self.height)
        cols = ((mu[:, None] + np.arange(4)[None, :])[:, :, None] * self.nbv
                + (mv[:, None] + np.arange(4)[None, :])[:, None, :]).reshape(len(uv), 16)
        vals = (bu[:, :, None] * bv[:, None, :]).reshape(len(uv), 16)
        return cols, vals

    def penalty(self):
        """The bending energy int (f_uu^2 + 2 f_uv^2 + f_vv^2) per component block, block-diagonal over all."""
        G0u, G1u, G2u = (_gram_1d(self.n_u, self.width, a, a) for a in (0, 1, 2))
        G0v, G1v, G2v = (_gram_1d(self.n_v, self.height, a, a) for a in (0, 1, 2))
        Om = np.kron(G2u, G0v) + 2 * np.kron(G1u, G1v) + np.kron(G0u, G2v)
        return Om

    def eval_points(self, cam, uv, theta=None):
        """(du, dv) of the spline (no coverage fill) at pixels uv (n, 2) of camera cam."""
        th = self.theta if theta is None else theta
        cols, vals = self.basis_rows(uv)
        out = np.zeros((len(uv), 2))
        for comp in (0, 1):
            t = th[self.index(cam, comp):self.index(cam, comp) + self.nb]
            out[:, comp] = (vals * t[cols]).sum(axis=1)
        return out

    def eval_grid(self, cam, theta=None):
        """The spline on every pixel of camera cam: (2, H, W)."""
        th = self.theta if theta is None else theta
        x = np.arange(self.width) + 0.5
        y = np.arange(self.height) + 0.5
        mu, bu = bspline_basis(x, self.n_u, self.width)
        mv, bv = bspline_basis(y, self.n_v, self.height)
        Bu = np.zeros((self.width, self.nbu))
        Bv = np.zeros((self.height, self.nbv))
        for k in range(4):
            Bu[np.arange(self.width), mu + k] = bu[:, k]
            Bv[np.arange(self.height), mv + k] = bv[:, k]
        out = np.zeros((2, self.height, self.width))
        for comp in (0, 1):
            T = th[self.index(cam, comp):self.index(cam, comp) + self.nb].reshape(self.nbu, self.nbv)
            out[comp] = Bv @ T.T @ Bu.T
        return out

    def make_dense(self, data_uv_per_cam, cover_px=COVER_PX):
        """The dense field of each camera with the coverage fill: pixels farther than cover_px from every fitted point
        take the value of the nearest covered pixel."""
        from scipy import ndimage
        self.dense = np.zeros((self.n_cam, 2, self.height, self.width), np.float32)
        self.covered = np.zeros((self.n_cam, self.height, self.width), bool)
        for c in range(self.n_cam):
            g = self.eval_grid(c)
            pts = np.asarray(data_uv_per_cam[c])
            occ = np.zeros((self.height, self.width), bool)
            if len(pts):
                iu = np.clip(np.rint(pts[:, 0]).astype(int), 0, self.width - 1)
                iv = np.clip(np.rint(pts[:, 1]).astype(int), 0, self.height - 1)
                occ[iv, iu] = True
                dist = ndimage.distance_transform_edt(~occ)
                cov = dist <= cover_px
            else:
                cov = np.zeros_like(occ)
            self.covered[c] = cov
            if cov.any() and not cov.all():
                _d, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
                g = g[:, iy, ix]
            elif not cov.any():
                g = np.zeros_like(g)
            self.dense[c] = g.astype(np.float32)
        return self.dense

    def eval_dense(self, cam, uv):
        """(du, dv) of the dense field (bilinear) at pixels uv (n, 2)."""
        from scipy import ndimage
        uv = np.asarray(uv, float)
        coords = np.vstack([np.clip(uv[:, 1], 0, self.height - 1), np.clip(uv[:, 0], 0, self.width - 1)])
        return np.column_stack([ndimage.map_coordinates(self.dense[cam, k], coords, order=1, mode="nearest")
                                for k in (0, 1)])

    def save(self, path):
        np.savez_compressed(path, theta=self.theta, n_cam=self.n_cam, n_u=self.n_u, n_v=self.n_v,
                            width=self.width, height=self.height,
                            dense=self.dense if self.dense is not None else np.zeros(0),
                            covered=self.covered if self.covered is not None else np.zeros(0),
                            cov=self.cov if self.cov is not None else np.zeros(0),
                            meta=json.dumps(self.meta, default=_json_default))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        f = cls(int(d["n_cam"]), int(d["n_u"]), int(d["n_v"]), int(d["width"]), int(d["height"]))
        f.theta = d["theta"]
        f.dense = d["dense"] if d["dense"].size else None
        f.covered = d["covered"] if d["covered"].size else None
        f.cov = d["cov"] if d["cov"].size else None
        f.meta = json.loads(str(d["meta"]))
        return f


class FitData:
    """The data of a field fit: per point the cameras seen (mask (n, n_cam)), detections d (n, n_cam, 2), residuals
    r = d - P(X) (n, n_cam, 2), Jacobians J (n, n_cam, 2, 3), the detections' box lengths L (n, n_cam, 2) for the noise
    model, labels (frame, track) for held-out splits, and the 3-D points X (n, 3) where known."""

    def __init__(self, mask, d, r, J, L, frame, track, X=None):
        self.mask, self.d, self.r, self.J, self.L = mask, d, r, J, L
        self.frame, self.track, self.X = frame, track, X

    def subset(self, sel):
        return FitData(self.mask[sel], self.d[sel], self.r[sel], self.J[sel], self.L[sel], self.frame[sel],
                       self.track[sel], None if self.X is None else self.X[sel])

    def __len__(self):
        return len(self.mask)

    @staticmethod
    def concat(parts):
        X = None if any(p.X is None for p in parts) else np.concatenate([p.X for p in parts])
        return FitData(*[np.concatenate([getattr(p, k) for p in parts]) for k in
                         ("mask", "d", "r", "J", "L", "frame", "track")], X=X)


def noise_sigma(L, noise):
    """Per detection axis: sqrt(s_f^2 + (beta L)^2), L the box length along the axis."""
    s_f, beta = noise
    return np.sqrt(s_f ** 2 + (beta * np.abs(L)) ** 2)


def _groups(fd: FitData):
    """Points grouped by their camera sets: {tuple(cams): indices}."""
    out = {}
    keys = [tuple(np.flatnonzero(m)) for m in fd.mask]
    for i, k in enumerate(keys):
        if len(k) >= 2:
            out.setdefault(k, []).append(i)
    return {k: np.array(v) for k, v in out.items()}


def projected_system(field: Field, fd: FitData, noise, unit=False, gauge_rows=False):
    """The projected rows of the fit: (Z sparse (rows, n_theta), y (rows,), point index of each row, dof of each
    point).  Each point's whitened residual space (2|S| numbers) minus the span of its whitened Jacobian; unit: no
    whitening.  With gauge_rows, also G (3 rows per point): the field's part along the span of the point's whitened
    Jacobian, i.e. how far the correction moves the point's least-squares triangulation (|G theta| = |J~ dX|)."""
    import scipy.sparse as sp
    rows_i, cols_i, vals_i, ys, pidx = [], [], [], [], []
    g_rows, g_cols, g_vals, gidx = [], [], [], []
    g0 = 0
    row0 = 0
    dof = np.zeros(len(fd), int)
    for cams_s, idx in _groups(fd).items():
        s = len(cams_s)
        cams_s = list(cams_s)
        r = fd.r[idx][:, cams_s].reshape(len(idx), 2 * s)
        J = fd.J[idx][:, cams_s].reshape(len(idx), 2 * s, 3)
        sig = np.ones_like(r) if unit else noise_sigma(fd.L[idx][:, cams_s].reshape(len(idx), 2 * s), noise)
        rw = r / sig
        Jw = J / sig[:, :, None]
        good = np.all(np.isfinite(Jw.reshape(len(idx), -1)), axis=1) & np.all(np.isfinite(rw), axis=1)
        idx, rw, Jw, sig = idx[good], rw[good], Jw[good], sig[good]
        if not len(idx):
            continue
        U, _S, _Vt = np.linalg.svd(Jw, full_matrices=True)
        Nb = U[:, :, 3:]                                    # (n, 2s, 2s-3)
        k = 2 * s - 3
        dof[idx] = k
        y = np.einsum("nik,ni->nk", Nb, rw)                 # (n, k)
        # the field's design: row (cam c, comp) of point p has 16 basis entries / sigma
        cols_all, vals_all = [], []
        for a, c in enumerate(cams_s):
            cols, vals = field.basis_rows(fd.d[idx, c])
            for comp in (0, 1):
                cols_all.append(cols + field.index(c, comp))
                vals_all.append(vals / sig[:, 2 * a + comp][:, None])
        Cc = np.stack(cols_all, axis=1)                     # (n, 2s, 16)
        Cv = np.stack(vals_all, axis=1)                     # (n, 2s, 16)
        # projected row q of point p: sum_i Nb[p, i, q] * design row i
        for q in range(k):
            v = Nb[:, :, q][:, :, None] * Cv                # (n, 2s, 16)
            r_idx = row0 + np.arange(len(idx)) * k + q
            rows_i.append(np.repeat(r_idx, 2 * s * 16))
            cols_i.append(Cc.reshape(len(idx), -1).ravel())
            vals_i.append(v.reshape(len(idx), -1).ravel())
        if gauge_rows:
            if gauge_rows == "move":    # rows of the triangulation's move dX = V S^-1 U_J^T c~ (mm), per point and axis
                coef = np.transpose(_Vt, (0, 2, 1)) / _S[:, None, :]        # (n, axis i, q)
                for i in range(3):
                    v = sum(coef[:, i, q][:, None, None] * U[:, :, q][:, :, None] for q in range(3)) * Cv
                    r_idx = g0 + np.arange(len(idx)) * 3 + i
                    g_rows.append(np.repeat(r_idx, 2 * s * 16))
                    g_cols.append(Cc.reshape(len(idx), -1).ravel())
                    g_vals.append(v.reshape(len(idx), -1).ravel())
                gidx.append(idx)
            else:
                for q in range(3):
                    v = U[:, :, q][:, :, None] * Cv
                    r_idx = g0 + np.arange(len(idx)) * 3 + q
                    g_rows.append(np.repeat(r_idx, 2 * s * 16))
                    g_cols.append(Cc.reshape(len(idx), -1).ravel())
                    g_vals.append(v.reshape(len(idx), -1).ravel())
            g0 += len(idx) * 3
        ys.append(y.ravel())
        pidx.append(np.repeat(idx, k))
        row0 += len(idx) * k
    n_theta = field.theta.size
    if not ys:
        empty = (sp.csr_matrix((0, n_theta)), np.zeros(0), np.zeros(0, int), dof)
        return empty + (sp.csr_matrix((0, n_theta)),) if gauge_rows else empty
    Z = sp.csr_matrix((np.concatenate(vals_i), (np.concatenate(rows_i), np.concatenate(cols_i))),
                      shape=(row0, n_theta))
    if gauge_rows:
        G = sp.csr_matrix((np.concatenate(g_vals), (np.concatenate(g_rows), np.concatenate(g_cols))),
                          shape=(g0, n_theta))
        if gauge_rows == "move":
            return Z, np.concatenate(ys), np.concatenate(pidx), dof, G, np.concatenate(gidx)
        return Z, np.concatenate(ys), np.concatenate(pidx), dof, G
    return Z, np.concatenate(ys), np.concatenate(pidx), dof


def point_norms(e, pidx, n_points):
    """Per point: the squared norm of its projected residual rows."""
    out = np.zeros(n_points)
    np.add.at(out, pidx, e ** 2)
    return out


def robust_weights(nrm, robust="huber", huber=1.5, cauchy_c=2.0):
    """IRLS weights of per-point residual norms (noise units per degree of freedom): Huber, or Cauchy (redescending)."""
    if robust == "cauchy":
        return 1.0 / (1.0 + (nrm / cauchy_c) ** 2)
    return np.where(nrm <= huber, 1.0, huber / np.maximum(nrm, 1e-12))


def robust_loss(t, robust="huber", huber=1.5, cauchy_c=2.0):
    a = np.abs(t)
    if robust == "cauchy":
        return 0.5 * cauchy_c ** 2 * np.log1p((a / cauchy_c) ** 2)
    return np.where(a <= huber, 0.5 * a * a, huber * (a - 0.5 * huber))


def fit_field(field: Field, fd: FitData, lam, noise=(0.3, 0.05), huber=1.5, iters=8, tau=1.0, unit=False,
              log=None, robust=None, cauchy_c=2.0, gauge=None, gauge_affine=None):
    """Projected least squares with the bending penalty lam (per component block), a prior on the field's size
    (|theta|^2 / tau^2, tau in px: the coefficients are the field's values near the knots) and Huber weights on each
    point's projected residual (in its noise units, per degree of freedom).  The prior shrinks the directions the
    data barely see (e.g. cam1's du with two cameras, whose epipolar lines run 0.1 deg from horizontal: seen with a
    weight of about 0.002) to zero, i.e. it leaves them as the model has them, and leaves the well-seen ones alone
    (their information is 100-1000 times 1/tau^2).  Sets field.theta and field.cov (the inverse normal matrix, in
    the noise units' scale); returns a record.
    The gauge (what the tracks cannot see): gauge > 0 penalises each point's triangulation move (|J~ dX|^2, in noise
    units); gauge_affine > 0 penalises only the affine part of the moves over the points (12 numbers: the least-
    squares affine map of dX on X, in mm), so that the correction does not shift, turn, scale or shear the
    reconstruction while it makes the images consistent."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    robust = ROBUST if robust is None else robust
    gauge = GAUGE if gauge is None else gauge
    gauge_affine = GAUGE_AFFINE if gauge_affine is None else gauge_affine
    Paff = None
    if gauge_affine > 0:
        if fd.X is None:
            raise ValueError("gauge_affine needs the points' X (FitData.X)")
        Z, y, pidx, dof, D, gidx = projected_system(field, fd, noise, unit, gauge_rows="move")
        Xp = fd.X[gidx]
        Xn = (Xp - Xp.mean(axis=0)) / np.maximum(Xp.std(axis=0), 1e-9)
        m = len(gidx)
        Wm = np.zeros((3 * m, 12))
        for i in range(3):
            Wm[i::3, 3 * i:3 * i + 3] = Xn
            Wm[i::3, 9 + i] = 1.0
        K = np.asarray((D.T @ Wm)).T                      # (12, n_theta)
        Paff = np.linalg.solve(Wm.T @ Wm, K)              # the affine map's 12 numbers per unit theta
        if gauge > 0:
            raise ValueError("gauge and gauge_affine together are not supported")
    elif gauge > 0:
        Z, y, pidx, dof, G = projected_system(field, fd, noise, unit, gauge_rows=True)
    else:
        Z, y, pidx, dof = projected_system(field, fd, noise, unit)
    Om = field.penalty()
    nblk = field.n_cam * 2
    Pen = sp.block_diag([Om] * nblk, format="csr")
    n = field.theta.size
    w = np.ones(len(fd))
    theta = np.zeros(n)
    R = lam * Pen + (1.0 / tau ** 2) * sp.identity(n, format="csr")
    if gauge > 0:          # the correction should not move the points' triangulations (the reconstruction)
        R = R + gauge * (G.T @ G)
    rec = {}
    for it in range(iters):
        wr = w[pidx]
        A = (Z.T @ sp.diags(wr) @ Z + R).tocsc()
        b = Z.T @ (wr * y)
        if Paff is not None:
            A = A.toarray() + gauge_affine * (Paff.T @ Paff)
            theta_new = np.linalg.solve(A, b)
        else:
            theta_new = spla.spsolve(A, b)
        e = y - Z @ theta_new
        nrm = np.sqrt(point_norms(e, pidx, len(fd)) / np.maximum(dof, 1))
        w = robust_weights(nrm, robust, huber, cauchy_c)
        w[dof == 0] = 0.0
        delta = float(np.max(np.abs(theta_new - theta)))
        theta = theta_new
        if delta < 1e-5:
            break
    iters_done = it + 1
    field.theta = theta
    try:
        field.cov = np.linalg.inv(A if isinstance(A, np.ndarray) else A.toarray())
    except np.linalg.LinAlgError:
        field.cov = None
    used = dof > 0
    rec.update({"iterations": iters_done, "points": int(used.sum()), "rows": int(len(y)), "lam": lam,
                "noise": list(noise), "huber": huber, "tau": tau, "robust": robust,
                "downweighted": float(np.mean(w[used] < 0.5 if robust == "cauchy" else w[used] < 1.0))
                if used.any() else None,
                "rms_before": float(np.sqrt(np.mean(y ** 2))) if len(y) else None,
                "rms_after": float(np.sqrt(np.mean(e ** 2))) if len(y) else None})
    if log:
        log(f"    fit: lam {lam:g}, {rec['points']} points ({rec['rows']} projected rows), {rec['iterations']} "
            f"iterations; projected RMS {rec['rms_before']:.3f} -> {rec['rms_after']:.3f} (noise units); "
            f"{rec['downweighted'] * 100:.1f} % of the points down-weighted")
    return rec


def heldout_score(field: Field, fd: FitData, noise, huber=1.5, unit=False, theta=None, robust=None):
    """On other points: the mean robust loss (Huber, or Cauchy) and the RMS of the projected residual, before (no
    field) and after."""
    robust = ROBUST if robust is None else robust
    Z, y, pidx, dof = projected_system(field, fd, noise, unit)
    th = field.theta if theta is None else theta
    e = y - Z @ th
    def hub(x):
        return robust_loss(x, robust, huber)
    per0 = np.sqrt(point_norms(y, pidx, len(fd)) / np.maximum(dof, 1))
    per1 = np.sqrt(point_norms(e, pidx, len(fd)) / np.maximum(dof, 1))
    used = dof > 0
    return {"points": int(used.sum()), "loss_before": float(np.mean(hub(per0[used]))),
            "loss_after": float(np.mean(hub(per1[used]))),
            "rms_before": float(np.sqrt(np.mean(y ** 2))), "rms_after": float(np.sqrt(np.mean(e ** 2))),
            "mad_before": float(1.4826 * np.median(np.abs(y))), "mad_after": float(1.4826 * np.median(np.abs(e)))}


def estimate_noise(field: Field, fd: FitData, theta=None, bins=8):
    """The noise model (s_f, beta) from the projected residual of the unwhitened system: per point and null
    direction the variance is s_f^2 + beta^2 l with l = sum_i N_i^2 L_i^2; robust variances (MAD^2) in quantile bins
    of l, then a straight line in l."""
    th = field.theta if theta is None else theta
    zs, ls = [], []
    for cams_s, idx in _groups(fd).items():
        s = len(cams_s)
        cams_s = list(cams_s)
        r = fd.r[idx][:, cams_s].reshape(len(idx), 2 * s)
        J = fd.J[idx][:, cams_s].reshape(len(idx), 2 * s, 3)
        L = fd.L[idx][:, cams_s].reshape(len(idx), 2 * s)
        c = np.zeros_like(r)
        for a, cc in enumerate(cams_s):
            c[:, 2 * a:2 * a + 2] = field.eval_points(cc, fd.d[idx, cc], th)
        good = np.all(np.isfinite(J.reshape(len(idx), -1)), axis=1) & np.all(np.isfinite(r), axis=1)
        U, _S, _Vt = np.linalg.svd(J[good], full_matrices=True)
        Nb = U[:, :, 3:]
        z = np.einsum("nik,ni->nk", Nb, (r - c)[good])
        l = np.einsum("nik,ni->nk", Nb ** 2, L[good] ** 2)
        zs.append(z.ravel())
        ls.append(l.ravel())
    z, l = np.concatenate(zs), np.concatenate(ls)
    edges = np.quantile(l, np.linspace(0, 1, bins + 1))
    lb, vb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (l >= lo) & (l <= hi)
        if m.sum() < 30:
            continue
        lb.append(float(np.median(l[m])))
        vb.append(float((1.4826 * np.median(np.abs(z[m] - np.median(z[m])))) ** 2))
    lb, vb = np.array(lb), np.array(vb)
    A = np.column_stack([np.ones_like(lb), lb])
    coef, *_ = np.linalg.lstsq(A, vb, rcond=None)
    s_f2, b2 = max(coef[0], 1e-4), max(coef[1], 0.0)
    return (float(math.sqrt(s_f2)), float(math.sqrt(b2))), {"l_bins": lb.tolist(), "var_bins": vb.tolist()}


def choose_lambda(field_proto, fd_fit, fd_test, lams, noise, log=None, unit=False, tau=1.0, robust=None):
    """The bending weight with the smallest held-out robust loss."""
    robust = ROBUST if robust is None else robust
    best, table = None, []
    for lam in lams:
        f = Field(field_proto.n_cam, field_proto.n_u, field_proto.n_v)
        fit_field(f, fd_fit, lam, noise, unit=unit, tau=tau, robust=robust)
        s = heldout_score(f, fd_test, noise, unit=unit, robust=robust)
        table.append({"lam": lam, **s})
        if log:
            log(f"    lam {lam:g}: held-out {robust} loss {s['loss_before']:.4f} -> {s['loss_after']:.4f}, projected MAD "
                f"{s['mad_before']:.3f} -> {s['mad_after']:.3f}")
        if best is None or s["loss_after"] < best[1]:
            best = (lam, s["loss_after"])
    return best[0], table


def field_se(field: Field, cam, uv, s2=1.0):
    """The standard error of the fitted field (du, dv) at pixels uv, from the inverse normal matrix (times s2)."""
    if field.cov is None:
        return np.full((len(uv), 2), np.nan)
    cols, vals = field.basis_rows(uv)
    out = np.zeros((len(uv), 2))
    for comp in (0, 1):
        off = field.index(cam, comp)
        C = field.cov[np.ix_(np.arange(off, off + field.nb), np.arange(off, off + field.nb))]
        for i in range(len(uv)):
            v = vals[i]
            cc = cols[i]
            out[i, comp] = math.sqrt(max(float(v @ C[np.ix_(cc, cc)] @ v) * s2, 0.0))
    return out


# ------------------------------------------------------------------ gathered data (gather.py)
def load_gather(name, frames=None):
    """The frames gathered by gather.py into data/gather/NAME/ (dicts of arrays), sorted by frame."""
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "gather", name, "frame_*.npz"))):
        z = np.load(f, allow_pickle=False)
        if frames is not None and int(z["t"]) not in frames:
            continue
        out.append({k: z[k] for k in z.files})
    return out


def fitdata(recs, gate=R_PAIR, min_cams=2):
    """FitData of gathered frames: the track points with a detection within gate px in at least min_cams cameras."""
    parts = []
    for g in recs:
        n_cam, n = int(g["n_cam"]), len(g["X"])
        if n == 0:
            continue
        mask = (g["pair"] >= 0) & (g["pair_dist"] <= gate)
        d = np.full((n, n_cam, 2), np.nan)
        L = np.zeros((n, n_cam, 2))
        for c in range(n_cam):
            ok = mask[:, c]
            dc = g[f"det{c}"]
            d[ok, c] = dc[g["pair"][ok, c], :2]
            L[ok, c] = dc[g["pair"][ok, c], 2:4]
        keep = (mask.sum(axis=1) >= min_cams) & np.all(np.isfinite(g["J"].reshape(n, -1)), axis=1)
        parts.append(FitData(mask[keep], d[keep], (d - g["P"])[keep], g["J"][keep], L[keep],
                             np.full(int(keep.sum()), int(g["t"])), g["track"][keep], g["X"][keep]))
    return FitData.concat(parts)


def fit_with_noise(field, fd_fit, fd_test, lams, log, rounds=2, noise0=(0.3, 0.05), tau=1.0, robust=None,
                   s_floor=None):
    """The bending weight by held-out loss and the noise model by the residuals, alternately (rounds times); then the
    final fit on fd_fit with both.  The estimator's defaults are ROBUST and S_FLOOR (set after V2's estimator check,
    v2_diag.py).  Returns (record, lam, noise)."""
    robust = ROBUST if robust is None else robust
    s_floor = S_FLOOR if s_floor is None else s_floor
    noise = noise0
    rec = {"rounds": [], "robust": robust, "s_floor": s_floor}
    lam = None
    for k in range(rounds):
        lam, table = choose_lambda(field, fd_fit, fd_test, lams, noise, log, tau=tau, robust=robust)
        f = Field(field.n_cam, field.n_u, field.n_v)
        fit_field(f, fd_fit, lam, noise, tau=tau, robust=robust)
        noise_new, nb = estimate_noise(f, fd_fit)
        noise_new = (max(noise_new[0], s_floor), noise_new[1])
        rec["rounds"].append({"lam": lam, "noise_in": list(noise), "noise_out": list(noise_new), "bins": nb,
                              "lambda_table": table})
        log(f"    round {k + 1}: lam {lam:g}; noise model s_f {noise_new[0]:.3f} px, beta {noise_new[1]:.4f} "
            f"(from s_f {noise[0]:.3f}, beta {noise[1]:.4f}; s_f held at >= {s_floor})")
        noise = noise_new
    rec["fit"] = fit_field(field, fd_fit, lam, noise, tau=tau, log=log, robust=robust)
    rec["lam"], rec["noise"] = lam, list(noise)
    return rec, lam, noise


def observable(fd: FitData, vec, noise=None):
    """Per point (2 cameras) the projected value N^T vec (unwhitened, unit N) of a stacked per-camera vector field
    vec (n, n_cam, 2): the offset across the epipolar curve it implies, in px of the null direction."""
    out = np.full(len(fd), np.nan)
    for cams_s, idx in _groups(fd).items():
        cams_s = list(cams_s)
        s = len(cams_s)
        if s != 2:
            continue
        J = fd.J[idx][:, cams_s].reshape(len(idx), 4, 3)
        v = vec[idx][:, cams_s].reshape(len(idx), 4)
        U, _S, _V = np.linalg.svd(J, full_matrices=True)
        Nb = U[:, :, 3]
        sgn = np.sign(Nb[:, 3])                 # orient: + along cam2's v
        out[idx] = sgn * np.einsum("ni,ni->n", Nb, v)
    return out


def field_at(field, fd: FitData, dense=True):
    """The field (n, n_cam, 2) at each point's detections (0 where a camera is not paired)."""
    out = np.zeros_like(fd.d)
    for c in range(fd.mask.shape[1]):
        ok = fd.mask[:, c]
        if ok.any() and c < field.n_cam:
            out[ok, c] = field.eval_dense(c, fd.d[ok, c]) if dense else field.eval_points(c, fd.d[ok, c])
    return out


def cell_stats(uv, val, cell=160.0, min_n=30, width=W_IMG, height=H_IMG):
    """Per cell of cell x cell px: count, median and the median's standard error (1.2533 x 1.4826 MAD / sqrt n)."""
    nx, ny = int(math.ceil(width / cell)), int(math.ceil(height / cell))
    iu = np.clip(((uv[:, 0] + 0.5) // cell).astype(int), 0, nx - 1)
    iv = np.clip(((uv[:, 1] + 0.5) // cell).astype(int), 0, ny - 1)
    n = np.zeros((ny, nx), int)
    med = np.full((ny, nx), np.nan)
    se = np.full((ny, nx), np.nan)
    ok = np.isfinite(val)
    for a in range(ny):
        for b in range(nx):
            m = ok & (iv == a) & (iu == b)
            n[a, b] = int(m.sum())
            if n[a, b] >= min_n:
                x = val[m]
                med[a, b] = float(np.median(x))
                se[a, b] = 1.2533 * 1.4826 * float(np.median(np.abs(x - med[a, b]))) / math.sqrt(n[a, b])
    return n, med, se


def map_rms(med, se, n=None):
    """The RMS over cells of the cell medians about their weighted mean, raw and with the noise taken off."""
    ok = np.isfinite(med)
    if not ok.any():
        return {"cells": 0, "rms": None, "rms_noise_corrected": None}
    w = np.ones(ok.sum()) if n is None else n[ok].astype(float)
    m = med[ok]
    mu = float(np.average(m, weights=w))
    raw = float(np.sqrt(np.average((m - mu) ** 2, weights=w)))
    noise2 = float(np.average(se[ok] ** 2, weights=w))
    return {"cells": int(ok.sum()), "mean": mu, "rms": raw, "rms_noise_corrected": math.sqrt(max(raw ** 2 - noise2, 0.0)),
            "se_rms": math.sqrt(noise2), "rms_with_mean": float(np.sqrt(np.average(m ** 2, weights=w)))}


def peak_centre(off, window=6.0, bw=0.25, sig=1.5, bg0=3.0):
    """T1-a's background-subtracted peak of signed offsets: the floor from |off| >= bg0, the centroid of the excess
    within +-sig of the peak (three passes from the maximum of the smoothed histogram)."""
    off = np.asarray(off, float)
    off = off[np.isfinite(off) & (np.abs(off) <= window)]
    e = np.arange(-window, window + 1e-9, bw)
    h, _ = np.histogram(off, bins=e)
    c = 0.5 * (e[1:] + e[:-1])
    fl = np.abs(c) >= bg0
    floor = float(h[fl].mean()) if fl.any() else 0.0
    hs = np.convolve(h - floor, np.ones(5) / 5, mode="same")
    inner = np.abs(c) <= 2.0
    peak = float(c[inner][np.argmax(hs[inner])])
    excess = 0.0
    for _ in range(3):
        sel = np.abs(c - peak) <= sig
        w = h[sel] - floor
        excess = float(w.sum())
        if excess <= 0:
            return float("nan"), 0.0, floor
        peak = float((c[sel] * w).sum() / excess)
    return peak, excess, floor


def peak_ml(off, window=4.0, sigma_bounds=(0.3, 1.5), mu_bound=2.0):
    """Maximum likelihood of the signed offsets within +-window: a floor of random pairs (flat with a slope) plus a
    Gaussian of true pairs (centre mu within +-mu_bound, spread sigma within sigma_bounds; truncated to the window).
    Returns (mu, its standard error from the profile curvature, sigma, the true pairs' share, the true pairs)."""
    from scipy import optimize
    from scipy.special import ndtr
    x = np.asarray(off, float)
    x = x[np.isfinite(x) & (np.abs(x) <= window)]
    n = len(x)
    if n < 50:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0.0

    def nll(p):
        mu, s, f, k = p
        norm = ndtr((window - mu) / s) - ndtr((-window - mu) / s)
        g = np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi) * max(norm, 1e-12))
        fl = (1.0 + k * x / window) / (2 * window)          # integrates to 1 over the window for |k| < 1
        return -float(np.sum(np.log(np.maximum(f * g + (1 - f) * fl, 1e-300))))

    bounds = [(-mu_bound, mu_bound), sigma_bounds, (1e-4, 0.9), (-0.9, 0.9)]
    best = None
    for mu0 in (-1.0, -0.4, 0.0, 0.4, 1.0):
        r = optimize.minimize(nll, [mu0, 0.8, 0.05, 0.0], method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    mu, s, f, k = best.x
    h = 1e-3
    f0 = best.fun
    d2 = (nll([mu + h, s, f, k]) - 2 * f0 + nll([mu - h, s, f, k])) / h ** 2
    se = 1.0 / math.sqrt(d2) if d2 > 0 else float("nan")
    return float(mu), float(se), float(s), float(f), float(f * n)


def null_grid(camfiles, vol, n=(14, 12, 24)):
    """The two cameras' null direction N(X) (unit, oriented with N_v2 > 0; the 4-vector across which the epipolar
    offset lies, in (u1, v1, u2, v2)) on a grid over the view volume, as an interpolator of X (trilinear; nodes no
    camera sees take the nearest valid node's value)."""
    from scipy.interpolate import RegularGridInterpolator
    gx = np.linspace(vol["x"][0], vol["x"][1], n[0])
    gy = np.linspace(vol["y"][0], vol["y"][1], n[1])
    gz = np.linspace(vol["z"][0], vol["z"][1], n[2])
    XX = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    J = np.stack([project_jacobian(f, XX)[1] for f in camfiles], axis=1).reshape(len(XX), 4, 3)
    good = np.all(np.isfinite(J.reshape(len(XX), -1)), axis=1)
    Nv = np.full((len(XX), 4), np.nan)
    U, _S, _V = np.linalg.svd(J[good], full_matrices=True)
    Nb = U[:, :, 3]
    Nb *= np.sign(Nb[:, 3])[:, None]
    Nv[good] = Nb
    if (~good).any():
        from scipy.spatial import cKDTree
        tr = cKDTree(XX[good])
        _d, j = tr.query(XX[~good])
        Nv[~good] = Nv[good][j]
    vals = Nv.reshape(n[0], n[1], n[2], 4)
    interp = RegularGridInterpolator((gx, gy, gz), vals, bounds_error=False, fill_value=None)

    def f(X):
        v = interp(np.asarray(X, float))
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return f, {"nodes": int(len(XX)), "seen": int(good.sum())}


def mixture_fit(field, d1, d2, Nv, y, Lv1, Lv2, lam, noise, window, iters=400, tau=1.0, frac0=None, kappa0=1.0,
                log=None, init_theta=None, tol=1e-8):
    """The field from every candidate pair, with no tracks: y = the pair's offset across the curve in the null
    direction (signed offset x N_v2), modelled as y = N . c(d) + noise for a true pair and uniform over the window
    (+-window x N_v2) for a random one.  EM: the responsibilities from the current field, then weighted least squares
    (weights resp / v, v = kappa^2 (N_v1^2 s(Lv1)^2 + N_v2^2 s(Lv2)^2), s the noise model), the bending penalty lam
    and the prior 1/tau^2; the background share and kappa re-estimated each pass, until the mean log-likelihood changes
    by less than tol (EM with a 95 % background converges slowly: hundreds of passes).  The background share starts
    from the floor of the offsets' histogram (|o| in [2/3, 1] x window), the field at init_theta (default 0).  Only the
    v components are fitted (with two cameras the u components enter with weights of about 0.002).  Returns a record
    (with the final mean log-likelihood); sets field.theta."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    n = len(y)
    s1, s2 = noise_sigma(Lv1, noise), noise_sigma(Lv2, noise)
    v0 = Nv[:, 1] ** 2 * s1 ** 2 + Nv[:, 3] ** 2 * s2 ** 2
    dens_bg = 1.0 / (2.0 * window * Nv[:, 3])            # uniform over the window, in y units
    # the design: v1 component of cam 0 and v2 component of cam 1
    c1, b1 = field.basis_rows(d1)
    c2, b2 = field.basis_rows(d2)
    cols = np.hstack([c1 + field.index(0, 1), c2 + field.index(1, 1)])
    vals = np.hstack([b1 * Nv[:, 1:2], b2 * Nv[:, 3:4]])
    nth = field.theta.size
    Pen = sp.block_diag([field.penalty()] * (field.n_cam * 2), format="csr")
    R = lam * Pen + (1.0 / tau ** 2) * sp.identity(nth, format="csr")
    theta = np.zeros(nth) if init_theta is None else np.array(init_theta, float)
    if frac0 is None:      # the floor of |o| in [2/3, 1] x window, extended over the window
        o = y / Nv[:, 3]
        outer = np.abs(o) >= (2.0 / 3.0) * window
        frac0 = min(0.999, float(outer.mean()) * 3.0)
    frac, kappa = frac0, kappa0
    rows = np.repeat(np.arange(n), cols.shape[1])
    Z = sp.csr_matrix((vals.ravel(), (rows, cols.ravel())), shape=(n, nth))
    hist = []
    ll_prev = -np.inf
    for it in range(iters):
        e = y - Z @ theta
        v = kappa ** 2 * v0
        g = np.exp(-0.5 * e * e / v) / np.sqrt(2 * math.pi * v)
        mix = (1 - frac) * g + frac * dens_bg
        ll = float(np.mean(np.log(mix)))
        resp = (1 - frac) * g / mix
        frac = float(1 - resp.mean())
        kappa = float(math.sqrt(np.sum(resp * e * e / v0) / max(np.sum(resp), 1e-9)))
        w = resp / (kappa ** 2 * v0)
        A = (Z.T @ sp.diags(w) @ Z + R).tocsc()
        b = Z.T @ (w * y)
        th_new = spla.spsolve(A, b)
        delta = float(np.max(np.abs(th_new - theta)))
        theta = th_new
        hist.append({"iter": it + 1, "true_share": 1 - frac, "true_pairs": float(resp.sum()), "kappa": kappa,
                     "delta": delta, "loglik": ll})
        if abs(ll - ll_prev) < tol and it > 5:
            break
        ll_prev = ll
    field.theta = theta
    rec = {"pairs": n, "iterations": len(hist), "true_share": 1 - frac, "true_pairs": float(resp.sum()),
           "kappa": kappa, "loglik": ll, "history": hist[::10] + [hist[-1]], "lam": lam, "noise": list(noise),
           "window_px": window, "frac0": frac0, "warm_start": init_theta is not None}
    if log:
        log(f"    mixture fit{' (warm start)' if init_theta is not None else ''}: {n} candidate pairs within +-{window} "
            f"px, {len(hist)} passes; true pairs {rec['true_pairs']:.0f} ({100 * rec['true_share']:.1f} %), their "
            f"spread {kappa:.3f} x the noise model; mean log-likelihood {ll:.6f}")
    return rec


def compose(f1: Field, f2: Field) -> Field:
    """The correction of two passes as one field of the original pixels: f1 was fitted on the original images, f2 on
    the images corrected by f1 (whose pixel q = d - c1(d)); the total moves d to d - c1(d) - c2(d - c1(d)), so
    c_tot(d) = c1(d) + c2(d - c1(d)), on the dense grid (f2's dense field read bilinearly at d - c1(d)).  Covered: f1's.
    theta is the sum (an approximation, not used by the warp or the re-rendering, which read the dense field)."""
    out = Field(f1.n_cam, f1.n_u, f1.n_v, f1.width, f1.height)
    out.theta = f1.theta + f2.theta
    out.dense = np.zeros_like(f1.dense)
    out.covered = f1.covered.copy()
    H, Wd = f1.height, f1.width
    qy, qx = np.mgrid[0:H, 0:Wd].astype(np.float64)
    for c in range(f1.n_cam):
        c1 = f1.dense[c].astype(np.float64)
        src = np.column_stack([(qx - c1[0]).ravel(), (qy - c1[1]).ravel()])
        c2 = f2.eval_dense(c, src).T.reshape(2, H, Wd)
        out.dense[c] = (c1 + c2).astype(np.float32)
    out.meta = {"composed_of": [f1.meta, f2.meta]}
    return out


# ------------------------------------------------------------------ the warp
def warp_maps(field: Field, cam, iters=2):
    """cv2.remap maps so that W(q) = I(q + c), with the correction taken where the image content came from: the fixed
    point d = q + c(d), d the source pixel of q."""
    H, Wd = field.height, field.width
    qx, qy = np.meshgrid(np.arange(Wd, dtype=np.float64), np.arange(H, dtype=np.float64))
    dx, dy = qx.copy(), qy.copy()
    for _ in range(iters + 1):
        c = field.eval_dense(cam, np.column_stack([dx.ravel(), dy.ravel()]))
        dx = qx + c[:, 0].reshape(H, Wd)
        dy = qy + c[:, 1].reshape(H, Wd)
    return dx, dy


def warp(img, maps, order=3):
    """W(q) = I(maps(q)) by cubic-spline interpolation at the exact positions (scipy.ndimage.map_coordinates; cv2.remap
    rounds positions to 1/32 px, an error up to 0.016 px), mirrored at the borders."""
    from scipy import ndimage
    return ndimage.map_coordinates(np.asarray(img, np.float32), [maps[1], maps[0]], order=order, mode="mirror",
                                   prefilter=True).astype(np.float32)
