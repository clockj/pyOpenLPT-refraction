"""Vectorised PINPLATE camera model (a pinhole behind flat parallel plates), with exact derivatives, for calibration.

It is the same camera as pyOpenLPT's C++ `RefractionPinholeCamera` (PINPLATE): a pinhole with OpenCV distortion
(k1, k2, p1, p2, k3) looking through a stack of parallel plates into the object medium, e.g. air | acrylic | water.
Like `cyl_model`, every function takes per-row camera parameters (`cyl_model.Rows`) and per-row window parameters
(`WRows`), so the rows of one call may belong to different cameras and windows, and it returns the derivatives a
Levenberg-Marquardt solver needs.  The C++ model stays the source of truth: tests/plate_wand/w0_model_vs_cpp.py
checks this one against it.

**Window.**  A `Window` has a unit normal n (camera -> object), its camera-side surface {X : n.X = d}, the plate
thicknesses t (camera side first) and the indices idx = (n_cam, n_plate..., n_obj).  The C++ stores the stack the
other way round, from a point on the object-side surface, with the indices from the object medium to the camera's
and the thicknesses from the object side; `to_cpp` converts.  A row without a window (`WRows.has` False) is a plain
pinhole camera, so the pinhole stage of the calibration uses the same code.

**Forward projection** (`project`) is a 1-D solve.  The ray from X to the camera stays in the plane spanned by the
normal and X - C, and Snell's law keeps n sin(theta) the same in every layer.  With tau = tan(theta) in the
camera's medium, the ray's lateral travel is

    F(tau) = h1 tau + sum_i H_i g(tau; n_i),     g(tau; n) = n0 tau / sqrt(n^2 + (n^2 - n0^2) tau^2),

summed over the plates (H_i = t_i) and the object medium (H = h_W, the point's depth below the object-side surface),
with h1 = d - n.C the camera's distance to the window.  It must equal rho, the lateral distance from C to X.  F is
increasing and, when no layer is optically thinner than the camera's medium, concave, so Newton from
tau = rho / F'(0) rises monotonically to the root (SymPy: tests/plate_wand/p_sympy_presize.py).  The camera-side ray
is then q = n + (tau / rho) l, with l = (X - C) - a n the lateral part of X - C, a = n.(X - C); the pixel follows from
the pinhole and the distortion in closed form.

**Derivatives** follow by implicit differentiation of F(tau) = rho (SymPy: tests/plate_wand/w0_sympy_derivations.py):
    d tau = (d rho - tau d h1 - g_W d h_W) / F'(tau),   g_W = g(tau; n_obj),
for the camera (`CAM_PARAMS` order, rotation increments R <- R exp([w]x) as in `cyl_model`), the window (`WIN_PARAMS`:
the normal's increment a t1 + b t2 on the sphere, and the offset delta in d = n.A + delta, so a turn of the normal
pivots the surface about the anchor point A) and the point X.  The plate thicknesses and indices are constants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from . import cyl_model as cm
except ImportError:  # imported as a top-level module (tests put this folder on sys.path)
    import cyl_model as cm

Camera = cm.Camera
Rows = cm.Rows
rows = cm.rows
distort = cm.distort
undistort = cm.undistort
pixels = cm.pixels
skew = cm.skew
exp_so3 = cm.exp_so3
log_so3 = cm.log_so3
rvec_of = cm.rvec_of
rot_of = cm.rot_of
rotation_angle_deg = cm.rotation_angle_deg
cpp_project = cm.cpp_project
cpp_rays = cm.cpp_rays
CAM_PARAMS = cm.CAM_PARAMS
WIN_PARAMS = ("a", "b", "delta")


# ------------------------------------------------------------------ the window
def tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(t1, t2) perpendicular to each unit row of n (m, 3), t2 = n x t1 (as `cylinder_wand.Frames.tangent_basis`):
    the coordinates (a, b) of a normal's increment a t1 + b t2."""
    n = np.atleast_2d(np.asarray(n, float))
    k = np.argmin(np.abs(n), axis=1)
    e = np.zeros_like(n)
    e[np.arange(len(n)), k] = 1.0
    t1 = e - np.einsum("ij,ij->i", e, n)[:, None] * n
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(n, t1)


def retract_normal(n: np.ndarray, ab) -> np.ndarray:
    """The normal moved by (a, b) along the unit sphere: exp_n(a t1 + b t2)."""
    n = np.asarray(n, float).reshape(3)
    t1, t2 = tangent_basis(n)
    v = float(ab[0]) * t1[0] + float(ab[1]) * t2[0]
    th = float(np.linalg.norm(v))
    if th == 0.0:
        return n.copy()
    out = math.cos(th) * n + (math.sin(th) / th) * v
    return out / np.linalg.norm(out)


@dataclass
class Window:
    """A stack of flat parallel plates: unit normal n (camera -> object), camera-side surface {X : n.X = d},
    thicknesses t (mm, camera side first), indices idx = (n_cam, n_plate..., n_obj)."""

    n: np.ndarray
    d: float
    t: Tuple[float, ...]
    idx: Tuple[float, ...]
    name: str = ""

    def __post_init__(self):
        n = np.asarray(self.n, float).reshape(3)
        nn = float(np.linalg.norm(n))
        if not nn > 0:
            raise ValueError("the window normal must be non-zero")
        self.n = n / nn
        self.d = float(self.d)
        self.t = tuple(float(v) for v in np.atleast_1d(self.t))
        self.idx = tuple(float(v) for v in self.idx)
        if len(self.idx) != len(self.t) + 2 or not self.t:
            raise ValueError("a window needs K >= 1 thicknesses and K + 2 indices (camera, plates, object)")
        if any(not v > 0 for v in self.t) or any(not v > 0 for v in self.idx):
            raise ValueError("thicknesses and indices must be positive")

    @property
    def T(self) -> float:
        """Total thickness."""
        return float(sum(self.t))

    @property
    def far_d(self) -> float:
        """The object-side surface is {X : n.X = far_d}."""
        return self.d + self.T

    def distance(self, C) -> float:
        """A camera centre's distance to the window (h1 > 0: on the camera side)."""
        return self.d - float(self.n @ np.asarray(C, float))

    def depth(self, X) -> np.ndarray:
        """Points' depth below the object-side surface (> 0: in the object medium)."""
        return np.asarray(X, float) @ self.n - self.far_d

    def copy(self) -> "Window":
        return Window(self.n.copy(), self.d, self.t, self.idx, self.name)


@dataclass
class WRows:
    """Per-row window parameters, the plates padded to a common count K (a missing plate has thickness 0 and the
    object's index, which changes nothing).  has[i] False: row i is a pinhole camera without a window."""

    n: np.ndarray        # (m, 3)
    d: np.ndarray        # (m,)
    t: np.ndarray        # (m, K)
    idx: np.ndarray      # (m, K + 2)
    A: np.ndarray        # (m, 3) the anchor of each row's window (pivot of the normal's increments)
    has: np.ndarray      # (m,) bool

    def __len__(self) -> int:
        return len(self.d)

    def take(self, sel) -> "WRows":
        return WRows(self.n[sel], self.d[sel], self.t[sel], self.idx[sel], self.A[sel], self.has[sel])


def wrows(windows: Sequence[Optional[Window]], win_of_row, anchors=None) -> WRows:
    """Rows for window indices `win_of_row` (int array; -1 or a None window: a pinhole row).  anchors: one point per
    window (default: the foot of the origin on its camera-side surface)."""
    win_of_row = np.asarray(win_of_row, int)
    m = len(win_of_row)
    K = max([len(w.t) for w in windows if w is not None] or [1])
    nW = len(windows)
    Nn = np.tile([0.0, 0.0, 1.0], (nW + 1, 1))
    Dd = np.zeros(nW + 1)
    Tt = np.zeros((nW + 1, K))
    Ii = np.ones((nW + 1, K + 2))
    Aa = np.zeros((nW + 1, 3))
    Hh = np.zeros(nW + 1, bool)
    for k, w in enumerate(windows):
        if w is None:
            continue
        Kw = len(w.t)
        Nn[k], Dd[k], Hh[k] = w.n, w.d, True
        Tt[k, :Kw] = w.t
        Ii[k, :Kw + 1] = w.idx[:Kw + 1]
        Ii[k, Kw + 1:] = w.idx[-1]
        Aa[k] = w.d * w.n if anchors is None or anchors[k] is None else np.asarray(anchors[k], float)
    j = np.where((win_of_row >= 0) & (win_of_row < nW), win_of_row, nW)
    j = np.where(Hh[j], j, nW)
    return WRows(Nn[j], Dd[j], Tt[j], Ii[j], Aa[j], Hh[j]) if m else WRows(
        np.zeros((0, 3)), np.zeros(0), np.zeros((0, K)), np.zeros((0, K + 2)), np.zeros((0, 3)), np.zeros(0, bool))


# ------------------------------------------------------------------ forward projection
@dataclass
class PlateProjection:
    uv: np.ndarray                        # (m, 2) pixels (NaN where not ok)
    xy: np.ndarray                        # (m, 2) normalised undistorted coordinates
    ok: np.ndarray                        # (m,)
    iters: int = 0
    J_cam: Optional[np.ndarray] = None    # (m, 2, 14) d uv / d CAM_PARAMS
    J_win: Optional[np.ndarray] = None    # (m, 2, 3)  d uv / d (a, b, delta); zero for pinhole rows
    J_X: Optional[np.ndarray] = None      # (m, 2, 3)  d uv / d X
    tau: Optional[np.ndarray] = None      # (m,) tan of the ray's angle to the normal in the camera's medium
    h1: Optional[np.ndarray] = None       # (m,) camera's distance to the window
    hW: Optional[np.ndarray] = None       # (m,) point's depth below the object-side surface


def _g(tau, N, n0):
    """g(tau; n) per layer and its tau-derivative; tau (m,), N (m, L), n0 (m,)."""
    den = N * N + (N * N - (n0 * n0)[:, None]) * (tau * tau)[:, None]
    sq = np.sqrt(den)
    return n0[:, None] * tau[:, None] / sq, n0[:, None] * N * N / (den * sq)


def solve_tau(rho, h1, H, N, n0, max_iter: int = 60, tol: float = 1e-9, polish: int = 2):
    """Newton for F(tau) = rho from tau = rho / F'(0) (monotone from below when every N >= n0).  Stops once a step is
    below tol (1 + tau), then takes `polish` more steps (quadratic: to round-off).  Returns (tau, F'(tau), g per layer
    at tau, converged, iterations)."""
    rho = np.asarray(rho, float)
    m = len(rho)
    Fp0 = h1 + np.sum(H * n0[:, None] / N, axis=1)
    tau = rho / Fp0
    conv = np.zeros(m, bool)
    extra = np.zeros(m, int)                            # Newton steps taken after the tolerance was reached
    active = np.arange(m)
    it = 0
    for it in range(1, max_iter + polish + 1):
        ta = tau[active]
        g, gp = _g(ta, N[active], n0[active])
        F = h1[active] * ta + np.sum(H[active] * g, axis=1)
        Fp = h1[active] + np.sum(H[active] * gp, axis=1)
        step = (rho[active] - F) / Fp
        new = ta + step
        bad = ~np.isfinite(new) | (new < 0.0)
        tau[active] = np.where(bad, ta, new)
        was = conv[active]
        extra[active[was]] += 1
        conv[active[~bad & (np.abs(step) <= tol * (1.0 + np.abs(new)))]] = True
        conv[active[bad]] = False
        done = bad | (was & (extra[active] >= polish))
        active = active[~done]
        if active.size == 0:
            break
    g, gp = _g(tau, N, n0)
    Fp = h1 + np.sum(H * gp, axis=1)
    return tau, Fp, g, conv, it


def project(rw: Rows, ww: WRows, X: np.ndarray, jac: bool = False) -> PlateProjection:
    """Pixels of world points X (m, 3), row i by camera rw[i] through window ww[i]; with jac the exact derivatives.

    A row is ok when the camera is on the camera side of its window (h1 > 0), the point in the object medium
    (h_W > 0), in front of the camera (p_z > 0), no layer is optically thinner than the camera's medium, Newton
    converged and everything is finite."""
    X = np.asarray(X, float)
    m = len(X)
    v = X - rw.C
    q = v.copy()                                            # pinhole rows: q = X - C
    ok = np.ones(m, bool)
    tau = np.zeros(m)
    h1 = np.full(m, np.nan)
    hW = np.full(m, np.nan)
    it = 0
    sel = np.flatnonzero(ww.has)
    if jac:
        JqX = np.tile(np.eye(3), (m, 1, 1))
        JqC = -JqX.copy()
        Jqn = np.zeros((m, 3, 2))
        Jqd = np.zeros((m, 3))
    if sel.size:
        n = ww.n[sel]
        vs = v[sel]
        a = np.einsum("ij,ij->i", n, vs)
        h1s = ww.d[sel] - np.einsum("ij,ij->i", n, rw.C[sel])
        T = ww.t[sel].sum(axis=1)
        hWs = a - h1s - T
        ell = vs - a[:, None] * n
        rho = np.linalg.norm(ell, axis=1)
        idx = ww.idx[sel]
        n0 = idx[:, 0]
        N = idx[:, 1:]
        H = np.column_stack([ww.t[sel], hWs])
        good = (h1s > 0) & (hWs > 0) & np.all(N >= n0[:, None], axis=1)
        H = np.where(good[:, None], H, 1.0)
        h1g = np.where(good, h1s, 1.0)
        rhog = np.where(good, rho, 0.0)
        taus, Fp, g, conv, it = solve_tau(rhog, h1g, H, N, n0)
        Fp0 = h1g + np.sum(H * n0[:, None] / N, axis=1)
        ratio = np.where(rhog > 0, taus / np.where(rhog > 0, rhog, 1.0), 1.0 / Fp0)
        q[sel] = n + ratio[:, None] * ell
        ok[sel] = good & conv
        tau[sel] = np.where(good, taus, np.nan)
        h1[sel], hW[sel] = h1s, hWs
        if jac:
            # e: the in-plane unit direction from C to X (any unit vector perpendicular to n when rho is round-off:
            # the derivatives are continuous there), e2 = n x e
            t1, _t2 = tangent_basis(n)
            tiny = rhog <= 1e-10 * (1.0 + np.linalg.norm(vs, axis=1))
            e = np.where(tiny[:, None], t1, ell / np.where(rhog > 0, rhog, 1.0)[:, None])
            e2 = np.cross(n, e)
            gW = g[:, -1]
            Fp_ = Fp[:, None, None]
            r_ = ratio[:, None, None]
            E2 = e2[:, :, None] * e2[:, None, :]
            EE = e[:, :, None] * e[:, None, :]
            JqX[sel] = e[:, :, None] * (e - gW[:, None] * n)[:, None, :] / Fp_ + r_ * E2
            JqC[sel] = e[:, :, None] * (taus[:, None] * n - e)[:, None, :] / Fp_ - r_ * E2
            Jqd[sel] = e * ((gW - taus) / Fp)[:, None]
            A = ww.A[sel]
            C = rw.C[sel]
            w_n = -a[:, None] * e - taus[:, None] * (A - C) - gW[:, None] * (X[sel] - A)
            Jn = (np.eye(3)[None] + e[:, :, None] * w_n[:, None, :] / Fp_
                  - r_ * (n[:, :, None] * vs[:, None, :] + a[:, None, None] * (np.eye(3)[None] - EE)))
            tb1, tb2 = tangent_basis(n)
            Jqn[sel] = np.matmul(Jn, np.stack([tb1, tb2], axis=2))
    p = np.matmul(rw.R, q[:, :, None])[:, :, 0]
    ok &= p[:, 2] > 0
    pz = np.where(ok, p[:, 2], 1.0)
    xy = p[:, :2] / pz[:, None]
    ok &= np.all(np.isfinite(xy), axis=1)
    xy = np.where(ok[:, None], xy, np.nan)
    out = PlateProjection(uv=np.full((m, 2), np.nan), xy=xy, ok=ok, iters=it, tau=tau, h1=h1, hW=hW)
    good = np.flatnonzero(ok)
    if jac:                                                 # NaN rows where not ok, even when no row is
        out.J_cam = np.full((m, 2, len(CAM_PARAMS)), np.nan)
        out.J_X = np.full((m, 2, 3), np.nan)
        out.J_win = np.full((m, 2, 3), np.nan)
    if good.size == 0:
        return out
    sub = rw.take(good)
    if not jac:
        out.uv[good] = pixels(xy[good], sub)
        return out
    xd, Jd, Jk = distort(xy[good], sub.dist, jac=True)
    out.uv[good] = np.column_stack([sub.f * xd[:, 0] + sub.cx, sub.f * xd[:, 1] + sub.cy])
    x, y, pzg = xy[good, 0], xy[good, 1], pz[good]
    Dp = np.zeros((len(good), 2, 3))
    Dp[:, 0, 0] = 1.0 / pzg
    Dp[:, 1, 1] = 1.0 / pzg
    Dp[:, 0, 2] = -x / pzg
    Dp[:, 1, 2] = -y / pzg
    G = sub.f[:, None, None] * np.matmul(Jd, Dp)                        # d uv / d p
    GR = np.matmul(G, sub.R)                                             # d uv / d q
    Jc = np.zeros((len(good), 2, len(CAM_PARAMS)))
    Jc[:, :, 0:3] = -np.matmul(GR, skew(q[good]))                       # p = R q, R <- R exp([w]x)
    Jc[:, :, 3:6] = np.matmul(GR, JqC[good])
    Jc[:, :, 6] = sub.f[:, None] * xd
    Jc[:, 0, 7] = 1.0
    Jc[:, 1, 8] = 1.0
    Jc[:, :, 9:14] = sub.f[:, None, None] * Jk
    out.J_cam[good] = Jc
    out.J_X[good] = np.matmul(GR, JqX[good])
    out.J_win[good, :, 0:2] = np.matmul(GR, Jqn[good])
    out.J_win[good, :, 2] = np.matmul(GR, Jqd[good][:, :, None])[:, :, 0]
    return out


# ------------------------------------------------------------------ back-projection
def trace(rw: Rows, ww: WRows, xy: np.ndarray):
    """Back-project normalised undistorted coordinates: (o, d, ok), o on the object-side surface (a pinhole row: the
    centre), d the unit direction in the object medium.  ok needs the ray to head into the window (n.q > 0), the
    camera on the camera side and no total internal reflection."""
    xy = np.asarray(xy, float)
    m = len(xy)
    mm = np.column_stack([xy[:, 0], xy[:, 1], np.ones(m)])
    q = np.matmul(mm[:, None, :], rw.R)[:, 0, :]                       # R^T m
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    o = rw.C.astype(float).copy()
    d = q.copy()
    ok = np.all(np.isfinite(q), axis=1)
    sel = np.flatnonzero(ww.has)
    if sel.size:
        n = ww.n[sel]
        oo = o[sel]
        dd = d[sel]
        pos = ww.d[sel].copy()
        oks = (np.einsum("ij,ij->i", n, dd) > 0) & (pos - np.einsum("ij,ij->i", n, oo) > 0)
        idx = ww.idx[sel]
        K = ww.t.shape[1]
        for k in range(K + 1):
            ci = np.einsum("ij,ij->i", n, dd)
            cis = np.where(ci > 0, ci, 1.0)
            s = (pos - np.einsum("ij,ij->i", n, oo)) / cis
            oo = oo + s[:, None] * dd
            mu = idx[:, k] / idx[:, k + 1]
            ct2 = 1.0 - mu * mu * (1.0 - ci * ci)
            oks &= ct2 > 0
            ct = np.sqrt(np.maximum(ct2, 0.0))
            dd = mu[:, None] * dd + (ct - mu * ci)[:, None] * n
            dd /= np.linalg.norm(dd, axis=1, keepdims=True)
            if k < K:
                pos = pos + ww.t[sel, k]
        o[sel], d[sel], ok[sel] = oo, dd, ok[sel] & oks
    return o, d, ok


def rays(rw: Rows, ww: WRows, uv: np.ndarray):
    """Lines of sight of pixels: (o, d, ok) as `trace`."""
    return trace(rw, ww, undistort(np.asarray(uv, float), rw))


# ------------------------------------------------------------------ the C++ model
def import_lpt():
    return cm.import_lpt()


def to_cpp(cam: Camera, win: Window, tol: float = 1e-12, nmax: int = 1000, far_pt=None):
    """The C++ PINPLATE camera of `cam` behind `win`.  The window's point is the foot of the camera centre on the
    object-side surface unless `far_pt` (on that surface) is given.  Asserts that the C++ rotation is the intended
    one: a rotation vector near a half-turn is easily wrong (tests/plate_wand/w0_logmap.py).  To 1e-12: the C++
    conversion rounds to about 1e-14 (case plate_20260727_T0 had a rotation 1.2e-14 off)."""
    lpt = import_lpt()
    R = np.asarray(cam.R, float)
    C = np.asarray(cam.C, float)
    c = lpt.Camera()
    c.setPinplateImageSize(int(cam.height), int(cam.width))
    c.setPinplateIntrinsics(float(cam.f), float(cam.f), float(cam.cx), float(cam.cy), [float(v) for v in cam.dist])
    c.setPinplateExtrinsics(lpt.Pt3D(*map(float, rvec_of(R))), lpt.Pt3D(*map(float, -R @ C)))
    if far_pt is None:
        far_pt = C + (win.far_d - float(win.n @ C)) * win.n
    c.setPinplateRefraction(lpt.Pt3D(*map(float, far_pt)), lpt.Pt3D(*map(float, win.n)),
                            [float(v) for v in reversed(win.idx)], [float(v) for v in reversed(win.t)],
                            float(tol), int(nmax), 0.1)
    c.commitPinplateUpdate(True, 255.0)
    pp = c._pinplate_param
    R_cpp = np.array([[pp.r_mtx[i, j] for j in range(3)] for i in range(3)])
    if not np.max(np.abs(R_cpp - R)) < 1e-12:
        raise AssertionError(f"the C++ rotation differs from the intended one by {np.max(np.abs(R_cpp - R)):.1e}")
    return c


def from_cpp(cam_cpp) -> Tuple[Camera, Window]:
    """The (Camera, Window) of a C++ PINPLATE camera (e.g. one loaded from a camFile)."""
    pp = cam_cpp._pinplate_param
    R = np.array([[pp.r_mtx[i, j] for j in range(3)] for i in range(3)])
    tv = np.array([pp.t_vec[i] for i in range(3)])
    K = np.array([[pp.cam_mtx[i, j] for j in range(3)] for i in range(3)])
    dist = list(pp.dist_coeff)[:5] if pp.is_distorted else [0.0] * 5
    C = -R.T @ tv
    cam = Camera(R, C, 0.5 * (K[0, 0] + K[1, 1]), K[0, 2], K[1, 2], int(pp.n_col), int(pp.n_row), dist)
    nvec = np.array([pp.plane.norm_vector[i] for i in range(3)])
    pt = np.array([pp.plane.pt[i] for i in range(3)])
    idx = tuple(reversed(list(pp.refract_array)))
    t = tuple(reversed(list(pp.w_array)))
    nvec = nvec / np.linalg.norm(nvec)
    far_d = float(nvec @ pt)
    return cam, Window(nvec, far_d - float(sum(t)), t, idx)
