"""Vectorised PINCYL camera model in the tube frame, with exact derivatives, for calibration.

It is the same camera as pyOpenLPT's C++ `RefractionCylinderCameraModel` (PINCYL): a pinhole in medium 1 with
OpenCV distortion (k1, k2, p1, p2, k3), looking through coaxial cylinders.  It is written for fitting: every
function takes per-row camera parameters (`Rows`), so the rows of one call may belong to different cameras, and
it returns the derivatives a Gauss-Newton or Levenberg-Marquardt solver needs.  The C++ model stays the source of
truth: `tests/cylinder_wand/t0_fast_vs_cpp.py` checks this one against it.

**Frame.**  Everything is in the tube frame: the axis is the Y axis through the origin (`to_tube_frame` converts
a camera from any frame).  R is the world -> camera rotation (OpenCV axes: x right, y down, z forward) and C the
camera centre, so a world point X has camera coordinates R (X - C).

**Back-projection** (`trace`) is closed form: the pinhole ray crosses each cylinder in turn and is refracted by
the vector form of Snell's law.  With `K` > 0 it also returns the tangents of the final ray (origin o on the
innermost interface, unit direction d) with respect to the first K of the inputs
    x, y (normalised undistorted image coordinates), w1, w2, w3 (rotation increment, R <- R exp([w]x)), C1, C2, C3,
by forward-mode differentiation of every step.

**Forward projection** (`project`) solves for the normalised, undistorted image coordinates (x, y) whose
back-projected ray passes through X, by Newton from a nearby start (the detection).  The residual is the offset
g = E^T (X - o) of X from the ray, in a basis E = [e1, e2] perpendicular to d.  Newton converges to the image
nearest its start, which is the one the detection belongs to: a point with several images (beyond the fold
radius) is still matched to the right one.  The pixel then follows in closed form, u = f xd + cx, v = f yd + cy
with (xd, yd) = distort(x, y).

**Derivatives of the forward projection** follow by implicit differentiation of g(x*, y*; theta) = 0:
d(x, y)/d theta = -g_xy^-1 g_theta, with g_theta = E^T (dX/d theta - do/d theta - s dd/d theta) at the solution,
s the depth of X along the ray.  The basis drops out there, so any perpendicular basis gives the same derivative
(SymPy checks in `tests/cylinder_wand/p0_sympy_derivations.py`, block 2).  The intrinsics enter only through the
closed-form pixel map, so their derivatives are exact: du/dlog f = f xd, du/dcx = 1, du/dk1 = f x r^2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# column order of the per-camera derivatives returned by `project`
CAM_PARAMS = ("w1", "w2", "w3", "C1", "C2", "C3", "logf", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
SMALLNUMBER = 1e-8          # the C++ ignores distortion when every coefficient is at most this (STBCommons.h)


# ------------------------------------------------------------------ rotations
def skew(v: np.ndarray) -> np.ndarray:
    """[v]x for one vector (3,) or a stack (n, 3) -> (3, 3) or (n, 3, 3)."""
    v = np.asarray(v, float)
    out = np.zeros(v.shape[:-1] + (3, 3))
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


def exp_so3(w: np.ndarray) -> np.ndarray:
    """exp([w]x), the rotation by |w| about w (Rodrigues)."""
    w = np.asarray(w, float).reshape(3)
    th = float(np.linalg.norm(w))
    K = skew(w)
    if th < 1e-8:                       # series: exact to round-off below 1e-8
        return np.eye(3) + K + 0.5 * (K @ K)
    return np.eye(3) + (math.sin(th) / th) * K + ((1.0 - math.cos(th)) / (th * th)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """The rotation vector w with exp([w]x) = R; accurate for small and near-pi angles."""
    R = np.asarray(R, float)
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(v))
    c = 0.5 * (float(np.trace(R)) - 1.0)
    th = math.atan2(s, c)
    if s < 1e-12 and c > 0:
        return v
    if th < 3.0:
        return v * (th / s)
    # near pi: the axis from the symmetric part
    B = 0.5 * (R + R.T) - c * np.eye(3)
    k = int(np.argmax(np.diag(B)))
    axis = B[:, k] / math.sqrt(max(B[k, k], 1e-300))
    if axis @ v < 0:
        axis = -axis
    return axis * th


def rvec_of(R: np.ndarray) -> np.ndarray:
    """OpenCV rotation vector of R."""
    if cv2 is not None:
        return cv2.Rodrigues(np.asarray(R, float))[0].ravel()
    return log_so3(R)


def rot_of(rvec) -> np.ndarray:
    return exp_so3(np.asarray(rvec, float))


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of Ra Rb^T in degrees, resolved to round-off for small angles."""
    m = np.asarray(Ra, float) @ np.asarray(Rb, float).T
    s = 0.5 * math.sqrt((m[2, 1] - m[1, 2]) ** 2 + (m[0, 2] - m[2, 0]) ** 2 + (m[1, 0] - m[0, 1]) ** 2)
    return math.degrees(math.atan2(s, 0.5 * (np.trace(m) - 1.0)))


# ------------------------------------------------------------------ tube and cameras
@dataclass(frozen=True)
class Tube:
    """Coaxial cylinders about the Y axis: radii R_1 > ... > R_{N-1} (mm), indices n_1 .. n_N with the pinhole's
    medium first, as the C++ `setRefraction` takes them."""

    radii: Tuple[float, ...]
    n: Tuple[float, ...]

    def __post_init__(self):
        radii = tuple(float(r) for r in self.radii)
        n = tuple(float(v) for v in self.n)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "n", n)
        if len(n) != len(radii) + 1 or not radii:
            raise ValueError("a tube needs N-1 radii for N media")
        if any(r <= 0 for r in radii) or any(a <= b for a, b in zip(radii, radii[1:])):
            raise ValueError("radii must be positive and strictly decreasing")
        if any(v <= 0 for v in n):
            raise ValueError("refractive indices must be positive")

    @property
    def r_inner(self) -> float:
        return self.radii[-1]

    @property
    def r_outer(self) -> float:
        return self.radii[0]

    def fold_radius(self) -> float:
        """r_c = min_{k<N} (n_k R_k) / n_N for rays square to the axis: beyond it a point can have two images
        or none (refract_curved.tex, far branch)."""
        return min(nk * rk for nk, rk in zip(self.n[:-1], self.radii)) / self.n[-1]

    def apparent_distance(self, D: float, depth: float = 0.0) -> float:
        """Air-equivalent distance along a radial line from a pinhole at distance D from the axis to a point at
        `depth` beyond the axis (negative: towards the camera).  It is the flat-window (paraxial) scale along the
        axis: image scale along the axis ~ f / apparent_distance."""
        n, R = self.n, self.radii
        out = (D - R[0]) * n[0] / n[0]
        for k in range(1, len(R)):
            out += (R[k - 1] - R[k]) * n[0] / n[k]
        return out + (R[-1] + depth) * n[0] / n[-1]


@dataclass
class Camera:
    """One PINCYL camera in the tube frame."""

    R: np.ndarray                     # (3, 3) world -> camera
    C: np.ndarray                     # (3,) centre, mm
    f: float                          # px (fx = fy)
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))   # k1, k2, p1, p2, k3

    def __post_init__(self):
        self.R = np.asarray(self.R, float).reshape(3, 3)
        self.C = np.asarray(self.C, float).reshape(3)
        self.dist = np.asarray(list(self.dist) + [0.0] * (5 - len(self.dist)), float)[:5]
        self.f, self.cx, self.cy = float(self.f), float(self.cx), float(self.cy)

    @property
    def tvec(self) -> np.ndarray:
        return -self.R @ self.C

    @property
    def rvec(self) -> np.ndarray:
        return rvec_of(self.R)

    def copy(self) -> "Camera":
        return Camera(self.R.copy(), self.C.copy(), self.f, self.cx, self.cy, self.width, self.height,
                      self.dist.copy())


@dataclass
class Rows:
    """Per-row camera parameters (row i belongs to some camera)."""

    R: np.ndarray        # (n, 3, 3)
    C: np.ndarray        # (n, 3)
    f: np.ndarray        # (n,)
    cx: np.ndarray       # (n,)
    cy: np.ndarray       # (n,)
    dist: np.ndarray     # (n, 5)

    def __len__(self) -> int:
        return len(self.f)

    def take(self, idx) -> "Rows":
        return Rows(self.R[idx], self.C[idx], self.f[idx], self.cx[idx], self.cy[idx], self.dist[idx])


def rows(cams: Sequence[Camera], idx) -> Rows:
    """Rows for camera indices `idx` (an int array)."""
    idx = np.asarray(idx, int)
    R = np.stack([c.R for c in cams])
    C = np.stack([c.C for c in cams])
    f = np.array([c.f for c in cams])
    cx = np.array([c.cx for c in cams])
    cy = np.array([c.cy for c in cams])
    dist = np.stack([c.dist for c in cams])
    return Rows(R[idx], C[idx], f[idx], cx[idx], cy[idx], dist[idx])


# ------------------------------------------------------------------ distortion
def distort(xy: np.ndarray, dist: np.ndarray, jac: bool = False):
    """OpenCV radial-tangential distortion of normalised coordinates, as the C++ `Camera::distort` (5 terms).

    Returns xd (n, 2), and with jac also d(xd)/d(xy) (n, 2, 2) and d(xd)/d(k1, k2, p1, p2, k3) (n, 2, 5)."""
    xy = np.asarray(xy, float)
    dist = np.broadcast_to(np.asarray(dist, float), (len(xy), 5))
    x, y = xy[:, 0], xy[:, 1]
    k1, k2, p1, p2, k3 = dist.T
    r2 = x * x + y * y
    c = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * c + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * c + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    out = np.column_stack([xd, yd])
    if not jac:
        return out
    dc = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)          # dc / d(r2)
    dcx, dcy = 2.0 * x * dc, 2.0 * y * dc
    J = np.empty((len(xy), 2, 2))
    J[:, 0, 0] = c + x * dcx + 2.0 * p1 * y + 6.0 * p2 * x
    J[:, 0, 1] = x * dcy + 2.0 * p1 * x + 2.0 * p2 * y
    J[:, 1, 0] = y * dcx + 2.0 * p1 * x + 2.0 * p2 * y
    J[:, 1, 1] = c + y * dcy + 6.0 * p1 * y + 2.0 * p2 * x
    Jk = np.empty((len(xy), 2, 5))
    Jk[:, 0, 0], Jk[:, 1, 0] = x * r2, y * r2
    Jk[:, 0, 1], Jk[:, 1, 1] = x * r2 * r2, y * r2 * r2
    Jk[:, 0, 2], Jk[:, 1, 2] = 2.0 * x * y, r2 + 2.0 * y * y
    Jk[:, 0, 3], Jk[:, 1, 3] = r2 + 2.0 * x * x, 2.0 * x * y
    Jk[:, 0, 4], Jk[:, 1, 4] = x * r2 ** 3, y * r2 ** 3
    return out, J, Jk


def undistort(uv: np.ndarray, rw: Rows, iters: int = 30) -> np.ndarray:
    """Normalised undistorted coordinates of pixels, by Newton on `distort` (to round-off, unlike the C++, which
    stops at 1e-5 px)."""
    uv = np.asarray(uv, float)
    target = np.column_stack([(uv[:, 0] - rw.cx) / rw.f, (uv[:, 1] - rw.cy) / rw.f])
    if not np.any(np.abs(rw.dist) > 0):
        return target
    xy = target.copy()
    for _ in range(iters):
        xd, J, _ = distort(xy, rw.dist, jac=True)
        res = xd - target
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        step = np.column_stack([(J[:, 1, 1] * res[:, 0] - J[:, 0, 1] * res[:, 1]) / det,
                                (-J[:, 1, 0] * res[:, 0] + J[:, 0, 0] * res[:, 1]) / det])
        xy -= step
        if np.max(np.abs(step)) < 1e-16:
            break
    return xy


def pixels(xy: np.ndarray, rw: Rows) -> np.ndarray:
    """Pixels of normalised undistorted coordinates."""
    xd = distort(xy, rw.dist)
    return np.column_stack([rw.f * xd[:, 0] + rw.cx, rw.f * xd[:, 1] + rw.cy])


# ------------------------------------------------------------------ back-projection
def trace(rw: Rows, xy: np.ndarray, tube: Tube, K: int = 0):
    """Back-project normalised undistorted coordinates through the tube.

    Returns (o, d, ok, do, dd): o (n, 3) the ray's point on the innermost interface, d (n, 3) its unit direction
    in the object medium, ok (n,) False where the ray misses an interface or is totally reflected, and for K > 0
    the tangents do, dd (n, 3, K) with respect to the first K of (x, y, w1, w2, w3, C1, C2, C3)."""
    xy = np.asarray(xy, float)
    n = len(xy)
    m = np.column_stack([xy[:, 0], xy[:, 1], np.ones(n)])
    v = np.einsum("nji,nj->ni", rw.R, m)                     # R^T m, the pinhole ray in the world
    nv = np.linalg.norm(v, axis=1)
    d = v / nv[:, None]
    o = rw.C.astype(float).copy()
    do = dd = None
    if K:
        dv = np.zeros((n, 3, K))
        dv[:, :, 0] = rw.R[:, 0, :]
        if K > 1:
            dv[:, :, 1] = rw.R[:, 1, :]
        if K > 2:
            # R(w) = R exp([w]x)  =>  R(w)^T m = exp(-[w]x) v = v + v x w  =>  d/dw = [v]x
            dv[:, :, 2:min(K, 5)] = skew(v)[:, :, :min(K, 5) - 2]
        dd = (dv - d[:, :, None] * np.einsum("ni,nik->nk", d, dv)[:, None, :]) / nv[:, None, None]
        do = np.zeros((n, 3, K))
        if K > 5:
            do[:, :, 5:K] = np.eye(3)[None, :, :K - 5]
    ok = np.ones(n, bool)
    for k, Rk in enumerate(tube.radii):
        ox, oz, dx, dz = o[:, 0], o[:, 2], d[:, 0], d[:, 2]
        A = dx * dx + dz * dz
        B = 2.0 * (ox * dx + oz * dz)
        Cq = ox * ox + oz * oz - Rk * Rk
        disc = B * B - 4.0 * A * Cq
        ok &= (disc >= 0.0) & (A > 0.0) & (B < 0.0) & (Cq > 0.0)
        sq = np.sqrt(np.maximum(disc, 0.0))
        q = 0.5 * (sq - B)                                    # > 0 for B < 0: the stable root pair
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ok, Cq / q, 0.0)                     # the entering (smaller) root
        h = o + t[:, None] * d
        hp = h.copy()
        hp[:, 1] = 0.0
        rh = np.linalg.norm(hp, axis=1)
        rh = np.where(rh > 0, rh, 1.0)
        N = hp / rh[:, None]                                  # outward normal
        r = tube.n[k] / tube.n[k + 1]
        cosi = -np.einsum("ij,ij->i", N, d)
        kk = 1.0 - r * r * (1.0 - cosi * cosi)
        ok &= kk >= 0.0
        ct = np.sqrt(np.maximum(kk, 0.0))
        dnew = r * d + (r * cosi - ct)[:, None] * N
        if K:
            hpd = np.einsum("ij,ij->i", hp, d)
            hpd = np.where(np.abs(hpd) > 0, hpd, 1.0)
            dt = -np.einsum("ni,nik->nk", hp, do + t[:, None, None] * dd) / hpd[:, None]
            dh = do + t[:, None, None] * dd + d[:, :, None] * dt[:, None, :]
            dN = dh.copy()
            dN[:, 1, :] = 0.0
            dN /= rh[:, None, None]
            dcos = -(np.einsum("nik,ni->nk", dN, d) + np.einsum("ni,nik->nk", N, dd))
            ctz = np.where(ct > 0, ct, 1.0)
            dct = (r * r) * cosi[:, None] * dcos / ctz[:, None]
            dd = (r * dd + N[:, :, None] * (r * dcos - dct)[:, None, :]
                  + (r * cosi - ct)[:, None, None] * dN)
            do = dh
        nd = np.linalg.norm(dnew, axis=1)
        d = dnew / np.where(nd > 0, nd, 1.0)[:, None]
        o = h
    return o, d, ok, do, dd


def _perp_basis(d: np.ndarray, ref: np.ndarray):
    """Orthonormal e1, e2 perpendicular to the unit rows of d; e1 is `ref` (rows) made perpendicular."""
    e1 = ref - np.einsum("ij,ij->i", ref, d)[:, None] * d
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    return e1, e2


def ray_offsets(X: np.ndarray, o: np.ndarray, d: np.ndarray, ref: np.ndarray):
    """(g (n, 2), s (n,), e1, e2): the offset of X from each ray in a perpendicular basis, and its depth s."""
    w = X - o
    s = np.einsum("ij,ij->i", w, d)
    e1, e2 = _perp_basis(d, ref)
    g = np.column_stack([np.einsum("ij,ij->i", w, e1), np.einsum("ij,ij->i", w, e2)])
    return g, s, e1, e2


# ------------------------------------------------------------------ forward projection
@dataclass
class Projection:
    uv: np.ndarray                    # (n, 2) pixels (NaN where not ok)
    xy: np.ndarray                    # (n, 2) normalised undistorted coordinates
    ok: np.ndarray                    # (n,)
    iters: int = 0
    J_cam: Optional[np.ndarray] = None    # (n, 2, 14) d uv / d CAM_PARAMS
    J_X: Optional[np.ndarray] = None      # (n, 2, 3)  d uv / d X
    det: Optional[np.ndarray] = None      # (n,) det of g_xy, scaled: small near a fold (caustic)


def project(rw: Rows, X: np.ndarray, xy0: np.ndarray, tube: Tube, jac: bool = False, max_iter: int = 40,
            tol: float = 1e-13, max_step: float = 0.05, gtol: float = 1e-10) -> Projection:
    """Image of each world point X nearest the start xy0 (normalised undistorted), by Newton on the ray offset.

    A row is ok when Newton converged (|step| < 1e-9 in normalised units) and the ray at the result is valid and
    passes within gtol * |X - C| of X.  With jac, the derivatives with respect to the camera (`CAM_PARAMS` order)
    and to X follow by implicit differentiation."""
    X = np.asarray(X, float)
    n = len(X)
    xy = _valid_start(rw, X, np.array(xy0, float, copy=True), tube)
    xy_prev = xy.copy()
    step_prev = np.zeros((n, 2))
    gn_prev = np.full(n, np.inf)
    halvings = np.zeros(n, int)
    last = np.full(n, np.inf)
    ok = np.ones(n, bool)
    active = np.arange(n)
    it = 0
    for it in range(1, max_iter + 1):
        sub = rw.take(active)
        o, d, good, do, dd = trace(sub, xy[active], tube, K=2)
        g, s, e1, e2 = ray_offsets(X[active], o, d, sub.R[:, 0, :])
        gn = np.linalg.norm(g, axis=1)
        # backtracking: a step that loses the ray or increases the offset is halved (the merit is |g|; the
        # 1e-12 mm slack keeps round-off near the solution from triggering it)
        worse = ~good | ~np.isfinite(gn) | (gn > gn_prev[active] * (1.0 + 1e-6) + 1e-12)
        back = worse & np.isfinite(gn_prev[active]) & (halvings[active] < 40)
        if np.any(back):
            b = active[back]
            step_prev[b] *= 0.5
            xy[b] = xy_prev[b] + step_prev[b]
            halvings[b] += 1
        fail = worse & ~back
        ok[active[fail]] = False
        acc = ~worse
        G = -(do[acc] + s[acc, None, None] * dd[acc])         # (m, 3, 2)
        J = np.stack([np.einsum("ni,nik->nk", e1[acc], G), np.einsum("ni,nik->nk", e2[acc], G)], axis=1)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        sing = ~(np.isfinite(det) & (np.abs(det) > 0))
        dets = np.where(sing, 1.0, det)
        ga = g[acc]
        step = -np.column_stack([(J[:, 1, 1] * ga[:, 0] - J[:, 0, 1] * ga[:, 1]) / dets,
                                 (-J[:, 1, 0] * ga[:, 0] + J[:, 0, 0] * ga[:, 1]) / dets])
        sn = np.linalg.norm(step, axis=1)
        big = sn > max_step
        step[big] *= (max_step / sn[big])[:, None]
        a = active[acc]
        ok[a[sing]] = False
        xy_prev[a] = xy[a]
        gn_prev[a] = gn[acc]
        step_prev[a] = step
        halvings[a] = 0
        xy[a] += np.where(sing[:, None], 0.0, step)
        last[a] = np.where(sing, np.inf, sn)
        done = np.zeros(len(active), bool)
        done[np.flatnonzero(acc)] = sing | (sn <= tol)
        done |= fail
        active = active[~done]
        if active.size == 0:
            break
    ok &= last < 1e-9
    out = Projection(uv=np.full((n, 2), np.nan), xy=xy, ok=ok, iters=it)
    if not np.any(ok):
        return out
    # a small Newton step is not enough: at a grazing edge the ray's derivatives blow up and the step shrinks to
    # nothing with X still off the ray.  The ray at the final coordinates must be valid and pass through X.
    idx = np.flatnonzero(ok)
    sub = rw.take(idx)
    o, d, good, do, dd = trace(sub, xy[idx], tube, K=8 if jac else 0)
    g, s, e1, e2 = ray_offsets(X[idx], o, d, sub.R[:, 0, :])
    fine = good & (np.linalg.norm(g, axis=1) <= gtol * np.linalg.norm(X[idx] - sub.C, axis=1))
    ok[idx[~fine]] = False
    if not np.all(fine):
        idx, sub = idx[fine], sub.take(np.flatnonzero(fine))
        o, d, g, s, e1, e2 = o[fine], d[fine], g[fine], s[fine], e1[fine], e2[fine]
        if jac:
            do, dd = do[fine], dd[fine]
    if idx.size == 0:
        return out
    if not jac:
        out.uv[idx] = pixels(xy[idx], sub)
        return out
    G = -(do + s[:, None, None] * dd)                         # dg/dtheta = E^T (dX - do - s dd), (m, 3, 8)
    E = np.stack([e1, e2], axis=1)                           # (m, 2, 3)
    Gt = np.einsum("nai,nik->nak", E, G)                     # (m, 2, 8)
    Jxy, Jth = Gt[:, :, :2], Gt[:, :, 2:]
    det = Jxy[:, 0, 0] * Jxy[:, 1, 1] - Jxy[:, 0, 1] * Jxy[:, 1, 0]
    inv = np.empty_like(Jxy)
    inv[:, 0, 0], inv[:, 1, 1] = Jxy[:, 1, 1] / det, Jxy[:, 0, 0] / det
    inv[:, 0, 1], inv[:, 1, 0] = -Jxy[:, 0, 1] / det, -Jxy[:, 1, 0] / det
    dxy_dth = -np.einsum("nab,nbk->nak", inv, Jth)            # (m, 2, 6): w1 w2 w3 C1 C2 C3
    dxy_dX = -np.einsum("nab,nbi->nai", inv, E)               # (m, 2, 3)
    xd, Jd, Jk = distort(xy[idx], sub.dist, jac=True)
    fJd = sub.f[:, None, None] * Jd
    Jc = np.zeros((len(idx), 2, len(CAM_PARAMS)))
    Jc[:, :, 0:6] = np.einsum("nab,nbk->nak", fJd, dxy_dth)
    Jc[:, :, 6] = sub.f[:, None] * xd
    Jc[:, 0, 7] = 1.0
    Jc[:, 1, 8] = 1.0
    Jc[:, :, 9:14] = sub.f[:, None, None] * Jk
    out.uv[idx] = np.column_stack([sub.f * xd[:, 0] + sub.cx, sub.f * xd[:, 1] + sub.cy])
    out.J_cam = np.full((n, 2, len(CAM_PARAMS)), np.nan)
    out.J_cam[idx] = Jc
    out.J_X = np.full((n, 2, 3), np.nan)
    out.J_X[idx] = np.einsum("nab,nbi->nai", fJd, dxy_dX)
    # |det g_xy| relative to the pinhole-like scale (path length)^2: it vanishes at a fold (caustic)
    path = np.linalg.norm(o - sub.C, axis=1) + np.abs(s)
    out.det = np.full(n, np.nan)
    out.det[idx] = np.abs(det) / np.maximum(path * path, 1e-300)
    return out


def pinhole_start(rw: Rows, X: np.ndarray) -> np.ndarray:
    """Normalised coordinates of X for a pinhole without the tube: a crude start for `project`."""
    pc = np.einsum("nij,nj->ni", rw.R, np.asarray(X, float) - rw.C)
    return pc[:, :2] / pc[:, 2:3]


def _valid_start(rw: Rows, X: np.ndarray, xy: np.ndarray, tube: Tube, iters: int = 50,
                 inside: float = 0.02) -> np.ndarray:
    """Starts whose ray misses the tube or is totally reflected (a detection just outside the tube's image under
    the current estimate, e.g. at a grazing edge) are moved onto the segment towards a valid target: the pinhole
    start of X, or else the pinhole start of the axis point at X's height (a ray aimed at the axis meets every
    wall square-on in the cross-section, so it neither grazes nor reflects).  The boundary is found by bisection
    and the start placed a fraction `inside` of the way past it, off the edge where the ray's derivatives blow
    up, so Newton still begins next to the detection."""
    _o, _d, ok, _, _ = trace(rw, xy, tube)
    bad = np.flatnonzero(~ok)
    if bad.size == 0:
        return xy
    sub = rw.take(bad)
    Xb = np.asarray(X, float)[bad]
    far = pinhole_start(sub, Xb)
    _o, _d, okf, _, _ = trace(sub, far, tube)
    ax = Xb.copy()
    ax[:, 0] = 0.0
    ax[:, 2] = 0.0
    far_ax = pinhole_start(sub, ax)
    _o, _d, oka, _, _ = trace(sub, far_ax, tube)
    use_ax = ~okf & oka
    far[use_ax] = far_ax[use_ax]
    valid = okf | oka
    lo = xy[bad].copy()                  # invalid end
    hi = far                             # valid end
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _o, _d, okm, _, _ = trace(sub, mid, tube)
        hi = np.where(okm[:, None], mid, hi)
        lo = np.where(okm[:, None], lo, mid)
    start = hi + inside * (far - hi)
    _o, _d, oks, _, _ = trace(sub, start, tube)
    start = np.where(oks[:, None], start, hi)
    out = xy.copy()
    out[bad[valid]] = start[valid]
    return out


# ------------------------------------------------------------------ frames and the C++ model
def axis_rotation(axis_dir) -> np.ndarray:
    """A rotation Q with Q @ axis_dir = (0, 1, 0): rows (ex', axis, ez')."""
    a = np.asarray(axis_dir, float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ez = ref - (ref @ a) * a
    ez /= np.linalg.norm(ez)
    ex = np.cross(a, ez)
    return np.vstack([ex, a, ez])


def to_tube_frame(R: np.ndarray, C: np.ndarray, axis_pt, axis_dir):
    """A camera (world -> camera R, centre C) in the tube frame of a port with axis (axis_pt, axis_dir).

    Returns (R_t, C_t, Q, p): X_t = Q (X - p) maps world points into the tube frame."""
    Q = axis_rotation(axis_dir)
    p = np.asarray(axis_pt, float)
    return np.asarray(R, float) @ Q.T, Q @ (np.asarray(C, float) - p), Q, p


def import_lpt():
    import pyopenlpt as lpt  # noqa: WPS433 (deferred: the fast model itself does not need the extension)
    return lpt


def to_cpp(cam: Camera, tube: Tube, tol: float = 1e-12, nmax: int = 200, axis_pt=(0.0, 0.0, 0.0),
           axis_dir=(0.0, 1.0, 0.0), R_world: np.ndarray = None, C_world: np.ndarray = None):
    """The C++ PINCYL camera of `cam` (tube frame by default; pass the world pose and axis for another frame)."""
    lpt = import_lpt()
    R = cam.R if R_world is None else np.asarray(R_world, float)
    C = cam.C if C_world is None else np.asarray(C_world, float)
    c = lpt.RefractionCylinderCameraModel()
    c.setImageSize(int(cam.height), int(cam.width))
    c.setIntrinsics(float(cam.f), float(cam.f), float(cam.cx), float(cam.cy), [float(v) for v in cam.dist])
    c.setExtrinsics(lpt.Pt3D(*map(float, rvec_of(R))), lpt.Pt3D(*map(float, -R @ C)))
    c.setRefraction(lpt.Pt3D(*map(float, axis_pt)), lpt.Pt3D(*map(float, axis_dir)), list(tube.n),
                    list(tube.radii))
    c.setSolverOptions(float(tol), int(nmax))
    c.commitUpdate()
    return c


def cpp_project(cam_cpp, X: np.ndarray) -> np.ndarray:
    """The C++ primary image of each point, (n, 2); NaN where it has none."""
    lpt = import_lpt()
    res = cam_cpp.projectBatchStatus([lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])) for p in X], False)
    out = np.full((len(X), 2), np.nan)
    for k, (good, pt, _msg) in enumerate(res):
        if good:
            out[k] = (pt[0], pt[1])
    return out


def cpp_images(cam_cpp, X: np.ndarray) -> List[np.ndarray]:
    """Every (non-ring) image of each point, a list of (m, 2) arrays."""
    lpt = import_lpt()
    out = []
    for p in X:
        rays, _cov = cam_cpp.projectAll(lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])))
        out.append(np.array([[r["pt"][0], r["pt"][1]] for r in rays if not r["is_ring"]], float).reshape(-1, 2))
    return out


def cpp_rays(cam_cpp, uv: np.ndarray):
    """The C++ lines of sight of pixels: (origins, unit directions, ok)."""
    lpt = import_lpt()
    res = cam_cpp.lineOfSightBatchStatus([lpt.Pt2D(float(a), float(b)) for a, b in uv])
    o = np.full((len(uv), 3), np.nan)
    d = np.full((len(uv), 3), np.nan)
    ok = np.zeros(len(uv), bool)
    for k, (good, line, _msg) in enumerate(res):
        if good:
            o[k] = (line.pt[0], line.pt[1], line.pt[2])
            d[k] = (line.unit_vector[0], line.unit_vector[1], line.unit_vector[2])
            ok[k] = bool(np.all(np.isfinite(o[k])) and np.linalg.norm(d[k]) > 0)
    return o, d, ok
