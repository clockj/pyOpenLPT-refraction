"""The cylinder-port camera of this case: its parameters, residuals and least-squares fit.

The projection itself is pyOpenLPT's C++ `RefractionCylinderCameraModel` (PINCYL); this module only builds it from a
parameter set and fits that set to the detected dots.

World frame: the board at z = 0, X along the rows (image right), Y along the columns (image up), Z towards the
camera.  Dot (i, j) of plane k sits at (px i + tau_x z_k, py j + tau_y z_k, z_k) with z_k = s z_nominal_k + dz_k
(dz_0 = 0); tau is the traverse tilt and s the stage scale (0 and 1 unless fitted).

Parameters, as increments about an anchor rig (re-anchored after every round, so each increment stays small):
    rot   (3)  camera rotation, R = R_anchor exp([w]x)  (world -> camera, as OpenCV)
    cen   (3)  camera centre, C = C_anchor + dC          (mm)
    logf  (1)  f = f_anchor exp(df)                      (fx = fy; principal point fixed)
    axdir (2)  axis direction, d = normalise(d_anchor + a e1 + b e2): a tilts it towards +X (within the board
               plane), b towards +Z (leaning towards the camera)
    axpt  (2)  axis position, p = p_anchor + g e1 + h e2 (moves the line along e1 ~ X and e2 ~ Z)
    dz    (K-1) plane offsets dz_1..dz_{K-1}               (mm)
    zs    (1)  stage scale s = s_anchor + ds              (the stage steps stretched by s)
    tau   (2)  traverse tilt                              (lateral mm per mm of z)
    shift (2(K-2)) sideways shift of each plane's board (stage play), no mean and no trend in z   (mm)

Residuals:
    forward      the reprojection error in pixels through the C++ projection (what is minimised and reported);
    backproject  the offset of each world point from its pixel's back-projected ray, scaled to pixels by
                 f / (optical path from the camera) -- fast and always defined, used for the multi-start.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

import case

lpt = case.import_lpt()

BLOCKS = ("rot", "cen", "logf", "axdir", "axpt", "axptz", "dz", "zs", "tau", "dist", "shift", "board", "boardcr",
          "wall")
SIZES = {"rot": 3, "cen": 3, "logf": 1, "axdir": 2, "axpt": 2, "axptz": 1, "zs": 1, "tau": 2, "dist": 1}
STEPS = {"rot": 1e-6, "cen": 1e-3, "logf": 1e-5, "axdir": 1e-5, "axpt": 1e-3, "axptz": 1e-3, "dz": 1e-3,
         "zs": 1e-5, "tau": 1e-5, "dist": 1e-2, "shift": 1e-4, "board": 1e-4, "boardcr": 1e-4, "wall": 1e-6}
WALL_SPEC = (0.0, 1280.0, 80.0)          # knots of the wall-deflection spline in u: lo, hi, spacing (px)
PENALTY_PX = 50.0                       # residual of a point that has no image (never seen at a converged fit)

# M1*: the planes held at their stage positions (the delivered family since the stage steps are known, 2.54 mm);
# M2*: every plane's position free (the first run's family, when the steps were thought to be 2 mm)
MODELS = {
    "M1": ("rot", "cen", "logf", "axdir", "axpt"),
    "M1b": ("rot", "cen", "logf", "axdir", "axpt", "board"),
    "M1c": ("rot", "cen", "logf", "axdir", "axpt", "boardcr"),
    "M1bs": ("rot", "cen", "logf", "axdir", "axpt", "zs", "board"),
    "M1bt": ("rot", "cen", "logf", "axdir", "axpt", "tau", "board"),
    "M1bk": ("rot", "cen", "logf", "axdir", "axpt", "dist", "board"),
    "M1bw": ("rot", "cen", "logf", "axdir", "axpt", "board", "wall"),
    "M1bx": ("rot", "cen", "logf", "axdir", "axpt", "shift", "board"),
    "M1bxw": ("rot", "cen", "logf", "axdir", "axpt", "shift", "board", "wall"),
    "M1x": ("rot", "cen", "logf", "axdir", "axpt", "shift"),
    "M1b_fixf": ("rot", "cen", "axdir", "axpt", "board"),
    "M2": ("rot", "cen", "logf", "axdir", "axpt", "dz"),
    "M3": ("rot", "cen", "logf", "axdir", "axpt", "dz", "tau"),
    "M2_fixf": ("rot", "cen", "axdir", "axpt", "dz"),
    "M1_fixf": ("rot", "cen", "axdir", "axpt"),
    "M2k": ("rot", "cen", "logf", "axdir", "axpt", "dz", "dist"),
    "M2b": ("rot", "cen", "logf", "axdir", "axpt", "dz", "board"),
    "M2c": ("rot", "cen", "logf", "axdir", "axpt", "dz", "boardcr"),
    "M2bw": ("rot", "cen", "logf", "axdir", "axpt", "dz", "board", "wall"),
    "M2w": ("rot", "cen", "logf", "axdir", "axpt", "dz", "wall"),
}


# ------------------------------------------------------------------ bases of the optional blocks
def bspline_basis(x, lo: float, hi: float, step: float) -> np.ndarray:
    """Cubic B-splines on uniform knots covering [lo, hi]; a fixed set of columns for any x.

    Basis q starts at knot lo + (q - 3) step and spans four intervals; in the interval of x only q = k-3..k are
    nonzero, with the uniform cubic B-spline weights (1-t)^3/6, (3t^3-6t^2+4)/6, (-3t^3+3t^2+3t+1)/6, t^3/6.
    """
    n = len(np.arange(lo - 3 * step, hi + 3.5 * step, step)) - 4
    x = np.asarray(x, float)
    s = (x - (lo - 3 * step)) / step
    k = np.floor(s).astype(int)
    t = s - k
    w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
    B = np.zeros((len(x), n))
    rows = np.arange(len(x))
    for m in range(4):
        col = k - 3 + m
        ok = (col >= 0) & (col < n)
        B[rows[ok], col[ok]] = w[m][ok]
    return B


def board_basis(board_ij: np.ndarray, pitch) -> np.ndarray:
    """Orthonormal basis of the per-dot corrections [dX_0, dY_0, dX_1, ...] with no affine part.

    The removed part (shift, rotation, scale and shear of the whole board) cannot be told from the camera pose,
    the focal length or the tube's magnification, so the nominal 3 mm grid keeps defining it.
    """
    X = pitch[0] * board_ij[:, 0].astype(float)
    Y = pitch[1] * board_ij[:, 1].astype(float)
    n = len(X)
    M = np.zeros((2 * n, 6))
    M[0::2, 0] = 1
    M[1::2, 1] = 1
    M[0::2, 2] = X
    M[0::2, 3] = Y
    M[1::2, 4] = X
    M[1::2, 5] = Y
    U, _s, _vt = np.linalg.svd(M, full_matrices=True)
    return U[:, 6:]


def board_basis_colrow(board_ij: np.ndarray, pitch) -> np.ndarray:
    """Map from column/row offsets to per-dot corrections [dX_0, dY_0, dX_1, ...]: every dot of column i moves
    by the same dX_i and every dot of row j by the same dY_j (an engraver's axis errors), with each set's mean and
    linear trend removed (they trade with the camera and the magnification)."""
    out = []
    for c, p in ((0, pitch[0]), (1, pitch[1])):
        vals = np.unique(board_ij[:, c])
        pos = p * vals.astype(float)
        A = np.column_stack([np.ones_like(pos), pos]).T          # 2 x n_vals
        _u, _s, vt = np.linalg.svd(A, full_matrices=True)
        N = vt[2:].T                                            # n_vals x (n_vals - 2)
        M = np.zeros((2 * len(board_ij), N.shape[1]))
        idx = np.searchsorted(vals, board_ij[:, c])
        M[c::2] = N[idx]
        out.append(M)
    return np.hstack(out)


def shift_basis(z) -> np.ndarray:
    """Orthonormal basis (K x (K-2)) of per-plane sideways shifts with no mean and no linear trend in z.

    The stage carriage moves the board a little sideways at every step (tests/d8).  A common shift of all planes
    cannot be told from the camera's lateral position, and a shift growing linearly with z is a traverse tilt
    (which the data cannot fix either), so both stay out: the board keeps moving square to itself on average.
    """
    z = np.asarray(z, float)
    A = np.column_stack([np.ones_like(z), z - z.mean()]).T
    _u, _s, vt = np.linalg.svd(A, full_matrices=True)
    return vt[2:].T


def wall_basis_null(spec) -> np.ndarray:
    """Coefficient combinations of the wall spline with no constant, linear or quadratic part in u (those trade
    with the camera, the axis and the tube's power)."""
    lo, hi, step = spec
    us = np.arange(lo, hi + 1e-9, 4.0)
    B = bspline_basis(us, lo, hi, step)
    t = (us - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    Q = np.column_stack([np.ones_like(t), t, t * t])
    A = Q.T @ B
    _u, _s, vt = np.linalg.svd(A, full_matrices=True)
    return vt[3:].T


# ------------------------------------------------------------------ rotations
def rodrigues(w) -> np.ndarray:
    w = np.asarray(w, float).reshape(3)
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)


def rvec_of(R) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def look_at(C, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """World -> camera rotation of a camera at C looking at `target`, image up along `up` (OpenCV axes)."""
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))       # image right ... then y = z x x points image-down
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def perp_basis(d) -> tuple:
    """(e1 ~ +X, e2) spanning the plane perpendicular to the unit vector d, with e2 = e1 x d ~ +Z for d ~ +Y."""
    d = np.asarray(d, float)
    e1 = np.array([1.0, 0.0, 0.0]) - d[0] * d
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, d)
    return e1, e2


# ------------------------------------------------------------------ data
@dataclass
class Obs:
    """The observations: one row per usable dot."""
    plane: np.ndarray          # (n,) plane index k
    i: np.ndarray              # (n,) column index
    j: np.ndarray              # (n,) row index (up)
    uv: np.ndarray             # (n, 2) detected pixel
    z_nom: np.ndarray          # (K,) nominal z of each plane
    pitch: tuple               # (px, py) mm
    plane_ids: np.ndarray = None   # (K,) original plane numbers (when only some planes are used)

    def subset(self, mask) -> "Obs":
        return Obs(self.plane[mask], self.i[mask], self.j[mask], self.uv[mask], self.z_nom, self.pitch,
                   self.plane_ids)

    @property
    def n_planes(self) -> int:
        return len(self.z_nom)

    def dots(self):
        """(index of each row's dot, (n_dots, 2) table of the distinct (i, j))."""
        if getattr(self, "_dots", None) is None:
            ij = np.column_stack([self.i, self.j]).astype(int)
            uniq, inv = np.unique(ij, axis=0, return_inverse=True)
            self._dots = (inv.ravel(), uniq)
        return self._dots


@dataclass
class Rig:
    R: np.ndarray
    C: np.ndarray
    f: float
    cx: float
    cy: float
    axis_pt: np.ndarray
    axis_dir: np.ndarray
    dz: np.ndarray
    tau: np.ndarray
    n: list
    radii: list
    width: int = 1280
    height: int = 800
    dist: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # optional: per-dot board corrections (mm), and a ray deflection in the object medium that depends on the
    # pixel column (rad; cubic B-spline in u on uniform knots wall_spec = (lo, hi, step)); neither is part of
    # pyOpenLPT's PINCYL camera -- the board corrections belong to the target, the wall field is diagnostic
    board_ij: np.ndarray = None
    board: np.ndarray = None
    wall_spec: tuple = None
    wall: np.ndarray = None
    zscale: float = 1.0               # stage scale: plane k at zscale * z_nominal_k + dz_k
    shift: np.ndarray = None          # (K, 2) sideways shift (dX, dY) of each plane's board, mm (stage play)

    @property
    def tvec(self) -> np.ndarray:
        return -self.R @ self.C

    def has_wall(self) -> bool:
        return self.wall is not None and np.any(self.wall != 0)

    def board_offsets(self, obs: "Obs") -> np.ndarray:
        """Board correction (dX, dY) of each observation row (zeros for dots without one)."""
        idx, uniq = obs.dots()
        off = np.zeros((len(uniq), 2))
        if self.board is not None:
            lut = {(int(a), int(b)): q for q, (a, b) in enumerate(self.board_ij)}
            for q, (a, b) in enumerate(uniq):
                k = lut.get((int(a), int(b)))
                if k is not None:
                    off[q] = self.board[k]
        return off[idx]

    def copy(self) -> "Rig":
        return copy.deepcopy(self)

    def camera(self, tol: float = 1e-10, nmax: int = 100):
        cam = lpt.RefractionCylinderCameraModel()
        cam.setImageSize(int(self.height), int(self.width))
        cam.setIntrinsics(float(self.f), float(self.f), float(self.cx), float(self.cy), list(self.dist))
        cam.setExtrinsics(lpt.Pt3D(*rvec_of(self.R)), lpt.Pt3D(*self.tvec))
        cam.setRefraction(lpt.Pt3D(*np.asarray(self.axis_pt, float)), lpt.Pt3D(*np.asarray(self.axis_dir, float)),
                          [float(v) for v in self.n], [float(v) for v in self.radii])
        cam.setSolverOptions(tol, nmax)
        cam.commitUpdate()
        return cam

    def plane_z(self, z_nom) -> np.ndarray:
        return np.asarray(z_nom, float) * self.zscale + self.dz

    def world_points(self, obs: Obs) -> np.ndarray:
        z = self.plane_z(obs.z_nom)[obs.plane]
        X = obs.pitch[0] * obs.i + self.tau[0] * z
        Y = obs.pitch[1] * obs.j + self.tau[1] * z
        if self.board is not None:
            off = self.board_offsets(obs)
            X, Y = X + off[:, 0], Y + off[:, 1]
        if self.shift is not None:
            X, Y = X + self.shift[obs.plane, 0], Y + self.shift[obs.plane, 1]
        return np.column_stack([X, Y, z])

    def axis_at_y0(self) -> np.ndarray:
        """The axis point where the axis crosses Y = 0."""
        d, p = self.axis_dir, self.axis_pt
        return p - (p[1] / d[1]) * d

    def summary(self) -> dict:
        d = self.axis_dir
        a0 = self.axis_at_y0()
        view = self.R[2]                                 # camera z axis (viewing direction) in world
        # distance of the camera centre from the axis line
        w = self.C - a0
        r_cam = float(np.linalg.norm(w - (w @ d) * d))
        return {
            "f_px": float(self.f), "f_mm": float(self.f) * 0.020,
            "camera_centre_mm": self.C.tolist(), "rvec": rvec_of(self.R).tolist(), "tvec": self.tvec.tolist(),
            "viewing_direction": view.tolist(),
            "view_tilt_from_minusZ_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, -view[2]))))),
            "axis_point_at_Y0_mm": a0.tolist(), "axis_direction": d.tolist(),
            "axis_tilt_in_board_plane_deg": float(math.degrees(math.atan2(d[0], d[1]))),
            "axis_lean_towards_camera_deg": float(math.degrees(math.atan2(d[2], d[1]))),
            "camera_distance_from_axis_mm": r_cam,
            "dz_mm": self.dz.tolist(), "tau": self.tau.tolist(), "stage_scale": float(self.zscale),
        }


# ------------------------------------------------------------------ parameter vector
class Layout:
    """Which blocks are free, and where each sits in the parameter vector."""

    def __init__(self, blocks, n_planes: int, rig: "Rig" = None, pitch=None, z=None):
        self.blocks = tuple(b for b in BLOCKS if b in blocks)
        self.n_planes = n_planes
        self.slices, k = {}, 0
        for b in self.blocks:
            if b == "dz":
                n = n_planes - 1
            elif b == "shift":
                if z is None:
                    raise ValueError("the shift block needs the plane positions (Layout(..., z=obs.z_nom))")
                self.Ns = shift_basis(z)
                n = 2 * self.Ns.shape[1]
            elif b == "board":
                self.Nb = board_basis(rig.board_ij, pitch)
                n = self.Nb.shape[1]
            elif b == "boardcr":
                self.Nb = board_basis_colrow(rig.board_ij, pitch)
                n = self.Nb.shape[1]
            elif b == "wall":
                self.Nw = wall_basis_null(rig.wall_spec)
                n = self.Nw.shape[1]
            else:
                n = SIZES[b]
            self.slices[b] = slice(k, k + n)
            k += n
        self.size = k

    def names(self) -> list:
        out = []
        for b in self.blocks:
            n = self.slices[b].stop - self.slices[b].start
            out += [f"{b}[{q + (1 if b == 'dz' else 0)}]" for q in range(n)]
        return out

    def steps(self) -> np.ndarray:
        h = np.zeros(self.size)
        for b in self.blocks:
            h[self.slices[b]] = STEPS[b]
        return h

    def apply(self, x: np.ndarray, anchor: Rig) -> Rig:
        r = anchor.copy()
        if "rot" in self.blocks:
            r.R = anchor.R @ rodrigues(x[self.slices["rot"]])
        if "cen" in self.blocks:
            r.C = anchor.C + x[self.slices["cen"]]
        if "logf" in self.blocks:
            r.f = anchor.f * math.exp(float(x[self.slices["logf"]][0]))
        e1, e2 = perp_basis(anchor.axis_dir)
        if "axdir" in self.blocks:
            a, b = x[self.slices["axdir"]]
            d = anchor.axis_dir + a * e1 + b * e2
            r.axis_dir = d / np.linalg.norm(d)
        if "axpt" in self.blocks:
            g, h = x[self.slices["axpt"]]
            r.axis_pt = anchor.axis_pt + g * e1 + h * e2
        if "axptz" in self.blocks:                     # depth only: the lateral position X_a held
            r.axis_pt = anchor.axis_pt + x[self.slices["axptz"]][0] * e2
        if "dz" in self.blocks:
            r.dz = anchor.dz.copy()
            r.dz[1:] += x[self.slices["dz"]]
        if "zs" in self.blocks:
            r.zscale = float(anchor.zscale + x[self.slices["zs"]][0])
        if "tau" in self.blocks:
            r.tau = anchor.tau + x[self.slices["tau"]]
        if "dist" in self.blocks:
            r.dist = list(anchor.dist)
            r.dist[0] = float(anchor.dist[0] + x[self.slices["dist"]][0])     # k1 (radial, normalised coords)
        if "shift" in self.blocks:
            m = self.Ns.shape[1]
            xs = x[self.slices["shift"]]
            r.shift = anchor.shift + np.column_stack([self.Ns @ xs[:m], self.Ns @ xs[m:]])
        for b in ("board", "boardcr"):
            if b in self.blocks:
                r.board = anchor.board + (self.Nb @ x[self.slices[b]]).reshape(-1, 2)
        if "wall" in self.blocks:
            r.wall = anchor.wall + self.Nw @ x[self.slices["wall"]]
        return r


def prepare(rig: "Rig", obs: "Obs", blocks) -> "Rig":
    """Give the rig the tables the optional blocks need: board corrections for exactly the observed dots
    (carrying over any it has), wall coefficients, and plane shifts (any it has projected onto the shift basis of
    these planes, so that a start from another plane set keeps no common shift or traverse tilt in them)."""
    r = rig.copy()
    if "shift" in blocks:
        if r.shift is None or len(r.shift) != obs.n_planes:
            r.shift = np.zeros((obs.n_planes, 2))
        Ns = shift_basis(obs.z_nom)
        r.shift = Ns @ (Ns.T @ np.asarray(r.shift, float))
    if "board" in blocks or "boardcr" in blocks:
        _idx, uniq = obs.dots()
        old = {}
        if r.board is not None:
            old = {(int(a), int(b)): r.board[q] for q, (a, b) in enumerate(r.board_ij)}
        r.board_ij = uniq.copy()
        r.board = np.array([old.get((int(a), int(b)), np.zeros(2)) for a, b in uniq], float).reshape(-1, 2)
    if "wall" in blocks and r.wall is None:
        r.wall_spec = WALL_SPEC
        r.wall = np.zeros(bspline_basis(np.zeros(1), *WALL_SPEC).shape[1])
    return r


def restrict(rig: "Rig", blocks) -> "Rig":
    """The rig with every optional correction that `blocks` does not fit reset to its neutral value: plane offsets
    0, stage scale 1, traverse tilt 0, k1 0, no plane shifts, no board corrections, no wall field.  A start taken
    from another model's fit must not carry corrections that the new model would then hold fixed.  The column/row
    board model (boardcr) starts from the nominal grid: per-dot corrections from another fit lie mostly outside
    its subspace and would stay in as fixed offsets."""
    r = rig.copy()
    if "dz" not in blocks:
        r.dz = np.zeros_like(np.asarray(r.dz, float))
    if "zs" not in blocks:
        r.zscale = 1.0
    if "tau" not in blocks:
        r.tau = np.zeros(2)
    if "dist" not in blocks:
        r.dist = [0.0] * len(r.dist)
    if "board" not in blocks:
        r.board, r.board_ij = None, None
    if "shift" not in blocks:
        r.shift = None
    if "wall" not in blocks:
        r.wall, r.wall_spec = None, None
    return r


def reanchor(r: Rig) -> Rig:
    """Normalise the representation without changing the model: the axis point moved to Y = 0."""
    r = r.copy()
    r.axis_dir = r.axis_dir / np.linalg.norm(r.axis_dir)
    if r.axis_dir[1] < 0:
        r.axis_dir = -r.axis_dir
    r.axis_pt = r.axis_at_y0()
    return r


# ------------------------------------------------------------------ projection
def project(cam, P: np.ndarray) -> np.ndarray:
    """Primary image of each world point, (n, 2); NaN where the point has no image."""
    res = cam.projectBatchStatus([lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])) for p in P], False)
    uv = np.full((len(P), 2), np.nan)
    for k, (ok, pt, _msg) in enumerate(res):
        if ok:
            uv[k] = (pt[0], pt[1])
    return uv


def project_all(cam, P: np.ndarray) -> list:
    """Every image of each world point: a list of (m, 2) arrays."""
    out = []
    for p in P:
        rays, _cov = cam.projectAll(lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])))
        out.append(np.array([[r["pt"][0], r["pt"][1]] for r in rays if not r["is_ring"]], float).reshape(-1, 2))
    return out


def rays(cam, uv: np.ndarray):
    """Back-projected rays in the object medium: (origins, unit directions, ok)."""
    res = cam.lineOfSightBatchStatus([lpt.Pt2D(float(a), float(b)) for a, b in uv])
    o = np.zeros((len(uv), 3))
    d = np.zeros((len(uv), 3))
    ok = np.zeros(len(uv), bool)
    for k, (good, line, _msg) in enumerate(res):
        if good:
            o[k] = (line.pt[0], line.pt[1], line.pt[2])
            d[k] = (line.unit_vector[0], line.unit_vector[1], line.unit_vector[2])
            nd = np.linalg.norm(d[k])
            if np.isfinite(o[k]).all() and nd > 1e-12:
                d[k] /= nd
                ok[k] = True
    return o, d, ok


# ------------------------------------------------------------------ fast vectorised projection
# The same model as the C++ PINCYL camera (pinhole in medium 1, radial distortion on normalised coordinates,
# coaxial cylinders), traced in numpy.  Back-projection is closed form; the forward projection solves for the
# pixel whose ray passes through the point, by Newton from a nearby pixel (the detection).  It is used inside
# the fits for speed and is checked against the C++ projection (`check_fast`); every reported residual is the
# C++ one.
def _undistort(x, y, dist, iters=30):
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    if not any((k1, k2, p1, p2, k3)):
        return x, y
    xu, yu = x.copy(), y.copy()
    for _ in range(iters):
        r2 = xu * xu + yu * yu
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        dx = 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu)
        dy = p1 * (r2 + 2 * yu * yu) + 2 * p2 * xu * yu
        xu, yu = (x - dx) / rad, (y - dy) / rad
    return xu, yu


def trace(rig: Rig, uv: np.ndarray):
    """Back-project pixels through the port: (origins on the inner wall, unit directions in the object medium, ok)."""
    x = (uv[:, 0] - rig.cx) / rig.f
    y = (uv[:, 1] - rig.cy) / rig.f
    x, y = _undistort(x, y, rig.dist)
    d = np.column_stack([x, y, np.ones_like(x)]) @ rig.R          # camera -> world: R^T d, as row vectors
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(rig.C, d.shape).copy()
    e = rig.axis_dir / np.linalg.norm(rig.axis_dir)
    a = rig.axis_pt
    ok = np.ones(len(uv), bool)
    n = list(rig.n)
    for k, radius in enumerate(rig.radii):
        w = o - a
        wp = w - np.outer(w @ e, e)
        dp = d - np.outer(d @ e, e)
        A = np.einsum("ij,ij->i", dp, dp)
        B = 2 * np.einsum("ij,ij->i", wp, dp)
        Cq = np.einsum("ij,ij->i", wp, wp) - radius * radius
        disc = B * B - 4 * A * Cq
        ok &= disc >= 0
        sq = np.sqrt(np.maximum(disc, 0.0))
        q = -0.5 * (B + np.where(B >= 0, sq, -sq))
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = q / A, Cq / q
        t = np.where((t1 > 0) & ((t1 <= t2) | (t2 <= 0)), t1, t2)
        ok &= t > 0
        h = o + t[:, None] * d
        hp = (h - a) - np.outer((h - a) @ e, e)
        N = hp / np.linalg.norm(hp, axis=1, keepdims=True)         # outward normal: towards the incident side
        r = n[k] / n[k + 1]
        cosi = -np.einsum("ij,ij->i", N, d)
        kk = 1 - r * r * (1 - cosi * cosi)
        ok &= kk >= 0
        d = r * d + (r * cosi - np.sqrt(np.maximum(kk, 0.0)))[:, None] * N
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = h
    if rig.has_wall():
        # diagnostic wall field: turn the ray in the object medium about the tube axis by delta(u)
        delta = bspline_basis(uv[:, 0], *rig.wall_spec) @ rig.wall
        c, s = np.cos(delta)[:, None], np.sin(delta)[:, None]
        d = d * c + np.cross(e, d) * s + np.outer(d @ e, e) * (1 - c)
    return o, d, ok


def _offsets(P, o, d, e1, e2):
    w = P - o
    wp = w - np.einsum("ij,ij->i", w, d)[:, None] * d
    return np.column_stack([np.einsum("ij,ij->i", wp, e1), np.einsum("ij,ij->i", wp, e2)])


def project_fast(rig: Rig, P: np.ndarray, uv0: np.ndarray, iters: int = 12, h: float = 1e-3):
    """Pixels whose back-projected rays pass through P, by Newton from uv0.  NaN where it does not converge."""
    uv = np.array(uv0, float, copy=True)
    step = np.full(len(P), np.inf)
    good = np.zeros(len(P), bool)
    for _ in range(iters):
        o, d, ok = trace(rig, uv)
        e1 = np.array([1.0, 0.0, 0.0])[None, :] - d[:, :1] * d
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(d, e1)
        r0 = _offsets(P, o, d, e1, e2)
        o1, d1, ok1 = trace(rig, uv + np.array([h, 0.0]))
        o2, d2, ok2 = trace(rig, uv + np.array([0.0, h]))
        J = np.stack([(_offsets(P, o1, d1, e1, e2) - r0) / h, (_offsets(P, o2, d2, e1, e2) - r0) / h], axis=2)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        good = ok & ok1 & ok2 & (np.abs(det) > 1e-300)
        du = np.zeros_like(uv)
        du[good, 0] = -(J[good, 1, 1] * r0[good, 0] - J[good, 0, 1] * r0[good, 1]) / det[good]
        du[good, 1] = -(-J[good, 1, 0] * r0[good, 0] + J[good, 0, 0] * r0[good, 1]) / det[good]
        step = np.linalg.norm(du, axis=1)
        du[step > 20.0] *= (20.0 / step[step > 20.0])[:, None]       # damp wild steps
        uv += du
        if np.all(good & (step < 1e-9)):
            break
    uv[~good | (step > 1e-6)] = np.nan
    return uv


FAST = True                                   # fits use project_fast; reports use the C++ projection


def check_fast(rig: Rig, obs: Obs) -> float:
    """Largest difference (px) between the fast projection and pyOpenLPT's C++ projection, over obs."""
    P = rig.world_points(obs)
    a = project_fast(rig, P, obs.uv)
    b = project(rig.camera(), P)
    return float(np.nanmax(np.abs(a - b)))


# ------------------------------------------------------------------ residuals
def forward_residual(rig: Rig, obs: Obs, cam=None, fast: bool = None) -> np.ndarray:
    fast = FAST if fast is None else fast
    if fast and cam is None:
        uv = project_fast(rig, rig.world_points(obs), obs.uv)
    else:
        cam = cam or rig.camera()
        uv = project(cam, rig.world_points(obs))
    r = uv - obs.uv
    r[~np.isfinite(r)] = PENALTY_PX
    return r.ravel()


def forward_residual_cpp(rig: Rig, obs: Obs) -> np.ndarray:
    """The reported residual: pyOpenLPT's own projection."""
    return forward_residual(rig, obs, fast=False)


def backproject_residual(rig: Rig, obs: Obs) -> np.ndarray:
    if FAST:
        o, d, ok = trace(rig, obs.uv)
    else:
        o, d, ok = rays(rig.camera(), obs.uv)
    P = rig.world_points(obs)
    w = P - o
    depth = np.einsum("ij,ij->i", w, d)
    perp = w - depth[:, None] * d
    # 2-D basis perpendicular to each ray: e1 ~ world X, e2 = d x e1
    ex = np.array([1.0, 0.0, 0.0])
    e1 = ex[None, :] - d[:, 0:1] * d
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    path = np.linalg.norm(o - rig.C[None, :], axis=1) + np.maximum(depth, 0.0)
    s = rig.f / np.maximum(path, 1e-6)
    r = np.column_stack([s * np.einsum("ij,ij->i", perp, e1), s * np.einsum("ij,ij->i", perp, e2)])
    r[~ok] = PENALTY_PX
    return r.ravel()


# ------------------------------------------------------------------ fitting
@dataclass
class FitResult:
    rig: Rig
    layout: Layout
    resid: np.ndarray                  # (n, 2) forward residuals at the solution
    rms: float                         # sqrt(mean of u^2 + v^2 over points) ... see rms_axis for per axis
    J: np.ndarray = None               # Jacobian at the solution, in the increment coordinates
    cov: np.ndarray = None
    sigma: np.ndarray = None
    history: list = None
    n_fail: int = 0
    fast_vs_cpp: float = float("nan")      # largest |fast - C++| projection difference at the solution (px)


def _jacobian(fun, x, layout: Layout, obs: Obs, anchor: Rig = None, rig_fun=None) -> np.ndarray:
    """Central differences, with two groupings that keep the cost independent of the number of planes and dots:

    * the plane offsets are perturbed together: each moves only its own plane's rows;
    * the plane shifts likewise: every plane's dX (then dY) is shifted at once on the rig, and the per-plane
      derivatives are mapped through the shift basis;
    * the board corrections: every dot's dX (then dY) is shifted at once on the rig itself -- each row depends
      only on its own dot -- and the per-dot derivatives are mapped through the board basis.
    `rig_fun(rig)` is the residual of a rig (needed for the shift and board groupings), `fun(x)` that of an
    increment.
    """
    h = layout.steps()
    r0 = fun(x)
    J = np.zeros((len(r0), layout.size))
    dz = layout.slices.get("dz")
    bd = sh = None
    if rig_fun is not None:
        bd = layout.slices.get("board", layout.slices.get("boardcr"))
        sh = layout.slices.get("shift")
    for q in range(layout.size):
        if dz is not None and dz.start <= q < dz.stop:
            continue
        if bd is not None and bd.start <= q < bd.stop:
            continue
        if sh is not None and sh.start <= q < sh.stop:
            continue
        xp, xm = x.copy(), x.copy()
        xp[q] += h[q]
        xm[q] -= h[q]
        J[:, q] = (fun(xp) - fun(xm)) / (2 * h[q])
    if dz is not None:
        xp, xm = x.copy(), x.copy()
        xp[dz] += h[dz]
        xm[dz] -= h[dz]
        dr = (fun(xp) - fun(xm)) / (2 * STEPS["dz"])
        rows_plane = np.repeat(obs.plane, 2)
        for k in range(1, obs.n_planes):
            rows = rows_plane == k
            J[rows, dz.start + k - 1] = dr[rows]
    if bd is not None:
        rig = layout.apply(x, anchor)
        hb = STEPS["board"]
        idx_obs, uniq = obs.dots()
        lut = {(int(a), int(b)): q for q, (a, b) in enumerate(rig.board_ij)}
        rowdot = np.array([lut[(int(a), int(b))] for a, b in uniq])[idx_obs]     # board row of each observation
        D = np.zeros((len(r0), 2))
        for c in (0, 1):
            rp, rm = rig.copy(), rig.copy()
            rp.board = rig.board.copy()
            rm.board = rig.board.copy()
            rp.board[:, c] += hb
            rm.board[:, c] -= hb
            D[:, c] = (rig_fun(rp) - rig_fun(rm)) / (2 * hb)
        # J[row, :] = D[row, 0] Nb[2 d, :] + D[row, 1] Nb[2 d + 1, :], d the row's dot
        d2 = np.repeat(rowdot, 2)
        J[:, bd] = D[:, 0:1] * layout.Nb[2 * d2] + D[:, 1:2] * layout.Nb[2 * d2 + 1]
    if sh is not None:
        rig = layout.apply(x, anchor)
        hs = STEPS["shift"]
        m = layout.Ns.shape[1]
        rows_plane = np.repeat(obs.plane, 2)
        for c in (0, 1):
            rp, rm = rig.copy(), rig.copy()
            rp.shift = rig.shift.copy()
            rm.shift = rig.shift.copy()
            rp.shift[:, c] += hs
            rm.shift[:, c] -= hs
            Dc = (rig_fun(rp) - rig_fun(rm)) / (2 * hs)
            J[:, sh.start + c * m: sh.start + (c + 1) * m] = Dc[:, None] * layout.Ns[rows_plane]
    return J


def fit(rig0: Rig, obs: Obs, blocks, residual: str = "forward", max_outer: int = 8,
        verbose: bool = False, max_nfev: int = 200) -> FitResult:
    """Least squares over the free blocks, re-anchored after every round."""
    anchor = reanchor(prepare(rig0, obs, blocks))
    layout = Layout(blocks, obs.n_planes, anchor, obs.pitch, z=obs.z_nom)
    history = []
    for outer in range(max_outer):
        if residual == "forward":
            def fun(x, anchor=anchor):
                return forward_residual(layout.apply(x, anchor), obs)

            def jac(x, anchor=anchor):
                return _jacobian(lambda y: fun(y, anchor), x, layout, obs, anchor,
                                 lambda rg: forward_residual(rg, obs))
            res = least_squares(fun, np.zeros(layout.size), jac=jac, method="trf", x_scale="jac",
                                ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=max_nfev)
        else:
            def fun(x, anchor=anchor):
                return backproject_residual(layout.apply(x, anchor), obs)
            res = least_squares(fun, np.zeros(layout.size), jac="2-point", method="trf", x_scale="jac",
                                diff_step=None, ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=max_nfev)
        anchor = reanchor(layout.apply(res.x, anchor))
        # size of this round's increment in units of the finite-difference steps
        rel = float(np.max(np.abs(res.x) / layout.steps())) if layout.size else 0.0
        rms = float(np.sqrt(np.mean(res.fun ** 2) * 2))
        history.append({"outer": outer, "rms_px": rms, "nfev": int(res.nfev), "status": int(res.status),
                        "max_step_over_fd": rel})
        if verbose:
            print(f"    [{residual} {'+'.join(layout.blocks)}] round {outer}: rms {rms:.4f} px, "
                  f"nfev {res.nfev}, max step/fd {rel:.2e}")
        if rel < 1e-2:
            break
    # the reported residual is pyOpenLPT's own projection, and the fast one must agree with it; a rig with the
    # diagnostic wall field has no C++ counterpart, so its residual is the fast tracer's
    if anchor.has_wall():
        r = forward_residual(anchor, obs).reshape(-1, 2)
    else:
        r = forward_residual_cpp(anchor, obs).reshape(-1, 2)
    n_fail = int(np.sum(np.all(r == PENALTY_PX, axis=1)))
    out = FitResult(anchor, layout, r, float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))), history=history,
                    n_fail=n_fail)
    if FAST and not anchor.has_wall():
        rf = forward_residual(anchor, obs).reshape(-1, 2)
        ok = np.all(r != PENALTY_PX, axis=1) & np.all(rf != PENALTY_PX, axis=1)
        out.fast_vs_cpp = float(np.max(np.abs(rf[ok] - r[ok]))) if ok.any() else float("nan")
    return out


def covariance(fr: FitResult, obs: Obs) -> FitResult:
    """Jacobian at the solution, covariance s^2 (J'J)^-1 and standard errors of the increments."""
    layout = fr.layout

    def fun(x):
        return forward_residual(layout.apply(x, fr.rig), obs)
    J = _jacobian(fun, np.zeros(layout.size), layout, obs, fr.rig, lambda rg: forward_residual(rg, obs))
    r = fr.resid.ravel()
    dof = max(len(r) - layout.size, 1)
    s2 = float(r @ r) / dof
    JTJ = J.T @ J
    cov = s2 * np.linalg.pinv(JTJ)
    fr.J, fr.cov, fr.sigma = J, cov, np.sqrt(np.diag(cov))
    return fr


def physical_sigmas(fr: FitResult) -> dict:
    """Standard errors in physical terms: camera centre (mm), rotation (deg), f (px), axis (deg, mm), dz (mm)."""
    L, s = fr.layout, fr.sigma
    out = {}
    if "rot" in L.blocks:
        out["rotation_deg"] = np.degrees(s[L.slices["rot"]]).tolist()
    if "cen" in L.blocks:
        out["camera_centre_mm"] = s[L.slices["cen"]].tolist()
    if "logf" in L.blocks:
        out["f_px"] = float(fr.rig.f * s[L.slices["logf"]][0])
    if "axdir" in L.blocks:
        out["axis_tilt_lean_deg"] = np.degrees(s[L.slices["axdir"]]).tolist()
    if "axpt" in L.blocks:
        out["axis_position_X_Z_mm"] = s[L.slices["axpt"]].tolist()
    if "dz" in L.blocks:
        out["dz_mm"] = [0.0] + s[L.slices["dz"]].tolist()
    if "zs" in L.blocks:
        out["stage_scale"] = float(s[L.slices["zs"]][0])
    if "tau" in L.blocks:
        out["tau"] = s[L.slices["tau"]].tolist()
    if "dist" in L.blocks:
        out["k1"] = float(s[L.slices["dist"]][0])
    if "shift" in L.blocks:
        sl, m = L.slices["shift"], L.Ns.shape[1]
        c = fr.cov[sl, sl]
        sx = np.sqrt(np.diag(L.Ns @ c[:m, :m] @ L.Ns.T))
        sy = np.sqrt(np.diag(L.Ns @ c[m:, m:] @ L.Ns.T))
        out["plane_shift_mm"] = np.column_stack([sx, sy]).tolist()
        out["plane_shift_mm_median"] = [float(np.median(sx)), float(np.median(sy))]
    if "board" in L.blocks or "boardcr" in L.blocks:
        sl = L.slices.get("board", L.slices.get("boardcr"))
        cb = L.Nb @ fr.cov[sl, sl] @ L.Nb.T
        sb = np.sqrt(np.diag(cb)).reshape(-1, 2)
        out["board_mm_median"] = [float(np.median(sb[:, 0])), float(np.median(sb[:, 1]))]
    if "wall" in L.blocks:
        sl = L.slices["wall"]
        us = np.arange(0.0, 1280.0, 8.0)
        Bw = bspline_basis(us, *fr.rig.wall_spec) @ L.Nw
        out["wall_mrad_median"] = float(np.median(np.sqrt(np.einsum("ij,jk,ik->i", Bw, fr.cov[sl, sl], Bw)))
                                        * 1e3)
    return out


def extras_summary(rig: Rig) -> dict:
    """Size of the optional corrections: board (mm) and wall deflection (mrad) and its shift at 100 mm."""
    out = {}
    if rig.board is not None:
        out["board_rms_mm"] = np.sqrt(np.mean(rig.board ** 2, axis=0)).tolist()
        out["board_max_mm"] = np.max(np.abs(rig.board), axis=0).tolist()
    if rig.shift is not None:
        out["plane_shift_rms_mm"] = np.sqrt(np.mean(rig.shift ** 2, axis=0)).tolist()
        out["plane_shift_max_mm"] = np.max(np.abs(rig.shift), axis=0).tolist()
    if rig.has_wall():
        us = np.arange(0.0, 1280.0, 8.0)
        dl = bspline_basis(us, *rig.wall_spec) @ rig.wall
        out["wall_rms_mrad"] = float(np.sqrt(np.mean(dl ** 2)) * 1e3)
        out["wall_max_mrad"] = float(np.max(np.abs(dl)) * 1e3)
        out["wall_shift_at_100mm_rms_um"] = float(np.sqrt(np.mean(dl ** 2)) * 100.0 * 1e3)
    return out


def correlations(fr: FitResult) -> np.ndarray:
    d = np.sqrt(np.diag(fr.cov))
    return fr.cov / np.outer(d, d)


# ------------------------------------------------------------------ starts
def start_rigs(obs: Obs, cfg: dict, scale_v0: float, f_lens_px: float = None) -> list:
    """Starting rigs for the multi-start: camera distance from the axis L, axis depth Z_a, camera lateral position.

    The camera looks at the middle of the observed board region; f follows from the vertical image scale of plane
    0 (px per mm along Y), using the flat-interface apparent depth along the axis direction.
    """
    port = cfg["port"]
    n_air, n_acr, n_w = port["n"]
    R1, R2 = port["R"]
    X = obs.pitch[0] * obs.i
    Y = obs.pitch[1] * obs.j
    mid = np.array([0.5 * (X.min() + X.max()), 0.5 * (Y.min() + Y.max()), float(np.mean(obs.z_nom))])
    starts = []
    for L in (400.0, 700.0, 1000.0, 1400.0):
        for Za in (-10.0, 0.0, 10.0, 20.0):
            for xmode in ("axis", "mid"):
                ax_pt = np.array([0.5, 0.0, Za])
                Cx = ax_pt[0] if xmode == "axis" else mid[0]
                C = np.array([Cx, 0.0, Za + L])
                # air-equivalent distance to plane 0 along the line of sight, then f from the vertical scale
                Dv = (L - R1) + (R1 - R2) * n_air / n_acr + (R2 + Za) * n_air / n_w
                f0 = scale_v0 * Dv
                R = look_at(C, mid)
                starts.append(Rig(R=R, C=C, f=f0, cx=cfg["camera"]["principal_point"][0],
                                  cy=cfg["camera"]["principal_point"][1], axis_pt=ax_pt,
                                  axis_dir=np.array([0.0, 1.0, 0.0]), dz=np.zeros(obs.n_planes),
                                  tau=np.zeros(2), n=list(port["n"]), radii=list(port["R"]),
                                  width=cfg["image"]["width"], height=cfg["image"]["height"]))
    return starts
