"""Wand calibration of cameras that look through one cylindrical port (PINCYL), from wand frames alone.

Inputs are the tube (radii and indices), the wand length L, each camera's image size and the detections of the
wand's two ends A and B in every frame.  No camera position, orientation or focal length is needed: the start
comes from the data (`bootstrap`).  Every camera's pose and focal length are estimated; k1 is optional.

    sol = calibrate(WandData(uv, L, sizes), Tube(radii, n))     # uv[frame, camera, end, (u, v)], NaN = not seen
    cylinder_wand_io.write_result(sol, folder)                   # PINCYL camFiles, result JSON, text report

**Frame and gauge.**  The world is the tube frame: the axis is the Y axis through the origin, and Z points
towards camera 1.  The tube then has no parameters.  What the data cannot fix is the tube's stabiliser, a spin
about the axis and a slide along it (2 dof).  It is removed by convention: camera 1's centre is held at X = 0 and
Y = 0, its distance from the axis free.  One discrete symmetry is left, a half-turn about Z (it reverses the axis
and keeps camera 1 in place); `canonical` fixes it by convention.  Afterwards the origin may slide along Y to the
wand's mean height (`relabel`), which changes no residual.

**Stages** (plan and tests: tests/cylinder_wand/):
  0  data: frames whose ends are both seen by at least two cameras (`observations`);
  1  affine bootstrap: factorisation of the frames seen by every camera, metric upgrade with L, both mirror
     images (`affine_factorization`, `orient`);
  2  tube placement: axis direction from the cameras' stretch, axis position by search, distances from the
     magnification ratios, f from the scale along the axis (`place`, `bootstrap`);
  3  coarse fit: back-projection residual with the points triangulated (`coarse_fit`), always defined;
  4  maximum likelihood: the reprojection error in pixels with explicit wand frames, outlier rejection and the
     multi-image guard (`bundle_adjust`, `refine`);
  5  covariance and checks (`covariance`, `camera_sigmas`, `predicted_projection_error`, `wand_lengths`; the
     files: cylinder_wand_io).

**Parametrisation (stage 4).**  Unknowns live on manifolds; each Levenberg-Marquardt iteration solves for a small
step in flat local coordinates at the current estimate and maps it back (and the estimate becomes the new
anchor):
  camera rotation  R <- R exp([dw]x)                         SO(3)
  camera centre    C <- C + dC                               R^3   (camera 1: only C3, the gauge)
  focal length     f <- f exp(dl)                            R+
  k1 (optional)    k1 <- k1 + dk
  wand midpoint    m <- m + dm                               R^3
  wand direction   w <- cos|d| w + sin|d| d/|d|, d = a t1 + b t2 (t1, t2 perpendicular to w)    S^2
The ends are m -+ (L/2) w, so the wand length is exact.  Each frame's 5 parameters touch only its own
observations, so the frames are eliminated (Schur complement) and only the camera system (7 per camera, 5 for
camera 1) is solved densely: the cost is linear in the number of frames.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import cyl_model as cm
except ImportError:  # imported as a top-level module (tests put this folder on sys.path)
    import cyl_model as cm

CAM_COLS = ("w1", "w2", "w3", "C1", "C2", "C3", "logf")
GAUGE_FIXED = ("C1", "C2")          # camera 1 (index 0): X and Y of its centre


# ------------------------------------------------------------------ data
@dataclass
class WandData:
    """Detections uv[frame, camera, end, (u, v)], NaN where not seen; end 0 is A, end 1 is B."""

    uv: np.ndarray
    L: float
    sizes: Sequence[Tuple[int, int]]           # (width, height) per camera

    def __post_init__(self):
        self.uv = np.asarray(self.uv, float)
        if self.uv.ndim != 4 or self.uv.shape[2:] != (2, 2):
            raise ValueError("uv must have shape (frames, cameras, 2 ends, 2)")
        if len(self.sizes) != self.uv.shape[1]:
            raise ValueError("one image size per camera")
        if not self.L > 0:
            raise ValueError("the wand length must be positive")

    @property
    def n_frames(self) -> int:
        return self.uv.shape[0]

    @property
    def n_cams(self) -> int:
        return self.uv.shape[1]


@dataclass
class Obs:
    """One row per detection, sorted by frame."""

    frame: np.ndarray        # (n,) index into the kept frames
    cam: np.ndarray          # (n,)
    end: np.ndarray          # (n,) 0 = A, 1 = B
    uv: np.ndarray           # (n, 2)
    n_frames: int
    n_cams: int
    frame_ids: np.ndarray    # (n_frames,) original frame numbers
    active: np.ndarray = None

    def __post_init__(self):
        if self.active is None:
            self.active = np.ones(len(self.frame), bool)

    def __len__(self) -> int:
        return len(self.frame)

    def subset_frames(self, keep: np.ndarray) -> "Obs":
        """The observations of the frames with keep[frame] True, renumbered."""
        keep = np.asarray(keep, bool)
        new_id = np.cumsum(keep) - 1
        rows = keep[self.frame]
        return Obs(new_id[self.frame[rows]], self.cam[rows], self.end[rows], self.uv[rows], int(keep.sum()),
                   self.n_cams, self.frame_ids[keep], self.active[rows].copy())


def observations(data: WandData, min_cams: int = 2) -> Obs:
    """Stage 0: the frames whose ends are both seen by at least `min_cams` cameras, as rows."""
    seen = np.all(np.isfinite(data.uv), axis=3)                     # (F, Nc, 2)
    good = np.all(seen.sum(axis=1) >= min_cams, axis=1)             # both ends
    fr, ca, en = np.nonzero(seen & good[:, None, None])
    order = np.lexsort((en, ca, fr))
    fr, ca, en = fr[order], ca[order], en[order]
    keep_ids = np.flatnonzero(good)
    new_id = np.full(data.n_frames, -1)
    new_id[keep_ids] = np.arange(len(keep_ids))
    return Obs(new_id[fr], ca, en, data.uv[fr, ca, en], len(keep_ids), data.n_cams, keep_ids)


# ------------------------------------------------------------------ frames
@dataclass
class Frames:
    m: np.ndarray            # (F, 3) midpoints
    w: np.ndarray            # (F, 3) unit directions A -> B

    def ends(self, L: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.m - 0.5 * L * self.w, self.m + 0.5 * L * self.w

    def copy(self) -> "Frames":
        return Frames(self.m.copy(), self.w.copy())

    def tangent_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        """(t1, t2) perpendicular to each w, t2 = w x t1."""
        w = self.w
        k = np.argmin(np.abs(w), axis=1)
        e = np.zeros_like(w)
        e[np.arange(len(w)), k] = 1.0
        t1 = e - np.einsum("ij,ij->i", e, w)[:, None] * w
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        return t1, np.cross(w, t1)

    def retract(self, df: np.ndarray) -> "Frames":
        """m <- m + dm, w <- exp_w(a t1 + b t2) (the exponential map of the unit sphere)."""
        t1, t2 = self.tangent_basis()
        d = df[:, 3:4] * t1 + df[:, 4:5] * t2
        th = np.linalg.norm(d, axis=1)
        safe = np.where(th > 0, th, 1.0)
        w = np.cos(th)[:, None] * self.w + (np.sin(th) / safe)[:, None] * d
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        return Frames(self.m + df[:, :3], w)


def frames_from_ends(XA: np.ndarray, XB: np.ndarray) -> Frames:
    d = XB - XA
    return Frames(0.5 * (XA + XB), d / np.linalg.norm(d, axis=1, keepdims=True))


# ------------------------------------------------------------------ parameter layout (gauge)
class Layout:
    """The free camera parameters: per camera the columns of CAM_COLS (+ k1), camera 1 without C1, C2 (the
    gauge); optionally the wand length L as one more global parameter."""

    def __init__(self, n_cams: int, fit_k1: bool = False, fit_L: bool = False, gauge: bool = True):
        self.cols = list(CAM_COLS) + (["k1"] if fit_k1 else [])
        self.model_idx = [cm.CAM_PARAMS.index(c) for c in self.cols]
        self.n_cams, self.fit_k1, self.fit_L, self.gauge = n_cams, fit_k1, fit_L, gauge
        self.pos = np.full((n_cams, len(self.cols)), -1, int)
        self.names: List[str] = []
        k = 0
        for c in range(n_cams):
            for j, name in enumerate(self.cols):
                if gauge and c == 0 and name in GAUGE_FIXED:
                    continue
                self.pos[c, j] = k
                self.names.append(f"cam{c + 1}.{name}")
                k += 1
        self.L_index = k if fit_L else -1
        if fit_L:
            self.names.append("L")
            k += 1
        self.P = k

    def apply(self, cams: List[cm.Camera], L: float, dx: np.ndarray):
        out = []
        for c, cam in enumerate(cams):
            k = cam.copy()
            v = np.zeros(len(self.cols))
            sel = self.pos[c] >= 0
            v[sel] = dx[self.pos[c][sel]]
            k.R = cam.R @ cm.exp_so3(v[0:3])
            k.C = cam.C + v[3:6]
            k.f = cam.f * math.exp(float(v[6]))
            if self.fit_k1:
                k.dist = cam.dist.copy()
                k.dist[0] += v[7]
            out.append(k)
        return out, (L + float(dx[self.L_index]) if self.fit_L else L)


# ------------------------------------------------------------------ residuals and Jacobians
@dataclass
class Eval:
    r: np.ndarray                     # (n, 2) reprojection residuals (NaN where the projection failed)
    ok: np.ndarray                    # (n,)
    Jg: Optional[np.ndarray] = None   # (n, 2, P) global (camera, L) columns
    Jf: Optional[np.ndarray] = None   # (n, 2, 5) frame columns (dm, a, b)
    det: Optional[np.ndarray] = None  # (n,) conditioning of the projection (0 at a fold)


def evaluate(cams, tube: cm.Tube, frames: Frames, L: float, obs: Obs, layout: Layout = None,
             rows: np.ndarray = None) -> Eval:
    """Reprojection residuals of the rows `rows` (default: all), and with a layout their Jacobians."""
    rows = np.arange(len(obs)) if rows is None else rows
    fr, ca, en = obs.frame[rows], obs.cam[rows], obs.end[rows]
    XA, XB = frames.ends(L)
    X = np.where((en == 0)[:, None], XA[fr], XB[fr])
    rw = cm.rows(cams, ca)
    pr = cm.project(rw, X, cm.undistort(obs.uv[rows], rw), tube, jac=layout is not None)
    r = pr.uv - obs.uv[rows]
    ev = Eval(r=r, ok=pr.ok.copy())
    if layout is None:
        return ev
    n = len(rows)
    Jg = np.zeros((n, 2, layout.P))
    for j, mi in enumerate(layout.model_idx):
        g = layout.pos[ca, j]
        sel = (g >= 0) & pr.ok
        Jg[np.flatnonzero(sel), :, g[sel]] = pr.J_cam[sel, :, mi]
    t1, t2 = frames.tangent_basis()
    sgn = np.where(en == 0, -0.5 * L, 0.5 * L)
    JX = np.where(pr.ok[:, None, None], pr.J_X, 0.0)
    Jf = np.zeros((n, 2, 5))
    Jf[:, :, 0:3] = JX
    Jf[:, :, 3] = np.einsum("nai,ni->na", JX, t1[fr]) * sgn[:, None]
    Jf[:, :, 4] = np.einsum("nai,ni->na", JX, t2[fr]) * sgn[:, None]
    if layout.fit_L:
        Jg[:, :, layout.L_index] = np.einsum("nai,ni->na", JX, frames.w[fr]) * np.where(en == 0, -0.5, 0.5)[:, None]
    ev.Jg, ev.Jf, ev.det = Jg, Jf, pr.det
    return ev


def _segment_sum(a: np.ndarray, starts: np.ndarray, n_seg: int, seg_of_start: np.ndarray) -> np.ndarray:
    """Sum rows of `a` (sorted by segment) over segments; segments without rows are zero."""
    out = np.zeros((n_seg,) + a.shape[1:])
    if len(a):
        out[seg_of_start] = np.add.reduceat(a, starts, axis=0)
    return out


def normal_blocks(ev: Eval, fr: np.ndarray, n_frames: int, weights: np.ndarray = None):
    """V = Jg'Jg, gV = Jg'r, U_i = Jf'Jf per frame, gU_i, W_i = Jg'Jf per frame (rows sorted by frame)."""
    r = np.where(ev.ok[:, None], ev.r, 0.0)
    Jg = np.where(ev.ok[:, None, None], ev.Jg, 0.0)
    Jf = np.where(ev.ok[:, None, None], ev.Jf, 0.0)
    if weights is not None:
        sw = np.sqrt(weights)[:, None]
        r = r * sw
        Jg = Jg * sw[:, :, None]
        Jf = Jf * sw[:, :, None]
    V = np.einsum("nai,naj->ij", Jg, Jg)
    gV = np.einsum("nai,na->i", Jg, r)
    starts = np.flatnonzero(np.r_[True, fr[1:] != fr[:-1]]) if len(fr) else np.zeros(0, int)
    seg = fr[starts] if len(fr) else np.zeros(0, int)
    U = _segment_sum(np.einsum("nai,naj->nij", Jf, Jf), starts, n_frames, seg)
    gU = _segment_sum(np.einsum("nai,na->ni", Jf, r), starts, n_frames, seg)
    W = _segment_sum(np.einsum("nai,naj->nij", Jg, Jf), starts, n_frames, seg)
    return V, gV, U, gU, W


def reduced_system(V, gV, U, gU, W, lam: float = 0.0):
    """Schur complement over the frames: S dc = rhs, and the back-substitution pieces."""
    dU = np.einsum("fii->fi", U)
    Ud = U + lam * np.einsum("fi,ij->fij", dU, np.eye(5))
    # a ridge of 1e-15 relative keeps a degenerate frame block invertible without breaking the stabiliser's exact
    # null space (T6); a frame without observations gets an identity block (its gradient and coupling are zero)
    scale = dU.max(axis=1)
    Ud = Ud + (1e-15 * scale)[:, None, None] * np.eye(5)
    Ud[scale <= 0] = np.eye(5)
    Uinv = np.linalg.inv(Ud)
    WUi = np.einsum("fpk,fkl->fpl", W, Uinv)
    S = V + lam * np.diag(np.diag(V)) - np.einsum("fpl,fql->pq", WUi, W)
    rhs = -gV + np.einsum("fpl,fl->p", WUi, gU)
    return S, rhs, Uinv, WUi


def solve_step(V, gV, U, gU, W, lam: float):
    S, rhs, Uinv, WUi = reduced_system(V, gV, U, gU, W, lam)
    dc = np.linalg.solve(S, rhs)
    df = -np.einsum("fkl,fl->fk", Uinv, gU + np.einsum("fpk,p->fk", W, dc))
    return dc, df


# ------------------------------------------------------------------ Levenberg-Marquardt
@dataclass
class Solution:
    cams: List[cm.Camera]
    tube: cm.Tube
    frames: Frames
    L: float
    obs: Obs
    layout: Layout
    rms: float = float("nan")
    history: List[dict] = field(default_factory=list)
    cov: Optional[np.ndarray] = None
    sigma_px: float = float("nan")
    info: Dict = field(default_factory=dict)


def cost_of(ev: Eval, rows_mask: np.ndarray = None) -> float:
    ok = ev.ok if rows_mask is None else ev.ok & rows_mask
    return float(np.sum(ev.r[ok] ** 2))


def bundle_adjust(cams, tube: cm.Tube, frames: Frames, L: float, obs: Obs, layout: Layout,
                  max_iter: int = 100, tol: float = 1e-10, lam0: float = 1e-3, verbose: bool = False,
                  log=print) -> Solution:
    """Levenberg-Marquardt on the manifold over the active observations, frames eliminated (Schur).

    Stops when the largest step (in units of each parameter's own scale: rad, mm per |C|, log f, mm per 10 mm for
    the frames) falls below `tol`, or when no step lowers the cost.  A trial step that makes a projection fail is
    rejected like one that raises the cost."""
    cams = [c.copy() for c in cams]
    frames = frames.copy()
    rows = np.flatnonzero(obs.active)
    hist = []
    ev = evaluate(cams, tube, frames, L, obs, layout, rows)
    good = ev.ok.copy()
    cost = cost_of(ev, good)
    lam, nu = lam0, 2.0
    t0 = time.time()
    for it in range(max_iter):
        fr = obs.frame[rows]
        V, gV, U, gU, W = normal_blocks(Eval(ev.r, good, ev.Jg, ev.Jf), fr, obs.n_frames)
        accepted = False
        for _inner in range(12):
            try:
                dc, df = solve_step(V, gV, U, gU, W, lam)
            except np.linalg.LinAlgError:
                lam *= nu
                nu *= 2
                continue
            # predicted cost from the linearisation
            lin = np.where(good[:, None], ev.r, 0.0) + np.einsum("nai,i->na", np.where(good[:, None, None], ev.Jg, 0),
                                                                  dc) + np.einsum("nai,ni->na", np.where(
                                                                      good[:, None, None], ev.Jf, 0), df[fr])
            pred = cost - float(np.sum(lin ** 2))
            cams_t, L_t = layout.apply(cams, L, dc)
            frames_t = frames.retract(df)
            ev_t = evaluate(cams_t, tube, frames_t, L_t, obs, layout, rows)
            if np.any(good & ~ev_t.ok):
                lam *= nu
                nu *= 2
                continue
            cost_t = cost_of(ev_t, good)
            rho = (cost - cost_t) / pred if pred > 0 else -1.0
            if rho > 0:
                step_rel = _step_size(dc, df, cams, frames, layout)
                cams, frames, L = cams_t, frames_t, L_t
                ev = ev_t
                good = good | ev.ok                  # rows whose projection now succeeds join the cost
                cost = cost_of(ev, good)
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                accepted = True
                break
            lam *= nu
            nu *= 2
        n_good = int(good.sum())
        rms = math.sqrt(cost / max(n_good, 1))
        hist.append({"iter": it, "rms_px": rms, "lambda": lam, "accepted": accepted,
                     "seconds": time.time() - t0})
        if verbose:
            log(f"    LM {it:3d}: rms {rms:.6f} px, lambda {lam:.1e}, {'ok' if accepted else 'stuck'}")
        if not accepted:
            break                      # no step lowers the cost even at large damping: converged to round-off
        # on the step, not the cost: along the near-telecentric f-distance direction the cost is so flat that a
        # relative decrease of 1e-14 still leaves the estimate 1e-5 sigma short (T6)
        if step_rel < tol or cost == 0.0:
            break
    sol = Solution(cams, tube, frames, L, obs, layout, rms=math.sqrt(cost / max(int(good.sum()), 1)),
                   history=hist)
    sol.info["n_fail"] = int((~good).sum())
    return sol


def _step_size(dc, df, cams, frames, layout) -> float:
    """Largest step relative to a natural scale: radians, mm over 1000 mm, log f, mm over the wand scale."""
    s = 0.0
    for c in range(layout.n_cams):
        for j, name in enumerate(layout.cols):
            g = layout.pos[c, j]
            if g < 0:
                continue
            v = abs(dc[g])
            if name.startswith("C"):
                v /= max(np.linalg.norm(cams[c].C), 1.0)
            s = max(s, v)
    if layout.fit_L:
        s = max(s, abs(dc[layout.L_index]) / 10.0)
    if len(df):
        s = max(s, float(np.max(np.abs(df[:, :3]))) / 10.0, float(np.max(np.abs(df[:, 3:]))))
    return s


# ------------------------------------------------------------------ stage 4 with outlier rejection
def refine(cams, tube: cm.Tube, frames: Frames, L: float, obs: Obs, layout: Layout, k_sigma: float = 5.0,
           max_rounds: int = 10, guard_px: float = 30.0, max_iter: int = 100, sigma_floor_px: float = 1e-3,
           verbose: bool = False, log=print) -> Solution:
    """Stage 4: bundle adjustment, then a new active set chosen from every usable observation (those active on
    entry): residual within k_sigma robust sigmas (Mahalanobis, `_update_active`), and no frame left with an end seen by fewer than two
    cameras.
    Repeat until the set no longer changes.  Rejections are not final: an observation rejected while the fit was
    still poor comes back once it fits (T5 found a start that lost 50 good observations to a one-way rejection).
    Then the multi-image guard (an observation whose point has another image on the sensor within guard_px of the
    detection is dropped), a last adjustment and the covariance.  `obs.active` is updated in place."""
    sol = None
    t0 = time.time()
    usable = obs.active.copy()
    changed = False
    for rnd in range(max_rounds):
        sol = bundle_adjust(cams, tube, frames, L, obs, layout, max_iter=max_iter, verbose=verbose, log=log)
        cams, frames, L = sol.cams, sol.frames, sol.L
        changed, n_out, n_back = _update_active(sol, usable, k_sigma, sigma_floor_px)
        frames = sol.frames
        if verbose:
            log(f"  round {rnd}: rms {sol.rms:.5f} px, {len(sol.history)} LM iterations, rejected {n_out}, "
                f"re-admitted {n_back}")
        if not changed:
            break
    if changed:                                  # the round limit was hit: fit the last set
        sol = bundle_adjust(cams, tube, frames, L, obs, layout, max_iter=max_iter, verbose=verbose, log=log)
        cams, frames, L = sol.cams, sol.frames, sol.L
    n_guard = multi_image_guard(sol, guard_px) if guard_px else 0
    if n_guard:
        kept = dict(sol.info)
        sol = bundle_adjust(cams, tube, frames, L, obs, layout, max_iter=max_iter, verbose=verbose, log=log)
        sol.info = {**kept, **sol.info}
    sol.info["guard_dropped"] = int(n_guard)
    sol.info["rounds"] = rnd + 1
    sol.info["seconds"] = time.time() - t0
    return covariance(sol)


def _update_active(sol: Solution, usable: np.ndarray, k_sigma: float, sigma_floor_px: float = 1e-3):
    """Choose the active set afresh from the usable observations at the current solution.  Frames with no active
    observation were not adjusted, so they are first re-triangulated from all their usable rays with the current
    cameras.  Returns (changed, number newly rejected, number re-admitted)."""
    obs = sol.obs
    act_frame = np.zeros(obs.n_frames, bool)
    act_frame[obs.frame[obs.active]] = True
    stale = ~act_frame
    if stale.any():
        rows = np.flatnonzero(usable & stale[obs.frame])
        if rows.size:
            XA, XB, cnt = triangulate(sol.cams, sol.tube, obs, rows)
            ok = stale & np.all(cnt >= 2, axis=1) & np.all(np.isfinite(XA), axis=1) & np.all(np.isfinite(XB), axis=1)
            if ok.any():
                fr = frames_from_ends(XA[ok], XB[ok])
                sol.frames = sol.frames.copy()
                sol.frames.m[ok], sol.frames.w[ok] = fr.m, fr.w
    ev = evaluate(sol.cams, sol.tube, sol.frames, sol.L, obs, Layout(obs.n_cams))
    act = obs.active & ev.ok
    if act.sum() < 10:
        return False, 0, 0
    # Each residual is judged by its Mahalanobis distance under its own predicted covariance, which follows from
    # its frame's 5 parameters (P = J_f U_f^-1 J_f^T, U_f the frame's information from the active rows; the camera
    # parameters' share, from thousands of observations, is neglected): sigma^2 (I - P) for an observation in the
    # fit, sigma^2 (I + P) for one predicted from the others.  d^2 / sigma^2 is then chi-square with 2 dof for every
    # rig.  The plain |r| is not: with two cameras a frame has 3 residual dof over 8 coordinates, each residual is
    # nearly one-dimensional and shrunk to ~0.6 sigma, and a 5-sigma Rayleigh cut on |r| rejected ~0.4 % of good
    # observations (T5, 2-camera rigs).
    Jf = np.where(ev.ok[:, None, None], ev.Jf, 0.0)
    U = np.zeros((obs.n_frames, 5, 5))
    np.add.at(U, obs.frame[act], np.einsum("nai,naj->nij", Jf[act], Jf[act]))
    Uinv = np.zeros_like(U)
    good_f = np.linalg.det(U) > 0
    Uinv[good_f] = np.linalg.inv(U[good_f])
    Pm = np.einsum("nai,nij,nbj->nab", Jf, Uinv[obs.frame], Jf)
    M = np.eye(2)[None] + np.where(obs.active, -1.0, 1.0)[:, None, None] * Pm
    lam, V = np.linalg.eigh(M)
    lam = np.maximum(lam, 1e-3)                    # an observation that alone fixes a direction of its frame
    rr = np.where(ev.ok[:, None], ev.r, 0.0)
    proj = np.einsum("nai,na->ni", V, rr)
    d = np.sqrt(np.sum(proj ** 2 / lam, axis=1))
    d[~ev.ok] = np.inf
    sigma = float(np.median(d[act])) / math.sqrt(2.0 * math.log(2.0))   # median of a Rayleigh(sigma)
    lim = k_sigma * max(sigma, sigma_floor_px)     # the floor keeps noise-free data from rejecting itself
    # hysteresis: a rejected observation comes back only well inside the limit, so the set cannot oscillate
    new = usable & ev.ok & np.where(obs.active, d <= lim, d <= 0.8 * lim)
    probe = Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, new)
    _drop_thin_frames(probe)
    new = probe.active
    n_out = int(np.sum(obs.active & ~new))
    n_back = int(np.sum(~obs.active & new))
    changed = bool(n_out or n_back)
    obs.active[:] = new
    sol.info["robust_sigma_px"] = sigma
    return changed, n_out, n_back


def _drop_thin_frames(obs: Obs, min_cams: int = 2) -> int:
    """Deactivate every observation of a frame one of whose ends has fewer than min_cams active rays."""
    cnt = np.zeros((obs.n_frames, 2), int)
    np.add.at(cnt, (obs.frame[obs.active], obs.end[obs.active]), 1)
    thin = np.any(cnt < min_cams, axis=1)
    bad = obs.active & thin[obs.frame]
    obs.active[bad] = False
    return int(bad.sum())


def multi_image_guard(sol: Solution, guard_px: float, probe_mm: float = 0.02) -> int:
    """Drop active observations whose point has another image on the sensor within guard_px of the detection (a
    blob there would be a blend of two images, or the wrong one).  Only points beyond the fold radius can have two
    images, so only those are checked, with the C++ image census.  Next to a fold the second image appears or
    vanishes within microns of the point, so the census is also taken at the point moved by +-probe_mm along each
    axis, and any of the seven showing a second image within guard_px drops the observation."""
    obs = sol.obs
    rows = np.flatnonzero(obs.active)
    XA, XB = sol.frames.ends(sol.L)
    X = np.where((obs.end[rows] == 0)[:, None], XA[obs.frame[rows]], XB[obs.frame[rows]])
    beyond = np.hypot(X[:, 0], X[:, 2]) > sol.tube.fold_radius()
    offsets = np.vstack([np.zeros(3), probe_mm * np.eye(3), -probe_mm * np.eye(3)])
    n_drop = 0
    multi = 0
    for c, cam in enumerate(sol.cams):
        sel = np.flatnonzero((obs.cam[rows] == c) & beyond)
        if not sel.size:
            continue
        cpp = cm.to_cpp(cam, sol.tube)
        for k in sel:
            i = rows[k]
            drop = False
            counted = False
            for off in offsets:
                im = cm.cpp_images(cpp, (X[k] + off)[None])[0]
                on = im[(im[:, 0] >= 0) & (im[:, 0] < cam.width) & (im[:, 1] >= 0) & (im[:, 1] < cam.height)]
                if len(on) < 2:
                    continue
                if not counted:
                    multi += 1
                    counted = True
                if np.sort(np.linalg.norm(on - obs.uv[i], axis=1))[1] < guard_px:
                    drop = True
                    break
            if drop:
                obs.active[i] = False
                n_drop += 1
    sol.info["multi_image_observations"] = multi
    sol.info["beyond_fold_observations"] = int(beyond.sum())
    if n_drop:
        n_drop += _drop_thin_frames(obs)
    return n_drop


# ------------------------------------------------------------------ stage 3: coarse fit (back-projection)
def _rays(cams, tube, obs: Obs, rows):
    rw = cm.rows(cams, obs.cam[rows])
    xy = cm.undistort(obs.uv[rows], rw)
    o, d, ok, _, _ = cm.trace(rw, xy, tube)
    return rw, o, d, ok


def backprojection_residual(cams, tube: cm.Tube, obs: Obs, L: float, rows, penalty: float = 30.0):
    """Stage 3 residual (n, 2): the offset of each frame's end from the observation's ray, in pixels (scaled by
    f / optical path), with the ends triangulated from the rays and then set exactly L apart about their midpoint
    (variable projection: the points never lag behind the cameras).  Always defined; rays that miss the tube and
    frames that cannot be triangulated get `penalty`."""
    rw, o, d, ok = _rays(cams, tube, obs, rows)
    F = obs.n_frames
    fr, en = obs.frame[rows], obs.end[rows]
    path0 = np.linalg.norm(o - rw.C, axis=1)
    wgt = np.where(ok, (rw.f / (path0 + 50.0)) ** 2, 0.0)
    Pm = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = wgt[:, None, None] * Pm
    b = np.einsum("nij,nj->ni", A, np.where(ok[:, None], o, 0.0))
    key = fr * 2 + en
    AA = np.zeros((2 * F, 3, 3))
    bb = np.zeros((2 * F, 3))
    cnt = np.zeros(2 * F, int)
    np.add.at(AA, key, A)
    np.add.at(bb, key, b)
    np.add.at(cnt, key, ok.astype(int))
    good = cnt >= 2
    Xe = np.full((2 * F, 3), np.nan)
    Xe[good] = np.linalg.solve(AA[good], bb[good][:, :, None])[:, :, 0]
    XA, XB = Xe[0::2], Xe[1::2]
    dv = XB - XA
    nd = np.linalg.norm(dv, axis=1)
    u = dv / np.where(nd > 0, nd, 1.0)[:, None]
    m = 0.5 * (XA + XB)
    X = np.where((en == 0)[:, None], (m - 0.5 * L * u)[fr], (m + 0.5 * L * u)[fr])
    g, s, _e1, _e2 = cm.ray_offsets(np.nan_to_num(X), o, d, rw.R[:, 0, :])
    r = g * (rw.f / (path0 + np.maximum(s, 0.0)))[:, None]
    bad = ~ok | ~np.isfinite(r).all(axis=1) | ~np.isfinite(X).all(axis=1)
    r[bad] = penalty
    return r


def coarse_fit(cams, tube: cm.Tube, obs: Obs, L: float, layout: Layout, max_outer: int = 8,
               loss: str = "soft_l1", f_scale: float = 2.0, max_nfev: int = 100, rows=None, verbose: bool = False,
               log=print):
    """Stage 3: least squares of the back-projection residual over the camera parameters (frames eliminated by
    triangulation), re-anchored after every round.  Returns (cameras, info)."""
    from scipy.optimize import least_squares
    rows = np.flatnonzero(obs.active) if rows is None else rows
    lay = Layout(layout.n_cams, fit_k1=layout.fit_k1, fit_L=False, gauge=layout.gauge)
    anchor = [c.copy() for c in cams]
    hist = []
    for outer in range(max_outer):
        def fun(x, anchor=anchor):
            cs, _ = lay.apply(anchor, L, x)
            return backprojection_residual(cs, tube, obs, L, rows).ravel()
        res = least_squares(fun, np.zeros(lay.P), method="trf", x_scale="jac", loss=loss, f_scale=f_scale,
                            max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        anchor, _ = lay.apply(anchor, L, res.x)
        rms = float(np.sqrt(np.mean(res.fun ** 2) * 2))
        step = _step_size(res.x, np.zeros((0, 5)), anchor, None, lay)
        hist.append({"outer": outer, "rms_px": rms, "nfev": int(res.nfev), "step": step})
        if verbose:
            log(f"    coarse round {outer}: rms {rms:.4f} px, nfev {res.nfev}, step {step:.1e}")
        if step < 1e-8:
            break
    return anchor, {"history": hist, "rms_px": hist[-1]["rms_px"]}


def initial_frames(cams, tube: cm.Tube, obs: Obs, L: float) -> Frames:
    """Frames from the triangulated ends, set exactly L apart about their midpoint; frames that cannot be
    triangulated get a placeholder and should be inactive."""
    XA, XB, cnt = triangulate(cams, tube, obs)
    bad = ~np.all(np.isfinite(XA), axis=1) | ~np.all(np.isfinite(XB), axis=1)
    XA[bad] = 0.0
    XB[bad] = np.array([0.0, 1.0, 0.0])
    fr = frames_from_ends(XA, XB)
    fr.w[~np.all(np.isfinite(fr.w), axis=1)] = np.array([0.0, 1.0, 0.0])
    return fr


# ------------------------------------------------------------------ covariance and reports
def information(cams, tube, frames, L, obs: Obs, layout: Layout, rows=None, sigma_px: float = 1.0):
    """Fisher information of the free camera parameters, the frames marginalised (Schur), for i.i.d. Gaussian
    pixel noise of `sigma_px`; and the evaluation used."""
    rows = np.flatnonzero(obs.active) if rows is None else rows
    ev = evaluate(cams, tube, frames, L, obs, layout, rows)
    V, gV, U, gU, W = normal_blocks(ev, obs.frame[rows], obs.n_frames)
    S, _rhs, _Ui, _WUi = reduced_system(V, gV, U, gU, W, 0.0)
    return S / sigma_px ** 2, ev


def covariance(sol: Solution, sigma_px: float = None) -> Solution:
    """Covariance of the free camera parameters, sigma^2 S^-1 with S the reduced (Schur) normal matrix.  With
    sigma_px None, sigma is estimated from the residuals with the degrees of freedom of the full problem."""
    rows = np.flatnonzero(sol.obs.active)
    info1, ev = information(sol.cams, sol.tube, sol.frames, sol.L, sol.obs, sol.layout, rows, 1.0)
    n_res = 2 * int(ev.ok.sum())
    dof = max(n_res - sol.layout.P - 5 * sol.obs.n_frames, 1)
    s2 = float(np.sum(ev.r[ev.ok] ** 2)) / dof if sigma_px is None else sigma_px ** 2
    sol.cov = s2 * np.linalg.inv(info1)
    sol.sigma_px = math.sqrt(s2)
    sol.info["dof"] = dof
    sol.info["cond_S"] = float(np.linalg.cond(info1))
    return sol


def camera_sigmas(cams, layout: Layout, cov: np.ndarray) -> List[dict]:
    """Per camera: sigma of the rotation (deg, about the world axes), centre (mm), f (relative and px), and the
    derived distance from the axis, azimuth and height."""
    out = []
    for c, cam in enumerate(cams):
        idx = layout.pos[c]
        full = np.zeros((len(layout.cols), len(layout.cols)))
        sel = np.flatnonzero(idx >= 0)
        full[np.ix_(sel, sel)] = cov[np.ix_(idx[sel], idx[sel])]
        sw = np.degrees(np.sqrt(np.diag(full)[0:3]))
        cC = full[3:6, 3:6]
        Cx, Cy, Cz = cam.C
        rho = math.hypot(Cx, Cz)
        # derived: distance from the axis, azimuth (deg), height
        Jd = np.array([[Cx / rho, 0.0, Cz / rho],
                       [math.degrees(Cz / rho ** 2), 0.0, math.degrees(-Cx / rho ** 2)],
                       [0.0, 1.0, 0.0]])
        cd = Jd @ cC @ Jd.T
        d = {"rotation_deg": sw.tolist(), "centre_mm": np.sqrt(np.diag(cC)).tolist(),
             "f_rel": math.sqrt(full[6, 6]), "f_px": cam.f * math.sqrt(full[6, 6]),
             "distance_mm": math.sqrt(max(cd[0, 0], 0.0)), "azimuth_deg": math.sqrt(max(cd[1, 1], 0.0)),
             "height_mm": math.sqrt(max(cd[2, 2], 0.0))}
        if layout.fit_k1:
            d["k1"] = math.sqrt(full[7, 7])
        out.append(d)
    return out


def point_jacobians(cams, tube, layout: Layout, P: np.ndarray):
    """For fixed world points P (m, 3): per camera c, d(uv)/d(free camera params) (m, 2, P_free), d(uv)/dX
    (m, 2, 3) and the pixels; points a camera cannot project get NaN."""
    out = []
    for c, cam in enumerate(cams):
        rw = cm.rows([cam], np.zeros(len(P), int))
        pr = cm.project(rw, P, cm.pinhole_start(rw, P), tube, jac=True)
        J = np.zeros((len(P), 2, layout.P))
        for j, mi in enumerate(layout.model_idx):
            g = layout.pos[c, j]
            if g >= 0:
                J[:, :, g] = pr.J_cam[:, :, mi]
        J[~pr.ok] = np.nan
        out.append((J, pr.J_X, pr.uv, pr.ok))
    return out


def predicted_projection_error(cams, tube, layout: Layout, cov: np.ndarray, P: np.ndarray, sizes=None):
    """Cramer-Rao prediction of the calibration's error on unseen points P (tube frame), RMS per 2-D image:
      fixed       projecting a known tube-frame point, in the gauge of the fit (camera 1 at X = Y = 0);
      stabiliser  the same, after the best common spin about the axis and slide along it of all the points:
                  the tube's stabiliser is not observable, so an error along it is a relabelling, not a
                  calibration error.  A near-telecentric camera 1 trades its height with its pitch, so in the
                  camera-1 gauge the whole scene can slide along the axis relative to camera 1's centre; this
                  removes that.  It is the figure of merit for projecting unseen points;
      retriang    each point re-triangulated from all cameras: what the cameras disagree on (gauge-free; what
                  particle tracking feels).
    Points not seen by every camera (sensor or tube) are skipped."""
    pj = point_jacobians(cams, tube, layout, P)
    ok = np.all([p[3] for p in pj], axis=0)
    if sizes is not None:
        for (J, JX, uv, okc), (w, h) in zip(pj, sizes):
            ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    fixed, retri = [], []
    per_cam = [[] for _ in cams]
    Jall, Gall = [], []
    ey = np.array([0.0, 1.0, 0.0])
    for i in np.flatnonzero(ok):
        Jc = np.vstack([pj[c][0][i] for c in range(len(cams))])          # (2 Nc, P)
        JX = np.vstack([pj[c][1][i] for c in range(len(cams))])          # (2 Nc, 3)
        Sig = Jc @ cov @ Jc.T
        for c in range(len(cams)):
            v = float(np.trace(Sig[2 * c:2 * c + 2, 2 * c:2 * c + 2]))
            fixed.append(v)
            per_cam[c].append(v)
        Pperp = np.eye(len(JX)) - JX @ np.linalg.pinv(JX)
        retri.append(float(np.trace(Pperp @ Sig @ Pperp.T)) / len(cams))
        Jall.append(Jc)
        Gall.append(JX @ np.column_stack([np.cross(ey, P[i]), ey]))     # spin, slide of the point
    out = {"n_points": int(ok.sum()),
           "fixed_rms_px": math.sqrt(np.mean(fixed)) if fixed else float("nan"),
           "fixed_rms_px_per_cam": [math.sqrt(np.mean(v)) if v else float("nan") for v in per_cam],
           "retriangulated_rms_px": math.sqrt(np.mean(retri)) if retri else float("nan")}
    if Jall:
        J = np.vstack(Jall)
        G = np.vstack(Gall)
        Jp = J - G @ np.linalg.lstsq(G, J, rcond=None)[0]
        L = np.linalg.cholesky(cov + 1e-300 * np.eye(len(cov)))
        A = Jp @ L
        out["stabiliser_rms_px"] = math.sqrt(float(np.sum(A * A)) / (len(Jall) * len(cams)))
    else:
        out["stabiliser_rms_px"] = float("nan")
    return out


def relabel(cams, frames: Frames = None):
    """Slide the origin along the axis to the wand's mean height (no residual changes)."""
    if frames is None:
        return cams, frames, 0.0
    y0 = float(np.mean(frames.m[:, 1]))
    out = []
    for cam in cams:
        k = cam.copy()
        k.C = cam.C - np.array([0.0, y0, 0.0])
        out.append(k)
    return out, Frames(frames.m - np.array([0.0, y0, 0.0]), frames.w.copy()), y0


def triangulate(cams, tube, obs: Obs, rows=None, n_frames: int = None):
    """Each frame's ends from the back-projected rays of their detections (least squares of the offsets, weighted
    to pixels by f / path).  Returns XA, XB (F, 3) and how many rays each end had."""
    rows = np.flatnonzero(obs.active) if rows is None else rows
    F = obs.n_frames if n_frames is None else n_frames
    rw = cm.rows(cams, obs.cam[rows])
    xy = cm.undistort(obs.uv[rows], rw)
    o, d, ok, _, _ = cm.trace(rw, xy, tube)
    path = np.linalg.norm(o - rw.C, axis=1) + 50.0
    wgt = np.where(ok, (rw.f / path) ** 2, 0.0)
    P = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = wgt[:, None, None] * P
    b = np.einsum("nij,nj->ni", A, o)
    key = obs.frame[rows] * 2 + obs.end[rows]
    AA = np.zeros((2 * F, 3, 3))
    bb = np.zeros((2 * F, 3))
    cnt = np.zeros(2 * F, int)
    np.add.at(AA, key, A)
    np.add.at(bb, key, b)
    np.add.at(cnt, key, ok.astype(int))
    X = np.full((2 * F, 3), np.nan)
    good = cnt >= 2
    X[good] = np.linalg.solve(AA[good], bb[good][:, :, None])[:, :, 0]
    return X[0::2], X[1::2], cnt.reshape(F, 2)


def wand_lengths(cams, tube, obs: Obs, rows=None):
    """Free triangulation (no length constraint) of every frame: the measured wand length per frame."""
    XA, XB, cnt = triangulate(cams, tube, obs, rows)
    Lm = np.linalg.norm(XB - XA, axis=1)
    Lm[np.any(cnt < 2, axis=1)] = np.nan
    return Lm


def triangulate_ml(cams, tube: cm.Tube, cam_idx: np.ndarray, uv: np.ndarray, point_idx: np.ndarray,
                   X0: np.ndarray, sigma_px: float = 1.0, iters: int = 15):
    """Maximum-likelihood triangulation: each point X_p minimises the reprojection error of its detections (rows with
    point_idx == p, camera cam_idx, pixel uv), by Gauss-Newton from X0 with the forward projection's d(uv)/dX.
    Returns X (P, 3), the residuals (n, 2) (NaN where a projection failed) and each point's covariance
    sigma^2 (J^T J)^-1 (P, 3, 3)."""
    X = np.array(X0, float, copy=True)
    P = len(X)
    rw = cm.rows(cams, cam_idx)
    xy0 = cm.undistort(uv, rw)
    r = np.full((len(uv), 2), np.nan)
    H = np.zeros((P, 3, 3))
    for _ in range(iters):
        pr = cm.project(rw, X[point_idx], xy0, tube, jac=True)
        ok = pr.ok
        r = np.where(ok[:, None], pr.uv - uv, np.nan)
        J = np.where(ok[:, None, None], pr.J_X, 0.0)
        rr = np.nan_to_num(r)
        H = np.zeros((P, 3, 3))
        g = np.zeros((P, 3))
        np.add.at(H, point_idx, np.einsum("nai,naj->nij", J, J))
        np.add.at(g, point_idx, np.einsum("nai,na->ni", J, rr))
        good = np.linalg.det(H) > 0
        dX = np.zeros((P, 3))
        dX[good] = -np.linalg.solve(H[good], g[good][:, :, None])[:, :, 0]
        X += dX
        if np.max(np.abs(dX)) < 1e-12:
            break
    pr = cm.project(rw, X[point_idx], xy0, tube, jac=True)
    r = np.where(pr.ok[:, None], pr.uv - uv, np.nan)
    J = np.where(pr.ok[:, None, None], pr.J_X, 0.0)
    H = np.zeros((P, 3, 3))
    np.add.at(H, point_idx, np.einsum("nai,naj->nij", J, J))
    cov = np.full((P, 3, 3), np.nan)
    good = np.linalg.det(H) > 0
    cov[good] = sigma_px ** 2 * np.linalg.inv(H[good])
    return X, r, cov


# ------------------------------------------------------------------ stages 1-2: the start without input
def gauge_fix(cams, points: np.ndarray = None):
    """Turn about the axis and slide along it so that camera 1's centre is at X = 0 (Z > 0) and Y = 0.  Points
    (n, 3), if given, follow.  No residual changes: this is the tube's stabiliser."""
    C0 = cams[0].C
    phi = math.atan2(C0[0], C0[2])
    c, s = math.cos(phi), math.sin(phi)
    Ry = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    shift = np.array([0.0, C0[1], 0.0])
    out = []
    for cam in cams:
        k = cam.copy()
        k.R = cam.R @ Ry.T
        k.C = Ry @ cam.C - shift
        out.append(k)
    if points is None:
        return out
    return out, np.asarray(points, float) @ Ry.T - shift


def canonical(cams, frames: Frames = None):
    """Fix the one discrete symmetry the gauge leaves: a half-turn about Z (X -> -X, Y -> -Y) maps the tube onto
    itself and keeps camera 1's centre where it is, so it only flips the sign of the axis direction.  Convention:
    +Y points up in camera 1's image, or right when the axis is closer to the image's horizontal.  Returns
    (cameras, frames, flipped)."""
    R1 = cams[0].R
    dx, dy = R1[0, 1], R1[1, 1]                      # image direction of +Y in camera 1 (x right, y down)
    flip = (dy > 0) if abs(dy) >= abs(dx) else (dx < 0)
    if not flip:
        return cams, frames, False
    Fz = np.diag([-1.0, -1.0, 1.0])
    out = []
    for cam in cams:
        k = cam.copy()
        k.R = cam.R @ Fz
        k.C = Fz @ cam.C
        out.append(k)
    if frames is not None:
        frames = Frames(frames.m @ Fz, frames.w @ Fz)
    return out, frames, True


@dataclass
class Affine:
    """Stage 1: metric affine cameras u = M_c X + t_c and the ends of the frames used, about their centroid."""

    M: np.ndarray            # (Nc, 2, 3)
    t: np.ndarray            # (Nc, 2)
    XA: np.ndarray           # (F', 3)
    XB: np.ndarray           # (F', 3)
    frames: np.ndarray       # (F',) frame indices (into obs) used
    rms_px: float            # rank-3 fit residual, RMS per 2-D image
    length_rms_rel: float    # metric upgrade residual, RMS of |d|^2/L^2 - 1
    Q_eig: np.ndarray        # eigenvalues of the metric form (all positive)


def affine_factorization(obs: Obs, L: float, k_reject: float = 5.0, rounds: int = 5, min_frames: int = 12) -> Affine:
    """Tomasi-Kanade factorisation of the frames whose ends every camera sees, with gross-error rejection; then the
    metric upgrade: every segment d_i of the affine reconstruction has d_i^T Q d_i = L^2, linear in the six
    entries of Q = A^T A, so A = Q^1/2 up to an orthogonal factor (a rotation and a mirror image)."""
    Nc, F = obs.n_cams, obs.n_frames
    act = obs.active
    seen = np.zeros((F, Nc, 2), bool)
    seen[obs.frame[act], obs.cam[act], obs.end[act]] = True
    use = np.flatnonzero(seen.all(axis=(1, 2)))
    if len(use) < min_frames:
        raise ValueError(f"only {len(use)} frames are seen whole by every camera; the affine start needs "
                         f"{min_frames}")
    col = np.full(F, -1)
    col[use] = np.arange(len(use))
    W = np.zeros((2 * Nc, 2 * len(use)))
    rows = np.flatnonzero(act & (col[obs.frame] >= 0))
    for i in rows:
        c, e, k = obs.cam[i], obs.end[i], col[obs.frame[i]]
        W[2 * c:2 * c + 2, 2 * k + e] = obs.uv[i]
    keep = np.ones(len(use), bool)
    for _ in range(rounds):
        cols = np.flatnonzero(np.repeat(keep, 2))
        Wk = W[:, cols]
        t = Wk.mean(axis=1)
        W0 = Wk - t[:, None]
        U, s, Vt = np.linalg.svd(W0, full_matrices=False)
        M = U[:, :3] * np.sqrt(s[:3])
        S = np.sqrt(s[:3])[:, None] * Vt[:3]
        Rz = (W0 - M @ S).reshape(Nc, 2, -1)
        per_img = np.sqrt(np.sum(Rz ** 2, axis=1))                       # (Nc, cols)
        per_frame = per_img.max(axis=0).reshape(-1, 2).max(axis=1)
        thr = k_reject * max(float(np.median(per_frame)), 1e-9)
        bad = per_frame > thr
        if not bad.any():
            break
        idx = np.flatnonzero(keep)
        keep[idx[bad]] = False
    rms = float(np.sqrt(np.mean(per_img ** 2)))
    d = S[:, 1::2] - S[:, 0::2]
    A = np.column_stack([d[0] ** 2, d[1] ** 2, d[2] ** 2, 2 * d[0] * d[1], 2 * d[0] * d[2], 2 * d[1] * d[2]])
    sel = np.ones(len(A), bool)
    res = np.zeros(len(A))
    q = None
    for _ in range(rounds):
        q = np.linalg.lstsq(A[sel], np.full(int(sel.sum()), L * L), rcond=None)[0]
        res = A @ q / (L * L) - 1.0
        mad = 1.4826 * float(np.median(np.abs(res[sel] - np.median(res[sel]))))
        bad = np.abs(res) > k_reject * max(mad, 1e-6)
        if np.array_equal(bad, ~sel):
            break
        sel = ~bad
    Q = np.array([[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]])
    lam, V = np.linalg.eigh(Q)
    if np.any(lam <= 0):
        raise ValueError("the metric upgrade failed (the wand-length form is not positive definite)")
    Am = np.sqrt(lam)[:, None] * V.T                                     # A^T A = Q
    Sm = Am @ S
    Mm = M @ np.linalg.inv(Am)
    fr = np.flatnonzero(keep)
    return Affine(M=Mm.reshape(Nc, 2, 3), t=t.reshape(Nc, 2), XA=Sm[:, 0::2].T[sel], XB=Sm[:, 1::2].T[sel],
                  frames=use[fr][sel], rms_px=rms, length_rms_rel=float(np.sqrt(np.mean(res[sel] ** 2))),
                  Q_eig=lam)


@dataclass
class Orientation:
    """Stage 2a, for one mirror choice: everything about the cameras the affine model knows, in the tube frame
    (axis = Y, Z towards camera 1), about the cloud's centroid."""

    mirror: bool
    R: List[np.ndarray]          # world -> camera rotations
    q: List[np.ndarray]          # a point of each principal ray (the ray of the principal point), in the water
    ratio: np.ndarray            # (Nc,) high / low magnification
    s_low: np.ndarray            # (Nc,) low magnification (along the axis), px per mm
    XA: np.ndarray
    XB: np.ndarray
    axis_view_cos: np.ndarray    # |a . z_c|: the axis should be across every camera's view


def orient(aff: Affine, cxcy, mirror: bool) -> Orientation:
    """Polar decomposition of each affine camera, M_c = P_c O_c: O_c's rows are the camera's image axes, the
    stretch P_c's singular values its two magnifications (across and along the tube).  The axis is perpendicular
    to every camera's high-magnification direction; Z points towards camera 1."""
    P = np.diag([1.0, 1.0, -1.0]) if mirror else np.eye(3)
    R, q, ratio, s_low, h = [], [], [], [], []
    for c in range(len(aff.M)):
        M = aff.M[c] @ P
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        O = U @ Vt
        R.append(np.vstack([O[0], O[1], np.cross(O[0], O[1])]))
        h.append(Vt[0])
        ratio.append(s[0] / s[1])
        s_low.append(s[1])
        q.append(np.linalg.pinv(M) @ (np.asarray(cxcy[c], float) - aff.t[c]))
    a = np.linalg.svd(np.vstack(h))[2][-1]
    if a @ (-R[0][1]) < 0:                      # Y up in camera 1's image (any choice works; this one is the case's)
        a = -a
    z1 = R[0][2]
    Z = -(z1 - (z1 @ a) * a)
    Z /= np.linalg.norm(Z)
    T = np.vstack([np.cross(a, Z), a, Z])
    return Orientation(mirror=mirror, R=[Rc @ T.T for Rc in R], q=[T @ qc for qc in q], ratio=np.array(ratio),
                       s_low=np.array(s_low), XA=(aff.XA @ P) @ T.T, XB=(aff.XB @ P) @ T.T,
                       axis_view_cos=np.array([abs(a @ Rc[2]) for Rc in R]))


def _distance_from_ratio(R, q, cloud, tube: cm.Tube, ratio_obs: float, s_low_obs: float, D_lo: float,
                         D_hi: float, n: int = 36, m_pts: int = 120):
    """Distance and f of a camera from its affine stretch, compared like for like.

    The model camera (f = 1) is put at distance s along its principal ray (from q) and projects the cloud; an
    affine camera fitted to those images has a stretch ratio (across / along the tube) and a low magnification.
    The distance is where the ratio matches the observed affine camera's, and f = observed low magnification /
    the model's.  (Comparing with the local ratio at the centroid instead is biased: over a deep cloud seen from
    close by, perspective inflates the affine ratio towards the tube's telecentric limit, and a 0.6 m camera was
    placed at 6 m -- T5 rig 37.)  Returns (s, f) or None."""
    z = R[2]
    pts = np.asarray(cloud, float)
    if len(pts) > m_pts:
        pts = pts[np.linspace(0, len(pts) - 1, m_pts).round().astype(int)]
    m = len(pts)
    s_grid = np.geomspace(D_lo, D_hi, n)
    C = q[None, :] - s_grid[:, None] * z[None, :]
    outside = np.hypot(C[:, 0], C[:, 2]) > tube.r_outer * 1.02
    if not outside.any():
        return None
    s_grid, C = s_grid[outside], C[outside]
    k = len(s_grid)
    rw = cm.Rows(np.repeat(R[None], k * m, axis=0), np.repeat(C, m, axis=0), np.ones(k * m), np.zeros(k * m),
                 np.zeros(k * m), np.zeros((k * m, 5)))
    X = np.tile(pts, (k, 1))
    pr = cm.project(rw, X, cm.pinhole_start(rw, X), tube)
    xy = pr.xy.reshape(k, m, 2)
    ok = pr.ok.reshape(k, m)
    rat = np.full(k, np.nan)
    slow = np.full(k, np.nan)
    A = np.column_stack([pts, np.ones(m)])
    for i in range(k):
        g = ok[i]
        if g.sum() < 12:
            continue
        coef = np.linalg.lstsq(A[g], xy[i][g], rcond=None)[0]
        sv = np.linalg.svd(coef[:3].T, compute_uv=False)
        rat[i], slow[i] = sv[0] / sv[1], sv[1]
    good = np.isfinite(rat)
    if not good.any():
        return None
    s_grid, rat, slow = s_grid[good], rat[good], slow[good]
    j0 = int(np.argmin(np.abs(rat - ratio_obs)))
    s_best, sl = float(s_grid[j0]), float(slow[j0])
    for j in (j0 - 1, j0):                       # linear refinement where neighbours bracket the observed ratio
        if 0 <= j < len(rat) - 1 and (rat[j] - ratio_obs) * (rat[j + 1] - ratio_obs) <= 0 and rat[j] != rat[j + 1]:
            w = (ratio_obs - rat[j]) / (rat[j + 1] - rat[j])
            s_best = float(np.exp((1 - w) * np.log(s_grid[j]) + w * np.log(s_grid[j + 1])))
            sl = float((1 - w) * slow[j] + w * slow[j + 1])
            break
    return s_best, s_low_obs / sl


def place(ori: Orientation, p_xz, tube: cm.Tube, sizes, cxcy, s_fixed=None, D_range=(150.0, 6000.0),
          f_range=None):
    """Stage 2b: cameras for an axis at p_xz (relative to the cloud's centroid).  Each camera keeps its affine
    rotation and principal ray; its distance along that ray and its f come from its affine stretch, compared like
    for like with the model's (`_distance_from_ratio`), or the distance is given (s_fixed) and f follows from the
    flat-window scale along the axis.  Returns (cameras in the gauge, ends stacked (XA; XB) in the same frame) or
    None."""
    p3 = np.array([p_xz[0], 0.0, p_xz[1]])
    cent = -p3
    cloud = np.vstack([ori.XA - p3, ori.XB - p3])
    cams = []
    for c in range(len(ori.R)):
        R = ori.R[c]
        q = ori.q[c] - p3
        if s_fixed is None:
            got = _distance_from_ratio(R, q, cloud, tube, ori.ratio[c], ori.s_low[c], *D_range)
            if got is None:
                return None
            s, f = got
        else:
            s, f = s_fixed[c], None
        C = q - s * R[2]
        D = float(np.hypot(C[0], C[2]))
        if D <= tube.r_outer * 1.02:
            return None
        if f is None:
            zxz = np.array([R[2][0], 0.0, R[2][2]])
            zxz /= max(np.linalg.norm(zxz), 1e-12)
            f = ori.s_low[c] * tube.apparent_distance(D, float(cent @ zxz))
        if f_range is not None:
            f = float(np.clip(f, *f_range))
        cams.append(cm.Camera(R, C, f, cxcy[c][0], cxcy[c][1], sizes[c][0], sizes[c][1]))
    return gauge_fix(cams, cloud)


def _subset_rows(obs: Obs, n_frames: int):
    F = obs.n_frames
    pick = np.zeros(F, bool)
    pick[np.unique(np.linspace(0, F - 1, min(n_frames, F)).round().astype(int))] = True
    return np.flatnonzero(obs.active & pick[obs.frame])


def _robust_cost(r, f_scale: float = 2.0) -> float:
    z = np.sum(r ** 2, axis=1) / f_scale ** 2
    return float(np.sum(2.0 * (np.sqrt(1.0 + z) - 1.0)))


def bootstrap(obs: Obs, tube: cm.Tube, L: float, sizes, cxcy=None, hints: dict = None, n_score_frames: int = 150,
              n_short: int = 6, n_quick_frames: int = 40, quick_nfev: int = 8, grid_step: float = None,
              verbose: bool = False, log=print):
    """Stages 1-2: candidate starting rigs from the data alone, best first.

    Stage 1 factorises the frames seen by every camera (affine cameras, metric upgrade with L).  For both mirror
    images, stage 2 takes each camera's rotation and principal ray from its affine camera, the axis direction from
    the cameras' high-magnification directions, and searches the axis position across the feasible disc (every
    point inside the inner wall); at each position the camera distances follow from the magnification ratios and
    f from the scale along the axis.  Every candidate gets a quick stage-3 fit on `n_quick_frames` frames and is
    ranked by its robust back-projection cost after it; the best `n_short` get a longer fit on `n_score_frames`
    frames and are ranked again.
    Hints (all optional): {"f_range": (lo, hi) px, "distance_range": (lo, hi) mm along each principal ray}.
    Returns (list of (cost, cameras), info)."""
    hints = hints or {}
    t0 = time.time()
    cxcy = cxcy or [((w - 1) / 2.0, (h - 1) / 2.0) for (w, h) in sizes]
    aff = affine_factorization(obs, L)
    rows = _subset_rows(obs, n_score_frames)
    layout = Layout(obs.n_cams)
    D_range = hints.get("distance_range", (150.0, 6000.0))
    f_range = hints.get("f_range")
    cands = []
    for mirror in (False, True):
        ori = orient(aff, cxcy, mirror)
        r_cloud = float(np.max(np.hypot(np.r_[ori.XA[:, 0], ori.XB[:, 0]], np.r_[ori.XA[:, 2], ori.XB[:, 2]])))
        rho = 0.98 * tube.r_inner - r_cloud
        if rho <= 0:
            continue
        step = grid_step or max(4.0, rho / 4.0)
        g = np.arange(-rho, rho + 1e-9, step)
        g = g - g[np.argmin(np.abs(g))]                     # include the centroid itself
        for px in g:
            for pz in g:
                if px * px + pz * pz > rho * rho:
                    continue
                pl = place(ori, (px, pz), tube, sizes, cxcy, D_range=D_range, f_range=f_range)
                if pl is None:
                    continue
                cams, _ends = pl
                r = backprojection_residual(cams, tube, obs, L, rows)
                cands.append((_robust_cost(r), mirror, (float(px), float(pz)), cams))
    if not cands:
        raise ValueError("no starting rig: the cloud does not fit inside the tube under the affine model")
    # every candidate gets a quick stage-3 fit on a few frames before it is ranked: the cost of the placement as it
    # stands is no guide, because with the affine rotations still ~1 deg off, placements far from the tube (near
    # telecentric) score better than the right one (T5 rig 37)
    quick_rows = _subset_rows(obs, n_quick_frames)
    quick = []
    for cost0, mirror, p, cams in cands:
        cs, _info = coarse_fit(cams, tube, obs, L, layout, max_outer=1, max_nfev=quick_nfev, rows=quick_rows)
        quick.append((_robust_cost(backprojection_residual(cs, tube, obs, L, quick_rows)), cs, mirror, p, cost0))
    quick.sort(key=lambda c: c[0])
    short = []
    for cost_q, cams, mirror, p, cost0 in quick[:n_short]:
        cs, _info = coarse_fit(cams, tube, obs, L, layout, max_outer=2, max_nfev=40, rows=rows)
        short.append((_robust_cost(backprojection_residual(cs, tube, obs, L, rows)), cs, mirror, p, cost0))
        if verbose:
            log(f"    candidate mirror {mirror} p ({p[0]:+.1f}, {p[1]:+.1f}) mm: cost as placed {cost0:.3g}, after a "
                f"quick fit {cost_q:.3g}, after a short fit {short[-1][0]:.3g}")
    short.sort(key=lambda c: c[0])
    info = {"affine_rms_px": aff.rms_px, "affine_frames": int(len(aff.frames)),
            "affine_length_rms_rel": aff.length_rms_rel, "candidates": len(cands),
            "best": {"mirror": bool(short[0][2]), "p": list(short[0][3])}, "seconds": time.time() - t0}
    return [(c[0], c[1]) for c in short], info


def calibrate(data: WandData, tube: cm.Tube, start=None, hints: dict = None, fit_k1: bool = False,
              fit_L: bool = False, k_sigma: float = 5.0, guard_px: float = 30.0, n_final: int = 2,
              verbose: bool = False, log=print) -> Solution:
    """Stages 0-5: from the detections to the cameras, with no camera input (or from `start`, cameras in the
    gauge).  The best `n_final` bootstrap candidates are carried through stages 3 and 4; the answer keeping the
    most observations, then with the lowest cost, wins.  With fit_L the wand length is estimated too (stages 1-3
    use the nominal data.L): the tube's radii fix the scale, so the fitted L against the nominal checks the tube's
    constants (tests/cylinder_wand T9)."""
    obs = observations(data)
    layout = Layout(data.n_cams, fit_k1=fit_k1, fit_L=fit_L)
    t0 = time.time()
    binfo = {}
    if start is not None:
        starts = [start]
    else:
        ranked, binfo = bootstrap(obs, tube, data.L, data.sizes, hints=hints, verbose=verbose, log=log)
        starts = [c for _cost, c in ranked[:n_final]]
    best = None
    active0 = obs.active.copy()
    for k, cams0 in enumerate(starts):
        o = Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, active0.copy())
        cams3, info3 = coarse_fit(cams0, tube, o, data.L, layout)
        sol = refine(cams3, tube, initial_frames(cams3, tube, o, data.L), data.L, o, layout, k_sigma=k_sigma,
                     guard_px=guard_px)
        sol.info["stage3"] = info3
        n_act = int(o.active.sum())
        if verbose:
            log(f"  start {k}: stage 3 rms {info3['rms_px']:.3f} px -> stage 4 rms {sol.rms:.5f} px, active "
                f"{n_act}/{len(o)}")
        key = (-n_act, sol.rms)
        if best is None or key < best[0]:
            best = (key, sol)
    sol = best[1]
    sol.cams, sol.frames, flipped = canonical(sol.cams, sol.frames)
    if flipped:
        sol = covariance(sol)            # the same covariance in the flipped coordinates
    sol.info["bootstrap"] = binfo
    sol.info["total_seconds"] = time.time() - t0
    return sol
