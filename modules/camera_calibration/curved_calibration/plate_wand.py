"""Wand calibration of cameras that look through flat windows (PINPLATE), from wand frames alone.

Inputs are the windows' plates (thicknesses and indices), which camera looks through which window, the wand length
L, each camera's image size and the detections of the wand's two ends A and B in every frame.  No camera position,
orientation or focal length, and no window position or orientation, is needed: the start comes from the data
(`calibrate`, stages 1-3).  Every camera's pose and focal length and every window's normal and position are
estimated; k1 is optional.

    rig = Rig([Window(n, d, (12.7,), (1.0, 1.49, 1.333))], cam_win=[0, 0, 0, 0])   # n, d ignored unless given
    sol = calibrate(WandData(uv, L, sizes), rig)        # uv[frame, camera, end, (u, v)], NaN = not seen
    plate_wand_io.write_result(sol, folder)              # PINPLATE camFiles, result JSON, text report

It is the cylinder calibration's machinery (`cylinder_wand`, tests/cylinder_wand/README.md) with the flat-window
camera model of `plate_model` (exact derivatives by implicit differentiation).

**Gauge.**  Nothing in the scene is fixed, so the residuals do not change under a rigid motion of everything (6 dof);
the wand length fixes the scale.  The gauge holds camera 1's pose at its start value, and the answer is then moved
to the output frame (`to_output_frame`): Z along window 1's normal (camera -> water), Y along n1 x n2 with two windows
or camera 1's image "up" in the window's plane with one, the origin at the wand's mean position.  The covariance is
in the gauge of the fit (camera 1 held).

**Stages** (plan and tests: tests/plate_wand/):
  0  data: frames whose ends are both seen by at least two cameras (`cylinder_wand.observations`);
  1  affine start: factorisation of the frames seen by every camera (or by a camera subset, the rest added by DLT),
     metric upgrade with L, both mirror images (`affine_start`, `cylinder_wand.affine_factorization`);
  2  pinhole start: each camera's rotation from its affine camera, its distance from perspective (a wand nearer the
     camera looks longer; this also tells the mirror images apart), then a pinhole bundle adjustment
     (`pinhole_start`): a camera behind a flat window is nearly a pinhole camera;
  3  windows: per window, candidate normals about the cameras' viewing direction and candidate offsets, the cameras
     put on their chief rays behind each candidate, resected at a fixed distance to it, the best refined together
     with the window (`window_search`, `resect`, `resect_window`), then a joint coarse fit of the back-projection
     residual (`coarse_fit`).  A rig with one window is started whole (`bootstrap`); with several, each window's
     cameras are calibrated as a group and the groups joined, a camera alone behind its window then placed against
     the others' points (`bootstrap_groups`); with one camera per window, by alternating placements
     (`bootstrap_singles`).  Design and evidence: tests/plate_wand/2026-10-02-plate-wand-implementation.md, section 10;
  4  maximum likelihood: the reprojection error in pixels with explicit wand frames, outlier rejection and per-camera
     noise weights (`bundle_adjust`, `refine`); on a rig that uses several windows, a few restarts perturbed about the
     answer, the cheapest refitted if it lowers the cost (`perturbed_restart`): a start can end pinned against a
     window, a wand end on its water-side surface; and a scan of each weakly seen window's normal (`weak_windows`,
     `scan_window_normal`): its true normal can lie in a narrow valley, tens of degrees from a wrong answer's;
  5  covariance and checks (`covariance`, `camera_sigmas`, `window_sigmas`, `predicted_projection_error`,
     `wand_lengths`; the files: plate_wand_io).

**Parametrisation (stage 4).**  As the cylinder's: R <- R exp([dw]x), C <- C + dC, f <- f exp(dl), k1 <- k1 + dk; a
window's normal moves along the unit sphere, n <- exp_n(da t1 + db t2), and its camera-side surface pivots about a
fixed anchor point A (the mean crossing of its cameras' principal rays), d = n.A + delta; the wand frames are the
midpoint and the direction on S^2, eliminated by the Schur complement.  Camera 1 has only logf (and k1) free.
"""
from __future__ import annotations

import itertools
import math
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import cyl_model as cm
    from . import plate_model as pm
    from . import cylinder_wand as cw
except ImportError:  # imported as a top-level module (tests put this folder on sys.path)
    import cyl_model as cm
    import plate_model as pm
    import cylinder_wand as cw

WandData = cw.WandData
Obs = cw.Obs
Frames = cw.Frames
Eval = cw.Eval
observations = cw.observations
frames_from_ends = cw.frames_from_ends
cost_of = cw.cost_of
noise_groups = cw.noise_groups
NOISE_MODES = cw.NOISE_MODES
_drop_thin_frames = cw._drop_thin_frames
_robust_cost = cw._robust_cost
_subset_rows = cw._subset_rows
Window = pm.Window

CAM_COLS = ("w1", "w2", "w3", "C1", "C2", "C3", "logf")


# ------------------------------------------------------------------ normal equations (cylinder_wand's, with BLAS products)
def _segment_sum(a: np.ndarray, starts: np.ndarray, n_seg: int, seg_of_start: np.ndarray) -> np.ndarray:
    out = np.zeros((n_seg,) + a.shape[1:])
    if len(a):
        out[seg_of_start] = np.add.reduceat(a, starts, axis=0)
    return out


def normal_blocks(ev: Eval, fr: np.ndarray, n_frames: int, weights: np.ndarray = None):
    """V = Jg'Jg, gV = Jg'r, U_i = Jf'Jf per frame, gU_i, W_i = Jg'Jf per frame (rows sorted by frame), as
    `cylinder_wand.normal_blocks`."""
    r = np.where(ev.ok[:, None], ev.r, 0.0)
    Jg = np.where(ev.ok[:, None, None], ev.Jg, 0.0)
    Jf = np.where(ev.ok[:, None, None], ev.Jf, 0.0)
    if weights is not None:
        sw = np.sqrt(weights)[:, None]
        r = r * sw
        Jg = Jg * sw[:, :, None]
        Jf = Jf * sw[:, :, None]
    P = Jg.shape[2]
    J2 = Jg.reshape(-1, P)
    V = J2.T @ J2
    gV = J2.T @ r.reshape(-1)
    starts = np.flatnonzero(np.r_[True, fr[1:] != fr[:-1]]) if len(fr) else np.zeros(0, int)
    seg = fr[starts] if len(fr) else np.zeros(0, int)
    JfT = Jf.transpose(0, 2, 1)
    U = _segment_sum(np.matmul(JfT, Jf), starts, n_frames, seg)
    gU = _segment_sum(np.matmul(JfT, r[:, :, None])[:, :, 0], starts, n_frames, seg)
    W = _segment_sum(np.matmul(Jg.transpose(0, 2, 1), Jf), starts, n_frames, seg)
    return V, gV, U, gU, W


def reduced_system(V, gV, U, gU, W, lam: float = 0.0):
    """Schur complement over the frames: S dc = rhs, and the back-substitution pieces (as cylinder_wand's)."""
    dU = np.einsum("fii->fi", U)
    Ud = U + lam * dU[:, :, None] * np.eye(5)[None]
    scale = dU.max(axis=1)
    Ud = Ud + (1e-15 * scale)[:, None, None] * np.eye(5)
    Ud[scale <= 0] = np.eye(5)
    Uinv = np.linalg.inv(Ud)
    WUi = np.matmul(W, Uinv)
    S = V + lam * np.diag(np.diag(V)) - np.tensordot(WUi, W, axes=([0, 2], [0, 2]))
    rhs = -gV + np.tensordot(WUi, gU, axes=([0, 2], [0, 1]))
    return S, rhs, Uinv, WUi


def solve_step(V, gV, U, gU, W, lam: float):
    S, rhs, Uinv, WUi = reduced_system(V, gV, U, gU, W, lam)
    dc = np.linalg.solve(S, rhs)
    df = -np.matmul(Uinv, (gU + np.matmul(W.transpose(0, 2, 1), dc))[:, :, None])[:, :, 0]
    return dc, df


GAUGE_FIXED = ("w1", "w2", "w3", "C1", "C2", "C3")     # camera 1 (index 0): its pose is the gauge
WIN_COLS = ("a", "b", "delta")


# ------------------------------------------------------------------ the rig and the parameter layout
@dataclass
class Rig:
    """The windows and which one each camera looks through (-1: none, a plain pinhole camera)."""

    windows: List[pm.Window]
    cam_win: List[int]

    def __post_init__(self):
        self.cam_win = [int(k) for k in self.cam_win]
        if any(k >= len(self.windows) for k in self.cam_win):
            raise ValueError("cam_win refers to a window that does not exist")

    def copy(self) -> "Rig":
        return Rig([w.copy() for w in self.windows], list(self.cam_win))

    def cams_of(self, w: int) -> List[int]:
        return [c for c, k in enumerate(self.cam_win) if k == w]

    def pinhole(self) -> "Rig":
        """The same cameras without windows."""
        return Rig([w.copy() for w in self.windows], [-1] * len(self.cam_win))


def anchors_for(cams, rig: Rig) -> List[Optional[np.ndarray]]:
    """Each window's anchor: the mean of where its cameras' principal rays meet its camera-side surface (the foot of
    the camera centre when a principal ray does not head into the window)."""
    out = []
    for w, win in enumerate(rig.windows):
        pts = []
        for c in rig.cams_of(w):
            z = cams[c].R[2]
            h1 = win.distance(cams[c].C)
            cz = float(win.n @ z)
            if cz > 0.05:
                pts.append(cams[c].C + (h1 / cz) * z)
            else:
                pts.append(cams[c].C + h1 * win.n)
        out.append(np.mean(pts, axis=0) if pts else win.d * win.n)
    return out


class Layout:
    """The free parameters: per camera the columns of CAM_COLS (+ k1), camera 1 without its pose (the gauge); per
    window not in fixed_windows its (a, b, delta) about its anchor -- only delta for a window in held_normals, whose
    normal is held (a hint-held start) --; optionally L.  `cams_free` limits the free cameras (e.g. a resection).
    Optional priors (`add_prior`) add residual rows on log f or on a camera's distance to its window."""

    def __init__(self, n_cams: int, n_wins: int = 0, fit_k1: bool = False, fit_L: bool = False, gauge: bool = True,
                 fixed_windows: Sequence[int] = (), anchors=None, cams_free: Sequence[int] = None,
                 held_normals: Sequence[int] = ()):
        self.cols = list(CAM_COLS) + (["k1"] if fit_k1 else [])
        self.model_idx = [cm.CAM_PARAMS.index(c) for c in self.cols]
        self.n_cams, self.n_wins, self.fit_k1, self.fit_L, self.gauge = n_cams, n_wins, fit_k1, fit_L, gauge
        self.fixed_windows = tuple(int(w) for w in fixed_windows)
        self.held_normals = tuple(int(w) for w in held_normals)
        self.anchors = list(anchors) if anchors is not None else [None] * n_wins
        self.pos = np.full((n_cams, len(self.cols)), -1, int)
        self.names: List[str] = []
        k = 0
        for c in range(n_cams):
            if cams_free is not None and c not in cams_free:
                continue
            for j, name in enumerate(self.cols):
                if gauge and c == 0 and name in GAUGE_FIXED:
                    continue
                self.pos[c, j] = k
                self.names.append(f"cam{c + 1}.{name}")
                k += 1
        self.wpos = np.full((n_wins, 3), -1, int)
        for w in range(n_wins):
            if w in self.fixed_windows:
                continue
            for j, name in enumerate(WIN_COLS):
                if j < 2 and w in self.held_normals:
                    continue
                self.wpos[w, j] = k
                self.names.append(f"win{w + 1}.{name}")
                k += 1
        self.L_index = k if fit_L else -1
        if fit_L:
            self.names.append("L")
            k += 1
        self.P = k
        self.priors: List[dict] = []

    def add_prior(self, kind: str, cam: int, mean: float, sigma: float):
        """kind "logf" (mean log f, sigma relative) or "distance" (mean h1 in mm, sigma mm) for camera `cam`."""
        if kind not in ("logf", "distance"):
            raise ValueError("a prior is on 'logf' or 'distance'")
        self.priors.append({"kind": kind, "cam": int(cam), "mean": float(mean), "sigma": float(sigma)})

    def apply(self, cams: List[pm.Camera], rig: Rig, L: float, dx: np.ndarray):
        out = []
        for c, cam in enumerate(cams):
            k = cam.copy()
            v = np.zeros(len(self.cols))
            sel = self.pos[c] >= 0
            v[sel] = dx[self.pos[c][sel]]
            k.R = cam.R @ cm.exp_so3(v[0:3])
            k.C = cam.C + v[3:6]
            k.f = cam.f * math.exp(float(np.clip(v[6], -30.0, 30.0)))     # a wild trial step must not overflow
            if self.fit_k1:
                k.dist = cam.dist.copy()
                k.dist[0] += v[7]
            out.append(k)
        wins = []
        for w, win in enumerate(rig.windows):
            if w >= self.n_wins or np.all(self.wpos[w] < 0):
                wins.append(win.copy())
                continue
            a, b, dl = (float(dx[g]) if g >= 0 else 0.0 for g in self.wpos[w])          # held: a = b = 0
            A = self.anchors[w] if self.anchors[w] is not None else win.d * win.n
            n_new = pm.retract_normal(win.n, (a, b))
            wins.append(pm.Window(n_new, float(n_new @ A) + (win.d - float(win.n @ A)) + dl, win.t, win.idx,
                                  win.name))
        return out, Rig(wins, list(rig.cam_win)), (L + float(dx[self.L_index]) if self.fit_L else L)


def prior_terms(cams, rig: Rig, layout: Layout):
    """The priors' residual rows r (k,) and their Jacobian J (k, P), in units of their sigma."""
    k = len(layout.priors)
    r = np.zeros(k)
    J = np.zeros((k, layout.P))
    for i, pr in enumerate(layout.priors):
        c = pr["cam"]
        if pr["kind"] == "logf":
            r[i] = (math.log(cams[c].f) - pr["mean"]) / pr["sigma"]
            g = layout.pos[c, CAM_COLS.index("logf")]
            if g >= 0:
                J[i, g] = 1.0 / pr["sigma"]
        else:
            w = rig.cam_win[c]
            win = rig.windows[w]
            h1 = win.distance(cams[c].C)
            r[i] = (h1 - pr["mean"]) / pr["sigma"]
            for j in range(3):                                   # dh1/dC = -n
                g = layout.pos[c, 3 + j]
                if g >= 0:
                    J[i, g] = -win.n[j] / pr["sigma"]
            if w < layout.n_wins and np.any(layout.wpos[w] >= 0):
                A = layout.anchors[w] if layout.anchors[w] is not None else win.d * win.n
                t1, t2 = pm.tangent_basis(win.n)
                dv = (float((A - cams[c].C) @ t1[0]), float((A - cams[c].C) @ t2[0]), 1.0)
                for j in range(3):
                    if layout.wpos[w, j] >= 0:
                        J[i, layout.wpos[w, j]] = dv[j] / pr["sigma"]
    return r, J


# ------------------------------------------------------------------ residuals and Jacobians
def evaluate(cams, rig: Rig, frames: Frames, L: float, obs: Obs, layout: Layout = None, rows: np.ndarray = None,
             anchors=None) -> Eval:
    """Reprojection residuals of the rows `rows` (default: all), and with a layout their Jacobians.  The Eval also
    carries each row's h1 (camera to window), hW (point's depth in the water) and tau (tan of the air angle)."""
    rows = np.arange(len(obs)) if rows is None else rows
    fr, ca, en = obs.frame[rows], obs.cam[rows], obs.end[rows]
    XA, XB = frames.ends(L)
    X = np.where((en == 0)[:, None], XA[fr], XB[fr])
    rw = cm.rows(cams, ca)
    wrow = np.asarray(rig.cam_win, int)[ca]
    anc = layout.anchors if (layout is not None and layout.n_wins) else anchors
    ww = pm.wrows(rig.windows, wrow, anchors=anc)
    pr = pm.project(rw, ww, X, jac=layout is not None)
    r = pr.uv - obs.uv[rows]
    ev = Eval(r=r, ok=pr.ok.copy())
    ev.h1, ev.hW, ev.tau = pr.h1, pr.hW, pr.tau
    if layout is None:
        return ev
    n = len(rows)
    Jg = np.zeros((n, 2, layout.P))
    for j, mi in enumerate(layout.model_idx):
        g = layout.pos[ca, j]
        sel = (g >= 0) & pr.ok
        Jg[np.flatnonzero(sel), :, g[sel]] = pr.J_cam[sel, :, mi]
    if layout.n_wins:
        wclip = np.clip(wrow, 0, layout.n_wins - 1)
        for j in range(3):
            g = np.where((wrow >= 0) & (wrow < layout.n_wins), layout.wpos[wclip, j], -1)
            sel = (g >= 0) & pr.ok
            Jg[np.flatnonzero(sel), :, g[sel]] = pr.J_win[sel, :, j]
    t1, t2 = frames.tangent_basis()
    sgn = np.where(en == 0, -0.5 * L, 0.5 * L)
    JX = np.where(pr.ok[:, None, None], pr.J_X, 0.0)
    Jf = np.zeros((n, 2, 5))
    Jf[:, :, 0:3] = JX
    Jf[:, :, 3] = np.matmul(JX, t1[fr][:, :, None])[:, :, 0] * sgn[:, None]
    Jf[:, :, 4] = np.matmul(JX, t2[fr][:, :, None])[:, :, 0] * sgn[:, None]
    if layout.fit_L:
        Jg[:, :, layout.L_index] = (np.matmul(JX, frames.w[fr][:, :, None])[:, :, 0]
                                    * np.where(en == 0, -0.5, 0.5)[:, None])
    ev.Jg, ev.Jf = Jg, Jf
    return ev


# ------------------------------------------------------------------ Levenberg-Marquardt
@dataclass
class Solution:
    cams: List[pm.Camera]
    rig: Rig
    frames: Frames
    L: float
    obs: Obs
    layout: Layout
    rms: float = float("nan")
    history: List[dict] = field(default_factory=list)
    cov: Optional[np.ndarray] = None
    sigma_px: float = float("nan")
    info: Dict = field(default_factory=dict)
    weights: Optional[np.ndarray] = None          # per observation, 1 / sigma^2 (px^-2); None: equal weights


def bundle_adjust(cams, rig: Rig, frames: Frames, L: float, obs: Obs, layout: Layout, max_iter: int = 100,
                  tol: float = 1e-10, lam0: float = 1e-3, verbose: bool = False, log=print,
                  weights: np.ndarray = None, prior_scale: float = 1.0) -> Solution:
    """Levenberg-Marquardt on the manifold over the active observations, frames eliminated (Schur), as
    `cylinder_wand.bundle_adjust`.  Stops when the largest step (rad; mm per |C|; log f; mm per the window's scale; mm
    per 10 mm and rad for the frames) falls below `tol`, or when no step lowers the cost.  A trial step that makes a
    projection fail is rejected like one that raises the cost.  With `weights` (per observation, 1 / sigma^2) the cost
    is sum w |r|^2; the reported RMS stays the unweighted one, in pixels.  Priors (layout.priors) add
    prior_scale^2 r_p^2 to the cost (prior_scale: the pixel noise the unweighted cost is measured in)."""
    cams = [c.copy() for c in cams]
    rig = rig.copy()
    frames = frames.copy()
    rows = np.flatnonzero(obs.active)
    wr = None if weights is None else np.asarray(weights, float)[rows]
    ps = 1.0 if weights is not None else float(prior_scale)

    def wcost(e, g, cs, rg):
        c0 = cost_of(e, g) if wr is None else float(np.sum(wr[g][:, None] * e.r[g] ** 2))
        if layout.priors:
            rp, _ = prior_terms(cs, rg, layout)
            c0 += ps * ps * float(rp @ rp)
        return c0

    hist = []
    ev = evaluate(cams, rig, frames, L, obs, layout, rows)
    good = ev.ok.copy()
    cost = wcost(ev, good, cams, rig)
    lam, nu = lam0, 2.0
    t0 = time.time()
    step_rel = np.inf
    for it in range(max_iter):
        fr = obs.frame[rows]
        V, gV, U, gU, W = normal_blocks(Eval(ev.r, good, ev.Jg, ev.Jf), fr, obs.n_frames, weights=wr)
        if layout.priors:
            rp, Jp = prior_terms(cams, rig, layout)
            V = V + ps * ps * (Jp.T @ Jp)
            gV = gV + ps * ps * (Jp.T @ rp)
        accepted = False
        r_g = np.where(good[:, None], ev.r, 0.0)
        Jg_g = np.where(good[:, None, None], ev.Jg, 0.0)
        Jf_g = np.where(good[:, None, None], ev.Jf, 0.0)
        for _inner in range(12):
            try:
                dc, df = solve_step(V, gV, U, gU, W, lam)
            except np.linalg.LinAlgError:
                lam *= nu
                nu *= 2
                continue
            lin = r_g + Jg_g @ dc + np.matmul(Jf_g, df[fr][:, :, None])[:, :, 0]
            pred = cost - float(np.sum(lin ** 2) if wr is None else np.sum(wr[:, None] * lin ** 2))
            if layout.priors:
                rl = rp + Jp @ dc
                pred -= ps * ps * float(rl @ rl)
            cams_t, rig_t, L_t = layout.apply(cams, rig, L, dc)
            frames_t = frames.retract(df)
            ev_t = evaluate(cams_t, rig_t, frames_t, L_t, obs, layout, rows)
            if np.any(good & ~ev_t.ok):
                lam *= nu
                nu *= 2
                continue
            cost_t = wcost(ev_t, good, cams_t, rig_t)
            rho = (cost - cost_t) / pred if pred > 0 else -1.0
            if rho > 0:
                step_rel = _step_size(dc, df, cams, rig, layout)
                cams, rig, frames, L = cams_t, rig_t, frames_t, L_t
                ev = ev_t
                good = good | ev.ok
                cost = wcost(ev, good, cams, rig)
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
                accepted = True
                break
            lam *= nu
            nu *= 2
        n_good = int(good.sum())
        rms = math.sqrt(cost_of(ev, good) / max(n_good, 1))
        hist.append({"iter": it, "rms_px": rms, "lambda": lam, "accepted": accepted, "seconds": time.time() - t0})
        if verbose:
            log(f"    LM {it:3d}: rms {rms:.6f} px, lambda {lam:.1e}, {'ok' if accepted else 'stuck'}")
        if not accepted:
            break
        if step_rel < tol or cost == 0.0:
            break
    sol = Solution(cams, rig, frames, L, obs, layout, rms=math.sqrt(cost_of(ev, good) / max(int(good.sum()), 1)),
                   history=hist, weights=None if weights is None else np.asarray(weights, float))
    sol.info["n_fail"] = int((~good).sum())
    sol.info["weighted_cost"] = cost
    return sol


def _step_size(dc, df, cams, rig: Rig, layout: Layout) -> float:
    """Largest step relative to a natural scale: radians; mm over |C| (the origin is near the wand cloud); log f; a
    window's offset in mm over its anchor's distance from its cameras; L and the frames' midpoints in mm over 10 mm,
    their directions in radians."""
    s = 0.0
    for c in range(layout.n_cams):
        for j, name in enumerate(layout.cols):
            g = layout.pos[c, j]
            if g < 0:
                continue
            v = abs(dc[g])
            if name.startswith("C"):
                v /= max(float(np.linalg.norm(cams[c].C)), 1.0)
            s = max(s, v)
    for w in range(layout.n_wins):
        if layout.wpos[w, 0] >= 0:
            s = max(s, abs(dc[layout.wpos[w, 0]]), abs(dc[layout.wpos[w, 1]]))
        if layout.wpos[w, 2] < 0:
            continue
        A = layout.anchors[w] if layout.anchors[w] is not None else rig.windows[w].d * rig.windows[w].n
        cs = [cams[c].C for c in rig.cams_of(w)] or [np.zeros(3)]
        scale = max(float(np.linalg.norm(A - np.mean(cs, axis=0))), 1.0)
        s = max(s, abs(dc[layout.wpos[w, 2]]) / scale)
    if layout.fit_L:
        s = max(s, abs(dc[layout.L_index]) / 10.0)
    if len(df):
        s = max(s, float(np.max(np.abs(df[:, :3]))) / 10.0, float(np.max(np.abs(df[:, 3:]))))
    return s


# ------------------------------------------------------------------ stage 4 with outlier rejection
def refine(cams, rig: Rig, frames: Frames, L: float, obs: Obs, layout: Layout, k_sigma: float = 5.0,
           max_rounds: int = 20, max_iter: int = 100, sigma_floor_px: float = 1e-3, verbose: bool = False, log=print,
           noise_weights: str = "camera", sigma_cam_px=None, min_group: int = 200, max_rel_se: float = 0.10,
           max_ratio: float = 10.0, sigma_tol: float = 1e-6, prior_scale: float = 1.0) -> Solution:
    """Stage 4, as `cylinder_wand.refine` (without its multi-image guard: a flat window gives one image): bundle
    adjustment, then a new active set chosen from every usable observation (residual within k_sigma robust sigmas,
    Mahalanobis; no frame left with an end seen by fewer than two cameras), repeated until the set and the per-group
    noise estimates (noise_weights "camera", "camera_end" or "none"; or fixed by sigma_cam_px) are steady; then the
    covariance.  Rejections are not final.  `obs.active` is updated in place."""
    sol = None
    t0 = time.time()
    usable = obs.active.copy()
    if noise_weights not in NOISE_MODES:
        raise ValueError(f"noise_weights must be one of {NOISE_MODES}")
    groups = noise_groups(obs, "camera" if sigma_cam_px is not None else noise_weights)
    n_groups = int(groups.max()) + 1 if len(groups) else 1
    # with two cameras each wand position's residual can be moved from one camera to the other, and the estimated
    # weights feed on that: equal weights (W5 seeds 2004, 2005, 2008)
    two_cams = sigma_cam_px is None and obs.n_cams == 2 and noise_weights != "none"
    estimate = sigma_cam_px is None and noise_weights != "none" and not two_cams
    sig2 = None
    if sigma_cam_px is not None:
        sig2 = np.asarray(sigma_cam_px, float) ** 2
        if sig2.shape != (obs.n_cams,) or not np.all(sig2 > 0):
            raise ValueError("sigma_cam_px needs one positive value per camera")
    vc = {}
    fallback = False
    converged = True
    changed = sig_changed = False
    rnd = 0
    lm_iters = 0
    budget = max_rounds
    steps = []                                    # each round's largest relative change of a group's sigma
    while rnd < budget:
        weights = None if sig2 is None else 1.0 / sig2[groups]
        sol = bundle_adjust(cams, rig, frames, L, obs, layout, max_iter=max_iter, verbose=verbose, log=log,
                            weights=weights, prior_scale=prior_scale)
        lm_iters += len(sol.history)
        cams, rig, frames, L = sol.cams, sol.rig, sol.frames, sol.L
        sig_fit = sig2
        sig_changed = False
        step = None
        if estimate and not fallback:
            cur = np.ones(n_groups) if sig2 is None else sig2
            new, vc = variance_components(sol, groups, cur, min_group=min_group, max_rel_se=max_rel_se,
                                          max_ratio=max_ratio, floor_px=sigma_floor_px)
            step = np.inf if sig2 is None else float(np.max(np.abs(np.sqrt(new / sig2) - 1.0)))
            sig_changed = step > sigma_tol
            if sig_changed:
                sig2 = new
            steps.append(step)
        sig_obs = None if sig_fit is None else np.sqrt(sig_fit[groups])
        changed, n_out, n_back = _update_active(sol, usable, k_sigma, sigma_floor_px, sig_obs=sig_obs,
                                                groups=None if sig_fit is None else groups)
        frames = sol.frames
        if verbose:
            log(f"  round {rnd}: rms {sol.rms:.5f} px, {len(sol.history)} LM iterations, rejected {n_out}, "
                f"re-admitted {n_back}" + ("" if sig2 is None else ", sigma per group " +
                                           ", ".join(f"{v:.4f}" for v in np.sqrt(sig2)) + " px"))
        rnd += 1
        # The weights are kept only if the data separate the groups.  That is judged as soon as the weights have
        # settled (within 1 %), when the standard errors mean something, and also when the updates stop shrinking:
        # with two cameras the estimate can drift away round by round, then distort the fit and reject good data
        # (tests/plate_wand W5 seed 2005: sigmas 0.95, 1.00 -> 0.14, 1.35 px over 16 rounds, 11 % rejected).
        # Falling back gives the equal-weight rounds a budget of their own.
        if estimate and not fallback and sig2 is not None:
            settled = step is not None and step < 0.01
            drifting = (len(steps) >= 4 and steps[-1] > steps[-2] > steps[-3]) or (len(steps) >= 6 and steps[-1] > 1e-3)
            if (settled and not vc.get("separable", False)) or drifting:
                fallback, sig2 = True, None
                vc["drifting"] = bool(drifting)
                budget = rnd + max_rounds
                continue
        if not changed and not sig_changed:
            break
    else:
        converged = False
        if estimate and not fallback and sig2 is not None and not vc.get("separable", False):
            fallback, sig2 = True, None
    weights = None if sig2 is None else 1.0 / sig2[groups]
    if changed or sig_changed or not converged:
        sol = bundle_adjust(cams, rig, frames, L, obs, layout, max_iter=max_iter, verbose=verbose, log=log,
                            weights=weights, prior_scale=prior_scale)
        lm_iters += len(sol.history)
    sol.info["rounds"] = rnd
    sol.info["lm_iterations"] = lm_iters
    sol.info["seconds"] = time.time() - t0
    sol.info["noise_weights"] = {"mode": noise_weights, "given": sigma_cam_px is not None, "estimated": estimate,
                                 "two_cameras": two_cams,
                                 "used": sig2 is not None, "fallback": fallback, "converged": converged,
                                 "sigma_group_px": None if sig2 is None else np.sqrt(sig2).tolist(), **vc}
    return covariance(sol, weights=weights, prior_scale=prior_scale)


def variance_components(sol: Solution, groups: np.ndarray, sig2: np.ndarray, min_group: int = 200,
                        max_rel_se: float = 0.10, max_ratio: float = 10.0, floor_px: float = 1e-3):
    """Each noise group's variance from the fitted residuals (MINQUE), as `cylinder_wand.variance_components`."""
    obs = sol.obs
    G = len(sig2)
    ev = evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs, Layout(obs.n_cams))
    act = obs.active & ev.ok
    cnt = np.bincount(groups[act], minlength=G)
    eff = np.where(cnt >= min_group, np.arange(G), G)
    _u, inv = np.unique(eff, return_inverse=True)
    E = len(_u)
    w = 1.0 / sig2[groups]
    q = np.zeros(E)
    M = np.zeros((E, E))
    rows = np.flatnonzero(act)
    if rows.size == 0:
        return np.asarray(sig2, float), {"groups": G, "separable": False, "counts": cnt.tolist()}
    rows = rows[np.argsort(obs.frame[rows], kind="stable")]
    fr = obs.frame[rows]
    cut = np.flatnonzero(np.r_[True, fr[1:] != fr[:-1], True])
    for a, b in zip(cut[:-1], cut[1:]):
        idx = rows[a:b]
        sw = np.sqrt(np.repeat(w[idx], 2))
        A = sw[:, None] * ev.Jf[idx].reshape(-1, 5)
        try:
            Hh = A @ np.linalg.solve(A.T @ A, A.T)
        except np.linalg.LinAlgError:
            continue
        R = np.eye(len(sw)) - Hh
        g = np.repeat(inv[groups[idx]], 2)
        Gm = np.zeros((len(g), E))
        Gm[np.arange(len(g)), g] = 1.0
        np.add.at(q, g, (sw * ev.r[idx].reshape(-1)) ** 2)
        M += Gm.T @ (R * R) @ Gm
    old_eff = np.array([np.mean(sig2[inv == e]) for e in range(E)])
    info = {"groups": G, "pooled_groups": int(np.sum(cnt < min_group)), "counts": cnt.tolist(),
            "max_rel_se": max_rel_se}
    separable = E >= 2
    tau = None
    if separable:
        try:
            Minv = np.linalg.inv(M)
            tau = Minv @ q
            rel_se = np.sqrt(np.maximum(np.diag(Minv), 0.0) / 2.0)
            info["rel_se"] = rel_se.tolist()
            info["rel_se_group"] = [float(rel_se[inv[g]]) for g in range(G)]
            info["sigma_px_separate"] = [float(math.sqrt(tau[inv[g]] * old_eff[inv[g]])) if tau[inv[g]] > 0 else None
                                         for g in range(G)]
            separable = bool(np.all(tau > 0) and np.max(rel_se) <= max_rel_se)
        except np.linalg.LinAlgError:
            separable = False
    info["separable"] = bool(separable)
    if tau is not None:
        new_eff = tau * old_eff
    else:
        r2 = np.sum(ev.r[act] ** 2)
        dof = max(2 * int(act.sum()) - 5 * len(cut[:-1]), 1)
        new_eff = np.full(E, r2 / dof)
    new_eff = np.maximum(new_eff, floor_px ** 2)
    new_eff = np.maximum(new_eff, np.max(new_eff) / max_ratio ** 2)
    new = new_eff[inv]
    info["sigma_px"] = np.sqrt(new).tolist()
    return new, info


def _update_active(sol: Solution, usable: np.ndarray, k_sigma: float, sigma_floor_px: float = 1e-3,
                   sig_obs: np.ndarray = None, groups: np.ndarray = None):
    """Choose the active set afresh from the usable observations at the current solution (Mahalanobis distances
    under each observation's predicted covariance, judged against its group's robust scale), as
    `cylinder_wand._update_active`.  Returns (changed, number newly rejected, number re-admitted)."""
    obs = sol.obs
    act_frame = np.zeros(obs.n_frames, bool)
    act_frame[obs.frame[obs.active]] = True
    stale = ~act_frame
    if stale.any():
        rows = np.flatnonzero(usable & stale[obs.frame])
        if rows.size:
            _restore_stale_frames(sol, rows, stale, k_sigma, sigma_floor_px, sig_obs)
    ev = evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs, Layout(obs.n_cams))
    act = obs.active & ev.ok
    if act.sum() < 10:
        return False, 0, 0
    Jf = np.where(ev.ok[:, None, None], ev.Jf, 0.0)
    wv = np.ones(len(obs)) if sig_obs is None else 1.0 / np.asarray(sig_obs, float) ** 2
    U = np.zeros((obs.n_frames, 5, 5))
    np.add.at(U, obs.frame[act], np.einsum("nai,naj->nij", Jf[act], Jf[act]) * wv[act][:, None, None])
    Uinv = np.zeros_like(U)
    good_f = np.linalg.det(U) > 0
    Uinv[good_f] = np.linalg.inv(U[good_f])
    Pm = np.einsum("nai,nij,nbj->nab", Jf, Uinv[obs.frame], Jf)
    if sig_obs is not None:
        Pm = Pm * wv[:, None, None]
    M = np.eye(2)[None] + np.where(obs.active, -1.0, 1.0)[:, None, None] * Pm
    lam, V = np.linalg.eigh(M)
    lam = np.maximum(lam, 1e-3)
    rr = np.where(ev.ok[:, None], ev.r, 0.0)
    proj = np.einsum("nai,na->ni", V, rr)
    d = np.sqrt(np.sum(proj ** 2 / lam, axis=1))
    d[~ev.ok] = np.inf
    sigma = float(np.median(d[act])) / math.sqrt(2.0 * math.log(2.0))
    if sig_obs is None:
        lim = k_sigma * max(sigma, sigma_floor_px)
    else:
        s_i = np.asarray(sig_obs, float)
        z = d / s_i
        c_obs = np.full(len(obs), float(np.median(z[act])) / math.sqrt(2.0 * math.log(2.0)))
        if groups is not None:
            for g in np.unique(groups[act]):
                sel = act & (groups == g)
                if sel.sum() >= 30:
                    c_obs[groups == g] = float(np.median(z[sel])) / math.sqrt(2.0 * math.log(2.0))
        lim = k_sigma * np.maximum(c_obs * s_i, sigma_floor_px)
        sol.info["robust_sigma_group_px"] = {int(g): float(np.median((c_obs * s_i)[groups == g]))
                                             for g in np.unique(groups)} if groups is not None else {}
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


def _restore_stale_frames(sol: Solution, rows: np.ndarray, stale: np.ndarray, k_sigma: float, sigma_floor_px: float,
                          sig_obs: np.ndarray = None, max_drop: int = 4):
    """Frames with no active observation were not adjusted; they are re-triangulated from their usable rays so that
    their good rays can come back.  Robustly: while a frame's worst ray lies beyond k_sigma robust sigmas of the
    others' fit (and dropping it leaves its end two rays), it is left out of the triangulation.  A gross outlier left
    in spoils the whole frame, whose good rays then never fit again: W7 (2 % outliers) lost the whole frame of every
    outlier, 1,001 good detections for 160 bad ones."""
    obs = sol.obs
    rw_w = None if sig_obs is None else 1.0 / np.asarray(sig_obs, float) ** 2
    act = np.flatnonzero(obs.active)
    ev0 = evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs, rows=act)
    rn = np.linalg.norm(np.where(ev0.ok[:, None], ev0.r, np.nan), axis=1)
    rn = rn[np.isfinite(rn)]
    scale = float(np.median(rn)) / math.sqrt(2.0 * math.log(2.0)) if rn.size else sigma_floor_px
    lim = k_sigma * max(scale, sigma_floor_px)
    keep = rows.copy()
    XA = XB = cnt = None
    for _ in range(max_drop + 1):
        XA, XB, cnt = triangulate(sol.cams, sol.rig, obs, keep, row_weights=rw_w)
        X = np.where((obs.end[keep] == 0)[:, None], XA[obs.frame[keep]], XB[obs.frame[keep]])
        fine = np.all(np.isfinite(X), axis=1)
        r = np.full(len(keep), np.inf)
        if fine.any():
            sub = keep[fine]
            pr = pm.project(cm.rows(sol.cams, obs.cam[sub]),
                            pm.wrows(sol.rig.windows, np.asarray(sol.rig.cam_win, int)[obs.cam[sub]]), X[fine])
            r[fine] = np.where(pr.ok, np.linalg.norm(pr.uv - obs.uv[sub], axis=1), np.inf)
        # per frame, the worst ray beyond the limit, if its end keeps two rays without it
        key = obs.frame[keep] * 2 + obs.end[keep]
        n_end = np.bincount(key, minlength=2 * obs.n_frames)
        order = np.lexsort((-r, obs.frame[keep]))
        first = np.r_[True, obs.frame[keep][order][1:] != obs.frame[keep][order][:-1]]
        worst = order[first]
        drop = worst[(r[worst] > lim) & (n_end[key[worst]] > 2)]
        if drop.size == 0:
            break
        keep = np.delete(keep, drop)
    ok = stale & np.all(cnt >= 2, axis=1) & np.all(np.isfinite(XA), axis=1) & np.all(np.isfinite(XB), axis=1)
    if ok.any():
        fr = frames_from_ends(XA[ok], XB[ok])
        sol.frames = sol.frames.copy()
        sol.frames.m[ok], sol.frames.w[ok] = fr.m, fr.w


def _log_likelihood_key(sol: Solution, noise_weights: str) -> float:
    """sum over noise groups of n_g log(mean squared residual of the group): smaller is more likely."""
    obs = sol.obs
    ev = evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs)
    act = obs.active & ev.ok
    groups = noise_groups(obs, "none" if noise_weights == "none" else noise_weights)
    key = 0.0
    for g in np.unique(groups[act]):
        sel = act & (groups == g)
        key += int(sel.sum()) * math.log(max(float(np.mean(np.sum(ev.r[sel] ** 2, axis=1))), 1e-300))
    return key


# ------------------------------------------------------------------ covariance and reports
def information(cams, rig: Rig, frames, L, obs: Obs, layout: Layout, rows=None, sigma_px: float = 1.0,
                weights: np.ndarray = None, prior_scale: float = 1.0):
    """Fisher information of the free parameters, the frames marginalised (Schur), for Gaussian pixel noise of
    sigma_px (or, with `weights` per observation, of variance sigma_px^2 / w); and the evaluation used.  Priors add
    their information."""
    rows = np.flatnonzero(obs.active) if rows is None else rows
    ev = evaluate(cams, rig, frames, L, obs, layout, rows)
    V, gV, U, gU, W = normal_blocks(ev, obs.frame[rows], obs.n_frames,
                                    weights=None if weights is None else np.asarray(weights, float)[rows])
    if layout.priors:
        ps = 1.0 if weights is not None else float(prior_scale)
        _rp, Jp = prior_terms(cams, rig, layout)
        V = V + ps * ps * (Jp.T @ Jp)
    S, _rhs, _Ui, _WUi = reduced_system(V, gV, U, gU, W, 0.0)
    return S / sigma_px ** 2, ev


def covariance(sol: Solution, sigma_px: float = None, weights: np.ndarray = None,
               prior_scale: float = 1.0) -> Solution:
    """Covariance of the free parameters, sigma^2 S^-1 with S the reduced (Schur) normal matrix, as
    `cylinder_wand.covariance` (with weights: s0^2 S_w^-1, s0^2 the variance factor)."""
    rows = np.flatnonzero(sol.obs.active)
    info1, ev = information(sol.cams, sol.rig, sol.frames, sol.L, sol.obs, sol.layout, rows, 1.0, weights=weights,
                            prior_scale=prior_scale)
    n_res = 2 * int(ev.ok.sum())
    dof = max(n_res - sol.layout.P - 5 * sol.obs.n_frames + len(sol.layout.priors), 1)
    s2 = float(np.sum(ev.r[ev.ok] ** 2)) / dof if sigma_px is None else sigma_px ** 2
    try:
        inv = np.linalg.inv(info1)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(info1)
    if weights is None:
        sol.cov = s2 * inv
    else:
        wr = np.asarray(weights, float)[rows]
        s02 = float(np.sum(wr[ev.ok][:, None] * ev.r[ev.ok] ** 2)) / dof if sigma_px is None else 1.0
        sol.cov = s02 * inv
        sol.info["variance_factor"] = s02
    sol.sigma_px = math.sqrt(s2)
    sol.info["dof"] = dof
    sol.info["cond_S"] = float(np.linalg.cond(info1))
    sol.info["h1_mm"] = [None if sol.rig.cam_win[c] < 0 else float(_win_of(sol.rig, c).distance(sol.cams[c].C))
                         for c in range(len(sol.cams))]
    return sol


def _win_of(rig: Rig, c: int):
    k = rig.cam_win[c]
    return rig.windows[k] if k >= 0 else None


def _cam_block(layout: Layout, cov: np.ndarray, c: int) -> np.ndarray:
    idx = layout.pos[c]
    full = np.zeros((len(layout.cols), len(layout.cols)))
    sel = np.flatnonzero(idx >= 0)
    full[np.ix_(sel, sel)] = cov[np.ix_(idx[sel], idx[sel])]
    return full


def camera_sigmas(cams, rig: Rig, layout: Layout, cov: np.ndarray) -> List[dict]:
    """Per camera: sigma of the rotation (deg, about the world axes), centre (mm), f (relative and px), and its
    distance to its window (mm, window and camera together)."""
    out = []
    for c, cam in enumerate(cams):
        full = _cam_block(layout, cov, c)
        d = {"rotation_deg": np.degrees(np.sqrt(np.maximum(np.diag(full)[0:3], 0.0))).tolist(),
             "centre_mm": np.sqrt(np.maximum(np.diag(full)[3:6], 0.0)).tolist(),
             "f_rel": math.sqrt(max(full[6, 6], 0.0)), "f_px": cam.f * math.sqrt(max(full[6, 6], 0.0))}
        if layout.fit_k1:
            d["k1"] = math.sqrt(max(full[7, 7], 0.0))
        w = rig.cam_win[c]
        if w >= 0:
            d["distance_mm"] = math.sqrt(max(_distance_var(cams, rig, layout, cov, c), 0.0))
        out.append(d)
    return out


def _distance_grad(cams, rig: Rig, layout: Layout, c: int) -> np.ndarray:
    """Gradient of camera c's distance to its window, h1 = d - n.C, over the free parameters."""
    g = np.zeros(layout.P)
    w = rig.cam_win[c]
    win = rig.windows[w]
    for j in range(3):
        k = layout.pos[c, 3 + j]
        if k >= 0:
            g[k] = -win.n[j]
    if w < layout.n_wins and np.any(layout.wpos[w] >= 0):
        A = layout.anchors[w] if layout.anchors[w] is not None else win.d * win.n
        t1, t2 = pm.tangent_basis(win.n)
        dv = (float((A - cams[c].C) @ t1[0]), float((A - cams[c].C) @ t2[0]), 1.0)
        for j in range(3):
            if layout.wpos[w, j] >= 0:
                g[layout.wpos[w, j]] = dv[j]
    return g


def _distance_var(cams, rig, layout, cov, c) -> float:
    g = _distance_grad(cams, rig, layout, c)
    return float(g @ cov @ g)


def window_sigmas(cams, rig: Rig, layout: Layout, cov: np.ndarray) -> List[dict]:
    """Per window: sigma of the normal's direction (deg, both tangent angles combined), of its offset at the anchor
    (mm), and of each of its cameras' distance to it (mm)."""
    out = []
    for w, win in enumerate(rig.windows):
        g = layout.wpos[w] if w < layout.n_wins else np.full(3, -1)
        d = {"fixed": bool(np.all(g < 0))}
        if g[0] >= 0:
            d["normal_deg"] = math.degrees(math.sqrt(max(cov[g[0], g[0]] + cov[g[1], g[1]], 0.0)))
        elif g[2] >= 0:
            d["normal_held"] = True
        if g[2] >= 0:
            d["offset_mm"] = math.sqrt(max(cov[g[2], g[2]], 0.0))
        d["camera_distance_mm"] = {int(c + 1): math.sqrt(max(_distance_var(cams, rig, layout, cov, c), 0.0))
                                   for c in rig.cams_of(w)}
        out.append(d)
    return out


def point_jacobians(cams, rig: Rig, layout: Layout, P: np.ndarray):
    """For fixed world points P (m, 3): per camera c, d(uv)/d(free params) (m, 2, P_free) (camera and its window),
    d(uv)/dX (m, 2, 3), the pixels and ok."""
    out = []
    P = np.asarray(P, float)
    for c, cam in enumerate(cams):
        rw = cm.rows([cam], np.zeros(len(P), int))
        ww = pm.wrows(rig.windows, np.full(len(P), rig.cam_win[c]), anchors=layout.anchors if layout.n_wins else None)
        pr = pm.project(rw, ww, P, jac=True)
        J = np.zeros((len(P), 2, layout.P))
        for j, mi in enumerate(layout.model_idx):
            g = layout.pos[c, j]
            if g >= 0:
                J[:, :, g] = pr.J_cam[:, :, mi]
        w = rig.cam_win[c]
        if 0 <= w < layout.n_wins:
            for j in range(3):
                g = layout.wpos[w, j]
                if g >= 0:
                    J[:, :, g] = pr.J_win[:, :, j]
        J[~pr.ok] = np.nan
        out.append((J, pr.J_X, pr.uv, pr.ok))
    return out


def _stabiliser(rig: Rig) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The rigid motions that map every window onto itself: translations along every window's plane (rows of the
    returned (k, 3) basis) and, when all normals are parallel, the rotation about that normal (else None)."""
    N = np.array([w.n for w in rig.windows]) if rig.windows else np.zeros((0, 3))
    if len(N) == 0:
        return np.eye(3), None
    _u, s, Vt = np.linalg.svd(N)
    rank = int(np.sum(s > 1e-9 * max(s[0], 1.0)))
    T = Vt[rank:]
    axis = N[0] if rank == 1 else None
    return T, axis


def predicted_projection_error(cams, rig: Rig, layout: Layout, cov: np.ndarray, P: np.ndarray, sizes=None):
    """Cramer-Rao prediction of the calibration's error on unseen points P, RMS per 2-D image:
      fixed       projecting a known point, in the gauge of the fit (camera 1 held);
      stabiliser  the same, after the best common motion of all the points that maps every window onto itself
                  (one window: slides along it and turns about its normal; two: the slide along their intersection):
                  a point fixed to the window(s) rather than to camera 1;
      retriang    each point re-triangulated from all cameras: what the cameras disagree on (gauge-free; what
                  particle tracking feels).
    Points not seen by every camera (sensor or window) are skipped."""
    pj = point_jacobians(cams, rig, layout, P)
    ok = np.all([p[3] for p in pj], axis=0)
    if sizes is not None:
        for (J, JX, uv, okc), (w, h) in zip(pj, sizes):
            ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    T, axis = _stabiliser(rig)
    fixed, retri = [], []
    per_cam = [[] for _ in cams]
    Jall, Gall = [], []
    for i in np.flatnonzero(ok):
        Jc = np.vstack([pj[c][0][i] for c in range(len(cams))])
        JX = np.vstack([pj[c][1][i] for c in range(len(cams))])
        Sig = Jc @ cov @ Jc.T
        for c in range(len(cams)):
            v = float(np.trace(Sig[2 * c:2 * c + 2, 2 * c:2 * c + 2]))
            fixed.append(v)
            per_cam[c].append(v)
        Pperp = np.eye(len(JX)) - JX @ np.linalg.pinv(JX)
        retri.append(float(np.trace(Pperp @ Sig @ Pperp.T)) / len(cams))
        Jall.append(Jc)
        gens = [t for t in T]
        if axis is not None:
            gens.append(np.cross(axis, P[i]))
        Gall.append(JX @ np.column_stack(gens) if gens else np.zeros((len(JX), 0)))
    out = {"n_points": int(ok.sum()),
           "fixed_rms_px": math.sqrt(np.mean(fixed)) if fixed else float("nan"),
           "fixed_rms_px_per_cam": [math.sqrt(np.mean(v)) if v else float("nan") for v in per_cam],
           "retriangulated_rms_px": math.sqrt(np.mean(retri)) if retri else float("nan")}
    if Jall and Gall[0].shape[1]:
        J = np.vstack(Jall)
        G = np.vstack(Gall)
        Jp = J - G @ np.linalg.lstsq(G, J, rcond=None)[0]
        Lc = np.linalg.cholesky(cov + 1e-300 * np.eye(len(cov)))
        A = Jp @ Lc
        out["stabiliser_rms_px"] = math.sqrt(float(np.sum(A * A)) / (len(Jall) * len(cams)))
    else:
        out["stabiliser_rms_px"] = out["fixed_rms_px"]
    return out


# ------------------------------------------------------------------ triangulation
def triangulate(cams, rig: Rig, obs: Obs, rows=None, n_frames: int = None, row_weights: np.ndarray = None):
    """Each frame's ends from the back-projected rays of their detections (least squares of the offsets, weighted to
    pixels by f / path, and by row_weights when given).  Returns XA, XB (F, 3) and how many rays each end had."""
    rows = np.flatnonzero(obs.active) if rows is None else rows
    F = obs.n_frames if n_frames is None else n_frames
    rw = cm.rows(cams, obs.cam[rows])
    ww = pm.wrows(rig.windows, np.asarray(rig.cam_win, int)[obs.cam[rows]])
    o, d, ok = pm.rays(rw, ww, obs.uv[rows])
    path = np.linalg.norm(o - rw.C, axis=1) + 50.0
    wgt = np.where(ok, (rw.f / path) ** 2, 0.0)
    if row_weights is not None:
        wgt = wgt * np.asarray(row_weights, float)[rows]
    Pm = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = wgt[:, None, None] * Pm
    b = np.einsum("nij,nj->ni", A, np.where(ok[:, None], o, 0.0))
    key = obs.frame[rows] * 2 + obs.end[rows]
    AA = np.zeros((2 * F, 3, 3))
    bb = np.zeros((2 * F, 3))
    cnt = np.zeros(2 * F, int)
    np.add.at(AA, key, A)
    np.add.at(bb, key, b)
    np.add.at(cnt, key, ok.astype(int))
    X = np.full((2 * F, 3), np.nan)
    good = cnt >= 2
    good[good] &= np.abs(np.linalg.det(AA[good])) > 0
    X[good] = np.linalg.solve(AA[good], bb[good][:, :, None])[:, :, 0]
    return X[0::2], X[1::2], cnt.reshape(F, 2)


def initial_frames(cams, rig: Rig, obs: Obs, L: float) -> Frames:
    """Frames from the triangulated ends, set exactly L apart about their midpoint; frames that cannot be
    triangulated get a placeholder (and should be inactive)."""
    XA, XB, _cnt = triangulate(cams, rig, obs)
    bad = ~np.all(np.isfinite(XA), axis=1) | ~np.all(np.isfinite(XB), axis=1)
    XA[bad] = 0.0
    XB[bad] = np.array([0.0, 0.0, 1.0])
    fr = frames_from_ends(XA, XB)
    fr.w[~np.all(np.isfinite(fr.w), axis=1)] = np.array([0.0, 0.0, 1.0])
    return fr


def wand_lengths(cams, rig: Rig, obs: Obs, rows=None):
    """Free triangulation (no length constraint) of every frame: the measured wand length per frame."""
    XA, XB, cnt = triangulate(cams, rig, obs, rows)
    Lm = np.linalg.norm(XB - XA, axis=1)
    Lm[np.any(cnt < 2, axis=1)] = np.nan
    return Lm


def triangulate_ml(cams, rig: Rig, cam_idx: np.ndarray, uv: np.ndarray, point_idx: np.ndarray, X0: np.ndarray,
                   sigma_px: float = 1.0, iters: int = 15):
    """Maximum-likelihood triangulation: each point X_p minimises the reprojection error of its detections (rows with
    point_idx == p), by Gauss-Newton from X0.  Returns X (P, 3), the residuals (n, 2) and each point's covariance
    sigma^2 (J^T J)^-1 (P, 3, 3)."""
    X = np.array(X0, float, copy=True)
    Pn = len(X)
    rw = cm.rows(cams, cam_idx)
    ww = pm.wrows(rig.windows, np.asarray(rig.cam_win, int)[cam_idx])
    for _ in range(iters):
        pr = pm.project(rw, ww, X[point_idx], jac=True)
        r = np.where(pr.ok[:, None], pr.uv - uv, np.nan)
        J = np.where(pr.ok[:, None, None], pr.J_X, 0.0)
        rr = np.nan_to_num(r)
        H = np.zeros((Pn, 3, 3))
        g = np.zeros((Pn, 3))
        np.add.at(H, point_idx, np.einsum("nai,naj->nij", J, J))
        np.add.at(g, point_idx, np.einsum("nai,na->ni", J, rr))
        good = np.linalg.det(H) > 0
        dX = np.zeros((Pn, 3))
        dX[good] = -np.linalg.solve(H[good], g[good][:, :, None])[:, :, 0]
        X += dX
        if np.max(np.abs(dX)) < 1e-12:
            break
    pr = pm.project(rw, ww, X[point_idx], jac=True)
    r = np.where(pr.ok[:, None], pr.uv - uv, np.nan)
    J = np.where(pr.ok[:, None, None], pr.J_X, 0.0)
    H = np.zeros((Pn, 3, 3))
    np.add.at(H, point_idx, np.einsum("nai,naj->nij", J, J))
    cov = np.full((Pn, 3, 3), np.nan)
    good = np.linalg.det(H) > 0
    cov[good] = sigma_px ** 2 * np.linalg.inv(H[good])
    return X, r, cov


# ------------------------------------------------------------------ fitting from a start
def solve(data: WandData, cams, rig: Rig, fit_k1: bool = False, fit_L: bool = False, k_sigma: float = 5.0,
          noise_weights: str = "camera", sigma_cam_px=None, fixed_windows: Sequence[int] = (), obs: Obs = None,
          frames: Frames = None, verbose: bool = False, log=print, priors: Sequence[dict] = (),
          prior_scale: float = 1.0, held_normals: Sequence[int] = ()) -> Solution:
    """Stage 4-5 from given cameras and windows (in the gauge: camera 1's pose is held): frames triangulated, then
    `refine`.  Used by `calibrate` and by the tests that start from the truth.  held_normals: windows whose normal is
    held (their offset is fitted)."""
    obs = observations(data) if obs is None else obs
    layout = Layout(obs.n_cams, len(rig.windows), fit_k1=fit_k1, fit_L=fit_L, fixed_windows=fixed_windows,
                    anchors=anchors_for(cams, rig), held_normals=held_normals)
    for p in priors:
        layout.add_prior(p["kind"], p["cam"], p["mean"], p["sigma"])
    if frames is None:
        frames = initial_frames(cams, rig, obs, data.L)
    sol = refine(cams, rig, frames, data.L, obs, layout, k_sigma=k_sigma, noise_weights=noise_weights,
                 sigma_cam_px=sigma_cam_px, verbose=verbose, log=log, prior_scale=prior_scale)
    return sol


# ------------------------------------------------------------------ stage 3 tools: back-projection residual, coarse fit
def backprojection_residual(cams, rig: Rig, obs: Obs, L: float, rows, penalty: float = 30.0):
    """Stage 3 residual (n, 2): the offset of each frame's end from the observation's ray, in pixels (scaled by f over
    the apparent path, the water part shortened by n_cam / n_obj), with the ends triangulated from the rays and then
    set exactly L apart about their midpoint (the points never lag behind the cameras).  An end that lies behind the
    ray's start (on the camera side of the window, where it cannot be seen) adds its distance from that surface,
    in the same pixel scale, so the fit cannot place a window beyond the wand.  Always defined: rays that miss a
    window and frames that cannot be triangulated get `penalty`."""
    rw = cm.rows(cams, obs.cam[rows])
    wrow = np.asarray(rig.cam_win, int)[obs.cam[rows]]
    ww = pm.wrows(rig.windows, wrow)
    o, d, ok = pm.rays(rw, ww, obs.uv[rows])
    F = obs.n_frames
    fr, en = obs.frame[rows], obs.end[rows]
    path0 = np.linalg.norm(o - rw.C, axis=1)
    ratio = np.where(ww.has, ww.idx[:, 0] / ww.idx[:, -1], 1.0)
    wgt = np.where(ok, (rw.f / (path0 + 50.0)) ** 2, 0.0)
    Pm = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = wgt[:, None, None] * Pm
    b = np.matmul(A, np.where(ok[:, None], o, 0.0)[:, :, None])[:, :, 0]
    key = fr * 2 + en
    AA = np.zeros((2 * F, 3, 3))
    bb = np.zeros((2 * F, 3))
    cnt = np.zeros(2 * F, int)
    np.add.at(AA, key, A)
    np.add.at(bb, key, b)
    np.add.at(cnt, key, ok.astype(int))
    good = cnt >= 2
    good[good] &= np.abs(np.linalg.det(AA[good])) > 0
    Xe = np.full((2 * F, 3), np.nan)
    Xe[good] = np.linalg.solve(AA[good], bb[good][:, :, None])[:, :, 0]
    XA, XB = Xe[0::2], Xe[1::2]
    dv = XB - XA
    nd = np.linalg.norm(dv, axis=1)
    u = dv / np.where(nd > 0, nd, 1.0)[:, None]
    m = 0.5 * (XA + XB)
    X = np.where((en == 0)[:, None], (m - 0.5 * L * u)[fr], (m + 0.5 * L * u)[fr])
    with np.errstate(invalid="ignore"):
        g, s, _e1, _e2 = cm.ray_offsets(np.nan_to_num(X), o, d, rw.R[:, 0, :])
        scale = rw.f / (path0 + ratio * np.maximum(s, 0.0) + 1e-9)
        r = g * scale[:, None]
        r += (scale * np.maximum(-s, 0.0) / math.sqrt(2.0))[:, None]
    bad = ~ok | ~np.isfinite(r).all(axis=1) | ~np.isfinite(X).all(axis=1)
    r[bad] = penalty
    return r


def coarse_fit(cams, rig: Rig, obs: Obs, L: float, layout: Layout, max_outer: int = 4, loss: str = "soft_l1",
               f_scale: float = 2.0, max_nfev: int = 40, rows=None, verbose: bool = False, log=print):
    """Stage 3: least squares of the back-projection residual over the free camera and window parameters (frames
    eliminated by triangulation), re-anchored after every round.  Returns (cameras, rig, info)."""
    from scipy.optimize import least_squares
    rows = np.flatnonzero(obs.active) if rows is None else rows
    anc_c, anc_r = [c.copy() for c in cams], rig.copy()
    hist = []
    for outer in range(max_outer):
        lay = Layout(layout.n_cams, layout.n_wins, fit_k1=layout.fit_k1, fit_L=False, gauge=layout.gauge,
                     fixed_windows=layout.fixed_windows, anchors=anchors_for(anc_c, anc_r),
                     held_normals=layout.held_normals)

        def fun(x, anc_c=anc_c, anc_r=anc_r, lay=lay):
            cs, rg, _ = lay.apply(anc_c, anc_r, L, x)
            return backprojection_residual(cs, rg, obs, L, rows).ravel()

        res = least_squares(fun, np.zeros(lay.P), method="trf", x_scale="jac", loss=loss, f_scale=f_scale,
                            max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        anc_c, anc_r, _ = lay.apply(anc_c, anc_r, L, res.x)
        rms = float(np.sqrt(np.mean(res.fun ** 2) * 2))
        step = _step_size(res.x, np.zeros((0, 5)), anc_c, anc_r, lay)
        hist.append({"outer": outer, "rms_px": rms, "nfev": int(res.nfev), "step": step})
        if verbose:
            log(f"    coarse round {outer}: rms {rms:.4f} px, nfev {res.nfev}, step {step:.1e}")
        if step < 1e-8:
            break
    return anc_c, anc_r, {"history": hist, "rms_px": hist[-1]["rms_px"]}


# ------------------------------------------------------------------ resection (one or several cameras, points fixed)
def _robust_rho(r2, f_scale):
    return 2.0 * (np.sqrt(1.0 + r2 / f_scale ** 2) - 1.0)


def resect(cams: List[pm.Camera], wins: List[Optional[pm.Window]], X: List[np.ndarray], uv: List[np.ndarray],
           iters: int = 10, f_scale: float = 2.0, penalty: float = 30.0, lam0: float = 1e-2, keep_h1: bool = False):
    """Each camera's (w, C, logf) with its points X[k] fixed, by Levenberg-Marquardt on the soft-L1 cost at f_scale px
    (IRLS), every camera independently but in one vectorised projection per iteration.  A detection whose point does
    not project costs as a residual of `penalty` px.  With keep_h1 a camera behind a window moves only parallel to it,
    so its distance to the window stays as given: a candidate window then stands for a window at that distance (left
    free, a narrow-view camera slides onto its window, where it acts as a pinhole in the water and fits any window
    nearly as well -- tests/plate_wand/d3_single.py).  Returns (cameras, robust cost per camera)."""
    K = len(cams)
    B = []
    for k in range(K):
        if keep_h1 and wins[k] is not None:
            t1, t2 = pm.tangent_basis(wins[k].n)
            B.append(np.column_stack([t1[0], t2[0]]))
        else:
            B.append(np.eye(3))
    cam_of = np.concatenate([np.full(len(x), k, int) for k, x in enumerate(X)])
    Xall = np.vstack(X)
    Uall = np.vstack(uv)
    pen = _robust_rho(penalty ** 2, f_scale)

    def run(cs):
        rw = cm.rows(cs, cam_of)
        ww = pm.wrows(wins, cam_of)
        pr = pm.project(rw, ww, Xall, jac=True)
        r = pr.uv - Uall
        r2 = np.where(pr.ok, np.sum(np.nan_to_num(r) ** 2, axis=1), 0.0)
        rho = np.where(pr.ok, _robust_rho(r2, f_scale), pen)
        cost = np.bincount(cam_of, weights=rho, minlength=K)
        return pr, r, r2, cost

    cs = [c.copy() for c in cams]
    pr, r, r2, cost = run(cs)
    lam = np.full(K, lam0)
    for _ in range(iters):
        wts = np.where(pr.ok, 1.0 / np.sqrt(1.0 + r2 / f_scale ** 2), 0.0)
        J = np.where(pr.ok[:, None, None], np.nan_to_num(pr.J_cam[:, :, :7]), 0.0)
        rr = np.where(pr.ok[:, None], np.nan_to_num(r), 0.0)
        trial = []
        for k in range(K):
            sel = cam_of == k
            Jk = J[sel]
            nb = B[k].shape[1]
            Jk = np.concatenate([Jk[:, :, 0:3], np.matmul(Jk[:, :, 3:6], B[k]), Jk[:, :, 6:7]], axis=2)
            wk = wts[sel]
            JT = Jk.transpose(0, 2, 1)
            H = np.einsum("n,nij->ij", wk, np.matmul(JT, Jk))
            g = np.einsum("n,ni->i", wk, np.matmul(JT, rr[sel][:, :, None])[:, :, 0])
            dg = np.diag(H).copy()
            dg[dg <= 0] = 1.0
            try:
                dx = -np.linalg.solve(H + lam[k] * np.diag(dg), g)
            except np.linalg.LinAlgError:
                dx = np.zeros(4 + nb)
            c = cs[k].copy()
            c.R = cs[k].R @ cm.exp_so3(dx[0:3])
            c.C = cs[k].C + B[k] @ dx[3:3 + nb]
            c.f = cs[k].f * math.exp(float(np.clip(dx[3 + nb], -1.0, 1.0)))
            trial.append(c)
        pr_t, r_t, r2_t, cost_t = run(trial)
        better = cost_t < cost
        if not better.any():
            lam *= 10.0
            if np.all(lam > 1e8):
                break
            continue
        for k in range(K):
            if better[k]:
                cs[k] = trial[k]
                lam[k] = max(lam[k] / 3.0, 1e-9)
            else:
                lam[k] *= 10.0
        if better.all():
            pr, r, r2, cost = pr_t, r_t, r2_t, cost_t
        else:
            pr, r, r2, cost = run(cs)
    return cs, cost


def resect_window(cams: List[pm.Camera], win: pm.Window, X: List[np.ndarray], uv: List[np.ndarray],
                  iters: int = 30, f_scale: float = 2.0, penalty: float = 30.0, lam0: float = 1e-2,
                  hold_normal: bool = False):
    """The cameras behind one window and the window itself -- each camera's (w, C, logf) and the window's (a, b,
    delta) about the cameras' mean chief-ray crossing, only delta with hold_normal -- with the points X[k] fixed:
    Levenberg-Marquardt on the soft-L1 cost at f_scale px (IRLS), exact derivatives.  Returns (cameras, window, robust
    cost per camera)."""
    K = len(cams)
    cam_of = np.concatenate([np.full(len(x), k, int) for k, x in enumerate(X)])
    Xall = np.vstack(X)
    Uall = np.vstack(uv)
    pen = _robust_rho(penalty ** 2, f_scale)
    P = 7 * K + (1 if hold_normal else 3)

    def run(cs, wn, A):
        rw = cm.rows(cs, cam_of)
        ww = pm.wrows([wn], np.zeros(len(cam_of), int), anchors=[A])
        pr = pm.project(rw, ww, Xall, jac=True)
        r = pr.uv - Uall
        r2 = np.where(pr.ok, np.sum(np.nan_to_num(r) ** 2, axis=1), 0.0)
        rho = np.where(pr.ok, _robust_rho(r2, f_scale), pen)
        return pr, r, r2, float(np.sum(rho)), np.bincount(cam_of, weights=rho, minlength=K)

    cs = [c.copy() for c in cams]
    wn = win.copy()
    A = anchors_for(cs, Rig([wn], [0] * K))[0]
    pr, r, r2, cost, per = run(cs, wn, A)
    lam = lam0
    for _ in range(iters):
        wts = np.where(pr.ok, 1.0 / np.sqrt(1.0 + r2 / f_scale ** 2), 0.0)
        J = np.zeros((len(cam_of), 2, P))
        Jc = np.where(pr.ok[:, None, None], np.nan_to_num(pr.J_cam[:, :, :7]), 0.0)
        for k in range(K):
            sel = cam_of == k
            J[sel, :, 7 * k:7 * k + 7] = Jc[sel]
        Jw = np.where(pr.ok[:, None, None], np.nan_to_num(pr.J_win), 0.0)
        J[:, :, 7 * K:] = Jw[:, :, 2:3] if hold_normal else Jw
        rr = np.where(pr.ok[:, None], np.nan_to_num(r), 0.0)
        J2 = (J * np.sqrt(wts)[:, None, None]).reshape(-1, P)
        H = J2.T @ J2
        g = J2.T @ (rr * np.sqrt(wts)[:, None]).reshape(-1)
        dg = np.diag(H).copy()
        dg[dg <= 0] = 1.0
        improved = False
        for _inner in range(8):
            try:
                dx = -np.linalg.solve(H + lam * np.diag(dg), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            ct = []
            for k in range(K):
                c = cs[k].copy()
                v = dx[7 * k:7 * k + 7]
                c.R = cs[k].R @ cm.exp_so3(v[0:3])
                c.C = cs[k].C + v[3:6]
                c.f = cs[k].f * math.exp(float(np.clip(v[6], -1.0, 1.0)))
                ct.append(c)
            a, b, dl = (0.0, 0.0, dx[7 * K]) if hold_normal else dx[7 * K:]
            n_new = wn.n if hold_normal else pm.retract_normal(wn.n, (a, b))
            wt = pm.Window(n_new, float(n_new @ A) + (wn.d - float(wn.n @ A)) + dl, wn.t, wn.idx, wn.name)
            pr_t, r_t, r2_t, cost_t, per_t = run(ct, wt, A)
            if cost_t < cost:
                cs, wn, pr, r, r2, cost, per = ct, wt, pr_t, r_t, r2_t, cost_t, per_t
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved:
            break
    return cs, wn, per


# ------------------------------------------------------------------ stage 1: affine start (with a camera subset)
def _restrict_obs(obs: Obs, cams_keep: Sequence[int]) -> Obs:
    """The rows of the cameras in cams_keep, renumbered 0..k-1; the frames keep their numbers; frames left with an
    end seen by fewer than two of them are inactive."""
    cams_keep = list(cams_keep)
    new = np.full(obs.n_cams, -1)
    new[cams_keep] = np.arange(len(cams_keep))
    sel = new[obs.cam] >= 0
    o = Obs(obs.frame[sel], new[obs.cam[sel]], obs.end[sel], obs.uv[sel], obs.n_frames, len(cams_keep),
            obs.frame_ids, obs.active[sel].copy())
    _drop_thin_frames(o)
    return o


def _whole_seen(obs: Obs) -> np.ndarray:
    seen = np.zeros((obs.n_frames, obs.n_cams, 2), bool)
    act = obs.active
    seen[obs.frame[act], obs.cam[act], obs.end[act]] = True
    return seen.all(axis=2)


def affine_start(obs: Obs, L: float, min_frames: int = 12):
    """Stage 1: `cylinder_wand.affine_factorization` of the frames every camera sees whole; with fewer than
    min_frames such frames, of the largest camera subset that has them (cameras dropped greedily, each time the one
    whose removal most increases the count).  Returns (Affine, subset)."""
    subset = list(range(obs.n_cams))
    whole = _whole_seen(obs)
    while whole[:, subset].all(axis=1).sum() < min_frames and len(subset) > 2:
        best = max(subset, key=lambda c: int(whole[:, [k for k in subset if k != c]].all(axis=1).sum()))
        subset.remove(best)
    if len(subset) == obs.n_cams:
        return cw.affine_factorization(obs, L, min_frames=min_frames), subset
    return cw.affine_factorization(_restrict_obs(obs, subset), L, min_frames=min_frames), subset


# ------------------------------------------------------------------ stage 2: pinhole start
def _distance_search(R, q, sc, X, uv, cxcy, rc, n: int = 24):
    """Fallback for `_distance_from_magnification`: the distance along the principal ray (from q) of a pinhole camera
    with rotation R and f = sc D that best fits the detections uv of the affine points X (a geometric grid from 3 to
    3000 cloud radii, then golden-section in log D).  Biased far: the affine points absorb perspective."""
    z = R[2]

    def rms_at(D):
        C = q - D * z
        f = sc * D
        pc = (X - C) @ R.T
        front = pc[:, 2] > 1e-9 * D
        if front.sum() < max(6, len(X) // 2):
            return np.inf
        u = f * pc[front, :2] / pc[front, 2:3] + cxcy
        e = np.sum((u - uv[front]) ** 2, axis=1)
        return float(np.sqrt(np.mean(e))) * (len(X) / front.sum())

    grid = np.geomspace(3.0 * rc, 3000.0 * rc, n)
    vals = np.array([rms_at(D) for D in grid])
    k = int(np.argmin(vals))
    lo, hi = math.log(grid[max(k - 1, 0)]), math.log(grid[min(k + 1, n - 1)])
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fa, fb = rms_at(math.exp(a)), rms_at(math.exp(b))
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - gr * (hi - lo)
            fa = rms_at(math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + gr * (hi - lo)
            fb = rms_at(math.exp(b))
    D = math.exp(0.5 * (lo + hi))
    return D, {"grid_min_at_end": k in (0, n - 1), "rms_px": rms_at(D)}


def _distance_from_magnification(R, XA, XB, uvA, uvB, L: float, centre):
    """A camera's distance from the wand cloud by perspective: a wand at depth z beyond the cloud's centre (along the
    camera's axis R[2]) looks f / (D + z) px per mm long across the view, so L |w_perp| / |uv_B - uv_A| = (D + z) / f
    is linear in z, with slope 1/f and intercept D/f.  The directions w and depths z come from the affine
    reconstruction, the length from the known L (the affine reconstruction stretches near wands to explain their
    larger images, so its own lengths hide the effect).  Wands at least half across the view are used, with a
    Huber-weighted line fit.  Returns (D, f, slope) -- the slope is negative for the wrong mirror image."""
    dX = XB - XA
    w = dX / np.linalg.norm(dX, axis=1, keepdims=True)
    z = (0.5 * (XA + XB) - centre) @ R[2]
    lp = L * np.linalg.norm(w - (w @ R[2])[:, None] * R[2], axis=1)
    lu = np.linalg.norm(uvB - uvA, axis=1)
    sel = (lp > 0.5 * L) & np.isfinite(lu) & (lu > 0)
    if sel.sum() < 8:
        return float("nan"), float("nan"), 0.0
    y = lp[sel] / lu[sel]
    A = np.column_stack([np.ones(int(sel.sum())), z[sel]])
    wts = np.ones(len(y))
    coef = np.zeros(2)
    for _ in range(10):
        coef = np.linalg.lstsq(A * np.sqrt(wts)[:, None], y * np.sqrt(wts), rcond=None)[0]
        r = y - A @ coef
        sr = 1.4826 * float(np.median(np.abs(r)))
        wts = np.minimum(1.0, 1.345 * sr / np.maximum(np.abs(r), 1e-300))
    b, a = float(coef[0]), float(coef[1])
    if not (a > 0 and b > 0):
        return float("nan"), float("nan"), a
    return b / a, 1.0 / a, a


def _affine_pinholes(aff, mirror: bool, obs_sub: Obs, cxcy, sizes, L: float, f_hint=None):
    """Pinhole cameras from the affine cameras of one mirror image: rotation from the polar decomposition, the
    principal ray from the principal point, the distance from a hint (f / magnification), else by perspective
    (`_distance_from_magnification`), else by `_distance_search`.  Returns (cameras, info per camera)."""
    P = np.diag([1.0, 1.0, -1.0]) if mirror else np.eye(3)
    XA, XB = aff.XA @ P, aff.XB @ P
    cloud = np.vstack([XA, XB])
    centre = cloud.mean(axis=0)
    rc = float(np.sqrt(np.mean(np.sum((cloud - centre) ** 2, axis=1))))
    col = np.full(obs_sub.n_frames, -1)
    col[aff.frames] = np.arange(len(aff.frames))
    cams, info = [], []
    for c in range(len(aff.M)):
        M = aff.M[c] @ P
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        O = U @ Vt
        R = np.vstack([O[0], O[1], np.cross(O[0], O[1])])
        sc = math.sqrt(s[0] * s[1])
        q = np.linalg.pinv(M) @ (np.asarray(cxcy[c], float) - aff.t[c])
        rows = np.flatnonzero(obs_sub.active & (obs_sub.cam == c) & (col[obs_sub.frame] >= 0))
        Xr = np.where((obs_sub.end[rows] == 0)[:, None], XA[col[obs_sub.frame[rows]]], XB[col[obs_sub.frame[rows]]])
        uvw = np.full((len(aff.frames), 2, 2), np.nan)
        uvw[col[obs_sub.frame[rows]], obs_sub.end[rows]] = obs_sub.uv[rows]
        D_m, f_m, slope = _distance_from_magnification(R, XA, XB, uvw[:, 0], uvw[:, 1], L, centre)
        if f_hint is not None and f_hint[c]:
            D, di = float(f_hint[c]) / sc, {"method": "hint"}
        elif np.isfinite(D_m):
            D, di = D_m, {"method": "perspective"}
        else:
            D, di = _distance_search(R, q, sc, Xr, obs_sub.uv[rows], np.asarray(cxcy[c], float), rc)
            di["method"] = "search"
        cams.append(pm.Camera(R, q - D * R[2], sc * D, cxcy[c][0], cxcy[c][1], sizes[c][0], sizes[c][1]))
        info.append({"D_mm": D, "f_px": sc * D, "perspective_slope": slope, **di})
    return cams, info


def _dlt(X, uv):
    """Direct linear transform: the 3 x 4 camera matrix of points X (n, 3) imaged at uv (n, 2), Hartley-normalised."""
    mx, mu = X.mean(axis=0), uv.mean(axis=0)
    sx = math.sqrt(3.0) / max(float(np.sqrt(np.mean(np.sum((X - mx) ** 2, axis=1)))), 1e-12)
    su = math.sqrt(2.0) / max(float(np.sqrt(np.mean(np.sum((uv - mu) ** 2, axis=1)))), 1e-12)
    Tx = np.diag([sx, sx, sx, 1.0])
    Tx[:3, 3] = -sx * mx
    Tu = np.diag([su, su, 1.0])
    Tu[:2, 2] = -su * mu
    Xh = (Tx @ np.column_stack([X, np.ones(len(X))]).T).T
    uh = (Tu @ np.column_stack([uv, np.ones(len(uv))]).T).T
    A = np.zeros((2 * len(X), 12))
    A[0::2, 0:4] = Xh
    A[0::2, 8:12] = -uh[:, 0:1] * Xh
    A[1::2, 4:8] = Xh
    A[1::2, 8:12] = -uh[:, 1:2] * Xh
    Pn = np.linalg.svd(A)[2][-1].reshape(3, 4)
    return np.linalg.inv(Tu) @ Pn @ Tx


def _camera_from_dlt(X, uv, cxcy, size):
    """A pinhole camera (centred principal point, f from the DLT) for points X seen at uv, polished by `resect`."""
    from scipy.linalg import rq
    P = _dlt(X, uv)
    M = P[:, :3]
    if np.linalg.det(M) < 0:
        P = -P
        M = -M
    K, R = rq(M)
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R
    C = -np.linalg.solve(M, P[:, 3])
    f = 0.5 * (K[0, 0] + K[1, 1]) / K[2, 2]
    cam = pm.Camera(R, C, f, cxcy[0], cxcy[1], size[0], size[1])
    cs, _cost = resect([cam], [None], [X], [uv], iters=30)
    return cs[0]


def _frame_subset(obs: Obs, max_frames: int) -> Obs:
    """At most max_frames frames (evenly spread over those with active rows), renumbered."""
    act = np.zeros(obs.n_frames, bool)
    act[obs.frame[obs.active]] = True
    idx = np.flatnonzero(act)
    if len(idx) <= max_frames:
        return obs
    keep = np.zeros(obs.n_frames, bool)
    keep[idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]] = True
    return obs.subset_frames(keep)


def pinhole_fit(cams, obs: Obs, L: float, k_sigma: float = 5.0, max_frames: int = 300, verbose: bool = False,
                log=print) -> Solution:
    """The pinhole bundle adjustment of stage 2 on at most max_frames frames (every camera without its window; camera
    1's pose held; f free; equal weights).  The refraction misfit (up to tens of px at steep incidence) only inflates
    the robust sigma.  The solution's observations are that frame subset."""
    rig0 = Rig([], [-1] * obs.n_cams)
    o = _frame_subset(obs, max_frames)
    o = Obs(o.frame, o.cam, o.end, o.uv, o.n_frames, o.n_cams, o.frame_ids, o.active.copy())
    layout = Layout(obs.n_cams, 0)
    frames = initial_frames(cams, rig0, o, L)
    return refine(cams, rig0, frames, L, o, layout, k_sigma=k_sigma, noise_weights="none", max_rounds=4,
                  max_iter=60, verbose=verbose, log=log)


def pinhole_start(obs: Obs, L: float, sizes, cxcy, rig: Rig, hints: dict = None, verbose: bool = False,
                  log=print):
    """Stages 1-2: the affine factorisation (of a camera subset if needed), pinhole cameras from it, cameras outside
    the subset added by DLT resection, and the pinhole bundle adjustment.  Perspective tells the two mirror images of
    the affine reconstruction apart (near wands look longer only in the right one); when every camera agrees only that
    one is tried, else both, and both are kept when their RMS are within 1.5x.  Returns (list of Solutions, info)."""
    hints = hints or {}
    aff, subset = affine_start(obs, L)
    f_hint = None
    if hints.get("f") is not None:
        # a hinted (true) f becomes the pinhole equivalent n_obj/n_cam times larger behind a window
        f_hint = []
        for c, fv in enumerate(hints["f"]):
            w = rig.cam_win[c]
            k = rig.windows[w].idx[-1] / rig.windows[w].idx[0] if w >= 0 else 1.0
            f_hint.append(None if not fv else float(fv) * k)
        f_hint = [f_hint[c] for c in subset]
    obs_sub = obs if len(subset) == obs.n_cams else _restrict_obs(obs, subset)
    out = []
    infos = []
    built = {}
    for mirror in (False, True):
        built[mirror] = _affine_pinholes(aff, mirror, obs_sub, [cxcy[c] for c in subset],
                                         [sizes[c] for c in subset], L, f_hint)
    votes = {m: sum(1 for i in built[m][1] if i["perspective_slope"] > 0) for m in (False, True)}
    k = len(subset)
    mirrors = [m for m in (False, True) if not (votes[m] == 0 and votes[not m] == k)]
    for mirror in mirrors:
        cams_s, info_s = built[mirror]
        added = []
        if len(subset) < obs.n_cams:
            sol_s = pinhole_fit(cams_s, obs_sub, L, verbose=verbose, log=log)
            XA, XB, _cnt = triangulate(sol_s.cams, Rig([], [-1] * len(subset)), obs_sub)
            cams = [None] * obs.n_cams
            for kk, c in enumerate(subset):
                cams[c] = sol_s.cams[kk]
            for c in range(obs.n_cams):
                if cams[c] is not None:
                    continue
                rows = np.flatnonzero(obs.active & (obs.cam == c))
                Xr = np.where((obs.end[rows] == 0)[:, None], XA[obs.frame[rows]], XB[obs.frame[rows]])
                fine = np.all(np.isfinite(Xr), axis=1)
                if fine.sum() < 6:
                    raise ValueError(f"camera {c + 1} sees fewer than 6 points the other cameras triangulate")
                cams[c] = _camera_from_dlt(Xr[fine], obs.uv[rows[fine]], cxcy[c], sizes[c])
                added.append(c)
        else:
            cams = cams_s
        sol = pinhole_fit(cams, obs, L, verbose=verbose, log=log)
        sol.info["mirror"] = mirror
        sol.info["affine_cameras"] = info_s
        sol.info["added_by_dlt"] = added
        out.append(sol)
        infos.append({"mirror": mirror, "rms_px": sol.rms, "active": int(sol.obs.active.sum()),
                      "distances": [i.get("D_mm") for i in info_s], "added_by_dlt": added})
        if verbose:
            log(f"  pinhole start, mirror {mirror}: rms {sol.rms:.3f} px, active {int(sol.obs.active.sum())}/"
                f"{len(sol.obs)}")
    best = min(s.rms for s in out)
    kept = [s for s in out if s.rms <= 1.5 * best]
    kept.sort(key=lambda s: s.rms)
    return kept, {"affine_rms_px": aff.rms_px, "affine_frames": int(len(aff.frames)),
                  "affine_length_rms_rel": aff.length_rms_rel, "subset": subset,
                  "mirror_votes": {str(m): v for m, v in votes.items()}, "mirrors": infos}


# ------------------------------------------------------------------ stage 3: the windows
def fibonacci_cap(n0: np.ndarray, theta_cap_deg: float, dtheta_deg: float) -> np.ndarray:
    """Unit vectors on the cap of half-angle theta_cap about n0, about dtheta apart (n0 first)."""
    th = math.radians(theta_cap_deg)
    dt = math.radians(dtheta_deg)
    N = max(1, int(round(2.0 * math.pi * (1.0 - math.cos(th)) / dt ** 2)))
    t1, t2 = pm.tangent_basis(n0)
    t1, t2 = t1[0], t2[0]
    out = [np.asarray(n0, float)]
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(N):
        z = 1.0 - (1.0 - math.cos(th)) * (i + 0.5) / N
        r = math.sqrt(max(1.0 - z * z, 0.0))
        ph = i * ga
        out.append(z * n0 + r * (math.cos(ph) * t1 + math.sin(ph) * t2))
    return np.array(out)


def small_angle_camera(cam_p: pm.Camera, win: pm.Window) -> pm.Camera:
    """The camera behind `win` that sees what the pinhole camera cam_p sees, to first order.  Its distance to the
    window is the paraxial one: a camera h1 from the window acts as a pinhole in the object medium n_obj (h1/n_cam +
    sum t_k/n_k) behind its object-side surface, with f' = f n_obj/n_cam (SymPy: tests/plate_wand/p_sympy_presize.py,
    claim E); h1 is at least 2 % of the pinhole's distance.  Its chief ray is exact: the pinhole's principal ray is
    the ray in the water, which is traced back through the plates by Snell's law to the camera side, and the camera
    is put on it, h1 from the window, looking along it (an oblique camera's air axis differs from the water ray by
    the refraction, 13 deg at 44 deg incidence, too much for a short resection to recover)."""
    n0, nobj = win.idx[0], win.idx[-1]
    n = win.n
    Dp = win.far_d - float(n @ cam_p.C)
    h1 = n0 * (Dp / nobj - sum(t / nk for t, nk in zip(win.t, win.idx[1:-1])))
    h1 = max(h1, 0.02 * Dp)
    c = cam_p.copy()
    c.f = cam_p.f * n0 / nobj
    dw = cam_p.R[2]
    cw_ = float(dw @ n)
    lat = dw - cw_ * n
    sw = float(np.linalg.norm(lat))
    sin_a = nobj / n0 * sw
    if cw_ <= 0.05 or sin_a >= 0.999:
        c.C = cam_p.C + (Dp - h1 - win.T) * n             # no usable chief ray: along the normal only
        return c
    e = lat / sw if sw > 0 else pm.tangent_basis(n)[0][0]
    pos = cam_p.C + ((win.far_d - float(n @ cam_p.C)) / cw_) * dw      # where the water ray leaves the window
    for t, nk in zip(reversed(win.t), reversed(win.idx[1:-1])):
        sk = nobj / nk * sw
        ck = math.sqrt(max(1.0 - sk * sk, 1e-12))
        pos = pos - (t / ck) * (ck * n + sk * e)
    ca = math.sqrt(1.0 - sin_a * sin_a)
    da = ca * n + sin_a * e
    c.C = pos - (h1 / ca) * da
    axis = np.cross(dw, da)
    sa = float(np.linalg.norm(axis))
    if sa > 1e-12:
        c.R = cam_p.R @ cm.exp_so3(axis / sa * math.atan2(sa, float(dw @ da))).T
    return c


def hint_to_frame(v, cam_p: pm.Camera, win: pm.Window) -> np.ndarray:
    """A camera's hinted window normal v -- in the camera's own frame (x along the image rows, y down, z along the
    optical axis), either sign -- in the frame of its pinhole camera cam_p.  A pinhole camera looks along the chief ray
    in the object medium (`small_angle_camera`), so the hint's angle theta from the camera's axis becomes theta_w,
    sin(theta_w) = (n_cam / n_obj) sin(theta), at the same azimuth about the axis: the inverse of
    `small_angle_camera` (tests/test_plate_wand.py)."""
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    st = math.hypot(float(v[0]), float(v[1]))
    if st < 1e-12:
        q = np.array([0.0, 0.0, 1.0])
    else:
        sw = min(win.idx[0] / win.idx[-1] * st, 1.0)
        q = np.array([v[0] / st * sw, v[1] / st * sw, math.sqrt(1.0 - sw * sw)])
    return cam_p.R.T @ q


def hinted_normal(hints: dict, cams_w: Sequence[int], cams_pin: Dict[int, pm.Camera], win: pm.Window):
    """A window's normal from hints["normals"] (one entry per camera: None, or the normal of the camera's window in
    its own frame, `hint_to_frame`) of its cameras cams_w, in the frame of their pinhole cameras cams_pin: the mean
    over the hinted ones.  None without a hint."""
    hn = (hints or {}).get("normals")
    if not hn:
        return None
    acc, k = np.zeros(3), 0
    for c in cams_w:
        if c >= len(hn) or hn[c] is None or c not in cams_pin:
            continue
        acc += hint_to_frame(hn[c], cams_pin[c], win)
        k += 1
    nrm = float(np.linalg.norm(acc))
    return acc / nrm if k and nrm > 1e-12 else None


def _cam_hint(hints: dict, c: int):
    """Camera c's hinted window normal (hints["normals"][c]) or None."""
    hn = (hints or {}).get("normals") or []
    return hn[c] if c < len(hn) else None


HINT_CAP_DEG = 20.0                    # the hint-searched start: a second window search about a hinted normal, its cap
HINT_STEP_DEG = 4.0                    # ... and its normals' spacing (a candidate within ~2.3 deg of any normal in it)


def _merge_hinted(base, extra, tol_deg: float = 1.0, tol_rel: float = 0.05):
    """The default window search's candidates [(score, window, {camera: Camera})] and those of the search about a hint,
    by score (the hint-searched start).  Every default candidate is kept.  A hint candidate is dropped when one already
    there has its normal within tol_deg and its plane, at the candidate's cameras, within tol_rel of their distance to
    the window: it would only repeat a start.  Returns (merged, whether the best came from the hint search)."""
    out = [(s, wn, cs, False) for s, wn, cs in base]
    for s, wn, cs in extra:
        dup = False
        if cs:
            Cm = np.mean([cam.C for cam in cs.values()], axis=0)
            h1 = max(abs(wn.distance(Cm)), 1e-9)
            dup = any(math.degrees(math.acos(min(1.0, float(wn.n @ w2.n)))) <= tol_deg
                      and abs(wn.distance(Cm) - w2.distance(Cm)) <= tol_rel * h1 for _s, w2, _c, _f in out)
        if not dup:
            out.append((s, wn, cs, True))
    out.sort(key=lambda x: x[0])
    return [(s, wn, cs) for s, wn, cs, _f in out], bool(out and out[0][3])


def window_search(sol_p: Solution, rig: Rig, w: int, theta_cap: float = 50.0, dtheta: float = 9.0,
                  phis=(0.2, 0.5, 0.8), n_quick_frames: int = 60, quick_iters: int = 4,
                  n_full_frames: int = 200, full_iters: int = 20, n_full: int = 8, n_keep: int = 3,
                  points=None, base_rig: Rig = None, n_guess=None, tri_cams=None, keep_h1: bool = True,
                  retriangulate: bool = True, hold_normal: bool = False, verbose: bool = False, log=print):
    """Stage 3 for window w: candidate normals on a cap about a guess (default: the window's cameras' mean viewing
    direction, the dominant eigenvector of sum z z^T, sign-free, turned towards the wand cloud), candidate offsets
    across the gap between the cameras and the cloud (the final fit recovers a normal 30 deg off and any offset across
    that gap, so the spacing is generous: tests/plate_wand/w4_basin.py); for each, the window's cameras (pinhole cameras in sol_p) moved
    behind it (`small_angle_camera`) and resected against fixed points (sol_p's frames, or `points` = (XA, XB)) with a
    few iterations on a few frames.  The best n_full get a full resection, a re-triangulation of the frames with the
    window's cameras refractive (the other cameras as in base_rig, default pinholes; only the cameras in tri_cams, default
    all) and a second resection (not with retriangulate False: points known to be accurate stay fixed, and a camera
    must not pull them towards itself).  The resections keep each camera's distance to the candidate window
    (`resect`'s keep_h1).  With hold_normal the window's normal is held at n_guess (a hint-held start): only the
    offsets are tried, and the resections fit only the window's offset.  Returns the best n_keep as (score, window,
    {camera: Camera}) and info."""
    obs = sol_p.obs
    L = sol_p.L
    cams_w = rig.cams_of(w)
    win0 = rig.windows[w]
    XA, XB = sol_p.frames.ends(L) if points is None else points
    act_frames = np.unique(obs.frame[obs.active])
    pts = np.vstack([XA[act_frames], XB[act_frames]])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if n_guess is None:
        S = sum(np.outer(sol_p.cams[c].R[2], sol_p.cams[c].R[2]) for c in cams_w)
        n_guess = np.linalg.eigh(S)[1][:, -1]
        cbar = np.mean([sol_p.cams[c].C for c in cams_w], axis=0)
        if float(n_guess @ (pts.mean(axis=0) - cbar)) < 0:
            n_guess = -n_guess
    n_guess = np.asarray(n_guess, float) / np.linalg.norm(n_guess)
    normals = n_guess[None] if hold_normal else fibonacci_cap(n_guess, theta_cap, dtheta)
    quick_rows = _subset_rows(obs, n_quick_frames)
    full_rows = _subset_rows(obs, n_full_frames)

    def cam_data(rows_all, XA_, XB_):
        Xs, Us = [], []
        for c in cams_w:
            rr = rows_all[obs.cam[rows_all] == c]
            Xr = np.where((obs.end[rr] == 0)[:, None], XA_[obs.frame[rr]], XB_[obs.frame[rr]])
            fine = np.all(np.isfinite(Xr), axis=1)
            Xs.append(Xr[fine])
            Us.append(obs.uv[rr[fine]])
        return Xs, Us

    Xq, Uq = cam_data(quick_rows, XA, XB)
    cands = []
    n_tried = 0
    for n in normals:
        lo = max(float(n @ sol_p.cams[c].C) for c in cams_w)
        hi = float(np.min(pts @ n)) - win0.T
        if not hi > lo:
            continue
        for phi in phis:
            d = lo + phi * (hi - lo)
            win = pm.Window(n, d, win0.t, win0.idx, win0.name)
            cs0 = [small_angle_camera(sol_p.cams[c], win) for c in cams_w]
            cs, cost = resect(cs0, [win] * len(cs0), Xq, Uq, iters=quick_iters, keep_h1=keep_h1)
            n_tried += 1
            cands.append((float(np.sum(cost)), win, cs))
    if not cands:
        raise ValueError(f"window {w + 1}: no candidate between its cameras and the wand cloud")
    cands.sort(key=lambda c: c[0])
    Xf, Uf = cam_data(full_rows, XA, XB)
    base = base_rig if base_rig is not None else Rig(list(rig.windows), [-1] * obs.n_cams)
    full = []
    for score_q, win, cs in cands[:n_full]:
        cs, cost = resect(cs, [win] * len(cs), Xf, Uf, iters=full_iters, keep_h1=keep_h1)
        if not retriangulate:
            cs, win, cost = resect_window(cs, win, Xf, Uf, iters=full_iters, hold_normal=hold_normal)
            full.append((float(np.sum(cost)), win, {c: cs[k] for k, c in enumerate(cams_w)}, score_q))
            continue
        cams_all = [c.copy() for c in sol_p.cams]
        for k, c in enumerate(cams_w):
            cams_all[c] = cs[k]
        wins_tmp = [x.copy() for x in base.windows]
        cw_tmp = list(base.cam_win)
        while len(wins_tmp) <= w:
            wins_tmp.append(win0.copy())
        wins_tmp[w] = win
        for c in cams_w:
            cw_tmp[c] = w
        use = np.isin(np.arange(len(obs)), full_rows) & obs.active
        if tri_cams is not None:
            use &= np.isin(obs.cam, list(tri_cams) + list(cams_w))
        o_tmp = Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, use)
        XA2, XB2, cnt = triangulate(cams_all, Rig(wins_tmp, cw_tmp), o_tmp)
        fine = np.all(cnt >= 2, axis=1) & np.all(np.isfinite(XA2), axis=1) & np.all(np.isfinite(XB2), axis=1)
        XA3, XB3 = XA.copy(), XB.copy()
        XA3[fine], XB3[fine] = XA2[fine], XB2[fine]
        Xf2, Uf2 = cam_data(full_rows, XA3, XB3)
        cs, win, cost = resect_window(cs, win, Xf2, Uf2, iters=full_iters, hold_normal=hold_normal)
        full.append((float(np.sum(cost)), win, {c: cs[k] for k, c in enumerate(cams_w)}, score_q))
    full.sort(key=lambda c: c[0])
    if verbose:
        for s, win, _cs, sq in full[:n_keep]:
            ang = math.degrees(math.acos(min(1.0, abs(float(win.n @ n_guess)))))
            log(f"    window {w + 1} candidate: {ang:.1f} deg from the guess, d {win.d:.1f}: quick {sq:.4g}, full {s:.4g}")
    info = {"normals": len(normals), "candidates": n_tried, "guess": n_guess.tolist()}
    return [(s, win, cs) for s, win, cs, _sq in full[:n_keep]], info


def _combos(per_win, max_combos: int):
    """Every window's best candidate; then each window's next ones with the others at their best, cheapest score
    increase first."""
    out = [[0] * len(per_win)]
    extra = []
    for w, lst in enumerate(per_win):
        for k in range(1, len(lst)):
            cmb = [0] * len(per_win)
            cmb[w] = k
            extra.append((lst[k][0] - lst[0][0], cmb))
    extra.sort(key=lambda e: e[0])
    return out + [c for _d, c in extra][:max(0, max_combos - 1)]


def _assemble(base_cams, rig: Rig, per_win, cmb):
    cams = [c.copy() for c in base_cams]
    wins = []
    for w, k in enumerate(cmb):
        _s, win, cs = per_win[w][k]
        wins.append(win.copy())
        for c, cam in cs.items():
            cams[c] = cam.copy()
    return cams, Rig(wins, rig.cam_win)


def bootstrap(obs: Obs, rig: Rig, L: float, sizes, hints: dict = None, cxcy=None, theta_cap: float = 50.0,
              dtheta: float = 9.0, n_coarse_frames: int = 150, max_combos: int = 3, passes: int = None,
              search: str = "length", hold: bool = False, widen: bool = False, verbose: bool = False, log=print):
    """Stages 1-3 on the whole rig at once, from the pinhole start: each window's best three candidates
    (`window_search`) are combined (every window's best; then the next ones, cheapest first, up to max_combos), and
    each combination gets the joint coarse fit (`coarse_fit`) on n_coarse_frames frames, which ranks them.  With
    several windows a second pass searches every window again against points re-triangulated with the first pass's
    best cameras (refractive), which are less biased than the pinhole start's.  Returns (list of (cost, rms, cameras,
    rig), info).
    search: how a window with one or two cameras is searched -- "length" scores each candidate by the wand-length
    constrained back-projection residual (`place_window_vp`), "fixed" by resections against fixed points
    (`window_search`); neither is right on every rig (W5), so `calibrate` pools the starts of both on rigs with such
    windows.  Windows with three or more cameras always use "fixed".
    hold: a hint-held start -- every window with a hinted camera (hints["normals"], `hinted_normal`) has its normal
    held at the hint through the search and the coarse fit.  widen: a hint-searched start -- every such window is
    searched a second time about its hint (HINT_CAP_DEG, HINT_STEP_DEG), the candidates merged with the default
    search's (`_merge_hinted`)."""
    hints = hints or {}
    t0 = time.time()
    cxcy = cxcy or [((w - 1) / 2.0, (h - 1) / 2.0) for (w, h) in sizes]
    pins, pinfo = pinhole_start(obs, L, sizes, cxcy, rig, hints=hints, verbose=verbose, log=log)
    t_pin = time.time() - t0
    rows = _subset_rows(obs, n_coarse_frames)
    passes = passes if passes is not None else (2 if len(rig.windows) > 1 else 1)
    starts = []
    winfo = []
    t_search = t_coarse = 0.0
    for sol_p in pins:
        t1 = time.time()
        per_win = None
        prev = None
        points = None
        base = None
        # a hint-held or hint-searched start: the hinted normals (hints["normals"]) in this pinhole start's frame
        n_hint = [hinted_normal(hints, rig.cams_of(w), {c: sol_p.cams[c] for c in rig.cams_of(w)}, rig.windows[w])
                  if (hold or widen) else None for w in range(len(rig.windows))]
        held = [w for w in range(len(rig.windows)) if hold and n_hint[w] is not None]
        for ps in range(passes):
            per_win = []
            for w in range(len(rig.windows)):
                if not rig.cams_of(w):
                    per_win.append([(0.0, rig.windows[w].copy(), {})])
                    continue
                hw = w in held
                wide = widen and n_hint[w] is not None
                hint_inf = {"held": True} if hw else {}
                if search == "length" and len(rig.cams_of(w)) <= 2:
                    # one or two cameras: their own points cannot judge a window, the wand length must
                    cams_cur, rig_c = (sol_p.cams, Rig(list(rig.windows), [-1] * obs.n_cams)) if base is None \
                        else (_assemble(sol_p.cams, rig, prev, [0] * len(prev))[0], base)
                    pins_w = {c: sol_p.cams[c] for c in rig.cams_of(w)}
                    best = place_window_vp(rig.cams_of(w), w, pins_w, cams_cur, rig_c, rig, sol_p.obs, L,
                                           theta_cap=theta_cap, dtheta=dtheta, n_guess=n_hint[w] if hw else None,
                                           hold_normal=hw)
                    if wide:
                        # the search again about the hinted normal: a wrong hint only adds candidates that lose
                        extra = place_window_vp(rig.cams_of(w), w, pins_w, cams_cur, rig_c, rig, sol_p.obs, L,
                                                theta_cap=HINT_CAP_DEG, dtheta=HINT_STEP_DEG, n_guess=n_hint[w])
                        best, hint_inf["best_from_hint"] = _merge_hinted(best, extra)
                    per_win.append(best)
                    winfo.append({"mirror": sol_p.info.get("mirror"), "window": w, "pass": ps, "method": "wand length",
                                  **hint_inf})
                    continue
                best, inf = window_search(sol_p, rig, w, theta_cap=theta_cap, dtheta=dtheta, points=points,
                                          base_rig=base, n_guess=n_hint[w] if hw else None, hold_normal=hw,
                                          verbose=verbose, log=log)
                if wide:
                    extra, _inf = window_search(sol_p, rig, w, theta_cap=HINT_CAP_DEG, dtheta=HINT_STEP_DEG,
                                                points=points, base_rig=base, n_guess=n_hint[w], verbose=verbose,
                                                log=log)
                    best, hint_inf["best_from_hint"] = _merge_hinted(best, extra)
                per_win.append(best)
                winfo.append({"mirror": sol_p.info.get("mirror"), "window": w, "pass": ps, **inf, **hint_inf})
            prev = per_win
            if ps + 1 < passes:
                cams1, rig1 = _assemble(sol_p.cams, rig, per_win, [0] * len(per_win))
                XA, XB, cnt = triangulate(cams1, rig1, sol_p.obs)
                okf = np.all(cnt >= 2, axis=1)
                XA[~okf], XB[~okf] = np.nan, np.nan
                points, base = (XA, XB), rig1
        t_search += time.time() - t1
        t2 = time.time()
        for cmb in _combos(per_win, max_combos):
            cams, rg = _assemble(sol_p.cams, rig, per_win, cmb)
            layout = Layout(obs.n_cams, len(rg.windows), held_normals=held)
            cs3, rg3, inf3 = coarse_fit(cams, rg, obs, L, layout, rows=rows, verbose=verbose, log=log)
            r = backprojection_residual(cs3, rg3, obs, L, rows)
            starts.append((_robust_cost(r), inf3["rms_px"], cs3, rg3))
            if verbose:
                log(f"  start (mirror {sol_p.info.get('mirror')}, combination {cmb}): coarse rms {inf3['rms_px']:.4f} px")
        t_coarse += time.time() - t2
    starts.sort(key=lambda s: s[0])
    info = {**pinfo, "method": "whole rig", "search": search, "passes": passes, "windows": winfo,
            "starts": len(starts),
            "seconds": {"pinhole": t_pin, "window_search": t_search, "coarse_fit": t_coarse,
                        "total": time.time() - t0}}
    return starts, info


# ------------------------------------------------------------------ stage 1-3 by window groups
def _rigid(P, Q):
    """Rigid (R, t) with R P + t ~ Q (Kabsch)."""
    mp, mq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - mp).T @ (Q - mq)
    U, _s, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def _move(cams, wins, R, t):
    """Cameras and windows moved by X -> R X + t."""
    oc = []
    for c in cams:
        k = c.copy()
        k.R = c.R @ R.T
        k.C = R @ c.C + t
        oc.append(k)
    ow = []
    for w in wins:
        n = R @ w.n
        ow.append(pm.Window(n, w.d + float(n @ t), w.t, w.idx, w.name))
    return oc, ow


def solve_obs(obs: Obs, L: float, cams, rig: Rig, **kw) -> Solution:
    """`solve` on given observations (e.g. a camera subset): frames triangulated, then `refine`."""
    fit_k1 = kw.pop("fit_k1", False)
    layout = Layout(obs.n_cams, len(rig.windows), fit_k1=fit_k1, fit_L=kw.pop("fit_L", False),
                    fixed_windows=kw.pop("fixed_windows", ()), anchors=anchors_for(cams, rig),
                    held_normals=kw.pop("held_normals", ()))
    frames = initial_frames(cams, rig, obs, L)
    return refine(cams, rig, frames, L, obs, layout, **kw)


def place_single_window(c: int, w: int, cams, rig: Rig, obs: Obs, L: float, XA, XB, cxcy, size,
                        theta_cap: float = 50.0, dtheta: float = 9.0, tri_cams=None, hint=None, widen: bool = False,
                        verbose: bool = False, log=print):
    """A camera alone behind its window, placed against points the other (calibrated) cameras triangulate: its
    pinhole start by DLT, then `window_search` with those points fixed (they are accurate, so the resection sees
    the window's refraction).  With a hint (the window's normal in the camera's own frame, `hint_to_frame`) the
    window's normal is held at it and only its offsets are searched (a hint-held start); with widen, the search runs
    instead a second time about the hint and the candidates are merged (`_merge_hinted`, a hint-searched start).
    Returns window_search's candidates and info."""
    rows = np.flatnonzero(obs.active & (obs.cam == c))
    Xr = np.where((obs.end[rows] == 0)[:, None], XA[obs.frame[rows]], XB[obs.frame[rows]])
    fine = np.all(np.isfinite(Xr), axis=1)
    if fine.sum() < 12:
        raise ValueError(f"camera {c + 1} sees fewer than 12 points the other cameras triangulate")
    cam_p = _camera_from_dlt(Xr[fine], obs.uv[rows[fine]], cxcy, size)
    cams_p = [x.copy() for x in cams]
    cams_p[c] = cam_p
    fr = Frames(np.where(np.isfinite(XA), 0.5 * (XA + XB), 0.0), np.tile([0.0, 0.0, 1.0], (len(XA), 1)))
    pseudo = Solution(cams_p, rig, fr, L, obs, Layout(obs.n_cams))
    base = rig.copy()
    base.cam_win = [k if q != c else -1 for q, k in enumerate(rig.cam_win)]
    n_hint = None if hint is None else hint_to_frame(hint, cam_p, rig.windows[w])
    hold = n_hint is not None and not widen
    cands, info = window_search(pseudo, rig, w, theta_cap=theta_cap, dtheta=dtheta, points=(XA, XB), base_rig=base,
                                n_guess=n_hint if hold else cam_p.R[2], tri_cams=tri_cams, retriangulate=False,
                                hold_normal=hold, verbose=verbose, log=log)
    if widen and n_hint is not None:
        extra, _inf = window_search(pseudo, rig, w, theta_cap=HINT_CAP_DEG, dtheta=HINT_STEP_DEG, points=(XA, XB),
                                    base_rig=base, n_guess=n_hint, tri_cams=tri_cams, retriangulate=False,
                                    verbose=verbose, log=log)
        cands, from_hint = _merge_hinted(cands, extra)
        info = {**info, "best_from_hint": from_hint}
    return cands, info


def bootstrap_groups(obs: Obs, rig: Rig, L: float, sizes, hints: dict = None, cxcy=None, theta_cap: float = 50.0,
                     dtheta: float = 9.0, max_combos: int = 3, search: str = "length", cache: dict = None,
                     hold: bool = False, widen: bool = False, verbose: bool = False, log=print):
    """Stages 1-3 for a rig whose windows have two or more cameras each (or some do): each such window's cameras are
    calibrated on their own as a one-window rig (`bootstrap` and stage 4 on their observations), the groups are put
    into one frame by the rigid motion that best maps their common wand positions onto each other (the wand length
    fixes every group's scale), and a camera alone behind its window (or without one) is then placed against the
    others (`place_window_vp` or, with search "fixed", `place_single_window`).  Returns (list of (cost, rms, cameras,
    rig), info).  `cache` (a dict) keeps the groups of three or more cameras between calls with different searches.
    search: how a window with one or two cameras is searched -- "length" scores each candidate by the wand-length
    constrained back-projection residual (`place_window_vp`), "fixed" by resections against fixed points
    (`window_search`); neither is right on every rig (W5), so `calibrate` pools the starts of both on rigs with such
    windows.  Windows with three or more cameras always use "fixed".
    hold: a hint-held start -- every window with a hinted camera (hints["normals"]) has its normal held at the hint in
    its group's own calibration (search and stage 4) or in its single camera's placement.  widen: a hint-searched start
    -- every such window is also searched about its hint, in its group's own calibration or in its single camera's
    placement (`bootstrap`'s widen)."""
    t0 = time.time()
    hints = hints or {}
    cxcy = cxcy or [((w_ - 1) / 2.0, (h - 1) / 2.0) for (w_, h) in sizes]
    nW = len(rig.windows)
    groups = [w for w in range(nW) if len(rig.cams_of(w)) >= 2]
    sols = {}
    ginfo = {}
    for w in groups:
        cs = rig.cams_of(w)
        if cache is not None and len(cs) >= 3 and w in cache:
            sols[w], ginfo[w] = cache[w]
            continue
        o_w = _restrict_obs(obs, cs)
        rig_w = Rig([rig.windows[w].copy()], [0] * len(cs))
        sub_h = {}
        if hints.get("f") is not None:
            sub_h["f"] = [hints["f"][c] for c in cs]
        hinted_w = any(_cam_hint(hints, c) is not None for c in cs)
        held_w, wide_w = hold and hinted_w, widen and hinted_w
        if held_w or wide_w:
            sub_h["normals"] = [_cam_hint(hints, c) for c in cs]
        st_w, inf_w = bootstrap(o_w, rig_w, L, [sizes[c] for c in cs], hints=sub_h, cxcy=[cxcy[c] for c in cs],
                                theta_cap=theta_cap, dtheta=dtheta, search=search, hold=held_w, widen=wide_w,
                                verbose=verbose, log=log)
        best = None
        for k, (_cost, rms3, c0, r0) in enumerate(st_w[:2]):
            o = Obs(o_w.frame, o_w.cam, o_w.end, o_w.uv, o_w.n_frames, o_w.n_cams, o_w.frame_ids, o_w.active.copy())
            try:
                sol = solve_obs(o, L, c0, r0, noise_weights="none", held_normals=(0,) if held_w else ())
            except (np.linalg.LinAlgError, ValueError):
                continue
            key = (sol.rms, -int(o.active.sum()))
            if best is None or key < best[0]:
                best = (key, sol)
        if best is None:
            raise ValueError(f"window {w + 1}: its cameras could not be calibrated on their own")
        sols[w] = best[1]
        ginfo[w] = {"cameras": cs, "rms_px": best[1].rms, "bootstrap_seconds": inf_w["seconds"]}
        if held_w or wide_w:
            ginfo[w]["held" if held_w else "widened"] = True
        if cache is not None and len(cs) >= 3:
            cache[w] = (sols[w], ginfo[w])
        if verbose:
            log(f"  window {w + 1} group {[c + 1 for c in cs]}: rms {best[1].rms:.4f} px")
    # one frame: the largest group's; the others joined in order of their common frames with the joined ones
    ref = max(groups, key=lambda w: (len(rig.cams_of(w)), int(sols[w].obs.active.sum())))
    cams = [None] * obs.n_cams
    wins = [rig.windows[w].copy() for w in range(nW)]
    ends = {}

    def frame_ends(sol, R=np.eye(3), t=np.zeros(3)):
        XA, XB = sol.frames.ends(L)
        act = np.zeros(sol.obs.n_frames, bool)
        act[sol.obs.frame[sol.obs.active]] = True
        XA = np.where(act[:, None], XA @ R.T + t, np.nan)
        XB = np.where(act[:, None], XB @ R.T + t, np.nan)
        return XA, XB

    def join(w, R, t):
        oc, ow = _move(sols[w].cams, sols[w].rig.windows, R, t)
        for k, c in enumerate(rig.cams_of(w)):
            cams[c] = oc[k]
        wins[w] = ow[0]
        ends[w] = frame_ends(sols[w], R, t)

    join(ref, np.eye(3), np.zeros(3))
    todo = [w for w in groups if w != ref]
    while todo:
        best = None
        for w in todo:
            XA_w, XB_w = frame_ends(sols[w])
            for v in ends:
                XA_v, XB_v = ends[v]
                common = np.all(np.isfinite(XA_w), axis=1) & np.all(np.isfinite(XA_v), axis=1)
                if best is None or common.sum() > best[0]:
                    best = (int(common.sum()), w, v, common)
        n_common, w, v, common = best
        if n_common < 3:
            raise ValueError(f"window {w + 1}'s cameras share fewer than 3 wand frames with the others")
        XA_w, XB_w = frame_ends(sols[w])
        XA_v, XB_v = ends[v]
        R, t = _rigid(np.vstack([XA_w[common], XB_w[common]]), np.vstack([XA_v[common], XB_v[common]]))
        join(w, R, t)
        ginfo[w]["joined_on_frames"] = n_common
        todo.remove(w)
    # cameras alone behind a window (or without one): placed against the joined cameras' points
    rest = [c for c in range(obs.n_cams) if cams[c] is None]
    placed = {}
    joined = [c for c in range(obs.n_cams) if cams[c] is not None]
    if rest:
        o_j = _restrict_obs(obs, joined)
        rig_j = Rig(wins, [rig.cam_win[c] for c in joined])
        XAj, XBj, cntj = triangulate([cams[c] for c in joined], rig_j, o_j)
        okj = np.all(cntj >= 2, axis=1)
        XAj[~okj], XBj[~okj] = np.nan, np.nan
        for c in rest:
            w = rig.cam_win[c]
            tmp_cams = [x if x is not None else pm.Camera(np.eye(3), np.zeros(3), 1000.0, cxcy[k][0], cxcy[k][1],
                                                          sizes[k][0], sizes[k][1]) for k, x in enumerate(cams)]
            if w < 0:
                rows = np.flatnonzero(obs.active & (obs.cam == c))
                Xr = np.where((obs.end[rows] == 0)[:, None], XAj[obs.frame[rows]], XBj[obs.frame[rows]])
                fine = np.all(np.isfinite(Xr), axis=1)
                cams[c] = _camera_from_dlt(Xr[fine], obs.uv[rows[fine]], cxcy[c], sizes[c])
                continue
            hint_c = _cam_hint(hints, c) if (hold or widen) else None
            if search == "fixed":
                cands, inf = place_single_window(c, w, tmp_cams, Rig(wins, rig.cam_win), obs, L, XAj, XBj, cxcy[c],
                                                 sizes[c], theta_cap=theta_cap, dtheta=dtheta, tri_cams=joined,
                                                 hint=hint_c, widen=widen, verbose=verbose, log=log)
                placed[c] = cands
                ginfo[f"single_cam{c + 1}"] = {"window": w, "scores": [x[0] for x in cands],
                                               **({"held": True} if hold and hint_c is not None else {}),
                                               **{k: inf[k] for k in ("best_from_hint",) if k in inf}}
                continue
            rows_c = np.flatnonzero(obs.active & (obs.cam == c))
            Xr = np.where((obs.end[rows_c] == 0)[:, None], XAj[obs.frame[rows_c]], XBj[obs.frame[rows_c]])
            fine = np.all(np.isfinite(Xr), axis=1)
            if fine.sum() < 12:
                raise ValueError(f"camera {c + 1} sees fewer than 12 points the other cameras triangulate")
            cam_p = _camera_from_dlt(Xr[fine], obs.uv[rows_c[fine]], cxcy[c], sizes[c])
            # the others as joined; cameras not yet placed are left out of the triangulation
            cw_cur = [rig.cam_win[q] if cams[q] is not None else -1 for q in range(obs.n_cams)]
            o_c = _restrict_obs(obs, sorted(set(joined) | {c}))
            sub = sorted(set(joined) | {c})
            cams_sub = [tmp_cams[q] for q in sub]
            cams_sub[sub.index(c)] = cam_p
            rig_sub = Rig(wins, [cw_cur[q] for q in sub])
            rig_full_sub = Rig(wins, [rig.cam_win[q] for q in sub])
            n_h = None if hint_c is None else hint_to_frame(hint_c, cam_p, rig.windows[w])
            hc = hold and n_h is not None
            cands_s = place_window_vp([sub.index(c)], w, {sub.index(c): cam_p}, cams_sub, rig_sub, rig_full_sub, o_c,
                                      L, theta_cap=theta_cap, dtheta=dtheta, n_guess=n_h if hc else None,
                                      hold_normal=hc)
            hint_inf = {"held": True} if hc else {}
            if widen and n_h is not None:
                # the search again about the hinted normal: a wrong hint only adds candidates that lose
                extra = place_window_vp([sub.index(c)], w, {sub.index(c): cam_p}, cams_sub, rig_sub, rig_full_sub,
                                        o_c, L, theta_cap=HINT_CAP_DEG, dtheta=HINT_STEP_DEG, n_guess=n_h)
                cands_s, hint_inf["best_from_hint"] = _merge_hinted(cands_s, extra)
            cands = [(sc, wn, {c: cd[sub.index(c)]}) for sc, wn, cd in cands_s]
            placed[c] = cands
            ginfo[f"single_cam{c + 1}"] = {"window": w, "scores": [x[0] for x in cands], **hint_inf}
    # starts: every single window's best candidate, then the next ones (cheapest first)
    singles = sorted(placed)
    per = [placed[c] for c in singles]
    starts = []
    rows = _subset_rows(obs, 150)
    for cmb in (_combos(per, max_combos) if per else [[]]):
        cs = [x.copy() if x is not None else None for x in cams]
        ws = [x.copy() for x in wins]
        for k, c in enumerate(singles):
            _s, win, cd = per[k][cmb[k]]
            ws[rig.cam_win[c]] = win.copy()
            cs[c] = cd[c].copy()
        rg = Rig(ws, rig.cam_win)
        r = backprojection_residual(cs, rg, obs, L, rows)
        starts.append((_robust_cost(r), float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))), cs, rg))
    starts.sort(key=lambda s: s[0])
    info = {"method": "window groups", "search": search, "groups": {str(k): v for k, v in ginfo.items()},
            "reference_window": ref,
            "starts": len(starts), "seconds": {"total": time.time() - t0}}
    return starts, info


def place_window_vp(cams_w: Sequence[int], w: int, cams_pin: Dict[int, pm.Camera], cams, rig_cur: Rig, rig: Rig,
                    obs: Obs, L: float, theta_cap: float = 50.0, dtheta: float = 9.0, phis=(0.2, 0.5, 0.8),
                    n_guess=None, n_quick_frames: int = 60, quick_nfev: int = 6, n_full: int = 6,
                    n_full_frames: int = 150, full_nfev: int = 40, n_keep: int = 3, hold_normal: bool = False):
    """The camera(s) behind window w (one or two), placed with the other cameras held as they are: each candidate
    window (normals on a cap about a guess -- default the cameras' mean viewing direction --, offsets across the
    camera-cloud gap) gets its cameras from their pinhole starts (`small_angle_camera`) and a short least-squares fit
    of them (rotation, slide parallel to the window, f) on the back-projection residual, which re-triangulates the
    wand with every camera and sets its ends L apart.  So the wand's length -- the only metric in the data -- judges
    the candidate; a fit to fixed points would not, because with one or two cameras the points carry the placed
    cameras' own earlier state (W5 seeds 1045, 1048).  The best n_full are refined with their window free.  With
    hold_normal the window's normal is held at n_guess (a hint-held start): only the offsets are tried, and the
    refinement frees only the window's offset.  Returns [(cost, window, {camera: Camera})] best first."""
    from scipy.optimize import least_squares
    cams_w = list(cams_w)
    k = len(cams_w)
    win0 = rig.windows[w]
    quick = _subset_rows(obs, n_quick_frames)
    full = _subset_rows(obs, n_full_frames)
    if n_guess is None:
        S = sum(np.outer(cams_pin[c].R[2], cams_pin[c].R[2]) for c in cams_w)
        n_guess = np.linalg.eigh(S)[1][:, -1]
        if float(n_guess @ cams_pin[cams_w[0]].R[2]) < 0:
            n_guess = -n_guess
    n_guess = np.asarray(n_guess, float) / np.linalg.norm(n_guess)
    XA, XB, cnt = triangulate(cams, rig_cur, obs, quick)
    pts = np.vstack([XA[np.all(cnt >= 2, axis=1)], XB[np.all(cnt >= 2, axis=1)]])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    cw_ = list(rig_cur.cam_win)
    for c in cams_w:
        cw_[c] = w

    def fit(cams0, win, rows, nfev, free_window):
        t1, t2 = pm.tangent_basis(win.n)
        T = np.column_stack([t1[0], t2[0]])
        A = np.mean([cm_.C + win.distance(cm_.C) * win.n for cm_ in cams0], axis=0)
        per = 7 if free_window else 6
        n_w = 0 if not free_window else (1 if free_window == "offset" else 3)    # the window's free parameters

        def build(x):
            out = []
            for j, cam0 in enumerate(cams0):
                v = x[per * j:per * j + per]
                kk = cam0.copy()
                kk.R = cam0.R @ cm.exp_so3(v[0:3])
                kk.C = cam0.C + T @ v[3:5] + (v[6] * win.n if free_window else 0.0)
                kk.f = cam0.f * math.exp(float(np.clip(v[5], -2.0, 2.0)))
                out.append(kk)
            wn = win
            if n_w == 3:
                a, b, dl = x[per * k:per * k + 3]
                nn = pm.retract_normal(win.n, (a, b))
                wn = pm.Window(nn, float(nn @ A) + win.d - float(win.n @ A) + dl, win.t, win.idx, win.name)
            elif n_w == 1:
                wn = pm.Window(win.n, win.d + float(x[per * k]), win.t, win.idx, win.name)
            return out, wn

        def fun(x):
            ks, wn = build(x)
            cs = list(cams)
            for j, c in enumerate(cams_w):
                cs[c] = ks[j]
            ws = list(rig_cur.windows)
            while len(ws) <= w:
                ws.append(win0)
            ws[w] = wn
            return backprojection_residual(cs, Rig(ws, cw_), obs, L, rows).ravel()

        n_x = per * k + n_w
        res = least_squares(fun, np.zeros(n_x), method="trf", x_scale="jac", loss="soft_l1", f_scale=2.0,
                            max_nfev=nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        ks, wn = build(res.x)
        return _robust_cost(res.fun.reshape(-1, 2)), wn, {c: ks[j] for j, c in enumerate(cams_w)}

    cands = []
    for n in ([n_guess] if hold_normal else fibonacci_cap(n_guess, theta_cap, dtheta)):
        lo = max(float(n @ cams_pin[c].C) for c in cams_w)
        hi = float(np.min(pts @ n)) - win0.T if len(pts) else lo
        if not hi > lo:
            continue
        for phi in phis:
            win = pm.Window(n, lo + phi * (hi - lo), win0.t, win0.idx, win0.name)
            cams0 = [small_angle_camera(cams_pin[c], win) for c in cams_w]
            if min(win.distance(cm_.C) for cm_ in cams0) <= 0:
                continue
            cands.append((fit(cams0, win, quick, quick_nfev, False), cams0))
    if not cands:
        raise ValueError(f"window {w + 1}: no candidate between its cameras and the wand cloud")
    cands.sort(key=lambda x: x[0][0])
    out = []
    for (_cost, win, cd), _c0 in cands[:n_full]:
        out.append(fit([cd[c] for c in cams_w], win, full, full_nfev, "offset" if hold_normal else True))
    out.sort(key=lambda x: x[0])
    return out[:n_keep]


def bootstrap_singles(obs: Obs, rig: Rig, L: float, sizes, hints: dict = None, cxcy=None, theta_cap: float = 50.0,
                      dtheta: float = 9.0, passes: int = 3, n_coarse_frames: int = 150, max_combos: int = 4,
                      search: str = "length", hold: bool = False, widen: bool = False, verbose: bool = False,
                      log=print):
    """Stages 1-3 for a rig whose windows have one camera each (e.g. two cameras through two walls): from the pinhole
    start, each camera's window is placed in turn with the other cameras as they are (pinholes at first, then
    refractive), scored by the wand-length-constrained back-projection residual (`place_window_vp`).  Later passes
    search a narrower cap about the window found before; the last pass's best candidates per window give several
    starts, each with the joint coarse fit.  Returns (list of (cost, rms, cameras, rig), info).  With search "fixed"
    each placement is instead a resection against the points all cameras triangulate in their current state
    (`window_search` without re-triangulation).
    search: how a window with one or two cameras is searched -- "length" scores each candidate by the wand-length
    constrained back-projection residual (`place_window_vp`), "fixed" by resections against fixed points
    (`window_search`); neither is right on every rig (W5), so `calibrate` pools the starts of both on rigs with such
    windows.  Windows with three or more cameras always use "fixed".
    hold: a hint-held start -- a window whose camera is hinted (hints["normals"]) has its normal held at the hint in
    every pass and in the coarse fit.  widen: a hint-searched start -- such a window is searched a second time about
    its hint in every pass, the candidates merged (`_merge_hinted`)."""
    hints = hints or {}
    t0 = time.time()
    cxcy = cxcy or [((w_ - 1) / 2.0, (h - 1) / 2.0) for (w_, h) in sizes]
    pins, pinfo = pinhole_start(obs, L, sizes, cxcy, rig, hints=hints, verbose=verbose, log=log)
    rows = _subset_rows(obs, n_coarse_frames)
    starts = []
    hist = []
    for sol_p in pins:
        o = sol_p.obs
        cams = [c.copy() for c in sol_p.cams]
        wins = [w.copy() for w in rig.windows]
        cam_win = [-1] * obs.n_cams
        found = {}
        last = {}
        n_hint = {w: hinted_normal(hints, rig.cams_of(w)[:1], {rig.cams_of(w)[0]: sol_p.cams[rig.cams_of(w)[0]]},
                                   rig.windows[w])
                  for w in range(len(rig.windows)) if (hold or widen) and rig.cams_of(w)}
        held = [w for w, n in n_hint.items() if hold and n is not None]
        for ps in range(passes):
            for w in range(len(rig.windows)):
                cs_w = rig.cams_of(w)
                if not cs_w:
                    continue
                c = cs_w[0]
                cur = Rig(wins, cam_win)
                hw = w in held
                guess = n_hint[w] if hw else found[w].n if w in found else None
                cap = theta_cap if w not in found else min(theta_cap, 20.0)
                step = dtheta if w not in found else min(dtheta, 5.0)
                from_hint = None
                wide = widen and n_hint.get(w) is not None
                if search == "length":
                    best = place_window_vp([c], w, {c: sol_p.cams[c]}, cams, cur, rig, o, L, theta_cap=cap,
                                           dtheta=step, n_guess=guess, hold_normal=hw)
                    if wide:
                        # the search again about the hinted normal (every pass): a wrong hint only adds candidates
                        # that lose
                        extra = place_window_vp([c], w, {c: sol_p.cams[c]}, cams, cur, rig, o, L,
                                                theta_cap=HINT_CAP_DEG, dtheta=HINT_STEP_DEG, n_guess=n_hint[w])
                        best, from_hint = _merge_hinted(best, extra)
                else:
                    XA, XB, cnt = triangulate(cams, cur, o)
                    okf = np.all(cnt >= 2, axis=1)
                    XA[~okf], XB[~okf] = np.nan, np.nan
                    pseudo_cams = [x.copy() for x in cams]
                    pseudo_cams[c] = sol_p.cams[c].copy()          # candidates are converted from the pinhole camera
                    fr = Frames(np.where(np.isfinite(XA), 0.5 * (XA + XB), 0.0),
                                np.tile([0.0, 0.0, 1.0], (len(XA), 1)))
                    pseudo = Solution(pseudo_cams, rig, fr, L, o, Layout(obs.n_cams))
                    best, _inf = window_search(pseudo, rig, w, theta_cap=cap, dtheta=step, points=(XA, XB),
                                               base_rig=cur, n_guess=guess, retriangulate=False, hold_normal=hw,
                                               verbose=verbose, log=log)
                    if wide:
                        extra, _inf = window_search(pseudo, rig, w, theta_cap=HINT_CAP_DEG, dtheta=HINT_STEP_DEG,
                                                    points=(XA, XB), base_rig=cur, n_guess=n_hint[w],
                                                    retriangulate=False, verbose=verbose, log=log)
                        best, from_hint = _merge_hinted(best, extra)
                score, win, cd = best[0]
                found[w] = win
                last[w] = best
                wins[w] = win.copy()
                cams[c] = cd[c].copy()
                cam_win[c] = w
                hist.append({"mirror": sol_p.info.get("mirror"), "pass": ps, "window": w, "score": score,
                             **({"held": True} if hw else {}),
                             **({} if from_hint is None else {"best_from_hint": from_hint})})
                if verbose:
                    log(f"  pass {ps}, window {w + 1}: score {score:.4g}")
        # a weakly seen window (e.g. a camera nearly square-on with a long lens) can leave several minima: the last
        # pass's best candidates per window are combined into several starts
        ws_used = [w for w in range(len(rig.windows)) if w in last]
        per = [last[w] for w in ws_used]
        for cmb in _combos(per, max_combos):
            cs_c = [c.copy() for c in cams]
            wn_c = [w.copy() for w in wins]
            for k, w in enumerate(ws_used):
                _sc, win, cd = per[k][cmb[k]]
                wn_c[w] = win.copy()
                for c, cam in cd.items():
                    cs_c[c] = cam.copy()
            rg = Rig(wn_c, rig.cam_win)
            cs3, rg3, inf3 = coarse_fit(cs_c, rg, obs, L, Layout(obs.n_cams, len(rg.windows), held_normals=held),
                                        rows=rows, verbose=verbose, log=log)
            r = backprojection_residual(cs3, rg3, obs, L, rows)
            starts.append((_robust_cost(r), inf3["rms_px"], cs3, rg3))
    starts.sort(key=lambda s_: s_[0])
    info = {**pinfo, "method": "one camera per window", "search": search, "passes": passes, "placements": hist,
            "starts": len(starts),
            "seconds": {"total": time.time() - t0}}
    return starts, info


# ------------------------------------------------------------------ perturbed restarts of the final fit
PERTURB_ROUNDS = 2                     # rounds of restarts at most: a second only about an answer the first improved


def _perturb_seed(data: WandData) -> int:
    """The restarts' seed, from the detections (CRC32 of their bytes): a rerun on the same data draws the same."""
    return zlib.crc32(np.ascontiguousarray(data.uv, dtype=float).tobytes())


def window_depths(cams, rig: Rig, frames: Frames, L: float, obs: Obs) -> List[Optional[float]]:
    """Per window, the smallest depth in the water (mm) of the active wand ends seen through it (None: none).  Near 0,
    the fit may be pinned against that window: every step that would lower the cost takes a wand end out of the
    water."""
    XA, XB = frames.ends(L)
    out = []
    for w, win in enumerate(rig.windows):
        sel = obs.active & np.isin(obs.cam, rig.cams_of(w))
        if not sel.any():
            out.append(None)
            continue
        X = np.where((obs.end[sel] == 0)[:, None], XA[obs.frame[sel]], XB[obs.frame[sel]])
        out.append(float(np.min(win.depth(X))))
    return out


def perturbed_draws(sol: Solution, obs: Obs, n: int, scale: float, rng, max_redraw: int = 10):
    """n perturbations of a stage-4 answer's cameras and windows: dx = scale L z with sol.cov = L L^T (camera 1's pose
    held, as in the fit) and z ~ N(0, I), applied as stage 4 applies a step, the frames re-triangulated.  A draw is
    drawn again (up to max_redraw times, then left out) when a camera is no longer on the air side of its window or an
    observation active in `obs` no longer projects (a wand end out of the water): its cost would leave that
    observation out.  Returns ([{"cams", "rig", "L", "frames", "dx", "redraws"}], the number left out)."""
    cov = 0.5 * (sol.cov + sol.cov.T)
    try:
        Lc = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh(cov)
        Lc = V * np.sqrt(np.maximum(ev, 0.0))
    rows = np.flatnonzero(obs.active)
    out, left_out = [], 0
    for _ in range(n):
        for k in range(max_redraw + 1):
            dx = scale * (Lc @ rng.standard_normal(sol.layout.P))
            cams, rig, L = sol.layout.apply(sol.cams, sol.rig, sol.L, dx)
            if any(rig.windows[w].distance(cams[c].C) <= 0 for c, w in enumerate(rig.cam_win) if w >= 0):
                continue
            frames = initial_frames(cams, rig, obs, L)
            if not evaluate(cams, rig, frames, L, obs, rows=rows).ok.all():
                continue
            out.append({"cams": cams, "rig": rig, "L": L, "frames": frames, "dx": dx, "redraws": k})
            break
        else:
            left_out += 1
    return out, left_out


def capped_cost(sol: Solution, obs: Obs, weights, s02: float, k_sigma: float = 5.0, prior_scale: float = 1.0) -> float:
    """Every observation's weighted squared residual at sol (with its own frames), capped at k_sigma^2 s02, the
    rejection limit; an observation that does not project counts the cap; priors as in `bundle_adjust`.  Two answers
    whose active sets differ compare on the same terms this way: an outlier counts the same in both, and an answer
    pays for every good observation it rejects.  weights: per observation (None: equal); s02: the variance per
    coordinate in the weighted metric."""
    ev = evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs)
    w = np.ones(len(obs)) if weights is None else np.asarray(weights, float)
    e = np.where(ev.ok, w * np.sum(np.where(ev.ok[:, None], ev.r, 0.0) ** 2, axis=1), np.inf)
    c = float(np.sum(np.minimum(e, k_sigma ** 2 * s02)))
    if sol.layout.priors:
        rp, _ = prior_terms(sol.cams, sol.rig, sol.layout)
        ps = 1.0 if weights is not None else float(prior_scale)
        c += ps * ps * float(rp @ rp)
    return c


SAME_MINIMUM_CHI2 = 9.0                # `answers_apart` below this: the same minimum (every parameter within 3 sigma)


def answers_apart(a: Solution, b: Solution, obs: Obs, weights, s02: float) -> float:
    """How far apart two answers are, in the data: their residuals' weighted squared differences, summed over the
    observations both keep and divided by s02.  Gauge-free (image residuals), and about the squared Mahalanobis
    distance between them (dr ~ J dp), which bounds every parameter's z^2: below 9, every parameter is within 3 sigma
    (`calibrate`: the same minimum).  weights: per observation (None: equal); s02: the variance per coordinate in the
    weighted metric."""
    rows = np.flatnonzero(a.obs.active & b.obs.active)
    ea = evaluate(a.cams, a.rig, a.frames, a.L, obs, rows=rows)
    eb = evaluate(b.cams, b.rig, b.frames, b.L, obs, rows=rows)
    ok = ea.ok & eb.ok
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, float)[rows]
    d2 = np.sum((ea.r - eb.r) ** 2, axis=1)
    return float(np.sum(w[ok] * d2[ok]) / s02)


def replaces(new: Solution, old: Solution, obs: Obs, weights, s02: float, k_sigma: float = 5.0, c_old: float = None):
    """Whether `new` replaces `old` as the answer: a `capped_cost` lower by more than 1e-6 relative, and either another
    minimum (`answers_apart` >= SAME_MINIMUM_CHI2) or exactly the same observations kept.  With the same observations a
    lower cost is simply a better fit of the same data; two answers at one minimum that keep different observations
    are left as they are.  (A weakly determined window can hold two minima only chi^2 = 5 apart: W5 seed 1001, H4a.)
    weights, s02: the terms they are compared on (`calibrate`: the current answer's).  Returns (replaces, the capped
    cost relative to old's, answers_apart)."""
    c_old = capped_cost(old, obs, weights, s02, k_sigma) if c_old is None else c_old
    c_new = capped_cost(new, obs, weights, s02, k_sigma)
    apart = answers_apart(new, old, obs, weights, s02)
    kept_same = bool(np.array_equal(new.obs.active, old.obs.active))
    return bool(c_new < (1.0 - 1e-6) * c_old and (apart >= SAME_MINIMUM_CHI2 or kept_same)), c_new / c_old - 1.0, apart


def perturbed_restart(sol: Solution, data: WandData, obs: Obs, active0: np.ndarray, n_draws: int, scale: float, rng,
                      solve_kw: dict, rel_gain: float = 1e-6, verbose: bool = False, log=print):
    """One round of restarts perturbed about a stage-4 answer.  A start can end pinned against a window: a wand end on
    its water-side surface, where every step that would lower the cost takes a wand end out of the water and is
    rejected (W5 seed 2000: tests/plate_wand/2026-10-03-perturbed-starts-report.md).  Each draw (`perturbed_draws`) is
    screened by one bundle adjustment at the answer's active observations and noise weights, so that its cost compares
    with the answer's on the same terms; the cheapest draw, if it lowers that cost by more than rel_gain (relative),
    then goes through the whole of stage 4 (`solve` on fresh observations, as a start does).  That fit is accepted
    (info["accepted"]) when its `capped_cost` is lower than the answer's by more than rel_gain, both with the answer's
    weights and scale: their active sets can differ, and the right minimum can reject an observation or two whose wand
    end lies just outside the water at its windows (R4 seed 3111), which the starts' rule (`_choose`, the most
    observations kept) would hold against it.  Returns (that fit or None, info)."""
    t0 = time.time()
    info = {"draws": int(n_draws), "scale": float(scale), "rms_before_px": sol.rms,
            "min_depth_mm": window_depths(sol.cams, sol.rig, sol.frames, sol.L, sol.obs)}
    if sol.cov is None or not np.all(np.isfinite(sol.cov)):
        info.update(skipped="no covariance", seconds=time.time() - t0)
        return None, info
    o_a = Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, sol.obs.active.copy())
    # the answer's own cost on these terms (bundle_adjust's cost, no iteration)
    cost0 = float(bundle_adjust(sol.cams, sol.rig, sol.frames, sol.L, o_a, sol.layout, max_iter=0,
                                weights=sol.weights).info["weighted_cost"])
    draws, left_out = perturbed_draws(sol, o_a, n_draws, scale, rng)
    info.update(screened=len(draws), left_out=left_out, redraws=[d["redraws"] for d in draws])
    costs, best = [], None
    for d in draws:
        try:
            b = bundle_adjust(d["cams"], d["rig"], d["frames"], d["L"], o_a, sol.layout, weights=sol.weights)
        except (np.linalg.LinAlgError, ValueError):
            costs.append(float("inf"))
            continue
        c = float(b.info["weighted_cost"]) if b.info["n_fail"] == 0 else float("inf")
        costs.append(c)
        if np.isfinite(c) and (best is None or c < best[0]):
            best = (c, b)
    info["cost_rel"] = [c / cost0 - 1.0 if np.isfinite(c) else None for c in costs]
    info["screen_seconds"] = time.time() - t0
    new = None
    if best is not None and best[0] < (1.0 - rel_gain) * cost0:
        b = best[1]
        o = Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, active0.copy())
        d_b = data if b.L == data.L else WandData(data.uv, b.L, data.sizes)
        try:
            new = solve(d_b, b.cams, b.rig, obs=o, **solve_kw)
        except (np.linalg.LinAlgError, ValueError) as exc:
            info["fit_error"] = repr(exc)
    info["accepted"] = False
    if new is not None:
        s02 = float(sol.info.get("variance_factor", 1.0)) if sol.weights is not None else sol.sigma_px ** 2
        k = float(solve_kw.get("k_sigma", 5.0))
        c_old = capped_cost(sol, obs, sol.weights, s02, k)
        c_new = capped_cost(new, obs, sol.weights, s02, k)
        info["capped_rel"] = c_new / c_old - 1.0
        info["accepted"] = bool(c_new < (1.0 - rel_gain) * c_old)
    info["rms_after_px"] = None if new is None else new.rms
    info["seconds"] = time.time() - t0
    if verbose:
        log(f"  perturbed restarts: {len(draws)} draws at {scale:g} sigma ({left_out} left out), screened costs "
            "relative to the answer's " + ", ".join("-" if v is None else f"{v:+.1e}" for v in info["cost_rel"])
            + (f"; the cheapest refitted: rms {sol.rms:.5f} -> {new.rms:.5f} px, capped cost "
               f"{info['capped_rel']:+.1e}: {'accepted' if info['accepted'] else 'not accepted'}" if new is not None
               else "; none lower"))
    return new, info


# ------------------------------------------------------------------ weakly seen windows: the scan of the normal
WEAK_VIEW_DEG = 6.0          # a window whose one or two cameras all see the wand under less than this is weak
SCAN_RADIUS_DEG = 30.0       # the scan of a weak window's normal: its grid's half-width ...
SCAN_STEP_DEG = 3.0          # ... and spacing (D8: 1001's true valley found from an answer 11 deg away)
SCAN_RELEASE = 3             # at most this many of the grid's minima released
SCAN_FRAMES = 300            # the scan's held fits use at most this many frames (the release uses all)


def view_angles_deg(cams, obs: Obs) -> List[Optional[float]]:
    """Per camera, the angle under which it sees the wand: the diagonal of the box holding the 1st to 99th percentile
    of its active detections, through the lens (2 atan(diagonal / 2f)).  None with fewer than 10 detections."""
    out = []
    for c, cam in enumerate(cams):
        uv = obs.uv[obs.active & (obs.cam == c)]
        if len(uv) < 10:
            out.append(None)
            continue
        lo, hi = np.percentile(uv, [1.0, 99.0], axis=0)
        out.append(math.degrees(2.0 * math.atan(math.hypot(*(hi - lo)) / (2.0 * cam.f))))
    return out


def weak_windows(cams, rig: Rig, obs: Obs) -> List[int]:
    """The windows seen only by one or two cameras that all see the wand under less than WEAK_VIEW_DEG.  Every wrong
    answer of W5 and R4 had one (5 of the 22 cases with one, none of the 98 without: tests/plate_wand/
    p_weak_windows.py): such a window's tilt shows only weakly, and the cost has several minima."""
    ang = view_angles_deg(cams, obs)
    return [w for w in range(len(rig.windows)) if 1 <= len(rig.cams_of(w)) <= 2
            and all(ang[c] is not None and ang[c] < WEAK_VIEW_DEG for c in rig.cams_of(w))]


def _tilted(cams, rig: Rig, frames: Frames, L: float, obs: Obs, w: int, n_new) -> Rig:
    """rig with window w's normal set to n_new, its plane turned about its anchor and, when that brings the active wand
    ends seen through it nearer its water-side surface than before, moved back towards the cameras by the difference
    (a start must not begin with wand ends out of the water)."""
    win = rig.windows[w]
    A = anchors_for(cams, rig)[w]
    n_new = np.asarray(n_new, float) / np.linalg.norm(n_new)
    wn = pm.Window(n_new, float(n_new @ A) + (win.d - float(win.n @ A)), win.t, win.idx, win.name)
    sel = obs.active & np.isin(obs.cam, rig.cams_of(w))
    if sel.any():
        XA, XB = frames.ends(L)
        X = np.where((obs.end[sel] == 0)[:, None], XA[obs.frame[sel]], XB[obs.frame[sel]])
        lost = float(np.min(win.depth(X))) - float(np.min(wn.depth(X)))
        if lost > 0:
            wn = pm.Window(n_new, wn.d - lost, win.t, win.idx, win.name)
    return Rig([wn if k == w else x.copy() for k, x in enumerate(rig.windows)], list(rig.cam_win))


def scan_window_normal(sol: Solution, data: WandData, obs: Obs, active0: np.ndarray, w: int, solve_kw: dict,
                       radius: float = SCAN_RADIUS_DEG, step: float = SCAN_STEP_DEG, n_release: int = SCAN_RELEASE,
                       max_frames: int = SCAN_FRAMES, max_iter: int = 60, verbose: bool = False, log=print):
    """The scan of a weak window's normal (tests/plate_wand/2026-10-03-angles-and-hints-plan.md, 7.3a; D8).  With a
    window seen only weakly, a start can end in a wrong minimum while the true normal lies in a narrow valley tens of
    degrees away, behind a ridge (W5 seed 1001: 11 deg).  Window w's normal is set on a grid about the answer's
    (`radius`, `step` apart; the plane turned about its anchor, `_tilted`) and held, and one bundle adjustment fits
    everything else on the answer's active observations with its weights, on at most max_frames frames evenly spread
    over them (None: all); each grid point starts from a neighbour already fitted, outwards from the answer.  The
    grid's local minima cheaper than the answer's own point (at most n_release, cheapest first) are then released:
    stage 4 on fresh observations of every frame, as a start (`solve`).  Returns (the released answers, info)."""
    t0 = time.time()
    o_a = Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, sol.obs.active.copy())
    weights, frames0 = sol.weights, sol.frames
    act = np.zeros(obs.n_frames, bool)
    act[o_a.frame[o_a.active]] = True
    idx = np.flatnonzero(act)
    if max_frames and len(idx) > max_frames:
        keep = np.zeros(obs.n_frames, bool)
        keep[idx[np.linspace(0, len(idx) - 1, max_frames).round().astype(int)]] = True
        rows = keep[o_a.frame]
        weights = None if weights is None else np.asarray(weights, float)[rows]
        o_a = o_a.subset_frames(keep)
        frames0 = Frames(sol.frames.m[keep], sol.frames.w[keep])
    n0 = sol.rig.windows[w].n.copy()

    def held(cams, rig, frames, L, n_new):
        rg = _tilted(cams, rig, frames, L, o_a, w, n_new)
        lay = Layout(obs.n_cams, len(rg.windows), fit_k1=sol.layout.fit_k1, fit_L=sol.layout.fit_L,
                     anchors=anchors_for(cams, rg), held_normals=(w,))
        for pr in sol.layout.priors:
            lay.add_prior(pr["kind"], pr["cam"], pr["mean"], pr["sigma"])
        try:
            b = bundle_adjust(cams, rg, frames, L, o_a, lay, max_iter=max_iter, weights=weights)
        except (np.linalg.LinAlgError, ValueError):
            return None
        return b if b.info["n_fail"] == 0 else None

    info = {"window": int(w), "radius_deg": float(radius), "step_deg": float(step), "frames": int(o_a.n_frames)}
    centre = held(sol.cams, sol.rig, frames0, sol.L, n0)
    if centre is None:
        info.update(skipped="the answer's own normal held did not fit", seconds=time.time() - t0)
        return [], info
    kmax = int(round(radius / step))
    pts = {(i, j) for i in range(-kmax, kmax + 1) for j in range(-kmax, kmax + 1) if i * i + j * j <= kmax * kmax}
    st = math.radians(step)
    fits, cost = {(0, 0): centre}, {(0, 0): float(centre.info["weighted_cost"])}
    queue, seen = deque([(0, 0)]), {(0, 0)}
    while queue:
        ij = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (ij[0] + di, ij[1] + dj)
            if nb in seen or nb not in pts:
                continue
            seen.add(nb)
            f = fits[ij]
            b = held(f.cams, f.rig, f.frames, f.L, pm.retract_normal(n0, (nb[0] * st, nb[1] * st)))
            if b is not None:
                fits[nb], cost[nb] = b, float(b.info["weighted_cost"])
                queue.append(nb)
    c0 = cost[(0, 0)]
    minima = sorted((k for k in cost if k != (0, 0) and cost[k] < (1.0 - 1e-6) * c0
                     and all(cost[k] < cost[(k[0] + a, k[1] + b)] for a in (-1, 0, 1) for b in (-1, 0, 1)
                             if (a, b) != (0, 0) and (k[0] + a, k[1] + b) in cost)), key=lambda k: cost[k])
    info.update(points=len(pts), fitted=len(fits),
                minima=[{"deg": math.hypot(*k) * step, "cost_rel": cost[k] / c0 - 1.0} for k in minima[:10]])
    released = []
    for k in minima[:n_release]:
        f = fits[k]
        o = Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, active0.copy())
        d_b = data if f.L == data.L else WandData(data.uv, f.L, data.sizes)
        try:
            released.append(solve(d_b, f.cams, f.rig, obs=o, **solve_kw))
        except (np.linalg.LinAlgError, ValueError):
            continue
    info.update(released_rms_px=[s.rms for s in released], seconds=time.time() - t0)
    if verbose:
        log(f"  scan of window {w + 1}'s normal: {len(fits)} of {len(pts)} grid points fitted, {len(minima)} minima "
            f"below the answer's; released: " + (", ".join(f"rms {x:.5f}" for x in info["released_rms_px"]) or "none")
            + f" ({info['seconds']:.0f} s)")
    return released, info


# ------------------------------------------------------------------ the output frame
def to_output_frame(sol: Solution) -> Solution:
    """Move the answer (no residual changes) so that Z is window 1's normal (camera -> water), Y is along n1 x n2 with
    two or more windows (when not parallel) or camera 1's image "up" in window 1's plane, X = Y x Z, and the origin
    is the mean of the active frames' midpoints.  The covariance is recomputed in the new frame (camera 1 held)."""
    cams, rig, frames = sol.cams, sol.rig, sol.frames
    if rig.windows:
        Z = rig.windows[0].n.copy()
    else:
        Z = cams[0].R[2].copy()
    Y = None
    if len(rig.windows) >= 2:
        y = np.cross(rig.windows[0].n, rig.windows[1].n)
        if np.linalg.norm(y) > 1e-3:
            Y = y / np.linalg.norm(y)
    if Y is None:
        up = -cams[0].R[1]
        Y = up - float(up @ Z) * Z
        if np.linalg.norm(Y) < 1e-6:
            Y = np.cross(Z, cams[0].R[0])
        Y /= np.linalg.norm(Y)
    X = np.cross(Y, Z)
    Q = np.vstack([X, Y, Z])
    act = np.zeros(sol.obs.n_frames, bool)
    act[sol.obs.frame[sol.obs.active]] = True
    O = frames.m[act].mean(axis=0) if act.any() else frames.m.mean(axis=0)
    new_c = []
    for c in cams:
        k = c.copy()
        k.R = c.R @ Q.T
        k.C = Q @ (c.C - O)
        new_c.append(k)
    new_w = [pm.Window(Q @ w.n, w.d - float(w.n @ O), w.t, w.idx, w.name) for w in rig.windows]
    sol.cams = new_c
    sol.rig = Rig(new_w, rig.cam_win)
    sol.frames = Frames((frames.m - O) @ Q.T, frames.w @ Q.T)
    sol.layout.anchors = [None if a is None else Q @ (np.asarray(a) - O) for a in sol.layout.anchors]
    sol.info["output_frame"] = {"Q": Q.tolist(), "origin": O.tolist(),
                                "note": "X_out = Q (X_fit - origin); covariance recomputed in the output frame, "
                                        "camera 1's pose held"}
    return covariance(sol, weights=sol.weights)


# ------------------------------------------------------------------ the whole calibration
def _choose(results: List[Solution], noise_weights: str) -> Solution:
    """The answer among stage-4 results: of those whose RMS is within 1.5x of the best, the one that keeps the most
    observations, then the most likely; the first of equals."""
    best_rms = min(s.rms for s in results)
    pool = [s for s in results if s.rms <= 1.5 * best_rms]
    return min(pool, key=lambda s: (-int(s.obs.active.sum()), _log_likelihood_key(s, noise_weights)))


def calibrate(data: WandData, rig: Rig, start=None, hints: dict = None, fit_k1: bool = False, fit_L: bool = False,
              k_sigma: float = 5.0, n_final: int = 2, noise_weights: str = "camera", sigma_cam_px=None,
              more_starts_px: float = 1.0, theta_cap: float = 50.0, dtheta: float = 9.0, output_frame: bool = True,
              max_starts: int = 6, perturb_starts: int = 4, perturb_scale: float = 2.0, perturb_all: bool = False,
              scan_weak: bool = True, verbose: bool = False, log=print) -> Solution:
    """Stages 0-5: from the detections to the cameras and windows, with no camera or window input (or from `start`,
    cameras in the frame of rig.windows, which are then taken as given).  A rig with one window is started whole
    (`bootstrap`); one with several windows of which some have two or more cameras window by window
    (`bootstrap_groups`); one whose windows have one camera each by alternating placements (`bootstrap_singles`).  The best n_final
    starts are carried through stage 4; when the best of them has a coarse-fit RMS above more_starts_px (noisy data), or
    when they end in different minima (RMS apart by more than 0.1 %), every start is, up to max_starts.  The answer is chosen among those whose RMS is within 1.5x of the best (a start stuck in a wrong
    minimum fits badly, and its inflated noise estimate rejects little, so it must not win on the count): most
    observations kept, then the highest likelihood (`_choose`).
    On a rig that uses several windows, perturb_starts restarts are then drawn about the answer from its covariance
    scaled by perturb_scale and screened, the cheapest refitted if it lowers the cost (`perturbed_restart`); the refit
    replaces the answer when it is better on the same terms (`capped_cost`), and a second round follows.  A start can
    end pinned against a window (W5 seed 2000).  perturb_starts=0 turns this off; perturb_all=True applies it to every rig and to a given
    start.  The draws' seed comes from the detections, so the same data give the same answer.
    On a rig that uses several windows, each weakly seen window (`weak_windows`) then has its normal scanned
    (`scan_window_normal`; scan_weak=False turns this off): a released fit replaces the answer, after its restarts,
    by `replaces` (as a hinted path's answer below).  info["scan"] reports it.
    hints (optional): {"f": [px per camera or None]: the lenses' focal lengths, used for the pinhole start's distances;
    "f_prior_rel": sigma: also a soft prior on log f in stage 4; "normals": [per camera None or the normal of its
    window in the camera's own frame -- x along the image rows, y down, z along the optical axis, either sign --, e.g.
    (sin 30 deg, 0, cos 30 deg) for a camera turned 30 deg to the left of square-on; a rough guess]: two more start
    paths then run beside the default one, which ignores the hints: "held", every hinted window's normal held at its
    hint (`hinted_normal`) while the start is built (the window search tries only its offsets; the window groups' own
    fits and the coarse fit hold it too) and freed in stage 4; "searched", every hinted window searched a second time
    about its hint, the candidates merged with the default search's (`_merge_hinted`).  Each path's answer gets its
    own restarts from the same seed, so the default path gives exactly the answer without hints; a path whose stage-4
    answer is the same minimum as one already found (`answers_apart` below SAME_MINIMUM_CHI2) skips its restarts.  A
    hinted path's answer replaces the default one only by `replaces` (a lower `capped_cost` with the default answer's
    weights, and another minimum or the same observations kept): a rough or wrong hint costs time, not the answer, and
    the answer's normals come from the data.  No prior.  info["hints"] holds each hint's angle from the answer, every
    path's outcome and the path chosen; info["starts"] and info["bootstrap"] describe the default path}.  With fit_L
    the wand length is estimated too: only the plates' thickness then fixes the scale, weakly."""
    hints = hints or {}
    obs = observations(data)
    t0 = time.time()
    multi = sum(1 for w in range(len(rig.windows)) if len(rig.cams_of(w)) >= 2)
    used = sum(1 for w in range(len(rig.windows)) if rig.cams_of(w))
    few = any(1 <= len(rig.cams_of(w)) <= 2 for w in range(len(rig.windows)))
    searches = ("length", "fixed") if few else ("fixed",)

    def make_starts(hold: bool = False, widen: bool = False):
        """Stages 1-3 (hold: the hint-held start; widen: the hint-searched start)."""
        if used > 1:
            # several windows: each search's starts are pooled (neither is right on every rig with a window of one or
            # two cameras; the final costs decide)
            per_search, binfo_, cache = [], {}, {}
            for sr in searches:
                try:
                    if multi >= 1:
                        st, inf = bootstrap_groups(obs, rig, data.L, data.sizes, hints=hints, theta_cap=theta_cap,
                                                   dtheta=dtheta, search=sr, cache=cache, hold=hold, widen=widen,
                                                   verbose=verbose, log=log)
                    else:
                        st, inf = bootstrap_singles(obs, rig, data.L, data.sizes, hints=hints, theta_cap=theta_cap,
                                                    dtheta=dtheta, search=sr, hold=hold, widen=widen,
                                                    verbose=verbose, log=log)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    # e.g. a window's cameras that never see the same frames cannot be calibrated as a group
                    t_f = time.time()
                    st, inf = bootstrap(obs, rig, data.L, data.sizes, hints=hints, theta_cap=theta_cap,
                                        dtheta=dtheta, search=sr, hold=hold, widen=widen, verbose=verbose, log=log)
                    inf = {**inf, "fallback_from": repr(exc)}
                    inf["seconds"]["total"] = inf["seconds"].get("total", time.time() - t_f)
                per_search.append(sorted(st, key=lambda s_: s_[0]))
                binfo_[sr] = inf
            # the searches' starts taken in turn, each search's best first: their costs are not comparable enough to
            # rank one search's starts behind the other's (W5 seed 1031: the right start ranked seventh)
            starts_ = [x for tup in itertools.zip_longest(*per_search) for x in tup if x is not None]
            binfo_["seconds"] = {"total": sum(v["seconds"]["total"] for v in binfo_.values())}
            return starts_, binfo_
        return bootstrap(obs, rig, data.L, data.sizes, hints=hints, theta_cap=theta_cap, dtheta=dtheta, hold=hold,
                         widen=widen, verbose=verbose, log=log)

    binfo = {}
    if start is not None:
        starts = [(0.0, float("nan"), [c.copy() for c in start], rig.copy())]
    else:
        starts, binfo = make_starts()
    priors = []
    if hints.get("f") is not None and hints.get("f_prior_rel"):
        for c, fv in enumerate(hints["f"]):
            if fv:
                priors.append({"kind": "logf", "cam": c, "mean": math.log(float(fv)),
                               "sigma": float(hints["f_prior_rel"])})
    solve_kw = dict(fit_k1=fit_k1, fit_L=fit_L, k_sigma=k_sigma, noise_weights=noise_weights,
                    sigma_cam_px=sigma_cam_px, priors=priors, verbose=False, log=log)
    active0 = obs.active.copy()

    def fit_starts(starts_):
        """Stage 4 from the starts; returns (the answer by `_choose`, every result, starts fitted, their coarse
        RMS)."""
        results_, stage3_, tried_ = [], [], 0
        for k, (_cost, rms3, cams0, rig0) in enumerate(starts_):
            if k >= max(n_final, max_starts):
                break
            if k >= n_final and min(stage3_, default=0.0) <= more_starts_px:
                rr = [x.rms for x in results_]
                # starts that ended in different minima (RMS apart by more than 0.1 %): the landscape has several, so
                # every start is tried
                if not rr or max(rr) <= (1.0 + 1e-3) * min(rr):
                    break
            o = Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, active0.copy())
            stage3_.append(rms3 if np.isfinite(rms3) else 0.0)
            try:
                s_ = solve(data, cams0, rig0, obs=o, **solve_kw)
            except (np.linalg.LinAlgError, ValueError) as exc:
                if verbose:
                    log(f"  start {k}: failed ({exc})")
                continue
            tried_ += 1
            results_.append(s_)
            if verbose:
                log(f"  start {k}: coarse rms {rms3:.3f} px -> stage 4 rms {s_.rms:.5f} px, active "
                    f"{int(o.active.sum())}/{len(o)}")
        if not results_:
            raise ValueError("no start converged")
        return _choose(results_, noise_weights), results_, tried_, stage3_

    # restarts perturbed about an answer: rigs that use several windows (every rig with perturb_all)
    used_wins = sum(1 for w in range(len(rig.windows)) if rig.cams_of(w))
    do_pert = perturb_starts > 0 and (perturb_all or (start is None and used_wins > 1))

    def restarts(sol_):
        """The restarts about an answer, from the restarts' seed; returns (the answer, their info or None)."""
        if not do_pert:
            return sol_, None
        t_p = time.time()
        rng = np.random.default_rng(_perturb_seed(data))
        pert_ = {"draws": int(perturb_starts), "scale": float(perturb_scale), "rms_before_px": sol_.rms, "rounds": [],
                 "improved": False}
        for _round in range(PERTURB_ROUNDS):
            new, pinf = perturbed_restart(sol_, data, obs, active0, perturb_starts, perturb_scale, rng, solve_kw,
                                          verbose=verbose, log=log)
            pert_["rounds"].append(pinf)
            if new is None or not pinf["accepted"]:
                break
            sol_ = new
            pert_["improved"] = True
        pert_["rms_px"] = sol_.rms
        pert_["seconds"] = time.time() - t_p
        return sol_, pert_

    def terms(s_):
        """An answer's weights and variance per coordinate in their metric: the terms answers are compared on."""
        return s_.weights, (float(s_.info.get("variance_factor", 1.0)) if s_.weights is not None else s_.sigma_px ** 2)

    hinted = start is None and any(_cam_hint(hints, c) is not None for c in range(obs.n_cams))
    t4 = time.time()
    try:
        sol, results, tried, stage3 = fit_starts(starts)
    except ValueError:
        if not hinted:
            raise
        sol, results, tried, stage3 = None, [], 0, []
    t_ml = time.time() - t4
    pert = None
    found = []                                  # (path, its stage-4 answer, its answer after the restarts)
    if sol is not None:
        sol4 = sol
        sol, pert = restarts(sol)
        found.append(("default", sol4, sol))
    # window-normal hints: two more paths, the hint-held start and the hint-searched start, each with its own restarts;
    # a hinted answer replaces the default one only when it is another minimum with a lower capped cost
    paths, chosen = None, "default"
    if hinted:
        paths = {}
        for name, kw_ in (("held", {"hold": True}), ("searched", {"widen": True})):
            t_h = time.time()
            p = {}
            try:
                starts_h, binfo_h = make_starts(**kw_)
                sol_h, results_h, tried_h, _st3 = fit_starts(starts_h)
            except (ValueError, np.linalg.LinAlgError) as exc:
                p["error"] = repr(exc)
            else:
                p.update(available=len(starts_h), tried=tried_h, stage4_rms_px=[s.rms for s in results_h],
                         stage4_answer_rms_px=sol_h.rms, bootstrap=binfo_h)
                same = None
                if found:
                    wts, s02 = terms(found[0][2])
                    same = next((nm for nm, s4, sf in found
                                 if min(answers_apart(sol_h, s4, obs, wts, s02),
                                        answers_apart(sol_h, sf, obs, wts, s02)) < SAME_MINIMUM_CHI2), None)
                if same is not None:
                    p["same_as"] = same                 # its restarts would repeat that answer's; still a candidate
                    found.append((name, sol_h, sol_h))
                else:
                    sol4_h = sol_h
                    sol_h, pert_h = restarts(sol_h)
                    p["perturbed"] = None if pert_h is None else {
                        k: pert_h[k] for k in ("improved", "rms_before_px", "rms_px", "seconds")}
                    found.append((name, sol4_h, sol_h))
                p["rms_px"] = sol_h.rms
            p["seconds"] = time.time() - t_h
            paths[name] = p
            if verbose:
                if "error" in p:
                    msg = p["error"]
                elif "same_as" in p:
                    msg = (f"stage 4 rms {p['stage4_answer_rms_px']:.5f} px, the same minimum as the {p['same_as']} "
                           "path's")
                else:
                    msg = f"stage 4 rms {p['stage4_answer_rms_px']:.5f} px, after the restarts {p['rms_px']:.5f} px"
                log(f"  hint path {name}: {msg}")
        if not found:
            raise ValueError("no start converged")
        base_name, _b4, base = found[0]
        wts, s02 = terms(base)
        c_base = capped_cost(base, obs, wts, s02, k_sigma)
        best = (c_base, base_name, base)
        for nm, _s4, sf in found[1:]:
            ok, crel, apart = replaces(sf, base, obs, wts, s02, k_sigma, c_base)
            paths[nm].update(capped_rel=crel, apart_chi2=apart, replaces=ok)
            if ok and (1.0 + crel) * c_base < best[0]:
                best = ((1.0 + crel) * c_base, nm, sf)
        chosen, sol = best[1], best[2]
        if verbose:
            log(f"  the answer: the {chosen} path's, rms {sol.rms:.5f} px")
    # weakly seen windows: the scan of each one's normal; a released fit replaces the answer on the same terms
    scan = None
    if scan_weak and start is None and used_wins > 1:
        weak = weak_windows(sol.cams, sol.rig, sol.obs)
        if weak:
            t_s = time.time()
            scan = {"windows": [], "improved": False, "rms_before_px": sol.rms}
            for w in weak:
                fits_r, sinf = scan_window_normal(sol, data, obs, active0, w, solve_kw, verbose=verbose, log=log)
                wts, s02 = terms(sol)
                c_sol = capped_cost(sol, obs, wts, s02, k_sigma)
                best_s, sinf["released"] = None, []
                for s_ in fits_r:
                    ok, crel, apart = replaces(s_, sol, obs, wts, s02, k_sigma, c_sol)
                    sinf["released"].append({"rms_px": s_.rms, "capped_rel": crel, "apart_chi2": apart,
                                             "replaces": ok})
                    if ok and (best_s is None or crel < best_s[0]):
                        best_s = (crel, s_)
                sinf["accepted"] = False
                if best_s is not None:
                    new, _pert_s = restarts(best_s[1])
                    ok, crel, _apart = replaces(new, sol, obs, wts, s02, k_sigma, c_sol)
                    if ok:
                        sinf.update(accepted=True, rms_px=new.rms, capped_rel=crel)
                        sol = new
                        scan["improved"] = True
                scan["windows"].append(sinf)
                if verbose:
                    log(f"  scan of window {w + 1}: " + (f"the answer IMPROVED, rms {sol.rms:.5f} px" if sinf["accepted"]
                                                         else "nothing better"))
            scan["rms_px"] = sol.rms
            scan["seconds"] = time.time() - t_s
    sol.info["sigmas_fit_frame"] = {"cameras": camera_sigmas(sol.cams, sol.rig, sol.layout, sol.cov),
                                    "windows": window_sigmas(sol.cams, sol.rig, sol.layout, sol.cov)}
    if hinted:
        # each hint's angle from the answer's normal, in its camera's frame (a frame change leaves it unchanged)
        off = []
        for c in range(obs.n_cams):
            v, w = _cam_hint(hints, c), sol.rig.cam_win[c]
            if v is None or w < 0:
                off.append(None)
                continue
            v = np.asarray(v, float)
            m = sol.cams[c].R @ sol.rig.windows[w].n
            off.append(math.degrees(math.acos(min(1.0, abs(float(v @ m)) / float(np.linalg.norm(v))))))
        sol.info["hints"] = {"normals_off_deg": off, "paths": paths, "chosen": chosen}
    sol.info["scan"] = scan
    if output_frame:
        sol = to_output_frame(sol)
    sol.info["bootstrap"] = binfo
    sol.info["starts"] = {"available": len(starts), "tried": tried, "n_final": n_final,
                          "more_starts_px": more_starts_px, "stage3_rms_px": stage3, "start_given": start is not None,
                          "stage4_rms_px": [s.rms for s in results], "perturbed": pert}
    sol.info["timing_s"] = {"bootstrap": binfo.get("seconds", {}), "stage4": t_ml,
                            "perturbed": None if pert is None else pert["seconds"],
                            "hint_paths": None if paths is None else {k: v["seconds"] for k, v in paths.items()},
                            "scan": None if scan is None else scan["seconds"], "total": time.time() - t0}
    return sol
