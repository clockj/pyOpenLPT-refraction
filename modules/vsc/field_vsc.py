"""
Field VSC: a per-camera image correction fitted from a run's long tracks, for cameras behind curved windows (PINCYL,
PINSPH) and any camera the library can project.

What it fits. Every long-track point X seen by a set S of cameras has detections d_c near its projections P_c(X).
The residuals r_c = d_c - P_c(X) mix two things: X itself is off (a 3-D move explains that part) and the cameras'
images sit off where the model puts them (the correction c_c(d), smooth over each sensor). Only the part of the
stacked residual (2|S| numbers) that no 3-D move explains says anything about the cameras: with two cameras it is one
number per point, the offset across the epipolar curve; with three or four, 3 or 5. The fit (projected least squares)
uses that part only, with:
  - a cubic B-spline of 10 x 6 intervals over each camera's sensor, per component (du, dv);
  - a bending penalty, its weight chosen by the held-out loss (alternate frames), and a 1 px prior on the size, which
    leaves the directions the tracks cannot see as the model has them;
  - Cauchy weights (c = 2) against ghosts and blended detections, in units of the detections' noise
    sqrt(s_f^2 + (beta L)^2) (L the line's box length), s_f >= 0.15 px, the noise model fitted from the residuals.
These are the settings of tests/cylinder_selfcal (2026-10-08-report.md), where two passes at a 1.0 px tolerance gave
T1 11 % more real long tracks with a fifth fewer ghosts.

Passes. The tracks were found with the tolerance selecting particles, so a pass under-reports the error: refit after
tracking again with the correction (two or three passes). A camera that already has a correction c1 gets the rest c2
fitted against its corrected model, and the total c(d) = c2(d) + c1(d - c2(d)) replaces it.

Output: camFile_VSC/vsc_cam<k>_corr_p<n>.txt per camera (modules/vsc/correction_io.py's format, read by the library
as config.json's "correction") and config.json pointing at them. The matching tolerance is left as it is (T1 gained
at 1.0 px after two passes, not at 1.5 px); the log says so.
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from modules import stb_config
from modules.vsc.correction_io import Correction, compose

N_U, N_V = 10, 6            # spline intervals across and down each sensor
TAU_PX = 1.0                # prior on the field's size
CAUCHY_C = 2.0              # robust weights, in noise units per degree of freedom
S_FLOOR = 0.15              # px: the noise model's floor
NOISE0 = (0.3, 0.05)        # the noise model's start (s_f px, beta)
LAMS = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)   # the self-calibration often chose its grid's lowest weight: one lower
ROUNDS = 2                  # rounds of (bending weight, noise model)
COVER_PX = 48.0             # the field is the fit within this of the data; beyond, a smooth fill
JAC_STEP_MM = 0.01          # central differences of the projections
NODE_STEP_PX = 8.0          # the correction file's grid
CURVED = ("PINCYL", "PINSPH")


# ------------------------------------------------------------------ the spline field
def bspline_basis(x, n_int: int, length: float):
    """Uniform cubic B-spline basis on [0, length] with n_int intervals: (first index (n,), values (n, 4))."""
    h = length / n_int
    s = np.clip(np.asarray(x, float) / h, 0.0, n_int - 1e-9)
    m = np.floor(s).astype(int)
    t = s - m
    t2, t3 = t * t, t * t * t
    vals = np.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], axis=-1)
    return m, vals


def _gram_1d(n_int: int, length: float, deriv: int, n_quad: int = 8) -> np.ndarray:
    """Integral over the domain of B_i^(deriv) B_j^(deriv) (cubic, uniform)."""
    h = length / n_int
    nb = n_int + 3
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    G = np.zeros((nb, nb))
    t = 0.5 * (xg + 1.0)
    w = 0.5 * wg * h
    if deriv == 0:
        A = np.stack([(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
                      (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6], axis=-1)
    elif deriv == 1:
        A = np.stack([-(1 - t) ** 2 / 2, (3 * t ** 2 - 4 * t) / 2, (-3 * t ** 2 + 2 * t + 1) / 2, t ** 2 / 2],
                     axis=-1) / h
    else:
        A = np.stack([(1 - t), (3 * t - 2), (-3 * t + 1), t], axis=-1) / h ** 2
    for m in range(n_int):
        G[m:m + 4, m:m + 4] += (A * w[:, None]).T @ A
    return G


class SplineField:
    """Per camera and component a tensor-product cubic B-spline over that camera's sensor."""

    def __init__(self, sizes: List[Tuple[int, int]], n_u: int = N_U, n_v: int = N_V):
        self.sizes = [(int(w), int(h)) for w, h in sizes]     # (n_col, n_row) per camera
        self.n_cam = len(sizes)
        self.n_u, self.n_v = n_u, n_v
        self.nbu, self.nbv = n_u + 3, n_v + 3
        self.nb = self.nbu * self.nbv
        self.theta = np.zeros(self.n_cam * 2 * self.nb)

    def index(self, cam: int, comp: int) -> int:
        return (cam * 2 + comp) * self.nb

    def basis_rows(self, cam: int, uv):
        """Columns (n, 16) within one component block and values (n, 16) at pixels uv (n, 2) of camera cam."""
        w, h = self.sizes[cam]
        uv = np.asarray(uv, float)
        mu, bu = bspline_basis(uv[:, 0] + 0.5, self.n_u, w)
        mv, bv = bspline_basis(uv[:, 1] + 0.5, self.n_v, h)
        cols = ((mu[:, None] + np.arange(4)[None, :])[:, :, None] * self.nbv
                + (mv[:, None] + np.arange(4)[None, :])[:, None, :]).reshape(len(uv), 16)
        vals = (bu[:, :, None] * bv[:, None, :]).reshape(len(uv), 16)
        return cols, vals

    def penalty(self, cam: int) -> np.ndarray:
        """The bending energy of one component block of camera cam."""
        w, h = self.sizes[cam]
        G0u, G1u, G2u = (_gram_1d(self.n_u, w, d) for d in (0, 1, 2))
        G0v, G1v, G2v = (_gram_1d(self.n_v, h, d) for d in (0, 1, 2))
        return np.kron(G2u, G0v) + 2 * np.kron(G1u, G1v) + np.kron(G0u, G2v)

    def eval_points(self, cam: int, uv, theta=None) -> np.ndarray:
        """(du, dv) of the spline at pixels uv (n, 2) of camera cam."""
        th = self.theta if theta is None else theta
        uv = np.atleast_2d(np.asarray(uv, float))
        cols, vals = self.basis_rows(cam, uv)
        out = np.zeros((len(uv), 2))
        for comp in (0, 1):
            t = th[self.index(cam, comp):self.index(cam, comp) + self.nb]
            out[:, comp] = (vals * t[cols]).sum(axis=1)
        return out


class FitData:
    """Per point: the cameras seen (mask (n, n_cam)), detections d (n, n_cam, 2), residuals r = d - P(X), Jacobians J
    (n, n_cam, 2, 3), the detections' box lengths L (n, n_cam, 2) for the noise model, and the frame."""

    def __init__(self, mask, d, r, J, L, frame):
        self.mask, self.d, self.r, self.J, self.L, self.frame = mask, d, r, J, L, frame

    def subset(self, sel) -> "FitData":
        return FitData(self.mask[sel], self.d[sel], self.r[sel], self.J[sel], self.L[sel], self.frame[sel])

    def __len__(self) -> int:
        return len(self.mask)

    @staticmethod
    def concat(parts: List["FitData"]) -> "FitData":
        return FitData(*[np.concatenate([getattr(p, k) for p in parts]) for k in ("mask", "d", "r", "J", "L", "frame")])


def noise_sigma(L, noise):
    s_f, beta = noise
    return np.sqrt(s_f ** 2 + (beta * np.abs(L)) ** 2)


def _groups(fd: FitData) -> Dict[tuple, np.ndarray]:
    out = {}
    for i, m in enumerate(fd.mask):
        k = tuple(np.flatnonzero(m))
        if len(k) >= 2:
            out.setdefault(k, []).append(i)
    return {k: np.array(v) for k, v in out.items()}


def projected_system(field: SplineField, fd: FitData, noise):
    """The projected rows: (Z sparse (rows, n_theta), y (rows,), point index of each row, dof of each point). Each
    point's whitened residual (2|S| numbers) minus the span of its whitened Jacobian."""
    import scipy.sparse as sp
    rows_i, cols_i, vals_i, ys, pidx = [], [], [], [], []
    row0 = 0
    dof = np.zeros(len(fd), int)
    for cams_s, idx in _groups(fd).items():
        s = len(cams_s)
        cams_s = list(cams_s)
        r = fd.r[idx][:, cams_s].reshape(len(idx), 2 * s)
        J = fd.J[idx][:, cams_s].reshape(len(idx), 2 * s, 3)
        sig = noise_sigma(fd.L[idx][:, cams_s].reshape(len(idx), 2 * s), noise)
        rw = r / sig
        Jw = J / sig[:, :, None]
        good = np.all(np.isfinite(Jw.reshape(len(idx), -1)), axis=1) & np.all(np.isfinite(rw), axis=1)
        idx, rw, Jw, sig = idx[good], rw[good], Jw[good], sig[good]
        if not len(idx):
            continue
        U, _S, _Vt = np.linalg.svd(Jw, full_matrices=True)
        Nb = U[:, :, 3:]
        k = 2 * s - 3
        dof[idx] = k
        y = np.einsum("nik,ni->nk", Nb, rw)
        cols_all, vals_all = [], []
        for a, c in enumerate(cams_s):
            cols, vals = field.basis_rows(c, fd.d[idx, c])
            for comp in (0, 1):
                cols_all.append(cols + field.index(c, comp))
                vals_all.append(vals / sig[:, 2 * a + comp][:, None])
        Cc = np.stack(cols_all, axis=1)
        Cv = np.stack(vals_all, axis=1)
        for q in range(k):
            v = Nb[:, :, q][:, :, None] * Cv
            r_idx = row0 + np.arange(len(idx)) * k + q
            rows_i.append(np.repeat(r_idx, 2 * s * 16))
            cols_i.append(Cc.reshape(len(idx), -1).ravel())
            vals_i.append(v.reshape(len(idx), -1).ravel())
        ys.append(y.ravel())
        pidx.append(np.repeat(idx, k))
        row0 += len(idx) * k
    n_theta = field.theta.size
    if not ys:
        return sp.csr_matrix((0, n_theta)), np.zeros(0), np.zeros(0, int), dof
    Z = sp.csr_matrix((np.concatenate(vals_i), (np.concatenate(rows_i), np.concatenate(cols_i))),
                      shape=(row0, n_theta))
    return Z, np.concatenate(ys), np.concatenate(pidx), dof


def _point_norms(e, pidx, n_points):
    out = np.zeros(n_points)
    np.add.at(out, pidx, e ** 2)
    return out


def _cauchy_weights(nrm):
    return 1.0 / (1.0 + (nrm / CAUCHY_C) ** 2)


def _cauchy_loss(t):
    return 0.5 * CAUCHY_C ** 2 * np.log1p((np.abs(t) / CAUCHY_C) ** 2)


def fit_field(field: SplineField, fd: FitData, lam: float, noise, iters: int = 8) -> dict:
    """Projected least squares with the bending penalty lam, the prior |theta|^2 / tau^2 and Cauchy weights on each
    point's projected residual. Sets field.theta; returns a record."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    Z, y, pidx, dof = projected_system(field, fd, noise)
    Pen = sp.block_diag([field.penalty(c) for c in range(field.n_cam) for _comp in (0, 1)], format="csr")
    n = field.theta.size
    R = lam * Pen + (1.0 / TAU_PX ** 2) * sp.identity(n, format="csr")
    w = np.ones(len(fd))
    theta = np.zeros(n)
    e = y
    it = 0
    for it in range(iters):
        wr = w[pidx]
        A = (Z.T @ sp.diags(wr) @ Z + R).tocsc()
        b = Z.T @ (wr * y)
        theta_new = spla.spsolve(A, b)
        e = y - Z @ theta_new
        nrm = np.sqrt(_point_norms(e, pidx, len(fd)) / np.maximum(dof, 1))
        w = _cauchy_weights(nrm)
        w[dof == 0] = 0.0
        delta = float(np.max(np.abs(theta_new - theta)))
        theta = theta_new
        if delta < 1e-5:
            break
    field.theta = theta
    used = dof > 0
    return {"iterations": it + 1, "points": int(used.sum()), "rows": int(len(y)), "lam": lam, "noise": list(noise),
            "downweighted": float(np.mean(w[used] < 0.5)) if used.any() else None,
            "rms_before": float(np.sqrt(np.mean(y ** 2))) if len(y) else None,
            "rms_after": float(np.sqrt(np.mean(e ** 2))) if len(y) else None}


def heldout_score(field: SplineField, fd: FitData, noise) -> dict:
    Z, y, pidx, dof = projected_system(field, fd, noise)
    e = y - Z @ field.theta
    per0 = np.sqrt(_point_norms(y, pidx, len(fd)) / np.maximum(dof, 1))
    per1 = np.sqrt(_point_norms(e, pidx, len(fd)) / np.maximum(dof, 1))
    used = dof > 0
    if not used.any():
        return {"points": 0, "loss_before": float("nan"), "loss_after": float("nan"), "mad_before": float("nan"),
                "mad_after": float("nan")}
    return {"points": int(used.sum()), "loss_before": float(np.mean(_cauchy_loss(per0[used]))),
            "loss_after": float(np.mean(_cauchy_loss(per1[used]))),
            "mad_before": float(1.4826 * np.median(np.abs(y))), "mad_after": float(1.4826 * np.median(np.abs(e)))}


def estimate_noise(field: SplineField, fd: FitData, bins: int = 8):
    """The noise model (s_f, beta) from the unwhitened projected residual: per null direction the variance is
    s_f^2 + beta^2 l, l = sum_i N_i^2 L_i^2; robust variances in quantile bins of l, then a line."""
    zs, ls = [], []
    for cams_s, idx in _groups(fd).items():
        s = len(cams_s)
        cams_s = list(cams_s)
        r = fd.r[idx][:, cams_s].reshape(len(idx), 2 * s)
        J = fd.J[idx][:, cams_s].reshape(len(idx), 2 * s, 3)
        L = fd.L[idx][:, cams_s].reshape(len(idx), 2 * s)
        c = np.zeros_like(r)
        for a, cc in enumerate(cams_s):
            c[:, 2 * a:2 * a + 2] = field.eval_points(cc, fd.d[idx, cc])
        good = np.all(np.isfinite(J.reshape(len(idx), -1)), axis=1) & np.all(np.isfinite(r), axis=1)
        if not good.any():
            continue
        U, _S, _Vt = np.linalg.svd(J[good], full_matrices=True)
        Nb = U[:, :, 3:]
        zs.append(np.einsum("nik,ni->nk", Nb, (r - c)[good]).ravel())
        ls.append(np.einsum("nik,ni->nk", Nb ** 2, L[good] ** 2).ravel())
    z, l = np.concatenate(zs), np.concatenate(ls)
    edges = np.quantile(l, np.linspace(0, 1, bins + 1))
    lb, vb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (l >= lo) & (l <= hi)
        if m.sum() < 30:
            continue
        lb.append(float(np.median(l[m])))
        vb.append(float((1.4826 * np.median(np.abs(z[m] - np.median(z[m])))) ** 2))
    if len(lb) < 2:
        return NOISE0
    A = np.column_stack([np.ones(len(lb)), np.array(lb)])
    coef, *_ = np.linalg.lstsq(A, np.array(vb), rcond=None)
    return float(math.sqrt(max(coef[0], 1e-4))), float(math.sqrt(max(coef[1], 0.0)))


def fit_with_noise(sizes, fd_fit: FitData, fd_test: FitData, fd_all: FitData,
                   log: Callable[[str], None], lams=LAMS) -> Tuple[SplineField, dict]:
    """The bending weight by held-out loss and the noise model by the residuals, alternately (ROUNDS times), on the
    fit and test frames; then the fit on every frame with both."""
    noise = NOISE0
    rec = {"rounds": []}
    lam = None
    for k in range(ROUNDS):
        best = None
        table = []
        for lm in lams:
            f = SplineField(sizes)
            fit_field(f, fd_fit, lm, noise)
            s = heldout_score(f, fd_test, noise)
            table.append({"lam": lm, **s})
            if best is None or s["loss_after"] < best[1]:
                best = (lm, s["loss_after"])
        lam = best[0]
        f = SplineField(sizes)
        fit_field(f, fd_fit, lam, noise)
        s_f, beta = estimate_noise(f, fd_fit)
        new = (max(s_f, S_FLOOR), beta)
        rec["rounds"].append({"lam": lam, "noise_in": list(noise), "noise_out": list(new), "lambda_table": table})
        log(f"    round {k + 1}: bending weight {lam:g} (held-out loss), noise model s_f {new[0]:.3f} px, "
            f"beta {new[1]:.4f}")
        noise = new
    f = SplineField(sizes)
    fit_field(f, fd_fit, lam, noise)
    rec["heldout"] = heldout_score(f, fd_test, noise)      # the fit frames' field on the others
    field = SplineField(sizes)
    rec["fit"] = fit_field(field, fd_all, lam, noise)
    rec["lam"], rec["noise"] = lam, list(noise)
    return field, rec


def point_offsets(fd: FitData, r) -> np.ndarray:
    """Per point the residual no 3-D move explains, in px: signed with two cameras (the offset across the epipolar
    curve, + along the second camera's v), its norm per degree of freedom with more."""
    out = np.full(len(fd), np.nan)
    for cams_s, idx in _groups(fd).items():
        cams_s = list(cams_s)
        s = len(cams_s)
        J = fd.J[idx][:, cams_s].reshape(len(idx), 2 * s, 3)
        v = r[idx][:, cams_s].reshape(len(idx), 2 * s)
        good = np.all(np.isfinite(J.reshape(len(idx), -1)), axis=1) & np.all(np.isfinite(v), axis=1)
        if not good.any():
            continue
        U, _S, _V = np.linalg.svd(J[good], full_matrices=True)
        Nb = U[:, :, 3:]
        z = np.einsum("nik,ni->nk", Nb, v[good])
        if s == 2:
            out[idx[good]] = np.sign(Nb[:, 3, 0]) * z[:, 0]
        else:
            out[idx[good]] = np.sqrt((z ** 2).sum(axis=1) / (2 * s - 3))
    return out


# ------------------------------------------------------------------ the field on the correction's grid
def node_correction(field: SplineField, cam: int, data_uv: np.ndarray, source: str = "") -> Tuple[Correction, dict]:
    """Camera cam's fitted field at the correction grid's nodes: the spline within COVER_PX of the data, a harmonic
    fill beyond (smooth, and no larger than the field at the data's edge). Zero without data."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from scipy.spatial import cKDTree
    w, h = field.sizes[cam]
    u0, v0, nu, nv = Correction.grid_for(w, h, NODE_STEP_PX)
    zero = Correction.zeros(w, h, NODE_STEP_PX, source=source)
    nodes = zero.nodes()
    if len(data_uv) == 0:
        return zero, {"covered_nodes": 0, "nodes": len(nodes)}
    dist, _ = cKDTree(np.asarray(data_uv, float)).query(nodes)
    cov = dist <= COVER_PX
    vals = field.eval_points(cam, nodes)
    if not cov.all():
        # Laplace's equation on the uncovered nodes, the covered ones fixed, mirrored at the grid's border
        n = nu * nv
        idx = np.arange(n).reshape(nv, nu)
        unk = np.flatnonzero(~cov)
        pos = -np.ones(n, int)
        pos[unk] = np.arange(len(unk))
        rows, cols, data = [], [], []
        rhs = np.zeros((len(unk), 2))
        for k, node in enumerate(unk):
            j, i = divmod(node, nu)
            nbrs = [(j, i - 1), (j, i + 1), (j - 1, i), (j + 1, i)]
            nbrs = [(jj, ii) for jj, ii in nbrs if 0 <= jj < nv and 0 <= ii < nu]
            rows.append(k)
            cols.append(k)
            data.append(float(len(nbrs)))
            for jj, ii in nbrs:
                m = idx[jj, ii]
                if cov[m]:
                    rhs[k] += vals[m]
                else:
                    rows.append(k)
                    cols.append(pos[m])
                    data.append(-1.0)
        A = sp.csr_matrix((data, (rows, cols)), shape=(len(unk), len(unk))).tocsc()
        for comp in (0, 1):
            vals[unk, comp] = spla.spsolve(A, rhs[:, comp])
    corr = Correction(w, h, u0, v0, NODE_STEP_PX, vals[:, 0].reshape(nv, nu), vals[:, 1].reshape(nv, nu), source)
    return corr, {"covered_nodes": int(cov.sum()), "nodes": len(nodes)}


# ------------------------------------------------------------------ tracks and frames
def read_last_tracks(track_dir: str, min_len: int) -> Dict[tuple, np.ndarray]:
    """The long tracks of a run's last ConvergeTrack files (LongTrackActive, LongTrackInactive, ExitTrack): {(kind,
    id): rows (frame, X, Y, Z)} with at least min_len points."""
    files = glob.glob(os.path.join(track_dir, "*.csv"))
    frames = [int(m.group(1)) for f in files for m in [re.search(r"_(\d+)\.csv$", f)] if m]
    if not frames:
        return {}
    last = max(frames)
    out = {}
    for kind in ("LongTrackActive", "LongTrackInactive", "ExitTrack"):
        p = os.path.join(track_dir, f"{kind}_{last}.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        rows = []
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 5:
                    try:
                        rows.append([float(x) for x in row[:5]])
                    except ValueError:
                        continue
        if not rows:
            continue
        a = np.array(rows)
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
        ids, start = np.unique(a[:, 0], return_index=True)
        ends = np.r_[start[1:], len(a)]
        for tid, s, e in zip(ids, start, ends):
            if e - s >= min_len:
                out[(kind, int(tid))] = a[s:e, 1:5]
    return out


def choose_frames(tracks: Dict[tuple, np.ndarray], sample_points: int,
                  stride: Optional[int] = None) -> Tuple[List[int], int, Dict[int, int]]:
    """Every k-th frame of the run, k the smallest stride that keeps the points near sample_points (or `stride`).
    (frames, k, points per frame)."""
    per = {}
    for t in tracks.values():
        for f in t[:, 0].astype(int):
            per[int(f)] = per.get(int(f), 0) + 1
    if not per:
        return [], 1, per
    total = sum(per.values())
    k = int(stride) if stride else max(1, int(math.floor(total / max(sample_points, 1))))
    f0, f1 = min(per), max(per)
    frames = [f for f in range(f0, f1 + 1, k) if per.get(f, 0) > 0]
    return frames, k, per


def points_at(tracks: Dict[tuple, np.ndarray], frame: int) -> np.ndarray:
    pts = []
    for t in tracks.values():
        m = t[:, 0].astype(int) == frame
        if m.any():
            pts.append(t[m, 1:4][0])
    return np.array(pts).reshape(-1, 3)


def pair_one_to_one(det_uv: np.ndarray, proj_uv: np.ndarray, r: float) -> np.ndarray:
    """Each projection's detection (index or -1): nearest pairs first within r px, each detection used once."""
    from scipy.spatial import cKDTree
    out = np.full(len(proj_uv), -1)
    ok = np.isfinite(proj_uv[:, 0])
    if not len(det_uv) or not ok.any():
        return out
    tree = cKDTree(det_uv)
    idx = np.flatnonzero(ok)
    cand = []
    for a, js in zip(idx, tree.query_ball_point(proj_uv[idx], r)):
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


# ------------------------------------------------------------------ per frame (a worker process each)
_W: dict = {}


def _worker_init(setup: dict) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    _W.clear()
    _W.update(setup)


def _lpt():
    if "lpt" not in _W:
        import pyopenlpt as lpt
        _W["lpt"] = lpt
    return _W["lpt"]


def _cameras():
    if "cams" not in _W:
        lpt = _lpt()
        _W["cams"] = [lpt.loadCameraModel(f, c or "") for f, c in zip(_W["cam_files"], _W["corr_files"])]
    return _W["cams"]


def _run_finder():
    """The tracking run's 2-D detection, set up once per worker as the run sets it up (main.cpp, IPR): the object's
    settings read from config.json, each camera's frames opened by ImageIO from the config's folder."""
    if "finder" not in _W:
        lpt = _lpt()
        settings = lpt.BasicSetting()
        settings.readConfig(_W["config_path"])
        cfg = lpt.BubbleConfig() if _W.get("obj_type") == "Bubble" else lpt.TracerConfig()
        cfg.readConfig(settings._object_config_paths[0], settings)
        root = os.path.dirname(os.path.abspath(_W["config_path"])).replace("\\", "/") + "/"
        ios = []
        for path in settings._image_file_paths:
            io = lpt.ImageIO()
            io.loadImgPath(root, path)
            ios.append(io)
        _W["obj_cfg"], _W["imgio"], _W["finder"] = cfg, ios, lpt.ObjectFinder2D()
    return _W["finder"], _W["obj_cfg"], _W["imgio"]


def _detect(c: int, frame: int) -> np.ndarray:
    """Detections (n, 4): u, v and the line's box lengths Lx, Ly (0 for spots), as the tracking run finds them on the
    frame: ImageIO's image, ObjectFinder2D.findObject2D with the camera's index (which picks its line-detector
    noise)."""
    finder, cfg, ios = _run_finder()
    if frame >= len(ios[c].getImgPath()):
        return np.zeros((0, 4))
    objs = finder.findObject2D(ios[c].loadImg(frame), cfg, c)
    out = np.zeros((len(objs), 4))
    for i, o in enumerate(objs):
        out[i, 0], out[i, 1] = o._pt_center[0], o._pt_center[1]
        if getattr(o, "_has_shape", False):
            out[i, 2], out[i, 3] = o._len_h_px, o._len_v_px
    return out


def _project_jacobian(cam, X: np.ndarray):
    """Projections (n, 2) and Jacobians (n, 2, 3) by central differences from each point's own hint."""
    lpt = _lpt()
    h = JAC_STEP_MM
    P = np.full((len(X), 2), np.nan)
    J = np.full((len(X), 2, 3), np.nan)
    for i, x in enumerate(X):
        ok, p, hint, _e = cam.projectWithHint(lpt.Pt3D(*map(float, x)), lpt.ProjectionHint())
        if not ok:
            continue
        P[i] = (p[0], p[1])
        for a in range(3):
            xp, xm = x.copy(), x.copy()
            xp[a] += h
            xm[a] -= h
            ok1, p1, _h1, _e1 = cam.projectWithHint(lpt.Pt3D(*map(float, xp)), hint)
            ok2, p2, _h2, _e2 = cam.projectWithHint(lpt.Pt3D(*map(float, xm)), hint)
            if not (ok1 and ok2):
                J[i] = np.nan
                break
            J[i, 0, a] = (p1[0] - p2[0]) / (2 * h)
            J[i, 1, a] = (p1[1] - p2[1]) / (2 * h)
    return P, J


def _worker_frame(job: Tuple[int, np.ndarray]) -> dict:
    frame, X = job
    cams = _cameras()
    n_cam = len(cams)
    n = len(X)
    out = {"frame": frame, "X": X, "P": np.full((n, n_cam, 2), np.nan), "J": np.full((n, n_cam, 2, 3), np.nan),
           "d": np.full((n, n_cam, 2), np.nan), "L": np.zeros((n, n_cam, 2)), "dist": np.full((n, n_cam), np.nan),
           "n_det": [0] * n_cam}
    for c in range(n_cam):
        det = _detect(c, frame)
        out["n_det"][c] = len(det)
        if not n:
            continue
        P, J = _project_jacobian(cams[c], X)
        out["P"][:, c], out["J"][:, c] = P, J
        pr = pair_one_to_one(det[:, :2], P, _W["pair_radius"][c])
        ok = pr >= 0
        out["d"][ok, c] = det[pr[ok], :2]
        out["L"][ok, c] = det[pr[ok], 2:4]
        out["dist"][ok, c] = np.hypot(*(det[pr[ok], :2] - P[ok]).T)
    return out


# ------------------------------------------------------------------ the VSC
class FieldVSC:
    """The field VSC of a project (config.json's cameras, the run's tracks in its output folder)."""

    def __init__(self, proj_dir: str, log: Callable[[str], None], min_track_len: int = 15,
                 sample_points: int = 20000, min_valid_points: int = 2000, tolerance_mode: str = "default",
                 tolerance_value: float = 3.0, workers: Optional[int] = None, frame_stride: Optional[int] = None,
                 lams=LAMS):
        self.proj_dir = proj_dir
        self.log = log
        self.min_track_len = int(min_track_len)
        self.sample_points = int(sample_points)
        self.frame_stride = frame_stride
        self.lams = tuple(lams)
        self.min_valid_points = int(min_valid_points)
        self.tolerance_mode = tolerance_mode
        self.tolerance_value = float(tolerance_value)
        self.workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))

    # -------------------------------------------------------------- the project
    def _setup(self) -> Tuple[bool, str]:
        import pyopenlpt as lpt
        if not hasattr(lpt, "ImageCorrection"):
            return False, ("this pyopenlpt has no image corrections (lpt.ImageCorrection): rebuild the library "
                           "(build_local.py) first, or tracking would ignore the correction")
        if "cam_id" not in (lpt.ObjectFinder2D.findObject2D.__doc__ or ""):
            return False, ("this pyopenlpt's findObject2D takes no camera index, so it cannot detect as the tracking "
                           "run does: rebuild the library (build_local.py) first")
        self.lpt = lpt
        self.config_path = stb_config.find_config(self.proj_dir)
        if self.config_path is None or not self.config_path.lower().endswith(".json"):
            return False, ("VSC for tube / sphere cameras writes each camera's correction into config.json, and this "
                           "project has only config.txt: convert it first (python -m modules.stb_config convert "
                           "config.txt writes config.json beside it, with every setting)")
        cfg = stb_config.read_json_raw(self.config_path)
        self.cfg = cfg
        cams = cfg.get("cameras", [])
        if not cams:
            return False, "config.json has no cameras"
        self.cam_files = [stb_config.resolve(self.config_path, c["file"]) for c in cams]
        self.corr_files = [stb_config.resolve(self.config_path, c["correction"]) if c.get("correction") else ""
                           for c in cams]
        self.image_lists = [stb_config.resolve(self.config_path, p) for p in cfg.get("image_lists", [])]
        if len(self.image_lists) != len(cams):
            return False, "config.json needs one image list per camera"
        # camera numbers as the other VSC names its files: the number in the camera file's name, else the position
        self.cam_ids = []
        for i, f in enumerate(self.cam_files):
            m = re.search(r"cam(\d+)", os.path.basename(f))
            self.cam_ids.append(int(m.group(1)) if m else i)
        if len(set(self.cam_ids)) != len(self.cam_ids):
            self.cam_ids = list(range(len(self.cam_files)))
        self.cams = []
        for f, c in zip(self.cam_files, self.corr_files):
            try:
                self.cams.append(lpt.loadCameraModel(f, c))
            except RuntimeError as e:
                return False, f"cannot load camera {f}" + (f" with its correction {c}" if c else "") + f": {e}"
        self.sizes = [(cam.getNCol(), cam.getNRow()) for cam in self.cams]
        obj = (cfg.get("objects") or [{}])[0]
        tracer = obj.get("tracer", {})
        self.obj = obj
        self.detector = str(tracer.get("detector", "peaks")).lower()
        # the line detector's noise as the run takes it (ObjectFinder2D): one value for every camera, else the
        # camera's; none (or not > 0): estimated from each frame
        noise = (tracer.get("lines") or {}).get("noise")
        n_cam = len(cams)
        values = [noise] * n_cam if isinstance(noise, (int, float)) else list(noise or [])
        if isinstance(noise, list) and len(noise) == 1:
            values = noise * n_cam
        self.noise = [float(values[c]) if c < len(values) and float(values[c]) > 0 else None for c in range(n_cam)]
        r_px = float(tracer.get("radius_px", 2.0))
        if self.tolerance_mode == "custom":
            self.pair_radius = [self.tolerance_value] * n_cam
        elif self.tolerance_mode == "reproj":
            self.pair_radius = []
            for f in self.cam_files:
                err = _calib_error(f)
                if err is None:
                    return False, f"{f} has no calibration error for the Reprojection Error tolerance mode"
                self.pair_radius.append(err[0] + 3.0 * err[1])
        else:
            self.pair_radius = [r_px + 1.0] * n_cam
        out = cfg.get("output_dir", "Results")
        self.output_dir = stb_config.resolve(self.config_path, out)
        self.track_dir = os.path.join(self.output_dir, "ConvergeTrack")
        return True, ""

    def _guard(self) -> Tuple[bool, str]:
        """A pass fits the tracks of a run made with the cameras as they are now: tracks older than a camera's
        correction were found without it."""
        files = glob.glob(os.path.join(self.track_dir, "*.csv"))
        if not files:
            return False, f"no tracks in {self.track_dir}: run tracking first"
        newest = max(os.path.getmtime(f) for f in files)
        for c in self.corr_files:
            if c and os.path.exists(c) and os.path.getmtime(c) > newest:
                return False, (f"the tracks in {self.track_dir} are older than the correction {os.path.basename(c)}: "
                               "rerun tracking with it first (each pass fits the tracks found with the last one)")
        return True, ""

    # -------------------------------------------------------------- the run
    def run(self) -> Tuple[bool, str, dict]:
        t0 = time.time()
        ok, msg = self._setup()
        if not ok:
            return False, msg, {}
        log = self.log
        n_cam = len(self.cams)
        types = [str(c.type()).split(".")[-1] for c in self.cams]
        log(f"  Field VSC (image corrections) for {n_cam} cameras: {', '.join(types)}")
        for cid, f, c, (w, h) in zip(self.cam_ids, self.cam_files, self.corr_files, self.sizes):
            log(f"    cam{cid}: {f} ({w} x {h})" + (f", corrected by {c}" if c else ", no correction yet"))
        ok, msg = self._guard()
        if not ok:
            return False, msg, {}
        tracks = read_last_tracks(self.track_dir, self.min_track_len)
        if not tracks:
            return False, f"no tracks of at least {self.min_track_len} points in {self.track_dir}", {}
        frames, k, per = choose_frames(tracks, self.sample_points, self.frame_stride)
        n_pts = sum(per[f] for f in frames)
        log(f"\n[Step 2] Tracks: {len(tracks)} long tracks of >= {self.min_track_len} points; one frame in {k}: "
            f"{len(frames)} frames, {n_pts} track points")
        noise_txt = ", ".join("from each frame" if x is None else f"{x:.2f}" for x in self.noise)
        log(f"  detector: the tracking run's ({self.detector}"
            + (f"; noise {noise_txt}" if self.detector == "lines" else "") + ")"
            + f"; pairing within {', '.join(f'{r:.2f}' for r in self.pair_radius)} px")
        # per frame, in worker processes
        setup = {"cam_files": self.cam_files, "corr_files": self.corr_files, "pair_radius": self.pair_radius,
                 "config_path": self.config_path, "obj_type": self.obj.get("type", "Tracer")}
        jobs = [(f, points_at(tracks, f)) for f in frames]
        log(f"\n[Step 3] Detections, projections and Jacobians on {len(jobs)} frames ({self.workers} processes)...")
        recs = self._map(setup, jobs)
        fd, used = self._fitdata(recs)
        counts = [int(fd.mask[:, c].sum()) for c in range(n_cam)]
        log(f"  {len(fd)} points paired in at least two cameras; per camera {counts}")
        if min(counts) < self.min_valid_points:
            return False, (f"only {min(counts)} paired points in a camera, need {self.min_valid_points} (Min Valid "
                           "Points)"), {}
        # the fit
        log("\n[Step 4] Fitting the correction (projected least squares; held-out frames choose the bending)...")
        sel = np.isin(fd.frame, frames[::2])
        field, rec = fit_with_noise(self.sizes, fd.subset(sel), fd.subset(~sel), fd, log, lams=self.lams)
        c_pts = np.zeros_like(fd.r)
        for c in range(n_cam):
            m = fd.mask[:, c]
            c_pts[m, c] = field.eval_points(c, fd.d[m, c])
        off0 = point_offsets(fd, fd.r)
        off1 = point_offsets(fd, fd.r - c_pts)
        rsd = lambda x: float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))  # noqa: E731
        log(f"  bending weight {rec['lam']:g}, noise s_f {rec['noise'][0]:.3f} px, beta {rec['noise'][1]:.4f}; "
            f"{100 * (rec['fit']['downweighted'] or 0):.1f} % of the points down-weighted")
        log(f"  the residual no 3-D move explains: median {np.nanmedian(off0):+.3f} -> {np.nanmedian(off1):+.3f} px, "
            f"robust SD {rsd(off0):.3f} -> {rsd(off1):.3f} px")
        # the files and the config
        log("\n[Step 5] Writing the corrections and config.json...")
        res = self._write(field, fd, rec, off0, off1, log)
        res.update({"tracks": len(tracks), "frames": frames, "stride": k, "points": int(len(fd)),
                    "points_per_camera": counts, "fit": rec, "seconds": round(time.time() - t0, 1)})
        data = self._gui_data(recs, used, fd, field)
        data["field_vsc"] = res
        return True, res["message"], data

    def _map(self, setup: dict, jobs) -> List[dict]:
        if self.workers <= 1:
            _worker_init(setup)
            return [_worker_frame(j) for j in jobs]
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx, initializer=_worker_init,
                                 initargs=(setup,)) as ex:
            return list(ex.map(_worker_frame, jobs))

    def _fitdata(self, recs: List[dict]) -> Tuple[FitData, list]:
        """The points with a detection, a projection and a Jacobian in at least two cameras."""
        parts, used = [], []
        for g in recs:
            n = len(g["X"])
            if not n:
                continue
            mask = (np.isfinite(g["d"][:, :, 0]) & np.isfinite(g["P"][:, :, 0])
                    & np.all(np.isfinite(g["J"].reshape(n, g["J"].shape[1], -1)), axis=2))
            keep = mask.sum(axis=1) >= 2
            r = g["d"] - g["P"]
            parts.append(FitData(mask[keep], g["d"][keep], r[keep], g["J"][keep], g["L"][keep],
                                 np.full(int(keep.sum()), int(g["frame"]))))
            used.append((g, keep))
        return FitData.concat(parts), used

    def _write(self, field: SplineField, fd: FitData, rec: dict, off0, off1, log) -> dict:
        cam_dir = os.path.join(self.proj_dir, "camFile_VSC")
        os.makedirs(cam_dir, exist_ok=True)
        cfg = stb_config.read_json_raw(self.config_path)
        stamp = time.strftime("%Y-%m-%d %H:%M")
        per_cam = []
        for c, cam_entry in enumerate(cfg["cameras"]):
            data_uv = fd.d[fd.mask[:, c], c]
            inc, cover = node_correction(field, c, data_uv)
            prev = None
            n_pass = 1
            if self.corr_files[c]:
                prev = Correction.read(self.corr_files[c])
                n_pass = _pass_of(self.corr_files[c]) + 1
            total = compose(prev, inc)
            name = f"vsc_cam{self.cam_ids[c]}_corr_p{n_pass}.txt"
            path = os.path.join(cam_dir, name)
            covered = _covered_nodes(total, data_uv)
            mag_inc = np.hypot(inc.du, inc.dv).ravel()[covered]
            mag_tot = np.hypot(total.du, total.dv).ravel()[covered]
            stats = {"camera": self.cam_ids[c], "file": path, "pass": n_pass, "points": int(len(data_uv)),
                     "covered_nodes": cover["covered_nodes"], "nodes": cover["nodes"],
                     "change_rms_px": float(np.sqrt(np.mean(mag_inc ** 2))) if covered.any() else 0.0,
                     "change_max_px": float(mag_inc.max()) if covered.any() else 0.0,
                     "total_rms_px": float(np.sqrt(np.mean(mag_tot ** 2))) if covered.any() else 0.0,
                     "total_max_px": float(mag_tot.max()) if covered.any() else 0.0,
                     "slope_bound": total.slope_bound(), "previous": self.corr_files[c] or None}
            total.write(path, [f"field VSC pass {n_pass}, {stamp}: from the tracks in {self.track_dir}",
                               f"{len(data_uv)} paired points; this pass changed it by {stats['change_rms_px']:.3f} px "
                               f"RMS ({stats['change_max_px']:.3f} max) over the {cover['covered_nodes']} nodes near "
                               f"the data",
                               f"bending weight {rec['lam']:g}, noise s_f {rec['noise'][0]:.3f} px, beta "
                               f"{rec['noise'][1]:.4f}"] + ([f"previous pass: {self.corr_files[c]}"] if prev else []))
            cam_entry["correction"] = os.path.relpath(path, os.path.dirname(os.path.abspath(self.config_path))) \
                .replace(os.sep, "/")
            per_cam.append(stats)
            log(f"  cam{self.cam_ids[c]}: pass {n_pass}, {stats['points']} points; the correction now "
                f"{stats['total_rms_px']:.3f} px RMS ({stats['total_max_px']:.3f} max) near the data, this pass "
                f"changed it by {stats['change_rms_px']:.3f} px RMS -> camFile_VSC/{name}")
        _backup_once(self.config_path, self.proj_dir, log)
        stb_config.write_json_raw(cfg, self.config_path)
        log("  config.json: every camera's \"correction\" points at its new file")
        tol = (self.obj.get("ipr") or {}).get("tol_2d_px")
        rsd = lambda x: float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))  # noqa: E731
        n_pass = max(s["pass"] for s in per_cam)
        # the tracking settings page keeps a tube project's tracer settings when it saves (the line detector's, the
        # OTF table's, the depth cue's), so the tolerance can be set there
        where = "in the tracking settings (or objects[0].ipr.tol_2d_px in config.json)"
        advice = (f"The 2-D tolerance is left at {tol} px. On the tube run T1 the correction paid at 1.0 px "
                  f"(two passes: +11 % real long tracks, a fifth fewer ghosts) and changed nothing at 1.5 px, so "
                  f"consider 1.0 px, set {where}." if tol is not None else
                  "The 2-D tolerance is left as it is; on T1 the correction paid at 1.0 px.")
        nxt = ("Rerun tracking with the corrected cameras, then VSC again for a second pass (two or three passes)."
               if n_pass < 2 else "Rerun tracking; another pass helps while a pass still changes the field.")
        message = (f"Field VSC pass {n_pass}: corrections written to camFile_VSC/ and set in config.json. The "
                   f"residual no 3-D move explains: robust SD {rsd(off0):.3f} -> {rsd(off1):.3f} px. {advice} {nxt}")
        log(f"  {advice}")
        log(f"  {nxt}")
        return {"cameras": per_cam, "pass": n_pass, "offset_before": {"median": float(np.nanmedian(off0)),
                "robust_sd": rsd(off0)}, "offset_after": {"median": float(np.nanmedian(off1)), "robust_sd": rsd(off1)},
                "tol_2d_px": tol, "message": message}

    def _gui_data(self, recs, used, fd: FitData, field: SplineField) -> dict:
        """The other VSC's result layout, for the tracking view: the points (pt3d, detections '2d_per_cam',
        projections before this pass '2d_csv_per_cam', after it '2d_corrected_per_cam'; keyed by the camera's place
        in config.json, as the view numbers its cameras), frame and id."""
        lpt = self.lpt
        cfg = stb_config.read_json_raw(self.config_path)
        new_cams = [lpt.loadCameraModel(f, stb_config.resolve(self.config_path, c["correction"]))
                    for f, c in zip(self.cam_files, cfg["cameras"])]
        pts = []
        cid = 0
        for g, keep in used:
            for i in np.flatnonzero(keep):
                X = g["X"][i]
                d2, p0, p1 = {}, {}, {}
                for c in range(len(self.cams)):
                    if not np.isfinite(g["d"][i, c, 0]):
                        continue
                    k = c
                    d2[k] = (float(g["d"][i, c, 0]), float(g["d"][i, c, 1]))
                    p0[k] = (float(g["P"][i, c, 0]), float(g["P"][i, c, 1]))
                    ok, q, _err = _project_status(new_cams[c], lpt, X)
                    if ok:
                        p1[k] = q
                pts.append({"pt3d": [float(x) for x in X], "2d_per_cam": d2, "2d_csv_per_cam": p0,
                            "2d_corrected_per_cam": p1, "frame_id": int(g["frame"]), "corr_id": cid})
                cid += 1
        return {"mode": "field", "valid_points": pts, "cameras_init": {}, "cameras_optim": {},
                "overlay_points_optim": {}}


def _project_status(cam, lpt, X):
    try:
        p = cam.project(lpt.Pt3D(*map(float, X)))
        return True, (float(p[0]), float(p[1])), ""
    except RuntimeError as e:
        return False, None, str(e)


def _covered_nodes(corr: Correction, data_uv) -> np.ndarray:
    from scipy.spatial import cKDTree
    nodes = corr.nodes()
    if len(data_uv) == 0:
        return np.zeros(len(nodes), bool)
    dist, _ = cKDTree(np.asarray(data_uv, float)).query(nodes)
    return dist <= COVER_PX


def _pass_of(path: str) -> int:
    """The pass a correction file records ("field VSC pass N"); 1 for one without."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                m = re.search(r"field VSC pass (\d+)", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return 1


def _calib_error(cam_file: str):
    """(mean, std) of a camera file's "Camera Calibration Error", or None."""
    try:
        with open(cam_file, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh]
        k = next(i for i, ln in enumerate(lines) if ln.startswith("# Camera Calibration Error"))
        a, b = (float(x) for x in lines[k + 1].split(",")[:2])
        return a, b
    except (OSError, StopIteration, ValueError):
        return None


def _backup_once(config_path: str, proj_dir: str, log) -> None:
    import shutil
    backup = os.path.join(proj_dir, "config_backup.json")
    if not os.path.exists(backup):
        shutil.copy(config_path, backup)
        log("  Backup saved to config_backup.json")


def is_field_project(proj_dir: str) -> bool:
    """True when the project's config (config.json, else config.txt) has a camera behind a curved window (PINCYL,
    PINSPH): the field VSC's (which then needs config.json, and says so)."""
    path = stb_config.find_config(proj_dir)
    if path is None:
        return False
    try:
        cfg = stb_config.read(path, inline_objects=False)
    except Exception:  # noqa: BLE001
        return False
    for c in cfg.get("cameras", []):
        f = stb_config.resolve(path, c.get("file", ""))
        if _model_token(f) in CURVED:
            return True
    return False


def _model_token(cam_file: str) -> str:
    try:
        with open(cam_file, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    return s.upper()
    except OSError:
        pass
    return ""
