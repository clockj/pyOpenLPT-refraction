"""Track predictors compared in Phase E of the stb_improve campaign (plan section "Phase E: the detailed steps").

Tracks are stored flat: P (N, 3) holds every point, track after track; a track's points are P[start:start+length].
A prediction for the point at flat index g uses only the points of its track before g; n is how many there are.

  wiener_one(hist)          today's STB::predictNext (src/srcSTB/STB.cpp) on one history, statement by statement
  wiener(P, g, n)           the same, vectorised over many points
  mean_step(P, g, n)        its limit far from the origin: y_m + (y_m - y_0)/m, m = min(n - 1, 5)
  poly(P, g, n, deg, k)     least-squares polynomial of degree deg through the last min(n, k) points, one frame ahead
  Kalman(kind, lam)         per-axis Kalman filter, "cv" (constant velocity) or "ca" (constant acceleration), from a
                            diffuse start, with q/r = lam (from a diffuse start nothing else matters):
                              .run(tr)       one-step predictions of every point of a Tracks set, recursively
                              .weights(nmax) the same predictions as fixed weights on the points before
                              .fir(P, g, n, N) the filter over the last N points only (a window)
  load_run(name)            the saved long tracks of a run (cached in runs/e1_cache/)

Units: mm and frames.  Process noise: "cv" has a white acceleration per frame (G = [1/2, 1]), "ca" a white change of
acceleration per frame (G = [1/2, 1, 1], Bar-Shalom's discrete Wiener process acceleration); q is its variance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "runs", "e1_cache")
X_RUNS = r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results\10Hz_r_b_1\stb_new_calib"
RUN_FILES = {   # name: (ConvergeTrack folder, save frame, description)
    "b300": (os.path.join(HERE, "runs", "b300_final3", "Results", "ConvergeTrack"), 300,
             "B300: new camFiles, frames 0-300, this campaign's final code"),
    "t4": (os.path.join(X_RUNS, "new_cams", "Results", "ConvergeTrack"), 500,
           "T4: new camFiles, frames 0-500, the code before this campaign"),
    "t5": (os.path.join(X_RUNS, "old_cams", "Results", "ConvergeTrack"), 500,
           "T5: old camFiles, frames 0-500, the code before this campaign"),
}
KINDS = ("LongTrackActive", "ExitTrack", "LongTrackInactive")   # kind codes 0, 1, 2

# constants of src/srcSTB/STB.cpp / inc/libMath/STBCommons.h
SMALLNUMBER = 1e-8
WIENER_MAX_ITER = 5000
K_SHIFT = 10.0


# ----------------------------------------------------------------------------------------------- today's predictor
def wiener_one(hist):
    """STB::predictNext for one history (oldest first, each a 3-vector); None with fewer than 3 points.

    Statement by statement as the C++, in the same order of operations, so the results agree to the last bit."""
    n = len(hist)
    if n < 3:
        return None
    order = n - 1 if n < 6 else 5
    start = n - 1 - order
    out = [0.0, 0.0, 0.0]
    for axis in range(3):
        series = [float(hist[start + j][axis]) for j in range(order + 1)]
        shifted = False
        if abs(series[order]) < 1.0:
            shifted = True
            series = [v + K_SHIFT for v in series]
        denom = 0.0
        for j in range(order):
            denom += series[j] * series[j]
        step = (1.0 / denom) if denom > 1e-12 else 0.0
        filt = [0.0] * order
        prediction = 0.0
        error = series[order] - prediction
        it = 0
        while abs(error) > SMALLNUMBER and it < WIENER_MAX_ITER and step > 0.0:
            prediction = 0.0
            for j in range(order):
                filt[j] += step * series[j] * error
                prediction += filt[j] * series[j]
            error = series[order] - prediction
            it += 1
        if order >= 1 and step == 0.0:
            prediction = series[order] + (series[order] - series[order - 1])
        else:
            prediction = 0.0
            for j in range(order):
                prediction += filt[j] * series[j + 1]
        if shifted:
            prediction -= K_SHIFT
        out[axis] = prediction
    return out


def _wiener_windows(S):
    """wiener_one on windows S (k, order+1, 3) of one order, vectorised with the same operations."""
    S = np.array(S, dtype=float)
    m = S.shape[1] - 1
    shifted = np.abs(S[:, m, :]) < 1.0
    S = np.where(shifted[:, None, :], S + K_SHIFT, S)
    denom = np.zeros(S[:, 0, :].shape)
    for j in range(m):
        denom = denom + S[:, j] * S[:, j]
    ok = denom > 1e-12
    step = np.zeros_like(denom)
    np.divide(1.0, denom, out=step, where=ok)
    filt = np.zeros((S.shape[0], m, 3))
    pred = np.zeros_like(denom)
    err = S[:, m] - pred
    active = (np.abs(err) > SMALLNUMBER) & (step > 0.0)
    it = 0
    while active.any() and it < WIENER_MAX_ITER:
        newpred = np.zeros_like(pred)
        for j in range(m):
            filt[:, j] = np.where(active, filt[:, j] + step * S[:, j] * err, filt[:, j])
            newpred = newpred + filt[:, j] * S[:, j]
        pred = np.where(active, newpred, pred)
        err = np.where(active, S[:, m] - pred, err)
        it += 1
        active &= np.abs(err) > SMALLNUMBER
    fwd = np.zeros_like(denom)
    for j in range(m):
        fwd = fwd + filt[:, j] * S[:, j + 1]
    fallback = S[:, m] + (S[:, m] - S[:, m - 1])
    out = np.where(step == 0.0, fallback, fwd)
    return np.where(shifted, out - K_SHIFT, out)


def wiener(P, g, n):
    """Today's prediction of the points at flat indices g from their n points before (NaN where n < 3)."""
    g, n = np.asarray(g), np.asarray(n)
    out = np.full((len(g), 3), np.nan)
    order = np.minimum(n - 1, 5)
    for m in (2, 3, 4, 5):
        sel = np.nonzero((order == m) & (n >= 3))[0]
        if len(sel):
            idx = g[sel, None] - (m + 1) + np.arange(m + 1)[None, :]
            out[sel] = _wiener_windows(P[idx])
    return out


def mean_step(P, g, n):
    """W's limit far from the origin: the last point plus the mean step over the last m = min(n-1, 5) frames."""
    g, n = np.asarray(g), np.asarray(n)
    out = np.full((len(g), 3), np.nan)
    m = np.minimum(n - 1, 5)
    ok = n >= 2
    last = P[g[ok] - 1]
    first = P[g[ok] - 1 - m[ok]]
    out[ok] = last + (last - first) / m[ok, None]
    return out


# ------------------------------------------------------------------------------------ least-squares polynomials
def poly_weights(deg, k):
    """Weights w (k,) with prediction = w @ y for y at t = 0..k-1, extrapolated to t = k (least squares, degree deg)."""
    t = np.arange(k, dtype=float) - (k - 1) / 2.0          # centred for conditioning
    V = np.vander(t, deg + 1, increasing=True)
    e = np.vander(np.array([k - (k - 1) / 2.0]), deg + 1, increasing=True)[0]
    return e @ np.linalg.solve(V.T @ V, V.T)


def poly(P, g, n, deg, k):
    """Least-squares polynomial (degree deg) through the last min(n, k) points, one frame ahead (NaN if too few)."""
    g, n = np.asarray(g), np.asarray(n)
    out = np.full((len(g), 3), np.nan)
    keff = np.minimum(n, k)
    for kk in np.unique(keff):
        if kk < deg + 1:
            continue
        sel = np.nonzero(keff == kk)[0]
        w = poly_weights(deg, int(kk))
        idx = g[sel, None] - kk + np.arange(kk)[None, :]
        out[sel] = np.einsum("j,ijk->ik", w, P[idx])
    return out


# ------------------------------------------------------------------------------------------------ Kalman filters
class Kalman:
    """Per-axis Kalman filter with a diffuse start; the three axes are filtered independently with the same q/r.

    The predictions depend only on lam = q/r (Q, R and the diffuse prior scale together), so r = 1 internally and the
    innovation variances come out in units of r."""

    def __init__(self, kind, lam):
        self.kind, self.lam = kind, float(lam)
        if kind == "cv":
            self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
            self.G = np.array([0.5, 1.0])
        elif kind == "ca":
            self.F = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
            self.G = np.array([0.5, 1.0, 1.0])
        else:
            raise ValueError(kind)
        self.d = self.F.shape[0]
        self.Q = self.lam * np.outer(self.G, self.G)
        self._cache = None

    def _start(self):
        """The diffuse start, exactly: the information filter over the first d measurements.

        Returns A (d, d), the filtered state after measurement d-1 as A @ y[0:d], and its covariance (units of r)."""
        d, F, G, q = self.d, self.F, self.G, self.lam
        Finv = np.linalg.inv(F)
        Y = np.zeros((d, d))          # information matrix
        B = np.zeros((d, d))          # information vector = B @ y[0:d]
        for j in range(d):
            Y[0, 0] += 1.0            # H^T H / r
            B[0, j] += 1.0            # H^T y_j / r
            if j == d - 1:
                break
            M = Finv.T @ Y @ Finv
            if q > 0:
                MG = M @ G
                Om = np.outer(MG, MG) / (G @ MG + 1.0 / q)
                Y = M - Om
                B = (np.eye(d) - np.outer(MG, G) / (G @ MG + 1.0 / q)) @ Finv.T @ B
            else:
                Y = M
                B = Finv.T @ B
        Pf = np.linalg.inv(Y)
        return Pf @ B, Pf

    def gains(self, L):
        """K[j] (gain of measurement j, j >= d), Ppred[j] (covariance before measurement j), S[j] (innovation variance
        of measurement j), all in units of r, for j < L; rows below d are NaN."""
        if self._cache is not None and self._cache[0] >= L:
            return self._cache[1]
        d, F, Q = self.d, self.F, self.Q
        A, Pf = self._start()
        K = np.full((L, d), np.nan)
        Pp = np.full((L, d, d), np.nan)
        S = np.full(L, np.nan)
        P = F @ Pf @ F.T + Q
        H = np.zeros(d)
        H[0] = 1.0
        for j in range(d, L):
            Pp[j] = P
            S[j] = P[0, 0] + 1.0
            k = P[:, 0] / S[j]
            K[j] = k
            IKH = np.eye(d) - np.outer(k, H)
            Pu = IKH @ P @ IKH.T + np.outer(k, k)          # Joseph form
            P = F @ Pu @ F.T + Q
            P = 0.5 * (P + P.T)
        out = (A, K, Pp, S)
        self._cache = (L, out)
        return out

    def run(self, tr):
        """One-step predictions (N, 3) of every point of a Tracks set from all points before it in its track (NaN for
        the first d points of each track), and each point's innovation variance in units of r (NaN likewise)."""
        P, starts, lengths = tr.P, tr.starts, tr.lengths
        order = np.argsort(-lengths, kind="stable")
        st, ln = starts[order], lengths[order]
        L = int(ln.max()) if len(ln) else 0
        d = self.d
        A, K, _, S = self.gains(max(L, d + 1))
        pred = np.full(P.shape, np.nan)
        svar = np.full(len(P), np.nan)
        n_act = np.searchsorted(-ln, -np.arange(L + 1), side="left")   # tracks longer than j (ln is descending)
        na = n_act[d - 1]                                   # tracks with at least d points
        if na == 0:
            return pred, svar
        first = np.stack([P[st[:na] + j] for j in range(d)], axis=1)        # (na, d, 3)
        x = np.einsum("ab,nbk->nka", A, first)                              # (na, 3, d) filtered after d-1
        x = x @ self.F.T                                                    # predicted for measurement d
        for j in range(d, L):
            na = n_act[j]
            if na == 0:
                break
            x = x[:na]
            idx = st[:na] + j
            pred[idx] = x[:, :, 0]
            svar[idx] = S[j]
            innov = P[idx] - x[:, :, 0]
            x = (x + K[j][None, None, :] * innov[:, :, None]) @ self.F.T
        return pred, svar

    def weights(self, nmax):
        """w[n] (n,) for n = d..nmax: the prediction of the next point from n points as w[n] @ y[0:n]."""
        d, F = self.d, self.F
        A, K, _, _ = self.gains(nmax + 1)
        H = np.zeros(d)
        H[0] = 1.0
        B = [A[:, i].copy() for i in range(d)]      # filtered state after n points = sum_i B[i] y_i
        w = {}
        for n in range(d, nmax + 1):
            w[n] = np.array([H @ F @ b for b in B])
            if n == nmax:
                break
            j = n                                   # the next measurement's index
            k = K[j]
            T = (np.eye(d) - np.outer(k, H)) @ F
            B = [T @ b for b in B] + [k.copy()]
        return w

    def fir(self, P, g, n, N):
        """The filter over the last min(n, N) points only, from a diffuse start at the window's first point."""
        g, n = np.asarray(g), np.asarray(n)
        out = np.full((len(g), 3), np.nan)
        w = self.weights(N)
        keff = np.minimum(n, N)
        for kk in np.unique(keff):
            if kk < self.d:
                continue
            sel = np.nonzero(keff == kk)[0]
            idx = g[sel, None] - kk + np.arange(kk)[None, :]
            out[sel] = np.einsum("j,ijk->ik", w[int(kk)], P[idx])
        return out


# ------------------------------------------------------------------------------------------------------ the data
@dataclass
class Tracks:
    P: np.ndarray          # (N, 3) mm
    starts: np.ndarray     # (T,)
    lengths: np.ndarray    # (T,)
    kind: np.ndarray       # (T,) 0 active at the save, 1 left the volume, 2 lost
    tid: np.ndarray        # (T,) the TrackID in its file
    frame0: np.ndarray     # (T,) first frame
    name: str = ""

    @property
    def index_in_track(self):
        return np.arange(len(self.P)) - np.repeat(self.starts, self.lengths)

    @property
    def track_of_point(self):
        return np.repeat(np.arange(len(self.starts)), self.lengths)


def _read_csv(path):
    import pandas as pd
    d = pd.read_csv(path, usecols=[0, 1, 2, 3, 4], dtype={0: np.int64, 1: np.int64, 2: np.float64, 3: np.float64,
                                                          4: np.float64})
    return d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy(), d.iloc[:, 2:5].to_numpy()


def load_run(name, refresh=False):
    """The saved long tracks of a run, checked (each track's rows together, frames consecutive) and cached."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(cache) and not refresh:
        z = np.load(cache)
        return Tracks(z["P"], z["starts"], z["lengths"], z["kind"], z["tid"], z["frame0"], name)
    folder, save, _ = RUN_FILES[name]
    Ps, st, ln, kd, td, f0 = [], [], [], [], [], []
    offset = 0
    for code, kind in enumerate(KINDS):
        tid, frame, xyz = _read_csv(os.path.join(folder, f"{kind}_{save}.csv"))
        if len(tid) == 0:
            continue
        brk = np.nonzero(np.diff(tid) != 0)[0] + 1
        s = np.concatenate([[0], brk])
        l = np.diff(np.concatenate([s, [len(tid)]]))
        if len(np.unique(tid[s])) != len(s):
            raise ValueError(f"{name} {kind}: a track's rows are not together")
        same = np.ones(len(tid) - 1, bool)
        same[brk - 1] = False
        if not np.all(np.diff(frame)[same] == 1):
            raise ValueError(f"{name} {kind}: frames not consecutive within a track")
        Ps.append(xyz)
        st.append(s + offset)
        ln.append(l)
        kd.append(np.full(len(s), code, np.int8))
        td.append(tid[s])
        f0.append(frame[s])
        offset += len(tid)
    tr = Tracks(np.concatenate(Ps), np.concatenate(st), np.concatenate(ln), np.concatenate(kd),
                np.concatenate(td), np.concatenate(f0), name)
    np.savez(cache, P=tr.P, starts=tr.starts, lengths=tr.lengths, kind=tr.kind, tid=tr.tid, frame0=tr.frame0)
    return tr


def linear_fit_residual(P, g):
    """STB::checkLinearFit's residual of the point g: distance from the least-squares line through it and the 3 points
    before, evaluated at g (weights -0.2, 0.1, 0.4, 0.7)."""
    w = np.array([-0.2, 0.1, 0.4, 0.7])
    idx = np.asarray(g)[:, None] - 3 + np.arange(4)[None, :]
    est = np.einsum("j,ijk->ik", w, P[idx])
    return np.linalg.norm(P[g] - est, axis=1)
