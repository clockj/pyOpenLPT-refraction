"""S2 (plan, Phase 6; L3): the shape model of T1's particle images, fitted from T1 itself.

    python tests/s2_shape_fit.py [--band 8] [--reuse]

The model (tests/cylinder_stb/shape_model.py): per camera a pupil of diameter A focused at depth z_f; a point X images
as a uniform ellipse of full axes L_h = A f |1/(z_tan + delta) - 1/z_f| and L_v = A f |1/(z_sag + delta) - 1/z_f|
blurred by sigma0, z_tan and z_sag from the calibrated PINCYL camera (the parallax of a displaced pinhole), delta a
depth offset (model B; 0 in model A): T0's calibration fixes each camera's f / D but not f and D (C1-T), and a camera
really at lambda D with focal length lambda f puts its virtual images about (lambda - 1) D further, which changes the
blur's slope with depth.  What is compared is the RMS width: the detector's sqrt(Lx^2/12 + sx^2) along u and
sqrt(Ly^2/12 + sy^2) along v (well determined where its box length and blur trade against each other; with lengths
the fit collapsed to A -> 0: before_s2_widths/), against the model's sqrt(L^2/16 + sigma0^2) (a uniform ellipse of
full axis L has variance L^2/16 along it).

Which pairs are true.  About 1,000 detections per image: each cam1 detection's curved image line on cam2 passes within
1.2 px of 2.6 cam2 detections, and T1-a (tests/t1a_epipolar_offset.py) shows the true matches as a peak about 2 px
wide on a flat floor of random pairs, one true pair to about ten random ones within 1 px.  The earlier fits
maximised the number of candidate pairs that agree with the model, which fitted the random pairs' widths
(before_s2_bgsub/: true-looking pairs agreed no more often than the rest).  This version separates the true pairs
statistically.  Every pair within BAND px of the curve is gathered with its distance d from it; the pairs with
d <= 1.5 px form the signal band, those with 4 <= d <= BAND the background band, which holds random pairs only.  For
any rule, (count in the signal band) - k (count in the background band), k = 1.5 / (BAND - 4), estimates the number
of true pairs in the signal band that it keeps; the same for sums (moments).
  1. Per camera, a grid over (delta, z_f, sigma0, A), maximising the estimated number of true pairs whose two widths
     agree within max(0.5 px, 20 %).  Along A the agreeing values form an interval for each pair, so the count over
     all A's is one pass; then a finer grid around the best.
  2. Model A (delta = 0) or B (the best delta), B only if it keeps more true pairs on the held-out frames by more
     than two standard errors.
  3. The amplitude: log flux against log(|L_h| + |L_v| + 2), a straight line per camera, by moments (the signal
     band's pairs that agree in both cameras, minus k times the background band's).
Judged (plan L3), on the held-out frames' true pairs: the residuals of the widths (68 % within 1.5 px, the plan's
RMS figure for lengths, widths being about a third of a length; the quantiles from the background-subtracted
distribution, which an RMS of heavy tails would not describe) and no trend with depth (the background-subtracted mean
residual in depth bins, its slope within 3 standard errors).  Also the depth cue's worth: the fraction of true pairs
the check keeps against the fraction of random pairs, over a range of tolerances, and the tolerances for
s3_track.py's tracer.shape_check: those that maximise (true kept) / sqrt(random kept), the significance of the true
matches among the candidates (no setting keeps 90 % of the true pairs: the residuals' tails, mostly blended
detections, are wide).
Frames: every 5th of S1's (i = 0, 5, ..., 250; raw frames 4 i); fitting i = 0, 10, ..., 250 (26), held out
i = 5, 15, ..., 245 (25).  Data: data/s2/pairs_band.npz (--reuse reads it).
Writes data/T1_shape.json, its OTF table (model "segment", box lengths of equal second moment, 15 x 11 x 17 nodes
over the view volume) data/T1_otf_shape.txt, and s2_shape_fit_out.txt, .json and s2_shape_fit.png beside itself.
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402

lpt = ts.lib()
import shape_model as sm  # noqa: E402  (tests/cylinder_stb, put on the path by ts.lib)

FIT = list(range(0, 251, 10))
TEST = list(range(5, 246, 10))
SIG = 1.5            # signal band: d <= SIG px
BG0 = 4.0            # background band: BG0 <= d <= band
TOL = (0.5, 0.2)     # agreement: |measured - predicted| <= max(TOL[0], TOL[1] x predicted)
A_GRID = (0.25, 0.05, 795)   # A0, dA, n: A = 0.25 ... 39.95 mm


def poly_dist(poly, Q):
    """Distance of points Q (m, 2) to a polyline (k, 2), and whether the nearest point is inside it (not its ends)."""
    A, B = poly[:-1], poly[1:]
    E = B - A
    e2 = np.maximum((E ** 2).sum(axis=1), 1e-24)
    best = np.full(len(Q), np.inf)
    tb = np.zeros(len(Q))
    kb = np.zeros(len(Q), int)
    for k, (a, e, ee) in enumerate(zip(A, E, e2)):
        t = np.clip(((Q - a) @ e) / ee, 0.0, 1.0)
        d = np.hypot(*(Q - (a + t[:, None] * e)).T)
        upd = d < best
        best[upd], tb[upd], kb[upd] = d[upd], t[upd], k
    inside = ~(((kb == 0) & (tb <= 0.0)) | ((kb == len(A) - 1) & (tb >= 1.0)))
    return best, inside


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


def candidate_pairs(cams, sm_, dets, vol, band):
    """(i, j, X, distance from the curve, closest approach) for every cam1 detection i and cam2 detection j whose
    distance from i's curve is at most band, its nearest point inside the curve, X inside the view volume."""
    d1, d2 = dets
    Q2 = d2[:, :2]
    los2 = [cams[1].lineOfSight(lpt.Pt2D(float(u), float(v))) for u, v in Q2]
    out = []
    for i, (u, v) in enumerate(d1[:, :2]):
        los1 = cams[0].lineOfSight(lpt.Pt2D(float(u), float(v)))
        ok, _chord, poly = sm_.makeCurve2DFromLOS3D(1, los1)
        if not ok:
            continue
        P = np.array([[p[0], p[1]] for p in poly])
        P = P[np.all(np.isfinite(P), axis=1)]
        if len(P) < 2:
            continue
        lo, hi = P.min(axis=0) - band, P.max(axis=0) + band
        sel = np.flatnonzero(np.all((Q2 >= lo) & (Q2 <= hi), axis=1))
        if not len(sel):
            continue
        dist, inside = poly_dist(P, Q2[sel])
        keep = (dist <= band) & inside
        for j, dd in zip(sel[keep], dist[keep]):
            X, gap = midpoint(los1, los2[j])
            if X is None or gap > 1.0:
                continue
            if not (vol["x"][0] <= X[0] <= vol["x"][1] and vol["y"][0] <= X[1] <= vol["y"][1]
                    and vol["z"][0] <= X[2] <= vol["z"][1]):
                continue
            out.append((i, int(j), X, float(dd), gap))
    return out


def gather(frames, cams, sm_, noise, vol, band, log):
    P = lpt.LineFinderParam()
    rows, counts = [], []
    for fi in frames:
        k = 4 * fi
        dets = []
        for c in (1, 2):
            img = cv2.imread(os.path.join(ts.PRE, f"cam{c}", f"frame{k:06d}.tif"), cv2.IMREAD_UNCHANGED)
            dets.append(np.array(lpt.detectLines(lpt.Image.from_numpy(img), float(noise[c - 1]), P)))
        pairs = candidate_pairs(cams, sm_, dets, vol, band)
        for (i, j, X, dd, gap) in pairs:
            a, b = dets[0][i], dets[1][j]
            rows.append([fi, i, j, *X, dd, gap, a[2], a[3], a[4], a[5], a[8], a[10],
                         b[2], b[3], b[4], b[5], b[8], b[10]])
        counts.append((fi, len(dets[0]), len(dets[1]), len(pairs)))
        log(f"    frame {k:4d}: detections {len(dets[0])} / {len(dets[1])}; pairs within {band} px {len(pairs)}")
    cols = ["frame", "i", "j", "X", "Y", "Z", "d_epi", "gap", "Lx1", "Ly1", "sx1", "sy1", "flux1", "score1",
            "Lx2", "Ly2", "sx2", "sy2", "flux2", "score2"]
    return np.array(rows), cols, np.array(counts)


# ------------------------------------------------------------------ the model in widths
def widths_meas(data, ci, c):
    Lx, Ly = data[:, ci[f"Lx{c + 1}"]], data[:, ci[f"Ly{c + 1}"]]
    sx, sy = data[:, ci[f"sx{c + 1}"]], data[:, ci[f"sy{c + 1}"]]
    return np.sqrt(Lx ** 2 / 12 + sx ** 2), np.sqrt(Ly ** 2 / 12 + sy ** 2)


def lengths_pred(p, f, z):
    A, zf, s0, dl = p
    return A * f * np.abs(1.0 / (z[:, 0] + dl) - 1.0 / zf), A * f * np.abs(1.0 / (z[:, 1] + dl) - 1.0 / zf)


def widths_pred(p, f, z):
    Lh, Lv = lengths_pred(p, f, z)
    return np.sqrt(Lh ** 2 / 16 + p[2] ** 2), np.sqrt(Lv ** 2 / 16 + p[2] ** 2)


def agree(wm, wp, tol=TOL):
    return np.abs(wm - wp) <= np.maximum(tol[0], tol[1] * wp)


def width_window(wm, tol=TOL):
    """The predicted widths that agree with a measured wm: [min(wm - tp, wm/(1 + tr)), max(wm + tp, wm/(1 - tr))]."""
    tp, tr = tol
    return np.minimum(wm - tp, wm / (1.0 + tr)), np.maximum(wm + tp, wm / (1.0 - tr))


def a_interval(w_lo, w_hi, g, s0):
    """The A's whose predicted width sqrt(A^2 g^2 / 16 + s0^2) lies in [w_lo, w_hi] (g = f |1/z - 1/z_f|)."""
    g = np.maximum(g, 1e-12)
    lo = 4.0 * np.sqrt(np.maximum(w_lo ** 2 - s0 ** 2, 0.0)) / g
    hi = np.where(w_hi > s0, 4.0 * np.sqrt(np.maximum(w_hi ** 2 - s0 ** 2, 0.0)) / g, -1.0)
    return lo, hi


def excess_over_A(lo, hi, w):
    """Sum of the weights w of the pairs whose A-interval [lo, hi] holds each grid value of A."""
    A0, dA, n = A_GRID
    i0 = np.clip(np.ceil((lo - A0) / dA), 0, n).astype(np.int64)
    i1 = np.clip(np.floor((hi - A0) / dA), -1, n - 1).astype(np.int64)
    ok = i1 >= i0
    d = np.bincount(i0[ok], weights=w[ok], minlength=n + 1) - np.bincount(i1[ok] + 1, weights=w[ok], minlength=n + 1)
    return np.cumsum(d)[:n]


class Bands:
    """The signal and background bands of a set of pairs: weights +1 and -k."""

    def __init__(self, data, ci, band):
        d = data[:, ci["d_epi"]]
        self.sig = d <= SIG
        self.bg = (d >= BG0) & (d <= band)
        self.k = SIG / (band - BG0)
        self.w = np.where(self.sig, 1.0, np.where(self.bg, -self.k, 0.0))
        self.n_sig, self.n_bg = int(self.sig.sum()), int(self.bg.sum())
        self.n_true = self.n_sig - self.k * self.n_bg
        self.n_true_se = math.sqrt(self.n_sig + self.k ** 2 * self.n_bg)

    def true_count(self, m):
        """Estimated true pairs (and its standard error) among the pairs with m."""
        a, b = float(m[self.sig].sum()), float(m[self.bg].sum())
        return a - self.k * b, math.sqrt(a + self.k ** 2 * b)

    def random_rate(self, m):
        return float(m[self.bg].mean())

    def true_mean(self, m, x):
        """Background-subtracted mean of x over the true pairs with m."""
        s, b = m & self.sig, m & self.bg
        n = s.sum() - self.k * b.sum()
        return float((x[s].sum() - self.k * x[b].sum()) / n) if n > 0 else float("nan"), float(n)

    def true_cdf(self, x, grid, m=None):
        """Background-subtracted fraction of the true pairs (with m) whose x <= each grid value."""
        m = np.ones(len(x), bool) if m is None else m
        s, b = np.sort(x[m & self.sig]), np.sort(x[m & self.bg])
        n = len(s) - self.k * len(b)
        return (np.searchsorted(s, grid, side="right") - self.k * np.searchsorted(b, grid, side="right")) / n


def grid_fit(wm, z, f, bands, deltas, zf_step, s0s, zf_span=40.0, around=None):
    """The (A, z_f, sigma0, delta) that keep the most true pairs in one camera (both widths agree)."""
    (wl_h, wh_h), (wl_v, wh_v) = width_window(wm[0]), width_window(wm[1])
    sel = bands.w != 0
    w = bands.w[sel]
    wl_h, wh_h, wl_v, wh_v, zz0 = wl_h[sel], wh_h[sel], wl_v[sel], wh_v[sel], z[sel]
    A0, dA, _n = A_GRID
    best = (-np.inf, None)
    for dl in deltas:
        zz = zz0 + dl
        ih, iv = 1.0 / zz[:, 0], 1.0 / zz[:, 1]
        if around is None:
            zfs = np.arange(np.nanmin(zz) - zf_span, np.nanmax(zz) + zf_span, zf_step)
        else:
            zfs = around[1] + np.arange(-zf_span, zf_span + 1e-9, zf_step)
        for zf in zfs:
            gh, gv = f * np.abs(ih - 1.0 / zf), f * np.abs(iv - 1.0 / zf)
            for s0 in s0s:
                lo_h, hi_h = a_interval(wl_h, wh_h, gh, s0)
                lo_v, hi_v = a_interval(wl_v, wh_v, gv, s0)
                E = excess_over_A(np.maximum(lo_h, lo_v), np.minimum(hi_h, hi_v), w)
                k = int(np.argmax(E))
                if E[k] > best[0]:
                    best = (float(E[k]), [float(A0 + k * dA), float(zf), float(s0), float(dl)])
    return best


def agree_cam(p, f, z, wm, tol=TOL):
    ph, pv = widths_pred(p, f, z)
    return agree(wm[0], ph, tol) & agree(wm[1], pv, tol)


def assess(ps, f_px, data, z, ci, band, depth_bins=6):
    """Held-out figures of a model: true and random pairs kept, residual quantiles, the trend with depth."""
    bands = Bands(data, ci, band)
    wm = [widths_meas(data, ci, c) for c in (0, 1)]
    ag = [agree_cam(ps[c], f_px[c], z[c], wm[c]) for c in (0, 1)]
    out = {"n_sig": bands.n_sig, "n_bg": bands.n_bg, "k": bands.k, "n_true": bands.n_true,
           "n_true_se": bands.n_true_se}
    for name, m in (("cam1", ag[0]), ("cam2", ag[1]), ("joint", ag[0] & ag[1])):
        t, se = bands.true_count(m)
        out[name] = {"true_kept": t / bands.n_true, "true_kept_se": se / bands.n_true,
                     "random_kept": bands.random_rate(m)}
    grid = np.arange(0.0, 6.0001, 0.01)
    for c in (0, 1):
        ph, pv = widths_pred(ps[c], f_px[c], z[c])
        other = ag[1 - c]
        rec = {}
        for nm, r in (("wh", wm[c][0] - ph), ("wv", wm[c][1] - pv)):
            F = bands.true_cdf(np.abs(r), grid)
            Fo = bands.true_cdf(np.abs(r), grid, other)
            q = {f"q{int(p * 100)}": float(grid[min(np.searchsorted(F, p), len(grid) - 1)]) for p in (0.5, 0.68, 0.9)}
            qo = {f"q{int(p * 100)}": float(grid[min(np.searchsorted(Fo, p), len(grid) - 1)]) for p in (0.5, 0.68, 0.9)}
            # the trend: the background-subtracted mean residual (|r| <= 3 px, the pairs agreeing in the other camera)
            zc = z[c][:, 0]
            m0 = other & (np.abs(r) <= 3.0)
            edges = np.quantile(zc[m0 & bands.sig], np.linspace(0, 1, depth_bins + 1))
            zb, rb, eb = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mb = m0 & (zc >= lo) & (zc <= hi)
                mu, n = bands.true_mean(mb, r)
                s, b = mb & bands.sig, mb & bands.bg
                var = ((r[s] - mu) ** 2).sum() + bands.k ** 2 * ((r[b] - mu) ** 2).sum()
                zb.append(0.5 * (lo + hi))
                rb.append(mu)
                eb.append(math.sqrt(var) / n if n > 0 else float("nan"))
            zb, rb, eb = np.array(zb), np.array(rb), np.array(eb)
            ok = np.isfinite(rb) & np.isfinite(eb) & (eb > 0)
            W = 1.0 / eb[ok] ** 2
            X = np.column_stack([zb[ok] - np.average(zb[ok], weights=W), np.ones(ok.sum())])
            cov = np.linalg.inv(X.T @ (W[:, None] * X))
            coef = cov @ (X.T @ (W * rb[ok]))
            rec[nm] = {**q, "agreeing_other": qo, "bias": float(bands.true_mean(other & (np.abs(r) <= 3.0), r)[0]),
                       "depth_bins_mm": zb.tolist(), "mean_resid_px": rb.tolist(), "mean_resid_se": eb.tolist(),
                       "slope_px_per_mm": float(coef[0]), "slope_se": float(math.sqrt(cov[0, 0])),
                       "slope_over_range_px": float(coef[0] * (zb[-1] - zb[0]))}
        out[f"cam{c + 1}_resid"] = rec
    out["trend"] = any(abs(out[f"cam{c}_resid"][k]["slope_px_per_mm"]) > 3 * out[f"cam{c}_resid"][k]["slope_se"]
                       for c in (1, 2) for k in ("wh", "wv"))
    out["q68_max"] = max(out[f"cam{c}_resid"][k]["q68"] for c in (1, 2) for k in ("wh", "wv"))
    return out, ag


def tolerance_scan(ps, f_px, data, z, ci, band):
    """Joint true and random pairs kept for shape_check's (tol_px, tol_rel)."""
    bands = Bands(data, ci, band)
    wm = [widths_meas(data, ci, c) for c in (0, 1)]
    rows = []
    for tp in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5):
        for tr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
            m = agree_cam(ps[0], f_px[0], z[0], wm[0], (tp, tr)) & agree_cam(ps[1], f_px[1], z[1], wm[1], (tp, tr))
            t, se = bands.true_count(m)
            rows.append({"tol_px": tp, "tol_rel": tr, "true_kept": t / bands.n_true, "true_kept_se": se / bands.n_true,
                         "random_kept": bands.random_rate(m)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=8.0)
    ap.add_argument("--reuse", action="store_true", help="read data/s2/pairs_band.npz instead of gathering")
    args = ap.parse_args()
    log = case.Log(os.path.join(HERE, "s2_shape_fit"))
    t0 = time.time()
    vol = ts.view_volume(lpt)
    noise = ts.noise_per_camera()
    fds = [sm.FocalDepths(lpt, f) for f in ts.CAMS]
    f_px = [math.sqrt(fd.K[0, 0] * fd.K[1, 1]) for fd in fds]
    log(f"S2: T1's shape model; build {ts.BUILD}; cameras {ts.CAMS}; view volume x {np.round(vol['x'], 1)}, "
        f"y {np.round(vol['y'], 1)}, z {np.round(vol['z'], 1)} mm (r <= {vol['r_max_mm']:.1f}); noise "
        f"{np.round(noise, 2)}; pairs within {args.band} px; signal band d <= {SIG} px, background {BG0} to "
        f"{args.band} px")
    npz = os.path.join(ts.DATA, "s2", "pairs_band.npz")
    if args.reuse:
        d = np.load(npz)
        D, T, cols = d["D"], d["T"], [str(c) for c in d["cols"]]
        zD, zT = [d["zD1"], d["zD2"]], [d["zT1"], d["zT2"]]
        log(f"  pairs read from {npz}")
    else:
        cfg_path = os.path.join(ts.DATA, "s2", "config.json")
        ts.stb_config(cfg_path, (0, 10), vol, {"intensity_threshold": 30, "radius_px": 2.0})
        bs = lpt.BasicSetting()
        bs.readConfig(cfg_path)
        tc = lpt.TracerConfig()
        tc.readConfig(bs._object_config_paths[0], bs)
        cams = list(bs._cam_list)
        sm_ = lpt.StereoMatch(cams, [[] for _ in cams], tc)
        log("  fitting frames:")
        D, cols, cD = gather(FIT, cams, sm_, noise, vol, args.band, log)
        log("  held-out frames:")
        T, _, cT = gather(TEST, cams, sm_, noise, vol, args.band, log)
        t1 = time.time()
        zD = [fd(D[:, 3:6]) for fd in fds]
        zT = [fd(T[:, 3:6]) for fd in fds]
        okD = np.all(np.isfinite(zD[0]), axis=1) & np.all(np.isfinite(zD[1]), axis=1)
        D, zD = D[okD], [z[okD] for z in zD]
        okT = np.all(np.isfinite(zT[0]), axis=1) & np.all(np.isfinite(zT[1]), axis=1)
        T, zT = T[okT], [z[okT] for z in zT]
        log(f"  focal depths in {time.time() - t1:.0f} s ({(~okD).sum() + (~okT).sum()} pairs without)")
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        np.savez(npz, D=D, T=T, zD1=zD[0], zD2=zD[1], zT1=zT[0], zT2=zT[1], cols=np.array(cols),
                 f_px=np.array(f_px), countsD=cD, countsT=cT, band=args.band)
    ci = {c: k for k, c in enumerate(cols)}
    bD, bT = Bands(D, ci, args.band), Bands(T, ci, args.band)
    log(f"  pairs: {len(D)} fitting, {len(T)} held out ({time.time() - t0:.0f} s)")
    for nm, b in (("fitting", bD), ("held out", bT)):
        log(f"  {nm}: signal band {b.n_sig}, background band {b.n_bg} (k = {b.k:.4f}); estimated true pairs in the "
            f"signal band {b.n_true:.0f} +- {b.n_true_se:.0f} ({b.n_true / b.n_sig * 100:.1f} % of it)")
    # the distance profile: true pairs per 0.25 px of d (all frames)
    dd = np.concatenate([D[:, ci["d_epi"]], T[:, ci["d_epi"]]])
    h, e = np.histogram(dd, bins=np.arange(0.0, args.band + 1e-9, 0.25))
    floor = h[e[:-1] >= BG0].mean()
    excess = h - floor
    cum = np.cumsum(excess)
    tot = cum[e[1:] <= 3.0][-1]
    prof = {f"{x:.2f}": float(cum[np.searchsorted(e[1:], x)] / tot) for x in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)}
    log(f"  true pairs within d (of those within 3 px): " + ", ".join(f"{k} px {v * 100:.0f} %" for k, v in prof.items())
        + f"; floor {floor:.0f} pairs per 0.25 px; true pairs within 3 px {tot:.0f} "
        f"({tot / (len(FIT) + len(TEST)):.0f} per frame)")

    wD = [widths_meas(D, ci, c) for c in (0, 1)]
    out = {"models": {}, "band": args.band, "signal_band_px": SIG, "background_from_px": BG0,
           "distance_profile": {"bins": e.tolist(), "hist": h.tolist(), "floor": float(floor),
                                "true_within": prof, "true_within_3px": float(tot)}}
    fits = {}
    for c in (0, 1):
        t1 = time.time()
        Ec, pc = grid_fit(wD[c], zD[c], f_px[c], bD, np.arange(-200.0, 200.1, 25.0), 2.0,
                          np.arange(0.7, 1.71, 0.1))
        log(f"  cam{c + 1} coarse grid (delta free): A {pc[0]:.2f} mm, z_f {pc[1]:.0f} mm, sigma0 {pc[2]:.1f} px, delta "
            f"{pc[3]:+.0f} mm: {Ec:.0f} true pairs agree ({time.time() - t1:.0f} s)")
        EA, pA = grid_fit(wD[c], zD[c], f_px[c], bD, [0.0], 2.0, np.arange(0.7, 1.71, 0.1))
        log(f"  cam{c + 1} coarse grid (delta 0): A {pA[0]:.2f} mm, z_f {pA[1]:.0f} mm, sigma0 {pA[2]:.1f} px: "
            f"{EA:.0f} true pairs agree")
        EfB, pfB = grid_fit(wD[c], zD[c], f_px[c], bD, pc[3] + np.arange(-20.0, 20.1, 4.0), 0.5,
                            pc[2] + np.arange(-0.1, 0.101, 0.025), zf_span=4.0, around=pc)
        EfA, pfA = grid_fit(wD[c], zD[c], f_px[c], bD, [0.0], 0.5, pA[2] + np.arange(-0.1, 0.101, 0.025),
                            zf_span=4.0, around=pA)
        log(f"  cam{c + 1} fine grid: model A: A {pfA[0]:.2f} mm, z_f {pfA[1]:.1f} mm, sigma0 {pfA[2]:.3f} px "
            f"({EfA:.0f}); model B: A {pfB[0]:.2f} mm, z_f {pfB[1]:.1f} mm, sigma0 {pfB[2]:.3f} px, delta {pfB[3]:+.0f} mm "
            f"({EfB:.0f})")
        fits[c] = {"A": pfA, "B": pfB}

    # model A or B: on the held-out frames, per camera, B's extra true pairs against two standard errors
    wT = [widths_meas(T, ci, c) for c in (0, 1)]
    use_B = []
    for c in (0, 1):
        aA = agree_cam(fits[c]["A"], f_px[c], zT[c], wT[c])
        aB = agree_cam(fits[c]["B"], f_px[c], zT[c], wT[c])
        dif = aB.astype(float) - aA.astype(float)
        gain = float(dif[bT.sig].sum() - bT.k * dif[bT.bg].sum())
        se = math.sqrt(float(np.abs(dif[bT.sig]).sum() + bT.k ** 2 * np.abs(dif[bT.bg]).sum()))
        use_B.append(bool(gain > 2 * se))
        log(f"  cam{c + 1}, held out: model B keeps {gain:+.0f} +- {se:.0f} true pairs more than A -> "
            f"{'B' if use_B[-1] else 'A'}")
    choice = "B" if any(use_B) else "A"
    for label in ("A", "B"):
        ps = [fits[c][label] for c in (0, 1)]
        ev, _ = assess(ps, f_px, T, zT, ci, args.band)
        out["models"][label] = {"params": ps, "heldout": ev}
        log(f"  model {label}, held out: estimated true pairs {ev['n_true']:.0f} +- {ev['n_true_se']:.0f}; kept: "
            + "; ".join(f"{nm} {ev[nm]['true_kept'] * 100:.0f} +- {ev[nm]['true_kept_se'] * 100:.0f} % of true, "
                        f"{ev[nm]['random_kept'] * 100:.1f} % of random" for nm in ("cam1", "cam2", "joint")))
        for c in (1, 2):
            for k in ("wh", "wv"):
                r = ev[f"cam{c}_resid"][k]
                log(f"    cam{c} {k}: |residual| of true pairs: 50 % <= {r['q50']:.2f}, 68 % <= {r['q68']:.2f}, "
                    f"90 % <= {r['q90']:.2f} px (agreeing in the other camera: {r['agreeing_other']['q50']:.2f}, "
                    f"{r['agreeing_other']['q68']:.2f}, {r['agreeing_other']['q90']:.2f}); bias {r['bias']:+.3f} px; "
                    f"slope with depth {r['slope_px_per_mm']:+.4f} +- {r['slope_se']:.4f} px/mm "
                    f"({r['slope_over_range_px']:+.2f} px over the depths)")
    ps = out["models"][choice]["params"]
    ev = out["models"][choice]["heldout"]
    passed = bool(ev["q68_max"] <= 1.5 and not ev["trend"])
    log(f"  chosen: model {choice}; judged (68 % of true pairs' residuals within 1.5 px, no trend with depth): "
        f"{'PASS' if passed else 'FAIL'} (largest q68 {ev['q68_max']:.2f} px; trend {'yes' if ev['trend'] else 'no'})")

    # the tolerances of shape_check
    scan = tolerance_scan(ps, f_px, T, zT, ci, args.band)
    rec = max(scan, key=lambda r: r["true_kept"] / math.sqrt(max(r["random_kept"], 1e-9)))
    log("  shape_check tolerances, held out (joint: true kept / random kept):")
    for tp in sorted({r["tol_px"] for r in scan}):
        log(f"    tol_px {tp:.1f}: " + "  ".join(f"rel {r['tol_rel']:.2f} {r['true_kept'] * 100:3.0f}/"
                                                f"{r['random_kept'] * 100:4.1f}" for r in scan if r["tol_px"] == tp))
    log(f"  recommended (largest true kept / sqrt(random kept)): tol_px {rec['tol_px']}, tol_rel {rec['tol_rel']}: keeps "
        f"{rec['true_kept'] * 100:.0f} +- {rec['true_kept_se'] * 100:.0f} % of true pairs and "
        f"{rec['random_kept'] * 100:.1f} % of random ones")

    # the amplitude, by moments, on the fitting frames' pairs agreeing in both cameras
    m = agree_cam(ps[0], f_px[0], zD[0], wD[0]) & agree_cam(ps[1], f_px[1], zD[1], wD[1])
    famp = []
    for c in (0, 1):
        Lh, Lv = lengths_pred(ps[c], f_px[c], zD[c])
        x = np.log(sm.BOX * (Lh + Lv) + 2.0)
        flux = D[:, ci[f"flux{c + 1}"]]
        ok = m & (flux > 0)
        y = np.log(np.where(flux > 0, flux, 1.0))
        wgt = np.where(ok, bD.w, 0.0)
        Xm = np.column_stack([x, np.ones_like(x)])
        coef = np.linalg.solve(Xm.T @ (wgt[:, None] * Xm), Xm.T @ (wgt * y))
        famp.append((float(coef[1]), float(coef[0])))
        log(f"  amplitude, cam{c + 1}: flux = {math.exp(coef[1]):.1f} x (|b| + |c| + 2)^{coef[0]:.3f} (by moments over "
            f"{int(ok[bD.sig].sum())} signal and {int(ok[bD.bg].sum())} background pairs; b, c the box lengths)")
    model = {"model": choice, "cameras": [{"camfile": ts.CAMS[c], "f_px": f_px[c], "A_mm": float(ps[c][0]),
                                           "z_f_mm": float(ps[c][1]), "sigma0_px": float(ps[c][2]),
                                           "delta_mm": float(ps[c][3]), "flux_log_a": famp[c][0],
                                           "flux_log_b": famp[c][1]} for c in (0, 1)],
             "fit_frames": FIT, "test_frames": TEST, "band_px": args.band,
             "agreement_rule": f"RMS widths within max({TOL[0]} px, {TOL[1] * 100:.0f} %) in both cameras",
             "shape_check": {"tol_px": rec["tol_px"], "tol_rel": rec["tol_rel"], "true_kept": rec["true_kept"],
                             "random_kept": rec["random_kept"]},
             "heldout": ev, "epipolar_true_within": prof}
    with open(os.path.join(ts.DATA, "T1_shape.json"), "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1)

    # the OTF table
    n = (15, 11, 17)
    vbox = ((vol["x"][0], vol["y"][0], vol["z"][0]), (vol["x"][1], vol["y"][1], vol["z"][1]))
    Pn = sm.grid_points(vbox, n)
    a = np.zeros((2, len(Pn)))
    b_, c_, al, sg = np.zeros_like(a), np.zeros_like(a), np.zeros_like(a), np.zeros_like(a)
    for c in (0, 1):
        z = fds[c](Pn)
        bad = ~np.all(np.isfinite(z), axis=1)
        if bad.any():
            good_ = np.flatnonzero(~bad)
            for i in np.flatnonzero(bad):
                z[i] = z[good_[np.argmin(np.linalg.norm(Pn[good_] - Pn[i], axis=1))]]
        A, zf, s0, dl = ps[c]
        Lh = sm.BOX * A * f_px[c] * (1.0 / (z[:, 0] + dl) - 1.0 / zf)      # signed box lengths
        Lv = sm.BOX * A * f_px[c] * (1.0 / (z[:, 1] + dl) - 1.0 / zf)
        fl = np.exp(famp[c][0]) * (np.abs(Lh) + np.abs(Lv) + 2.0) ** famp[c][1]
        a[c] = fl / (sm.profile_integral(Lh, s0) * sm.profile_integral(Lv, s0))
        b_[c], c_[c], sg[c] = Lh, Lv, s0
        al[c] = 0.0   # no roll: lines within +-0.3 deg of the image axes (P0b)
    sm.write_otf(os.path.join(ts.DATA, "T1_otf_shape.txt"), vbox, n, a, b_, c_, al, sg)
    log(f"  written data/T1_shape.json and data/T1_otf_shape.txt ({n[0]} x {n[1]} x {n[2]} nodes); "
        f"{time.time() - t0:.0f} s")

    # figure: the distance profile, the true pairs' residuals, the trend with depth
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    cc = 0.5 * (e[1:] + e[:-1])
    ax[0, 0].bar(cc, h, width=0.25, color="0.6")
    ax[0, 0].axhline(floor, color="k", lw=0.8)
    ax[0, 0].axvspan(0, SIG, color="C2", alpha=0.15)
    ax[0, 0].axvspan(BG0, args.band, color="C3", alpha=0.08)
    ax[0, 0].set_xlabel("distance from the curve d [px]")
    ax[0, 0].set_ylabel("pairs per 0.25 px")
    ax[0, 0].set_title("signal (green) and background (red) bands")
    wTm = [widths_meas(T, ci, c) for c in (0, 1)]
    bins = np.arange(-4.0, 4.0001, 0.2)
    for c in (0, 1):
        ph, pv = widths_pred(ps[c], f_px[c], zT[c])
        for k2, (nm, r) in enumerate((("wh", wTm[c][0] - ph), ("wv", wTm[c][1] - pv))):
            hs, _ = np.histogram(r[bT.sig], bins=bins)
            hb, _ = np.histogram(r[bT.bg], bins=bins)
            a_ = ax[0, 1 + c] if k2 == 0 else ax[1, 1 + c]
            a_.step(0.5 * (bins[1:] + bins[:-1]), hs - bT.k * hb, where="mid", color="C0", label="true (subtracted)")
            a_.step(0.5 * (bins[1:] + bins[:-1]), bT.k * hb, where="mid", color="C3", lw=0.8, label="random (scaled)")
            a_.axhline(0, color="k", lw=0.5)
            a_.set_xlabel(f"cam{c + 1} {nm}: measured - predicted [px]")
            a_.legend(fontsize=7)
    for c in (0, 1):
        for nm, mk in (("wh", "o"), ("wv", "s")):
            r = ev[f"cam{c + 1}_resid"][nm]
            ax[1, 0].errorbar(np.array(r["depth_bins_mm"]) - np.mean(r["depth_bins_mm"]), r["mean_resid_px"],
                              yerr=r["mean_resid_se"], marker=mk, ls="-", label=f"cam{c + 1} {nm}")
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].set_xlabel("z_tan - its mean [mm]")
    ax[1, 0].set_ylabel("true pairs' mean residual [px]")
    ax[1, 0].legend(fontsize=7)
    fig.suptitle(f"S2: T1's width model {choice} (held-out frames)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s2_shape_fit.png"), dpi=100)
    plt.close(fig)
    out.update({"chosen": choice, "model": model, "pass": passed, "pairs_fit": int(len(D)), "pairs_test": int(len(T)),
                "tolerance_scan": scan})
    log.save(out)


if __name__ == "__main__":
    main()
