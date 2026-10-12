"""M0 and M1 (plan, section 3): T1's shape model from its tracked particles' measured widths.

    python m01_shape.py

M0, the data.  The long tracks' points with a detection within 3 px in both cameras (gather.py, every 5th frame), their
measured RMS widths w = sqrt(L^2/12 + s^2) along u and v, and the focal depths z_tan, z_sag of each camera at the
track's 3-D point, from two runs:
  - the 15 px run (s3_final_m15): its tracks passed the depth cue at 0.8 px / 20 %, so their widths are censored towards
    the current model;
  - the run with the cue loosened to 1.5 px / 40 % (s3_final_m15cue, this plan's), keeping only its long (>= 20
    points), swirl-consistent tracks (T1-e's test: the mean |v_r| / |v| over the steps >= 8 mm from the axis at most
    0.3), so that the looser cue's extra ghosts stay out.
  Per camera and axis: the residual against S2's model (A, z_f, sigma0; cases/cyl_20261007/data/T1_shape.json), its
  depth trend, the share inside the cue, and the share of the residual's variance that is per particle (between
  tracks, the intra-class correlation).
M1, the refit.  Per camera, by robust least squares (Cauchy loss, scale 0.5 px) on both widths, on the tracks of even
id (the odd ones held out), models:
  A   A, z_f, sigma0 (S2's model A, refitted on the tracks);
  B   + delta, one depth offset of both focal depths (S2's model B);
  C   + delta_h, delta_v: separate offsets of z_tan and z_sag (the trends T1-c and S2 found have opposite signs in w_h
      and w_v: the astigmatism's split, not the overall depth, is off);
  D   C with separate sigma0 along u and v.
  Judged on held-out data, truth-free, as S2 judges: on the held-out frames 5, 15, ..., 245, every candidate pair of all
  detections (t1_m15's gather), the depth cue (both widths in both cameras within max(0.8 px, 20 %) of the model at the
  pair's midpoint) under each model: the true pairs it keeps (signal band d <= 1.5 px minus k x the band 4-8 px) and
  the random ones (the band 4-8 px), against S2's model; also the tolerances that maximise true / sqrt(random) (S2's
  rule).  The plan's bar: >= 65 % of the true pairs kept with no more random ones; stop if the error is per-particle
  scatter (a refit cannot remove that).  The flux law (log flux against log(|b| + |c| + 2)) is refitted for the
  chosen model, and its OTF table written for M2 (data/shape/T1_otf_<model>.txt; the case's own table untouched).
Writes m01_shape_out.txt, .json and m01_shape.png.
"""
from __future__ import annotations

import math
import os

import numpy as np

import selfcal as sc

GATE = 3.0
TOL = (0.8, 0.2)
SIG, BG0, BAND = 1.5, 4.0, 8.0
HELD = list(range(5, 246, 10))


def swirl_per_track(tracks, r_min=8.0):
    """T1-e's swirl statistic per track key: the mean of |v_r| / |v| over its steps at least r_min from the axis."""
    out = {}
    for k, tr in tracks.items():
        if len(tr) < 4 or np.any(np.diff(tr[:, 0]) != 1):
            out[k] = np.nan
            continue
        X = tr[:, 1:4]
        v = np.diff(X, axis=0)
        m = 0.5 * (X[1:] + X[:-1])
        r = np.hypot(m[:, 0], m[:, 2])
        sp = np.linalg.norm(v, axis=1)
        ok = (r >= r_min) & (sp > 0)
        vr = (m[:, 0] * v[:, 0] + m[:, 2] * v[:, 2]) / np.maximum(r, 1e-9)
        out[k] = float(np.mean(np.abs(vr[ok]) / sp[ok])) if ok.any() else np.nan
    return out


def points(name, run_dir, keep_tracks=None):
    """Track points paired in both cameras: widths (n, 2 cams, 2 axes), focal depths (n, 2, 2), detections' flux and
    box lengths, track keys."""
    recs = sc.load_gather(name)
    W, Z, F, Lb, keys, fr = [], [], [], [], [], []
    for g in recs:
        if not len(g["X"]):
            continue
        m = ((g["pair"] >= 0) & (g["pair_dist"] <= GATE)).all(axis=1)
        kk = [(str(a), int(b)) for a, b in zip(g["track_kind"], g["track_id"])]
        if keep_tracks is not None:
            m &= np.array([k in keep_tracks for k in kk])
        idx = np.flatnonzero(m)
        w = np.zeros((len(idx), 2, 2))
        f = np.zeros((len(idx), 2))
        lb = np.zeros((len(idx), 2, 2))
        for c in (0, 1):
            d = g[f"det{c}"][g["pair"][idx, c]]
            w[:, c, 0] = np.sqrt(d[:, 2] ** 2 / 12 + d[:, 4] ** 2)
            w[:, c, 1] = np.sqrt(d[:, 3] ** 2 / 12 + d[:, 5] ** 2)
            f[:, c] = d[:, 8]
            lb[:, c] = d[:, 2:4]
        z = np.stack([g["z_track0"][idx], g["z_track1"][idx]], axis=1)
        W.append(w)
        Z.append(z)
        F.append(f)
        Lb.append(lb)
        keys += [kk[i] for i in idx]
        fr.append(np.full(len(idx), int(g["t"])))
    return {"w": np.concatenate(W), "z": np.concatenate(Z), "flux": np.concatenate(F), "Lbox": np.concatenate(Lb),
            "keys": keys, "frame": np.concatenate(fr)}


def widths(p, f_px, z, model):
    """Predicted RMS widths (n, 2) for the parameters p of a model at focal depths z (n, 2)."""
    if model == "A":
        A, zf, s0 = p
        dh = dv = 0.0
        sh = sv = s0
    elif model == "B":
        A, zf, s0, d = p
        dh = dv = d
        sh = sv = s0
    elif model == "C":
        A, zf, s0, dh, dv = p
        sh = sv = s0
    else:
        A, zf, sh, sv, dh, dv = p
    Lh = A * f_px * np.abs(1.0 / (z[:, 0] + dh) - 1.0 / zf)
    Lv = A * f_px * np.abs(1.0 / (z[:, 1] + dv) - 1.0 / zf)
    return np.column_stack([np.sqrt(Lh ** 2 / 16 + sh ** 2), np.sqrt(Lv ** 2 / 16 + sv ** 2)])


def start(model, s2):
    A, zf, s0 = s2["A_mm"], s2["z_f_mm"], s2["sigma0_px"]
    return {"A": [A, zf, s0], "B": [A, zf, s0, 0.0], "C": [A, zf, s0, 0.0, 0.0], "D": [A, zf, s0, s0, 0.0, 0.0]}[model]


def fit_model(w, z, f_px, model, s2):
    from scipy.optimize import least_squares
    ok = np.all(np.isfinite(z), axis=1) & np.all(np.isfinite(w), axis=1)

    def res(p):
        return (widths(p, f_px, z[ok], model) - w[ok]).ravel()
    lo = {"A": [1.0, 300.0, 0.3], "B": [1.0, 300.0, 0.3, -300.0], "C": [1.0, 300.0, 0.3, -300.0, -300.0],
          "D": [1.0, 300.0, 0.3, 0.3, -300.0, -300.0]}[model]
    hi = {"A": [60.0, 3000.0, 3.0], "B": [60.0, 3000.0, 3.0, 300.0], "C": [60.0, 3000.0, 3.0, 300.0, 300.0],
          "D": [60.0, 3000.0, 3.0, 3.0, 300.0, 300.0]}[model]
    best = None
    for dA in (0.7, 1.0, 1.4):
        p0 = np.array(start(model, s2), float)
        p0[0] *= dA
        r = least_squares(res, p0, bounds=(lo, hi), loss="cauchy", f_scale=0.5, x_scale="jac", max_nfev=2000)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, float(best.cost)


def residual_stats(w, z, f_px, p, model):
    ok = np.all(np.isfinite(z), axis=1)
    pr = widths(p, f_px, z[ok], model)
    r = w[ok] - pr                         # measured - predicted (T1-c's sign)
    q = lambda x: np.percentile(np.abs(x), [50, 68, 90]).tolist()  # noqa: E731
    inside = np.all(np.abs(r) <= np.maximum(TOL[0], TOL[1] * pr), axis=1)
    return {"n": int(ok.sum()), "q_h": q(r[:, 0]), "q_v": q(r[:, 1]), "median_h": float(np.median(r[:, 0])),
            "median_v": float(np.median(r[:, 1])), "inside_cue": float(inside.mean())}, r, ok


def icc(r, keys):
    """The share of the residual's variance that lies between tracks (a robust ANOVA: track medians)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for x, k in zip(r, keys):
        if np.isfinite(x):
            groups[k].append(x)
    means = np.array([np.mean(v) for v in groups.values() if len(v) >= 3])
    within = np.concatenate([np.array(v) - np.mean(v) for v in groups.values() if len(v) >= 3])
    nbar = np.mean([len(v) for v in groups.values() if len(v) >= 3])
    vb = np.var(means) - np.var(within) / nbar
    return float(max(vb, 0.0) / (max(vb, 0.0) + np.var(within))), int(len(means))


def depth_trend(x, r, nb=6):
    ok = np.isfinite(x) & np.isfinite(r)
    e = np.quantile(x[ok], np.linspace(0, 1, nb + 1))
    c, m = [], []
    for lo, hi in zip(e[:-1], e[1:]):
        s = ok & (x >= lo) & (x <= hi)
        c.append(float(np.median(x[s])))
        m.append(float(np.median(r[s])))
    return c, m


def fit_map(uv, y, lam=1e4, iters=6):
    """A smooth scalar map over the sensor (one B-spline component of selfcal.Field), Cauchy-weighted."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    f = sc.Field(1)
    cols, vals = f.basis_rows(uv)
    A = sp.csr_matrix((vals.ravel(), (np.repeat(np.arange(len(uv)), 16), cols.ravel())), shape=(len(uv), f.nb))
    R = lam * sp.csr_matrix(f.penalty()) + sp.identity(f.nb)
    w = np.ones(len(uv))
    th = np.zeros(f.nb)
    s = 1.4826 * np.median(np.abs(y - np.median(y)))
    for _ in range(iters):
        th = spla.spsolve((A.T @ sp.diags(w) @ A + R).tocsc(), A.T @ (w * y))
        e = (y - A @ th) / max(s, 1e-6)
        w = 1.0 / (1.0 + (e / 2.0) ** 2)
    f.theta[:f.nb] = th
    return f


def eval_map(f, uv):
    return f.eval_points(0, uv)[:, 0]


def cue_stats(recs, params, f_px, models, tol=TOL, wmaps=None, use=()):
    """S2's held-out statistics of the depth cue on every candidate pair: (true kept, its SE, random kept); with
    wmaps, the measured widths are first corrected by the maps of (camera, axis) in use."""
    keep, d = [], []
    for g in recs:
        C = g["cand"]
        if not len(C):
            continue
        ok = np.ones(len(C), bool)
        for c in (0, 1):
            z = g[f"z_cand{c}"]
            good = np.all(np.isfinite(z), axis=1)
            det = g[f"det{c}"][C[:, c].astype(int)]
            wm = np.column_stack([np.sqrt(det[:, 2] ** 2 / 12 + det[:, 4] ** 2),
                                  np.sqrt(det[:, 3] ** 2 / 12 + det[:, 5] ** 2)])
            for k in (0, 1):
                if wmaps is not None and (c, k) in use:
                    wm[:, k] = wm[:, k] - eval_map(wmaps[(c, k)], det[:, :2])
            zz = np.where(good[:, None], z, 1e3)
            pr = widths(params[c], f_px[c], zz, models[c])
            ok &= good & np.all(np.abs(wm - pr) <= np.maximum(tol[0], tol[1] * pr), axis=1)
        keep.append(ok)
        d.append(C[:, 5])
    keep, d = np.concatenate(keep), np.concatenate(d)
    sig, bg = d <= SIG, (d >= BG0) & (d <= BAND)
    k = SIG / (BAND - BG0)
    n_true = sig.sum() - k * bg.sum()
    t = keep[sig].sum() - k * keep[bg].sum()
    se = math.sqrt(keep[sig].sum() + k ** 2 * keep[bg].sum())
    return float(t / n_true), float(se / n_true), float(keep[bg].mean()), float(n_true / len(recs))


def main():
    log = sc.Log("m01_shape")
    shape = sc.load_json(os.path.join(sc.CASE_DATA, "T1_shape.json"))
    s2 = shape["cameras"]
    f_px = [c["f_px"] for c in s2]
    p_s2 = [[c["A_mm"], c["z_f_mm"], c["sigma0_px"]] for c in s2]
    out = {"s2_model": s2}
    log(f"M0/M1: S2's model A: cam1 A {p_s2[0][0]:.2f} mm, z_f {p_s2[0][1]:.1f} mm, sigma0 {p_s2[0][2]:.3f} px; cam2 A "
        f"{p_s2[1][0]:.2f}, z_f {p_s2[1][1]:.1f}, sigma0 {p_s2[1][2]:.3f}")

    # M0: the data
    t_cue, _last = sc.read_tracks(os.path.join(sc.CASE_RUNS, "s3_final_m15cue"))
    sw = swirl_per_track(t_cue)
    keep = {k for k, tr in t_cue.items() if len(tr) >= 20 and np.isfinite(sw[k]) and sw[k] <= 0.3}
    log(f"  the loosened-cue run: {len(t_cue)} long tracks; {len(keep)} kept (>= 20 points, swirl <= 0.3; "
        f"{sum(1 for k, tr in t_cue.items() if len(tr) >= 20)} have >= 20 points)")
    data = {"m15": points("t1_m15", sc.RUN_M15), "cue": points("t1_m15cue", None, keep)}
    out["M0"] = {}
    for nm, D in data.items():
        rec = {"points": len(D["keys"]), "tracks": len(set(D["keys"]))}
        for c in (0, 1):
            st, r, ok = residual_stats(D["w"][:, c], D["z"][:, c], f_px[c], p_s2[c], "A")
            ich, _ = icc(r[:, 0], [k for k, o in zip(D["keys"], ok) if o])
            icv, ntr = icc(r[:, 1], [k for k, o in zip(D["keys"], ok) if o])
            zz = D["z"][ok, c]
            th = depth_trend(zz[:, 0] - s2[c]["z_f_mm"], r[:, 0])
            tv = depth_trend(zz[:, 1] - s2[c]["z_f_mm"], r[:, 1])
            st.update({"icc_h": ich, "icc_v": icv, "icc_tracks": ntr, "trend_h": th, "trend_v": tv})
            rec[f"cam{c + 1}"] = st
            log(f"  M0 {nm}, cam{c + 1}: {st['n']} points; |measured - S2's model| 50/68/90 %: w_h "
                f"{'/'.join(f'{x:.2f}' for x in st['q_h'])}, w_v {'/'.join(f'{x:.2f}' for x in st['q_v'])} px (medians "
                f"{st['median_h']:+.2f}, {st['median_v']:+.2f}); inside the cue {100 * st['inside_cue']:.0f} %; per "
                f"particle (between tracks) {100 * ich:.0f} % / {100 * icv:.0f} % of the variance; trend with z - z_f: "
                f"w_h {', '.join(f'{m:+.2f}' for m in th[1])}; w_v {', '.join(f'{m:+.2f}' for m in tv[1])}")
        out["M0"][nm] = rec

    # M1: the refits on the even tracks, held out on the odd
    held_recs = sc.load_gather("t1_m15", frames=set(HELD))
    base_cue = cue_stats(held_recs, p_s2, f_px, ["A", "A"])
    log(f"  M1: the held-out frames' candidate pairs ({base_cue[3]:.0f} true pairs per frame in the signal band): S2's "
        f"model keeps {100 * base_cue[0]:.1f} +- {100 * base_cue[1]:.1f} % of the true pairs and {100 * base_cue[2]:.2f} "
        f"% of the random ones")
    out["M1"] = {"s2_model_cue": base_cue, "fits": {}}
    fits = {}
    for nm in ("m15", "cue"):
        D = data[nm]
        even = np.array([k[1] % 2 == 0 for k in D["keys"]])
        for model in ("A", "B", "C", "D"):
            ps, rows = [], {}
            for c in (0, 1):
                p, cost = fit_model(D["w"][even, c], D["z"][even, c], f_px[c], model, s2[c])
                st_te, r_te, _ok = residual_stats(D["w"][~even, c], D["z"][~even, c], f_px[c], p, model)
                ps.append(p.tolist())
                rows[f"cam{c + 1}"] = {"params": p.tolist(), "cost": cost, "heldout_tracks": st_te}
            cs = cue_stats(held_recs, ps, f_px, [model, model])
            fits[(nm, model)] = (ps, cs)
            out["M1"]["fits"][f"{nm}_{model}"] = {**rows, "cue": cs}
            log(f"  M1 {nm} model {model}: cam1 {np.round(ps[0], 2).tolist()}, cam2 {np.round(ps[1], 2).tolist()}; "
                f"held-out tracks |resid| 68 %: cam1 {rows['cam1']['heldout_tracks']['q_h'][1]:.2f}/"
                f"{rows['cam1']['heldout_tracks']['q_v'][1]:.2f}, cam2 {rows['cam2']['heldout_tracks']['q_h'][1]:.2f}/"
                f"{rows['cam2']['heldout_tracks']['q_v'][1]:.2f} px; cue on the held-out frames: true kept "
                f"{100 * cs[0]:.1f} +- {100 * cs[1]:.1f} %, random {100 * cs[2]:.2f} %")
    # the best by the held-out cue: most true pairs with no more random ones than S2's model
    ok_fits = {k: v for k, v in fits.items() if v[1][2] <= base_cue[2] * 1.05}
    best = max(ok_fits or fits, key=lambda k: (ok_fits or fits)[k][1][0])
    ps, cs = fits[best]
    log(f"  best (the most true pairs kept with at most 5 % more random ones than S2's): {best[0]} model {best[1]}: "
        f"true {100 * cs[0]:.1f} % (S2's {100 * base_cue[0]:.1f} %), random {100 * cs[2]:.2f} % ({100 * base_cue[2]:.2f} %)")
    # the tolerance scan of the best (S2's rule: maximise true / sqrt(random))
    scan = []
    for tp in (0.5, 0.6, 0.8, 1.0, 1.2):
        for tr in (0.1, 0.15, 0.2, 0.25, 0.3):
            t, se, r, _n = cue_stats(held_recs, ps, f_px, [best[1], best[1]], (tp, tr))
            scan.append({"tol_px": tp, "tol_rel": tr, "true_kept": t, "true_kept_se": se, "random_kept": r})
    rec_t = max(scan, key=lambda x: x["true_kept"] / math.sqrt(max(x["random_kept"], 1e-9)))
    base_t = max([cue_stats(held_recs, p_s2, f_px, ["A", "A"], (tp, tr)) + (tp, tr)
                  for tp in (0.5, 0.6, 0.8, 1.0, 1.2) for tr in (0.1, 0.15, 0.2, 0.25, 0.3)],
                 key=lambda x: x[0] / math.sqrt(max(x[2], 1e-9)))
    log(f"  tolerances (true / sqrt(random) at its largest): the best model at {rec_t['tol_px']} px / {rec_t['tol_rel']}: "
        f"true {100 * rec_t['true_kept']:.1f} %, random {100 * rec_t['random_kept']:.2f} %; S2's model at "
        f"{base_t[4]} px / {base_t[5]}: true {100 * base_t[0]:.1f} %, random {100 * base_t[2]:.2f} %")
    out["M1"].update({"best": {"data": best[0], "model": best[1], "params": ps, "cue": cs}, "scan": scan,
                      "best_tolerances": rec_t, "s2_best_tolerances": list(base_t)})
    # a fair comparison: each model's true pairs kept at the same random rate (its tolerance scan's upper envelope,
    # interpolated at S2's random rate and at 2 and 5 %)
    tps = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)
    trs = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
    roc = {}
    cand_models = {"S2": (p_s2, "A")}
    for key in (("m15", "A"), ("m15", "C"), ("cue", "A"), ("cue", "C"), best):
        cand_models[f"{key[0]}_{key[1]}"] = (fits[key][0], key[1])
    for nm, (pp, md) in cand_models.items():
        pts = sorted((cue_stats(held_recs, pp, f_px, [md, md], (tp, tr))[2],
                      cue_stats(held_recs, pp, f_px, [md, md], (tp, tr))[0]) for tp in tps for tr in trs)
        env, top = [], -1.0
        for r_, t_ in pts:                      # the upper envelope: the most true pairs at or below each random rate
            if t_ > top:
                env.append((r_, t_))
                top = t_
        er, et = np.array(env).T
        at = {f"{x:.4f}": float(np.interp(x, er, et)) for x in (0.02, base_cue[2], 0.05)}
        roc[nm] = {"envelope": env, "true_at": at}
        log(f"  ROC {nm}: true pairs kept at the same random rate: 2 % -> {100 * at['0.0200']:.1f} %, "
            f"{100 * base_cue[2]:.2f} % -> {100 * at[f'{base_cue[2]:.4f}']:.1f} %, 5 % -> {100 * at['0.0500']:.1f} %")
    out["M1"]["roc"] = roc
    gain = max(v["true_at"][f"{base_cue[2]:.4f}"] for k, v in roc.items() if k != "S2") \
        - roc["S2"]["true_at"][f"{base_cue[2]:.4f}"]
    out["M1"]["roc_gain_at_s2_random"] = gain
    log(f"  the refits' best gain at S2's random rate: {100 * gain:+.1f} points of the true pairs")
    # is the per-particle part a map over the image (e.g. field curvature), which a model could take up?
    D = data["m15"]
    maps = {}
    h1 = D["frame"] <= 125
    for c in (0, 1):
        st, r, ok = residual_stats(D["w"][:, c], D["z"][:, c], f_px[c], p_s2[c], "A")
        recs_m = sc.load_gather("t1_m15")
        uv = np.concatenate([g[f"det{c}"][g["pair"][((g["pair"] >= 0) & (g["pair_dist"] <= GATE)).all(axis=1), c], :2]
                             for g in recs_m if len(g["X"])])[ok]
        for k, axn in ((0, "w_h"), (1, "w_v")):
            n, med, se = sc.cell_stats(uv, r[:, k])
            n1, m1, _s1 = sc.cell_stats(uv[h1[ok]], r[h1[ok], k])
            n2, m2, _s2 = sc.cell_stats(uv[~h1[ok]], r[~h1[ok], k])
            from v0_pattern import corr
            cc, nc = corr(m1, m2)
            stt = sc.map_rms(med, se, n)
            maps[f"cam{c + 1}_{axn}"] = {"median": med.tolist(), "stats": stt, "halves_corr": cc, "cells": nc}
            log(f"  width residual map cam{c + 1} {axn}: {stt['cells']} cells, RMS of the cell medians "
                f"{stt['rms']:.3f} px ({stt['rms_noise_corrected']:.3f} with their noise taken off); halves "
                f"correlation {cc:.2f}")
    out["M0"]["width_maps"] = maps
    # M1b: the widths' map over the image as a model term (a lens's field curvature changes the blur across the field
    # at one depth, which the model's z_f cannot): per camera and axis a smooth map (B-splines 10 x 6, bending 1e4,
    # Cauchy) of the residual against S2's model, fitted on the even tracks of the 15 px run, subtracted from every
    # held-out candidate pair's measured widths, then the cue's ROC as above
    wmaps = {}
    even = np.array([k[1] % 2 == 0 for k in D["keys"]])
    for c in (0, 1):
        st, r, ok = residual_stats(D["w"][:, c], D["z"][:, c], f_px[c], p_s2[c], "A")
        recs_m = sc.load_gather("t1_m15")
        uv = np.concatenate([g[f"det{c}"][g["pair"][((g["pair"] >= 0) & (g["pair_dist"] <= GATE)).all(axis=1), c], :2]
                             for g in recs_m if len(g["X"])])[ok]
        ev = even[ok]
        for k in (0, 1):
            wmaps[(c, k)] = fit_map(uv[ev], r[ev, k])
            # held out on the odd tracks: the residual's |.| 68 % before and after the map
            pred = eval_map(wmaps[(c, k)], uv[~ev])
            q0 = float(np.percentile(np.abs(r[~ev, k]), 68))
            q1 = float(np.percentile(np.abs(r[~ev, k] - pred), 68))
            log(f"  M1b map cam{c + 1} {'w_h' if k == 0 else 'w_v'}: its RMS {np.sqrt(np.mean(pred ** 2)):.3f} px at the "
                f"odd tracks; their |residual| 68 % {q0:.3f} -> {q1:.3f} px")
    roc_m = {}
    for nm, use in (("S2+maps", {(0, 0), (0, 1), (1, 0), (1, 1)}), ("S2+cam1_wh_map", {(0, 0)})):
        pts = sorted((cue_stats(held_recs, p_s2, f_px, ["A", "A"], (tp, tr), wmaps, use)[2],
                      cue_stats(held_recs, p_s2, f_px, ["A", "A"], (tp, tr), wmaps, use)[0]) for tp in tps for tr in trs)
        env, top = [], -1.0
        for r_, t_ in pts:
            if t_ > top:
                env.append((r_, t_))
                top = t_
        er, et = np.array(env).T
        at = {f"{x:.4f}": float(np.interp(x, er, et)) for x in (0.02, base_cue[2], 0.05)}
        roc_m[nm] = {"envelope": env, "true_at": at}
        log(f"  ROC {nm}: true pairs kept at 2 % random {100 * at['0.0200']:.1f} %, at "
            f"{100 * base_cue[2]:.2f} % {100 * at[f'{base_cue[2]:.4f}']:.1f} %, at 5 % {100 * at['0.0500']:.1f} % "
            f"(S2 alone: {100 * roc['S2']['true_at']['0.0200']:.1f}, {100 * roc['S2']['true_at'][f'{base_cue[2]:.4f}']:.1f}, "
            f"{100 * roc['S2']['true_at']['0.0500']:.1f} %)")
    out["M1"]["roc_maps"] = roc_m
    judged = cs[0] >= 0.65 and cs[2] <= base_cue[2] * 1.05
    out["M1"]["judged_ge_65"] = judged
    log(f"M1 judged: {'PASS' if judged else 'not met'} (the plan's bar: >= 65 % of the true pairs with no more random "
        f"ones; S2's {100 * base_cue[0]:.1f} %)")
    sc.save_json(os.path.join(sc.DATA, "shape", "m01_best.json"), out["M1"]["best"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    D = data["cue"]
    for c in (0, 1):
        for k, ax_ in ((0, "w_h vs z_tan"), (1, "w_v vs z_sag")):
            a = ax[c, k]
            ok = np.all(np.isfinite(D["z"][:, c]), axis=1)
            x = D["z"][ok, c, k] - s2[c]["z_f_mm"]
            a.scatter(x, D["w"][ok, c, k], s=1, alpha=0.2, c="0.5", label="measured (loosened-cue tracks)")
            xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 200)
            zz = np.column_stack([xs + s2[c]["z_f_mm"], xs + s2[c]["z_f_mm"]])
            a.plot(xs, widths(p_s2[c], f_px[c], zz, "A")[:, k], "r-", label="S2's model")
            a.plot(xs, widths(ps[c], f_px[c], zz, best[1])[:, k], "b-", label=f"refit ({best[0]} {best[1]})")
            a.set_xlabel(f"cam{c + 1}: z - z_f [mm]")
            a.set_ylabel(f"{ax_.split()[0]} [px]")
            a.set_ylim(0, 8)
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, "m01_shape.png"), dpi=100)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
