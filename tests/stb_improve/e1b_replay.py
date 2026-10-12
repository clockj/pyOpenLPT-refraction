"""E1b and E2 (Phase E): replay one-step track predictions on the saved tracks of B300, T4 and T5.

For every saved long track (active at the last save, left the volume, lost), every point from the fifth on (index
>= 4, the points STB got by shaking a prediction) is predicted from the points before it by:
  W          today's STB::predictNext (predictors.wiener, identical to the C++ to the last bit: E1a)
  MD         W's limit far from the origin, the mean step over the last m = min(n-1, 5) frames
  LIN-k      least-squares line through the last k points (k = 3, 4, 6, 8, 10, 15)
  QUAD-k     least-squares parabola through the last k points (k = 5, 6, 8, 10, 15, 20)
  KF-CV/CA   Kalman filters (constant velocity / acceleration) over the whole track from a diffuse start
The last point of a lost track that failed the linear-fit test (a "jump") is not scored with the others; for the
lost tracks of >= 9 points the script looks at where each predictor placed it.

E2: the tracks are split at random in two halves (by track).  On half A the Kalman ratio lam = q/r is scanned for
each model and chosen by the mean squared one-step error, k is chosen likewise for LIN and QUAD; everything is then
scored on half B.  The innovation likelihood (which fixes q and r together) and the position noise of the tracks
(T6b's second-difference estimate) are the cross-checks of r.  Each run's lam is applied to the other runs.

Writes results/e1b_replay.txt, results/e1b_replay.json and figures results/e1b_*.png.
    C:\\Users\\zcloc\\.conda\\envs\\refract\\python.exe tests/stb_improve/e1b_replay.py [runs...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predictors as pr  # noqa: E402

RUNS = ("b300", "t4", "t5")
SEED = 20261006
LAM_GRID = 10.0 ** np.arange(-7.0, 1.01, 0.25)
LIN_K = (3, 4, 6, 8, 10, 15)
QUAD_K = (5, 6, 8, 10, 15, 20)
THRESH = (0.032, 0.064, 0.1)          # the shake's first step, about 1 px, 0.1 mm
LF_LIMIT = 0.05                       # MAX_ERR_LINEARFIT
HIST_BINS = ((4, 4), (5, 5), (6, 6), (7, 10), (11, 30), (31, 10 ** 6))
BANDS = ((0, 1), (1, 3), (3, 10), (10, 1e9))
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def metrics(e):
    """e: 3-D error norms."""
    if len(e) == 0:
        return {"n": 0}
    q = np.percentile(e, [50, 90, 99, 99.9])
    out = {"n": int(len(e)), "median": float(q[0]), "rms": float(np.sqrt(np.mean(e.astype(float) ** 2))),
           "p90": float(q[1]), "p99": float(q[2]), "p999": float(q[3])}
    for t in THRESH:
        out[f"gt_{t:g}"] = float(np.mean(e > t))
    return out


def axis_metrics(ea):
    """ea: per-axis signed errors, flattened."""
    if len(ea) == 0:
        return {"n": 0}
    a = np.abs(ea)
    q = np.percentile(a, [50, 99])
    return {"n": int(len(a)), "median_abs": float(q[0]), "p99_abs": float(q[1]), "mean": float(np.mean(ea)),
            "rms": float(np.sqrt(np.mean(ea.astype(float) ** 2)))}


def position_noise(tr, exclude):
    """Per-track noise per axis from second differences (T6b): 1.4826 MAD / sqrt(6); tracks of >= 10 points."""
    import pandas as pd
    ii = tr.index_in_track
    g = np.nonzero((ii >= 2) & ~exclude)[0]
    d2 = tr.P[g] - 2 * tr.P[g - 1] + tr.P[g - 2]
    t = tr.track_of_point[g]
    keep = tr.lengths[t] >= 10
    df = pd.DataFrame(d2[keep], columns=["x", "y", "z"])
    df["t"] = t[keep]
    med = df.groupby("t")[["x", "y", "z"]].transform("median")
    dev = (df[["x", "y", "z"]] - med).abs()
    dev["t"] = df["t"]
    mad = dev.groupby("t")[["x", "y", "z"]].median()
    sig = 1.4826 * mad / np.sqrt(6.0)
    return sig.median().to_numpy(), len(sig)


class Run:
    def __init__(self, name):
        t0 = time.time()
        self.name = name
        self.tr = tr = pr.load_run(name)
        self.ii = ii = tr.index_in_track
        tp = tr.track_of_point
        L = tr.lengths[tp]
        kind = tr.kind[tp]
        last = ii == L - 1
        lost_last = np.nonzero(last & (kind == 2))[0]
        res = pr.linear_fit_residual(tr.P, lost_last)
        self.jump_any = np.zeros(len(tr.P), bool)
        self.jump_any[lost_last[res > LF_LIMIT]] = True
        self.lost_last_resid = res
        self.U = np.nonzero(ii >= 4)[0]                       # every point a prediction was made for
        self.scored_U = ~self.jump_any[self.U]                 # within U
        self.jumps_U = self.jump_any[self.U] & (L[self.U] >= 9)
        rng = np.random.default_rng(SEED)
        half = rng.random(len(tr.starts)) < 0.5                # True: half A (fit)
        self.A_U = half[tp[self.U]]
        self.n_hist = ii[self.U]                               # points before each
        log(f"\n=== {name}: {pr.RUN_FILES[name][2]}")
        log(f"  {len(tr.starts)} saved long tracks (active {np.sum(tr.kind == 0)}, left the volume "
            f"{np.sum(tr.kind == 1)}, lost {np.sum(tr.kind == 2)}), {len(tr.P)} points; scored "
            f"{int(self.scored_U.sum())} (half A {int((self.scored_U & self.A_U).sum())}, half B "
            f"{int((self.scored_U & ~self.A_U).sum())})")
        log(f"  lost tracks ending with a linear-fit failure: {int(np.sum(res > LF_LIMIT))} of {len(res)}; with >= 9 "
            f"points (the jumps examined): {int(self.jumps_U.sum())}  [{time.time() - t0:.0f} s]")

    def err(self, pred_U):
        return pred_U - self.tr.P[self.U]


def scan_kalman(run, kind):
    """lam scan: MSE, p99 and median on both halves, and the profile likelihood on half A."""
    rows = []
    sA = run.scored_U & run.A_U
    sB = run.scored_U & ~run.A_U
    for lam in LAM_GRID:
        kf = pr.Kalman(kind, lam)
        pred, svar = kf.run(run.tr)
        e = run.err(pred[run.U])
        n3 = np.linalg.norm(e, axis=1)
        s = svar[run.U]
        z = (e[sA] ** 2 / s[sA, None]).ravel()
        r_ml = float(np.mean(z))
        nll = float(0.5 * np.mean(np.log(2 * np.pi * r_ml * np.repeat(s[sA], 3)) + z / r_ml))
        rows.append({"lam": float(lam),
                     "A": {"mse": float(np.mean(n3[sA] ** 2)), "p99": float(np.percentile(n3[sA], 99)),
                           "median": float(np.median(n3[sA]))},
                     "B": {"mse": float(np.mean(n3[sB] ** 2)), "p99": float(np.percentile(n3[sB], 99)),
                           "median": float(np.median(n3[sB]))},
                     "ml": {"nll_per_sample": nll, "r": r_ml, "sigma": float(np.sqrt(r_ml))}})
    return rows


def volume_grid(P_U, val, lo, hi, nb=6):
    """Median and p99 of val in nb x nb bins of each coordinate plane."""
    out = {}
    for name, (a, b) in (("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))):
        ia = np.clip(((P_U[:, a] - lo[a]) / (hi[a] - lo[a]) * nb).astype(int), 0, nb - 1)
        ib = np.clip(((P_U[:, b] - lo[b]) / (hi[b] - lo[b]) * nb).astype(int), 0, nb - 1)
        med = np.full((nb, nb), np.nan)
        p99 = np.full((nb, nb), np.nan)
        cnt = np.zeros((nb, nb), int)
        key = ia * nb + ib
        order = np.argsort(key, kind="stable")
        ks = key[order]
        bounds = np.searchsorted(ks, np.arange(nb * nb + 1))
        for k in range(nb * nb):
            v = val[order[bounds[k]:bounds[k + 1]]]
            cnt.flat[k] = len(v)
            if len(v) >= 200:
                med.flat[k] = np.median(v)
                p99.flat[k] = np.percentile(v, 99)
        out[name] = {"median": med.tolist(), "p99": p99.tolist(), "count": cnt.tolist()}
    return out


def analyse(name, figs):
    run = Run(name)
    tr, U = run.tr, run.U
    out = {"description": pr.RUN_FILES[name][2], "n_tracks": int(len(tr.starts)), "n_points": int(len(tr.P)),
           "n_scored": int(run.scored_U.sum()), "n_lost": int(np.sum(tr.kind == 2)),
           "n_lost_lf_fail": int(np.sum(run.lost_last_resid > LF_LIMIT)), "n_jumps": int(run.jumps_U.sum())}
    sig, nt = position_noise(tr, run.jump_any)
    out["noise_sigma_mm"] = sig.tolist()
    log(f"  position noise per axis (median over {nt} tracks of >= 10 points): "
        f"{sig[0]:.4f} / {sig[1]:.4f} / {sig[2]:.4f} mm (x / y / z)")

    sA = run.scored_U & run.A_U
    sB = run.scored_U & ~run.A_U
    keep = {}        # per-axis errors of the predictors kept for the splits (float32)
    norms = {}       # 3-D error norms of every predictor (float32)

    def add(label, pred_U, keep_axes=False):
        e = run.err(pred_U)
        norms[label] = np.linalg.norm(e, axis=1).astype(np.float32)
        if keep_axes:
            keep[label] = e.astype(np.float32)

    t0 = time.time()
    add("W", pr.wiener(tr.P, U, run.n_hist), keep_axes=True)
    add("MD", pr.mean_step(tr.P, U, run.n_hist), keep_axes=True)
    for k in LIN_K:
        add(f"LIN-{k}", pr.poly(tr.P, U, run.n_hist, 1, k))
    for k in QUAD_K:
        add(f"QUAD-{k}", pr.poly(tr.P, U, run.n_hist, 2, k))
    log(f"  W, MD, LIN, QUAD predictions [{time.time() - t0:.0f} s]")

    # E2: choose k and lam on half A by the mean squared error
    chosen = {}
    for fam, ks in (("LIN", LIN_K), ("QUAD", QUAD_K)):
        mse = {k: float(np.mean(norms[f"{fam}-{k}"][sA].astype(float) ** 2)) for k in ks}
        chosen[fam] = min(mse, key=mse.get)
        out[f"mse_A_{fam}"] = mse
    scans = {}
    for kind in ("cv", "ca"):
        t0 = time.time()
        rows = scan_kalman(run, kind)
        scans[kind] = rows
        i_mse = int(np.argmin([r["A"]["mse"] for r in rows]))
        i_p99 = int(np.argmin([r["A"]["p99"] for r in rows]))
        i_ml = int(np.argmin([r["ml"]["nll_per_sample"] for r in rows]))
        chosen[f"KF-{kind.upper()}"] = rows[i_mse]["lam"]
        out[f"lam_{kind}"] = {"mse": rows[i_mse]["lam"], "p99": rows[i_p99]["lam"], "ml": rows[i_ml]["lam"],
                              "ml_sigma_mm": rows[i_ml]["ml"]["sigma"],
                              "edge_of_grid": i_mse in (0, len(rows) - 1)}
        log(f"  KF-{kind.upper()}: lam by MSE {rows[i_mse]['lam']:.3g}, by p99 {rows[i_p99]['lam']:.3g}, by the "
            f"innovation likelihood {rows[i_ml]['lam']:.3g} (its sigma {rows[i_ml]['ml']['sigma']:.4f} mm)"
            f"{'  [EDGE OF GRID]' if i_mse in (0, len(rows) - 1) else ''}  [{time.time() - t0:.0f} s]")
    out["scan"] = scans
    out["chosen"] = chosen
    for kind in ("cv", "ca"):
        kf = pr.Kalman(kind, chosen[f"KF-{kind.upper()}"])
        pred, _ = kf.run(tr)
        add(f"KF-{kind.upper()}", pred[U], keep_axes=True)
    best_lin, best_quad = f"LIN-{chosen['LIN']}", f"QUAD-{chosen['QUAD']}"
    for lab in (best_lin, best_quad):
        e = run.err(pr.poly(tr.P, U, run.n_hist, 1 if lab.startswith("LIN") else 2, int(lab.split("-")[1])))
        keep[lab] = e.astype(np.float32)
    main = ["W", "MD", best_lin, best_quad, "KF-CV", "KF-CA"]
    out["main"] = main

    # scores on the test half (and W on all scored points)
    out["test"] = {lab: metrics(n[sB]) for lab, n in norms.items()}
    out["all_scored"] = {lab: metrics(norms[lab][run.scored_U]) for lab in main}
    log(f"\n  one-step 3-D error on half B ({int(sB.sum())} points), mm; shares above 0.032 / 0.064 / 0.1 mm:")
    log(f"  {'predictor':10s} {'median':>8s} {'RMS':>8s} {'p90':>8s} {'p99':>8s} {'p99.9':>8s}   "
        f"{'>0.032':>7s} {'>0.064':>7s} {'>0.1':>7s}")
    for lab in norms:
        m = out["test"][lab]
        star = " *" if lab in main else ""
        log(f"  {lab:10s} {m['median']:8.4f} {m['rms']:8.4f} {m['p90']:8.4f} {m['p99']:8.4f} {m['p999']:8.4f}   "
            f"{100 * m['gt_0.032']:6.2f}% {100 * m['gt_0.064']:6.2f}% {100 * m['gt_0.1']:6.2f}%{star}")
    alts = [lab for lab in main if lab not in ("W",)]
    best = min(alts, key=lambda lab: out["test"][lab]["p99"])
    out["best_alternative"] = best
    gain = 1 - out["test"][best]["p99"] / out["test"]["W"]["p99"]
    out["p99_gain_best"] = gain
    log(f"  best alternative by p99: {best}, p99 {out['test'][best]['p99']:.4f} against W's "
        f"{out['test']['W']['p99']:.4f} mm ({100 * gain:+.1f} % lower)")

    # splits
    nh = run.n_hist
    out["by_history"] = {}
    for lab in main:
        out["by_history"][lab] = {f"{a}-{b}" if b < 10 ** 6 else f">{a - 1}":
                                  metrics(norms[lab][run.scored_U & (nh >= a) & (nh <= b)]) for a, b in HIST_BINS}
    log("\n  by the number of points before the prediction (all scored points): median / p99, mm")
    log("  " + f"{'points':>8s} " + " ".join(f"{lab:>17s}" for lab in main))
    for a, b in HIST_BINS:
        key = f"{a}-{b}" if b < 10 ** 6 else f">{a - 1}"
        cells = []
        for lab in main:
            m = out["by_history"][lab][key]
            cells.append(f"{m['median']:.4f} / {m['p99']:.4f}" if m["n"] else "-")
        log(f"  {key:>8s} " + " ".join(f"{c:>17s}" for c in cells) + f"   n={out['by_history']['W'][key]['n']}")

    last = tr.P[U - 1]                                     # the point W's shift rule looks at
    out["by_band"] = {}
    log("\n  per axis, by the last point's distance from 0 on that axis (all scored points): "
        "mean signed error x sign(coordinate) / p99 |error|, mm")
    log("  " + f"{'band':>9s} " + " ".join(f"{lab:>17s}" for lab in main) + "   share of axis samples")
    for lo, hi in BANDS:
        key = f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}"
        cells, res = [], {}
        for lab in main:
            ea, sg = [], []
            for ax in range(3):
                c = np.abs(last[:, ax])
                sel = run.scored_U & (c >= lo) & (c < hi)
                ea.append(keep[lab][sel, ax])
                sg.append(np.sign(last[sel, ax]))
            ea, sg = np.concatenate(ea), np.concatenate(sg)
            m = axis_metrics(ea)
            m["mean_outward"] = float(np.mean(ea * sg)) if len(ea) else float("nan")
            res[lab] = m
            cells.append(f"{m['mean_outward']:+.4f} / {m['p99_abs']:.4f}")
        out["by_band"][key] = res
        log(f"  {key:>9s} " + " ".join(f"{c:>17s}" for c in cells) +
            f"   {res['W']['n'] / (3 * run.scored_U.sum()):.3f}")

    # over the view volume
    P_U = tr.P[U][run.scored_U]
    lo, hi = P_U.min(axis=0), P_U.max(axis=0)
    out["volume"] = {"lo": lo.tolist(), "hi": hi.tolist(),
                     "W": volume_grid(P_U, norms["W"][run.scored_U], lo, hi),
                     best: volume_grid(P_U, norms[best][run.scored_U], lo, hi)}
    rat = []
    for plane in ("xy", "xz", "yz"):
        a = np.array(out["volume"]["W"][plane]["p99"])
        b = np.array(out["volume"][best][plane]["p99"])
        ok = np.isfinite(a) & np.isfinite(b)
        rat.append(b[ok] / a[ok])
    rat = np.concatenate(rat)
    out["volume"]["p99_ratio_best_to_W"] = {"min": float(rat.min()), "median": float(np.median(rat)),
                                            "max": float(rat.max()), "n_bins": int(len(rat))}
    wp = np.concatenate([np.array(out["volume"]["W"][p]["p99"]).ravel() for p in ("xy", "xz", "yz")])
    wp = wp[np.isfinite(wp)]
    log(f"\n  over the view volume (6 x 6 bins per plane, >= 200 points): W's p99 {wp.min():.4f}-{wp.max():.4f} mm; "
        f"{best}'s p99 / W's p99 per bin: {rat.min():.2f}-{rat.max():.2f} (median {np.median(rat):.2f})")

    # the jumps
    J = run.jumps_U
    jd = {}
    preds_at_jumps = {}
    for lab in main:
        if lab == "W":
            p = pr.wiener(tr.P, U[J], run.n_hist[J])
        elif lab == "MD":
            p = pr.mean_step(tr.P, U[J], run.n_hist[J])
        elif lab.startswith("LIN") or lab.startswith("QUAD"):
            p = pr.poly(tr.P, U[J], run.n_hist[J], 1 if lab.startswith("LIN") else 2, int(lab.split("-")[1]))
        else:
            p = tr.P[U[J]] + keep[lab][J]
        preds_at_jumps[lab] = p
    PJ = tr.P[U[J]]
    for lab in main:
        d = np.linalg.norm(PJ - preds_at_jumps[lab], axis=1)
        dw = np.linalg.norm(preds_at_jumps["W"] - preds_at_jumps[lab], axis=1)
        dw_ord = np.linalg.norm(keep["W"][run.scored_U] - keep[lab][run.scored_U], axis=1) if lab != "W" else None
        jd[lab] = {"dist_last_point": {"p10": float(np.percentile(d, 10)), "median": float(np.median(d)),
                                       "p90": float(np.percentile(d, 90)),
                                       "share_lt_0.032": float(np.mean(d < 0.032))},
                   "dist_from_W_at_jumps": {"median": float(np.median(dw)), "p90": float(np.percentile(dw, 90)),
                                            "p99": float(np.percentile(dw, 99)),
                                            "share_gt_0.032": float(np.mean(dw > 0.032))}}
        if dw_ord is not None:
            jd[lab]["dist_from_W_ordinary"] = {"median": float(np.median(dw_ord)),
                                               "p99": float(np.percentile(dw_ord, 99)),
                                               "share_gt_0.032": float(np.mean(dw_ord > 0.032))}
    dW = np.linalg.norm(PJ - preds_at_jumps["W"], axis=1)
    dB = np.linalg.norm(PJ - preds_at_jumps[best], axis=1)
    sep = np.linalg.norm(preds_at_jumps["W"] - preds_at_jumps[best], axis=1)
    caused = (dW < 0.032) & (dB > 0.064) & (sep > 0.032)
    jd["candidates_caused_by_W"] = {"n": int(caused.sum()), "share": float(np.mean(caused)) if len(caused) else 0.0,
                                    "rule": "last point within 0.032 mm of W's prediction, more than 0.064 mm from "
                                            f"{best}'s, and the two predictions more than 0.032 mm apart"}
    ordW = norms["W"][run.scored_U]
    jd["W_error_ordinary"] = {"median": float(np.median(ordW)), "p99": float(np.percentile(ordW, 99))}
    out["jumps"] = jd
    log(f"\n  the jumps ({int(J.sum())} lost tracks of >= 9 points whose last point failed the linear-fit test):")
    log(f"  {'predictor':10s} {'|last - pred|: p10':>19s} {'median':>8s} {'p90':>8s} {'<0.032':>7s}   "
        f"{'|pred - W| at jumps: median':>28s} {'p99':>8s}   {'ordinary: median':>17s} {'p99':>8s}")
    for lab in main:
        a, b = jd[lab]["dist_last_point"], jd[lab]["dist_from_W_at_jumps"]
        o = jd[lab].get("dist_from_W_ordinary")
        log(f"  {lab:10s} {a['p10']:19.4f} {a['median']:8.4f} {a['p90']:8.4f} {100 * a['share_lt_0.032']:6.1f}%   "
            f"{b['median']:28.4f} {b['p99']:8.4f}   " + (f"{o['median']:17.4f} {o['p99']:8.4f}" if o else ""))
    log(f"  W's error on ordinary points: median {jd['W_error_ordinary']['median']:.4f}, "
        f"p99 {jd['W_error_ordinary']['p99']:.4f} mm")
    log(f"  jumps that W's prediction may have caused ({jd['candidates_caused_by_W']['rule']}): "
        f"{caused.sum()} ({100 * jd['candidates_caused_by_W']['share']:.1f} %)")

    figs[name] = {"norms": {lab: norms[lab][sB] for lab in main},
                  "scan": scans, "chosen": chosen,
                  "band": out["by_band"], "main": main}
    return out


def transfer(results):
    """Each run's chosen lam applied to the other runs (half B MSE and p99, relative to the run's own choice)."""
    out = {}
    for kind in ("cv", "ca"):
        tab = {}
        for src in results:
            lam = results[src]["chosen"][f"KF-{kind.upper()}"]
            row = {}
            for dst in results:
                rows = results[dst]["scan"][kind]
                i = int(np.argmin([abs(np.log(r["lam"] / lam)) for r in rows]))
                own = results[dst]["chosen"][f"KF-{kind.upper()}"]
                j = int(np.argmin([abs(np.log(r["lam"] / own)) for r in rows]))
                row[dst] = {"mse_ratio": rows[i]["B"]["mse"] / rows[j]["B"]["mse"],
                            "p99_ratio": rows[i]["B"]["p99"] / rows[j]["B"]["p99"]}
            tab[src] = {"lam": lam, "on": row}
        out[kind] = tab
    return out


def make_figures(figs, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # fixed categorical order (Okabe-Ito, colour-blind safe), one colour per predictor family
    col = {"W": "#000000", "MD": "#999999", "LIN": "#0072B2", "QUAD": "#009E73", "KF-CV": "#E69F00",
           "KF-CA": "#D55E00"}

    def colour(lab):
        return col.get(lab, col.get(lab.split("-")[0], "#56B4E9"))
    names = list(figs)
    fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, name in zip(axs, names):
        for lab in figs[name]["main"]:
            e = np.sort(figs[name]["norms"][lab])
            y = 1 - np.arange(1, len(e) + 1) / len(e)
            ax.loglog(e, np.maximum(y, 1e-6), color=colour(lab), lw=1.5, label=lab)
        for t in THRESH:
            ax.axvline(t, color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel("one-step error (mm)")
        ax.set_title(f"{name.upper()} (test half)")
        ax.grid(True, which="major", color="#eeeeee", lw=0.6)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    axs[0].set_ylabel("share of points with a larger error")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "e1b_error_tails.png"), dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, kind in zip(axs, ("cv", "ca")):
        for name in names:
            rows = figs[name]["scan"][kind]
            lam = [r["lam"] for r in rows]
            p99 = [r["B"]["p99"] for r in rows]
            line, = ax.semilogx(lam, p99, lw=1.5, label=name.upper())
            ax.axhline(results[name]["test"]["W"]["p99"], color=line.get_color(), lw=0.8, ls="--")
            ax.plot([figs[name]["chosen"][f"KF-{kind.upper()}"]],
                    [rows[int(np.argmin([abs(np.log(r["lam"] / figs[name]["chosen"][f"KF-{kind.upper()}"]))
                                         for r in rows]))]["B"]["p99"]], "o", color=line.get_color(), ms=5)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 3)
        ax.set_xlabel("q / r")
        ax.set_ylabel("p99 one-step error, test half (mm)")
        ax.set_title(f"KF-{kind.upper()} (dashed: today's predictor; dot: chosen on half A)", fontsize=9)
        ax.grid(True, color="#eeeeee", lw=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "e1b_kalman_scan.png"), dpi=130)
    plt.close(fig)


def main():
    runs = sys.argv[1:] or list(RUNS)
    t0 = time.time()
    results, figs = {}, {}
    for name in runs:
        results[name] = analyse(name, figs)
    if len(results) > 1:
        results["_transfer"] = tr = transfer({k: v for k, v in results.items() if not k.startswith("_")})
        log("\n=== transfer: each run's lam on the others (half B; p99 / MSE relative to the run's own lam)")
        for kind in ("cv", "ca"):
            for src, row in tr[kind].items():
                log(f"  KF-{kind.upper()} lam from {src} ({row['lam']:.3g}): " + ", ".join(
                    f"on {dst} p99 x{v['p99_ratio']:.3f}, MSE x{v['mse_ratio']:.3f}" for dst, v in row["on"].items()))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    make_figures(figs, results)
    # decision rule of the plan (E3): best alternative's p99 at least 30 % below W's on every run
    real = [k for k in results if not k.startswith("_")]
    gains = {k: results[k]["p99_gain_best"] for k in real}
    results["_decision"] = {"p99_gain_best": gains, "all_ge_30pct": all(g >= 0.30 for g in gains.values()),
                            "jump_candidates_share": {k: results[k]["jumps"]["candidates_caused_by_W"]["share"]
                                                      for k in real}}
    log("\n=== E3 threshold (plan): the best alternative's p99 at least 30 % below W's on every run: "
        + ", ".join(f"{k} {100 * g:+.1f} % ({results[k]['best_alternative']})" for k, g in gains.items())
        + f" -> {'MET' if results['_decision']['all_ge_30pct'] else 'not met'}")
    log(f"(total {time.time() - t0:.0f} s)")
    with open(os.path.join(HERE, "results", "e1b_replay.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(_lines) + "\n")
    with open(os.path.join(HERE, "results", "e1b_replay.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
