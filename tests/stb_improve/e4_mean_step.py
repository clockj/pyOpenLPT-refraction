"""E4 (Phase E, approved 6 Oct): test the origin-free predictor (the mean step) that replaces the LMS fit.

    port      E4a: the `md` build's STB.predictNext against predictors.mean_step on E1a's 20,104 histories (to the last
              bit), and shifted histories: the prediction must move by exactly the shift (1e-12 mm); today's
              predictor (predictors.wiener, identical to today's C++: E1a) for contrast.
    compare   E4b-d: b300_md (the mean step) and b300_wjit (the control: today's predictor plus an offset as large as
              |W - MD| in a pseudo-random direction) against b300_final3 (today's), frames 0-300:
                - frames 0-3 identical to b300_final3 (no prediction there), first frame that differs;
                - per-frame counters from frame 4 (active long tracks, added, removed, linear-fit and shaking
                  failures, IPR objects, exits, time) and the saved tracks (long tracks, tracked points, lengths);
                - near the coordinate planes, per axis by the distance from 0: the prediction's mean error away from 0
                  and the shake's move from it (each run with its own predictor), linear-fit losses per tracked
                  point, tracked points per mm.

Writes results/e4_port.txt/.json or results/e4_compare.txt/.json.
    C:\\Users\\zcloc\\.conda\\envs\\refract\\python.exe tests/stb_improve/e4_mean_step.py port|compare
"""
from __future__ import annotations

import csv
import filecmp
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import predictors as pr  # noqa: E402

RUNS = os.path.join(HERE, "runs")
LF_LIMIT = 0.05
BANDS = ((0, 1), (1, 3), (3, 10), (10, 1e9))
_lines = []
results = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def check(name, ok, detail):
    results.setdefault("checks", {})[name] = {"ok": bool(ok), "detail": detail}
    log(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def save(stem):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{stem}.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(_lines) + "\n")
    with open(os.path.join(HERE, "results", f"{stem}.json"), "w") as f:
        json.dump(results, f, indent=1)


# ------------------------------------------------------------------------------------------------------- E4a
def histories():
    """E1a's histories: the same seed and procedure (19,933 sampled from B300) plus its 171 edge cases."""
    import e1a_port_check as e1a
    tr = pr.load_run("b300")
    rng = np.random.default_rng(20261006)
    n_hist = rng.integers(3, 8, size=20000)
    tidx = rng.integers(0, len(tr.starts), size=20000)
    keep = tr.lengths[tidx] >= n_hist
    n_hist, tidx = n_hist[keep], tidx[keep]
    off = (rng.random(len(tidx)) * (tr.lengths[tidx] - n_hist + 1)).astype(int)
    starts = tr.starts[tidx] + off
    H = [tr.P[s:s + n] for s, n in zip(starts, n_hist)]
    E = e1a.edge_histories()
    return H + E, len(E)


def flat_eval(hists, shift=0.0):
    flat = np.concatenate([np.vstack([h + shift, np.zeros((1, 3))]) for h in hists])
    lens = np.array([len(h) + 1 for h in hists])
    st = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return flat, st + lens - 1, lens - 1


def port(build):
    t0 = time.time()
    bdir = os.path.join(REPO, "build-stb", build, "out")
    sys.path.insert(0, bdir)
    import pyopenlpt as lpt
    import e1a_port_check as e1a
    assert os.path.normcase(os.path.dirname(lpt.__file__)) == os.path.normcase(bdir), lpt.__file__
    info = json.load(open(os.path.join(bdir, "build_info.json")))
    log(f"E4a: module {lpt.__file__} (build {build}: HEAD {info['git_head']} + {info['git_diff_stat'].splitlines()[-1]})")
    stb = e1a.make_stb(lpt)
    hists, n_edge = histories()
    cpp = np.array([e1a.cpp_predict(lpt, stb, h) for h in hists])
    flat, g, n = flat_eval(hists)
    md = pr.mean_step(flat, g, n)
    same = np.all(cpp == md, axis=1)
    d = np.abs(cpp - md).max()
    check("E4a-1 the md build's predictNext = the mean step", same.all(),
          f"identical to the last bit: {same.sum()} of {len(hists)} histories ({len(hists) - n_edge} sampled, "
          f"{n_edge} edge cases); max |diff| {d:.3g} mm")
    W = pr.wiener(flat, g, n)
    dw_all = np.linalg.norm(W - md, axis=1)
    dw = dw_all[:len(hists) - n_edge]
    log(f"    today's predictor against the mean step on the sampled histories: median {np.median(dw):.4f}, p99 "
        f"{np.percentile(dw, 99):.4f}, max {dw.max():.4f} mm (edge cases: max {dw_all[-n_edge:].max():.4f} mm, from "
        f"the fallback branch and the 10 mm shift)")
    shifts = (0.5, -0.5, 1.0, -1.0, 7.3, -7.3, 100.0, -100.0)
    sub = list(range(0, len(hists) - n_edge, 10)) + list(range(len(hists) - n_edge, len(hists)))
    hs = [hists[i] for i in sub]
    base = cpp[sub]
    worst_md, worst_w = 0.0, 0.0
    fsub, gsub, nsub = flat_eval(hs)
    W0 = pr.wiener(fsub, gsub, nsub)
    per = {}
    n_s = len(hs) - n_edge                     # the sampled histories come first in hs
    worst_w_sampled = 0.0
    for C in shifts:
        cs = np.array([e1a.cpp_predict(lpt, stb, h + C) for h in hs]) - C
        dm = float(np.abs(cs - base).max())
        fs, gs, ns = flat_eval(hs, C)
        dWa = np.abs(pr.wiener(fs, gs, ns) - C - W0).max(axis=1)
        dW, dWs = float(dWa.max()), float(dWa[:n_s].max())
        per[f"{C:g}"] = {"md_max_mm": dm, "wiener_max_mm": dW, "wiener_max_sampled_mm": dWs}
        worst_md, worst_w = max(worst_md, dm), max(worst_w, dW)
        worst_w_sampled = max(worst_w_sampled, dWs)
    check("E4a-2 shifting a history shifts the md prediction by the same amount", worst_md <= 1e-12,
          f"max |p(P + C) - C - p(P)| {worst_md:.3g} mm over C = {', '.join(f'{c:g}' for c in shifts)} mm and "
          f"{len(hs)} histories ({n_s} sampled, {n_edge} edge cases); today's predictor moves by up to "
          f"{worst_w_sampled:.4f} mm on the sampled ones ({worst_w:.1f} mm on the edge cases)")
    results["shift"] = per
    results["wiener_vs_md"] = {"median": float(np.median(dw)), "p99": float(np.percentile(dw, 99)),
                               "max": float(dw.max())}
    n_fail = sum(not c["ok"] for c in results["checks"].values())
    log(f"E4a: {len(results['checks']) - n_fail} of {len(results['checks'])} checks passed ({time.time() - t0:.0f} s)")
    save("e4_port")
    return n_fail


# ---------------------------------------------------------------------------------------------------- E4b-d
def read_per_frame(run):
    with open(os.path.join(RUNS, run, "per_frame.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


TIME_KEYS = ("t_",)
COUNTERS = ("ipr_found", "la_n", "la_add", "la_rm", "fail_lf", "fail_shake", "ex_add", "li_add", "sa_n", "sa_add")


def track_stats(tr):
    L = tr.lengths
    out = {"long_tracks": int(len(L)), "points": int(L.sum()), "active": int(np.sum(tr.kind == 0)),
           "exited": int(np.sum(tr.kind == 1)), "lost": int(np.sum(tr.kind == 2))}
    for k in (50, 100, 200):
        out[f"tracks_ge_{k}"] = int(np.sum(L >= k))
        out[f"points_in_ge_{k}"] = int(L[L >= k].sum())
    out["median_length"] = float(np.median(L))
    return out


def bands(tr, predict):
    """Per axis and band: the prediction's mean error away from 0, |error| median and p99 (if predict), jumps per
    tracked point, tracked points per mm."""
    ii = tr.index_in_track
    tp = tr.track_of_point
    L = tr.lengths[tp]
    last_idx = np.nonzero((ii == L - 1) & (tr.kind[tp] == 2))[0]
    res = pr.linear_fit_residual(tr.P, last_idx)
    jump = np.zeros(len(tr.P), bool)
    jump[last_idx[res > LF_LIMIT]] = True
    U = np.nonzero(ii >= 4)[0]
    scored = U[~jump[U]]
    J = np.nonzero(jump)[0]
    lastpt = tr.P[scored - 1]
    e = None
    out = {}
    if predict is not None:
        e = predict(tr.P, scored, ii[scored]) - tr.P[scored]
        n3 = np.linalg.norm(e, axis=1)
        q = np.percentile(n3, [50, 90, 99])
        out["all_3d"] = {"median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                         "gt_0.064": float(np.mean(n3 > 0.064))}
    for lo, hi in BANDS:
        key = f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}"
        r = {"n_axis_samples": 0, "jumps": 0}
        ea, sg = [], []
        for ax in range(3):
            c = np.abs(lastpt[:, ax])
            sel = (c >= lo) & (c < hi)
            r["n_axis_samples"] += int(sel.sum())
            cj = np.abs(tr.P[J - 1, ax])
            r["jumps"] += int(np.sum((cj >= lo) & (cj < hi)))
            if e is not None:
                ea.append(e[sel, ax])
                sg.append(np.sign(lastpt[sel, ax]))
        r["jumps_per_1000_points"] = 1000.0 * r["jumps"] / max(r["n_axis_samples"], 1)
        r["jumps_per_1000_points_err"] = 1000.0 * np.sqrt(max(r["jumps"], 1)) / max(r["n_axis_samples"], 1)
        width = 2 * ((hi if hi < 1e9 else np.nan) - lo)
        r["points_per_mm"] = r["n_axis_samples"] / width if np.isfinite(width) else None
        if e is not None:
            ea, sg = np.concatenate(ea), np.concatenate(sg)
            r["mean_outward_mm"] = float(np.mean(ea * sg))
            r["mean_outward_se_mm"] = float(np.std(ea * sg) / np.sqrt(len(ea)))
            r["median_abs_mm"] = float(np.median(np.abs(ea)))
            r["p99_abs_mm"] = float(np.percentile(np.abs(ea), 99))
        out[key] = r
    return out, int(len(scored)), int(len(J))


def compare():
    t0 = time.time()
    runs = {"W": "b300_final3", "MD": "b300_md", "Wjit": "b300_wjit"}
    cache = {"W": "b300", "MD": "b300_md", "Wjit": "b300_wjit"}       # b300 is b300_final3's cache (E1)
    for lab in ("MD", "Wjit"):
        pr.RUN_FILES[cache[lab]] = (os.path.join(RUNS, runs[lab], "Results", "ConvergeTrack"), 300,
                                    f"{runs[lab]} ({lab})")
    # --- frames 0-3 identical, first differing frame
    pf = {lab: read_per_frame(run) for lab, run in runs.items()}
    keys = [k for k in pf["W"][0].keys() if not k.startswith(TIME_KEYS)]
    res_frames = {}
    for lab in ("MD", "Wjit"):
        first_diff = None
        for a, b in zip(pf["W"], pf[lab]):
            if any(a[k] != b[k] for k in keys):
                first_diff = int(a["frame"])
                break
        init_same = all(filecmp.cmp(os.path.join(RUNS, runs["W"], "Results", "InitialTrack", f),
                                    os.path.join(RUNS, runs[lab], "Results", "InitialTrack", f), shallow=False)
                        for f in os.listdir(os.path.join(RUNS, runs["W"], "Results", "InitialTrack")))
        res_frames[lab] = {"first_differing_frame": first_diff, "initial_tracks_identical": init_same,
                           "frames_done": len(pf[lab])}
        check(f"E4b {lab}: frames 0-3 as today's (no prediction there)", init_same and first_diff is not None and
              first_diff >= 4 and len(pf[lab]) == len(pf["W"]),
              f"InitialTrack files identical: {init_same}; first frame with a different counter: {first_diff}; "
              f"frames done {len(pf[lab])} of {len(pf['W'])}")
    results["frames"] = res_frames

    # --- per-frame counters from frame 4
    def series(lab, key, lo=4, hi=300):
        return np.array([num(r[key]) for r in pf[lab] if lo <= int(r["frame"]) <= hi and num(r[key]) is not None])
    rows = {}
    log("\nper-frame counters, frames 4-300 (mean) and 281-300 (mean); MD and the control against today's (W):")
    log(f"  {'counter':>11s} {'W':>10s} {'MD':>10s} {'Wjit':>10s}   {'MD - W':>8s} {'Wjit - W':>9s}   "
        f"{'W 281-300':>10s} {'MD':>10s} {'Wjit':>10s}")
    for key in COUNTERS + ("t_total", "t_pred", "t_shake_pred"):
        m = {lab: float(np.mean(series(lab, key))) for lab in runs}
        late = {lab: float(np.mean(series(lab, key, 281, 300))) for lab in runs}
        rows[key] = {"mean_4_300": m, "mean_281_300": late,
                     "md_minus_w_pct": 100 * (m["MD"] / m["W"] - 1) if m["W"] else None,
                     "wjit_minus_w_pct": 100 * (m["Wjit"] / m["W"] - 1) if m["W"] else None}
        log(f"  {key:>11s} {m['W']:10.2f} {m['MD']:10.2f} {m['Wjit']:10.2f}   "
            f"{rows[key]['md_minus_w_pct']:+7.2f}% {rows[key]['wjit_minus_w_pct']:+8.2f}%   "
            f"{late['W']:10.2f} {late['MD']:10.2f} {late['Wjit']:10.2f}")
    results["per_frame"] = rows

    # --- the saved tracks
    trs = {lab: pr.load_run(cache[lab], refresh=(lab != "W")) for lab in runs}
    ts = {lab: track_stats(tr) for lab, tr in trs.items()}
    results["tracks"] = ts
    log("\nsaved tracks at frame 300 (all long tracks: active, exited, lost):")
    log(f"  {'':>16s} {'W':>10s} {'MD':>10s} {'Wjit':>10s}   {'MD - W':>8s} {'Wjit - W':>9s}")
    for key in ts["W"]:
        a, b, c = ts["W"][key], ts["MD"][key], ts["Wjit"][key]
        log(f"  {key:>16s} {a:10.0f} {b:10.0f} {c:10.0f}   {100 * (b / a - 1):+7.2f}% {100 * (c / a - 1):+8.2f}%")
    # verdict: MD no worse than W beyond the control's spread, on the yield measures
    good_up = ["points", "tracks_ge_50", "tracks_ge_100", "tracks_ge_200", "points_in_ge_100"]
    verdict = {}
    for key in good_up:
        dm = ts["MD"][key] / ts["W"][key] - 1
        dc = abs(ts["Wjit"][key] / ts["W"][key] - 1)
        verdict[key] = {"md_minus_w": dm, "control_spread": dc, "ok": dm >= -max(dc, 0.001)}
    lf = rows["fail_lf"]
    dm = lf["mean_4_300"]["MD"] / lf["mean_4_300"]["W"] - 1
    dc = abs(lf["mean_4_300"]["Wjit"] / lf["mean_4_300"]["W"] - 1)
    verdict["fail_lf"] = {"md_minus_w": dm, "control_spread": dc, "ok": dm <= max(dc, 0.001)}
    results["verdict_yield"] = verdict
    check("E4b/c MD's yield is no worse than today's beyond the control's spread", all(v["ok"] for v in verdict.values()),
          "; ".join(f"{k} {100 * v['md_minus_w']:+.2f} % (control spread {100 * v['control_spread']:.2f} %)"
                    for k, v in verdict.items()))

    # --- near the coordinate planes
    pred = {"W": pr.wiener, "MD": pr.mean_step, "Wjit": None}
    bd = {}
    log("\nnear the coordinate planes (per axis, by the last point's distance from 0; each run with its own predictor):")
    for lab in runs:
        b, n_sc, n_j = bands(trs[lab], pred[lab])
        bd[lab] = {"bands": b, "n_scored": n_sc, "n_jumps": n_j}
    results["bands"] = bd
    log(f"  {'band':>8s} | {'mean error away from 0, mm':^26s} | {'shake move |e| median / p99, mm':^33s} | "
        f"{'losses per 1000 points':^29s} | {'points per mm':^27s}")
    log(f"  {'':>8s} | {'W':>12s} {'MD':>12s} | {'W':>16s} {'MD':>16s} | {'W':>9s} {'MD':>9s} {'Wjit':>9s} | "
        f"{'W':>8s} {'MD':>8s} {'Wjit':>8s}")
    for key in [k for k in bd["W"]["bands"] if k != "all_3d"]:
        w, m, j = bd["W"]["bands"][key], bd["MD"]["bands"][key], bd["Wjit"]["bands"][key]
        ppm = lambda r: f"{r['points_per_mm']:8.0f}" if r["points_per_mm"] else f"{'-':>8s}"
        log(f"  {key:>8s} | {w['mean_outward_mm']:+12.4f} {m['mean_outward_mm']:+12.4f} | "
            f"{w['median_abs_mm']:7.4f} / {w['p99_abs_mm']:.4f} {m['median_abs_mm']:7.4f} / {m['p99_abs_mm']:.4f} | "
            f"{w['jumps_per_1000_points']:9.3f} {m['jumps_per_1000_points']:9.3f} {j['jumps_per_1000_points']:9.3f} | "
            f"{ppm(w)} {ppm(m)} {ppm(j)}")
    aw, am = bd["W"]["bands"]["all_3d"], bd["MD"]["bands"]["all_3d"]
    log(f"  the shake's 3-D move from each run's own prediction, all scored points: W median {aw['median']:.4f}, p90 "
        f"{aw['p90']:.4f}, p99 {aw['p99']:.4f} mm ({100 * aw['gt_0.064']:.2f} % above 0.064 mm); MD median "
        f"{am['median']:.4f}, p90 {am['p90']:.4f}, p99 {am['p99']:.4f} mm ({100 * am['gt_0.064']:.2f} %)")
    mo = [abs(bd["MD"]["bands"][k]["mean_outward_mm"]) for k in bd["MD"]["bands"] if k != "all_3d"]
    check("E4d MD's predictions show no bias away from 0 in any band (|mean| <= 0.001 mm)", max(mo) <= 0.001,
          f"largest |mean| {max(mo):.4f} mm; today's at 1-3 mm {bd['W']['bands']['1-3']['mean_outward_mm']:+.4f} mm")
    w13, m13 = bd["W"]["bands"]["1-3"], bd["MD"]["bands"]["1-3"]
    z = (m13["jumps_per_1000_points"] - w13["jumps_per_1000_points"]) / np.hypot(
        m13["jumps_per_1000_points_err"], w13["jumps_per_1000_points_err"])
    check("E4d no more losses near the planes with MD (1-3 mm band, within 2 standard errors)", z <= 2.0,
          f"losses per 1000 points at 1-3 mm: W {w13['jumps_per_1000_points']:.3f} +- "
          f"{w13['jumps_per_1000_points_err']:.3f}, MD {m13['jumps_per_1000_points']:.3f} +- "
          f"{m13['jumps_per_1000_points_err']:.3f} (z = {z:+.2f})")

    n_fail = sum(not c["ok"] for c in results["checks"].values())
    log(f"\nE4b-d: {len(results['checks']) - n_fail} of {len(results['checks'])} checks passed "
        f"({time.time() - t0:.0f} s)")
    save("e4_compare")
    return n_fail


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "port"
    if mode == "port":
        sys.exit(1 if port(sys.argv[2] if len(sys.argv) > 2 else "md") else 0)
    elif mode == "compare":
        sys.exit(1 if compare() else 0)
    else:
        sys.exit(__doc__)
