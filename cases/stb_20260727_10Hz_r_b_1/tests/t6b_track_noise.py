"""T6b: why long tracks fail the linear-fit test (an addition to T6, 2026-10-04-plan.md section 6a).

STB removes a long track when the last of its last four points lies more than MAX_ERR_LINEARFIT = 0.05 mm (fixed in
inc/libMath/STBCommons.h) from the least-squares line through those four points (STB::checkLinearFit).  From the
tracks each run saved (Results/ConvergeTrack: exit and inactive tracks of every save, and the long tracks still
active at the last one) this measures, per run:
  - the speed of the tracked particles (mm per frame, median per track);
  - their position noise: the robust spread of second differences, sigma = 1.4826 MAD(d2) / sqrt(6) per axis;
  - the linear-fit residual (as checkLinearFit computes it) of every point after the fourth: the points that passed,
    and the last point of each inactive track, which is the point that ended it when the linear fit did.
The old run's positions are in the old calibration's frame; they are also given multiplied by s = 1.229 (T0), the
scale that turns old-frame lengths into true ones.

Writes t6b_track_noise.json, t6b_track_noise_out.txt and t6b_residuals.png beside this script.
    python tests/t6b_track_noise.py [new|old ...]
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402

OUT_JSON = os.path.join(case.TESTS, "t6b_track_noise.json")
OUT_TXT = os.path.join(case.TESTS, "t6b_track_noise_out.txt")
RUNS = {"new": "new_cams", "old": "old_cams"}
LIMIT = 0.05
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def load(conv_dir):
    """{kind: DataFrame(TrackID, FrameID, X, Y, Z)} with TrackID made unique per file."""
    files = glob.glob(os.path.join(conv_dir, "*.csv"))
    saves = sorted({int(m.group(1)) for f in files if (m := re.search(r"_(\d+)\.csv$", f))})
    out = {}
    for kind, pats in (("exit", ["ExitTrack_{}.csv"]), ("inactive", ["LongTrackInactive_{}.csv"]),
                       ("active", ["LongTrackActive_{}.csv"])):
        parts = []
        use = saves if kind != "active" else saves[-1:]
        for s in use:
            for pat in pats:
                p = os.path.join(conv_dir, pat.format(s))
                if os.path.exists(p):
                    d = pd.read_csv(p, usecols=[0, 1, 2, 3, 4])
                    d.columns = ["tid", "frame", "x", "y", "z"]
                    d["tid"] = d["tid"].astype(np.int64) + (s * 10_000_000)
                    parts.append(d)
        if parts:
            out[kind] = pd.concat(parts, ignore_index=True).sort_values(["tid", "frame"], kind="stable")
    return out, saves


def lf_residuals(P):
    """checkLinearFit's residual for every point i >= 3 of one track P (n, 3): distance of P[i] from the LS line
    through P[i-3..i] evaluated at i."""
    n = len(P)
    if n < 4:
        return np.empty(0)
    # LS line over t = 0..3, value at t = 3: weights (-0.2, 0.1, 0.4, 0.7) (= kernel row of checkLinearFit)
    w = np.array([-0.2, 0.1, 0.4, 0.7])
    W = np.lib.stride_tricks.sliding_window_view(P, (4, 3))[:, 0]           # (n-3, 4, 3)
    est = np.einsum("k,nkj->nj", w, W)
    return np.linalg.norm(P[3:] - est, axis=1)


def analyse(dfs, scale):
    speeds, sig, res_pass, res_last_inactive, lengths = [], [], [], [], {}
    for kind, d in dfs.items():
        g = d.groupby("tid", sort=False)
        idx = g.indices
        P_all = d[["x", "y", "z"]].to_numpy(float) * scale
        L = []
        for tid, ii in idx.items():
            P = P_all[ii]
            L.append(len(P))
            if len(P) < 7:
                continue
            v = np.linalg.norm(np.diff(P, axis=0), axis=1)
            speeds.append(np.median(v))
            d2 = P[2:] - 2 * P[1:-1] + P[:-2]
            mad = np.median(np.abs(d2 - np.median(d2, axis=0)), axis=0)
            sig.append(1.4826 * mad / np.sqrt(6))
            r = lf_residuals(P)
            if kind == "inactive":
                res_last_inactive.append(r[-1])
                res_pass.append(r[:-1])
            else:
                res_pass.append(r)
        lengths[kind] = np.array(L)
    sig = np.array(sig)
    res_pass = np.concatenate(res_pass) if res_pass else np.empty(0)
    res_last = np.array(res_last_inactive)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")  # noqa: E731
    return {
        "n_tracks_ge7": int(len(speeds)),
        "speed_mm_per_frame": {"p10": q(speeds, 10), "median": q(speeds, 50), "p90": q(speeds, 90)},
        "sigma_mm_per_axis_median": sig.mean(axis=0).tolist() if len(sig) else None,
        "sigma_mm_median": {ax: q(sig[:, i], 50) for i, ax in enumerate("xyz")} if len(sig) else None,
        "res_pass_mm": {"median": q(res_pass, 50), "p90": q(res_pass, 90), "p99": q(res_pass, 99),
                        "n": int(len(res_pass))},
        "res_last_inactive_mm": {"p10": q(res_last, 10), "median": q(res_last, 50), "p90": q(res_last, 90),
                                 "share_over_limit": float(np.mean(res_last > LIMIT)) if len(res_last) else None,
                                 "share_under_2x": float(np.mean(res_last <= 2 * LIMIT)) if len(res_last) else None,
                                 "n": int(len(res_last))},
        "lengths": {k: {"n": int(len(v)), "median": float(np.median(v)) if len(v) else None} for k, v in
                    lengths.items()},
        "_res_pass": res_pass, "_res_last": res_last,
    }


def main():
    which = sys.argv[1:] or list(RUNS)
    cfg = case.load_config()
    with open(os.path.join(case.TESTS, "t0_frame_map.json"), encoding="utf-8") as f:
        s_old = json.load(f)["mapping"]["similarity"]["s"]
    res = {"limit_mm": LIMIT, "s_old_to_true": s_old, "runs": {}}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            res["runs"].update(json.load(f).get("runs", {}))
    hists = {}
    for k in which:
        conv = os.path.join(cfg["out_dir"], RUNS[k], "Results", "ConvergeTrack")
        dfs, saves = load(conv)
        if not dfs:
            log(f"{k}: no saved tracks in {conv}")
            continue
        scales = {"frame": 1.0} if k == "new" else {"frame": 1.0, "true (x s)": s_old}
        rk = {"saves": saves}
        for sname, sc in scales.items():
            a = analyse(dfs, sc)
            hists[f"{k} {sname}"] = (a.pop("_res_pass"), a.pop("_res_last"))
            rk[sname] = a
            log(f"\n== {k} camFiles, positions in {sname} mm; saves {saves}")
            log(f"   tracks >= 7 points: {a['n_tracks_ge7']}; lengths by file kind: {a['lengths']}")
            log(f"   speed (median per track), mm/frame: p10 {a['speed_mm_per_frame']['p10']:.4f}, median "
                f"{a['speed_mm_per_frame']['median']:.4f}, p90 {a['speed_mm_per_frame']['p90']:.4f}")
            if a["sigma_mm_median"]:
                log("   position noise per axis (median over tracks), mm: " +
                    ", ".join(f"{ax} {v:.4f}" for ax, v in a["sigma_mm_median"].items()))
            rp, rl = a["res_pass_mm"], a["res_last_inactive_mm"]
            log(f"   linear-fit residual of the points that passed: median {rp['median']:.4f}, p90 {rp['p90']:.4f},"
                f" p99 {rp['p99']:.4f} mm (n {rp['n']})")
            log(f"   last point of the inactive tracks: p10 {rl['p10']:.4f}, median {rl['median']:.4f}, p90 "
                f"{rl['p90']:.4f} mm; over {LIMIT} mm: {rl['share_over_limit']:.3f}; within 2x the limit: "
                f"{rl['share_under_2x']:.3f} (n {rl['n']})")
        res["runs"][k] = rk
    if hists:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                             "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        bins = np.logspace(-4, 1, 101)
        col = {"new frame": "#2a78d6", "old true (x s)": "#eb6834"}
        for key, (rp, rl) in hists.items():
            if key not in col:
                continue
            ax.hist(rl, bins=bins, histtype="step", lw=2, color=col[key], label=f"{key.split()[0]} camFiles: last "
                    f"point of the inactive tracks")
            ax.hist(rp, bins=bins, histtype="step", lw=1, ls="--", color=col[key],
                    weights=np.full(len(rp), len(rl) / max(1, len(rp))),
                    label=f"{key.split()[0]} camFiles: points that passed (scaled to the same count)")
        ax.axvline(LIMIT, color="#52514e", lw=1)
        ax.annotate("0.05 mm limit", (LIMIT, ax.get_ylim()[1] * 0.95), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, va="top")
        ax.set_xscale("log")
        ax.set_xlim(5e-4, 0.3)
        ax.set_xlabel("linear-fit residual, mm (true scale)")
        ax.set_ylabel("points")
        ax.set_title("Linear-fit residuals of tracked points", loc="left", fontweight="bold", fontsize=10)
        ax.grid(True, color="#e7e6e1", lw=0.6)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(case.TESTS, "t6b_residuals.png"), dpi=150)
        plt.close(fig)
    case.save_json(OUT_JSON, res)
    with open(OUT_TXT, "a" if os.path.exists(OUT_TXT) and len(which) < len(RUNS) else "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT_JSON}\n      {OUT_TXT}")


if __name__ == "__main__":
    main()
