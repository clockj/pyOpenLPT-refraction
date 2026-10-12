"""T6: compare the STB runs T4 (new camFiles) and T5 (old camFiles) (2026-10-04-plan.md, section 4).

From each run folder on X: (stb_new_calib/new_cams, old_cams) it reads log.txt (parse_log.py), run_info.json,
resources.csv and the saved tracks in Results/ConvergeTrack, and reports:
  - speed: time per frame (the C++ "Total time for frame", which excludes loading the images) over frame windows,
    without the frames that save tracks; matching time per camera combination; the other steps; memory and CPU use;
  - IPR: objects found per frame, objects matched on all four cameras at 2 and 3 px, detections left in the residual
    images;
  - tracks: active long tracks, long tracks removed per frame (linear-fit failures, shaking failures, exits) as a share
    of the active long tracks; lengths of the saved long tracks.
The old run's logs (log_0_1500.txt: frames 0-1740, an earlier tracerConfig; log.txt: frames 3501-4687) are added for
context.  A run still in progress is read as far as it got.

Writes beside this script: t4_per_frame.csv, t5_per_frame.csv, t6_compare.json, t6_compare_out.txt and the figures
t6_*.png.
    python tests/t6_compare.py [--no-tracks]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402
import parse_log  # noqa: E402

OUT_JSON = os.path.join(case.TESTS, "t6_compare.json")
OUT_TXT = os.path.join(case.TESTS, "t6_compare_out.txt")
RUNS = {"new": ("new_cams", "t4_per_frame.csv"), "old": ("old_cams", "t5_per_frame.csv")}
WINDOWS = [(4, 100), (100, 300), (300, 500), (500, 1001)]
COMBOS = ["0123", "012", "013", "023", "123"]
COLORS = {"new": "#2a78d6", "old": "#eb6834", "restart": "#1baf7a", "context": "#8a8984"}
LABELS = {"new": "new camFiles (T4)", "old": "old camFiles (T5)"}
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def frame_table(path):
    recs = parse_log.parse(path)
    df = pd.DataFrame([parse_log.flatten(r) for r in recs])
    if len(df) and "la_n" in df:
        conv = df["phase"] == "conv"
        for k in ("la_rm", "fail_lf", "fail_shake", "ex_add", "la_add"):
            df.loc[conv, f"rate_{k}"] = df.loc[conv, k] / df.loc[conv, "la_n"].clip(lower=1)
    return recs, df


def wstats(df, col, lo, hi, skip_saved=True):
    m = (df["frame"] >= lo) & (df["frame"] < hi) & (df["phase"] == "conv")
    if skip_saved and "saved" in df:
        m &= ~df["saved"].astype(bool)
    if col not in df:
        return None
    x = df.loc[m, col].dropna().to_numpy(float)
    if len(x) == 0:
        return None
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(np.mean(x)),
            "p90": float(np.percentile(x, 90)), "max": float(np.max(x))}


def track_lengths(conv_dir):
    """Lengths (frames) of the long tracks saved in ConvergeTrack: finished ones (exit + inactive files of every save)
    and the ones still active at the last save (censored)."""
    out = {}
    files = glob.glob(os.path.join(conv_dir, "*.csv"))
    saves = sorted({int(m.group(1)) for f in files if (m := re.search(r"_(\d+)\.csv$", f))})
    if not saves:
        return out, saves
    last = saves[-1]
    for kind, pat in (("exit", "ExitTrack_{}.csv"), ("inactive", "LongTrackInactive_{}.csv")):
        L = []
        for s in saves:
            p = os.path.join(conv_dir, pat.format(s))
            if os.path.exists(p):
                d = pd.read_csv(p, usecols=["TrackID", "FrameID"])
                L.append(d.groupby("TrackID")["FrameID"].size().to_numpy())
        out[kind] = np.concatenate(L) if L else np.array([], int)
    for kind, pat in (("active_long", "LongTrackActive_{}.csv"), ("active_short", "ShortTrackActive_{}.csv")):
        p = os.path.join(conv_dir, pat.format(last))
        if os.path.exists(p):
            d = pd.read_csv(p, usecols=["TrackID", "FrameID"])
            out[kind] = d.groupby("TrackID")["FrameID"].size().to_numpy()
    return out, saves


def len_stats(L):
    L = np.asarray(L)
    if len(L) == 0:
        return {"n": 0}
    return {"n": int(len(L)), "median": float(np.median(L)), "mean": float(L.mean()),
            "p90": float(np.percentile(L, 90)), "max": int(L.max()),
            **{f"ge{k}": int((L >= k).sum()) for k in (10, 20, 50, 100, 200, 500)}, "points": int(L.sum())}


def rolling(x, w=21):
    return pd.Series(x).rolling(w, center=True, min_periods=1).median().to_numpy()


def figures(tables, context, resources, lengths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})

    def finish(ax, title, ylabel, xlabel="frame"):
        ax.set_title(title, loc="left", color="#0b0b0b")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#e7e6e1", lw=0.6)
        ax.set_axisbelow(True)

    def label_end(ax, x, y, text, color, dy=0):
        ok = np.isfinite(y)
        if ok.any():
            i = np.flatnonzero(ok)[-1]
            ax.annotate(text, (x[i], y[i]), xytext=(4, dy), textcoords="offset points", va="center", fontsize=8,
                        color="#0b0b0b")

    def conv(df):
        m = (df["phase"] == "conv") & ~df["saved"].astype(bool)
        return df[m]

    figs = []
    # 1. time per frame
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, df in tables.items():
        d = conv(df)
        ax.plot(d["frame"], rolling(d["t_total"].to_numpy()), color=COLORS[k], lw=2, label=LABELS[k])
        label_end(ax, d["frame"].to_numpy(), rolling(d["t_total"].to_numpy()), LABELS[k].split(" (")[0], COLORS[k])
    if "restart" in context:
        d = conv(context["restart"])
        y = rolling(d["t_total"].to_numpy(), 11)
        ax.plot(d["frame"], y, color=COLORS["restart"], lw=2, label="new camFiles, fresh process from frame 500 (T4c)")
        label_end(ax, d["frame"].to_numpy(), y, "T4c restart", COLORS["restart"])
    if "log_0_1500" in context:
        d = conv(context["log_0_1500"])
        d = d[d["frame"] <= 1000]
        ax.plot(d["frame"], rolling(d["t_total"].to_numpy()), color=COLORS["context"], lw=1.2, ls="--",
                label="old run, log_0_1500 (earlier tracerConfig)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(bottom=0)
    finish(ax, "Time per frame (convergence phase, 21-frame median)", "s per frame")
    fig.tight_layout()
    figs.append(("t6_time_per_frame.png", fig))
    # 2. active long tracks
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, df in tables.items():
        d = df[df["phase"] == "conv"]
        ax.plot(d["frame"], d["la_new"], color=COLORS[k], lw=2, label=LABELS[k])
        label_end(ax, d["frame"].to_numpy(), d["la_new"].to_numpy(float), LABELS[k].split(" (")[0], COLORS[k])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(bottom=0)
    finish(ax, "Active long tracks after each frame", "tracks")
    fig.tight_layout()
    figs.append(("t6_active_long_tracks.png", fig))
    # 3. removal rate
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ends = {}
    for k, df in tables.items():
        d = df[df["phase"] == "conv"]
        y = rolling(100 * d["rate_la_rm"].to_numpy())
        ax.plot(d["frame"], y, color=COLORS[k], lw=2, label=LABELS[k])
        ends[k] = (d["frame"].to_numpy(), y)
    order = sorted(ends, key=lambda k: ends[k][1][np.isfinite(ends[k][1])][-1])
    for j, k in enumerate(order):            # the lower end label below, the upper one above
        label_end(ax, *ends[k], LABELS[k].split(" (")[0], COLORS[k], dy=(-6 if j == 0 else 6) if len(order) > 1 else 0)
    if "log_0_1500" in context:
        d = context["log_0_1500"]
        d = d[(d["phase"] == "conv") & (d["frame"] <= 1000)]
        ax.plot(d["frame"], rolling(100 * d["rate_la_rm"].to_numpy()), color=COLORS["context"], lw=1.2, ls="--",
                label="old run, log_0_1500")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)
    finish(ax, "Long tracks removed per frame, share of the active long tracks (21-frame median)", "% per frame")
    fig.tight_layout()
    figs.append(("t6_removal_rate.png", fig))
    # 4. objects found by IPR per frame
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for k, df in tables.items():
        d = df[df["phase"] == "conv"]
        y = rolling(d["ipr_found"].to_numpy(float))
        ax.plot(d["frame"], y, color=COLORS[k], lw=2, label=LABELS[k])
        label_end(ax, d["frame"].to_numpy(), y, LABELS[k].split(" (")[0], COLORS[k])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)
    finish(ax, "New objects found by IPR in the residual images per frame (21-frame median)", "objects")
    fig.tight_layout()
    figs.append(("t6_ipr_found.png", fig))
    # 5. track lengths (CCDF)
    if lengths:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for k, L in lengths.items():
            fin = np.concatenate([L.get("exit", []), L.get("inactive", []), L.get("active_long", [])]).astype(int)
            if len(fin) == 0:
                continue
            xs = np.arange(1, fin.max() + 1)
            cc = np.array([(fin >= x).sum() for x in xs])
            ax.step(xs, cc, where="post", color=COLORS[k], lw=2, label=LABELS[k])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8)
        finish(ax, "Saved long tracks at least L frames long (finished and still active)", "tracks", "L (frames)")
        fig.tight_layout()
        figs.append(("t6_track_lengths.png", fig))
    # 6. memory of the process
    if resources:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for k, r in resources.items():
            ax.plot(r["t_s"] / 3600, r["private_MB"] / 1024, color=COLORS[k], lw=2, label=LABELS[k])
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(bottom=0)
        finish(ax, "Memory of the STB process (private bytes)", "GB", "hours since start")
        fig.tight_layout()
        figs.append(("t6_memory.png", fig))
    for name, fig in figs:
        fig.savefig(os.path.join(case.TESTS, name), dpi=150)
        plt.close(fig)
    return [n for n, _ in figs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tracks", action="store_true", help="skip reading the saved track files")
    args = ap.parse_args()
    cfg = case.load_config()
    res = {"runs": {}, "context": {}}
    tables, resources, lengths = {}, {}, {}
    for k, (sub, csv_name) in RUNS.items():
        d = os.path.join(cfg["out_dir"], sub)
        lp = os.path.join(d, "log.txt")
        if not os.path.exists(lp):
            continue
        recs, df = frame_table(lp)
        if df.empty:
            continue
        df.to_csv(os.path.join(case.TESTS, csv_name), index=False)
        tables[k] = df
        info = {}
        if os.path.exists(os.path.join(d, "run_info.json")):
            with open(os.path.join(d, "run_info.json"), encoding="utf-8") as f:
                info = json.load(f)
        r = {"frames": [int(df["frame"].min()), int(df["frame"].max())], "n_frames": int(len(df)), "info": info}
        r["t_total_sum"] = float(df["t_total"].sum())
        r["saved_frames"] = {int(a): float(b) for a, b in zip(df.loc[df["saved"], "frame"], df.loc[df["saved"],
                                                                                                    "t_total"])}
        cols = ["t_total", "t_match_all", "t_shake_ipr", "t_shake_pred", "t_overlap", "t_link", "ipr_found",
                "det_first", "matched_0123_tol2", "matched_0123_tol3", "la_n", "la_new", "sa_new", "la_add", "la_rm",
                "fail_lf", "fail_shake", "ex_add", "rate_la_rm", "rate_fail_lf", "rate_fail_shake", "rate_ex_add"]
        cols += [f"t_match_{c}" for c in COMBOS] + [f"matched_{c}" for c in COMBOS]
        r["windows"] = {f"{lo}-{hi - 1}": {c: wstats(df, c, lo, hi) for c in cols} for lo, hi in WINDOWS}
        rp = os.path.join(d, "resources.csv")
        if os.path.exists(rp):
            rr = pd.read_csv(rp)
            if len(rr):
                resources[k] = rr
                r["resources"] = {"private_MB_start": float(rr["private_MB"].iloc[0]),
                                  "private_MB_end": float(rr["private_MB"].iloc[-1]),
                                  "working_set_MB_max": float(rr["working_set_MB"].max()),
                                  "proc_cores_median": float(rr["proc_cores"].median()),
                                  "others_cores_median": float(rr["others_cores"].median()),
                                  "others_cores_p90": float(rr["others_cores"].quantile(0.9))}
        if not args.no_tracks:
            L, saves = track_lengths(os.path.join(d, "Results", "ConvergeTrack"))
            if L:
                lengths[k] = L
                fin = np.concatenate([L.get("exit", []), L.get("inactive", [])])
                r["tracks"] = {"saves": saves, "finished": len_stats(fin), "exit": len_stats(L.get("exit", [])),
                               "inactive": len_stats(L.get("inactive", [])),
                               "active_long_at_end": len_stats(L.get("active_long", [])),
                               "active_short_at_end": len_stats(L.get("active_short", [])),
                               "all_long": len_stats(np.concatenate([fin, L.get("active_long", [])]))}
        res["runs"][k] = r
    # context: T4c (a fresh process from T4's tracks at frame 500) and the old run's logs
    context = {}
    rp = os.path.join(cfg["out_dir"], "new_cams_restart500", "log.txt")
    if os.path.exists(rp):
        _, context["restart"] = frame_table(rp)
    for name in cfg["old_logs"]:
        p = case.case_path(cfg, name)
        _, df = frame_table(p)
        key = os.path.splitext(name)[0]
        context[key] = df
        lo, hi = int(df["frame"].min()), int(df["frame"].max())
        wins = [(lo + 4, lo + 100), (lo + 100, lo + 300), (lo + 300, lo + 500), (lo + 500, lo + 1001)] \
            if key == "log" else WINDOWS
        res["context"][key] = {"frames": [lo, hi], "windows": {
            f"{a}-{b - 1}": {c: wstats(df, c, a, b) for c in ("t_total", "t_match_all", "ipr_found", "la_new",
                                                              "rate_la_rm", "rate_fail_lf", "matched_0123_tol2",
                                                              "matched_0123_tol3")} for a, b in wins}}

    # ---- the printed comparison
    def g(k, w, c, s="median"):
        try:
            return res["runs"][k]["windows"][w][c][s]
        except (KeyError, TypeError):
            return float("nan")

    log("T6: STB with the new camFiles (T4) against the old ones (T5); medians per frame over each window,")
    log("    convergence phase, frames that save tracks excluded")
    for k, r in res["runs"].items():
        log(f"  {k}: frames {r['frames']} ({r['n_frames']} done), sum of frame times {r['t_total_sum'] / 3600:.2f} h"
            + (f", wall {r['info']['wall_s'] / 3600:.2f} h" if r["info"].get("wall_s") else "")
            + f"; save frames {r['saved_frames']}")
    rows = [("time per frame, s", "t_total", "{:.2f}"), ("matching time, s", "t_match_all", "{:.2f}"),
            ("  [0,1,2,3]", "t_match_0123", "{:.2f}"), ("  [0,1,2]", "t_match_012", "{:.2f}"),
            ("  [0,1,3]", "t_match_013", "{:.2f}"), ("  [0,2,3]", "t_match_023", "{:.2f}"),
            ("  [1,2,3]", "t_match_123", "{:.2f}"), ("shaking the predictions, s", "t_shake_pred", "{:.2f}"),
            ("IPR shaking, s", "t_shake_ipr", "{:.2f}"),
            ("detections in the residual images (sum, 1st round)", "det_first", "{:.0f}"),
            ("objects found by IPR", "ipr_found", "{:.0f}"),
            ("  matched on 4 cameras at 2 px", "matched_0123_tol2", "{:.0f}"),
            ("  matched on 4 cameras at 3 px", "matched_0123_tol3", "{:.0f}"),
            ("active long tracks", "la_new", "{:.0f}"), ("active short tracks", "sa_new", "{:.0f}"),
            ("long tracks added per frame", "la_add", "{:.0f}"), ("long tracks removed per frame", "la_rm", "{:.0f}"),
            ("  failed the linear fit", "fail_lf", "{:.0f}"), ("  failed shaking", "fail_shake", "{:.0f}"),
            ("  left the view volume", "ex_add", "{:.0f}"),
            ("removed, % of active long tracks", "rate_la_rm", "{:.2%}"),
            ("  linear-fit failures, %", "rate_fail_lf", "{:.2%}")]
    wnames = [f"{lo}-{hi - 1}" for lo, hi in WINDOWS]
    log("  " + " " * 52 + "".join(f"{w:>22s}" for w in wnames))
    log("  " + " " * 52 + "".join(f"{'new':>11s}{'old':>11s}" for _ in wnames))
    for label, col, fmt in rows:
        cells = ""
        for w in wnames:
            for k in ("new", "old"):
                v = g(k, w, col)
                cells += f"{fmt.format(v) if np.isfinite(v) else '-':>11s}"
        log(f"  {label:<52s}{cells}")
    for k, r in res["runs"].items():
        if "resources" in r:
            rr = r["resources"]
            log(f"  {k} resources: private bytes {rr['private_MB_start']:.0f} -> {rr['private_MB_end']:.0f} MB, "
                f"working set max {rr['working_set_MB_max']:.0f} MB, cores used median {rr['proc_cores_median']:.1f}, "
                f"other programs median {rr['others_cores_median']:.1f} (p90 {rr['others_cores_p90']:.1f}) cores")
        if "tracks" in r:
            t = r["tracks"]
            for kind in ("finished", "exit", "inactive", "active_long_at_end", "all_long"):
                s = t[kind]
                if s.get("n"):
                    log(f"  {k} tracks, {kind:>18s}: n {s['n']:6d}, length median {s['median']:.0f}, mean "
                        f"{s['mean']:.1f}, p90 {s['p90']:.0f}, max {s['max']}; >=50: {s['ge50']}, >=100: {s['ge100']},"
                        f" >=200: {s['ge200']}; points {s['points']}")
    log("  context, the old run's logs (medians):")
    for key, c in res["context"].items():
        for w, st in c["windows"].items():
            vals = {k: (v["median"] if v else float("nan")) for k, v in st.items()}
            log(f"    {key} frames {w}: time {vals['t_total']:.2f} s, matching {vals['t_match_all']:.2f} s, found "
                f"{vals['ipr_found']:.0f}, active long {vals['la_new']:.0f}, removed {vals['rate_la_rm']:.2%}, "
                f"linear-fit {vals['rate_fail_lf']:.2%}, 4 cams at 2/3 px {vals['matched_0123_tol2']:.0f}/"
                f"{vals['matched_0123_tol3']:.0f}")
    res["figures"] = figures(tables, context, resources, lengths)
    case.save_json(OUT_JSON, res)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT_JSON}\n      {OUT_TXT}\n      figures {res['figures']}")


if __name__ == "__main__":
    main()
