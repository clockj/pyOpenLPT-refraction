"""Tables and figures of the stb_improve campaign from its runs (runs/<name>/, written by stbrun.py).

    python analyze.py s1    identical outputs (S1), the text/JSON run pair (C1 end to end)
    python analyze.py p2    the save-interval run (A3): saves at 20, 40 and 45, freed tracks
    python analyze.py p1    benchmarks B300 and restart tests R300: time per frame, growth, memory, cores
    python analyze.py b     Phase B: matching counters and times ([profile] lines), B4's overlap time, B2's yield
Each writes results/<cmd>_*.txt (+ .json) and figures results/<cmd>_*.png; runs that do not exist yet are skipped.
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
import compare_runs as cr  # noqa: E402

B300 = [("b300_base", "HEAD (baseline)"), ("b300_a1_9", "A1, inline 9"), ("b300_a1_3", "A1, inline 3"),
        ("b300_b1", "A1 + B0, B1 (caching only)"),
        ("b300_b134", "A1 + B0, B1, B3, B4 (identical outputs)"),
        ("b300_b2", "A1 + B0-B4 (B2: more matches)"),
        ("b300_final", "final, inline 9"), ("b300_final3", "final, inline 3"),
        ("b300_final_si200", "final, inline 9, save every 200")]
R300 = [("b300_base", "r300_base"), ("b300_a1_9", "r300_a1_9"), ("b300_a1_3", "r300_a1_3"),
        ("b300_final", "r300_final")]
COLORS = {"b300_base": "#4a4a4a", "b300_a1_9": "#1f77b4", "b300_a1_3": "#17becf", "b300_b1": "#bcbd22",
          "b300_b134": "#d62728",
          "b300_b2": "#ff7f0e", "b300_final": "#2ca02c", "b300_final3": "#98df8a", "b300_final_si200": "#9467bd"}
# the figure of the time per frame shows these (the final runs overlap b2's line)
FIG_RUNS = ["b300_base", "b300_a1_9", "b300_b1", "b300_b134", "b300_final3"]


def exists(run):
    return os.path.exists(os.path.join(RUNS, run, "log.txt")) and os.path.exists(os.path.join(RUNS, run,
                                                                                              "run_info.json"))


def write(name, lines, data=None):
    os.makedirs(RESULTS, exist_ok=True)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, name + ".txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    if data is not None:
        with open(os.path.join(RESULTS, name + ".json"), "w") as f:
            json.dump(data, f, indent=1, default=str)


# ------------------------------------------------------------------------------------------------ S1
def cmd_s1():
    pairs = [("s50_base", "s50_base_2", "the baseline run twice (determinism)"),
             ("s50_base", "s50_a1_9", "A1, inline size 9"),
             ("s50_base", "s50_a1_3", "A1, inline size 3"),
             ("s50_base", "s50_b134", "A1+A3+A4 and B0, B1, B3, B4 (profile on)"),
             ("s20_text", "s20_json", "the same run from config.txt and from config.json (C1)")]
    lines, data = ["S1 identical outputs (frames 0-50; C1: frames 0-20)", ""], {}
    for a, b, what in pairs:
        if not (exists(a) and exists(b)):
            lines.append(f"{what}: {a} / {b} not run yet")
            continue
        out, d = cr.cmd_identical(a, b, None)
        data[f"{a}|{b}"] = d
        lines.append(f"{what}:")
        lines += ["  " + ln for ln in out[1:]]
    write("s1_identical", lines, data)


# ------------------------------------------------------------------------------------------------ P2
def cmd_p2():
    run = "p2_si20"
    if not exists(run):
        print("p2_si20 not run yet")
        return
    folder = os.path.join(RUNS, run, "Results", "ConvergeTrack")
    files = sorted(os.listdir(folder))
    fr = cr.frames_of(run)
    lines = [f"P2 save interval: {run}, frames 0-45 with save_interval = 20", f"ConvergeTrack files: {files}"]
    want = {"LongTrackInactive_20.csv", "LongTrackInactive_40.csv", "LongTrackInactive_45.csv", "ExitTrack_20.csv",
            "ExitTrack_40.csv", "ExitTrack_45.csv", "LongTrackActive_45.csv", "ShortTrackActive_45.csv"}
    ok_files = set(files) == want
    lines.append(f"files as expected (inactive and exit tracks at 20, 40, 45; active snapshots only at 45): {ok_files}")
    saved = [f for f in sorted(fr) if fr[f]["saved"] and fr[f]["phase"] == "conv"]
    lines.append(f"convergence frames that saved inside the frame: {saved} (the last frame saves after its timing; "
                 "frame 3 writes InitialTrack/ at the end of the initial phase)")
    li = {f: (fr[f].get("li_n"), fr[f].get("li_new")) for f in (20, 21, 40, 41) if f in fr}
    lines.append(f"inactive long tracks (before, after) at frames 20, 21, 40, 41: {li} -> 0 before frames 21 and 41 "
                 "means the save freed them")

    def rows(name):
        with open(os.path.join(folder, name)) as f:
            return sum(1 for _ in f) - 1
    counts = {name: rows(name) for name in files}
    lines.append(f"rows per file: {counts}")
    ok = ok_files and saved == [20, 40] and li.get(21, (None,))[0] == 0 and li.get(41, (None,))[0] == 0
    lines.append("P2 logic PASSED" if ok else "P2 logic FAILED")
    write("p2_save_interval", lines, {"files": files, "saved": saved, "li": li, "rows": counts, "ok": ok})


# ------------------------------------------------------------------------------------------------ P1
def frame_times(run):
    fr = cr.frames_of(run)
    return {f: fr[f]["t_total"] for f in sorted(fr) if fr[f]["phase"] == "conv" and not fr[f]["saved"]}, fr


def resources(run):
    with open(os.path.join(RUNS, run, "resources.csv")) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def cmd_p1():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines, data = ["P1 benchmarks: frames 0-300 from scratch, 26 threads, local disk", ""], {}
    lines.append(f"{'run':12s} {'what':38s} {'f4-23 s':>8s} {'f281-300':>9s} {'growth':>7s} {'sum min':>8s} "
                 f"{'vs base':>8s} {'peak GB':>8s} {'cores':>6s} {'others':>7s}")
    base_late = None
    series = {}
    for run, what in B300:
        if not exists(run):
            lines.append(f"{run:12s} {what:38s} not run yet")
            continue
        t, fr = frame_times(run)
        frames = sorted(t)
        early = [t[f] for f in frames if 4 <= f <= 23]
        late = [t[f] for f in frames if 281 <= f <= 300]
        res = resources(run)
        cores = statistics.mean(r["proc_cores"] for r in res[1:]) if len(res) > 1 else float("nan")
        others = statistics.mean(r["others_cores"] for r in res[1:]) if len(res) > 1 else float("nan")
        peak = max(r["private_MB"] for r in res) / 1024
        total = sum(fr[f]["t_total"] for f in fr) / 60
        e, l_ = statistics.mean(early), statistics.mean(late)
        if run == "b300_base":
            base_late = l_
        rel = (base_late / l_) if base_late else float("nan")
        lines.append(f"{run:12s} {what:38s} {e:8.2f} {l_:9.2f} {l_ / e:7.2f} {total:8.1f} {rel:8.2f} {peak:8.2f} "
                     f"{cores:6.1f} {others:7.2f}")
        data[run] = {"what": what, "early_s": e, "late_s": l_, "growth": l_ / e, "sum_min": total,
                     "late_speedup_vs_base": rel, "peak_private_GB": peak, "mean_cores": cores,
                     "others_cores": others}
        series[run] = (frames, [t[f] for f in frames], res)
    lines.append("f4-23: mean time of frames 4-23 (the first convergence frames); f281-300: of frames 281-300; growth: "
                 "their ratio; vs base: how many times faster than the baseline at frames 281-300; cores: mean CPU "
                 "cores the run used; others: mean cores other programs used meanwhile")

    lines += ["", "R300 restart tests: frames 201-300 in the aged process (B300) and in a fresh process restarted from "
              "tracks saved at frame 200"]
    for aged, fresh in R300:
        if not (exists(aged) and exists(fresh)):
            lines.append(f"  {aged} / {fresh}: not run yet")
            continue
        ta, _ = frame_times(aged)
        tf, _ = frame_times(fresh)
        common = [f for f in sorted(tf) if 201 <= f <= 300 and f in ta]
        ma, mf = statistics.mean(ta[f] for f in common), statistics.mean(tf[f] for f in common)
        _out, rdat = cr.cmd_restart(aged, fresh, (201, 300))
        lines.append(f"  {aged:10s} aged {ma:6.2f} s/frame, fresh {mf:6.2f} s/frame, aged/fresh {ma / mf:5.2f} "
                     f"({len(common)} frames); logs agree in {rdat['log_agree']} of {rdat['frames']} frames"
                     + (f" (first difference at {rdat['first_diff']})" if rdat["first_diff"] is not None else ""))
        data[f"{aged}|{fresh}"] = {"aged_s": ma, "fresh_s": mf, "ratio": ma / mf, "frames": len(common),
                                   "log_agree": rdat["log_agree"], "first_diff": rdat["first_diff"]}
        if fresh not in series:
            series[fresh] = (common, [tf[f] for f in common], resources(fresh))
    write("p1_benchmarks", lines, data)

    # figures: time per frame, memory
    def roll(y, k=5):
        out = []
        for i in range(len(y)):
            w = y[max(0, i - k + 1):i + 1]
            out.append(sum(w) / len(w))
        return out
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, what in B300:
        if run in series and run in FIG_RUNS:
            f, y, _ = series[run]
            ax.plot(f, roll(y), lw=2, color=COLORS[run], label=what)
    for aged, fresh in R300:
        if fresh in series and aged in FIG_RUNS:
            f, y, _ = series[fresh]
            ax.plot(f, roll(y), lw=2, ls="--", color=COLORS[aged], label=f"{fresh}: restarted at 200")
    ax.set_xlabel("frame")
    ax.set_ylabel("time per frame, s (5-frame mean)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Time per frame, frames 0-300 (26 threads)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "p1_time_per_frame.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, what in B300:
        if run in series:
            res = series[run][2]
            ax.plot([r["t_s"] / 60 for r in res], [r["private_MB"] / 1024 for r in res], lw=2, color=COLORS[run],
                    label=what)
    ax.set_xlabel("minutes since start")
    ax.set_ylabel("private memory, GB")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Memory of the tracking process, frames 0-300")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "p1_memory.png"), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ B
RE_PROF = re.compile(r"\[profile\] refs (\d+), built (\d+), checked (\d+), selected (\d+), objects (\d+) \| los (\d+), "
                     r"proj (\d+), tri (\d+), enum (\d+), strip pts (\d+), cands (\d+), dfs (\d+) \| t tables ([\d.]+) "
                     r"s, build\+check ([\d.]+) s, sort ([\d.]+) s, prune ([\d.]+) s, tri ([\d.]+) s")
RE_PROF0 = re.compile(r"\[profile\] refs (\d+), built (\d+), checked (\d+), selected (\d+), objects (\d+) \| los (\d+), "
                      r"proj (\d+), tri (\d+), enum (\d+), strip pts (\d+), cands (\d+), dfs (\d+) \| t build\+check "
                      r"([\d.]+) s, sort ([\d.]+) s, prune ([\d.]+) s, tri ([\d.]+) s")
PROF_KEYS = ["refs", "built", "checked", "selected", "objects", "los", "proj", "tri", "enum", "strip_pts", "cands",
             "dfs", "t_tables", "t_build_check", "t_sort", "t_prune", "t_tri"]


def profile_rounds(run):
    """[profile] lines in log order, with the frame each belongs to."""
    out, frame = [], None
    with open(os.path.join(RUNS, run, "log.txt"), encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.match(r"^(Initial|Convergence) phase at frame (\d+)", ln)
            if m:
                frame = int(m.group(2))
                continue
            m = RE_PROF.search(ln)
            if m:
                vals = [float(x) for x in m.groups()]
                out.append({"frame": frame, **dict(zip(PROF_KEYS, vals))})
                continue
            m = RE_PROF0.search(ln)
            if m:
                vals = [float(x) for x in m.groups()]
                keys = [k for k in PROF_KEYS if k != "t_tables"]
                out.append({"frame": frame, "t_tables": 0.0, **dict(zip(keys, vals))})
    return out


def profile_by_combo(run):
    """[profile] lines with the frame and the camera combination (the "Combination ... cams=[..]" line before)."""
    out, frame, combo = [], None, None
    with open(os.path.join(RUNS, run, "log.txt"), encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = re.match(r"^(Initial|Convergence) phase at frame (\d+)", ln)
            if m:
                frame = int(m.group(2))
                continue
            m = re.match(r"^Combination \d+ cams=\[([\d, ]+)\]", ln)
            if m:
                combo = m.group(1).replace(" ", "").replace(",", "")
                continue
            m = RE_PROF.search(ln)
            if m:
                out.append({"frame": frame, "combo": combo, **dict(zip(PROF_KEYS, [float(x) for x in m.groups()]))})
    return out


def cmd_combo(run):
    """The work counters and times of each camera combination, early and late in a run."""
    pr = profile_by_combo(run)
    combos = sorted({p["combo"] for p in pr}, key=lambda c: (-len(c), c))
    lines = [f"[profile] per camera combination, {run}: mean per round (each combination runs 2 rounds per frame)"]
    keys = ["refs", "dfs", "cands", "strip_pts", "built", "checked", "selected", "t_tables", "t_build_check"]
    lines.append(f"{'combo':6s} {'frames':9s} " + " ".join(f"{k:>13s}" for k in keys))
    data = {}
    for c in combos:
        for a, b in [(4, 23), (95, 105), (190, 200), (281, 300)]:
            sel = [p for p in pr if p["combo"] == c and p["frame"] is not None and a <= p["frame"] <= b]
            if not sel:
                continue
            v = {k: statistics.mean(p[k] for p in sel) for k in keys}
            data[f"{c}|{a}-{b}"] = v
            lines.append(f"{c:6s} {a:3d}-{b:3d}   " + " ".join(f"{v[k]:13.4g}" for k in keys))
    write(f"b_combo_{run}", lines, data)


def per_frame_sum(run, key):
    fr = cr.frames_of(run)
    out = {}
    for f, r in fr.items():
        if key == "t_match_all":
            out[f] = sum(x.get("t_match", 0.0) for x in r["rounds"])
        elif key == "matched_4cam_2px":   # from the main thread's lines (see compare_runs.main_thread_rounds)
            out[f] = sum(m for c, m, t, _a in r["rounds_main"] if c == "0123" and t == 2.0)
        elif key == "matched_all":
            out[f] = sum(m for _c, m, _t, _a in r["rounds_main"])
        else:
            out[f] = r.get(key)
    return out


def cmd_b():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines, data = ["Phase B: stereo matching (frames 0-300 unless noted)", ""], {}

    # 1. profile counters, totals over the run
    for run in ("s50_b134", "s50_b2", "b300_b134", "b300_b2"):
        if not exists(run):
            continue
        pr = profile_rounds(run)
        tot = {k: sum(p[k] for p in pr) for k in PROF_KEYS}
        data[f"profile_{run}"] = tot
        lines.append(f"[profile] totals of {run} ({len(pr)} matching rounds): " + ", ".join(
            f"{k} {tot[k]:.3g}" for k in PROF_KEYS))
    # 2. matching and overlap time per frame against the baseline
    lines += ["", f"{'run':12s} {'match s/frame':>14s} {'f281-300':>9s} {'overlap s':>10s} {'IPR objects':>12s} "
                  f"{'4cam@2px':>9s} {'active long':>12s} {'fail lin.fit':>13s}"]
    rows = {}
    for run, what in B300:
        if not exists(run):
            continue
        tm = per_frame_sum(run, "t_match_all")
        to = per_frame_sum(run, "t_overlap")
        found = per_frame_sum(run, "ipr_found")
        m4 = per_frame_sum(run, "matched_4cam_2px")
        la = per_frame_sum(run, "la_new")
        flf = per_frame_sum(run, "fail_lf")
        conv = [f for f in sorted(tm) if f >= 4]
        late = [f for f in conv if f >= 281]
        mean = lambda d, fs: statistics.mean(d[f] for f in fs if d.get(f) is not None)  # noqa: E731
        rows[run] = {"match_s": mean(tm, conv), "match_late_s": mean(tm, late), "overlap_s": mean(to, conv),
                     "ipr_found": mean(found, conv), "matched_4cam_2px": mean(m4, conv), "active_long": mean(la, late),
                     "fail_lf": mean(flf, conv)}
        r = rows[run]
        lines.append(f"{run:12s} {r['match_s']:14.2f} {r['match_late_s']:9.2f} {r['overlap_s']:10.3f} "
                     f"{r['ipr_found']:12.0f} {r['matched_4cam_2px']:9.0f} {r['active_long']:12.0f} {r['fail_lf']:13.0f}")
    lines.append("match: IPR matching time per frame (all rounds); overlap: STB::checkRepeat; IPR objects, 4cam@2px "
                 "(matched in the 4-camera rounds at 2 px), fail lin.fit: means over frames 4-300; active long: mean "
                 "over frames 281-300")
    data["per_run"] = rows
    write("b_matching", lines, data)

    if rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for run, what in B300:
            if run in rows:
                tm = per_frame_sum(run, "t_match_all")
                fs = [f for f in sorted(tm) if f >= 4]
                ax.plot(fs, [tm[f] for f in fs], lw=1.5, color=COLORS[run], label=what)
        ax.set_xlabel("frame")
        ax.set_ylabel("IPR matching time per frame, s")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("Stereo-matching time per frame")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "b_matching_time.png"), dpi=130)
        plt.close(fig)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "s1"
    if cmd == "combo":
        cmd_combo(sys.argv[2])
        return
    {"s1": cmd_s1, "p2": cmd_p2, "p1": cmd_p1, "b": cmd_b}[cmd]()


if __name__ == "__main__":
    main()
