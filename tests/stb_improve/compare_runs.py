"""Compare benchmark runs of the stb_improve campaign (runs/RUN, written by stbrun.py).

    python compare_runs.py identical RUN_A RUN_B [--frames A B] [--out NAME]
        Test S1: the same results.  Compares, frame by frame, everything the log reports except times (phase, IPR
        rounds: cameras, detections, matched objects, objects after shaking; objects found; the track counters; the
        failures), and every file under Results/ byte by byte (SHA-256).  Prints IDENTICAL or the differences.
    python compare_runs.py timing RUN ... [--out NAME]
        Per run: frames, time per frame early (frames 4-23), late (last 20) and their ratio, total frame time, peak
        private memory, and the CPU other programs used (cores) while it ran.
    python compare_runs.py restart AGED FRESH [--frames A B] [--out NAME]
        Test R300: the aged process's frames against the fresh process's (restarted from saved tracks): times per
        frame and their ratio, and whether the counters agree.

--out NAME also writes the report to results/NAME.txt (and a JSON beside it).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
CASE_DIR = os.path.join(REPO, "cases", "stb_20260727_10Hz_r_b_1")


def _parse_log_module():
    spec = importlib.util.spec_from_file_location("parse_log", os.path.join(CASE_DIR, "parse_log.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PL = _parse_log_module()
COUNTERS = ["phase", "ipr_found", "sa_n", "sa_add", "sa_rm", "sa_new", "la_n", "la_add", "la_rm", "la_new", "ex_n",
            "ex_add", "ex_new", "li_n", "li_add", "li_new", "ex_del", "fail_shake", "fail_lf", "saved"]


RE_FRAME = re.compile(r"^(Initial|Convergence) phase at frame (\d+)")
RE_COMBO = re.compile(r"^Combination \d+ cams=\[([\d, ]+)\]")
RE_MATCHED = re.compile(r"Matched (\d+) objects\. \( 2D tol: ([\d.]+), t:[\d.]+ s\)")
RE_AFTER = re.compile(r"After shaking: (\d+) objects remain")


def main_thread_rounds(run):
    """Per frame, the IPR rounds as (cameras, matched, 2D tol, after shaking), read only from lines the main thread
    prints ("Combination ... cams=", "Matched ...", "After shaking ..."), so HEAD's racy detection line cannot hide
    a round."""
    out, frame, combo = {}, None, None
    with open(os.path.join(RUNS, run, "log.txt"), encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = RE_FRAME.match(ln)
            if m:
                frame, combo = int(m.group(2)), None
                out[frame] = []
                continue
            if frame is None:
                continue
            m = RE_COMBO.match(ln)
            if m:
                combo = m.group(1).replace(" ", "").replace(",", "")
            for m in RE_MATCHED.finditer(ln):
                out[frame].append([combo, int(m.group(1)), float(m.group(2)), None])
            m = RE_AFTER.search(ln)
            if m and out[frame]:
                out[frame][-1][3] = int(m.group(1))
    return {k: [tuple(r) for r in v] for k, v in out.items()}


def frames_of(run):
    recs = {r["frame"]: r for r in PL.parse(os.path.join(RUNS, run, "log.txt"))}
    rounds = main_thread_rounds(run)
    for f, r in recs.items():
        r["rounds_main"] = rounds.get(f, [])
    return recs


def signature(rec):
    """Everything a frame's log says except times."""
    sig = {k: rec.get(k) for k in COUNTERS}
    # The per-camera detection counts are left out: HEAD's IPR prints them from inside a parallel loop over the
    # cameras, so they come in the threads' finishing order and sometimes with their digits interleaved (the output
    # files decide identity anyway)
    sig["rounds"] = rec["rounds_main"]
    return sig


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def result_files(run):
    root = os.path.join(RUNS, run, "Results")
    out = {}
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = p
    return out


def in_range(frames, rng):
    return [f for f in sorted(frames) if rng is None or rng[0] <= f <= rng[1]]


def cmd_identical(a, b, rng):
    fa, fb = frames_of(a), frames_of(b)
    lines, diffs = [f"S1 identical outputs: {a} vs {b}"], []
    common = [f for f in in_range(fa, rng) if f in fb]
    only = sorted(set(in_range(fa, rng)) ^ set(f for f in in_range(fb, rng)))
    if only:
        diffs.append(f"frames in one run only: {only[:10]}{'...' if len(only) > 10 else ''}")
    for f in common:
        sa, sb = signature(fa[f]), signature(fb[f])
        if sa != sb:
            keys = [k for k in sa if sa[k] != sb[k]]
            diffs.append(f"frame {f}: log differs in {keys}: {[(sa[k], sb[k]) for k in keys][:2]}")
    lines.append(f"log: {len(common)} frames compared ({common[0] if common else '-'}-{common[-1] if common else '-'}), "
                 f"{sum(1 for d in diffs if d.startswith('frame '))} differ")
    ra, rb = result_files(a), result_files(b)
    n_same = 0
    for name in sorted(set(ra) | set(rb)):
        if name not in ra or name not in rb:
            diffs.append(f"file only in {'A' if name in ra else 'B'}: {name}")
            continue
        if os.path.getsize(ra[name]) != os.path.getsize(rb[name]) or sha256(ra[name]) != sha256(rb[name]):
            diffs.append(f"file differs: {name}")
        else:
            n_same += 1
    lines.append(f"Results/: {len(set(ra) | set(rb))} files, {n_same} byte-identical")
    lines.append("IDENTICAL" if not diffs else f"DIFFERENT ({len(diffs)}):")
    lines += ["  " + d for d in diffs[:40]]
    return lines, {"a": a, "b": b, "frames_compared": len(common), "files": len(set(ra) | set(rb)),
                   "files_identical": n_same, "identical": not diffs, "diffs": diffs}


def resources(run):
    path = os.path.join(RUNS, run, "resources.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def summary(run):
    fr = frames_of(run)
    order = sorted(fr)
    conv = [f for f in order if fr[f]["phase"] == "conv" and not fr[f]["saved"]]
    t = [fr[f]["t_total"] for f in conv]
    early = t[:20]
    late = t[-20:]
    res = resources(run)
    info = {}
    p = os.path.join(RUNS, run, "run_info.json")
    if os.path.exists(p):
        with open(p) as f:
            info = json.load(f)
    others = [float(r["others_cores"]) for r in res] if res else []
    return {
        "run": run, "build": info.get("build", ""), "frames": f"{order[0]}-{order[-1]}" if order else "",
        "n_frames": len(order), "conv_unsaved": len(conv),
        "early_s": statistics.mean(early) if early else float("nan"),
        "late_s": statistics.mean(late) if late else float("nan"),
        "growth": (statistics.mean(late) / statistics.mean(early)) if early and late else float("nan"),
        "sum_frame_s": sum(fr[f]["t_total"] for f in order),
        "peak_private_MB": max((float(r["private_MB"]) for r in res), default=float("nan")),
        "others_cores_mean": statistics.mean(others) if others else float("nan"),
        "others_cores_max": max(others) if others else float("nan"),
        "wall_s": info.get("wall_s"),
    }


def cmd_timing(runs):
    rows = [summary(r) for r in runs]
    lines = [f"{'run':28s} {'build':10s} {'frames':9s} {'early s':>8s} {'late s':>8s} {'growth':>7s} "
             f"{'sum min':>8s} {'peak GB':>8s} {'others':>7s}"]
    for s in rows:
        lines.append(f"{s['run']:28s} {s['build']:10s} {s['frames']:9s} {s['early_s']:8.2f} {s['late_s']:8.2f} "
                     f"{s['growth']:7.2f} {s['sum_frame_s'] / 60:8.1f} {s['peak_private_MB'] / 1024:8.2f} "
                     f"{s['others_cores_mean']:7.2f}")
    lines.append("early: mean time of the first 20 convergence frames; late: of the last 20 (frames that save are "
                 "left out); others: mean cores used by other programs")
    return lines, rows


def cmd_restart(aged, fresh, rng):
    fa, ff = frames_of(aged), frames_of(fresh)
    common = [f for f in in_range(ff, rng) if f in fa and not fa[f]["saved"] and not ff[f]["saved"]]
    ta = [fa[f]["t_total"] for f in common]
    tf = [ff[f]["t_total"] for f in common]
    # the fresh process starts with no lost or exited tracks, so their running totals are left out; its tracks come
    # from files (positions rounded to 1e-6 mm), so small differences can grow over the frames
    def restart_sig(rec):
        sig = signature(rec)
        for k in ("li_n", "li_new", "ex_n", "ex_new"):
            sig.pop(k, None)
        return sig
    same = sum(1 for f in common if restart_sig(fa[f]) == restart_sig(ff[f]))
    first_diff = next((f for f in common if restart_sig(fa[f]) != restart_sig(ff[f])), None)
    lines = [f"R300 restart test: aged {aged} vs fresh {fresh}, frames {common[0]}-{common[-1]} ({len(common)})",
             f"mean time per frame: aged {statistics.mean(ta):.2f} s, fresh {statistics.mean(tf):.2f} s, "
             f"aged/fresh {statistics.mean(ta) / statistics.mean(tf):.2f}",
             f"median: aged {statistics.median(ta):.2f} s, fresh {statistics.median(tf):.2f} s",
             f"frames whose log (all but times and the lost/exited totals) agrees: {same} of {len(common)}"
             + (f" (first difference at frame {first_diff})" if first_diff is not None else "")]
    return lines, {"aged": aged, "fresh": fresh, "frames": len(common), "mean_aged": statistics.mean(ta),
                   "mean_fresh": statistics.mean(tf), "ratio": statistics.mean(ta) / statistics.mean(tf),
                   "log_agree": same, "first_diff": first_diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["identical", "timing", "restart"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "identical":
        lines, data = cmd_identical(args.runs[0], args.runs[1], args.frames)
    elif args.cmd == "timing":
        lines, data = cmd_timing(args.runs)
    else:
        lines, data = cmd_restart(args.runs[0], args.runs[1], args.frames)
    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(RESULTS, exist_ok=True)
        with open(os.path.join(RESULTS, args.out + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        with open(os.path.join(RESULTS, args.out + ".json"), "w") as f:
            json.dump(data, f, indent=1, default=str)
    if args.cmd == "identical" and not data["identical"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
