"""Short test runs before the batch (S1-S3): the batch's settings on a few frames.

    python tests/smoke_test.py run NAME RUN F0 F1 [--threads T] [--threshold I]
        STB on frames F0-F1 of RUN with the batch's settings (the staging config of RUN, frames and threads changed)
        into <staging_root>\\smoke\\NAME (config.json, log.txt, run_info.json, resources.csv, Results_tracer\\).
        --threshold changes the tracer finder's intensity threshold (tracerConfig: 20).
    python tests/smoke_test.py identical NAME REF_DIR [--out OUT]
        S1: the same results as a reference run (a run folder with log.txt and Results\\): every per-frame counter
        of the logs (the stb_improve campaign's compare_runs.signature) and every output file byte by byte.
    python tests/smoke_test.py summary NAME ... [--out OUT]
        Per run: time per frame, detections per camera, matched objects and objects left after shaking per IPR
        round, objects found, tracks.

--out writes the report to tests/OUT.txt and tests/OUT.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import make_configs  # noqa: E402
import parse_log  # noqa: E402

CMP_PATH = os.path.join(case.REPO, "tests", "stb_improve", "compare_runs.py")


def compare_runs():
    spec = importlib.util.spec_from_file_location("compare_runs", CMP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def smoke_dir(cfg, name):
    return os.path.join(cfg["staging_root"], "smoke", name)


def write_config(cfg, name, run, f0, f1, threads=None, threshold=None):
    """The config of a test run in <staging_root>\\smoke\\NAME; returns its path."""
    d = smoke_dir(cfg, name)
    if os.path.exists(os.path.join(d, "log.txt")):
        raise SystemExit(f"{d} already holds a run")
    n = make_configs.frames_of(run)
    c = make_configs.stage_config(cfg, n, frames=(f0, f1), threads=threads)
    sd = case.stage_dir(cfg, run)
    c["cameras"] = [{"file": os.path.join(sd, "camFile", f), "max_intensity": 255} for f in cfg["camfiles"]]
    c["image_lists"] = [os.path.join(sd, "imgLists", f"cam{k}ImageNames.txt") for k in range(4)]
    if threshold is not None:
        c["objects"][0]["tracer"]["intensity_threshold"] = threshold
    c["_smoke"] = {"run": run, "frames": [f0, f1], "threads": c["threads"],
                   "threshold": c["objects"][0]["tracer"]["intensity_threshold"]}
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "config.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(make_configs.json_text(c))
    return path


def cmd_run(cfg, name, run, f0, f1, threads, threshold=None):
    path = write_config(cfg, name, run, f0, f1, threads, threshold)
    cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "run_stb.py"), path, "--quiet"]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_under(root):
    out = {}
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = p
    return out


def cmd_identical(cfg, name, ref):
    cr = compare_runs()
    mine = smoke_dir(cfg, name)
    fa, fb = cr.frames_of(ref), cr.frames_of(mine)   # absolute folders: os.path.join keeps them
    lines, diffs = [f"S1 identical: {name} ({mine}) against {ref}"], []
    common = sorted(set(fa) & set(fb))
    only = sorted(set(fa) ^ set(fb))
    if only:
        diffs.append(f"frames in one run only: {only[:10]}")
    for f in common:
        sa, sb = cr.signature(fa[f]), cr.signature(fb[f])
        if sa != sb:
            keys = [k for k in sa if sa[k] != sb[k]]
            diffs.append(f"frame {f}: log differs in {keys}: {[(sa[k], sb[k]) for k in keys][:2]}")
    lines.append(f"log: {len(common)} frames compared ({common[0]}-{common[-1]}), "
                 f"{sum(1 for d in diffs if d.startswith('frame '))} differ")
    ra = files_under(os.path.join(ref, "Results"))
    rb = files_under(os.path.join(mine, cfg["output_folder"]))
    same = 0
    for k in sorted(set(ra) | set(rb)):
        if k not in ra or k not in rb:
            diffs.append(f"file only in {'the reference' if k in ra else name}: {k}")
        elif os.path.getsize(ra[k]) != os.path.getsize(rb[k]) or sha256(ra[k]) != sha256(rb[k]):
            diffs.append(f"file differs: {k}")
        else:
            same += 1
    lines.append(f"output files: {len(set(ra) | set(rb))}, {same} byte-identical")
    lines.append("IDENTICAL" if not diffs else f"DIFFERENT ({len(diffs)}):")
    lines += ["  " + d for d in diffs[:40]]
    return lines, {"name": name, "ref": ref, "frames": len(common), "files": len(set(ra) | set(rb)),
                   "files_identical": same, "identical": not diffs, "diffs": diffs}


def run_summary(cfg, name):
    d = smoke_dir(cfg, name)
    recs = parse_log.parse(os.path.join(d, "log.txt"))
    with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    info = {}
    if os.path.exists(os.path.join(d, "run_info.json")):
        with open(os.path.join(d, "run_info.json"), encoding="utf-8") as f:
            info = json.load(f)
    conv = [r for r in recs if r["phase"] == "conv" and not r["saved"]]
    first = [r["rounds"][0] for r in recs if r["rounds"]]
    return {
        "name": name, "run": c.get("_smoke", {}).get("run"), "threads": c["threads"],
        "frames": f"{recs[0]['frame']}-{recs[-1]['frame']}" if recs else "-", "n": len(recs),
        "t_init": statistics.mean(r["t_total"] for r in recs if r["phase"] == "init") if recs else None,
        "t_conv": statistics.mean(r["t_total"] for r in conv) if conv else None,
        "t_conv_last5": statistics.mean(r["t_total"] for r in conv[-5:]) if conv else None,
        "det_cam": [statistics.mean(r["det"][k] for r in first if len(r["det"]) == 4) for k in range(4)] if first else None,
        "matched_first": statistics.mean(r.get("matched", 0) for r in first) if first else None,
        "after_shake_first": statistics.mean(r.get("after_shake", 0) for r in first) if first else None,
        "ipr_found": statistics.mean(r["ipr_found"] for r in recs if "ipr_found" in r) if any("ipr_found" in r for r in recs) else None,
        "la_last": recs[-1].get("la_new") if recs else None,
        "sa_last": recs[-1].get("sa_new") if recs else None,
        "wall_s": info.get("wall_s"), "status": info.get("status"),
        "per_frame": [{"frame": r["frame"], "phase": r["phase"], "t": r["t_total"], "found": r.get("ipr_found"),
                       "la": r.get("la_new"), "sa": r.get("sa_new"),
                       "rounds": [(x.get("cams"), x.get("det"), x.get("matched"), x.get("after_shake"),
                                   x.get("t_match"), x.get("t_shake")) for x in r["rounds"]]} for r in recs],
    }


def cmd_summary(cfg, names):
    rows = [run_summary(cfg, n) for n in names]
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    f0 = lambda x: "-" if x is None else f"{x:.0f}"  # noqa: E731
    lines = [f"{'name':26s} {'run':11s} {'thr':>3s} {'frames':7s} {'s/init':>7s} {'s/conv':>7s} {'last5':>7s} "
             f"{'detections per camera (1st round)':>36s} {'matched':>8s} {'shaken':>7s} {'found':>7s} "
             f"{'long':>6s} {'short':>6s}"]
    for s in rows:
        det = " ".join(f"{x:8.0f}" for x in s["det_cam"]) if s["det_cam"] else "-"
        lines.append(f"{s['name']:26s} {str(s['run']):11s} {s['threads']:3d} {s['frames']:7s} {f2(s['t_init']):>7s} "
                     f"{f2(s['t_conv']):>7s} {f2(s['t_conv_last5']):>7s} {det:>36s} {f0(s['matched_first']):>8s} "
                     f"{f0(s['after_shake_first']):>7s} {f0(s['ipr_found']):>7s} {str(s['la_last']):>6s} "
                     f"{str(s['sa_last']):>6s}")
    lines.append("s/init, s/conv: mean time per frame in the initial and the convergence phase (frames that save left "
                 "out); last5: the last five convergence frames; detections, matched, shaken: the first IPR round "
                 "(all four cameras), mean over frames; found: objects IPR found; long/short: active tracks at the end")
    return lines, rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("run")
    p.add_argument("f0", type=int)
    p.add_argument("f1", type=int)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--threshold", type=int, default=None, help="the tracer finder's intensity threshold")
    p = sub.add_parser("identical")
    p.add_argument("name")
    p.add_argument("ref")
    p.add_argument("--out")
    p = sub.add_parser("summary")
    p.add_argument("names", nargs="+")
    p.add_argument("--out")
    args = ap.parse_args()
    cfg = case.load_config()
    if args.cmd == "run":
        sys.exit(cmd_run(cfg, args.name, args.run, args.f0, args.f1, args.threads, args.threshold))
    if args.cmd == "identical":
        lines, data = cmd_identical(cfg, args.name, args.ref)
    else:
        lines, data = cmd_summary(cfg, args.names)
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(os.path.join(case.TESTS, args.out + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        case.save_json(os.path.join(case.TESTS, args.out + ".json"), data)


if __name__ == "__main__":
    main()
