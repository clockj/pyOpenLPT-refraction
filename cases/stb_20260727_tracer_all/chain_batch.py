"""Start the next batch when the running one ends (your choice of 11 Oct: after 10Hz_r_b_2, track 10Hz_r_b_1 again and
then the ten 15-35 Hz runs, with the new settings; report section 12).

    python chain_batch.py --after PID [--tag spacing30] [--dry-run] RUN ...

1. Waits for the batch process PID to exit.  The handle is opened at once, so a reused PID cannot fool it.
2. Checks that the batch ended well: every run it was tracking or copying has DONE.json (its delivery checked).
   Otherwise it starts nothing and says why in batch_log.txt.
3. Checks each RUN's staging folder: config.json read by the C++ reader with the case's settings (frames, linking
   spacing, threshold).  A RUN that still has DONE.json there (an earlier run not set aside) is left out, with a
   warning; run_all.py would skip it anyway.  Any other problem: nothing is started.
4. Keeps the earlier record of a RUN that is tracked again under previous_<tag> in batch_status.json.
5. Runs run_all.py with the RUNs and waits for it (its output goes where this process's goes).
--dry-run: the same checks, printed, without stopping at a failed one; nothing is written or started.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys

import case
import make_configs
from run_all import Batch, now

HERE = os.path.dirname(os.path.abspath(__file__))
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INFINITE = 0xFFFFFFFF
UNFINISHED = ("tracking", "tracked", "copying", "tracking failed", "copy failed")


def wait_for(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        raise SystemExit(f"no process {pid} to wait for")
    k32.WaitForSingleObject(h, INFINITE)
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value


def save_status(batch):
    tmp = batch.status_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(batch.status, f, indent=2)
    os.replace(tmp, batch.status_path)


def check_config(cfg, run):
    """The staging config of a run must hold the case's settings and be read by the C++ reader."""
    p = os.path.join(case.stage_dir(cfg, run), "config.json")
    with open(p, encoding="utf-8") as f:
        obj = json.load(f)["objects"][0]
    want = make_configs.run_object(cfg, run)
    for sect, field in (("convergence_phase", "avg_spacing_vox"), ("tracer", "intensity_threshold")):
        assert obj[sect][field] == want[sect][field], f"{sect}.{field} {obj[sect][field]} != {want[sect][field]}"
    return make_configs.check_cpp(p, make_configs.last_frame(cfg, make_configs.frames_of(run)), obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--after", type=int, required=True, help="pid of the running batch")
    ap.add_argument("--tag", default="spacing30", help="earlier records are kept as previous_<tag>")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = case.load_config()
    dry = args.dry_run
    say = print if dry else Batch(cfg).say
    say(f"chain (pid {os.getpid()}{', dry run' if dry else ''}): waits for the batch (pid {args.after}) to end, "
        f"then starts {args.runs}")
    code = wait_for(args.after)

    batch = Batch(cfg)  # the status as the ended batch left it
    undelivered = [r for r, st in batch.status.items() if st.get("state") in UNFINISHED
                   and not os.path.exists(os.path.join(case.stage_dir(cfg, r), "DONE.json"))]
    if undelivered:
        say(f"chain: the batch (pid {args.after}) ended (exit {code}) with {undelivered} not delivered; the next batch "
            f"is NOT started")
        if not dry:
            return 1

    runs, problems = [], []
    for run in args.runs:
        sd = case.stage_dir(cfg, run)
        if os.path.exists(os.path.join(sd, "DONE.json")):
            say(f"chain: WARNING {run} still has DONE.json in {sd} (the earlier run was not set aside); left out")
            continue
        try:
            got = check_config(cfg, run)
            if dry:
                print(f"  {run}: config ok, C++ reads frames {got['frames']}, spacing {got['avg_spacing_vox']:g}, "
                      f"threshold {got['threshold']}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{run}: {type(e).__name__} {e}")
            continue
        runs.append(run)
    if problems:
        say(f"chain: the next batch is NOT started; config problems: {problems}")
        if not dry:
            return 1

    for run in runs:
        st = batch.status.get(run)
        if st and st.get("state") in ("done", "tracking") and (st.get("state") == "done" or dry):
            new = {k: v for k, v in st.items() if k.startswith("previous_")}
            new[f"previous_{args.tag}"] = {k: v for k, v in st.items() if not k.startswith("previous_")}
            new["state"] = "queued again"
            batch.status[run] = new
            say(f"chain: {run}: its earlier record kept as previous_{args.tag}" +
                (f": {json.dumps(new)[:300]}" if dry else ""))
    if not dry:
        save_status(batch)

    cmd = [sys.executable, os.path.join(HERE, "run_all.py"), *runs]
    say(f"chain: the batch (pid {args.after}) ended (exit {code}); starting run_all.py {runs}")
    if dry:
        print("  would run:", " ".join(cmd))
        return 0
    rc = subprocess.call(cmd, cwd=HERE)
    say(f"chain: run_all.py ended (exit {rc}) at {now()}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
