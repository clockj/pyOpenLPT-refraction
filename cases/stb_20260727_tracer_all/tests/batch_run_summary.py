"""The batch's finished runs, whole, from their logs (report section 12, end).  Read only.

For each run given that has DONE.json in its staging folder:
  - when it was tracked and delivered (batch_status.json, DONE.json);
  - its time in all, the sum of STB's own frame times and its saves' (count, median, sum), and the time outside them;
  - per 2,000 frames: the mean frame time (saves apart), the particles IPR found, the long tracks promoted, lost
    (to the inactive ones) and exited per frame, and the active long tracks at the block's end;
  - the active long tracks at the last frame (`la_end`, as run_all.py records it).

    python tests/batch_run_summary.py RUN ... > tests/batch_run_summary.txt
"""
from __future__ import annotations

import datetime
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import parse_log  # noqa: E402

BLOCK = 2000


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summary(cfg, status, run):
    sd = case.stage_dir(cfg, run)
    done_path = os.path.join(sd, "DONE.json")
    if not os.path.exists(done_path):
        print(f"{run}: not finished (no DONE.json)\n")
        return
    done, st = read_json(done_path), status[run]
    recs = parse_log.parse(os.path.join(sd, "log.txt"))
    n, last = len(recs), recs[-1]["frame"]
    tot = sum(r["t_total"] for r in recs)
    sv = [r["t_total"] for r in recs if r["saved"]]
    wall = st["wall_s"]
    print(f"{run}: tracked {st['start']} to {st['end']}, {n} frames (0..{last}); delivered {done['delivered']}, "
          f"{done['files']} files, {done['GB']:.1f} GB")
    print(f"  in all {wall / 3600:.2f} h = {wall / n:.2f} s per frame; STB's frame times {tot / 3600:.2f} h = "
          f"{tot / n:.2f} s per frame, with {len(sv)} saves (median {statistics.median(sv):.1f} s, together "
          f"{sum(sv) / 3600:.2f} h); outside them {(wall - tot) / n:.2f} s per frame")
    print("  frames         frame time  particles  promoted  lost  exited  long tracks")
    print("                 (no save)   found      ------ per frame -----  at the end")
    for a in range(0, last + 1, BLOCK):
        w = [r for r in recs if a <= r["frame"] < a + BLOCK and "la_n" in r]
        if not w:
            continue
        m = lambda k: statistics.mean(r.get(k, 0) for r in w)  # noqa: E731
        t = statistics.mean(r["t_total"] for r in w if not r["saved"])
        print(f"  {a:5d}-{min(a + BLOCK - 1, last):<5d}    {t:6.2f} s  {m('ipr_found'):9.0f}  {m('la_add'):8.0f}  "
              f"{m('li_add'):4.0f}  {m('ex_add'):6.1f}  {w[-1]['la_new']:11d}")
    print(f"  active long tracks at the last frame: {st['la_end']}\n")


def main():
    cfg = case.load_config()
    status = read_json(os.path.join(cfg["staging_root"], "batch_status.json"))
    print(f"The batch's finished runs, whole ({datetime.datetime.now():%d %b %Y, %H:%M}); report section 12, end.\n")
    for run in sys.argv[1:]:
        summary(cfg, status, run)


if __name__ == "__main__":
    main()
