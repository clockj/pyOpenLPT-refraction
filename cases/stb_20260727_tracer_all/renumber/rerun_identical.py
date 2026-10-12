"""R1: the new 10Hz_r_b_1 run on the renumbered images against the old run (read only).

Files 0-255 did not change in the renumbering, so frames 0-255 of the new run must repeat the old run's log exactly:
every counter of every frame (long and short tracks made, removed and kept, IPR particles, ...), timings left out.
From frame 256 on the old run saw a copy of frame 255 (nothing moved) and the new run the real frame 256, so they
must differ there.

    python renumber/rerun_identical.py [RUN]     (default 10Hz_r_b_1)
Writes renumber/rerun_identical_<RUN>.txt.  The new run is the batch's run with tracerConfig.txt's settings: for
10Hz_r_b_1, set aside on 11 Oct as <staging>\10Hz_r_b_1_spacing30 when the run was queued again with spacing 5.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import parse_log  # noqa: E402

TIMING = ("t_total", "t_", "time")


def counters(rec):
    """The record without timings, at every level (each IPR round carries its own t_match and t_shake)."""
    if isinstance(rec, dict):
        return {k: counters(v) for k, v in rec.items() if not any(k.startswith(t) for t in TIMING)}
    if isinstance(rec, list):
        return [counters(v) for v in rec]
    return rec


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "10Hz_r_b_1"
    cfg = case.load_config()
    new_dir = case.stage_dir(cfg, run) + "_spacing30"
    if not os.path.isdir(new_dir):
        new_dir = case.stage_dir(cfg, run)
    new_log = os.path.join(new_dir, "log.txt")
    old_log = os.path.join(case.stage_dir(cfg, run) + "_old_images", "log.txt")
    old = {r["frame"]: counters(r) for r in parse_log.parse(old_log)}
    new = {r["frame"]: counters(r) for r in parse_log.parse(new_log)}
    upto = min(255, max(new))
    diff = [f for f in range(0, upto + 1) if old.get(f) != new.get(f)]
    lines = [f"R1 {run}: the new run (renumbered images) against the old run, frame by frame, timings left out", "",
             f"frames compared: 0-{upto} ({upto + 1} frames; the new run is at frame {max(new)})",
             f"counters compared per frame: {sorted(next(iter(new.values())).keys())}",
             f"frames that differ in 0-{upto}: {len(diff)}" + (f" (first {diff[:5]})" if diff else "")]
    if diff:
        f = diff[0]
        lines.append(f"  frame {f}: old {old.get(f)}")
        lines.append(f"  frame {f}: new {new.get(f)}")
    if max(new) >= 256:
        lines.append(f"frame 256: old {old.get(256)}")
        lines.append(f"frame 256: new {new.get(256)}")
        lines.append(f"frame 256 differs, as it must: {old.get(256) != new.get(256)}")
    ok = not diff and upto == 255 and (max(new) < 256 or old.get(256) != new.get(256))
    lines += ["", "PASS" if ok else ("NOT YET (the new run has not reached frame 256)" if upto < 255 and not diff
                                     else "FAIL")]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, f"rerun_identical_{run}.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
