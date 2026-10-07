"""P1 (diagnosis before the fixes plan, 2026-10-06-fixes-plan.md): why does the cylinder calibration run so long?

    python tests/cylinder_wand_gui/p1_log_digest.py

Reads the delivered calibrations' logs of runs 20261006/T0 and T2 (calibration_log.txt, written by
cylinder_wand_gui.run through the cases' calibrate.py: cylinder_wand.calibrate with noise weights "camera", six
starts).  Per start: its stage-4 rounds (refine), the time they took, the per-camera noise sigmas each round
estimated, how many detections each round rejected and re-admitted, and whether the sigmas flip between two states
(a sigma within 3 % of its value two rounds earlier and more than 30 % off the last round's).
Writes p1_log_digest_out.txt and p1_log_digest.json beside this script.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = {"T0": r"X:\Shijie Zhong\Refraction\Processed\20261006\T0\Results\calibration_log.txt",
        "T2": r"X:\Shijie Zhong\Refraction\Processed\20261006\T2\Results\calibration_log.txt"}
LINE = re.compile(r"^\[\s*([\d.]+) s\] ([^:]+): (.*)$")
ROUND = re.compile(r"round (\d+): rms ([\d.]+) px, (\d+) LM iterations, rejected (\d+), re-admitted (\d+)"
                   r"(?:, sigma per group ([\d., ]+) px)?")
START = re.compile(r"start (\d+): stage 3 rms ([\d.]+) px -> stage 4 rms ([\d.]+) px, active (\d+)/(\d+)")


def flips(sig):
    """First round whose sigmas are within 3 % of two rounds earlier and > 30 % off the last round's."""
    for k in range(2, len(sig)):
        a, b, c = sig[k], sig[k - 1], sig[k - 2]
        if a and b and c:
            d2 = max(abs(x / z - 1) for x, z in zip(a, c))
            d1 = max(abs(x / y - 1) for x, y in zip(a, b))
            if d2 < 0.03 and d1 > 0.30:
                return k
    return None


def main():
    out, lines = {}, []

    def say(s=""):
        print(s)
        lines.append(s)
    say("P1: the stage-4 rounds of the delivered calibrations of T0 and T2")
    for run, path in LOGS.items():
        starts, cur, t_prev = [], [], 0.0
        t_start = None
        for raw in open(path, encoding="utf-8"):
            m = LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            t, msg = float(m.group(1)), m.group(3)
            r = ROUND.search(msg)
            if r and "coarse" not in msg:
                if t_start is None:
                    t_start = t_prev
                sig = [float(x) for x in r.group(6).split(",")] if r.group(6) else None
                cur.append({"t": t, "rms": float(r.group(2)), "lm": int(r.group(3)), "rejected": int(r.group(4)),
                            "readmitted": int(r.group(5)), "sigma": sig})
            s = START.search(msg)
            if s:
                starts.append({"start": int(s.group(1)), "stage3_rms": float(s.group(2)),
                               "stage4_rms": float(s.group(3)), "active": int(s.group(4)), "n": int(s.group(5)),
                               "rounds": cur, "seconds": t - (t_start if t_start is not None else t)})
                cur, t_start = [], None
            t_prev = t
        say(f"\n{run}: {path}")
        say("  start  stage-3 rms  rounds  seconds  stage-4 rms  active      flip from round   sigmas (cam1, cam2) of "
            "the last four rounds; rejected / re-admitted")
        for st in starts:
            rd = st["rounds"]
            k = flips([x["sigma"] for x in rd])
            tail = "; ".join(f"({x['sigma'][0]:.3f}, {x['sigma'][1]:.3f}) {x['rejected']}/{x['readmitted']}"
                             for x in rd[-4:] if x["sigma"])
            say(f"  {st['start']:5d}  {st['stage3_rms']:11.3f}  {len(rd):6d}  {st['seconds']:7.1f}  "
                f"{st['stage4_rms']:11.5f}  {st['active']:5d}/{st['n']:<5d}  {str(k):>15s}   {tail}")
        tot = sum(st["seconds"] for st in starts)
        say(f"  stage 4 over all starts: {tot:.0f} s; rounds per start {[len(s['rounds']) for s in starts]} "
            f"(the limit is 20)")
        out[run] = starts
    with open(os.path.join(HERE, "p1_log_digest_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "p1_log_digest.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
