"""F7 (fix of the four-round flip, 2026-10-07): the flip rule (`cylinder_wand._flips`) applied offline to every logged
stage-4 run of the real data, before the reruns.

    python tests/cylinder_wand_gui/f7_flip_rule.py

Input: the calibration logs that f7_keep_before.py copied to data/f7_logs/ (runs 20261006/T0 and T2: the delivered
calibrations, T0's first delivery, every variant) and F3's logs (data/f3_*_before_flip4/).  Each start's stage 4
logs one noise estimate per round ("sigma per group", 4 decimals) until the weights fall back to equal ones; the
round the old rule (two states) fired logs that round's estimate in its message.  Per start: its rounds, whether it
reached the round limit (20), the round each rule fires at on its estimates ("alternates": the two-state rule F-a;
"jumps": more than 30 % in each of four rounds in a row, the new one), and which fires first.
Also on made-up sequences: a damped oscillation and a converging one (neither may fire), a two-state flip, a
four-round cycle.
Writes f7_flip_rule_out.txt and f7_flip_rule.json beside this script.
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "modules", "camera_calibration", "curved_calibration"))
import cylinder_wand as cw  # noqa: E402

ROUND = re.compile(r"round (\d+): rms ([\d.]+) px, \d+ LM iterations, rejected \d+, re-admitted \d+"
                   r"(?:, sigma per group ([\d., ]+) px)?")
FLIP = re.compile(r"round (\d+): the noise estimate flips between two states every round \(([\d., ]+) px and "
                  r"([\d., ]+) px per group\)")
START = re.compile(r"start (\d+): stage 3 rms [\d.]+ px(, as start (\d+)'s)?")


def vec(s):
    return np.array([float(x) for x in s.split(",")])


def starts_of(path):
    """Per start whose stage 4 ran: its rounds and its logged noise estimates (one per round, while estimated)."""
    out, cur = [], {"rounds": 0, "est": []}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ROUND.search(line)
            if m and "coarse round" not in line:
                cur["rounds"] += 1
                if m.group(3):
                    cur["est"].append(vec(m.group(3)))
                continue
            f = FLIP.search(line)
            if f:
                cur["est"].append(vec(f.group(3)))     # the round's own estimate (the log shows the weights after)
                cur["old_flip_logged"] = int(f.group(1))
                continue
            s = START.search(line)
            if s:
                if s.group(2) is None:
                    out.append({"start": int(s.group(1)), **cur})
                cur = {"rounds": 0, "est": []}
    return out


def first(hist, kind_wanted=None):
    """The round at which `_flips` first fires on the estimates (and which rule), as refine applies it."""
    for r in range(len(hist)):
        k = cw._flips(hist[:r + 1])
        if k and (kind_wanted is None or k == kind_wanted):
            return r, k
    return None, None


def old_rule(hist):
    """The rule before this fix: only "alternates"."""
    for r in range(2, len(hist)):
        a, b, c = hist[r], hist[r - 1], hist[r - 2]
        if np.max(np.abs(a / c - 1.0)) < cw.FLIP_CLOSE and np.max(np.abs(a / b - 1.0)) > cw.FLIP_JUMP:
            return r
    return None


def main():
    lines, rec = [], {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say(f"F7: the flip rule on the logged stage-4 runs (thresholds: close {cw.FLIP_CLOSE}, jump {cw.FLIP_JUMP}, "
        f"run {cw.FLIP_RUN})")
    paths = sorted(glob.glob(os.path.join(HERE, "data", "f7_logs", "*_calibration_log.txt")))
    paths += sorted(glob.glob(os.path.join(HERE, "data", "f3_*_before_flip4", "calibration_log.txt")))
    say("\nlog                                start  rounds  limit  estimates  old rule  new: first  rule        "
        "max jump of the last 4")
    n_limit = n_limit_caught = n_conv = n_conv_fired = 0
    for p in paths:
        name = (os.path.basename(p).replace("_calibration_log.txt", "") if "f7_logs" in p
                else os.path.basename(os.path.dirname(p)))
        for st in starts_of(p):
            h = st["est"]
            old = old_rule(h)
            r_new, k_new = first(h)
            limit = st["rounds"] >= 20
            jumps = [float(np.max(np.abs(h[-k] / h[-k - 1] - 1.0))) for k in range(1, min(4, len(h) - 1) + 1)]
            say(f"{name:34s} {st['start']:5d}  {st['rounds']:6d}  {'yes' if limit else 'no':5s}  {len(h):9d}  "
                f"{'-' if old is None else old:>8}  {'-' if r_new is None else r_new:>10}  {k_new or '-':10s}  "
                + ", ".join(f"{100 * j:.0f} %" for j in jumps[::-1]))
            rec.setdefault(name, []).append({"start": st["start"], "rounds": st["rounds"], "limit": limit,
                                              "estimates": [x.tolist() for x in h], "old_rule_round": old,
                                              "new_rule_round": r_new, "new_rule": k_new,
                                              "old_flip_logged": st.get("old_flip_logged")})
            if limit:
                n_limit += 1
                n_limit_caught += r_new is not None
            elif st.get("old_flip_logged") is None and len(h) == st["rounds"]:
                n_conv += 1                              # estimated in every round and no fallback: converged
                n_conv_fired += r_new is not None
    say(f"\nstarts that reached the round limit: {n_limit}; the new rule fires on {n_limit_caught} of them")
    for name, sts in rec.items():
        for s in sts:
            if s["limit"] and s["new_rule_round"] is None:
                say(f"    not caught: {name} start {s['start']}")
    say(f"starts whose weights converged without a fallback: {n_conv}; the new rule fires on {n_conv_fired} of them")
    early = [(n, s) for n, sts in rec.items() for s in sts if not s["limit"] and s["old_rule_round"] is not None]
    say(f"starts that ended before the limit after the old rule fired: {len(early)}; the new rule fires "
        + ", ".join(f"{d} round(s) earlier on {sum(1 for _n, s in early if s['old_rule_round'] - s['new_rule_round'] == d)}"
                    for d in sorted({s["old_rule_round"] - s["new_rule_round"] for _n, s in early})))
    other = [(n, s) for n, sts in rec.items() for s in sts if not s["limit"] and s["old_rule_round"] is None
             and s["estimates"]]
    say(f"starts that ended before the limit without the old rule (stopped by the repeated active set, or settled "
        f"and then not separable): {len(other)}; the new rule fires on "
        f"{sum(1 for _n, s in other if s['new_rule_round'] is not None)} of them:")
    for n, s in other:
        if s["new_rule_round"] is not None:
            say(f"    {n} start {s['start']}: {s['rounds']} rounds before; the new rule at round {s['new_rule_round']}")

    say("\nmade-up sequences (two groups)")
    t = np.arange(14)
    seqs = {
        "damped oscillation, factor 0.5 per round": [np.array([0.5, 0.5 * (1 + 0.8 * (-0.5) ** k)]) for k in t],
        "damped oscillation, factor 0.7 per round": [np.array([0.5, 0.5 * (1 + 0.8 * (-0.7) ** k)]) for k in t],
        "converging from a clamp (x0.1) without overshoot": [np.array([0.5, 0.5 * (1 - 0.9 * 0.5 ** k)]) for k in t],
        "two states": [np.array([0.7, 0.07]) if k % 2 else np.array([0.55, 0.55]) for k in t],
        "four-round cycle": [np.array(v) for v in [(0.70, 0.12), (0.42, 0.57), (0.78, 0.078), (0.57, 0.54)] * 4],
    }
    synth = {}
    for label, h in seqs.items():
        r, k = first(h)
        synth[label] = {"round": r, "rule": k}
        say(f"  {label:50s} fires at round {'-' if r is None else r} ({k or 'never'})")
    ok = (synth["damped oscillation, factor 0.5 per round"]["round"] is None
          and synth["converging from a clamp (x0.1) without overshoot"]["round"] is None
          and synth["two states"]["rule"] == "alternates" and synth["four-round cycle"]["round"] is not None)
    say(f"\nF7 {'PASS' if ok and n_limit_caught == n_limit and n_conv_fired == 0 else 'FAIL'} (criteria: every start "
        "that reached the limit is caught; no start whose weights converged is; the made-up sequences behave): "
        f"{n_limit_caught} of {n_limit} caught; {n_conv_fired} of {n_conv} converged; made-up "
        f"{'as expected' if ok else 'NOT as expected'}"
        + ("" if synth["damped oscillation, factor 0.7 per round"]["round"] is None else
           "; the slowly damped oscillation (factor 0.7) is caught too: a limit of the rule, see the report"))
    with open(os.path.join(HERE, "f7_flip_rule.json"), "w", encoding="utf-8") as fh:
        json.dump({"logs": rec, "made_up": synth}, fh, indent=1)
    with open(os.path.join(HERE, "f7_flip_rule_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
