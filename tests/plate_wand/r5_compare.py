"""R5 (2026-10-02-perturbed-starts-plan.md): W7 (rigs A and B, bad data, from scratch) rerun on the code with the
perturbed restarts, against the run before it (runs_v4/w7_bad_data_v2.json).

Criteria: every answer still at its data's minimum (the same minimum as the fit started from the truth on the same
data); its distance from the clean data's answer (z RMS and max, in its own sigma) unchanged or smaller.  Rig A has one
window (no restarts: unchanged to the last digit expected); rig B has two windows with two cameras each.
Output: r5_compare_out.txt
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

before = json.loads((HERE / "runs_v4" / "w7_bad_data_v2.json").read_text())
after = json.loads((HERE / "w7_bad_data.json").read_text())
log = prigs.Tee(HERE / "r5_compare_out.txt")
log("R5: W7 on the code with the perturbed restarts, against runs_v4/w7_bad_data_v2.json")
ok_all = True
for key in [k for k in before if k != "pass"]:
    b, a = before[key], after[key]
    p = a.get("perturbed") or {}
    at_min = bool(a.get("same_as_truth_start"))
    no_further = (a["z_vs_clean_rms"] <= b["z_vs_clean_rms"] + 1e-6 and a["z_vs_clean_max"] <= b["z_vs_clean_max"] + 1e-6)
    identical = (a["rms_px"] == b["rms_px"] and a["active"] == b["active"] and a["z_vs_clean_rms"] == b["z_vs_clean_rms"])
    ok = at_min and no_further
    ok_all &= ok
    rnds = p.get("rounds") or []
    log(f"  {key:11s}: at its data's minimum {b.get('same_as_truth_start')} -> {at_min}; z vs the clean fit RMS "
        f"{b['z_vs_clean_rms']:.3f} -> {a['z_vs_clean_rms']:.3f}, max {b['z_vs_clean_max']:.3f} -> "
        f"{a['z_vs_clean_max']:.3f}; rms {b['rms_px']:.6f} -> {a['rms_px']:.6f}, active {b['active']} -> {a['active']}"
        f"; {'identical' if identical else 'CHANGED'}; W7 verdict {b.get('pass')} -> {a.get('pass')}"
        + (f"; restarts: {len(rnds)} round(s), {'improved' if p.get('improved') else 'no change'}, "
           f"{p.get('seconds', 0.0):.0f} s of {a['seconds']:.0f} s" if p else "; no restarts")
        + ("" if ok else "  <-- FAIL"))
log(f"R5: {'PASS' if ok_all else 'FAIL'}")
