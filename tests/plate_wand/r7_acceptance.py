"""R7 (2026-10-03, after R4; the report's section 7, approved "ok, go ahead and compare"): how a restart is accepted,
compared.

Three versions of the calibration on the same data:
  before    no restarts (runs_v4/: W5 run 4, W7 run 2; R4's runs "without");
  choose    the restarts, their refit judged with the starts by `_choose` (the most observations kept, then the
            likelihood; runs_v5/: W5 run 5, W7 run 3, R4 run 1);
  capped    the restarts, their refit accepted when its capped cost (every usable observation, each capped at the 5
            sigma rejection limit) is lower than the answer's (the current code: w5_from_scratch.json,
            r4_group_rigs.json, w7_bad_data.json).
Per case with restarts (W5's 23 multi-window rigs, R4's 60 cases, W7's 6 rig-B scenarios): the verdict against the fit
started from the truth (same / equivalent / DIFFERENT), the RMS and the active observations; every case whose answer
differs between "choose" and "capped"; and the counts.  "Worse" (the plan's sense): a different minimum with a higher
RMS than before.
Output: r7_acceptance_out.txt / .json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

log = prigs.Tee(HERE / "r7_acceptance_out.txt")


def load(p):
    return json.loads((HERE / p).read_text())


def verdict(r):
    if r is None or "error" in r:
        return "ERROR"
    return "same" if r["ok"] else "equivalent" if r["equivalent"] else "DIFFERENT"


def rounds_note(r):
    p = (r.get("starts") or {}).get("perturbed") or {}
    out = []
    for k, rd in enumerate(p.get("rounds") or []):
        if rd.get("rms_after_px") is None:
            out.append(f"round {k + 1}: none lower")
        elif "accepted" in rd:
            out.append(f"round {k + 1}: refit {rd['rms_after_px']:.6f}, capped {rd.get('capped_rel', float('nan')):+.1e}, "
                       f"{'accepted' if rd['accepted'] else 'not accepted'}")
        else:
            out.append(f"round {k + 1}: refit {rd['rms_after_px']:.6f}, {'chosen' if rd.get('chosen') else 'not chosen'}")
    return "; ".join(out)


rows = []
# ---------------------------------------------------------------- W5
w5 = {v: {r["seed"]: r for r in load(p)["runs"]} for v, p in
      (("before", "runs_v4/w5_from_scratch_v4.json"), ("choose", "runs_v5/w5_from_scratch_v5.json"),
       ("capped", "w5_from_scratch.json"))}
for s in sorted(w5["capped"]):
    if w5["capped"][s].get("kind") == "one window":
        continue
    rows.append({"set": "W5", "case": f"seed {s} ({w5['capped'][s]['noise_px']:g} px)",
                 **{v: w5[v][s] for v in w5}})
# ---------------------------------------------------------------- R4
r4_now = {(r["seed"], r["noise_px"], r["mode"]): r for r in load("r4_group_rigs.json")["runs"]}
r4_old = {(r["seed"], r["noise_px"], r["mode"]): r for r in load("runs_v5/r4_group_rigs_v1.json")["runs"]}
for (s, v, m) in sorted(r4_now):
    if m != "with":
        continue
    rows.append({"set": "R4", "case": f"seed {s} ({v:g} px)", "before": r4_now[(s, v, "without")],
                 "choose": r4_old[(s, v, "with")], "capped": r4_now[(s, v, "with")],
                 "before_v1": r4_old[(s, v, "without")]})
same_without = all(abs(r["before"]["rms_px"] - r["before_v1"]["rms_px"]) <= 1e-12 * r["before_v1"]["rms_px"]
                   for r in rows if r["set"] == "R4")
# ---------------------------------------------------------------- W7, rig B
w7 = {v: load(p) for v, p in (("before", "runs_v4/w7_bad_data_v2.json"), ("choose", "runs_v5/w7_bad_data_v3.json"),
                               ("capped", "w7_bad_data.json"))}


def w7_rec(d):
    return {"ok": bool(d["same_as_truth_start"]), "equivalent": bool(d["same_as_truth_start"]), "rms_px": d["rms_px"],
            "active": d["active"], "starts": {"perturbed": d.get("perturbed")}, "z_vs_clean_rms": d["z_vs_clean_rms"]}


for key in [k for k in w7["capped"] if k.startswith("B_")]:
    rows.append({"set": "W7", "case": key, **{v: w7_rec(w7[v][key]) for v in w7}})

# ---------------------------------------------------------------- per case
log("R7: how a restart is accepted -- before (no restarts), choose (the starts' rule), capped (the capped cost)")
log(f"  R4's runs without the restarts are the same in both R4 runs (RMS to 1e-12): {same_without}")
changed, worse = [], []
for r in rows:
    vb, vc, vn = verdict(r["before"]), verdict(r["choose"]), verdict(r["capped"])
    diff = (abs(r["capped"]["rms_px"] - r["choose"]["rms_px"]) > 1e-9 * r["choose"]["rms_px"]
            or r["capped"]["active"] != r["choose"]["active"] or vn != vc)
    is_worse = bool(vn != "same" and vb != "ERROR" and (vb == "same" or r["capped"]["rms_px"] > r["before"]["rms_px"] * (1 + 1e-6)))
    r.update({"v_before": vb, "v_choose": vc, "v_capped": vn, "changed": diff, "worse": is_worse})
    if diff:
        changed.append(r)
    if is_worse:
        worse.append(r)
    if diff or vb != "same" or vn != "same":
        log(f"  {r['set']} {r['case']:18s}: {vb} -> {vc} -> {vn}; rms {r['before']['rms_px']:.6f} -> "
            f"{r['choose']['rms_px']:.6f} -> {r['capped']['rms_px']:.6f}; active {r['before']['active']} -> "
            f"{r['choose']['active']} -> {r['capped']['active']}"
            + (f" (truth-started {r['capped'].get('ref_rms_px', float('nan')):.6f}, "
               f"{r['capped'].get('active_ref', '-')})" if r["set"] != "W7" else "")
            + ("  [CHANGED by the capped rule]" if diff else "") + ("  <-- WORSE than before" if is_worse else ""))
        if diff:
            log(f"      choose: {rounds_note(r['choose'])}")
            log(f"      capped: {rounds_note(r['capped'])}")
# ---------------------------------------------------------------- counts
log("counts (cases with restarts):")
summ = {}
for st in ("W5", "R4", "W7"):
    rr = [r for r in rows if r["set"] == st]
    c = {v: {"same": sum(r[f"v_{v}"] == "same" for r in rr),
             "equivalent_or_better": sum(r[f"v_{v}"] in ("same", "equivalent") for r in rr),
             "different": sum(r[f"v_{v}"] == "DIFFERENT" for r in rr)} for v in ("before", "choose", "capped")}
    summ[st] = {"cases": len(rr), **c}
    log(f"  {st} ({len(rr)} cases): same {c['before']['same']} -> {c['choose']['same']} -> {c['capped']['same']}; "
        f"equivalent or better {c['before']['equivalent_or_better']} -> {c['choose']['equivalent_or_better']} -> "
        f"{c['capped']['equivalent_or_better']}; different {c['before']['different']} -> {c['choose']['different']} -> "
        f"{c['capped']['different']}")
log(f"  changed by the capped rule against 'choose': {len(changed)} ({', '.join(r['set'] + ' ' + r['case'] for r in changed)})")
log(f"  worse than before the restarts: {len(worse)} ({', '.join(r['set'] + ' ' + r['case'] for r in worse)})")
(HERE / "r7_acceptance.json").write_text(json.dumps({
    "summary": summ, "same_without": same_without,
    "changed": [{k: r[k] for k in ("set", "case", "v_before", "v_choose", "v_capped")} for r in changed],
    "worse": [{k: r[k] for k in ("set", "case")} for r in worse],
    "cases": [{"set": r["set"], "case": r["case"], "v_before": r["v_before"], "v_choose": r["v_choose"],
               "v_capped": r["v_capped"], "rms": [r[v]["rms_px"] for v in ("before", "choose", "capped")],
               "active": [r[v]["active"] for v in ("before", "choose", "capped")]} for r in rows]},
    indent=1, default=float), encoding="utf-8")
