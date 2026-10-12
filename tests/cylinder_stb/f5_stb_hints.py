"""F5 (plan, Phase 1): STB through the tube with the hints on and off.

    python f5_stb_hints.py [--case f5_spots] [--build p1] [--base head1007]

The synthetic case data/CASE (synth.py: two PINCYL cameras of 6 Oct T2 at tolerance 1e-9, round spots, truth) is
tracked three times, each in a fresh process with the Python module:
  - full: build BUILD with OPENLPT_PROJECTION_HINTS=0, every projection the full search;
  - hints: build BUILD with the hints;
  - base: build BASE (HEAD without this work).
Both runs of BUILD have OPENLPT_CURVED_LINES=0 (Phase 3's curved image lines off), so that on a build with Phase 3
only the hints differ between them, and full is today's code path.
Judged by (plan): base and full identical (the switch restores today's code path exactly); hints and full give the same
tracks, and positions equal to the printed resolution (1e-6 mm) - or, where not, the differences reported and
traced.  Also reported: wall time of each run and the truth's recovery (cstb_eval).
Writes f5_stb_hints_out.txt and f5_stb_hints.json; the runs in runs/f5_* (git-ignored).
"""
import argparse
import os

import cstb
import stbruns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="f5_spots")
    ap.add_argument("--build", default="p1")
    ap.add_argument("--base", default="head1007")
    args = ap.parse_args()
    log = cstb.Log(os.path.join(cstb.HERE, "f5_stb_hints"))
    case_cfg = os.path.join(cstb.DATA, args.case, "config.json")
    runs = os.path.join(cstb.HERE, "runs")
    log(f"F5: STB on data/{args.case} with the projection hints on and off; builds {args.build}, {args.base}")
    out = {"case": args.case, "runs": {}}
    # the curved image lines (Phase 3) are switched off in both, so that only the hints differ (and full is today's
    # code path on a build that has them; a build without them ignores the variable)
    plan = [("full", args.build, {"OPENLPT_PROJECTION_HINTS": "0", "OPENLPT_CURVED_LINES": "0"}),
            ("hints", args.build, {"OPENLPT_CURVED_LINES": "0"}),
            ("base", args.base, None)]
    for tag, build, env in plan:
        rd = os.path.join(runs, f"f5_{tag}")
        info = stbruns.run(case_cfg, rd, build, env=env)
        out["runs"][tag] = info
        log(f"  {tag}: build {build}{' with ' + str(env) if env else ''}: exit {info['returncode']}, "
            f"wall {info['wall_s']:.1f} s")
    c1 = stbruns.compare(os.path.join(runs, "f5_base"), os.path.join(runs, "f5_full"))
    c2 = stbruns.compare(os.path.join(runs, "f5_full"), os.path.join(runs, "f5_hints"))
    out["base_vs_full"] = c1
    out["full_vs_hints"] = c2
    log(f"\n  base vs full: {c1['identical_files']} of {c1['compared_files']} files identical; "
        f"only in one: {c1['only_a'] + c1['only_b']}")
    log(f"  full vs hints: {c2['identical_files']} of {c2['compared_files']} files identical; "
        f"only in one: {c2['only_a'] + c2['only_b']}")
    for f, rec in c2["files"].items():
        if rec != "identical":
            log(f"    {f}: rows {rec.get('rows')}, largest differences "
                + ", ".join(f"{k} {v:.3g}" for k, v in (rec.get('max_abs_diff') or {}).items() if v > 0))
    log.save(out)


if __name__ == "__main__":
    main()
