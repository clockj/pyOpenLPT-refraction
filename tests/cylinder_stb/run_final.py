"""Reruns the cylinder STB tests on one build (the final one) and collects their verdicts.

    python run_final.py [--build p4c] [--only F1 F2 ...]

Each test is run in its own process with --build BUILD (its outputs are rewritten beside it; the earlier ones, on
the builds of their phase, are named in the report), then its judged line is read from its _out.txt.  Writes
run_final_out.txt and run_final.json (with --only: run_final_<ids>_out.txt and .json, so that the full record stays).
"""
import argparse
import json
import os
import subprocess
import time

import cstb

PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"
TESTS = [
    ("F1", ["f1_agreement.py"], "f1_agreement_out.txt"),
    ("F1b", ["f1b_residuals.py"], "f1b_residuals_out.txt"),
    ("F2", ["f2_los_hints.py"], "f2_los_hints_out.txt"),
    ("F3", ["f3_junction.py"], "f3_junction_out.txt"),
    ("F3b", ["f3b_junction_diag.py"], "f3b_junction_diag_out.txt"),
    ("F4", ["f4_speed.py"], "f4_speed_out.txt"),
    ("F5", ["f5_stb_hints.py"], "f5_stb_hints_out.txt"),
    ("E1", ["e1_coverage.py"], "e1_coverage_out.txt"),
    ("E3", ["e3_large_box.py", "--builds", "head1007", "{build}"], "e3_large_box_out.txt"),
    ("V1", ["v1_statuses.py"], "v1_statuses_out.txt"),
    ("V2", ["v2_shadow_edge.py", "--builds", "p3", "{build}"], "v2_shadow_edge_out.txt"),
    ("V3", ["v3_line_roi.py"], "v3_line_roi_out.txt"),
    ("L5", ["l5_port.py", "--t1-frames", "0", "500"], "l5_port_out.txt"),
    ("L5b", ["l5b_port_isolated.py"], "l5b_port_isolated_out.txt"),
    ("L6", ["sympy_checks.py"], "sympy_checks_out.txt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p4c")
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    name_out = "run_final" + ("_" + "_".join(args.only) if args.only else "")   # a partial run keeps the full record
    log = cstb.Log(os.path.join(cstb.HERE, name_out))
    log(f"The cylinder STB tests on build {args.build}")
    out = {"build": args.build, "tests": {}}
    for name, cmd, outfile in TESTS:
        if args.only and name not in args.only:
            continue
        cmd = [c.replace("{build}", args.build) for c in cmd]
        if name not in ("E3", "V2", "L6"):
            cmd += ["--build", args.build]
        t0 = time.time()
        r = subprocess.run([PY, "-u"] + cmd, cwd=cstb.HERE, capture_output=True, text=True)
        secs = time.time() - t0
        verdict = []
        p = os.path.join(cstb.HERE, outfile)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                for ln in fh:
                    if ("judged" in ln or "PASS" in ln or "FAIL" in ln or "checks:" in ln) and len(verdict) < 8:
                        verdict.append(ln.rstrip())
        out["tests"][name] = {"cmd": cmd, "returncode": r.returncode, "seconds": round(secs, 1), "verdict": verdict,
                              "stderr_tail": r.stderr[-1500:] if r.returncode else ""}
        log(f"  {name}: exit {r.returncode}, {secs:.0f} s")
        for v in verdict:
            log(f"      {v.strip()}")
        with open(os.path.join(cstb.HERE, f"{name_out}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    log.save(out)


if __name__ == "__main__":
    main()
