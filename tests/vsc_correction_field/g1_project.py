"""G1's test projects: T1 as the GUI sees a project (config.json + its Results/), in projects/ here.

    python g1_project.py NAME [--results RUN]      # RUN: a run folder whose Results/ to copy (default s3_final_m15)

projects/NAME/config.json is the 15 px run's config (cases/cyl_20261007/data/runs/s3_final_m15/config_final_m15.json:
the case's frames, T0's cameras and the OTF table, all by absolute path, read only), and projects/NAME/Results/ a copy
of RUN's results (the tracks VSC reads). Writes g1_project_<NAME>_out.txt here.
"""
from __future__ import annotations

import argparse
import os
import shutil

import vcf_common as vc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--results", default=os.path.join(vc.CASE_RUNS, "s3_final_m15"))
    a = ap.parse_args()
    proj = os.path.join(vc.HERE, "projects", a.name)
    if os.path.exists(proj):
        raise SystemExit(f"{proj} exists: remove it by hand")
    os.makedirs(proj)
    cfg = vc.load_json(os.path.join(vc.CASE_RUNS, "s3_final_m15", "config_final_m15.json"))
    cfg["output_dir"] = "Results"
    cfg["_comment"] = (f"G1 (tests/vsc_correction_field): T1's 15 px config as a GUI project; Results/ copied from "
                       f"{a.results}")
    vc.save_json(os.path.join(proj, "config.json"), cfg)
    shutil.copytree(os.path.join(a.results, "Results"), os.path.join(proj, "Results"))
    log = vc.Log(f"g1_project_{a.name}")
    n = sum(len(fs) for _d, _s, fs in os.walk(os.path.join(proj, "Results")))
    log(f"{proj}: config.json (cameras {[c['file'] for c in cfg['cameras']]}, tol_2d_px "
        f"{cfg['objects'][0]['ipr']['tol_2d_px']}), Results/ from {a.results} ({n} files)")
    log.save({"project": proj, "results_from": a.results})


if __name__ == "__main__":
    main()
