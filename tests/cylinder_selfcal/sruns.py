"""The synthetic tracking runs of V2 (plan: section 3), with 2, 3 and 4 cameras, scored against the truth.

    python sruns.py NAME [NAME ...] [--threads 12]
    python sruns.py --list

Every run as the study's (tests/cylinder_stb_study/s_runs.py, g2_realism.py): build p4c, frames 0-39, T1's view volume,
the line detector told the set's noise per camera (the noise measured on the preprocessed images where the set records
it, as on T1), the set's OTF table with its ROIs, the depth cue at 0.8 px / 0.2, IPR 3 loops at the run's 2-D tolerance
and 2 voxels in 3-D; with 3 and 4 cameras 2 more loops with one camera allowed missing.  Configs in
runs/configs/, runs in runs/NAME/ (git-ignored); each scored by the study's eval_study.evaluate (recall of the central
particles seen by all of the run's cameras, by n - 1 and by 2; ghosts; position RMS; long tracks, purity, track
recall).  A run whose cam2 is shifted (a control) is not scored (its tracks are all ghosts).
Runs (case, set, cameras, 2-D tolerance [px], OTF, cam2 shift, use the measured noise):
"""
from __future__ import annotations

import argparse
import os

import selfcal as sc

import eval_study  # noqa: E402  (the study's scorer)
import s_runs  # noqa: E402
import stbruns  # noqa: E402

RUNS = {
    # the 4-camera set with an error field in every camera (synth_sc.py cal4): before the correction
    "Gcal4_n8_r3_tol1.5": ("cal4", "Gcal4_n8", 3, 1.5, "otf_r3.txt", 0, False),
    "Gcal4_n8_r4_tol1.5": ("cal4", "Gcal4_n8", 4, 1.5, "otf_r4.txt", 0, False),
    # after: the images moved by -c of the field fitted from the run's own tracks (v2_fit.py; synth_sc.py corrected)
    "Gcal4c2_n8_r2_tol1.0": ("cal4", "Gcal4c2_n8", 2, 1.0, "otf_r2.txt", 0, False),
    "Gcal4c3_n8_r3_tol1.0": ("cal4", "Gcal4c3_n8", 3, 1.0, "otf_r3.txt", 0, False),
    "Gcal4c4_n8_r4_tol1.0": ("cal4", "Gcal4c4_n8", 4, 1.0, "otf_r4.txt", 0, False),
    "Gcal4c2_n8_r2_tol1.5": ("cal4", "Gcal4c2_n8", 2, 1.5, "otf_r2.txt", 0, False),
    # G-all matched to T1: without the error field (a), the uncorrected at 1.0 px, corrected (re-rendered) and warped
    "Gallm0_r2_tol1.5": ("gallm", "Gallm0", 2, 1.5, "otf_t1.txt", 0, True),
    "Gallm0_r2_tol1.0": ("gallm", "Gallm0", 2, 1.0, "otf_t1.txt", 0, True),
    "Gallm_r2_tol1.0": ("gallm", "Gallm", 2, 1.0, "otf_t1.txt", 0, True),
    "Gallmc_r2_tol1.5": ("gallm", "Gallmc", 2, 1.5, "otf_t1.txt", 0, True),
    "Gallmc_r2_tol1.0": ("gallm", "Gallmc", 2, 1.0, "otf_t1.txt", 0, True),
    "Gallmw_r2_tol1.5": ("gallm", "Gallmw", 2, 1.5, "otf_t1.txt", 0, True),
    "Gallmw_r2_tol1.0": ("gallm", "Gallmw", 2, 1.0, "otf_t1.txt", 0, True),
    # the second pass (VSC's iteration): the field refitted from the first pass's tracks, composed with the first
    # (compose_fields.py), the original images warped (Gallm) or re-rendered (Gcal4, 2 cameras) by the total
    "Gallmw2_r2_tol1.5": ("gallm", "Gallmw2", 2, 1.5, "otf_t1.txt", 0, True),
    "Gallmw2_r2_tol1.0": ("gallm", "Gallmw2", 2, 1.0, "otf_t1.txt", 0, True),
    "Gcal4c2b_n8_r2_tol1.0": ("cal4", "Gcal4c2b_n8", 2, 1.0, "otf_r2.txt", 0, False),
}
__doc__ += "\n".join(f"  {k}: {v}" for k, v in RUNS.items()) + "\nWrites sruns_<NAMES>_out.txt and .json here (one record per run).\n"


def run_one(name, threads, log):
    case, set_name, n, tol, otf, shift, measured = RUNS[name]
    case_dir = os.path.join(sc.DATA, case)
    spec = sc.load_json(os.path.join(case_dir, "sets.json"))[set_name]
    tracer = s_runs.work_tracer(case_dir, spec, n, otf)
    if measured:
        tracer["lines"] = {"noise": [float(x) for x in spec["noise_measured"][:n]]}
    lists = [os.path.join(case_dir, "imgLists", set_name, f"cam{k + 1}ImageNames.txt") for k in range(n)]
    if shift:
        with open(lists[1], encoding="utf-8") as fh:
            names = [ln for ln in fh.read().splitlines() if ln.strip()]
        lst = os.path.join(sc.RUNS, "configs", f"{name}_cam2_shift{shift}.txt")
        os.makedirs(os.path.dirname(lst), exist_ok=True)
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names[shift:]) + "\n")
        lists[1] = lst
    cfg = s_runs.write_config(os.path.join(sc.RUNS, "configs", f"{name}.json"), case_dir, set_name, n, (0, 39), tracer,
                              s_runs.ipr_for(n, tol), image_lists=lists,
                              comment=f"self-calibration plan (tests/cylinder_selfcal): {name}")
    rd = os.path.join(sc.RUNS, name)
    info = stbruns.run(cfg, rd, sc.BUILD, env={"OMP_NUM_THREADS": str(threads), "PYTHONDONTWRITEBYTECODE": "1"})
    rec = {"case": case, "set": set_name, "cameras": n, "tol_2d_px": tol, "otf": otf, "cam2_shift": shift,
           "tracer": tracer, "info": info}
    if not shift:
        res = eval_study.evaluate(rd, case_dir, set_name, list(range(n)))
        res.pop("per_frame", None)
        rec["score"] = res
        log(f"  {eval_study.line(name, res)}; exit {info['returncode']}, wall {info['wall_s']:.0f} s")
    else:
        log(f"  {name}: exit {info['returncode']}, wall {info['wall_s']:.0f} s (a control: not scored)")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(RUNS))
        return
    if a.threads > sc.THREADS_MAX:
        raise SystemExit(f"--threads {a.threads}: at most {sc.THREADS_MAX}")
    for name in a.names:
        if name not in RUNS:
            raise SystemExit(f"unknown run {name}")
    log = sc.Log("sruns_" + "_".join(a.names))
    log(f"synthetic runs {a.names}: build {sc.BUILD}, OMP_NUM_THREADS {a.threads}")
    recs = {}
    for name in a.names:
        recs[name] = run_one(name, a.threads, log)
        log.save({"runs": recs})


if __name__ == "__main__":
    main()
