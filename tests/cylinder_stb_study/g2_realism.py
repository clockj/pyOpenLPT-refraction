"""G2 (plan: 2026-10-08-plan.md, Part G): the synthetic case made realistic one factor at a time, 2 cameras.

    python g2_realism.py [--only NAME ...] [--threads 16]

Runs, with this work's settings as s_runs.py (build p4c, T1's final settings; the line detector with the set's noise
per camera), each in runs/NAME, scored by eval_study:
  Gcal_n8_r2_tol1.0     data/main's Gcal_n8 (images moved by the 0.7 px error field; the tracker keeps the true
                        cameras), 2-D tolerance 1.0 px;
  Gcal_n8_r2_tol1.5     the same at 1.5 px (T1's);
  Gshape_nT1_r2         data/main's Gshape_nT1 (T1's shape model with +-25 % scatter and its flux law, T1's noise),
                        T1's OTF table (otf_t1.txt: what the tracker can know, the model without the scatter), 1.0 px;
  Gall_r2               data/gall's Gall (everything: particles throughout the tube at G-all's density, the error
                        field, T1's shape model with scatter, T1's noise), frames 0-39, T1's OTF table, 1.5 px;
  Gall_r2_control       G3: as Gall_r2 with cam2's images 40 frames later (its frames 40-79), so that no true match
                        exists (as T1-e's control on T1).
Diagnostics of G-shape, added after its run (recall 0.18):
  Gshape_nT1_r2_nocue   Gshape_nT1 with the depth cue off (tracer.shape_check false);
  Gshape0_nT1_r2        data/main's Gshape0_nT1: T1's model without the scatter (synth_study.py gshape0), T1's OTF;
  Gshape12_nT1_r2       data/main's Gshape12_nT1: half the scatter (synth_study.py gshape12), added when T1's depth cue
                        (52 % of true pairs kept, D1) fell between the exact model's (74 %) and G-shape's (33 %);
  Gall12_r2, Gall12_r2_control  data/gall's Gall12 (synth_study.py gall12): G-all at half the scatter, and its control.
D1's detector settings tracked on the most T1-like set (does the gain in true pairs reach the tracks?):
  Gall12mn_r2           Gall12 with the detector told the noise measured on it (S1's method: 9.08, 9.73, with the
                        out-of-focus texture) instead of the rendered noise: the reference for the two below;
  Gall12m15_r2          Gall12_m15 (Gall12 preprocessed with a 15 px median), the detector told the noise measured
                        on it (as S1 does on T1);
  Gall12m15k5_r2        the same with k_detect 5.
  Gall12mn_r2_control, Gall12m15_r2_control, Gall12m15k5_r2_control  their controls (cam2 40 frames later), added for
                        T1's rerun with a 15 px median (r1_t1_rerun.py): how T1-e's estimators respond to the median
                        where the truth is known (g3_control.py --run).
G-all matched to T1 (data/gallm, synth_study.py gallm: its texture, detections and depth cue set to T1's), the
detector told the noise measured on it as on T1:
  Gallm_r2, Gallm_r2_control
G-base and G-tube are s_runs.py's S_n8_r2 and O10_n8_r2.  Writes g2_realism_out.txt and g2_realism.json.
"""
from __future__ import annotations

import argparse
import os

import study
import eval_study
import s_runs
import stbruns

RUNS = {  # case, set, 2-D tolerance, OTF table, cam2's frame shift, tracer overrides
    "Gcal_n8_r2_tol1.0": ("main", "Gcal_n8", 1.0, "otf_r2.txt", 0, {}),
    "Gcal_n8_r2_tol1.5": ("main", "Gcal_n8", 1.5, "otf_r2.txt", 0, {}),
    "Gshape_nT1_r2": ("main", "Gshape_nT1", 1.0, "otf_t1.txt", 0, {}),
    "Gall_r2": ("gall", "Gall", 1.5, "otf_t1.txt", 0, {}),
    "Gall_r2_control": ("gall", "Gall", 1.5, "otf_t1.txt", 40, {}),
    "Gshape_nT1_r2_nocue": ("main", "Gshape_nT1", 1.0, "otf_t1.txt", 0, {"shape_check": False}),
    "Gshape0_nT1_r2": ("main", "Gshape0_nT1", 1.0, "otf_t1.txt", 0, {}),
    "Gshape12_nT1_r2": ("main", "Gshape12_nT1", 1.0, "otf_t1.txt", 0, {}),
    "Gall12_r2": ("gall", "Gall12", 1.5, "otf_t1.txt", 0, {}),
    "Gall12_r2_control": ("gall", "Gall12", 1.5, "otf_t1.txt", 40, {}),
    "Gall12mn_r2": ("gall", "Gall12", 1.5, "otf_t1.txt", 0, {"lines": "measured"}),
    "Gall12m15_r2": ("gall", "Gall12_m15", 1.5, "otf_t1.txt", 0, {"lines": "measured"}),
    "Gall12m15k5_r2": ("gall", "Gall12_m15", 1.5, "otf_t1.txt", 0, {"lines": "measured", "k_detect": 5.0}),
    "Gall12mn_r2_control": ("gall", "Gall12", 1.5, "otf_t1.txt", 40, {"lines": "measured"}),
    "Gall12m15_r2_control": ("gall", "Gall12_m15", 1.5, "otf_t1.txt", 40, {"lines": "measured"}),
    "Gall12m15k5_r2_control": ("gall", "Gall12_m15", 1.5, "otf_t1.txt", 40, {"lines": "measured", "k_detect": 5.0}),
    "Gallm_r2": ("gallm", "Gallm", 1.5, "otf_t1.txt", 0, {"lines": "measured"}),
    "Gallm_r2_control": ("gallm", "Gallm", 1.5, "otf_t1.txt", 40, {"lines": "measured"}),
}


def run_one(name, threads, log, out, out_path):
    case, set_name, tol, otf, shift, over = RUNS[name]
    case_dir = os.path.join(study.DATA, case)
    s = study.load_json(os.path.join(case_dir, "sets.json"))[set_name]
    tracer = s_runs.work_tracer(case_dir, s, 2, otf)
    over = dict(over)
    if over.pop("lines", None) == "measured":          # the noise measured on the preprocessed images
        tracer["lines"] = {"noise": [float(x) for x in s["noise_measured"]]}
    if "k_detect" in over:
        tracer["lines"]["k_detect"] = over.pop("k_detect")
    tracer.update(over)
    lists = [os.path.join(case_dir, "imgLists", set_name, f"cam{k + 1}ImageNames.txt") for k in range(2)]
    if shift:
        with open(lists[1], encoding="utf-8") as fh:
            names = [ln for ln in fh.read().splitlines() if ln.strip()]
        lst = os.path.join(study.RUNS, "configs", f"{name}_cam2_shift{shift}.txt")
        os.makedirs(os.path.dirname(lst), exist_ok=True)
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names[shift:]) + "\n")
        lists[1] = lst
    cfg = s_runs.write_config(os.path.join(study.RUNS, "configs", f"{name}.json"), case_dir, set_name, 2, (0, 39),
                              tracer, s_runs.ipr_for(2, tol), image_lists=lists,
                              comment=f"G2: {name}" + (f", cam2 shifted by {shift} frames" if shift else ""))
    rd = os.path.join(study.RUNS, name)
    info = stbruns.run(cfg, rd, study.BUILD, env={"OMP_NUM_THREADS": str(threads)})
    rec = {"case": case, "set": set_name, "tol_2d_px": tol, "otf": otf, "cam2_shift": shift,
           "tracer_overrides": RUNS[name][5], "tracer": tracer,
           "info": info}
    if not shift:
        res = eval_study.evaluate(rd, case_dir, set_name, [0, 1])
        res.pop("per_frame", None)
        rec["score"] = res
        log(f"  {eval_study.line(name, res)}; exit {info['returncode']}, wall {info['wall_s']:.0f} s")
    else:
        log(f"  {name}: exit {info['returncode']}, wall {info['wall_s']:.0f} s (scored by g3_control.py)")
    out["runs"][name] = rec
    if os.path.exists(out_path):
        prev = study.load_json(out_path)
        prev["runs"].update(out["runs"])
        out["runs"] = prev["runs"]
    study.save_json(out_path, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    names = args.only or list(RUNS)
    log = study.Log("g2_realism" if not args.only else "g2_realism_" + "_".join(args.only))
    out = {"runs": {}}
    log(f"G2: {', '.join(names)}; OMP_NUM_THREADS {args.threads}")
    for name in names:
        run_one(name, args.threads, log, out, os.path.join(study.HERE, "g2_realism.json"))
    log.save()


if __name__ == "__main__":
    main()
