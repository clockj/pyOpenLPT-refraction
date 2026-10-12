"""L3: the library's correction against re-rendered images, on synthetic sets with 2, 3 and 4 cameras (truth known).

    python l3_synthetic.py run N        # N = 2, 3, 4: the library run (runs/l3_lib_r<N>)
    python l3_synthetic.py score        # every run scored against the truth

The self-calibration's 4-camera set Gcal4_n8 (tests/cylinder_selfcal/data/cal4, read only) has an error field in every
camera: its images sit off where the model projects. V2 fitted a one-pass field from each run's tracks (cal4_r2,
cal4_r3, cal4_r4: 2, 3, 4 cameras) and re-rendered the images moved by it (sets Gcal4c<N>_n8), which were tracked at
1.0 px. Here the original images are tracked at 1.0 px with the same fields as the cameras' corrections in config.json
(convert_fields.py's files): the same geometry, the model moved instead of the images. Scored by the study's
eval_study.evaluate against the truth: recall of the central particles seen by all cameras, ghosts, position RMS,
long tracks and their purity. References (read only): the re-rendered runs Gcal4c<N>_n8_r<N>_tol1.0 and the
uncorrected runs (Gcal_n8_r2_tol1.0 of the study; Gcal4_n8_r3/r4_tol1.5 of V2). Pass: recall, ghost share and
position RMS within 3 % (relative; 3 points of the ghost share) of the re-rendered set's.
Writes l3_synthetic_<part>_out.txt and .json here.
"""
from __future__ import annotations

import os
import sys

import vcf_common as vc

import stbruns  # noqa: E402  (tests/cylinder_stb)

STUDY = os.path.join(vc.REPO, "tests", "cylinder_stb_study")
sys.path.insert(0, STUDY)
CAL4 = os.path.join(vc.SELFCAL, "data", "cal4")


def run(n: int):
    base = vc.load_json(os.path.join(vc.SELFCAL, "runs", "configs", f"Gcal4c{n}_n8_r{n}_tol1.0.json"))
    base["image_lists"] = [os.path.join(CAL4, "imgLists", "Gcal4_n8", f"cam{k + 1}ImageNames.txt") for k in range(n)]
    for k, cam in enumerate(base["cameras"]):
        cam["correction"] = os.path.join(vc.DATA, "fields", f"cal4_r{n}_cam{k + 1}.txt")
    base["_comment"] = (f"L3 (tests/vsc_correction_field): set Gcal4_n8, {n} cameras, at 1.0 px, the cameras corrected "
                        f"by V2's one-pass field cal4_r{n}")
    cfg = os.path.join(vc.RUNS, "configs", f"l3_lib_r{n}.json")
    vc.save_json(cfg, base)
    rd = os.path.join(vc.RUNS, f"l3_lib_r{n}")
    info = stbruns.run(cfg, rd, "vcf", env={"OMP_NUM_THREADS": str(vc.THREADS_MAX), "PYTHONDONTWRITEBYTECODE": "1"})
    log = vc.Log(f"l3_synthetic_run_r{n}")
    log(f"l3_lib_r{n}: {cfg} -> {rd}: exit {info['returncode']}, {info['wall_s']:.0f} s")
    log.save(info)


def score():
    import eval_study
    log = vc.Log("l3_synthetic_score")
    runs = {}
    for n in (2, 3, 4):
        runs[f"lib_r{n}"] = (os.path.join(vc.RUNS, f"l3_lib_r{n}"), "Gcal4_n8", n)
        runs[f"rerendered_r{n}"] = (os.path.join(vc.SELFCAL, "runs", f"Gcal4c{n}_n8_r{n}_tol1.0"), f"Gcal4c{n}_n8", n)
    runs["uncorrected_r2_tol1.0"] = (os.path.join(STUDY, "runs", "Gcal_n8_r2_tol1.0"), "Gcal4_n8", 2)
    runs["uncorrected_r3_tol1.5"] = (os.path.join(vc.SELFCAL, "runs", "Gcal4_n8_r3_tol1.5"), "Gcal4_n8", 3)
    runs["uncorrected_r4_tol1.5"] = (os.path.join(vc.SELFCAL, "runs", "Gcal4_n8_r4_tol1.5"), "Gcal4_n8", 4)
    res = {}
    for name, (rd, set_name, n) in runs.items():
        if not os.path.isdir(os.path.join(rd, "Results")):
            log(f"{name}: no run at {rd}")
            continue
        case_dir = CAL4
        if name == "uncorrected_r2_tol1.0":
            # the study's 2-camera set Gcal_n8 is Gcal4_n8's first two cameras; scored with its own truth
            case_dir = os.path.join(STUDY, "data", "main")
            set_name = "Gcal_n8"
        r = eval_study.evaluate(rd, case_dir, set_name, list(range(n)))
        r.pop("per_frame", None)
        res[name] = r
        log(f"  {eval_study.line(name, r)}")
    out = {"scores": res, "checks": {}}
    ok = True
    for n in (2, 3, 4):
        a, b = res.get(f"lib_r{n}"), res.get(f"rerendered_r{n}")
        if a is None or b is None:
            ok = False
            continue
        ra, rb = a["recall"]["all_in"], b["recall"]["all_in"]
        ga, gb = a["ghost_share"], b["ghost_share"]
        pa = max(a["pos_rms_um"]) if a["pos_rms_um"] else float("nan")
        pb = max(b["pos_rms_um"]) if b["pos_rms_um"] else float("nan")
        c = {"recall": [ra, rb, abs(ra - rb) <= 0.03 * rb], "ghost_share": [ga, gb, abs(ga - gb) <= 0.03],
             "pos_rms_um_max": [pa, pb, abs(pa - pb) <= 0.03 * pb]}
        good = all(v[2] for v in c.values())
        ok &= good
        out["checks"][f"r{n}"] = {**c, "pass": bool(good)}
        log(f"{n} cameras: library {ra:.3f} recall, {100 * ga:.1f} % ghosts, position RMS (largest axis) {pa:.1f} um; "
            f"re-rendered {rb:.3f}, {100 * gb:.1f} %, {pb:.1f} um -> {'PASS' if good else 'FAIL'}")
    out["pass"] = bool(ok)
    log(f"L3: {'PASS' if ok else 'FAIL'}")
    log.save(out)


def main():
    if sys.argv[1] == "run":
        run(int(sys.argv[2]))
    else:
        score()


if __name__ == "__main__":
    main()
