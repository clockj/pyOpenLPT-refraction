"""T1's tagged runs of this plan (L2, G1): S3's final and control runs, as the self-calibration's t1_runs.py makes them,
on build vcf with the cameras' image corrections in config.json.

    python t1_lib_runs.py final   --tag TAG --base CONFIG [--corr C1 C2] [--tol2d 1.0] [--threads 12]
    python t1_lib_runs.py control --tag TAG --base CONFIG [--corr C1 C2] [--tol2d 1.0] [--shift 100]

- base: a T1 config to start from (the 15 px frames, T0's cameras, the run's settings), e.g. the uncorrected 1.0 px
  run's (cases/cyl_20261007/data/runs/s3_final_m15t10/config_final_m15t10.json) or a project's config.json that the
  GUI's VSC updated (its cameras' "correction" kept, unless --corr gives others);
- --corr: image correction files for cam1 and cam2 (config.json's "correction");
- final: frames 0-250; control: frames 0-100 with cam2's images --shift steps later (every track a ghost, T1-e);
- ipr.tol_2d_px is set to --tol2d (the tracking setting; the field VSC leaves it, decision D1 (a)).
The run goes to the case's data/runs/s3_<final|control>_<TAG>/ (a new tag: the case's runs are not overwritten), its
config to runs/configs/ here, and the summary to t1_lib_runs_<final|control>_<TAG>_out.txt and .json here.
"""
from __future__ import annotations

import argparse
import os
import sys

import vcf_common as vc

import stbruns  # noqa: E402  (tests/cylinder_stb)

sys.path.insert(0, os.path.join(vc.CASE, "tests"))
from s3_track import parse_log  # noqa: E402  (the case's tests)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["final", "control"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--corr", nargs=2, default=None)
    ap.add_argument("--tol2d", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--shift", type=int, default=100)
    ap.add_argument("--build", default="vcf")
    a = ap.parse_args()
    if a.threads > vc.THREADS_MAX:
        raise SystemExit(f"--threads {a.threads}: at most {vc.THREADS_MAX}")
    name = f"{a.kind}_{a.tag}"
    rd = os.path.join(vc.CASE_RUNS, f"s3_{name}")
    if os.path.exists(rd):
        raise SystemExit(f"{rd} exists: remove it by hand or choose another tag")
    base_dir = os.path.dirname(os.path.abspath(a.base))
    c = vc.load_json(a.base)
    # paths of the base made absolute, so that the config works from runs/configs/
    for cam in c["cameras"]:
        for key in ("file", "correction"):
            if cam.get(key) and not os.path.isabs(cam[key]):
                cam[key] = os.path.normpath(os.path.join(base_dir, cam[key]))
    c["image_lists"] = [p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
                        for p in c["image_lists"]]
    otf = c["objects"][0]["tracer"].get("otf_file")
    if otf and not os.path.isabs(otf):
        c["objects"][0]["tracer"]["otf_file"] = os.path.normpath(os.path.join(base_dir, otf))
    if a.corr:
        for cam, f in zip(c["cameras"], a.corr):
            cam["correction"] = os.path.abspath(f)
    c["objects"][0]["ipr"]["tol_2d_px"] = a.tol2d
    c["output_dir"] = "Results"
    cfg_dir = os.path.join(vc.RUNS, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    if a.kind == "final":
        c["frames"] = [0, 250]
    else:
        c["frames"] = [0, 100]
        with open(c["image_lists"][1], encoding="utf-8") as fh:
            names = [ln for ln in fh.read().splitlines() if ln.strip()]
        if 100 + a.shift >= len(names):
            raise SystemExit(f"frames 0-100 + shift {a.shift} pass the {len(names)} images")
        lst = os.path.join(cfg_dir, f"cam2ImageNames_shift{a.shift}_{a.tag}.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names[a.shift:]) + "\n")
        c["image_lists"][1] = lst
    c["_comment"] = (f"T1 of run 20261007, correction-field plan (tests/vsc_correction_field), run {name}: from "
                     f"{a.base}" + (f", cam2 {a.shift} steps later (a control: every track a ghost)"
                                    if a.kind == "control" else ""))
    cfg = os.path.join(cfg_dir, f"config_{name}.json")
    vc.save_json(cfg, c)
    log = vc.Log(f"t1_lib_runs_{name}")
    log(f"T1 {a.kind} {a.tag}: build {a.build}, frames {c['frames']}, base {a.base}; cameras "
        f"{[{k: v for k, v in cam.items()} for cam in c['cameras']]}; images {c['image_lists']}; ipr "
        f"{c['objects'][0]['ipr']}; OMP_NUM_THREADS {a.threads}; run dir {rd}")
    info = stbruns.run(cfg, rd, a.build, env={"OMP_NUM_THREADS": str(a.threads), "PYTHONDONTWRITEBYTECODE": "1"})
    res = parse_log(os.path.join(rd, "log.txt"))
    log(f"  exit {info['returncode']}, wall {info['wall_s']:.0f} s; {res['frames_done']} frames, "
        f"{res['seconds_per_frame_mean']:.2f} s per frame; active long tracks at the end {res['active_long_tracks_last']}")
    log.save({"info": info, "config": cfg, "run_dir": rd, **res})


if __name__ == "__main__":
    main()
