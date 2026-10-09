"""T1's tagged runs of this plan (V3, M0, M2): S3's final and control runs (cases/cyl_20261007/tests/s3_track.py, the
same settings) on given frames, tolerances and OTF table.

    python t1_runs.py final   --tag TAG --pre DIR [--tol2d 1.5] [--shape-tol 0.8 0.2] [--otf FILE] [--threads 12]
    python t1_runs.py control --tag TAG --pre DIR [...] [--shift 100]

  - final: frames 0-250 of DIR (S1's every 4th raw frame), T0's tracking cameras, T1's view volume; the line detector
    told the noise DIR's manifest records; the OTF table (default: the case's data/T1_otf_shape.txt, S2's model) with
    its ROIs; the depth cue at --shape-tol (default S2's 0.8 px / 0.2); IPR 3 loops at --tol2d, 3-D tolerance 2 voxels;
  - control: the same on frames 0-100 with cam2's images --shift steps later, so that every track is a ghost (T1-e).
The run goes to the case's data/runs/s3_<final|control>_<TAG>/ (git-ignored; the tag keeps the case's own runs
untouched), its config to runs/configs/ here, and the summary to t1_runs_<final|control>_<TAG>_out.txt and .json here.
Build p4c, OMP_NUM_THREADS = --threads (at most 12).
"""
from __future__ import annotations

import argparse
import os
import sys

import selfcal as sc

import stbruns  # noqa: E402  (tests/cylinder_stb)
import t1_setup as ts  # noqa: E402  (the case)
from s3_track import parse_log  # noqa: E402  (the case's tests)


def noise_of(pre):
    man = sc.load_json(os.path.join(pre, "manifest.json"))
    return [float(man["cameras"][f"cam{c}"]["noise_8bit_median"]) for c in (1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["final", "control"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pre", required=True, help="the frames' folder (cam1/, cam2/, cam*ImageNames.txt, manifest.json)")
    ap.add_argument("--tol2d", type=float, default=1.5)
    ap.add_argument("--shape-tol", type=float, nargs=2, default=None)
    ap.add_argument("--otf", default=os.path.join(sc.CASE_DATA, "T1_otf_shape.txt"))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--shift", type=int, default=100)
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    args = ap.parse_args()
    if args.threads > sc.THREADS_MAX:
        raise SystemExit(f"--threads {args.threads}: at most {sc.THREADS_MAX}")
    if args.tag in ("", "m15", "m15k5"):
        raise SystemExit("a new tag, please: the case's runs and the study's are not overwritten")
    pre = os.path.abspath(args.pre)
    name = f"{args.kind}_{args.tag}"
    rd = os.path.join(sc.CASE_RUNS, f"s3_{name}")
    if os.path.exists(rd):
        raise SystemExit(f"{rd} exists: remove it by hand or choose another tag")
    log = sc.Log(f"t1_runs_{name}")
    vol = sc.view_volume()
    shape = sc.load_json(os.path.join(sc.CASE_DATA, "T1_shape.json"))["shape_check"]
    tol = args.shape_tol or [shape["tol_px"], shape["tol_rel"]]
    tracer = {"intensity_threshold": 28, "radius_px": 2.0, "detector": "lines", "lines": {"noise": noise_of(pre)},
              "otf_file": os.path.abspath(args.otf), "roi_from_otf": True, "roi_sigma": 3.0,
              "shape_check": True, "shape_tol_px": float(tol[0]), "shape_tol_rel": float(tol[1])}
    ipr = {"loops": 3, "reduced_cameras": 0, "reduced_loops": 0, "tol_2d_px": args.tol2d, "tol_3d_vox": 2.0}
    lists = [os.path.join(pre, f"cam{c}ImageNames.txt") for c in (1, 2)]
    frames = args.frames or ((0, 250) if args.kind == "final" else (0, 100))
    cfg_dir = os.path.join(sc.RUNS, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    if args.kind == "control":
        with open(lists[1], encoding="utf-8") as fh:
            names = [ln for ln in fh.read().splitlines() if ln.strip()]
        if frames[1] + args.shift >= len(names):
            raise SystemExit(f"frames {frames} + shift {args.shift} pass the {len(names)} images")
        lst = os.path.join(cfg_dir, f"cam2ImageNames_shift{args.shift}_{args.tag}.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names[args.shift:]) + "\n")
        lists[1] = lst
    cfg = os.path.join(cfg_dir, f"config_{name}.json")
    ts.stb_config(cfg, frames, vol, tracer, ipr=ipr, extra={"image_lists": lists})
    c = sc.load_json(cfg)
    c["_comment"] = (f"T1 of run 20261007, self-calibration plan (tests/cylinder_selfcal), run {name}: frames of {pre}"
                     + (f", cam2 {args.shift} steps later (a control: every track a ghost)" if args.kind == "control"
                        else ""))
    sc.save_json(cfg, c)
    log(f"T1 {args.kind} {args.tag}: build {sc.BUILD}, frames {frames[0]}-{frames[1]}, images {pre}; tracer {tracer}; "
        f"ipr {ipr}; OMP_NUM_THREADS {args.threads}; run dir {rd}")
    info = stbruns.run(cfg, rd, sc.BUILD, env={"OMP_NUM_THREADS": str(args.threads), "PYTHONDONTWRITEBYTECODE": "1"})
    res = parse_log(os.path.join(rd, "log.txt"))
    log(f"  exit {info['returncode']}, wall {info['wall_s']:.0f} s; {res['frames_done']} frames, "
        f"{res['seconds_per_frame_mean']:.2f} s per frame; active long tracks at the end {res['active_long_tracks_last']}")
    log.save({"info": info, "tracer": tracer, "ipr": ipr, "frames": frames, "pre": pre, "run_dir": rd,
              "shift": args.shift if args.kind == "control" else None, **res})


if __name__ == "__main__":
    main()
