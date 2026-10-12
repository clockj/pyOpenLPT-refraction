"""Parts S and O (plan: 2026-10-08-plan.md, sections 3 and 7): STB on the study's synthetic sets with 2, 3 and 4
cameras, scored against the truth.

    python s_runs.py [--only NAME ...] [--stream work|today|all] [--threads 16] [--list] [--tag T]

Several streams can run at once: each run is claimed first (runs/claims/NAME, created exclusively), and a stream skips
a run another stream has claimed or that s_runs.json already scores.

Runs (data/main's sets, synth_study.py), each in a fresh process with the Python module (stbruns.run), in runs/NAME:
  - this work's settings (build p4c), as T1's final run: the line detector with the set's noise per camera, the true
    OTF table of the rig's cameras (otf_r<n>.txt, model "segment") and its ROIs (roi_sigma 3), the depth cue
    (shape_tol_px 0.8, shape_tol_rel 0.2); voxel 0.03 mm, IPR 3 loops, 2-D tolerance 1.0 px (exact cameras), 3-D
    tolerance 2 voxels; with 3 and 4 cameras 2 more IPR loops with one camera allowed missing (reduced_cameras 1):
        S_n4, S_n8, S_n16 (Part S) and O05_n8, O10_n8 (Part O), on rigs 2, 3 and 4: NAME = <set>_r<n>
  - today's code (build head1007) with today's settings on S_n8: the peak finder at 4 sigma (intensity threshold 32),
    radius 2 px, the OTF estimated from the images, the same IPR, voxel and volume: NAME = today_S_n8_r<n>
The view volume of every run: T1's (study.view_volume()).  Frames 0-39.  OMP_NUM_THREADS = THREADS (two streams can
run at once on the 32 cores).  Scored by eval_study.evaluate; writes s_runs_out.txt and s_runs.json (merged with the
runs already scored).
"""
from __future__ import annotations

import argparse
import json
import os

import study
import eval_study
import stbruns

CASE = os.path.join(study.DATA, "main")
# the order the streams take them in: Part G's references (S_n8_r2, O10_n8_r2) and noise 8 first
WORK = ["S_n4_r2", "S_n4_r3", "S_n8_r2", "O10_n8_r2", "S_n8_r3", "S_n8_r4", "O05_n8_r2", "O10_n8_r3", "O10_n8_r4",
        "O05_n8_r3", "O05_n8_r4", "S_n4_r4", "S_n16_r2", "S_n16_r3", "S_n16_r4"]
TODAY = [f"today_S_n8_r{n}" for n in (2, 3, 4)]


def claim(name):
    """True if this stream may run NAME: it is not scored yet and no other stream has claimed it."""
    out_path = os.path.join(study.HERE, "s_runs.json")
    if os.path.exists(out_path) and name in study.load_json(out_path).get("runs", {}):
        return False
    d = os.path.join(study.RUNS, "claims")
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def ipr_for(n, tol2d):
    if n == 2:
        return {"loops": 3, "reduced_cameras": 0, "reduced_loops": 0, "tol_2d_px": tol2d, "tol_3d_vox": 2.0}
    return {"loops": 3, "reduced_cameras": 1, "reduced_loops": 2, "tol_2d_px": tol2d, "tol_3d_vox": 2.0}


def work_tracer(case_dir, set_spec, n, otf):
    return {"intensity_threshold": 30, "radius_px": 2.0, "detector": "lines",
            "lines": {"noise": [float(x) for x in set_spec["noise"][:n]]},
            "otf_file": os.path.join(case_dir, otf), "roi_from_otf": True, "roi_sigma": 3.0,
            "shape_check": True, "shape_tol_px": 0.8, "shape_tol_rel": 0.2}


def write_config(path, case_dir, set_name, n, frames, tracer, ipr, voxel=0.03, image_lists=None, comment=""):
    vol = study.view_volume()
    cfg = {
        "_comment": f"cylinder STB study (tests/cylinder_stb_study): set {set_name}, {n} cameras. {comment}".strip(),
        "frames": [int(frames[0]), int(frames[1])], "fps": 1000, "threads": 0,
        "cameras": [{"file": os.path.join(case_dir, "camFile", f"cam{k + 1}.txt"), "max_intensity": 255}
                    for k in range(n)],
        "image_lists": image_lists or [os.path.join(case_dir, "imgLists", set_name, f"cam{k + 1}ImageNames.txt")
                                       for k in range(n)],
        "view_volume_mm": {"x": vol["x"], "y": vol["y"], "z": vol["z"]},
        "voxel_mm": voxel, "output_dir": "Results",
        "objects": [{
            "type": "Tracer",
            "initial_phase": {"search_radius_vox": 10.0, "frames": 4},
            "convergence_phase": {"avg_spacing_vox": 30.0},
            "shake": {"width_vox": 0.25, "loops": 4, "ghost_threshold": 0.01},
            "predictive_field": {"grid": [51, 51, 51], "search_radius_vox": 25.0},
            "ipr": ipr,
            "tracer": tracer,
        }],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    return path


def spec(name):
    """(set, n cameras, build, tracer kind) of a run name."""
    today = name.startswith("today_")
    base = name[len("today_"):] if today else name
    set_name, rn = base.rsplit("_r", 1)
    return set_name, int(rn), (study.TODAY if today else study.BUILD), today


def run_one(name, threads, log, out, out_path):
    set_name, n, build, today = spec(name)
    sets = study.load_json(os.path.join(CASE, "sets.json"))
    s = sets[set_name]
    if today:
        tracer = {"intensity_threshold": 32, "radius_px": 2.0}
    else:
        tracer = work_tracer(CASE, s, n, f"otf_r{n}.txt")
    cfg = write_config(os.path.join(study.RUNS, "configs", f"{name}.json"), CASE, set_name, n, (0, 39), tracer,
                       ipr_for(n, 1.0))
    rd = os.path.join(study.RUNS, name)
    info = stbruns.run(cfg, rd, build, env={"OMP_NUM_THREADS": str(threads)})
    res = eval_study.evaluate(rd, CASE, set_name, list(range(n)))
    res.pop("per_frame", None)
    out["runs"][name] = {"set": set_name, "cameras": n, "build": build, "threads": threads, "info": info,
                         "score": res}
    log(f"  {eval_study.line(name, res)}; exit {info['returncode']}, wall {info['wall_s']:.0f} s")
    # merge with what other streams wrote meanwhile
    if os.path.exists(out_path):
        prev = study.load_json(out_path)
        prev["runs"].update(out["runs"])
        out["runs"] = prev["runs"]
    study.save_json(out_path, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--stream", choices=["work", "today", "all"], default="all")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tag", default="", help="a suffix for this stream's log (several streams at once)")
    args = ap.parse_args()
    names = args.only or {"work": WORK, "today": TODAY, "all": WORK + TODAY}[args.stream]
    if args.list:
        print("\n".join(names))
        return
    stem = "s_runs" if not args.only else "s_runs_" + "_".join(args.only)
    stem = stem if args.stream == "all" or args.only else f"s_runs_{args.stream}"
    if args.tag:
        stem += f"_{args.tag}"
    log = study.Log(stem)
    out_path = os.path.join(study.HERE, "s_runs.json")
    out = {"runs": {}}
    log(f"Parts S and O: {len(names)} runs, OMP_NUM_THREADS {args.threads}: {', '.join(names)}")
    for name in names:
        if not claim(name):
            log(f"  {name}: scored or claimed by another stream, skipped")
            continue
        run_one(name, args.threads, log, out, out_path)
    log.save()


if __name__ == "__main__":
    main()
