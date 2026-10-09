"""S3 (plan: 2026-10-08-plan.md, Phase T): STB on T1 with three cameras.

    python tests/s3_track.py final [--frames 0 249] [--tol2d 1.5] [--no-cue | --shape-tol PX REL] [--reduced 1|0]
                                   [--tag TAG]
    python tests/s3_track.py control [--frames 0 149] [--shift 50 100] (the same options as final)
    python tests/s3_track.py pair12 [--frames 0 249] [--tol2d 1.5] [--tag TAG]
    python tests/s3_track.py control12 [--frames 0 149] [--shift 100 0] (cams 1-2, cam2 SHIFT[0] frames later)
    python tests/s3_track.py copy-to-x [--tag TAG]

S1's preprocessed frames (data/T1pre: raw frames 0-249, every frame, the user's first 250), T0's tracking cameras
(camFile_track), the view volume at least two cameras see (t1_setup.view_volume), voxel 0.03 mm (about 1.4 px on
cams 1-2, 2 px on cam3), the frame rate given to STB as 4,000 frames/s.
  - final: build p4c (7 Oct's work): the line detector (tracer.detector "lines", each camera's noise from S1), S2's
    shape model as the OTF (data/T1_otf_shape.txt, model "segment") with regions from it (roi_from_otf), the depth cue
    (shape_check, S2's tolerances; --no-cue: off); 3 IPR loops with every camera, then 2 loops that accept a match
    in two of the three (reduced_cameras 1: the plan's D3), 2-D tolerance --tol2d, 3-D tolerance 2 voxels;
  - control (T1-e): final's settings with cam2's images SHIFT[0] frames later and cam3's SHIFT[1] frames later, so
    that no camera pair holds a true match and every track is a ghost; its active long tracks against final's at the
    same frame estimate the ghosts in final (frames 0-149 use raw frames up to 249);
  - pair12: final's settings on cams 1 and 2 only (the OTF's first two cameras, the same view volume, IPR with two
    cameras): what the third camera adds;
  - control12: pair12's settings with cam2's images SHIFT[0] frames later (default 100): pair12's ghosts.
Runs in data/runs/s3_<name>[_TAG]/ (git-ignored), each with its config, log.txt and Results/; writes
s3_track_<name>[_TAG]_out.txt and .json beside itself (frames, time per frame, objects per frame from the log).
copy-to-x copies the final run to <tracers.out_dir> (config.json that reruns there, config_local.json, log.txt,
Results/, the frames it read as images\\, the cameras it used as camFile\\ (cam3 cropped, S1b), the OTF table and
the model) and the checks' outputs to its checks\\.
It refuses to overwrite a different run that is there.
"""
import argparse
import json
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402

sys.path.insert(0, os.path.join(ts.REPO, "tests", "cylinder_stb"))
import stbruns  # noqa: E402

FPS = 4000


def shape_tolerances():
    with open(os.path.join(ts.DATA, "T1_shape.json"), encoding="utf-8") as fh:
        sc = json.load(fh)["shape_check"]
    return {"shape_tol_px": sc["tol_px"], "shape_tol_rel": sc["tol_rel"]}


def parse_log(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    times = [float(x) for x in re.findall(r"Total time for frame \d+: ([0-9.]+)", text)]
    dets = [tuple(int(v) for v in m.group(1).split())
            for m in re.finditer(r"2D detections per active camera: ([0-9 \t]+)", text)]
    longs = [int(m.group(1)) for m in re.finditer(r"No\. of active long tracks: \d+ \+ \d+ - \d+ = (\d+)", text)]
    return {"frames_done": len(times), "seconds_per_frame_mean": sum(times) / max(1, len(times)),
            "active_long_tracks_last": longs[-1] if longs else None, "active_long_tracks": longs,
            "detections_first_ipr_loop": dets[:3]}


def otf_subset(src, dst, keep):
    """The OTF table of src with only the cameras `keep` (indices)."""
    with open(src, encoding="utf-8") as fh:
        L = [ln.rstrip("\n") for ln in fh]
    n_cam, nx, ny, nz, ng = (int(v) for v in L[1].split(","))
    out = [L[0], ",".join(str(v) for v in (len(keep), nx, ny, nz, ng)), L[2], L[3]]
    i = 4
    while i < len(L):
        ln = L[i]
        if ln.startswith("# ") and "(n_cam,n_grid)" in ln:
            rows = L[i + 1:i + 1 + n_cam]
            out.append(ln)
            out += [rows[k] for k in keep]
            i += 1 + n_cam
        else:
            out.append(ln)
            i += 1
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def shifted_list(cam, shift, n_needed, tag):
    with open(os.path.join(ts.PRE, f"cam{cam}ImageNames.txt"), encoding="utf-8") as fh:
        names = [ln for ln in fh.read().splitlines() if ln.strip()]
    if n_needed + shift > len(names):
        raise SystemExit(f"cam{cam}: {n_needed} frames + shift {shift} pass the {len(names)} images")
    lst = os.path.join(ts.DATA, "s3", f"cam{cam}ImageNames_shift{shift}{tag}.txt")
    os.makedirs(os.path.dirname(lst), exist_ok=True)
    with open(lst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(names[shift:]) + "\n")
    return lst


def copy_to_x(tag):
    cfg = case.load_config()
    src = os.path.join(ts.DATA, "runs", f"s3_final{tag}")
    dst = cfg["tracers"]["out_dir"]
    cfg_name = f"config_final{tag}.json"
    with open(os.path.join(src, cfg_name), encoding="utf-8") as fh:
        run_cfg = json.load(fh)
    info_x = os.path.join(dst, "run_info.json")
    if os.path.exists(info_x):
        with open(info_x, encoding="utf-8") as fh:
            there = os.path.basename(json.load(fh)["config"])
        if there != cfg_name:
            raise SystemExit(f"X: holds another run ({there}); move it aside first")
    os.makedirs(dst, exist_ok=True)
    # the frames the run read, with image lists that point to the copies
    imgs = os.path.join(dst, "images")
    for k, lst in enumerate(run_cfg["image_lists"]):
        with open(lst, encoding="utf-8") as fh:
            paths = [ln for ln in fh.read().splitlines() if ln.strip()]
        os.makedirs(os.path.join(imgs, f"cam{k + 1}"), exist_ok=True)
        out = []
        for q in paths:
            out.append(os.path.join(imgs, f"cam{k + 1}", os.path.basename(q)))
            if not os.path.exists(out[-1]) or os.path.getsize(out[-1]) != os.path.getsize(q):
                shutil.copy2(q, out[-1])
        with open(os.path.join(imgs, f"cam{k + 1}ImageNames.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
    shutil.copy2(os.path.join(ts.PRE, "manifest.json"), os.path.join(imgs, "manifest.json"))
    # the cameras the run used (cam3 may be cropped: S1b), copied beside it as camFile\\
    os.makedirs(os.path.join(dst, "camFile"), exist_ok=True)
    for c in run_cfg["cameras"]:
        shutil.copy2(c["file"], os.path.join(dst, "camFile", os.path.basename(c["file"])))
    run_cfg["_comment"] += " Copied to X: with the paths of the copies there."
    run_cfg["cameras"] = [{**c, "file": os.path.join(dst, "camFile", os.path.basename(c["file"]))} for c in run_cfg["cameras"]]
    run_cfg["image_lists"] = [os.path.join(imgs, f"cam{k + 1}ImageNames.txt") for k in range(len(run_cfg["cameras"]))]
    run_cfg["objects"][0]["tracer"]["otf_file"] = "otf_shape.txt"
    run_cfg["output_dir"] = "Results"
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_cfg, fh, indent=1)
    shutil.copy2(os.path.join(src, cfg_name), os.path.join(dst, "config_local.json"))
    for name in ("log.txt", "run_info.json"):
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    if os.path.isdir(os.path.join(dst, "Results")):
        shutil.rmtree(os.path.join(dst, "Results"))
    shutil.copytree(os.path.join(src, "Results"), os.path.join(dst, "Results"))
    shutil.copy2(os.path.join(ts.DATA, "T1_otf_shape.txt"), os.path.join(dst, "otf_shape.txt"))
    shutil.copy2(os.path.join(ts.DATA, "T1_shape.json"), os.path.join(dst, "shape_model.json"))
    chk = os.path.join(dst, "checks")
    os.makedirs(chk, exist_ok=True)
    run = f"s3_final{tag}"
    stems = ("p0b_t1_survey", "p0c_t1_motion", "s1_preprocess", "s1b_crop_cam3", "s0_detector", "s2_shape_fit",
             f"t1_checks_{run}", f"t1e_ghost_control_{run}", f"t1e_ghost_control_s3_pair12{tag}", f"s3_track_final{tag}",
             f"s3_track_control{tag}", f"s3_track_pair12{tag}", f"s3_track_control12{tag}", "t1_compare")
    files = [os.path.join(HERE, f"{s}{e}") for s in stems for e in ("_out.txt", ".json", ".png")]
    files += [os.path.join(HERE, n) for n in (f"t1_tracks_{run}.png", f"t1_flow_{run}.png", "p0c_k8.png",
                                              "p0b_t1_crops.png")]
    copied = [f for f in files if os.path.exists(f)]
    for f in copied:
        shutil.copy2(f, os.path.join(chk, os.path.basename(f)))
    print(f"copied {src} -> {dst}; {len(copied)} check files -> {chk}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["final", "control", "pair12", "control12", "copy-to-x"])
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    ap.add_argument("--build", default=None)
    ap.add_argument("--tol2d", type=float, default=1.5)
    ap.add_argument("--shift", type=int, nargs=2, default=[50, 100], help="control: cam2's and cam3's shifts [frames]")
    ap.add_argument("--no-cue", action="store_true", help="the depth cue (shape_check) off")
    ap.add_argument("--shape-tol", type=float, nargs=2, default=None,
                    help="the depth cue's tol_px, tol_rel (default: S2's recommendation)")
    ap.add_argument("--loops", type=int, default=3, help="IPR loops per frame (each detects on every image)")
    ap.add_argument("--k-detect", type=float, default=None, help="the line detector's threshold (default 6)")
    ap.add_argument("--reduced", type=int, default=1, choices=[0, 1],
                    help="1: the reduced loops accept matches in two of the three cameras (D3); 0: three required")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    if args.what == "copy-to-x":
        copy_to_x(tag)
        return
    lpt = ts.lib()
    vol = ts.view_volume(lpt)
    log = case.Log(os.path.join(HERE, f"s3_track_{args.what}{tag}"))
    rd = os.path.join(ts.DATA, "runs", f"s3_{args.what}{tag}")
    build = args.build or ts.BUILD
    noise = ts.noise_per_camera()
    tracer = {"intensity_threshold": 28, "radius_px": 2.0, "detector": "lines", "lines": {"noise": noise},
              "otf_file": os.path.join(ts.DATA, "T1_otf_shape.txt"), "roi_from_otf": True, "roi_sigma": 3.0,
              "shape_check": not args.no_cue}
    if args.k_detect is not None:
        tracer["lines"]["k_detect"] = args.k_detect
    if not args.no_cue:
        tracer.update(shape_tolerances() if args.shape_tol is None else
                      {"shape_tol_px": args.shape_tol[0], "shape_tol_rel": args.shape_tol[1]})
    ipr = ({"loops": args.loops, "reduced_cameras": 1, "reduced_loops": 2, "tol_2d_px": args.tol2d, "tol_3d_vox": 2.0}
           if args.reduced else
           {"loops": args.loops, "reduced_cameras": 0, "reduced_loops": 0, "tol_2d_px": args.tol2d, "tol_3d_vox": 2.0})
    extra = None
    if args.what == "final":
        frames = args.frames or (0, 249)
    elif args.what == "control":
        frames = args.frames or (0, 149)
        n = frames[1] + 1
        extra = {"image_lists": [os.path.join(ts.PRE, "cam1ImageNames.txt"),
                                 shifted_list(2, args.shift[0], n, tag), shifted_list(3, args.shift[1], n, tag)]}
    else:                                                    # pair12, control12: cams 1-2 only
        frames = args.frames or ((0, 249) if args.what == "pair12" else (0, 149))
        otf2 = os.path.join(ts.DATA, "s3", "T1_otf_shape_cams12.txt")
        os.makedirs(os.path.dirname(otf2), exist_ok=True)
        otf_subset(os.path.join(ts.DATA, "T1_otf_shape.txt"), otf2, [0, 1])
        tracer["otf_file"] = otf2
        tracer["lines"] = {"noise": noise[:2], **({"k_detect": args.k_detect} if args.k_detect is not None else {})}
        ipr = {"loops": args.loops, "reduced_cameras": 0, "reduced_loops": 0, "tol_2d_px": args.tol2d, "tol_3d_vox": 2.0}
        if args.what == "control12":
            extra = {"image_lists": [os.path.join(ts.PRE, "cam1ImageNames.txt"),
                                     shifted_list(2, args.shift[0], frames[1] + 1, tag + "_c12")]}
    cfg_path = os.path.join(ts.DATA, "s3", f"config_{args.what}{tag}.json")
    ts.stb_config(cfg_path, frames, vol, tracer, ipr=ipr, extra=extra, fps=FPS)
    if args.what in ("pair12", "control12"):
        with open(cfg_path, encoding="utf-8") as fh:
            c = json.load(fh)
        c["cameras"] = c["cameras"][:2]
        c["image_lists"] = c["image_lists"][:2]
        c["_comment"] += f" Cams 1 and 2 only ({args.what})."
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(c, fh, indent=1)
    log(f"S3 {args.what}: build {build}, frames {frames[0]}-{frames[1]}, tracer {tracer}, ipr {ipr}"
        + (f", cam2 shifted by {args.shift[0]} and cam3 by {args.shift[1]} frames" if args.what == "control" else "")
        + (f", cams 1-2 only, cam2 shifted by {args.shift[0]} frames" if args.what == "control12" else "")
        + f"; OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS')}; frames {ts.PRE}, cameras {ts.CAMS_DIR}")
    info = stbruns.run(cfg_path, rd, build)
    res = parse_log(os.path.join(rd, "log.txt"))
    log(f"  exit {info['returncode']}, wall {info['wall_s']:.0f} s; {res['frames_done']} frames, "
        f"{res['seconds_per_frame_mean']:.2f} s per frame; active long tracks at the end {res['active_long_tracks_last']}")
    log.save({"info": info, "tracer": tracer, "ipr": ipr, "frames": frames,
              "shift": args.shift if args.what in ("control", "control12") else None, **res})


if __name__ == "__main__":
    main()
