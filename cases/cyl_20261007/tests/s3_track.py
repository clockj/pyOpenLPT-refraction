"""S3 (plan, Phase 6): STB on T1.

    python tests/s3_track.py final [--frames 0 250] [--build p4c] [--tol2d 1.0]
    python tests/s3_track.py control [--frames 0 100] [--shift 100]
    python tests/s3_track.py today [--frames 0 60]
    python tests/s3_track.py copy-to-x

S1's preprocessed frames (every 4th raw frame: frame index i = raw frame 4 i), T0's tracking cameras (camFile_track),
the view volume both cameras see (t1_setup.view_volume), voxel 0.03 mm (about 1 px in both cameras).
  - final: this work's code (build BUILD): the line detector (tracer.detector "lines", each camera's noise from S1),
    S2's shape model as the OTF (tracer.otf_file: data/T1_otf_shape.txt, model "segment") with regions from it
    (tracer.roi_from_otf), and the depth cue (tracer.shape_check, tolerances from S2's held-out residuals); 3 IPR
    loops, 2D tolerance 1 px (C1's residual pattern reaches 0.5-1 px), 3D tolerance 2 voxels;
  - control (T1-e, added to the plan's T1-d): final's settings with cam2's images SHIFT steps later (0.1 s for 100:
    the particles have moved far), so that no true match exists and every track is a ghost.  Its active long tracks
    against final's at the same frame estimate the share of ghosts in final;
  - today: today's code (build head1007) with today's settings on the same frames: the peak finder at 4 sigma of each
    camera's 8-bit noise (cam1 32, cam2 24 -> one threshold, 28), radius 2 px, the OTF estimated from the images.
Runs in data/runs/s3_<name>/ (git-ignored), each with its config.json, log.txt and Results/; copy-to-x copies the
final run to <tracers.out_dir> (config.json, log.txt, Results/, the OTF table and the model) and the checks' outputs
(S2, S2b, T1-a to T1-e, the runs' summaries) to its checks\.  Writes s3_track_<name>_out.txt
and .json beside itself (frames, time per frame, objects per frame from the log).
"""
import argparse
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402

sys.path.insert(0, os.path.join(ts.REPO, "tests", "cylinder_stb"))
import stbruns  # noqa: E402


def shape_tolerances():
    """S2's tolerances for tracer.shape_check: the narrowest that keep 90 % of the held-out true pairs."""
    with open(os.path.join(ts.DATA, "T1_shape.json"), encoding="utf-8") as fh:
        sc = json.load(fh)["shape_check"]
    return {"shape_tol_px": sc["tol_px"], "shape_tol_rel": sc["tol_rel"]}


def parse_log(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    times = [float(x) for x in re.findall(r"Total time for frame \d+: ([0-9.]+)", text)]
    dets = [tuple(int(v) for v in m.groups()) for m in re.finditer(r"2D detections per active camera: (\d+)\s+(\d+)", text)]
    longs = [int(m.group(1)) for m in re.finditer(r"No\. of active long tracks: \d+ \+ \d+ - \d+ = (\d+)", text)]
    return {"frames_done": len(times), "seconds_per_frame_mean": sum(times) / max(1, len(times)),
            "active_long_tracks_last": longs[-1] if longs else None, "active_long_tracks": longs,
            "detections_first_ipr_loop": dets[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["final", "control", "today", "copy-to-x"])
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    ap.add_argument("--build", default=None)
    ap.add_argument("--tol2d", type=float, default=1.0, help="final, control: IPR's 2D tolerance [px]")
    ap.add_argument("--shift", type=int, default=100, help="control: cam2's images SHIFT steps later")
    args = ap.parse_args()
    cfg = case.load_config()
    if args.what == "copy-to-x":
        src = os.path.join(ts.DATA, "runs", "s3_final")
        dst = cfg["tracers"]["out_dir"]
        # config.json on X: the run's settings with paths to the copies there (images\, otf_shape.txt) and to T0's
        # tracking camFiles, so that the folder reruns as it is; the run's own config (local paths): config_local.json
        with open(os.path.join(src, "config_final.json"), encoding="utf-8") as fh:
            run_cfg = json.load(fh)
        t0_track = os.path.join(os.path.dirname(dst), "T0", "camFile_track")
        run_cfg["_comment"] += " Copied to X: with the paths of the copies there."
        run_cfg["cameras"] = [{**c, "file": os.path.join(t0_track, os.path.basename(c["file"]))}
                              for c in run_cfg["cameras"]]
        run_cfg["image_lists"] = [os.path.join(dst, "images", f"cam{k + 1}ImageNames.txt")
                                  for k in range(len(run_cfg["cameras"]))]
        run_cfg["objects"][0]["tracer"]["otf_file"] = "otf_shape.txt"
        run_cfg["output_dir"] = "Results"
        with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(run_cfg, fh, indent=1)
        shutil.copy2(os.path.join(src, "config_final.json"), os.path.join(dst, "config_local.json"))
        for name in ("log.txt", "run_info.json"):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
        if os.path.isdir(os.path.join(dst, "Results")):
            shutil.rmtree(os.path.join(dst, "Results"))
        shutil.copytree(os.path.join(src, "Results"), os.path.join(dst, "Results"))
        shutil.copy2(os.path.join(ts.DATA, "T1_otf_shape.txt"), os.path.join(dst, "otf_shape.txt"))
        shutil.copy2(os.path.join(ts.DATA, "T1_shape.json"), os.path.join(dst, "shape_model.json"))
        # the checks: S2, S2b, T1-a to T1-e and the runs' summaries, as written beside these scripts
        chk = os.path.join(dst, "checks")
        os.makedirs(chk, exist_ok=True)
        names = [f"{stem}{ext}" for stem in ("s2_shape_fit", "s2b_planes_check", "t1a_epipolar_offset", "t1_checks",
                                             "t1_checks_s3_today", "t1e_ghost_control",
                                             "t1e_ghost_control_s3_today", "s3_track_final",
                                             "s3_track_control", "s3_track_today")
                 for ext in ("_out.txt", ".json", ".png")]
        names += ["s2_shape_fit_console.txt", "t1_tracks.png", "t1_flow.png", "t1_tracks_s3_today.png",
                  "t1_flow_s3_today.png"]
        copied = [n for n in names if os.path.exists(os.path.join(HERE, n))]
        for n in copied:
            shutil.copy2(os.path.join(HERE, n), os.path.join(chk, n))
        print(f"copied {src} -> {dst}; {len(copied)} check files -> {chk}")
        return
    lpt = ts.lib()
    vol = ts.view_volume(lpt)
    log = case.Log(os.path.join(HERE, f"s3_track_{args.what}"))
    rd = os.path.join(ts.DATA, "runs", f"s3_{args.what}")
    ipr, extra = None, None
    if args.what in ("final", "control"):
        frames = args.frames or ((0, 250) if args.what == "final" else (0, 100))
        build = args.build or ts.BUILD
        tol = shape_tolerances()
        tracer = {"intensity_threshold": 28, "radius_px": 2.0, "detector": "lines",
                  "lines": {"noise": ts.noise_per_camera()},
                  "otf_file": os.path.join(ts.DATA, "T1_otf_shape.txt"), "roi_from_otf": True, "roi_sigma": 3.0,
                  "shape_check": True, **tol}
        ipr = {"loops": 3, "reduced_cameras": 0, "reduced_loops": 0, "tol_2d_px": args.tol2d, "tol_3d_vox": 2.0}
        if args.what == "control":
            with open(os.path.join(ts.PRE, "cam2ImageNames.txt"), encoding="utf-8") as fh:
                names = [ln for ln in fh.read().splitlines() if ln.strip()]
            if frames[1] + args.shift >= len(names):
                raise SystemExit(f"frames {frames} + shift {args.shift} pass the {len(names)} images")
            lst = os.path.join(ts.DATA, "s3", f"cam2ImageNames_shift{args.shift}.txt")
            os.makedirs(os.path.dirname(lst), exist_ok=True)
            with open(lst, "w", encoding="utf-8") as fh:
                fh.write("\n".join(names[args.shift:]) + "\n")
            extra = {"image_lists": [os.path.join(ts.PRE, "cam1ImageNames.txt"), lst]}
    else:
        frames = args.frames or (0, 60)
        build = args.build or "head1007"
        tracer = {"intensity_threshold": 28, "radius_px": 2.0}
    cfg_path = os.path.join(ts.DATA, "s3", f"config_{args.what}.json")
    ts.stb_config(cfg_path, frames, vol, tracer, ipr=ipr, extra=extra)
    log(f"S3 {args.what}: build {build}, frames {frames[0]}-{frames[1]} (raw {4 * frames[0]}-{4 * frames[1]}), "
        f"tracer {tracer}, ipr {ipr}" + (f", cam2 shifted by {args.shift} steps" if args.what == "control" else ""))
    info = stbruns.run(cfg_path, rd, build)
    res = parse_log(os.path.join(rd, "log.txt"))
    log(f"  exit {info['returncode']}, wall {info['wall_s']:.0f} s; {res['frames_done']} frames, "
        f"{res['seconds_per_frame_mean']:.2f} s per frame; active long tracks at the end {res['active_long_tracks_last']}")
    log.save({"info": info, "tracer": tracer, "ipr": ipr, "frames": frames,
              "shift": args.shift if args.what == "control" else None, **res})


if __name__ == "__main__":
    main()
