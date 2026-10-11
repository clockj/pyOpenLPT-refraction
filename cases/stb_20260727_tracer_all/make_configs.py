"""Write the tracking configs of every run (I3).

Each run (test case, e.g. 15Hz_r_b_2) gets two configs with the same settings:

  on X: (written with --case), what the GUI's Save writes:
    <run>\\config.json                           frames 0..N-1 (or 0..frames_end of config.json if earlier; the image
                                                lists still name all N files), 5000 Hz, 26 threads, the new camFiles,
                                                imgFile_mask,
                                                output Results_tracer, save_interval 200, view volume and voxel of
                                                10Hz_r_b_1's new_cams run, the tracer settings of its tracerConfig.txt
                                                with config.json's tracking_changes and the run's threshold (the
                                                user's choices of 11 Oct, report section 12)
    <run>\\Results_tracer\\camFile\\cam0..3.txt     byte copies of T0\\calib_plate_wand_ref_centre\\camFile
    <run>\\imgFile_mask\\cam0..3ImageNames.txt     image lists where missing (10Hz_r_b_1 keeps its own), as the GUI's
                                                preprocessing writes them: paths relative to the run folder

  on D: (written with --stage), the folder STB runs from, so that its saves go to a local disk:
    <stage>\\<run>\\config.json                   the same settings; camFiles and output relative to this folder
    <stage>\\<run>\\camFile\\cam0..3.txt            byte copies of the same camFiles
    <stage>\\<run>\\imgLists\\cam0..3ImageNames.txt the images on X:, by absolute path

Existing files are left alone unless they hold exactly what would be written (then "same"); --force rewrites the
configs (never an existing image list).  Both configs are read back with the C++ reader (BasicSetting.readConfig,
TracerConfig.readConfig), which must find the frames, the linking spacing and the finder threshold.
The tests build their configs with stage_config() without a run: tracerConfig.txt's settings, unchanged.

    python make_configs.py --stage [--case] [RUN ...]
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import shutil

import case
from modules import stb_config

CASE_COMMENT = ("Written by pyOpenLPT/cases/stb_20260727_tracer_all/make_configs.py on {date}: tracer tracking with "
                "the flat-window wand calibration (Results\\T0\\calib_plate_wand_ref_centre, copied to "
                "Results_tracer\\camFile), as run 10Hz_r_b_1\\stb_new_calib\\new_cams.  The batch runs STB from a "
                "copy on D: (saves to X: are slow) and copies its output to Results_tracer when the run ends.  "
                "Tracking settings: 10Hz_r_b_1's tracerConfig.txt with linking spacing (avg_spacing_vox) {spacing:g} "
                "voxels and finder threshold {threshold} (chosen 11 Oct 2026).")


def tracer_object(cfg):
    obj = stb_config.read_text_object(cfg["tracer_config_source"], "Tracer")
    return obj


def run_object(cfg, run):
    """The tracer settings of a run in the batch: tracerConfig.txt's, with config.json's tracking_changes and the
    run's finder threshold (config.json "thresholds"; a run not listed keeps tracerConfig.txt's)."""
    obj = tracer_object(cfg)
    for key, val in cfg.get("tracking_changes", {}).items():
        sect, field = key.split(".")
        assert field in obj[sect], key
        obj[sect][field] = val
    if run in cfg.get("thresholds", {}):
        assert "intensity_threshold" in obj["tracer"]
        obj["tracer"]["intensity_threshold"] = cfg["thresholds"][run]
    return obj


def last_frame(cfg, n_frames):
    """The last frame tracked: the run's last image, or the case's frames_end if that is earlier (the image lists
    keep naming every file)."""
    return min(n_frames - 1, cfg.get("frames_end", n_frames - 1))


def base_config(cfg, n_frames, run=None):
    """The settings both configs share; with a run, its batch settings (run_object), without, tracerConfig.txt's."""
    vv = cfg["view_volume_mm"]
    return {
        "frames": [0, last_frame(cfg, n_frames)],
        "fps": cfg["fps"],
        "threads": cfg["threads"],
        "view_volume_mm": {"x": vv[0:2], "y": vv[2:4], "z": vv[4:6]},
        "voxel_mm": cfg["voxel_mm"],
        "output_dir": cfg["output_folder"],
        "save_interval": cfg["save_interval"],
        "objects": [run_object(cfg, run) if run else tracer_object(cfg)],
    }


def case_config(cfg, n_frames, run):
    """The config in the run folder on X:, as the GUI writes it."""
    out = cfg["output_folder"]
    c = base_config(cfg, n_frames, run)
    c["cameras"] = [{"file": f"{out}/camFile/{f}", "max_intensity": 255} for f in cfg["camfiles"]]
    c["image_lists"] = [f"{cfg['image_folder']}/cam{k}ImageNames.txt" for k in range(4)]
    c["load_tracks"] = {"enabled": False, "frame": 0, "folder": f"{out}/ConvergeTrack/"}
    obj = c["objects"][0]
    c["_comment"] = CASE_COMMENT.format(date=datetime.date.today().isoformat(),
                                        spacing=obj["convergence_phase"]["avg_spacing_vox"],
                                        threshold=obj["tracer"]["intensity_threshold"])
    return c


def stage_config(cfg, n_frames, frames=None, threads=None, output=None, run=None):
    """The config STB runs from, in the staging folder on D: (frames, threads and output folder can be changed for
    a test; with a run, its batch settings, without, tracerConfig.txt's)."""
    c = base_config(cfg, n_frames, run)
    c["cameras"] = [{"file": f"camFile/{f}", "max_intensity": 255} for f in cfg["camfiles"]]
    c["image_lists"] = [f"imgLists/cam{k}ImageNames.txt" for k in range(4)]
    c["load_tracks"] = {"enabled": False}
    if frames is not None:
        c["frames"] = list(frames)
    if threads is not None:
        c["threads"] = threads
    if output is not None:
        c["output_dir"] = output
    return c


def json_text(c):
    return json.dumps(stb_config.to_json_dict(c, keep_unknown=True), indent=2) + "\n"


def write_text(path, text, force, newline="\n"):
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            old = f.read()
        if old == text.replace("\n", newline):
            return "same"
        if not force:
            raise FileExistsError(f"{path} exists with other contents (use --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
    return "written"


def copy_exact(src, dst):
    with open(src, "rb") as f:
        data = f.read()
    if os.path.exists(dst):
        with open(dst, "rb") as f:
            if f.read() == data:
                return "same"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return "written"


def image_entries(cfg, cam, n_frames):
    return [f"{cfg['image_folder']}/cam{cam}/" + cfg["image_name"].format(i) for i in range(n_frames)]


def check_cpp(path, last, obj):
    """Read a config with the C++ reader, which must find the frames and the object's linking spacing and finder
    threshold; return its frames, output folder, save interval, spacing and threshold."""
    lpt = case.import_lpt()
    s = lpt.BasicSetting()
    s.readConfig(path)
    assert (s._frame_start, s._frame_end) == (0, last) or last is None, (s._frame_start, s._frame_end)
    tc = lpt.TracerConfig()
    tc.readConfig(path + "#0", s)
    spacing = tc._stb_param._radius_search_track / s._voxel_to_mm
    assert abs(spacing - obj["convergence_phase"]["avg_spacing_vox"]) < 1e-9, (path, spacing)
    assert tc._min_obj_int == obj["tracer"]["intensity_threshold"], (path, tc._min_obj_int)
    return {"frames": [s._frame_start, s._frame_end], "threads": s._n_thread, "output": s._output_path,
            "save_interval": s._save_interval, "voxel_mm": s._voxel_to_mm, "n_cam": s._n_cam,
            "avg_spacing_vox": round(spacing, 9), "threshold": tc._min_obj_int}


def write_case(cfg, run, n_frames, force):
    d = case.case_dir(cfg, run)
    st = {}
    for f in cfg["camfiles"]:
        st[f"camFile/{f}"] = copy_exact(os.path.join(cfg["camfile_source"], f),
                                        os.path.join(case.output_dir(cfg, run), "camFile", f))
    for k in range(4):
        p = os.path.join(d, cfg["image_folder"], f"cam{k}ImageNames.txt")
        want = image_entries(cfg, k, n_frames)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                have = [ln.strip() for ln in fh if ln.strip()]
            assert have == want, f"{p} exists and names other images"
            st[f"cam{k}ImageNames.txt"] = "kept (exists, names the same images)"
        else:
            # as the GUI's preprocessing writes them (text mode: CRLF on Windows)
            st[f"cam{k}ImageNames.txt"] = write_text(p, "".join(e + "\n" for e in want), False, newline="\r\n")
    p = os.path.join(d, "config.json")
    c = case_config(cfg, n_frames, run)
    st["config.json"] = write_text(p, json_text(c), force)
    st["cpp"] = check_cpp(p, last_frame(cfg, n_frames), c["objects"][0])
    return st


def write_stage(cfg, run, n_frames, force):
    d = case.stage_dir(cfg, run)
    st = {}
    for f in cfg["camfiles"]:
        st[f"camFile/{f}"] = copy_exact(os.path.join(cfg["camfile_source"], f), os.path.join(d, "camFile", f))
    root = case.case_dir(cfg, run)
    for k in range(4):
        text = "".join(os.path.join(root, e.replace("/", os.sep)) + "\n" for e in image_entries(cfg, k, n_frames))
        st[f"imgLists/cam{k}"] = write_text(os.path.join(d, "imgLists", f"cam{k}ImageNames.txt"), text, force)
    p = os.path.join(d, "config.json")
    c = stage_config(cfg, n_frames, run=run)
    st["config.json"] = write_text(p, json_text(c), force)
    st["cpp"] = check_cpp(p, last_frame(cfg, n_frames), c["objects"][0])
    return st


def frames_of(run):
    with open(os.path.join(case.TESTS, "inventory.json"), encoding="utf-8") as f:
        inv = json.load(f)
    r = inv["runs"][run]
    assert r["ok"], f"{run}: the inventory found a problem"
    return r["frames"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--stage", action="store_true", help="write the staging configs on D:")
    ap.add_argument("--case", action="store_true", help="write the configs, camFiles and image lists on X:")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = case.load_config()
    runs = args.runs or cfg["cases"]
    for run in runs:
        n = frames_of(run)
        if args.stage:
            st = write_stage(cfg, run, n, args.force)
            print(f"{run} stage: {case.stage_dir(cfg, run)}  {st}")
        if args.case:
            st = write_case(cfg, run, n, args.force)
            print(f"{run} case:  {case.case_dir(cfg, run)}  {st}")


if __name__ == "__main__":
    main()
