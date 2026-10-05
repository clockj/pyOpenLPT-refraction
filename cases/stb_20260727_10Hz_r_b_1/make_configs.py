"""Write the run folders of the case on X: (2026-10-04-plan.md, section 5).

    out_dir (stb_new_calib)
      README.md
      imgLists\cam0..3ImageNames.txt          image lists for configs one level down (new_cams, old_cams)
      new_cams\  config.txt, tracerConfig.txt, camFile\cam0..3.txt     T4: the new camFiles
      old_cams\  config.txt, tracerConfig.txt, camFile\vsc_cam0..3.txt T5: the old camFiles (control)
      ipr_probe\imgLists\...                   image lists for configs two levels down
      ipr_probe\<set>_f<frame>\config.txt      T3: one IPR on one frame (frame range [frame, frame])

A config resolves relative paths (camFiles, image lists, and the image paths inside the lists) from its own folder,
so every list holds paths relative to its configs' depth.  tracerConfig.txt is the case's own, copied unchanged.  The
new config's view volume and voxel size come from T0 (tests/t0_frame_map.json); everything else is the case's
config.txt.  Existing files are never overwritten unless --force is given; the case's own files are only read.

    python make_configs.py            # all folders
    python make_configs.py --force    # rewrite them
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import case

CONFIG_TEMPLATE = """# Frame Range: [startID,endID]
{f0},{f1}
# Frame Rate: [Hz]
{fps}
# Number of Threads: (0: use as many as possible)
{threads}
# Number of Cameras:
4
# Camera File Path, Max Intensity
{cams}
# Image File Path
{imgs}
# View Volume: (xmin,xmax,ymin,ymax,zmin,zmax)
{vv}
# Voxel to MM: e.g. use 1000^3 voxel, (xmax-xmin)/1000
{voxel}
# Output Folder Path:
Results
# Object Types:
Tracer
# STB Config Files:
{tracer}
# Flag to load previous track files, previous frameID
0
"""


def read_old_config(cfg):
    """The case's config.txt as its data lines (comments dropped)."""
    with open(case.case_path(cfg, cfg["old_config"]), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def write(path, text, force):
    if os.path.exists(path) and not force:
        with open(path, encoding="utf-8") as f:
            if f.read() == text:
                return "same"
        raise FileExistsError(f"{path} exists with other contents (use --force)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return "written"


def copy_exact(src, dst, force):
    """A byte-exact copy (camFiles and tracerConfig keep their own line endings)."""
    with open(src, "rb") as f:
        data = f.read()
    if os.path.exists(dst) and not force:
        with open(dst, "rb") as f:
            if f.read() == data:
                return "same"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return "written"


def image_lists(cfg, folder, depth, force):
    """Image lists whose entries point to the case's images from configs `depth` levels below the case folder."""
    up = "../" * depth
    out = []
    for i, rel in enumerate(cfg["image_lists"]):
        with open(case.case_path(cfg, rel), encoding="utf-8") as f:
            entries = [ln.strip().replace("\\", "/") for ln in f if ln.strip()]
        assert all(not os.path.isabs(e) for e in entries), "expected relative image paths"
        dst = os.path.join(folder, f"cam{i}ImageNames.txt")
        write(dst, "".join(up + e + "\n" for e in entries), force)
        out.append(dst)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = case.load_config()
    with open(os.path.join(case.TESTS, "t0_frame_map.json"), encoding="utf-8") as f:
        t0 = json.load(f)
    old_lines = read_old_config(cfg)
    fps = int(old_lines[1])
    out = cfg["out_dir"]
    tracer_src = case.case_path(cfg, cfg["tracer_config"])

    sets = {
        "new": {"dir": "new_cams", "vv": t0["new_view_volume"], "voxel": t0["new_voxel_to_mm"],
                "camfiles": case.camfile_paths(cfg, "new"), "frames": cfg["frames"]["t4"]},
        "old": {"dir": "old_cams", "vv": cfg["old_view_volume"], "voxel": cfg["old_voxel_to_mm"],
                "camfiles": case.camfile_paths(cfg, "old"), "frames": cfg["frames"]["t5"]},
    }
    image_lists(cfg, os.path.join(out, "imgLists"), 2, args.force)
    image_lists(cfg, os.path.join(out, "ipr_probe", "imgLists"), 3, args.force)
    for name, s in sets.items():
        run = os.path.join(out, s["dir"])
        cam_rel = []
        for src in s["camfiles"]:
            copy_exact(src, os.path.join(run, "camFile", os.path.basename(src)), True)
            cam_rel.append("camFile/" + os.path.basename(src))
        copy_exact(tracer_src, os.path.join(run, "tracerConfig.txt"), True)
        vv = ",".join(f"{v:.1f}" for v in s["vv"])
        txt = CONFIG_TEMPLATE.format(f0=s["frames"][0], f1=s["frames"][1], fps=fps, threads=cfg["threads"],
                                     cams="\n".join(c + ",255" for c in cam_rel),
                                     imgs="\n".join(f"../imgLists/cam{i}ImageNames.txt" for i in range(4)),
                                     vv=vv, voxel=f"{s['voxel']:g}", tracer="tracerConfig.txt")
        write(os.path.join(run, "config.txt"), txt, args.force)
        print(f"{name}: {run}\\config.txt  frames {s['frames']}  view volume {vv}  voxel {s['voxel']:g}")
        for fr in cfg["ipr_probe_frames"]:
            probe = os.path.join(out, "ipr_probe", f"{name}_f{fr:04d}")
            txt = CONFIG_TEMPLATE.format(f0=fr, f1=fr, fps=fps, threads=cfg["threads"],
                                         cams="\n".join(f"../../{s['dir']}/{c},255" for c in cam_rel),
                                         imgs="\n".join(f"../imgLists/cam{i}ImageNames.txt" for i in range(4)),
                                         vv=vv, voxel=f"{s['voxel']:g}",
                                         tracer=f"../../{s['dir']}/tracerConfig.txt")
            write(os.path.join(probe, "config.txt"), txt, args.force)
        print(f"   ipr_probe: frames {cfg['ipr_probe_frames']}")
    readme = f"""# Tracking with the new flat-window calibration (case stb_20260727_10Hz_r_b_1)

Written by `pyOpenLPT/cases/stb_20260727_10Hz_r_b_1/make_configs.py` (plan `2026-10-04-plan.md`, report there).
Every config runs pyOpenLPT's STB unchanged; the case's own files (`..\\config.txt`, `..\\tracerConfig.txt`,
`..\\camFile_VSC`, `..\\imgFile_mask`, `..\\Results`) are not touched.

| folder | contents |
|---|---|
| `new_cams\\` | T4: STB on frames {sets['new']['frames'][0]}-{sets['new']['frames'][1]}, stopped after frame {cfg['frames'].get('t4_stopped_after', sets['new']['frames'][1])}, with the new camFiles (copies of `Results\\T0\\calib_plate_wand_ref_centre\\camFile`). View volume {sets['new']['vv']} mm and voxel {sets['new']['voxel']:g} mm: the old view volume mapped into the new calibration's frame and clipped to the water (test T0). `log.txt`, `Results\\`. |
| `old_cams\\` | T5: the same frames with the old camFiles (copies of `..\\camFile_VSC`) and the old view volume {sets['old']['vv']}, voxel {sets['old']['voxel']:g} mm: the control. |
| `new_cams_restart500\\` | T4c: a fresh process that loads T4's active tracks at frame 500 (`load\\`) and runs frames 501-600 (written by `tests\\t4c_restart.py`). |
| `ipr_probe\\<set>_f<frame>\\` | T3: one IPR, without tracking, on one frame with each camFile set. |
| `imgLists\\`, `ipr_probe\\imgLists\\` | image lists pointing to `..\\imgFile_mask` (a config resolves relative image paths from its own folder). |

The world frame of the new camFiles differs from the old one: x and y change sign (a half-turn about the C1/C2 wall's
normal), and lengths are 1.229 x the old frame's.  So `new_cams` and `old_cams` give positions in different frames.
"""
    write(os.path.join(out, "README.md"), readme, args.force)
    print(f"README: {out}\\README.md")


if __name__ == "__main__":
    main()
