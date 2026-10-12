"""T1's results after VSC into a new subfolder of T1's deliverable on X: (asked 8 Oct: "create a new subfolder to save
the results after vsc and the corresponding results there").  T1's own deliverable (T1\, the 15 px run) and its
run_31px\ stay as they are.

    python v3_to_x.py [--name after_vsc] [--dry-run]

Writes <tracers.out_dir>\<NAME>\ (refuses if it exists):
  - the two-pass run (s3_final_m15vc2): Results\, log.txt, run_info.json; config_local.json (as run, local paths) and
    config.json (the same with the paths of the copies here and of ..\..\T0\camFile_track\: reruns as it is with build
    p4c); otf_shape.txt and shape_model.json (T1's, unchanged);
  - images\cam1, cam2: the frames it tracked (the 15 px frames warped by the VSC field), image lists with X: paths,
    manifest.json (each frame's noise);
  - control\: its control run's config (cam2 100 steps later, X: paths; reruns as it is), log and run_info (the counts
    estimate of the ghost share reads its log);
  - vsc\: the correction fields (pass 1, pass 2 and their total, the one the frames were warped by: .npz), a figure of
    the total, and a README of the format;
  - checks\: T1-a to T1-e on this run (v3_t1checks.py: the case's checks, ported), the five runs compared
    (v3_compare.py), the positions against T1\'s (v3_positions_t1.py), V0's pattern, V1's held-out benefit, the warp
    check, the fits and warps' outputs, the synthetic tests' tables (summary_sc), and this plan's report and plan;
  - README.md.
Writes v3_to_x_out.txt and .json here (what was copied, the sizes, sha256 of the track files).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np

import selfcal as sc

RUN, CONTROL, PRE = "s3_final_m15vc2", "s3_control_m15vc2", os.path.join(sc.DATA, "T1pre_s4_m15vc2")
FIELDS = {"pass1.npz": "t1_m15_all.npz", "pass2.npz": "t1_vc_pass2.npz", "total.npz": "t1_vc2_total.npz"}
CHECKS = ["t1_checks_s3_final_m15vc2_out.txt", "t1_checks_s3_final_m15vc2.json", "t1_tracks_s3_final_m15vc2.png",
          "t1_flow_s3_final_m15vc2.png", "t1e_ghost_control_s3_final_m15vc2_out.txt",
          "t1e_ghost_control_s3_final_m15vc2.json", "t1e_ghost_control_s3_final_m15vc2.png",
          "v3_compare_out.txt", "v3_compare.json", "v3_compare.png", "v3_positions_t1_out.txt", "v3_positions_t1.json",
          "v0_pattern_out.txt", "v0_pattern.json", "v0_pattern.png", "v0_fields.png",
          "v0b_truthfree_out.txt", "v0b_truthfree.json", "v0b_truthfree.png",
          "v1_benefit_out.txt", "v1_benefit.json", "v1_benefit.png",
          "v3a_warpcheck_out.txt", "v3a_warpcheck.json", "v3a_warpcheck.png",
          "fit_t1_t1_vc_out.txt", "fit_t1_t1_vc.json", "compose_fields_t1_vc2_total_out.txt",
          "compose_fields_t1_vc2_total.json", "v3_warp_vc_out.txt", "v3_warp_vc.json", "v3_warp_vc2_out.txt",
          "v3_warp_vc2.json", "t1_runs_final_m15vc2_out.txt", "t1_runs_final_m15vc2.json",
          "t1_runs_control_m15vc2_out.txt", "t1_runs_control_m15vc2.json",
          "summary_sc_out.txt", "summary_sc.png", "v3_xcheck_out.txt", "v3_xcheck.json", "2026-10-08-report.md",
          "2026-10-08-plan.md"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def field_figure(path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = sc.Field.load(os.path.join(sc.DATA, "fields", FIELDS["total.npz"]))
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for c in (0, 1):
        for k, nm in ((0, "du"), (1, "dv")):
            a = ax[c, k]
            img = np.where(f.covered[c], f.dense[c][k], np.nan)
            im = a.imshow(img, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            a.set_title(f"cam{c + 1} {nm} [px]: the VSC correction (two passes); RMS "
                        f"{np.sqrt(np.mean(f.dense[c][k][f.covered[c]] ** 2)):.2f} px", fontsize=9)
            plt.colorbar(im, ax=a, fraction=0.03)
    fig.suptitle("c(u, v) = where each camera's images sit relative to the model's projection; the frames here are "
                 "warped by -c (blank: beyond the data, where c holds its edge value)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path_png, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="after_vsc")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg_case = sc.load_json(os.path.join(sc.CASE, "config.json"))
    t1_dir = cfg_case["tracers"]["out_dir"]
    dst = os.path.join(t1_dir, a.name)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists: choose another --name or remove it by hand")
    t0_track = os.path.join(os.path.dirname(t1_dir), "T0", "camFile_track")
    if not os.path.isdir(t0_track):
        raise SystemExit(f"no {t0_track}")
    src_run, src_ctl = os.path.join(sc.CASE_RUNS, RUN), os.path.join(sc.CASE_RUNS, CONTROL)
    missing = [p for p in [os.path.join(sc.HERE, n) for n in CHECKS]
               + [os.path.join(sc.DATA, "fields", v) for v in FIELDS.values()] if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    log = sc.Log("v3_to_x" if not a.dry_run else "v3_to_x_dryrun")
    log(f"T1 after VSC -> {dst}{' (dry run)' if a.dry_run else ''}")
    plan = {"run": src_run, "control": src_ctl, "frames": PRE, "dst": dst}
    n_img = sum(len(os.listdir(os.path.join(PRE, c))) for c in ("cam1", "cam2"))
    size_img = sum(os.path.getsize(os.path.join(PRE, c, n)) for c in ("cam1", "cam2")
                   for n in os.listdir(os.path.join(PRE, c)))
    log(f"  {n_img} frames ({size_img / 1e6:.0f} MB), the run's Results\\, its control's config and log, 3 fields, "
        f"{len(CHECKS)} check files")
    if a.dry_run:
        log.save(plan)
        return
    t0 = time.time()
    os.makedirs(dst)
    # the run
    shutil.copytree(os.path.join(src_run, "Results"), os.path.join(dst, "Results"))
    for n in ("log.txt", "run_info.json"):
        shutil.copy2(os.path.join(src_run, n), os.path.join(dst, n))
    run_cfg = sc.load_json(os.path.join(src_run, f"config_{RUN[3:]}.json"))
    shutil.copy2(os.path.join(src_run, f"config_{RUN[3:]}.json"), os.path.join(dst, "config_local.json"))
    shutil.copy2(os.path.join(sc.CASE_DATA, "T1_otf_shape.txt"), os.path.join(dst, "otf_shape.txt"))
    shutil.copy2(os.path.join(sc.CASE_DATA, "T1_shape.json"), os.path.join(dst, "shape_model.json"))
    # the frames
    img_dst = os.path.join(dst, "images")
    lists = {}
    for c in ("cam1", "cam2"):
        os.makedirs(os.path.join(img_dst, c))
        with open(os.path.join(PRE, f"{c}ImageNames.txt"), encoding="utf-8") as fh:
            names = [os.path.basename(ln.strip()) for ln in fh.read().splitlines() if ln.strip()]
        for n in names:
            shutil.copy2(os.path.join(PRE, c, n), os.path.join(img_dst, c, n))
        lists[c] = [os.path.join(img_dst, c, n) for n in names]
        with open(os.path.join(img_dst, f"{c}ImageNames.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lists[c]) + "\n")
    shutil.copy2(os.path.join(PRE, "manifest.json"), os.path.join(img_dst, "manifest.json"))
    # config.json: the run's settings with the paths here and T0's tracking cameras on X:
    run_cfg["_comment"] = ("T1 of run 20261007 after VSC (pyOpenLPT/tests/cylinder_selfcal, two passes, 2-D tolerance "
                           "1.0 px), copied to X: with the paths of the copies here; reruns as it is with build p4c.")
    run_cfg["cameras"] = [{**c, "file": os.path.join(t0_track, os.path.basename(c["file"]))} for c in run_cfg["cameras"]]
    run_cfg["image_lists"] = [os.path.join(img_dst, f"cam{k}ImageNames.txt") for k in (1, 2)]
    run_cfg["objects"][0]["tracer"]["otf_file"] = "otf_shape.txt"
    run_cfg["output_dir"] = "Results"
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_cfg, fh, indent=1)
    # the control
    ctl = os.path.join(dst, "control")
    os.makedirs(ctl)
    for n in ("log.txt", "run_info.json"):
        shutil.copy2(os.path.join(src_ctl, n), os.path.join(ctl, n))
    ctl_cfg = sc.load_json(os.path.join(src_ctl, f"config_{CONTROL[3:]}.json"))
    shutil.copy2(os.path.join(src_ctl, f"config_{CONTROL[3:]}.json"), os.path.join(ctl, "config_local.json"))
    with open(os.path.join(ctl, "cam2ImageNames_shift100.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lists["cam2"][100:]) + "\n")
    ctl_cfg["_comment"] = ("The control of the run above (cam2's frames 100 steps later: every track a ghost), copied to "
                           "X: with the paths of the copies; reruns as it is with build p4c (its Results are not kept).")
    ctl_cfg["cameras"] = [{**c, "file": os.path.join(t0_track, os.path.basename(c["file"]))} for c in ctl_cfg["cameras"]]
    ctl_cfg["image_lists"] = [os.path.join(img_dst, "cam1ImageNames.txt"),
                              os.path.join(ctl, "cam2ImageNames_shift100.txt")]
    ctl_cfg["objects"][0]["tracer"]["otf_file"] = os.path.join(dst, "otf_shape.txt")
    ctl_cfg["output_dir"] = "Results"
    with open(os.path.join(ctl, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(ctl_cfg, fh, indent=1)
    # the fields
    vsc = os.path.join(dst, "vsc")
    os.makedirs(vsc)
    for n, src in FIELDS.items():
        shutil.copy2(os.path.join(sc.DATA, "fields", src), os.path.join(vsc, n))
    field_figure(os.path.join(vsc, "total.png"))
    with open(os.path.join(vsc, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(VSC_README)
    # the checks
    chk = os.path.join(dst, "checks")
    os.makedirs(chk)
    for n in CHECKS:
        shutil.copy2(os.path.join(sc.HERE, n), os.path.join(chk, n))
    with open(os.path.join(dst, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README)
    # the record
    tf = sorted(os.listdir(os.path.join(dst, "Results", "ConvergeTrack")))
    rec = {**plan, "seconds": time.time() - t0, "frames_copied": n_img,
           "bytes": sum(os.path.getsize(os.path.join(r, f)) for r, _d, fs in os.walk(dst) for f in fs),
           "track_files_sha256": {f: sha(os.path.join(dst, "Results", "ConvergeTrack", f)) for f in tf
                                  if f.endswith("_250.csv")},
           "track_files_equal_local": all(sha(os.path.join(dst, "Results", "ConvergeTrack", f))
                                          == sha(os.path.join(src_run, "Results", "ConvergeTrack", f))
                                          for f in tf if f.endswith("_250.csv"))}
    log(f"  copied in {rec['seconds']:.0f} s: {rec['bytes'] / 1e6:.0f} MB; the final track files equal the local run's: "
        f"{rec['track_files_equal_local']}")
    log.save(rec)


VSC_README = """# The VSC correction fields of T1 (run 20261007)

Fitted by `pyOpenLPT/tests/cylinder_selfcal` (8 Oct 2026) from T1's own long tracks: per camera a smooth field
c(u, v) = (du, dv) [px], where the camera's images sit relative to where T0's calibration (`..\\..\\..\\T0\\camFile_track\\`)
projects them. The frames in `..\\images\\` are T1's 15 px frames warped by -c (every image moved onto its model), so
that the unchanged cameras fit them.

| file | what |
|---|---|
| `pass1.npz` | the first pass: fitted from the 15 px run's tracks (`T1\\`, 1,228 long tracks) |
| `pass2.npz` | the second pass: fitted from the run on the frames corrected by pass 1 (in their coordinates) |
| `total.npz` | the two composed, c(d) = c1(d) + c2(d - c1(d)), in the original frames' pixels: what the frames here were warped by |
| `total.png` | the total per camera and component |

Each .npz (numpy): `dense` (2 cameras, 2 components (du, dv), 800 rows, 1280 columns) float32 in px, at the pixel
centres (u = column, v = row); `covered` (2, 800, 1280) bool, the pixels within 48 px of the data (beyond, `dense`
holds the nearest covered pixel's value); `theta`, `n_u`, `n_v`, ... the B-spline coefficients (10 x 6 intervals per
camera and component) and `meta` (JSON: the fit's settings). To correct a detection d of camera k: d - c_k(d) (read
`dense[k]` bilinearly at d). With two cameras only the offset across the epipolar curves is seen; the part along them
(du, and the split of dv between the cameras) is set by the fit's prior, so the fields are a consistent correction
for matching, not each camera's true error.
"""

README = r"""# Run 20261007 / T1 after VSC: the cameras corrected from T1's own tracks

Made on 8 October 2026 by `pyOpenLPT/tests/cylinder_selfcal` (plan `2026-10-08-plan.md`, report
`2026-10-08-report.md`, both copied to `checks\`). The tracking settings and code are those of T1's deliverable in the
folder above (`..\`, the 15 px run), with two differences: the frames are corrected by a VSC field, and the 2-D
matching tolerance is 1.0 px instead of 1.5. The deliverable above is unchanged.

## Read this first

- **What VSC did.** T1's long tracks showed a smooth, stable pattern in where the cameras' images sit relative to T0's
  calibration: 0.32 px RMS across the epipolar curves, mostly in cam2's v (`checks\v0_pattern_*`). A smooth field
  per camera was fitted to it (`vsc\`), T1's frames were warped by it, T1 was tracked again, the field left in
  those tracks was fitted again (it under-reports the error near the matching tolerance), and T1 was tracked on the
  frames warped by both: two passes.
- **More real tracks, fewer ghosts** (`checks\v3_compare_*`; frame 100, where the control ends):

  | | the folder above (15 px, 1.5 px) | here (VSC, 1.0 px) |
  |---|---|---|
  | real long tracks at frame 100, by the control's counts | 296 | **329 (+11 %)** |
  | ghost share, by the counts / by the swirl | 50 / 53 % | **39 / 48 %** |
  | long tracks in the whole run, their positions | 1,228, 112,383 | 1,132, 105,885 |
  | particles per frame in long tracks (from frame 150) | 571 | 536 |

  Fewer tracks in all, more of them real. The swirl estimate, which understates gains on synthetic cases, gives 279
  against 278 real tracks. Both estimates read 1.5–3.5 times the true ghost share on synthetic cases, so compare them
  between the folders rather than take them at face value.
- **The checks** (`checks\t1_checks_s3_final_m15vc2_*`, `checks\t1e_ghost_control_s3_final_m15vc2_*`, the case's T1-a to
  T1-e as for the folder above):
  - the tracked particles' images fit their detections (RMS 0.27–0.69 px; above 0.24–0.76 px), and the shape model as
    before (68 % within 0.43–0.65 px);
  - the epipolar residual has no mean any more: −0.009 ± 0.030 px (above +0.069 ± 0.034) and RMS 0.91 px (1.00);
  - **the cameras' time offset is gone:** −0.08 ± 0.28 raw frames (above 0.59 ± 0.30, 1.95 standard errors): that was
    the calibration error, not a sync error;
  - the flow is the same: speed 33–108 mm/s (10–90 %, median 70), track length median 80 steps (77), a little smoother
    (second difference over the step 0.54 against 0.55).
- **Positions.** Where both folders track the same particle, x and z agree (median difference 0.0 µm, spread 1 µm)
  and y agrees on average (+0.1 µm) but differs smoothly across the volume by 8 µm RMS (up to 22 µm,
  `checks\v3_positions_t1_*`). With two cameras the tracks see only the offset across the epipolar curves; how a
  correction splits between the cameras moves the reconstruction, and nothing in T1 says which split is right. T0's
  wand fixes the metric: treat y differences of that size between the folders as calibration uncertainty.
- **Ghosts** remain the main limit with two cameras: random pairs near the epipolar curves far outnumber true ones at
  this density. VSC removes the calibration's part of the problem; the shape model's error and the detections are the
  larger part (`checks\2026-10-08-report.md`, sections 1 and 7). A third camera would halve the ghosts.
- **Frame numbers:** `FrameID` in the track files is the processed frame i: raw frame 4 i, time i ms.

## Files

| file | what |
|---|---|
| `images\cam1\`, `images\cam2\` | the frames tracked: the folder above's 15 px frames ((background − frame) − its 15 px median, × gain) warped by the VSC field before the rounding, 8 bits; `images\cam*ImageNames.txt` list them; `images\manifest.json` has each frame's noise (3 % below the unwarped frames': the interpolation) |
| `config.json` | the STB settings, with the paths of the copies here (images, `otf_shape.txt`) and of `..\..\T0\camFile_track\`: this folder reruns as it is with build `p4c` |
| `config_local.json` | the same, as run (local paths) |
| `log.txt`, `run_info.json` | STB's log; the build, time and command |
| `Results\ConvergeTrack\` | the tracks at the last frame (250): `LongTrackActive_250.csv`, `LongTrackInactive_250.csv`, `ExitTrack_250.csv` (long tracks), `ShortTrackActive_250.csv`. Columns: TrackID, FrameID, WorldX, WorldY, WorldZ (mm; the tube's axis is the y axis), then each camera's image (col, row) in the corrected frames |
| `Results\InitialTrack\`, `Results\PredField_*.csv` | STB's initial phase and predictive fields |
| `otf_shape.txt`, `shape_model.json` | the OTF table and the shape model: the folder above's, unchanged |
| `control\` | the control run (cam2's frames 100 steps later, so every track is a ghost): its config (reruns as it is), log and run info; the ghost estimate reads its log |
| `vsc\` | the correction fields (pass 1, pass 2, their total: the one the frames were warped by), a figure, a README of the format |
| `checks\` | `t1_checks_s3_final_m15vc2_*` (T1-a to T1-d, the figures `t1_tracks_*`, `t1_flow_*`), `t1e_ghost_control_s3_final_m15vc2_*` (T1-e), `v3_compare_*` (this run against the folder above, the 15 px frames uncorrected at 1.0 px, and one pass at 1.5 and 1.0 px), `v3_positions_t1_*` (positions against the folder above), `v0_pattern_*` and `v0b_truthfree_*` (the pattern VSC found), `v1_benefit_*` (the gain on held-out frames), `v3a_warpcheck_*` (the warped frames against corrected detections), `fit_t1_*`, `compose_fields_*`, `v3_warp_*` (the second pass's fit and the warps), `t1_runs_*` (the runs), `summary_sc_*` (the synthetic tests, 2, 3 and 4 cameras), `v3_xcheck_*` (this folder's config rerun from X: for frames 0-3: the same objects as the run), the report and the plan |

## How it was made

From `pyOpenLPT/tests/cylinder_selfcal`, with the refract environment and build `p4c` (the README there lists every
step):

1. `gather.py t1_m15 ...` and `v0_pattern.py`: the first field, from the 15 px run's tracks (`data/fields/t1_m15_all.npz`).
2. `v3_warp.py vc ...`, then `t1_runs.py final --tag m15vc --pre data/T1pre_s4_m15vc --tol2d 1.5`: the first pass.
3. `prep_vc2.sh`: the first pass's tracks gathered, the field left fitted (`fit_t1.py`), the two composed
   (`compose_fields.py`), the 15 px frames warped by the total (`v3_warp.py vc2`).
4. `t1_runs.py final --tag m15vc2 --pre data/T1pre_s4_m15vc2 --tol2d 1.0` and `t1_runs.py control --tag m15vc2 ...`:
   this folder's run and its control.
5. `v3_t1checks.py`, `v3_compare.py`, `v3_positions_t1.py`: the checks; `v3_to_x.py`: this folder.
"""


if __name__ == "__main__":
    main()
