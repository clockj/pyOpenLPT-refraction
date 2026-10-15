# Case: wand calibration of run 20261013/T0 (three cameras, the same tube)

The wand recording of 13 October 2026 through the water-filled acrylic tube (the same size as 6–8 Oct), with 8 Oct's
three cameras:
- **cam1, cam2:** Phantom VEO 710L, 1280 × 800 px, 20 µm pixels;
- **cam3:** Phantom v2640, 2048 × 1952 px, 13.5 µm pixels.

All three saved images 0–12,000 at 100 frames/s; no camera starts late this time. T1–T20 of the same day are other
recordings with these cameras.

- **Plan:** `2026-10-15-plan.md` (approved 15 Oct: "go ahead").
- **Report:** `2026-10-15-report.md`. Its section 3 gives the settings and how to choose them.
- **Raw images:** `X:\Shijie Zhong\Refraction\Raw\20261013\T0`.
- **Results:** `X:\Shijie Zhong\Refraction\Processed\20261013\T0`, with a `README.md`. Use `camFile_track\` for
  tracking.
- **In one line:** 0.980 px over 9,139 frames. The cameras agree to 1.6–2.7 px (8 Oct: 1.1–2.2). They moved since
  8 Oct. f is undetermined, as before.

## Running it

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder. Every script
writes `<name>_out.txt` (and `.json`, figures) beside itself. The full order is in the report's section 9.

    python tests/p0a_files.py                 # Phase 0 (read-only): files and headers
    python tests/p0d_wand_survey.py scanfine  # the rod opening per camera
    python tests/p0d_wand_survey.py survey 50 # every 50th image through the detector
    python tests/p0e_wand_views.py            # the settings drawn on the images
    python detect.py                          # C1: every image (24 min over the share)
    python calibrate.py                       # C1: selection, fit, outputs and detection record on X: (14 min)
    python tests/c1t_variants.py              # the halves, quarters and camera pairs (35 min)

| code | does |
|---|---|
| `config.json` | every number of the case: paths, cameras, tube, wand, the per-camera detector settings (with how they were found), the selection, the calibration, the tracking camFiles |
| `case.py` | cases/cyl_20261008's, unchanged |
| `detect.py` | cases/cyl_20261008's, unchanged |
| `calibrate.py` | cases/cyl_20261008's, plus the detection record (the GUI's `wand_points_detection.json`, and the Job's `detection` for the results page) |
| `tests/p0*.py` | Phase 0: files, the opening scans, the survey, the figures |
| `tests/c1*.py`, `tests/c4_track_camfiles.py` | 8 Oct's checks (C4 counts images on each camera's own sensor; C1-H loads C4's files directly) |
| `tests/c1t_f_over_d.py` | f and f / distance over the parts |
| `tests/d1_residual_causes.py`, `tests/v1_variants.py`, `tests/x1_cross_run.py` | the diagnosis: where the residuals come from; variant fits; 8 Oct's cameras on today's frames |
| `tests/v2_all_three.py`, `tests/v2b_wand_length_same_frames.py` | your follow-up: the fit on the frames all three cameras keep, compared on the same frames (report section 9) |
| `tests/d2_focus_or_path.py` | your follow-up: is the larger error focus or the wand's path? Both runs' detections by stretch, distance from the axis and depth (report section 10) |
