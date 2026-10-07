# Case: wand calibration of two cameras through the tube, run 20261006 / T0

Two Phantom VEO 710L cameras (cam1, cam2; 1280 × 800 px, 20 µm pixels, 180 mm macro lenses) look through the
water-filled acrylic tube (OD 7.9375 in, wall 0.125 in) at a wand hanging on a string: two dark balls of 3 mm and
4 mm, 10 mm apart, on a thick rod, backlit. 10,640 frames per camera. The model is pyOpenLPT's cylinder-port camera
(PINCYL), calibrated with `cylinder_wand` exactly as the GUI's "Pinhole+Cylinder" runs it. This is the first real
data set for that calibration.

- Plan: `2026-10-06-plan.md` (approved 6 Oct). Report: `2026-10-06-report.md`.
- **Result (6 Oct):** 1,519 usable frames; the calibration fits with 0.45 px RMS and 22 µm wand-length scatter, and
  is usable inside the wand's volume (0.1-0.2 % in scale), but this recording does not determine f: its halves and
  quarters give cam1 f from 32,800 to 40,200 px (T3). The 2 mm rod limits the ball centres (T1 b). Report sections 1
  and 6 (what to change for the next recording).
- **Redelivered (6 Oct, night)** with the stage-4 fix (`tests/cylinder_wand_gui/2026-10-06-fixes-report.md`):
  - **The change:** 6,064 of 6,076 detections instead of 5,720; RMS 0.47 px, 23 µm; f −0.33 % / −0.58 %, under half
    a standard error.
  - **Better?** Every independent check agrees with the first delivery within its noise (report, top).
  - **Where things are:** every test rerun; the first delivery's outputs in `before_stage4_fix\` here and in
    `Results\before_stage4_fix\`.
- **Variants rerun (7 Oct)** with the four-round flip fix (fixes report, section 9):
  - **What changed:** h2 and fit_k1, which had still looped, and D2's calibration; the rest only in the last digits.
  - **The delivered camFiles:** unchanged.
  - **Where things are:** the earlier outputs in `before_flip4\` here and in `Results\before_flip4\`; the page is
    republished (version 5).
- Raw images: `X:\Shijie Zhong\Refraction\Raw\20261006\T0\cam1`, `cam2`.
- Outputs: `X:\Shijie Zhong\Refraction\Processed\20261006\T0\Results` (its own `README.md` lists them).

## Running

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder:

    python detect.py                 # S1: the balls in all frames of both cameras (~19 min over the network share)
    python calibrate.py              # S2: selection, wand_points.csv, the calibration and the GUI's output files
    python tests/t0_gui_csv.py       # wand_points.csv loads in the GUI
    python tests/t1_detection.py     # T1: overlays, rod settings, A/B tracking, rod width and blur
    python tests/t2_residuals.py     # T2: residual maps and wand-length trends
    python tests/t3_t6_variants.py   # T3-T6: halves and quarters, wand length free, k1 free, thinned frames
    python tests/t7_vs_20260930.py   # T7: cam1 against the dot-board calibration of 30 Sep
    python tests/p0_survey.py        # P0 (before the plan): usable frames, the opening window
    python tests/d1_sync.py          # D1-D8: the diagnoses after T1-T3 failed (report section 3)
    python tests/d2_shape_prior.py   #   centres with the predicted image shape (~15 min, writes Results\checks)
    python tests/d3_jitter.py        #   frame-to-frame noise
    python tests/d4_residual_vs_pose.py
    python tests/d5_jitter_cameras.py
    python tests/d6_length_vs_direction.py
    python tests/d7_length_autocorr.py
    python tests/d8_held_out.py      #   the variants' cameras on the frames they were not fitted on

Set `CASE_PYOPENLPT_DIR` to a folder holding a copy of `build-py/Release/pyopenlpt*.pyd` to import that copy
instead (so that pyOpenLPT can be rebuilt while the case runs).

`config.json` holds every number of the case: paths, cameras, image size, the wand, the tube, the detector's and
the acceptance's settings, the calibration's options and the 30 Sep reference.

## Code

| file | does |
|---|---|
| `case.py` | Reads the config; imports the library (unchanged); reads images; runs the GUI's "ellipse" detector; reads and writes the detection table; keeps or rejects camera-frames (`classify`); builds the calibration's input and the GUI's CSV; runs `cylinder_wand_gui.run` and `write_outputs`. |
| `detect.py` | S1, in parallel processes. |
| `calibrate.py` | S2, the delivered result. |
| `tests/` | P0, T0-T7, each writing `<name>_out.txt` (and `.json`, figures) beside itself. |
| `make_page.py`, `page_template.html` | The results page, from the saved outputs only: `python make_page.py [--fragment PATH]` writes `<out_dir>\wand_calibration_20261006_T0.html` (standalone) and, with `--fragment`, the copy without its document wrapper that is published as the artifact https://claude.ai/artifact/BSQS8D25a9Fypt3wWoP6vB (republish the fragment, not the standalone file). Since the evening of 6 Oct (your request, during run T2) it also shows each camera's 2-D reprojection-error distribution (`err2d`, the same chart as the T2 page). |
