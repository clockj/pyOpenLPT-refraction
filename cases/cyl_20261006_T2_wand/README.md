# Case: wand calibration of two cameras through the tube, run 20261006 / T2

The second wand recording of 6 October 2026, about an hour after T0 (`cases/cyl_20261006_T0_wand`): the same two
Phantom VEO 710L cameras (cam1, cam2; 1280 × 800 px, 20 µm pixels, 180 mm macro lenses) looking through the
water-filled acrylic tube (OD 7.9375 in, wall 0.125 in) at the same wand, two dark balls of 3 mm and 4 mm, 10 mm
apart, on a 2 mm rod, backlit. 12,716 frames per camera. The model is pyOpenLPT's cylinder-port camera (PINCYL),
calibrated with `cylinder_wand` exactly as the GUI's "Pinhole+Cylinder" runs it.

- Plan: `2026-10-06-plan.md` (approved 6 Oct). Report: `2026-10-06-report.md`.
- **Result (6 Oct):**
  - **The calibration:** 1,966 usable frames; it fits with 0.60 px RMS and 31 µm wand-length scatter.
  - **Usable inside the wand's volume:** held-out wand lengths within 0.06 % (D8).
  - **The cameras moved by about 1 px between T0 and T2 (X1):** use each run's own camFiles.
  - **f is still not determined (T3):** cam1 f 36,500–40,900 px over the parts, along one f / distance line on
    which the 30 Sep dot board also lies.
  - **k1 does not help (T5):** −3.5 % RMS, implausible values.
  - **Where to read:** report sections 1 and 6.
- **T3, T5 and D8 rerun (7 Oct):**
  - **Why:** they had run on the library before the stage-4 fixes, and every start had looped (the fixes report,
    `tests/cylinder_wand_gui/2026-10-06-fixes-report.md`, section 9).
  - **What changed:** the 2nd quarter no longer puts cam2 at 4.7 m.
  - **Where things are:** the earlier outputs in `before_flip4\` here and in `Results\before_flip4\`; the page is
    republished (version 3).
- Raw images: `X:\Shijie Zhong\Refraction\Raw\20261006\T2\cam1`, `cam2`.
- Outputs: `X:\Shijie Zhong\Refraction\Processed\20261006\T2\Results` (its own `README.md` lists them).

## Running

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder:

    python tests/p0_survey.py          # P0 (before the plan): the headers, the light, a survey of every 50th frame
    python tests/p0_t0_cameras.py      #   T0's calibration on the surveyed frames: the cameras moved
    python detect.py                   # S1: the balls in all frames of both cameras (~22 min over the network share)
    python calibrate.py                # S2: selection, wand_points.csv, the calibration and the GUI's output files
    python tests/t0_gui_csv.py         # T0: wand_points.csv loads in the GUI
    python tests/t1_detection.py       # T1: overlays (a third before the light changed); A and B never swap
    python tests/t2_residuals.py       # T2: residual maps and wand-length trends
    python tests/t3_variants.py        # T3: the calibration on each half and quarter of the recording
    python tests/t5_k1.py              # T5 (added at your question): the calibration with k1 free
    python tests/x1_cross_runs.py      # X1: T0's and T2's calibrations on each other's frames
    python tests/d1_sync.py            # D1: the cameras' time offset
    python tests/d8_held_out.py        # D8: T3's cameras on the frames they were not fitted on (after T3)
    python tests/t7_references.py      # T7: against the 30 Sep dot board (cam1) and T0's cameras
    python make_page.py                # the overview page (after all the tests)

Set `CASE_PYOPENLPT_DIR` to a folder holding a copy of `build-py/Release/pyopenlpt*.pyd` to import that copy
instead (so that pyOpenLPT can be rebuilt while the case runs).

`config.json` holds every number of the case: paths, cameras, image size, the wand, the tube, the detector's and
the acceptance's settings (T0's, unchanged), the calibration's options, and the two references (T0's results and the
30 Sep dot board).

## Code

The code is T0's case code, copied: `case.py` is unchanged but for its docstring, and the scripts differ only in
paths, T2's frame ranges and texts. The P0 scripts use T0's `case.py` itself.

| file | does |
|---|---|
| `case.py` | Reads the config; imports the library (unchanged); reads images; runs the GUI's "ellipse" detector; reads and writes the detection table; keeps or rejects camera-frames (`classify`); builds the calibration's input and the GUI's CSV; runs `cylinder_wand_gui.run` and `write_outputs`. |
| `detect.py` | S1, in parallel processes. |
| `calibrate.py` | S2, the delivered result. |
| `tests/` | P0, T0-T3, T5, T7, X1, D1, D8, each writing `<name>_out.txt` (and `.json`, figures) beside itself. |
| `make_page.py`, `page_template.html` | The overview page, from the saved outputs only (T2's and T0's): `python make_page.py [--fragment PATH]` writes `<out_dir>\wand_calibration_20261006_T2.html` (standalone) and, with `--fragment`, the copy without its document wrapper that is published as the artifact https://claude.ai/artifact/Mu8jHCGGPtcVAZ4F36MNhc (republish the fragment). `page_template.html` was assembled from T0's styles and chart helpers plus T2's sections and charts. |
