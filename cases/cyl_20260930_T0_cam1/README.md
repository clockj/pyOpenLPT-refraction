# Case: cam1 through the water-filled tube, run 20260930 / T0

Camera calibration of one Phantom VEO710 camera (180 mm macro lens) looking through a water-filled acrylic tube
(OD 7.9375 in, wall 0.125 in) at a dot board (1.5 mm dots, 3 mm pitch). The board was traversed in 20 planes, 0.1 in
(2.54 mm) apart; the files are labelled `z_0mm` ... `z_38mm`. The model is pyOpenLPT's cylinder-port camera
(PINCYL).

- The current calibration is the redo with the corrected inputs: plan `2026-10-01-plan-redo.md`, report
  `2026-10-01-report.md`.
- The first calibration assumed 2 mm steps and a 2 mm dot size. Its plan is `2026-09-30-plan.md` and its report
  `2026-09-30-report.md` (superseded). Its test results are in `tests/v1_2mm_steps/`.
- **A check of f without the tube (2026-10-05), in `focal/`.**
  - **Data:** 15 images of the board in air, at different tilts, in `X:\Shijie Zhong\Refraction\Raw\20260930\T0\focal`.
  - **Result:** f = 41,610 px from these images, 0.4 % from the cylinder calibration's 41,459 px. A full analysis was
    planned (`focal/2026-10-05-plan.md`) but not built, at your request.
  - **Explainer page** (why a 180 mm lens gives f of about 41,600 px), for your labmate:
    `X:\Shijie Zhong\Refraction\Processed\20260930\T0\focal\focal_length_explained.html`, written by
    `focal/make_f_explainer.py`.
- Raw images: `X:\Shijie Zhong\Refraction\Raw\20260930\T0\cam1`.
- Outputs: `X:\Shijie Zhong\Refraction\Processed\20260930\T0\cam1` (its own `README.md` lists them). The first
  run's outputs are in its `superseded_v1_2mm_steps\` folder.

## Running

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder:

    python run_calibration.py                    # detect, index, fit (model M1bx), write every output (~2 min)
    python run_calibration.py --model M1b --no-outputs --state S.pkl   # another model, fit only, state saved
    python run_calibration.py --outputs-only     # rewrite the outputs from the saved fit
    python tests/t1_detection_indexing.py        # each test writes its results beside itself (tests/*_out.txt)

`config.json` holds every number of the case: paths, the port, the indices, the pitch and dot size, the plane labels
and positions, the setup distances from the photo, and the detector settings.

## Code

| file | does |
|---|---|
| `case.py` | Reads the config, lists the planes (file label and position), reads images, imports the local pyOpenLPT build. |
| `dots.py` | Finds the dark dots. Background normalisation, then the sub-pixel centre as an ellipse fitted to edge points at half the local edge height, plus three check estimators. |
| `grid_index.py` | Gives each dot its (i, j), tracked plane to plane so that the column entering on the right never shifts the indices. |
| `cylfit.py` | The model: a rig maps to a PINCYL camera. It has the parameter blocks: pose, f, axis, plane offsets, stage scale, traverse tilt, k1, plane shifts (stage play), board corrections, and a diagnostic wall field. It also has the forward and back-projection residuals, the least squares and the covariance. A numpy copy of the projection is used inside the fits for speed; tests/t0 checks it against the C++. |
| `run_calibration.py` | The pipeline and the output writer. |
| `figures.py`, `viz.py` | Diagnostic figures (including the dot-spacing check of f) and overlays. |
| `viz3d.py` | The interactive 3-D view of the fitted rig (`figures/rig_3d.html`), with a self-check of its rays against the model. `figures.py` calls it; `python viz3d.py` rewrites it alone. |
| `tests/` | **Tests:** T0, T0b (fast projection, Jacobian), T1, T1b, T2 (detection, indexing), T3 (synthetic recovery), T4 (held-out planes), T5 (model comparison and choice), T6 (visibility, reload, centre bias), T7 (error budget). **Diagnoses:** D1-D6 (first run: residual structure, board pattern, mechanism, sensor columns, lateral profile), D7 (spread of near-equal fits). |
