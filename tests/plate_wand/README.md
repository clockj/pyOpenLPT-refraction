# Flat-window wand calibration (PINPLATE): summary

Last updated: 2026-10-02. Start here. The plan is `2026-10-02-plate-wand-plan.md`, the working design and every
deviation from it `2026-10-02-plate-wand-implementation.md` (sections 9–10), and the full account of the results
`2026-10-02-plate-wand-report.md`.

**In short.**
- A new calibration of cameras behind flat windows from wand data alone: no focal length, camera pose, window position
  or window orientation is needed, only the plates' thicknesses and indices and which camera looks through which
  window. Code: `modules/camera_calibration/curved_calibration/plate_model.py`, `plate_wand.py`, `plate_wand_io.py`.
- **Correct and efficient:** the model matches the C++ PINPLATE camera to 1e-10 px; from the truth the fit reaches the
  Cramér–Rao bound (errors 0.97–0.99 of it) and its error bars are honest (z standard deviation 0.97–1.00).
- **From scratch:** every one-window rig reached the global minimum (37/37 in W5), in a median 35 s for 4 cameras and
  1,000 frames. Rigs with several windows: 11/13 with camera groups and 1/3 with one camera per window at 0.1 px.
- **Against the GUI's calibration (W6, 18 cases):** 63x faster (median; 38 s against 29 min).
  - It solved the square-on rig on which the GUI failed twice.
  - The same accuracy on the reference rigs, although the GUI was given the true focal lengths.
  - On random rigs a 3-D shape error of 1.5 against 59 µm (medians).
- **Not solved:** two-camera configurations with long lenses (a window seen by one or two narrow-view cameras):
  4 of 60 W5 rigs end in a wrong minimum and 1 in a minimum 3.9 sigma from the global one.
- Synthetic data only; the GUI integration is a separate step for your approval.

## 1. Progress log

**2026-10-02** (all phases of the plan, in one day)
- Phase 1: the `cyl_model.py` rotation fix (two lines), the cylinder pytest subset rerun (7 passed).
- Phases 2–5: the model, the solver, the start, the files, and tests W0–W9 (section 2).
- Phase 6–7: W6 against the GUI, the Monte Carlo, bad data, wrong constants.
- Phase 8: `tests/test_plate_wand.py`, the report.
- **Figures page:** `figures/plate_wand_results.html`, also at
  https://claude.ai/artifact/ALBPbfKJKbMddmNBp7m2iV.
- Found on the way and fixed in the new code: the planned stage-2 distance search, the camera sliding onto its window,
  the scoring of windows with few cameras, two-camera noise weights, the re-admission of an outlier's frame
  (implementation doc, section 10). The last two also affect `cylinder_wand.refine` (logged in
  `tests/cylinder_wand/README.md`, open item 7; the cylinder code is unchanged).

## 2. Results

| test | what | result |
|---|---|---|
| W0 | SymPy checks of the projection's derivatives (`w0_sympy_derivations.py`) | 77 PROVED, 2 SKIPPED (limits argued, and checked numerically in W1) |
| W0 | the model against the C++ (`w0_model_vs_cpp.py`), 2,000 random cameras, windows (1–2 plates) and points | projections 6.5e-11 px; rays 1.3e-13; with distortion the C++'s own undistortion limits to 1e-5 px; the C++ projects a point on the camera side of a window as its mirror image (recorded, not changed) |
| W0 | the log map (`w0_logmap.py`) | round trip 1.7e-15 rad at every angle (cv2.Rodrigues: 2.0e-5 rad) |
| W1 | derivatives against central differences (`w1_jacobians.py`) | every column within 1.3e-7 (model) and 3e-7 (bundle adjustment) |
| W2 | Cramér–Rao on rigs A–E (`w2_observability.py`) | f to 0.06 % (A, B, D), 0.2 % (C, square-on), 0.6 % (E, macro); window normal 0.015–0.14 deg; 3-D shape error of unseen points 0.6–26 um |
| W3 | noise-free from the truth (`w3_noise_free.py`) | RMS 1e-12 px, every parameter at round-off, also from a perturbed start |
| W3 | Monte Carlo, 40 sets per rig at 0.1 px, and a 0.02–2 px sweep (`w3_monte_carlo.py`) | errors / CRB 0.97–0.99; z SD 0.97–1.00; no bias beyond 2.4 SE; linear in the noise |
| W4 | basin of the final fit (`w4_basin.py`) | recovers the window normal 30 deg off (45 deg except the steep rig), any offset, f +-40 % (macro: +-10–20 %) |
| W5 | from scratch, 50 random rigs at 0.1 px, 10 at 1 px (`w5_from_scratch.py`) | 46/50 and 5/10 reach the truth-started fit's minimum (9/10 an equivalent one); one-window rigs 37/37; median 35 s for 4 cameras |
| W6 | against the GUI's calibration (`w6_vs_gui.py`) | see section 3 |
| W7 | bad data (`w7_bad_data.py`) | outliers and swapped ends all rejected, 0–6 good detections lost; a 5x noisier camera's noise estimated (0.503 px); no frame seen by every camera handled; every answer at the true minimum of its data |
| W8 | wrong constants (`w8_constants.py`) | thickness 10 % and water index 0.002 off: invisible and nearly harmless; wand 0.1 % long: a 0.1 % scale error; an unmodelled k1 0.05: RMS +0.3 %, 3-D error x2 (fitting k1 recovers it); the wand length is not identifiable through flat windows (+-5–18 mm) |
| W9 | the files (`w9_files.py`) | camFiles reload in the C++ and project within 7e-12 px; the CSV round trip exact; the GUI's own loader reads the CSV |
| W10 | real data | not run: no data set |

## 3. Against the GUI's calibration (W6)

The GUI's current refractive calibration (`RefractiveWandCalibrator.calibrate`, run headless as `view.py` runs it,
with the true focal lengths in its camera table) and the new one, on the same data: rigs A–D at 0.1 and 0.5 px and
ten W5 rigs, 1,000 frames each, every frame seen by every camera. Table: `w6_vs_gui_out.txt`.

- **Time:** 7–112x faster (median 63x): the GUI 685–4,222 s (median 29 min),
  the new calibration 21–153 s (median 38 s).
- **Failures:** the GUI failed both square-on cases (rig C: a collapsed scale; no valid camera pair); the new one none.
- **Accuracy:**
  - on rigs A, B, D the two are comparable (3-D shape error medians 6.9 and 6.2 µm, near the Cramér–Rao bound);
  - on the ten random rigs the new one is far better: 1.5 against 59 µm, f errors 0.12 % against 2.7 %, unseen points
    reprojected within 0.012 against 0.54 px (medians);
  - the GUI is better on 1 of 10.

  The GUI had the true focal lengths, the new one none.
- The per-case table is in the report, section 3.6.

## 4. How to run

```python
from plate_wand import Rig, WandData, calibrate        # modules/camera_calibration/curved_calibration on sys.path
from plate_model import Window
import plate_wand_io as pio
cd = pio.read_gui_csv("wand.csv")                       # the GUI's "Export wand data" CSV
rig = Rig([Window([0, 0, 1], 0.0, (12.7,), (1.0, 1.49, 1.333))], cam_win=[0, 0, 0, 0])  # n, d: placeholders
sol = calibrate(WandData(cd.uv, 25.0, [(1280, 800)] * 4), rig)
pio.write_result(sol, "out")                            # out/camFile/cam1.txt ..., calibration_result.json, report
```
Or `pio.calibrate_csv(csv, config_json, out_folder)` with a rig configuration file (`plate_wand_io.RigConfig`).

Tests: `pytest tests/test_plate_wand.py` (6 fast checks, and a calibration from scratch marked slow; all pass; the
full suite: 243 passed, 7 skipped, 2 failed, both known and unrelated). Each W script
runs on its own from the repository root with the `refract` interpreter and writes `<name>_out.txt` and `.json` here;
`PLATE_WORKERS` sets the number of processes of the parallel ones.

## 5. Open items

1. **Two-camera configurations with long lenses** (W5 seeds 1001, 1031, 1034, 1045, 2000): the start lands in a wrong
   minimum. A hint (an approximate window normal or camera distance), a third camera, or a different start would be
   needed; report section 5.
2. **GUI integration** (a separate step for your approval): `view.py`'s refractive path calling `plate_wand.calibrate`
   and writing the camFiles where the GUI expects them, the current path kept as an option.
3. **Real data (W10):** the GUI's wand CSV, the window settings, the wand length and the current calibration's
   camFiles of one experiment.
4. **The cylinder's `refine`:** port the two fixes (`tests/cylinder_wand/README.md`, open item 7).

## 6. Files

- Plans and report: `2026-10-02-plate-wand-plan.md`, `2026-10-02-plate-wand-implementation.md`,
  `2026-10-02-plate-wand-report.md`.
- Pre-plan sizing: `p_sympy_presize.py`, `p_pre_sizing.py`, `p_logmap_check.py`.
- Helpers: `prigs.py` (rigs A–E, random rigs, wand data from the C++, alignment, metrics).
- Tests W0–W9 and their outputs (`w*_out.txt`, `w*.json`); `w1_block2.py` (W1's bundle-adjustment block);
  `w6_run.py` (one W6 calibration as its own process); `w6_cases/` (W6's data and both calibrations' working files,
  git-ignored: `w6_vs_gui.py prepare` and `run` rebuild it).
- Diagnoses: `d1_multiwindow.py`, `d2_groups.py`, `d3_single.py` and their outputs.
- Earlier runs kept for the record: `runs_v1/` (W5 run 1, W7 before the re-admission fix), `runs_v2/`, `runs_v3/`
  (W5 runs 2 and 3).
- `w9_result/`: an example of the written result (camFiles, JSON, report).
- `figures/plate_wand_results.html` (open it in a browser), also at
  https://claude.ai/artifact/ALBPbfKJKbMddmNBp7m2iV.
  - `make_figures.py` rebuilds it from the saved results; nothing is re-fitted.
  - It fills `figures/template.html` and writes the data to `figures/figure_data.json`.
  - `--fragment PATH` also writes the page without its document skeleton, for republishing.
