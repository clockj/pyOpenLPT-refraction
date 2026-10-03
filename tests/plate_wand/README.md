# Flat-window wand calibration (PINPLATE): summary and progress

Last updated: 2026-10-02.

Start here to catch up. This file gathers what the flat-window wand tests concluded, and what that means for a real
experiment.
- It is kept up to date: each new result on this calibration adds a dated entry to the progress log (section 1,
  newest first) and revises the sections it changes.
- The numbers come from the reports listed in section 7, which name the scripts and data behind them.
- The figures page draws them: `figures/plate_wand_results.html`, also at
  https://claude.ai/artifact/ALBPbfKJKbMddmNBp7m2iV.

**In short.**
- On synthetic data the calibration works for cameras behind one window:
  - it finds the right cameras and window from scratch, with no camera input, focal lengths included (37 of 37
    random rigs);
  - it is as precise as the data allow (errors 0.97–0.99 of the Cramér–Rao bound);
  - its error bars are honest.
- Rigs with several windows work when two or more cameras share a window (11 of 13), less often with one camera per
  window (1 of 3). The unsolved case is a window seen by only one or two long-lens cameras: 5 of the 60 random rigs end
  in a different minimum.
- Against the GUI's current calibration on the same data:
  - 63× faster (median 38 s against 29 min);
  - no failures, where the GUI failed twice;
  - a 3-D error of 1.5 µm against 59 µm on random rigs (medians), although only the GUI was given the true focal
    lengths.
- Two differences from the tube:
  - **the wand length is the only scale.** Fitting it gives no check of the constants, so it must be measured
    independently;
  - **the window constants hardly matter.** A plate 10 % too thick or a water index 0.002 off leave the answer almost
    unchanged.
- No real wand data have been tried yet, and the GUI does not call this calibration yet.

## 1. Progress log

**2026-10-02**
- **Why W5 seed 2000 misses although each window has two cameras** (`d4_seed2000.py`, your question).
  - **The rig is sound.** Each window's two cameras are 21° and 30° apart (in the water), and the walls are
    perpendicular. At 1 px noise the right answer is itself loose: σ(f) 0.5–1.2 %, distances 3–20 mm, normals 0.4°.
  - **The start lands in a small wrong basin.** Each window's pair is first calibrated on its own, then the two are
    joined. Alone, a pair is 1.2–2.5× less precise than the full rig, which sees the same wand through both walls.
    - Each pair landed within 0.2–2.0 of its own σ, as well as two cameras can do at 1 px.
    - Joined, that is 2–3.4 full-rig σ off in several parameters at once.
    - Stage 4 then settles in a nearby minimum: RMS +0.3 % (50 px² more in squared residuals), cameras 3 and 4's f
      off by +2.0 % and −2.1 % (3.8σ, 2.8σ).
  - **That basin is narrow.** Stage 4 restarted a quarter of the way (or more) from that answer towards the global
    minimum reaches the global minimum.
  - **The whole-rig start that solved it in W5 run 1 no longer does** (it was changed since, because it failed on 4 of
    9 multi-window rigs): 1.34 px and 18–20 px.
  - **It is the only such miss.** Every rig whose windows all have two or more cameras reached the global minimum
    at 0.1 px (8 of 8); at 1 px, 2 of 4 did, plus one equivalent answer.
  - A possible remedy, untested: a few stage-4 starts perturbed around the joined start (open item 1).
- **A camera looking straight through a window costs nothing; cameras close together do** (`p_mixed_views.py`, your
  question: one camera square-on and one at an angle through the same window).
  - Cramér–Rao at 0.1 px, 1,000 frames, on rigs built like rig A.
  - **With one square-on and one angled camera, the square-on camera is as precise as the angled one,** or slightly
    more: σ(f) 0.145 % against 0.162 %, its distance to the window 3.5 mm against 4.0 mm (0° + 20°).
    - The rig matches a pair at −10°/+10°, the same 20° apart: σ(f) 0.151 %, 3-D shape 13.6 µm against 13.5 µm.
    - All three mixed rigs calibrated from scratch reached the global minimum (11–15 s).
  - **Correction: rig C's weakness comes from its cameras being 5° apart, not from its view straight through the
    wall.** The same layouts moved to 18–25° incidence are as weak:

    | layout | 0–5° incidence: σ(f), σ(distance), 3-D shape | 18–25° incidence: σ(f), σ(distance), 3-D shape |
    |---|---|---|
    | two cameras 5° apart | 0.31 %, 10 mm, 47 µm | 0.35 %, 7 mm, 46 µm |
    | rig C, three cameras 5° apart | 0.20 %, 4.4 mm, 26 µm | 0.19 %, 3.7 mm, 25 µm |

  - Sections 3 and 4, the report (section 3.2) and the figures page said rig C was weak because it is square-on, and
    advised keeping cameras 15° or more off the normal. They are corrected: what matters is the angle between the
    cameras' lines of sight (section 3, item 6).
- **This summary rewritten as a catch-up report,** in the format of the cylinder's (`tests/cylinder_wand/README.md`).
- **Figures page:** `figures/plate_wand_results.html`, also at https://claude.ai/artifact/ALBPbfKJKbMddmNBp7m2iV.
  - `make_figures.py` rebuilds it from the saved results.
  - Checking its captions against the results corrected two statements of the report (its section 3.2):
    - the scene's position relative to camera 1: 0.43–0.48 mm on rigs A, B and D, 1.6 mm on C and 7.3 mm on E;
    - how much weaker the square-on rig is: 3–9×, not about 5×.
- **The calibration was built and tested on synthetic data (W0–W9),** all phases of the approved plan.
  - **The plan's from-scratch target was not met:** 46/50 and 5/10, against ≥ 49/50 and 10/10 (section 3, item 2).
  - **Found on the way and fixed in the new code** (implementation doc, section 10):
    - the planned stage-2 distance search: the affine points bias it about 10× too far, so the distance now comes
      from perspective;
    - cameras sliding onto their window during resection: their distance is now held, and they are placed on their
      exact chief rays;
    - windows with one or two cameras, which a fit to fixed points cannot judge: now also scored by the wand-length
      residual;
    - per-camera noise weights drifting with two cameras;
    - a gross outlier costing its whole frame on re-admission.
  - The last two also affect `cylinder_wand.refine`. They are logged in the cylinder summary as open item 7; the
    cylinder code is unchanged.
  - The approved two-line rotation fix in `cyl_model.py` (`log_so3`, `rvec_of`).
  - In pyOpenLPT commit `5391cb3` (branch `Refraction_new`).

## 2. Status

- **What it does:** it calibrates cameras that look through one or more flat windows (the walls of a tank), from
  wand frames alone.
  - It estimates every camera's pose and f, every window's normal and position, each camera's noise, and the error
    bars of all of them.
  - It needs no focal length, camera pose, window position or window orientation. It needs only:
    - the detections and the wand length;
    - the image sizes;
    - each window's plate thicknesses and indices;
    - which camera looks through which window.
  - It writes pyOpenLPT PINPLATE camFiles, which the C++ reloads and projects within 7e-12 px (W9), a result JSON and a
    text report.
- **How** (`calibrate`, stages 1–5):
  1. an affine reconstruction of the wand points, made metric by the wand length;
  2. a pinhole start:
     - each camera's rotation from its affine camera;
     - its distance from perspective, since a wand's image shrinks with depth;
     - then a pinhole bundle adjustment;
  3. the windows:
     - **candidates:** normals on a 50° cap at 9° spacing, and three offsets;
     - **for each candidate:** the cameras are placed on their exact chief rays and resected, and the best candidates
       are refined together with the window;
     - **a window with one or two cameras** is also scored by the wand-length residual;
     - **several windows:** each group of cameras sharing a window is calibrated on its own, the groups are joined,
       and single cameras are placed;
     - **then a coarse fit;**
  4. the maximum-likelihood bundle adjustment, with:
     - the wand frames eliminated and the wand length exact;
     - outlier rejection and re-admission;
     - per-camera noise weights (equal with two cameras);
     - several starts when they disagree;
  5. the covariance, the output frame and the files.
- **Tested on:** synthetic data only.
  - The data come from pyOpenLPT's C++ PINPLATE camera, through air, acrylic (n 1.49) and water (n 1.333). The walls
    are 12.7 mm thick on the reference rigs and 6–25.4 mm on the random ones.
  - **Rigs:** the five reference rigs A–E (section 3, item 4) and 60 random rigs, with 2–6 cameras and 1–3 walls of a
    box tank.
  - **Wands:** 25 mm long (10 mm on rig E), and 0.2–0.35 of the cloud's size on the random rigs.
- **Not done yet:** real data (W10) and the GUI integration.
- **Code,** in `modules/camera_calibration/curved_calibration/`:
  - `plate_model.py`: the PINPLATE camera, vectorised, with exact derivatives. It is checked against the C++ (W0),
    and its derivatives against finite differences (W1);
  - `plate_wand.py`: the calibration (`calibrate`). It imports the model-free pieces of `cylinder_wand.py`;
  - `plate_wand_io.py`: the GUI's wand CSV, a rig configuration file, camFiles, the result JSON and the report.
- **Tests:**
  - `tests/test_plate_wand.py`: 6 fast checks and one calibration from scratch (marked slow); all pass.
  - The campaign is in this folder.
  - pyOpenLPT's full suite: 243 passed, 7 skipped, 2 failed, both known and unrelated.
- **Run time** (from scratch, 1,000 frames):
  - **one window:** a median 35 s for 4 cameras (20–79 s for 2–6 cameras), with the machine fully loaded. In one
    quiet process, rig A takes about 26 s;
  - **several windows:** a median 121 s (up to 354 s). Each window group is calibrated on its own, and two searches
    are pooled.
- **Git:** pyOpenLPT commit `5391cb3` (branch `Refraction_new`).

Use:

```python
import plate_wand as pw, plate_wand_io as pio, plate_model as pm   # modules/camera_calibration/curved_calibration
cd = pio.read_gui_csv("wand.csv")                     # the GUI's "Export wand data" CSV
rig = pw.Rig([pm.Window([0, 0, 1], 0.0, (12.7,), (1.0, 1.49, 1.333))],   # plates (mm); indices camera side first;
             cam_win=[0, 0, 0, 0])                    #   normal and offset are placeholders: calibrate finds them
sol = pw.calibrate(pw.WandData(cd.uv, 25.0, [(1280, 800)] * 4), rig)    # options: fit_k1=True; hints={"f": [...]}
pio.write_result(sol, "out")                          # out/camFile/cam1.txt ..., calibration_result.json, report
```

Or, with a rig configuration file, `pio.calibrate_csv("wand.csv", "rig.json", "out")`. The file has one camera entry
per camera:

```
{"windows": [{"name": "front", "thickness_mm": [12.7], "n": [1.0, 1.49, 1.333]}],
 "cam_to_window": [0, 0, 0, 0],
 "cameras": [{"name": "cam1", "width": 1280, "height": 800, "pixel_mm": 0.02}, ...],
 "wand_length_mm": 25.0}
```

The output frame:
- Z is window 1's normal, from the cameras into the water;
- Y is along n1 × n2 with two windows that are not parallel, otherwise camera 1's image "up" within window 1's plane;
- the origin is at the wand's mean position.

## 3. What the tests concluded

1. **The camera model is exact (W0, W1).**
   - It matches pyOpenLPT's C++ PINPLATE camera to 6.5e-11 px, and the rays to 1.3e-13. The comparison used 2,000
     random cameras, windows (one and two plates, incidence up to 60°) and points.
   - Its derivatives match central differences to 1.3e-7 (the bundle adjustment's to 3e-7).
   - SymPy proves 77 steps of its derivation. Two limits are SKIPPED there; W1 checks them numerically.
   - **A C++ quirk, recorded and not changed:** the C++ projects a point on the camera side of a window (in the plate
     or the air) as its mirror image in the window (600 of 600). The new model reports such a point as not projecting.
     The GUI's pipeline uses the C++ and can be misled by it.
2. **From scratch, one window always succeeds; several windows usually do** (W5: 60 random rigs, no hints).

   | rig kind | 0.1 px: same minimum | 1 px: same / equivalent | time (median) |
   |---|---|---|---|
   | one window | 34/34 | 3/3 / 3/3 | 35 s for 4 cameras |
   | several windows, two or more cameras on some | 11/13 | 2/4 / 3/4 | 121 s |
   | several windows, one camera each | 1/3 | 0/3 / 3/3 | 121 s |
   | all | **46/50** | **5/10 / 9/10** | |

   - **What the columns mean:**
     - **same minimum:** every parameter within 0.01σ of the fit started from the truth, the same detections kept and
       the same RMS;
     - **equivalent:** every parameter within 3σ and the RMS within 2 %, or a lower RMS.
   - **Every miss has a window seen by only one or two cameras.** Four of the five also have a camera with f of
     24,000–37,000 px.
     - A narrow view sees a window's tilt only weakly, so the cost has several minima.
     - Even the right minimum is then poorly determined: seed 1001's second window has σ 2.1° and 438 mm.
   - The error bars warn of it: a weakly determined window shows a large σ.
   - **Rigs whose windows all have two or more cameras** reached the global minimum at 0.1 px every time (8 of 8).
     The one miss among them is seed 2000 at 1 px: its pairs, calibrated one window at a time, start it in a narrow
     wrong basin next to the global minimum (RMS +0.3 %; `d4_seed2000.py`, progress log).
3. **It reaches the theoretical limit, and its error bars match the real scatter (W3).**
   - Rigs A–E, 40 data sets each at 0.1 px, each fitted from the truth:

     | rig | parameters | RMS error / Cramér–Rao | z standard deviation | largest bias |
     |---|---|---|---|---|
     | A | 25 | 0.98 | 0.98 | 1.7 SE |
     | B | 28 | 0.99 | 1.00 | 1.8 SE |
     | C | 18 | 0.97 | 0.98 | 2.3 SE |
     | D | 18 | 0.99 | 0.99 | 2.4 SE |
     | E | 18 | 0.97 | 0.97 | 1.3 SE |

   - Errors grow in proportion to the detection noise from 0.02 to 2 px (rig A).
   - Without noise, a fit from the truth reaches an RMS of 1e-12 px, with every parameter at round-off.
4. **What the data determine (W2, Cramér–Rao at 0.1 px, 1,000 frames).**
   - **The reference rigs:**
     - a 100 mm cube of wand positions, its centre 150 mm behind a 12.7 mm acrylic wall;
     - 1280 × 800 cameras, 600 mm from the wall along their centre rays, with f of 3,900–4,400 px;
     - rig E is the exception: a 16 mm cube 100 mm behind the wall, seen with f = 40,000 px from 1,100 mm.

   | rig | f | window normal | camera's distance to window | 3-D shape of unseen points (calibration) | 25 mm wand, calibration / with detection noise |
   |---|---|---|---|---|---|
   | A, one wall, 4 cameras at 21° | 0.063 % | 0.026° | 0.47 mm | 5.5 µm | 1.9 / 30 µm |
   | B, two walls, 2 + 2 cameras at 15° | 0.055 % | 0.018°, 0.016° | 0.95 mm | 2.4 µm | 0.9 / 16 µm |
   | C, 3 cameras 5° apart, looking straight through | 0.20 % | 0.14° | 4.4 mm | 26 µm | 8.9 / 148 µm |
   | D, steep, 35–45° | 0.059 % | 0.015° | 0.37 mm | 4.3 µm | 1.5 / 23 µm |
   | E, macro, f 40,000 px | 0.61 % | 0.016° | 6.7 mm | 0.6 µm | 0.6 / 8 µm (10 mm wand) |

   - Everything scales with the noise: 10× these values at 1 px.
   - **The calibration's share is small.** On rig A a single point triangulated from 0.1 px detections is itself
     uncertain by 37 µm (10–198 µm across the rigs). The calibration adds 5.5 µm.
   - Outside the calibrated volume the shape error grows about linearly with the distance: 2× at twice the volume.
5. **Shapes are known far better than positions relative to the cameras (W2).**
   - The scene, the windows and f can slide together along the viewing direction.
   - Relative to camera 1, the scene's position is known to:
     - 0.43–0.48 mm on rigs A, B and D;
     - 1.6 mm on C;
     - 7.3 mm on E.
   - Its shape is known to 0.6–26 µm.
   - Distances, velocities and shapes measured inside the volume carry only the shape error.
6. **Rig design (W2, W4, `p_mixed_views.py`).**
   - **The angle between the cameras' lines of sight sets the precision.** With two cameras through one window:

     | angle between the cameras | σ(f) | 3-D shape of unseen points |
     |---|---|---|
     | 5° | 0.31–0.35 % | 46–47 µm |
     | 20° | 0.15–0.16 % | 13.5 µm |
     | 35–40° | 0.09–0.11 % | 7.6–8.6 µm |

     - Rig C, three cameras 5° apart, is 3–9× weaker than rig A on every measure for this reason.
     - The GUI's calibration failed on it twice (W6), in its pinhole start.
   - **Looking straight through the window costs nothing.** Moving rig C's layout from 0–5° to 18–25° incidence
     changes σ(f) by 7 %, the cameras' distances by 17 % and the 3-D shape by 5 %; the window's normal is even 30 %
     less certain at 20°.
     - With one camera square-on and another at an angle, the square-on camera is as precise as the angled one
       (σ(f) 0.145 % against 0.162 % at 0° + 20°).
     - In principle a camera looking straight through cannot see the window's position paraxially: moving the window
       towards it and the scene away by a third as much leaves its images unchanged (checked with SymPy). In these
       rigs that barely matters, presumably because the rays towards the image's edges are oblique enough.
   - **Steep views are fine.** Rig D, at 35–45°, is as precise as A or slightly better.
   - **Two perpendicular walls give the best shape:** rig B, 2.4 µm against 5.5 µm on A.
   - **Long lenses:** f and the camera's distance trade almost freely.
     - Rig E, with f = 40,000 px, has f to 0.61 % and the distance to 6.7 mm.
     - Its shape is still the best of all: 0.6 µm on a 16 mm cloud.
   - **How close the start must be (W4).** The final fit recovers:
     - a window normal 60° off, on every rig but the steep D, which needs it within 30°;
     - any window offset between the cameras and the cloud;
     - f within ±40 % on rigs A, B and D (C: 9 of 10 at 40 %), but only within ±10 % on the macro rig E.
   - **Camera rotations are the weak spot** of that basin: A 8/10 at 5°; C 5/10 and E 1/10 at 10°. The start
     therefore takes them from the wand data, which gives them to about 1°.
7. **Against the GUI's current calibration on the same data (W6, 18 cases).**
   - **The setup:**
     - the GUI's `RefractiveWandCalibrator`, run headless as `view.py` runs it and **given the true focal lengths**;
       the new calibration was given none;
     - 1,000 frames: rigs A–D at 0.1 and 0.5 px, and ten W5 rigs at 0.1 px;
     - each judged on its exported camFiles through the C++.
   - **Time:** 7–112× faster, median 63×: 38 s against 29 min (medians).
   - **Failures:** the GUI failed both cases of rig C, whose cameras are 5° apart:
     - at 0.1 px its scale collapsed;
     - at 0.5 px it found no valid camera pair.

     The new calibration solved both, at about the precision W2 predicts.
   - **Accuracy on rigs A, B and D:** comparable. The 3-D shape error medians are 6.9 µm (GUI) and 6.2 µm (new),
     both near the Cramér–Rao bound.
   - **Accuracy on the ten random rigs:** the new calibration is far better.
     - 3-D shape error: 1.5 µm against 59 µm (medians), better on 9 of 10, by up to 400×.
     - f error: 0.12 % against 2.7 % (medians). The GUI's f is held within ±5 % of the value it was given, and two of
       its runs ended on that bound.
8. **Bad data are handled (W7: rigs A and B, from scratch).**
   - **2 % gross outliers of 20–200 px:** all 160 rejected; 0–6 good detections lost.
   - **1 % of camera-frames with the ends swapped:** all rejected; no good detection lost.
   - **A camera missing from 40 % of the frames:** the answer moved only as much as the lost data explain.
   - **A camera with 5× the noise:** its noise was estimated at 0.503 px (true 0.5 px), and it was weighted
     accordingly.
   - **No frame seen by every camera:** handled.
     - Rig A uses a camera-subset start.
     - On rig B, a window group that never shares a frame falls back to the whole-rig start.
     - Rig B's answer moved 1.11σ RMS from the clean answer, against 0.72σ expected. That is a marginal miss of the
       1.5× limit, but it is still at its data's own minimum.
9. **The window constants hardly matter; the wand length is the only scale (W8, rigs A and D).**

   | given | residual RMS | 3-D shape of unseen points | what shows it |
   |---|---|---|---|
   | correct | — | 4.3–5.4 µm | — |
   | plate 10 % thicker | unchanged | unchanged | nothing |
   | water index +0.002 | unchanged | 5.3–5.4 µm | nothing |
   | wand 0.1 % long | unchanged | 49–51 µm (a 0.1 % scale error) | only an independent length |
   | unmodelled k1 = 0.05 | +0.3–0.4 % | 7–8 µm | barely the residuals |

   - **The free wand length is not identifiable.** With it free (`fit_L=True`), the fit returns 9–91 mm ± 5–18 mm for
     a 25 mm wand.
   - Only the plate's thickness carries scale, and 12.7 mm of acrylic is a poor ruler.
   - So `fit_L` is no check of the constants here. Through the tube it is: there it detects a water index 0.002 off at
     33σ.
10. **An unmodelled lens distortion hides in the residuals (W8).**
    - A k1 of 0.05 left out of the model raises the residual RMS by only 0.3–0.4 %, but the 3-D error 1.3–1.9×.
    - With `fit_k1=True` the fit recovers it (k1 0.043–0.058 for the true 0.05), and the 3-D error returns to that of
      the correct model.
11. **The files work (W9).**
    - The camFiles reload in the C++ and project within 7e-12 px, with the camera and window read back exactly.
    - The GUI's wand CSV round-trips exactly, and the GUI's own loader reads the CSV written here.

## 4. For a real experiment

**Before recording**
- **Calibrate with the experiment's exact optics.** Keep the lens settings (focus, aperture), the cameras, the tank
  and the water as they will be during the measurement.
- **Measure the wand length independently, to 0.1 % or better** (25 µm on 25 mm).
  - It is the only scale: a 0.1 % error makes every distance 0.1 % wrong.
  - Nothing in the wand data shows it (W8).
- **Enter the windows' constants as well as you know them, but they are not critical.**
  - A plate 10 % too thick or a water index 0.002 off changed the 3-D shape error by at most 1 µm on the reference
    rigs (W8).
  - Give each window its plate thicknesses and its indices, both in order from the camera side.
- **Spread the cameras' lines of sight: 20° or more between them, 35–40° better.**
  - Cameras 5° apart give 2× the f error and 3.5× the 3-D error of cameras 20° apart (`p_mixed_views.py`). Rig C's
    layout also defeated the GUI's calibration (W6).
  - A camera may look straight through the window: that costs nothing, also next to angled cameras.
  - Steep views (up to 45° tested) are fine.
  - Views through two perpendicular walls give the best shape (W2, rig B).
- **Give every window two or more cameras, or at least one with a wide view.** A window seen by only one or two
  long-lens cameras (f above about 10,000 px) is the case that can fail (W5).
- **If the lens may distort, fit k1** (`fit_k1=True`). An unmodelled k1 barely shows in the residual RMS, but
  raises the 3-D error up to 1.9× (W8).
- **Make the two ends distinguishable,** for example with two ball sizes.
  - Label A and B the same way in every camera.
  - Occasional swaps are rejected as outliers (W7).
- **Check the shape of the balls' images on the first recordings.**
  - Through a window at an angle a ball's image can be slightly elongated.
  - Through the tube, pyOpenLPT's circle detector failed on 1.3 : 1 images (cylinder T13).
  - This was not tested for flat windows (section 5). If the images are clearly elongated, use an ellipse fit, as
    T13 recommends.

**While recording**
- **Move the wand through the whole volume you will measure in,** including its full depth as seen from each camera.
  The calibration is only good where the wand went: the shape error doubles at twice the volume (W2).
- **Record about 1,000 frames with both ends seen by at least two cameras.**
  - The tests used 500–1,500. Other frames are dropped.
  - The start needs at least 12 frames that show the whole wand in two or more cameras, ideally in all of them.
  - To join the cameras of different windows, their groups must share at least 3 frames.
- **Use three or more cameras if you can.**
  - With two cameras the noise weights stay equal.
  - Windows with only one or two cameras are the hard case.
- **Aim for about 0.1 px detection noise.** Every error scales with it (W3).
- **Synchronise the cameras, or move the wand slowly.** The tests assume exactly simultaneous frames.

**Checking the result** (the RMS alone can hide a model error)
- **Read the noise section of `calibration_report.txt`.**
  - Each camera's noise should match your detector's. One camera much noisier than the others points to blur or
    focus.
  - With two cameras the weights stay equal, and the report says why.
- **Compare that noise with a still wand imaged over many frames,** which gives the detection noise alone. If the
  fit's noise is clearly larger, suspect model error, motion or synchronisation.
- **Read the windows' and cameras' σ in the report.**
  - A window whose normal is uncertain by degrees, or a camera whose distance is uncertain by tens of mm, is poorly
    determined (W5 seed 1001: 2.1° and 438 mm).
  - Add a camera behind that window, or record more depth.
- **Do not use `fit_L` as a check.** The free length is not identifiable through flat windows (W8).
- **Validate on something of known size instead,** such as a second wand of measured length, or a dot target imaged
  through the window and triangulated.
- **Plot each camera's residuals binned over the image.** A pattern fixed to each image hides in the RMS but shows up
  as structure (cylinder T11e).

## 5. Not covered by any test

- Real data (W10).
- The GUI integration. The CSV reader and writer are done and tested (W9).
- Fitting k1 in a calibration from scratch. k1 was fitted only from the truth (W8). Data with an unmodelled k1 were
  calibrated from scratch too, and gave the same answer as from the truth.
- Windows of two or more plates in a calibration. The model handles them (W0).
- Windows that are not walls of a box tank, more than 3 windows, and more than 6 cameras.
- Cameras with no window (`cam_to_window` −1): accepted, but never run.
- Ball centres through flat windows. The tube's T13 has no flat-window counterpart.
- Patterns fixed to each image (cylinder T11e), noise that varies with position or depth, a bending wand, and camera
  synchronisation.

## 6. Open items and possible next steps

None of these is started; each needs your go-ahead.

1. **Windows seen by one or two long-lens cameras** (W5's misses). What would help, in order of cost (report,
   section 5):
   - **a rough hint:** the window's normal within about 20°, or a camera's distance within about 50 %. W4 shows the
     final fit then converges. `calibrate` accepts focal-length hints today (`hints={"f": [...]}`); a normal or
     distance hint would be new code;
   - **a third camera** behind the window;
   - **a pairwise start** built for this case.

   For seed 2000's kind of miss (every window with two cameras, at 1 px): a few stage-4 starts perturbed around the
   joined start, by about the pairs' own σ. Its wrong basin is narrow (`d4_seed2000.py`), and a stage-4 run costs
   about 4 s on that rig.
2. **GUI integration.** `view.py`'s refractive path would call `plate_wand.calibrate` and write the camFiles where the
   GUI expects them, with the current path kept as an option.
3. **The first real data set (W10):** the GUI's wand CSV, the window settings, the wand length and the current
   calibration's camFiles of one experiment. Run it with the checks of section 4.
4. **Port the two `refine` fixes to the cylinder calibration** (its summary, open item 7):
   - the robust re-triangulation of a frame before re-admission;
   - equal noise weights with two cameras.
5. **The C++ mirror-image quirk** (section 3, item 1). The C++ library was left unchanged. A fix there would report a
   point on the camera side of a window as not projected.

## 7. Files

All in this folder unless stated.

**Plans and reports**, in order:

| file | what |
|---|---|
| `2026-10-02-plate-wand-plan.md` | the approved plan: model, solver, start, tests W0–W10 |
| `2026-10-02-plate-wand-implementation.md` | the working design: section 9 the checklist, section 10 every deviation from the plan with its evidence |
| `2026-10-02-plate-wand-report.md` | the full report: what was built, each test's result, the limitations |

**The tests:**

| test | what | script |
|---|---|---|
| W0 | the model: SymPy checks, against the C++, the rotation log map | `w0_sympy_derivations.py`, `w0_model_vs_cpp.py`, `w0_logmap.py` |
| W1 | derivatives against finite differences | `w1_jacobians.py`, `w1_block2.py` |
| W2 | what the data determine (Cramér–Rao) | `w2_observability.py` |
| W3 | from the truth: noise-free, Monte Carlo, noise sweep | `w3_noise_free.py`, `w3_monte_carlo.py` |
| W4 | how close the start must be | `w4_basin.py` |
| W5 | from scratch, 60 random rigs | `w5_from_scratch.py` |
| W6 | against the GUI's calibration | `w6_vs_gui.py`, `w6_run.py` |
| W7 | bad data | `w7_bad_data.py` |
| W8 | wrong constants, free wand length, k1 | `w8_constants.py` |
| W9 | camFiles and the GUI's CSV | `w9_files.py` |
| W10 | real data | not run: no data set |

- Each script writes `<name>_out.txt` beside it, and most also `<name>.json`.
- Shared code: `prigs.py`, with rigs A–E, the random rigs, wand data from the C++, alignment and metrics.
- Diagnoses of the multi-window start: `d1_multiwindow.py`, `d2_groups.py`, `d3_single.py`, `d4_seed2000.py`, with
  their outputs.
- Pre-plan sizing: `p_sympy_presize.py`, `p_pre_sizing.py`, `p_logmap_check.py`.
- Sizing studies for later questions: `p_mixed_views.py` (square-on and angled cameras, and cameras close together).
- Earlier runs, kept for the record:
  - `runs_v1/`: W5 run 1, and W7 before the re-admission fix;
  - `runs_v2/`, `runs_v3/`: W5 runs 2 and 3.
- `w6_cases/`: W6's data and both calibrations' working files. It is git-ignored, and `w6_vs_gui.py prepare` and
  `run` rebuild it.
- `tests/test_plate_wand.py` (from the pyOpenLPT root): the pytest checks.

**Figures:**
- `figures/plate_wand_results.html` (open it in a browser), also at
  https://claude.ai/artifact/ALBPbfKJKbMddmNBp7m2iV.
- `make_figures.py` rebuilds it from the saved results; nothing is re-fitted.
  - It fills `figures/template.html` and writes the data to `figures/figure_data.json`.
  - `--fragment PATH` also writes the page without its document skeleton, for republishing.

**Example output:** `w9_result/` holds a complete result for rig A: `camFile/`, `calibration_result.json` and
`calibration_report.txt`.

**Rerunning** (from the pyOpenLPT root):

```
python tests/plate_wand/w0_sympy_derivations.py                          # base python (sympy)
C:\Users\zcloc\.conda\envs\refract\python.exe tests/plate_wand/w0_model_vs_cpp.py   # and w0_logmap, w1_jacobians,
                                    # w2_observability, w3_noise_free, w9_files: seconds each
... w3_monte_carlo.py, w4_basin.py, w5_from_scratch.py, w7_bad_data.py, w8_constants.py   # parallel: 2-10 min each
... w6_vs_gui.py prepare | run | evaluate                                 # run: about an hour (the GUI's runs)
pytest tests/test_plate_wand.py                                           # 6 fast checks; -m slow: one from scratch
```

`PLATE_WORKERS` sets the parallel scripts' process count (default 32).
