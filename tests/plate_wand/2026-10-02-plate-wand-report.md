# Wand calibration through flat windows (PINPLATE): report

Date: 2026-10-02. Plan: `2026-10-02-plate-wand-plan.md` (approved 2026-10-02, the six inputs at their defaults).
Working design, checklist and every deviation from the plan with its evidence: `2026-10-02-plate-wand-implementation.md`
(sections 9–10). Summary and file index: `README.md`. Synthetic data only.

**Your request:** "can you also try to improve the currect wand calibration algorithm for refractive plates? i think
the previous gui calibration is slow and not very robust. are you able to develop a new algorithm for that? use your
successful experience on cylinder calibration"

## 1. Summary

**Yes, for the rigs a flat-window experiment normally uses; not yet for two-camera rigs with long lenses.**
- **What was built:** a calibration from wand data alone. It needs only:
  - the detections;
  - the wand length;
  - the image sizes;
  - each window's plates (thicknesses and indices);
  - which camera looks through which window.

  It needs no focal length, camera pose, window position or window orientation. It estimates every camera's pose and
  f, every window's normal and position, per-camera noise, and error bars. It writes PINPLATE camFiles that the C++
  loads. It is the cylinder calibration's machinery with an exact flat-window model; most of the start is new.
- **It is right, and as precise as the data allow.** From the truth on noisy data, its errors are 0.97–0.99 of the
  Cramér–Rao bound on all five reference rigs, its error bars are honest (z standard deviation 0.97–1.00), and it is
  unbiased (W3).
- **From scratch:**
  - every one-window rig in W5 reached the global minimum: 37/37, at 0.1 and 1 px;
  - a median 35 s for 4 cameras and 1,000 frames, measured with the machine fully loaded;
  - rigs with several windows: 11/13 with two or more cameras per window group, 1/3 with one camera per window at
    0.1 px.
- **Against the GUI's calibration on the same data (W6, 18 cases):**
  - 7–112x faster (median 63x): 38 s against 29 min (medians);
  - no failures, where the GUI failed both square-on cases;
  - the same accuracy on the reference rigs, although the GUI was handed the true focal lengths;
  - on the random rigs a 3-D shape error of 1.5 µm against 59 µm (medians), better on 9 of 10.
- **What it does not solve:**
  - **Two-camera configurations with long lenses:** a window seen by one or two narrow-view cameras. 4 of 60 W5 rigs
    end in a wrong minimum, and 1 in a minimum 3.9 σ from the global one. All five have a window seen by at most two
    cameras. Four of them have a camera with f of 24,000–37,000 px. The fifth (seed 2000) has two two-camera groups at
    1 px noise.
  - **The free wand length is not identifiable through flat windows (W8).** The constants check that worked through
    the tube does not exist here.
- **The plan's W5 target was not met:** 46/50 and 5/10 against ≥ 49/50 and 10/10. Section 5 says what would close it.

## 2. What was built

### 2.1 Code

New, in `modules/camera_calibration/curved_calibration/`:
- **`plate_model.py` (440 lines): the PINPLATE camera, vectorised over observations.**
  - The projection is the 1-D Newton of the plan's section 3.1, from τ = ρ/F′(0). It converges monotonically and is
    exact to round-off.
  - Derivatives by implicit differentiation, for the camera, the window (normal and offset about an anchor) and the
    point.
  - Back-projection by vector Snell's law.
  - Conversion to and from the C++ camera.
- **`plate_wand.py` (2,350 lines): the calibration.** The data, the maximum-likelihood fit, outlier rejection, noise
  weights and covariance follow the cylinder's design (`cylinder_wand.py`, of which the model-free pieces are imported);
  the start (stages 1–3) is new.
- **`plate_wand_io.py` (430 lines): files.**
  - the GUI's wand CSV, read and written;
  - a rig configuration file;
  - camFiles through the C++ writer;
  - the result JSON and a text report.

Changed:
- `cyl_model.py`: the approved two-line rotation fix. `log_so3` normalises the axis of a half-turn, and `rvec_of`
  uses it instead of `cv2.Rodrigues`. The cylinder pytest subset still passes (7/7).
- `.gitignore`: negations for `tests/test_plate_wand.py` and `tests/plate_wand/`, whose `w6_cases/` (regenerable data)
  stays ignored.
- `tests/cylinder_wand/README.md`: progress-log entries.

Not touched: the GUI, the C++ library, `cylinder_wand*.py`, `pose_calibrator.py`, the case folder, and `paper/`.

### 2.2 How it works

```python
sol = plate_wand.calibrate(WandData(uv, L, sizes), Rig(windows, cam_win))   # uv[frame, camera, end, (u, v)]
plate_wand_io.write_result(sol, folder)
```

0. **Data.** Frames whose two ends are each seen by at least two cameras; missing detections are allowed.
1. **Affine start.** The cylinder's factorisation and metric upgrade with the wand length. When too few frames are
   seen by every camera, a camera subset is factorised and the other cameras are added by DLT.
2. **Pinhole start.**
   - Each camera's rotation comes from its affine camera.
   - **Its distance comes from perspective.** A wand at depth z looks L|w⊥|/|Δuv| = (D + z)/f, a straight line in z.
     The wand's direction and depth come from the affine reconstruction, its length from the known L. This replaced the
     plan's 1-D search, which the affine points bias about 10x too far.
   - The sign of that line's slope picks the right mirror image.
   - Then a pinhole bundle adjustment.
3. **Windows.** For each window:
   - **Candidates:** normals on a 50° cap about the cameras' viewing direction, and offsets across the camera–cloud gap.
   - **Each candidate's cameras** are placed on their exact chief rays (Snell's law back through the plates), at a
     fixed distance to the candidate window, and resected. The best candidates are refined together with their window.
   - **A window with one or two cameras** is also searched with each candidate scored by the wand-length-constrained
     back-projection residual (`place_window_vp`). A fit to fixed points cannot judge such a window.
   - **Rigs with several windows:**
     - each window with two or more cameras is calibrated on its own;
     - the groups are joined by the rigid motion of their common wand positions;
     - a camera alone behind its window is then placed against the others.
     - With one camera per window, the placements alternate (three passes).
   - **Then a joint coarse fit** of the back-projection residual.
4. **Maximum likelihood.**
   - The pixel reprojection error, with the wand frames eliminated (Schur) and the wand length exact.
   - Levenberg–Marquardt with exact derivatives, stopped on the step size.
   - Outliers rejected and re-admitted.
   - Per-camera noise weights, except with two cameras (section 4).
   - Several starts when the first ones disagree. The answer is chosen among those within 1.5x the best RMS: most
     observations kept, then the likelihood.
5. **Output.**
   - The covariance, in the gauge where camera 1's pose is held.
   - Per-camera and per-window σ.
   - The output frame: Z along window 1's normal, the origin at the wand's mean position.
   - camFiles, JSON and report (`plate_wand_io`).

Typical timing (rig A, 4 cameras, 1,000 frames, one process): pinhole start 1 s, window search 8 s, coarse fit 6 s,
final fit 11 s (two starts).

## 3. Results

### 3.1 The model (W0, W1)

- **SymPy** (`w0_sympy_derivations.py`): every closed form of the derivatives, the ρ → 0 limits, the rotation and
  pinhole steps, and the vector Snell step. 77 PROVED. The 2 SKIPPED are limits argued in the text and checked
  numerically in W1.
- **Against the C++** (`w0_model_vs_cpp.py`): 2,000 random cameras, windows (one and two plates, incidence to 60°)
  and points.
  - Projections agree to 6.5e-11 px, rays to 1.3e-13. Points on the camera's normal line agree to 2.7e-7 px, because
    the C++ treats a lateral distance below 1e-8 mm as zero.
  - With distortion the rays agree to 1e-5 px, the tolerance at which the C++ stops undistorting.
  - **A C++ quirk, recorded and not changed:** a point on the camera side of a window (inside the plate, or in the air)
    is projected as its mirror image in the window (600 of 600, to 8e-11 px). The C++ uses the unsigned distance to
    the window. The new model reports such a point as not projecting. The GUI's pipeline uses the C++ and can be
    misled by it.
- **Derivatives** (`w1_jacobians.py`): every column within 1.3e-7 of central differences, including points on and next
  to the normal line. The bundle adjustment's Jacobian (`evaluate`) is within 3e-7.
- **The rotation conversion** (`w0_logmap.py`): 1.7e-15 rad at every angle after the `cyl_model` fix. `cv2.Rodrigues`
  loses 2.0e-5 rad on the same rotations.

### 3.2 What the data determine (W2, Cramér–Rao at 0.1 px, 1,000 frames)

| rig | f | window normal | camera's distance to window | 3-D shape of unseen points (calibration) | 25 mm wand, calibration / with detection noise |
|---|---|---|---|---|---|
| A, one wall, 4 cameras at ~21° | 0.063 % | 0.026° | 0.47 mm | 5.5 µm | 1.9 / 30 µm |
| B, two walls, 2 + 2 | 0.055 % | 0.018°, 0.016° | 0.95 mm | 2.4 µm | 0.9 / 16 µm |
| C, square-on, 3 within 5° | 0.20 % | 0.14° | 4.4 mm | 26 µm | 8.9 / 148 µm |
| D, steep, 35–45° | 0.059 % | 0.015° | 0.37 mm | 4.3 µm | 1.5 / 23 µm |
| E, macro, f 40,000 px | 0.61 % | 0.016° | 6.7 mm | 0.6 µm | 0.6 / 8 µm (10 mm wand) |

- **Shape against position.** The shape of what the cameras measure is known 100–10,000 times better than its position
  relative to camera 1. The scene, the window and f can slide together along the viewing direction: 0.43–0.48 mm on
  A, B and D, 1.6 mm on C, 7.3 mm on E (W2's `gauge_mm`).
- **Rig C** is 3–9x weaker than rig A on every measure. The cause is its cameras' small angle to each other (5°), not
  their view straight through the window: the same layout moved to 18–25° incidence is as weak (`p_mixed_views.py`,
  added later; README section 3, item 6).
- **Points outside the calibrated volume:** the shape error grows about linearly with the distance (2x at twice the
  volume).

### 3.3 Precision and error bars (W3)

- **Noise-free, from the truth:**
  - RMS 1e-12 px, every parameter at round-off;
  - from a start 2 % off in f, 5 mm off in position, 0.3° in rotation, 1° in the normal and 3 mm in the offset, too:
    15–82 iterations, at most 3 s.
- **Monte Carlo, 40 data sets per rig at 0.1 px:**

| rig | parameters | RMS error / Cramér–Rao | z standard deviation | largest bias |
|---|---|---|---|---|
| A | 25 | 0.98 | 0.98 | 1.7 SE |
| B | 28 | 0.99 | 1.00 | 1.8 SE |
| C | 18 | 0.97 | 0.98 | 2.3 SE |
| D | 18 | 0.99 | 0.99 | 2.4 SE |
| E | 18 | 0.97 | 0.97 | 1.3 SE |

- **Noise sweep on A, 0.02–2 px:** the efficiency stays at 0.86–1.14 with 10 sets per level, so the errors scale
  linearly with the noise.

### 3.4 How close the start must be (W4)

The final fit (coarse fit, then maximum likelihood) was started from the truth with one thing wrong. It recovers:
- **the window normal:** 30° off on every rig, 45° except the steep rig D (8/10), 60° except D;
- **the window offset:** anywhere across the gap between the cameras and the cloud (±90 %);
- **f:** ±40 %, except the macro rig E (7/10 at 20 %, 4/10 at 40 %), where f and distance trade almost freely;
- **camera rotations:** these are the weak spot (A 8/10 at 5°; C 5/10 and E 1/10 at 10°), which is why stage 2 gets
  them from the affine cameras (about 1°).

The stage-3 spacing follows: a 9° grid on a 50° cap and 3 offsets.

### 3.5 From scratch (W5)

50 random rigs at 0.1 px and 10 at 1 px. The rigs have:
- 2–6 cameras;
- 1–3 walls of a box tank;
- incidence 0–45°, 150–1,500 mm from the window;
- f 1,500–40,000 px;
- 30–300 mm clouds and 500–1,500 frames.

`calibrate` gets placeholder window planes. Success means reaching the minimum of the fit started from the truth on the
same data: every parameter within 0.01 σ, the same active set, and the RMS within 1e-6.

| rig kind | 0.1 px: same minimum | 1 px: same / equivalent | time (median) |
|---|---|---|---|
| one window | 34/34 | 3/3 / 3/3 | 35 s for 4 cameras (20–79 s, 2–6 cameras) |
| several windows, camera groups | 11/13 | 2/4 / 3/4 | 121 s |
| several windows, one camera each | 1/3 | 0/3 / 3/3 | 121 s |
| all | **46/50** | **5/10 / 9/10** | |

"Equivalent" means every parameter within 3 σ of the reference and the RMS within 2 %, or a lower RMS.

The timings were taken with 16 W5 processes and the GUI runs of W6 sharing 32 cores.

**The misses** (`w5_from_scratch_out.txt`). All have a window seen by at most two cameras, and most have long lenses:

| seed | cameras per window | f (px) | result |
|---|---|---|---|
| 1001 | 1, 1 (9° and 44°) | 12,300 and 24,500 | a nearby minimum, RMS +2 %. Window 2 is barely determined even at the truth (σ 2.1°, 438 mm) |
| 1045 | 1, 1 (33° and 37°) | 12,700 and 29,100 | RMS 0.35 against 0.086 px |
| 1031 | 2, 1 | 6,400–31,700 | the two-camera group (13,500 and 31,700 px) cannot be calibrated on its own; 5.3 px |
| 1034 | 2, 1, 1 | 4,400–37,200 | 1.8 px |
| 2000 (1 px) | 2, 2 | 2,000–11,000 | 3.9 σ from the global minimum, RMS +0.3 % |

### 3.6 Against the GUI's calibration (W6)

The GUI's current refractive calibration (`RefractiveWandCalibrator.calibrate`, run headless exactly as `view.py`'s
worker runs it, with the **true focal lengths** in its camera table) and the new one (no focal lengths), on the same
data: rigs A–D at 0.1 and 0.5 px and the first ten W5 rigs at 0.1 px, 1,000 frames each, every frame seen by every
camera (the GUI drops a camera missing from any frame), all 36 calibrations in parallel. Each calibration is judged
by its exported camFiles, loaded through the C++:
- the 3-D shape error of 500 unseen points (their exact pixels triangulated, then the best rigid motion onto the truth);
- the length error of 300 held-out wands;
- the reprojection of the unseen points;
- the largest f error.

**Summary:**
- **Time.** The GUI took 685–4,222 s (median 1,720 s, 29 min); the new calibration took 21–153 s (median 38 s). That is
  7–112x faster, median 63x. Part of the GUI's time overlapped the other tests' runs, which slowed it by perhaps
  10–30 %.
- **Failures.** The GUI failed on both square-on cases (rig C):
  - at 0.1 px its pinhole start collapsed the scale ("scale sanity failed: median 3.11 mm, target 25 mm");
  - at 0.5 px it found no "geometry-valid camera pair".

  The new calibration solved both, at about the precision W2 predicts (33 µm against 26 µm at 0.1 px).
- **Rigs A, B, D, where both work:** comparable. The 3-D shape error medians are 6.9 µm (GUI) and 6.2 µm (new), both
  near the Cramér–Rao bound. The GUI is helped by being given the true f.
- **The random rigs:** the new calibration is far more accurate.
  - 3-D shape error: median 1.5 µm against 58.7 µm, better on 9 of 10, by up to 400x;
  - the largest f error: 0.12 % against 2.7 % (median). The GUI's f is bounded to ±5 % of the true value it was given,
    and two runs ended on that bound;
  - unseen points reproject within 0.012 px against 0.54 px (median).
  - The GUI's worst runs (rand1000, rand1005: 0.5–0.9 mm) are what its bounded start-and-polish produces when the start
    is poor.

| case | calibration | time | largest f error | 3-D shape error (µm) | wand error, mean / RMS (µm) | reprojection of unseen points (px) |
|---|---|---|---|---|---|---|
| A_0.1px | GUI | 1742 s | 0.06 % | 5.2 | -0.3 / 1.6 | 0.020 |
|  | new | 34 s | 0.10 % | 6.6 | -0.1 / 2.1 | 0.025 |
| A_0.5px | GUI | 1373 s | 0.28 % | 19.9 | -3.6 / 6.7 | 0.080 |
|  | new | 30 s | 0.55 % | 10.3 | +0.5 / 3.6 | 0.045 |
| B_0.1px | GUI | 1271 s | 0.03 % | 1.4 | +0.0 / 0.5 | 0.008 |
|  | new | 94 s | 0.05 % | 2.1 | +0.1 / 0.7 | 0.011 |
| B_0.5px | GUI | 1353 s | 0.22 % | 7.5 | -3.0 / 3.4 | 0.048 |
|  | new | 97 s | 0.35 % | 5.5 | -0.2 / 2.0 | 0.043 |
| C_0.1px | GUI | failed after 171 s (scale sanity check) |  |  |  |  |
|  | new | 30 s | 0.06 % | 33.1 | -5.4 / 10.9 | 0.123 |
| C_0.5px | GUI | failed after 6 s (no valid camera pair) |  |  |  |  |
|  | new | 153 s | 0.64 % | 163.0 | -6.9 / 52.0 | 0.640 |
| D_0.1px | GUI | 1541 s | 0.05 % | 6.4 | +0.1 / 1.9 | 0.024 |
|  | new | 24 s | 0.08 % | 5.8 | +0.2 / 1.8 | 0.022 |
| D_0.5px | GUI | 1461 s | 0.32 % | 23.5 | -9.2 / 10.3 | 0.096 |
|  | new | 24 s | 0.72 % | 25.1 | -6.3 / 9.4 | 0.088 |
| rand1000_0.1px | GUI | 4185 s | 2.42 % | 530.5 | -34.8 / 245.3 | 1.977 |
|  | new | 65 s | 0.05 % | 1.6 | +0.5 / 0.8 | 0.008 |
| rand1001_0.1px | GUI | 951 s | 2.44 % | 28.6 | -0.4 / 9.6 | 0.583 |
|  | new | 140 s | 1.29 % | 3.8 | -0.1 / 2.0 | 0.060 |
| rand1002_0.1px | GUI | 2886 s | 2.93 % | 5.3 | -0.7 / 3.5 | 0.056 |
|  | new | 52 s | 0.14 % | 0.7 | +0.1 / 0.4 | 0.009 |
| rand1003_0.1px | GUI | 3723 s | 4.91 % | 140.7 | +30.7 / 54.3 | 1.338 |
|  | new | 54 s | 0.11 % | 0.3 | +0.1 / 0.2 | 0.007 |
| rand1004_0.1px | GUI | 1698 s | 2.30 % | 3.3 | +0.3 / 2.4 | 0.056 |
|  | new | 26 s | 0.14 % | 0.6 | +0.1 / 0.4 | 0.012 |
| rand1005_0.1px | GUI | 4222 s | 6.62 % | 939.4 | +8.2 / 295.2 | 2.080 |
|  | new | 38 s | 0.02 % | 13.6 | +4.5 / 5.1 | 0.019 |
| rand1006_0.1px | GUI | 3459 s | 0.71 % | 9.0 | +0.1 / 5.1 | 0.031 |
|  | new | 38 s | 0.18 % | 2.6 | +0.5 / 1.3 | 0.011 |
| rand1007_0.1px | GUI | 1782 s | 5.00 % | 88.9 | -4.9 / 58.0 | 0.496 |
|  | new | 21 s | 0.06 % | 1.4 | -0.4 / 0.8 | 0.008 |
| rand1008_0.1px | GUI | 4050 s | 5.00 % | 214.7 | +15.8 / 65.9 | 1.988 |
|  | new | 47 s | 0.08 % | 1.2 | -0.3 / 0.4 | 0.017 |
| rand1009_0.1px | GUI | 685 s | 0.13 % | 3.8 | -0.6 / 1.7 | 0.025 |
|  | new | 30 s | 0.22 % | 4.7 | -0.3 / 2.0 | 0.031 |

Rig C at 0.5 px took the new calibration 153 s: the coarse fit's RMS exceeded 1 px, so every start was carried
through. rand1001 is one of W5's hard cases (two cameras through two walls); here its answer is close to the true
one.

### 3.7 Bad data (W7)

Each case was calibrated from scratch on rigs A and B (1,000 frames, 0.1 px) and compared with the clean data's
answer:
- **2 % gross outliers (20–200 px):** all 160 rejected; 0–6 good detections lost.
- **1 % swapped ends:** all rejected; no good detection lost.
- **A camera missing from 40 % of the frames:** the answer moved by the amount the lost data explain (z 0.39 against
  0.39 expected).
- **A camera 5x noisier:** its noise was estimated at 0.503 px (true 0.5 px), and the others at 0.10 px.
- **No frame seen by every camera** (two cameras each missing a different half):
  - rig A uses the camera-subset start, the fourth camera added by DLT;
  - rig B: the window group whose two cameras never see the same frame falls back to the whole-rig start.

Every answer is at the minimum of a fit started from the truth on the same data.

**One marginal miss of the criterion:** rig B with no common frames moved 1.11 σ RMS from the clean answer, against 0.72
σ expected for the data it lost (limit 1.08). It is nevertheless at its data's own minimum.

**Fixed on the way:** before the fix, the 2 % outliers cost 1,001 good detections. A gross outlier's frame lost all its
rays, because the frame was re-triangulated with the outlier included. Section 4 has the fix.

### 3.8 Wrong constants (W8)

Rigs A and D, fit from the truth (and three cases from scratch, which give the same answers):

| given | residual RMS | 3-D shape of unseen points | what shows it |
|---|---|---|---|
| correct | — | 4.3–5.4 µm | — |
| plate 10 % thicker | unchanged | unchanged | nothing |
| water index +0.002 | unchanged | 5.3–5.4 µm | nothing |
| wand 0.1 % long | unchanged | 49–51 µm (a 0.1 % scale error) | only an independent length |
| unmodelled k1 = 0.05 | +0.3–0.4 % | 7–8 µm | barely the residuals; fitting k1 recovers it (0.043–0.058) |

**The wand length is not identifiable through flat windows.** With it free, the fit returns 10–90 mm ± 5–18 mm for a
25 mm wand. Only the plate's thickness carries scale (SymPy, plan section 3.1), and 12.7 mm of acrylic is a poor ruler.
`fit_L` is therefore not a check of the constants here, unlike the tube.

So in practice:
- the wand length must be right, because it is the scale;
- the plates' constants hardly matter;
- k1 should be fitted when the lens may distort.

### 3.9 Files (W9) and real data (W10)

- **camFiles:** written through the C++ writer, they reload with `pyopenlpt.Camera(path)` and project within 7e-12 px.
  The camera and window read back exactly.
- **The GUI's CSV:** round-trips, and the GUI's own loader reads ours.
- An example result is in `w9_result/`.
- **W10 was not run:** there is no real data set.

## 4. What changed from the plan

Each item is detailed with its evidence in the implementation doc, section 10. The ones that matter:
- **Stage 2's distance:** from perspective, not the planned search on the affine points.
- **Stage 3:**
  - the cameras are put on their exact chief rays, at a fixed distance to the candidate window;
  - the short list is refined together with the window.
  - A window with one or two cameras is also judged by the wand-length residual.
  - Rigs with several windows are started window group by window group, plus placements.
- **The back-projection residual** now penalises a wand end on the camera side of a window. Without it, the coarse fit
  placed a window beyond the wand.
- **Stage 4:**
  - **The choice among starts:** a stuck start inflates its noise estimate and keeps more observations, so it no longer
    wins on that count.
  - **Equal weights with two cameras:** the per-camera weights drifted and rejected 11 % of good data.
  - **Re-admission:** a frame is re-triangulated without its worst ray.

  The last two also apply to `cylinder_wand.refine`, which was not changed. They are logged in
  `tests/cylinder_wand/README.md`, open item 7.

## 5. Limitations and what would close them

1. **Two-camera configurations with long lenses (W5's misses).**
   - With one or two narrow-view cameras behind a window, the window's tilt changes the images only weakly. The cost
     then has several minima, and even the right one can be poorly determined (seed 1001: σ 2.1° and 438 mm).
   - What would help, in order of cost:
     - a rough hint: the window's normal within about 20°, or a camera's distance within about 50%. W4 shows the final
       fit then converges;
     - a third camera behind the window;
     - a pairwise start built for this case.
   - The error bars already warn: a weak window shows a large σ.
2. **No constants check through flat windows** (W8). A wand of independently measured length is the only check of the
   scale.
3. **Synthetic only.** Real detections bring patterns fixed to each image (the cylinder's T11e), blob-centre biases and
   wand bending, none of which was tested.
4. **The GUI integration is not done.** It is a separate step for your approval. `view.py`'s refractive path would call
   `plate_wand.calibrate` and write the camFiles where the GUI expects them, with the current path kept as an option.

## 6. Reproducing

```
python tests/plate_wand/w0_sympy_derivations.py                          # base python (sympy)
C:\Users\zcloc\.conda\envs\refract\python.exe tests/plate_wand/w0_model_vs_cpp.py   # and w0_logmap, w1_jacobians,
                                    # w2_observability, w3_noise_free, w9_files: seconds each
... w3_monte_carlo.py, w4_basin.py, w5_from_scratch.py, w7_bad_data.py, w8_constants.py   # parallel: 2-10 min each
... w6_vs_gui.py prepare | run | evaluate                                 # run: about an hour (the GUI's runs)
pytest tests/test_plate_wand.py                                           # 6 fast checks; -m slow: one from scratch
```

`PLATE_WORKERS` sets the parallel scripts' process count (default 32).
