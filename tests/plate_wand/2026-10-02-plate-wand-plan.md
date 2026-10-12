# Wand calibration through flat windows (PINPLATE): plan

Date: 2026-10-02. Status: **approved by you on 2026-10-02 ("go ahead to finish all phases follow the plan").**
- The six inputs of section 6 were not answered, so their defaults apply:
  - synthetic data only;
  - rigs A–E;
  - thickness and indices fixed per window, the principal point at the centre, k1 optional;
  - the GUI's focal length only as an optional prior;
  - the GUI integration later, as a separate step;
  - the two-line `cyl_model.py` fix applied.
- The working design, and a checklist of the phases, are in `2026-10-02-plate-wand-implementation.md`.
- **Implemented 2026-10-02**, all phases. Results: `2026-10-02-plate-wand-report.md`; summary: `README.md`. The plan's
  W5 target (≥ 49/50, 10/10) was not met: 46/50 and 5/10 (9/10 equivalent); every one-window rig succeeded, and the
  misses are two-camera configurations with long lenses (report, section 5). Stage 2 and stage 3 differ from section
  4.4 as recorded in the implementation doc, section 10.

The pre-plan sizing (section 3) is done, with its scripts and outputs in this folder:
- `p_sympy_presize.py`;
- `p_pre_sizing.py`;
- `p_logmap_check.py`.

**Your request:** "can you also try to improve the currect wand calibration algorithm for refractive plates? i think
the previous gui calibration is slow and not very robust. are you able to develop a new algorithm for that? use your
successful experience on cylinder calibration"

Scope:
- pyOpenLPT only, and new files only. The following are read, not changed:
  - the GUI's wand calibration (`modules/camera_calibration/wand_calibration/`, `view.py`);
  - the C++ library;
  - the cylinder calibration.
- Nothing in `D:\Refraction\paper` is touched.
- Per the debug-workflow, each test writes its outputs beside itself in this folder, and the final report goes here
  too.
- Putting the new calibration into the GUI is a separate step for your approval (section 7).

## 1. The short answer

**Yes.** I expect a new calibration that is much faster and finds the right answer from scratch, for three reasons.
My estimates are under a minute against tens of minutes now; W6 measures both on the same data.
- **The current pipeline's problems are structural, not tuning** (section 2):
  - a start that ignores refraction;
  - bounds that keep the fit close to that start;
  - an objective that is not the pixel error;
  - a coordinate frame held only by bounds;
  - derivatives by finite differences that redo all the work in Python for every parameter.
- **The cylinder calibration solved the same problems for a harder port** (section 4.1):
  - a start from the wand data alone;
  - the exact model, with exact derivatives;
  - the maximum-likelihood fit, with the wand frames eliminated;
  - a fixed gauge;
  - automatic outlier rejection and per-camera noise weights;
  - honest error bars.

  All 50 random cylinder rigs reached the global minimum from scratch.
- **A flat window is simpler than the tube in one key way:** the projection is one equation in one unknown, and
  Newton's method always finds its root (section 3.1, checked with SymPy).

**One physical limit stays.** A camera that looks nearly straight through a window behaves almost exactly like a
pinhole camera in the water, so its distance from the window barely shows in the images (section 3.2). The new
calibration converges along that direction anyway and reports how well the distance is known. It also matters
little for measurements inside the calibrated volume. W2 measures both.

## 2. What the current pipeline does, and why it is slow and fragile

This was read from the code at HEAD `2cfe474`; none of it was run. W6 measures the speed and accuracy directly.
Paths are under `modules/camera_calibration/wand_calibration/`.

**The GUI path** ("Run Calibration" with Pinhole+Refraction), in order:
1. **Pair ranking:** a full pinhole calibration, used only to rank camera pairs (`wand_calibrator.py:2217`).
2. **Pinhole bootstrap ("P0" in the code):** each camera's focal length is frozen at the GUI value
   (`refractive_bootstrap.py`).
   - An essential matrix from a seed pair, then PnP for the other cameras, then a bundle adjustment.
   - Up to 3 seed pairs are tried, plus one outer retry.
3. **Windows:** the planes come from rules of thumb (`PlaneInitializer`, `plane_d_solver.py`).
4. **Bundle adjustment in rounds** (`refraction_calibration_BA.py`):
   - planes only;
   - then cameras only;
   - then both together;
   - then both, with f, thickness and k1/k2 added.
5. **Output:** the world frame is re-aligned, then the camFiles and a report are written.

**Why it is slow**
- **Finite-difference derivatives.**
  - Every derivative is a dense finite difference: one full evaluation per parameter, 30–44 parameters for 4 cameras
    and 2 windows.
  - Each evaluation (`refraction_calibration_BA.py:669-1083`):
    - re-sends every camera to C++;
    - rebuilds every ray with single-point C++ calls;
    - triangulates every frame in a Python loop, with an SVD per point.
  - Estimated cost: 0.3–0.6 s per evaluation for 4 cameras × 1,000 frames, so 15–20 s per iteration.
- **Restarts.**
  - The last two rounds run in up to 22 chunks. Each is a fresh solver call with a new starting Jacobian and a reset
    trust region (`:1589-1691`).
  - An early stop reruns the whole schedule once.
- **Work done twice.** The pinhole problem is solved twice before refraction is touched.
- **Estimate if every budget is spent:** about an hour (planes 2 min, cameras 9 min, joint 11 min, final 33 min), and
  up to twice that with the retries. All of it runs on one thread.

**Why it is fragile.** The git history records each of these failing.
1. **The start ignores refraction.**
   - A pinhole model with the GUI's focal length frozen cannot represent a camera behind water.
     - Square-on, the best pinhole has f ×1.33 and sits 1.33× farther away (section 3.2; also
       `D:\Refraction\reports\2026-09-17-calibration.md` §2.1).
     - At 15–45° it misses by 2.5–28 px RMS (section 3.2). RANSAC then treats refraction as outliers: the authors saw
       36 % inliers and a collapsed scale (commit `fd004f6`).
   - The pinhole bootstrap failed with 1e5–6e5 px errors in 7 of 29 robustness cases (`499e00f`). The fix was rules for
     choosing camera pairs.
2. **The windows start from rules of thumb.**
   - The normal is the mean line of sight from the cameras to the wand cloud (`refraction_wand_calibrator.py:479-484`).
     That is off by the viewing angle.
   - The distance is the median range / 1.333 (`:486-493`).
   - The analytic solver for the distance:
     - ignores the window's thickness;
     - may move the distance by at most ±50 %;
     - otherwise falls back silently (`plane_d_solver.py:212-239`).
   - If the start is not between a camera and its nearest point, the distance is frozen
     (`refraction_calibration_BA.py:2985-2991`).
3. **The fit can only move a little from that start.**
   - There is one plane/camera pass (`:3098`, "prevent cumulative plane drift").
   - Each round has box bounds:
     - plane tilt ±2.5°, then ±10° and ±5°;
     - plane distance 5–18 mm;
     - cameras ±5° and ±15 mm;
     - f and thickness ±5 %.
   - In total it can reach about 17° on the normal and a few tens of mm on the distance.
   - An exhausted budget is accepted as a partial result with a warning (`:2461-2464`).
4. **The objective is not the pixel error.**
   - It is distances to rays in millimetres, with one scale for every camera.
   - The wand length is a soft term with an arbitrary weight (λ = 2 × cameras, σ = 2 % of L, `:1375-1381`).
   - A side barrier is added.
   - The weights are re-estimated between chunks, so the objective changes during the fit.
   - It weights cameras and points by their millimetres per pixel, and is not the maximum-likelihood estimate.
5. **The coordinate frame is not fixed.** The six rigid motions are held only by bounds and repeated re-alignments.
   The "drifting planes" fix (`a7649e0`) tightened the bounds and cut the loops from 6 to 1.
6. **The stopping rules are loose.**
   - A round stops after two chunks that each gain less than 0.5 %.
   - ftol is 5e-4.
   - The budgets are 50–100 evaluations.

   It can stop while a parameter is still several standard errors from its optimum.
7. **No automatic outlier rejection:** the loss is linear everywhere. Cleaning is manual, using the pinhole
   precalibration, which ignores refraction.
8. **Smaller defects:**
   - A camera missing from even one frame is dropped from the whole calibration
     (`refraction_wand_calibrator.py:243-254`).
   - One set of indices and thickness is copied to every window (`view.py:7439-7446`).
   - The bootstrap cache's key ignores the focal lengths, the image size and the detections (`:145-206`).
   - Failed projections are left out of the reported errors, which makes them look better than they are.
   - A batch of 14 principled fixes (Huber loss, cheirality checks, gauge fixing) was reverted as "unusable"
     (`72cb3cf`).

| current | new (section 4) |
|---|---|
| pinhole start with f frozen; windows by rules of thumb | start from the wand data alone: affine, then pinhole with f free, then windows found by a search scored through the exact refractive model |
| finite-difference derivatives redoing everything in Python | exact derivatives from a vectorised model; frames eliminated (cost linear in the number of frames) |
| millimetre ray distances, soft wand length, barrier | pixel reprojection error (maximum likelihood), with the wand length exact |
| bounds, caps, chunks, one alternation pass | Levenberg–Marquardt to convergence, no bounds; a step that breaks a projection is rejected |
| free frame held by bounds | gauge fixed exactly |
| linear loss, manual cleaning | outliers rejected automatically, with re-admission; per-camera noise weights |
| cameras must be in every frame | every detection is used where it exists |
| no error bars | covariance, per-camera noise, predicted precision |

## 3. Pre-plan sizing (done)

### 3.1 The projection through a flat window (SymPy)

- **The equation.** Every interface has the same normal, so the ray stays in the plane through the normal and the
  point. Snell's law keeps n sin θ the same in every layer.
- **The unknown.** With τ = tan θ in the air, the projection is one equation:

  F(τ) = h₁ τ + Σᵢ hᵢ n₁ τ / √(nᵢ² + (nᵢ² − n₁²) τ²) = ρ

  - h₁ is the camera's distance from the window;
  - hᵢ are the window's thickness and the point's depth in the water;
  - ρ is the point's distance from the camera's normal line.
- **SymPy proved** (`p_sympy_presize.py`, output `p_sympy_presize_out.txt`: 13 PROVED, 1 SKIPPED):
  - the formula from Snell's law;
  - each layer's term is increasing, and concave when its index exceeds the camera side's (air);
  - F(0) = 0;
  - F′(0) = h₁ + t n₁/n₂ + Z n₁/n₃;
  - F scales with the lengths, so with the thickness held only its term fixes the scale (section 4.3).
- **So Newton's method always converges.** Started from the small-angle value τ = ρ / F′(0), it rises monotonically to
  the root and cannot overshoot. That is the SKIPPED step, an argument resting on the proved concavity. There is one
  image, and it is always found.
- **Derivatives** follow by implicit differentiation: dτ = (dρ − τ dh₁ − Σ gᵢ dhᵢ) / F′(τ), proved for h₁ and ρ.
  - There is no 2-D Newton and no image census, unlike the tube.
  - The C++ uses the same equation in another variable, with a safeguarded Newton and a fallback.
- **SymPy also proved the small-angle equivalence.** A camera looking straight through the window projects, to first
  order, exactly like a pinhole camera in the water with:
  - focal length f n₃/n₁;
  - its centre n₃ (h₁/n₁ + t/n₂) behind the window's water-side surface.

  This is why the window's distance is hard to see.

### 3.2 How far is a camera behind a flat window from a pinhole? (`p_pre_sizing.py`, the C++ model)

Setup:
- **Camera:** 400 mm from a 12.7 mm acrylic window, measured along its chief ray, with water behind the window.
- **Views:** incidence 0–45°.
- **Lenses:** a 1280 × 800 sensor with f = 3,000 px (a ±12° field) or 12,000 px (±3°).
- **Points:** 585, from 20 to 120 mm deep in the water, with their exact images.
- **Caveat:** every fit has the points known, as with a dot target. A wand gives less; W2 measures that.

**The best pinhole camera** (cx and cy at the centre). The first two columns give the misfit, RMS / max over the
points:

| incidence | f = 3,000 px | f = 12,000 px | the pinhole's f, and its distance, relative to the true camera |
|---|---|---|---|
| 0° | 0.17 / 0.64 px (with k1: 0.06 px) | 0.011 / 0.041 px (with k1: 0.004 px) | ×1.33–1.34, ×1.33–1.34 |
| 15° | 5.0 / 13.7 px | 2.5 / 5.3 px | ×1.29–1.34, ×1.29–1.35 |
| 30° | 13.2 / 37.5 px | 10.1 / 19.6 px | ×1.17–1.38, ×1.19–1.44 |
| 45° | 27.9 / 76.7 px | 26.1 / 50.1 px | ×1.01–1.45, ×1.05–1.64 |

- **Square-on, the camera is a pinhole:** to 0.01–0.17 px, and to 0.004–0.06 px with k1. Its f and distance are
  scaled by 1.33, the water's index, as section 3.1 predicts.
- **Oblique, the pinhole misses by 2.5–28 px RMS.**
  - Freeing the principal point brings that down to 0.5–24 px.
  - It does so only by moving the principal point off the sensor: cx from −9,400 to 3,865 px on a 1,280 px sensor.
    That is the 2026-09-17 finding.

**How visible the window is.** The window is moved or tilted, the camera is refitted (rotation, centre, f), and the
table gives the RMS left. Each cell is f = 3,000 / f = 12,000 px:

| change | 0° | 15° | 30° | 45° |
|---|---|---|---|---|
| window 10 % closer to the camera (28–40 mm) | 0.08 / 0.005 px | 0.58 / 0.40 px | 1.5 / 1.5 px | 2.3 / 2.9 px |
| window 25 % closer (71–100 mm) | 0.19 / 0.012 px | 1.5 / 1.0 px | 3.9 / 3.7 px | 6.1 / 6.7 px |
| window 15 mm farther | 0.03 / 0.002 px | 0.22 / 0.16 px | 0.61 / 0.66 px | 1.1 / 1.6 px |
| normal tilted 2°, in the plane of incidence | 0.79 / 0.21 px | 1.1 / 0.78 px | 2.0 / 2.0 px | 3.5 / 4.3 px |
| normal tilted 2°, out of it | 0.57 / 0.16 px | 0.99 / 0.82 px | 1.8 / 1.8 px | 2.8 / 3.4 px |

- **A square-on camera with a long lens cannot see the window's distance.**
  - Moving the window 100 mm is absorbed to 0.012 px.
  - Anything that relies on that distance there fights the noise. The current pipeline's distance solver and bounds
    do.
  - The same fact makes the error harmless for points like these: the refitted camera reproduces their images to
    0.01 px. W2 checks this with wand data and outside the calibrated depth.
- **From 15° on, the distance shows clearly** (0.4–2.9 px for a 10 % change), and so does the tilt (0.8–4.3 px for
  2°).
- **The tilt shows even square-on** (0.16–0.79 px for 2°).
- **What this means for the current start.** Its pinhole with f frozen at the in-air value is off by ×1.33 in f and
  in distance even square-on. At 15–45° it misses by 2.5–28 px, which RANSAC treats as outliers.

### 3.3 Found on the way: two rotation conversions that fail near a half-turn

The sizing builds C++ cameras from rotation matrices, and the C++ takes a rotation vector. Cameras placed with
"look at" are often exact half-turns. Measured round trips, 200 random rotations per row (`p_logmap_check.py`, output
`p_logmap_check_out.txt`):

| distance from a half-turn | `cv2.Rodrigues` | `cyl_model.log_so3` | corrected log map |
|---|---|---|---|
| 0 (exact) | 2e-8 rad | 1.3 rad | 7e-16 rad |
| 1e-7 – 1e-5 rad | 2e-7 – 2e-5 rad | 1.3 rad | 1e-15 rad |
| 1e-4 rad | 2e-5 rad | 1.3 rad | 8e-16 rad |
| 1e-3 – 0.05 rad | 4e-11 – 7e-13 rad | 1.3 rad | 7e-16 rad |

- **`cyl_model.log_so3` (cylinder code) is wrong** for every rotation within 0.14 rad of a half-turn. Its half-turn
  branch does not normalise the axis, so it returns a rotation of about 1.41·π instead of π.
  - The cylinder calibration uses it only:
    - as `rvec_of`'s fallback when OpenCV is missing;
    - in one test helper's error metric (`tests/cylinder_wand/rigs.py:206`).

    So no cylinder result depends on it.
- **`cv2.Rodrigues` loses up to 2e-5 rad** within about 1e-4 rad of a half-turn. At 1 m and f = 10,000 px that is
  about 0.2 px.
  - `cyl_model.rvec_of` uses it to build C++ cylinder cameras (`to_cpp`), which feeds the T0/T10 checks and the
    double-image guard.
  - The camFiles are not affected: the C++ reads the rotation matrix and ignores the vector on load.
- **The proposed fix** (two lines in `cyl_model.py`, for your approval):
  - normalise the axis in `log_so3`'s half-turn branch;
  - have `rvec_of` use `log_so3` instead of OpenCV.

  The corrected version is accurate to 1e-15 rad at every angle. The new plate code uses the corrected map either
  way.

## 4. The new calibration

### 4.1 What carries over from the cylinder

Every piece below is tested there (`tests/cylinder_wand/README.md`):
- **The start from the data alone.** An affine factorisation of the frames, made metric by the wand length. Candidate
  placements are ranked only after a quick fit, because a start's cost as placed is misleading.
- **The exact model.** It is vectorised over observations, with exact derivatives by implicit differentiation, and
  checked against the C++ before use.
- **The maximum-likelihood fit.** The pixel reprojection error is minimised by Levenberg–Marquardt on manifolds.
  - Each frame is a midpoint and a direction, so the wand length is exact.
  - The frames are eliminated by the Schur complement, so the cost is linear in the number of frames.
  - It stops on the step size, not the cost: along a nearly flat direction a small cost change can still hide a real
    error.
- **The gauge is fixed exactly**, so nothing drifts.
- **Outlier rejection with re-admission.** Each residual is judged by its Mahalanobis distance. An observation
  rejected while the fit was still poor comes back once it fits.
- **Per-camera noise weights**, estimated from the residuals. When the data cannot separate the cameras, the weights
  stay equal.
- **The covariance and checks:** per-camera error bars, the predicted error on unseen points, and the free-length
  check.
- **The test campaign:** the C++ as the truth, Cramér–Rao bounds, Monte Carlo error-bar checks, and random rigs from
  scratch.

### 4.2 The model (`plate_model.py`)

- **It is pyOpenLPT's PINPLATE camera:**
  - a pinhole in air, with fx = fy = f, cx and cy, and OpenCV distortion;
  - a stack of plane-parallel layers: air, then window, then water, with any number of plates.
- **Projection:** the 1-D Newton of section 3.1. Points on the camera side of the window, or behind the camera, are
  reported as having no image.
  - Reading the C++ code, it measures those distances without their sign. It would then return the image of the
    point's mirror image in the window.
  - W0 checks this C++ quirk and records it. The C++ is not changed.
- **Back-projection:** closed form (vector Snell's law), for triangulation and the coarse fit.
- **Exact derivatives** with respect to every parameter:
  - the camera's rotation, centre, f and k1;
  - the window's normal and distance;
  - the 3-D point.
- **C++ cameras for checks and export** are built through `setPinplate*`, which takes a rotation vector. That vector
  comes from an accurate log map, for the reasons in section 3.3.

### 4.3 Unknowns and gauge

- **Per camera:**
  - rotation;
  - centre;
  - f;
  - k1, optional.

  cx and cy are fixed at the image centre, or at values you give. They are never fitted to refracted data: through
  oblique windows they run off the sensor (`2026-09-17-calibration.md` §2.2, and section 3.2 here).
- **Per window:**
  - **The unknowns:** its normal (2) and its distance (1), parameterised about the point where its cameras look
    through it. A tilt then pivots the plane where it is used, rather than about a distant origin.
  - **Known inputs:** the thickness and the indices, per window. That fixes the GUI's single shared set.
- **The wand frames:** a midpoint and a direction each, with the length exact.
  - Optionally the length is also free, as the check of the constants.
  - Through a flat window that check is much weaker than through the tube. Scaling every length leaves the image
    unchanged, so only the window's thickness fixes the scale (SymPy, section 3.1). W8 measures how weak it is.
- **The gauge:** camera 1's pose is held. That fixes the six rigid motions exactly, while the windows stay free.
- **The output frame:** the camFiles are written in the GUI's convention, after a rigid motion that changes no
  residual:
  - Z along window 1's normal;
  - the origin at the wand cloud's centroid;
  - with two windows, Y along their intersection.

### 4.4 Stages (`plate_wand.py`)

0. **Data.** Frames whose two ends are each seen by at least two cameras. Missing detections are allowed.
1. **Affine start** (the cylinder's): factorise the frames seen whole by every camera, and make it metric with the
   wand length. Both mirror images are kept.
   - **If too few frames are seen by every camera:** factorise the largest group of cameras that shares enough frames,
     and add the others by resection. This is new; the cylinder needed 12 frames seen by all cameras.
2. **Pinhole start.**
   - Each camera's distance along its principal ray comes from the perspective, by a 1-D search with the affine points
     held, and its f from its scale.
   - Then a pinhole bundle adjustment of all frames, with f free and the principal point fixed.
   - This is a complete pinhole wand calibration with no camera input, and could serve the GUI's Pinhole model too.
3. **Windows by search.** For each window, candidate normals and candidate distances:
   - the normals lie on a cap around the sign-free mean of its cameras' viewing directions (the eigenvector method of
     the 2026-09-17 report);
   - the distances are fractions of the way between the cameras and the wand cloud.

   For each candidate, every camera behind that window is refitted through the exact refractive model, with the
   stage-2 points held. That is a resection with 7 unknowns and exact derivatives, started from the small-angle
   conversion of its pinhole (section 3.1).
   - The points are then re-triangulated through the refitted cameras, and the resection is repeated once. The
     stage-2 points carry the pinhole's misfit, which reaches tens of pixels at steep views (section 3.2).
   - The candidate's score is the fitted residual.
   - Windows are searched one at a time.
   - The best candidates go on to a joint coarse fit, which minimises the back-projection residual with the points
     triangulated and a robust loss, over all cameras and windows.
   - The size of the cap and of the distance grid is set by W4's basin measurements.
4. **Maximum likelihood** (the cylinder's `refine`), with the window parameters added. Several starts are carried
   through when the data are noisy.
5. **Covariance, checks and files.**

### 4.5 Inputs and outputs (`plate_wand_io.py`)

- **Inputs:**
  - the GUI's wand CSV ("Export wand data"), where Small is end A and Large is end B;
  - or a `uv[frame, camera, end, (u, v)]` array;
  - the wand length;
  - the image sizes;
  - the windows, each with its thickness(es) and indices, and the camera-to-window mapping.
- **Optional hints:**
  - a focal-length range, or a prior per camera;
  - a camera-to-window distance with its uncertainty, a soft prior, for rigs where the data cannot fix it;
  - the principal point.
- **Outputs:**
  - PINPLATE camFiles in the C++ format, at full precision, which the C++ reloads (checked in W9);
  - a result JSON;
  - a text report with each camera's noise, error bars, the windows' σ, the rejected frames, and the predicted
    precision on unseen points.

### 4.6 Speed

Estimated for 4 cameras and 1,000 frames:
- the affine and pinhole starts: about 1 s;
- the window search: 5–20 s;
- the coarse fit: a few seconds;
- maximum likelihood with outlier rounds: a few seconds.

That is 15–60 s from scratch. W5 measures it.

### 4.7 What I expect to be hard

- **Steep views (rig D):** the pinhole start misses by tens of pixels there, so the window search has to bridge that.
  W4 and W5 show whether the search's cap is wide enough.
- **Square-on views (rig C):** the window's distance is nearly invisible. The fit must converge along that flat
  direction without drifting into a camera crossing its window. A step that makes a projection fail is rejected. The
  error bars should then show the distance's large uncertainty.
- **Long lenses (rig E):** f may be weakly determined, as it was through the tube for a ±8 mm cloud. W2 says whether
  the GUI's focal length should be used as a prior.
- **Rigs where no frame is seen by every camera:** the affine start needs the camera-group extension of stage 1, which
  is new.

## 5. Tests (W0–W10)

The tests are named W, because the current pipeline already calls its pinhole start "P0". The synthetic data come
from pyOpenLPT's C++ PINPLATE camera, never from the model being tested.

**The reference rigs (W2–W4):**
- **A, one wall:** 4 cameras about 0.6 m from an acrylic wall (12.7 mm), 15–20° off its normal, with a 100 mm wand
  cloud.
- **B, two walls:** 2 + 2 cameras through two perpendicular walls.
- **C, square-on:** 3 cameras within 5° of the normal (the worst case for the window's distance).
- **D, steep:** 3 cameras at 35–45° (strong refraction).
- **E, macro:** a long lens, f ≈ 40,000 px like cam1, through a flat window.

| test | what | criterion |
|---|---|---|
| W0 | `plate_model` against the C++: projections and back-projections. Random cameras, windows and points, including steep incidence, points near the camera's normal line and multi-plate windows. Plus the SymPy checks | projections within 1e-6 px; rays within 1e-10 (C++ tolerance 1e-12 mm); every SymPy step PROVED or explained |
| W1 | exact derivatives against central differences (camera, window, point) | relative error ≤ 1e-6 |
| W2 | Cramér–Rao on rigs A–E: what the wand determines (f, poses, window normal and distance, 3-D shape), at 0.1 px | reported. It sets the expectations, and shows whether a hint is needed for E or C |
| W3 | from the truth. Noise-free: the answer is the truth. Monte Carlo: 40 data sets per rig at 0.1 px, and a sweep from 0.02 to 2 px | noise-free RMS ≤ 1e-8 px; errors / Cramér–Rao 0.8–1.25; z standard deviation 0.9–1.1; no bias beyond 3 SE |
| W4 | basin of convergence of stages 3–4 from perturbed starts: normal 5–60° off, distance ±90 %, f ±40 %, rotations up to 20° | reported. It sizes the stage-3 search |
| W5 | from scratch, no hints: 50 random rigs at 0.1 px (2–6 cameras, 1–3 windows, 0–45° incidence, 0.2–2 m, f 1,500–40,000 px, wand clouds 30–300 mm) and 10 at 1 px | ≥ 49/50 and 10/10 reach the minimum of the truth-started fit (the same RMS, every parameter within 0.01 σ); median time ≤ 60 s for 4 cameras × 1,000 frames |
| W6 | **the GUI's current calibration on the same data**, run headless through `RefractiveWandCalibrator.calibrate` with the true focal lengths. Rigs A–D at 0.1 and 0.5 px, and 10 of W5's random rigs | reported side by side: time, failures, f, 3-D shape error, held-out wand lengths, the error on unseen points |
| W7 | bad data: gross outliers, swapped ends, a camera missing from many frames, a camera 5× noisier | the answer within 1 σ of the clean data's; error bars still honest |
| W8 | wrong constants: thickness 10 % off, water index 0.002 off, wand length 0.1 % off, an unmodelled k1 | reported: what the residuals, the free length and the error bars show |
| W9 | the files: camFiles reload in the C++ and project like the fit; the CSV reader reproduces the detections | within 1e-6 px; identical |
| W10 | **real data, if you send a data set:** the new and current calibrations compared | reported: residuals, wand-length errors, held-out frames |

Plus a pytest subset, `tests/test_plate_wand.py`. It needs a `.gitignore` negation, like `tests/cylinder_wand`.

## 6. Inputs to confirm (defaults in brackets)

1. **A real flat-window wand data set?** [none: synthetic only]
   - What is needed: the GUI's exported wand CSV, the window settings (number, mapping, thicknesses, indices), the
     wand length, and the camFiles from the current calibration.
   - It would let W10 compare the two calibrations on your data.
2. **Your typical rig:** cameras, windows, viewing angles, distances, lenses and wand length. [rigs A–E: 1280 × 800 px
   sensors, a 25 mm wand in a 100 mm cloud, 1,000 frames, 0.1 px noise]
3. **What is held fixed:** [the thickness and the indices, given per window; cx and cy at the image centre; k1
   optional, off by default]
4. **The GUI's focal-length input:** [not used; an optional prior only where W2 shows f is weakly determined, e.g.
   rig E]
5. **GUI integration now?** [no: a separate step after the tests, keeping the current path as an option]
6. **The `cyl_model.py` rotation fix of section 3.3?** [yes, as its own two-line change, with the cylinder pytest
   subset rerun]

## 7. Deliverables and cost

- **Code** (new), in `modules/camera_calibration/curved_calibration/`, beside the cylinder code it reuses:
  - `plate_model.py`;
  - `plate_wand.py`;
  - `plate_wand_io.py`.
- **Tests:** this folder (the scripts, their outputs, the report and a summary README), plus `tests/test_plate_wand.py`.
- **Report:** `2026-10-0X-plate-wand-report.md`.
- **The GUI** (a separate step for your approval):
  - `view.py`'s refractive path would call the new calibration and write the camFiles where the GUI expects them;
  - the current path stays as an option.
- **Cost:**
  - about a day of work;
  - computing: W5 about 30 min and W6 1–3 h in parallel (the current pipeline is the slow part); the rest takes
    minutes.
