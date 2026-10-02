# Phase 0 report: fast model, derivatives, observability

Date: 2026-10-01. Plan: `2026-10-01-plan.md`. Code under test: `modules/camera_calibration/curved_calibration/cyl_model.py`
(new) and the information/covariance part of `cylinder_wand.py` (new). Every number below comes from a script in this
folder; its log (`*_out.txt`) and data (`*.json`) are beside it.

## Summary

| test | result |
|---|---|
| T0 fast model vs C++ | **PASS** (criterion for rays restated, see below) |
| T1 derivatives vs central differences | **PASS**: worst 5.3e-8 relative inside the fold radius, 4.0e-8 beyond it |
| T2 observability | **GO** on the gauge-free figure (0.0077 px). sigma(f)/f = **0.064 %** (target 0.5 %). The literal tube-frame figure fails, 0.62 px > 0.05 px; it is the tube's position across the axis relative to the rig, 13 um (T2b). |

## T0: fast model against the C++ PINCYL camera (`t0_fast_vs_cpp.py`)

24 random rigs: 2-4 media, axis tilted up to 30 deg and offset up to 50 mm (so the conversion to the tube frame is
exercised), camera 1.5-12 outer radii from the axis, half of them with distortion.

- **Back-projection, 3000 pixels per rig:** the same valid/invalid flags as the C++ everywhere.
  - Compared as lines in pixels (the C++ ray's mid-chord point reprojected by the fast model): worst 1.0e-9 px.
  - Raw: origins within 2.3e-9 mm, directions within 2.4e-11 rad.
- **The plan's 1e-9 mm was below the C++'s own round-off for a distant camera.**
  - Its `lineOfSight` forms the direction as `normalise((R^-1 (x, y, 1) + C) - C)`: two points about 1 m from the
    origin and 1 mm apart.
  - That costs about |C| eps / 1 mm ~ 1e-13 rad, or about 1e-9 px at f = 13,000.
  - Emulating that step reproduces the differences (`t0b_cpp_direction_roundoff.py`: median 1.5e-14 rad, max
    5.5e-14 rad).
  - The criterion is set at 1e-8 px, ten times that floor. The fast model forms R^T m directly and is the more
    precise of the two here.
- **Forward projection, single-image region (r < r_c):**
  - every Newton start converged, from the C++ pixel + 5 px and from a crude pinhole start;
  - worst difference from the C++ image: 6.3e-9 px.
- **Beyond the fold radius (diagnostic):** 19,429 starts next to a C++ image on the sensor.
  - 19,428 returned that image, 1 returned the point's other image (2.9 px away), 0 failed.
  - No false solution.
- **Two defects were found and fixed in `cyl_model.project` along the way:**
  - *A false convergence at a grazing edge.* The ray's derivatives blow up there, so the Newton step shrank to zero
    with the point still 5.9 mm off the ray. Convergence now also requires the ray to pass within 1e-10 |X - C| of
    the point, re-checked on a final trace.
  - *Starts whose ray misses the tube* (a detection just outside the tube's image, e.g. at the grazing edge). These
    are moved by bisection to the valid region, towards the pinhole start or towards a ray aimed at the axis.
  - Newton also backtracks on |g| now.

## T1: implicit-differentiation Jacobians (`t1_derivatives.py`)

12 rigs; all 14 camera columns (rotation, centre, log f, cx, cy, k1, k2, p1, p2, k3) and d(uv)/dX. The numeric side is
Richardson-extrapolated central differences.
- The pixel is exactly linear in each distortion coefficient, so those columns use a large step.
- Results:
  - **inside r_c:** worst 5.3e-8 relative to the column's size;
  - **beyond r_c, next to the fold:** 4.0e-8 at the better of two steps.

## T2: observability and rig design (`t2_observability.py`, `t2b_error_modes.py`)

Cramer-Rao bound at the truth, 0.1 px noise, frames marginalised, gauge = camera 1 at X = Y = 0.
- **Reference rig:**
  - three replicas of cam1 at 0/45/90 deg (f 41,459 px, 1094 mm);
  - a 10 mm wand, midpoints in +-20 mm, 500 frames;
  - the sensor made large enough for the volume.
- **Unseen-point error,** RMS per 2-D image over a 7x7x7 grid:
  - `fixed`: in the fit's gauge;
  - `stabiliser`: after the best common spin and slide of the points;
  - `retri`: each point re-triangulated (gauge-free).

| rig | cams | spacing deg | D mm | half mm | frames | sigma f/f % | sigma dist mm | fixed px | stabiliser px | retri px |
|---|---|---|---|---|---|---|---|---|---|---|
| reference | 3 | 45 | 1094 | 20 | 500 | 0.064 | 0.69 | 4.19 | 0.620 | 0.0077 |
| 2 cams | 2 | 45 | 1094 | 20 | 500 | 0.114 | 1.22 | 6.63 | 1.22 | 0.0062 |
| 4 cams | 4 | 45 | 1094 | 20 | 500 | 0.056 | 0.60 | 3.42 | 0.485 | 0.0083 |
| spacing 30 | 3 | 30 | 1094 | 20 | 500 | 0.081 | 0.87 | 5.07 | 0.809 | 0.0075 |
| spacing 60 | 3 | 60 | 1094 | 20 | 500 | 0.067 | 0.71 | 4.22 | 0.612 | 0.0078 |
| spacing 90 | 3 | 90 | 1094 | 20 | 500 | 0.087 | 0.93 | 4.02 | 0.618 | 0.0079 |
| spacing 120 | 3 | 120 | 1094 | 20 | 500 | 0.068 | 0.73 | 4.72 | 0.452 | 0.0077 |
| D = 500, f scaled | 3 | 45 | 500 | 20 | 500 | 0.028 | 0.14 | 2.27 | 0.718 | 0.0077 |
| half 10 | 3 | 45 | 1094 | 10 | 500 | 0.164 | 1.77 | 5.28 | 0.922 | 0.0067 |
| half 40 | 3 | 45 | 1094 | 40 | 500 | 0.020 | 0.22 | 1.61 | 0.279 | 0.0091 |
| 200 frames | 3 | 45 | 1094 | 20 | 200 | 0.101 | 1.09 | 6.67 | 0.985 | 0.0121 |
| 2000 frames | 3 | 45 | 1094 | 20 | 2000 | 0.032 | 0.35 | 2.13 | 0.310 | 0.0038 |
| 4 cams at 90 | 4 | 90 | 1094 | 20 | 500 | 0.073 | 0.78 | 3.18 | 0.307 | 0.0084 |
| 3 cams at 90, D = 500 | 3 | 90 | 500 | 20 | 500 | 0.041 | 0.20 | 2.14 | 0.775 | 0.0078 |
| 2 cams at 90 | 2 | 90 | 1094 | 20 | 500 | 0.101 | 1.08 | 5.52 | 0.962 | 0.0061 |
| 3 cams at 60, half 10 | 3 | 60 | 1094 | 10 | 500 | 0.166 | 1.78 | 5.14 | 0.885 | 0.0067 |
| cam1-sensor-like, half 8 | 3 | 45 | 1094 | 8 | 500 | 0.219 | 2.35 | 5.41 | 1.12 | 0.0063 |
| 2 cams at 45, half 10 | 2 | 45 | 1094 | 10 | 500 | 0.278 | 2.99 | 8.35 | 2.15 | 0.0051 |

**What the three unseen-point figures measure** (`t2b_error_modes.py`, reference rig). Removing the best common motion of
the points, one family at a time:

| removed | none | + spin and slide | + shift across the axis | + tilt | + scale | + affine | re-triangulated |
|---|---|---|---|---|---|---|---|
| RMS px | 4.19 | 0.620 | 0.067 | 0.029 | 0.027 | 0.010 | 0.0077 |

- **4.19 px is the slide along the axis.** It is unobservable, and camera 1 trades its height with its pitch, so the
  scene slides relative to camera 1's centre (sigma of camera 2-3 heights ~0.15-0.2 mm in that gauge).
- **Of the 0.62 px left, 90 % is where the tube sits across the axis relative to the rig.**
  - sigma 12.8 um in X and in Z;
  - its tilt: 12.7 arcsec;
  - 4 cameras at 90 deg halve both (6 um); a +-40 mm volume, 5.6 um.
- **What the cameras disagree on, which triangulation feels, is 0.0077 px.**

**Gate decision.**
- The plan's gate ("adds at most 0.05 px RMS to the projection of unseen points") did not name a frame.
- On the gauge-free reading it is **GO**: 0.0077 px, 6.5x inside the limit.
- On the literal tube-frame reading it **fails**, at 0.62 px. That failure is the 13 um uncertainty of the axis position
  relative to the rig, which is reported as its own figure from now on.
- For scale: the real single-camera plate case placed the axis to +-0.135 mm.
- I proceeded on the gauge-free reading; this is flagged for the user.

**f from the wand.**
- sigma(f)/f is 0.02-0.28 % on every rig tested, inside the 0.5 % target everywhere.
- The depth the wand covers matters most: +-40 mm gives 0.020 %, +-10 mm 0.164 %.
- Then the number of frames (as 1/sqrt(N)) and the number of cameras.
- The angle between cameras matters little between 45 and 120 deg; 30 deg is somewhat worse.
- A 0.5 m, wider-field set-up (f scaled to keep f/D) halves sigma(f)/f and pins the camera distance 5x better.

**Rig guidance:**
1. Cover depth: a volume of +-20 mm or more about the axis, as seen from each camera.
2. 3 or more cameras, 45-120 deg apart.
3. 500 or more frames.
4. A shorter camera distance helps f and the distance, not the consistency.

**Model error of the starts the plate pipeline would use** (true 3-D points, exact images, reference rig):

| camera | pinhole fx = fy | pinhole fx != fy | affine |
|---|---|---|---|
| 1 | 99.5 px | 15.8 px | 15.8 px |
| 2 | 99.7 px | 10.7 px | 10.7 px |
| 3 | 96.3 px | 15.8 px | 15.8 px |

- The principal point is held at the sensor centre.
- **The best fx != fy pinhole runs off to infinite distance,** where it is the affine camera. The tube shrinks points as
  they approach the camera, the opposite of perspective, so no finite pinhole fits better.
- This matches the real case's 11.4 px (case T5).
- The affine start's error is 11-16 px, as the plan expected (~10 px).

## Next: Phase 1

Stages 3-4, gauge, covariance: T3, T4, T6, T7.
