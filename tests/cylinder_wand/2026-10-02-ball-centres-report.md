# Wand balls seen through the tube: their centres (report, T13)

Date: 2026-10-02. Plan: `2026-10-02-ball-centres-plan.md` (approved: "go ahead and do the test"). The inputs of the
plan's section 3 were not given, so its defaults were used: 1.0 and 1.5 mm balls, 10 mm apart on a 0.3 mm rod; all
four lightings; the blur the plate case measured on the real camera.

**The question (the user):** a ball seen through the tube is not imaged as a circle. How does that affect the
measured centre, which the wand calibration uses, and is there a way to help?

Nothing in the library, the GUI, the C++, the case folder or `paper/` was changed. Tests and outputs are in this
folder.

## 1. Result

1. **The shape itself does not move the centre.**
   - Through the tube a ball's image is an ellipse 1.26–1.34 times wider than tall.
   - The centre of its exact silhouette is within 0.002 px of the image of the ball's centre, for balls up to 2 mm
     in the ±8 mm volume (pre-plan sizing).
   - The renderer used below reproduces this to 0.0005 px (T13a).
2. **pyOpenLPT's wand detector cannot measure these images.**
   - It fits circles. On perfect ellipses without the tube (T13a, 2b):
     - its "fast" mode (a RANSAC circle on Canny edges) holds up to 1.15 : 1 and fails from 1.25 : 1, missing the
       centre by 3–13 px along the long axis;
     - its "reliable" mode (the C++ CircleIdentifier) fails from 1.15–1.2 : 1. It returns two circles near the
       two ends of each ellipse (T13a2).
   - On rendered wand images through the tube (T13b):
     - "fast" put both balls within 5 px of the truth in only 16–58 of 1142 camera-frames. The detection nearest
       a ball was typically 5–7 px off, and up to 20 px.
     - "reliable" measured none of 286 (typically 7–8 px off).
3. **A thresholded-blob centroid is pulled by the rod.**
   - Backlit, the rod joins the two balls into one blob.
   - With the rod removed by an opening, the error is still 0.43 px RMS backlit, and 1.0–1.6 px in front light,
     where the dark rod hides part of a bright ball.
4. **The plate case's ellipse estimator works once the rod is masked.**
   - **The estimator:** edge points at half the local edge height along 96 rays, a robust ellipse fit, and the edge
     points in the rod's strip left out ("ellipse_rodmask").
   - **Best focus, backlit:** bias ≤ 0.001 px, RMS 0.026 px, largest error 0.14 px.
   - **Best focus, evenly front-lit:** bias ≤ 0.0004 px, RMS 0.028 px, largest error 0.16 px.
   - **Defocused:** 0.04–0.06 px RMS.
   - **Without the rod mask:** 0.07 px RMS backlit, and occasional errors up to 4.9 px in front light.
   - **With overlapping balls:** a guard that drops a camera's pair when the two fitted ellipses come within 3 px
     returns no wrong centre, and drops no good pair (T13b2).
5. **Uneven lighting is now the largest error.**
   - **One side 35 % brighter:** the centres shift by about 0.2 px. Most of it is a constant shift per camera, the
     same for both balls: its spread is 0.02–0.06 px.
   - **A specular highlight:** 0.1–0.3 px, with more scatter.
6. **What the calibration makes of it (T13c).**
   - With pyOpenLPT's detector or with blob centroids, the calibration comes out wrong by percent to tens of percent
     in f.
   - With the rod-masked ellipse fit, in backlit or even light, it comes close to the calibration from exact
     centres:
     - f within 0.07 % (baseline 0.06 % RMS);
     - 3-D shape 0.12–0.13 µm (baseline 0.07);
     - an unbiased held-out wand length.
   - A small systematic remains: f +0.03 to +0.07 % (t up to 3.5) and error bars 1.5× too small. So the literal
     criteria fail narrowly, and T13d tries a model-based correction.
   - Uneven lighting harms the calibration even with this detector: f up to 0.1 % off with one-sided light, and up
     to 0.56 % with a highlight.
7. **A model-based correction removes what is left (T13d).**
   - **The method:** render the image the fitted calibration predicts for every detection, measure it with the same
     detector, subtract that offset, and refit.
   - **f bias:** t falls from up to 3.5 to at most 1.6.
   - **3-D shape error:** 0.08–0.10 µm (baseline 0.07).
   - **Error bars:** z falls from 1.5–1.6 to 1.22 ± 0.11 backlit and 1.30 ± 0.12 even (baseline 1.11 ± 0.11).
   - **Assumption:** the blur and the lighting must be known to the model. In the test they were known exactly.
   - **Literal verdict:** z still misses its limit of 1.2, by 0.02 and 0.10. That limit was set tighter than z's own
     standard error with 8 data sets (section 4).
8. **The answer to the question:**
   - the ball's distorted shape is harmless;
   - the detector is what matters;
   - fit ellipses with the rod masked;
   - light the balls evenly or from behind.

   - **With that:** f lands within 0.07 % of the truth, against 0.06 % RMS for a calibration from exact centres.
   - **The model-based correction** closes even that gap.

## 2. What was run

| script | what | outputs |
|---|---|---|
| `t13_pre_sizing.py` | before the plan: the exact silhouette's centre against the image of the ball's centre | `t13_pre_sizing_out.txt`, `.json` |
| `t13_common.py` | the renderer (exact refracted rays, supersampled outlines, four lightings, the plate case's astigmatic blur, noise), the detectors, the scoring | — |
| `t13_ellipse.py` | the plate case's estimator: six functions copied verbatim from `cases/cyl_20260930_T0_cam1/dots.py`, plus `detect_balls` for wand balls | — |
| `t13a_render.py` | the renderer against the exact silhouette; the detectors on perfect circles and ellipses without the tube; example images | `t13a_render_out.txt`, `.json`, `t13a_examples/` |
| `t13a2_circle_split.py` | what pyOpenLPT's detector returns for ellipses of growing aspect ratio (diagnostic) | `t13a2_circle_split_out.txt` |
| `t13b_detectors.py` | six detectors on 1164 rendered camera-frames × 4 lightings × 2 focus settings, with and without noise | `t13b_detectors_out.txt`, `.json`, `_rows.npz` |
| `t13b2_overlap_guard.py` | the overlap guard for the best detector | `t13b2_overlap_guard_out.txt`, `.json` |
| `t13b3_symmetric.py` | after T13c: a symmetric rod mask (96 and 192 rays) against the one-sided one | `t13b3_symmetric_out.txt`, `.json` |
| `t13c_calibration.py` | 8 data sets × 600 frames, five detectors × four lightings, calibrated from scratch and measured against the truth | `t13c_calibration_out.txt`, `.json`, `_uv.npz` |
| `t13d_correction.py` | the model-based correction of the rod-masked ellipse fit's remaining error, backlit and even | `t13d_correction_out.txt`, `.json` |
| `t13d_zscore_bootstrap.py` | bootstrap standard errors of z's RMS over the 8 data sets (T13c, T13d) | `t13d_zscore_bootstrap_out.txt` |

**The rendering (`t13_common.py`):**
- **Rig:** three replicas of cam1 at 0, 45 and 90°, with f 41,459 px, 1094 mm from the axis, 1280 × 800 px sensors
  (cam1's Phantom), and the case's tube.
- **Geometry:** a (sub)pixel shows the object its refracted ray meets first, a ball or the rod. Pixels on an
  outline are supersampled 8 × 8.
- **Lighting:**
  - backlit: a dark silhouette on a bright background;
  - even: matte balls evenly lit, on a dark background, with a dark rod;
  - side: as even, with one side 35 % brighter;
  - highlight: as even, plus a specular spot.
- **Blur:** a Gaussian per ball, fitted exactly through the three blur states the plate case measured on the real
  camera. Across the tube's axis σ is 4.7, 2.8 and 3.8 px; along it, 1.5, 2.0 and 4.6 px.
  - "Best focus" puts the volume between the tube's two foci: σ 2.7–3.7 px across and 1.2–2.3 px along.
  - "Defocused" puts it 22 mm further along that traverse: up to 4.6 px along the axis.
- **Noise and output:** 2 grey levels of noise, as in the plate case; 8-bit images.
- Example images are in `t13a_examples/`.

**The detectors:**
- **pyOpenLPT's wand detector**, imported unchanged:
  - "fast" mode, with its C++ fallback;
  - "reliable" mode.
  - Radius range 10–60 px, sensitivity 0.85 (the GUI's default).
- **Thresholded-blob centroids**, with and without an opening that removes the rod.
- **The plate case's ellipse estimator**, with and without the rod's strip masked.

## 3. Results by test

### Pre-plan sizing (`t13_pre_sizing.py`)

The exact silhouette's centroid (or the centre of an ellipse fitted to it), minus the image of the ball's centre, at
its largest over a 5 × 5 × 5 grid and the three cameras:

| ball | ±8 mm | ±20 mm |
|---|---|---|
| 1 mm | 0.0005 px | 0.003 px |
| 2 mm | 0.002 px | 0.011 px |
| 3 mm | 0.0045 px | 0.025 px |

- The offset grows exactly as the square of the ball's radius, and is almost all across the axis.
- The shape is an ellipse, 1.26–1.34 times wider than tall over ±8 mm.

### T13a: the renderer, and the detectors without the tube

- **The renderer against the exact silhouette** (36 balls, no rod, no blur, no noise): the rendered centroid's
  error at each supersampling factor is

  | SS | RMS | largest |
  |---|---|---|
  | 4 | 0.0053 px | 0.013 px |
  | 8 | 0.0018 px | 0.0042 px |
  | 16 | 0.0008 px | 0.0015 px |
  | 32 | 0.0003 px | 0.0005 px |

  **Pass** at SS 32 (limit 1e-3 px). The production SS 8 leaves 0.0018 px RMS, the floor of everything below.
- **Pixel convention:** on perfect discs, every detector's mean error is within 0.021 px, so none has a convention
  offset. Their RMS is:
  - ellipse estimator: 0.0014 px;
  - pyOpenLPT "fast": 0.04 px;
  - blob: 0.04–0.05 px;
  - blob with the opening: 0.08 px;
  - pyOpenLPT "reliable": 0.12 px.
- **On perfect 1.3 : 1 ellipses:** both pyOpenLPT modes fail on all 20 images. The others are unchanged.
- **The aspect-ratio sweep** (median distance to the nearest detection, 20 balls each):

  | a / b | "fast" | "reliable" | blob | ellipse |
  |---|---|---|---|---|
  | 1.00 | 0.03 px | 0.09 px | 0.05 px | 0.00 px |
  | 1.15 | 0.07 px | 0.41 px (1 image failed) | 0.04 px | 0.00 px |
  | 1.20 | 0.13 px (1 failed) | 5.2 px (all failed) | 0.03 px | 0.00 px |
  | 1.25 | 3.4 px (all failed) | 7.7 px | 0.04 px | 0.00 px |
  | 1.30 | 6.3 px | 9.3 px | 0.04 px | 0.00 px |
  | 1.40 | 8.1 px | 13.0 px | 0.04 px | 0.00 px |

  The tube gives 1.26–1.34.

### T13b: the detectors on rendered tube images

1164 camera-frames, of which 22 are crowded (the balls' images within 10 px of each other) and 17 overlap. The table
is for the 1142 uncrowded ones at best focus. "Measured" means both balls within 5 px, with the right labels; the
bias and RMS are over the measured ones, with noise.

| detector | backlit: measured, bias, RMS, max | even: measured, bias, RMS, max | side: bias, RMS | highlight: bias, RMS |
|---|---|---|---|---|
| pyOpenLPT fast | 16 / 1142 (median 5.7 px off) | 34 / 1142 (6.9 px off) | — | — |
| pyOpenLPT reliable (1 frame in 4) | 0 / 286 (8.2 px off) | 0 / 286 (7.4 px off) | — | — |
| blob | 0 (one blob: the rod) | 1142; (−0.02, 0.00); 1.03; 3.2 px | (0.17, −0.31); 1.09 | (−0.02, 0.00); 1.03 |
| blob, rod opened | 1142; (0.00, 0.00); 0.43; 0.82 px | 927; (−0.03, 0.03); 1.56; 5.0 px | (0.13, −0.28); 1.60 | (−0.03, 0.04); 1.55 |
| ellipse | 1142; (0.001, 0.000); 0.073; 0.24 px | 1089; (0.001, 0.001); 0.151; 4.9 px | (0.10, −0.17); 0.23 | (0.03, 0.15); 0.28 |
| **ellipse, rod masked** | **1142; (−0.001, −0.000); 0.026; 0.14 px** | **1103; (0.000, −0.000); 0.028; 0.16 px** | (0.10, −0.17); 0.22 | (0.03, 0.14); 0.32 |

- **Systematic and random parts** of the masked ellipse estimator (the noise-free image gives the systematic part):
  - backlit: systematic 0.010 px RMS, random 0.024 px;
  - even: 0.013 and 0.026 px.
  - Its error percentiles, backlit and even: median 0.014–0.016 px, 99.9th 0.125–0.127 px, largest 0.14–0.16 px.
  - Unmasked, the even lighting had 12 errors above 0.5 px, none at a crowded frame. The rod mask removes them
    all.
- **Defocused:**

  | detector | backlit RMS | even RMS |
  |---|---|---|
  | ellipse, rod masked | 0.043 px | 0.061 px |
  | ellipse | 0.18 px | 0.12 px |
  | blob, rod opened | 0.71 px | 1.5 px |

  pyOpenLPT "fast" is as before.
- **Front-lit misses:** the masked ellipse estimator missed about 3 % of camera-frames in front light (none
  backlit), where the rod in front of a ball breaks its blob.
- **The lighting bias, masked ellipse estimator:**
  - **Side light:** each camera sees its own nearly constant shift, the same for both balls:
    - cam1 (0.16, −0.15) px;
    - cam2 (0.13, −0.18) px;
    - cam3 (0.00, −0.19) px.
    - The spread about that shift is 0.02–0.06 px, and ball B minus ball A is within 0.02 px.
  - **Highlight:** the shifts reach 0.22 px, with a spread of up to 0.26 px.
  - **Backlit and even:** every camera's mean is within 0.001 px.
- **Literal verdict:** every detector fails at least one of T13b's criteria (section 4).
- **Time per 1280 × 800 image:**

  | detector | time |
  |---|---|
  | pyOpenLPT fast | 0.23 s |
  | pyOpenLPT reliable (on a crop) | 0.82 s; about 6 s on a full frame |
  | blob | 0.005 s |
  | blob, rod opened | 0.017 s |
  | ellipse | 0.26 s |
  | ellipse, rod masked | 0.27 s |

### T13b2: a guard against overlapping balls

- **The guard:** a camera's pair is dropped when the two fitted ellipses come within 3 px of each other.
- **The data:** T13b's 22 crowded camera-frames, of which 17 overlap, and 150 uncrowded ones, re-rendered with new
  noise.
- **Wrong centres for overlapping balls, without and with the guard:**

  | lighting | without the guard | with it |
  |---|---|---|
  | backlit | 0 | 0 |
  | even | 3 | 0 |
  | side | 2 | 0 |
  | highlight | 3 | 0 |

- **Good pairs dropped:** none of the 150 uncrowded pairs, in all four lightings.
- **Pass.**

### T13c: effect on the calibration

8 data sets of 500 frames (plus 100 held out) on the cam1-like rig, at best focus. Each detector's two centres go
straight into a calibration from scratch, labelled by size, with no knowledge of the truth.
- **The start:** the baseline calibration from scratch reached the minimum of the one started at the true cameras in
  8 of 8 data sets (RMS within 2e-9 px, f within 8e-8). The start works on this small-field rig.
- **The centres the detectors hand over** (23,314 visible balls per detector and lighting):

  | detector | share measured | share more than 1 px off | share more than 5 px off |
  |---|---|---|---|
  | pyOpenLPT fast | 100 % | 86–93 % | 57–65 % |
  | blob, backlit | 0 | — | — |
  | blob, front-lit | 99 % | 26–28 % | — |
  | blob, rod opened, backlit | 99 % | 0 | — |
  | blob, rod opened, front-lit | 96 % | 32 % | 8 % |
  | ellipse | 96–99 % | up to 1 % | 0.6 % |
  | ellipse, rod masked | 96–99 % | 0.00–0.02 % | 0.00–0.01 % |

- **The calibrations** (f: mean error per camera with its t over the 8 data sets; "length": the held-out wand length
  from the detector's centres; z: the error over the fit's own σ):

| centres | f mean (t) | length mean (t), RMS | 3-D shape / relative to the tube | z RMS | fit RMS, rejected |
|---|---|---|---|---|---|
| ideal: exact + 0.02 px noise | +0.003, −0.021, −0.019 % (t ≤ 1.1) | 0.00 µm (t 0.0), 0.50 µm | 0.07 / 9.4 µm | 1.11 | 0.021 px, 0 % |
| pyOpenLPT fast, backlit | −63 % in every camera | —, 145 µm | 10.6 mm / 37 mm | 558 | 3.7 px, 1.1 % |
| pyOpenLPT fast, front-lit | wrong by 36–87 % RMS | —, 200–220 µm | 5–7 mm / 20–28 mm | 234–335 | 4.7–4.9 px, 2–3 % |
| blob, backlit | all 8 failed: the rod joins the balls, so no frame is seen whole | | | | |
| blob, front-lit | +6.4, −0.5, +5.4 % (t up to 14) | +4.0 µm (t 6.8), 14 µm | 22 µm / 0.37 mm | 2.7 | 0.38 px, 11.5 % |
| blob, rod opened, backlit | +0.77, +0.38, +0.98 % (t 4.6–5.8) | −0.25 µm (t −2.3), 4.0 µm | 21 µm / 0.66 mm | 9.3 | 0.145 px, 0 % |
| blob, rod opened, front-lit | −68 % (a wrong minimum) | —, 96 µm | 10.6 mm / 38 mm | ~1050 | 2.2 px, 7–8 % |
| ellipse, backlit | +0.25, +0.06, +0.28 % (t up to 10.6) | −0.13 µm (t −6.2), 0.71 µm | 2.9 µm / 129 µm | 8.8 | 0.026 px, 3.6 % |
| ellipse, even | −0.17, +0.14, −0.23 % (t up to 23) | +1.35 µm (t 2.5), 9.9 µm | 1.3 µm / 79 µm | 4.0 | 0.024 px, 16.4 % |
| ellipse, highlight | +0.29, +0.33, +0.33 % | +1.23 µm (t 3.4), 10.6 µm | 3.5 µm / 194 µm | 6.6 | 0.073 px, 8.3 % |
| **ellipse, rod masked, backlit** | +0.048, +0.032, +0.042 % (t 3.1, 1.9, 2.7) | −0.01 µm (t −0.2), 0.64 µm | 0.13 / 8.0 µm | 1.51 | 0.017 px, 1.6 % |
| **ellipse, rod masked, even** | +0.055, +0.067, +0.031 % (t 1.9, 3.5, 1.8) | −0.05 µm (t −2.1), 0.67 µm | 0.12 / 10.3 µm | 1.61 | 0.017 px, 1.8 % |
| ellipse, rod masked, side | +0.058, −0.048, −0.105 % (t up to 5.6) | +0.02 µm (t 0.6), 1.12 µm | 0.39 / 16.9 µm | 2.77 | 0.028 px, 2.9 % |
| ellipse, rod masked, highlight | +0.10, +0.56, +0.55 % (t up to 9.6) | −0.09 µm (t −0.9), 2.66 µm | 0.95 / 75 µm | 2.45 | 0.157 px, 0.3 % |

- **pyOpenLPT's detector, and blob centroids, ruin the calibration.** f is off by percent to tens of percent, and
  3-D shapes by tens of µm to millimetres. The calibration converges anyway, to a wrong answer with large residuals
  (0.1–4.9 px).
- **The plain ellipse fit:** the rod's pull leaves f 0.1–0.3 % off and the error bars 4–9× too small.
- **The rod-masked ellipse fit comes close to the baseline,** in backlit and even light:
  - f within 0.07 %, against 0.06 % RMS for the baseline;
  - 3-D shape 0.12–0.13 µm (baseline 0.07);
  - the held-out wand length unbiased (t −0.2 and −2.1), RMS 0.64–0.67 µm (baseline 0.50).
  - What remains is small but measurable: f +0.03–0.07 % (t up to 3.5), and error bars 1.5× too small (z 1.5–1.6
    against 1.11).
  - It comes from the detector's systematic error of about 0.01 px (T13b). T13b3 traced it to the one-sided rod
    mask: a symmetric mask cuts it to 0.003–0.005 px, but leaves too few edge points on the small ball.
  - The calibration rejects 1.6–1.8 % of the centres. That matches the detector's tail: T13b's 99th percentile,
    0.085–0.091 px, sits at the 5σ limit of about 5 × 0.017 px. The overlapping pairs add to it, since T13c ran
    without T13b2's guard.
- **Uneven lighting harms the calibration even with the best detector:**
  - one-sided light: f off by up to 0.1 % (t 5.6), 3-D shape 0.39 µm, z 2.8;
  - a highlight: f off by up to 0.56 % (t 9.6), 3-D shape 0.95 µm, z 2.5.
- **Literal verdict:** no detector and lighting passes all of T13c's criteria. The rod-masked ellipse fit fails
  narrowly in backlit and even light (t 3.1 and 3.5 against 3; z 1.5 and 1.6 against 1.2). That triggers T13d.

### T13b3: a symmetric rod mask (after T13c)

The first 150 of T13b's frames, best focus. The systematic part is from the noise-free image, over the measured
camera-frames.

| lighting | detector | measured | systematic RMS | random RMS |
|---|---|---|---|---|
| backlit | ellipse, rod masked (one-sided) | 423 / 423 | 0.0105 px | 0.0245 px |
| backlit | symmetric mask, 96 rays | 98 / 423 | 0.0034 px | 0.0193 px |
| backlit | symmetric mask, 192 rays | 423 / 423 | 0.0069 px | 0.0288 px |
| even | one-sided | 408 / 423 | 0.0134 px | 0.0231 px |
| even | symmetric, 96 rays | 102 / 423 | 0.0053 px | 0.0184 px |
| even | symmetric, 192 rays | 389 / 423 | 0.0085 px | 0.0258 px |

- **The cause is confirmed.** Leaving out the far strip as well makes the points symmetric, and cuts the systematic
  error by a factor of 3. The cause was the one-sided mask, under which the blurred outline's angle-dependent inward
  shift moves the fitted centre.
- **But it is not a usable fix.**
  - With 96 rays, the two 18 px strips leave too few points on the 1 mm ball: three quarters of the frames go
    unmeasured.
  - With 192 rays, the systematic part falls by a third, and the random part rises by about as much.
- **Lighting is unchanged:** the side light and highlight biases stay as they were.

### T13d: a model-based correction

T13c's 8 data sets and rod-masked ellipse centres, backlit and even. The steps:
1. Calibrate on the measured centres.
2. For every detection, render the noise-free image the fit predicts:
   - the fitted camera;
   - the fitted wand ends, or the held-out ends triangulated with the fitted cameras;
   - the balls and rod;
   - the lighting and the blur model.
3. Run the same detector on that image; its offset from the projected centre is the predicted error.
4. Subtract the predicted errors from the measured centres, and refit.

- **Predicted errors:**
  - backlit: RMS 0.0095 px over 13,798 camera-frames (1 not predicted);
  - even: RMS 0.018 px over 13,308 (19 not predicted).
  - They are comparable to the detector's systematic part in T13b (0.010 and 0.013 px over the uncrowded frames).
  - What is left after the correction, 0.025 and 0.026 px RMS, is about the random part.
- **The calibrations:**

| centres | f mean (t) | f RMS | length mean (t), RMS | 3-D shape / tube | z RMS ± SE |
|---|---|---|---|---|---|
| ideal (T13c) | +0.003, −0.021, −0.019 % (t ≤ 1.1) | 0.05–0.06 % | 0.00 µm (t 0.0), 0.50 µm | 0.07 / 9.4 µm | 1.11 ± 0.11 |
| rod masked, backlit (T13c) | +0.048, +0.032, +0.042 % (t 3.1, 1.9, 2.7) | 0.06 % | −0.01 µm (t −0.2), 0.64 µm | 0.13 / 8.0 µm | 1.51 ± 0.11 |
| **… corrected** | −0.003, −0.012, −0.020 % (t −0.2, −0.8, −1.5) | 0.04–0.05 % | −0.01 µm (t −0.3), 0.63 µm | 0.08 / 6.4 µm | **1.22 ± 0.11** |
| rod masked, even (T13c) | +0.055, +0.067, +0.031 % (t 1.9, 3.5, 1.8) | 0.05–0.09 % | −0.05 µm (t −2.1), 0.67 µm | 0.12 / 10.3 µm | 1.61 ± 0.14 |
| **… corrected** | −0.016, +0.007, −0.028 % (t −0.7, +0.7, −1.6) | 0.03–0.06 % | −0.03 µm (t −1.7), 0.64 µm | 0.10 / 8.3 µm | **1.30 ± 0.12** |

- **What the correction removes:** the f bias (largest |t| 1.6) and most of the error bars' excess.
- **Backlit:** z is within 1 standard error of the baseline's, and 2.0 of its own standard errors above 1.
- **Even:** z is within 2 standard errors of the baseline's, and 2.5 of its own above 1.
- **Literal verdict:** fail, because z RMS 1.22 and 1.30 exceed 1.2 (section 4).
- **One correction round was used.** The predicted errors change very little with the parameters, whose errors are
  under 0.1 %.

## 4. Criteria, and where they were restated (honest accounting)

- **T13c and T13d: the z limit (RMS ≤ 1.2) was too tight for 8 data sets.**
  - A calibration's parameters move together, so z's RMS over 8 data sets has a bootstrap standard error of about
    0.11. Even the baseline from exact centres reads 1.11 ± 0.11.
  - T12's lesson already said the windows should allow 3 standard errors. The plan's fixed 1.2 did not.
  - **Restated:** z RMS within 3 of its own standard errors of 1 (honest error bars).
    - Baseline: 1.0 SE above.
    - T13d's corrected fits pass: 2.0 and 2.5 SE above.
    - T13c's uncorrected fits still fail: 4.6 and 4.4 SE above.
  - The f and length criteria were met literally after the correction.
- **T13b's literal verdict is "fail" for every detector.** The masked ellipse estimator met the bias, RMS and
  labelling criteria in both required lightings. It missed one: "no wrong centre for overlapping balls", with 3 of
  17 in front light. The guard of T13b2, added after T13b, fixes that (pass).
- **Changes from the plan:**
  - **The masked ellipse estimator was added after T13b's smoke test,** which showed the rod pulling the fit by
    0.1–0.2 px. It is the plate case's estimator with the edge points in the rod's strip left out. The unmasked
    estimator was kept, so the effect of the mask is measured.
  - **pyOpenLPT's "reliable" detector** takes about 6 s per full frame. In T13b it measured every fourth frame, at
    best focus, on a crop around the wand. It is left out of T13c: it measured nothing in T13b, and its failure is
    structural (T13a, 2b).
  - **T13b2 (the guard) is new.**
  - **T13b3 (the symmetric mask) is new.** It was a cheaper alternative to T13d, tried after T13c pointed at the
    one-sided mask.
  - **T13d ran as the plan provides,** because T13c's best detector missed its criteria. Only backlit and even light
    were corrected: a correction needs a model of the lighting, and one-sided light or a highlight would have to be
    measured first.
  - **The radius range** given to pyOpenLPT's detector is 10–60 px. The GUI's default lower limit, 20 px, would
    miss the 1 mm ball, whose inscribed radius is 19.4 px. A user would set the range for the balls.
- **What the renderer assumes:**
  - **Blur:** a Gaussian per ball, fitted to three measured states, with the volume's centre at 18 mm of the plate
    case's traverse. Real defocus blur has the aperture's shape.
  - **Balls:** perfectly round, uniform and matte, apart from the highlight variant.
  - **Rod:** as dark as the background in front light.

## 5. What to do

- **Do not use pyOpenLPT's current wand detector for balls seen through the tube.** Its circle fit cannot follow
  the 1.3 : 1 images. Its wrong centres come with normal-looking radii: at 1.3 : 1 it put ball B 8.3 px off with
  radius 29 px, for a 39 × 30 px image (T13a2). The GUI's later filter, which sorts detections by radius, has
  nothing to catch them by.
- **Use an ellipse fit,** as tested here:
  1. edge points at half the local edge height along rays from the blob's centre;
  2. a robust fit;
  3. the edge points in the rod's strip left out;
  4. the camera's pair dropped when the two ellipses come within 3 px.
  - Tell the balls apart by the ellipse's area.
  - The code is `t13_ellipse.detect_balls(img, polarity, open_px, rod_mask_px=...)` with the guard of
    `t13b2_overlap_guard.py`. Putting it into pyOpenLPT is a separate step for your approval, for example a module
    used by a future wand CSV path or the GUI.
- **Light the balls evenly:** backlight them, or use diffuse light on matte balls. One-sided light and highlights
  shift the centres by 0.1–0.3 px, which biases f by up to 0.1 % and 0.56 % even with the best detector (T13c).
- **For the last 0.03–0.07 % in f, and honest error bars, add the model-based correction** (T13d).
  - The model needs the blur, and it needs to know that the lighting is even or from behind.
  - The blur can be measured from the balls' own edge profiles, as the plate case did for its dots.
  - Without the correction, the error bars of a calibration from the best detector are about 1.5× too small.
- **Keep the measurement volume near best focus.** Defocus roughly doubles the error of the best estimator.
- **Choose the rod and the opening together.** The opening must be wider than the rod's image (here 25 px for a
  0.3 mm rod), and the mask strip about the rod's half-width plus twice its blur (here 18 px).

## 6. Not covered

- **Real images.** The blur, the lighting and the balls' surfaces are models. A real ball may also reflect the tube
  wall or the light source.
- **A moving wand's motion blur.**
- **Other ball and rod sizes.** The opening and the rod mask were sized for this wand.
- **Image-fixed errors of the tube wall,** a separate risk (T11e).
- **A correction with a blur model that is itself estimated, and so slightly wrong.** In T13d the blur model was
  the renderer's own.
