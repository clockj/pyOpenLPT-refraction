# Detection noise and the cylinder wand calibration: plan (T11)

Date: 2026-10-01. Status: **approved by the user on 2026-10-01 ("go ahead"), T11e included; implemented and run
2026-10-01/02.** Results, the one criterion that failed literally (and why), and the changes from this plan are in
`2026-10-01-noise-report.md`. The body below is the plan as approved. Asked: "can you also test how will noise
affect the calibration?"

Scope: as for the main campaign (`2026-10-01-plan.md`).
- Only pyOpenLPT is touched; nothing in `D:\Refraction\paper`.
- The calibration code is not changed during these tests. A defect they find is reported with a proposed fix, and
  fixed only after approval.
- Scripts, outputs and the report live in this folder (debug-workflow).

## 1. What is known, and what is not

Every test so far used one noise level: i.i.d. Gaussian, 0.1 px per coordinate.

**What theory already gives.** For small noise, every Cramér–Rao σ is exactly proportional to the noise σ, so T2's
table scales by σ / 0.1 px. For σ(f)/f:
- the reference rig (3 cameras, 45° apart, ±20 mm cloud, 500 frames) has 0.064 % at 0.1 px, so it crosses the plan's
  0.5 % target at about **0.78 px**;
- a ±8 mm cloud, which is what cam1's present sensor sees, has 0.22 % at 0.1 px, so it crosses at about **0.23 px**.

For scale: at cam1, 1 px is about 20 µm across the axis and 26 µm along it, in the water. The real plate case fitted
at 0.20 px RMS.

**What theory cannot tell, and these tests measure:**
1. **The fits at larger noise.** Do they stay at the bound, or do nonlinearity and bias appear?
2. **The start from scratch.** Does the bootstrap (affine factorisation, candidate search, robust stage 3) still find
   the global minimum at 0.5–5 px?
3. **The error bars.** Is the σ the fit reports still honest at every noise level?
4. **Noise that is not uniform i.i.d. Gaussian.** The calibration treats all detections alike: unweighted least
   squares, and one pooled robust σ both for rejecting outliers and for the covariance.
   - **Prediction:** a camera with 5× the others' noise loses about 40 % of its detections to the 5σ rejection.
     Its error bars also come out too small.
   - This matters for a real rig, where one camera may be out of focus or see smaller blobs.

## 2. Tests

All on synthetic data from the C++ PINCYL camera, in the case's tube, with a 10 mm wand. Unless stated: the reference
rig, 500 frames, ±20 mm cloud.

### T11a: noise sweep (`t11a_noise_sweep.py`)
- **Noise levels:** σ = 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2 px (Gaussian, per coordinate).
- **Runs per level:**
  - 20 data sets, each with a new wand path and new noise, fitted from a perturbed start as in T4 (2°, 20 mm, 5 % in f);
  - 5 of them also from scratch (T11b).
- **Measured per level:**
  - **Accuracy:** RMS error over the 20 fits of f (%), camera distance (mm), azimuth (°), height (mm) and rotation (°),
    against the Cramér–Rao σ.
  - **Honest error bars:** pooled z-scores; criterion mean within ±0.25, standard deviation 0.8–1.2.
  - **Bias:** each parameter's mean error over its standard error; flag |t| > 3.
  - **σ estimate:** the fit's σ (from its residuals) against the true noise; criterion within 5 %.
  - **False rejections:** observations dropped, against the χ² expectation (about 0 per 3000 at 5σ).
  - **3-D effect, unseen points:** 7 × 7 × 7 grid over ±20 mm, re-triangulated with the fitted cameras. Error in µm in
    the tube frame, after removing the unobservable spin about and slide along the axis. Also in px, as T2/T7.
  - **3-D effect, wand length:** 100 held-out noisy frames per data set, triangulated. RMS length error in µm, i.e. what
    you would see in practice. The share due to the calibration comes from noise-free copies of the same frames.
- **Criteria:** realised error / Cramér–Rao within 0.8–1.2 up to 1 px; beyond that, reported.

### T11b: start from scratch under noise (`t11b_noise_scratch.py`)
- **Reference rig:** the 5 from-scratch runs per level of T11a.
- **Random rigs:** the first 10 rigs of T5 (2–4 cameras), with each rig's own seed for the geometry, at σ = 0.5, 1,
  2 and 5 px.
- **Success:** the same answer as the fit started at the truth (as T5: RMS within 1e-6 px, every parameter within
  0.01σ).
- **Reported:** success rate and run time per level, and where the start breaks down. Any failure is diagnosed
  (which stage, which candidate) before anything else.

### T11c: unequal noise (`t11c_unequal_noise.py`)
- **c1:** one camera (cam2) at 0.5 px, the others at 0.1 px.
- **c2:** wand end A at 0.2 px, end B at 0.1 px (balls of different size).
- **Runs:** 20 data sets each from a perturbed start, plus 5 from scratch.
- **Measured:**
  - observations rejected, per camera and per end;
  - z-scores per camera (is the noisy camera's σ too small?);
  - accuracy against the Cramér–Rao bound of the correctly weighted problem (how much the equal weighting costs).
- **If the prediction holds:** I report it with a proposed fix: a robust σ per camera (and per end) for the
  rejection, and the matching weights in the adjustment and the covariance. It is implemented only after approval.

### T11d: heavy-tailed noise (`t11d_heavy_tails.py`)
- **d1:** contaminated Gaussian: 95 % at 0.1 px, 5 % at 0.5 px.
- **d2:** Student-t with 3 degrees of freedom, scale 0.1 px.
- **Runs:** 20 data sets each from a perturbed start, plus 5 from scratch.
- **Measured:**
  - rejections, and how many of the wide-tail detections they caught;
  - bias, and the σ estimate;
  - z-scores;
  - accuracy against Gaussian noise of the same RMS (T11a, interpolated).

### T11e (optional): systematic image errors (`t11e_systematic.py`)
Not noise, but the case that worried the real data: an image-fixed pattern that PINCYL cannot carry. The real case
fitted 0.1–0.2 px of it.
- **Pattern:** per camera, a smooth random field (low-order polynomials over the sensor) at 0.1 and 0.2 px RMS.
- **Runs:** 10 data sets each.
- **Measured:**
  - bias of f and of the camera positions, in σ units;
  - the rise in RMS;
  - whether the free-wand-length check (T9) flags it.

## 3. Deliverables
- `t11*_*.py`, with `*_out.txt` and `*.json` beside each.
- `2026-10-01-noise-report.md`. Its centre is a practical table: detection noise against σ(f)/f, camera position, 3-D
  point error and wand-length error. It covers the reference rig and, by scaling T2, the other designs, including
  the ±8 mm cloud of cam1's sensor.
- A "Noise" section added to the figures page, at the same link:
  - errors against noise, log-log, with the Cramér–Rao line and the 0.5 % target;
  - success from scratch against noise;
  - error-bar honesty against noise;
  - unequal and heavy-tailed noise.

## 4. Cost
About 75 min of computing if run one after another. The runs are independent, so I run them in parallel processes
(one BLAS thread each, 32 cores): about 15–20 min.

## 5. Not covered
- **Sphere-centre bias.** The centroid of a large wand ball imaged through the tube is not the image of its centre.
  This is a systematic error that depends on position and ball size; a separate test if wanted.
- **Real data.**
