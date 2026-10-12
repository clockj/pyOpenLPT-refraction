# Detection noise and the cylinder wand calibration: report (T11)

Date: 2026-10-01/02. Plan: `2026-10-01-noise-plan.md` (approved, "go ahead"; T11e included). All synthetic data:
- detections are the C++ PINCYL camera's exact images, in the case's tube;
- the wand is 10 mm long;
- unless stated: the reference rig (three cam1 replicas 45° apart, 1094 mm from the axis), 500 frames, ±20 mm cloud.

No calibration code was changed. Two weaknesses found are proposed below (section 6), not fixed. The figures are
section 07 of `figures/cylinder_wand_results.html`.

## 1. Result

**Random Gaussian noise does what theory says.**
- Every error grows in proportion to the noise and sits on the Cramér–Rao prediction, from 0.02 to 2 px.
- The fit's error bars are honest at every level:
  - z standard deviation 0.95–1.02;
  - the fit's σ within 0.3 % of the true noise.
- No detectable bias.
- 4 false rejections in 840,000 observations.

| noise σ (px) | f error, worst camera | camera distance | 3-D error relative to the tube | 3-D shape and scale | wand length as measured | of which the calibration |
|---|---|---|---|---|---|---|
| 0.02 | 0.014 % | 0.16 mm | 3.5 µm | 0.15 µm | 0.47 µm | 0.05 µm |
| 0.05 | 0.034 % | 0.36 mm | 8.7 µm | 0.38 µm | 1.2 µm | 0.11 µm |
| 0.1 | 0.069 % | 0.74 mm | 19 µm | 0.76 µm | 2.4 µm | 0.23 µm |
| 0.2 | 0.14 % | 1.5 mm | 39 µm | 1.5 µm | 4.6 µm | 0.47 µm |
| 0.5 | 0.33 % | 3.6 mm | 82 µm | 3.8 µm | 11 µm | 1.2 µm |
| 1 | 0.76 % | 8.2 mm | 156 µm | 7.0 µm | 24 µm | 2.2 µm |
| 2 | 1.3 % | 14 mm | 341 µm | 15 µm | 47 µm | 4.4 µm |

How to read the table:
- **Scale:** the errors are RMS over 40 data sets per level. Per pixel of noise: f about 0.7 %, 3-D shape 7.5 µm,
  wand length 24 µm.
- **3-D relative to the tube** is what absolute positions in the tube frame feel. It is dominated by where the tube
  sits relative to the rig, which the wand pins down only loosely.
- **3-D shape and scale** is the error left after the best rigid motion: what distances and velocities feel. It is
  about 25 times smaller.
- **Wand length as measured** is on 100 frames left out of the fit, so it includes those frames' own detection noise.
  The last column is the calibration's share alone.

**Noise thresholds.**
- **Reference rig:** f reaches the plan's 0.5 % at about 0.7–0.8 px of noise. The measured curve crosses at
  0.69 px; the Cramér–Rao bound (worst camera) at 0.78 px.
- **A ±8 mm cloud** (what cam1's present sensor sees) reaches 0.5 % already at 0.23 px.
- Because the scaling is linear, any T2 design's threshold is 0.05 px / (its σ(f)/f in % at 0.1 px). The figures
  page's T2 table now lists it.

**From scratch (no hints).**
- The reference rig succeeds 5/5 at every level from 0.02 to 5 px.
- T5's first ten random rigs (two with 2 cameras, five with 3, three with 4) succeed:
  - 10/10 at 0.5, 1 and 2 px;
  - 9/10 at 5 px.
- **The miss:** rig 7 at 5 px ended in a near-telecentric wrong minimum.
  - cam3's f came out 37 % too long and the camera 360 mm too far away.
  - The RMS was only 0.03 px higher than at the global minimum.
  - The right start was on the bootstrap's short list, but fourth.
  - `calibrate(..., n_final=6)` finds it, an existing argument (80 s instead of 34 s).

**Noise that is not uniform Gaussian.**
- **One camera at 5× the others' noise (cam2 at 0.5 px, others at 0.1 px).**
  - My prediction was wrong: almost nothing is rejected (0.2 % of cam2's detections, not 40 %). The pooled robust σ
    rises with the noisy camera (0.29 px), so the 5σ limit does too.
  - The error bars stay roughly honest: z RMS per camera 0.87 / 1.09 / 0.95.
  - **The real cost is precision.** Weighting every detection alike makes every camera's σ 1.7–2.1× worse than a
    weighted fit would. The noisy camera degrades the quiet ones through the shared wand positions: f errors
    0.16–0.25 % where 0.09–0.17 % is possible.
- **Wand end A at 0.2 px, end B at 0.1 px:** mild. Equal weights cost ×1.1 and z RMS is 1.08.
- **Occasional large errors** (5 % of detections at 0.5 px, or Student-t with ν = 3):
  - the error bars stay honest (z std 1.03 and 0.95);
  - the 5σ rejection removes 38 % of the wide detections;
  - precision lands between least squares keeping everything (×0.87 / 0.82 of it) and the ideal
    (×1.23 / 1.21 of it).
- **A fixed pattern on each image, which PINCYL cannot carry, is the dangerous case.**
  - A smooth pattern of 0.07 px or 0.15 px RMS at the detections (plus 0.1 px noise):
    - about 60 % of it is absorbed into the parameters;
    - f is biased by 0.22 % / 0.47 % RMS, 3–7× its noise-only σ;
    - the fit's own error bars come out 3–4× too small.
  - Only 40 % of the pattern is left in the residuals: the RMS rises from 0.107 to 0.127 / 0.165 px, which looks like
    ordinary noise.
  - The free-wand-length check (T9) flags it in only 7 / 12 of 20 data sets.
  - Your real cam1 case showed a 0.1–0.2 px image-fixed pattern. The same in every camera of a wand rig would bias f
    by roughly 0.2–0.5 % behind error bars that look fine.

## 2. What was run

| script | what | outputs |
|---|---|---|
| `t11_common.py` | shared code: noise models, data sets, fits, Cramér–Rao / sandwich bounds, measurements, parallel runner (16 processes, 1 BLAS thread each) | — |
| `t11a_noise_sweep.py` | 7 levels × 40 data sets, fitted from a start near the truth | `t11a_noise_sweep_out.txt`, `.json` |
| `t11a_recheck.py` | 0.2 px again, 80 new data sets (section 4) | `t11a_recheck_out.txt`, `.json` |
| `t11a_recheck_levels.py` | 0.1, 0.15, 0.2, 0.3 px, 80 new data sets each | `t11a_recheck_levels_out.txt`, `.json` |
| `t11b_noise_scratch.py` | from scratch: reference rig 8 levels × 5, T5 rigs 0–9 × 4 levels | `t11b_noise_scratch_out.txt`, `.json` |
| `t11b_diag_rig7.py` | the 5 px miss: the six short-listed starts carried through | `t11b_diag_rig7_out.txt`, `.json` |
| `t11b_diag_rig7_nfinal.py` | the same data with `n_final=6` | `t11b_diag_rig7_nfinal_out.txt`, `.json` |
| `t11c_unequal_noise.py` | one camera / one end noisier; 40 + 5 per case | `t11c_unequal_noise_out.txt`, `.json` |
| `t11d_heavy_tails.py` | contaminated Gaussian, Student-t; 40 + 5 per case | `t11d_heavy_tails_out.txt`, `.json` |
| `t11d_rejection_split.py` | why good detections are rejected under contamination | `t11d_rejection_split_out.txt`, `.json` |
| `t11e_systematic.py` | image-fixed pattern, 2 amplitudes × 20 data sets, L fixed and free | `t11e_systematic_out.txt`, `.json` (first pass: `t11e_systematic_first_pass_*`) |

Each run's console output is in `*_console.txt`. Total computing time: about 15 minutes in parallel.

Shared measurements (`t11_common.py`):
- **Parameter errors** in the fit's local coordinates (as T4), as z-scores against three covariances:
  - the correctly weighted Cramér–Rao bound;
  - the fit's own covariance (one pooled σ);
  - the unweighted estimator's sandwich covariance S⁻¹MS⁻¹ under the true per-detection σ.
- **Per camera:** f, distance from the axis, azimuth, height.
- **3-D error of unseen points:** a 7 × 7 × 7 grid over ±20 mm, re-triangulated by maximum likelihood from exact
  images, in three forms:
  - raw;
  - after the unobservable spin about and slide along the axis (relative to the tube);
  - after the best rigid motion (shape and scale).
- **Wand length** on 100 held-out frames, from their noisy and their exact detections.

## 3. Results by test

### T11a: noise sweep
| σ (px) | f / CRB | distance / CRB | azimuth / CRB | height / CRB | z (fit) mean / std | fit σ / σ | rejected (of 120 000) | max \|t\| | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 1.06 | 1.07 | 0.83 | 1.00 | −0.084 / 1.005 | 1.0001 | 1 | 1.9 | pass |
| 0.05 | 1.02 | 1.02 | 0.97 | 1.04 | −0.025 / 0.972 | 1.0008 | 0 | 1.1 | pass |
| 0.1 | 1.00 | 1.00 | 1.01 | 1.02 | −0.000 / 1.021 | 0.9984 | 0 | 1.4 | pass |
| 0.2 | 1.03 | 1.03 | 1.07 | **0.71** | −0.041 / 0.951 | 0.9972 | 0 | 2.3 | **fail** (section 4) |
| 0.5 | 1.00 | 1.00 | 1.02 | 1.02 | −0.073 / 0.974 | 0.9999 | 2 | 1.9 | pass |
| 1 | 1.06 | 1.06 | 0.83 | 0.90 | −0.046 / 0.966 | 1.0009 | 0 | 1.7 | pass |
| 2 | 0.90 | 0.91 | 0.89 | 1.11 | +0.018 / 0.969 | 1.0001 | 1 | 2.1 | reported |

"/ CRB" is the realised RMS error over the Cramér–Rao σ, pooled over cameras; 1 is ideal. No parameter's mean error
reaches |t| = 3 at any level, so there is no bias even at 2 px. Each fit takes about 7 s.

### T11b: from scratch
| | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 | 1 | 2 | 5 px |
|---|---|---|---|---|---|---|---|---|
| reference rig (5 data sets per level) | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| T5 rigs 0–9 | · | · | 10/10 (T5) | · | 10/10 | 10/10 | 10/10 | 9/10 |

- Every success agrees with the fit started at the truth to 1.6e-5 σ.
- Median times: 33–36 s on the reference rig, 39–47 s on the random rigs.
- **Rig 7 at 5 px** (3 cameras at 0°, 82°, 200°; 428, 622, 962 mm; f 21k, 23k, 49k px):
  - the bootstrap short-listed six starts, ranked by a short robust fit;
  - the top two, which `calibrate` keeps (n_final = 2), both end in the wrong minimum above;
  - ranks 3 and 4 (counting from 0) reach the global minimum;
  - rank 5 ends far off, at 10.3 px.
  - Calibrate's own rule (most observations, then lowest RMS) picks rank 3 from all six, and `n_final=6` confirms it.
  - At 5 px the short-fit ranking is no longer reliable, but the full stage 3 is: the right starts end at 4.86 px
    against 5.04 px for the others.

### T11c: unequal noise
| | rejected per camera | z RMS per camera: fit's σ / sandwich / weighted bound | equal-weights cost (σ ratio) | f error, realised / weighted bound | from scratch |
|---|---|---|---|---|---|
| c1: cam2 0.5 px, others 0.1 px | 0.0 / 0.2 / 0.0 % | 0.87 / 1.09 / 0.95; 0.97 / 1.02 / 1.05; 1.99 / 1.76 / 2.24 | 2.03 / 1.71 / 2.14 | 0.16–0.25 % / 0.09–0.17 % | 5/5 |
| c2: end A 0.2 px, B 0.1 px | 0.1 % each | 1.09 / 1.06 / 1.10; 1.08 / 1.05 / 1.09; 1.22 / 1.20 / 1.19 | 1.11 / 1.13 / 1.09 | 0.11–0.13 % / 0.08–0.09 % | 5/5 |

- **The fit behaves exactly as least-squares theory says** (sandwich z ≈ 1).
- **The pooled σ makes the reported bars roughly right** for this split, but not per camera:
  - they are 15 % too wide for cam1;
  - they are 9 % too narrow for the noisy cam2.
- In c2 the sandwich z reads slightly high (pooled std 1.07). That is probably chance, but noted.

### T11d: occasional large errors
- **d1, 95 % of detections at 0.1 px and 5 % at 0.5 px** (RMS 0.149 px):
  - **Rejections:** 3.5 % of all detections. That is 38 % of the wide ones and 1.7 % of the others, and 97 % of the
    latter went with whole frames (`t11d_rejection_split`).
  - **Why whole frames:** once a wide detection goes, an end may be left in one camera, so the frame goes too.
  - **σ and error bars:** the fit's σ is 0.113 px against 0.116 px for the noise it kept; z std 1.03.
  - **Possible small bias in f:** the mean z of log f is −0.41 / −0.10 / −0.48 (t = −2.5 / −0.5 / −3.3), i.e. f about
    0.03 % low. It is at the edge of significance: two correlated parameters of cam3 (its f and its x position) reach
    |t| = 3.3, out of 38 parameters in T11d. It is small against the 0.08 % precision.
- **d2, Student-t, ν = 3, scale 0.1 px** (RMS 0.173 px):
  - 1.9 % rejected;
  - the fit's σ is 0.144 px against 0.147 px kept;
  - z std 0.95.
- **Precision against least squares keeping everything / the best possible:** ×0.87 / ×1.23 (d1), ×0.82 / ×1.21 (d2).
  The best possible is 0.70 / 0.68 of least squares.

### T11e: an image-fixed pattern
| | pattern at the detections | absorbed into the parameters | left in residuals | f bias from the pattern alone | realised f error (× noise-only σ) | fit's z RMS | fit RMS | free L flags it | 3-D shape | wand length (calibration share) |
|---|---|---|---|---|---|---|---|---|---|---|
| e1 | 0.073 px | 58 % (0.056 px) | 0.047 px | 0.22 % RMS | 0.23–0.25 % (×3.8) | 2.9 | 0.127 px | 7/20 | 2.0 µm | 2.7 µm (0.58) |
| e2 | 0.145 px | 60 % (0.113 px) | 0.091 px | 0.47 % RMS | 0.43–0.50 % (×7.4) | 4.2 | 0.165 px | 12/20 | 3.9 µm | 3.6 µm (1.20) |

For comparison, 0.1 px of noise alone gives: f error 0.069 %, fit RMS 0.107 px, 3-D shape 0.76 µm, wand length
2.4 µm (calibration share 0.23 µm).

- **The pattern.**
  - Each camera gets a smooth random field over the box the wand covers: Legendre products of degree 2–3. A new
    pattern is drawn for each data set.
  - It was scaled to 0.1 / 0.2 px RMS over the box. At the detections, which cluster mid-box, it measures 0.07 /
    0.15 px.
- **How "absorbed" is measured.**
  - It is exact and noise-free: one Gauss–Newton step at the truth with the pattern as the only residual.
  - The first pass measured it from the fitted residuals instead. That mixed in the noise the wand positions absorb
    and read a meaningless −21 %. It is kept as `t11e_systematic_first_pass_*`; its other numbers are unchanged.
- **The cost to a tracking run is moderate:** 3-D shape errors are 2.6× / 5× the noise-only ones, and wand lengths
  barely change.
- **The cost to the calibration report is large:** the parameters are off by 3–4 of the fit's own quoted σ on
  average. For f, the error is 4–7 of its noise-only σ.

## 4. Criteria, and where they were restated (honest accounting)

**T11a, 0.2 px, camera heights: literal FAIL** (0.71 against the 0.8 limit; the rolls w3 read 0.72–0.77 in the same
run).
- **The diagnosis:**
  - An unbiased fit cannot beat the bound on average, so the result was rerun twice.
  - The reruns at 0.2 px gave 0.80 (80 new data sets) and 0.99 (80 more).
  - The control and neighbouring levels gave 0.97 (0.1 px), 0.95 (0.15 px) and 0.89 (0.3 px).
  - Pooled over T11a's 280 data sets, the heights sit at 0.98, with no heavy tails (kurtosis +0.1).
  - The group that reads low moves between runs: in the second rerun at 0.2 px it was a rotation, w1, at 0.80.
- **Why it happens:** cam2's and cam3's height errors correlate at 0.86–0.93, so each level is about 40 draws of a
  single mode. Their RMS has a standard error of about 11 %, and 12 level-runs × 7 parameter groups give about 84
  such ratios. A −2.6 SE low among them is expected.
- **Conclusion:** chance, not an effect. The criterion's window (0.8–1.2 with 40 data sets) was too tight for a
  quantity driven by one mode; ±3 SE (0.67–1.33) would have been the right gate. All other criteria pass at every
  level up to 1 px.

Changes from the plan:
- **More data sets:** 40 per level in T11a and per case in T11c/T11d instead of 20, for tighter ratios; 20 per case
  in T11e instead of 10.
- **A new 3-D measure:** the shape-and-scale error, after a rigid motion. The tube-frame error turned out to be
  dominated by the tube's position, which tells little about distances.
- **T11e's absorption measure was replaced,** as described above.
- **The T11c prediction (about 40 % rejected) was wrong.** The weakness is in precision, not rejection.

## 5. For real data

1. **Budget the noise.** With the reference-like rig, about 0.3 px of detection noise keeps f within 0.2 %.
   With cam1's present ±8 mm field, f reaches the 0.5 % target already at 0.23 px, so aim for 0.1 px or a deeper
   cloud.
   - Precision ∝ σ / √frames, so 4× the frames buys back 2× the noise.
   - A deeper wand cloud helps f most (T2).
2. **Read the residual maps, not just the RMS.** A fixed pattern hides in the RMS: 0.127 px looks like 0.107 px of
   noise. Per-camera residual maps binned over the image show it as structure.
3. **Run the free-wand-length check** (`fit_L=True`, T9). It catches the larger patterns and wrong tube constants.
   Treat an offset beyond 3σ as a warning that the error bars are too small.
4. **Watch for a blurred or distant camera.** If one camera's residuals are clearly larger, the calibration still
   works, but every camera loses precision (section 6.1).
5. **For very noisy data (σ above ~2 px), use `calibrate(..., n_final=6)`.** It costs about 45 s more.

## 6. Proposed changes (not made; each needs your approval)

**Update 2026-10-02:** items 1 and 2 were approved and implemented as T12 (`2026-10-02-weights-plan.md`,
`2026-10-02-weights-report.md`). Per-camera weights are now the default, and they recover T11c's factor of
1.7–2.1. The extra starts at high noise give 10/10 at 5 px. Item 3 is open. The T11 results above are for the code
before T12; the T11 scripts keep that behaviour via `t11_common.T11_REFINE` / `T11_CALIBRATE`.

1. **Per-camera noise weights.**
   - **What:** estimate a robust σ per camera (and optionally per end) in `_update_active`. Then weight the
     adjustment and the covariance by 1/σ², and reject against each camera's own σ.
   - **Gain:** it recovers the factor 1.7–2.1 of T11c and makes each camera's error bars honest.
   - **Test:** T11c with the weights against the weighted bound.
2. **More starts when the noise is high.**
   - **What:** carry all six short-listed starts through stages 3–4 when the stage-3 RMS exceeds about 1 px, or raise
     the default `n_final`.
   - **Cost:** about +45 s per calibration.
   - **Test:** T11b's 5 px column.
3. **Image-fixed patterns (larger).** Fit and check a model that can carry them:
   - the distortion terms beyond k1;
   - or a low-order per-camera correction field, with a guard against it absorbing the geometry.
   - T11e is the test bench.

## 7. Not covered

- **Sphere-centre bias.** The blob centroid of a large wand ball seen through the tube is not the image of its centre.
- **Real data.**
