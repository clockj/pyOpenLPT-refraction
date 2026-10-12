# T1's tracer images: survey, preprocessing and a line detector (P0b, P0c, L1, L2) — report

7 October 2026. Plan: `2026-10-07-plan.md` (Phase 0 steps P0b and P0c; Phase 4 steps L1 and L2; approved with all
phases). Every number below is read from an output file in `tests/`, written by the script named beside it.

**In short.**
- **T1's images:** about 1,500 (cam1) and 2,100 (cam2) objects per frame stand above 4 sigma: horizontal lines,
  vertical lines, ellipses and round spots. The background is stable over the 1,001 frames. Of hp's noise
  (1.35 / 1.48 counts) only 1.05 counts is temporal; the rest is a texture of faint, blurred particles.
- **Frame step 4:** particles move about 0.5 px (cam1) and 0.3 px (cam2) per frame, so 4 frames give a median
  2.1 / 1.2 px.
- **Preprocessing (L1):** removes the dark clouds (tile scatter 0.55 → 0.20 counts on T1) and centres the
  background on 0. Gains of 6 (cam1) and 4 (cam2) put the noise at 8.1 and 5.9 counts in the 8-bit image, with no
  saturation.
- **The detector (L2), isolated objects:** it meets the plan's targets at SNR ≥ 5 (recall 99.9 %, no duplicates,
  lines 0.28–0.30 px along and 0.08–0.09 px across). Today's C++ finder gives 1.3–2.6 extra detections per object
  and errors of 0.8–0.9 px along.
- **The detector at T1's density:** in a synthetic population calibrated to T1, recall at SNR ≥ 5 falls to 80 %
  (cam1) and 51 % (cam2), and errors grow to about 1 px along. Almost every object there overlaps another (98–99 %).
  It still makes no duplicates and finds about as many objects as on T1 itself: on T1 it finds 1,001 (cam1) and
  862 (cam2) objects per frame, where today's finder gives 10,000–12,000 local maxima at 4 sigma.
- **For L3:**
  - The objects do not lie on a single astigmatic locus per camera.
  - The fitted flux grows 6× (cam1) and 9× (cam2) from the smallest images to the largest, so the amplitude
    model must be empirical.

## 1. What was run

All scripts run from the case folder with the `refract` interpreter. They use the local copy of T1 and a frozen copy
of the module (`data/pyd_lines/`, from `build-py/Release` of 7 Oct); nothing outside the case was changed.

| step | script | outputs (beside it) |
|---|---|---|
| local copy of T1 (2 × 1,001 frames, 2.05 GB) | `tests/p0b_copy_t1.py` (uses `t1.py`) | `p0b_copy_t1_out.txt`, `data/T1raw/manifest.json` (3 frames per camera checked by sha256 against the share: all match) |
| P0b, the images | `tests/p0b_t1_survey.py` | `p0b_t1_survey_out.txt`, `.json`, `p0b_shapes.png`, `p0b_cam1_hp.png`, `p0b_cam2_hp.png`; backgrounds in `data/T1bg/` |
| P0c, the motion | `tests/p0c_t1_motion.py` | `p0c_t1_motion_out.txt`, `.json`, `p0c_k8.png` |
| L1, preprocessing checks | `tests/l1_preprocess.py` | `l1_preprocess_out.txt`, `.json`, `l1_synthetic.png` |
| L1, preprocessed frames | `tests/l1_write_pre.py` | `l1_write_pre_out.txt`; all 1,001 frames of both cameras in `data/T1pre/` with `manifest.json` (gain, the noise of every frame) |
| L2, the mathematics | `tests/l2_selftest.py` | `l2_selftest_out.txt`, `.json`: 11 of 11 checks pass |
| L2, synthetic | `tests/l2_synthetic.py` | `l2_synthetic_out.txt`, `.json`, `l2_synthetic.png` |
| L2, on T1 | `tests/l2_t1.py` (every 100th frame, 11 per camera) | `l2_t1_out.txt`, `.json`, `l2_t1_shapes.png`, `l2_t1_cam1_overlay.png`, `l2_t1_cam2_overlay.png`; detections in `data/T1det/` |
| L2, the design choices | `tests/l2_variants.py` (k_detect on pure noise, the templates' noise on T1, the bounds, the suppression, the clipped model) | `l2_variants_out.txt`, `.json` |

The code is in `lines.py`, the module meant to be ported (preprocessing, rendering, detector, scoring), and `t1.py`
(T1's files).

## 2. P0b: the images

`tests/p0b_t1_survey_out.txt`. hp = (background − frame) − median₃₁, recentred (L1).

| | cam1 | cam2 |
|---|---|---|
| grey level of the frames (median) | 105, the same at every 100th frame | 146, the same |
| background (per-pixel 90th percentile of 1,001 frames) | median 107 | median 149 |
| every 10th frame against all: \|diff\| 99th percentile | 1.0 count | 1.0 count |
| first half against second half: \|diff\| 99th percentile (max) | 2.0 (4) counts | 2.0 (8) counts |
| hp noise (negative tail) | 1.35 counts (MAD 1.54) | 1.48 counts (MAD 1.98) |
| temporal noise (consecutive frames' difference / √2, MAD) | 1.05 counts | 1.05 counts |
| neighbour correlation of hp's noise (horizontal / vertical) | 0.41 / 0.39 | 0.51 / 0.47 |
| background level of hp in 64 px tiles: mean, scatter (before the high-pass) | −0.03, 0.20 (0.55) counts | −0.02, 0.24 (0.65) counts |
| objects (components of ≥ 6 px) per frame, 4 σ / 6 σ | 1,491 / 606 | 2,088 / 993 |
| classes at 4 σ: H / V / ellipse / round / cross | 27 / 26 / 21 / 19 / 8 % | 27 / 29 / 19 / 17 / 8 % |
| H core length above 4 σ, 10/50/90/99 % | 5/10/18/29 px | 6/10/20/33 px |
| V core length above 4 σ, 10/50/90/99 % | 6/9/13/20 px | 6/9/13/21 px |
| thickness, 10/50/90 % | 2/3/6 px | 2/3/6 px |
| orientation of long lines (moments): H, V from vertical | +0.30°, +0.22° | −0.04°, +0.02° |
| peak hp of the objects: median, 99 %, 99.9 % | 9.5, 25.0, 36.7 counts | 11.1, 32.6, 46.7 counts |

**What this says:**
- **1,001 frames are more than enough for the background:** every 10th frame gives the same to 1 count. The halves
  differ by at most 2 counts at the 99th percentile. Particles never stay long on a pixel, since they move.
- **The noise is not white.** The pixel-to-pixel noise is 1.05 counts, but hp reads 1.35–1.48 counts, correlated
  between neighbours (0.4–0.5). The rest is a texture of faint, blurred particles that moves with the flow; it
  cancels between consecutive frames. It sets the detection limit (section 5.1): long templates collect it, and
  their noise is 3.5–8.6× the white-noise value (`l2_variants_out.txt`).
- **The lines are horizontal and vertical to within a fraction of a degree:** the cameras are not rolled against
  the tube's axis, so an axis-aligned model suffices.
- **Contrast against length:** the summed signal of a component grows with its length, 155 → 1,590 counts from
  4–7 px to 30–60 px (cam1, H). The fits on T1 confirm it (section 6.3).

## 3. P0c: the motion and the frame step

`tests/p0c_t1_motion_out.txt`. 64 px windows, normalised cross-correlation; the search for each larger gap is
guided by the window's displacement at half the gap. An unguided search at 8 and 16 frames locks onto other lines.

| gap k (frames) | cam1: median du, dv; \|d\| 50/95 % | correlation | cam2: median du, dv; \|d\| 50/95 % | correlation |
|---|---|---|---|---|
| 1 | −0.33, −0.02; 0.34/0.66 px | 0.79 | −0.19, +0.07; 0.22/0.43 px | 0.87 |
| 2 | −0.77, −0.05; 0.79/1.59 px | 0.74 | −0.41, +0.14; 0.46/0.97 px | 0.83 |
| **4** | **−2.04, −0.11; 2.10/3.36 px** | 0.63 | **−0.96, +0.32; 1.17/2.31 px** | 0.71 |
| 8 | −3.39, −0.26; 4.14/7.40 px | 0.44 | −1.10, +0.76; 3.00/5.40 px | 0.52 |
| 16 | unreliable (112 % off the k = 4 trend) | 0.39 | unreliable (244 % off) | 0.41 |

- **The flow is steady over the recording:** the medians at start frames 0, 200, 400, 600 and 800 agree within
  about 15 %.
- **The speed comes from k = 4:** about 0.51 px per frame horizontally in cam1, and 0.24 / 0.08 px per frame in
  cam2.
- **k = 1 reads 35 % (cam1) and 19 % (cam2) low against it.** The 3-point peak fit locks sub-pixel shifts towards
  whole pixels (a known effect; not verified separately here).
- **Beyond 8 frames the windows decorrelate (≤ 0.52):** the images change shape as the particles move in depth.

**Recommended frame step: 4.** That gives a median 2.1 px (cam1) and 1.2 px (cam2) per step, 95 % within 3.4 and
2.3 px, and 251 processed frames of the 1,001 (1,000 steps/s). Step 2 halves the shape change between steps if the
tracker has trouble with it.

## 4. L1: preprocessing

`lines.preprocess`, steps in order:
1. **signal** = background − frame, with the background the per-pixel 90th percentile of the recording;
2. **hp** = signal − median₃₁(signal). The median is computed by OpenCV on 4 × signal + 64 (0.25-count resolution),
   then smoothed by a 15 px box;
3. **recentre:** hp −= the noise's level read from its negative tail, 2 Q(15.87 %) − Q(2.275 %). The particles
   pull the local median up, and they never reach that tail;
4. **8-bit output** = clip(round(gain × hp), 0, 255).

The noise is read from the same tail: σ = Q(15.87 %) − Q(2.275 %) on values jittered by ±½. The jitter is there
because whole-count data make quantiles snap, which halved the estimate on a flat synthetic background. The
jitter's 1/12 is removed from the variance.

**Synthetic check** (`l1_preprocess_out.txt`): 201 frames of 640 × 512 px at T1's density, with drifting clouds up
to 7.4 counts deep and noise 1.59 counts.
- **Clouds:** the tile scatter of the background drops from 0.61 counts before the high-pass to 0.074 after it,
  centred at −0.07 counts.
- **Noise read:** 1.64 counts against the true 1.59.
- **Flux kept (bright particles, peak > 6 σ):** median 0.88 (semi-axis 0–4 px) and 0.91 (4–8 px) of the true
  flux. The 10–90 % spread is mostly noise. About 10 % of the flux is lost; the cause was not isolated (the
  median's pull, the recentring or the background percentile). The centres are not affected (section 5).

**On T1:**
- The background after the high-pass is flat to 0.20–0.24 counts between 64 px tiles, at −0.02/−0.03 counts.
- **The gain** is the largest integer that keeps the 99.9th percentile of the objects' peaks at or below
  230 counts: **6 for cam1, 4 for cam2.** The noise in the 8-bit image is then 8.1 (7.8–8.3) and 5.9 (5.8–6.1)
  counts, and the saturated pixels are at most 0.0004 % of the frame.
- A single gain of 5 for both would also stay below saturation (99.9 % peaks at 184 and 234 counts).
- 46.7 % (cam1) and 46.2 % (cam2) of a preprocessed frame's pixels are 0, the clipped negative half of the noise
  (`l2_variants_out.txt`, frame 500).

## 5. L2: the detector

### 5.1 The algorithm (lines.py, `detect`)

**The model of one object** (Lx, Ly ≥ 0):

    m(x, y) = A · P(x − u; Lx, sx) · P(y − v; Ly, sy) + c

- It is a box of lengths (Lx, Ly) blurred by a Gaussian (sx, sy) and integrated over each pixel.
- P is closed form: P(t; L, s) = (s²√2π / L)[ψ(z₁) − ψ(z₂) − ψ(z₃) + ψ(z₄)], with ψ(z) = zΦ(z) + φ(z) and
  zᵢ = (t ± ½ ± L/2)/s. Its derivatives in t, L and s are analytic (section 5.2).
- A horizontal line is Ly ≈ 0, a vertical line Lx ≈ 0, an ellipse or round spot both small.
- The flux 2πA·sx·sy does not depend on the lengths.
- The object's RMS widths wx = √(Lx²/12 + sx²) and wy are well determined, while L and s alone are not, for short
  objects.

**Steps:**
1. **Matched filters.** One template per pair (Lx, Ly) of `lengths_x` × `lengths_y` (8 × 7), blurred by `sigma0`,
   each response minus the template's sum times the mean of a box around it. Each response is divided by its own
   noise level (MAD over every 4th row and column). Normalised that way, the templates are 1.6–8.6× noisier on T1
   than white noise predicts, the long ones (24 px or more) 3.5–8.6×, because of the texture of faint particles
   (`l2_variants_out.txt`).
2. **Candidates:** 3 × 3 local maxima of the best normalised response, at least `k_detect`.
3. **Suppression,** by decreasing score. A candidate is dropped if its centre lies inside an accepted object's box
   (|du| ≤ Lx/2 + `core_px`, the same in v), or if an accepted object's centre lies inside its own box. Without the
   second rule every bright compact object had off-centre large-template companions.
4. **Fit.** Levenberg-Marquardt of the 8 parameters on each object's window (L/2 + 3s + 2 px, capped at 28), all
   objects at once in batches, for 2 Jacobi passes with the other objects' models held fixed in each window.
   - With `clipped`, the model is the expectation of the clipped image, m Φ(m/σ) + σ φ(m/σ).
   - The centre stays within ±(0.25 L + 2.5 px) of the candidate, and |c| ≤ 1.5 σ.
   - Unbounded, c ran to −20 σ with the clipped model, trading against A and the blur.
5. **Second pass** on the residual image: new candidates outside the accepted boxes (grown by 2 px) are fitted
   together with their neighbours. It adds 21–23 objects per frame on T1.
6. **Validation.** An object stays only if its own matched-filter score, on the image minus every other object's
   model, still reaches `k_detect`. This drops pieces of long lines and copies.
7. **A final fit** of every object with all the others held fixed.

**Outputs per object:** u, v, Lx, Ly, sx, sy, wx, wy, A, c, flux, peak, score, χ², pixels, class (H, V, ellipse,
round) and the pass that found it.

**Parameters** (`lines.DET`, the future keys of `tracer.detector: "lines"`):

| key | value | key | value |
|---|---|---|---|
| lengths_x | 0, 3, 5, 8, 12, 17, 24, 34 px | lengths_y | 0, 3, 5, 8, 12, 17, 24 px |
| sigma0 | 1.0 px | bg_guard | 4 px |
| k_detect | 6.0 (pure clipped noise: 69 candidates per frame at 5, 20 at 5.5, 6.3 at 6, 1.7 at 6.5) | core_px | 2.0 px |
| nms_symmetric | true | fit_margin_sigma, fit_max_half | 3, 28 px |
| fit_iters | 40 | deblend_passes | 2 |
| second_pass, validate | true, true | clipped | true |
| centre_box_frac | 0.25 | dynamic_bounds | true (lengths up to L_max) |
| L_min, L_max | 0.05, 60 px | sigma_min, sigma_max | 0.5, 4 px |
| c_max_sigma | 1.5 | noise | from L1 (gain × the noise of hp), per camera |

**The design choices** (`l2_variants_out.txt`):
- **dynamic_bounds** replaced a bound of the lengths by the template's length (1.5 L_t + 8 px).
  - On T1 (cam1 frame 500) that bound pinned 55 % of the objects at exactly 8 px: the texture makes the long
    templates noisy, so objects are found with short ones. Freed, the H lines' median length is 12.4 px instead of
    8.0, and 1,016 objects are found instead of 1,139.
  - On a dense synthetic frame the bound gave a higher recall at SNR ≥ 5 (97.1 % against 95.9 %), but a larger
    error along the H lines (0.95 against 0.82 px).
- **The symmetric suppression** finds more objects on T1 than the asymmetric one (1,016 against 874); the
  asymmetric one lets large-template companions through, which then split the signal.
- **The clipped model** reads flux and RMS widths without bias on isolated objects: flux 1.036 of the truth,
  widths 1.010–1.018. The plain model reads them low: flux 0.933, widths 0.966–0.971. Centres and recall are the
  same, and the plain model is about 30 % faster.

### 5.2 Its mathematics (`tests/l2_selftest.py`: 11 of 11 pass)

Checked numerically:
- **P against quadrature:** to 3.3e-8.
- **Its three derivatives against differences:** to 9e-9. The L → 0 branch agrees to 2.4e-11, and P is continuous
  there (1.5e-10).
- **The clipped expectation:** against Monte Carlo to 0.0018, and its derivative to 9e-11.
- **The renderer of the synthetic tests:**
  - flux to 5e-7 and centroid to 2e-6 px;
  - a line's profile against the semicircle ⊗ Gaussian to 1.6e-4 of the peak;
  - a disk against 20× supersampling to 4.2e-4.
- **The noise estimators:**
  - from the negative tail, within 4 % with up to 15 % of the pixels covered by objects (the MAD reads +15 %);
  - the clipped-image fallback, within 3 %.
- **The fit:** recovers its own model exactly, in centres, model and RMS widths, given 100 iterations. With the
  default 40, a 2.5 px box stops in the flat L–s valley, with flux and RMS width kept.

### 5.3 Synthetic tests against today's finder (`tests/l2_synthetic_out.txt`)

The images are made like T1's preprocessed frames:
- **Objects:** uniform ellipses, the aperture's image under astigmatism; a line is the degenerate ellipse, brightest
  in its middle. So the detector's box model does not match them exactly.
- **Noise:** as P0b measured it, white 0.91 (cam1) and 0.86 (cam2) plus a texture of 0.99 and 1.21 blurred by
  0.9 px, which reproduces the neighbour correlation.
- **Pipeline:** times L1's gain, rounded and clipped.

SNR is an object's noiseless peak over hp's noise. The C++ finder ran at 3, 4 and 5 noise σ.

**Set A, isolated objects** (6 frames, 1,368 objects, 64 px grid; cam1's noise and gain):

| | lines.detect | C++ 3 σ | C++ 4 σ | C++ 5 σ |
|---|---|---|---|---|
| recall, SNR ≥ 5 | **99.9 %** | 100.0 % | 100.0 % | 99.3 % |
| recall, SNR 3–5 | 85.2 % | 97.4 % | 83.1 % | 47.2 % |
| duplicates per object (objects with one) | **0.00 % (0 %)** | 259 % (70 %) | 183 % (58 %) | 134 % (47 %) |
| false positives per frame | 18 | 1,680 | 50 | 0.3 |
| H lines, SNR ≥ 5: along / across | **0.278 / 0.082 px** | 0.883 / 0.155 | 0.883 / 0.155 | 0.893 / 0.161 |
| V lines, SNR ≥ 5: along / across | **0.297 / 0.094 px** | 0.776 / 0.255 | 0.776 / 0.255 | 0.856 / 0.253 |
| ellipses and round, SNR ≥ 5: u / v | 0.150 / 0.147 px | 0.302 / 0.307 | 0.302 / 0.307 | 0.320 / 0.317 |

**By SNR, lines.detect, H lines (along / across):**
- SNR 5: 0.48 / 0.14 px;
- SNR 7: 0.38 / 0.08;
- SNR 10: 0.27 / 0.06;
- SNR 25: 0.12 / 0.02.

V lines are similar. No duplicates at any SNR. The 18 false positives per frame come from the texture.

**Set B, T1's population.** The population is calibrated so that P0b's analysis of the synthetic hp gives T1's
counts and core lengths. The fitted parameters per frame:
- **cam1:** 4,487 objects, flux median 200, α 13.4, β 9.2 px;
- **cam2:** 6,488 objects, flux median 215, α 12.7, β 9.4 px.

At that density 98–99 % of the objects overlap a neighbour (boxes within 4 px). The results:

| | cam1: lines.detect | cam1: C++ 4 σ | cam2: lines.detect | cam2: C++ 4 σ |
|---|---|---|---|---|
| detections per frame | 1,031 | 9,207 | 826 | 11,748 |
| recall, SNR ≥ 5 (isolated objects only) | 80.2 % (87.1 %) | 99.6 % | 51.4 % (50.8 %) | 98.5 % |
| recall, SNR 8–12 | 94.2 % | 99.3 % | 89.4 % | 98.8 % |
| duplicates per object | **0.00 %** | 118 % | **0.00 %** | 93 % |
| H lines, SNR ≥ 5: along / across (isolated) | 1.06 / 0.30 (0.35 / 0.08) px | 1.10 / 0.26 | 1.15 / 0.30 (0.66 / 0.15) | 1.22 / 0.32 |
| V lines, SNR ≥ 5: along / across | 0.79 / 0.23 px | 0.96 / 0.25 | 0.91 / 0.27 | 1.09 / 0.26 |
| time per frame (prototype) | 23 s | — | 17 s | — |

**At this density the C++ finder's recall means little.** It puts 9,000–12,000 detections per frame into a field
where every footprint overlaps others, so almost anything matches something; 46–52 % of the objects get
duplicates. Without the second pass lines.detect finds 78.4 % at SNR ≥ 5 (instead of 80.2 %). With the plain
(unclipped) model it finds 80.3 %, 37 % faster, but with flux and widths biased low (section 5.1).

**Against the plan's L2 targets:**
- **Isolated objects:** all met at SNR ≥ 5 taken together:
  - recall ≥ 95 %: 99.9 %;
  - duplicates ≤ 1 %: 0 %;
  - across ≤ 0.1 px: 0.08 / 0.09;
  - along ≤ 0.3 px: 0.28 / 0.30.
  At exactly SNR 5 the errors are 0.48 / 0.14 px; the targets hold from SNR ≈ 8–10.
- **T1's density:**
  - recall is not met (80 % and 51 %);
  - the centre errors are not met in crowds (≈ 1 px along, 0.23–0.30 px across);
  - duplicates are met (0 %).
  The cause is overlap, not the noise: isolated objects in the same images do much better.

### 5.4 On T1 (`tests/l2_t1_out.txt`, every 100th frame, 11 per camera)

| | cam1 | cam2 |
|---|---|---|
| lines.detect, objects per frame (median) | 1,001 | 862 |
| classes: H / V / ellipse / round | 27 / 47 / 9 / 17 % | 22 / 46 / 8 / 24 % |
| from the second pass | 23 | 21 |
| C++ finder: local maxima above 3 σ / 4 σ | 18,008 / 10,200 | 18,865 / 12,241 |
| time per frame (prototype) | 36 s | 21 s |

- **The synthetic set B gave 1,031 and 826 per frame,** close to these counts, so it is a fair stand-in for T1.
- **The overlays** (`l2_t1_cam1_overlay.png`, `l2_t1_cam2_overlay.png`, frame 500) show one box per line, ellipse or
  spot, against several C++ maxima on most of them. They also show many faint horizontal lines left undetected:
  their normalised scores stay below 6 against the texture.

## 6. What the shapes say, for L3

`tests/l2_t1_out.txt` and `l2_t1_shapes.png`. Objects with score ≥ 10: 5,336 (cam1) and 4,083 (cam2).

**Blur and lengths:**
- The blur s₀ (median across-width of long lines) is 1.48 and 1.40 px, wider than the 1.0 px assumed in the
  synthetic tests.
- **Fitted box lengths, 10/50/90 %:** H 6.7/12.2/19.9 px and 6.0/12.0/19.9 px; V 6.1/8.9/12.2 px and
  5.8/8.8/12.4 px.
- **Aperture-image semi-axes,** a = 2√(wx² − s₀²), b likewise:
  - H lines a_H 8.7 px (10–90 %: 4.6–13.2) and 8.8 px;
  - V lines b_V 5.4 px (3.6–8.4) and 5.5 px.
  - The two cameras agree.

**The astigmatic locus is not sharp.** With one (a_H, b_V) per camera, objects between the two focal depths would
lie on a/a_H + b/b_V = 1, and those beyond on |a/a_H − b/b_V| = 1. The ellipses scatter instead:
- q = a/a_H + b/b_V has median 1.34 (cam1) and 1.23 (cam2), 10–90 % 0.7–1.9; only 38–45 % lie within 0.7–1.3;
- r = |a/a_H − b/b_V| has median 0.5–0.6.

**Little field dependence:** a_H 8.1–9.6 px and b_V 4.8–5.8 px over a 3 × 3 grid of the image. The class fractions
vary more (V 30–51 %).

**The flux grows with the image's size:** the fitted flux rises from 1,047 (larger semi-axis 0–2 px) to 6,240
(16–30 px) in cam1, and from 759 to 6,830 in cam2. The peak falls only from 88 to 48. Two candidate causes, neither
tested here:
- selection: long lines are seen only when bright;
- the particles may be transparent: polyamide in water acts on the light mostly as a phase object, whose shadow
  contrast grows with defocus.

**Consequences for L3 and L4:**
- Fit the shape model per ray (z_tan, z_sag from the PINCYL model), not one (a_H, b_V) per camera.
- Treat the box model's lengths of ellipses with care: the model is not the ellipse.
- Make the amplitude model empirical: flux against blur, not conserved.
- The depth cue's interval should be wide enough for the scatter of q (≈ ±0.4 about the locus).

## 7. Recommended settings

- **Background:** the per-pixel 90th percentile over all 1,001 frames (every 10th frame gives the same within
  1 count).
- **Preprocessing:** `lines.preprocess` with a 31 px median, recentring, and gains 6 (cam1) and 4 (cam2), or 5 for
  both. Record the noise per frame (L1's manifest does).
- **Frame step:** 4.
- **Detector:** `lines.DET` as in section 5.1, with the noise from L1, per camera:
  - cam1: 8.1 counts at gain 6;
  - cam2: 5.9 counts at gain 4.
- **Today's finder** should not be used on these images. Even on isolated lines it gives 1.3–2.6 extra detections
  per object, with errors of 0.8–0.9 px along the lines.

## 8. Open issues

1. **Crowding** limits recall (80 % / 51 % at SNR ≥ 5) and the accuracy along lines (≈ 1 px). Possible next steps:
   - joint fits of overlapping groups instead of Jacobi passes;
   - an ellipse-shaped model (closer to the real images than the box);
   - a lower threshold for the second pass;
   - the depth cue (L4), to keep only consistent matches.
2. **The texture of faint particles** sets the detection limit for long lines: their templates are 3.5–8.6×
   noisier than white noise predicts. Many faint lines on T1 stay below k = 6. Lowering k to 5 would cost about 11×
   more false positives in pure noise (69 against 6.3 per frame).
3. **About 10 % of a particle's flux is lost** in the preprocessing (synthetic; cause not isolated).
4. **'Cross' components** (8 % of P0b's) were not scored separately. The detector fits them as one or two objects.
5. **Speed.** The prototype takes 15–36 s per frame per camera in Python; the fitting steps take 60–80 % of it.
   Rough operation counts, an estimate only, put a C++ port at well under a second per frame per camera:
   - matched filters: 56 separable templates;
   - fits: 1,000 objects × ~40 iterations × ~360 px × 8 parameters, about 10⁸–10⁹ multiply-adds per pass.
6. **The bias of k = 1 window correlation** (sub-pixel peak locking) was inferred, not measured. The frame-step
   choice does not depend on it.
