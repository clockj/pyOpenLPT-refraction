# Noise weights per camera, and more starts at high noise: plan (T12)

Date: 2026-10-02. Status: **approved by the user on 2026-10-02 ("go ahead"); implemented and tested.** Results, the one
criterion that failed literally and why, and the changes from this plan (the σ tolerance went from 1 % to 1e-6, and
there were 35 SymPy checks instead of 7) are in `2026-10-02-weights-report.md`. The body below is the plan as
approved. It comes from `2026-10-01-noise-report.md`, section
6, items 1 and 2. The user asked whether the noise weights need user input or can be judged during the calibration.

Scope:
- Only `modules/camera_calibration/curved_calibration/cylinder_wand.py` changes, with tests in this folder.
- Nothing in `D:\Refraction\paper`, the C++ library, the GUI or the case folder.

## 1. Noise weights per camera, judged from the data (no user input)

**The finding (T11c).** One camera at 5× the others' noise is not rejected and keeps roughly honest error bars. But
equal weights cost every camera 1.7–2.1× in σ, because the noisy camera degrades the others through the shared wand
positions.

**The change: estimate each camera's noise level as the calibration runs.** This is variance component estimation,
the standard method in photogrammetric bundle adjustment.
1. **Measure.** After each adjustment, take a robust σ per camera from its own active residuals. They are already
   leverage-corrected (the Mahalanobis distances of `_update_active`), so the frames' share is accounted for.
2. **Reweight and refit.** Weight each detection by 1/σ² in the adjustment (`normal_blocks` already takes weights)
   and refit. Repeat until no σ moves by more than 1 %, typically 2–4 rounds.
3. **Reject per camera.** Outliers are judged against their own camera's σ, not the pooled one.
4. **Covariance from the weighted fit.** Each camera's error bars then use its own σ.

Further details:
- **Optional groups by wand end:** for balls of different size (T11c c2), groups by camera and end.
- **Guards:**
  - weights stay within 1:100 of each other (σ within 1:10), so an early bad round cannot switch a camera off;
  - a σ is estimated only from a group with at least 200 active detections.
- **Optional input:** per-camera σ given by the user overrides the estimate; it is never required.

**Can the data tell the cameras apart?** SymPy, 7 checks, all PROVED (`p12_sympy_noise_weights.py` in this folder,
output `p12_sympy_noise_weights_out.txt`):
- **One point seen by two cameras.** The residual space is one-dimensional. The residuals then measure only the two
  cameras' combined noise. How the residual splits between the cameras is fixed by the geometry and does not depend
  on which camera is noisier.
- **A wand frame seen by two cameras.** Here the length constraint does separate them, but weakly. In the checked
  example the split moves from 0.78 to 1.01 when a 25:1 noise ratio is swapped between the cameras.
- **One point seen by three cameras.** The split moves from 0.72 to 2.73 for the same swap.

So: with three or more cameras the noise levels are well determined. With two cameras they are determined only
through the wand's length. The code will measure how well each σ is pinned down (the variance components' own
standard errors) and keep equal weights when the data cannot separate the cameras. That is today's behaviour, so the
change can only help.

## 2. More starts when the noise is high

**The finding (T11b).** At 5 px one rig in ten fell into a near-telecentric wrong minimum. The right start was fourth
on the short list, but only the best two are carried forward. `n_final=6` fixes it at about +45 s.

**The change.** When the best start's stage-3 RMS exceeds 1 px, carry every short-listed start through stages 3–4.
The selection rule is unchanged (most observations, then lowest RMS). Low-noise runs keep today's speed.

## 3. Tests (T12), with criteria

| test | what | criterion |
|---|---|---|
| T12a | T11c again (cam2 at 0.5 px; end A at 0.2 px), 40 data sets each, with the weights | realised σ / weighted bound 0.8–1.2 per camera (was 1.7–2.1); z per camera 0.8–1.2 against the fit's own covariance; estimated σ within 5 % of the truth |
| T12b | equal noise: T11a at 0.1 and 1 px, 40 data sets each | no loss: weights within 5 % of equal; ratios and z as in T11a |
| T12c | a two-camera rig, one camera at 0.5 px | reported: the estimated σ, their standard errors, whether the fallback triggers; never worse than equal weights (accuracy ratio ≤ 1.05) |
| T12d | T5's 50 random rigs from scratch at 0.1 px | 50/50 still reach the global minimum; run time reported |
| T12e | T11b's ten rigs at 5 px, with the automatic extra starts | 10/10; run time reported |
| — | pytest subset and the full suite | no new failures |

The ratio windows use ±3 standard errors for quantities driven by one mode (the lesson of T11a's 0.2 px heights).

Deliverables:
- the code change;
- `t12*_*.py` with their outputs;
- `2026-10-02-weights-report.md`;
- the figures page updated.

Cost: about 30–40 min of computing in parallel.
