# Noise weights per camera, and more starts at high noise: report (T12)

Date: 2026-10-02. Plan: `2026-10-02-weights-plan.md` (approved, "go ahead"). It comes from
`2026-10-01-noise-report.md`, section 6, items 1 and 2.

In the library, two files changed:
- `modules/camera_calibration/curved_calibration/cylinder_wand.py`: the weights and the starts;
- `cylinder_wand_io.py`: the report's noise section, added at the user's request ("so that I can know what actually
  happens"; section 2).

Tests and outputs are in this folder, and nothing is committed. Nothing in `paper/`, the C++ library, the GUI or the
case folder was touched.

## 1. Result

The user asked whether the noise weights need user input. They do not: the calibration estimates each camera's noise
from the residuals.

| case | before (equal weights, T11) | after (T12, the new default) |
|---|---|---|
| one camera at 5× the noise (cam2 0.5 px, others 0.1 px): accuracy per camera against the best possible | ×1.99 / 1.76 / 2.24 | **×1.00 / 1.05 / 1.05** |
| … the same: error bars, z RMS per camera (1 = honest) | 0.87 / 1.09 / 0.95 | 1.01 / 1.06 / 1.06 |
| … f error per camera | 0.16 / 0.25 / 0.19 % | **0.09 / 0.18 / 0.10 %** (best possible 0.09 / 0.17 / 0.09 %) |
| … 3-D shape error / relative to the tube | 2.1 µm / 51 µm | **1.0 µm / 23 µm** |
| … good detections rejected (of 120 000) | 89 | 14 |
| wand end A at 2× the noise of B: accuracy per camera | ×1.22 / 1.20 / 1.19 | ×0.96 / 0.99 / 1.03 |
| equal noise, 0.1 px and 1 px | — | unchanged (×1.003, ×0.998; estimates move 0.03–0.04σ) |
| two cameras, one at 5× (45° / 90° apart) | — | weights adopted in 35 / 40 of 40 data sets; **×0.76 / ×0.72** of equal weights' error |
| two cameras, equal noise | — | unchanged (×1.00) |
| from scratch, T5's 50 rigs at 0.1 px | 50/50 | 50/50 |
| from scratch, ten rigs at 5 px | 9/10 | **10/10** |
| time of a calibration from scratch | — | ×1.09 median (×1.07–1.24), like for like |

How to read it:
- "Accuracy" is the realised error over the correctly weighted Cramér–Rao bound; 1 is the best any calibration can
  do.
- The estimated noise levels are within 2–5 % RMS of the truth. That is the scatter their own predicted standard
  errors give, and their means lie between −1.6 % and +1.1 %.

## 2. What changed in `cylinder_wand.py`

**The weights.**
- **`variance_components(sol, groups, sig2, ...)`, new.** Each noise group's variance comes from the fitted
  residuals, by MINQUE: τ = M⁻¹q.
  - Per frame, with the whitened residual operator R̃ = I − W^½J(JᵀWJ)⁻¹JᵀW^½ (the frame's 5 columns), q_g is the
    group's whitened squared residuals and M_gh = Σ tr(D_g R̃ D_h R̃).
  - It also gives each σ's relative standard error, √((M⁻¹)_gg/2).
  - "separable" is False when a relative standard error exceeds 10 % or an estimate is not positive. The estimates
    are returned anyway, with a non-positive one clamped, and `refine` decides.
  - It also records the separate estimates (`sigma_px_separate`) and the standard errors per original group
    (`rel_se_group`), even when they are not used.
  - Groups with fewer than 200 active detections share one variance.
  - The σ are kept within a factor of 10 of each other and above the 1e-3 px floor.
- **`refine(..., noise_weights="camera", sigma_cam_px=None, min_group=200, max_rel_se=0.10, max_ratio=10.0,
  sigma_tol=1e-6)`, with `max_rounds` now 20.**
  - **Modes:** "camera" (the default), "camera_end" (per camera and wand end), "none" (equal weights: the old
    behaviour), or given σ per camera.
  - **Each round:** adjust with the current weights, estimate the variances, judge outliers, and repeat until both the
    active set and every σ are steady.
  - **Separability is judged once the weights have converged.** The standard errors mean something only when the
    weights are about right (defect 4, section 4). If the groups cannot be separated even then, the fit is redone
    with equal weights.
- **`bundle_adjust(..., weights=None)`:** weighted least squares. The reported RMS stays the unweighted one in pixels.
- **`_update_active(..., sig_obs=None, groups=None)`:** the leverage is the weighted one, P = w J_f U_f⁻¹ J_fᵀ. Each
  observation is judged against its own group's robust (median) scale, with the weights of the fit being judged.
- **`covariance(..., weights=None)`:** s0² S_w⁻¹, with the variance factor s0² (median 1.005 in T12a). `sigma_px`
  still reports the pooled unweighted σ.
- **`information(..., weights=None)`** and **`triangulate(..., row_weights=None)`:** the stale frames are
  re-triangulated with the weights.
- **`Solution.weights`:** the weights the fit used.
- **What `sol.info["noise_weights"]` records:**
  - whether the weights were used, fell back ("fallback") or ran out of rounds ("converged");
  - whether the groups were separable;
  - the σ per group (used, and the separate estimates) and their relative standard errors.
- **What `calibrate`'s `sol.info["starts"]` records:** the bootstrap starts short-listed and tried, whether extra
  starts were used, and each one's stage-3 RMS.

**More starts.**
- `calibrate(..., more_starts_px=1.0)`: when the best of the top `n_final` starts still has a stage-3 RMS above
  1 px, every short-listed start is carried through stages 3–4.
- The winner keeps the most observations, then has the highest likelihood: the lowest Σ_g n_g log(mean squared
  residual of group g). With one group that is the RMS order, as before.
- An existing call keeps working unchanged and gets the new defaults. `noise_weights="none",
  more_starts_px=float("inf")` restores the old behaviour exactly; the T11 helpers in `t11_common.py` now pass this,
  so T11 reruns mean what they meant.

**The report** (`cylinder_wand_io.noise_summary`, `noise_text`).
- `write_result` now adds a "noise" block to `calibration_result.json` and a section "Noise levels and weights" to
  `calibration_report.txt`.
- **One row per noise group (camera, or camera and wand end):**
  - the estimated noise σ (px per coordinate) and its relative standard error;
  - detections used out of the total, and how many were not used (rejected, or dropped with their frame);
  - the residual RMS (px per coordinate);
  - the robust scale outliers were judged against.
- **Then:**
  - whether the weights were applied, and if not, why (for example "largest relative standard error 15.7 %,
    limit 10 %"), with the estimates still listed and marked "not used";
  - the pooled σ, the variance factor, the rounds and the noisiest-to-quietest ratio;
  - how many bootstrap starts were tried.
- `t12f_report.py` writes complete result folders for five cases into `t12f_report/<case>/`, so the reports can be
  read as a user would read them. One of them (three cameras, cam2 at 0.5 px):

```
Noise levels and weights (noise_weights='camera')
  weights APPLIED: estimated from the residuals; each detection weighted by 1 / sigma^2 of its group
  group          noise sigma  +- rel. SE    used / total  not used  residual RMS  outlier scale
                        (px)                                         (px/coord.)           (px)
  cam1                0.0980       3.8 %    1000 / 1000          0        0.0597         0.0977
  cam2                0.4930       1.6 %    1000 / 1000          0        0.4867         0.4910
  cam3                0.1042       3.5 %    1000 / 1000          0        0.0662         0.1022
  pooled sigma 0.3750 px per coordinate (unweighted); variance factor 1.005 (...); 5 rounds; noisiest / quietest group 5.03x
```

- Two cameras with only 150 frames show the fallback:
  - "weights NOT applied: the data could not separate the groups' noise (largest relative standard error 15.7 %,
    limit 10 %)";
  - the unused estimates are listed: 0.112 and 0.499 px.
  - With equal weights the quiet camera's residual RMS (0.19 px) is larger than its own noise (0.11 px), because it
    shows part of the noisy camera's error. The report's notes explain this.

**SymPy** (`p12_sympy_noise_weights.py`, output `p12_sympy_noise_weights_out.txt`): 35 checks, all PROVED.
- **Identifiability:**
  - one point seen by two cameras: the split of the residual between them does not depend on their noise levels;
  - a wand frame seen by two cameras: the split does depend on them, through the length constraint only;
  - one point seen by three cameras: it depends strongly.
- **The estimator:**
  - R̃ symmetric and idempotent;
  - E[q] = Mτ;
  - Cov(q) = 2M at τ = 1;
  - Σ_h M_gh = the group's redundancy;
  - the weighted residual covariance used by the outlier test, Cov(r) = W⁻¹ − JU⁻¹Jᵀ, and its blocks.
- **How they were checked:** exact rational examples, with the variances symbolic and two weight sets. With symbolic
  square-rooted weights SymPy's simplification did not finish in 20 minutes.

## 3. Tests

| test | what | result |
|---|---|---|
| T12a | T11c's unequal noise: the same 40 data sets and starts, with the weights; 5 from scratch | c1 **pass**. c2 passes everything but one criterion (section 5). From scratch 5/5 and 5/5 |
| T12b | T11a's data at 0.1 and 1 px, with the weights, against T11a's equal-weight fits of the same data | **pass**. σ within 2.2–3.3 % RMS; RMS z ×1.003 / ×0.998; z (fit) std 1.025 / 0.965 |
| T12c | two-camera rigs (45°, 90°), one camera at 0.5 px or both at 0.1 px; each data set fitted with and without weights | **pass** (never worse): ×0.762 / 0.999 / 0.718 / 0.999 |
| T12d | T5's 50 rigs from scratch with the new defaults | **pass** 50/50 (largest difference from the truth-started fit 2.5e-4σ) |
| T12e | T11b's ten rigs at 5 px from scratch with the new defaults | **pass** 10/10 (2e-5σ, RMS within 7e-8 px) |
| T12d timing | six T5 rigs, old against new behaviour, under the same load | ×1.07–1.24, median ×1.09 (about 3–7 s per calibration) |
| T12f | the report's noise section in five cases (`t12f_report/<case>/`) | **pass** 5/5: weights applied in four cases, with the σ within 4 of their standard errors of the truth; one fallback, explained |
| pytest | `tests/test_cylinder_wand.py` (the export test now also checks the noise section) | 7 passed |
| full suite | `pytest` (`t12_full_suite_out.txt`) | 236 passed, 2 failed, 7 skipped, 2 errors. Against the baseline, one new failure, not caused by this work (section 5) |

More detail:
- **T12a c1 (noise_weights "camera"):**
  - σ estimates RMS 4.9 / 1.8 / 3.7 % (truth 0.1 / 0.5 / 0.1 px);
  - accuracy against the weighted bound 1.00 / 1.05 / 1.05 (±0.06–0.07);
  - z against the fit's own covariance 1.01 / 1.06 / 1.06;
  - 14 rejections in 120 000;
  - 5–9 rounds per fit.
- **T12a c2 (noise_weights "camera_end", six groups):**
  - accuracy 0.96 / 0.99 / 1.03;
  - z 0.97 / 1.00 / 1.04;
  - 15 rejections in 120 000 (T11c: 72).
- **T12c, two cameras:**
  - With one noisy camera the weights are adopted in 35/40 (45°) and 40/40 (90°) data sets. The quiet camera's
    relative standard error is near the 10 % limit (median 9.0 % / 7.7 %), and in the 5 cases above it the fit
    falls back to equal weights.
  - The weights cut the error to ×0.71 / 0.81 per camera at 45° and ×0.69 / 0.75 at 90°. At 90° the weighted fits
    reach the weighted bound (RMS z 1.01).
  - With equal noise the weights are adopted in 40/40 and change nothing.

## 4. Defects found while testing, and fixed

1. **Rejection judged with weights the fit had not used.**
   - In the first version the first round's rejection used the newly estimated weights while that round's adjustment
     had used equal ones.
   - On a c1 smoke test this threw out 30 good detections, against 3 with equal weights.
   - Fixed: the rejection always uses the weights of the fit it judges; the new variances weight the next round.
     Stale frames are re-triangulated with the weights.
2. **Weights not converged tightly enough.**
   - Iterating the variances to 1 % left runs from different starts on weights up to 1 % apart, and so on answers up
     to 2e-3σ apart.
   - T12e's first pass (`t12e_scratch_5px_first_pass_*`) failed 8/10 by that: rigs 4 and 7 reached the same minimum,
     with the same active set, but their RMS differed by 7e-5 and 1e-5 px.
   - A tolerance of 1e-4 gave 9/10 (`..._second_pass_*`; rig 7: 2e-4σ). Rig 7's far camera is nearly telecentric,
     so its flat f–distance direction amplifies small weight differences.
   - 1e-6 gives 10/10 for 1–2 more rounds.
   - T12a–d's first runs used 1 % (`*_first_pass_*`); all of T12 was rerun on the final code.
3. **SymPy statements.** Matrix-valued claims came back UNCHECKED (the checker samples scalars), so they were restated
   as sums of squared entries.
4. **Separability decided at the first round.** Found while adding the report section, by reading why some
   two-camera fits had kept equal weights although their recorded standard errors were 4 %.
   - The first round fits with equal weights, and from that fit a quiet camera next to a noisy one can get a
     negative variance estimate: the noisy camera's error swamps it.
   - The code had marked such data "not separable" for good, although one round later the estimates were fine.
   - The standard errors mean something only when the weights are about right. So now negative estimates are clamped
     and iterated, and separability is judged at convergence; if the groups cannot be separated then, the fit is redone
     with equal weights.
   - **Effect, T12c:** weights adopted in 35/40 and 40/40 instead of 25/40 and 33/40; error ×0.76 and ×0.72 instead
     of ×0.86 and ×0.76.
   - The three-camera tests, where the first round always separates, are unchanged.
   - Those cases need up to 10 rounds, so `max_rounds` went from 10 to 20.
   - The earlier T12 outputs are kept as `*_before_convergence_fix_*`. All of T12 was rerun on the final code.

## 5. Criteria, and where they were restated (honest accounting)

**T12a c2, "estimated σ within 5 %": literal FAIL.**
- One of six groups (camera × wand end, about 500 detections each) scattered 5.01 % RMS over 40 data sets.
- The estimator's own predicted standard error for that group is 4.56 %. The limit sat at the estimator's precision,
  so it was a criterion set too tight, not a defect.
- Across all groups, realised over predicted scatter is 0.88–1.33, and the means are −1.6 to +1.1 %.
- Accuracy, error bars, rejections and the starts from scratch all pass.
- The precision of the noise estimate barely matters for the fit, because weights act at second order: T12b's
  estimates move only 0.03σ against equal weights.

**Full suite: `tests/test_refract_oracles.py::test_vendored_cylinder_oracle_matches_notes_repo` now fails.**
- It checks that pyOpenLPT's vendored `test/refs/project_cylinder.py` is byte-identical to the notes repository's
  `D:\Refraction\code\project_cylinder.py`.
- That file was modified on 2026-10-02 at 01:09 and is uncommitted. This session did not touch it; the vendored copy
  is unchanged since 2026-09-24.
- Re-copy the vendored file once that edit is final, as the test asks. I did not, because it is your file.
- The other failures are the known pre-existing ones listed in pyOpenLPT's CLAUDE.md.

**Time.**
- T12d's median, 54 s against T5's 31 s, compares 16 processes at once with one, so it measures the load, not the
  code. Like for like (T12d timing), the cost is ×1.09.
- At 5 px the extra starts make a calibration about 2.9× longer (113 s against 39 s under the same parallel load).
  That happens only when the stage-3 RMS exceeds 1 px.

## 6. Limits and not done

- **The weights do nothing against an image-fixed pattern (T11e).** That remains the main risk on real data.
- **Each camera's noise is assumed uniform over its image.** Noise that varies with position or depth, as blob size
  does, is not modelled.
- **Two-camera rigs:** whether the noise levels separate depends on the geometry and the number of frames. Above the
  10 % standard-error limit the weights are skipped, and the report says so. More frames help: with 150 frames
  (T12f case 4) the quiet camera's error was 15.7 %.
