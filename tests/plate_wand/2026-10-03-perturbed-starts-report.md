# Perturbed restarts of the final fit: report (R1–R6)

Date: 2026-10-03. Plan: `2026-10-02-perturbed-starts-plan.md`, approved 2026-10-03 ("go ahead and implement the plan").
Code changed, as planned: `modules/camera_calibration/curved_calibration/plate_wand.py` and one report line in
`plate_wand_io.py`. Tests: this folder and two new tests in `tests/test_plate_wand.py`. Nothing else was touched.

**Status: implemented and tested.**
- R4 failed its criterion: 2 of 5 misses fixed, against at least half.
- The change to how a restart's result is accepted (section 7) was approved ("ok, go ahead and compare") and
  implemented.
- R7 compares it with the first version (section 10). It fixes the blocked case and makes nothing worse. R4's strict
  count stays at 2 of 5, with 4 of 5 now at least equivalent.

## 1. Summary

- **What changed.** On a rig that uses several windows, `calibrate` now restarts the final fit from four perturbations
  of its answer.
  - The perturbations are drawn from the answer's own covariance, at twice its σ.
  - Each is screened by one bundle adjustment.
  - The cheapest is refitted in full if it lowers the cost.
  - One-window rigs and given starts are unchanged.
- **What it fixed.**
  - **W5 seed 2000.**
  - **W5 seed 2008:** from an equivalent answer to the same minimum as the fit started from the truth.
  - **R4 seeds 3029 and 3100 at 1 px:** from a different answer to the same minimum.
  - **R4 seed 3111 at 0.5 px:** from a different answer to an equivalent one.
  - **W5 at 1 px:** 7 of 10 rigs now reach the same minimum (5 before) and 10 of 10 an equivalent answer (9 before).
- **Nothing got worse,** in 60 W5 rigs, 60 R4 cases and 12 W7 cases.
  - The 37 one-window rigs are unchanged to the last digit.
- **Cost:** the restarts add a median 9 % to a W5 multi-window calibration (8 s, at most 30 s) and 13 % to an R4 one
  (at most 82 %: 184 s on 5 cameras and 1,373 frames).
- **Why these misses happen (new).** Three of the four answers that the restarts brought to the global minimum had
  been **pinned against a window**: W5 seed 2000 and R4 seeds 3029 and 3100. A wand end lay exactly on the window's
  water-side surface, at a depth of 3e-13 to 9e-10 mm. The fourth, W5 seed 2008, had already been an equivalent
  answer.
  - The final fit rejects any step that would take a wand end out of the water. So it stops on that surface, where it
    is a minimum only because the fit cannot cross it.
  - **In numbers,** across W5's multi-window rigs and R4:
    - 8 of the 14 answers that missed the global minimum were pinned;
    - none of the 69 that reached it was.
- **R4 failed its criterion.** Two of its 5 misses were fixed, against at least half.
  - **Seed 3111 at 1 px:** a restart reached the global minimum's RMS (1.231995 px against the truth-started fit's
    1.231979), but `calibrate`'s selection rule kept the pinned answer. That rule prefers the answer that keeps more
    observations, and the restart kept 13,728 against 13,730.
  - **Seed 3111 at 0.5 px:** reached an equivalent answer.
  - **Seed 3110 at 1 px:** was not pinned; it is an ordinary wrong minimum that the draws do not leave.
- **Proposed, then approved and implemented** (sections 7 and 10): accept a restart when it is better on the same terms
  as the answer. Every usable observation counts, its squared residual capped at the 5σ rejection limit.
  - Of 89 cases with restarts, this changed 2:
    - R4 seed 3111 at 1 px, from a different answer to an equivalent one;
    - W5 seed 1031, still wrong, from 5.29 to 3.59 px.
  - Nothing got worse.
  - Equivalent or better: 59 of 60 R4 cases (58 with the first version, 55 without restarts).

## 2. What R1 found

### R1: sizing on W5 seed 2000 (`r1_sizing.py`, `r1_sizing_out.txt`)

- **The rig and its answer.** Seed 2000 has 4 cameras behind 2 windows, 2 per window, at 1 px.
  - **The global minimum** (the fit from the truth): 1.155353 px.
  - **The answer from scratch:** 1.158848 px, its largest difference 3.9σ (camera 3's centre).
  - **The distance between them** is 16.0σ of the answer in the multivariate (Mahalanobis) metric, over 28 parameters.
    That is far, although no single parameter is more than 3.6σ off.
- **Two centres,** each with 20 draws at s = 1, 2, 3 and 4 times the answer's covariance:
  - the answer;
  - the joined start it came from (4.9σ from the answer).
- **Each draw was:**
  - screened by one bundle adjustment at the answer's observations and noise weights;
  - then fitted in full, both from the screened result and from the draw itself.
- **Control:** a screen started at the global minimum ends 0.65 % below the answer's cost. The screen therefore ranks
  the global minimum first.

| centre, scale | per-draw success | of the draws whose observations all projected |
|---|---|---|
| answer, s = 1, 2, 3 | 0.45 | 9 of 9 |
| answer, s = 4 | 0.40 | 8 of 9 (the ninth screened 0.45 % below the answer, but its full fit ended at 1.1706 px) |
| joined start, s = 1, 2 | 0.45 | 9 of 9 |
| joined start, s = 3 | 0.50 | 10 of 10 |
| joined start, s = 4 | 0.40 | 8 of 9 |

- **Each failed draw ended its screen with observations that did not project,** 4–3,076 of them, and the screen's cost
  left them out. At s = 1 and 3, R1b shows they failed from the draw's start.
- **Every draw whose observations all projected reached the global minimum** at s ≤ 3.
- **The screen ranked the draws correctly** in every cell: each draw that reached the global minimum was cheaper than
  every draw that did not.
- How far a draw stepped towards the global minimum did not predict its success. The same 9 draws succeeded at s = 1,
  2 and 3.
- **Timing,** with all 32 processes loaded: a screen took a median of 3.7–4.4 s, and a full fit 11–17 s.

### R1b: why half the draws failed (`r1b_failed_draws.py`)

- **Every failed observation was a wand end outside the water.** That is, h_W ≤ 0 behind window 1, seen by cameras 1
  and 3.
- **What decides it is where a draw leaves window 1 relative to the re-triangulated wand ends.** At s = 1:
  - after every valid draw, the nearest wand end lies 3.5–33 mm inside the water;
  - after every failed one, it lies 0.9–39 mm outside.

  Most failed draws had moved window 1's offset towards the cloud (7 of 11). The others' normals and cameras moved
  enough to do the same.
- **At the answer, one wand end sits on window 1's water-side surface,** 3.5e-12 mm deep. The next nearest are 0.6 and
  0.8 mm deep.

| | nearest wand end to window 1's water side |
|---|---|
| the answer from scratch | 3.5e-12 mm (1 of 1,538 ends within 0.01 mm) |
| the global minimum | 19.7 mm |
| the true wand box | 55.7 mm |

- **The answer is pinned against window 1.** The bundle adjustment rejects a trial step that makes a projection fail,
  like one that raises the cost.
  - Here every step that would lower the cost takes that wand end out of the water. So the fit stops on the surface,
    at a point that is not a minimum of the cost.
  - Its last bundle adjustment converged normally (1 iteration, accepted), so nothing in the fit itself shows it.
- **This is what makes the basin look narrow** in `d4_seed2000.py`.
  - A draw that moves the window off the cloud frees the fit, and stage 4 then descends to the global minimum.
  - A draw that pushes the window further into the cloud cannot be screened fairly: the observations that no longer
    project drop out of its cost.
- **A boundary point depends on the path.** With 4 BLAS threads instead of 1, R4b's pinned answer came out at
  1.244745 px instead of 1.245158 px.

## 3. The change

In `plate_wand.py`:
- **`perturbed_draws(sol, obs, n, scale, rng)`** draws the perturbations.
  - Each draw is dx = scale · L z, with sol.cov = L Lᵀ (camera 1's pose held) and z ~ N(0, I). It is applied as stage 4
    applies a step, and the frames are re-triangulated.
  - A draw is drawn again, up to 10 times, when a camera leaves the air side of its window or **an active observation
    no longer projects**.
- **`perturbed_restart(...)`** runs one round of restarts.
  - **The screen:** each draw gets one bundle adjustment at the answer's active observations and noise weights. Its
    cost is set against the answer's, on the same terms.
  - **The full fit:** the cheapest draw, if it is more than 1e-6 cheaper (relative), goes through the whole of stage 4
    on fresh observations, as a start does.
- **`calibrate(..., perturb_starts=4, perturb_scale=2.0, perturb_all=False)`** runs the restarts on rigs that use
  several windows.
  - `perturb_starts=0` turns them off.
  - `perturb_all=True` applies them to every rig, and to a given start.
  - **First version:** the restart's result joined the starts' results under the existing rule, now the function
    `_choose`.
  - **Since section 7 was approved:** it replaces the answer when its `capped_cost` is more than 1e-6 lower
    (`info["accepted"]`, `info["capped_rel"]`). `_choose` still picks among the starts.
  - A second round follows when the restart is accepted; `PERTURB_ROUNDS = 2`.
  - The draws' seed is the CRC32 of the detections, so the same data give the same answer.
- **`window_depths(...)`** returns each window's smallest wand-end depth: near 0, the answer is pinned.
- **`capped_cost(sol, obs, weights, s02, k_sigma)`** scores an answer on every usable observation. Each observation's
  weighted squared residual is capped at k_sigma² s02; an observation that does not project counts the cap; priors
  are added as in the bundle adjustment.
- **Recorded:**
  - `info["starts"]["perturbed"]`: the draws, scale, redraws, screened costs, the depths before, the RMS before and
    after, whether the restart was chosen, the seconds;
  - `info["timing_s"]["perturbed"]`.

In `plate_wand_io.py`, `calibration_report.txt` gets one line. For seed 2000 it reads: "restarts perturbed about the
answer: 8 draws at 2 sigma in 2 round(s); the answer IMPROVED, rms 1.15885 -> 1.15535 px (its start had ended in a
wrong minimum)".

**The scale and the number of draws.**
- **The scale, s = 2.** Every valid draw succeeded at s = 1, 2 and 3 (9 of 9 each); s = 2 is the middle of the three.
- **The number of draws, K = 4.** The plan's rule gives K = 1 here, from p = 9/9.
  - The 95 % lower bound on that p is 0.72, from 9 draws on one rig. At that bound K = 3 gives 98 %.
  - One more draw is for rigs harder than seed 2000.
  - Four screens cost about one to one and a half final fits (R1).

## 4. Tests

### R2: the draws alone (`r2_draws.py`): PASS

- **Rig B's answer,** 2,000 draws at s = 2:
  - each parameter's sample σ is 0.964–1.030 of 2σ (the target: within 5 %; the sampling error is 1.6 %);
  - the largest correlation difference is 0.081 (the sampling error about 0.022, across 378 pairs);
  - re-expressed through the log maps, each draw equals its dx to 5.7e-14 (the first 200);
  - camera 1's pose is untouched in every draw;
  - no redraws, 11 ms a draw.
- **Determinism:** the same detections give the same seed and draws. One detection moved by 1e-9 px gives other draws.
- **Seed 2000's pinned answer** (calibrated with `perturb_starts=0`; the same 1.158848 px and a depth of 3.5e-12 mm):
  - 200 draws were kept and none left out;
  - a try is kept 52 % of the time;
  - every kept draw has every camera on the air side and every active observation projecting.

### R3: W5 rerun (`w5_from_scratch.py`, compared by `r3_compare.py`): PASS

| | 0.1 px: same | 1 px: same | 1 px: equivalent |
|---|---|---|---|
| before | 46/50 | 5/10 | 9/10 |
| after | 46/50 | 7/10 | 10/10 |

| criterion | result |
|---|---|
| seed 2000 at the global minimum | PASS (1.158848 → 1.155353 px, max \|z\| 3.8e-7) |
| no rig worse | PASS |
| one-window rigs unchanged (RMS to 1e-12, the same active count and verdict) | PASS, 37 of 37 |
| median time increase on multi-window rigs ≤ 25 % | PASS: the restarts take a median 9 % of the rest of the run (at most 21 %) |

- The two whole runs differ by a median of +5 % in time, but wall times drift 10–20 % between runs here.
- **The restarts changed 6 of the 23 multi-window answers:**

  | seed | before → after | RMS (truth-started fit) | pinned before |
  |---|---|---|---|
  | 2000 (1 px) | different → same | 1.158848 → 1.155353 (1.155353) | yes |
  | 2008 (1 px) | equivalent → same | 0.877174 → 0.877146 (0.877146) | no |
  | 2004 (1 px, one camera per window) | equivalent → equivalent | 0.863713 → 0.860512 (0.860768) | yes |
  | 2005 (1 px, one camera per window) | equivalent → equivalent | 0.864359 → 0.852606 (0.852625) | yes |
  | 1034 (0.1 px) | different → different | 1.841685 → 0.178590 (0.117511) | yes |
  | 1001 (0.1 px) | different → different | 0.087867 → 0.087479 (0.085946) | no |

- **Notes on these rigs:**
  - **Seed 1034** was freed from the pinned window into a second wrong minimum. Its second round's draws all returned
    to that minimum (relative cost −1e-13).
  - **Seeds 2004 and 2005** end at or below the truth-started fit's RMS.
    - Their first round's answers were still pinned. With two cameras at 1 px, the data's best fit may itself touch the
      window.
    - Seed 2004's answer keeps 16 observations fewer than the truth-started fit.
  - **Seed 1031** (0.1 px): a restart was refitted from 5.29 px to 3.57 px, but the selection rule kept the 5.29 px
    answer, which keeps more observations. Both are wrong (the truth-started fit gives 0.109 px), so nothing got worse.

### R4: 30 new rigs with two or more cameras on every window (`r4_group_rigs.py`): FAIL

- **The cases:**
  - seeds 3008–3135, the first 30 random rigs with 2–3 windows and two or more cameras on each;
  - each at 0.5 and 1 px, with the same frames;
  - each calibrated without the restarts (the code before the change) and with them: 120 calibrations in 603 s.

| | 0.5 px misses | 1 px misses | equivalent or better | worse |
|---|---|---|---|---|
| without the restarts | 1 of 30 | 4 of 30 | 55 of 60 | |
| with them | 1 of 30 | 2 of 30 | 58 of 60 | 0 |

| case | without → with | pinned before | why |
|---|---|---|---|
| 3029, 1 px | different → same | yes (window 2, 1.8e-12 mm) | fixed |
| 3100, 1 px | different → same | yes (window 1, 9.3e-10 mm) | fixed |
| 3111, 0.5 px | different → equivalent | yes (window 2, 5.6e-11 mm) | the right RMS (0.615991 against 0.615974), 12 observations fewer |
| 3111, 1 px | different → different | yes (window 2, 3.0e-10 mm) | a restart reached 1.231995 px, but the selection rule kept the pinned answer (R4b) |
| 3110, 1 px | different → different | no | an ordinary wrong minimum (max \|z\| 47); the draws return to it |

- **Pinning and misses:** 4 of the 5 misses were pinned, and none of the 55 cases that reached the global minimum was.
- **No case is worse:** PASS.
- **At least half of the misses fixed:** 2 of 5, **FAIL**. With the equivalent answer it is 3 of 5.
- **Time:** the restarts take a median 13 % of the rest of the run (at most 82 %). With against without, the median
  whole run is +13 %.

### R4b: the blocked fix (`r4b_blocked_restart.py`, a diagnosis)

- **Seed 3111 at 1 px, recomputed single-threaded as in R4:**
  - the pinned answer: 1.245158 px, 13,730 observations, window 2's nearest wand end at 3.0e-10 mm;
  - one round of restarts: the screens lowered the cost by 1.4–2.1 %, and the cheapest was refitted to 1.231995 px with
    13,728 observations.
- **`_choose` keeps the pinned answer,** because it keeps 2 more observations.
- **The 2 observations the restart dropped do not project under it:** their wand ends lie just outside window 2's
  water side. So a comparison on the answer's own observations would also refuse the restart, since its cost there
  would leave them out.
- **The capped cost of section 7 keeps the restart:** 20,888.28 against 21,287.24, which is −1.9 %. The restart pays
  the cap for its 2 dropped observations.
- **The restart against the truth-started fit:** max |z| 0.70 and 2 observations fewer. That is equivalent, not the
  same minimum.

### R5: W7 rerun (`w7_bad_data.py`, compared by `r5_compare.py`): PASS

- **All 12 answers are identical** to the run before: the RMS, the active observations and the z against the clean
  fit, to the last digit.
- **Rig A** has one window, so it has no restarts.
- **Rig B** had one round in each scenario, with no change, in 3–4 s.
- **W7's own verdict is unchanged:** rig B "partial" still misses its 1.5× limit (1.11 against 0.72 expected), at its
  data's own minimum.
- `w7_bad_data.py` now also records the restarts' information.

### R6: pytest: PASS

- **`tests/test_plate_wand.py`:** 9 passed in 88 s (7 fast, 2 slow). Two of them are new:
  - `test_perturbed_draws` (fast): the seed and the draws are deterministic, camera 1 is held, every draw projects, and
    the spread is right;
  - `test_restarts_free_a_pinned_answer` (slow): seed 2000 from scratch reaches the global minimum, and the restarts
    report the improvement.
- **The full suite:** 245 passed, 7 skipped, 2 failed and 2 collection errors, all four known and unrelated:
  - the radius test `test_p1_round4_radius_frame_mismatch` (CLAUDE.md);
  - the oracle-drift test: `D:\Refraction\code\project_cylinder.py` changed in commit 8a76a83 (2 Oct), and its vendored
    copy in pyOpenLPT is left unchanged, as you asked;
  - the two modules that cannot be imported (CLAUDE.md).

## 5. Cost

| | the restarts' share of the rest of the run (median, max) | seconds (median, max) |
|---|---|---|
| W5, 23 multi-window rigs | 9 %, 21 % | 8.2, 30.3 |
| R4, 60 cases | 13 %, 82 % | 14.5, 184 |

- **A screen costs about a third of a final fit** on small rigs, and more on large ones. On R4 seed 3111, at 13,730
  observations, the four screens took 50–65 s.
- **The worst case,** seed 3111 at 0.5 px, ran two rounds: four screens and a full fit each.
- **Without improvement** the restarts cost one round of four screens (W5's median: 8 s).

## 6. Deviations from the plan

1. **The redraw rule was widened.** A draw is drawn again also when an active observation no longer projects, not only
   when a camera leaves the air side.
   - Such a draw's screened cost would leave those observations out, so it cannot be compared with the answer's.
   - On a pinned answer these draws are exactly the ones that push the window further into the cloud (R1b).
2. **K = 4 rather than the plan rule's K = 1** (section 3), at s = 2.
3. **Added for diagnosis:**
   - `window_depths`, and the depths before each round in the restarts' information;
   - R1b and R4b, diagnoses not in the plan.
4. **W7's script records the restarts' information** (output only).
5. **R1 was sized from the full fits** as planned. Its screening data also show that the screen alone ranks the draws
   correctly; the planned code relies on that.

## 7. Proposal: accept a restart on the same terms (approved and implemented; results in section 10)

Decision, 3 Oct: "ok, go ahead and compare". The proposal follows as it was put.

**The problem.**
- The restart's result joins the starts' results under `_choose`. Within 1.5× of the best RMS, `_choose` prefers the
  answer that keeps the most observations.
- That rule was made for starts, to stop a wrong minimum that rejects much from winning on its RMS.
- Against a pinned answer it is the wrong test.
  - A pinned answer keeps every observation because it holds the window against the cloud.
  - The right minimum may drop one or two observations whose wand ends lie just outside the water at its windows.
- Seed 3111 at 1 px lost its fix this way, and W5 seed 1031 a smaller improvement.

**The rule proposed.**
- Keep the restart's result instead of the answer when C(restart) < (1 − 1e-6) C(answer), where C is:
  - C(X) = Σ over every usable observation of min(w_i |r_i(X)|², 25 s²);
  - w is the answer's noise weights (1 when they are equal), and s² the answer's variance per coordinate (its variance
    factor with weights);
  - an observation that does not project at X counts 25 s²;
  - each answer has its own fitted frames.
- **The 5σ cap is the rejection limit.**
  - An outlier counts the same in both answers.
  - A restart pays for every good observation it drops.
  - A restart that rejects much to lower its RMS cannot win.
- **The starts' rule (`_choose`) stays as it is.** Only the restart-against-answer decision changes. That is about 15
  lines in `calibrate` and a helper.

**The evidence.** R4b: the restart's 20,888 against the pinned answer's 21,287, so the restart is kept.

**The expected effect.**
- R4 seed 3111 at 1 px becomes an equivalent answer at the global minimum's RMS (max |z| 0.70, 2 observations fewer).
  R4 would then have 2 misses fixed and 2 more made equivalent, out of 5.
- The strict criterion (the same minimum) would still count 2 of 5. A restart that ends with a wand end just outside the
  water drops it, so it cannot keep exactly the truth-started fit's observations.

**Then:** rerun R3, R4, R5 and R6 (about 25 min), and update this report, the README and the figures.

**The alternatives:**
- keep the code as it is now (2 of 5 fixed and 1 equivalent in R4; no harm anywhere);
- or remove the restarts (`perturb_starts=0` by default), which gives up seed 2000, 2008, 3029 and 3100.

## 8. Open items

1. **The section 7 proposal:** done (section 10).
2. **Wrong minima with wide basins are not addressed:** W5 seeds 1001, 1031, 1045, 1034's second minimum, and R4 seed
   3110. They need better starts (README, open item 1), not restarts.
3. **Pinning as a warning.** `window_depths` near 0 at the answer means a wand end sits on a window's surface.
   - In these tests that marked the misses (8 of 14) and no answer that reached the global minimum (0 of 69).
   - The report could warn of it, for example "a wand end lies on window 2's surface: the answer may be pinned; check
     that window's position". It would be one more report line, not yet done.
4. **Two cameras at 1 px (seeds 2004 and 2005):** the data's best fit may itself touch a window. Pinning is then
   not an error but a sign of weak data: the window's position is uncertain by more than the gap between the wand and
   the window.
5. **The cost on large rigs.** On R4 seed 3111 (5 cameras, 13,730 observations) a screen takes 12–16 s under full
   load. Two rounds there cost 189–194 s, about as much as the rest of the calibration.
   - The second round is worth its cost: it improved 3111 at 0.5 px (0.6191 → 0.6160 px, to an equivalent answer),
     W5 seed 2005 slightly, and seed 1031.
   - A cheaper option is to cap the screen's iterations: it took 15–35 in R1. Not done.

## 9. Files

| file | what |
|---|---|
| `2026-10-02-perturbed-starts-plan.md` | the plan (approved) |
| `r1_sizing.py`, `_out.txt`, `.json`, `_console.txt` | R1: sizing on seed 2000, 160 draws |
| `r1b_failed_draws.py`, `_out.txt` | R1b: the failed draws and the pinned window |
| `r2_draws.py`, `_out.txt`, `.json`, `_console.txt` | R2: the draws alone |
| `w5_from_scratch.*`, `r3_compare.py`, `_out.txt`, `.json` | R3: W5 rerun and its comparison |
| `r4_group_rigs.py`, `_out.txt`, `.json`, `_console.txt` | R4: 30 new multi-window rigs, with and without |
| `r4b_blocked_restart.py`, `_out.txt`, `_console.txt` | R4b: the blocked fix and the proposed rule |
| `w7_bad_data.*`, `r5_compare.py`, `_out.txt` | R5: W7 rerun and its comparison |
| `r7_acceptance.py`, `_out.txt`, `.json` | R7: the two acceptance rules and no restarts, compared |
| `runs_v4/` | the W5 (v4) and W7 (v2) runs before this change |
| `runs_v5/` | the runs of the first version (`_choose` judged the restarts): W5 v5, W7 v3, R4 v1, with R3's and R5's comparisons |
| `tests/test_plate_wand.py` | R6: the three new tests (`test_perturbed_draws`, `test_capped_cost`, `test_restarts_free_a_pinned_answer`) |

## 10. R7: the acceptance rule compared (after your approval)

The capped-cost acceptance of section 7 was implemented (`capped_cost`; `perturbed_restart` sets `info["accepted"]`).
Then W5, R4 and W7 were rerun and pytest run again. `r7_acceptance.py` compares three versions on the same data:
- **before:** no restarts (`runs_v4/`, and R4's runs without them);
- **choose:** the first version, its refit judged with the starts by `_choose` (`runs_v5/`);
- **capped:** the restart accepted when its capped cost is lower (the current code).

R4's runs without restarts came out identical in both R4 runs, to 1e-12 in RMS.

| cases with restarts | same minimum: before → choose → capped | equivalent or better | different |
|---|---|---|---|
| W5, 23 multi-window rigs | 14 → 16 → 16 | 18 → 19 → 19 | 5 → 4 → 4 |
| R4, 60 cases | 55 → 57 → 57 | 55 → 58 → 59 | 5 → 2 → 1 |
| W7, 6 rig-B scenarios | 6 → 6 → 6 | 6 → 6 → 6 | 0 → 0 → 0 |

**The capped rule changed 2 of the 89 answers:**

| case | choose | capped |
|---|---|---|
| R4 seed 3111, 1 px | the pinned answer kept, 1.245158 px (DIFFERENT) | the restart accepted: 1.231995 px, 13,728 observations (the truth-started fit: 1.231979 px, 13,730), equivalent. Capped cost −1.9 % in round 1; round 2 refitted 1.231998 px at +5.5e-6, not accepted |
| W5 seed 1031, 0.1 px | 5.292115 px kept | 3.588919 px: round 1 −50 %, round 2 −6.7 %. Still wrong (truth-started: 0.109 px) |

- **Every other answer is unchanged,** to the last digit.
- **No answer is worse than before the restarts.**

**The plan's criteria, on the current code:**

| test | result |
|---|---|
| R3, W5 | PASS on all four: seed 2000 at the global minimum; no rig worse; one-window rigs identical; the restarts cost a median 9 % (8.1 s, at most 32 s) |
| R4 | no case worse: PASS. At least half the misses fixed: 2 of 5, **FAIL**; at least equivalent: 4 of 5 (3029, 3100, 3111 at 0.5 and at 1 px). Only 3110 remains, not pinned |
| R5, W7 | PASS: all 12 answers identical to the run before the restarts |
| R6 | `tests/test_plate_wand.py`: 10 passed (8 fast, 2 slow), including the new `test_capped_cost`. Full suite: 246 passed, 7 skipped, the same 2 known failures and 2 collection errors |

- **Why 3111 counts as equivalent, not the same minimum.** Its accepted answer drops 2 observations whose wand ends lie
  just outside the water at its windows (R4b), against the truth-started fit's 13,730. Their parameters differ by up to
  0.70σ. The capped rule cannot change that: the drop is the refit's own.
- **Cost.** On R4 the restarts take a median 14 % of the rest of the run.
  - On seed 3111 at 1 px the second round now also runs, after the accepted first: 194 s of restarts, 109 % of the
    rest.
  - Open item 5 has the cheaper option.
- **Conclusion.**
  - The capped rule does what section 7 intended: it accepts the found fix that `_choose` blocked, and changes nothing
    else except one wrong answer made less wrong.
  - With it, R4's misses go from 5 to 1 by the equivalent measure, and from 5 to 3 by the strict one.
