# Report: the GUI's cylinder wand calibration on real data — the slow final fit and the wand-point import

**Date:** 2026-10-06 (night). **Plan:** `2026-10-06-fixes-plan.md` (approved: "Go ahead"). **Nothing is committed.**
**7 October:** section 9 adds F7, the fix of a flip that repeats every four rounds, and reruns both runs' variants.

---

## 1. Result

- **The calibration no longer loops; it is 3.5–6× faster on your runs.**
  - **With the thorough start search (six starts):** T0 takes 96 s, down from 576 s, and T2 takes 234 s, down from
    816 s.
  - **With the GUI's default (two starts):** 60 s and 65 s.
  - **Rounds:** no start reaches the 20-round limit any more; the most is 15.
- **The answers:**
  - **T2's result is exactly the delivered one.**
  - **T0's result changes, and I recommend redelivering it** (with your OK).
    - **Why it changes:** T0's delivered camFiles were fitted without 356 good detections. The fixed calibration
      keeps them and leaves out only two frames that fail the outlier test.
    - **By how much:** f −0.33 % and −0.58 %, and the distances from the axis −3.3 and −6.0 mm, under half a standard
      error each, along the f/D line.
- **The GUI's detection now runs on this PC.**
  - Each worker commits 0.33 GB instead of 4.2 GB.
  - A detection of 1,200 T2 frames with the GUI's 25 workers completed; the memory in use rose by 8.7 GB.
- **A detection that pairs no frame now says so.**
  - **The warning:** per camera, the frames with two blobs, one blob and none, and the likely cause (the rod opening).
  - **Loading such a file:** it explains what the file holds and what to do, instead of "Success: Loaded 0 valid
    frames".
  - **After a failure** the GUI no longer asks "Detection complete. Export?".
- **More cameras:**
  - **The weight flip:** never seen with three or four cameras. In the synthetic tests it never fired at all.
  - **The leave-and-return loop:** it did happen with three cameras (32 of the 35 synthetic fits that looped), and
    the fix stops it there too.
- **Tests:**
  - F1 5/5;
  - F2 11/11;
  - F3 and F5 pass;
  - F4: every fit that converged before is unchanged; the 35 that looped now stop in 3–6 rounds with the same
    accuracy; every test's verdict is the same;
  - F6: the GUI tests G3–G9 and pytest are unchanged from before the fixes (pytest: 254 passed, the same 4 old
    failures).

## 2. What was wrong (diagnoses P1–P7, before any change)

**The final fit (stage 4, `cylinder_wand.refine`) looped on runs 20261006/T0 and T2**, so every start ran the full 20
rounds: 10 min (T0) and 14 min (T2) with six starts.
- **Loop 1, the noise weights flipped** (P1). With the GUI's default "Per camera" weights, the estimate alternated
  every round between cam2 clamped at a tenth of cam1 and a balanced state, and the outlier test, judged with the
  clamped weights, rejected 350 (T0) or 600 (T2) good detections every other round. The library falls back to
  equal weights when the cameras' noise cannot be separated, but only once the weights have converged, which never
  happened. T0's delivered fit was made in the rejecting state, without 356 good detections (89 frames).
- **Loop 2, two frames left and came back** (P5, P6), also with equal weights. In the fit, all four detections of
  T0's frames 272 and 284 were 10–15 % over the outlier limit, so the frames were dropped. A dropped frame is
  re-triangulated from its own rays and re-tested with its raw residuals, without the correction for the pose
  having been fitted to those rays (up to 3.8× for these detections), so it passed (0.67–0.90 of the limit for
  coming back), came back, and failed again. T2 showed the same as a swap of two frames. The "Equal" weights I had
  suggested as a workaround therefore did not help.
- **The starts repeated each other** (P7). Five of T2's six starts and all of T0's ended stage 3 with the same
  detections and RMS, up to 2e-4 apart in f along the flat f/D valley, and ran the same stage 4.

**The wand points saved after a detection were all "Raw"** (P4, your description). The GUI's detection saves every
blob as a "Raw" row, then pairs a frame only when every camera has a small and a large ball; only the paired rows
load as wand points. With the GUI's default rod opening (25 px) this wand's rod (76–105 px wide in the images) joined
the two balls into one blob: on 1,200 T2 frames, cam1 found one blob in 557 of 565 camera-frames and cam2 in all
835, the pairing kept no frame, the CSV had 1,408 rows all "Raw", and loading it said "Loaded 0 valid frames" in a
Success box. Nothing in the GUI said the pairing had failed. With the case's settings (95 / 50 px, radius
40–120 px) the same frames paired 159 frames, which load (P3). Two more ways to a raw-only file: Stop, and a failed
detection's rescue export, after which the GUI also asked "Detection complete. Export?".

**The GUI's detection ran out of memory on this PC** (P2): each of its 25 worker processes committed 4.2 GB for
OpenBLAS's buffers (0.33 GB with one BLAS thread), 105 GB in all against the PC's 68 GB.

## 3. What changed

| fix | where | change |
|---|---|---|
| F-a | `curved_calibration/cylinder_wand.py`: `refine`, new `_flips` | When the noise estimate is within 3 % of its value two rounds earlier and more than 30 % off the last round's, the rounds go on with equal weights, as when the groups cannot be separated. Recorded as noise_weights "flipped". |
| F-a2 | `cylinder_wand.py`: `refine`, new `_same_sigma` | When the active set comes back to one it had after an earlier round, with the same weights, the rounds stop; the detections that alternated stay out (in the fit they fail the outlier test), then the last adjustment. Recorded as info "cycle". If the weights are estimated and cannot be separated at that point, equal weights first, as at convergence. |
| F-b | `cylinder_wand.py`: `calibrate(..., share_equal_starts=True)` | A start whose stage-3 result has the same active set as an earlier start's and a stage-3 RMS within 1e-5 (relative) shares that start's stage 4. Recorded as starts "stage4_shared". |
| — | `curved_calibration/cylinder_wand_io.py` | The report's noise section says when the weights flipped (instead of "could not separate"), when the rounds stopped on a repeated active set and how many detections were left out, and how many starts shared a stage 4. |
| F-c | `wand_calibration/wand_calibrator.py`: new `_one_blas_thread_for_workers`, used by `detect_wand_points_generator` and `detect_single_frame` | The detection pools start their workers with OPENBLAS / OMP / MKL _NUM_THREADS = 1; the variables are restored when the pool ends. |
| F-d | `view.py`: `_on_process_finished`, new `_wand_pairing_counts`, `_wand_pairing_failure_text` | After Process All Frames the status line says how many frames were paired. When none, a warning gives per camera the frames with two or more blobs, one, and none, the likely cause (the rod joining the balls: the rod opening; the radius range) and that the file holds only raw detections. "Detection complete. Export?" now comes only after a completed detection, with the paired count. After Stop the status line says the file is not paired yet and names Resume; the rescue question after a failure says so too. |
| F-e | `view.py`: `_load_wand_points_for_calibration` | A file with raw detections but no paired frames gives a warning (its counts, the two causes, what to do) instead of "Success: Loaded 0 valid frames". The loader is unchanged: Process All Frames / Resume reads such files. |

Not changed: the noise model and the outlier rule themselves, the start search, the detector and its default
settings, the pairing rule, the CSV format, the GUI's layout.

## 4. Tests

| test | what | result |
|---|---|---|
| F1 detection memory (`f1_detection_memory.py`) | the GUI's detection on T2 frames 4000–5199 (case settings), the BLAS variables unset as when the GUI starts; its own pool of 25 workers | **pass** 5/5: one worker 0.33 GB (4.22 GB before); the run completes; committed memory +8.7 GB (without the fix about 105 GB by P2's measurement, more than the PC's 68 GB); the same 159 frames and centres as P3 (difference 0) |
| F2 the GUI's messages (`f2_gui_messages.py`) | the fixed handlers offscreen, after P4's detections, a Stop, two failures, and loading P4's raw-only CSV and P3's CSV | **pass** 11/11 (rerun after the last edit of the messages: 11/11) |
| F3 real data (`f3_real_data.py`) | T0 and T2 with per-camera weights (as delivered and the GUI's default), with equal weights, with two starts | **pass**: no start reaches the round limit (at most 9 rounds on T0, 15 on T2); times below; T2's result identical to the delivered one; T0's within half a standard error of it, with 344 more detections |
| F4 synthetic (`f4_synthetic/`) | T11a, T11c–e, T12a–e run on the library before (git HEAD) and after the fixes | **pass**, below |
| F5 shared starts | T0 and T2 with and without F-b | **pass**: the same detections and RMS (to 1e-6 px), cameras 0.0000 σ apart; 96 s against 265 s (T0), 234 s against 432 s (T2) |
| F6 regression (`f6_regression.py`, `f6_regression_out.txt`) | G3, G5/G9, G6, G8, G4 rerun in place; pytest | **pass**: G3 13/13, G5/G9 8/8, G6 6/6, G8 9/9, G4 16/16, each as before the fixes; pytest 254 passed, with the same 2 failures and 2 collection errors as the baseline, all old |

**F3 in detail** (`f3_real_data_out.txt`; calibrations alone on the PC, one BLAS thread):

| run | settings | time (delivered) | stage 4 per start | result |
|---|---|---|---|---|
| T0 | per camera, six starts | **96 s** (576 s) | 9 rounds, 34 s (5 starts shared) | 6,064 of 6,076 detections, RMS 0.469 px |
| T0 | equal weights, six starts | 67 s | 3 rounds, 7 s | the same |
| T0 | per camera, six starts, not shared (F5) | 265 s | 9 rounds, 30–34 s each | the same |
| T0 | per camera, two starts (the GUI's default) | **60 s** | 9 rounds (1 start shared) | the same |
| T2 | per camera, six starts | **234 s** (816 s) | 10 rounds, 49 s; start 4 (another stage-3 minimum) 15 rounds, 78 s | 7,856 of 7,864, RMS 0.601 px: the delivered result exactly |
| T2 | equal weights, six starts | 141 s | 3 and 6 rounds | the same |
| T2 | per camera, six starts, not shared (F5) | 432 s | 10–15 rounds, 47–69 s each | the same |
| T2 | per camera, two starts (the GUI's default) | **65 s** | 10 rounds (1 start shared) | the same |

- **With per-camera weights the estimate flipped in every start** (F-a at round 5; T2's start 4 at round 8), so every
  run ended on equal weights.
- **A repeated active set then stopped the rounds (F-a2),** with 8 detections (two frames) left out: at round 8 (T0),
  9 (T2) and 14 (T2's start 4); with equal weights from the start, at round 2 (T0) and at rounds 2 and 5 (T2).
- **Where T2's 234 s go now:**
  - stage 4: 127 s;
  - the six stage-3 fits and the start search: about 107 s, a share stage 4 used to hide.
- **T0 against its delivered camFiles:** with the 356 detections back and the two alternating frames out, both
  cameras move by under half a standard error, along the f/D line:
  - cam1: f 38,551 px (38,678; −0.33 %, −0.41 σ), 1,006.8 mm from the axis (1,010.1; −0.42 σ);
  - cam2: f 35,812 px (36,022; −0.58 %, −0.46 σ), 1,036.1 mm (1,042.1; −0.47 σ).
- **Against P5** (equal weights before the fixes, which ended with frames 272 and 284 in):
  - cam2: f +0.75 % (+0.64 σ). Those two frames alone moved it that much: f is that weakly determined on this
    data.
  - cam1: +0.06 %.

**F4 in detail** (`f4_synthetic/f4_compare_out.txt`): 1,060 fit records (T12e's ten rigs are summarised, not
recorded per fit), two to four cameras, equal and per-camera weights.

| test | fits | converged before: identical after | ran to the round limit before | now | z RMS of those (error / bound), before → after | verdicts |
|---|---|---|---|---|---|---|
| T11a Gaussian noise | 280 | 280 | 0 | — | — | 8/8 the same |
| T11c one camera 5× noisier | 80 | 77 | 3 | 3 rounds (F-a2) | 1.81 → 1.78 | — |
| T11d heavy tails | 80 | 53 | 27 | 4–6 rounds (F-a2) | 0.94 → 0.94 | — |
| T11e systematic errors | 40 | 38 | 2 | 3–4 rounds (F-a2) | 6.63 → 6.63 | — |
| T12a weights, one camera noisier | 80 | 80 | 0 | — | — | 3/3 the same |
| T12b weights, equal noise | 80 | 80 | 0 | — | — | 3/3 the same |
| T12c two cameras | 320 | 317 | 3 (equal weights) | 3 rounds (F-a2) | 1.11 → 1.11 | 5/5 the same |
| T12d random rigs from scratch | 100 | 70; the 30 from scratch within 5e-7 (relative; F-b chose another of equal starts) | 0 | — | — | 1/1 the same: 50/50 from scratch, within 2.4e-4 σ of the truth-started fit (2.6e-4 before); median 52 s (58) |
| T12e ten rigs at 5 px from scratch | — | — | — | — | — | 1/1 the same: 10/10; median 108 s per calibration (136) |

- **F-a never acted on synthetic data.** In particular not on T12c's two-camera rigs, whose weights converge in up to
  12 rounds: the flip test does not fire where the weights converge.
- **F-a2 acted only on the 35 fits that ran to the limit before** (32 of them with three cameras), leaving out 2–17
  alternating detections each, without changing their accuracy.

## 5. Honest accounting

- **My workaround was wrong.** I had suggested Noise weights "Equal" before checking it. P5 showed loop 2 runs every
  start to the limit with equal weights too, on both runs.
- **I had also said 178 frames** for T0's 356 left-out detections. It is 89 frames, 178 detections in each camera.
- **F1 failed on its first run, through the test itself** (`f1_detection_memory_first_run_out.txt`).
  - **What happened:** the script cleared the BLAS variables at its top level. A spawned worker runs that top level
    again (as `__mp_main__`), so it undid what the pool gave it: 4.22 GB.
  - **The fix:** the clearing moved into `main()`. The library was not changed for it.
  - **The GUI is not affected:** its start script, `GUI.py`, does nothing at its top level but import.
- **F3's first calls ran as smoke tests while F4's sixteen processes ran** (`f3_real_data_smoke_under_load*`). They
  gave the same results, more slowly. The times in section 4 come from the rerun on a quiet machine.
- **The F4 harness patches its copies of the tests,** two short additions asserted in `f4_synthetic.py`:
  - the path to the library before the fixes;
  - two more fields in each fit's record.
- **The before run reproduces the campaign's saved runs,** with two differences:
  - T11's fits that loop now run to 20 rounds, not the 10 they had then;
  - T12c has one more fit that needed 10 or more rounds than its saved output has (10 against 9).
  - Both are differences in the library before this work, not caused by it.
- **G4 (the results pages) joined F6,** because the report's noise section changed.
- **pytest stopped at collection on its first try** (`f6_pytest_first_try_out.txt`).
  - **Why:** plain `pytest -q` stops at the two test files that cannot be imported.
  - **The rerun** used the baseline's command, `pytest tests -m "not slow" --continue-on-collection-errors`, from
    tests/stb_improve's report.
- **G6 passes 6/6 but its process exits with 0xC0000409 at shutdown,** where the earlier report recorded 127 for the
  same pre-existing teardown fault.
  - **The check** (`f6_g6_exit_codes.py`): run four times on git HEAD's code (exported read-only to a scratch folder)
    and four on the fixed code, it exits the same way every time.
  - **So it is not caused by the fixes;** G6 calls none of the changed handlers.

## 6. Limits and not done

- **The flip test's thresholds** (3 % and 30 %) come from T0 and T2. No other real data has tried them yet.
- **Loop 2's cause stays:** a dropped frame is still re-tested more leniently than in the fit. F-a2 stops the loop,
  but does not change that test. The alternative is in the plan, with its trade-off.
- **The flat-window calibration (`plate_wand`) has the same re-admission step.** Not changed here.
- **The GUI's pairing still needs every camera to see both balls,** and the default rod opening is still 25 px. The
  new warning explains when the opening is too small for a wand.
- **A built GUI executable,** if you use one, needs a rebuild to pick these changes up. The source runs them.
- **T0's delivered camFiles and page come from the looping fit.** I have not redelivered them; that waits for your OK
  (section 1).

## 7. Files

**Library and GUI** (uncommitted):
- `modules/camera_calibration/curved_calibration/cylinder_wand.py`: `refine` (F-a, F-a2), `_flips`, `_same_sigma`,
  and `calibrate(share_equal_starts=True)` (F-b).
- `modules/camera_calibration/curved_calibration/cylinder_wand_io.py`: the report's noise section.
- `modules/camera_calibration/wand_calibration/wand_calibrator.py`: `_one_blas_thread_for_workers` and its two uses
  (F-c).
- `modules/camera_calibration/view.py`: `_on_process_finished`, `_wand_pairing_counts`,
  `_wand_pairing_failure_text` (F-d), and `_load_wand_points_for_calibration` (F-e).

**This folder:**
- **Plan and report:** `2026-10-06-fixes-plan.md` and this report.
- **Diagnoses P1–P7:** `p1_log_digest.py`, `p2_worker_memory.py`, `p3_import_roundtrip.py`, `p4_gui_defaults.py`,
  `p5_equal_weights.py`, `p6_readmission_cycle.py` and `p7_stage3_starts.py`, each with its output.
- **Tests F1–F6:**
  - `f1_detection_memory.py` (and its first run);
  - `f2_gui_messages.py`;
  - `f3_real_data.py`, which also runs F5 (and its smoke runs);
  - `f4_synthetic/`: the runner, the comparison, the library before the fixes, and `run_before/` and `run_after/`;
  - `f6_regression.py` (pytest's output in `f6_pytest_out.txt`, its first try in `f6_pytest_first_try_out.txt`) and
    `f6_g6_exit_codes.py`, plus the GUI tests' earlier outputs kept as `*_before_fixes*`.
- **Data in `data/`:**
  - the P4 CSVs and the F1 CSV;
  - the P5 and F3 calibrations: each the GUI's files and its log.

## 8. Afterwards (6 Oct, night): T0 redelivered, the radius range, and a flip F-a misses

**You asked:** "redeliver and rebuild. after redelivering T0, is the result better? also, i want to know why should
we set radius range for ellipse case? i thought that was useless?"

**Rebuild: nothing to rebuild.**
- **Your GUI runs from source:** the shortcut "OpenLPT (new)" runs `D:\Refraction\pyOpenLPT\GUI.py` with the refract
  env, so the GUI loads the fixed Python files at its next start (no GUI was running).
- **The compiled parts are current:** `build\Release\OpenLPT.exe` and the pyopenlpt module are from 6 Oct 10:31, newer
  than every C++ source (the newest, `src\srcSTB\STB.cpp`, 9:17). These fixes changed no C++.

**T0 redelivered** (`cases/cyl_20261006_T0_wand`: calibrate.py, every test rerun, the page republished; the first
delivery kept in `Results\before_stage4_fix\` and the case's `before_stage4_fix\`):

| check | first delivery | redelivered |
|---|---|---|
| detections used | 5,720 of 6,076 | 6,064 of 6,076 |
| RMS (of the detections used) | 0.453 px | 0.469 px |
| free-triangulation wand length, its own frames | 22.1 µm (1,430 frames) | 23.1 µm (1,516 frames) |
| D8: wand length on all 1,519 kept frames | 23.4 µm, mean +0.3 µm | 23.4 µm, mean −0.0 µm |
| D8: the halves held out | 0.07–0.13 %, 25–38 µm | the same |
| X1 (T2's case): T0's cameras on T2's frames | +1.010 px vertical | +1.010 px |
| T7 (T2's case): against T2's fit | cam1 −1.1 %, cam2 +3.9 % | cam1 −0.8 %, cam2 +4.5 % |
| T7: against the 30 Sep dot board | cam1 −6.7 % | cam1 −7.0 % |
| T3: the halves and quarters | cam1 32,800–40,200 px | the same |
| D1, D4–D7 (sync, pose, jitter, length) | | the same conclusions |

- **The answer to "is the result better?":** not measurably, and not worse.
  - **On every independent check** the redelivered cameras agree with the first delivery within its noise.
  - **The cameras slid along the f / distance line** by under half a standard error. The recording cannot fix that
    line, and only a hold-f option (or another camera) would.
  - **What is better is how the result was reached:** one rule for every detection, instead of a fit that ended in
    whichever state its loop stopped in.
  - **The first delivery's lower RMS was not a better fit.** It came from leaving out the 356 worst-fitting
    detections.

**The radius range in the ellipse mode** (`r1_radius_range.py`; T2 frames 4000–5199, rod opening 95 px, mask 50 px):
- **What the detector uses it for** (`ball_ellipses.detect_wand_balls`):
  - the background window, 2.5 × the largest radius (at least 101 px), which must be wider than a ball;
  - a blob filter, areas from π(min/2)² to π(1.5 max)².
- **The measurements:**

  | range | background window | frames paired | centres against 40–120 (RMS) |
  |---|---|---|---|
  | 40–120 px (the case's) | 301 px | 159 | — |
  | 20–200 px (the GUI's default) | 501 px | 151 | 0.02 / 0.04 px |
  | 20–60 px (largest radius below the balls') | 151 px | 108 | 0.02 / 0.01 px |

- **So the range does not move the centres** (0.01–0.04 px, a tenth of the detection noise), **but it sets how many
  frames survive.**
  - **Too small a maximum lost a third of them:** the window became narrower than the large ball.
  - **The default lost 5 %** (cam1).
  - **Best:** the minimum a little below the small ball's radius, the maximum a little above the large one's.

**A flip F-a misses** (found in the rerun of T0's variants):
- **What happened:** fit_k1 (k1 free) and h2 (the second half) still ran their stage 4 to the 20-round limit.
- **Why:** their noise estimates jump by more than 30 % every round, but repeat only every four rounds (fit_k1) or not
  exactly (h2). `_flips` tests for an exact two-round repeat.
- **Their results are identical to the first delivery's;** T0's and T2's main calibrations are not affected (theirs
  repeat every two rounds, caught at round 5).
- **The proposed fix,** not made (it needs your OK): also count as a flip four rounds in a row with a jump of more than
  30 %, and recheck with F3, F4 and these variants.

## 9. 7 October: the four-round flip (F7)

**You said:** "go on and fix them" (section 8's proposal).

**The change** (`cylinder_wand.py`, `cylinder_wand_io.py`):
- **The new rule:** `_flips` now also fires when the noise estimate moved by more than 30 % in some group in each of
  the last four rounds.
- **What it returns:** which rule fired: "alternates" (F-a, unchanged) or "jumps" (the new one, called F-a3 in the
  tests). The thresholds are module constants: `FLIP_CLOSE`, `FLIP_JUMP` and `FLIP_RUN` (0.03, 0.30, 4).
- **What it records:** noise_weights "flipped" has the rule as its "kind". The log line and the report's noise section
  name the rule: "did not converge: it moved by more than 30 % in each of 4 rounds in a row".
- Nothing else changed.

**A correction to section 8 first:**
- **Other T0 variants hit the limit too:** besides fit_k1 and h2, one start each of q1, q3 and D2 (the shape-prior
  calibration) reached the round limit.
- **T2's own variants had never been rerun after the fixes of 6 Oct.** Its halves, quarters and k1 fit had run on the
  library of 6 Oct afternoon, and every start of each had looped. I reran only T2's X1 and T7 after T0's
  redelivery.

**Tests** (outputs beside this report; everything the reruns overwrite was kept first by `f7_keep_before.py`, as
`*_before_flip4*` here and `before_flip4\` in the cases and on X:):

| test | what | result |
|---|---|---|
| F7 the rule offline (`f7_flip_rule.py`) | `_flips` on the noise estimates of every logged stage-4 start of T0 and T2 (100 starts in 28 logs: the delivered fits, T0's first delivery, every variant, F3's runs), and on made-up sequences | **FAIL by its own criteria** (set before the run), for a reason outside the rule: **60 of the 61 starts that reached the limit are caught**, 51 at round 4, 7 at round 5, one each at 6 and 8. The 61st (T0 q1, start 1) is not a flip: its weights sit at the 10:1 clamp, every round's fit stops at its 100-iteration cap, and the RMS still falls (0.71 → 0.60 px). It is a slow fit in the f/D valley, not a loop. The criterion "no converged start is caught" could not be tested: with two cameras every run ends on equal weights. The made-up sequences behave, except that an oscillation damped by only 0.7 per round is caught at round 4 (a damping of 0.5 is not), a limit of the rule. |
| F4 synthetic (`f4_synthetic.py flip4`, `f4_compare.py after flip4`, `before flip4`) | the campaign's T11a, T11c–e and T12a–e on the library with the new rule | **pass**: all 1,060 fits identical to F4's run after the fixes of 6 Oct (to 1e-12); **the new rule never fired**, including on T12c's 320 two-camera fits whose weights take up to 12 rounds to converge; every verdict the same; against the library before all fixes, the comparison is line for line F4's |
| F3 real data (rerun) | F3's eight calibrations (`f3_real_data.py`) | **pass**: no start runs more than 10 rounds. Every result equals the delivered one (T0 and T2; per-camera and equal weights; six and two starts): 0.00σ, f within 0.003 px (8e-8). F5 passes again (0.0000σ). The new rule fired at round 4 in every per-camera run (the old one at round 5). Rounds per start: T0 9 → 8, T2 10 → 9, T2's start 4 15 → 10. Times: T0 72 s (96), T2 158 s (234); with two starts 44 s (60) and 55 s (65); not shared 198 s (265) and 295 s (432). The equal-weight runs, whose code did not change, also ran 17–19 % faster (55 s against 67, 114 s against 141): part of every gain is a quieter machine, and the rounds are the fair measure. |
| T0's variants (`cases/cyl_20261006_T0_wand/tests/t3_t6_variants.py`, `d8_held_out.py`) | all nine, then D8 | **no start at the limit except q1's start 1** (the slow fit). h2: 3,796 of 3,836 detections (3,724), cam1 f 35,229 px (34,548), largest \|z\| 19.4 (26.8). fit_k1: 6,060 of 6,076 detections (5,704), RMS 0.449 px (0.427), k1 +14.3 / +6.7 (+13.8 / +8.7). The others moved in the last digits. All nine took 159 s (290 s). D8: h2's held-out length 36.4 µm RMS (38.2). |
| T2's variants (`cases/cyl_20261006_T2_wand/tests/t3_variants.py`, `t5_k1.py`, `d8_held_out.py`) | all six parts and the k1 fit, then D8 | **no start at the limit** (every start had reached it before). q2 had put cam2 at f 166,557 px and 4.7 m; now f 41,408 px and 1,201 mm, 2.1σ from the full fit. The spread over the parts is cam1 36,521–40,908 px (33,927–41,152) and cam2 32,137–41,408 px (32,137–166,557). k1: +18.5 / −2.9 (+18.8 / −2.9), RMS 0.5795 px (0.5769). Six parts in 128 s (432 s); k1 252 s (884 s). D8: held-out RMS 32–34 µm (32–44), reprojection 0.24–0.30 px (0.24–0.37). |
| T0's D2 (`d2_shape_prior.py`) | the shape-prior calibration, rerun | **no start at the limit** (one had been): the new rule caught it, then a repeated active set stopped it. RMS 0.619 px (0.637), wand length 31.7 µm (31.8); still worse than S2's 0.47 px and 23 µm, so D2's conclusion stands. |
| F7 regression (`f7_regression.py`, `f7_regression_out.txt`) | F6's GUI tests and pytest, rerun | **pass**: G3 13/13, G5/G9 8/8, G6 6/6 (exit 0xC0000409 at shutdown, as before), G8 9/9, G4 16/16, each as in F6; pytest 254 passed with the same 2 failures and 2 collection errors as F6 and the baseline |

**What this means:**
- **No real calibration loops any more,** except the one slow fit below. The real runs finish in 8–10 rounds per start.
- **The delivered calibrations need no redelivery:** the fixed library gives the same camFiles. Only the wording of
  the noise section in their reports would now differ: "did not converge … to round 4" instead of "flipped … from
  round 5".
- **The variants of both runs are now consistent with the library.** T0's page (version 5) and T2's (version 3) were
  republished, and both cases' reports and READMEs and the X: READMEs were updated.
- **T2's page caption** about an off-chart part is now written from the data. It named q2's runaway, which no longer
  exists.

**Limits:**
- **A slowly damped oscillation would be cut short.** If the weights oscillate while converging slowly (damped by about
  0.7 per round or less), the new rule falls back to equal weights. Neither the 1,060 synthetic fits nor the real
  runs show one.
- **The slow fit still runs to the limit.** In T0's q1, start 1 has its weights at the 10:1 clamp, and every round's
  fit stops at its 100-iteration cap. On 184 frames that costs about 20 s; on a full recording it could cost minutes.
  It is a different problem from the loops and is not changed here. A rule for it (for example, equal weights once
  the estimate stays at the clamp for some rounds) would need its own plan.
- **The thresholds** (30 %, four rounds) come from T0 and T2, like the earlier ones.

**Files (F7):**
- **Here:**
  - **New scripts and their outputs:**
    - `f7_keep_before.py`, which keeps every output before it is overwritten, with its output;
    - `f7_flip_rule.py` and its outputs;
    - `f7_regression.py`, with `f7_regression_out.txt` and `f7_pytest_out.txt`.
  - **The earlier outputs, kept:**
    - the GUI tests' outputs from F6, as `*_before_flip4*`;
    - F3's outputs, as `f3_real_data_before_flip4*` and `data/f3_*_before_flip4/`;
    - every real calibration log, in `data/f7_logs/` (F7's input).
- **In `f4_synthetic/`:**
  - the new run, `run_flip4/`, with `f4_flip4_console.txt`;
  - the comparisons `f4_compare_after_flip4_*` and `f4_compare_before_flip4_*`;
  - `f4_compare.py` now takes the two runs to compare.
- **The cases:**
  - in each, `before_flip4/tests/`;
  - T2's `make_page.py`, whose caption is now written from the data.
- **On X:** each run's `Results\before_flip4\`.
- **Library:** `cylinder_wand.py` (`_flips` and the `FLIP_*` constants, `refine`'s record and log line) and
  `cylinder_wand_io.py` (the "why" text).
