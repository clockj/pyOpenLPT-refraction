# The GUI's cylinder wand calibration on real data: the slow calibration and the wand-point import — plan

6 October 2026 (night; revised after your answer). **Status: approved ("Go ahead"); built the same night. Report:
`2026-10-06-fixes-report.md`.** One addition while testing: G4 (the results pages) joined F6, because the report's
noise section changed.

You asked:
- how long a calibration from the detected points usually takes, and why in the GUI "it just keeps going without too
  much improvement";
- why you cannot import the wand points from the GUI's detection. You made it precise: "after detection, if I save
  the detected points, the status is always Raw … if I load that csv it cannot find valid points";
- "what about more than 2 cams?"

You asked me to fix them if they are bugs. They are; this plan says what I found and how I would fix and test it.
Library and GUI code change only after your approval.

## 1. What I found (read-only; scripts and outputs beside this plan)

### A. Stage 4 is caught in two loops, so every start runs to the round limit (`cylinder_wand.refine`)

- **The loop:** stage 4 repeats rounds (an adjustment, each camera's noise estimated from the residuals, the outliers
  judged) and stops when nothing changes. On T0 and T2 something changes every round, in two ways.
- **Loop 1, the noise weights flip** (`p1_log_digest.py`, the delivered calibrations; the GUI's default noise weights
  "Per camera"):
  - **The flip:** the estimate alternates every round between cam2 clamped at a tenth of cam1 (T2: 1.08 / 0.108 px;
    T0: 0.79 / 0.079 px) and a balanced state (T2: 0.58 / 0.85 px; T0: 0.58 / 0.56 px).
  - **What it does to the detections:** judged with the clamped weights, about 600 (T2) or 350 (T0) good detections
    are rejected; the next round brings them back.
  - **Why the library does not stop it:** its rule, equal weights when the data cannot separate the cameras' noise
    (often so with two cameras), is applied only once the weights have converged, which never happens here.
- **Loop 2, two frames leave and come back** (`p5_equal_weights.py`, `p6_readmission_cycle.py`: T0 with equal
  weights):
  - **What happens:** frames 272 and 284 are dropped in one round and re-admitted in the next, for all 20 rounds of
    every start (RMS 0.4726 / 0.4693 px in turn).
  - **Why:**
    - In the fit, all four detections of each frame are 10–15 % over the outlier limit (2.7 px), so the frame is
      dropped. (With two cameras even one rejected detection drops its frame: an end seen by one camera cannot be
      triangulated.)
    - A dropped frame is re-triangulated from its own four rays and tested again with its raw residuals. That test
      leaves out the correction for the pose having been fitted to those same rays, which for these detections
      would raise them by a factor of up to 3.8.
    - So the same detections come out at 0.67–0.90 of the limit for coming back, and come back. In the next fit they
      fail again.
  - **So "Equal" noise weights, the workaround I gave you, does not help:** T0's six starts each still ran 20 rounds.
- **What the loops cost:**
  - **Every start runs the full 20 rounds,** at 4–6 s a round: 80–120 s per start.
  - **With the thorough start search (six starts)** a calibration takes 14 min (T2, 1,966 frames) or 10 min (T0,
    1,519 frames). Stage 4 is 90 % of that time.
  - **In the GUI's log** the RMS alternates for minutes. This is the "keeps going without too much improvement".
- **Loop 1 changed a delivered result.**
  - **T0:** all six starts ended in the rejecting state. T0's camFiles were fitted without 356 detections (89
    frames, 178 in each camera): those with cam2's largest residuals, judged against a clamped noise of 0.079 px.
  - **T2:** its odd start ended in the other state and won, with 7,856 of 7,864 detections.
- **The starts repeat each other.** Five of T2's six starts, and all of T0's, reach the same stage-3 RMS (0.502 and
  0.408 px) and then run the same stage-4 rounds.

### B. More than two cameras

- **Loop 1 comes from having two cameras.** With two cameras, how each residual splits between the cameras is fixed
  by the weights, except through the wand-length constraint (`tests/cylinder_wand/p12_sympy_noise_weights.py`,
  PROVED). The noise estimate then follows the weights it was computed with, and the rounds can bounce. With three
  or more cameras each point's residual separates the cameras' noise strongly: the estimate hardly depends on the
  weights, and the rounds settle.
- **Loop 2 does not need two cameras, only a frame dropped whole.**
  - With two cameras any rejection drops its frame.
  - With three or more, a frame is dropped whole only when an end is left with one camera: a frame that only two
    cameras see, or a frame that does not fit anywhere.
- **Evidence (synthetic, the saved runs of `tests/cylinder_wand`):**

  | campaign | cameras | weights | fits | stage-4 rounds | at the round limit |
  |---|---|---|---|---|---|
  | T12a (one camera 5× noisier) | 3 | per camera; per camera and end | 80 | 5–9 | 0 |
  | T12b (equal noise, 0.1 and 1 px) | 3 | per camera | 80 | 3–6 | 0 |
  | T12d (T5's random rigs, from the truth and from scratch) | 2 / 3 / 4 | per camera | 20 / 50 / 30 | 3–8 / 4–6 / 4–6 | 0 |
  | T12c (one camera 5× noisier, or equal noise) | 2 | per camera | 160 | 3–12 | 0 |
  | T12c | 2 | equal | 160 | 1–2 | 3 |
  | T11a (Gaussian noise, 0.1–5 px) | 3 | equal | 280 | 1–2 | 0 |
  | T11c (one camera 5× noisier) | 3 | equal | 80 | 1–3 | 3 (limit then 10) |
  | T11d (heavy tails: 5 % of detections at 5×, or Student-t) | 3 | equal | 80 | 3–6 | **27** (limit then 10) |
  | T11e (systematic errors) | 3 | equal | 40 | 1–3 | 2 (limit then 10) |

  - No fit with per-camera weights and three or four cameras flipped (loop 1).
  - The fits at the limit with equal weights are loop 2: with no weights to flip, only the active set can change.
    In T11d, 97 % of the good detections rejected went with whole frames (`t11d_rejection_split`), because frames
    seen by two of the three cameras lose an end.
- **The limits:**
  - These are synthetic.
  - There is no real data with three or more cameras through the tube yet.
- **The fixes below stop both loops with any number of cameras.**
- **Time** grows with the number of detections (frames × cameras): a round costs 4–6 s at T0/T2's 6,000–7,900.
- **One more limit with more cameras, not changed here:** the GUI's pairing keeps only frames in which every camera
  finds both balls; the cylinder calibration itself accepts frames that some cameras miss. In G3 (three cameras,
  200 rendered frames) the pairing dropped 3 frames because one camera saw the two balls merged into one blob.

### C. "The status is always Raw": the detection's pairing found no frame (`p4_gui_defaults.py`)

- **How the GUI's detection works:** two steps.
  1. Every image gives its blobs: the "Raw" rows of the CSV.
  2. **Pairing:** per camera, a radius histogram finds the small and the large ball; a frame is kept when every
     camera has one of each. Its two balls become the "Filtered_Small" and "Filtered_Large" rows. Only these rows
     are calibrated, and only these the loader reads as wand points.
- **P4, the GUI's default ellipse settings** (rod opening 25 px, rod mask 18 px, radius 20–200 px) on T2 frames
  4000–5199:
  - cam1 found one blob in 557 of 565 camera-frames and two in 8; cam2 one blob in all 835;
  - this wand's rod (2 mm; 76–105 px wide in these images, T0's test T1 d) is wider than the 25 px opening, so it
    joins the two balls into one blob;
  - the pairing's radius histogram then finds one peak per camera where it needs two (cam1 92.7 / 94.5 px, cam2
    81.9 / 81.8 px), and no frame has two balls in both cameras, so it keeps none ("Pass 1 Done. Valid Frames:
    0 / 1200", in the console only);
  - the CSV: 1,408 rows, all "Raw"; the GUI's loader: "Loaded 0 valid frames" in a Success box.
- **With the case's settings** (opening 95 px, mask 50 px, radius 40–120 px) the same frames give clean peaks
  (cam1 65.5 / 88.2 px, cam2 58.5 / 79.2 px) and 159 paired frames, and their CSV loads them (P4's third setting,
  as P3). Their 636 paired rows start at line 2,875, after the 2,874 raw ones.
- **P4, the GUI's default mode, "fast"** (radius 20–200 px): 398 of the 1,200 frames paired, and the CSV loads them;
  17 min for the 1,200 frames on 9 workers (the ellipse mode: under 3 min). But "fast" fits circles, and through the
  tube its centres are off (T13: calibrations from them were off in f by a few to tens of per cent); the cylinder
  calibration warns about points from it.
- **Nothing in the GUI says the pairing failed.**
  - The status line says "Done: Detection completed."
  - The only hints are the console's "Pass 1 Done. Valid Frames: 0 / 1200" and the frame list, all in red
    strike-through.
- **Two more ways to a file with only Raw rows:**
  - **Stop:** the autosave is written before the pairing.
  - **A failed detection** (D below): the rescue export. After a failure the GUI also asks "Detection complete. Do
    you want to export the point data?", which writes the same raw-only file.
- **In a good file the paired rows are at the end,** after every raw row (`export_wand_data` writes the raw ones
  first). Opened in Excel, such a file shows thousands of "Raw" lines first.

### D. The GUI's detection runs out of memory on this PC (`wand_calibrator`)

- **The workers:** the detection runs in a pool of 25 worker processes (80 % of 32 cores).
- **What each one loads:** a worker imports the GUI's package: PySide6, matplotlib, cv2, scipy and numpy. Its numpy
  sets up OpenBLAS for all 32 cores.
- **The cost (`p2_worker_memory.py`):** each worker commits **4.2 GB** (0.33 GB with one BLAS thread). 25 workers need
  about 105 GB, and this PC allows 68 GB (64 GB RAM and a 4 GB page file).
- **What happened in my test:** the GUI's detection on 1,200 frames of T2 failed with "The paging file is too small"
  and a broken process pool. I do not rerun that here: it starves every other program of memory too.
- **Then:** "Detection Failed … partial data was captured … export it?", and the raw-only file of C.

### E. Equal weights on T0 and T2, and the delivered results (`p5_equal_weights.py`)

- **The delivered fits** (their `calibration_result.json`):
  - **T0:** weights applied (0.579 / 0.563 px, the balanced state), 5,720 of 6,076 detections;
  - **T2:** equal weights (the library's fallback at the round limit), 7,856 of 7,864.
- **T0 with equal weights:** every one of the six starts ran 20 rounds (loop 2: frames 272 and 284).
  - **Result:** 6,072 of 6,076 detections, RMS 0.473 px.
  - **Against the delivered:** f 38,528 / 35,545 px, that is −0.39 % / −1.33 % (0.5 and 1.0 of their standard
    errors); distances from the axis 1,006.7 / 1,028.9 mm against 1,010.1 / 1,042.1.
  - **So T0's answer moves by up to one standard error** when its 356 detections come back, close to the f/D line of
    T2's report (cam2: −13.2 mm and −477 px, 36 px/mm against the line's 34.5).
- **T2 with equal weights:** the same loop, as a swap: each round one frame (4 detections) leaves and another comes
  back; 20 rounds for each of the six starts.
  - **Result:** 7,860 of 7,864 detections, RMS 0.602 px.
  - **Against the delivered** (which also ended on equal weights): f +0.01 % / −0.28 %, distances 1,001.3 / 1,082.1 mm
    against 1,001.2 / 1,085.1. T2's delivered answer stands.
- **Time with equal weights:** 495 s (T0) and 493 s (T2), against the delivered 576 s and 816 s: a little faster per
  round, still 20 rounds a start.

## 2. The fixes

| | where | change |
|---|---|---|
| F-a | `cylinder_wand.refine` | **Loop 1.** When the noise estimate flips between two states, fall back to equal weights at once, as the library already does when it cannot separate the noise. The test: every sigma within 3 % of its value two rounds earlier, and one more than 30 % off the last round's. In T0 and T2 it fires at round 5. |
| F-a2 | `cylinder_wand.refine` | **Loop 2.** When the active set comes back to one it had in an earlier round with the same weights, the rounds stop. The detections that alternate stay out (in the fit they fail the outlier test), and the last adjustment is made without them. The log and the report say how many. |
| F-b | `cylinder_wand.calibrate` | A start whose stage-3 result matches an earlier start's (the same active set, and a stage-3 RMS within 1e-5, relative) is not refined again; the earlier start's stage 4 is reused. `p7_stage3_starts.py`: T0's six starts end stage 3 with the same active set and RMS within 3e-6, at points up to 2e-4 apart in f and 0.18 mm in position (about 0.02 of their standard errors), along the flat f/D valley; all six then reach the same stage-4 answer (the delivered log, and P5). T2's starts 0–3 and 5 likewise (RMS within 6e-8, f within 2e-5); its start 4 stops stage 3 elsewhere (RMS 0.703 px) and keeps its own stage 4. So T0 would refine once instead of six times, T2 twice. |
| F-c | `wand_calibrator.detect_wand_points_generator`, `detect_single_frame` | The detection workers start with one BLAS thread (OPENBLAS / OMP / MKL _NUM_THREADS = 1, set while the pool runs, then restored): 0.33 GB each instead of 4.2 GB. The detection itself does not use BLAS threads. |
| F-d | `view._on_process_finished` | After Process All Frames, the GUI says how many frames were paired. When none: per camera, in how many frames it found one blob and in how many two or more; the likely cause (the rod joining the balls: set the rod opening a little wider than the rod's image and check with Test Detect; the radius range); and that the saved file holds only raw detections. "Detection complete. Export?" is asked only after a detection that completed. After Stop, the status line says that the file is not paired yet and that Resume finishes it. |
| F-e | `view._load_wand_points_for_calibration` | Loading a file with raw detections but no paired frames gives a warning instead of "Success": what the file holds, why, and what to do (stopped: Process All Frames → Resume with it; the pairing found nothing: new settings and detect again). The loader itself is unchanged: Resume needs it to read raw-only files. |

**F-a against the flat-window calibration's rule.** `plate_wand` never estimates weights with two cameras (its test
W5 found them drifting). F-a keeps them where they converge: T12c's two-camera rigs with one camera 5× noisier got
0.72–0.76 of equal weights' error from them. On T0 and T2 both give equal weights.

**F-a2 against a fix of loop 2's cause.**
- **The alternative:** test a dropped frame the way the fit tests it, with the correction included.
- **Why F-a2 instead:**
  - The alternative changes which frames come back in every fit, including those that converge now.
  - It could also keep good detections out for good once a bad state of loop 1 has rejected them; today they come
    back ("Rejections are not final", T5).
  - F-a2 changes nothing where the rounds converge, and in a loop keeps out only what the fit itself rejects.

**Expected:**
- T0's and T2's calibrations in about 2–4 min with the thorough search (now 10 and 14), every good detection kept
  except frames that fail the outlier test. [F3 measures it.]
- The GUI's detection runs on this PC.
- A detection that pairs nothing says so, and why.

**Not changed:**
- the noise model and the outlier rule themselves;
- the start search;
- the detector, its default settings and the pairing rule;
- the CSV format;
- the GUI's layout.

**Options, not in this plan unless you want them:**
- the CSV with the paired rows first (they are now at the end);
- a "suggest" for the rod opening: Test Detect tries several openings on the current frame and proposes the smallest
  that separates the balls in every camera (the case found 91–101 px by such a scan);
- pairing that keeps frames seen by two or more cameras (section 1 B), for rigs with three or more cameras;
- the same stop for loop 2 in the flat-window calibration (`plate_wand`): it has the same re-admission step. It
  already falls back to equal weights with two cameras, and when the weights drift (its test W5).

## 3. Tests (F1–F6, beside this plan, each with its output)

| test | what | judged by |
|---|---|---|
| F1 detection memory | The GUI's detection on T2 frames 4000–5199, with the BLAS variables unset in the shell (as the GUI is started), the library fixed. | completes; one worker commits ≤ 0.5 GB (P2's measurement after the fix); the same 159 frames and centres as P3 |
| F2 the GUI's messages | The fixed handlers, offscreen, with their dialogs answered by the test: after P4's default-settings detection; after the case-settings detection; after a Stop; after a failure; loading P4's raw-only CSV and P3's CSV. | the counts and the cause in the text; no "Detection complete" after a failure; the raw-only file refused with the new message; P3's CSV loads its 159 frames as before |
| F3 real data | T0's and T2's calibrations (the cases' inputs, the GUI's settings: per-camera weights, six starts), and with equal weights, with the fixed library; against the delivered ones. | no start reaches the round limit; time; detections kept; RMS, f, distances and the free wand length, compared |
| F4 synthetic | `tests/cylinder_wand` T11a, T11c–T11e and T12 (two to four cameras; equal and per-camera weights) rerun. | fits from a given start that converged before: the same numbers; fits from scratch (F-b can pick another of equal starts): within F5's tolerance; fits that ran to the limit: now stop early, with their accuracy and rejections against before |
| F5 starts | T0 and T2 with and without F-b. | the same active set; cameras within 0.01 of their standard errors (and within the spread of the six starts' own answers) |
| F6 regression | pytest (baseline: 254 passed and the 4 known failures), and this folder's G3 (the GUI path) and G5–G9. | unchanged |

- **Expected time:** about 2 h, most of it F3 and F4.
- **After F3:** if T0's calibration with its detections back differs materially from the delivered one, I would
  redeliver T0's camFiles and its page (and T2's if it changes), with your OK.
