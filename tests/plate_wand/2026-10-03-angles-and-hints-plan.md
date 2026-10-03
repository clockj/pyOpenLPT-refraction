# Windows at other angles, and window-normal hints for the failed cases: plan (A1, H1)

Date: 2026-10-03.

Status:
- approved 2026-10-03, revised;
- A1 done (PASS); H1a done (FAIL) and diagnosed;
- the revision H2 (section 5) approved ("Ok go ahead"), built, and H2a run (FAIL), then diagnosed;
- the further revision H3 (section 6) approved with the third start: "go ahead to try to fix it as much as possible.
  in the end, report when the calibration cannot work so that i can pay attention to in the real experiments". How
  it is built, and what follows it: section 7;
- H3 built; H3a run: no hinted run worse (0 of 65), 48 of 65 at least equivalent, strict criterion FAIL on windows
  behind a single long-lens camera;
- D8 run (section 7.3a); H4, the scan of weak windows' normals, built; its smoke test fixes W5 seed 1001 without
  hints; H4a run (FAIL; section 7.3b), refined (H4b: the acceptance rule, a 300-frame scan) and H4b run: none worse
  with or without hints, 56 of 65 hinted runs at least equivalent, 3110 and 1001 right without hints; strict
  criterion FAIL on seed 1045 alone;
- H5, the warnings in the report, built (section 7.4); H5a run: PASS (118 of 120 answers without hints unchanged, 2
  better, none worse; at 0.1 px every wrong answer had a warning, 3 of 47 right ones did);
- H1b run on the final design: PASS (none of 120 worse with hints 0–10° off; the 3 still wrong without hints right
  with them);
- the final report for real experiments: `2026-10-03-when-calibration-fails.md`. Done, apart from the user's review.

Results: `2026-10-03-angles-and-hints-report.md`. It answers your questions "How to avoid the failed
cases? Have you tested for two windows with different angles?". Your decision on the first draft: "For H1, start with
the second part for hints on window normal, but that hints is not guaranteed to be super accurate. The reason why we
do not start with the first part is that it is hard to guess for f in real experiments. Now you can go ahead for A1
and H1's 2nd part". So the focal-length hints (the first draft's H1 part 1) are dropped, and H1 now tests hints with
errors up to 30°.

Scope:
- **A1** is a test only: a new script in this folder, no change to the calibration code.
- **H1** adds a window-normal hint to `plate_wand.py` (section 2), plus one report line in `plate_wand_io.py` and fast
  tests in `tests/test_plate_wand.py`.
- Nothing in `D:\Refraction\paper`, the C++ library, the GUI, the case folder or `cylinder_wand*.py`.

## 1. What is known

**Window angles.**
- Every multi-window rig so far used walls of a box tank:
  - two adjacent walls, 90° apart (rig B, W5's and R4's two-window rigs);
  - in three-window rigs, also the opposite wall, 180°.
- No other angle has been tested.
- Nothing in the code assumes 90°:
  - each window has its own normal, searched within 50° of its cameras' viewing direction;
  - window groups are joined through their common wand positions, not through the windows' geometry.
- Real rigs use other angles: polygonal tanks (hexagon 60°, octagon 45°) and water prisms set square to an angled
  camera.

**The failed cases.**
- **Where they are:** all 4 wrong answers left in W5 have a window seen only by one or two long-lens cameras
  (13,500–37,000 px). R4 seed 3110 is the one wrong answer among 72 rigs with two or more cameras on every window.
- **Where seed 1001 fails** (`d3_single.py`): the window search for its lone 24,500 px camera chose a candidate 18° off.
  The true window scores better (2.0 against 5.0), but the score is not monotonic between them (18.8 at 4.6° from the
  truth), so the search's 9° grid missed the true window's narrow basin.
- **Where seed 1031 fails** (`d2_groups.py`): its window-1 pair (13,500 and 31,700 px), calibrated on its own, ends
  with the window 22° off.
- **What a lab knows:** each camera's angle to its window, from the mounts or by eye, to perhaps 5–10°.

## 2. The window-normal hint (H1's code change)

**Input.** `hints["normals"]`: one entry per camera, None or the normal of the window that camera looks through, in the
camera's own frame (x along the image rows, y down the image, z along the optical axis; the sign does not matter).
Example: a camera turned 30° to the left of square-on, about its vertical axis, sees the normal at
(sin 30°, 0, cos 30°). Not every camera needs one. In the rig file (`plate_wand_io`), `"hints": {"normals": [...]}`.

**Use: in the window search only (stage 3).** For each window with a hint from at least one of its cameras:
1. **Into the search's frame.** Each hint is turned into the frame of its camera's pinhole start. A pinhole camera
   looks along the chief ray in the water (`small_angle_camera`), so a hint at angle θ from the camera's axis becomes θ_w,
   with sin θ_w = (n_air / n_water) sin θ, at the same azimuth: the exact inverse of `small_angle_camera`. A window with
   several hinted cameras takes the mean.
2. **A second search about the hint.** The default search runs unchanged: a 50° cap about the cameras' viewing
   direction, normals 9° apart. Then the same search runs about the hinted normal: a 20° cap, normals 4° apart, with
   its own quick fits and full fits. 4° puts a candidate within about 2.3° of any normal in the cap, inside seed
   1001's narrow basin.
3. **Merged by score.** The hint search's candidates join the default's. A hint candidate within 1° and 5 % of the
   camera's distance of one already there is dropped, so a duplicate takes no start. A wrong hint only adds
   candidates that lose on score; the default's candidates are all kept.
4. **Every pass.** The hint search runs in every pass of the search: `bootstrap`'s two passes, `bootstrap_singles`'s
   three, the window groups' own calibrations and the cameras placed alone.

**Not used:** no prior in the final fit. The hints are too rough for one; the data decide the answer.

**Reported:** per hinted camera, the angle between its hint and the answer's normal, in `info["hints"]` and in
`calibration_report.txt`. A large angle points to a wrong hint or a wrong answer.

**Without hints** the code path is unchanged, and so are the results.

**Cost:** the hint search costs about as much as the default search of its window, so the window search takes up to
twice as long on hinted windows.

## 3. Tests

### A1: two windows at other angles (`a1_window_angles.py`)

- **The rigs:**
  - window 1's normal +Z; window 2's normal turned about Y by α = 30°, 45°, 60°, 90°, 120°, 150° and 180°;
  - both windows 12.7 mm acrylic, the 100 mm wand cloud's centre 150 mm behind each (its corners at least 79 mm);
  - two cameras behind each window, on either side of its normal, 600 mm along the chief ray, 1280 × 800;
  - a 25 mm wand and 1,000 frames.
- **(i) What the data determine,** Cramér–Rao at 0.1 px, as W2: per α, σ(f), the windows' normals, the cameras'
  distances, the 3-D shape of unseen points. The fixed rig per α has its cameras at 20° incidence (window 1's
  horizontally, window 2's vertically, as rig B) and f filling 80 % of the sensor.
- **(ii) From scratch:** per α, 4 rigs with each camera's incidence (15–25°), azimuth (the window's two cameras about
  180° apart, ±30°), distance (400–900 mm), aim, roll and f (fill 50–90 %) drawn at random, at 0.1 and 1 px. That is
  56 calibrations, each against the fit from the truth (W5's test).
- **Criteria:**
  - 0.1 px: every case at the same minimum;
  - 1 px: every case at least equivalent;
  - (i) is reported, not judged.

  A miss is diagnosed, not fixed, in this round.
- About 10 min.

### H1: window-normal hints on the failed cases (`h1_hints.py`, `h1b_hints_regression.py`)

- **H1a, the failed cases:** W5 seeds 1001, 1031, 1034 and 1045 (0.1 px) and R4 seed 3110 (1 px).
  - Each is run without hints and with every camera hinted, the hints off by δ = 0°, 5°, 10°, 20° and 30° in random
    directions (independent per camera; three draws per δ above 0). That is 70 calibrations.
  - **Criteria:**
    - δ up to 10°: every case at least equivalent to the fit from the truth, and the four 0.1 px cases at the same
      minimum;
    - δ of 20° and 30°: no case worse than without hints;
    - without hints: the same answers as before (the code path is unchanged).
- **H1b, nothing else changes:** W5 (60 rigs) and R4 (30 rigs at 0.5 and 1 px) again, with every camera hinted, each
  hint off by a random 0–10°.
  - **Criteria:** no rig worse than without hints (a "same" stays the same, an "equivalent" stays at least equivalent);
    the median time increase reported.
- **pytest:** the hint's conversion against `small_angle_camera` (exact); the merge (the default's candidates kept,
  duplicates dropped); the existing checks still pass.
- About 35 min of computing.

**Stop points:** if H1a's criteria fail for δ up to 10°, I report the diagnosis before changing the design again. If
H1b finds a rig made worse, I stop and report.

## 4. Deliverables

- The code change (section 2), the scripts and their outputs.
- `2026-10-03-angles-and-hints-report.md`.
- README updated: the progress log, the rig advice, the open items.

## 5. Revision after H1a: hint-held starts (H2; approved, "Ok go ahead")

H1a failed (report, section 4).
- **What the diagnosis found** (D5, D6; report, section 5):
  - the hints arrive accurately (exact hints 0.4–6.5° off in the search's frame);
  - the final fit converges from a weak window 20° off when the rest of the start is right (144 of 144);
  - the hint is lost in between: the window searches' scores and the window groups' own fits override it.
- **The change** (`plate_wand.py`):
  1. **A held normal:** `Layout` can hold a window's normal while its offset stays free (`held_normals`).
  2. **A hint-held start:** with hints, the start is built a second time with every hinted window's normal held at its
     hint:
     - the window search tries the hinted normal at its three offsets, and every fit in it keeps the normal;
     - the window groups' own final fits and the coarse fit keep it too.
  3. **The final fit frees everything,** as for the default start.
  4. **The hinted answer replaces the default answer only when its capped cost is lower** (the restarts' rule). The
     restarts then run as now.
  5. **Removed:** the second search about the hint (section 2). It costs time and led 1031 and 1045 into worse minima.

  Kept from section 2: the hint's input and conversion, the report line, the fast tests. Without hints nothing
  changes.
- **How a rough hint is refined** (your question: "Will you refine the hints later? Because the hint cannot guarantee
  to be super accurate"):
  - the hint is used only to build the start. The final fit, the full bundle adjustment on all the data, frees every
    hinted normal and estimates it from the data, so the answer's normal is the data's, with its own σ, not the
    hint's;
  - the report gives each hint's angle from the answer's normal: how far the data moved it;
  - how far off a hint may be and still be refined is what H2a measures (hints off by 0, 5, 10, 20 and 30°). D6
    suggests at least 20° when the rest of the start is right;
  - a hint too rough to be refined costs time, not the answer: the default start still runs, and the hinted answer
    is used only when it fits the data better;
  - **if H2a shows hints 10–20° off are not refined reliably:** several held starts spread around the hint (the hint
    and, e.g., four directions 10° from it), each refined the same way, the best fit kept. That costs more time, and
    I would ask first.
- **Tests:**
  - **H2a:** H1a again (the five cases, hints off by 0–30°), with H1a's criteria;
  - **H1b:** as in section 3: W5 and R4 with every camera hinted 0–10° off, no rig worse;
  - **pytest:** a held normal stays fixed while its offset moves.
  - About 45 min of computing.
- **Stop point:** if H2a fails for hints up to 10° off, I stop and report. Those rigs would then need more data, not
  hints: a third camera behind the window, or a moderate lens.

## 6. Revision after H2a: compare after the restarts (H3; approved with the third start, section 7)

H2a failed (report, section 6a).
- **What worked:** exact hints fixed all five cases. Rough hints (5–30°) were corrected on windows with two cameras
  (22 of 24 runs).
- **What did not:** rough hints were rarely corrected on windows behind a single long-lens camera (5 of 36).
- **Three answers came out worse than without hints.** D7 found why: the hint-held answer was better when compared,
  but the restarts then ran only on it. On the default answer they would have done better (1034: 1.84 → 0.179 px).
- **The two versions fix different cases:** the first version's second search fixed 1034 at every error; the held
  start fixed 3110 and 1031. Either one: 46 of the 65 hinted runs.

**The change:**
1. **The restarts run on the default answer and on each hinted answer,** each from the restarts' own seed, so the
   default answer's restarts give exactly the answer without hints.
2. **Then the answers are compared,** on the capped cost with the default answer's weights; every weighting agreed in
   D7. The answer without hints is always a candidate, so a hint cannot make the result worse.
3. **Optional:** the first version's second search back, as a third start (default, second search, hint-held), each
   with its own final fits and restarts. With hints, a calibration then takes about 2–2.5× the time without them,
   against 1.0–1.8× now.

**Tests:**
- **H3a:** H2a's 65 hinted runs again. Criteria: δ up to 10° at least equivalent and the 0.1 px cases the same; none
  worse at any δ.
- **H1b:** as planned.
- **pytest.**
- About 1 h of computing.

**Stop point:** if a hint still makes an answer worse, I stop and report.

## 7. H3 as approved, and the rounds after it

Your decision: "go ahead to try to fix it as much as possible. in the end, report when the calibration cannot work so
that i can pay attention to in the real experiments". So H3 is built with the optional third start. If H3a leaves
misses, each further fix is written here before it is built, tested the same way, and the last design then goes
through H1b and pytest. The round ends with a report of when the calibration cannot work.

### 7.1 The change (H3, `plate_wand.py`)

1. **Three start paths with hints.** `calibrate` builds:
   - **default:** the start without the hints, exactly as without hints;
   - **held:** H2's hint-held start;
   - **searched:** the first version's start (section 2): every hinted window searched a second time about its hint
     (20° cap, normals 4° apart), the candidates merged with the default search's by score.

   The searched path comes back as an option of the bootstraps (`widen=True`), so the default path does not change.
2. **Each path gets its own restarts.** Each path's stage-4 answer goes through the restarts, each from the restarts'
   seed. So the default path's answer is exactly the answer without hints.
3. **No repeated work.** A path whose stage-4 answer is the same minimum as an answer already found skips its
   restarts: they would repeat that answer's. "The same minimum": the two answers' residuals, on the observations
   both keep, differ by less than χ² = 9 (`answers_apart`). That is about the squared Mahalanobis distance between
   them, so every parameter is within 3σ.
4. **The comparison, after the restarts.** The answer is the default one unless a hinted path's answer:
   - is a different minimum (χ² ≥ 9 apart), and
   - has a lower capped cost, by more than 1e-6 relative, with the default answer's weights and scale.

   Of several such answers, the lowest wins. So a hint cannot make the answer worse on the capped cost. It changes the
   answer only when it finds another minimum.
5. **Reported:** per path, its stage-4 and final RMS, its capped cost against the default's, and which path gave the
   answer: `info["hints"]["paths"]` and `["chosen"]`, plus a line in `calibration_report.txt`.

### 7.2 Tests

- **H3a** (`h3a_hint_paths.py`): H2a's 70 runs (the five cases, without hints and with hints 0–30° off).
  - Criteria:
    - without hints, the stored answers;
    - no hinted run worse than without hints, at any δ;
    - δ up to 10°: at least equivalent, and the 0.1 px cases the same.

    H2a's verdicts are reported beside them.
- **pytest:** fast checks of `_merge_hinted`, `answers_apart`, and the paths' bookkeeping; the slow hinted test
  updated.
- About 25 min on 32 processes.

### 7.3 After H3a

- **If misses remain** (expected: windows behind one long-lens camera with rough hints, section 6):
  1. a diagnosis (D8) of the remaining wrong minima;
  2. one more fix (H4), written here first;
  3. its test.
- **D8** (`d8_profile.py`, a diagnosis): the cost over a weak window's normal, from the answer without hints, on 1001,
  1034 and 1045 (windows behind one long-lens camera). The weak window's normal is held at each point while
  everything else is fitted:
  - along the great circle to the true normal, 1° apart;
  - on a grid about the answer's normal, 30° wide, 3° apart.

  The three cheapest grid points are then released. It asks whether a scan of that normal, without any hint, would
  find the global minimum.
  - **H3a's outcome adds a question:** 1045's remaining misses lie 2–6° from the true normals. So D8 also maps a fine
    grid about the true normal (12° wide, 1.5° apart), to see whether the cost has minima that close to the truth.
  - **What D8 decides:**
    - **if that normal's cost, with everything else fitted, falls steadily towards the truth:** H4 is a descent of
      the weak window's normal from the answer, a few tens of held fits, then the release;
    - **if it has several minima:** H4 is a grid scan of that normal, a few hundred held fits, then the release of
      its best minima. That costs more time, and only on rigs with a weak window (warning 1 of 7.4).
  - **Either way,** the scan's answer is accepted only on H3's terms: another minimum with a lower capped cost. So it
    cannot make an answer worse.
- **Then H1b** on the final design (W5 and R4, every camera hinted 0–10° off; no rig worse).
- **Then the final report:** when the calibration cannot work, from every campaign so far:
  - rig geometry, lenses, cameras per window, noise, wand length, hints;
  - for each condition, what goes wrong, how to see it in `calibration_report.txt`, and what to do in the
    experiment.

### 7.3a D8's result, and the scan of a weak window's normal (H4)

**D8** (`d8_profile_out.txt`). The weak window's normal was held along the path to the true normal and on grids:
- **1001** (window 2, one camera of 24,500 px; the answer 11.2° off):
  - along the path the cost first rises, by 49 % at 5–6°, then falls into a narrow valley at the truth;
  - the grid about the answer (30° wide, 3° apart) finds that valley (a grid point 0.6° from the truth);
  - its three best points, released, all reach the global minimum (equivalent, RMS 0.085945–0.086069).
  - **A scan fixes 1001 without any hint.**
- **1034** (window 3, one camera of 23,300 px; the answer 26.6° off):
  - the fits held at wrong normals get pinned against a window: 228 of the 317 grid points;
  - continuing from them carries the pinned state along, so the grid near the truth is no better;
  - the releases return to the answer.
- **1045** (two windows, one camera each, both wrong: 48.8° and 14.0°):
  - scanning one window while the other is far off cannot reach the minimum;
  - near the truth the cost has minima 1.5–3° apart: released 1.5° from the true normal of window 2, the fit ends in
    another minimum (RMS 0.0871 against 0.0862).
- **From the truth** every grid is one valley 3–4° wide, with no other minimum within 10°.

**The change** (`plate_wand.py`; after the start paths and their comparison, on a rig with several windows):
1. **Which windows:** every weak window, with warning 1's rule. Its one or two cameras all see the wand under less
   than 6°.
2. **The grid:** its normal on a grid about the answer's, 30° wide and 3° apart. At each point the normal is held, and
   one bundle adjustment fits everything else, on the answer's active observations with its weights. Each point
   starts from a neighbour already fitted, outwards from the answer.
3. **The release:** the three cheapest grid points with a cost below the answer's are fitted with everything free, as
   a start is (stage 4 on fresh observations).
4. **Accepted on H3's terms:** a released answer must be another minimum (`answers_apart` ≥ 9) with a lower capped
   cost than the answer.
   - The best such one gets the restarts, and is compared again.
   - So the scan cannot make an answer worse.
5. **Reported** in `info["scan"]` per window: the grid's size and time, the released fits' RMS, and whether one was
   accepted.

**Cost:**
- D8's held fits took 0.6–4 s each, so about 3–20 min per weak window.
- It runs only on rigs with a weak window: 22 of the 120 W5 and R4 cases.

**Expected:**
- fixes 1001-type misses, with or without hints;
- not 1034's or 1045's without hints;
- maybe some of 1045's misses with hints, which lie 2–6° from the truth.

**Test (H4a):** H3a's 70 runs again, with H3a's criteria. Reported besides: what the scan changed, and its time.

### 7.3b H4a's result, and two refinements (H4b)

**H4a** (`h4a_scan_out.txt`): FAIL.
- **Without hints, the scan fixes 2 of the 5 cases:** R4 seed 3110 (now the same minimum) and W5 seed 1001
  (equivalent). 1031, 1034 and 1045 are unchanged.
- **With hints:** 55 of 65 runs are at least equivalent (H3a 48).
- **3 runs come out "worse than without hints",** because the answers without hints improved:
  - two on 3110 are cosmetic. The hinted answers have a lower RMS than the truth-started fit and |z| = 0.02; they
    count as "equivalent" only because one observation more or fewer is kept;
  - one is real: 1001 with hints 20° off.
    - The scan released a fit at RMS 0.085950, essentially the global minimum, with a capped cost 0.39 % lower.
    - It was refused, because it was only χ² = 4.9 from the answer: "the same minimum" by the rule of 7.1.
    - That window is so weakly determined (σ 2.1°), and the cost so far from quadratic, that two different minima
      lie that close.
- **Time:** the scans took a median of 59 min on the 6-camera rig (3110), with 32 processes at once.

**The refinements** (`plate_wand.py`):
1. **The acceptance rule:** a lower capped cost (by more than 1e-6 relative), and either another minimum
   (`answers_apart` ≥ 9) or exactly the same observations kept.
   - With the same observations, a lower capped cost is simply a better fit of the same data.
   - The swap the "another minimum" test guards against differs only in which observations are kept.
   - This rule applies to the paths' comparison and to the scan.
   - A hinted path whose stage-4 answer repeats a minimum already found skips its restarts, as before, but stays a
     candidate.
2. **The scan's held fits on at most 300 frames,** evenly spread over the answer's active ones (`SCAN_FRAMES`). The
   release still fits every frame.

**Test (H4b):** H4a's 70 runs again, with H4a's criteria. A run that is "worse" only because one observation more or
fewer is kept, at a lower RMS than the fit from the truth with |z| below 0.1, is reported as such.

### 7.4 Which rigs fail, and warnings in the report (H5)

**The study** (`p_weak_windows.py`; no calibration run, done 3 Oct):
- It covers the 120 stored cases of W5 and R4, without hints, on the current code.
- Every wrong answer (5 of 120) had a window seen by only one or two cameras that all see the wand cloud under
  6° or less (long lenses, or far cameras).
- 5 of the 22 cases with such a window ended wrong; none of the other 98 did.

**The change** (`plate_wand_io.py`, the report only; the calibration is unchanged): a list `warnings` in
`calibration_result.json` and a WARNINGS block at the top of `calibration_report.txt`. Each warning names the window
or camera, the evidence from the tests, and what to do:
1. **A weak window:** seen by one or two cameras that all see its wand ends under less than 6°. The angle is measured
   from the detections' spread in the image and f.
2. **A pinned window:** an active wand end within 1e-6 mm of the window's water-side surface. 8 of the 14
   multi-window misses were pinned, none of the 69 right answers.
3. **A hint far from the answer:** a hinted normal more than 15° from the answer's. Either the hint is off by that
   much or the answer is wrong.
4. **Cameras looking the same way:** every camera's line of sight within 10° of the others (rig C: precision 3–9×
   worse).
5. **High noise on a multi-window rig:** σ above 0.5 px. The remaining wrong answers at 1 px came there (R4 3110).
6. **Few frames:** fewer than 300 active frames (the tests used 500–1,500).
7. **The wand length fitted:** not identifiable through flat windows (W8).

**The thresholds' evidence (H5a):**
- W5 and R4 rerun without hints on the final code. This is also the final check that the answers without hints are
  unchanged.
- Counted: the warnings on wrong and on right answers.
- A fast pytest checks each warning on a constructed case.
