# Flat-window wand calibration (PINPLATE): summary and progress

Last updated: 2026-10-04.

Start here to catch up. This file gathers what the flat-window wand tests concluded, and what that means for a real
experiment.
- It is kept up to date: each new result on this calibration adds a dated entry to the progress log (section 1,
  newest first) and revises the sections it changes.
- The numbers come from the reports listed in section 7, which name the scripts and data behind them.
- The figures page draws them: `figures/plate_wand_results.html`, also at
  https://claude.ai/artifact/ALBPbfKJKbMddmNBp7m2iV.

**In short** (as of 3 Oct).
- **Ready for a first real data set** on the rigs a flat-window experiment normally uses:
  - one window, with any number of cameras;
  - several windows, each seen by two or more cameras.

  Not reliable when a window is seen only by one or two cameras with narrow views (each sees the wand under less
  than about 6°: long lenses or far cameras), above all with one camera per window.
- **When it cannot work, and what to watch:** `2026-10-03-when-calibration-fails.md`, written at your request ("in
  the end, report when the calibration cannot work so that i can pay attention to in the real experiments").
  - As a page with two figures: `2026-10-03-when-calibration-fails.html`, also at
    https://claude.ai/artifact/Ue22gHGbyjWDNn2GnRBAyf.
- **The evidence, on synthetic data:**
  - **one window:** the right cameras and window from scratch, with no camera input, focal lengths included: 37 of 37
    random rigs;
  - **several windows, each with two or more cameras:** 68 of 72 cases at the global minimum, 71 equivalent (W5 and
    R4);
  - **a window behind a single camera:** fine with moderate lenses (5 of 5 W5 rigs whose lenses were all under
    11,000 px reached at least an equivalent answer);
  - **the weak case:** every wrong answer in W5 and R4 had a window seen only by one or two narrow-view cameras. The
    scan of such a window's normal (3 Oct) fixed 2 of the 5; 3 of 120 cases remain wrong without hints (W5 1031,
    1034, 1045);
  - **precision:** as precise as the data allow (errors 0.97–0.99 of the Cramér–Rao bound), with honest error bars.
- **New (3 Oct): two windows at any angle work** (A1). From 30° to 180° apart, 56 of 56 calibrations from scratch
  reached the global minimum; the angle changes only the precision (best through perpendicular walls).
- **New (3 Oct): window-normal hints, a scan of weak windows, and warnings** (`2026-10-03-angles-and-hints-report.md`).
  - **Hints** (`hints={"normals": [...]}`): the calibration runs three start paths and keeps a hinted answer only when
    it fits the data better.
    - On the five hard cases, no hinted run was worse than without hints (0 of 65).
    - Within 20° of the truth, 45 of 50 runs are right, every miss on seed 1045 (one camera per window, long lenses).
  - **The scan:** a weakly seen window's normal is scanned by the calibration itself. Without hints, it fixed R4 3110
    and W5 1001 and changed no other answer (H5a: 118 of 120 unchanged, 2 better).
  - **Warnings:** `calibration_report.txt` opens with WARNINGS. At 0.1 px, every wrong answer in W5 and R4 had one
    (3 of 3), and only 3 of the 47 right answers did.
- **New (3 Oct): restarts perturbed about the answer,** on rigs with several windows.
  - They free an answer pinned against a window: a wand end exactly on the window's water-side surface. That caused 8
    of the 14 multi-window misses.
  - **W5 at 1 px:** 7 of 10 at the same minimum (5 before), 10 of 10 equivalent.
  - **R4:** 59 of 60 at least equivalent (55 without); its strict target, half the misses fixed, was missed with 2 of 5.
  - Nothing got worse, for a median 9–14 % more time.
- Against the GUI's current calibration on the same data:
  - 63× faster (median 38 s against 29 min);
  - no failures, where the GUI failed twice;
  - a 3-D error of 1.5 µm against 59 µm on random rigs (medians), although only the GUI was given the true focal
    lengths.
- Two differences from the tube:
  - **the wand length is the only scale.** Fitting it gives no check of the constants, so it must be measured
    independently;
  - **the window constants hardly matter.** A plate 10 % too thick or a water index 0.002 off leave the answer almost
    unchanged.
- **New (3 Oct): the first real data set,** case `cases/plate_20260727_T0`. Four cameras look through two
  perpendicular tank walls, two cameras per wall (report `cases/plate_20260727_T0/2026-10-03-report.md`).
  - **The calibration works:** 0.215 px median reprojection error and 0.0067 mm median wand error on every frame. That
    is 6× and 8× better than the GUI's calibration of the same data, which held f at the nominal value.
  - **Lens distortion (k1) must be fitted.**
  - **From scratch it can end in a wrong minimum** (2 of 9 frame sets). The cause is one wall seen by its two cameras
    under only about 6°. Starting from a verified fit avoids it, and so do rough hints.
  - **Two library bugs** were found and fixed.
  - **New (4 Oct): the channel.** The user's channel (15 × 10 cm) showed one wall about 9 cm off.
    - **New options, off by default:** fitted lens terms (`fit_lens`) and held facts about the rig (`constraints`: the
      channel, the cameras' aim, a camera's distance to its wall).
    - **The fix:** with the principal point fitted and the channel held, the wall is in place, and the fit is better
      on held-out frames too (progress log, 2026-10-04).
- The GUI does not call this calibration yet. Section 4 says how to run and check a real experiment; the full report
  has the same advice (`2026-10-02-plate-wand-report.md`, section 6).

## 1. Progress log

**2026-10-04**
- **An illustrated guide to both wand calibrations, flat walls and tube:**
  `modules/camera_calibration/curved_calibration/guide/WAND_CALIBRATION_GUIDE.html`, also at
  https://claude.ai/artifact/BR616e7TfeF63xsX4rrA46 (private until you share it).
  - **What it shows:**
    - a flow chart of the stages;
    - both rigs in 3-D;
    - what a wall and a tube do to an image;
    - each stage with T0's real numbers (flat walls) and a synthetic rig (tube).
  - **How it is made:** `make_guide.py` beside it computes every number and drawn ray with the calibration's own
    models. It also checks that its replay of `cylinder_wand.bootstrap`'s placement search matches the library.
- **Case `plate_20260727_T0`, your channel** (plan and report `cases/plate_20260727_T0/2026-10-03-walls-*.md`).
  - **The problem:** the channel is 15 × 10 cm inside, with C1 and C2 on the long side. The delivered calibration put
    the C3/C4 wall about 9 cm too close to its cameras, so the wand reached 225 mm into the 150 mm channel.
  - **The causes, found by tests W0–W2:**
    - the fit was pinned against the C1/C2 wall: a wand end sat on its surface, so every step was refused;
    - each lens's principal point was held at the image centre.
    - Freed of both, the data alone move the wall about 70 mm towards the channel.
  - **Two library options, approved and off by default, so earlier results are unchanged; committed by you in
    `e2f3934`:**
    - **L1, `fit_lens`:** fits the principal point, k2 or the tangential terms.
    - **L2, `constraints`:**
      - **channel:** every wand end stays inside it, each ball its radius from every wall; held by a squared hinge,
        starts moved into it;
      - **aim:** where cameras' optical axes meet their wall;
      - **distance:** a camera's distance to its wall.
      - `constraint_report` lists them at the answer.
    - Three new tests in `tests/test_plate_wand.py`; the file passes, 21 tests with the slow ones.
  - **Your choice:** the principal point held at the image centre, `Results\T0\calib_plate_wand_ref_centre` (W4c).
    - Accuracy: 0.218 px and 0.0068 mm median on every frame, as the first calibration's.
    - The wand: 14–99 mm from the C1/C2 wall, 16–132 mm from the C3/C4 wall.
  - **The variant with the principal point fitted, `Results\T0\calib_plate_wand_ref`** (W4):

    | measure | the first calibration | with the channel |
    |---|---|---|
    | reprojection error, median, every frame | 0.215 px | 0.210 px |
    | wand length error, median | 0.0067 mm | 0.0065 mm |
    | the wand from the C3/C4 wall | 109–225 mm | 16–131 mm |

    - **Settings:** k1 and the principal point fitted; the channel and the cameras' aim held.
    - **Held-out halves:** better too.
    - **A cross-check:** C3 and C4 aim at the middle of their wall unprompted.
  - **What the real data showed:**
    - **"pinned" is not harmless for the walls.** It hides how freely they move.
    - **The walls' positions** rest on effects of a few tenths of a pixel. A lens term held wrongly can move one by
      9 cm.
    - **Facts about the rig should be constraints.** A channel's size or a wall the wand touched works where a hint
      cannot, because a hint never overrides the data.
    - **The channel also guards against wrong minima.** From scratch on every frame with the channel held, both lens
      models reached the right answer in 55–70 min on one thread. Without the channel, the same data had ended in
      wrong minimum kind A.

**2026-10-03**
- **The first real data set: case `cases/plate_20260727_T0`, run 20260727/T0** (plan and report in the case folder).
  - **The rig:** four cameras with 180 mm lenses, two per wall, behind two perpendicular 0.5 in acrylic walls.
  - **The data:** your circle finder's detections, 10,936 frames.
  - **Two library bugs, fixed with your approval, each with a regression test in `tests/test_plate_wand.py`:**
    - `cylinder_wand.affine_factorization` crashed when its last rejection round still rejected frames. Real gross
      detection errors cause that. It now returns the frames it factorised.
    - `plate_model.to_cpp` refused a rotation that the C++ side rounds 1.2e-14 off. Its check is now 1e-12.
    - All suites pass after both fixes: plate 14 + 4, cylinder 6 + 1.
  - **The result** (`Results\T0\calib_plate_wand` on X:):

    | measure | this calibration | GUI's calib_3 |
    |---|---|---|
    | reprojection error, median | 0.215 px | 1.28 px |
    | wand length error, median | 0.0067 mm | 0.052 mm |
    | wand length SD | 0.010 mm | |
    | walls angle | 90.17° | |
    | f | 18,750–21,940 px | 13,333 px (held) |

    Holding f at 13,333 px put calib_3's cameras about 30 % too close.
  - **What the real data showed:**
    - **k1 is needed:** about 6 px of distortion at the image corners.
    - **Wrong minima:** from scratch, the calibration ended in one on 2 of 9 frame sets, including every frame. The
      wall seen by C3 and C4 under only 5.3–6.0° is the weak window. The "noisy" warning caught 2 of 4 wrong answers.
      A start from a verified fit, or rough hints, avoided them.
    - **"pinned":** caused by a few frames where the wand touched a wall while its ball was cut by another camera's
      image border. It costs about 0.5 % in f and nothing in the measurement.
    - **The walls' positions** are uncertain by a few cm. The measurement inside the wand's volume is not affected.
    - **No drift** between the halves of the recording.
- **The scan of a weak window's normal (H4, H4b), warnings (H5), and the final checks (H5a, H1b): all criteria met
  except H4b's strict one, on seed 1045** (report sections 6d–6g; plan sections 7.3a, 7.3b and 7.4).
  - **D8** (`d8_profile.py`) held the weak window's normal at points between the answer and the truth, and on grids,
    fitting everything else. On 1001, 1034 and 1045 it found:
    - **the answers are separate minima:** the cost rises before it falls into the truth's valley, which is only a few
      degrees wide;
    - **a grid scan of the normal (30° wide, 3° apart) finds that valley when the rest of the answer is sound**
      (1001), not when the held fits get pinned against a window (1034), nor when two windows are wrong at once (1045).
  - **H4** (`plate_wand.py`): on rigs with several windows, `calibrate` scans the normal of every weak window. A
    released fit is kept only on H3's terms, so the scan cannot make an answer worse.
  - **Smoke test** (`h4_smoke.py`): **W5 seed 1001 is now right without hints.** 1034 and 1045 are unchanged. The scans
    took 3–9 min.
  - **H4a** (H3a's 70 runs):
    - without hints, 3110 and 1001 are now right;
    - with hints, 55 of 65 runs are at least equivalent;
    - but the "another minimum" test refused one real improvement (1001, χ² 4.9 apart on a window with σ 2.1°);
    - the scans took up to an hour on a 6-camera rig.
  - **Refined (H4b):**
    - a fit also replaces the answer when its capped cost is lower and it keeps exactly the same observations;
    - the scan's held fits use at most 300 frames.
  - **H4b** (`h4b_scan.py`, the final code):
    - without hints, 2 of the 5 failed cases are now right (3110, 1001), none worse;
    - with hints, **56 of 65** runs are at least equivalent and **none is worse**. 45 of 50 are right within 20°,
      every miss on 1045;
    - scans 4–6× faster: a median of 2–11 min per case under full load;
    - strict criterion **FAIL**, on 1045 alone. It has two cameras in all, one per window, with long lenses, and it
      stays wrong without hints and with most hints 10° or more off.
  - **Left wrong without hints:** 1031, 1034 and 1045. Each has two weak windows wrong at once, or held fits pinned
    against a window.
  - **H5a** (W5 and R4 rerun without hints, the final code): **PASS.**
    - 118 of 120 answers unchanged, the other 2 better (1001, 3110), none worse;
    - 3 of 120 still wrong (W5 1031, 1034, 1045);
    - at 0.1 px every wrong answer had a warning, and 3 of 47 right answers did.
  - **H1b** (W5 and R4 with every camera hinted 0–10° off, against H5a's answers): **PASS.**
    - none of 120 worse;
    - 1031, 1034 and 1045 right with their hints;
    - a median 2.55× the time without hints.
  - **The report for real experiments:** `2026-10-03-when-calibration-fails.md`.
- **Three start paths with hints (H3): built on your approval, tested; no hint makes an answer worse any more**
  (report `2026-10-03-angles-and-hints-report.md`, section 6c; plan section 7).
  - **Your decision:** "go ahead to try to fix it as much as possible. in the end, report when the calibration cannot
    work so that i can pay attention to in the real experiments".
  - **The change** (`plate_wand.py`):
    - with hints, `calibrate` runs three start paths: the default one (hints ignored), the hint-held one (H2), and the
      hint-searched one (the first version, back as `widen`);
    - each path's answer gets its own restarts; a path that reaches a minimum already found stops there;
    - the answer is the default one unless a hinted answer is another minimum with a lower capped cost.
  - **H3a** (the five failed cases, the same 65 hinted runs):
    - **no run worse than without hints** (H2a: 3);
    - 48 of 65 at least equivalent (H2a 31, H1a 22);
    - every run right on two-camera windows with hints up to 20° off (3110, 1031), and all 13 on 1034;
    - still wrong on windows behind a single long-lens camera: 1045 three times and 1001 once with hints up to 10° off.
      Their answers lie 2–6° from the truth, so the cost seems to have other minima nearby. A diagnosis (D8) maps
      them;
    - **time with hints:** 1.8–3.0× the time without.
  - **Strict criterion: FAIL**, on those single long-lens windows.
  - **Tests:** pytest 11 fast checks pass, with new ones for the merge, `answers_apart` and the report's warnings
    (next item).
  - **Warnings in the report (H5, plan section 7.4):**
    - `calibration_report.txt` now opens with a WARNINGS block naming what the tests associate with a wrong or weak
      answer;
    - **the evidence for the first warning:** all 5 wrong answers in W5 and R4 had a window seen only by one or two
      cameras with narrow views (5 of 22 such cases; none of 98 others; `p_weak_windows.py`);
    - checked on W5 and R4 in H5a (previous item).
- **Hint-held starts (H2): built on your approval ("Ok go ahead"), tested, FAIL; a fix (H3) proposed, approved since**
  (report `2026-10-03-angles-and-hints-report.md`, sections 6a and 6b; plan section 6).
  - **Your question first:** "Will you refine the hints later? Because the hint cannot guarantee to be super
    accurate". Yes: the hint only builds the start, and the final fit estimates the normal from the data.
  - **The change** (`plate_wand.py`):
    - a window's normal can be held while its offset is fitted (`Layout(held_normals=...)`);
    - with hints, the start is built a second time with every hinted normal held, through the window search, the
      window groups' own fits and the coarse fit;
    - the final fit frees everything, and the hint-held answer replaces the default one when its capped cost is lower;
    - the first version's second search is gone.
  - **H2a** (the five failed cases, the same hints as H1a):
    - exact hints fix all five (first version: two);
    - rough hints (5–30°) are corrected on windows with two cameras (3110, 1031: 22 of 24 runs), rarely on windows
      behind a single long-lens camera (1001, 1045, 1034: 5 of 36);
    - 3 runs came out worse than without hints;
    - time with hints 1.0–1.8× (medians).
  - **Why worse (D7, `d7_held_acceptance.py`):** the comparison came too early.
    - The hint-held answer was better when compared, under every weighting.
    - The restarts then ran only on it. On the default answer they would have done better: 1034 from 1.84 to 0.179 px,
      against 0.453 px for the hint-held answer.
  - **The two versions fix different cases:** the first fixed 1034 at every error, the second 3110 and 1031. Either one
    fixes 46 of the 65 hinted runs.
  - **Proposed (H3):**
    - each answer gets its own restarts before they are compared, so a hint cannot make things worse;
    - optionally, the first version's search back as a third start, at about 2–2.5× the time without hints.
  - **Tests:** `tests/test_plate_wand.py` 12 passed: 9 fast, and 3 slow including a new one (rig B, hints 10° off,
    the global minimum).
  - **Limit, on this evidence:** a window behind a single long-lens camera needs a hint within a few degrees, or
    better, a second camera.
- **Windows at other angles, and window-normal hints: A1 passed, H1a failed; a revision proposed, approved since**
  (plan `2026-10-03-angles-and-hints-plan.md`, report `2026-10-03-angles-and-hints-report.md`).
  - **Your questions and decision:** "How to avoid the failed cases? Have you tested for two windows with different
    angles?". Then: hints on the window normal first, "not guaranteed to be super accurate", and no focal-length hints,
    which are hard to guess in a real experiment.
  - **A1, two windows 30°–180° apart: PASS.** Every multi-window rig before used box-tank walls, 90° or 180° apart.
    - 56 of 56 calibrations from scratch reached the global minimum, at 0.1 and 1 px. Each rig had two moderate-lens
      cameras per window.
    - Cramér–Rao: the precision follows the angles between the lines of sight. The 3-D shape error is 2.6 µm through
      perpendicular walls, 1.5× that at 30° and 150°, and 2.2× at 180°, where the cameras face each other.
  - **The hint, as built** (`plate_wand.py`; tested in `test_window_normal_hints`):
    - `hints={"normals": [...]}`: per camera, its window's normal in the camera's own frame (e.g.
      (sin 30°, 0, cos 30°) for a camera turned 30° to the left of square-on);
    - turned into the search's frame exactly (the inverse of `small_angle_camera`);
    - a second window search about it (a 20° cap, 4° apart), merged with the default search by score;
    - the report gives each hint's angle from the answer.
  - **H1a, the five failed cases with hints 0–30° off: FAIL.**
    - Fixed: seed 1034 at every hint error, and 1045 with exact hints or some rough ones.
    - Not fixed, even with exact hints: 1001, 1031 and 3110.
    - Some rough hints led 1031 and 1045 into other wrong minima with a higher RMS (1045: 0.380 against 0.349 px).
    - Without hints, every answer is identical to before.
  - **Why (D5, D6, diagnoses):**
    - **the hints arrive accurately:** 0.4–6.5° from the truth in the search's frame;
    - **the final fit would converge from them:** from the truth with one weak window tilted up to 20°, 144 of 144 runs
      reached the global minimum;
    - **the hint is lost in between.** The window searches score candidates against cameras that are still only roughly
      placed, and a window group calibrated on its own lets a weakly seen window drift (3110: from 0.7° to 20°).
  - **Proposed (H2, report section 6): hold the hinted normals fixed while the start is built.**
    - The final fit frees them.
    - The hinted answer replaces the default one only when its capped cost is lower.
    - The second search is dropped.
  - **H1b (W5 and R4 with hints) was not run.** It waits for the design you approve.
  - **The code is uncommitted** (the restarts too). Without hints it behaves as before.
- **The conclusion and the real-experiment advice brought up to date** (your question):
  - "In short" above, and section 4;
  - the full report `2026-10-02-plate-wand-report.md`: a new summary, the W5 section with the restarts, the
    limitations, and a new section 6 on running a real experiment.
  - The evidence for the advice:
    - rigs with two or more cameras on every window reach the global minimum in 68 of 72 cases;
    - every wrong answer left in W5 has a window seen only by one or two long-lens cameras (13,500–37,000 px);
    - the multi-window failures cluster at 1 px.
- **How a restart is accepted: changed and compared (R7), on your approval** ("ok, go ahead and compare"; report,
  sections 7 and 10).
  - **The rule:** a restart's refit now replaces the answer when its cost on the same terms is lower. Every usable
    observation counts, its squared residual capped at the 5σ rejection limit (`capped_cost`). `_choose` still picks
    among the starts.
  - **Compared on the 89 cases with restarts:** no restarts, the first version (`_choose`), and the capped rule.
    - The capped rule changed exactly 2 answers:
      - **R4 seed 3111 at 1 px:** from a different answer to an equivalent one (1.231995 px, the truth-started fit
        1.231979);
      - **W5 seed 1031:** from 5.29 to 3.59 px, still wrong.
    - Nothing is worse than without restarts.
    - **Equivalent or better:** R4 55 → 58 → 59 of 60; W5's multi-window rigs 18 → 19 → 19 of 23.
  - **The tests again:**
    - **R3:** PASS, all four criteria; the restarts cost a median 9 %.
    - **R4:** no case worse; still 2 of 5 misses fixed strictly, so FAIL, but 4 of 5 at least equivalent.
    - **R5:** PASS, identical.
    - **R6:** `tests/test_plate_wand.py` 10 passed, with a new `test_capped_cost`.
  - **Cost:** on R4's largest rigs the second round now also runs: at most 194 s, 109 % of the rest of the run. The
    median is still 14 %.
- **Restarts perturbed about the answer: built and tested (R1–R6); one criterion failed.** The full report is
  `2026-10-03-perturbed-starts-report.md`.
  - **Why seed 2000 missed (R1, R1b): its answer was pinned against window 1.**
    - One wand end lay exactly on the window's water-side surface (3.5e-12 mm deep; at the global minimum the nearest
      is 19.7 mm deep).
    - The final fit rejects any step that takes a wand end out of the water, so it stopped there.
    - This, not a narrow basin, is why `d4` saw restarts from a quarter of the way succeed.
    - Every draw that moved the window back off the cloud reached the global minimum (27 of 27 at 1–3σ). The draws
      that pushed it further in left wand ends out of the water.
  - **Pinning marks misses.** Across W5's multi-window rigs and R4:
    - 8 of the 14 answers that missed the global minimum were pinned;
    - none of the 69 that reached it was.
  - **The change** (`plate_wand.py`, plus one line in the report):
    - on a rig that uses several windows, `calibrate` draws 4 perturbations of its answer from its covariance at 2σ;
    - a draw is drawn again when a camera leaves the air side or an observation stops projecting;
    - each draw is screened by one bundle adjustment on the answer's observations and weights;
    - the cheapest is refitted in full if it lowers the cost, and a second round follows if it wins;
    - one-window rigs are unchanged;
    - options: `perturb_starts=4`, `perturb_scale=2.0`, `perturb_all=False`.
  - **R2 (the draws):** PASS. Their spread is within 3.6 % of the target, camera 1 is held, they are deterministic,
    and every kept draw is valid.
  - **R3 (W5 rerun):** PASS on all four criteria.
    - Seed 2000 now reaches the global minimum.
    - At 1 px: 7 of 10 at the same minimum (5 before) and 10 of 10 equivalent (9 before). At 0.1 px: unchanged, 46 of
      50.
    - No rig got worse, and the 37 one-window rigs are identical.
    - The restarts cost a median 9 % (8 s, at most 30 s).
    - Seeds 1034, 2004 and 2005 improved, and 2008 went from equivalent to the same minimum.
  - **R4 (30 new rigs, two or more cameras on every window, 0.5 and 1 px, without and with the restarts): FAIL.**
    - Misses: 5 before, 3 after. Fixed: 3029 and 3100 at 1 px, both pinned before. Equivalent now: 3111 at 0.5 px.
      Nothing got worse.
    - The target was at least half of the misses fixed.
    - **On 3111 at 1 px the fix was found but blocked:**
      - a restart reached the global minimum's RMS;
      - `calibrate`'s selection rule kept the pinned answer, because it keeps 2 more observations (R4b);
      - the 2 that the restart dropped have wand ends just outside the water at its windows.
    - 3110 is an ordinary wrong minimum, not pinned.
  - **R5 (W7 rerun):** PASS. All 12 answers are identical; rig B's restarts found nothing better, in 3–4 s.
  - **R6:** PASS.
    - `tests/test_plate_wand.py`: 9 passed, including 2 new tests.
    - The full suite: 245 passed and 7 skipped, with only the known failures.
  - **Proposed** (report, section 7; approved and done, see the entry above): accept a restart when it is better on
    the same terms.
    - Every usable observation counts, its squared residual capped at the 5σ rejection limit; one that does not
      project counts the cap.
    - On 3111 this keeps the restart (−1.9 %).
  - **The figures page was rebuilt:**
    - each W5 rig's hover box shows its restarts;
    - Fig. 2's caption names the fixes.

**2026-10-02**
- **Plan written** (approved 3 Oct; next day's entry): perturbed restarts of the final fit on multi-window rigs
  (`2026-10-02-perturbed-starts-plan.md`, tests R1–R6), the remedy for seed 2000's kind of miss (next entry).
- **Why W5 seed 2000 misses although each window has two cameras** (`d4_seed2000.py`, your question).
  - **The rig is sound.** Each window's two cameras are 21° and 30° apart (in the water), and the walls are
    perpendicular. At 1 px noise the right answer is itself loose: σ(f) 0.5–1.2 %, distances 3–20 mm, normals 0.4°.
  - **The start lands in a small wrong basin.** Each window's pair is first calibrated on its own, then the two are
    joined. Alone, a pair is 1.2–2.5× less precise than the full rig, which sees the same wand through both walls.
    - Each pair landed within 0.2–2.0 of its own σ, as well as two cameras can do at 1 px.
    - Joined, that is 2–3.4 full-rig σ off in several parameters at once.
    - Stage 4 then settles in a nearby minimum: RMS +0.3 % (50 px² more in squared residuals), cameras 3 and 4's f
      off by +2.0 % and −2.1 % (3.8σ, 2.8σ).
  - **That basin is narrow.** Stage 4 restarted a quarter of the way (or more) from that answer towards the global
    minimum reaches the global minimum.
  - **The whole-rig start that solved it in W5 run 1 no longer does** (it was changed since, because it failed on 4 of
    9 multi-window rigs): 1.34 px and 18–20 px.
  - **It is the only such miss.** Every rig whose windows all have two or more cameras reached the global minimum
    at 0.1 px (8 of 8); at 1 px, 2 of 4 did, plus one equivalent answer.
  - A possible remedy, untested: a few stage-4 starts perturbed around the joined start (open item 1).
- **A camera looking straight through a window costs nothing; cameras close together do** (`p_mixed_views.py`, your
  question: one camera square-on and one at an angle through the same window).
  - Cramér–Rao at 0.1 px, 1,000 frames, on rigs built like rig A.
  - **With one square-on and one angled camera, the square-on camera is as precise as the angled one,** or slightly
    more: σ(f) 0.145 % against 0.162 %, its distance to the window 3.5 mm against 4.0 mm (0° + 20°).
    - The rig matches a pair at −10°/+10°, the same 20° apart: σ(f) 0.151 %, 3-D shape 13.6 µm against 13.5 µm.
    - All three mixed rigs calibrated from scratch reached the global minimum (11–15 s).
  - **Correction: rig C's weakness comes from its cameras being 5° apart, not from its view straight through the
    wall.** The same layouts moved to 18–25° incidence are as weak:

    | layout | 0–5° incidence: σ(f), σ(distance), 3-D shape | 18–25° incidence: σ(f), σ(distance), 3-D shape |
    |---|---|---|
    | two cameras 5° apart | 0.31 %, 10 mm, 47 µm | 0.35 %, 7 mm, 46 µm |
    | rig C, three cameras 5° apart | 0.20 %, 4.4 mm, 26 µm | 0.19 %, 3.7 mm, 25 µm |

  - Sections 3 and 4, the report (section 3.2) and the figures page said rig C was weak because it is square-on, and
    advised keeping cameras 15° or more off the normal. They are corrected: what matters is the angle between the
    cameras' lines of sight (section 3, item 6).
- **This summary rewritten as a catch-up report,** in the format of the cylinder's (`tests/cylinder_wand/README.md`).
- **Figures page:** `figures/plate_wand_results.html`, also at https://claude.ai/artifact/ALBPbfKJKbMddmNBp7m2iV.
  - `make_figures.py` rebuilds it from the saved results.
  - Checking its captions against the results corrected two statements of the report (its section 3.2):
    - the scene's position relative to camera 1: 0.43–0.48 mm on rigs A, B and D, 1.6 mm on C and 7.3 mm on E;
    - how much weaker the square-on rig is: 3–9×, not about 5×.
- **The calibration was built and tested on synthetic data (W0–W9),** all phases of the approved plan.
  - **The plan's from-scratch target was not met:** 46/50 and 5/10, against ≥ 49/50 and 10/10 (section 3, item 2).
  - **Found on the way and fixed in the new code** (implementation doc, section 10):
    - the planned stage-2 distance search: the affine points bias it about 10× too far, so the distance now comes
      from perspective;
    - cameras sliding onto their window during resection: their distance is now held, and they are placed on their
      exact chief rays;
    - windows with one or two cameras, which a fit to fixed points cannot judge: now also scored by the wand-length
      residual;
    - per-camera noise weights drifting with two cameras;
    - a gross outlier costing its whole frame on re-admission.
  - The last two also affect `cylinder_wand.refine`. They are logged in the cylinder summary as open item 7; the
    cylinder code is unchanged.
  - The approved two-line rotation fix in `cyl_model.py` (`log_so3`, `rvec_of`).
  - In pyOpenLPT commit `5391cb3` (branch `Refraction_new`).

## 2. Status

- **What it does:** it calibrates cameras that look through one or more flat windows (the walls of a tank), from
  wand frames alone.
  - It estimates every camera's pose and f, every window's normal and position, each camera's noise, and the error
    bars of all of them.
  - It needs no focal length, camera pose, window position or window orientation. It needs only:
    - the detections and the wand length;
    - the image sizes;
    - each window's plate thicknesses and indices;
    - which camera looks through which window.
  - It writes pyOpenLPT PINPLATE camFiles, which the C++ reloads and projects within 7e-12 px (W9), a result JSON and a
    text report.
- **How** (`calibrate`, stages 1–5):
  1. an affine reconstruction of the wand points, made metric by the wand length;
  2. a pinhole start:
     - each camera's rotation from its affine camera;
     - its distance from perspective, since a wand's image shrinks with depth;
     - then a pinhole bundle adjustment;
  3. the windows:
     - **candidates:** normals on a 50° cap at 9° spacing, and three offsets;
     - **for each candidate:** the cameras are placed on their exact chief rays and resected, and the best candidates
       are refined together with the window;
     - **a window with one or two cameras** is also scored by the wand-length residual;
     - **several windows:** each group of cameras sharing a window is calibrated on its own, the groups are joined,
       and single cameras are placed;
     - **with window-normal hints** (`hints={"normals": [...]}`, optional; 3 Oct): the start is built three ways:
       without the hints, with every hinted normal held, and with a second search about each hint. Each answer gets
       the restarts of stage 4, and a hinted one is kept only when it fits better (`replaces`), so a hint cannot make
       the answer worse;
     - **then a coarse fit;**
  4. the maximum-likelihood bundle adjustment, with:
     - the wand frames eliminated and the wand length exact;
     - outlier rejection and re-admission;
     - per-camera noise weights (equal with two cameras);
     - several starts when they disagree;
     - on rigs with several windows, restarts perturbed about the answer: four draws at 2σ, screened, the cheapest
       refitted if it lowers the cost, and kept if it is better on the same terms (a cost capped at the rejection
       limit);
     - on rigs with several windows, a scan of every weakly seen window's normal (one or two cameras that all see the
       wand under less than 6°; 3 Oct):
       - a 30°-wide grid of fits with that normal held, 3° apart;
       - the grid's best minima released, and kept only when they fit better (`replaces`);
  5. the covariance, the output frame and the files. `calibration_report.txt` opens with WARNINGS: what the tests
     associate with a wrong or weak answer, each with its evidence and what to do (3 Oct).
- **Tested on:** synthetic data only.
  - The data come from pyOpenLPT's C++ PINPLATE camera, through air, acrylic (n 1.49) and water (n 1.333). The walls
    are 12.7 mm thick on the reference rigs and 6–25.4 mm on the random ones.
  - **Rigs:** the five reference rigs A–E (section 3, item 4) and 60 random rigs, with 2–6 cameras and 1–3 walls of a
    box tank.
  - **Wands:** 25 mm long (10 mm on rig E), and 0.2–0.35 of the cloud's size on the random rigs.
- **Not done yet:** real data (W10) and the GUI integration.
- **Code,** in `modules/camera_calibration/curved_calibration/`:
  - `plate_model.py`: the PINPLATE camera, vectorised, with exact derivatives. It is checked against the C++ (W0),
    and its derivatives against finite differences (W1);
  - `plate_wand.py`: the calibration (`calibrate`). It imports the model-free pieces of `cylinder_wand.py`;
  - `plate_wand_io.py`: the GUI's wand CSV, a rig configuration file, camFiles, the result JSON and the report.
- **Tests:**
  - `tests/test_plate_wand.py`: 12 fast checks and four calibrations from scratch (marked slow); all pass.
  - The campaign is in this folder.
  - pyOpenLPT's full suite (3 Oct): 245 passed, 7 skipped, 2 failed, both known and unrelated.
- **Run time** (from scratch, 1,000 frames):
  - **one window:** a median 35 s for 4 cameras (20–79 s for 2–6 cameras), with the machine fully loaded. In one
    quiet process, rig A takes about 26 s;
  - **several windows:** a median 121 s (up to 354 s). Each window group is calibrated on its own, and two searches
    are pooled. The restarts add a median 8 s (9 %), at most 30 s, on W5's rigs. On R4's larger rigs they add a
    median 14.5 s (13 %), at most 184 s.
- **Run time with the 3 Oct additions:**
  - **a weak window** adds its scan: 2–13 min per case with 32 calibrations at once, less on a machine running one;
  - **hints** make a calibration a median 2.55× as long (1.2–3.4×; H1b).
- **Git:** pyOpenLPT commit `e1c52da` (branch `Refraction_new`). The restarts, the window-normal hints, the scan and the
  warnings (3 Oct) are not committed yet.

Use:

```python
import plate_wand as pw, plate_wand_io as pio, plate_model as pm   # modules/camera_calibration/curved_calibration
cd = pio.read_gui_csv("wand.csv")                     # the GUI's "Export wand data" CSV
rig = pw.Rig([pm.Window([0, 0, 1], 0.0, (12.7,), (1.0, 1.49, 1.333))],   # plates (mm); indices camera side first;
             cam_win=[0, 0, 0, 0])                    #   normal and offset are placeholders: calibrate finds them
sol = pw.calibrate(pw.WandData(cd.uv, 25.0, [(1280, 800)] * 4), rig)    # options: fit_k1=True; hints={"f": [...]}
pio.write_result(sol, "out")                          # out/camFile/cam1.txt ..., calibration_result.json, report
```

Or, with a rig configuration file, `pio.calibrate_csv("wand.csv", "rig.json", "out")`. The file has one camera entry
per camera:

```
{"windows": [{"name": "front", "thickness_mm": [12.7], "n": [1.0, 1.49, 1.333]}],
 "cam_to_window": [0, 0, 0, 0],
 "cameras": [{"name": "cam1", "width": 1280, "height": 800, "pixel_mm": 0.02}, ...],
 "wand_length_mm": 25.0}
```

The output frame:
- Z is window 1's normal, from the cameras into the water;
- Y is along n1 × n2 with two windows that are not parallel, otherwise camera 1's image "up" within window 1's plane;
- the origin is at the wand's mean position.

## 3. What the tests concluded

1. **The camera model is exact (W0, W1).**
   - It matches pyOpenLPT's C++ PINPLATE camera to 6.5e-11 px, and the rays to 1.3e-13. The comparison used 2,000
     random cameras, windows (one and two plates, incidence up to 60°) and points.
   - Its derivatives match central differences to 1.3e-7 (the bundle adjustment's to 3e-7).
   - SymPy proves 77 steps of its derivation. Two limits are SKIPPED there; W1 checks them numerically.
   - **A C++ quirk, recorded and not changed:** the C++ projects a point on the camera side of a window (in the plate
     or the air) as its mirror image in the window (600 of 600). The new model reports such a point as not projecting.
     The GUI's pipeline uses the C++ and can be misled by it.
2. **From scratch, one window always succeeds; several windows usually do** (W5: 60 random rigs, no hints; rerun on
   3 Oct with the restarts, R3).

   | rig kind | 0.1 px: same minimum | 1 px: same / equivalent | time (median) |
   |---|---|---|---|
   | one window | 34/34 | 3/3 / 3/3 | 35 s for 4 cameras |
   | several windows, two or more cameras on some | 11/13 | 3/4 / 4/4 (before the restarts 2/4 / 3/4) | 121 s, + 8 s for the restarts |
   | several windows, one camera each | 1/3 | 1/3 / 3/3 (before 0/3 / 3/3) | 121 s, + 8 s |
   | all | **46/50** | **7/10 / 10/10** (before 5/10 / 9/10) | |

   - **What the columns mean:**
     - **same minimum:** every parameter within 0.01σ of the fit started from the truth, the same detections kept and
       the same RMS;
     - **equivalent:** every parameter within 3σ and the RMS within 2 %, or a lower RMS.
   - **The four different answers (seeds 1001, 1031, 1034 and 1045, all at 0.1 px) have a window seen by only one or
     two cameras,** and a camera with f of 24,000–37,000 px.
     - A narrow view sees a window's tilt only weakly, so the cost has several minima.
     - Even the right minimum is then poorly determined: seed 1001's second window has σ 2.1° and 438 mm.
   - The error bars warn of it: a weakly determined window shows a large σ.
   - **Rigs whose windows all have two or more cameras** reached the global minimum at 0.1 px every time (8 of 8).
     - At 1 px, 3 of 4 do, and the fourth is equivalent.
     - Seed 2000 was the miss among them until the restarts.
     - On 30 new such rigs at 0.5 and 1 px (R4), 57 of 60 cases reach the global minimum with the restarts (55
       without), and 59 an equivalent answer or better.
   - **A wrong answer can be pinned against a window** (R1b, report of 3 Oct).
     - A wand end lies exactly on the window's water-side surface. The final fit rejects any step that takes a wand end
       out of the water, so it stops there although it is not a minimum of the cost.
     - 8 of the 14 multi-window misses in W5 and R4 were pinned, and none of the 69 answers at the global minimum was.
     - The restarts free such an answer when they move the window back off the cloud.
       - Fixed: W5 seed 2000 and R4 seeds 3029 and 3100.
       - Equivalent now: R4 seed 3111 at 0.5 and 1 px.
       - Improved but still wrong: W5 seed 1034, freed into a second wrong minimum.
3. **It reaches the theoretical limit, and its error bars match the real scatter (W3).**
   - Rigs A–E, 40 data sets each at 0.1 px, each fitted from the truth:

     | rig | parameters | RMS error / Cramér–Rao | z standard deviation | largest bias |
     |---|---|---|---|---|
     | A | 25 | 0.98 | 0.98 | 1.7 SE |
     | B | 28 | 0.99 | 1.00 | 1.8 SE |
     | C | 18 | 0.97 | 0.98 | 2.3 SE |
     | D | 18 | 0.99 | 0.99 | 2.4 SE |
     | E | 18 | 0.97 | 0.97 | 1.3 SE |

   - Errors grow in proportion to the detection noise from 0.02 to 2 px (rig A).
   - Without noise, a fit from the truth reaches an RMS of 1e-12 px, with every parameter at round-off.
4. **What the data determine (W2, Cramér–Rao at 0.1 px, 1,000 frames).**
   - **The reference rigs:**
     - a 100 mm cube of wand positions, its centre 150 mm behind a 12.7 mm acrylic wall;
     - 1280 × 800 cameras, 600 mm from the wall along their centre rays, with f of 3,900–4,400 px;
     - rig E is the exception: a 16 mm cube 100 mm behind the wall, seen with f = 40,000 px from 1,100 mm.

   | rig | f | window normal | camera's distance to window | 3-D shape of unseen points (calibration) | 25 mm wand, calibration / with detection noise |
   |---|---|---|---|---|---|
   | A, one wall, 4 cameras at 21° | 0.063 % | 0.026° | 0.47 mm | 5.5 µm | 1.9 / 30 µm |
   | B, two walls, 2 + 2 cameras at 15° | 0.055 % | 0.018°, 0.016° | 0.95 mm | 2.4 µm | 0.9 / 16 µm |
   | C, 3 cameras 5° apart, looking straight through | 0.20 % | 0.14° | 4.4 mm | 26 µm | 8.9 / 148 µm |
   | D, steep, 35–45° | 0.059 % | 0.015° | 0.37 mm | 4.3 µm | 1.5 / 23 µm |
   | E, macro, f 40,000 px | 0.61 % | 0.016° | 6.7 mm | 0.6 µm | 0.6 / 8 µm (10 mm wand) |

   - Everything scales with the noise: 10× these values at 1 px.
   - **The calibration's share is small.** On rig A a single point triangulated from 0.1 px detections is itself
     uncertain by 37 µm (10–198 µm across the rigs). The calibration adds 5.5 µm.
   - Outside the calibrated volume the shape error grows about linearly with the distance: 2× at twice the volume.
5. **Shapes are known far better than positions relative to the cameras (W2).**
   - The scene, the windows and f can slide together along the viewing direction.
   - Relative to camera 1, the scene's position is known to:
     - 0.43–0.48 mm on rigs A, B and D;
     - 1.6 mm on C;
     - 7.3 mm on E.
   - Its shape is known to 0.6–26 µm.
   - Distances, velocities and shapes measured inside the volume carry only the shape error.
6. **Rig design (W2, W4, `p_mixed_views.py`).**
   - **The angle between the cameras' lines of sight sets the precision.** With two cameras through one window:

     | angle between the cameras | σ(f) | 3-D shape of unseen points |
     |---|---|---|
     | 5° | 0.31–0.35 % | 46–47 µm |
     | 20° | 0.15–0.16 % | 13.5 µm |
     | 35–40° | 0.09–0.11 % | 7.6–8.6 µm |

     - Rig C, three cameras 5° apart, is 3–9× weaker than rig A on every measure for this reason.
     - The GUI's calibration failed on it twice (W6), in its pinhole start.
   - **Looking straight through the window costs nothing.** Moving rig C's layout from 0–5° to 18–25° incidence
     changes σ(f) by 7 %, the cameras' distances by 17 % and the 3-D shape by 5 %; the window's normal is even 30 %
     less certain at 20°.
     - With one camera square-on and another at an angle, the square-on camera is as precise as the angled one
       (σ(f) 0.145 % against 0.162 % at 0° + 20°).
     - In principle a camera looking straight through cannot see the window's position paraxially: moving the window
       towards it and the scene away by a third as much leaves its images unchanged (checked with SymPy). In these
       rigs that barely matters, presumably because the rays towards the image's edges are oblique enough.
   - **Steep views are fine.** Rig D, at 35–45°, is as precise as A or slightly better.
   - **Two perpendicular walls give the best shape:** rig B, 2.4 µm against 5.5 µm on A.
   - **Long lenses:** f and the camera's distance trade almost freely.
     - Rig E, with f = 40,000 px, has f to 0.61 % and the distance to 6.7 mm.
     - Its shape is still the best of all: 0.6 µm on a 16 mm cloud.
   - **How close the start must be (W4).** The final fit recovers:
     - a window normal 60° off, on every rig but the steep D, which needs it within 30°;
     - any window offset between the cameras and the cloud;
     - f within ±40 % on rigs A, B and D (C: 9 of 10 at 40 %), but only within ±10 % on the macro rig E.
   - **Camera rotations are the weak spot** of that basin: A 8/10 at 5°; C 5/10 and E 1/10 at 10°. The start
     therefore takes them from the wand data, which gives them to about 1°.
7. **Against the GUI's current calibration on the same data (W6, 18 cases).**
   - **The setup:**
     - the GUI's `RefractiveWandCalibrator`, run headless as `view.py` runs it and **given the true focal lengths**;
       the new calibration was given none;
     - 1,000 frames: rigs A–D at 0.1 and 0.5 px, and ten W5 rigs at 0.1 px;
     - each judged on its exported camFiles through the C++.
   - **Time:** 7–112× faster, median 63×: 38 s against 29 min (medians).
   - **Failures:** the GUI failed both cases of rig C, whose cameras are 5° apart:
     - at 0.1 px its scale collapsed;
     - at 0.5 px it found no valid camera pair.

     The new calibration solved both, at about the precision W2 predicts.
   - **Accuracy on rigs A, B and D:** comparable. The 3-D shape error medians are 6.9 µm (GUI) and 6.2 µm (new),
     both near the Cramér–Rao bound.
   - **Accuracy on the ten random rigs:** the new calibration is far better.
     - 3-D shape error: 1.5 µm against 59 µm (medians), better on 9 of 10, by up to 400×.
     - f error: 0.12 % against 2.7 % (medians). The GUI's f is held within ±5 % of the value it was given, and two of
       its runs ended on that bound.
8. **Bad data are handled (W7: rigs A and B, from scratch).**
   - **2 % gross outliers of 20–200 px:** all 160 rejected; 0–6 good detections lost.
   - **1 % of camera-frames with the ends swapped:** all rejected; no good detection lost.
   - **A camera missing from 40 % of the frames:** the answer moved only as much as the lost data explain.
   - **A camera with 5× the noise:** its noise was estimated at 0.503 px (true 0.5 px), and it was weighted
     accordingly.
   - **No frame seen by every camera:** handled.
     - Rig A uses a camera-subset start.
     - On rig B, a window group that never shares a frame falls back to the whole-rig start.
     - Rig B's answer moved 1.11σ RMS from the clean answer, against 0.72σ expected. That is a marginal miss of the
       1.5× limit, but it is still at its data's own minimum.
9. **The window constants hardly matter; the wand length is the only scale (W8, rigs A and D).**

   | given | residual RMS | 3-D shape of unseen points | what shows it |
   |---|---|---|---|
   | correct | — | 4.3–5.4 µm | — |
   | plate 10 % thicker | unchanged | unchanged | nothing |
   | water index +0.002 | unchanged | 5.3–5.4 µm | nothing |
   | wand 0.1 % long | unchanged | 49–51 µm (a 0.1 % scale error) | only an independent length |
   | unmodelled k1 = 0.05 | +0.3–0.4 % | 7–8 µm | barely the residuals |

   - **The free wand length is not identifiable.** With it free (`fit_L=True`), the fit returns 9–91 mm ± 5–18 mm for
     a 25 mm wand.
   - Only the plate's thickness carries scale, and 12.7 mm of acrylic is a poor ruler.
   - So `fit_L` is no check of the constants here. Through the tube it is: there it detects a water index 0.002 off at
     33σ.
10. **An unmodelled lens distortion hides in the residuals (W8).**
    - A k1 of 0.05 left out of the model raises the residual RMS by only 0.3–0.4 %, but the 3-D error 1.3–1.9×.
    - With `fit_k1=True` the fit recovers it (k1 0.043–0.058 for the true 0.05), and the 3-D error returns to that of
      the correct model.
11. **The files work (W9).**
    - The camFiles reload in the C++ and project within 7e-12 px, with the camera and window read back exactly.
    - The GUI's wand CSV round-trips exactly, and the GUI's own loader reads the CSV written here.
12. **Two windows at any angle (A1, 3 Oct).**
    - Window 2 turned 30°–180° from window 1, two cameras behind each: 56 of 56 calibrations from scratch reached the
      global minimum, at 0.1 and 1 px.
    - The precision follows the angles between the lines of sight (Cramér–Rao at 0.1 px, 1,000 frames):

      | angle between the windows | 30° | 45° | 60° | 90° | 120° | 150° | 180° |
      |---|---|---|---|---|---|---|---|
      | 3-D shape of unseen points | 3.9 µm | 3.3 µm | 2.8 µm | 2.6 µm | 2.8 µm | 3.9 µm | 5.6 µm |

    - Opposite walls (180°) are the weakest: the cameras face each other, so their lines of sight are nearly parallel
      as lines.
13. **Hints on the windows' normals rescue most hard cases, and never make an answer worse** (H1a–H4b, D5–D8;
    3 Oct).
    - **What was learnt on the way:**
      - the final fit converges from a weak window 20° off when the rest of the start is right (D6);
      - the hard cases fail because their starts are wrong in several things at once;
      - the answers must be compared after their restarts (D7).
    - **The design kept (H3):** three start paths (without hints, hints held, a search about the hints), each with its
      own restarts; a hinted answer is kept only when it fits better.
    - **On the five hard cases (H4b, 65 hinted runs):**
      - none worse than without hints;
      - 56 at least equivalent; within 20° of the truth, 45 of 50;
      - every miss is on seed 1045, two cameras each alone behind its window with a long lens.
    - **On W5 and R4** (H1b, every camera hinted 0–10° off):
      - none of 120 cases worse;
      - the three still wrong without hints (1031, 1034, 1045) were right;
      - W5 at 0.1 px: 50 of 50 at least equivalent;
      - a median 2.55× the time.
14. **A weakly seen window's normal is now scanned, which fixed 2 of the 5 remaining wrong answers without hints** (D8,
    H4, H4b; 3 Oct).
    - **What D8 found:** the wrong answers are separate minima, 10–50° from the truth, behind a ridge in the cost. The
      truth's valley is only a few degrees wide.
    - **The scan:** a grid of fits with the normal held, the best minima released.
    - **Without hints, now right:** R4 3110 and W5 1001. Still wrong: W5 1031, 1034 and 1045. Each has two windows
      wrong at once, or fits pinned against a window, which a scan of one normal cannot leave.
    - **Nothing else changed:** H5a, W5 and R4 rerun: 118 of 120 answers unchanged, the other 2 better.
15. **The report now warns** (H5, H5a; 3 Oct). `calibration_report.txt` opens with WARNINGS on:
    - a weak window;
    - an answer pinned against a window;
    - a hint far from the answer;
    - cameras all looking the same way;
    - large residuals on a multi-window rig;
    - few frames;
    - a fitted wand length.

    At 0.1 px every wrong answer in W5 and R4 had a warning (3 of 3), and only 3 of the 47 right answers did. At
    0.5–1 px most multi-window cases get the residual warning, as they should.

## 4. For a real experiment

**How to run it today** (the GUI does not call it yet):
1. export the wand detections from the GUI (its wand CSV);
2. write the rig file of section 2;
3. run `plate_wand_io.calibrate_csv("wand.csv", "rig.json", "out")`.

The camFiles it writes load in pyOpenLPT's C++ camera (W9).

**Before recording**
- **Calibrate with the experiment's exact optics.** Keep the lens settings (focus, aperture), the cameras, the tank
  and the water as they will be during the measurement.
- **Measure the wand length independently, to 0.1 % or better** (25 µm on 25 mm).
  - It is the only scale: a 0.1 % error makes every distance 0.1 % wrong.
  - Nothing in the wand data shows it (W8).
- **Enter the windows' constants as well as you know them, but they are not critical.**
  - A plate 10 % too thick or a water index 0.002 off changed the 3-D shape error by at most 1 µm on the reference
    rigs (W8).
  - Give each window its plate thicknesses and its indices, both in order from the camera side.
- **Spread the cameras' lines of sight: 20° or more between them, 35–40° better.**
  - Cameras 5° apart give 2× the f error and 3.5× the 3-D error of cameras 20° apart (`p_mixed_views.py`). Rig C's
    layout also defeated the GUI's calibration (W6).
  - A camera may look straight through the window: that costs nothing, also next to angled cameras.
  - Steep views (up to 45° tested) are fine.
  - Views through two perpendicular walls give the best shape (W2, rig B).
  - **Windows at any angle to each other work** (A1: 30°–180°), e.g. the walls of a hexagonal or octagonal tank, or a
    prism. Opposite walls give the weakest shape (2.2× the 3-D error of perpendicular walls), because the cameras
    face each other.
- **Put two or more cameras behind every window, at least one of them with a wide enough view.**
  - **Wide enough:** the wand's positions span at least about f/9 pixels across its image diagonal, i.e. it sees the
    wand under 6° or more.
  - **The evidence:** every wrong answer in W5 and R4 had a window seen only by one or two cameras with narrower views
    (`2026-10-03-when-calibration-fails.md`). The scan of such a window's normal now fixes some of them, not all.
  - **Never one camera per window with long lenses** (W5 seed 1045). It is the one rig still wrong with most hints.
  - **If a window must have only narrow-view cameras, give window-normal hints for them** (`"hints": {"normals":
    [...]}` in the rig file):
    - within 20° they fixed every two-camera window tested;
    - a window behind a single long-lens camera needs them within about 5°;
    - they never made an answer worse (H4b).
- **If the lens may distort, fit k1** (`fit_k1=True`). An unmodelled k1 barely shows in the residual RMS, but
  raises the 3-D error up to 1.9× (W8).
- **Make the two ends distinguishable,** for example with two ball sizes.
  - Label A and B the same way in every camera.
  - Occasional swaps are rejected as outliers (W7).
- **Check the shape of the balls' images on the first recordings.**
  - Through a window at an angle a ball's image can be slightly elongated.
  - Through the tube, pyOpenLPT's circle detector failed on 1.3 : 1 images (cylinder T13).
  - This was not tested for flat windows (section 5). If the images are clearly elongated, use an ellipse fit, as
    T13 recommends.

**While recording**
- **Move the wand through the whole volume you will measure in,** including its full depth as seen from each camera.
  The calibration is only good where the wand went: the shape error doubles at twice the volume (W2).
- **Record about 1,000 frames with both ends seen by at least two cameras.**
  - The tests used 500–1,500. Other frames are dropped.
  - The start needs at least 12 frames that show the whole wand in two or more cameras, ideally in all of them.
  - To join the cameras of different windows, their groups must share at least 3 frames.
- **Use three or more cameras if you can.**
  - With two cameras the noise weights stay equal.
  - Windows with only one or two cameras are the hard case.
- **Aim for about 0.1 px detection noise:** sharp focus, short exposure, well-lit balls.
  - Every error scales with it (W3).
  - The multi-window failures cluster at high noise: 6 of the 8 pinned answers came at 1 px. At 0.1 px, every W5 rig
    with two or more cameras on every window reached the global minimum (8 of 8).
- **Synchronise the cameras, or move the wand slowly.** The tests assume exactly simultaneous frames.

**Checking the result** (the RMS alone can hide a model error)
- **Read the WARNINGS block at the top of `calibration_report.txt` first** (`2026-10-03-when-calibration-fails.md`,
  section 7).
  - At 0.1 px, every wrong answer in the tests had a warning.
  - A `weak_window` or residual warning on a rig with several windows is the main sign of a possibly wrong answer.
- **Read the noise section of `calibration_report.txt`.**
  - Each camera's noise should match your detector's. One camera much noisier than the others points to blur or
    focus.
  - With two cameras the weights stay equal, and the report says why.
- **Compare that noise with a still wand imaged over many frames,** which gives the detection noise alone. If the
  fit's noise is clearly larger, suspect model error, motion or synchronisation.
- **Read the windows' and cameras' σ in the report.**
  - A window whose normal is uncertain by degrees, or a camera whose distance is uncertain by tens of mm, is poorly
    determined (W5 seed 1001: 2.1° and 438 mm).
  - Add a camera behind that window, or record more depth.
- **With several windows, read the report's line on the restarts.**
  - "IMPROVED" means the first answer was a wrong minimum, usually one pinned against a window; the restarts' answer
    is the one written.
  - `calibration_result.json` has each window's nearest wand end before each round of restarts
    (`info.starts.perturbed.rounds[k].min_depth_mm`).
  - A value near 0 that remains after the restarts, together with a window σ of tens of mm, means the data hold that
    window's position only loosely. More cameras behind it, or less detection noise, would help.
- **Do not use `fit_L` as a check.** The free length is not identifiable through flat windows (W8).
- **Validate on something of known size instead,** such as a second wand of measured length, or a dot target imaged
  through the window and triangulated.
- **Plot each camera's residuals binned over the image.** A pattern fixed to each image hides in the RMS but shows up
  as structure (cylinder T11e).
- **For the first data set** (W10): run this calibration and the GUI's on the same wand CSV, and compare:
  - each camera's noise;
  - the free-triangulation wand length;
  - a known target.

  Calibrating each half of the frames on its own and comparing the cameras within their σ is a further check, not
  part of these tests.

## 5. Not covered by any test

- Real data (W10).
- The GUI integration. The CSV reader and writer are done and tested (W9).
- Fitting k1 in a calibration from scratch. k1 was fitted only from the truth (W8). Data with an unmodelled k1 were
  calibrated from scratch too, and gave the same answer as from the truth.
- Windows of two or more plates in a calibration. The model handles them (W0).
- Windows at other angles were tested only as two windows with two moderate-lens cameras each, their normals turned
  about one axis (A1). Not tested: a single camera or long lenses at other angles, more than 3 windows, and more than
  6 cameras.
- Cameras with no window (`cam_to_window` −1): accepted, but never run.
- Ball centres through flat windows. The tube's T13 has no flat-window counterpart.
- Patterns fixed to each image (cylinder T11e), noise that varies with position or depth, a bending wand, and camera
  synchronisation.

## 6. Open items and possible next steps

None of these is started; each needs your go-ahead.

1. **Windows seen by one or two long-lens cameras** (W5's misses). Done on 3 Oct: hints in three start paths (H3), the
   scan of a weak window's normal (H4), and warnings (H5). Left wrong without hints: W5 1031, 1034 and 1045. Ideas
   that remain, none started:
   - **a joint scan of two weak windows' normals,** for rigs like 1045. It is four-dimensional, so about 100× the cost
     of a scan;
   - **a scan that does not carry a pinned fit from point to point** (1034);
   - **the weak-window rule judged on the hints or the data,** not on the answer's focal lengths, which a wrong answer
     can distort (1031);
   - **focal-length hints** exist (`hints={"f": [...]}`) but were set aside: f is hard to guess in a real experiment.

   **Answers pinned against a window** (seed 2000's kind of miss): the restarts perturbed about the answer are done
   (3 Oct; `2026-10-03-perturbed-starts-report.md`). Two follow-ups:
   - **How a restart is accepted: done (3 Oct, R7).**
     - It is now judged on a cost capped at the rejection limit, not by the starts' rule.
     - That rule had blocked a found fix in R4 (seed 3111 at 1 px).
   - **A warning in the report** when a wand end lies on a window's surface: done (3 Oct, H5, `pinned`).
2. **GUI integration.** `view.py`'s refractive path would call `plate_wand.calibrate` and write the camFiles where the
   GUI expects them, with the current path kept as an option.
3. **The first real data set (W10):** the GUI's wand CSV, the window settings, the wand length and the current
   calibration's camFiles of one experiment. Run it with the checks of section 4.
4. **Port the two `refine` fixes to the cylinder calibration** (its summary, open item 7):
   - the robust re-triangulation of a frame before re-admission;
   - equal noise weights with two cameras.
5. **The C++ mirror-image quirk** (section 3, item 1). The C++ library was left unchanged. A fix there would report a
   point on the camera side of a window as not projected.

## 7. Files

All in this folder unless stated.

**Plans and reports**, in order:

| file | what |
|---|---|
| `2026-10-02-plate-wand-plan.md` | the approved plan: model, solver, start, tests W0–W10 |
| `2026-10-02-plate-wand-implementation.md` | the working design: section 9 the checklist, section 10 every deviation from the plan with its evidence |
| `2026-10-02-plate-wand-report.md` | the full report: what was built, each test's result, the limitations |
| `2026-10-02-perturbed-starts-plan.md` | the plan for perturbed restarts of the final fit on multi-window rigs (R1–R6); approved 3 Oct |
| `2026-10-03-perturbed-starts-report.md` | its report: the pinned answers, the change, R1–R6, the proposal on accepting a restart |
| `2026-10-03-angles-and-hints-plan.md` | the plan for two windows at other angles (A1) and window-normal hints for the failed cases (H1); approved 3 Oct; section 5 the revision H2, section 6 H3, section 7 what followed your "fix it as much as possible": H3 as built, D8, the scan of weak windows (H4, H4b) and the report's warnings (H5) |
| `2026-10-03-angles-and-hints-report.md` | its report: A1 passed; H1a failed (D5, D6); H2a failed (D7); H3 (6c), D8 (6d), H4 (6e) and the final tests |
| `2026-10-03-when-calibration-fails.md` | **for a real experiment:** when the calibration cannot work, how the report shows it, and what to do (your request of 3 Oct) |
| `2026-10-03-when-calibration-fails.html` | the same as a page, with two figures (the weak-window rule on W5 and R4; the hard cases with hints); built by `make_fails_page.py` from the saved results and `figures/fails_template.html`; also at https://claude.ai/artifact/Ue22gHGbyjWDNn2GnRBAyf |

**The tests:**

| test | what | script |
|---|---|---|
| W0 | the model: SymPy checks, against the C++, the rotation log map | `w0_sympy_derivations.py`, `w0_model_vs_cpp.py`, `w0_logmap.py` |
| W1 | derivatives against finite differences | `w1_jacobians.py`, `w1_block2.py` |
| W2 | what the data determine (Cramér–Rao) | `w2_observability.py` |
| W3 | from the truth: noise-free, Monte Carlo, noise sweep | `w3_noise_free.py`, `w3_monte_carlo.py` |
| W4 | how close the start must be | `w4_basin.py` |
| W5 | from scratch, 60 random rigs | `w5_from_scratch.py` |
| W6 | against the GUI's calibration | `w6_vs_gui.py`, `w6_run.py` |
| W7 | bad data | `w7_bad_data.py` |
| W8 | wrong constants, free wand length, k1 | `w8_constants.py` |
| W9 | camFiles and the GUI's CSV | `w9_files.py` |
| W10 | real data | not run: no data set |
| R1–R7 | the restarts perturbed about the answer: sizing, why draws fail, the draws, W5 and W7 again, 30 new multi-window rigs, the blocked fix, the two acceptance rules compared | `r1_sizing.py`, `r1b_failed_draws.py`, `r2_draws.py`, `r3_compare.py`, `r4_group_rigs.py`, `r4b_blocked_restart.py`, `r5_compare.py`, `r7_acceptance.py` |
| A1 | two windows at 30°–180°: Cramér–Rao and from scratch | `a1_window_angles.py` |
| H1a, H1b | window-normal hints: on the five failed cases (first version); on W5 and R4 with the final design | `h1_hints.py`, `h1b_hints_regression.py` |
| H2a | the hint-held starts on the five failed cases | `h2a_hints_held.py` |
| H3a | the three start paths with their own restarts, on the five failed cases | `h3a_hint_paths.py` |
| H4 smoke, H4a, H4b | the scan of weak windows' normals: three cases; the five failed cases, before and after two refinements | `h4_smoke.py`, `h4a_scan.py`, `h4b_scan.py` |
| H5a | W5 and R4 without hints on the final code: the answers, and the report's warnings against the verdicts | `h5a_warnings.py` |

- Each script writes `<name>_out.txt` beside it, and most also `<name>.json`.
- Shared code: `prigs.py`, with rigs A–E, the random rigs, wand data from the C++, alignment and metrics.
- Diagnoses of the multi-window start: `d1_multiwindow.py`, `d2_groups.py`, `d3_single.py`, `d4_seed2000.py`, with
  their outputs. Of the hints (3 Oct): `d5_hints.py` (where a hint is lost, first version), `d6_basin_failed.py` (the
  final fit's basin on the failed cases), `d7_held_acceptance.py` (why a hint-held answer came out worse) and
  `d8_profile.py` (the cost over a weak window's normal).
- Pre-plan sizing: `p_sympy_presize.py`, `p_pre_sizing.py`, `p_logmap_check.py`.
- Sizing studies for later questions:
  - `p_mixed_views.py`: square-on and angled cameras, and cameras close together;
  - `p_weak_windows.py`: which rigs W5 and R4 got wrong (the weak-window rule);
  - `p_held_fit_timing.py`: the time of one held fit.
- Earlier runs, kept for the record:
  - `runs_v1/`: W5 run 1, and W7 before the re-admission fix;
  - `runs_v2/`, `runs_v3/`: W5 runs 2 and 3;
  - `runs_v4/`: W5 run 4 and W7 run 2, the last before the restarts (3 Oct);
  - `runs_v5/`: W5 run 5, W7 run 3 and R4 run 1, with the first version of the restarts (judged by `_choose`).
- `w6_cases/`: W6's data and both calibrations' working files. It is git-ignored, and `w6_vs_gui.py prepare` and
  `run` rebuild it.
- `tests/test_plate_wand.py` (from the pyOpenLPT root): the pytest checks.

**Figures:**
- `figures/plate_wand_results.html` (open it in a browser), also at
  https://claude.ai/artifact/ALBPbfKJKbMddmNBp7m2iV.
- `make_figures.py` rebuilds it from the saved results; nothing is re-fitted.
  - It fills `figures/template.html` and writes the data to `figures/figure_data.json`.
  - `--fragment PATH` also writes the page without its document skeleton, for republishing.

**Example output:** `w9_result/` holds a complete result for rig A: `camFile/`, `calibration_result.json` and
`calibration_report.txt`.

**Rerunning** (from the pyOpenLPT root):

```
python tests/plate_wand/w0_sympy_derivations.py                          # base python (sympy)
C:\Users\zcloc\.conda\envs\refract\python.exe tests/plate_wand/w0_model_vs_cpp.py   # and w0_logmap, w1_jacobians,
                                    # w2_observability, w3_noise_free, w9_files: seconds each
... w3_monte_carlo.py, w4_basin.py, w5_from_scratch.py, w7_bad_data.py, w8_constants.py   # parallel: 2-10 min each
... w6_vs_gui.py prepare | run | evaluate                                 # run: about an hour (the GUI's runs)
... r1_sizing.py (5 min), r2_draws.py (1.5 min), r4_group_rigs.py (10 min); r3_compare.py, r5_compare.py and
    r7_acceptance.py after w5_from_scratch.py and w7_bad_data.py          # the restarts (3 Oct)
... a1_window_angles.py (4 min), h2a_hints_held.py (19 min), d6_basin_failed.py (3 min),
    d7_held_acceptance.py (15 min)                                        # angles and hints (3 Oct); h1_hints.py and
                                    # d5_hints.py ran on the first version, which H2 replaced
... h3a_hint_paths.py (31 min), d8_profile.py (40 min, 4 processes), h4a_scan.py (85 min), h4b_scan.py,
    h5a_warnings.py, then h1b_hints_regression.py (it compares with H5a's answers)   # hints and the scan (3 Oct)
... p_weak_windows.py (2 min, one process)                                # the weak-window rule
... make_fails_page.py [--fragment PATH]                                  # the when-it-fails page (seconds; no fits)
pytest tests/test_plate_wand.py                                           # 12 fast checks; -m slow: four from scratch
```

`PLATE_WORKERS` sets the parallel scripts' process count (default 32).
