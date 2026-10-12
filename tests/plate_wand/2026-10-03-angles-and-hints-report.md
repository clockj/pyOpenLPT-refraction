# Windows at other angles, window-normal hints, the scan of weak windows and warnings: report (A1, H1–H5, D5–D8)

Date: 2026-10-03. Plan: `2026-10-03-angles-and-hints-plan.md`, approved with one change ("For H1, start with the
second part for hints on window normal, but that hints is not guaranteed to be super accurate ... Now you can go ahead
for A1 and H1's 2nd part"). It answers "How to avoid the failed cases? Have you tested for two windows with different
angles?".

Status:
- **A1 passed.**
- **H1a failed** (the first version). Its diagnosis (D5, D6) is in section 5.
- **The revision H2,** hint-held starts, was approved ("Ok go ahead"), built and tested (section 6a). **H2a failed** too,
  and its diagnosis (D7) is in section 6a.
- **The further revision H3** (section 6b) was approved with its third start: "go ahead to try to fix it as much as
  possible. in the end, report when the calibration cannot work so that i can pay attention to in the real
  experiments". It is built and tested (section 6c).
  - **H3a: no hinted run is worse than without hints any more;** 48 of 65 reach at least an equivalent answer.
  - It still fails its strict criterion on windows behind a single long-lens camera.
- **What followed, under that approval:**
  - **D8** (section 6d) mapped the cost over a weak window's normal;
  - **H4,** the scan of a weak window's normal (section 6e), was built, tested (H4a), refined and tested again (H4b);
  - **H5,** warnings in the report (section 6f), was built and tested with W5 and R4 rerun (H5a).
- **H4b:**
  - none worse with or without hints;
  - 56 of 65 hinted runs at least equivalent;
  - two of the five hard cases right without hints;
  - strict criterion FAIL, on seed 1045 alone.
- **H5a:** answers without hints unchanged or better (118 of 120 unchanged, 2 better).
- **H1b** (W5 and R4 with hints): section 6g.
- **For a real experiment,** the conclusions are gathered in `2026-10-03-when-calibration-fails.md`.

## 1. Summary

- **Two windows at other angles (A1): no problem.**
  - 56 of 56 calibrations from scratch reached the global minimum, at α = 30°, 45°, 60°, 90°, 120°, 150° and 180°,
    at 0.1 and 1 px. Each window had two moderate-lens cameras.
  - The angle changes only the precision, through the angles between the cameras' lines of sight. The 3-D shape
    error is smallest through perpendicular walls (2.6 µm), 1.5× that at 30° and 150°, and 2.2× at 180°, where the
    cameras face each other.
- **The window-normal hint as built (H1a): FAIL.**
  - The second window search about the hint fixed seed 1034 at every hint error (0–30°), and 1045 with an exact
    hint. It did not fix 1001, 1031 or 3110, even with exact hints.
  - Some rough hints moved 1031 and 1045 into other wrong minima with a higher RMS.
  - Without hints, every answer was identical to the stored one.
- **Why it failed (D5, D6).**
  - **The hints arrive accurately.** In the frame where the windows are searched, an exact hint lands 0.4–6.5° from
    the true normal (one exception, 10°). The default guess is 1–37° off.
  - **The final fit would converge from them.** From the truth with one weak window tilted by up to 20°, it reached
    the global minimum in 144 of 144 runs on these rigs.
  - **The hint is lost in between.** The window searches score candidates while the other cameras are only roughly
    placed, and end 5–43° off even where a hint's candidate won. A window group calibrated on its own, with only its
    one or two long-lens cameras, lets the normal drift: on 3110 from 0.7° to 20°, on 1031 from 5.9° to 22°.
- **Proposed then (section 6), approved and built (section 6a): hold the hinted normals fixed while the start is
  built, and free them only in the final fit (H2).**
- **H2a: better, but FAIL.**
  - Exact hints now fix all five cases.
  - Rough hints (5–30°) are corrected when the weak window has two cameras: 22 of 24 runs.
  - They rarely are on windows behind a single long-lens camera: 5 of 36.
  - Three answers came out worse than without hints.
- **Why worse (D7): the comparison came too early.** The hint-held answer was better when compared, but the restarts
  then ran only on it; on the default answer they would have done better.
- **Proposed (section 6b, H3):**
  - each answer gets its own restarts before they are compared, so a hint cannot make the result worse;
  - optionally, the first version's search as a third start, since the two versions fix different cases (46 of 65
    hinted runs between them).
- **H3, built with the third start (section 6c):** no hinted run worse (0 of 65); 48 of 65 at least equivalent.
- **D8 (section 6d): the remaining wrong answers are separate minima,** 10–50° from the truth behind a ridge in the
  cost; the truth's valley is a few degrees wide. A grid scan of the weak window's normal finds it when the rest of the
  answer is sound.
- **H4 and H4b (section 6e): the calibration now scans every weak window's normal.**
  - A weak window is seen only by one or two cameras that all see the wand under less than 6°.
  - A released fit replaces the answer only when its capped cost is lower and it is another minimum or keeps the same
    observations.
  - **Without hints:** R4 3110 and W5 1001 are now right; W5 1031, 1034 and 1045 are not.
  - **With hints:** 56 of 65 runs at least equivalent, none worse; within 20°, 45 of 50, every miss on 1045.
- **H5 (section 6f): `calibration_report.txt` opens with warnings.** At 0.1 px every wrong answer had one (3 of 3), and
  only 3 of 47 right answers did.
- **Not fixed:** seed 1045, two cameras each alone behind its window with a long lens. Without hints, the other two
  misses (1031, 1034) also stay wrong. Each has two windows wrong at once, or fits pinned against a window.

## 2. A1: two windows at other angles

**The rigs** (`a1_window_angles.py`):
- window 1's normal +Z; window 2's turned about Y by α = 30°, 45°, 60°, 90°, 120°, 150° and 180° (walls of a 12-, 8-,
  6- or 4-sided tank);
- 12.7 mm acrylic between air and water; the 100 mm wand cloud's centre 150 mm behind each window's camera-side surface,
  its corners at least 79 mm behind the water side;
- two cameras behind each window, on either side of its normal; 1280 × 800; a 25 mm wand; 1,000 frames.

**(i) What the data determine** (Cramér–Rao at 0.1 px). One fixed rig per angle: 20° incidence (window 1's cameras
horizontally, window 2's vertically, as rig B), 600 mm from the window, f 3,550–4,320 px. The columns give the range
over the four cameras or two windows.

| α | lines of sight across the windows | σ(f) | windows' normals | cameras' distance to the window | 3-D shape of unseen points | one point's triangulation noise |
|---|---|---|---|---|---|---|
| 30° | 22° | 0.061–0.073 % | 0.018–0.019° | 0.66–0.82 mm | 3.9 µm | 30 µm |
| 45° | 32° | 0.057–0.072 % | 0.016° | 0.67–0.84 mm | 3.3 µm | 25 µm |
| 60° | 45° | 0.054–0.068 % | 0.015° | 0.68–0.80 mm | 2.8 µm | 23 µm |
| 90° | 72° | 0.053–0.060 % | 0.013–0.015° | 0.67–0.74 mm | 2.6 µm | 20 µm |
| 120° | 45° | 0.057–0.067 % | 0.013–0.016° | 0.74–0.79 mm | 2.8 µm | 22 µm |
| 150° | 22° | 0.066–0.076 % | 0.016–0.018° | 0.82–0.90 mm | 3.9 µm | 29 µm |
| 180° | 26° | 0.075–0.078 % | 0.025–0.026° | 1.03 mm | 5.6 µm | 37 µm |

- "Lines of sight across the windows": the smallest angle between a window-1 camera's line of sight to the cloud's
  centre and a window-2 camera's, as lines (cameras facing each other count as parallel).
- Precision follows that angle, as found for one window (`p_mixed_views.py`): best through perpendicular walls; the
  3-D shape error is 1.5× larger at 30° and 150°, and 2.2× at 180°, where the cameras face each other.

**(ii) From scratch.** Per angle, 4 rigs with each camera's incidence (15–25°), azimuth (a window's two cameras 180° ±
30° apart), distance (374–847 mm from the window), aim, roll and f (1,780–6,160 px) drawn at random; at 0.1 and 1 px.

- **56 of 56 at the same minimum** as the fit from the truth: every angle, both noise levels. The criteria (0.1 px all
  the same; 1 px all at least equivalent) pass.
- Their lines of sight across the windows were 3°–68° apart.
- The restarts improved none: no answer was wrong to begin with.
- Time: a median 100 s at 0.1 px and 129 s at 1 px (at most 141 s), with the machine fully loaded.

**Conclusion.** The angle between two windows makes no difference to finding the answer, on rigs with two moderate
lenses per window. It changes only the precision, through the angles between the lines of sight. Not covered: a
window seen by a single camera at these angles, long lenses, normals turned about more than one axis, three windows.

## 3. The window-normal hint as built

As planned (plan, section 2), in `plate_wand.py` (`hint_to_frame`, `hinted_normal`, `_merge_hinted`, and the window
searches of `bootstrap`, `bootstrap_groups`, `bootstrap_singles` and `place_single_window`):
- **Input:** `hints={"normals": [...]}`, one entry per camera: None, or the normal of its window in the camera's own
  frame (x along the image rows, y down the image, z along the optical axis; either sign). A camera turned 30° to the
  left of square-on, about its vertical axis, sees (sin 30°, 0, cos 30°). In the rig file of `plate_wand_io`:
  `"hints": {"normals": [[0.5, 0, 0.866], null, ...]}`.
- **Into the search's frame:** through the camera's pinhole start, whose axis is the chief ray in the water. The
  hint's angle θ from the camera's axis becomes θ_w with sin θ_w = (n_air / n_water) sin θ, at the same azimuth. This is
  the exact inverse of `small_angle_camera`: `test_window_normal_hints` checks it to 1e-12 on 40 random cameras.
- **A second search about the hinted normal** (a 20° cap, normals 4° apart; `HINT_CAP_DEG`, `HINT_STEP_DEG`, or
  `normal_cap_deg` and `normal_step_deg` in the hints), in every pass and for every window with a hinted camera. Its
  candidates are merged with the default search's by score. Every default candidate is kept, and a hint candidate that
  repeats one (normal within 1°, plane within 5 % of the camera's distance) is dropped.
- **No prior in the final fit.**
- **Reported:** `info["hints"]["normals_off_deg"]`, each hint's angle from the answer's normal, and one line in
  `calibration_report.txt`. Whether a hint's candidate won each search is in the bootstrap info (`best_from_hint`,
  `hint_won`).
- **Without hints** the code path is unchanged.

## 4. H1a: the hints on the failed cases (`h1_hints.py`)

Every camera hinted, its hint off by δ in a random direction (independent per camera; three draws per δ above 0).
Verdicts against the fit from the truth; "worse" means a lower verdict than without hints, or another wrong minimum with
a higher RMS.

| case (windows' cameras; f) | without hints | exact | 5° | 10° | 20° | 30° |
|---|---|---|---|---|---|---|
| 1034, 0.1 px ([2, 1, 1]; 4,400–37,200 px) | different | same | 3/3 same | 3/3 same | 3/3 same | 3/3 same |
| 1045, 0.1 px ([1, 1]; 12,700 and 29,100 px) | different | same | 0/3, 3 worse | 2/3 same, 1 worse | 2/3 same, 1 worse | 0/3, 1 worse |
| 1001, 0.1 px ([1, 1]; 12,300 and 24,500 px) | different | different | 0/3 | 1/3 equivalent, 2 "worse" | 0/3, 1 "worse" | 1/3 equivalent, 1 "worse" |
| 1031, 0.1 px ([2, 1]; 6,400–31,700 px) | different | different | 0/3, 2 worse | 0/3, 1 worse | 0/3 | 0/3, 1 worse |
| 3110, 1 px ([2, 2, 2]; 4,900–34,500 px) | different | different | 0/3 | 1/3 same | 1/3 same | 0/3 |

- **Criteria:** δ up to 10° all at least equivalent with the 0.1 px cases the same: FAIL. δ of 20–30° none worse:
  FAIL. Without hints the stored answers: PASS (all five identical).
- **1045's "worse"** answers are another wrong minimum: RMS 0.3796 against 0.3488 px without hints.
- **1031's** rise from 3.589 to 3.60–3.63 px. Most of its hinted runs end at 1.04–1.09 px instead, still wrong: window 1
  within about 1°, but window 2 42° off.
- **1001's "worse"** are not really worse: an RMS 0.002 % higher, in the same wrong minimum (window 2 11.1–11.4° off
  in every run but the two equivalent ones).
- **Time** (median of the hinted runs against the run without): 3110 362 → 450 s, 1034 374 → 291 s, 1045 400 → 392 s,
  1031 258 → 281 s, 1001 157 → 249 s.

## 5. Diagnosis (D5, D6)

**D5: where the hint is lost** (`d5_hints.py`, exact hints; `calibrate`'s start generation replayed, every start fitted).
- **The hints arrive accurately.** In the frame of each pinhole start, aligned with the truth by its wand points, an
  exact hint lands:
  - 0.4–4.2° from the true normal on the whole-rig pinhole starts;
  - 0.7–6.5° on the window groups' own pinhole starts;
  - the exception: 9.7–10.6° on 3110's window-2 group, whose pinhole start is the mirror solution at 2.7 px.

  The default guess (the camera's viewing direction) is 1.3–37.2° off.
- **Then the starts drift away from the truth:**
  - **1001:** all 8 starts have window 2 4.7–19.7° off, and none reaches the global minimum. The one nearest, at
    4.7°, also has camera 2's f 60 % too large and ends at 0.219 px.
  - **1045:** the four length-search starts are within 1.9–2.9° (window 1) and 0.6–0.9° (window 2). Their final fits
    end just short of the global minimum (0.08621–0.08634 px against 0.08615; max |z| 2.2–3.8), and the restarts
    finish the job. The four fixed-search starts have window 1 42° off. With hints 5° off, every start has a window
    37.9–42.5° off.
  - **1031:** window 1's group, its two long-lens cameras calibrated on their own, ends at 2.87 px (the hint was 5.9°
    off in its frame). Every start built from it has window 1 21.9° off.
  - **3110:** window 1's group (20,700 and 16,400 px; the hint 0.7° off in its frame) is calibrated on its own, and
    both starts built from the groups have window 1 20° off. The final fit keeps it (1.266172 px against 1.255149).

**D6: how close a start must be** (`d6_basin_failed.py`; W4's test on these rigs). From the truth with one weak window's
normal tilted by 2°, 5°, 10° and 20° (4 directions each; 9 windows), then the coarse fit and the final fit:
- **144 of 144 at the global minimum:**
  - 111 at the same minimum;
  - 32 of seed 1001's at the same RMS (0.08595 px; one at 0.08605) but not within 0.01σ in its flat valley (max |z|
    0.01–1.6), so counted equivalent;
  - one of 1045's at 20°, equivalent.
- So the final fit's basin around the truth is at least 20° wide for these windows, when the rest of the start is
  right.

**Conclusion.** These rigs are not hard because the right answer has a narrow basin. They are hard because the start
gets several things wrong at once:
- the window searches judge candidates against cameras that are still only roughly placed;
- a window group's own fit cannot pin its window with one or two long-lens cameras, and drifts.

The second search hands the hint to exactly those steps, and they override it.

## 6. Proposed revision after H1a: hint-held starts (approved 3 Oct; built and tested in 6a)

**The change** (`plate_wand.py`):
1. **`Layout` can hold a window's normal** (`held_normals`): its offset stays free.
2. **With hints, the start is built a second time, with every hinted window's normal held at its hint:**
   - the window search tries only the hinted normal, at its three offsets, and every fit in it keeps the normal; the
     offset and the cameras are fitted;
   - the window groups' own final fits and the coarse fit keep it too.
3. **Then the final fit frees everything,** from the hint-held start as from the default one.
4. **The hinted answer replaces the default answer only when its capped cost is lower.** This is the restarts' rule:
   every usable observation counts, each capped at the 5σ rejection limit. The restarts then run as now.
5. **The second search about the hint (this round's change) is removed.** It costs time and moved 1031 and 1045 into
   worse minima on some hints.

Kept from this round: the hint's input, its conversion (`hint_to_frame`, `hinted_normal`), the report line and the fast
tests. Without hints, nothing changes.

**Why it should work, and its limits:**
- D6 shows the final fit converges from a weak window 20° off when the rest of the start is right. With the normal
  held, the rest of the start is built around a window δ off, not 20–40° off.
- **Not shown yet:** that the rest of a held start comes out close enough. The cameras' f and distances, fitted with a
  normal δ off, carry some of that error. H2a tests it.
- A wrong hint costs time but not the answer: the default start still runs, and the costs decide.

**Tests:**
- **H2a:** H1a again: the five cases × δ = 0–30°, 70 calibrations. Same criteria.
- **H1b:** as planned, W5 and R4 with every camera hinted 0–10° off: no rig worse.
- **pytest:** a held normal does not move in the fit, while its offset does.
- About 45 min of computing.
- **Cost with hints, to be measured:** the start is built twice and the held start gets its own final fits. The held
  search itself is cheap: 3 candidates per window, against about 230 in this round's second search. I expect 30–60 %
  more time with hints.

**Stop point:** if H2a fails for hints up to 10° off, I stop and report. The cases would then need what the data
lack: a third camera behind the window, or a moderate lens.

## 6a. H2: hint-held starts, built and tested (after your approval, "Ok go ahead")

**Built as proposed** (`plate_wand.py`):
- **Held normals in the fit:** `Layout(held_normals=...)`; `solve`, `solve_obs` and `coarse_fit` pass them on. Every
  use of the window parameters handles a window whose normal is held and whose offset is free.
- **Held searches:** `window_search`, `place_window_vp` and `resect_window` take `hold_normal`. Only the hinted normal
  is tried, at its three offsets, and the fits free only the offset.
- **Hint-held starts:** `bootstrap`, `bootstrap_groups`, `bootstrap_singles` and `place_single_window` take `hold`:
  every hinted window's normal is held through its search, its group's own final fit and the coarse fit.
- **`calibrate`** builds the default start and, with hints, the hint-held start; each goes through stage 4. The
  hint-held answer replaces the default one when its capped cost is lower; then the restarts.
- **Removed:** the first version's second search (`_merge_hinted`).

**Tests:**
- `tests/test_plate_wand.py`: 9 fast checks pass. The hint test now checks that a held normal stays put while its
  offset moves. A new slow test (rig B, hints 10° off) reaches the global minimum, in 52 s.
- **H2a** (`h2a_hints_held.py`; H1a's cases and the same hints): **FAIL.**

| case | exact | 5° | 10° | 20° | 30° | the hint-held answer used | time without → with hints |
|---|---|---|---|---|---|---|---|
| 3110, 1 px ([2, 2, 2]) | same | 3/3 | 3/3 | 2/3 | 2/3 | 11 of 13 | 390 → 697 s |
| 1031, 0.1 px ([2, 1]) | same | 3/3 | 3/3 | 3/3 | 2/3, 1 worse | 13 of 13 | 317 → 312 s |
| 1034, 0.1 px ([2, 1, 1]) | same | 2/3 | 0/3, 1 worse | 0/3, 1 worse | 1/3 | 6 of 13 | 413 → 606 s |
| 1045, 0.1 px ([1, 1]) | same | 2/3 | 0/3 | 0/3 | 0/3 | 5 of 13 | 449 → 503 s |
| 1001, 0.1 px ([1, 1]) | equivalent | 0/3 | 0/3 | 0/3 | 0/3 | 1 of 13 | 176 → 184 s |

(Counts are runs at least equivalent to the fit from the truth; time is the median with hints.)

- **Exact hints fix all five cases** (H1a: two).
- **Rough hints are corrected when the weak window has two cameras:** 3110 and 1031, 22 of 24 runs at 5–30°.
- **Rough hints are rarely corrected when a window has a single long-lens camera:** 1001, 1045 and 1034, 5 of 36 runs
  at 5–30°.
  - Held 5–10° off, such a camera's f and position bend to fit the wrong normal, and the free fit then finds another
    minimum: on 1001 the hint-held answers end at 0.0882–2.3 px, against 0.0860 at the global minimum.
  - D6's wide basin (20°) assumed the rest of the start was right.
- **Three answers came out worse than without hints:** 1034 at 10° and 20°, 1031 at 30°.
- **The two versions are complementary,** with the same hints:
  - the first (a second search about the hint) fixed 1034 at every error, where the held start fails;
  - the held start fixed 3110 and 1031, where the first version failed;
  - either one: 46 of the 65 hinted runs, against 31 for H2 and 22 for H1a.

**D7: why three answers came out worse** (`d7_held_acceptance.py`): the comparison came too early.
- When `calibrate` compared the two answers, the hint-held one was the better one in all seven cases checked:
  - under every weighting (the default answer's, the held answer's, equal);
  - including the three that ended worse.
- But the restarts then ran only on the answer kept. Without hints they would have improved the default answer:
  - **1034:** the default answer was at 1.84 px. Its restarts would have reached 0.179 px (the stored answer). The
    hint-held answer, 0.453 px, was kept, and its restarts found nothing better.
  - **1031:** the default answer was at 5.29 px; its restarts would have reached 3.59 px. The hint-held answer
    (3.78 px) was kept.
- So the acceptance rule is fine, and the fix is to give each answer its own restarts before comparing them.

## 6b. Proposed revision: compare after the restarts (H3; approved, built and tested in 6c)

1. **The restarts run on the default answer and on each hinted answer,** each with the restarts' own seed, so the
   default answer's restarts give exactly the answer without hints.
2. **The answers are compared after their restarts,** on the capped cost with the default answer's weights (all
   weightings agreed in D7). The answer without hints is always a candidate, so a hint cannot make the result worse
   than without it.
3. **Optional: the first version's second search back, as a third start** (default, second search, held), each with
   its own stage 4 and restarts. It fixed 1034 at every hint error and 1045 at some.
   - With both, 46 of H2a's 65 hinted runs would have reached at least an equivalent answer, if the comparison picks
     the right one. It did in all seven D7 cases.
   - Cost with hints: about 2–2.5× the time without hints (three starts, each with its own final fits and restarts),
     against 1.0–1.8× now.
4. **Tests:** H3a (H2a's 65 hinted runs again; criteria: δ up to 10° at least equivalent, the 0.1 px cases the same;
   none worse at any δ), then H1b (W5 and R4 with hints 0–10° off, no rig worse).

**What hints could not do, after H2a:** neither version reliably refines a rough hint on a window seen by a single
long-lens camera (seeds 1001, 1045, 1034).
- With hints 5–30° off: the held start fixed 5 of these 36 runs; the first version 18, 12 of them on 1034; either one
  20.
- Seed 1001 reached an equivalent answer in only 3 of its 26 hinted runs, across both versions.

For such a window, a second camera, or a moderate lens, is worth more than a hint.

## 6c. H3: three start paths, each with its own restarts (built and tested after your approval)

**Built** (`plate_wand.py`; plan section 7.1):
- **The hint-searched start is back** as an option of the bootstraps (`widen`): every hinted window is searched a
  second time about its hint, and the candidates are merged with the default search's (`_merge_hinted`, the first
  version's code).
- **`calibrate` runs up to three paths:**
  - **default:** the hints ignored, exactly as without hints;
  - **held:** H2's start;
  - **searched:** the first version's start.
- **Each path's stage-4 answer gets its own restarts,** from the same seed. A path whose stage-4 answer is the same
  minimum as one already found skips them. "The same": `answers_apart` below 9, which is the residual difference on
  the observations both keep, in χ²; it bounds every parameter within 3σ.
- **The answer is the default one unless a hinted path's answer is another minimum** (`answers_apart` ≥ 9) **with a
  lower capped cost** (the default answer's weights).
- **Reported:** `info["hints"]["paths"]` (each path's RMS, whose minimum it repeated, its capped cost) and
  `["chosen"]`; one line in `calibration_report.txt`.

**Tests:**
- pytest: 11 fast checks pass. New: `test_hint_paths_helpers`, the merge and `answers_apart`. The slow test
  `test_hint_paths` follows the new bookkeeping.
- **H3a** (`h3a_hint_paths.py`; H2a's 70 runs and hints): **FAIL** on the strict criterion, with no run worse.

| case | exact | 5° | 10° | 20° | 30° | the answer from: default / held / searched | time without → with hints |
|---|---|---|---|---|---|---|---|
| 3110, 1 px ([2, 2, 2]) | same | 3/3 | 3/3 | 2/3 | 2/3 | 2 / 11 / 0 | 417 → 1146 s |
| 1034, 0.1 px ([2, 1, 1]) | same | 3/3 | 3/3 | 3/3 | 3/3 | 0 / 6 / 7 | 432 → 985 s |
| 1045, 0.1 px ([1, 1]) | same | 2/3 | 1/3 | 1/3 | 0/3 | 3 / 8 / 2 | 469 → 845 s |
| 1031, 0.1 px ([2, 1]) | same | 3/3 | 3/3 | 3/3 | 2/3 | 1 / 12 / 0 | 274 → 596 s |
| 1001, 0.1 px ([1, 1]) | equivalent | 3/3 | 2/3 | 0/3 | 1/3 | 4 / 7 / 2 | 143 → 432 s |

(Counts are runs at least equivalent to the fit from the truth; time is the median with hints.)

- **No hinted run came out worse than without hints** (0 of 65; H2a: 3). The three of H2a are now right, or the
  answer without hints:
  - 1034 at 10° and 20°: right, from the hint-searched path;
  - 1031 at 30°: the answer without hints.
- **48 of the 65 hinted runs are at least equivalent** (H2a 31, H1a 22). That is above the 46 the two versions reached
  separately, because each path's restarts now run on its own answer.
- **Without hints, every answer is the stored one.**
- **Two-camera windows** (3110, 1031): every run with hints up to 20° off is right, and 4 of 6 at 30°.
- **1034** (a two-camera window and two single long-lens windows): all 13 right, mostly from the hint-searched path.
- **The strict criterion fails on windows behind a single long-lens camera** (1045, 1001):
  - Wrong with hints up to 10° off: 1045 three times, 1001 once.
  - 1001's "equivalent" answers are, in practice, the global minimum: they have a slightly *lower* RMS than the
    truth-started fit (0.085945 against 0.085946 px), with |z| ≤ 0.4 and one observation more or fewer kept.
- **The remaining wrong answers lie near the truth:** 1045's three misses with hints up to 10° off are 2–6° from the
  true normals, at an RMS 0.4–7 % above the global minimum. So the cost seems to have other minima within a few
  degrees of the truth. D8 (below) maps them.
- **Time with hints:** 1.8–3.0× the time without (medians per case). The hint-searched path often reproduced the
  default minimum, and then skipped its restarts.

## 6d. D8: the cost over a weak window's normal (`d8_profile.py`)

**The question:** do the remaining misses come from a cost with several minima in the weak window's normal, and would
a scan of that normal find the right one, without any hint?

**The method:**
- On W5 seeds 1001, 1034 and 1045, from the answer without hints, the weak window's normal is held at each point and
  everything else is fitted (equal weights).
- The points:
  - along the great circle to the true normal, 1° apart;
  - on a grid about the answer's normal, 30° wide, 3° apart;
  - on a grid about the true normal, 12° wide, 1.5° apart.
- Each fit starts from a neighbour already fitted. The cheapest grid points are then released (everything free).

| case, window | answer's error | along the path to the truth | grid about the answer | grid about the truth |
|---|---|---|---|---|
| 1001, window 2 (one camera, 24,500 px) | 11.2° | rises to +49 % at 5–6°, then falls into the truth's valley | finds that valley; the 3 best points released: all at the global minimum (equivalent) | one valley; released from 0–3° off: equivalent |
| 1034, window 3 (one camera, 23,300 px) | 26.6° | rises sixfold, pinned against a window from 5° on | 228 of 317 points pinned; released: back to the answer | one valley; released from 0–4° off: the same minimum |
| 1045, window 1 (one camera, 29,100 px) | 48.8° | rises, pinned near 20°, degenerate beyond | the truth is outside the grid | released 1.5° off: equivalent |
| 1045, window 2 (one camera, 12,700 px) | 14.0° | rises 2.9-fold and stays | nearest the truth: a bad state (window 1 is still 49° off) | released 1.5° off: **another minimum** (RMS 0.0871 against 0.0862) |

- **The answers without hints are separate minima.** The cost along the way to the truth rises before it falls, so no
  descent from the answer leaves them. That is why the restarts, which perturb by 2σ, never did.
- **The truth's valley is narrow:** a few degrees wide (`p_held_fit_timing.py`). Held 3° off, the RMS is already 13 %
  higher on 1001, and 2.7× higher on 1045.
- **A scan of the normal finds it when the rest of the answer is sound** (1001: one wrong window).
- **The scan fails in two cases:**
  - when the fits held at wrong normals get pinned against a window: their bad state is carried from point to point
    (1034);
  - when two windows are wrong at once (1045).
- **1045's window 2 has other minima within 1.5–3° of the truth.** That explains H3a's near misses with hints (2–6°
  off).
- **A held fit takes 0.4–1.4 s** (all frames). A 317-point scan takes 3–10 min.

## 6e. H4: the scan of weak windows' normals (built after D8; plan section 7.3a)

**Built** (`plate_wand.py`):
- **`weak_windows`:** the windows seen only by one or two cameras that all see the wand under less than 6°
  (`view_angles_deg`, from the detections' spread and f). The report's warning uses the same rule.
- **`scan_window_normal`:**
  - D8's grid about the answer: 30° wide, 3° apart;
  - at each point the normal is held (the plane turned about its anchor, and moved back if that brought wand ends
    nearer the water-side surface), and one bundle adjustment fits everything else on the answer's observations;
  - each point starts from a neighbour already fitted;
  - the grid's local minima cheaper than the answer's own point are released, up to 3: stage 4 on fresh
    observations.
- **In `calibrate`,** after the start paths and their comparison, on rigs with several windows: every weak window is
  scanned in turn.
  - A released fit is accepted only on H3's terms: another minimum (`answers_apart` ≥ 9) with a lower capped cost.
  - It gets the restarts, and is compared again.
  - `scan_weak=False` turns the scan off. `info["scan"]` and a line in `calibration_report.txt` report it.

**The smoke test** (`h4_smoke.py`; no hints):

| case | before | the scan | after | the scan's time |
|---|---|---|---|---|
| 1001 | DIFFERENT (rms 0.087479) | window 2: 315 points; one minimum 15° from the answer, released and accepted | **equivalent** (rms 0.085945, \|z\| 0.15) | 191 s |
| 1034 | DIFFERENT | windows 1 and 3: no point cheaper than the answer | unchanged | 527 s |
| 1045 | DIFFERENT | window 1: no point cheaper | unchanged | 237 s |

(Three processes at once. The calibrations' total times are not comparable with H3a's, which ran 32 at once.)

So W5 seed 1001 is now right without any hint.

**H4a** (`h4a_scan.py`, H3a's 70 runs): **FAIL**, but a clear gain.

| case | without hints (H3a's) | exact | 5° | 10° | 20° | 30° | the scan improved | its time (median) |
|---|---|---|---|---|---|---|---|---|
| 3110, 1 px | **same** (DIFFERENT) | same | 3/3 | 3/3 | 3/3 | 3/3 | 3 of 14 | 59 min |
| 1034 | DIFFERENT (DIFFERENT) | same | 3/3 | 3/3 | 3/3 | 3/3 | 0 of 14 | 26 min |
| 1045 | DIFFERENT (DIFFERENT) | same | 2/3 | 1/3 | 1/3 | 0/3 | 0 of 14 | 13 min |
| 1031 | DIFFERENT (DIFFERENT) | same | 3/3 | 3/3 | 3/3 | 2/3 | 0 of 14 | 8 min |
| 1001 | **equivalent** (DIFFERENT) | equivalent | 3/3 | 3/3 | 2/3 | 3/3 | 6 of 14 | 4.5 min |

(32 calibrations at once, so the times are about twice those of a machine running one.)

- **Without hints, 2 of the 5 cases are now right:** 3110 and 1001.
- **With hints, 55 of 65 runs are at least equivalent** (H3a 48). 3110 is right at every hint error, and 1001 in 12
  of 13.
- **"Worse than without hints": 3 runs,** because the answers without hints got better:
  - two (3110) are the same answer: a lower RMS than the truth-started fit, |z| = 0.02, one observation more or
    fewer kept;
  - one is real (1001, 20°): the scan released a fit at the global minimum, but the rule refused it as "the same
    minimum" (χ² 4.9 apart, below 9). That window's normal has σ 2.1°, and its cost is far from quadratic.
- **The scans are slow on big rigs:** a median of 59 min on the 6-camera rig.
- **Refined after H4a (H4b, plan section 7.3b):**
  - a fit also replaces the answer when its capped cost is lower and it keeps exactly the same observations;
  - the scan's held fits use at most 300 frames.

**H4b** (`h4b_scan.py`, the same 70 runs, on the final code): **FAIL** on the strict criterion only, from seed 1045.

| case | without hints | exact | 5° | 10° | 20° | 30° | the scan improved | its time (median) |
|---|---|---|---|---|---|---|---|---|
| 3110, 1 px ([2, 2, 2]) | **same** | same | 3/3 | 3/3 | 3/3 | 3/3 | 3 of 14 | 9.5 min |
| 1034, 0.1 px ([2, 1, 1]) | DIFFERENT | same | 3/3 | 3/3 | 3/3 | 3/3 | 0 of 14 | 11 min |
| 1045, 0.1 px ([1, 1]) | DIFFERENT | same | 2/3 | 1/3 | 1/3 | 0/3 | 0 of 14 | 2.4 min |
| 1031, 0.1 px ([2, 1]) | DIFFERENT | same | 3/3 | 3/3 | 3/3 | 2/3 | 0 of 14 | 2.5 min |
| 1001, 0.1 px ([1, 1]) | **equivalent** | equivalent | 3/3 | 3/3 | 3/3 | 3/3 | 7 of 14 | 2.2 min |

(32 calibrations at once.)

- **Criteria:**
  - without hints, none worse than the stored answers: **PASS** (2 of 5 better);
  - with hints, none worse than without: **PASS**. Two runs on 3110 are the same answer with one observation more or
    fewer kept: a lower RMS than the fit from the truth, |z| = 0.02;
  - hints up to 10° off all at least equivalent, the 0.1 px cases the same: **FAIL**.
    - 1045 is wrong three times.
    - 1001's and 1031's "equivalent" answers have a lower RMS than the truth-started fit; they keep one observation
      more or fewer.
- **56 of the 65 hinted runs are at least equivalent** (H4a 55, H3a 48, H2a 31, H1a 22). Within 20°: 45 of 50, every
  miss on 1045.
- **The refined rule** fixed 1001 at 20° (H4a's real "worse" run). 1001 is now right in all 13 hinted runs.
- **The 300-frame scan is 4–6× faster** (3110: a median of 9.5 min against 59 min).
  - 68 of the 70 verdicts are H4a's.
  - The two that differ are better: 1001 at 20° (the refined rule), and 1034 at 30°, now the same minimum rather than
    equivalent.
  - Otherwise only one wrong answer moved: 1045 at 20° is still wrong, at a lower RMS (0.08639 against 0.08686 px).
  - The rest differ only within the same minimum (RMS changes below 2e-6 relative).
- **Time,** median per case (32 calibrations at once):
  - without hints: 267 s (1031, no weak window flagged) to 1,159 s (1034, two weak windows scanned);
  - with hints: 610–1,701 s.

**What is left:**
- **Without hints: 1031, 1034 and 1045.** Each has two windows wrong at once, or held fits that get pinned. A scan of
  one normal cannot leave these minima.
  - **1031:** its wrong answer's focal lengths hide its weak window (the views it computes are wider than 6°), so its
    normal is not scanned. Its residuals (3.6 px from 0.1 px detections) do raise the noise warning.
- **With hints, 1045:** two cameras in all, one per window, with 29,100 and 12,700 px lenses. It is right with an
  exact hint and in 2 of 3 runs at 5°, but wrong in most runs with hints 10° or more off. Its cost has minima 1.5–3°
  from the truth (D8).
- **This is the weakest rig the tests contain:** two cameras, each alone behind its own window, both with narrow
  views.

## 6f. H5: warnings in the report; H5a, W5 and R4 without hints on the final code

**Built** (`plate_wand_io.py`, `assess`; plan section 7.4):
- seven warnings, in `calibration_result.json` (`"warnings"`) and as a WARNINGS block at the top of
  `calibration_report.txt`;
- each names the window or camera, the evidence from the tests and what to do;
- codes: `weak_window`, `pinned`, `hint_far`, `same_way`, `noisy`, `few_frames`, `wand_length_fitted`;
- `weak_window` uses the scan's own rule (`plate_wand.weak_windows`) and says whether the scan found a better answer;
- `noisy` says both readings: noisy detections, or a wrong answer.

**H5a** (`h5a_warnings.py`): W5's 60 cases and R4's 60, without hints, on the final code.
- **Answers:** **PASS.**
  - 118 of 120 are exactly as stored;
  - the other 2 are better: W5 1001 and R4 3110, both fixed by the scan;
  - none is worse.
- **The scan** ran on 21 cases: a median of 9.4 min, at most 13 min, with 32 calibrations at once. It improved 2, and
  changed none of the other 19.
- **Wrong answers left:** 3 of 120, all at 0.1 px: W5 1031, 1034 and 1045.
- **The warnings against the verdicts:**

| detection noise | cases | wrong, with a warning | right, with a warning | the right answers' warnings |
|---|---|---|---|---|
| 0.1 px | 50 | 3 of 3 | 3 of 47 | `weak_window` |
| 0.5 px | 30 | — | 22 of 30 | mostly `noisy`; `weak_window`, `pinned` |
| 1 px | 40 | — | 37 of 40 | mostly `noisy`; `weak_window`, `pinned` |

- **At realistic noise (0.1 px) the warnings separate cleanly:**
  - every wrong answer has one (1034 and 1045 `weak_window`; 1031 `pinned` and `noisy`, at 3.3 px per coordinate);
  - only 3 of the 47 right answers do. One of them is 1001, whose warning says the scan found the better answer.
- **At 0.5–1 px almost every multi-window case gets `noisy`,** as it should: the detections are that noisy. No answer
  was wrong there.
- **Per warning, over all 120 cases:**
  - `weak_window`: 2 of the 3 wrong answers, 19 of the 117 right ones. Of the 21 cases it marks, 2 are wrong (10 %),
    down from 5 of 22 before the scan;
  - `pinned`: 1 wrong, 5 right. The right ones are two-camera windows at 1 px, pinned at their own best fit;
  - `noisy`: 1 wrong, 56 right;
  - `hint_far`, `same_way`, `few_frames`, `wand_length_fitted`: none. These conditions are not in W5 or R4; their
    pytest builds each one.
- **1031's wrong answer is not marked `weak_window`.** Its wrong focal lengths make its narrow view look wider than
  6°. Its residuals mark it.

## 6g. H1b: W5 and R4 with every camera hinted 0–10° off (`h1b_hints_regression.py`)

**The test:**
- W5's 60 cases and R4's 30 rigs at 0.5 and 1 px, with every camera hinted (each hint off by a random 0–10°), on the
  final code.
- Each answer is compared with the same case's answer without hints from H5a, on the same code.
- **Criterion:** no case worse.

**Result: PASS.** No case is worse, and three are better.

| set | the same minimum, without → with hints | at least equivalent, without → with hints |
|---|---|---|
| W5, 0.1 px (50) | 46 → 49 | 47 → **50** |
| W5, 1 px (10) | 7 → 7 | 10 → 10 |
| R4, 0.5 px (30) | 29 → 29 | 30 → 30 |
| R4, 1 px (30) | 29 → 29 | 30 → 30 |

- **Better: W5 1031, 1034 and 1045,** the three cases still wrong without hints, all now at the same minimum.
  - Their hints happened to be 1–8°, 4–10° and 0–1° off.
  - H4b shows 1045 needs hints that close: with hints 10° or more off, it is right in only 2 of 9 runs.
- **A hinted path gave the answer in 6 of 120 cases:** the three above, and three within their minimum, verdicts
  unchanged:
  - 3110 at 1 px: the same minimum, as without hints;
  - 3111 at 0.5 px: equivalent, as without, at a slightly lower RMS;
  - 1001: equivalent, as without, but at a slightly higher RMS (0.085956 against 0.085945 px, |z| 0.99 against
    0.11). It is the hint-held answer; the scan from it found nothing better.

  Everywhere else the hinted paths reached the default answer's minimum and stopped there.
- **Time with hints:** a median 2.55× the time without (1.2–3.4×), with 32 calibrations at once.

## 7. For a real experiment, now

- **Windows at any angle are fine** (A1). Polygonal tanks and prisms need no special handling. Perpendicular walls
  give the best precision, opposite walls (cameras facing each other) the worst, 2.2× the 3-D error.
- **Window-normal hints** (H4b):
  - give them for windows seen only by narrow-view cameras: they never made an answer worse;
  - within 20° they fixed every two-camera window tested;
  - a window behind a single long-lens camera needs them within about 5° (1045).
- **The scan of weak windows runs by itself.** It adds a few minutes per weak window.
- **Read the WARNINGS block of `calibration_report.txt` first.**
- **The rig advice stands, sharpened:**
  - two or more cameras behind every window, at least one seeing the wand under 6° or more (its positions spanning
    about f/9 pixels of the image diagonal);
  - never one camera per window with long lenses;
  - lines of sight 20° or more apart;
  - about 0.1 px detection noise.
- **The full list,** with how the report shows each condition: `2026-10-03-when-calibration-fails.md`.

## 8. Deviations from the plan

- **H1's part 1** (focal-length hints) was dropped at your request.
- **H1b ran only on the final design** (after H4b and H5a), not after H1a, H2a or H3a: each of those showed the design
  must change.
- **H1b's baseline is H5a's answers without hints,** on the same code, not the stored ones. The scan changed two
  answers without hints, and H1b must compare like with like. Its log's header still says "the stored answers".
- **Diagnoses added at the stop points:** D5 (`d5_hints.py`) and D6 (`d6_basin_failed.py`) after H1a, D7
  (`d7_held_acceptance.py`) after H2a, D8 (`d8_profile.py`) after H3a.
- **Steps added after your "fix it as much as possible",** each written into the plan (section 7) before it was
  built: H4 (the scan), H4b (its refinement), H5 (the warnings), and the studies `p_weak_windows.py` and
  `p_held_fit_timing.py`.
- **H4a's acceptance rule was changed after the test** (H4b): it refused a real improvement. H4b reran all 70 runs.
- **The criteria were not loosened.** H4b reports apart, for information, the runs that are "worse" or "equivalent"
  only because one observation more or fewer is kept, at a lower RMS than the truth-started fit.
- **The first version's code came back** in H3 as the hint-searched path (`widen`). `h1_hints.py` and `d5_hints.py`
  ran on the first version: rerun now, the first exercises the final design, and the second replays the default path
  only.
- **The A1 output's table header** is misaligned with its rows (cosmetic); the JSON holds the same numbers.

## 9. Files and reproducing

- `a1_window_angles.py` → `a1_window_angles_out.txt`, `.json`, `_console.txt` (4 min on 32 processes).
- `h1_hints.py` → `h1_hints_out.txt`, `.json`, `_console.txt` (14 min).
- `h1b_hints_regression.py` → `h1b_hints_regression_out.txt`, `.json`, `_console.txt` (run after H5a: it reads
  `h5a_warnings.json`).
- `d5_hints.py SEED NOISE DELTA DRAW` → `d5_hints_<seed>_<noise>_<delta>_<draw>_out.txt` (2–3 min each). Run for
  1001, 1031, 3110 and 1045 with exact hints, and for 1045 with hints 5° off.
- `d6_basin_failed.py` → `d6_basin_failed_out.txt`, `.json` (3 min).
- `h2a_hints_held.py` → `h2a_hints_held_out.txt`, `.json`, `_console.txt` (19 min).
- `d7_held_acceptance.py` → `d7_held_acceptance_out.txt`, `.json` (about 15 min).
- `h3a_hint_paths.py` → `h3a_hint_paths_out.txt`, `.json`, `_console.txt` (31 min).
- `d8_profile.py` → `d8_profile_out.txt`, `.json`, `_console.txt` (40 min, 4 processes).
- `p_weak_windows.py` → `p_weak_windows_out.txt`, `.json` (2 min, one process); `p_held_fit_timing.py` →
  `p_held_fit_timing_out.txt` (1 min).
- `h4_smoke.py` → `h4_smoke_out.txt`, `.json`, `h4_smoke_<seed>_log.txt` (calibrate's verbose log).
- `h4a_scan.py` → `h4a_scan_out.txt`, `.json`, `_console.txt` (85 min); `h4b_scan.py` → `h4b_scan_out.txt`, `.json`,
  `_console.txt` (44 min).
- `h5a_warnings.py` → `h5a_warnings_out.txt`, `.json`, `_console.txt` (18 min).
- `tests/test_plate_wand.py`: 12 fast checks and 4 slow ones; all 16 pass on the final code (366 s;
  `pytest_final_console.txt`). The hint and scan tests:
  - fast: `test_window_normal_hints`, `test_hint_paths_helpers`, `test_weak_windows_and_tilt`, `test_warnings`;
  - slow: `test_hint_paths` (rig B, hints 10° off), `test_scan_fixes_a_weak_window` (W5 1001).
