# Plan: tuning the tracking for the dense runs (15–35 Hz)

11 October 2026. Your choice: tune the tracking first (report section 9, "Next steps"), before the ten dense runs are
queued. Status: **approved on 11 Oct; ran 01:52–05:49** (`tests/tune_tests.py run`, log `tests/u_queue.log`).
Results: section 12 of `2026-10-10-report.md`. Spacing 5 voxels passed in all three runs; U1 and U3 failed, so U4
did not run.

## 1. What the logs already show

Read from the 400-frame tests T3 and the 10 Hz batch, means per frame over frames 200–250. No new runs.

| | 10Hz_r_b_1 | 35Hz_r_b_2 at 80 | 35Hz_r_b_1 at 30 |
|---|---|---|---|
| IPR particles | 13,607 | 54,836 | 38,586 |
| active long tracks | 13,738 | 2,459 | 5,388 |
| short tracks promoted to long | 1,728 | 6,373 | 4,589 |
| long tracks failing the line test | 1,700 | 6,361 | 4,558 |
| established long tracks (7+ points) ending: lost + left the volume | 71 + 8 | 39 + 9 | 35 + 26 |
| loss rate of established long tracks | 0.57 % | 1.95 % | 1.12 % |
| promotions that last (net gain + losses) | 97 | 50 | 66 |
| level-off = lasting promotions / loss rate | 16,800 | 2,600 | 5,900 |

**What that says:**
- **Almost every promoted track fails the line test in the frame it is promoted.** Those are 4-point short tracks that
  do not follow one particle. So section 9's "turnover of 85 % and 260 % per frame" is this, not long tracks dying.
  Established long tracks end at 1.1–2.0 % per frame, 2–3.4 times the 10 Hz rate.
- **The level-off is predicted by the two rates** in all three runs. The 10 Hz run levels off at 15,800–17,300, and
  35Hz_r_b_2 at 2,000–2,500. So there are two levers: more promotions that last, and fewer losses.
- **Why so few promotions last (to be tested):** the convergence phase links each short track to the nearest IPR
  particle within 30 voxels (3.9 mm), widening to 5.7 mm when it finds none. That is 12 times the 0.33 mm step.
  - **It nearly always finds one:** at 35 Hz, 8–11 IPR particles lie within 3.9 mm of any point, and the nearest is
    about 1.6 mm away.
  - **Real particles get taken:** short tracks started on ghosts or on wrong particles can link a real particle.
    When such a track fails the line test, the real particle's track start is lost with it.
  - **The setting:** `30.0 # Avg Interparticle spacing. (vox) to identify neighbour tracks` in `tracerConfig.txt`
    (`convergence_phase.avg_spacing_vox`). It also sets the radius, three times larger, in which neighbouring long
    tracks give a short track its velocity.
  - **A correction to section 9:** it named the initial phase's search radius (10 voxels). That one is used only in
    frames 0–3.
- **Why established tracks end more often:**
  - **Predictions miss more at 35 Hz:** 34–48 % of them miss by more than the shake's starting width of 0.032 mm,
    against 21 % at 10 Hz (T4).
  - **The shake catches its particle less often from farther away:** from 0.032 mm it catches 71 %, and from 0.1 mm
    50 % (stb_improve E1c, 10 Hz images).
- **The 2D tolerance is not a good lever.** The plate cameras' epipolar lines are straight chords (stb_improve B2):
  - with exact pixels, 92 % of the 4-camera particles lie within 2 px of them, but only 46 % within 1 px;
  - so a lower tolerance would cut real particles along with the ghosts;
  - curved lines, as built for the tube, would remove that limit. That is a code change (section 6), not in this plan.

## 2. Tests

**Common to all tests:**
- **Build and settings:** the batch's build, and the batch's settings unless a test says otherwise. Only the settings
  change, nothing in the code.
- **Images:** the renumbered images.
- **Runs:** 12 threads beside the 10 Hz batch, one test after another, each into `<staging>\smoke\u<k>_<run>`. Saves at
  frame 200 and at the last frame.
- **Baseline:** T3's runs. Their frames 0–255 used images that the renumbering did not change, so they stay valid as
  the baseline. U0 checks this.

| test | run, threshold | frames | change | why |
|---|---|---|---|---|
| U0 | 35Hz_r_b_1, 30 | 0–30 | none | must repeat T3's log exactly, every counter of every frame; then T3 is the baseline |
| U1 | 35Hz_r_b_1, 30 | 0–250 | spacing 30 → 10 voxels (1.3 mm) | fewer false links |
| U2 | 35Hz_r_b_1, 30 | 0–250 | spacing 30 → 5 voxels (0.65 mm, twice the step) | fewer still |
| U3 | 35Hz_r_b_1, 30 | 0–250 | shake width 0.25 → 0.5 voxels (0.064 mm) | catch more of the predictions |
| U4 | 35Hz_r_b_1, 30 | 0–250 | the better spacing together with U3's width | only if U3 and U1 or U2 help |
| U5 | 35Hz_r_b_2, 80 | 0–250 | the chosen change(s) | the densest run, against T3 |
| U6 | 10Hz_r_b_1, 20 | 0–250 | the chosen change(s) | whether the 10 Hz runs would gain too, against the batch's log |

**Notes:**
- **U2 and the step:** with 5 voxels the search widens to 0.94 mm, still well above the step (median 0.32 mm, 90th
  percentile 0.39–0.42 mm).
- **Neighbour velocities:** the neighbour search shrinks with the spacing too, so a short track finds fewer neighbours
  for its velocity and more often predicts no motion. The measures below show what that costs.

## 3. Measures

**For each test against its baseline:**
- **Yield:** active long tracks at frames 50, 100, 150, 200 and 250.
- **Rates,** from the log's counters as in section 1, means over frames 150–250:
  - IPR particles and short tracks;
  - promotions and line-test failures;
  - lasting promotions, the loss rate, and the level-off they predict.
- **Cost:** seconds per frame, and hours per run.

**Quality of the active long tracks,** at frame 200 (T3 and the batch saved there too) and at the last frame:
- **Step:** median and 90th percentile.
- **Line-fit residual:** median and 90th percentile (T2's measures).
- **Neighbour check, new:**
  - **What:** each track's velocity against the median velocity of the other long tracks within 4 mm, for tracks
    with at least 3 such neighbours; median and 90th percentile of the relative difference.
  - **Why:** a track that follows a wrong particle or a ghost moves unlike its neighbours.

## 4. Decision rule

- **Better:** at least 5 % more long tracks at frame 250 than the baseline and a higher predicted level-off, with the
  quality kept:
  - step median within 3 % of the baseline;
  - line-fit residual and neighbour-check medians no more than 10 % above the baseline.
- **U4** runs only if U3 and at least one spacing test pass. Otherwise U5 and U6 take the best single change, or the
  campaign stops after U3 if none passes.
- **Stop rule, as in T1 and T3:** a test is stopped if its median frame over 10 frames takes more than 150 s.

## 5. Time, files and what comes after

- **Time:**
  - U0 takes about 6 min;
  - U1–U4 about 50 min each, at about 11 s per frame on 12 threads;
  - U5 about 40 min, and U6 about 20 min.
  - In all about 4.5 h, beside the batch, which runs 10Hz_r_b_2 until about 21:15 today. The batch slows a little,
    as it did during T1 and T3.
- **Files:**
  - `tests/tune_tests.py`: `run` and `summary`. It writes `tests/u_summary.txt` and `.json`, `tests/u_tracks.png` and
    `tests/u_queue.log`.
  - `tests/track_quality.py`: the quality measures, written to `tests/u_quality.txt` and `.json`.
  - The results go to a new section 12 of `2026-10-10-report.md`, which also corrects section 9's turnover and radius.
- **After the tests, your choices:**
  - **The settings for the ten dense runs.** Their staging configs are then rewritten (they also need `frames_end`
    12,603), and they are queued after 10Hz_r_b_2.
  - **The 10 Hz runs,** if U6 shows that they would gain too.

## 6. Not in this plan

- **P: curved epipolar lines for the plate cameras.**
  - **The change:** the tube cameras already have curved lines (7 Oct). The plate cameras would turn them on: one
    camera method, plus its projection along the line of sight.
  - **The gain:** the 2D tolerance could then drop to about 1 px, which cuts ghosts and keeps the real particles.
  - **What it needs:** your approval of a code change, a separate build (the batch's build stays untouched), and its
    own tests: matching with exact pixels, speed, and a tracking run.
- **The line-test limit (0.05 mm):** a constant in the code, not a setting.
