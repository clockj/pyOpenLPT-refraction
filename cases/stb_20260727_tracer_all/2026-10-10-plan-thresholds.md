# Threshold tests for the 15–35 Hz runs: plan

10 October 2026. **Status: approved on 10 Oct and done the same day; results in section 8 of the report.** You chose
"Test thresholds first". The other options were the same settings anyway, or only the 10 Hz runs for now. The batch
report is `2026-10-10-report.md`; its section 6 lists the options.

## Question

Which tracer finder threshold makes the dense runs trackable at a cost close to the 10 Hz runs'? "Trackable" means the
long tracks survive the convergence phase and grow, as in 10Hz_r_b_1, rather than collapsing as in S2.

## Tests (T1)

Every test runs frames 0–50 with the batch's settings: the same camFiles, view volume, voxel, tracer settings and save
interval. Only the finder's `intensity_threshold` changes (tracerConfig: 20). The tests use 12 threads, beside the
running batch.

| test | run | thresholds | why |
|---|---|---|---|
| T1a | `10Hz_r_b_1` | 20 | the reference under the same conditions (12 threads, beside the batch), so speeds compare |
| T1b | `35Hz_r_b_2` | 100, 80, 60 | the densest run: at 20 it has 15.4× the detections of 10Hz_r_b_1; at 100, 80 and 60, 1.5×, 2.4× and 4.3× (I4) |
| T1c | `35Hz_r_b_1` | 50, 40, 30 | the lightest dense run (5.8× at 20): 1.1×, 1.7× and 3.0×. Its cameras 1 and 3 are dim, so a single threshold costs them most |

- **Order:** T1a first, then each run from its highest threshold down.
- **Stopping rule:** a test whose frames take over 150 s each is stopped after 10 frames, and its partial result is
  kept.

## What is reported

For each test:
- detections per camera in IPR's first round, and the objects IPR finds per frame;
- the time per frame, and its ratio to T1a. The projected time per frame alone at 26 threads is that ratio × 2.48 s
  (10Hz_r_b_1 alone), and from it the time per run;
- active long and short tracks at the end of each frame. A figure compares them with T1a, which reaches about 6,700
  long tracks by frame 50.

Then a proposal: one threshold per dense run, chosen from I4's detection counts so that each run's detections are
about those of the threshold that worked. That is for you to choose; the batch takes it as a config change for those
runs only.

## Files

- **Scripts:** `tests/smoke_test.py run ... --threshold T` and `tests/threshold_tests.py`, which queues the tests,
  summarises them and draws the figure.
- **Results:** `tests/t1_*`.
- **The runs themselves:** `D:\Shijie Zhong\STB_staging\20260727\smoke\t1_*`.

## T3: 400 frames at the best thresholds (approved 10 Oct; done the same day, results in section 9 of the report)

You chose "400-frame test first" after T1 (report section 8). The other options were to queue all ten runs now, to
tune the tracking for fast flow, or to stop at the 10 Hz runs.
- **Runs:** `35Hz_r_b_2` at 80 and `35Hz_r_b_1` at 30, the best of T1, one after the other on 12 threads beside the
  batch, frames 0–400. Saves come every 200 frames, as in the batch.
- **Reference:** the batch's own run of `10Hz_r_b_1` (its log, frames 0–400). Speeds compare with T1a.
- **Reported:** active long tracks at frames 50, 100, 200, 300 and 400, and the gain per frame over frames 300–400,
  which shows whether they level off. Also the time per frame and hours per run, as in T1, and the motion and
  linear-fit residual of the tracks alive at frame 400.
- **Files:** `tests/long_tests.py` → `tests/t3_summary.txt`, `t3_summary.json`, `t3_tracks.png`, `t3_queue.log`; the
  runs in `D:\Shijie Zhong\STB_staging\20260727\smoke\t3_*`.
