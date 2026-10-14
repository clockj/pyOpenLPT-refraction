# Case: tracer tracking of every run of 20260727 with the flat-window calibration

Your request (10 Oct 2026): "use the new calibrated cam file like what we use for 10Hz_r_b_1 to track all cases from
10Hz to 35Hz. save interval should be set to 200 frames; create a subfolder called Results_tracer inside each test
case; use imgFile_mask as the images for tracking."

**Status (13 Oct, 22:20):**
- **Your choice of 11 Oct (report section 12):** every run links short tracks within 5 voxels instead of 30
  (20–32 % more long tracks in the tests), and the dense runs have their own finder thresholds.
- **The queue:** `chain_batch.py` started `run_all.py` (pid 6780) at 23:28 on 11 Oct. Both 10 Hz runs and
  `15Hz_r_b_1`, the first dense run, are tracked with spacing 5 and delivered. `15Hz_r_b_1` took 33.6 h (12:32 on
  12 Oct to 22:05 on 13 Oct, 24.2 GB), steady from start to end; `15Hz_r_b_2` has tracked since 22:05. At that
  pace the nine runs left take until about 26 Oct.
- **10 Hz runs:**
  - **On the renumbered images** (frames 0–12,603), with spacing 5, in `<run>\Results_tracer`: `10Hz_r_b_2`
    (07:50–23:18 on 11 Oct, 60.0 GB) and `10Hz_r_b_1` (23:28–12:32, 36.1 GB). `10Hz_r_b_1`'s spacing-30 run
    (18:22–07:50 on 10–11 Oct) is kept as `10Hz_r_b_1\Results_tracer_spacing30` on X:.
  - **R1:** frames 0–255 repeat the old run exactly, and the drop at frame 256 is gone.
  - **The run on the old images** is kept as `10Hz_r_b_1\Results_tracer_old_images` on X:.
  - **Settings check:** the batch's settings reproduce the tested run byte for byte (S1).
- **15–35 Hz runs:** tracking (above). Their images are 6–15 times denser, and their particles move 3.3 times faster.
  - **With the 10 Hz settings,** IPR finds about 390,000 mostly ghost particles per frame, 2.5 min per frame, and
    almost no tracks survive (S2).
  - **With a higher threshold** (T1, T3; report sections 8–9), they take 25–30 h per run. After 255 frames they keep
    about 2,500 long tracks (35Hz_r_b_2 at 80) and 5,500 (35Hz_r_b_1 at 30), against 14,300 at 10 Hz.
  - **Tuning (your choice; U0–U6, report section 12):** short tracks linking within 5 voxels instead of 30
    (`convergence_phase.avg_spacing_vox`) gives 20–25 % more long tracks in both dense runs tested and 32 % at
    10 Hz, with the same quality and nearly the same cost.
  - **Configs:** written on D: and X: (`config.json`, `Results_tracer\camFile`, image lists), with spacing 5 and
    I5's second-row thresholds, and checked with the C++ reader.
- **Frame 256 was a copy of frame 255** in every camera of every run, in `imgFile`, `imgFile_bb` and
  `imgFile_mask` (D1, D2), until the renumbering. It cut about half the long tracks of a dense run, and 1.4 % at
  10 Hz.
  - **Cause (D3, report section 10):** pyOpenLPT's image preprocessing read the .cine videos in chunks of 256 and
    handed pycine the camera's frame number where pycine counts from 1.
  - **Effect:** from 257 on each file was one frame late, and the video's last frame was missing.
  - **Fixed in pyOpenLPT** (your choice; `tests/cine_frame_order`, all tests pass; committed as 1685d63).
  - **Renumbering on X:** your choice B2 (`renumber/`): drop the copy, shift the later files down, add the last
    frame. All 144 folders are done and checked (22:25). The masks' last frame (12,604) awaits your masking
    tool, so tracking stops at 12,603 (`frames_end`).
- **Report:** `2026-10-10-report.md`.

## Settings

Every run is tracked as `10Hz_r_b_1\stb_new_calib\new_cams` was (case `stb_20260727_10Hz_r_b_1`), over all its frames:

| setting | value |
|---|---|
| runs | `10Hz_r_b_1`, `10Hz_r_b_2`, `15Hz_r_b_1` … `35Hz_r_b_2` (12 runs, 12,605 frames each) |
| camFiles | `Results\T0\calib_plate_wand_ref_centre\camFile` (the flat-window wand calibration), copied to each run's `Results_tracer\camFile` |
| images | `imgFile_mask\cam0..3` (the masked images) |
| view volume, voxel | x [-81, 54], y [-42, 59], z [-36, 52.5] mm, 0.129 mm (test T0 of the 10 Hz case) |
| tracking settings | `10Hz_r_b_1\tracerConfig.txt`, with `convergence_phase.avg_spacing_vox` 5 instead of 30 (since 11 Oct; `config.json` `tracking_changes`) |
| finder threshold | 20 in the 10 Hz runs; in the dense runs I5's second row (`config.json` `thresholds`): 15 Hz 45 / 54, 20 Hz 42 / 39, 25 Hz 51 / 67, 30 Hz 65 / 34, 35 Hz 30 / 71 |
| frames, rate, threads | 0–12,603 (`frames_end`; the masks lack 12,604), 5000 Hz, 26 threads |
| save interval | 200 frames |
| output | `<run>\Results_tracer` |
| code | pyOpenLPT's current build (`build-py\Release`), the same as the GUI's |

## What is written where

**In each run's folder on X:** (`X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results\<run>\`)
- `config.json`: the run's tracking settings, as the GUI's Save writes them. The GUI opens it with the project, and
  can run it again. In `10Hz_r_b_1` it takes over from the old `config.txt`, as after any GUI Save (the old file stays).
- `imgFile_mask\cam0..3ImageNames.txt`: the image lists, as the GUI's preprocessing writes them. `10Hz_r_b_1` keeps
  its own (they name the same files).
- `Results_tracer\`: `camFile\` (the camFiles used) and, when the run is done, STB's output (`ConvergeTrack\`,
  `InitialTrack\`, `PredField_*.csv`) with the run's `log.txt`, `run_info.json`, `resources.csv` and
  `config_staging.json`.

**On D: while a run tracks** (`D:\Shijie Zhong\STB_staging\20260727\<run>\`): STB writes its output to a local disk,
because each save to X: takes about 12 min there (0.9 MB/s; frame 500 of the 10 Hz test took 14.9 min). The finished
run is copied to `Results_tracer` on X:.

## Files

| file | contents |
|---|---|
| `config.json` | the case: runs, camFiles, settings, folders |
| `case.py` | reads it; the paths of a run |
| `make_configs.py` | writes the configs: `--case` on X:, `--stage` on D: |
| `run_stb.py`, `parse_log.py` | one STB run with its log, run info and resource samples; a log as one record per frame (from the 10 Hz case) |
| `run_all.py` | the batch: every run in turn, delivered to X:; resumes an interrupted run from its last save; `--status` |
| `chain_batch.py` | starts the next batch when the running one has ended with every run delivered and every config checked |
| `renumber/` | the renumbering of the images on X: (B2): `renumber.py` (`--check`, `--undo`), its logs, checks and summaries; `mock_check.py` (R0) and `rerun_identical.py` (R1) with their results |
| `tests/` | the checks before the batch (I1, I2, I4, S1, S2, B1), the tests of the dense runs (T1–T4, I5), of the copied frame 256 (D1–D3) and the tuning tests (U0–U6), and the checks of the batch's runs (`batch_run_summary.py`; `track_quality.py` → `batch_quality.txt`), each with its results beside it |

## Running

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder.

    python tests/inventory.py                 # I1: the runs' images and lists (read only, ~2 min)
    python make_configs.py --stage            # the staging configs on D: (all runs)
    python make_configs.py --case RUN ...     # config.json, Results_tracer\camFile and image lists on X:
    python tests/smoke_test.py run NAME RUN F0 F1 [--threads T]   # a short test run (S1, S2)
    python tests/smoke_test.py identical NAME REF_DIR --out OUT   # S1: against a reference run
    python tests/batch_steps.py               # B1: the end-of-run check and delivery on a copy of S1
    python run_all.py RUN ...                 # the batch (start it detached; it logs to <staging>\batch_log.txt)
    python run_all.py --status                # progress and time left
    python tests/threshold_tests.py run|summary   # T1: thresholds on frames 0-50 (beside the batch)
    python tests/long_tests.py run|summary        # T3: 400 frames at the best thresholds
    python tests/frame256.py                      # D1: each camera's images around frame 256
    python tests/frame_copies.py RUN ...          # D2: copied frames at the chunk starts
    python tests/cine_frames.py RUN [--cams 0 1]  # D3: which video frame is in each imgFile file
    python tests/tune_tests.py run|summary        # U0-U6: tuning the tracking (beside the batch)
    python tests/step_tail.py                     # U1-U6: the fastest steps against the baselines
    python tests/config_change_check.py RUN ...   # a run's configs on disk against what make_configs.py writes
    python tests/track_quality.py RUN_DIR F [--max-save S]   # the long tracks' quality at a save frame F
    python tests/batch_run_summary.py RUN ...     # finished batch runs, whole, from their logs
    python chain_batch.py --after PID RUN ...     # the next batch after the running one ([--dry-run])

After a reboot, `python run_all.py` with all twelve runs resumes the unfinished run from its last save and tracks
the rest; runs with `DONE.json` in their staging folder are skipped.
