# Case: tracking run 20260727 / 10Hz_r_b_1 with the new flat-window calibration

Tracking (STB) of the bubble-shear run `10Hz_r_b_1` (4 Phantom cameras through two perpendicular acrylic walls of a
15 × 10 cm channel, 12,605 frames at 5 kHz).
- **Old camFiles:** the GUI's calibration after VSC, `camFile_VSC`.
- **New camFiles:** the flat-window wand calibration of case `plate_20260727_T0`, delivered in
  `Results\T0\calib_plate_wand_ref_centre`.

The question: how fast the tracking runs with the new camFiles, and whether it still removes many tracks in the
convergence phase.

- **Plan:** `2026-10-04-plan.md` (approved 4 Oct, done 5 Oct). **Report:** `2026-10-04-report.md`; its section 1 is
  the summary.
- **Result (5 Oct):**
  - **Speed:** the new camFiles track frames 0–500 in 27 % less time than the old ones.
  - **Tracks:** they keep 69 % more long tracks and give 77 % more tracked points.
  - **Removed tracks:** the many removals are mostly spurious short tracks; real long tracks end only when their
    particle is lost (a 3 px jump).
  - **The main cost is a slowdown that builds up inside the running STB process,** with either camFile set: the same
    frames run 6× faster after a restart from saved tracks.
  - **Matching:** it draws epipolar lines as straight lines; with the old camFiles that made 4-camera matches
    impossible.
- **Runs, configs and results on X:** `X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results\10Hz_r_b_1\stb_new_calib\`.
  Its `README.md` lists the folders: `new_cams` (T4), `old_cams` (T5), `ipr_probe` (T3).
- **The case's own files are not changed:** `config.txt`, `tracerConfig.txt`, `camFile_VSC`, `imgFile_mask`, `Results`
  and the logs.

Cameras: cam0 = C1 (image folder cam1), cam1 = C2, cam2 = C3, cam3 = C4. C1 and C2 look through one wall, C3 and C4
through the other.

## Running

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder. pyOpenLPT's local
build (`build-py\Release`) is put on the path by `case.py`.

    python tests/t0_frame_map.py        # T0: map the old view volume into the new frame -> new view volume, voxel size
    python make_configs.py              # write the run folders on X: (configs, image lists, camFile copies)
    python tests/t1_projection_cost.py  # T1: cost of the camFiles' projection tolerance
    python tests/t2_epipolar_lines.py   # T2: true matches against the straight epipolar lines of StereoMatch
    python tests/t3_ipr_probe.py        # T3: one IPR on raw frames 0 ... 4600, both camFile sets (~2 min)
    python run_stb.py X:\...\stb_new_calib\new_cams\config.txt   # T4: STB frames 0-1000, new camFiles (~2 h)
    python tests/t4c_restart.py --prepare   # T4c: copy T4's tracks at frame 500 (before T4 reaches 1000)
    python tests/t4c_restart.py --run       #      a fresh process from those tracks, frames 501-600 (~25 min)
    python tests/t4c_restart.py --compare   #      T4 against T4c
    python tests/t4d_heap_probe.py          # T4d: project / lineOfSight with a fragmented heap (one thread)
    python run_stb.py X:\...\stb_new_calib\old_cams\config.txt   # T5: frames 0-500, old camFiles (~3.3 h)
    python tests/t6_compare.py          # T6: the comparison (tables, figures)
    python tests/t6b_track_noise.py new old    # T6b: speed, position noise and linear-fit residuals of saved tracks
    python tests/t6c_deleted_tracks.py new     # T6c: why long tracks end, whether they continue; the short tracks

`run_stb.py` writes `log.txt`, `run_info.json` and `resources.csv` beside the config. `resources.csv` holds memory and
CPU use every 15 s. `parse_log.py LOG --csv OUT` turns any STB log into a per-frame table, old logs included.

## Code

| file | contents |
|---|---|
| `config.json` | every path and number of the case: folders, camFile sets, frames, the old view volume, the channel |
| `case.py` | reads the config; the C++ cameras with vectorised project, line of sight and triangulation |
| `make_configs.py` | writes `stb_new_calib\` on X: |
| `run_stb.py` | runs one config with the local build; log, run info, resource samples |
| `parse_log.py` | an STB log as one record per frame: step times, IPR rounds, track counters |
| `tests/` | T0–T6c, each test's results beside its script; `t4_per_frame.csv` and `t5_per_frame.csv` are the two runs' per-frame tables |
