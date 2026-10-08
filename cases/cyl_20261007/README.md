# Case: tracking tracers through the tube, run 20261007

The recordings of 7 October 2026 through the water-filled acrylic tube, with the two Phantom VEO 710L cameras of
6 Oct:
- `T0`: wand calibration, 15,001 frames at 100 frames/s;
- `Planes`: the dot board at three positions;
- `T1`: tracers, 1,001 frames at 4,000 frames/s, backlit; each particle images as a short horizontal or vertical
  line, because of the tube's astigmatism.

- **Plan:** `2026-10-07-plan.md`. **Status: all phases done (7 Oct).**
- **Final report: `2026-10-07-report.md`.** Phase reports: `2026-10-07-phaseC-report.md` (the calibration),
  `2026-10-07-lines-report.md` (T1's images, preprocessing, the detector's prototype).
- **Raw images:** `X:\Shijie Zhong\Refraction\Raw\20261007\{T0,Planes,T1}`.
- **Results:** `X:\Shijie Zhong\Refraction\Processed\20261007\{T0,Planes,T1}`, each with a `README.md`. Use
  `T0\camFile_track\` for tracking; T1's tracks are in `T1\Results\`.
- **`p0_evidence/`:** the scratch probes behind the plan, with their outputs and figures.

The library work (fast projection, image statuses, curved epipolar lines, the OTF segment model, the line detector,
the depth cue) has its tests and report in `tests/cylinder_stb/`.

## Running Phase C (the calibration)

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder, with
`CASE_PYOPENLPT_DIR` set to a copy of `build-py/Release/pyopenlpt*.pyd` (`data/pyd/`, git-ignored), so that
pyOpenLPT can be rebuilt while the case runs:

    python tests/p0a_files.py              # P0a: files and headers
    python tests/p0d_wand_survey.py        # P0d: T0, every 50th frame through the detector
    python tests/p0e_planes_survey.py      # P0e: the board's dots and their blur
    python detect.py                       # C1: the balls in all frames (28 min over the share)
    python calibrate.py                    # C1: the calibration (6 min; two starts, see config.json)
    python tests/c1_gui_csv.py             # C1: wand_points.csv loads in the GUI
    python tests/c1_residuals.py           # C1: residual maps, wand-length trends
    python tests/c1_references.py          # C1: against 30 Sep's board and 6 Oct's calibrations
    python tests/c1x_cross_runs.py         # C1-X: 6 Oct T2 and this run on each other's frames
    python tests/c1d_sync.py               # C1-D: the cameras' time offset
    python tests/c1t_variants.py           # C1-T: halves and quarters
    python tests/c1t_held_out.py           # C1-T: the parts' cameras on the frames they did not fit
    python tests/c2a_board_check.py        # C2a: the board triangulated: flatness, pitch, steps
    python tests/c2b_f_board.py            # C2b: each camera fitted to the board alone
    python tests/c2b_valley_board.py       # C2b: the board along the wand's f / distance valley
    python tests/c3_hold_f.py scan         # C3: cam2's f held at 26,000-30,000 px
    python tests/c3_hold_f.py fit 29035 48306 c3_held_f
    python tests/c3_hold_f.py parts 29035 48306
    python tests/c4_track_camfiles.py      # C4: the tracking camFiles (tolerance 1e-9)

## Running T1's steps (the lines report and Phase 6)

The T1 scripts use the library build `build-stb/p4c/out` (`t1_setup.BUILD`; `CYL_STB_BUILD` overrides it) and the
local copies in `data/` (git-ignored):

    python tests/p0b_copy_t1.py            # the local copy of T1 (2 x 1,001 frames, 2 GB)
    python tests/p0b_t1_survey.py          # P0b: the images; the backgrounds (data/T1bg)
    python tests/p0c_t1_motion.py          # P0c: the motion; the frame step (4)
    python tests/l1_preprocess.py          # L1: preprocessing checks
    python tests/l2_selftest.py            # L2: the detector's mathematics
    python tests/l2_synthetic.py           # L2: the prototype against today's finder, synthetic
    python tests/l2_t1.py                  # L2: the prototype on T1
    python tests/s1_preprocess.py --to-x   # S1: every 4th frame preprocessed, local and on X:
    python tests/t1a_epipolar_offset.py    # T1-a: offsets across T0's epipolar curves
    python tests/s2_shape_fit.py           # S2: the shape model (45 min; --reuse refits only)
    python tests/s2b_planes_check.py       # S2b: the model against the Planes' dot edges
    python tests/s3_track.py final --tol2d 1.5      # S3: STB on all 251 frames
    python tests/s3_track.py control --tol2d 1.5    # T1-e: cam2 shifted by 100 steps, every track a ghost
    python tests/s3_track.py today                  # today's code on frames 0-60
    python tests/t1_checks.py              # T1-a to T1-d (--run s3_today for today's run)
    python tests/t1e_ghost_control.py      # T1-e
    python tests/s3_track.py copy-to-x     # the final run, the model and the checks to T1\ on X:

| code | does |
|---|---|
| `config.json` | every number of the case: paths, cameras, tube, wand, detector, selection, calibration (`n_final` 2, with the reason), the Planes, T1 |
| `case.py`, `detect.py`, `calibrate.py` | the 6 Oct T2 case's wand code, copied (paths in `config.json`) |
| `planes.py` | the board: the 30 Sep case's dot detector (loaded unchanged) after masking cam2's dark band, lattice indexing, pairing the two cameras' dots, the outline blur |
| `lines.py` | the line detector's prototype (ported to C++ as `LineFinder`), T1's preprocessing, rendering, scoring |
| `t1.py`, `t1_setup.py` | T1's files; the STB config, view volume and noise for the Phase 6 runs |
| `tests/` | the scripts above, each writing `<name>_out.txt` (and `.json`, figures) beside itself; `before_*/` hold the outputs of earlier versions that the reports cite |
| `data/` (git-ignored) | local copies (T1 raw and preprocessed frames, the tracking camFiles), S2's pairs and model, the STB runs |
| `wrong_minimum_nfinal6/` | the first calibration run (six starts), whose wrong minimum won |
