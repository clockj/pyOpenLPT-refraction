# Case: tracking tracers through the tube with three cameras, run 20261008

The recordings of 8 October 2026 through the water-filled acrylic tube, with three cameras:
- **cam1 and cam2:** the two Phantom VEO 710L of 6–7 Oct (1280 × 800 px, 20 µm pixels), moved and refocused since
  7 Oct;
- **cam3:** a Phantom v2640 (2048 × 1952 px, 13.5 µm pixels);
- **lenses:** 180 mm on every camera, at roughly the same distance from the tube.

The recordings:
- **`T0`:** wand calibration, 100 frames/s. cams 1–2 saved images 0–13000; **cam3 saved images 2396–13000**, so its
  `frame000000.tif` is image 2396 (the `.chd` headers' FirstImageNo).
- **`T1`:** tracers, 1,001 frames at 4,000 frames/s. All three start at image 0. Tracked: the first 250 frames (your
  choice, 8 Oct).

- **Plan:** `2026-10-08-plan.md`. **Status:** approved 8 Oct ("go on and generate results and reports, update
  gui"; the wand is 6–7 Oct's; T0 and T1 share the cameras' settings; cams 1–2 were moved and refocused).
- **Final report:** `2026-10-08-report.md`; the HTML report (3-D tracks, figures):
  https://claude.ai/artifact/TDj26gBjtZskKriczxYZ6G.
- **Raw images:** `X:\Shijie Zhong\Refraction\Raw\20261008\{T0,T1}`.
- **Results:** `X:\Shijie Zhong\Refraction\Processed\20261008\{T0,T1}`, each with a `README.md`. Use
  `T0\camFile_track\` for tracking; T1's tracks are in `T1\Results\`.
- **The GUI's part** (a pixel size per camera, cameras lined up by their Phantom headers) has its tests and report in
  `tests/wand_gui_pixel_size/`.

## Running it

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder. Frames are counted
by image number from the common trigger (`case.py`). Every script writes `<name>_out.txt` (and `.json`, figures)
beside itself.

Phase 0 (read-only):

    python tests/p0a_files.py              # files and headers (cam3's FirstImageNo 2396)
    python tests/p0d_wand_survey.py scan   # the balls' sizes and the rod opening, per camera
    python tests/p0d_wand_survey.py survey 50
    python tests/p0b_t1_survey.py          # T1's images per camera (crops: tests/p0b_t1_crops.png)

Phase C, the calibration (results on X: `T0\`):

    python detect.py                       # C1: the balls in every image of every camera (about 2 h over the share)
    python calibrate.py                    # C1: the three-camera fit
    python tests/c1_gui_csv.py             # wand_points.csv loads in the GUI
    python tests/c4_track_camfiles.py      # C4: the tracking camFiles (tolerance 1e-9)
    python tests/c1_residuals.py           # residual maps, wand-length trends
    python tests/c1d_sync.py               # C1-D: the cameras' time offsets (cam3's 2396)
    python tests/c1h_third_camera.py       # C1-H: each camera predicted from the other two
    python tests/c1t_variants.py           # C1-T, C1-P: halves, quarters, and the three camera pairs

Phase T, T1's first 250 frames (local copies in `data/`, git-ignored; results on X: `T1\`):

    python tests/t_copy_t1.py              # the local copy of frames 0-249
    python tests/p0c_t1_motion.py          # P0c: the motion per frame (step 1); the p90 backgrounds
    python tests/s1_preprocess.py --step 1 # S1: bright particles on black, 15 / 15 / 23 px medians
    python tests/s2_shape_fit.py           # S2: the shape model from three-camera matches (20 min)
    python tests/s0_detector.py            # S0: the line detector against the standard finder
    python tests/s0_detector.py --k 6 5 4 --frames 20 70 120 170 220 --tag k    # S0: the threshold k_detect
    python tests/s1b_crop_cam3.py          # S1b: cam3's frames and camFile cropped to rows 360-1575 (speed)
    set CYL_T1_PRE=T1pre_c3crop            # (bash: export) the runs read the cropped cam3
    set CYL_T1_CAMS=cams_T0_c3crop
    python tests/s3_track.py final --tol2d 2.0 --shape-tol 0.8 0.2 --reduced 0 --loops 2 --tag c3
    python tests/s3_track.py control --tol2d 2.0 --shape-tol 0.8 0.2 --reduced 0 --loops 2 --tag c3
    python tests/s3_track.py pair12 --frames 0 149 --tol2d 2.0 --shape-tol 0.8 0.2 --loops 2 --tag c3
    python tests/s3_track.py control12 --frames 0 149 --shift 100 0 --tol2d 2.0 --shape-tol 0.8 0.2 --loops 2 --tag c3
    python tests/t1_checks.py --run s3_final_c3
    python tests/t1e_ghost_control.py --final s3_final_c3 --control s3_control_c3
    python tests/t1e_ghost_control.py --final s3_pair12_c3 --control s3_control12_c3
    python tests/t1_compare.py --final s3_final_c3 --control s3_control_c3 --pair s3_pair12_c3 --control12 s3_control12_c3
    python tests/s3_track.py copy-to-x --tag c3

The run settings, and why they differ from the plan's, are in the plan's section 6: a 2.0 px tolerance (S2: 89 % of
the true matches within it), the depth cue at 0.8 px / 20 %, every match in all three cameras, two IPR loops, and
cam3 cropped (the line detector's time grows with the image's area). OMP_NUM_THREADS sets each run's threads.

| code | does |
|---|---|
| `config.json` | every number of the case: paths, cameras (sizes, pixel sizes, first image numbers), tube, wand, detector (per camera), selection, calibration, T1 |
| `case.py`, `detect.py`, `calibrate.py` | the 7 Oct case's wand code, for cameras that differ |
| `lines.py` | 7 Oct's line detector prototype and T1's preprocessing (copied unchanged) |
| `t1.py`, `t1_setup.py` | T1's files; the STB config, view volume and noise, three cameras (CYL_T1_PRE, CYL_T1_CAMS choose the frames and the camFiles) |
| `tests/` | the scripts above |
| `data/` (git-ignored) | local copies (T1 raw and preprocessed frames, the tracking camFiles), S2's triplets and model, the STB runs |
