# Case: four cameras through two flat tank walls, run 20260727 / T0

Wand calibration of four Phantom cameras (2048 × 1952 px, 13.5 µm pixels, 180 mm lenses, in air) that look through
two perpendicular acrylic walls (0.5 in) of a water tank: C1 above C2 through one wall, C3 above C4 through the other.
The wand is 10 mm long, with a small ball (end A) and a large ball (end B), backlit. The model is pyOpenLPT's
flat-window camera (PINPLATE), fitted by its wand calibration `plate_wand.calibrate` from the detections alone, with
no camera, wall or focal-length input.

- **Plan:** `2026-10-03-plan.md`. **Report:** `2026-10-03-report.md`; its section 1 is the summary.
- **The report as a page:** `2026-10-03-report.html`, with four figures and the wand volume's table:
  - the rig in plan view;
  - the rig in 3-D, interactive, with the cameras' refracted fields of view;
  - every fit, right and wrong;
  - the GUI comparison.

  It is built by `make_report_page.py` from `page/report_template.html`, `tests/fits_summary.json`,
  `tests/rig_3d.json` and `tests/t7_vs_gui.json`.
  - It is also at https://claude.ai/artifact/WpyBdf4aid6EATqpxsZNQN, private until you share it.
- **Result (3 Oct):** the calibration in `Results\T0\calib_plate_wand\`. Every frame of the detection table,
  triangulated with it, gives:
  - a reprojection error of 0.215 px (median) and a wand length error of 0.0067 mm (median);
  - 6× and 8× lower than the GUI's calib_3.

  The cameras' f are 18,750–21,940 px, and k1 is fitted. From scratch the calibration can end in a wrong minimum on
  this rig, so the delivered fit was started from a verified one (report, sections 4 and 6).
- **Wand images:** `X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\T0\cam1` … `cam4`, 12,605 frames each.
- **Detections, inputs:** the user's circle finder (`circle_results_table_new.csv`: both balls in all four cameras in
  10,936 frames) and the GUI's filtered sets behind its calibrations (`wandpoints_filtered.csv`,
  `wandpoints_filtered_2.csv`), all in `X:\...\20260727\Results\T0`.
- **Earlier GUI calibrations,** kept for comparison: `Results\T0\calib`, `calib_2`, `calib_3`.
- **Outputs:** `Results\T0\calib_plate_wand\` (its own `README.md` lists them).

Cameras: cam0 = C1 (image folder cam1), cam1 = C2 (cam2), cam2 = C3 (cam3), cam3 = C4 (cam4); the CSVs and the
camFiles number them 0-3.

## Running

Use the `refract` environment (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder:

    # T2, the delivered result: every frame, k1 fitted, started from the first half's fit (5 min); writes the outputs
    python run_calibration.py --fit-k1 --start tests/states/t3k_h1.pkl --state tests/states/t2_all.pkl --log tests/t2_calibration_console.txt
    python run_calibration.py --fit-k1 --every 5 --offset 0 --out none --state tests/states/x.pkl  # a subset, fit only
    python run_calibration.py --readme-only            # rewrite the output folder's README.md
    python evaluate.py calib_3                         # any calibration judged on the detections
    python tests/run_batch.py t3 t3k t5 t6             # subset fits in parallel (groups: t3 t3k t5 t6 t6k t5h2 d3)
    python tests/t3_repeatability.py t3k               # each test writes its results beside itself

Options of `run_calibration.py`:
- `--fit-k1`: fit radial distortion; needed here.
- `--start STATE`: start from a saved fit's cameras and walls.
- `--hints photo` or `--hints gui:calib_3`: window-normal hints.
- `--edge-margin PX`: drop detections whose ball crosses the image border.
- `--n-water`, `--thickness`: override the water index or the wall thickness.
- `--first`, `--last`, `--every`, `--offset`: frame subsets.

The state `tests/states/t3k_h1.pkl` is not tracked by git. To rebuild it:

    python run_calibration.py --first 0.5 --fit-k1 --out none --state tests/states/t3k_h1.pkl   # 45 min

`config.json` holds every number of the case: paths, walls, indices, wand, cameras, the photo hints.

## Code

| file | does |
|---|---|
| `case.py` | Reads the config, the wand CSVs, the images and the GUI's camFiles; describes a rig independently of its frame (f, distances and angles to the walls, baselines); saves and reloads fits. |
| `run_calibration.py` | The pipeline: `plate_wand.calibrate` on a detection set or a subset of its frames, then the outputs (`plate_wand_io.write_result`, `evaluation.txt`, `README.md`). |
| `evaluate.py` | Judges any calibration on the detections: free triangulation of every frame, reprojection errors and wand lengths. |
| `make_report_page.py`, `page/report_template.html` | The report page (`2026-10-03-report.html`; `--fragment PATH` for publishing). Its fit table comes from `tests/fits_summary.py`, which summarises every saved fit into `tests/fits_summary.json`. |
| `tests/` | **Tests:** T1 (detections), T2 (`t2_*`: the delivered fit, its from-scratch run, the camFiles' round trip), T3 (repeatability over frame subsets and halves, k1 held and fitted), T4 (held-out frames), T5 (hints), T6 (water index, wall thickness, k1), T7 (against the GUI's calibrations). **Diagnoses:** D1 (how well the data fix each wall's position), D2 (the frames that pin a wall), D3 (the border-cut detections dropped). `run_batch.py` runs the subset fits; their states go to `tests/states/` (not tracked by git: regenerable), their consoles to `tests/logs/`. |

The case found two library bugs on real data and fixed them on 3 Oct, with your approval:
- a crash in `cylinder_wand.affine_factorization`, stage 1 of both wand calibrations;
- a rotation check in `plate_model.to_cpp` too strict for rounding.

Each has a regression test in `tests/test_plate_wand.py`. Records: `tests/fix_*_console.txt` and the report, sections 3
and 4.
