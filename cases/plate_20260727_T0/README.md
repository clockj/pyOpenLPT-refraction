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
- **Use this one (4 Oct, your choice): `Results\T0\calib_plate_wand_ref_centre\`.** It holds your channel: 15 × 10
  cm inside, C1 and C2 on the long side.
  - **Fitted:** k1, with each lens's principal point held at the image centre.
  - **Held:** the channel, and C1 and C2 aiming at the middle of their side.
  - **What changed:** the first calibration put the C3/C4 wall about 9 cm too close. The new one puts the wand inside
    the channel, as accurately as the first: 0.218 px and 0.0068 mm median on every frame.
  - **The variant with the principal point fitted** is in `Results\T0\calib_plate_wand_ref\` (0.210 px, 0.0065 mm).
  - **Plan and report:** `2026-10-03-walls-plan.md`, `2026-10-03-walls-report.md` (section 6b). Tests W0–W4.
- **Result (3 Oct), the first calibration:** in `Results\T0\calib_plate_wand\`. Every frame of the detection table,
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

    # T2, the first result: every frame, k1 fitted, started from the first half's fit (5 min); writes calib_plate_wand
    python run_calibration.py --fit-k1 --start tests/states/t3k_h1.pkl --state tests/states/t2_all.pkl --log tests/t2_calibration_console.txt
    # W4c, the result with the channel and the principal point at the centre (4 min), from T2; writes
    # calib_plate_wand_ref_centre (tests/w4_delivery.py --centre runs it, the halves and the camFiles' round trip)
    python run_calibration.py --fit-k1 --channel --aim centre --start tests/states/t2_all.pkl --out ref_centre --state tests/states/w4c_ref.pkl
    # W4, the same with the principal point fitted; writes calib_plate_wand_ref
    python run_calibration.py --fit-k1 --fit-lens pp --channel --aim centre --start tests/states/t2_all.pkl --out ref --state tests/states/w4_ref.pkl
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
- `--fit-lens pp[,k2,tangential]`: fit lens terms besides f and k1 (plate_wand's `fit_lens`).
- `--channel` (`--channel-tol MM`): hold the channel of `config.json`'s `constraints` (the wand stays inside it).
- `--aim centre` or `--aim CAMS:WALL:MM` (`--aim-sigma MM`): hold where cameras aim.
- `--out ref_centre` or `--out ref`: write to `config.json`'s `out_dir_ref_centre` (`calib_plate_wand_ref_centre`) or
  `out_dir_ref` (`calib_plate_wand_ref`).
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
| `tests/` | **Tests:** T1 (detections), T2 (`t2_*`: the delivered fit, its from-scratch run, the camFiles' round trip), T3 (repeatability over frame subsets and halves, k1 held and fitted), T4 (held-out frames), T5 (hints), T6 (water index, wall thickness, k1), T7 (against the GUI's calibrations). **Diagnoses:** D1 (how well the data fix each wall's position), D2 (the frames that pin a wall), D3 (the border-cut detections dropped). **The walls against the channel:** W0 (`w0_*`: the images, refraction's own distortion, every fit's wand against the channel), W1 (walls held in the channel), W2 (lens terms), W3 (the constraints), W4 (the delivery with the channel). `run_batch.py` runs the subset fits; their states go to `tests/states/` (not tracked by git: regenerable), their consoles to `tests/logs/`. |

The case found two library bugs on real data and fixed them on 3 Oct, with your approval:
- a crash in `cylinder_wand.affine_factorization`, stage 1 of both wand calibrations;
- a rotation check in `plate_model.to_cpp` too strict for rounding.

Each has a regression test in `tests/test_plate_wand.py`. Records: `tests/fix_*_console.txt` and the report, sections 3
and 4. Both are committed (`180063f`).

For your channel it added two options to `plate_wand` on 3 Oct, with your approval. Both are off by default and not
committed:
- `fit_lens`: the principal point, k2, the tangential terms.
- `constraints`: a channel the wand stays in, where cameras aim, a camera's distance to its wall.

Three regression tests cover them (`tests/l12_pytest_console.txt`); the walls report has the details, section 7.
