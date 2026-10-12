# The GUI's flat-window wand calibration: the real case's options (9 Oct 2026)

You asked: "Improve gui for refract+plate". Approved with:
- **D1 (b):** "Radial k1" by default for Pinhole+Refraction;
- **D2 (a):** values per window;
- **D3 (a):** the panel's values also for a refit;
- **D4 (a):** constraint fields in the panel.

| file | what |
|---|---|
| `2026-10-09-plan.md` | the plan |
| `2026-10-09-report.md` | the report |
| `pgcommon.py` | shared helpers: paths (the real case, its results on X:, read only), G8's offscreen Qt helpers, the check log, rig B's wand points as the GUI holds them, a view set up as G8 sets it up, `catch_job` (the Job the view builds, taken instead of starting the process) |
| `p0_unchanged.py` | P0: with no new option set, today's view and run (commit cffed7d's, by `git show`) and the change give the same Job and bit-identical results, with k1 off and on; Precalibrate too |
| `p1_real_case.py` | P1: the real case's delivered fit W4c (`calib_plate_wand_ref_centre`) made through the GUI's path: Load CSV, the panel, the saved start (T2's folder), the constraints |
| `p2_options.py` | P2: each option on a synthetic rig (B', window 1 10 mm): per-window thickness, the saved start (JSON, camFiles), D3, constraints, lens terms, hints, refusals |
| `p2b_thickness.py` | P2 (b): what a window's thickness changes in the fit (after P2's case A failed its second criterion) |
| `p3_panel.py` | P3: the panel's controls and the Job they make, offscreen (D1's default, the Windows table, Start / Hints, Precalibrate, bad inputs) |
| `p3b_screens.py` | P3 (b): pictures of the panel from the main window on the real platform (`p3b_panel_1-3.png`) and its width in its 358 px column |
| `queue_p.sh` | everything above on the final code, then the wand tab's regression (`tests/wand_gui_fixes/w11_regression.py --tag p4pwg`) |

Each script writes `<name>_out.txt` and `<name>.json` here. Run data (written calibrations, `git show` copies) goes to
`data/` (git-ignored).

```
python p3_panel.py; python p0_unchanged.py; python p1_real_case.py; python p2_options.py; python p2b_thickness.py
python p3b_screens.py
bash queue_p.sh            # all of it, then W11
```
