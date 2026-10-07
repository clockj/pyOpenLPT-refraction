# The wand tab's fixes (7 Oct 2026): tests

**Start with the reports:**
- `2026-10-07-report.md`: B1–B10, the bugs found while rewriting the user guide;
- `2026-10-07-report-b11-b12.md`: B11–B13, the Pinhole model's bugs and the fast detection mode.

**Their plans** sit beside them: `2026-10-07-plan.md` and `2026-10-07-plan-b11-b12.md`.

**Each test** is a script `<name>.py` with its outputs beside it:
- `<name>_out.txt`: what it printed, with every check;
- `<name>.json`: its numbers;
- `<name>_console.txt`: the whole console.

**Earlier runs** of a test are kept under the same name with a suffix, and the reports say why:
- `_first_try`: a test corrected after its first run;
- `_before_b11_b12`, `_before_b13`: a rerun after a later fix;
- `_before`: W8's run before its fix.

## Reproducing

**Environment:** the `refract` conda environment (pyOpenLPT's `CLAUDE.md`), with pyOpenLPT built (`build_local.py`).
Run each script from this folder with that interpreter:

```
python make_before_lib.py           # first: before_lib/, the code before the fixes (git 99e2d52), for W0, W5, W9/W10, X1, S2
python w1_bright_balls.py           # needs run 20261006/T2 on X: (OPENLPT_T2_RAW overrides the place); writes data/w1/
python w0_plate_wand_unchanged.py
python w2_w3_plate_precheck_notes.py
python w4_del_marks.py
python w5_pause.py                  # uses W1's images
python w6_w7_cameras_decimals.py    # uses W1's images
python w8_panel_widths.py           # the GUI's own window, off the screen (not offscreen Qt)
python w9_w10_pinhole.py
python f1_pinhole_ids.py
python x1_pinhole_unchanged.py
python x2_x3_pinhole_sizes_ids.py
python d3_two_camera_flatness.py
python d2_fast_mode.py              # renders data/d2/; its T2 part uses W1's images
python s1_s2_fast_seeded.py         # re-renders data/d2/; its T2 part uses W1's images
python w11_regression.py [--tag NAME] [g3 g5_g9 g6 g8 g4 pytest]   # tests/cylinder_wand_gui's GUI tests and pytest
```

**What depends on what:**
- **The synthetic data** is drawn from fixed seeds. The pinhole calibrations, plate_wand and the seeded detectors are
  deterministic, so a rerun gives the same numbers. The exceptions: times, and the fast detection before B13 (the
  `_before_b13` outputs).
- **Each result reflects the code of its run.** W1 and D2 were rerun after B13, F1 and W9/W10 after B11/B12; the
  other W tests ran on the code of B1–B10. The reports give each run's state.
- **`w11_regression.py` runs tests/cylinder_wand_gui's GUI tests in place,** then moves their outputs here (into
  `w11/`, or `w11_NAME/` with `--tag`) and restores that folder byte for byte.

## Not tracked

- **`data/`** (about 40 MB): the images and run outputs the scripts write. These are W1's T2 frames (copied from
  X:), D2's synthetic images, and the GUI runs' camFiles and pages. The scripts write them again.
- **`before_lib/`:** git history (`make_before_lib.py` writes it).
- **`w11*/*.png`:** the regression's screenshots of the results pages, which G4 draws again. The G tests' verdicts
  and numbers are in their `.json` and `_out.txt`, which are tracked.
