# A pixel size per camera in the wand tab (8 Oct 2026): tests

For run 20261008, whose cam3 (Phantom v2640, 13.5 µm pixels) sees the wand's balls 1.5 times larger than cams 1–2
(VEO 710L, 20 µm), and saved a shorter range of images.

- **Plan:** `2026-10-08-plan.md` (from the case plan `cases/cyl_20261008/2026-10-08-plan.md`, Phase G).
- **Report:** `2026-10-08-report.md`.

**Each test** is a script `<name>.py` with its outputs beside it: `<name>_out.txt` (what it printed, with every
check), `<name>.json` (its numbers) and `<name>_console.txt` (the whole console).

## Reproducing

The `refract` environment (pyOpenLPT's `CLAUDE.md`), from this folder. The tests read run 20261008/T0 on X:
(`OPENLPT_T0_1008` overrides the place):

```
python g1_scaled_detection.py     # the scaled detection (no Qt)
python g2_unscaled_unchanged.py   # without pixel sizes, bitwise as git HEAD (writes before_lib/, git-ignored)
python g3_alignment.py            # the Phantom header and the alignment (no Qt)
python g4_gui.py                  # the wand tab offscreen (checks only: offscreen Qt has no fonts)
python g4b_screens.py             # pictures from the GUI's own window, off the screen
python g4c_table_width.py         # the camera table's columns at three window sizes
```

| file | what |
|---|---|
| `common.py` | paths (T0's folders, the cameras' first image numbers and pixel sizes), the check log, the offscreen Qt application and its dialogs (tests/wand_gui_fixes/common.py's) |
| `g1_*` … `g4_*` | the tests |
| `before_lib/` (git-ignored) | git HEAD's `wand_calibrator.py`, written by G2 |
| `first_run/` | G4's and G4b's first runs, before the camera table's widths were fixed (G4c) |
