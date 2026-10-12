# Phase C report: the calibration of run 20261007 (T0's wand, checked with the Planes)

7 October 2026. Plan: `2026-10-07-plan.md`, Phase 0 step P0e and Phase C (approved with all phases). Every number
below is from a saved output in `tests/` (named in each section) or in `X:\Shijie Zhong\Refraction\Processed\20261007`.

## 1. In short

- **Delivered:** `X:\...\Processed\20261007\T0\camFile\` (the wand fit, f free) and, for tracking,
  `camFile_track\` (the same, projection tolerance 1e-9). 3,036 frames, reprojection RMS 0.600 px, wand length RMS
  34.4 um.
- **f is not determined by the wand**, as on 6 Oct: the halves and quarters give cam1 41,351-53,010 px and cam2
  23,639-30,417 px. Inside the wand's volume every part's cameras agree (held-out wand lengths to 0.02-0.12 %).
- **The board agrees with the delivered fit within its precision.** Triangulated with it, the board is flat
  (18-47 um RMS), square (pitch along the axis = pitch across it to 0.11 %; 90.04-90.06 deg) and its planes are
  parallel. Its pitch reads 0.36-0.50 % above 3 mm: the wand and the board disagree in scale by that much.
- **The stage steps were 2.54 and 2.46 mm**, not the 3.175 mm assumed (about 0.1 in, not 0.125 in).
- **The rig changed since 6 Oct** (C1-X: 9 px and 4.9 % on each other's frames): this day needs its own calibration,
  which it has.
- **Holding f (C3)** was built and tested (synthetic: passes), but on T0 it does not make the parts agree, so the
  free fit is delivered.
- **A library problem found:** with the thorough start search (six starts) a start in a wrong minimum (cam1 at
  253 mm, RMS 1.38 px) won `calibrate`'s selection, because its larger residuals widen its outlier limit, so it
  "kept more observations" (section 3). The case uses two starts; the library was not changed (outside this phase).

## 2. What was run

| step | script (in `tests/`) | judged by (plan) | result |
|---|---|---|---|
| P0e | `p0e_planes_survey.py` | facts for C2 and L3 | dots found, blur measured (section 4) |
| C1 | `../calibrate.py`, `c1_gui_csv.py`, `c1_residuals.py`, `c1_references.py` | converges; no residual pattern above ~0.1 px; page checks explained | converges at 0.600 px; **image-fixed pattern up to 0.46 / 0.96 px** (as on 6 Oct); the page raises no check |
| C1-T | `c1t_variants.py`, `c1t_held_out.py` | an f spread as on 6 Oct means f undetermined → C3 | **f undetermined** (cam1 ±12 %, cam2 −16 … +7 %) |
| C1-X | `c1x_cross_runs.py` | reported | **the rig changed since 6 Oct** |
| C1-D | `c1d_sync.py` | offset below 0.01 frame | −0.0006 ± 0.0003 frames: **pass** |
| C2a | `c2a_board_check.py` | pitch 3 mm within ~0.1 %; planes flat; steps match the stage | flat; square; **pitch +0.36 … +0.50 %; steps 2.54 / 2.46 mm, not 3.175** |
| C2b | `c2b_f_board.py`, `c2b_valley_board.py` | f and distance with error bars, against C1's line | one camera alone: **not determined**; both cameras: the board fixes cam2's f along the valley at about 29,000 px |
| C3 | `../../../tests/cylinder_wand/t14_hold_f.py`, `c3_hold_f.py` | synthetic: truth within σ; real: C2a passes and the parts agree | synthetic **pass**; real: **the parts still disagree** with f held → not delivered |
| C4 | `c4_track_camfiles.py` | same pixels as the 1e-12 files | **pass** |

## 3. C1, the wand calibration

**First run: a wrong minimum won (kept in `wrong_minimum_nfinal6/` here and on X:).** With the 6 Oct settings
(the thorough start search, `n_final` 6) the calibration delivered cam1 f 10,995 px at 253 mm from the axis and cam2
7,709 px at 210 mm, RMS 1.38 px (`wrong_minimum_nfinal6/calibrate_out.txt`). Its log shows why
(`calibration_log.txt` there):
- starts 0-2 reached the right minimum: stage-4 RMS 0.600, 0.5995, 0.5995 px, with 11,876 or 11,872 of 12,144
  detections active;
- start 3 reached the wrong one: RMS 1.380 px, but with 12,088 active;
- `calibrate` ranks answers by the observations kept first and the likelihood second, so start 3 won. A worse
  minimum has larger residuals, a larger noise estimate and so a wider 5-sigma outlier limit: it keeps more.

The case therefore uses `n_final` 2, the GUI's default (`config.json`, with the reason): the first two starts both
reach the 0.600 px minimum. **The same can happen to a GUI user who ticks "Thorough start search".** A fix (e.g.
rank by the likelihood over a common set of observations, or reject an answer whose RMS is far above another's) is
a library change for the user to decide on; nothing in `cylinder_wand` was changed for it.

**The delivered fit** (`calibrate_out.txt`, `calibrate.json`):
- 3,036 frames kept in both cameras (cam1 kept 5,966 camera-frames, cam2 8,334 of 15,001 each); 11,876 of 12,144
  detections used; RMS 0.6003 px (cam1 0.534, cam2 0.660 px); wand length of a free triangulation RMS 34.4 um, mean
  −0.002 um.
- cam1: f 48,494 ± 427 px, 1,178.65 ± 10.6 mm from the axis. cam2: f 28,306 ± 284 px, 835.00 ± 7.9 mm, azimuth
  60.30 deg from cam1, 16.8 mm higher.
- The noise weights were not applied (the per-camera estimate flipped between two states: equal weights).
- `wand_points.csv` loads in the GUI with every frame and the selection's centres (`c1_gui_csv_out.txt`: 5 of 5).

**Residuals** (`c1_residuals_out.txt`, `c1_residual_maps.png`): per ball the mean residual is below 0.006 px, but
the 10 × 6 cell means reach 0.46 px (cam1) and 0.96 px (cam2), with 35 of 41 and 36 of 37 cells above 0.1 px, mostly
horizontal in cam2. The 6 Oct runs showed the same (0.53-0.84 px). The wand length trends with X (−1.30 ± 0.22 um per
mm) and Z (+0.61 ± 0.12 um per mm). Fails the plan's 0.1 px; it is the image-fixed model error the wand campaign
warned of, and it limits any camera through this tube at the 0.1-1 px level.

**References** (`c1_references_out.txt`): f differs from 6 Oct's by +26 % (cam1) and −21 % (cam2); f / distance
by +7.5 % (cam1: 41.14 against 38.29 and 38.21 px/mm) and −1.7 to −1.9 % (cam2). The valley's line does not pass
through the origin (C1-T's parts: cam1 f / D 39.7-41.9), so f / D compares only to a few %; with C1-X (below) the
cameras clearly changed since 6 Oct.

**C1-X** (`c1x_cross_runs_out.txt`, `c1x_offsets.png`): 6 Oct T2's cameras on this run's frames measure the wand
+485 ± 7 um long (4.9 %) and the cameras disagree vertically by −8.85 px; this run's cameras on T2's frames −402 um
and +9.21 px. On its own frames this run's cameras agree to 0.000 px, with a drift of −0.0124 ± 0.0013 px per 1,000
frames (−0.19 px over the 150 s recording).

**C1-D** (`c1d_sync_out.txt`): offset cam1 − cam2 −0.0006 ± 0.0003 frames; the wand moves 22 px per frame. Pass.

## 4. P0e and C2a, the board

**P0e** (`p0e_planes_survey_out.txt`, `p0e_blur.png`, overlays):
- cam1 (face-on) finds 44-45 usable dots per position; cam2 (oblique) 103 / 94 / 85.
- cam2 sees a dark band at the board's edge on the left (wider as the board moves away from cam1). The 30 Sep
  detector took it as its reference dot size and dropped every dot as a speck, so `planes.mask_border_blobs` fills
  it first. In cam2 at positions 1 and 2 the band shows refracted copies of dots; they are not used.
- The dots' outlines are blurred differently along the rows and the columns. cam1: along the rows 3.1 / 3.6 / 4.3 px
  (median, sigma of an erf edge), along the columns 2.0 / 1.9 / 1.8 px at positions 0 / 1 / 2. cam2: both grow with u
  (the board's depth across its view): along the rows from 2.0 to 4.6 px and along the columns from 2.3 to 9.2 px
  between u = 200 and 1080 px at position 0. These are for L3 (the line-image shape model).

**C2a** (`c2a_board_check_out.txt`, `c2a_board.png`; dot tables on X:):
- **Pairing the two cameras' dots** is unambiguous: the best lattice map has a median line-of-sight distance of 16.8,
  18.0 and 14.9 um at the three positions, the next best 81-190 um (4.8-12.8 times).
- **Triangulated** (maximum likelihood, both cameras): reprojection RMS 0.32-0.35 px (cam1) and 0.38-0.41 px (cam2).
- **Flatness** RMS 18.3 / 39.0 / 47.1 um, largest 44 / 103 / 105 um; dots about the fitted lattice 26 / 43 / 51 um.
- **Square:** angle 90.061 / 90.060 / 90.041 deg; the pitch along j (the tube's axis) exceeds that along i by 0.08-0.13 %.
- **Pitch:** along i 3.0107 / 3.0113 / 3.0128 mm (+0.36 / +0.38 / +0.43 %), along j 3.0147 / 3.0148 / 3.0151 mm
  (+0.49 / +0.49 / +0.50 %), standard errors 0.001-0.003 mm. **Fails the plan's ~0.1 %**: inside the wand's volume
  (the board sits at X 10.5-11.1 mm, Z −0.8 to −5.8 mm, where the wand was) the wand calibration's scale is 0.4-0.5 %
  larger than the board's nominal 3 mm. One of the two references is off by that much (the wand's 10 mm is the
  user's figure; the board's pitch is nominal); this case cannot tell which.
- **Steps:** 2.539 mm (0 → 1, 44 dots) and 2.461 mm (1 → 2, 43 dots), 1.0 and 1.3 deg from cam1's line of sight,
  away from cam1, 2.5 deg from the board's normal. Not the 3.175 mm assumed: −20 % and −22 %. The first is 0.1 in
  (the 30 Sep stage's step); the user's "about one full turn" was approximate.

## 5. C2b and C3: f

**C2b, each camera on its own** (`c2b_f_board_out.txt`): cyl_model fitted to the board's three positions (camera
rotation, distance, f; board pose and translation). It does not determine f: cam1 f 46,319 (translation free),
38,032 (step held at 3.175 mm, which C2a shows is wrong) and 43,404 px (with per-dot corrections); cam2 33,292,
31,750 and 13,000 px (the last a collapse, 216 parameters). The single-camera fits leave 0.8-1.6 px of residual:
the board's real departures from a flat nominal grid (tens of um, as on 30 Sep) are larger than what three
positions 2.5 mm apart can separate from f.

**C2b, both cameras along the valley** (`c2b_valley_board_out.txt`): the board triangulated with C1 and the six
parts. Flatness, lattice residual, steps and the planes' tilt do not change along the valley. The pitch along the
tube's axis does: from −2.1 % to +5.1 % while the pitch across stays +0.15 to +0.45 %, and the angle from 89.78 to
90.52 deg, both mostly with cam2's f. A square board therefore fixes cam2's f; cam1's (face-on) hardly matters.

**C3, holding f.** The option, in `cylinder_wand` (the user's library, the one change allowed):
- `Layout(..., fix_f=...)` leaves the log f column of the chosen cameras out;
- `calibrate(..., fix_f=...)` passes it on (it needs `start`, whose f is held), and `coarse_fit` keeps it;
- `cylinder_wand_gui.Job.fix_f` passes it from the GUI's job (written to the result's settings only when set);
- default empty: nothing changes. `pytest tests/test_cylinder_wand.py`: 7 passed.

Synthetic test `tests/cylinder_wand/t14_hold_f.py` (2 cameras 1,000 mm from the axis, 61 deg apart, f 38,000 px,
2,000 frames, 0.4 px noise, 4 seeds; `t14_hold_f_out.txt`):
- f free: within 0.87 % (sigma 0.42-0.47 %);
- f held at the truth: the distances within |z| ≤ 1.21 and the centres within |z| ≤ 1.83 of the truth;
- f held 1 % high: the distances move +0.98 % (along the valley), the RMS by under 0.001 px, the held-out wand length by
  under 0.05 um.

Pass.

**On T0** (`c3_hold_f_scan_out.txt`, `c3_hold_f_fit_out.txt`, `c3_hold_f_parts_out.txt`):
- **Scan:** cam2's f held at 26,000-30,000 px, cam1's free. cam1's f follows weakly (49,208 to 48,016 px); the RMS
  is 0.6018-0.6076 px (C1: 0.6003). The board's pitch difference B − A crosses 0 at cam2 f = 29,035 px (−0.246 % per
  1,000 px), so the board puts cam2's f 2.6 % above the free fit's 28,306. The angle reaches 90 deg only by
  extrapolation (30,761 px; −0.0245 deg per 1,000 px), a weaker measure.
- **Fit with f held** at 48,306 (cam1: the scan's line at 29,035) and 29,035 px (`checks\c3_held_f`): distances
  1,173.0 and 855.2 mm, RMS 0.6026 px, board B − A −0.02 %.
- **Its halves and quarters with the same f held:** cam1's distance still ranges 1,155-1,206 mm (f / D 40.07-41.82)
  and cam2's 845-875 mm. The board's B − A on them ranges −2.0 % to +3.8 %, as wide as with f free (−2.6 … +5.0 %).
  The parts' disagreement is not only along f.

**Decision: the free fit is delivered.**
- The held fit changes cam2's f and distance by 2.6 % and is not better on any measure: RMS 0.6026 against 0.6003 px,
  the same board within its precision.
- It does not meet C3's criterion that the parts agree.
- The board alone does not pin cam1's f.
- Inside the wand's volume, where both agree, the tracking will be the same with either. Outside it, neither is
  validated.

## 6. C4, the tracking camFiles

`c4_track_camfiles_out.txt`: `camFile_track\cam1.txt` and `cam2.txt` differ from `camFile\` in the tolerance's line
only (1e-12 → 1e-9). On 20,000 points in the wand's box (inside the water) and 4,000 in each camera's near/far junction
plane, both versions give the same pixels (difference 0 px) wherever either gives one, and no point loses its image
at 1e-9. (This sample had no point that failed at 1e-12 either; 6 Oct T2's cameras had 3 in 40,000.)

## 7. Open issues

1. **The calibration's selection rule** can pick a worse minimum when more starts are carried (section 3). It is a
   library fix for the user to approve.
2. **f stays undetermined** (±12-16 % over the parts); the report's and page's error bars (0.9-1.0 %) are far too small
   and the page has no warning. The board fixes cam2's f to about ±3 % at best; nothing here fixes cam1's.
3. **Scale:** the wand's 10 mm and the board's 3 mm disagree by 0.4-0.5 %. Measuring the wand (or a known length in
   the tube) would settle it.
4. **The image-fixed residual pattern** (0.46 / 0.96 px cells) remains the limit of this camera model on real images.
5. **The stage:** 2.54 / 2.46 mm per position, not 3.175 mm.
6. **The cameras drift slightly during T0:** −0.19 px of vertical disagreement over the recording.

## 8. Rerunning

From the case folder, with `CASE_PYOPENLPT_DIR` set to a copy of the module (`data/pyd/`, git-ignored):

    python tests/p0e_planes_survey.py
    python detect.py                       # 28 min over the share
    python calibrate.py                    # 6 min (two starts)
    python tests/c1_gui_csv.py ; python tests/c1_residuals.py ; python tests/c1_references.py
    python tests/c1x_cross_runs.py ; python tests/c1d_sync.py
    python tests/c1t_variants.py ; python tests/c1t_held_out.py      # 3 min, 6 processes
    python tests/c2a_board_check.py ; python tests/c2b_f_board.py ; python tests/c2b_valley_board.py
    python tests/c3_hold_f.py scan ; python tests/c3_hold_f.py fit 29035 48306 c3_held_f ; python tests/c3_hold_f.py parts 29035 48306
    python tests/c4_track_camfiles.py
    python ../../tests/cylinder_wand/t14_hold_f.py      # the hold-f option, synthetic
