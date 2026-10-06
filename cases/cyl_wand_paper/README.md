# Case: the paper's synthetic wand calibration through the tube

The synthetic wand-calibration evidence for Section 9.2.2 of the Optics Express paper (`D:\Refraction\paper`).

**The calibration the paper describes** (Section 9.1, as rewritten in the paper's commit `150325e`) is pyOpenLPT's
`cylinder_wand` with two changes:
- **the residual's image comes from the paper's projection,** the C++ PINCYL `projectAll`, which starts Newton
  (`cyl_model.project`); the library starts Newton at the detection;
- **no double-image guard.** The wand is kept in a central single-image volume instead.

`projection_fit.py` makes both changes in memory. No library file is edited.

**Status (2026-10-06): done.** See `2026-10-06-report.md`.
- On the campaign's data (`tests/cylinder_wand`), the paper's calibration and the library's give the same residuals
  and the same fits (E1, E2). Section 9.2.2 therefore quotes the campaign's results. `paper_numbers.py` lists each
  number with its source.
- Every wand end of the campaign lies in the single-image volume (E0).
- **Found:** the C++ projection misses images whose ray grazes the point's own radius, 27 of 907,000 pairs (E0b).
  The Python reference finds them. Not fixed, by request (reminder below).

**History:**
- `2026-10-03-plan.md`: the first plan, with P0–P5 reruns and a single-image rule. It is superseded: the paper
  dropped the rule, and the campaign already had the results.
- `2026-10-06-plan.md`: the plan that was run.

The tube is the one of the dot-board trial `cases/cyl_20260930_T0_cam1`:
- OD 7.9375 in, wall 0.125 in;
- air, acrylic and water at 1.0, 1.49 and 1.333.

## Reminders (not fixed, by request: no other pyOpenLPT file is to change)

1. **The C++ projection's blind band at the branch junction** (report, E0b).
   - Points about 90–94° in azimuth from the camera get no image when their ray grazes their own radius within about
     10⁻⁸ of Λmax.
   - The Python reference's fixes for this region are in `D:\Refraction\code\tests\classify_grazing\`.
2. **`cylinder_wand.multi_image_guard` checks only points beyond r_c at zero tilt** (`2026-10-03-plan.md` §6). The
   paper no longer uses a guard.
3. **`tests/cylinder_wand/README.md`** needs the E0b finding and the corrections to its 0.1 px table (report, "The
   numbers for the paper"). It was left alone at your request of 2026-10-03.
