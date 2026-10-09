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
- Every wand end of the campaign lies in the single-image volume: 907,000 point–camera pairs, each with exactly
  one image (E0, E0c).
- **No C++ bug** (E0b, E0c). The first census's cameras, made by `cyl_model.to_cpp` with a tolerance of 10⁻¹², skipped
  27 rays that graze the point's own radius, where rounding limits the residual to about 10⁻¹²–4×10⁻¹¹. The projection's
  own default, 10⁻⁹, finds them all.

**`projection_cpp/` (2026-10-06):** the paper's comparison of the projection with iterative projection (Section 9.2.1,
Fig. 14, S8) in C++. A module of its own compiles OpenLPT's search unmodified, together with C++ ports of the iterative
methods of the paper's `code/paper_baselines.py`, and times them alike. See its `2026-10-06-plan.md` and
`2026-10-06-report.md`. Nothing outside that folder is changed or built.

**`experiment/` (2026-10-08):** the paper's experiment (Section 9.3, Figs. 16 and 17, Table 3, S8's paragraphs on
the experiment).
- **The paper's run, 20261008 (three cameras):** case `cases/cyl_20261008`, its deliverables on X:.
  `experiment_numbers_20261008.py` prints every number the section quotes with its source
  (`experiment_numbers_20261008_out.txt`, `.json`). It also writes the figures' data (`experiment_data_20261008.npz`,
  copied into the paper by `code/make_paper_experiment_figs.py --refresh`).
- **What it computes:** nothing is fitted or tracked afresh. It smooths the tracks with the group's Gaussian filter
  (width 3 steps, the author's choice), reruns the line detector on ten frames (the particles detected per frame,
  and one sample frame), and traces the rig figure's lines of sight.
- **Running it:** use the refract interpreter, on one core (about 12 min). `CYL_EXPERIMENT_PYD` names a folder with
  a copy of `build-stb/p4c/out`'s `.pyd`, so that the build is not locked.
- **Run 20261007 (two cameras), which the section quoted first:** `experiment_numbers.py` (`experiment_numbers_out.txt`,
  `experiment_data.npz`), and `ghost_filter.py`, the neighbour and swirl tests its track figure used
  (`ghost_filter_check*_out.txt`).

**History:**
- `2026-10-03-plan.md`: the first plan, with P0–P5 reruns and a single-image rule. It is superseded: the paper
  dropped the rule, and the campaign already had the results.
- `2026-10-06-plan.md`: the plan that was run.

The tube is the one of the dot-board trial `cases/cyl_20260930_T0_cam1`:
- OD 7.9375 in, wall 0.125 in;
- air, acrylic and water at 1.0, 1.49 and 1.333.

## Notes (nothing to fix now)

1. **`cyl_model.to_cpp`'s default tolerance of 10⁻¹²** is below the rounding floor next to grazing. Raising it to the
   projection's 10⁻⁹ would be a one-line change; it is left as is, since a rounding-level effect needs no extra
   effort (your rule of 2026-10-06).
2. **`cylinder_wand.multi_image_guard` checks only points beyond r_c at zero tilt** (`2026-10-03-plan.md` §6). The
   paper no longer uses a guard.
