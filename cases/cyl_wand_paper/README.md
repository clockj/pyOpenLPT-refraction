# Case: the paper's synthetic wand calibration through the tube

The synthetic wand-calibration tests behind Section 9.2.2 of the Optics Express paper (`D:\Refraction\paper`). The fit
is pyOpenLPT's `cylinder_wand` with two changes:
- **its residuals come from the paper's projection** (the C++ PINCYL `projectAll`), polished by `cyl_model.project`;
- **only the single-image region is used.**

Both changes are made in memory, inside this folder's code. No library file is edited.

**Status (2026-10-04): plan only, waiting for approval.** See `2026-10-03-plan.md`. Nothing has been built or run,
and on 4 Oct you decided to change no code for now.

The plan was amended on 4 Oct, after you asked whether the single-image rule could steer the fit (its section 7):
- the rule runs once, on the converged answer, and a recount afterwards warns on drops;
- where the C++ projection finds no image, the residual falls back to the library's Newton. The C++ does not look
  outside the water, so without the fallback the fit would refuse steps that the library accepts;
- P0 gains a barrier test.

The tube is the one of the dot-board trial `cases/cyl_20260930_T0_cam1`:
- OD 7.9375 in, wall 0.125 in;
- air, acrylic and water at 1.0, 1.49 and 1.333.

## Reminder: the library's double-image guard (not fixed, by request)

`modules/camera_calibration/curved_calibration/cylinder_wand.py`, `multi_image_guard`, has a gap.

- **What it checks.** Only points beyond `Tube.fold_radius()`, which is r_c at zero tilt (75.62 mm here).
- **Why that misses points.** Tilted rays fold inside that radius:
  - r_c is 74.04 mm at C_z = 0.3;
  - a point at r = 74.5 mm, Δφ = 130.50°, Δz = 357.5 mm, seen from 1.1 m, has two images.
  - The paper also notes that multilayer cylinders can fold inside r_c.
- **Effect so far.** Nothing reported is affected: wand rigs have C_z ≈ 0.02, which leaves a 7 µm band.
- **Fix later.** Take the census of every observation; the single-image rule of this case does so.
- **Logged** in `tests/cylinder_wand/README.md` on 2026-10-04 (progress log and open item 8).

Details: `2026-10-03-plan.md`, section 6.
