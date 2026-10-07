# Case: tracking tracers through the tube, run 20261007

The recordings of 7 October 2026 through the water-filled acrylic tube, with the two Phantom VEO 710L cameras of
6 Oct:
- `T0`: wand calibration, 15,001 frames at 100 frames/s;
- `Planes`: the dot board at three positions;
- `T1`: tracers, 1,001 frames at 4,000 frames/s, backlit; each particle images as a short horizontal or vertical
  line, because of the tube's astigmatism.

- **Plan:** `2026-10-07-plan.md`. **Status: waiting for approval.**
- **Raw images:** `X:\Shijie Zhong\Refraction\Raw\20261007\{T0,Planes,T1}`.
- **Results:** `X:\Shijie Zhong\Refraction\Processed\20261007\{T0,Planes,T1}` (not created yet).
- **`p0_evidence/`:** the scratch probes behind the plan, with their outputs and figures.

The library work the plan needs (fast projection, curved epipolar lines, a line detector, the shape model) will have
its own tests in `tests/cylinder_stb/`.
