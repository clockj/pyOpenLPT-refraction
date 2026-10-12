# Evidence for the plan (7 Oct 2026)

Scratch probes that informed `../2026-10-07-plan.md`. They were run on 7 Oct from the session's scratch folder, with
the `refract` environment and a frozen copy of the 6 Oct build of the module (`build-py/Release/pyopenlpt*.pyd`).

**These are kept as they ran, and they will not run from here.**
- **The relative paths are the scratch folder's:**
  - `pyd/`: the module copy;
  - `cams/`: T2's camFiles of 6 Oct (`X:\...\Processed\20261006\T2\Results\camFile`);
  - `variants/`: modified copies of them;
  - and T1's raw images on X:.
- **Phase 0 repeats the measurements on today's cameras.**

| script | measured | output |
|---|---|---|
| `probe_rig.py` | on T2's cameras:<br>• A1: image census over the volume both cameras see<br>• A2: cost of `project()` (PINCYL at tolerance 1e-9 and 1e-12, against a PINHOLE with the same K, R, t) and of `lineOfSight()`<br>• A3: curvature of the epipolar curve<br>• A4: tangential and sagittal focal depths (parallax of a pinhole displaced by ±0.5 mm) and the image Jacobian's anisotropy | `probe_rig_out.json` |
| `probe_cost.py`, `probe_cost_print.py` | where a projection spends its time (`classify_near`, `classify_far`, a warm-started near-branch Newton) | `probe_cost_out.txt` |
| `probe_fail12.py` | the projections that return no image at tolerance 1e-12 but do at 1e-9 | `probe_fail12_out.txt` |
| `probe_epi.py` | distance of true matches from StereoMatch's straight image line (box entry/exit anchored) | `probe_epi_out.txt` |
| `probe_detect.py` | the C++ tracer finder on synthetic stretched spots, against a 1 px pre-smoothing and a 2-D Gaussian fit | `probe_detect_out.json` |
| `recon_t1.py` | T1 frame 0: background, noise, objects and their shapes; displacement between frames 0, 1 and 4 | `recon_t1_out.json` |
| `hdr.py` | the `.chd` headers: size, serial, frame rate, exposure | — (printed: T0 100 frames/s, T1 cam2 4,000 frames/s, both 100 µs) |

**`figures/`** (contrast-stretched views):
- T0, T1 and two of the Planes, frame 0 of both cameras;
- `t1_cam1_crop.png` and `t1_cam2_crop.png`: a 240 × 160 px region of T1 enlarged 4×;
- `t1_cam1_highpass_crop.png`: the same region of cam1 after background removal and the high-pass.
