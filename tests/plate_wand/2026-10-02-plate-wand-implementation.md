# Flat-window (PINPLATE) wand calibration: implementation plan

Date: 2026-10-02. This is the working design behind `2026-10-02-plate-wand-plan.md`, which you approved on 2026-10-02
("go ahead to finish all phases follow the plan"). It is written so the work can continue after a context compaction
with nothing lost.
- **Status of each phase:** the checklist in section 9, kept current.
- **If the plan and this file disagree:** the plan wins on scope and criteria, this file on how.

## 0. Rules for this work

- **Python:** `C:\Users\zcloc\.conda\envs\refract\python.exe`. Its BLAS must stay OpenBLAS. Set `OMP_NUM_THREADS=1`
  and `OPENBLAS_NUM_THREADS=1` in parallel workers.
  - SymPy scripts run with the base `python`.
  - Never use `conda run`, which swallows output.
- **Patches:** write patch scripts with the Write tool, asserting each match. Bash heredocs eat backslashes here.
- **Git:** no commits.
- **Do not touch:**
  - `D:\Refraction\paper`;
  - the C++ library;
  - the GUI (`view.py`, `modules/camera_calibration/wand_calibration/*`);
  - the case folder;
  - `pose_calibrator.py`;
  - `cylinder_wand*.py`.
- **The one approved change to existing code:** the two-line `cyl_model.py` fix (section 1).
- **Results:** every test script writes `<name>_out.txt` (and `.json`) beside itself in `tests/plate_wand/`.
- **Logs:**
  - Keep `tests/cylinder_wand/README.md` current with anything that touches the cylinder code.
  - Write `tests/plate_wand/README.md` as the plate summary.
- **Parallel runs:** the machine has 32 cores. Use `multiprocessing` with spawn, and pass every option through the task
  arguments: spawned workers re-import the modules, so parent-side overrides are lost.

## 1. Phase 1: the `cyl_model.py` rotation fix (approved: input 6 default)

- `log_so3`, half-turn branch: `axis = B[:, k] / max(np.linalg.norm(B[:, k]), 1e-300)` instead of dividing by
  `sqrt(B[k, k])`.
- `rvec_of`: `return log_so3(np.asarray(R, float))`, with no `cv2.Rodrigues`. OpenCV loses up to 2e-5 rad near a
  half-turn (`p_logmap_check.py`).
- **Check:**
  - rerun `p_logmap_check.py`, and add random rotations at all angles: the round trip within 1e-14 rad;
  - `pytest tests/test_cylinder_wand.py` (7 tests, about 1–2 min);
  - log it in the cylinder README's progress log and open item 6.

## 2. `modules/camera_calibration/curved_calibration/plate_model.py`

### 2.1 What it reuses from `cyl_model` (import)

`Camera` (R world→camera with OpenCV axes, C, f, cx, cy, width, height, dist[5]), `Rows`, `rows`, `distort` (with
jac), `undistort`, `pixels`, `skew`, `exp_so3`, `log_so3` (fixed), `rvec_of`, `rot_of`, `rotation_angle_deg`,
`CAM_PARAMS` (`w1 w2 w3 C1 C2 C3 logf cx cy k1 k2 p1 p2 k3`). Rotation increments are R ← R exp([w]x), so w is in
world axes, as in the cylinder.

### 2.2 The window

```
@dataclass Window:
    n: (3,) unit normal, camera -> object
    d: float          camera-side surface = {X : n.X = d}
    t: tuple          plate thicknesses, camera side first (usually one)
    idx: tuple        indices, camera side first: (n_cam, n_plate..., n_obj)   e.g. (1.0, 1.49, 1.333)
    far_d = d + sum(t)   object-side surface
```

- The C++ conventions differ, and `to_cpp` converts:
  - `plane.pt` is on the farthest (object-side) interface;
  - `refract_array` = [n_obj, plates reversed..., n_cam];
  - `w_array` = thicknesses, farthest first.
- Per-row window data: `WRows(n (m,3), d (m,), t (m,K), idx (m,K+2), has (m,) bool)`. A row with `has=False` is a
  plain pinhole camera. The pinhole stage uses that, so one code path serves both.

### 2.3 Forward projection `project(rw, ww, X, jac=False) -> PlateProjection`

Per row, with n̂ = n, n₀ = idx[0], the layers i = plates and the object medium:
1. v = X − C, a = n̂·v.
   - h₁ = d − n̂·C, the camera's distance to the window; it must be > 0.
   - h_W = n̂·X − far_d, the point's depth; it must be > 0.
2. ℓ = v − a n̂, ρ = |ℓ|, ê = ℓ/ρ. If ρ < 1e-12 (1 + |v|): ê is any unit vector ⊥ n̂, and τ/ρ := 1/F′(0).
3. g(τ; n) = n₀ τ / √(n² + (n² − n₀²) τ²), and F(τ) = h₁ τ + Σ_k t_k g(τ; n_k) + h_W g(τ; n_obj).
   - F′(τ) = h₁ + Σ h_i n₀ n_i² / (n_i² + (n_i² − n₀²) τ²)^{3/2}.
   - Start τ = ρ / F′(0), where F′(0) = h₁ + Σ t_k n₀/n_k + h_W n₀/n_obj.
   - Newton τ ← τ + (ρ − F)/F′ until |Δτ| ≤ 4e-16 (1 + τ), at most 60 iterations.
   - This converges monotonically when every layer index is ≥ n₀ (SymPy, `p_sympy_presize.py`). Otherwise mark the
     row as failed; it never happens for an air camera.
4. q = n̂ + (τ/ρ) ℓ, the air-side direction, not normalised. Then p = R q; it must have p_z > 0. Then x = p_x/p_z,
   y = p_y/p_z, distortion, and pixels.
5. **ok** = h₁ > 0, h_W > 0, p_z > 0, everything finite, and Newton converged.

**Derivatives** (jac=True). Let e₂ = n̂ × ê, g_W = g(τ; n_obj), F′ = F′(τ), and A = the window's anchor point (the
parameterisation of section 3.2: d = n̂·A + δ, so changing n̂ pivots the plane about A). Then:

```
J_qX = ê (ê − g_W n̂)ᵀ / F′ + (τ/ρ) e₂ e₂ᵀ
J_qC = ê (τ n̂ − ê)ᵀ / F′ − (τ/ρ) e₂ e₂ᵀ
J_qδ = ê (g_W − τ) / F′
J_qn = I + ê [−a ê − τ (A − C) − g_W (X − A)]ᵀ / F′ − (τ/ρ) [n̂ vᵀ + a (I − ê êᵀ)]     (w.r.t. dn̂; times T = [t₁ t₂])
J_pw = −R [q]×,   J_p• = R J_q•
dx = (dp_x − x dp_z)/p_z, dy = (dp_y − y dp_z)/p_z
u = f xd + cx: du/dlogf = f xd, du/dcx = 1, du/ddist = f D_dist, and du/d(x,y) = f D_xy
```

- **Where they come from:**
  - dτ = (dρ − τ dh₁ − g_W dh_W) / F′, with the thicknesses fixed;
  - dρ = ê·dv − a ê·dn̂;
  - dℓ = P_n dv − (v·dn̂) n̂ − a dn̂;
  - dh₁ = dd − C·dn̂ − n̂·dC;
  - dh_W = X·dn̂ + n̂·dX − dd;
  - dd = A·dn̂ + dδ.
- **At ρ → 0** the combinations are finite. ê ê ᵀ/F′ + (τ/ρ) e₂e₂ᵀ → P_n/F′(0), and the J_qn terms in ê cancel to
  −(a/F′) I. So any ê ⊥ n̂ works there, with τ/ρ = 1/F′(0).
- **Returned** per row:
  - `uv` (m,2), `ok`;
  - `J_cam` (m,2,14), in CAM_PARAMS order;
  - `J_win` (m,2,3), for (a, b, δ);
  - `J_X` (m,2,3);
  - `tau`, `h1`, `hW`.
- **Pinhole rows** (has=False): q = X − C, J_qX = I, J_qC = −I, and no window columns.
- **W0** checks these formulas in SymPy (implicit τ as a function symbol; n̂ = e_z without loss of generality). **W1**
  checks them against central differences.

### 2.4 Back-projection `trace(rw, ww, xy) -> (o, d, ok)`

- Undistorted normalised (x, y) → q = Rᵀ(x, y, 1), normalised.
- For each interface in turn (n̂·X = d, then d + t₁, ...): intersect, then refract with vector Snell. With
  μ = n_old/n_new and c_i = n̂·q > 0, take c_t = √(1 − μ²(1 − c_i²)) and q ← μ q + (c_t − μ c_i) n̂.
- **Returns:** o on the object-side surface, d in the object medium (unit), and ok. ok needs: c_i > 0, the camera on
  the camera side (h₁ > 0), and no total internal reflection.
- Pinhole rows: o = C, d = q.

### 2.5 C++ helpers

- `to_cpp(cam, win, tol=1e-12, nmax=1000)`:
  - `lpt.Camera()`;
  - `setPinplateImageSize(H, W)`;
  - `setPinplateIntrinsics(f, f, cx, cy, dist5)`;
  - `setPinplateExtrinsics(Pt3D(rvec_of(R)), Pt3D(−R C))`;
  - `setPinplateRefraction(Pt3D(far point), Pt3D(n), [n_obj, plates reversed..., n_cam], [t reversed], tol, nmax, 0.1)`;
  - `commitPinplateUpdate(True, 255.0)`, which is mandatory: `lineOfSight` reads `plane_array`;
  - then assert `max|_pinplate_param.r_mtx[i, j] − R| < 1e-14`.
- `cpp_project(cam_cpp, X)` goes through `projectBatchStatus(list of Pt3D, False)` → (ok, Pt2D, msg); a failure gives
  NaN.
- `cpp_rays(cam_cpp, uv)` goes through `lineOfSightBatchStatus(list of Pt2D)` → Line3D(pt on the far interface,
  unit_vector).
- **C++ facts** (reading, `src/srcMath/Camera.cpp`):
  - projection: a 1-D Snell solve, with a Levenberg–Marquardt fallback;
  - `proj_tol` is an absolute length in mm;
  - a point on the camera side of the far interface is measured with an unsigned distance (C:1059-1061), and probably
    returns its mirror image's pixel. W0 records what it does;
  - `lr` is unused.

## 3. `plate_wand.py` (same folder)

### 3.1 What it reuses from `cylinder_wand` (import as `cw`)

`WandData`, `Obs`, `observations`, `Frames`, `frames_from_ends`, `Eval`, `normal_blocks`, `reduced_system`,
`solve_step`, `cost_of`, `noise_groups`, `NOISE_MODES`, `_drop_thin_frames`, `affine_factorization`, `Affine`,
`_robust_cost`, `_subset_rows`. These are all model-free.

Re-implemented for plates: everything that calls a projection, adapted from the cylinder's code:
- `evaluate`, `bundle_adjust`, `_step_size`;
- `refine`, `variance_components`, `_update_active`, `_log_likelihood_key`;
- `information`, `covariance`;
- `triangulate`, `triangulate_ml`, `wand_lengths`;
- `point_jacobians`, `predicted_projection_error`.

There is no multi-image guard: a flat window gives one image.

### 3.2 The rig and the parameters

- **`Rig`:**
  - `windows`: a list of `Window`;
  - `cam_win`: an int per camera, the index into windows, or −1 for a pinhole camera (the pinhole stage).
- **`Layout(n_cams, n_wins, fit_k1=False, fit_L=False, fixed_windows=())`:**
  - **Per camera:** w1, w2, w3, C1, C2, C3, logf, plus k1 when fitted.
  - **Camera 1 (index 0):** the six pose columns are not free (the gauge). Only logf (and k1) are free.
  - **Per window:** a, b, δ, in the tangent basis (t₁, t₂ ⊥ n̂) and about the anchor A_w.
  - **Plus:** L.
  - **Optional soft priors** (residual rows):
    - log f_c around a hint, with its σ;
    - a camera–window distance with its σ.
- **`apply(cams, rig, L, dx)`:**
  - R ← R exp([w]x), C ← C + dC, f ← f e^{dlogf}, k1 += dk;
  - n̂ ← exp_{S²}(a t₁ + b t₂) (as `Frames.retract`), then d ← n̂_new·A + (d_old − n̂_old·A) + dδ;
  - A_w is fixed per fit: the mean, over the window's cameras, of where their principal rays cross the plane.
- **The frames:** the cylinder's midpoint m and direction w on S². The ends are m ∓ (L/2) w. There are 5 columns,
  Schur-eliminated.

### 3.3 `evaluate(cams, rig, frames, L, obs, layout=None, rows=None) -> cw.Eval`

The cylinder's structure:
- X per row from the frames;
- `pm.project` for all rows;
- r = uv − obs.uv.
- **Jg:**
  - the camera columns from J_cam (CAM_PARAMS index → layout position);
  - the window columns from J_win;
  - the L column = J_X·w·(∓½).
- **Jf** = [J_X, J_X t₁ s, J_X t₂ s], with s = ∓L/2 (as `cw.evaluate`).
- Also return h₁ and h_W per row, for the reports and checks.

### 3.4 Levenberg–Marquardt, refine, covariance

- **`bundle_adjust`:** copy `cw.bundle_adjust`. The LM loop, the trial-step rejection when a projection fails, and
  `good |= ev.ok` are unchanged. It stops on `_step_size` < 1e-10.
  - `_step_size`: rotation in rad; C in mm per max(|C|, 1); logf; k1; window a, b in rad; δ in mm per max(|A − C̄|, 1);
    frames in mm per 10, and rad.
- **`refine`:** copy `cw.refine`. Remove the multi-image guard. The noise weights default to "camera".
- **`covariance`:** the Schur-reduced information S/σ², as in the cylinder. It is expressed in the camera-1 gauge.
- **`camera_sigmas` / `window_sigmas`:**
  - per camera: rotation (deg), centre (mm), f (relative and px), and the distance to its window (mm);
  - per window: the normal (deg, as √(σ_a² + σ_b²)), δ (mm), and each camera's distance to it.

### 3.5 The start (stages 1–3)

**Stage 1, affine:**
- `cw.affine_factorization(obs, L)` needs ≥ 12 frames seen whole by every camera.
- If there are fewer:
  - greedily drop the camera whose removal most increases the count of frames seen whole, until the subset has ≥ 12;
  - factorise the subset;
  - add the others in stage 2, by DLT resection on points triangulated from the subset's pinhole cameras (≥ 6 points;
    centred principal point, f from the DLT), then the pinhole bundle adjustment.

**Stage 2, pinhole** (`pinhole_start`):
- For each mirror image of the affine reconstruction (P = I or diag(1, 1, −1)), and each camera:
  - polar decomposition M = U S Vᵀ, with O = U Vᵀ;
  - R = [O₀; O₁; O₀ × O₁];
  - the scale s = √(s₁ s₂) px/mm;
  - q_c = pinv(M)(c − t), on the principal ray.
- Its distance D along the principal ray:
  - **with an f hint:** D = f/s;
  - **otherwise:** a 1-D search. A geometric grid of 24 values from 3× to 3000× the cloud's RMS radius, with C = q − D z
    and f = s D. Score each camera's own pinhole reprojection RMS on the affine points, pick the minimum, then a
    golden-section refinement in log D.
- Then the pinhole bundle adjustment:
  - all cameras `cam_win = −1`;
  - the gauge is camera 1's pose;
  - f is free;
  - the frames come from the pinhole triangulation;
  - `refine` with outlier rejection, k_sigma 5. The refraction misfit (up to tens of px) inflates the robust σ, which
    is fine: only gross errors go.
- Keep both mirror images if their final RMS are within a factor 1.5; else keep the better one.

**Stage 3, windows** (`window_search`), one window at a time:
- **The normal:**
  - the dominant eigenvector of Σ z_c z_cᵀ over the window's cameras (sign-free; the 2026-09-17 lesson);
  - orient it so that n̂·(X̄ − C̄) > 0;
  - the candidates are a Fibonacci cap of half-angle θ_cap around it, at spacing Δθ.
- **The distance:** d ∈ (max_c n̂·C_c, min_j n̂·X_j − T), at fractions φ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}. Skip a normal
  whose interval is empty.
- **The start of each camera** behind the window (small-angle conversion):
  - D′ = far_d − n̂·C′, the pinhole centre's distance from the water-side surface;
  - h₁ = n₀ (D′/n_obj − Σ t_k/n_k);
  - if h₁ ≤ 0.02 D′, set h₁ = 0.02 D′;
  - C = C′ + (D′ − h₁ − T) n̂;
  - f = f′ n₀/n_obj;
  - R = R′.
- **The resection:** an LM over (w, C, logf) with the points held at the stage-2 ends. Exact Jacobians, with soft-L1
  weights at f_scale 2 px (IRLS).
  - Then re-triangulate the window's frames: the window's cameras refractive, the others as they are.
  - Then run the resection once more.
- **The score:** the sum of the window's cameras' `_robust_cost`.
- **Coarse to fine:**
  - first pass: every candidate, a 4-iteration resection on 60 frames;
  - the top 8 get the full resection;
  - the top 3 are kept.
- **The joint coarse fit** (`coarse_fit`, adapted from `cw.coarse_fit`):
  - the back-projection residual (ray offsets × f/path, with the ends triangulated then set L apart);
  - over all free camera and window parameters;
  - scipy `least_squares`, trf, soft_l1, f_scale 2 px, re-anchored every outer round;
  - the combinations tried: every window's best, plus each window's second and third with the others at their best.
- **θ_cap, Δθ and the φ grid** are set by W4. The defaults until then: θ_cap 45°, Δθ 9°.

### 3.6 `calibrate(data, rig, hints=None, fit_k1=False, fit_L=False, k_sigma=5, n_final=2, noise_weights="camera", sigma_cam_px=None, more_starts_px=1.0, verbose=False)`

- Stages 0 → 3 give the ranked starts.
- Then `refine` on the best `n_final`, or on all of them when the best stage-3 RMS exceeds `more_starts_px` (the
  cylinder rule).
- **The winner:** most observations kept, then the likelihood key.
- Then the covariance, the output frame (section 3.7) and `info` (timings per stage, starts, candidate counts).
- `rig.windows` give each window's t and idx. Their n and d are ignored unless `hints["windows_given"]`, which skips
  stage 3; W3 and W4 use it.

### 3.7 The output frame (no residual changes)

- Z = window 1's normal (camera → water).
- Y: with ≥ 2 windows, along n₁ × n₂; with one, camera 1's image "up" (−R₁[1]) projected into the plane. Then X = Y × Z.
- The origin is the mean of the frames' midpoints.
- Cameras: R ← R Qᵀ, C ← Q(C − O). Windows: n ← Q n, d ← d − n_old·O. Frames: m ← Q(m − O), w ← Q w.
- The covariance stays in the fit's gauge (camera 1); the report says so.
- This matches the GUI's final alignment (`refractive_geometry.align_world_y_to_plane_intersection`).

## 4. `plate_wand_io.py`

- **`read_gui_csv(path)`:**
  - the header is `Frame, Camera, Status, PointIdx, X, Y, Radius, Metric`;
  - only rows whose Status starts with "Filtered" are read: Small = end A (0), Large = end B (1);
  - at most 2 per camera per frame;
  - returns `uv[frame, camera, end, 2]` (NaN where missing), the frame ids, the camera ids and the radii.
- **`write_gui_csv(path, uv, radii=None)`:** for the W6 synthetic data. Write the raw rows too, the same points with
  Status "Raw", so the GUI's loader is happy.
- **`RigConfig` JSON:**
  - the windows: thicknesses, indices camera side first, and an optional name;
  - `cam_to_window`;
  - the image sizes;
  - the wand length;
  - optional hints;
  - the camera names.
- **`write_camfile(path, cam, win, proj_err=None, ray_err=None)`:** the C++ PINPLATE format.
  - Line by line: "PINPLATE"; the projection error "mean,std"; the ray error "mean,std"; "n_row,n_col"; K (3 rows);
    dist (5, one token); rvec; R; R⁻¹; t; t_inv (= C); the far point; the normal; refract_array (one token); w_array;
    proj_tol 1e-6; proj_nmax 1000; lr 0.1.
  - Values at 17 significant digits.
  - Check: reload with `lpt.Camera().loadParameters(path)` or `lpt.loadCameraModel(path)` (W9 finds which works), then
    project.
- **`write_result(sol, folder, names, ...)`:**
  - the camFiles;
  - `result.json` (cameras, windows, σ, noise, rejected observations, timings);
  - `report.txt`: per camera noise and σ; windows; rejected frames; predicted precision; wand lengths.

## 5. Tests in `tests/plate_wand/`

**The helper module `prigs.py`:**
- paths and sys.path;
- `build_cpp(cam, win)`, which uses `plate_model.to_cpp`;
- the rig makers A–E and `random_rig(rng)`;
- `make_wand(...)`, the C++ projections;
- `align_to_truth(sol, truth)`, the camera-1 alignment;
- the metrics.

**The rigs** (0.1 px noise, 1,000 frames and 1280 × 800 sensors unless stated). Each window has 12.7 mm of acrylic
(1.49) and water (1.333). The cloud is a box in the water. The wand is 25 mm long, with random directions and both
ends inside the box.

| rig | windows | cameras | cloud |
|---|---|---|---|
| A, one wall | 1 | 4 at ±18° horizontally and ±12° vertically, 600 mm from the window along their chief rays, aimed at the cloud's centre; f such that the cloud fills about 80 % of the sensor (≈ 6,000 px) | 100 mm, centred 150 mm behind the window |
| B, two walls | 2 perpendicular (Z = 0 and X = 0 planes) | 2 + 2, each pair at ±15° | 100 mm, near the corner |
| C, square-on | 1 | 3 within 5° | as A |
| D, steep | 1 | 3 at 35–45° | as A |
| E, macro | 1 | 3 at ±15°, f ≈ 40,000 px, 1,100 mm | 16 mm (±8 mm), wand 10 mm |

**`random_rig(rng)`:**
- 2–6 cameras;
- 1 window (50 %), 2 (35 %) or 3 (15 %); the walls are those of a box tank;
- incidence 0–45°;
- the camera's distance to its window 150–1,500 mm;
- the cloud 30–300 mm, with the wand 0.2–0.35 of it, 30–300 mm behind the walls;
- f so that the cloud fills 50–90 % of the sensor, which gives 1,500–40,000 px;
- a roll of ±10°;
- every camera's chief ray hits its own window;
- at least 2 cameras see each end in ≥ 90 % of the frames;
- 500–1,500 frames.

**The tests:**
- **W0** (`w0_sympy_derivations.py`, `w0_model_vs_cpp.py`):
  - SymPy for the Jacobian formulas of 2.3;
  - 2,000 random camera/window/point triples, with 1 and 2 plates, incidence to 60°, ρ from 0 to 1e-6 mm, compared
    with the C++: projections within 1e-6 px, rays within 1e-10;
  - the C++ quirk for points on the camera side, recorded;
  - the log map at every angle.
- **W1** (`w1_jacobians.py`): every column of J_cam, J_win and J_X against central differences, relative error
  ≤ 1e-6; plus `evaluate`'s Jg and Jf against differences of the whole residual.
- **W2** (`w2_observability.py`): Cramér–Rao on A–E from the information at the truth:
  - σ of f, poses, window normal and distance, held-out wand length;
  - the 3-D error on unseen points (fixed and retriangulated);
  - points outside the calibrated depth.
- **W3** (`w3_truth_start.py`):
  - noise-free from the truth: RMS ≤ 1e-8 px, and the parameters equal to the truth to 1e-6 relative;
  - 40 noisy data sets per rig at 0.1 px: errors / Cramér–Rao in 0.8–1.25, z standard deviation in 0.9–1.1, no bias
    beyond 3 SE;
  - a sweep from 0.02 to 2 px on A (10 sets per level).
- **W4** (`w4_basin.py`): starts from the truth with the window's normal tilted 5–60°, the distance moved ±90 %, f
  ±40 % and the rotations 5–20°. Run the coarse fit and refine, and report the convergence rates. This sets θ_cap, Δθ
  and φ.
- **W5** (`w5_from_scratch.py`):
  - 50 random rigs at 0.1 px and 10 at 1 px;
  - success = the same minimum as the truth-started fit: an RMS difference below 1e-9 relative, and every parameter
    within 0.01 σ;
  - the time per run.
- **W6** (`w6_vs_gui.py`): the GUI pipeline headless (section 6), on A–D at 0.1 and 0.5 px, and on 10 W5 rigs.
  - Both calibrations get the same frames: only frames seen by every camera, because the GUI drops a camera missing
    from any frame.
  - Each run has its own parent folder, because of the stale bootstrap cache.
  - The time limit is 2 h per run. Runs go in parallel, single-threaded.
  - The metrics: time; failures; f; the 3-D shape error on unseen points (triangulated with each calibration, rigid
    best-fit to the truth); held-out wand lengths; the reprojection error of unseen points after the alignment.
- **W7** (`w7_bad_data.py`):
  - 2 % gross outliers (20–200 px);
  - 1 % swapped ends;
  - a camera missing from 40 % of the frames;
  - a camera with 5× the noise;
  - a criterion of within 1 σ of the clean answer, with honest error bars.
- **W8** (`w8_constants.py`): the thickness +10 %, the water index +0.002, L +0.1 %, an unmodelled k1 = 0.05. Fit with
  and without `fit_L`, and report what the residuals, the free length and the error bars show.
- **W9** (`w9_files.py`):
  - the camFile round trip through the C++, projections within 1e-6 px;
  - the CSV round trip, identical;
  - the report written.
- **W10:** no real data; recorded as skipped.
- **`tests/test_plate_wand.py`:**
  - the model against the C++ (small);
  - the Jacobians (small);
  - noise-free from the truth on rig A;
  - one rig from scratch (≤ 60 s);
  - the camFile round trip;
  - a `.gitignore` negation for it and for `tests/plate_wand/`, like the cylinder's.

## 6. Running the GUI's current calibration headless (W6)

```python
from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator    # imports PySide6, no QApplication needed
from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import RefractiveWandCalibrator
base = WandCalibrator(); ok, msg = base.load_wand_data_from_csv(csv)
base.wand_length = L; base.dist_coeff_num = 0; base.initial_focal = f_cam0
base.camera_settings = {cid: {'focal': f_true[cid], 'width': W, 'height': H}}
base.image_size = (H, W); base.cameras = {cid: {} for cid in cids}      # must NOT set base.cam_params
ok, cam_params, report, dataset = RefractiveWandCalibrator(base).calibrate(
    num_windows, cam_to_window, window_media, out_path=str(run_dir / "camFile"))
# window_media[wid] = {'n1': 1.0, 'n2': 1.49, 'n3': 1.333, 'thickness': 12.7}
```

- **To check against the code:**
  - the signature of `calibrate` (`refraction_wand_calibrator.py`, around line 1990);
  - its `verbosity` and `progress_callback` arguments;
  - the names of the exported camFiles (`<out_path>/cam{cid}.txt`).
- **Then** load the exported camFiles with the C++ loader and evaluate them like the new ones.
- **Known risks** (from reading):
  - a camera skipped in its P0 can raise a KeyError later;
  - `bootstrap_cache.json` in the out folder's parent is reused when the key matches (camera ids, frame count, L).

## 7. The output frame and the truth in the tests

- The fit's gauge is camera 1's pose. To compare with the truth, map the fit rigidly so that camera 1's pose equals
  the truth's (`align_to_truth`). Then compare every parameter directly, and the z-scores come from the gauge
  covariance.
- When the bootstrap's camera 1 differs from the truth, the alignment absorbs it.
- The 3-D shape error is gauge-free: the best rigid fit of the triangulated unseen points to the truth.

## 8. Expected numbers (from the sizing), to sanity-check the results

- **Square-on:** the window's distance is nearly invisible. Moving the window 100 mm is absorbed to 0.012 px at
  f = 12,000 px and 0.19 px at f = 3,000 px, with the points known. W2's σ of the distance on rig C should be large.
- **15–45°:** a 10 % change of the distance shows at 0.4–2.9 px, and a 2° tilt at 0.8–4.3 px.
- **The best pinhole:** f ×1.33 square-on; it misses by 2.5–28 px RMS at 15–45°.
- **The scale** is fixed only through the thickness (SymPy G1–G2), so W8's free length will be weak.

## 9. Checklist (update as phases finish)

- [x] Phase 1: the `cyl_model.py` fix, the pytest cylinder subset, the cylinder README. Done 2026-10-02:
  `w0_logmap.py` 1.7e-15 rad at every angle; `tests/test_cylinder_wand.py` 7 passed.
- [x] Phase 2: `plate_model.py`; W0 (SymPy, against the C++, the log map); W1. Done 2026-10-02:
  `w0_sympy_derivations.py` 77 PROVED, 2 SKIPPED; `w0_model_vs_cpp.py` 6.5e-11 px, rays 1.3e-13 (the C++ projects a
  point on the camera side as its mirror image); `w1_jacobians.py` all columns within 3e-7 (block 2: `evaluate`).
- [x] Phase 3: `plate_wand.py` core (Layout, evaluate, LM, refine, covariance, triangulation); W3 noise-free; W2.
  Done 2026-10-02: `w3_noise_free.py` RMS 1e-12 px, parameters at round-off, also from a perturbed start;
  `w2_observability.py` written and run. `prigs.py` holds rigs A-E (f comes out about 3,900-4,400 px at 80 % fill,
  not 6,000) and `random_rig`.
- [x] Phase 4: the start (stages 1–3, `calibrate`); W4; then W5. Done 2026-10-02: W5 runs 1–3 (`runs_v1/`–`runs_v3/`)
  found the problems listed in section 10, fixed; the final run on the final code: 46/50 at 0.1 px (one-window rigs
  34/34), 5/10 at 1 px (9/10 equivalent) -- below the plan's 49/50 and 10/10 (the report, section 5). W4 rerun on the
  final code: the same basin. Diagnoses: `d1_multiwindow.py`, `d2_groups.py`, `d3_single.py`.
- [x] Phase 5 (early): `plate_wand_io.py`; W9 passes (`w9_files_out.txt`).
- [x] Phase 6: W6 against the GUI (`w6_vs_gui.py`, `w6_run.py`): 18 cases, both calibrations, run headless in
  parallel; the table in `w6_vs_gui_out.txt` and the report, section 3.6.
- [x] Phase 7: W3 Monte Carlo and sweep (PASS); W7 (all scenarios at their data's minimum; one marginal miss of the
  1-sigma criterion, explained); W8 (a study).
- [x] Phase 8: `tests/test_plate_wand.py` (6 fast + 1 slow, all pass) and the `.gitignore` negation; the full pytest
  suite (243 passed, 7 skipped, 2 failed -- both known and unrelated: `test_p1_round4_radius_frame_mismatch` and the
  cylinder-oracle copy); the report `2026-10-02-plate-wand-report.md`; `README.md`; the plan's status; memory.

## 10. What changed during the implementation (deviations from sections 2–3, with the evidence)

- **Stage 2's distance (section 3.5).** The planned 1-D search scored each pinhole camera on the affine points; those
  points absorb perspective, so every camera fit best ~10x too far (rig A: 8–10 m for 0.77 m). Replaced by
  `_distance_from_magnification`: a wand at depth z looks L|w_perp|/|duv| = (D + z)/f, a line in z, with the wand's
  direction and depth from the affine points but its length the known L. Distances then land within 5–40 % of the
  apparent ones on A–E, and the slope's sign picks the right mirror image (wrong one: negative on every camera).
  The old search is the fallback.
- **The pinhole fit** runs on at most 300 frames and 4 rounds (it is only a start).
- **Stage 3's candidate cameras (`small_angle_camera`)** are put on the exact chief ray (the pinhole's principal ray is
  the water ray; traced back by Snell's law), looking along the air direction. Keeping the pinhole's rotation left
  an oblique camera 13 deg off at 44 deg incidence (`d3_single.py`).
- **Resections keep each camera's distance to the candidate window** (`resect(keep_h1=True)`). Left free, a narrow-view
  camera slides onto its window (h1 -> 0), where it acts as a pinhole in the water and fits any window nearly as
  well; the search then picked windows 30–45 deg off (`d3_single.py`).
- **The short-listed candidates are refined with their window** (`resect_window`: cameras and window together, points
  fixed). With accurate points an oblique single camera then lands within 0.7–2.3 deg of the true window.
- **The back-projection residual** now penalises a wand end behind the ray's start (on the camera side of the window).
  Without it the coarse fit placed a window beyond the wand (W5 seed 1041: coarse RMS 0.108 px, then stage 4 could
  project none of that window's observations and rejected them all).
- **Choosing among the starts:** only answers within 1.5x of the best RMS compete (then most observations kept, then
  the likelihood). A start stuck in a wrong minimum inflates its own noise estimate, rejects little, and won on the
  count (W5 seed 1032: 11.9 px chosen over 0.69 px).
- **When the first starts end in different minima** (RMS apart by more than 0.1 %), every start is tried.
- **Multi-window rigs (new `bootstrap_groups`).** Each window with two or more cameras is calibrated on its own as a
  one-window rig; the groups are joined by the rigid motion of their common wand positions; a camera alone behind
  its window is then placed against the points the others triangulate (`place_single_window`: no re-triangulation
  with that camera, which would pull the points to itself and flatter wrong windows). The whole-rig search failed
  on 4 of 9 multi-window rigs: the pinhole points carry a 3–12 px misfit that swamps a window's tilt.
- **Windows with one camera each (new `bootstrap_singles`):** alternating placements, three passes; the last pass's best
  candidates give several starts.
- **Windows with one or two cameras are also searched by the wand length (new `place_window_vp`).** Each candidate's
  cameras get a short least-squares fit on the back-projection residual (the wand re-triangulated with every camera,
  its ends set L apart), the other cameras held. A fit to fixed points cannot judge such a window: the points carry the
  placed cameras' own earlier state (with two cameras every candidate fitted its points to 0.02–0.07 px; W5 seeds
  1045, 1048). Neither search is right on every rig, so on a rig with such a window `calibrate` pools the starts of
  both ("length" and "fixed"), taken in turn, each search's best first (sorted together, the right start of seed 1031
  ranked seventh), and the final costs decide (`max_starts` 6).
- **A window whose cameras never see the same frames** cannot be calibrated as a group: `calibrate` then falls back to
  the whole-rig start (W7's "partial" case on rig B).
- **Per-camera noise weights with two cameras (stage 4):** equal weights. Each wand position's residual can be moved
  from one camera to the other, and the estimate fed on it: seed 2005 drifted from 0.95, 1.00 px to 0.14, 1.35 px over
  16 rounds, then rejected 11 % of good data (RMS 1.11 against 0.85). Also: the weights are judged as soon as they
  settle (1 %), and fall back to equal when they keep moving after six rounds; the fallback gets its own rounds.
- **Spacing (W4):** the final fit recovers a normal 30 deg off (45 deg on all but the steep rig D) and any offset
  across the camera–cloud gap, so 3 offset fractions (0.2, 0.5, 0.8) suffice; the cap is 50 deg (incidence up to 45
  deg plus spread), the spacing 9 deg.
- **Speed:** BLAS products in the normal equations and the projection (einsum had taken 40 % of the time).
- **`plate_model.project`** returns NaN Jacobians when no row projects (W5's TypeError).
- **W5's "same minimum" test:** the RMS within 1e-6 relative (not 1e-9): with per-camera weights estimated to 1e-6, two
  runs at one minimum differ in the unweighted RMS by up to ~1e-8; the 0.01-sigma parameter test is unchanged.
- **Hard cases found** (W5 run 1, `d1`–`d3`): a window seen by one camera nearly square-on with a long lens is barely
  determined even at the truth (seed 1001: sigma 2.1 deg and 438 mm) and the cost has several minima there; two-camera
  groups with long lenses (seed 1031: f 13,500 and 31,700 px) can calibrate wrongly on their own.
- **W5 run 3** (`w5_from_scratch_out.txt`): 46/50 at 0.1 px reach the truth-started fit's minimum -- every one-window
  rig (34/34), 11/13 rigs with camera groups, 1/3 with one camera per window; at 1 px 5/10 exactly and 9/10 an
  equivalent answer (within 3 sigma, or a lower RMS). The misses are all two-camera configurations with long lenses:
  1001, 1045 (two cameras through two walls), 1031, 1034 (a two-camera group, f up to 31,700-37,000 px), 2000 (two
  two-camera groups at 1 px, 3.9 sigma). Stopped there: the remaining cases need more than a better search (see the
  report).
