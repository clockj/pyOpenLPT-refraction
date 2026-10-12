# Wand balls seen through the tube: their centres (plan, T13)

Date: 2026-10-02. Status: **approved by the user on 2026-10-02 ("go ahead and do the test"); implemented and
tested.** The results are in `2026-10-02-ball-centres-report.md`.
- **Inputs:** those of section 3 were not given, so the defaults were used: 1.0 and 1.5 mm balls, a 0.3 mm rod, all
  four lightings, and the plate case's measured blur.
- **Changes from this plan** (report, section 4):
  - a rod-masked ellipse detector;
  - an overlap guard (T13b2);
  - a symmetric-mask variant (T13b3);
  - pyOpenLPT's "reliable" detector run on a subsample;
  - T13d run, since T13c's best detector missed its criteria narrowly.

The body below is the plan as approved.

**The user's question:** "In real wand calibration, the sphere projected through cylinder will not be a perfect
sphere; I am not sure how will that affect the accuracy on the center measurement which will be used to find 2D
pairs. Is there any way to help with that?"

Scope:
- Tests and scripts in this folder only.
- No change to the library, the GUI's wand detector, the C++ or the case folder. The case's detector is copied
  here, not edited.
- Nothing in `D:\Refraction\paper`.

## 1. What is already known

**The geometry alone does not bias the centre** (pre-plan sizing, `t13_pre_sizing.py`, output
`t13_pre_sizing_out.txt`).
- **How it was computed.** The silhouette of an opaque ball is exactly the set of pixels whose ray, refracted into
  the water, passes within the ball's radius of its centre. The script finds its outline (1440 points) for every
  camera of the reference rig, with ball centres on a 5 × 5 × 5 grid. It compares the outline's centre with the
  PINCYL image of the ball's centre.
- **The shape:** the image is an ellipse 1.26–1.34 times wider (across the axis) than tall over ±8 mm, and 1.21–1.41
  over ±20 mm. A 1 mm ball is 50 × 39 px; a 2 mm ball is 101 × 78 px.
- **The centre:** the silhouette's area centroid, and the centre of an ellipse fitted to its outline, lie this close
  to the image of the ball's centre (largest over the grid and the three cameras):

  | ball diameter | ±8 mm (cam1's present common volume) | ±20 mm |
  |---|---|---|
  | 0.5 mm | 0.0001 px | 0.0007 px |
  | 1 mm | 0.0005 px | 0.003 px |
  | 2 mm | 0.002 px | 0.011 px |
  | 3 mm | 0.0045 px | 0.025 px |

  - The offset grows exactly as the square of the ball's size, and is almost all horizontal.
  - Against 0.1 px of detection noise it is negligible for balls up to 2 mm. No model-based correction is needed for
    the shape.
- **The plate case found the same for its dots:** 1.44 mm dot outlines moved by 0.003 px RMS, 0.009 px at most
  (`cases/cyl_20260930_T0_cam1/2026-09-30-report.md`, T6).

**On real images the errors came from elsewhere** (the same plate case, the same camera and tube):
- **Astigmatism.** The tube separates the horizontal and vertical foci. At best focus the left and right edges were
  ~12 px wide (10–90 %), and the top and bottom ~4 px. Both were blurred at the end of the traverse, and the residual
  tripled there.
- **Uneven edges.** A rim darker on one side pulled the centres by:
  - 0.03–0.08 px with its best estimator;
  - 0.10–0.19 px with an iso-contour (thresholded-blob) centroid.
- **Its best estimator:** an ellipse fitted to edge points taken along rays from the centre, each at half the local
  edge height (`dots.py`). On synthetic dots with even rims it had a bias of at most 0.016 px and an RMS of at most
  0.048 px, in all three blur states.

**pyOpenLPT's wand detector fits circles** (`modules/camera_calibration/wand_calibration/wand_calibrator.py`):
- **Edges:** Canny edges (whole pixels), kept in an annulus of 0.7–1.3 times the inscribed radius. One point per
  degree is kept, the one nearest the inscribed radius.
- **Fit:** a RANSAC circle (3 px threshold, at least 40 % of the angles covered), then a least-squares circle.
- **Fallback:** the C++ `CircleIdentifier`.
- **Ends:** the two balls are told apart by their radius.
- **What the tube does to it:**
  - with a 1.3 : 1 ellipse, the long-axis edges sit at the annulus's 1.3 limit (the inscribed radius is the short
    semi-axis);
  - a circle can match only part of the outline;
  - the edge points are whole pixels, with no sub-pixel position.

  How large its error is on these images is not known. It is the main open risk.

**"2-D pairs".** Matching the two balls across cameras needs only their labels (A, B), which come from their sizes.
Errors of a fraction of a pixel do not affect it. The centres' accuracy matters for the calibration itself.

## 2. Tests (T13)

| test | what | criterion |
|---|---|---|
| T13a | **Rendered wand images through the tube.** The rig is cam1-like: three cameras 45° apart, 1280 × 800 px sensors, a ±8 mm cloud. The reference rig (±20 mm) is the second case. **Rendering:** each pixel's coverage by the two balls and the rod comes from the exact refracted rays, supersampled. **Blur:** separate horizontal and vertical widths that grow away from focus, set to the plate case's measured edges. **Noise:** as in the plate case. **Lighting:** (i) backlit silhouette; (ii) evenly lit matte balls; (iii) one side 30 % brighter; (iv) a small highlight. **Wand:** 1.0 and 1.5 mm balls, 10 mm apart, on a 0.3 mm rod. | Renderer check: the noise-free, blur-free silhouette centroids reproduce the pre-sizing offsets (to 1e-3 px); example images saved for you to compare with real ones |
| T13b | **The detectors on those images**, against the exact image of each ball's centre: (1) pyOpenLPT's current detector, imported unchanged; (2) the area centroid of the thresholded blob (Otsu and contour moments, as the GUI's dot picker), with and without the rod masked; (3) the plate case's estimator adapted to balls, with a robust ellipse fit to half-height edges, so the rod's edges are rejected. Reported for each: bias and how it varies with position and focus, RMS, largest error, failures, A/B told apart, overlapping balls rejected | For the recommended detector, with lighting (i) and (ii): bias ≤ 0.02 px, RMS ≤ 0.05 px (the plate case's levels), A/B told apart in every frame, and no centre returned for overlapping balls. The others are reported |
| T13c | **Effect on the calibration.** 500-frame data sets from the rendered images, calibrated from scratch with each detector and lighting. They are compared with exact centres plus noise of the same size. Measured: f, 3-D shape error, held-out wand length, error-bar honesty (z), the free-length check, the report's noise section | For the recommended detector: no bias beyond 3 standard errors in f or the wand length; z RMS ≤ 1.2. The others are reported |
| T13d | **Only if T13c shows a bias that matters:** a model-based correction. Render each detection's predicted image from the fitted calibration, run the same detector on it, subtract its offset from the measurement, and refit | T13c's criteria met on its worst case |

## 3. Inputs to confirm (defaults in brackets)

- The ball diameters and the rod diameter [1.0 and 1.5 mm balls, 0.3 mm rod].
- The lighting you plan to use [all four variants].
- The aperture and focus [the plate case's measured blur].

## 4. Deliverables

- `t13*_*.py`, with `_out.txt` and `.json` outputs and a few rendered example images (PNG);
- `2026-10-0X-ball-centres-report.md`, with the recommended detector and its measured accuracy;
- `README.md` updated.

Adding a recommended detector to pyOpenLPT, for example for a future wand CSV path or the GUI, would be a separate
step for your approval.

Cost: about 1 hour of computing in parallel.
