# Case 20260727 / T0: the walls against the channel — report

3–4 October 2026. **Status: done.** Plan: `2026-10-03-walls-plan.md`, approved on 3 Oct. It follows the case's first report,
`2026-10-03-report.md`. The tests' scripts and results are in `tests/` (`w0_*` … `w4_*`).

You wrote: "my channel has cross section area 15cm*10cm, the part that cam 1 and cam 2 look at is the longer edge; but
your 3d visualization does not seem to be this case? maybe for the hints, you can also assign their reference point as
well?"

## 1. In short

- **You were right.** The first calibration (`calib_plate_wand`) put the C3/C4 wall about 9 cm too close to C3 and C4.
  - The wand then reached 225 mm from that wall, which your 150 mm channel cannot hold.
  - C1 and C2 aimed 168 mm along their wall, outside the channel.
- **The two causes.**
  - **The fit was pinned against the C1/C2 wall.** A wand end sat on its surface, so every step was refused.
  - **Each lens's centre, the principal point, was held at the image centre.**
  - With both freed, the data alone move the C3/C4 wall about 70 mm towards the channel (W2).
- **The fix: two optional settings in `plate_wand`, both off by default.**
  - **Lens terms** (`fit_lens`): the principal point, k2, the tangential terms.
  - **Constraints** (`constraints`): the channel the wand stays in, where cameras aim, a camera's distance to its wall.
  - They are kept apart from the hints. A hint only chooses where a fit starts; a constraint overrides the data.
- **Your choice (4 Oct): `calib_plate_wand_ref_centre`** (W4c, section 6b). It holds the channel and the aim with
  each lens's principal point held at the image centre.
  - It is as accurate as the first calibration: 0.218 px median and 0.0068 mm wand error on every frame.
  - The wand lies 14–99 mm from the C1/C2 wall and 16–132 mm from the C3/C4 wall.
- **The variant with the principal point fitted, `calib_plate_wand_ref`** (W4).
  - **Settings:** k1 and the principal point fitted; the channel held; C1 and C2 aimed at the middle of the long side.
  - **Fit:** better on every measure (section 6), also on the half of the recording each half's fit did not use.
  - **The wand** lies 2–86 mm from the C1/C2 wall and 16–131 mm from the C3/C4 wall, inside the channel.
  - **A cross-check:** C3 and C4 aim at 47 of 100 mm, the middle of their wall, without being told to.
- **From scratch, with the channel held,** the calibration reached the right answer on every frame. Without it, the
  same data had ended in a wrong minimum (section 9).
- **What the walls' places rest on.**
  - The C3/C4 wall rests on the aim: an error in where C1 and C2 really look moves it by the same amount.
  - The C1/C2 wall rests on the wand touching it; the two halves of the recording differ by 16 mm there.
  - Measurements inside the wand's volume do not depend on either. The volume's size and the wand length are the same
    in both calibrations.

| measure | first: `calib_plate_wand` | channel, centre held: `calib_plate_wand_ref_centre` | channel, centre fitted: `calib_plate_wand_ref` |
|---|---|---|---|
| reprojection RMS, the fit's own | 0.2734 px | 0.2773 px | 0.2661 px |
| reprojection error, median, every frame | 0.2153 px | 0.2177 px | 0.2100 px |
| wand error, median / SD, every frame | 0.0067 / 0.0101 mm | 0.0068 / 0.0102 mm | 0.0065 / 0.0099 mm |
| the wand from the C1/C2 wall (channel: 100 mm) | 0–85 mm | 14–99 mm | 2–86 mm |
| the wand from the C3/C4 wall (channel: 150 mm) | 109–225 mm | 16–132 mm | 16–131 mm |
| report warnings | `pinned` | none | none |

## 2. The problem (W0)

**The delivered fit against the channel** (`tests/rig_3d_out.txt`, 21,872 wand ends):

| | nearest end | median | farthest end | the channel allows |
|---|---|---|---|---|
| from wall 1 (C1, C2: the long side) | 0 mm | 30 mm | 85 mm | 0–100 mm |
| from wall 2 (C3, C4: the short side) | 109 mm | 178 mm | 225 mm | 0–150 mm |

**W0's checks:**
- **No camera sees a wall's edge or the channel's corner.** The background is a uniform backlight. A frame of each
  camera and its minimum over 31 frames show only the wand and its hanging nut (`w0_full_frames.png`).
- **Refraction cannot pass for k1.** Beyond what a homography absorbs, refraction itself bends the images by at most
  0.06 px for the square-on cameras (C1, C4) and 0.28 px for the oblique ones (C2, C3). The fitted k1 moves the image
  corners by about 6 px. Those 0.1–0.3 px are also the signal that places the walls.
  (`w0_refraction_distortion_out.txt`)
- **Every saved fit says the same** (`w0_wall_bands_out.txt`, 31 fits).
  - Each right fit needs wall 2 moved by 85–126 mm to centre the wand.
  - The wrong minima put the wand 0.8–1.2 m from at least one wall.

## 3. What the channel costs with the first model (W1)

From the delivered fit, wall 2 was held at three shifts that put the wand inside the channel, and everything else was
refitted on every frame (`w1_channel_hold_out.txt`). Positive shifts are away from C3 and C4.

| case | shift of wall 2 | delta chi² | RMS | per camera C1 / C2 / C3 / C4 | median, every frame | wand error |
|---|---|---|---|---|---|---|
| free | 0 | 0 | 0.2734 px | 0.265 / 0.259 / 0.289 / 0.279 | 0.2153 px | 0.0067 mm |
| far end on the far wall | +77 mm | 113,438 | 0.3815 px | 0.288 / 0.267 / 0.469 / 0.457 | 0.2820 px | 0.0069 mm |
| centred | +92 mm | 160,932 | 0.4185 px | 0.300 / 0.269 / 0.527 / 0.512 | 0.3009 px | 0.0070 mm |
| near end on wall 2 | +107 mm | 218,272 | 0.4593 px | 0.313 / 0.271 / 0.589 / 0.572 | 0.3210 px | 0.0072 mm |

- **The cost is in C3 and C4.** The residual maps (`w1_residual_maps.png`) show an outward radial pattern of
  0.5–1 px in both, centred left of the image centre.
- **The wand length hardly moves:** 0.0067 → 0.0070 mm.
- **This overstated the cost.** The fits stayed pinned against wall 1, which W3 later showed.

## 4. The lens model (W2)

From the delivered fit, the lens model was extended one term at a time (`w2_lens_model_out.txt`). Every fit holds each
ball at least its radius (1.5 and 2 mm) inside every wall. That unpins the fit: without it, every fit on every frame
stopped after one step.

**Walls free, every frame:**

| lens terms (with f, k1) | delta chi² | RMS | the wand from wall 2 | shift that centres it | median, every frame | wand error / SD |
|---|---|---|---|---|---|---|
| none | 0 | 0.2732 px | 119–236 mm | +102 mm | 0.2152 px | 0.0067 / 0.0101 mm |
| principal point | −5,943 | 0.2663 px | 48–164 mm | +31 mm | 0.2100 px | 0.0065 / 0.0099 mm |
| k2 | −214 | 0.2729 px | 119–235 mm | +102 mm | 0.2150 px | 0.0067 / 0.0101 mm |
| tangential | −5,522 | 0.2668 px | 49–165 mm | +32 mm | 0.2103 px | 0.0065 / 0.0099 mm |
| all three | −7,114 | 0.2649 px | 93–209 mm | +76 mm | 0.2094 px | 0.0064 / 0.0097 mm |

**Fitted on every 5th frame, judged on the other 80 %:**

| lens terms | median | wand error / SD | shift that centres the wand |
|---|---|---|---|
| none | 0.2150 px | 0.0068 / 0.0102 mm | +100 mm |
| principal point | 0.2099 px | 0.0065 / 0.0099 mm | +23 mm |
| tangential | 0.2104 px | 0.0066 / 0.0100 mm | +19 mm (and wall 1 39 mm out of the channel) |
| all three | 0.2092 px | 0.0065 / 0.0098 mm | +79 mm |

**Verdict per term:**
- **The principal point is the term that matters.**
  - Its offsets from the image centre are plausible and the same on every frame and on the subset.

    | | C1 | C2 | C3 | C4 |
    |---|---|---|---|---|
    | every frame | −47, −16 | +17, −23 | −74, −81 | +5, +1 |
    | every 5th frame | −43, −20 | +23, −25 | −79, −84 | +2, +2 |

    Offsets are in px (right, down), ± 3 px on every frame and ± 6–8 px on the subset.
  - It lowers the error on unused frames by 2–3 %.
  - It moves wall 2 most of the way to the channel.
- **The tangential terms** do as much for the fit, but they move wall 1 out of the channel instead.
- **k2** does nothing.
- **All three together** need implausible values to hold wall 2 in the channel: principal points up to 480 px off,
  tangential terms of 3 %.

## 5. The constraints (L2) and W3

**How they work:**
- **The channel.**
  - **Inputs:** the inside width across each wall (100 mm across the C1/C2 wall, 150 mm across the C3/C4 wall), a
    tolerance for the size's error (3 mm), and the balls' radii (1.5 and 2 mm, from their image radii).
  - **The rule:** every wand end must lie at least its ball's radius inside every wall, and inside the far wall across
    it.
  - **How it is held:** a breach adds a residual row of the breach over 0.01 mm. This squared hinge is zero inside and
    steep outside, so the fit can touch a wall but not cross it.
  - **The start:** each stage-4 start is first moved along the walls' normals into the channel.
- **The aim.** The mean depth, below another wall's water side, of the points where the named cameras' optical axes
  meet their own wall. It is held as a stiff prior row.
- **The distance.** A camera's distance to its wall: the library's existing prior.
- **Why they are held, not soft:** the data's own sigma for a wall's position is well under 1 mm, while frame sets and
  lens models move it by tens of mm. A prior of honest width would be ignored.
- **The report:** `constraint_report` lists each constraint at the answer. For the channel: each wall's band of shifts
  that would still hold the wand, the ends on a limit, and the largest breach.

**W3** (`w3_constraints_out.txt`, `w3_constraints_pp_out.txt`), every frame through the case's calibration, from each
model's free fit:

| lens; constraints | RMS | median, every frame | wand error | wand from wall 2 | wand from wall 1 | C1, C2 aim |
|---|---|---|---|---|---|---|
| k1; none (the first calibration) | 0.2734 px | 0.2153 px | 0.0067 mm | 109–225 mm | 0–85 mm | 168 mm |
| k1; channel | 0.2760 px | 0.2167 px | 0.0068 mm | 36–151 mm | 16–101 mm | 94 mm |
| k1; channel and aim | 0.2773 px | 0.2177 px | 0.0068 mm | 16–132 mm | 14–99 mm | 75 mm |
| k1; aim only | 0.3148 px | 0.2417 px | 0.0069 mm | 16–132 mm | 0–84 mm (pinned) | 75 mm |
| + principal point; channel | 0.2675 px | 0.2100 px | 0.0065 mm | 36–151 mm | 2–86 mm | 95 mm |
| **+ principal point; channel and aim** | **0.2661 px** | **0.2100 px** | **0.0065 mm** | **16–131 mm** | **2–86 mm** | **75 mm** |
| + principal point; aim only | 0.3663 px | 0.2193 px | 0.0067 mm | 16–131 mm | 0–84 mm (pinned) | 75 mm |

- **The channel is almost free once the fit is unpinned.** With k1 alone it costs 0.003 px of RMS. The pin, not the
  data, made W1's cost so high.
- **The aim alone does not unpin the fit.** Without the balls' radii at the walls, both aim-only fits stay pinned and
  fit worse.
- **Channel alone or channel and aim.**
  - With the channel alone, the C3/C4 wall goes to the edge of its band, with the far end of the wand on the far wall.
  - The aim centres it, at no cost: 0.2661 against 0.2675 px.
- **From scratch, with the channel** (section 9): with k1 alone and with the principal point, the calibration reached
  the right answer on every frame. Without the channel it had fallen into a wrong minimum.

## 6. The delivery (W4)

- **Settings:** `run_calibration.py --fit-k1 --fit-lens pp --channel --aim centre --start tests/states/t2_all.pkl --out
  ref` (`w4_delivery.py`, 5.5 min).
- **Output:** `X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results\T0\calib_plate_wand_ref`, with
  camFiles, `calibration_result.json`, the report, `evaluation.txt` and `README.md`.
- **Unchanged:** the first calibration stays in `calib_plate_wand`.

| | first (T2) | with the channel (W4) |
|---|---|---|
| RMS, the fit's own | 0.2734 px | 0.2661 px |
| every frame: median, inliers RMS | 0.2153, 0.1890 px | 0.2100, 0.1845 px |
| every frame: wand error median, p90, SD | 0.0067, 0.0170, 0.0101 mm | 0.0065, 0.0165, 0.0099 mm |
| first half's fit on the second half: median, wand error | 0.2190 px, 0.0066 mm | 0.2128 px, 0.0065 mm |
| second half's fit on the first half: median, wand error | 0.2158 px, 0.0069 mm | 0.2103 px, 0.0067 mm |
| f, C1–C4 | 18,752 / 19,146 / 21,488 / 21,936 px | 18,915 / 19,260 / 21,482 / 21,830 px |
| principal point, C1–C4 (px, right, down) | the image centre, held | −48, −18 / +16, −25 / −76, −88 / +2, +18 |
| cameras to their walls | 1,067 / 1,034 / 772 / 845 mm | 1,075 / 1,038 / 848 / 911 mm |
| angles to the walls' normals | 0.70 / 12.80 / 18.10 / 1.24° | 0.65 / 12.92 / 16.34 / 0.37° |
| walls' angle | 90.17° | 90.18° |
| C1–C2, C3–C4 spacing | 242.6, 286.8 mm | 244.5, 285.7 mm |
| the wand from the C1/C2 wall, its far wall | 0–85, 15–100 mm | 2–86, 14–99 mm |
| the wand from the C3/C4 wall, its far wall | 109–225 mm; −75 to +41 mm (negative: beyond it) | 16–131, 19–135 mm |

- **The halves:** each half's fit with the channel beats the plain fit of the same half on the other half, in both
  directions.
  - Their principal points differ by 10–35 px, more than their sigmas. Treat them as known to a few tens of px.
  - The C1/C2 wall differs by 16 mm between the halves. The full fit agrees with the first half, which holds the
    frames where the wand touched that wall.
- **The round trip:** the camFiles read back give the fit's numbers exactly (`w4_camfile_roundtrip_out.txt`).
- **The volume:** its size is unchanged, 68 × 101 × 76 mm (1–99 %), hull 503 cm³. Only its place relative to the walls
  moved.

## 6b. Your choice: the principal point held at the centre (W4c, 4 Oct)

You wrote: "i still prefer center held, and i think 0.277 px error is good enough for real experiment".

- **Settings:** `run_calibration.py --fit-k1 --channel --aim centre --start tests/states/t2_all.pkl --out ref_centre`
  (`w4_delivery.py --centre`, 4 min).
- **Output:** `Results\T0\calib_plate_wand_ref_centre`. It is the same answer as W3's `w3_AB`.
- **The variant with the principal point fitted** stays in `calib_plate_wand_ref`.

| | first (T2) | centre held (W4c) | principal point fitted (W4) |
|---|---|---|---|
| RMS, the fit's own | 0.2734 px | 0.2773 px | 0.2661 px |
| every frame: median, wand error, SD | 0.2153 px, 0.0067, 0.0101 mm | 0.2177 px, 0.0068, 0.0102 mm | 0.2100 px, 0.0065, 0.0099 mm |
| first half's fit on the second half | 0.2190 px, 0.0066 mm | 0.2232 px, 0.0066 mm | 0.2128 px, 0.0065 mm |
| second half's fit on the first half | 0.2158 px, 0.0069 mm | 0.2167 px, 0.0069 mm | 0.2103 px, 0.0067 mm |
| the wand from the C1/C2 wall | 0–85 mm | 14–99 mm | 2–86 mm |
| the wand from the C3/C4 wall | 109–225 mm | 16–132 mm | 16–131 mm |
| C3, C4 aim, from the C1/C2 wall | 47 mm | 62 mm | 47 mm |
| C1–C2, C3–C4 spacing | 242.6, 286.8 mm | 242.9, 288.2 mm | 244.5, 285.7 mm |
| report warnings | `pinned` | none | none |

- **Accuracy:** the centre-held version is as accurate as the first calibration on every frame. On the half it did not
  use, it is within 2 % of the plain fit of the same half. 0.28 px RMS is well within what particle tracking needs:
  matching tolerances are about 1 px, and calib_3 had 1.28 px.
- **Wall 1 differs between the two versions.**
  - With the centre held, the C1/C2 wall sits 13 mm further from the wand, and the wand no longer touches it.
  - With the centre fitted, the wand touches it at a ball's radius, as the earlier frames near that wall suggested
    (report, section 7).
  - C3 and C4 aim at 62 mm of 100 instead of the middle, 47.
  - So distances to the C1/C2 wall carry about 13 mm of model uncertainty. Measurements inside the volume do not.
- **The round trip:** the camFiles read back give the fit's numbers exactly (`w4c_camfile_roundtrip_out.txt`).

## 6c. Do the balls look like circles? (D4, 4 Oct)

You asked: "do they still look like circle or ellipse? that might bring into some error which cause the 0.27 px".

**The test** (`d4_ball_image_shape_out.txt`):
- **The ball:** a 2 mm ball (ball B) through the delivered calibration, with perspective, refraction and k1.
- **Where:** 9 image points per camera, at depths of 10–120 mm in the water.
- **How:** its image is the hull of 20,000 projected surface points.

**Results:**

| camera | the image's elongation, at most | circle-fit centre from the image of the ball's centre, at most |
|---|---|---|
| C1 | 0.9 % | 0.014 px |
| C2 | 1.7 % | 0.020 px |
| C3 | 2.4 % | 0.030 px |
| C4 | 0.8 % | 0.021 px |

- **The balls look like ellipses, slightly.** At the image corners k1's pincushion stretches them radially by up to
  about 0.9 %. The oblique view through the wall adds more in C2 and C3.
- **That does not move their centres.** An ellipse's centre is where a circle fit puts it, and the outline's centre
  lies within 0.03 px of the image of the ball's centre. That is a tenth of the 0.27 px RMS, and next to nothing in
  quadrature.
- **What the 0.27 px is made of:**
  - **Random scatter** of the detected centres: about 0.24 px per detection.
  - **A pattern across each image:** about 0.1 px (W1's residual maps). The principal point takes up part of it.
  - **The rod:** it pulls the ball centres along the wand by at most 0.07 px (`along_wand` in the calibration's
    console).
  - **To go lower,** improve the detection: blur, motion, the thread and rod on the ball's outline, image noise.

## 7. Library changes (L1, L2), approved on 3 Oct, not committed

- **`plate_wand.py`:**
  - `LENS_TERMS` and `fit_lens` in `Layout`, `solve`, `solve_obs` and `calibrate`.
  - `CONSTRAINT_SIGMA_MM`, `set_constraints`, `channel_terms`, `channel_bounds`, `into_channel`, `aim_depth` and
    `constraint_report`.
  - Channel rows in `bundle_adjust`, `information` and `capped_cost`; the aim prior in `prior_terms`; `constraints` in
    `solve`, `solve_obs` and `calibrate` (`info["constraints"]`).
  - The parameter layout, step scale and sigmas know the new terms.
- **`plate_wand_io.py`:** `fit_lens` in the result JSON; the lens terms and the constraints in the text report.
- **`tests/test_plate_wand.py`:** three new tests.
  - `test_lens_terms`: a known principal point, k2, p1 and p2 recovered on rig B.
  - `test_channel_rows_jacobian`: the channel rows against central differences.
  - `test_channel_and_aim_constraints`: a channel the truth satisfies changes nothing; a broken one is met at its band's
    edge; an aim is held.
  - The whole file passes, 21 tests with the slow ones (`tests/l12_pytest_console.txt`).
- **Defaults:** both options are off, and the existing tests are unchanged.
- **The case code:** `run_calibration.py` gained `--fit-lens`, `--channel`, `--channel-tol`, `--aim` (`centre` or
  `CAMS:WALL:MM`), `--aim-sigma` and `--out ref`. `config.json` holds `constraints` and `out_dir_ref`.

## 8. For the experiment

- **Use `calib_plate_wand_ref_centre`**, your choice. `calib_plate_wand_ref` is the variant with the principal point
  fitted.
- **Tape-measure** the C1–C2 and C3–C4 spacings: 243 and 288 mm.
- **For the next calibration of this rig:** `run_calibration.py --fit-k1 --channel --aim centre --start <a verified
  fit>`. From a verified start it takes 4–6 min; from scratch, about an hour (below).
- **Why from scratch takes about an hour.** The 55–70 min runs were the full calibration on all 10,936 frames, each on
  one CPU thread.
  - About 70 % of the time is stages 1–3, the search for both walls' orientation and position with nothing known: many
    candidate normals per wall, each with camera fits.
  - The rest is the final fit (about 16 min) and its restarts.
  - More threads barely help, because the work is memory-bound.
  - From scratch on every 5th frame takes about 18 min; then 4–6 min refine it on every frame.
  - Running the start search's candidates in parallel on several cores would be a library change; not done.
- **When recording the wand:**
  - Touch each wall for a second or two with a ball fully in every view. With the channel held, a touch pins that
    wall to a millimetre or two, and C1 and C2's aim is no longer needed.
  - Keep both balls inside every image.
- **The GUI** does not call these options yet.

## 9. From scratch with the channel (W3)

Every frame, no start given, the channel held; one thread each (`w3_constraints_out.txt`, `logs/w3_A_scratch*.txt`).

| run | result | RMS | walls | the wand from wall 1 / wall 2 | time |
|---|---|---|---|---|---|
| k1, no channel (T2's first run, report section 6) | wrong minimum, kind A | 0.586 px | 93.5° | (the cameras 0.2–0.3 m from their walls) | 65 min |
| k1, channel (`w3_A_scratch`) | **right**: the same answer as from T2's start (`w3_A`) | 0.2760 px | 90.18° | 16–101 / 36–151 mm | 55 min |
| k1 and principal point, channel (`w3_A_scratch_pp`) | **right**: the same answer as from W2's start (`w3_A_pp`) | 0.2661 px | 90.18° | 2–86 / 36–151 mm | 70 min |

- **The channel kept both calibrations out of the wrong minimum** that the same data reached from scratch without it.
- **Each answer matches the run started from a verified fit,** with the same walls, cameras and evaluation:
  - `w3_A_scratch` matches `w3_A`: 0.2167 px median, 0.0068 mm wand error;
  - `w3_A_scratch_pp` matches `w3_A_pp`: 0.2100 px, 0.0065 mm.
  - With the channel alone, both end with the wand's far end on the far wall across wall 2, the band's edge; the aim
    then centres it (section 5).
- **This is one data set.** It shows that the channel's limits can rule out a wrong start, as section 5 expected
  (the wrong minima put the wand 0.8–1.2 m from a wall). It is not yet a proof for other rigs.

## 10. Files

| file | contents |
|---|---|
| `2026-10-03-walls-plan.md` | the plan, your answers |
| `tests/w0_full_frames.*`, `tests/w0_refraction_distortion.*`, `tests/w0_wall_bands.*` | W0 |
| `tests/w1_channel_hold.*`, `tests/w1_residual_maps.png` | W1 |
| `tests/w2_lens_model.*` | W2 |
| `tests/w3_constraints.*`, `tests/w3_constraints_pp_*` | W3 |
| `tests/w4_delivery.*`, `tests/w4_calibration_console.txt`, `tests/w4_camfile_roundtrip_out.txt` | W4 |
| `tests/w4c_delivery_out.txt`, `tests/w4c_delivery.json`, `tests/w4c_calibration_console.txt`, `tests/w4c_camfile_roundtrip_out.txt` | W4c, your choice |
| `tests/l12_pytest_console.txt` | the library's tests |
| `tests/states/w*_*.pkl` | the fits (not tracked by git) |
