# Case 20260727 / T0: the walls against the channel — plan

3 October 2026. **Status: approved on 3 Oct; done on 4 Oct.** The report is `2026-10-03-walls-report.md`. It follows
the case's first report, `2026-10-03-report.md`.

**Your choice after W4 (4 Oct):** "i still prefer center held, and i think 0.277 px error is good enough for real
experiment". So the principal point is held at the image centre in `calib_plate_wand_ref_centre` (W4c, report
section 6b). The variant with it fitted stays in `calib_plate_wand_ref`.

Your answers to section 7:
1. **The size:** the inside, as taken (your earlier note: "not 100% accurate, it might have some small erros").
2. **The aim:** "I only know that the cameras should somehow look at the center for this case but no exactly accurate".
   - So the aim at the centre is used only as an approximate reference (B), beside (A).
   - C3 and C4 already look at the middle of their wall in every fit (47 of 100 mm), which supports it.
3. **L1 and L2:** "Approve".
4. **The output folder** `Results\T0\calib_plate_wand_ref`: "Ok".

You wrote: "my channel has cross section area 15cm*10cm, the part that cam 1 and cam 2 look at is the longer edge; but
your 3d visualization does not seem to be this case? maybe for the hints, you can also assign their reference point as
well?"

Your follow-up, the same day:
- "the size i report is not 100% accurate, it might have some small erros".
- "i feel that we usually think there is no strong distortion on our lens, although i am not sure whether that is
  correct".
- "will you add that as an optional setting? and how are you going to add thost constraints? constrain on wall location
  or wand spread or something else?"

Sections 3, 5 and 7 answer these.

## 1. What the delivered calibration says about the channel

You are right, and the problem is in the calibration, not only in the drawing. The wand's ends in the delivered fit
(T2, all 21,872 ends, `tests/rig_3d_out.txt`) lie at these distances from each wall's water side:

| | nearest end | median | farthest end | a 150 × 100 mm inside allows |
|---|---|---|---|---|
| from wall 1 (C1, C2: the long side) | 0 mm | 30 mm | 85 mm | 0–100 mm: fits |
| from wall 2 (C3, C4: the short side) | 109 mm | 178 mm | 225 mm | 0–150 mm: does not fit |

- **Wall 1 fits.** The wand probably touched it (report, section 7), and its farthest end is 85 mm out, inside 100 mm.
- **Wall 2 does not.** The farthest end lies 225 mm from it, 75 mm beyond the far wall.
  - For the wand to fit, wall 2's water side must lie 75–109 mm further from C3 and C4 than the fit puts it.
  - A shift of 92 mm centres the wand along the long side.
- **The cameras' aim says the same.**
  - C3's and C4's optical axes meet wall 2 47 mm from wall 1, the middle of the 100 mm side.
  - C1's and C2's axes meet wall 1 167–169 mm from wall 2, which is outside the channel.
  - Aimed at the middle of the long side (75 mm), C1 and C2 would be 93 mm closer to wall 2. That is the same shift
    that centres the wand.
- **Every fit says the same** (`tests/w0_wall_bands_out.txt`, all 31 saved fits).
  - Each right fit (k1 fitted, RMS 0.26–0.28 px) needs wall 2 moved by 85–126 mm to centre the wand. It fits wall 1
    as it is, or with wall 1 moved by at most 4 mm.
  - The wrong minima (section 6 of the report) put the wand 0.8–1.2 m from at least one wall, which no channel allows.
- **Figure 2 hid it.** It draws each wall as a long slab from the corner and never draws the channel's other two walls.
- **If 15 × 10 cm is the outside,** the inside is 124.6 × 74.6 mm. Wall 2 would then be 101–109 mm off. Wall 1
  would conflict too: the wand's full 85 mm span across it would not fit in 74.6 mm, although its 1–99 % span (68 mm)
  would.

## 2. Why hints alone cannot fix it

- **Hints only choose starting points.** By design, a hint never overrides the data: a hinted start is kept only if it
  ends with a lower cost. Wall 2's cost has a single minimum along its normal (D1's profile on subset s4). A start
  placed at the channel's position would slide back.
- **The data push wall 2 to where it is.**
  - D1 held wall 2 80 mm further from its cameras and refitted everything else. The RMS rose from 0.27 to 0.39 px on
    two frame subsets, while the wand length error stayed at 0.007 mm.
  - The fit is statistically sure of wall 2's position (sigma 0.6 mm on a fifth of the frames), yet that position moves:
    - by up to 41 mm between frame sets and detection filters. The centring shift is 85 mm on the first half, 116 mm
      on the second, and 126 mm with the border-cut balls dropped;
    - by 20–55 mm, in either direction depending on the frames, when k1 is held at 0.
  - So something the model lacks is being absorbed into wall 2's position.
- **A prior of honest width would be ignored.** Against the data's own sigma, a reference good to ±5 mm would move the
  wall by about 1 mm. The reference has to be held as a constraint, or the model fixed so that the data agree with it.
- **A point on a wall fixes nothing by itself.**
  - The calibration's frame floats, so any two perpendicular walls can be placed with their corner anywhere.
  - A reference helps only once it is tied to a camera or to the wand. Examples: where a camera's image centre falls
    on its wall, or a camera's distance to its wall.
- **The images cannot locate the walls.** No camera sees a wall's edge or the corner: the background is a uniform
  backlight. A frame of each camera, and its minimum over 31 frames, show only the wand and its hanging nut
  (`tests/w0_full_frames.png`, from `tests/w0_full_frames.py`).

## 3. The main suspect: the lens model

- **What it fits:** `plate_wand` fits f and k1 per camera. The principal point stays at the image centre, and k2 and
  the tangential terms stay at 0.
- **How strong the distortion is:** the fit finds about 6 px of pincushion at the image corners, 0.4 % of the image's
  half-diagonal (k1 0.76–1.06).
  - That is small by photographic standards, so it agrees with your sense that the lenses have no strong distortion.
  - It is real at this precision. Fitting it lowered the reprojection RMS from 0.30–0.42 to 0.27–0.28 px, and the wand
    length error on unused frames from 0.0096 to 0.0067 mm, on every subset (report, section 5).
  - Refraction cannot be the cause. Beyond what a homography absorbs, refraction itself bends the images of the
    square-on cameras (C1, C4) by at most 0.06 px, and of the oblique ones (C2, C3) by at most 0.28 px
    (`tests/w0_refraction_distortion_out.txt`).
- **Why that matters for the walls:**
  - Those 0.1–0.3 px in the oblique cameras are the signal that places the walls. An unfitted lens term of 0.1 px can
    therefore move them.
  - Wall 2's position rests on effects of a tenth of a pixel: 80 mm costs only 0.12 px of RMS (0.27 to 0.39 px).
  - Holding k1 at 0 moves wall 2 by 20–55 mm, in either direction depending on the frames. So the lens model drives
    where wall 2 lands.
  - The parts of the lens model that are not fitted may account for the rest: the distortion's centre (the principal
    point) and k2.
  - The camera model already carries these terms and their derivatives. Only `plate_wand`'s list of fitted parameters
    leaves them out.
- **Other inputs matter less.** The water index and the wall thickness moved wall 2 by less than 4 mm (T6).

## 4. Tests

The tests are run in this order. Each test's script and results go in `tests/`, as before.

| test | what | judged by |
|---|---|---|
| W1 what the channel costs (case code only) | From the delivered fit, on every frame: hold wall 2 at the shifts that put the wand inside the channel (+75, +92 centred, +109 mm), and hold both walls centred (wall 1 about 8 mm towards C1/C2). Refit everything else, as D1 did on subsets. Record the RMS overall and per camera, f, k1, distances and angles, the baselines, and `evaluate.py` on every frame. Also draw residual maps: each camera's mean residual over an 8 × 8 grid of image cells, for the free fit, the centred fit and their difference. | The RMS cost of the channel on every frame. The image pattern that wall 2's position absorbs, and in which cameras. |
| W2 the lens model (needs L1) | From the delivered fit, walls free, refit with one extra lens term at a time: the principal point; k2; p1 and p2; then all of them. Then all of them again with wall 2 held centred. A test only: a term goes into W4 only if it lowers the error on unused frames and its value is plausible. | Whether a term lowers the RMS and moves wall 2 most of the way to the channel, or makes holding it there nearly free. Its values must be plausible (principal point within about 100 px of the centre, small k2), and the wand length must stay as good. |
| W3 the constraints (needs L2) | On every frame, from the delivered fit: (A) alone (the wand inside the channel); then (A) with your aim reference (B), if you have one (question 2). Then (A) from scratch on every frame, where T2's first run fell into a wrong minimum. | The wand inside the channel; where wall 2 ends within its band; the RMS and the wand length against W1 and W2. From scratch: whether the channel keeps the fit out of the wrong minimum without `--start`. |
| W4 delivery | T2 rerun on every frame with the lens model W2 chose and the constraints of W3. Then: `evaluate.py` on every frame, a held-out check (fit on the first half, judged on the second), the camFiles' round trip, and the comparison with the delivered calibration. | The wand inside the channel, and the measurement (wand length, reprojection) as good as the delivered fit or better. |

**W4 also updates the pages:**
- **Figure 2** draws the channel itself: all four walls at your dimensions. The far two are drawn faintly, because they
  are not in the model. It shows both the delivered fit and the new one.
- **The volume table** gives the wand's distance to all four walls.
- **The report** (`2026-10-03-walls-report.md`), the README, the page and its artifact are updated.

**If W2 finds nothing,** the constraints hold wall 2 inside the channel at a cost of about 0.1 px in RMS (W1 measures
it on every frame).
- **The measurement inside the volume:** D1 found the wand length unchanged when wall 2 was moved, so it should not
  suffer.
- **Positions relative to wall 2** move with the wall.
- **Your choice:** the best fit, or the fit that matches the channel.

## 5. Library changes, each needing your approval

Both are options that are off by default, so the current results are reproduced bit for bit. Each gets regression
tests in `tests/test_plate_wand.py` and an entry in `tests/plate_wand/README.md`.

- **L1, optional lens terms** (`plate_wand`):
  - `fit_lens` in `Layout`, `solve` and `calibrate`. It fits any of the principal point (cx, cy), k2, and p1 and p2,
    beside f and k1.
  - The parameter layout, the step scale, the sigmas and the report learn these terms. The camFiles already carry
    them.
  - Tests: a synthetic rig with a known principal point and k2 is recovered within its sigma; with the option off, an
    existing fit is unchanged.
- **L2, constraints from the channel** (`plate_wand`):
  - **Where they go:** `constraints`, a new argument of `calibrate` and `solve`, and options of `run_calibration.py`.
    They stay out of the hints. By design, a hint only chooses a starting point and never overrides the data, while a
    constraint must.
  - **Why they are held, not soft:**
    - The data's own sigma for wall 2's position is under 1 mm.
    - Yet frame sets and lens models move it by tens of mm: up to 41 mm between frame sets, and 20–55 mm with k1
      held.
    - So a soft prior of honest width (±5 mm) would move the wall by about 1 mm.
  - **(A) The wand stays inside the channel** (the wand's spread). It needs only the channel's size.
    - You give the inside width across each wall (100 mm across wall 1, 150 mm across wall 2) and a tolerance for the
      size's error (default 3 mm).
    - Each wall gets a far wall parallel to it, at that width plus the tolerance. Every wand end must lie between the
      two walls with its ball inside: the ball's centre at least its radius from either wall.
    - How it is enforced: an end that crosses a far wall adds a residual that grows steeply with the crossing and is
      zero while the end is inside. An end can therefore touch the far wall but not cross it. The near walls keep the
      fit's present rule (no end in or before a wall).
    - What it gives is a band, not a point. Wall 2 can sit anywhere in a band 34 mm wide (150 mm minus the wand's
      116 mm spread). Because the data push wall 2 towards C3 and C4, the fit will end with the farthest end against
      the far wall: the band's edge, a shift of about 75 mm, where 92 would centre the wand.
    - Size errors: a few mm of error in the width moves the wall by the same few mm.
    - It also rules out this rig's wrong minima. They put the wand 0.8–1.2 m from at least one wall. With (A) on, the
      report warns when the wand does not fit the channel, and an answer that breaks it is not chosen.
  - **(B) Reference points tied to a camera** (the wall's location). It needs a fact about the rig.
    - "aim": camera c's optical axis meets its wall s mm from the other wall's water side. An example is "C1 and C2
      look at the middle of the long wall" (s = 75 mm).
    - "distance": camera c is h mm from its wall. This is the library's existing distance prior.
    - How it is enforced: held as an equality (a very stiff residual). A width can be given instead, to make it a
      prior.
  - **The report** lists each constraint: whether it is active, its residual, and the band left for each wall. A wrong
    reference shows up as a large cost.
  - **Tests, on synthetic rigs:**
    - a constraint the truth satisfies leaves a right answer unchanged;
    - a broken one is met;
    - with no constraint, the results are unchanged.
- **For future recordings:** touch each wall with the wand for a second or two. With (A), a touch pins that wall to a
  millimetre or two (a ball's radius). Wall 1 probably fits the channel already because the wand touched it.

## 6. What stays as it is

- **The delivered calibration:** `Results\T0\calib_plate_wand` keeps it. A new calibration goes to a new folder.
- **The paper:** nothing in `paper\` is touched.
- **Git:** nothing is committed.

## 7. Questions

1. **The size.** I take 15 × 10 cm as the inside, within 3 mm. If it were the outside (inside 124.6 × 74.6 mm), the
   wand's 85 mm spread across wall 1 would not fit.
2. **The aim.** Do you know where the cameras were aimed during the recording, for example at the middle of each wall?
   Or can you measure it, if the rig has not moved since: a ruler on the wall and the live image give it to about
   2 mm. A tape distance from each camera to its wall is good to only 2–3 cm, because the pinhole lies somewhere inside
   the lens. Without any of these, I use (A) alone, and the report gives wall 2's band.
3. **The library.** Do you approve L1 (for the W2 test) and L2?
4. **The output.** Is `Results\T0\calib_plate_wand_ref` right for the new folder?
