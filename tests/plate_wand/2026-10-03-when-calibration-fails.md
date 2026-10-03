# When the flat-window wand calibration cannot work: what to watch in a real experiment

Date: 2026-10-03.

**As a page, with two figures:** `2026-10-03-when-calibration-fails.html` in this folder (open it in a browser). It is
also at https://claude.ai/artifact/Ue22gHGbyjWDNn2GnRBAyf (private until you share it). `make_fails_page.py` builds it
from the saved results, with `figures/fails_template.html`.

Your request: "go ahead to try to fix it as much as possible. in the end, report when the calibration cannot work so
that i can pay attention to in the real experiments".

This report gathers, from every test of the flat-window calibration (`plate_wand.calibrate`), the conditions under
which:
- its answer can be wrong (it stops in a wrong minimum);
- its answer is right but imprecise;
- it was never tested.

For each it gives the evidence, how `calibration_report.txt` shows it, and what to do in the experiment. Section 9
lists what was changed today to fix as much as possible.
- **Data:** all synthetic. The detections come from pyOpenLPT's own C++ camera model, with Gaussian noise. No real
  wand data have been calibrated yet.
- **Sources:** the README's section 7 lists every test; the scripts and their outputs are in this folder.
- **Terms:** each answer is compared with the fit started from the truth on the same data.
  - **The same minimum:** every parameter within 0.01σ of it, the same detections kept.
  - **Equivalent:** every parameter within 3σ and the RMS within 2 %, or a lower RMS.
  - **Wrong:** neither.

  "Right" below means at least equivalent.

## 1. In short

1. **Wrong answers come from one rig feature: a window seen only by one or two cameras with narrow views.**
   - Narrow: each sees the wand under less than about 6°. That means a long lens, or a camera far from the wand's
     volume.
   - **Before today's changes:**
     - every wrong answer in W5 and R4 (5 of 120 cases) had such a window;
     - 5 of the 22 cases with one went wrong; none of the 98 others did.
   - **Now:** the calibration scans such a window's normal itself. That fixed 2 of the 5 (R4 3110, W5 1001), and
     changed no other answer (H5a: 118 of 120 exactly as before).
   - **Still wrong without hints (W5 1031, 1034 and 1045):** in each, two windows are wrong at once, or the fit gets
     pinned against a window.
   - **The worst rig tested:** two cameras in all, each alone behind its own window with a long lens (seed 1045). It
     stays wrong even with most rough hints.
2. **Window-normal hints help, and never made an answer worse.**
   - Within 20° of the truth they gave an answer at least equivalent to the global minimum in 45 of 50 runs on the
     five hard cases, every miss on seed 1045.
   - Over those 65 hinted runs, none was worse than without hints.
   - Over W5 and R4 with every camera hinted 0–10° off (H1b, 120 cases), none was worse. The three cases still wrong
     without hints were right with them.
3. **High detection noise on rigs with several windows** lets the fit stop pinned against a window (6 of the 8 pinned
   answers came at 1 px). The restarts fix most. Aim for about 0.1 px.
4. **The wand length is the only scale.** Measure it to 0.1 % or better: the data cannot check it.
5. **An unmodelled lens distortion hides in the residuals.** If the lens may distort, fit k1.
6. **Calibrate over the whole measurement volume.** The error grows outside it, 2× at twice the volume.
7. **`calibration_report.txt` now opens with a WARNINGS block** naming these conditions when it finds them.
   - At 0.1 px detection noise, every wrong answer in W5 and R4 had one (3 of 3), and only 3 of the 47 right answers
     did.
   - At 0.5–1 px almost every rig with several windows gets the noise warning, as it should.

## 2. When the answer can be wrong

| condition | what goes wrong | evidence | how the report shows it | what to do |
|---|---|---|---|---|
| **a window seen only by one or two cameras that all see the wand under < 6°** | the window's tilt shows only weakly. The cost has several minima: the truth's valley is a few degrees wide, and a wrong answer 10–50° away is a separate minimum | W5, R4: every wrong answer had one (5 of 22 such cases, 0 of 98 others; `p_weak_windows.py`). D8 mapped the minima | WARNINGS: "window k is seen only by ...", with whether the scan found a better answer; a large σ on that window (W5 seed 1001 at the truth: 2.1°, 438 mm) | put a second camera with a wider view behind that window; failing that, give window-normal hints (section 6) |
| ... **and two such windows wrong at once, or one camera per window** | the scan of one window's normal cannot leave the minimum. W5 1045 (two cameras, one per window, 29,100 and 12,700 px) stays wrong without hints and with most hints 10° or more off | H4b: 1031, 1034 and 1045 still wrong without hints; 1045 wrong in 8 of 13 hinted runs | the same warning on each window; often a high residual (see the next rows) | do not build a rig with one camera per window and long lenses |
| **a wrong answer that leaves large residuals** | the cost's wrong minimum fits the data badly: W5 1031 without hints ends at 3.6 px RMS from 0.1 px detections | W5 | WARNINGS: "the residuals are X px per coordinate ..."; the noise section shows every camera far noisier than the detector | compare with a still wand imaged over many frames: much smaller detection noise means the answer is wrong |
| **detection noise near 1 px on a rig with several windows** | the fit can stop pinned against a window: a wand end exactly on its water-side surface | 6 of the 8 pinned answers came at 1 px; the restarts fixed most. R4 3110 at 1 px was the one wrong answer left in R4; the scan now fixes it | WARNINGS: "the residuals are ..."; "a wand end lies on window k's water-side surface"; the restarts' line "IMPROVED" | sharper detections (focus, exposure, bright balls); more cameras per window |
| **a window-normal hint far from the truth** | nothing: a hinted answer is kept only when it fits better than the answer without hints | H3a, H4b: 0 of 65 hinted runs worse; H1b: 0 of 120 | WARNINGS: "the answer's window normal is X° from camN's hint" | if you trust the hint better than X°, doubt the answer |

## 3. When the answer is right but imprecise

| condition | effect | evidence | what to do |
|---|---|---|---|
| cameras' lines of sight within ~5° of each other | f, the distances and the 3-D shape 3–9× worse than with lines 20–40° apart; the GUI's current calibration failed on this layout | W2 rig C, `p_mixed_views.py`, W6 | spread the cameras: 20° or more between lines of sight, 35–40° better |
| two windows facing each other (180°) | 3-D shape 2.2× worse than through perpendicular walls | A1 | prefer perpendicular windows |
| long lenses (f ≈ 40,000 px) | f and the camera's distance trade: σ(f) 0.6 %, distance 6.7 mm; the 3-D shape is still the best (0.6 µm) | W2 rig E | nothing, if only shapes and distances matter |
| the scene's position along the viewing direction | known to 0.4–7 mm relative to camera 1; shapes to 0.6–26 µm | W2 | measure distances and shapes inside the volume, not positions relative to the cameras |
| points outside the calibrated volume | the shape error grows about linearly: 2× at twice the volume | W2 | move the wand through the whole volume, its full depth included |
| detection noise | every error scales with it (0.02–2 px) | W3 | about 0.1 px |
| fewer frames | errors expected to grow as 1/√frames (the Cramér–Rao bound; not swept in the tests, which used 500–1,500) | W2 (1,000 frames), W5 | about 1,000 frames with both ends seen by two or more cameras |

## 4. What does not matter (tested)

- **The window constants:** a plate 10 % too thick, or a water index 0.002 off, left the 3-D shape error within 1 µm
  (W8).
- **The angle between windows:** 30°–180° all reach the global minimum (A1: 56 of 56).
- **A camera looking straight through its window:** no loss, also next to angled cameras (`p_mixed_views.py`).
- **Steep views,** up to 45° incidence (W2 rig D).
- **Bad detections:** 2 % gross outliers, 1 % swapped ends, a camera missing from 40 % of the frames, one camera five
  times noisier: all handled (W7).
- **No frame seen by every camera:** handled (W7).

## 5. Not tested: unknown territory

- **Real data** (W10).
- **Lens distortion fitted from scratch:** k1 was fitted only from the truth (W8).
- **Windows of two or more plates in a calibration:** the model handles them (W0), but no calibration ran.
- **Other window angles with a single camera or with long lenses:** A1 used two moderate-lens cameras per window.
- **More than 3 windows or more than 6 cameras.**
- **Cameras without a window.**
- **Ball-centre bias through a window:** elliptical images, as the tube's T13 found.
- **Image-fixed patterns, noise varying with depth, a bending wand, unsynchronised cameras.**

## 6. Window-normal hints: when to give them and what they do

- **When:** a window is seen only by one or two narrow-view cameras. A first calibration's WARNINGS block names such
  windows.
- **What:** per camera, the normal of its window in the camera's own frame: x along the image rows, y down the image,
  z along the optical axis, either sign. A camera turned 30° to the left of square-on, about its vertical axis, sees
  (sin 30°, 0, cos 30°). In the rig file: `"hints": {"normals": [[0.5, 0, 0.866], null, ...]}`, null for a camera
  without one.
- **What the calibration does with them:** it builds the start three ways and refits each:
  - without the hints;
  - with each hinted normal held;
  - with a second search about each hint.

  It keeps a hinted answer only when it fits the data better. The final fit estimates every normal from the data;
  the hint only steers the start.
- **How accurate they must be,** on the five hard cases (H4b):
  - windows with two cameras: hints up to 20° off were always enough (3110, 1031); at 30°, 3 of 3 runs on 3110 and 2
    of 3 on 1031;
  - one two-camera window and two single long-lens ones (1034): up to 30° off, all 13 runs right;
  - one camera per window with long lenses (1045): right with an exact hint and in 2 of 3 runs at 5°, mostly wrong
    beyond.
- **Never worse:**
  - none of the 65 hinted runs on the hard cases was worse than without hints (H4b);
  - over W5 and R4, with every camera hinted 0–10° off, none of 120 cases was worse (H1b);
  - the three cases still wrong without hints were right with them: their hints happened to be within 1–10°, 1045's
    within 1°.
- **The report gives each hint's angle from the answer.** A large angle means a rough hint or a wrong answer.
- **Cost:** with hints the calibration takes a median 2.55× as long (1.2–3.4×, H1b).

## 7. The warnings in `calibration_report.txt`

`calibration_report.txt` opens with a WARNINGS block. `calibration_result.json` has the same list under
`"warnings"`, each with a `code`. Each warning gives its evidence and what to do.

| code | when | what it means | evidence; in H5a (120 W5 and R4 cases, final code) |
|---|---|---|---|
| `weak_window` | a window is seen only by one or two cameras that all see the wand under less than 6° | the window's tilt shows only weakly; the cost can have several minima, and the answer may be one of the wrong ones. The calibration has scanned that window's normal; the line says whether the scan found a better answer | before the scan, all 5 wrong answers of W5 and R4 had such a window (5 of 22 such cases wrong). H5a: on 2 of the 3 wrong answers and 19 of the 117 right ones; 2 of its 21 cases wrong |
| `pinned` | an active wand end lies within 1e-6 mm of a window's water-side surface | the fit may be held against the window rather than at a minimum | before the restarts, 8 of the 14 wrong multi-window answers were pinned, none of the 69 right ones. H5a: 1 wrong (1031), 5 right (two-camera windows at 1 px, pinned at their own best fit) |
| `hint_far` | a hinted normal is more than 15° from the answer's | the hint was off by that much, or the answer is wrong | H3a, H4b: a right answer lies about as far from each hint as the hint is off (hints 30° off: 28–31°); wrong answers lay up to 70° from theirs. Not raised in H1b (hints 0–10° off) |
| `same_way` | every camera's line of sight is within 10° of the others | f, the distances and the 3-D shape are poorly determined | rig C (5° apart): 3–9× worse. Not in W5 or R4 |
| `noisy` | residuals above 0.5 px per coordinate on a rig with several windows | either the detections are that noisy (answers pinned against a window came mostly at 1 px), or the answer is wrong: a wrong minimum leaves large residuals. A still wand imaged over many frames tells the two apart | 6 of the 8 pinned answers came at 1 px. H5a: 1 wrong (1031: 3.3 px per coordinate from 0.1 px detections); 56 right, all at 0.5–1 px |
| `few_frames` | fewer than 300 frames used | outside the tested range (500–1,500) | — |
| `wand_length_fitted` | the wand length was fitted (`fit_L=True`) | through flat windows it is not identifiable, so the scale is unchecked | W8 |

No warning does not mean the answer is right. It means none of the conditions the tests linked to wrong or weak
answers is present.

## 8. Checklist for the experiment

**Designing the rig**
1. **Every window: two or more cameras, at least one with a wide enough view.**
   - The wand's positions should span at least about f/9 pixels across that camera's image diagonal (f in pixels).
     That is a view of 6° or more: the angle is 2 atan(diagonal / 2f).
   - Example: with f = 20,000 px this means about 2,100 px. A 100 mm cloud seen from 1 m spans 5.7°: just too narrow.
   - If a window must have only one or two narrow-view cameras: plan window-normal hints for them, accurate to about
     5° if the window has a single camera.
2. **Never one camera per window with long lenses** (seed 1045's rig). It is the one rig the calibration could not
   solve even with most hints.
3. **Spread the lines of sight:** 20° or more apart, 35–40° better. Perpendicular windows give the best shape.
4. **Synchronise the cameras,** or move the wand slowly.

**Before recording**
1. **Measure the wand length** to 0.1 % or better (25 µm on 25 mm).
2. **Enter the windows' constants** as well as you know them: plate thicknesses and indices, from the camera side. They
   are not critical.
3. **Make the two ends distinguishable,** for example with two ball sizes.
4. **If the lens may distort,** plan to fit k1 (`fit_k1=True`).

**While recording**
1. **About 1,000 frames** with both ends seen by two or more cameras.
2. **The wand through the whole volume,** its full depth included.
3. **Sharp, well-lit detections:** about 0.1 px.
4. **Frames shared between the cameras of different windows:** at least 3 per pair of window groups, ideally many.

**After calibrating**
1. **Read the WARNINGS block** first. A `weak_window` or `noisy` warning on a rig with several windows is the main sign
   of a possibly wrong answer.
2. **Read each window's and camera's σ.** Degrees on a normal, or tens of mm on a distance, mean the data hold it
   loosely.
3. **Compare the fit's noise with a still wand's.** The fit's should not be clearly larger.
4. **Validate on something of known size:** a second wand, or a dot target triangulated through the window.
5. **Plot each camera's residuals over the image.** A pattern fixed to the image hides in the RMS.

## 9. What was changed today to fix as much as possible

All in `plate_wand.py` and `plate_wand_io.py`. Without hints, a rig with no weak window gives exactly the answer it
gave before: in H5a, 118 of the 120 W5 and R4 answers are unchanged, and the other 2 are better.

1. **Window-normal hints run three start paths,** each with its own restarts (H3):
   - without the hints;
   - with the hinted normals held;
   - with a second search about the hints.

   A hinted answer replaces the default one only when its capped cost is lower and it is another minimum or keeps the
   same observations. So a hint cannot make the answer worse.
2. **A weak window's normal is scanned** (H4): a 30°-wide grid of held fits about the answer, 3° apart, then the best
   minima released and kept only on the same terms.
   - Without hints, it fixes 2 of the 5 known wrong answers (R4 3110, W5 1001).
   - It adds about 2–13 min to a rig with weak windows (H4b, H5a: 32 calibrations at once; less on a machine running
     one).
3. **The report warns** (H5): seven warnings, each with its evidence and what to do.
4. **Tests:** H3a, D8, H4a, H4b, H5a and H1b (`2026-10-03-angles-and-hints-report.md`, sections 6c–6g), and pytest:
   12 fast checks and 4 slow, all passing.
5. **Nothing is committed.**
