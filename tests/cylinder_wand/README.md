# Cylinder wand calibration (PINCYL): summary and progress

Last updated: 2026-10-15.

Start here to catch up. This file gathers what all the wand tests so far concluded, and what that means for a real
experiment.
- It is kept up to date: each new result on the wand calibration adds a dated entry to the progress log (section 1,
  newest first) and revises the sections it changes.
- The numbers come from the reports listed in section 7, which name the scripts and data behind them.

**In short.**
- On synthetic data the calibration works:
  - it finds the right cameras from scratch with no camera input, focal lengths included;
  - it is as precise as the data allow;
  - its error bars are honest.
- That holds while PINCYL and the tube constants describe the optics exactly.
- The main risk for real data is a model error fixed to each image (section 3, item 8). The real cam1 plate case
  already shows it.
- **First real wand data (6 Oct, two long-lens cameras, runs T0 and T2):**
  - **The fit:** 0.45–0.60 px, usable inside the wand's volume.
  - **f is not determined:** the parts of each run disagree by ±10 %.
  - **The cameras moved by about 1 px between the two runs**, so each run needs its own calibration.
  - **The final fit looped on these runs** (10–14 min, and 356 good detections left out of T0's fit). Fixed on 6 Oct,
    with a four-round flip caught on 7 Oct: T0 now takes 72 s and T2 158 s with six starts, under 1 min with the
    GUI's default two. T0 is redelivered; its checks agree with the first delivery within their noise. Both runs'
    variants were rerun on 7 Oct.
  - See the entries of 6–7 Oct below.
- **The GUI's wand tab (7 Oct):** ten bugs are fixed, and then the two Pinhole-model bugs found with them (the
  progress log's entries of 7 Oct). The "fast" detection mode is kept, and now repeatable (your choice). The results
  page now shows the distribution of the reprojection errors.
- **Third real run (20261007/T0, 7 Oct, `cases/cyl_20261007`):** 0.600 px, f again undetermined (±12–16 % over the
  parts), the rig changed since 6 Oct. Two new things: a **hold-f option** (`fix_f`), and a **selection problem**
  with six starts, where a wrong minimum won (the entry of 7 Oct, late night).
- **The GUI's detection (13 Oct):** each camera can now have its own Radius Range, Sensitivity, Rod opening and Rod
  mask, for cameras that see the balls at different sizes. How the points were detected is recorded beside them
  (`<points>_detection.json`), in `calibration_result.json` and on the results page.
- **Run 20261013/T0 (15 Oct, `cases/cyl_20261013`):**
  - **The fit:** three cameras, 0.980 px over 9,139 frames.
  - **Between cameras:** they agree to 1.6–2.7 px, noisier than 8 Oct for reasons outside the settings.
  - **The cameras moved since 8 Oct.**
  - **The settings:** each camera needs its own rod opening (51, 57 and 85 px). The case report's section 3 says how
    to choose every setting.

## 1. Progress log

**2026-10-15: run 20261013/T0, three cameras through the same tube** (your question: "are we able to calibrate my new
experimental results with the same size cylinder in ...\20261013\ subfolder T0? how to choose the parameters"; then
"go ahead")
- **Where:**
  - the case: `cases/cyl_20261013/` (plan, report, scripts);
  - the results: `X:\Shijie Zhong\Refraction\Processed\20261013\T0` (README). Use `camFile_track\` for T1–T20.
- **The data:** 8 Oct's three cameras. All three saved images 0–12,000, so there is no offset this time.
- **The settings:** ellipse mode, Dark on Bright, a row per camera:

  | camera | radius range | opening | mask |
  |---|---|---|---|
  | cam1 | 20–70 px | 51 px | 30 px |
  | cam2 | 20–70 px | 57 px | 30 px |
  | cam3 | 30–105 px | 85 px | 44 px |

  The rod opening's working window is narrow (the 2 mm rod is 2/3 of the 3 mm ball) and differs between cameras. The
  report's section 3 gives the steps for choosing each setting.
- **The fit:**
  - 9,139 frames, 0.980 px (8 Oct: 0.703 px);
  - each camera predicted from the other two misses by 1.6–2.7 px RMS (8 Oct: 1.1–2.2);
  - the wand length's error has an RMS of 72 µm.
- **Why it is noisier than 8 Oct:**
  - **Both balls:** 30–50 % noisier in every camera.
  - **Not the settings:** variant fits with 8 Oct's aspect limit, with weights per camera and ball, or with k1 predict
    the cameras within 1 % as well.
  - **cam3's ellipse fits are noisier** (edge RMS 0.25 against 0.18 px), which points to its focus or light.
- **The cameras moved since 8 Oct:** 8 Oct's cameras miss today's balls by 29–72 px RMS (X1). Their layout and
  f / distance are 8 Oct's to within a degree and 1 %.
- **f is undetermined, as before:** the parts give −14 to +17 %, with f / distance within 0.8 %.
- **Tracking:** matching through all three cameras needs about 2.5–3 px.
- **Then (your question: "what if you only use the frames that all cameras can see for calibration? will that reduce
  the error?"): no.**
  - **The test:** a fit on the 5,596 frames all three cameras keep, the GUI's rule.
  - **The result:** on the same frames the cameras agree within 1 % of the delivered fit's, and the wand length is
    45 µm RMS for both.
  - **Where the error is larger:** the two-camera frames give about 100 µm with either fit (cams 1–3: 128 µm). That
    is the geometry of two cameras, not the calibration.
  - **So:** the GUI's rule gives the same calibration (report section 9).
- **Then ("so the main reason is that this experiment does not focus the wand points well, right?"): not mainly.**
  The answer differs by camera (report section 10).
  - **cam2:** its rise comes entirely from where the wand went (more stretched views). At the same stretch it is as
    good as on 8 Oct.
  - **cam3:** noisier at the same place, with rougher ellipse outlines: focus or lighting.
  - **cam1:** noisier at the same place, but its outlines are as clean as before. Not plain blur; possibly
    direction-dependent blur through the tube, or uneven backlight.
  - **Depth of field:** not the cause.

**2026-10-13: detection settings per camera in the wand tab** (your question: "i still cannot set object parameters
for each camera? for example, the wand opening, wand mask, radius range? i think that is important if my cameras have
different resolutions?")
- **Where:** `tests/wand_per_camera_detect/2026-10-13-plan.md` and `-report.md`.
- **New:** a box, "Different settings for each camera", above Test Detect. Ticked, a table gives each camera its own
  R min, R max, Sensitivity, Rod opening and Rod mask, in its own pixels.
  - **Before,** the only per-camera handle was the Pixel (µm) column, which scales the shared values by the pixel
    size. That is right only when the cameras differ in pixel size alone, not in magnification.
  - **The Sensitivity** shows only where it is used (your question: the ellipse mode does not use it). Opening and
    Mask show in the ellipse mode only.
  - **Test Detect, Process All Frames and the axis points' Detect** all use the table.
- **Nothing changes until a row is edited.**
  - With the box off, the tasks are exactly as before.
  - Ticking fills the rows from the shared settings and keeps their exact values: on six T2 frames, the same points
    to the bit in all three modes, with and without pixel sizes.
  - The circle finders' points can move by up to 0.5 px when a radius bound moves by 0.002 px, so the exact values
    matter.
- **Checked** in the GUI's window on T2 frames (17 of 17). With camera 1's row set below its balls, camera 1 finds
  none and camera 0 is unchanged, in Test Detect and in Process All Frames. The panel still fits its width.
- **Then (your OK: "record the detection settings on the results page and save it for reproduction"):** every
  detection now writes a record beside its points, `wand_points.csv` → `wand_points_detection.json`.
  - **It holds** the mode, the Wand Type, and each camera's settings as its detector got them, and its images.
  - **Resume and Export:** a resume adds an entry, and Export Data copies the record beside the exported file.
  - **Read back:** Load Wand Points reads it, so the "Detection mode unknown" note no longer appears for such files.
  - **Into the results:** the calibration saves it in `calibration_result.json`, and the results page lists it:
    Detection, per-camera rows, Images, and the record's path.
  - **Checked:** 12 of 12 (`r_record.py`, 24 T2 frames); report `2026-10-13-report-record.md`.

**2026-10-07 (late night): the third real run, 20261007/T0; a hold-f option; a selection problem** (the tracking
case's plan, Phase C: `cases/cyl_20261007/2026-10-07-plan.md`; report `cases/cyl_20261007/2026-10-07-phaseC-report.md`)
- **The calibration:** 3,036 frames, 0.600 px, wand length RMS 34.4 µm; cam1 f 48,494 px at 1,179 mm, cam2 28,306 px
  at 835 mm, 60.3° apart. Results in `X:\Shijie Zhong\Refraction\Processed\20261007\T0`.
- **f is undetermined again:** the halves and quarters give cam1 41,351–53,010 px and cam2 23,639–30,417 px. Inside
  the wand's volume each part's cameras measure the others' wand lengths to 0.02–0.12 %.
- **The rig changed since 6 Oct:** 6 Oct T2's cameras disagree by 9 px on this run's frames and measure the wand
  4.9 % long. The f / distance ratio of cam1 is 7.5 % higher than on 6 Oct.
- **A selection problem with the thorough start search** (six starts): a start reached a wrong minimum (cam1 f
  10,995 px at 253 mm, RMS 1.38 px) but kept 12,088 of 12,144 detections, against 11,876 for the right one (RMS
  0.600 px). `calibrate` ranks by the detections kept first, so the wrong one won: its larger residuals widen its
  outlier limit. The case used the GUI's default two starts. **Not fixed** (a library change for you to decide; GUI
  users with "Thorough start search" ticked are exposed). Section 6, item 9.
- **The hold-f option** (approved as the plan's C3): `calibrate(..., fix_f=(camera indices))` holds those cameras'
  f at their value in `start`; `Layout(fix_f=...)` leaves their log f out, `coarse_fit` keeps it, and
  `cylinder_wand_gui.Job.fix_f` passes it from a GUI job. Default empty: nothing changes (`test_cylinder_wand.py` 7
  passed). **T14** (`t14_hold_f.py`, 2 cameras like the real rig, 4 seeds): with f held at the truth the distances
  are within |z| ≤ 1.21 and the centres ≤ 1.83; held 1 % high, the distances move +0.98 % (along the valley) and
  the RMS and held-out wand lengths do not change.
- **On the real run,** the dot board (three stage positions, triangulated with the wand's cameras) is square only
  for one value of cam2's f: 29,035 px, 2.6 % above the free fit. Held there (and cam1 at the line's 48,306 px), the
  halves and quarters still disagree (distances ±2.2 %), so the free fit was delivered.
- **The board and the wand disagree in scale by 0.4–0.5 %** (the board's 3 mm pitch reads 3.011–3.015 mm).

**2026-10-07 (night): the results page shows the reprojection errors' distributions** (your message: "for the gui,
please add 2d projection error pdf for your wand calibration reports: du, dv, 2d magnitude"; then "do the optional
a,b")
- **Where:** `tests/wand_report_error_pdf/2026-10-07-plan.md` and `-report.md`.
- **A new section on every results page** (the Report tab, all three models). Per camera it shows:
  - the probability density of du, dv and |e| = √(du² + dv²), on a linear or log scale;
  - a table of their statistics;
  - du against dv in 2-D, the T0/T2 case pages' chart.
- **T0 and T2 on X:** their `calibration_report.html` is rendered again with the new section. I checked before
  writing that nothing else on them changed.
- **What it shows on the real runs:**
  - **du is about twice as wide as dv** in both cameras, as the case pages found. T2: du 0.56 and 0.50 px, against
    dv 0.27 and 0.29 px.
  - **T2's du has a sharp spike at zero,** in both cameras at once. It is in the data; its cause was not looked into.
- **Checked:** the six pages draw in QtWebEngine without an error (34 of 34). No further tests (your call).

**2026-10-07 (evening): the Pinhole model's two bugs fixed; the "fast" mode measured** (your message: "fix B11 to
make it able to accept different size of image. For B12, fix it. for the "Fast" detection, shall we just remove
it?")
- **Where:** `tests/wand_gui_fixes/2026-10-07-plan-b11-b12.md` and `-report-b11-b12.md`.
- **Cameras of different image sizes now calibrate** (B11). A camera of 1024 × 768 px among cameras of 1280 × 800 px
  gives the answer of the rig of one size with the same noise, within the calibrator's own numerical noise. Cameras
  of one size are unchanged to the bit.
- **Cameras numbered with a gap now calibrate** (B12): the ids 0, 2, 3 give the answer of 0, 1, 2 to the bit.
- **The "fast" mode, measured on synthetic images of a wand in air:**
  - **fast:** 0.07 px median centre error, 0.13 s per image, two runs within 0.04 px;
  - **reliable:** 0.16 px and 4.2 s;
  - **ellipse:** 0.02 px and 0.16 s.

  Its big jumps (up to 27 px) appear only on tube images, where "ellipse" is the mode to use. **My answer: keep
  it,** and optionally make it repeatable with a fixed seed.
- **Your choice, "use 2.": done (B13).** The fast mode now draws its RANSAC samples with a fixed seed.
  - **Repeatable:** an image gives the same points every run, in any process, in Test Detect and in Process All
    Frames.
  - **The change, once:** its points moved by 0.04–0.06 px, as much as two unseeded runs differed.
  - **Accuracy:** unchanged (0.072 px median on D2's images).
- **The regression is as before:** the pinhole page and the results pages pass, and pytest has the same four old
  failures.

**2026-10-07 (later): the wand tab's bugs fixed** (your OK: "please go on and fix those bugs. for the
Precalibration, i think we should just remove that or show the results from the start stage?").
- **Where:** `tests/wand_gui_fixes/2026-10-07-plan.md` and `-report.md`.
- **All ten fixes are in.** Every test passes (W0–W11), and the regression matches F7's.
  - **"Bright on Dark" now detects bright balls.** It had run as "Dark on Bright". On inverted T2 frames it gives
    the originals' detections exactly (reliable, ellipse).
  - **The flat-window "Precalibrate to Check" is plate_wand's own check.** It runs the start search and coarse fit
    (stages 1–3) in a separate process, about 40 s, and writes no files. plate_wand's answers are bit-identical after
    the change.
  - **Deleted frames can be restored,** with a new button. New points start with none deleted, and a Del box clicked
    by hand survives a filter change.
  - **Pause pauses the detection:** during a 10 s pause only the 26 tasks in flight finished. Before the fix, 587 ran.
  - **The smaller fixes:**
    - Num Cameras keeps the remaining cameras' settings.
    - The indices take four decimals.
    - The panels fit their column.
    - Pinhole camFiles carry each camera's own image size.
    - The Pinhole error table shows the red crosses.
    - The axis-point notes are right.
- **Found, not fixed (your decision):**
  - **The Pinhole model cannot calibrate cameras of different image sizes:** it stops with an error (B11).
  - **It also fails when the cameras are not numbered 0, 1, 2, … without gaps** (B12).
  - **The "fast" detector is not repeatable:** two runs differ by up to 4.4 px.
  - Fixes are proposed in the report. (B11 and B12 were fixed that evening, and the fast mode measured: see the entry
    above.)
- **The user guide** is updated, with its screenshots retaken.

**2026-10-07: the four-round flip fixed; both runs' variants rerun; the install scripts fixed**
- **The fix** (your OK: "go on and fix them"): the flip test of `cylinder_wand.refine` now also fires when the noise
  estimate jumps by more than 30 % in each of four rounds in a row. See `tests/cylinder_wand_gui/2026-10-06-fixes-report.md`,
  section 9 (test F7).
  - **On the real runs' logs (100 starts):** it catches 60 of the 61 starts that ran to the round limit, most at
    round 4.
  - **The one it misses** (T0's q1, start 1) is not a loop but a slow fit: its RMS still falls every round. Its
    result is not the one kept.
  - **Synthetic (F4):** all 1,060 fits are unchanged, and the new rule never fires.
  - **The delivered calibrations (F3):** unchanged; f is within 0.003 px.
  - **Time:** T0 now takes 72 s and T2 158 s with six starts (96 s and 234 s last night). Part of that comes from a
    quieter machine; the fair measure is the rounds per start, down from 9 to 8 (T0) and from 10 to 9 (T2).
- **A correction to last night:**
  - **More T0 variants looped:** besides the two I named, one start each of q1, q3 and D2 ran to the limit.
  - **T2's own variants had never been rerun after the fixes of 6 Oct:** its halves, quarters and k1 fit. Every one of
    their starts had looped.
- **The reruns:**
  - **T0:** two variants changed. h2: cam1 f 35,229 px (34,548 before). fit_k1: 6,060 detections kept (5,704
    before), k1 +14.3 / +6.7. The rest changed only in the last digits.
  - **T2:** the 2nd quarter no longer puts cam2 at 4.7 m and f 166,557 px. It now lies with the other parts, at
    1.2 m and 41,408 px. T2's spread is cam1 36,500–40,900 px and cam2 32,100–41,400 px.
  - **Both pages republished**, and the reports and READMEs updated. The earlier outputs are kept in `before_flip4\`.
- **The install scripts (your question)** — see `tests/install_scripts/2026-10-07-plan.md` and `-report.md`.
  - **What was wrong:**
    - every installer stopped at its third step, because `requirements.txt` lists pycine, which is only on PyPI;
    - the Windows and macOS installers needed mamba, and this PC has only conda;
    - on Windows conda-forge's solve picks MKL, which crashes numpy here;
    - QtWebEngine (the Report tab) was in no list;
    - the scripts made Python 3.10, not the tested 3.11.
  - **Fixed, with your change "keep pycine":** its line is marked pip-only, and the installers pip-install it.
    - **Windows:** the installer works from scratch here in 2.7 min. The new environment has Python 3.11 and
      OpenBLAS, the GUI opens, and the Report tab shows a results page. The updater works too.
    - **Linux and macOS:** their package sets solve, and the scripts parse. They were not run.
- **The GUI's wand calibration user guide, rewritten** (your request).
  - **What it is:** `modules/camera_calibration/wand_calibration/WAND_CALIBRATION_USER_GUIDE.md` and its HTML, which
    the wand tab's Tutorial opens.
  - **What it covers:** every setting (what it is, its default, how to set it) for detection (fast / reliable /
    ellipse, rod opening and mask), the three camera models and the results.
  - **Screenshots:** six, taken from the real GUI on run T2 by `make_user_guide.py`.
  - **Your question:** the GUI's "Pinhole+Refraction" runs plate_wand, unless "Use the previous calibrator" is ticked
    (test G8). Its "Precalibrate to Check" still runs the plain pinhole check, and the Plate Calibration (dot target)
    tab keeps its own calibration.
  - **GUI bugs found while writing it (not fixed; proposals):** "Bright on Dark" behaves as "Dark on Bright"; the
    flat-window result's note about axis points is wrong; and smaller ones. They are listed in the guide's "Known
    issues". (Fixed later that day: see the entry above.)

**2026-10-06 (later still): T0 redelivered; is it better; the radius range in the ellipse mode**
- **T0 redelivered** on your OK, with every test of its case rerun and its page republished.
  - **The change:** 6,064 of 6,076 detections; RMS 0.47 px; 23 µm.
  - **Better?** Not measurably, and not worse. On all 1,519 frames the wand length is 23.4 µm RMS as before (D8). The
    cross-run step to T2 is unchanged (+1.01 px). The halves and quarters still spread f by about ±10 %.
  - **The real difference:** the fit now follows one rule for every detection. The first delivery's lower RMS came
    from leaving out the 356 worst-fitting detections.
- **Rebuild:** not needed. Your "OpenLPT (new)" shortcut runs the repo's `GUI.py`, so the GUI picks up the fixes at
  its next start. The C++ exe and module are newer than every C++ source.
- **The radius range does matter in the ellipse mode** (R1):
  - **What it does:** it sizes the background window (2.5 × the largest radius) and filters blobs by area.
  - **On 1,200 T2 frames:** 40–120 px paired 159 frames, 20–200 px 151, and 20–60 px (below the balls) 108. The
    centres moved by only 0.01–0.04 px.
- **Found while rerunning T0's variants:** a weight flip that repeats every four rounds (k1 free; one half) still runs
  to the round limit. The fix's flip test only catches two-round flips. A further fix is proposed, waiting for your
  OK (`tests/cylinder_wand_gui/2026-10-06-fixes-report.md`, section 8).

**2026-10-06 (late night): the GUI's calibration looped and its wand points would not load; both fixed**
- **Your questions:** how long a calibration from detected points takes; why the GUI "keeps going without too much
  improvement"; why the GUI's saved wand points would not load ("the status is always Raw"); and what happens with
  more than two cameras.
- **Where:** `tests/cylinder_wand_gui/2026-10-06-fixes-plan.md` (approved) and `2026-10-06-fixes-report.md`, with
  diagnoses P1–P7 and tests F1–F6 beside them. Uncommitted.
- **Stage 4 was caught in two loops on T0 and T2,** so each of the six starts ran the full 20 rounds (10 and 14 min).
  - **Loop 1:** the per-camera noise weights flipped every round between two states. T0's delivered fit left out 356
    good detections (89 frames) because of it.
  - **Loop 2:** two frames that fail the outlier test in the fit were dropped, re-tested from their own
    re-triangulation without the fit's correction, passed, came back and failed again, with equal weights too.
- **The fixes (`cylinder_wand.refine`, `calibrate`):**
  - **A flip** falls back to equal weights at once.
  - **A repeated active set** stops the rounds, with the alternating detections left out.
  - **Starts that end stage 3 at the same point** share one stage 4.
- **Now, on the real runs:**
  - **Time:** T0 96 s (was 576 s) and T2 234 s (was 816 s) with six starts; 60 s and 65 s with the GUI's default two.
    No start reaches the round limit.
  - **T2:** its delivered result exactly.
  - **T0:** 344 more detections. f −0.33 % / −0.58 % and the distances −3.3 / −6.0 mm against the delivered camFiles,
    under half a standard error. Its camFiles and page await redelivery (your OK).
- **More cameras:**
  - **Loop 1** is a two-camera effect: no synthetic fit with three or four cameras ever flipped.
  - **Loop 2** needs a frame dropped whole and also happens with three cameras: 32 synthetic fits ran to the round
    limit, 27 of them with heavy-tailed noise (T11d).
  - **On the synthetic campaign before and after (F4):**
    - every fit that converged is unchanged;
    - the 35 that looped now stop in 3–6 rounds, with the same accuracy;
    - every test's verdict is the same.
- **The wand points:**
  - **Why they were all "Raw":** with the GUI's default rod opening (25 px) this wand's rod (76–105 px wide) joins
    the balls into one blob. The detection then pairs no frame and saves only "Raw" rows, and the loader reported
    "0 valid frames" as a success.
  - **What works:** the case's settings (95 / 50 px, radius 40–120 px) pair 159 of 1,200 T2 frames, which load.
  - **Now:** the GUI says when a detection pairs nothing, and why, and loading a raw-only file explains itself.
  - **Memory:** the GUI's detection workers start with one BLAS thread, 0.33 GB each instead of 4.2 GB, so the
    detection no longer runs out of memory on this PC.

**2026-10-06 (night): the second recording, T2**
- **The data:** run 20261006/T2, an hour after T0, with the same cameras and wand. 12,716 frames; 1,966 usable in both
  cameras.
  - **Case:** `cases/cyl_20261006_T2_wand/` (plan, report, P0, T0–T3, T5, T7, X1, D1, D8). It copies T0's code, with
    T0's detector settings.
  - **Results:** `X:\Shijie Zhong\Refraction\Processed\20261006\T2\Results`, with an overview page (artifact
    https://claude.ai/artifact/Mu8jHCGGPtcVAZ4F36MNhc).
- **The calibration:**
  - reprojection RMS 0.60 px;
  - free-triangulation wand length 31 µm RMS;
  - cam1 f 38,252 px at 1,001 mm, cam2 f 37,409 px at 1,085 mm.
- **The cameras moved between T0 and T2 (X1, new):**
  - **The test:** each run's calibration triangulates the other run's frames.
  - **The step:** the two cameras' lines of sight miss each other by 1.0 px vertically (+1.01 / −1.05 px). The step
    is level along all of T2, before and after the light changed mid-run, while the wand lengths barely notice
    (+1.7 / −5.0 µm).
  - **So:** a calibration holds only while the cameras stay untouched. Calibrate close to each experiment.
- **f is still not determined (T3):**
  - **The spread:** the parts give cam1 f 33,900–41,200 px, and one quarter's cam2 ran to 4.7 m.
  - **One line per camera:** every fit of both runs lies on one f / distance line per camera. cam1's 30 Sep dot board
    lies on it too (37.9 against 38.2 px/mm).
  - **So:** a "hold f" option fed by the dot board would fix it. It is a library change, not built; it needs your
    approval.
- **Inside the volume, better than T0 (D8):** held-out wand lengths within 0.06 % on average, the halves within
  0.01–0.02 %.
- **k1 does not help (T5, your question):**
  - **Barely lower:** the RMS falls by 4 %.
  - **Implausible values:** k1 +18.8 (cam1, 3.5 px at the image's edge) and −2.9 (cam2), opposite signs.
  - **The pattern stays:** the residual pattern is unchanged.
- **Both overview pages now show each camera's 2-D reprojection-error distribution** (your request; the T0 page was
  republished).
  - **What it shows:** the errors are about twice as wide across the image as up and down, along the roughly level
    rod.
  - **Why:** the rod hides part of each ball's outline there.

**2026-10-06 (evening): the first real wand data**
- **The data:** run 20261006/T0, two cameras (VEO 710L, 180 mm macro, 100 frames/s) through your tube. The wand is
  10 mm long with 3 mm and 4 mm balls on a 2 mm rod, hanging on a string. 10,640 frames; 1,519 usable in both
  cameras. Case: `cases/cyl_20261006_T0_wand/` (plan, report, tests T0–T7, diagnoses D1–D8). Results:
  `X:\Shijie Zhong\Refraction\Processed\20261006\T0\Results`.
- **The detector works on real images:** the "ellipse" mode, with rod settings sized for this wand (opening 95 px,
  mask 50 px). The fits follow the outlines, and A and B never swap.
  - **But the rod limits it.** The 2 mm rod is about 75 % of the small ball's height in the image, so the small
    ball's centre moves by 0.6–2 px RMS with the rod mask (T1 b).
- **The calibration:**
  - reprojection RMS 0.45 px;
  - free-triangulation wand length 22 µm RMS;
  - cam1 f 38,678 px, cam2 f 36,022 px, the cameras 61° apart around the tube.
- **This recording does not determine f (T3 failed).**
  - **The spread:** the halves and quarters give cam1 f from 32,800 to 40,200 px. Each time the distance changes in
    proportion, the near-telecentric valley.
  - **The error bars:** 0.8–1.3 %, 10–30 times too small.
  - **The free wand length (T4):** 11.07 ± 0.10 mm, no conclusion about the tube.
- **Inside the wand's volume the cameras agree (D8):** each half predicts the other half's wand lengths to 0.07–0.13 %
  (mean), 25–38 µm RMS. So the camFiles are usable there, but f and the distances are not physical.
- **Ruled out:** synchronisation (D1), motion blur (100 µs exposures), the wand's bouncing (D5), a stretched calibration
  (D6). Each camera's image also shifts by 0.2–0.3 px horizontally from frame to frame on its own (vibration?), which
  the fit absorbs.
- **Lessons:**
  - **Rod thickness:** this detector needs a rod well below the small ball's size (T13 used about 1/4).
  - **f with two long lenses:** they need an independent f, from a dot board through the tube. Holding it in the
    calibration is a small library option, not yet built.
  - **The page's checks:** a single calibration's page cannot see T3's problem. A split-half check in the GUI would.

**2026-10-06 (afternoon)**
- **The cylinder wand calibration is in the GUI.**
  - **Where:** Wand Calibration → Calibration → Camera Model "Pinhole+Cylinder", with a Tube panel. The defaults are
    your tube.
  - **Detection:** a new mode, "ellipse", measures balls seen through the tube. It is T13's estimator with the rod
    masked and the overlap guard, now in the library (`curved_calibration/ball_ellipses.py`).
  - **What a run writes:** PINCYL camFiles, the result JSON and report, and a results page (`calibration_report.html`).
  - **What the GUI shows:** the page in a new Report tab, the error table and overlays, and the tube in the 3-D view.
    The tracking settings pick up the camFiles.
  - **Tests** (`tests/cylinder_wand_gui/`: plan, report, G0–G9):
    - on rendered images through your tube, going from the images to the camFiles through the GUI: f within 0.02 %
      (σ 0.06 %), RMS 0.017 px;
    - the GUI's runner gives exactly `calibrate`'s answer;
    - pytest is unchanged.
  - **Library change:** `calibrate` and `bootstrap` gained `log_fits` (off by default) for the GUI's progress and
    Stop.
  - **Still to try:** real images through the tube.

**2026-10-06**
- **The paper's §9.2.2 now quotes this campaign** (`cases/cyl_wand_paper/2026-10-06-report.md`; every number, with
  its source file and key, in that case's `paper_numbers_out.txt`).
  - **The paper's calibration differs in two ways:** it starts each image's Newton iteration at the image of the C++
    projection instead of at the detection, and it has no guard.
  - **E1, E2:** on this campaign's data it gives the same residuals as `cylinder_wand`, to 3.4e-11 px, and the same
    fits: the same observations, parameters within 1.7e-5 σ.
  - **E0, E0c:** every wand end of the campaign lies in the single-image volume. 907,000 point–camera pairs each
    have exactly one image. The ends reach 33.2 mm from the axis on the reference rig and 49.3 mm on T5's rigs.
  - **E2b:** today's library, with per-camera noise weights, moves T4's fits of 1 Oct by at most 0.038 Cramér–Rao σ.
    With T12b, the equal-weight results of T4–T11 therefore stand for the present calibration.
- **`cyl_model.to_cpp`'s default tolerance, 1e-12, is below the rounding floor next to grazing.**
  - **Where it bites:** a ray that grazes the point's own radius, the junction of the near and far branches.
    Rounding limits the projection's residual there to about 1e-12–4e-11.
  - **Effect:** C++ cameras built by `to_cpp` (in `rigs.wand_frames`, in the guard, in E0) skip such rays: 27 of
    907,000 pairs in the census, 7 of 227,000 wand-end images in the regenerated data sets.
  - **At the projection's own default, 1e-9,** all are found.
  - No result changes. Not fixed: a rounding-level effect (your rule of 2026-10-06).
- **Corrected in section 3 item 2:**
  - the 0.1 px table comes from T11a's 0.1 px level (40 data sets) and from T7, not from T4;
  - T4's z-score SD is 0.925;
  - the campaign's sensor is 3200×2400 px.

**2026-10-04**
- **Double images during the fit (your question for the paper's §9.1).** You keep the calibration to observations in
  the single-image region. No code was changed, by your decision. The plan of the paper's variant
  (`cases/cyl_wand_paper/2026-10-03-plan.md`) was amended instead; its section 7 has the details.
  - **The library's guard has a gap,** found on 3 Oct and not fixed (section 6, item 8).
    - `multi_image_guard` checks only points beyond r_c at zero tilt, and drops an observation only for a second
      image on the sensor within 30 px.
    - r_c shrinks for tilted rays: 75.62 mm at C_z = 0, 74.04 at 0.3, 70.65 at 0.5 (checked with SymPy).
    - No wand rig so far is affected. C_z ≈ 0.02 leaves a 7 µm band, and the reference rig's wand ends stay within
      31 mm of the axis.
  - **What a fit does when a wand end leaves the region mid-way** (scratch probe: cam1 of the reference rig, 85 mm from
    the axis):
    - **In a two-image band** it keeps following its image. The new image appears about 1,400 px away.
    - **In a no-image band** the projection fails and the fit refuses the step, as the flat-wall fit does at a wall.
      The band's edge images at the tube's outline, at u ≈ 5,135 px, far off the 2,600 px sensor. A fit would need a
      huge residual to get there.
    - **Beyond the inner wall** `cyl_model.project` continues the ray, but the C++ projection does not look outside
      the water. The paper's variant would therefore refuse steps that the library accepts.
  - **The amended plan:**
    - the single-image rule runs once, since a census repeated inside the iterations could walk the fit away from the
      data;
    - a recount after the last adjustment warns on drops;
    - the residual falls back to the library's Newton where the C++ finds no image;
    - P0 tests starts across the wall and the bands.

    It still waits for your approval.
- **An illustrated guide to both wand calibrations, tube and flat walls:**
  `modules/camera_calibration/curved_calibration/guide/WAND_CALIBRATION_GUIDE.html`, also at
  https://claude.ai/artifact/BR616e7TfeF63xsX4rrA46 (private until you share it).
  - **What it shows:** a flow chart, both rigs in 3-D, a 2 mm ball's image through the tube, and each stage.
  - **How it is made:** `make_guide.py` beside it computes everything with the calibration's own models, on a
    synthetic copy of the reference rig (500 frames, 0.1 px).
  - **The run behind it:**
    - **Calibration:** from scratch in 28 s, with f within 0.1 % on all three cameras.
    - **Start search:** its replay of `bootstrap`'s placement search matches the library.
    - **The stretch ratio** puts the start's cameras 20–40 % too far away, even with the axis in its true place.
      The ratio-against-distance curve flattens with distance, and stages 3–4 then correct it.
- **The flat-wall calibration's "pinned" failure cannot happen at the tube's wall** (checked for your question). A
  no-image band can still stop a step, and the paper's variant differs at the wall: see the entry above.
  - On the plate, a fit can freeze with a wand end on a wall's water side: the model refuses that point, so the fit
    refuses every step.
  - `cyl_model.project` refuses no point at the tube's wall. Points up to 102 mm from the axis still project; the
    inner wall is at 97.6 mm.
  - The tube does not move, and the start keeps the wand inside it.

**2026-10-03**
- **A crash in `cylinder_wand.affine_factorization` (stage 1), found and fixed on real flat-window data** (case
  `cases/plate_20260727_T0`, report section 3).
  - **Cause:** the outlier rejection runs at most 5 rounds. When the last round still rejected frames, the frames it
    returned no longer matched its factorisation, giving an IndexError.
  - **Why only now:** real detections with a few gross errors trigger it; the synthetic tests never did.
  - **Fix:** the function returns the frames the last factorisation used. Wherever the rejection converged, which is
    every earlier run that did not crash, the result is identical.
  - **Checks:** a regression test is in `tests/test_plate_wand.py`. The cylinder suites still pass (6 fast + 1 slow).
  - This is the first change to `cylinder_wand.py` since 2 Oct, made with your approval.

**2026-10-02**
- **Two weaknesses of `cylinder_wand.refine`, found by the flat-window calibration's tests** (`tests/plate_wand/`,
  which copies this machinery; fixed there, not here: the cylinder code is unchanged).
  - **A gross outlier costs its whole frame.** A frame left with no active observation is re-triangulated from all its
    rays, the outlier included, so its good rays never fit well enough to come back. Flat-window W7 (2 % outliers of
    20–200 px): 160 outliers rejected, and with them 1,001 good detections. The plate code now drops a frame's worst ray
    while it is far off before re-triangulating (0–6 good detections lost). The cylinder's T8 and T11d used smaller
    or fewer outliers and did not show it.
  - **With two cameras the per-camera noise weights can drift.** Each wand position's residual can move from one camera
    to the other, and the estimate feeds on it. Flat-window W5 seed 2005 (1 px): 0.95, 1.00 px became 0.14, 1.35 px over
    16 rounds, then 11 % of good data were rejected (RMS 1.11 against 0.85 with equal weights). The plate code uses
    equal weights with two cameras and falls back when the weights do not settle. The cylinder's two-camera rigs
    (T12) have not shown it, but nothing stops it.
  - Both are open item 7.
- **A latent rotation bug in `cyl_model.py`, found while planning the flat-window calibration**
  (`tests/plate_wand/2026-10-02-plate-wand-plan.md`, section 3.3; `tests/plate_wand/p_logmap_check.py`).
  - **`log_so3` is wrong near a half-turn.** For every rotation within 0.14 rad of one, its half-turn branch does not
    normalise the axis, and returns a rotation 1.3 rad off.
    - The calibration uses it only as `rvec_of`'s fallback when OpenCV is missing, and in one test helper's error
      metric (`rigs.py:206`, small angles only).
    - No reported result depends on it.
  - **`rvec_of` uses `cv2.Rodrigues`, which loses up to 2e-5 rad** within about 1e-4 rad of a half-turn.
    - That reaches `to_cpp`: the C++ cameras of the T0/T10 checks and of the double-image guard.
    - The camFiles are not affected: the C++ reads the matrix and ignores the vector on load.
  - **Fixed the same day (two lines, approved with the flat-window plan):** `log_so3` now normalises the axis, and
    `rvec_of` uses `log_so3` instead of `cv2.Rodrigues`.
    - The round trip is now within 1.7e-15 rad at every angle, half-turns included
      (`tests/plate_wand/w0_logmap.py`, 36,000 rotations); `cv2.Rodrigues` loses 2.0e-5 rad on the same rotations.
    - `pytest tests/test_cylinder_wand.py`: 7 passed (40 s), as before.
- **T13 done: measuring the wand balls' centres through the tube** (`2026-10-02-ball-centres-report.md`).
  - **What was run:** wand images rendered exactly through the tube, with the plate case's astigmatic blur, four
    lightings and the rod. Six detectors were scored against the truth, and their centres were fed to the
    calibration (8 data sets).
  - **pyOpenLPT's wand detector cannot be used through the tube.**
    - It fits circles, which fail from about 1.2 : 1. The tube's ellipses are 1.3 : 1.
    - Its centres are 5–7 px off, and the calibration comes out wrong by tens of percent in f.
    - Thresholded-blob centroids are pulled by the rod: 0.4–1.6 px, and f off by 0.4–68 %.
  - **The plate case's ellipse fit works, with the rod's edge points left out** and an overlap guard:
    - 0.026–0.028 px RMS, no bias, backlit or evenly lit;
    - f within 0.07 % of the truth, 3-D shape 0.12–0.13 µm, an unbiased wand length.
    - What remains is small: f +0.03 to +0.07 %, and error bars 1.5× too small.
  - **A model-based correction removes that remainder (T13d):** render each detection's predicted image, run the
    same detector, subtract the offset.
  - **Uneven lighting harms it even then:** f up to 0.1 % off with one-sided light, and 0.56 % with a highlight.
- **How a ball's image through the tube affects its measured centre** (your question).
  - An exact calculation (`t13_pre_sizing.py`) shows that the shape itself does not matter. The image is an ellipse
    about 1.3× wider than tall, but its centre is within 0.002 px of the image of the ball's centre for balls up to
    2 mm in the ±8 mm volume (section 3, item 9).
  - The real risks are elsewhere:
    - pyOpenLPT's wand detector fits circles to these ellipses;
    - the tube separates the horizontal and vertical foci;
    - uneven lighting.
  - The T13 plan, which tests detectors on rendered wand images: `2026-10-02-ball-centres-plan.md` (approved and
    done, above).
- This summary started.
- **The calibration report shows each camera's noise and weights.** They are in `calibration_report.txt` and in the
  "noise" block of `calibration_result.json`:
  - the estimated noise σ, with its standard error;
  - the detections used and not used, and the residual RMS;
  - whether the weights were applied, and if not, why;
  - the starts tried.
  - Five example reports are in `t12f_report/`. Details: weights report, section 2.
- **A defect found while adding it, and fixed.**
  - Whether the cameras' noise levels could be told apart was judged from the first round, which uses equal weights.
    It is now judged once the weights have converged.
  - Two-camera rigs with one noisier camera now use the weights in 35 and 40 of 40 data sets (cameras 45° and 90°
    apart), instead of 25 and 33.
  - Details: weights report, section 4, defect 4.
- **T12: noise weights per camera (the new default), and more starts for noisy data.**
  - The calibration estimates each camera's noise from its residuals and weights by it. Nothing needs to be entered.
  - A camera with 5× the noise no longer costs the others 1.7–2.1× in precision.
  - Calibrations from scratch at 5 px noise now succeed 10 of 10 (before: 9).
  - The cost is ×1.09 in time.
  - `2026-10-02-weights-plan.md`, `2026-10-02-weights-report.md`.

**2026-10-01/02**
- **T11: how detection noise affects the calibration.**
  - It covered Gaussian noise from 0.02 to 5 px, one noisier camera or wand end, occasional large errors, and a
    pattern fixed to each image.
  - It found the two weaknesses that T12 fixed, and the one that remains (section 3, item 8).
  - `2026-10-01-noise-plan.md`, `2026-10-01-noise-report.md`.

**2026-10-01**
- **The calibration was built and tested on synthetic data (T0–T10):** several cameras through one tube, from wand
  frames alone. `2026-10-01-plan.md`, `2026-10-01-phase0-report.md`, `2026-10-01-report.md`.
- **Figures page**, updated since with T11 and T12: `figures/cylinder_wand_results.html`, also at
  https://claude.ai/artifact/2UGkfg2d9L7JHvZbA9npR3.

## 2. Status

- **What it does:** it calibrates several cameras that look through one water-filled tube, from wand frames alone,
  with no camera input and focal lengths included. It writes:
  - pyOpenLPT PINCYL camFiles, which reload and project identically (T10);
  - a result JSON;
  - a text report.
- **How** (`calibrate`, stages 1–5):
  1. an affine reconstruction of the wand points, made metric by the wand length;
  2. the tube placed relative to it: about 90 candidate placements, each given a quick fit;
  3. a coarse fit on the back-projected rays;
  4. the maximum-likelihood bundle adjustment, with outlier rejection, per-camera noise weights and a guard against
     double images;
  5. the covariance and checks.
- **Tested on:** synthetic data only.
  - The data come from pyOpenLPT's C++ PINCYL camera in the real case's tube: OD 7.9375 in, wall 0.125 in; air,
    acrylic 1.49, water 1.333.
  - The wand is 10 mm long, and the rigs have 2–4 cameras.
- **Not done yet:** real data, the GUI's wand CSV reader, and the GUI integration. These have been out of scope
  since 2026-10-01.
- **Code,** in `modules/camera_calibration/curved_calibration/`:
  - `cyl_model.py`: a fast vectorised PINCYL with exact derivatives. It is checked against the C++ (T0), and its
    derivatives against finite differences (T1);
  - `cylinder_wand.py`: the calibration;
  - `cylinder_wand_io.py`: the rig config, camFiles, result JSON and report.
- **Tests:**
  - `tests/test_cylinder_wand.py`: 7 pass. One of them calibrates from scratch, in about 25 s.
  - The campaign is in this folder.
  - pyOpenLPT's full suite shows no failure caused by this work. One test fails for a reason outside it (section 6).
- **Run time:**
  - 20–60 s per calibration from scratch (500 frames, 2–4 cameras);
  - about 3× that when the data are very noisy (stage-3 RMS above 1 px), because every short-listed start is then
    tried.
- **Git:** the work through T12 is in pyOpenLPT commit `ed4d6b6` (branch `Refraction_new`).

Use:

```python
import cylinder_wand as cw, cylinder_wand_io as cio      # (package: modules.camera_calibration.curved_calibration)
tube = cw.cm.Tube(radii=(100.80625, 97.63125), n=(1.0, 1.49, 1.333))   # radii outer, inner (mm); air, acrylic, water
data = cw.WandData(uv, 10.0, sizes)   # uv[frame, camera, end A/B, (u, v)], NaN where not seen; sizes (width, height)
sol = cw.calibrate(data, tube)        # options: fit_L=True; noise_weights="camera_end" or "none"; sigma_cam_px=[...];
                                      #   hints={"f_range": (lo, hi)}; fit_k1=True; guard_px=30
cio.write_result(sol, "out", names=["cam1", "cam2", "cam3"], pixel_mm=[0.02] * 3)
```

The output frame is the tube frame:
- the axis is Y, through the origin;
- Z points towards camera 1;
- the origin sits at the wand's mean height.

## 3. What the tests concluded

1. **It finds the right answer from scratch.**
   - All 50 random rigs reached the global minimum without hints (T5, and again with the weights in T12d). There were
     10 rigs with 2 cameras, 25 with 3 and 15 with 4, with cameras 0.4–1.5 m from the axis.
   - It still works at 2 px of noise on every rig tried, and at 5 px since T12 (10 of 10).
2. **It reaches the theoretical limit, and its error bars match the real scatter.**
   - Errors grow in proportion to the detection noise and sit on the Cramér–Rao bound from 0.02 to 2 px.
   - The z-scores' standard deviations are 0.92–1.02 (T4: 0.925; T11a: 0.95–1.02).
   - The table is for the reference rig: three copies of cam1, 45° apart, a 3200×2400 px sensor, a ±20 mm wand
     cloud and 500 frames. It comes from T11a's 0.1 px level (40 data sets); the cameras' agreement on unseen
     points comes from T7.

   | quantity | error at 0.1 px noise |
   |---|---|
   | f | 0.07 % |
   | 3-D shape and scale (distances, velocities) | 0.8 µm |
   | 3-D position relative to the tube | 19 µm |
   | length of a held-out 10 mm wand | 2.4 µm (the calibration's share: 0.23 µm) |
   | agreement between the cameras on unseen points | 0.008 px |

   Everything scales with the noise. Per pixel of noise: f 0.7 %, 3-D shape 7.5 µm, wand length 24 µm.
3. **Distances are about 25× more certain than positions relative to the tube.**
   - Most of the position error is where the tube sits relative to the cameras: σ 13 µm across the axis, and
     13 arcsec of tilt.
   - The wand cannot fix the spin about the axis or the slide along it at all. The output sets them by convention
     (section 2).
   - A point's distance from the axis is fully determined.
4. **Rig design (T2, Cramér–Rao).**
   - The depth of the wand cloud matters most, then the number of frames, then the number of cameras.
   - Angles of 45–120° between cameras are all fine; 30° is worse.

   | change from the reference rig | σ(f)/f at 0.1 px |
   |---|---|
   | wand cloud ±40 / ±20 / ±10 mm | 0.020 / 0.064 / 0.164 % |
   | ±8 mm: the common volume cam1's present sensor sees | 0.219 % |
   | 200 / 500 / 2000 frames | 0.101 / 0.064 / 0.032 % |
   | 2 / 3 / 4 cameras, 45° apart | 0.114 / 0.064 / 0.056 % |
   | cameras at 0.5 m instead of 1.1 m (f scaled) | 0.028 % |

5. **Bad data are handled** (T8, T11d).
   - Gross outliers and swapped A/B ends were all rejected. The answer stayed within 0.8σ of the clean-data fit.
   - Points with a second image within 30 px (near the wall) are dropped. Only points beyond r_c at zero tilt are
     checked (section 6, item 8).
   - With occasional large errors the error bars stay honest.
6. **A noisier camera is handled automatically since T12.**
   - Before T12, one camera at 5× the noise cost every camera 1.7–2.1× in precision, because the cameras share the
     wand positions.
   - Now each camera's noise is estimated from the residuals and weighted. That brings the precision back to the
     limit (×1.00–1.05).
   - With only two cameras, the wand length alone tells their noise levels apart, so sometimes they cannot be
     separated. The fit then keeps equal weights, and the report says why.
7. **The tube fixes the scale (T9).**
   - With the wand length left free (`fit_L=True`), the tube's radii set the scale to 0.02 %.
   - The free length is therefore a sharp test of the constants:
     - a water index off by 0.002 shows at 33σ, and the residuals grow by 8 %;
     - an outer radius off by 0.25 mm shows at 12σ;
     - a nominal wand length off by 0.1 % shows at about 5σ.
8. **The danger: a pattern fixed to each image, which PINCYL cannot represent (T11e).**
   - With 0.07–0.15 px of it:
     - about 60 % is absorbed into the camera parameters;
     - f is biased by 0.2–0.5 %;
     - the error bars come out 3–4× too small;
     - the residual RMS (0.127 px against 0.107 px) looks like ordinary noise.
   - The free-length test caught it in only 7–12 of 20 data sets, and the noise weights do not help.
   - The real cam1 plate case (`cases/cyl_20260930_T0_cam1`) shows 0.1–0.2 px of this kind of error.
9. **A ball's distorted image does not bias its centre** (exact pre-plan calculation, `t13_pre_sizing.py`).
   - Through the tube a ball's image is an ellipse, 1.26–1.34 times wider than tall over ±8 mm (1.21–1.41 over
     ±20 mm). At cam1's magnification a 1 mm ball is 50 × 39 px.
   - The centre of its exact silhouette (area centroid or fitted ellipse) is close to the image of the ball's centre:

     | ball diameter | ±8 mm | ±20 mm |
     |---|---|---|
     | 1 mm | 0.0005 px | 0.003 px |
     | 2 mm | 0.002 px | 0.011 px |
     | 3 mm | 0.0045 px | 0.025 px |

   - The offset grows as the square of the ball's size. The plate case found the same for its dots (0.003 px RMS).
   - What can bias a real ball's centre is the detector and the image, not the geometry. T13 measured it:
     - **pyOpenLPT's circle detector** fails on the 1.3 : 1 images. Its centres are 5–7 px off, and the
       calibration comes out wrong by tens of percent in f.
     - **Blob centroids** are pulled by the rod: 0.4–1.6 px.
     - **The rod-masked ellipse fit** gives 0.026 px RMS with no bias, and f within 0.07 % of the truth. A
       model-based correction removes even that remainder.
     - **Uneven lighting** still biases f, by 0.1–0.56 %.

## 4. For a real experiment

**Before recording**
- **Calibrate with the experiment's exact optics.**
  - Keep the lens settings (focus, aperture): refocusing moves f and the projection centre.
  - Keep the cameras and the tube fixed, and use the same water.
- **Use the right water index.**
  - 1.333 is water at 589 nm and 20 °C. Use the value for your light's wavelength: blue light near 475 nm gives about
    1.3376, the value the plate case preferred.
  - The index also falls by roughly 1e-4 per °C, so match the water temperature.
  - Measure the tube's outer diameter and wall thickness.
- **Measure the wand length independently,** to 0.1 % (10 µm on 10 mm) or better. The free-length run then tests
  the tube constants: a 0.002 error in the water index changes the length by 0.65 %.
- **Make the two ends distinguishable**, for example with two ball sizes.
  - Label A and B the same way in every camera. Occasional swaps are rejected as outliers (T8).
- **Ball size and detection.**
  - Balls of 1–2 mm are plenty: 50–100 px wide images at cam1's magnification, with a shape offset of 0.002 px or
    less (section 3, item 9).
  - **Fit ellipses, not circles.** The circle finders fail on these 1.3 : 1 images (T13). The GUI's "ellipse"
    detection mode fits ellipses: the plate case's estimator, with the rod's edge points left out and the overlap
    guard. It tells the balls apart by size.
  - **Choose its settings per camera** (`cases/cyl_20261013`'s report, section 3):
    - R min ≈ 0.6 × the small ball's radius, R max ≈ 1.5 × the large ball's;
    - the rod opening between the rod's width and the small ball's, found with Test Detect;
    - the rod mask about 0.55–0.6 × the opening.
  - **A thinner rod helps.** With a 2 mm rod on a 3 mm ball the opening's window is only a few pixels wide, and it
    differs between cameras.
  - **Light the balls evenly** (backlit, or diffuse light on matte balls). One-sided light or a highlight biased f
    by 0.1–0.56 % even with the best detector (T13c).
  - Keep the volume near best focus, with a small aperture. The tube separates the horizontal and vertical foci.

**While recording**
- **Move the wand through the whole volume you will measure in,** including its full depth as seen from each camera.
  The calibration is only good where the wand went.
- **Cam1's present field gives a common volume of only about ±8 mm.**
  - f is then 0.22 % uncertain at 0.1 px, 3.4× worse than with ±20 mm. More frames (2000 halve the error) or lower
    noise make up for it.
  - Triangulation inside that volume stays consistent (0.006 px). What suffers is f, the camera distances and the
    tube's position relative to the cameras.
- **Record 500 or more frames with both ends seen by at least two cameras.**
  - Other frames are dropped.
  - At least 12 frames must show the whole wand in every camera, because the automatic start needs them.
- **Use three or more cameras if you can.** Two work, but f comes out about 1.7× worse, and their noise levels may
  not separate.
- **Aim for about 0.1 px detection noise.**
  - 0.3 px still keeps f within 0.2 % on the reference rig.
  - The macro's shallow depth of field will make the noise grow away from focus, and the weights assume one noise
    level per camera.
- **Synchronise the cameras, or move the wand slowly.** The tests assume exactly simultaneous frames.

**Checking the result** (the RMS alone can hide a model error)
- **Read the noise section of the report.** Each camera's noise should match your detector's. One camera much
  noisier than the others points to blur or focus.
- **Compare that noise with a still wand imaged over many frames,** which gives the detection noise alone. If the
  fit's noise is clearly larger, suspect model error, or motion or synchronisation.
- **Run once with `fit_L=True`.** If the fitted length is more than 3σ from your measurement, the constants or the
  model are off, and the error bars are too small.
- **Plot each camera's residuals binned over the image.** A fixed pattern hides in the RMS but shows up as structure.
- **Validate on something you know,** for example the dot board on the stage, imaged by all cameras and
  triangulated.

## 5. Not covered by any test

- Real data.
- The GUI's wand CSV reader. Until it exists, build the `uv[frame, camera, end, (u, v)]` array yourself.
- 5 or more cameras.
- The k1 distortion term in a full calibration. Only its derivatives were checked (T1b).
- Rigs with little common view. The start needs 12 frames seen whole by every camera.
- Noise that varies with position or depth. The weights assume one level per camera, or per camera and wand end.
- Ball centres on real images. T13 used rendered ones: modelled blur, lighting and ball surfaces. It also did not
  cover motion blur, or other ball and rod sizes.
- Camera synchronisation.

## 6. Open items and possible next steps

None of these is started; each needs your go-ahead.

1. **A ball detector for real wand images through the tube** (T13's recommendation).
   - Put the rod-masked ellipse fit with the overlap guard into pyOpenLPT, for a wand CSV path or the GUI.
   - Optionally add the model-based correction, with the blur measured from the balls' edges.
2. **The first real wand data set.**
   - Run it with the checks of section 4.
   - With a wand of independently measured length and `fit_L=True`, it also settles the plate case's open 1.1 %
     mismatch: is it the water index or the stage scale?
     - A wrong index would make the wand come out roughly 1.5 % long (scaled from T9).
     - The stage plays no part in a wand fit.
3. **A model for patterns fixed to each image** (T11's proposal 3).
   - Options: distortion terms beyond k1, or a low-order correction field per camera with a guard against it
     absorbing the geometry.
   - T11e is the test bench. This is worth doing if the real data show the pattern.
4. **GUI:** the wand CSV reader and the integration.
5. **The untested options:** k1 in a full calibration; 5 or more cameras; a pairwise start for rigs with little common
   view.
6. **Housekeeping:**
   - `tests/test_refract_oracles.py::test_vendored_cylinder_oracle_matches_notes_repo` fails (checked 2026-10-02).
     - The cause is that `D:\Refraction\code\project_cylinder.py` was edited on 2026-10-02, outside this work.
     - Re-copy it to `test/refs/project_cylinder.py` once that edit is final. It is your file, so it was left alone.
   - The `cyl_model.py` rotation fix (progress log, 2026-10-02): applied 2026-10-02.
7. **Port the flat-window calibration's two `refine` fixes** (progress log, 2026-10-02): the robust re-triangulation
   of a frame before re-admission, and the two-camera noise weights. Each is a few lines in `cylinder_wand.py`, with
   T8/T11d/T12 rerun.
8. **The double-image guard's gap** (progress log, 2026-10-04; `cases/cyl_wand_paper/README.md`).
   - `multi_image_guard` checks only points beyond r_c at zero tilt. The fix is to take the census of every
     observation, as the paper's variant will: a few seconds per round.
   - Not urgent: no wand rig so far is affected. On 4 Oct you chose to leave the code as it is for now.
9. **The answer's selection when several starts are carried** (progress log, 2026-10-07 late night). `calibrate`
   ranks the starts' answers by the detections kept, then by the likelihood. A worse minimum keeps more (its larger
   residuals widen its 5σ limit), and on 20261007/T0 it won with six starts. Options: rank by the likelihood over the
   detections common to both answers, or drop an answer whose RMS is well above another's.
10. **f from another source, held:** the option exists (`fix_f`, T14). What is still missing is a reliable f: on
    20261007 the dot board fixed only cam2's f (to about ±3 %), and holding it did not make the parts agree.

## 7. Files

All in this folder unless stated.

**Plans and reports**, in order:

| file | what |
|---|---|
| `2026-10-01-plan.md` | the plan of the calibration and of T0–T10 |
| `2026-10-01-phase0-report.md` | T0–T2: the fast model against the C++, observability, rig design |
| `2026-10-01-report.md` | the main report: T0–T10, what was built, the defects found and fixed |
| `2026-10-01-noise-plan.md`, `2026-10-01-noise-report.md` | T11: detection noise |
| `2026-10-02-weights-plan.md`, `2026-10-02-weights-report.md` | T12: noise weights, more starts, the report's noise section |
| `2026-10-02-ball-centres-plan.md`, `2026-10-02-ball-centres-report.md` | T13: wand balls' centres through the tube (detectors, effect on the calibration, model-based correction); pre-plan sizing `t13_pre_sizing.py` |
| `t14_hold_f.py`, `t14_hold_f_out.txt`, `.json` | T14 (7 Oct): the hold-f option on synthetic data; its plan and report are the tracking case's (`cases/cyl_20261007/2026-10-07-plan.md` step C3, `2026-10-07-phaseC-report.md` section 5) |

**Figures:**
- `figures/cylinder_wand_results.html` (open it in a browser), also at
  https://claude.ai/artifact/2UGkfg2d9L7JHvZbA9npR3.
- `make_figures.py` rebuilds it from the saved results.

**Example output:** `t12f_report/<case>/` holds complete results (camFile/, calibration_result.json,
calibration_report.txt) for five cases:
- three cameras, one noisier;
- three cameras, equal noise;
- two cameras 90° apart;
- two cameras with only 150 frames, where the weights are not applied (and the report says why);
- wand ends with different noise.

**Scripts:**
- One script per test (`t0_*.py` to `t12f_report.py`). Each writes `<name>_out.txt` beside it, and most also
  `<name>.json`.
- `run_all.py --with-t5` reruns T0–T10.
- Shared code:
  - `rigs.py`: the synthetic rigs and wand data;
  - `t11_common.py`, `t12_common.py`: the shared code of T11 and T12.
- `p12_sympy_noise_weights.py`: the SymPy checks of T12.

**The real single-camera plate calibration** these tests refer to: `cases/cyl_20260930_T0_cam1/2026-10-01-report.md`
(from the pyOpenLPT root).
