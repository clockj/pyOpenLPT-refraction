# Plan: record the detection settings, for the results page and for reproduction (13 Oct 2026)

**Your message:** "what are the two options? i think it is a good idea to record the detection settings on the
results page and save it for reproduction"

**The two options** (from `2026-10-13-plan.md`, section 5):
- **(a)** The results page records the detection settings used. You want this one, and the settings saved.
- **(b)** Retake the user guide's screenshot of the detection panel so that it shows the new box. Not part of this
  plan.

## 1. Now

**What the GUI keeps.** It remembers only the detection mode (`_wand_detect_mode_used`) and where the points came
from. These go into the calibration's settings (`Job.settings()`, saved in `calibration_result.json`), and the
results page prints "Detection: ellipse" and "Points from: …".

**What is lost:**
- the Radius Range, Sensitivity, Rod opening and Rod mask, shared or per camera;
- the Wand Type and the pixel factors;
- the images each camera's points came from.

**Points loaded from a CSV** know nothing: the page says "unknown (points read from a file)", and the tube model's
checks add the note "Detection mode unknown".

## 2. What I propose

**1. A record of every detection.** It holds:
- the mode and the Wand Type;
- whether the settings were shared or per camera;
- each camera's **effective** values: R min, R max, the Sensitivity where the mode uses it, the Rod opening and mask
  in the ellipse mode;
- each camera's pixel factor, when the shared settings were scaled;
- each camera's images: folder, first file, count;
- the time, and whether it resumed an earlier file.

"Effective" means the exact numbers the detector got. Typing them into the per-camera table reproduces the
detection.

**2. Saved beside the points.** The record goes in a small file next to the wand-points CSV:
`wand_points.csv` → `wand_points_detection.json`.
- **Process All Frames / Resume** writes it when it starts, so a stopped run has it too. A resume adds its own entry,
  so a file detected in two goes, with different settings, shows both.
- **Export Data and the rescue export** write it beside their file as well.

**3. Read back with the points.** Load Wand Points reads that file when it is there.
- The calibration then knows the mode and the settings. The "Detection mode unknown" note goes away for such files.
- Older CSVs without a record behave as now.

**4. Carried into the results.**
- **The calibration's settings:** the tube and flat-window Jobs get a `detection` entry, saved in
  `calibration_result.json`.
- **The pinhole page** gets it directly.
- **The page's "Settings and files" table** gets rows like these:

  ```
  Detection         ellipse mode, Dark on Bright; each camera's own settings
  Detection, cam0   radius 40–120 px; Rod opening 95 px; Rod mask 50 px
  Detection, cam1   radius 30–90 px; Rod opening 70 px; Rod mask 38 px
  Images, cam0      X:\...\T2\cam1: 12,716 images from frame000000.tif
  Detection record  X:\...\wand_points_detection.json
  Points from       X:\...\wand_points.csv
  ```

  A file detected in more than one run with different settings gets a note saying so.

## 3. The code changes

1. **`wand_calibrator.py`:**
   - `detection_record_path(csv_path)`;
   - `save_detection_record(csv_path, record, resume)`, which on a resume adds to the file's entries;
   - `load_detection_record(csv_path)`.

   These have no Qt.
2. **`view.py`:**
   - `_wand_detection_record()`, which builds the record from the panel through `camera_detect_settings`, so it holds
     exactly what the tasks get;
   - Process All, Export, rescue and Load Wand Points save or read it;
   - the three calibrations pass it (`detection=`).
3. **`curved_calibration/cylinder_wand_gui.py` and `plate_wand_gui.py`:** `Job.detection` (optional), included in
   `settings()`.
4. **`results_page/page.py`:** the rows above, for all three models. When a report has no record, there are no new
   rows, so older results show as before.
5. **The user guide:** the record file in §3.1 step 5 and in the table of files (§6.5); the HTML rebuilt.
6. **The two wand READMEs:** a log entry.

**Not touched:** the detection itself and the calibrations.

## 4. The check (light)

**One script, `r_record.py`,** in the GUI's window, on the six T2 frames:
- **R1:** Process All Frames writes the record beside the CSV, equal to the settings the detection used: per camera
  (ellipse), and shared with pixel factors.
- **R2:** a Resume with changed settings keeps both entries.
- **R3:** Load Wand Points reads the record back (mode and settings), and a CSV without one stays "unknown".
- **R4:** the tube calibration's Job carries the record. The Job is captured before it runs, so no fit is run.
- **R5:** `report_from_cylinder` and `report_from_pinhole` show the rows (G3's saved result, with a record added), and
  a page with them draws in QtWebEngine without an error. I look at a screenshot.

Results and the report go in this folder: `2026-10-13-report-record.md`.

## 5. An option (not planned unless you want it)

- **(c)** Apply a record to the panel. A button, or a question when Load Wand Points finds a record, sets the
  detection panel to the file's settings, so that the same detection can be run again with one click.

## 6. Your answer (13 Oct)

"go ahead. i think c is not necessary at this point". Built as planned, without (c).

**One change to the check:** R used 24 T2 frames, not W1's six. The pairing needs 10 detections per camera to learn
the two balls' sizes ("Cam 0: Not enough points (7) for stats"), so six frames pair nothing, and Load Wand Points
refuses a file without pairs.
