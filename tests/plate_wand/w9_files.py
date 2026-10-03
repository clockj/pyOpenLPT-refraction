"""W9: the files.

1. camFile round trip: plate_wand_io.write_camfile (the C++ writer) -> pyopenlpt.Camera(path) -> projections of 500
   points against plate_model's, within 1e-6 px; and load_camfile gives back the same camera and window.
2. CSV round trip: write_gui_csv -> read_gui_csv returns the same detections (to the 6 decimals written); the GUI's own
   loader (WandCalibrator.load_wand_data_from_csv, read-only use) finds the same frames.
3. The result: a calibration of rig A written with write_result (camFiles, calibration_result.json,
   calibration_report.txt in w9_result/); its camFiles reload and reproduce the solution's projections.
Output: w9_files_out.txt / .json and w9_result/.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

pm = prigs.pm
cm = prigs.cm
import plate_wand as pw  # noqa: E402
import plate_wand_io as pio  # noqa: E402

log = prigs.Tee(HERE / "w9_files_out.txt")
rng = np.random.default_rng(9)
lpt = pm.import_lpt()
res = {}
ok_all = True

# 1. camFile round trip, every rig's cameras (B has two windows), with and without distortion
tmp = HERE / "w9_tmp"
tmp.mkdir(exist_ok=True)
worst_px = worst_par = 0.0
n_files = 0
for name, fn in prigs.RIGS.items():
    rig_t = fn()
    for c, cam in enumerate(rig_t.cams):
        for distorted in (False, True):
            cam2 = cam.copy()
            if distorted:
                cam2.dist = np.array([0.05, -0.02, 1e-4, -2e-4, 0.01])
            win = rig_t.win_of(c)
            path = tmp / f"{name}_cam{c + 1}_{int(distorted)}.txt"
            pio.write_camfile(cam2, win, str(path), (0.1, 0.05), (0.01, 0.005))
            cpp = lpt.Camera(str(path))
            P = prigs.cloud_points(rig_t, rng, 500)
            u_cpp = cm.cpp_project(cpp, P)
            pr = pm.project(pm.rows([cam2], np.zeros(len(P), int)), pm.wrows([win], np.zeros(len(P), int)), P)
            d = np.linalg.norm(u_cpp - pr.uv, axis=1)
            worst_px = max(worst_px, float(np.nanmax(d)))
            c3, w3 = pio.load_camfile(str(path))
            dpar = max(float(np.max(np.abs(c3.R - cam2.R))), float(np.max(np.abs(c3.C - cam2.C))) / 1000.0,
                       abs(c3.f / cam2.f - 1.0), float(np.max(np.abs(c3.dist - cam2.dist))),
                       float(np.max(np.abs(w3.n - win.n))), abs(w3.d - win.d) / 1000.0, abs(w3.T - win.T))
            worst_par = max(worst_par, dpar)
            n_files += 1
            text = path.read_text()
            assert "0.1,0.05" in text and "0.01,0.005" in text, "error lines not filled"
ok1 = worst_px <= 1e-6 and worst_par <= 1e-12
ok_all &= ok1
res["camfile"] = {"files": n_files, "worst_px": worst_px, "worst_param": worst_par, "pass": bool(ok1)}
log(f"1. camFile round trip: {n_files} files; C++ projections from the files vs plate_model: worst {worst_px:.2e} px; "
    f"parameters back: worst {worst_par:.1e} (relative / per metre)" + ("" if ok1 else "   <-- FAIL"))

# 2. CSV round trip, and the GUI's loader
rig_t = prigs.rig_A()
wand = prigs.make_wand(rig_t, 300, rng, noise_px=0.1)
uv = wand.uv.copy()
uv[5, 2] = np.nan                                    # a camera missing from a frame
uv[7, 1, 0] = np.nan                                 # a camera seeing only one end (gets no rows)
csv_path = tmp / "wand.csv"
n_rows = pio.write_gui_csv(str(csv_path), uv)
cd = pio.read_gui_csv(str(csv_path))
expect = uv.copy()
expect[7, 1] = np.nan
same_nan = bool(np.array_equal(np.isnan(cd.uv), np.isnan(expect)))
dmax = float(np.nanmax(np.abs(cd.uv - expect)))
ok2 = same_nan and dmax <= 5e-7 and cd.uv.shape == uv.shape
gui = {}
try:
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
    base = WandCalibrator()
    okg, msg = base.load_wand_data_from_csv(str(csv_path))
    gui = {"ok": bool(okg), "message": msg, "frames": len(base.wand_points)}
    # the GUI keeps a frame when every camera listed in it has both ends: all 300 here
    ok2 &= okg and len(base.wand_points) == 300
except Exception as exc:  # noqa: BLE001
    gui = {"error": repr(exc)}
ok_all &= ok2
res["csv"] = {"filtered_rows": n_rows, "max_diff_px": dmax, "same_missing": same_nan, "gui_loader": gui,
              "pass": bool(ok2)}
log(f"2. CSV round trip: {n_rows} Filtered rows; detections back to {dmax:.1e} px, missing ones identical: {same_nan}; "
    f"the GUI's loader: {gui}" + ("" if ok2 else "   <-- FAIL"))

# 3. a result written and reloaded
data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
sol = pw.calibrate(data, pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx) for w in rig_t.wins], rig_t.cam_win))
out = HERE / "w9_result"
if out.exists():
    shutil.rmtree(out)
result = pio.write_result(sol, str(out), names=[f"cam{k + 1}" for k in range(rig_t.n_cams)])
worst3 = 0.0
XA, XB = sol.frames.ends(sol.L)
P = np.vstack([XA[:200], XB[:200]])
for c in range(rig_t.n_cams):
    cpp = lpt.Camera(str(out / "camFile" / f"cam{c + 1}.txt"))
    u_cpp = cm.cpp_project(cpp, P)
    pr = pm.project(pm.rows([sol.cams[c]], np.zeros(len(P), int)),
                    pm.wrows([sol.rig.windows[sol.rig.cam_win[c]]], np.zeros(len(P), int)), P)
    worst3 = max(worst3, float(np.nanmax(np.linalg.norm(u_cpp - pr.uv, axis=1))))
files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
report = (out / "calibration_report.txt").read_text()
ok3 = worst3 <= 1e-6 and "PINPLATE" in report and len(files) == rig_t.n_cams + 2
ok_all &= ok3
res["result"] = {"files": files, "worst_px": worst3, "rms_px": sol.rms, "pass": bool(ok3)}
log(f"3. result written ({', '.join(files)}); camFiles reproduce the solution's projections to {worst3:.2e} px")
log("   report:")
for ln in report.splitlines():
    log("   | " + ln)
shutil.rmtree(tmp)
res["pass"] = bool(ok_all)
log("W9: " + ("PASS" if ok_all else "FAIL"))
(HERE / "w9_files.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
sys.exit(0 if ok_all else 1)
