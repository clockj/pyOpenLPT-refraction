"""W6 helper: one calibration of one case, run as its own process by w6_vs_gui.py.

    python w6_run.py CASE_DIR gui|new

CASE_DIR holds case.json (the rig's sizes, windows' plates, cam_to_window, wand length and the true f for the GUI's
table) and wand.csv (the GUI's format).  "gui" runs the GUI's current refractive wand calibration headless, exactly
as view.py's RefractiveCalibWorker does (a real WandCalibrator loaded from the CSV provides the wand points and the
precalibration; the camera table gets the true f), writing CASE_DIR/gui_parent/camFile/cam<id>.txt; its own parent
folder keeps its bootstrap cache from being shared.  "new" runs plate_wand.calibrate on the same CSV and writes
CASE_DIR/new/.  Each writes <mode>_status.json (ok, seconds, error) and its console output to <mode>_console.txt.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402,F401  (paths)


def run_gui(case_dir: Path, case: dict):
    from modules.camera_calibration.wand_calibration.refraction_wand_calibrator import RefractiveWandCalibrator
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
    cids = list(range(len(case["sizes"])))
    W, H = case["sizes"][0]
    wc = WandCalibrator()
    ok, msg = wc.load_wand_data_from_csv(str(case_dir / "wand.csv"))
    if not ok:
        raise RuntimeError(f"CSV not loaded: {msg}")
    settings = {cid: {"focal": float(case["f_true"][cid]), "width": int(case["sizes"][cid][0]),
                      "height": int(case["sizes"][cid][1])} for cid in cids}
    wc.cameras = {cid: {} for cid in cids}
    wc.camera_settings = settings
    wc.image_size = (H, W)
    wc.wand_length = float(case["L"])
    wc.dist_coeff_num = 0
    wc.initial_focal = float(case["f_true"][0])

    class MockWandCalibrator:
        pass

    mb = MockWandCalibrator()
    mb.wand_points = wc.wand_points
    mb.wand_points_filtered = getattr(wc, "wand_points_filtered", None)
    mb.active_cam_ids = cids
    mb.cams = wc.cameras
    mb.cameras = wc.cameras
    mb.dist_coeff_num = 0
    mb.wand_length = float(case["L"])
    mb.initial_focal = float(case["f_true"][0])
    mb.image_size = (H, W)
    mb.camera_settings = settings
    mb.run_precalibration_check = wc.run_precalibration_check
    cal = RefractiveWandCalibrator(mb)
    window_media = {wid: {"n1": w["n"][0], "n2": w["n"][1], "n3": w["n"][2], "thickness": w["thickness_mm"][0]}
                    for wid, w in enumerate(case["windows"])}
    cam_to_window = {cid: int(case["cam_to_window"][cid]) for cid in cids}
    out = case_dir / "gui_parent" / "camFile"
    out.parent.mkdir(exist_ok=True)
    success, cam_params, report, dataset = cal.calibrate(num_windows=len(case["windows"]), cam_to_window=cam_to_window,
                                                         window_media=window_media, out_path=str(out), verbosity=1)
    files = sorted(p.name for p in out.glob("cam*.txt")) if out.exists() else []
    return {"success": bool(success), "camfiles": files}


def run_new(case_dir: Path, case: dict):
    import plate_wand as pw
    import plate_wand_io as pio
    cd = pio.read_gui_csv(str(case_dir / "wand.csv"), cam_ids=list(range(len(case["sizes"]))))
    data = pw.WandData(cd.uv, float(case["L"]), [tuple(s) for s in case["sizes"]])
    rig = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, tuple(w["thickness_mm"]), tuple(w["n"]))
                  for w in case["windows"]], case["cam_to_window"])
    sol = pw.calibrate(data, rig)
    res = pio.write_result(sol, str(case_dir / "new"), names=[f"cam{k}" for k in range(len(case["sizes"]))])
    return {"success": True, "rms_px": res["rms_px"], "timing": sol.info["timing_s"]}


def main():
    case_dir = Path(sys.argv[1])
    mode = sys.argv[2]
    case = json.loads((case_dir / "case.json").read_text())
    log_path = case_dir / f"{mode}_console.txt"
    status = {"mode": mode}
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = fh
        try:
            status.update(run_gui(case_dir, case) if mode == "gui" else run_new(case_dir, case))
            status["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            status.update({"ok": False, "error": repr(exc), "trace": traceback.format_exc()[-4000:]})
        finally:
            sys.stdout, sys.stderr = old_out, old_err
    status["seconds"] = time.time() - t0
    (case_dir / f"{mode}_status.json").write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
