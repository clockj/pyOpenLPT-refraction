"""P3 (diagnosis before the fixes plan, 2026-10-06-fixes-plan.md): does the GUI's import of its own wand points work
when its detection finishes?

    python tests/cylinder_wand_gui/p3_import_roundtrip.py [FIRST LAST]      (default: T2 frames 4000-5199)

The GUI's path, headless and with the library unchanged: WandCalibrator.detect_wand_points_generator in "ellipse"
mode with the GUI's detect_opts (rod opening 95 px, rod mask 50 px, guard 3 px), dark balls, radius 40-120 px, and
its autosave CSV (export_wand_data); then the GUI's "Load Wand Points (from CSV)" (load_wand_data_from_csv) on that
CSV; then what "Pinhole+Cylinder" does with the points (cylinder_wand_gui.uv_from_wand_points, Job.check).  The BLAS
thread variables are set to 1 here, so that the 25 workers fit in this PC's memory (P2); the GUI does not set them.
Writes p3_import_roundtrip_out.txt beside this script and the CSV to data/p3_wand_points_gui.csv.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.environ.get("CASE_PYOPENLPT_DIR", ""), os.path.join(REPO, "build-py", "Release"), REPO]

import numpy as np  # noqa: E402

RAW = r"X:\Shijie Zhong\Refraction\Raw\20261006\T2"


def main():
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
    from modules.camera_calibration.curved_calibration import cylinder_wand_gui as cg
    first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (4000, 5199)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    csv_path = os.path.join(HERE, "data", "p3_wand_points_gui.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    images = {k: [os.path.join(RAW, cam, f"frame{f:06d}.tif") for f in range(first, last + 1)]
              for k, cam in enumerate(("cam1", "cam2"))}
    opts = {"open_px": 95, "rod_mask_px": 50.0, "guard_px": 3.0, "full": False}     # view._get_wand_detect_opts
    say(f"P3: the GUI's detection, export and import on {RAW}, frames {first}-{last}")
    wc = WandCalibrator()
    for _ in wc.detect_wand_points_generator(images, "dark", 40, 120, 0.85, autosave_path=csv_path, resume=False,
                                             detect_mode="ellipse", detect_opts=opts):
        pass
    say(f"  detection: {len(wc.wand_points)} frames paired (small and large ball in both cameras) of "
        f"{len(wc.wand_data_raw)}")
    with open(csv_path, encoding="utf-8") as fh:
        rows = fh.read().splitlines()
    status = {}
    for ln in rows[1:]:
        s = ln.split(",")[2]
        status[s] = status.get(s, 0) + 1
    say(f"  the autosaved CSV: {len(rows) - 1} rows, by status {status}")
    wc2 = WandCalibrator()
    ok, msg = wc2.load_wand_data_from_csv(csv_path)
    say(f"  the GUI's loader: {ok}, \"{msg}\"")
    uv1, f1, c1 = cg.uv_from_wand_points(wc.wand_points)
    uv2, f2, c2 = cg.uv_from_wand_points(wc2.wand_points)
    same = f1 == f2 and uv1.shape == uv2.shape
    say(f"  the same frames and cameras: {same} ({len(f1)} frames, cameras {c1})"
        + (f"; largest centre difference {np.nanmax(np.abs(uv1 - uv2)):.1e} px (the CSV keeps 3 decimals)" if same else ""))
    for name, uv, fids, cids in (("as detected", uv1, f1, c1), ("as loaded", uv2, f2, c2)):
        job = cg.Job(uv=uv, frame_ids=fids, cam_ids=cids, sizes=[(1280, 800)] * len(cids), wand_length=10.0,
                     radii=(100.80625, 97.63125), n=(1.0, 1.49, 1.333))
        try:
            job.check()
            say(f"  Pinhole+Cylinder's input check, {name}: accepted")
        except ValueError as e:
            say(f"  Pinhole+Cylinder's input check, {name}: refused: {e}")
    # what a raw-only file (the GUI's "rescue" export after a failed detection) gives
    raw_only = os.path.join(HERE, "data", "p3_raw_only.csv")
    with open(raw_only, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join([rows[0]] + [ln for ln in rows[1:] if ln.split(",")[2] == "Raw"]) + "\n")
    wc3 = WandCalibrator()
    ok, msg = wc3.load_wand_data_from_csv(raw_only)
    say(f"  the same CSV with only its Raw rows (as a rescue export after a failed detection): {ok}, \"{msg}\"")
    with open(os.path.join(HERE, "p3_import_roundtrip_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
