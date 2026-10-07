"""P4 (diagnosis before the fixes plan, 2026-10-06-fixes-plan.md): when does a finished GUI detection save only "Raw"
rows, so that loading its CSV gives no valid frames?

    python tests/cylinder_wand_gui/p4_gui_defaults.py [FIRST LAST]      (default: T2 frames 4000-5199, as P3)

The GUI's path, headless and with the library unchanged: WandCalibrator.detect_wand_points_generator with its
autosave CSV (export_wand_data), then the GUI's loader (load_wand_data_from_csv), for the settings a user starts from:
  - "ellipse defaults": the ellipse mode with the GUI's default rod opening 25 px and rod mask 18 px, and the
    default radius range 20-200 px;
  - "fast defaults": the GUI's default mode ("fast") and radius range;
  - "ellipse, case settings": P3's settings (rod opening 95 px, rod mask 50 px, radius 40-120 px), for reference.
All dark balls, sensitivity 0.85.  The pairing's console messages (radius peaks per camera, Pass 1 / Pass 2 counts,
the first rejection reasons) are kept.  The BLAS thread variables are set to 1, as in P3, and the pool is held at 9
workers (os.cpu_count patched in this process only), so the run leaves the PC usable.
Writes p4_gui_defaults_out.txt beside this script and the CSVs to data/p4_<setting>.csv.
"""
import contextlib
import io
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.environ.get("CASE_PYOPENLPT_DIR", ""), os.path.join(REPO, "build-py", "Release"), REPO]

RAW = r"X:\Shijie Zhong\Refraction\Raw\20261006\T2"
SETTINGS = {
    "ellipse_defaults": ("ellipse", 20, 200, {"open_px": 25, "rod_mask_px": 18.0, "guard_px": 3.0, "full": False}),
    "fast_defaults": ("fast", 20, 200, None),
    "ellipse_case": ("ellipse", 40, 120, {"open_px": 95, "rod_mask_px": 50.0, "guard_px": 3.0, "full": False}),
}
KEEP = ("PeakFilter", "Final Stats", "Pass 1", "Pass 2", "Refined", "rejected", "Could not find", "Not enough",
        "No valid stats", "Falling back", "Refinement Done")


def main():
    os.cpu_count = lambda: 12           # the generator starts int(0.8 * cpu_count) = 9 workers
    from modules.camera_calibration.wand_calibration.wand_calibrator import WandCalibrator
    first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (4000, 5199)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    images = {k: [os.path.join(RAW, cam, f"frame{f:06d}.tif") for f in range(first, last + 1)]
              for k, cam in enumerate(("cam1", "cam2"))}
    say(f"P4: the GUI's detection and import with the settings a user starts from; {RAW}, frames {first}-{last}")
    for name, (mode, rmin, rmax, opts) in SETTINGS.items():
        csv_path = os.path.join(HERE, "data", f"p4_{name}.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        wc = WandCalibrator()
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            for _ in wc.detect_wand_points_generator(images, "dark", rmin, rmax, 0.85, autosave_path=csv_path,
                                                     resume=False, detect_mode=mode, detect_opts=opts):
                pass
        dt = time.time() - t0
        n_det = {c: sum(len(v.get(c, [])) for v in wc.wand_data_raw.values()) for c in (0, 1)}
        n_frames_any = sum(1 for v in wc.wand_data_raw.values() if any(len(p) for p in v.values()))
        say(f"\n[{name}] mode {mode}, radius {rmin}-{rmax} px, ellipse options {opts}  ({dt:.0f} s)")
        say(f"  raw detections: {n_det[0]} in cam 0, {n_det[1]} in cam 1; frames with any detection "
            f"{n_frames_any} of {last - first + 1}")
        for ln in buf.getvalue().splitlines():
            if any(k in ln for k in KEEP):
                say(f"  | {ln.strip()}")
        say(f"  paired (wand_points): {len(wc.wand_points)} frames")
        status = {}
        with open(csv_path, encoding="utf-8") as fh:
            rows = fh.read().splitlines()
        for ln in rows[1:]:
            s = ln.split(",")[2]
            status[s] = status.get(s, 0) + 1
        first_filtered = next((i for i, ln in enumerate(rows[1:], 1) if ln.split(",")[2] != "Raw"), None)
        say(f"  the saved CSV: {len(rows) - 1} rows, by status {status}; first non-Raw row: line {first_filtered}")
        ok, msg = WandCalibrator().load_wand_data_from_csv(csv_path)
        say(f"  the GUI's loader on it: {ok}, \"{msg}\"")
    with open(os.path.join(HERE, "p4_gui_defaults_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
