"""G2: without pixel sizes the detection is unchanged (plan: 2026-10-08-plan.md, test G2).  No Qt.

    python g2_unscaled_unchanged.py

The code before this change is git HEAD's wand_calibrator.py, written to before_lib/wand_calibrator_head.py and
loaded from there.  On 6 image numbers of run 20261008/T0, all three cameras, in the "ellipse" mode (cams 1-2's
settings) and the "fast" mode (Radius Range 20-70 px):
  1. detect_single_frame now, with cam_scales None and with every scale 1, against HEAD's: bitwise the same rows;
  2. detect_wand_points_generator now (cam_scales None) against HEAD's, ellipse mode: the same raw balls.
Writes g2_unscaled_unchanged_out.txt and .json beside itself.
"""
import common as C  # noqa: I001

import importlib.util  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("g2_unscaled_unchanged")
IMAGES = [3000, 4800, 6600, 8400, 10200, 12000]
OPTS = {"open_px": 51, "rod_mask_px": 28.0, "guard_px": 3.0, "full": True}


def head_module():
    d = C.HERE / "before_lib"
    d.mkdir(exist_ok=True)
    path = d / "wand_calibrator_head.py"
    src = subprocess.run(["git", "show", "HEAD:modules/camera_calibration/wand_calibration/wand_calibrator.py"],
                         cwd=str(C.REPO), capture_output=True, check=True).stdout
    path.write_bytes(src)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("wand_calibrator_head", str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules["wand_calibrator_head"] = m
    spec.loader.exec_module(m)
    return m


def main():
    from modules.camera_calibration.wand_calibration import wand_calibrator as wc
    old = head_module()
    log(f"HEAD's code: {old.__file__}")
    out = {"images": IMAGES}

    log("\n1. detect_single_frame: now (scales None / all 1) against HEAD")
    diffs = []
    n = 0
    for mode, opts in (("ellipse", OPTS), ("fast", None)):
        for f in IMAGES:
            paths = {k: C.frame_path(k + 1, f) for k in range(3)}
            ref = old.WandCalibrator().detect_single_frame(paths, "dark", 20, 70, 0.85, mode, detect_opts=opts)
            for scales in (None, {0: 1.0, 1: 1.0, 2: 1.0}):
                now = wc.WandCalibrator().detect_single_frame(paths, "dark", 20, 70, 0.85, mode, detect_opts=opts,
                                                              cam_scales=scales)
                for k in range(3):
                    n += 1
                    if not C.same_rows(now.get(k), ref.get(k)):
                        diffs.append((mode, f, k, "none" if scales is None else "ones"))
    log.check("every camera-frame, both modes, both ways of 'no scaling': bitwise the same as HEAD", not diffs,
              f"{n} comparisons; different: {diffs[:6]}")

    log("\n2. detect_wand_points_generator (ellipse): now against HEAD")
    paths = {k: [C.frame_path(k + 1, f) for f in IMAGES[:4]] for k in range(3)}
    raws = []
    for mod in (old, wc):
        cal = mod.WandCalibrator()
        tmp = os.path.join(tempfile.mkdtemp(prefix="g2_"), "wand_points.csv")
        for _ in cal.detect_wand_points_generator(paths, "dark", 20, 70, 0.85, autosave_path=tmp,
                                                  detect_mode="ellipse", detect_opts=OPTS):
            pass
        raws.append(cal.wand_data_raw)
    a, b = raws
    same = set(a) == set(b) and all(set(a[f]) == set(b[f]) and all(np.array_equal(np.asarray(a[f][k], float),
                                                                                  np.asarray(b[f][k], float))
                                                                   for k in a[f]) for f in a)
    log.check("the same raw balls in every camera-frame", same, f"{sum(len(v) for v in a.values())} camera-frames "
              "with balls")
    log.save(out)


if __name__ == "__main__":
    main()
