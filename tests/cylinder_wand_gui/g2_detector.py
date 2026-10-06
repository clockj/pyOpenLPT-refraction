"""G2: the "ellipse" detection mode inside the GUI's detection function (plan: 2026-10-06-plan.md, section 5).

    python g2_detector.py

On rendered images through the real tube (common.render_set), backlit (dark balls, the GUI's "Dark on Bright") and
evenly front-lit (bright balls, "Bright on Dark"):
  (a) the same centres as T13's estimator: wand_calibrator.run_detection_task(..., "ellipse") against
      t13_ellipse.detect_balls with the rod mask and the 3 px overlap guard (tests/cylinder_wand, T13b2), called the
      same way, must give identical centres (Radius Range up to 40 px, so the background window is T13's 101 px);
  (b) the accuracy T13 found: against the exact images of the balls' centres, RMS within 0.035 px, mean error
      within 0.01 px per coordinate, largest error under 0.25 px, no centre more than 1 px off, smallest blob = ball A;
  (c) with a Radius Range up to 60 px (a wider background window) the centres move by less than 0.02 px RMS;
  (d) "fast" and "reliable" unchanged: the new run_detection_task returns byte-identical arrays to HEAD's (loaded
      from git) on the same images, with 8-argument and 9-argument tasks.  "fast" samples its RANSAC from numpy's
      global generator, so it is seeded alike before every call (unseeded, two calls of the same code differ:
      the first run of this test compared them unseeded and failed, 12 of 52).
Writes g2_detector_out.txt and g2_detector.json.
"""
import common as C  # noqa: I001  (first: paths and one BLAS thread)

import importlib.util  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import t13_ellipse as te  # noqa: E402
from modules.camera_calibration.wand_calibration import wand_calibrator as wc_new  # noqa: E402

log = C.Log("g2_detector")
OPEN_PX, MASK_PX, GUARD_PX = 25, 18.0, 3.0
SETS = [("g2_backlit", 60, 1306, "backlit", "dark"), ("g2_even", 60, 1307, "even", "bright")]


def head_module():
    """HEAD's wand_calibrator.py (before this work), imported from git as a separate module."""
    src = subprocess.run(["git", "show", "HEAD:modules/camera_calibration/wand_calibration/wand_calibrator.py"],
                         cwd=str(C.REPO), capture_output=True, text=True, check=True,
                         env={**C.os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout
    path = C.HERE / "data" / "wand_calibrator_head.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("wand_calibrator_head", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wand_calibrator_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def t13_reference(img, polarity):
    """T13b2's detect_guarded: the rod-masked ellipse estimator and the overlap guard."""
    r = te.detect_balls(img, polarity, OPEN_PX, rod_mask_px=MASK_PX)
    if len(r) >= 2 and te_gap(r[0], r[1]) < GUARD_PX:
        return []
    return [(x[0], x[1]) for x in r]


def te_gap(e1, e2):
    du, dv = e2[0] - e1[0], e2[1] - e1[1]
    phi = math.atan2(dv, du)

    def rad(e, ang):
        return 1.0 / math.sqrt((math.cos(ang - e[5]) / e[3]) ** 2 + (math.sin(ang - e[5]) / e[4]) ** 2)
    return math.hypot(du, dv) - rad(e1, phi) - rad(e2, phi + math.pi)


def main():
    log("G2: the 'ellipse' detection mode in wand_calibrator.run_detection_task")
    res = {}
    for name, n, seed, lighting, wand_type in SETS:
        t0 = time.time()
        folder = C.render_set(name, n, seed, lighting)
        truth = C.load_truth(folder)
        lists = C.image_lists(folder)
        log(f"\n{name}: {n} frames, {lighting} ('{wand_type}' wand), rendered in {time.time() - t0:.0f} s")
        same, diff_cf, errs, wrong, dropped, missing, n_cf = 0, [], [], 0, 0, 0, 0
        moved = []
        t_det = 0.0
        for c in range(3):
            for f in range(n):
                tA, tB = truth["UV"][f, c]
                if not (np.all(np.isfinite(tA)) and np.all(np.isfinite(tB))):
                    continue
                n_cf += 1
                path = lists[c][f]
                opts = {"open_px": OPEN_PX, "rod_mask_px": MASK_PX, "guard_px": GUARD_PX}
                t1 = time.time()
                _f, _c, rows = wc_new.run_detection_task((f, c, path, wand_type, 10.0, 40.0, 0.85, "ellipse", opts))
                t_det += time.time() - t1
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                ref = t13_reference(img, "dark" if wand_type == "dark" else "bright")
                got = [] if rows is None else [(float(r[0]), float(r[1])) for r in rows]
                if got == [(float(u), float(v)) for u, v in ref]:
                    same += 1
                else:
                    diff_cf.append((f, c, got, ref))
                if rows is None:
                    if ref == []:
                        dropped += 1
                    missing += 1
                    continue
                if len(got) < 2:
                    missing += 1
                    continue
                eA = np.array(got[0]) - tA
                eB = np.array(got[1]) - tB
                if max(np.hypot(*eA), np.hypot(*eB)) > 1.0:
                    wrong += 1
                    continue
                errs += [eA, eB]
                _f2, _c2, rows60 = wc_new.run_detection_task((f, c, path, wand_type, 10.0, 60.0, 0.85, "ellipse",
                                                              opts))
                if rows60 is not None and len(rows60) >= 2:
                    moved += [rows60[0][:2] - np.array(got[0]), rows60[1][:2] - np.array(got[1])]
        E = np.array(errs)
        M = np.array(moved)
        rms = float(np.sqrt(np.mean(np.sum(E ** 2, axis=1)))) if len(E) else float("nan")
        mean = E.mean(axis=0) if len(E) else np.array([np.nan, np.nan])
        mx = float(np.max(np.hypot(E[:, 0], E[:, 1]))) if len(E) else float("nan")
        mv = float(np.sqrt(np.mean(np.sum(M ** 2, axis=1)))) if len(M) else float("nan")
        log(f"  camera-frames with both balls seen whole: {n_cf}; measured {n_cf - missing}, not measured {missing} "
            f"(of which {dropped} dropped by the overlap guard in both codes); {t_det / max(n_cf, 1) * 1e3:.0f} ms per "
            "image")
        log(f"  error against the exact centres: RMS {rms:.4f} px over {len(E)} balls, mean ({mean[0]:+.4f}, "
            f"{mean[1]:+.4f}) px, largest {mx:.3f} px; more than 1 px off: {wrong}")
        log(f"  Radius Range up to 60 px instead of 40: the centres move by {mv:.4f} px RMS")
        log.check(f"{name} (a): identical to T13's estimator and guard", same == n_cf,
                  f"{same}/{n_cf} camera-frames identical" + (f"; first difference {diff_cf[0]}" if diff_cf else ""))
        log.check(f"{name} (b): accuracy as T13 (RMS <= 0.035, |mean| <= 0.01, max < 0.25 px, none > 1 px)",
                  rms <= 0.035 and np.all(np.abs(mean) <= 0.01) and mx < 0.25 and wrong == 0,
                  f"RMS {rms:.4f}, mean ({mean[0]:+.4f}, {mean[1]:+.4f}), max {mx:.3f}, wrong {wrong}")
        log.check(f"{name} (c): a wider background window moves the centres < 0.02 px RMS", mv < 0.02, f"{mv:.4f} px")
        res[name] = {"camera_frames": n_cf, "identical": same, "not_measured": missing, "guard_dropped": dropped,
                     "rms_px": rms, "mean_px": mean.tolist(), "max_px": mx, "wrong": wrong, "moved_rms_px": mv,
                     "ms_per_image": t_det / max(n_cf, 1) * 1e3}

    # (d) fast and reliable unchanged
    log("\n(d) 'fast' and 'reliable' against HEAD's run_detection_task")
    wc_head = head_module()
    folder = C.DATA / "g2_backlit"
    lists = C.image_lists(folder)
    picks = [(f, c) for f in range(0, 60, 6) for c in (0, 1)][:20]
    n_same = n_all = 0
    first = None
    for mode, sub in (("fast", picks), ("reliable", picks[:6])):
        t0 = time.time()
        for f, c in sub:
            args8 = (f, c, lists[c][f], "dark", 10.0, 60.0, 0.85, mode)
            # "fast" draws its RANSAC samples from numpy's global generator: seed it alike before every call
            np.random.seed(1000 * f + c)
            _a, _b, old = wc_head.run_detection_task(args8)
            np.random.seed(1000 * f + c)
            _a, _b, new8 = wc_new.run_detection_task(args8)
            np.random.seed(1000 * f + c)
            _a, _b, new9 = wc_new.run_detection_task(args8 + (None,))
            for new in (new8, new9):
                n_all += 1
                ok = (old is None and new is None) or (old is not None and new is not None and old.dtype == new.dtype
                                                        and old.shape == new.shape and old.tobytes() == new.tobytes())
                n_same += ok
                if not ok and first is None:
                    first = (mode, f, c, old, new)
        log(f"  {mode}: {len(sub)} images in {time.time() - t0:.0f} s")
    log.check("(d) 'fast' and 'reliable' byte-identical to HEAD (8- and 9-argument tasks)", n_same == n_all,
              f"{n_same}/{n_all}" + (f"; first difference {first}" if first else ""))
    res["fast_reliable"] = {"identical": n_same, "compared": n_all}
    return log.save({"sets": res})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
