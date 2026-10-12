"""S1 and S2 (B13): the fast detection mode, seeded, is repeatable, and its points moved once by no more than its old
run-to-run spread.

    python s1_s2_fast_seeded.py

Images: D2's 60 synthetic images of a wand in air (data/d2/, re-rendered here with D2's generator for their truth)
and W1's 12 T2 frames (data/w1/dark/: balls through the tube; no truth).  Settings: D2's (Radius Range 15-70 px) and
T2's (40-120 px); Sensitivity 0.85; Wand Type "Dark on Bright".
S1, repeatable:
  - in one process, numpy's global seed set to 1 and then to 2 before each call: the same points to the bit;
  - numpy's global random state the same before and after a call (the fit no longer draws from it);
  - in fresh worker processes: Test Detect's path (detect_single_frame) on 6 images of each set, and Process All
    Frames' path (detect_wand_points_generator, the 60 D2 images as two cameras' frames): the same points as in this
    process, to the bit.
S2, the change: on D2's images, the seeded points against four unseeded runs of git HEAD's detector (before_lib/,
numpy's global seed 11, 12, 13, 14 before each run):
  - the seeded points differ from each unseeded run by no more than 1.5 x the largest difference between two
    unseeded runs;
  - the centre errors against the truth (median, 95 %, largest) within the unseeded runs' range.
Writes s1_s2_fast_seeded_out.txt and .json.
"""
import common as C  # noqa: I001

import importlib.util  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("s1_s2_fast_seeded")
SENS = 0.85


def same(a, b):
    if (a is None) != (b is None):
        return False
    return a is None or np.asarray(a, float).tobytes() == np.asarray(b, float).tobytes() and \
        np.asarray(a).shape == np.asarray(b).shape


def nearest_shift(a, b):
    """The largest distance from a point of a to the nearest point of b (None if either has none)."""
    if a is None or b is None or not len(a) or not len(b):
        return None
    A, B = np.asarray(a, float)[:, :2], np.asarray(b, float)[:, :2]
    return float(max(np.min(np.hypot(B[:, 0] - p[0], B[:, 1] - p[1])) for p in A))


def main():
    import openlpt  # noqa: F401  (the local pyopenlpt build first on the path, as the GUI)
    import pyopenlpt  # noqa: F401
    import modules.camera_calibration.wand_calibration.wand_calibrator as new
    spec = importlib.util.spec_from_file_location("wand_calibrator_before", C.HERE / "before_lib" / "wand_calibrator.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["wand_calibrator_before"] = old
    spec.loader.exec_module(old)
    sys.path.insert(0, str(C.HERE))
    import d2_fast_mode as d2

    folder = C.DATA / "d2"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)                     # D2's images and their truth, as D2 renders them
    truth, d2_paths = {}, []
    for i in range(d2.N):
        p = folder / f"img_{i:03d}.png"
        truth[i] = d2.render(rng, p)
        d2_paths.append(str(p))
    t2_paths = [str(p) for p in sorted((C.DATA / "w1" / "dark").glob("cam*/*.png"))]
    sets = {"D2": (d2_paths, 15, 70), "T2": (t2_paths, 40, 120)}
    log(f"S1/S2: {len(d2_paths)} D2 images, {len(t2_paths)} T2 frames; fast mode, FAST_RANSAC_SEED = "
        f"{new.FAST_RANSAC_SEED}")

    # ---------------- S1
    here = {}
    for name, (paths, rmin, rmax) in sets.items():
        n_same, state_same = 0, 0
        here[name] = []
        for p in paths:
            np.random.seed(1)
            st = np.random.get_state()
            a = new.run_detection_task((0, 0, p, "dark", rmin, rmax, SENS, "fast", None))[2]
            st2 = np.random.get_state()
            state_same += all(np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(st, st2))
            np.random.seed(2)
            b = new.run_detection_task((0, 0, p, "dark", rmin, rmax, SENS, "fast", None))[2]
            n_same += same(a, b)
            here[name].append(a)
        log.check(f"{name}: the same points with numpy's global seed 1 and 2, to the bit", n_same == len(paths),
                  f"{n_same} of {len(paths)}")
        log.check(f"{name}: numpy's global random state unchanged by a call", state_same == len(paths),
                  f"{state_same} of {len(paths)}")
        wc = new.WandCalibrator()
        sub = list(range(0, len(paths), max(1, len(paths) // 6)))[:6]
        n_ok = 0
        for i in sub:
            r = wc.detect_single_frame({0: paths[i]}, "dark", rmin, rmax, SENS, "fast")
            n_ok += same(r.get(0), here[name][i])
        log.check(f"{name}: Test Detect's path (fresh worker processes) gives the same points, to the bit",
                  n_ok == len(sub), f"{n_ok} of {len(sub)} images")
    wc = new.WandCalibrator()
    gen = wc.detect_wand_points_generator({0: d2_paths, 1: d2_paths}, "dark", 15, 70, SENS, detect_mode="fast")
    for _ in gen:
        pass
    n_ok = sum(same(wc.wand_data_raw.get(i, {}).get(c), here["D2"][i]) for i in range(len(d2_paths)) for c in (0, 1))
    log.check("D2: Process All Frames' path (the pool, two cameras of the same images) gives the same points, to the bit",
              n_ok == 2 * len(d2_paths), f"{n_ok} of {2 * len(d2_paths)}")

    # ---------------- S2
    paths, rmin, rmax = sets["D2"]
    unseeded = []
    for s in (11, 12, 13, 14):
        run = []
        for p in paths:
            np.random.seed(s)
            run.append(old.run_detection_task((0, 0, p, "dark", rmin, rmax, SENS, "fast", None))[2])
        unseeded.append(run)

    def max_shift(r1, r2):
        v = [nearest_shift(a, b) for a, b in zip(r1, r2)]
        return max(x for x in v if x is not None)
    between = [max_shift(unseeded[i], unseeded[j]) for i in range(4) for j in range(i + 1, 4)]
    to_seeded = [max_shift(here["D2"], u) for u in unseeded]
    log.check("D2: the seeded points differ from each unseeded run by no more than 1.5 x two unseeded runs do",
              max(to_seeded) <= 1.5 * max(between),
              f"seeded against unseeded: {', '.join(f'{v:.3f}' for v in to_seeded)} px; between unseeded runs: "
              f"{', '.join(f'{v:.3f}' for v in between)} px")

    def errors(run):
        ce = []
        for i, pts in enumerate(run):
            for (x, y, _r) in truth[i]:
                if pts is None or not len(pts):
                    continue
                P = np.asarray(pts, float)
                d = np.hypot(P[:, 0] - x, P[:, 1] - y)
                if d.min() <= 5:
                    ce.append(float(d.min()))
        return [float(np.median(ce)), float(np.percentile(ce, 95)), float(max(ce)), len(ce)]
    e_new = errors(here["D2"])
    e_old = [errors(u) for u in unseeded]
    lo = [min(e[k] for e in e_old) for k in range(3)]
    hi = [max(e[k] for e in e_old) for k in range(3)]
    tol = max(between)
    within = all(lo[k] - tol <= e_new[k] <= hi[k] + tol for k in range(3)) and e_new[3] == 2 * len(paths)
    log.check("D2: the accuracy is as before (median, 95 %, largest centre error within the unseeded runs' range, "
              "give or take their spread); every ball found", within,
              f"seeded {e_new[0]:.3f} / {e_new[1]:.3f} / {e_new[2]:.3f} px ({e_new[3]} balls); unseeded "
              + "; ".join(f"{e[0]:.3f} / {e[1]:.3f} / {e[2]:.3f}" for e in e_old) + f" px; spread {tol:.3f} px")
    return log.save({"seeded_errors": e_new, "unseeded_errors": e_old, "between_unseeded_px": between,
                     "seeded_to_unseeded_px": to_seeded})


if __name__ == "__main__":
    ok = main()
    sys.stdout.flush()
    os._exit(0 if ok else 1)
