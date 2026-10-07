"""D2 (for the question "shall we just remove [the fast detection]?"): the three detection modes on images whose
truth is known, as the GUI runs them (run_detection_task, Wand Type "Dark on Bright").

    python d2_fast_mode.py

60 synthetic images of a wand in air, 1280 x 800 px (data/d2/): a bright backlight with a gentle gradient (180-220),
two dark balls (grey 40) of radii 22-30 px and 38-50 px at random places at least 30 px apart, the rod between them
and a handle out of the large ball (6 px wide), the edges blurred (Gaussian, sigma 1.5 px), noise of 2 grey levels.
The balls are drawn with sub-pixel centres (cv2.circle with shift and anti-aliasing).
Per mode (fast, reliable, ellipse with its defaults; Radius Range 15-70 px; Sensitivity 0.85):
  - found: the images in which both balls are found (a detection within 5 px of each true centre);
  - centre and radius errors of the found balls (median, 95th percentile, largest);
  - extra points returned per image;
  - time per image (one process, as each worker runs);
  - fast only: a second run (other random draws): the largest change of a found ball's centre.
Also W1's six T2 frames (the tube recording; no truth): points per image, and fast's change between two runs.
(The first try ran without the C++ module on the path, so "reliable" and fast's fallback silently found nothing:
d2_fast_mode_first_try_out.txt.)
Writes d2_fast_mode_out.txt and .json.
"""
import common as C  # noqa: I001

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("d2_fast_mode")
N, W, H = 60, 1280, 800
RMIN, RMAX, SENS = 15, 70, 0.85


def render(rng, path):
    import cv2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    gx, gy = rng.uniform(-1, 1, 2)
    bg = 200.0 + 20.0 * (gx * (xx / W - 0.5) + gy * (yy / H - 0.5))     # 180-220
    img = np.zeros((H, W), np.uint8)                       # the dark mask (255 = object), drawn anti-aliased
    rs, rl = rng.uniform(22, 30), rng.uniform(38, 50)
    while True:
        a = np.array([rng.uniform(rl + 20, W - rl - 20), rng.uniform(rl + 20, H - rl - 20)])
        b = np.array([rng.uniform(rl + 20, W - rl - 20), rng.uniform(rl + 20, H - rl - 20)])
        if np.linalg.norm(a - b) > rs + rl + 30 and np.linalg.norm(a - b) < 500:
            break
    s = 8                                                  # sub-pixel bits

    def P(p):
        return (int(round(p[0] * 2 ** s)), int(round(p[1] * 2 ** s)))
    cv2.line(img, P(a), P(b), 255, 6, cv2.LINE_AA, s)
    d = (b - a) / np.linalg.norm(b - a)
    handle_end = b + d * 200.0
    cv2.line(img, P(b), P(handle_end), 255, 6, cv2.LINE_AA, s)
    cv2.circle(img, P(a), int(round(rs * 2 ** s)), 255, -1, cv2.LINE_AA, s)
    cv2.circle(img, P(b), int(round(rl * 2 ** s)), 255, -1, cv2.LINE_AA, s)
    cover = img.astype(np.float64) / 255.0
    out = bg * (1 - cover) + 40.0 * cover
    out = cv2.GaussianBlur(out, (0, 0), 1.5)
    out = out + rng.normal(0, 2.0, out.shape)
    cv2.imwrite(str(path), np.clip(np.rint(out), 0, 255).astype(np.uint8))
    return [(a[0], a[1], rs), (b[0], b[1], rl)]


def score(pts, truth):
    """For each true ball: (centre error, radius error) of the nearest detection within 5 px, or None."""
    out = []
    P = np.zeros((0, 3)) if pts is None else np.asarray(pts, float)[:, :3]
    for (x, y, r) in truth:
        if not len(P):
            out.append(None)
            continue
        dd = np.hypot(P[:, 0] - x, P[:, 1] - y)
        k = int(np.argmin(dd))
        out.append((float(dd[k]), float(P[k, 2] - r)) if dd[k] <= 5.0 else None)
    return out


def main():
    import openlpt  # noqa: F401  (as the GUI: the local pyopenlpt build first on the path; without it "reliable"
    #                               and fast's fallback find nothing: run_detection_task returns None on ImportError)
    import pyopenlpt  # noqa: F401  (fails loudly here instead)
    from modules.camera_calibration.wand_calibration.wand_calibrator import run_detection_task
    folder = C.DATA / "d2"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    truth = {}
    for i in range(N):
        p = folder / f"img_{i:03d}.png"
        truth[i] = render(rng, p)
    log(f"D2: {N} synthetic images of a wand in air ({W} x {H} px), radii 22-30 / 38-50 px, blur 1.5 px, noise 2")
    modes = {"fast": None, "reliable": None,
             "ellipse": {"open_px": 25, "rod_mask_px": 18.0, "guard_px": 3.0, "full": False}}
    res, summary = {}, {}
    for mode, opts in modes.items():
        res[mode], times = {}, []
        for i in range(N):
            np.random.seed(1000 + i)
            t0 = time.perf_counter()
            _f, _c, pts = run_detection_task((i, 0, str(folder / f"img_{i:03d}.png"), "dark", RMIN, RMAX, SENS, mode, opts))
            times.append(time.perf_counter() - t0)
            res[mode][i] = pts
        found, ce, re_, extra = 0, [], [], []
        for i in range(N):
            sc = score(res[mode][i], truth[i])
            if all(s is not None for s in sc):
                found += 1
            ce += [s[0] for s in sc if s is not None]
            re_ += [s[1] for s in sc if s is not None]
            extra.append(0 if res[mode][i] is None else max(0, len(res[mode][i]) - 2))
        rec = {"found_both": found, "centre_err_px": [float(np.median(ce)), float(np.percentile(ce, 95)), float(max(ce))]
               if ce else None, "radius_err_px": [float(np.median(np.abs(re_))), float(np.percentile(np.abs(re_), 95)),
                                                  float(max(np.abs(re_)))] if re_ else None,
               "extra_points_mean": float(np.mean(extra)), "seconds_per_image": float(np.median(times))}
        if mode == "fast":
            ch = []
            for i in range(N):
                np.random.seed(5000 + i)
                _f, _c, pts2 = run_detection_task((i, 0, str(folder / f"img_{i:03d}.png"), "dark", RMIN, RMAX, SENS,
                                                   "fast", None))
                s1, s2 = score(res["fast"][i], truth[i]), score(pts2, truth[i])
                for a_, b_ in zip(s1, s2):
                    if a_ is not None and b_ is not None:
                        p1 = np.asarray(res["fast"][i], float)[:, :2]
                        p2 = np.asarray(pts2, float)[:, :2]
                        # the matched detections' centres: nearest to each other
                        for q in p1:
                            ch.append(float(np.min(np.hypot(p2[:, 0] - q[0], p2[:, 1] - q[1]))))
                        break
            rec["second_run_change_px"] = [float(np.median(ch)), float(np.percentile(ch, 95)), float(max(ch))] if ch else None
        summary[mode] = rec
        log(f"  {mode:<9}: both balls in {found} of {N}; centre error median/95%/max "
            + ("/".join(f"{v:.3f}" for v in rec["centre_err_px"]) if rec["centre_err_px"] else "-")
            + " px; radius error " + ("/".join(f"{v:.2f}" for v in rec["radius_err_px"]) if rec["radius_err_px"] else "-")
            + f" px; extra points {rec['extra_points_mean']:.1f}; {1000 * rec['seconds_per_image']:.0f} ms per image"
            + (f"; second run: change median/95%/max {'/'.join(f'{v:.3f}' for v in rec['second_run_change_px'])} px"
               if rec.get("second_run_change_px") else ""))

    # W1's T2 frames (the tube): points per image and fast's change between two runs
    t2 = sorted((C.DATA / "w1" / "dark").glob("cam*/*.png"))
    t2_rec = {}
    if t2:
        opts_t2 = {"open_px": 95, "rod_mask_px": 50.0, "guard_px": 3.0, "full": False}
        for mode in ("fast", "reliable", "ellipse"):
            n_pts, ch, t_ = [], [], []
            for p in t2:
                np.random.seed(11)
                t0 = time.perf_counter()
                _f, _c, a = run_detection_task((0, 0, str(p), "dark", 40, 120, SENS, mode,
                                                opts_t2 if mode == "ellipse" else None))
                t_.append(time.perf_counter() - t0)
                n_pts.append(0 if a is None else len(a))
                if mode == "fast" and a is not None:
                    np.random.seed(12)
                    _f, _c, b = run_detection_task((0, 0, str(p), "dark", 40, 120, SENS, "fast", None))
                    if b is not None:
                        A_, B_ = np.asarray(a, float)[:, :2], np.asarray(b, float)[:, :2]
                        ch += [float(np.min(np.hypot(B_[:, 0] - q[0], B_[:, 1] - q[1]))) for q in A_]
            t2_rec[mode] = {"points": n_pts, "seconds_per_image": float(np.median(t_)),
                            "second_run_change_max_px": max(ch) if ch else None}
            log(f"  T2 ({len(t2)} images), {mode:<9}: points per image {n_pts}; {1000 * float(np.median(t_)):.0f} ms per "
                f"image" + (f"; fast's second run: largest change {max(ch):.2f} px" if ch else ""))
    return log.save({"synthetic": summary, "t2": t2_rec})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
