"""S0 (plan: 2026-10-08-plan.md, Phase T): which detector for T1's images, dots on cams 2-3 and lines on cam1?

    python tests/s0_detector.py [--frames 0 25 50 ... ] [--workers 10]
    python tests/s0_detector.py --k 6 5 4 --frames 20 70 120 170 220 --tag k   # the line detector's threshold k_detect

On 10 of S1's frames, two detectors on every camera:
  - "lines": 7 Oct's line detector (the C++ LineFinder via lpt.detectLines, S1's noise), which models a particle as a
    segment of any length (a dot is a segment of length about 0);
  - "peaks": OpenLPT's standard tracer finder (ObjectFinder2D, radius 2 px, intensity threshold 4 x each camera's
    8-bit noise, as today's code would be set).
Truth-free score: the three-camera matches, as S2 gathers them (s2_shape_fit.gather: the pairs of cams 1-2 within
8 px of the epipolar curve, triangulated, and a cam3 detection within 1.5 px of the projection), counted in the
signal band (d <= 1.5 px) minus k times the background band (4-8 px): the true triplets per frame.  Random triplets
are rare, so this counts true particles all three cameras detect.  Also the detections per camera.
Writes s0_detector_out.txt and .json beside itself.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
sys.path.insert(0, HERE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402
import s2_shape_fit as s2  # noqa: E402

lpt = s2.lpt


def peaks(img8, noise8):
    cfg = lpt.TracerConfig()
    cfg._radius_obj = 2.0
    cfg._min_obj_int = int(round(4 * noise8))
    objs = lpt.ObjectFinder2D().findObject2D(lpt.Image.from_numpy(img8), cfg)
    if not objs:
        return np.zeros((0, 14))
    # rows shaped like detectLines' (u, v, Lx, Ly, sx, sy, ...): lengths 0, blur 1 px, the rest 0
    out = np.zeros((len(objs), 14))
    out[:, 0] = [o._pt_center[0] for o in objs]
    out[:, 1] = [o._pt_center[1] for o in objs]
    out[:, 4] = out[:, 5] = 1.0
    return out


def task(args):
    frames, detector = args[:2]
    k_detect = args[2] if len(args) > 2 else None
    cv2.setNumThreads(1)
    vol = ts.view_volume(lpt)
    cams, sm_ = s2.matcher(vol)
    noise = ts.noise_per_camera()
    P = lpt.LineFinderParam()
    if k_detect is not None:
        P.k_detect = float(k_detect)
    res = []
    for k in frames:
        dets = []
        for c in range(3):
            img = cv2.imread(os.path.join(ts.PRE, f"cam{c + 1}", f"frame{k:06d}.tif"), cv2.IMREAD_UNCHANGED)
            if detector == "lines":
                dets.append(np.array(lpt.detectLines(lpt.Image.from_numpy(img), float(noise[c]), P)))
            else:
                dets.append(peaks(img, noise[c]))
        pairs = s2.candidate_pairs(cams, sm_, dets, vol, 8.0, (0, 1))
        from scipy.spatial import cKDTree
        tree = cKDTree(dets[2][:, :2]) if len(dets[2]) else None
        sig = bg = 0
        for (i, j, X, dd, gap) in pairs:
            if tree is None:
                break
            try:
                q = cams[2].project(lpt.Pt3D(*map(float, X)))
            except RuntimeError:
                continue
            d3, _l = tree.query([q[0], q[1]])
            if d3 > s2.R3:
                continue
            sig += dd <= s2.SIG
            bg += s2.BG0 <= dd <= 8.0
        res.append((k, detector if k_detect is None else f"lines_k{k_detect:g}", [len(d) for d in dets], len(pairs),
                    int(sig), int(bg)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=list(range(0, 250, 25)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--k", type=float, nargs="+", default=None, help="compare the line detector at these k_detect")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    log = case.Log(os.path.join(HERE, "s0_detector" + (f"_{args.tag}" if args.tag else "")))
    noise = ts.noise_per_camera()
    log(f"S0: the line detector against the standard tracer finder (threshold 4 x noise: "
        f"{[int(round(4 * n)) for n in noise]}) on frames {args.frames}")
    from concurrent.futures import ProcessPoolExecutor
    if args.k:
        jobs = [([f], "lines", k) for k in args.k for f in args.frames]
        names = [f"lines_k{k:g}" for k in args.k]
    else:
        jobs = [([f], d) for d in ("lines", "peaks") for f in args.frames]
        names = ["lines", "peaks"]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(task, jobs):
            rows += r
    k_bg = s2.SIG / (8.0 - s2.BG0)
    out = {"frames": args.frames, "k_background": k_bg, "detectors": {}}
    for d in names:
        rr = [r for r in rows if r[1] == d]
        dets = np.array([r[2] for r in rr])
        true = np.array([r[4] - k_bg * r[5] for r in rr])
        out["detectors"][d] = {"detections_median": np.median(dets, axis=0).tolist(),
                               "pairs_median": float(np.median([r[3] for r in rr])),
                               "triplets_signal_median": float(np.median([r[4] for r in rr])),
                               "triplets_background_median": float(np.median([r[5] for r in rr])),
                               "true_triplets_per_frame": float(true.mean()),
                               "true_triplets_per_frame_range": [float(true.min()), float(true.max())]}
        o = out["detectors"][d]
        log(f"  {d}: detections per camera (median) {np.round(o['detections_median']).astype(int).tolist()}; cam1-cam2 "
            f"pairs within 8 px {o['pairs_median']:.0f}; triplets in the signal band {o['triplets_signal_median']:.0f}, "
            f"background band {o['triplets_background_median']:.0f}; true triplets per frame "
            f"{o['true_triplets_per_frame']:.0f} ({o['true_triplets_per_frame_range'][0]:.0f}-"
            f"{o['true_triplets_per_frame_range'][1]:.0f})")
    better = max(out["detectors"], key=lambda d: out["detectors"][d]["true_triplets_per_frame"])
    out["better"] = better
    log(f"  more true three-camera matches per frame: {better}")
    log.save(out)


if __name__ == "__main__":
    main()
