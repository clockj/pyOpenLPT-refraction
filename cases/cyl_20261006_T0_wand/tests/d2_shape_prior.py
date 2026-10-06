"""D2 (diagnosis after T1 and T2): ball centres fitted with the ball's image shape predicted by the calibration.

    python tests/d2_shape_prior.py [--model FOLDER] [--name NAME]

T1 b: with this wand the small ball's centre moves 0.6-2 px RMS when the rod mask changes by 10 px; T1 d: the rod
(2 mm) is about 75 % of the small ball's height in the image, so a mask that removes the rod leaves less than half
the small ball's outline, and a free ellipse (5 parameters) is then poorly fixed along the rod.  T2: residuals
0.45 px RMS, varying from cell to cell (not a smooth model error).  Hypothesis: the centres are the problem.

Here the ellipse's shape is not fitted but predicted: the calibration (default: S2's, in <out_dir>) projects each
ball centre X with its Jacobian J = d(uv)/dX; the ray's direction d at X is J's null vector, and a sphere of radius
rho images (to first order in rho) as the ellipse with matrix rho^2 J (I - d d^T) J^T.  Its aspect and orientation
are held; the centre and the size are fitted (3 parameters) to the edge points of ball_ellipses.radial_edges (the
GUI estimator's edge points), with the rod's strip masked wide enough for any rod angle (T1 d: half-width + 2 sigma,
90th percentile, plus a margin), robustly (the GUI estimator's rejection rule), in two passes.
Steps and outputs:
  1. the predicted shapes against the free fits of S1 (aspect, orientation, size), ball A and ball B;
  2. new centres for every kept frame; their shift from S1; their sensitivity to the mask (+-10 px) and to a 3 px
     shift of the start;
  3. the calibration (cylinder_wand_gui.run, as S2) on the new centres -> <out_dir>/checks/<NAME>/ (default
     d2_shape_prior), and T2's residual statistics on it.
Writes d2_<NAME>_out.txt and .json beside this script, and the new centres to <out_dir>/checks/<NAME>/centres.csv.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

RHO_MM = (1.5, 2.0)                    # ball radii (3 mm and 4 mm balls), end A then B
MASK_PX = {1: 75.0, 2: 66.0}           # T1 d: rule's 90th percentile 72.6 / 63.3 px, plus a margin
MIN_POINTS = 24


# ------------------------------------------------------------------ the predicted shapes
def cameras_from_result(lib, result):
    cams = []
    for c in result["cameras"]:
        cams.append(lib.cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"],
                                  c["width"], c["height"], np.array(c["dist"])))
    return cams


def predicted_shapes(lib, folder):
    """{(frame id, camera id, end): (u, v, a, b, theta)} predicted by the calibration in `folder`."""
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        result = json.load(fh)
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    cams = cameras_from_result(lib, result)
    tube = lib.cm.Tube(tuple(result["tube"]["radii_mm"]), tuple(result["tube"]["n"]))
    ids = [int(c["name"][3:]) for c in result["cameras"]]
    fids = data["frame_ids"]
    ends = [np.array(data["ends_A_mm"], float), np.array(data["ends_B_mm"], float)]
    out = {}
    for k, cam in enumerate(cams):
        for e in (0, 1):
            X = ends[e]
            ok = np.all(np.isfinite(X), axis=1)
            idx = np.flatnonzero(ok)
            rw = lib.cm.rows([cam], np.zeros(len(idx), int))
            pr = lib.cm.project(rw, X[idx], lib.cm.pinhole_start(rw, X[idx]), tube, jac=True)
            for j, i in enumerate(idx):
                if not pr.ok[j]:
                    continue
                J = pr.J_X[j]
                d = np.linalg.svd(J)[2][-1]            # the ray's direction at X: J's null vector
                P = np.eye(3) - np.outer(d, d)
                M = RHO_MM[e] ** 2 * J @ P @ J.T
                lam, vec = np.linalg.eigh(M)
                a, b = math.sqrt(lam[1]), math.sqrt(max(lam[0], 0.0))
                th = math.atan2(vec[1, 1], vec[0, 1])
                out[(int(fids[i]), ids[k], e)] = (float(pr.uv[j, 0]), float(pr.uv[j, 1]), a, b, th)
    return out


# ------------------------------------------------------------------ the constrained fit
def radial_res(p, pts, alpha, th):
    u, v, r = p
    c, s = math.cos(th), math.sin(th)
    dx, dy = pts[:, 0] - u, pts[:, 1] - v
    xe, ye = c * dx + s * dy, -s * dx + c * dy
    rho = np.hypot(xe, ye)
    phi = np.arctan2(ye, xe)
    return rho - 1.0 / np.sqrt((np.cos(phi) / (alpha * r)) ** 2 + (np.sin(phi) / r) ** 2)


def fit_fixed_shape(pts, alpha, th, p0, rounds=4, k_mad=3.5):
    """Centre and size of an ellipse with aspect alpha (a / b) and orientation th held; points beyond k_mad robust
    sigmas dropped (ball_ellipses.robust_ellipse's rule).  Returns (u, v, b, rms, n) or None."""
    keep = np.ones(len(pts), bool)
    p = np.asarray(p0, float)
    for _ in range(rounds):
        if keep.sum() < MIN_POINTS:
            return None
        sol = least_squares(radial_res, p, args=(pts[keep], alpha, th), method="lm")
        p = sol.x
        res = radial_res(p, pts, alpha, th)
        sig = 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))) + 1e-3
        new = np.abs(res) < k_mad * sig + 0.05
        if np.array_equal(new, keep):
            break
        keep = new
    res = radial_res(p, pts, alpha, th)
    return float(p[0]), float(p[1]), float(p[2]), float(np.sqrt(np.mean(res[keep] ** 2))), int(keep.sum())


def refit_task(task):
    """One camera-frame: both balls refitted with the predicted shapes, for each mask and start offset."""
    path, seeds, shapes, masks, offsets, det = task
    cv2.setNumThreads(1)
    be = case.ball_ellipses()
    img = case.read_image(path).astype(np.float32)
    dd = be.settings_for(float(det["max_r_px"]))
    N, _bg, _seg = be.normalised_darkness(img, dd)
    Ns = cv2.GaussianBlur(N, (0, 0), float(dd["smooth_sigma_px"]))
    out = {}
    for m in masks:
        for off in offsets:
            res = []
            for i in (0, 1):
                u0, v0, a0, b0, th0 = seeds[i]
                alpha, th = shapes[i]
                c = np.array([u0 + off, v0 + off])
                o = np.array(seeds[1 - i][:2])
                a_cur, b_cur, th_cur = a0, b0, th0
                fit = None
                for _ in range(2):
                    ct, st = math.cos(th_cur), math.sin(th_cur)
                    rx = math.sqrt((a_cur * ct) ** 2 + (b_cur * st) ** 2)
                    ry = math.sqrt((a_cur * st) ** 2 + (b_cur * ct) ** 2)
                    pts, _angs, complete = be.radial_edges(Ns, c, rx, ry, dd)
                    if not complete or len(pts) < MIN_POINTS:
                        fit = None
                        break
                    e = (o - c) / max(float(np.linalg.norm(o - c)), 1e-9)
                    rel = pts - c[None, :]
                    drop = (np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0]) < m) & (rel @ e > 0)
                    pts = pts[~drop]
                    if len(pts) < MIN_POINTS:
                        fit = None
                        break
                    fit = fit_fixed_shape(pts, alpha, th, (c[0], c[1], b_cur))
                    if fit is None:
                        break
                    c = np.array(fit[:2])
                    a_cur, b_cur, th_cur = alpha * fit[2], fit[2], th
                res.append(fit)
            out[(m, off)] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="result folder whose calibration predicts the shapes")
    ap.add_argument("--name", default="d2_shape_prior")
    args = ap.parse_args()
    cfg = case.load_config()
    model = args.model or cfg["out_dir"]
    log = case.Log(os.path.join(HERE, f"d2_{args.name}" if not args.name.startswith("d2") else args.name))
    lib = case.library()
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    cams = case.cam_ids(cfg)
    log(f"D2 ({args.name}): shapes predicted by the calibration in {model}")
    shapes = predicted_shapes(lib, model)

    # 1. predicted against S1's free fits
    for c in cams:
        for e, name in ((0, "A"), (1, "B")):
            da, dth, ds = [], [], []
            for f in frames:
                key = (f, c, e)
                if key not in shapes:
                    continue
                pu, pv, pa, pb, pth = shapes[key]
                row = kept[(f, c)][e]
                a, b, th = row[4], row[5], row[6]
                da.append((a / b) / (pa / pb) - 1.0)
                d = (th - pth + math.pi / 2) % math.pi - math.pi / 2
                dth.append(math.degrees(d))
                ds.append(math.sqrt(a * b) / math.sqrt(pa * pb) - 1.0)
            da, dth, ds = map(np.array, (da, dth, ds))
            log(f"  cam{c} ball {name}: {len(da)} fits; free-fit aspect / predicted - 1: median {100 * np.median(da):+.2f} %, "
                f"spread (16-84 %) {50 * (np.percentile(da, 84) - np.percentile(da, 16)):.2f} %; orientation difference "
                f"median {np.median(dth):+.2f} deg, spread {0.5 * (np.percentile(dth, 84) - np.percentile(dth, 16)):.2f} deg; "
                f"size / predicted - 1: median {100 * np.median(ds):+.2f} %")

    # 2. the refit
    tasks, keys = [], []
    for f in frames:
        for c in cams:
            if (f, c, 0) not in shapes or (f, c, 1) not in shapes:
                continue
            seeds = [tuple(kept[(f, c)][e][i] for i in (0, 1, 4, 5, 6)) for e in (0, 1)]
            shp = [(shapes[(f, c, e)][2] / shapes[(f, c, e)][3], shapes[(f, c, e)][4]) for e in (0, 1)]
            M = MASK_PX[c]
            sens = (f % 10 == 0)                     # the sensitivity runs on every 10th kept frame
            masks = [M - 10.0, M, M + 10.0] if sens else [M]
            offsets = [0.0, 3.0] if sens else [0.0]
            tasks.append((case.image_path(cfg, c, f), seeds, shp, masks, offsets, cfg["detect"]))
            keys.append((f, c))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=int(cfg["detect"].get("workers", 16))) as ex:
        res = list(ex.map(refit_task, tasks, chunksize=4))
    log(f"  refit {len(tasks)} camera-frames in {time.time() - t0:.0f} s")
    new = {}
    lost = 0
    sens = {"mask-10": [], "mask+10": [], "start+3px": []}
    for (f, c), r in zip(keys, res):
        M = MASK_PX[c]
        main = r[(M, 0.0)]
        if any(x is None for x in main):
            lost += 1
            continue
        new[(f, c)] = main
        for name, key in (("mask-10", (M - 10.0, 0.0)), ("mask+10", (M + 10.0, 0.0)), ("start+3px", (M, 3.0))):
            if key in r and all(x is not None for x in r[key]):
                for e in (0, 1):
                    sens[name].append((e, math.hypot(r[key][e][0] - main[e][0], r[key][e][1] - main[e][1])))
    log(f"  new centres for {len(new)} camera-frames ({lost} lost: too few edge points)")
    shift = {0: [], 1: []}
    for (f, c), fit in new.items():
        for e in (0, 1):
            row = kept[(f, c)][e]
            shift[e].append(math.hypot(fit[e][0] - row[0], fit[e][1] - row[1]))
    for e, name in ((0, "A"), (1, "B")):
        s = np.array(shift[e])
        rr = np.array([new[k][e][3] for k in new])
        log(f"  ball {name}: shift from S1 RMS {np.sqrt(np.mean(s ** 2)):.3f} px, median {np.median(s):.3f}, 99th "
            f"percentile {np.percentile(s, 99):.3f}, largest {s.max():.3f}; edge RMS about the fixed-shape fit median "
            f"{np.median(rr):.3f} px")
    out = {"sensitivity": {}}
    for name, lst in sens.items():
        for e, bname in ((0, "A"), (1, "B")):
            s = np.array([d for (ee, d) in lst if ee == e])
            if len(s):
                out["sensitivity"][f"{name}_{bname}"] = {"n": int(len(s)), "rms": float(np.sqrt(np.mean(s ** 2))),
                                                         "max": float(s.max())}
                log(f"  sensitivity {name:>9s}, ball {bname}: {len(s)} camera-frames, centre moves RMS "
                    f"{np.sqrt(np.mean(s ** 2)):.4f} px, largest {s.max():.4f} px")

    folder = os.path.join(cfg["out_dir"], "checks", args.name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "centres.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "camera", "end", "u", "v", "b", "edge_rms", "n_points"])
        for (f, c), fit in sorted(new.items()):
            for e in (0, 1):
                w.writerow([f, c, e] + [f"{x:.5f}" for x in fit[e][:4]] + [fit[e][4]])

    # 3. the calibration on the new centres
    fr = [f for f in frames if all((f, c) in new for c in cams)]
    uv = np.full((len(fr), len(cams), 2, 2), np.nan)
    for i, f in enumerate(fr):
        for j, c in enumerate(cams):
            for e in (0, 1):
                uv[i, j, e] = new[(f, c)][e][:2]
    job = case.make_job(lib, cfg, fr, uv)
    log(f"  calibration on {len(fr)} frames -> {folder}")
    _sol, result, secs = case.run_calibration(lib, job, folder, log)
    for line in lib.cg.summary_lines(result):
        log("    " + line)
    s = case.summary(result)
    s["seconds"] = secs
    out["summary"] = s
    log.save(out)


if __name__ == "__main__":
    main()
