"""T8: robustness of the whole pipeline (from scratch, no hints) to bad data.

Reference rig (three cam1 replicas at 0/45/90 deg), 0.1 px noise, 10 mm wand.

(a) 500 frames in +-20 mm, and a corrupted copy, all at once:
      2 % of the detections moved by 20-200 px in a random direction (gross outliers);
      1 % of the frames with A and B swapped in one camera;
      10 % of the detections deleted (missing).
    Both calibrated from scratch.  Criteria: every planted outlier and swapped detection rejected; every camera
    parameter of the corrupted fit within 1 sigma of the clean fit (the corrupted data are a subset of the clean
    ones, so the two differ by about 0.4 sigma per parameter); and the corrupted fit consistent with the truth (its
    largest |error| / own sigma over the 19 parameters below 3.5, which a correct fit exceeds with ~1 % chance).
(b) Frames near the wall, on a sensor wide enough to see them.  400 ordinary frames, plus up to 30 frames with an
    end just beyond the fold radius (75.6-76.6 mm) whose point has two images within 30 px of each other in some
    camera (the detection is the C++ primary image; a real blob there would be a blend), plus 30 other frames
    anywhere beyond the fold radius.  Close double images are rare: elsewhere the two images of a point are
    hundreds of px apart.  Criteria: the multi-image guard drops every observation whose point has another image within 30 px of
    the detection; the fit is consistent with the truth (as in (a)).
Outputs: t8_robustness_out.txt, t8_robustness.json.
"""
import json
import time

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

OUT = rigs.HERE / "t8_robustness"
log = rigs.Tee(str(OUT) + "_out.txt")
GUARD_PX = 30.0


def zmax(sol, ref_cams, cov):
    lay = sol.layout
    dl = rigs.local_delta(sol.cams, ref_cams)
    d = np.array([dl[c][j] for c in range(len(ref_cams)) for j in range(len(lay.cols)) if lay.pos[c, j] >= 0])
    return float(np.max(np.abs(d) / np.sqrt(np.diag(cov))))


def part_a():
    tube = rigs.CASE_TUBE
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    sizes = [(c.width, c.height) for c in truth]
    rng = np.random.default_rng(808)
    wd = rigs.wand_frames(truth, tube, 500, 10.0, rng, half=(20, 20, 20), noise_px=0.1)
    clean = cw.calibrate(cw.WandData(wd.uv, 10.0, sizes), tube)
    uv = wd.uv.copy()
    F, Nc = uv.shape[:2]
    det = np.argwhere(np.isfinite(uv[..., 0]))                      # (k, 3): frame, cam, end
    pick = det[rng.choice(len(det), int(round(0.02 * len(det))), replace=False)]
    planted = set()
    for f, c, e in pick:
        ang = rng.uniform(0, 2 * np.pi)
        uv[f, c, e] += rng.uniform(20, 200) * np.array([np.cos(ang), np.sin(ang)])
        planted.add((int(f), int(c), int(e)))
    swapped = set()
    for f in rng.choice(F, int(round(0.01 * F)), replace=False):
        c = int(rng.integers(Nc))
        uv[f, c, [0, 1]] = uv[f, c, [1, 0]]
        swapped |= {(int(f), c, 0), (int(f), c, 1)}
    miss = det[rng.choice(len(det), int(round(0.10 * len(det))), replace=False)]
    for f, c, e in miss:
        uv[f, c, e] = np.nan
    t0 = time.time()
    sol = cw.calibrate(cw.WandData(uv, 10.0, sizes), tube)
    obs = sol.obs
    key = lambda i: (int(obs.frame_ids[obs.frame[i]]), int(obs.cam[i]), int(obs.end[i]))  # noqa: E731
    active = {key(i) for i in np.flatnonzero(obs.active)}
    present = {key(i) for i in range(len(obs))}
    bad = (planted | swapped) & present
    missed = bad & active
    clean_rejected = (present - bad) - active
    dropped_frames = len({k[0] for k in clean_rejected})
    z_clean = zmax(sol, clean.cams, clean.cov)
    z_truth = zmax(sol, truth, sol.cov)
    z_truth_clean = zmax(clean, truth, clean.cov)
    rec = {"planted_outliers": len(planted & present), "swapped_detections": len(swapped & present),
           "missing": int(len(miss)), "outliers_left_active": len(missed),
           "clean_observations_rejected": len(clean_rejected), "frames_they_belong_to": dropped_frames,
           "max_diff_from_clean_sigma": z_clean, "max_error_vs_truth_sigma": z_truth,
           "clean_fit_max_error_vs_truth_sigma": z_truth_clean, "rms_px": sol.rms, "clean_rms_px": clean.rms,
           "seconds": time.time() - t0}
    rec["pass"] = bool(len(missed) == 0 and z_clean <= 1.0 and z_truth < 3.5)
    log(f"(a) {rec['planted_outliers']} gross outliers, {rec['swapped_detections']} swapped detections, "
        f"{len(miss)} missing: left active {len(missed)}; {len(clean_rejected)} clean observations went with "
        f"{dropped_frames} frames left with an end in fewer than two cameras; rms {sol.rms:.4f} px (clean "
        f"{clean.rms:.4f}); vs the clean fit {z_clean:.2f} sigma; vs the truth {z_truth:.2f} sigma (clean fit "
        f"{z_truth_clean:.2f}) | {'PASS' if rec['pass'] else 'FAIL'}")
    return rec


def close_double(cpp, cam, X):
    """For points X: (has two images on the sensor within GUARD_PX of each other, the primary image)."""
    out, prim = [], []
    for im in cm.cpp_images(cpp, X):
        on = im[(im[:, 0] >= 0) & (im[:, 0] < cam.width) & (im[:, 1] >= 0) & (im[:, 1] < cam.height)]
        close = False
        if len(on) >= 2:
            d = np.linalg.norm(on[:, None] - on[None], axis=2) + np.eye(len(on)) * 1e9
            close = bool(d.min() < GUARD_PX)
        out.append(close)
    return np.array(out)


def part_b():
    tube = rigs.CASE_TUBE
    truth = rigs.reference_rig(3, 45.0, sensor=(10000, 2400))
    sizes = [(c.width, c.height) for c in truth]
    cpp = [cm.to_cpp(c, tube) for c in truth]
    rng = np.random.default_rng(809)
    base = rigs.wand_frames(truth, tube, 400, 10.0, rng, half=(20, 20, 20), noise_px=0.1)
    rc = tube.fold_radius()
    # candidate points just beyond the fold radius, where close double images concentrate (t8 log: separations
    # are hundreds of px elsewhere); those with a close double image in some camera
    m = 40000
    r = np.sqrt(rng.uniform(rc ** 2, (rc + 1.0) ** 2, m))
    th = rng.uniform(0, 2 * np.pi, m)
    Pn = np.column_stack([r * np.cos(th), rng.uniform(-15, 15, m), r * np.sin(th)])
    cd = np.zeros(m, bool)
    for cam, cc in zip(truth, cpp):
        cd |= close_double(cc, cam, Pn)
    log(f"(b) near-wall candidates: {cd.sum()} of {m} points have two images within {GUARD_PX:.0f} px in some camera")

    def frame_from(A):
        """A frame with end A, the other end 10 mm away inside the tube; (uv, A, B) or None."""
        for _ in range(50):
            w = rng.normal(size=3)
            w /= np.linalg.norm(w)
            B = A + 10.0 * w
            if np.hypot(B[0], B[2]) > 0.985 * tube.r_inner:
                continue
            uv = np.full((len(truth), 2, 2), np.nan)
            for c, (cam, cc) in enumerate(zip(truth, cpp)):
                q = cm.cpp_project(cc, np.vstack([A, B]))
                on = np.isfinite(q[:, 0]) & (q[:, 0] >= 0) & (q[:, 0] < cam.width) & (q[:, 1] >= 0) & (
                    q[:, 1] < cam.height)
                if on.all():
                    uv[c] = q
            if np.sum(np.all(np.isfinite(uv[:, :, 0]), axis=1)) >= 2:
                return uv, A, B
        return None
    planted, other = [], []
    for i in rng.permutation(np.flatnonzero(cd)):
        if len(planted) >= 30:
            break
        fr_ = frame_from(Pn[i])
        if fr_ is not None:
            planted.append(fr_)
    # other near-wall frames: anywhere beyond the fold radius
    r2 = np.sqrt(rng.uniform(rc ** 2, (0.985 * tube.r_inner) ** 2, 2000))
    th2 = rng.uniform(0, 2 * np.pi, 2000)
    P2 = np.column_stack([r2 * np.cos(th2), rng.uniform(-15, 15, 2000), r2 * np.sin(th2)])
    for i in range(2000):
        if len(other) >= 30:
            break
        fr_ = frame_from(P2[i])
        if fr_ is not None:
            other.append(fr_)
    extra = np.array([f[0] for f in planted + other])
    ends_true = [(f[1], f[2]) for f in planted + other]
    uv_all = np.concatenate([base.uv, extra + rng.normal(scale=0.1, size=extra.shape)])
    t0 = time.time()
    sol = cw.calibrate(cw.WandData(uv_all, 10.0, sizes), tube, guard_px=GUARD_PX)
    obs = sol.obs
    # every observation of the extra frames whose (true) point has another image within GUARD_PX of the detection
    must_drop = []
    for i in range(len(obs)):
        fid = int(obs.frame_ids[obs.frame[i]])
        if fid < 400:
            continue
        X = ends_true[fid - 400][int(obs.end[i])]
        cam = truth[obs.cam[i]]
        ims = cm.cpp_images(cpp[obs.cam[i]], X[None])[0]
        on = ims[(ims[:, 0] >= 0) & (ims[:, 0] < cam.width) & (ims[:, 1] >= 0) & (ims[:, 1] < cam.height)]
        if len(on) >= 2 and np.sort(np.linalg.norm(on - obs.uv[i], axis=1))[1] < GUARD_PX:
            must_drop.append(i)
    left = [i for i in must_drop if obs.active[i]]
    # the guard sees the fitted points; next to the fold the second image moves fast with the point, so report the
    # distances of any left active at the truth and at the fit
    XAf, XBf = sol.frames.ends(sol.L)
    for i in left:
        fid = int(obs.frame_ids[obs.frame[i]])
        for lab, X in (("truth", ends_true[fid - 400][int(obs.end[i])]),
                       ("fit", (XAf if obs.end[i] == 0 else XBf)[obs.frame[i]])):
            ims = cm.cpp_images(cpp[obs.cam[i]], X[None])[0]
            dd = np.sort(np.linalg.norm(ims - obs.uv[i], axis=1))
            log(f"    left active: observation {i} (frame {fid}, camera {obs.cam[i] + 1}): images at {lab} point, "
                f"distances from the detection {', '.join(f'{v:.1f}' for v in dd)} px")
    z_truth = zmax(sol, truth, sol.cov)
    rec = {"planted_frames": len(planted), "other_near_wall_frames": len(other),
           "observations_with_second_image_within_guard": len(must_drop), "left_active": len(left),
           "guard_dropped": sol.info.get("guard_dropped"), "multi_image_observations": sol.info.get(
               "multi_image_observations"), "max_error_vs_truth_sigma": z_truth, "rms_px": sol.rms,
           "seconds": time.time() - t0}
    rec["pass"] = bool(len(must_drop) > 0 and len(left) == 0 and z_truth < 3.5)
    log(f"(b) {len(planted)} frames with a close double image + {len(other)} other near-wall frames + 400 ordinary: "
        f"{len(must_drop)} observations have a second image within {GUARD_PX:.0f} px, left active {len(left)} (guard "
        f"dropped {sol.info.get('guard_dropped')}, multi-image observations {sol.info.get('multi_image_observations')});"
        f" rms {sol.rms:.4f} px; vs the truth {z_truth:.2f} sigma | {'PASS' if rec['pass'] else 'FAIL'}")
    return rec


def main():
    res = {"corrupted": part_a(), "near_wall": part_b()}
    res["pass"] = bool(res["corrupted"]["pass"] and res["near_wall"]["pass"])
    log(f"  {'PASS' if res['pass'] else 'FAIL'}  robustness")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
