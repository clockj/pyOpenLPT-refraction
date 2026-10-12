"""V1 (plan, Phase 2): the per-camera image statuses on a rig that sees the whole tube.

    python v1_statuses.py [--build p123] [--n 50000]

A synthetic wide-view PINCYL camera (the tube of the real rig: radii 100.81 / 97.63 mm, n = 1, 1.49, 1.333; pinhole
400 mm from the axis, f = 1,000 px, 1280 x 800 px), so that the shadow and the double-image bands next to the wall
(r > r_c = 75.6 mm) are in view.  N random points of the water (r < 97.6 mm, |y| < 60 mm along the axis).  For each:
  - Camera.imageStatus(point, margin 2 px, merge 8 px) (C++);
  - the same rule written independently here from Camera.projectAll(): None when no image lies on the sensor
    enlarged by the margin; One when exactly one does, no other image lies within the merge distance of it, and it is
    the image project() returns; Ambiguous otherwise;
  - Camera.provablySingleImage(point).
Judged (plan): the statuses agree for every point; every point with provablySingleImage has exactly one image in
projectAll (so the census is never needed inside r_c); reported: the statuses by region (r below / above r_c).
Writes v1_statuses_out.txt and v1_statuses.json.
"""
import argparse
import math
import os

import numpy as np

import cstb

N_IDX = [1.0, 1.49, 1.333]
RADII = [100.80625, 97.63125]
R_C = 1.0 * RADII[0] / 1.333


def wide_camera(lpt, azimuth_deg=0.0, dist=400.0, f=1000.0):
    """A PINCYL camera on a circle of radius dist about the world y axis, looking at the axis."""
    cam = lpt.RefractionCylinderCameraModel()
    cam.setImageSize(800, 1280)
    cam.setIntrinsics(f, f, 639.5, 399.5, [0.0] * 5)
    a = math.radians(azimuth_deg)
    C = np.array([dist * math.sin(a), 0.0, dist * math.cos(a)])
    z_c = -C / np.linalg.norm(C)                 # forward: towards the axis
    y_c = np.array([0.0, -1.0, 0.0])             # image down = world -y
    x_c = np.cross(y_c, z_c)
    R = np.vstack([x_c, y_c, z_c])               # world -> camera
    t = -R @ C
    import cv2
    rvec, _ = cv2.Rodrigues(R)
    cam.setExtrinsics(lpt.Pt3D(*map(float, rvec.ravel())), lpt.Pt3D(*map(float, t)))
    cam.setRefraction(lpt.Pt3D(0.0, 0.0, 0.0), lpt.Pt3D(0.0, 1.0, 0.0), N_IDX, RADII)
    cam.setSolverOptions(1e-9, 80)
    cam.commitUpdate()
    return cam


def census_status(rays, primary, W, H, margin, merge):
    """The rule, independently: (status, the one image)."""
    if any(r["is_ring"] for r in rays):
        return "ambiguous", None
    on = [r for r in rays if -margin <= r["pt"][0] < W + margin and -margin <= r["pt"][1] < H + margin]
    if not on:
        return "none", None
    if len(on) > 1:
        return "ambiguous", None
    one = on[0]
    for r in rays:
        if r is one:
            continue
        if math.hypot(r["pt"][0] - one["pt"][0], r["pt"][1] - one["pt"][1]) <= merge:
            return "ambiguous", None
    if primary is None or math.hypot(primary[0] - one["pt"][0], primary[1] - one["pt"][1]) > 1e-9:
        return "ambiguous", None
    return "one", one["pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p123")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "v1_statuses"))
    rng = np.random.default_rng(args.seed)
    cam = wide_camera(lpt)
    W, H = cam.getNCol(), cam.getNRow()
    margin, merge = 2.0, 8.0
    log(f"V1: image statuses on a wide-view rig; build {build['name']} ({build['git_head']} + diff "
        f"{(build['git_diff_sha256'] or '')[:12]}); margin {margin} px, merge {merge} px; r_c = {R_C:.2f} mm")
    # points uniform in the water cylinder r < 97.6, |y| < 60
    r = RADII[1] * np.sqrt(rng.random(args.n)) * (1 - 1e-6)
    phi = 2 * np.pi * rng.random(args.n)
    y = rng.uniform(-60.0, 60.0, args.n)
    X = np.column_stack([r * np.cos(phi), y, r * np.sin(phi)])
    agree, disagree = 0, []
    counts = {}
    single_bad = 0
    single_n = 0
    inside_rc_not_single = 0
    n_images_hist = {}
    for k in range(args.n):
        P = cstb.pt3(lpt, X[k])
        st, _pt = cam.imageStatus(P, margin, merge)
        rays, _cov = cam.projectAll(P)
        try:
            pr = cam.project(P)
            primary = (pr[0], pr[1])
        except RuntimeError:
            primary = None
        rays = [{"pt": (d["pt"][0], d["pt"][1]) if not d["is_ring"] else (float("nan"),) * 2,
                 "is_ring": d["is_ring"]} for d in rays]
        ref, _ = census_status(rays, primary, W, H, margin, merge)
        region = "r < r_c" if r[k] < R_C else "r > r_c"
        counts.setdefault(region, {}).setdefault(st, 0)
        counts[region][st] += 1
        n_images_hist.setdefault(region, {}).setdefault(len(rays), 0)
        n_images_hist[region][len(rays)] += 1
        if st == ref:
            agree += 1
        elif len(disagree) < 20:
            disagree.append({"X": X[k].tolist(), "cpp": st, "rule": ref, "n_images": len(rays)})
        single = cam.provablySingleImage(P)
        if single:
            single_n += 1
            if len(rays) != 1:
                single_bad += 1
        elif r[k] < R_C * (1 - 1e-6):
            inside_rc_not_single += 1
    out = {"build": build, "n": args.n, "agree": agree, "disagree_examples": disagree, "counts": counts,
           "images_per_point": n_images_hist, "provably_single": single_n,
           "provably_single_but_not_one_image": single_bad, "inside_rc_without_certificate": inside_rc_not_single}
    out["pass"] = bool(agree == args.n and single_bad == 0)
    for region in sorted(counts):
        log(f"  {region}: statuses {counts[region]}; images per point {dict(sorted(n_images_hist[region].items()))}")
    log(f"  agreement with the rule from projectAll: {agree} of {args.n}")
    for d in disagree[:5]:
        log(f"    differs: {d}")
    log(f"  provablySingleImage: {single_n} points, of which {single_bad} have other than one image; points inside "
        f"r_c without the certificate: {inside_rc_not_single}")
    log(f"  judged: {'PASS' if out['pass'] else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
