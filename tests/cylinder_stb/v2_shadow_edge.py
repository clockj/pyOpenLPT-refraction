"""V2 (plan, Phase 2): a shake next to one camera's shadow.

    python v2_shadow_edge.py [--builds p3 p123]

Three wide-view PINCYL cameras of V1's kind (400 mm from the axis, f = 1,000 px) at azimuths 0, +70 and -70 deg.  A
particle X0 at r > r_c that camera A (0 deg) sees once (status One), close to A's shadow: moving it by e_s mm along a
direction e puts it into the shadow (no image in A), while B and C see it all along.  Images: a round Gaussian spot
(sigma 1.2 px, peak 120) at X0's image in each camera.  Along X0 + s e, s from -3 e_s to +3 e_s, the shake's cost
TracerShakeStrategy.calShakeResidue with all three cameras chosen (as at the start of a shake at X0), with each
build in its own process:
  - p3 (today's Shake): a trial position with no image in A drops A from the cost, which then averages B and C only;
  - p123 (Phase 2): A stays in, with the cost of predicting nothing (its measured spot unexplained).
The images' own OTF (a Gaussian of sigma 1.2 px, peak 120) is given as tracer.otf_file.
Then shakeOneObject from X0 + 0.5 e_s (A still sees it), shake width 0.6 e_s, five times with the width halved.
Judged (plan): with p123 the cost is never lower in the shadow than at the last position A sees, so a shake is not
drawn into the shadow, and the shakes end at X0 (within the last width); p3's are reported beside it.
Writes v2_shadow_edge_out.txt and v2_shadow_edge.json.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

import cstb
from v1_statuses import wide_camera

PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"
WORK = os.path.join(cstb.DATA, "v2")


def spot_image(shape, uv, sigma=1.2, amp=120.0):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W))
    if uv is not None:
        img += amp * np.exp(-((xx - uv[0]) ** 2 + (yy - uv[1]) ** 2) / (2 * sigma ** 2))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def write_gauss_otf(path, n_cam, vol, sigma=1.2, amp=120.0):
    """The images' own OTF (a uniform Gaussian), so that the cost is about 0 at X0: with one particle per image the
    estimate from the images falls back to its defaults."""
    n = 125
    lines = ["# Size: (n_cam,nx,ny,nz,n_grid)", f"{n_cam},5,5,5,{n}", "# Boundary: (xmin,xmax,ymin,ymax,zmin,zmax)",
             ",".join(f"{v:.6f}" for v in (vol["x"][0], vol["x"][1], vol["y"][0], vol["y"][1], vol["z"][0],
                                           vol["z"][1]))]
    for name, val in (("a", amp), ("b", 0.5 / sigma ** 2), ("c", 0.5 / sigma ** 2), ("alpha", 0.0)):
        lines.append(f"# {name}: (n_cam,n_grid)")
        lines += [",".join([f"{val:.9f}"] * n) for _ in range(n_cam)]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def setup(lpt):
    """Cameras, X0, e, e_s; images and config written to WORK."""
    if os.path.isdir(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    cams = [wide_camera(lpt, a) for a in (0.0, 70.0, -70.0)]
    files = []
    for k, c in enumerate(cams):
        f = os.path.join(WORK, f"cam{k + 1}.txt")
        c.saveParameters(f)
        files.append(f)
    rng = np.random.default_rng(41)
    # a point next to A's shadow: A sees it, B and C see it, and a small move along +/-e loses A's image
    for _ in range(200000):
        r = rng.uniform(80.0, 96.0)
        phi = rng.uniform(0, 2 * np.pi)
        X = np.array([r * math.cos(phi), rng.uniform(-10, 10), r * math.sin(phi)])
        P = cstb.pt3(lpt, X)
        if any(c.imageStatus(P, 2.0, 8.0)[0] != "one" for c in cams):
            continue
        # tangential and radial directions in the transverse plane
        for e in (np.array([-math.sin(phi), 0, math.cos(phi)]), np.array([math.cos(phi), 0, math.sin(phi)])):
            for sgn in (1.0, -1.0):
                ee = sgn * e
                # the first s (0.005 mm steps) where A loses the image
                s_edge = None
                for k in range(1, 61):
                    s = 0.005 * k
                    st = cams[0].imageStatus(cstb.pt3(lpt, X + s * ee), 2.0, 8.0)[0]
                    if st != "one":
                        if st == "none":
                            s_edge = s
                        break
                if s_edge is None or s_edge < 0.05:
                    continue
                ok_bc = all(c.imageStatus(cstb.pt3(lpt, X + s * ee), 2.0, 8.0)[0] == "one"
                            for c in cams[1:] for s in np.linspace(-3 * s_edge, 3 * s_edge, 13))
                if ok_bc:
                    return cams, files, X, ee, s_edge
    raise SystemExit("no point next to the shadow found")


def worker(build, out_path):
    lpt, rec = cstb.use_build(build)
    meta = json.load(open(os.path.join(WORK, "setup.json"), encoding="utf-8"))
    cfg_path = os.path.join(WORK, "config.json")
    bs = lpt.BasicSetting()
    bs.readConfig(cfg_path)
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    imgs = [lpt.Image.from_numpy(cv2.imread(os.path.join(WORK, f"cam{k + 1}_0.tif"), cv2.IMREAD_UNCHANGED))
            for k in range(3)]
    X0, e, s_edge = np.array(meta["X0"]), np.array(meta["e"]), meta["s_edge"]
    shake = lpt.Shake(cams, tc)
    o0 = lpt.Tracer3D(cstb.pt3(lpt, X0))
    o0.projectObject2D(cams)
    shake.calResidueImage([o0], imgs)                # the residual: the images less the particle, as runShake
    roi = shake.buildROIInfo(o0, imgs)
    strat = shake.strategy
    profile = []
    for s in np.linspace(-3 * s_edge, 3 * s_edge, 61):
        o = lpt.Tracer3D(cstb.pt3(lpt, X0 + s * e))
        o.projectObject2D(cams)
        seen_a = o[0]._pt_center[0] > -1e9
        profile.append((float(s), float(strat.calShakeResidue(o, roi, [True, True, True])), bool(seen_a)))
    # shakes from X0 + 0.5 s_edge e
    start = X0 + 0.5 * s_edge * e
    o = lpt.Tracer3D(cstb.pt3(lpt, start))
    o.projectObject2D(cams)
    delta = 0.6 * s_edge
    path = [start.tolist()]
    for _ in range(5):
        shake.calResidueImage([o], imgs)
        roi = shake.buildROIInfo(o, imgs)
        _res, o = shake.shakeOneObject(o, roi, delta, [True, True, True])
        path.append([o._pt_center[i] for i in range(3)])
        delta *= 0.5
    end = np.array(path[-1])
    json.dump({"build": rec, "profile": profile, "path": path,
               "end_from_X0_mm": float(np.linalg.norm(end - X0)),
               "end_s_mm": float((end - X0) @ e)}, open(out_path, "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=["p3", "p123"])
    ap.add_argument("--worker", nargs=2, default=None)
    args = ap.parse_args()
    if args.worker:
        worker(*args.worker)
        return
    lpt, build = cstb.use_build(args.builds[-1])
    log = cstb.Log(os.path.join(cstb.HERE, "v2_shadow_edge"))
    cams, files, X0, e, s_edge = setup(lpt)
    H, W = cams[0].getNRow(), cams[0].getNCol()
    lists = []
    for k, c in enumerate(cams):
        p = c.project(cstb.pt3(lpt, X0))
        img = spot_image((H, W), (p[0], p[1]))
        names = []
        for t in range(5):
            f = os.path.join(WORK, f"cam{k + 1}_{t}.tif")
            cv2.imwrite(f, img)
            names.append(f)
        lp = os.path.join(WORK, f"cam{k + 1}ImageNames.txt")
        open(lp, "w", encoding="utf-8").write("\n".join(names) + "\n")
        lists.append(lp)
    vol = {"x": [-100.0, 100.0], "y": [-30.0, 30.0], "z": [-100.0, 100.0]}
    cfg = {"frames": [0, 4], "fps": 1000, "threads": 0,
           "cameras": [{"file": f, "max_intensity": 255} for f in files], "image_lists": lists,
           "view_volume_mm": vol, "voxel_mm": 0.1, "output_dir": "Results",
           "objects": [{"type": "Tracer", "initial_phase": {"search_radius_vox": 10.0, "frames": 4},
                        "convergence_phase": {"avg_spacing_vox": 30.0},
                        "shake": {"width_vox": 0.25, "loops": 4, "ghost_threshold": 0.01},
                        "predictive_field": {"grid": [51, 51, 51], "search_radius_vox": 25.0},
                        "ipr": {"loops": 2, "reduced_cameras": 1, "reduced_loops": 1, "tol_2d_px": 1.0,
                                "tol_3d_vox": 1.0},
                        "tracer": {"intensity_threshold": 20, "radius_px": 2.0,
                                   "otf_file": os.path.join(WORK, "otf.txt")}}]}
    json.dump(cfg, open(os.path.join(WORK, "config.json"), "w", encoding="utf-8"), indent=1)
    write_gauss_otf(os.path.join(WORK, "otf.txt"), 3, vol)
    json.dump({"X0": X0.tolist(), "e": e.tolist(), "s_edge": s_edge}, open(os.path.join(WORK, "setup.json"), "w"),
              indent=1)
    log(f"V2: a shake next to camera A's shadow; X0 = {np.round(X0, 3).tolist()} mm (r = "
        f"{math.hypot(X0[0], X0[2]):.2f} mm), A loses its image {s_edge:.3f} mm along e = {np.round(e, 3).tolist()}")
    out = {"X0": X0.tolist(), "e": e.tolist(), "s_edge": s_edge, "builds": {}}
    for b in args.builds:
        p = os.path.join(WORK, f"v2_{b}.json")
        r = subprocess.run([PY, __file__, "--worker", b, p], capture_output=True, text=True, cwd=cstb.HERE)
        if r.returncode != 0:
            log(r.stdout[-3000:] + r.stderr[-3000:])
            raise SystemExit(f"worker {b} failed")
        res = json.load(open(p, encoding="utf-8"))
        out["builds"][b] = res
        prof = res["profile"]
        last_seen = max((c for s, c, a in prof if a and s > 0), default=None, key=lambda c: c)
        seen_costs = [c for s, c, a in prof if a]
        shadow_costs = [c for s, c, a in prof if not a]
        edge_seen = [c for s, c, a in prof if a][-1]
        res["min_cost_seen"] = min(seen_costs)
        res["min_cost_shadow"] = min(shadow_costs) if shadow_costs else None
        res["cost_last_seen"] = edge_seen
        log(f"  {b}: cost where A sees it: min {min(seen_costs):.2f}, at the last such position {edge_seen:.2f}; "
            f"in the shadow: min {min(shadow_costs):.2f}, max {max(shadow_costs):.2f}; shake from +0.5 e_s ends "
            f"{res['end_from_X0_mm'] * 1e3:.1f} um from X0 (s = {res['end_s_mm'] * 1e3:+.1f} um)")
    new = out["builds"][args.builds[-1]]
    out["pass"] = bool(new["min_cost_shadow"] is not None and new["min_cost_shadow"] >= new["cost_last_seen"]
                       and new["end_from_X0_mm"] <= 0.6 * s_edge / 16 + 1e-3)
    log(f"  judged (p123: no lower cost in the shadow; the shake ends at X0): {'PASS' if out['pass'] else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
