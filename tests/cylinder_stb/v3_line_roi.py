"""V3 (plan, Phase 2): line images are removed from the residual when the regions come from the OTF.

    python v3_line_roi.py [--build p123] [--case f5_spots] [--n 150]

Cameras: the synthetic case's (6 Oct T2's PINCYL cameras at tolerance 1e-9).  N particles of the volume both cameras
see, chosen so that their images lie at least 100 px apart in both cameras (no overlap, so each particle's signal
can be measured alone).  For each line length L = 5, 10, 20, 30 px (horizontal, width sigma 1.2 px, peak 120 grey
levels; the segment model of OTF::intensity, sampled at the pixel centres, rounded to 8 bits), the residual image
Shake.calResidualImage leaves with the particles at their true positions, for:
  - "today": the config as it is (radius_px 2: square regions of 4 x 4 px; the Gaussian OTF estimated from the first
    five images);
  - "otf": tracer.otf_file = the segment model with this L (a 5x5x5 grid, every cell the same) and
    tracer.roi_from_otf = true (regions out to 3 sigma).
Per particle the share of its signal removed: 1 - (residual summed over a box 10 px beyond the line) / (the image
summed there).  Judged (plan): with the OTF's regions >= 98 % removed for every particle at every length; today's
square region is reported beside it.  Writes v3_line_roi_out.txt and v3_line_roi.json.
"""
import argparse
import json
import math
import os
import shutil

import cv2
import numpy as np
from scipy.special import erf

import cstb

AMP, SIGMA = 120.0, 1.2


def segment_image(shape, centres, L, sigma=SIGMA, amp=AMP):
    H, W = shape
    img = np.zeros((H, W))
    hx = int(math.ceil(L / 2 + 5 * sigma)) + 1
    hy = int(math.ceil(5 * sigma)) + 1
    k = 1.0 / (sigma * math.sqrt(2))
    for (u, v) in centres:
        cu, cv_ = int(round(u)), int(round(v))
        xs = np.arange(cu - hx, cu + hx + 1)
        ys = np.arange(cv_ - hy, cv_ + hy + 1)
        xx = xs[None, :] - u
        yy = ys[:, None] - v
        gx = (erf((xx + L / 2) * k) - erf((xx - L / 2) * k)) / (2 * erf(L / 2 * k)) if L > 1e-3 * sigma else \
            np.exp(-0.5 * xx ** 2 / sigma ** 2)
        gy = np.exp(-0.5 * yy ** 2 / sigma ** 2)
        y0, x0 = max(0, ys[0]), max(0, xs[0])
        y1, x1 = min(H, ys[-1] + 1), min(W, xs[-1] + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        img[y0:y1, x0:x1] += (amp * gy * gx)[y0 - ys[0]:y1 - ys[0], x0 - xs[0]:x1 - xs[0]]
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def write_segment_otf(path, n_cam, vol, L, sigma=SIGMA, amp=AMP):
    nx = ny = nz = 5
    n = nx * ny * nz
    lines = ["# Size: (n_cam,nx,ny,nz,n_grid)", f"{n_cam},{nx},{ny},{nz},{n}",
             "# Boundary: (xmin,xmax,ymin,ymax,zmin,zmax)",
             ",".join(f"{v:.6f}" for v in (vol["x"][0], vol["x"][1], vol["y"][0], vol["y"][1], vol["z"][0], vol["z"][1]))]
    for name, val in (("a", amp), ("b", L), ("c", 0.0), ("alpha", 0.0)):
        lines.append(f"# {name}: (n_cam,n_grid)")
        lines += [",".join([f"{val:.6f}"] * n) for _ in range(n_cam)]
    lines += ["# Model: (gauss or segment; segment: b, c are lengths [px])", "segment", "# sigma: (n_cam,n_grid)"]
    lines += [",".join([f"{sigma:.6f}"] * n) for _ in range(n_cam)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p123")
    ap.add_argument("--case", default="f5_spots")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "v3_line_roi"))
    rng = np.random.default_rng(args.seed)
    src = os.path.join(cstb.DATA, args.case)
    with open(os.path.join(src, "config.json"), encoding="utf-8") as fh:
        base_cfg = json.load(fh)
    work = os.path.join(cstb.DATA, "v3")
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    # cameras and well-separated particles
    cams_files = [c["file"] for c in base_cfg["cameras"]]
    cams = [lpt.loadCameraModel(f) for f in cams_files]
    H, W = cams[0].getNRow(), cams[0].getNCol()
    vol = base_cfg["view_volume_mm"]
    box = ((vol["x"][0], vol["y"][0], vol["z"][0]), (vol["x"][1], vol["y"][1], vol["z"][1]))
    pts, pix = cstb.sample_shared(lpt, cams, 3000, box, rng)
    keep = []
    for k in range(len(pts)):
        u = pix[k]
        if np.any(u[:, 0] < 40) or np.any(u[:, 0] > W - 40) or np.any(u[:, 1] < 20) or np.any(u[:, 1] > H - 20):
            continue
        if all(np.all(np.hypot(*(pix[j] - u).T) > 100) for j in keep):
            keep.append(k)
        if len(keep) == args.n:
            break
    pts, pix = pts[keep], pix[keep]
    log(f"V3: line images and the residual; build {build['name']} ({build['git_head']} + diff "
        f"{(build['git_diff_sha256'] or '')[:12]}); {len(pts)} particles >= 100 px apart in both cameras")
    out = {"build": build, "particles": len(pts), "lengths": {}}
    all_ok = True
    for L in (5.0, 10.0, 20.0, 30.0):
        d = os.path.join(work, f"L{int(L)}")
        os.makedirs(d)
        imgs_np, lists = [], []
        for c in range(len(cams)):
            img = segment_image((H, W), pix[:, c], L)
            imgs_np.append(img)
            names = []
            for t in range(5):   # the OTF estimate reads the first five frames
                p = os.path.join(d, f"cam{c + 1}_{t}.tif")
                cv2.imwrite(p, img)
                names.append(p)
            lp = os.path.join(d, f"cam{c + 1}ImageNames.txt")
            with open(lp, "w", encoding="utf-8") as fh:
                fh.write("\n".join(names) + "\n")
            lists.append(lp)
        rec = {}
        for mode in ("today", "otf"):
            cfg = json.loads(json.dumps(base_cfg))
            cfg["frames"] = [0, 4]
            cfg["image_lists"] = lists
            tr = cfg["objects"][0]["tracer"]
            if mode == "otf":
                otf_path = os.path.join(d, "otf_segment.txt")
                write_segment_otf(otf_path, len(cams), vol, L)
                tr["otf_file"] = otf_path
                tr["roi_from_otf"] = True
            cfg_path = os.path.join(d, f"config_{mode}.json")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=1)
            bs = lpt.BasicSetting()
            bs.readConfig(cfg_path)
            tc = lpt.TracerConfig()
            tc.readConfig(bs._object_config_paths[0], bs)
            shake = lpt.Shake(cams, tc)
            objs = []
            for X in pts:
                o = lpt.Tracer3D(cstb.pt3(lpt, X))
                o.projectObject2D(cams)
                objs.append(o)
            imgs = [lpt.Image.from_numpy(im) for im in imgs_np]
            res = shake.calResidualImage(objs, imgs)
            removed = []
            for c in range(len(cams)):
                r_np = np.array(res[c].to_list())
                orig = imgs_np[c].astype(float)
                hx, hy = int(L / 2 + 10), 10
                for (u, v) in pix[:, c]:
                    cu, cv_ = int(round(u)), int(round(v))
                    win = (slice(max(0, cv_ - hy), cv_ + hy + 1), slice(max(0, cu - hx), cu + hx + 1))
                    s0 = orig[win].sum()
                    removed.append(1.0 - r_np[win].sum() / s0 if s0 > 0 else float("nan"))
            removed = np.array(removed)
            rec[mode] = {"removed_min": float(np.nanmin(removed)), "removed_median": float(np.nanmedian(removed)),
                         "removed_p5": float(np.nanpercentile(removed, 5))}
            if mode == "otf":
                rec[mode]["pass"] = bool(np.nanmin(removed) >= 0.98)
                all_ok &= rec[mode]["pass"]
        out["lengths"][str(L)] = rec
        log(f"  L = {L:.0f} px: removed (min / 5th percentile / median) today {rec['today']['removed_min']:.3f} / "
            f"{rec['today']['removed_p5']:.3f} / {rec['today']['removed_median']:.3f}; from the OTF "
            f"{rec['otf']['removed_min']:.3f} / {rec['otf']['removed_p5']:.3f} / {rec['otf']['removed_median']:.3f}"
            f" ({'PASS' if rec['otf']['pass'] else 'FAIL'})")
    out["pass"] = bool(all_ok)
    log(f"  judged (>= 98 % removed for every particle with the OTF's regions): {'PASS' if all_ok else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
