"""Synthetic line-image recordings through the tube, with their truth (plan: Phase 5).

    python synth_lines.py NAME [--cams DIR] [--particles 3000] [--frames 40] [--step-px 1.5] [--seed 1]
                               [--A 15] [--sigma0 1.0] [--flux 2500] [--noise 7] [--zf-offset -8]
    python synth_lines.py NAME --otf-only     (rebuilds otf_truth.txt from shape.json and synth.json; images stay)

Writes data/NAME/ (git-ignored), as synth.py does (camFile/, img/camK/, imgLists/, config.json, truth.csv,
synth.json) plus shape.json (the shape model's parameters) and otf_truth.txt (the shape model's own OTF table, model
"segment", for the runs that are given the true model).
  - Cameras: DIR's camFiles at the tracking tolerance 1e-9 (6 Oct T2's by default).
  - Particles: uniform in the water the cameras look into (r < 30 mm about the axis, the depth both cameras span),
    kept if at least one camera sees them on its sensor, so that, as in T1, many images have no partner in the other
    camera.  PARTICLES of them, moving with synth.py's swirl, scaled so that the median image step is STEP_PX.
  - Images: the physical shape, not the tracker's model: per camera and particle the shape model's blur lengths
    (shape_model.py: A, z_f per camera, sigma0) give a uniform ellipse with full axes |L_h| x |L_v| (turned by the
    image angle of the direction across the axis), of integral FLUX x (0.6-1.4, per particle) grey levels px^2,
    blurred by a Gaussian of sigma0 (rendered at 4x4 subpixels), plus Gaussian noise NOISE, as 8-bit images of the
    preprocessed kind (particles bright on 0).  z_f per camera = the particles' median z_tan + ZF_OFFSET mm, so that
    vertical lines (tangential focus) and horizontal lines (sagittal focus) both occur.
truth.csv: frame, id, X, Y, Z, then per camera u, v, L_h, L_v (empty where the camera has no image).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil

import cv2
import numpy as np

import cstb
import shape_model as sm
from synth import swirl_positions, write_config

SUB = 4                                      # subpixels per pixel


def render_ellipse(img, u, v, Lh, Lv, alpha, flux, sigma0):
    """Adds a uniform ellipse (full axes |Lh| along the direction alpha, |Lv| across it) of integral flux, blurred by
    sigma0, centred at pixel (u, v) (pixel centres at integers)."""
    H, W = img.shape
    ax, ay = 0.5 * abs(Lh), 0.5 * abs(Lv)
    ext = int(math.ceil(max(ax, ay) + 4 * sigma0)) + 2
    cu, cv_ = int(round(u)), int(round(v))
    x0, y0 = cu - ext, cv_ - ext
    n = 2 * ext + 1
    if x0 + n <= 0 or y0 + n <= 0 or x0 >= W or y0 >= H:
        return
    # subpixel centres of the patch, relative to (u, v)
    s = (np.arange(n * SUB) + 0.5) / SUB - 0.5
    xs = x0 + s - u
    ys = y0 + s - v
    X, Y = np.meshgrid(xs, ys)
    ca, sa = math.cos(alpha), math.sin(alpha)
    xx = X * ca + Y * sa
    yy = -X * sa + Y * ca
    ax_ = max(ax, 0.5 / SUB)
    ay_ = max(ay, 0.5 / SUB)
    mask = ((xx / ax_) ** 2 + (yy / ay_) ** 2 <= 1.0).astype(np.float64)
    if mask.sum() == 0:
        mask[np.unravel_index(np.argmin(xx ** 2 + yy ** 2), mask.shape)] = 1.0
    mask *= flux / mask.sum()                   # integral = flux (per subpixel area handled by the block sum below)
    k = int(math.ceil(4 * sigma0 * SUB)) | 1
    blurred = cv2.GaussianBlur(mask, (2 * k + 1, 2 * k + 1), sigma0 * SUB, borderType=cv2.BORDER_CONSTANT)
    patch = blurred.reshape(n, SUB, n, SUB).sum(axis=(1, 3))
    ya, xa = max(0, y0), max(0, x0)
    yb, xb = min(H, y0 + n), min(W, x0 + n)
    img[ya:yb, xa:xb] += patch[ya - y0:yb - y0, xa - x0:xb - x0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--cams", default=cstb.CAMS_T2)
    ap.add_argument("--build", default="p123")
    ap.add_argument("--particles", type=int, default=3000)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--step-px", type=float, default=1.5)
    ap.add_argument("--A", type=float, default=15.0, help="entrance pupil diameter [mm]")
    ap.add_argument("--sigma0", type=float, default=1.0)
    ap.add_argument("--flux", type=float, default=2500.0)
    ap.add_argument("--noise", type=float, default=7.0)
    ap.add_argument("--zf-offset", type=float, default=-8.0)
    ap.add_argument("--voxel-mm", type=float, default=0.026)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--otf-only", action="store_true",
                    help="rebuild data/NAME/otf_truth.txt from its shape.json and synth.json (the images stay)")
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    rng = np.random.default_rng(args.seed)
    case_dir = os.path.join(cstb.DATA, args.name)
    if args.otf_only:
        with open(os.path.join(case_dir, "shape.json"), encoding="utf-8") as fh:
            params = json.load(fh)
        with open(os.path.join(case_dir, "synth.json"), encoding="utf-8") as fh:
            vol = json.load(fh)["view_volume_mm"]
        files = [os.path.join(case_dir, "camFile", f"cam{c + 1}.txt") for c in range(len(params))]
        sm.build_otf(lpt, files, params, vol, (9, 7, 11), os.path.join(case_dir, "otf_truth.txt"))
        print(f"rebuilt {os.path.join(case_dir, 'otf_truth.txt')} (build {build})")
        return
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(os.path.join(case_dir, "camFile"))
    src = cstb.cams_with_tol(args.cams, 1e-9)
    files = []
    for c, f in enumerate(src):
        dst = os.path.join(case_dir, "camFile", f"cam{c + 1}.txt")
        shutil.copy2(f, dst)
        files.append(dst)
    cams = [lpt.loadCameraModel(f) for f in files]
    n_cam = len(cams)
    H, W = cams[0].getNRow(), cams[0].getNCol()

    # particles: in the water near the axis (r < 30 mm), seen by at least one camera
    box = ((-30.0, -15.0, -30.0), (30.0, 15.0, 30.0))
    pts = []
    while len(pts) < args.particles:
        X = np.array(box[0]) + (np.array(box[1]) - np.array(box[0])) * rng.random((4000, 3))
        X = X[np.hypot(X[:, 0], X[:, 2]) < 30.0]
        P = [lpt.Pt3D(*map(float, x)) for x in X]
        res = [cam.projectBatchStatus(P) for cam in cams]
        for i in range(len(X)):
            seen = 0
            for r in res:
                if r[i][0] and 0 <= r[i][1][0] < W and 0 <= r[i][1][1] < H:
                    seen += 1
            if seen >= 1:
                pts.append(X[i])
                if len(pts) == args.particles:
                    break
    X0 = np.array(pts)
    # the swirl's scale: median image step
    from synth import project_all
    trial = swirl_positions(X0[:300], 2, 1e-3, 0.0)
    Pt = project_all(lpt, cams, trial)
    d = np.hypot(*(Pt[1] - Pt[0]).transpose(2, 0, 1))
    omega = 1e-3 * args.step_px / float(np.nanmedian(d))
    vy = 0.3 * omega * float(np.median(np.hypot(X0[:, 0], X0[:, 2])))
    Xt = swirl_positions(X0, args.frames, omega, vy)
    P = project_all(lpt, cams, Xt)                 # (frames, cam, n, 2)

    # the shape model per camera
    params, Lh, Lv, AL = [], np.full(P.shape[:3], np.nan), np.full(P.shape[:3], np.nan), np.zeros(P.shape[:3])
    for c, f in enumerate(files):
        fd = sm.FocalDepths(lpt, f)
        R, t, K = sm.camera_geometry(f)
        f_px = math.sqrt(K[0, 0] * K[1, 1])
        z = fd(Xt.reshape(-1, 3)).reshape(args.frames, args.particles, 2)
        zf = float(np.nanmedian(z[0, :, 0])) + args.zf_offset
        params.append({"A": args.A, "z_f": zf, "sigma0": args.sigma0, "F": args.flux, "f_px": f_px})
        lh, lv = sm.blur_lengths(z[..., 0], z[..., 1], f_px, args.A, zf)
        Lh[:, c, :], Lv[:, c, :] = lh, lv
        for i in range(args.particles):
            if np.all(np.isfinite(P[0, c, i])):
                AL[:, c, i] = sm.axis_angle(lpt, cams[c], Xt[0, i], (0.0, 1.0, 0.0))
    flux_i = args.flux * rng.uniform(0.6, 1.4, args.particles)

    for c in range(n_cam):
        d_img = os.path.join(case_dir, "img", f"cam{c + 1}")
        os.makedirs(d_img)
        names = []
        for t in range(args.frames):
            img = np.zeros((H, W))
            for i in range(args.particles):
                u, v = P[t, c, i]
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                render_ellipse(img, u, v, Lh[t, c, i], Lv[t, c, i], AL[t, c, i], flux_i[i], args.sigma0)
            img += rng.normal(0.0, args.noise, img.shape)
            path = os.path.join(d_img, f"img{t:06d}.tif")
            cv2.imwrite(path, np.clip(np.rint(img), 0, 255).astype(np.uint8))
            names.append(path)
        os.makedirs(os.path.join(case_dir, "imgLists"), exist_ok=True)
        with open(os.path.join(case_dir, "imgLists", f"cam{c + 1}ImageNames.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")

    with open(os.path.join(case_dir, "truth.csv"), "w", encoding="utf-8") as fh:
        fh.write("frame,id,X,Y,Z," + ",".join(f"u{c + 1},v{c + 1},Lh{c + 1},Lv{c + 1}" for c in range(n_cam)) + "\n")
        for t in range(args.frames):
            for i in range(args.particles):
                cols = []
                for c in range(n_cam):
                    u, v = P[t, c, i]
                    on = np.isfinite(u) and 0 <= u < W and 0 <= v < H
                    cols.append(f"{u:.6f},{v:.6f},{Lh[t, c, i]:.4f},{Lv[t, c, i]:.4f}" if on else ",,,")
                fh.write(f"{t},{i},{Xt[t, i, 0]:.9f},{Xt[t, i, 1]:.9f},{Xt[t, i, 2]:.9f}," + ",".join(cols) + "\n")
    # the view volume: where both cameras see particles
    both = np.all(np.isfinite(P[0, :, :, 0]), axis=0) & np.all(
        (P[0, :, :, 0] >= 0) & (P[0, :, :, 0] < W) & (P[0, :, :, 1] >= 0) & (P[0, :, :, 1] < H), axis=0)
    lo = Xt[:, both].reshape(-1, 3).min(axis=0) - 1.0
    hi = Xt[:, both].reshape(-1, 3).max(axis=0) + 1.0
    vol = (lo.round(2).tolist(), hi.round(2).tolist())
    tracer = {"intensity_threshold": 30, "radius_px": 2.0}
    write_config(case_dir, args.frames, n_cam, vol, args.voxel_mm, tracer)
    # the true shape model's OTF table over the view volume
    sm.build_otf(lpt, files, params, vol, (9, 7, 11), os.path.join(case_dir, "otf_truth.txt"))
    steps = np.hypot(*(P[1:] - P[:-1]).transpose(3, 0, 1, 2))
    seen = np.isfinite(P[..., 0]) & (P[..., 0] >= 0) & (P[..., 0] < W) & (P[..., 1] >= 0) & (P[..., 1] < H)
    meta = {"args": vars(args), "build": build, "params": params, "omega_rad_per_frame": omega,
            "vy_mm_per_frame": vy, "view_volume_mm": vol,
            "images_per_camera_per_frame": [float(seen[:, c].sum(axis=1).mean()) for c in range(n_cam)],
            "particles_in_both": int(both.sum()),
            "image_step_px_median": float(np.nanmedian(steps)),
            "Lh_px_pcts": [np.nanpercentile(np.abs(Lh[:, c]), [10, 50, 90]).tolist() for c in range(n_cam)],
            "Lv_px_pcts": [np.nanpercentile(np.abs(Lv[:, c]), [10, 50, 90]).tolist() for c in range(n_cam)]}
    with open(os.path.join(case_dir, "synth.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    with open(os.path.join(case_dir, "shape.json"), "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=1)
    print(json.dumps({k: meta[k] for k in meta if k not in ("args", "build")}, indent=1))


if __name__ == "__main__":
    main()
