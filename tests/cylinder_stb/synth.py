"""Synthetic tracer recordings through the tube, with their truth (plan: Phases 1 F5, 2 and 5).

    python synth.py NAME --cams DIR [--particles 400] [--frames 30] [--render spot] [--step-px 1.5] [--seed 1]

Writes data/NAME/ (git-ignored):
  - camFile/cam1.txt, cam2.txt: the cameras (copies of DIR's);
  - img/camK/img%06d.tif: 8-bit images, bright particles on a dark background (as OpenLPT's tracer finder expects,
    i.e. after the shadowgraph preprocessing);
  - imgLists/camKImageNames.txt (absolute paths), config.json (frames, cameras, view volume, a tracer config with
    radius_px = 2 and threshold 20 unless set), truth.csv (frame, id, X, Y, Z, then u, v per camera of the primary
    image, empty where the particle has no image) and synth.json (the parameters).
Particles: uniform in the volume both cameras see (cstb.sample_shared, the box BOX), moving with a solid-body swirl
about the tube's axis plus an axial drift, scaled so that the median image displacement per frame is STEP_PX.
Render "spot": a round Gaussian of sigma SIGMA_PX and peak AMP grey levels at each image (no noise unless --noise);
other renderers are added by Phase 5.
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

BOX = ((-20.0, -15.0, -30.0), (25.0, 15.0, 35.0))


def swirl_positions(X0, n_frames, omega, vy, axis_point=(0.0, 0.0, 0.0)):
    """Positions (n_frames, n, 3): rotation by omega*t (rad) about the world y axis through axis_point, plus vy*t
    along it."""
    out = np.empty((n_frames,) + X0.shape)
    c = np.asarray(axis_point, float)
    for t in range(n_frames):
        a = omega * t
        ca, sa = math.cos(a), math.sin(a)
        d = X0 - c
        out[t, :, 0] = c[0] + ca * d[:, 0] + sa * d[:, 2]
        out[t, :, 2] = c[2] - sa * d[:, 0] + ca * d[:, 2]
        out[t, :, 1] = X0[:, 1] + vy * t
    return out


def render_spots(shape, uv, amp, sigma, noise, rng):
    H, W = shape
    img = np.zeros((H, W), np.float64)
    r = int(math.ceil(4 * sigma))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for (u, v), a in zip(uv, amp):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        cu, cv_ = int(round(u)), int(round(v))
        if cu < -r or cu >= W + r or cv_ < -r or cv_ >= H + r:
            continue
        g = a * np.exp(-((xx + cu - u) ** 2 + (yy + cv_ - v) ** 2) / (2 * sigma * sigma))
        y0, y1, x0, x1 = cv_ - r, cv_ + r + 1, cu - r, cu + r + 1
        gy0, gx0 = max(0, -y0), max(0, -x0)
        y0c, y1c, x0c, x1c = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
        img[y0c:y1c, x0c:x1c] += g[gy0:gy0 + (y1c - y0c), gx0:gx0 + (x1c - x0c)]
    if noise > 0:
        img += rng.normal(0.0, noise, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def project_all(lpt, cams, X):
    """(n_cam, n, 2) primary pixels by hinted projections along each particle's track; NaN where none."""
    n_f, n, _ = X.shape
    out = np.full((n_f, len(cams), n, 2), np.nan)
    for c, cam in enumerate(cams):
        for i in range(n):
            h = lpt.ProjectionHint()
            for t in range(n_f):
                ok, p, h, _e = cam.projectWithHint(cstb.pt3(lpt, X[t, i]), h)
                if ok:
                    out[t, c, i] = (p[0], p[1])
    return out


def write_config(case_dir, n_frames, n_cam, vol, voxel_mm, tracer, ipr=None, shake=None, extra=None):
    cfg = {
        "_comment": "Synthetic case written by tests/cylinder_stb/synth.py.",
        "frames": [0, n_frames - 1], "fps": 1000, "threads": 0,
        "cameras": [{"file": os.path.join(case_dir, "camFile", f"cam{c + 1}.txt"), "max_intensity": 255}
                    for c in range(n_cam)],
        "image_lists": [os.path.join(case_dir, "imgLists", f"cam{c + 1}ImageNames.txt") for c in range(n_cam)],
        "view_volume_mm": {"x": [vol[0][0], vol[1][0]], "y": [vol[0][1], vol[1][1]], "z": [vol[0][2], vol[1][2]]},
        "voxel_mm": voxel_mm,
        "output_dir": "Results",
        "objects": [{
            "type": "Tracer",
            "initial_phase": {"search_radius_vox": 10.0, "frames": 4},
            "convergence_phase": {"avg_spacing_vox": 30.0},
            "shake": shake or {"width_vox": 0.25, "loops": 4, "ghost_threshold": 0.01},
            "predictive_field": {"grid": [51, 51, 51], "search_radius_vox": 25.0},
            "ipr": ipr or {"loops": 2, "reduced_cameras": 0, "reduced_loops": 0, "tol_2d_px": 1.0, "tol_3d_vox": 1.0},
            "tracer": tracer,
        }],
    }
    if extra:
        cfg.update(extra)
    with open(os.path.join(case_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--cams", default=cstb.CAMS_T2)
    ap.add_argument("--tol", type=float, default=1e-9, help="the cameras' projection tolerance (C4: 1e-9)")
    ap.add_argument("--build", default="p1")
    ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--render", default="spot")
    ap.add_argument("--sigma-px", type=float, default=1.2)
    ap.add_argument("--amp", type=float, default=180.0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--step-px", type=float, default=1.5)
    ap.add_argument("--voxel-mm", type=float, default=0.026)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    rng = np.random.default_rng(args.seed)
    case_dir = os.path.join(cstb.DATA, args.name)
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(os.path.join(case_dir, "camFile"))
    src = cstb.cams_with_tol(args.cams, args.tol)
    for c, f in enumerate(src):
        shutil.copy2(f, os.path.join(case_dir, "camFile", f"cam{c + 1}.txt"))
    files = [os.path.join(case_dir, "camFile", f"cam{c + 1}.txt") for c in range(len(src))]
    cams = [lpt.loadCameraModel(f) for f in files]
    H, W = cams[0].getNRow(), cams[0].getNCol()

    X0, _pix = cstb.sample_shared(lpt, cams, args.particles, BOX, rng)
    # scale the swirl so the median image step is step_px: one trial frame
    trial = swirl_positions(X0, 2, 1e-3, 0.0)
    P = project_all(lpt, cams, trial)
    d = np.hypot(*(P[1] - P[0]).transpose(2, 0, 1))
    omega = 1e-3 * args.step_px / float(np.nanmedian(d))
    vy = 0.3 * omega * float(np.median(np.hypot(X0[:, 0], X0[:, 2])))
    X = swirl_positions(X0, args.frames, omega, vy)
    P = project_all(lpt, cams, X)
    steps = np.hypot(*(P[1:] - P[:-1]).transpose(3, 0, 1, 2))
    lo = X.reshape(-1, 3).min(axis=0) - 0.5
    hi = X.reshape(-1, 3).max(axis=0) + 0.5
    vol = (lo.round(2).tolist(), hi.round(2).tolist())

    amp = args.amp * (0.8 + 0.4 * rng.random(args.particles))
    for c in range(len(cams)):
        d_img = os.path.join(case_dir, "img", f"cam{c + 1}")
        os.makedirs(d_img)
        names = []
        for t in range(args.frames):
            if args.render == "spot":
                img = render_spots((H, W), P[t, c], amp, args.sigma_px, args.noise, rng)
            else:
                raise SystemExit(f"unknown renderer {args.render}")
            path = os.path.join(d_img, f"img{t:06d}.tif")
            cv2.imwrite(path, img)
            names.append(path)
        os.makedirs(os.path.join(case_dir, "imgLists"), exist_ok=True)
        with open(os.path.join(case_dir, "imgLists", f"cam{c + 1}ImageNames.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
    with open(os.path.join(case_dir, "truth.csv"), "w", encoding="utf-8") as fh:
        fh.write("frame,id,X,Y,Z," + ",".join(f"u{c + 1},v{c + 1}" for c in range(len(cams))) + "\n")
        for t in range(args.frames):
            for i in range(args.particles):
                uv = ",".join("" if not np.isfinite(P[t, c, i, 0]) else f"{P[t, c, i, 0]:.6f},{P[t, c, i, 1]:.6f}"
                              for c in range(len(cams)))
                fh.write(f"{t},{i},{X[t, i, 0]:.9f},{X[t, i, 1]:.9f},{X[t, i, 2]:.9f},{uv}\n")
    tracer = {"intensity_threshold": 20, "radius_px": 2.0}
    write_config(case_dir, args.frames, len(cams), vol, args.voxel_mm, tracer)
    meta = {"args": vars(args), "build": build, "omega_rad_per_frame": omega, "vy_mm_per_frame": vy,
            "image_step_px_median": float(np.nanmedian(steps)), "image_step_px_p90": float(np.nanpercentile(steps, 90)),
            "view_volume_mm": vol, "on_sensor_fraction": float(np.mean(
                (P[..., 0] >= 0) & (P[..., 0] < W) & (P[..., 1] >= 0) & (P[..., 1] < H)))}
    with open(os.path.join(case_dir, "synth.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({k: meta[k] for k in meta if k not in ("args", "build")}, indent=1))


if __name__ == "__main__":
    main()
