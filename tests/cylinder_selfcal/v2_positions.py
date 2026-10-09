"""V2: what the correction does to positions, without detection noise.  (Added while running V2: with 4 cameras the
corrected run's x and z positions stayed at 14-16 um against the clean rig's 11-12.)

    python v2_positions.py [--frame 20]

For the central particles of the 4-camera set (data/cal4) inside the view volume at one frame, each seen by the rig's
cameras: least-squares triangulation (Gauss-Newton on the cameras' projections) of
  - the exact images (the clean positions): the cameras' own error, about 0;
  - the images moved by the error field (what the tracker sees);
  - the same moved back by the field fitted from the tracks with 2, 3 and 4 cameras (data/fields/cal4_r<n>.npz);
against the true positions: the RMS error per axis, and the RMS reprojection residual.  The part of the error field
that a smooth deformation of the world explains moves the triangulated points without leaving a residual; no
self-calibration can see it, so it stays in the positions after the correction.
Writes v2_positions_out.txt and .json here.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import selfcal as sc


def triangulate(cams, uv, X0, iters=6):
    """Gauss-Newton least squares of the projections onto pixels uv (n, k, 2) from X0 (n, 3)."""
    X = X0.copy()
    for _ in range(iters):
        P = np.zeros_like(uv)
        J = np.zeros(uv.shape[:2] + (2, 3))
        for c, f in enumerate(cams):
            P[:, c], J[:, c] = sc.project_jacobian(f, X)
        r = (uv - P).reshape(len(X), -1)
        Jm = J.reshape(len(X), -1, 3)
        dX = np.linalg.solve(np.einsum("nij,nik->njk", Jm, Jm), np.einsum("nij,ni->nj", Jm, r)[..., None])[..., 0]
        X = X + dX
    P = np.zeros_like(uv)
    for c, f in enumerate(cams):
        P[:, c] = sc.project_jacobian(f, X)[0]
    return X, float(np.sqrt(np.mean((uv - P) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=20)
    a = ap.parse_args()
    log = sc.Log("v2_positions")
    case_dir = os.path.join(sc.DATA, "cal4")
    arr = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"), mmap_mode="r")  # noqa: E731
    t = a.frame
    X = np.asarray(arr("X")[t])
    layer = np.asarray(arr("layer"))
    cams_all = [os.path.join(case_dir, "camFile", f"cam{c}.txt") for c in (1, 2, 3, 4)]
    out = {"frame": t}
    for n in (2, 3, 4):
        cams = cams_all[:n]
        uv0 = np.stack([np.asarray(arr(f"uv_clean_c{c}")[t]) for c in range(n)], axis=1)
        uv1 = np.stack([np.asarray(arr(f"uv_cal4_c{c}")[t]) for c in range(n)], axis=1)
        on = np.all(np.isfinite(uv0[..., 0]) & (uv0[..., 0] >= 0) & (uv0[..., 0] < 1280) & (uv0[..., 1] >= 0)
                    & (uv0[..., 1] < 800), axis=1)
        sel = (layer == 0) & sc.in_volume(X, sc.view_volume()) & on
        Xt, u0, u1 = X[sel], uv0[sel], uv1[sel]
        f = sc.Field.load(os.path.join(sc.DATA, "fields", f"cal4_r{n}.npz"))
        u2 = u1.copy()
        for c in range(n):
            u2[:, c] = u1[:, c] - f.eval_dense(c, u1[:, c])
        rec = {"particles": int(len(Xt))}
        for nm, u in (("exact images", u0), ("with the error field", u1), ("corrected by the fitted field", u2)):
            Xe, rr = triangulate(cams, u, Xt)
            e = 1e3 * np.sqrt(np.mean((Xe - Xt) ** 2, axis=0))
            rec[nm] = {"pos_rms_um": e.tolist(), "reproj_rms_px": rr}
            log(f"  {n} cameras ({len(Xt)} particles seen by all), {nm}: position RMS x/y/z {e[0]:.1f} / {e[1]:.1f} / "
                f"{e[2]:.1f} um; reprojection RMS {rr:.3f} px")
        out[f"{n}_cameras"] = rec
    log.save(out)


if __name__ == "__main__":
    main()
