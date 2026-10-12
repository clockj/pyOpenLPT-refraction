"""T0: self-tests of selfcal.py before any result rests on it.

    python t0_selftest.py

  1. B-splines: the basis sums to 1 everywhere; the bending energy of a plane is 0, of a quadratic its integral.
  2. The projected fit recovers a known field: synthetic 3-D points seen by T1's two cameras (and by the study's 4-camera
     rig), detections = projections + a known smooth field + noise; the fit must reproduce the field's observable part
     (2 cameras: across the epipolar curves; 4 cameras: all but a 3-D deformation) within the noise.
  3. The warp moves image content by -c: a Gaussian spot rendered at p appears at p - c after warping by a constant c,
     and at p - c(p) for a smooth field.
  4. One T1 frame (index 100 of the 15 px run): detection, one-to-one pairing of the long tracks' projections,
     Jacobians (the computed projections against the stored ones), the curved epipolar line and S2's candidate pairs
     (timings).
Writes t0_selftest_out.txt and .json here.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

import selfcal as sc


def check(log, name, ok, detail):
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return bool(ok)


def test_bsplines(log):
    out = {}
    x = np.linspace(-0.5, 1279.5, 5001)
    m, v = sc.bspline_basis(x + 0.5, 10, 1280)
    out["partition"] = float(np.max(np.abs(v.sum(axis=1) - 1)))
    ok1 = check(log, "B-spline partition of unity", out["partition"] < 1e-12, f"max |sum - 1| = {out['partition']:.1e}")
    f = sc.Field(1, 10, 6)
    Om = f.penalty()
    # a plane a + b u + c v: coefficients reproduce it exactly (cubic B-splines reproduce linear functions)
    gu = (np.arange(f.nbu) - 1.0) * (1280 / 10) - 0.5      # Greville abscissae of uniform cubic B-splines
    gv = (np.arange(f.nbv) - 1.0) * (800 / 6) - 0.5
    T = 0.3 + 0.002 * gu[:, None] - 0.001 * gv[None, :]
    th = T.ravel()
    e_plane = float(th @ Om @ th)
    uv = np.column_stack([np.random.default_rng(0).uniform(0, 1279, 200), np.random.default_rng(1).uniform(0, 799, 200)])
    f.theta[:f.nb] = th
    vals = f.eval_points(0, uv)[:, 0]
    exact = 0.3 + 0.002 * uv[:, 0] - 0.001 * uv[:, 1]
    out["plane_reproduction"] = float(np.max(np.abs(vals - exact)))
    out["plane_energy"] = e_plane
    ok2 = check(log, "a plane: reproduced and zero bending energy",
                out["plane_reproduction"] < 1e-9 and abs(e_plane) < 1e-9,
                f"max error {out['plane_reproduction']:.1e} px, energy {e_plane:.1e}")
    # a quadratic in u: f = a u^2 has f_uu = 2a, energy = 4 a^2 * area; least-squares spline of it
    a = 1e-4
    G = np.column_stack([np.ones(len(gu))])  # noqa: F841
    # fit the spline to the quadratic on a dense grid
    gx, gy = np.meshgrid(np.linspace(0, 1279, 120), np.linspace(0, 799, 80))
    P = np.column_stack([gx.ravel(), gy.ravel()])
    cols, valsb = f.basis_rows(P)
    A = np.zeros((len(P), f.nb))
    np.add.at(A, (np.repeat(np.arange(len(P)), 16), cols.ravel()), valsb.ravel())
    target = a * (P[:, 0] + 0.5) ** 2
    th2 = np.linalg.lstsq(A, target, rcond=None)[0]
    e_q = float(th2 @ Om @ th2)
    e_exact = 4 * a * a * 1280 * 800
    out["quadratic_energy_ratio"] = e_q / e_exact
    ok3 = check(log, "a quadratic's bending energy", abs(e_q / e_exact - 1) < 1e-6,
                f"spline {e_q:.6g} vs exact {e_exact:.6g} (ratio {e_q / e_exact:.8f})")
    return ok1 and ok2 and ok3, out


def synth_fit(log, camfiles, label, n_pts=6000, noise=0.15, seed=3):
    """Points in T1's view volume seen by all cameras; detections = projections + true field + noise."""
    rng = np.random.default_rng(seed)
    vol = sc.view_volume()
    lo = np.array([vol["x"][0], vol["y"][0], vol["z"][0]])
    hi = np.array([vol["x"][1], vol["y"][1], vol["z"][1]])
    X = lo + (hi - lo) * rng.random((4 * n_pts, 3))
    n_cam = len(camfiles)
    P = np.zeros((len(X), n_cam, 2))
    J = np.zeros((len(X), n_cam, 2, 3))
    for c, f in enumerate(camfiles):
        P[:, c], J[:, c] = sc.project_jacobian(f, X)
    ok = np.all(np.isfinite(P[..., 0]), axis=1) & np.all((P[..., 0] >= 0) & (P[..., 0] < 1280) & (P[..., 1] >= 0)
                                                         & (P[..., 1] < 800), axis=1)
    ok &= np.all(np.isfinite(J.reshape(len(X), -1)), axis=1)
    X, P, J = X[ok][:n_pts], P[ok][:n_pts], J[ok][:n_pts]
    # a true field per camera: smooth, about 0.6 px RMS, not in the spline's space exactly (sines)
    def true_field(c, uv):
        u, v = uv[:, 0] / 1280, uv[:, 1] / 800
        return np.column_stack([0.5 * np.sin(2.1 * u + 0.7 * c) + 0.3 * v * v - 0.2,
                                0.6 * np.cos(1.7 * v + 2.3 * u + 0.5 * c) - 0.25 * u])
    E = np.stack([true_field(c, P[:, c]) for c in range(n_cam)], axis=1)
    d = P + E + noise * rng.standard_normal(P.shape)
    # the tracker's point: least squares on the detections (so the residuals are what tracks show)
    Xh = X.copy()
    for _ in range(3):
        Pc = np.zeros_like(P)
        Jc = np.zeros_like(J)
        for c, f in enumerate(camfiles):
            Pc[:, c], Jc[:, c] = sc.project_jacobian(f, Xh)
        r = (d - Pc).reshape(len(X), -1)
        Jm = Jc.reshape(len(X), -1, 3)
        dX = np.linalg.solve(np.einsum("nij,nik->njk", Jm, Jm), np.einsum("nij,ni->nj", Jm, r)[..., None])[..., 0]
        Xh = Xh + dX
    Pc = np.zeros_like(P)
    Jc = np.zeros_like(J)
    for c, f in enumerate(camfiles):
        Pc[:, c], Jc[:, c] = sc.project_jacobian(f, Xh)
    r = d - Pc
    L = np.zeros_like(r)
    fd = sc.FitData(np.ones((len(X), n_cam), bool), d, r, Jc, L, np.arange(len(X)) % 10, np.arange(len(X)))
    fit = sc.Field(n_cam, 10, 6)
    noise_model = (noise, 0.0)
    half = (np.arange(len(X)) % 2) == 0
    lam, table = sc.choose_lambda(fit, fd.subset(half), fd.subset(~half), [1e1, 1e3, 1e5, 1e7], noise_model, log)
    rec = sc.fit_field(fit, fd, lam, noise_model, log=log)
    # the observable part: per point, N^T (E - c) against N^T E (unwhitened: noise is the same on all axes)
    C = np.stack([fit.eval_points(c, d[:, c]) for c in range(n_cam)], axis=1)
    Jm = Jc.reshape(len(X), -1, 3)
    U, _S, _V = np.linalg.svd(Jm, full_matrices=True)
    Nb = U[:, :, 3:]
    oe = np.einsum("nik,ni->nk", Nb, E.reshape(len(X), -1))
    oc = np.einsum("nik,ni->nk", Nb, C.reshape(len(X), -1))
    rms_obs = float(np.sqrt(np.mean(oe ** 2)))
    rms_err = float(np.sqrt(np.mean((oe - oc) ** 2)))
    full_err = float(np.sqrt(np.mean((E - C) ** 2)))
    full_true = float(np.sqrt(np.mean(E ** 2)))
    res = {"cameras": n_cam, "points": int(len(X)), "lam": lam, "fit": rec, "observable_rms_true": rms_obs,
           "observable_rms_error": rms_err, "full_rms_true": full_true, "full_rms_error": full_err,
           "noise_px": noise, "lambda_table": table}
    ok = rms_err < 0.25 * rms_obs and full_err < 2 * full_true
    check(log, f"{label}: the fit recovers the field's observable part, and stays bounded where the data do not see",
          ok, f"observable RMS {rms_obs:.3f} px, error {rms_err:.3f} px (noise {noise} px per axis); full field RMS "
              f"{full_true:.3f}, error {full_err:.3f} px (the unseen part: left at 0 by the prior)")
    return ok, res


def test_warp(log):
    import cv2
    H, W = 800, 1280
    f = sc.Field(1, 10, 6)
    out = {}
    # constant field (0.3, -0.7): every coefficient the constant
    f.theta[:f.nb] = 0.3
    f.theta[f.nb:2 * f.nb] = -0.7
    f.make_dense([np.array([[640.0, 400.0]])], cover_px=1e6)
    yy, xx = np.mgrid[0:H, 0:W]
    p = np.array([500.2, 300.6])
    img = 100 * np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * 1.5 ** 2))
    w = sc.warp(img, sc.warp_maps(f, 0))
    q = p - np.array([0.3, -0.7])
    m = (np.abs(xx - q[0]) <= 10) & (np.abs(yy - q[1]) <= 10)
    c = np.array([(w * xx)[m].sum() / w[m].sum(), (w * yy)[m].sum() / w[m].sum()])
    out["constant_shift_centroid"] = c.tolist()
    e1 = float(np.hypot(*(c - q)))
    ok1 = check(log, "warp by a constant c moves content by -c", e1 < 0.002,
                f"spot at {p.tolist()} -> {np.round(c, 4).tolist()}, expected {np.round(p - [0.3, -0.7], 4).tolist()} "
                f"(error {e1:.4f} px)")
    # a smooth field: spots at several places
    g = sc.Field(1, 10, 6)
    rng = np.random.default_rng(2)
    g.theta[:2 * g.nb] = rng.normal(0, 0.6, 2 * g.nb)
    g.make_dense([np.array([[640.0, 400.0]])], cover_px=1e6)
    errs = []
    maps = sc.warp_maps(g, 0)
    for _ in range(20):
        p = np.array([rng.uniform(100, 1180), rng.uniform(100, 700)])
        img = 100 * np.exp(-((xx - p[0]) ** 2 + (yy - p[1]) ** 2) / (2 * 1.5 ** 2))
        w = sc.warp(img, maps)
        cp = g.eval_dense(0, p[None])[0]
        q = p - cp
        win = (np.abs(xx - q[0]) <= 10) & (np.abs(yy - q[1]) <= 10)
        c = np.array([(w * xx)[win].sum() / w[win].sum(), (w * yy)[win].sum() / w[win].sum()])
        errs.append(float(np.hypot(*(c - q))))
    out["smooth_field_errors"] = errs
    ok2 = check(log, "warp by a smooth field moves content at p by -c(p)", max(errs) < 0.005,
                f"20 spots: max error {max(errs):.4f} px (field RMS {np.sqrt(np.mean(g.dense ** 2)):.2f} px)")
    del cv2
    return ok1 and ok2, out


def test_t1_frame(log):
    import cv2  # noqa: F401
    out = {}
    tracks, last = sc.read_tracks(sc.RUN_M15)
    n_pts = sum(len(t) for t in tracks.values())
    log(f"  the 15 px run: {len(tracks)} long tracks, {n_pts} points, last frame {last}")
    out["tracks"], out["points"] = len(tracks), n_pts
    fi = 100
    ti, ff, X, UV, keys = sc.track_points(tracks, [fi], min_len=10)
    noise = [7.514877384110587, 5.502475966380505]
    t0 = time.time()
    dets = [sc.detect(sc.read8(os.path.join(sc.PRE_M15, f"cam{c}", f"frame{4 * fi:06d}.tif")), noise[c - 1])
            for c in (1, 2)]
    out["detect_s"] = time.time() - t0
    pr = [sc.pair_one_to_one(dets[c][:, :2], UV[:, c]) for c in (0, 1)]
    both = (pr[0] >= 0) & (pr[1] >= 0)
    t0 = time.time()
    PJ = [sc.project_jacobian(sc.CAMS_T1[c], X) for c in (0, 1)]
    out["jacobian_s"] = time.time() - t0
    dproj = max(float(np.nanmax(np.abs(PJ[c][0] - UV[:, c]))) for c in (0, 1))
    out["stored_vs_computed_projection_px"] = dproj
    log(f"  frame {fi}: {len(X)} long-track points (>= 10 frames), detections {len(dets[0])} / {len(dets[1])} "
        f"({out['detect_s']:.1f} s); paired in cam1 {int((pr[0] >= 0).sum())}, cam2 {int((pr[1] >= 0).sum())}, both "
        f"{int(both.sum())}; Jacobians {out['jacobian_s']:.1f} s")
    ok1 = check(log, "stored projections = the cameras' projections of the stored points", dproj < 1e-4,
                f"max difference {dproj:.2e} px (the CSV keeps 6 decimals)")
    cfg = sc.write_matcher_config(sc.CAMS_T1, os.path.join(sc.WORK, "matcher_T1"),
                                  os.path.join(sc.CASE_DATA, "T1_otf_shape.txt"))
    cams, sm_ = sc.load_matcher(cfg)
    t0 = time.time()
    offs = []
    for k in np.flatnonzero(both)[:200]:
        P, _los = sc.curve(cams, sm_, 0, 1, dets[0][pr[0][k], :2])
        if P is None:
            continue
        offs.append(float(dets[1][pr[1][k], 1] - sc.curve_v(P, np.array([dets[1][pr[1][k], 0]]))[0]))
    out["curve_ms"] = (time.time() - t0) / max(len(offs), 1) * 1e3
    offs = np.array(offs)
    log(f"  epipolar offsets of 200 paired points: median {np.nanmedian(offs):+.3f} px, RMS "
        f"{np.sqrt(np.nanmean(offs ** 2)):.3f} px ({out['curve_ms']:.1f} ms per curve)")
    t0 = time.time()
    C, Wn = sc.candidate_pairs(cams, sm_, dets[0], dets[1], sc.view_volume(), 8.0, window=15.0)
    out["candidate_pairs_s"] = time.time() - t0
    log(f"  S2's candidate pairs within 8 px: {len(C)}; signed offsets within 15 px: {len(Wn)} "
        f"({out['candidate_pairs_s']:.1f} s)")
    out["candidate_pairs"] = int(len(C))
    out["window_offsets"] = int(len(Wn))
    # S2's own function on the same frame: the same pairs?
    import s2_shape_fit as s2
    t0 = time.time()
    ref = s2.candidate_pairs(cams, sm_, dets, sc.view_volume(), 8.0)
    out["s2_candidate_pairs_s"] = time.time() - t0
    A = {(int(r[0]), int(r[1])) for r in C}
    B = {(i, j) for (i, j, _X, _d, _g) in ref}
    ok2 = check(log, "the vectorised candidate pairs = S2's", A == B,
                f"{len(A)} vs {len(B)} pairs ({len(A ^ B)} differ); {out['candidate_pairs_s']:.1f} s vs "
                f"{out['s2_candidate_pairs_s']:.1f} s")
    return ok1 and ok2, out


def main():
    log = sc.Log("t0_selftest")
    lpt = sc.lib()
    log(f"T0 self-tests of selfcal.py; build {sc.build_record()['name']} ({lpt.__file__})")
    res = {}
    oks = []
    log("1. B-splines")
    ok, res["bsplines"] = test_bsplines(log)
    oks.append(ok)
    log("2. the projected fit recovers a known field")
    ok, res["fit_2cam"] = synth_fit(log, sc.CAMS_T1, "T1's 2 cameras")
    oks.append(ok)
    rig = [os.path.join(sc.STUDY, "data", "rig", f"cam{c}.txt") for c in (1, 2, 3, 4)]
    ok, res["fit_4cam"] = synth_fit(log, rig, "the study's 4-camera rig")
    oks.append(ok)
    log("3. the warp")
    ok, res["warp"] = test_warp(log)
    oks.append(ok)
    log("4. one T1 frame")
    ok, res["t1_frame"] = test_t1_frame(log)
    oks.append(ok)
    res["all_pass"] = all(oks)
    log(f"T0: {'all pass' if all(oks) else 'FAILURES'} ({sum(oks)} of {len(oks)} groups)")
    log.save(res)


if __name__ == "__main__":
    main()
