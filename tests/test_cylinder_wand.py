"""Wand calibration through a cylindrical port (PINCYL): the fast pytest subset.

The full campaign (plan, tests T0-T10, their outputs and the report) is in tests/cylinder_wand/; these are reduced
versions of its checks that run in well under a minute (`-m "not slow"` skips the calibration from scratch):
  * the fast model against the C++ PINCYL camera, both directions (T0);
  * its implicit-differentiation derivatives, and the bundle adjustment's Jacobian, against central differences
    (T1, T1b);
  * noiseless recovery through stages 3-4 (T3);
  * the gauge: full rank with it, and without it exactly the tube's stabiliser as the null space (T6);
  * the exported camFiles reload exactly (T10);
  * slow: a calibration from scratch reaches the same minimum as one started at the truth (T5).
The modules are imported from their folder directly, so the GUI package (Qt) is not loaded.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "tests" / "cylinder_wand"
for _p in (str(CAMPAIGN),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

lpt = pytest.importorskip("pyopenlpt", reason="extension not built")

import rigs  # noqa: E402  (puts curved_calibration/ on sys.path)
from rigs import cm  # noqa: E402
import cylinder_wand as cw  # noqa: E402
import cylinder_wand_io as cio  # noqa: E402

TUBE = rigs.CASE_TUBE


def _rig():
    return rigs.reference_rig(3, 45.0, sensor=(3200, 2400))


# ------------------------------------------------------------------ T0
@pytest.mark.parametrize("distorted", [False, True])
def test_fast_model_matches_cpp(distorted):
    rng = np.random.default_rng(11 + distorted)
    dist = np.array([0.05, -0.02, 2e-4, -1e-4, 0.005]) if distorted else None
    cam = rigs.ring_camera(30.0, 400.0, 3000.0, 2048, 2048, height_mm=10.0, aim=(5.0, -3.0, 2.0), roll_deg=7.0,
                           dist=dist)
    cpp = cm.to_cpp(cam, TUBE, tol=1e-13, nmax=300)
    # forward, single-image region, from 5 px off the C++ image
    rc = TUBE.fold_radius()
    r = np.sqrt(rng.uniform(0, (0.95 * rc) ** 2, 300))
    th = rng.uniform(0, 2 * np.pi, 300)
    P = np.column_stack([r * np.cos(th), rng.uniform(-30, 30, 300), r * np.sin(th)])
    uv = cm.cpp_project(cpp, P)
    keep = np.isfinite(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < 2048) & (uv[:, 1] >= 0) & (uv[:, 1] < 2048)
    P, uv = P[keep], uv[keep]
    rw = cm.rows([cam], np.zeros(len(P), int))
    pr = cm.project(rw, P, cm.undistort(uv + rng.normal(scale=5.0, size=uv.shape), rw), TUBE)
    assert pr.ok.all()
    assert np.max(np.abs(pr.uv - uv)) < 1e-6
    # back-projection: the C++ ray's mid-chord point reprojects onto the pixel (no distortion: the C++ undistorts
    # only to 1e-5 px)
    if not distorted:
        q = np.column_stack([rng.uniform(0, 2048, 300), rng.uniform(0, 2048, 300)])
        o, d, ok = cm.cpp_rays(cpp, q)
        o2, d2, ok2, _, _ = cm.trace(cm.rows([cam], np.zeros(300, int)), cm.undistort(q, cm.rows(
            [cam], np.zeros(300, int))), TUBE)
        assert np.array_equal(ok, ok2)
        assert np.max(np.linalg.norm(o[ok] - o2[ok], axis=1)) < 1e-8
        assert np.max(np.linalg.norm(d[ok] - d2[ok], axis=1)) < 1e-11


# ------------------------------------------------------------------ T1
def test_projection_derivatives():
    rng = np.random.default_rng(5)
    cam = rigs.ring_camera(-40.0, 700.0, 20000.0, 2048, 2048, height_mm=-5.0, aim=(2.0, 1.0, -3.0), roll_deg=4.0,
                           dist=np.array([0.05, 0.0, 0.0, 0.0, 0.0]))
    cpp = cm.to_cpp(cam, TUBE)
    P = rng.uniform(-25, 25, (60, 3))
    uv = cm.cpp_project(cpp, P)
    P, uv = P[np.isfinite(uv[:, 0])], uv[np.isfinite(uv[:, 0])]
    rw = cm.rows([cam], np.zeros(len(P), int))
    base = cm.project(rw, P, cm.undistort(uv, rw), TUBE, jac=True, tol=1e-16, max_iter=60)

    def proj(c, X):
        return cm.project(cm.rows([c], np.zeros(len(X), int)), X, base.xy, TUBE, tol=1e-16, max_iter=60).uv

    def rich(fun, h):
        d1 = (fun(h) - fun(-h)) / (2 * h)
        d2 = (fun(h / 2) - fun(-h / 2)) / h
        return (4 * d2 - d1) / 3

    def cam_with(name, h):
        k = cam.copy()
        if name.startswith("w"):
            w = np.zeros(3)
            w[int(name[1]) - 1] = h
            k.R = cam.R @ cm.exp_so3(w)
        elif name.startswith("C"):
            k.C[int(name[1]) - 1] += h
        elif name == "logf":
            k.f = cam.f * math.exp(h)
        return k
    for name, h in (("w1", 1e-5), ("w2", 1e-5), ("w3", 1e-5), ("C1", 1e-2), ("C2", 1e-2), ("C3", 1e-2),
                    ("logf", 1e-6)):
        num = rich(lambda hh: proj(cam_with(name, hh), P), h)
        col = base.J_cam[:, :, cm.CAM_PARAMS.index(name)]
        assert np.max(np.linalg.norm(col - num, axis=1)) / np.sqrt(np.mean(np.sum(col ** 2, axis=1))) < 1e-6, name
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        num = rich(lambda hh: proj(cam, P + hh * e), 1e-3)
        col = base.J_X[:, :, i]
        assert np.max(np.linalg.norm(col - num, axis=1)) / np.sqrt(np.mean(np.sum(col ** 2, axis=1))) < 1e-6


def test_bundle_adjustment_jacobian():
    rng = np.random.default_rng(3)
    truth = _rig()
    wd = rigs.wand_frames(truth, TUBE, 15, 10.0, rng, half=(15, 15, 15), noise_px=0.0)
    obs = cw.observations(cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in truth]))
    cams = rigs.perturb_rig(truth, rng, rot_deg=0.05, trans_mm=0.5, f_rel=0.002)
    fr = cw.frames_from_ends(wd.XA[obs.frame_ids] + 0.05, wd.XB[obs.frame_ids] - 0.03)
    lay = cw.Layout(3, fit_L=True)
    rows = np.arange(len(obs))
    ev = cw.evaluate(cams, TUBE, fr, 10.0, obs, lay, rows)

    def res(cs, frames, L):
        return cw.evaluate(cs, TUBE, frames, L, obs, None, rows).r
    for g in range(lay.P):
        h = 1e-5 if (".w" in lay.names[g] or "logf" in lay.names[g]) else 1e-2
        dx = np.zeros(lay.P)
        dx[g] = h
        cp, Lp = lay.apply(cams, 10.0, dx)
        cmn, Lm = lay.apply(cams, 10.0, -dx)
        num = (res(cp, fr, Lp) - res(cmn, fr, Lm)) / (2 * h)
        col = ev.Jg[:, :, g]
        assert np.max(np.linalg.norm(col - num, axis=1)) / np.sqrt(np.mean(np.sum(col ** 2, axis=1))) < 1e-5
    for k in range(5):
        h = 1e-3 if k < 3 else 1e-5
        df = np.zeros((obs.n_frames, 5))
        df[:, k] = h
        num = (res(cams, fr.retract(df), 10.0) - res(cams, fr.retract(-df), 10.0)) / (2 * h)
        col = ev.Jf[:, :, k]
        assert np.max(np.linalg.norm(col - num, axis=1)) / np.sqrt(np.mean(np.sum(col ** 2, axis=1))) < 1e-5


# ------------------------------------------------------------------ T3, T6
def test_noiseless_recovery_and_gauge():
    rng = np.random.default_rng(31)
    truth = _rig()
    wd = rigs.wand_frames(truth, TUBE, 150, 10.0, rng, half=(20, 20, 20), noise_px=0.0)
    obs = cw.observations(cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in truth]))
    start = rigs.perturb_rig(truth, rng, rot_deg=1.0, trans_mm=10.0, f_rel=0.03)
    lay = cw.Layout(3)
    cams3, _ = cw.coarse_fit(start, TUBE, obs, 10.0, lay)
    sol = cw.refine(cams3, TUBE, cw.initial_frames(cams3, TUBE, obs, 10.0), 10.0, obs, lay)
    assert sol.rms < 1e-6
    for e in rigs.compare(sol.cams, truth):
        assert e["rot_deg"] < 1e-6 and e["centre_mm"] < 1e-5 and abs(e["f_rel"]) < 1e-8
    # gauge: full rank with it; without it, two null directions spanned by the stabiliser
    S_g, _ = cw.information(sol.cams, TUBE, sol.frames, sol.L, obs, lay)
    d = np.sqrt(np.diag(S_g))
    sv = np.linalg.svd(S_g / np.outer(d, d), compute_uv=False)
    assert sv[-1] / sv[0] > 1e-12
    free = cw.Layout(3, gauge=False)
    S_f, _ = cw.information(sol.cams, TUBE, sol.frames, sol.L, obs, free)
    d = np.sqrt(np.diag(S_f))
    _u, sf, vt = np.linalg.svd(S_f / np.outer(d, d))
    assert sf[-2] / sf[0] < 1e-12 and sf[-3] > 1e3 * sf[-2]
    ey = np.array([0.0, 1.0, 0.0])
    gen = np.zeros((free.P, 2))
    for c, cam in enumerate(sol.cams):
        for j, nm in enumerate(free.cols):
            g = free.pos[c, j]
            if nm in ("w1", "w2", "w3"):
                gen[g, 0] = -ey[j]
            elif nm in ("C1", "C2", "C3"):
                gen[g, 0] = np.cross(ey, cam.C)[j - 3]
                gen[g, 1] = ey[j - 3]
    q, _ = np.linalg.qr(gen * d[:, None])
    cosines = np.linalg.svd(q.T @ vt[-2:].T, compute_uv=False)
    assert np.min(cosines) > 1 - 1e-10


# ------------------------------------------------------------------ T10
def test_export_roundtrip(tmp_path):
    rng = np.random.default_rng(10)
    truth = _rig()
    wd = rigs.wand_frames(truth, TUBE, 80, 10.0, rng, half=(15, 15, 15), noise_px=0.1)
    data = cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in truth])
    sol = cw.calibrate(data, TUBE, start=truth)
    res = cio.write_result(sol, str(tmp_path))
    cams_out, _fr, y0 = cw.relabel(sol.cams, sol.frames)
    P = rng.uniform(-15, 15, (200, 3))
    for c, cam in enumerate(cams_out):
        loaded = lpt.Camera(str(tmp_path / res["cameras"][c]["camfile"]))
        a = cm.cpp_project(loaded, P)
        b = cm.cpp_project(cm.to_cpp(cam, TUBE), P)
        ok = np.isfinite(a[:, 0])
        assert ok.sum() > 150
        assert np.max(np.abs(a[ok] - b[ok])) <= 1e-9
    assert (tmp_path / "calibration_result.json").exists() and (tmp_path / "calibration_report.txt").exists()
    # the report says what the noise weighting did (T12): one row per camera, with the detections accounted for
    nz = res["noise"]
    assert nz["mode"] == "camera" and len(nz["groups"]) == len(truth)
    assert all(g["n_used"] + g["n_not_used"] == g["n_total"] for g in nz["groups"])
    assert "Noise levels and weights" in (tmp_path / "calibration_report.txt").read_text(encoding="utf-8")
    # the rig configuration round-trips
    cfg = cio.RigConfig(radii=list(TUBE.radii), n=list(TUBE.n), wand_length_mm=10.0,
                        cameras=[{"name": f"cam{c + 1}", "width": cam.width, "height": cam.height, "pixel_mm": 0.02}
                                 for c, cam in enumerate(truth)])
    cio.save_config(cfg, str(tmp_path / "rig.json"))
    back = cio.load_config(str(tmp_path / "rig.json"))
    assert back.tube == TUBE and back.sizes == [(c.width, c.height) for c in truth] and back.wand_length_mm == 10.0


# ------------------------------------------------------------------ T5
@pytest.mark.slow
def test_from_scratch_reaches_the_global_minimum():
    rng = np.random.default_rng(5003)
    truth, centre, half = rigs.random_rig(rng)
    wd = rigs.wand_frames(truth, TUBE, 300, 10.0, rng, half=(half, half, half), centre=centre, noise_px=0.1)
    data = cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in truth])
    ref = cw.calibrate(data, TUBE, start=truth)
    sol = cw.calibrate(data, TUBE)
    assert abs(sol.rms - ref.rms) < 1e-6
    for e in rigs.compare(sol.cams, ref.cams):
        assert e["rot_deg"] < 1e-4 and e["centre_mm"] < 1e-2 and abs(e["f_rel"]) < 1e-5
