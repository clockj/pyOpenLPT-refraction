"""Wand calibration through flat windows (PINPLATE): the fast pytest subset.

The full campaign (plan, tests W0-W9, their outputs and the report) is in tests/plate_wand/; these are reduced versions
of its checks (`-m "not slow"` skips the calibration from scratch):
  * the model against the C++ PINPLATE camera: projections and lines of sight (W0);
  * its derivatives, and the bundle adjustment's Jacobian, against central differences (W1);
  * noiseless recovery from the truth and from a perturbed start (W3);
  * the camFile and CSV round trips (W9);
  * slow: a calibration from scratch reaches the same minimum as one started at the truth (W5, rig D).
The modules are imported from their folder directly, so the GUI package (Qt) is not loaded.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "tests" / "plate_wand"
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))

lpt = pytest.importorskip("pyopenlpt", reason="extension not built")

import prigs  # noqa: E402  (puts curved_calibration/ on sys.path)
from prigs import cm, pm  # noqa: E402
import plate_wand as pw  # noqa: E402
import plate_wand_io as pio  # noqa: E402


def _case(rng, two_plates=False, distorted=False):
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    t, idx = ((10.0, 5.0), (1.0, 1.49, 1.6, 1.333)) if two_plates else ((12.7,), (1.0, 1.49, 1.333))
    win = pm.Window(n, rng.uniform(-50, 50), t, idx)
    t1, t2 = pm.tangent_basis(n)
    th = math.radians(rng.uniform(0, 45))
    u = math.cos(th) * n + math.sin(th) * t1[0]
    C = win.d * n - 500.0 / math.cos(th) * u
    R = prigs.look_dir(u, up=rng.normal(size=3), roll_deg=rng.uniform(-180, 180))
    dist = np.array([0.05, -0.02, 2e-4, -1e-4, 0.005]) if distorted else np.zeros(5)
    return pm.Camera(R, C, 4000.0, 1023.5, 1023.5, 2048, 2048, dist), win


@pytest.mark.parametrize("two_plates", [False, True])
def test_model_matches_cpp(two_plates):
    rng = np.random.default_rng(3 + two_plates)
    for _ in range(20):
        cam, win = _case(rng, two_plates)
        cpp = pm.to_cpp(cam, win)
        uv = np.column_stack([rng.uniform(0, 2047, 30), rng.uniform(0, 2047, 30)])
        o, d, ok = cm.cpp_rays(cpp, uv)
        rw, ww = pm.rows([cam], np.zeros(30, int)), pm.wrows([win], np.zeros(30, int))
        om, dm, okm = pm.rays(rw, ww, uv)
        sel = ok & okm
        assert sel.sum() >= 20
        assert np.max(np.abs(dm[sel] - d[sel])) < 1e-10
        assert np.max(np.linalg.norm(om[sel] - o[sel], axis=1)) < 1e-8
        X = o[sel] + rng.uniform(1, 300, (sel.sum(), 1)) * d[sel]
        pr = pm.project(pm.rows([cam], np.zeros(len(X), int)), pm.wrows([win], np.zeros(len(X), int)), X)
        assert pr.ok.all()
        assert np.max(np.abs(pr.uv - cm.cpp_project(cpp, X))) < 1e-6
        assert np.max(np.abs(pr.uv - uv[sel])) < 1e-6


def test_projection_derivatives():
    rng = np.random.default_rng(5)
    cam, win = _case(rng, distorted=True)
    o, d, _ok = pm.rays(pm.rows([cam], np.zeros(10, int)), pm.wrows([win], np.zeros(10, int)),
                        np.column_stack([rng.uniform(200, 1800, 10), rng.uniform(200, 1800, 10)]))
    X = o + rng.uniform(5, 200, (10, 1)) * d
    A = win.d * win.n + 50.0 * pm.tangent_basis(win.n)[0][0]

    def proj(c, w, P):
        return pm.project(pm.rows([c], np.zeros(len(P), int)), pm.wrows([w], np.zeros(len(P), int), anchors=[A]), P,
                          jac=True)

    base = proj(cam, win, X)
    for i, h in ((0, 1e-6), (3, 1e-4), (6, 1e-7), (9, 1e-6)):          # w1, C1, logf, k1
        cp, cn = cam.copy(), cam.copy()
        if i == 0:
            cp.R, cn.R = cam.R @ pm.exp_so3([h, 0, 0]), cam.R @ pm.exp_so3([-h, 0, 0])
        elif i == 3:
            cp.C, cn.C = cam.C + [h, 0, 0], cam.C - [h, 0, 0]
        elif i == 6:
            cp.f, cn.f = cam.f * math.exp(h), cam.f * math.exp(-h)
        else:
            cp.dist, cn.dist = cam.dist + [h, 0, 0, 0, 0], cam.dist - [h, 0, 0, 0, 0]
        fd = (proj(cp, win, X).uv - proj(cn, win, X).uv) / (2 * h)
        assert np.max(np.abs(fd - base.J_cam[:, :, i])) <= 1e-6 * np.max(np.abs(base.J_cam[:, :, i]))
    for j, h in ((0, 1e-7), (2, 1e-5)):                                 # a, delta
        if j == 0:
            wp = pm.retract_normal(win.n, (h, 0.0))
            wn = pm.retract_normal(win.n, (-h, 0.0))
            Wp = pm.Window(wp, float(wp @ A) + win.d - float(win.n @ A), win.t, win.idx)
            Wn = pm.Window(wn, float(wn @ A) + win.d - float(win.n @ A), win.t, win.idx)
        else:
            Wp = pm.Window(win.n, win.d + h, win.t, win.idx)
            Wn = pm.Window(win.n, win.d - h, win.t, win.idx)
        fd = (proj(cam, Wp, X).uv - proj(cam, Wn, X).uv) / (2 * h)
        assert np.max(np.abs(fd - base.J_win[:, :, j])) <= 1e-6 * np.max(np.abs(base.J_win[:, :, j]))
    fd = (proj(cam, win, X + [0, 0, 1e-5]).uv - proj(cam, win, X - [0, 0, 1e-5]).uv) / 2e-5
    assert np.max(np.abs(fd - base.J_X[:, :, 2])) <= 1e-6 * np.max(np.abs(base.J_X[:, :, 2]))


def test_bundle_adjustment_jacobian():
    rig_t = prigs.rig_B()
    wand = prigs.make_wand(rig_t, 20, np.random.default_rng(2), noise_px=0.3)
    obs = pw.observations(pw.WandData(wand.uv, wand.L, rig_t.sizes))
    rig = pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win)
    frames = pw.frames_from_ends(wand.XA[obs.frame_ids], wand.XB[obs.frame_ids])
    layout = pw.Layout(rig_t.n_cams, 2, fit_L=True, anchors=pw.anchors_for(rig_t.cams, rig))
    ev = pw.evaluate(rig_t.cams, rig, frames, wand.L, obs, layout)
    for k in range(layout.P):
        h = 1e-7 if (".w" in layout.names[k] or layout.names[k].endswith((".a", ".b")) or "logf" in layout.names[k]) \
            else 1e-5
        dx = np.zeros(layout.P)
        dx[k] = h
        cp, rp, Lp = layout.apply(rig_t.cams, rig, wand.L, dx)
        cn, rn, Ln = layout.apply(rig_t.cams, rig, wand.L, -dx)
        fd = (pw.evaluate(cp, rp, frames, Lp, obs).r - pw.evaluate(cn, rn, frames, Ln, obs).r) / (2 * h)
        assert np.max(np.abs(fd - ev.Jg[:, :, k])) <= 1e-6 * max(np.max(np.abs(ev.Jg[:, :, k])), 1e-12), layout.names[k]


def test_noiseless_recovery():
    rig_t = prigs.rig_A()
    wand = prigs.make_wand(rig_t, 300, np.random.default_rng(4), noise_px=0.0)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    cams0 = [c.copy() for c in rig_t.cams]
    for c in cams0[1:]:
        c.R = c.R @ pm.exp_so3([0.004, -0.003, 0.002])
        c.C = c.C + [3.0, -2.0, 4.0]
        c.f *= 1.02
    w0 = rig_t.wins[0]
    n0 = pm.retract_normal(w0.n, (0.01, -0.01))
    sol = pw.solve(data, cams0, pw.Rig([pm.Window(n0, w0.d + 3.0, w0.t, w0.idx)], rig_t.cam_win),
                   noise_weights="none")
    assert sol.rms < 1e-8
    for c, t in zip(sol.cams, rig_t.cams):
        assert pm.rotation_angle_deg(c.R, t.R) < 1e-9
        assert np.linalg.norm(c.C - t.C) < 1e-7
        assert abs(c.f / t.f - 1) < 1e-10
    assert np.linalg.norm(sol.rig.windows[0].n - w0.n) < 1e-10
    assert abs(sol.rig.windows[0].d - w0.d) < 1e-7


def test_files_roundtrip(tmp_path):
    rig_t = prigs.rig_B()
    rng = np.random.default_rng(6)
    P = prigs.cloud_points(rig_t, rng, 100)
    for c, cam in enumerate(rig_t.cams):
        path = tmp_path / f"cam{c}.txt"
        pio.write_camfile(cam, rig_t.win_of(c), str(path), (0.1, 0.02), (0.01, 0.002))
        cpp = lpt.Camera(str(path))
        pr = pm.project(pm.rows([cam], np.zeros(len(P), int)), pm.wrows([rig_t.win_of(c)], np.zeros(len(P), int)), P)
        assert np.nanmax(np.abs(cm.cpp_project(cpp, P) - pr.uv)) < 1e-6
        c2, w2 = pio.load_camfile(str(path))
        assert np.max(np.abs(c2.R - cam.R)) < 1e-14 and abs(w2.d - rig_t.win_of(c).d) < 1e-9
    wand = prigs.make_wand(rig_t, 50, rng, noise_px=0.1)
    pio.write_gui_csv(str(tmp_path / "wand.csv"), wand.uv)
    back = pio.read_gui_csv(str(tmp_path / "wand.csv"))
    assert np.nanmax(np.abs(back.uv - wand.uv)) < 1e-6


@pytest.mark.slow
def test_from_scratch_reaches_the_global_minimum():
    rig_t = prigs.rig_D()
    wand = prigs.make_wand(rig_t, 600, np.random.default_rng(7), noise_px=0.1)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    sol = pw.calibrate(data, pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx) for w in rig_t.wins], rig_t.cam_win))
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    cmp = prigs.same_minimum(sol, ref)
    assert cmp["same"], cmp
