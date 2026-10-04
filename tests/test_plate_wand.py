"""Wand calibration through flat windows (PINPLATE): the fast pytest subset.

The full campaign (plan, tests W0-W9, their outputs and the report) is in tests/plate_wand/; these are reduced versions
of its checks (`-m "not slow"` skips the calibration from scratch):
  * the model against the C++ PINPLATE camera: projections and lines of sight (W0);
  * its derivatives, and the bundle adjustment's Jacobian, against central differences (W1);
  * noiseless recovery from the truth and from a perturbed start (W3);
  * the camFile and CSV round trips (W9);
  * the perturbed restarts' draws: deterministic, camera 1's pose held, every observation projecting, their spread
    (R2 of tests/plate_wand/2026-10-02-perturbed-starts-plan.md);
  * the window-normal hints: their conversion into a pinhole camera's frame, a held normal in the fit, the
    hint-searched start's merge and how far apart two answers are (`answers_apart`); slow: a calibration with hints
    10 deg off runs the hinted start paths and reaches the global minimum (2026-10-03-angles-and-hints-plan.md);
  * weakly seen windows: the rule, the tilt the scan uses, and the report's warnings; slow: the scan of a weak
    window's normal brings W5 seed 1001 to the global minimum without hints (H4);
  * slow: a calibration from scratch reaches the same minimum as one started at the truth (W5, rig D), and the
    perturbed restarts free W5 seed 2000's answer, pinned against a window (R6);
  * the optional lens terms (fit_lens: principal point, k2, tangential) and constraints (a channel the wand stays in,
    where cameras aim), added for the real case cases/plate_20260727_T0 (its 2026-10-03-walls-plan.md, L1 and L2):
    the terms recovered on noiseless data, the channel rows' Jacobian, a channel the truth satisfies changes nothing
    and one it breaks is met, an aim held.
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


def test_perturbed_draws():
    rig_t = prigs.rig_B()
    wand = prigs.make_wand(rig_t, 150, np.random.default_rng(8), noise_px=0.3)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    sol = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    seed = pw._perturb_seed(data)
    assert seed == pw._perturb_seed(pw.WandData(wand.uv.copy(), wand.L, rig_t.sizes))
    draws, left_out = pw.perturbed_draws(sol, sol.obs, 400, 2.0, np.random.default_rng(seed))
    again, _ = pw.perturbed_draws(sol, sol.obs, 5, 2.0, np.random.default_rng(seed))
    assert left_out == 0 and len(draws) == 400
    assert all(np.array_equal(a["dx"], b["dx"]) for a, b in zip(draws, again))
    rows = np.flatnonzero(sol.obs.active)
    for d in draws[:20]:
        assert np.array_equal(d["cams"][0].R, sol.cams[0].R) and np.array_equal(d["cams"][0].C, sol.cams[0].C)
        assert pw.evaluate(d["cams"], d["rig"], d["frames"], d["L"], sol.obs, rows=rows).ok.all()
    ratio = np.std([d["dx"] for d in draws], axis=0, ddof=1) / (2.0 * np.sqrt(np.diag(sol.cov)))
    assert np.all(np.abs(ratio - 1.0) < 0.2), ratio


def test_capped_cost():
    rig_t = prigs.rig_B()
    wand = prigs.make_wand(rig_t, 150, np.random.default_rng(8), noise_px=0.3)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    sol = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win),
                   noise_weights="none")
    obs = sol.obs
    r2 = np.sum(pw.evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs).r ** 2, axis=1)
    s02 = sol.sigma_px ** 2
    cap = 25.0 * s02
    assert r2.max() < cap                                       # nothing near the cap: the plain sum
    assert pw.capped_cost(sol, obs, None, s02) == pytest.approx(float(r2.sum()), rel=1e-12)
    out = pw.Obs(obs.frame, obs.cam, obs.end, obs.uv.copy(), obs.n_frames, obs.n_cams, obs.frame_ids, obs.active)
    out.uv[0] += 100.0                                          # an outlier counts the cap
    assert pw.capped_cost(sol, out, None, s02) == pytest.approx(float(r2.sum() - r2[0] + cap), rel=1e-12)
    w0 = sol.rig.windows[0]                                     # window 1 moved behind its cameras: none projects
    d_bad = min(float(w0.n @ sol.cams[c].C) for c in sol.rig.cams_of(0)) - 1.0
    rig2 = pw.Rig([pm.Window(w0.n, d_bad, w0.t, w0.idx)] + [w.copy() for w in sol.rig.windows[1:]], sol.rig.cam_win)
    bad = np.isin(obs.cam, sol.rig.cams_of(0))
    sol2 = pw.Solution(sol.cams, rig2, sol.frames, sol.L, obs, sol.layout)
    assert pw.capped_cost(sol2, obs, None, s02) == pytest.approx(float(r2[~bad].sum() + cap * bad.sum()), rel=1e-12)


def test_window_normal_hints():
    """A hinted normal (in its camera's frame) turned into its pinhole camera's frame is the inverse of
    small_angle_camera, for either sign; several cameras' hints average; a camera turned 30 deg to the left sees
    (sin 30, 0, cos 30) (the docstrings' example); a held normal (a hint-held start) stays put in the fit while its
    window's offset moves, and the other window stays free."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = rng.normal(size=3)
        win = pm.Window(n, float(rng.uniform(-50, 50)), (12.7,), (1.0, 1.49, 1.333))
        cams_p = []
        for _k in range(2):
            e = np.cross(win.n, rng.normal(size=3))
            e /= np.linalg.norm(e)
            th = math.radians(rng.uniform(2.0, 40.0))            # the chief ray in the water, from the normal
            dw = math.cos(th) * win.n + math.sin(th) * e
            cam_p = pm.Camera(prigs.look_dir(dw, roll_deg=float(rng.uniform(-180, 180))), win.d * win.n - 600.0 * dw,
                              3000.0, 639.5, 399.5, 1280, 800)
            c = pw.small_angle_camera(cam_p, win)
            v = c.R @ win.n
            assert np.allclose(pw.hint_to_frame(v, cam_p, win), win.n, atol=1e-12)
            assert np.allclose(pw.hint_to_frame(-3.0 * v, cam_p, win), win.n, atol=1e-12)
            cams_p.append((cam_p, v))
        hints = {"normals": [cams_p[0][1], None, cams_p[1][1]]}
        got = pw.hinted_normal(hints, [0, 2], {0: cams_p[0][0], 2: cams_p[1][0]}, win)
        assert np.allclose(got, win.n, atol=1e-12)
        assert pw.hinted_normal(hints, [1], {1: cams_p[0][0]}, win) is None
        assert pw.hinted_normal({}, [0], {0: cams_p[0][0]}, win) is None
    a = math.radians(30.0)
    assert prigs.look_dir([0.0, 0.0, 1.0])[0][0] < 0             # looking along +Z, the image's left is +X ...
    R = prigs.look_dir([math.sin(a), 0.0, math.cos(a)])          # ... so this camera is turned 30 deg to the left
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [0.5, 0.0, math.cos(a)], atol=1e-12)
    rig_t = prigs.rig_B()
    wand = prigs.make_wand(rig_t, 150, np.random.default_rng(8), noise_px=0.1)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    w0, w1 = rig_t.wins
    n_held = pm.retract_normal(w0.n, (0.01, -0.02))                # 1.3 deg off, as a rough hint
    w1_off = pm.Window(pm.retract_normal(w1.n, (0.005, 0.0)), w1.d, w1.t, w1.idx)
    start = pw.Rig([pm.Window(n_held, w0.d + 2.0, w0.t, w0.idx), w1_off], rig_t.cam_win)
    sol = pw.solve(data, [c.copy() for c in rig_t.cams], start, held_normals=(0,), noise_weights="none")
    assert np.allclose(sol.rig.windows[0].n, n_held, atol=1e-14)    # held
    assert abs(sol.rig.windows[0].d - (w0.d + 2.0)) > 0.1            # its offset was fitted
    assert pw.Layout(4, 2, held_normals=(0,)).wpos[0].tolist()[:2] == [-1, -1]
    ws = pw.window_sigmas(sol.cams, sol.rig, sol.layout, sol.cov)
    assert ws[0].get("normal_held") and ws[0]["offset_mm"] > 0 and ws[1]["normal_deg"] > 0
    assert prigs.compare_wins([sol.rig.windows[1]], [w1_off])[0]["normal_deg"] > 0.05    # the other one is free


def test_hint_paths_helpers():
    """The hint-searched start's merge keeps every default candidate, by score, and drops a hint candidate that
    repeats one (normal within 1 deg, plane within 5 % of the camera's distance); `answers_apart` is 0 for one answer,
    tiny for the same minimum reached from another start, and large for another state of the rig."""
    win = pm.Window([0.0, 0.0, 1.0], 100.0, (12.7,), (1.0, 1.49, 1.333))
    cam = pm.Camera(np.eye(3), np.array([0.0, 0.0, -400.0]), 3000.0, 639.5, 399.5, 1280, 800)
    tilted = pm.Window(pm.retract_normal(win.n, (0.2, 0.0)), 100.0, win.t, win.idx)
    base = [(1.0, win, {0: cam}), (3.0, tilted, {0: cam})]
    repeat = (0.5, pm.Window(pm.retract_normal(win.n, (0.01, 0.0)), 101.0, win.t, win.idx), {0: cam})  # 0.6 deg, 1 mm
    other = pm.Window(pm.retract_normal(win.n, (0.0, 0.1)), 100.0, win.t, win.idx)                     # 5.7 deg
    merged, from_hint = pw._merge_hinted(base, [repeat, (2.0, other, {0: cam})])
    assert [m[0] for m in merged] == [1.0, 2.0, 3.0] and merged[1][1] is other and not from_hint
    merged, from_hint = pw._merge_hinted(base, [(0.5, other, {0: cam})])
    assert [m[0] for m in merged] == [0.5, 1.0, 3.0] and from_hint
    rig_t = prigs.rig_B()
    wand = prigs.make_wand(rig_t, 150, np.random.default_rng(8), noise_px=0.3)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    sol = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win),
                   noise_weights="none")
    s02 = sol.sigma_px ** 2
    assert pw.answers_apart(sol, sol, sol.obs, None, s02) == 0.0
    cams2 = [c.copy() for c in rig_t.cams]
    cams2[1].f *= 1.01
    cams2[2].C = cams2[2].C + np.array([1.0, -1.0, 0.5])
    sol2 = pw.solve(data, cams2, pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win), noise_weights="none")
    assert pw.answers_apart(sol, sol2, sol.obs, None, s02) < 1e-2                  # the same minimum
    w0 = sol.rig.windows[0]
    moved = pw.Rig([pm.Window(pm.retract_normal(w0.n, (0.01, 0.0)), w0.d, w0.t, w0.idx)] + sol.rig.windows[1:],
                   sol.rig.cam_win)
    sol3 = pw.Solution(sol.cams, moved, sol.frames, sol.L, sol.obs, sol.layout)
    assert pw.answers_apart(sol, sol3, sol.obs, None, s02) > 100.0                # window 1 turned 0.6 deg


def _truth_solution(rig_t, n_frames, seed, sigma_px=0.1):
    """A Solution at the truth, not fitted: the true cameras, windows and wand frames, every observation active."""
    wand = prigs.make_wand(rig_t, n_frames, np.random.default_rng(seed), noise_px=0.1)
    obs = pw.observations(pw.WandData(wand.uv, wand.L, rig_t.sizes))
    rig = pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win)
    sol = pw.Solution([c.copy() for c in rig_t.cams], rig, pw.frames_from_ends(wand.XA[obs.frame_ids],
                                                                                 wand.XB[obs.frame_ids]),
                      wand.L, obs, pw.Layout(len(rig_t.cams), len(rig.windows)))
    sol.sigma_px = sigma_px
    return sol


def test_warnings(tmp_path):
    """Each warning of the report (plate_wand_io.assess) on a case built for it, none on a sound rig, and the WARNINGS
    block in calibration_report.txt."""
    def codes(s):
        return [x["code"] for x in pio.assess(s)]
    sol = _truth_solution(prigs.rig_B(), 400, 8)
    assert codes(sol) == []                                         # two windows, two moderate lenses each
    sol.sigma_px = 1.0
    assert codes(sol) == ["noisy"]
    sol.sigma_px = 0.1
    sol.info["hints"] = {"normals_off_deg": [20.0, 2.0, None, 1.0]}
    w = pio.assess(sol)
    assert [x["code"] for x in w] == ["hint_far"] and w[0]["cameras"] == [0]
    sol.info = {}
    d0 = pw.window_depths(sol.cams, sol.rig, sol.frames, sol.L, sol.obs)[0]
    w0 = sol.rig.windows[0]                                         # window 1 moved onto its nearest wand end
    sol.rig = pw.Rig([pm.Window(w0.n, w0.d + d0, w0.t, w0.idx)] + sol.rig.windows[1:], sol.rig.cam_win)
    w = pio.assess(sol)
    assert [x["code"] for x in w] == ["pinned"] and w[0]["window"] == 0
    assert codes(_truth_solution(prigs.rig_C(), 400, 8)) == ["same_way"]          # three cameras 5 deg apart
    weak = pio.assess(_truth_solution(prigs.random_rig(np.random.default_rng(1001)), 400, 8))
    assert {x["code"] for x in weak} == {"weak_window"}                  # W5 seed 1001: one camera per window ...
    assert any(x["window"] == 1 and x["view_deg"][0] < 3.0 for x in weak)  # ... window 2's of 24,500 px
    sol = _truth_solution(prigs.rig_A(), 150, 8)
    sol.layout = pw.Layout(len(sol.cams), 1, fit_L=True)
    assert codes(sol) == ["few_frames", "wand_length_fitted"]
    rig_t = prigs.rig_B()
    wand = prigs.make_wand(rig_t, 150, np.random.default_rng(8), noise_px=0.1)
    fit = pw.solve(pw.WandData(wand.uv, wand.L, rig_t.sizes), [c.copy() for c in rig_t.cams],
                   pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    res = pio.write_result(fit, str(tmp_path))
    assert [x["code"] for x in res["warnings"]] == ["few_frames"]
    text = (tmp_path / "calibration_report.txt").read_text(encoding="utf-8")
    assert "WARNINGS (1):" in text and "frames used" in text


def test_weak_windows_and_tilt():
    """The weak-window rule (one or two cameras that all see the wand under less than 6 deg): none on rig B, window 2
    of W5 seed 1001's rig (one camera of 24,500 px); a window tilted for the scan keeps every active wand end at least
    as deep in the water as before, its plane turned about the anchor otherwise."""
    sol = _truth_solution(prigs.rig_B(), 400, 8)
    assert pw.weak_windows(sol.cams, sol.rig, sol.obs) == []
    assert all(4.0 < a < 60.0 for a in pw.view_angles_deg(sol.cams, sol.obs))
    s1001 = _truth_solution(prigs.random_rig(np.random.default_rng(1001)), 400, 8)
    assert pw.weak_windows(s1001.cams, s1001.rig, s1001.obs) == [1]
    for w in (0, 1):
        win = sol.rig.windows[w]
        sel = sol.obs.active & np.isin(sol.obs.cam, sol.rig.cams_of(w))
        XA, XB = sol.frames.ends(sol.L)
        X = np.where((sol.obs.end[sel] == 0)[:, None], XA[sol.obs.frame[sel]], XB[sol.obs.frame[sel]])
        d0 = float(np.min(win.depth(X)))
        for ab in ((0.05, 0.0), (-0.05, 0.03), (0.0, -0.1)):
            n_new = pm.retract_normal(win.n, ab)
            rg = pw._tilted(sol.cams, sol.rig, sol.frames, sol.L, sol.obs, w, n_new)
            wn = rg.windows[w]
            assert np.allclose(wn.n, n_new, atol=1e-15) and rg.windows[1 - w] is not sol.rig.windows[1 - w]
            assert float(np.min(wn.depth(X))) >= d0 - 1e-9
            A = pw.anchors_for(sol.cams, sol.rig)[w]
            pivot = float(n_new @ A) + (win.d - float(win.n @ A))
            assert wn.d <= pivot + 1e-12                              # moved only towards the cameras, if at all


def test_cpp_rotation_check_tolerates_rounding():
    """pm.to_cpp's check of the C++ rotation tolerates rounding: this camera (case plate_20260727_T0, the first
    calibration of every frame) came back from the C++ conversion 1.2e-14 off and was refused at the former 1e-14."""
    R = np.array([[0.053715160341603334, -0.20084687981247923, 0.9781488702743903],
                  [0.06691431986580397, -0.9766372411229803, -0.20421109922963687],
                  [0.9963117761756936, 0.0764213983205725, -0.03902069364268137]])
    cam = pm.Camera(R, [-1205.3832328455674, -67.82844825984218, 68.90381958980866], 21532.829558676105, 1023.5,
                    975.5, 2048, 1952, [-1.0966510309524398, 0.0, 0.0, 0.0, 0.0])
    win = pm.Window([0.9981182067080382, 2.862293735361731e-17, -0.06131920937137082], -958.199937622995, (12.7,),
                    (1.0, 1.49, 1.333))
    pp = pm.to_cpp(cam, win)._pinplate_param
    R_cpp = np.array([[pp.r_mtx[i, j] for j in range(3)] for i in range(3)])
    assert np.max(np.abs(R_cpp - R)) < 1e-12


def test_affine_start_rejection_in_last_round():
    """Stage 1 (cylinder_wand.affine_factorization) with gross errors still being rejected in its last round: the
    frames it returns must be those it factorised (real detections of case plate_20260727_T0 crashed it)."""
    rng = np.random.default_rng(11)
    F, Nc, L = 400, 4, 10.0
    M = rng.normal(size=(Nc, 2, 3)) * 14.0
    t = rng.uniform(500.0, 1500.0, size=(Nc, 2))
    m = rng.uniform(-40.0, 40.0, size=(F, 3))
    w = rng.normal(size=(F, 3))
    w /= np.linalg.norm(w, axis=1)[:, None]
    X = np.stack([m - 0.5 * L * w, m + 0.5 * L * w], axis=1)
    uv = np.einsum("cij,fej->fcei", M, X) + t[None, :, None, :] + rng.normal(scale=0.2, size=(F, Nc, 2, 2))
    bad = rng.choice(F, 20, replace=False)
    uv[bad, rng.integers(0, Nc, 20)] += rng.normal(scale=30.0, size=(20, 2, 2))
    obs = pw.observations(pw.WandData(uv, L, [(2048, 2048)] * Nc))
    for rounds in (1, 2, 5):
        aff = pw.cw.affine_factorization(obs, L, rounds=rounds)
        assert len(aff.frames) == len(aff.XA) == len(aff.XB) > F // 2
        assert np.all(np.isfinite(aff.XA)) and np.all(np.isfinite(aff.XB))


def test_lens_terms():
    """L1: rig B with every camera's principal point moved 20-40 px and k2, p1, p2 set, noiseless (the C++ cameras'
    images).  Started from the truth with those terms at their defaults (principal point at the image centre, k2 = p1 =
    p2 = 0), the fit with fit_lens recovers them; without fit_lens it cannot fit the data.  The layout without fit_lens
    is the former one, and the sigmas name the new terms."""
    rig_t = prigs.rig_B()
    rng = np.random.default_rng(31)
    for cam in rig_t.cams:
        cam.cx += rng.choice([-1, 1]) * rng.uniform(20, 40)
        cam.cy += rng.choice([-1, 1]) * rng.uniform(20, 40)
        cam.dist = np.array([0.05, -0.05, 3e-4, -2e-4, 0.0])
    wand = prigs.make_wand(rig_t, 400, np.random.default_rng(32), noise_px=0.0)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    cams0 = []
    for cam in rig_t.cams:
        k = cam.copy()
        k.cx, k.cy, k.dist = (k.width - 1) / 2.0, (k.height - 1) / 2.0, np.array([0.05, 0.0, 0.0, 0.0, 0.0])
        cams0.append(k)
    rig0 = pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win)
    assert pw.Layout(4, 2, fit_k1=True).cols == list(pw.CAM_COLS) + ["k1"]
    with pytest.raises(ValueError):
        pw.Layout(4, 2, fit_lens=("k3",))
    plain = pw.solve(data, cams0, rig0, fit_k1=True, noise_weights="none")
    full = pw.solve(data, cams0, rig0, fit_k1=True, fit_lens=("pp", "k2", "tangential"), noise_weights="none")
    assert full.layout.cols[-5:] == ["cx", "cy", "k2", "p1", "p2"]
    assert plain.rms > 0.05 and full.rms < 1e-5, (plain.rms, full.rms)
    for c, t in zip(full.cams, rig_t.cams):
        assert abs(c.cx - t.cx) < 1e-3 and abs(c.cy - t.cy) < 1e-3
        assert np.max(np.abs(c.dist[:4] - t.dist[:4]) / np.array([0.05, 0.05, 3e-4, 2e-4])) < 1e-3
    sig = pw.camera_sigmas(full.cams, full.rig, full.layout, full.cov)
    assert all(set(s) >= {"cx_px", "cy_px", "k1", "k2", "p1", "p2"} for s in sig)


def _channel_case():
    """Rig B's truth (its wand ends 100-200 mm below both walls' water side) as a Solution with a channel layout."""
    sol = _truth_solution(prigs.rig_B(), 300, 41)
    lay = pw.Layout(4, 2, fit_k1=True, anchors=pw.anchors_for(sol.cams, sol.rig))
    return sol, lay


def test_channel_rows_jacobian():
    """L2: the channel's residual rows (a channel narrower than the wand's spread, so that many ends breach both a far
    wall and a near one) against central differences, over every free parameter and over the frames."""
    sol, lay = _channel_case()
    pw.set_constraints(lay, {"channel": {"width_mm": [150.0, 170.0], "tolerance_mm": 2.0, "ball_radius_mm": [1.0, 1.5],
                                         "sigma_mm": 0.5}})
    rig = pw.Rig([pm.Window(w.n, w.d + dd, w.t, w.idx) for w, dd in zip(sol.rig.windows, (-25.0, 105.0))],
                 sol.rig.cam_win)
    r, fr, Jg, Jf = pw.channel_terms(rig, sol.frames, sol.L, sol.obs, lay)
    assert len(r) > 50 and np.all(r > 0)
    for k in range(lay.P):
        h = 1e-6 if lay.names[k].endswith((".a", ".b")) else 1e-4
        dx = np.zeros(lay.P)
        dx[k] = h
        _cp, rp, _ = lay.apply(sol.cams, rig, sol.L, dx)
        _cn, rn, _ = lay.apply(sol.cams, rig, sol.L, -dx)
        a = pw.channel_terms(rp, sol.frames, sol.L, sol.obs, lay, jac=False)[0]
        b = pw.channel_terms(rn, sol.frames, sol.L, sol.obs, lay, jac=False)[0]
        assert len(a) == len(b) == len(r)
        assert np.max(np.abs((a - b) / (2 * h) - Jg[:, k])) <= 1e-6 * max(np.max(np.abs(Jg[:, k])), 1.0), lay.names[k]
    for j in range(5):
        h = 1e-6
        dfp = np.zeros((sol.obs.n_frames, 5))
        dfp[:, j] = h
        a = pw.channel_terms(rig, sol.frames.retract(dfp), sol.L, sol.obs, lay, jac=False)[0]
        b = pw.channel_terms(rig, sol.frames.retract(-dfp), sol.L, sol.obs, lay, jac=False)[0]
        assert np.max(np.abs((a - b) / (2 * h) - Jf[:, j])) <= 1e-5 * max(np.max(np.abs(Jf[:, j])), 1.0), j


def test_channel_and_aim_constraints():
    """L2 on rig B (noise 0.1 px): a channel the truth satisfies leaves the fit unchanged; a channel 30 mm narrower
    than the wand's place across wall 2 needs moves the wall to the band's edge (every end inside, some on the far
    wall), and the report gives the band; an aim held 5 mm off the truth is met; into_channel moves a window only as
    far as needed."""
    rig_t = prigs.rig_B()
    wand = prigs.make_wand(rig_t, 300, np.random.default_rng(42), noise_px=0.1)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)

    def fit(cons=None):
        return pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win),
                        fit_k1=True, constraints=cons)

    free = fit()
    loose = fit({"channel": {"width_mm": [230.0, 230.0], "ball_radius_mm": 1.0}})
    assert abs(loose.rms - free.rms) < 1e-9
    assert np.max(np.abs(loose.rig.windows[1].d - free.rig.windows[1].d)) < 1e-6
    d2 = loose.rig.windows[1].depth(np.vstack(loose.frames.ends(loose.L)))
    tight = fit({"channel": {"width_mm": [230.0, float(d2.max()) - 30.0], "ball_radius_mm": 1.0}})
    rep = pw.constraint_report(tight)
    w2 = rep["channel"]["windows"][1]
    assert w2["depth_mm"][1] <= rep["channel"]["width"][1] - 1.0 + 0.01      # every end inside the far wall ...
    assert rep["channel"]["active_rows"] >= 1 and abs(w2["band_mm"][0]) < 0.05  # ... some on it: the band's edge
    assert tight.rms > free.rms
    aim0 = pw.aim_depth(free.cams, free.rig, [0, 1], 1)
    aimed = fit({"aim": [{"cams": [0, 1], "wall": 1, "mm": aim0 + 5.0}]})
    assert abs(pw.aim_depth(aimed.cams, aimed.rig, [0, 1], 1) - (aim0 + 5.0)) < 0.05
    lay = pw.Layout(4, 2)
    pw.set_constraints(lay, {"channel": {"width_mm": [None, float(d2.max()) - 30.0], "ball_radius_mm": 1.0}})
    moved = pw.into_channel(free.rig, free.frames, free.L, free.obs, lay)
    assert moved.windows[0].d == free.rig.windows[0].d
    assert abs((moved.windows[1].d - free.rig.windows[1].d) - 31.0) < 0.5


@pytest.mark.slow
def test_scan_fixes_a_weak_window():
    """W5 seed 1001 (one camera per window; window 2's of 24,500 px sees the wand under 1.4 deg), without hints: the
    start ends 11 deg off in window 2's normal; the scan of that normal finds the global minimum (D8, H4)."""
    from h1_hints import case
    rig_t, data = case(1001, 0.1)
    sol = pw.calibrate(data, pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                                    rig_t.cam_win))
    assert sol.info["scan"]["improved"] and [s["window"] for s in sol.info["scan"]["windows"]] == [1]
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    cmp = prigs.same_minimum(sol, ref)
    assert cmp["same"] or sol.rms < ref.rms or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms), cmp


@pytest.mark.slow
def test_restarts_free_a_pinned_answer():
    """W5 seed 2000 (two windows, two cameras each, 1 px): its start ends pinned against window 1, a wand end on its
    water-side surface; the perturbed restarts reach the global minimum."""
    rng = np.random.default_rng(2000)
    rig_t = prigs.random_rig(rng)
    wand = prigs.make_wand(rig_t, int(rng.integers(500, 1501)), rng, noise_px=1.0)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    sol = pw.calibrate(data, pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                                    rig_t.cam_win))
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    assert sol.info["starts"]["perturbed"]["improved"]
    cmp = prigs.same_minimum(sol, ref)
    assert cmp["same"], cmp


@pytest.mark.slow
def test_hint_paths():
    """Window-normal hints 10 deg off on rig B (two windows, two cameras each): the hint-held and hint-searched starts
    are built and fitted beside the default one, the answer is the global minimum, and each hint's angle from it is
    reported (about 10 deg)."""
    rig_t = prigs.rig_B()
    wand = prigs.make_wand(rig_t, 400, np.random.default_rng(21), noise_px=0.1)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    rng = np.random.default_rng(5)
    normals = []
    for c, cam in enumerate(rig_t.cams):
        v = cam.R @ rig_t.win_of(c).n
        e = np.cross(v, rng.normal(size=3))
        e /= np.linalg.norm(e)
        normals.append((math.cos(math.radians(10.0)) * v + math.sin(math.radians(10.0)) * e).tolist())
    sol = pw.calibrate(data, pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                                    rig_t.cam_win), hints={"normals": normals})
    hi = sol.info["hints"]
    assert set(hi["paths"]) == {"held", "searched"} and hi["chosen"] in ("default", "held", "searched"), hi
    for p in hi["paths"].values():
        assert "error" not in p and p["tried"] >= 1 and p["rms_px"] > 0, p
    assert all(abs(a - 10.0) < 1.0 for a in hi["normals_off_deg"])
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    cmp = prigs.same_minimum(sol, ref)
    assert cmp["same"], cmp


@pytest.mark.slow
def test_from_scratch_reaches_the_global_minimum():
    rig_t = prigs.rig_D()
    wand = prigs.make_wand(rig_t, 600, np.random.default_rng(7), noise_px=0.1)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    sol = pw.calibrate(data, pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx) for w in rig_t.wins], rig_t.cam_win))
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    cmp = prigs.same_minimum(sol, ref)
    assert cmp["same"], cmp
