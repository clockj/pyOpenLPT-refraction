"""pytest subset of the cylinder STB work (tests/cylinder_stb/; plan cases/cyl_20261007/2026-10-07-plan.md).

The campaign's full tests (F1-F5, E1-E4, V1-V4, L5, the synthetic end-to-end runs) are in tests/cylinder_stb/; these
are fast checks of the same properties on cameras built here, skipped on a build without the new code:
  - projection hints (Phase 1): the hinted projection equals project(), from a nearby point's hint and from a line
    of sight; the hint takes the fast path;
  - the single-image certificate and the per-camera image statuses (Phase 2) on a camera that sees the whole tube;
  - the curved image lines of StereoMatch (Phase 3);
  - the OTF's segment model (Phase 2/4): its peak, its integral, its extent;
  - the line detector (Phase 4) on a rendered line.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

lpt = pytest.importorskip("pyopenlpt")
pytestmark = pytest.mark.skipif(not hasattr(lpt, "ProjectionHint"),
                                reason="this build of pyopenlpt has no projection hints (cylinder STB work)")

N_IDX = [1.0, 1.49, 1.333]
RADII = [100.80625, 97.63125]


def _camera(azimuth_deg, dist, f, size=(800, 1280)):
    """A PINCYL camera on a circle about the world y axis (the tube's axis), looking at the axis."""
    import cv2
    cam = lpt.RefractionCylinderCameraModel()
    cam.setImageSize(*size)
    cam.setIntrinsics(f, f, (size[1] - 1) / 2, (size[0] - 1) / 2, [0.0] * 5)
    a = math.radians(azimuth_deg)
    C = np.array([dist * math.sin(a), 0.0, dist * math.cos(a)])
    z_c = -C / np.linalg.norm(C)
    y_c = np.array([0.0, -1.0, 0.0])
    x_c = np.cross(y_c, z_c)
    R = np.vstack([x_c, y_c, z_c])
    rvec, _ = cv2.Rodrigues(R)
    cam.setExtrinsics(lpt.Pt3D(*map(float, rvec.ravel())), lpt.Pt3D(*map(float, -R @ C)))
    cam.setRefraction(lpt.Pt3D(0.0, 0.0, 0.0), lpt.Pt3D(0.0, 1.0, 0.0), N_IDX, RADII)
    cam.setSolverOptions(1e-9, 80)
    cam.commitUpdate()
    return cam


@pytest.fixture(scope="module")
def rig():
    """Two long-lens cameras as the real rig (about 1 m from the axis, 60 deg apart)."""
    return [_camera(-30.0, 1000.0, 38000.0), _camera(30.0, 1000.0, 38000.0)]


@pytest.fixture(scope="module")
def wide():
    """A camera that sees the whole tube, shadow and double images included."""
    return _camera(0.0, 400.0, 1000.0)


def _points(rng, n, half=(8.0, 6.0, 8.0)):
    return rng.uniform(-np.array(half), np.array(half), (n, 3))


def _pt(x):
    return lpt.Pt3D(*map(float, x))


def test_hinted_projection_equals_project(rig):
    rng = np.random.default_rng(1)
    for cam in rig:
        assert cam.usesProjectionHint()
        for X in _points(rng, 150):
            ok, _p, hint, _e = cam.projectWithHint(_pt(X), lpt.ProjectionHint())
            assert ok and hint.valid
            d = rng.normal(size=3)
            Y = X + 0.05 * d / np.linalg.norm(d)
            slow = cam.project(_pt(Y))
            ok, fast, _h, _e = cam.projectWithHint(_pt(Y), hint)
            assert ok
            assert math.hypot(slow[0] - fast[0], slow[1] - fast[1]) < 1e-6


def test_hint_from_line_of_sight(rig):
    rng = np.random.default_rng(2)
    for cam in rig:
        for X in _points(rng, 100):
            P = _pt(X)
            pix = cam.project(P)
            ok, hint = cam.hintFromLineOfSight(cam.lineOfSight(pix), P)
            assert ok and hint.valid
            ok, fast, _h, _e = cam.projectWithHint(P, hint)
            assert ok and math.hypot(fast[0] - pix[0], fast[1] - pix[1]) < 1e-6


def test_certificate_and_statuses(wide):
    rng = np.random.default_rng(3)
    W, H = wide.getNCol(), wide.getNRow()
    seen = {"none": 0, "one": 0, "ambiguous": 0}
    for _ in range(400):
        r = RADII[1] * math.sqrt(rng.random()) * (1 - 1e-6)
        phi = 2 * math.pi * rng.random()
        X = np.array([r * math.cos(phi), rng.uniform(-40, 40), r * math.sin(phi)])
        P = _pt(X)
        rays, _cov = wide.projectAll(P)
        if wide.provablySingleImage(P):
            assert len(rays) == 1
        st, _one = wide.imageStatus(P, 2.0, 8.0)
        seen[st] += 1
        on = [d for d in rays if -2 <= d["pt"][0] < W + 2 and -2 <= d["pt"][1] < H + 2]
        if not on:
            assert st == "none"
        elif len(on) > 1:
            assert st == "ambiguous"
    assert seen["one"] > 0 and seen["none"] + seen["ambiguous"] > 0   # the band beyond r_c is in view


def test_segment_otf_intensity():
    # a horizontal line of length 20 px, sigma 1.2: peak a at the centre, integral a w(20) w(0)
    a, L, s = 50.0, 20.0, 1.2
    p = [a, L, 0.0, 1.0, 0.0, s]
    xs = np.arange(-40, 41)
    ys = np.arange(-10, 11)
    img = np.array([[lpt.OTF.intensity(1, x, y, 0.0, 0.0, p) for x in xs] for y in ys])
    assert img[10, 40] == pytest.approx(a, rel=1e-12)
    w = lambda L_: L_ / math.erf(L_ / (2 * math.sqrt(2) * s)) if L_ > 0 else s * math.sqrt(2 * math.pi)
    assert img.sum() == pytest.approx(a * w(L) * w(0.0), rel=1e-3)
    dx, dy = lpt.OTF.halfExtent(1, p, 3.0)
    assert dx == pytest.approx(L / 2 + 3 * s) and dy == pytest.approx(3 * s)


def _stereo_match(tmp_path, cams, half=10.0):
    """A StereoMatch on the cameras, through a config.json in tmp_path (blank images: only the geometry is used)."""
    import json
    import cv2
    files, lists = [], []
    for k, cam in enumerate(cams):
        f = tmp_path / f"cam{k + 1}.txt"
        cam.saveParameters(str(f))
        img = tmp_path / f"cam{k + 1}_0.tif"
        cv2.imwrite(str(img), np.zeros((cam.getNRow(), cam.getNCol()), np.uint8))
        lst = tmp_path / f"cam{k + 1}ImageNames.txt"
        lst.write_text(f"{img}\n" * 5)                 # the OTF estimate reads 5 frames
        files.append(str(f))
        lists.append(str(lst))
    cfg = {"frames": [0, 0], "fps": 1000, "threads": 1,
           "cameras": [{"file": f, "max_intensity": 255} for f in files], "image_lists": lists,
           "view_volume_mm": {"x": [-half, half], "y": [-half, half], "z": [-half, half]}, "voxel_mm": 0.03,
           "output_dir": str(tmp_path / "Results"),
           "objects": [{"type": "Tracer", "initial_phase": {"search_radius_vox": 10.0, "frames": 4},
                        "convergence_phase": {"avg_spacing_vox": 30.0},
                        "shake": {"width_vox": 0.25, "loops": 4, "ghost_threshold": 0.01},
                        "predictive_field": {"grid": [51, 51, 51], "search_radius_vox": 25.0},
                        "ipr": {"loops": 1, "reduced_cameras": 0, "reduced_loops": 0, "tol_2d_px": 1.0,
                                "tol_3d_vox": 2.0},
                        "tracer": {"intensity_threshold": 30, "radius_px": 2.0}}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    bs = lpt.BasicSetting()
    bs.readConfig(str(path))
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams_ = list(bs._cam_list)
    return lpt.StereoMatch(cams_, [[] for _ in cams_], tc), cams_


def test_curved_image_line(rig, tmp_path):
    # a point's image in cam2 lies on the curved image of cam1's line of sight through it; the straight chord can miss
    if not hasattr(lpt.StereoMatch, "makeCurve2DFromLOS3D"):
        pytest.skip("no curved image lines in this build")
    sm, cams = _stereo_match(tmp_path, rig)
    rng = np.random.default_rng(4)
    worst_curve, worst_chord = 0.0, 0.0
    for X in _points(rng, 60):
        P = _pt(X)
        los = cams[0].lineOfSight(cams[0].project(P))
        ok, chord, poly = sm.makeCurve2DFromLOS3D(1, los)
        assert ok and len(poly) >= 2
        q = cams[1].project(P)
        worst_curve = max(worst_curve, math.sqrt(lpt.dist2ToPolyline(q, poly)))
        worst_chord = max(worst_chord, math.sqrt(lpt.dist2ToPolyline(q, [poly[0], poly[-1]])))
    assert worst_curve < 0.05
    assert worst_chord > worst_curve


def test_line_detector_isolated():
    # isolated line images (the segment model, on 8 bits with noise): centres and box lengths recovered
    if not hasattr(lpt, "detectLines"):
        pytest.skip("no line detector in this build")
    rng = np.random.default_rng(5)
    H, W, s = 120, 240, 1.2
    truth = [(60.3, 40.6, 14.0, 0.0), (150.7, 70.2, 0.0, 9.0), (200.2, 30.9, 0.0, 0.0)]   # u, v, Lx, Ly
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W))
    for u, v, Lx, Ly in truth:
        p = [120.0, Lx, Ly, 1.0, 0.0, s]
        win = (np.abs(xx - u) <= Lx / 2 + 5 * s) & (np.abs(yy - v) <= Ly / 2 + 5 * s)
        img[win] += [lpt.OTF.intensity(1, float(x), float(y), u, v, p) for x, y in zip(xx[win], yy[win])]
    img = np.clip(np.rint(img + rng.normal(0.0, 1.0, img.shape)), 0, 255).astype(np.uint8)   # background 0, as S1's
    det = np.array(lpt.detectLines(lpt.Image.from_numpy(img), 1.0, lpt.LineFinderParam()))
    assert len(det) == len(truth)
    for u, v, Lx, Ly in truth:
        k = int(np.argmin(np.hypot(det[:, 0] - u, det[:, 1] - v)))
        assert math.hypot(det[k, 0] - u, det[k, 1] - v) < 0.05
        assert abs(math.sqrt(det[k, 2] ** 2 / 12 + det[k, 4] ** 2) - math.sqrt(Lx ** 2 / 12 + s ** 2)) < 0.1
        assert abs(math.sqrt(det[k, 3] ** 2 / 12 + det[k, 5] ** 2) - math.sqrt(Ly ** 2 / 12 + s ** 2)) < 0.1
