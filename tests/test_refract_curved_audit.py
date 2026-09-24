"""Regression tests for the bugs found by the audit of the curved-port projection (2026-09-24).

Each test names the audit item it guards (the notes repository's
reports/2026-09-24-pyopenlpt-projection-audit.md):

- A1: the projection's tolerances act on dimensionless numbers, so a rig gives
  the same rays in metres and in millimetres, and rays next to grazing are not
  lost in millimetres.
- A3: next to a fold the box search matches the reference (double-double
  residual), rather than overrunning its cap and returning copies of one ray.
- A4: an azimuth a few ulp below pi gives the reference's rays.
- B1-B4: inputs outside the model are reported as such, never as "no image".
- B5: a strict proj_tol is met rather than failed.
- B6, C2: distortion coefficients survive a reload and short vectors are padded.
- B7: matrix indexing is bounds-checked.
- B8: synthetic_ports.look_at gives image x right and y down.
- B9: the loader rejects non-positive indices and radii.

The follow-up of the same day (the notes repository's code/tests/openlpt_followup/):

- C3: project() returns an image on the sensor when the preferred one is off it.
- C5: every ring on a sphere's axial line, not only the first.
- D: the ring's pixel is NaN; an invisible point is reported as such; solveFar
  counts the iterations it took; camera files round-trip exactly; the on-axis
  branch label is the limit of the general case; the reference's azimuth is the
  C++ remainder; the radial action is finite at Lambda = 0 on the far branch.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFS = REPO_ROOT / "test" / "refs"
if str(REFS) not in sys.path:
    sys.path.insert(0, str(REFS))

import project_cylinder as pc  # noqa: E402
import project_sphere as ps  # noqa: E402

cv2 = pytest.importorskip("cv2")
lpt = pytest.importorskip("pyopenlpt", reason="extension not built")
rc = pytest.importorskip("pyopenlpt.refract_curved", reason="refract_curved bindings not built")

REF_N = [1.0, 1.49, 1.33]
REF_R_M = [0.100, 0.094]
REF_RP_M = 0.350


# ----------------------------------------------------------------- helpers
def _stacks(seed, count):
    """The reference port, then random admissible stacks (in metres)."""
    yield REF_N, REF_R_M, REF_RP_M
    rng = np.random.default_rng(seed)
    for _ in range(count - 1):
        k = int(rng.integers(2, 5))
        n = [1.0] + list(rng.uniform(1.05, 2.0, k))
        radii = sorted(rng.uniform(0.03, 0.12, k), reverse=True)
        yield n, [float(r) for r in radii], float(rng.uniform(radii[0] * 1.2, 0.6))


def _grazing_point(rng, n, R, rP, far, lo=-12.0, hi=-6.0, c_z=None):
    """A point reached by a ray with 1 - Lambda/Lmax log-uniform in [10^lo, 10^hi]."""
    while True:
        C = float(rng.uniform(-0.9, 0.9) * min(n)) if c_z is None else c_z
        rX = float(rng.uniform(0.3, 0.999) * R[-1])
        L = pc.Lmax_kstar(n, R, rX, abs(C))[0] * (1 - 10 ** rng.uniform(lo, hi))
        pz = pc.Phi_Z(L, abs(C), n, R, rP, rX, far)
        if pz is not None and pz[0] < math.pi:
            return np.array([rX * math.cos(pz[0]), rX * math.sin(pz[0]), math.copysign(pz[1], C)])


def _cyl(n, R, rP, X, s, impl):
    Rs, Ps, Xs = [r * s for r in R], [rP * s, 0.0, 0.0], [float(v) * s for v in X]
    if impl == "C++":
        rays = [(b, L, C) for b, L, C, _ in rc.project_cylinder(n, Rs, Ps, Xs)[0]]
    else:
        rays = [(b, L, C) for b, L, C, _ in pc.project(n, Rs, np.array(Ps), np.array(Xs))]
    return sorted((b, L / s, C) for b, L, C in rays)


def _sph(n, R, rP, X, s, impl):
    Rs, Ps, Xs = [r * s for r in R], [rP * s, 0.0, 0.0], [float(v) * s for v in X]
    if impl == "C++":
        rays = rc.project_sphere(n, Rs, [0.0, 0.0, 0.0], Ps, Xs)
    else:
        rays = ps.project(n, Rs, np.zeros(3), np.array(Ps), np.array(Xs))
    return sorted((b, L / s) for b, L, _ in rays)


def _same(a, b, lam_tol=1e-9):
    assert len(a) == len(b), f"{len(a)} rays against {len(b)}: {a} / {b}"
    for x, y in zip(a, b):
        assert x[0] == y[0], (x, y)
        assert abs(x[1] - y[1]) <= lam_tol * max(abs(x[1]), 1e-3), (x, y)
        if len(x) > 2:
            assert abs(x[2] - y[2]) <= 1e-9, (x, y)


def _look_at_cv(center, target, up=(0.0, 0.0, 1.0)):
    """OpenCV convention (x right, y down, z forward): (rvec, tvec)."""
    center = np.asarray(center, float)
    z = np.asarray(target, float) - center
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rmtx = np.vstack([x, y, z])
    return cv2.Rodrigues(rmtx)[0].ravel(), -rmtx @ center


def _camera(kind="cylinder", center=(350.0, 0.0, 0.0), target=(0.0, 0.0, 0.0),
            n=REF_N, radii=(100.0, 94.0), dist=(0.0, 0.0, 0.0, 0.0, 0.0), tol=1e-10, nmax=100):
    cam = lpt.RefractionSphereCameraModel() if kind == "sphere" else lpt.RefractionCylinderCameraModel()
    cam.setImageSize(1024, 1280)
    cam.setIntrinsics(1800.0, 1800.0, 640.0, 512.0, list(dist))
    rvec, tvec = _look_at_cv(center, target)
    cam.setExtrinsics(lpt.Pt3D(*rvec), lpt.Pt3D(*tvec))
    cam.setRefraction(lpt.Pt3D(0.0, 0.0, 0.0), lpt.Pt3D(0.0, 0.0, 1.0), list(n), list(radii))
    cam.setSolverOptions(tol, nmax)
    cam.commitUpdate()
    return cam


# ---------------------------------------------------------------- A1: units
@pytest.mark.parametrize("impl", ["C++", "Python"])
def test_a1_metres_and_mm_give_the_same_rays(impl):
    rng = np.random.default_rng(31)
    for n, R, rP in _stacks(7, 3):
        pts = [_grazing_point(rng, n, R, rP, bool(i % 2)) for i in range(12)]
        for _ in range(12):
            r, a = R[-1] * math.sqrt(rng.uniform(0.0, 0.999)), rng.uniform(-math.pi, math.pi)
            pts.append(np.array([r * math.cos(a), r * math.sin(a), rng.uniform(-2, 2) * R[0]]))
        for X in pts:
            _same(_cyl(n, R, rP, X, 1.0, impl), _cyl(n, R, rP, X, 1e3, impl))
        # The sphere: in its ray plane it is the cylinder at C_z = 0.
        for i in range(24):
            if i < 12:
                Xs = _grazing_point(rng, n, R, rP, bool(i % 2), c_z=0.0)
            else:
                v = rng.normal(size=3)
                Xs = v / np.linalg.norm(v) * R[-1] * rng.uniform(0.0, 0.999) ** (1 / 3)
            _same(_sph(n, R, rP, Xs, 1.0, impl), _sph(n, R, rP, Xs, 1e3, impl))


@pytest.mark.parametrize("far", [False, True])
def test_a1_near_grazing_rays_are_found_in_mm(far):
    """Rays within 1e-13..1e-9 of grazing: every one is returned, in mm, by both implementations."""
    rng = np.random.default_rng(32 + far)
    n, R, rP = REF_N, [100.0, 94.0], 350.0
    for _ in range(25):
        X = _grazing_point(rng, n, R, rP, far, lo=-13.0, hi=-9.0)
        for impl in ("C++", "Python"):
            rays = (rc.project_cylinder(n, R, [rP, 0.0, 0.0], list(X))[0] if impl == "C++"
                    else pc.project(n, R, np.array([rP, 0.0, 0.0]), X))
            misses = []
            for b, _L, _C, d in rays:
                if (b == "far") == far:
                    Xh = pc.trace_to_radius(np.array([rP, 0.0, 0.0]), np.asarray(d, float), n, R,
                                            np.hypot(X[0], X[1]), far)
                    misses.append(np.inf if Xh is None else float(np.linalg.norm(Xh - X)))
            assert misses and min(misses) < 1e-6, f"{impl}: no {'far' if far else 'near'} ray to {X}"


# ------------------------------------------------------------- A3: the folds
FOLD_POINTS = [  # (r_X, azimuth, dz) in mm, the reference port; from the audit's r06
    (85.8475, -2.3273, 0.5294),
    (75.7379, -2.0433, 0.8014),
    (81.2850, -2.2371, 1.0579),
    (77.5388, -2.1292, -0.4945),
    (77.48887327009759, -2.1273951118472776, 2.198658456020221),
]


@pytest.mark.parametrize("pt", FOLD_POINTS)
def test_a3_fold_points_match_the_reference(pt):
    rX, az, dz = pt
    n, R, rP = REF_N, [100.0, 94.0], 350.0
    stats = {}
    py_seeds = pc.classify_far(n, R, rP, rX, abs(az), dz, stats=stats)
    cpp_seeds, cst = rc.classify_far(n, R, rP, rX, abs(az), dz)
    assert not cst["capped"]
    assert cst["boxes"] == stats["boxes"] and cst["precise"] == stats["precise"]
    assert len(cpp_seeds) == len(py_seeds)
    X = [rX * math.cos(az), rX * math.sin(az), dz]
    rays, cov = rc.project_cylinder(n, R, [rP, 0.0, 0.0], X)
    assert cov == {"near_branch": True, "far_branch": True}
    assert len(rays) == len(pc.project(n, R, np.array([rP, 0.0, 0.0]), np.array(X)))


def test_a3_a_capped_search_is_reported():
    """The fold search needs 1163 boxes; with a cap of 50 the stats must say it overran."""
    rX, az, dz = FOLD_POINTS[-1]
    args = (REF_N, [100.0, 94.0], 350.0, rX, abs(az), dz)
    _seeds, st = rc.classify_far(*args)
    assert st["boxes"] > 50 and st["capped"] is False
    _seeds, st = rc.classify_far(*args, maxboxes=50)
    assert st["capped"] is True


# ---------------------------------------------------------- A4: just below pi
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 6, 8, 13, 20, 40])
def test_a4_azimuths_just_below_pi(k):
    n, R, rP = REF_N, [100.0, 94.0], 350.0
    az = math.pi - k * 2.220446049250313e-16
    X = np.array([70.0 * math.cos(az), 70.0 * math.sin(az), 60.0])
    cpp = rc.project_cylinder(n, R, [rP, 0.0, 0.0], list(X))[0]
    py = pc.project(n, R, np.array([rP, 0.0, 0.0]), X)
    assert sorted(b for b, *_ in cpp) == sorted(b for b, *_ in py)
    for (b1, L1, C1, _), (b2, L2, C2, _) in zip(sorted(cpp, key=lambda r: (r[0], r[1])),
                                                sorted(py, key=lambda r: (r[0], r[1]))):
        assert abs(L1 - L2) <= 1e-9 * 100.0 and abs(C1 - C2) <= 1e-9


# ------------------------------------------------------ B1-B4: out of model
@pytest.mark.parametrize("rX", [97.0, 94.0, 150.0])
def test_b1_points_outside_the_innermost_medium_are_not_certified(rX):
    n, R, rP = REF_N, [100.0, 94.0], 350.0
    X = [rX * math.cos(0.5236), rX * math.sin(0.5236), 10.0]
    rays, cov = rc.project_cylinder(n, R, [rP, 0.0, 0.0], X)
    assert rays == [] and cov == {"near_branch": False, "far_branch": False}
    Xs = [rX * math.cos(0.5236), rX * math.sin(0.5236), 0.0]
    assert rc.project_sphere(n, R, [0.0, 0.0, 0.0], [rP, 0.0, 0.0], Xs) == []
    for kind, P in (("cylinder", X), ("sphere", Xs)):
        rays, cov = _camera(kind).projectAll(lpt.Pt3D(*P))
        assert rays == [] and cov == {"near_branch": False, "far_branch": False}
    # just inside, both see it
    Xin = [93.999 * math.cos(0.5236), 93.999 * math.sin(0.5236), 10.0]
    rays, cov = rc.project_cylinder(n, R, [rP, 0.0, 0.0], Xin)
    assert rays and cov == {"near_branch": True, "far_branch": True}


@pytest.mark.parametrize("x", [200.0, 97.0, 500.0])
def test_b2_sphere_axial_points_outside_get_no_ray(x):
    assert rc.project_sphere(REF_N, [100.0, 94.0], [0.0, 0.0, 0.0], [350.0, 0.0, 0.0], [x, 0.0, 0.0]) == []


@pytest.mark.parametrize("r_p", [97.0, 60.0, 100.0])
def test_b3_pinhole_inside_the_port_is_an_error(r_p):
    n, R = REF_N, [100.0, 94.0]
    X = [50.0, 20.0, 10.0]
    rays, cov = rc.project_cylinder(n, R, [r_p, 0.0, 0.0], X)
    assert rays == [] and cov == {"near_branch": False, "far_branch": False}
    assert rc.project_sphere(n, R, [0.0, 0.0, 0.0], [r_p, 0.0, 0.0], X) == []
    if r_p == R[0]:
        return  # a camera rebuilds its pinhole from its pose, within rounding of R_1 on either side
    for kind in ("cylinder", "sphere"):
        cam = _camera(kind, center=(r_p, 0.0, 0.0))
        with pytest.raises(RuntimeError, match="pinhole"):
            cam.projectAll(lpt.Pt3D(*X))
        with pytest.raises(RuntimeError, match="pinhole"):
            cam.lineOfSight(lpt.Pt2D(640.0, 512.0))


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_b4_a_camera_facing_away_sees_nothing(kind):
    away = _camera(kind, target=(700.0, 0.0, 0.0))
    toward = _camera(kind)
    for X in ([50.0, 10.0, 20.0], [0.0, 0.0, 0.0], [-40.0, 30.0, -10.0]):
        assert away.projectAll(lpt.Pt3D(*X))[0] == []
        assert len(toward.projectAll(lpt.Pt3D(*X))[0]) >= 1


# -------------------------------------------------------- B5: solver options
@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_b5_a_strict_proj_tol_keeps_every_image(kind):
    rng = np.random.default_rng(12)
    pts = []
    while len(pts) < 150:
        r, a = 94.0 * math.sqrt(rng.uniform()), rng.uniform(-math.pi, math.pi)
        pts.append([r * math.cos(a), r * math.sin(a), rng.uniform(-94.0, 94.0) if kind == "cylinder" else 0.0])
    counts = []
    for tol in (1e-9, 1e-12, 1e-13):
        cam = _camera(kind, tol=tol)
        counts.append(sum(len(cam.projectAll(lpt.Pt3D(*p))[0]) for p in pts))
    assert counts[0] > 0 and counts[1] == counts[0] and counts[2] == counts[0], counts


def test_b5_itmax_is_accepted():
    """proj_nmax reaches the solvers as itmax. The seeds of step (3) already meet the
    tolerance at generic points, so even itmax = 0 returns the same rays there."""
    n, R, rP = REF_N, [100.0, 94.0], 350.0
    X = [30.0, 40.0, 25.0]
    full = rc.project_cylinder(n, R, [rP, 0.0, 0.0], X)[0]
    for itmax in (0, 1, 5, 200):
        got = rc.project_cylinder(n, R, [rP, 0.0, 0.0], X, itmax=itmax)[0]
        assert [(b, round(L, 6)) for b, L, *_ in got] == [(b, round(L, 6)) for b, L, *_ in full]
    Xs = [30.0, 40.0, 0.0]
    full = rc.project_sphere(n, R, [0.0, 0.0, 0.0], [rP, 0.0, 0.0], Xs)
    got = rc.project_sphere(n, R, [0.0, 0.0, 0.0], [rP, 0.0, 0.0], Xs, itmax=200)
    assert [(b, round(L, 6)) for b, L, _ in got] == [(b, round(L, 6)) for b, L, _ in full]


# ------------------------------------------------------ B6, C2: distortion
def test_b6_negative_distortion_survives_a_reload(tmp_path):
    X = lpt.Pt3D(30.0, 40.0, 25.0)
    for dist in ([-0.2, 0.0, 0.0, 0.0, 0.0], [0.0, -0.05, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0]):
        cam = _camera(dist=dist)
        before = cam.projectAll(X)[0][0]["pt"]
        path = str(tmp_path / "cam.txt")
        cam.saveParameters(path)
        after = lpt.loadCameraModel(path).projectAll(X)[0][0]["pt"]
        assert math.hypot(before[0] - after[0], before[1] - after[1]) < 1e-3, dist


def test_c2_short_distortion_vectors_are_padded():
    X = lpt.Pt3D(30.0, 40.0, 25.0)
    ref = _camera(dist=[0.1, 0.0, 0.0, 0.0]).projectAll(X)[0][0]["pt"]
    for dist in ([0.1], [0.1, 0.0], [0.1, 0.0, 0.0]):
        got = _camera(dist=dist).projectAll(X)[0][0]["pt"]
        assert (got[0], got[1]) == pytest.approx((ref[0], ref[1]), abs=1e-9)
    ref8 = _camera(dist=[0.1, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0]).projectAll(X)[0][0]["pt"]
    got8 = _camera(dist=[0.1, 0.0, 0.0, 0.0, 0.0, 0.01]).projectAll(X)[0][0]["pt"]
    assert (got8[0], got8[1]) == pytest.approx((ref8[0], ref8[1]), abs=1e-9)
    # OpenCV's 14 coefficients stay as given; distort() reads the first 12.
    ref14 = _camera(dist=[0.1] + [0.0] * 11).projectAll(X)[0][0]["pt"]
    got14 = _camera(dist=[0.1] + [0.0] * 13).projectAll(X)[0][0]["pt"]
    assert (got14[0], got14[1]) == pytest.approx((ref14[0], ref14[1]), abs=1e-9)


# ---------------------------------------------------------- B7: indexing
def test_b7_matrix_indexing_is_bounds_checked():
    assert list(lpt.Pt2D(1.0, 2.0)) == [1.0, 2.0]
    assert list(lpt.Pt3D(1.0, 2.0, 3.0)) == [1.0, 2.0, 3.0]
    p = lpt.Pt2D(1.0, 2.0)
    for bad in (2, -1, 100):
        with pytest.raises(IndexError):
            p[bad]
        with pytest.raises(IndexError):
            p[bad] = 0.0


# --------------------------------------------------------- B8: look_at
def test_b8_look_at_gives_x_right_and_y_down():
    import synthetic_ports as sp

    spec = sp.look_at([350.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    Rm = cv2.Rodrigues(np.asarray(spec.rvec, float))[0]
    assert np.linalg.det(Rm) == pytest.approx(1.0)
    # Looking along -x with +z up: right is +y and down is -z.
    assert Rm[0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert Rm[1] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)
    cam = sp.build_curved_camera(spec, sp.CurvedPort("cylinder"))
    centre = cam.projectAll(lpt.Pt3D(0.0, 0.0, 0.0))[0][0]["pt"]
    above = cam.projectAll(lpt.Pt3D(0.0, 0.0, 30.0))[0][0]["pt"]
    right = cam.projectAll(lpt.Pt3D(0.0, 30.0, 0.0))[0][0]["pt"]
    assert above[1] < centre[1] and right[0] > centre[0]


# ---------------------------------------------------------- B9: the loader
@pytest.mark.parametrize("which,new", [
    ("# Interface Radii", "1.0e+02,-9.4e+01"),
    ("# Interface Radii", "1.0e+02,0"),
    ("# Refractive Index Array", "1,1.49,-1.33"),
    ("# Refractive Index Array", "1,0,1.33"),
])
def test_b9_loader_rejects_nonpositive_indices_and_radii(tmp_path, which, new):
    path = tmp_path / "cyl.txt"
    _camera().saveParameters(str(path))
    lines = path.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(which))
    lines[i + 1] = new
    bad = tmp_path / "bad.txt"
    bad.write_text("\n".join(lines) + "\n")
    with pytest.raises(Exception):
        lpt.loadCameraModel(str(bad))
    assert lpt.loadCameraModel(str(path)) is not None


# ============================================================== follow-up
def _r21_points():
    """The audit's scan r21: 3000 points next to the wall, on the far side of the port."""
    rng = np.random.default_rng(29)
    out = []
    for _ in range(3000):
        r = rng.uniform(60.0, 93.5)
        a = rng.uniform(1.8, math.pi) * rng.choice([-1, 1])
        out.append((r * math.cos(a), r * math.sin(a), rng.uniform(-150.0, 150.0)))
    return out


# ----------------------------------------------- C3: the primary image
def test_c3_project_prefers_an_image_on_the_sensor():
    """Near branch first, then the smallest |Lambda|, among the images on the sensor."""
    cam = _camera()
    on = lambda r: 0.0 <= r["pt"][0] < 1280 and 0.0 <= r["pt"][1] < 1024
    best = lambda rays: min(rays, key=lambda r: (r["is_far"], abs(r["lambda"])))
    two, cases = 0, 0
    for X in _r21_points():
        rays = [r for r in cam.projectAll(lpt.Pt3D(*X))[0] if not r["is_ring"]]
        if len(rays) < 2:
            continue
        two += 1
        vis = [r for r in rays if on(r)]
        want = best(vis) if vis else best(rays)
        got = cam.project(lpt.Pt3D(*X))
        assert (got[0], got[1]) == (want["pt"][0], want["pt"][1]), X
        cases += bool(vis) and not on(best(rays))
    assert (two, cases) == (212, 5)


def test_d_an_invisible_point_is_reported_as_such():
    cam = _camera()
    for X in _r21_points():
        rays, cov = cam.projectAll(lpt.Pt3D(*X))
        if not rays and cov["near_branch"] and cov["far_branch"]:
            with pytest.raises(RuntimeError, match="has no image"):
                cam.project(lpt.Pt3D(*X))
            return
    pytest.fail("the scan has no invisible point")


# --------------------------------------------- C5: rings on the axial line
SHELL_N, SHELL_R = [1.0, 1.362, 1.503, 1.962, 2.607], [1.0, 0.867, 0.653, 0.487]


@pytest.mark.parametrize("rP,rX,branches", [(8.0, 0.3625, ["far", "far"]), (20.0, 0.365, ["far", "near"])])
def test_c5_every_ring_on_the_axial_line(rP, rX, branches):
    """A shell found by code/tests/openlpt_followup/f2_ring_search.py, in mm (R_1 = 50 mm)."""
    s = 50.0
    n, R, P0, X = SHELL_N, [r * s for r in SHELL_R], [rP * s, 0.0, 0.0], [-rX * s, 0.0, 0.0]
    ref = ps.ring_roots(n, R, rP * s, rX * s)
    assert sorted(b for b, _ in ref) == branches
    want = sorted(L for _, L in ref)
    cpp = sorted(L for b, L, _ in rc.project_sphere(n, R, [0.0, 0.0, 0.0], P0, X) if b == "ring")
    py = sorted(L for b, L, _ in ps.project(n, R, np.zeros(3), np.array(P0), np.array(X)) if b == "ring")
    assert cpp == pytest.approx(want, rel=1e-9) and py == pytest.approx(want, rel=1e-9)
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    for L in want:  # each ring is a real cone of rays: traced by vector Snell's law, it meets X
        p, d = ps.trace(np.array(P0), ps.direction_at_pinhole(L, n[0], rP * s, e1, e2), n, R, np.zeros(3))
        v = np.array(X) - p
        assert np.linalg.norm(v - np.dot(v, d) * d) < 1e-9 * s


def test_c5_single_spheres_keep_their_one_ring_or_none():
    """Section ring: a ball lens has one ring on the axial line, a water sphere none."""
    for n2, count in ((1.52, 1), (2.0, 1), (1.33, 0)):
        rays = rc.project_sphere([1.0, n2], [50.0], [0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [-49.0, 0.0, 0.0])
        assert sum(b == "ring" for b, _, _ in rays) == count, n2


def test_d_ring_entries_carry_no_pixel():
    s = 50.0
    cam = _camera("sphere", center=(20.0 * s, 0.0, 0.0), n=SHELL_N, radii=[r * s for r in SHELL_R])
    X = lpt.Pt3D(-0.365 * s, 0.0, 0.0)
    rays, _ = cam.projectAll(X)
    rings = [r for r in rays if r["is_ring"]]
    assert len(rings) == 2
    assert all(math.isnan(r["pt"][0]) and math.isnan(r["pt"][1]) for r in rings)
    central = [r for r in rays if not r["is_ring"]]
    assert len(central) == 1
    got = cam.project(X)
    assert (got[0], got[1]) == (central[0]["pt"][0], central[0]["pt"][1])


# ------------------------------------------------------- D: cosmetic items
def test_d_solve_far_counts_the_iterations_it_took():
    """A target no far ray reaches: Newton stops on a failed line search, well before itmax (40)."""
    R = [100.0 / 256.0, 94.0 / 256.0]
    ok, _L, _C, res, iters = rc.solve_far(REF_N, R, 350.0 / 256.0, 0.3580817014515325, 2.310065610100996,
                                          -0.17086909970221584, 0.2613300501012747, 0.05)
    assert not ok and res > 1e-3
    assert iters < 40


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_d_curved_camera_files_round_trip_exactly(tmp_path, kind):
    """Every parameter is written with 17 significant digits (8 moved a pixel by up to 1.7e-6)."""
    cam = lpt.RefractionSphereCameraModel() if kind == "sphere" else lpt.RefractionCylinderCameraModel()
    cam.setImageSize(1024, 1280)
    cam.setIntrinsics(1800.0 + 1 / 3, 1800.0 - 1 / 7, 640.0 + 1 / 9, 512.0 - 1 / 11,
                      [0.1 / 3, -0.01 / 7, 0.001 / 3, 0.0, 0.0])
    cam.setExtrinsics(lpt.Pt3D(0.1 / 3, -1.5 + 1 / 7, 0.2 / 9), lpt.Pt3D(1 / 3, -2 / 7, 350.0 + 1 / 11))
    cam.setRefraction(lpt.Pt3D(1 / 3, -1 / 7, 0.0), lpt.Pt3D(0.0, 0.0, 1.0), [1.0, 1.49 + 1e-9 / 3, 1.33],
                      [100.0 + 1 / 3, 94.0 - 1 / 7])
    cam.setSolverOptions(1e-10, 100)
    cam.commitUpdate()
    path = str(tmp_path / "cam.txt")
    cam.saveParameters(path)
    back = lpt.loadCameraModel(path)
    for X in ((30.0, 40.0, 25.0), (-20.0, 10.0, -30.0), (5.0, -60.0, 70.0)):
        a, b = cam.projectAll(lpt.Pt3D(*X))[0], back.projectAll(lpt.Pt3D(*X))[0]
        assert [tuple(r["pt"]) for r in a] == [tuple(r["pt"]) for r in b], X


@pytest.mark.parametrize("name", ["cam1.txt", "cam1_poly.txt", "cam1_refract.txt"])
def test_d_other_camera_files_round_trip(tmp_path, name):
    """Pinhole, polynomial and plate files: a save and load changes no pixel, and saving again changes no digit."""
    first = lpt.loadCameraModel(str(REPO_ROOT / "test" / "inputs" / "test_Camera" / name))
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    first.saveParameters(str(a))
    second = lpt.loadCameraModel(str(a))
    second.saveParameters(str(b))
    assert a.read_text() == b.read_text()
    X = lpt.Pt3D(10.0, 20.0, 30.0)
    p, q = first.project(X), second.project(X)
    assert (p[0], p[1]) == (q[0], q[1])


@pytest.mark.parametrize("az,branch", [(1.0, "near"), (3.0, "far")])
@pytest.mark.parametrize("impl", ["C++", "Python"])
def test_d_on_axis_label_is_the_limit_of_the_general_case(impl, az, branch):
    """Below r_X = 1e-15 (in units of 2^floor(log2 r_P)) step (1) takes the radial ray; its label must be the
    general path's just above that."""
    for rX in (1e-16 * REF_RP_M, 1e-12 * REF_RP_M):
        X = [rX * math.cos(az), rX * math.sin(az), 0.05]
        assert [b for b, _, _ in _cyl(REF_N, REF_R_M, REF_RP_M, X, 1.0, impl)] == [branch], rX
    assert [b for b, _, _ in _cyl(REF_N, REF_R_M, REF_RP_M, [0.0, 0.0, 0.05], 1.0, impl)] == ["near"]


@pytest.mark.parametrize("az", [1e-16, -1e-16, 3e-16, math.pi - 3e-16, -(math.pi - 3e-16), math.pi - 1e-12])
def test_d_reference_and_cpp_agree_where_the_old_reduction_rounded(az):
    """The reference now reduces the azimuth by math.remainder, as the C++ port does by std::remainder;
    (d + pi) % 2 pi - pi rounded twice and flushed |d| < 2.2e-16 to 0. That error is below the projection's
    tolerance, so no ray moves: this checks that the two implementations agree at such azimuths."""
    X = [0.05 * math.cos(az), 0.05 * math.sin(az), 0.02]
    _same(_cyl(REF_N, REF_R_M, REF_RP_M, X, 1.0, "Python"), _cyl(REF_N, REF_R_M, REF_RP_M, X, 1.0, "C++"))


def test_d_radial_action_is_finite_at_zero_on_the_far_branch():
    n, R, rP, rX, C = REF_N, REF_R_M, REF_RP_M, 0.05, 0.1
    w0 = rc.radial_action(0.0, C, n, R, rP, rX, True)
    assert w0 is not None and math.isfinite(w0)
    assert w0 == pytest.approx(rc.radial_action(1e-15, C, n, R, rP, rX, True), abs=1e-12)
    assert pc.W_closed(0.0, C, n, R, rP, rX, far=True) == pytest.approx(w0, rel=1e-12)
