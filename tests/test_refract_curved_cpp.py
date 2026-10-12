"""The C++ curved-port core against the Python reference implementations.

`pyopenlpt.refract_curved` is a term-for-term port of `test/refs/project_cylinder.py`
(and, for spheres, `project_sphere.py`).  These tests are the gate on that port:
every closed form is compared on random admissible stacks, then the whole sphere
projector is compared ray list against ray list, then the rays the C++ returns
are re-traced with vector Snell's law so that agreement with a wrong reference
would still be caught.

Tolerances are tight on purpose.  The two implementations evaluate the same
expressions in the same order, so anything looser than ~1e-12 relative would hide
a genuine difference in formula rather than in rounding.
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

pyopenlpt = pytest.importorskip("pyopenlpt", reason="extension not built")
rc = pytest.importorskip(
    "pyopenlpt.refract_curved", reason="refract_curved bindings not built"
)

REF_N = [1.0, 1.49, 1.33]
REF_R = [100.0, 94.0]
REF_RP = 350.0
BALL_N = [1.0, 2.00]
BALL_R = [50.0]
BALL_RP = 500.0

ABS = 1e-11
REL = 1e-11


def _stacks(seed, count=30):
    """Random admissible stacks, as the reference implementation's self-test builds them."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n_media = int(rng.integers(2, 6))
        n = [1.0] + list(rng.uniform(1.05, 2.0, n_media - 1))
        radii = np.sort(rng.uniform(0.03, 0.12, n_media - 1))[::-1].tolist()
        r_p = rng.uniform(radii[0] * 1.05, 1.0)
        r_x = rng.uniform(1e-3, radii[-1])
        yield n, radii, r_p, r_x, rng


def _close(got, want, what):
    if want is None:
        assert got is None, f"{what}: C++ returned {got}, reference returned None"
        return
    assert got is not None, f"{what}: C++ returned None, reference returned {want}"
    assert got == pytest.approx(want, rel=REL, abs=ABS), f"{what}: {got} vs {want}"


# ----------------------------------------------------------- the closed forms
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_closed_forms_match_reference(seed):
    for n, radii, r_p, r_x, rng in _stacks(seed):
        for _ in range(8):
            c_z = float(rng.uniform(-0.6, 0.6))
            l_max = pc.Lmax_kstar(n, radii, r_x, c_z)[0]
            lam = float(rng.uniform(0.0, 1.0) * l_max * (1 - 1e-9))
            for far in (False, True):
                tag = f"n={len(n)} c_z={c_z:.4f} lam/Lmax={lam / l_max:.4f} far={far}"

                want = pc.Phi_Z(lam, c_z, n, radii, r_p, r_x, far)
                got = rc.phi_z(lam, c_z, n, radii, r_p, r_x, far)
                if want is None:
                    assert got is None, f"phi_z {tag}"
                else:
                    _close(got[0], want[0], f"phi {tag}")
                    _close(got[1], want[1], f"z {tag}")

                _close(
                    rc.radial_action(lam, c_z, n, radii, r_p, r_x, far),
                    pc.W_closed(lam, c_z, n, radii, r_p, r_x, far),
                    f"W {tag}",
                )
                _close(
                    rc.dphi_dlam(lam, c_z, n, radii, r_p, r_x, far),
                    pc.dPhi_dLam(lam, c_z, n, radii, r_p, r_x, far),
                    f"dPhi/dLam {tag}",
                )
                _close(
                    rc.dphi_dcz(lam, c_z, n, radii, r_p, r_x, far),
                    pc.dPhi_dCz(lam, c_z, n, radii, r_p, r_x, far),
                    f"dPhi/dCz {tag}",
                )
                _close(
                    rc.dz_dcz(lam, c_z, n, radii, r_p, r_x, far),
                    pc.dZ_dCz(lam, c_z, n, radii, r_p, r_x, far),
                    f"dZ/dCz {tag}",
                )


@pytest.mark.parametrize("seed", [11, 12])
def test_lmax_domain_and_z0_match_reference(seed):
    for n, radii, r_p, r_x, rng in _stacks(seed):
        for _ in range(10):
            c_z = float(rng.uniform(-0.9, 0.9))
            want_lmax, want_kstar = pc.Lmax_kstar(n, radii, r_x, c_z)
            got_lmax, got_kstar = rc.lmax_kstar(n, radii, r_x, c_z)
            _close(got_lmax, want_lmax, "Lmax")
            assert got_kstar == want_kstar, f"k*: {got_kstar} vs {want_kstar}"

            _close(rc.z0(c_z, n, radii, r_p, r_x), pc.Z0(c_z, n, radii, r_p, r_x), "Z0")

            lam = float(rng.uniform(0.0, 1.2) * want_lmax)
            assert rc.in_domain(lam, c_z, n, radii, r_x) == pc.in_domain(
                lam, c_z, n, radii, r_x
            )

            np.testing.assert_allclose(
                rc.nus(n, c_z), pc.nus(n, c_z), rtol=REL, atol=ABS
            )


# ------------------------------------------------------------- the far branch
@pytest.mark.parametrize("seed", [21, 22])
def test_far_turns_match_reference(seed):
    """The turning points are what cut Phi_far into monotone arcs; losing one loses rays."""
    for n, radii, r_p, r_x, rng in _stacks(seed, count=20):
        for _ in range(4):
            c_z = float(rng.uniform(-0.5, 0.5))
            want = pc.far_turns(c_z, n, radii, r_p, r_x)
            got = rc.far_turns(c_z, n, radii, r_p, r_x)
            assert len(got) == len(want), (
                f"turning-point count {len(got)} vs {len(want)} "
                f"(n={n}, R={radii}, r_p={r_p}, r_x={r_x}, c_z={c_z})"
            )
            for g, w in zip(sorted(got), sorted(want)):
                _close(g, w, "turning point")


def test_far_roots_match_reference_on_the_reference_port():
    for r_x in (20.0, 50.0, 80.0, 93.0):
        for target_deg in (30.0, 90.0, 150.0, 179.0):
            target = math.radians(target_deg)
            want = pc.far_roots(0.0, REF_N, REF_R, REF_RP, r_x, target)
            got = rc.far_roots(0.0, REF_N, REF_R, REF_RP, r_x, target)
            assert len(got) == len(want), (
                f"root count at r_x={r_x}, target={target_deg} deg: "
                f"{len(got)} vs {len(want)}"
            )
            for (gl, gs, gz), (wl, ws, wz) in zip(got, want):
                _close(gl, wl, "far root Lambda")
                assert gs == ws, f"arc slope {gs} vs {ws}"
                _close(gz, wz, "far root Z")


# ------------------------------------------------------------------- spheres
def test_sweeps_and_chord_bound_match_reference():
    for n, radii, r_p in ((REF_N, REF_R, REF_RP), (BALL_N, BALL_R, BALL_RP)):
        _close(rc.chord_bound(n, radii, r_p), ps.chord_bound(n, radii, r_p), "chord")
        for r_x in (0.2 * radii[-1], 0.6 * radii[-1], radii[-1] * (1 - 1e-12)):
            got_near, got_m0 = rc.sweeps(n, radii, r_p, r_x)
            want_near, want_m0 = ps.sweeps(n, radii, r_p, r_x)
            _close(got_near, want_near, f"near reach r_x={r_x}")
            _close(got_m0, want_m0, f"M(0) r_x={r_x}")


def test_cpp_sweeps_reproduce_the_document_numbers():
    """The same four numbers of doc/refract_curved.tex:922, straight out of C++."""
    near, m0 = rc.sweeps(BALL_N, BALL_R, BALL_RP, 49.0)
    assert math.degrees(near) == pytest.approx(84.94, abs=5e-3)
    assert math.degrees(m0) == pytest.approx(203.58, abs=5e-3)
    assert 360.0 - math.degrees(m0) == pytest.approx(156.42, abs=5e-3)

    _, m0_ref = rc.sweeps(REF_N, REF_R, REF_RP, 94.0 * (1 - 1e-12))
    assert math.degrees(m0_ref) == pytest.approx(180.0, abs=1e-6)


def test_ray_plane_matches_reference():
    rng = np.random.default_rng(99)
    centre = [0.0, 0.0, 0.0]
    for _ in range(50):
        pinhole = list(rng.uniform(-400.0, 400.0, 3))
        point = list(rng.uniform(-90.0, 90.0, 3))
        got = rc.ray_plane(centre, pinhole, point)
        want = ps.ray_plane(centre, pinhole, point)
        _close(got[0], want[0], "r_p")
        _close(got[1], want[1], "r_x")
        _close(got[2], want[2], "dphi")
        np.testing.assert_allclose(got[3], want[3], rtol=REL, atol=ABS)
        assert got[5] == (want[4] is None), "axial flag disagrees"
        if want[4] is not None:
            np.testing.assert_allclose(got[4], want[4], rtol=REL, atol=ABS)


def _sorted_rays(rays):
    """Rays are returned per target, so compare as a set ordered by (branch, Lambda)."""
    return sorted(((b, round(lam, 10)) for b, lam, _ in rays))


@pytest.mark.parametrize("seed", [31, 32])
def test_project_sphere_matches_reference_ray_for_ray(seed):
    rng = np.random.default_rng(seed)
    centre = [0.0, 0.0, 0.0]
    pinhole = [REF_RP, 0.0, 0.0]
    n_compared = 0
    for _ in range(80):
        r_x = rng.uniform(1.0, 93.0)
        theta, phi = rng.uniform(0.0, math.pi), rng.uniform(0.0, 2 * math.pi)
        point = [
            r_x * math.cos(theta),
            r_x * math.sin(theta) * math.cos(phi),
            r_x * math.sin(theta) * math.sin(phi),
        ]
        got = rc.project_sphere(REF_N, REF_R, centre, pinhole, point)
        want = ps.project(REF_N, REF_R, centre, pinhole, point)
        assert _sorted_rays(got) == _sorted_rays(want), (
            f"ray lists differ at r_x={r_x:.3f}: C++ {_sorted_rays(got)} "
            f"vs reference {_sorted_rays(want)}"
        )
        for (_, _, gd), (_, _, wd) in zip(got, want):
            if wd is not None:
                np.testing.assert_allclose(gd, wd, rtol=1e-10, atol=1e-12)
        n_compared += len(got)
    assert n_compared > 50, f"suspiciously few rays compared ({n_compared})"


def test_project_sphere_rays_reach_their_point():
    """Independent of the reference: re-trace what C++ returns with vector Snell."""
    rng = np.random.default_rng(4242)
    centre = np.zeros(3)
    pinhole = np.array([REF_RP, 0.0, 0.0])
    n_traced = 0
    for _ in range(60):
        r_x = rng.uniform(1.0, 93.0)
        theta, phi = rng.uniform(0.0, math.pi), rng.uniform(0.0, 2 * math.pi)
        point = centre + r_x * np.array(
            [math.cos(theta), math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi)]
        )
        for branch, _lam, direction in rc.project_sphere(
            REF_N, REF_R, list(centre), list(pinhole), list(point)
        ):
            if branch == "ring":
                continue
            hit, dirn = ps.trace(pinhole, np.asarray(direction), REF_N, REF_R, centre)
            assert hit is not None, f"{branch} ray does not cross the stack"
            w = point - hit
            assert np.linalg.norm(w - np.dot(w, dirn) * dirn) < 1e-9
            n_traced += 1
    assert n_traced > 40, f"suspiciously few rays traced ({n_traced})"


def test_project_sphere_axial_ring():
    """Section ring: the ball lens carries a ring on the axial line, the weak sphere does not."""
    centre, pinhole = [0.0, 0.0, 0.0], [BALL_RP, 0.0, 0.0]
    point = [-49.0, 0.0, 0.0]
    assert [b for b, _, _ in rc.project_sphere(BALL_N, BALL_R, centre, pinhole, point)] == [
        "far",
        "ring",
    ]
    assert [
        b for b, _, _ in rc.project_sphere([1.0, 1.33], BALL_R, centre, pinhole, point)
    ] == ["far"]


# ------------------------------------------------------------------ cylinders
@pytest.mark.parametrize("seed", [41, 42])
def test_solve_cylinder_matches_reference(seed):
    """The 2-D Newton solve must land on the same (Lambda, C_z) as the reference."""
    rng = np.random.default_rng(seed)
    n_solved = 0
    for n, radii, r_p, r_x, _ in _stacks(seed, count=15):
        for _ in range(6):
            c_z = float(rng.uniform(-0.4, 0.4))
            l_max = pc.Lmax_kstar(n, radii, r_x, c_z)[0]
            lam = float(rng.uniform(0.05, 0.9) * l_max)
            truth = pc.Phi_Z(lam, c_z, n, radii, r_p, r_x, False)
            if truth is None:
                continue
            dphi, dz = truth
            lam0, cz0 = 0.5 * lam, 0.5 * c_z
            ok, got_lam, got_cz, res, _ = rc.solve_cylinder(
                n, radii, r_p, r_x, dphi, dz, lam0, cz0, False
            )
            want, _, want_res = pc.solve(
                n, radii, r_p, r_x, dphi, dz, lam0, cz0, far=False
            )
            if not ok or want is None or want_res > 1e-10:
                continue
            _close(got_lam, want[0], "solved Lambda")
            _close(got_cz, want[1], "solved C_z")
            assert res < 1e-10
            # And it must be the value the system was built from.
            assert got_lam == pytest.approx(lam, rel=1e-8, abs=1e-10)
            assert got_cz == pytest.approx(c_z, rel=1e-8, abs=1e-10)
            n_solved += 1
    assert n_solved > 20, f"suspiciously few solves compared ({n_solved})"


def test_components_match_reference():
    """Property reachpieces: losing a component loses every near-branch ray in it."""
    for r_x in (20.0, 60.0, 93.0):
        for dphi_deg in (5.0, 30.0, 60.0, 80.0):
            dphi = math.radians(dphi_deg)
            want = pc.components(REF_N, REF_R, REF_RP, r_x, dphi)
            got = rc.components(REF_N, REF_R, REF_RP, r_x, dphi)
            assert len(got) == len(want), (
                f"component count at r_x={r_x}, dphi={dphi_deg} deg: "
                f"{len(got)} vs {len(want)}"
            )
            for (glo, ghi), (wlo, whi) in zip(got, want):
                _close(glo, wlo, "component lo")
                _close(ghi, whi, "component hi")


@pytest.mark.parametrize("seed", [61, 62])
def test_classify_near_matches_reference(seed):
    """The near-branch seed search must agree case for case, including its misses."""
    rng = np.random.default_rng(seed)
    n_found = 0
    for _ in range(120):
        r_x = float(rng.uniform(5.0, 93.0))
        dphi = float(rng.uniform(0.01, 1.2))
        dz = float(rng.uniform(-300.0, 300.0))
        want = pc.classify(REF_N, REF_R, REF_RP, r_x, dphi, dz)
        got = rc.classify_near(REF_N, REF_R, REF_RP, r_x, dphi, dz)
        assert (got is None) == (want is None), (
            f"disagree on existence at r_x={r_x:.3f}, dphi={dphi:.4f}, dz={dz:.3f}: "
            f"C++ {got}, reference {want}"
        )
        if want is not None:
            _close(got[0], want[0], "seed Lambda")
            _close(got[1], want[1], "seed C_z")
            n_found += 1
    assert n_found > 20, f"suspiciously few seeds found ({n_found})"


def test_project_cylinder_searches_both_branches():
    """Both branches are searched now, so an empty list really does mean invisible."""
    _rays, coverage = rc.project_cylinder(
        REF_N, REF_R, [REF_RP, 0.0, 0.0], [50.0, 10.0, 20.0]
    )
    assert coverage["near_branch"] is True
    assert coverage["far_branch"] is True


@pytest.mark.parametrize("seed", [81, 82])
def test_classify_far_matches_reference(seed):
    """The certified box search must find the same seeds as the reference.

    This is the part of the algorithm that could silently lose rays, so it is
    compared directly rather than only through project_cylinder.
    """
    rng = np.random.default_rng(seed)
    n_with_rays = 0
    for _ in range(80):
        r_x = float(rng.uniform(10.0, 93.0))
        dphi = float(rng.uniform(0.1, 3.1))
        dz = float(rng.uniform(1.0, 400.0))
        want = sorted(pc.classify_far(REF_N, REF_R, REF_RP, r_x, dphi, dz))
        got_seeds, stats = rc.classify_far(REF_N, REF_R, REF_RP, r_x, dphi, dz)
        got = sorted(tuple(g) for g in got_seeds)

        assert len(got) == len(want), (
            f"far-seed count at r_x={r_x:.3f}, dphi={dphi:.4f}, dz={dz:.3f}: "
            f"C++ {len(got)} vs reference {len(want)}; stats={stats}"
        )
        for (gl, gc), (wl, wc) in zip(got, want):
            assert gl == pytest.approx(wl, rel=1e-7, abs=1e-9)
            assert gc == pytest.approx(wc, rel=1e-7, abs=1e-9)
        if want:
            n_with_rays += 1
    assert n_with_rays > 10, (
        f"only {n_with_rays} cases had far-branch rays; the sample is not "
        "exercising the box search"
    )


def _ray_key(rays, places=8):
    """Rays come back per target, so compare as a set ordered by (branch, Lambda, C_z)."""
    return sorted((b, round(lam, places), round(c_z, places)) for b, lam, c_z, *_ in rays)


@pytest.mark.parametrize("seed", [71, 72])
def test_project_cylinder_matches_reference_ray_for_ray(seed):
    """The whole projector: both branches, both senses, wrapped targets."""
    rng = np.random.default_rng(seed)
    pinhole = np.array([REF_RP, 0.0, 0.0])
    n_rays = 0
    for _ in range(80):
        r_x = float(rng.uniform(5.0, 93.0))
        azimuth = float(rng.uniform(-math.pi, math.pi))
        point = np.array(
            [r_x * math.cos(azimuth), r_x * math.sin(azimuth), float(rng.uniform(-250.0, 250.0))]
        )
        rays, _ = rc.project_cylinder(REF_N, REF_R, list(pinhole), list(point))
        want = pc.project(REF_N, REF_R, pinhole, point)
        assert _ray_key(rays) == _ray_key(want), (
            f"ray lists differ at r_x={r_x:.4f}, azimuth={azimuth:.5f}, "
            f"dz={point[2]:.4f}:\n  C++ {_ray_key(rays)}\n  ref {_ray_key(want)}"
        )
        n_rays += len(rays)
    assert n_rays > 50, f"suspiciously few cylinder rays compared ({n_rays})"


def test_project_cylinder_matches_reference_in_the_multi_image_regime():
    """Wide azimuth near the wall, where the far branch folds and points get two images."""
    rng = np.random.default_rng(29)
    pinhole = np.array([REF_RP, 0.0, 0.0])
    counts = {}
    for _ in range(150):
        r_x = float(rng.uniform(60.0, 93.5))
        azimuth = float(rng.uniform(1.8, math.pi))
        point = np.array(
            [r_x * math.cos(azimuth), r_x * math.sin(azimuth), float(rng.uniform(-150.0, 150.0))]
        )
        rays, _ = rc.project_cylinder(REF_N, REF_R, list(pinhole), list(point))
        want = pc.project(REF_N, REF_R, pinhole, point)
        assert _ray_key(rays) == _ray_key(want), (
            f"ray lists differ at r_x={r_x:.4f}, azimuth={azimuth:.5f}, "
            f"dz={point[2]:.4f}:\n  C++ {_ray_key(rays)}\n  ref {_ray_key(want)}"
        )
        counts[len(want)] = counts.get(len(want), 0) + 1
    assert counts.get(2, 0) >= 3, (
        f"expected some two-image points in this regime, got {counts}"
    )
    assert counts.get(0, 0) >= 1, f"expected some invisible points, got {counts}"


@pytest.mark.parametrize("dz", [5.0, 60.0, 200.0])
def test_project_cylinder_point_opposite_the_pinhole(dz):
    """Delta phi = pi: the far branch's radial ray, recorded once for both senses."""
    pinhole = np.array([REF_RP, 0.0, 0.0])
    point = np.array([-70.0, 0.0, dz])
    rays, _ = rc.project_cylinder(REF_N, REF_R, list(pinhole), list(point))
    want = pc.project(REF_N, REF_R, pinhole, point)
    assert _ray_key(rays) == _ray_key(want)


@pytest.mark.parametrize("r_x", [20.0, 50.0, 80.0, 90.0, 93.0])
def test_project_cylinder_level_with_the_pinhole(r_x):
    """dz = 0 exactly: classify_far takes its own path, solving Phi_far alone at C_z = 0.

    Random sampling never hits this, so it is exercised deliberately.
    """
    pinhole = np.array([REF_RP, 0.0, 0.0])
    for azimuth_deg in (10, 45, 90, 120, 150, 170, 179):
        azimuth = math.radians(azimuth_deg)
        point = np.array([r_x * math.cos(azimuth), r_x * math.sin(azimuth), 0.0])
        rays, _ = rc.project_cylinder(REF_N, REF_R, list(pinhole), list(point))
        want = pc.project(REF_N, REF_R, pinhole, point)
        assert _ray_key(rays) == _ray_key(want), (
            f"ray lists differ at r_x={r_x}, azimuth={azimuth_deg} deg, dz=0:\n"
            f"  C++ {_ray_key(rays)}\n  ref {_ray_key(want)}"
        )
        for _b, lam, c_z, _d in rays:
            # The far branch short-circuits to C_z = 0 exactly; the near branch
            # reaches it through Newton, so it lands within rounding of zero.
            assert abs(c_z) < 1e-15, (
                f"a ray level with the pinhole has tilt {c_z}"
            )


@pytest.mark.parametrize("seed", [91, 92])
def test_project_cylinder_matches_reference_on_random_stacks(seed):
    """Not just the reference port: 2 to 4 media, random radii and indices."""
    rng = np.random.default_rng(seed)
    n_points = 0
    for _ in range(12):
        n_media = int(rng.integers(2, 5))
        n = [1.0] + list(rng.uniform(1.05, 1.9, n_media - 1))
        radii = np.sort(rng.uniform(30.0, 120.0, n_media - 1))[::-1].tolist()
        r_p = float(rng.uniform(radii[0] * 1.2, 600.0))
        pinhole = np.array([r_p, 0.0, 0.0])
        for _ in range(10):
            r_x = float(rng.uniform(1.0, radii[-1] * 0.98))
            azimuth = float(rng.uniform(-math.pi, math.pi))
            point = np.array(
                [r_x * math.cos(azimuth), r_x * math.sin(azimuth), float(rng.uniform(-300.0, 300.0))]
            )
            rays, _ = rc.project_cylinder(n, radii, list(pinhole), list(point))
            want = pc.project(n, radii, pinhole, point)
            assert _ray_key(rays, 7) == _ray_key(want, 7), (
                f"ray lists differ for n={n}, R={radii}, r_p={r_p:.2f}, "
                f"r_x={r_x:.4f}, azimuth={azimuth:.5f}, dz={point[2]:.3f}:\n"
                f"  C++ {_ray_key(rays, 7)}\n  ref {_ray_key(want, 7)}"
            )
            n_points += 1
    assert n_points == 120


@pytest.mark.parametrize("seed", [101, 102])
def test_project_cylinder_rays_reach_their_point(seed):
    """Independent of the reference: re-trace every ray, on both branches."""
    rng = np.random.default_rng(seed)
    pinhole = np.array([REF_RP, 0.0, 0.0])
    n_near = n_far = 0
    for _ in range(100):
        r_x = float(rng.uniform(20.0, 93.0))
        azimuth = float(rng.uniform(-math.pi, math.pi))
        point = np.array(
            [r_x * math.cos(azimuth), r_x * math.sin(azimuth), float(rng.uniform(-200.0, 200.0))]
        )
        rays, _ = rc.project_cylinder(REF_N, REF_R, list(pinhole), list(point))
        for branch, lam, c_z, direction in rays:
            hit = pc.trace_to_radius(
                pinhole, np.asarray(direction), REF_N, REF_R, r_x, far=(branch == "far")
            )
            assert hit is not None, (
                f"{branch} ray (Lambda={lam:g}, C_z={c_z:g}) never reaches r_X"
            )
            assert np.linalg.norm(hit - point) < 1e-8, (
                f"{branch} ray misses by {np.linalg.norm(hit - point):.3e} mm"
            )
            n_near += branch == "near"
            n_far += branch == "far"
    assert n_far > 20, f"only {n_far} far-branch rays traced; regime too easy"
    assert n_near > 20, f"only {n_near} near-branch rays traced"


def test_direction_from_invariants_matches_reference():
    rng = np.random.default_rng(55)
    for _ in range(50):
        pinhole = [float(rng.uniform(200.0, 500.0)), float(rng.uniform(-200.0, 200.0)), 0.0]
        c_z = float(rng.uniform(-0.5, 0.5))
        n1, nu1 = 1.0, math.sqrt(1.0 - c_z * c_z)
        r_p = math.hypot(pinhole[0], pinhole[1])
        lam = float(rng.uniform(-0.9, 0.9)) * nu1 * r_p
        np.testing.assert_allclose(
            rc.direction_from_invariants(pinhole, lam, c_z, n1, nu1),
            pc.direction_from_invariants(np.array(pinhole), lam, c_z, n1, nu1),
            rtol=1e-12,
            atol=1e-14,
        )
