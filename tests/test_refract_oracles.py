"""Validation of the reference projectors in `test/refs/`.

These are the oracles the C++ implementation is checked against, so they are
pinned here first.  Three kinds of check:

  * **Drift** - the vendored `project_cylinder.py` must stay byte-identical to
    the notes repo's copy, which is the source of truth.
  * **Against the document** - numbers `doc/refract_curved.tex` states outright
    (Property shadow, Section ring) must come back out of the code.
  * **Against physics** - every ray returned is re-traced with vector Snell's
    law and must land on the point it claims to reach.

The third kind is the one that would catch a wrong formula; the second catches a
right formula wired up to the wrong case.
"""
import hashlib
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

# The reference port of doc/refract_curved.tex Section "Numerical verification".
REF_N = [1.0, 1.49, 1.33]
REF_R = [100.0, 94.0]
REF_RP = 350.0

# The ball lens of Section ring: a single sphere, R = 50 mm, in air.
BALL_N = [1.0, 2.00]
BALL_R = [50.0]
BALL_RP = 500.0

TRACE_TOL_MM = 1e-9


# --------------------------------------------------------------------- drift
def test_vendored_cylinder_oracle_matches_notes_repo():
    """`test/refs/project_cylinder.py` is a verbatim copy; catch silent drift.

    The notes repo is the source of truth for the algorithm.  If it is not
    checked out beside this one the test skips rather than passing vacuously.
    """
    original = REPO_ROOT.parent / "code" / "project_cylinder.py"
    if not original.is_file():
        pytest.skip(f"notes repo not checked out at {original}")
    vendored = REFS / "project_cylinder.py"
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert h(vendored) == h(original), (
        "test/refs/project_cylinder.py has drifted from the notes repo copy; "
        "re-copy it rather than editing the vendored file"
    )


# ------------------------------------------------- against the document
def test_sphere_property_shadow_numbers():
    """doc/refract_curved.tex:922 - the rod / ball lens at r_X = 49 mm.

    Its gap is (84.94, 177.87) degrees but its shadow only (84.94, 156.42),
    because M(0) = Phi_far(Lmax) = 203.58 exceeds pi and the excess is reached
    the long way round.
    """
    r_x = 49.0
    near, m0 = ps.sweeps(BALL_N, BALL_R, BALL_RP, r_x)
    l_max = pc.Lmax_kstar(BALL_N, BALL_R, r_x, 0.0)[0] * (1 - 1e-13)
    far_vals = [math.pi, pc.Phi_Z(l_max, 0.0, BALL_N, BALL_R, BALL_RP, r_x, far=True)[0]]
    far_vals += [
        pc.Phi_Z(lam, 0.0, BALL_N, BALL_R, BALL_RP, r_x, far=True)[0]
        for lam in pc.far_turns(0.0, BALL_N, BALL_R, BALL_RP, r_x)
        if 0.0 < lam < l_max
    ]

    assert math.degrees(near) == pytest.approx(84.94, abs=5e-3)
    assert math.degrees(min(far_vals)) == pytest.approx(177.87, abs=5e-3)
    assert math.degrees(m0) == pytest.approx(203.58, abs=5e-3)
    assert 360.0 - math.degrees(m0) == pytest.approx(156.42, abs=5e-3)


def test_sphere_ray_at_958_percent_of_lmax_sweeps_190_degrees():
    """Same passage: the ray Lambda = 0.958 Lmax sweeps 190 deg, landing at 170."""
    r_x = 49.0
    l_max = pc.Lmax_kstar(BALL_N, BALL_R, r_x, 0.0)[0] * (1 - 1e-13)
    swept = pc.Phi_Z(0.958 * l_max, 0.0, BALL_N, BALL_R, BALL_RP, r_x, far=True)[0]
    assert math.degrees(swept) == pytest.approx(190.0, abs=0.1)


def test_sphere_ring_threshold():
    """doc/refract_curved.tex:1942 - the axial line carries a ring above r_X = 30.80 mm."""
    lo, hi = 1.0, 49.9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ps.ring(BALL_N, BALL_R, BALL_RP, mid) is None else (lo, mid)
    assert 0.5 * (lo + hi) == pytest.approx(30.80, abs=5e-3)


def test_reference_port_as_sphere_never_wraps():
    """doc/refract_curved.tex:1931 - Phi_far(Lmax) < 180, so the largest sweep is exactly pi.

    Nothing is reached the long way round, so `_targets` must offer one target.
    """
    r_x = 94.0 * (1 - 1e-12)
    _, m0 = ps.sweeps(REF_N, REF_R, REF_RP, r_x)
    assert math.degrees(m0) == pytest.approx(180.0, abs=1e-6)
    assert ps._targets(math.radians(120.0), REF_N, REF_R, REF_RP, r_x) == [
        (pytest.approx(math.radians(120.0)), 1.0)
    ]


def test_ball_lens_wraps():
    """doc/refract_curved.tex:1938 - M(0) = 204.26 at the wall, so azimuths past 155.74 wrap."""
    r_x = 50.0 * (1 - 1e-12)
    _, m0 = ps.sweeps(BALL_N, BALL_R, BALL_RP, r_x)
    assert math.degrees(m0) == pytest.approx(204.26, abs=5e-3)

    # An azimuth inside the wrapped band gets a second target, swept the other way.
    targets = ps._targets(math.radians(170.0), BALL_N, BALL_R, BALL_RP, r_x)
    senses = sorted(s for _, s in targets)
    assert senses == [-1.0, 1.0], f"expected both senses, got {targets}"


# ----------------------------------------------------------- against physics
def _sphere_miss(n, radii, centre, pinhole, point, direction):
    """Distance from `point` to the traced ray inside the innermost medium."""
    hit, dirn = ps.trace(pinhole, direction, n, radii, centre)
    if hit is None:
        return math.inf
    w = np.asarray(point, float) - hit
    return float(np.linalg.norm(w - np.dot(w, dirn) * dirn))


@pytest.mark.parametrize("seed", [20260917, 11, 4242])
def test_sphere_every_returned_ray_reaches_its_point(seed):
    """Vector-Snell round trip: each ray of project() must land on X."""
    rng = np.random.default_rng(seed)
    centre = np.zeros(3)
    pinhole = np.array([REF_RP, 0.0, 0.0])
    n_rays = 0
    for _ in range(150):
        r_x = rng.uniform(1.0, 93.0)
        theta, phi = rng.uniform(0.0, math.pi), rng.uniform(0.0, 2 * math.pi)
        point = centre + r_x * np.array(
            [math.cos(theta), math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi)]
        )
        for branch, _lam, direction in ps.project(REF_N, REF_R, centre, pinhole, point):
            if branch == "ring":
                continue
            assert _sphere_miss(REF_N, REF_R, centre, pinhole, point, direction) < TRACE_TOL_MM
            n_rays += 1
    assert n_rays > 100, f"suspiciously few rays traced ({n_rays})"


def test_sphere_axial_line_gives_ring_only_for_the_ball_lens():
    """Section ring: the reference port has no ring on the axial line; the ball lens does."""
    centre = np.zeros(3)
    ref = ps.project(
        BALL_N, BALL_R, centre, np.array([BALL_RP, 0.0, 0.0]), np.array([-49.0, 0.0, 0.0])
    )
    assert [b for b, _, _ in ref] == ["far", "ring"]

    weak = ps.project(
        [1.0, 1.33], BALL_R, centre,
        np.array([BALL_RP, 0.0, 0.0]), np.array([-49.0, 0.0, 0.0]),
    )
    assert [b for b, _, _ in weak] == ["far"]


def test_sphere_point_between_centre_and_pinhole_is_the_radial_ray():
    """Section ring at Delta phi = 0: the answer is Lambda = 0, straight in."""
    centre = np.zeros(3)
    rays = ps.project(
        REF_N, REF_R, centre, np.array([REF_RP, 0.0, 0.0]), np.array([40.0, 0.0, 0.0])
    )
    assert len(rays) == 1
    branch, lam, direction = rays[0]
    assert (branch, lam) == ("near", 0.0)
    np.testing.assert_allclose(direction, [-1.0, 0.0, 0.0], atol=1e-14)


def test_huge_sphere_tends_to_the_flat_plate():
    """Section remaining-work: the plate is the large-radius limit of the sphere."""
    big = 1.0e7
    n, radii = [1.0, 1.49, 1.33], [big, big - 6.0]
    centre = np.zeros(3)
    pinhole = np.array([big + REF_RP, 0.0, 0.0])
    point = np.array([big - 6.0 - 50.0, 30.0, 0.0])
    rays = ps.project(n, radii, centre, pinhole, point)
    assert len(rays) == 1
    assert _sphere_miss(n, radii, centre, pinhole, point, rays[0][2]) < 1e-8


# ------------------------------------------------------------------- cylinder
@pytest.mark.parametrize("seed", [7, 2718])
def test_cylinder_every_returned_ray_reaches_its_point(seed):
    """The same round trip for the cylinder, over both branches and both senses."""
    rng = np.random.default_rng(seed)
    pinhole = np.array([REF_RP, 0.0, 0.0])
    n_rays = 0
    for _ in range(40):
        r_x = rng.uniform(5.0, 93.0)
        azimuth = rng.uniform(-math.pi, math.pi)
        point = np.array(
            [r_x * math.cos(azimuth), r_x * math.sin(azimuth), rng.uniform(-200.0, 200.0)]
        )
        for branch, lam, c_z, direction in pc.project(REF_N, REF_R, pinhole, point):
            # trace_to_radius returns the crossing of radius r_X itself, and the
            # branch says which of the two crossings the ray claims.
            hit = pc.trace_to_radius(
                pinhole, direction, REF_N, REF_R, r_x, far=(branch == "far")
            )
            assert hit is not None, (
                f"{branch} ray (Lambda={lam:g}, C_z={c_z:g}) never reaches r_X = {r_x:g}"
            )
            assert np.linalg.norm(hit - point) < TRACE_TOL_MM, (
                f"{branch} ray misses by {np.linalg.norm(hit - point):.3e} mm"
            )
            n_rays += 1
    assert n_rays > 30, f"suspiciously few rays traced ({n_rays})"
