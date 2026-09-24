"""Forward projection of a 3-D point through concentric spherical interfaces.

Implements the algorithm of Section spherealg of doc/refract_curved.tex.  It is
deliberately thin: in its ray plane a sphere *is* the cylinder at C_z = 0
(Section sphere_action), so every closed form is imported from
``project_cylinder`` rather than restated.  What is added here is only what the
sphere does not share with the cylinder:

  * the ray plane itself, formed from O, P_0 and X (Section rayplane), which
    reduces the two unknowns of the cylinder to the single unknown Lambda;
  * the axial-ring degeneracy on the line through O and P_0 (Section ring);
  * the wrapped targets, enumerated up to M(0) rather than decided by the
    cylinder's test (wraptest) (Section wrap).

``project()`` returns every ray, as ``(branch, Lambda_signed, direction)`` with
the direction a unit vector at the pinhole, matching the tuple shape of
``project_cylinder.project()`` minus its C_z entry.  The sign of Lambda carries
the sense of rotation in the ray plane, towards +e2 when positive.

    python project_sphere.py        self-test against a brute-force ray tracer
"""
import math

import numpy as np

from project_cylinder import (
    Lmax_kstar,
    Phi_Z,
    _bisect,
    dPhi_dLam,
    far_roots,
    far_turns,
    lam_at,
    refract,
)

_EPS = np.finfo(float).eps


# ------------------------------------------------------------- the ray plane
def ray_plane(O, P0, X):
    """Section rayplane: (r_P, r_X, Delta phi, e1, e2), the plane through O, P_0 and X.

    e1 points from the centre to the pinhole and e2 completes the plane on the
    side of X, so Delta phi lies in [0, pi] and a ray of positive Lambda sweeps
    towards X.  When O, P_0 and X are collinear the plane is undetermined and e2
    is returned as None; Delta phi is then 0 or pi and Section ring applies.
    """
    O, P0, X = (np.asarray(v, float) for v in (O, P0, X))
    u, v = P0 - O, X - O
    rP, rX = np.linalg.norm(u), np.linalg.norm(v)
    if rP <= 0.0:
        raise ValueError("pinhole is at the centre of the sphere")
    e1 = u / rP
    if rX <= 0.0:                                  # X at the centre: the radial ray
        return rP, 0.0, 0.0, e1, None
    w = v - np.dot(v, e1) * e1
    nw = np.linalg.norm(w)
    if nw <= 1e-12 * rX:                           # collinear: Section ring
        return rP, rX, (0.0 if np.dot(v, e1) > 0 else np.pi), e1, None
    e2 = w / nw
    return rP, rX, float(np.arctan2(nw, np.dot(v, e1))), e1, e2


def direction_at_pinhole(Lam, n1, rP, e1, e2):
    """Step (4): the ray direction at the pinhole, in the ray plane.

    At C_z = 0 the cylinder's nu_1 is n_1, so the direction is the unit vector at
    angle alpha to the inward radial -e1, with sin alpha = Lambda/(n_1 r_P).  The
    sign of Lambda is the sense.  This is direction_from_invariants() of
    project_cylinder with C_z = 0 and (rhat, phat) = (e1, e2).
    """
    sa = np.clip(Lam / (n1 * rP), -1.0, 1.0)
    ca = np.sqrt(max(1.0 - sa * sa, 0.0))
    if e2 is None:                                 # radial ray; any e2 gives the same vector
        return -ca * np.asarray(e1, float)
    return -ca * np.asarray(e1, float) + sa * np.asarray(e2, float)


# ------------------------------------------------------------------ the reach
def chord_bound(n, R, rP):
    """Property chord: no ray sweeps more than sum_{k<N} arccos(R_k/R_{k-1}) + pi."""
    return sum(np.arccos(ri / ro) for ro, ri in zip([rP] + list(R), R)) + np.pi


def sweeps(n, R, rP, rX):
    """(Phi_near(Lmax), M(0)) of step (2): the near-branch reach and the largest sweep.

    Phi_far starts at pi and turns where far_turns() says, so its supremum is the
    largest of pi, its turning values and its value at the edge of the domain.
    M(0) is the larger of that and the near-branch reach.
    """
    Lm = Lmax_kstar(n, R, rX, 0.0)[0]
    edge = Lm                                      # at Lmax itself: Phi_Z clamps the grazing medium's rounding
    near = Phi_Z(edge, 0.0, n, R, rP, rX)
    near = 0.0 if near is None else near[0]
    vals = [np.pi, Phi_Z(edge, 0.0, n, R, rP, rX, far=True)[0]]
    vals += [Phi_Z(L, 0.0, n, R, rP, rX, far=True)[0]
             for L in far_turns(0.0, n, R, rP, rX) if 0.0 < L < edge]
    return float(near), float(max(max(vals), near))


def _targets(dphi, n, R, rP, rX):
    """Step (3): (target, sense) for the short way and every wrap that M(0) allows.

    Delta phi is kept, 2 pi - Delta phi is swept the other way, Delta phi + 2 pi
    the first way again, and so on while the target stays below both M(0) and the
    chord bound (Property chord).  At Delta phi = 0 or pi the two senses coincide,
    so each target is emitted once.
    """
    _, M0 = sweeps(n, R, rP, rX)
    cap = min(M0, chord_bound(n, R, rP))
    one_sense = dphi <= 8 * _EPS or abs(dphi - np.pi) <= 8 * _EPS
    out, m = [(dphi, 1.0)], 0
    if dphi > cap:
        return []
    while True:
        nxt = [(2 * np.pi * (m + 1) - dphi, -1.0), (dphi + 2 * np.pi * (m + 1), 1.0)]
        added = False
        for t, s in nxt:
            if t <= cap and not (one_sense and s < 0):
                out.append((t, s))
                added = True
        if not added:
            return out
        m += 1


# ------------------------------------------------------------------- the ring
def ring(n, R, rP, rX, tol=1e-12):
    """Section ring: the second root of Phi_far = pi on the axial line, or None.

    On the line through O and P_0 at Delta phi = pi the ray plane is undetermined,
    so a root at Lambda > 0 is swept at every azimuth at once and the image is a
    circle, not a point.  Lambda = 0 (the ray straight through the centre) is the
    first root and is returned separately by project().
    """
    roots = [L for L, _, _ in far_roots(0.0, n, R, rP, rX, np.pi) if L > tol]
    return float(roots[0]) if roots else None


# -------------------------------------------------------------- the projector
def project(n, R, O, P0, X, tol=1e-9):
    """Steps (1)-(4): every ray from the pinhole P0 to X, as (branch, Lambda, direction).

    Near branch: Phi_near increases from 0 to its reach, so a target at most the
    reach has exactly one root (Remark newton), found by bisection on the
    monotone arc and polished by Newton.  Far branch: far_roots() cuts [0, Lmax)
    at the turning points that far_turns() brackets and takes the one root of each
    monotone arc.  Both are solved for every target of _targets().

    On the axial line the answer is Section ring's: the radial ray at Delta phi =
    0, or the central ray plus possibly a ring at Delta phi = pi.  A ring is
    returned as ("ring", Lambda, None), since it is a set of rays, not one.

    Every length is divided by u = 2^floor(log2 r_P) on entry and Lambda is
    multiplied by u on exit, as in project_cylinder.project(), so the
    tolerances are dimensionless and a rig gives the same rays in metres and in
    millimetres. The division by a power of two is exact.
    """
    O, P0, X = (np.asarray(v, float) for v in (O, P0, X))
    rP = float(np.linalg.norm(P0 - O))
    unit = math.ldexp(1.0, math.frexp(rP)[1] - 1) if rP > 0 else 1.0   # ray_plane() rejects r_P = 0
    out = _project_unit(list(map(float, n)), [float(r) / unit for r in R],
                        (P0 - O) / unit, (X - O) / unit, tol)
    return [(b, L * unit, d) for b, L, d in out]


def _project_unit(n, R, P0, X, tol):
    """project() with the centre at the origin and every length in units of u = 2^floor(log2 r_P)."""
    rP, rX, dphi, e1, e2 = ray_plane(np.zeros(3), P0, X)
    if R and not (rP > R[0]):
        raise ValueError("the pinhole is not outside the outermost interface")
    if rX <= 0.0:                                          # X at the centre
        return [("near", 0.0, direction_at_pinhole(0.0, n[0], rP, e1, e2))]
    if R and not (R[-1] > rX):
        raise ValueError(f"the point is not inside the innermost interface (r_X / R = {rX / R[-1]:.9g})")

    out = []
    axial = e2 is None
    if axial and dphi <= 8 * _EPS:                         # X between O and P_0
        return [("near", 0.0, direction_at_pinhole(0.0, n[0], rP, e1, e2))]
    if axial:                                              # X beyond O: through the centre
        out.append(("far", 0.0, direction_at_pinhole(0.0, n[0], rP, e1, e2)))
        Lr = ring(n, R, rP, rX)
        if Lr is not None:
            out.append(("ring", Lr, None))
        return out

    near_reach, _ = sweeps(n, R, rP, rX)
    for target, sense in _targets(dphi, n, R, rP, rX):
        if target <= near_reach:
            L = lam_at(0.0, n, R, rP, rX, target)
            if L is not None:
                L = _polish(L, 0.0, n, R, rP, rX, target, far=False)
                if abs(Phi_Z(L, 0.0, n, R, rP, rX)[0] - target) < tol:
                    out.append(("near", sense * L,
                                direction_at_pinhole(sense * L, n[0], rP, e1, e2)))
        for L, _slope, _z in far_roots(0.0, n, R, rP, rX, target):
            L = _polish(L, 0.0, n, R, rP, rX, target, far=True)
            if abs(Phi_Z(L, 0.0, n, R, rP, rX, far=True)[0] - target) < tol:
                out.append(("far", sense * L,
                            direction_at_pinhole(sense * L, n[0], rP, e1, e2)))
    return out


def _polish(Lam, Cz, n, R, rP, rX, target, far, itmax=30, tol=1e-14):
    """Newton on Phi(Lambda) = target from a bracketed start, clamped into the domain."""
    Lm = Lmax_kstar(n, R, rX, Cz)[0]
    L = min(max(Lam, 0.0), Lm * (1 - 1e-15))
    for _ in range(itmax):
        v = Phi_Z(L, Cz, n, R, rP, rX, far=far)
        if v is None:
            return L
        g = v[0] - target
        if abs(g) < tol:
            return L
        d = dPhi_dLam(L, Cz, n, R, rP, rX, far=far)
        if not np.isfinite(d) or d == 0.0:
            return L
        step = -g / d
        Ln = L + step
        if not (0.0 <= Ln < Lm):                    # clamp, halving towards the domain
            for _ in range(60):
                step *= 0.5
                Ln = L + step
                if 0.0 <= Ln < Lm:
                    break
            else:
                return L
        L = Ln
    return L


# ------------------------------------------------------------ brute-force check
def trace(P0, d, n, R, O, kind_tol=1e-12):
    """Vector-Snell trace of a ray through the concentric spheres, for validation.

    Mirrors project_cylinder.trace_to_radius() but intersects spheres: the ray is
    advanced to each interface in turn, refracted there, and the point of entry
    into the innermost medium is returned with the direction it carries.
    """
    O = np.asarray(O, float)
    p, dd = np.asarray(P0, float) - O, np.asarray(d, float) / np.linalg.norm(d)
    for k, Rk in enumerate(R):
        b = np.dot(p, dd)
        c = np.dot(p, p) - Rk * Rk
        disc = b * b - c
        if disc <= 0.0:
            return None, None
        t = -b - np.sqrt(disc)                     # the near intersection, entering
        if t <= kind_tol:
            return None, None
        p = p + t * dd
        nrm = p / np.linalg.norm(p)                # outward normal
        dd = refract(dd, nrm, n[k], n[k + 1])
        if dd is None:
            return None, None
    return p + O, dd


def _selftest():
    """Round-trip every returned ray through the tracer and measure where it lands."""
    P = lambda *a: print(*a, flush=True)
    rng = np.random.default_rng(20260917)

    def miss(n, R, O, P0, X, branch, Lam, d):
        """Distance from X to the traced ray's line inside the innermost medium."""
        q, dd = trace(P0, d, n, R, O)
        if q is None:
            return np.inf
        w = np.asarray(X, float) - q
        return float(np.linalg.norm(w - np.dot(w, dd) * dd))

    P("sphere: reference port as a sphere, n = (1, 1.49, 1.33), R = (100, 94) mm")
    n, R, O = [1.0, 1.49, 1.33], [100.0, 94.0], np.zeros(3)
    P0 = np.array([350.0, 0.0, 0.0])
    worst, count, rings = 0.0, 0, 0
    for _ in range(400):
        rX = rng.uniform(1.0, 93.0)
        th = rng.uniform(0.0, np.pi)
        ph = rng.uniform(0.0, 2 * np.pi)
        X = O + rX * np.array([np.cos(th),
                               np.sin(th) * np.cos(ph),
                               np.sin(th) * np.sin(ph)])
        for branch, Lam, d in project(n, R, O, P0, X):
            if branch == "ring":
                rings += 1
                continue
            e = miss(n, R, O, P0, X, branch, Lam, d)
            worst = max(worst, e)
            count += 1
    P(f"    {count} rays traced, worst miss {worst:.3e} mm, {rings} rings")

    P("sphere: the flat-plate sanity check, a huge sphere is nearly a plate")
    n2, R2 = [1.0, 1.49, 1.33], [1.0e7, 1.0e7 - 6.0]
    P02 = np.array([1.0e7 + 350.0, 0.0, 0.0])
    X2 = np.array([1.0e7 - 6.0 - 50.0, 30.0, 0.0])
    rays = project(n2, R2, np.zeros(3), P02, X2)
    P(f"    {len(rays)} ray(s); worst miss "
      f"{max(miss(n2, R2, np.zeros(3), P02, X2, *r) for r in rays):.3e} mm")

    P("sphere: the axial line, Delta phi = pi (Section ring)")
    for n2v in (1.33, 2.00):
        nr, Rr = [1.0, n2v], [50.0]
        P0r, Xr = np.array([500.0, 0.0, 0.0]), np.array([-49.0, 0.0, 0.0])
        got = project(nr, Rr, np.zeros(3), P0r, Xr)
        kinds = [b for b, _, _ in got]
        P(f"    n_2 = {n2v}: {kinds}")


if __name__ == "__main__":
    _selftest()
