"""Forward projection of a 3-D point through coaxial cylindrical interfaces.

Implements the algorithm (Section cylalg) of doc/refract_curved.tex on both
branches.  Near branch: classify by the reach and zeta (Property zeta), then
solve the 2-D system (Phi, Z) = (Delta phi, Delta z) as the convex program of
Property consequences(d).  Far branch: search boxes in (theta*, C_z), Lambda = Lmax sin(theta*),
dropping those whose bounds (Property farbox) exclude the point and certifying the rest by Krawczyk's
test, so that no ray is lost (Properties farbox and zbound), then polish by Newton.  project() returns
every ray, including those that reach the point the long way round (Section
"When the sweep exceeds pi"), and rebuilds each ray direction at the pinhole.

The radial action of eq. (W) is elementary,
    int sqrt(nu^2 - Lam^2/r^2) dr = nu [ sqrt(r^2-b^2) - b*arccos(b/r) ],  b = Lam/nu,
so no quadrature is needed anywhere.

Running this file checks the whole chain against a brute-force vector-Snell ray
tracer: solve, rebuild the direction, trace it, and measure how far the traced
ray lands from the point it was supposed to reach.

    python project_cylinder.py            self-test
    python project_cylinder.py images     how many images points have (Section images of the notes):
                                          a census of traced rays, and the maps of Figure imgmap
    python project_cylinder.py sampling   what step (3) of the algorithm can miss (Section cylalg of the notes)
"""
import math

import numpy as np

rng = np.random.default_rng(2718)
P = lambda *a: print(*a, flush=True)

nus = lambda n, Cz: np.sqrt(np.asarray(n, float) ** 2 - Cz ** 2)
q = lambda a, L: 1.0 / np.sqrt(a * a - L * L)
# At Lambda = Lmax the binding medium's impact parameter b = Lambda/nu equals its radius, but rounding can put it a few
# ulp above; such a b is the grazing ray, not a ray that misses, so it is clamped to the radius (see Phi_Z).
_GRAZE = 8 * np.finfo(float).eps


# ------------------------------------------------------------ closed forms
def Lmax_kstar(n, R, rX, Cz):
    nu = nus(n, Cz)
    v = [nu[k] * r for k, r in enumerate(list(R) + [rX])]
    return min(v), int(np.argmin(v)) + 1


def Phi_Z(Lam, Cz, n, R, rP, rX, far=False):
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    phi = z = 0.0
    for k in range(len(n)):
        b = Lam / nu[k]
        if b > Rin[k]:
            if b > Rin[k] * (1 + _GRAZE):
                return None
            b = Rin[k]                            # Lambda = Lmax to rounding: the grazing ray of this medium
        a1 = np.arcsin(np.clip(b / Rin[k], -1, 1))
        a2 = np.arcsin(np.clip(b / Rout[k], -1, 1))
        s1 = np.sqrt(max(Rout[k] ** 2 - b * b, 0.0))
        s2 = np.sqrt(max(Rin[k] ** 2 - b * b, 0.0))
        if far and k == len(n) - 1:
            phi += np.pi - a1 - a2
            z += Cz * (s1 + s2) / nu[k]
        else:
            phi += a1 - a2
            z += Cz * (s1 - s2) / nu[k]
    return phi, z


def W_closed(Lam, Cz, n, R, rP, rX, far=False):
    """Radial action in elementary form."""
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    F = lambda r, b, v: v * (np.sqrt(max(r * r - b * b, 0.0)) - b * np.arccos(np.clip(b / r, -1, 1)))
    tot = 0.0
    for k in range(len(n)):
        b = Lam / nu[k]
        if b > Rin[k]:
            if b > Rin[k] * (1 + _GRAZE):
                return None
            b = Rin[k]                            # as in Phi_Z
        if far and k == len(n) - 1:
            tot += (F(Rout[k], b, nu[k]) - F(b, b, nu[k])) + (F(Rin[k], b, nu[k]) - F(b, b, nu[k]))
        else:
            tot += F(Rout[k], b, nu[k]) - F(Rin[k], b, nu[k])
    return tot


def dPhi_dLam(Lam, Cz, n, R, rP, rX, far=False):
    nu = nus(n, Cz)
    S = sum(q(nu[k] * R[k], Lam) - q(nu[k + 1] * R[k], Lam) for k in range(len(R)))
    return S + (-1.0 if far else 1.0) * q(nu[-1] * rX, Lam) - q(nu[0] * rP, Lam)


def dPhi_dCz(Lam, Cz, n, R, rP, rX, far=False):
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    v = Cz * Lam * sum((q(nu[k] * Rin[k], Lam) - q(nu[k] * Rout[k], Lam)) / nu[k] ** 2
                       for k in range(len(n)))
    if far:
        v -= 2 * Cz * Lam * q(nu[-1] * rX, Lam) / nu[-1] ** 2
    return v


def dZ_dCz(Lam, Cz, n, R, rP, rX, far=False, h=1e-7):
    """Central difference; one-sided where Lambda is so close to Lmax that one side leaves the domain (the tilt that
    lowers nu_k lowers Lmax), None only if both sides do."""
    a = Phi_Z(Lam, Cz + h, n, R, rP, rX, far)
    b = Phi_Z(Lam, Cz - h, n, R, rP, rX, far)
    if a is not None and b is not None:
        return (a[1] - b[1]) / (2 * h)
    c = Phi_Z(Lam, Cz, n, R, rP, rX, far)
    if c is None or (a is None and b is None):
        return None
    return (a[1] - c[1]) / h if a is not None else (c[1] - b[1]) / h


def in_domain(Lam, Cz, n, R, rX):
    if abs(Cz) >= min(n) - 1e-12:
        return False
    Lm, _ = Lmax_kstar(n, R, rX, Cz)
    return 0.0 <= Lam < Lm


# ------------------------------------------------------------ the solver
def solve(n, R, rP, rX, dphi, dz, Lam0, Cz0, far=False, itmax=80, tol=1e-10):
    """Damped Newton on G = -W - Lam*dphi - Cz*dz  (grad G = (Phi-dphi, Z-dz))."""
    def G(L, C):
        w = W_closed(L, C, n, R, rP, rX, far)
        return None if w is None else -w - L * dphi - C * dz

    L, C = Lam0, Cz0
    if not in_domain(L, C, n, R, rX):
        Lm, _ = Lmax_kstar(n, R, rX, C)
        L = min(max(L, 0.0), 0.5 * Lm)
    for it in range(itmax):
        pz = Phi_Z(L, C, n, R, rP, rX, far)
        if pz is None:
            return None, it, np.inf
        g = np.array([pz[0] - dphi, pz[1] - dz])
        if np.linalg.norm(g) < tol:
            return (L, C), it, np.linalg.norm(g)
        j11 = dPhi_dLam(L, C, n, R, rP, rX, far)
        j12 = dPhi_dCz(L, C, n, R, rP, rX, far)
        j22 = dZ_dCz(L, C, n, R, rP, rX, far)
        if j22 is None:                           # no Newton step: the caller judges the iterate by its residual
            return (L, C), it, np.linalg.norm(g)
        J = np.array([[j11, j12], [j12, j22]])
        try:
            p = -np.linalg.solve(J, g)
        except np.linalg.LinAlgError:
            return (L, C), it, np.linalg.norm(g)
        g0 = G(L, C)
        slope = np.dot(p, g)
        # G is stationary at the solution, so its value saturates while |g| is
        # still ~sqrt(eps); below that floor an Armijo test on G is meaningless
        # and the full Newton step is taken.
        if abs(slope) < 1e-14 * (1.0 + abs(g0)):
            Ln, Cn = L + p[0], C + p[1]
            if in_domain(Ln, Cn, n, R, rX):
                L, C = Ln, Cn
                continue
        s = 1.0
        for _ in range(60):                       # backtracking, staying inside D
            Ln, Cn = L + s * p[0], C + s * p[1]
            if in_domain(Ln, Cn, n, R, rX):
                gn = G(Ln, Cn)
                if gn is not None and gn <= g0 + 1e-4 * s * np.dot(p, g):
                    break
            s *= 0.5
        else:
            return (L, C), it, np.linalg.norm(g)
        L, C = L + s * p[0], C + s * p[1]
    pz = Phi_Z(L, C, n, R, rP, rX, far)
    return (L, C), itmax, (np.inf if pz is None else
                           np.linalg.norm([pz[0] - dphi, pz[1] - dz]))


# ------------------------------------------------------------ ray tracing
def refract(d, nrm, na, nb):
    c = -np.dot(nrm, d)
    if c < 0:
        nrm, c = -nrm, -c
    e = na / nb
    k = 1 - e * e * (1 - c * c)
    return None if k < 0 else e * d + (e * c - np.sqrt(k)) * nrm


def trace_to_radius(P0, d, n, R, rX, far):
    """Trace inward and return the crossing of radius rX in the last medium."""
    p, dd = np.array(P0, float), np.array(d, float)
    for k, Rk in enumerate(R):
        a = dd[0] ** 2 + dd[1] ** 2
        b = 2 * (p[0] * dd[0] + p[1] * dd[1])
        c = p[0] ** 2 + p[1] ** 2 - Rk ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        ts = sorted(t for t in ((-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)) if t > 1e-12)
        if not ts:
            return None
        p = p + ts[0] * dd
        nrm = np.array([p[0], p[1], 0.0])
        nrm /= np.linalg.norm(nrm)
        dd = refract(dd, nrm, n[k], n[k + 1])
        if dd is None:
            return None
    a = dd[0] ** 2 + dd[1] ** 2
    b = 2 * (p[0] * dd[0] + p[1] * dd[1])
    c = p[0] ** 2 + p[1] ** 2 - rX ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t1 = (-b - np.sqrt(disc)) / (2 * a)
    t2 = (-b + np.sqrt(disc)) / (2 * a)
    t = t2 if far else t1
    return None if t <= 1e-12 else p + t * dd


def trace_many(n, R, rP, rX, C, sa):
    """Vectorised vector-Snell trace of rays from P0 = (r_P, 0, 0), independent of the closed forms.

    The ray in medium 1 has d_z = C/n_1 and its transverse part at angle alpha from the inward radial, sin(alpha) = sa.
    Returns, for the near and the far crossing of r = r_X, the azimuth swept (summed segment by segment, so it may
    exceed pi) and the axial offset, NaN where the ray is reflected or misses r_X."""
    C, sa = (np.ravel(np.asarray(v, float)) for v in np.broadcast_arrays(C, sa))
    m = C.size
    dp = np.sqrt(np.maximum(1.0 - (C / n[0]) ** 2, 0.0))
    d = np.stack([-dp * np.sqrt(np.maximum(1.0 - sa * sa, 0.0)), dp * sa, C / n[0]], 1)
    p = np.tile([rP, 0.0, 0.0], (m, 1))
    ok, sweep = np.ones(m, bool), np.zeros(m)
    turn = lambda a, b: np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])

    def hit(p, d, r, far):
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (p[:, 0] * d[:, 0] + p[:, 1] * d[:, 1])
        disc = b * b - 4 * a * (p[:, 0] ** 2 + p[:, 1] ** 2 - r * r)
        t = (-b + (1 if far else -1) * np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
        return p + t[:, None] * d, (disc >= 0) & (t > 1e-15)

    for k, Rk in enumerate(R):
        q, okk = hit(p, d, Rk, False)
        ok &= okk
        sweep += np.where(ok, turn(p, q), 0.0)
        p = np.where(ok[:, None], q, p)
        nrm = np.stack([p[:, 0], p[:, 1], np.zeros(m)], 1) / np.hypot(p[:, 0], p[:, 1])[:, None]
        c = -(nrm * d).sum(1)
        nrm, c = np.where((c < 0)[:, None], -nrm, nrm), np.abs(c)
        e = n[k] / n[k + 1]
        kk = 1 - e * e * (1 - c * c)
        ok &= kk >= 0
        d = e * d + (e * c - np.sqrt(np.maximum(kk, 0.0)))[:, None] * nrm
    out = {}
    for name, far in (("near", False), ("far", True)):
        q, okk = hit(p, d, rX, far)
        good = ok & okk
        out[name] = (np.where(good, sweep + turn(p, q), np.nan), np.where(good, q[:, 2], np.nan))
    return out


def direction_from_invariants(P0, Lam, Cz, n1, nu1):
    """Step (5) of the algorithm: rebuild d at the pinhole."""
    rP = np.hypot(P0[0], P0[1])
    rhat = np.array([P0[0] / rP, P0[1] / rP, 0.0])
    phat = np.array([-P0[1] / rP, P0[0] / rP, 0.0])
    sa = np.clip(Lam / (nu1 * rP), -1.0, 1.0)
    ca = np.sqrt(max(1.0 - sa * sa, 0.0))
    dperp = (nu1 / n1) * (-ca * rhat + sa * phat)
    return dperp + np.array([0.0, 0.0, Cz / n1])




# ------------------------------------------------------- visibility (step 3)
def lam_at(Cz, n, R, rP, rX, dphi, iters=90):
    """Bisection for Phi_near(Lambda; Cz) = dphi.  None if out of reach at this Cz.

    Relies only on Property convexnear: Phi_near is strictly increasing in Lambda.  The reach is taken at Lmax itself,
    and the bisection runs in s with Lambda = Lmax (1 - s^2).  Phi_near has a square-root edge at Lmax, so a margin of
    m in Lambda would cost about sqrt(m) in Phi (4.5e-7 rad for m = 1e-13 at the reference port) and hide every tilt
    whose reach exceeds dphi by less; in s the edge is regular.  The result stays strictly below Lmax, inside the domain
    of solve()."""
    if abs(Cz) >= min(n):
        return None
    Lm, _ = Lmax_kstar(n, R, rX, Cz)
    top = Phi_Z(Lm, Cz, n, R, rP, rX)
    if top is None or top[0] < dphi:
        return None
    lo, hi = 0.0, 1.0                             # s; Phi_near falls as s grows, from the reach at s = 0
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        v = Phi_Z(Lm * (1 - m * m), Cz, n, R, rP, rX)
        lo, hi = (lo, m) if (v is not None and v[0] < dphi) else (m, hi)
    s = 0.5 * (lo + hi)
    return min(Lm * (1 - s * s), np.nextafter(Lm, 0.0))


def switch_tilts(n, R, rX, chi):
    """Tilts in (0, chi) at which two caps nu_k R_k of (adm) are equal (R_N = r_X); k* is fixed between them."""
    caps = list(R) + [rX]
    out = set()
    for j in range(len(n)):
        for k in range(j + 1, len(n)):
            den = caps[j] ** 2 - caps[k] ** 2
            if den != 0.0:
                c2 = (n[j] ** 2 * caps[j] ** 2 - n[k] ** 2 * caps[k] ** 2) / den
                if 0.0 < c2 < chi * chi:
                    out.add(float(np.sqrt(c2)))
    return sorted(out)


def reach_terms(Cz, n, R, rP, rX):
    """The reach Phi_near(Lmax(C_z); C_z) medium by medium; each term is monotone in C_z while k* is fixed."""
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    Lm = min(nu[k] * Rin[k] for k in range(len(n)))
    return np.array([np.arcsin(min(Lm / (nu[k] * Rin[k]), 1.0)) - np.arcsin(Lm / (nu[k] * Rout[k]))
                     for k in range(len(n))])


def components(n, R, rP, rX, dphi, chi=None, tol=1e-13):
    """Connected pieces of {C_z in [0, chi] : reach(C_z) >= dphi}, found exactly (Property reachpieces).

    The range is cut at the tilts where k* can switch.  On a piece every medium's term of the reach is monotone
    in C_z, so on [a, b] the reach lies between sum_k min(T_k(a), T_k(b)) and sum_k max(T_k(a), T_k(b)): an
    interval is dropped when the upper bound is below dphi, kept when the lower bound reaches it, and halved
    otherwise.  An interval still undecided at width tol is kept, so no component is lost; a spurious sliver
    only costs classify() a failed look for zeta."""
    chi = min(n) * (1 - 1e-9) if chi is None else chi
    cuts = [0.0] + switch_tilts(n, R, rX, chi) + [chi]
    kept = []
    for a0, b0 in zip(cuts[:-1], cuts[1:]):
        stack = [(a0, b0, reach_terms(a0, n, R, rP, rX), reach_terms(b0, n, R, rP, rX))]
        while stack:
            a, b, ta, tb = stack.pop()
            if np.maximum(ta, tb).sum() < dphi:
                continue
            if np.minimum(ta, tb).sum() >= dphi or b - a < tol:
                kept.append((a, b))
                continue
            m = 0.5 * (a + b)
            tm = reach_terms(m, n, R, rP, rX)
            stack += [(m, b, tm, tb), (a, m, ta, tm)]
    out = []
    for a, b in sorted(kept):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def classify(n, R, rP, rX, dphi, dz, iters=60):
    """Step (3): is (dphi, dz) reachable on the near branch?  Seed (Lam, Cz) or None.

    zeta(Cz) = Z(Lambda(Cz), Cz) is strictly increasing on each component
    (Property zeta: zeta' = det J / Phi_Lambda > 0), so one bisection per component
    is both sound and complete there.  At a component end where the reach is dphi,
    Lambda(Cz) = Lmax: the ray is the grazing ray, and zeta there is its axial offset
    Z(Lmax, Cz) in closed form.  The bisection therefore brackets the whole component.
    (Approaching the ends from inside instead lost the rays of tilts next to an end,
    within about 1e-10 of grazing: evidence E1 of the paper, and the tests in
    code/tests/classify_grazing/.)"""
    sgn = 1.0 if dz >= 0 else -1.0                # zeta is odd in C_z
    target = abs(dz)

    def zeta(c):
        """zeta at c; at a tilt whose reach is dphi to rounding, the grazing ray's Z(Lmax, c)."""
        v = zeta_at(c, n, R, rP, rX, dphi)
        if v is None:
            pz = Phi_Z(Lmax_kstar(n, R, rX, c)[0], c, n, R, rP, rX)
            if pz is not None and pz[0] >= dphi - 1e-9:
                v = pz[1]
        return v

    def zin(c0, c1):
        """First finite zeta walking from c0 towards c1: the fallback for an end where zeta() has no value.

        C_z = 0 is a regular point with zeta(0) = 0, so it is used as it stands."""
        steps = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
        if c0 == 0.0:
            steps = (0.0,) + steps
        for t in steps:
            c = c0 + t * (c1 - c0)
            v = zeta(c)
            if v is not None:
                return c, v
        return None, None

    for (c1, c2) in components(n, R, rP, rX, dphi):
        if c2 <= c1:
            continue
        a, za = c1, zeta(c1)
        if za is None:
            a, za = zin(c1, c2)
        b, zb = c2, zeta(c2)
        if zb is None:
            b, zb = zin(c2, c1)
        if a is None or b is None or b <= a:
            continue
        if not (min(za, zb) <= target <= max(za, zb)):
            continue
        for _ in range(iters):
            mm = 0.5 * (a + b)
            v = zeta(mm)
            if v is None:
                break
            a, b = (mm, b) if (v - target) * (za - target) > 0 else (a, mm)
        c = 0.5 * (a + b)
        L = lam_at(c, n, R, rP, rX, dphi)
        if L is None:                             # c is within rounding of the end: the grazing ray
            Lm = Lmax_kstar(n, R, rX, c)[0]
            pz = Phi_Z(Lm, c, n, R, rP, rX)
            if pz is None or pz[0] < dphi - 1e-9:
                continue
            L = np.nextafter(Lm, 0.0)
        return L, sgn * c
    return None


def zeta_at(Cz, n, R, rP, rX, dphi):
    L = lam_at(Cz, n, R, rP, rX, dphi, iters=60)
    if L is None:
        return None
    v = Phi_Z(L, Cz, n, R, rP, rX)
    return None if v is None else v[1]



# ------------------------------------------------------- far branch (steps 3, 4)
def Z0(Cz, n, R, rP, rX):
    """Eq. (z0): the axial offset of the radial ray; no ray of tilt C_z has |Z| below it (Property zbound)."""
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    return abs(Cz) * sum((Rout[k] - Rin[k]) / nu[k] for k in range(len(n)))


def _bisect(f, lo, hi, it=70):
    """Bisection on a sign change of f in [lo, hi]; lo itself if f vanishes there."""
    f0 = f(lo)
    if f0 == 0:
        return lo
    up = f0 > 0
    for _ in range(it):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if (f(m) > 0) == up else (lo, m)
    return 0.5 * (lo + hi)


_ARC = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, 801))    # Lambda/Lmax, dense at both ends


def far_turns(Cz, n, R, rP, rX, tol=1e-14):
    """Every turning point of Phi_far(Lambda; C_z) on (0, Lmax), none missed (Section fold, eq. intervalbound).

    With t = Lambda^2, Phi_far' = F(t) = sum_j c_j (a_j^2 - t)^(-1/2) with the (c_j, a_j) of (Flist), and every
    term increases in t.  On [t1, t2], F lies between (positive terms at t1) + (negative terms at t2) and the
    reverse, so a cell whose bounds exclude 0 holds no turning point.  Beyond Lmax^2 - c_0^2/M^2 the terms at
    a_j = Lmax dominate and fix the sign.  Cells of a grid dense at both ends are decided at once; the rest are
    halved, and every sign change is refined to tol Lmax^2.  A pair of turning points closer than that is a
    tangency of Phi_far' with 0, whose arc sweeps nothing."""
    N = len(n)
    nu = nus(n, Cz)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    c = np.array([1.0 if k < N - 1 else -1.0 for k in range(N)] + [-1.0] * N)
    A = np.array([(nu[k] * Rin[k]) ** 2 for k in range(N)] + [(nu[k] * Rout[k]) ** 2 for k in range(N)])
    A0 = A.min()
    edge = A <= A0 * (1 + 1e-12)
    c0, M = c[edge].sum(), np.sum(np.abs(c[~edge]) / np.sqrt(A[~edge] - A0))
    t_end = A0 * (1 - 1e-14)
    if c0 != 0.0:
        t_end = min(t_end, A0 - (c0 / M) ** 2 if M > 0 else -1.0)
    if t_end <= 0.0:
        return []
    pos, neg = c > 0, c < 0
    Fv = lambda t: float(np.sum(c / np.sqrt(A - t)))

    def bounds(v1, v2):
        return v1[pos].sum() + v2[neg].sum(), v2[pos].sum() + v1[neg].sum()

    def halve(t1, t2, f1, f2):
        lo, hi = bounds(c / np.sqrt(A - t1), c / np.sqrt(A - t2))
        if lo > 0 or hi < 0:
            return []
        if t2 - t1 <= tol * A0:
            return [0.5 * (t1 + t2)] if (f1 > 0) != (f2 > 0) else []
        tm = 0.5 * (t1 + t2)
        fm = Fv(tm)
        return halve(t1, tm, f1, fm) + halve(tm, t2, fm, f2)

    t = t_end * _ARC ** 2
    V = c[None, :] / np.sqrt(A[None, :] - t[:, None])
    Fg = V.sum(1)
    lo = V[:-1][:, pos].sum(1) + V[1:][:, neg].sum(1)
    hi = V[1:][:, pos].sum(1) + V[:-1][:, neg].sum(1)
    out = []
    for i in np.flatnonzero(~((lo > 0) | (hi < 0))):
        out += halve(t[i], t[i + 1], Fg[i], Fg[i + 1])
    return [float(np.sqrt(x)) for x in out]


def far_roots(Cz, n, R, rP, rX, target):
    """Roots of Phi_far(Lambda; C_z) = target as (Lambda, slope sign of their arc, Z_far), by increasing Lambda.

    [0, Lmax) is split at the turning points of Phi_far, all of them located by far_turns(); an arc whose end
    values bracket the target holds exactly one root.  A root exactly at an end of an arc, such as Lambda = 0 at
    target pi, is counted once, as the root of the arc it starts.  The last arc ends at Lmax itself: a margin of m
    there would drop the roots of the targets within about sqrt(m) of Phi_far(Lmax), as in lam_at()."""
    Lm = Lmax_kstar(n, R, rX, Cz)[0]
    cuts = [0.0] + [L for L in far_turns(Cz, n, R, rP, rX) if 0.0 < L < Lm] + [Lm]
    phi = lambda L: Phi_Z(L, Cz, n, R, rP, rX, far=True)[0]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        fa, fb = phi(a), phi(b)
        if fa == target:
            L = a
        elif fb != target and (fa > target) != (fb > target):
            L = _bisect(lambda x: phi(x) - target, a, b)
        else:
            continue
        out.append((L, 1 if fb > fa else -1, Phi_Z(L, Cz, n, R, rP, rX, far=True)[1]))
    return out


def sweep_far(n, R, rP, rX, dphi, dz, samples=160, tol=1e-13):
    """The sampled sweep that classify_far() replaced, kept to show what sampling misses (python ... sampling).

    Tilts up to C_max, Z0(C_max) = |dz|, are sampled evenly in Z0.  Each root of Phi_far = dphi is followed across
    the samples: an interval over which the set of roots changes is halved until the change is isolated, and each
    sign change of zeta - |dz| is refined by bisection.  zeta is monotone along a decreasing arc (Property farfold),
    so nothing is missed there; on an increasing arc two rays closer than the sampling can be."""
    sgn, dz = (1.0 if dz >= 0 else -1.0), abs(dz)
    if dz == 0.0:
        return [(L, 0.0) for L, _, _ in far_roots(0.0, n, R, rP, rX, dphi)]
    z0 = lambda C: Z0(C, n, R, rP, rX)
    cmax = _bisect(lambda C: z0(C) - dz, 0.0, min(n) * (1 - 1e-15), it=200)
    tilts = ([0.0] + [_bisect(lambda C: z0(C) - dz * i / samples, 0.0, cmax, it=100) for i in range(1, samples)]
             + [cmax])
    roots = lambda C: far_roots(C, n, R, rP, rX, dphi)
    found = []

    def sweep(Ca, Sa, Cb, Sb):
        same = [k for _, k, _ in Sa] == [k for _, k, _ in Sb]
        if same and not any((x[2] >= dz) != (y[2] >= dz) for x, y in zip(Sa, Sb)):
            return
        if Cb - Ca < tol:
            for x in Sa:                             # pair each root with the nearest root of the same kind
                ys = [y for y in Sb if y[1] == x[1]]
                y = min(ys, key=lambda y: abs(y[0] - x[0])) if ys else None
                if y is not None and (x[2] >= dz) != (y[2] >= dz):
                    found.append((0.5 * (x[0] + y[0]), sgn * 0.5 * (Ca + Cb)))
            return
        Cm = 0.5 * (Ca + Cb)
        Sm = roots(Cm)
        sweep(Ca, Sa, Cm, Sm)
        sweep(Cm, Sm, Cb, Sb)

    states = [roots(C) for C in tilts]
    for i in range(samples):
        sweep(tilts[i], states[i], tilts[i + 1], states[i + 1])
    return found


# ---------------------------------------------- far branch across tilts: boxes (steps 3, 4)
# Lambda = Lmax(C_z) sin(theta*), eq. (thetastar).  Between two cuts (eq. cuts) k* is fixed, and every term of
# Phi_far and Z_far is monotone in theta* and in C_z, or a product of two nonnegative such factors, so the values at
# the corners of a box bound them over it (Property farbox).  The Jacobian in (theta*, C_z) is bounded term by term
# as well, and Krawczyk's test (eq. krawczyk) then shows that a box holds exactly one ray, or none.
_EPS = np.finfo(float).eps


def _far_stack(n, R, rP, rX):
    """Indices, inner and outer radii of the N media, and the medium k_j, radius rho_j and sign c_j of the 2N terms of
    (Flist)."""
    n = np.asarray(n, float)
    N = len(n)
    Rin, Rout = np.array(list(R) + [rX], float), np.array([rP] + list(R), float)
    med = np.repeat(np.arange(N), 2)
    rad = np.column_stack([Rin, Rout]).ravel()
    sig = np.column_stack([np.where(np.arange(N) < N - 1, 1.0, -1.0), -np.ones(N)]).ravel()
    return n, Rin, Rout, med, rad, sig


def box_bounds(st, ta, tb, ca, cb, ks, far=True):
    """Eq. (farbox): lower and upper bounds on Phi and Z of one branch over the boxes [ta, tb] x [ca, cb], k* = ks.

    b_k = R_k* (nu_k*/nu_k) sin(theta*) increases with theta* and is monotone in C_z between cuts; medium k's sweep
    and chord increase with b_k, except that medium N's decrease on the far branch; C_z/nu_k increases with C_z."""
    n, Rin, Rout = st[:3]
    i = np.arange(ca.size)
    nua, nub = np.sqrt(n ** 2 - ca[:, None] ** 2), np.sqrt(n ** 2 - cb[:, None] ** 2)
    ra, rb = Rin[ks][:, None] * nua[i, ks][:, None] / nua, Rin[ks][:, None] * nub[i, ks][:, None] / nub
    blo = np.minimum(np.sin(ta)[:, None] * np.minimum(ra, rb), Rin)
    bhi = np.minimum(np.sin(tb)[:, None] * np.maximum(ra, rb), Rin)
    sweep = lambda b: np.arcsin(b / Rin) - np.arcsin(b / Rout)
    chord = lambda b: np.sqrt(Rout ** 2 - b * b) - np.sqrt(np.maximum(Rin ** 2 - b * b, 0.0))
    plo, phi, slo, shi = sweep(blo), sweep(bhi), chord(blo), chord(bhi)
    fsweep = lambda b: np.pi - np.arcsin(b / Rin[-1]) - np.arcsin(b / Rout[-1])
    fchord = lambda b: np.sqrt(Rout[-1] ** 2 - b * b) + np.sqrt(np.maximum(Rin[-1] ** 2 - b * b, 0.0))
    if far:
        plo[:, -1], phi[:, -1] = fsweep(bhi[:, -1]), fsweep(blo[:, -1])
        slo[:, -1], shi[:, -1] = fchord(bhi[:, -1]), fchord(blo[:, -1])
    zlo, zhi = (ca[:, None] / nua * slo).sum(1), (cb[:, None] / nub * shi).sum(1)
    amp = 1.0 + (n ** 2 / nub ** 2).max(1)                  # rounding grows as nu -> 0
    pad = 64 * _EPS * amp
    return plo.sum(1) - pad * 10, phi.sum(1) + pad * 10, zlo * (1 - pad) - pad * 1e-3, zhi * (1 + pad) + pad * 1e-3


def _imul(alo, ahi, blo, bhi):
    p = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    bad = np.isnan(p).any(0)
    return np.where(bad, -np.inf, p.min(0)), np.where(bad, np.inf, p.max(0))


def far_box_jacobian(st, ta, tb, ca, cb, ks):
    """Eq. (farjac): bounds on [[dPhi/dtheta*, dPhi/dC_z], [dZ/dtheta*, dZ/dC_z]] of the far branch over the boxes.

    Term j of (Flist), with radius rho_j in medium k_j, kappa_j = Lmax/a_j and g_j = sqrt(1 - kappa_j^2 sin^2), gives
        dPhi/dtheta*:  c_j kappa_j cos/g_j                   dPhi/dC_z:  c_j kappa_j' sin/g_j
        dZ/dtheta*:    c_j rho_j (C_z/nu) kappa_j^2 sin cos/g_j
        dZ/dC_z:      -c_j rho_j [(n^2/nu^3) g_j - (C_z/nu) kappa_j kappa_j' sin^2/g_j],   n, nu of medium k_j,
    with kappa_j' = (R_k*/rho_j) C_z (n_k*^2 - n^2)/(nu_k* nu^3).  Every factor but c_j and kappa_j' is nonnegative
    and monotone on a box, so each is bounded by its end values.  The term that grazes, kappa_j = 1, gives c_j, 0,
    c_j rho_j (C_z/nu) sin and -c_j rho_j (n^2/nu^3) cos.  Returns lo, hi of shape (boxes, 2, 2)."""
    n, Rin, Rout, med, rad, sig = st
    K = ca.size
    nua, nub = np.sqrt(n ** 2 - ca[:, None] ** 2)[:, med], np.sqrt(n ** 2 - cb[:, None] ** 2)[:, med]
    nsa, nsb = np.sqrt(n[ks] ** 2 - ca ** 2)[:, None], np.sqrt(n[ks] ** 2 - cb ** 2)[:, None]
    Rs, nk = Rin[ks][:, None], n[med]
    ma, mb = Rs * nsa / (nua * rad), Rs * nsb / (nub * rad)
    mlo, mhi = np.minimum(ma, mb), np.maximum(ma, mb)
    kc = Rs / rad * (n[ks][:, None] ** 2 - nk ** 2)           # kappa' = kc C_z/(nu_k* nu^3); C_z/(nu_k* nu^3) increases
    fa, fb = ca[:, None] / (nsa * nua ** 3), cb[:, None] / (nsb * nub ** 3)
    dlo, dhi = np.where(kc >= 0, kc * fa, kc * fb), np.where(kc >= 0, kc * fb, kc * fa)
    sa, sb = np.sin(ta)[:, None], np.sin(tb)[:, None]
    co_lo, co_hi = np.cos(tb)[:, None], np.cos(ta)[:, None]
    xlo, xhi = np.minimum(mlo * sa, 1.0), np.minimum(mhi * sb, 1.0)
    Dlo, Dhi = np.sqrt(1.0 - xhi ** 2), np.sqrt(1.0 - xlo ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        iDlo, iDhi = 1.0 / Dhi, 1.0 / Dlo
        glo, ghi = ca[:, None] / nua, cb[:, None] / nub
        hlo, hhi = nk ** 2 / nua ** 3, nk ** 2 / nub ** 3
        pt = (mlo * co_lo * iDlo, mhi * co_hi * iDhi)
        pc = _imul(dlo, dhi, sa * iDlo, sb * iDhi)
        zt = (rad * glo * mlo ** 2 * sa * co_lo * iDlo, rad * ghi * mhi ** 2 * sb * co_hi * iDhi)
        q = _imul(dlo, dhi, glo * mlo * sa * sa * iDlo, ghi * mhi * sb * sb * iDhi)
        zc = (rad * (hlo * Dlo - q[1]), rad * (hhi * Dhi - q[0]))
    graze = (med[None, :] == ks[:, None]) & (np.arange(med.size)[None, :] % 2 == 0)
    pt = (np.where(graze, 1.0, pt[0]), np.where(graze, 1.0, pt[1]))
    pc = (np.where(graze, 0.0, pc[0]), np.where(graze, 0.0, pc[1]))
    zt = (np.where(graze, rad * glo * sa, zt[0]), np.where(graze, rad * ghi * sb, zt[1]))
    zc = (np.where(graze, rad * hlo * co_lo, zc[0]), np.where(graze, rad * hhi * co_hi, zc[1]))
    lo, hi = np.empty((K, 2, 2)), np.empty((K, 2, 2))
    for (r, c), (a, b), s in (((0, 0), pt, sig), ((0, 1), pc, sig), ((1, 0), zt, sig), ((1, 1), zc, -sig)):
        with np.errstate(invalid="ignore"):
            l_, h_ = np.where(s > 0, a, -b).sum(1), np.where(s > 0, b, -a).sum(1)
        grow = 64 * _EPS * (np.abs(l_) + np.abs(h_))
        lo[:, r, c] = np.where(np.isnan(l_), -np.inf, l_ - grow)
        hi[:, r, c] = np.where(np.isnan(h_), np.inf, h_ + grow)
    return lo, hi


def _far_point(st, th, C, ks):
    """Phi_far, Z_far and their Jacobian in (theta*, C_z) at points, with a bound on the rounding error of each value.

    The term that grazes is evaluated as theta* and cos(theta*), which keeps full accuracy as theta* -> pi/2.  The
    argument x_j = kappa_j sin(theta*) carries a relative error of a few eps, more where nu is small, and arcsin and
    sqrt magnify it by x_j/g_j."""
    n, Rin, Rout, med, rad, sig = st
    nu = np.sqrt(n ** 2 - C[:, None] ** 2)
    x = np.minimum(nu[np.arange(C.size), ks][:, None] * Rin[ks][:, None] * np.sin(th)[:, None] / (nu[:, med] * rad), 1.0)
    graze = (med[None, :] == ks[:, None]) & (np.arange(med.size)[None, :] % 2 == 0)
    asn = np.where(graze, th[:, None], np.arcsin(x))
    root = np.where(graze, np.cos(th)[:, None], np.sqrt(1.0 - x * x))
    zj = -sig * rad * (C[:, None] / nu[:, med]) * root
    phi = np.pi + (sig * asn).sum(1)
    z = zj.sum(1)
    amp = (n ** 2 + C[:, None] ** 2) / nu ** 2
    ex = 6.0 + amp[np.arange(C.size), ks][:, None] + amp[:, med]      # relative error of x_j, in units of eps
    with np.errstate(divide="ignore", invalid="ignore"):
        grow = np.nan_to_num(np.where(graze, 0.0, x / root), posinf=1e300)
    terms = 2 * med.size + 2
    ephi = _EPS * ((grow * ex).sum(1) + terms * (np.pi + np.abs(asn).sum(1)))
    ez = _EPS * ((np.abs(zj) * (grow ** 2 * ex + 6.0 + amp[:, med])).sum(1) + terms * np.abs(zj).sum(1))
    lo, hi = far_box_jacobian(st, th, th, C, C, ks)
    return phi, z, 0.5 * (lo + hi), ephi, ez


def _far_residual_mp(st, th, C, k, dphi, dz):
    """(Phi_far - dphi, Z_far - dz) at one point, in 34-digit arithmetic, for a box whose Krawczyk test is limited by
    the rounding of the double-precision value (near a caustic, where the inverse Jacobian is large)."""
    import mpmath as mpm
    n, Rin, Rout, med, rad, sig = st
    with mpm.workdps(34):
        th_, C_ = (v if isinstance(v, mpm.mpf) else mpm.mpf(float(v)) for v in (th, C))
        nu = [mpm.sqrt(mpm.mpf(float(v)) ** 2 - C_ ** 2) for v in n]
        Lm = nu[k] * mpm.mpf(float(Rin[k]))
        phi, z = mpm.pi, mpm.mpf(0)
        for j in range(med.size):
            nj, Rj = nu[med[j]], mpm.mpf(float(rad[j]))
            if med[j] == k and j % 2 == 0:
                a, r = th_, mpm.cos(th_)
            else:
                x = Lm * mpm.sin(th_) / (nj * Rj)
                a, r = mpm.asin(x), mpm.sqrt(1 - x * x)
            phi += float(sig[j]) * a
            z -= float(sig[j]) * Rj * (C_ / nj) * r
        return [phi - mpm.mpf(float(dphi)), z - mpm.mpf(float(dz))]


def classify_far(n, R, rP, rX, dphi, dz, tol=1e-13, stats=None, maxboxes=200000):
    """Steps (3) and (4), far branch: every (Lambda, C_z) with Phi_far = dphi and Z_far = dz, polished.

    Boxes in (theta*, C_z), Lambda = Lmax(C_z) sin(theta*), cover [0, pi/2] x [0, C_max] between the cuts of (cuts); no
    steeper ray reaches the point (Property zbound).  A box is dropped when the bounds (farbox) exclude the target,
    or when Krawczyk's test (krawczyk) on the box grown by 10% shows it holds no ray; it yields a ray when the test
    shows it holds exactly one, and Newton polishes that ray inside it; otherwise it is halved.  The test carries a
    bound on the rounding error of the residual at the centre, and where that bound is not small against the box,
    as next to a caustic, the residual is recomputed to 34 digits, so no box is dropped on rounding.  A box still
    undecided at width tol either holds a ray on the edge of its piece (Lambda = 0 at dphi = pi, or a cut) or lies
    within rounding of a caustic; Newton is run from its centre and a ray it converges to is kept unless already found.
    stats, if a dict, receives the numbers of boxes examined, of residuals recomputed to 34 digits, of undecided boxes
    and of rays found from them."""
    sgn, dz = (1.0 if dz >= 0 else -1.0), abs(dz)
    if dz == 0.0:
        return [(L, 0.0) for L, _, _ in far_roots(0.0, n, R, rP, rX, dphi)]
    st = _far_stack(n, R, rP, rX)
    nmin = st[0].min()
    cmax = _bisect(lambda C: Z0(C, n, R, rP, rX) - dz, 0.0, nmin * (1 - 1e-15), it=200)
    top = min(cmax * (1 + 1e-12) + 1e-15, nmin * (1 - 1e-15))
    cuts = [0.0] + switch_tilts(n, R, rX, top) + [top]
    ca, cb = np.array(cuts[:-1]), np.array(cuts[1:])
    ta, tb = np.zeros(ca.size), np.full(ca.size, 0.5 * np.pi)
    pa, pb = ca.copy(), cb.copy()                               # the piece between cuts that each box lies in
    ks = np.argmin(np.sqrt(st[0] ** 2 - (0.5 * (ca + cb))[:, None] ** 2) * st[1], 1)
    found, loose, examined, precise = [], [], 0, 0
    while ta.size:
        examined += ta.size
        if examined > maxboxes:                                 # never expected; keep every box rather than drop it
            loose += [(0.5 * (a + b), 0.5 * (c + d), p, q, k) for a, b, c, d, p, q, k in zip(ta, tb, ca, cb, pa, pb, ks)]
            break
        plo, phi, zlo, zhi = box_bounds(st, ta, tb, ca, cb, ks)
        keep = (plo <= dphi) & (dphi <= phi) & (zlo <= dz) & (dz <= zhi)
        ta, tb, ca, cb, pa, pb, ks = (a[keep] for a in (ta, tb, ca, cb, pa, pb, ks))
        if not ta.size:
            break
        tm, cm, wt, wc = 0.5 * (ta + tb), 0.5 * (ca + cb), tb - ta, cb - ca
        Ta, Tb = np.maximum(ta - 0.1 * wt, 0.0), np.minimum(tb + 0.1 * wt, 0.5 * np.pi)
        Ca, Cb = np.maximum(ca - 0.1 * wc, pa), np.minimum(cb + 0.1 * wc, pb)
        ph, z, J, ephi, ez = _far_point(st, tm, cm, ks)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            Y = np.stack([J[:, 1, 1], -J[:, 0, 1], -J[:, 1, 0], J[:, 0, 0]], 1).reshape(-1, 2, 2) / det[:, None, None]
            Y = np.where(np.isfinite(Y).all((1, 2))[:, None, None], Y, 0.0)
            v = -np.einsum("kij,kj->ki", Y, np.stack([ph - dphi, z - dz], 1))
            err = np.abs(Y[:, :, 0]) * ephi[:, None] + np.abs(Y[:, :, 1]) * ez[:, None]
            half = 0.5 * np.stack([Tb - Ta, Cb - Ca], 1)
            for k in np.flatnonzero((err > 0.125 * half).any(1) & (det != 0) & np.isfinite(det)):
                F_ = _far_residual_mp(st, tm[k], cm[k], ks[k], dphi, dz)
                v[k] = [-float(float(Y[k, i, 0]) * F_[0] + float(Y[k, i, 1]) * F_[1]) for i in range(2)]
                err[k] = 4 * _EPS * (np.abs(v[k]) + np.array([tm[k], cm[k]])) + 1e-30 * np.abs(Y[k]).sum(1)
                precise += 1
            Jlo, Jhi = far_box_jacobian(st, Ta, Tb, Ca, Cb, ks)
            Klo = np.stack([tm, cm], 1) + v - err
            Khi = np.stack([tm, cm], 1) + v + err
            lo_off, hi_off = np.stack([Ta - tm, Ca - cm], 1), np.stack([Tb - tm, Cb - cm], 1)
            for r in range(2):
                for c in range(2):
                    y0lo, y0hi = _imul(Y[:, r, 0], Y[:, r, 0], Jlo[:, 0, c], Jhi[:, 0, c])
                    y1lo, y1hi = _imul(Y[:, r, 1], Y[:, r, 1], Jlo[:, 1, c], Jhi[:, 1, c])
                    slack = 8 * _EPS * (np.abs(y0lo) + np.abs(y0hi) + np.abs(y1lo) + np.abs(y1hi))
                    klo, khi = _imul(float(r == c) - (y0hi + y1hi) - slack, float(r == c) - (y0lo + y1lo) + slack,
                                     lo_off[:, c], hi_off[:, c])
                    Klo[:, r] += klo
                    Khi[:, r] += khi
            Blo, Bhi = np.stack([Ta, Ca], 1), np.stack([Tb, Cb], 1)
            fine = np.isfinite(Klo).all(1) & np.isfinite(Khi).all(1) & (det != 0)
            one = fine & ((Klo > Blo) & (Khi < Bhi)).all(1)
            none = fine & ((Khi < Blo) | (Klo > Bhi)).any(1)
        for k in np.flatnonzero(one):
            found.append((tm[k], cm[k], Ta[k], Tb[k], Ca[k], Cb[k], ks[k], Y[k]))
        open_ = ~one & ~none
        small = open_ & (wt < tol) & (wc < tol)
        loose += [(tm[k], cm[k], pa[k], pb[k], ks[k]) for k in np.flatnonzero(small)]
        split = open_ & ~small
        ta, tb, ca, cb, pa, pb, ks, tm, cm, wt, wc, J = (a[split] for a in (ta, tb, ca, cb, pa, pb, ks, tm, cm, wt, wc, J))
        zs = max(dz, 1e-9)
        with np.errstate(invalid="ignore", over="ignore"):
            st_ = wt * (np.abs(J[:, 0, 0]) + np.abs(J[:, 1, 0]) / zs)
            sc_ = wc * (np.abs(J[:, 0, 1]) + np.abs(J[:, 1, 1]) / zs)
        st_, sc_ = np.where(np.isfinite(st_), st_, wt), np.where(np.isfinite(sc_), sc_, wc)
        by_t = ((st_ >= sc_) & (wt >= tol)) | (wc < tol)
        ta, tb, ca, cb = (np.concatenate([ta, np.where(by_t, tm, ta)]), np.concatenate([np.where(by_t, tm, tb), tb]),
                          np.concatenate([ca, np.where(by_t, ca, cm)]), np.concatenate([np.where(by_t, cb, cm), cb]))
        pa, pb, ks = np.concatenate([pa, pa]), np.concatenate([pb, pb]), np.concatenate([ks, ks])

    def newton(x, lo, hi, k, Y=None):
        """Newton on (Phi_far - dphi, Z_far - dz) in (theta*, C_z) inside [lo, hi]; the fixed Y when a step leaves it.
        Stops once the residual is within the rounding error of the values."""
        for _ in range(60):
            p_, z_, J_, e1, e2 = _far_point(st, x[:1], x[1:], np.array([k]))
            F_ = np.array([p_[0] - dphi, z_[0] - dz])
            if abs(F_[0]) <= e1[0] and abs(F_[1]) <= e2[0]:
                break
            try:
                y = x - np.linalg.solve(J_[0], F_)
            except np.linalg.LinAlgError:
                y = x
            if not ((lo <= y) & (y <= hi)).all():
                y = x - Y @ F_ if Y is not None else np.clip(y, lo, hi)
            if np.all(np.abs(y - x) <= 4 * _EPS * np.maximum(1.0, np.abs(x))):
                break
            x = y
        return x, np.abs(F_), (e1[0], e2[0])

    rays = []                                                   # (theta and C_z, k*, box)
    for tm_, cm_, Ta_, Tb_, Ca_, Cb_, k_, Y_ in found:
        lo, hi = np.array([Ta_, Ca_]), np.array([Tb_, Cb_])
        x, _, _ = newton(np.array([tm_, cm_]), lo, hi, k_, Y_)
        # each of two boxes holds exactly one ray, so rays lying in each other's boxes are the same ray
        if not any(((r[2] <= x) & (x <= r[3])).all() and ((lo <= r[0]) & (r[0] <= hi)).all() for r in rays):
            rays.append((x, k_, lo, hi))
    extra = 0
    for tm_, cm_, pa_, pb_, k_ in loose:
        x, res, (e1, e2) = newton(np.array([tm_, cm_]), np.array([0.0, pa_]), np.array([0.5 * np.pi, pb_]), k_)
        if (res[0] <= max(64 * e1, 1e-13) and res[1] <= max(64 * e2, 1e-13 * max(dz, 1e-3))
                and not any(np.all(np.abs(x - r[0]) <= 1e-12 + 1e-9 * np.abs(r[0])) for r in rays)):
            rays.append((x, k_, x, x))
            extra += 1
    if stats is not None:
        stats.update(boxes=examined, precise=precise, undecided=len(loose), from_undecided=extra)
    nu = lambda C, k: np.sqrt(st[0][k] ** 2 - C * C)
    return [(float(nu(x[1], k_) * st[1][k_] * np.sin(x[0])), sgn * float(x[1])) for x, k_, _, _ in rays]


def no_wrap_below(n, R, rP, rX, c_hi, tol=1e-10, maxboxes=10 ** 6):
    """Whether no ray of either branch with |C_z| <= c_hi sweeps more than pi, decided by boxes (test wraptest).

    A box is cleared when its upper bound (farbox) on the sweep is at most pi, or, on the far branch and touching
    Lambda = 0, where Phi_far = pi, when its upper bound (farjac) on dPhi_far/dtheta* is at most 0.  Other boxes are
    halved.  Returns (True, boxes examined), or (False, (theta*, C_z, far)) for a box whose every ray sweeps past pi
    or that is not cleared by width tol."""
    st = _far_stack(n, R, rP, rX)
    cuts = [0.0] + switch_tilts(n, R, rX, c_hi) + [c_hi]
    examined = 0
    for far in (False, True):
        ca, cb = np.array(cuts[:-1]), np.array(cuts[1:])
        ta, tb = np.zeros(ca.size), np.full(ca.size, 0.5 * np.pi)
        ks = np.argmin(np.sqrt(st[0] ** 2 - (0.5 * (ca + cb))[:, None] ** 2) * st[1], 1)
        while ta.size:
            examined += ta.size
            if examined > maxboxes:
                return False, None
            lo, hi = box_bounds(st, ta, tb, ca, cb, ks, far)[:2]
            if (lo > np.pi).any():                              # every ray of this box sweeps past pi
                k = np.flatnonzero(lo > np.pi)[0]
                return False, (0.5 * (ta[k] + tb[k]), 0.5 * (ca[k] + cb[k]), far)
            clear = hi <= np.pi
            at0 = far & (ta == 0.0) & ~clear
            if at0.any():
                clear[at0] = far_box_jacobian(st, ta[at0], tb[at0], ca[at0], cb[at0], ks[at0])[1][:, 0, 0] <= 0.0
            ta, tb, ca, cb, ks = (a[~clear] for a in (ta, tb, ca, cb, ks))
            wt, wc = tb - ta, cb - ca
            stuck = np.flatnonzero((wt < tol) & (wc < tol))
            if stuck.size:
                k = stuck[0]
                return False, (0.5 * (ta[k] + tb[k]), 0.5 * (ca[k] + cb[k]), far)
            tm, cm = 0.5 * (ta + tb), 0.5 * (ca + cb)
            by_t = wt / (0.5 * np.pi) >= wc / max(c_hi, 1e-300)
            ta, tb, ca, cb = (np.concatenate([ta, np.where(by_t, tm, ta)]), np.concatenate([np.where(by_t, tm, tb), tb]),
                              np.concatenate([ca, np.where(by_t, ca, cm)]), np.concatenate([np.where(by_t, cb, cm), cb]))
            ks = np.concatenate([ks, ks])
    return True, examined


def solve_far(n, R, rP, rX, dphi, dz, Lam0, Cz0, itmax=40, tol=1e-10):
    """Step (4), far branch: Newton on (Phi_far - dphi, Z_far - dz) with the Jacobian of Property symmetry, from a
    step-(3) seed, backtracking on the residual and staying inside D (this branch has no convex program)."""
    def residual(L, C):
        pz = Phi_Z(L, C, n, R, rP, rX, far=True)
        return None if pz is None else np.array([pz[0] - dphi, pz[1] - dz])

    L, C = Lam0, Cz0
    g = residual(L, C)
    for it in range(itmax):
        if g is None:
            return None, it, np.inf
        if np.linalg.norm(g) < tol:
            return (L, C), it, np.linalg.norm(g)
        j12 = dPhi_dCz(L, C, n, R, rP, rX, far=True)
        j22 = dZ_dCz(L, C, n, R, rP, rX, far=True)
        if j22 is None:
            break
        try:
            p = -np.linalg.solve(np.array([[dPhi_dLam(L, C, n, R, rP, rX, far=True), j12], [j12, j22]]), g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-12:
            Ln, Cn = L + t * p[0], C + t * p[1]
            gn = residual(Ln, Cn) if in_domain(Ln, Cn, n, R, rX) else None
            if gn is not None and np.linalg.norm(gn) < np.linalg.norm(g):
                L, C, g = Ln, Cn, gn
                break
            t *= 0.5
        else:
            break
    return (L, C), itmax, (np.inf if g is None else np.linalg.norm(g))


def project(n, R, P0, X, tol=1e-9):
    """Steps (1)-(5): every ray from the pinhole P0 to the point X, as (branch, Lambda, C_z, direction at P0).

    The sign of Lambda is the sense of rotation about the axis.  Beside Delta phi the targets
    2 pi - Delta phi, Delta phi + 2 pi, ... are searched while they stay below the bound
    sum_{k<N} arccos(R_k/R_{k-1}) + pi on any sweep (Property chord); test (wraptest) of the notes would
    let them be skipped when |Delta z| < Z0(C_pi), and the box search of the far branch discards them quickly.

    A point opposite the pinhole, Delta phi = pi, is reached by the radial ray of the far branch, Lambda = 0, which
    sweeps pi at every tilt.  Its root lies on the edge Lambda = 0 of the far search for both senses, where the
    rounding of pi alone can put it just outside for both or just inside for both, so it is added here, once.  The
    targets pi and 2 pi - pi coincide there, so each is searched once and every other ray is returned in both senses.

    Every length is divided on entry by u = 2^floor(log2 r_P), the power of two in (r_P/2, r_P], and Lambda is
    multiplied by u on exit.  The tolerances, tol here and those of the solvers, act on (Delta phi, Delta z), so they
    then act on Delta z / u, and a rig gives the same rays in metres and in millimetres.  In millimetres a tolerance
    on Delta z itself would be 1000 times stricter than in metres, beyond what the closed forms resolve next to
    grazing.  Dividing by a power of two is exact, so the problem solved is the one given to the last bit; dividing
    by r_P itself would round every length, and next to grazing that rounding alone costs about eps/sqrt(delta) in
    Phi (audit A1, code/tests/openlpt_audit/)."""
    P0, X = np.asarray(P0, float), np.asarray(X, float)
    unit = math.ldexp(1.0, math.frexp(np.hypot(P0[0], P0[1]))[1] - 1)     # u = 2^floor(log2 r_P), exact scaling
    R, P0, X = [float(r) / unit for r in R], P0 / unit, X / unit
    rP, rX = np.hypot(P0[0], P0[1]), np.hypot(X[0], X[1])
    dz = X[2] - P0[2]
    if rX < 1e-15:                                   # step (1): the radial ray
        C = np.sign(dz) * _bisect(lambda c: Z0(c, n, R, rP, rX) - abs(dz), 0.0, min(n) * (1 - 1e-15), it=200)
        return [("near", 0.0, C, direction_from_invariants(P0, 0.0, C, n[0], nus(n, C)[0]))]
    d = (np.arctan2(X[1], X[0]) - np.arctan2(P0[1], P0[0]) + np.pi) % (2 * np.pi) - np.pi
    turn, dphi = (1.0 if d >= 0 else -1.0), abs(d)
    opposite = abs(dphi - np.pi) <= 8 * _EPS                 # within 4 ulp of pi, the rounding of the azimuths
    top = sum(np.arccos(ri / ro) for ro, ri in zip([rP] + list(R[:-1]), R)) + np.pi
    rays, m = [], 0
    if opposite:                                     # step (1): the far branch's radial ray, Z_far(0, C_z) increasing
        dphi = np.pi
        C = np.sign(dz) * _bisect(lambda c: Phi_Z(0.0, c, n, R, rP, rX, far=True)[1] - abs(dz), 0.0,
                                  min(n) * (1 - 1e-15), it=200)
        rays.append(("far", 0.0, C))
    senses = (1.0, -1.0) if opposite else None
    while dphi + 2 * np.pi * m < top:
        for target, sense in ((dphi + 2 * np.pi * m, 1.0), (2 * np.pi * (m + 1) - dphi, -1.0)):
            if target >= top or (opposite and sense < 0):
                continue
            seed = classify(n, R, rP, rX, target, dz)
            if seed is not None:
                sol, _, res = solve(n, R, rP, rX, target, dz, seed[0], seed[1])
                if sol is not None and res < tol:
                    rays += [("near", turn * s * sol[0], sol[1]) for s in (senses or (sense,))]
            for s0 in classify_far(n, R, rP, rX, target, dz):
                sol, _, res = solve_far(n, R, rP, rX, target, dz, s0[0], s0[1])
                if sol is not None and res < tol:
                    if opposite and abs(sol[0]) <= 1e-9 * Lmax_kstar(n, R, rX, sol[1])[0]:
                        continue                     # the radial ray, already added
                    rays += [("far", turn * s * sol[0], sol[1]) for s in (senses or (sense,))]
        m += 1
    return [(b, L * unit, C, direction_from_invariants(P0, L, C, n[0], nus(n, C)[0])) for b, L, C in rays]


# ==================================================================== self-test
def _selftest():
    rng = np.random.default_rng(2718)
    P = lambda *a: print(*a, flush=True)

    P("radial action: closed form vs. its own derivative")
    worst = 0.0
    for _ in range(1500):
        N = int(rng.integers(2, 6))
        n = [1.0] + list(rng.uniform(1.05, 2.0, N - 1))
        R = np.sort(rng.uniform(0.03, 0.12, N - 1))[::-1].tolist()
        rP = rng.uniform(R[0] * 1.05, 1.0)
        rX = rng.uniform(1e-3, R[-1])
        Cz = rng.uniform(-0.6, 0.6)
        Lm, _ = Lmax_kstar(n, R, rX, Cz)
        Lam = rng.uniform(0.05, 0.7) * Lm
        for far in (False, True):
            h = 1e-6 * Lm
            a, b = W_closed(Lam + h, Cz, n, R, rP, rX, far), W_closed(Lam - h, Cz, n, R, rP, rX, far)
            if a is None or b is None:
                continue
            ref = Phi_Z(Lam, Cz, n, R, rP, rX, far)[0]
            worst = max(worst, abs(-(a - b) / (2 * h) - ref) / (abs(ref) + 1e-12))
    P(f"    max rel. error of -dW/dLambda = Phi : {worst:.1e}\n")

    n, R, rP = [1.0, 1.49, 1.33], [0.100, 0.094], 0.35
    P0 = np.array([rP, 0.0, 0.0])
    P("end-to-end, near branch: solve, rebuild the direction, trace, compare with the target")
    P("  reference port n=(1,1.49,1.33), R=(100,94) mm, r_P=350 mm")
    for label, zspan in (("coplanar   ", 0.0), ("out of plane", 0.20), ("steep rays ", 0.60)):
        ok = miss = fail = notvis = 0
        its, wm = [], 0.0
        for _ in range(700):
            rX = rng.uniform(0.002, R[-1] * 0.999)
            phX = rng.uniform(0.0, np.pi)
            zX = rng.uniform(-zspan, zspan)
            X = np.array([rX * np.cos(phX), rX * np.sin(phX), zX])
            seed = classify(n, R, rP, rX, phX, zX, iters=8)   # coarse: Newton polishes
            if seed is None:
                notvis += 1                       # no near-branch ray at any C_z
                continue
            sol, it, res = solve(n, R, rP, rX, phX, zX, seed[0], seed[1])
            if sol is None or res > 1e-9:
                fail += 1
                continue
            its.append(it)
            L, C = sol
            d = direction_from_invariants(P0, L, C, n[0], nus(n, C)[0])
            Xh = trace_to_radius(P0, d, n, R, rX, far=False)
            if Xh is None:
                miss += 1
                continue
            e = np.linalg.norm(Xh - X)
            wm = max(wm, e)
            ok += e < 1e-9
            miss += e >= 1e-9
        P(f"  {label}  solved+traced {ok:5d}   trace mismatch {miss}   solver failed {fail}"
          f"   not on the near branch {notvis}")
        if its:
            P(f"                 Newton iterations mean {np.mean(its):.2f}, median "
              f"{np.median(its):.0f}, max {max(its)};  worst miss {wm:.1e} m")

    P("\nround trip, both branches: trace random rays from the pinhole, project their end points back")
    for label, lo, hi, count in (("across the tube", 0.0, 0.6, 60), ("oblique       ", 0.6, 0.97, 60),
                                 ("along the tube ", 0.97, 0.995, 40)):
        got = lost = extra = far_rays = 0
        worst = 0.0
        for _ in range(count):
            rX = rng.uniform(0.002, R[-1] * 0.999)
            far = bool(rng.uniform() < 0.6)
            Cz = rng.uniform(lo, hi) * (1.0 if rng.uniform() < 0.5 else -1.0)
            Lm, _ = Lmax_kstar(n, R, rX, Cz)
            s = 1 - rng.uniform() ** 2 if rng.uniform() < 0.5 else rng.uniform()
            Lam = min(s, 1 - 1e-6) * Lm * (1.0 if rng.uniform() < 0.5 else -1.0)
            X = trace_to_radius(P0, direction_from_invariants(P0, Lam, Cz, n[0], nus(n, Cz)[0]), n, R, rX, far)
            if X is None:
                continue
            rays = project(n, R, P0, X)
            hit = any(b == ("far" if far else "near") and abs(L - Lam) < 1e-7 * Lm and abs(C - Cz) < 1e-7
                      for b, L, C, _ in rays)
            got, lost, extra = got + hit, lost + (not hit), extra + len(rays) - hit
            far_rays += far
            for b, _, _, dd in rays:
                Xh = trace_to_radius(P0, dd, n, R, rX, b == "far")
                worst = max(worst, np.inf if Xh is None else np.linalg.norm(Xh - X))
        P(f"  {label}  {got + lost} rays ({far_rays} far): recovered {got}, lost {lost}; "
          f"{extra} other images found; every ray returned lands within {worst:.1e} m")


# ==================================================================== image census
def image_census(seed, count, steep):
    """How many images the end points of random rays have at the reference port (Section images of the notes).

    steep: half the tilts in 0.8 < |C_z| < 0.995, where far-branch rays can sweep past pi; otherwise |C_z| < 0.78.
    Returns ({images: points}, [(r_X, dphi, dz, [(branch, C_z, Lambda/Lmax, sweep, the long way round)])] for
    every point with three or more images)."""
    n, R, rP = [1.0, 1.49, 1.33], [0.100, 0.094], 0.35
    P0 = np.array([rP, 0.0, 0.0])
    rng_c = np.random.default_rng(seed)
    hist, many = {}, []
    for _ in range(count):
        rX = rng_c.uniform(0.002, R[-1] * 0.999)
        far = bool(rng_c.uniform() < 0.5)
        if steep:
            Cz = (rng_c.uniform(0.8, 0.995) if rng_c.uniform() < 0.5 else rng_c.uniform(0.0, 0.8)) * (1 if rng_c.uniform() < 0.5 else -1)
        else:
            Cz = rng_c.uniform(0.0, 0.78) * (1 if rng_c.uniform() < 0.5 else -1)
        Lm = Lmax_kstar(n, R, rX, Cz)[0]
        if steep:
            s = 1 - rng_c.uniform() ** 3 if rng_c.uniform() < 0.5 else rng_c.uniform()
        else:
            s = 1 - 0.2 * rng_c.uniform() ** 2 if rng_c.uniform() < 0.7 else rng_c.uniform()
        X = trace_to_radius(P0, direction_from_invariants(P0, min(s, 1 - 1e-7) * Lm, Cz, n[0], nus(n, Cz)[0]), n, R, rX, far)
        if X is None:
            continue
        rays = project(n, R, P0, X)
        hist[len(rays)] = hist.get(len(rays), 0) + 1
        if len(rays) >= 3:
            dphi = abs(np.arctan2(X[1], X[0]))
            info = []
            for b, L, C, _ in rays:
                sweep = Phi_Z(abs(L), abs(C), n, R, rP, rX, b == "far")[0]
                info.append((b, C, abs(L) / Lmax_kstar(n, R, rX, C)[0], sweep, abs(sweep - dphi) > 1e-6))
            many.append((rX, dphi, X[2], info))
    return hist, many



# ==================================================================== image maps
def _phi_z_grid(L, C, n, R, rP, rX, far):
    """Phi and Z of eqs. (Phinear), (Phifar) and (Z), vectorised over arrays L and C of one shape."""
    n = np.asarray(n, float)
    Ro, Ri = np.array([rP] + list(R)), np.array(list(R) + [rX])
    nu = np.sqrt(n ** 2 - C[..., None] ** 2)
    b = L[..., None] / nu
    ai, ao = np.arcsin(np.clip(b / Ri, -1, 1)), np.arcsin(np.clip(b / Ro, -1, 1))
    si, so = np.sqrt(np.maximum(Ri ** 2 - b ** 2, 0)), np.sqrt(np.maximum(Ro ** 2 - b ** 2, 0))
    ph, dz = ai - ao, (so - si) / nu
    if far:
        ph[..., -1] = np.pi - ai[..., -1] - ao[..., -1]
        dz[..., -1] = (so[..., -1] + si[..., -1]) / nu[..., -1]
    return ph.sum(-1), C * dz.sum(-1)


def image_counts(n, R, rP, rX, ztop, dphi_px=0.25, dz_px=0.0025, m=1500):
    """How many images each point on the circle rX has, over 0 < dphi < 180 deg and 0 < dz < ztop (Section images).

    Returns (short, long): counts[j, i] of the rays reaching dphi = (i + 1/2) dphi_px degrees, dz = (j + 1/2) dz_px
    the short way and the long way round.  The maps (Lambda, C_z) -> (Phi, Z) of both branches are sampled with
    C_z >= 0 evenly in Z0 up to Z0 = ztop (no steeper ray gets that close, Property zbound), plus the tilts where k*
    switches, so that the tip of the shadow there is a grid point (Remark czslice), and Lambda/Lmax =
    1 - (1 - t)^2; every grid cell is cut into two triangles, and a pixel centre counts the triangles that cover
    it, which is its number of preimages away from the images of fold curves and of the grid edges.  A sweep
    Phi > pi reaches the point at 2 pi - Phi, turning the other way."""
    n = np.asarray(n, float)
    Rout, Rin = [rP] + list(R), list(R) + [rX]
    cs = np.linspace(0.0, n.min() * (1 - 1e-12), 200001)
    z0s = cs * sum((Rout[k] - Rin[k]) / np.sqrt(n[k] ** 2 - cs ** 2) for k in range(len(n)))
    C = np.interp(np.linspace(0.0, ztop, m), z0s, cs)
    C = np.union1d(C, switch_tilts(n, R, rX, C[-1]))            # a row at each switch of k*, the tip of a cusp
    t = np.linspace(0.0, 1.0, m)
    s = 1 - (1 - t) ** 2
    s[-1] = 1 - 1e-15
    Lm = np.array([Lmax_kstar(n, R, rX, c)[0] for c in C])
    nx, ny = int(round(360 / dphi_px)), int(round(ztop / dz_px))
    cnt = np.zeros((ny, nx), np.int64)
    for far in (False, True):
        ph, z = _phi_z_grid(s[None, :] * Lm[:, None], np.repeat(C[:, None], m, 1), n, R, rP, rX, far)
        assert ph.max() < 2 * np.pi
        X, Y = np.degrees(ph) / dphi_px - 0.5, z / dz_px - 0.5          # pixel centres at integers
        for tri in ((X[:-1, :-1], X[:-1, 1:], X[1:, 1:], Y[:-1, :-1], Y[:-1, 1:], Y[1:, 1:]),
                    (X[:-1, :-1], X[1:, 1:], X[1:, :-1], Y[:-1, :-1], Y[1:, 1:], Y[1:, :-1])):
            x0, x1, x2, y0, y1, y2 = (a.ravel() for a in tri)
            i0 = np.maximum(np.ceil(np.minimum(np.minimum(x0, x1), x2)), 0).astype(np.int64)
            i1 = np.minimum(np.floor(np.maximum(np.maximum(x0, x1), x2)), nx - 1).astype(np.int64)
            j0 = np.maximum(np.ceil(np.minimum(np.minimum(y0, y1), y2)), 0).astype(np.int64)
            j1 = np.minimum(np.floor(np.maximum(np.maximum(y0, y1), y2)), ny - 1).astype(np.int64)
            idx = np.flatnonzero((i1 >= i0) & (j1 >= j0))
            w, h = (i1 - i0 + 1)[idx], (j1 - j0 + 1)[idx]
            for part in np.array_split(np.arange(idx.size), int((w * h).sum() // 4000000) + 1):
                k, wk, tk = idx[part], w[part], (w * h)[part]
                rep = np.repeat(np.arange(k.size), tk)
                off = np.arange(rep.size) - np.repeat(np.cumsum(tk) - tk, tk)
                pi_, pj = i0[k][rep] + off % wk[rep], j0[k][rep] + off // wk[rep]
                ax, ay, bx, by, cx, cy = (a[k][rep] for a in (x0, y0, x1, y1, x2, y2))
                e0 = (bx - ax) * (pj - ay) - (by - ay) * (pi_ - ax)
                e1 = (cx - bx) * (pj - by) - (cy - by) * (pi_ - bx)
                e2 = (ax - cx) * (pj - cy) - (ay - cy) * (pi_ - cx)
                inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
                np.add.at(cnt, (pj[inside], pi_[inside]), 1)
    return cnt[:, :nx // 2], cnt[:, ::-1][:, :nx // 2]


def first_wrap(n, R, rP, rX):
    """C_pi and Z of the grazing far-branch ray there, the first ray to sweep past pi at the reference port
    (Section wrap)."""
    g = lambda C: Phi_Z(Lmax_kstar(n, R, rX, C)[0] * (1 - 1e-13), C, n, R, rP, rX, far=True)[0] - np.pi
    cs = np.linspace(0.0, min(n) * (1 - 1e-9), 4001)
    i = next(k for k in range(len(cs)) if g(cs[k]) > 0)
    C = _bisect(g, cs[i - 1], cs[i], it=100)
    return C, Phi_Z(Lmax_kstar(n, R, rX, C)[0] * (1 - 1e-13), C, n, R, rP, rX, far=True)[1]


def _map_check(per=8, seed=4):
    """The image-count maps of the reference port (Figure imgmap of the notes) against project() and a trace."""
    P = lambda *a: print(*a, flush=True)
    n, R, rP = [1.0, 1.49, 1.33], [0.100, 0.094], 0.35
    rng_m = np.random.default_rng(seed)
    checked = bad = 0
    for rX in (0.050, 0.085):
        short, far_way = image_counts(n, R, rP, rX, 2.0)
        tot = short + far_way
        dphi, dz = (np.arange(tot.shape[1]) + 0.5) * 0.25, (np.arange(tot.shape[0]) + 0.5) * 0.0025
        j0, _ = np.nonzero(tot == 0)
        j3, i3 = np.nonzero(tot == 3)
        jw, iw = np.nonzero(far_way > 0)
        Cpi, zw = first_wrap(n, R, rP, rX)
        P("  map at r_X = %.0f mm, 0 < |dz| < 2 m: share by number of images %s, most %d (long way round at most %d)"
          % (1e3 * rX, ", ".join("%d: %.2f%%" % (c, 100 * np.mean(tot == c)) for c in np.unique(tot)), tot.max(),
             far_way.max()))
        P("    no image from |dz| = %.3f m; three images at dphi %.2f..%.2f deg, |dz| %.3f..%.3f m; rays the long way "
          "round from |dz| = %.3f m at dphi = %.2f deg (first wrapped ray, C_pi = %.4f: %.3f m)"
          % (dz[j0].min(), dphi[i3].min(), dphi[i3].max(), dz[j3].min(), dz[j3].max(), dz[jw].min(),
             dphi[iw[np.argmin(jw)]], Cpi, zw))
        edges = np.flatnonzero(np.diff(tot[0]))
        P("    three images: %d pixels, %d of them two rays the short way and one the long way; one ray the long way "
          "beside one the short way at %d pixels; along |dz| -> 0 the count changes at %s deg"
          % ((tot == 3).sum(), ((short == 2) & (far_way == 1)).sum(), ((short == 1) & (far_way == 1)).sum(),
             ", ".join("%.2f (%d to %d)" % (dphi[i] + 0.125, tot[0, i], tot[0, i + 1]) for i in edges) or "no azimuth"))
        win = np.lib.stride_tricks.sliding_window_view(np.pad(tot, 4, mode="edge"), (9, 9))
        flat = win.min(axis=(-1, -2)) == win.max(axis=(-1, -2))
        for c in np.unique(tot):
            jj, ii = np.nonzero((tot == c) & flat)
            for k in rng_m.choice(jj.size, size=min(per, jj.size), replace=False):
                a = np.radians(dphi[ii[k]])
                got = len(project(n, R, np.array([rP, 0.0, 0.0]), np.array([rX * np.cos(a), rX * np.sin(a), dz[jj[k]]])))
                checked, bad = checked + 1, bad + (got != c)
        # the shadow against a brute-force trace: no traced ray may land in a no-image pixel 2 px or more from an edge
        seen = np.lib.stride_tricks.sliding_window_view(np.pad(tot > 0, 2, mode="edge"), (5, 5)).any(axis=(-1, -2))
        crossings = inside = 0
        for _ in range(40):
            C = np.minimum(np.sqrt(rng_m.uniform(0.0, 1.0, 200000)) * min(n), min(n) * (1 - 1e-9))
            s = 1 - (1 - rng_m.uniform(0.0, 1.0, C.size)) ** 2
            nu = np.sqrt(np.asarray(n) ** 2 - C[:, None] ** 2)
            caps = nu * np.array(list(R) + [rX])
            for ph_, z_ in trace_many(n, R, rP, rX, C, s * caps.min(1) / (rP * nu[:, 0])).values():
                g = np.isfinite(ph_) & (np.abs(z_) < 2.0)
                fold = np.abs((ph_[g] + np.pi) % (2 * np.pi) - np.pi)
                i = np.minimum((np.degrees(fold) / 0.25).astype(int), tot.shape[1] - 1)
                j = np.minimum((np.abs(z_[g]) / 0.0025).astype(int), tot.shape[0] - 1)
                crossings, inside = crossings + g.sum(), inside + int((~seen[j, i]).sum())
        P("    %d traced crossings of r_X within 2 m; in the no-image region, 2 px or more from its edge: %d"
          % (crossings, inside))
    P("  project() at %d pixels away from region edges disagrees with the maps at %d" % (checked, bad))


def _traced_rays(n, R, rP, rX, dphi, dz, m=24, its=60):
    """Every ray from P0 = (r_P, 0, 0) to the point (r_X, dphi, dz) that Newton finds from an m x m grid of starts on
    the traced map of trace_many(), as (far, sin alpha, C_z): the landing azimuth against dphi modulo 2 pi and z
    against dz, with a finite-difference Jacobian.  Independent of the closed forms; it can miss a ray whose basin of
    attraction falls between the starts."""
    smax, cap = min(R[0] / rP, 1.0) * (1 - 1e-12), min(n) * (1 - 1e-9)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    h = np.array([1e-7 * smax, 1e-7])
    out = []
    for far in (False, True):
        def F(x):
            with np.errstate(invalid="ignore"):
                ph, z = trace_many(n, R, rP, rX, x[:, 1], x[:, 0])["far" if far else "near"]
            return np.stack([wrap(ph - dphi), z - dz], 1)

        s0 = smax * np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, m + 2)[1:-1])
        c0 = np.sign(dz) * cap * np.linspace(0.0, 1.0, m + 2)[1:-1] ** 2 if dz != 0.0 else np.zeros(1)
        x = np.stack([a.ravel() for a in np.meshgrid(s0, c0)], 1)
        f = F(x)
        live = np.isfinite(f).all(1)
        for _ in range(its):
            a = np.flatnonzero(live & (np.abs(f).max(1) > 1e-15))
            if not a.size:
                break
            J = np.empty((a.size, 2, 2))
            for j in range(2 if dz != 0.0 else 1):
                e = np.zeros(2)
                e[j] = h[j]
                d_ = F(x[a] + e) - F(x[a] - e)
                d_[:, 0] = wrap(d_[:, 0])
                J[:, :, j] = d_ / (2 * h[j])
            with np.errstate(divide="ignore", invalid="ignore"):
                if dz != 0.0:
                    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                    step = -np.stack([J[:, 1, 1] * f[a, 0] - J[:, 0, 1] * f[a, 1],
                                      J[:, 0, 0] * f[a, 1] - J[:, 1, 0] * f[a, 0]], 1) / det[:, None]
                else:
                    step = np.stack([-f[a, 0] / J[:, 0, 0], np.zeros(a.size)], 1)
            norm, moved, lam = np.linalg.norm(f[a], axis=1), np.zeros(a.size, bool), 1.0
            for _ in range(30):
                y = x[a] + lam * step
                ok = ~moved & np.isfinite(y).all(1) & (np.abs(y[:, 0]) < smax) & (np.abs(y[:, 1]) < cap)
                fy = np.full((a.size, 2), np.nan)
                if ok.any():
                    fy[ok] = F(y[ok])
                better = ok & np.isfinite(fy).all(1) & (np.linalg.norm(np.nan_to_num(fy, nan=np.inf), axis=1) < norm)
                x[a[better]], f[a[better]] = y[better], fy[better]
                moved |= better
                lam *= 0.5
            live[a[~moved]] = False
        good = np.isfinite(f).all(1) & (np.abs(f).max(1) < 1e-12)
        for s_, c_ in x[good]:
            if not any(b == far and abs(s_ - u) < 1e-7 * smax and abs(c_ - v) < 1e-7 for b, u, v in out):
                out.append((far, s_, c_))
    return out


def _rod_check():
    """Figure imgways (c) and (d) of the notes: the rod's rays from project() against a trace of every ray that
    enters the rod.  Level with the pinhole C_z = 0, so the rays that can reach a point form one family,
    sin alpha in (-R_1/r_P, R_1/r_P); where the swept azimuth crosses the target continuously there is one ray."""
    P = lambda *a: print(*a, flush=True)
    n, R, rP, rX = [1.0, 2.0], [0.050], 0.5, 0.049
    P0 = np.array([rP, 0.0, 0.0])
    sa = (R[0] / rP) * np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, 400001)[1:-1])
    swept = trace_many(n, R, rP, rX, np.zeros(sa.size), sa)

    def traced(dphi):
        """(far, sin alpha) of every ray reaching the azimuth dphi, located between two neighbouring traced rays."""
        hits = []
        for key, (ph, _) in swept.items():
            for base in (dphi, dphi - 2 * np.pi, dphi + 2 * np.pi):
                g = ph - base
                for i in np.flatnonzero((np.sign(g[:-1]) != np.sign(g[1:])) & (np.abs(np.diff(g)) < np.pi)):
                    hits.append((key == "far", sa[i] - g[i] * (sa[i + 1] - sa[i]) / (g[i + 1] - g[i])))
        return hits

    dphi = np.radians(178.5)
    X = np.array([rX * np.cos(dphi), rX * np.sin(dphi), 0.0])
    rays = project(n, R, P0, X)
    ref = traced(dphi)
    land = max(np.linalg.norm(trace_to_radius(P0, d, n, R, rX, b == "far") - X) for b, _, _, d in rays)
    Lm = Lmax_kstar(n, R, rX, 0.0)[0]
    P("  rod of Figure imgways (c), (d) at dphi = 178.5 deg: project() %d rays, Lambda/Lmax %s, each landing within "
      "%.1e m; the trace of %d rays entering the rod finds %d, at most %.1e from project()'s in sin alpha"
      % (len(rays), ", ".join("%+.4f (%s, %.1f deg)" % (L / Lm, b, np.degrees(Phi_Z(abs(L), 0.0, n, R, rP, rX, b == "far")[0]))
                              for b, L, _, _ in sorted(rays, key=lambda r: abs(r[1]))),
         land, sa.size, len(ref),
         max(min(abs(s - L / (n[0] * rP)) for f_, s in ref if f_ == (b == "far")) for b, L, _, _ in rays)))
    rows, bad = [], 0
    for deg in np.round(np.arange(150.0, 180.0, 0.05), 2):
        a = np.radians(deg)
        got = len(project(n, R, P0, np.array([rX * np.cos(a), rX * np.sin(a), 0.0])))
        want = len(traced(a))
        bad += got != want
        if not rows or rows[-1][1] != got:
            rows.append((deg, got))
    P("    600 azimuths from 150 to 179.95 deg: project() and the trace disagree at %d; images %s"
      % (bad, ", ".join("%d from %.2f deg" % (c, d) for d, c in rows)))


def _census():
    P = lambda *a: print(*a, flush=True)
    total, many = {}, []
    for seed, count, steep, label in ((5, 400, True, "half the tilts steep"), (9, 500, False, "|C_z| < 0.78")):
        hist, m = image_census(seed, count, steep)
        P("  %d rays, %s: images per point %s" % (count, label, dict(sorted(hist.items()))))
        for k, v in hist.items():
            total[k] = total.get(k, 0) + v
        many += m
    P("  all: %s" % dict(sorted(total.items())))
    if many:
        wrapped = [any(w for *_, w in info) for _, _, _, info in many]
        P("  %d points with three or more images, %d of them with a ray the long way round; most images %d"
          % (len(many), sum(wrapped), max(len(info) for *_, info in many)))
        P("  dphi %.1f..%.1f deg, |dz| %.3f..%.3f m, r_X %.1f..%.1f mm; long-way rays at Lambda/Lmax >= %.3f"
          % (np.degrees(min(d for _, d, _, _ in many)), np.degrees(max(d for _, d, _, _ in many)),
             min(abs(z) for _, _, z, _ in many), max(abs(z) for _, _, z, _ in many),
             1e3 * min(r for r, _, _, _ in many), 1e3 * max(r for r, _, _, _ in many),
             min(q for *_, info in many for _, _, q, _, w in info if w)))
        P("  branches: %s" % sorted(set(tuple(sorted(b for b, *_ in info)) for *_, info in many)))
    _rod_check()
    _map_check()


# ==================================================================== what step (3) can miss
def _pair_near(n, R, rP, rX, dphi, dz, L0, C0):
    """For a far-branch ray (L0, C0) of the point (dphi, dz): the nearest other ray of that point on the same level
    curve of Phi_far, the angle between the two at the pinhole (deg), and the largest |zeta - dz| between them.

    The level curve is followed in Lambda, solving Phi_far(Lambda, C_z) = dphi for C_z, so it passes smoothly
    through the tilt at which its two roots merge.  Between two rays zeta has an extremum on the curve, where
    det J = 0: the caustic.  The largest |zeta - dz| between them bounds how far inside it the point lies.
    None if no other ray lies within 5% of Lmax along the curve."""
    Lm = Lmax_kstar(n, R, rX, C0)[0]
    P0 = np.array([rP, 0.0, 0.0])

    def tilt(L, c0):
        g = lambda c: Phi_Z(L, c, n, R, rP, rX, far=True)
        v0 = g(c0)
        if v0 is None:
            return None
        h = 1e-10
        while h < 0.05:
            for c1 in (c0 - h, c0 + h):
                v1 = g(c1) if 0.0 <= c1 < min(n) else None
                if v1 is not None and (v1[0] > dphi) != (v0[0] > dphi):
                    return _bisect(lambda c: g(c)[0] - dphi, c0, c1, it=90)
            h *= 2.0
        return None

    for window in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        best = None
        for sgn in (1.0, -1.0):
            track, c_prev = [], C0
            for j in range(1, 201):
                L = L0 + sgn * window * Lm * j / 200.0
                if not 0.0 < L < Lm * (1 - 1e-12):
                    break
                c = tilt(L, c_prev)
                if c is None:
                    break
                track.append((L, c, Phi_Z(L, c, n, R, rP, rX, far=True)[1] - dz))
                c_prev = c
                if len(track) >= 2 and (track[-1][2] > 0) != (track[-2][2] > 0):
                    (La, ca, ga), (Lb, cb, gb) = track[-2], track[-1]

                    def zeta_minus_dz(x):
                        cx = tilt(x, ca)
                        return np.nan if cx is None else Phi_Z(x, cx, n, R, rP, rX, far=True)[1] - dz
                    Lp = _bisect(zeta_minus_dz, La, Lb, it=60)
                    cp = tilt(Lp, ca)
                    if cp is None:
                        Lp, cp = Lb, cb
                    d0 = direction_from_invariants(P0, L0, C0, n[0], nus(n, C0)[0])
                    dp = direction_from_invariants(P0, Lp, cp, n[0], nus(n, cp)[0])
                    ang = np.degrees(np.arccos(np.clip(np.dot(d0, dp) / np.linalg.norm(d0) / np.linalg.norm(dp), -1, 1)))
                    cand = (abs(Lp - L0), ang, max(abs(t[2]) for t in track))
                    best = cand if best is None or cand[0] < best[0] else best
                    break
        if best is not None:
            return best[1], best[2]
    return None


def _sampling_check(seed=17):
    """Section cylalg, step (3): what a sampled search misses, and what the exact searches find.

    Near branch: rays close to grazing at tilts close to the switch of k*, where the reach has a square-root cusp;
    a grid of tilts loses the component that holds such a ray when no grid tilt falls inside it.  Far branch at one
    tilt: far_turns() against a dense grid.  Far branch across tilts: rays generated close to a turning point of
    Phi_far, and whether sampling the tilts (sweep_far) and the box search (classify_far) find them again; then
    points ever closer to a caustic, where two rays merge."""
    import time
    P = lambda *a: print(*a, flush=True)
    rng_s = np.random.default_rng(seed)
    n, R, rP = [1.0, 1.49, 1.33], [0.100, 0.094], 0.35
    grid = np.linspace(0.0, min(n) * (1 - 1e-9), 400)
    P("near branch, rays within 10% of grazing and 0.003 of the switch of k* (the cusp of the reach)")
    for rX in (0.030, 0.050, 0.070):
        Cs = np.sqrt((n[2] ** 2 * rX ** 2 - n[0] ** 2 * R[0] ** 2) / (rX ** 2 - R[0] ** 2))
        total = grid_lost = exact_lost = 0
        while total < 300:
            C = Cs + rng_s.uniform(-0.003, 0.003)
            Lm = Lmax_kstar(n, R, rX, C)[0]
            L = Lm * (1 - 10 ** rng_s.uniform(-6, -1))
            ph, z = Phi_Z(L, C, n, R, rP, rX)
            if ph >= np.pi:
                continue
            total += 1
            comp = [(a, b) for a, b in components(n, R, rP, rX, ph) if a <= C <= b]
            grid_lost += not (comp and np.any((grid >= comp[0][0]) & (grid <= comp[0][1])))
            seed_ = classify(n, R, rP, rX, ph, z)
            sol = solve(n, R, rP, rX, ph, z, seed_[0], seed_[1]) if seed_ is not None else (None, 0, np.inf)
            exact_lost += not (sol[0] is not None and sol[2] < 1e-9 and abs(sol[0][1] - C) < 1e-6)
        P("  r_X = %.0f mm, switch at C_z = %.4f: of %d rays a 400-tilt grid misses the component of %d; "
          "the exact search misses none of the components, and classify + solve lose %d"
          % (1e3 * rX, Cs, total, grid_lost, exact_lost))

    from far_branch import _rand
    agree = total = 0
    for _ in range(1000):
        N = int(rng_s.integers(2, 7))
        ns, Rs, rPs, rXs = _rand(rng_s, N, bool(rng_s.integers(0, 2)))
        Cz = rng_s.uniform(0.0, 0.95) * min(ns)
        Lm = Lmax_kstar(ns, Rs, rXs, Cz)[0]
        s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, 200001))
        d = dPhi_dLam(s * Lm * (1 - 1e-13), Cz, ns, Rs, rPs, rXs, far=True)
        total += 1
        agree += len(far_turns(Cz, ns, Rs, rPs, rXs)) == int(np.sum(np.sign(d[:-1]) != np.sign(d[1:])))
    P("far branch at one tilt: far_turns() and a 200001-point grid count the same turning points on %d of %d "
      "random tilted stacks" % (agree, total))

    P("far branch across tilts: rays generated a relative distance 10^-6..10^-1 from a turning point of Phi_far")
    for rX in (0.085, 0.094 * 0.999):
        stats, pairs, unexplained = {}, [], 0
        done = box_lost = undecided = 0
        t_sweep = t_box = 0.0
        while done < 120:
            C = rng_s.uniform(0.0, 0.9)
            Lm = Lmax_kstar(n, R, rX, C)[0]
            f = lambda L_: dPhi_dLam(L_, C, n, R, rP, rX, far=True)
            if not (f(1e-9 * Lm) < 0 < f(Lm * (1 - 1e-13))):
                continue
            Lf = _bisect(f, 1e-9 * Lm, Lm * (1 - 1e-13), it=200)
            side = 1 if rng_s.uniform() < 0.5 else -1
            L = Lf * (1 + side * 10 ** rng_s.uniform(-6, -1))
            if not 0.0 < L < Lm:
                continue
            ph, z = Phi_Z(L, C, n, R, rP, rX, far=True)
            if ph >= np.pi:
                continue
            done += 1
            t0 = time.perf_counter()
            hit = False
            for s0 in sweep_far(n, R, rP, rX, ph, z):
                sol, _, res = solve_far(n, R, rP, rX, ph, z, s0[0], s0[1])
                hit |= sol is not None and res < 1e-9 and abs(sol[1] - C) < 1e-6 and abs(sol[0] - L) < 1e-6 * Lm
            t1 = time.perf_counter()
            box = {}
            rays = classify_far(n, R, rP, rX, ph, z, stats=box)
            t_sweep, t_box = t_sweep + t1 - t0, t_box + time.perf_counter() - t1
            box_lost += not any(abs(C_ - C) < 1e-9 and abs(L_ - L) < 1e-9 * Lm for L_, C_ in rays)
            undecided += box["undecided"]
            key = "increasing" if side > 0 else "decreasing"
            stats[key] = (stats.get(key, (0, 0))[0] + (not hit), stats.get(key, (0, 0))[1] + 1)
            if not hit:
                pr = _pair_near(n, R, rP, rX, ph, z, L, C)
                if pr is None:
                    unexplained += 1
                else:
                    pairs.append(pr)
        P("  r_X = %.1f mm: sampling the tilts (sweep_far) misses on decreasing arcs %d of %d, on increasing arcs %d of %d"
          % (1e3 * rX, *stats.get("decreasing", (0, 0)), *stats.get("increasing", (0, 0))))
        if pairs:
            P("    each ray missed has a partner on its level curve: the two at most %.2e deg apart at the pinhole, "
              "the point at most %.1e m from the caustic between them in dz" % (max(a for a, _ in pairs),
                                                                              max(d for _, d in pairs)))
        P("    rays missed without such a partner: %d" % unexplained)
        P("    the box search (classify_far) misses %d of %d, with %d boxes left undecided; %.0f ms a point against "
          "%.0f ms for the sweep" % (box_lost, done, undecided, 1e3 * t_box / done, 1e3 * t_sweep / done))

    rX, dphi = 0.085, np.radians(135.0)
    P("far branch against a caustic: r_X = 85 mm, dphi = 135 deg (Section tilt of the notes)")
    zpk, cpk = _zeta_peak_mp(n, R, rP, rX, dphi, 0.2766783, 0.9383, 0.9385)
    P("  zeta peaks on the increasing arc at dz* = %s m, C_z = %s (36 digits); det J = 0 there"
      % (_mp_str(zpk, 16), _mp_str(cpk, 12)))
    for off in (1e-6, 1e-9, 1e-12, 1e-15, 1e-16, 2e-17):
        row = []
        for sgn_ in (-1.0, 1.0):
            dz_ = float(zpk) + sgn_ * off
            box = {}
            rays = classify_far(n, R, rP, rX, dphi, dz_, stats=box)
            row.append("dz - dz* = %+.1e m: %d rays, %d boxes undecided" % (_mp_diff(dz_, zpk), len(rays),
                                                                          box["undecided"]))
        P("  " + ";  ".join(row))

    P("bounds (farbox) and (farjac): values and Jacobians at random points of random boxes on random stacks")
    worst_fd, outside, points = 0.0, 0, 0
    for _ in range(300):
        N = int(rng_s.integers(2, 7))
        ns, Rs, rPs, rXs = _rand(rng_s, N, bool(rng_s.integers(0, 2)))
        st = _far_stack(ns, Rs, rPs, rXs)
        top = min(ns) * rng_s.uniform(0.3, 0.99)
        cuts = [0.0] + switch_tilts(ns, Rs, rXs, top) + [top]
        i = int(rng_s.integers(0, len(cuts) - 1))
        C1, C2 = np.sort(rng_s.uniform(cuts[i], cuts[i + 1], 2))
        t1, t2 = np.sort(rng_s.uniform(0.0, 0.5 * np.pi, 2))
        ks = np.array([np.argmin(np.sqrt(st[0] ** 2 - (0.5 * (C1 + C2)) ** 2) * st[1])])
        a_ = lambda v: np.array([v])
        bounds = {far: box_bounds(st, a_(t1), a_(t2), a_(C1), a_(C2), ks, far) for far in (False, True)}
        Jlo, Jhi = far_box_jacobian(st, a_(t1), a_(t2), a_(C1), a_(C2), ks)
        for _ in range(20):
            th, C = rng_s.uniform(t1, t2), rng_s.uniform(C1, C2)
            L = Lmax_kstar(ns, Rs, rXs, C)[0] * np.sin(th)
            for far in (False, True):
                ph, z = Phi_Z(L, C, ns, Rs, rPs, rXs, far)
                lo_p, hi_p, lo_z, hi_z = (float(v[0]) for v in bounds[far])
                outside += not (lo_p - 1e-12 <= ph <= hi_p + 1e-12 and lo_z - 1e-15 <= z <= hi_z * (1 + 1e-12) + 1e-15)
            _, _, J, _, _ = _far_point(st, a_(th), a_(C), ks)
            outside += not ((Jlo[0] <= J[0]) & (J[0] <= Jhi[0])).all()
            if points % 4 == 0:                             # (farjac) against central differences to 34 digits
                import mpmath as mpm
                with mpm.workdps(34):
                    t_, c_, h = mpm.mpf(th), mpm.mpf(C), mpm.mpf("1e-12")
                    cols = [[(p - q) / (2 * h) for p, q in zip(_far_residual_mp(st, t_ + dt, c_ + dc, ks[0], 0.0, 0.0),
                                                              _far_residual_mp(st, t_ - dt, c_ - dc, ks[0], 0.0, 0.0))]
                            for dt, dc in ((h, 0), (0, h))]
                    Jn = np.array([[float(cols[0][0]), float(cols[1][0])], [float(cols[0][1]), float(cols[1][1])]])
                worst_fd = max(worst_fd, float(np.max(np.abs(Jn - J[0]) / (1e-12 + np.abs(Jn)))))
            points += 1
    P("  %d points in 300 boxes: outside the bounds %d; Jacobian (farjac) against 34-digit central differences at "
      "%d of them, worst relative difference %.1e" % (points, outside, (points + 3) // 4, worst_fd))

    P("box search against Newton from a 12 x 12 grid of starts, on random stacks")
    lost = extra_ms = searches = und = 0
    for _ in range(400):
        if searches == 60:
            break
        N = int(rng_s.integers(2, 7))
        ns, Rs, rPs, rXs = _rand(rng_s, N, bool(rng_s.integers(0, 2)))
        C = rng_s.uniform(0.0, 0.95) * min(ns)
        Lm = Lmax_kstar(ns, Rs, rXs, C)[0]
        L = Lm * (1 - rng_s.uniform() ** 2)
        pz = Phi_Z(L, C, ns, Rs, rPs, rXs, far=True)
        if pz is None or not 0.0 < pz[0] < np.pi or pz[1] <= 0.0:
            continue
        searches += 1
        box = {}
        rays = classify_far(ns, Rs, rPs, rXs, pz[0], pz[1], stats=box)
        und += box["undecided"]
        lost += not any(abs(a - L) < 1e-8 * Lm and abs(b - C) < 1e-8 for a, b in rays)
        cmax = _bisect(lambda c: Z0(c, ns, Rs, rPs, rXs) - pz[1], 0.0, min(ns) * (1 - 1e-15), it=200)
        ref = []
        for c0 in np.linspace(0.0, cmax, 13)[1:]:
            for s0 in np.sin(np.linspace(0.02, 1.55, 12)):
                sol, _, res = solve_far(ns, Rs, rPs, rXs, pz[0], pz[1], s0 * Lmax_kstar(ns, Rs, rXs, c0)[0], c0)
                if sol is not None and res < 1e-10 and not any(abs(sol[0] - a) < 1e-8 * Lm and abs(sol[1] - b) < 1e-8
                                                               for a, b in ref):
                    ref.append(sol)
        extra_ms += sum(not any(abs(a - x) < 1e-7 * Lm and abs(b - y) < 1e-7 for x, y in rays) for a, b in ref)
    P("  %d searches: the box search lost the generating ray %d times; rays found from the starts but not by the "
      "boxes %d; boxes left undecided %d" % (searches, lost, extra_ms, und))

    P("test (wraptest): C_pi checked by boxes, no ray with |C_z| <= C_pi - 1e-6 sweeps more than pi")
    for rX in (0.010, 0.030, 0.050, 0.070, 0.085, 0.094 * (1 - 1e-9)):
        cp, zw = first_wrap(n, R, rP, rX)
        ok, cnt = no_wrap_below(n, R, rP, rX, cp - 1e-6)
        E = Lmax_kstar(n, R, rX, cp + 1e-6)[0] * (1 - 1e-13)
        P("  reference port, r_X = %.1f mm: C_pi = %.5f, below it %s (%s boxes); the grazing far ray at C_pi + 1e-6 "
          "sweeps past pi: %s; Z_0(C_pi) = %.3f m" % (1e3 * rX, cp, "certified" if ok else "NOT certified", cnt,
                                                     Phi_Z(E, cp + 1e-6, n, R, rP, rX, far=True)[0] > np.pi,
                                                     Z0(cp, n, R, rP, rX)))
    for tag, ns, Rs, rPs, rXs in (("a", [1.0, 2.0, 2.8], [0.100, 0.050], 0.5, 0.035),
                                  ("b", [1.0, 1.8, 2.2], [0.100, 0.055], 0.5, 0.04505),
                                  ("c", [1.0, 2.0], [0.050], 0.5, 0.049)):
        ok0, where = no_wrap_below(ns, Rs, rPs, rXs, 1e-6)
        if not ok0:
            P("  Figure imgways (%s): a ray with C_z <= 1e-6 sweeps past pi (box at C_z = %.1e), so C_pi = 0 and the "
              "test fails at every point" % (tag, where[1]))
            continue
        cp, zw = first_wrap(ns, Rs, rPs, rXs)
        ok, cnt = no_wrap_below(ns, Rs, rPs, rXs, cp - 1e-6)
        P("  Figure imgways (%s): C_pi = %.4f, below it %s; Z_0(C_pi) = %.4f m; the grazing far ray there reaches "
          "%.4f m" % (tag, cp, "certified" if ok else "NOT certified", Z0(cp, ns, Rs, rPs, rXs), zw))

    P("points in the plane of the pinhole and the axis, dphi = 0 or pi: project() against Newton on the traced map "
      "from 24 x 24 starts")
    worst = 0.0
    for tag, ns, Rs, rPs, rXs in (("rod of Figure imgways (c)", [1.0, 2.0], [0.050], 0.5, 0.049),
                                  ("reference port, r_X = 50 mm", n, R, rP, 0.050),
                                  ("reference port, r_X = 85 mm", n, R, rP, 0.085)):
        row = []
        for phase, off, label in ((0.0, 0.0, "pi"), (1.0, 0.0, "pi, pinhole turned"), (0.0, 1e-16, "pi - 1e-16"),
                                  (0.0, 1e-14, "pi - 1e-14"), (None, 0.0, "0")):
            for dz_ in (0.0, 0.02, 0.3):
                a = rng_s.uniform(-np.pi, np.pi) if phase else 0.0
                P0_ = np.array([rPs * np.cos(a), rPs * np.sin(a), 0.1])
                if phase is None:
                    X = np.array([rXs, 0.0, 0.1 + dz_])
                else:
                    X = np.array([-rXs * np.cos(a) - off * rXs * np.sin(a), -rXs * np.sin(a) + off * rXs * np.cos(a),
                                  0.1 + dz_])
                rays = project(ns, Rs, P0_, X)
                ref = _traced_rays(ns, Rs, rPs, np.hypot(X[0], X[1]), 0.0 if phase is None else np.pi - off, dz_)
                got = [(b == "far", L / (np.sqrt(ns[0] ** 2 - C * C) * rPs), C) for b, L, C, _ in rays]
                same = lambda g, r: g[0] == r[0] and abs(g[1] - r[1]) < 1e-6 * Rs[0] / rPs and abs(g[2] - r[2]) < 1e-6
                agree = len(got) == len(ref) and all(any(same(g, r) for r in ref) for g in got)
                for b, _, _, d in rays:
                    Xh = trace_to_radius(P0_, d, ns, Rs, np.hypot(X[0], X[1]), b == "far")
                    worst = max(worst, np.inf if Xh is None else np.linalg.norm(Xh - X))
                row.append("%d%s" % (len(rays), "" if agree else " (traced %d)" % len(ref)))
            row[-3:] = ["%s: %s" % (label, "/".join(row[-3:]))]
        P("  %s, dz = 0/0.02/0.3 m: %s rays" % (tag, "; ".join(row)))
    P("  every ray returned lands within %.1e m; a count in brackets is the traced one where the two differ" % worst)


def _zeta_peak_mp(n, R, rP, rX, dphi, c0, s_lo, s_hi, dps=36):
    """The peak of zeta along the level curve Phi_far = dphi, in 36-digit arithmetic: golden section over Lambda/Lmax in
    [s_lo, s_hi] (Lmax taken at the tilt c0), the tilt at each Lambda by bisection.  Returns (dz*, C_z) as mpf."""
    import mpmath as mpm
    with mpm.workdps(dps):
        n_, R_ = [mpm.mpf(float(v)) for v in n], [mpm.mpf(float(v)) for v in R]
        rP_, rX_, d_ = mpm.mpf(float(rP)), mpm.mpf(float(rX)), mpm.mpf(float(dphi))
        Rin, Rout = R_ + [rX_], [rP_] + R_
        c0 = mpm.mpf(float(c0))

        def phiz(L, C):
            ph, z = mpm.mpf(0), mpm.mpf(0)
            for k in range(len(n_)):
                nu = mpm.sqrt(n_[k] ** 2 - C * C)
                b = L / nu
                ai, ao = mpm.asin(b / Rin[k]), mpm.asin(b / Rout[k])
                si, so = mpm.sqrt(Rin[k] ** 2 - b * b), mpm.sqrt(Rout[k] ** 2 - b * b)
                far = k == len(n_) - 1
                ph += (mpm.pi - ai - ao) if far else (ai - ao)
                z += C * ((so + si) if far else (so - si)) / nu
            return ph, z

        def zeta(L):
            g = lambda c: phiz(L, c)[0] - d_
            h = mpm.mpf("1e-7")
            while (g(c0 - h) > 0) == (g(c0 + h) > 0):
                h *= 2
            lo, hi = c0 - h, c0 + h
            up = g(lo) > 0
            for _ in range(int(3.4 * dps) + 20):
                m = (lo + hi) / 2
                lo, hi = (m, hi) if (g(m) > 0) == up else (lo, m)
            c = (lo + hi) / 2
            return phiz(L, c)[1], c

        Lm = min(mpm.sqrt(n_[k] ** 2 - c0 ** 2) * Rin[k] for k in range(len(n_)))
        a, b = mpm.mpf(s_lo) * Lm, mpm.mpf(s_hi) * Lm
        gr = (mpm.sqrt(5) - 1) / 2
        for _ in range(2 * dps + 5):
            x1, x2 = b - gr * (b - a), a + gr * (b - a)
            if zeta(x1)[0] > zeta(x2)[0]:
                b = x2
            else:
                a = x1
        return zeta((a + b) / 2)


def _mp_str(x, digits):
    import mpmath as mpm
    return mpm.nstr(x, digits)


def _mp_diff(a, b):
    """float(a) - b to full accuracy, for a double a and an mpf b."""
    import mpmath as mpm
    with mpm.workdps(40):
        return float(mpm.mpf(float(a)) - b)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["images"]:
        _census()
    elif sys.argv[1:] == ["sampling"]:
        _sampling_check()
    else:
        _selftest()
