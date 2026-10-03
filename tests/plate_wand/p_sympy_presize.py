"""Pre-plan SymPy checks of the flat-window projection the plate wand plan rests on (2026-10-02-plate-wand-plan.md,
section 3.1); W0's w0_sympy_derivations.py will extend it.  Run with the base `python` (sympy); output:
p_sympy_presize_out.txt.

Setting: camera centre C in medium 1 (index n1), plane-parallel layers below it with thicknesses h_i and indices
n_i (i = 2..N), the point X in the last layer.  Every interface normal is the same unit vector nhat, so the ray
stays in the plane spanned by nhat and X - C, and Snell's law keeps q = n_i sin(theta_i) the same in every layer.
rho = lateral distance of X from C (perpendicular to nhat), h_1 = distance from C to the first interface, h_N =
depth of X below the last interface.

Claim A: with tau = tan(theta_1), layer i's lateral travel is h_i n1 tau / sqrt(n_i^2 + (n_i^2 - n1^2) tau^2).
Claim B: that term is increasing in tau, and concave on tau >= 0 when n_i > n1 (camera side the lowest index).
Claim C: F(tau) = sum of the lateral travels has F(0) = 0 and F'(0) = h1 + sum_i h_i n1 / n_i.
Claim D: (argument) Newton on F(tau) = rho from the paraxial tau_p = rho / F'(0) increases monotonically to the root.
Claim E: paraxial equivalence, normal view: u = f tau ~ f rho / (h1 + t n1/n2 + Z n1/n3) equals a pinhole in the
         water with f' = f n3/n1 placed n3 (h1/n1 + t/n2) behind the water-side surface.
Claim F: implicit derivative of the root: dtau/dtheta = -F_theta / F_tau, e.g. dtau/dh1 = -tau / F'(tau),
         dtau/drho = 1 / F'(tau).
"""
import sys

sys.path.insert(0, r"D:\Refraction\code")
import sympy as sp

from symcheck import check_eq, check_positive, check_limit, skip, summary, RANGES

tau, n1, ni, h, k = sp.symbols("tau n1 n_i h k", positive=True)
th1 = sp.atan(tau)

# ---- A: layer i's lateral travel from Snell's law
s1 = sp.sin(th1)
si = n1 * s1 / ni
tan_i = si / sp.sqrt(1 - si ** 2)
claimed = h * n1 * tau / sp.sqrt(ni ** 2 + (ni ** 2 - n1 ** 2) * tau ** 2)
RANGES[ni] = (1.0, 2.0)
RANGES[n1] = (1.0, 1.0001)
check_eq("A  h tan(theta_i) with sin(theta_i) = n1 sin(theta_1)/n_i, tau = tan(theta_1)", h * tan_i, claimed,
         domain=[ni - n1], squared=True)

# ---- B: increasing, and concave when n_i > n1 (parametrise n_i = n1 sqrt(1 + k^2), k > 0)
g = n1 * tau / sp.sqrt(ni ** 2 + (ni ** 2 - n1 ** 2) * tau ** 2)
gp = sp.diff(g, tau)
check_eq("B1 g'(tau) = n1 n_i^2 / (n_i^2 + (n_i^2 - n1^2) tau^2)^(3/2)", gp,
         n1 * ni ** 2 / (ni ** 2 + (ni ** 2 - n1 ** 2) * tau ** 2) ** sp.Rational(3, 2))
gk = g.subs(ni, n1 * sp.sqrt(1 + k ** 2))
check_positive("B2 g'(tau) > 0 for n_i = n1 sqrt(1 + k^2) > n1", sp.simplify(sp.diff(gk, tau)))
gk2 = sp.simplify(sp.diff(gk, tau, 2))
check_positive("B3 -g''(tau) > 0 (concave) for n_i = n1 sqrt(1 + k^2) > n1, tau > 0", -gk2)
check_eq("B4 g''(tau) = -3 n1 n_i^2 (n_i^2 - n1^2) tau / (n_i^2 + (n_i^2 - n1^2) tau^2)^(5/2)",
         sp.diff(g, tau, 2), -3 * n1 * ni ** 2 * (ni ** 2 - n1 ** 2) * tau
         / (ni ** 2 + (ni ** 2 - n1 ** 2) * tau ** 2) ** sp.Rational(5, 2))
check_limit("B5 g(tau) -> n1 / sqrt(n_i^2 - n1^2) as tau -> oo (each lower layer's travel per unit h is bounded)",
            g, tau, sp.oo, n1 / sp.sqrt(ni ** 2 - n1 ** 2), dir="-", domain=[ni - n1])

# ---- C: three layers (air, glass, water) as the case: F(0) = 0, F'(0)
h1, t, Z, n2, n3, rho = sp.symbols("h1 t Z n2 n3 rho", positive=True)
RANGES[n2] = (1.3, 1.8)
RANGES[n3] = (1.2, 1.5)


def lat(hh, nn):
    return hh * n1 * tau / sp.sqrt(nn ** 2 + (nn ** 2 - n1 ** 2) * tau ** 2)


F = h1 * tau + lat(t, n2) + lat(Z, n3)
check_eq("C1 F(0) = 0", F.subs(tau, 0), 0)
check_eq("C2 F'(0) = h1 + t n1/n2 + Z n1/n3", sp.diff(F, tau).subs(tau, 0), h1 + t * n1 / n2 + Z * n1 / n3)
skip("D  Newton from tau_p = rho/F'(0) increases monotonically to the root",
     "argument, not algebra: F concave increasing with F(0) = 0 (B3, C1) gives F(tau_p) <= F'(0) tau_p = rho, so "
     "tau_p <= root; from any tau_k <= root concavity gives F(root) <= F(tau_k) + F'(tau_k)(root - tau_k), so "
     "tau_k <= tau_{k+1} <= root; a bounded monotone sequence converges, to a fixed point, where F = rho")

# ---- E: paraxial equivalence for a normal view: u/rho as rho -> 0 (tau -> 0)
# the root tau(rho) has dtau/drho = 1/F'(tau); at rho = 0, tau = 0 and dtau/drho = 1/F'(0)
f = sp.symbols("f", positive=True)
du_drho_refr = f / sp.diff(F, tau).subs(tau, 0)
fp = f * n3 / n1
Zc = -n3 * (h1 / n1 + t / n2)
du_drho_pin = fp / (Z - Zc)
check_eq("E  normal view, paraxial: refractive du/drho = pinhole with f' = f n3/n1 at depth -n3 (h1/n1 + t/n2)",
         du_drho_refr, du_drho_pin)

# ---- F: implicit derivatives of the root of F(tau; h1, rho) = rho
Fp = sp.diff(F, tau)
G = F - rho
check_eq("F1 dtau/dh1 = -G_h1 / G_tau = -tau / F'(tau)", -sp.diff(G, h1) / sp.diff(G, tau), -tau / Fp)
check_eq("F2 dtau/drho = -G_rho / G_tau = 1 / F'(tau)", -sp.diff(G, rho) / sp.diff(G, tau), 1 / Fp)

# ---- G: scale.  Scaling every length (h1, t, Z, rho) by s scales the equation by s, so the root tau and hence the
# image are unchanged; with the thickness t held, only its term breaks the invariance.  So the window fixes the scale
# only through its thickness (plan 4.3).
s_ = sp.symbols("s", positive=True)
Fs = F.subs({h1: s_ * h1, t: s_ * t, Z: s_ * Z}, simultaneous=True)
check_eq("G1 F(tau; s h1, s t, s Z) - s rho = s (F(tau; h1, t, Z) - rho)", Fs - s_ * rho, s_ * (F - rho))
Ft = F.subs({h1: s_ * h1, Z: s_ * Z}, simultaneous=True)
check_eq("G2 with t held: F(tau; s h1, t, s Z) - s rho = s (F - rho) - (s - 1) t g_2(tau)", Ft - s_ * rho,
         s_ * (F - rho) - (s_ - 1) * lat(t, n2))

summary()
