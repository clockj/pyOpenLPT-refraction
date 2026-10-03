"""W0 (SymPy): the derivatives of the flat-window projection that plate_model.project uses (implementation plan,
section 2.3), and the vector Snell step of plate_model.trace.  Run with the base `python` (sympy); output:
w0_sympy_derivations_out.txt.

Setting (plate_model's docstring): camera centre C, point X, window normal n (camera -> object), camera-side surface
n.X = d with d = n.A + delta (anchor A), one plate of thickness t and index n2, the point in medium n3, the camera in
n0.  v = X - C, a = n.v, l = v - a n, rho = |l|, h1 = d - n.C, hW = n.X - d - t, and tau solves
    Phi = h1 tau + t g(tau; n2) + hW g(tau; n3) - rho = 0,    g(tau; m) = n0 tau / sqrt(m^2 + (m^2 - n0^2) tau^2),
q = n + (tau / rho) l is the camera-side ray.  The derivatives of q with respect to X, C, delta and the normal's
increment (a, b) (n = exp_n(a t1 + b t2)) follow from the implicit function theorem, dtau/dtheta = -Phi_theta/Phi_tau,
and are compared with the closed forms in plate_model (with F' = Phi_tau, g_W = g(tau; n3), e = l/rho, e2 = n x e):
    J_qX = e (e - g_W n)^T / F' + (tau/rho) e2 e2^T
    J_qC = e (tau n - e)^T / F' - (tau/rho) e2 e2^T
    J_qd = e (g_W - tau) / F'
    J_qn = [I + e (-a e - tau (A - C) - g_W (X - A))^T / F' - (tau/rho) (n v^T + a (I - e e^T))] [t1 t2]
Without loss of generality the normal is e_z at the point of differentiation (then t1 = e_x, t2 = e_y, as
plate_model.tangent_basis gives), and after differentiating, C1 = C2 = X2 = 0 so that e = e_x, e2 = e_y.  tau is a
free symbol: the implicit derivative -Phi_theta/Phi_tau is an algebraic identity whether or not Phi = 0.
"""
import sys
from pathlib import Path

sys.path.insert(0, r"D:\Refraction\code")
import sympy as sp  # noqa: E402

import symcheck  # noqa: E402
from symcheck import check_eq, skip, summary, RANGES  # noqa: E402

HERE = Path(__file__).resolve().parent
lines = []
_print = print


def tee(*args):
    s = " ".join(str(a) for a in args)
    _print(s)
    lines.append(s)


symcheck.print = tee  # symcheck reports through print

X1, X2, X3, C1, C2, C3, A1, A2, A3, dl = sp.symbols("X1 X2 X3 C1 C2 C3 A1 A2 A3 delta", real=True)
al, be = sp.symbols("alpha beta", real=True)
tau, t, n0, n2, n3, r = sp.symbols("tau t n0 n2 n3 r", positive=True)
RANGES[n0] = (1.0, 1.05)
RANGES[n2] = (1.3, 1.8)
RANGES[n3] = (1.2, 1.5)
RANGES[tau] = (0.01, 2.0)

X = sp.Matrix([X1, X2, X3])
C = sp.Matrix([C1, C2, C3])
A = sp.Matrix([A1, A2, A3])
nv = sp.Matrix([al, be, 1]) / sp.sqrt(1 + al ** 2 + be ** 2)     # first order: e_z + alpha e_x + beta e_y


def g(m):
    return n0 * tau / sp.sqrt(m ** 2 + (m ** 2 - n0 ** 2) * tau ** 2)


d = (nv.T * A)[0] + dl
v = X - C
a = (nv.T * v)[0]
ell = v - a * nv
rho = sp.sqrt((ell.T * ell)[0])
h1 = d - (nv.T * C)[0]
hW = (nv.T * X)[0] - d - t
Phi = h1 * tau + t * g(n2) + hW * g(n3) - rho
q = nv + (tau / rho) * ell

params = [X1, X2, X3, C1, C2, C3, al, be, dl]
at0 = {al: 0, be: 0}
special = {C1: 0, C2: 0, X2: 0}
Phi_tau = sp.diff(Phi, tau)


def total(j):
    """d q / d params[j] at the special point."""
    p = params[j]
    dq = sp.diff(q, p) + sp.diff(q, tau) * (-sp.diff(Phi, p) / Phi_tau)
    return dq.subs(at0).subs(special).subs(X1, r)


def at(e):
    return sp.sympify(e).subs(at0).subs(special).subs(X1, r)


# the claimed closed forms at n = e_z, e = e_x, e2 = e_y, rho = r
nz = sp.Matrix([0, 0, 1])
e = sp.Matrix([1, 0, 0])
e2 = sp.Matrix([0, 1, 0])
Fp = at(Phi_tau)
gW = g(n3)
aa = at(a)
vv = at(v)
ratio = tau / r
JqX = e * (e - gW * nz).T / Fp + ratio * e2 * e2.T
JqC = e * (tau * nz - e).T / Fp - ratio * e2 * e2.T
Jqd = e * (gW - tau) / Fp
Ac = A
Cc = at(C)
Xc = at(X)
wn = -aa * e - tau * (Ac - Cc) - gW * (Xc - Ac)
Jn = sp.eye(3) + e * wn.T / Fp - ratio * (nz * vv.T + aa * (sp.eye(3) - e * e.T))
Tb = sp.Matrix([[1, 0], [0, 1], [0, 0]])
Jqn = Jn * Tb

dom = [n2 - n0, n3 - n0]
tee("-- block 1: the implicit derivatives of the camera-side ray q (n = e_z, e = e_x)")
check_eq("1.0 rho = r at the special point (e = e_x)", at(rho), r)
check_eq("1.1 Phi_tau = F'(tau) = h1 + t g'(tau; n2) + hW g'(tau; n3)", Fp,
         at(h1) + t * n0 * n2 ** 2 / (n2 ** 2 + (n2 ** 2 - n0 ** 2) * tau ** 2) ** sp.Rational(3, 2)
         + at(hW) * n0 * n3 ** 2 / (n3 ** 2 + (n3 ** 2 - n0 ** 2) * tau ** 2) ** sp.Rational(3, 2), domain=dom)
names = ["X1", "X2", "X3", "C1", "C2", "C3"]
for j in range(3):
    col = total(j)
    for i in range(3):
        check_eq(f"1.2 J_qX[{i},{j}] (d q_{i + 1} / d {names[j]})", col[i], JqX[i, j], domain=dom)
for j in range(3):
    col = total(3 + j)
    for i in range(3):
        check_eq(f"1.3 J_qC[{i},{j}] (d q_{i + 1} / d {names[3 + j]})", col[i], JqC[i, j], domain=dom)
col = total(8)
for i in range(3):
    check_eq(f"1.4 J_qdelta[{i}]", col[i], Jqd[i], domain=dom)
for j, nm in enumerate(("alpha (t1)", "beta (t2)")):
    col = total(6 + j)
    for i in range(3):
        check_eq(f"1.5 J_qn[{i},{j}] (d q_{i + 1} / d {nm}, the surface pivoting about A)", col[i], Jqn[i, j],
                 domain=dom)

tee("-- block 2: the limits at rho -> 0 (the point on the camera's normal line)")
# tau(rho) / rho -> 1/F'(0): tau(0) = 0 and dtau/drho = -Phi_rho/Phi_tau = 1/F'(tau)
Fp0 = at(h1) + t * n0 / n2 + at(hW) * n0 / n3
check_eq("2.1 F'(0) = h1 + t n0/n2 + hW n0/n3", Fp.subs(tau, 0), Fp0)
rr = sp.Symbol("rho", positive=True)
Phi_r = at(h1) * tau + t * g(n2) + at(hW) * g(n3) - rr
check_eq("2.2 dtau/drho = -Phi_rho / Phi_tau = 1 / F'(tau) (so tau/rho -> 1/F'(0) as rho -> 0)",
         -sp.diff(Phi_r, rr) / sp.diff(Phi_r, tau), 1 / Fp, domain=dom)
skip("2.2b tau/rho -> 1/F'(0) as rho -> 0", "follows from 2.2 and tau(0) = 0 by l'Hopital; plate_model uses "
     "ratio = 1/F'(0) when rho == 0 and checks it against central differences in W1")
check_eq("2.3 g(0; n3) = 0", g(n3).subs(tau, 0), 0)
phi = sp.symbols("phi", real=True)
for nm, ee in (("e = e_x", sp.Matrix([1, 0, 0])), ("e = (cos phi, sin phi, 0)", sp.Matrix([sp.cos(phi), sp.sin(phi), 0]))):
    ee2 = nz.cross(ee)
    lim_X = ee * ee.T / Fp0 + ee2 * ee2.T / Fp0
    P = sp.eye(3) - nz * nz.T
    for i in range(3):
        for j in range(3):
            check_eq(f"2.4 J_qX -> (I - n n^T)/F'(0) at rho = 0, {nm}, [{i},{j}]", lim_X[i, j], P[i, j] / Fp0)
    va = aa * nz                                   # rho = 0: v is along n
    lim_n = (sp.eye(3) + ee * (-aa * ee).T / Fp0 - (nz * va.T + aa * (sp.eye(3) - ee * ee.T)) / Fp0) * Tb
    for i in range(3):
        for j in range(2):
            check_eq(f"2.5 J_qn -> (1 - a/F'(0)) [t1 t2] at rho = 0, {nm}, [{i},{j}]", lim_n[i, j],
                     (1 - aa / Fp0) * Tb[i, j])
skip("2.6 the (A - C) and (X - A) terms of J_qn at rho = 0", "they are multiplied by tau and g_W, both 0 at rho = 0")

tee("-- block 3: the rotation and the pinhole")
w1, w2, w3, q1, q2, q3 = sp.symbols("w1 w2 w3 q1 q2 q3", real=True)
W = sp.Matrix([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
qv = sp.Matrix([q1, q2, q3])
expW = sp.eye(3) + W + W * W / 2                   # exp([w]x) to second order: enough for the derivative at 0
Qx = sp.Matrix([[0, -q3, q2], [q3, 0, -q1], [-q2, q1, 0]])
for i, wi in enumerate((w1, w2, w3)):
    dcol = sp.diff(expW * qv, wi).subs({w1: 0, w2: 0, w3: 0})
    for k in range(3):
        check_eq(f"3.1 d(exp([w]x) q)/dw{i + 1} = -[q]x e{i + 1}, row {k}", dcol[k], (-Qx)[k, i])
p1, p2, p3 = sp.symbols("p1 p2 p3", real=True)
xq = p1 / p3
check_eq("3.2 dx/dp1 = 1/p3", sp.diff(xq, p1), 1 / p3, domain=[p3])
check_eq("3.3 dx/dp3 = -x/p3", sp.diff(xq, p3), -xq / p3, domain=[p3])

tee("-- block 4: the vector Snell step of trace (q unit, n = e_z)")
s1, s2, mu = sp.symbols("s1 s2 mu", real=True)
RANGES[s1] = (-0.4, 0.4)
RANGES[s2] = (-0.4, 0.4)
RANGES[mu] = (0.6, 1.0)
ci = sp.sqrt(1 - s1 ** 2 - s2 ** 2)
qq = sp.Matrix([s1, s2, ci])
ct = sp.sqrt(1 - mu ** 2 * (1 - ci ** 2))
qn = mu * qq + (ct - mu * ci) * nz
dom4 = [1 - s1 ** 2 - s2 ** 2, mu, 1 - mu ** 2 * (s1 ** 2 + s2 ** 2)]
check_eq("4.1 |q_new|^2 = 1", sp.expand((qn.T * qn)[0]), 1, domain=dom4)
check_eq("4.2 n.q_new = c_t (the ray keeps going into the window)", qn[2], ct, domain=dom4)
check_eq("4.3 tangential part scales by mu = n_old/n_new (n sin(theta) conserved), x", qn[0], mu * s1, domain=dom4)
check_eq("4.3 tangential part scales by mu, y", qn[1], mu * s2, domain=dom4)

tee("")
try:
    summary()
finally:
    (HERE / "w0_sympy_derivations_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
