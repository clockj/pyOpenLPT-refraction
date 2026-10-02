"""SymPy checks of the derivation steps in the cylinder wand-calibration plan (conversation of 2026-10-01).

Blocks:
  1. metric upgrade of an affine reconstruction from the wand length;
  2. the projection Jacobian by implicit differentiation (generic, rescaling invariance, a pinhole instance with the
     same perpendicular basis as cylfit / the planned cyl_model);
  3. paraxial magnification of a point at the centre of curvature, and the flat-window factor along the axis.
Run with plain `python` from anywhere; imports code/symcheck.py.
"""
import sys

sys.path.insert(0, r"D:\Refraction\code")
import sympy as sp  # noqa: E402

from symcheck import check_eq, check_value, skip, summary  # noqa: E402

# ------------------------------------------------------------------ 1. metric upgrade
A = sp.Matrix(3, 3, sp.symbols("a0:9", real=True))
d = sp.Matrix(sp.symbols("d0:3", real=True))
Q = A.T * A
check_eq("1.1 |A d|^2 = d^T (A^T A) d", (A * d).dot(A * d), (d.T * Q * d)[0])

q00, q01, q02, q11, q12, q22 = sp.symbols("q00 q01 q02 q11 q12 q22", real=True)
Qs = sp.Matrix([[q00, q01, q02], [q01, q11, q12], [q02, q12, q22]])
d0, d1, d2 = d
lin = q00 * d0**2 + q11 * d1**2 + q22 * d2**2 + 2 * q01 * d0 * d1 + 2 * q02 * d0 * d2 + 2 * q12 * d1 * d2
check_eq("1.2 d^T Q d is linear in the 6 entries of symmetric Q (explicit form)", (d.T * Qs * d)[0], lin)
for qa in (q00, q01, q02, q11, q12, q22):
    for qb in (q00, q01, q02, q11, q12, q22):
        if sp.diff((d.T * Qs * d)[0], qa, qb) != 0:
            raise SystemExit("1.2: a second derivative in Q is nonzero")
check_eq("1.2b all second derivatives of d^T Q d in Q vanish (linear)", sp.Integer(0), sp.Integer(0))

# Q fixes A only up to an orthogonal factor: if A'^T A' = A^T A with A invertible, O = A' A^-1 is orthogonal:
# O^T O = A^-T (A'^T A') A^-1 = A^-T (A^T A) A^-1. With A^-1 = adj(A)/det(A) (1.3a) this is
# adj^T (A^T A) adj / det^2, and (1.3b) adj^T A^T A adj = (A adj)^T (A adj) = det^2 I, a polynomial identity.
adj = A.adjugate()
Aadj = A * adj
for i in range(3):
    for j in range(3):
        check_eq(f"1.3a A adj(A) = det(A) I, entry ({i},{j})", Aadj[i, j], A.det() if i == j else 0,
                 how=(sp.expand,))
M13 = adj.T * Q * adj
for i in range(3):
    for j in range(i, 3):
        check_eq(f"1.3b adj^T (A^T A) adj = det(A)^2 I, entry ({i},{j})", M13[i, j],
                 A.det() ** 2 if i == j else 0, how=(sp.expand,))
P = sp.diag(1, 1, -1)
check_eq("1.4a the mirror P = diag(1,1,-1) has det -1", P.det(), -1)
PA = P * A
for i in range(3):
    for j in range(3):
        check_eq(f"1.4b (P A)^T (P A) = A^T A, entry ({i},{j})", (PA.T * PA)[i, j], Q[i, j])

# ------------------------------------------------------------------ 2. implicit differentiation
# 2.1 generic: g(u(t), t) = 0 for all t  =>  g_u u' + g_t = 0  =>  u' = -g_u^-1 g_t
gu = sp.Matrix(2, 2, sp.symbols("gu0:4", real=True))
gt = sp.Matrix(sp.symbols("gt0:2", real=True))
uprime = -gu.inv() * gt
res = gu * uprime + gt
for k in range(2):
    check_eq(f"2.1 g_u (-g_u^-1 g_t) + g_t = 0, row {k}", res[k], 0, domain=[gu.det() ** 2], how=(sp.cancel,))

# 2.2 rescaling invariance: gt = S g with S invertible; at g = 0, gt_u = S g_u and gt_t = S g_t, so the implicit
# derivative is the same for any such S (the basis e1, e2 and its normalisation do not matter).
S = sp.Matrix(2, 2, sp.symbols("s0:4", real=True))
lhs22 = -(S * gu).inv() * (S * gt)
for k in range(2):
    check_eq(f"2.2 -(S g_u)^-1 (S g_t) = -g_u^-1 g_t, row {k}", lhs22[k], uprime[k],
             domain=[S.det() ** 2, gu.det() ** 2], how=(sp.cancel,))
skip("2.0 the implicit function theorem itself (existence and smoothness of u*(theta))",
     "a theorem, not algebra; it needs g_u invertible at the solution, which T1 tests on the cylinder model")

# 2.3 pinhole instance. The implementation's basis is dhat = normalise((u-cx)/f, (v-cy)/f, 1),
# e1 = normalise(ex - (ex.dhat) dhat), e2 = dhat x e1. Up to positive scalar factors (a diagonal S in 2.2) this is
# E1 = |d|^2 ex - d_x d and E2 = d x E1 with d = ((u-cx)/f, (v-cy)/f, 1); both give the same implicit derivative
# (2.2), so the exact check uses the rational form (2.3a-c) and the normalised form is sampled (2.3e).
u, v, cx, cy = sp.symbols("u v cx cy", real=True)
f = sp.Symbol("f", positive=True)
x, y = sp.symbols("x y", real=True)
z = sp.Symbol("z", positive=True)
dvec = sp.Matrix([(u - cx) / f, (v - cy) / f, 1])
ex = sp.Matrix([1, 0, 0])
X = sp.Matrix([x, y, z])
star = {u: cx + f * x / z, v: cy + f * y / z}          # the closed-form pinhole projection
J_X_true = sp.Matrix([[f / z, 0, -f * x / z**2], [0, f / z, -f * y / z**2]])


def implicit(e1, e2, how):
    g = sp.Matrix([e1.dot(X), e2.dot(X)])
    Gu = g.jacobian([u, v]).subs(star)
    inv = -Gu.inv()
    return (inv * g.jacobian([x, y, z]).subs(star), inv * g.jacobian([f]).subs(star),
            inv * g.jacobian([cx, cy]).subs(star), g)


E1 = dvec.dot(dvec) * ex - dvec[0] * dvec
E2 = dvec.cross(E1)
J_X, J_f, J_c, g_rat = implicit(E1, E2, None)
for i in range(2):
    for j in range(3):
        check_eq(f"2.3a pinhole (rational basis): implicit d(u,v)/dX = closed form, entry ({i},{j})",
                 J_X[i, j], J_X_true[i, j], how=(sp.cancel,))
    check_eq(f"2.3b pinhole (rational basis): implicit d(u,v)/df = (x/z, y/z), row {i}", J_f[i, 0],
             [x / z, y / z][i], how=(sp.cancel,))
    for j in range(2):
        check_eq(f"2.3c pinhole (rational basis): implicit d(u,v)/d(cx,cy) = I, entry ({i},{j})", J_c[i, j],
                 1 if i == j else 0, how=(sp.cancel,))
# the rational basis is the normalised one times positive factors, and it is perpendicular to the ray
dhat = dvec / sp.sqrt(dvec.dot(dvec))
e1n = ex - ex.dot(dhat) * dhat
e1n = e1n / sp.sqrt(e1n.dot(e1n))
e2n = dhat.cross(e1n)
check_eq("2.3d E1 . d = 0 (rational basis perpendicular to the ray)", E1.dot(dvec), 0, how=(sp.expand,))
check_eq("2.3d E2 . d = 0", E2.dot(dvec), 0, how=(sp.expand,))
# the implemented (normalised) basis is the rational one divided by its lengths, a positive diagonal S of 2.2
nE1 = sp.sqrt(E1.dot(E1))
nE2 = sp.sqrt(E2.dot(E2))
for j in range(3):
    check_eq(f"2.3e e1 (normalised) = E1/|E1|, component {j}", e1n[j], E1[j] / nE1, how=(sp.cancel,))
    check_eq(f"2.3e e2 (normalised) = E2/|E2|, component {j}", e2n[j], E2[j] / nE2, how=(sp.cancel,))
g_n = sp.Matrix([e1n.dot(X), e2n.dot(X)])
# g_X is the perpendicular basis itself: g is linear in X with coefficients e1, e2
GXn = g_n.jacobian([x, y, z])
for j in range(3):
    check_eq(f"2.3f g_X row 0 = e1, component {j}", GXn[0, j], e1n[j])
    check_eq(f"2.3f g_X row 1 = e2, component {j}", GXn[1, j], e2n[j])

# ------------------------------------------------------------------ 3. paraxial magnification
# Inputs (Gaussian convention): n1/s + n2/s' = (n2 - n1)/R and m = -(n1 s')/(n2 s); light goes from the object
# medium n1 to n2; R > 0 when the centre of curvature is on the outgoing side.
rho, n1, n2, n3 = sp.symbols("rho n1 n2 n3", positive=True)
s_img = -rho                                   # claimed image distance for an object at the centre
R_signed = -rho                                # centre of curvature on the incoming (object) side
s_obj = rho                                    # object at the centre: a distance rho in front of the surface
check_eq("3.1 object at the centre of curvature images onto itself: s' = -rho solves n1/s + n2/s' = (n2-n1)/R",
         n1 / s_obj + n2 / s_img, (n2 - n1) / R_signed)
m1 = -(n1 * s_img) / (n2 * s_obj)
check_eq("3.2 its lateral magnification is n1/n2", m1, n1 / n2)
# two concentric surfaces (water n1 -> acrylic n2 -> air n3): the centre is imaged onto itself by each in turn
m_total = (n1 / n2) * (n2 / n3)
check_eq("3.3 two concentric surfaces: total magnification n_obj/n_cam", m_total, n1 / n3)
check_value("3.4 C++ probe on cam1, across the axis at the axis point: 50.517/37.892 px/mm vs n_water = 1.333",
            sp.Float(50.517) / sp.Float(37.892), "1.333", 3)
# along the axis the interfaces are straight lines (zero curvature along the generator): flat-window apparent
# distance D_app = (D - R_out) + (R_out - R_in) n_air/n_acr + R_in n_air/n_w, scale ratio D / D_app
D, Ro, Ri, na, nw = (sp.Float("1094.13"), sp.Float("100.80625"), sp.Float("97.63125"), sp.Float("1.49"),
                     sp.Float("1.333"))
D_app = (D - Ro) + (Ro - Ri) / na + Ri / nw
check_value("3.5 along the axis: D/D_app (flat window) vs the probe's x1.0239", D / D_app, "1.024", 3)
skip("3.6 'along the axis the tube acts as a flat window' as a physical statement",
     "physics (the cylinder's generator is straight), supported by 3.5 against the C++ model")

summary()
