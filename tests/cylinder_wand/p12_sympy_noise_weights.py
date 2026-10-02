"""SymPy check: can each camera's noise level be judged from the residuals?  (Conversation derivation, 2026-10-02.)

Claim 1 (one point, two cameras): the residual space is one-dimensional, P = n n^T, so E[r r^T] = P Sigma P =
(n^T Sigma n) n n^T and camera A's expected squared residual is (n^T D_A n)(n^T Sigma n).  The split between the
cameras, E|r_A|^2 / E|r_B|^2 = (n1^2 + n2^2) / (n3^2 + n4^2), does not depend on sigma_A, sigma_B: the residuals
only measure the combination n^T Sigma n, never which camera is noisier.
Claim 2 (a wand frame, two cameras, the length constraint): the residual space is 3-D and the split does depend on
the noise levels, so they are identifiable in principle; how strongly is shown on an example.
Claim 3 (one point, three cameras): the split depends on the noise levels.
"""
import sys

sys.path.insert(0, r"D:\Refraction\code")
import sympy as sp  # noqa: E402

from symcheck import check_eq, check_positive, check_value, summary  # noqa: E402

n1, n2, n3, n4 = sp.symbols("n1:5", real=True)
a, b, c = sp.symbols("a b c", positive=True)          # sigma_A^2, sigma_B^2, sigma_C^2

# ---- claim 1
n = sp.Matrix([n1, n2, n3, n4])
Sig = sp.diag(a, a, b, b)
DA, DB = sp.diag(1, 1, 0, 0), sp.diag(0, 0, 1, 1)
P = n * n.T
EA = (DA * P * Sig * P).trace()
EB = (DB * P * Sig * P).trace()
check_eq("1a: E|r_A|^2 = (n^T D_A n)(n^T Sigma n)", EA, (n.T * DA * n)[0] * (n.T * Sig * n)[0])
ratio = EA / EB
check_eq("1b: split E|r_A|^2/E|r_B|^2 independent of sigma_A^2", sp.diff(ratio, a), 0, domain=[n3 ** 2 + n4 ** 2 > 0])
check_eq("1c: split independent of sigma_B^2", sp.diff(ratio, b), 0, domain=[n3 ** 2 + n4 ** 2 > 0])
# the residual projector of a full-rank 4x3 Jacobian has rank 1 (rank P = 4 - rank J), on an exact example
J4 = sp.Matrix([[3, 1, -2], [1, 4, 1], [-2, 1, 5], [2, -3, 1]])
P4 = sp.eye(4) - J4 * (J4.T * J4).inv() * J4.T
check_eq("1d: residual projector rank (exact 4x3 example)", P4.rank(), 1)

# ---- claim 2: wand frame seen by two cameras; parameters m (3) and two direction angles; L/2 = 1, w = e_z
T = sp.Matrix([[1, 0], [0, 1], [0, 0]])                # dX_A/d(angles) = -T, dX_B/d(angles) = +T
GA = sp.Matrix([[5, 1, -2], [1, 4, 3]])               # camera A's point Jacobian (generic rationals)
GB = sp.Matrix([[2, -3, 1], [4, 1, -5]])              # camera B's
J8 = sp.Matrix.vstack(sp.Matrix.hstack(GA, -GA * T), sp.Matrix.hstack(GA, GA * T),
                      sp.Matrix.hstack(GB, -GB * T), sp.Matrix.hstack(GB, GB * T))
P8 = sp.eye(8) - J8 * (J8.T * J8).inv() * J8.T
check_eq("2a: frame residual projector rank 3 (8 - 5)", P8.rank(), 3)
Sig8 = sp.diag(*([a] * 4 + [b] * 4))
D8A = sp.diag(*([1] * 4 + [0] * 4))
D8B = sp.diag(*([0] * 4 + [1] * 4))
r8 = sp.simplify((D8A * P8 * Sig8 * P8).trace() / (D8B * P8 * Sig8 * P8).trace())
swap = sp.simplify(r8.subs({a: 1, b: 25}) - r8.subs({a: 25, b: 1}))
check_positive("2b: split changes when the noise moves from A to B (frame, length constraint)", swap ** 2)
v1, v2 = float(r8.subs({a: 1, b: 25})), float(r8.subs({a: 25, b: 1}))
print(f"   frame example: split A/B = {v1:.4f} with A quiet (1:25), {v2:.4f} with A noisy (25:1)")

# ---- claim 3: one point seen by three cameras
GC = sp.Matrix([[-1, 2, 4], [3, 2, 1]])
J6 = sp.Matrix.vstack(GA, GB, GC)
P6 = sp.eye(6) - J6 * (J6.T * J6).inv() * J6.T
Sig6 = sp.diag(a, a, b, b, c, c)
D6A = sp.diag(1, 1, 0, 0, 0, 0)
D6B = sp.diag(0, 0, 1, 1, 0, 0)
r6 = sp.simplify((D6A * P6 * Sig6 * P6).trace() / (D6B * P6 * Sig6 * P6).trace())
swap3 = sp.simplify(r6.subs({a: 1, b: 25, c: 1}) - r6.subs({a: 25, b: 1, c: 1}))
check_positive("3a: three cameras, one point: the split changes when the noise moves", swap3 ** 2)
w1, w2 = float(r6.subs({a: 1, b: 25, c: 1})), float(r6.subs({a: 25, b: 1, c: 1}))
print(f"   three-camera example: split A/B = {w1:.4f} with A quiet, {w2:.4f} with A noisy")
# ---- the estimator used by the code (variance components, MINQUE) on an exact example: one frame, 3 observations
# of 2 coordinates (6 rows), 2 frame parameters, groups = observations 1 | 2 | 3 (cameras)
t1, t2, t3 = sp.symbols("tau1:4", positive=True)          # true variances / current ones, per group
w1, w2, w3 = sp.symbols("w1:4", positive=True)           # weights 1/sigma^2 (symbolic in 4f-4g)
J = sp.Matrix([[2, 1], [1, -1], [3, 2], [-1, 1], [1, 3], [2, -2]])
D = [sp.diag(*([1] * 2 + [0] * 4)), sp.diag(*([0] * 2 + [1] * 2 + [0] * 2)), sp.diag(*([0] * 4 + [1] * 2))]
Sigt = t1 * D[0] + t2 * D[1] + t3 * D[2]                                   # whitened noise covariance
# 4a-4e hold for any weights (they follow from R~ being a symmetric idempotent); checked exactly, with the variances
# symbolic, for two weight sets (symbolic square-rooted weights made SymPy's simplification too slow)
for tag, ws in (("w = 1, 4, 9", (1, 4, 9)), ("w = 1/4, 1, 16", (sp.Rational(1, 4), 1, 16))):
    Wh = sp.diag(*[sp.sqrt(sp.nsimplify(x)) for x in ws for _ in range(2)])
    Aw = Wh * J
    Rt = sp.eye(6) - Aw * (Aw.T * Aw).inv() * Aw.T                        # whitened residual operator
    check_eq(f"4a ({tag}): R~ is symmetric (sum of squared entries of R~ - R~^T)", sum(e ** 2 for e in Rt - Rt.T), 0)
    check_eq(f"4b ({tag}): R~ is idempotent (sum of squared entries of R~^2 - R~)", sum(e ** 2 for e in Rt * Rt - Rt), 0)
    Mg = [[sp.nsimplify((D[g] * Rt * D[h] * Rt).trace()) for h in range(3)] for g in range(3)]
    for g in range(3):
        lhs = sp.expand((Rt * D[g] * Rt * Sigt).trace())                   # E[q_g] = E[e~^T R~ D_g R~ e~]
        check_eq(f"4c ({tag}): E[q_{g + 1}] = sum_h tau_h M_{g + 1}h", lhs,
                 sp.expand(sum(tt * Mg[g][h] for h, tt in enumerate((t1, t2, t3)))))
    for g in range(3):
        for h in range(g, 3):
            cov = 2 * (Rt * D[g] * Rt * Rt * D[h] * Rt).trace()            # Gaussian, tau = 1
            check_eq(f"4d ({tag}): Cov(q_{g + 1}, q_{h + 1}) = 2 M_{g + 1}{h + 1} at tau = 1", cov, 2 * Mg[g][h])
    check_eq(f"4e ({tag}): sum_h M_1h = redundancy trace(D_1 R~)", sum(Mg[0]), (D[0] * Rt).trace())
# weighted residual covariance used by the outlier test: with Sigma = W^-1 and H = J U^-1 J^T W, U = J^T W J,
# Cov(r) = (I - H) W^-1 (I - H)^T = W^-1 - J U^-1 J^T, so observation i's block is sigma_i^2 (I - w_i J_i U^-1 J_i^T).
# (Algebra: (I - H) W^-1 = W^-1 - J U^-1 J^T, and J U^-1 J^T W J U^-1 J^T = J U^-1 J^T.)  Checked exactly for the
# same two weight sets (with symbolic weights SymPy's cancellation of the 36 entries was too slow).
for tag, ws in (("w = 1, 4, 9", (1, 4, 9)), ("w = 1/4, 1, 16", (sp.Rational(1, 4), 1, 16))):
    W = sp.diag(*[sp.nsimplify(x) for x in ws for _ in range(2)])
    U = J.T * W * J
    H = J * U.inv() * J.T * W
    covr = (sp.eye(6) - H) * W.inv() * (sp.eye(6) - H).T
    check_eq(f"4f ({tag}): Cov(r) = W^-1 - J U^-1 J^T (sum of squared entries of the difference)",
             sum(e ** 2 for e in covr - (W.inv() - J * U.inv() * J.T)), 0)
    blk = (W.inv() - J * U.inv() * J.T)[0:2, 0:2]
    w_1 = sp.nsimplify(ws[0])
    check_eq(f"4g ({tag}): block 1 = sigma_1^2 (I - w_1 J_1 U^-1 J_1^T) (sum of squared entries of the difference)",
             sum(e ** 2 for e in blk - (1 / w_1) * (sp.eye(2) - w_1 * J[0:2, :] * U.inv() * J[0:2, :].T)), 0)
summary()
