"""SymPy checks of the short derivations behind Phase E (plan section "Phase E: the detailed steps").

The claims, about STB::predictNext (src/srcSTB/STB.cpp) and the predictors E1 compares it with:
  (1) its normalised-LMS loop stops after one pass, with p = y_m * sum_j y_j y_(j+1) / sum_j y_j^2 (j = 0..m-1);
  (2) far from the origin (every position shifted by C -> oo) p tends to the mean step, y_m + (y_m - y_0)/m;
  (3) for linear motion y_j = a + b j the error of p is (m+1) b^2 / (2a) to first order, away from 0; the numbers
      the plan quotes (0.009-0.037 mm at 1-3 mm from 0, 0.003 mm at 10 mm, for 0.1 mm/frame);
  (4) under constant acceleration A the mean step (m = 5) misses by 3A and a 6-point least-squares line by 14A/3;
      3A = 0.024 mm at A = 0.008 mm/frame^2;
  (5) the noise variance of the mean step is (1 + 1/m)^2 + 1/m^2 times the position noise, of the 6-point line
      extrapolated one frame 13/15 times;
  (6) a Kalman filter's gains do not change when P, Q and r are scaled together, so from a diffuse start its
      predictions depend only on q/r;
  (7) the shake's first step, 0.25 voxel at 0.129 mm per voxel, is 0.032 mm.

Uses D:/Refraction/code/symcheck.py (the project's SymPy helpers).  Writes results/e1_sympy_check.txt.
    python tests/stb_improve/e1_sympy_check.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
SYMCHECK_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "code"))   # D:/Refraction/code
sys.path.insert(0, SYMCHECK_DIR)
import symcheck as sc  # noqa: E402
from symcheck import check_eq, check_limit, check_positive, check_value, skip  # noqa: E402


def lms_one_pass(y):
    """The C++ loop's first pass on the samples y = [y_0..y_m]: returns (fit of y_m after the pass, prediction)."""
    m = len(y) - 1
    S = sum(y[j] ** 2 for j in range(m))
    step = 1 / S
    error = y[m] - 0                              # prediction starts at 0
    filt = [step * y[j] * error for j in range(m)]  # filter starts at 0; one update each
    fit = sum(filt[j] * y[j] for j in range(m))
    pred = sum(filt[j] * y[j + 1] for j in range(m))
    return fit, pred


def wiener_formula(y):
    m = len(y) - 1
    return y[m] * sum(y[j] * y[j + 1] for j in range(m)) / sum(y[j] ** 2 for j in range(m))


def wiener_value(ys):
    """The formula with the C++ shift rule (+10 mm when |y_m| < 1), in exact rationals."""
    ys = [sp.nsimplify(v, rational=True) for v in ys]
    shift = 10 if abs(ys[-1]) < 1 else 0
    return wiener_formula([v + shift for v in ys]) - shift


def main():
    a = sp.Symbol("a", positive=True)          # the coordinate (mm), positive side
    b = sp.Symbol("b", real=True)              # velocity (mm/frame)
    A = sp.Symbol("A", real=True)              # acceleration (mm/frame^2)
    c0, c1 = sp.symbols("c0 c1", real=True)
    C = sp.Symbol("C", positive=True)          # a shift of the origin
    mm = sp.Symbol("m", positive=True, integer=True)
    j = sp.Symbol("j", integer=True, nonnegative=True)

    print("(1) the LMS loop stops after one pass")
    for m in (3, 4, 5):
        y = sp.symbols(f"y0:{m + 1}", real=True)
        fit, pred = lms_one_pass(list(y))
        check_eq(f"(1) m={m}: after one pass the fit of y_m is exact (error 0, the loop ends)", fit, y[m])
        check_eq(f"(1) m={m}: prediction = y_m sum y_j y_j+1 / sum y_j^2", pred, wiener_formula(list(y)))

    print("\n(2) far from the origin: the mean step")
    for m in (3, 4, 5):
        y = sp.symbols(f"y0:{m + 1}", real=True)
        shifted = wiener_formula([v + C for v in y]) - C
        check_limit(f"(2) m={m}: p(y + C) - C -> y_m + (y_m - y_0)/m as C -> oo", shifted, C, sp.oo,
                    y[m] + (y[m] - y[0]) / m)

    print("\n(3) linear motion y_j = a + b j: the error near the origin")
    for m in (3, 4, 5):
        y = [a + b * k for k in range(m + 1)]
        err = wiener_formula(y) - (a + b * (m + 1))
        check_limit(f"(3) m={m}: error / b^2 -> (m+1)/(2a) as b -> 0", err / b ** 2, b, 0, sp.Rational(m + 1, 2) / a)
        # sign: the error points away from 0 whenever all positions are on the positive side
        num = sp.factor(sp.together(err).as_numer_denom()[0])
        den = sp.factor(sp.together(err).as_numer_denom()[1])
        print(f"    m={m}: error = {num} / ({den})")
    # general m, with the sums in closed form
    S1 = sp.summation((a + b * j) * (a + b * (j + 1)), (j, 0, mm - 1))
    S2 = sp.summation((a + b * j) ** 2, (j, 0, mm - 1))
    err_m = (a + b * mm) * S1 / S2 - (a + b * (mm + 1))
    check_limit("(3) any m: error / b^2 -> (m+1)/(2a) as b -> 0", err_m / b ** 2, b, 0, (mm + 1) / (2 * a))
    # m = 5: away from 0 for every motion that keeps y_0..y_5 positive (linear motion: y_0 = a > 0, y_5 = v > 0)
    v = sp.Symbol("v", positive=True)          # v = a + 5 b, the last position
    bv = (v - a) / 5
    y5 = [a + bv * k for k in range(6)]
    err5 = wiener_formula(y5) - (a + 6 * bv)
    form = bv ** 2 * (3 * a + 4 * bv) / ((a + 2 * bv) ** 2 + 2 * bv ** 2)
    check_eq("(3) m=5: error = b^2 (3a + 4b) / ((a + 2b)^2 + 2b^2)", err5, form)
    check_positive("(3) m=5: 3a + 4b > 0 when y_0, y_5 > 0", sp.expand(3 * a + 4 * bv))
    check_positive("(3) m=5: (a + 2b)^2 + 2b^2 > 0 when y_0, y_5 > 0", sp.factor(sp.expand((a + 2 * bv) ** 2)) +
                   2 * bv ** 2)
    check_positive("(3) m=5: so error >= 0 (away from 0; 0 only at rest) when y_0, y_5 > 0", form, strict=False)
    # the numbers the plan quotes (0.1 mm/frame, coordinates 1-3 mm and 10 mm)
    def err_at(ym, v):
        ys = [ym - v * (5 - k) for k in range(6)]
        return wiener_value(ys) - (sp.nsimplify(ym, rational=True) + sp.nsimplify(v, rational=True))
    grid = [sp.Rational(100 + 5 * k, 100) for k in range(41)]                  # 1.00 .. 3.00 mm
    vals = [err_at(g, v) for g in grid for v in (sp.Rational(1, 10), -sp.Rational(1, 10))]
    check_value("(3) largest error at 1-3 mm from 0, 0.1 mm/frame (plan: 0.037 mm)", max(vals), 0.037, 3)
    check_value("(3) smallest error at 1-3 mm from 0, 0.1 mm/frame (plan: 0.009 mm)", min(vals), 0.009, 3)
    check_positive("(3) all those errors point away from 0", min(vals))
    check_value("(3) error at 10 mm, +0.1 mm/frame (plan: 0.003 mm)", err_at(10, sp.Rational(1, 10)), 0.003, 3)
    check_value("(3) error at 10 mm, -0.1 mm/frame (plan: 0.003 mm)", err_at(10, -sp.Rational(1, 10)), 0.003, 3)

    print("\n(4) constant acceleration: the misses")
    y = [c0 + c1 * k + A * k ** 2 / 2 for k in range(7)]
    md = y[5] + (y[5] - y[0]) / 5
    check_eq("(4) mean step (m=5) misses by 3A", y[6] - md, 3 * A)
    t = sp.Matrix([[1, k] for k in range(6)])
    coef = (t.T * t).inv() * t.T * sp.Matrix(y[:6])
    lin6 = coef[0] + coef[1] * 6
    check_eq("(4) 6-point least-squares line misses by 14A/3", sp.simplify(y[6] - lin6), sp.Rational(14, 3) * A)
    check_value("(4) 3A at A = 0.008 mm/frame^2 (plan: 0.024 mm)", 3 * sp.Rational(8, 1000), 0.024, 3)

    print("\n(5) noise: variance factors")
    w_md = {5: 1 + sp.Rational(1, 5), 0: -sp.Rational(1, 5)}
    check_eq("(5) mean step, m=5: sum of squared weights = (1+1/m)^2 + 1/m^2",
             sum(v ** 2 for v in w_md.values()), (1 + sp.Rational(1, 5)) ** 2 + sp.Rational(1, 25))
    w_lin = (sp.Matrix([[1, 6]]) * (t.T * t).inv() * t.T)
    check_eq("(5) 6-point line at one frame ahead: sum of squared weights = 13/15",
             sum(v ** 2 for v in w_lin), sp.Rational(13, 15))
    check_eq("(5) the line's weights sum to 1 (a constant is predicted exactly)", sum(w_lin), 1)

    print("\n(6) Kalman gains depend only on q/r")
    p11, p12, p22, r, q, cc = sp.symbols("p11 p12 p22 r q c", positive=True)
    P = sp.Matrix([[p11, p12], [p12, p22]])
    F = sp.Matrix([[1, 1], [0, 1]])
    G = sp.Matrix([sp.Rational(1, 2), 1])
    H = sp.Matrix([[1, 0]])

    def step(P, r, q):
        S = (H * P * H.T)[0, 0] + r
        K = P * H.T / S
        Pp = (sp.eye(2) - K * H) * P
        return K, F * Pp * F.T + q * G * G.T
    K1, Pn1 = step(P, r, q)
    K2, Pn2 = step(cc * P, cc * r, cc * q)
    for i in range(2):
        check_eq(f"(6) CV gain K[{i}] unchanged when P, q, r are scaled by c", K2[i], K1[i])
    for i, k in ((0, 0), (0, 1), (1, 1)):
        check_eq(f"(6) CV predicted covariance P[{i},{k}] scales by c", Pn2[i, k], cc * Pn1[i, k])
    skip("(6) diffuse start: P0 = kappa I with kappa -> oo for both scalings",
         "an induction on the step above: P0 scaled by c is again diffuse, so every gain depends on q/r only "
         "(the same algebra holds for the constant-acceleration model; e1d checks both numerically)")

    print("\n(7) the shake's first step")
    check_value("(7) 0.25 voxel x 0.129 mm (plan: 0.032 mm)", sp.Rational(25, 100) * sp.Rational(129, 1000), 0.032, 3)


if __name__ == "__main__":
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)
    code = 0
    with contextlib.redirect_stdout(Tee()):
        try:
            main()
            sc.summary()
        except SystemExit as e:
            code = e.code or 0
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "e1_sympy_check.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(buf.getvalue())
    sys.exit(code)
