"""L6 (plan, Phase 4): SymPy checks of the shape model's steps (shape_model.py's docstring), by the sympy-check
method of the notes repository (D:/Refraction/.claude/skills/sympy-check, helpers code/symcheck.py).

    python sympy_checks.py          (exits 1 on any FAIL)

Steps:
  A. the parallax of a displaced pinhole gives the depth: du = -f h / z, z = -2 f h / (u(+h) - u(-h));
  B. the defocus blur of a pupil of diameter A focused at z_f, for a (virtual) point at depth z: the rays spread over
     A |1 - z_f / z| in the plane in focus, f A |1/z - 1/z_f| pixels on the sensor;
  C. the segment profile g(t; L) of OTF::intensity: peak 1, integral w(L) = L / erf(L / (2 sqrt2 sigma)) (an
     antiderivative of Phi and the limit of the integral), and w -> sigma sqrt(2 pi) as L -> 0;
  D. the amplitude a = F / (w(L_h) w(L_v)) of a separable image of integral F;
  E. the second moments that a moment-matched Gaussian would take: a uniform segment L^2/12, a uniform disk of
     diameter D D^2/16 along any axis.
Writes sympy_checks_out.txt beside itself.
"""
import io
import os
import sys
from contextlib import redirect_stdout

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES_CODE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "code"))   # D:/Refraction/code
sys.path.insert(0, NOTES_CODE)
import symcheck as sc  # noqa: E402

buf = io.StringIO()


class Tee:
    def write(self, s):
        sys.__stdout__.write(s)
        buf.write(s)

    def flush(self):
        sys.__stdout__.flush()


def main():
    # ---- A. parallax of a displaced pinhole (camera coordinates: x_c across, z_c the depth)
    f, h, xc, zc = sp.symbols("f h x_c z_c", positive=True)
    u = lambda shift: f * (xc - shift) / zc          # the pinhole moved by `shift` along the camera's x
    sc.check_eq("A1 du = u(h) - u(0) = -f h / z", u(h) - u(0), -f * h / zc)
    sc.check_eq("A2 z = -2 f h / (u(+h) - u(-h)) is the depth", -2 * f * h / (u(h) - u(-h)), zc)

    # ---- B. defocus blur
    a, s, z, zf, A = sp.symbols("a s z z_f A", positive=True)
    x_at = a * (z - s) / z                            # the ray from the point (0, z) through the pupil point (a, 0)
    sc.check_eq("B1a the ray passes through the point (s = z)", x_at.subs(s, z), 0)
    sc.check_eq("B1b the ray passes through the pupil point (s = 0)", x_at.subs(s, 0), a)
    spread = (x_at.subs({s: zf, a: A / 2}) - x_at.subs({s: zf, a: -A / 2}))
    sc.check_eq("B2 spread in the plane in focus = A (1 - z_f / z)", spread, A * (1 - zf / z))
    blur_px = (f / zf) * spread                        # magnification f / z_f [px per mm] at the plane in focus
    sc.check_eq("B3 blur on the sensor = f A (1/z_f - 1/z) [px] (signed; |.| is the length)",
                blur_px, f * A * (1 / zf - 1 / z))
    sc.check_eq("B4 in focus (z = z_f) there is no blur", blur_px.subs(z, zf), 0)

    # ---- C. the segment profile
    t, L, sig = sp.symbols("t L sigma", positive=True)
    tt = sp.Symbol("t", real=True)
    Phi = lambda x: (1 + sp.erf(x / sp.sqrt(2))) / 2
    g_num = Phi((tt + L / 2) / sig) - Phi((tt - L / 2) / sig)
    den = 2 * Phi(L / (2 * sig)) - 1
    sc.check_eq("C1 the peak: g(0) = 1", (g_num / den).subs(tt, 0), 1)
    sc.check_eq("C2 2 Phi(L / 2 sigma) - 1 = erf(L / (2 sqrt2 sigma))", den, sp.erf(L / (2 * sp.sqrt(2) * sig)))
    uu = sp.Symbol("u", real=True)
    G = uu * Phi(uu) + sp.exp(-uu ** 2 / 2) / sp.sqrt(2 * sp.pi)
    sc.check_eq("C3 G(u) = u Phi(u) + phi(u) is an antiderivative of Phi", sp.diff(G, uu), Phi(uu))
    T = sp.Symbol("T", positive=True)
    integral_T = sig * (G.subs(uu, (T + L / 2) / sig) - G.subs(uu, (-T + L / 2) / sig)
                        - G.subs(uu, (T - L / 2) / sig) + G.subs(uu, (-T - L / 2) / sig))
    sc.check_eq("C4 d/dT of the integral over [-T, T] is the integrand at both ends",
                sp.diff(integral_T, T), g_num.subs(tt, T) + g_num.subs(tt, -T), how=(sp.simplify,))
    sc.check_limit("C5 the integral of the unnormalised profile over the line -> L", integral_T, T, sp.oo, L)
    w = L / sp.erf(L / (2 * sp.sqrt(2) * sig))
    sc.check_limit("C6 w(L) -> sigma sqrt(2 pi) as L -> 0", w, L, 0, sig * sp.sqrt(2 * sp.pi))

    # ---- D. the amplitude of a separable image
    F, wh, wv = sp.symbols("F w_h w_v", positive=True)
    amp = F / (wh * wv)
    sc.check_eq("D1 the image a g(x; L_h) g(y; L_v) integrates to a w_h w_v = F", amp * wh * wv, F)

    # ---- E. second moments
    D, R, r, th = sp.symbols("D R r theta", positive=True)
    var_seg = sp.integrate(tt ** 2, (tt, -L / 2, L / 2)) / L
    sc.check_eq("E1 a uniform segment of length L has variance L^2 / 12", var_seg, L ** 2 / 12)
    var_disk = sp.integrate(sp.integrate((r * sp.cos(th)) ** 2 * r, (r, 0, D / 2)), (th, 0, 2 * sp.pi)) \
        / (sp.pi * (D / 2) ** 2)
    sc.check_eq("E2 a uniform disk of diameter D has variance D^2 / 16 along any axis", var_disk, D ** 2 / 16)
    sc.skip("E3 a convolution adds the variances (segment or disk blurred by sigma0: + sigma0^2)",
            "a property of independent sums, not a derivation of this model")
    sc.skip("F1 alpha: the image direction across the tube's axis",
            "a definition (the projected axis turned by 90 deg), computed from the camera, not derived")
    sc.summary()


if __name__ == "__main__":
    with redirect_stdout(Tee()):
        try:
            main()
        except SystemExit as e:
            code = e.code
        else:
            code = 0
    with open(os.path.join(HERE, "sympy_checks_out.txt"), "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    sys.exit(code)
