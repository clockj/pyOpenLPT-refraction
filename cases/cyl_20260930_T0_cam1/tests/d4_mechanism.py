"""D4 (diagnosis): which mechanism explains the systematic residual best.

    python tests/d4_mechanism.py STATE.pkl [tag]

Each hypothesis is fitted to the residuals (detected minus model) of a fit with comparable freedom, u and v
separately, and the RMS left over is compared:

  H1 board error       r = P(dot)                       each dot displaced on the board (constant in px, to ~10 %)
  H2 image-fixed       r = H(u)    (or H(v) for v)      a lens/sensor error: fixed in the image, same in every plane
  H3 wall deflection   r = G(u) * L_k                   the tube wall deflects the ray at each pixel by a fixed
                                                        angle; the shift grows with the distance L_k from the
                                                        inner wall to plane k (rays are nearly parallel, so a pixel
                                                        always crosses the wall at the same place)
  H3b                  r = G(u) * L_k + H(u)            both
  H1+H2                r = P(dot) + H(u)
  H1+H3                r = P(dot) + G(u) * L_k        (H1+H2 against H1+H3 is the decisive pair)

H(.) and G(.) are cubic B-splines in the image coordinate with knots every 64 px (21 coefficients); P has one value
per dot (54).  L_k is taken from the fit: the distance along the viewing direction from the front of the inner
wall to plane k.  Writes d4_<tag>_out.txt beside this script.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import run_calibration as RC                  # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def bspline_basis(x, lo, hi, step):
    """Cubic B-spline basis on uniform knots covering [lo, hi]."""
    knots = np.arange(lo - 3 * step, hi + 4 * step, step)
    n = len(knots) - 4
    B = np.zeros((len(x), n))
    for q in range(n):
        t = (x - knots[q]) / step
        b = np.zeros_like(t)
        m0 = (t >= 0) & (t < 1)
        m1 = (t >= 1) & (t < 2)
        m2 = (t >= 2) & (t < 3)
        m3 = (t >= 3) & (t < 4)
        b[m0] = t[m0] ** 3 / 6
        b[m1] = (-3 * t[m1] ** 3 + 12 * t[m1] ** 2 - 12 * t[m1] + 4) / 6
        b[m2] = (3 * t[m2] ** 3 - 24 * t[m2] ** 2 + 60 * t[m2] - 44) / 6
        b[m3] = (4 - t[m3]) ** 3 / 6
        B[:, q] = b
    return B[:, B.any(axis=0)]


def left(r, G):
    coef, *_ = np.linalg.lstsq(G, r, rcond=None)
    rem = r - G @ coef
    return float(np.sqrt(np.mean(rem ** 2))), G.shape[1]


def main():
    st = RC.load_state(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "fit"
    obs, r, rig = st["obs"], st["resid"], st["rig"]
    z = rig.plane_z(obs.z_nom)
    a0 = rig.axis_at_y0()
    # distance from the front of the inner wall to each plane, along the viewing direction (~ -Z)
    wall_z = a0[2] + rig.radii[1]
    L = (wall_z - z)[obs.plane]
    say(f"D4 mechanism of the systematic residual ({sys.argv[1]}, model {st['info']['model']}); "
        f"L from {L.max():.1f} mm (z=0) to {L.min():.1f} mm (last plane)")
    keys = list(zip(obs.i.astype(int), obs.j.astype(int)))
    dots = sorted(set(keys))
    Pd = np.zeros((len(r), len(dots)))
    for q, k in enumerate(dots):
        Pd[[n for n, kk in enumerate(keys) if kk == k], q] = 1.0
    for comp, name, coord, span in ((0, "u", obs.uv[:, 0], (0, 1280)), (1, "v", obs.uv[:, 1], (0, 800))):
        rc = r[:, comp]
        Bu = bspline_basis(coord, span[0], span[1], 64.0)
        one = np.ones((len(rc), 1))
        say(f"\n{name} residual: rms {np.sqrt(np.mean(rc ** 2)):.4f} px")
        rows = [("H0 a constant", one),
                ("H1 board error P(dot)", Pd),
                (f"H2 image-fixed H({name})", Bu),
                (f"H3 wall deflection G({name}) L", Bu * L[:, None]),
                (f"H3b G({name}) L + H({name})", np.hstack([Bu * L[:, None], Bu])),
                (f"H1+H2 P(dot) + H({name})", np.hstack([Pd, Bu])),
                (f"H1+H3 P(dot) + G({name}) L", np.hstack([Pd, Bu * L[:, None]]))]
        if name == "v":
            Bu2 = bspline_basis(obs.uv[:, 0], 0, 1280, 64.0)
            rows += [("H3 with G(u) instead", Bu2 * L[:, None]), ("H1+H3(u)", np.hstack([Pd, Bu2 * L[:, None]]))]
        for label, G in rows:
            rm, p = left(rc, G)
            say(f"  {label:34s} {p:3d} coefficients -> leaves {rm:.4f} px")
    with open(os.path.join(HERE, f"d4_{tag}_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
