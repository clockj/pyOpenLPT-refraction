"""T13a2: what pyOpenLPT's wand detector returns for an ellipse (diagnostic for T13a's aspect-ratio sweep).  Plan:
2026-10-02-ball-centres-plan.md.

Two ellipses, (20 a) x 20 px and (30 a) x 30 px, at known positions, dark on bright, blur 1.5 px, no noise, for
aspect ratios a = 1.0 to 1.35.  Prints every circle that the "fast" mode (detect_circles_robust) and the C++
CircleIdentifier ("reliable") return, with the truth, so the failure mode can be seen.
Output: t13a2_circle_split_out.txt.
"""
import t13_common as T  # first: one BLAS thread per process

import numpy as np  # noqa: E402

import rigs  # noqa: E402
from t13a_render import draw_ellipses  # noqa: E402

log = rigs.Tee(str(rigs.HERE / "t13a2_circle_split") + "_out.txt")


def main():
    cA, cB = np.array([150.3, 150.2]), np.array([300.1, 149.7])
    log("T13a2: what pyOpenLPT's wand detector returns for two ellipses (truth: A at "
        f"({cA[0]}, {cA[1]}), b = 20 px; B at ({cB[0]}, {cB[1]}), b = 30 px; dark on bright)")
    for asp in (1.0, 1.1, 1.2, 1.25, 1.3, 1.35):
        img = draw_ellipses(450, 300, [(cA[0], cA[1], 20 * asp, 20), (cB[0], cB[1], 30 * asp, 30)], "dark")
        g = 255 - img
        fast = T._wand_calibrator().detect_circles_robust(g, T.MIN_R, T.MAX_R)
        cpp = T._cpp_circles(g)
        log(f"  a / b = {asp:.2f} (A {20 * asp:.0f} x 20 px, B {30 * asp:.0f} x 30 px)")
        log("    fast:     " + "; ".join(f"({p[0]:.2f}, {p[1]:.2f}) r {p[2]:.2f}" for p in fast))
        log("    reliable: " + "; ".join(f"({p[0]:.2f}, {p[1]:.2f}) r {p[2]:.2f}" for p in cpp))
    log.close()


if __name__ == "__main__":
    main()
