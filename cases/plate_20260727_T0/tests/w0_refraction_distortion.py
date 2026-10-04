"""W0 (2026-10-03-walls-plan.md, section 3): how much distortion does refraction itself put into each camera's image,
beyond what a homography absorbs?  It checks that refraction cannot pass for the fitted k1 (about 6 px at the image
corners), and gives the size of the signal that places the walls.

For each camera of the delivered fit (tests/states/t2_all.pkl) with k1 set to 0, so that only refraction bends the
rays: points on planes parallel to its wall, at water depths 30 and 85 mm (the wand's span from wall 1), on a grid
filling the image about the optical axis, projected through the plate model.  The best homography from each plane to
the image is fitted (DLT, then least squares).  Its largest residual is the refraction's distortion.

    python tests/w0_refraction_distortion.py
Writes w0_refraction_distortion_out.txt beside this script.
"""
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

DEPTHS = (30.0, 85.0)
HALF_MM, STEPS = 80.0, 41


def homography_residual(P2, uv):
    """Distances (px) between uv and the best homography of the plane points P2 (m, 2)."""
    A = []
    for (x, y), (u, v) in zip(P2, uv):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    H = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)

    def res(h):
        q = (np.append(h, 1.0).reshape(3, 3) @ np.column_stack([P2, np.ones(len(P2))]).T).T
        return (q[:, :2] / q[:, 2:3] - uv).ravel()

    return np.linalg.norm(least_squares(res, (H / H[2, 2]).ravel()[:8]).fun.reshape(-1, 2), axis=1)


def main():
    st = case.load_state(os.path.join(HERE, "states", "t2_all.pkl"))
    cams, rig = st["cams"], st["rig"]
    log = case.Tee(os.path.join(HERE, "w0_refraction_distortion_out.txt"))
    log("W0: refraction's distortion beyond a homography, delivered fit (T2) with k1 = 0; planes parallel to each "
        "camera's wall")
    for c, cam in enumerate(cams):
        k = cam.copy()
        k.dist = np.zeros_like(cam.dist)
        w = rig.cam_win[c]
        win = rig.windows[w]
        t1, t2 = (b[0] for b in case.pm.tangent_basis(win.n))
        o, d, _ = case.pm.rays(case.pw.cm.rows([k], np.zeros(1, int)), case.pm.wrows(rig.windows, np.array([w])),
                               np.array([[k.cx, k.cy]]))
        for depth in DEPTHS:
            X0 = o[0] + depth / float(d[0] @ win.n) * d[0]
            g = np.linspace(-HALF_MM, HALF_MM, STEPS)
            G = np.array([X0 + a * t1 + b * t2 for a in g for b in g])
            pr = case.pm.project(case.pw.cm.rows([k], np.zeros(len(G), int)),
                                 case.pm.wrows(rig.windows, np.full(len(G), w)), G)
            uv = pr.uv
            inside = (pr.ok & (uv[:, 0] >= 0) & (uv[:, 0] <= k.width - 1) & (uv[:, 1] >= 0)
                      & (uv[:, 1] <= k.height - 1))
            r = homography_residual(np.column_stack([(G - X0) @ t1, (G - X0) @ t2])[inside], uv[inside])
            log(f"  C{c + 1} ({case.angle_deg(win.n, k.R[2]):.1f} deg to its wall's normal), water depth {depth:.0f} "
                f"mm: {int(inside.sum())} points in the image; max {r.max():.3f} px, RMS {np.sqrt(np.mean(r ** 2)):.3f} px")
    log("  For comparison, the fitted k1 (0.76-1.06) moves the image corners by about 6 px.")
    log.close()


if __name__ == "__main__":
    main()
