"""D4 (asked 4 Oct: "do they still look like circle or ellipse? that might bring into some error which cause the
0.27 px"): how far from a circle is a wand ball's image, and how far does its outline's centre lie from the image of the
ball's centre, which is what the calibration assumes the circle finder returns?

With the delivered calibration (tests/states/w4c_ref.pkl: the principal point at the image centre, the channel held), a
ball of radius 2 mm (ball B; A is 1.5 mm, and the effects grow as the radius squared) is placed in the water on the ray
through 9 image points of each camera (the centre, the mid-edges and the corners, 100 px in), at depths of 10, 50 and
120 mm below the camera's wall: across the wand's volume.  20,000 points of its surface are projected through the plate
model (perspective, refraction, k1).  Their convex hull is the ball's image, its outline the silhouette.  Measured:
  * the image's radius and its axis ratio (major / minor, from the filled outline's second moments of area);
  * the offset of the outline's centre (a least-squares circle through the outline, as a circle finder fits) and of its
    area centroid (as a blob-centroid finder gives) from the projection of the ball's centre.

    python tests/d4_ball_image_shape.py
Writes d4_ball_image_shape_out.txt beside this script.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

R_MM = 2.0
DEPTHS_MM = (10.0, 50.0, 120.0)
N_SURF = 20000
INSET = 100.0


def sphere_points(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def polygon_moments(P: np.ndarray):
    """Area, centroid and the second central moments (xx, yy, xy) of the polygon P (m, 2), counter-clockwise."""
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    A = c.sum() / 2
    cx = ((x + x1) * c).sum() / (6 * A)
    cy = ((y + y1) * c).sum() / (6 * A)
    ixx = ((x * x + x * x1 + x1 * x1) * c).sum() / 12 - A * cx * cx
    iyy = ((y * y + y * y1 + y1 * y1) * c).sum() / 12 - A * cy * cy
    ixy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * c).sum() / 24 - A * cx * cy
    return A, np.array([cx, cy]), np.array([[ixx, ixy], [ixy, iyy]])


def circle_fit(P: np.ndarray) -> np.ndarray:
    """Centre of the least-squares circle (algebraic) through points P (m, 2)."""
    M = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    b = -(P[:, 0] ** 2 + P[:, 1] ** 2)
    D, E, _F = np.linalg.lstsq(M, b, rcond=None)[0]
    return np.array([-D / 2, -E / 2])


def resample(P: np.ndarray, n: int = 720) -> np.ndarray:
    """n points evenly spaced along the closed polygon P."""
    Q = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])])


def main():
    st = case.load_state(os.path.join(HERE, "states", "w4c_ref.pkl"))
    cams, rig = st["cams"], st["rig"]
    S = sphere_points(N_SURF) * R_MM
    log = case.Tee(os.path.join(HERE, "d4_ball_image_shape_out.txt"))
    log(f"D4: a {R_MM:g} mm ball's image through the delivered calibration (w4c_ref), at 9 image points per camera "
        f"(100 px in from the edges) and depths {', '.join(f'{d:g}' for d in DEPTHS_MM)} mm below the camera's wall")
    log("  camera  point          depth   radius px   axis ratio   circle-fit centre offset px   area-centroid offset px")
    summary = []
    for c, cam in enumerate(cams):
        w = rig.cam_win[c]
        win = rig.windows[w]
        xs = [INSET, (cam.width - 1) / 2, cam.width - 1 - INSET]
        ys = [INSET, (cam.height - 1) / 2, cam.height - 1 - INSET]
        names = {0: "left", 1: "mid", 2: "right"}, {0: "top", 1: "mid", 2: "bottom"}
        worst = {"ratio": 1.0, "circle": 0.0, "centroid": 0.0}
        for j, v in enumerate(ys):
            for i, u in enumerate(xs):
                o, d, ok = case.pm.rays(case.pw.cm.rows([cam], np.zeros(1, int)),
                                        case.pm.wrows(rig.windows, np.array([w])), np.array([[u, v]]))
                if not ok[0]:
                    continue
                for depth in DEPTHS_MM:
                    X0 = o[0] + depth / float(win.n @ d[0]) * d[0]
                    X = X0 + S
                    idx = np.zeros(len(X) + 1, int)
                    pr = case.pm.project(case.pw.cm.rows([cam], idx), case.pm.wrows(rig.windows, np.full(len(idx), w)),
                                         np.vstack([X0, X]))
                    if not pr.ok.all():
                        continue
                    c0, uv = pr.uv[0], pr.uv[1:]
                    hull = ConvexHull(uv)
                    poly = uv[hull.vertices]                      # counter-clockwise
                    A, cen, I = polygon_moments(poly)
                    ev = np.linalg.eigvalsh(I)
                    ratio = float(np.sqrt(ev[1] / ev[0]))
                    radius = float(np.sqrt(A / np.pi))
                    dc = float(np.linalg.norm(circle_fit(resample(poly)) - c0))
                    dg = float(np.linalg.norm(cen - c0))
                    worst = {"ratio": max(worst["ratio"], ratio), "circle": max(worst["circle"], dc),
                             "centroid": max(worst["centroid"], dg)}
                    log(f"  C{c + 1}      {names[1][j]:6s} {names[0][i]:6s} {depth:5.0f}   {radius:8.2f}    {ratio:.5f}"
                        f"       {dc:.4f}                        {dg:.4f}")
        summary.append(worst)
    log("  the largest per camera: axis ratio, circle-fit centre offset, area-centroid offset")
    for c, wst in enumerate(summary):
        log(f"    C{c + 1}: {wst['ratio']:.5f} (the image {100 * (wst['ratio'] - 1):.2f} % longer one way), "
            f"{wst['circle']:.4f} px, {wst['centroid']:.4f} px")
    log.close()


if __name__ == "__main__":
    main()
