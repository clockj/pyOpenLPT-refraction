"""Pre-plan sizing for T13 (wand balls seen through the tube): how far is the centre of a ball's image from the image
of the ball's centre?  A quick exact calculation, made to answer the user's question and to size the T13 plan; it
changes no code and is not one of T13's tests.

The silhouette of an opaque ball (radius rho, centre X0, in the water) is exactly the set of pixels whose ray, refracted
into the water, passes within rho of X0.  For each camera and ball the script finds that silhouette's boundary by
bisection along N directions from p0, the image of X0 (the PINCYL projection), and compares with p0:
  - the area centroid of the silhouette (what a perfect thresholded-blob centroid measures);
  - the centre of a conic (ellipse) fitted to the boundary;
  - the shape: horizontal / vertical extent (across / along the tube axis).
Cameras: the reference rig (three replicas of cam1, 45 deg apart, 1094 mm from the axis, f 41,459 px), the case's
tube.  Ball centres on a 5 x 5 x 5 grid over +-8 mm (the common volume cam1's sensor sees) and over +-20 mm.
Output: t13_pre_sizing_out.txt and t13_pre_sizing.json beside this script.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cyl_model as cm  # noqa: E402

OUT = rigs.HERE / "t13_pre_sizing"
log = rigs.Tee(str(OUT) + "_out.txt")
N_DIR = 1440
RHOS = (0.25, 0.5, 1.0, 1.5)          # ball radius, mm (diameters 0.5, 1, 2, 3 mm)
HALVES = (8.0, 20.0)


def ray_dist(rw1, cam, X0, uv, tube):
    """Distance from X0 to the water rays of pixels uv (n, 2)."""
    xy = np.column_stack([(uv[:, 0] - cam.cx) / cam.f, (uv[:, 1] - cam.cy) / cam.f])
    rw = rw1.take(np.zeros(len(uv), int))
    o, d, ok, _, _ = cm.trace(rw, xy, tube)
    w = X0[None, :] - o
    dist = np.linalg.norm(np.cross(w, d), axis=1)
    return np.where(ok, dist, np.inf)


def silhouette(cam, rw1, tube, X0, rho, p0, scale, n_dir=N_DIR):
    """Boundary points of the ball's silhouette, found by bisection along N_DIR directions from p0."""
    th = np.arange(n_dir) * (2 * np.pi / n_dir)
    e = np.column_stack([np.cos(th), np.sin(th)])
    lo = np.zeros(n_dir)
    hi = np.full(n_dir, 4.0 * rho * scale)
    assert np.all(ray_dist(rw1, cam, X0, p0 + hi[:, None] * e, tube) > rho), "bracket"
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        inside = ray_dist(rw1, cam, X0, p0 + mid[:, None] * e, tube) <= rho
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return 0.5 * (lo + hi)[:, None] * e          # relative to p0


def polygon_centroid(P):
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = 0.5 * cr.sum()
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6.0 * A)


def conic_centre(P):
    """Centre of the conic x^2 + B xy + C y^2 + D x + E y + F = 0 fitted to P by least squares (P relative to p0,
    scaled to O(1) first)."""
    s = np.max(np.abs(P))
    x, y = P[:, 0] / s, P[:, 1] / s
    A = np.column_stack([x * y, y * y, x, y, np.ones_like(x)])
    B, C, D, E, _ = np.linalg.lstsq(A, -x * x, rcond=None)[0]
    M = np.array([[2.0, B], [B, 2.0 * C]])
    return np.linalg.solve(M, [-D, -E]) * s


def main():
    t0 = time.time()
    tube = rigs.CASE_TUBE
    cams = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    cpp = [cm.to_cpp(c, tube) for c in cams]
    rws = [cm.rows([c], [0]) for c in cams]
    log("T13 pre-plan sizing: centre of a ball's image through the tube vs the image of the ball's centre")
    log(f"  reference rig (3 cameras at 0/45/90 deg, D {rigs.CAM1_D} mm, f {rigs.CAM1_F} px), case tube; "
        f"{N_DIR} boundary points per silhouette; px = pixels")

    # convergence check of the boundary discretisation, one case
    res = {"rhos": RHOS, "halves": HALVES, "cases": []}
    X0 = np.array([8.0, 8.0, 8.0])
    out = []
    for n in (360, 1440):
        p0 = cm.cpp_project(cpp[1], X0[None])[0]
        P = silhouette(cams[1], rws[1], tube, X0, 1.0, p0, 60.0, n_dir=n)
        out.append(polygon_centroid(P))
    log(f"  discretisation check (cam2, ball at (8, 8, 8) mm, rho 1 mm): centroid offset with 360 vs 1440 "
        f"directions differs by {np.linalg.norm(out[0] - out[1]):.1e} px")

    for half in HALVES:
        g = np.linspace(-half, half, 5)
        grid = np.array([[x, y, z] for x in g for y in g for z in g])
        log("")
        log(f"Ball centres on a 5 x 5 x 5 grid over +-{half:g} mm")
        log(f"  {'ball diam.':>10s} {'camera':>6s} {'image size h x v (px)':>22s} {'h / v':>12s} "
            f"{'centroid offset: RMS':>21s} {'max':>7s} {'max horiz.':>10s} {'max vert.':>9s} "
            f"{'conic centre: RMS':>18s} {'max':>7s}")
        for rho in RHOS:
            for c, (cam, cc, rw1) in enumerate(zip(cams, cpp, rws)):
                p0s = cm.cpp_project(cc, grid)
                offs, conics, sizes = [], [], []
                for X0, p0 in zip(grid, p0s):
                    P = silhouette(cam, rw1, tube, X0, rho, p0, 60.0)
                    offs.append(polygon_centroid(P))
                    conics.append(conic_centre(P))
                    sizes.append([P[:, 0].max() - P[:, 0].min(), P[:, 1].max() - P[:, 1].min()])
                offs, conics, sizes = np.array(offs), np.array(conics), np.array(sizes)
                ratio = sizes[:, 0] / sizes[:, 1]
                rms = float(np.sqrt(np.mean(np.sum(offs ** 2, axis=1))))
                mx = float(np.max(np.linalg.norm(offs, axis=1)))
                crms = float(np.sqrt(np.mean(np.sum(conics ** 2, axis=1))))
                cmx = float(np.max(np.linalg.norm(conics, axis=1)))
                log(f"  {2 * rho:>8.1f} mm {c + 1:>6d} {sizes[:, 0].mean():>9.1f} x {sizes[:, 1].mean():>6.1f} px "
                    f"{ratio.min():>5.3f}-{ratio.max():<5.3f} {rms:>17.4f} px {mx:>7.4f} "
                    f"{np.max(np.abs(offs[:, 0])):>10.4f} {np.max(np.abs(offs[:, 1])):>9.4f} "
                    f"{crms:>15.4f} px {cmx:>7.4f}")
                res["cases"].append({"half_mm": half, "rho_mm": rho, "camera": c + 1,
                                     "size_px_mean": sizes.mean(axis=0).tolist(),
                                     "h_over_v": [float(ratio.min()), float(ratio.max())],
                                     "centroid_offset_rms_px": rms, "centroid_offset_max_px": mx,
                                     "centroid_offset_max_h_px": float(np.max(np.abs(offs[:, 0]))),
                                     "centroid_offset_max_v_px": float(np.max(np.abs(offs[:, 1]))),
                                     "conic_centre_rms_px": crms, "conic_centre_max_px": cmx,
                                     "offsets_px": offs.tolist(), "grid_mm": grid.tolist()})
    # scaling with the ball's size: offset(rho) / rho^2 should be nearly constant if the effect is second order
    log("")
    log("Scaling with the ball's size (RMS centroid offset / rho^2, px per mm^2; constant = second order in rho):")
    for half in HALVES:
        for c in range(3):
            v = [r["centroid_offset_rms_px"] / r["rho_mm"] ** 2 for r in res["cases"]
                 if r["half_mm"] == half and r["camera"] == c + 1]
            log(f"  +-{half:g} mm, camera {c + 1}: " + ", ".join(f"{x:.4f}" for x in v)
                + f"   (rho {', '.join(f'{r:g}' for r in RHOS)} mm)")
    log("")
    log(f"done in {time.time() - t0:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    log.close()


if __name__ == "__main__":
    main()
