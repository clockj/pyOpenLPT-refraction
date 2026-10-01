"""D1 (diagnosis): the structure of the reprojection residuals of a fit.

    python tests/d1_residual_structure.py STATE.pkl [tag]

Breaks the residuals (detected minus model) down to find what a model misses:
  * per plane: mean (a uniform image shift), and a 2-D affine part (scale / shear / rotation of each plane's dot
    pattern), the part a mis-set plane position or tilt would leave;
  * against the image position, pooled over planes: a radial (distortion-like) term about the sensor centre, and
    low-order polynomial trends in u and v;
  * against the dot index, pooled over planes (a board error would follow the dots, not the image);
  * how much of the RMS each of these explains.
Writes d1_<tag>_out.txt beside this script.
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


def rms(r):
    return float(np.sqrt(np.mean(np.sum(np.asarray(r).reshape(-1, 2) ** 2, axis=1))))


def explained(r, G):
    """Least-squares fit of the residual components r (n,2) on the design G (n,p): (fitted, remainder)."""
    coef, *_ = np.linalg.lstsq(G, r, rcond=None)
    fit = G @ coef
    return fit, r - fit, coef


def main():
    path = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "fit"
    st = RC.load_state(path)
    obs, r, rig = st["obs"], st["resid"], st["rig"]
    say(f"D1 residual structure of {path} (model {st['info']['model']}, {len(r)} dots): rms {rms(r):.4f} px "
        f"(u {np.sqrt(np.mean(r[:, 0] ** 2)):.4f}, v {np.sqrt(np.mean(r[:, 1] ** 2)):.4f})")
    s = rig.summary()
    say(f"  f {s['f_px']:.1f} px; camera {np.round(s['camera_centre_mm'], 2)} mm; camera-axis distance "
        f"{s['camera_distance_from_axis_mm']:.1f} mm; axis at Y=0 {np.round(s['axis_point_at_Y0_mm'], 3)}; tilt "
        f"{s['axis_tilt_in_board_plane_deg']:.3f} lean {s['axis_lean_towards_camera_deg']:.3f} deg")
    u, v = obs.uv[:, 0], obs.uv[:, 1]
    K = obs.n_planes

    # per plane: mean and affine part
    say("\nper plane: n, rms, mean (du, dv), rms left after removing the plane's own affine map of (u, v)")
    left_aff = np.zeros_like(r)
    for k in range(K):
        m = obs.plane == k
        G = np.column_stack([np.ones(m.sum()), u[m] - 640, v[m] - 400])
        _f, rem, coef = explained(r[m], G)
        left_aff[m] = rem
        say(f"  z={obs.z_nom[k]:4.0f} dz {rig.dz[k]:+.3f}: n {m.sum():3d} rms {rms(r[m]):.3f} mean "
            f"({r[m, 0].mean():+.3f}, {r[m, 1].mean():+.3f})  affine du/du {coef[1, 0] * 1e3:+.3f} du/dv "
            f"{coef[2, 0] * 1e3:+.3f} dv/du {coef[1, 1] * 1e3:+.3f} dv/dv {coef[2, 1] * 1e3:+.3f} (x1e-3)  "
            f"left {rms(rem):.3f}")
    say(f"  -> removing each plane's mean leaves "
        f"{rms(np.vstack([r[obs.plane == k] - r[obs.plane == k].mean(0) for k in range(K)])):.4f} px; "
        f"its affine map leaves {rms(left_aff):.4f} px")

    # pooled image-position trends
    x, y = (u - 639.5) / 640.0, (v - 399.5) / 640.0
    rr2 = x * x + y * y
    G_rad = np.column_stack([x * rr2, y * rr2])          # radial cubic about the centre (k1-like), two columns
    Grad = np.column_stack([np.ones_like(x), x, y])
    _f, rem1, _ = explained(r, Grad)
    rad = np.column_stack([x * rr2, y * rr2])
    # a k1 term moves (du, dv) = c (x r^2, y r^2): one coefficient for both components
    rhs = np.concatenate([rem1[:, 0], rem1[:, 1]])
    A = np.concatenate([rad[:, 0], rad[:, 1]])[:, None]
    c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    rem_rad = rem1 - c[0] * rad
    say(f"\npooled over planes (image coordinates normalised by 640 px):")
    say(f"  a global affine trend leaves {rms(rem1):.4f} px; adding one radial cubic term (k1-like, coefficient "
        f"{c[0]:+.3f} px at r = 640 px) leaves {rms(rem_rad):.4f} px")
    for deg in (2, 3):
        cols = [x ** a * y ** b for a in range(deg + 1) for b in range(deg + 1 - a)]
        Gp = np.column_stack(cols)
        _f, remp, _ = explained(r, Gp)
        say(f"  a full polynomial of degree {deg} in (u, v), the same for every plane, leaves {rms(remp):.4f} px")
    # per-plane polynomial (would absorb anything smooth within a plane)
    rem_pp = np.zeros_like(r)
    for k in range(K):
        m = obs.plane == k
        cols = [x[m] ** a * y[m] ** b for a in range(3) for b in range(3 - a)]
        _f, rem_pp[m], _ = explained(r[m], np.column_stack(cols))
    say(f"  a quadratic polynomial per plane leaves {rms(rem_pp):.4f} px (the floor for smooth per-plane errors)")

    # by dot index, pooled over planes (board errors follow the dots)
    keys = sorted(set(zip(obs.i.astype(int), obs.j.astype(int))))
    mean_by_dot = {}
    for key in keys:
        m = (obs.i == key[0]) & (obs.j == key[1])
        mean_by_dot[key] = r[m].mean(axis=0)
    rem_dot = r - np.array([mean_by_dot[(int(a), int(b))] for a, b in zip(obs.i, obs.j)])
    say(f"\nremoving each dot's mean over the planes (a board error) leaves {rms(rem_dot):.4f} px")
    say("  dot mean residuals (i, j): du, dv")
    for jj in sorted({k[1] for k in keys}, reverse=True):
        say("   j=%+d " % jj + "  ".join(f"({mean_by_dot[(ii, jj)][0]:+.2f},{mean_by_dot[(ii, jj)][1]:+.2f})"
                                          if (ii, jj) in mean_by_dot else "      --      "
                                          for ii in range(0, 9)))
    # by image cell, pooled (an image-fixed error such as distortion stays put while dots move)
    say("\nresidual by image cell (u in 160-px bins, v in 160-px bins), pooled over planes: mean du, dv (n)")
    for vb in range(0, 800, 160):
        row = []
        for ub in range(0, 1280, 160):
            m = (u >= ub) & (u < ub + 160) & (v >= vb) & (v < vb + 160)
            row.append(f"({r[m, 0].mean():+.2f},{r[m, 1].mean():+.2f})" if m.sum() >= 5 else "     --     ")
        say(f"  v {vb:3d}-{vb + 160:3d}: " + " ".join(row))
    with open(os.path.join(HERE, f"d1_{tag}_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
