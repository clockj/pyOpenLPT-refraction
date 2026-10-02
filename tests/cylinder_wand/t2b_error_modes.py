"""T2b: which 3-D motion carries the reference rig's projection error on unseen points (diagnostic for T2).

The Cramer-Rao covariance of the reference rig (T2) is mapped to the pixel errors of projecting a 7x7x7 grid of
fixed tube-frame points with every camera.  The RMS is reported after removing, cumulatively, the best common
motion of the points from these families (each is a linear map of the points' pixel errors through d(uv)/dX):
  stabiliser   spin about the axis, slide along it (not observable: a relabelling)
  + shift      translation across the axis (X, Z): where the tube sits relative to the rig
  + tilt       rotation about X and Z: the tube's axis direction relative to the rig
  + scale      a uniform scale about the origin
  + affine     a general affine map of the points (12 dof)
and, as the gauge-free floor, each point re-triangulated (T2's `retriangulated`).
Outputs: t2b_error_modes_out.txt.
"""
import math

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw
import t2_observability as t2

log = rigs.Tee(str(rigs.HERE / "t2b_error_modes") + "_out.txt")


def run(name, **kw):
    cams, tube, fr, obs, wd = t2.build(**kw)
    layout = cw.Layout(len(cams))
    info, ev = cw.information(cams, tube, fr, t2.L_WAND, obs, layout, sigma_px=t2.SIGMA)
    cov = np.linalg.inv(info)
    half = kw["half"]
    g = np.linspace(-half, half, 7)
    P = np.array([[x, y, z] for x in g for y in g for z in g])
    pj = cw.point_jacobians(cams, tube, layout, P)
    ok = np.all([p[3] for p in pj], axis=0)
    ex, ey, ez = np.eye(3)
    J, fam = [], {k: [] for k in ("stab", "shift", "tilt", "scale", "affine")}
    for i in np.flatnonzero(ok):
        X = P[i]
        Jc = np.vstack([pj[c][0][i] for c in range(len(cams))])
        JX = np.vstack([pj[c][1][i] for c in range(len(cams))])
        J.append(Jc)
        fam["stab"].append(JX @ np.column_stack([np.cross(ey, X), ey]))
        fam["shift"].append(JX @ np.column_stack([ex, ez]))
        fam["tilt"].append(JX @ np.column_stack([np.cross(ex, X), np.cross(ez, X)]))
        fam["scale"].append(JX @ X[:, None])
        fam["affine"].append(JX @ np.hstack([np.kron(np.eye(3), X[None, :]).T.reshape(3, 9) if False else
                                             np.column_stack([X[0] * np.eye(3), X[1] * np.eye(3), X[2] * np.eye(3)]),
                                             np.eye(3)]))
    J = np.vstack(J)
    Lc = np.linalg.cholesky(cov)
    n_img = int(ok.sum()) * len(cams)
    A = J @ Lc
    out = {"none (camera-1 gauge)": math.sqrt(float(np.sum(A * A)) / n_img)}
    G = np.zeros((len(J), 0))
    for k in ("stab", "shift", "tilt", "scale", "affine"):
        G = np.hstack([G, np.vstack(fam[k])])
        Ap = A - G @ np.linalg.lstsq(G, A, rcond=None)[0]
        out["+" + k] = math.sqrt(float(np.sum(Ap * Ap)) / n_img)
    pe = cw.predicted_projection_error(cams, tube, layout, cov, P)
    out["re-triangulated (gauge-free floor)"] = pe["retriangulated_rms_px"]
    log(f"{name}: " + " | ".join(f"{k} {v:.4f}" for k, v in out.items()))
    # sigma of the tube's position across the axis and of its tilt relative to the rig: the coefficients of the
    # best common shift and tilt (with the stabiliser) explaining the error field, propagated from the covariance
    G = np.hstack([np.vstack(fam["stab"]), np.vstack(fam["shift"]), np.vstack(fam["tilt"])])
    C = np.linalg.lstsq(G, A, rcond=None)[0]                 # (6, P) coefficients per unit whitened error
    cc = C @ C.T
    sd = np.sqrt(np.diag(cc))
    out["sigma_tube_shift_X_mm"], out["sigma_tube_shift_Z_mm"] = float(sd[2]), float(sd[3])
    out["sigma_tube_tilt_about_X_deg"] = float(np.degrees(sd[4]))
    out["sigma_tube_tilt_about_Z_deg"] = float(np.degrees(sd[5]))
    log(f"    tube relative to the rig: shift across the axis sigma X {sd[2] * 1e3:.1f} um, Z {sd[3] * 1e3:.1f} um; "
        f"tilt sigma {np.degrees(sd[4]) * 3600:.1f} arcsec about X, {np.degrees(sd[5]) * 3600:.1f} arcsec about Z")
    return out


if __name__ == "__main__":
    log("RMS pixel error on unseen points (Cramer-Rao, 0.1 px), after removing the best common motion of the points:")
    run("reference (3 cams 45 deg, 1094 mm, +-20 mm, 500 frames)", n_cams=3, spacing=45.0, D=rigs.CAM1_D, half=20.0,
        frames=500)
    run("4 cams at 90", n_cams=4, spacing=90.0, D=rigs.CAM1_D, half=20.0, frames=500)
    run("D = 500 mm", n_cams=3, spacing=45.0, D=500.0, half=20.0, frames=500)
    run("half = 40 mm", n_cams=3, spacing=45.0, D=rigs.CAM1_D, half=40.0, frames=500)
    run("2000 frames", n_cams=3, spacing=45.0, D=rigs.CAM1_D, half=20.0, frames=2000)
    log.close()
