"""T6: gauge and frame.

(a) Rank.  At the solution of a noisy fit (reference rig), the reduced (Schur) camera system with the gauge, scaled
    to unit diagonal, has full rank (19), and without the gauge (camera 1's C1 and C2 free, 21 parameters) exactly two near-null
    directions, which are the tube's stabiliser acting on the cameras: a spin about the axis
    (w_c = -alpha e_y, dC_c = alpha e_y x C_c) and a slide along it (dC_c = beta e_y).
(b) Camera order.  The same data calibrated with the cameras in another order (camera 1 is then a different
    physical camera, so the gauge differs) gives the same rig once the second answer is put in the first one's gauge
    (a turn about the axis and a slide along it).
(c) Input frame.  The same physical rig generated in a world frame where the tube is tilted 20 deg and offset
    (C++ cameras with a general axis) gives the same pixels to round-off, hence the same calibration in the tube
    frame.
Criteria (plan section 7): rank = number of parameters, and without the gauge two singular values at round-off
(below 1e-12 of the largest, 1000x below the next) whose singular vectors are the stabiliser (principal angles below
1e-6 rad); camera order: the same output to 1e-4 of its sigmas; input frame: pixels to 1e-6 px, output to 1e-6 mm
and 1e-9 in f.
Outputs: t6_gauge_frame_out.txt, t6_gauge_frame.json.
"""
import json
import math

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

OUT = rigs.HERE / "t6_gauge_frame"
log = rigs.Tee(str(OUT) + "_out.txt")


def fit(cams_start, tube, uv, sizes):
    data = cw.WandData(uv, 10.0, sizes)
    obs = cw.observations(data)
    layout = cw.Layout(len(cams_start))
    cams3, _ = cw.coarse_fit(cams_start, tube, obs, 10.0, layout)
    sol = cw.refine(cams3, tube, cw.initial_frames(cams3, tube, obs, 10.0), 10.0, obs, layout, k_sigma=1e9)
    return sol


def main():
    tube = rigs.CASE_TUBE
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    sizes = [(c.width, c.height) for c in truth]
    rng = np.random.default_rng(606)
    wd = rigs.wand_frames(truth, tube, 400, 10.0, rng, half=(20, 20, 20), noise_px=0.1)
    start = rigs.perturb_rig(truth, rng, rot_deg=1.0, trans_mm=10.0, f_rel=0.02)
    sol = fit(start, tube, wd.uv, sizes)
    res = {}
    # ---- (a) rank and null space
    obs = sol.obs
    S_g, _ = cw.information(sol.cams, tube, sol.frames, sol.L, obs, sol.layout)
    lay_free = cw.Layout(3, gauge=False)
    S_f, _ = cw.information(sol.cams, tube, sol.frames, sol.L, obs, lay_free)
    # the columns mix radians, millimetres and log f, so rank and null space are read off the matrices scaled
    # to unit diagonal (Jacobi); the generators are mapped to the same scaled coordinates
    dg = np.sqrt(np.diag(S_g))
    sg = np.linalg.svd(S_g / np.outer(dg, dg), compute_uv=False)
    df_ = np.sqrt(np.diag(S_f))
    U, sf, Vt = np.linalg.svd(S_f / np.outer(df_, df_))
    # stabiliser generators in the free layout's coordinates
    ey = np.array([0.0, 1.0, 0.0])
    gen = np.zeros((lay_free.P, 2))
    for c, cam in enumerate(sol.cams):
        for j, nm in enumerate(lay_free.cols):
            g = lay_free.pos[c, j]
            if nm in ("w1", "w2", "w3"):
                gen[g, 0] = -ey[j]
            elif nm in ("C1", "C2", "C3"):
                gen[g, 0] = np.cross(ey, cam.C)[j - 3]
                gen[g, 1] = ey[j - 3]
    null = Vt[-2:].T
    q1, _ = np.linalg.qr(gen * df_[:, None])
    cosines = np.linalg.svd(q1.T @ null, compute_uv=False)
    angles = np.arccos(np.clip(cosines, -1, 1))
    res["rank"] = {"gauge_params": int(sol.layout.P), "gauge_rank": int(np.sum(sg > sg[0] * 1e-12)),
                   "gauge_cond": float(sg[0] / sg[-1]), "free_params": int(lay_free.P),
                   "free_smallest_sv_rel": (sf[-3:] / sf[0]).tolist(),
                   "null_vs_stabiliser_principal_angles_rad": angles.tolist()}
    log(f"(a) with the gauge (scaled to unit diagonal): {sol.layout.P} parameters, rank "
        f"{res['rank']['gauge_rank']}, condition {res['rank']['gauge_cond']:.2e}")
    log(f"    without it: {lay_free.P} parameters, smallest singular values / largest "
        f"{', '.join(f'{v:.1e}' for v in sf[-3:] / sf[0])}; null space vs the stabiliser: principal angles "
        f"{', '.join(f'{a:.1e}' for a in angles)} rad")
    ok_a = (res["rank"]["gauge_rank"] == sol.layout.P and sf[-2] / sf[0] < 1e-12 and sf[-3] > 1e3 * sf[-2]
            and np.max(angles) < 1e-6)
    # ---- (b) camera order
    perm = [1, 2, 0]
    uv_p = wd.uv[:, perm]
    start_p = rigs.gauge_fix([start[k] for k in perm])
    sol_p = fit(start_p, tube, uv_p, [sizes[k] for k in perm])
    back = [None] * 3
    for i, k in enumerate(perm):
        back[k] = sol_p.cams[i]
    back = rigs.gauge_fix(back)                         # physical camera 0 at X = Y = 0 again
    cmp_b = rigs.compare(back, sol.cams)
    # the same in sigma units: the difference in the fit's local coordinates over the reported sigmas
    lay = sol.layout
    dl = rigs.local_delta(back, sol.cams)
    dvec = np.array([dl[c][j] for c in range(3) for j in range(len(lay.cols)) if lay.pos[c, j] >= 0])
    zmax = float(np.max(np.abs(dvec) / np.sqrt(np.diag(sol.cov))))
    res["order"] = {"cmp": cmp_b, "rms_a": sol.rms, "rms_b": sol_p.rms, "max_diff_in_sigma": zmax}
    log(f"(b) cameras reordered {perm}: after re-gauging, max differences {max(e['rot_deg'] for e in cmp_b):.1e} deg, "
        f"{max(e['centre_mm'] for e in cmp_b):.1e} mm, f {max(abs(e['f_rel']) for e in cmp_b):.1e}, at most "
        f"{zmax:.1e} sigma; rms {sol.rms:.6f} vs {sol_p.rms:.6f} px")
    ok_b = zmax < 1e-4
    # ---- (c) input frame: the rig in a tilted, offset world frame
    a = np.array([0.6, 0.0, 0.8])
    Qw = cm.exp_so3(a * math.radians(20.0))              # tube -> world
    p = np.array([12.0, -30.0, 25.0])
    XAw, XBw = wd.XA @ Qw.T + p, wd.XB @ Qw.T + p
    uv_w = np.full_like(wd.uv, np.nan)
    noise = wd.uv - np.stack([np.stack([cm.cpp_project(cm.to_cpp(c, tube), wd.XA),
                                        cm.cpp_project(cm.to_cpp(c, tube), wd.XB)], axis=1)
                              for c in truth], axis=1)
    for c, cam in enumerate(truth):
        cpp = cm.to_cpp(cam, tube, axis_pt=p, axis_dir=Qw @ np.array([0.0, 1.0, 0.0]), R_world=cam.R @ Qw.T,
                        C_world=Qw @ cam.C + p)
        uv_w[:, c, 0] = cm.cpp_project(cpp, XAw)
        uv_w[:, c, 1] = cm.cpp_project(cpp, XBw)
    uv_w = uv_w + noise                                   # the same noise
    dpx = np.nanmax(np.abs(uv_w - wd.uv))
    sol_w = fit(start, tube, uv_w, sizes)
    cmp_c = rigs.compare(sol_w.cams, sol.cams)
    res["frame"] = {"max_pixel_difference": float(dpx), "cmp": cmp_c}
    log(f"(c) tube tilted 20 deg and offset in the world: pixels differ by at most {dpx:.1e} px; calibrations differ "
        f"by {max(e['rot_deg'] for e in cmp_c):.1e} deg, {max(e['centre_mm'] for e in cmp_c):.1e} mm, f "
        f"{max(abs(e['f_rel']) for e in cmp_c):.1e}")
    ok_c = dpx < 1e-6 and max(e["centre_mm"] for e in cmp_c) < 1e-6 and max(abs(e["f_rel"]) for e in cmp_c) < 1e-9
    res["pass"] = {"rank_and_null_space": bool(ok_a), "camera_order": bool(ok_b), "input_frame": bool(ok_c)}
    for k, v in res["pass"].items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
