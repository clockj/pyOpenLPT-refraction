"""E0b: the (point, camera) pairs for which the C++ projectAll returned no image in E0: where they are, and what the
Python reference implementation (D:\\Refraction\\code\\project_cylinder.py, the paper's algorithm) and the library's
Newton (cyl_model.project from the pinhole start) return there.

E0 found a few such pairs well inside r_c, where every point has exactly one image.  This regenerates E0's points
with E0's seeds, finds every pair with no C++ image, and reports per pair: r_X, the azimuth difference dphi from the
pinhole, the axial offset dz, the C++ projectAll and projectBatchStatus results, the Python reference's image count,
and whether the library's Newton finds an image whose ray passes through the point.
Outputs: e0b_zero_images_out.txt, e0b_zero_images.json.
"""
import json
import sys

import numpy as np

import projection_fit as pf
from projection_fit import cm, rigs
import e0_census as e0

sys.path.insert(0, r"D:\Refraction\code")
import project_cylinder as pc                            # noqa: E402  the paper's reference implementation

L = e0.L


def zero_pairs(cams, tube, P):
    """(point index, camera) pairs with no C++ image."""
    lpt = cm.import_lpt()
    out = []
    for c, cam in enumerate(cams):
        cc = cm.to_cpp(cam, tube)
        for i, p in enumerate(P):
            rays, _cov = cc.projectAll(lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])))
            if not [r for r in rays if not r["is_ring"]]:
                out.append((i, c))
    return out


def describe(cam, tube, X):
    """The pair in the paper's frame (axis z): the code frame's (X, Y, Z) is the paper's (y, z, x)."""
    P0 = np.array([cam.C[2], cam.C[0], cam.C[1]])
    Xp = np.array([X[2], X[0], X[1]])
    rX, rP = float(np.hypot(Xp[0], Xp[1])), float(np.hypot(P0[0], P0[1]))
    dphi = float(np.degrees(np.angle(np.exp(1j * (np.arctan2(Xp[1], Xp[0]) - np.arctan2(P0[1], P0[0]))))))
    py = pc.project(list(tube.n), list(tube.radii), P0, Xp)
    lpt = cm.import_lpt()
    cc = cm.to_cpp(cam, tube)
    prim = cc.projectBatchStatus([lpt.Pt3D(*map(float, X))], False)[0]
    rw = cm.rows([cam], np.array([0]))
    pr = cm.project(rw, np.array([X], float), cm.pinhole_start(rw, np.array([X], float)), tube)
    # how close each reference ray is to grazing the point's own radius: 1 - |Lambda| / Lmax(C_z), with Lmax the
    # bound of the paper's domain (r_N = r_X); near 0 the near and far branches meet
    graze = [float(1.0 - abs(lam) / pc.Lmax_kstar(list(tube.n), list(tube.radii), rX, cz)[0]) for _b, lam, cz, _d in py]
    return {"r_X_mm": rX, "r_P_mm": rP, "dphi_deg": dphi, "dz_mm": float(Xp[2] - P0[2]),
            "python_images": len(py), "python_branches": [b for b, *_ in py], "python_one_minus_lam_ratio": graze,
            "cpp_primary_ok": bool(prim[0]), "cpp_primary_msg": str(prim[2]),
            "library_newton_ok": bool(pr.ok[0]), "library_uv": [float(v) for v in pr.uv[0]] if pr.ok[0] else None}


def main():
    run = pf.Run("e0b_zero_images")
    tube = rigs.CASE_TUBE
    r_max = tube.fold_radius() * 0.98
    found = []
    # the reference rig: E0's data and volume
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    ends = []
    for seed in range(20):
        wd = rigs.wand_frames(truth, tube, 500, L, np.random.default_rng(1000 + seed), half=(20, 20, 20),
                              noise_px=0.1)
        ends += [wd.XA, wd.XB]
    wd = rigs.wand_frames(truth, tube, 2000, L, np.random.default_rng(909), half=(20, 20, 20), noise_px=0.1)
    ends += [wd.XA, wd.XB]
    sets = [("reference rig, data", truth, np.concatenate(ends)),
            ("reference rig, volume", truth, e0.box_sample(np.random.default_rng(42), (0, 0, 0), (20, 20, 20),
                                                            e0.N_BOX_REF, r_max))]
    # the T5 rigs E0 flagged
    flagged = [r["rig"] for r in json.load(open(pf.HERE / "e0_census.json", encoding="utf-8"))["t5"]
               if not r["all_one"]]
    for k in flagged:
        rng = np.random.default_rng(5000 + k)
        cams, centre, half = rigs.random_rig(rng)
        wd = rigs.wand_frames(cams, tube, 500, L, rng, half=(half, half, half), centre=centre, noise_px=0.1)
        sets.append((f"T5 rig {k}, data", cams, np.concatenate([wd.XA, wd.XB])))
        sets.append((f"T5 rig {k}, volume", cams, e0.box_sample(np.random.default_rng(7000 + k), centre,
                                                                  (half, half, half), e0.N_BOX_T5, r_max)))
    for name, cams, P in sets:
        for i, c in zero_pairs(cams, tube, P):
            d = describe(cams[c], tube, P[i])
            d.update({"set": name, "camera": c + 1, "point_code_frame_mm": P[i].tolist()})
            found.append(d)
            run.log(f"{name:24s} cam {c + 1}: r_X {d['r_X_mm']:7.3f} mm, dphi {d['dphi_deg']:+9.4f} deg, dz "
                    f"{d['dz_mm']:+8.3f} mm | Python reference: {d['python_images']} image(s) {d['python_branches']}, "
                    f"1 - |Lam|/Lmax {['%.1e' % g for g in d['python_one_minus_lam_ratio']]} | "
                    f"C++ primary {'ok' if d['cpp_primary_ok'] else 'none'} ({d['cpp_primary_msg']}) | library Newton "
                    f"{'finds it' if d['library_newton_ok'] else 'fails'}")
    run.log(f"E0b: {len(found)} pairs with no C++ image; the Python reference finds "
            f"{sum(d['python_images'] >= 1 for d in found)} of them, the library's Newton "
            f"{sum(d['library_newton_ok'] for d in found)}")
    run.finish({"pairs": found})


if __name__ == "__main__":
    main()
