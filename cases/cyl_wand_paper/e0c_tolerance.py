"""E0c: why the C++ cameras of E0 returned no image for 27 point-camera pairs, and E0's census at the projection's
own tolerance.

(a) The 27 pairs of E0b, each through the C++ core (`refract_curved.project_cylinder`) at tol 1e-9 (its default, and
    the paper's) and 1e-12 (what cyl_model.to_cpp sets, and so E0's cameras and the campaign's data generator), and
    through a camera built by cyl_model.to_cpp at each tolerance.  For the ray the reference returns: its closeness to
    grazing, delta = 1 - |Lambda|/Lmax, the residual that step (4) reaches (e0b_trace.py showed that the C++ and the
    Python reach the same one, step by step), and eps/sqrt(delta), the size of the rounding error
    in Phi next to the square-root edge at grazing.
(b) E0's census again, the same points, with the cameras at tol 1e-9.
Criterion of (b): every count is 1.
Outputs: e0c_tolerance_out.txt, e0c_tolerance.json.
"""
import json
import math
import sys

import numpy as np

import projection_fit as pf
from projection_fit import cm, rigs
import e0_census as e0

sys.path.insert(0, r"D:\Refraction\code")
import project_cylinder as pc                            # noqa: E402

L = e0.L
TOLS = (1e-9, 1e-12)


def census(cams, tube, P, tol):
    lpt = cm.import_lpt()
    counts = np.zeros((len(P), len(cams)), int)
    for c, cam in enumerate(cams):
        cc = cm.to_cpp(cam, tube, tol=tol)
        for i, p in enumerate(P):
            rays, _cov = cc.projectAll(lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])))
            counts[i, c] = len([r for r in rays if not r["is_ring"]])
    return counts


def pairs_part(run):
    """(a): the 27 pairs at both tolerances."""
    lpt = cm.import_lpt()
    rc = lpt.refract_curved
    tube = rigs.CASE_TUBE
    N, R = list(tube.n), list(tube.radii)
    out = []
    for p in json.load(open(pf.HERE / "e0b_zero_images.json", encoding="utf-8"))["pairs"]:
        rP, rX, dphi, dz = p["r_P_mm"], p["r_X_mm"], abs(math.radians(p["dphi_deg"])), p["dz_mm"]
        P0, X = [rP, 0.0, 0.0], [rX * math.cos(dphi), rX * math.sin(dphi), dz]
        core = {tol: len(rc.project_cylinder(N, R, P0, X, tol)[0]) for tol in TOLS}
        # the reference ray, its closeness to grazing, and the residual step (4) reaches (scaled lengths)
        u = math.ldexp(1.0, math.frexp(rP)[1] - 1)
        Rs, rPs, rXs, dzs = [r / u for r in R], rP / u, rX / u, dz / u
        b, lam, cz, _d = pc.project(N, R, np.array(P0), np.array(X))[0]
        delta = 1.0 - abs(lam / u) / pc.Lmax_kstar(N, Rs, rXs, cz)[0]
        if b == "far":
            _sol, _it, res = pc.solve_far(N, Rs, rPs, rXs, dphi, dzs, abs(lam) / u, cz)
        else:
            _sol, _it, res = pc.solve(N, Rs, rPs, rXs, dphi, dzs, abs(lam) / u, cz)
        rec = {"set": p["set"], "camera": p["camera"], "r_X_mm": rX, "dphi_deg": p["dphi_deg"], "branch": b,
               "delta": delta, "residual": float(res), "eps_over_sqrt_delta": 2.220446049250313e-16 / math.sqrt(delta),
               "core_rays_tol_1e-9": core[1e-9], "core_rays_tol_1e-12": core[1e-12]}
        out.append(rec)
        run.log(f"{p['set']:24s} cam {p['camera']}: {b:4s} ray, delta {delta:.1e}, residual {float(res):.1e} "
                f"(eps/sqrt(delta) {rec['eps_over_sqrt_delta']:.1e}) | C++ core rays: tol 1e-9 -> {core[1e-9]}, "
                f"tol 1e-12 -> {core[1e-12]}")
    return out


def main():
    run = pf.Run("e0c_tolerance")
    tube = rigs.CASE_TUBE
    r_max = tube.fold_radius() * 0.98
    pairs = pairs_part(run)
    above = [q for q in pairs if q["residual"] > 1e-12]
    run.log(f"(a) {len(pairs)} pairs: the C++ core finds the ray in {sum(q['core_rays_tol_1e-9'] == 1 for q in pairs)} "
            f"at tol 1e-9 and in {sum(q['core_rays_tol_1e-12'] == 1 for q in pairs)} at 1e-12; the residual step (4) "
            f"reaches is above 1e-12 in {len(above)} (range {min(q['residual'] for q in pairs):.1e} to "
            f"{max(q['residual'] for q in pairs):.1e}), against eps/sqrt(delta) "
            f"{min(q['eps_over_sqrt_delta'] for q in pairs):.1e} to {max(q['eps_over_sqrt_delta'] for q in pairs):.1e}")

    # (b) E0's census at tol 1e-9, the same points
    totals = {}
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
    for k in range(50):
        rng = np.random.default_rng(5000 + k)
        cams, centre, half = rigs.random_rig(rng)
        wd = rigs.wand_frames(cams, tube, 500, L, rng, half=(half, half, half), centre=centre, noise_px=0.1)
        sets.append((f"T5 rig {k}, data", cams, np.concatenate([wd.XA, wd.XB])))
        sets.append((f"T5 rig {k}, volume", cams, e0.box_sample(np.random.default_rng(7000 + k), centre,
                                                                  (half, half, half), e0.N_BOX_T5, r_max)))
    per_set = []
    for name, cams, P in sets:
        cnt = census(cams, tube, P, 1e-9)
        s = e0.summary(cnt)
        per_set.append({"set": name, "counts": s})
        for kk, v in s.items():
            totals[kk] = totals.get(kk, 0) + v
    ok = set(totals) == {1}
    run.log(f"(b) E0's census with the cameras at tol 1e-9: {sum(totals.values())} point-camera pairs, image counts "
            f"{dict(sorted(totals.items()))} -> {'PASS: every count is 1' if ok else 'FAIL'}")
    run.finish({"pairs": pairs, "census_tol_1e-9": {"totals": totals, "per_set": per_set}, "pass": ok})


if __name__ == "__main__":
    main()
