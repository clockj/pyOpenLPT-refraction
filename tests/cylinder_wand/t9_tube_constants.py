"""T9: wrong tube constants (diagnostic).

Data from the true tube (reference rig, 2000 frames for power, 0.1 px noise).  Calibrated (from the truth's start)
with the tube's constants and the wand length perturbed one at a time: n_water +-0.002, R_outer +-0.25 mm (the wall
thickness kept), the nominal L +-0.01 mm.  Reported for each:
  the RMS change (does the misfit show in the residuals?);
  the camera shifts in units of their sigmas (largest);
  the error on unseen points against the truth, gauge-free (re-triangulated) and modulo the stabiliser;
  the L-free fit: L estimated with the camera parameters (the tube's radii also fix the scale), its sigma and its
  offset from the nominal in sigmas -- does it detect the mismatch?
Outputs: t9_tube_constants_out.txt, t9_tube_constants.json.
"""
import json
import math

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

OUT = rigs.HERE / "t9_tube_constants"
log = rigs.Tee(str(OUT) + "_out.txt")


def fit(data, tube, truth, L, fit_L=False):
    obs = cw.observations(cw.WandData(data.uv, L, data.sizes))
    layout = cw.Layout(len(truth), fit_L=fit_L)
    cams3, _ = cw.coarse_fit(truth, tube, obs, L, layout)
    return cw.refine(cams3, tube, cw.initial_frames(cams3, tube, obs, L), L, obs, layout)


def main():
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    sizes = [(c.width, c.height) for c in truth]
    tube0 = rigs.CASE_TUBE
    rng = np.random.default_rng(909)
    wd = rigs.wand_frames(truth, tube0, 2000, 10.0, rng, half=(20, 20, 20), noise_px=0.1)
    data = cw.WandData(wd.uv, 10.0, sizes)
    g = np.linspace(-20, 20, 7)
    P = np.array([[x, y, z] for x in g for y in g for z in g])
    Ro, Ri = tube0.radii
    cases = [("true constants", tube0, 10.0),
             ("n_water +0.002", cm.Tube(tube0.radii, (1.0, 1.49, 1.335)), 10.0),
             ("n_water -0.002", cm.Tube(tube0.radii, (1.0, 1.49, 1.331)), 10.0),
             ("R_outer +0.25 mm", cm.Tube((Ro + 0.25, Ri + 0.25), tube0.n), 10.0),
             ("R_outer -0.25 mm", cm.Tube((Ro - 0.25, Ri - 0.25), tube0.n), 10.0),
             ("L +0.01 mm", tube0, 10.01),
             ("L -0.01 mm", tube0, 9.99)]
    res = {"cases": []}
    base = None
    for name, tube, L in cases:
        sol = fit(data, tube, truth, L)
        solL = fit(data, tube, truth, L, fit_L=True)
        sL = math.sqrt(solL.cov[solL.layout.L_index, solL.layout.L_index])
        # unseen-point errors against the truth (the fitted cameras with the assumed tube, the truth with the true)
        fixed, stab, retri = rigs.realised_unseen(sol.cams, tube, truth, tube0, P)
        rec = {"case": name, "rms_px": sol.rms, "unseen_stabiliser_px": stab, "unseen_retriangulated_px": retri,
               "L_free_mm": solL.L, "L_free_sigma_mm": sL, "L_free_offset_sigma": (solL.L - L) / sL}
        if base is None:
            base = sol
            rec["max_shift_sigma"] = 0.0
        else:
            lay = base.layout
            dl = rigs.local_delta(sol.cams, base.cams)
            d = np.array([dl[c][j] for c in range(3) for j in range(len(lay.cols)) if lay.pos[c, j] >= 0])
            z = np.abs(d) / np.sqrt(np.diag(base.cov))
            rec["max_shift_sigma"] = float(z.max())
            rec["max_shift_param"] = lay.names[int(np.argmax(z))]
            rec["f_shift_rel"] = [float(c.f / b.f - 1) for c, b in zip(sol.cams, base.cams)]
        res["cases"].append(rec)
        log(f"{name:18s}: rms {sol.rms:.5f} px | camera shift max {rec['max_shift_sigma']:7.2f} sigma "
            f"({rec.get('max_shift_param', '-')}) | unseen: stabiliser {stab:.3f} px, re-triangulated {retri:.4f} px | "
            f"L free {solL.L:.4f} +- {sL:.4f} mm, {rec['L_free_offset_sigma']:+.1f} sigma from the nominal {L}")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
