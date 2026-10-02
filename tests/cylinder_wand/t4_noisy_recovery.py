"""T4: noisy recovery over 20 seeds -- is the reported covariance the actual scatter?

Reference rig (three cam1 replicas at 0/45/90 deg), 500 frames of a 10 mm wand in +-20 mm, 0.1 px Gaussian noise,
a new wand path and noise per seed.  Each fit starts 2 deg / 20 mm / 5 % in f off and runs stages 3 and 4.  For every
free camera parameter (19) the error in the fit's local coordinates,
    rotation w with R_est = R_true exp([w]x), centre C_est - C_true, log(f_est / f_true),
is divided by its Cramer-Rao sigma (0.1 px, computed at the truth with that seed's frames): a z-score.  Also the
whitened vector L^-1 delta (cov = L L^T), whose 19 entries are independent N(0, 1) if the covariance is right, and
the covariance the fit itself reports (sigma estimated from the residuals) against the true one.

Criteria (plan section 7): z-scores pooled over all parameters and seeds have mean within +-0.25 and standard
deviation 0.8-1.2.
The fits are saved (t4_fits.json) for T7.
Outputs: t4_noisy_recovery_out.txt, t4_noisy_recovery.json, t4_fits.json.
"""
import json
import math
import time

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

OUT = rigs.HERE / "t4_noisy_recovery"
log = rigs.Tee(str(OUT) + "_out.txt")
SIGMA = 0.1
N_SEEDS = 20


def cam_dict(c):
    return {"R": c.R.tolist(), "C": c.C.tolist(), "f": c.f, "cx": c.cx, "cy": c.cy, "width": c.width,
            "height": c.height, "dist": c.dist.tolist()}


def main():
    t0 = time.time()
    tube = rigs.CASE_TUBE
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    layout = cw.Layout(3)
    zs, white, ratios, fits = [], [], [], []
    per_param = {nm: [] for nm in layout.names}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        wd = rigs.wand_frames(truth, tube, 500, 10.0, rng, half=(20, 20, 20), noise_px=SIGMA)
        data = cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in truth])
        obs = cw.observations(data)
        fr_true = cw.frames_from_ends(wd.XA[obs.frame_ids], wd.XB[obs.frame_ids])
        info, _ev = cw.information(truth, tube, fr_true, 10.0, obs, layout, sigma_px=SIGMA)
        cov = np.linalg.inv(info)
        start = rigs.perturb_rig(truth, rng, rot_deg=2.0, trans_mm=20.0, f_rel=0.05)
        cams3, info3 = cw.coarse_fit(start, tube, obs, 10.0, layout)
        sol = cw.refine(cams3, tube, cw.initial_frames(cams3, tube, obs, 10.0), 10.0, obs, layout)
        dl = rigs.local_delta(sol.cams, truth)
        delta = np.zeros(layout.P)
        for c in range(3):
            for j, nm in enumerate(layout.cols):
                g = layout.pos[c, j]
                if g >= 0:
                    delta[g] = dl[c][j]
        sd = np.sqrt(np.diag(cov))
        z = delta / sd
        Lc = np.linalg.cholesky(cov)
        wz = np.linalg.solve(Lc, delta)
        zs.append(z)
        white.append(wz)
        for g, nm in enumerate(layout.names):
            per_param[nm].append(float(z[g]))
        ratios.append(np.sqrt(np.diag(sol.cov)) / sd)
        fits.append({"seed": seed, "fit": [cam_dict(c) for c in sol.cams], "rms_px": sol.rms,
                     "sigma_px_est": sol.sigma_px, "active": int(obs.active.sum()), "n_obs": int(len(obs)),
                     "cov_crb": cov.tolist(), "cov_fit": sol.cov.tolist()})
        log(f"seed {seed:2d}: rms {sol.rms:.4f} px (sigma est {sol.sigma_px:.4f}), active {obs.active.sum()}/"
            f"{len(obs)}, LM {len(sol.history)} | |z| max {np.max(np.abs(z)):.2f}, chi2 {float(wz @ wz):6.1f} "
            f"(dof {layout.P}) | f err {[f'{100 * (c.f / t.f - 1):+.3f}%' for c, t in zip(sol.cams, truth)]} | "
            f"{time.time() - t0:.0f} s")
    Z = np.concatenate(zs)
    Wz = np.concatenate(white)
    chi2 = np.array([float(w @ w) for w in white])
    ratio = np.concatenate(ratios)
    res = {"seeds": N_SEEDS, "params": layout.names, "z_mean": float(Z.mean()), "z_std": float(Z.std(ddof=1)),
           "white_mean": float(Wz.mean()), "white_std": float(Wz.std(ddof=1)),
           "chi2_mean": float(chi2.mean()), "chi2_dof": layout.P,
           "fit_sigma_over_crb_median": float(np.median(ratio)),
           "per_param_rms_z": {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in per_param.items()},
           "seconds": time.time() - t0}
    res["pass"] = bool(abs(res["z_mean"]) <= 0.25 and 0.8 <= res["z_std"] <= 1.2)
    log("")
    log(f"pooled z over {len(Z)} values: mean {res['z_mean']:+.3f}, std {res['z_std']:.3f}")
    log(f"whitened: mean {res['white_mean']:+.3f}, std {res['white_std']:.3f}; chi2 mean {res['chi2_mean']:.1f} "
        f"(expected {layout.P})")
    log(f"the fit's own sigmas over the Cramer-Rao ones: median {res['fit_sigma_over_crb_median']:.3f}")
    log("RMS z per parameter: " + " ".join(f"{k}:{v:.2f}" for k, v in res["per_param_rms_z"].items()))
    log(f"  {'PASS' if res['pass'] else 'FAIL'}  z mean within +-0.25 and std 0.8-1.2")
    log(f"time {res['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    with open(str(rigs.HERE / "t4_fits.json"), "w", encoding="utf-8") as fh:
        json.dump({"truth": [cam_dict(c) for c in truth], "layout_names": layout.names, "sigma_px": SIGMA,
                   "fits": fits}, fh)
    log.close()


if __name__ == "__main__":
    main()
