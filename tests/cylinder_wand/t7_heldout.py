"""T7: held-out errors against the Cramer-Rao prediction.

(a) Unseen points.  For each of T4's 20 fits (reference rig, 0.1 px), a 7x7x7 grid over the +-20 mm volume is
    projected with the fitted and the true cameras.  The realised errors, pooled over seeds as an RMS, against the
    prediction from that seed's Cramer-Rao covariance (`cylinder_wand.predicted_projection_error`):
      fixed          projecting a known point in the fit's gauge;
      stabiliser     after the best common spin and slide of the points;
      retriangulated the true pixels re-triangulated (maximum likelihood) with the fitted cameras, the
                     residual RMS per 2-D image (gauge-free).
    For a single fit the realised error is one draw of the calibration error, dominated by a few modes, so the
    comparison pools the 20 fits.
(b) Held-out wand frames (5-fold, two seeds).  Fit on four fifths of the frames; on the held-out fifth each end is
    triangulated by maximum likelihood with the fitted cameras, without the length constraint:
      reprojection RMS per 2-D image, predicted sqrt(sigma^2 (2 - 3 / N_seen) + calibration part);
      wand length error (measured - L), predicted from each frame's triangulation covariance.
Criterion (plan section 7): realised within 1.2x the prediction (and above 0.8x) for the gauge-free and held-out
figures; the gauge-dependent ones (fixed, stabiliser) are reported.
Outputs: t7_heldout_out.txt, t7_heldout.json.
"""
import json
import math
import time

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

OUT = rigs.HERE / "t7_heldout"
log = rigs.Tee(str(OUT) + "_out.txt")
SIGMA = 0.1
ey = np.array([0.0, 1.0, 0.0])


def cam_of(d):
    return cm.Camera(np.array(d["R"]), np.array(d["C"]), d["f"], d["cx"], d["cy"], d["width"], d["height"],
                     np.array(d["dist"]))


def realised_unseen(fit, truth, tube, P):
    return rigs.realised_unseen(fit, tube, truth, tube, P)


def part_a():
    data = json.load(open(rigs.HERE / "t4_fits.json"))
    truth = [cam_of(d) for d in data["truth"]]
    tube = rigs.CASE_TUBE
    layout = cw.Layout(3)
    g = np.linspace(-20, 20, 7)
    P = np.array([[x, y, z] for x in g for y in g for z in g])
    real = {"fixed": [], "stabiliser": [], "retriangulated": []}
    pred = {"fixed": [], "stabiliser": [], "retriangulated": []}
    for fd in data["fits"]:
        fit = [cam_of(d) for d in fd["fit"]]
        cov = np.array(fd["cov_crb"])
        f_, s_, r_ = realised_unseen(fit, truth, tube, P)
        pe = cw.predicted_projection_error(truth, tube, layout, cov, P)
        for k, v in (("fixed", f_), ("stabiliser", s_), ("retriangulated", r_)):
            real[k].append(v)
        pred["fixed"].append(pe["fixed_rms_px"])
        pred["stabiliser"].append(pe["stabiliser_rms_px"])
        pred["retriangulated"].append(pe["retriangulated_rms_px"])
        log(f"  seed {fd['seed']:2d}: realised fixed {f_:.3f}, stabiliser {s_:.4f}, retri {r_:.5f} px | predicted "
            f"{pe['fixed_rms_px']:.3f}, {pe['stabiliser_rms_px']:.4f}, {pe['retriangulated_rms_px']:.5f}")
    out = {}
    for k in real:
        rr = math.sqrt(float(np.mean(np.square(real[k]))))
        pp = math.sqrt(float(np.mean(np.square(pred[k]))))
        out[k] = {"realised_rms_px": rr, "predicted_rms_px": pp, "ratio": rr / pp}
        log(f"(a) unseen points, {k:14s}: realised {rr:.5f} px, predicted {pp:.5f} px, ratio {rr / pp:.3f}")
    return out


def part_b(seed):
    tube = rigs.CASE_TUBE
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    rng = np.random.default_rng(7000 + seed)
    wd = rigs.wand_frames(truth, tube, 500, 10.0, rng, half=(20, 20, 20), noise_px=SIGMA)
    sizes = [(c.width, c.height) for c in truth]
    folds = np.arange(wd.uv.shape[0]) % 5
    reps, lens, pred_rep, pred_len, insample = [], [], [], [], []
    for k in range(5):
        train = folds != k
        data = cw.WandData(wd.uv[train], 10.0, sizes)
        obs = cw.observations(data)
        layout = cw.Layout(3)
        start = rigs.perturb_rig(truth, rng, rot_deg=1.0, trans_mm=10.0, f_rel=0.02)
        cams3, _ = cw.coarse_fit(start, tube, obs, 10.0, layout)
        sol = cw.refine(cams3, tube, cw.initial_frames(cams3, tube, obs, 10.0), 10.0, obs, layout)
        insample.append(sol.rms)
        # held-out frames: every detected end, triangulated without the length constraint
        uvh = wd.uv[~train]
        Fh = uvh.shape[0]
        rows = [(f, c, e) for f in range(Fh) for e in range(2) for c in range(3) if np.all(np.isfinite(uvh[f, c, e]))]
        fr_ = np.array([r[0] for r in rows])
        ca_ = np.array([r[1] for r in rows])
        en_ = np.array([r[2] for r in rows])
        pt_idx = fr_ * 2 + en_
        X0 = np.zeros((2 * Fh, 3))
        X0[0::2] = wd.XA[~train]
        X0[1::2] = wd.XB[~train]
        X, r, cov = cw.triangulate_ml(sol.cams, tube, ca_, uvh[fr_, ca_, en_], pt_idx, X0, sigma_px=SIGMA)
        reps.append(r)
        nseen = np.bincount(pt_idx, minlength=2 * Fh)
        pred_rep.append(SIGMA ** 2 * (2.0 - 3.0 / nseen[pt_idx]))
        XA, XB = X[0::2], X[1::2]
        u = (XB - XA) / np.linalg.norm(XB - XA, axis=1, keepdims=True)
        lens.append(np.linalg.norm(XB - XA, axis=1) - 10.0)
        pred_len.append(np.einsum("ni,nij,nj->n", u, cov[0::2] + cov[1::2], u))
    r = np.vstack(reps)
    rep = math.sqrt(float(np.nanmean(np.sum(r ** 2, axis=1))))
    rep_pred = math.sqrt(float(np.mean(np.concatenate(pred_rep))))
    le = np.concatenate(lens)
    len_rms = math.sqrt(float(np.mean(le ** 2)))
    len_pred = math.sqrt(float(np.mean(np.concatenate(pred_len))))
    out = {"seed": seed, "heldout_reproj_rms_px": rep, "heldout_reproj_pred_px": rep_pred,
           "reproj_ratio": rep / rep_pred, "length_mean_mm": float(le.mean()), "length_rms_mm": len_rms,
           "length_pred_mm": len_pred, "length_ratio": len_rms / len_pred,
           "insample_rms_px": float(np.mean(insample))}
    log(f"(b) seed {seed}: held-out reprojection {rep:.4f} px (predicted {rep_pred:.4f}, ratio {rep / rep_pred:.3f}); "
        f"wand length {le.mean() * 1e3:+.2f} um mean, {len_rms * 1e3:.2f} um RMS (predicted {len_pred * 1e3:.2f}, "
        f"ratio {len_rms / len_pred:.3f}); in-sample 2-D rms {np.mean(insample):.4f} px")
    return out


def main():
    t0 = time.time()
    res = {"unseen": part_a(), "heldout_frames": [part_b(0), part_b(1)], "seconds": None}
    ratios = [res["unseen"]["retriangulated"]["ratio"]]
    for hb in res["heldout_frames"]:
        ratios += [hb["reproj_ratio"], hb["length_ratio"]]
    res["gated_ratios"] = ratios
    res["pass"] = bool(all(0.8 <= x <= 1.2 for x in ratios))
    res["seconds"] = time.time() - t0
    log(f"  {'PASS' if res['pass'] else 'FAIL'}  gauge-free and held-out errors within 0.8-1.2x the prediction "
        f"({', '.join(f'{x:.3f}' for x in ratios)})")
    log(f"time {res['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
