"""V0, the truth-free side (plan: "a truth-free estimate that uses no tracks ... to show ghosts don't drive it"),
done two ways, after v0_pattern.py's first estimator (the T1-a peak per cell) proved too noisy per cell.

    python v0b_truthfree.py [--name t1_m15]

Data: every one of S2's candidate pairs gathered by gather.py (all detections, curve distance <= 8 px, inside the curve,
the lines of sight within 1 mm, the midpoint in the view volume) with its signed offset o (cam2's detection minus cam1's
curve).  True pairs form a peak about the local offset; random pairs a flat floor.
  1. The null direction: the two cameras' 4-vector N(X) across which the offset lies, on a grid over the view volume
     (selfcal.null_grid); checked on the track points: o = (projected residual) / N_v2.
  2. Per region, by maximum likelihood (a Gaussian on a flat floor, selfcal.peak_ml): the centre of the true pairs'
     offsets per cam2 cell (160 and 320 px), against the tracks' medians (v0_pattern's map, the same cells).
  3. A truth-free field: selfcal.mixture_fit on every pair within +-3 px (EM: true pairs ~ the field's offset plus the
     noise model, random ones uniform); fitted on frames 0, 10, ... and on all frames; the bending weight by the held-out
     log-likelihood.  Its offset at the track points against the tracks' field (v0_pattern's), and its held-out
     sharpening of the true pairs (the mixture's spread on frames 5, 15, ... with and without it).
Writes v0b_truthfree_out.txt, .json, v0b_truthfree.png and data/fields/<name>_tf_{fit,all}.npz.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import selfcal as sc
from v0_pattern import corr

WIN = 3.0
LAMS = [1e3, 1e4, 1e5, 1e6]


def pairs_table(recs):
    """All candidate pairs: frame, cam1 detection (u, v, Ly), cam2 detection (u, v, Ly), X, signed offset."""
    rows = []
    for g in recs:
        C = g["cand"]
        if not len(C):
            continue
        i, j = C[:, 0].astype(int), C[:, 1].astype(int)
        d0, d1 = g["det0"], g["det1"]
        rows.append(np.column_stack([np.full(len(C), int(g["t"])), d0[i, 0], d0[i, 1], d0[i, 3], d1[j, 0], d1[j, 1],
                                     d1[j, 3], C[:, 2:5], C[:, 7]]))
    return np.vstack(rows)


def loglik(field, d1, d2, Nv, y, Lv1, Lv2, noise, rec, window):
    """The mean log-likelihood of pairs under the mixture with the fitted share and spread."""
    c = np.zeros(len(y))
    c += Nv[:, 1] * field.eval_points(0, d1)[:, 1] + Nv[:, 3] * field.eval_points(1, d2)[:, 1]
    e = y - c
    v = rec["kappa"] ** 2 * (Nv[:, 1] ** 2 * sc.noise_sigma(Lv1, noise) ** 2
                             + Nv[:, 3] ** 2 * sc.noise_sigma(Lv2, noise) ** 2)
    g = np.exp(-0.5 * e * e / v) / np.sqrt(2 * math.pi * v)
    f = 1 - rec["true_share"]
    return float(np.mean(np.log((1 - f) * g + f / (2 * window * Nv[:, 3]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t1_m15")
    args = ap.parse_args()
    log = sc.Log("v0b_truthfree")
    recs = sc.load_gather(args.name)
    v0 = sc.load_json(os.path.join(sc.HERE, "v0_pattern.json"))
    noise = tuple(v0["field"]["noise"])
    lam_tr = v0["field"]["lam"]
    out = {"gather": args.name, "noise_model": noise}
    T = pairs_table(recs)
    fr, d1, Lv1, d2, Lv2, X, o = T[:, 0], T[:, 1:3], T[:, 3], T[:, 4:6], T[:, 6], T[:, 7:10], T[:, 10]
    log(f"V0b on {args.name}: {len(T)} candidate pairs in {len(recs)} frames; noise model {noise}")

    # 1. the null direction
    Nf, info = sc.null_grid(sc.CAMS_T1, sc.view_volume())
    fd = sc.fitdata(recs)
    Xt = np.concatenate([g["X"][((g["pair"] >= 0) & (g["pair_dist"] <= sc.R_PAIR)).all(axis=1)
                                & np.all(np.isfinite(g["J"].reshape(len(g["X"]), -1)), axis=1)] for g in recs])
    e12 = np.concatenate([g["epi12"][((g["pair"] >= 0) & (g["pair_dist"] <= sc.R_PAIR)).all(axis=1)
                                     & np.all(np.isfinite(g["J"].reshape(len(g["X"]), -1)), axis=1)] for g in recs])
    obs0 = sc.observable(fd, fd.r)
    Nt = Nf(Xt)
    pred = obs0 / Nt[:, 3]
    ok = np.isfinite(pred) & np.isfinite(e12)
    dev = pred[ok] - e12[ok]
    log(f"  1. the null direction on {info['nodes']} nodes ({info['seen']} seen by both cameras); N at the volume's "
        f"centre {np.round(Nf(np.array([[np.mean(sc.view_volume()['x']), np.mean(sc.view_volume()['y']), np.mean(sc.view_volume()['z'])]]))[0], 4).tolist()}; "
        f"on the track points the offset = projected / N_v2 to {np.median(np.abs(dev)):.4f} px (median), "
        f"{np.percentile(np.abs(dev), 95):.4f} (95 %)")
    out["null"] = {"nodes": info, "check_median_abs_px": float(np.median(np.abs(dev))),
                   "check_p95_abs_px": float(np.percentile(np.abs(dev), 95))}

    # 2. per cell by maximum likelihood
    out["cells"] = {}
    for cell in (160.0, 320.0):
        nx, ny = int(math.ceil(1280 / cell)), int(math.ceil(800 / cell))
        iu = np.clip(((d2[:, 0] + 0.5) // cell).astype(int), 0, nx - 1)
        iv = np.clip(((d2[:, 1] + 0.5) // cell).astype(int), 0, ny - 1)
        est = np.full((ny, nx), np.nan)
        ses = np.full((ny, nx), np.nan)
        ntrue = np.zeros((ny, nx))
        for a in range(ny):
            for b in range(nx):
                m = (iv == a) & (iu == b)
                if m.sum() < 2000:
                    continue
                mu, se, s, sh, nt = sc.peak_ml(o[m])
                if nt >= 100 and np.isfinite(se):
                    est[a, b], ses[a, b], ntrue[a, b] = mu, se, nt
        n_t, med_t, se_t = sc.cell_stats(fd.d[:, 1], e12, cell=cell)
        cc, nc = corr(est, med_t)
        dd = est - med_t
        okd = np.isfinite(dd)
        # the expected scatter of the difference from the two estimates' standard errors
        exp_sd = float(np.sqrt(np.nanmean((ses ** 2 + se_t ** 2)[okd]))) if okd.any() else None
        st_tf = sc.map_rms(est, ses, ntrue.astype(int))
        st_tr = sc.map_rms(med_t, se_t, n_t)
        out["cells"][str(int(cell))] = {"truth_free": est.tolist(), "se": ses.tolist(), "true_pairs": ntrue.tolist(),
                                        "tracks": med_t.tolist(), "tracks_se": se_t.tolist(), "corr": cc,
                                        "cells_compared": nc,
                                        "rms_difference": float(np.sqrt(np.mean(dd[okd] ** 2))) if okd.any() else None,
                                        "expected_difference_sd": exp_sd, "truth_free_stats": st_tf,
                                        "tracks_stats": st_tr}
        log(f"  2. cells of {cell:.0f} px: {nc} cells; truth-free (ML) centres' RMS about their mean "
            f"{st_tf['rms']:.3f} px ({st_tf['rms_noise_corrected']:.3f} with their noise ({st_tf['se_rms']:.3f}) taken "
            f"off), the tracks' {st_tr['rms']:.3f} ({st_tr['rms_noise_corrected']:.3f}); correlation {cc:.2f}; RMS "
            f"difference {out['cells'][str(int(cell))]['rms_difference']:.3f} px against {exp_sd:.3f} expected from the "
            f"two estimates' standard errors")

    # 3. the truth-free field
    sel = np.abs(o) <= WIN
    Nv = Nf(X[sel])
    y = o[sel] * Nv[:, 3]
    fsel = fr[sel]
    fit_m = fsel % 10 == 0
    args_all = (d1[sel], d2[sel], Nv, y, Lv1[sel], Lv2[sel])
    def sub(m):
        return tuple(a[m] for a in args_all)
    log(f"  3. the truth-free field: {int(sel.sum())} pairs within +-{WIN} px ({int(fit_m.sum())} on frames 0, 10, ...)")
    table = []
    best = None
    for lam in LAMS:
        f = sc.Field(2)
        r = sc.mixture_fit(f, *sub(fit_m), lam, noise, WIN)
        ll1 = loglik(f, *sub(~fit_m), noise, r, WIN)
        f0 = sc.Field(2)
        ll0 = loglik(f0, *sub(~fit_m), noise, r, WIN)
        table.append({"lam": lam, "heldout_loglik": ll1, "heldout_loglik_no_field": ll0, "fit": r})
        log(f"    lam {lam:g}: true pairs {r['true_pairs']:.0f} ({100 * r['true_share']:.1f} %), spread {r['kappa']:.3f};"
            f" held-out log-likelihood per pair {ll0:.5f} -> {ll1:.5f}")
        if best is None or ll1 > best[1]:
            best = (lam, ll1)
    lam = best[0]
    out["lambda_table"] = table
    # EM finds a local maximum: also start it from the tracks' field, and keep the higher likelihood
    f_tr0 = sc.Field.load(os.path.join(sc.DATA, "fields", f"{args.name}_fit.npz"))
    starts = {}
    for st, th0 in (("cold", None), ("warm", f_tr0.theta)):
        f = sc.Field(2)
        r = sc.mixture_fit(f, *sub(fit_m), lam, noise, WIN, init_theta=th0, log=log)
        starts[st] = (f, r)
    pick = max(starts, key=lambda k: starts[k][1]["loglik"])
    log(f"    frames 0, 10, ...: mean log-likelihood from zero {starts['cold'][1]['loglik']:.6f}, from the tracks' field "
        f"{starts['warm'][1]['loglik']:.6f}: the {pick} start's maximum is kept")
    out["starts"] = {k: {kk: v[1][kk] for kk in ("loglik", "true_pairs", "true_share", "kappa", "iterations")}
                     for k, v in starts.items()}
    fields = {}
    for nm, m in (("fit", fit_m), ("all", np.ones(len(y), bool))):
        f = sc.Field(2)
        th0 = starts[pick][0].theta if pick == "warm" or nm == "fit" else None
        if nm == "fit":
            f, r = starts[pick]
        else:
            r = sc.mixture_fit(f, *sub(m), lam, noise, WIN, init_theta=(f_tr0.theta if pick == "warm" else None),
                               log=log)
        f.make_dense([d1[sel][m], d2[sel][m]])
        f.meta = {"gather": args.name, "frames": nm, "lam": lam, "noise": list(noise), "mixture": r,
                  "kind": "truth-free (all candidate pairs, EM)", "start": pick}
        f.save(os.path.join(sc.DATA, "fields", f"{args.name}_tf_{nm}.npz"))
        fields[nm] = (f, r)
        del th0
    # held out: the mixture's spread on frames 5, 15, ... with the fitting frames' field, against no field
    f_fit, r_fit = fields["fit"]
    def spread(field):
        d1h, d2h, Nh, yh, l1, l2 = sub(~fit_m)
        g = sc.Field(2)
        g.theta = field.theta.copy() if field is not None else g.theta
        c = Nh[:, 1] * g.eval_points(0, d1h)[:, 1] + Nh[:, 3] * g.eval_points(1, d2h)[:, 1]
        g2 = sc.Field(2)
        rr = sc.mixture_fit(g2, d1h, d2h, Nh, yh - c, l1, l2, 1e12, noise, WIN, tau=1e-6)   # field frozen at 0
        return rr
    r_no = spread(None)
    r_tf = spread(f_fit)
    f_tr = sc.Field.load(os.path.join(sc.DATA, "fields", f"{args.name}_fit.npz"))
    r_trk = spread(f_tr)
    log(f"    held out (frames 5, 15, ...), the mixture refitted with the field frozen: no field: true pairs "
        f"{r_no['true_pairs']:.0f}, spread {r_no['kappa']:.3f}; the truth-free field: {r_tf['true_pairs']:.0f}, "
        f"{r_tf['kappa']:.3f}; the tracks' field (v0_pattern, frames 0, 10, ...): {r_trk['true_pairs']:.0f}, "
        f"{r_trk['kappa']:.3f}")
    out["heldout_spread"] = {"none": {k: r_no[k] for k in ("true_pairs", "true_share", "kappa")},
                             "truth_free": {k: r_tf[k] for k in ("true_pairs", "true_share", "kappa")},
                             "tracks": {k: r_trk[k] for k in ("true_pairs", "true_share", "kappa")}}
    # the two fields' offsets at the track points
    f_all = fields["all"][0]
    f_tr_all = sc.Field.load(os.path.join(sc.DATA, "fields", f"{args.name}_all.npz"))
    p_tf = sc.observable(fd, sc.field_at(f_all, fd))
    p_tr = sc.observable(fd, sc.field_at(f_tr_all, fd))
    cc, _n = corr(p_tf, p_tr)
    rd = float(np.sqrt(np.nanmean((p_tf - p_tr) ** 2)))
    a, b = np.polyfit(p_tr[np.isfinite(p_tr) & np.isfinite(p_tf)], p_tf[np.isfinite(p_tr) & np.isfinite(p_tf)], 1)
    log(f"    at the track points, the offset each field predicts (null direction): truth-free RMS "
        f"{np.sqrt(np.nanmean(p_tf ** 2)):.3f} px, the tracks' {np.sqrt(np.nanmean(p_tr ** 2)):.3f}; correlation "
        f"{cc:.2f}, RMS difference {rd:.3f} px; truth-free = {a:.2f} x tracks' {b:+.3f}")
    out["fields_compared"] = {"rms_truth_free": float(np.sqrt(np.nanmean(p_tf ** 2))),
                              "rms_tracks": float(np.sqrt(np.nanmean(p_tr ** 2))), "corr": cc, "rms_difference": rd,
                              "slope": float(a), "intercept": float(b), "lam": lam}
    for c in (0, 1):
        g1, g2 = f_all.dense[c][1], f_tr_all.dense[c][1]
        cov = f_all.covered[c] & f_tr_all.covered[c]
        log(f"    cam{c + 1} dv over the pixels both cover: truth-free RMS {np.sqrt(np.mean(g1[cov] ** 2)):.3f}, tracks' "
            f"{np.sqrt(np.mean(g2[cov] ** 2)):.3f} px; correlation {np.corrcoef(g1[cov], g2[cov])[0, 1]:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for k, cell in enumerate(("160", "320")):
        c = out["cells"][cell]
        a = ax[0, k]
        a.errorbar(np.array(c["tracks"]).ravel(), np.array(c["truth_free"]).ravel(),
                   xerr=np.array(c["tracks_se"]).ravel(), yerr=np.array(c["se"]).ravel(), fmt="o", ms=3, capsize=2)
        a.plot([-1.5, 1.5], [-1.5, 1.5], "k-", lw=0.5)
        a.set_xlabel("tracks: cell median of the offset [px]")
        a.set_ylabel("truth-free (ML) centre [px]")
        a.set_title(f"cam2 cells of {cell} px: corr {c['corr']:.2f}", fontsize=9)
    a = ax[0, 2]
    a.scatter(p_tr, p_tf, s=1, alpha=0.3)
    a.plot([-1.5, 1.5], [-1.5, 1.5], "k-", lw=0.5)
    a.set_xlabel("the tracks' field: offset at the track points [px]")
    a.set_ylabel("the truth-free field's [px]")
    a.set_title(f"corr {cc:.2f}", fontsize=9)
    for c in (0, 1):
        a = ax[1, c]
        img = np.where(f_all.covered[c], f_all.dense[c][1], np.nan)
        im = a.imshow(img, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        a.set_title(f"cam{c + 1} dv: the truth-free field [px]", fontsize=9)
        plt.colorbar(im, ax=a, fraction=0.03)
    a = ax[1, 2]
    h, e = np.histogram(o, bins=np.arange(-8, 8.001, 0.25))
    a.bar(0.5 * (e[1:] + e[:-1]), h, width=0.25, color="0.6")
    a.set_xlabel("signed offset of all candidate pairs [px]")
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, "v0b_truthfree.png"), dpi=100)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
