"""V0 (plan, section 3): is there a pattern in T1's residuals that an image-space field can correct?

    python v0_pattern.py [--name t1_m15] [--gate 3]

Data: gather.py's frames of T1's 15 px run (every 5th of frames 0-250; the long tracks of at least 10 points; each
point paired one to one with the line detector's detections within GATE px in both cameras).
  1. The pairing: the share of points paired at 2, 3, 4 and 6 px, and the residuals' spread at each gate.
  2. The residuals per camera, du, dv = detection - the track's projection (whose 3-D point the tracker fitted to both
     cameras), and their spread against the detection's box length along each axis.
  3. Maps by region (cells of 160 px): per camera the medians of du and dv, and the signed offset across the epipolar
     curve (cam2's detection minus cam1's curve, by cam2's cell; cam1's against cam2's curve, by cam1's cell); the RMS of
     the cell medians, also with the medians' noise taken off.
  4. Stability: the same maps from frames 0-125 and 130-250; the correlation of the cell medians.
  5. A truth-free estimate that uses no tracks: per cam2 cell, the centre of the true pairs' peak in the signed offsets
     of all of S2's candidate pairs, against the tracks' cell medians: if ghosts drove the tracks' pattern, the two would
     differ.  T1-a's method (the excess's centroid about the histogram's maximum) proved too noisy per 160 px cell
     (the true pairs stand about 1.7 sigma above the floor per bin), so the judgement uses the maximum-likelihood
     centre (a Gaussian on a sloped floor, selfcal.peak_ml) per 320 px cell; all three are reported.
  6. The field: the projected fit (selfcal.fit_field) on all points, its bending weight by held-out loss (fitted on
     frames 0, 10, ..., judged on 5, 15, ...) and the noise model by the residuals; the fit on each half of the frames;
     the fields' agreement between the halves in the observable (the epipolar offset each predicts at the held-out
     points); the coverage and the standard error.
  7. Depth against image position: the offsets binned by each camera's depth (z_tan - z_f, the side and distance from
     focus) before the fit, and the residual after it; the image field's own size (the RMS of the offset it predicts)
     against the residual's depth trend (the RMS of its depth-bin medians, noise taken off).
Judged (plan): a real pattern when the regional medians' RMS (noise taken off) is at least 0.3 px, the halves agree
(correlation >= 0.7) and the two estimates agree; stop if the offsets depend on depth more than on image position.
Writes v0_pattern_out.txt, .json, v0_pattern.png, v0_fields.png and data/fields/t1_m15_{all,h1,h2,fit}.npz.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import selfcal as sc

LAMS = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
HALF = 125


def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def depth_bins(x, y, nb=8):
    """Quantile bins of x: centres, medians of y, their standard errors."""
    ok = np.isfinite(x) & np.isfinite(y)
    edges = np.quantile(x[ok], np.linspace(0, 1, nb + 1))
    cs, ms, ses, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (x >= lo) & (x <= hi)
        if m.sum() < 20:
            continue
        v = y[m]
        md = float(np.median(v))
        cs.append(float(np.median(x[m])))
        ms.append(md)
        ses.append(1.2533 * 1.4826 * float(np.median(np.abs(v - md))) / math.sqrt(m.sum()))
        ns.append(int(m.sum()))
    cs, ms, ses, ns = map(np.array, (cs, ms, ses, ns))
    mu = float(np.average(ms, weights=ns))
    raw = float(np.sqrt(np.average((ms - mu) ** 2, weights=ns)))
    corrected = math.sqrt(max(raw ** 2 - float(np.average(ses ** 2, weights=ns)), 0.0))
    return {"centres": cs.tolist(), "medians": ms.tolist(), "se": ses.tolist(), "n": ns.tolist(), "rms": raw,
            "rms_noise_corrected": corrected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t1_m15")
    ap.add_argument("--gate", type=float, default=sc.R_PAIR)
    args = ap.parse_args()
    log = sc.Log("v0_pattern")
    recs = sc.load_gather(args.name)
    frames = [int(g["t"]) for g in recs]
    out = {"gather": args.name, "frames": frames, "gate_px": args.gate}
    log(f"V0 on {args.name}: {len(recs)} frames ({frames[0]}..{frames[-1]}), gate {args.gate} px")
    shape = sc.load_json(os.path.join(sc.CASE_DATA, "T1_shape.json"))
    zf = [c["z_f_mm"] for c in shape["cameras"]]

    # 1. the pairing at several gates
    n_all = sum(len(g["X"]) for g in recs)
    pg = {}
    for gate in (2.0, 3.0, 4.0, 6.0):
        fdg = sc.fitdata(recs, gate)
        rr = fdg.r
        pg[gate] = {"points": len(fdg), "share": len(fdg) / max(n_all, 1),
                    "rms": [[float(np.sqrt(np.mean(rr[:, c, k] ** 2))) for k in (0, 1)] for c in (0, 1)]}
        log(f"  gate {gate:.0f} px: {len(fdg)} of {n_all} long-track points paired in both cameras "
            f"({100 * pg[gate]['share']:.0f} %); RMS du, dv cam1 {pg[gate]['rms'][0][0]:.3f}, {pg[gate]['rms'][0][1]:.3f}"
            f", cam2 {pg[gate]['rms'][1][0]:.3f}, {pg[gate]['rms'][1][1]:.3f} px")
    out["pairing"] = {str(k): v for k, v in pg.items()}
    fd = sc.fitdata(recs, args.gate)
    log(f"  using gate {args.gate} px: {len(fd)} points, {len(np.unique(fd.track))} tracks")

    # epipolar offsets of the same points (gather.py: epi12 by cam2's detection, epi21 by cam1's)
    e12 = np.concatenate([g["epi12"][((g["pair"] >= 0) & (g["pair_dist"] <= args.gate)).all(axis=1)
                                     & np.all(np.isfinite(g["J"].reshape(len(g["X"]), -1)), axis=1)]
                          for g in recs if len(g["X"])])
    e21 = np.concatenate([g["epi21"][((g["pair"] >= 0) & (g["pair_dist"] <= args.gate)).all(axis=1)
                                     & np.all(np.isfinite(g["J"].reshape(len(g["X"]), -1)), axis=1)]
                          for g in recs if len(g["X"])])
    zt = [np.concatenate([g[f"z_track{c}"][((g["pair"] >= 0) & (g["pair_dist"] <= args.gate)).all(axis=1)
                                           & np.all(np.isfinite(g["J"].reshape(len(g["X"]), -1)), axis=1)]
                          for g in recs if len(g["X"])]) for c in (0, 1)]
    assert len(e12) == len(fd)
    obs0 = sc.observable(fd, fd.r)
    ok_o = np.isfinite(obs0) & np.isfinite(e12)
    k_lin = float(np.polyfit(obs0[ok_o], e12[ok_o], 1)[0])
    log(f"  the epipolar offset (cam2 detection - cam1's curve): median {np.nanmedian(e12):+.3f} px, RMS "
        f"{np.sqrt(np.nanmean(e12 ** 2)):.3f}, robust SD {1.4826 * np.nanmedian(np.abs(e12 - np.nanmedian(e12))):.3f}; "
        f"the projected residual of the same points: robust SD {1.4826 * np.nanmedian(np.abs(obs0 - np.nanmedian(obs0))):.3f}"
        f" (offset = {k_lin:.3f} x projected; correlation {corr(obs0, e12)[0]:.4f})")
    out["epipolar_offset"] = {"median": float(np.nanmedian(e12)), "rms": float(np.sqrt(np.nanmean(e12 ** 2))),
                              "offset_per_projected": k_lin, "corr_projected": corr(obs0, e12)[0]}

    # 2. residuals against the box lengths
    out["residuals"] = {}
    for c in (0, 1):
        rec = {}
        for k, ax in ((0, "u"), (1, "v")):
            r = fd.r[:, c, k]
            L = np.abs(fd.L[:, c, k])
            bins = [0, 2, 5, 10, 20, 60]
            rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (L >= lo) & (L < hi)
                if m.sum() < 20:
                    continue
                rows.append({"L": [lo, hi], "n": int(m.sum()), "median": float(np.median(r[m])),
                             "robust_sd": float(1.4826 * np.median(np.abs(r[m] - np.median(r[m]))))})
            rec[ax] = {"median": float(np.median(r)), "rms": float(np.sqrt(np.mean(r ** 2))),
                       "robust_sd": float(1.4826 * np.median(np.abs(r - np.median(r)))), "by_length": rows}
            log(f"  cam{c + 1} d{ax}: median {rec[ax]['median']:+.3f}, RMS {rec[ax]['rms']:.3f}, robust SD "
                f"{rec[ax]['robust_sd']:.3f} px; by the box length along {ax}: "
                + "; ".join(f"{q['L'][0]}-{q['L'][1]} px: {q['robust_sd']:.2f} (n {q['n']})" for q in rows))
        out["residuals"][f"cam{c + 1}"] = rec

    # 3. maps by region; 4. the halves
    maps = {}
    h1 = fd.frame <= HALF
    def mk(uv, val, sel=None):
        s = np.ones(len(val), bool) if sel is None else sel
        n, med, se = sc.cell_stats(uv[s], val[s])
        return n, med, se
    items = {"cam1_du": (fd.d[:, 0], fd.r[:, 0, 0]), "cam1_dv": (fd.d[:, 0], fd.r[:, 0, 1]),
             "cam2_du": (fd.d[:, 1], fd.r[:, 1, 0]), "cam2_dv": (fd.d[:, 1], fd.r[:, 1, 1]),
             "epi12_by_cam2": (fd.d[:, 1], e12), "epi21_by_cam1": (fd.d[:, 0], e21)}
    for key, (uv, val) in items.items():
        n, med, se = mk(uv, val)
        n1, m1, s1 = mk(uv, val, h1)
        n2, m2, s2 = mk(uv, val, ~h1)
        st = sc.map_rms(med, se, n)
        cc, nc = corr(m1, m2)
        maps[key] = {"n": n.tolist(), "median": med.tolist(), "se": se.tolist(), "stats": st,
                     "half1": m1.tolist(), "half2": m2.tolist(), "halves_corr": cc, "halves_cells": nc}
        log(f"  map {key}: {st['cells']} cells (>= 30 points); cell medians about their mean {st['mean']:+.3f}: RMS "
            f"{st['rms']:.3f} px, {st['rms_noise_corrected']:.3f} with the medians' noise ({st['se_rms']:.3f}) taken off;"
            f" halves: correlation {cc:.2f} over {nc} cells")
    out["maps"] = maps

    # 5. the truth-free estimate per cam2 cell from all candidate pairs
    cu, cv_ = [], []
    offs = []
    for g in recs:
        C = g["cand"]
        if not len(C):
            continue
        j = C[:, 1].astype(int)
        cu.append(g["det1"][j, 0])
        cv_.append(g["det1"][j, 1])
        offs.append(C[:, 7])
    cu, cv_, offs = map(np.concatenate, (cu, cv_, offs))
    pk_all, ex_all, fl_all = sc.peak_centre(offs)
    out["truth_free"] = {"all_pairs_peak": pk_all, "all_pairs_excess": ex_all, "floor_per_bin": fl_all}
    log(f"  truth-free (all {len(offs)} candidate pairs, no tracks): the peak of the signed offsets at {pk_all:+.3f} px "
        f"({ex_all:.0f} pairs above a floor of {fl_all:.0f} per 0.25 px)")
    # per cam2 cell: T1-a's peak (the first estimator: the centroid of the excess about the histogram's maximum,
    # too noisy per 160 px cell, where the true pairs' peak stands about 1.7 sigma above the floor per bin) and the
    # maximum-likelihood centre (a Gaussian on a sloped floor, selfcal.peak_ml), at 160 and 320 px
    for cell, method in ((160.0, "t1a"), (160.0, "ml"), (320.0, "ml")):
        nx, ny = int(math.ceil(1280 / cell)), int(math.ceil(800 / cell))
        est = np.full((ny, nx), np.nan)
        ses = np.full((ny, nx), np.nan)
        iu = np.clip(((cu + 0.5) // cell).astype(int), 0, nx - 1)
        iv = np.clip(((cv_ + 0.5) // cell).astype(int), 0, ny - 1)
        for a in range(ny):
            for b in range(nx):
                m = (iv == a) & (iu == b)
                if m.sum() < 2000:
                    continue
                if method == "t1a":
                    p, e, _f = sc.peak_centre(offs[m])
                    if e >= 100:
                        est[a, b] = p
                else:
                    mu, se, _s, _sh, nt = sc.peak_ml(offs[m])
                    if nt >= 100 and np.isfinite(se):
                        est[a, b], ses[a, b] = mu, se
        n_t, med_t, se_t = sc.cell_stats(fd.d[:, 1], e12, cell=cell)
        cc, nc = corr(est, med_t)
        dd = est - med_t
        okd = np.isfinite(dd)
        exp_sd = float(np.sqrt(np.nanmean((ses ** 2 + se_t ** 2)[okd]))) if method == "ml" and okd.any() else None
        key = f"{method}_{int(cell)}"
        out["truth_free"][key] = {"cells": est.tolist(), "se": ses.tolist(), "tracks": med_t.tolist(),
                                  "corr_with_tracks": cc, "cells_compared": nc,
                                  "rms_difference": float(np.sqrt(np.mean(dd[okd] ** 2))) if okd.any() else None,
                                  "expected_difference_sd": exp_sd}
        log(f"  truth-free per cam2 cell ({method}, {cell:.0f} px): correlation with the tracks' medians {cc:.2f} over "
            f"{nc} cells, RMS difference {out['truth_free'][key]['rms_difference']:.3f} px"
            + (f" (expected from the two estimates' standard errors {exp_sd:.3f})" if exp_sd else ""))
        if key == "ml_320":
            out["truth_free"]["corr_with_tracks"] = cc
        if key == "t1a_160":
            est160, trk = est, med_t

    # 6. the field
    fit_m = fd.frame % 10 == 0
    log("  6. the field (2 cameras, B-splines 10 x 6 intervals; bending weight by held-out loss, noise by the residuals)")
    field = sc.Field(2)
    rec_all, lam, noise = sc.fit_with_noise(field, fd.subset(fit_m), fd.subset(~fit_m), LAMS, log)
    f_fit = field                              # fitted on frames 0, 10, ... (for V1)
    f_fit.make_dense([fd.subset(fit_m).d[fd.subset(fit_m).mask[:, c], c] for c in (0, 1)])
    f_fit.meta = {"gather": args.name, "frames": "t % 10 == 0", "lam": lam, "noise": noise}
    os.makedirs(os.path.join(sc.DATA, "fields"), exist_ok=True)
    f_fit.save(os.path.join(sc.DATA, "fields", f"{args.name}_fit.npz"))
    s_test = sc.heldout_score(f_fit, fd.subset(~fit_m), noise)
    log(f"    held out (frames 5, 15, ...): projected MAD {s_test['mad_before']:.3f} -> {s_test['mad_after']:.3f}, "
        f"{sc.ROBUST} loss {s_test['loss_before']:.4f} -> {s_test['loss_after']:.4f}")
    fields = {}
    for nm, sel in (("all", np.ones(len(fd), bool)), ("h1", h1), ("h2", ~h1)):
        f = sc.Field(2)
        r = sc.fit_field(f, fd.subset(sel), lam, noise, log=log)
        f.make_dense([fd.subset(sel).d[fd.subset(sel).mask[:, c], c] for c in (0, 1)])
        f.meta = {"gather": args.name, "frames": nm, "lam": lam, "noise": noise, "fit": r}
        f.save(os.path.join(sc.DATA, "fields", f"{args.name}_{nm}.npz"))
        fields[nm] = f
    # the observable each predicts at every point (2 cameras: the offset across the curve, in the null direction)
    pred = {nm: sc.observable(fd, sc.field_at(f, fd)) for nm, f in fields.items()}
    c12, _n = corr(pred["h1"][~h1], pred["h2"][~h1])
    c21, _n = corr(pred["h1"][h1], pred["h2"][h1])
    rms_pred = float(np.sqrt(np.nanmean((pred["all"] - np.nanmean(pred["all"])) ** 2)))
    rms_pred_m = float(np.sqrt(np.nanmean(pred["all"] ** 2)))
    diff_h = float(np.sqrt(np.nanmean((pred["h1"] - pred["h2"]) ** 2)))
    obs_after = obs0 - pred["all"]
    sd0 = 1.4826 * np.nanmedian(np.abs(obs0 - np.nanmedian(obs0)))
    sd1 = 1.4826 * np.nanmedian(np.abs(obs_after - np.nanmedian(obs_after)))
    log(f"    the field's offset across the curves (at the points): RMS {rms_pred_m:.3f} px ({rms_pred:.3f} about its "
        f"mean {np.nanmean(pred['all']):+.3f}); the halves' fields predict it alike: correlation {c12:.2f} (on the "
        f"second half's points) and {c21:.2f} (first half's), RMS difference {diff_h:.3f} px; the points' offset robust "
        f"SD {sd0:.3f} -> {sd1:.3f} px after it")
    se_med = []
    for c in (0, 1):
        uvs = fd.d[fd.mask[:, c], c]
        sel = np.random.default_rng(0).choice(len(uvs), size=min(400, len(uvs)), replace=False)
        se = sc.field_se(fields["all"], c, uvs[sel])
        se_med.append(np.median(se, axis=0).tolist())
        cov = fields["all"].covered[c]
        g = fields["all"].dense[c]
        log(f"    cam{c + 1}: the field over its covered pixels ({100 * cov.mean():.0f} % of the sensor): du RMS "
            f"{np.sqrt(np.mean(g[0][cov] ** 2)):.3f}, dv RMS {np.sqrt(np.mean(g[1][cov] ** 2)):.3f} px (max |c| "
            f"{np.sqrt(g[0][cov] ** 2 + g[1][cov] ** 2).max():.2f}); its standard error at the data (median) du "
            f"{se_med[-1][0]:.3f}, dv {se_med[-1][1]:.3f} px")
    out["field"] = {"lam": lam, "noise": noise, "selection": rec_all, "heldout": s_test,
                    "offset_rms_px": rms_pred_m, "offset_rms_about_mean_px": rms_pred, "offset_mean_px":
                    float(np.nanmean(pred["all"])), "halves_corr": [c12, c21], "halves_rms_diff_px": diff_h,
                    "points_offset_sd_before_after": [float(sd0), float(sd1)], "se_median_px": se_med,
                    "covered_share": [float(fields["all"].covered[c].mean()) for c in (0, 1)]}

    # 7. depth against image position
    dep = {}
    for c in (0, 1):
        x = zt[c][:, 0] - zf[c]
        b0 = depth_bins(x, obs0)
        b1 = depth_bins(x, obs_after)
        dep[f"cam{c + 1}"] = {"before": b0, "after": b1}
        log(f"    by cam{c + 1}'s z_tan - z_f ({np.nanpercentile(x, 5):+.0f} to {np.nanpercentile(x, 95):+.0f} mm): the "
            f"projected offset's bin medians RMS {b0['rms_noise_corrected']:.3f} px before the field, "
            f"{b1['rms_noise_corrected']:.3f} after (noise taken off); medians after "
            + ", ".join(f"{m:+.2f}" for m in b1["medians"]))
    trend = max(dep[k]["after"]["rms_noise_corrected"] for k in dep)
    out["depth"] = dep
    stop = trend > rms_pred
    log(f"  7. the image field's size (RMS of its offset about the mean) {rms_pred:.3f} px against the depth trend left "
        f"after it {trend:.3f} px: {'STOP (depth dominates)' if stop else 'image position dominates'}")

    # the judgement
    st = maps["epi12_by_cam2"]["stats"]
    real = st["rms_noise_corrected"] >= 0.3
    halves = maps["epi12_by_cam2"]["halves_corr"] >= 0.7
    agree = (out["truth_free"]["corr_with_tracks"] or 0) >= 0.7
    out["judged"] = {"pattern_rms_ge_0.3": real, "halves_corr_ge_0.7": halves, "estimates_agree_corr_ge_0.7": agree,
                     "stop_depth": stop}
    log(f"V0 judged: regional medians' RMS (epipolar offset, noise taken off) {st['rms_noise_corrected']:.3f} px "
        f"{'>=' if real else '<'} 0.3; halves correlation {maps['epi12_by_cam2']['halves_corr']:.2f} "
        f"{'>=' if halves else '<'} 0.7; the truth-free estimate's correlation (ML, 320 px cells) "
        f"{out['truth_free']['corr_with_tracks']:.2f} {'>=' if agree else '<'} 0.7; depth "
        f"{'dominates: STOP' if stop else 'does not dominate'}")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 4, figsize=(19, 10.5))
    ext = [0, 1280, 800, 0]
    for k, key in enumerate(("cam1_du", "cam1_dv", "cam2_du", "cam2_dv")):
        a = ax[0, k]
        im = a.imshow(np.array(maps[key]["median"]), extent=ext, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
        a.set_title(f"{key}: cell medians [px]", fontsize=9)
        plt.colorbar(im, ax=a, fraction=0.03)
    for k, key in enumerate(("epi12_by_cam2", "epi21_by_cam1")):
        a = ax[1, k]
        im = a.imshow(np.array(maps[key]["median"]), extent=ext, cmap="RdBu_r", vmin=-1.2, vmax=1.2)
        a.set_title(f"{key}: cell medians [px]", fontsize=9)
        plt.colorbar(im, ax=a, fraction=0.03)
    tf3 = out["truth_free"]["ml_320"]
    a = ax[1, 2]
    im = a.imshow(np.array(tf3["cells"], float), extent=ext, cmap="RdBu_r", vmin=-1.2, vmax=1.2)
    a.set_title("truth-free: ML centre of all candidate pairs, by 320 px cam2 cell", fontsize=9)
    plt.colorbar(im, ax=a, fraction=0.03)
    a = ax[1, 3]
    for key, mk in (("t1a_160", "x"), ("ml_160", "."), ("ml_320", "o")):
        t = out["truth_free"][key]
        a.scatter(np.array(t["tracks"], float).ravel(), np.array(t["cells"], float).ravel(), s=14, marker=mk,
                  label=f"{key}: corr {t['corr_with_tracks']:.2f}")
    a.plot([-1.5, 1.5], [-1.5, 1.5], "k-", lw=0.5)
    a.set_xlabel("tracks: epi12 cell median [px]")
    a.set_ylabel("truth-free centre [px]")
    a.legend(fontsize=7)
    a.set_title("per cam2 cell: the two estimates", fontsize=9)
    a = ax[2, 0]
    m1 = np.array(maps["epi12_by_cam2"]["half1"]).ravel()
    m2 = np.array(maps["epi12_by_cam2"]["half2"]).ravel()
    a.scatter(m1, m2, s=12)
    a.plot([-1.5, 1.5], [-1.5, 1.5], "k-", lw=0.5)
    a.set_xlabel("frames 0-125")
    a.set_ylabel("frames 130-250")
    a.set_title(f"epi12 cell medians, halves: corr {maps['epi12_by_cam2']['halves_corr']:.2f}", fontsize=9)
    for k, c in enumerate((0, 1)):
        a = ax[2, 1 + k]
        for nm, lab in (("before", "before the field"), ("after", "after")):
            b = dep[f"cam{c + 1}"][nm]
            a.errorbar(b["centres"], b["medians"], yerr=b["se"], marker="o", capsize=2, label=lab)
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel(f"cam{c + 1}: z_tan - z_f [mm]")
        a.set_ylabel("projected offset, bin median [px]")
        a.legend(fontsize=8)
    a = ax[2, 3]
    h, e = np.histogram(offs, bins=np.arange(-8, 8.001, 0.25))
    a.bar(0.5 * (e[1:] + e[:-1]), h, width=0.25, color="0.6")
    a.set_xlabel("signed offset of all candidate pairs [px]")
    a.set_title(f"T1-a's histogram, 51 frames: peak {pk_all:+.2f} px", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, "v0_pattern.png"), dpi=100)
    plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5))
    for c in (0, 1):
        g = fields["all"].dense[c]
        cov = fields["all"].covered[c]
        for k, nm in ((0, "du"), (1, "dv")):
            a = ax[c, k]
            img = np.where(cov, g[k], np.nan)
            im = a.imshow(img, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            a.set_title(f"cam{c + 1} {nm}: the fitted field (all frames) [px]", fontsize=9)
            plt.colorbar(im, ax=a, fraction=0.03)
        a = ax[c, 2]
        sel = fd.mask[:, c]
        a.scatter(fd.d[sel, c, 0], fd.d[sel, c, 1], s=0.2, c="k")
        a.set_xlim(0, 1280)
        a.set_ylim(800, 0)
        a.set_title(f"cam{c + 1}: the paired points ({int(sel.sum())})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, "v0_fields.png"), dpi=100)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
