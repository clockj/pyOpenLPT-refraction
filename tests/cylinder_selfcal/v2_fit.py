"""V2 (b) (plan, section 3), with 2, 3 and 4 cameras: the field fitted from a synthetic run's own tracks, against the
true error field.

    python v2_fit.py NAME --gather G --case CASE --set SET [--err ERR] [--truth-frame 20] [--tf]

The run's tracks are gathered first (gather.py G: every frame 0-39, the long tracks of >= 10 points, detections within
3 px; with --tf also S2's candidate pairs).  The fit is V0's (selfcal.fit_with_noise: the projected least squares with
the bending weight by held-out loss, frames t even against odd, and the noise model by the residuals), on every
point seen by at least two of the run's cameras.  With --tf (two cameras) also the truth-free fit of V0b
(selfcal.mixture_fit on every candidate pair within +-3 px, warm-started from the tracks' field).
The truth (synth_sc.py): data/<CASE>/arrays/<ERR>_c<k>.npy, the error each image was moved by.  Judged on the central
particles inside the view volume at one frame (TRUTH_FRAME), each with the cameras that see it:
  - the observable part: the stacked true errors projected on the complement of the point's Jacobian (what the
    cameras can see: 2 cameras 1 number, 3 cameras 3, 4 cameras 5), against the same projection of the fitted field;
    RMS of the truth, of the fit's error, the fit's slope on the truth (1 = no shrinkage);
  - the full field per camera and component: RMS of the truth, of the fit's error (with 2 cameras the part along the
    curves is unseen and stays at 0; with 3-4 cameras a smooth 3-D deformation of the world stays unseen);
  - the epipolar consistency it buys: per camera pair, the RMS offset across the curves of the true images with and
    without the field (the null-direction value of the pair's 4 numbers).
Writes v2_fit_<NAME>_out.txt, .json and data/fields/<NAME>.npz (and <NAME>_tf.npz).
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

import selfcal as sc

LAMS = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]


def nullspace_project(J, v):
    """Per point: the projection of v (n, 2s) on the complement of J's columns (n, 2s, 3): (n, 2s - 3)."""
    U, _S, _V = np.linalg.svd(J, full_matrices=True)
    Nb = U[:, :, 3:]
    return np.einsum("nik,ni->nk", Nb, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--gather", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--err", default=None, help="the true error arrays' stem (default: err_<the set's variant>)")
    ap.add_argument("--uv", default=None, help="the images' positions variant (default: the set's variant; for a warped "
                                                "set its variant_effective)")
    ap.add_argument("--closed", default=None, choices=["cal4", "gallm"],
                    help="whose closed-form error fields to use for the detections' deviation (default: the case's; in "
                         "a second pass, minus the first pass's field)")
    ap.add_argument("--truth-frame", type=int, default=20)
    ap.add_argument("--tf", action="store_true")
    a = ap.parse_args()
    log = sc.Log(f"v2_fit_{a.name}")
    case_dir = os.path.join(sc.DATA, a.case)
    spec = sc.load_json(os.path.join(case_dir, "sets.json"))[a.set]
    uv_var = a.uv or spec.get("variant_effective", spec["variant"])
    err = a.err or f"err_{uv_var}"
    recs = sc.load_gather(a.gather)
    info = sc.load_json(os.path.join(sc.DATA, "gather", a.gather, f"{a.gather}.json"))
    n_cam = len(info["cams"])
    cams = info["cams"]
    fd = sc.fitdata(recs, sc.R_PAIR, min_cams=2)
    seen = fd.mask.sum(axis=1)
    log(f"V2 fit {a.name}: gather {a.gather} ({len(recs)} frames, {n_cam} cameras), case {a.case}, set {a.set}, truth "
        f"{err}; {len(fd)} track points seen by >= 2 cameras (" + ", ".join(f"{k} cameras: {int((seen == k).sum())}"
                                                                          for k in range(2, n_cam + 1)) + ")")
    out = {"name": a.name, "gather": a.gather, "case": a.case, "set": a.set, "cameras": n_cam, "points": len(fd),
           "points_by_cameras": {str(k): int((seen == k).sum()) for k in range(2, n_cam + 1)}}
    even = fd.frame % 2 == 0
    field = sc.Field(n_cam)
    rec, lam, noise = sc.fit_with_noise(field, fd.subset(even), fd.subset(~even), LAMS, log)
    f_all = sc.Field(n_cam)
    rec_all = sc.fit_field(f_all, fd, lam, noise, log=log)
    f_all.make_dense([fd.d[fd.mask[:, c], c] for c in range(n_cam)])
    f_all.meta = {"name": a.name, "gather": a.gather, "lam": lam, "noise": noise, "fit": rec_all, "kind": "tracks"}
    os.makedirs(os.path.join(sc.DATA, "fields"), exist_ok=True)
    f_all.save(os.path.join(sc.DATA, "fields", f"{a.name}.npz"))
    out["fit"] = {"selection": rec, "lam": lam, "noise": list(noise), "all": rec_all}
    fields = {"tracks": f_all}

    if a.tf:
        from v0b_truthfree import pairs_table
        T = pairs_table(recs)
        o = T[:, 10]
        sel = np.abs(o) <= 3.0
        Nf, _i = sc.null_grid(cams, sc.view_volume())
        Nv = Nf(T[sel, 7:10])
        y = o[sel] * Nv[:, 3]
        g = sc.Field(2)
        r_tf = sc.mixture_fit(g, T[sel, 1:3], T[sel, 4:6], Nv, y, T[sel, 3], T[sel, 6], lam, noise, 3.0,
                              init_theta=f_all.theta, log=log)
        g.make_dense([T[sel, 1:3], T[sel, 4:6]])
        g.meta = {"name": a.name, "kind": "truth-free (EM, warm start from the tracks' field)", "mixture": r_tf,
                  "lam": lam, "noise": list(noise)}
        g.save(os.path.join(sc.DATA, "fields", f"{a.name}_tf.npz"))
        fields["truth_free"] = g
        out["truth_free_fit"] = {k: r_tf[k] for k in ("pairs", "true_pairs", "true_share", "kappa", "loglik",
                                                      "iterations")}

    # the detections' own smooth bias (v2_diag.py found one, 0.15 px on the 4-camera set: what the detector makes of
    # each image's shape and neighbours, which a correction rightly takes up).  With the error fields in closed form
    # (synth_sc.cal4_fields; for Gallm the study's error_fields), the same estimator on the detections' error alone
    # (detection - projection - the true error at the detection), so that the fit can also be judged against the truth
    # plus that bias, i.e. against what makes the detections consistent.
    bias = None
    closed = None
    kind = a.closed or a.case
    if kind == "cal4":
        import synth_sc
        closed = synth_sc.cal4_fields()[0]
    elif kind == "gallm":
        import synth_study
        closed = synth_study.error_fields(2)[0]
    pass1 = spec.get("corrected_by") or spec.get("warped_by")      # a second pass: the images were corrected by f1
    f1 = sc.Field.load(pass1) if pass1 else None
    if closed is not None:
        r_true = np.zeros_like(fd.r)
        for c in range(n_cam):
            ok = fd.mask[:, c]
            q = fd.d[ok, c]
            if f1 is None:
                r_true[ok, c] = closed[c](q[:, 0], q[:, 1])
            else:             # the error left at q: e(d) - c1(d), d = q + c1(d) (two fixed-point steps)
                d0 = q + f1.eval_dense(c, q)
                d0 = q + f1.eval_dense(c, d0)
                r_true[ok, c] = closed[c](d0[:, 0], d0[:, 1]) - f1.eval_dense(c, d0)
        fb = sc.FitData(fd.mask, fd.d, fd.r - r_true, fd.J, fd.L, fd.frame, fd.track)
        bias = sc.Field(n_cam)
        sc.fit_field(bias, fb, lam, noise)
        bias.make_dense([fd.d[fd.mask[:, c], c] for c in range(n_cam)])

    # the truth: central particles inside the view volume at TRUTH_FRAME
    arr = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"), mmap_mode="r")  # noqa: E731
    t = a.truth_frame
    X = np.asarray(arr("X")[t])
    layer = np.asarray(arr("layer"))
    vol = sc.view_volume()
    uv = np.stack([np.asarray(arr(f"uv_{uv_var}_c{c}")[t]) for c in range(n_cam)], axis=1)     # the images' positions
    E = np.stack([np.asarray(arr(f"{err}_c{c}")[t]) for c in range(n_cam)], axis=1)             # their errors
    on = np.isfinite(uv[..., 0]) & (uv[..., 0] >= 0) & (uv[..., 0] < 1280) & (uv[..., 1] >= 0) & (uv[..., 1] < 800)
    sel = (layer == 0) & sc.in_volume(X, vol) & (on.sum(axis=1) >= 2)
    X, uv, E, on = X[sel], uv[sel], E[sel], on[sel]
    J = np.full((len(X), n_cam, 2, 3), np.nan)
    for c in range(n_cam):
        J[:, c] = sc.project_jacobian(cams[c], X)[1]
    log(f"  the truth at frame {t}: {len(X)} central particles in the view volume seen by >= 2 cameras")
    out["truth"] = {"frame": t, "particles": int(len(X))}
    Bv = np.zeros_like(E)
    if bias is not None:
        for c in range(n_cam):
            ok = on[:, c]
            Bv[ok, c] = bias.eval_dense(c, uv[ok, c])
    for fname, f in fields.items():
        C = np.zeros_like(E)
        for c in range(n_cam):
            ok = on[:, c]
            C[ok, c] = f.eval_dense(c, uv[ok, c])
        res = {"observable": {}, "full": {}, "pairs": {}}
        if bias is not None:      # against the truth plus the detections' bias
            pt_b, pe_b, pb = [], [], []
            for k in range(2, n_cam + 1):
                for S in itertools.combinations(range(n_cam), k):
                    m = np.all(on[:, list(S)], axis=1) & (on.sum(axis=1) == k)
                    if not m.any():
                        continue
                    Js = J[m][:, list(S)].reshape(int(m.sum()), 2 * k, 3)
                    good = np.all(np.isfinite(Js.reshape(len(Js), -1)), axis=1)
                    TB = (E + Bv)[m][:, list(S)].reshape(int(m.sum()), 2 * k)[good]
                    a1 = nullspace_project(Js[good], TB)
                    a2 = nullspace_project(Js[good], C[m][:, list(S)].reshape(int(m.sum()), 2 * k)[good])
                    a3 = nullspace_project(Js[good], Bv[m][:, list(S)].reshape(int(m.sum()), 2 * k)[good])
                    pt_b.append(a1.ravel())
                    pe_b.append((a1 - a2).ravel())
                    pb.append(a3.ravel())
            pt_b, pe_b, pb = map(np.concatenate, (pt_b, pe_b, pb))
            res["vs_truth_plus_bias"] = {"bias_rms": float(np.sqrt(np.mean(pb ** 2))),
                                         "rms_true": float(np.sqrt(np.mean(pt_b ** 2))),
                                         "rms_error": float(np.sqrt(np.mean(pe_b ** 2))),
                                         "slope": float(np.sum(pt_b * (pt_b - pe_b)) / np.sum(pt_b * pt_b))}
            v = res["vs_truth_plus_bias"]
            log(f"  {fname} field against the truth plus the detections' bias (observable RMS {v['bias_rms']:.3f} px): "
                f"{v['rms_true']:.3f} px, the fit's error {v['rms_error']:.3f} px (slope {v['slope']:.2f})")
        # the observable part, per camera set (all the cameras that see the particle)
        tot_t, tot_e, xs, ys = [], [], [], []
        for k in range(2, n_cam + 1):
            for S in itertools.combinations(range(n_cam), k):
                m = np.all(on[:, list(S)], axis=1) & (on.sum(axis=1) == k)
                if not m.any():
                    continue
                Js = J[m][:, list(S)].reshape(int(m.sum()), 2 * k, 3)
                good = np.all(np.isfinite(Js.reshape(len(Js), -1)), axis=1)
                pe = nullspace_project(Js[good], E[m][:, list(S)].reshape(int(m.sum()), 2 * k)[good])
                pc = nullspace_project(Js[good], C[m][:, list(S)].reshape(int(m.sum()), 2 * k)[good])
                tot_t.append(pe.ravel())
                tot_e.append((pe - pc).ravel())
                xs.append(pe.ravel())
                ys.append(pc.ravel())
        pt, pe_, x_, y_ = map(np.concatenate, (tot_t, tot_e, xs, ys))
        slope = float(np.sum(x_ * y_) / np.sum(x_ * x_))
        res["observable"] = {"rms_true": float(np.sqrt(np.mean(pt ** 2))), "rms_error": float(np.sqrt(np.mean(pe_ ** 2))),
                             "slope": slope, "explained": float(1 - np.mean(pe_ ** 2) / np.mean(pt ** 2))}
        log(f"  {fname} field: the observable part: truth RMS {res['observable']['rms_true']:.3f} px, the fit's error "
            f"{res['observable']['rms_error']:.3f} px ({100 * res['observable']['explained']:.0f} % explained); the fit "
            f"= {slope:.2f} x the truth")
        for c in range(n_cam):
            ok = on[:, c]
            rr = {}
            for comp, nm in ((0, "du"), (1, "dv")):
                rr[nm] = {"rms_true": float(np.sqrt(np.mean(E[ok, c, comp] ** 2))),
                          "rms_fit": float(np.sqrt(np.mean(C[ok, c, comp] ** 2))),
                          "rms_error": float(np.sqrt(np.mean((E[ok, c, comp] - C[ok, c, comp]) ** 2))),
                          "corr": float(np.corrcoef(E[ok, c, comp], C[ok, c, comp])[0, 1])
                          if np.std(C[ok, c, comp]) > 0 else None}
            res["full"][f"cam{c + 1}"] = rr
            log(f"    cam{c + 1}: du true {rr['du']['rms_true']:.3f}, fit {rr['du']['rms_fit']:.3f}, error "
                f"{rr['du']['rms_error']:.3f} (corr {rr['du']['corr'] if rr['du']['corr'] is None else round(rr['du']['corr'], 2)}); "
                f"dv true {rr['dv']['rms_true']:.3f}, fit {rr['dv']['rms_fit']:.3f}, error {rr['dv']['rms_error']:.3f} "
                f"(corr {rr['dv']['corr'] if rr['dv']['corr'] is None else round(rr['dv']['corr'], 2)}) px")
        # per camera pair: the offset across the curves of the true images (the pair's null direction)
        for c1, c2 in itertools.combinations(range(n_cam), 2):
            m = on[:, c1] & on[:, c2]
            Js = J[m][:, [c1, c2]].reshape(int(m.sum()), 4, 3)
            good = np.all(np.isfinite(Js.reshape(len(Js), -1)), axis=1)
            pe = nullspace_project(Js[good], E[m][:, [c1, c2]].reshape(int(m.sum()), 4)[good])[:, 0]
            pc = nullspace_project(Js[good], C[m][:, [c1, c2]].reshape(int(m.sum()), 4)[good])[:, 0]
            res["pairs"][f"{c1 + 1}{c2 + 1}"] = {"n": int(good.sum()), "rms_before": float(np.sqrt(np.mean(pe ** 2))),
                                               "rms_after": float(np.sqrt(np.mean((pe - pc) ** 2)))}
        log("    camera pairs, RMS offset across the curves before -> after: "
            + "; ".join(f"cams {k[0]}-{k[1]}: {v['rms_before']:.3f} -> {v['rms_after']:.3f} px"
                        for k, v in res["pairs"].items()))
        out[fname] = res
    log.save(out)


if __name__ == "__main__":
    main()
