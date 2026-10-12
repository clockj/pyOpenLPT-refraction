"""G0: the GUI's field VSC estimator (modules/vsc/field_vsc.py) against the self-calibration's, on the same data.

    python g0_port.py

The self-calibration's gathered T1 frames (tests/cylinder_selfcal/data/gather/t1_m15: every 5th frame of the 15 px
run, read only), turned into fit data as its selfcal.fitdata does (pairs within 3 px, two cameras):
  a. the module's fit with V0's protocol (bending weights 1e4 ... 1e9 chosen on frames 0, 10, ... against 5, 15, ...;
     two rounds of the noise model; then every frame) against V0's field t1_m15_all: its coefficients, the bending
     weight and noise model, and the field at the data. Pass: the same weight, coefficients equal to 1e-6 px;
  b. the module's own grid (1e2 ... 1e7): the weight it picks and how far its field departs from (a) at the data;
  c. the correction file's grid (8 px nodes): the field near the data against V0's dense map, and the fill beyond
     (harmonic, the module's; nearest covered value, V0's).
Writes g0_port_out.txt and .json here.
"""
from __future__ import annotations

import glob
import os

import numpy as np

import vcf_common as vc
from modules.vsc import field_vsc as fv

V0_LAMS = (1e4, 1e5, 1e6, 1e7, 1e8, 1e9)


def load_fitdata(name: str, gate: float = 3.0) -> fv.FitData:
    parts = []
    for f in sorted(glob.glob(os.path.join(vc.SELFCAL, "data", "gather", name, "frame_*.npz"))):
        g = np.load(f, allow_pickle=False)
        n_cam, n = int(g["n_cam"]), len(g["X"])
        if n == 0:
            continue
        mask = (g["pair"] >= 0) & (g["pair_dist"] <= gate)
        d = np.full((n, n_cam, 2), np.nan)
        L = np.zeros((n, n_cam, 2))
        for c in range(n_cam):
            ok = mask[:, c]
            dc = g[f"det{c}"]
            d[ok, c] = dc[g["pair"][ok, c], :2]
            L[ok, c] = dc[g["pair"][ok, c], 2:4]
        keep = (mask.sum(axis=1) >= 2) & np.all(np.isfinite(g["J"].reshape(n, -1)), axis=1)
        parts.append(fv.FitData(mask[keep], d[keep], (d - g["P"])[keep], g["J"][keep], L[keep],
                                np.full(int(keep.sum()), int(g["t"]))))
    return fv.FitData.concat(parts)


def field_at_data(field: fv.SplineField, fd: fv.FitData) -> np.ndarray:
    out = []
    for c in range(field.n_cam):
        m = fd.mask[:, c]
        out.append(field.eval_points(c, fd.d[m, c]))
    return np.vstack(out)


def main():
    log = vc.Log("g0_port")
    fd = load_fitdata("t1_m15")
    log(f"t1_m15 (the self-calibration's gather): {len(fd)} points in both cameras, frames {fd.frame.min()}-"
        f"{fd.frame.max()}")
    z = np.load(os.path.join(vc.SELFCAL, "data", "fields", "t1_m15_all.npz"), allow_pickle=False)
    import json
    meta = json.loads(str(z["meta"]))
    ref = fv.SplineField([(1280, 800), (1280, 800)])
    ref.theta = np.asarray(z["theta"], float)
    sizes = [(1280, 800), (1280, 800)]
    fit_m = fd.frame % 10 == 0
    out = {"points": len(fd), "v0": {"lam": meta.get("lam"), "noise": meta.get("noise")}}

    # a. V0's protocol
    a, rec_a = fv.fit_with_noise(sizes, fd.subset(fit_m), fd.subset(~fit_m), fd, log, lams=V0_LAMS)
    dth = float(np.max(np.abs(a.theta - ref.theta)))
    da = np.hypot(*(field_at_data(a, fd) - field_at_data(ref, fd)).T)
    same_lam = rec_a["lam"] == meta.get("lam")
    noise_diff = float(np.max(np.abs(np.array(rec_a["noise"]) - np.array(meta.get("noise")))))
    ok_a = same_lam and dth <= 1e-6 and noise_diff <= 1e-9
    log(f"a. V0's protocol: bending weight {rec_a['lam']:g} (V0 {meta.get('lam'):g}); noise {rec_a['noise']} (V0 "
        f"{meta.get('noise')}, largest difference {noise_diff:.1e}); coefficients differ by <= {dth:.2e}; the field at "
        f"the data by <= {da.max():.2e} px -> {'PASS' if ok_a else 'FAIL'}")
    out["a"] = {"lam": rec_a["lam"], "noise": rec_a["noise"], "theta_max_diff": dth, "field_max_diff_px": float(da.max()),
                "pass": bool(ok_a)}

    # b. the module's grid
    b, rec_b = fv.fit_with_noise(sizes, fd.subset(fit_m), fd.subset(~fit_m), fd, log)
    db = np.hypot(*(field_at_data(b, fd) - field_at_data(a, fd)).T)
    mag = np.hypot(*field_at_data(a, fd).T)
    table = {t["lam"]: round(t["loss_after"], 5) for t in rec_b["rounds"][-1]["lambda_table"]}
    log(f"b. the module's grid {fv.LAMS}: bending weight {rec_b['lam']:g} (held-out loss per weight {table}); its field "
        f"departs from (a)'s at the data by {np.sqrt(np.mean(db ** 2)):.3f} px RMS, {db.max():.3f} max (the field there: "
        f"{np.sqrt(np.mean(mag ** 2)):.3f} px RMS)")
    out["b"] = {"lam": rec_b["lam"], "noise": rec_b["noise"], "lambda_table": table,
                "rms_diff_px": float(np.sqrt(np.mean(db ** 2))), "max_diff_px": float(db.max())}

    # c. the correction's grid against V0's dense map
    dense = np.asarray(z["dense"], float)
    covered = np.asarray(z["covered"], bool)
    out["c"] = {}
    for c in range(2):
        data_uv = fd.d[fd.mask[:, c], c]
        corr, cov = fv.node_correction(a, c, data_uv)
        H, W = dense.shape[2:]
        yy, xx = np.mgrid[0:H, 0:W]
        uv = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
        dif = np.hypot(*(corr.at(uv) - dense[c].reshape(2, -1).T).T).reshape(H, W)
        cv_ = covered[c]
        far = ~cv_
        log(f"c. cam{c + 1}: {cov['covered_nodes']} of {cov['nodes']} nodes near the data; the correction against V0's "
            f"dense map: near the data {np.sqrt(np.mean(dif[cv_] ** 2)):.4f} px RMS ({dif[cv_].max():.3f} max); beyond "
            f"(the fills differ) {np.sqrt(np.mean(dif[far] ** 2)) if far.any() else 0:.3f} px RMS "
            f"({dif[far].max() if far.any() else 0:.3f} max); slope bound {corr.slope_bound():.4f}")
        out["c"][f"cam{c + 1}"] = {"covered_rms_px": float(np.sqrt(np.mean(dif[cv_] ** 2))),
                                   "covered_max_px": float(dif[cv_].max()),
                                   "beyond_rms_px": float(np.sqrt(np.mean(dif[far] ** 2))) if far.any() else 0.0,
                                   "beyond_max_px": float(dif[far].max()) if far.any() else 0.0,
                                   "slope_bound": corr.slope_bound(), **cov}
    out["pass"] = bool(ok_a)
    log(f"G0: {'PASS' if ok_a else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
