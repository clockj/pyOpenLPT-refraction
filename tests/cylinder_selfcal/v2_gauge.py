"""V2: the correction's gauge.  (Added while running V2: v2_positions.py found that the fitted fields, which make the
images consistent, also move the triangulated points: with 4 cameras x and z by 8 and 12 um RMS against 4 and 6
with no correction.  The part of a field that a smooth deformation of the world explains is unseen by the tracks, and
the prior fills it with the smallest field, which is not the deformation-free one.)

    python v2_gauge.py

For the 4-camera set's runs with 2, 3 and 4 cameras (gathers cal4_r<n>, v2_fit.py's bending weight and noise), the
fit with a penalty on the correction's part along each point's Jacobian (selfcal.fit_field(gauge=g): how far it
moves the point's triangulation), g = 0 (as fitted so far), 1, 10; and with a penalty on the affine part of those moves only (gauge_affine = 1e7, 1e9, 1e11: the correction may not shift, turn, scale or shear the reconstruction).  Each judged as v2_fit.py judges (the observable
part against the truth, and against the truth plus the tracked points' deviation) and as v2_positions.py does (the
true images, moved by the error field, corrected by the field and triangulated: position RMS per axis against the
truth).  The fields of g = 1 and gauge_affine = 1e9 are saved as data/fields/cal4_r<n>_g1.npz, _ga.npz.
Writes v2_gauge_out.txt and .json here.
"""
from __future__ import annotations

import itertools
import os

import numpy as np

import selfcal as sc
from v2_fit import nullspace_project
from v2_positions import triangulate

GAUGES = [("point", 0.0), ("point", 1.0), ("point", 10.0), ("affine", 1e7), ("affine", 1e9), ("affine", 1e11)]


def main():
    log = sc.Log("v2_gauge")
    import synth_sc
    closed = synth_sc.cal4_fields()[0]
    case_dir = os.path.join(sc.DATA, "cal4")
    arr = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"), mmap_mode="r")  # noqa: E731
    t = 20
    X = np.asarray(arr("X")[t])
    layer = np.asarray(arr("layer"))
    cams_all = [os.path.join(case_dir, "camFile", f"cam{c}.txt") for c in (1, 2, 3, 4)]
    out = {}
    for n in (2, 3, 4):
        rec_fit = sc.load_json(os.path.join(sc.HERE, f"v2_fit_cal4_r{n}.json"))
        lam, noise = rec_fit["fit"]["lam"], tuple(rec_fit["fit"]["noise"])
        fd = sc.fitdata(sc.load_gather(f"cal4_r{n}"), sc.R_PAIR, 2)
        cams = cams_all[:n]
        # the truth at frame 20 (as v2_fit.py) and the images seen by all n cameras (as v2_positions.py)
        uv = np.stack([np.asarray(arr(f"uv_cal4_c{c}")[t]) for c in range(n)], axis=1)
        E = np.stack([np.asarray(arr(f"err_cal4_c{c}")[t]) for c in range(n)], axis=1)
        on = np.isfinite(uv[..., 0]) & (uv[..., 0] >= 0) & (uv[..., 0] < 1280) & (uv[..., 1] >= 0) & (uv[..., 1] < 800)
        sel = (layer == 0) & sc.in_volume(X, sc.view_volume()) & (on.sum(axis=1) >= 2)
        Xs, uvs, Es, ons = X[sel], uv[sel], E[sel], on[sel]
        J = np.stack([sc.project_jacobian(cams[c], Xs)[1] for c in range(n)], axis=1)
        allc = np.all(ons, axis=1)
        # the detections' deviation (as v2_fit.py), for the judgement against truth + deviation
        r_true = np.zeros_like(fd.r)
        for c in range(n):
            ok = fd.mask[:, c]
            r_true[ok, c] = closed[c](fd.d[ok, c, 0], fd.d[ok, c, 1])
        dev = sc.Field(n)
        sc.fit_field(dev, sc.FitData(fd.mask, fd.d, fd.r - r_true, fd.J, fd.L, fd.frame, fd.track), lam, noise,
                     gauge=0.0)
        dev.make_dense([fd.d[fd.mask[:, c], c] for c in range(n)])
        B = np.zeros_like(Es)
        for c in range(n):
            B[ons[:, c], c] = dev.eval_dense(c, uvs[ons[:, c], c])
        rec_n = {}
        for mode, g in GAUGES:
            f = sc.Field(n)
            if mode == "point":
                r = sc.fit_field(f, fd, lam, noise, gauge=g, gauge_affine=0.0)
            else:
                r = sc.fit_field(f, fd, lam, noise, gauge=0.0, gauge_affine=g)
            key = f"{mode} {g:g}"
            f.make_dense([fd.d[fd.mask[:, c], c] for c in range(n)])
            if key in ("point 1", "affine 1e+09"):
                tag = "g1" if mode == "point" else "ga"
                f.meta = {"name": f"cal4_r{n}_{tag}", "lam": lam, "noise": list(noise), "gauge": key, "fit": r}
                f.save(os.path.join(sc.DATA, "fields", f"cal4_r{n}_{tag}.npz"))
            C = np.zeros_like(Es)
            for c in range(n):
                C[ons[:, c], c] = f.eval_dense(c, uvs[ons[:, c], c])
            pt, pc, ptb = [], [], []
            for k in range(2, n + 1):
                for S in itertools.combinations(range(n), k):
                    m = np.all(ons[:, list(S)], axis=1) & (ons.sum(axis=1) == k)
                    if not m.any():
                        continue
                    Js = J[m][:, list(S)].reshape(int(m.sum()), 2 * k, 3)
                    gd = np.all(np.isfinite(Js.reshape(len(Js), -1)), axis=1)
                    sh = lambda A: A[m][:, list(S)].reshape(int(m.sum()), 2 * k)[gd]  # noqa: E731
                    pt.append(nullspace_project(Js[gd], sh(Es)).ravel())
                    pc.append(nullspace_project(Js[gd], sh(C)).ravel())
                    ptb.append(nullspace_project(Js[gd], sh(Es + B)).ravel())
            pt, pc, ptb = map(np.concatenate, (pt, pc, ptb))
            # positions: the images seen by all n cameras, corrected and triangulated
            u2 = uvs[allc] - C[allc]
            Xe, rr = triangulate(cams, u2, Xs[allc])
            pos = (1e3 * np.sqrt(np.mean((Xe - Xs[allc]) ** 2, axis=0))).tolist()
            rec_n[key] = {"obs_error_vs_truth": float(np.sqrt(np.mean((pt - pc) ** 2))),
                             "obs_error_vs_truth_plus_dev": float(np.sqrt(np.mean((ptb - pc) ** 2))),
                             "pos_rms_um": pos, "reproj_rms_px": rr,
                             "field_rms_px": [float(np.sqrt(np.mean(f.dense[c][:, f.covered[c]] ** 2)))
                                              for c in range(n)]}
            q = rec_n[key]
            log(f"  {n} cameras, gauge {key}: observable error {q['obs_error_vs_truth']:.3f} px against the truth, "
                f"{q['obs_error_vs_truth_plus_dev']:.3f} against truth + deviation; positions of {int(allc.sum())} "
                f"particles corrected and triangulated: RMS x/y/z {pos[0]:.1f} / {pos[1]:.1f} / {pos[2]:.1f} um "
                f"(reprojection {rr:.3f} px); the field's RMS per camera "
                f"{np.round(q['field_rms_px'], 3).tolist()} px")
        Xe0, rr0 = triangulate(cams, uvs[allc], Xs[allc])
        rec_n["uncorrected"] = {"pos_rms_um": (1e3 * np.sqrt(np.mean((Xe0 - Xs[allc]) ** 2, axis=0))).tolist(),
                                "reproj_rms_px": rr0}
        p0 = rec_n["uncorrected"]["pos_rms_um"]
        log(f"  {n} cameras, no correction: RMS x/y/z {p0[0]:.1f} / {p0[1]:.1f} / {p0[2]:.1f} um (reprojection {rr0:.3f} px)")
        out[f"{n}_cameras"] = rec_n
    log.save(out)


if __name__ == "__main__":
    main()
