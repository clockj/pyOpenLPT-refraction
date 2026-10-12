"""V2's estimator check: which noise model and robust loss recover a known field best?  (Added while running V2: on
the 4-camera synthetic set the fitted noise model collapsed to s_f = 0.01 px, the Huber fit down-weighted 39 % of the
points and invented ~1 px fields in the directions the cameras barely see.)

    python v2_diag.py GATHER CASE SET [--frames-truth 20]

On one gathered synthetic run, variants of the fit, each judged against the truth as v2_fit.py judges (the observable
part's error and slope, the full field's error per camera):
  ml_huber     the noise model by the residuals (as fitted so far), Huber 1.5
  floor_huber  the same with s_f held at >= 0.15 px
  floor_cauchy s_f >= 0.15 px, Cauchy weights 1 / (1 + (t / 2)^2) (redescending: wrong pairings and blends count
               little at any size)
  unit_huber   one noise for every axis (0.3 px), Huber 1.5
Writes v2_diag_<GATHER>_out.txt and .json.
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

import selfcal as sc
from v2_fit import nullspace_project

LAMS = [1e3, 1e4, 1e5, 1e6, 1e7]


def judge(field, case_dir, spec, cams, t, n_cam):
    arr = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"), mmap_mode="r")  # noqa: E731
    X = np.asarray(arr("X")[t])
    layer = np.asarray(arr("layer"))
    uv = np.stack([np.asarray(arr(f"uv_{spec['variant']}_c{c}")[t]) for c in range(n_cam)], axis=1)
    E = np.stack([np.asarray(arr(f"err_{spec['variant']}_c{c}")[t]) for c in range(n_cam)], axis=1)
    on = np.isfinite(uv[..., 0]) & (uv[..., 0] >= 0) & (uv[..., 0] < 1280) & (uv[..., 1] >= 0) & (uv[..., 1] < 800)
    sel = (layer == 0) & sc.in_volume(X, sc.view_volume()) & (on.sum(axis=1) >= 2)
    X, uv, E, on = X[sel], uv[sel], E[sel], on[sel]
    J = np.stack([sc.project_jacobian(cams[c], X)[1] for c in range(n_cam)], axis=1)
    C = np.zeros_like(E)
    for c in range(n_cam):
        C[on[:, c], c] = field.eval_dense(c, uv[on[:, c], c])
    pt, pe = [], []
    for k in range(2, n_cam + 1):
        for S in itertools.combinations(range(n_cam), k):
            m = np.all(on[:, list(S)], axis=1) & (on.sum(axis=1) == k)
            if not m.any():
                continue
            Js = J[m][:, list(S)].reshape(int(m.sum()), 2 * k, 3)
            g = np.all(np.isfinite(Js.reshape(len(Js), -1)), axis=1)
            a = nullspace_project(Js[g], E[m][:, list(S)].reshape(int(m.sum()), 2 * k)[g])
            b = nullspace_project(Js[g], C[m][:, list(S)].reshape(int(m.sum()), 2 * k)[g])
            pt.append(a.ravel())
            pe.append(b.ravel())
    pt, pc = np.concatenate(pt), np.concatenate(pe)
    full = [float(np.sqrt(np.mean((E[on[:, c], c] - C[on[:, c], c]) ** 2))) for c in range(n_cam)]
    return {"obs_true": float(np.sqrt(np.mean(pt ** 2))), "obs_error": float(np.sqrt(np.mean((pt - pc) ** 2))),
            "slope": float(np.sum(pt * pc) / np.sum(pt * pt)), "full_error": full,
            "full_true": [float(np.sqrt(np.mean(E[on[:, c], c] ** 2))) for c in range(n_cam)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gather")
    ap.add_argument("case")
    ap.add_argument("set")
    ap.add_argument("--frame-truth", type=int, default=20)
    a = ap.parse_args()
    log = sc.Log(f"v2_diag_{a.gather}")
    case_dir = os.path.join(sc.DATA, a.case)
    spec = sc.load_json(os.path.join(case_dir, "sets.json"))[a.set]
    info = sc.load_json(os.path.join(sc.DATA, "gather", a.gather, f"{a.gather}.json"))
    cams = info["cams"]
    n_cam = len(cams)
    fd = sc.fitdata(sc.load_gather(a.gather), sc.R_PAIR, 2)
    even = fd.frame % 2 == 0
    log(f"V2 estimator check on {a.gather}: {len(fd)} points, {n_cam} cameras")
    out = {}
    orig_huber = sc.fit_field.__defaults__
    for name in ("ml_huber", "floor_huber", "floor_cauchy", "unit_huber"):
        floor = 0.15 if name.startswith("floor") else 0.0
        cauchy = name.endswith("cauchy")
        unit = name.startswith("unit")
        noise = (0.3, 0.0) if unit else (0.3, 0.05)
        for rnd in range(2):
            lam, _t = sc.choose_lambda(sc.Field(n_cam), fd.subset(even), fd.subset(~even), LAMS, noise, None,
                                       robust="cauchy" if cauchy else "huber")
            if unit:
                break
            f = sc.Field(n_cam)
            sc.fit_field(f, fd.subset(even), lam, noise, robust="cauchy" if cauchy else "huber")
            nz, _b = sc.estimate_noise(f, fd.subset(even))
            noise = (max(nz[0], floor), nz[1])
        f = sc.Field(n_cam)
        r = sc.fit_field(f, fd, lam, noise, robust="cauchy" if cauchy else "huber")
        f.make_dense([fd.d[fd.mask[:, c], c] for c in range(n_cam)])
        j = judge(f, case_dir, spec, cams, a.frame_truth, n_cam)
        out[name] = {"lam": lam, "noise": list(noise), "fit": r, "judge": j}
        log(f"  {name}: lam {lam:g}, noise ({noise[0]:.3f}, {noise[1]:.4f}), down-weighted "
            f"{100 * r['downweighted']:.0f} %: observable truth {j['obs_true']:.3f}, error {j['obs_error']:.3f} px "
            f"(slope {j['slope']:.2f}); full field error per camera {np.round(j['full_error'], 3).tolist()} (truth "
            f"{np.round(j['full_true'], 3).tolist()})")
    del orig_huber
    # the oracle (cal4 only, whose fields are closed forms): the same estimator fitted to the true errors at the track
    # points' detections instead of the residuals, i.e. exact detections; its error is the representation's and the
    # regularisation's, the rest of the data fit's error is the data's
    if a.case == "cal4":
        import synth_sc
        fields, _i = synth_sc.cal4_fields()
        r_true = np.zeros_like(fd.r)
        for c in range(n_cam):
            ok = fd.mask[:, c]
            r_true[ok, c] = fields[c](fd.d[ok, c, 0], fd.d[ok, c, 1])
        fo = sc.FitData(fd.mask, fd.d, r_true, fd.J, fd.L, fd.frame, fd.track)
        for name, noise, robust in (("oracle_floor_cauchy", out["floor_cauchy"]["noise"], "cauchy"),
                                    ("oracle_unit", (0.3, 0.0), "huber")):
            f = sc.Field(n_cam)
            r = sc.fit_field(f, fo, out["floor_cauchy"]["lam"], tuple(noise), robust=robust)
            f.make_dense([fd.d[fd.mask[:, c], c] for c in range(n_cam)])
            j = judge(f, case_dir, spec, cams, a.frame_truth, n_cam)
            out[name] = {"noise": list(noise), "fit": r, "judge": j}
            log(f"  {name}: observable truth {j['obs_true']:.3f}, error {j['obs_error']:.3f} px (slope {j['slope']:.2f}); "
                f"full field error per camera {np.round(j['full_error'], 3).tolist()}")
        # the data's own deviation from the truth at the track points: detection - (projection + true error) is the
        # detection's error plus a move of X; its projected part, per number of cameras
        dev = fd.r - r_true
        for k in range(2, n_cam + 1):
            m = fd.mask.sum(axis=1) == k
            if not m.any():
                continue
            sub = fd.subset(m)
            z = []
            for S in sorted({tuple(np.flatnonzero(mm)) for mm in sub.mask}):
                mm = np.all(sub.mask[:, list(S)], axis=1)
                Js = sub.J[mm][:, list(S)].reshape(int(mm.sum()), 2 * k, 3)
                z.append(nullspace_project(Js, dev[m][mm][:, list(S)].reshape(int(mm.sum()), 2 * k)).ravel())
            z = np.concatenate(z)
            mad = 1.4826 * np.median(np.abs(z - np.median(z)))
            log(f"  the detections' own error at points seen by {k} cameras, projected: median {np.median(z):+.3f}, "
                f"robust SD {mad:.3f}, RMS {np.sqrt(np.mean(z ** 2)):.3f} px, beyond 1 px {100 * np.mean(np.abs(z) > 1):.1f} %")
            out[f"detection_error_{k}cams"] = {"median": float(np.median(z)), "robust_sd": float(mad),
                                               "rms": float(np.sqrt(np.mean(z ** 2))),
                                               "beyond_1px": float(np.mean(np.abs(z) > 1))}
        # is part of the detections' error a smooth field (a detector bias that varies with the image's shape across
        # the sensor)?  The same estimator fitted to dev alone; then the data fit against the truth plus that field
        fb = sc.FitData(fd.mask, fd.d, dev, fd.J, fd.L, fd.frame, fd.track)
        f_b = sc.Field(n_cam)
        sc.fit_field(f_b, fb, out["floor_cauchy"]["lam"], tuple(out["floor_cauchy"]["noise"]), robust="cauchy")
        f_b.make_dense([fd.d[fd.mask[:, c], c] for c in range(n_cam)])
        jb = judge_vs(f_b, None, case_dir, spec, cams, a.frame_truth, n_cam)
        f_d = sc.Field(n_cam)
        sc.fit_field(f_d, fd, out["floor_cauchy"]["lam"], tuple(out["floor_cauchy"]["noise"]), robust="cauchy")
        f_d.make_dense([fd.d[fd.mask[:, c], c] for c in range(n_cam)])
        jd = judge_vs(f_d, f_b, case_dir, spec, cams, a.frame_truth, n_cam)
        log(f"  the detections' smooth bias (the estimator on their error alone): its observable RMS {jb['fit_rms']:.3f} px;"
            f" the data fit against the truth plus that bias: error {jd['obs_error']:.3f} px of {jd['obs_true']:.3f} "
            f"(slope {jd['slope']:.2f})")
        out["detector_bias"] = {"observable_rms": jb["fit_rms"], "data_fit_vs_truth_plus_bias": jd}
    log.save(out)


def judge_vs(field, bias, case_dir, spec, cams, t, n_cam):
    """As judge, against the truth plus a bias field (or, with bias None, the field's own observable RMS)."""
    arr = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"), mmap_mode="r")  # noqa: E731
    X = np.asarray(arr("X")[t])
    layer = np.asarray(arr("layer"))
    uv = np.stack([np.asarray(arr(f"uv_{spec['variant']}_c{c}")[t]) for c in range(n_cam)], axis=1)
    E = np.stack([np.asarray(arr(f"err_{spec['variant']}_c{c}")[t]) for c in range(n_cam)], axis=1)
    on = np.isfinite(uv[..., 0]) & (uv[..., 0] >= 0) & (uv[..., 0] < 1280) & (uv[..., 1] >= 0) & (uv[..., 1] < 800)
    sel = (layer == 0) & sc.in_volume(X, sc.view_volume()) & (on.sum(axis=1) >= 2)
    X, uv, E, on = X[sel], uv[sel], E[sel], on[sel]
    J = np.stack([sc.project_jacobian(cams[c], X)[1] for c in range(n_cam)], axis=1)
    C = np.zeros_like(E)
    B = np.zeros_like(E)
    for c in range(n_cam):
        C[on[:, c], c] = field.eval_dense(c, uv[on[:, c], c])
        if bias is not None:
            B[on[:, c], c] = bias.eval_dense(c, uv[on[:, c], c])
    T = E + B if bias is not None else np.zeros_like(E)
    pt, pc = [], []
    for k in range(2, n_cam + 1):
        for S in itertools.combinations(range(n_cam), k):
            m = np.all(on[:, list(S)], axis=1) & (on.sum(axis=1) == k)
            if not m.any():
                continue
            Js = J[m][:, list(S)].reshape(int(m.sum()), 2 * k, 3)
            g = np.all(np.isfinite(Js.reshape(len(Js), -1)), axis=1)
            pt.append(nullspace_project(Js[g], T[m][:, list(S)].reshape(int(m.sum()), 2 * k)[g]).ravel())
            pc.append(nullspace_project(Js[g], C[m][:, list(S)].reshape(int(m.sum()), 2 * k)[g]).ravel())
    pt, pc = np.concatenate(pt), np.concatenate(pc)
    if bias is None:
        return {"fit_rms": float(np.sqrt(np.mean(pc ** 2)))}
    return {"obs_true": float(np.sqrt(np.mean(pt ** 2))), "obs_error": float(np.sqrt(np.mean((pt - pc) ** 2))),
            "slope": float(np.sum(pt * pc) / np.sum(pt * pt))}


if __name__ == "__main__":
    main()
