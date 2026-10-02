"""T13c: what each ball detector does to the calibration.  Plan: 2026-10-02-ball-centres-plan.md.

K data sets on the cam1-like rig (three cam1 replicas at 0/45/90 deg, 1280 x 800 px), each of 500 wand frames for the
calibration and 100 held out (midpoints in +-8 mm), rendered through the tube at best focus in the four lightings.
Each detector measures every image.  Its two detections, smallest first, go to the calibration as balls A and B
(NaN when it finds fewer than two), with no knowledge of the truth, as they would in practice.
  - The calibration runs from scratch with calibrate's defaults.
  - The baseline ("ideal") is the same frames with the exact centres plus Gaussian noise of SIGMA_IDEAL per
    coordinate, about the detectors' random part in T13b.
  - The baseline is also calibrated from the true cameras ("ideal_truth"), to check that the start finds the global
    minimum on this rig.
Measured against the truth (cameras aligned to the truth's gauge: camera 1 at azimuth 0 and height 0):
  - f per camera;
  - the 3-D error of unseen points (a 7 x 7 x 7 grid over +-8 mm, re-triangulated from their exact images):
    relative to the tube, and shape and scale (after the best rigid motion);
  - the held-out wand lengths, triangulated without the length constraint from the detector's centres and from the
    exact ones (the calibration's share);
  - z = parameter error / the fit's own sigma (honest error bars: RMS 1);
  - the fit's RMS, its noise estimate and its rejections.
Criteria for the recommended detector: no bias beyond 3 standard errors (over the K data sets) in any camera's f or in
the held-out wand length, and z RMS <= 1.2.
Outputs: t13c_calibration_out.txt, t13c_calibration.json, t13c_calibration_uv.npz (every detector's centres).
"""
import t13_common as T  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
from rigs import cm  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t13c_calibration"
log = rigs.Tee(str(OUT) + "_out.txt")
K = 8
N_TRAIN, N_HELD = 500, 100
CHUNK = 50
SEED = 1303
SIGMA_IDEAL = 0.02
DETS = ("pyopenlpt_fast", "blob", "blob_open", "ellipse", "ellipse_rodmask")
_g = np.linspace(-T.HALF, T.HALF, 7)
GRID8 = np.array([[x, y, z] for x in _g for y in _g for z in _g])


def keys():
    return [f"{d}|{li}" for d in DETS for li in T.LIGHTINGS]


# ------------------------------------------------------------------ phase 1: render and detect
def detect_task(args):
    k, ch, XA, XB, UV = args
    rng = np.random.default_rng(SEED + 1000 * k + ch)
    cams = T.cam1_rig()
    n = len(XA)
    out = {key: np.full((n, len(cams), 2, 2), np.nan) for key in keys()}
    for f in range(n):
        for c, cam in enumerate(cams):
            if not np.all(np.isfinite(UV[f, c])):
                continue
            geo = T.render_geometry(cam, XA[f], XB[f], UV[f, c, 0], UV[f, c, 1])
            crop = T.crop_around(geo, cam)
            bA, bB = T.ball_blur(cam, XA[f]), T.ball_blur(cam, XB[f])
            for light in T.LIGHTINGS:
                img = T.compose(cam, geo, light, bA, bB, rng=rng)
                for d in DETS:
                    det = T.detect(d, img, T.POLARITY[light], crop=crop)
                    if len(det) >= 2:
                        out[f"{d}|{light}"][f, c, 0] = det[0][:2]
                        out[f"{d}|{light}"][f, c, 1] = det[1][:2]
    return {"k": k, "chunk": ch, "uv": out, "line": f"data set {k}, frames {ch * CHUNK}-{ch * CHUNK + n - 1}"}


# ------------------------------------------------------------------ phase 2: calibrate and measure
def align_to_truth(cams, truth):
    """The fitted cameras in the truth's gauge: spin about and slide along the axis so camera 1 sits at azimuth 0 and
    height 0, and the half-turn about Z if the fit took the other sign of the axis."""
    fixed = rigs.gauge_fix(cams)
    if cm.rotation_angle_deg(fixed[1].R, truth[1].R) > 90.0:
        Hm = np.diag([-1.0, -1.0, 1.0])
        flipped = []
        for c in fixed:
            q = c.copy()
            q.C = Hm @ c.C
            q.R = c.R @ Hm
            flipped.append(q)
        fixed = rigs.gauge_fix(flipped)
    return fixed


def lengths(fit_cams, uv, XA, XB):
    """Held-out frames triangulated without the length constraint: length - L (um) per frame."""
    F = uv.shape[0]
    rows = [(f, c, e) for f in range(F) for e in range(2) for c in range(uv.shape[1])
            if np.all(np.isfinite(uv[f, c, e]))]
    if not rows:
        return np.array([])
    fr_ = np.array([q[0] for q in rows])
    ca_ = np.array([q[1] for q in rows])
    en_ = np.array([q[2] for q in rows])
    # a point needs two cameras
    cnt = np.zeros((F, 2), int)
    np.add.at(cnt, (fr_, en_), 1)
    keep = cnt[fr_, en_] >= 2
    fr_, ca_, en_ = fr_[keep], ca_[keep], en_[keep]
    X0 = np.zeros((2 * F, 3))
    X0[0::2], X0[1::2] = XA, XB
    X, _r, _cov = cw.triangulate_ml(fit_cams, T.TUBE, ca_, uv[fr_, ca_, en_], fr_ * 2 + en_, X0)
    good = (cnt[:, 0] >= 2) & (cnt[:, 1] >= 2)
    d = np.linalg.norm(X[1::2] - X[0::2], axis=1) - T.L_WAND
    return 1e3 * d[good]


def calib_task(args):
    k, name, uv, UVt, XA, XB = args
    cams = T.cam1_rig()
    t0 = time.time()
    if name.startswith("ideal"):
        rng = np.random.default_rng(SEED + 77 + k)
        uv = UVt + rng.normal(0.0, SIGMA_IDEAL, UVt.shape)
    data = cw.WandData(uv[:N_TRAIN], T.L_WAND, [(c.width, c.height) for c in cams])
    try:
        sol = cw.calibrate(data, T.TUBE, start=cams if name == "ideal_truth" else None)
    except Exception as exc:  # noqa: BLE001 -- a failed calibration is a result
        return {"k": k, "name": name, "ok": False, "error": repr(exc)[:300], "seconds": time.time() - t0,
                "line": f"FAILED {repr(exc)[:80]}"}
    fit = align_to_truth(sol.cams, cams)
    phys = T.tc.physical(fit, cams)
    un = T.tc.unseen(fit, cams, T.TUBE, P=GRID8)
    d_det = lengths(fit, uv[N_TRAIN:], XA[N_TRAIN:], XB[N_TRAIN:])
    d_ex = lengths(fit, UVt[N_TRAIN:], XA[N_TRAIN:], XB[N_TRAIN:])
    z = T.tc.delta_vec(fit, cams, sol.layout) / np.sqrt(np.diag(sol.cov))
    nw = sol.info.get("noise_weights") or {}
    obs = sol.obs
    rec = {"k": k, "name": name, "ok": True, "seconds": time.time() - t0,
           "f_rel": [p["f_rel"] for p in phys], "distance_mm": [p["distance_mm"] for p in phys],
           "unseen": un, "held_det_um": d_det.tolist(), "held_exact_um": d_ex.tolist(),
           "z": z.tolist(), "rms_px": sol.rms, "sigma_px": sol.sigma_px,
           "sigma_cam_px": nw.get("sigma_group_px"), "weights_used": nw.get("used"),
           "n_obs": int(len(obs)), "rejected": int((~obs.active).sum()),
           "rejected_per_cam": [int(np.sum(~obs.active & (obs.cam == c))) for c in range(obs.n_cams)],
           "starts_tried": sol.info.get("starts_tried")}
    rec["line"] = (f"f {', '.join(f'{100 * v:+.3f}' for v in rec['f_rel'])} %, shape {un['rigid_um']:.2f} um, "
                   f"held {np.mean(d_det):+.2f} um, z {math.sqrt(float(np.mean(z ** 2))):.2f}, "
                   f"rms {sol.rms:.3f} px, rejected {rec['rejected']}")
    return rec


# ------------------------------------------------------------------ aggregation
def summary(recs):
    ok = [r for r in recs if r.get("ok")]
    out = {"n": len(recs), "n_ok": len(ok)}
    if not ok:
        return out
    F = np.array([r["f_rel"] for r in ok])
    n = len(F)
    se = F.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.full(F.shape[1], np.nan)
    out["f_mean_pct"] = (100 * F.mean(axis=0)).tolist()
    out["f_rms_pct"] = (100 * np.sqrt(np.mean(F ** 2, axis=0))).tolist()
    out["f_t"] = (F.mean(axis=0) / se).tolist()
    Hm = np.array([np.mean(r["held_det_um"]) if r["held_det_um"] else np.nan for r in ok])
    out["held_mean_um"] = float(np.nanmean(Hm))
    out["held_t"] = float(np.nanmean(Hm) / (np.nanstd(Hm, ddof=1) / math.sqrt(np.sum(np.isfinite(Hm))))) \
        if np.sum(np.isfinite(Hm)) > 1 else float("nan")
    out["held_rms_um"] = float(np.sqrt(np.mean(np.concatenate([np.asarray(r["held_det_um"]) ** 2 for r in ok]))))
    He = np.concatenate([np.asarray(r["held_exact_um"]) for r in ok])
    out["held_exact_mean_um"] = float(np.mean(He))
    out["held_exact_rms_um"] = float(np.sqrt(np.mean(He ** 2)))
    out["rigid_um"] = float(np.sqrt(np.mean([r["unseen"]["rigid_um"] ** 2 for r in ok])))
    out["stab_um"] = float(np.sqrt(np.mean([r["unseen"]["stab_um"] ** 2 for r in ok])))
    Z = np.concatenate([np.asarray(r["z"]) for r in ok])
    out["z_rms"] = float(np.sqrt(np.mean(Z ** 2)))
    out["rms_px"] = float(np.mean([r["rms_px"] for r in ok]))
    out["sigma_px"] = float(np.mean([r["sigma_px"] for r in ok]))
    out["rejected_frac"] = float(np.sum([r["rejected"] for r in ok]) / np.sum([r["n_obs"] for r in ok]))
    out["n_obs"] = float(np.mean([r["n_obs"] for r in ok]))
    return out


def main():
    t0 = time.time()
    cams = T.cam1_rig()
    log("T13c: what each ball detector does to the calibration (cam1-like rig, 1280 x 800 px, best focus)")
    log(f"  {K} data sets x ({N_TRAIN} + {N_HELD} held-out) frames; detectors {', '.join(DETS)}; "
        f"lightings {', '.join(T.LIGHTINGS)}; baseline: exact centres + {SIGMA_IDEAL} px noise")
    sets = []
    tasks = []
    for k in range(K):
        rng = np.random.default_rng(SEED + k)
        XA, XB, UV = T.wand_frames(cams, N_TRAIN + N_HELD, rng)
        sets.append((XA, XB, UV))
        for ch in range(math.ceil((N_TRAIN + N_HELD) / CHUNK)):
            s = slice(ch * CHUNK, (ch + 1) * CHUNK)
            tasks.append((k, ch, XA[s], XB[s], UV[s]))
    log("")
    log("Phase 1: render and detect")
    res = T.tc.run_parallel(detect_task, tasks, log, label=lambda t: f"set {t[0]} chunk {t[1]}")
    bad = [r for r in res if "error" in r]
    if bad:
        log(bad[0]["error"])
        raise SystemExit("a detection task failed")
    UVD = {k: {key: np.concatenate([r["uv"][key] for r in sorted((q for q in res if q["k"] == k),
                                                                    key=lambda q: q["chunk"])])
               for key in keys()} for k in range(K)}
    np.savez_compressed(str(OUT) + "_uv.npz", **{f"{k}|{key}": UVD[k][key] for k in range(K) for key in keys()},
                        **{f"{k}|truth": sets[k][2] for k in range(K)})
    # how many observations each detector delivers, and how many are far off (> 1 px) or wrong (> 5 px)
    log("")
    log("Detections delivered to the calibration (training frames, all data sets): share of the visible balls "
        "measured, and of those the share more than 1 px / 5 px from the truth")
    det_stats = {}
    for key in keys():
        vis = tot = far1 = far5 = 0
        for k in range(K):
            UVt = sets[k][2][:N_TRAIN]
            D = UVD[k][key][:N_TRAIN]
            v = np.all(np.isfinite(UVt), axis=-1)
            got = v & np.all(np.isfinite(D), axis=-1)
            e = np.hypot(*(D - UVt).transpose(3, 0, 1, 2))
            vis += int(v.sum())
            tot += int(got.sum())
            far1 += int(np.sum(got & (e > 1.0)))
            far5 += int(np.sum(got & (e > 5.0)))
        det_stats[key] = {"visible": vis, "measured": tot, "over_1px": far1, "over_5px": far5}
        log(f"  {key:<30s} measured {tot}/{vis} ({100 * tot / vis:.1f} %), > 1 px {100 * far1 / max(tot, 1):.2f} %, "
            f"> 5 px {100 * far5 / max(tot, 1):.2f} %")
    log("")
    log("Phase 2: calibrate from scratch and measure")
    ctasks = []
    for k in range(K):
        XA, XB, UVt = sets[k]
        ctasks.append((k, "ideal", None, UVt, XA, XB))
        ctasks.append((k, "ideal_truth", None, UVt, XA, XB))
        for key in keys():
            ctasks.append((k, key, UVD[k][key], UVt, XA, XB))
    cres = T.tc.run_parallel(calib_task, ctasks, log, label=lambda t: f"set {t[0]} {t[1]}")
    # the start: did the ideal calibration from scratch reach the truth-started one's minimum?
    log("")
    log("Global minimum on this rig (ideal data, from scratch vs started at the truth):")
    starts = []
    for k in range(K):
        a = next(r for r in cres if r.get("k") == k and r.get("name") == "ideal")
        b = next(r for r in cres if r.get("k") == k and r.get("name") == "ideal_truth")
        if a.get("ok") and b.get("ok"):
            dr = abs(a["rms_px"] - b["rms_px"])
            df = max(abs(x - y) for x, y in zip(a["f_rel"], b["f_rel"]))
            starts.append({"k": k, "rms_diff_px": dr, "f_diff": df, "same": bool(dr < 1e-6 and df < 1e-7)})
            log(f"  set {k}: RMS difference {dr:.1e} px, largest f difference {df:.1e} -> "
                f"{'same minimum' if starts[-1]['same'] else 'DIFFERENT'}")
        else:
            starts.append({"k": k, "same": False})
            log(f"  set {k}: a calibration failed ({a.get('error') or b.get('error')})")
    out = {"K": K, "sigma_ideal_px": SIGMA_IDEAL, "detections": det_stats, "starts": starts, "configs": {}}
    log("")
    log("Results over the data sets (f: mean and RMS of the relative error per camera, with the mean's t; held-out "
        "wand length from the detector's centres: mean, t, RMS; from exact centres: RMS; 3-D shape and relative to "
        "the tube: RMS; z RMS; fit RMS; rejected)")
    for name in ["ideal"] + keys():
        recs = [r for r in cres if r.get("name") == name]
        s = summary(recs)
        out["configs"][name] = {"summary": s, "runs": recs}
        if not s.get("n_ok"):
            log(f"  {name:<30s} all {s['n']} calibrations failed: {recs[0].get('error')}")
            continue
        log(f"  {name:<30s} ok {s['n_ok']}/{s['n']}  f mean {', '.join(f'{v:+.3f}' for v in s['f_mean_pct'])} % "
            f"(t {', '.join(f'{v:+.1f}' for v in s['f_t'])}), RMS {', '.join(f'{v:.3f}' for v in s['f_rms_pct'])} % | "
            f"held {s['held_mean_um']:+.2f} um (t {s['held_t']:+.1f}), RMS {s['held_rms_um']:.2f}; exact "
            f"{s['held_exact_rms_um']:.2f} | shape {s['rigid_um']:.2f}, tube {s['stab_um']:.1f} um | z {s['z_rms']:.2f} | "
            f"fit {s['rms_px']:.3f} px, rejected {100 * s['rejected_frac']:.1f} %")
    # verdict
    log("")
    log("Criteria: |t| <= 3 for every camera's f and for the held-out wand length; z RMS <= 1.2")
    verdict = {}
    for name in keys():
        s = out["configs"][name]["summary"]
        if not s.get("n_ok") or s["n_ok"] < K:
            verdict[name] = False
        else:
            verdict[name] = bool(max(abs(v) for v in s["f_t"]) <= 3.0 and abs(s["held_t"]) <= 3.0
                                 and s["z_rms"] <= 1.2)
        log(f"  {name:<30s} {'PASS' if verdict[name] else 'fail'}")
    out["verdict"] = verdict
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    log("")
    log(f"done in {time.time() - t0:.0f} s")
    log.close()


if __name__ == "__main__":
    main()
