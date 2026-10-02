"""T13d: a model-based correction of the detector's remaining bias.  Plan: 2026-10-02-ball-centres-plan.md, T13d
("only if T13c shows a bias that matters": T13c's best detector, ellipse_rodmask, failed its criteria narrowly, with
f biased +0.03 to +0.07 % (t up to 3.5) and z RMS 1.5-1.6 in backlit and even light).

For each of T13c's 8 data sets, in backlit and in even light (the lightings a model can describe):
  1. calibrate from scratch on T13c's ellipse_rodmask centres (as T13c);
  2. for every detection, render the image the fitted calibration predicts: the fitted camera, the fitted wand ends
     (held-out frames: triangulated with the fitted cameras), the known balls and rod, the lighting, and the blur
     model of the renderer; noise-free.  Run the same detector on it; its offset from the projected centre is the
     detector's predicted error.  The model's blur is assumed known; in practice it would come from the edge
     profiles, as the plate case measured it;
  3. subtract the predicted errors from the measured centres and calibrate again (from the first fit);
  4. measure as T13c.
Criteria (T13c's): |t| <= 3 for every camera's f and for the held-out wand length; z RMS <= 1.2.
Outputs: t13d_correction_out.txt, t13d_correction.json.
"""
import t13_common as T  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
from rigs import cm  # noqa: E402
import cylinder_wand as cw  # noqa: E402
import t13c_calibration as C  # noqa: E402

OUT = rigs.HERE / "t13d_correction"
log = rigs.Tee(str(OUT) + "_out.txt")
DET = "ellipse_rodmask"
LIGHTS = ("backlit", "even")
CHUNK = 50


def sizes(cams):
    return [(c.width, c.height) for c in cams]


def fit_task(args):
    k, light, uv = args
    cams = T.cam1_rig()
    t0 = time.time()
    sol = cw.calibrate(cw.WandData(uv[:C.N_TRAIN], T.L_WAND, sizes(cams)), T.TUBE)
    XA, XB = sol.frames.ends(sol.L)
    ids = sol.obs.frame_ids
    EA = np.full((C.N_TRAIN, 3), np.nan)
    EB = np.full((C.N_TRAIN, 3), np.nan)
    EA[ids], EB[ids] = XA, XB
    return {"k": k, "light": light, "cams": sol.cams, "EA": EA, "EB": EB, "seconds": time.time() - t0,
            "line": f"rms {sol.rms:.4f} px"}


def predict_task(args):
    """Predicted detector errors for frames [f0, f0 + n) of one data set and lighting."""
    k, light, f0, uv, cams_fit, EA, EB, cloud = args
    n = len(uv)
    delta = np.full_like(uv, np.nan)
    cpp = [cm.to_cpp(c, T.TUBE) for c in cams_fit]
    pol = T.POLARITY[light]
    n_ok = n_bad = 0
    for i in range(n):
        f = f0 + i
        have = [c for c in range(len(cams_fit)) if np.all(np.isfinite(uv[i, c]))]
        if len(have) < 2:
            continue
        if f < C.N_TRAIN and np.all(np.isfinite(EA[i])):
            XA, XB = EA[i], EB[i]
        else:
            # held out (or not in the fit): triangulate the two ends with the fitted cameras, in the fit's frame,
            # from the fitted cloud's centre
            ca = np.array([c for c in have for _e in range(2)])
            en = np.array([e for _c in have for e in range(2)])
            X, r, _cov = cw.triangulate_ml(cams_fit, T.TUBE, ca, uv[i, ca, en], en, np.array([cloud, cloud]))
            if not np.all(np.isfinite(r)) or float(np.max(np.abs(r))) > 1.0:
                n_bad += len(have)
                continue
            XA, XB = X[0], X[1]
        for c in have:
            cam = cams_fit[c]
            pA, pB = cm.cpp_project(cpp[c], np.array([XA, XB]))
            if not (np.all(np.isfinite(pA)) and np.all(np.isfinite(pB))):
                n_bad += 1
                continue
            geo = T.render_geometry(cam, XA, XB, pA, pB)
            img = T.compose(cam, geo, light, T.ball_blur(cam, XA), T.ball_blur(cam, XB))
            d = T.detect(DET, img, pol)
            if len(d) < 2:
                n_bad += 1
                continue
            dA, dB = np.array(d[0][:2]) - pA, np.array(d[1][:2]) - pB
            if max(np.hypot(*dA), np.hypot(*dB)) > 2.0:          # the prediction itself failed: no correction
                n_bad += 1
                continue
            delta[i, c, 0], delta[i, c, 1] = dA, dB
            n_ok += 1
    return {"k": k, "light": light, "f0": f0, "delta": delta, "n_ok": n_ok, "n_bad": n_bad,
            "line": f"{n_ok} predicted, {n_bad} not"}


def refit_task(args):
    k, light, uv_corr, cams_start, UVt, XAt, XBt = args
    cams = T.cam1_rig()
    t0 = time.time()
    sol = cw.calibrate(cw.WandData(uv_corr[:C.N_TRAIN], T.L_WAND, sizes(cams)), T.TUBE, start=cams_start)
    fit = C.align_to_truth(sol.cams, cams)
    phys = T.tc.physical(fit, cams)
    un = T.tc.unseen(fit, cams, T.TUBE, P=C.GRID8)
    d_det = C.lengths(fit, uv_corr[C.N_TRAIN:], XAt[C.N_TRAIN:], XBt[C.N_TRAIN:])
    d_ex = C.lengths(fit, UVt[C.N_TRAIN:], XAt[C.N_TRAIN:], XBt[C.N_TRAIN:])
    z = T.tc.delta_vec(fit, cams, sol.layout) / np.sqrt(np.diag(sol.cov))
    rec = {"k": k, "name": f"{DET}|{light}|corrected", "ok": True, "seconds": time.time() - t0,
           "f_rel": [p["f_rel"] for p in phys], "distance_mm": [p["distance_mm"] for p in phys], "unseen": un,
           "held_det_um": d_det.tolist(), "held_exact_um": d_ex.tolist(), "z": z.tolist(), "rms_px": sol.rms,
           "sigma_px": sol.sigma_px, "n_obs": int(len(sol.obs)), "rejected": int((~sol.obs.active).sum())}
    rec["line"] = (f"f {', '.join(f'{100 * v:+.3f}' for v in rec['f_rel'])} %, shape {un['rigid_um']:.2f} um, "
                   f"z {math.sqrt(float(np.mean(z ** 2))):.2f}, rms {sol.rms:.4f} px")
    return rec


def main():
    t0 = time.time()
    cams = T.cam1_rig()
    npz = np.load(str(rigs.HERE / "t13c_calibration_uv.npz"))
    prev = json.load(open(str(rigs.HERE / "t13c_calibration.json"), encoding="utf-8"))
    sets = []
    for k in range(C.K):
        XA, XB, UV = T.wand_frames(cams, C.N_TRAIN + C.N_HELD, np.random.default_rng(C.SEED + k))
        assert np.allclose(np.nan_to_num(UV), np.nan_to_num(npz[f"{k}|truth"])), "frames differ from T13c's"
        sets.append((XA, XB, UV))
    log(f"T13d: model-based correction of {DET}'s remaining bias (T13c's data sets and centres; lightings "
        f"{', '.join(LIGHTS)})")
    log("")
    log("1. Calibrate on the measured centres (as T13c)")
    fits = T.tc.run_parallel(fit_task, [(k, li, npz[f"{k}|{DET}|{li}"]) for li in LIGHTS for k in range(C.K)], log,
                             label=lambda t: f"set {t[0]} {t[1]}", workers=30)
    log("")
    log("2. Predict each detection's error from the fitted calibration")
    ptasks = []
    for r in fits:
        k, li = r["k"], r["light"]
        uv = npz[f"{k}|{DET}|{li}"]
        XA, XB, _UV = sets[k]
        for f0 in range(0, C.N_TRAIN + C.N_HELD, CHUNK):
            s = slice(f0, f0 + CHUNK)
            EA = np.full((CHUNK, 3), np.nan)
            EB = np.full((CHUNK, 3), np.nan)
            m = min(f0 + CHUNK, C.N_TRAIN)
            if f0 < C.N_TRAIN:
                EA[:m - f0], EB[:m - f0] = r["EA"][f0:m], r["EB"][f0:m]
            cloud = np.nanmean(np.concatenate([r["EA"], r["EB"]]), axis=0)
            ptasks.append((k, li, f0, uv[s], r["cams"], EA, EB, cloud))
    pres = T.tc.run_parallel(predict_task, ptasks, log, label=lambda t: f"set {t[0]} {t[1]} frames {t[2]}-",
                             workers=30)
    bad = [p for p in pres if "error" in p]
    if bad:
        log(bad[0]["error"])
        raise SystemExit("a prediction task failed")
    log("")
    log("Predicted errors (the correction), over all detections:")
    corr = {}
    stats = {}
    for li in LIGHTS:
        for k in range(C.K):
            uv = npz[f"{k}|{DET}|{li}"]
            delta = np.full_like(uv, np.nan)
            for p in pres:
                if p["k"] == k and p["light"] == li:
                    delta[p["f0"]:p["f0"] + len(p["delta"])] = p["delta"]
            corr[(k, li)] = np.where(np.isfinite(delta), uv - delta, uv)
            # how well does the predicted error match the actual one (known here from the truth)?
            act = uv - sets[k][2]
            both = np.all(np.isfinite(delta), axis=-1) & np.all(np.isfinite(act), axis=-1)
            stats.setdefault(li, []).append((delta[both], act[both]))
        D = np.concatenate([s[0] for s in stats[li]])
        A = np.concatenate([s[1] for s in stats[li]])
        n_ok = sum(p["n_ok"] for p in pres if p["light"] == li)
        n_bad = sum(p["n_bad"] for p in pres if p["light"] == li)
        rms_d = float(np.sqrt(np.mean(np.sum(D ** 2, axis=1))))
        rms_left = float(np.sqrt(np.mean(np.sum((A - D) ** 2, axis=1))))
        rms_a = float(np.sqrt(np.mean(np.sum(A ** 2, axis=1))))
        stats[li] = {"camera_frames_predicted": n_ok, "not_predicted": n_bad, "predicted_rms_px": rms_d,
                     "actual_rms_px": rms_a, "left_after_correction_rms_px": rms_left}
        log(f"  {li:<8s} {n_ok} camera-frames predicted ({n_bad} not); predicted error RMS {rms_d:.4f} px; actual "
            f"error RMS {rms_a:.4f} px; left after the correction {rms_left:.4f} px")
    log("")
    log("3. Calibrate again on the corrected centres, and measure")
    rtasks = []
    for r in fits:
        k, li = r["k"], r["light"]
        XA, XB, UV = sets[k]
        rtasks.append((k, li, corr[(k, li)], r["cams"], UV, XA, XB))
    rres = T.tc.run_parallel(refit_task, rtasks, log, label=lambda t: f"set {t[0]} {t[1]}", workers=30)
    out = {"correction": stats, "configs": {}}
    log("")
    log("Results over the data sets (as T13c; the uncorrected rows are T13c's)")
    rows = [("ideal", prev["configs"]["ideal"]["summary"])]
    for li in LIGHTS:
        rows.append((f"{DET}|{li}", prev["configs"][f"{DET}|{li}"]["summary"]))
        s = C.summary([r for r in rres if r.get("name") == f"{DET}|{li}|corrected"])
        out["configs"][f"{DET}|{li}|corrected"] = {"summary": s,
                                                   "runs": [r for r in rres if r.get("name") ==
                                                            f"{DET}|{li}|corrected"]}
        rows.append((f"{DET}|{li}|corrected", s))
    for name, s in rows:
        log(f"  {name:<34s} f mean {', '.join(f'{v:+.3f}' for v in s['f_mean_pct'])} % "
            f"(t {', '.join(f'{v:+.1f}' for v in s['f_t'])}), RMS {', '.join(f'{v:.3f}' for v in s['f_rms_pct'])} % | "
            f"held {s['held_mean_um']:+.2f} um (t {s['held_t']:+.1f}), RMS {s['held_rms_um']:.2f} | shape "
            f"{s['rigid_um']:.2f}, tube {s['stab_um']:.1f} um | z {s['z_rms']:.2f} | fit {s['rms_px']:.4f} px, "
            f"rejected {100 * s['rejected_frac']:.1f} %")
    log("")
    log("Criteria: |t| <= 3 for every camera's f and for the held-out wand length; z RMS <= 1.2")
    verdict = {}
    for li in LIGHTS:
        s = out["configs"][f"{DET}|{li}|corrected"]["summary"]
        verdict[li] = bool(s.get("n_ok") == C.K and max(abs(v) for v in s["f_t"]) <= 3.0 and abs(s["held_t"]) <= 3.0
                           and s["z_rms"] <= 1.2)
        log(f"  {DET}|{li}|corrected  {'PASS' if verdict[li] else 'fail'}")
    out["verdict"] = verdict
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    log("")
    log(f"done in {time.time() - t0:.0f} s")
    log.close()


if __name__ == "__main__":
    main()
