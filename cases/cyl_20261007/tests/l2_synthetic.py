"""L2 (plan: Phase 4): the line detector (lines.detect) against today's C++ tracer finder (ObjectFinder2D) on
synthetic images with a known truth.

    python tests/l2_synthetic.py           (after p0b_t1_survey.py and l1_preprocess.py)

The images are made as T1's preprocessed frames of one camera: the particles' signal (lines.render: uniform ellipses,
the image of the aperture under astigmatism, blurred by sigma0 and integrated over the pixels; a line is a degenerate
ellipse, brightest in its middle), plus noise as P0b measured it on T1 (hp's noise sigma, of which 1.05 counts is
white -- the temporal noise -- and the rest a texture correlated between neighbouring pixels, here white noise
blurred by a Gaussian of 0.9 px, scaled so that the neighbour correlation is P0b's), times L1's gain, rounded and
clipped to 0..255.  The truth's SNR is the object's noiseless peak over hp's noise sigma.  Both detectors run on the
same 8-bit images: lines.detect (DET's defaults) with the noise known, and the C++ finder (4-neighbour local maxima
above min_obj_int, 3-point log-parabola; radius 2 px) with min_obj_int at 3, 4 and 5 noise sigmas.

Set A, isolated objects (accuracy; cam1's noise and gain): 6 frames, objects on a 64 px grid (no overlaps): H lines
  a = 3, 6, 10, 15 px (b = 0); V lines b = 3, 5, 8 px (a = 0); ellipses (4, 2), (3, 6); round a = b = 0, 1.5, 3;
  blur sigma0 0.8, 1.0, 1.2 px; peak SNR 3, 4, 5, 7, 10, 15, 25; sub-pixel positions at random.
Set B, the astigmatic population at T1's density (recall, duplicates, false positives), per camera, 4 frames of
  1280 x 800 px: depth t uniform in [-1, 1], a = alpha |t - 0.3|, b = beta |t + 0.3|, blur 0.8-1.2 px, flux
  log-normal (30 %); (number, median flux, alpha, beta) calibrated so that P0b's analysis of the synthetic hp
  (components above 4 and 6 sigma, H and V core lengths) matches T1's (P0b) for that camera.  Objects are 'crowded'
  when another object's box (a + 2, b + 2) overlaps theirs; scores are given for all, isolated and crowded objects.
  On cam1's set B also the detector without its second pass, and with the plain (unclipped) model.
Scores (lines.score_detections): recall per SNR bin and at SNR >= 5; duplicates (an extra detection inside a found
object's footprint |du| <= a + 2, |dv| <= b + 2, per object) and the fraction of objects with one; false positives
(in no footprint); centre errors along and across the lines (H, V) or in u and v (others), all and at SNR >= 5.
The plan's targets: recall >= 95 % at SNR >= 5, duplicates <= 1 %, centre error <= 0.1 px across and <= 0.3 px
along.  Writes l2_synthetic_out.txt, l2_synthetic.json and l2_synthetic.png beside itself.
"""
import importlib.util
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
sys.path.insert(0, os.path.join(CASE, "data", "pyd_lines"))
import pyopenlpt as lpt  # noqa: E402  (the frozen copy of build-py/Release)
import lines  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_spec = importlib.util.spec_from_file_location("p0b", os.path.join(HERE, "p0b_t1_survey.py"))
p0b_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p0b_mod)

LOG = []
H, W = 800, 1280
TX, TY = 0.3, -0.3
TEXTURE_SIGMA_PX = 0.9
WHITE = 1.05


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def noise_model(sigma, corr):
    """(white sigma, texture sigma): sigma^2 = w^2 + c^2, neighbour correlation = c^2 rho / sigma^2 with rho the
    texture's own neighbour correlation (Gaussian blur of 0.9 px: exp(-1/(4 s^2)) = 0.73)."""
    rho = math.exp(-1.0 / (4 * TEXTURE_SIGMA_PX ** 2))
    c2 = min(corr * sigma ** 2 / rho, sigma ** 2 - 0.25)
    return math.sqrt(sigma ** 2 - c2), math.sqrt(c2)


def make_noise(rng, shape, w, c):
    tex = ndimage.gaussian_filter(rng.standard_normal(shape), TEXTURE_SIGMA_PX)
    tex *= c / tex.std()
    return w * rng.standard_normal(shape) + tex


def cpp_find(img8, thr):
    cfg = lpt.TracerConfig()
    cfg._radius_obj = 2.0
    cfg._min_obj_int = int(round(thr))
    objs = lpt.ObjectFinder2D().findObject2D(lpt.Image.from_numpy(img8), cfg)
    return np.array([[o._pt_center[0], o._pt_center[1]] for o in objs]) if objs else np.zeros((0, 2))


def population(rng, n, alpha, beta, flux_med):
    t = rng.uniform(-1, 1, n)
    return np.column_stack([rng.uniform(3, W - 4, n), rng.uniform(3, H - 4, n), alpha * np.abs(t - TX),
                            beta * np.abs(t - TY), rng.uniform(0.8, 1.2, n), flux_med * np.exp(rng.normal(0, 0.3, n))])


def p0b_stats(sig, noise, sn):
    hp = sig + noise
    o4 = p0b_mod.components(hp, 4 * sn)
    o6 = p0b_mod.components(hp, 6 * sn)
    LH = [o["w"] for o in o4 if o["cls"] == "H"]
    LV = [o["h"] for o in o4 if o["cls"] == "V"]
    return {"n4": len(o4), "n6": len(o6), "H_med": float(np.median(LH)) if LH else 0.0,
            "V_med": float(np.median(LV)) if LV else 0.0,
            "frac_H": len(LH) / max(len(o4), 1), "frac_V": len(LV) / max(len(o4), 1)}


def calibrate(cam, p0b, w, c, log):
    """(n, flux_med, alpha, beta) whose synthetic hp gives P0b's 4- and 6-sigma counts and H, V core lengths."""
    sn = p0b["noise_counts"]["median"]
    tgt = {"n4": p0b["objects_per_frame_4sigma"]["median"], "n6": p0b["objects_per_frame_6sigma"]["median"],
           "H_med": p0b["shapes_4sigma"]["H_length_px_p10_50_90_99"][1],
           "V_med": p0b["shapes_4sigma"]["V_length_px_p10_50_90_99"][1]}
    n, flux, alpha, beta = 1.5 * tgt["n4"], 150.0, 12.0, 6.0
    noise = make_noise(np.random.default_rng(99), (H, W), w, c)
    best = None
    for it in range(14):
        sig, _ = lines.render((H, W), population(np.random.default_rng(100), int(n), alpha, beta, flux))
        st = p0b_stats(sig, noise, sn)
        err = (abs(math.log(st["n4"] / tgt["n4"])) + abs(math.log(max(st["n6"], 1) / tgt["n6"]))
               + 0.5 * abs(math.log(max(st["H_med"], 1) / tgt["H_med"])) + 0.5 * abs(math.log(max(st["V_med"], 1) / tgt["V_med"])))
        if best is None or err < best[0]:
            best = (err, int(n), flux, alpha, beta, st)
        log(f"    calibration {it}: n {int(n)}, flux {flux:.0f}, alpha {alpha:.1f}, beta {beta:.1f} -> 4 sigma {st['n4']} "
            f"(T1 {tgt['n4']:.0f}), 6 sigma {st['n6']} ({tgt['n6']:.0f}), H core {st['H_med']:.0f} px ({tgt['H_med']:.0f}), "
            f"V core {st['V_med']:.0f} px ({tgt['V_med']:.0f})")
        if err < 0.06:
            break
        # the 6/4 ratio follows the flux, the 4-sigma count the number; the core lengths follow alpha, beta
        r64, r64t = st["n6"] / max(st["n4"], 1), tgt["n6"] / tgt["n4"]
        flux *= (r64t / max(r64, 1e-3)) ** 1.2
        n *= (tgt["n4"] / max(st["n4"], 1)) ** 0.9
        alpha *= (tgt["H_med"] / max(st["H_med"], 1)) ** 0.8
        beta *= (tgt["V_med"] / max(st["V_med"], 1)) ** 0.8
    err, n, flux, alpha, beta, st = best
    return {"n": n, "flux_median": flux, "alpha": alpha, "beta": beta, "stats": st, "target": tgt, "mismatch": err}


def crowded(objs):
    tree = cKDTree(objs[:, :2])
    out = np.zeros(len(objs), bool)
    reach = 2 * (objs[:, 2].max() + objs[:, 3].max()) + 8
    for i, nb in enumerate(tree.query_ball_point(objs[:, :2], r=reach)):
        for j in nb:
            if j != i and abs(objs[i, 0] - objs[j, 0]) <= objs[i, 2] + objs[j, 2] + 4 and \
                    abs(objs[i, 1] - objs[j, 1]) <= objs[i, 3] + objs[j, 3] + 4:
                out[i] = True
                break
    return out


def pool(dets, truths, snrs):
    D, T, S = [], [], []
    for i, (d, t, s) in enumerate(zip(dets, truths, snrs)):
        D.append(np.asarray(d, float).reshape(-1, 2) + np.array([0.0, 10000.0 * i]))
        tt = t.copy()
        tt[:, 1] += 10000.0 * i
        T.append(tt)
        S.append(s)
    return np.vstack(D), np.vstack(T), np.concatenate(S)


def score(dets, truths, snrs, subset=None):
    D, T, S = pool(dets, truths, snrs)
    st, tm, dm, inside = lines.score_detections(D, T, S)
    if subset is not None:
        sub = np.concatenate(subset)
        # recall and errors restricted to the subset of true objects (duplicates and false positives: all)
        m = tm >= 0
        s5 = sub & (S >= 5)
        st["subset_n"] = int(sub.sum())
        st["subset_recall_snr_ge5"] = float((tm[s5] >= 0).mean()) if s5.any() else None
        a, b = T[:, 2], T[:, 3]
        cls = np.where((a >= 2) & (a >= 2 * b), "H", np.where((b >= 2) & (b >= 2 * a), "V", "other"))
        errs = {}
        for c in ("H", "V", "other"):
            s = m & sub & (cls == c) & (S >= 5)
            if not s.any():
                continue
            du = D[tm[s], 0] - T[s, 0]
            dv = D[tm[s], 1] - T[s, 1]
            al, ac = (du, dv) if c in ("H", "other") else (dv, du)
            errs[c] = {"n": int(s.sum()), "along_rms": float(np.sqrt(np.mean(al ** 2))),
                       "across_rms": float(np.sqrt(np.mean(ac ** 2))),
                       "along_p95": float(np.percentile(np.abs(al), 95)), "across_p95": float(np.percentile(np.abs(ac), 95))}
        st["subset_errors_snr_ge5"] = errs
    return st, (tm, dm, inside, D, T, S)


def report(label, st):
    log(f"    {label}: recall at SNR >= 5 {100 * st['recall_snr_ge5']:.1f} %, all {100 * st['recall_all']:.1f} %; "
        f"duplicates {100 * st['duplicates_per_object']:.2f} % per object ({100 * st['objects_with_duplicate']:.2f} % of "
        f"objects have one); false positives {st['false_positives']} ({st['n_det']} detections, {st['n_true']} objects)")
    log("      recall by SNR: " + "; ".join(f"{b['snr'][0]}-{b['snr'][1] or 'inf'}: {100 * b['recall']:.1f} % (n {b['n']})"
                                         for b in st["recall_bins"]))
    for c, e in st["centre_errors_px"].items():
        a5 = "-" if e["along_rms_snr_ge5"] is None else f"{e['along_rms_snr_ge5']:.3f}"
        c5 = "-" if e["across_rms_snr_ge5"] is None else f"{e['across_rms_snr_ge5']:.3f}"
        log(f"      {c:5s} n {e['n']}: along RMS {e['along_rms']:.3f} (bias {e['along_bias']:+.3f}), across RMS "
            f"{e['across_rms']:.3f} (bias {e['across_bias']:+.3f}) px; at SNR >= 5 (n {e['n_snr_ge5']}): along {a5}, "
            f"across {c5} px")


def report_subset(label, st):
    errs = st["subset_errors_snr_ge5"]
    log(f"      {label} (n {st['subset_n']}): recall at SNR >= 5 {100 * st['subset_recall_snr_ge5']:.1f} %; errors at SNR >= 5: "
        + "; ".join(f"{c} along {e['along_rms']:.3f} (95 % {e['along_p95']:.2f}), across {e['across_rms']:.3f} "
                    f"(95 % {e['across_p95']:.2f}) px, n {e['n']}" for c, e in errs.items()))


def per_class_a(pooled, meta_all):
    tm, dm, inside, D, T, S = pooled
    cls = np.array([m[0] for m in meta_all])
    snr_set = np.array([m[1] for m in meta_all])
    out = {}
    for c in ("H", "V", "ellipse", "round"):
        rows = []
        for snr in (3, 4, 5, 7, 10, 15, 25):
            s = (cls == c) & (snr_set == snr)
            if not s.any():
                continue
            idx = np.flatnonzero(s)
            ext = [i for i in idx if tm[i] >= 0]
            du = D[tm[ext], 0] - T[ext, 0]
            dv = D[tm[ext], 1] - T[ext, 1]
            al, ac = (dv, du) if c == "V" else (du, dv)
            dups = int(((dm < 0) & np.isin(inside, idx)).sum())
            rows.append({"snr": snr, "n": int(s.sum()), "recall": float((tm[s] >= 0).mean()), "dups": dups,
                         "along_rms": float(np.sqrt(np.mean(al ** 2))) if len(al) else None,
                         "across_rms": float(np.sqrt(np.mean(ac ** 2))) if len(ac) else None})
        out[c] = rows
    return out


def run(frames, w, c, gain, sn, rng, variants=(("default", {}),), cpp=True):
    """frames: list of (signal, objs, peaks).  Returns per variant the detections, and the C++ finder's."""
    res = {name: [] for name, _ in variants}
    cppd = {k: [] for k in (3, 4, 5)}
    times = {name: [] for name, _ in variants}
    noise_out = gain * sn
    for (sig, objs, peaks) in frames:
        img8 = np.clip(np.round(gain * (sig + make_noise(rng, sig.shape, w, c))), 0, 255).astype(np.uint8)
        for name, over in variants:
            tm = {}
            d = lines.detect(img8, noise_out, lines.det_params(**over), timing=tm)
            res[name].append(np.column_stack([d["u"], d["v"]]))
            times[name].append(tm)
        if cpp:
            for k in (3, 4, 5):
                cppd[k].append(cpp_find(img8, k * noise_out))
    return res, cppd, times


def main():
    rng = np.random.default_rng(2026)
    with open(os.path.join(HERE, "p0b_t1_survey.json"), encoding="utf-8") as fh:
        p0b = json.load(fh)["cameras"]
    with open(os.path.join(HERE, "l1_preprocess.json"), encoding="utf-8") as fh:
        l1 = json.load(fh)["T1"]
    P = lines.det_params()
    out = {"params": P, "cameras": {}}
    log("L2 synthetic: lines.detect (DET defaults) against the C++ finder on T1-like images")
    nm = {}
    for cam in ("cam1", "cam2"):
        sn = p0b[cam]["noise_counts"]["median"]
        corr = 0.5 * (p0b[cam]["noise_correlation"]["horizontal"] + p0b[cam]["noise_correlation"]["vertical"])
        w, c = noise_model(sn, corr)
        nm[cam] = (sn, w, c, float(l1[cam]["gain"]))
        log(f"  {cam}: hp noise {sn:.2f} counts (neighbour correlation {corr:.2f}) = white {w:.2f} + texture {c:.2f}; "
            f"gain {l1[cam]['gain']} -> {l1[cam]['gain'] * sn:.1f} counts in the 8-bit image")

    # ------------------------------------------------------------ set A (cam1)
    sn, w, c, gain = nm["cam1"]
    log("\nSet A: isolated objects on a 64 px grid, 6 frames (cam1's noise and gain)")
    shapes = [("H", 3, 0), ("H", 6, 0), ("H", 10, 0), ("H", 15, 0), ("V", 0, 3), ("V", 0, 5), ("V", 0, 8),
              ("ellipse", 4, 2), ("ellipse", 3, 6), ("round", 0, 0), ("round", 1.5, 1.5), ("round", 3, 3)]
    snrs = [3, 4, 5, 7, 10, 15, 25]
    frames_a, meta_a = [], []
    for fr in range(6):
        objs = []
        for gy in range(40, H - 40, 64):
            for gx in range(40, W - 40, 64):
                cls, a, b = shapes[rng.integers(len(shapes))]
                s0 = [0.8, 1.0, 1.2][rng.integers(3)]
                snr = snrs[rng.integers(len(snrs))]
                u, v = gx + rng.uniform(-0.5, 0.5), gy + rng.uniform(-0.5, 0.5)
                _, pk = lines.render((H, W), np.array([(u, v, a, b, s0, 1.0)]))
                objs.append((u, v, a, b, s0, snr * sn / pk[0]))
                meta_a.append((cls, snr))
        objs = np.array(objs)
        sig, peaks = lines.render((H, W), objs)
        frames_a.append((sig, objs, peaks))
    t0 = time.time()
    ra, ca, ta = run(frames_a, w, c, gain, sn, rng)
    log(f"  (ran in {time.time() - t0:.0f} s)")
    truths = [f[1] for f in frames_a]
    snr_a = [f[2] / sn for f in frames_a]
    st, pooled = score(ra["default"], truths, snr_a)
    report("lines.detect", st)
    out["A"] = {"new": {"pooled": st, "per_class": per_class_a(pooled, meta_a)}, "cpp": {}}
    for cls_, rows in out["A"]["new"]["per_class"].items():
        log(f"      {cls_}: " + "; ".join(
            f"SNR {r['snr']}: {100 * r['recall']:.0f} %, dups {r['dups']}, along "
            f"{'-' if r['along_rms'] is None else round(r['along_rms'], 3)}, across "
            f"{'-' if r['across_rms'] is None else round(r['across_rms'], 3)}" for r in rows))
    for k in (3, 4, 5):
        stc, pc = score(ca[k], truths, snr_a)
        report(f"C++ finder, {k} sigma", stc)
        out["A"]["cpp"][str(k)] = {"pooled": stc, "per_class": per_class_a(pc, meta_a)}
    out["A"]["timing_s_per_frame"] = float(np.median([t["total_s"] for t in ta["default"]]))

    # ------------------------------------------------------------ set B (both cameras)
    for cam in ("cam1", "cam2"):
        sn, w, c, gain = nm[cam]
        log(f"\nSet B, {cam}: calibrating the astigmatic population to P0b")
        cal = calibrate(cam, p0b[cam], w, c, log)
        log(f"  calibrated: {cal['n']} objects per frame, flux median {cal['flux_median']:.0f}, alpha {cal['alpha']:.1f}, "
            f"beta {cal['beta']:.1f} px (H lines up to {2 * 1.3 * cal['alpha']:.0f} px long, V lines up to "
            f"{2 * 1.3 * cal['beta']:.0f} px); P0b's analysis of it: {cal['stats']}")
        frames_b, crowd_b = [], []
        for fr in range(4):
            objs = population(rng, cal["n"], cal["alpha"], cal["beta"], cal["flux_median"])
            sig, peaks = lines.render((H, W), objs)
            frames_b.append((sig, objs, peaks))
            crowd_b.append(crowded(objs))
        variants = (("default", {}),)
        if cam == "cam1":
            variants = (("default", {}), ("no second pass", {"second_pass": False}),
                        ("plain model (unclipped)", {"clipped": False}))
        t0 = time.time()
        rb, cb, tb = run(frames_b, w, c, gain, sn, rng, variants)
        log(f"  (ran in {time.time() - t0:.0f} s; crowded objects {100 * np.mean(np.concatenate(crowd_b)):.0f} %)")
        truths = [f[1] for f in frames_b]
        snr_b = [f[2] / sn for f in frames_b]
        res_cam = {"calibration": cal, "crowded_fraction": float(np.mean(np.concatenate(crowd_b))), "variants": {}}
        for name, _ in variants:
            st, pooled = score(rb[name], truths, snr_b)
            report(f"lines.detect ({name})", st)
            sti, _ = score(rb[name], truths, snr_b, subset=[~cb_ for cb_ in crowd_b])
            stc, _ = score(rb[name], truths, snr_b, subset=crowd_b)
            report_subset("isolated", sti)
            report_subset("crowded", stc)
            tmed = {k: float(np.median([t[k] for t in tb[name]])) for k in tb[name][0]}
            log("      time per frame (median, s): " + ", ".join(f"{k} {v:.1f}" for k, v in tmed.items()))
            res_cam["variants"][name] = {"all": st, "isolated": {k: sti[k] for k in ("subset_n", "subset_recall_snr_ge5",
                                                                                     "subset_errors_snr_ge5")},
                                         "crowded": {k: stc[k] for k in ("subset_n", "subset_recall_snr_ge5",
                                                                         "subset_errors_snr_ge5")},
                                         "timing_s_per_frame": tmed}
        for k in (3, 4, 5):
            st, _ = score(cb[k], truths, snr_b)
            report(f"C++ finder, {k} sigma", st)
            res_cam[f"cpp_{k}"] = st
        out["cameras"][cam] = res_cam

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for cam, ls in (("cam1", "-"), ("cam2", "--")):
        rc = out["cameras"][cam]
        for lab, st in ((f"{cam} lines.detect", rc["variants"]["default"]["all"]), (f"{cam} C++ 4 sigma", rc["cpp_4"])):
            x = [0.5 * (b["snr"][0] + (b["snr"][1] or 30)) for b in st["recall_bins"]]
            ax[0].plot(x, [b["recall"] for b in st["recall_bins"]], "o" + ls, label=lab)
    ax[0].set_xlabel("peak SNR (bin centre)")
    ax[0].set_ylabel("recall")
    ax[0].set_title("set B (T1-like populations)")
    ax[0].axhline(0.95, color="k", lw=0.5)
    ax[0].legend(fontsize=7)
    for cls_, mk in (("H", "o"), ("V", "s"), ("ellipse", "^"), ("round", "x")):
        rows = out["A"]["new"]["per_class"][cls_]
        ax[1].plot([r["snr"] for r in rows], [r["along_rms"] for r in rows], mk + "-", label=f"{cls_} along")
        ax[2].plot([r["snr"] for r in rows], [r["across_rms"] for r in rows], mk + "-", label=f"{cls_} across")
    for a_, lim, t in ((ax[1], 0.3, "centre error along (px), set A"), (ax[2], 0.1, "centre error across (px), set A")):
        a_.axhline(lim, color="k", lw=0.5)
        a_.set_xscale("log")
        a_.set_yscale("log")
        a_.set_xlabel("peak SNR")
        a_.set_title(t)
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "l2_synthetic.png"), dpi=110)
    with open(os.path.join(HERE, "l2_synthetic.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(HERE, "l2_synthetic_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
