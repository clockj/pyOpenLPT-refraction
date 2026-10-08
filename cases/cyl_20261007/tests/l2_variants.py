"""L2 (plan: Phase 4): the evidence behind the detector's design choices (lines.DET), kept with the other tests.

    python tests/l2_variants.py          (after l1_preprocess.py; about 5 min)

1. k_detect: candidates per frame (3 x 3 local maxima of the best normalised score, after the suppression) on pure
   clipped Gaussian noise (sigma 6 counts, 1280 x 800, 3 frames), for k = 4 ... 7.
2. The texture: on T1 cam1 frame 500 (preprocessed as L1), each template's noise level (the MAD of its response)
   against what white noise of the image's sigma would give (sqrt(norm2) * sigma).
3. The bounds and the suppression on T1 (cam1 frame 500): DET (dynamic bounds, symmetric suppression) against
   lengths bounded by the template (1.5 L_t + 8 px) and against asymmetric suppression: objects found, classes, H
   and V lengths, and the fraction pinned at exactly 8 px.
4. The fit model on isolated objects (64 px grid, 228 objects; SNR 4, 6, 10, 20; cam1's noise with its texture and
   gain): the clipped model (DET) against the plain one: recall, flux and RMS widths against the truth, centres.
5. A dense synthetic frame (1,500 objects on 1280 x 800, a = 11 |t - 0.3|, b = 5 |t + 0.3|, flux 400 counts
   median; white noise 1.5 counts, gain 4): DET against bounded lengths and against the plain model.
6. The fraction of zero pixels in L1's 8-bit frames (cam1 and cam2, frame 500).
Writes l2_variants_out.txt and l2_variants.json beside itself.
"""
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import lines  # noqa: E402
import t1  # noqa: E402

LOG = []
OUT = {}


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def main():
    with open(os.path.join(HERE, "l1_preprocess.json"), encoding="utf-8") as fh:
        l1 = json.load(fh)["T1"]
    with open(os.path.join(HERE, "p0b_t1_survey.json"), encoding="utf-8") as fh:
        p0b = json.load(fh)["cameras"]

    # 1 ------------------------------------------------------------------
    log("1. candidates per frame on pure clipped noise (sigma 6, 1280 x 800), after the suppression")
    rng = np.random.default_rng(7)
    P = lines.det_params()
    ks = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)
    cnt = {k: [] for k in ks}
    for f in range(3):
        img8 = np.clip(np.round(rng.normal(0, 6.0, (800, 1280))), 0, 255).astype(np.uint8)
        sc, idx, tpl = lines.matched_filter(img8.astype(np.float64), P)
        for k in ks:
            r, c = lines.local_maxima(sc, k)
            Ls = [(tpl[t][0], tpl[t][1]) for t in idx[r, c]]
            cnt[k].append(len(lines.suppress(r, c, sc[r, c], Ls, P)))
    OUT["pure_noise_candidates_per_frame"] = {str(k): float(np.mean(v)) for k, v in cnt.items()}
    log("   " + ", ".join(f"k {k}: {np.mean(v):.1f}" for k, v in cnt.items()))

    # 2 ------------------------------------------------------------------
    log("\n2. the templates' noise on T1 (cam1 frame 500) against white noise of the image's sigma")
    gain = float(l1["cam1"]["gain"])
    bg = np.load(os.path.join(t1.DATA, "T1bg", "cam1_p90.npy"))
    hp, img8 = lines.preprocess(t1.read("cam1", 500), bg, gain)
    noise = gain * lines.noise_left(hp)
    sc, idx, tpl = lines.matched_filter(img8.astype(np.float64), P)
    rows = []
    for (Lx, Ly, kx, ky, ssum, norm2, sig) in tpl:
        white = math.sqrt(max(norm2, 0.0)) * noise
        rows.append({"Lx": Lx, "Ly": Ly, "noise": sig, "white": white, "ratio": sig / white})
    OUT["template_noise_T1_cam1_500"] = rows
    sel = [r for r in rows if r["Lx"] in (0, 5, 12, 24, 34) and r["Ly"] in (0, 5, 12, 24)]
    for r in sel:
        log(f"   ({r['Lx']:4.0f}, {r['Ly']:4.0f}): {r['noise']:6.1f} against {r['white']:5.1f} (x {r['ratio']:.1f})")
    ratios = np.array([r["ratio"] for r in rows])
    log(f"   all 56 templates: ratio {ratios.min():.1f}-{ratios.max():.1f}; templates with Lx >= 24 or Ly >= 24: "
        f"{min(r['ratio'] for r in rows if r['Lx'] >= 24 or r['Ly'] >= 24):.1f}-"
        f"{max(r['ratio'] for r in rows if r['Lx'] >= 24 or r['Ly'] >= 24):.1f}")

    # 3 ------------------------------------------------------------------
    log("\n3. bounds and suppression on T1 (cam1 frame 500)")
    out3 = {}
    for name, over in (("DET", {}), ("lengths bounded by the template", {"dynamic_bounds": False}),
                       ("asymmetric suppression", {"nms_symmetric": False})):
        tm = {}
        d = lines.detect(img8, noise, lines.det_params(**over), timing=tm)
        cc = Counter(d["cls"].tolist())
        Hl, Vl = d[d["cls"] == "H"], d[d["cls"] == "V"]
        pinned = float(np.mean(np.isclose(d["Lx"], 8.0) | np.isclose(d["Ly"], 8.0)))
        out3[name] = {"n": int(len(d)), "classes": dict(cc), "H_Lx_p10_50_90": np.percentile(Hl["Lx"], [10, 50, 90]).tolist(),
                      "V_Ly_p10_50_90": np.percentile(Vl["Ly"], [10, 50, 90]).tolist(), "pinned_at_8px": pinned,
                      "seconds": tm["total_s"]}
        log(f"   {name}: {len(d)} objects ({', '.join(f'{k} {v}' for k, v in sorted(cc.items()))}); H Lx 10/50/90 % "
            + "/".join(f"{x:.1f}" for x in out3[name]["H_Lx_p10_50_90"]) + " px; V Ly "
            + "/".join(f"{x:.1f}" for x in out3[name]["V_Ly_p10_50_90"]) + f" px; pinned at 8 px: {100 * pinned:.0f} %")
    OUT["T1_bounds_suppression"] = out3

    # 4 ------------------------------------------------------------------
    log("\n4. the fit model on isolated objects (cam1's noise and texture, gain 6)")
    sn = p0b["cam1"]["noise_counts"]["median"]
    corr = 0.5 * (p0b["cam1"]["noise_correlation"]["horizontal"] + p0b["cam1"]["noise_correlation"]["vertical"])
    rho = math.exp(-1.0 / (4 * 0.9 ** 2))
    c2 = min(corr * sn ** 2 / rho, sn ** 2 - 0.25)
    w, c = math.sqrt(sn ** 2 - c2), math.sqrt(c2)
    rng = np.random.default_rng(5)
    H, W = 800, 1280
    objs, meta = [], []
    for gy in range(40, H - 40, 64):
        for gx in range(40, W - 40, 64):
            a, b = [(12, 0), (0, 6), (3, 3), (0, 0)][rng.integers(4)]
            snr = [4, 6, 10, 20][rng.integers(4)]
            u, v = gx + rng.uniform(-0.5, 0.5), gy + rng.uniform(-0.5, 0.5)
            _, pk = lines.render((H, W), np.array([(u, v, a, b, 1.0, 1.0)]))
            objs.append((u, v, a, b, 1.0, snr * sn / pk[0]))
            meta.append(snr)
    objs, meta = np.array(objs), np.array(meta)
    sig, peaks = lines.render((H, W), objs)
    tex = ndimage.gaussian_filter(rng.standard_normal(sig.shape), 0.9)
    tex *= c / tex.std()
    img = np.clip(np.round(gain * (sig + w * rng.standard_normal(sig.shape) + tex)), 0, 255).astype(np.uint8)
    out4 = {}
    for name, over in (("clipped model (DET)", {}), ("plain model", {"clipped": False})):
        d = lines.detect(img, gain * sn, lines.det_params(**over))
        st, tm_, dm, ins = lines.score_detections(np.column_stack([d["u"], d["v"]]), objs, peaks / sn)
        m = tm_ >= 0
        fl = d["flux"][tm_[m]] / gain / objs[m, 5]
        twx = np.sqrt(objs[m, 2] ** 2 / 4 + 1.0)
        twy = np.sqrt(objs[m, 3] ** 2 / 4 + 1.0)
        out4[name] = {"recall": st["recall_all"], "duplicates": st["duplicates"], "false_positives": st["false_positives"],
                      "flux_ratio_median": float(np.median(fl)), "wx_ratio_median": float(np.median(d["wx"][tm_[m]] / twx)),
                      "wy_ratio_median": float(np.median(d["wy"][tm_[m]] / twy)), "centre_errors": st["centre_errors_px"]}
        e = st["centre_errors_px"]
        log(f"   {name}: recall {100 * st['recall_all']:.1f} %, duplicates {st['duplicates']}, false positives "
            f"{st['false_positives']}; flux / true {np.median(fl):.3f}, RMS widths / true {out4[name]['wx_ratio_median']:.3f} "
            f"(x), {out4[name]['wy_ratio_median']:.3f} (y); H along/across {e['H']['along_rms']:.3f}/{e['H']['across_rms']:.3f}, "
            f"V {e['V']['along_rms']:.3f}/{e['V']['across_rms']:.3f} px")
    OUT["isolated_fit_model"] = out4

    # 5 ------------------------------------------------------------------
    log("\n5. a dense synthetic frame (1,500 objects, white noise 1.5 counts, gain 4)")
    rng = np.random.default_rng(3)
    n = 1500
    t = rng.uniform(-1, 1, n)
    objs = np.column_stack([rng.uniform(3, W - 4, n), rng.uniform(3, H - 4, n), 11 * np.abs(t - 0.3),
                            5 * np.abs(t + 0.3), rng.uniform(0.8, 1.2, n), 400 * np.exp(rng.normal(0, 0.3, n))])
    sig, peaks = lines.render((H, W), objs)
    img = np.clip(np.round(4 * (sig + rng.normal(0, 1.5, sig.shape))), 0, 255).astype(np.uint8)
    out5 = {}
    for name, over in (("DET", {}), ("lengths bounded by the template", {"dynamic_bounds": False}),
                       ("plain model", {"clipped": False})):
        tm = {}
        d = lines.detect(img, 6.0, lines.det_params(**over), timing=tm)
        st, _, _, _ = lines.score_detections(np.column_stack([d["u"], d["v"]]), objs, peaks / 1.5)
        e = st["centre_errors_px"]
        out5[name] = {"n": int(len(d)), "recall_snr_ge5": st["recall_snr_ge5"], "duplicates_per_object":
                      st["duplicates_per_object"], "false_positives": st["false_positives"], "centre_errors": e,
                      "seconds": tm["total_s"]}
        log(f"   {name}: {len(d)} objects, recall at SNR >= 5 {100 * st['recall_snr_ge5']:.1f} %, duplicates "
            f"{100 * st['duplicates_per_object']:.2f} %, false positives {st['false_positives']}; at SNR >= 5 H along/across "
            f"{e['H']['along_rms_snr_ge5']:.3f}/{e['H']['across_rms_snr_ge5']:.3f}, V "
            f"{e['V']['along_rms_snr_ge5']:.3f}/{e['V']['across_rms_snr_ge5']:.3f} px; {tm['total_s']:.1f} s")
    OUT["dense_synthetic"] = out5

    # 6 ------------------------------------------------------------------
    log("\n6. zero pixels in L1's 8-bit frames (frame 500)")
    for cam in t1.CAMS:
        bgc = np.load(os.path.join(t1.DATA, "T1bg", f"{cam}_p90.npy"))
        _, o8 = lines.preprocess(t1.read(cam, 500), bgc, float(l1[cam]["gain"]))
        z = float((o8 == 0).mean())
        OUT[f"zero_fraction_{cam}"] = z
        log(f"   {cam}: {100 * z:.1f} % of the pixels are 0")
    with open(os.path.join(HERE, "l2_variants.json"), "w", encoding="utf-8") as fh:
        json.dump(OUT, fh, indent=1, default=float)
    with open(os.path.join(HERE, "l2_variants_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
