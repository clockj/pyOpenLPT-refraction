"""L1 (plan: Phase 4): shadowgraph preprocessing for T1, checked on a synthetic recording and on T1.

    python tests/l1_preprocess.py          (after p0b_t1_survey.py)

lines.preprocess: signal = background - frame (background: the per-pixel 90th percentile over the recording),
hp = signal - median_31(signal), out = clip(round(gain * hp), 0, 255).

1. Synthetic recording (201 frames, 640 x 512 px, the density of T1):
   - background 107 counts with a smooth 3 % illumination pattern and a 0.5 % fixed pixel pattern; four dark clouds
     (Gaussian blobs, sigma 40-80 px, 3-10 counts deep) that drift by 1 px per frame (so that the per-pixel
     background does not hold them and the high-pass has to);
   - particles: the astigmatic family (a = 7.5 |t - 0.3|, b = 4 |t + 0.3| px for a depth t uniform in [-1, 1]: H
     lines near t = -0.3, V lines near t = 0.3), blur sigma0 0.8-1.2 px, flux log-normal (median 160, 30 %), each
     moving at 0.25-0.75 px per frame; a particle darkens the background in proportion to the local brightness;
   - noise sigma^2 = 0.8^2 + 0.017 I (1.57 counts at 107), 8-bit.
   Frame 100 preprocessed with the background of all 201 frames: each particle's recovered peak and flux (hp summed
   over its footprint) against the truth (the darkening it caused), by length; the noise lines.noise_left reads
   against the true noise; the flatness of hp away from particles (64 px tile means against noise / sqrt(n)); the
   clouds' residue.
2. T1: the gain.  From P0b's object peaks: gain = the largest integer for which the 99.9th percentile of the peak
   hp stays at or below 230 counts after the gain, at least 2.  Reported with the noise after the gain and the
   fraction of saturated pixels on every 50th frame.
Writes l1_preprocess_out.txt and l1_preprocess.json beside itself, and l1_synthetic.png (truth, frame, hp).
"""
import json
import math
import os
import sys

import cv2
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import lines  # noqa: E402
import t1  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def astig_population(n, rng, shape_hw, alpha=7.5, beta=4.0, tx=0.3, ty=-0.3, flux_med=160.0, flux_spread=0.3):
    H, W = shape_hw
    t = rng.uniform(-1, 1, n)
    a = alpha * np.abs(t - tx)
    b = beta * np.abs(t - ty)
    s0 = rng.uniform(0.8, 1.2, n)
    flux = flux_med * np.exp(rng.normal(0, flux_spread, n))
    u = rng.uniform(5, W - 6, n)
    v = rng.uniform(5, H - 6, n)
    return np.column_stack([u, v, a, b, s0, flux]), t


def synthetic_recording(rng, H=512, W=640, n_frames=201, density_per_mpx=1500):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    illum = 1.0 + 0.03 * np.cos(2 * np.pi * xx / 900 + 0.4) * np.cos(2 * np.pi * yy / 700 - 0.2)
    fixed = 1.0 + 0.005 * rng.standard_normal((H, W))
    B0 = 107.0
    clouds = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(40, 80), rng.uniform(3, 10),
               rng.uniform(0, 2 * np.pi)) for _ in range(4)]
    n = int(round(density_per_mpx * H * W / 1e6))
    objs, _ = astig_population(n, rng, (H, W))
    ang = rng.uniform(0, 2 * np.pi, n)
    spd = rng.uniform(0.25, 0.75, n)
    vel = np.column_stack([spd * np.cos(ang), spd * np.sin(ang)])
    frames, truths = [], []
    for f in range(n_frames):
        cl = np.zeros((H, W))
        for (cx, cy, s, d, th) in clouds:
            cx2, cy2 = cx + 1.0 * f * math.cos(th), cy + 1.0 * f * math.sin(th)
            cl += d * np.exp(-0.5 * ((xx - cx2) ** 2 + (yy - cy2) ** 2) / s ** 2)
        bright = B0 * illum * fixed - cl
        o = objs.copy()
        o[:, 0] = np.mod(objs[:, 0] + vel[:, 0] * f - 5, W - 10) + 5
        o[:, 1] = np.mod(objs[:, 1] + vel[:, 1] * f - 5, H - 10) + 5
        dark, _ = lines.render((H, W), o)
        dark *= bright / B0
        I = bright - dark
        noise = rng.standard_normal((H, W)) * np.sqrt(0.8 ** 2 + 0.017 * np.maximum(I, 0))
        frames.append(np.clip(np.round(I + noise), 0, 255).astype(np.uint8))
        truths.append((o, dark, cl, bright))
    return frames, truths


def main():
    rng = np.random.default_rng(11)
    out = {}
    log("L1: shadowgraph preprocessing (lines.preprocess)")
    # ---------------------------------------------------------------- 1. synthetic
    log("\n1. synthetic recording: 201 frames, 640 x 512 px, the density of T1 (about 1,500 objects per Mpx)")
    frames, truths = synthetic_recording(rng)
    stack = np.stack(frames)
    bg = lines.background(stack.astype(np.float32), 90)
    f = 100
    o, dark, cl, bright = truths[f]
    hp = lines.highpass(frames[f], bg)
    true_noise = float(np.sqrt(0.8 ** 2 + 0.017 * 107.0 + 1.0 / 12.0))   # with the rounding to whole counts
    sn = lines.noise_left(hp)
    log(f"  noise: true {true_noise:.2f} counts at the background (with the rounding); noise_left(hp) {sn:.2f}; MAD "
        f"{lines.robust_sigma(hp):.2f}; background level of hp (noise_level) {lines.noise_level(hp):+.2f}")
    # per-particle recovery
    rows = []
    for k, (u, v, a, b, s0, flux) in enumerate(o):
        r0, c0 = int(round(v)), int(round(u))
        hx, hy = int(math.ceil(a + 3 * s0)) + 1, int(math.ceil(b + 3 * s0)) + 1
        if r0 - hy < 0 or c0 - hx < 0 or r0 + hy >= hp.shape[0] or c0 + hx >= hp.shape[1]:
            continue
        sl = (slice(r0 - hy, r0 + hy + 1), slice(c0 - hx, c0 + hx + 1))
        # the particle alone (truth), and what hp holds there minus the other particles' truth
        alone, _ = lines.render(hp.shape, np.array([(u, v, a, b, s0, flux)]))
        alone = alone[sl] * (bright[sl] / 107.0)
        others = dark[sl] - alone
        rec = hp[sl] - others
        rows.append((max(a, b), alone.max(), rec[alone > 0.25 * alone.max()].max() if (alone > 0.25 * alone.max()).any() else np.nan,
                     alone.sum(), rec.sum()))
    rows = np.array(rows)
    ratio_peak = rows[:, 2] / rows[:, 1]
    ratio_flux = rows[:, 4] / rows[:, 3]
    bins = [(0, 4), (4, 8), (8, 12), (12, 20)]
    res_bins = []
    for lo, hi in bins:
        s = (rows[:, 0] >= lo) & (rows[:, 0] < hi) & (rows[:, 1] > 6 * true_noise)
        if s.sum() >= 5:
            res_bins.append({"semi_axis_px": [lo, hi], "n": int(s.sum()),
                             "flux_ratio_median": float(np.median(ratio_flux[s])),
                             "flux_ratio_p10_p90": np.percentile(ratio_flux[s], [10, 90]).tolist()})
    out["synthetic"] = {"n_particles": int(len(o)), "noise_true": true_noise, "noise_left": sn,
                        "flux_recovery_by_size": res_bins}
    log("  recovered flux / true flux (hp summed over each particle's footprint, the other particles' truth "
        "removed), particles with peak > 6 sigma:")
    for r in res_bins:
        log(f"    semi-axis {r['semi_axis_px'][0]}-{r['semi_axis_px'][1]} px: n {r['n']}, median "
            f"{r['flux_ratio_median']:.3f} (10-90 %: {r['flux_ratio_p10_p90'][0]:.3f}-{r['flux_ratio_p10_p90'][1]:.3f})")
    # flatness away from particles, and the clouds' residue
    m = dark > 0.3 * true_noise
    m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    tiles = []
    for r0 in range(0, hp.shape[0] - 63, 64):
        for c0 in range(0, hp.shape[1] - 63, 64):
            sub = hp[r0:r0 + 64, c0:c0 + 64][~m[r0:r0 + 64, c0:c0 + 64]]
            if sub.size > 400:
                tiles.append((sub.mean(), sub.size))
    tm = np.array([t[0] for t in tiles])
    exp_sd = true_noise / math.sqrt(np.median([t[1] for t in tiles]))
    sig_raw = (bg - frames[f]).astype(np.float64)
    tiles_raw = []
    for r0 in range(0, hp.shape[0] - 63, 64):
        for c0 in range(0, hp.shape[1] - 63, 64):
            sub = sig_raw[r0:r0 + 64, c0:c0 + 64][~m[r0:r0 + 64, c0:c0 + 64]]
            if sub.size > 400:
                tiles_raw.append(sub.mean())
    out["synthetic"]["flatness"] = {"hp_tile_mean_std": float(tm.std()), "hp_tile_mean_mean": float(tm.mean()),
                                    "expected_std": float(exp_sd),
                                    "before_highpass_tile_mean_std": float(np.std(tiles_raw)),
                                    "clouds_depth_max_counts": float(cl.max())}
    log(f"  flatness: 64 px tile means of hp away from particles scatter {tm.std():.3f} counts about {tm.mean():+.3f} "
        f"(noise alone: {exp_sd:.3f}); before the high-pass (background - frame): {np.std(tiles_raw):.3f} "
        f"(drifting clouds up to {cl.max():.1f} counts deep)")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].imshow(dark[150:350, 200:500], cmap="gray_r")
    ax[0].set_title("synthetic: the particles' darkening (truth)")
    ax[1].imshow(frames[f][150:350, 200:500], cmap="gray")
    ax[1].set_title("synthetic frame 100")
    ax[2].imshow(hp[150:350, 200:500], cmap="gray_r", vmin=-3, vmax=20)
    ax[2].set_title("hp (background - frame - median 31)")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "l1_synthetic.png"), dpi=100)

    # ---------------------------------------------------------------- 2. T1: the gain
    log("\n2. T1: the gain of the 8-bit output")
    with open(os.path.join(HERE, "p0b_t1_survey.json"), encoding="utf-8") as fh:
        p0b = json.load(fh)
    out["T1"] = {}
    for cam in t1.CAMS:
        c = p0b["cameras"][cam]
        p999 = c["peak_percentiles_4sigma"][3]
        gain = max(2, int(math.floor(230.0 / p999)))
        bgc = np.load(os.path.join(t1.DATA, "T1bg", f"{cam}_p90.npy"))
        sat, noise_out = [], []
        for k in t1.frames()[::50]:
            hpk, o8 = lines.preprocess(t1.read(cam, k), bgc, gain)
            sat.append(float((o8 == 255).mean()))
            noise_out.append(gain * lines.noise_left(hpk))
        out["T1"][cam] = {"peak_p999_counts": p999, "gain": gain, "noise_out_counts": float(np.median(noise_out)),
                          "saturated_fraction_max": float(max(sat)), "saturated_fraction_median": float(np.median(sat)),
                          "noise_hp_counts": c["noise_counts"]["median"]}
        log(f"  {cam}: object peaks 99.9 % {p999:.1f} counts -> gain {gain}; noise after the gain "
            f"{np.median(noise_out):.1f} counts; saturated pixels {100 * np.median(sat):.4f} % (max {100 * max(sat):.4f} %)")
    with open(os.path.join(HERE, "l1_preprocess.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(HERE, "l1_preprocess_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
