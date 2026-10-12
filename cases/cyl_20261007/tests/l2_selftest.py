"""L2, step 0 (plan: Phase 4): the detector's mathematics in lines.py checked numerically before it is used.

    python tests/l2_selftest.py

1. profile(t; L, s), the pixel-integrated box (L) blurred by a Gaussian (s), against brute-force quadrature
   (2,000 points across the pixel, 4,000 along the box), on a grid of t, L, s; and its L -> 0 limit.
2. Its derivatives d/dt, d/dL, d/ds against central differences (h = 1e-5).
3. clip_expect(m, s): E[max(0, m + noise)] and its derivative against Monte Carlo (10^6 samples) and differences.
4. render(): the flux of a rendered object equals its flux parameter, for lines, ellipses and round spots, and its
   centroid is the given centre (to the window's truncation, 4 sigma + 1 px); a line's profile along it against
   the semicircle density blurred by the Gaussian (fine quadrature); a disk against a 20x supersampled disk.
5. The noise estimators on Gaussian noise (sigma 1.5 counts float, 6 counts clipped) with 0-15 % of the pixels
   covered by an exponential positive signal: noise_left (float image) and estimate_noise_clipped (8-bit).
6. fit_objects on single noiseless objects of the detector's own model, with 100 iterations: centres, model and RMS
   widths sqrt(L^2/12 + s^2) are recovered (L and s alone are nearly degenerate for short objects: with the default
   40 iterations a 2.5 px box with s 1.3 ends at L 0.09, s 1.50, 0.3 % of the peak off, flux and RMS width kept).
Writes l2_selftest_out.txt and l2_selftest.json beside itself; every check prints PASS or FAIL.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lines  # noqa: E402

LOG, OUT = [], {}


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def check(name, ok, detail):
    log(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    OUT[name] = {"pass": bool(ok), "detail": detail}


def brute_profile(t, L, s, n_px=2000, n_box=4000):
    xs = t - 0.5 + (np.arange(n_px) + 0.5) / n_px            # across the pixel
    if L < 1e-9:
        return float(np.mean(np.exp(-0.5 * (xs / s) ** 2)))
    q = -L / 2 + L * (np.arange(n_box) + 0.5) / n_box        # along the box
    S = np.exp(-0.5 * ((xs[:, None] - q[None, :]) / s) ** 2).mean(axis=1)
    return float(S.mean())


log("L2 self-test of lines.py")
# 1. profile against quadrature
worst = 0.0
for L in (0.0, 0.05, 0.5, 3.0, 12.0, 34.0):
    for s in (0.6, 1.0, 2.0):
        for t in (-20.0, -8.3, -3.0, -0.7, 0.0, 0.4, 1.0, 5.5, 17.2):
            p = float(lines.profile(t, L, s))
            b = brute_profile(t, L, s)
            worst = max(worst, abs(p - b))
check("profile vs quadrature", worst < 2e-6, f"max |P - quadrature| = {worst:.2e} (quadrature error ~1e-7)")

# 2. derivatives
worst = {"t": 0.0, "L": 0.0, "s": 0.0}
h = 1e-5
for L in (0.05, 0.5, 3.0, 12.0, 34.0):
    for s in (0.6, 1.0, 2.0):
        for t in (-20.0, -8.3, -3.0, -0.7, 0.0, 0.4, 1.0, 5.5, 17.2):
            _, dt, dL, ds = lines.profile(t, L, s, deriv=True)
            nt = (lines.profile(t + h, L, s) - lines.profile(t - h, L, s)) / (2 * h)
            nL = (lines.profile(t, L + h, s) - lines.profile(t, L - h, s)) / (2 * h)
            ns = (lines.profile(t, L, s + h) - lines.profile(t, L, s - h)) / (2 * h)
            worst["t"] = max(worst["t"], abs(float(dt - nt)))
            worst["L"] = max(worst["L"], abs(float(dL - nL)))
            worst["s"] = max(worst["s"], abs(float(ds - ns)))
check("profile derivatives vs differences", max(worst.values()) < 1e-6,
      ", ".join(f"d/d{k} {v:.1e}" for k, v in worst.items()))
# L -> 0: the derivative formulas at L = 0 (limit branch) against differences in t and s
_, dt0, dL0, ds0 = lines.profile(0.7, 0.0, 1.1, deriv=True)
nt0 = (lines.profile(0.7 + h, 0.0, 1.1) - lines.profile(0.7 - h, 0.0, 1.1)) / (2 * h)
ns0 = (lines.profile(0.7, 0.0, 1.1 + h) - lines.profile(0.7, 0.0, 1.1 - h)) / (2 * h)
check("L = 0 branch derivatives", abs(float(dt0 - nt0)) < 1e-8 and abs(float(ds0 - ns0)) < 1e-8,
      f"d/dt {abs(float(dt0 - nt0)):.1e}, d/ds {abs(float(ds0 - ns0)):.1e}")
cont = abs(float(lines.profile(0.7, 1e-4, 1.1) - lines.profile(0.7, 0.0, 1.1)))
check("continuity at L -> 0", cont < 1e-8, f"|P(L = 1e-4) - P(L = 0)| = {cont:.1e}")

# 3. clip_expect
rng = np.random.default_rng(1)
worst_mc = 0.0
worst_d = 0.0
for m in (-6.0, -2.0, -0.5, 0.0, 0.7, 3.0, 10.0):
    e, d = lines.clip_expect(m, 2.0)
    mc = float(np.maximum(0.0, m + rng.normal(0, 2.0, 10 ** 6)).mean())
    worst_mc = max(worst_mc, abs(float(e) - mc))
    nd = (lines.clip_expect(m + 1e-5, 2.0)[0] - lines.clip_expect(m - 1e-5, 2.0)[0]) / 2e-5
    worst_d = max(worst_d, abs(float(d - nd)))
check("clip_expect vs Monte Carlo", worst_mc < 0.01, f"max |E - MC| = {worst_mc:.4f} (MC error ~0.003)")
check("clip_expect derivative", worst_d < 1e-6, f"max |d - difference| = {worst_d:.1e}")

# 4. render: flux and centroid
objs = np.array([(100.3, 60.7, 15.0, 0.0, 1.0, 500.0), (200.6, 60.2, 0.0, 7.0, 1.1, 300.0),
                 (300.1, 60.9, 4.0, 3.0, 1.0, 400.0), (400.45, 60.5, 0.0, 0.0, 1.2, 250.0)])
img, peaks = lines.render((120, 500), objs)
worst_f = worst_c = 0.0
for (u, v, a, b, s0, flux) in objs:
    c0, r0 = int(round(u)), int(round(v))
    patch = img[r0 - 30:r0 + 31, c0 - 30:c0 + 31]
    f = patch.sum()
    rr, cc = np.mgrid[r0 - 30:r0 + 31, c0 - 30:c0 + 31]
    worst_f = max(worst_f, abs(f / flux - 1))
    worst_c = max(worst_c, abs((patch * cc).sum() / f - u), abs((patch * rr).sum() / f - v))
check("render flux and centroid", worst_f < 1e-5 and worst_c < 1e-5,
      f"max flux error {worst_f:.1e} (relative), max centroid error {worst_c:.1e} px")
a, s0, u = 15.0, 1.0, 100.3
cols = np.arange(80, 121)
q = np.linspace(-a, a, 200001)
wq = np.sqrt(np.clip(1 - (q / a) ** 2, 0, None))
wq /= wq.sum()
ideal = np.array([(wq * (special.ndtr((c + 0.5 - u - q) / s0) - special.ndtr((c - 0.5 - u - q) / s0))).sum() for c in cols])
img1, _ = lines.render((120, 200), np.array([(u, 60.0, a, 0.0, s0, 1.0)]))
rip = float(np.abs(img1.sum(axis=0)[cols] - ideal).max() / ideal.max())
ss, R0 = 20, 4.0
yy, xx = np.mgrid[0:60 * ss, 0:60 * ss]
xx = (xx + 0.5) / ss - 0.5
yy = (yy + 0.5) / ss - 0.5
from scipy import ndimage
disk = ndimage.gaussian_filter((((xx - 30.2) ** 2 + (yy - 29.7) ** 2) <= R0 ** 2).astype(float), 0.8 * ss)
bf = disk.reshape(60, ss, 60, ss).mean(axis=(1, 3))
bf /= bf.sum()
img2, _ = lines.render((60, 60), np.array([(30.2, 29.7, R0, R0, 0.8, 1.0)]))
dd = float(np.abs(img2 - bf).max() / bf.max())
check("render line profile and disk", rip < 1e-3 and dd < 2e-3,
      f"line (a = 15 px) vs semicircle * Gaussian: {rip:.1e} of the peak; disk (r = 4 px) vs 20x supersampling: {dd:.1e}")
lp = img[61, 86:116]
log(f"  (info) a = 15 px line, row through it, columns 86-115: " + " ".join(f"{x:.1f}" for x in lp))

# 5. noise estimators
res, resf = [], []
for frac in (0.0, 0.05, 0.10, 0.15):
    x = rng.normal(0.0, 6.0, (800, 1280))
    mask = rng.random(x.shape) < frac
    x[mask] += rng.exponential(25.0, mask.sum())
    res.append((frac, lines.estimate_noise_clipped(np.clip(np.round(x), 0, 255).astype(np.uint8))))
    xf = rng.normal(0.0, 1.5, (800, 1280))
    xf[mask] += rng.exponential(6.0, mask.sum())
    resf.append((frac, lines.noise_left(xf), lines.robust_sigma(xf)))
worst_f = max(abs(e / 1.5 - 1) for _, e, _ in resf)
check("noise_left (float image)", worst_f < 0.04,
      "true 1.5: " + ", ".join(f"{100 * f:.0f} % covered -> {e:.3f} (MAD {m:.3f})" for f, e, m in resf))
worst_n = max(abs(e / 6.0 - 1) for _, e in res)
check("estimate_noise_clipped (8-bit fallback)", worst_n < 0.06,
      "true 6.0: " + ", ".join(f"{100 * f:.0f} % covered -> {e:.2f}" for f, e in res))

# 6. fit_objects recovers the detector's own model
P = lines.det_params(clipped=False, fit_iters=100)
true = np.array([[40.0, 50.3, 40.6, 22.0, 0.05, 1.1, 0.9, 0.0],
                 [60.0, 150.7, 40.2, 0.05, 9.0, 1.0, 1.2, 0.0],
                 [80.0, 250.2, 40.8, 2.5, 2.0, 1.3, 1.0, 0.0]])
H, W = 90, 300
I = lines.render_models((H, W), true, P)
start = true.copy()
start[:, 0] *= 0.7
start[:, 1] += 0.4
start[:, 2] -= 0.3
start[:, 3] = np.maximum(start[:, 3] * 0.8, 0.5)
start[:, 4] = np.maximum(start[:, 4] * 0.8, 0.5)
start[:, 5:7] = 1.0
fit, chi2, npx, conv = lines.fit_objects(I, start, P, noise=1.0)
err = np.abs(fit - true)
Ifit = lines.render_models((H, W), fit, P)
wtrue = np.sqrt(true[:, 3:5] ** 2 / 12 + true[:, 5:7] ** 2)
wfit = np.sqrt(fit[:, 3:5] ** 2 / 12 + fit[:, 5:7] ** 2)
model_err = float(np.abs(Ifit - I).max() / I.max())
w_err = float(np.abs(wfit / wtrue - 1).max())
check("fit_objects recovers its model (noiseless)", float(err[:, 1:3].max()) < 1e-4 and model_err < 1e-3 and w_err < 2e-3,
      f"max |du|,|dv| {err[:, 1:3].max():.1e} px, model {model_err:.1e} of the peak, RMS widths {w_err:.1e}; "
      "(Lx, sx | Ly, sy) fitted: " + "; ".join(f"{p[3]:.2f}, {p[5]:.2f} | {p[4]:.2f}, {p[6]:.2f}" for p in fit))

OUT["all_pass"] = all(v["pass"] for v in OUT.values() if isinstance(v, dict))
log(f"\nall checks pass: {OUT['all_pass']}")
with open(os.path.join(HERE, "l2_selftest_out.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
with open(os.path.join(HERE, "l2_selftest.json"), "w", encoding="utf-8") as fh:
    json.dump(OUT, fh, indent=1)
