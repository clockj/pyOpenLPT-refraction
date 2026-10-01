"""T2: the dot detector on synthetic images with known centres.

The dots are rendered to look like the real ones (T1b profiles):
  * an elliptical outline, a = 36, b = 27.5 px;
  * a dark rim inside the edge, darker at the top than at the bottom, and paler at the sides;
  * horizontal streaks inside;
  * anisotropic Gaussian blur in three states measured on the real images:
      "z0"  sigma_u = 4.7, sigma_v = 1.5   (left/right edges blurred, top/bottom sharp)
      "z18" sigma_u = 2.8, sigma_v = 2.0
      "z38" sigma_u = 3.8, sigma_v = 4.6
  * background 115 with a gentle gradient, Gaussian noise sigma = 2 grey levels, 8-bit quantisation, a few dirt
    specks.

Rendering is supersampled 4x before area averaging, so the edges carry the right sub-pixel information.
Each state is run with the asymmetric rim and with a symmetric one, to separate the rim's effect.

Passes (plan, section 5) if the primary estimator has |bias| < 0.02 px and RMS < 0.05 px per axis in every state.
Writes t2_out.txt beside this script.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import dots as D                              # noqa: E402

LOG = []
SS = 4                                        # supersampling factor


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def render(centres, W, H, rng, sig_u, sig_v, rim, a=36.0, b=27.5, d_int=0.28, w_rim=2.5, specks=12):
    """8-bit image of dark dots at `centres` (pixel coordinates, pixel centres at integers)."""
    d_top, d_bot, d_side = rim
    Ws, Hs = W * SS, H * SS
    dark = np.zeros((Hs, Ws), np.float32)
    # supersampled pixel centres: fine pixel q covers [q/SS - 0.5, (q+1)/SS - 0.5) in coarse coordinates
    for (cu, cv) in centres:
        x0, x1 = int((cu - a - 8) * SS), int((cu + a + 8) * SS)
        y0, y1 = int((cv - b - 8) * SS), int((cv + b + 8) * SS)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, Ws), min(y1, Hs)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        X = (xx + 0.5) / SS - 0.5 - cu
        Y = (yy + 0.5) / SS - 0.5 - cv
        rho = np.sqrt((X / a) ** 2 + (Y / b) ** 2)
        phi = np.arctan2(Y, X)
        re = 1.0 / np.sqrt((np.cos(phi) / a) ** 2 + (np.sin(phi) / b) ** 2)
        r_in = (1.0 - rho) * re
        s = np.sin(phi)
        d_rim = d_side + (d_top - d_side) * np.clip(-s, 0, 1) ** 2 + (d_bot - d_side) * np.clip(s, 0, 1) ** 2
        streak = 1.0 + 0.10 * np.sin(2 * np.pi * (Y + cv) / 3.1)
        val = d_int * streak + (d_rim - d_int * streak) * np.exp(-np.clip(r_in, 0, None) / w_rim)
        val = np.where(rho <= 1.0, val, 0.0)
        dark[y0:y1, x0:x1] = np.maximum(dark[y0:y1, x0:x1], val.astype(np.float32))
    dark = cv2.GaussianBlur(dark, (0, 0), sigmaX=sig_u * SS, sigmaY=sig_v * SS)
    dark = cv2.resize(dark, (W, H), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = 115.0 * (1.0 + 0.04 * (xx / W - 0.5) - 0.03 * (yy / H - 0.5))
    img = bg * (1.0 - dark)
    for _ in range(specks):                   # small dark specks anywhere, some on dots
        su, sv = rng.uniform(0, W), rng.uniform(0, H)
        rr = rng.uniform(0.8, 2.0)
        m = (xx - su) ** 2 + (yy - sv) ** 2 < rr * rr
        img[m] *= rng.uniform(0.4, 0.7)
    img = img + rng.normal(0, 2.0, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.float32)


def run_state(name, sig_u, sig_v, rim, det, rng, n_img=5):
    W, H = 1280, 800
    errs = {k: [] for k in ("halfmax", "gradient", "iso", "sat")}
    found, total = 0, 0
    for _ in range(n_img):
        cen = []
        for i in range(7):
            for j in range(6):
                cen.append((120 + 150 * i + rng.uniform(-0.5, 0.5) + rng.uniform(0, 20),
                            75 + 117 * j + rng.uniform(-0.5, 0.5) + rng.uniform(0, 15)))
        cen = np.array(cen)
        img = render(cen, W, H, rng, sig_u, sig_v, rim)
        r = D.detect(img, det)
        ok = [d for d in r["dots"] if d["ok"]]
        P = np.array([[d["u"], d["v"]] for d in ok])
        total += len(cen)
        for c in cen:
            dd = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
            k = int(np.argmin(dd))
            if dd[k] > 10:
                continue
            found += 1
            d = ok[k]
            errs["halfmax"].append((d["u"] - c[0], d["v"] - c[1]))
            errs["gradient"].append((d["u_grad"] - c[0], d["v_grad"] - c[1]))
            errs["iso"].append((d["u_iso"] - c[0], d["v_iso"] - c[1]))
            errs["sat"].append((d["u_sat"] - c[0], d["v_sat"] - c[1]))
    say(f"state {name}: sigma_u {sig_u}, sigma_v {sig_v}, rim top/bottom/side {rim}; found {found}/{total}")
    worst = 0.0
    for k, e in errs.items():
        e = np.array(e, float)
        e = e[np.all(np.isfinite(e), axis=1)]
        bias, rms = e.mean(axis=0), np.sqrt((e ** 2).mean(axis=0))
        std = e.std(axis=0)
        say(f"   {k:9s} bias ({bias[0]:+.3f}, {bias[1]:+.3f})  std ({std[0]:.3f}, {std[1]:.3f})  "
            f"rms ({rms[0]:.3f}, {rms[1]:.3f})  max |e| {np.abs(e).max():.3f}  n {len(e)}")
        if k == "halfmax":
            worst = max(worst, float(np.abs(bias).max()) / 0.02, float(rms.max()) / 0.05)
    return worst, found == total


def main():
    cfg = case.load_config()
    det = cfg["detect"]
    rng = np.random.default_rng(20260930)
    say("T2 detector on synthetic dots (truth known); errors in px, estimate minus truth")
    states = {"z0": (4.7, 1.5), "z18": (2.8, 2.0), "z38": (3.8, 4.6)}
    rims = {"asym": (0.85, 0.70, 0.45), "sym": (0.78, 0.78, 0.45)}
    worst, all_found = 0.0, True
    for sname, (su, sv) in states.items():
        for rname, rim in rims.items():
            w, f = run_state(f"{sname}/{rname}", su, sv, rim, det, rng)
            worst = max(worst, w)
            all_found &= f
    # how the synthetic profiles compare with the real ones (T1b): peak darkness at top/bottom/sides after blur
    say("")
    say(f"worst primary metric relative to its threshold: {worst:.2f} (<= 1 passes)")
    say("RESULT: " + ("PASS" if worst <= 1.0 and all_found else "FAIL"))
    with open(os.path.join(HERE, "t2_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
