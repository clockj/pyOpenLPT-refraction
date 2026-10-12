"""How does OpenLPT's tracer detector do on stretched particle images? (scratch)

Synthetic 8-bit images: elliptical Gaussian particles (pixel-integrated by 7x7 supersampling) on a 40 px grid,
same total flux for every shape, background 10, read noise 3 counts. Detectors:
  A  the C++ ObjectFinder2D as it is (4-neighbour local max + 3-point log-parabola in x and y)
  B  the same C++ finder on the image smoothed by a Gaussian of 1 px first
  C  a 2-D elliptical Gaussian least-squares fit (7 parameters) seeded by A, window from the true shape
"""
import json
import math
import os
import sys

import numpy as np
from scipy import ndimage, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyd"))
import pyopenlpt as lpt  # noqa: E402

RNG = np.random.default_rng(3)
SIZE, STEP, SS = 1000, 40, 7
A0, S0 = 180.0, 0.8           # peak of the round particle and its sigma
BG, NOISE = 10.0, 3.0


def render(shape):
    sx, sy, th = shape
    A = A0 * S0 * S0 / (sx * sy)  # same total flux
    img = np.zeros((SIZE, SIZE))
    truth = []
    c, s = math.cos(th), math.sin(th)
    off = (np.arange(SS) + 0.5) / SS - 0.5
    for gy in range(STEP, SIZE - STEP + 1, STEP):
        for gx in range(STEP, SIZE - STEP + 1, STEP):
            xc, yc = gx + RNG.uniform(-0.5, 0.5), gy + RNG.uniform(-0.5, 0.5)
            truth.append((xc, yc))
            hw = int(math.ceil(4 * max(sx, sy))) + 1
            cols = np.arange(int(xc) - hw, int(xc) + hw + 1)
            rows = np.arange(int(yc) - hw, int(yc) + hw + 1)
            X = cols[None, :, None, None] + off[None, None, None, :] - xc
            Y = rows[:, None, None, None] + off[None, None, :, None] - yc
            xx = X * c + Y * s
            yy = -X * s + Y * c
            val = A * np.exp(-0.5 * (xx ** 2 / sx ** 2 + yy ** 2 / sy ** 2)).mean(axis=(2, 3))
            img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] += val
    img += BG + RNG.normal(0, NOISE, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8), np.array(truth)


def cpp_find(img_u8):
    cfg = lpt.TracerConfig()
    cfg._radius_obj = 2.0
    cfg._min_obj_int = 30
    objs = lpt.ObjectFinder2D().findObject2D(lpt.Image.from_numpy(img_u8), cfg)
    return np.array([[o._pt_center[0], o._pt_center[1]] for o in objs]) if objs else np.zeros((0, 2))


def gauss_fit(img, x0, y0, shape):
    sx, sy, th = shape
    hx = int(math.ceil(3 * max(sx, sy))) + 1
    cx, cy = int(round(x0)), int(round(y0))
    rows, cols = np.mgrid[cy - hx:cy + hx + 1, cx - hx:cx + hx + 1]
    z = img[cy - hx:cy + hx + 1, cx - hx:cx + hx + 1].astype(float)

    def model(p):
        A, xc, yc, a, b, t, bg = p
        X, Y = cols - xc, rows - yc
        xx = X * math.cos(t) + Y * math.sin(t)
        yy = -X * math.sin(t) + Y * math.cos(t)
        return A * np.exp(-0.5 * (xx ** 2 / a ** 2 + yy ** 2 / b ** 2)) + bg

    p0 = [z.max() - BG, x0, y0, sx, sy, th, BG]
    r = optimize.least_squares(lambda p: (model(p) - z).ravel(), p0, method="lm")
    return r.x[1], r.x[2]


def score(det, truth, shape):
    rad = max(3 * max(shape[0], shape[1]), 2.0)
    n_found = n_dup = 0
    errs = []
    nearest = []
    for (xc, yc) in truth:
        if len(det) == 0:
            nearest.append(None)
            continue
        d = np.hypot(det[:, 0] - xc, det[:, 1] - yc)
        k = int(np.sum(d < rad))
        if k >= 1:
            n_found += 1
            j = int(np.argmin(d))
            errs.append((det[j, 0] - xc, det[j, 1] - yc))
            nearest.append(j)
        else:
            nearest.append(None)
        if k >= 2:
            n_dup += 1
    e = np.array(errs)
    return {
        "found": n_found / len(truth), "dup": n_dup / len(truth),
        "x_bias": float(e[:, 0].mean()), "x_rms": float(np.sqrt((e[:, 0] ** 2).mean())),
        "y_bias": float(e[:, 1].mean()), "y_rms": float(np.sqrt((e[:, 1] ** 2).mean())),
    }, nearest


CASES = {
    "round 0.8x0.8": (0.8, 0.8, 0.0),
    "1.6x0.8": (1.6, 0.8, 0.0),
    "2.4x0.8": (2.4, 0.8, 0.0),
    "3.2x0.8": (3.2, 0.8, 0.0),
    "2.4x0.8 rotated 20 deg": (2.4, 0.8, math.radians(20)),
}
out = {}
for name, shape in CASES.items():
    img, truth = render(shape)
    detA = cpp_find(img)
    sm = np.clip(np.round(ndimage.gaussian_filter(img.astype(float), 1.0)), 0, 255).astype(np.uint8)
    detB = cpp_find(sm)
    sA, nearA = score(detA, truth, shape)
    sB, _ = score(detB, truth, shape)
    # C: fit seeded by A's nearest detection
    detC = []
    for (xc, yc), j in zip(truth, nearA):
        if j is None:
            continue
        try:
            detC.append(gauss_fit(img, detA[j, 0], detA[j, 1], shape))
        except Exception:
            pass
    sC, _ = score(np.array(detC), truth, shape)
    out[name] = {"A_cpp_as_is": sA, "B_cpp_on_smoothed": sB, "C_2d_gauss_fit": sC, "n": len(truth)}
    print(name, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v
                            for k, v in out[name].items()}))
with open(os.path.join(HERE, "probe_detect_out.json"), "w") as f:
    json.dump(out, f, indent=1)
