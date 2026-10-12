"""Planning reconnaissance of 20261007/T1 (scratch, read-only): background, object count and shapes, and how far
particles move per frame.
"""
import json
import os
import sys

import cv2
import numpy as np

RAW = r"X:\Shijie Zhong\Refraction\Raw\20261007\T1"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def read(cam, k):
    with open(os.path.join(RAW, cam, "frame%06d.tif" % k), "rb") as f:
        return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32)


for cam in ("cam1", "cam2"):
    stack = np.stack([read(cam, k) for k in range(0, 1001, 40)])  # 26 frames over the recording
    bg_max = stack.max(axis=0)
    bg_p90 = np.percentile(stack, 90, axis=0)
    f0, f1, f4 = read(cam, 0), read(cam, 1), read(cam, 4)
    res = {}
    # dark objects: background minus image, then a local high-pass to remove the large dark clouds
    d = np.clip(bg_p90 - f0, 0, None)
    hp = d - cv2.blur(cv2.medianBlur(np.clip(d, 0, 255).astype(np.uint8), 31).astype(np.float32), (15, 15))
    noise = 1.4826 * np.median(np.abs(hp - np.median(hp)))
    res["noise_counts"] = float(noise)
    for k in (4, 6):
        mask = (hp > k * noise).astype(np.uint8)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[1:]
        keep = stats[:, cv2.CC_STAT_AREA] >= 6
        w, h = stats[keep, cv2.CC_STAT_WIDTH], stats[keep, cv2.CC_STAT_HEIGHT]
        horiz = (w >= 2 * h)
        vert = (h >= 2 * w)
        res["thr_%dsigma" % k] = {
            "objects": int(keep.sum()),
            "per_Mpx": float(keep.sum() / (f0.size / 1e6)),
            "horizontal_frac": float(horiz.mean()), "vertical_frac": float(vert.mean()),
            "h_len_px_pcts": np.percentile(w[horiz], [10, 50, 90]).tolist() if horiz.any() else None,
            "v_len_px_pcts": np.percentile(h[vert], [10, 50, 90]).tolist() if vert.any() else None,
            "thickness_px_median": float(np.median(np.concatenate([h[horiz], w[vert]]))) if (horiz.any() or vert.any()) else None,
        }
    # displacement per frame: local cross-correlation of 96 px windows of the high-passed images
    def hpimg(f):
        dd = np.clip(bg_p90 - f, 0, None)
        return dd - cv2.blur(dd, (25, 25))
    a0, a1, a4 = hpimg(f0), hpimg(f1), hpimg(f4)
    shifts = {"1": [], "4": []}
    W = 96
    for y in range(64, 800 - W - 64, 96):
        for x in range(64, 1280 - W - 64, 96):
            t = a0[y:y + W, x:x + W]
            for key, b in (("1", a1), ("4", a4)):
                srch = b[y - 32:y + W + 32, x - 32:x + W + 32]
                r = cv2.matchTemplate(srch, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                if mx > 0.3:
                    shifts[key].append((loc[0] - 32, loc[1] - 32, mx))
    for key in shifts:
        s = np.array(shifts[key])
        if len(s):
            res["shift_frame0_to_%s" % key] = {
                "windows": int(len(s)), "dx_pcts": np.percentile(s[:, 0], [10, 50, 90]).tolist(),
                "dy_pcts": np.percentile(s[:, 1], [10, 50, 90]).tolist(),
                "speed_px_per_frame_median": float(np.median(np.hypot(s[:, 0], s[:, 1])) / int(key)),
                "corr_median": float(np.median(s[:, 2]))}
    res["bg_p90_median"] = float(np.median(bg_p90))
    res["bg_max_minus_p90_median"] = float(np.median(bg_max - bg_p90))
    OUT[cam] = res
    cv2.imwrite(os.path.join(HERE, "t1", "%s_hp_view.png" % cam),
                np.clip(255 - hp * (255 / max(10 * noise, 1)), 0, 255).astype(np.uint8))
print(json.dumps(OUT, indent=1))
with open(os.path.join(HERE, "recon_t1_out.json"), "w") as f:
    json.dump(OUT, f, indent=1)
