"""P0b (plan: Phase 0): the tracer recording T1 of run 20261008, read-only: what the particles' images look like on
each camera.

    python tests/p0b_t1_survey.py

1. Each camera's background: the per-pixel median of 63 frames (every 16th), kept in data/T1bg/cam<k>.npy
   (git-ignored) for the later steps.
2. On 5 frames (100, 300, 500, 700, 900): the difference d = background - image (dark particles positive), its
   noise (1.4826 x MAD), and its local maxima above 5 noise after a Gaussian smoothing of 1 px (cams 1-2) or 1.5 px
   (cam3, 1.48x smaller pixels): their number per frame.
3. The shape of every such image: second moments of d (clipped at 0) in a window of 15 px (cam3 23 px) about the
   peak, after subtracting the window's 20th percentile.  sigma_u and sigma_v (px), the elongation
   e = log2(sigma_u / sigma_v): about 0 for a round dot, > 0.8 for a horizontal line, < -0.8 for a vertical one.
   Reported for the strongest third (peak above 10 noise) and for all.
4. p0b_t1_crops.png: a 200 px crop (cam3 300 px) of each camera's frame 500, raw and d, and the 16 strongest images.
The same statistics for 7 Oct's T1 (frames 100 and 500, cams 1-2), for comparison, if its local copy exists.
Writes p0b_t1_survey_out.txt and .json beside itself.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = os.path.join(CASE_DIR, "data")
FRAMES = (100, 300, 500, 700, 900)
OLD_DATA = os.path.join(os.path.dirname(CASE_DIR), "cyl_20261007", "data")


def t1_path(cam, frame, root=None):
    return case.image_path(CFG, cam, frame, run="T1", root=root)


def background(cam, log):
    path = os.path.join(DATA, "T1bg", f"cam{cam}.npy")
    if os.path.exists(path):
        return np.load(path)
    stack = np.stack([case.read_image(t1_path(cam, f)).astype(np.float32) for f in range(0, 1001, 16)])
    bg = np.median(stack, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, bg)
    log(f"  cam{cam}: background from {len(stack)} frames -> {path}")
    return bg


def shapes(d, noise, smooth, half):
    sm = cv2.GaussianBlur(d, (0, 0), smooth)
    k = 2 * int(round(2 * smooth)) + 1
    peak = (sm == cv2.dilate(sm, np.ones((k, k), np.uint8))) & (sm > 5 * noise)
    peak[:half + 1, :] = False
    peak[-half - 1:, :] = False
    peak[:, :half + 1] = False
    peak[:, -half - 1:] = False
    ys, xs = np.nonzero(peak)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    out = []
    for y, x in zip(ys, xs):
        w = d[y - half:y + half + 1, x - half:x + half + 1]
        w = np.clip(w - np.percentile(w, 20), 0, None)
        s = float(w.sum())
        if s <= 0:
            continue
        mu = float((w * xx).sum() / s)
        mv = float((w * yy).sum() / s)
        su = float(np.sqrt(max((w * (xx - mu) ** 2).sum() / s, 1e-6)))
        sv = float(np.sqrt(max((w * (yy - mv) ** 2).sum() / s, 1e-6)))
        out.append((float(sm[y, x]) / noise, su, sv, float(np.log2(su / sv))))
    return np.array(out) if out else np.zeros((0, 4))


def describe(name, A, log):
    if len(A) == 0:
        log(f"    {name}: none")
        return {}
    e = A[:, 3]
    res = {"n": int(len(A)), "sigma_u_median": float(np.median(A[:, 1])), "sigma_v_median": float(np.median(A[:, 2])),
           "share_round": float(np.mean(np.abs(e) < 0.4)), "share_horizontal": float(np.mean(e > 0.8)),
           "share_vertical": float(np.mean(e < -0.8)), "e_p10_p50_p90": [float(np.percentile(e, q)) for q in (10, 50, 90)]}
    log(f"    {name}: {res['n']} images; sigma_u {res['sigma_u_median']:.2f}, sigma_v {res['sigma_v_median']:.2f} px "
        f"(medians); round (|e| < 0.4) {100 * res['share_round']:.0f} %, horizontal (e > 0.8) "
        f"{100 * res['share_horizontal']:.0f} %, vertical (e < -0.8) {100 * res['share_vertical']:.0f} %; e p10/50/90 "
        + "/".join(f"{v:.2f}" for v in res["e_p10_p50_p90"]))
    return res


def main():
    log = case.Log(os.path.join(HERE, "p0b_t1_survey"))
    log(f"P0b: run 20261008/T1 ({os.path.join(CFG['raw_root'], 'T1')}), read-only")
    out = {"cameras": {}}
    fig, ax = plt.subplots(3, 3, figsize=(16, 16))
    for j, cam in enumerate(case.cam_ids(CFG)):
        bg = background(cam, log)
        smooth, half, crop = (1.5, 11, 150) if cam == 3 else (1.0, 7, 100)
        per, allA = [], []
        for f in FRAMES:
            img = case.read_image(t1_path(cam, f)).astype(np.float32)
            d = bg - img
            noise = 1.4826 * float(np.median(np.abs(d - np.median(d))))
            A = shapes(d, noise, smooth, half)
            per.append({"frame": f, "noise": noise, "n_5sigma": int(len(A)), "n_10sigma": int((A[:, 0] > 10).sum())})
            allA.append(A)
            if f == 500:
                H, W = img.shape
                cy, cx = H // 2, W // 2
                ax[j, 0].imshow(img[cy - crop:cy + crop, cx - crop:cx + crop], cmap="gray", interpolation="nearest")
                ax[j, 0].set_title(f"cam{cam} frame 500, raw, centre {2 * crop} px", fontsize=9)
                ax[j, 1].imshow(d[cy - crop:cy + crop, cx - crop:cx + crop], cmap="magma", vmin=-2 * noise,
                                vmax=12 * noise, interpolation="nearest")
                ax[j, 1].set_title(f"cam{cam}: background - image (noise {noise:.1f})", fontsize=9)
                sm = cv2.GaussianBlur(d, (0, 0), smooth)
                sm[:30, :] = 0
                sm[-30:, :] = 0
                sm[:, :30] = 0
                sm[:, -30:] = 0
                order = np.argsort(sm.ravel())[::-1]
                picks = []
                for i in order[:50000]:
                    y, x = divmod(int(i), W)
                    if all(abs(y - py) > 40 or abs(x - px) > 40 for py, px in picks):
                        picks.append((y, x))
                    if len(picks) >= 16:
                        break
                t = 2 * half
                tiles = [d[y - t:y + t + 1, x - t:x + t + 1] for (y, x) in picks]
                rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)]
                ax[j, 2].imshow(np.vstack(rows), cmap="magma", vmin=-2 * noise, vmax=15 * noise, interpolation="nearest")
                ax[j, 2].set_title(f"cam{cam}: the 16 strongest images ({2 * t + 1} px tiles)", fontsize=9)
        A = np.vstack(allA)
        log(f"\n  cam{cam}: noise {np.mean([p['noise'] for p in per]):.2f} grey levels; images above 5 noise per frame "
            + ", ".join(str(p["n_5sigma"]) for p in per) + "; above 10 noise " + ", ".join(str(p["n_10sigma"]) for p in per))
        res = {"per_frame": per, "all": describe("all above 5 noise", A, log),
               "strong": describe("above 10 noise", A[A[:, 0] > 10], log)}
        out["cameras"][f"cam{cam}"] = res
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "p0b_t1_crops.png"), dpi=60)
    plt.close(fig)
    # 7 Oct's T1, the same statistics (local copy of the raw frames, if present)
    old_raw = os.path.join(OLD_DATA, "T1raw")
    if os.path.isdir(old_raw):
        log("\n7 Oct's T1 (cams 1-2, frames 100 and 500, background = median of every 16th frame of its local copy)")
        out["T1_20261007"] = {}
        for cam in (1, 2):
            folder = os.path.join(old_raw, f"cam{cam}")
            names = sorted(n for n in os.listdir(folder) if n.endswith(".tif"))
            stack = np.stack([case.read_image(os.path.join(folder, n)).astype(np.float32) for n in names[::16]])
            bg = np.median(stack, axis=0)
            allA, per = [], []
            for f in (100, 500):
                img = case.read_image(os.path.join(folder, f"frame{f:06d}.tif")).astype(np.float32)
                d = bg - img
                noise = 1.4826 * float(np.median(np.abs(d - np.median(d))))
                A = shapes(d, noise, 1.0, 7)
                per.append({"frame": f, "noise": noise, "n_5sigma": int(len(A)), "n_10sigma": int((A[:, 0] > 10).sum())})
                allA.append(A)
            A = np.vstack(allA)
            log(f"  cam{cam}: noise {np.mean([p['noise'] for p in per]):.2f}; images above 5 noise per frame "
                + ", ".join(str(p["n_5sigma"]) for p in per))
            out["T1_20261007"][f"cam{cam}"] = {"per_frame": per, "all": describe("all above 5 noise", A, log),
                                               "strong": describe("above 10 noise", A[A[:, 0] > 10], log)}
    else:
        log(f"\n(7 Oct's local copy {old_raw} not found: no comparison)")
    log.save(out)


CFG = case.load_config()

if __name__ == "__main__":
    main()
