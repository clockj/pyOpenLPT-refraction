"""D2: which frames hold a wall "pinned" (plate_wand_io.assess: a wand end on a wall's water-side surface)?  For a saved
fit: every active frame with a wand end within 1 mm of a wall, its detections' reprojection residuals (as fitted, wand
length held), its free-triangulation length, and the frame's detections drawn on its four images
(d2_pinned_<fit>_<frame>.png, for the first MAX_IMAGES such frames of each wall).  A gross detection error that the fit
keeps pins the wall without the data asking for it.

    python tests/d2_pinned_frames.py t3k_s4 [...]
Writes d2_pinned_frames_<fit>_out.txt beside this script.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import evaluate as ev  # noqa: E402

NEAR_MM = 1.0
MAX_IMAGES = 8           # frames drawn per wall (the text lists them all)


def main(name: str):
    cfg = case.load_config()
    st = case.load_state(os.path.join(HERE, "states", f"{name}.pkl"))
    rc, cd, rows, data, obs, frames = case.rebuild(st, cfg)
    cams, rig, L = st["cams"], st["rig"], st["L"]
    log = case.Tee(os.path.join(HERE, f"d2_pinned_frames_{name}_out.txt"))
    e = case.pw.evaluate(cams, rig, frames, L, obs)
    XA, XB = frames.ends(L)
    act_f = np.zeros(obs.n_frames, bool)
    act_f[obs.frame[obs.active]] = True
    sig = st["sigma_px"]
    log(f"D2 {name}: {len(rows)} frames, RMS {st['rms']:.4f} px, sigma {sig:.4f} px")
    fids = cd.frame_ids[rows]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for w, win in enumerate(rig.windows):
        dA, dB = win.depth(XA), win.depth(XB)
        near = np.flatnonzero(act_f & (np.minimum(dA, dB) < NEAR_MM))
        log(f"  wall {w + 1}: {len(near)} active frame(s) with a wand end within {NEAR_MM} mm of its water side; the "
            f"next nearest frame's end lies {np.sort(np.minimum(dA, dB)[act_f])[len(near)]:.2f} mm deep")
        for k_near, f in enumerate(near):
            fid = int(fids[obs.frame_ids[f]])
            sel = np.flatnonzero(obs.frame == f)
            Lfree = ev.evaluate(cams, rig, cd.uv[[rows[obs.frame_ids[f]]]], L, rc.sizes)["wand_length_mm"]["median"]
            log(f"    frame {fid}: depths A {dA[f]:.3g}, B {dB[f]:.3g} mm; free-triangulation length {Lfree:.3f} mm")
            for i in sel:
                r = e.r[i]
                log(f"      cam{obs.cam[i]} end {'AB'[obs.end[i]]}: detection ({obs.uv[i, 0]:.2f}, {obs.uv[i, 1]:.2f}), "
                    f"residual ({r[0]:+.2f}, {r[1]:+.2f}) px = {np.hypot(*r) / sig:.1f} sigma, "
                    f"{'active' if obs.active[i] else 'rejected'}{'' if e.ok[i] else ', projection failed'}")
            if k_near >= MAX_IMAGES:
                continue
            fig, axs = plt.subplots(1, len(cams), figsize=(3.2 * len(cams), 3.4))
            for c in range(len(cams)):
                img = case.read_image(case.image_path(cfg, c, fid))
                uvc = cd.uv[rows[obs.frame_ids[f]], c]
                rad = cd.radii[rows[obs.frame_ids[f]], c]
                mid = np.nanmean(uvc, axis=0)
                half = int(max(np.linalg.norm(uvc[1] - uvc[0]), 60) / 2 + 2.5 * np.nanmax(rad))
                x0, y0 = max(int(mid[0]) - half, 0), max(int(mid[1]) - half, 0)
                crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
                lo, hi = np.percentile(crop, [1, 99.5])
                ax = axs[c]
                ax.imshow(np.clip((crop - lo) / max(hi - lo, 1), 0, 1), cmap="gray",
                          extent=(x0 - 0.5, x0 + crop.shape[1] - 0.5, y0 + crop.shape[0] - 0.5, y0 - 0.5))
                for k, col in ((0, "tab:orange"), (1, "tab:cyan")):
                    ax.add_patch(plt.Circle(tuple(uvc[k]), rad[k], fill=False, color=col, lw=1))
                    ax.text(uvc[k, 0], uvc[k, 1] - rad[k] - 3, "AB"[k], color=col, ha="center", va="bottom", fontsize=9)
                ax.set_title(f"cam{c} ({rc.cameras[c]['photo']})", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"frame {fid}: a wand end on wall {w + 1}'s water side in fit {name}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(HERE, f"d2_pinned_{name}_{fid}.png"), dpi=80)
            plt.close(fig)
    log.close()


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["t3k_s4"]:
        main(nm)
