"""P0e (plan: Phase 0): the Planes recording, read-only: the board's dots in both cameras at the three stage
positions, and the blur of their outlines.

    python tests/p0e_planes_survey.py

For each position (0, 1, 2) and camera: the dots found by the 30 Sep case's detector (planes.detect: dots.py,
unchanged, after masking the dark band at the board's edge that cam2 sees), by kind (dot = usable, border = cut by
the image's border or too close to it, small / big / irregular), their outline sizes (semi-axes a, b of the
half-height ellipse) and edge RMS; and per usable dot the blur of its outline, as the sigma of an erf step fitted
across the outline along the rows (left and right edges: sigma_h) and along the columns (top and bottom: sigma_v)
(planes.edge_blur).  Through the tube the blur along the rows and along the columns differ (astigmatism), and in
cam2's oblique view the board's depth changes across the image, so the blur is shown against u.
Writes p0e_planes_survey_out.txt, p0e_planes_survey.json, p0e_blur.png and p0e_overlay_pos{0,1,2}_cam{1,2}.jpg (the
usable dots' ellipses in green, the others in red, at 0.6 scale) beside this script.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import planes  # noqa: E402

import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KEYS = ("u", "v", "a", "b", "theta", "edge_rms", "area", "core")


def overlay(img, ds, path):
    vis = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for d in ds:
        if d["ok"]:
            cv2.ellipse(vis, ((d["u"], d["v"]), (2 * d["a"], 2 * d["b"]), np.degrees(d["theta"])), (0, 200, 0), 2,
                        cv2.LINE_AA)
        else:
            x, y, w, h = d["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, d["kind"], (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    vis = cv2.resize(vis, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    cfg = case.load_config()
    pl = cfg["planes"]
    log = case.Log(os.path.join(HERE, "p0e_planes_survey"))
    log(f"P0e: the Planes ({pl['raw_dir']}), read-only; detector settings config.planes.detect (the 30 Sep case's)")
    out = {"positions": {}}
    for pos in pl["positions"]:
        for cam in case.cam_ids(cfg):
            det = dict(pl["detect"])
            det.update(pl.get("detect_overrides", {}).get(f"cam{cam}", {}))
            img = planes.read(cfg, pos, cam)
            r = planes.detect(img, det)
            ds = r["dots"]
            kinds = Counter(d["kind"] if not (d["kind"] == "dot" and not d["ok"]) else "dot (no fit)" for d in ds)
            ok = [d for d in ds if d["ok"]]
            rows = []
            for d in ok:
                rec = {k: float(d[k]) for k in KEYS}
                rec.update(planes.edge_blur(r["N"], d))
                rows.append(rec)
            A = np.array([[q["a"], q["b"], q["edge_rms"], q["sigma_h"], q["sigma_v"]] for q in rows], float)
            rec = {"kinds": dict(kinds), "usable": len(ok), "a_ref_px2": float(r["a_ref"]),
                   "masked_px": int(r["masked"].sum()), "dots": rows,
                   "median": {"a": float(np.median(A[:, 0])), "b": float(np.median(A[:, 1])),
                              "edge_rms": float(np.median(A[:, 2])), "sigma_h": float(np.nanmedian(A[:, 3])),
                              "sigma_v": float(np.nanmedian(A[:, 4]))}}
            uu = np.array([q["u"] for q in rows])
            vv = np.array([q["v"] for q in rows])
            rec["extent_px"] = {"u": [float(uu.min()), float(uu.max())], "v": [float(vv.min()), float(vv.max())]}
            # the blur's trend across the image (cam2: the board's depth changes along u)
            for key in ("sigma_h", "sigma_v"):
                y = A[:, 3 if key == "sigma_h" else 4]
                m = np.isfinite(y)
                if m.sum() > 5:
                    p = np.polyfit(uu[m], y[m], 1)
                    rec[f"{key}_trend_per_100px"] = float(100.0 * p[0])
                    rec[f"{key}_at_u"] = {str(u0): float(np.polyval(p, u0)) for u0 in (200, 640, 1080)}
            out["positions"].setdefault(pos, {})[f"cam{cam}"] = rec
            md = rec["median"]
            log(f"\n  position {pos}, cam{cam}: {len(ds)} dark regions: " +
                ", ".join(f"{n} {k}" for k, n in kinds.most_common()) +
                (f"; {rec['masked_px']} px masked (the board's edge)" if rec["masked_px"] else ""))
            log(f"    usable dots {len(ok)}, u {uu.min():.0f}-{uu.max():.0f}, v {vv.min():.0f}-{vv.max():.0f} px; "
                f"outline a {md['a']:.1f}, b {md['b']:.1f} px (median), edge RMS {md['edge_rms']:.3f} px")
            log(f"    outline blur (erf sigma, median): along the rows {md['sigma_h']:.2f} px, along the columns "
                f"{md['sigma_v']:.2f} px" +
                (f"; trend with u per 100 px: {rec.get('sigma_h_trend_per_100px', float('nan')):+.3f} (rows), "
                 f"{rec.get('sigma_v_trend_per_100px', float('nan')):+.3f} (columns)" if "sigma_h_trend_per_100px" in rec
                 else ""))
            if "sigma_h_at_u" in rec:
                log("    fitted line at u = 200 / 640 / 1080 px: rows " +
                    " / ".join(f"{rec['sigma_h_at_u'][k]:.2f}" for k in ("200", "640", "1080")) + ", columns " +
                    " / ".join(f"{rec['sigma_v_at_u'][k]:.2f}" for k in ("200", "640", "1080")) + " px")
            overlay(img, ds, os.path.join(HERE, f"p0e_overlay_pos{pos}_cam{cam}.jpg"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, cam in zip(axes, case.cam_ids(cfg)):
        for k, pos in enumerate(pl["positions"]):
            rows = out["positions"][pos][f"cam{cam}"]["dots"]
            u = np.array([q["u"] for q in rows])
            ax.plot(u, [q["sigma_h"] for q in rows], "o", ms=3, color=f"C{k}", label=f"position {pos}: along rows")
            ax.plot(u, [q["sigma_v"] for q in rows], "^", ms=3, mfc="none", color=f"C{k}",
                    label=f"position {pos}: along columns")
        ax.set_xlabel("u (px)")
        ax.set_ylabel("outline blur sigma (px)")
        ax.set_title(f"cam{cam}: blur of the dots' outlines", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "p0e_blur.png"), dpi=110)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
