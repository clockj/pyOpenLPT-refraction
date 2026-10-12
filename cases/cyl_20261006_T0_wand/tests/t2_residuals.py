"""T2: the residuals of the delivered calibration (plan: 2026-10-06-plan.md, section 2).

    python tests/t2_residuals.py [RESULT_FOLDER]      (default: config out_dir, S2's result)

From the result's calibration_data.json (the detections, the model's residuals du, dv = model - detection, the wand
ends and each frame's wand length from a free triangulation):
  - per camera and ball: RMS of the residuals used, and their mean (a bias of one ball, e.g. from the one-sided rod
    mask, shows here), also split along and across the wand's image;
  - maps: the mean residual in 10 x 6 cells of the image (cells with 20 or more detections); the largest cell mean,
    and its significance against the scatter in the cell;
  - the wand length of the free triangulation: RMS, and its trend with time (frame number) and with the wand's
    position in the tube's frame (X across, Y along the axis, Z towards camera 1), as fitted slopes with their
    standard errors;
  - figures: t2_residual_maps.png (arrows: the cell means, scaled) and t2_wand_length.png beside this script.
Judged by (plan): no pattern larger than about 0.1 px.  Writes t2_residuals_out.txt and t2_residuals.json.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NU, NV, MIN_CELL = 10, 6, 20


def slope(x, y):
    """Least-squares slope of y on x with its standard error (from the residual scatter)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.column_stack([x - x.mean(), np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    s2 = float(r @ r) / max(len(y) - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[0]), float(math.sqrt(cov[0, 0]))


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "t2_residuals"))
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        result = json.load(fh)
    ob = data["observations"]
    fid = np.array(ob["frame_id"])
    cam = np.array(ob["cam_id"])
    end = np.array(ob["end"])
    u, v = np.array(ob["u"], float), np.array(ob["v"], float)
    du = np.array([np.nan if x is None else x for x in ob["du"]], float)
    dv = np.array([np.nan if x is None else x for x in ob["dv"]], float)
    act = np.array(ob["active"], bool) & np.isfinite(du)
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    log(f"T2: residuals of {folder}: {result['frames']} frames, {int(act.sum())} of {len(act)} detections used, "
        f"RMS {result['rms_px']:.4f} px")
    # the direction of the wand's image per frame and camera (A -> B), to split the residuals along / across it
    key = {}
    for i in range(len(fid)):
        key.setdefault((fid[i], cam[i]), {})[int(end[i])] = (u[i], v[i])
    along = np.full(len(fid), np.nan)
    across = np.full(len(fid), np.nan)
    for i in range(len(fid)):
        k = key[(fid[i], cam[i])]
        if 0 in k and 1 in k:
            e = np.array(k[1]) - np.array(k[0])
            e /= np.linalg.norm(e)
            s = 1.0 if end[i] == 1 else -1.0           # positive = away from the other ball
            along[i] = s * (du[i] * e[0] + dv[i] * e[1])
            across[i] = -du[i] * e[1] + dv[i] * e[0]
    out = {"cameras": {}}
    fig, axes = plt.subplots(1, len(set(cam)), figsize=(6.5 * len(set(cam)), 4.6))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, sorted(set(cam))):
        sel = act & (cam == c)
        rec = {"n": int(sel.sum()), "rms_px": float(np.sqrt(np.mean(du[sel] ** 2 + dv[sel] ** 2)))}
        for e_, name in ((0, "A"), (1, "B")):
            s = sel & (end == e_)
            n = int(s.sum())
            rec[name] = {"n": n, "mean_du": float(du[s].mean()), "mean_dv": float(dv[s].mean()),
                         "se_px": float(np.sqrt((du[s].var() + dv[s].var()) / 2 / n)),
                         "mean_along_outward": float(np.nanmean(along[s])),
                         "se_along": float(np.nanstd(along[s]) / math.sqrt(n)),
                         "mean_across": float(np.nanmean(across[s])),
                         "rms_px": float(np.sqrt(np.mean(du[s] ** 2 + dv[s] ** 2)))}
        iu = np.clip((u * NU / W).astype(int), 0, NU - 1)
        iv = np.clip((v * NV / H).astype(int), 0, NV - 1)
        cells = []
        for a in range(NU):
            for b in range(NV):
                s = sel & (iu == a) & (iv == b)
                n = int(s.sum())
                if n < MIN_CELL:
                    continue
                mu, mv = float(du[s].mean()), float(dv[s].mean())
                se = float(np.sqrt((du[s].var() + dv[s].var()) / 2 / n))
                cells.append({"cell": [a, b], "n": n, "mean_du": mu, "mean_dv": mv, "mean_px": math.hypot(mu, mv),
                              "z": math.hypot(mu, mv) / se if se > 0 else float("inf"),
                              "centre": [(a + 0.5) * W / NU, (b + 0.5) * H / NV]})
        big = max(cells, key=lambda d: d["mean_px"]) if cells else None
        rec["cells"] = cells
        rec["largest_cell"] = big
        out["cameras"][f"cam{c}"] = rec
        log(f"  cam{c}: {rec['n']} detections, RMS {rec['rms_px']:.4f} px")
        for name in ("A", "B"):
            r = rec[name]
            log(f"    ball {name}: n {r['n']}, RMS {r['rms_px']:.4f} px, mean (du, dv) ({r['mean_du']:+.4f}, "
                f"{r['mean_dv']:+.4f}) px (SE {r['se_px']:.4f}); along the wand, outward {r['mean_along_outward']:+.4f} "
                f"px (SE {r['se_along']:.4f}), across {r['mean_across']:+.4f} px")
        if big:
            log(f"    cells ({NU} x {NV}, >= {MIN_CELL} detections): {len(cells)}; largest mean {big['mean_px']:.4f} px "
                f"at cell {big['cell']} (n {big['n']}, {big['z']:.1f} standard errors); cells above 0.1 px: "
                f"{sum(1 for d in cells if d['mean_px'] > 0.1)}")
        ax.scatter(u[sel], v[sel], s=1, c="0.8")
        sc = 500.0
        for d in cells:
            ax.annotate("", xy=(d["centre"][0] + sc * d["mean_du"], d["centre"][1] + sc * d["mean_dv"]),
                        xytext=tuple(d["centre"]), arrowprops=dict(arrowstyle="->", color="C3", lw=1.2))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.set_title(f"cam{c}: mean residual per cell (arrows x{sc:.0f}; 0.1 px = {0.1 * sc:.0f} px)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "t2_residual_maps.png"), dpi=110)
    plt.close(fig)

    # the wand length, frame by frame
    L = np.array([np.nan if x is None else x for x in data["wand_length_free_mm"]], float)
    fids = np.array(data["frame_ids"])
    XA, XB = np.array(data["ends_A_mm"], float), np.array(data["ends_B_mm"], float)
    mid = 0.5 * (XA + XB)
    ok = np.isfinite(L) & np.array(data["frame_active"], bool)
    dL = 1e3 * (L[ok] - float(result["wand_length_mm"]))
    rec = {"frames": int(ok.sum()), "mean_um": float(dL.mean()), "rms_um": float(np.sqrt(np.mean(dL ** 2))),
           "sd_um": float(dL.std()), "slopes": {}}
    log(f"  wand length (free triangulation), {rec['frames']} frames: mean error {rec['mean_um']:+.2f} um, SD "
        f"{rec['sd_um']:.2f} um, RMS {rec['rms_um']:.2f} um")
    for name, x, unit in (("frame", fids[ok], "per 1000 frames"), ("X", mid[ok, 0], "per mm"),
                          ("Y", mid[ok, 1], "per mm"), ("Z", mid[ok, 2], "per mm")):
        scale = 1000.0 if name == "frame" else 1.0
        s, se = slope(np.asarray(x, float) / scale, dL)
        rec["slopes"][name] = {"um_" + unit.replace(" ", "_"): s, "se": se, "z": s / se if se > 0 else 0.0,
                               "range": [float(np.min(x)), float(np.max(x))]}
        log(f"    trend with {name}: {s:+.3f} +- {se:.3f} um {unit} ({s / se:+.1f} SE); range "
            f"{np.min(x):.1f} to {np.max(x):.1f}")
    out["wand_length"] = rec
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].plot(fids[ok], dL, ".", ms=2)
    ax[0].set_xlabel("frame")
    ax[0].set_ylabel("wand length error (um)")
    ax[1].scatter(mid[ok, 0], mid[ok, 2], c=dL, s=4, cmap="coolwarm", vmin=-3 * rec["sd_um"], vmax=3 * rec["sd_um"])
    ax[1].set_xlabel("X (mm, across)")
    ax[1].set_ylabel("Z (mm, towards cam1)")
    ax[1].set_aspect("equal")
    ax[1].set_title("wand length error by the wand's position (seen along the axis)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "t2_wand_length.png"), dpi=110)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
