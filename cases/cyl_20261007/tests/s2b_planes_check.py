"""S2b (plan L3: the Planes as the shape model's second source): S2's model against the board's blurred dot edges.

    python tests/s2b_planes_check.py          (after s2_shape_fit.py)

The Planes were recorded with T1's focus and aperture (your answer 3), so the dots' outlines are blurred by the
same defocus: a uniform ellipse with full axes L_h, L_v (the model's), whose line spread function along each axis
has variance L^2/16, convolved with the base blur.  P0e fitted an erf step across each usable dot's outline
(sigma_h from its left and right edges, across the rows; sigma_v from its top and bottom) and C2a fitted each
position's plane.  Each dot is placed where its line of sight meets its position's plane, and the model's widths
there, sqrt(L^2/16 + sigma0^2), are compared with P0e's sigma_h and sigma_v: per camera and position, the medians
and their ratio, and in cam2 (whose view of the board spans a range of depths) the trend with depth.
This is a check, not a fit: the printed dots' own edges add to the blur, a particle's image adds its size (60 um,
about 2 px) to sigma0 that a dot edge does not have, and an erf fitted to a disk's edge is only close to its RMS
width.  Agreement within about 30 % is what to expect; a factor of two, or a trend with depth the model does not
follow, is not.  Writes s2b_planes_check_out.txt, .json and s2b_planes_check.png beside itself.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402

lpt = ts.lib()
import shape_model as sm  # noqa: E402


def widths(cm, z):
    A, zf, s0, dl, f = cm["A_mm"], cm["z_f_mm"], cm["sigma0_px"], cm["delta_mm"], cm["f_px"]
    Lh = A * f * np.abs(1.0 / (z[:, 0] + dl) - 1.0 / zf)
    Lv = A * f * np.abs(1.0 / (z[:, 1] + dl) - 1.0 / zf)
    return np.sqrt(Lh ** 2 / 16 + s0 ** 2), np.sqrt(Lv ** 2 / 16 + s0 ** 2), Lh, Lv


def main():
    log = case.Log(os.path.join(HERE, "s2b_planes_check"))
    with open(os.path.join(ts.DATA, "T1_shape.json"), encoding="utf-8") as fh:
        model = json.load(fh)
    with open(os.path.join(HERE, "p0e_planes_survey.json"), encoding="utf-8") as fh:
        p0e = json.load(fh)["positions"]
    with open(os.path.join(HERE, "c2a_board_check.json"), encoding="utf-8") as fh:
        c2a = json.load(fh)["positions"]
    cams = [lpt.loadCameraModel(f) for f in ts.CAMS]
    fds = [sm.FocalDepths(lpt, f) for f in ts.CAMS]
    log(f"S2b: S2's model {model['model']} against the Planes' dot edges; " + "; ".join(
        f"cam{c + 1} A {m['A_mm']:.2f} mm, z_f {m['z_f_mm']:.1f} mm, sigma0 {m['sigma0_px']:.2f} px, delta "
        f"{m['delta_mm']:+.0f} mm" for c, m in enumerate(model["cameras"])))
    out = {"model": model["cameras"], "positions": {}}
    rows = []
    for p in ("0", "1", "2"):
        ctr = np.array(c2a[p]["plane"]["centre"], float)
        nrm = np.array(c2a[p]["plane"]["normal"], float)
        out["positions"][p] = {}
        for c in (0, 1):
            dots = [d for d in p0e[p][f"cam{c + 1}"]["dots"]
                    if np.isfinite(d.get("sigma_h", np.nan)) or np.isfinite(d.get("sigma_v", np.nan))]
            X = []
            for d in dots:
                los = cams[c].lineOfSight(lpt.Pt2D(float(d["u"]), float(d["v"])))
                q = np.array([los.pt[i] for i in range(3)])
                e = np.array([los.unit_vector[i] for i in range(3)])
                X.append(q + e * float((ctr - q) @ nrm / (e @ nrm)))
            X = np.array(X)
            z = fds[c](X)
            wh, wv, Lh, Lv = widths(model["cameras"][c], z)
            sh = np.array([d.get("sigma_h", np.nan) for d in dots], float)
            sv = np.array([d.get("sigma_v", np.nan) for d in dots], float)
            rec = {"dots": len(dots), "z_tan_median_mm": float(np.nanmedian(z[:, 0])),
                   "z_sag_median_mm": float(np.nanmedian(z[:, 1]))}
            for nm, meas, pred, L in (("h", sh, wh, Lh), ("v", sv, wv, Lv)):
                ok = np.isfinite(meas) & np.isfinite(pred)
                rec[nm] = {"n": int(ok.sum()), "measured_median_px": float(np.median(meas[ok])),
                           "predicted_median_px": float(np.median(pred[ok])),
                           "ratio_median": float(np.median(meas[ok] / pred[ok])),
                           "ratio_iqr": np.percentile(meas[ok] / pred[ok], [25, 75]).round(3).tolist(),
                           "L_median_px": float(np.median(L[ok]))}
            out["positions"][p][f"cam{c + 1}"] = rec
            for k in range(len(dots)):
                rows.append((int(p), c, z[k, 0], z[k, 1], sh[k], sv[k], wh[k], wv[k], dots[k]["u"]))
            log(f"  position {p}, cam{c + 1}: {len(dots)} dots, z_tan {rec['z_tan_median_mm']:.1f} mm, z_sag "
                f"{rec['z_sag_median_mm']:.1f} mm; along u: measured {rec['h']['measured_median_px']:.2f}, model "
                f"{rec['h']['predicted_median_px']:.2f} px (ratio {rec['h']['ratio_median']:.2f}, IQR "
                f"{rec['h']['ratio_iqr']}); along v: measured {rec['v']['measured_median_px']:.2f}, model "
                f"{rec['v']['predicted_median_px']:.2f} px (ratio {rec['v']['ratio_median']:.2f}, IQR "
                f"{rec['v']['ratio_iqr']})")
    R = np.array(rows, float)
    # cam2: the trend with depth, measured against predicted
    for c in (0, 1):
        m = R[:, 1] == c
        for k, nm in ((4, "h"), (5, "v")):
            ok = m & np.isfinite(R[:, k]) & np.isfinite(R[:, k + 2])
            zc = R[ok, 2] if nm == "h" else R[ok, 3]
            sm_ = np.polyfit(zc, R[ok, k], 1)[0]
            sp = np.polyfit(zc, R[ok, k + 2], 1)[0]
            out[f"cam{c + 1}_slope_{nm}"] = {"measured_px_per_mm": float(sm_), "predicted_px_per_mm": float(sp),
                                             "depth_range_mm": float(np.ptp(zc))}
            log(f"  cam{c + 1} along {'u' if nm == 'h' else 'v'}: slope with depth measured {sm_:+.4f}, model "
                f"{sp:+.4f} px/mm over {np.ptp(zc):.1f} mm of depth")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for c in (0, 1):
        m = R[:, 1] == c
        for k, nm, col in ((4, "h", "C0"), (5, "v", "C1")):
            zc = R[m, 2] if nm == "h" else R[m, 3]
            ax[c].scatter(zc, R[m, k], s=6, color=col, label=f"P0e sigma_{nm}")
            o = np.argsort(zc)
            ax[c].plot(zc[o], R[m, k + 2][o], color=col, lw=1.2, label=f"model w_{nm}")
        ax[c].set_xlabel("focal depth z_tan (h) / z_sag (v) [mm]")
        ax[c].set_ylabel("width [px]")
        ax[c].set_title(f"cam{c + 1}: the board's dot edges against the model")
        ax[c].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s2b_planes_check.png"), dpi=110)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
