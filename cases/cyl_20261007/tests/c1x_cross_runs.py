"""C1-X (plan: 2026-10-07-plan.md, Phase C; the 6 Oct T2 case's x1_cross_runs.py, adapted): did the rig move since
6 Oct?  This run's T0 calibration and the 6 Oct T2 calibration on each other's wand frames.

    python tests/c1x_cross_runs.py      (after calibrate.py; T2's results from config.reference_T2)

For each pair (calibration, recording) of the two runs' delivered cameras and kept frames (run "T2" = 6 Oct T2,
run "T0" = this case, 7 Oct T0):
  - the wand length of a free triangulation (no wand length, no fitted poses), mean and RMS against 10 mm;
  - the residual of a maximum-likelihood triangulation of each ball from both cameras (as T0's D8).  It has one
    degree of freedom per ball, and with the cameras side by side around the vertical tube it is nearly vertical in
    both images.  Per camera: its mean (u, v) and the RMS about the mean.  A camera that moved between the runs
    leaves a common offset in the other run's frames; on a calibration's own frames the mean is zero;
  - frame by frame: the cameras' vertical disagreement dv = (v residual of cam1) - (v residual of cam2), the mean of
    the two balls, smoothed over WINDOW consecutive kept frames, along each recording; its linear trend (px per
    1,000 frames, with its standard error).  A step between the runs shows as level lines; a drift as a slope.
    6 Oct T2 is also split at its light change (its P0: between frames 2,000 and 3,000).
Figure c1x_offsets.png beside this script.  Writes c1x_cross_runs_out.txt and c1x_cross_runs.json.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WINDOW = 50
LIGHT = (2000, 3000)


def load_cams(lib, folder):
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    cams = [lib.cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"],
                          c["height"], np.array(c["dist"])) for c in res["cameras"]]
    tube = lib.cm.Tube(tuple(res["tube"]["radii_mm"]), tuple(res["tube"]["n"]))
    return res, cams, tube


def evaluate(lib, cams, tube, uv, L):
    """Free triangulation of every frame: length error (um) per frame, and the maximum-likelihood residuals
    arranged as res[frame, camera, ball, (du, dv)] (px)."""
    cw = lib.cw
    obs = cw.observations(cw.WandData(uv, L, [(1280, 800)] * len(cams)))
    XA, XB, cnt = cw.triangulate(cams, tube, obs, np.arange(len(obs)))
    Lm = np.linalg.norm(XB - XA, axis=1)
    Lm[np.any(cnt < 2, axis=1)] = np.nan
    point = obs.frame * 2 + obs.end
    X0 = np.empty((2 * obs.n_frames, 3))
    X0[0::2], X0[1::2] = XA, XB
    ok = np.all(np.isfinite(X0), axis=1)
    X0[~ok] = 0.0
    _X, r, _cov = cw.triangulate_ml(cams, tube, obs.cam, obs.uv, point, X0)
    res = np.full((obs.n_frames, len(cams), 2, 2), np.nan)
    res[obs.frame, obs.cam, obs.end] = r
    return 1e3 * (Lm - L), res


def trend(x, y):
    """Slope of y on x (per unit of x) with its standard error."""
    A = np.column_stack([x - x.mean(), np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    s2 = float(r @ r) / max(len(y) - 2, 1)
    return float(coef[0]), float(math.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0]))


def rolling_median(y, w):
    h = w // 2
    return np.array([np.median(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


def main():
    cfg0 = case.load_config()                                    # this case: 7 Oct T0
    cfg2 = case.load_config(os.path.normpath(os.path.join(CASE_DIR, cfg0["reference_T2"]["case_dir"], "config.json")))
    log = case.Log(os.path.join(HERE, "c1x_cross_runs"))
    lib = case.library()
    L = float(cfg0["wand"]["length_mm"])
    runs = {}
    for name, cfg in (("T2", cfg2), ("T0", cfg0)):
        _d, kept, _r, _m, frames = case.load_selection(cfg)
        runs[name] = {"frames": np.array(frames), "uv": case.uv_array(kept, frames, case.cam_ids(cfg))}
    cals = {"T2": load_cams(lib, cfg2["out_dir"]), "T0": load_cams(lib, cfg0["out_dir"])}
    log(f"C1-X: the calibrations of 6 Oct T2 and 7 Oct T0 on each other's frames (T2: {len(runs['T2']['frames'])} "
        f"kept frames, T0: {len(runs['T0']['frames'])})")
    out = {"window": WINDOW, "pairs": {}, "series": {}}
    log(f"\n{'cameras':>8s} {'frames':>7s}   wand length error: mean, RMS (um)   residual per camera: mean (du, dv), "
        "RMS about it (px)")
    for cal_name, (res, cams, tube) in cals.items():
        for run_name, run in runs.items():
            dL, r = evaluate(lib, cams, tube, run["uv"], L)
            d = dL[np.isfinite(dL)]
            rec = {"length_mean_um": float(np.mean(d)), "length_rms_um": float(np.sqrt(np.mean(d ** 2))),
                   "length_se_um": float(np.std(d) / math.sqrt(len(d))), "frames": int(len(d)), "cameras": {}}
            parts = []
            for j, c in enumerate(res["cameras"]):
                rc = r[:, j].reshape(-1, 2)
                rc = rc[np.all(np.isfinite(rc), axis=1)]
                mean = rc.mean(axis=0)
                about = float(np.sqrt(np.mean(np.sum((rc - mean) ** 2, axis=1))))
                rec["cameras"][c["name"]] = {"mean_px": mean.tolist(), "rms_about_mean_px": about,
                                             "se_mean_v_px": float(np.std(rc[:, 1]) / math.sqrt(len(rc)))}
                parts.append(f"{c['name']} ({mean[0]:+.3f}, {mean[1]:+.3f}), {about:.3f}")
            log(f"{cal_name + ' cams':>8s} {run_name:>7s}   {rec['length_mean_um']:+6.1f} +- {rec['length_se_um']:.1f}, "
                f"{rec['length_rms_um']:5.1f}            " + "; ".join(parts))
            # the cameras' vertical disagreement, frame by frame
            dv = np.nanmean(r[:, 0, :, 1] - r[:, 1, :, 1], axis=1)
            fr = run["frames"]
            okf = np.isfinite(dv)
            s, se = trend(fr[okf] / 1000.0, dv[okf])
            rec["dv_mean_px"] = float(np.mean(dv[okf]))
            rec["dv_trend_px_per_1000_frames"] = [s, se]
            line = (f"{'':>8s} {'':>7s}   vertical disagreement cam1 - cam2: mean {np.mean(dv[okf]):+.3f} px, trend "
                    f"{s:+.4f} +- {se:.4f} px per 1000 frames")
            if run_name == "T2":
                for tag, sel in (("before the light change", fr < LIGHT[0]), ("after", fr > LIGHT[1])):
                    m = okf & sel
                    rec[f"dv_mean_px_{'before' if tag.startswith('before') else 'after'}_light"] = float(np.mean(dv[m]))
                    line += f"; {tag} {np.mean(dv[m]):+.3f} ({int(m.sum())} frames)"
            log(line)
            out["pairs"][f"{cal_name}_cams_on_{run_name}"] = rec
            sm = rolling_median(dv[okf], WINDOW)
            keep = np.arange(0, okf.sum(), 5)                 # every 5th point is enough for the figure and the page
            out["series"][f"{cal_name}_cams_on_{run_name}"] = {"frame": fr[okf][keep].tolist(),
                                                                "dv_smooth": np.round(sm[keep], 4).tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, run_name in zip(axes, ("T2", "T0")):
        for cal_name, col in (("T2", "C3"), ("T0", "C0")):
            s = out["series"][f"{cal_name}_cams_on_{run_name}"]
            ax.plot(s["frame"], s["dv_smooth"], color=col, lw=1.2,
                    label=f"{'6 Oct T2' if cal_name == 'T2' else '7 Oct T0'}'s cameras")
        ax.axhline(0.0, color="0.6", lw=0.8)
        if run_name == "T2":
            ax.axvspan(*LIGHT, color="0.9", zorder=0, label="light changes")
        ax.set_title(f"recording {'6 Oct T2' if run_name == 'T2' else '7 Oct T0'}: the cameras' vertical "
                     f"disagreement (median of {WINDOW} kept frames)", fontsize=9)
        ax.set_xlabel("frame")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("v residual cam1 - cam2 (px)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "c1x_offsets.png"), dpi=110)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
