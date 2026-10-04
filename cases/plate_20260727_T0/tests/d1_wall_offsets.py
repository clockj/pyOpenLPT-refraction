"""D1: how well do the data fix each wall's position along its normal (the probe fit's "pinned" warning: wall 1 ended
against the nearest wand end, its offset sigma 11 mm, wall 2's 8 mm)?

A profile of the cost: from a saved fit (tests/states/<name>.pkl), the wall's camera-side surface is moved along its
normal by delta (negative: towards its cameras) and held there with its normal; everything else (the cameras, f, the
other wall, the wand frames; camera 1's pose is the gauge) is refitted by bundle adjustment on the fit's active
detections with its noise weights.  Delta chi^2 = the weighted cost minus the fit's own (in units of the noise
variance): a value of 1 is one standard deviation.  Also recorded: each camera's distance to the moved wall, f, the
wand ends' smallest depth below each wall, and each profile fit judged on the frames the fit did not use (evaluate.py:
wand length, reprojection) -- whether the wall's position matters for measuring in the volume.

    python tests/d1_wall_offsets.py t3_s0 [t3k_s0 ...]
Writes d1_wall_offsets_<name>_out.txt and .json beside this script.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import evaluate as ev  # noqa: E402

DELTAS = {0: [-80.0, -60.0, -45.0, -30.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0],
          1: [-40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0]}


def min_depths(sol):
    XA, XB = sol.frames.ends(sol.L)
    act = np.zeros(sol.obs.n_frames, bool)
    act[sol.obs.frame[sol.obs.active]] = True
    P = np.vstack([XA[act], XB[act]])
    return [float(w.depth(P).min()) for w in sol.rig.windows]


def profile(name: str):
    cfg = case.load_config()
    st = case.load_state(os.path.join(HERE, "states", f"{name}.pkl"))
    rc, cd, rows, data, obs, frames = case.rebuild(st, cfg)
    fit_k1 = bool(st["layout"]["fit_k1"])
    log = case.Tee(os.path.join(HERE, f"d1_wall_offsets_{name}_out.txt"))
    log(f"D1 {name}: {len(rows)} frames, fit_k1 {fit_k1}; the saved fit: RMS {st['rms']:.4f} px, sigma "
        f"{st['sigma_px']:.4f} px")
    # the fit's own cost with its weights, wall free (a few iterations from the saved answer)
    w = st.get("weights")
    if w is None:
        base = case.pw.solve(data, st["cams"], st["rig"], fit_k1=fit_k1, obs=obs, frames=frames)
        w, obs, frames = base.weights, base.obs, base.frames
        cams0, rig0 = base.cams, base.rig
    else:
        cams0, rig0 = st["cams"], st["rig"]
    lay0 = case.pw.Layout(obs.n_cams, len(rig0.windows), fit_k1=fit_k1, anchors=case.pw.anchors_for(cams0, rig0))
    b0 = case.pw.bundle_adjust(cams0, rig0, frames, st["L"], obs, lay0, weights=w, max_iter=200)
    c0 = b0.info["weighted_cost"]
    log(f"  free walls: weighted cost {c0:.2f} ({int(obs.active.sum())} detections), RMS {b0.rms:.4f} px; smallest "
        f"depth below the walls {', '.join(f'{d:.3g}' for d in min_depths(b0))} mm")
    out = {"name": name, "fit_k1": fit_k1, "frames": int(len(rows)), "cost_free": c0, "walls": []}
    other = np.setdiff1d(np.arange(cd.uv.shape[0]), rows)

    def unused(s):
        e = ev.evaluate(s.cams, s.rig, cd.uv[other], rc.wand_length_mm, rc.sizes)
        w_ = e["wand_error_mm"]
        return {"median_px": e["euclidean_px"]["median"], "rms_inliers_px": e["rms_inliers_px"],
                "wand_abs_median_mm": w_["abs"]["median"], "wand_mean_mm": e["L"] + w_["clean_frames"]["mean"],
                "wand_clean_sd_mm": w_["clean_frames"]["std"]}

    u0 = unused(b0)
    log(f"  free walls on the {len(other)} unused frames: median {u0['median_px']:.4f} px, inliers RMS "
        f"{u0['rms_inliers_px']:.4f}; wand |dL| median {u0['wand_abs_median_mm']:.4f} mm, mean L (clean frames) "
        f"{u0['wand_mean_mm']:.4f}, SD {u0['wand_clean_sd_mm']:.4f}")
    out["unused_free"] = u0
    for wi, deltas in DELTAS.items():
        rows_w = []
        log(f"  wall {wi + 1} moved by delta (mm, negative = towards its cameras) and held:")
        log("    delta   dchi2    RMS px   " + "  ".join(f"cam{c} to wall" for c in b0.rig.cams_of(wi))
            + "   f (px) of every camera     smallest depths (mm)   unused frames: |dL| median, clean mean L, inl. RMS")
        for dl in deltas:
            rig_d = b0.rig.copy()
            wn = rig_d.windows[wi]
            rig_d.windows[wi] = case.pm.Window(wn.n, wn.d + dl, wn.t, wn.idx, wn.name)
            dep = min_depths(case.pw.Solution(b0.cams, rig_d, b0.frames, b0.L, obs, lay0))
            if min(dep) <= 0:
                log(f"    {dl:+6.1f}   (a wand end would lie in or before the wall: smallest depth {min(dep):.2f} mm)")
                continue
            lay = case.pw.Layout(obs.n_cams, len(rig_d.windows), fit_k1=fit_k1, fixed_windows=[wi],
                                 anchors=case.pw.anchors_for(b0.cams, rig_d))
            s = case.pw.bundle_adjust(b0.cams, rig_d, b0.frames, b0.L, obs, lay, weights=w, max_iter=300)
            dchi = s.info["weighted_cost"] - c0
            dist = [s.rig.windows[wi].distance(s.cams[c].C) for c in s.rig.cams_of(wi)]
            dd = min_depths(s)
            u = unused(s)
            rows_w.append({"delta_mm": dl, "dchi2": dchi, "rms_px": s.rms, "distances_mm": dist,
                           "f_px": [c.f for c in s.cams], "k1": [float(c.dist[0]) for c in s.cams],
                           "min_depth_mm": dd, "n_fail": s.info.get("n_fail"), "unused": u})
            log(f"    {dl:+6.1f} {dchi:8.2f}  {s.rms:.4f}   " + "  ".join(f"{d:12.1f}" for d in dist)
                + "   " + " ".join(f"{c.f:6.0f}" for c in s.cams) + "   " + ", ".join(f"{d:.2f}" for d in dd)
                + f"   {u['wand_abs_median_mm']:.4f}, {u['wand_mean_mm']:.4f}, {u['rms_inliers_px']:.4f}")
        out["walls"].append({"wall": wi + 1, "profile": rows_w})
    with open(os.path.join(HERE, f"d1_wall_offsets_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["t3_s0"]:
        profile(nm)
