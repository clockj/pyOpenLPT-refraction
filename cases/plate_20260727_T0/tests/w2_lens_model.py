"""W2 (2026-10-03-walls-plan.md, section 4): is it the lens model?

From the delivered fit (T2, tests/states/t2_all.pkl), bundle adjustment with T2's active detections and noise weights
(as W1; T2 ended with equal weights), the lens model extended by plate_wand's optional lens terms (L1, fit_lens) one at
a time:
  none  f and k1, T2's model          pp    + the principal point (the distortion's centre)
  k2    + k2                          tan   + the tangential terms p1, p2
  all   + all of them
each in three modes:
  free   every frame, the walls free: where does wall 2 go, and what does each term take?
  w2mid  every frame, wall 2 held where the channel (150 mm inside across it) centres the wand, as W1's w2_mid
  s0     every 5th frame (subset s0), the walls free, judged on the other 80 % (evaluate.py): do the terms carry over?
In every mode each ball's centre is held at least its radius (1.5, 2 mm) below each wall's water side (plate_wand's
channel without far walls): T2 is pinned against wall 1, a wand end on its surface, where every step is refused.
Recorded per fit: the cost against `none` in the same frames (delta chi^2), the RMS overall and per camera, each
camera's lens terms with their sigmas, f, the distances to the walls, the wand ends' depths below both walls and the
shift of wall 2 that would centre them in the channel, and evaluate.py on the frames named.

    python tests/w2_lens_model.py              # every case in parallel (subprocesses), then the summary
    python tests/w2_lens_model.py --case NAME  # one case, NAME = <variant>_<mode>
    python tests/w2_lens_model.py --summary
Writes w2_lens_model_out.txt and w2_lens_model.json beside this script; each case's fit and numbers to
tests/states/w2_<case>.pkl and .json, its console to tests/logs/w2_<case>.txt.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
sys.path.insert(0, HERE)
import case  # noqa: E402
import evaluate as ev  # noqa: E402
import w1_channel_hold as w1  # noqa: E402

VARIANTS = {"none": (), "pp": ("pp",), "k2": ("k2",), "tan": ("tangential",), "all": ("pp", "k2", "tangential")}
MODES = ("free", "w2mid", "s0")
MAX_ITER = 500
WIDTH2_MM = w1.WIDTH_MM[1]
BALL_MM = [1.5, 2.0]             # the balls' radii (A, B): every wand end at least this far below each wall


def cases():
    return [f"{v}_{m}" for v in VARIANTS for m in MODES]


def run_case(name: str):
    variant, mode = name.rsplit("_", 1)
    cfg = case.load_config()
    st, rc, cd, rows, data, obs, frames = w1.load_t2(cfg)
    cams, rig, L = st["cams"], st["rig"].copy(), st["L"]
    weights = None if st["weights"] is None else np.asarray(st["weights"], float)    # T2: equal weights (None)
    fixed = []
    if mode == "w2mid":
        dl = w1.planned_shifts(st, obs, frames)["w2_mid"][1]
        wn = rig.windows[1]
        rig.windows[1] = case.pm.Window(wn.n, wn.d + dl, wn.t, wn.idx, wn.name)
        fixed = [1]
    other = None
    if mode == "s0":
        keep = np.zeros(obs.n_frames, bool)
        keep[::5] = True
        sel_rows = keep[obs.frame]
        weights = None if weights is None else weights[sel_rows]
        obs = obs.subset_frames(keep)
        frames = case.pw.Frames(frames.m[keep], frames.w[keep])
        other = np.flatnonzero(~keep)
    lay = case.pw.Layout(obs.n_cams, len(rig.windows), fit_k1=True, fit_lens=VARIANTS[variant], fixed_windows=fixed,
                         anchors=case.pw.anchors_for(cams, rig))
    # T2 is pinned against wall 1 (a wand end on its water-side surface: every step is refused, report section 7);
    # the balls' radii as the near walls' limit (plate_wand's channel without far walls) unpin it: wall 1 moves ~2 mm
    case.pw.set_constraints(lay, {"channel": {"width_mm": [None, None], "ball_radius_mm": BALL_MM}})
    rig = case.pw.into_channel(rig, frames, L, obs, lay)
    t0 = time.time()
    sol = case.pw.bundle_adjust(cams, rig, frames, L, obs, lay, weights=weights, max_iter=MAX_ITER, verbose=True)
    secs = time.time() - t0
    sol = case.pw.covariance(sol, weights=weights)
    sig = case.pw.camera_sigmas(sol.cams, sol.rig, sol.layout, sol.cov)
    d = case.describe(sol.cams, sol.rig)
    dep = w1.active_depths(sol.cams, sol.rig, sol.frames, sol.L, sol.obs)
    e_rows = np.flatnonzero(sol.obs.active)
    e = case.pw.evaluate(sol.cams, sol.rig, sol.frames, sol.L, sol.obs, None, e_rows)
    r2 = np.sum(e.r ** 2, axis=1)
    cam_rms = [float(np.sqrt(np.mean(r2[e.ok & (sol.obs.cam[e_rows] == c)]))) for c in range(sol.obs.n_cams)]
    uv_eval = cd.uv if other is None else cd.uv[other]
    evl = ev.evaluate(sol.cams, sol.rig, uv_eval, rc.wand_length_mm, rc.sizes)
    wc = evl["wand_error_mm"]["clean_frames"]
    lens = []
    for c, cam in enumerate(sol.cams):
        s = sig[c]
        lens.append({"cx_off_px": cam.cx - (cam.width - 1) / 2.0, "cy_off_px": cam.cy - (cam.height - 1) / 2.0,
                     "sigma_cx_px": s.get("cx_px"), "sigma_cy_px": s.get("cy_px"), "k1": float(cam.dist[0]),
                     "sigma_k1": s.get("k1"), "k2": float(cam.dist[1]), "sigma_k2": s.get("k2"),
                     "p1": float(cam.dist[2]), "sigma_p1": s.get("p1"), "p2": float(cam.dist[3]),
                     "sigma_p2": s.get("p2"), "f_px": cam.f, "sigma_f_px": s.get("f_px")})
    out = {"case": name, "variant": variant, "mode": mode, "fit_lens": list(VARIANTS[variant]),
           "frames": int(sol.obs.n_frames), "active": int(sol.obs.active.sum()), "rms_px": sol.rms,
           "weighted_cost": sol.info["weighted_cost"], "lm_iterations": len(sol.history), "seconds": secs,
           "n_fail": sol.info["n_fail"], "rms_per_camera_px": cam_rms, "lens": lens,
           "distance_to_wall_mm": [c["distance_to_wall_mm"] for c in d["cameras"]],
           "incidence_deg": [c["incidence_deg"] for c in d["cameras"]], "walls_angle_deg": d["walls_angle_deg"],
           "baselines_mm": {f"C{p['cams'][0] + 1}-C{p['cams'][1] + 1}": p["baseline_mm"] for p in d["pairs"]},
           "depth_wall1_mm": [float(dep[0].min()), float(dep[0].max())],
           "depth_wall2_mm": [float(dep[1].min()), float(dep[1].max())],
           "wall2_centring_shift_mm": float((dep[1].min() + dep[1].max() - WIDTH2_MM) / 2),
           "evaluated_on": "every frame" if other is None else f"the other {len(other)} frames",
           "evaluation": {"median_px": evl["euclidean_px"]["median"], "rms_inliers_px": evl["rms_inliers_px"],
                          "wand_abs_median_mm": evl["wand_error_mm"]["abs"]["median"],
                          "wand_clean_mean_mm": evl["L"] + wc["mean"], "wand_clean_sd_mm": wc["std"],
                          "outlier_detections": evl["outlier_detections"]}}
    sol.info["seconds"] = secs
    case.save_state(os.path.join(HERE, "states", f"w2_{name}.pkl"), sol,
                    extra={"settings": st["extra"]["settings"], "w2_case": name})
    with open(os.path.join(HERE, "states", f"w2_{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=lambda v: None if v is None else float(v))
    print(f"w2 {name}: RMS {sol.rms:.4f} px, {len(sol.history)} LM iterations, {secs:.0f} s")


def run_all(names):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def one(name):
        t0 = time.time()
        with open(os.path.join(HERE, "logs", f"w2_{name}.txt"), "w", encoding="utf-8") as fh:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--case", name], cwd=CASE_DIR, env=env,
                               stdout=fh, stderr=subprocess.STDOUT, text=True)
        return name, p.returncode, time.time() - t0

    with ThreadPoolExecutor(max_workers=len(names)) as ex:
        for name, rc, secs in ex.map(one, names):
            print(f"  {name}: exit {rc}, {secs:.0f} s")


def summary():
    res = {}
    for name in cases():
        p = os.path.join(HERE, "states", f"w2_{name}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                res[name] = json.load(fh)
    log = case.Tee(os.path.join(HERE, "w2_lens_model_out.txt"))
    s2 = case.load_state(w1.T2_STATE)["sigma_px"] ** 2
    log("W2: the lens model extended (plate_wand fit_lens), from the delivered fit T2 with its active detections and "
        "equal weights (T2's); dchi2 against 'none' (T2's model: f, k1) in the same frames and mode: the change of the "
        f"sum of squared residuals over T2's variance per coordinate ({s2 ** 0.5:.4f} px)^2")
    for mode in MODES:
        base = res.get(f"none_{mode}") if mode != "w2mid" else res.get("none_free")
        if base is None:
            continue
        head = {"free": "every frame, walls free",
                "w2mid": f"every frame, wall 2 held where the channel ({WIDTH2_MM:.0f} mm) centres the wand; dchi2 "
                         "against none_free",
                "s0": "every 5th frame, walls free; evaluated on the other 80 %"}[mode]
        log(f"\n  {mode}: {head}")
        log("    variant  dchi2      RMS px   per camera (px)            wand below wall 2 (mm)  centring shift (mm)   "
            "evaluation: median px, inliers RMS, |dL| median, clean SD (mm)")
        for v in VARIANTS:
            o = res.get(f"{v}_{mode}")
            if o is None:
                continue
            e = o["evaluation"]
            log(f"    {v:7s} {(o['weighted_cost'] - base['weighted_cost']) / s2:9.0f}  {o['rms_px']:.4f}   "
                + " ".join(f"{x:.3f}" for x in o["rms_per_camera_px"])
                + f"    {o['depth_wall2_mm'][0]:6.1f}..{o['depth_wall2_mm'][1]:6.1f}       {o['wall2_centring_shift_mm']:+7.1f}"
                  f"          {e['median_px']:.4f}, {e['rms_inliers_px']:.4f}, {e['wand_abs_median_mm']:.4f}, "
                  f"{e['wand_clean_sd_mm']:.4f}")
    log("\n  lens terms per camera (free mode, every frame): principal point offset from the image centre (px), k1, k2, "
        "p1, p2 (+- sigma); f (px); distance to its wall (mm)")
    for v in VARIANTS:
        o = res.get(f"{v}_free")
        if o is None:
            continue
        log(f"    {v}:")
        for c, l in enumerate(o["lens"]):
            sd = lambda k: "" if l.get("sigma_" + k) is None else f" +- {l['sigma_' + k]:.2g}"
            parts = []
            if l["sigma_cx_px"] is not None:
                parts.append(f"pp ({l['cx_off_px']:+.1f} +- {l['sigma_cx_px']:.1f}, {l['cy_off_px']:+.1f} +- "
                             f"{l['sigma_cy_px']:.1f})")
            parts.append(f"k1 {l['k1']:.3f}{sd('k1')}")
            for k in ("k2", "p1", "p2"):
                if l.get("sigma_" + k) is not None:
                    parts.append(f"{k} {l[k]:.3g}{sd(k)}")
            log(f"      C{c + 1}: " + ", ".join(parts) + f"; f {l['f_px']:.0f}; {o['distance_to_wall_mm'][c]:.0f} mm")
    with open(os.path.join(HERE, "w2_lens_model.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    if "--case" in sys.argv:
        run_case(sys.argv[sys.argv.index("--case") + 1])
    elif "--summary" in sys.argv:
        summary()
    else:
        only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else cases()
        run_all(only)
        summary()
