"""W1 (2026-10-03-walls-plan.md, section 4): what does the channel cost?

The delivered fit (T2, tests/states/t2_all.pkl, every frame) puts the wand 109-225 mm from wall 2's water side, which a
channel 150 mm wide inside cannot hold.  Here the walls are moved along their normals to where the channel puts them
and held there; everything else (the cameras, f, k1, the wand frames; camera 1's pose is the gauge) is refitted by
bundle adjustment on T2's active detections with T2's weights (equal: its per-camera noise weights fell back), as D1
did on subsets.  The cases:
  free      walls free (the baseline: T2 refitted the same way)
  w2_lo     wall 2 moved so the farthest wand end lies a ball's radius inside the far wall (the least shift)
  w2_mid    wall 2 moved so the wand is centred along the channel's long side
  w2_hi     wall 2 moved so the nearest wand end lies a ball's radius from wall 2 (the largest shift)
  both_mid  wall 2 as w2_mid, and wall 1 moved so the wand is centred across the short side
A shift is positive away from the wall's cameras (towards the wand).  Recorded: the cost against free (delta chi^2: the
sum of squared residuals over T2's variance per coordinate), the RMS overall and per camera, f, k1, the cameras'
distances and angles to their walls,
the baselines, the walls' angle, the wand ends' depths below both walls, and evaluate.py on every frame.  Residual maps:
each camera's mean residual over an 8 x 8 grid of image cells, for free, w2_mid and their difference -- the pattern
that wall 2's position absorbs.

    python tests/w1_channel_hold.py              # every case in parallel (subprocesses), then the summary
    python tests/w1_channel_hold.py --case NAME  # one case
    python tests/w1_channel_hold.py --summary    # the summary of the saved cases
Writes w1_channel_hold_out.txt, w1_channel_hold.json and w1_residual_maps.png beside this script, each case's fit to
tests/states/w1_<case>.pkl and its console to tests/logs/w1_<case>.txt.
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
import case  # noqa: E402
import evaluate as ev  # noqa: E402

T2_STATE = os.path.join(HERE, "states", "t2_all.pkl")
WIDTH_MM = (100.0, 150.0)        # the channel's inside width across wall 1 and across wall 2 (the user's 15 x 10 cm)
BALL_MM = 2.0                    # a ball's centre at least this far from a wall (radii about 1.4 and 1.8 mm)
CASES = ("free", "w2_lo", "w2_mid", "w2_hi", "both_mid")
MAX_ITER = 300
GRID = 8


def active_depths(cams, rig, frames, L, obs):
    """The depth of every active wand end below each wall's water-side surface (mm), per wall."""
    XA, XB = frames.ends(L)
    act = np.zeros(obs.n_frames, bool)
    act[obs.frame[obs.active]] = True
    P = np.vstack([XA[act], XB[act]])
    return [P @ w.n - w.far_d for w in rig.windows]


def planned_shifts(st, obs, frames):
    """{case: {wall index: shift (mm)}} from the T2 fit's wand ends."""
    d1, d2 = active_depths(st["cams"], st["rig"], frames, st["L"], obs)
    lo2 = float(d2.max() - (WIDTH_MM[1] - BALL_MM))
    hi2 = float(d2.min() - BALL_MM)
    mid2 = float((d2.min() + d2.max() - WIDTH_MM[1]) / 2)
    mid1 = float((d1.min() + d1.max() - WIDTH_MM[0]) / 2)
    return {"free": {}, "w2_lo": {1: lo2}, "w2_mid": {1: mid2}, "w2_hi": {1: hi2}, "both_mid": {0: mid1, 1: mid2}}


def load_t2(cfg):
    st = case.load_state(T2_STATE)
    rc, cd, rows, data, obs, frames = case.rebuild(st, cfg)
    return st, rc, cd, rows, data, obs, frames


def run_case(name: str):
    cfg = case.load_config()
    st, rc, cd, rows, data, obs, frames = load_t2(cfg)
    shifts = planned_shifts(st, obs, frames)[name]
    rig_d = st["rig"].copy()
    for wi, dl in shifts.items():
        wn = rig_d.windows[wi]
        rig_d.windows[wi] = case.pm.Window(wn.n, wn.d + dl, wn.t, wn.idx, wn.name)
    lay = case.pw.Layout(obs.n_cams, len(rig_d.windows), fit_k1=bool(st["layout"]["fit_k1"]),
                         fixed_windows=sorted(shifts), anchors=case.pw.anchors_for(st["cams"], rig_d))
    t0 = time.time()
    sol = case.pw.bundle_adjust(st["cams"], rig_d, frames, st["L"], obs, lay, weights=st["weights"],
                                max_iter=MAX_ITER, verbose=True)
    secs = time.time() - t0
    sol.info["seconds"] = secs
    sol.info["lm_iterations"] = len(sol.history)
    sol.info["shifts_mm"] = {str(k): v for k, v in shifts.items()}
    case.save_state(os.path.join(HERE, "states", f"w1_{name}.pkl"), sol,
                    extra={"settings": st["extra"]["settings"], "w1_case": name,
                           "shifts_mm": {str(k): v for k, v in shifts.items()}})
    print(f"w1 {name}: shifts {shifts}, RMS {sol.rms:.4f} px, weighted cost {sol.info['weighted_cost']:.2f}, "
          f"{len(sol.history)} LM iterations, {secs:.0f} s")


def run_all():
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")

    def one(name):
        t0 = time.time()
        with open(os.path.join(HERE, "logs", f"w1_{name}.txt"), "w", encoding="utf-8") as fh:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--case", name], cwd=CASE_DIR, env=env,
                               stdout=fh, stderr=subprocess.STDOUT, text=True)
        return name, p.returncode, time.time() - t0

    with ThreadPoolExecutor(max_workers=len(CASES)) as ex:
        for name, rc, secs in ex.map(one, CASES):
            print(f"  {name}: exit {rc}, {secs:.0f} s")


def residual_grid(cams, cam_of_row, uv, r, ok):
    """Per camera: the mean residual (u, v) and the count in each of GRID x GRID image cells, from the rows' camera
    indices, detections uv, residuals r and projection flags ok."""
    out = []
    for c, cam in enumerate(cams):
        sel = ok & (cam_of_row == c)
        p, q = uv[sel], r[sel]
        iu = np.clip((p[:, 0] / cam.width * GRID).astype(int), 0, GRID - 1)
        iv = np.clip((p[:, 1] / cam.height * GRID).astype(int), 0, GRID - 1)
        mean = np.full((GRID, GRID, 2), np.nan)
        cnt = np.zeros((GRID, GRID), int)
        for a in range(GRID):
            for b in range(GRID):
                m = (iv == a) & (iu == b)
                cnt[a, b] = int(m.sum())
                if cnt[a, b] >= 20:
                    mean[a, b] = q[m].mean(axis=0)
        out.append({"mean": mean, "count": cnt})
    return out


def summary():
    cfg = case.load_config()
    st0, rc, cd, rows, data, obs, frames0 = load_t2(cfg)
    log = case.Tee(os.path.join(HERE, "w1_channel_hold_out.txt"))
    res = {"width_mm": list(WIDTH_MM), "ball_mm": BALL_MM, "cases": {}}
    rows_act = np.flatnonzero(obs.active)
    fits = {}
    for name in CASES:
        p = os.path.join(HERE, "states", f"w1_{name}.pkl")
        if not os.path.exists(p):
            log(f"  {name}: no saved fit")
            continue
        fits[name] = case.load_state(p)
    c_free = fits["free"]["info"]["weighted_cost"]
    # T2 ended with equal weights (its per-camera weights fell back), so the cost is the sum of squared residuals in
    # px^2; delta chi^2 divides it by T2's variance per coordinate
    s2 = 1.0 if st0["weights"] is not None else st0["sigma_px"] ** 2
    log(f"W1: the delivered fit (T2) on every frame ({obs.n_frames} frames, {len(rows_act)} active detections, T2's "
        f"weights: equal), walls held where a channel {WIDTH_MM[1]:.0f} x {WIDTH_MM[0]:.0f} mm inside puts them, "
        f"everything else refitted; dchi2 = the change of the sum of squared residuals over ({st0['sigma_px']:.4f} "
        f"px)^2")
    log(f"  shifts along each wall's normal, positive away from its cameras; ball centres at least {BALL_MM:g} mm "
        f"inside the channel")
    grids = {}
    for name, st in fits.items():
        cams, rig = st["cams"], st["rig"]
        frames = case.pw.Frames(np.asarray(st["frames_m"]), np.asarray(st["frames_w"]))
        e = case.pw.evaluate(cams, rig, frames, st["L"], obs, None, rows_act)
        ok = e.ok
        r2 = np.sum(e.r ** 2, axis=1)
        cam_rms = [float(np.sqrt(np.mean(r2[ok & (obs.cam[rows_act] == c)]))) for c in range(obs.n_cams)]
        d = case.describe(cams, rig)
        dep = active_depths(cams, rig, frames, st["L"], obs)
        evl = ev.evaluate(cams, rig, cd.uv, rc.wand_length_mm, rc.sizes)
        wc = evl["wand_error_mm"]["clean_frames"]
        shifts = {int(k): v for k, v in st["extra"]["shifts_mm"].items()}
        dcost = st["info"]["weighted_cost"] - c_free
        out = {"shifts_mm": shifts, "dcost_px2": dcost, "dchi2": dcost / s2, "rms_px": st["rms"],
               "rms_per_camera_px": cam_rms, "lm_iterations": st["info"].get("lm_iterations"),
               "seconds": st["info"].get("seconds"), "n_fail": st["info"].get("n_fail"),
               "f_px": [c["f_px"] for c in d["cameras"]], "k1": [c["k1"] for c in d["cameras"]],
               "distance_to_wall_mm": [c["distance_to_wall_mm"] for c in d["cameras"]],
               "incidence_deg": [c["incidence_deg"] for c in d["cameras"]],
               "baselines_mm": {f"C{p['cams'][0] + 1}-C{p['cams'][1] + 1}": p["baseline_mm"] for p in d["pairs"]},
               "walls_angle_deg": d["walls_angle_deg"],
               "depth_wall1_mm": [float(dep[0].min()), float(dep[0].max())],
               "depth_wall2_mm": [float(dep[1].min()), float(dep[1].max())],
               "every_frame": {"median_px": evl["euclidean_px"]["median"], "rms_inliers_px": evl["rms_inliers_px"],
                               "wand_abs_median_mm": evl["wand_error_mm"]["abs"]["median"],
                               "wand_clean_mean_mm": evl["L"] + wc["mean"], "wand_clean_sd_mm": wc["std"],
                               "outlier_detections": evl["outlier_detections"]}}
        res["cases"][name] = out
        if name in ("free", "w2_mid"):
            grids[name] = residual_grid(cams, obs.cam[rows_act], obs.uv[rows_act], e.r, ok)
    log("  case      shifts (mm)            dchi2     RMS px   per camera (px)             f (px)                   "
        "k1                       to wall (mm)            depth below wall 1 / wall 2 (mm)")
    for name, o in res["cases"].items():
        sh = ", ".join(f"wall {k + 1} {v:+.1f}" for k, v in sorted(o["shifts_mm"].items())) or "none"
        log(f"  {name:9s} {sh:22s} {o['dchi2']:9.0f}  {o['rms_px']:.4f}   "
            + " ".join(f"{v:.3f}" for v in o["rms_per_camera_px"]) + "   "
            + " ".join(f"{v:5.0f}" for v in o["f_px"]) + "   " + " ".join(f"{v:.3f}" for v in o["k1"]) + "   "
            + " ".join(f"{v:5.0f}" for v in o["distance_to_wall_mm"]) + "   "
            + f"{o['depth_wall1_mm'][0]:5.1f}..{o['depth_wall1_mm'][1]:5.1f} / {o['depth_wall2_mm'][0]:5.1f}.."
              f"{o['depth_wall2_mm'][1]:5.1f}")
    log("  case      angles to wall normals (deg)   baselines C1-C2, C3-C4 (mm)   walls (deg)   every frame: median px, "
        "inliers RMS, |dL| median (mm), clean mean L, SD, outliers")
    for name, o in res["cases"].items():
        b = o["baselines_mm"]
        f = o["every_frame"]
        log(f"  {name:9s} " + " ".join(f"{v:5.2f}" for v in o["incidence_deg"]) + f"            {b['C1-C2']:7.2f}, "
            f"{b['C3-C4']:7.2f}               {o['walls_angle_deg']:.3f}       {f['median_px']:.4f}, "
            f"{f['rms_inliers_px']:.4f}, {f['wand_abs_median_mm']:.4f}, {f['wand_clean_mean_mm']:.4f}, "
            f"{f['wand_clean_sd_mm']:.4f}, {f['outlier_detections']}")
    if "free" in grids and "w2_mid" in grids:
        res["residual_grid"] = {}
        log("  residual maps (w1_residual_maps.png): per camera, the RMS over cells of the mean residual (px), and of "
            "its change from free to w2_mid")
        for c in range(obs.n_cams):
            a, b = grids["free"][c]["mean"], grids["w2_mid"][c]["mean"]
            dm = b - a
            rms = lambda m: float(np.sqrt(np.nanmean(np.sum(m ** 2, axis=2))))
            log(f"    C{c + 1}: free {rms(a):.4f}, w2_mid {rms(b):.4f}, change {rms(dm):.4f}; largest change "
                f"{np.nanmax(np.linalg.norm(dm, axis=2)):.4f} px")
            res["residual_grid"][f"C{c + 1}"] = {"free": np.round(a, 5).tolist(), "w2_mid": np.round(b, 5).tolist(),
                                                 "count": grids["free"][c]["count"].tolist()}
        plot_grids(fits["free"]["cams"], grids)
    with open(os.path.join(HERE, "w1_channel_hold.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda v: None if v is None else float(v))
    log.close()


def plot_grids(cams, grids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("free", grids["free"]), ("w2_mid", grids["w2_mid"]),
            ("change, w2_mid - free", [{"mean": b["mean"] - a["mean"]} for a, b in zip(grids["free"],
                                                                                   grids["w2_mid"])])]
    k = 500.0                                              # arrow length in image px per px of residual
    fig, axes = plt.subplots(3, len(cams), figsize=(4.2 * len(cams), 12))
    for i, (title, g) in enumerate(rows):
        for c, cam in enumerate(cams):
            ax = axes[i, c]
            m = g[c]["mean"]
            cu = (np.arange(GRID) + 0.5) * cam.width / GRID
            cv = (np.arange(GRID) + 0.5) * cam.height / GRID
            U, Vv = np.meshgrid(cu, cv)
            ax.quiver(U, Vv, m[:, :, 0] * k, m[:, :, 1] * k, angles="xy", scale_units="xy", scale=1.0,
                      color="C0" if i < 2 else "C3", width=0.006)
            ax.quiver([150], [cam.height - 120], [0.5 * k], [0], angles="xy", scale_units="xy", scale=1.0,
                      color="k", width=0.006)
            ax.text(150, cam.height - 200, "0.5 px", fontsize=8)
            ax.set_xlim(0, cam.width)
            ax.set_ylim(cam.height, 0)
            ax.set_aspect("equal")
            ax.set_title(f"C{c + 1}: {title}", fontsize=10)
    fig.suptitle("W1: mean residual per image cell (8 x 8), T2's frames; w2_mid = wall 2 held where the channel "
                 "centres the wand", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "w1_residual_maps.png"), dpi=70)


if __name__ == "__main__":
    if "--case" in sys.argv:
        run_case(sys.argv[sys.argv.index("--case") + 1])
    elif "--summary" in sys.argv:
        summary()
    else:
        run_all()
        summary()
