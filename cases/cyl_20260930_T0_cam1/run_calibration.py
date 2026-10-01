"""End to end: detect, index, fit the cylinder model and write every output.

    python run_calibration.py                 # the calibration (model M1bx) and all outputs in config["out_dir"]
    python run_calibration.py --no-outputs    # fit only (prints the summary)
    python run_calibration.py --model M2b --no-outputs --state S.pkl   # another model, state saved for diagnosis

Steps
  1. detect the dots of the 20 planes (dots.py) and index them by tracking (grid_index.py);
  2. multi-start (32 starts over camera distance, axis depth and camera lateral position) with the fast
     back-projection residual, model M1;
  3. forward-projection refinement of the 3 best starts with M1, then the model from the best.  M1bx, delivered:
     the planes at their stage positions (2.54 mm steps), a sideways shift of each plane's board (the stage
     carriage's play, a few um; no common shift and no trend in z) and per-dot board corrections (the board's dots
     are off the nominal grid by 10-50 um, tests/v1_2mm_steps/d1-d2).  M1b leaves out the plane shifts; M2b frees
     every plane's position instead (the first run's model);
  4. outliers beyond 5 robust sigma removed, the model refitted, covariance from the Jacobian;
  5. outputs: camFile/cam1.txt (pyOpenLPT PINCYL), calibration CSVs, per-plane dot tables, overlays, figures,
     calibration_result.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402
import dots as D                              # noqa: E402
import grid_index as G                        # noqa: E402
import viz                                    # noqa: E402

ESTIMATORS = {"halfmax": ("u", "v"), "iso": ("u_iso", "v_iso"), "gradient": ("u_grad", "v_grad"),
              "sat": ("u_sat", "v_sat")}


# ------------------------------------------------------------------ observations
def collect(cfg: dict) -> list:
    """Detect and index every plane.  Returns one dict per plane (increasing z)."""
    det = cfg["detect"]
    planes = []
    for (z, path), lab in zip(case.plane_files(cfg), case.plane_labels(cfg)):
        img = case.read_image(path)
        r = D.detect(img, det)
        dots = r["dots"]
        P = np.array([[d["u"], d["v"]] if d["ok"] else [d["u_rough"], d["v_rough"]] for d in dots], float)
        ok = np.array([d["ok"] for d in dots], bool)
        planes.append({"z": z, "label": lab, "path": path, "dots": dots, "P": P, "ok": ok})
    idx = G.index_planes(planes, cfg["frame"]["origin_row_v"], cfg["index"]["track_gate_px"],
                         cfg["index"]["lattice_tol_frac"])
    for k, pl in enumerate(planes):
        pl["ij"] = idx["ij"][k]
        pl["lattice"] = idx["lattice"][k]
    planes[0]["index_checks"] = idx["checks"]
    return planes


def build_obs(planes: list, cfg: dict, estimator: str = "halfmax", drop=None, keep_planes=None):
    """Observations from the usable dots.  Returns (Obs, keys) with keys[n] = (plane k, dot index).

    With keep_planes, only those planes are used and renumbered 0.. in increasing z; obs.plane_ids maps them back.
    The first kept plane then holds dz = 0.
    """
    cu, cv = ESTIMATORS[estimator]
    kp = list(range(len(planes))) if keep_planes is None else sorted(keep_planes)
    remap = {k: q for q, k in enumerate(kp)}
    rows, keys = [], []
    for k in kp:
        pl = planes[k]
        for n, d in enumerate(pl["dots"]):
            if not d["ok"] or (drop and (k, n) in drop):
                continue
            u, v = d[cu], d[cv]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            rows.append((remap[k], pl["ij"][n][0], pl["ij"][n][1], u, v))
            keys.append((k, n))
    a = np.array(rows, float)
    z_nom = np.array([planes[k]["z"] for k in kp], float)
    obs = CF.Obs(plane=a[:, 0].astype(int), i=a[:, 1], j=a[:, 2], uv=a[:, 3:5], z_nom=z_nom,
                 pitch=tuple(cfg["target"]["pitch_mm"]), plane_ids=np.array(kp))
    return obs, keys


def vertical_scale(planes: list, cfg: dict) -> float:
    """Image scale along Y at plane 0 (px per mm), from its fitted lattice."""
    _p0, A = planes[0]["lattice"]
    return float(np.linalg.norm(A[:, 1])) / float(cfg["target"]["pitch_mm"][1])


def scale_from_obs(obs) -> float:
    """The same scale from the observations of the first plane (affine fit of uv against i, j)."""
    m = obs.plane == 0
    G_ = np.column_stack([np.ones(m.sum()), obs.i[m], obs.j[m]])
    coef, *_ = np.linalg.lstsq(G_, obs.uv[m], rcond=None)
    return float(np.linalg.norm(coef[2])) / float(obs.pitch[1])


# ------------------------------------------------------------------ the fit
N_JOBS = min(16, os.cpu_count() or 1)


def _single_thread_env():
    """Workers must not each start a full BLAS/OpenMP thread pool: 16 workers x 32 threads thrash the machine.
    Spawned workers read these at numpy import, so they are set before the pool starts."""
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"


def parallel_map(fn, items, n_jobs: int = N_JOBS) -> list:
    """map() over worker processes (each builds its own C++ cameras); serial when n_jobs <= 1."""
    items = list(items)
    if n_jobs <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    from concurrent.futures import ProcessPoolExecutor
    _single_thread_env()
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(items))) as ex:
        return list(ex.map(fn, items))


def _multistart_one(args):
    s, obs, blocks = args if len(args) == 3 else (*args, CF.MODELS["M1"])
    try:
        fr = CF.fit(s, obs, blocks, residual="backproject", max_outer=4, max_nfev=300)
        bp = CF.backproject_residual(fr.rig, obs).reshape(-1, 2)
        return (float(np.sqrt(np.mean(np.sum(bp ** 2, axis=1)))), fr.rms, fr, s)
    except Exception:                                # a start can leave the model's domain
        return (float("inf"), float("inf"), None, s)


def _fit_one(args):
    """Forward fit of the given blocks from a rig: args = (rig, obs, blocks)."""
    rig, obs, blocks = args
    return CF.fit(rig, obs, blocks)


def multistart(obs, cfg, scale_v0, verbose=True, n_jobs: int = N_JOBS, starts=None, blocks=None) -> list:
    """Back-projection fits (M1 unless `blocks`) from every start; returns [(bp_rms, fwd_rms, FitResult, start)]
    sorted by the back-projection rms."""
    starts = starts if starts is not None else CF.start_rigs(obs, cfg, scale_v0)
    blocks = blocks or CF.MODELS["M1"]
    table = parallel_map(_multistart_one, [(s, obs, blocks) for s in starts], n_jobs)
    table.sort(key=lambda t: t[0])
    return table


def robust_outliers(resid: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Rows whose u or v residual exceeds k robust sigmas (1.4826 MAD)."""
    out = np.zeros(len(resid), bool)
    for a in (0, 1):
        r = resid[:, a]
        s = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-6
        out |= np.abs(r - np.median(r)) > k * s
    return out


def fit_obs(obs, cfg, scale_v0, model="M2", verbose=True, start_rig=None, outlier_k=5.0) -> dict:
    """Fit `model` to the observations: multi-start (unless start_rig is given), M1, then the model, then one
    outlier pass.  Returns {"fit", "obs" (outliers removed), "kept" (mask over the input rows), "info"}."""
    t0 = time.time()
    info = {"model": model, "n_obs_initial": len(obs.plane)}
    if start_rig is None:
        table = multistart(obs, cfg, scale_v0, verbose)
        info["multistart"] = [{"bp_rms": t[0], "fwd_rms": t[1],
                               "start_L": float(np.linalg.norm((t[3].C - t[3].axis_pt)[[0, 2]])),
                               "start_Za": float(t[3].axis_pt[2]), "start_Cx": float(t[3].C[0]),
                               "end_f": (t[2].rig.f if t[2] else None),
                               "end_C": (t[2].rig.C.tolist() if t[2] else None)} for t in table]
        if verbose:
            print(f"  multi-start: best back-projection rms {table[0][0]:.3f} px "
                  f"(forward {table[0][1]:.3f} px); worst finite {max(t[0] for t in table if np.isfinite(t[0])):.2f}")
        m1_blocks = CF.MODELS["M1" if "fixf" not in model else "M1_fixf"]
        cands = parallel_map(_fit_one, [(t[2].rig, obs, m1_blocks) for t in table[:3] if t[2] is not None])
        m1 = min(cands, key=lambda fr: fr.rms)
        info["M1_candidates_rms"] = [fr.rms for fr in cands]
    else:
        start_rig = CF.restrict(start_rig, CF.MODELS[model])
        m1 = CF.fit(start_rig, obs, CF.MODELS["M1" if "fixf" not in model else "M1_fixf"], verbose=verbose)
    info["M1_rms"] = m1.rms
    fr = m1
    if model not in ("M1", "M1_fixf"):
        fr = CF.fit(m1.rig, obs, CF.MODELS[model], verbose=verbose)
    # outliers, then the final fit
    kept = np.ones(len(obs.plane), bool)
    if outlier_k:
        bad = robust_outliers(fr.resid, outlier_k)
        if bad.any():
            kept = ~bad
            obs = obs.subset(kept)
            fr = CF.fit(fr.rig, obs, CF.MODELS[model], verbose=verbose)
    fr = CF.covariance(fr, obs)
    info.update(rms=fr.rms, n_obs=int(kept.sum()), n_outliers=int((~kept).sum()), n_fail=fr.n_fail,
                seconds=time.time() - t0)
    if verbose:
        print(f"  {model}: rms {fr.rms:.4f} px over {int(kept.sum())} dots ({int((~kept).sum())} outliers "
              f"dropped), {time.time() - t0:.0f} s")
    return {"fit": fr, "obs": obs, "kept": kept, "info": info}


def calibrate(planes, cfg, estimator="halfmax", model="M2", verbose=True, keep_planes=None, start_rig=None,
              outlier_k=5.0) -> dict:
    """The full fit on detected dots.  With start_rig given, the multi-start is skipped and the fit starts there.
    Returns {"fit", "obs", "keys" (plane, dot index) per row, "drop" (outlier keys), "info"}."""
    obs, keys = build_obs(planes, cfg, estimator, keep_planes=keep_planes)
    out = fit_obs(obs, cfg, vertical_scale(planes, cfg), model, verbose, start_rig, outlier_k)
    kept = out["kept"]
    out["drop"] = {keys[n] for n in np.where(~kept)[0]}
    out["keys"] = [keys[n] for n in np.where(kept)[0]]
    out["info"]["estimator"] = estimator
    return out


# ------------------------------------------------------------------ outputs
def per_plane_rms(fr, obs) -> list:
    """Per-plane residual statistics; the means are of detected minus model (fr.resid is model minus detected)."""
    out = []
    for k in range(obs.n_planes):
        m = obs.plane == k
        r = -fr.resid[m]
        out.append({"plane": k, "z_nominal": float(obs.z_nom[k]), "n": int(m.sum()),
                    "rms_px": float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))) if m.any() else None,
                    "rms_u": float(np.sqrt(np.mean(r[:, 0] ** 2))) if m.any() else None,
                    "rms_v": float(np.sqrt(np.mean(r[:, 1] ** 2))) if m.any() else None,
                    "mean_u": float(np.mean(r[:, 0])) if m.any() else None,
                    "mean_v": float(np.mean(r[:, 1])) if m.any() else None})
    return out


def spacing_check(obs, fr, cfg) -> dict:
    """The model-free check of f and the pinhole distance (figures.dot_spacing): the vertical dot spacing against
    the plane position, and how much it would grow with the pinhole at the lens front or the camera body."""
    import figures
    rig = fr.rig
    z = rig.plane_z(obs.z_nom)
    sv, sh, nv, nh = figures.spacing_per_plane(obs, obs.uv)
    pv, ph, _, _ = figures.spacing_per_plane(obs, obs.uv - fr.resid)
    za = float(rig.axis_at_y0()[2])
    setup = cfg.get("setup", {})
    out = {"vertical_spacing_px": sv.tolist(), "horizontal_spacing_px": sh.tolist(), "pairs": [nv, nh],
           "vertical_growth_measured": float(sv[-1] / sv[0] - 1), "vertical_growth_model": float(pv[-1] / pv[0] - 1),
           "horizontal_change_measured": float(sh[-1] / sh[0] - 1),
           "horizontal_change_model": float(ph[-1] / ph[0] - 1),
           "line_through_1_over_vertical_spacing": figures.spacing_line(z, sv, cfg)}
    for key in ("lens_front_to_tube_wall_mm", "camera_body_front_to_tube_wall_mm"):
        if setup.get(key) is not None:
            out[f"vertical_growth_if_pinhole_at_{key.replace('_to_tube_wall_mm', '')}"] = float(
                figures.pinhole_growth([z[-1] - z[0]], setup[key], cfg, za - z[0])[0])
    return out


def git_rev(path) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_outputs(planes, cfg, res, extra: dict = None):
    out = case.ensure_dir(cfg["out_dir"])
    for sub in ("camFile", "dots", "overlays", "figures"):
        case.ensure_dir(os.path.join(out, sub))
    fr, obs, keys, drop = res["fit"], res["obs"], res["keys"], res["drop"]
    rig = fr.rig
    cam_id = int(cfg["camera_id"])
    cam = rig.camera()
    cam_path = os.path.join(out, "camFile", f"cam{cam_id}.txt")
    cam.saveParameters(cam_path)

    # predictions for every usable dot (outliers included) and every border dot with an index; the residual columns
    # of the dot tables are detected minus model, as the overlays draw them (fr.resid is model minus detected)
    resid_of = {keys[n]: -fr.resid[n] for n in range(len(keys))}
    z_fit = rig.plane_z(obs.z_nom)
    P_fit = rig.world_points(obs)                    # model plane positions, board corrections included
    # calibration point tables: nominal (as the GUI would build them), fitted Z (only when the model fits the plane
    # positions), and the model's positions with the corrected board
    hdr = ["CameraID", "ImagePath", "Plane", "PixelX", "PixelY", "WorldX", "WorldY", "WorldZ"]
    files = {"nominal": "calibration_points.csv"}
    if any(b in fr.layout.blocks for b in ("dz", "zs", "tau")):
        files["fittedZ"] = "calibration_points_fittedZ.csv"
    if rig.board is not None:
        files["corrected"] = "calibration_points_corrected.csv"
    handles = {k: open(os.path.join(out, v), "w", newline="") for k, v in files.items()}
    writers = {k: csv.writer(h) for k, h in handles.items()}
    for w in writers.values():
        w.writerow(hdr)
    for n, (k, dn) in enumerate(keys):
        pl = planes[k]
        X, Y = cfg["target"]["pitch_mm"][0] * obs.i[n], cfg["target"]["pitch_mm"][1] * obs.j[n]
        u, v = obs.uv[n]
        base = [cam_id, pl["path"], k, f"{u:.4f}", f"{v:.4f}"]
        writers["nominal"].writerow(base + [f"{X:.4f}", f"{Y:.4f}", f"{obs.z_nom[k]:.4f}"])
        if "fittedZ" in writers:
            writers["fittedZ"].writerow(base + [f"{X + rig.tau[0] * z_fit[k]:.4f}",
                                                f"{Y + rig.tau[1] * z_fit[k]:.4f}", f"{z_fit[k]:.4f}"])
        if "corrected" in writers:
            writers["corrected"].writerow(base + [f"{P_fit[n][0]:.4f}", f"{P_fit[n][1]:.4f}", f"{P_fit[n][2]:.4f}"])
    for h in handles.values():
        h.close()
    if rig.board is not None:
        sig_b = None
        if fr.cov is not None and ("board" in fr.layout.blocks or "boardcr" in fr.layout.blocks):
            sl = fr.layout.slices.get("board", fr.layout.slices.get("boardcr"))
            sig_b = np.sqrt(np.diag(fr.layout.Nb @ fr.cov[sl, sl] @ fr.layout.Nb.T)).reshape(-1, 2)
        with open(os.path.join(out, "board_corrections.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["i", "j", "X_nominal_mm", "Y_nominal_mm", "dX_mm", "dY_mm", "X_corrected_mm",
                        "Y_corrected_mm", "sigma_dX_mm", "sigma_dY_mm"])
            for q, (i, j) in enumerate(rig.board_ij):
                Xn, Yn = cfg["target"]["pitch_mm"][0] * i, cfg["target"]["pitch_mm"][1] * j
                dX, dY = rig.board[q]
                sx, sy = (sig_b[q] if sig_b is not None else (np.nan, np.nan))
                w.writerow([int(i), int(j), f"{Xn:.4f}", f"{Yn:.4f}", f"{dX:+.5f}", f"{dY:+.5f}", f"{Xn + dX:.5f}",
                            f"{Yn + dY:.5f}", f"{sx:.5f}", f"{sy:.5f}"])

    # per-plane dot tables and overlays
    for k, pl in enumerate(planes):
        tag = case.plane_tag(pl["label"])
        ij = np.array(pl["ij"], float).reshape(-1, 2)
        obs_k = CF.Obs(np.full(len(ij), k), ij[:, 0], ij[:, 1], np.zeros((len(ij), 2)), obs.z_nom, obs.pitch)
        pred = CF.project(cam, rig.world_points(obs_k))
        flags = []
        with open(os.path.join(out, "dots", tag + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["i", "j", "kind", "used_in_fit", "outlier", "u", "v", "u_iso", "v_iso", "u_grad", "v_grad",
                        "u_sat", "v_sat", "u_rough", "v_rough", "ellipse_a", "ellipse_b", "ellipse_theta_rad",
                        "edge_rms", "core_darkness", "u_pred", "v_pred", "res_u", "res_v"])
            for n, d in enumerate(pl["dots"]):
                used = (k, n) in resid_of
                out_flag = (k, n) in drop
                r = resid_of.get((k, n), (np.nan, np.nan))
                if out_flag:
                    r = (d["u"] - pred[n][0], d["v"] - pred[n][1])
                flags.append("out" if out_flag else "")
                w.writerow([pl["ij"][n][0], pl["ij"][n][1], d["kind"], int(used), int(out_flag)] +
                           [f"{d[c]:.4f}" for c in ("u", "v", "u_iso", "v_iso", "u_grad", "v_grad", "u_sat",
                                                    "v_sat", "u_rough", "v_rough", "a", "b", "theta",
                                                    "edge_rms", "core")] +
                           [f"{pred[n][0]:.4f}", f"{pred[n][1]:.4f}", f"{r[0]:.4f}", f"{r[1]:.4f}"])
        img = case.read_image(pl["path"])
        zt = f"z = {pl['z']:.2f} mm" + (f" (fit {z_fit[k]:.3f})" if "fittedZ" in files else "")
        vis = viz.draw_dots(img, pl["dots"], [tuple(v) for v in pl["ij"]], pred=pred, flags=flags,
                            title=f"{os.path.basename(pl['path'])}: {zt}; arrows: residual x100; cyan: model")
        cv2.imwrite(os.path.join(out, "overlays", tag + ".png"), vis)

    # the result file
    sig = CF.physical_sigmas(fr)
    result = {
        "case": cfg["case"], "date": time.strftime("%Y-%m-%d %H:%M"),
        "camera_file": cam_path, "model": res["info"]["model"],
        "frame": cfg["frame"], "units": "mm, px (u = column, v = row, pixel centres at integers)",
        "port": cfg["port"], "target": cfg["target"], "setup": cfg.get("setup"),
        "plane_labels": cfg["planes"]["labels"], "planes_nominal_mm": cfg["planes"]["z_mm"],
        "intrinsics": {"f_px": rig.f, "f_mm_at_20um": rig.f * cfg["image"]["pixel_mm"],
                       "cx": rig.cx, "cy": rig.cy, "dist": rig.dist, "fixed": ["cx", "cy", "dist"]},
        "rig": rig.summary(), "sigma": sig, "corrections": CF.extras_summary(rig),
        "fast_vs_cpp_px": fr.fast_vs_cpp,
        "planes_fitted_z_mm": z_fit.tolist(),
        "plane_shifts_mm": (rig.shift.tolist() if rig.shift is not None else None),
        "residuals": {"rms_px": fr.rms, "rms_u": float(np.sqrt(np.mean(fr.resid[:, 0] ** 2))),
                      "rms_v": float(np.sqrt(np.mean(fr.resid[:, 1] ** 2))),
                      "max_px": float(np.max(np.linalg.norm(fr.resid, axis=1))),
                      "per_plane": per_plane_rms(fr, obs)},
        "spacing_check": spacing_check(obs, fr, cfg),
        "fit_info": res["info"], "parameter_names": fr.layout.names(),
        "correlation_max_offdiag": float(np.max(np.abs(CF.correlations(fr) - np.eye(fr.layout.size)))),
        "software": {"pyOpenLPT_commit": git_rev(case.REPO), "python": platform.python_version(),
                     "numpy": np.__version__},
    }
    if extra:
        result.update(extra)
    with open(os.path.join(out, "calibration_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    save_state(res, os.path.join(out, "fit_state.pkl"))
    return result


def save_state(res: dict, path: str):
    """The fitted rig, observations and covariance, for the tests (pickle; needs cylfit importable)."""
    import pickle
    fr = res["fit"]
    state = {"rig": fr.rig, "blocks": fr.layout.blocks, "obs": res["obs"], "keys": res["keys"],
             "drop": sorted(res["drop"]), "cov": fr.cov, "sigma": fr.sigma, "resid": fr.resid, "rms": fr.rms,
             "info": res["info"], "names": fr.layout.names()}
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_state(path: str) -> dict:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def summarise(fr, obs):
    s = fr.rig.summary()
    sig = CF.physical_sigmas(fr)
    print(f"  f = {s['f_px']:.1f} +- {sig.get('f_px', float('nan')):.1f} px ({s['f_mm']:.2f} mm at 20 um)")
    print(f"  camera centre = {np.round(s['camera_centre_mm'], 3)} mm, +- {np.round(sig['camera_centre_mm'], 3)}")
    print(f"  camera distance from the axis = {s['camera_distance_from_axis_mm']:.2f} mm; viewing direction "
          f"{np.round(s['viewing_direction'], 5)} ({s['view_tilt_from_minusZ_deg']:.3f} deg from -Z)")
    print(f"  axis at Y=0: {np.round(s['axis_point_at_Y0_mm'], 3)} mm, +- {np.round(sig['axis_position_X_Z_mm'], 3)};"
          f" direction {np.round(s['axis_direction'], 6)}; tilt in board plane "
          f"{s['axis_tilt_in_board_plane_deg']:.3f} deg, lean towards camera {s['axis_lean_towards_camera_deg']:.3f}"
          f" deg (+- {np.round(sig['axis_tilt_lean_deg'], 3)})")
    if "dz_mm" in sig:
        print("  plane z (fitted - nominal), mm: " +
              " ".join(f"{z:.2f}:{d:+.3f}" for z, d in zip(obs.z_nom, fr.rig.dz)))
        print("  sigma dz, mm: " + " ".join(f"{s_:.3f}" for s_ in sig["dz_mm"]))
    if "stage_scale" in sig:
        print(f"  stage scale {fr.rig.zscale:.5f} +- {sig['stage_scale']:.5f} (plane k at scale x z_nominal_k)")
    ex = CF.extras_summary(fr.rig)
    if ex:
        print("  extras: " + ", ".join(f"{k} {np.round(v, 4)}" for k, v in ex.items()) +
              (f"; sigma board (median) {np.round(sig['board_mm_median'], 4)} mm" if "board_mm_median" in sig else "")
              + (f"; sigma wall (median) {sig['wall_mrad_median']:.4f} mrad" if "wall_mrad_median" in sig else ""))
    print(f"  fast vs C++ projection at the solution: {fr.fast_vs_cpp:.2e} px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-outputs", action="store_true")
    ap.add_argument("--estimator", default="halfmax", choices=list(ESTIMATORS))
    ap.add_argument("--model", default="M1bx", choices=list(CF.MODELS))
    ap.add_argument("--state", default=None, help="also save the fit state (pickle) to this path")
    ap.add_argument("--start", default=None, help="start from the rig in this state pickle (skips the multi-start)")
    ap.add_argument("--outputs-only", action="store_true",
                    help="rewrite every output from out_dir/fit_state.pkl without refitting")
    args = ap.parse_args()
    cfg = case.load_config()
    t0 = time.time()
    planes = collect(cfg)
    print(f"detected and indexed {sum(int(p['ok'].sum()) for p in planes)} dots in {len(planes)} planes "
          f"({time.time() - t0:.0f} s)")
    if args.outputs_only:
        st = load_state(os.path.join(cfg["out_dir"], "fit_state.pkl"))
        layout = CF.Layout(st["blocks"], st["obs"].n_planes, st["rig"], st["obs"].pitch, z=st["obs"].z_nom)
        fr = CF.FitResult(st["rig"], layout, st["resid"], st["rms"], cov=st["cov"], sigma=st["sigma"])
        rf = CF.forward_residual(fr.rig, st["obs"]).reshape(-1, 2)
        fr.fast_vs_cpp = float(np.max(np.abs(rf - fr.resid)))
        res = {"fit": fr, "obs": st["obs"], "keys": st["keys"], "drop": set(map(tuple, st["drop"])),
               "info": st["info"]}
        result = write_outputs(planes, cfg, res)
        import figures
        figures.make_all(cfg, planes, res, result)
        print(f"outputs rewritten from the saved fit ({time.time() - t0:.0f} s)")
        return
    start = None
    if args.start:
        start = load_state(args.start)["rig"]
        if len(start.dz) != len(planes):
            start.dz = np.zeros(len(planes))
    res = calibrate(planes, cfg, args.estimator, args.model, start_rig=start)
    summarise(res["fit"], res["obs"])
    if args.state:
        save_state(res, args.state)
    if not args.no_outputs:
        result = write_outputs(planes, cfg, res)
        print(f"outputs written to {cfg['out_dir']}")
        try:
            import figures
            figures.make_all(cfg, planes, res, result)
        except ImportError:
            pass
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
