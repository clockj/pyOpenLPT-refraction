"""Any calibration of the rig judged on wand detections, the same way for all: each frame's two ends triangulated
freely (maximum likelihood, every camera that sees the end, no length constraint), then each detection's
reprojection error and each frame's wand length.

    python evaluate.py calib_3                     # one of the GUI's calibrations (config gui_calibrations)
    python evaluate.py X:\\...\\calib_plate_wand\\camFile --detections gui_5563
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

import case


def triangulate_all(cams, rig, uv: np.ndarray, L: float, sizes):
    """Free ML triangulation of every frame of uv[frame, camera, end, (u, v)]: (obs, X (2F, 3) ends A, B
    interleaved, residuals (n, 2))."""
    data = case.pw.WandData(uv, L, sizes)
    obs = case.pw.observations(data)
    rows = np.arange(len(obs))
    XA, XB, _cnt = case.pw.triangulate(cams, rig, obs, rows=rows)
    X0 = np.empty((2 * obs.n_frames, 3))
    X0[0::2], X0[1::2] = XA, XB
    X0 = np.where(np.isfinite(X0), X0, 0.0)
    X, r, _cov = case.pw.triangulate_ml(cams, rig, obs.cam, obs.uv, obs.frame * 2 + obs.end, X0)
    return obs, X, r


def _stats(x: np.ndarray) -> Dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(np.mean(x)), "std": float(np.std(x)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)), "max": float(np.max(x))}


OUTLIER_PX = 3.0      # a detection this far (Euclidean) from its free triangulation's projection is a gross error


def evaluate(cams, rig, uv: np.ndarray, L: float, sizes, frame_ids=None, keep_arrays: bool = False) -> Dict:
    """Reprojection errors (px) per camera and wand lengths (mm) of the free triangulation of every frame.  Besides
    the plain statistics (which a few gross detection errors dominate), the inliers': detections within OUTLIER_PX,
    and frames all of whose detections are inliers."""
    obs, X, r = triangulate_all(cams, rig, uv, L, sizes)
    e = np.linalg.norm(r, axis=1)
    inl = np.isfinite(e) & (e < OUTLIER_PX)
    clean = np.ones(obs.n_frames, bool)
    clean[obs.frame[~inl]] = False
    out = {"L": float(L), "frames": int(obs.n_frames), "detections": int(len(obs)),
           "failed_projections": int(np.sum(~np.isfinite(e))), "outlier_px": OUTLIER_PX, "outlier_detections": int(np.sum(~inl)), "clean_frames": int(clean.sum()),
           "per_camera": []}
    for c in range(len(cams)):
        sel = obs.cam == c
        rc = r[sel]
        ok = np.all(np.isfinite(rc), axis=1)
        ic = inl[sel]
        out["per_camera"].append({
            "camera": c, "euclidean_px": _stats(e[sel]),
            "rms_per_coordinate_px": float(np.sqrt(np.mean(rc[ok] ** 2))) if ok.any() else None,
            "rms_inliers_px": float(np.sqrt(np.mean(rc[ic] ** 2))) if ic.any() else None,
            "outliers": int(np.sum(~ic)),
            "mean_offset_px": rc[ic].mean(axis=0).tolist() if ic.any() else None})
    allok = np.all(np.isfinite(r), axis=1)
    out["rms_per_coordinate_px"] = float(np.sqrt(np.mean(r[allok] ** 2)))
    out["rms_inliers_px"] = float(np.sqrt(np.mean(r[inl] ** 2)))
    out["euclidean_px"] = _stats(e)
    Lm = np.linalg.norm(X[1::2] - X[0::2], axis=1)
    err = Lm - L
    out["wand_length_mm"] = _stats(Lm)
    out["wand_error_mm"] = {"mean": float(np.nanmean(err)), "std": float(np.nanstd(err)),
                            "abs": _stats(np.abs(err)), "frames_over_0.1mm": int(np.sum(np.abs(err) > 0.1)),
                            "frames_over_0.2mm": int(np.sum(np.abs(err) > 0.2)),
                            "clean_frames": {"mean": float(np.mean(err[clean])), "std": float(np.std(err[clean])),
                                             "abs": _stats(np.abs(err[clean]))}}
    fid = np.asarray(frame_ids)[obs.frame_ids] if frame_ids is not None else obs.frame_ids
    worst = np.argsort(-np.nan_to_num(np.abs(err), nan=np.inf))[:10]
    out["worst_frames"] = [{"frame": int(fid[k]), "wand_error_mm": float(err[k])} for k in worst]
    if keep_arrays:
        out["_arrays"] = {"obs": obs, "X": X, "r": r, "wand_error": err, "frame_ids": fid}
    return out


def text(name: str, ev: Dict) -> str:
    w = ev["wand_error_mm"]
    wc = w["clean_frames"]
    lines = [f"{name}: {ev['frames']} frames, {ev['detections']} detections"
             + (f", {ev['failed_projections']} failed projections" if ev["failed_projections"] else "")
             + f"; {ev['outlier_detections']} detections over {ev['outlier_px']:g} px (gross errors), "
               f"{ev['clean_frames']} frames without one",
             f"  reprojection: inliers RMS {ev['rms_inliers_px']:.4f} px per coordinate (all: "
             f"{ev['rms_per_coordinate_px']:.3f}); Euclidean median {ev['euclidean_px']['median']:.4f}, p99 "
             f"{ev['euclidean_px']['p99']:.3f}, max {ev['euclidean_px']['max']:.1f} px"]
    for pc in ev["per_camera"]:
        s = pc["euclidean_px"]
        lines.append(f"    cam{pc['camera']}: inliers RMS {pc['rms_inliers_px']:.4f} px/coord., Euclidean median "
                     f"{s['median']:.4f}, p99 {s['p99']:.3f} px, {pc['outliers']} outliers; inliers' mean offset "
                     f"({pc['mean_offset_px'][0]:+.3f}, {pc['mean_offset_px'][1]:+.3f}) px")
    lines.append(f"  wand length, frames without a gross error: mean {L_of(ev) + wc['mean']:.4f} mm, SD "
                 f"{wc['std']:.4f}; |error| mean {wc['abs']['mean']:.4f}, median {wc['abs']['median']:.4f}, p90 "
                 f"{wc['abs']['p90']:.4f}, p99 {wc['abs']['p99']:.4f}, max {wc['abs']['max']:.3f} mm")
    lines.append(f"  wand length, every frame: median {ev['wand_length_mm']['median']:.4f} mm; |error| mean "
                 f"{w['abs']['mean']:.4f}, p99 {w['abs']['p99']:.4f}, max {w['abs']['max']:.2f} mm; frames over "
                 f"0.1 / 0.2 mm: {w['frames_over_0.1mm']} / {w['frames_over_0.2mm']}")
    return "\n".join(lines)


def L_of(ev: Dict) -> float:
    """The wand length the evaluation was made with."""
    return ev["L"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration", help="a GUI calibration's name in config.json, or a camFile folder")
    ap.add_argument("--detections", default="all", help="all, gui_8962 or gui_5563 (config detections)")
    args = ap.parse_args()
    cfg = case.load_config()
    if args.calibration in cfg["gui_calibrations"]:
        cams, rig = case.load_gui(cfg, args.calibration)
    else:
        cams, wins = [], {}
        for c, cc in enumerate(cfg["rig"]["cameras"]):
            cam, win = case.pio.load_camfile(os.path.join(args.calibration, f"{cc['name']}.txt"))
            cams.append(cam)
            wins.setdefault(cfg["rig"]["cam_to_window"][c], win)
        rig = case.pw.Rig([wins[w] for w in sorted(wins)], list(cfg["rig"]["cam_to_window"]))
    rc = case.rig_config(cfg)
    cd = case.read_detections(cfg, args.detections)
    ev = evaluate(cams, rig, cd.uv, rc.wand_length_mm, rc.sizes, frame_ids=cd.frame_ids)
    print(text(args.calibration, ev))
    print(json.dumps(case.describe(cams, rig), indent=1)[:2000])


if __name__ == "__main__":
    main()
