"""T5 (added on 6 Oct at your question: "if we also fit k1, will that helps to reduce the projection error?"): the
delivered calibration with k1 free, T0's T5 on T2.

    python tests/t5_k1.py      (after calibrate.py and tests/t2_residuals.py)

cylinder_wand_gui.run with fit_k1 (each camera's first radial distortion coefficient, OpenCV's convention: a point
rho px from the principal point moves outward by about k1 (rho / f)^2 rho px) on all of S2's frames; the GUI's files
go to <out_dir>/checks/fit_k1/.  Compared with S2:
  - the reprojection RMS, overall and per camera, and the wand-length scatter of a free triangulation;
  - k1 and its sigma per camera, and what it does at the image's side edge (rho = 640 px) and corner (755 px);
  - f and the distance from the axis;
  - the residual pattern over the image as T2 measures it: the largest cell mean and the cells above 0.1 px.
Judged by (T0's T5): k1 is kept only if it clearly lowers the residuals, and is plausible for the lens.
Writes t5_k1_out.txt and t5_k1.json beside this script.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

NU, NV, MIN_CELL = 10, 6, 20          # as tests/t2_residuals.py


def cells(folder, cfg):
    """Per camera: (largest cell mean, cells above 0.1 px, cells) of the active residuals, T2's binning."""
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        ob = json.load(fh)["observations"]
    cam = np.array(ob["cam_id"])
    u, v = np.array(ob["u"], float), np.array(ob["v"], float)
    du = np.array([np.nan if x is None else x for x in ob["du"]], float)
    dv = np.array([np.nan if x is None else x for x in ob["dv"]], float)
    act = np.array(ob["active"], bool) & np.isfinite(du)
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    iu = np.clip((u * NU / W).astype(int), 0, NU - 1)
    iv = np.clip((v * NV / H).astype(int), 0, NV - 1)
    out = {}
    for c in sorted(set(cam)):
        means = []
        for a in range(NU):
            for b in range(NV):
                s = act & (cam == c) & (iu == a) & (iv == b)
                if s.sum() >= MIN_CELL:
                    means.append(math.hypot(du[s].mean(), dv[s].mean()))
        out[f"cam{c}"] = {"largest_px": max(means), "above_0p1": sum(1 for m in means if m > 0.1), "cells": len(means)}
    return out


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "t5_k1"))
    with open(os.path.join(os.path.dirname(HERE), "calibrate.json"), encoding="utf-8") as fh:
        ref = json.load(fh)["summary"]
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    cams = case.cam_ids(cfg)
    lib = case.library()
    job = case.make_job(lib, cfg, frames, case.uv_array(kept, frames, cams), fit_k1=True)
    folder = os.path.join(cfg["out_dir"], "checks", "fit_k1")
    log(f"T5: the calibration with k1 free, on S2's {len(frames)} frames")
    _sol, result, secs = case.run_calibration(lib, job, folder, log)
    s = case.summary(result)
    c_ref, c_k1 = cells(cfg["out_dir"], cfg), cells(folder, cfg)
    log(f"  done in {secs:.0f} s; written to {folder}")
    log(f"  reprojection RMS {ref['rms_px']:.4f} -> {s['rms_px']:.4f} px ({100 * (s['rms_px'] / ref['rms_px'] - 1):+.1f} %); "
        f"detections used {ref['active']} -> {s['active']} of {s['observations']}")
    log(f"  wand length, free triangulation: RMS {ref['wand_free_rms_um']:.2f} -> {s['wand_free_rms_um']:.2f} um, "
        f"mean {ref['wand_free_mean_um']:+.2f} -> {s['wand_free_mean_um']:+.2f} um")
    out = {"S2": ref, "fit_k1": s, "seconds": secs, "cameras": {}}
    for cr, ck in zip(ref["cameras"], s["cameras"]):
        k1, sk = ck["k1"], ck["k1_sigma"] or float("nan")
        f = ck["f_px"]
        edge = k1 * (640.0 / f) ** 2 * 640.0
        corner = k1 * (755.0 / f) ** 2 * 755.0
        name = ck["name"]
        rec = {"k1": k1, "k1_sigma": sk, "shift_edge_px": edge, "shift_corner_px": corner,
               "rms_px": [cr["rms_px"], ck["rms_px"]], "f_px": [cr["f_px"], ck["f_px"]],
               "distance_mm": [cr["distance_mm"], ck["distance_mm"]], "cells_S2": c_ref[name], "cells_k1": c_k1[name]}
        out["cameras"][name] = rec
        log(f"  {name}: k1 {k1:+.3f} +- {sk:.3f} ({k1 / sk:+.1f} sigma): {edge:+.2f} px at the side edge (640 px), "
            f"{corner:+.2f} px at the corner; RMS {cr['rms_px']:.4f} -> {ck['rms_px']:.4f} px; f {cr['f_px']:,.0f} -> "
            f"{ck['f_px']:,.0f} px; distance {cr['distance_mm']:,.1f} -> {ck['distance_mm']:,.1f} mm")
        log(f"      residual cells (T2's binning): largest mean {c_ref[name]['largest_px']:.3f} -> "
            f"{c_k1[name]['largest_px']:.3f} px; above 0.1 px {c_ref[name]['above_0p1']} of {c_ref[name]['cells']} -> "
            f"{c_k1[name]['above_0p1']} of {c_k1[name]['cells']}")
    if result.get("checks"):
        log("  checks on the results page: " + "; ".join(c["title"] for c in result["checks"]))
    log.save(out)


if __name__ == "__main__":
    main()
