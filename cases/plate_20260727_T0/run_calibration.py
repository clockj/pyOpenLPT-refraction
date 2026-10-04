"""The case's calibration: pyOpenLPT's flat-window wand calibration (plate_wand.calibrate, unchanged) on the
configured wand detections, then the outputs (PINPLATE camFiles, result JSON, report, evaluation, README).

    python run_calibration.py                                   # T2: every frame of "all"; outputs to config out_dir
    python run_calibration.py --every 5 --offset 0 --out none --state tests/s.pkl     # a subset, fit only
    python run_calibration.py --detections gui_5563 --n-water 1.33 --fit-k1 --hints photo --out none

Each run prints its progress and ends with a summary; --state saves the fit (case.save_state) for the tests.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import numpy as np

import case
import evaluate as ev


def calibrate_case(cfg: dict, detections: str = "all", rows=None, n_water: float = None, thickness_mm: float = None,
                   fit_k1: bool = False, hints: str = None, edge_margin: float = None, start_state: str = None,
                   verbose: bool = False, log=print, **kw):
    """Fit the rig to one configured detection set (rows: indices of its frames; default all).  hints: None or
    "photo" (config photo_hints).  edge_margin: drop the detections whose ball crosses the image border (None: keep
    every detection).  start_state: a saved fit (case.save_state) whose cameras and walls start the calibration
    (plate_wand.calibrate's `start`: stages 1-3 skipped).  Returns (sol, rig config, CSV data, the fitted frames' CSV
    numbers, seconds)."""
    if hints == "photo":
        h = {k: v for k, v in cfg["photo_hints"].items() if not k.startswith("_")}
    elif hints and hints.startswith("gui:"):
        # each camera's wall normal in its own frame, from one of the GUI's earlier calibrations
        gcams, grig = case.load_gui(cfg, hints[4:])
        h = {"normals": [(gc.R @ grig.windows[grig.cam_win[c]].n).tolist() for c, gc in enumerate(gcams)]}
    else:
        h = None
    rc = case.rig_config(cfg, n_water=n_water, thickness_mm=thickness_mm, hints=h)
    cd = case.read_detections(cfg, detections)
    if edge_margin is not None:
        n_cut = case.drop_truncated(cd, rc.sizes, edge_margin)
        log(f"edge filter: {n_cut} detections whose ball crosses the image border (margin {edge_margin} px) dropped")
    rows = np.arange(cd.uv.shape[0]) if rows is None else np.asarray(rows)
    data = case.wand_data(cd, rc, rows)
    log(f"calibrate: {detections} ({case.detections_path(cfg, detections)}), {len(rows)} of {cd.uv.shape[0]} frames; "
        f"walls {[w['thickness_mm'] for w in rc.windows]} mm, n {[w['n'] for w in rc.windows]}; L {rc.wand_length_mm} "
        f"mm; fit_k1 {fit_k1}; hints {hints if h else 'none'}"
        + (f" (normals {[[round(x, 3) for x in v] for v in h['normals']]})" if h else ""))
    t0 = time.time()
    if start_state:
        st = case.load_state(start_state)
        rig0 = st["rig"].copy()
        for w, wc in zip(rig0.windows, rc.windows):        # this run's plates and indices, the start's planes
            w.t, w.idx = tuple(float(v) for v in wc["thickness_mm"]), tuple(float(v) for v in wc["n"])
        log(f"start: the cameras and walls of {start_state} (RMS {st['rms']:.4f} px there)")
        sol = case.pw.calibrate(data, rig0, start=[c.copy() for c in st["cams"]], hints=rc.hints or None,
                                fit_k1=fit_k1, verbose=verbose, log=log, **kw)
    else:
        sol = case.pw.calibrate(data, rc.rig(), hints=rc.hints or None, fit_k1=fit_k1, verbose=verbose, log=log, **kw)
    secs = time.time() - t0
    fids = cd.frame_ids[rows]
    log(f"calibrate: done in {secs:.0f} s; RMS {sol.rms:.4f} px, sigma {sol.sigma_px:.4f} px, "
        f"{int(sol.obs.active.sum())} of {len(sol.obs)} detections active")
    return sol, rc, cd, fids, secs


def summary(sol, rc) -> str:
    """The fitted rig in a few lines (output frame, walls' frame description)."""
    d = case.describe(sol.cams, sol.rig)
    sig = case.pw.camera_sigmas(sol.cams, sol.rig, sol.layout, sol.cov) if sol.cov is not None else [None] * len(
        sol.cams)
    out = [f"walls {d['walls_angle_deg']:.3f} deg apart"]
    for c, (cd_, s) in enumerate(zip(d["cameras"], sig)):
        fs = f" +- {s['f_px']:.0f}" if s and s.get("f_px") is not None else ""
        out.append(f"  {rc.names[c]}: f {cd_['f_px']:.0f}{fs} px ({cd_['f_px'] * rc.cameras[c]['pixel_mm']:.1f} mm), "
                   f"{cd_['distance_to_wall_mm']:.1f} mm from its wall, {cd_['incidence_deg']:.2f} deg to its normal"
                   + (f", k1 {cd_['k1']:.3g}" if sol.layout.fit_k1 else ""))
    for p in d["pairs"]:
        out.append(f"  cams {p['cams']}: baseline {p['baseline_mm']:.1f} mm, views {p['views_angle_deg']:.2f} deg apart")
    return "\n".join(out)


def along_wand(sol) -> dict:
    """Mean reprojection residual of each camera's ends along and across the wand's image direction (A -> B), on the
    active detections: a detector that pulls the ball centres towards the rod gives end A + and end B - along it."""
    e = case.pw.evaluate(sol.cams, sol.rig, sol.frames, sol.L, sol.obs)
    XA, XB = sol.frames.ends(sol.L)
    out = []
    for c in range(len(sol.cams)):
        res = {}
        for end in (0, 1):
            sel = sol.obs.active & (sol.obs.cam == c) & (sol.obs.end == end) & e.ok
            fr = sol.obs.frame[sel]
            # image direction of the wand in this camera: from the observed A to B of the same frame
            uvA = {f: u for f, u in zip(sol.obs.frame[(sol.obs.cam == c) & (sol.obs.end == 0)],
                                        sol.obs.uv[(sol.obs.cam == c) & (sol.obs.end == 0)])}
            uvB = {f: u for f, u in zip(sol.obs.frame[(sol.obs.cam == c) & (sol.obs.end == 1)],
                                        sol.obs.uv[(sol.obs.cam == c) & (sol.obs.end == 1)])}
            dirs = np.array([uvB[f] - uvA[f] if (f in uvA and f in uvB) else (np.nan, np.nan) for f in fr])
            nrm = np.linalg.norm(dirs, axis=1)
            t = dirs / nrm[:, None]
            r = e.r[sel]
            along = np.sum(r * t, axis=1)
            across = r[:, 0] * -t[:, 1] + r[:, 1] * t[:, 0]
            res["AB"[end]] = {"along_px": float(np.nanmean(along)), "across_px": float(np.nanmean(across)),
                              "n": int(np.sum(np.isfinite(along)))}
        out.append(res)
    return {"per_camera": out, "note": "residual = model - detection; positive along = the model's end lies further "
                                       "along A->B than the detection"}


def write_outputs(sol, rc, cfg, out_dir: str, detections: str, fids, secs: float, settings: dict, log=print):
    """camFiles, calibration_result.json, calibration_report.txt (plate_wand_io), evaluation.json/.txt and a
    README.md into out_dir, a new folder (or one this script wrote before)."""
    mine = os.path.join(out_dir, "calibration_result.json")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        ok = False
        if os.path.isfile(mine):
            with open(mine, encoding="utf-8") as fh:
                ok = json.load(fh).get("case") == cfg["case"]
        if not ok:
            raise SystemExit(f"{out_dir} exists and was not written by this case; not overwritten")
    extra = {"case": cfg["case"], "written": datetime.datetime.now().isoformat(timespec="seconds"),
             "detections": case.detections_path(cfg, detections), "frames_used": int(len(fids)),
             "frame_numbers_used": [int(fids.min()), int(fids.max())], "seconds": secs, "settings": settings,
             "cameras_photo": [c["photo"] for c in rc.cameras],
             "image_folders": [c["folder"] for c in rc.cameras]}
    res = case.pio.write_result(sol, out_dir, names=rc.names, pixel_mm=[c["pixel_mm"] for c in rc.cameras],
                                extra=extra)
    # the same judgement as every other calibration of this case (evaluate.py) on all detections
    cd_all = case.read_detections(cfg, "all")
    e_all = ev.evaluate(sol.cams, sol.rig, cd_all.uv, rc.wand_length_mm, rc.sizes, frame_ids=cd_all.frame_ids)
    with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump({"all_detections": e_all}, fh, indent=1)
    with open(os.path.join(out_dir, "evaluation.txt"), "w", encoding="utf-8") as fh:
        fh.write("Free triangulation of every frame of the circle finder's table (evaluate.py)\n"
                 + ev.text("calib_plate_wand", e_all) + "\n")
    write_readme(out_dir, res, e_all)
    log(f"outputs written to {out_dir}")
    return res, e_all


def write_readme(out_dir: str, res: dict = None, e_all: dict = None) -> None:
    """The output folder's README.md, from its calibration_result.json and evaluation.json (read when not given)."""
    if res is None:
        with open(os.path.join(out_dir, "calibration_result.json"), encoding="utf-8") as fh:
            res = json.load(fh)
    if e_all is None:
        with open(os.path.join(out_dir, "evaluation.json"), encoding="utf-8") as fh:
            e_all = json.load(fh)["all_detections"]
    st = res.get("settings") or {}
    w = e_all["wand_error_mm"]

    def base(i, j):
        return float(np.linalg.norm(np.subtract(res["cameras"][i]["centre_mm"], res["cameras"][j]["centre_mm"])))
    start = st.get("start")
    how = ("from the wand detections alone: no camera, wall or focal-length input" if not start else
           f"from the wand detections, started from an earlier fit of the same data (`{os.path.basename(start)}`, "
           "the first half of the recording; a start from scratch on every frame ended in a wrong minimum: case "
           "report, section 4). Every camera, both walls and k1 are refitted on every frame")
    readme = [
        f"# Flat-window wand calibration, run 20260727 / T0 ({res.get('written', '')[:10]})", "",
        f"Calibrated by pyOpenLPT's flat-window wand calibration (`plate_wand.calibrate`), {how}. Code, plan, tests "
        "and report (`2026-10-03-report.md`): `D:\\Refraction\\pyOpenLPT\\cases\\plate_20260727_T0`.", "",
        "| file | contents |", "|---|---|",
        "| `camFile\\cam0.txt` ... `cam3.txt` | the cameras as PINPLATE camFiles for OpenLPT (same names as calib_3) |",
        "| `calibration_result.json` | every fitted value with its standard deviation, the noise, the warnings |",
        "| `calibration_report.txt` | the same as a short report, with a WARNINGS block when there is one |",
        "| `evaluation.txt`, `evaluation.json` | every frame of the circle finder's table triangulated with these "
        "cameras: reprojection errors and wand lengths (the same measure the case applies to calib, calib_2, calib_3) |",
        "| `comparison_with_calib_3.txt` | this calibration against your GUI calibrations (case test T7) |", "",
        "Cameras: cam0 = C1 (image folder cam1), cam1 = C2 (cam2), cam2 = C3 (cam3), cam3 = C4 (cam4). C1 and C2 look "
        "through one wall, C3 and C4 through the other.", "",
        f"Inputs: `{os.path.basename(res.get('detections', ''))}`, {res.get('frames_used')} frames; walls "
        f"{res['windows'][0]['thickness_mm'][0]} mm acrylic, indices air / acrylic / water = "
        f"{' / '.join(str(v) for v in res['windows'][0]['n'])}; wand {res['wand_length_mm']} mm; lens distortion k1 "
        f"{'fitted' if res.get('fit_k1') else 'held at 0'}.", "",
        f"Result: reprojection RMS {res['rms_px']:.3f} px (fit). Free triangulation of every frame of the table: "
        f"reprojection error median {e_all['euclidean_px']['median']:.3f} px; wand length error median "
        f"{w['abs']['median']:.4f} mm, p90 {w['abs']['p90']:.4f} mm; over the frames without a gross detection error "
        f"the wand measures {e_all['L'] + w['clean_frames']['mean']:.4f} mm on average, SD {w['clean_frames']['std']:.4f} mm "
        f"({w['frames_over_0.2mm']} frames have a gross detection error).", "",
        "Warnings: " + ("none." if not res["warnings"] else
                        "; ".join(x.get("code", "?") for x in res["warnings"])
                        + " (calibration_report.txt). The case report, section 7, explains the `pinned` warning: a "
                          "few frames where the wand came right up to the C1/C2 wall; it costs about 0.5 % in C1's and "
                          "C2's f and nothing measurable in the 3-D measurement."), "",
        f"Check: the spacing of C1 and C2 is {base(0, 1):.1f} mm in this calibration and of C3 and C4 "
        f"{base(2, 3):.1f} mm (calib_3: 170 and 173 mm); a tape measure between the lens axes decides.", ""]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(readme))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="all")
    ap.add_argument("--every", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--first", type=float, default=None, help="use the first fraction of the frames (0-1)")
    ap.add_argument("--last", type=float, default=None, help="use the last fraction of the frames (0-1)")
    ap.add_argument("--n-water", type=float, default=None)
    ap.add_argument("--thickness", type=float, default=None)
    ap.add_argument("--fit-k1", action="store_true")
    ap.add_argument("--hints", default=None,
                    help="window-normal hints: 'photo' (config photo_hints) or 'gui:<name>' (from a GUI calibration)")
    ap.add_argument("--edge-margin", type=float, default=None,
                    help="drop detections whose ball crosses the image border (this margin in px; default: keep all)")
    ap.add_argument("--start", default=None, help="start from this saved fit's cameras and walls (a state pickle)")
    ap.add_argument("--out", default=None, help="output folder (default: config out_dir; 'none': no outputs)")
    ap.add_argument("--state", default=None, help="save the fit here (pickle)")
    ap.add_argument("--log", default=None, help="also write the console to this file")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--readme-only", action="store_true",
                    help="only rewrite the output folder's README.md from its result files (no fit)")
    args = ap.parse_args()
    if args.readme_only:
        write_readme(case.load_config()["out_dir"] if args.out is None else args.out)
        return
    cfg = case.load_config()
    log = case.Tee(args.log) if args.log else print
    n = case.read_detections(cfg, args.detections).uv.shape[0]
    rows = case.interleaved(n, args.offset, args.every)
    if args.first is not None:
        rows = rows[rows < int(round(args.first * n))]
    if args.last is not None:
        rows = rows[rows >= n - int(round(args.last * n))]
    settings = {k: getattr(args, k) for k in ("detections", "every", "offset", "first", "last", "n_water", "thickness",
                                              "fit_k1", "hints", "edge_margin", "start")}
    sol, rc, cd, fids, secs = calibrate_case(cfg, args.detections, rows, args.n_water, args.thickness, args.fit_k1,
                                             args.hints, edge_margin=args.edge_margin, start_state=args.start,
                                             verbose=args.verbose, log=log)
    log(summary(sol, rc))
    aw = along_wand(sol)
    log("residual along the wand (A, B), px: " + "; ".join(
        f"cam{c}: {r['A']['along_px']:+.3f}, {r['B']['along_px']:+.3f}" for c, r in enumerate(aw["per_camera"])))
    warn = case.pio.assess(sol, rc.names)
    log("warnings: " + ("none" if not warn else "; ".join(w.get("code", "?") + ": " + w.get("text", "")[:200]
                                                         for w in warn)))
    if args.state:
        case.save_state(args.state, sol, extra={"frame_numbers": fids, "detections": args.detections,
                                                "settings": settings, "seconds": secs, "along_wand": aw})
        log(f"state saved to {args.state}")
    out = cfg["out_dir"] if args.out is None else (None if args.out == "none" else args.out)
    if out:
        write_outputs(sol, rc, cfg, out, args.detections, fids, secs, settings, log=log)
    if args.log:
        log.close()


if __name__ == "__main__":
    main()
