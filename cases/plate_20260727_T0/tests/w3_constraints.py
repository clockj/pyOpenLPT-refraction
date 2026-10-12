"""W3 (2026-10-03-walls-plan.md, section 4): the constraints (plate_wand's L2), run through the case's calibration
(run_calibration.py: plate_wand.calibrate, stage 4 with outlier rejection and noise weights) on every frame:
  w3_A          the channel (config constraints.channel: 100 x 150 mm inside, +3 mm, ball radii 1.5 / 2 mm)
  w3_AB         the channel and the aim: C1 and C2 look at the middle of the long side (config aim_centre), held
  w3_B          the aim alone
  w3_A_scratch  the channel from scratch (no start): T2's first run fell into a wrong minimum from scratch
Each with the lens model given by --fit-lens (default: f and k1, T2's), started from that model's free fit of every
frame (T2, or W2's unpinned w2_<terms>_free; STARTS).  The summary gives per fit the RMS, the
cameras (f, k1, lens terms, distances and angles to their walls), the walls' angle, the wand ends' depths below both
walls and the band each wall could still move in (plate_wand.constraint_report), the aim, and evaluate.py on every
frame, beside T2's.

    python tests/w3_constraints.py [--fit-lens TERMS] [--only JOBS]   # the fits in parallel, then the summary
    python tests/w3_constraints.py --summary [--fit-lens TERMS]
Writes w3_constraints_out.txt and w3_constraints.json beside this script (with --fit-lens: w3_constraints_<terms>_*),
the fits to tests/states/<job>.pkl and their consoles to tests/logs/<job>.txt.
"""
from __future__ import annotations

import json
import os
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
import run_batch  # noqa: E402
import w1_channel_hold as w1  # noqa: E402

# each lens model starts from its own free fit of every frame: T2 (f, k1), or W2's unpinned free fit with the terms
STARTS = {"": "t2_all", "pp": "w2_pp_free", "tangential": "w2_tan_free", "pp,k2,tangential": "w2_all_free"}
JOBS = {"w3_A": ["--channel"], "w3_AB": ["--channel", "--aim", "centre"], "w3_B": ["--aim", "centre"],
        "w3_A_scratch": ["--channel"]}


def job_names(lens: str):
    tag = "" if not lens else "_" + lens.replace(",", "+")
    start = ["--start", os.path.join("tests", "states", f"{STARTS[lens]}.pkl")]
    return {f"{k}{tag}": ["--fit-k1"] + ([] if k.endswith("scratch") else start) + v
            + (["--fit-lens", lens] if lens else []) for k, v in JOBS.items()}


def describe_fit(st, cd, rc) -> dict:
    cams, rig = st["cams"], st["rig"]
    frames = case.pw.Frames(np.asarray(st["frames_m"]), np.asarray(st["frames_w"]))
    obs = case.pw.Obs(np.asarray(st["obs_frame"]), np.asarray(st["obs_cam"]), np.asarray(st["obs_end"]), None,
                      int(len(frames.m)), len(cams), np.asarray(st["frame_ids"]), np.asarray(st["active"], bool))
    d = case.describe(cams, rig)
    dep = w1.active_depths(cams, rig, frames, st["L"], obs)
    evl = ev.evaluate(cams, rig, cd.uv, rc.wand_length_mm, rc.sizes)
    wc = evl["wand_error_mm"]["clean_frames"]
    return {"rms_px": st["rms"], "sigma_px": st["sigma_px"], "active": int(np.sum(st["active"])),
            "fit_lens": st["layout"].get("fit_lens", []),
            "f_px": [c["f_px"] for c in d["cameras"]], "k1": [c["k1"] for c in d["cameras"]],
            "pp_off_px": [[c.cx - (c.width - 1) / 2.0, c.cy - (c.height - 1) / 2.0] for c in cams],
            "dist": [c.dist.tolist() for c in cams],
            "distance_to_wall_mm": [c["distance_to_wall_mm"] for c in d["cameras"]],
            "incidence_deg": [c["incidence_deg"] for c in d["cameras"]], "walls_angle_deg": d["walls_angle_deg"],
            "baselines_mm": {f"C{p['cams'][0] + 1}-C{p['cams'][1] + 1}": p["baseline_mm"] for p in d["pairs"]},
            "depth_wall1_mm": [float(dep[0].min()), float(dep[0].max())],
            "depth_wall2_mm": [float(dep[1].min()), float(dep[1].max())],
            "aim_C1C2_from_wall2_mm": case.pw.aim_depth(cams, rig, [0, 1], 1),
            "aim_C3C4_from_wall1_mm": case.pw.aim_depth(cams, rig, [2, 3], 0),
            "constraints": st["info"].get("constraints"), "warnings": None,
            "seconds": (st.get("extra") or {}).get("seconds"),
            "every_frame": {"median_px": evl["euclidean_px"]["median"], "rms_inliers_px": evl["rms_inliers_px"],
                            "wand_abs_median_mm": evl["wand_error_mm"]["abs"]["median"],
                            "wand_clean_mean_mm": evl["L"] + wc["mean"], "wand_clean_sd_mm": wc["std"],
                            "outlier_detections": evl["outlier_detections"]}}


def summary(lens: str):
    cfg = case.load_config()
    rc = case.rig_config(cfg)
    cd = case.read_detections(cfg, "all")
    tag = "" if not lens else "_" + lens.replace(",", "+")
    names = list(dict.fromkeys(["t2_all", STARTS[lens]] + list(job_names(lens))))
    res = {}
    for nm in names:
        p = os.path.join(HERE, "states", f"{nm}.pkl")
        if os.path.exists(p):
            res[nm] = describe_fit(case.load_state(p), cd, rc)
    log = case.Tee(os.path.join(HERE, f"w3_constraints{tag}_out.txt"))
    log(f"W3: the constraints through the case's calibration, every frame; lens terms: {lens or 'none (f, k1)'}; "
        "t2_all = the delivered fit, no constraint")
    log("  fit              RMS px  sigma   active   f (px)                    k1                       to wall (mm)"
        "             walls deg   wand below wall 1 / wall 2 (mm)       aim C1C2 / C3C4 (mm)")
    for nm, o in res.items():
        log(f"  {nm:16s} {o['rms_px']:.4f}  {o['sigma_px']:.4f}  {o['active']:6d}   "
            + " ".join(f"{v:5.0f}" for v in o["f_px"]) + "   " + " ".join(f"{v:.3f}" for v in o["k1"]) + "   "
            + " ".join(f"{v:5.0f}" for v in o["distance_to_wall_mm"]) + f"   {o['walls_angle_deg']:.3f}     "
            + f"{o['depth_wall1_mm'][0]:5.1f}..{o['depth_wall1_mm'][1]:5.1f} / {o['depth_wall2_mm'][0]:5.1f}.."
              f"{o['depth_wall2_mm'][1]:5.1f}      {o['aim_C1C2_from_wall2_mm']:6.1f} / {o['aim_C3C4_from_wall1_mm']:5.1f}")
    log("  fit              every frame: median px, inliers RMS, |dL| median (mm), clean mean L, SD, outliers;  "
        "baselines C1-C2, C3-C4 (mm)")
    for nm, o in res.items():
        f = o["every_frame"]
        b = o["baselines_mm"]
        log(f"  {nm:16s} {f['median_px']:.4f}, {f['rms_inliers_px']:.4f}, {f['wand_abs_median_mm']:.4f}, "
            f"{f['wand_clean_mean_mm']:.4f}, {f['wand_clean_sd_mm']:.4f}, {f['outlier_detections']};  "
            f"{b['C1-C2']:.2f}, {b['C3-C4']:.2f}")
    log("  constraints at the answers (plate_wand.constraint_report): per wall the band of shifts that keeps the wand "
        "inside (mm, + away from its cameras); the channel's ends on a limit; the aims")
    for nm, o in res.items():
        c = o["constraints"]
        if not c:
            continue
        parts = []
        ch = c.get("channel")
        if ch:
            for k, w in enumerate(ch["windows"]):
                lo, hi = w["band_mm"]
                parts.append(f"wall {k + 1} band {'-inf' if lo is None else f'{lo:+.1f}'}..{hi:+.1f}")
            parts.append(f"{ch['active_rows']} ends on a limit, largest breach {ch['max_breach_mm']:.3f} mm")
        for p in c.get("priors") or []:
            parts.append(f"{p['kind']} {p['value']:.2f} (given {p['mean']:.2f})")
        log(f"  {nm:16s} " + "; ".join(parts))
    with open(os.path.join(HERE, f"w3_constraints{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    log.close()


def main():
    args = sys.argv[1:]
    lens = args[args.index("--fit-lens") + 1] if "--fit-lens" in args else ""
    if "--summary" in args:
        summary(lens)
        return
    jobs = job_names(lens)
    if "--only" in args:
        keep = set(args[args.index("--only") + 1].split(","))
        jobs = {k: v for k, v in jobs.items() if k in keep}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for job, a, rcode, secs in ex.map(lambda kv: run_batch.run(*kv), jobs.items()):
            print(f"  {job}: exit {rcode}, {secs:.0f} s  ({a})")
    print(f"W3: {len(jobs)} fits in {time.time() - t0:.0f} s")
    summary(lens)


if __name__ == "__main__":
    main()
