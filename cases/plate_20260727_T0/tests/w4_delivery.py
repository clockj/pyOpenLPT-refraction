"""W4 (2026-10-03-walls-plan.md, section 4): the calibration with the channel, delivered.

The settings W2 and W3 chose: k1 and the principal point fitted (plate_wand fit_lens "pp"), the channel held (config
constraints.channel) and C1 and C2 aiming at the middle of the long side (config aim_centre).  With --centre, the
user's choice of 4 Oct instead: the principal point held at the image centre (k1 only), the rest the same (prefix w4c,
output folder config out_dir_ref_centre).
  w4_ref  every frame, started from the delivered fit (T2); outputs to config out_dir_ref (calib_plate_wand_ref)
  w4_h1   the first half of the recording, started from its plain fit (t3k_h1); judged on the second half
  w4_h2   the second half, started from its plain fit (t3k_h2_from_h1); judged on the first half
The halves are judged with evaluate.py on the frames they did not use, beside the plain fits of the same halves (k1,
no constraint) on the same frames.  The camFiles written to calib_plate_wand_ref are read back and judged on every
frame (the round trip: the same numbers as the fit's own evaluation).

    python tests/w4_delivery.py [--centre]            # the three fits in parallel, the round trip, the summary
    python tests/w4_delivery.py [--centre] --summary
Writes w4_delivery_out.txt, w4_delivery.json, w4_camfile_roundtrip_out.txt and w4_calibration_console.txt beside
this script (w4c_* with --centre); the fits to tests/states/w4_*.pkl (w4c_*).
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

VARIANTS = {
    "pp": {"prefix": "w4", "out": "ref", "out_key": "out_dir_ref", "what": "k1 and the principal point fitted",
           "settings": ["--fit-k1", "--fit-lens", "pp", "--channel", "--aim", "centre"]},
    "centre": {"prefix": "w4c", "out": "ref_centre", "out_key": "out_dir_ref_centre",
               "what": "k1 fitted, the principal point held at the image centre",
               "settings": ["--fit-k1", "--channel", "--aim", "centre"]},
}
VAR = VARIANTS["centre" if "--centre" in sys.argv else "pp"]
PFX, SETTINGS = VAR["prefix"], VAR["settings"]
JOBS = {
    f"{PFX}_ref": SETTINGS + ["--start", os.path.join("tests", "states", "t2_all.pkl"), "--out", VAR["out"],
                              "--log", os.path.join("tests", f"{PFX}_calibration_console.txt")],
    f"{PFX}_h1": SETTINGS + ["--first", "0.5", "--start", os.path.join("tests", "states", "t3k_h1.pkl"),
                             "--out", "none", "--log", os.path.join("tests", "logs", f"{PFX}_h1.txt")],
    f"{PFX}_h2": SETTINGS + ["--last", "0.5", "--start", os.path.join("tests", "states", "t3k_h2_from_h1.pkl"),
                             "--out", "none", "--log", os.path.join("tests", "logs", f"{PFX}_h2.txt")],
}
PLAIN = {f"{PFX}_h1": "t3k_h1", f"{PFX}_h2": "t3k_h2_from_h1"}


def run(job, args):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(CASE_DIR, "run_calibration.py"), *args,
                        "--state", os.path.join(HERE, "states", f"{job}.pkl")], cwd=CASE_DIR, env=env,
                       capture_output=True, text=True)
    with open(os.path.join(HERE, "logs", f"{job}_stdout.txt"), "w", encoding="utf-8") as fh:
        fh.write(p.stdout + "\n" + p.stderr)
    return job, p.returncode, time.time() - t0


def judged(st, uv, rc) -> dict:
    e = ev.evaluate(st["cams"], st["rig"], uv, rc.wand_length_mm, rc.sizes)
    wc = e["wand_error_mm"]["clean_frames"]
    return {"median_px": e["euclidean_px"]["median"], "rms_inliers_px": e["rms_inliers_px"],
            "wand_abs_median_mm": e["wand_error_mm"]["abs"]["median"], "wand_abs_p90_mm": e["wand_error_mm"]["abs"]["p90"],
            "wand_clean_mean_mm": e["L"] + wc["mean"], "wand_clean_sd_mm": wc["std"],
            "outlier_detections": e["outlier_detections"], "frames": e["frames"]}


def summary():
    cfg = case.load_config()
    rc = case.rig_config(cfg)
    cd = case.read_detections(cfg, "all")
    n = cd.uv.shape[0]
    h = int(round(0.5 * n))
    first, second = np.arange(h), np.arange(n - h, n)
    out = {"settings": SETTINGS}
    REF = f"{PFX}_ref"
    log = case.Tee(os.path.join(HERE, f"{PFX}_delivery_out.txt"))
    log(f"W4{'c' if PFX == 'w4c' else ''}: {VAR['what']}, the channel held, C1 and C2 aiming at the middle of the long "
        "side")
    t2 = case.load_state(os.path.join(HERE, "states", "t2_all.pkl"))
    ref = case.load_state(os.path.join(HERE, "states", f"{REF}.pkl"))
    out["every_frame"] = {"t2_all": judged(t2, cd.uv, rc), REF: judged(ref, cd.uv, rc)}
    others = [nm for nm in ("w4_ref",) if nm != REF and os.path.exists(os.path.join(HERE, "states", f"{nm}.pkl"))]
    for nm in others:                                  # the other variant, for comparison
        out["every_frame"][nm] = judged(case.load_state(os.path.join(HERE, "states", f"{nm}.pkl")), cd.uv, rc)
    out[REF] = {"rms_px": ref["rms"], "sigma_px": ref["sigma_px"], "active": int(np.sum(ref["active"])),
                "constraints": ref["info"].get("constraints"), "seconds": ref["extra"].get("seconds")}
    log(f"  every frame (its own): T2 RMS {t2['rms']:.4f} px, {REF} RMS {ref['rms']:.4f} px (sigma "
        f"{ref['sigma_px']:.4f}), {out[REF]['active']} detections active, {out[REF]['seconds']:.0f} s")
    for nm in ["t2_all", REF] + others:
        j = out["every_frame"][nm]
        log(f"    {nm:15s} evaluate.py: median {j['median_px']:.4f} px, inliers RMS {j['rms_inliers_px']:.4f}; wand "
            f"|dL| median {j['wand_abs_median_mm']:.4f}, p90 {j['wand_abs_p90_mm']:.4f} mm, clean mean "
            f"{j['wand_clean_mean_mm']:.4f}, SD {j['wand_clean_sd_mm']:.4f} mm")
    out["held_out"] = {}
    log("  held out: each half's fit judged on the other half (evaluate.py), beside the plain fit (k1, no constraint) "
        "of the same half")
    for job, frames, other in ((f"{PFX}_h1", first, "second"), (f"{PFX}_h2", second, "first")):
        rows = second if other == "second" else first
        res = {}
        for nm in (PLAIN[job], job):
            p = os.path.join(HERE, "states", f"{nm}.pkl")
            if os.path.exists(p):
                st = case.load_state(p)
                res[nm] = {**judged(st, cd.uv[rows], rc), "rms_px": st["rms"]}
        out["held_out"][job] = res
        for nm, j in res.items():
            log(f"    {nm:15s} (fit RMS {j['rms_px']:.4f}) on the {other} half: median {j['median_px']:.4f} px, inliers "
                f"RMS {j['rms_inliers_px']:.4f}; wand |dL| median {j['wand_abs_median_mm']:.4f}, p90 "
                f"{j['wand_abs_p90_mm']:.4f} mm, clean mean {j['wand_clean_mean_mm']:.4f}, SD {j['wand_clean_sd_mm']:.4f}")
    # the round trip: the camFiles written, read back
    folder = os.path.join(cfg[VAR["out_key"]], "camFile")
    cams, wins = [], {}
    for c, cc in enumerate(cfg["rig"]["cameras"]):
        cam, win = case.pio.load_camfile(os.path.join(folder, f"{cc['name']}.txt"))
        cams.append(cam)
        wins.setdefault(cfg["rig"]["cam_to_window"][c], win)
    rig = case.pw.Rig([wins[w] for w in sorted(wins)], list(cfg["rig"]["cam_to_window"]))
    e_rt = ev.evaluate(cams, rig, cd.uv, rc.wand_length_mm, rc.sizes)
    with open(os.path.join(HERE, f"{PFX}_camfile_roundtrip_out.txt"), "w", encoding="utf-8") as fh:
        fh.write(ev.text(folder, e_rt) + "\n")
    rt = judged({"cams": cams, "rig": rig}, cd.uv, rc)
    d_pp = max(max(abs(a.cx - b.cx), abs(a.cy - b.cy)) for a, b in zip(cams, ref["cams"]))
    out["roundtrip"] = {**rt, "max_principal_point_change_px": d_pp}
    same = all(abs(rt[k] - out["every_frame"][REF][k]) < 1e-6 for k in ("median_px", "wand_abs_median_mm"))
    log(f"  camFiles read back ({folder}): median {rt['median_px']:.4f} px, wand |dL| median "
        f"{rt['wand_abs_median_mm']:.4f} mm; the principal points within {d_pp:.2e} px of the fit's: "
        + ("the same as the fit" if same else "DIFFERENT from the fit"))
    with open(os.path.join(HERE, f"{PFX}_delivery.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    log.close()


def main():
    if "--summary" not in sys.argv:
        os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
        with ThreadPoolExecutor(max_workers=len(JOBS)) as ex:
            for job, rcode, secs in ex.map(lambda kv: run(*kv), JOBS.items()):
                print(f"  {job}: exit {rcode}, {secs:.0f} s")
    summary()


if __name__ == "__main__":
    main()
