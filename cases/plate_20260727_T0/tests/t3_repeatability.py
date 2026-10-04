"""T3: repeatability.  The same calibration on 5 interleaved subsets of the frames (every 5th, offsets 0-4) and on the
first and second halves of the recording (tests/run_batch.py t3 / t3k; states in tests/states/).

For every fit: RMS, noise, detections kept, warnings, and the rig's frame-independent description (case.describe): f,
each camera's distance and angle to its wall, the baselines and angles between cameras, the angle between the
walls.  Then the spread over the 5 subsets against the fits' own sigmas (f, distance to the wall), and the two halves'
difference against their combined sigma (drift during the recording).  Pass: every subset within about 3 sigma of the
others, no subset in another minimum.

    python tests/t3_repeatability.py t3        # k1 held at 0
    python tests/t3_repeatability.py t3k       # k1 fitted
Writes t3_repeatability_<group>_out.txt and .json beside this script.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

SUBS = ["s0", "s1", "s2", "s3", "s4"]
HALVES = ["h1", "h2", "h2_from_h1"]      # h2_from_h1: the second half started from the first half's answer (--start)


def solution_of(st, obs, frames):
    """A plate_wand Solution of a saved fit (for the warnings and the sigmas)."""
    lay = case.pw.Layout(obs.n_cams, len(st["rig"].windows), fit_k1=bool(st["layout"]["fit_k1"]),
                         anchors=case.pw.anchors_for(st["cams"], st["rig"]))
    if list(lay.names) != list(st["layout"]["names"]):
        raise ValueError("layout differs from the saved fit's")
    return case.pw.Solution(st["cams"], st["rig"], frames, st["L"], obs, lay, rms=st["rms"], cov=st["cov"],
                            sigma_px=st["sigma_px"], info=st["info"], weights=st.get("weights"))


def one(group: str, sub: str, cfg) -> dict:
    st = case.load_state(os.path.join(HERE, "states", f"{group}_{sub}.pkl"))
    rc, cd, rows, data, obs, frames = case.rebuild(st, cfg)
    sol = solution_of(st, obs, frames)
    d = case.describe(sol.cams, sol.rig)
    sig = case.pw.camera_sigmas(sol.cams, sol.rig, sol.layout, sol.cov)
    wsig = case.pw.window_sigmas(sol.cams, sol.rig, sol.layout, sol.cov)
    warn = case.pio.assess(sol, rc.names)
    XA, XB = sol.frames.ends(sol.L)
    act = np.zeros(obs.n_frames, bool)
    act[obs.frame[obs.active]] = True
    P = np.vstack([XA[act], XB[act]])
    return {"sub": sub, "frames": int(len(rows)), "rms_px": st["rms"], "sigma_px": st["sigma_px"],
            "active": int(obs.active.sum()), "detections": int(len(obs)), "seconds": st["extra"].get("seconds"),
            "describe": d, "sigma_f_px": [s["f_px"] for s in sig], "sigma_distance_mm": [s.get("distance_mm") for s in sig],
            "sigma_k1": [s.get("k1") for s in sig], "sigma_wall_normal_deg": [w.get("normal_deg") for w in wsig],
            "sigma_wall_offset_mm": [w.get("offset_mm") for w in wsig],
            "min_depth_mm": [float(w.depth(P).min()) for w in sol.rig.windows],
            "warnings": [w["code"] for w in warn], "chosen": (st["info"].get("hints") or {}).get("chosen")}


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "t3"
    cfg = case.load_config()
    names = [c["name"] for c in cfg["rig"]["cameras"]]
    res = {s: one(group, s, cfg) for s in SUBS + HALVES
           if os.path.isfile(os.path.join(HERE, "states", f"{group}_{s}.pkl"))}
    log = case.Tee(os.path.join(HERE, f"t3_repeatability_{group}_out.txt"))
    log(f"T3 ({group}): repeatability over frame subsets{' (k1 fitted)' if group.endswith('k') else ''}")
    log("  fit   frames  RMS px  sigma px  kept     s    walls deg   f (px) cam0..cam3            "
        "distance to wall (mm) cam0..cam3     smallest depth below wall 1, 2 (mm)  warnings")
    for s, r in res.items():
        d = r["describe"]
        log(f"  {s:4s} {r['frames']:6d}  {r['rms_px']:.4f}  {r['sigma_px']:.4f}  {r['active']:6d} {r['seconds'] or 0:5.0f}"
            f"  {d['walls_angle_deg']:8.3f}   " + " ".join(f"{c['f_px']:6.0f}" for c in d["cameras"]) + "    "
            + " ".join(f"{c['distance_to_wall_mm']:7.1f}" for c in d["cameras"]) + "    "
            + ", ".join(f"{x:.3g}" for x in r["min_depth_mm"]) + "   " + (", ".join(r["warnings"]) or "-"))
    subs = [res[s] for s in SUBS if s in res]
    summary = {}
    if len(subs) >= 2:
        log("  spread over the interleaved subsets (SD) against the fits' own sigma (median):")
        rows = []
        for c in range(len(names)):
            for key, skey, unit in (("f_px", "sigma_f_px", "px"), ("distance_to_wall_mm", "sigma_distance_mm", "mm")):
                vals = np.array([r["describe"]["cameras"][c][key] for r in subs])
                sig = np.median([r[skey][c] for r in subs])
                rows.append({"camera": c, "quantity": key, "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                             "sigma": float(sig), "ratio": float(vals.std(ddof=1) / sig)})
                log(f"    {names[c]} {key:20s}: mean {vals.mean():9.2f} {unit}, SD {vals.std(ddof=1):7.2f}, sigma "
                    f"{sig:7.2f}, SD/sigma {vals.std(ddof=1) / sig:5.2f}; range {vals.min():.2f}-{vals.max():.2f}")
            k1s = [r["describe"]["cameras"][c]["k1"] for r in subs]
            if any(k1s):
                sk = np.median([r["sigma_k1"][c] for r in subs])
                log(f"    {names[c]} k1                  : mean {np.mean(k1s):.3e}, SD {np.std(k1s, ddof=1):.1e}, "
                    f"sigma {sk:.1e}")
        for key in ("walls_angle_deg",):
            vals = np.array([r["describe"][key] for r in subs])
            log(f"    walls angle: mean {vals.mean():.4f} deg, SD {vals.std(ddof=1):.4f}")
        for p in range(len(subs[0]["describe"]["pairs"])):
            pr = subs[0]["describe"]["pairs"][p]["cams"]
            b = np.array([r["describe"]["pairs"][p]["baseline_mm"] for r in subs])
            a = np.array([r["describe"]["pairs"][p]["views_angle_deg"] for r in subs])
            log(f"    cams {pr}: baseline mean {b.mean():8.2f} mm, SD {b.std(ddof=1):6.2f}; views angle mean "
                f"{a.mean():7.3f} deg, SD {a.std(ddof=1):.3f}")
        summary["subsets"] = rows
    if "h1" in res and ("h2" in res or "h2_from_h1" in res):
        a = res["h1"]
        # a second half that ended in another minimum than the first (its walls far from them) is compared through
        # its refit from the first half's answer
        b_key = "h2_from_h1" if "h2_from_h1" in res else "h2"
        b = res[b_key]
        log(f"  first half against second half ({b_key}; difference / combined sigma):")
        drift = []
        for c in range(len(names)):
            for key, skey in (("f_px", "sigma_f_px"), ("distance_to_wall_mm", "sigma_distance_mm")):
                va, vb = a["describe"]["cameras"][c][key], b["describe"]["cameras"][c][key]
                sg = float(np.hypot(a[skey][c], b[skey][c]))
                drift.append({"camera": c, "quantity": key, "h1": va, "h2": vb, "z": (vb - va) / sg})
                log(f"    {names[c]} {key:20s}: {va:9.2f} -> {vb:9.2f}  ({(vb - va) / sg:+.2f} sigma)")
        summary["halves"] = drift
    with open(os.path.join(HERE, f"t3_repeatability_{group}.json"), "w", encoding="utf-8") as fh:
        json.dump({"fits": res, "summary": summary}, fh, indent=1, default=float)
    log.close()


if __name__ == "__main__":
    main()
