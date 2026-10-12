"""Every fit of the case in one table (tests/fits_summary.json), for the report page: from the saved states
(tests/states/*.pkl, not tracked by git), each fit's settings, RMS, warnings, the rig's frame-independent description
(case.describe) and whether it is the right answer (both walls' cameras farther than 0.5 m from their walls and the
walls within 1 degree of the right answer's 90.2); and, for the plan view, the delivered fit's and calib_3's cameras
placed against the two walls, with the volume the wand covered (and, once it exists, the calibration with the
channel, W4's w4_ref, and the channel's size).  The walls plan's W1 and W2 states are held bundle adjustments, not
calibrations, and are left out.

    python tests/fits_summary.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

SKIP = {"probe20"}
SKIP_PREFIX = ("w1_", "w2_")     # the walls plan's held bundle adjustments (W1, W2): tests, not calibrations
LABELS = {"t2_all": "T2, delivered (every frame, from the first half's fit)",
          "t2_scratch": "T2 from scratch (every frame)",
          "t3k_h2_from_h1": "second half, from the first half's fit",
          "w4_ref": "W4, with the channel (every frame; principal point fitted; channel and aim held)",
          "w4c_ref": "W4c, delivered with the channel (every frame; principal point at the centre; channel and aim held)"}


def plan(cams, rig, P=None):
    """Each camera against the two walls: x = past wall 2's camera-side surface along its normal, y = past wall 1's
    (mm; negative on the camera side), its viewing direction in those axes; the walls' thicknesses; with P (wand ends)
    the 1-99 % extent of the wand's volume in the same axes."""
    w1, w2 = rig.windows[0], rig.windows[1]
    out = {"cameras": [], "t": [w1.T, w2.T], "angle_deg": case.angle_deg(w1.n, w2.n)}
    for c, cam in enumerate(cams):
        z = cam.R[2]
        out["cameras"].append({"x": float(w2.n @ cam.C - w2.d), "y": float(w1.n @ cam.C - w1.d),
                               "dx": float(w2.n @ z), "dy": float(w1.n @ z), "wall": rig.cam_win[c] + 1})
    if P is not None:
        X, Y = P @ w2.n - w2.d, P @ w1.n - w1.d
        out["wand"] = {"x": [float(np.percentile(X, 1)), float(np.percentile(X, 99))],
                       "y": [float(np.percentile(Y, 1)), float(np.percentile(Y, 99))]}
    return out


def wand_ends(st):
    fr = case.pw.Frames(np.asarray(st["frames_m"]), np.asarray(st["frames_w"]))
    XA, XB = fr.ends(st["L"])
    act = np.zeros(len(XA), bool)
    act[np.asarray(st["obs_frame"])[np.asarray(st["active"], bool)]] = True
    return np.vstack([XA[act], XB[act]])


def main():
    cfg = case.load_config()
    names = [c["name"] for c in cfg["rig"]["cameras"]]
    fits = []
    for fn in sorted(os.listdir(os.path.join(HERE, "states"))):
        name = fn[:-4]
        if not fn.endswith(".pkl") or name in SKIP or name.startswith(SKIP_PREFIX):
            continue
        st = case.load_state(os.path.join(HERE, "states", fn))
        s = st["extra"].get("settings") or {}
        d = case.describe(st["cams"], st["rig"])
        P = wand_ends(st)
        right = (min(c["distance_to_wall_mm"] for c in d["cameras"]) > 500.0 and abs(d["walls_angle_deg"] - 90.2) < 1.0)
        info = st["info"]
        hints = (info.get("hints") or {})
        fits.append({
            "name": name, "label": LABELS.get(name, name), "frames": int(len(st["frame_ids"])),
            "detections": s.get("detections") or "all", "every": s.get("every"), "offset": s.get("offset"),
            "first": s.get("first"), "last": s.get("last"), "fit_k1": bool(st["layout"]["fit_k1"]),
            "hints": s.get("hints"), "edge_margin": s.get("edge_margin"),
            "start": os.path.basename(s["start"])[:-4] if s.get("start") else None,
            "n_water": s.get("n_water"), "thickness": s.get("thickness"),
            "fit_lens": list(st["layout"].get("fit_lens") or []), "constraints": s.get("constraints"),
            "pp_off_px": [[c.cx - (c.width - 1) / 2.0, c.cy - (c.height - 1) / 2.0] for c in st["cams"]],
            "depth_mm": [[float(w.depth(P).min()), float(w.depth(P).max())] for w in st["rig"].windows],
            "rms_px": st["rms"], "sigma_px": st["sigma_px"], "seconds": st["extra"].get("seconds"),
            "walls_angle_deg": d["walls_angle_deg"], "f_px": [c["f_px"] for c in d["cameras"]],
            "k1": [c["k1"] for c in d["cameras"]], "distance_mm": [c["distance_to_wall_mm"] for c in d["cameras"]],
            "incidence_deg": [c["incidence_deg"] for c in d["cameras"]],
            "baseline_mm": {f"{p['cams'][0]}-{p['cams'][1]}": p["baseline_mm"] for p in d["pairs"]},
            "min_depth_mm": [float(w.depth(P).min()) for w in st["rig"].windows],
            "warnings": warnings_of(st, cfg, names),
            "hint_path": hints.get("chosen"), "right": bool(right)})
    st2 = case.load_state(os.path.join(HERE, "states", "t2_all.pkl"))
    g_cams, g_rig = case.load_gui(cfg, "calib_3")
    out = {"fits": fits, "plan": {"T2": plan(st2["cams"], st2["rig"], wand_ends(st2)),
                                  "calib_3": plan(g_cams, g_rig)}}
    for key, state in (("ref", "w4c_ref"), ("ref_pp", "w4_ref")):   # the walls plan's deliveries (W4c chosen, W4)
        p_ref = os.path.join(HERE, "states", f"{state}.pkl")
        if os.path.exists(p_ref):
            st_r = case.load_state(p_ref)
            out["plan"][key] = plan(st_r["cams"], st_r["rig"], wand_ends(st_r))
            out["plan"]["channel"] = {k: v for k, v in cfg["constraints"]["channel"].items() if not k.startswith("_")}
    with open(os.path.join(HERE, "fits_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"{len(fits)} fits; right {sum(f['right'] for f in fits)}, wrong {sum(not f['right'] for f in fits)}")
    for f in fits:
        print(f"  {f['name']:18s} {f['frames']:6d}  RMS {f['rms_px']:.4f}  walls {f['walls_angle_deg']:7.3f}  "
              f"to walls {', '.join(f'{x:.0f}' for x in f['distance_mm'])}  {'right' if f['right'] else 'WRONG'}  "
              f"{', '.join(f['warnings']) or '-'}")


def warnings_of(st, cfg, names):
    """The report's warnings for a saved fit, on its own detections (plate_wand_io.assess)."""
    from t3_repeatability import solution_of
    _rc, _cd, _rows, _data, obs, frames = case.rebuild(st, cfg)
    return [w["code"] for w in case.pio.assess(solution_of(st, obs, frames), names)]


if __name__ == "__main__":
    main()
