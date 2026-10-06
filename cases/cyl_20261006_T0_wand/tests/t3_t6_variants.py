"""T3-T6: the calibration's variants (plan: 2026-10-06-plan.md, section 2), each compared with S2's result.

    python tests/t3_t6_variants.py [NAME ...]      (default: all; after calibrate.py)

Each variant is a full calibration (cylinder_wand_gui.run, the settings of config.json unless said), run in its own
process, with the GUI's files written to <out_dir>/checks/<name>/:
  T3  h1, h2: the first and second half of the recording (frames 0-5319, 5320-10639);
      q1-q4: its quarters.  The wand's poses differ between them.
  T4  fit_L: the wand length free (the tube's radii and indices then set the scale).
  T5  fit_k1: k1 free.
  T6  thinned: the kept frames thinned so that a ball moves by 10 px or more (in either camera) between kept
      frames; consecutive frames repeat almost the same pose.
Compared with S2 (calibrate.json) per camera: f, the distance from the tube's axis, cam2's azimuth about the axis
from cam1, cam2's height along the axis relative to cam1, and the angle between the viewing directions.  For the
subsets (T3, T6) a difference is scaled by sqrt(sigma_subset^2 - sigma_S2^2), the spread expected between nested
fits; T4 and T5 report the fitted L and k1 against their sigmas.
Judged by (plan): T3 every part within about 3 sigma, none in another minimum; T4 the fitted length agrees with
10 mm within its uncertainty; T5 k1 kept only if it clearly lowers the residuals; T6 the same answer within sigma.
Writes t3_t6_variants_out.txt and t3_t6_variants.json beside this script.
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

N_ALL = 10640
VARIANTS = {
    "h1": {"frames": (0, 5319)}, "h2": {"frames": (5320, 10639)},
    "q1": {"frames": (0, 2659)}, "q2": {"frames": (2660, 5319)},
    "q3": {"frames": (5320, 7979)}, "q4": {"frames": (7980, 10639)},
    "fit_L": {"over": {"fit_L": True}}, "fit_k1": {"over": {"fit_k1": True}},
    "thinned": {"thin_px": 10.0},
}


def thin(frames, uv, step_px):
    """Keep a frame when a ball has moved by step_px or more, in any camera, since the last kept frame."""
    keep = [0]
    for i in range(1, len(frames)):
        d = np.linalg.norm(uv[i] - uv[keep[-1]], axis=-1)
        if np.nanmax(d) >= step_px:
            keep.append(i)
    return keep


def run_variant(name):
    cfg = case.load_config()
    spec = VARIANTS[name]
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    cams = case.cam_ids(cfg)
    uv = case.uv_array(kept, frames, cams)
    idx = list(range(len(frames)))
    if "frames" in spec:
        lo, hi = spec["frames"]
        idx = [i for i, f in enumerate(frames) if lo <= f <= hi]
    if "thin_px" in spec:
        idx = thin(frames, uv, float(spec["thin_px"]))
    fr = [frames[i] for i in idx]
    lib = case.library()
    job = case.make_job(lib, cfg, fr, uv[idx], **spec.get("over", {}))
    folder = os.path.join(cfg["out_dir"], "checks", name)
    lines = []
    try:
        _sol, result, secs = case.run_calibration(lib, job, folder, log=lambda m: lines.append(m))
    except Exception as e:                               # report it, keep the other variants
        return name, {"error": f"{type(e).__name__}: {e}", "frames": len(fr)}
    s = case.summary(result)
    s["seconds"] = secs
    s["n_frames_given"] = len(fr)
    s["checks"] = [c["title"] for c in result.get("checks", [])]
    return name, s


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


def compare(name, s, ref, log):
    rows = []
    nested = name not in ("fit_L", "fit_k1")
    for c, (cv, cr) in enumerate(zip(s["cameras"], ref["cameras"])):
        for q, sq in (("f_px", "f_sigma_px"), ("distance_mm", "distance_sigma_mm"), ("azimuth_deg", "azimuth_sigma_deg"),
                      ("height_rel_mm", "height_sigma_mm")):
            if c == 0 and q in ("azimuth_deg", "height_rel_mm"):
                continue                                  # cam1 defines the azimuth and the height origin
            d = cv[q] - cr[q]
            sv, sr = cv.get(sq), cr.get(sq)
            if nested and sv and sr and sv > sr:
                scale = math.sqrt(sv * sv - sr * sr)
            else:
                scale = sv or sr
            z = d / scale if scale else float("nan")
            rows.append({"camera": cv["name"], "quantity": q, "variant": cv[q], "S2": cr[q], "diff": d,
                         "sigma_used": scale, "z": z})
        rows.append({"camera": cv["name"], "quantity": "viewing direction (deg)",
                     "diff": angle_deg(cv["viewing_direction"], cr["viewing_direction"])})
    zs = [abs(r["z"]) for r in rows if "z" in r and np.isfinite(r["z"])]
    worst = max(zs) if zs else float("nan")
    log(f"  {name}: {s['frames']} frames ({s['n_frames_given']} given), RMS {s['rms_px']:.4f} px, "
        f"{s['active']}/{s['observations']} used, {s['seconds']:.0f} s; largest |z| {worst:.2f}")
    for r in rows:
        if "z" in r:
            log(f"      {r['camera']} {r['quantity']:<14s} {r['variant']:>12.4f}  S2 {r['S2']:>12.4f}  "
                f"diff {r['diff']:+10.4f}  (sigma {r['sigma_used']:.4f}, z {r['z']:+.2f})")
        else:
            log(f"      {r['camera']} {r['quantity']:<14s} diff {r['diff']:.4f}")
    if s.get("L_sigma_mm"):
        zL = (s["wand_length_mm"] - 10.0) / s["L_sigma_mm"]
        log(f"      fitted wand length {s['wand_length_mm']:.5f} mm +- {s['L_sigma_mm']:.5f} "
            f"({1e3 * (s['wand_length_mm'] - 10.0):+.1f} um, {zL:+.2f} sigma from 10 mm)")
    for cv in s["cameras"]:
        if cv.get("k1_sigma"):
            log(f"      {cv['name']} k1 {cv['k1']:+.4e} +- {cv['k1_sigma']:.4e} ({cv['k1'] / cv['k1_sigma']:+.2f} sigma)")
    if s.get("checks"):
        log(f"      checks: {'; '.join(s['checks'])}")
    return {"rows": rows, "largest_abs_z": worst}


def main():
    cfg = case.load_config()
    names = sys.argv[1:] or list(VARIANTS)
    log = case.Log(os.path.join(HERE, "t3_t6_variants"))
    with open(os.path.join(os.path.dirname(HERE), "calibrate.json"), encoding="utf-8") as fh:
        ref = json.load(fh)["summary"]
    log(f"T3-T6: {len(names)} variants against S2 (RMS {ref['rms_px']:.4f} px, {ref['frames']} frames)")
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=min(len(names), 9)) as ex:
        for name, s in ex.map(run_variant, names):
            res[name] = s
    log(f"  all variants ran in {time.time() - t0:.0f} s")
    out = {"S2": ref, "variants": {}}
    for name in names:
        s = res[name]
        if "error" in s:
            log(f"  {name}: FAILED: {s['error']}")
            out["variants"][name] = s
            continue
        out["variants"][name] = {"summary": s, "comparison": compare(name, s, ref, log)}
    log.save(out)


if __name__ == "__main__":
    main()
