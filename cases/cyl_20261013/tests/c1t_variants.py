"""C1-T and C1-P: the calibration on each half and quarter of the recording, and on each pair of cameras (plan:
2026-10-15-plan.md, Phase C), each compared with C1's result.  cases/cyl_20261008's c1t_variants.py, unchanged in
its code: the halves and quarters split the image numbers every camera saved (here 0-12000), and the pairs p12,
p13, p23 use all of C1's frames with two of the three cameras.

    python tests/c1t_variants.py [NAME ...]      (default: all; after calibrate.py)

Each variant is a full calibration (cylinder_wand_gui.run, the settings of config.json), run in its own process,
with the GUI's files written to <out_dir>/checks/<name>/:
  h1, h2: the first and second half of the image numbers every camera saved (all three cameras);
  q1-q4: its quarters.  The wand's poses differ between them;
  p12, p13, p23: two cameras, every frame both keep.
Compared with C1 (calibrate.json) per camera: f, the distance from the tube's axis, each camera's azimuth about the
axis and height along it relative to the part's first camera, and the angle between the viewing directions.  A difference
is scaled by sqrt(sigma_part^2 - sigma_C1^2), the spread expected between nested fits.
Judged by (plan): f and the distances agree within their sigma; the pairs agree with the three-camera fit.
Writes c1t_variants_out.txt and c1t_variants.json beside this script.
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


def variants(cfg):
    """h1, h2, q1-q4 as image-number ranges (inclusive) over the images every camera saved; p12, p13, p23 as pairs of
    cameras over every frame."""
    first = max(case.first_image(cfg, c) for c in case.cam_ids(cfg))
    last = int(cfg["frames"]["last"])
    n = last - first + 1
    out = {}
    for k in range(2):
        out[f"h{k + 1}"] = {"frames": (first + k * n // 2, first + (k + 1) * n // 2 - 1)}
    for k in range(4):
        out[f"q{k + 1}"] = {"frames": (first + k * n // 4, first + (k + 1) * n // 4 - 1)}
    cams = case.cam_ids(cfg)
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            out[f"p{a}{b}"] = {"frames": (int(cfg["frames"]["first"]), last), "cameras": [a, b]}
    return out


def run_variant(name):
    cfg = case.load_config()
    spec = variants(cfg)[name]
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    cams = case.cam_ids(cfg)
    uv = case.uv_array(kept, frames, cams)
    lo, hi = spec["frames"]
    use = spec.get("cameras", cams)
    ci = [cams.index(c) for c in use]
    seen = np.all(np.isfinite(uv[:, ci]), axis=(2, 3))                  # (F, len(use)): both balls kept
    idx = [i for i, f in enumerate(frames) if lo <= f <= hi and int(seen[i].sum()) >= 2]
    fr = [frames[i] for i in idx]
    lib = case.library()
    job = case.make_job(lib, cfg, fr, uv[idx][:, ci])
    job.cam_ids = list(use)
    job.sizes = [case.size(cfg, c) for c in use]
    job.pixel_mm = [float(case.cam_cfg(cfg, c)["pixel_mm"]) for c in use]
    folder = os.path.join(cfg["out_dir"], "checks", name)
    lines = []
    try:
        _sol, result, secs = case.run_calibration(lib, job, folder, log=lambda m: lines.append(m))
    except Exception as e:                               # report it, keep the other variants
        return name, {"error": f"{type(e).__name__}: {e}", "frames": len(fr)}
    s = case.summary(result, cfg)
    s["cameras_used"] = list(use)
    s["seconds"] = secs
    s["n_frames_given"] = len(fr)
    s["frame_range"] = [lo, hi]
    s["checks"] = [c["title"] for c in result.get("checks", [])]
    return name, s


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


def compare(name, s, ref, log):
    """The part's cameras against C1's same cameras; azimuth and height relative to the part's first camera (C1's
    taken relative to the same camera)."""
    rows = []
    byname = {c["name"]: c for c in ref["cameras"]}
    first = byname[s["cameras"][0]["name"]]
    ref_cams = []
    for cv in s["cameras"]:
        cr = dict(byname[cv["name"]])
        cr["azimuth_deg"] = (cr["azimuth_deg"] - first["azimuth_deg"] + 180.0) % 360.0 - 180.0
        cr["height_rel_mm"] = cr["height_rel_mm"] - first["height_rel_mm"]
        ref_cams.append(cr)
    s_cams = []
    for cv in s["cameras"]:
        cv = dict(cv)
        cv["azimuth_deg"] = (cv["azimuth_deg"] - s["cameras"][0]["azimuth_deg"] + 180.0) % 360.0 - 180.0
        s_cams.append(cv)
    for c, (cv, cr) in enumerate(zip(s_cams, ref_cams)):
        for q, sq in (("f_px", "f_sigma_px"), ("distance_mm", "distance_sigma_mm"), ("azimuth_deg", "azimuth_sigma_deg"),
                      ("height_rel_mm", "height_sigma_mm")):
            if c == 0 and q in ("azimuth_deg", "height_rel_mm"):
                continue                                  # cam1 defines the azimuth and the height origin
            d = cv[q] - cr[q]
            sv, sr = cv.get(sq), cr.get(sq)
            if sv and sr and sv > sr:
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
    log(f"  {name} (frames {s['frame_range'][0]}-{s['frame_range'][1]}): {s['frames']} frames, RMS {s['rms_px']:.4f} px, "
        f"{s['active']}/{s['observations']} used, {s['seconds']:.0f} s; largest |z| {worst:.2f}")
    for r in rows:
        if "z" in r:
            log(f"      {r['camera']} {r['quantity']:<14s} {r['variant']:>12.4f}  C1 {r['S2']:>12.4f}  "
                f"diff {r['diff']:+10.4f}  (sigma {r['sigma_used']:.4f}, z {r['z']:+.2f})")
        else:
            log(f"      {r['camera']} {r['quantity']:<14s} diff {r['diff']:.4f}")
    if s.get("checks"):
        log(f"      checks: {'; '.join(s['checks'])}")
    return {"rows": rows, "largest_abs_z": worst}


def main():
    cfg = case.load_config()
    names = sys.argv[1:] or list(variants(cfg))
    log = case.Log(os.path.join(HERE, "c1t_variants"))
    with open(os.path.join(os.path.dirname(HERE), "calibrate.json"), encoding="utf-8") as fh:
        ref = json.load(fh)["summary"]
    log(f"C1-T: {len(names)} parts of the recording against C1 (RMS {ref['rms_px']:.4f} px, {ref['frames']} frames)")
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=min(len(names), 6)) as ex:
        for name, s in ex.map(run_variant, names):
            res[name] = s
    log(f"  all parts ran in {time.time() - t0:.0f} s")
    out = {"S2": ref, "variants": {}}
    for name in names:
        s = res[name]
        if "error" in s:
            log(f"  {name}: FAILED: {s['error']}")
            out["variants"][name] = s
            continue
        out["variants"][name] = {"summary": s, "comparison": compare(name, s, ref, log)}
    for cr in ref["cameras"]:
        fs = [c["f_px"] for n in names if "summary" in out["variants"][n]
              for c in out["variants"][n]["summary"]["cameras"] if c["name"] == cr["name"]]
        if fs:
            log(f"  f over the parts: {cr['name']} {min(fs):.0f}-{max(fs):.0f} px (C1 {cr['f_px']:.0f} +- "
                f"{cr['f_sigma_px'] or float('nan'):.0f})")
    log.save(out)


if __name__ == "__main__":
    main()
