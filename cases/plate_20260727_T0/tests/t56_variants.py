"""T5 and T6: the same frames fitted with different inputs, against a plain fit of those frames.
  On subset s2 against t3k_s2 (k1 fitted, n_water 1.333, walls 12.7 mm, no hints):
  T5  t5k_hints_s2    rough window-normal hints from the photo (each camera square-on to its wall);
  T6  t6k_nw1.33_s2   n_water 1.33 (the GUI's value);
      t6k_t12.4_s2    walls 12.4 mm;   t6k_t13.0_s2  walls 13.0 mm;
      t3_s2           k1 held at 0.
  On subset s0 against t3_s0 (k1 held at 0; it ended in a wrong minimum): the same variants (t5_hints_s0, t6_*_s0)
  and t3k_s0 (k1 fitted).
For each: RMS, noise, warnings, how far the rig moved (f, distances to the walls, the walls' angle, baselines; in the
plain fit's sigmas), how far apart the two answers' predictions are (answers_apart, chi^2 on the plain fit's
detections and weights), and both judged on the 80 % of the frames neither used (evaluate.py).
Pass (T5): the same answer as without hints.  T6: reported, k1 kept only if it clearly lowers the residuals.

    python tests/t56_variants.py
Writes t56_variants_out.txt and .json beside this script.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import evaluate as ev  # noqa: E402
from t3_repeatability import solution_of  # noqa: E402

COMPARISONS = [("t3k_s2", ["t5k_hints_s2", "t6k_nw1.33_s2", "t6k_t12.4_s2", "t6k_t13.0_s2", "t3_s2"]),
               ("t3_s0", ["t5_hints_s0", "t6_nw1.33_s0", "t6_t12.4_s0", "t6_t13.0_s0", "t3k_s0"])]


def main():
    cfg = case.load_config()
    log = case.Tee(os.path.join(HERE, "t56_variants_out.txt"))
    out = {}
    for base, variants in COMPARISONS:
        out[base] = compare(cfg, base, variants, log)
        log("")
    with open(os.path.join(HERE, "t56_variants.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    log.close()


def compare(cfg, BASE, VARIANTS, log):
    names = [c["name"] for c in cfg["rig"]["cameras"]]
    stb = case.load_state(os.path.join(HERE, "states", f"{BASE}.pkl"))
    rc, cd, rows, data, obs_b, frames_b = case.rebuild(stb, cfg)
    sol_b = solution_of(stb, obs_b, frames_b)
    sig = case.pw.camera_sigmas(sol_b.cams, sol_b.rig, sol_b.layout, sol_b.cov)
    db = case.describe(sol_b.cams, sol_b.rig)
    other = np.setdiff1d(np.arange(cd.uv.shape[0]), rows)
    s02 = stb["sigma_px"] ** 2

    def judged(st):
        e = ev.evaluate(st["cams"], st["rig"], cd.uv[other], rc.wand_length_mm, rc.sizes)
        w = e["wand_error_mm"]
        return {"median_px": e["euclidean_px"]["median"], "rms_inliers_px": e["rms_inliers_px"],
                "outliers": e["outlier_detections"], "wand_abs_median_mm": w["abs"]["median"],
                "wand_abs_p90_mm": w["abs"]["p90"], "wand_mean_mm": e["L"] + w["clean_frames"]["mean"],
                "wand_clean_sd_mm": w["clean_frames"]["std"]}

    jb = judged(stb)
    log(f"T5/T6: {len(rows)} frames, against {BASE}: RMS {stb['rms']:.4f} px, sigma {stb['sigma_px']:.4f} px")
    log(f"  {BASE}: f " + ", ".join(f"{c['f_px']:.0f}" for c in db["cameras"]) + " px; distance to wall "
        + ", ".join(f"{c['distance_to_wall_mm']:.1f}" for c in db["cameras"]) + f" mm; walls {db['walls_angle_deg']:.3f}"
        f" deg; unused frames: median {jb['median_px']:.4f} px, inliers RMS {jb['rms_inliers_px']:.4f}, |dL| median "
        f"{jb['wand_abs_median_mm']:.4f} mm, p90 {jb['wand_abs_p90_mm']:.4f}, mean L (clean frames) "
        f"{jb['wand_mean_mm']:.4f}")
    out = {"base": {"name": BASE, "describe": db, "unused": jb, "rms_px": stb["rms"], "sigma_px": stb["sigma_px"]},
           "variants": {}}
    for v in VARIANTS:
        p = os.path.join(HERE, "states", f"{v}.pkl")
        if not os.path.isfile(p):
            log(f"  {v}: no state")
            continue
        st = case.load_state(p)
        _rc, _cd, rows_v, _data, obs_v, frames_v = case.rebuild(st, cfg)
        sol_v = solution_of(st, obs_v, frames_v)
        d = case.describe(sol_v.cams, sol_v.rig)
        warn = [w["code"] for w in case.pio.assess(sol_v, names)]
        # how far apart the predictions are, on the plain fit's detections and weights
        sol_vb = case.pw.Solution(sol_v.cams, sol_v.rig, sol_v.frames, sol_v.L, obs_b, sol_v.layout)
        try:
            apart = float(case.pw.answers_apart(sol_b, sol_vb, obs_b, stb["weights"], s02))
        except Exception as exc:  # noqa: BLE001
            apart = repr(exc)
        jv = judged(st)
        rec = {"rms_px": st["rms"], "sigma_px": st["sigma_px"], "warnings": warn, "describe": d, "apart_chi2": apart,
               "unused": jv, "hints": st["info"].get("hints"), "k1": [c["k1"] for c in d["cameras"]]}
        out["variants"][v] = rec
        log(f"  {v}: RMS {st['rms']:.4f} px, sigma {st['sigma_px']:.4f}; warnings {', '.join(warn) or '-'}; answers "
            f"apart chi2 {apart if isinstance(apart, str) else f'{apart:.2f}'}")
        log("     f " + ", ".join(f"{c['f_px']:.0f} ({(c['f_px'] - b['f_px']) / s['f_px']:+.1f} s)"
                                  for c, b, s in zip(d["cameras"], db["cameras"], sig))
            + "; distance to wall " + ", ".join(
                f"{c['distance_to_wall_mm']:.1f} ({(c['distance_to_wall_mm'] - b['distance_to_wall_mm']) / s['distance_mm']:+.1f} s)"
                for c, b, s in zip(d["cameras"], db["cameras"], sig))
            + f"; walls {d['walls_angle_deg']:.3f} deg")
        log("     baselines " + ", ".join(f"{p_['baseline_mm'] - q['baseline_mm']:+.2f}" for p_, q in
                                          zip(d["pairs"], db["pairs"])) + " mm against the plain fit"
            + ("; k1 " + ", ".join(f"{k:.3g}" for k in rec["k1"]) if any(rec["k1"]) else ""))
        log(f"     unused frames: median {jv['median_px']:.4f} px, inliers RMS {jv['rms_inliers_px']:.4f}, outliers "
            f"{jv['outliers']}, |dL| median {jv['wand_abs_median_mm']:.4f} mm, p90 {jv['wand_abs_p90_mm']:.4f}, mean L (clean) "
            f"{jv['wand_mean_mm']:.4f}, clean SD {jv['wand_clean_sd_mm']:.4f}")
        if rec["hints"]:
            h = rec["hints"]
            log(f"     hints: normals off by " + ", ".join(f"{x:.1f}" for x in (h.get("normals_off_deg") or []))
                + f" deg from the answer; path chosen: {h.get('chosen')}")
    return out


if __name__ == "__main__":
    main()
