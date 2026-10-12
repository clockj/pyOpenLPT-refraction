"""T4: held-out frames.  Each T3 fit's cameras judged by evaluate.py (free triangulation of every frame: reprojection
error and wand length) on the frames it was fitted to and on those it never saw: the other 80 % of the frames for an
interleaved subset, the other half for a half.  Pass: as good on the unused frames as on its own.

    python tests/t4_heldout.py t3 [t3k]
Writes t4_heldout_out.txt and t4_heldout.json beside this script.
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

SUBS = ["s0", "s1", "s2", "s3", "s4", "h1", "h2", "h2_from_h1"]


def brief(e: dict) -> dict:
    w = e["wand_error_mm"]
    return {"frames": e["frames"], "median_px": e["euclidean_px"]["median"], "p99_px": e["euclidean_px"]["p99"],
            "rms_inliers_px": e["rms_inliers_px"], "outliers": e["outlier_detections"],
            "wand_abs_median_mm": w["abs"]["median"], "wand_abs_p90_mm": w["abs"]["p90"],
            "wand_mean_mm": e["L"] + w["clean_frames"]["mean"], "wand_clean_sd_mm": w["clean_frames"]["std"],
            "wand_clean_abs_mean_mm": w["clean_frames"]["abs"]["mean"]}


def main():
    groups = sys.argv[1:] or ["t3", "t3k"]
    cfg = case.load_config()
    log = case.Tee(os.path.join(HERE, "t4_heldout_out.txt"))
    log("T4: each fit's cameras on its own frames and on the frames it did not use (evaluate.py)")
    log("  (mean L and SD over the frames without a gross detection error, evaluate.py)")
    log("  fit        frames  median px  p99 px  inl.RMS  outl.  |dL| median  p90 mm  mean L mm  clean SD  "
        "| unused: frames  median px  p99 px  inl.RMS  outl.  |dL| median  p90 mm  mean L mm  clean SD")
    out = {}
    for g in groups:
        for s in SUBS:
            p = os.path.join(HERE, "states", f"{g}_{s}.pkl")
            if not os.path.isfile(p):
                continue
            st = case.load_state(p)
            rc, cd, rows, data, obs, frames = case.rebuild(st, cfg)
            n = cd.uv.shape[0]
            other = np.setdiff1d(np.arange(n), rows)
            res = {}
            for k, rr in (("own", rows), ("unused", other)):
                e = ev.evaluate(st["cams"], st["rig"], cd.uv[rr], rc.wand_length_mm, rc.sizes,
                                frame_ids=cd.frame_ids[rr])
                res[k] = brief(e)
            out[f"{g}_{s}"] = res
            a, b = res["own"], res["unused"]
            log(f"  {g + '_' + s:9s} {a['frames']:6d}  {a['median_px']:.4f}   {a['p99_px']:6.3f}  {a['rms_inliers_px']:.4f}"
                f" {a['outliers']:5d}    {a['wand_abs_median_mm']:.4f}   {a['wand_abs_p90_mm']:.4f}   "
                f"{a['wand_mean_mm']:.4f}   {a['wand_clean_sd_mm']:.4f}  |  {b['frames']:6d}    {b['median_px']:.4f}"
                f"   {b['p99_px']:6.3f}  {b['rms_inliers_px']:.4f} {b['outliers']:5d}    {b['wand_abs_median_mm']:.4f}"
                f"   {b['wand_abs_p90_mm']:.4f}   {b['wand_mean_mm']:.4f}   {b['wand_clean_sd_mm']:.4f}")
    with open(os.path.join(HERE, "t4_heldout.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
