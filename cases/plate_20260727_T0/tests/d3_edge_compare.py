"""D3: the detections whose ball crosses the image border dropped before fitting (run_calibration.py --edge-margin
2): does that change the measurement?  Each pair (the same frames fitted without and with the filter) is judged by
evaluate.py on the frames neither fit used, once on every detection and once without the border-cut ones (they are
biased: a truncated ball's centre moves inwards).  Also the rigs side by side.

    python tests/d3_edge_compare.py
Writes d3_edge_compare_out.txt and .json beside this script.
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

PAIRS = [("t3k_s4", "d3_edge_s4k"), ("t3k_s0", "d3_edge_s0k")]


def main():
    cfg = case.load_config()
    rc = case.rig_config(cfg)
    cd_all = case.read_detections(cfg, "all")
    cd_cut = case.read_detections(cfg, "all")
    n_cut = case.drop_truncated(cd_cut, rc.sizes, 2.0)
    log = case.Tee(os.path.join(HERE, "d3_edge_compare_out.txt"))
    log(f"D3: fits without and with the edge filter ({n_cut} border-cut detections dropped), judged on the frames "
        "neither used")
    out = {}
    for a, b in PAIRS:
        sa = case.load_state(os.path.join(HERE, "states", f"{a}.pkl"))
        sb = case.load_state(os.path.join(HERE, "states", f"{b}.pkl"))
        rows = case.subset_rows(cd_all.uv.shape[0], sa["extra"]["settings"])
        other = np.setdiff1d(np.arange(cd_all.uv.shape[0]), rows)
        log(f"\n== {a} (no filter) against {b} (filter): {len(other)} unused frames")
        for name, st in ((a, sa), (b, sb)):
            d = case.describe(st["cams"], st["rig"])
            log(f"  {name}: RMS {st['rms']:.4f} px; walls {d['walls_angle_deg']:.3f} deg; f "
                + ", ".join(f"{c['f_px']:.0f}" for c in d["cameras"]) + "; to wall "
                + ", ".join(f"{c['distance_to_wall_mm']:.1f}" for c in d["cameras"]) + " mm; k1 "
                + ", ".join(f"{c['k1']:.3f}" for c in d["cameras"]))
            for tag, cd in (("every detection", cd_all), ("border-cut dropped", cd_cut)):
                e = ev.evaluate(st["cams"], st["rig"], cd.uv[other], rc.wand_length_mm, rc.sizes)
                w = e["wand_error_mm"]
                rec = {"median_px": e["euclidean_px"]["median"], "rms_inliers_px": e["rms_inliers_px"],
                       "outliers": e["outlier_detections"], "wand_abs_median_mm": w["abs"]["median"],
                       "wand_abs_p90_mm": w["abs"]["p90"], "wand_clean_mean_mm": e["L"] + w["clean_frames"]["mean"],
                       "wand_clean_sd_mm": w["clean_frames"]["std"]}
                out.setdefault(f"{a}|{b}", {}).setdefault(name, {})[tag] = rec
                log(f"     {tag:18s}: median {rec['median_px']:.4f} px, inliers RMS {rec['rms_inliers_px']:.4f}, "
                    f"outliers {rec['outliers']}; |dL| median {rec['wand_abs_median_mm']:.4f} mm, p90 "
                    f"{rec['wand_abs_p90_mm']:.4f}; clean mean L {rec['wand_clean_mean_mm']:.4f}, SD "
                    f"{rec['wand_clean_sd_mm']:.4f}")
    with open(os.path.join(HERE, "d3_edge_compare.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
