"""V2b (after v2_all_three.py): the wand length of C1 and of the all-three fit on the same frames.

    python tests/v2b_wand_length_same_frames.py

V2 printed each fit's wand-length RMS over its own frames (C1: every frame kept in at least two cameras; all3: the
frames all three keep), which mixes a frame effect into the comparison: a ball triangulated from three cameras is
better placed than from two.  This reads each fit's free-triangulation wand lengths (calibration_data.json,
wand_length_free_mm: each frame triangulated by that fit's cameras from every camera that kept it) on the frames both
fits used, which are the frames all three cameras keep.
Writes v2b_wand_length_same_frames_out.txt and .json beside this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402


def lengths(folder):
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    L = np.array([np.nan if x is None else x for x in d["wand_length_free_mm"]], float)
    ok = np.isfinite(L) & np.array(d["frame_active"], bool)
    return {int(f): float(x) for f, x, k in zip(d["frame_ids"], L, ok) if k}


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "v2b_wand_length_same_frames"))
    L0 = float(cfg["wand"]["length_mm"])
    a = lengths(cfg["out_dir"])
    b = lengths(os.path.join(cfg["out_dir"], "checks", "all3"))
    common = sorted(set(a) & set(b))
    only_c1 = sorted(set(a) - set(b))
    out = {"common_frames": len(common), "c1_only_frames": len(only_c1)}
    log(f"V2b: the wand length's free-triangulation error (um); {len(common)} frames both fits used (all three cameras), "
        f"{len(only_c1)} frames only C1 used (two cameras)")
    for name, d, fr in (("C1, frames both used", a, common), ("all3, frames both used", b, common),
                        ("C1, its two-camera frames", a, only_c1)):
        e = 1e3 * (np.array([d[f] for f in fr]) - L0)
        rec = {"n": int(len(e)), "rms_um": float(np.sqrt(np.mean(e ** 2))), "mean_um": float(e.mean()),
               "median_abs_um": float(np.median(np.abs(e))), "p95_abs_um": float(np.percentile(np.abs(e), 95))}
        out[name] = rec
        log(f"  {name}: n {rec['n']}, RMS {rec['rms_um']:.1f}, mean {rec['mean_um']:+.1f}, median |e| "
            f"{rec['median_abs_um']:.1f}, 95 % {rec['p95_abs_um']:.1f}")
    log.save(out)


if __name__ == "__main__":
    main()
