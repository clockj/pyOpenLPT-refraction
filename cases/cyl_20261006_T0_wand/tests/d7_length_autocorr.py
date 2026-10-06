"""D7 (diagnosis after D6): how fast does the wand-length error change from frame to frame?

    python tests/d7_length_autocorr.py [RESULT_FOLDER]      (default: config out_dir, S2's result)

The autocorrelation of S2's free-triangulation length error between frames f and f+k (both kept), k = 1 ... 300,
and the same for each camera's image distance A-B after removing its smooth trend (a 61-frame running median), to
see whether the length error is frame-to-frame noise or persists over many frames (a pose- or time-dependent error).
Writes d7_length_autocorr_out.txt and .json beside this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

LAGS = (1, 2, 3, 5, 10, 20, 50, 100, 300)


def autocorr(series: dict, lag: int):
    keys = [f for f in series if f + lag in series]
    if len(keys) < 30:
        return None, 0
    a = np.array([series[f] for f in keys])
    b = np.array([series[f + lag] for f in keys])
    return float(np.corrcoef(a, b)[0, 1]), len(keys)


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    log = case.Log(os.path.join(HERE, "d7_length_autocorr"))
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    Lf = data["wand_length_free_mm"]
    act = data["frame_active"]
    s = {int(f): 1e3 * (v - 10.0) for f, v, a in zip(data["frame_ids"], Lf, act) if v is not None and a}
    out = {"length": {}}
    log(f"D7: autocorrelation of the length error ({len(s)} frames), lag in frames: correlation (pairs)")
    log("  " + "; ".join(f"{k}: {c:+.2f} ({n})" for k in LAGS for c, n in [autocorr(s, k)] if c is not None))
    for k in LAGS:
        c, n = autocorr(s, k)
        out["length"][k] = c
    log.save(out)


if __name__ == "__main__":
    main()
