"""D8 (plan): do the parts' cameras agree inside the measured volume, whatever their f?

    python tests/d8_held_out.py      (after tests/t3_variants.py)

In T0 the halves and quarters of the recording gave f from 32,800 to 40,200 px (cam1), each with its distance from
the axis changed in proportion (f / D about constant: the near-telecentric valley), yet each half's cameras measured
the other half's wand lengths to 0.07-0.13 % (T0's D8).  This evaluates every part's cameras (S2, T3's h1, h2,
q1-q4) on all of S1's kept frames, by free triangulation (no wand length, no fitted poses):
  - the wand length per frame, against 10 mm;
  - each ball's reprojection residual after a maximum-likelihood triangulation from both cameras (the two cameras'
    disagreement; 1 degree of freedom per ball);
both over the frames the variant was not fitted on (held out) and over its own frames.
Writes d8_held_out_out.txt and d8_held_out.json beside this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

VARIANTS = ["S2", "h1", "h2", "q1", "q2", "q3", "q4"]


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "d8_held_out"))
    lib = case.library()
    cw, cm = lib.cw, lib.cm
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    cams_ids = case.cam_ids(cfg)
    uv = case.uv_array(kept, frames, cams_ids)
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    data = cw.WandData(uv, float(cfg["wand"]["length_mm"]), [(W, H)] * len(cams_ids))
    obs = cw.observations(data)
    fid = np.array([frames[i] for i in obs.frame_ids])           # frame number of each obs frame
    log(f"D8: every variant's cameras on all {obs.n_frames} kept frames (free triangulation)")
    log(f"  {'variant':>8s} {'f cam1':>8s} {'f cam2':>8s}   length error RMS (um): held out / own   "
        "reprojection RMS (px): held out / own")
    out = {}
    for name in VARIANTS:
        folder = cfg["out_dir"] if name == "S2" else os.path.join(cfg["out_dir"], "checks", name)
        with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
            res = json.load(fh)
        cams = [cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"],
                          c["height"], np.array(c["dist"])) for c in res["cameras"]]
        tube = cm.Tube(tuple(res["tube"]["radii_mm"]), tuple(res["tube"]["n"]))
        own = set(int(f) for f in res["gui"]["settings"]["frame_ids"])
        L_ref = float(res["wand_length_mm"])
        rows = np.arange(len(obs))
        XA, XB, cnt = cw.triangulate(cams, tube, obs, rows)
        Lm = np.linalg.norm(XB - XA, axis=1)
        Lm[np.any(cnt < 2, axis=1)] = np.nan
        dL = 1e3 * (Lm - L_ref)
        # maximum-likelihood triangulation of every ball, from the least-squares start
        point = obs.frame * 2 + obs.end
        X0 = np.empty((2 * obs.n_frames, 3))
        X0[0::2], X0[1::2] = XA, XB
        okp = np.all(np.isfinite(X0), axis=1)
        X0[~okp] = 0.0
        _X, r, _cov = cw.triangulate_ml(cams, tube, obs.cam, obs.uv, point, X0)
        rn = np.linalg.norm(r, axis=1)
        held_f = np.array([int(f) not in own for f in fid])
        held_r = held_f[obs.frame]
        rec = {"f": [c["f_px"] for c in res["cameras"]], "frames_own": int((~held_f).sum()),
               "frames_held_out": int(held_f.sum())}
        for tag, sf, sr in (("held_out", held_f, held_r), ("own", ~held_f, ~held_r)):
            d = dL[sf & np.isfinite(dL)]
            rr = rn[sr & np.isfinite(rn)]
            rec[tag] = {"length_rms_um": float(np.sqrt(np.mean(d ** 2))) if len(d) else None,
                        "length_mean_um": float(np.mean(d)) if len(d) else None,
                        "reproj_rms_px": float(np.sqrt(np.mean(rr ** 2))) if len(rr) else None}
        out[name] = rec

        def fmt(v, p):
            return f"{v:{p}}" if v is not None else "   -  "
        log(f"  {name:>8s} {rec['f'][0]:8.0f} {rec['f'][1]:8.0f}   "
            f"{fmt(rec['held_out']['length_rms_um'], '6.1f')} / {fmt(rec['own']['length_rms_um'], '6.1f')}"
            f" (mean {fmt(rec['held_out']['length_mean_um'], '+6.1f')} / {fmt(rec['own']['length_mean_um'], '+6.1f')})"
            f"        {fmt(rec['held_out']['reproj_rms_px'], '6.3f')} / {fmt(rec['own']['reproj_rms_px'], '6.3f')}")
    log.save(out)


if __name__ == "__main__":
    main()
