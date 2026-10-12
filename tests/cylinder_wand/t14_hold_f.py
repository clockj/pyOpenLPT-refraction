"""T14 (case cyl_20261007, plan step C3): the hold-f option of the wand calibration, on synthetic data.

    python tests/cylinder_wand/t14_hold_f.py [N_SEEDS]      (default 4)

cylinder_wand.calibrate(..., fix_f=(...)) holds the f of the chosen cameras at the value of the start (Layout leaves
their log f column out).  With two long-lens cameras the wand leaves f along a near-telecentric valley (the 6 Oct
runs: the halves and quarters of a recording disagreed by +-10 % in f, with the distance in proportion), so f from a
dot board is to be held instead.

A rig like the real one (cases/cyl_20261007): two cameras 1,000 mm from the axis, 61 deg apart about it, f 38,000 px
(f / D 38 px/mm), sensor 1280 x 800 px, looking at the axis; the case's tube; 2,000 wand frames (10 mm) with
midpoints in a box of +-12 x +-8 x +-12 mm about the axis, detections from the C++ PINCYL camera plus 0.4 px of
Gaussian noise.  Per seed, three fits from the truth (in the gauge):
  - "free": f free (today's default);
  - "held": both f held at the truth;
  - "held +1 %": both f held 1 % above the truth.
Reported: f and the distance D from the axis against the truth (relative), the rotation and centre errors, the
RMS, and the wand length of a free triangulation of 500 held-out frames (mean error, um).
Judged by (case plan, C3): with f held at the truth the cameras are recovered within their error bars (|z| < 3);
with f held 1 % off the fit moves D by about 1 % (it stays on the f / D valley), the RMS barely changes, and the
held-out wand length shows how much a 1 % error in f matters inside the volume.
Writes t14_hold_f_out.txt and t14_hold_f.json beside this script.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

import rigs
from rigs import cm

import cylinder_wand as cw  # noqa: E402

# a frozen copy of the extension, if given, before rigs' build-py/Release (Windows locks a loaded .pyd)
if os.environ.get("CASE_PYOPENLPT_DIR"):
    sys.path.insert(0, os.environ["CASE_PYOPENLPT_DIR"])

HERE = rigs.HERE
TUBE = rigs.CASE_TUBE
L = 10.0
NOISE = 0.4
N_FRAMES = 2000
N_HELD = 500


def rig():
    cams = [rigs.ring_camera(0.0, 1000.0, 38000.0, 1280, 800), rigs.ring_camera(61.0, 1000.0, 38000.0, 1280, 800)]
    return rigs.gauge_fix(cams)


def held_out_length(cams, wand):
    data = cw.WandData(wand.uv, L, [(c.width, c.height) for c in cams])
    obs = cw.observations(data)
    XA, XB, cnt = cw.triangulate(cams, TUBE, obs)
    Lm = np.linalg.norm(XB - XA, axis=1)
    ok = np.all(cnt >= 2, axis=1) & np.isfinite(Lm)
    d = 1e3 * (Lm[ok] - L)
    return float(np.mean(d)), float(np.sqrt(np.mean(d ** 2)))


def run(seed, log):
    rng = np.random.default_rng(1000 + seed)
    truth = rig()
    wand = rigs.wand_frames(truth, TUBE, N_FRAMES, L, rng, half=(12.0, 8.0, 12.0), noise_px=NOISE)
    held = rigs.wand_frames(truth, TUBE, N_HELD, L, rng, half=(12.0, 8.0, 12.0), noise_px=NOISE)
    data = cw.WandData(wand.uv, L, [(c.width, c.height) for c in truth])
    out = {}
    for name, fscale, fix in (("free", 1.0, ()), ("held", 1.0, (0, 1)), ("held +1 %", 1.01, (0, 1))):
        start = [c.copy() for c in truth]
        for c in start:
            c.f *= fscale
        t0 = time.time()
        sol = cw.calibrate(data, TUBE, start=start, fix_f=fix, n_final=1, verbose=False, log=lambda *a: None)
        sol = cw.covariance(sol)
        sig = cw.camera_sigmas(sol.cams, sol.layout, sol.cov)
        rec = {"seconds": time.time() - t0, "rms_px": float(sol.rms), "cameras": []}
        for c, (e, t, s) in enumerate(zip(sol.cams, truth, sig)):
            De, Dt = math.hypot(e.C[0], e.C[2]), math.hypot(t.C[0], t.C[2])
            cmp = rigs.compare([e], [t])[0]
            dC = e.C - t.C
            zC = [float(dC[k] / s["centre_mm"][k]) if s["centre_mm"][k] > 0 else float("nan") for k in range(3)]
            rec["cameras"].append({"f_rel": float(e.f / t.f - 1), "f_sigma_rel": s["f_rel"],
                                   "D_rel": float(De / Dt - 1), "D_sigma_mm": s["distance_mm"],
                                   "z_D": float((De - Dt) / s["distance_mm"]) if s["distance_mm"] > 0 else float("nan"),
                                   "rot_deg": cmp["rot_deg"], "centre_mm": cmp["centre_mm"], "z_centre": zC})
        rec["held_out_length_um"] = held_out_length(sol.cams, held)
        out[name] = rec
        c1, c2 = rec["cameras"]
        log(f"  seed {seed} {name:<10s}: RMS {rec['rms_px']:.4f} px; f {100 * c1['f_rel']:+.3f} / {100 * c2['f_rel']:+.3f} %"
            f" (sigma {100 * c1['f_sigma_rel']:.3f} / {100 * c2['f_sigma_rel']:.3f} %); D {100 * c1['D_rel']:+.3f} / "
            f"{100 * c2['D_rel']:+.3f} % (z {c1['z_D']:+.2f} / {c2['z_D']:+.2f}); centre |z| max "
            f"{max(abs(v) for c in rec['cameras'] for v in c['z_centre'] if np.isfinite(v)):.2f}; rotation "
            f"{c1['rot_deg']:.4f} / {c2['rot_deg']:.4f} deg; held-out wand length mean "
            f"{rec['held_out_length_um'][0]:+.2f} um, RMS {rec['held_out_length_um'][1]:.2f} um; {rec['seconds']:.0f} s")
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    log = rigs.Tee(HERE / "t14_hold_f_out.txt")
    log(f"T14: hold f, {n} seeds; rig: 2 cameras 1000 mm from the axis, 61 deg apart, f 38000 px, {N_FRAMES} frames, "
        f"noise {NOISE} px")
    res = {}
    for s in range(n):
        res[str(s)] = run(s, log)
    # summary
    for name in ("free", "held", "held +1 %"):
        f = np.array([[c["f_rel"] for c in res[s][name]["cameras"]] for s in res])
        D = np.array([[c["D_rel"] for c in res[s][name]["cameras"]] for s in res])
        zD = np.array([[c["z_D"] for c in res[s][name]["cameras"]] for s in res])
        zc = np.array([v for s in res for c in res[s][name]["cameras"] for v in c["z_centre"] if np.isfinite(v)])
        Lm = np.array([res[s][name]["held_out_length_um"][0] for s in res])
        log(f"  {name:<10s}: f error {100 * np.abs(f).max():.3f} % at most; D error mean {100 * D.mean():+.3f} %, "
            f"range {100 * D.min():+.3f} to {100 * D.max():+.3f} %; |z| of D at most {np.nanmax(np.abs(zD)):.2f}, of the "
            f"centre coordinates at most {np.max(np.abs(zc)):.2f}; held-out wand length mean {Lm.mean():+.2f} um "
            f"(range {Lm.min():+.2f} to {Lm.max():+.2f})")
    with open(HERE / "t14_hold_f.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
