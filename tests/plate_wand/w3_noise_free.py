"""W3 (noise-free part): the stage-4 solver from the truth on noise-free data, rigs A-E.

The cameras, windows and wand length are the truth; the frames are triangulated from the data; camera 1's pose (the
gauge) is the truth's.  Criteria: RMS <= 1e-8 px, and every parameter equal to the truth to 1e-6 relative (rotation
in rad, centres and window offsets in mm over the camera's distance to the cloud, f relative, the normal in rad).
Also from a start near the truth (f +-2 %, centres moved 5 mm, rotations 0.3 deg, normals 1 deg, offsets 3 mm), to see
the solver converge from off the minimum.  Output: w3_noise_free_out.txt / .json.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

pm = prigs.pm
import plate_wand as pw  # noqa: E402

log = prigs.Tee(HERE / "w3_noise_free_out.txt")
rng = np.random.default_rng(3)


def errors(sol, rig_t):
    scale = float(np.mean([np.linalg.norm(c.C - rig_t.centre) for c in rig_t.cams]))
    ce = prigs.compare_cams(sol.cams, rig_t.cams)
    we = prigs.compare_wins(sol.rig.windows, rig_t.wins)
    return {"rot_rad": max(math.radians(e["rot_deg"]) for e in ce),
            "centre_rel": max(e["centre_mm"] for e in ce) / scale,
            "f_rel": max(abs(e["f_rel"]) for e in ce),
            "normal_rad": max(math.radians(e["normal_deg"]) for e in we),
            "offset_rel": max(abs(e["d_mm"]) for e in we) / scale}


def perturbed(rig_t):
    cams = []
    for k, c in enumerate(rig_t.cams):
        c2 = c.copy()
        if k > 0:                               # camera 1's pose is the gauge
            a = rng.normal(size=3)
            c2.R = c.R @ pm.exp_so3(math.radians(0.3) * a / np.linalg.norm(a))
            t = rng.normal(size=3)
            c2.C = c.C + 5.0 * t / np.linalg.norm(t)
        c2.f = c.f * (1.0 + rng.choice([-0.02, 0.02]))
        cams.append(c2)
    wins = []
    for w in rig_t.wins:
        ab = rng.normal(size=2)
        n2 = pm.retract_normal(w.n, math.radians(1.0) * ab / np.linalg.norm(ab))
        wins.append(pm.Window(n2, w.d + 3.0, w.t, w.idx, w.name))
    return cams, wins


res = {}
ok_all = True
for name, fn in prigs.RIGS.items():
    rig_t = fn()
    wand = prigs.make_wand(rig_t, 1000, rng, noise_px=0.0)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    for start in ("truth", "near"):
        if start == "truth":
            cams0, wins0 = [c.copy() for c in rig_t.cams], [w.copy() for w in rig_t.wins]
        else:
            cams0, wins0 = perturbed(rig_t)
        t0 = time.time()
        sol = pw.solve(data, cams0, pw.Rig(wins0, rig_t.cam_win), noise_weights="none")
        dt = time.time() - t0
        e = errors(sol, rig_t)
        good = sol.rms <= 1e-8 and all(v <= 1e-6 for v in e.values())
        ok_all &= good
        res[f"{name}_{start}"] = {"rms_px": sol.rms, **e, "seconds": dt, "lm_iterations": sol.info["lm_iterations"],
                                  "active": int(sol.obs.active.sum()), "rows": len(sol.obs), "pass": bool(good)}
        log(f"rig {name}, start {start:5s}: rms {sol.rms:.2e} px; errors: rotation {e['rot_rad']:.1e} rad, centre "
            f"{e['centre_rel']:.1e} rel, f {e['f_rel']:.1e}, normal {e['normal_rad']:.1e} rad, offset "
            f"{e['offset_rel']:.1e} rel; {sol.info['lm_iterations']} LM iterations, {dt:.1f} s; active "
            f"{int(sol.obs.active.sum())}/{len(sol.obs)}" + ("" if good else "   <-- FAIL"))
res["pass"] = bool(ok_all)
log("W3 noise-free: " + ("PASS" if ok_all else "FAIL"))
(HERE / "w3_noise_free.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
sys.exit(0 if ok_all else 1)
