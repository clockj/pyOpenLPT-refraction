"""R2 (2026-10-02-perturbed-starts-plan.md): the perturbed restarts' draws on their own, no fitting.

  1. 2,000 draws (plate_wand.perturbed_draws, the default scale s = 2) about rig B's answer (the fit from the truth on
     1,000 frames at 0.1 px): their sample covariance against s^2 Sigma, each parameter's sigma within 5 %; each draw
     re-expressed as a step from the answer through the log maps (independently of Layout.apply) equals its dx;
     camera 1's pose untouched in every draw; the redraws.
  2. determinism: the seed from the same detections is the same, and so are the draws; one detection moved by 1e-9 px
     gives another seed and other draws.
  3. the redraw rule on W5 seed 2000's pinned answer (calibrate without restarts: a wand end on window 1's water-side
     surface): 200 draws; every draw kept has each camera on the air side of its window and each active observation
     projecting; how often a first try is kept (R1b: 9 of 20 at s = 1 and at s = 3).
Output: r2_draws_out.txt / .json
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
import r1_sizing as r1  # noqa: E402

pm = prigs.pm
import plate_wand as pw  # noqa: E402

N = 2000
SCALE = 2.0
log = prigs.Tee(HERE / "r2_draws_out.txt")
t_all = time.time()
out = {}

# ---------------------------------------------------------------- 1. the draws about rig B's answer
rig_t = prigs.rig_B()
wand = prigs.make_wand(rig_t, 1000, np.random.default_rng(777), noise_px=0.1)
data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
ans = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
lay = ans.layout
log(f"R2: draws about rig B's answer (fit from the truth: rms {ans.rms:.5f} px, {lay.P} free parameters), "
    f"{N} draws at s = {SCALE:g}")
t0 = time.time()
rng = np.random.default_rng(pw._perturb_seed(data))
draws, left_out = pw.perturbed_draws(ans, ans.obs, N, SCALE, rng)
secs = time.time() - t0
dx = np.array([d["dx"] for d in draws])
sd_t = SCALE * np.sqrt(np.diag(ans.cov))
sd_s = dx.std(axis=0, ddof=1)
ratio = sd_s / sd_t
corr_t = ans.cov / np.outer(np.sqrt(np.diag(ans.cov)), np.sqrt(np.diag(ans.cov)))
corr_s = np.corrcoef(dx.T)
worst = int(np.argmax(np.abs(ratio - 1.0)))
ok_sd = bool(np.all(np.abs(ratio - 1.0) <= 0.05))
log(f"  {len(draws)} draws kept, {left_out} left out, {sum(d['redraws'] for d in draws)} redraws, {secs:.0f} s "
    f"({1e3 * secs / N:.0f} ms a draw)")
log(f"  sample sigma / (s sigma): {ratio.min():.3f} to {ratio.max():.3f} (worst {lay.names[worst]}); "
    f"the sampling error of a sigma from {N} draws is {100 / math.sqrt(2 * (N - 1)):.1f} %; within 5 %: "
    f"{'yes' if ok_sd else 'NO'}")
log(f"  correlations: largest difference from the covariance's {np.max(np.abs(corr_s - corr_t)):.3f} "
    f"(sampling error about {1 / math.sqrt(N):.3f})")


def step_of(sol, cams, rig):
    """cams, rig (in sol's frame) as a step from sol through the log maps, as r1_sizing.step_to without the
    alignment."""
    dxx = np.zeros(sol.layout.P)
    for c, (cs, co) in enumerate(zip(sol.cams, cams)):
        vals = list(pm.log_so3(cs.R.T @ co.R)) + list(co.C - cs.C) + [math.log(co.f / cs.f)]
        for j in range(7):
            g = sol.layout.pos[c, j]
            if g >= 0:
                dxx[g] = vals[j]
    for w, (ws, wo) in enumerate(zip(sol.rig.windows, rig.windows)):
        a, b = prigs.sphere_log(ws.n, wo.n)
        A = sol.layout.anchors[w]
        dl = (wo.d - float(wo.n @ A)) - (ws.d - float(ws.n @ A))
        for j, v in enumerate((a, b, dl)):
            dxx[sol.layout.wpos[w, j]] = v
    return dxx


rt = max(float(np.max(np.abs(step_of(ans, d["cams"], d["rig"]) - d["dx"]))) for d in draws[:200])
cam1 = all(np.array_equal(d["cams"][0].R, ans.cams[0].R) and np.array_equal(d["cams"][0].C, ans.cams[0].C)
           for d in draws)
ok_rt = rt < 1e-9
log(f"  each draw re-expressed through the log maps equals its dx to {rt:.1e} (first 200; tolerance 1e-9): "
    f"{'yes' if ok_rt else 'NO'}; camera 1's pose untouched in every draw: {'yes' if cam1 else 'NO'}")
out["rig_B"] = {"kept": len(draws), "left_out": left_out, "redraws": sum(d["redraws"] for d in draws),
                "sigma_ratio_min": float(ratio.min()), "sigma_ratio_max": float(ratio.max()),
                "corr_max_diff": float(np.max(np.abs(corr_s - corr_t))), "roundtrip": rt, "camera1_untouched": cam1,
                "ms_per_draw": 1e3 * secs / N}

# ---------------------------------------------------------------- 2. determinism
s1, s2 = pw._perturb_seed(data), pw._perturb_seed(pw.WandData(wand.uv.copy(), wand.L, rig_t.sizes))
d_a, _ = pw.perturbed_draws(ans, ans.obs, 20, SCALE, np.random.default_rng(s1))
d_b, _ = pw.perturbed_draws(ans, ans.obs, 20, SCALE, np.random.default_rng(s2))
same = s1 == s2 and all(np.array_equal(a["dx"], b["dx"]) for a, b in zip(d_a, d_b))
uv2 = wand.uv.copy()
f0, c0 = np.argwhere(np.isfinite(uv2[:, :, 0, 0]))[0]
uv2[f0, c0, 0, 0] += 1e-9
s3 = pw._perturb_seed(pw.WandData(uv2, wand.L, rig_t.sizes))
d_c, _ = pw.perturbed_draws(ans, ans.obs, 20, SCALE, np.random.default_rng(s3))
differ = s3 != s1 and not any(np.array_equal(a["dx"], c["dx"]) for a, c in zip(d_a, d_c))
log(f"  determinism: the same detections give seed {s1} twice and the same 20 draws: {'yes' if same else 'NO'}; one "
    f"detection moved by 1e-9 px gives seed {s3} and other draws: {'yes' if differ else 'NO'}")
out["determinism"] = {"seed": s1, "same": same, "seed_moved": s3, "differ": differ}

# ---------------------------------------------------------------- 3. the redraw rule on seed 2000's pinned answer
rig_2, data_2 = r1.seed_2000()
placeholder = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_2.wins], rig_2.cam_win)
t0 = time.time()
pin = pw.calibrate(data_2, placeholder, output_frame=False, perturb_starts=0)
dep = pw.window_depths(pin.cams, pin.rig, pin.frames, pin.L, pin.obs)
log(f"seed 2000 without restarts (perturb_starts=0): rms {pin.rms:.6f} px (R1: 1.158848), its wand ends' smallest "
    f"depth per window " + ", ".join(f"{v:.2e}" for v in dep) + f" mm; {time.time() - t0:.0f} s")
o_p = pw.Obs(pin.obs.frame, pin.obs.cam, pin.obs.end, pin.obs.uv, pin.obs.n_frames, pin.obs.n_cams,
             pin.obs.frame_ids, pin.obs.active.copy())
t0 = time.time()
dd, lo = pw.perturbed_draws(pin, o_p, 200, SCALE, np.random.default_rng(pw._perturb_seed(data_2)))
rows = np.flatnonzero(o_p.active)
valid = all(all(d["rig"].windows[w].distance(d["cams"][c].C) > 0 for c, w in enumerate(d["rig"].cam_win) if w >= 0)
            and pw.evaluate(d["cams"], d["rig"], d["frames"], d["L"], o_p, rows=rows).ok.all() for d in dd)
first = sum(1 for d in dd if d["redraws"] == 0)
n_tries = sum(d["redraws"] + 1 for d in dd) + 11 * lo
log(f"  200 draws at s = {SCALE:g}: {len(dd)} kept, {lo} left out; every kept draw has its cameras on the air side "
    f"and every active observation projecting: {'yes' if valid else 'NO'}; a try is kept {len(dd)}/{n_tries} = "
    f"{len(dd) / n_tries:.2f} of the time ({first} draws on their first try); {time.time() - t0:.0f} s")
out["seed2000"] = {"rms_pinned": pin.rms, "min_depth_mm": dep, "kept": len(dd), "left_out": lo, "valid": valid,
                   "first_try": first, "tries": n_tries}

ok = ok_sd and ok_rt and cam1 and same and differ and valid
log(f"R2: {'PASS' if ok else 'FAIL'}; {time.time() - t_all:.0f} s")
out["pass"] = ok
(HERE / "r2_draws.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
