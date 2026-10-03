"""D1 (diagnosis): why W5's multi-window rigs 1031, 1032, 1039, 1041 reached a wrong minimum.

For each seed: the rig (cameras per window, incidences), the pinhole start, each window's stage-3 candidates (normal
error against the truth in degrees, offset error in mm, score), the coarse fits of the combinations, and stage 4 of
each start (RMS, active observations), against the truth.  Output: d1_multiwindow_out.txt.
Usage: d1_multiwindow.py [seeds...]
"""
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

log = prigs.Tee(HERE / "d1_multiwindow_out.txt")
seeds = [int(s) for s in sys.argv[1:]] or [1031, 1032, 1039, 1041]
for seed in seeds:
    rng = np.random.default_rng(seed)
    rig_t = prigs.random_rig(rng)
    n_frames = int(rng.integers(500, 1501))
    wand = prigs.make_wand(rig_t, n_frames, rng, noise_px=0.1)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    obs = pw.observations(data)
    rig = pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx) for w in rig_t.wins], rig_t.cam_win)
    inc = [math.degrees(math.acos(min(1.0, float(c.R[2] @ rig_t.win_of(k).n)))) for k, c in enumerate(rig_t.cams)]
    log(f"seed {seed}: {rig_t.n_cams} cams, windows {rig_t.cam_win}, incidence {[round(x) for x in inc]} deg, f "
        f"{[round(c.f) for c in rig_t.cams]}, h1 {[round(rig_t.win_of(k).distance(c.C)) for k, c in enumerate(rig_t.cams)]}"
        f" mm, cloud {rig_t.info['cloud_mm']:.0f} mm, {len(wand.uv)} frames")
    t0 = time.time()
    pins, pinfo = pw.pinhole_start(obs, data.L, data.sizes, [((w - 1) / 2, (h - 1) / 2) for w, h in data.sizes], rig)
    log(f"  pinhole start: {len(pins)} kept, votes {pinfo['mirror_votes']}, rms {[round(p.rms, 3) for p in pins]}, "
        f"{time.time() - t0:.1f} s")
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    for sol_p in pins:
        # the truth in the pinhole solution's frame (by camera 1), to judge the candidates
        Q, p = prigs.rigid_from_cam(rig_t.cams[0], sol_p.cams[0])
        cams_tt, wins_tt = prigs.transform(rig_t.cams, rig_t.wins, Q, p)
        for w in range(len(rig.windows)):
            t1 = time.time()
            best, inf = pw.window_search(sol_p, rig, w)
            msg = []
            for s, win, cs in best:
                ang = math.degrees(math.acos(min(1.0, float(win.n @ wins_tt[w].n))))
                msg.append(f"[{ang:.1f} deg, d {win.d - wins_tt[w].d:+.1f} mm, score {s:.4g}]")
            log(f"  window {w + 1} (cameras {rig.cams_of(w)}): guess {math.degrees(math.acos(min(1, float(np.array(inf['guess']) @ wins_tt[w].n)))):.1f} deg "
                f"from the truth; best: {' '.join(msg)}; {time.time() - t1:.1f} s")
    t2 = time.time()
    starts, binfo = pw.bootstrap(obs, rig, data.L, data.sizes)
    log(f"  bootstrap {time.time() - t2:.1f} s: {len(starts)} starts, coarse rms {[round(s[1], 3) for s in starts]}")
    for k, (cost, rms3, cams0, rig0) in enumerate(starts[:3]):
        o = pw.Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, obs.active.copy())
        sol = pw.solve(data, cams0, rig0, obs=o)
        cmp = prigs.same_minimum(sol, ref)
        act_c = [int(np.sum(o.active & (o.cam == c))) for c in range(rig_t.n_cams)]
        log(f"  start {k}: coarse rms {rms3:.3f} -> stage 4 rms {sol.rms:.5f} (truth start {ref.rms:.5f}), active per "
            f"camera {act_c}, max |z| {cmp['max_abs_z']:.2e} ({cmp['worst']}), likelihood key "
            f"{pw._log_likelihood_key(sol, 'camera'):.1f}")
