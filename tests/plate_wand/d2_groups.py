"""D2 (diagnosis): the window-group start (plate_wand.bootstrap_groups) step by step on one W5 seed, against the truth.

  1. each group of cameras behind one window, calibrated on its own: its RMS, and its cameras and window against the
     truth after aligning the group's first camera;
  2. the merge: the rigid motion's residual on the common wand positions;
  3. each single camera's window from place_single_window against the truth (merged frame aligned to the truth by
     camera 1).
Output: d2_groups_<seed>_out.txt.  Usage: d2_groups.py SEED
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

pm = prigs.pm
import plate_wand as pw  # noqa: E402

seed = int(sys.argv[1])
log = prigs.Tee(HERE / f"d2_groups_{seed}_out.txt")
rng = np.random.default_rng(seed)
rig_t = prigs.random_rig(rng)
n_frames = int(rng.integers(500, 1501))
wand = prigs.make_wand(rig_t, n_frames, rng, noise_px=1.0 if seed >= 2000 else 0.1)
data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
obs = pw.observations(data)
rig = pw.Rig([pm.Window([0, 0, 1], 0.0, w.t, w.idx) for w in rig_t.wins], rig_t.cam_win)
inc = [round(math.degrees(math.acos(min(1.0, float(c.R[2] @ rig_t.win_of(k).n))))) for k, c in enumerate(rig_t.cams)]
log(f"seed {seed}: windows {rig_t.cam_win}, incidence {inc}, h1 "
    f"{[round(rig_t.win_of(k).distance(c.C)) for k, c in enumerate(rig_t.cams)]}, f {[round(c.f) for c in rig_t.cams]}")
cxcy = [((w - 1) / 2.0, (h - 1) / 2.0) for (w, h) in data.sizes]


def ang(a, b):
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


for w in range(len(rig.windows)):
    cs = rig.cams_of(w)
    if len(cs) < 2:
        continue
    o_w = pw._restrict_obs(obs, cs)
    rig_w = pw.Rig([rig.windows[w].copy()], [0] * len(cs))
    st, inf = pw.bootstrap(o_w, rig_w, data.L, [data.sizes[c] for c in cs], cxcy=[cxcy[c] for c in cs])
    for k, (cost, rms3, c0, r0) in enumerate(st[:3]):
        o = pw.Obs(o_w.frame, o_w.cam, o_w.end, o_w.uv, o_w.n_frames, o_w.n_cams, o_w.frame_ids, o_w.active.copy())
        sol = pw.solve_obs(o, data.L, c0, r0, noise_weights="none")
        Q, p = prigs.rigid_from_cam(sol.cams[0], rig_t.cams[cs[0]])
        ca, wa = prigs.transform(sol.cams, sol.rig.windows, Q, p)
        ce = prigs.compare_cams(ca, [rig_t.cams[c] for c in cs])
        log(f"  group window {w + 1} cams {cs}: start {k} coarse {rms3:.3f} -> rms {sol.rms:.4f}, active "
            f"{int(o.active.sum())}/{len(o)}; normal err {ang(wa[0].n, rig_t.wins[w].n):.3f} deg, d err "
            f"{wa[0].d - rig_t.wins[w].d:+.2f} mm; cams: rot {max(e['rot_deg'] for e in ce):.3f} deg, centre "
            f"{max(e['centre_mm'] for e in ce):.2f} mm, f {max(abs(e['f_rel']) for e in ce) * 100:.2f} %")
log("whole bootstrap_groups:")
starts, binfo = pw.bootstrap_groups(obs, rig, data.L, data.sizes)
log(f"  info: {binfo['groups']}")
for k, (cost, rms3, c0, r0) in enumerate(starts[:3]):
    Q, p = prigs.rigid_from_cam(c0[0], rig_t.cams[0])
    ca, wa = prigs.transform(c0, r0.windows, Q, p)
    ce = prigs.compare_cams(ca, rig_t.cams)
    log(f"  start {k}: back-projection rms {rms3:.3f}; normal errors {[round(ang(a.n, b.n), 3) for a, b in zip(wa, rig_t.wins)]} deg; "
        f"d err {[round(a.d - b.d, 2) for a, b in zip(wa, rig_t.wins)]}; cam rot err {[round(e['rot_deg'], 3) for e in ce]}, "
        f"centre {[round(e['centre_mm'], 1) for e in ce]}, f % {[round(100 * e['f_rel'], 2) for e in ce]}")
