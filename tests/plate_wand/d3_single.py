"""D3 (diagnosis): one camera alone behind its window, placed against known points (plate_wand.place_single_window).

Seed 1039's camera 3 (43 deg incidence, f 5,220 px) and others: the wand ends are the truth's (exact), the other
cameras the truth's.  Reported: the best candidates' normal error, camera error and score, and the score of the true
camera and window resected the same way, to tell a search miss from an ambiguity.  Also the resection cost along
a line from the truth to the chosen candidate.  Output: d3_single_out.txt.  Usage: d3_single.py SEED CAM
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

seed, cam = int(sys.argv[1]), int(sys.argv[2])
log = prigs.Tee(HERE / f"d3_single_{seed}_cam{cam}_out.txt")
rng = np.random.default_rng(seed)
rig_t = prigs.random_rig(rng)
n_frames = int(rng.integers(500, 1501))
wand = prigs.make_wand(rig_t, n_frames, rng, noise_px=0.1)
data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
obs = pw.observations(data)
w = rig_t.cam_win[cam]
c_t = rig_t.cams[cam]
inc = math.degrees(math.acos(float(c_t.R[2] @ rig_t.wins[w].n)))
log(f"seed {seed} camera {cam + 1}: window {w + 1}, incidence {inc:.1f} deg, f {c_t.f:.0f}, h1 "
    f"{rig_t.wins[w].distance(c_t.C):.0f} mm, sensor {c_t.width}x{c_t.height}")
XA, XB = wand.XA[obs.frame_ids], wand.XB[obs.frame_ids]
rig = pw.Rig([x.copy() for x in rig_t.wins], rig_t.cam_win)
cams = [x.copy() for x in rig_t.cams]
cxcy = ((c_t.width - 1) / 2.0, (c_t.height - 1) / 2.0)
cands, inf = pw.place_single_window(cam, w, cams, rig, obs, data.L, XA, XB, cxcy, (c_t.width, c_t.height),
                                    tri_cams=[c for c in range(rig_t.n_cams) if c != cam])


def ang(a, b):
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


for s, win, cd in cands:
    cc = cd[cam]
    log(f"  candidate: score {s:.4g}; normal error {ang(win.n, rig_t.wins[w].n):.2f} deg, camera rotation "
        f"{pm.rotation_angle_deg(cc.R, c_t.R):.2f} deg, centre {np.linalg.norm(cc.C - c_t.C):.1f} mm, f "
        f"{100 * (cc.f / c_t.f - 1):+.2f} %, h1 {win.distance(cc.C):.0f} mm")
# the truth resected the same way (full phase: 200 frames, 20 iterations), and the line truth -> best candidate
rows = pw._subset_rows(obs, 200)
rr = rows[obs.cam[rows] == cam]
Xr = np.where((obs.end[rr] == 0)[:, None], XA[obs.frame[rr]], XB[obs.frame[rr]])
cs, cost = pw.resect([c_t], [rig_t.wins[w]], [Xr], [obs.uv[rr]], iters=20)
log(f"  the truth resected: score {cost[0]:.4g} (at the truth itself, before resection: "
    f"{pw.resect([c_t], [rig_t.wins[w]], [Xr], [obs.uv[rr]], iters=0)[1][0]:.4g})")
s_b, win_b, cd_b = cands[0]
for t in (0.0, 0.25, 0.5, 0.75, 1.0):
    # interpolate the window normal and offset, then resect the camera from the truth
    n = (1 - t) * rig_t.wins[w].n + t * win_b.n
    n /= np.linalg.norm(n)
    A = c_t.C + rig_t.wins[w].distance(c_t.C) * rig_t.wins[w].n
    d = (1 - t) * rig_t.wins[w].d + t * win_b.d
    win_t = pm.Window(n, d, rig_t.wins[w].t, rig_t.wins[w].idx)
    c0 = pw.small_angle_camera(pw.solve_obs.__globals__["_camera_from_dlt"](Xr, obs.uv[rr], cxcy, (c_t.width, c_t.height)), win_t) if t > 0 else c_t
    csx, costx = pw.resect([c0], [win_t], [Xr], [obs.uv[rr]], iters=40)
    log(f"  t = {t:.2f}: normal {ang(n, rig_t.wins[w].n):.1f} deg from the truth -> resected score {costx[0]:.4g}")
