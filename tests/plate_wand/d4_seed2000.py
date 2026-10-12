"""D4 (diagnosis, 2026-10-02, a question): why W5 seed 2000 misses the global minimum although each of its two
windows has two cameras.

Seed 2000: 1 px noise, 769 frames, cameras 1-4 behind windows [1, 2, 1, 2].  W5 run 1 (the whole-rig start) reached
the global minimum; the window-group start (`bootstrap_groups`, from run 2 on) ends 0.3 % above it in RMS and 3.9
sigma away (camera 3's centre).  Steps:
  1. the rig, and the angle between each window's two cameras' lines of sight;
  2. the fit from the truth (the global minimum): each camera's and window's sigma at 1 px;
  3. each window's two cameras calibrated on their own (the group start's first step): their errors against the
     truth, in mm and degrees and in the sigma that the group's own data allow (Cramer-Rao on its observations);
  4. calibrate() as in W5: its starts, and its answer against the global minimum (the parameters that differ most);
  5. is that answer a separate minimum?  Stage 4 restarted from it, and from the points 1/4, 1/2 and 3/4 of the way
     to the global minimum;
  6. the whole-rig start (`bootstrap`, W5 run 1's path) on today's code: each of its starts' stage-4 answer.
Output: d4_seed2000_out.txt
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

SEED = 2000
log = prigs.Tee(HERE / "d4_seed2000_out.txt")
t_all = time.time()
rng = np.random.default_rng(SEED)                    # exactly as w5_from_scratch.run_one
rig_t = prigs.random_rig(rng)
n_frames = int(rng.integers(500, 1501))
wand = prigs.make_wand(rig_t, n_frames, rng, noise_px=1.0)
data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
obs = pw.observations(data)
placeholder = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins], rig_t.cam_win)
true_rig = pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win)
cxcy = [((w - 1) / 2.0, (h - 1) / 2.0) for (w, h) in data.sizes]


def ang(a, b):
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))


def refract(u, n, n1, n2):
    """Vector Snell's law: direction u (unit) crossing a surface with normal n (along the travel) from n1 into n2."""
    c = float(u @ n)
    r = n1 / n2
    return r * (u - c * n) + math.sqrt(1.0 - r * r * (1.0 - c * c)) * n


def z_all(sol, ref):
    """Every free parameter's difference from ref, in ref's sigma, after aligning sol's camera 1 on ref's (as
    prigs.param_z, but all of them): a list of (|z|, name), largest first."""
    Q, p = prigs.rigid_from_cam(sol.cams[0], ref.cams[0])
    cams_a, wins_a = prigs.transform(sol.cams, sol.rig.windows, Q, p)
    lay, cov = ref.layout, ref.cov
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    out = []
    for c, (cs, cr) in enumerate(zip(cams_a, ref.cams)):
        dw = pm.log_so3(cr.R.T @ cs.R)
        vals = {"w1": dw[0], "w2": dw[1], "w3": dw[2], "C1": cs.C[0] - cr.C[0], "C2": cs.C[1] - cr.C[1],
                "C3": cs.C[2] - cr.C[2], "logf": math.log(cs.f / cr.f), "k1": cs.dist[0] - cr.dist[0]}
        for j, nm in enumerate(lay.cols):
            g = lay.pos[c, j]
            if g >= 0:
                out.append((abs(vals[nm]) / sd[g], f"cam{c + 1}.{nm}", vals[nm], sd[g]))
    for w, (ws, wr) in enumerate(zip(wins_a, ref.rig.windows)):
        a, b = prigs.sphere_log(wr.n, ws.n)
        A = lay.anchors[w]
        dl = (ws.d - float(ws.n @ A)) - (wr.d - float(wr.n @ A))
        for j, v in enumerate((a, b, dl)):
            g = lay.wpos[w, j]
            out.append((abs(v) / sd[g], f"win{w + 1}.{'ab'[j] if j < 2 else 'delta'}", v, sd[g]))
    return sorted(out, reverse=True), cams_a, wins_a


def toward(cams_a, wins_a, ref, t):
    """The point a fraction t of the way from (cams_a, wins_a) to ref's cameras and windows."""
    cams = []
    for ca, cr in zip(cams_a, ref.cams):
        k = ca.copy()
        k.R = ca.R @ pm.exp_so3(t * pm.log_so3(ca.R.T @ cr.R))
        k.C = (1 - t) * ca.C + t * cr.C
        k.f = math.exp((1 - t) * math.log(ca.f) + t * math.log(cr.f))
        cams.append(k)
    wins = []
    for wa, wr in zip(wins_a, ref.rig.windows):
        th = ang(wa.n, wr.n)
        if th < 1e-12:
            n = wr.n.copy()
        else:
            ax = np.cross(wa.n, wr.n)
            ax /= np.linalg.norm(ax)
            n = pm.exp_so3(t * math.radians(th) * ax) @ wa.n
        wins.append(pm.Window(n, (1 - t) * wa.d + t * wr.d, wa.t, wa.idx, wa.name))
    return cams, pw.Rig(wins, ref.rig.cam_win)


# ---------------------------------------------------------------- 1. the rig
log(f"seed {SEED}: {rig_t.n_cams} cameras behind windows {[w + 1 for w in rig_t.cam_win]}, {len(data.uv)} frames, "
    f"1 px noise; cloud {rig_t.info['cloud_mm']:.0f} mm, wand {rig_t.L:.1f} mm, plates {rig_t.wins[0].t} mm")
for k, c in enumerate(rig_t.cams):
    w = rig_t.win_of(k)
    log(f"  cam{k + 1}: window {rig_t.cam_win[k] + 1} ({w.name}), incidence {ang(c.R[2], w.n):.1f} deg, "
        f"{w.distance(c.C):.0f} mm from it, f {c.f:.0f} px, sensor {c.width} x {c.height}, half-width field "
        f"{math.degrees(math.atan(c.width / 2 / c.f)):.1f} deg")
for w in range(len(rig_t.wins)):
    cs = rig_t.cams_of(w) if hasattr(rig_t, "cams_of") else [k for k, x in enumerate(rig_t.cam_win) if x == w]
    a, b = rig_t.cams[cs[0]], rig_t.cams[cs[1]]
    win = rig_t.wins[w]
    wa = refract(a.R[2], win.n, 1.0, win.idx[-1])
    wb = refract(b.R[2], win.n, 1.0, win.idx[-1])
    log(f"  window {w + 1}: cameras {cs[0] + 1} and {cs[1] + 1}, their lines of sight {ang(a.R[2], b.R[2]):.1f} deg "
        f"apart in air, {ang(wa, wb):.1f} deg in the water")
c1, c2 = rig_t.cams[0], rig_t.cams[1]
log(f"  the two windows' normals are {ang(rig_t.wins[0].n, rig_t.wins[1].n):.0f} deg apart")

# ---------------------------------------------------------------- 2. the global minimum and its sigmas
t0 = time.time()
ref = pw.solve(data, [c.copy() for c in rig_t.cams], true_rig.copy())
log(f"fit from the truth (the global minimum): rms {ref.rms:.5f} px, active {int(ref.obs.active.sum())}/"
    f"{len(ref.obs)}, {time.time() - t0:.0f} s")
cs_ref = pw.camera_sigmas(ref.cams, ref.rig, ref.layout, ref.cov)
ws_ref = pw.window_sigmas(ref.cams, ref.rig, ref.layout, ref.cov)
for k, s in enumerate(cs_ref):
    log(f"  sigma cam{k + 1}: f {100 * s['f_rel']:.2f} %, distance to its window {s['distance_mm']:.1f} mm, centre "
        f"{max(s['centre_mm']):.1f} mm, rotation {max(s['rotation_deg']):.3f} deg")
for k, s in enumerate(ws_ref):
    log(f"  sigma window {k + 1}: normal {s['normal_deg']:.3f} deg, offset {s['offset_mm']:.1f} mm")

# ---------------------------------------------------------------- 3. each window's pair on its own
log("each window's two cameras calibrated on their own (bootstrap_groups' first step, search 'length'):")
for w in range(len(placeholder.windows)):
    cs = placeholder.cams_of(w)
    o_w = pw._restrict_obs(obs, cs)
    rig_w = pw.Rig([placeholder.windows[w].copy()], [0] * len(cs))
    t0 = time.time()
    st, _inf = pw.bootstrap(o_w, rig_w, data.L, [data.sizes[c] for c in cs], cxcy=[cxcy[c] for c in cs],
                            search="length")
    best = None
    for (_cost, rms3, c0, r0) in st[:2]:
        o = pw.Obs(o_w.frame, o_w.cam, o_w.end, o_w.uv, o_w.n_frames, o_w.n_cams, o_w.frame_ids, o_w.active.copy())
        sol_w = pw.solve_obs(o, data.L, c0, r0, noise_weights="none")
        if best is None or sol_w.rms < best.rms:
            best = sol_w
    # the group's own Cramer-Rao bound, at the truth, on its data alone (its first camera's pose held)
    cams_tw = [rig_t.cams[c].copy() for c in cs]
    rig_tw = pw.Rig([rig_t.wins[w].copy()], [0] * len(cs))
    data_w = pw.WandData(wand.uv[:, cs], wand.L, [data.sizes[c] for c in cs])
    obs_w = pw.observations(data_w)
    lay_w = pw.Layout(len(cs), 1, anchors=pw.anchors_for(cams_tw, rig_tw))
    fr_w = pw.frames_from_ends(wand.XA[obs_w.frame_ids], wand.XB[obs_w.frame_ids])
    S_w, _ = pw.information(cams_tw, rig_tw, fr_w, wand.L, obs_w, lay_w, sigma_px=1.0)
    cov_w = np.linalg.inv(S_w)
    sc_w = pw.camera_sigmas(cams_tw, rig_tw, lay_w, cov_w)
    sw_w = pw.window_sigmas(cams_tw, rig_tw, lay_w, cov_w)
    Q, p = prigs.rigid_from_cam(best.cams[0], rig_t.cams[cs[0]])
    ca, wa = prigs.transform(best.cams, best.rig.windows, Q, p)
    e_n = ang(wa[0].n, rig_t.wins[w].n)
    d_err = [ca[k].C - rig_t.cams[c].C for k, c in enumerate(cs)]
    f_err = [ca[k].f / rig_t.cams[c].f - 1.0 for k, c in enumerate(cs)]
    h_err = [wa[0].distance(ca[k].C) - rig_t.wins[w].distance(rig_t.cams[c].C) for k, c in enumerate(cs)]
    log(f"  window {w + 1}, cameras {[c + 1 for c in cs]}: rms {best.rms:.4f} px, {time.time() - t0:.0f} s")
    log(f"    window normal off by {e_n:.2f} deg (the pair's own sigma {sw_w[0]['normal_deg']:.2f} deg; in the full "
        f"rig {ws_ref[w]['normal_deg']:.2f} deg)")
    for k, c in enumerate(cs):
        log(f"    cam{c + 1}: f off by {100 * f_err[k]:+.2f} % (pair's sigma {100 * sc_w[k]['f_rel']:.2f} %, full rig "
            f"{100 * cs_ref[c]['f_rel']:.2f} %); distance to the window off by {h_err[k]:+.1f} mm (pair's sigma "
            f"{sc_w[k]['distance_mm']:.1f} mm, full rig {cs_ref[c]['distance_mm']:.1f} mm); centre off by "
            f"{np.linalg.norm(d_err[k]):.1f} mm")

# ---------------------------------------------------------------- 4. calibrate, as in W5
t0 = time.time()
sol = pw.calibrate(data, placeholder)
log(f"calibrate (as W5): rms {sol.rms:.5f} px against {ref.rms:.5f}, {time.time() - t0:.0f} s; starts "
    f"{sol.info['starts']}; bootstrap methods: "
    f"{ {k: v.get('method') for k, v in sol.info['bootstrap'].items() if isinstance(v, dict) and 'method' in v} }")
n_act = int(sol.obs.active.sum())
dchi2 = (sol.rms ** 2 - ref.rms ** 2) * n_act
log(f"  the same {n_act} observations; their squared residuals sum to {dchi2:.0f} px^2 more than at the global minimum")
zs, cams_a, wins_a = z_all(sol, ref)
log("  the parameters that differ most from the global minimum (camera 1's pose aligned):")
for z, nm, v, sd in zs[:8]:
    unit = "mm" if ".C" in nm or nm.endswith("delta") else "rad"
    if nm.endswith("logf"):
        log(f"    {nm:12s} {z:5.2f} sigma ({100 * v:+.2f} % in f; sigma {100 * sd:.2f} %)")
    else:
        log(f"    {nm:12s} {z:5.2f} sigma ({v:+.4g} {unit}; sigma {sd:.3g} {unit})")
hist = sol.history
if hist:
    log(f"  its last LM round: {len(hist)} iterations, the last {'accepted' if hist[-1]['accepted'] else 'stuck'}, "
        f"rms {hist[-1]['rms_px']:.6f} px")

# ---------------------------------------------------------------- 5. a separate minimum?
log("stage 4 restarted from points between that answer (t = 0) and the global minimum (t = 1):")
for t in (0.0, 0.25, 0.5, 0.75):
    cams_t, rig_tt = toward(cams_a, wins_a, ref, t)
    t0 = time.time()
    s_t = pw.solve(data, cams_t, rig_tt)
    same = prigs.same_minimum(s_t, ref)
    log(f"  t = {t:.2f}: rms {s_t.rms:.5f} px -> {'the global minimum' if same['same'] else 'NOT the global minimum'}"
        f" (max |z| {same['max_abs_z']:.2g}, {same['worst']}), {time.time() - t0:.0f} s")

# ---------------------------------------------------------------- 6. the whole-rig start
log("the whole-rig start (bootstrap on all four cameras, W5 run 1's path), each start through stage 4:")
for search in ("length", "fixed"):
    t0 = time.time()
    st, inf = pw.bootstrap(obs, placeholder, data.L, data.sizes, search=search)
    log(f"  search '{search}': {len(st)} starts, {time.time() - t0:.0f} s")
    for k, (_cost, rms3, c0, r0) in enumerate(st[:3]):
        s_k = pw.solve(data, c0, r0)                  # fresh observations, as calibrate's stage 4
        same = prigs.same_minimum(s_k, ref)
        log(f"    start {k}: coarse rms {rms3:.4f} px -> stage 4 rms {s_k.rms:.5f} px: "
            f"{'the global minimum' if same['same'] else 'NOT the global minimum'} (max |z| {same['max_abs_z']:.2g})")
log(f"done, {time.time() - t_all:.0f} s")
