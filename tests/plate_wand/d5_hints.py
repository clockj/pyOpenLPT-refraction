"""D5 (diagnosis, H1a's stop point): where a window-normal hint is lost on the cases H1a did not fix.

For one case and one set of hints (as h1_hints.py: SEED NOISE DELTA DRAW), `calibrate`'s start generation is replayed
step by step (the same searches, the same pooling of their starts):
  1. the hints in the search's frame: each camera's hint turned into its pinhole start's frame (`hint_to_frame`), and
     the default guess (the camera's viewing direction), compared with the true normal after aligning the start's wand
     points with the truth's (Kabsch) -- for the whole rig's pinhole start and for each window group's own;
  2. every start the searches produce (not only those calibrate fits): its windows' normal errors and cameras' errors
     against the truth (aligned by camera 1), its coarse cost and RMS, and where stage 4 takes it (`solve`: its RMS,
     the same minimum as the truth-started fit or not, the windows' normal errors);
  3. which of them calibrate would have fitted (n_final 2, more when they disagree, up to 6), and what it chose.
The bootstrap's own log (candidates, placements) is in the output too.
Output: d5_hints_SEED_NOISE_DELTA_DRAW_out.txt.  Usage: d5_hints.py SEED NOISE DELTA DRAW   (DELTA "none": no hints)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402
from h1_hints import make_hints  # noqa: E402

pm = prigs.pm
import plate_wand as pw  # noqa: E402

seed, noise = int(sys.argv[1]), float(sys.argv[2])
delta = None if sys.argv[3] == "none" else float(sys.argv[3])
draw = int(sys.argv[4]) if len(sys.argv) > 4 else 0
tag = f"{seed}_{noise:g}_{'none' if delta is None else f'{delta:g}'}_{draw}"
log = prigs.Tee(HERE / f"d5_hints_{tag}_out.txt")
_rng = np.random.default_rng(seed)                     # the data as in W5 / R4 / H1a
rig_t = prigs.random_rig(_rng)
wand = prigs.make_wand(rig_t, int(_rng.integers(500, 1501)), _rng, noise_px=noise)
data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
hints = None if delta is None else make_hints(rig_t, delta, np.random.default_rng([seed, int(delta), draw]))
obs = pw.observations(data)
rig = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins], rig_t.cam_win)
ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
log(f"D5 seed {seed} ({noise:g} px), hints {'none' if delta is None else f'off by {delta:g} deg, draw {draw}'}: "
    f"{rig_t.n_cams} cameras, windows {rig_t.cam_win}, f {', '.join(f'{c.f:.0f}' for c in rig_t.cams)} px, incidence "
    f"{', '.join(f'{math.degrees(math.acos(float(c.R[2] @ rig_t.win_of(k).n))):.0f}' for k, c in enumerate(rig_t.cams))}"
    f" deg; truth-started fit rms {ref.rms:.5f} px")


def ang(a, b):
    return math.degrees(math.atan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b))))




def align_rotation(sol_p, cams_idx):
    """Rotation taking the pinhole start's frame to the truth's: Kabsch on the active frames' ends."""
    o = sol_p.obs
    XA, XB = sol_p.frames.ends(sol_p.L)
    act = np.zeros(o.n_frames, bool)
    act[o.frame[o.active]] = True
    P = np.vstack([XA[act], XB[act]])
    fid = o.frame_ids[act]                             # the data's frame numbers (subsets renumber their frames)
    Q = np.vstack([wand.XA[fid], wand.XB[fid]])
    R, _t, rms = prigs.best_rigid(P, Q)
    return R, rms


# 1. the hints in the search's frame
log("")
log("1. Hints in the search's frame (pinhole starts aligned to the truth by their wand points)")
if hints is not None:
    sizes = data.sizes
    cxcy = [((w_ - 1) / 2.0, (h - 1) / 2.0) for (w_, h) in sizes]
    pins, _pinf = pw.pinhole_start(obs, data.L, sizes, cxcy, rig, hints=hints)
    for sol_p in pins:
        R_al, rms_al = align_rotation(sol_p, range(obs.n_cams))
        parts = []
        for c in range(obs.n_cams):
            w = rig.cam_win[c]
            n_true = rig_t.wins[w].n
            n_h = R_al @ pw.hint_to_frame(hints["normals"][c], sol_p.cams[c], rig.windows[w])
            n_d = R_al @ sol_p.cams[c].R[2]
            parts.append(f"cam {c + 1}: hint {ang(n_h, n_true):.1f} deg, default guess {ang(n_d, n_true):.1f} deg")
        log(f"  whole-rig pinhole start (mirror {sol_p.info.get('mirror')}, rms {sol_p.rms:.3f} px, alignment "
            f"{rms_al:.2f} mm): " + "; ".join(parts))
    for w in range(len(rig.windows)):
        cs = rig.cams_of(w)
        if len(cs) < 2:
            continue
        o_w = pw._restrict_obs(obs, cs)
        rig_w = pw.Rig([rig.windows[w].copy()], [0] * len(cs))
        sub_h = {"normals": [hints["normals"][c] for c in cs]}
        try:
            pins_w, _ = pw.pinhole_start(o_w, data.L, [sizes[c] for c in cs], [cxcy[c] for c in cs], rig_w, hints=sub_h)
        except (ValueError, np.linalg.LinAlgError) as exc:
            log(f"  window {w + 1} group {[c + 1 for c in cs]}: pinhole start failed ({exc})")
            continue
        for sol_p in pins_w:
            R_al, rms_al = align_rotation(sol_p, cs)
            parts = []
            for k, c in enumerate(cs):
                n_h = R_al @ pw.hint_to_frame(hints["normals"][c], sol_p.cams[k], rig.windows[w])
                n_d = R_al @ sol_p.cams[k].R[2]
                parts.append(f"cam {c + 1}: hint {ang(n_h, rig_t.wins[w].n):.1f} deg, default guess "
                             f"{ang(n_d, rig_t.wins[w].n):.1f} deg")
            n_mean = R_al @ pw.hinted_normal(sub_h, list(range(len(cs))), dict(enumerate(sol_p.cams)), rig.windows[w])
            log(f"  window {w + 1} group {[c + 1 for c in cs]} pinhole start (mirror {sol_p.info.get('mirror')}, rms "
                f"{sol_p.rms:.3f} px, alignment {rms_al:.2f} mm): " + "; ".join(parts)
                + f"; the window's hinted normal {ang(n_mean, rig_t.wins[w].n):.1f} deg")

# 2. every start, as calibrate builds them
log("")
log("2. The bootstrap (its log), then every start and where stage 4 takes it")
t0 = time.time()
multi = sum(1 for w in range(len(rig.windows)) if len(rig.cams_of(w)) >= 2)
few = any(1 <= len(rig.cams_of(w)) <= 2 for w in range(len(rig.windows)))
searches = ("length", "fixed") if few else ("fixed",)
per_search, cache = [], {}
for sr in searches:
    log(f"  -- search {sr}")
    try:
        if multi >= 1:
            st, inf = pw.bootstrap_groups(obs, rig, data.L, data.sizes, hints=hints, search=sr, cache=cache,
                                          verbose=True, log=lambda *a: log("    " + " ".join(str(x) for x in a)))
        else:
            st, inf = pw.bootstrap_singles(obs, rig, data.L, data.sizes, hints=hints, search=sr, verbose=True,
                                           log=lambda *a: log("    " + " ".join(str(x) for x in a)))
    except (ValueError, np.linalg.LinAlgError) as exc:
        log(f"    fell back to the whole-rig start ({exc!r})")
        st, inf = pw.bootstrap(obs, rig, data.L, data.sizes, hints=hints, search=sr, verbose=True,
                               log=lambda *a: log("    " + " ".join(str(x) for x in a)))
    per_search.append([(x, sr) for x in sorted(st, key=lambda s_: s_[0])])
    log(f"    {len(st)} starts; info: {json.dumps(inf, default=str)[:2500]}")
starts = [x for tup in itertools.zip_longest(*per_search) for x in tup if x is not None]
log(f"  bootstrap {time.time() - t0:.0f} s, {len(starts)} starts (the searches' starts taken in turn)")
solve_kw = dict(fit_k1=False, fit_L=False, k_sigma=5.0, noise_weights="camera", sigma_cam_px=None, priors=[],
                verbose=False)
results = []
for k, ((cost, rms3, cams0, rig0), sr) in enumerate(starts):
    Q, p = prigs.rigid_from_cam(cams0[0], rig_t.cams[0])
    ca, wa = prigs.transform(cams0, rig0.windows, Q, p)
    werr = [x["normal_deg"] for x in prigs.compare_wins(wa, rig_t.wins)]
    cerr = prigs.compare_cams(ca, rig_t.cams)
    o = pw.Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, obs.active.copy())
    try:
        sol = pw.solve(data, cams0, rig0, obs=o, **solve_kw)
    except (np.linalg.LinAlgError, ValueError) as exc:
        log(f"  start {k} ({sr}): coarse cost {cost:.4g}, rms {rms3:.3f} px; windows "
            f"{', '.join(f'{x:.1f}' for x in werr)} deg off -> stage 4 FAILED ({exc})")
        results.append(None)
        continue
    cmp = prigs.same_minimum(sol, ref)
    Q2, p2 = prigs.rigid_from_cam(sol.cams[0], rig_t.cams[0])
    _c2, w2 = prigs.transform(sol.cams, sol.rig.windows, Q2, p2)
    werr2 = [x["normal_deg"] for x in prigs.compare_wins(w2, rig_t.wins)]
    results.append((sol, cmp))
    f_txt = ", ".join("%+.0f" % (100 * x["f_rel"]) for x in cerr)
    r_txt = ", ".join("%.1f" % x["rot_deg"] for x in cerr)
    v_txt = "SAME as the truth-started fit" if cmp["same"] else "different, max |z| %.1e %s" % (cmp["max_abs_z"],
                                                                                              cmp["worst"])
    log(f"  start {k} ({sr}): coarse cost {cost:.4g}, rms {rms3:.3f} px; windows "
        f"{', '.join('%.1f' % x for x in werr)} deg off, cameras' f {f_txt} %, rotations {r_txt} deg -> stage 4 rms "
        f"{sol.rms:.6f} ({v_txt}), windows {', '.join('%.2f' % x for x in werr2)} deg off, active "
        f"{cmp['active']}/{cmp['active_ref']}")

# 3. which calibrate fits, and what it chooses
tried, stage3, run = 0, [], []
for k, ((cost, rms3, cams0, rig0), sr) in enumerate(starts):
    if k >= 6:
        break
    if k >= 2 and min(stage3, default=0.0) <= 1.0:
        rr = [results[j][0].rms for j in run if results[j] is not None]
        if not rr or max(rr) <= (1.0 + 1e-3) * min(rr):
            break
    stage3.append(rms3 if np.isfinite(rms3) else 0.0)
    run.append(k)
pool = [results[j][0] for j in run if results[j] is not None]
log("")
if pool:
    ch = pw._choose(pool, "camera")
    jj = [j for j in run if results[j] is not None and results[j][0] is ch][0]
    log(f"3. calibrate fits starts {run}; it chooses start {jj} (rms {ch.rms:.6f}, "
        f"{'the same minimum' if results[jj][1]['same'] else 'a different minimum'}) before the restarts")
any_same = [k for k, r in enumerate(results) if r is not None and r[1]["same"]]
log(f"   starts whose stage 4 reaches the truth-started fit's minimum: {any_same if any_same else 'none'}")
log(f"total {time.time() - t0:.0f} s")
