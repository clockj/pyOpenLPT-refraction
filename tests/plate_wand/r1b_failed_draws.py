"""R1b (diagnosis, 2026-10-03, after R1): why half of R1's draws start with observations that do not project.

In R1 every draw whose observations all projected reached the global minimum (s = 1-3, both centres); every draw that
failed had hundreds of active observations that did not project at the screen's start (`n_fail` > 0) and stayed out
of its cost.  Here the same draws (the answer as centre, s = 1 and 3) are rebuilt and, before any fitting, every
active observation is projected from the draw's cameras and windows at the re-triangulated frames.  Per draw: how many
fail and why (plate_model.project: the camera not on the air side, h1 <= 0; the point not in the water, h_W <= 0;
the point behind the camera; Newton), by camera; how far each window moved (its normal, its offset at the anchor, in
mm and in the answer's sigma); the wand cloud's nearest point to each window's water side, before and after.
Output: r1b_failed_draws_out.txt
"""
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

log = prigs.Tee(HERE / "r1b_failed_draws_out.txt")
t_all = time.time()
rig_t, data = r1.seed_2000()
placeholder = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins], rig_t.cam_win)
obs0 = pw.observations(data)
ans = pw.calibrate(data, placeholder, output_frame=False)
lay = ans.layout
log(f"R1b: seed {r1.SEED}, the answer from scratch: rms {ans.rms:.6f} px (as R1)")
try:
    Lc = np.linalg.cholesky(ans.cov)
except np.linalg.LinAlgError:
    ev_, V_ = np.linalg.eigh(ans.cov)
    Lc = V_ * np.sqrt(np.maximum(ev_, 0.0))
sd = np.sqrt(np.diag(ans.cov))
o_a = r1.fresh(obs0, ans.obs.active)
rows = np.flatnonzero(o_a.active)
XA0, XB0 = ans.frames.ends(data.L)


def water_gap(rig, XA, XB, frames_ok):
    """Each window's nearest wand end (of the frames seen through it) to its water-side surface, mm (< 0: outside)."""
    out = []
    for w, win in enumerate(rig.windows):
        cs = rig.cams_of(w)
        fr = np.unique(o_a.frame[rows][np.isin(o_a.cam[rows], cs)])
        fr = fr[frames_ok[fr]]
        X = np.vstack([XA[fr], XB[fr]])
        out.append(float(np.min(win.depth(X))) if len(X) else float("nan"))
    return out


log(f"  the answer: each window's nearest wand end to its water side: "
    + ", ".join(f"window {w + 1} {g:.1f} mm" for w, g in
                enumerate(water_gap(ans.rig, XA0, XB0, np.ones(o_a.n_frames, bool))))
    + "; sigma of the windows' offsets at their anchors: "
    + ", ".join(f"{sd[lay.wpos[w, 2]]:.1f} mm" for w in range(len(ans.rig.windows))))
# is the answer pinned against window 1's water side?  The wand ends' depths in the water, at the answer, at the global
# minimum (the fit from the truth) and in the truth itself
ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
for name, sol_ in (("the answer", ans), ("the global minimum", ref)):
    XA_, XB_ = sol_.frames.ends(data.L)
    act_f = np.zeros(sol_.obs.n_frames, bool)
    act_f[sol_.obs.frame[sol_.obs.active]] = True
    for w, win in enumerate(sol_.rig.windows):
        cs = sol_.rig.cams_of(w)
        fr_w = np.unique(sol_.obs.frame[sol_.obs.active & np.isin(sol_.obs.cam, cs)])
        dep = np.sort(np.concatenate([win.depth(XA_[fr_w]), win.depth(XB_[fr_w])]))
        log(f"  {name}, window {w + 1}: the wand ends' depths in the water, smallest five "
            + ", ".join(f"{v:.3g}" for v in dep[:5]) + f" mm; within 0.01 / 0.1 / 1 mm of the surface: "
            f"{int(np.sum(dep < 0.01))} / {int(np.sum(dep < 0.1))} / {int(np.sum(dep < 1.0))} of {len(dep)}")
gap_t = [float(np.min(np.concatenate([w.depth(prigs.box_corners(rig_t.centre, rig_t.half))])))
         for w in rig_t.wins]
log("  the truth: the wand box's nearest corner to each window's water side "
    + ", ".join(f"window {w + 1} {g:.1f} mm" for w, g in enumerate(gap_t)))
hist = ans.history
log(f"  the answer's last bundle adjustment: {len(hist)} iterations, the last "
    f"{'accepted' if hist and hist[-1]['accepted'] else 'NOT accepted (stuck: every trial step raised the cost or made a projection fail)'}; "
    f"stage 4: {ans.info.get('rounds')} rounds, {ans.info.get('lm_iterations')} LM iterations")
rng = np.random.default_rng(r1.DRAW_SEED)
Z = rng.standard_normal((r1.N_DRAWS, r1.MAX_REDRAW + 1, lay.P))
for s in (1.0, 3.0):
    log(f"scale s = {s:g} (the first z of each draw, as R1 used: R1 had no redraws)")
    for k in range(r1.N_DRAWS):
        dx = s * (Lc @ Z[k, 0])
        cams_p, rig_p, _ = lay.apply(ans.cams, ans.rig, data.L, dx)
        fr = pw.initial_frames(cams_p, rig_p, o_a, data.L)
        XA, XB, cnt = pw.triangulate(cams_p, rig_p, o_a)
        tri_ok = np.all(cnt >= 2, axis=1) & np.all(np.isfinite(XA), axis=1) & np.all(np.isfinite(XB), axis=1)
        XAf, XBf = fr.ends(data.L)
        X = np.where((o_a.end[rows] == 0)[:, None], XAf[o_a.frame[rows]], XBf[o_a.frame[rows]])
        rw = pm.rows(cams_p, o_a.cam[rows])
        ww = pm.wrows(rig_p.windows, np.asarray(rig_p.cam_win, int)[o_a.cam[rows]])
        pr = pm.project(rw, ww, X)
        bad = ~pr.ok
        why = {"h1<=0": int(np.sum(bad & ~(pr.h1 > 0))), "hW<=0": int(np.sum(bad & (pr.h1 > 0) & ~(pr.hW > 0))),
               "other": int(np.sum(bad & (pr.h1 > 0) & (pr.hW > 0)))}
        by_cam = np.bincount(o_a.cam[rows][bad], minlength=o_a.n_cams)
        wmov = []
        for w in range(len(rig_p.windows)):
            a, b, dl = (dx[g] for g in lay.wpos[w])
            wmov.append(f"window {w + 1}: normal {math.degrees(math.hypot(a, b)):.2f} deg, offset {dl:+.1f} mm "
                        f"({dl / sd[lay.wpos[w, 2]]:+.1f} sigma)")
        gaps = water_gap(rig_p, XAf, XBf, tri_ok)
        log(f"  #{k:2d}: {int(bad.sum()):4d} of {len(rows)} fail ({why}; by camera {by_cam.tolist()}); "
            f"{int((~tri_ok).sum())} frames not triangulated; " + "; ".join(wmov)
            + "; nearest end to the water side " + ", ".join(f"{g:.1f}" for g in gaps) + " mm")
log(f"done, {time.time() - t_all:.0f} s")
