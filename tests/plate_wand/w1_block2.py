"""W1 block 2 (imported by w1_jacobians.py): plate_wand.evaluate's global columns (cameras, windows, L) and frame
columns against central differences of the whole residual, moving the parameters exactly as the solver does
(Layout.apply, Frames.retract).  Rigs: B (two windows, k1 and L free), A as a pinhole rig (the pinhole stage), and D
with window 1 fixed."""
import math

import numpy as np

import prigs

pm = prigs.pm
import plate_wand as pw  # noqa: E402


def _check(rig_t: prigs.TrueRig, rig: pw.Rig, fit_k1, fit_L, fixed, rng, label, log):
    wand = prigs.make_wand(rig_t, 40, rng, noise_px=0.3)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    obs = pw.observations(data)
    cams = [c.copy() for c in rig_t.cams]
    for c in cams:
        c.dist = np.array([0.03, 0.0, 0.0, 0.0, 0.0]) if fit_k1 else c.dist
    frames = pw.frames_from_ends(wand.XA[obs.frame_ids], wand.XB[obs.frame_ids]).retract(
        rng.normal(scale=[0.3, 0.3, 0.3, 0.01, 0.01], size=(obs.n_frames, 5)))
    layout = pw.Layout(rig_t.n_cams, len(rig.windows), fit_k1=fit_k1, fit_L=fit_L, fixed_windows=fixed,
                       anchors=pw.anchors_for(cams, rig))
    ev = pw.evaluate(cams, rig, frames, wand.L, obs, layout)
    worst_g, worst_f = 0.0, 0.0
    worst_name = ""
    for k in range(layout.P):
        name = layout.names[k]
        h = 1e-7 if (".w" in name or name.endswith((".a", ".b")) or "logf" in name or "k1" in name) else 1e-5
        dx = np.zeros(layout.P)
        dx[k] = h
        cp, rp, Lp = layout.apply(cams, rig, wand.L, dx)
        cn, rn, Ln = layout.apply(cams, rig, wand.L, -dx)
        up = pw.evaluate(cp, rp, frames, Lp, obs).r
        un = pw.evaluate(cn, rn, frames, Ln, obs).r
        fd = (up - un) / (2 * h)
        an = ev.Jg[:, :, k]
        sel = ev.ok & np.all(np.isfinite(fd), axis=1)
        scale = float(np.max(np.abs(an[sel])))
        if scale == 0.0:
            continue
        e = float(np.max(np.abs(fd[sel] - an[sel]))) / scale
        if e > worst_g:
            worst_g, worst_name = e, name
    for j in range(5):
        h = 1e-5 if j < 3 else 1e-7
        df = np.zeros((obs.n_frames, 5))
        df[:, j] = h
        up = pw.evaluate(cams, rig, frames.retract(df), wand.L, obs).r
        un = pw.evaluate(cams, rig, frames.retract(-df), wand.L, obs).r
        fd = (up - un) / (2 * h)
        an = ev.Jf[:, :, j]
        sel = ev.ok
        scale = float(np.max(np.abs(an[sel])))
        worst_f = max(worst_f, float(np.max(np.abs(fd[sel] - an[sel]))) / scale)
    log(f"  {label}: {layout.P} global columns, worst relative error {worst_g:.2e} ({worst_name}); frame columns "
        f"{worst_f:.2e}")
    return worst_g, worst_f


def run(log, rng):
    log("block 2: plate_wand.evaluate against differences of the whole residual")
    out = {}
    rb = prigs.rig_B()
    out["B, k1 and L free"] = _check(rb, pw.Rig([w.copy() for w in rb.wins], rb.cam_win), True, True, (), rng,
                                     "rig B (two windows), k1 and L free", log)
    ra = prigs.rig_A()
    out["A pinhole"] = _check(ra, pw.Rig([], [-1] * ra.n_cams), False, False, (), rng, "rig A as pinhole cameras",
                              log)
    rd = prigs.rig_D()
    out["D window fixed"] = _check(rd, pw.Rig([w.copy() for w in rd.wins], rd.cam_win), False, False, (0,), rng,
                                   "rig D, window fixed", log)
    worst = max(max(v) for v in out.values())
    ok = worst <= 1e-6
    log(f"  block 2 worst {worst:.2e}: " + ("PASS" if ok else "FAIL"))
    return ok, {k: {"global": v[0], "frames": v[1]} for k, v in out.items()}
