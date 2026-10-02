"""T1b: the bundle adjustment's Jacobian (camera and frame columns through the manifold updates) against central
differences of the full residual.

At a point near (not at) the truth of two rigs, `cylinder_wand.evaluate` gives d(residual)/d(free camera
parameters, L) and d(residual)/d(frame parameters).  Each free camera column is compared with a central
difference of the residual under `Layout.apply` (R <- R exp([w]x), C <- C + dC, f <- f exp(dl), L <- L + dL); the
frame columns with `Frames.retract` (m <- m + dm, w <- the exponential map of S^2), Richardson-extrapolated.
Criterion: 1e-6 relative to each column's size (as T1).
Outputs: t1b_ba_jacobian_out.txt.
"""
import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

log = rigs.Tee(str(rigs.HERE / "t1b_ba_jacobian") + "_out.txt")


def residual(cams, tube, frames, L, obs, rows):
    ev = cw.evaluate(cams, tube, frames, L, obs, None, rows)
    return ev.r, ev.ok


def check(name, cams_true, fit_k1=False):
    rng = np.random.default_rng(3)
    tube = rigs.CASE_TUBE
    wd = rigs.wand_frames(cams_true, tube, 40, 10.0, rng, half=(15, 15, 15), noise_px=0.0)
    data = cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in cams_true])
    obs = cw.observations(data)
    cams = rigs.perturb_rig(cams_true, rng, rot_deg=0.05, trans_mm=0.5, f_rel=0.002)
    fr = cw.frames_from_ends(wd.XA[obs.frame_ids] + 0.05, wd.XB[obs.frame_ids] - 0.03)
    L = 10.0
    layout = cw.Layout(len(cams), fit_k1=fit_k1, fit_L=True)
    rows = np.arange(len(obs))
    ev = cw.evaluate(cams, tube, fr, L, obs, layout, rows)
    worst = 0.0
    lines = []

    def rich(fun, h):
        def cd(hh):
            rp, okp = fun(+hh)
            rm, okm = fun(-hh)
            return (rp - rm) / (2 * hh), okp & okm
        d1, o1 = cd(h)
        d2, o2 = cd(h / 2)
        return (4 * d2 - d1) / 3, o1 & o2

    for g, nm in enumerate(layout.names):
        h = 1e-5 if (".w" in nm or "logf" in nm) else (1e-3 if "k1" in nm else 1e-2)

        def fun(hh, g=g):
            dx = np.zeros(layout.P)
            dx[g] = hh
            cs, Lh = layout.apply(cams, L, dx)
            return residual(cs, tube, fr, Lh, obs, rows)
        num, ok = rich(fun, h)
        col = ev.Jg[:, :, g]
        ok &= ev.ok
        sc = np.sqrt(np.mean(np.sum(col[ok] ** 2, axis=1))) + 1e-300
        e = float(np.max(np.linalg.norm(col[ok] - num[ok], axis=1)) / sc)
        worst = max(worst, e)
        lines.append(f"{nm}:{e:.0e}")
    # frame columns: perturb every frame at once (each residual row depends on its own frame only)
    for k, nm in enumerate(("m1", "m2", "m3", "a", "b")):
        h = 1e-3 if k < 3 else 1e-5

        def fun(hh, k=k):
            df = np.zeros((obs.n_frames, 5))
            df[:, k] = hh
            return residual(cams, tube, fr.retract(df), L, obs, rows)
        num, ok = rich(fun, h)
        col = ev.Jf[:, :, k]
        ok &= ev.ok
        sc = np.sqrt(np.mean(np.sum(col[ok] ** 2, axis=1))) + 1e-300
        e = float(np.max(np.linalg.norm(col[ok] - num[ok], axis=1)) / sc)
        worst = max(worst, e)
        lines.append(f"frame.{nm}:{e:.0e}")
    log(f"{name}: {len(obs)} observations, {layout.P} free global parameters; worst {worst:.2e}")
    log("   " + " ".join(lines))
    return worst


if __name__ == "__main__":
    w1 = check("reference rig (3 cams, L free)", rigs.reference_rig(3, 45.0, sensor=(3200, 2400)))
    w2 = check("irregular 4-camera rig (L free, k1 free)", rigs.irregular_rig(), fit_k1=True)
    worst = max(w1, w2)
    log(f"  {'PASS' if worst < 1e-6 else 'FAIL'}  bundle-adjustment Jacobian agrees with central differences to 1e-6 "
        f"(worst {worst:.2e})")
    log.close()
