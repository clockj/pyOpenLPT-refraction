"""T4: held-out planes on the real data.

(a) Even/odd.  The delivered model is fitted from scratch (multi-start included) on the even planes z = 0, 5.08,
    ..., 45.72 mm and tested on the odd planes z = 2.54, 7.62, ..., 48.26 mm, three ways:
      * stage z: the plane at its stage position, nothing fitted (tests everything, the stage included);
      * own shift: only the plane's sideways shift fitted (2 parameters; the stage carriage's play), z held;
      * own z: only the plane's z fitted (1 parameter), everything else held.
(b) Leave-one-plane-out.  20 fits on 19 planes, each started from the full calibration (outliers as removed
    there), each predicting the left-out plane the same three ways.

Passes (plan 2026-10-01, section 4) if the held-out RMS at the stage position, and with the plane's own z, are
below 1.5x the training RMS in (a) and (b).  Writes t4_out.txt beside this script.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402
import run_calibration as RC                  # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def rms2(r) -> float:
    r = np.asarray(r).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


def plane_obs(obs: CF.Obs, k: int) -> CF.Obs:
    m = obs.plane == k
    return CF.Obs(np.zeros(m.sum(), int), obs.i[m], obs.j[m], obs.uv[m], np.array([obs.z_nom[k]]), obs.pitch,
                  np.array([k]))


def predict_plane(rig: CF.Rig, pobs: CF.Obs) -> dict:
    """Residuals of one held-out plane: rms at the stage position, with its own sideways shift fitted, and with its
    own z fitted; the fitted shift (mm) and dz (mm).

    The plane sits at the rig's stage scale times its stage position.  Projected with pyOpenLPT's C++ camera; a
    rig with the diagnostic wall field has no C++ counterpart and is projected with the numpy tracer instead.
    """
    cam = None if rig.has_wall() else rig.camera()
    base = rig.copy()
    base.dz = np.zeros(1)
    base.shift = None

    def res_shift(s):
        r = base.copy()
        r.shift = np.array([[s[0], s[1]]])
        return CF.forward_residual(r, pobs, cam)

    def res_z(dz):
        r = base.copy()
        r.dz = np.array([dz[0]])
        return CF.forward_residual(r, pobs, cam)
    r0 = res_z([0.0])
    s1 = least_squares(res_shift, [0.0, 0.0], x_scale=[0.01, 0.01], diff_step=[1e-4, 1e-4])
    s2 = least_squares(res_z, [0.0], x_scale=[0.1], diff_step=[1e-4])
    return {"stage": rms2(r0), "shift": rms2(s1.fun), "z": rms2(s2.fun), "shift_mm": s1.x.tolist(),
            "dz_mm": float(s2.x[0]), "resid_stage": r0.reshape(-1, 2), "resid_shift": s1.fun.reshape(-1, 2)}


def error_stats(name: str, e: np.ndarray, where: list) -> str:
    """One line of per-point statistics of the errors e (n, 2), given as detected minus model (px): the error
    length |e| (mean, std, RMS, median, 95th percentile, max and where), and each component (mean, std, max |.|).
    `where` labels each row (plane z, i, j)."""
    L = np.linalg.norm(e, axis=1)
    k = int(np.argmax(L))
    z, i, j = where[k]
    return (f"    {name:30s} n {len(L):4d} | |e| mean {L.mean():.3f} +- std {L.std():.3f}, RMS "
            f"{np.sqrt(np.mean(L ** 2)):.3f}, median {np.median(L):.3f}, 95% {np.percentile(L, 95):.3f}, max "
            f"{L.max():.3f} (z {z:.2f}, dot {i},{j}) | u {e[:, 0].mean():+.3f} +- {e[:, 0].std():.3f}, max "
            f"{np.abs(e[:, 0]).max():.3f} | v {e[:, 1].mean():+.3f} +- {e[:, 1].std():.3f}, max "
            f"{np.abs(e[:, 1]).max():.3f}")


def subset_obs(obs: CF.Obs, keep: list) -> CF.Obs:
    """Rows of the kept planes, renumbered 0.. (the first kept plane then holds dz = 0)."""
    remap = {k: q for q, k in enumerate(keep)}
    m = np.isin(obs.plane, keep)
    return CF.Obs(np.array([remap[k] for k in obs.plane[m]]), obs.i[m], obs.j[m], obs.uv[m],
                  obs.z_nom[keep], obs.pitch, np.array(keep))


def start_for_subset(rig: CF.Rig, keep: list) -> CF.Rig:
    """The full rig expressed in a subset's frame (its first plane at dz = 0): shift everything in Z together.  The
    plane shifts are carried over (the fit re-projects them onto the subset's shift basis)."""
    r = rig.copy()
    dz = rig.dz[keep]
    off = float(dz[0])
    r.dz = dz - off
    r.C = r.C - np.array([0.0, 0.0, off])
    r.axis_pt = r.axis_pt - np.array([0.0, 0.0, off])
    if rig.shift is not None:
        r.shift = rig.shift[keep]
    return r


def summary_line(name, vals):
    v = np.asarray(vals, float)
    return f"{name} {np.sqrt(np.mean(v ** 2)):.4f} (worst {v.max():.3f})"


def main():
    cfg = case.load_config()
    state = RC.load_state(os.path.join(cfg["out_dir"], "fit_state.pkl"))
    rig, obs, blocks, model = state["rig"], state["obs"], state["blocks"], state["info"]["model"]
    K = obs.n_planes
    say(f"T4 held-out planes, model {model}; full calibration rms {state['rms']:.4f} px over {len(obs.plane)} dots")

    # (a) even / odd, from scratch
    t0 = time.time()
    even = list(range(0, K, 2))
    odd = list(range(1, K, 2))
    obs_e = subset_obs(obs, even)
    out = RC.fit_obs(obs_e, cfg, RC.scale_from_obs(obs_e), model=model, verbose=False, outlier_k=0)
    fe = out["fit"]
    say(f"\n(a) trained on the even planes ({len(obs_e.plane)} dots) from the multi-start: training rms "
        f"{fe.rms:.4f} px ({time.time() - t0:.0f} s)")
    s_f, s_c = fe.rig.summary(), rig.summary()
    say(f"    vs the full calibration: f {s_f['f_px']:.1f} vs {s_c['f_px']:.1f} px; camera centre "
        f"{np.round(np.array(s_f['camera_centre_mm']) - np.array(s_c['camera_centre_mm']), 3)} mm apart; axis at "
        f"Y=0 {np.round(np.array(s_f['axis_point_at_Y0_mm']) - np.array(s_c['axis_point_at_Y0_mm']), 3)} mm apart")
    say("    test plane z   stage-z rms   own-shift rms   own-z rms   fitted shift (um)   fitted dz (mm)")
    a = {"stage": [], "shift": [], "z": []}
    pts = {"a_stage": [], "a_shift": [], "b_stage": [], "b_shift": [], "a_where": [], "b_where": []}
    for k in odd:
        pob = plane_obs(obs, k)
        p = predict_plane(fe.rig, pob)
        for key in a:
            a[key].append(p[key])
        pts["a_stage"].append(-p["resid_stage"])
        pts["a_shift"].append(-p["resid_shift"])
        pts["a_where"] += [(obs.z_nom[k], int(i), int(j)) for i, j in zip(pob.i, pob.j)]
        say(f"    {obs.z_nom[k]:6.2f}      {p['stage']:9.4f}   {p['shift']:11.4f}   {p['z']:9.4f}   "
            f"{np.round(np.array(p['shift_mm']) * 1000, 1)!s:>18s}   {p['dz_mm']:+10.4f}")
    say("    odd planes, rms over planes: " + ", ".join(summary_line(n, a[n]) for n in ("stage", "shift", "z")) +
        f"; training {fe.rms:.4f}")
    pass_a = (np.sqrt(np.mean(np.square(a["stage"]))) < 1.5 * fe.rms and
              np.sqrt(np.mean(np.square(a["z"]))) < 1.5 * fe.rms)

    # (b) leave one plane out, in parallel
    t0 = time.time()
    jobs = []
    for k in range(K):
        keep = [q for q in range(K) if q != k]
        jobs.append((start_for_subset(rig, keep), subset_obs(obs, keep), blocks))
    fits = RC.parallel_map(RC._fit_one, jobs)
    say(f"\n(b) leave one plane out: {K} fits ({time.time() - t0:.0f} s)")
    say("    left out   train rms   stage-z rms   own-shift rms   own-z rms   own dz (mm)   f (px)    camera centre (mm)")
    b = {"stage": [], "shift": [], "z": []}
    b_tr, Cs, fs = [], [], []
    for k, fk in enumerate(fits):
        r_k = fk.rig
        keep = [q for q in range(K) if q != k]
        pob = plane_obs(obs, k)
        p = predict_plane(r_k, pob)
        for key in b:
            b[key].append(p[key])
        pts["b_stage"].append(-p["resid_stage"])
        pts["b_shift"].append(-p["resid_shift"])
        pts["b_where"] += [(obs.z_nom[k], int(i), int(j)) for i, j in zip(pob.i, pob.j)]
        b_tr.append(fk.rms)
        C = r_k.C + (np.array([0.0, 0.0, rig.dz[keep[0]]]) if keep[0] != 0 else 0.0)
        Cs.append(C)
        fs.append(r_k.f)
        say(f"    {obs.z_nom[k]:6.2f}   {fk.rms:9.4f}   {p['stage']:11.4f}   {p['shift']:13.4f}   {p['z']:9.4f}   "
            f"{p['dz_mm']:+11.4f}   {r_k.f:8.1f}  {np.round(C, 2)}")
    b_tr, Cs, fs = np.array(b_tr), np.array(Cs), np.array(fs)
    tr = float(np.sqrt(np.mean(b_tr ** 2)))
    say("    held-out rms over planes: " + ", ".join(summary_line(n, b[n]) for n in ("stage", "shift", "z")) +
        f"; training {tr:.4f}")
    say(f"    spread over the {K} fits (std): f {fs.std():.2f} px, camera centre {np.round(Cs.std(axis=0), 3)} mm")
    worst = float(np.max(np.array(b["stage"]) / b_tr))
    say(f"    worst plane: stage-z held-out / training = {worst:.2f} "
        f"(z = {obs.z_nom[int(np.argmax(np.array(b['stage']) / b_tr))]:.2f})")
    pass_b = (np.sqrt(np.mean(np.square(b["stage"]))) < 1.5 * tr and np.sqrt(np.mean(np.square(b["z"]))) < 1.5 * tr)

    # per-point statistics (the RMS figures above are over planes); errors given as detected minus model
    say("\nper-point errors, detected minus model (px):")
    say(error_stats("(a) odd planes, stage z", np.vstack(pts["a_stage"]), pts["a_where"]))
    say(error_stats("(a) odd planes, own shift", np.vstack(pts["a_shift"]), pts["a_where"]))
    say(error_stats("(b) each plane left out, stage z", np.vstack(pts["b_stage"]), pts["b_where"]))
    say(error_stats("(b) each plane left out, own shift", np.vstack(pts["b_shift"]), pts["b_where"]))
    where_fit = [(obs.z_nom[q], int(i), int(j)) for q, i, j in zip(obs.plane, obs.i, obs.j)]
    say(error_stats("calibration fit, all planes", -state["resid"], where_fit))
    for name, v in (("(a) stage z", a["stage"]), ("(b) stage z", b["stage"]), ("(b) own shift", b["shift"])):
        v = np.asarray(v)
        say(f"    per-plane RMS, {name:14s}: mean {v.mean():.3f} +- std {v.std():.3f}, min {v.min():.3f}, "
            f"max {v.max():.3f} over {len(v)} planes")
    say("RESULT: " + ("PASS" if pass_a and pass_b else f"FAIL (a {'ok' if pass_a else 'fail'}, "
                                                          f"b {'ok' if pass_b else 'fail'})"))
    with open(os.path.join(HERE, "t4_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
