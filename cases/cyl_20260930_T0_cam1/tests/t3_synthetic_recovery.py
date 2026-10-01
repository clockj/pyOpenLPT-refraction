"""T3: synthetic end-to-end recovery.

Truth: the final calibrated rig (fit_state.pkl in the processed folder), with its plane shifts (if any) acting as the
stage's sideways play and its board corrections as the board's errors.  Observations: the same dots (planes and
indices) as the real calibration, projected through the truth, plus Gaussian noise with the real per-axis residual
RMS.  The whole automatic pipeline (multi-start, M1, the final model, outlier pass) then runs from scratch, three
noise seeds.

Passes (plan 2026-10-01, section 4) if
  * f, the camera centre, the axis (position and direction) and any stage scale or plane offsets are recovered
    within 3 reported sigma (z-scores);
  * the per-plane shifts (40 z-scores per seed, 36 of them independent) are recovered with z-scores of rms between
    0.5 and 1.5 and none beyond 4 (with 108 independent ones, one beyond 3 turns up by chance about one run in
    four);
  * the held-out pixel error at fresh points (between the planes and 2 mm beyond either end) is below 0.05 px rms.
Writes t3_out.txt beside this script.
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402
import run_calibration as RC                  # noqa: E402

LOG = []
HELD_MAX_PX = 0.05


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def rot_angle_deg(Ra, Rb) -> float:
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def fresh_z(obs: CF.Obs) -> list:
    """Depths the calibration never saw: halfway between neighbouring planes, and 2 mm beyond either end."""
    z = np.asarray(obs.z_nom, float)
    return list(0.5 * (z[1:] + z[:-1])) + [z[0] - 2.0, z[-1] + 2.0]


def fresh_points(obs: CF.Obs) -> np.ndarray:
    """Fresh points over the seen X, Y range at the depths of fresh_z, on a 1.5 mm grid."""
    X = obs.pitch[0] * obs.i
    Y = obs.pitch[1] * obs.j
    xs = np.arange(X.min(), X.max() + 1e-9, 1.5)
    ys = np.arange(Y.min(), Y.max() + 1e-9, 1.5)
    return np.array([(x, y, z) for z in fresh_z(obs) for x in xs for y in ys], float)


def main():
    cfg = case.load_config()
    state = RC.load_state(os.path.join(cfg["out_dir"], "fit_state.pkl"))
    truth, obs, model = state["rig"], state["obs"], state["info"]["model"]
    sig_u = float(np.sqrt(np.mean(state["resid"][:, 0] ** 2)))
    sig_v = float(np.sqrt(np.mean(state["resid"][:, 1] ** 2)))
    cam_t = truth.camera()
    uv_t = CF.project(cam_t, truth.world_points(obs))
    say(f"T3 synthetic end-to-end recovery; truth = the calibrated rig (model {model}), {len(obs.plane)} dots, "
        f"noise sigma ({sig_u:.3f}, {sig_v:.3f}) px")
    P_fresh = fresh_points(obs)
    uv_fresh_t = CF.project(cam_t, P_fresh)
    zf = fresh_z(obs)
    z_main, z_shift = [], []
    held_ok = True
    for seed in (11, 12, 13):
        t0 = time.time()
        rng = np.random.default_rng(seed)
        uv = uv_t + rng.normal(0.0, 1.0, uv_t.shape) * np.array([sig_u, sig_v])
        obs_s = CF.Obs(obs.plane, obs.i, obs.j, uv, obs.z_nom, obs.pitch, obs.plane_ids)
        out = RC.fit_obs(obs_s, cfg, RC.scale_from_obs(obs_s), model=model, verbose=False)
        fr = out["fit"]
        sig = CF.physical_sigmas(fr)
        r = fr.rig
        zs = {}
        zs["f"] = (r.f - truth.f) / sig["f_px"]
        zs["C"] = ((r.C - truth.C) / np.array(sig["camera_centre_mm"])).tolist()
        a_t, a_f = truth.axis_at_y0(), r.axis_at_y0()
        zs["axis_X"] = (a_f[0] - a_t[0]) / sig["axis_position_X_Z_mm"][0]
        zs["axis_Z"] = (a_f[2] - a_t[2]) / sig["axis_position_X_Z_mm"][1]
        tilt = lambda d: (math.atan2(d[0], d[1]), math.atan2(d[2], d[1]))      # noqa: E731
        tt, tf = tilt(truth.axis_dir), tilt(r.axis_dir)
        zs["axis_tilt"] = math.degrees(tf[0] - tt[0]) / sig["axis_tilt_lean_deg"][0]
        zs["axis_lean"] = math.degrees(tf[1] - tt[1]) / sig["axis_tilt_lean_deg"][1]
        if "dz_mm" in sig:
            zs["dz"] = ((r.dz[1:] - truth.dz[1:]) / np.array(sig["dz_mm"][1:])).tolist()
        if "stage_scale" in sig:
            zs["stage_scale"] = (r.zscale - truth.zscale) / sig["stage_scale"]
        flat = [abs(v) for k, v in zs.items() for v in (v if isinstance(v, list) else [v])]
        z_main += flat
        zsh = None
        if truth.shift is not None and r.shift is not None:
            zsh = ((r.shift - truth.shift) / np.array(sig["plane_shift_mm"])).ravel()
            z_shift += list(zsh)
        uv_fresh_f = CF.project(r.camera(), P_fresh)
        d = uv_fresh_f - uv_fresh_t
        ok = np.all(np.isfinite(d), axis=1)
        held = float(np.sqrt(np.mean(np.sum(d[ok] ** 2, axis=1))))
        held_max = float(np.max(np.linalg.norm(d[ok], axis=1)))
        held_ok &= held < HELD_MAX_PX
        say(f"\nseed {seed}: fit rms {fr.rms:.4f} px, {out['info']['n_outliers']} outliers dropped, "
            f"{time.time() - t0:.0f} s")
        say(f"  errors: f {r.f - truth.f:+.2f} px (sigma {sig['f_px']:.2f}); camera centre "
            f"{np.round(r.C - truth.C, 3)} mm (sigma {np.round(sig['camera_centre_mm'], 3)}); rotation "
            f"{rot_angle_deg(r.R, truth.R):.5f} deg")
        say(f"  axis at Y=0 {np.round(a_f - a_t, 4)} mm (sigma X,Z {np.round(sig['axis_position_X_Z_mm'], 4)}); "
            f"tilt {math.degrees(tf[0] - tt[0]):+.4f} deg, lean {math.degrees(tf[1] - tt[1]):+.4f} deg "
            f"(sigma {np.round(sig['axis_tilt_lean_deg'], 4)})")
        if "dz" in zs:
            say(f"  plane offsets: error rms {np.sqrt(np.mean((r.dz - truth.dz) ** 2)):.4f} mm, max "
                f"{np.max(np.abs(r.dz - truth.dz)):.4f} mm; median sigma {np.median(sig['dz_mm'][1:]):.4f} mm")
        if zsh is not None:
            e = r.shift - truth.shift
            say(f"  plane shifts: error rms ({np.sqrt(np.mean(e[:, 0] ** 2)) * 1e3:.2f}, "
                f"{np.sqrt(np.mean(e[:, 1] ** 2)) * 1e3:.2f}) um against true rms "
                f"({np.sqrt(np.mean(truth.shift[:, 0] ** 2)) * 1e3:.1f}, "
                f"{np.sqrt(np.mean(truth.shift[:, 1] ** 2)) * 1e3:.1f}) um; z-scores rms "
                f"{np.sqrt(np.mean(zsh ** 2)):.2f}, max |z| {np.max(np.abs(zsh)):.2f}")
        if truth.board is not None and r.board is not None:
            lut = {(int(a), int(b)): q for q, (a, b) in enumerate(truth.board_ij)}
            e = np.array([r.board[q] - truth.board[lut[(int(a), int(b))]] for q, (a, b) in enumerate(r.board_ij)
                          if (int(a), int(b)) in lut])
            say(f"  board corrections: error rms ({np.sqrt(np.mean(e[:, 0] ** 2)) * 1e3:.2f}, "
                f"{np.sqrt(np.mean(e[:, 1] ** 2)) * 1e3:.2f}) um against true rms "
                f"({np.sqrt(np.mean(truth.board[:, 0] ** 2)) * 1e3:.1f}, "
                f"{np.sqrt(np.mean(truth.board[:, 1] ** 2)) * 1e3:.1f}) um")
        say(f"  z-scores: f {zs['f']:+.2f}, C {np.round(zs['C'], 2)}, axis X {zs['axis_X']:+.2f} Z "
            f"{zs['axis_Z']:+.2f} tilt {zs['axis_tilt']:+.2f} lean {zs['axis_lean']:+.2f}" +
            (f", dz max |z| {np.max(np.abs(zs['dz'])):.2f}" if "dz" in zs else "") +
            (f", stage scale {zs['stage_scale']:+.2f}" if "stage_scale" in zs else ""))
        say(f"  held-out ({ok.sum()} fresh points at z = {zf[0]:.2f} ... {zf[-3]:.2f} between the planes and "
            f"{zf[-2]:.2f}, {zf[-1]:.2f} beyond): rms {held:.4f} px, max {held_max:.4f} px")
    z_main = np.array(z_main)
    say(f"\ncamera and port z-scores: n {len(z_main)}, rms {np.sqrt(np.mean(z_main ** 2)):.2f}, "
        f"max {z_main.max():.2f}, beyond 3: {int(np.sum(z_main > 3))}")
    ok_shift = True
    if z_shift:
        z_shift = np.abs(np.array(z_shift))
        zr = float(np.sqrt(np.mean(z_shift ** 2)))
        ok_shift = 0.5 <= zr <= 1.5 and z_shift.max() < 4.0
        say(f"plane-shift z-scores: n {len(z_shift)}, rms {zr:.2f}, max {z_shift.max():.2f}, "
            f"beyond 3: {int(np.sum(z_shift > 3))}")
    say("RESULT: " + ("PASS" if z_main.max() <= 3.0 and ok_shift and held_ok else "FAIL"))
    with open(os.path.join(HERE, "t3_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
