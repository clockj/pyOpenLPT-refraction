"""P6 (diagnosis before the fixes plan, 2026-10-06-fixes-plan.md): why, with equal weights, do the same detections of
T0 leave and come back every round of stage 4 (P5)?

    python tests/cylinder_wand_gui/p6_readmission_cycle.py [T0|T2]      (default: T0)

The library unchanged, the run's delivered inputs as in P5, noise_weights "none".  cylinder_wand.calibrate's first
start by hand: bootstrap, coarse_fit (stage 3), then refine (stage 4) with 4 rounds, its outlier step
(_update_active) wrapped to record, per round, the active set before and after, which frames had no active
detection (these are re-triangulated from all their rays before the test), and each detection's test value: the
Mahalanobis distance d against the limit (k_sigma robust sigmas; 0.8 of it to come back), recomputed exactly as the
step computes it.  For every detection that changes state twice: its frame, camera and end, and d over its limit in
both states.  Writes p6_readmission_cycle_out.txt and p6_readmission_cycle.json beside this script.
"""
import importlib.util
import json
import math
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CASES = {"T0": os.path.join(REPO, "cases", "cyl_20261006_T0_wand"),
         "T2": os.path.join(REPO, "cases", "cyl_20261006_T2_wand")}


def load_case(run):
    spec = importlib.util.spec_from_file_location(f"case_{run}", os.path.join(CASES[run], "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_values(cw, sol, active, k_sigma, floor):
    """d and the limit as _update_active computes them with equal weights, for the active set `active` and the
    solution's current frames (after its re-triangulation of the frames with no active detection)."""
    obs = sol.obs
    probe = cw.Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, active.copy())
    ev = cw.evaluate(sol.cams, sol.tube, sol.frames, sol.L, probe, cw.Layout(obs.n_cams))
    act = probe.active & ev.ok
    Jf = np.where(ev.ok[:, None, None], ev.Jf, 0.0)
    U = np.zeros((obs.n_frames, 5, 5))
    np.add.at(U, obs.frame[act], np.einsum("nai,naj->nij", Jf[act], Jf[act]))
    Uinv = np.zeros_like(U)
    good = np.linalg.det(U) > 0
    Uinv[good] = np.linalg.inv(U[good])
    Pm = np.einsum("nai,nij,nbj->nab", Jf, Uinv[obs.frame], Jf)
    M = np.eye(2)[None] + np.where(probe.active, -1.0, 1.0)[:, None, None] * Pm
    lam, V = np.linalg.eigh(M)
    lam = np.maximum(lam, 1e-3)
    rr = np.where(ev.ok[:, None], ev.r, 0.0)
    proj = np.einsum("nai,na->ni", V, rr)
    d = np.sqrt(np.sum(proj ** 2 / lam, axis=1))
    d[~ev.ok] = np.inf
    sigma = float(np.median(d[act])) / math.sqrt(2.0 * math.log(2.0))
    lim = k_sigma * max(sigma, floor)
    rnorm = np.where(ev.ok, np.linalg.norm(rr, axis=1), np.inf)
    return d, lim, rnorm, lam.min(axis=1)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "T0"
    case = load_case(run)
    cfg = case.load_config()
    lib = case.library()
    cw = lib.cw
    cams = case.cam_ids(cfg)
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    job = case.make_job(lib, cfg, frames, case.uv_array(kept, frames, cams), noise_weights="none")
    data, tube = job.data(), job.tube
    lines, records = [], []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    orig = cw._update_active

    def wrapped(sol, usable, k_sigma, sigma_floor_px=1e-3, sig_obs=None, groups=None):
        obs = sol.obs
        before = obs.active.copy()
        act_frame = np.zeros(obs.n_frames, bool)
        act_frame[obs.frame[before]] = True
        res = orig(sol, usable, k_sigma, sigma_floor_px, sig_obs=sig_obs, groups=groups)
        d, lim, rnorm, lam_min = test_values(cw, sol, before, k_sigma, sigma_floor_px)
        records.append({"before": before, "after": obs.active.copy(), "stale_frame": ~act_frame, "d": d,
                        "lim": lim, "rnorm": rnorm, "lam_min": lam_min, "result": res})
        return res
    cw._update_active = wrapped
    obs = cw.observations(data)
    layout = cw.Layout(data.n_cams, fit_k1=bool(job.fit_k1), fit_L=bool(job.fit_L))
    ranked, _binfo = cw.bootstrap(obs, tube, data.L, data.sizes, verbose=False, log=print, log_fits=False)
    cams0 = ranked[0][1]
    o = cw.Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, obs.active.copy())
    cams3, info3 = cw.coarse_fit(cams0, tube, o, data.L, layout, verbose=False, log=print)
    say(f"P6: {run}, {len(frames)} frames, {len(o)} detections; first start; stage 3 rms {info3['rms_px']:.4f} px")
    records.clear()                     # keep stage 4's rounds only
    sol = cw.refine(cams3, tube, cw.initial_frames(cams3, tube, o, data.L), data.L, o, layout, noise_weights="none",
                    max_rounds=4, verbose=True, log=lambda s: say("  |" + s) if "round" in s else None)
    say("")
    for k, rec in enumerate(records):
        n_out = int(np.sum(rec["before"] & ~rec["after"]))
        n_back = int(np.sum(~rec["before"] & rec["after"]))
        say(f"round {k}: limit {rec['lim']:.3f} (equal weights, d in px); rejected {n_out}, re-admitted {n_back}; "
            f"frames with no active detection before the test: {int(rec['stale_frame'].sum())}")
    # detections whose state changes in two consecutive rounds
    out = {"run": run, "rounds": [], "toggling": []}
    if len(records) >= 3:
        r1, r2 = records[1], records[2]
        tog = np.flatnonzero((r1["before"] != r1["after"]) & (r2["before"] != r2["after"]))
        say(f"\ndetections that change state in rounds 1 and 2: {len(tog)}")
        say("  frame  cam  end   in the fit: d / limit (smallest eigenvalue of I - P)   "
            "frame out: d / (0.8 limit), |r| px   frame stale then")
        fr = o.frame_ids if getattr(o, "frame_ids", None) is not None else None
        for i in tog:
            fit_rec = r1 if r1["before"][i] else r2           # the round that judged it while in the fit
            out_rec = r2 if r1["before"][i] else r1           # the round that judged it while out
            f = int(o.frame[i])
            fid = int(fr[f]) if fr is not None else f
            stale = bool(out_rec["stale_frame"][f])
            say(f"  {fid:5d}  {int(o.cam[i]):3d}  {'AB'[int(o.end[i])]:>3s}   "
                f"{fit_rec['d'][i] / fit_rec['lim']:6.2f}  ({fit_rec['lam_min'][i]:.3f})"
                f"                                  {out_rec['d'][i] / (0.8 * out_rec['lim']):6.2f}, "
                f"{out_rec['rnorm'][i]:.2f}               {stale}")
            out["toggling"].append({"frame": fid, "cam": int(o.cam[i]), "end": int(o.end[i]),
                                    "fit_d_over_lim": float(fit_rec["d"][i] / fit_rec["lim"]),
                                    "fit_lam_min": float(fit_rec["lam_min"][i]),
                                    "out_d_over_readmit_lim": float(out_rec["d"][i] / (0.8 * out_rec["lim"])),
                                    "out_rnorm_px": float(out_rec["rnorm"][i]), "frame_stale_when_out": stale})
        # within the toggling frames: which detection exceeds the limit in the fit (the cause) and which were only
        # dropped with it (the thin-frame rule: an end left with one camera drops the frame)
        frames_t = sorted({int(o.frame[i]) for i in tog})
        for f in frames_t:
            rows = np.flatnonzero(o.frame == f)
            fit_rec = r1 if r1["before"][rows[0]] else r2
            over = [int(i) for i in rows if fit_rec["d"][i] > fit_rec["lim"]]
            say(f"  frame {int(fr[f]) if fr is not None else f}: {len(rows)} detections; over the limit in the fit: "
                + ", ".join(f"cam {int(o.cam[i])} end {'AB'[int(o.end[i])]} (d {fit_rec['d'][i]:.2f} px)" for i in over))
    for rec in records:
        out["rounds"].append({"limit": rec["lim"], "rejected": int(np.sum(rec["before"] & ~rec["after"])),
                              "readmitted": int(np.sum(~rec["before"] & rec["after"])),
                              "stale_frames": int(rec["stale_frame"].sum())})
    with open(os.path.join(HERE, f"p6_readmission_cycle_out{'' if run == 'T0' else '_' + run}.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, f"p6_readmission_cycle{'' if run == 'T0' else '_' + run}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
