"""D7 (diagnosis, H2a's stop point): why the hint-held answer was accepted when it was worse.

calibrate keeps the hint-held start's answer when its capped cost, computed with the DEFAULT answer's noise weights and
scale, is lower than the default answer's.  H2a accepted three answers that were further from the global minimum than
the default one (1034 at 10 and 20 deg, 1031 at 30 deg).  For those and for four correct acceptances, both answers are
rebuilt as calibrate builds them (the default path and the hint-held path, each through `_choose`, before the
restarts) and compared:
  - per camera: RMS, active observations, the noise sigma each answer estimated;
  - the capped cost with the default answer's weights and scale (calibrate's rule), with the held answer's, and with
    equal weights at the smaller of the two scales;
  - which rule would have kept the right answer.
Output: d7_held_acceptance_out.txt / .json.  Runs in parallel (one process per case).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# (seed, noise, delta, draw, what H2a did)
CASES = [(1034, 0.1, 10.0, 1, "accepted, worse"), (1034, 0.1, 20.0, 1, "accepted, worse"),
         (1031, 0.1, 30.0, 2, "accepted, worse"), (3110, 1.0, 0.0, 0, "accepted, right"),
         (1031, 0.1, 20.0, 1, "accepted, then the restarts reached the global minimum"),
         (1045, 0.1, 5.0, 1, "accepted, then the restarts reached the global minimum"),
         (1031, 0.1, 0.0, 0, "accepted, right")]


def paths(data, rig, hints, obs):
    """calibrate's default and hint-held answers before the restarts (its make_starts and fit_starts)."""
    import plate_wand as pw
    multi = sum(1 for w in range(len(rig.windows)) if len(rig.cams_of(w)) >= 2)
    used = sum(1 for w in range(len(rig.windows)) if rig.cams_of(w))
    few = any(1 <= len(rig.cams_of(w)) <= 2 for w in range(len(rig.windows)))
    searches = ("length", "fixed") if few else ("fixed",)

    def make_starts(hold):
        if used <= 1:
            return pw.bootstrap(obs, rig, data.L, data.sizes, hints=hints, hold=hold)[0]
        per_search, cache = [], {}
        for sr in searches:
            try:
                if multi >= 1:
                    st, _ = pw.bootstrap_groups(obs, rig, data.L, data.sizes, hints=hints, search=sr, cache=cache,
                                                hold=hold)
                else:
                    st, _ = pw.bootstrap_singles(obs, rig, data.L, data.sizes, hints=hints, search=sr, hold=hold)
            except (ValueError, np.linalg.LinAlgError):
                st, _ = pw.bootstrap(obs, rig, data.L, data.sizes, hints=hints, search=sr, hold=hold)
            per_search.append(sorted(st, key=lambda s_: s_[0]))
        return [x for tup in itertools.zip_longest(*per_search) for x in tup if x is not None]

    def fit_starts(starts):
        results, stage3 = [], []
        for k, (_c, rms3, cams0, rig0) in enumerate(starts):
            if k >= 6:
                break
            if k >= 2 and min(stage3, default=0.0) <= 1.0:
                rr = [x.rms for x in results]
                if not rr or max(rr) <= (1.0 + 1e-3) * min(rr):
                    break
            o = pw.Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids,
                       obs.active.copy())
            stage3.append(rms3 if np.isfinite(rms3) else 0.0)
            try:
                results.append(pw.solve(data, cams0, rig0, obs=o))
            except (np.linalg.LinAlgError, ValueError):
                continue
        return pw._choose(results, "camera")

    return fit_starts(make_starts(False)), fit_starts(make_starts(True))


def describe(pw, sol, obs):
    per = []
    ev = pw.evaluate(sol.cams, sol.rig, sol.frames, sol.L, obs)
    r2 = np.where(ev.ok, np.sum(np.where(ev.ok[:, None], ev.r, 0.0) ** 2, axis=1), np.nan)
    sig = (sol.info.get("noise") or {}).get("sigma_px") if isinstance(sol.info.get("noise"), dict) else None
    for c in range(obs.n_cams):
        rows = (obs.cam == c) & sol.obs.active
        per.append({"cam": c + 1, "rms_px": float(np.sqrt(np.nanmean(r2[rows]))) if rows.any() else None,
                    "active": int(rows.sum())})
    return {"rms_px": sol.rms, "sigma_px": sol.sigma_px, "active": int(sol.obs.active.sum()), "per_camera": per,
            "variance_factor": sol.info.get("variance_factor"), "has_weights": sol.weights is not None,
            "camera_sigma": sig}


def run_one(task):
    import prigs
    import plate_wand as pw
    from h1_hints import case, make_hints
    seed, noise, delta, draw, what = task
    t0 = time.time()
    try:
        rig_t, data = case(seed, noise)
        rig = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins], rig_t.cam_win)
        hints = make_hints(rig_t, delta, np.random.default_rng([seed, int(delta), draw]))
        obs = pw.observations(data)
        sol_d, sol_h = paths(data, rig, hints, obs)
        ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
        out = {"seed": seed, "noise_px": noise, "delta_deg": delta, "draw": draw, "h2a": what, "ref_rms_px": ref.rms}
        for name, s in (("default", sol_d), ("held", sol_h)):
            d = describe(pw, s, obs)
            d.update(prigs.same_minimum(s, ref))
            out[name] = d
        rules = {}
        for tag, base in (("default_weights", sol_d), ("held_weights", sol_h)):
            s02 = float(base.info.get("variance_factor", 1.0)) if base.weights is not None else base.sigma_px ** 2
            cd = pw.capped_cost(sol_d, obs, base.weights, s02, 5.0)
            ch = pw.capped_cost(sol_h, obs, base.weights, s02, 5.0)
            rules[tag] = {"default": cd, "held": ch, "held_rel": ch / cd - 1.0}
        s02 = min(sol_d.sigma_px, sol_h.sigma_px) ** 2
        cd = pw.capped_cost(sol_d, obs, None, s02, 5.0)
        ch = pw.capped_cost(sol_h, obs, None, s02, 5.0)
        rules["equal_weights"] = {"default": cd, "held": ch, "held_rel": ch / cd - 1.0}
        out["rules"] = rules
        out["seconds"] = time.time() - t0
        return out
    except Exception as exc:  # noqa: BLE001
        return {"seed": seed, "delta_deg": delta, "draw": draw, "error": repr(exc),
                "trace": traceback.format_exc()[-1500:]}


def main():
    import prigs
    log = prigs.Tee(HERE / "d7_held_acceptance_out.txt")
    log("D7: the hint-held answer against the default one, under three ways of weighting their residuals")
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(CASES)) as pool:
        res = pool.map(run_one, CASES, chunksize=1)
    for r in res:
        log("")
        if "error" in r:
            log(f"seed {r['seed']} delta {r['delta_deg']} draw {r['draw']}: ERROR {r['error']}")
            continue
        log(f"seed {r['seed']} ({r['noise_px']:g} px), hints {r['delta_deg']:g} deg off, draw {r['draw']} (H2a: "
            f"{r['h2a']}); truth-started fit rms {r['ref_rms_px']:.6f} px")
        for name in ("default", "held"):
            d = r[name]
            v = "the same minimum" if d["same"] else f"different (max |z| {d['max_abs_z']:.1e})"
            pc = ", ".join(f"cam {p['cam']} {p['rms_px']:.3f}" if p["rms_px"] is not None else f"cam {p['cam']} -"
                           for p in d["per_camera"])
            log(f"  {name:7s}: rms {d['rms_px']:.6f} px, sigma {d['sigma_px']:.4f}, active {d['active']}, {v}; "
                f"per camera {pc}")
        right = "held" if (r["held"]["rms_px"] < r["default"]["rms_px"] and not r["default"]["same"]) or \
            r["held"]["same"] else "default"
        for tag, v in r["rules"].items():
            pick = "held" if v["held"] < (1 - 1e-6) * v["default"] else "default"
            log(f"  capped cost, {tag:15s}: held {v['held_rel']:+.2e} relative -> keeps the {pick} answer"
                f"{'' if pick == right else '  (WRONG)'}")
        both = all(v["held"] < (1 - 1e-6) * v["default"] for k, v in r["rules"].items() if k != "equal_weights")
        log(f"  better on both answers' weights: {'held' if both else 'default'}"
            f"{'' if ('held' if both else 'default') == right else '  (WRONG)'}  [the right answer: {right}]")
    (HERE / "d7_held_acceptance.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
