"""W7: bad data.  The whole calibration from scratch on rig A (and B) with the data spoiled one way at a time, against
the same calibration of the clean data.

Scenarios (1,000 frames, 0.1 px):
  outliers     2 % of the detections moved 20-200 px in a random direction;
  swapped      1 % of the camera-frames with the two ends swapped (A <-> B in that camera only);
  missing      camera 2 missing from 40 % of the frames;
  noisy        camera 3 with 5x the noise (0.5 px);
  partial      cameras 1 and 2 each missing a different half of the frames, so that no frame is seen by every
               camera: the stage-1 fallback (affine start on a camera subset, the rest added by DLT resection).
Criteria: the spoiled fit's parameters against the clean fit's, in the spoiled fit's sigma: RMS within 1 or, where
data are lost, within 1.5x what losing them should cost (for data nested in the clean data each parameter moves by
z^2 = 1 - sigma_clean^2 / sigma_spoiled^2 on average), and within 3.5 at worst; every spoiled detection rejected and
at most 0.5 % of the good ones; and the same minimum as the fit started from the truth on the same data.  Runs in parallel.  Output: w7_bad_data_out.txt / .json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCENARIOS = ["clean", "outliers", "swapped", "missing", "noisy", "partial"]


def spoil(rig_t, scenario, rng):
    """Data for a scenario (the clean data's noise realisation is shared), and the corrupted detections' mask."""
    import prigs
    wand = prigs.make_wand(rig_t, 1000, np.random.default_rng(777), noise_px=0.1)
    uv = wand.uv.copy()
    bad = np.zeros(uv.shape[:3], bool)                  # (F, Nc, 2) corrupted detections
    F, Nc = uv.shape[:2]
    if scenario == "outliers":
        seen = np.argwhere(np.isfinite(uv[..., 0]))
        pick = seen[rng.choice(len(seen), int(0.02 * len(seen)), replace=False)]
        for f, c, e in pick:
            a = rng.uniform(0, 2 * math.pi)
            uv[f, c, e] += rng.uniform(20, 200) * np.array([math.cos(a), math.sin(a)])
            bad[f, c, e] = True
    elif scenario == "swapped":
        seen = np.argwhere(np.all(np.isfinite(uv[..., 0]), axis=2))
        pick = seen[rng.choice(len(seen), int(0.01 * len(seen)), replace=False)]
        for f, c in pick:
            uv[f, c] = uv[f, c, ::-1].copy()
            bad[f, c] = True
    elif scenario == "missing":
        drop = rng.random(F) < 0.4
        uv[drop, 1] = np.nan
    elif scenario == "noisy":
        uv[:, 2] = wand.clean[:, 2] + rng.normal(scale=0.5, size=uv[:, 2].shape)
    elif scenario == "partial":
        half = rng.random(F) < 0.5
        uv[half, 0] = np.nan
        uv[~half, 1] = np.nan
    return wand, uv, bad


def run_one(task):
    import prigs
    import plate_wand as pw
    rig_name, scenario = task
    rig_t = prigs.RIGS[rig_name]()
    rng = np.random.default_rng(70 + SCENARIOS.index(scenario))
    wand, uv, bad = spoil(rig_t, scenario, rng)
    data = pw.WandData(uv, wand.L, rig_t.sizes)
    placeholder = pw.Rig([prigs.pm.Window([0, 0, 1], 0.0, w.t, w.idx) for w in rig_t.wins], rig_t.cam_win)
    t0 = time.time()
    out = {"rig": rig_name, "scenario": scenario}
    try:
        sol = pw.calibrate(data, placeholder)
    except Exception as exc:  # noqa: BLE001
        import traceback
        out.update({"ok": False, "error": repr(exc), "trace": traceback.format_exc()[-2000:]})
        return out
    out["seconds"] = time.time() - t0
    obs = sol.obs
    fid = obs.frame_ids[obs.frame]
    is_bad = bad[fid, obs.cam, obs.end]
    out.update({"ok": True, "rms_px": sol.rms, "rows": len(obs), "active": int(obs.active.sum()),
                "bad_rows": int(is_bad.sum()), "bad_rejected": int(np.sum(is_bad & ~obs.active)),
                "good_rejected": int(np.sum(~is_bad & ~obs.active)),
                "subset": sol.info["bootstrap"].get("subset"),
                "added_by_dlt": [m.get("added_by_dlt") for m in sol.info["bootstrap"].get("mirrors", [])],
                "noise": sol.info.get("noise_weights", {}).get("sigma_group_px"),
                "perturbed": sol.info["starts"].get("perturbed")})
    # the solution and the truth-started reference, for the comparisons in main()
    ref_truth = pw.solve(pw.WandData(uv, wand.L, rig_t.sizes), [c.copy() for c in rig_t.cams],
                         pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    out["same_as_truth_start"] = prigs.same_minimum(sol, ref_truth)
    Q, p = prigs.rigid_from_cam(sol.cams[0], rig_t.cams[0])
    cams_a, wins_a = prigs.transform(sol.cams, sol.rig.windows, Q, p)
    out["cams"] = [{"R": c.R.tolist(), "C": c.C.tolist(), "f": c.f} for c in cams_a]
    out["wins"] = [{"n": w.n.tolist(), "d": w.d} for w in wins_a]
    out["cov"] = ref_truth.cov.tolist()
    return out


def as_cams(d, rig_t):
    import prigs
    pm = prigs.pm
    cams = []
    for c, ct in zip(d["cams"], rig_t.cams):
        k = ct.copy()
        k.R, k.C, k.f = np.array(c["R"]), np.array(c["C"]), c["f"]
        cams.append(k)
    wins = [pm.Window(np.array(w["n"]), w["d"], wt.t, wt.idx) for w, wt in zip(d["wins"], rig_t.wins)]
    return cams, wins


def main():
    import prigs
    import plate_wand as pw
    import w3_monte_carlo as w3
    log = prigs.Tee(HERE / "w7_bad_data_out.txt")
    tasks = [(r, s) for r in "AB" for s in SCENARIOS]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(tasks)) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    log(f"W7: {len(tasks)} calibrations from scratch in {time.time() - t0:.0f} s wall")
    out = {}
    ok_all = True
    for rig_name in "AB":
        rig_t = prigs.RIGS[rig_name]()
        clean = [r for r in res if r["rig"] == rig_name and r["scenario"] == "clean"][0]
        sd_clean = np.sqrt(np.diag(np.array(clean["cov"])))
        cams_c, wins_c = as_cams(clean, rig_t)
        cams0 = [c.copy() for c in rig_t.cams]
        lay = pw.Layout(rig_t.n_cams, len(rig_t.wins), anchors=pw.anchors_for(cams0, pw.Rig(rig_t.wins, rig_t.cam_win)))
        d_clean = w3.deltas(cams_c, wins_c, rig_t, lay)
        for r in res:
            if r["rig"] != rig_name:
                continue
            key = f"{rig_name}_{r['scenario']}"
            if not r.get("ok"):
                ok_all = False
                out[key] = r
                log(f"rig {rig_name}, {r['scenario']:8s}: FAILED {r.get('error')}")
                continue
            cams_s, wins_s = as_cams(r, rig_t)
            d_s = w3.deltas(cams_s, wins_s, rig_t, lay)
            sd = np.sqrt(np.diag(np.array(r["cov"])))
            z_clean = (d_s - d_clean) / sd
            z_truth = d_s / sd
            rec = {k: r.get(k) for k in ("rms_px", "rows", "active", "bad_rows", "bad_rejected", "good_rejected",
                                         "seconds", "subset", "added_by_dlt", "noise", "perturbed")}
            nested = r["scenario"] != "noisy"
            exp_z = float(np.sqrt(np.mean(np.maximum(1.0 - (sd_clean / sd) ** 2, 0.0)))) if nested else None
            rec.update({"z_vs_clean_rms": float(np.sqrt(np.mean(z_clean ** 2))),
                        "z_vs_clean_max": float(np.max(np.abs(z_clean))),
                        "z_expected_rms": exp_z,
                        "z_vs_truth_rms": float(np.sqrt(np.mean(z_truth ** 2))),
                        "same_as_truth_start": r["same_as_truth_start"]["same"]})
            lim = max(1.0, 1.5 * exp_z) if exp_z is not None else 1.0
            good = (r["scenario"] == "clean" or (rec["z_vs_clean_rms"] <= lim and rec["z_vs_clean_max"] <= 3.5))
            good &= rec["bad_rejected"] == rec["bad_rows"] and rec["good_rejected"] <= 0.005 * rec["rows"]
            good &= rec["same_as_truth_start"]
            rec["pass"] = bool(good)
            ok_all &= good
            out[key] = rec
            log(f"rig {rig_name}, {r['scenario']:8s}: rms {r['rms_px']:.4f} px, {r['active']}/{r['rows']} active; "
                f"spoiled {r['bad_rows']} rejected {r['bad_rejected']}, good rejected {r['good_rejected']}; vs the "
                f"clean fit: z RMS {rec['z_vs_clean_rms']:.2f} (expected "
                f"{'-' if exp_z is None else f'{exp_z:.2f}'}), max {rec['z_vs_clean_max']:.2f}; vs the truth: z RMS "
                f"{rec['z_vs_truth_rms']:.2f}; same minimum as from the truth: {rec['same_as_truth_start']}; "
                f"{r['seconds']:.0f} s" + (f"; stage-1 subset {r['subset']}, added by DLT {r['added_by_dlt']}"
                                           if r['scenario'] == 'partial' else "")
                + (f"; noise per camera {', '.join(f'{v:.3f}' for v in r['noise'])} px" if r['noise'] and
                   r['scenario'] == 'noisy' else "")
                + ("; the perturbed restarts improved it" if (r.get("perturbed") or {}).get("improved") else "")
                + ("" if good else "   <-- FAIL"))
    out["pass"] = bool(ok_all)
    log("W7: " + ("PASS" if ok_all else "FAIL"))
    (HERE / "w7_bad_data.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
