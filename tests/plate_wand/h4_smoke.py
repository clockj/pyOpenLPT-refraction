"""H4 smoke test (before H4a): the scan of weak windows' normals on W5 seeds 1001, 1034 and 1045 (0.1 px), without
hints, calibrate run verbose.  Per case: the weak windows, the scan's grid and releases, the answer's verdict against
the truth-started fit, and the time.  Output: h4_smoke_out.txt (summary), h4_smoke_<seed>_log.txt (calibrate's log).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CASES = [(1001, 0.1), (1034, 0.1), (1045, 0.1)]


def run_one(task):
    import prigs
    import plate_wand as pw
    from h1_hints import case
    seed, noise = task
    rig_t, data = case(seed, noise)
    placeholder = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                         rig_t.cam_win)
    lines = []

    def log(msg=""):
        lines.append(str(msg))
    t0 = time.time()
    try:
        sol = pw.calibrate(data, placeholder, verbose=True, log=log)
    except Exception as exc:  # noqa: BLE001
        (HERE / f"h4_smoke_{seed}_log.txt").write_text("\n".join(lines), encoding="utf-8")
        return {"seed": seed, "error": repr(exc), "trace": traceback.format_exc()[-1500:]}
    (HERE / f"h4_smoke_{seed}_log.txt").write_text("\n".join(lines), encoding="utf-8")
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    cmp = prigs.same_minimum(sol, ref)
    eq = bool(cmp["same"] or sol.rms < ref.rms or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    return {"seed": seed, "noise_px": noise, "rms_px": sol.rms, "ref_rms_px": ref.rms,
            "verdict": "same" if cmp["same"] else "equivalent" if eq else "DIFFERENT", "max_abs_z": cmp["max_abs_z"],
            "scan": sol.info.get("scan"), "timing": sol.info["timing_s"], "seconds": time.time() - t0}


def main():
    import prigs
    log = prigs.Tee(HERE / "h4_smoke_out.txt")
    log("H4 smoke test: the scan of weak windows' normals, W5 seeds 1001, 1034, 1045 at 0.1 px, no hints")
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(CASES)) as pool:
        res = pool.map(run_one, CASES, chunksize=1)
    for r in res:
        if "error" in r:
            log(f"seed {r['seed']}: ERROR {r['error']}\n{r['trace']}")
            continue
        sc = r["scan"] or {}
        log(f"seed {r['seed']}: {r['verdict']} (rms {r['rms_px']:.6f} against {r['ref_rms_px']:.6f}, max |z| "
            f"{r['max_abs_z']:.1e}); scan {'IMPROVED' if sc.get('improved') else 'no change'} "
            f"({sc.get('seconds', 0):.0f} s); total {r['seconds']:.0f} s")
        for s in sc.get("windows") or []:
            log(f"  window {s['window'] + 1}: {s.get('fitted')} of {s.get('points')} fitted, minima "
                + ", ".join(f"{m['deg']:.1f} deg {m['cost_rel']:+.2e}" for m in (s.get("minima") or [])[:5])
                + "; released " + ", ".join(f"rms {q['rms_px']:.5f} (capped {q['capped_rel']:+.1e}, apart "
                                            f"{q['apart_chi2']:.1e})" for q in s.get("released") or [])
                + f"; accepted {s.get('accepted')}; {s.get('seconds', 0):.0f} s")
    (HERE / "h4_smoke.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
