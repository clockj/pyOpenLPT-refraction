"""W5: the whole calibration from scratch on random rigs.

50 random rigs at 0.1 px and 10 at 1 px (prigs.random_rig: 2-6 cameras, 1-3 walls of a box tank, incidence 0-45 deg,
150-1,500 mm from the window, 30-300 mm clouds, f 1,500-40,000 px, 500-1,500 frames).  calibrate() gets only the
detections, the wand length, the image sizes, each window's plates (thickness and indices) and which camera looks
through which window: the windows' normals and offsets it is given are placeholders (normal +Z, d = 0).
Success: the same minimum as the fit started from the truth on the same data (prigs.same_minimum: RMS within 1e-6
relative, the same active set, every parameter within 0.01 sigma after aligning camera 1).  Reported besides: whether
the answer is statistically equivalent (every parameter within 3 sigma of that fit's, RMS within 2 %, or a lower RMS
than it), the rig's kind (one window; several, some with two or more cameras; one camera per window) and the time.
Runs in parallel (one process per rig, single-threaded).  Output: w5_from_scratch_out.txt / .json.
Usage: w5_from_scratch.py [n_low n_high]   (defaults 50 10)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run_one(task):
    import prigs
    import plate_wand as pw
    seed, noise = task
    rng = np.random.default_rng(seed)
    rig_t = prigs.random_rig(rng)
    n_frames = int(rng.integers(500, 1501))
    wand = prigs.make_wand(rig_t, n_frames, rng, noise_px=noise)
    data = pw.WandData(wand.uv, wand.L, rig_t.sizes)
    placeholder = pw.Rig([prigs.pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins],
                         rig_t.cam_win)
    inc = []
    for c, cam in enumerate(rig_t.cams):
        wn = rig_t.win_of(c)
        inc.append(math.degrees(math.acos(min(1.0, float(cam.R[2] @ wn.n)))))
    out = {"seed": seed, "noise_px": noise, "n_cams": rig_t.n_cams, "n_windows": len(rig_t.wins),
           "frames": int(len(wand.uv)), "incidence_deg": inc, "f_px": [c.f for c in rig_t.cams],
           "h1_mm": [rig_t.win_of(c).distance(cam.C) for c, cam in enumerate(rig_t.cams)],
           "cloud_mm": rig_t.info.get("cloud_mm"), "L": rig_t.L}
    t0 = time.time()
    try:
        sol = pw.calibrate(data, placeholder)
    except Exception as exc:  # noqa: BLE001
        out.update({"ok": False, "error": repr(exc), "trace": traceback.format_exc()[-1500:],
                    "seconds": time.time() - t0})
        return out
    out["seconds"] = time.time() - t0
    out["timing"] = sol.info["timing_s"]
    out["starts"] = sol.info["starts"]
    out["rms_px"] = sol.rms
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    out["ref_rms_px"] = ref.rms
    cmp = prigs.same_minimum(sol, ref)
    out.update(cmp)
    out["ok"] = cmp["same"]
    out["equivalent"] = bool(cmp["same"] or sol.rms < ref.rms
                             or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    per = [len([c for c in rig_t.cam_win if c == w]) for w in range(len(rig_t.wins))]
    out["kind"] = ("one window" if len(per) == 1 else "groups" if max(per) >= 2 else "one camera per window")
    out["cams_per_window"] = per
    return out


def main():
    n_low = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_high = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    tasks = [(1000 + k, 0.1) for k in range(n_low)] + [(2000 + k, 1.0) for k in range(n_high)]
    import prigs
    log = prigs.Tee(HERE / "w5_from_scratch_out.txt")
    log(f"W5: {n_low} random rigs at 0.1 px, {n_high} at 1 px")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(int(os.environ.get("PLATE_WORKERS", "32")), len(tasks))) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    for r in res:
        if "error" in r:
            log(f"  seed {r['seed']} ({r['noise_px']} px): ERROR {r['error']}")
            continue
        log(f"  seed {r['seed']} ({r['noise_px']} px): {r['n_cams']} cams, {r['n_windows']} win, incidence "
            f"{min(r['incidence_deg']):.0f}-{max(r['incidence_deg']):.0f} deg, f {min(r['f_px']):.0f}-"
            f"{max(r['f_px']):.0f}, {r['frames']} frames, {r['kind']} {r['cams_per_window']}: "
            f"{'same minimum' if r['ok'] else 'equivalent' if r['equivalent'] else 'DIFFERENT'} (rms "
            f"{r['rms_px']:.5f} vs {r['ref_rms_px']:.5f}, max |z| {r['max_abs_z']:.1e} {r['worst']}, active "
            f"{r['active']}/{r['active_ref']}), {r['seconds']:.1f} s")
    summ = {}
    for noise in (0.1, 1.0):
        rr = [r for r in res if r["noise_px"] == noise]
        if not rr:
            continue
        ok = [r for r in rr if r.get("ok")]
        secs = [r["seconds"] for r in rr if "seconds" in r]
        eq = [r for r in rr if r.get("equivalent")]
        kinds = {}
        for r in rr:
            k = r.get("kind", "?")
            kinds.setdefault(k, [0, 0, 0])
            kinds[k][0] += 1
            kinds[k][1] += int(bool(r.get("ok")))
            kinds[k][2] += int(bool(r.get("equivalent")))
        summ[str(noise)] = {"runs": len(rr), "same_minimum": len(ok), "equivalent": len(eq),
                            "errors": sum(1 for r in rr if "error" in r), "by_kind": kinds,
                            "seconds_median": float(np.median(secs)), "seconds_max": float(np.max(secs))}
        log(f"{noise} px: {len(ok)}/{len(rr)} reached the same minimum as the truth-started fit, {len(eq)}/{len(rr)} an "
            f"equivalent one; time median {np.median(secs):.1f} s, max {np.max(secs):.1f} s")
        for k, (n_k, s_k, e_k) in kinds.items():
            log(f"    {k}: {s_k}/{n_k} same, {e_k}/{n_k} equivalent")
    log(f"wall time {time.time() - t0:.0f} s")
    allok = all(v["same_minimum"] == v["runs"] for v in summ.values())
    log("W5: " + ("PASS" if allok else "FAIL (see the runs marked DIFFERENT / ERROR)"))
    (HERE / "w5_from_scratch.json").write_text(json.dumps({"summary": summ, "runs": res}, indent=1, default=float),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
