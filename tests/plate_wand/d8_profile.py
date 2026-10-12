"""D8 (diagnosis, after H3a): the cost over a weak window's normal, on the cases rough hints do not always rescue
(windows behind one long-lens camera: W5 seeds 1001, 1034 and 1045 at 0.1 px).

Questions: is the cost, as a function of that window's normal, one valley about the truth, or does it have other
minima, near the truth or far from it?  Would a scan of the normal, run from the answer calibrate gives without hints,
find the global minimum?  Per case and weak window w, the fit with w's normal held at each point (everything else
free, equal weights, outliers rejected as in stage 4), started from a neighbour already fitted:
  1. the path: from the answer's normal (moved into the truth's frame by camera 1) to the true one along the great
     circle, 1 deg at a time;
  2. the grid about the answer: normals up to 30 deg from the answer's, 3 deg apart, outwards from the answer;
  3. the grid about the truth: normals up to 12 deg from the true one, 1.5 deg apart, outwards from the
     truth-started fit;
  4. the release: the three cheapest points of each grid, and the path's end, fitted with everything free and compared
     with the truth-started fit (W5's test).
Per point: the held fit's RMS, its capped cost relative to the answer's normal held (equal weights, the truth-started
fit's scale), its active count, and each window's smallest wand-end depth in the water (near 0: pinned).
A grid point is a local minimum when its cost is below all its fitted neighbours' (8 around it).
Output: d8_profile_out.txt / .json.  The four scans run in parallel.
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
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TASKS = [(1001, 0.1, 1), (1034, 0.1, 2), (1045, 0.1, 0), (1045, 0.1, 1)]     # seed, noise, weak window (0-based)
GRIDS = {"answer": (30.0, 3.0), "truth": (12.0, 1.5)}                          # radius, spacing (deg)
PATH_STEP_DEG = 1.0


def angle_deg(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


def held_fit(pw, pm, data, cams, rig, w, n_new):
    """The fit with window w's normal held at n_new (its plane turned about its anchor), everything else free."""
    win = rig.windows[w]
    A = pw.anchors_for(cams, rig)[w]
    n_new = np.asarray(n_new, float) / np.linalg.norm(n_new)
    wn = pm.Window(n_new, float(n_new @ A) + (win.d - float(win.n @ A)), win.t, win.idx, win.name)
    rg = pw.Rig([wn if k == w else x.copy() for k, x in enumerate(rig.windows)], rig.cam_win)
    return pw.solve(data, [c.copy() for c in cams], rg, held_normals=(w,), noise_weights="none")


def record(pw, sol, obs, s02, c0):
    depths = pw.window_depths(sol.cams, sol.rig, sol.frames, sol.L, sol.obs)
    c = pw.capped_cost(sol, obs, None, s02, 5.0)
    return {"rms_px": sol.rms, "active": int(sol.obs.active.sum()), "capped_rel": c / c0 - 1.0,
            "min_depth_mm": [None if d is None else float(d) for d in depths]}


def verdict_of(prigs, sol, ref):
    cmp = prigs.same_minimum(sol, ref)
    eq = bool(cmp["same"] or sol.rms < ref.rms or (cmp["max_abs_z"] <= 3.0 and sol.rms <= 1.02 * ref.rms))
    return ("same" if cmp["same"] else "equivalent" if eq else "DIFFERENT"), cmp


def scan(pw, pm, data, obs, s02, c0, centre_fit, n_c, w, radius, step, n_true):
    """Held fits on a grid about n_c (tangent-plane angles, `step` apart, within `radius`), outwards from centre_fit.
    Returns ({(i, j): record}, {(i, j): fit})."""
    kmax = int(round(radius / step))
    pts = {(i, j) for i in range(-kmax, kmax + 1) for j in range(-kmax, kmax + 1) if i * i + j * j <= kmax * kmax}
    st = math.radians(step)
    fits = {(0, 0): centre_fit}
    grid = {(0, 0): {**record(pw, centre_fit, obs, s02, c0), "off_true_deg": angle_deg(n_c, n_true)}}
    queue, seen = deque([(0, 0)]), {(0, 0)}
    while queue:
        ij = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (ij[0] + di, ij[1] + dj)
            if nb in seen or nb not in pts:
                continue
            seen.add(nb)
            n_nb = pm.retract_normal(n_c, (nb[0] * st, nb[1] * st))
            try:
                s_nb = held_fit(pw, pm, data, fits[ij].cams, fits[ij].rig, w, n_nb)
            except (np.linalg.LinAlgError, ValueError) as exc:
                grid[nb] = {"error": repr(exc)}
                continue
            grid[nb] = {**record(pw, s_nb, obs, s02, c0), "off_true_deg": angle_deg(n_nb, n_true)}
            fits[nb] = s_nb
            queue.append(nb)
    return grid, fits


def summarise(grid):
    good = {k: v for k, v in grid.items() if "error" not in v}
    local_min = [k for k, v in good.items()
                 if all(good[(k[0] + a, k[1] + b)]["capped_rel"] > v["capped_rel"]
                        for a in (-1, 0, 1) for b in (-1, 0, 1)
                        if (a, b) != (0, 0) and (k[0] + a, k[1] + b) in good)]
    near = min(good, key=lambda k: good[k]["off_true_deg"])
    pinned = sum(1 for v in good.values() if any(d is not None and d < 1e-6 for d in v["min_depth_mm"]))
    return good, {"points": len(grid), "failed": len(grid) - len(good), "pinned_points": pinned,
                  "local_minima": sorted([{"ij": list(k), "capped_rel": good[k]["capped_rel"],
                                           "off_true_deg": good[k]["off_true_deg"], "rms_px": good[k]["rms_px"],
                                           "pinned": any(d is not None and d < 1e-6 for d in good[k]["min_depth_mm"])}
                                          for k in local_min], key=lambda m: m["capped_rel"]),
                  "nearest_truth": {"ij": list(near), **{q: good[near][q] for q in ("capped_rel", "off_true_deg",
                                                                                    "rms_px", "active")}}}


def release(pw, prigs, data, ref, fit):
    try:
        s_r = pw.solve(data, [c.copy() for c in fit.cams], fit.rig.copy())
    except (np.linalg.LinAlgError, ValueError) as exc:
        return {"error": repr(exc)}
    v, cmp = verdict_of(prigs, s_r, ref)
    return {"verdict": v, "rms_px": s_r.rms, "max_abs_z": cmp["max_abs_z"]}


def run_task(task):
    import prigs
    import plate_wand as pw
    from prigs import pm
    from h1_hints import case
    seed, noise, w = task
    t0 = time.time()
    try:
        rig_t, data = case(seed, noise)
        obs = pw.observations(data)
        truth = pw.Rig([x.copy() for x in rig_t.wins], rig_t.cam_win)
        ref = pw.solve(data, [c.copy() for c in rig_t.cams], truth)
        ref0 = pw.solve(data, [c.copy() for c in rig_t.cams], truth.copy(), noise_weights="none")
        s02 = ref0.sigma_px ** 2
        placeholder = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, x.t, x.idx, x.name) for x in rig_t.wins], rig_t.cam_win)
        sol = pw.calibrate(data, placeholder)
        Q, p = prigs.rigid_from_cam(sol.cams[0], rig_t.cams[0])
        cams_a, wins_a = prigs.transform(sol.cams, sol.rig.windows, Q, p)
        rig_a = pw.Rig(wins_a, rig_t.cam_win)
        n_ans, n_true = wins_a[w].n.copy(), rig_t.wins[w].n.copy()
        out = {"seed": seed, "noise_px": noise, "window": w, "cams": [c for c, k in enumerate(rig_t.cam_win) if k == w],
               "f_px": [c.f for c in rig_t.cams], "ref_rms_px": ref.rms, "answer_rms_px": sol.rms,
               "answer_verdict": verdict_of(prigs, sol, ref)[0], "answer_off_deg": angle_deg(n_ans, n_true),
               "answer_normal_err_deg": [angle_deg(a.n, b.n) for a, b in zip(wins_a, rig_t.wins)]}
        centre = held_fit(pw, pm, data, cams_a, rig_a, w, n_ans)
        c0 = pw.capped_cost(centre, obs, None, s02, 5.0)
        out["centre"] = record(pw, centre, obs, s02, c0)
        out["ref_capped_rel"] = pw.capped_cost(ref0, obs, None, s02, 5.0) / c0 - 1.0
        # 1. the path from the answer's normal to the true one
        total = angle_deg(n_ans, n_true)
        path = []
        if total > 1e-6:
            axis = np.cross(n_ans, n_true)
            axis /= np.linalg.norm(axis)
            prev = centre
            for k in range(1, int(math.ceil(total / PATH_STEP_DEG)) + 1):
                th = math.radians(min(k * PATH_STEP_DEG, total))
                n_k = math.cos(th) * n_ans + math.sin(th) * np.cross(axis, n_ans)
                try:
                    s_k = held_fit(pw, pm, data, prev.cams, prev.rig, w, n_k)
                except (np.linalg.LinAlgError, ValueError) as exc:
                    path.append({"deg": math.degrees(th), "error": repr(exc)})
                    continue
                path.append({**record(pw, s_k, obs, s02, c0), "deg": math.degrees(th)})
                prev = s_k
            out["path_end_release"] = release(pw, prigs, data, ref, prev)
        out["path"] = path
        # 2. and 3. the grids, about the answer's normal and about the true one
        held_truth = held_fit(pw, pm, data, [c.copy() for c in rig_t.cams], truth.copy(), w, n_true)
        out["grids"] = {}
        for name, (radius, step) in GRIDS.items():
            centre_fit, n_c = (centre, n_ans) if name == "answer" else (held_truth, n_true)
            grid, fits = scan(pw, pm, data, obs, s02, c0, centre_fit, n_c, w, radius, step, n_true)
            good, summ = summarise(grid)
            best3 = sorted(good, key=lambda k: good[k]["capped_rel"])[:3]
            summ["release"] = [{"ij": list(k), "capped_rel": good[k]["capped_rel"],
                                "off_true_deg": good[k]["off_true_deg"], **release(pw, prigs, data, ref, fits[k])}
                               for k in best3]
            summ["radius_deg"], summ["step_deg"] = radius, step
            summ["grid"] = {f"{k[0]},{k[1]}": v for k, v in grid.items()}
            out["grids"][name] = summ
        out["seconds"] = time.time() - t0
        return out
    except Exception as exc:  # noqa: BLE001
        return {"seed": seed, "window": w, "error": repr(exc), "trace": traceback.format_exc()[-2000:],
                "seconds": time.time() - t0}


def main():
    import prigs
    log = prigs.Tee(HERE / "d8_profile_out.txt")
    log("D8: the cost over a weak window's normal (W5 seeds 1001, 1034, 1045 at 0.1 px; no hints)")
    log("held fits: the window's normal held, everything else free, equal weights; cost = capped cost relative to the "
        "answer's normal held; grids: " + ", ".join(f"about the {k} {r:g} deg at {s:g} deg" for k, (r, s) in
                                                     GRIDS.items()) + f"; path {PATH_STEP_DEG:g} deg")
    t_all = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(TASKS)) as pool:
        res = pool.map(run_task, TASKS, chunksize=1)
    for r in res:
        log("")
        if "error" in r:
            log(f"seed {r['seed']} window {r['window'] + 1}: ERROR {r['error']}")
            log(r.get("trace", ""))
            continue
        f_txt = ", ".join(f"{r['f_px'][c]:.0f}" for c in r["cams"])
        log(f"seed {r['seed']} ({r['noise_px']:g} px), window {r['window'] + 1} (camera(s) "
            f"{[c + 1 for c in r['cams']]}, f {f_txt} px): the answer without hints is {r['answer_verdict']} (rms "
            f"{r['answer_rms_px']:.6f} against {r['ref_rms_px']:.6f}; normal errors "
            f"{', '.join(f'{x:.2f}' for x in r['answer_normal_err_deg'])} deg)")
        log(f"  the truth-started fit's cost: {r['ref_capped_rel']:+.3e} relative to the answer's normal held")
        good_path = [q for q in r["path"] if "error" not in q]
        if r["path"]:
            log(f"  path to the true normal ({r['answer_off_deg']:.1f} deg), cost every 1 deg: " + ", ".join(
                f"{q['capped_rel']:+.2e}" if "error" not in q else "failed" for q in r["path"]))
            if good_path:
                pk = max(good_path, key=lambda q: q["capped_rel"])
                lo = min(good_path, key=lambda q: q["capped_rel"])
                pinned = [f"{q['deg']:.0f}" for q in good_path if any(d is not None and d < 1e-6
                                                                      for d in q["min_depth_mm"])]
                log(f"    highest {pk['capped_rel']:+.2e} at {pk['deg']:.0f} deg, lowest {lo['capped_rel']:+.2e} at "
                    f"{lo['deg']:.0f} deg; pinned at {pinned or 'none'}; released from the end: "
                    f"{r['path_end_release']}")
        for name, g in r["grids"].items():
            log(f"  grid about the {name} ({g['radius_deg']:g} deg at {g['step_deg']:g}): {g['points']} normals, "
                f"{g['failed']} failed, {g['pinned_points']} pinned; {len(g['local_minima'])} local minima:")
            for m in g["local_minima"][:8]:
                log(f"    {m['off_true_deg']:5.1f} deg from the truth: {m['capped_rel']:+.3e} (rms {m['rms_px']:.6f})"
                    f"{' PINNED' if m['pinned'] else ''}")
            nt = g["nearest_truth"]
            log(f"    nearest the truth ({nt['off_true_deg']:.1f} deg from it): {nt['capped_rel']:+.3e}, rms "
                f"{nt['rms_px']:.6f}")
            for q in g["release"]:
                log("    released from " + (f"{q['off_true_deg']:.1f} deg from the truth ({q['capped_rel']:+.2e}): "
                                            f"{q['verdict']}, rms {q['rms_px']:.6f}, max |z| {q['max_abs_z']:.1e}"
                                            if "error" not in q else f"{q['ij']}: failed {q['error']}"))
        log(f"  {r['seconds']:.0f} s")
    log(f"wall time {time.time() - t_all:.0f} s")
    (HERE / "d8_profile.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
