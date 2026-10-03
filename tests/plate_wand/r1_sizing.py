"""R1 (2026-10-02-perturbed-starts-plan.md): sizing the perturbed restarts on W5 seed 2000, before any code change.

Seed 2000 (1 px noise, two windows with two cameras each) ends 0.3 % above the global minimum when calibrated from
scratch: its joined-pairs start lands in a narrow wrong basin (d4_seed2000.py).  Draws are taken around two centres,
that wrong answer and the joined start it came from, from the answer's own covariance (camera 1's pose held), scaled
by s = 1, 2, 3, 4: dx = s L z with Sigma = L L^T and z ~ N(0, I), applied as stage 4 applies a step.  A draw that puts
a camera on the water side of its window is redrawn (up to 10 times).  The same 20 z's serve every centre and scale.
Each draw is
  screened  by one Levenberg-Marquardt fit (bundle_adjust) from it, the frames re-triangulated, at the answer's active
            observations and noise weights: its cost against the answer's on the same terms;
  fitted    by the whole final fit (`solve` on fresh observations, as calibrate's stage 4), from the screened result
            and from the draw itself: does it reach the global minimum (prigs.same_minimum against the fit started from
            the truth, W5's test)?
Reported per centre and scale: the per-draw success p, from the screened result and from the draw; the chance that the
planned rule (K draws screened, the cheapest fully fitted if it lowers the cost by more than 1e-6) finds the global
minimum, exactly from the 20 draws' ranks; the time.  Also the answer's distance from the global minimum in its own
sigma (Mahalanobis), and each draw's step along that direction as a fraction of it.
Runs in parallel (one process per draw, single-threaded).  Output: r1_sizing_out.txt / .json
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
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SEED = 2000
SCALES = (1.0, 2.0, 3.0, 4.0)
N_DRAWS = 20
DRAW_SEED = 20261002
MAX_REDRAW = 10
K_MAX = 8
REL_GAIN = 1e-6
CTX = {}


def seed_2000():
    """W5's seed 2000, exactly as w5_from_scratch.run_one builds it."""
    import prigs
    import plate_wand as pw
    rng = np.random.default_rng(SEED)
    rig_t = prigs.random_rig(rng)
    n_frames = int(rng.integers(500, 1501))
    wand = prigs.make_wand(rig_t, n_frames, rng, noise_px=1.0)
    return rig_t, pw.WandData(wand.uv, wand.L, rig_t.sizes)


def fresh(obs0, active=None):
    import plate_wand as pw
    act = obs0.active.copy() if active is None else np.asarray(active, bool).copy()
    return pw.Obs(obs0.frame, obs0.cam, obs0.end, obs0.uv, obs0.n_frames, obs0.n_cams, obs0.frame_ids, act)


def step_to(sol, other):
    """`other`'s cameras and windows, aligned on sol's camera 1, as a step dx in sol's layout:
    sol.layout.apply(sol.cams, sol.rig, L, dx) gives them back (cameras by the exp map, normals by the sphere's)."""
    import prigs
    pm = prigs.pm
    Q, p = prigs.rigid_from_cam(other.cams[0], sol.cams[0])
    cams_o, wins_o = prigs.transform(other.cams, other.rig.windows, Q, p)
    lay = sol.layout
    dx = np.zeros(lay.P)
    for c, (cs, co) in enumerate(zip(sol.cams, cams_o)):
        vals = list(pm.log_so3(cs.R.T @ co.R)) + list(co.C - cs.C) + [math.log(co.f / cs.f)]
        for j in range(7):
            g = lay.pos[c, j]
            if g >= 0:
                dx[g] = vals[j]
    for w, (ws, wo) in enumerate(zip(sol.rig.windows, wins_o)):
        a, b = prigs.sphere_log(ws.n, wo.n)
        A = lay.anchors[w]
        dl = (wo.d - float(wo.n @ A)) - (ws.d - float(ws.n @ A))
        for j, v in enumerate((a, b, dl)):
            dx[lay.wpos[w, j]] = v
    return dx, cams_o, wins_o


def init_worker(payload):
    CTX.update(payload)


def run_draw(task):
    import prigs
    import plate_wand as pw
    out = {k: task[k] for k in ("centre", "scale", "draw", "redraws", "along", "norm")}
    data, ref, lay, obs0 = CTX["data"], CTX["ref"], CTX["layout"], CTX["obs0"]
    cams_p, rig_p = task["cams"], task["rig"]
    b = None
    t0 = time.time()
    try:
        o_s = fresh(obs0, CTX["active"])
        fr = pw.initial_frames(cams_p, rig_p, o_s, data.L)
        b = pw.bundle_adjust(cams_p, rig_p, fr, data.L, o_s, lay, weights=CTX["weights"])
        cmp_b = prigs.same_minimum(b, ref)
        out["screen"] = {"cost": float(b.info["weighted_cost"]), "n_fail": int(b.info["n_fail"]), "rms": b.rms,
                         "iters": len(b.history), "max_abs_z": cmp_b["max_abs_z"], "worst": cmp_b["worst"],
                         "seconds": time.time() - t0}
    except Exception as exc:  # noqa: BLE001
        out["screen"] = {"error": repr(exc), "trace": traceback.format_exc()[-800:], "seconds": time.time() - t0}
    for key, start in (("from_screen", None if b is None else (b.cams, b.rig)), ("from_draw", (cams_p, rig_p))):
        if start is None:
            out[key] = {"same": False, "error": "no screened result"}
            continue
        t0 = time.time()
        try:
            f = pw.solve(data, [c.copy() for c in start[0]], start[1].copy(), obs=fresh(obs0))
            m = prigs.same_minimum(f, ref)
            out[key] = {"rms": f.rms, "same": m["same"], "max_abs_z": m["max_abs_z"], "worst": m["worst"],
                        "active": m["active"], "seconds": time.time() - t0}
        except Exception as exc:  # noqa: BLE001
            out[key] = {"same": False, "error": repr(exc), "seconds": time.time() - t0}
    return out


def p_rule(rr, cost_ans, K):
    """The chance that the planned rule succeeds with K of these draws taken at random: the cheapest screened draw
    among the K is fully fitted if it lowers the answer's cost by more than REL_GAIN, and it succeeds if that fit
    reaches the global minimum.  Exact over the draws' ranks: the cheapest of a random K-subset is the i-th cheapest
    with probability C(n-1-i, K-1) / C(n, K)."""
    n = len(rr)
    if K > n:
        return float("nan")
    order = sorted(rr, key=lambda r: eff_cost(r))
    succ = [bool(eff_cost(r) < (1.0 - REL_GAIN) * cost_ans and r["from_screen"].get("same")) for r in order]
    return sum(s * math.comb(n - 1 - i, K - 1) for i, s in enumerate(succ)) / math.comb(n, K)


def eff_cost(r):
    sc = r.get("screen", {})
    if "cost" not in sc or sc.get("n_fail", 1) > 0 or not np.isfinite(sc["cost"]):
        return float("inf")
    return sc["cost"]


def main():
    import prigs
    import plate_wand as pw
    pm = prigs.pm
    log = prigs.Tee(HERE / "r1_sizing_out.txt")
    t_all = time.time()
    rig_t, data = seed_2000()
    placeholder = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins], rig_t.cam_win)
    obs0 = pw.observations(data)
    log(f"R1: the perturbed restarts sized on W5 seed {SEED} ({rig_t.n_cams} cameras behind windows "
        f"{[w + 1 for w in rig_t.cam_win]}, {len(data.uv)} frames, 1 px noise); {N_DRAWS} draws per centre and scale")

    # the global minimum, and the answer from scratch (in the fit frame: camera 1's pose as its start's)
    t0 = time.time()
    ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
    log(f"  the global minimum (fit from the truth): rms {ref.rms:.6f} px, active {int(ref.obs.active.sum())}/"
        f"{len(ref.obs)}, {time.time() - t0:.0f} s")
    t0 = time.time()
    ans = pw.calibrate(data, placeholder, output_frame=False)
    m_ans = prigs.same_minimum(ans, ref)
    log(f"  calibrate from scratch (as W5): rms {ans.rms:.6f} px, active {m_ans['active']}, max |z| "
        f"{m_ans['max_abs_z']:.2f} ({m_ans['worst']}): "
        f"{'the global minimum' if m_ans['same'] else 'NOT the global minimum'}, {time.time() - t0:.0f} s; "
        f"noise weights {'used' if ans.weights is not None else 'equal'}")
    lay = ans.layout
    a0 = pw.bundle_adjust(ans.cams, ans.rig, ans.frames, data.L, fresh(obs0, ans.obs.active), lay, max_iter=0,
                          weights=ans.weights)
    cost_ans = float(a0.info["weighted_cost"])
    log(f"  the answer's cost at its own active set and weights: {cost_ans:.4f} (its stage 4 reported "
        f"{ans.info.get('weighted_cost', float('nan')):.4f}); {lay.P} free parameters")

    # the joined start it came from: the start with the same camera-1 pose (stage 4 holds it)
    t0 = time.time()
    cache, starts = {}, []
    o_b = fresh(obs0)                               # one Obs for both searches, as calibrate passes
    for sr in ("length", "fixed"):
        st, _inf = pw.bootstrap_groups(o_b, placeholder, data.L, data.sizes, search=sr, cache=cache)
        starts += [(sr, s_) for s_ in st]
    src = [(sr, s_) for sr, s_ in starts
           if np.array_equal(s_[2][0].R, ans.cams[0].R) and np.array_equal(s_[2][0].C, ans.cams[0].C)]
    if not src:
        raise RuntimeError("no start has the answer's camera-1 pose")
    sr0, (_c0, rms3, cams_j, rig_j) = src[0]
    joined = SimpleNamespace(cams=cams_j, rig=rig_j)
    log(f"  the joined start it came from: search '{sr0}' (of {len(starts)} starts), coarse rms {rms3:.4f} px, "
        f"{time.time() - t0:.0f} s")

    # distances in the answer's sigma
    try:
        Lc = np.linalg.cholesky(ans.cov)
    except np.linalg.LinAlgError:
        ev_, V_ = np.linalg.eigh(ans.cov)
        Lc = V_ * np.sqrt(np.maximum(ev_, 0.0))
    d_glob, cams_g, wins_g = step_to(ans, ref)
    cams_chk, rig_chk, _ = lay.apply(ans.cams, ans.rig, data.L, d_glob)
    chk = max(max(float(np.max(np.abs(a.R - b.R))) for a, b in zip(cams_chk, cams_g)),
              max(float(np.max(np.abs(a.C - b.C))) for a, b in zip(cams_chk, cams_g)),
              max(abs(a.f / b.f - 1.0) for a, b in zip(cams_chk, cams_g)),
              max(float(np.max(np.abs(a.n - b.n))) for a, b in zip(rig_chk.windows, wins_g)),
              max(abs(a.d - b.d) for a, b in zip(rig_chk.windows, wins_g)))
    wg = np.linalg.solve(Lc, d_glob)
    D_ans = float(np.linalg.norm(wg))
    u_ans = wg / D_ans
    d_j, _cj, _wj = step_to(ans, joined)
    wj = np.linalg.solve(Lc, d_j)
    wjg = wg - wj                                  # from the joined start to the global minimum, whitened
    D_join = float(np.linalg.norm(wjg))
    u_join = wjg / D_join
    sd = np.sqrt(np.diag(ans.cov))
    big = np.argsort(-np.abs(d_glob / sd))[:4]
    log(f"  the global minimum lies {D_ans:.2f} of the answer's sigma from it (Mahalanobis; the step reproduces it to "
        f"{chk:.1e}); largest single parameters: "
        + ", ".join(f"{lay.names[g]} {d_glob[g] / sd[g]:+.2f}" for g in big))
    log(f"  the joined start lies {float(np.linalg.norm(wj)):.2f} sigma from the answer and {D_join:.2f} from the global "
        f"minimum")

    # the screen at the global minimum itself: does its cost rank it first?
    o_g = fresh(obs0, ans.obs.active)
    fr_g = pw.initial_frames(cams_chk, rig_chk, o_g, data.L)
    bg = pw.bundle_adjust(cams_chk, rig_chk, fr_g, data.L, o_g, lay, weights=ans.weights)
    log(f"  control: the screen started at the global minimum ends at cost {bg.info['weighted_cost']:.4f} "
        f"({bg.info['weighted_cost'] / cost_ans - 1.0:+.2e} relative to the answer's), max |z| "
        f"{prigs.same_minimum(bg, ref)['max_abs_z']:.3f} from the global minimum, n_fail {bg.info['n_fail']}")

    # the draws (common random numbers: draw k uses the same z at every centre and scale, unless redrawn)
    rng = np.random.default_rng(DRAW_SEED)
    Z = rng.standard_normal((N_DRAWS, MAX_REDRAW + 1, lay.P))
    tasks, invalid = [], []
    for centre, (cams_c, rig_c), (u, D) in (("answer", (ans.cams, ans.rig), (u_ans, D_ans)),
                                            ("joined start", (cams_j, rig_j), (u_join, D_join))):
        for s in SCALES:
            for k in range(N_DRAWS):
                for a in range(MAX_REDRAW + 1):
                    cams_p, rig_p, _ = lay.apply(cams_c, rig_c, data.L, s * (Lc @ Z[k, a]))
                    if all(rig_p.windows[w].distance(cams_p[c].C) > 0 for c, w in enumerate(rig_p.cam_win) if w >= 0):
                        break
                else:
                    invalid.append((centre, s, k))
                    continue
                tasks.append({"centre": centre, "scale": s, "draw": k, "redraws": a,
                              "along": float(s * (Z[k, a] @ u) / D), "norm": float(s * np.linalg.norm(Z[k, a])),
                              "cams": cams_p, "rig": rig_p})
    n_red = sum(t["redraws"] for t in tasks)
    log(f"  {len(tasks)} draws ({n_red} redraws for a camera on the water side; {len(invalid)} given up); each draw's "
        f"step along the line to the global minimum is given below as a fraction of that distance ('along')")

    payload = {"data": data, "ref": ref, "layout": lay, "weights": ans.weights,
               "active": ans.obs.active.copy(), "obs0": obs0}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(int(os.environ.get("PLATE_WORKERS", "32")), len(tasks)), initializer=init_worker,
                  initargs=(payload,)) as pool:
        res = pool.map(run_draw, tasks, chunksize=1)
    log(f"  the draws ran in {time.time() - t0:.0f} s")

    # per draw
    log("per draw: centre, scale, draw, redraws, along, |dx| (sigma); the screen's cost relative to the answer's, its "
        "max |z| from the global minimum; the full fit from the screened result and from the draw (G = the global "
        "minimum)")
    for r in res:
        sc = r["screen"]
        c_rel = (eff_cost(r) / cost_ans - 1.0) if np.isfinite(eff_cost(r)) else float("nan")
        fs, fd = r["from_screen"], r["from_draw"]

        def tag(f):
            if "error" in f:
                return "error"
            return f"{'G' if f['same'] else '-'} {f['rms']:.5f}"
        log(f"  {r['centre']:12s} s={r['scale']:.0f} #{r['draw']:2d} red {r['redraws']} along {r['along']:+.2f} "
            f"|dx| {r['norm']:5.1f}: screen {c_rel:+.2e} (|z| {sc.get('max_abs_z', float('nan')):5.2f}, "
            f"{sc.get('iters', 0):3d} it, fail {sc.get('n_fail', -1)}, {sc.get('seconds', 0):.1f} s) | from screen "
            f"{tag(fs)} | from draw {tag(fd)}")

    # per centre and scale
    log("summary: per-draw success p (from the screened result / from the draw); the planned rule's chance with K draws "
        "(exact over the 20 draws' ranks)")
    summ = {}
    for centre in ("answer", "joined start"):
        for s in SCALES:
            rr = [r for r in res if r["centre"] == centre and r["scale"] == s]
            n = len(rr)
            p_s = sum(bool(r["from_screen"].get("same")) for r in rr) / n
            p_d = sum(bool(r["from_draw"].get("same")) for r in rr) / n
            improving = [r for r in rr if eff_cost(r) < (1.0 - REL_GAIN) * cost_ans]
            g_imp = [r for r in improving if r["from_screen"].get("same")]
            # do the global minimum's draws have the lowest screened costs?
            c_g = [eff_cost(r) for r in rr if r["from_screen"].get("same")]
            c_n = [eff_cost(r) for r in rr if not r["from_screen"].get("same")]
            ranked = bool(not c_g or not c_n or max(c_g) < min(c_n))
            pk = {K: p_rule(rr, cost_ans, K) for K in range(1, K_MAX + 1)}
            secs_s = [r["screen"].get("seconds", float("nan")) for r in rr]
            secs_f = [r["from_screen"].get("seconds", float("nan")) for r in rr]
            summ[f"{centre}|{s:g}"] = {"centre": centre, "scale": s, "n": n, "p_screen": p_s, "p_draw": p_d,
                                       "improving": len(improving), "improving_global": len(g_imp), "ranked": ranked,
                                       "p_rule": pk, "screen_seconds_median": float(np.nanmedian(secs_s)),
                                       "fit_seconds_median": float(np.nanmedian(secs_f))}
            log(f"  {centre:12s} s = {s:.0f}: p {p_s:.2f} / {p_d:.2f}; {len(improving)} draws lower the cost, "
                f"{len(g_imp)} of them the global minimum; ranked {'yes' if ranked else 'NO'}; rule "
                + " ".join(f"K{K} {pk[K]:.2f}" for K in (1, 2, 4, 6, 8))
                + f"; screen {np.nanmedian(secs_s):.1f} s, fit {np.nanmedian(secs_f):.1f} s (medians)")

    # the choice
    best_s = max(SCALES, key=lambda s: (summ[f"answer|{s:g}"]["p_screen"], -s))
    p = summ[f"answer|{best_s:g}"]["p_screen"]
    K_formula = next((K for K in range(1, K_MAX + 1) if 1.0 - (1.0 - p) ** K >= 0.95), None)
    K_rule = next((K for K in range(1, K_MAX + 1) if summ[f"answer|{best_s:g}"]["p_rule"][K] >= 0.95), None)
    stop = all(summ[f"answer|{s:g}"]["p_screen"] < 0.2 for s in SCALES)
    log(f"choice (centre: the answer): s = {best_s:g}, p = {p:.2f}; K by 1 - (1 - p)^K >= 0.95: {K_formula}; K by the "
        f"rule's exact chance >= 0.95: {K_rule}" + (" -- p < 0.2 at every scale: STOP (plan, R1)" if stop else ""))
    log(f"done, {time.time() - t_all:.0f} s")
    (HERE / "r1_sizing.json").write_text(json.dumps({
        "seed": SEED, "scales": SCALES, "n_draws": N_DRAWS, "draw_seed": DRAW_SEED,
        "ref_rms": ref.rms, "answer_rms": ans.rms, "answer_same": m_ans, "cost_answer": cost_ans,
        "distance_sigma": {"answer_to_global": D_ans, "joined_to_global": D_join,
                           "joined_to_answer": float(np.linalg.norm(wj))},
        "control_global_screen_cost": float(bg.info["weighted_cost"]),
        "summary": summ, "choice": {"scale": best_s, "p": p, "K_formula": K_formula, "K_rule": K_rule, "stop": stop},
        "invalid": invalid, "draws": res}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
