"""C5: the projection comparison of the paper's Section 9.2.1 and Supplement S8, every method in C++.

Methods (module projbench, built by build.py): our search, pyOpenLPT's C++ core (RefractCurved.cpp, read-only; C2:
the same rays as OpenLPT's extension to the last bit), and the C++ ports of the iterative methods of
code/paper_baselines.py (C3: the same runs as the Python to the last bit). Every call is timed in C++.

The protocol is that of paper_baselines.py (evidence E1 of the paper), imported from it and unchanged:
- point sets: point_set() (uniform, shadow edge, two-image band, steep tilts, near the axis) with the counts of PLAN,
  and for the maps a 40 x 50 grid of the plane z = 0 of the reference window;
- the images of each point (the ground truth): our rays confirmed without the closed forms (truth(): the traced
  census, or the traced line offset), plus every ray our list lacks that the endpoint bracket (endpoint_rays()) or a
  baseline's answer finds and that survives the Newton polish (polish()) and the vector-Snell re-trace (landing());
- the score of every returned ray: its vector-Snell re-trace (landing()), correct within TOL = 1e-9 m;
- the outcome codes per point and method (every image, one of several, no answer, a false image);
- the multi-start: start 1 the paper's, starts 2..K rays from ms_starts(), every converged answer re-traced (in C++,
  timed with the solve; C4) and kept within TOL, the kept rays placed among the images untimed.

Timing: per point, every method in turn (ours, AFRT, GN, LM, shooting), PASSES passes; the time of a call is its
fastest over the passes (the run is at below-normal priority, so that the other work on the machine keeps it; a call
the scheduler interrupts is slow in one pass only). The multi-start is timed in MS_PASSES passes the same way. The
results must be the same in every pass. One thread.

    PYTHONDONTWRITEBYTECODE=1 C:\\Users\\zcloc\\.conda\\envs\\refract\\python.exe run_comparison.py [--quick] [--only ref]

Writes run_comparison_out.txt (every number, each line keyed as the paper's E1 output is), run_comparison.json (the
summary numbers) and results/comparison_data.npz (every per-point outcome, for the figure).
"""
import argparse
import math
import time

import numpy as np

from common import HERE, Log, SCENES, args, cpp, e1

BASELINES, METHODS = e1.BASELINES, e1.METHODS
ALL, SOME, NONE_VIS, NONE_INV, FALSE, UNMATCHED = e1.ALL, e1.SOME, e1.NONE_VIS, e1.NONE_INV, e1.FALSE, e1.UNMATCHED
TOL, STATUS, SAME_RAY = e1.TOL, e1.STATUS, e1.SAME_RAY
STATUS_NAME = {v: k for k, v in STATUS.items()}
NAMES = {"ours": "ours", "afrt": "AFRT", "gn": "dewarp GN", "lm": "dewarp LM", "shoot": "shooting LM"}

# (scene, set): (points in the full run, in --quick, seed); the map is a grid (no seed)
PLAN = {
    ("ref", "uniform"): (2000, 20, 20261101), ("ref", "edge"): (500, 8, 20261102), ("ref", "band"): (500, 6, 20261103),
    ("ref", "steep"): (500, 6, 20261104), ("ref", "axis"): (500, 6, 20261105), ("ref", "map"): (2000, 20, None),
    ("pa", "uniform"): (2000, 20, 20261111), ("pa", "steep"): (500, 6, 20261114), ("pa", "axis"): (500, 6, 20261115),
    ("sph", "uniform"): (2000, 20, 20261121), ("sph", "edge"): (500, 8, 20261122), ("sph", "band"): (500, 6, 20261123),
    ("sph", "axis"): (500, 6, 20261125),
}
MS_SETS = {"ref": ("uniform", "edge", "band", "steep", "axis"), "sph": ("uniform", "edge", "band", "axis")}
GRID = (40, 50)        # the map: radii 60..0.999 R_N, azimuths 40..179 deg
PASSES = 5
MS_PASSES = 3
K_FULL, K_QUICK = 256, 8


def grid_points(sc, nr, na):
    RN = sc.R[-1]
    return [e1._cyl_point(rX, dphi, 0.0) for rX in np.linspace(0.060, 0.999 * RN, nr)
            for dphi in np.radians(np.linspace(40.0, 179.0, na))]


def ours_rays(sc, raw):
    """The module's rays (branch, Lambda, C_z, direction) in the form of paper_baselines.ours(): (branch, sin alpha,
    C_z, direction); a sphere's ring (a point on its axial line, none in these sets) is counted and left out."""
    out, rings = [], 0
    for b, L, C, d in raw:
        if b == "ring":
            rings += 1
            continue
        if sc.cyl:
            out.append((b, L / (math.sqrt(sc.n[0] ** 2 - C * C) * sc.rP), C, np.asarray(d, float)))
        else:
            out.append((b, L / (sc.n[0] * sc.rP), 0.0, np.asarray(d, float)))
    return out, rings


def same_ray(sc):
    return lambda a, b: a[0] == b[0] and abs(a[1] - b[1]) < 1e-6 * sc.R[0] / sc.rP and abs(a[2] - b[2]) < 1e-6


def baseline_out(s, k, i):
    """One baseline's single-start result at point i of a single_start() output."""
    st = STATUS_NAME[int(s["status"][k, i])]
    d = np.array(s["d"][k, i]) if s["has_d"][k, i] else None
    answers = [d] if st in ("zero", "stationary") and d is not None else []
    return answers, st, int(s["ev"][k, i]), int(s["it"][k, i]), int(s["evj"][k, i]), d


def evaluate(sc, X, raw, s, i, t_ours, t_base):
    """paper_baselines.evaluate() for one point, with the C++ results: our rays (raw) and the baselines' runs (single
    start output s, row i), timed in C++ (t_ours, t_base: seconds, median over the passes)."""
    rays, rings = ours_rays(sc, raw)
    how, m, lost, extra = e1.truth(sc, X, rays)
    tmiss = [e1.landing(sc, X, d) for _, _, _, d in rays]
    good = [r for r, (lm_, _, _) in zip(rays, tmiss) if lm_ <= TOL]
    same = same_ray(sc)
    added = []
    for r in e1.endpoint_rays(sc, X, good):
        if e1.landing(sc, X, r[3])[0] <= TOL and not any(same(r, g) for g in good):
            added.append((r, "endpoint"))
    out = {"ours": ([d for *_, d in rays], "list", 0, 0, t_ours, 0, None)}
    for k, name in enumerate(BASELINES):
        answers, st, ev, it, evj, d = baseline_out(s, k, i)
        out[name] = (answers, st, ev, it, t_base[k], evj, d)
    truth_rays = list(good) + [r for r, _ in added]
    for name in BASELINES:
        for d in out[name][0]:
            line, _, far = e1.landing(sc, X, d)
            if line > TOL or any(e1._angle(d, g[3]) <= SAME_RAY for g in truth_rays):
                continue
            r = e1.polish(sc, X, d, far)
            if r is None or e1.landing(sc, X, r[3])[0] > TOL or any(same(r, g) for g in truth_rays):
                continue
            truth_rays.append(r)
            added.append((r, name))
    wrapped = [e1._sweep(sc, X, r) > math.pi for r in truth_rays]
    rec = {"X": X, "n_true": len(truth_rays), "n_ours": len(rays), "rings": rings, "dense_how": how, "dense_m": m,
           "dense_agree": how is not None, "dense_lost": len(lost), "dense_extra": len(extra),
           "true_far": sum(r[0] == "far" for r in truth_rays), "true_wrap": sum(wrapped),
           "added_endpoint": sum(w == "endpoint" for _, w in added), "added_by": [w for _, w in added if w != "endpoint"],
           "added_grazing": [e1._from_grazing(sc, X, r) for r, _ in added],
           "added_depth": [e1._reach_depth(sc, X, r) for r, _ in added],
           "ours_line": max([t[0] for t in tmiss], default=0.0), "ours_cross": max([t[1] for t in tmiss], default=0.0),
           "truth_rays": truth_rays}
    for name in METHODS:
        answers, status, ev, it, dt, evj, dlast = out[name]
        found, n_false, unmatched, worst, fmiss, misses = set(), 0, 0, 0.0, math.inf, []
        last = math.nan
        if name != "ours" and status in ("cap", "breakdown") and dlast is not None:
            last = e1.landing(sc, X, dlast)[0]
        for d in answers:
            line, _, far = e1.landing(sc, X, d)
            misses.append(line)
            if line > TOL:
                n_false += 1
                fmiss = min(fmiss, line)
                continue
            worst = max(worst, line)
            k = min(range(len(truth_rays)), key=lambda q: e1._angle(d, truth_rays[q][3])) if truth_rays else None
            if k is not None and e1._angle(d, truth_rays[k][3]) <= SAME_RAY:
                found.add(k)
                continue
            r = e1.polish(sc, X, d, far) if name != "ours" else None
            k = next((q for q, g in enumerate(truth_rays) if r is not None and same(r, g)), None)
            if k is None:
                unmatched += 1
            else:
                found.add(k)
        n_img = len(truth_rays)
        if n_false:
            code = FALSE
        elif unmatched:
            code = UNMATCHED
        elif not answers:
            code = NONE_VIS if n_img else NONE_INV
        else:
            code = ALL if len(found) == n_img else SOME
        rec[name] = {"status": status, "ev": ev, "evj": evj, "it": it, "t": dt, "code": code, "found": len(found),
                     "false": n_false, "unmatched": unmatched, "miss": worst,
                     "far": sum(truth_rays[q][0] == "far" for q in found), "wrap": sum(wrapped[q] for q in found),
                     "last": last, "fmiss": fmiss, "misses": misses}
    return rec


def run_set(sc, kind, count, seed, say):
    """The points of one set, their images and every method's single-start outcome, timed."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    pts, pre = [], []                # the points kept, and their (raw rays, single-start output, row) before timing
    drawn = 0
    while len(pts) < count:
        if kind == "map":
            batch = grid_points(sc, *GRID) if count >= GRID[0] * GRID[1] else grid_points(sc, 4, 5)
        else:
            need = count - len(pts)
            batch = e1.point_set(sc, kind, need if kind != "band" else max(4, need // 2), rng)
        Xb = np.array(batch, float)
        s1 = cpp.single_start(*args(sc), Xb, 1)
        for i, X in enumerate(batch):
            drawn += 1
            if kind == "band":       # keep the points with two or more images, as paper_baselines.run_set()
                rec = evaluate(sc, tuple(X), s1["ours"][i], s1, i, 0.0, [0.0] * 4)
                if rec["n_true"] < 2:
                    continue
            pts.append(tuple(X))
            pre.append((s1["ours"][i], {k: np.array(v[:, i:i + 1]) if isinstance(v, np.ndarray) and v.ndim >= 2
                                        and v.shape[1] == len(batch) else v for k, v in s1.items()}))
            if len(pts) == count:
                break
        if kind == "map":
            break
    X = np.array(pts, float)
    s = cpp.single_start(*args(sc), X, PASSES)        # the timed run: every method per point in turn, PASSES passes
    t_ours = np.min(s["ours_t_ns"], axis=0) / 1e9       # each call's fastest pass
    t_base = np.min(s["t_ns"], axis=0) / 1e9            # (4, m)
    pass_med = [float(np.median(s["ours_t_ns"][p])) / 1e3 for p in range(PASSES)]
    changed = sum(not _same_single(pre[i], s, i) for i in range(len(pts)))
    recs = [evaluate(sc, pts[i], s["ours"][i], s, i, float(t_ours[i]), [float(t_base[k, i]) for k in range(4)])
            for i in range(len(pts))]
    say(f"[9.2] {sc.key}/{kind}: {len(recs)} points (drawn {drawn}); single start timed in {PASSES} passes (results "
        f"differ between passes at {s['passes_differ']} calls; differ from the untimed run at {changed} points; our "
        f"median per pass " + "/".join(f"{v:.1f}" for v in pass_med) + f" us); {time.time() - t0:.0f} s for this set")
    return recs, drawn


def _same_single(pre, s, i):
    raw, s1 = pre
    if len(raw) != len(s["ours"][i]) or any(a[0] != b[0] or a[1] != b[1] or a[2] != b[2] for a, b in
                                            zip(raw, s["ours"][i])):
        return False
    return all(np.array_equal(s1[f][:, 0], s[f][:, i]) for f in ("status", "ev", "it", "evj", "has_d")) and \
        np.array_equal(s1["d"][:, 0], s["d"][:, i])


# ---------------------------------------------------------------------------------------------- printing (in us)
def pct(k, m):
    return 100.0 * k / m if m else float("nan")


def summarize(label, recs, drawn, say):
    """paper_baselines.summarize(), with times in microseconds; our search's closed-form evaluations are not counted
    in C++ (the C++ core is compiled unmodified)."""
    N = len(recs)
    nt = np.array([r["n_true"] for r in recs])
    hist = [int(np.sum(np.minimum(nt, 3) == k)) for k in range(4)]
    say(f"[9.2] {label}: {N} points (drawn {drawn}); images 0/1/2/3+: {hist[0]}/{hist[1]}/{hist[2]}/{hist[3]}; "
        f"points with a far-branch image {sum(r['true_far'] > 0 for r in recs)}, with an image the long way round "
        f"(sweep > pi) {sum(r['true_wrap'] > 0 for r in recs)}; sphere rings left out {sum(r['rings'] for r in recs)}")
    how = [r["dense_how"] for r in recs]
    cen = {m: sum(r["dense_how"] == "census" and r["dense_m"] == m for r in recs) for m in (12, 24, 48)}
    lin = {m: sum(r["dense_how"] == "line" and r["dense_m"] == m for r in recs) for m in (12, 24, 48)}
    say(f"[9.2] {label}: ground truth confirmed without the closed forms at {N - how.count(None)}/{N} points: by the "
        f"traced census at {how.count('census')} (12/24/48 starts: {cen[12]}/{cen[24]}/{cen[48]}), by the traced line "
        f"offset at {how.count('line')} ({lin[12]}/{lin[24]}/{lin[48]}); unconfirmed {how.count(None)} (rays of ours "
        f"not found {sum(r['dense_lost'] for r in recs if r['dense_how'] is None)}, traced rays ours lacks "
        f"{sum(r['dense_extra'] for r in recs if r['dense_how'] is None)})")
    ep = sum(r["added_endpoint"] for r in recs)
    by = [w for r in recs for w in r["added_by"]]
    gz = [g for r in recs for g in r["added_grazing"]]
    say(f"[9.2] {label}: images our list lacks, each polished by Newton and re-traced onto X: {ep + len(by)} at "
        f"{sum(r['added_endpoint'] + len(r['added_by']) > 0 for r in recs)} points; {ep} from the endpoint bracket, "
        f"{len(by)} found only by a baseline"
        + (f" ({', '.join(f'{NAMES[w]} {by.count(w)}' for w in sorted(set(by)))})" if by else "")
        + (f"; 1 - Lambda/Lmax of these rays {min(gz):.1e} to {max(gz):.1e}" if gz else ""))
    cross = [r["ours_cross"] for r in recs if r["n_ours"] and np.isfinite(r["ours_cross"])]
    say(f"[9.2] {label}: our rays re-traced: worst line miss {max(r['ours_line'] for r in recs):.1e} m"
        + (f", worst crossing miss {max(cross):.1e} m" if cross else ""))
    out = {}
    for name in METHODS:
        s = e1.stats(recs, name)
        out[name] = s
        say(f"[9.2] {label} | {NAMES[name]:11s} | of {s['vis']} points with an image: (i) >=1 correct "
            f"{pct(s['i1'], s['vis']):5.1f}%, (ii) all images {pct(s['i2'], s['vis']):5.1f}%, one of several {s['some']}, "
            f"(iv) no answer {pct(s['i4v'], s['vis']):5.1f}% | of all {N}: (iii) false image {pct(s['i3'], N):5.1f}%, "
            f"(iv) no answer {pct(s['i4'], N):5.1f}% | of {s['inv']} without an image: no answer {s['i4i']}"
            + (" (the empty list)" if name == "ours" else "") + f", false image {s['i3i']} | correct rays "
            f"not in our list {s['um']} | exits {s['status']}")
        if s["i3"]:
            fm = np.array(s["fmiss"]) * 1e3
            say(f"[9.2] {label} | {NAMES[name]:11s} | false images: {s['i3v']} at points with an image, {s['i3i']} at "
                f"points without; they pass X at {fm.min():.2g} to {fm.max():.2g} mm (median {np.median(fm):.2g} mm)")
        misses = np.array([m for r in recs for m in r[name]["misses"] if m <= TOL])
        acc = (f"landing miss of its correct rays: median {np.median(misses):.1e}, max {misses.max():.1e} m"
               if misses.size else "no correct ray")
        cost = ("C++ core, evaluations not counted" if name == "ours" else
                f"ray-path evaluations per point median {s['ev_med']:.0f} mean {s['ev_mean']:.1f} (without FD Jacobians "
                f"{s['evnj_mean']:.1f}; iterations {s['it_mean']:.1f})")
        say(f"[9.2] {label} | {NAMES[name]:11s} | {cost} | us per point median {1e3 * s['t_med']:.2f} p95 "
            f"{1e3 * s['t_p95']:.2f}, per image found {1e3 * s['t_img']:.2f} | {acc} | far-branch images {s['far']}, "
            f"long-way-round images {s['wrap']}")
    return out


# ---------------------------------------------------------------------------------------------- multi-start
def multistart(sc, recs, K):
    """paper_baselines.multistart() for every point of a set, the solves and re-traces in C++ (timed there), the
    placement of the kept rays among the images in Python (untimed). Returns per point {name: per-start arrays}."""
    X = np.array([r["X"] for r in recs], float)
    starts = np.array([e1.ms_starts(sc, tuple(x), K - 1) for x in X], float).reshape(len(X), K - 1, 3)
    o = cpp.multistart(*args(sc), X, starts)
    for _ in range(MS_PASSES - 1):                     # each start's fastest pass; the runs must not change
        o2 = cpp.multistart(*args(sc), X, starts)
        for f in ("status", "ev", "it", "traced", "kept", "d"):
            if not np.array_equal(o[f], o2[f]):
                raise SystemExit(f"multi-start results changed between passes ({f})")
        o["t_ns"] = np.minimum(o["t_ns"], o2["t_ns"])
        del o2
    same = same_ray(sc)
    runs = []
    for i, rec in enumerate(recs):
        truth = rec["truth_rays"]
        run = {}
        for k, name in enumerate(BASELINES):
            img, new = np.full(K, -1, np.int8), np.zeros(K, bool)
            listed, placed = [], []
            for j in np.flatnonzero(o["kept"][k, i]):
                d = o["d"][k, i, j]
                new[j] = not any(e1._angle(d, q) <= SAME_RAY for q in listed)
                if new[j]:
                    listed.append(d)
                ang = [e1._angle(d, g[3]) for g in truth]
                if ang and min(ang) <= SAME_RAY:
                    img[j] = int(np.argmin(ang))
                    continue
                c = next((c for q, c in placed if e1._angle(d, q) <= SAME_RAY), None)
                if c is None:
                    r = e1.polish(sc, rec["X"], d, e1.landing(sc, rec["X"], d)[2])
                    c = next((q for q, g in enumerate(truth) if r is not None and same(r, g)), -2)
                    placed.append((d, c))
                img[j] = c
            run[name] = dict(code=o["status"][k, i].astype(np.int8), ev=o["ev"][k, i].astype(np.int32),
                             traced=o["traced"][k, i].astype(bool), it=o["it"][k, i].astype(np.int32),
                             t=o["t_ns"][k, i] / 1e9, img=img, new=new)
        runs.append(run)
    return runs


def ms_summarize(label, recs, runs, tal, ours, say):
    """paper_baselines.ms_summarize(), with times in microseconds."""
    K = runs[0][BASELINES[0]]["t"].size
    target = ours["i2"] / ours["vis"] if ours["vis"] else 1.0
    for name in BASELINES:
        for k in e1.ms_grid(K):
            s = e1.ms_stats(recs, tal[name], k)
            say(f"[9.2] multi-start {label} | {NAMES[name]:11s} | K = {k:3d} | of {s['vis']} points with an image: "
                f"(ii) all images {pct(s['i2'], s['vis']):5.1f}%, (i) >=1 correct {pct(s['i1'], s['vis']):5.1f}%, silent "
                f"{pct(s['sv'], s['vis']):5.1f}% | of {s['inv']} without: silent {s['si']} | ray paths per point median "
                f"{s['ev_med']:.0f} | us per point median {1e3 * s['t_med']:.2f} p95 {1e3 * s['t_p95']:.2f}")
        s = e1.ms_stats(recs, tal[name], K)
        code = np.concatenate([run[name]["code"][1:] for run in runs])
        tt = np.concatenate([run[name]["t"][1:] for run in runs])
        say(f"[9.2] multi-start {label} | {NAMES[name]:11s} | starts 2..{K}: exits "
            + ", ".join(f"{st} {pct(np.sum(code == c), code.size):.1f}%" for st, c in STATUS.items() if st != "list")
            + f"; us per start mean {1e6 * tt.mean() if tt.size else math.nan:.2f}, the capped ones "
            f"{pct(tt[code == STATUS['cap']].sum(), tt.sum()):.0f}% of the time | at K = {K}: kept rays that are no image "
            f"{s['um']}, points whose list differs from the images found {s['dup']}")
        line = []
        for share, what in ((target, f"ours ({100 * target:.1f}%)"), (0.99, "99%")):
            k = e1.ms_first(recs, tal[name], share)
            if k is None:
                line.append(f"{what} not within K = {K}")
            else:
                s = e1.ms_stats(recs, tal[name], k)
                line.append(f"{what} at K = {k} (us per point median {1e3 * s['t_med']:.2f} p95 {1e3 * s['t_p95']:.2f}, "
                            f"ray paths median {s['ev_med']:.0f})")
        say(f"[9.2] multi-start {label} | {NAMES[name]:11s} | (ii) all images first reaches " + "; ".join(line))
    say(f"[9.2] multi-start {label} | {NAMES['ours']:11s} | (ii) all images {pct(ours['i2'], ours['vis']):5.1f}% | us per "
        f"point median {1e3 * ours['t_med']:.2f} p95 {1e3 * ours['t_p95']:.2f} (one call, no start)")


def run_multistart(data, K, say):
    ms = {}
    for key, kinds in MS_SETS.items():
        sc = SCENES[key]
        for kind in kinds:
            recs = data.get((key, kind))
            if not recs:
                continue
            t0 = time.time()
            runs = multistart(sc, recs, K)
            tal = {name: e1.ms_tally(recs, runs, name) for name in BASELINES}
            ms[(key, kind)] = (runs, tal)
            e1.ms_check(f"{key}/{kind}", sc, recs, runs, lambda m: say(m.replace("[8.4]", "[9.2]")))
            nt = np.array([r["n_true"] for r in recs])
            vis = nt >= 1
            for name in BASELINES if vis.any() else ():
                every = tal[name]["found"][vis] == nt[vis, None]
                say(f"[9.2] multi-start {key}/{kind} | {NAMES[name]:11s} | (ii) all images at K = "
                    + "/".join(str(k) for k in e1.ms_grid(K)) + ": "
                    + "/".join(f"{100.0 * every[:, k - 1].mean():.1f}" for k in e1.ms_grid(K))
                    + f" % of {int(vis.sum())} points with an image")
            say(f"[9.2] multi-start {key}/{kind}: {len(recs)} points x {K} starts x {len(BASELINES)} baselines; "
                f"{time.time() - t0:.0f} s for this set")
        for sets, what in ((kinds[:1], "uniform"), (kinds[1:], "stratified sets pooled")):
            parts = [(data[(key, k)], ms[(key, k)]) for k in sets if (key, k) in ms]
            if not parts:
                continue
            recs = [r for p in parts for r in p[0]]
            runs = [r for p in parts for r in p[1][0]]
            ms_summarize(f"{key}/{what}", recs, runs, e1.ms_pool([p[1][1] for p in parts]), e1.stats(recs, "ours"), say)
    return ms


# ---------------------------------------------------------------------------------------------- saving
def save(data, ms, path):
    """paper_baselines.save()'s arrays, plus what the new figure needs: the number of rays in our list, and the
    landing miss of every returned ray (ours: one entry per ray, with its point's index; a baseline: its answer's miss,
    NaN without an answer)."""
    e1.save(data, ms, path)
    z = dict(np.load(path))
    for (key, kind), recs in data.items():
        p = f"{key}/{kind}/"
        z[p + "ours/n_list"] = np.array([r["n_ours"] for r in recs], int)
        idx = [(i, m) for i, r in enumerate(recs) for m in r["ours"]["misses"]]
        z[p + "ours/miss_point"] = np.array([i for i, _ in idx], int)
        z[p + "ours/miss"] = np.array([m for _, m in idx], float)
        for name in BASELINES:
            z[p + name + "/miss"] = np.array([r[name]["misses"][0] if r[name]["misses"] else np.nan for r in recs])
    np.savez_compressed(path, **z)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="smoke run: a few points per set, K = 8")
    ap.add_argument("--only", default=None, help="comma-separated scene keys (ref, pa, sph)")
    a = ap.parse_args()
    log = Log("run_comparison_quick" if a.quick else "run_comparison")
    keys = a.only.split(",") if a.only else list(SCENES)
    K = K_QUICK if a.quick else K_FULL
    log(f"[9.2] settings: TOL = {TOL:g} m; AFRT k_max = {e1.AFRT_KMAX}; GN {e1.GN_KMAX} iterations; LM {e1.LM_KMAX} "
        f"trial steps, mu_0 = {e1.LM_TAU:g} max diag(J^T J); gtol = {e1.GTOL:g}; xtol = {e1.XTOL:g}; ours: "
        f"RefractCurved projectCylinder/projectSphere, tol 1e-9; each call's fastest of {PASSES} timing passes (multi-start: "
        f"of {MS_PASSES}); multi-start K = {K}")
    data, drawn = {}, {}
    for (key, kind), (full, quick, seed) in PLAN.items():
        if key not in keys:
            continue
        sc = SCENES[key]
        recs, dr = run_set(sc, kind, quick if a.quick else full, seed, log)
        data[(key, kind)], drawn[(key, kind)] = recs, dr
        summarize(f"{key}/{kind}", recs, dr, log)
    summary = {}
    for key, sets, title in (("ref", ("uniform",), "ref/uniform"), ("ref", ("edge", "band", "steep", "axis"),
                                                                     "ref/stratified sets pooled"),
                             ("pa", ("uniform", "steep", "axis"), "pa/all sets pooled"),
                             ("sph", ("uniform",), "sph/uniform"), ("sph", ("edge", "band", "axis"),
                                                                    "sph/stratified sets pooled")):
        recs = [r for k in sets for r in data.get((key, k), [])]
        if recs:
            summary[title] = summarize(title, recs, sum(drawn.get((key, k), 0) for k in sets), log)
    ms = run_multistart(data, K, log)
    out = HERE / "results" / ("comparison_quick.npz" if a.quick else "comparison_data.npz")
    out.parent.mkdir(exist_ok=True)
    save(data, ms, str(out))
    log(f"[9.2] saved {out.relative_to(HERE)}")
    slim = {t: {n: {k: v for k, v in s.items() if k != "fmiss"} for n, s in by.items()} for t, by in summary.items()}
    log.close(summary=slim, plan={f"{k}/{s}": v for (k, s), v in PLAN.items()}, K=K, passes=PASSES,
              ms_passes=MS_PASSES)


if __name__ == "__main__":
    main()
