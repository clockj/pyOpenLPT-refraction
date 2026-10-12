"""E1a (Phase E): the Python predictors against the C++ code, and their own consistency checks.

  1. predictors.wiener_one (today's STB::predictNext, ported) against the C++ function itself, called through the
     module's STB.predictNext (build-py/Release, the build the GUI runs; read only):
       - 20,000 histories of 3-7 points sampled at random from B300's saved tracks;
       - edge cases: an axis at 0, +-0.999999, +-1, +-1.000001 mm, negative coordinates, a particle at rest, 3-point
         tracks, the fallback branch (all earlier samples 0), coordinates of 80 mm.
     Judged by: agreement to 1e-12 mm (and how many agree to the last bit).
  2. predictors.wiener (vectorised) against wiener_one on the same histories: to the last bit.
  3. mean_step is W's limit far from the origin: W(P + C) - C against mean_step(P) for C = 1e6 mm.
  4. Least-squares weights reproduce polynomials up to their degree exactly (1e-9 mm).
  5. Kalman filters (cv, ca): with q = 0 the recursive filter equals the least-squares line / parabola through all
     points (1e-9 mm); for q/r = 1e-4, 1e-2, 1 the recursive run equals the fixed-weight form (1e-9 mm).

Writes results/e1a_port_check.txt and .json.
    C:\\Users\\zcloc\\.conda\\envs\\refract\\python.exe tests/stb_improve/e1a_port_check.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(REPO, "build-py", "Release")
sys.path.insert(0, HERE)
import predictors as pr  # noqa: E402

RUN_DIR = os.path.join(HERE, "runs", "b300_final3")
_lines = []
results = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def check(name, ok, detail):
    results[name] = {"ok": bool(ok), "detail": detail}
    log(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def make_stb(lpt):
    bs = lpt.BasicSetting()
    if not bs.readConfig(os.path.join(RUN_DIR, "config.txt")):
        raise RuntimeError("cannot read the B300 config")
    return lpt.STB(bs, "Tracer", bs._object_config_paths[0])


def cpp_predict(lpt, stb, hist):
    tr = lpt.Track()
    for t, p in enumerate(hist):
        tr.add_next(lpt.Tracer3D(lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])), 2.0), t)
    o = stb.predictNext(tr)
    c = o._pt_center
    return [c[0], c[1], c[2]]


def edge_histories():
    """Hand-made histories for the branches of predictNext."""
    H = []
    base = np.array([[12.3, -40.1, 7.7], [12.4, -40.0, 7.6], [12.5, -39.9, 7.5], [12.6, -39.8, 7.4],
                     [12.7, -39.7, 7.3], [12.8, -39.6, 7.2], [12.9, -39.5, 7.1]])
    for last in (0.0, 0.999999, -0.999999, 1.0, -1.0, 1.000001, -1.000001, 0.5, -0.5, 2.0, -2.0):
        for v in (0.1, -0.1, 0.0):
            for n in (3, 4, 5, 6, 7):
                h = base[:n].copy()
                h[:, 0] = last - v * (n - 1 - np.arange(n))       # x ends at `last`
                h[:, 1] = -last + 0.07 * (n - 1 - np.arange(n))   # y ends at -last
                H.append(h)
    H.append(np.zeros((6, 3)))                                    # all zero: every axis shifted
    H.append(np.full((5, 3), 3.25))                               # at rest
    H.append(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [5.0, 6.0, -7.0]]))    # fallback: earlier samples all 0
    H.append(np.array([[1e-7, 0, 0], [0, 1e-7, 0], [5.0, 6.0, -7.0]]))         # fallback on a 3-point track
    H.append(np.array([[80.0, -80.0, 52.0], [79.9, -79.9, 52.1], [79.8, -79.8, 52.2], [79.7, -79.7, 52.3]]))
    H.append(np.array([[-81.0, 59.0, -36.0], [-80.9, 58.9, -35.9], [-80.8, 58.8, -35.8]]))
    return H


def main():
    t0 = time.time()
    sys.path.insert(0, BUILD)
    import pyopenlpt as lpt
    assert os.path.normcase(os.path.dirname(lpt.__file__)) == os.path.normcase(BUILD), lpt.__file__
    log(f"E1a: module {lpt.__file__}")
    stb = make_stb(lpt)

    tr = pr.load_run("b300")
    log(f"B300: {len(tr.starts)} saved long tracks, {len(tr.P)} points")
    rng = np.random.default_rng(20261006)
    n_hist = rng.integers(3, 8, size=20000)
    tidx = rng.integers(0, len(tr.starts), size=20000)
    keep = tr.lengths[tidx] >= n_hist
    n_hist, tidx = n_hist[keep], tidx[keep]
    off = (rng.random(len(tidx)) * (tr.lengths[tidx] - n_hist + 1)).astype(int)
    starts = tr.starts[tidx] + off
    hists = [tr.P[s:s + n] for s, n in zip(starts, n_hist)]
    hists += edge_histories()
    n_edge = len(edge_histories())
    log(f"histories: {len(hists) - n_edge} sampled (3-7 points), {n_edge} edge cases")

    cpp = np.array([cpp_predict(lpt, stb, h) for h in hists])
    py1 = np.array([pr.wiener_one(h) for h in hists])
    d = np.abs(cpp - py1)
    same = np.all(cpp == py1, axis=1)
    check("1 port: wiener_one = C++ predictNext", d.max() <= 1e-12,
          f"max |diff| {d.max():.3g} mm over {len(hists)} histories; identical to the last bit: {same.sum()} "
          f"({100 * same.mean():.2f} %); edge cases max |diff| {d[-n_edge:].max():.3g} mm")
    if not same.all():
        bad = np.nonzero(~same)[0][:5]
        for b in bad:
            log(f"    differs: history {b} n={len(hists[b])} cpp={cpp[b].tolist()} py={py1[b].tolist()}")

    # 2. vectorised against scalar, on a flat layout of the same histories
    flat = np.concatenate([np.vstack([h, np.zeros((1, 3))]) for h in hists])
    lens = np.array([len(h) + 1 for h in hists])
    st = np.concatenate([[0], np.cumsum(lens)[:-1]])
    g = st + lens - 1
    pyv = pr.wiener(flat, g, lens - 1)
    check("2 vectorised wiener = wiener_one", np.array_equal(pyv, py1),
          f"identical to the last bit: {np.all(pyv == py1, axis=1).sum()} of {len(hists)}")

    # 3. far from the origin W tends to the mean step: the difference falls as 1/C
    sampled = slice(0, len(hists) - n_edge)
    md = pr.mean_step(flat, g, lens - 1)[sampled]
    dC = {}
    for C in (1e4, 1e5, 1e6):
        wfar = pr.wiener(flat + C, g, lens - 1)[sampled] - C
        dC[C] = np.abs(wfar - md).max()
    ratios = [dC[1e4] / dC[1e5], dC[1e5] / dC[1e6]]
    check("3 W(P + C) - C -> mean step, as 1/C", all(9 < r < 11 for r in ratios),
          "max |diff| " + ", ".join(f"{d:.3g} mm at C = {C:g}" for C, d in dC.items())
          + f"; ratios {ratios[0]:.2f}, {ratios[1]:.2f} (1/C: 10)")

    # 4. polynomial weights
    worst = 0.0
    for deg in (1, 2):
        for k in range(deg + 1, 25):
            w = pr.poly_weights(deg, k)
            t = np.arange(k + 1, dtype=float)
            for c in rng.normal(size=(5, deg + 1)) * np.array([50, 0.1, 0.01][:deg + 1]):
                y = np.polyval(c[::-1], t)
                worst = max(worst, abs(w @ y[:k] - y[k]))
    check("4 least-squares weights reproduce polynomials", worst < 1e-9, f"max |error| {worst:.3g} mm")

    # 5. Kalman filters
    lens5 = rng.integers(4, 61, size=300)
    P5 = []
    for L in lens5:
        t = np.arange(L, dtype=float)
        p0 = rng.uniform(-80, 60, 3)
        v0 = rng.normal(0, 0.1, 3)
        a0 = rng.normal(0, 0.005, 3)
        P5.append(p0 + v0 * t[:, None] + 0.5 * a0 * t[:, None] ** 2 + rng.normal(0, 0.006, (L, 3)))
    st5 = np.concatenate([[0], np.cumsum(lens5)[:-1]])
    T5 = pr.Tracks(np.concatenate(P5), st5, lens5, np.zeros(len(lens5), np.int8), np.arange(len(lens5)),
                   np.zeros(len(lens5), int))
    ii = T5.index_in_track
    for kind, deg in (("cv", 1), ("ca", 2)):
        kf = pr.Kalman(kind, 0.0)
        pred, _ = kf.run(T5)
        sel = np.nonzero(ii >= kf.d)[0]
        ref = np.full_like(pred, np.nan)
        for kk in np.unique(ii[sel]):
            s = sel[ii[sel] == kk]
            ref[s] = pr.poly(T5.P, s, ii[s], deg, int(kk))
        dd = np.nanmax(np.abs(pred[sel] - ref[sel]))
        check(f"5 Kalman {kind}, q = 0: = least squares through all points", dd < 1e-9, f"max |diff| {dd:.3g} mm")
        for lam in (1e-4, 1e-2, 1.0):
            kf = pr.Kalman(kind, lam)
            pred, _ = kf.run(T5)
            fir = kf.fir(T5.P, sel, ii[sel], int(lens5.max()))
            dd = np.nanmax(np.abs(pred[sel] - fir))
            check(f"5 Kalman {kind}, q/r = {lam:g}: recursive = fixed weights", dd < 1e-9, f"max |diff| {dd:.3g} mm")

    n_fail = sum(not r["ok"] for r in results.values())
    log(f"\nE1a: {len(results) - n_fail} of {len(results)} checks passed ({time.time() - t0:.0f} s)")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "e1a_port_check.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(_lines) + "\n")
    with open(os.path.join(HERE, "results", "e1a_port_check.json"), "w") as f:
        json.dump({"module": lpt.__file__, "n_histories": len(hists), "n_edge": n_edge, "checks": results}, f,
                  indent=1)
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
