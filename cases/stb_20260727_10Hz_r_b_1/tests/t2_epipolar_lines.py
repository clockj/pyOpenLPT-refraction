"""T2: how far true matches lie from the straight epipolar lines that stereo matching draws (2026-10-04-plan.md).

StereoMatch draws a line of sight of camera i in camera j as the straight 2-D line through the images of two points 1
and 2 mm past camera i's far wall (makeLine2DFromLOS3D).  Through flat windows the true image of that line of sight is
a curve.  For random points X of the view volume, exact pixels in every camera (no calibration or detection error):
  - code:  distance in camera j of X's pixel from the line built from camera i's line of sight of X's pixel;
  - chord: the same for a line through the images of the points where that line of sight enters and leaves the view
           volume (for comparison: what a line anchored inside the volume would give).
A 4-camera match needs every ordered pair of its cameras within tol_2d (2 px, then 3 px): forward lines and
back-projections both use this construction.  So per camera combination, the share of points with every pair within
1, 2 and 3 px is the share that matching can find at all with a perfect calibration.

Writes t2_epipolar_lines.json and t2_epipolar_lines_out.txt beside this script.
    python tests/t2_epipolar_lines.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402

OUT_JSON = os.path.join(case.TESTS, "t2_epipolar_lines.json")
OUT_TXT = os.path.join(case.TESTS, "t2_epipolar_lines_out.txt")
N = 20000
TOLS = (1.0, 2.0, 3.0, 4.5)
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def clip_line(o, d, vv):
    """Parameters (t_in, t_out) of o + t d inside the box vv (slab method); NaN where it misses."""
    lo = np.array(vv[0::2])
    hi = np.array(vv[1::2])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    miss = tmax < tmin
    tmin[miss] = np.nan
    tmax[miss] = np.nan
    return tmin, tmax


def line_dist(p, a, b):
    """Distance of points p from the 2-D lines through a and b (all (n, 2))."""
    e = b - a
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    w = p - a
    return np.abs(w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0])


def stats(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()),
            **{f"gt{t:g}": float(np.mean(x > t)) for t in TOLS}}


def run_set(which, cams, vv, seed):
    X = case.box_random(vv, N, seed)
    uv, seen = [], []
    for c in cams:
        u, ok = case.project(c, X)
        uv.append(u)
        seen.append(ok & case.in_image(u, c))
    seen = np.array(seen)
    los = []
    for c, u, s in zip(cams, uv, seen):
        o, d, ok = case.lines_of_sight(c, np.where(s[:, None], u, 1000.0))
        los.append((o, d, ok & s))
    out = {"view_volume": vv, "n_points": N, "seen_share": seen.mean(1), "pairs": {}, "combos": {}}
    dev_code = np.full((4, 4, N), np.nan)
    dev_chord = np.full((4, 4, N), np.nan)
    depth = np.full((4, 4, N), np.nan)          # distance along camera i's line of sight from its wall to X, mm
    log(f"\n== {which} camFiles, view volume {vv}; {N} random points; share on each sensor "
        f"{np.round(seen.mean(1), 3).tolist()}")
    log("   ordered pair i->j: distance (px) of X in camera j from the line of camera i's line of sight: "
        "median / p95 / max, share > 2 px; and for the chord line inside the volume")
    for i, j in itertools.permutations(range(4), 2):
        o, d, oki = los[i]
        sel = oki & seen[j]
        a, oka = case.project(cams[j], o[sel] + d[sel])
        b, okb = case.project(cams[j], o[sel] + 2 * d[sel])
        p = uv[j][sel]
        dc = np.where(oka & okb, line_dist(p, a, b), np.inf)      # the matcher skips a line it cannot build
        tin, tout = clip_line(o[sel], d[sel], vv)
        ai, oki2 = case.project(cams[j], o[sel] + tin[:, None] * d[sel])
        bi, oki3 = case.project(cams[j], o[sel] + tout[:, None] * d[sel])
        dh = np.where(oki2 & oki3 & np.isfinite(tin), line_dist(p, ai, bi), np.inf)
        idx = np.flatnonzero(sel)
        dev_code[i, j, idx] = dc
        dev_chord[i, j, idx] = dh
        depth[i, j, idx] = np.einsum("ni,ni->n", X[sel] - o[sel], d[sel])
        sc, sh = stats(dc), stats(dh)
        out["pairs"][f"{i}->{j}"] = {"code": sc, "chord": sh, "n_line_fail": int((~(oka & okb)).sum())}
        log(f"   {i}->{j}: code {sc['median']:7.3f} / {sc['p95']:7.3f} / {sc['max']:8.3f}, >2px {sc['gt2']:.3f} | "
            f"chord {sh['median']:6.3f} / {sh['p95']:6.3f} / {sh['max']:7.3f}, >2px {sh['gt2']:.3f}")
    # deviation against the depth along the line of sight (10 mm bins), all pairs pooled
    dd = depth[np.isfinite(dev_code)]
    dv = dev_code[np.isfinite(dev_code)]
    bins = np.arange(0, np.nanmax(dd) + 10, 10)
    prof = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dd >= lo) & (dd < hi)
        if m.sum() > 20:
            prof.append({"depth_mm": [float(lo), float(hi)], "n": int(m.sum()), "median_px": float(np.median(dv[m])),
                         "p95_px": float(np.percentile(dv[m], 95))})
    out["depth_profile_code"] = prof
    log("   code-line deviation against the distance from the source camera's wall (all pairs): "
        + "; ".join(f"{p['depth_mm'][0]:.0f}-{p['depth_mm'][1]:.0f} mm: {p['median_px']:.2f}/{p['p95_px']:.2f}"
                    for p in prof))
    # per camera combination: every ordered pair within tolerance
    log("   share of points seen by every camera of a combination with all its ordered pairs within 1 / 2 / 3 / 4.5 px")
    for combo in [(0, 1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
        allseen = np.all(seen[list(combo)], axis=0)
        worst_c = np.max([dev_code[i, j] for i, j in itertools.permutations(combo, 2)], axis=0)[allseen]
        worst_h = np.max([dev_chord[i, j] for i, j in itertools.permutations(combo, 2)], axis=0)[allseen]
        rc = {f"le{t:g}": float(np.mean(worst_c <= t)) for t in TOLS}
        rh = {f"le{t:g}": float(np.mean(worst_h <= t)) for t in TOLS}
        out["combos"][str(list(combo))] = {"n_seen": int(allseen.sum()), "code": rc, "chord": rh,
                                           "worst_code": stats(worst_c), "worst_chord": stats(worst_h)}
        log(f"   {list(combo)} ({allseen.sum()} points): code " + " / ".join(f"{rc[f'le{t:g}']:.3f}" for t in TOLS)
            + " | chord " + " / ".join(f"{rh[f'le{t:g}']:.3f}" for t in TOLS))
    return out


def main():
    cfg = case.load_config()
    with open(os.path.join(case.TESTS, "t0_frame_map.json"), encoding="utf-8") as f:
        vv_new = json.load(f)["new_view_volume"]
    res = {"n_points": N, "tolerances_px": TOLS}
    res["new"] = run_set("new", case.load_cameras(case.camfile_paths(cfg, "new")), vv_new, 21)
    res["old"] = run_set("old", case.load_cameras(case.camfile_paths(cfg, "old")), cfg["old_view_volume"], 22)
    case.save_json(OUT_JSON, res)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT_JSON}\n      {OUT_TXT}")


if __name__ == "__main__":
    main()
