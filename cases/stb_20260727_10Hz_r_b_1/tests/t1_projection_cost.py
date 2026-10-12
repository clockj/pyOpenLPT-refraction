"""T1: what the new camFiles' projection tolerance costs (2026-10-04-plan.md, section 4).

The flat-window projection iterates (1-D Snell solve, with an LM fallback when it fails).  The delivered camFiles carry
proj_tol 1e-12 / 1000 iterations, the old ones 1e-6 / 50.  For every camera this times C++ project() on two point
sets, with the delivered settings and with the other ones:
  - "volume": 100k random points of the view volume (shaking, back-projection, triangulation checks);
  - "matcher": the points that stereo matching projects, 1 and 2 mm past the far wall along other cameras' lines of
    sight of pixels spread over their images (StereoMatch::makeLine2DFromLOS3D).
It counts failures and the pixel difference between the two settings, and times lineOfSight.  Each batch is one
projectBatchStatus call (a C++ loop); its per-point Python overhead is the same for both settings.

Writes t1_projection_cost.json, t1_projection_cost_out.txt and the variant camFiles in t1_camfiles/.
    python tests/t1_projection_cost.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402

OUT_JSON = os.path.join(case.TESTS, "t1_projection_cost.json")
OUT_TXT = os.path.join(case.TESTS, "t1_projection_cost_out.txt")
CAMDIR = os.path.join(case.TESTS, "t1_camfiles")
REPEAT = 3
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def with_solver(src: str, dst: str, tol: float, nmax: int) -> str:
    """A copy of a camFile with another projection tolerance and iteration limit."""
    txt = open(src, encoding="utf-8").read()
    txt, n1 = re.subn(r"(# Projection Tolerance:[^\n]*\n)[^\n]*", lambda m: m.group(1) + f"{tol:.17g}", txt)
    txt, n2 = re.subn(r"(# Maximum Number of Iterations for Projection:[^\n]*\n)[^\n]*",
                      lambda m: m.group(1) + f"{nmax:d}", txt)
    assert n1 == 1 and n2 == 1, (src, n1, n2)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(txt)
    return dst


def timed_project(cam, pts):
    """Best of REPEAT wall times (s) of one projectBatchStatus over pts, and its results."""
    best, res = np.inf, None
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        res = cam.projectBatchStatus(pts, False)
        best = min(best, time.perf_counter() - t0)
    uv = np.full((len(pts), 2), np.nan)
    ok = np.zeros(len(pts), bool)
    for i, (good, p, _m) in enumerate(res):
        if good and abs(p[0]) < 1e12 and abs(p[1]) < 1e12:
            uv[i] = (p[0], p[1])
            ok[i] = True
    return best, uv, ok


def matcher_points(cams, src: int, n: int, seed: int) -> np.ndarray:
    """Points 1 and 2 mm past camera src's far wall along lines of sight of n pixels spread over its image."""
    rng = np.random.default_rng(seed)
    c = cams[src]
    uv = np.stack([rng.uniform(0, c.getNCol() - 1, n), rng.uniform(0, c.getNRow() - 1, n)], axis=1)
    o, d, ok = case.lines_of_sight(c, uv)
    o, d = o[ok], d[ok]
    return np.concatenate([o + d, o + 2 * d])


def main():
    lpt = case.import_lpt()
    cfg = case.load_config()
    with open(os.path.join(case.TESTS, "t0_frame_map.json"), encoding="utf-8") as f:
        vv = {"new": json.load(f)["new_view_volume"], "old": cfg["old_view_volume"]}
    os.makedirs(CAMDIR, exist_ok=True)
    settings = {"delivered": None, "tol1e-6_n50": (1e-6, 50), "tol1e-12_n1000": (1e-12, 1000)}
    res = {"repeat": REPEAT, "sets": {}}
    for which in ("new", "old"):
        paths = case.camfile_paths(cfg, which)
        base = case.load_cameras(paths)
        vol = case.box_random(vv[which], 100_000, seed=10)
        vol_pts = [lpt.Pt3D(*map(float, p)) for p in vol]
        log(f"\n== {which} camFiles, view volume {vv[which]}")
        rs = {}
        for i, p in enumerate(paths):
            # the matcher's points for camera i: from the other cameras' lines of sight
            mp = np.concatenate([matcher_points(base, j, 6000, seed=100 + 10 * i + j) for j in range(4) if j != i])
            m_pts = [lpt.Pt3D(*map(float, q)) for q in mp]
            rc = {}
            for sname, sv in settings.items():
                if sv is None:
                    cam = base[i]
                    pl = case.plate_of(cam)
                    label = f"delivered ({pl['proj_tol']:g}/{pl['proj_nmax']})"
                else:
                    dst = os.path.join(CAMDIR, f"{which}_cam{i}_{sname}.txt")
                    cam = lpt.Camera(with_solver(p, dst, *sv))
                    label = sname
                tv, uv_v, ok_v = timed_project(cam, vol_pts)
                tm, uv_m, ok_m = timed_project(cam, m_pts)
                rc[sname] = {"label": label, "volume_us": 1e6 * tv / len(vol_pts), "volume_fail": int((~ok_v).sum()),
                             "matcher_us": 1e6 * tm / len(m_pts), "matcher_fail": int((~ok_m).sum()),
                             "n_volume": len(vol_pts), "n_matcher": len(m_pts), "_uv_v": uv_v, "_uv_m": uv_m}
            ref = rc["delivered"]
            for sname, r in rc.items():
                dv = np.linalg.norm(r["_uv_v"] - ref["_uv_v"], axis=1)
                dm = np.linalg.norm(r["_uv_m"] - ref["_uv_m"], axis=1)
                r["volume_dpx_max"] = float(np.nanmax(dv)) if np.isfinite(dv).any() else None
                r["matcher_dpx_max"] = float(np.nanmax(dm)) if np.isfinite(dm).any() else None
            # lineOfSight cost
            uvs = [lpt.Pt2D(float(u), float(v)) for u, v in np.stack(
                [np.random.default_rng(7).uniform(0, 2047, 50_000), np.random.default_rng(8).uniform(0, 1951, 50_000)],
                axis=1)]
            best = np.inf
            for _ in range(REPEAT):
                tt = time.perf_counter()
                base[i].lineOfSightBatchStatus(uvs)
                best = min(best, time.perf_counter() - tt)
            for r in rc.values():
                r.pop("_uv_v")
                r.pop("_uv_m")
            rc["lineOfSight_us"] = 1e6 * best / len(uvs)
            rs[f"cam{i}"] = rc
            log(f"cam{i}: lineOfSight {rc['lineOfSight_us']:.2f} us/call")
            for sname in settings:
                r = rc[sname]
                log(f"   {r['label']:>24s}: volume {r['volume_us']:6.2f} us/call, fail {r['volume_fail']:6d}/"
                    f"{r['n_volume']}, max |d| {r['volume_dpx_max']:.2e} px | matcher {r['matcher_us']:6.2f} us/call, "
                    f"fail {r['matcher_fail']:6d}/{r['n_matcher']}, max |d| {r['matcher_dpx_max']:.2e} px")
        res["sets"][which] = rs

    case.save_json(OUT_JSON, res)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT_JSON}\n      {OUT_TXT}")


if __name__ == "__main__":
    main()
