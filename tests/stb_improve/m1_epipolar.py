"""Test M1 (plan B2): how well the C++ code's straight image lines follow the curved image of a line of sight.

    python m1_epipolar.py [--builds base b2] [--n 20000]

For random points X of this case's view volume and the exact pixels of X in every camera (no calibration or
detection error), the C++ StereoMatch::makeLine2DFromLOS3D of each build (through the debug binding) draws camera
i's line of sight of X's pixel as a line in camera j.  A 4-camera match needs every ordered pair of its cameras
within tol_2d (forward lines and back-projections both use this line), so per camera combination the share of
points with every ordered pair within 1, 2, 3 and 4.5 px is the share stereo matching can find with a perfect
calibration.  "base" is HEAD (the line through the points 1 and 2 mm past the window), "b2" anchors it where the
line of sight enters and leaves the view volume; the case's T2 test (Python model of both rules) predicted 37 % ->
91 % at 2 px for the 4-camera set.

Each build runs in its own process; writes results/m1_epipolar.txt and .json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
CASE_DIR = os.path.join(REPO, "cases", "stb_20260727_10Hz_r_b_1")
PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"
TOLS = (1.0, 2.0, 3.0, 4.5)
COMBOS = [(0, 1, 2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def worker(build_dir, n, seed, out_path):
    """Runs inside a process with the chosen build first on sys.path."""
    sys.path.insert(0, CASE_DIR)
    import numpy as np
    import case  # noqa: F401 (puts build-py on sys.path; the chosen build goes in front below)
    sys.path.insert(0, build_dir)
    import pyopenlpt as lpt
    assert os.path.normcase(os.path.dirname(lpt.__file__)) == os.path.normcase(build_dir), lpt.__file__

    cfg = os.path.join(HERE, "runs", "c1_config", "config.txt")   # the benchmark setup (c1_config.py writes it)
    bs = lpt.BasicSetting()
    bs.readConfig(cfg)
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    sm = lpt.StereoMatch(cams, [[] for _ in cams], tc)
    al = bs._axis_limit
    vv = [al.x_min, al.x_max, al.y_min, al.y_max, al.z_min, al.z_max]
    X = case.box_random(vv, n, seed)

    uv, seen, los = [], [], []
    for c in cams:
        u, ok = case.project(c, X)
        s = ok & case.in_image(u, c)
        uv.append(u)
        seen.append(s)
        res = c.lineOfSightBatchStatus([lpt.Pt2D(float(a), float(b)) for a, b in np.where(s[:, None], u, 1000.0)])
        los.append([(good and bool(s[k])) and line for k, (good, line, _m) in enumerate(res)])
    seen = np.array(seen)
    dev = np.full((4, 4, n), np.inf)
    pairs = {}
    for i, j in itertools.permutations(range(4), 2):
        idx = np.flatnonzero(seen[i] & seen[j])
        n_fail = 0
        for k in idx:
            L3 = los[i][k]
            if not L3:
                continue
            ok, L2 = sm.makeLine2DFromLOS3D(j, L3)
            if not ok:
                n_fail += 1
                continue
            px, py = L2.pt[0], L2.pt[1]
            ux, uy = L2.unit_vector[0], L2.unit_vector[1]
            qx, qy = uv[j][k]
            dev[i, j, k] = abs((qy - py) * ux - (qx - px) * uy)
        d = dev[i, j, idx]
        fin = d[np.isfinite(d)]
        pairs[f"{i}->{j}"] = {"n": int(idx.size), "line_failed": n_fail,
                              "median_px": float(np.median(fin)) if fin.size else None,
                              "p95_px": float(np.percentile(fin, 95)) if fin.size else None,
                              "max_px": float(fin.max()) if fin.size else None}
    combos = {}
    for combo in COMBOS:
        allseen = np.all(seen[list(combo)], axis=0)
        worst = np.max([dev[i, j] for i, j in itertools.permutations(combo, 2)], axis=0)[allseen]
        combos[str(list(combo))] = {"n_seen": int(allseen.sum()),
                                    **{f"le{t:g}": float(np.mean(worst <= t)) for t in TOLS}}
    with open(out_path, "w") as f:
        json.dump({"build_dir": build_dir, "pyopenlpt": lpt.__file__, "view_volume": vv, "n": n, "seed": seed,
                   "pairs": pairs, "combos": combos}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=["base", "b2"])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--worker", nargs=2, metavar=("BUILD_DIR", "OUT"), default=None)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], args.n, args.seed, args.worker[1])
        return
    sys.path.insert(0, HERE)
    import stbrun
    os.makedirs(RESULTS, exist_ok=True)
    data, lines = {}, [f"M1 epipolar lines: {args.n} random points of the view volume (seed {args.seed}), exact "
                       "pixels; share of points seen by all cameras of a combination with every ordered pair within "
                       + " / ".join(f"{t:g}" for t in TOLS) + " px"]
    for b in args.builds:
        out = os.path.join(RESULTS, f"m1_epipolar_{b}.json")
        r = subprocess.run([PY, __file__, "--n", str(args.n), "--seed", str(args.seed), "--worker",
                            stbrun.build_dir(b), out], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            sys.exit(f"worker for {b} failed")
        with open(out) as f:
            data[b] = json.load(f)
        lines.append(f"\nbuild {b} ({data[b]['pyopenlpt']}):")
        for combo, v in data[b]["combos"].items():
            lines.append(f"  {combo:14s} ({v['n_seen']:5d} points): "
                         + " / ".join(f"{v[f'le{t:g}']:.3f}" for t in TOLS))
        lines.append("  ordered pairs, median / p95 / max px (lines that could not be built):")
        for pr, v in data[b]["pairs"].items():
            if v["median_px"] is None:
                lines.append(f"    {pr}: no points")
                continue
            lines.append(f"    {pr}: {v['median_px']:.3f} / {v['p95_px']:.3f} / {v['max_px']:.3f} "
                         f"({v['line_failed']} failed of {v['n']})")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "m1_epipolar.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    with open(os.path.join(RESULTS, "m1_epipolar.json"), "w") as f:
        json.dump(data, f, indent=1)


if __name__ == "__main__":
    main()
