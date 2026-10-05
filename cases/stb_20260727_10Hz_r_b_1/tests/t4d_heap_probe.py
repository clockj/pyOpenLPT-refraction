"""T4d: does a heap full of small Pt3D blocks slow down the cameras' geometry? (an addition, 2026-10-04-plan.md 6a)

pyOpenLPT's Matrix keeps its elements in `new T[n]`, so every Pt2D / Pt3D / Line3D is a heap allocation: projection
and line of sight create and free many per call, and every tracked point keeps about 11 heap blocks alive for the
track's life (Object3D, its Pt3D, the 2-D objects and their Pt2D).  T4c showed that the same frames run 34x faster in a
fresh process.  This probe times one camera's project() and lineOfSight() on fixed inputs in a fresh process, then
after filling the heap with millions of Pt3D objects (made through the bindings, so each one is a C++ Matrix plus its
`new double[3]`) and freeing every other one, then after freeing them all.  One thread only: it tests fragmentation,
not contention between threads.

Writes t4d_heap_probe.json and t4d_heap_probe_out.txt beside this script.
    python tests/t4d_heap_probe.py [N_MILLION]     (default 12 million Pt3D)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402

OUT_JSON = os.path.join(case.TESTS, "t4d_heap_probe.json")
OUT_TXT = os.path.join(case.TESTS, "t4d_heap_probe_out.txt")
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def bench(cam, pts, uvs, reps=5):
    """Best of reps: microseconds per project() and per lineOfSight() call (C++ batch loops)."""
    tp, tl = np.inf, np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        cam.projectBatchStatus(pts, False)
        tp = min(tp, time.perf_counter() - t0)
        t0 = time.perf_counter()
        cam.lineOfSightBatchStatus(uvs)
        tl = min(tl, time.perf_counter() - t0)
    return 1e6 * tp / len(pts), 1e6 * tl / len(uvs)


def main():
    n_million = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    lpt = case.import_lpt()
    cfg = case.load_config()
    cam = case.load_cameras(case.camfile_paths(cfg, "new"))[2]
    import json
    with open(os.path.join(case.TESTS, "t0_frame_map.json"), encoding="utf-8") as f:
        vv = json.load(f)["new_view_volume"]
    X = case.box_random(vv, 100_000, seed=5)
    pts = [lpt.Pt3D(*map(float, p)) for p in X]
    rng = np.random.default_rng(6)
    uvs = [lpt.Pt2D(float(u), float(v)) for u, v in zip(rng.uniform(0, 2047, 50_000), rng.uniform(0, 1951, 50_000))]
    res = {"n_pt3d": int(n_million * 1e6), "camera": "new cam2"}
    res["fresh"] = bench(cam, pts, uvs)
    log(f"fresh process: project {res['fresh'][0]:.3f} us, lineOfSight {res['fresh'][1]:.3f} us")
    t0 = time.perf_counter()
    keep = [lpt.Pt3D(1.0, 2.0, 3.0) for _ in range(int(n_million * 1e6))]
    t_alloc = time.perf_counter() - t0
    res["full"] = bench(cam, pts, uvs)
    log(f"after allocating {len(keep) / 1e6:.0f} M Pt3D ({t_alloc:.1f} s): project {res['full'][0]:.3f} us, "
        f"lineOfSight {res['full'][1]:.3f} us")
    del keep[::2]
    res["half_freed"] = bench(cam, pts, uvs)
    log(f"after freeing every other one: project {res['half_freed'][0]:.3f} us, lineOfSight {res['half_freed'][1]:.3f} us")
    # interleave: allocate and free temporaries of the same size among the survivors, as frames do
    churn = []
    for _ in range(3):
        churn += [lpt.Pt3D(0.0, 0.0, 0.0) for _ in range(len(keep) // 4)]
        del churn[::3]
    res["churned"] = bench(cam, pts, uvs)
    log(f"after churn (allocate and free more of the same size): project {res['churned'][0]:.3f} us, "
        f"lineOfSight {res['churned'][1]:.3f} us")
    del keep, churn
    res["all_freed"] = bench(cam, pts, uvs)
    log(f"after freeing everything: project {res['all_freed'][0]:.3f} us, lineOfSight {res['all_freed'][1]:.3f} us")
    case.save_json(OUT_JSON, res)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT_JSON}\n      {OUT_TXT}")


if __name__ == "__main__":
    main()
