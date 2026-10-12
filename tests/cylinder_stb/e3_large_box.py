"""E3 (plan, Phase 3): a view volume that reaches past the tube's wall.

    python e3_large_box.py [--builds head1007 p3] [--case f5_spots] [--n 5000]

Today StereoMatch draws an image line through the images of the points where a line of sight enters and leaves the
view volume; through a curved port a point outside the water has no image, so with a view volume reaching past the
wall project() fails and the line is dropped (the observation cannot be matched at all).  Phase 3 clips the line of
sight to the water first.  The synthetic case's config is copied with the view volume widened to +-150 mm across the
tube (the inner wall is at 97.6 mm) and run through each build's StereoMatch (debug bindings, one process per build):
for N points both cameras see, camera i's line of sight through the point's pixel, drawn on camera j by the table's
function of that build (head1007: makeLine2DFromLOS3D; p3: makeCurve2DFromLOS3D for these cameras), and the distance
from the point's true pixel.  Judged (plan): with the wide volume the lines are built (today they are dropped).
Writes e3_large_box_out.txt and e3_large_box.json.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

import cstb

PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"


def worker(build, case_cfg, n, seed, out_path):
    lpt, rec = cstb.use_build(build)
    bs = lpt.BasicSetting()
    bs.readConfig(case_cfg)
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    sm = lpt.StereoMatch(cams, [[] for _ in cams], tc)
    rng = np.random.default_rng(seed)
    box = ((-11.0, -9.0, -20.0), (19.0, 12.0, 24.0))      # where the points are (inside the water)
    pts, pix = cstb.sample_shared(lpt, cams, n, box, rng)
    res = {"build": rec, "pairs": {}}
    curved = hasattr(sm, "makeCurve2DFromLOS3D")
    for i, j in ((0, 1), (1, 0)):
        ok_n, d = 0, []
        for k in range(len(pts)):
            los = cams[i].lineOfSight(lpt.Pt2D(float(pix[k, i, 0]), float(pix[k, i, 1])))
            q = lpt.Pt2D(float(pix[k, j, 0]), float(pix[k, j, 1]))
            if curved:
                ok, _chord, poly = sm.makeCurve2DFromLOS3D(j, los)
                if ok:
                    d.append(float(np.sqrt(lpt.dist2ToPolyline(q, poly))))
            else:
                ok, line = sm.makeLine2DFromLOS3D(j, los)
                if ok:
                    ux, uy = line.unit_vector[0], line.unit_vector[1]
                    d.append(abs((q[1] - line.pt[1]) * ux - (q[0] - line.pt[0]) * uy))
            ok_n += bool(ok)
        res["pairs"][f"cam{i + 1}->cam{j + 1}"] = {"points": len(pts), "lines_built": ok_n,
                                                   "max_px": max(d) if d else None,
                                                   "median_px": float(np.median(d)) if d else None}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=["head1007", "p3"])
    ap.add_argument("--case", default="f5_spots")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--worker", nargs=2, default=None)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker[0], os.path.join(cstb.DATA, "e3_bigbox", "config.json"), args.n, args.seed, args.worker[1])
        return
    log = cstb.Log(os.path.join(cstb.HERE, "e3_large_box"))
    with open(os.path.join(cstb.DATA, args.case, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["view_volume_mm"] = {"x": [-150.0, 150.0], "y": cfg["view_volume_mm"]["y"], "z": [-150.0, 150.0]}
    os.makedirs(os.path.join(cstb.DATA, "e3_bigbox"), exist_ok=True)
    with open(os.path.join(cstb.DATA, "e3_bigbox", "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    log(f"E3: a view volume past the wall (x, z in +-150 mm; inner wall radius 97.6 mm); case {args.case}, "
        f"{args.n} points; builds {args.builds}")
    out = {"builds": {}}
    for b in args.builds:
        p = os.path.join(cstb.DATA, "e3_bigbox", f"e3_{b}.json")
        r = subprocess.run([PY, __file__, "--n", str(args.n), "--seed", str(args.seed), "--worker", b, p],
                           capture_output=True, text=True, cwd=cstb.HERE)
        if r.returncode != 0:
            log(r.stdout[-2000:] + r.stderr[-2000:])
            raise SystemExit(f"worker {b} failed")
        with open(p, encoding="utf-8") as fh:
            out["builds"][b] = json.load(fh)
        for pr, v in out["builds"][b]["pairs"].items():
            log(f"  {b} {pr}: lines built for {v['lines_built']} of {v['points']} points"
                + (f"; true pixel from the line: median {v['median_px']:.4f}, max {v['max_px']:.4f} px"
                   if v["max_px"] is not None else ""))
    new = out["builds"][args.builds[-1]]["pairs"]
    out["pass"] = bool(all(v["lines_built"] == v["points"] for v in new.values()))
    log(f"  judged (every line built with the wide volume, build {args.builds[-1]}): "
        f"{'PASS' if out['pass'] else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
