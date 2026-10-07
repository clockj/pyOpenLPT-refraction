"""Fraction of true matches outside StereoMatch's straight image line (box entry/exit anchored), T2 rig."""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyd"))
import pyopenlpt as lpt
RNG = np.random.default_rng(5)
cams = [lpt.loadCameraModel(os.path.join(HERE, "variants", "cam%d_tol9.txt" % i)) for i in (1, 2)]
out = json.load(open(os.path.join(HERE, "probe_rig_out.json")))
lo, hi = np.array(out["A1"]["common_bbox_mm"][0]) - 0.75, np.array(out["A1"]["common_bbox_mm"][1]) + 0.75
NROW, NCOL = 800, 1280
def uv(p): return np.array([p[0], p[1]])
def P(x): return lpt.Pt3D(*map(float, x))
def onsens(p): return 0 <= p[0] < NCOL and 0 <= p[1] < NROW
def clip(o, u):
    t_in, t_out = 0.0, math.inf
    for d in range(3):
        if abs(u[d]) < 1e-15:
            if o[d] < lo[d] or o[d] > hi[d]: return None
            continue
        ta, tb = sorted(((lo[d] - o[d]) / u[d], (hi[d] - o[d]) / u[d]))
        t_in, t_out = max(t_in, ta), min(t_out, tb)
        if t_in > t_out: return None
    return t_in, t_out
res = {}
for a, b in ((0, 1), (1, 0)):
    d = []
    while len(d) < 3000:
        X = RNG.uniform(lo, hi)
        try:
            pa = uv(cams[a].project(P(X))); pb = uv(cams[b].project(P(X)))
        except RuntimeError:
            continue
        if not (onsens(pa) and onsens(pb)): continue
        los = cams[a].lineOfSight(lpt.Pt2D(*pa))
        o = np.array([los.pt[i] for i in range(3)]); u = np.array([los.unit_vector[i] for i in range(3)])
        tt = clip(o, u)
        if tt is None: continue
        e0 = uv(cams[b].project(P(o + tt[0] * u))); e1 = uv(cams[b].project(P(o + tt[1] * u)))
        ab = e1 - e0
        d.append(abs((pb[0] - e0[0]) * ab[1] - (pb[1] - e0[1]) * ab[0]) / np.hypot(*ab))
    d = np.array(d)
    res["cam%d->cam%d" % (a + 1, b + 1)] = {"dist_px_pcts_50_90_99_max": np.percentile(d, [50, 90, 99, 100]).round(3).tolist(),
        "frac_gt_0.5px": round(float((d > 0.5).mean()), 3), "frac_gt_1.0px": round(float((d > 1.0).mean()), 4)}
print(json.dumps(res, indent=1))
