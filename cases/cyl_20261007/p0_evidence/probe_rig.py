"""Probe of particle tracking through the T2 tube rig (scratch, not part of any repo).

A1 image census over the volume the two cameras share
A2 cost of one projection: PINCYL (file tol and default tol) against a PINHOLE with the same K, R, t
A3 epipolar curvature: the image of a camera-1 line of sight in camera 2 against StereoMatch's straight line
A4 astigmatism: tangential and sagittal virtual-image depths from the parallax of a displaced pinhole,
   and the anisotropy of the image Jacobian
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pyd"))
import pyopenlpt as lpt  # noqa: E402

CAM_FILES = [os.path.join(HERE, "cams", "cam%d.txt" % i) for i in (1, 2)]
VAR = os.path.join(HERE, "variants")
os.makedirs(VAR, exist_ok=True)
RNG = np.random.default_rng(7)
OUT = {}


def load_text(p):
    with open(p) as f:
        return f.read().splitlines()


def value_index(lines, header):
    for i, l in enumerate(lines):
        if l.startswith(header):
            return i + 1
    raise KeyError(header)


def fmt(v):
    return ",".join("%.17e" % x for x in v)


def read_rt(src):
    lines = load_text(src)
    i_r = value_index(lines, "# Rotation Matrix")
    i_t = value_index(lines, "# Translation Vector")
    i_k = value_index(lines, "# Camera Matrix")
    R = np.array([[float(x) for x in lines[i_r + k].split(",")] for k in range(3)])
    t = np.array([float(x) for x in lines[i_t].split(",")])
    K = np.array([[float(x) for x in lines[i_k + k].split(",")] for k in range(3)])
    i_s = value_index(lines, "# Image Size")
    n_row, n_col = (int(x) for x in lines[i_s].split(","))
    return R, t, K, n_row, n_col


def variant(src, tag, t_shift=None, tol=None, nmax=None, pinhole=False):
    lines = load_text(src)
    if t_shift is not None:
        R, t, _, _, _ = read_rt(src)
        t2 = t - np.asarray(t_shift, float)  # the pinhole moves by +R^T t_shift
        lines[value_index(lines, "# Translation Vector")] = fmt(t2)
        lines[value_index(lines, "# Inverse of Translation Vector")] = fmt(-R.T @ t2)
    if tol is not None:
        lines[value_index(lines, "# Projection Tolerance")] = "%.17e" % tol
    if nmax is not None:
        lines[value_index(lines, "# Maximum Number of Iterations for Projection")] = "%d" % nmax
    if pinhole:
        lines[1] = "PINHOLE"
        lines = lines[: value_index(lines, "# Inverse of Translation Vector") + 1]
    out = os.path.join(VAR, os.path.basename(src).replace(".txt", "_%s.txt" % tag))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return lpt.loadCameraModel(out)


def P(x):
    return lpt.Pt3D(float(x[0]), float(x[1]), float(x[2]))


def uv(p2):
    return float(p2[0]), float(p2[1])


cams = [variant(f, "tol9", tol=1e-9, nmax=80) for f in CAM_FILES]
cams_file = [lpt.loadCameraModel(f) for f in CAM_FILES]
cams_pin = [variant(f, "pinhole", pinhole=True) for f in CAM_FILES]
geo = [read_rt(f) for f in CAM_FILES]
NROW, NCOL = geo[0][3], geo[0][4]


def all_images(cam, X):
    try:
        rays, cov = cam.projectAll(P(X))
    except RuntimeError:
        return None
    return [(uv(r["pt"]), r["is_far"], r["lambda"], r["c_z"]) for r in rays if not r["is_ring"]]


def on_sensor(p, margin=0.0):
    u, v = p
    return -margin <= u < NCOL + margin and -margin <= v < NROW + margin


# ---------------------------------------------------------------- A1 census
xs = np.arange(-30.0, 30.0 + 1e-9, 1.5)
ys = np.arange(-20.0, 20.0 + 1e-9, 2.0)  # the tube axis is world y
zs = np.arange(-30.0, 30.0 + 1e-9, 1.5)
grid = np.array([(x, y, z) for x in xs for y in ys for z in zs])
counts = np.zeros((len(grid), 2), int)       # images on the sensor
counts_all = np.zeros((len(grid), 2), int)   # images anywhere
t0 = time.perf_counter()
for i, X in enumerate(grid):
    for c in range(2):
        imgs = all_images(cams[c], X)
        if imgs is None:
            continue
        counts_all[i, c] = len(imgs)
        counts[i, c] = sum(on_sensor(p) for p, *_ in imgs)
t_census = time.perf_counter() - t0
both = (counts[:, 0] >= 1) & (counts[:, 1] >= 1)
common = grid[both]
lo, hi = common.min(axis=0), common.max(axis=0)
OUT["A1"] = {
    "grid_points": int(len(grid)),
    "census_seconds": t_census,
    "per_camera_on_sensor_hist": [np.bincount(counts[:, c], minlength=3).tolist() for c in range(2)],
    "per_camera_any_hist": [np.bincount(counts_all[:, c], minlength=3).tolist() for c in range(2)],
    "common_points": int(both.sum()),
    "common_bbox_mm": [lo.tolist(), hi.tolist()],
    "common_with_2plus_images_any_cam": int(((counts[both] >= 2).any(axis=1)).sum()),
    "max_radius_in_common_mm": float(np.hypot(common[:, 0], common[:, 2]).max()),
}
print("A1", json.dumps(OUT["A1"], indent=1))

# ---------------------------------------------------------------- A2 timing
pts = common[RNG.integers(0, len(common), 20000)] + RNG.uniform(-0.75, 0.75, (20000, 3))
pts_lpt = [P(x) for x in pts]


def time_batch(cam, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        res = cam.projectBatchStatus(pts_lpt)
        best = min(best, time.perf_counter() - t0)
    ok = sum(1 for r in res if r[0])
    return best / len(pts_lpt) * 1e6, ok, res


A2 = {}
for name, cl in (("pinhole", cams_pin), ("pincyl_tol1e-9", cams), ("pincyl_file_tol1e-12", cams_file)):
    A2[name] = []
    for c in range(2):
        us, ok, _ = time_batch(cl[c])
        A2[name].append({"us_per_call": us, "ok": ok})
# lineOfSight cost
pix = []
for c in range(2):
    res = cams[c].projectBatchStatus(pts_lpt[:5000])
    pix.append([r[1] for r in res if r[0]])
A2["lineOfSight_pincyl"] = []
for c in range(2):
    t0 = time.perf_counter()
    cams[c].lineOfSightBatchStatus(pix[c])
    A2["lineOfSight_pincyl"].append((time.perf_counter() - t0) / len(pix[c]) * 1e6)
# the two tolerances agree?
_, _, r9 = time_batch(cams[0], 1)
_, _, r12 = time_batch(cams_file[0], 1)
d = [math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]) for a, b in zip(r9, r12) if a[0] and b[0]]
A2["max_px_diff_tol9_vs_tol12"] = max(d)
OUT["A2"] = A2
print("A2", json.dumps(A2, indent=1))

# ---------------------------------------------------------------- A3 epipolar curvature
box_lo, box_hi = lo - 0.75, hi + 0.75


def clip(pt, u):
    t_in, t_out = 0.0, math.inf
    for d in range(3):
        if abs(u[d]) < 1e-15:
            if pt[d] < box_lo[d] or pt[d] > box_hi[d]:
                return None
            continue
        ta, tb = (box_lo[d] - pt[d]) / u[d], (box_hi[d] - pt[d]) / u[d]
        ta, tb = min(ta, tb), max(ta, tb)
        t_in, t_out = max(t_in, ta), min(t_out, tb)
        if t_in > t_out:
            return None
    return t_in, t_out


def line_dist(q, a, b):
    ab = np.subtract(b, a)
    n = np.hypot(*ab)
    return abs((q[0] - a[0]) * ab[1] - (q[1] - a[1]) * ab[0]) / n


dev_all, dev_vis, span_px = [], [], []
for X in common[RNG.integers(0, len(common), 400)]:
    p1 = uv(cams[0].project(P(X)))
    los = cams[0].lineOfSight(lpt.Pt2D(*p1))
    o = np.array([los.pt[0], los.pt[1], los.pt[2]])
    u = np.array([los.unit_vector[0], los.unit_vector[1], los.unit_vector[2]])
    tt = clip(o, u)
    if tt is None:
        continue
    ts = np.linspace(tt[0], tt[1], 61)
    q = []
    for t in ts:
        try:
            q.append(uv(cams[1].project(P(o + t * u))))
        except RuntimeError:
            q.append(None)
    if q[0] is None or q[-1] is None:
        continue
    a, b = q[0], q[-1]
    dists = [line_dist(p, a, b) for p in q if p is not None]
    vis = [line_dist(p, a, b) for p in q if p is not None and on_sensor(p)]
    dev_all.append(max(dists))
    if vis:
        dev_vis.append(max(vis))
    span_px.append(math.hypot(b[0] - a[0], b[1] - a[1]))
OUT["A3"] = {
    "n_lines": len(dev_all),
    "chord_length_px_median": float(np.median(span_px)),
    "max_dev_px_pcts_all": np.percentile(dev_all, [50, 90, 99, 100]).tolist(),
    "max_dev_px_pcts_on_sensor": np.percentile(dev_vis, [50, 90, 99, 100]).tolist(),
}
print("A3", json.dumps(OUT["A3"], indent=1))

# ---------------------------------------------------------------- A4 astigmatism
h = 0.5
shifted = []
for c in range(2):
    shifted.append({
        "x+": variant(CAM_FILES[c], "xp", t_shift=(h, 0, 0), tol=1e-9, nmax=80),
        "x-": variant(CAM_FILES[c], "xm", t_shift=(-h, 0, 0), tol=1e-9, nmax=80),
        "y+": variant(CAM_FILES[c], "yp", t_shift=(0, h, 0), tol=1e-9, nmax=80),
        "y-": variant(CAM_FILES[c], "ym", t_shift=(0, -h, 0), tol=1e-9, nmax=80),
        "pin_x+": variant(CAM_FILES[c], "pxp", t_shift=(h, 0, 0), pinhole=True),
        "pin_x-": variant(CAM_FILES[c], "pxm", t_shift=(-h, 0, 0), pinhole=True),
    })

A4 = []
sample = common[RNG.integers(0, len(common), 300)]
for c in range(2):
    R, t, K, _, _ = geo[c]
    fx, fy = K[0, 0], K[1, 1]
    rows = []
    for X in sample:
        z_true = (R @ X + t)[2]
        s = shifted[c]
        try:
            up = uv(s["x+"].project(P(X)))[0]
            um = uv(s["x-"].project(P(X)))[0]
            vp = uv(s["y+"].project(P(X)))[1]
            vm = uv(s["y-"].project(P(X)))[1]
            pu = uv(s["pin_x+"].project(P(X)))[0]
            pm = uv(s["pin_x-"].project(P(X)))[0]
        except RuntimeError:
            continue
        z_tan = -fx * 2 * h / (up - um)
        z_sag = -fy * 2 * h / (vp - vm)
        z_pin = -fx * 2 * h / (pu - pm)  # sanity: the pinhole must give z_true
        # image Jacobian d(u,v)/dX
        J = np.zeros((2, 3))
        e = 0.01
        for k in range(3):
            dX = np.zeros(3)
            dX[k] = e
            a = uv(cams[c].project(P(X + dX)))
            b = uv(cams[c].project(P(X - dX)))
            J[:, k] = (np.subtract(a, b)) / (2 * e)
        U, S, Vt = np.linalg.svd(J)
        major = U[:, 0]
        rows.append({
            "X": X.tolist(), "r_mm": float(math.hypot(X[0], X[2])),
            "z_true": z_true, "z_tan": z_tan, "z_sag": z_sag, "z_pin_check": z_pin,
            "blur_px_per_mm_aperture": abs(1 / z_tan - 1 / z_sag) * math.sqrt(fx * fy),
            "jac_sv": S.tolist(), "jac_ratio": float(S[0] / S[1]),
            "major_axis_deg_from_u": float(math.degrees(math.atan2(major[1], major[0])) % 180),
        })
    zt = np.array([r["z_true"] for r in rows])
    A4.append({
        "camera": c + 1,
        "n": len(rows),
        "pinhole_check_max_mm": float(max(abs(r["z_pin_check"] - r["z_true"]) for r in rows)),
        "z_true_minus_z_tan_mm_pcts": np.percentile([r["z_true"] - r["z_tan"] for r in rows], [0, 50, 100]).tolist(),
        "z_true_minus_z_sag_mm_pcts": np.percentile([r["z_true"] - r["z_sag"] for r in rows], [0, 50, 100]).tolist(),
        "z_tan_minus_z_sag_mm_pcts": np.percentile([r["z_tan"] - r["z_sag"] for r in rows], [0, 50, 100]).tolist(),
        "blur_px_per_mm_aperture_pcts": np.percentile([r["blur_px_per_mm_aperture"] for r in rows], [0, 50, 100]).tolist(),
        "jac_ratio_pcts": np.percentile([r["jac_ratio"] for r in rows], [0, 50, 100]).tolist(),
        "major_axis_deg_pcts": np.percentile([r["major_axis_deg_from_u"] for r in rows], [0, 50, 100]).tolist(),
        "z_true_range": [float(zt.min()), float(zt.max())],
    })
OUT["A4"] = A4
print("A4", json.dumps(A4, indent=1))

with open(os.path.join(HERE, "probe_rig_out.json"), "w") as f:
    json.dump(OUT, f, indent=1)
