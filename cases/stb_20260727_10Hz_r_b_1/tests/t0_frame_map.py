"""T0: map the old view volume into the new camFiles' world frame and derive the new config's view volume and voxel
size (2026-10-04-plan.md, section 4).

A grid filling the old view volume is projected with the old cameras; those pixels are triangulated with the new
cameras.  The mapped grid's bounding box, clipped to the channel's water (near walls from the new camFiles, far walls at
the inside widths), is the new view volume; a similarity fit X_new = s R X_old + t gives the scale s, and the voxel
size becomes 0.105 * s mm.  Also: how many cameras see each part of both volumes.

Writes t0_frame_map.json and t0_frame_map_out.txt beside this script.
    python tests/t0_frame_map.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402

OUT_JSON = os.path.join(case.TESTS, "t0_frame_map.json")
OUT_TXT = os.path.join(case.TESTS, "t0_frame_map_out.txt")
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _lines.append(s)


def umeyama(A: np.ndarray, B: np.ndarray):
    """Similarity B ~ s R A + t (least squares): (s, R, t)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    t = mb - s * R @ ma
    return s, R, t


def rot_axis_angle(R: np.ndarray):
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    w, V = np.linalg.eig(R)
    ax = np.real(V[:, np.argmin(np.abs(w - 1))])
    return ang, ax / np.linalg.norm(ax)


def clip_box(vv, a, c):
    """Shrink the axis-aligned box vv along the axis of a's largest component so that a.X >= c holds on the box."""
    vv = list(vv)
    k = int(np.argmax(np.abs(a)))
    others = [j for j in range(3) if j != k]
    # the most demanding corner of the other two axes
    worst = sum(min(a[j] * vv[2 * j], a[j] * vv[2 * j + 1]) for j in others)
    bound = (c - worst) / a[k]
    if a[k] > 0:
        vv[2 * k] = max(vv[2 * k], bound)
    else:
        vv[2 * k + 1] = min(vv[2 * k + 1], bound)
    return vv


def visibility(cams, X):
    seen = np.zeros((len(X), len(cams)), bool)
    for i, c in enumerate(cams):
        uv, ok = case.project(c, X)
        seen[:, i] = ok & case.in_image(uv, c)
    k = seen.sum(1)
    return seen, {f">={m}": float(np.mean(k >= m)) for m in (2, 3, 4)} | \
        {f"cam{i}": float(seen[:, i].mean()) for i in range(len(cams))}


def main():
    cfg = case.load_config()
    old = case.load_cameras(case.camfile_paths(cfg, "old"))
    new = case.load_cameras(case.camfile_paths(cfg, "new"))
    res = {"old_view_volume": cfg["old_view_volume"], "old_voxel_to_mm": cfg["old_voxel_to_mm"]}

    log("== cameras (centre, water-side point, normal, thickness, n camera side first, proj_tol, proj_nmax)")
    plates = {}
    for name, cams in (("old", old), ("new", new)):
        plates[name] = []
        for i, c in enumerate(cams):
            p = case.plate_of(c)
            plates[name].append(p)
            log(f"{name} cam{i}: C {np.round(p['centre'], 2)}  water {np.round(p['water_pt'], 2)}  "
                f"n {np.round(p['normal'], 5)}  t {p['thickness']:.4f}  idx {p['n']}  tol {p['proj_tol']:g}  "
                f"nmax {p['proj_nmax']}")
    res["plates"] = plates

    # ---- the mapping: old grid -> old pixels -> new triangulation
    vv_old = cfg["old_view_volume"]
    G = case.box_grid(vv_old, 21)
    uv_old, ok = [], np.ones(len(G), bool)
    for c in old:
        uv, o = case.project(c, G)
        uv_old.append(uv)
        ok &= o
    log(f"\n== mapping: {len(G)} grid points of the old view volume, {ok.sum()} project in all four old cameras")
    origins, dirs, ok2 = [], [], ok.copy()
    for c, uv in zip(new, uv_old):
        o, d, g = case.lines_of_sight(c, np.where(ok[:, None], uv, 1000.0))
        origins.append(o)
        dirs.append(d)
        ok2 &= g
    sel = ok2
    Xn, dist = case.triangulate([o[sel] for o in origins], [d[sel] for d in dirs])
    Xo = G[sel]
    # reprojection of the mapped points in the new cameras against the old pixels
    rep = []
    for c, uv in zip(new, uv_old):
        un, _ = case.project(c, Xn)
        rep.append(np.linalg.norm(un - uv[sel], axis=1))
    rep = np.array(rep)
    log(f"   triangulated {sel.sum()} points; distance to the four lines: median {np.median(dist):.4f}, "
        f"p95 {np.percentile(dist, 95):.4f}, max {dist.max():.4f} mm")
    log("   reprojection of the mapped points in the new cameras vs the old pixels (px), median / p95 / max:")
    for i in range(4):
        log(f"     cam{i}: {np.median(rep[i]):.3f} / {np.percentile(rep[i], 95):.3f} / {rep[i].max():.3f}")
    s, R, t = umeyama(Xo, Xn)
    ang, ax = rot_axis_angle(R)
    fit = (s * (R @ Xo.T)).T + t
    r = np.linalg.norm(fit - Xn, axis=1)
    M, *_ = np.linalg.lstsq(np.c_[Xo, np.ones(len(Xo))], Xn, rcond=None)
    sv = np.linalg.svd(M[:3].T, compute_uv=False)
    aff = np.linalg.norm(np.c_[Xo, np.ones(len(Xo))] @ M - Xn, axis=1)
    log(f"   similarity X_new = s R X_old + t: s = {s:.5f}, rotation {ang:.3f} deg about {np.round(ax, 4)}, "
        f"t = {np.round(t, 3)}")
    log(f"   R = {np.round(R, 5).tolist()}")
    log(f"   residual of the similarity: RMS {np.sqrt(np.mean(r ** 2)):.3f}, max {r.max():.3f} mm; "
        f"affine: RMS {np.sqrt(np.mean(aff ** 2)):.3f}, max {aff.max():.3f} mm, singular values {np.round(sv, 4)}")
    res["mapping"] = {"n_points": int(sel.sum()), "tri_dist_mm": {"median": float(np.median(dist)),
                      "p95": float(np.percentile(dist, 95)), "max": float(dist.max())},
                      "reproj_px": {f"cam{i}": {"median": float(np.median(rep[i])),
                                                "p95": float(np.percentile(rep[i], 95)),
                                                "max": float(rep[i].max())} for i in range(4)},
                      "similarity": {"s": float(s), "R": R, "t": t, "angle_deg": float(ang), "axis": ax,
                                     "rms_mm": float(np.sqrt(np.mean(r ** 2))), "max_mm": float(r.max())},
                      "affine": {"rms_mm": float(np.sqrt(np.mean(aff ** 2))), "max_mm": float(aff.max()),
                                 "singular_values": sv}}

    # ---- the mapped box and the channel
    lo, hi = Xn.min(0), Xn.max(0)
    vv_map = [float(np.floor(lo[0])), float(np.ceil(hi[0])), float(np.floor(lo[1])), float(np.ceil(hi[1])),
              float(np.floor(lo[2])), float(np.ceil(hi[2]))]
    log(f"\n== the old view volume mapped into the new frame (bounding box, rounded out to 1 mm): {vv_map}")
    log(f"   sizes {vv_map[1] - vv_map[0]:.0f} x {vv_map[3] - vv_map[2]:.0f} x {vv_map[5] - vv_map[4]:.0f} mm "
        f"(old {vv_old[1] - vv_old[0]:.0f} x {vv_old[3] - vv_old[2]:.0f} x {vv_old[5] - vv_old[4]:.0f})")
    ch = cfg["channel"]
    vv = list(vv_map)
    walls = []
    for w, (cams, width) in enumerate(zip(ch["wall_cams"], ch["inside_mm"])):
        p = plates["new"][cams[0]]
        for j in cams[1:]:
            q = plates["new"][j]
            dn = np.degrees(np.arccos(np.clip(p["normal"] @ q["normal"], -1, 1)))
            dd = float(p["normal"] @ (q["water_pt"] - p["water_pt"]))
            log(f"   wall {w + 1}: cam{cams[0]} and cam{j} planes differ by {dn:.2e} deg, {dd:.2e} mm")
        nvec, c0 = p["normal"], float(p["normal"] @ p["water_pt"])
        walls.append({"cams": cams, "normal": nvec, "water_side": c0, "inside_mm": width})
        vv = clip_box(vv, nvec, c0 + ch["margin_mm"])              # inside the near wall
        vv = clip_box(vv, -nvec, -(c0 + width - ch["margin_mm"]))  # inside the far wall
        log(f"   wall {w + 1} (cams {cams}): normal {np.round(nvec, 6)}, water side n.X = {c0:.3f} mm, "
            f"far wall at n.X = {c0 + width:.3f} mm")
    vv_new = [round(float(np.ceil(vv[0] * 10) / 10), 1), round(float(np.floor(vv[1] * 10) / 10), 1),
              round(float(np.ceil(vv[2] * 10) / 10), 1), round(float(np.floor(vv[3] * 10) / 10), 1),
              round(float(np.ceil(vv[4] * 10) / 10), 1), round(float(np.floor(vv[5] * 10) / 10), 1)]
    clipped = [a != b for a, b in zip(vv_new, vv_map)]
    log(f"   clipped to the water ({ch['margin_mm']} mm from each wall): {vv_new}  (faces moved: {clipped})")
    voxel = round(cfg["old_voxel_to_mm"] * s, 4)
    log(f"   voxel size 0.105 x s = {cfg['old_voxel_to_mm'] * s:.5f} -> {voxel} mm "
        f"({(vv_new[1] - vv_new[0]) / voxel:.0f} voxels across x)")
    res["mapped_box"] = vv_map
    res["walls"] = walls
    res["new_view_volume"] = vv_new
    res["new_voxel_to_mm"] = voxel

    # ---- how much each camera sees
    log("\n== visibility (share of random points on the sensor)")
    _, vis_old = visibility(old, case.box_random(vv_old, 20000, 1))
    _, vis_new = visibility(new, case.box_random(vv_new, 20000, 2))
    _, vis_map = visibility(new, case.box_random(vv_map, 20000, 3))
    for name, v in (("old cams, old box", vis_old), ("new cams, new box", vis_new),
                    ("new cams, mapped box (unclipped)", vis_map)):
        log(f"   {name}: " + ", ".join(f"{k} {v[k]:.3f}" for k in v))
    res["visibility"] = {"old": vis_old, "new": vis_new, "mapped_unclipped": vis_map}

    # ---- the cameras: old centres mapped vs new centres
    log("\n== camera centres: old mapped by the similarity vs new (mm)")
    for i in range(4):
        co = s * R @ plates["old"][i]["centre"] + t
        cn = plates["new"][i]["centre"]
        log(f"   cam{i}: mapped old {np.round(co, 1)}  new {np.round(cn, 1)}  |diff| {np.linalg.norm(co - cn):.1f}")

    case.save_json(OUT_JSON, res)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT_JSON}\n      {OUT_TXT}")


if __name__ == "__main__":
    main()
