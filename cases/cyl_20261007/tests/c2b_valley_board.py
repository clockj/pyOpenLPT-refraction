"""C2b, second form (plan: 2026-10-07-plan.md, Phase C): which point of the wand's f / D valley does the board
prefer?  The board triangulated with every wand fit along the valley.

    python tests/c2b_valley_board.py [FOLDER ...]      (default: C1's result and C1-T's parts in <out_dir>/checks/)

c2b_f_board.py fits each camera to the board on its own; with three positions 2.5 mm apart that does not determine f
(its variants disagree by 10-50 %).  Here both cameras are used together: each wand fit of the valley (C1, and the
halves and quarters of C1-T, whose f differ by up to 25 %) triangulates the board's dots (the pairs and maps of C2a,
found with C1's cameras), and the board's known shape scores it:
  - per position: the flatness (RMS distance from the fitted plane), the lattice residual (RMS of the dots about the
    best square-ish lattice X = O + i A + j B), the pitches |A| and |B| against 3 mm and the angle between A and B;
  - between positions: the translation's length and its angle to the board's normal (a stage moves the board
    parallel to itself: the three planes should be parallel);
  - the reprojection RMS of the triangulation per camera (each fit's own consistency with the board's pixels).
A point of the valley whose cameras distort space shows as a board that is not flat, not square or whose planes are
not parallel; a clear minimum along the valley would pin f.
Writes c2b_valley_board_out.txt and c2b_valley_board.json beside this script.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import planes  # noqa: E402

import numpy as np  # noqa: E402

GATE_PX = 25.0


def load_cams(lib, folder):
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    cams = [lib.cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"],
                          c["height"], np.array(c["dist"])) for c in res["cameras"]]
    tube = lib.cm.Tube(tuple(res["tube"]["radii_mm"]), tuple(res["tube"]["n"]))
    return res, cams, tube


def rays(lib, cams, tube, k, P):
    rw = lib.cm.rows(cams, np.full(len(P), k, int))
    xy = lib.cm.undistort(np.asarray(P, float), rw)
    o, d, ok, _do, _dd = lib.cm.trace(rw, xy, tube)
    return o, d, ok


def pairs_of_c1(lib, cfg):
    """Detect, index and pair the dots of every position as C2a does, with C1's cameras."""
    pl = cfg["planes"]
    _res, cams, tube = load_cams(lib, cfg["out_dir"])
    out = []
    prev1 = None
    for pos in pl["positions"]:
        P = {}
        for cam in (1, 2):
            d = dict(pl["detect"])
            d.update(pl.get("detect_overrides", {}).get(f"cam{cam}", {}))
            r = planes.detect(planes.read(cfg, pos, cam), d)
            P[cam] = np.array([[q["u"], q["v"]] for q in r["dots"] if q["ok"]], float)
        ij1, _h, _r = planes.index_dots(P[1], seed_xy=(640, 400))
        ij2, _h, _r = planes.index_dots(P[2], seed_xy=(640, 400))
        if prev1 is not None:
            Pp, ijp = prev1
            dmin = np.hypot(P[1][:, None, 0] - Pp[None, :, 0], P[1][:, None, 1] - Pp[None, :, 1])
            nn = np.argmin(dmin, axis=1)
            close = dmin[np.arange(len(P[1])), nn] < GATE_PX
            offs = ijp[nn[close]] - ij1[close]
            vals, cnt = np.unique(offs, axis=0, return_counts=True)
            ij1 = ij1 + vals[np.argmax(cnt)]
        prev1 = (P[1], ij1)
        o1, d1, ok1 = rays(lib, cams, tube, 0, P[1])
        o2, d2, ok2 = rays(lib, cams, tube, 1, P[2])
        dist = planes.skew_distances(o1, d1, o2, d2)
        dist[~ok1, :] = np.inf
        dist[:, ~ok2] = np.inf
        best = planes.correspondences(ij1, ij2, dist)[0]
        a = np.array([p[0] for p in best["pairs"]])
        b = np.array([p[1] for p in best["pairs"]])
        out.append({"pos": pos, "uv1": P[1][a], "uv2": P[2][b], "ij": ij1[a], "median_um": 1e3 * best["median_mm"]})
    return out


def score(lib, cams, tube, data):
    recs = []
    Xs = []
    for d in data:
        n = len(d["ij"])
        # start: the midpoints of the two lines of sight
        o1, d1, _ = rays(lib, cams, tube, 0, d["uv1"])
        o2, d2, _ = rays(lib, cams, tube, 1, d["uv2"])
        w0 = o1 - o2
        a11 = np.sum(d1 * d1, axis=1)
        a12 = -np.sum(d1 * d2, axis=1)
        a22 = np.sum(d2 * d2, axis=1)
        r1 = -np.sum(d1 * w0, axis=1)
        r2 = np.sum(d2 * w0, axis=1)
        det = a11 * a22 - a12 * a12
        s = (r1 * a22 - a12 * r2) / det
        t = (a11 * r2 - a12 * r1) / det
        X0 = 0.5 * (o1 + s[:, None] * d1 + o2 + t[:, None] * d2)
        cam_idx = np.concatenate([np.zeros(n, int), np.ones(n, int)])
        uv = np.vstack([d["uv1"], d["uv2"]])
        point = np.concatenate([np.arange(n), np.arange(n)])
        X, r, _cov = lib.cw.triangulate_ml(cams, tube, cam_idx, uv, point, X0)
        c = X.mean(axis=0)
        _u, _s, vt = np.linalg.svd(X - c)
        nrm = vt[2]
        if nrm @ (cams[0].C - c) < 0:
            nrm = -nrm
        pdist = (X - c) @ nrm
        G = np.column_stack([np.ones(n), d["ij"][:, 0], d["ij"][:, 1]])
        coef, *_ = np.linalg.lstsq(G, X, rcond=None)
        A, B = coef[1], coef[2]
        lres = X - G @ coef
        recs.append({"pos": d["pos"], "flat_rms_um": 1e3 * float(np.sqrt(np.mean(pdist ** 2))),
                     "lattice_rms_um": 1e3 * float(np.sqrt(np.mean(np.sum(lres ** 2, axis=1)))),
                     "pitch_A": float(np.linalg.norm(A)), "pitch_B": float(np.linalg.norm(B)),
                     "angle_deg": math.degrees(math.acos(float(A @ B) / (np.linalg.norm(A) * np.linalg.norm(B)))),
                     "reproj1_px": float(np.sqrt(np.nanmean(np.sum(r[:n] ** 2, axis=1)))),
                     "reproj2_px": float(np.sqrt(np.nanmean(np.sum(r[n:] ** 2, axis=1)))),
                     "normal": nrm.tolist()})
        Xs.append((d["ij"], X))
    steps = []
    for (ij0, X0_), (ij1_, X1_), r0 in zip(Xs[:-1], Xs[1:], recs[:-1]):
        k0 = {tuple(v): k for k, v in enumerate(ij0)}
        common = [(k0[tuple(v)], k) for k, v in enumerate(ij1_) if tuple(v) in k0]
        D = np.array([X1_[b] - X0_[a] for a, b in common])
        T = D.mean(axis=0)
        nrm = np.array(r0["normal"])
        steps.append({"length_mm": float(np.linalg.norm(T)),
                      "angle_to_normal_deg": math.degrees(math.acos(min(1.0, abs(float(T @ nrm)) / np.linalg.norm(T))))})
    normals = [np.array(r["normal"]) for r in recs]
    para = [math.degrees(math.acos(min(1.0, abs(float(a @ b))))) for a, b in zip(normals[:-1], normals[1:])]
    return recs, steps, para


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "c2b_valley_board"))
    lib = case.library()
    folders = sys.argv[1:] or ([cfg["out_dir"]] + [os.path.join(cfg["out_dir"], "checks", n)
                                                     for n in ("h1", "h2", "q1", "q2", "q3", "q4")])
    data = pairs_of_c1(lib, cfg)
    log("C2b (valley): the board triangulated with each wand fit along the f / D valley (pairs and maps of C2a, "
        "found with C1's cameras: median line distances " + ", ".join(f"{d['median_um']:.1f}" for d in data) + " um)")
    log(f"  {'fit':<34s} {'f cam1':>7s} {'f cam2':>7s}  flatness RMS (um)    lattice RMS (um)    pitch A, B (% off 3 mm), "
        "mean         angle (deg)   steps (mm)        planes' tilt (deg)   reprojection (px)")
    out = {"fits": {}}
    for folder in folders:
        res, cams, tube = load_cams(lib, folder)
        recs, steps, para = score(lib, cams, tube, data)
        name = os.path.relpath(folder, cfg["out_dir"]) if folder != cfg["out_dir"] else "C1"
        f1, f2 = res["cameras"][0]["f_px"], res["cameras"][1]["f_px"]
        pa = np.mean([100 * (r["pitch_A"] / 3.0 - 1) for r in recs])
        pb = np.mean([100 * (r["pitch_B"] / 3.0 - 1) for r in recs])
        out["fits"][name] = {"folder": folder, "f": [f1, f2], "positions": recs, "steps": steps,
                             "planes_tilt_deg": para, "pitch_A_pct": pa, "pitch_B_pct": pb}
        log(f"  {name:<34s} {f1:7.0f} {f2:7.0f}  " + " ".join(f"{r['flat_rms_um']:5.1f}" for r in recs) + "   "
            + " ".join(f"{r['lattice_rms_um']:5.1f}" for r in recs) + f"   {pa:+.3f} {pb:+.3f}        "
            + " ".join(f"{r['angle_deg']:.3f}" for r in recs) + "   "
            + " ".join(f"{s['length_mm']:.3f}" for s in steps) + "   " + " ".join(f"{p:.3f}" for p in para)
            + "   " + " ".join(f"{r['reproj1_px']:.2f}/{r['reproj2_px']:.2f}" for r in recs))
    log.save(out)


if __name__ == "__main__":
    main()
