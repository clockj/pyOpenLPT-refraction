"""C2a (plan: 2026-10-07-plan.md, Phase C): the board, triangulated with the wand calibration's cameras.

    python tests/c2a_board_check.py [RESULT_FOLDER]      (default: config out_dir, C1's result)

The board of 30 Sep (3 mm pitch) at the three stage positions of the Planes recording, 3.175 mm apart (the user:
about one turn of the stage's screw), each further from cam1.  If the cameras are right, the triangulated dots of
each position lie on a plane, on a square grid of 3 mm, and the planes step by the stage's translation.

1. The dots of each image (planes.detect, as P0e), indexed on each image's own lattice (planes.index_dots).
2. Which cam2 dot is which cam1 dot: every map ij2 = M ij1 + t between the two lattices (M a symmetry of the square
   lattice, t an integer offset) is scored by the median distance between the lines of sight (cyl_model.trace of
   the result's cameras) of the pairs it implies; the best map is used.  With the two cameras side by side about
   the vertical tube the epipolar lines run nearly along the board's rows, so a map shifted by one column along a
   row can also come close: the runner-up's score is reported beside the best's.
3. The pairs, triangulated by maximum likelihood (cylinder_wand.triangulate_ml): reprojection residuals per camera.
4. Per position: a plane fitted to the points (flatness: RMS and largest distance), and a lattice in it,
   X = O + i A + j B by least squares: the pitches |A| and |B| against 3 mm, the angle between A and B, and the
   residual of the dots about the lattice.
5. Between positions: cam1's dots are followed from position to position (they move a few px), so a dot keeps its
   index; the translation of the board between positions (mean over the dots seen in both), its length against
   3.175 mm, and its angle to the board's normal and to cam1's line of sight.
Writes c2a_board_check_out.txt, c2a_board_check.json and c2a_board.png beside this script, and the dot tables
(<out_root>/Planes/c2a_dots_pos{0,1,2}.csv: indices, both cameras' pixels, the triangulated point, residuals).
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import planes  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GATE_PX = 25.0


def load_cams(lib, folder):
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    cams = [lib.cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"],
                          c["height"], np.array(c["dist"])) for c in res["cameras"]]
    tube = lib.cm.Tube(tuple(res["tube"]["radii_mm"]), tuple(res["tube"]["n"]))
    return res, cams, tube


def rays(lib, cams, k, P):
    rw = lib.cm.rows(cams, np.full(len(P), k, int))
    xy = lib.cm.undistort(np.asarray(P, float), rw)
    o, d, ok, _do, _dd = lib.cm.trace(rw, xy, tube=TUBE[0])
    return o, d, ok


def plane_fit(X):
    c = X.mean(axis=0)
    _u, s, vt = np.linalg.svd(X - c)
    n = vt[2]
    dist = (X - c) @ n
    return c, n, dist


def lattice_fit(X, ij):
    G = np.column_stack([np.ones(len(X)), ij[:, 0], ij[:, 1]])
    coef, *_ = np.linalg.lstsq(G, X, rcond=None)
    O, A, B = coef[0], coef[1], coef[2]
    res = X - G @ coef
    return O, A, B, res


def pitch_se(X, ij, res):
    """Standard errors of |A| and |B| from the lattice fit's residual scatter."""
    G = np.column_stack([np.ones(len(X)), ij[:, 0], ij[:, 1]])
    s2 = float(np.sum(res ** 2)) / max(3 * (len(X) - 3), 1)
    cov = s2 * np.linalg.inv(G.T @ G)
    return math.sqrt(cov[1, 1]), math.sqrt(cov[2, 2])


TUBE = [None]


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    pl = cfg["planes"]
    log = case.Log(os.path.join(HERE, "c2a_board_check"))
    lib = case.library()
    res, cams, tube = load_cams(lib, folder)
    TUBE[0] = tube
    pitch = float(pl["pitch_mm"][0])
    step_nom = float(pl["step_mm_nominal"])
    out_dir = pl["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log(f"C2a: the board triangulated with the cameras of {folder} (f {res['cameras'][0]['f_px']:.0f} / "
        f"{res['cameras'][1]['f_px']:.0f} px)")
    out = {"cameras": folder, "positions": {}}
    prev1 = None                         # cam1's dots and indices of the previous position, to carry the indices
    allX = {}
    for pos in pl["positions"]:
        det = dict(pl["detect"])
        P = {}
        for cam in (1, 2):
            d = dict(det)
            d.update(pl.get("detect_overrides", {}).get(f"cam{cam}", {}))
            r = planes.detect(planes.read(cfg, pos, cam), d)
            P[cam] = np.array([[q["u"], q["v"]] for q in r["dots"] if q["ok"]], float)
        ij1, _H1, res1 = planes.index_dots(P[1], seed_xy=(640, 400))
        ij2, _H2, res2 = planes.index_dots(P[2], seed_xy=(640, 400))
        # carry cam1's indices from the previous position: the dots move only a few px
        if prev1 is not None:
            Pp, ijp = prev1
            dmin = np.hypot(P[1][:, None, 0] - Pp[None, :, 0], P[1][:, None, 1] - Pp[None, :, 1])
            nn = np.argmin(dmin, axis=1)
            close = dmin[np.arange(len(P[1])), nn] < GATE_PX
            offs = ijp[nn[close]] - ij1[close]
            vals, cnt = np.unique(offs, axis=0, return_counts=True)
            shift = vals[np.argmax(cnt)]
            ij1 = ij1 + shift
            agree = int(np.sum(np.all(offs == shift, axis=1)))
            log(f"\n  position {pos}: cam1's indices carried from the previous position: {agree} of {int(close.sum())} "
                f"dots followed (gate {GATE_PX:g} px) agree on the shift {shift.tolist()}")
        else:
            log(f"\n  position {pos}: cam1's indices start here (the dot nearest the image centre is (0, 0))")
        prev1 = (P[1], ij1)
        o1, d1, ok1 = rays(lib, cams, 0, P[1])
        o2, d2, ok2 = rays(lib, cams, 1, P[2])
        dist = planes.skew_distances(o1, d1, o2, d2)
        dist[~ok1, :] = np.inf
        dist[:, ~ok2] = np.inf
        cands = planes.correspondences(ij1, ij2, dist)
        best = cands[0]
        runner = next((c for c in cands[1:] if c["M"] != best["M"] or c["t"] != best["t"]), None)
        log(f"    dots: cam1 {len(P[1])}, cam2 {len(P[2])}; lattice maps scored: {len(cands)}")
        log(f"    best map M {best['M']}, t {best['t']}: {best['n']} pairs, median line distance "
            f"{1e3 * best['median_mm']:.1f} um")
        for c in cands[1:4]:
            log(f"      next: M {c['M']}, t {c['t']}: {c['n']} pairs, median {1e3 * c['median_mm']:.1f} um "
                f"({c['median_mm'] / best['median_mm']:.1f} x the best)")
        a_idx = np.array([p[0] for p in best["pairs"]])
        b_idx = np.array([p[1] for p in best["pairs"]])
        n = len(a_idx)
        # maximum-likelihood triangulation of every pair from both cameras
        mid = []
        for a, b in zip(a_idx, b_idx):
            w0 = o1[a] - o2[b]
            A11, A12, A22 = d1[a] @ d1[a], -(d1[a] @ d2[b]), d2[b] @ d2[b]
            rhs = np.array([-(d1[a] @ w0), d2[b] @ w0])
            s, t = np.linalg.solve(np.array([[A11, A12], [A12, A22]]), rhs)
            mid.append(0.5 * (o1[a] + s * d1[a] + o2[b] + t * d2[b]))
        X0 = np.array(mid)
        cam_idx = np.concatenate([np.zeros(n, int), np.ones(n, int)])
        uv = np.vstack([P[1][a_idx], P[2][b_idx]])
        point = np.concatenate([np.arange(n), np.arange(n)])
        X, r, _cov = lib.cw.triangulate_ml(cams, tube, cam_idx, uv, point, X0)
        r1, r2 = r[:n], r[n:]
        ij = ij1[a_idx]
        c, nrm, pd = plane_fit(X)
        O, A, B, lres = lattice_fit(X, ij)
        sA, sB = pitch_se(X, ij, lres)
        ang = math.degrees(math.acos(float(A @ B) / (np.linalg.norm(A) * np.linalg.norm(B))))
        if nrm @ (cams[0].C - c) < 0:
            nrm = -nrm                                           # the normal towards cam1
        los1 = cams[0].C - c
        los1 /= np.linalg.norm(los1)
        rec = {"n_cam1": len(P[1]), "n_cam2": len(P[2]), "pairs": n,
               "map": {"M": best["M"], "t": best["t"], "median_um": 1e3 * best["median_mm"]},
               "runner_up": ({"M": runner["M"], "t": runner["t"], "median_um": 1e3 * runner["median_mm"],
                              "n": runner["n"]} if runner else None),
               "reproj_rms_px": {"cam1": float(np.sqrt(np.nanmean(np.sum(r1 ** 2, axis=1)))),
                                 "cam2": float(np.sqrt(np.nanmean(np.sum(r2 ** 2, axis=1))))},
               "plane": {"centre": c.tolist(), "normal": nrm.tolist(), "rms_um": 1e3 * float(np.sqrt(np.mean(pd ** 2))),
                         "max_um": 1e3 * float(np.max(np.abs(pd))),
                         "normal_vs_cam1_los_deg": math.degrees(math.acos(min(1.0, abs(float(nrm @ los1)))))},
               "lattice": {"pitch_A_mm": float(np.linalg.norm(A)), "pitch_B_mm": float(np.linalg.norm(B)),
                           "se_A_mm": sA, "se_B_mm": sB, "angle_deg": ang,
                           "rms_um": 1e3 * float(np.sqrt(np.mean(np.sum(lres ** 2, axis=1)))),
                           "A": A.tolist(), "B": B.tolist(), "O": O.tolist()}}
        out["positions"][pos] = rec
        allX[pos] = (ij, X)
        log(f"    triangulated {n} dots: reprojection RMS cam1 {rec['reproj_rms_px']['cam1']:.3f} px, cam2 "
            f"{rec['reproj_rms_px']['cam2']:.3f} px")
        log(f"    plane: centre {np.round(c, 2).tolist()} mm, normal {np.round(nrm, 4).tolist()} ({rec['plane']['normal_vs_cam1_los_deg']:.2f} deg "
            f"from cam1's line of sight); flatness RMS {rec['plane']['rms_um']:.1f} um, largest {rec['plane']['max_um']:.1f} um")
        la = rec["lattice"]
        log(f"    lattice: pitch along i {la['pitch_A_mm']:.5f} +- {la['se_A_mm']:.5f} mm "
            f"({100 * (la['pitch_A_mm'] / pitch - 1):+.3f} %), along j {la['pitch_B_mm']:.5f} +- {la['se_B_mm']:.5f} mm "
            f"({100 * (la['pitch_B_mm'] / pitch - 1):+.3f} %); angle {la['angle_deg']:.3f} deg; dots about the lattice "
            f"RMS {la['rms_um']:.1f} um")
        with open(os.path.join(out_dir, f"c2a_dots_pos{pos}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["i", "j_down", "u1", "v1", "u2", "v2", "X", "Y", "Z", "du1", "dv1", "du2", "dv2"])
            for k in range(n):
                w.writerow([int(ij[k, 0]), int(ij[k, 1])] + [f"{v:.4f}" for v in P[1][a_idx[k]]] +
                           [f"{v:.4f}" for v in P[2][b_idx[k]]] + [f"{v:.5f}" for v in X[k]] +
                           [f"{v:.4f}" for v in r1[k]] + [f"{v:.4f}" for v in r2[k]])
    # the translations between positions, over the dots seen at both
    log("\n  Between positions (mean over the dots triangulated at both):")
    out["steps"] = {}
    names = pl["positions"]
    for p0, p1 in zip(names[:-1], names[1:]):
        ij0, X0_ = allX[p0]
        ij1_, X1_ = allX[p1]
        k0 = {tuple(v): k for k, v in enumerate(ij0)}
        common = [(k0[tuple(v)], k) for k, v in enumerate(ij1_) if tuple(v) in k0]
        D = np.array([X1_[b] - X0_[a] for a, b in common])
        T = D.mean(axis=0)
        sd = D.std(axis=0)
        nrm = np.array(out["positions"][p0]["plane"]["normal"])
        cvec = np.array(out["positions"][p0]["plane"]["centre"])
        los1 = cams[0].C - cvec
        los1 /= np.linalg.norm(los1)
        L = float(np.linalg.norm(T))
        rec = {"dots": len(common), "translation_mm": T.tolist(), "length_mm": L, "sd_mm": sd.tolist(),
               "vs_nominal_pct": 100 * (L / step_nom - 1),
               "angle_to_normal_deg": math.degrees(math.acos(min(1.0, abs(float(T @ nrm)) / L))),
               "angle_to_cam1_los_deg": math.degrees(math.acos(min(1.0, abs(float(T @ los1)) / L))),
               "away_from_cam1": bool(float(T @ los1) < 0)}
        out["steps"][f"{p0}->{p1}"] = rec
        log(f"    {p0} -> {p1}: {len(common)} dots; translation {np.round(T, 4).tolist()} mm, length {L:.4f} mm "
            f"({rec['vs_nominal_pct']:+.2f} % of {step_nom}), scatter over the dots {np.round(1e3 * sd, 1).tolist()} um; "
            f"{rec['angle_to_normal_deg']:.2f} deg from the board's normal, {rec['angle_to_cam1_los_deg']:.2f} deg from "
            f"cam1's line of sight, {'away from' if rec['away_from_cam1'] else 'towards'} cam1")
    # figure: the three boards seen along the tube's axis (X, Z) and face-on (X, Y)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, pos in enumerate(names):
        ij, X = allX[pos]
        ax[0].plot(X[:, 0], X[:, 2], "o", ms=3, color=f"C{k}", label=f"position {pos}")
        ax[1].plot(X[:, 0], X[:, 1], "o", ms=3, color=f"C{k}", label=f"position {pos}")
    ax[0].set_xlabel("X (mm, across)")
    ax[0].set_ylabel("Z (mm, towards cam1)")
    ax[0].set_title("the triangulated boards, seen along the tube's axis", fontsize=9)
    ax[0].set_aspect("equal")
    ax[1].set_xlabel("X (mm)")
    ax[1].set_ylabel("Y (mm, along the axis)")
    ax[1].set_title("seen from cam1's side", fontsize=9)
    ax[1].set_aspect("equal")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "c2a_board.png"), dpi=110)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
