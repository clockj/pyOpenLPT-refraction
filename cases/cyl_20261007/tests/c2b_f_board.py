"""C2b (plan: 2026-10-07-plan.md, Phase C): f of each camera from the board alone.

    python tests/c2b_f_board.py [RESULT_FOLDER]      (default: config out_dir, C1's result; after c2a_board_check.py)

Each camera on its own, as the 30 Sep case fitted cam1 (cases/cyl_20260930_T0_cam1/cylfit.py), but with this
recording's three positions: the board's dots at X = T0 + R_b p (i, j, 0) + k t for position k = 0, 1, 2 (one pose
for the board, tied from position to position by the stage's translation t; pitch p = 3 mm), seen through the tube
(cyl_model, the PINCYL model the wand calibration fits; tube of config.json, axis = Y through the origin).
Unknowns: the camera's rotation (3), its distance from the axis D (1; its azimuth about the axis and its height
along it are the tube's symmetry and held at 0), f (1, fx = fy, principal point at the image centre, no distortion,
as the wand calibration), the board's rotation and position (6) and t (3), or t's direction only (2) with its
length held at the nominal 3.175 mm.  Least squares on the reprojection error (scipy least_squares, Jacobian from
cyl_model.project's exact derivatives), started from the wand calibration's camera and C2a's triangulated board,
re-anchored until the parameters stop moving.  Covariance from the Jacobian, scaled by the residual variance.
Variants:
  - "t free": t free (3);
  - "t held": |t| held at 3.175 mm;
  - "t free + dots": with per-dot corrections of the board (each dot's own offset in the board's plane, the same
    at every position, without their affine part, which the camera cannot tell from its pose and f: the 30 Sep
    board's dots lie 10-50 um off the nominal grid).
The dots: each camera's own (cam2 sees more than cam1), indexed on cam1's lattice through C2a's map.
Reported per camera and variant: f and D with standard errors, f / D, the residual RMS, the step |t|; against the
wand calibration's f and D (and the f / D line of its parts, C1-T, if c1t_variants.json exists).
Writes c2b_f_board_out.txt and c2b_f_board.json beside this script.
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
from scipy.optimize import least_squares  # noqa: E402


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def expso3(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = np.asarray(w, float) / th
    K = skew(k)
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def board_basis(ij, pitch):
    """Orthonormal basis of per-dot in-plane corrections [dx_0, dy_0, dx_1, ...] without their affine part."""
    X = pitch * ij[:, 0].astype(float)
    Y = pitch * ij[:, 1].astype(float)
    n = len(X)
    M = np.zeros((2 * n, 6))
    M[0::2, 0] = 1
    M[1::2, 1] = 1
    M[0::2, 2] = X
    M[0::2, 3] = Y
    M[1::2, 4] = X
    M[1::2, 5] = Y
    U, _s, _vt = np.linalg.svd(M, full_matrices=True)
    return U[:, 6:]


class Fit:
    """One camera, the board at three positions."""

    def __init__(self, lib, tube, cam0, obs, pitch, board0, mode, step_nom):
        self.cm = lib.cm
        self.tube = tube
        self.pitch = pitch
        self.mode = mode
        self.step_nom = step_nom
        # observations: per row (position k, board index (i, j_up), dot id, uv)
        self.k = obs["k"]
        self.ij = obs["ij"]
        self.uv = obs["uv"]
        self.dot = obs["dot"]
        self.n_dots = int(self.dot.max()) + 1
        self.dot_ij = obs["dot_ij"]
        # the anchor
        self.R, self.C, self.f = cam0.R.copy(), cam0.C.copy(), float(cam0.f)
        self.cx, self.cy, self.W, self.H = cam0.cx, cam0.cy, cam0.width, cam0.height
        self.Rb, self.T0, self.t = board0["Rb"].copy(), board0["T0"].copy(), board0["t"].copy()
        if mode == "t held":
            self.t *= step_nom / np.linalg.norm(self.t)
        self.corr = np.zeros(2 * self.n_dots)
        self.Bc = board_basis(self.dot_ij, pitch) if mode == "t free + dots" else np.zeros((2 * self.n_dots, 0))
        self.nt = 2 if mode == "t held" else 3

    def sizes(self):
        return 5 + 6 + self.nt + self.Bc.shape[1]

    def unpack(self, x):
        w, dD, dlf = x[0:3], x[3], x[4]
        wb, dT0 = x[5:8], x[8:11]
        R = self.R @ expso3(w)
        C = self.C + np.array([0.0, 0.0, dD])
        f = self.f * math.exp(dlf)
        Rb = self.Rb @ expso3(wb)
        T0 = self.T0 + dT0
        if self.mode == "t held":
            u0 = self.t / np.linalg.norm(self.t)
            E = np.linalg.svd(u0[None, :])[2][1:].T
            t = self.step_nom * (u0 + E @ x[11:13])
            t *= self.step_nom / np.linalg.norm(t)
            off = 13
        else:
            t = self.t + x[11:14]
            off = 14
        corr = self.corr + self.Bc @ x[off:]
        return R, C, f, Rb, T0, t, corr

    def points(self, Rb, T0, t, corr):
        q = np.column_stack([self.pitch * self.ij[:, 0] + corr[2 * self.dot],
                             self.pitch * self.ij[:, 1] + corr[2 * self.dot + 1], np.zeros(len(self.ij))])
        return T0[None, :] + q @ Rb.T + self.k[:, None] * t[None, :], q

    def camera(self, R, C, f):
        return self.cm.Camera(R, C, f, self.cx, self.cy, self.W, self.H, np.zeros(5))

    def residual_jac(self, x, jac=True):
        R, C, f, Rb, T0, t, corr = self.unpack(x)
        X, q = self.points(Rb, T0, t, corr)
        cam = self.camera(R, C, f)
        rw = self.cm.rows([cam], np.zeros(len(X), int))
        xy0 = self.cm.undistort(self.uv, rw)
        pr = self.cm.project(rw, X, xy0, self.tube, jac=jac)
        r = (pr.uv - self.uv)
        ok = pr.ok & np.all(np.isfinite(r), axis=1)
        r = np.where(ok[:, None], r, 50.0)
        if not jac:
            return r.ravel(), ok, None
        Jc = pr.J_cam                                    # (n, 2, 14)
        JX = pr.J_X                                      # (n, 2, 3)
        n = len(X)
        J = np.zeros((n, 2, self.sizes()))
        J[:, :, 0:3] = Jc[:, :, 0:3]
        J[:, :, 3] = Jc[:, :, 5]                         # C3
        J[:, :, 4] = Jc[:, :, 6]                         # log f
        # board rotation: X = T0 + Rb exp([wb]x) q -> dX/dwb = -Rb [q]x
        dXdwb = -np.einsum("ij,njk->nik", Rb, np.array([skew(v) for v in q]))
        J[:, :, 5:8] = np.einsum("nai,nik->nak", JX, dXdwb)
        J[:, :, 8:11] = JX
        if self.mode == "t held":
            u0 = self.t / np.linalg.norm(self.t)
            E = np.linalg.svd(u0[None, :])[2][1:].T
            J[:, :, 11:13] = self.k[:, None, None] * np.einsum("nai,ik->nak", JX, self.step_nom * E)
            off = 13
        else:
            J[:, :, 11:14] = self.k[:, None, None] * JX
            off = 14
        if self.Bc.shape[1]:
            # in-plane offset of dot d: X += Rb (cx, cy, 0)
            Dx = np.zeros((n, 3, 2 * self.n_dots))
            rows = np.arange(n)
            Dx[rows, :, 2 * self.dot] = Rb[:, 0][None, :]
            Dx[rows, :, 2 * self.dot + 1] = Rb[:, 1][None, :]
            J[:, :, off:] = np.einsum("nai,nij,jk->nak", JX, Dx, self.Bc)
        J = np.where(ok[:, None, None], J, 0.0)
        return r.ravel(), ok, J.reshape(2 * n, -1)

    def reanchor(self, x):
        R, C, f, Rb, T0, t, corr = self.unpack(x)
        self.R, self.C, self.f, self.Rb, self.T0, self.t, self.corr = R, C, f, Rb, T0, t, corr

    def run(self, outer=6):
        for it in range(outer):
            x0 = np.zeros(self.sizes())
            fun = lambda x: self.residual_jac(x, jac=False)[0]
            jacf = lambda x: self.residual_jac(x, jac=True)[2]
            sol = least_squares(fun, x0, jac=jacf, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400)
            self.reanchor(sol.x)
            if np.max(np.abs(sol.x[:5])) < 1e-12:
                break
        r, ok, J = self.residual_jac(np.zeros(self.sizes()), jac=True)
        m, p = int(2 * ok.sum()), J.shape[1]
        s2 = float(np.sum(r[np.repeat(ok, 2)] ** 2)) / max(m - p, 1)
        cov = s2 * np.linalg.pinv(J.T @ J)
        return {"rms_px": float(np.sqrt(np.mean(r[np.repeat(ok, 2)] ** 2) * 2)), "ok": int(ok.sum()),
                "n": int(len(ok)), "params": p, "f_px": self.f, "f_sigma_px": self.f * math.sqrt(cov[4, 4]),
                "D_mm": float(self.C[2]), "D_sigma_mm": math.sqrt(cov[3, 3]),
                "corr_f_D": float(cov[3, 4] / math.sqrt(cov[3, 3] * cov[4, 4])),
                "f_over_D": self.f / float(self.C[2]), "t_mm": self.t.tolist(), "step_mm": float(np.linalg.norm(self.t)),
                "step_sigma_mm": (float(math.sqrt(max(0.0, (self.t / np.linalg.norm(self.t)) @ cov[11:14, 11:14] @
                                                     (self.t / np.linalg.norm(self.t)))))
                                  if self.mode != "t held" else 0.0),
                "dot_corr_rms_um": 1e3 * float(np.sqrt(np.mean(self.corr ** 2))) if self.Bc.shape[1] else 0.0,
                "outer_iterations": it + 1}


def gauge(R, C, pts):
    """Turn about the axis (Y) and slide along it so that the camera's centre is at X = 0, Y = 0, Z > 0."""
    a = math.atan2(C[0], C[2])
    Q = rot_y(-a)
    shift = np.array([0.0, (Q @ C)[1], 0.0])
    return R @ Q.T, Q @ C - shift, (lambda P: (np.asarray(P) @ Q.T) - shift), Q


def main():
    cfg = case.load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["out_dir"]
    pl = cfg["planes"]
    log = case.Log(os.path.join(HERE, "c2b_f_board"))
    lib = case.library()
    with open(os.path.join(HERE, "c2a_board_check.json"), encoding="utf-8") as fh:
        c2a = json.load(fh)
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    tube = lib.cm.Tube(tuple(res["tube"]["radii_mm"]), tuple(res["tube"]["n"]))
    pitch = float(pl["pitch_mm"][0])
    step_nom = float(pl["step_mm_nominal"])
    log(f"C2b: f of each camera from the board alone (start: the cameras of {folder}, the board of C2a)")
    # the dots of each camera, on cam1's lattice
    dets = {}
    for kk, pos in enumerate(pl["positions"]):
        P = {}
        ijs = {}
        for cam in (1, 2):
            d = dict(pl["detect"])
            d.update(pl.get("detect_overrides", {}).get(f"cam{cam}", {}))
            r = planes.detect(planes.read(cfg, pos, cam), d)
            P[cam] = np.array([[q["u"], q["v"]] for q in r["dots"] if q["ok"]], float)
            ijs[cam] = planes.index_dots(P[cam], seed_xy=(640, 400))[0]
        dets[pos] = (P, ijs)
    # cam1's indices carried from position to position as in C2a, and cam2's mapped onto them by C2a's maps
    obs = {1: {"k": [], "ij": [], "uv": []}, 2: {"k": [], "ij": [], "uv": []}}
    prev = None
    for kk, pos in enumerate(pl["positions"]):
        P, ijs = dets[pos]
        ij1 = ijs[1].copy()
        if prev is not None:
            Pp, ijp = prev
            dmin = np.hypot(P[1][:, None, 0] - Pp[None, :, 0], P[1][:, None, 1] - Pp[None, :, 1])
            nn = np.argmin(dmin, axis=1)
            close = dmin[np.arange(len(P[1])), nn] < 25.0
            offs = ijp[nn[close]] - ij1[close]
            vals, cnt = np.unique(offs, axis=0, return_counts=True)
            ij1 = ij1 + vals[np.argmax(cnt)]
        prev = (P[1], ij1)
        mp = c2a["positions"][pos]["map"]
        M, t = np.array(mp["M"]), np.array(mp["t"])
        # C2a's map is on cam1's carried indices: ij2 = M ij1 + t
        ij2_in_1 = (np.linalg.inv(M) @ (ijs[2] - t).T).T.round().astype(int)
        for cam, ij in ((1, ij1), (2, ij2_in_1)):
            ij_up = np.column_stack([ij[:, 0], -ij[:, 1]])
            obs[cam]["k"].append(np.full(len(ij), kk, float))
            obs[cam]["ij"].append(ij_up.astype(float))
            obs[cam]["uv"].append(P[cam])
    # the board's start from C2a (position 0's lattice, in the wand calibration's frame)
    la = c2a["positions"][pl["positions"][0]]["lattice"]
    O, A, B = np.array(la["O"]), np.array(la["A"]), np.array(la["B"])
    e1 = A / np.linalg.norm(A)
    e2 = -B - (-B @ e1) * e1
    e2 /= np.linalg.norm(e2)
    Rb0 = np.column_stack([e1, e2, np.cross(e1, e2)])
    tr = c2a["steps"]
    t0 = np.mean([np.array(v["translation_mm"]) for v in tr.values()], axis=0)
    out = {"cameras": {}}
    c1t = None
    if os.path.exists(os.path.join(HERE, "c1t_variants.json")):
        with open(os.path.join(HERE, "c1t_variants.json"), encoding="utf-8") as fh:
            c1t = json.load(fh)
    for j, cres in enumerate(res["cameras"]):
        cam_id = j + 1
        o = {k: np.concatenate(v) for k, v in obs[cam_id].items()}
        keys = [tuple(v) for v in o["ij"].astype(int)]
        uniq = sorted(set(keys))
        dot_of = {k: n for n, k in enumerate(uniq)}
        o["dot"] = np.array([dot_of[k] for k in keys])
        o["dot_ij"] = np.array(uniq, float)
        R, C = np.array(cres["R"], float), np.array(cres["centre_mm"], float)
        Rg, Cg, mapP, Q = gauge(R, C, None)
        cam0 = lib.cm.Camera(Rg, Cg, cres["f_px"], cres["cx"], cres["cy"], cres["width"], cres["height"], np.zeros(5))
        board0 = {"Rb": Q @ Rb0, "T0": mapP(O[None, :])[0], "t": Q @ t0}
        D_wand = float(np.hypot(C[0], C[2]))
        log(f"\n  {cres['name']}: {len(o['uv'])} detections of {len(uniq)} dots over {len(pl['positions'])} positions; "
            f"wand calibration: f {cres['f_px']:.0f} px, D {D_wand:.1f} mm, f / D {cres['f_px'] / D_wand:.3f} px/mm")
        out["cameras"][cres["name"]] = {"wand": {"f_px": cres["f_px"], "D_mm": D_wand,
                                                 "f_sigma_px": (cres.get("sigma") or {}).get("f_px")}, "variants": {}}
        for mode in ("t free", "t held", "t free + dots"):
            fit = Fit(lib, tube, cam0, o, pitch, board0, mode, step_nom)
            r = fit.run()
            out["cameras"][cres["name"]]["variants"][mode] = r
            log(f"    {mode:<14s}: f {r['f_px']:9.0f} +- {r['f_sigma_px']:6.0f} px, D {r['D_mm']:8.1f} +- "
                f"{r['D_sigma_mm']:5.1f} mm (corr {r['corr_f_D']:+.3f}), f / D {r['f_over_D']:.3f} px/mm; RMS "
                f"{r['rms_px']:.3f} px ({r['ok']}/{r['n']} projected, {r['params']} parameters); step {r['step_mm']:.4f}"
                + (f" +- {r['step_sigma_mm']:.4f}" if r["step_sigma_mm"] else " (held)") + " mm"
                + (f"; dot corrections RMS {r['dot_corr_rms_um']:.1f} um" if r["dot_corr_rms_um"] else "")
                + f"; f vs the wand {100 * (r['f_px'] / cres['f_px'] - 1):+.2f} %")
        if c1t:
            pts = [(v["summary"]["cameras"][j]["f_px"], v["summary"]["cameras"][j]["distance_mm"])
                   for v in c1t["variants"].values() if "summary" in v]
            pts.append((cres["f_px"], D_wand))
            F, Dd = np.array(pts).T
            k_line = float(np.sum(F * Dd) / np.sum(Dd * Dd))
            out["cameras"][cres["name"]]["wand_line_px_per_mm"] = k_line
            log(f"    the wand fits (C1 and C1-T's parts) lie along f = {k_line:.3f} D (spread of f / D "
                f"{np.std(F / Dd):.3f}); the board's f / D: " + ", ".join(
                    f"{m} {out['cameras'][cres['name']]['variants'][m]['f_over_D']:.3f}"
                    for m in out["cameras"][cres["name"]]["variants"]))
    log.save(out)


if __name__ == "__main__":
    main()
