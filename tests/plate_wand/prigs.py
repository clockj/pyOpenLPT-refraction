"""Shared helpers for the flat-window (PINPLATE) wand-calibration tests: paths, synthetic rigs A-E and random rigs,
wand data from the C++ camera, the alignment to the truth and the metrics (implementation plan, section 5).

Synthetic detections come from pyOpenLPT's C++ PINPLATE camera (the source of truth), never from the fast model being
tested.  Each window is 12.7 mm of acrylic (1.49) between air (1.0) and water (1.333) unless stated.  A camera is
placed by its chief ray: the ray leaves the camera at a given incidence to its window's normal, refracts through the
window and passes through an aim point in the water; the camera looks along it, so the aim point is imaged at the
principal point.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CURVED = REPO / "modules" / "camera_calibration" / "curved_calibration"
for _p in (str(REPO / "build-py" / os.environ.get("CONFIG", "Release")), str(CURVED), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cyl_model as cm  # noqa: E402
import plate_model as pm  # noqa: E402

N_AIR, N_ACRYLIC, N_WATER = 1.0, 1.49, 1.333
T_WALL = 12.7
SENSOR = (1280, 800)


class Tee:
    """Print to the console and to a log file beside the test (opened at the first line written)."""

    def __init__(self, path):
        self.path = path
        self.f = None

    def __call__(self, *args):
        s = " ".join(str(a) for a in args)
        print(s, flush=True)
        if self.f is None:
            self.f = open(self.path, "w", encoding="utf-8")
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()


def look_dir(z, up=(0.0, 1.0, 0.0), roll_deg: float = 0.0) -> np.ndarray:
    """World -> camera rotation (OpenCV axes) of a camera looking along z, image up along `up`, then rolled."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    up = np.asarray(up, float)
    if abs(float(up @ z)) > 0.99:
        up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ R
    return R


def wall(normal, d: float, t=(T_WALL,), idx=(N_AIR, N_ACRYLIC, N_WATER), name="") -> pm.Window:
    return pm.Window(np.asarray(normal, float), d, tuple(t), tuple(idx), name)


def chief_ray_camera(win: pm.Window, aim, theta_deg: float, phi_deg: float, dist: float):
    """Centre C and air direction u of a camera whose chief ray meets the window at incidence theta (azimuth phi about
    the normal, from the window's first tangent vector), passes through `aim` (in the water) after refraction, and
    leaves the camera `dist` mm before the window."""
    n = win.n
    t1, t2 = pm.tangent_basis(n)
    t1, t2 = t1[0], t2[0]
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    e = math.cos(ph) * t1 + math.sin(ph) * t2
    u = math.cos(th) * n + math.sin(th) * e
    tau = math.tan(th)
    aim = np.asarray(aim, float)
    hW = float(aim @ n) - win.far_d
    if hW <= 0:
        raise ValueError("the aim point is not in the object medium")
    n0 = win.idx[0]
    lat = sum(tk * n0 * tau / math.sqrt(nk * nk + (nk * nk - n0 * n0) * tau * tau)
              for tk, nk in zip(win.t + (hW,), win.idx[1:]))
    E = aim - lat * e - (float(aim @ n) - win.d) * n          # the entry point on the camera-side surface
    return E - dist * u, u


def fit_f(cam: pm.Camera, win: pm.Window, corners: np.ndarray, fill: float) -> float:
    """The f at which the projected box (its corners) spans `fill` of the sensor in its limiting direction."""
    c = cam.copy()
    c.f = 1.0
    pr = pm.project(pm.rows([c], np.zeros(len(corners), int)), pm.wrows([win], np.zeros(len(corners), int)), corners)
    if not np.all(pr.ok):
        raise ValueError("a corner of the cloud is not projected")
    xy = pr.xy
    span = xy.max(axis=0) - xy.min(axis=0)
    return fill * min(cam.width / span[0], cam.height / span[1])


def box_corners(centre, half) -> np.ndarray:
    c = np.asarray(centre, float)
    h = np.asarray(half, float)
    return np.array([c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


@dataclass
class TrueRig:
    name: str
    cams: List[pm.Camera]
    wins: List[pm.Window]
    cam_win: List[int]
    centre: np.ndarray            # the cloud box
    half: np.ndarray
    L: float
    info: dict = field(default_factory=dict)

    @property
    def n_cams(self) -> int:
        return len(self.cams)

    @property
    def sizes(self):
        return [(c.width, c.height) for c in self.cams]

    def win_of(self, c: int) -> pm.Window:
        return self.wins[self.cam_win[c]]

    def cpp(self, tol: float = 1e-12):
        return [pm.to_cpp(c, self.win_of(k), tol=tol) for k, c in enumerate(self.cams)]


def place(win, aim, theta, phi, dist, corners, fill, sensor=SENSOR, roll=0.0, up=(0.0, 1.0, 0.0), f=None,
          dist_coef=None) -> pm.Camera:
    C, u = chief_ray_camera(win, aim, theta, phi, dist)
    R = look_dir(u, up=up, roll_deg=roll)
    w, h = sensor
    cam = pm.Camera(R, C, 1.0, (w - 1) / 2.0, (h - 1) / 2.0, w, h, np.zeros(5) if dist_coef is None else dist_coef)
    cam.f = fit_f(cam, win, corners, fill) if f is None else float(f)
    return cam


def _hv(win, h_deg, v_deg):
    """Incidence and azimuth for horizontal / vertical angles h, v (tan-composed) about a window whose tangent basis
    is (t1, t2): the chief ray's direction is n + tan(h) t_h + tan(v) t_v with t_h the horizontal tangent."""
    n = win.n
    t1, t2 = pm.tangent_basis(n)
    t1, t2 = t1[0], t2[0]
    th_h = np.cross(np.array([0.0, 1.0, 0.0]), n)
    if np.linalg.norm(th_h) < 1e-9:
        th_h = np.array([1.0, 0.0, 0.0])
    th_h /= np.linalg.norm(th_h)
    tv = np.cross(n, th_h)
    lat = math.tan(math.radians(h_deg)) * th_h + math.tan(math.radians(v_deg)) * tv
    theta = math.degrees(math.atan(np.linalg.norm(lat)))
    phi = math.degrees(math.atan2(float(lat @ t2), float(lat @ t1))) if np.linalg.norm(lat) > 0 else 0.0
    return theta, phi


def rig_A(fill: float = 0.8) -> TrueRig:
    """One wall (Z = 0, normal +Z); four cameras at +-18 deg horizontally and +-12 deg vertically, 600 mm from the
    window along their chief rays, aimed at the centre of a 100 mm cloud 150 mm behind the window."""
    w = wall([0, 0, 1], 0.0, name="wall Z")
    centre, half = np.array([0.0, 0.0, T_WALL + 150.0]), np.full(3, 50.0)
    corners = box_corners(centre, half)
    cams = []
    for h, v in ((-18.0, -12.0), (18.0, -12.0), (-18.0, 12.0), (18.0, 12.0)):
        th, ph = _hv(w, h, v)
        cams.append(place(w, centre, th, ph, 600.0, corners, fill))
    return TrueRig("A", cams, [w], [0, 0, 0, 0], centre, half, 25.0)


def rig_B(fill: float = 0.8) -> TrueRig:
    """Two perpendicular walls of a tank (Z = 0 with normal +Z, X = 0 with normal +X); two cameras behind each, at
    +-15 deg (horizontally behind the first wall, vertically behind the second), 600 mm from their windows; a 100 mm
    cloud 100 mm from both walls."""
    w1 = wall([0, 0, 1], 0.0, name="wall Z")
    w2 = wall([1, 0, 0], 0.0, name="wall X")
    centre, half = np.array([T_WALL + 150.0, 0.0, T_WALL + 150.0]), np.full(3, 50.0)
    corners = box_corners(centre, half)
    cams = []
    for h in (-15.0, 15.0):
        th, ph = _hv(w1, h, 0.0)
        cams.append(place(w1, centre, th, ph, 600.0, corners, fill))
    for v in (-15.0, 15.0):
        th, ph = _hv(w2, 0.0, v)
        cams.append(place(w2, centre, th, ph, 600.0, corners, fill))
    return TrueRig("B", cams, [w1, w2], [0, 0, 1, 1], centre, half, 25.0)


def rig_C(fill: float = 0.8) -> TrueRig:
    """Square-on: one wall, three cameras within 5 deg of its normal (0; 5 deg horizontally; 5 deg at 120 deg)."""
    w = wall([0, 0, 1], 0.0, name="wall Z")
    centre, half = np.array([0.0, 0.0, T_WALL + 150.0]), np.full(3, 50.0)
    corners = box_corners(centre, half)
    cams = []
    for h, v in ((0.0, 0.0), (5.0, 0.0), (-2.5, 4.33)):
        th, ph = _hv(w, h, v)
        cams.append(place(w, centre, th, ph, 600.0, corners, fill))
    return TrueRig("C", cams, [w], [0, 0, 0], centre, half, 25.0)


def rig_D(fill: float = 0.8) -> TrueRig:
    """Steep: one wall, three cameras at 35, 40 and 45 deg incidence (two horizontally, one vertically)."""
    w = wall([0, 0, 1], 0.0, name="wall Z")
    centre, half = np.array([0.0, 0.0, T_WALL + 150.0]), np.full(3, 50.0)
    corners = box_corners(centre, half)
    cams = []
    for h, v in ((35.0, 0.0), (-40.0, 0.0), (0.0, 45.0)):
        th, ph = _hv(w, h, v)
        cams.append(place(w, centre, th, ph, 600.0, corners, fill))
    return TrueRig("D", cams, [w], [0, 0, 0], centre, half, 25.0)


def rig_E() -> TrueRig:
    """Macro: one wall, three cameras at +-15 deg, f = 40,000 px, 1,100 mm from the window; a 16 mm cloud 100 mm
    behind the window; a 10 mm wand."""
    w = wall([0, 0, 1], 0.0, name="wall Z")
    centre, half = np.array([0.0, 0.0, T_WALL + 100.0]), np.full(3, 8.0)
    corners = box_corners(centre, half)
    cams = []
    for h, v in ((-15.0, 0.0), (15.0, 0.0), (0.0, 15.0)):
        th, ph = _hv(w, h, v)
        cams.append(place(w, centre, th, ph, 1100.0, corners, 0.8, f=40000.0))
    return TrueRig("E", cams, [w], [0, 0, 0], centre, half, 10.0)


RIGS = {"A": rig_A, "B": rig_B, "C": rig_C, "D": rig_D, "E": rig_E}


def random_rig(rng, max_tries: int = 500) -> TrueRig:
    """A random rig (implementation plan, section 5): 2-6 cameras; 1 (50 %), 2 (35 %) or 3 (15 %) walls of a box tank
    (Z = 0 with normal +Z, X = 0 with normal +X, the far wall with normal -Z); incidence 0-45 deg; 150-1,500 mm from
    the window along the chief ray; a 30-300 mm cloud 30-300 mm behind each wall used; the wand 0.2-0.35 of the cloud;
    f so that the cloud fills 50-90 % of the sensor, kept within 1,500-40,000 px; roll +-10 deg."""
    for _ in range(max_tries):
        nw = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))
        size = float(math.exp(rng.uniform(math.log(30.0), math.log(300.0))))
        half = np.full(3, size / 2.0) * rng.uniform(0.7, 1.0, 3)
        gap = [float(math.exp(rng.uniform(math.log(30.0), math.log(300.0)))) for _ in range(3)]
        t_acr = float(rng.choice([6.0, 10.0, 12.7, 19.05, 25.4]))
        cx = (t_acr + gap[1] + half[0]) if nw >= 2 else rng.uniform(-100.0, 100.0)
        centre = np.array([cx, rng.uniform(-50.0, 50.0), t_acr + gap[0] + half[2]])
        wins = [wall([0, 0, 1], 0.0, t=(t_acr,), name="wall Z")]
        if nw >= 2:
            wins.append(wall([1, 0, 0], 0.0, t=(t_acr,), name="wall X"))
        if nw >= 3:
            z_out = centre[2] + half[2] + gap[2] + t_acr
            wins.append(wall([0, 0, -1], -z_out, t=(t_acr,), name="far wall"))
        n_cams = int(rng.integers(2, 7))
        cam_win = [k % nw for k in range(n_cams)]
        used = sorted(set(cam_win))                  # two cameras and three walls: the third wall is not used
        wins = [wins[k] for k in used]
        cam_win = [used.index(k) for k in cam_win]
        nw = len(wins)
        sensor_choices = [(1280, 800), (2048, 2048), (2560, 1600)]
        corners = box_corners(centre, half)
        cams = []
        okr = True
        for c in range(n_cams):
            w = wins[cam_win[c]]
            sensor = sensor_choices[int(rng.integers(0, 3))]
            theta = float(rng.uniform(0.0, 45.0))
            phi = float(rng.uniform(0.0, 360.0))
            dist = float(math.exp(rng.uniform(math.log(150.0), math.log(1500.0))))
            aim = centre + rng.uniform(-0.15, 0.15, 3) * half
            try:
                cam = place(w, aim, theta, phi, dist, corners, float(rng.uniform(0.5, 0.9)), sensor=sensor,
                            roll=float(rng.uniform(-10.0, 10.0)))
            except ValueError:
                okr = False
                break
            if not (1500.0 <= cam.f <= 40000.0) or w.distance(cam.C) <= 0:
                okr = False
                break
            cams.append(cam)
        if not okr:
            continue
        L = float(size * rng.uniform(0.2, 0.35))
        rig = TrueRig("random", cams, wins, cam_win, centre, half, L,
                      info={"n_windows": nw, "thickness": t_acr, "cloud_mm": size})
        return rig
    raise RuntimeError("no random rig")


# ------------------------------------------------------------------ wand data
@dataclass
class Wand:
    uv: np.ndarray            # (F, Nc, 2, 2): frame, camera, end (A, B), (u, v); NaN where not seen
    XA: np.ndarray            # (F, 3) truth
    XB: np.ndarray            # (F, 3) truth
    L: float
    clean: np.ndarray = None  # (F, Nc, 2, 2) the noise-free detections


def make_wand(rig: TrueRig, n_frames: int, rng, noise_px: float = 0.1, min_cams: int = 2, all_cams: bool = False,
              cpp_tol: float = 1e-12, max_tries: int = 200, noise_cam=None) -> Wand:
    """Wand frames with midpoints uniform in the rig's box and directions uniform on the sphere, both ends inside
    the box.  Detections are the C++ images, on the sensor, plus Gaussian noise (noise_px, or per camera noise_cam).
    A frame is kept when both ends are seen by at least `min_cams` cameras (by every camera with all_cams); a camera
    that sees only one end of a frame loses that frame."""
    cpp = rig.cpp(tol=cpp_tol)
    L = rig.L
    uvs, XA, XB = [], [], []
    tries = 0
    while len(uvs) < n_frames and tries < max_tries:
        tries += 1
        m = n_frames - len(uvs)
        k = max(2 * m, 64)
        mid = rig.centre + rng.uniform(-1, 1, size=(k, 3)) * rig.half
        w = rng.normal(size=(k, 3))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        A, B = mid - 0.5 * L * w, mid + 0.5 * L * w
        inside = np.all(np.abs(A - rig.centre) <= rig.half, axis=1) & np.all(np.abs(B - rig.centre) <= rig.half,
                                                                               axis=1)
        A, B = A[inside], B[inside]
        uv = np.full((len(A), rig.n_cams, 2, 2), np.nan)
        for c, (cam, cc) in enumerate(zip(rig.cams, cpp)):
            for e, P in enumerate((A, B)):
                q = cm.cpp_project(cc, P)
                on = (q[:, 0] >= 0) & (q[:, 0] < cam.width) & (q[:, 1] >= 0) & (q[:, 1] < cam.height)
                q[~on] = np.nan
                uv[:, c, e] = q
        seen = np.all(np.isfinite(uv[..., 0]), axis=2)
        n_seen = seen.sum(axis=1)
        good = n_seen == rig.n_cams if all_cams else n_seen >= min_cams
        for i in np.flatnonzero(good):
            for c in range(rig.n_cams):
                if not seen[i, c]:
                    uv[i, c] = np.nan
        uvs += list(uv[good][:m])
        XA += list(A[good][:m])
        XB += list(B[good][:m])
    clean = np.array(uvs)
    uv = clean.copy()
    if noise_cam is not None:
        uv = uv + rng.normal(size=uv.shape) * np.asarray(noise_cam, float)[None, :, None, None]
    elif noise_px > 0:
        uv = uv + rng.normal(scale=noise_px, size=uv.shape)
    return Wand(uv=uv, XA=np.array(XA), XB=np.array(XB), L=float(L), clean=clean)


def cloud_points(rig: TrueRig, rng, n: int = 500, grow: float = 1.0) -> np.ndarray:
    """Points uniform in the rig's box (grown by `grow`), e.g. unseen points for the 3-D error."""
    return rig.centre + rng.uniform(-1, 1, size=(n, 3)) * rig.half * grow


# ------------------------------------------------------------------ alignment and metrics
def rigid_from_cam(cam_fit: pm.Camera, cam_true: pm.Camera):
    """(Q, p) with X_true = Q X_fit + p mapping the fit's camera 1 onto the truth's: R_true = R_fit Q^T,
    C_true = Q C_fit + p."""
    Q = cam_true.R.T @ cam_fit.R
    p = cam_true.C - Q @ cam_fit.C
    return Q, p


def transform(cams, wins, Q, p, points=None):
    """Cameras, windows (and points) mapped by X -> Q X + p."""
    oc = []
    for c in cams:
        k = c.copy()
        k.R = c.R @ Q.T
        k.C = Q @ c.C + p
        oc.append(k)
    ow = []
    for w in wins:
        n = Q @ w.n
        ow.append(pm.Window(n, w.d + float(n @ p), w.t, w.idx, w.name))
    if points is None:
        return oc, ow
    return oc, ow, np.asarray(points, float) @ Q.T + p


def compare_cams(est, truth) -> List[dict]:
    out = []
    for e, t in zip(est, truth):
        out.append({"rot_deg": cm.rotation_angle_deg(e.R, t.R), "centre_mm": float(np.linalg.norm(e.C - t.C)),
                    "f_rel": float(e.f / t.f - 1.0)})
    return out


def compare_wins(est, truth) -> List[dict]:
    out = []
    for e, t in zip(est, truth):
        ang = math.degrees(math.atan2(np.linalg.norm(np.cross(e.n, t.n)), float(e.n @ t.n)))
        out.append({"normal_deg": ang, "d_mm": float(e.d - t.d)})
    return out


def best_rigid(P: np.ndarray, Q: np.ndarray):
    """Rigid (R, t) minimising |R P + t - Q| (Kabsch); returns (R, t, rms)."""
    mp, mq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - mp).T @ (Q - mq)
    U, _s, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = mq - R @ mp
    res = P @ R.T + t - Q
    return R, t, float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))


def sphere_log(n_ref, n):
    """(a, b) with n = exp_{n_ref}(a t1 + b t2), the tangent basis of plate_model at n_ref."""
    t1, t2 = pm.tangent_basis(n_ref)
    c = float(np.clip(n_ref @ n, -1.0, 1.0))
    v = n - c * n_ref
    sv = float(np.linalg.norm(v))
    if sv == 0.0:
        return 0.0, 0.0
    th = math.atan2(sv, c)
    v = v / sv * th
    return float(v @ t1[0]), float(v @ t2[0])


def param_z(cams, wins, L, ref) -> dict:
    """Each free parameter's difference from the reference solution `ref` (a plate_wand.Solution with its
    covariance), in units of the reference's sigma; the cameras and windows must already be in ref's frame (e.g.
    aligned by camera 1).  Returns {"max_abs_z": ..., "worst": name}."""
    lay, cov = ref.layout, ref.cov
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    worst, name_w = 0.0, ""
    for c, (cs, cr) in enumerate(zip(cams, ref.cams)):
        dw = pm.log_so3(cr.R.T @ cs.R)
        vals = {"w1": dw[0], "w2": dw[1], "w3": dw[2], "C1": cs.C[0] - cr.C[0], "C2": cs.C[1] - cr.C[1],
                "C3": cs.C[2] - cr.C[2], "logf": math.log(cs.f / cr.f), "k1": cs.dist[0] - cr.dist[0]}
        for j, nm in enumerate(lay.cols):
            g = lay.pos[c, j]
            if g < 0:
                continue
            z = abs(vals[nm]) / sd[g]
            if z > worst:
                worst, name_w = z, f"cam{c + 1}.{nm}"
    for w, (ws, wr) in enumerate(zip(wins, ref.rig.windows)):
        if w >= lay.n_wins or lay.wpos[w, 0] < 0:
            continue
        a, b = sphere_log(wr.n, ws.n)
        A = lay.anchors[w]
        dl = (ws.d - float(ws.n @ A)) - (wr.d - float(wr.n @ A))
        for j, v in enumerate((a, b, dl)):
            z = abs(v) / sd[lay.wpos[w, j]]
            if z > worst:
                worst, name_w = z, f"win{w + 1}.{'ab'[j] if j < 2 else 'delta'}"
    if lay.fit_L:
        z = abs(L - ref.L) / sd[lay.L_index]
        if z > worst:
            worst, name_w = z, "L"
    return {"max_abs_z": float(worst), "worst": name_w}


def same_minimum(sol, ref) -> dict:
    """Compare a solution (any frame) with the reference: align by camera 1, then the RMS difference (relative) and
    the largest parameter difference in reference sigmas."""
    Q, p = rigid_from_cam(sol.cams[0], ref.cams[0])
    cams_a, wins_a = transform(sol.cams, sol.rig.windows, Q, p)
    z = param_z(cams_a, wins_a, sol.L, ref)
    d_rms = abs(sol.rms - ref.rms) / ref.rms
    n_act = int(sol.obs.active.sum())
    n_ref = int(ref.obs.active.sum())
    # the RMS is the unweighted one, while the fits minimise the weighted cost with weights estimated to 1e-6: two
    # runs at the same minimum differ in it by up to ~1e-8 (W5); the parameters (0.01 sigma) decide
    same = bool(d_rms <= 1e-6 and z["max_abs_z"] <= 0.01 and n_act == n_ref)
    return {"rms_rel_diff": float(d_rms), **z, "active": n_act, "active_ref": n_ref, "same": same}
