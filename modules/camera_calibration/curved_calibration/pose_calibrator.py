"""Refraction-aware, pose-only bundle adjustment through a fixed port.

Solves camera extrinsics (and optionally focal length) against observations seen
through a refractive port whose geometry is *given* -- a flat plate, a coaxial
cylinder or a concentric sphere.  Nothing about the port is optimised here.

**Gauge depends on the observation kind, and it is easy to get wrong.**  With
3-D/2-D pairs the known world points pin the frame outright, so there is nothing
to fix.  With a wand the 3-D positions are unknown and the port is the only
anchor, so whatever transform preserves the port is free: rotating every camera
*and* every wand point about a sphere's centre changes no residual at all.  That
freedom is exactly the port's stabiliser -- 3 dof for a sphere, 2 for a cylinder
(spin about the axis and slide along it), 3 for a plate (slide in the plane and
spin about the normal).  Left alone the solve drifts along that orbit and reaches
a *perfect* fit at visibly wrong poses.  `PortGauge` removes it by projecting each
round's answer back onto the reference, which changes no residual.

Three deliberate choices, in decreasing order of how much they matter:

**Residuals are built from back-projection, not forward projection.**  Through a
curved port the forward map is multi-valued and can have no solution at all,
which makes a reprojection residual discontinuous and undefined outside the
visible set.  Back-projection is closed-form, single-valued and always defined,
so the residual is the offset of the known (or triangulated) point from the
observed ray.  It is scaled by `f / (optical path from the camera)`, which makes
it the reprojection error in pixels to first order rather than a raw millimetre
distance -- without that scaling the fit is biased towards whichever points sit
closest to the cameras.  The path must be measured from the camera, not from the
ray's origin on the port surface; see `_ray_residual`.

**The offset is expressed in a 2-D basis perpendicular to the ray**, not as a
scalar distance.  A scalar distance is non-negative and has a kink at zero, which
is exactly where the solution lives; Gauss-Newton behaves badly there.  Two
signed components behave like a reprojection residual and have no kink.

**Poses are updated on the manifold.**  The increment is
`R <- R exp(dw^)`, `C <- C + dt`, re-anchored after every solve, rather than
added to a Rodrigues vector.  Additive axis-angle updates degrade near
|rvec| = pi, which is where a camera looking back down the world axes sits -- a
routine configuration for cameras ringing a port.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
from scipy.optimize import least_squares

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import pyopenlpt as lpt
except ImportError:  # pragma: no cover
    lpt = None


# ------------------------------------------------------------------ SO(3)
def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation matrix from an axis-angle vector."""
    rvec = np.asarray(rvec, float).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-14:
        return np.eye(3)
    k = rvec / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def rot_to_rodrigues(rmtx: np.ndarray) -> np.ndarray:
    """Axis-angle vector from a rotation matrix."""
    if cv2 is not None:
        return cv2.Rodrigues(np.asarray(rmtx, float))[0].ravel()
    cos = np.clip((np.trace(rmtx) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos))
    if theta < 1e-12:
        return np.zeros(3)
    v = np.array([rmtx[2, 1] - rmtx[1, 2], rmtx[0, 2] - rmtx[2, 0],
                  rmtx[1, 0] - rmtx[0, 1]])
    return v * (theta / (2.0 * np.sin(theta)))


@dataclass
class Pose:
    """Camera pose plus the intrinsics the fit may touch."""

    rvec: np.ndarray
    tvec: np.ndarray
    f: float
    cx: float
    cy: float

    def rmtx(self) -> np.ndarray:
        return rodrigues(self.rvec)

    def center(self) -> np.ndarray:
        return -self.rmtx().T @ np.asarray(self.tvec, float)

    def retract(self, dw: np.ndarray, dt: np.ndarray, dlogf: float = 0.0) -> "Pose":
        """R <- R exp(dw^), C <- C + dt, f <- f exp(dlogf)."""
        rmtx = self.rmtx() @ rodrigues(dw)
        center = self.center() + np.asarray(dt, float)
        return Pose(rot_to_rodrigues(rmtx), -rmtx @ center,
                    float(self.f) * float(np.exp(dlogf)), self.cx, self.cy)


# ----------------------------------------------------------------- config
@dataclass
class PoseFitConfig:
    optimize_focal: bool = False
    # Re-anchor rounds: the retraction is only a good chart near delta = 0.
    max_outer: int = 6
    max_nfev: int = 300
    ftol: float = 1e-12
    xtol: float = 1e-12
    gtol: float = 1e-12
    # "soft_l1" or "huber" for real data with outliers; f_scale is in pixels.
    loss: str = "linear"
    f_scale: float = 2.0
    # Stop once a whole round moves less than this.
    outer_tol: float = 1e-9
    wand_sigma_mm: float = 1.0
    # Observations are NOT equally informative. Back-projection is near-singular
    # for grazing rays, so at the wall of a curved port a 0.2 px detection error
    # becomes a 4-7 px ray residual -- 20-35x the amplification at the centre.
    # Left unweighted those observations dominate the least-squares cost and drag
    # the fit. The weight below is 1/sensitivity, measured by perturbing the
    # pixel, so every observation enters at its true worth instead.
    weight_by_sensitivity: bool = True
    weight_probe_px: float = 0.5  # pixel perturbation used to measure the Jacobian
    # Singular values of the pixel->residual Jacobian are clamped to this range
    # before inversion, so a near-singular direction cannot be given unbounded
    # weight and a numerically flat one cannot be amplified without limit. The
    # ideal pinhole value is 1.
    weight_svd_min: float = 0.2
    weight_svd_max: float = 50.0
    # A single-camera resection through a curved port is not convex: it can
    # converge from a 3 deg start to a stationary point tens of pixels out while
    # its sibling cameras sit at the noise floor. That is detectable -- the bad
    # camera's residual is an outlier against the others -- so retry it from
    # perturbed starts and keep the best. Set retry_starts = 0 to disable.
    retry_starts: int = 6
    retry_factor: float = 4.0      # retry a camera this far above the median rms
    retry_floor_px: float = 0.5    # ...but never chase noise below this
    retry_rot_deg: float = 6.0     # spread of the retry starts
    retry_trans_mm: float = 30.0
    retry_seed: int = 20260918
    # A wand fit can stall the same way, but it is one joint fit: a stalled
    # camera drags the triangulated points, so every camera's residual rises
    # with it (all within 1.7x of each other in the stall of the synthetic tests)
    # and no camera stands out. So a wand fit whose rms stays above this is
    # rerun from retry_starts perturbed starts instead. Its rms mixes ray
    # residuals in px with wand-length residuals in units of wand_sigma_mm; a
    # converged fit sits near 0.6x the pixel noise (0.11-0.15 at 0.2 px), and the
    # stalls seen had 7.6 and 8.2.
    wand_retry_rms: float = 1.0
    verbosity: int = 0


# ------------------------------------------------------------------ gauge
@dataclass
class PortGauge:
    """The transforms that leave a port unchanged, and so leave every residual
    unchanged when applied to the cameras and the (unknown) points together.

    `kind` is "none" (3-D/2-D data, or no symmetry to remove), "sphere"
    (rotations about `center`), "cylinder" (spin about `axis` and slide along it)
    or "plate" (slide in the plane and spin about its normal `axis`).
    """

    kind: str = "none"
    center: np.ndarray = None
    axis: np.ndarray = None

    def project(self, poses: Dict[int, Pose],
                reference: Dict[int, Pose]) -> Dict[int, Pose]:
        """Move `poses` along the gauge orbit to sit as close to `reference` as
        possible. Residuals are invariant under this, so it only removes drift."""
        if self.kind == "none" or not poses:
            return poses
        cids = sorted(set(poses) & set(reference))
        if not cids:
            return poses
        cur = np.array([poses[c].center() for c in cids], float)
        ref = np.array([reference[c].center() for c in cids], float)

        if self.kind == "sphere":
            o = np.asarray(self.center, float)
            rot = _procrustes_rotation(cur - o, ref - o)
            return _apply_rigid(poses, rot, o - rot @ o)

        axis = np.asarray(self.axis, float)
        axis = axis / np.linalg.norm(axis)
        o = np.asarray(self.center, float)

        if self.kind == "cylinder":
            # Only the spin about the axis and the slide along it are free.
            ang = _best_axis_angle(cur - o, ref - o, axis)
            rot = rodrigues(axis * ang)
            shift = float(np.mean((ref - (rot @ (cur - o).T).T - o) @ axis))
            return _apply_rigid(poses, rot, o - rot @ o + shift * axis)

        if self.kind == "plate":
            # Spin about the normal, and slide within the plane.
            ang = _best_axis_angle(cur - o, ref - o, axis)
            rot = rodrigues(axis * ang)
            resid = ref - (rot @ (cur - o).T).T - o
            shift = resid.mean(axis=0)
            shift = shift - float(shift @ axis) * axis  # in-plane only
            return _apply_rigid(poses, rot, o - rot @ o + shift)

        raise ValueError(f"unknown gauge kind {self.kind!r}")


def _procrustes_rotation(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation taking `src` onto `dst` in the least-squares sense."""
    H = np.asarray(src, float).T @ np.asarray(dst, float)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _best_axis_angle(src: np.ndarray, dst: np.ndarray, axis: np.ndarray) -> float:
    """Angle about `axis` that best rotates `src` onto `dst`."""
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(float(e1 @ axis)) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - float(e1 @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    su, sv = src @ e1, src @ e2
    du, dv = dst @ e1, dst @ e2
    return float(np.arctan2(np.sum(su * dv - sv * du), np.sum(su * du + sv * dv)))


def _apply_rigid(poses: Dict[int, Pose], rot: np.ndarray,
                 trans: np.ndarray) -> Dict[int, Pose]:
    """Move every camera by X -> rot X + trans."""
    out = {}
    for cid, p in poses.items():
        rmtx_new = p.rmtx() @ rot.T
        center_new = rot @ p.center() + trans
        out[cid] = Pose(rot_to_rodrigues(rmtx_new), -rmtx_new @ center_new,
                        p.f, p.cx, p.cy)
    return out


# ---------------------------------------------------------------- helpers
def _perp_basis(d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `d`."""
    d = np.asarray(d, float)
    trial = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, d) * d
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def rays_from_pixels(cam, uvs: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project pixels. Returns (origins (N,3), directions (N,3), ok (N,))."""
    pts = [lpt.Pt2D(float(uv[0]), float(uv[1])) for uv in uvs]
    results = cam.lineOfSightBatchStatus(pts)
    origins = np.zeros((len(pts), 3))
    dirs = np.zeros((len(pts), 3))
    ok = np.zeros(len(pts), dtype=bool)
    for i, rr in enumerate(results):
        good, line = rr[0], rr[1]
        if not good:
            continue
        o = np.array([line.pt[0], line.pt[1], line.pt[2]], float)
        d = np.array([line.unit_vector[0], line.unit_vector[1], line.unit_vector[2]], float)
        nd = np.linalg.norm(d)
        if not np.isfinite(o).all() or nd < 1e-12:
            continue
        origins[i] = o
        dirs[i] = d / nd
        ok[i] = True
    return origins, dirs, ok


def triangulate(origins: np.ndarray, dirs: np.ndarray) -> Optional[np.ndarray]:
    """Least-squares meet of several rays.

    Because back-projection through any of these ports gives a *straight* line in
    the innermost medium, this is ordinary linear least squares -- which is what
    lets the wand's 3-D points be eliminated in closed form instead of carried as
    unknowns. That elimination is the Schur complement of the point block, not an
    approximation.
    """
    if len(origins) < 2:
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    try:
        if np.linalg.cond(A) > 1e12:
            return None
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def _ray_residual(point: np.ndarray, origin: np.ndarray, direction: np.ndarray,
                  focal: float, cam_center: np.ndarray) -> Tuple[float, float]:
    """Perpendicular offset of `point` from a ray, in two signed components.

    Scaled by focal / (optical path from the camera) so the pair reads as a
    reprojection error in pixels.

    The path length must be measured from the **camera centre**, not from the
    ray's origin. `lineOfSight` returns the segment inside the innermost medium,
    whose origin sits on the port's inner surface, so a point just behind the
    port has a depth of a millimetre or two while its true distance from the
    camera is hundreds. Scaling by that depth would weight such points hundreds
    of times too heavily and wreck the conditioning -- measured on the reference
    sphere the two differ by a factor of 90 across one camera's points.

    Using the geometric path `|origin - C| + depth` ignores how the bends change
    the angular magnification, which is a second-order effect; it reduces to the
    exact pinhole relation when there is no port.
    """
    w = np.asarray(point, float) - origin
    depth = float(np.dot(w, direction))
    perp = w - depth * direction
    e1, e2 = _perp_basis(direction)
    path = float(np.linalg.norm(np.asarray(origin, float) - np.asarray(cam_center, float)))
    scale = focal / max(path + max(depth, 0.0), 1e-6)
    return scale * float(np.dot(perp, e1)), scale * float(np.dot(perp, e2))


# ------------------------------------------------------------- calibrator
class PoseCalibrator:
    """Fit camera poses through a fixed refractive port.

    `make_camera(pose)` must return a configured C++ camera model for that pose;
    everything about the port is baked into that closure, so the same calibrator
    serves plate, cylinder and sphere without knowing which it has.
    """

    def __init__(self, make_camera: Callable[[Pose], Any], poses: Dict[int, Pose],
                 cfg: Optional[PoseFitConfig] = None,
                 gauge: Optional[PortGauge] = None,
                 fixed_cameras: Optional[Sequence[int]] = None):
        if lpt is None:
            raise RuntimeError("pyopenlpt is required for pose calibration")
        self.make_camera = make_camera
        self.poses = {cid: p for cid, p in poses.items()}
        self.cfg = cfg or PoseFitConfig()
        # Only wand fits need this; with known 3-D points there is no orbit.
        self.gauge = gauge or PortGauge()
        self._gauge_reference = {cid: p for cid, p in poses.items()}
        # Holding one camera pins the gauge outright: no transform that preserves
        # the port also leaves that camera where it is. Everything else is then
        # determined relative to it, which is the usual practical remedy when the
        # data are wand frames and the port supplies the only other anchor.
        self.fixed_cameras = set(fixed_cameras or ())
        self.cam_ids = sorted(self.poses)
        self.free_ids = [c for c in self.cam_ids if c not in self.fixed_cameras]
        self._n_par = 7 if self.cfg.optimize_focal else 6

    # -- parameter vector ------------------------------------------------
    def _apply(self, x: np.ndarray, anchors: Dict[int, Pose]) -> Dict[int, Pose]:
        out = {cid: anchors[cid] for cid in self.fixed_cameras if cid in anchors}
        for i, cid in enumerate(self.free_ids):
            blk = x[i * self._n_par:(i + 1) * self._n_par]
            dlogf = float(blk[6]) if self.cfg.optimize_focal else 0.0
            out[cid] = anchors[cid].retract(blk[0:3], blk[3:6], dlogf)
        return out

    def _cams(self, poses: Dict[int, Pose]) -> Dict[int, Any]:
        return {cid: self.make_camera(poses[cid]) for cid in self.cam_ids}

    # -- residuals -------------------------------------------------------

    def _whitening_matrices(self, pose: Pose, pts3d, uvs) -> np.ndarray:
        """Per-observation 2x2 whitening from the pixel -> residual Jacobian.

        Observations are not equally informative. Back-projection is
        near-singular for grazing rays, so at the wall of a curved port a 0.2 px
        detection error becomes a 4-7 px ray residual -- 20-35x the amplification
        at the centre. Left unweighted those observations dominate the
        least-squares cost and drag the fit.

        The amplification is also *anisotropic*: a grazing ray is stretched along
        one direction far more than the other, so a single scalar per observation
        cannot describe it. With `J = d(residual)/d(pixel)`, isotropic pixel noise
        of variance s^2 gives the residual covariance `s^2 J J^T`, and whitening
        is a left-multiplication by `(J J^T)^{-1/2} = U diag(1/S) U^T` from the SVD
        `J = U S V^T`. Each observation then enters the cost with unit variance in
        every direction, which is what least squares assumes.

        Measured at the pose each outer round starts from, and held fixed within
        the round, so least_squares sees one cost. Measured once, at a start a few
        degrees off, the weights made the answer depend on the start: two starts
        of the same data ended 0.014 deg and 0.25 mm apart (D3 of the synthetic
        tests), more than the unweighted fit's spread.
        """
        cam = self.make_camera(pose)
        centre = pose.center()
        step = float(self.cfg.weight_probe_px)
        n = len(uvs)
        out = np.tile(np.eye(2), (n, 1, 1))

        probes = []
        for axis in (0, 1):
            up = np.array(uvs, float)
            dn = np.array(uvs, float)
            up[:, axis] += step
            dn[:, axis] -= step
            probes.append((rays_from_pixels(cam, up), rays_from_pixels(cam, dn)))

        for j in range(n):
            jac = np.zeros((2, 2))
            ok = True
            for axis in (0, 1):
                (o_u, d_u, ok_u), (o_d, d_d, ok_d) = probes[axis]
                if not (ok_u[j] and ok_d[j]):
                    ok = False
                    break
                a = np.array(_ray_residual(pts3d[j], o_u[j], d_u[j], pose.f, centre))
                b = np.array(_ray_residual(pts3d[j], o_d[j], d_d[j], pose.f, centre))
                jac[:, axis] = (a - b) / (2.0 * step)
            if not ok or not np.isfinite(jac).all():
                continue
            u_mtx, sing, _vt = np.linalg.svd(jac)
            sing = np.clip(sing, self.cfg.weight_svd_min, self.cfg.weight_svd_max)
            out[j] = u_mtx @ np.diag(1.0 / sing) @ u_mtx.T
        return out

    def _residuals_one_cam(self, x, anchor: Pose, pts3d, uvs, weights=None):
        """Residuals for a single camera. With the 3-D points known, each camera's
        residuals depend on its own pose alone, so the fit separates."""
        dlogf = float(x[6]) if self.cfg.optimize_focal else 0.0
        pose = anchor.retract(x[0:3], x[3:6], dlogf)
        cam = self.make_camera(pose)
        cam_center = pose.center()
        origins, dirs, ok = rays_from_pixels(cam, uvs)
        out = np.zeros(2 * len(uvs))
        for j in range(len(uvs)):
            if not ok[j]:
                # Total internal reflection at the port edge: the ray does not
                # exist, so there is nothing to measure. A large constant would
                # add a flat, gradient-free plateau; zero simply drops it.
                continue
            out[2 * j], out[2 * j + 1] = _ray_residual(
                pts3d[j], origins[j], dirs[j], pose.f, cam_center)
        if weights is not None:
            pairs = out.reshape(-1, 2)
            out = np.einsum("nij,nj->ni", weights, pairs).ravel()
        return out

    @staticmethod
    def _wand_layout(frames, known_ids):
        """Fix which (frame, endpoint, camera) slots exist, once.

        The residual vector must keep the same length at every evaluation, and
        which rays succeed depends on the pose: a ray can graze into total
        internal reflection, or a triangulation can go singular, as the cameras
        move. Deciding the layout from the *data* and filling failures with zeros
        keeps the length fixed; deriving it from what happened to succeed does
        not, and scipy fails with a shape mismatch part-way through a fit.
        """
        layout = []
        for fr in frames:
            entry = {}
            for tag in ("A", "B"):
                seen = fr["uv_by_cam"].get(tag, {})
                cids = sorted(c for c in seen if c in known_ids)
                entry[tag] = cids if len(cids) >= 2 else []
            layout.append(entry)
        return layout

    def _residuals_wand(self, x, anchors, frames, layout):
        poses = self._apply(x, anchors)
        cams = self._cams(poses)
        out = []
        for fr, entry in zip(frames, layout):
            solved = {}
            for tag in ("A", "B"):
                cids = entry[tag]
                if not cids:
                    solved[tag] = None
                    continue
                o_list, d_list, f_list, c_list, ok_list = [], [], [], [], []
                for cid in cids:
                    o, d, ok = rays_from_pixels(cams[cid], [fr["uv_by_cam"][tag][cid]])
                    ok_list.append(bool(ok[0]))
                    o_list.append(o[0])
                    d_list.append(d[0])
                    f_list.append(poses[cid].f)
                    c_list.append(poses[cid].center())
                good = [k for k, g in enumerate(ok_list) if g]
                point = None
                if len(good) >= 2:
                    point = triangulate(np.asarray([o_list[k] for k in good]),
                                        np.asarray([d_list[k] for k in good]))
                solved[tag] = (point, o_list, d_list, f_list, c_list, ok_list)

            for tag in ("A", "B"):
                cids = entry[tag]
                if not cids:
                    continue
                point, o_list, d_list, f_list, c_list, ok_list = solved[tag]
                for k in range(len(cids)):
                    if point is None or not ok_list[k]:
                        out.extend((0.0, 0.0))
                        continue
                    out.extend(_ray_residual(point, o_list[k], d_list[k],
                                             f_list[k], c_list[k]))

            # The wand length is what fixes scale along the port's weak direction.
            pa = solved["A"][0] if entry["A"] else None
            pb = solved["B"][0] if entry["B"] else None
            if pa is None or pb is None:
                out.append(0.0)
            else:
                got = float(np.linalg.norm(pa - pb))
                out.append((got - float(fr["length"])) / max(self.cfg.wand_sigma_mm, 1e-9))
        return np.asarray(out, float)

    # -- drivers ---------------------------------------------------------
    def _solve(self, residual_fn, n_res_hint: str) -> Dict[str, Any]:
        anchors = {cid: self.poses[cid] for cid in self.cam_ids}
        history = []
        for outer in range(self.cfg.max_outer):
            x0 = np.zeros(len(self.free_ids) * self._n_par)
            res = least_squares(
                residual_fn, x0, args=(anchors,), method="trf",
                loss=self.cfg.loss, f_scale=self.cfg.f_scale,
                max_nfev=self.cfg.max_nfev, ftol=self.cfg.ftol,
                xtol=self.cfg.xtol, gtol=self.cfg.gtol,
            )
            anchors = self._apply(res.x, anchors)
            if not self.fixed_cameras:  # a held camera already pins the gauge
                anchors = self.gauge.project(anchors, self._gauge_reference)
            step = float(np.linalg.norm(res.x))
            rms = float(np.sqrt(2.0 * res.cost / max(res.fun.size, 1)))
            history.append({"outer": outer, "step": step, "rms": rms,
                            "nfev": int(res.nfev), "status": int(res.status)})
            if self.cfg.verbosity:
                print(f"  [{n_res_hint}] outer {outer}: rms {rms:.5f}, "
                      f"step {step:.3e}, nfev {res.nfev}")
            if step < self.cfg.outer_tol:
                break
        self.poses = anchors
        return {"poses": anchors, "history": history,
                "rms": history[-1]["rms"] if history else float("nan"),
                "success": bool(history)}

    def fit_3d2d(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fit to known 3-D points with their image coordinates.

        Solved one camera at a time: with the 3-D points known there is no term
        coupling two cameras, so a joint solve would only make the same problem
        larger and its numeric Jacobian N times more expensive.
        """
        obs_by_cam: Dict[int, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
        for obs in observations:
            X = np.asarray(obs["X_world"], float)
            for cid, uv in obs["uv_by_cam"].items():
                if cid not in self.poses:
                    continue
                pts, uvs = obs_by_cam.setdefault(cid, ([], []))
                pts.append(X)
                uvs.append(np.asarray(uv, float))

        history, rms_all, fitted = {}, [], {}
        for cid in self.free_ids:
            if cid not in obs_by_cam:
                continue
            pts3d = np.asarray(obs_by_cam[cid][0], float)
            uvs = np.asarray(obs_by_cam[cid][1], float)
            anchor = self.poses[cid]
            rounds = []
            for outer in range(self.cfg.max_outer):
                weights = (self._whitening_matrices(anchor, pts3d, uvs)
                           if self.cfg.weight_by_sensitivity else None)
                res = least_squares(
                    self._residuals_one_cam, np.zeros(self._n_par),
                    args=(anchor, pts3d, uvs, weights), method="trf",
                    loss=self.cfg.loss, f_scale=self.cfg.f_scale,
                    max_nfev=self.cfg.max_nfev, ftol=self.cfg.ftol,
                    xtol=self.cfg.xtol, gtol=self.cfg.gtol,
                )
                dlogf = float(res.x[6]) if self.cfg.optimize_focal else 0.0
                anchor = anchor.retract(res.x[0:3], res.x[3:6], dlogf)
                step = float(np.linalg.norm(res.x))
                rms = float(np.sqrt(2.0 * res.cost / max(res.fun.size, 1)))
                rounds.append({"outer": outer, "step": step, "rms": rms,
                               "nfev": int(res.nfev), "status": int(res.status)})
                if self.cfg.verbosity:
                    print(f"  [3d2d cam{cid}] outer {outer}: rms {rms:.5f} px, "
                          f"step {step:.3e}, nfev {res.nfev}")
                if step < self.cfg.outer_tol:
                    break
            self.poses[cid] = anchor
            history[cid] = rounds
            rms_all.append(rounds[-1]["rms"])
            fitted[cid] = (pts3d, uvs, rounds[-1]["rms"])

        retried = self._retry_outliers(fitted, history)

        rms_all = [history[c][-1]["rms"] for c in history]
        return {"poses": self.poses, "history": history, "retried": retried,
                "rms": float(np.sqrt(np.mean(np.square(rms_all)))) if rms_all else float("nan"),
                "success": bool(rms_all)}


    def _retry_outliers(self, fitted, history):
        """Re-fit any camera whose residual is an outlier against its siblings.

        The comparison is to the median across cameras rather than an absolute
        threshold, because the noise floor is not known in advance; the floor
        only stops it chasing noise when every camera is already good.
        """
        if self.cfg.retry_starts <= 0 or len(fitted) < 2:
            return []
        med = float(np.median([v[2] for v in fitted.values()]))
        cutoff = max(med * self.cfg.retry_factor, self.cfg.retry_floor_px)
        rng = np.random.default_rng(self.cfg.retry_seed)
        retried = []
        for cid, (pts3d, uvs, rms) in sorted(fitted.items()):
            if rms <= cutoff:
                continue
            best_pose, best_rms = self.poses[cid], rms
            for _ in range(self.cfg.retry_starts):
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                start = self.poses[cid].retract(
                    axis * np.radians(self.cfg.retry_rot_deg),
                    rng.normal(size=3) * self.cfg.retry_trans_mm)
                anchor = start
                for _ in range(self.cfg.max_outer):
                    weights = (self._whitening_matrices(anchor, pts3d, uvs)
                               if self.cfg.weight_by_sensitivity else None)
                    res = least_squares(
                        self._residuals_one_cam, np.zeros(self._n_par),
                        args=(anchor, pts3d, uvs, weights), method="trf",
                        loss=self.cfg.loss, f_scale=self.cfg.f_scale,
                        max_nfev=self.cfg.max_nfev, ftol=self.cfg.ftol,
                        xtol=self.cfg.xtol, gtol=self.cfg.gtol)
                    dlogf = float(res.x[6]) if self.cfg.optimize_focal else 0.0
                    anchor = anchor.retract(res.x[0:3], res.x[3:6], dlogf)
                    if float(np.linalg.norm(res.x)) < self.cfg.outer_tol:
                        break
                got = float(np.sqrt(2.0 * res.cost / max(res.fun.size, 1)))
                if got < best_rms:
                    best_rms, best_pose = got, anchor
            if best_rms < rms:
                self.poses[cid] = best_pose
                history[cid].append({"outer": "retry", "rms": best_rms,
                                     "was": rms, "starts": self.cfg.retry_starts})
                retried.append({"camera": cid, "rms_before": rms, "rms_after": best_rms})
                if self.cfg.verbosity:
                    print(f"  [3d2d cam{cid}] retried: rms {rms:.4f} -> {best_rms:.4f} px")
        return retried

    def fit_wand(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fit to wand frames: two endpoints a known distance apart, positions unknown.

        A fit whose rms stays above `wand_retry_rms` is rerun from perturbed
        starts and the best answer is kept (see `_retry_wand`).
        """
        layout = self._wand_layout(frames, set(self.poses))
        residual_fn = lambda x, anchors: self._residuals_wand(x, anchors, frames, layout)
        start = dict(self.poses)
        out = self._solve(residual_fn, "wand")
        out["retried"] = self._retry_wand(residual_fn, out, start)
        return out

    def _retry_wand(self, residual_fn, out, start):
        """Rerun a stalled wand fit from perturbed starts; keep the best answer.

        A wand fit can stop at a local minimum degrees from the truth: in the
        synthetic tests one noise-free fit ended 6.8 deg and 42 mm out with rms
        8.2, where the same frames fitted from other starts reach 3e-11.

        Up to `retry_starts` restarts perturb every free camera about the fit's
        own `start`; if none succeeds, as many more perturb them about the
        stalled answer, as `_retry_outliers` does for one camera. On four stalls
        of the synthetic tests the first kind succeeded at 10 of 24 restarts and
        recovered every stall by its third, the second at 4 of 24 and missed one
        stall (the notes repository's code/tests/openlpt_followup/, D4). The
        first restart that brings the rms below `wand_retry_rms` ends the
        search. `out` is updated in place.
        """
        was = out["rms"]
        if (self.cfg.retry_starts <= 0 or not self.free_ids
                or not (was > self.cfg.wand_retry_rms)):
            return []
        stalled = dict(self.poses)
        best = (was, stalled, out["history"])
        tried = 0
        for centre in (start, stalled):
            rng = np.random.default_rng(self.cfg.retry_seed)
            for _ in range(self.cfg.retry_starts):
                trial = dict(centre)
                for cid in self.free_ids:
                    axis = rng.normal(size=3)
                    axis /= np.linalg.norm(axis)
                    trial[cid] = centre[cid].retract(
                        axis * np.radians(self.cfg.retry_rot_deg),
                        rng.normal(size=3) * self.cfg.retry_trans_mm)
                self.poses = trial
                got = self._solve(residual_fn, f"wand retry {tried}")
                tried += 1
                if got["rms"] < best[0]:
                    best = (got["rms"], dict(self.poses), got["history"])
                if best[0] <= self.cfg.wand_retry_rms:
                    break
            if best[0] <= self.cfg.wand_retry_rms:
                break
        self.poses = best[1]
        if best[0] >= was:
            return []
        history = best[2] + [{"outer": "retry", "rms": best[0], "was": was,
                              "starts": tried}]
        out.update(poses=self.poses, history=history, rms=best[0])
        if self.cfg.verbosity:
            print(f"  [wand] retried: rms {was:.4f} -> {best[0]:.4f} ({tried} starts)")
        return [{"rms_before": was, "rms_after": best[0], "starts": tried}]

    # -- diagnostics -----------------------------------------------------
    def reprojection_rmse(self, observations: List[Dict[str, Any]]) -> float:
        """RMS forward-reprojection error in pixels, as an independent check.

        The fit never evaluates the forward model, so this is not the quantity
        that was minimised -- which makes it a meaningful check rather than a
        restatement of the cost.
        """
        cams = self._cams(self.poses)
        sq, n = 0.0, 0
        for obs in observations:
            X = lpt.Pt3D(*np.asarray(obs["X_world"], float))
            for cid, uv in obs["uv_by_cam"].items():
                if cid not in cams:
                    continue
                res = cams[cid].projectBatchStatus([X], False)[0]
                if not res[0]:
                    continue
                du = float(res[1][0]) - float(uv[0])
                dv = float(res[1][1]) - float(uv[1])
                if not (np.isfinite(du) and np.isfinite(dv)):
                    continue
                sq += du * du + dv * dv
                n += 1
        return float(np.sqrt(sq / max(n, 1)))
