"""Refraction-aware initialisation for flat-plate calibration.

`cv2.calibrateCamera` fits a *non-refractive* pinhole model to observations that
were refracted, so its answer is biased by a known physical factor: an object at
true depth z behind a flat interface appears at z/n, and the fit compensates by
pushing the camera back and inflating the focal length, both by roughly the index
ratio n_object/n_air.  On the reference rig that is a 33% focal error and a 280 mm
centre error, against optimiser bounds of +-5% and +-50 mm -- so the refinement
starts outside its own reachable set and can never recover.

This module replaces that start.  The plane's offset along its normal is the one
parameter that cannot be guessed, so it is scanned; at each candidate offset the
camera poses are re-solved *through the refractive model* by
`PoseCalibrator.fit_3d2d`, and the offset with the lowest residual wins.  Because
each camera's resection is an independent 6- or 7-parameter problem the scan is
cheap.

The scan is over a scalar, so it needs no initial guess for the offset at all --
only a bracket, which the geometry supplies: the plate must lie between the
cameras and the measurement volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pose_calibrator import Pose, PoseCalibrator, PoseFitConfig


@dataclass
class PlateBootstrapResult:
    poses: Dict[int, Pose]
    plane_pt: np.ndarray
    plane_n: np.ndarray
    offset: float          # signed distance moved along the normal from `origin`
    rms: float             # ray residual in pixels at the chosen offset
    scanned: List[Tuple[float, float]]  # (offset, rms) for every candidate tried


def index_debias(f_cv2: float, n_object: float, n_air: float = 1.0) -> float:
    """The focal length cv2 would have found without refraction.

    A first-order correction only: it removes the bulk of the bias so a bounded
    refinement can reach the rest. Exactness comes from the scan that follows.
    """
    return float(f_cv2) * float(n_air) / float(n_object)


def robust_plane_normal(axes: Sequence[np.ndarray],
                        toward: Optional[np.ndarray] = None) -> np.ndarray:
    """Average a set of optical axes into a plane normal, without sign flips.

    `_init_windows` averages the axes after aligning each one's sign against the
    first, `dot(a, ref) < 0 -> a = -a`. That heuristic breaks in two ways:

    * For an opposed pair of cameras the true dot is already small -- 0.246 for
      cameras tilted 37.9 deg from the normal -- so a modest pose error flips it.
      On the reference rig the 13 deg refraction bias took it to -0.212, one axis
      was flipped, and the averaged normal came out **50.6 deg** from truth.
    * Beyond 45 deg of tilt the true dot is negative anyway, so it flips a
      *correct* axis even from perfect poses.

    Axes are directions without a sign, so the right average is the dominant
    eigenvector of sum(a a^T) -- which is what a line fit through the origin
    means, and is invariant to flipping any input. The sign is then chosen once,
    at the end, to point along `toward` (the direction away from the cameras).
    """
    A = np.asarray([np.asarray(a, float) / np.linalg.norm(a) for a in axes], float)
    _vals, vecs = np.linalg.eigh(A.T @ A)
    normal = vecs[:, -1]  # eigenvector of the largest eigenvalue
    if toward is not None:
        toward = np.asarray(toward, float)
        if float(normal @ toward) < 0.0:
            normal = -normal
    return normal / np.linalg.norm(normal)


def trustworthy_intrinsics(poses: Dict[int, Pose], width: float, height: float,
                           n_object: float, n_air: float = 1.0,
                           principal: Optional[Dict[int, Tuple[float, float]]] = None,
                           ) -> Dict[int, Pose]:
    """Undo what fitting a non-refractive model did to the intrinsics.

    Two corrections, and the second matters more than the first:

    * **Focal**: divide out the index ratio (see `index_debias`). Leaves ~1%,
      which the scan's free focal then recovers exactly.
    * **Principal point**: discard it. This is the one that actually blocks the
      fit. Refraction at oblique incidence is asymmetric, and `cv2` absorbs that
      asymmetry by sliding the principal point -- on the reference rig to
      cy = 1388 px on a 1024 px sensor, 876 px *outside the image*. From there
      even PnP on warped points starts 25 deg out and the refinement cannot
      recover (it ran the focal away to 1.5x instead). Reset to the sensor centre
      and the same fit lands at 1e-06 deg with the focal back to 1.0000.

    The principal point is a property of the lens and sensor, so it should come
    from an in-air calibration or the sensor centre -- never from a fit to
    refracted observations. Pass `principal` to supply known values per camera.
    """
    out = {}
    for cid, pose in poses.items():
        cx, cy = (principal or {}).get(cid, (0.5 * float(width), 0.5 * float(height)))
        out[cid] = Pose(pose.rvec, pose.tvec,
                        index_debias(pose.f, n_object, n_air), cx, cy)
    return out


def apparent_points(points: np.ndarray, plane_pt: np.ndarray, normal: np.ndarray,
                    thickness: float, n_air: float, n_window: float,
                    n_object: float) -> np.ndarray:
    """Where a flat port makes the points *appear* to be, to first order.

    Rays from a point at perpendicular depth `t` behind an interface, refracted
    into a lighter medium, extend back to a virtual point at depth `t * n_air /
    n_object` -- the everyday "the pool is shallower than it looks" effect. The
    window slab adds its own shift.

    This matters because it turns the bootstrap from a local refinement into a
    global one: warp the known points here, run ordinary PnP, and the answer is
    already close, instead of asking a local solver to climb out of the 13-degree
    basin that fitting a straight-line model to bent rays leaves it in.

    `plane_pt` is the interface nearest the camera; `normal` points away from it
    into the object medium.
    """
    points = np.atleast_2d(np.asarray(points, float))
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    plane_pt = np.asarray(plane_pt, float)

    # Depth of each point beyond the far interface, along the normal.
    far_pt = plane_pt + float(thickness) * normal
    depth = (points - far_pt) @ normal
    depth = np.maximum(depth, 0.0)

    shift = depth * (1.0 - float(n_air) / float(n_object))
    shift = shift + float(thickness) * (1.0 - float(n_air) / float(n_window))
    return points - np.outer(shift, normal)


def pnp_pose(points: np.ndarray, uvs: np.ndarray, focal: float, cx: float,
             cy: float, rvec0=None, tvec0=None):
    """Ordinary PnP against (already warped) points. Returns (rvec, tvec) or None."""
    import cv2 as _cv2

    obj = np.asarray(points, np.float64).reshape(-1, 1, 3)
    img = np.asarray(uvs, np.float64).reshape(-1, 1, 2)
    if len(obj) < 6:
        return None
    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    ok, rvec, tvec = _cv2.solvePnP(
        obj, img, K, np.zeros(5),
        useExtrinsicGuess=rvec0 is not None,
        rvec=None if rvec0 is None else np.asarray(rvec0, float).reshape(3, 1).copy(),
        tvec=None if tvec0 is None else np.asarray(tvec0, float).reshape(3, 1).copy(),
        flags=_cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return rvec.ravel(), tvec.ravel()


def scan_plane_offset(
    make_camera: Callable[[Pose, np.ndarray, np.ndarray], Any],
    poses_init: Dict[int, Pose],
    observations: List[Dict[str, Any]],
    origin: np.ndarray,
    normal: np.ndarray,
    offsets: Sequence[float],
    media: Dict[str, float],
    cfg: Optional[PoseFitConfig] = None,
    refine: bool = True,
    verbosity: int = 0,
) -> PlateBootstrapResult:
    """Pick the plane offset whose refraction-aware resection fits best.

    `make_camera(pose, plane_pt, plane_n)` must build a camera for that pose and
    that plane. `offsets` are signed distances along `normal` from `origin`.
    `media` carries thickness / n_air / n_window / n_object for the warp.
    """
    # The focal must be free. cv2's is inflated by the index ratio, and de-biasing
    # leaves ~1% -- which a fixed-focal fit absorbs into camera distance instead
    # (8.6 mm on the reference rig) rather than showing up as residual. Freed from
    # the warp+PnP start it comes back exact; freed from cv2's own 13-degree start
    # it runs away to zero, which is why the warp matters first.
    cfg = cfg or PoseFitConfig(optimize_focal=True, max_outer=3, max_nfev=120)

    obs_by_cam: Dict[int, Any] = {}
    for obs in observations:
        X = np.asarray(obs["X_world"], float)
        for cid, uv in obs["uv_by_cam"].items():
            if cid not in poses_init:
                continue
            pts, uvs = obs_by_cam.setdefault(cid, ([], []))
            pts.append(X)
            uvs.append(np.asarray(uv, float))
    obs_by_cam = {cid: (np.asarray(p), np.asarray(u))
                  for cid, (p, u) in obs_by_cam.items()}
    origin = np.asarray(origin, float)
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)

    scanned: List[Tuple[float, float]] = []
    best: Optional[PlateBootstrapResult] = None

    def evaluate(offset: float) -> PlateBootstrapResult:
        plane_pt = origin + offset * normal
        # Restart the poses from PnP on the warped points at *this* offset rather
        # than refining whatever the previous offset produced: the start is then
        # global, and a bad offset cannot poison its neighbours.
        start = {}
        for cid, pose in poses_init.items():
            pts, uvs = obs_by_cam.get(cid, (None, None))
            if pts is None or len(pts) < 6:
                start[cid] = pose
                continue
            warped = apparent_points(pts, plane_pt, normal, media["thickness"],
                                     media["n_air"], media["n_window"],
                                     media["n_object"])
            got = pnp_pose(warped, uvs, pose.f, pose.cx, pose.cy)
            start[cid] = pose if got is None else Pose(got[0], got[1], pose.f,
                                                       pose.cx, pose.cy)
        cal = PoseCalibrator(
            lambda pose, _pt=plane_pt: make_camera(pose, _pt, normal), start, cfg)
        try:
            res = cal.fit_3d2d(observations)
            rms = float(res["rms"])
        except Exception:
            rms = float("inf")
        if not np.isfinite(rms):
            rms = float("inf")
        return PlateBootstrapResult(dict(cal.poses), plane_pt, normal.copy(),
                                    float(offset), rms, [])

    for offset in offsets:
        cand = evaluate(offset)
        scanned.append((float(offset), cand.rms))
        if verbosity:
            print(f"    offset {offset:9.2f} mm -> rms {cand.rms:10.4f} px")
        if best is None or cand.rms < best.rms:
            best = cand

    if best is None:
        raise RuntimeError("plane offset scan produced no candidates")

    # Golden-section style tightening around the best grid point, which is enough
    # because the residual is smooth and unimodal in the offset once the poses
    # are re-solved at each candidate.
    if refine and len(offsets) >= 3:
        grid = np.asarray(sorted(offsets), float)
        i = int(np.argmin(np.abs(grid - best.offset)))
        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, len(grid) - 1)]
        for _ in range(6):
            if hi - lo < 1e-3:
                break
            mids = [lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0]
            cands = [evaluate(m) for m in mids]
            for m, c in zip(mids, cands):
                scanned.append((float(m), c.rms))
                if verbosity:
                    print(f"    refine {m:9.2f} mm -> rms {c.rms:10.4f} px")
                if c.rms < best.rms:
                    best = c
            if cands[0].rms <= cands[1].rms:
                hi = mids[1]
            else:
                lo = mids[0]

    best.scanned = scanned
    return best


def default_offsets(cam_centers: Sequence[np.ndarray], points: np.ndarray,
                    origin: np.ndarray, normal: np.ndarray, thickness: float = 0.0,
                    n_steps: int = 13) -> np.ndarray:
    """A bracket for the plate, from geometry alone: no initial guess needed.

    `normal` points away from the cameras into the object medium, so a plate at
    offset `d` is physically possible only when every camera is on the near side
    and every point beyond the far interface:

        max_cameras (C - origin).n  <  d  <  min_points (X - origin).n - thickness

    Returns `n_steps` offsets spanning that interval, inset slightly so no
    candidate sits exactly on a camera or a point.

    Note the bracket is one-sided in each variable -- the *nearest* camera and the
    *nearest* point are what bound it. Taking a min and a max of the two sets
    independently gives an interval that need not contain the plate at all.
    """
    origin = np.asarray(origin, float)
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    s_pts = (np.asarray(points, float) - origin) @ normal
    s_cams = np.array([(np.asarray(c, float) - origin) @ normal for c in cam_centers])

    lo = float(s_cams.max())
    hi = float(s_pts.min()) - float(thickness)
    if not (hi > lo):
        raise ValueError(
            f"no room for the plate along this normal: cameras reach {lo:.3f} but "
            f"the nearest point is at {hi:.3f}; check the normal's sign"
        )
    pad = 0.05 * (hi - lo)
    return np.linspace(lo + pad, hi - pad, int(n_steps))
