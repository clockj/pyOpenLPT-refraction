"""Calibration through a refractive port, against synthetic ground truth.

Observations come from `test/refs/synthetic_ports.py`, which projects known 3-D
points through the C++ camera models. The forward model is checked elsewhere
(`test_refract_curved_*.py`); here the only question is whether the optimiser
recovers the poses it was given.

Three groups:

* **Curved ports with the port given** - cylinder and sphere, from 3-D/2-D pairs
  and from a wand. Pose only, which is the case the project actually needs.
* **Gauge** - with wand data the port's stabiliser is a free direction, and a fit
  that ignores it reaches a *perfect* residual at wrong poses. That failure is
  pinned here deliberately, because a passing residual would otherwise hide it.
* **Flat plate** - that `cv2.calibrateCamera` is biased by the refractive index
  ratio, and that the refraction-aware bootstrap removes the bias.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFS = REPO_ROOT / "test" / "refs"
for _p in (str(REPO_ROOT), str(REFS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

cv2 = pytest.importorskip("cv2")
lpt = pytest.importorskip("pyopenlpt", reason="extension not built")

import synthetic_ports as sp  # noqa: E402
from modules.camera_calibration.curved_calibration.pose_calibrator import (  # noqa: E402
    Pose,
    PortGauge,
    PoseCalibrator,
    PoseFitConfig,
    triangulate,
)

REFRACT = (1.0, 1.49, 1.33)
RADII = (100.0, 94.0)
FOCAL = 1800.0


def _port(kind):
    return sp.CurvedPort(kind, np.zeros(3), np.array([0.0, 0.0, 1.0]), REFRACT, RADII)


def _rig(kind):
    height = 150.0 if kind == "sphere" else 0.0
    return {i: s for i, s in enumerate(
        sp.ring_of_cameras(4, 350.0, height, target=(0, 0, 0.0), span_deg=270.0, f=FOCAL))}


def _bounds(kind):
    return ([-85, -85, -85], [85, 85, 85]) if kind == "sphere" else \
           ([-90, -90, -120], [90, 90, 120])


def _factory(port):
    def make(pose):
        spec = sp.CameraSpec(pose.rvec, pose.tvec, pose.f, pose.cx, pose.cy, 1280, 1024)
        return sp.build_curved_camera(spec, port)
    return make


def _as_pose(spec):
    return Pose(spec.rvec, spec.tvec, spec.f, spec.cx, spec.cy)


def _as_spec(pose):
    return sp.CameraSpec(pose.rvec, pose.tvec, pose.f, pose.cx, pose.cy, 1280, 1024)


def _worst(poses, specs):
    errs = [sp.pose_error(_as_spec(p), specs[c]) for c, p in poses.items()]
    return max(a for a, _ in errs), max(d for _, d in errs)


def _perturbed(specs, rot_deg, trans_mm, rng):
    return {cid: _as_pose(sp.perturb(s, math.radians(rot_deg), trans_mm, rng))
            for cid, s in specs.items()}


# ------------------------------------------------- curved ports, 3D-2D pairs
@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_pose_recovery_from_3d2d(kind):
    """With the port given, known 3-D points must pin the poses exactly."""
    rng = np.random.default_rng(7)
    port, specs = _port(kind), _rig(kind)
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    points = sp.volume_points(400, _bounds(kind), rng, port=port)
    obs = sp.make_3d2d_observations(cams, points, rng)
    assert len(obs) > 150, f"only {len(obs)} points visible to 2+ cameras"

    cal = PoseCalibrator(_factory(port), _perturbed(specs, 3.0, 15.0, rng),
                         PoseFitConfig())
    res = cal.fit_3d2d(obs)
    rot, cen = _worst(res["poses"], specs)
    assert rot < 1e-3, f"rotation error {rot:.3e} deg"
    assert cen < 1e-3, f"centre error {cen:.3e} mm"
    # The fit never evaluates the forward model, so this is an independent check.
    assert cal.reprojection_rmse(obs) < 1e-3


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_pose_recovery_survives_pixel_noise(kind):
    """0.2 px of detection noise must degrade the pose gracefully, not break it."""
    rng = np.random.default_rng(11)
    port, specs = _port(kind), _rig(kind)
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    points = sp.volume_points(400, _bounds(kind), rng, port=port)
    obs = sp.make_3d2d_observations(cams, points, rng, noise_px=0.2)

    cal = PoseCalibrator(_factory(port), _perturbed(specs, 3.0, 15.0, rng),
                         PoseFitConfig())
    res = cal.fit_3d2d(obs)
    rot, cen = _worst(res["poses"], specs)
    assert rot < 0.1, f"rotation error {rot:.4f} deg under 0.2 px noise"
    assert cen < 1.0, f"centre error {cen:.4f} mm under 0.2 px noise"
    assert 0.05 < res["rms"] < 1.0, f"residual {res['rms']:.3f} px is not noise-like"


def test_cylinder_pose_recovery_from_a_poor_start():
    """10 deg and 116 mm off: still recovered, because the retraction is on SO(3)."""
    rng = np.random.default_rng(3)
    port, specs = _port("cylinder"), _rig("cylinder")
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    points = sp.volume_points(400, _bounds("cylinder"), rng, port=port)
    obs = sp.make_3d2d_observations(cams, points, rng)

    init = _perturbed(specs, 10.0, 60.0, rng)
    rot0, cen0 = _worst(init, specs)
    assert rot0 > 5.0 and cen0 > 50.0, "the start was not actually poor"

    cal = PoseCalibrator(_factory(port), init, PoseFitConfig())
    res = cal.fit_3d2d(obs)
    rot, cen = _worst(res["poses"], specs)
    assert rot < 1e-3 and cen < 1e-3, f"rot {rot:.3e} deg, centre {cen:.3e} mm"


# ------------------------------------------------------- curved ports, wand
@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_wand_is_exact_in_the_gauge_quotient(kind):
    """Wand data determines the poses exactly -- but only modulo the port symmetry.

    Scoring the raw poses against truth measures the unobservable direction too,
    and so understates the fit badly. Projecting onto the truth along the gauge
    orbit first -- which changes no residual -- shows what the data actually fixed.
    """
    rng = np.random.default_rng(5)
    port, specs = _port(kind), _rig(kind)
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    frames = sp.make_wand_observations(cams, 50, 60.0, _bounds(kind), rng, port=port)
    assert len(frames) >= 35, f"only {len(frames)} wand frames generated"

    gauge = PortGauge(kind=kind, center=np.asarray(port.center, float),
                      axis=np.asarray(port.axis, float))
    cal = PoseCalibrator(_factory(port), _perturbed(specs, 3.0, 15.0, rng),
                         PoseFitConfig(), gauge)
    res = cal.fit_wand(frames)

    truth = {cid: _as_pose(s) for cid, s in specs.items()}
    rot, cen = _worst(gauge.project(res["poses"], truth), specs)
    assert rot < 1e-3, f"rotation error in the quotient {rot:.3e} deg"
    assert cen < 1e-3, f"centre error in the quotient {cen:.3e} mm"


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_wand_recovers_absolute_poses_when_one_camera_is_held(kind):
    """The practical remedy: no port-preserving transform leaves a held camera put.

    With one camera fixed the gauge is pinned outright and the remaining poses come
    out in absolute terms, which is what a calibration has to deliver.
    """
    rng = np.random.default_rng(5)
    port, specs = _port(kind), _rig(kind)
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    frames = sp.make_wand_observations(cams, 50, 60.0, _bounds(kind), rng, port=port)

    init = _perturbed(specs, 3.0, 15.0, rng)
    init[0] = _as_pose(specs[0])  # the held camera must start at its known pose
    cal = PoseCalibrator(_factory(port), init, PoseFitConfig(), fixed_cameras={0})
    res = cal.fit_wand(frames)

    rot, cen = _worst(res["poses"], specs)
    assert rot < 0.05, f"rotation error {rot:.4f} deg"
    assert cen < 0.5, f"centre error {cen:.4f} mm"


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_wand_without_gauge_fitting_drifts_along_the_port_symmetry(kind):
    """The failure the gauge exists to prevent, pinned so it cannot creep back.

    Rotating every camera and every wand point about a sphere's centre (or about
    a cylinder's axis) changes no residual, so an unconstrained wand fit reaches a
    near-perfect cost at poses that are visibly wrong -- and the error is a pure
    rotation about that axis, not random drift.
    """
    rng = np.random.default_rng(5)
    port, specs = _port(kind), _rig(kind)
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    frames = sp.make_wand_observations(cams, 40, 60.0, _bounds(kind), rng, port=port)

    cal = PoseCalibrator(_factory(port), _perturbed(specs, 3.0, 15.0, rng),
                         PoseFitConfig())  # no gauge
    res = cal.fit_wand(frames)
    rot, cen = _worst(res["poses"], specs)

    assert res["rms"] < 0.05, (
        f"expected a near-perfect residual along the gauge orbit, got {res['rms']:.4f}"
    )
    assert rot > 0.2, (
        f"expected drift along the port symmetry, but the fit stayed put "
        f"({rot:.4f} deg) -- if the calibrator now fixes the gauge by default, "
        f"this test should be updated rather than deleted"
    )
    # The drift is a rotation about the port axis, so centre error tracks radius.
    radius = float(np.mean([np.linalg.norm(s.center()[:2]) for s in specs.values()]))
    assert cen > 0.3 * radius * math.radians(rot), "drift is not rotation-like"


# --------------------------------------------- target on several depth planes
def _plane_stack(port, depths, grid, extent=62.0):
    return sp.plane_stack_points(depths, extent, normal=(1.0, 0.0, 0.0),
                                 grid=grid, port=port)


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_coplanar_target_is_far_worse_conditioned_than_a_depth_stack(kind):
    """One plane is the degenerate layout; extra stations are what fix it.

    Grids are sized so every layout carries about the same number of points, so
    what is compared is the distribution in depth, not the count.
    """
    port, specs = _port(kind), _rig(kind)
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}

    def condition(depths, grid):
        rng = np.random.default_rng(5)
        points = _plane_stack(port, depths, grid)
        obs = sp.make_3d2d_observations(cams, points, rng, noise_px=0.2)
        cid = 0
        pts = np.array([o["X_world"] for o in obs if cid in o["uv_by_cam"]])
        uvs = np.array([o["uv_by_cam"][cid] for o in obs if cid in o["uv_by_cam"]])
        cal = PoseCalibrator(_factory(port), {cid: _as_pose(specs[cid])}, PoseFitConfig())
        step = np.array([1e-6] * 3 + [1e-3] * 3)
        base = cal._residuals_one_cam(np.zeros(6), _as_pose(specs[cid]), pts, uvs)
        jac = np.zeros((base.size, 6))
        for k in range(6):
            e = np.zeros(6)
            e[k] = step[k]
            jac[:, k] = (cal._residuals_one_cam(e, _as_pose(specs[cid]), pts, uvs)
                         - cal._residuals_one_cam(-e, _as_pose(specs[cid]), pts, uvs)) / (2 * step[k])
        return float(np.linalg.cond(jac.T @ jac))

    one = condition([0.0], (16, 15))
    many = condition([-60.0, 0.0, 60.0], (9, 9))
    assert one > 5 * many, (
        f"a single plane should be markedly worse conditioned: "
        f"cond {one:.3e} vs {many:.3e} for three planes"
    )


def test_depth_stack_target_recovers_the_poses():
    """The realistic workflow: a flat target traversed to five stations."""
    rng = np.random.default_rng(5)
    port, specs = _port("cylinder"), _rig("cylinder")
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    points = _plane_stack(port, [-70.0, -35.0, 0.0, 35.0, 70.0], (7, 7))
    obs = sp.make_3d2d_observations(cams, points, rng, noise_px=0.2)
    assert len(obs) > 120, f"only {len(obs)} target points visible to 2+ cameras"

    cal = PoseCalibrator(_factory(port), _perturbed(specs, 3.0, 15.0, rng), PoseFitConfig())
    res = cal.fit_3d2d(obs)
    rot, cen = _worst(res["poses"], specs)
    assert rot < 0.05, f"rotation error {rot:.4f} deg"
    assert cen < 0.3, f"centre error {cen:.4f} mm"


def test_a_camera_stuck_in_a_local_minimum_is_detected_and_retried():
    """The resection is not convex; from a bad start one camera can stall.

    It stalls at a residual tens of times its siblings', which is what makes it
    detectable. Measured frequency on this layout: about 1 start in 25. The guard
    must notice and re-fit rather than return the stuck pose.
    """
    rng = np.random.default_rng(5)
    port, specs = _port("cylinder"), _rig("cylinder")
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    points = _plane_stack(port, [-60.0, 0.0, 60.0], (9, 9))
    obs = sp.make_3d2d_observations(cams, points, rng, noise_px=0.2)
    # The starts must come from the SAME stream, continued after the noise draws:
    # which start lands in the basin is what this test is about, and a fresh
    # generator here silently picks a different (well-behaved) one.
    init = _perturbed(specs, 3.0, 15.0, rng)

    stuck = PoseCalibrator(_factory(port), dict(init),
                           PoseFitConfig(retry_starts=0)).fit_3d2d(obs)
    assert stuck["rms"] > 1.0, (
        "this layout no longer reproduces the local minimum; if the solver was "
        "improved, update this test rather than deleting it"
    )

    fixed = PoseCalibrator(_factory(port), dict(init), PoseFitConfig()).fit_3d2d(obs)
    assert fixed["retried"], "the guard did not fire on a camera that was stuck"
    assert fixed["rms"] < 0.5, f"after retry the residual is still {fixed['rms']:.3f} px"
    rot, cen = _worst(fixed["poses"], specs)
    assert rot < 0.05 and cen < 0.3, f"after retry: {rot:.4f} deg, {cen:.4f} mm"


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_heldout_projection_error_is_at_the_noise_floor(kind):
    """The metric a calibration exists to deliver: unseen points, projected.

    Measured against the *true* camera's own projection with no detection noise
    added, so it is model error alone. Held-out points span the whole working
    volume, including places the training points did not reach.
    """
    rng = np.random.default_rng(5)
    port, specs = _port(kind), _rig(kind)
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    train = sp.volume_points(400, _bounds(kind), rng, port=port)
    obs = sp.make_3d2d_observations(cams, train, rng, noise_px=0.2)

    cal = PoseCalibrator(_factory(port), _perturbed(specs, 3.0, 15.0, rng), PoseFitConfig())
    cal.fit_3d2d(obs)

    held = sp.volume_points(1200, _bounds(kind), np.random.default_rng(9991), port=port)
    fitted = {c: _factory(port)(p) for c, p in cal.poses.items()}
    err = sp.heldout_projection_error(fitted, cams, held)

    assert err["n"] > 1000, f"only {err['n']} held-out projections compared"
    assert err["median"] < 0.5, f"median held-out projection error {err['median']:.3f} px"
    assert err["p95"] < 1.5, f"p95 held-out projection error {err['p95']:.3f} px"
    # A point one camera can see and the other cannot is itself a failure.
    assert err["visibility_mismatch"] < 0.02 * err["n"], (
        f"{err['visibility_mismatch']} visibility mismatches out of {err['n']}"
    )


def test_a_coplanar_target_extrapolates_worse_than_a_depth_stack():
    """Why the fit residual is not enough to judge a calibration.

    One plane can fit its own points as well as a depth stack fits its own -- the
    residuals are comparable -- and still be markedly worse on points elsewhere in
    the volume. Only held-out projections show it.
    """
    port, specs = _port("cylinder"), _rig("cylinder")
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    held = sp.volume_points(1200, _bounds("cylinder"), np.random.default_rng(9991), port=port)

    def fit(depths, grid):
        rng = np.random.default_rng(5)
        train = _plane_stack(port, depths, grid)
        obs = sp.make_3d2d_observations(cams, train, rng, noise_px=0.2)
        cal = PoseCalibrator(_factory(port), _perturbed(specs, 3.0, 15.0, rng), PoseFitConfig())
        res = cal.fit_3d2d(obs)
        fitted = {c: _factory(port)(p) for c, p in cal.poses.items()}
        return res["rms"], sp.heldout_projection_error(fitted, cams, held)

    rms_one, err_one = fit([0.0], (16, 15))
    rms_many, err_many = fit([-70.0, -35.0, 0.0, 35.0, 70.0], (7, 7))

    # Both explain their own data about equally well...
    assert abs(rms_one - rms_many) < 0.15, (
        f"fit residuals differ more than expected: {rms_one:.3f} vs {rms_many:.3f} px"
    )
    # ...but the coplanar one generalises worse.
    assert err_one["p95"] > err_many["p95"], (
        f"expected the coplanar target to extrapolate worse: p95 "
        f"{err_one['p95']:.3f} vs {err_many['p95']:.3f} px"
    )


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_forward_and_backward_models_agree_everywhere(kind):
    """Project a point, back-project the pixel, measure the miss.

    `projectAll` solves in the invariant variables and never traces a ray;
    `lineOfSight` traces quadric intersections with vector Snell and never
    solves. They share almost no code, so agreement is real evidence that both
    are right -- and unlike the existing round-trip tests this covers **every**
    image of a multi-image point and the far branch explicitly, which is where
    the calibration study found the observations that misbehave.

    It matters that this passes at the wall: it shows the near-caustic trouble is
    conditioning, not a disagreement between the two halves of the model.
    """
    port, specs = _port(kind), _rig(kind)
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    points = sp.volume_points(600, _bounds(kind), np.random.default_rng(3), port=port)
    radius = sp.port_radius(port, points)

    worst = {"near": 0.0, "far": 0.0, "wall": 0.0}
    seen = {"near": 0, "far": 0}
    for cid, cam in cams.items():
        n_row, n_col = cam.getNRow(), cam.getNCol()
        for i, p in enumerate(points):
            rays, _cov = cam.projectAll(lpt.Pt3D(*p))
            for r in rays:
                if r["is_ring"]:
                    continue
                u, v = float(r["pt"][0]), float(r["pt"][1])
                if not (0 <= u < n_col and 0 <= v < n_row):
                    continue
                los = cam.lineOfSight(lpt.Pt2D(u, v))
                d = _line_miss(los, p)
                tag = "far" if r["is_far"] else "near"
                seen[tag] += 1
                worst[tag] = max(worst[tag], d)
                if radius[i] > 0.9 * port.inner_radius():
                    worst["wall"] = max(worst["wall"], d)

    assert seen["near"] > 200 and seen["far"] > 100, (
        f"both branches must be exercised, got {seen}"
    )
    # Machine precision, not a tolerance: these are the same geometry computed
    # two independent ways.
    for tag, limit in (("near", 1e-6), ("far", 1e-6), ("wall", 1e-6)):
        assert worst[tag] < limit, f"{tag}: worst miss {worst[tag]:.3e} mm"


def _line_miss(line, point):
    o = np.array([line.pt[0], line.pt[1], line.pt[2]])
    d = np.array([line.unit_vector[0], line.unit_vector[1], line.unit_vector[2]])
    d = d / np.linalg.norm(d)
    w = np.asarray(point, float) - o
    return float(np.linalg.norm(w - np.dot(w, d) * d))


def test_triangulation_of_straight_rays_is_exact():
    """Back-projection gives straight lines, so triangulation is linear least squares."""
    rng = np.random.default_rng(1)
    port, specs = _port("sphere"), _rig("sphere")
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    from modules.camera_calibration.curved_calibration.pose_calibrator import (
        rays_from_pixels,
    )
    worst = 0.0
    for _ in range(40):
        X = sp.volume_points(1, _bounds("sphere"), rng, port=port)[0]
        origins, dirs = [], []
        for cid, cam in cams.items():
            uv, valid = sp.project_points(cam, X[None, :])
            if not valid[0]:
                continue
            o, d, ok = rays_from_pixels(cam, [uv[0]])
            if ok[0]:
                origins.append(o[0])
                dirs.append(d[0])
        if len(origins) < 2:
            continue
        got = triangulate(np.asarray(origins), np.asarray(dirs))
        worst = max(worst, float(np.linalg.norm(got - X)))
    assert worst < 1e-6, f"triangulation off by {worst:.3e} mm"


# ---------------------------------------------------------------- the plate
def _plate_case(seed=0, noise_px=0.0):
    rng = np.random.default_rng(seed)
    port = sp.PlatePort(pt=np.zeros(3), normal=np.array([0.0, 0.0, -1.0]),
                        thickness=10.0, n_air=1.0, n_window=1.49, n_object=1.33)
    specs = {i: s for i, s in enumerate(
        sp.ring_of_cameras(4, 420.0, 430.0, target=(0.0, 0.0, -110.0), f=FOCAL))}
    cams = {i: sp.build_camera(s, port) for i, s in specs.items()}
    points = sp.volume_points(400, ([-130, -130, -210], [130, 130, -35]), rng)
    obs = sp.make_3d2d_observations(cams, points, rng, noise_px=noise_px)
    return port, specs, cams, obs


def test_pinhole_init_is_biased_by_the_refractive_index_ratio():
    """Why plate calibration started outside its own bounds.

    `cv2.calibrateCamera` fits a non-refractive model to refracted observations,
    so it inflates the focal length and pushes the camera back by roughly
    n_object / n_air. The refinement then allows only +-5% on focal and +-50 mm on
    translation, which cannot undo a 33% / 280 mm bias.
    """
    import copy

    from modules.camera_calibration.plate_calibration.refraction_plate_calibration import (
        RefractionPlateCalibrator,
        RefractionPlateConfig,
    )

    port, specs, _cams, obs = _plate_case()
    cal = RefractionPlateCalibrator(
        cfg=RefractionPlateConfig(verbosity=0),
        observations=copy.deepcopy(obs),
        cam_intrinsics={c: {"width": s.width, "height": s.height, "focal_px": s.f}
                        for c, s in specs.items()},
        cam_to_window={c: 0 for c in specs},
        window_media={0: {"n1": port.n_air, "n2": port.n_window,
                          "n3": port.n_object, "thickness": port.thickness}},
        dist_mode=0,
    )
    cal._init_pinhole_per_camera()

    ratio = port.n_object / port.n_air
    for cid in cal.cam_ids:
        vec = cal.cam_ref[cid]
        f_ratio = float(vec[6]) / specs[cid].f
        assert f_ratio == pytest.approx(ratio, rel=0.05), (
            f"cam{cid}: focal inflated by {f_ratio:.4f}, expected ~{ratio}"
        )
        rmtx = cv2.Rodrigues(vec[0:3].reshape(3, 1))[0]
        center = -rmtx.T @ vec[3:6]
        dist_ratio = np.linalg.norm(center) / np.linalg.norm(specs[cid].center())
        assert dist_ratio > 1.2, f"cam{cid}: camera not pushed back ({dist_ratio:.3f})"

    # And the bias exceeds what the refinement is allowed to undo.
    bias_mm = max(
        np.linalg.norm(-cv2.Rodrigues(cal.cam_ref[c][0:3].reshape(3, 1))[0].T
                       @ cal.cam_ref[c][3:6] - specs[c].center())
        for c in cal.cam_ids
    )
    assert bias_mm > RefractionPlateConfig().limit_trans_mm, (
        f"centre bias {bias_mm:.1f} mm should exceed the +-"
        f"{RefractionPlateConfig().limit_trans_mm} mm bound"
    )


def test_pinhole_init_throws_the_principal_point_off_the_sensor():
    """The worst of the three plate defects, and the least obvious.

    Refraction at oblique incidence is asymmetric, and cv2 absorbs the asymmetry
    by sliding the principal point -- here to cy = 1388 px on a 1024 px sensor.
    Nothing downstream ever corrects it: `RefractionPlateCalibrator._layout`
    offers cam_f, cam_k1 and cam_k2 but no principal-point parameter at all, so a
    calibration inherits it permanently.
    """
    import copy

    from modules.camera_calibration.plate_calibration.refraction_plate_calibration import (
        RefractionPlateCalibrator,
        RefractionPlateConfig,
    )

    port, specs, _cams, obs = _plate_case()
    cal = RefractionPlateCalibrator(
        cfg=RefractionPlateConfig(verbosity=0),
        observations=copy.deepcopy(obs),
        cam_intrinsics={c: {"width": s.width, "height": s.height, "focal_px": s.f}
                        for c, s in specs.items()},
        cam_to_window={c: 0 for c in specs},
        window_media={0: {"n1": port.n_air, "n2": port.n_window,
                          "n3": port.n_object, "thickness": port.thickness}},
        dist_mode=0,
    )
    cal._init_pinhole_per_camera()

    for cid, spec in specs.items():
        cy = float(cal.cam_ref[cid][8])
        assert cy > spec.height, (
            f"cam{cid}: expected the principal point to land off the sensor "
            f"(height {spec.height}), got cy = {cy:.1f}"
        )

    # And the refinement has no parameter that could ever bring it back.
    layout_kinds = {t for t, _pid, _sub in cal._layout(
        enable_cam_f=True, enable_cam_k1=True, enable_cam_k2=True)}
    assert not (layout_kinds & {"cam_cx", "cam_cy"}), (
        "a principal-point parameter now exists; update this test rather than "
        "deleting it"
    )


def test_sign_aligned_axis_average_flips_and_wrecks_the_plane_normal():
    """`_init_windows` averages optical axes after aligning signs to the first one.

    Optical axes are directions without a sign, and for an opposed camera pair the
    true dot product is already small -- 0.246 at 37.9 deg of tilt here -- so the
    refraction bias pushes it negative, one axis is flipped, and the flipped axis
    no longer cancels. The averaged normal comes out tens of degrees wrong. Taking
    the dominant eigenvector of sum(a a^T) instead is sign-free and survives it.
    """
    import copy

    from modules.camera_calibration.curved_calibration.plate_bootstrap import (
        robust_plane_normal,
    )
    from modules.camera_calibration.plate_calibration.refraction_plate_calibration import (
        RefractionPlateCalibrator,
        RefractionPlateConfig,
    )

    port, specs, _cams, obs = _plate_case()
    cal = RefractionPlateCalibrator(
        cfg=RefractionPlateConfig(verbosity=0),
        observations=copy.deepcopy(obs),
        cam_intrinsics={c: {"width": s.width, "height": s.height, "focal_px": s.f}
                        for c, s in specs.items()},
        cam_to_window={c: 0 for c in specs},
        window_media={0: {"n1": port.n_air, "n2": port.n_window,
                          "n3": port.n_object, "thickness": port.thickness}},
        dist_mode=0,
    )
    cal._init_pinhole_per_camera()
    cal._init_windows()

    truth = np.asarray(port.normal, float)
    angle = lambda n: math.degrees(
        math.acos(min(1.0, abs(float(np.asarray(n, float) @ truth)))))

    assert angle(cal.win_ref[0]["plane_n"]) > 20.0, (
        "the sign-aligned average no longer fails here; if _init_windows was "
        "fixed, update this test rather than deleting it"
    )

    axes, centers = [], []
    for cid in cal.cam_ids:
        rmtx = cv2.Rodrigues(cal.cam_ref[cid][0:3].reshape(3, 1))[0]
        axes.append(rmtx.T @ np.array([0.0, 0.0, 1.0]))
        centers.append(-rmtx.T @ cal.cam_ref[cid][3:6])
    toward = np.mean([o["X_world"] for o in obs], axis=0) - np.mean(centers, axis=0)
    assert angle(robust_plane_normal(axes, toward=toward)) < 2.0, (
        "the eigenvector estimate should survive the same biased poses"
    )


def test_window_midpoint_initialisation_lands_outside_the_plane_bound():
    """The other half of the same problem: the plane's initial guess is arbitrary."""
    import copy

    from modules.camera_calibration.plate_calibration.refraction_plate_calibration import (
        RefractionPlateCalibrator,
        RefractionPlateConfig,
    )

    port, specs, _cams, obs = _plate_case()
    cal = RefractionPlateCalibrator(
        cfg=RefractionPlateConfig(verbosity=0),
        observations=copy.deepcopy(obs),
        cam_intrinsics={c: {"width": s.width, "height": s.height, "focal_px": s.f}
                        for c, s in specs.items()},
        cam_to_window={c: 0 for c in specs},
        window_media={0: {"n1": port.n_air, "n2": port.n_window,
                          "n3": port.n_object, "thickness": port.thickness}},
        dist_mode=0,
    )
    cal._init_pinhole_per_camera()
    cal._init_windows()

    normal = cal.win_ref[0]["plane_n"]
    offset = abs(float(normal @ cal.win_ref[0]["plane_pt"]) - float(normal @ port.pt))
    assert offset > RefractionPlateConfig().limit_plane_d_mm, (
        f"plane initialised {offset:.1f} mm from truth, against a +-"
        f"{RefractionPlateConfig().limit_plane_d_mm} mm bound"
    )


def test_apparent_point_warp_gives_a_usable_pnp_start():
    """The warp is what turns the plate bootstrap from local into global.

    Fitting a straight-line model to bent rays leaves cv2 13 deg and 285 mm out.
    Warping the known points to where the port makes them *appear* and running
    ordinary PnP lands an order of magnitude closer, inside the basin the
    refraction-aware resection converges from.
    """
    from modules.camera_calibration.curved_calibration.plate_bootstrap import (
        apparent_points,
        pnp_pose,
    )

    port, specs, _cams, obs = _plate_case()
    media = dict(thickness=port.thickness, n_air=port.n_air,
                 n_window=port.n_window, n_object=port.n_object)
    for cid, spec in specs.items():
        pts = np.array([o["X_world"] for o in obs if cid in o["uv_by_cam"]])
        uvs = np.array([o["uv_by_cam"][cid] for o in obs if cid in o["uv_by_cam"]])
        warped = apparent_points(pts, port.pt, port.normal, **media)
        got = pnp_pose(warped, uvs, spec.f, spec.cx, spec.cy)
        assert got is not None, f"PnP failed for camera {cid}"
        ang, dist = sp.pose_error(
            sp.CameraSpec(got[0], got[1], spec.f, spec.cx, spec.cy, 1280, 1024), spec)
        assert ang < 5.0, f"cam{cid}: warp+PnP rotation {ang:.3f} deg"
        assert dist < 50.0, f"cam{cid}: warp+PnP centre {dist:.3f} mm"


@pytest.mark.slow
def test_refraction_aware_bootstrap_recovers_the_plate():
    """The fix end to end: scan the plane offset, warp+PnP then resect at each."""
    from modules.camera_calibration.curved_calibration.plate_bootstrap import (
        index_debias,
        scan_plane_offset,
    )

    port, specs, _cams, obs = _plate_case()
    media = dict(thickness=port.thickness, n_air=port.n_air,
                 n_window=port.n_window, n_object=port.n_object)
    rng = np.random.default_rng(2)
    # Start where cv2 would leave it: focal inflated by the index ratio, then
    # de-biased, and the pose several degrees out.
    init = {cid: _as_pose(sp.perturb(s, math.radians(3.0), 40.0, rng))
            for cid, s in specs.items()}
    init = {cid: Pose(p.rvec, p.tvec,
                      index_debias(p.f * port.n_object, port.n_object), p.cx, p.cy)
            for cid, p in init.items()}

    def make(pose, plane_pt, plane_n):
        plate = sp.PlatePort(pt=np.asarray(plane_pt, float),
                             normal=np.asarray(plane_n, float), **media)
        return sp.build_plate_camera(_as_spec(pose), plate)

    normal = np.array([0.0, 0.0, -1.0])
    res = scan_plane_offset(make, init, obs, np.zeros(3), normal,
                            np.linspace(-160.0, 160.0, 9), media,
                            PoseFitConfig(max_outer=2, max_nfev=80))

    plane_err = abs(float(normal @ res.plane_pt) - float(normal @ port.pt))
    assert plane_err < 5.0, f"plane offset off by {plane_err:.3f} mm"
    rot, cen = _worst(res.poses, specs)
    assert rot < 1.0, f"rotation error {rot:.4f} deg"
    assert cen < 10.0, f"centre error {cen:.4f} mm"
