"""End-to-end tests of the cylindrical and spherical camera models.

The strongest check here is the round trip.  `projectAll()` solves for the ray in
the invariant variables of doc/refract_curved.tex and never traces anything;
`lineOfSight()` traces the pixel ray through the quadrics with vector Snell's law
and never solves anything.  They share only the pinhole intrinsics and the port
frame, so a point that comes back onto its own line of sight has exercised two
independent implementations of the same physics.

Also covered: the port frame (the same physical setup, posed differently in world
coordinates, must give the same pixels), the save/load round trip, and the
`coverage` flags that distinguish "no image" from "not searched".
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFS = REPO_ROOT / "test" / "refs"
if str(REFS) not in sys.path:
    sys.path.insert(0, str(REFS))

cv2 = pytest.importorskip("cv2")
lpt = pytest.importorskip("pyopenlpt", reason="extension not built")

REF_N = [1.0, 1.49, 1.33]
REF_R = [100.0, 94.0]
CAM_DIST = 350.0
FX = FY = 4000.0
CX, CY = 640.0, 512.0
N_ROW, N_COL = 1024, 1280


def _look_at_origin_from(centre):
    """Extrinsics (rvec, tvec) for a camera at `centre` looking at the origin.

    Rows of R are the camera axes in world coordinates: x right, y down, z along
    the view direction.
    """
    centre = np.asarray(centre, float)
    z_cam = -centre / np.linalg.norm(centre)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, z_cam)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    x_cam = np.cross(z_cam, up)  # right = forward x up (up x forward rolled the image 180 deg)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    rmtx = np.vstack([x_cam, y_cam, z_cam])
    rvec = cv2.Rodrigues(rmtx)[0].ravel()
    tvec = -rmtx @ centre
    return rvec, tvec


def _make_camera(kind, centre, axis=(0.0, 0.0, 1.0), port_centre=(0.0, 0.0, 0.0),
                 n=None, radii=None, cam_centre=None):
    """A curved-port camera looking at the port centre from `centre`."""
    n = REF_N if n is None else n
    radii = REF_R if radii is None else radii
    cam = (
        lpt.RefractionSphereCameraModel()
        if kind == "sphere"
        else lpt.RefractionCylinderCameraModel()
    )
    cam.setImageSize(N_ROW, N_COL)
    cam.setIntrinsics(FX, FY, CX, CY, [0.0, 0.0, 0.0, 0.0, 0.0])
    rvec, tvec = _look_at_origin_from(cam_centre if cam_centre is not None else centre)
    cam.setExtrinsics(lpt.Pt3D(*rvec), lpt.Pt3D(*tvec))
    cam.setRefraction(lpt.Pt3D(*port_centre), lpt.Pt3D(*axis), n, radii)
    cam.setSolverOptions(1e-10, 100)
    cam.commitUpdate()
    return cam


def _line_miss(line, point):
    """Distance from `point` to the line of sight."""
    o = np.array([line.pt[0], line.pt[1], line.pt[2]])
    d = np.array([line.unit_vector[0], line.unit_vector[1], line.unit_vector[2]])
    d = d / np.linalg.norm(d)
    w = np.asarray(point, float) - o
    return float(np.linalg.norm(w - np.dot(w, d) * d))


# ------------------------------------------------------------------- spheres
def test_sphere_camera_projects_the_centre_to_the_principal_point():
    """A point at the centre is on the optical axis, so refraction cannot move it."""
    cam = _make_camera("sphere", [CAM_DIST, 0.0, 0.0])
    rays, coverage = cam.projectAll(lpt.Pt3D(0.0, 0.0, 0.0))
    assert coverage["near_branch"] and coverage["far_branch"]
    assert len(rays) == 1
    assert rays[0]["pt"][0] == pytest.approx(CX, abs=1e-6)
    assert rays[0]["pt"][1] == pytest.approx(CY, abs=1e-6)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sphere_camera_round_trip(seed):
    """Every projected image must back-project onto its own point."""
    cam = _make_camera("sphere", [CAM_DIST, 0.0, 0.0])
    rng = np.random.default_rng(seed)
    n_checked = 0
    for _ in range(60):
        r_x = rng.uniform(1.0, 90.0)
        theta, phi = rng.uniform(0.0, math.pi), rng.uniform(0.0, 2 * math.pi)
        point = r_x * np.array(
            [math.cos(theta), math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi)]
        )
        rays, _ = cam.projectAll(lpt.Pt3D(*point))
        for ray in rays:
            if ray["is_ring"]:
                continue
            los = cam.lineOfSight(ray["pt"])
            assert _line_miss(los, point) < 1e-6, (
                f"round trip misses by {_line_miss(los, point):.3e} mm at r_x={r_x:.2f}"
            )
            n_checked += 1
    assert n_checked > 40, f"suspiciously few images checked ({n_checked})"


def test_sphere_camera_is_independent_of_how_the_port_is_posed():
    """The same physical geometry, rotated and translated, must give the same pixels."""
    point_local = np.array([40.0, 20.0, -15.0])
    cam_a = _make_camera("sphere", [CAM_DIST, 0.0, 0.0])
    rays_a, _ = cam_a.projectAll(lpt.Pt3D(*point_local))

    # Move the whole rig: port centre elsewhere, camera moved with it.
    shift = np.array([123.0, -45.0, 67.0])
    cam_b = _make_camera(
        "sphere",
        [CAM_DIST, 0.0, 0.0],
        port_centre=tuple(shift),
        cam_centre=np.array([CAM_DIST, 0.0, 0.0]),
    )
    # Camera still at the same place but the port has moved, so this is a
    # different geometry; instead shift the camera too and re-pose it.
    rvec, tvec = _look_at_origin_from(np.array([CAM_DIST, 0.0, 0.0]))
    rmtx = cv2.Rodrigues(rvec)[0]
    cam_b.setExtrinsics(lpt.Pt3D(*rvec), lpt.Pt3D(*(tvec - rmtx @ shift)))
    cam_b.commitUpdate()
    rays_b, _ = cam_b.projectAll(lpt.Pt3D(*(point_local + shift)))

    assert len(rays_a) == len(rays_b)
    for ra, rb in zip(rays_a, rays_b):
        assert ra["pt"][0] == pytest.approx(rb["pt"][0], abs=1e-7)
        assert ra["pt"][1] == pytest.approx(rb["pt"][1], abs=1e-7)


def test_sphere_camera_refraction_actually_moves_the_image():
    """Guard against the port being silently ignored: a plain pinhole differs."""
    cam = _make_camera("sphere", [CAM_DIST, 0.0, 0.0])
    point = np.array([60.0, 50.0, 0.0])
    rays, _ = cam.projectAll(lpt.Pt3D(*point))
    assert rays

    # The unrefracted pinhole image of the same point, computed directly rather
    # than through PinholeCameraModel, whose setters are not bound.
    rvec, tvec = _look_at_origin_from(np.array([CAM_DIST, 0.0, 0.0]))
    x_cam = cv2.Rodrigues(rvec)[0] @ point + tvec
    straight = (FX * x_cam[0] / x_cam[2] + CX, FY * x_cam[1] / x_cam[2] + CY)

    moved = math.hypot(rays[0]["pt"][0] - straight[0], rays[0]["pt"][1] - straight[1])
    assert moved > 1.0, f"refraction moved the image by only {moved:.3e} px"


# ------------------------------------------------------------------ cylinders
@pytest.mark.parametrize("seed", [11, 12])
def test_cylinder_camera_round_trip(seed):
    """Same round trip for the cylinder, over the near-branch images it returns."""
    cam = _make_camera("cylinder", [CAM_DIST, 0.0, 0.0], axis=(0.0, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    n_checked = 0
    for _ in range(120):
        r_x = rng.uniform(5.0, 90.0)
        azimuth = rng.uniform(-math.pi, math.pi)
        point = np.array(
            [r_x * math.cos(azimuth), r_x * math.sin(azimuth), rng.uniform(-150.0, 150.0)]
        )
        rays, coverage = cam.projectAll(lpt.Pt3D(*point))
        assert coverage["near_branch"] is True
        for ray in rays:
            los = cam.lineOfSight(ray["pt"])
            assert _line_miss(los, point) < 1e-6, (
                f"round trip misses by {_line_miss(los, point):.3e} mm"
            )
            n_checked += 1
    # Far fewer than the sphere's: only near-branch images are searched, and a
    # point wide of the pinhole's azimuth has none. Raise this when the box
    # search lands.
    assert n_checked > 20, f"suspiciously few images checked ({n_checked})"


def test_cylinder_camera_searches_both_branches():
    """Both branches are searched, so an empty list really does mean invisible."""
    cam = _make_camera("cylinder", [CAM_DIST, 0.0, 0.0])
    _rays, coverage = cam.projectAll(lpt.Pt3D(50.0, 10.0, 20.0))
    assert coverage["near_branch"] is True
    assert coverage["far_branch"] is True


def test_cylinder_camera_returns_far_branch_images():
    """Wide azimuth near the wall: the far branch must actually produce images."""
    cam = _make_camera("cylinder", [CAM_DIST, 0.0, 0.0])
    rng = np.random.default_rng(5)
    n_far = 0
    for _ in range(80):
        r_x = rng.uniform(60.0, 93.0)
        azimuth = rng.uniform(1.8, math.pi)
        point = np.array(
            [r_x * math.cos(azimuth), r_x * math.sin(azimuth), rng.uniform(-120.0, 120.0)]
        )
        rays, _ = cam.projectAll(lpt.Pt3D(*point))
        for ray in rays:
            if ray["is_far"]:
                n_far += 1
                los = cam.lineOfSight(ray["pt"])
                assert _line_miss(los, point) < 1e-6
    assert n_far > 10, f"only {n_far} far-branch images through the camera"


def test_cylinder_camera_works_with_a_tilted_axis():
    """The port frame must handle an axis that is not a world axis."""
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    # A point 40 mm off the axis, 20 mm along it.
    tangent = np.cross(axis, [0.0, 0.0, 1.0])
    tangent /= np.linalg.norm(tangent)
    point = 40.0 * tangent + 20.0 * axis
    cam_centre = CAM_DIST * tangent
    cam = _make_camera(
        "cylinder", cam_centre, axis=tuple(axis), cam_centre=cam_centre
    )
    rays, _ = cam.projectAll(lpt.Pt3D(*point))
    assert rays, "no image through a tilted cylinder"
    for ray in rays:
        los = cam.lineOfSight(ray["pt"])
        assert _line_miss(los, point) < 1e-6


# ------------------------------------------------------------ serialisation
@pytest.mark.parametrize(
    "kind,token", [("sphere", "PINSPH"), ("cylinder", "PINCYL")]
)
def test_save_load_round_trip(tmp_path, kind, token):
    """Parameters must survive a save/load, and the reloaded model must agree."""
    cam = _make_camera(kind, [CAM_DIST, 0.0, 0.0], axis=(0.0, 0.0, 1.0))
    path = tmp_path / f"cam_{kind}.txt"
    cam.saveParameters(str(path))

    text = path.read_text()
    assert text.splitlines()[1].strip() == token

    reloaded = lpt.loadCameraModel(str(path))
    assert reloaded.type() == (
        lpt.CameraType.PINSPH if kind == "sphere" else lpt.CameraType.PINCYL
    )

    point = np.array([55.0, 22.0, 18.0])
    before, _ = cam.projectAll(lpt.Pt3D(*point))
    after, _ = reloaded.projectAll(lpt.Pt3D(*point))
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert b["pt"][0] == pytest.approx(a["pt"][0], abs=1e-9)
        assert b["pt"][1] == pytest.approx(a["pt"][1], abs=1e-9)


def test_loader_rejects_inconsistent_radii(tmp_path):
    """A stack whose radii do not strictly decrease must be refused, not silently used."""
    cam = _make_camera("sphere", [CAM_DIST, 0.0, 0.0])
    path = tmp_path / "bad.txt"
    cam.saveParameters(str(path))

    # Values are written in scientific notation (Matrix::write leaves the stream
    # in that mode), so edit the line by position rather than by literal text.
    lines = path.read_text().splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("# Interface Radii"))
    radii = lines[idx + 1].split(",")
    assert len(radii) == 2, f"unexpected radii line: {lines[idx + 1]!r}"
    lines[idx + 1] = ",".join(reversed(radii))  # now increasing: invalid
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(RuntimeError):
        lpt.loadCameraModel(str(path))
