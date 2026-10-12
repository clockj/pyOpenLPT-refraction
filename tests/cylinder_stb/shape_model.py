"""The astigmatic shape of particle images through the tube, and its OTF table (plan: Phase 4, L3).

The model, per camera:
  - The camera is the calibrated PINCYL model: a pinhole (the entrance pupil's centre) and the tube.  Its lens has an
    entrance pupil of diameter A [mm] and is focused at the depth z_f [mm] (along the optical axis, from the pupil).
  - Through the tube a point X has two virtual images: one at depth z_tan(X) for the fan of rays across the tube's axis
    (spread along the image's u) and one at z_sag(X) for the fan along it (v).  Each is measured as the parallax of a
    displaced pinhole: moving the pinhole by h along the camera's x moves the image by du = -f h / z_tan, along y by
    dv = -f h / z_sag (a plain pinhole gives the point's own depth for both, the self-test below).
  - Defocus: the rays from a virtual image at depth z through a pupil of diameter A spread, in the plane in focus, over
    A |1 - z_f / z|; seen from the pupil that is A |1/z - 1/z_f| radians, f A |1/z - 1/z_f| pixels.  So the image is
    the pupil compressed to the blur lengths
        L_h = A f (1/z_tan - 1/z_f),   L_v = A f (1/z_sag - 1/z_f)          [px, signed: the side of focus]
    (a uniform ellipse with full axes |L_h|, |L_v|), blurred by a Gaussian of sigma0 [px] (diffraction, the particle's
    own size, the pixels).  A particle in tangential focus (z_tan = z_f) images as a vertical line, in sagittal focus as
    a horizontal one.
  - Intensity: defocus spreads the particle's shadow without changing its integral F, so the peak of the OTF's
    segment model is a = F / (w(|L_h|) w(|L_v|)), with w(L) the integral of the peak-1 profile g(t; L) of
    OTF::intensity (w = L / (2 Phi(L / 2 sigma0) - 1), and sigma0 sqrt(2 pi) at L = 0).
The OTF table (OTF::loadParam's file, model "segment"): on a grid over the view volume, per camera, b and c the box
lengths with the second moments of the ellipse's axes (a uniform ellipse of full axis L has variance L^2/16 along it,
a box of length L_box L_box^2/12, so L_box = (sqrt(3)/2) L; signed, the C++ uses |b|, |c|), a = F/(w(b) w(c)),
alpha = the image angle of the direction across the tube's axis (0 when the axis is vertical in the image), sigma =
sigma0.  The line detector's RMS widths sqrt(L^2/12 + s^2) are then the model's sqrt(L_h^2/16 + sigma0^2), which is
what the depth cue compares (StereoMatch's tracer check).  The sympy checks of these steps: sympy_checks.py.
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.special import erf

import cstb


# ------------------------------------------------------------------ cameras with a displaced pinhole
def _lines(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().split("\n")


def _idx(lines, header):
    return next(k for k, ln in enumerate(lines) if ln.startswith(header)) + 1


def _rt(lines):
    i_r = _idx(lines, "# Rotation Matrix")
    i_t = _idx(lines, "# Translation Vector")
    i_k = _idx(lines, "# Camera Matrix")
    R = np.array([[float(x) for x in lines[i_r + k].split(",")] for k in range(3)])
    t = np.array([float(x) for x in lines[i_t].split(",")])
    K = np.array([[float(x) for x in lines[i_k + k].split(",")] for k in range(3)])
    return R, t, K


def camera_geometry(path):
    """(R world->camera, t, K) of a camFile."""
    return _rt(_lines(path))


def displaced_camfile(src, dst, shift_cam, tol=1e-9, nmax=80):
    """src with its pinhole moved by shift_cam (camera axes, mm), orientation kept; projection tolerance tol."""
    lines = _lines(src)
    R, t, _K = _rt(lines)
    t2 = t - np.asarray(shift_cam, float)          # the pinhole moves by R^T shift_cam
    fmt = lambda v: ",".join("%.16e" % x for x in v)
    lines[_idx(lines, "# Translation Vector")] = fmt(t2)
    lines[_idx(lines, "# Inverse of Translation Vector")] = fmt(-R.T @ t2)
    if any(ln.startswith("# Projection Tolerance") for ln in lines):        # curved ports only
        lines[_idx(lines, "# Projection Tolerance")] = "%.17e" % tol
        lines[_idx(lines, "# Maximum Number of Iterations for Projection")] = "%d" % nmax
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return dst


class FocalDepths:
    """z_tan(X), z_sag(X) of one camera by the parallax of its pinhole displaced by +-h along its x and y."""

    def __init__(self, lpt, camfile, h=0.5, work=None):
        self.lpt = lpt
        self.R, self.t, self.K = camera_geometry(camfile)
        self.h = h
        work = work or os.path.join(cstb.DATA, "shape_model")
        stem = os.path.splitext(os.path.basename(camfile))[0] + "_" + str(abs(hash(os.path.abspath(camfile))) % 10 ** 8)
        self.cams = {}
        for tag, s in (("x+", (h, 0, 0)), ("x-", (-h, 0, 0)), ("y+", (0, h, 0)), ("y-", (0, -h, 0))):
            p = displaced_camfile(camfile, os.path.join(work, f"{stem}_{tag}.txt"), s)
            self.cams[tag] = lpt.loadCameraModel(p)

    def __call__(self, X):
        """(z_tan, z_sag) [mm] for points X (n, 3); NaN where a displaced camera has no image.  Each point is first
        projected by the x+ camera (the full search), and its ray starts the other three (hinted projections: the
        pinholes are 1 mm apart, so the rays nearly coincide)."""
        X = np.atleast_2d(np.asarray(X, float))
        fx, fy = self.K[0, 0], self.K[1, 1]
        out = np.full((len(X), 2), np.nan)
        lpt = self.lpt
        for i, x in enumerate(X):
            P = lpt.Pt3D(*map(float, x))
            ok, p0, h0, _e = self.cams["x+"].projectWithHint(P, lpt.ProjectionHint())
            if not ok:
                continue
            uv = {"x+": p0}
            good = True
            for tag in ("x-", "y+", "y-"):
                ok, p, _h, _e = self.cams[tag].projectWithHint(P, h0)
                if not ok:
                    good = False
                    break
                uv[tag] = p
            if not good:
                continue
            du = uv["x+"][0] - uv["x-"][0]
            dv = uv["y+"][1] - uv["y-"][1]
            out[i, 0] = -fx * 2 * self.h / du
            out[i, 1] = -fy * 2 * self.h / dv
        return out


# ------------------------------------------------------------------ the model
BOX = math.sqrt(3.0) / 2.0     # box length / full axis of a uniform ellipse with the same second moment


def blur_lengths(z_tan, z_sag, f, A, z_f):
    """Signed blur lengths (L_h, L_v) [px] of a pupil of diameter A [mm] focused at z_f [mm]."""
    return A * f * (1.0 / z_tan - 1.0 / z_f), A * f * (1.0 / z_sag - 1.0 / z_f)


def profile_integral(L, sigma):
    """w(L): the integral of the peak-1 profile g(t; L) of OTF::intensity's segment model."""
    L = np.abs(np.asarray(L, float))
    small = L < 1e-3 * sigma
    with np.errstate(invalid="ignore", divide="ignore"):
        w = L / erf(L / (2 * math.sqrt(2) * sigma))
    return np.where(small, sigma * math.sqrt(2 * math.pi), w)


def axis_angle(lpt, cam, X, axis, eps=0.05):
    """The image angle [rad] of the direction across the tube's axis at X: the angle of the axis' image minus 90 deg
    (0 when the axis is vertical in the image, pointing down)."""
    a = cam.project(lpt.Pt3D(*map(float, X)))
    b = cam.project(lpt.Pt3D(*map(float, np.asarray(X) + eps * np.asarray(axis))))
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    alpha = ang - math.pi / 2
    # fold into (-pi/2, pi/2]: the segment model is symmetric
    while alpha <= -math.pi / 2:
        alpha += math.pi
    while alpha > math.pi / 2:
        alpha -= math.pi
    return alpha


# ------------------------------------------------------------------ the OTF table
def grid_points(vol, n):
    """Grid nodes in OTF order (x_id * ny*nz + y_id * nz + z_id) over vol = ((x0,y0,z0),(x1,y1,z1))."""
    gx = np.linspace(vol[0][0], vol[1][0], n[0])
    gy = np.linspace(vol[0][1], vol[1][1], n[1])
    gz = np.linspace(vol[0][2], vol[1][2], n[2])
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def write_otf(path, vol, n, a, b, c, alpha, sigma):
    """OTF::saveParam's format with model "segment"; a, b, c, alpha, sigma are (n_cam, n_grid) arrays."""
    n_cam = a.shape[0]
    n_grid = n[0] * n[1] * n[2]
    lines = ["# Size: (n_cam,nx,ny,nz,n_grid)", f"{n_cam},{n[0]},{n[1]},{n[2]},{n_grid}",
             "# Boundary: (xmin,xmax,ymin,ymax,zmin,zmax)",
             ",".join("%.9g" % v for v in (vol[0][0], vol[1][0], vol[0][1], vol[1][1], vol[0][2], vol[1][2]))]
    for name, arr in (("a", a), ("b", b), ("c", c), ("alpha", alpha)):
        lines.append(f"# {name}: (n_cam,n_grid)")
        lines += [",".join("%.9g" % v for v in arr[k]) for k in range(n_cam)]
    lines += ["# Model: (gauss or segment; segment: b, c are lengths [px])", "segment", "# sigma: (n_cam,n_grid)"]
    lines += [",".join("%.9g" % v for v in sigma[k]) for k in range(n_cam)]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def build_otf(lpt, camfiles, params, vol, n, path, axis=(0.0, 1.0, 0.0)):
    """The OTF table of the shape model.  params[k] = {"A": mm, "z_f": mm, "sigma0": px, "F": grey levels x px^2,
    optionally "delta": mm, a depth offset added to z_tan and z_sag}.  Returns the arrays and writes path."""
    P = grid_points(vol, n)
    n_cam = len(camfiles)
    a = np.zeros((n_cam, len(P)))
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    alpha = np.zeros_like(a)
    sig = np.zeros_like(a)
    for k, f in enumerate(camfiles):
        fd = FocalDepths(lpt, f)
        z = fd(P)
        bad = ~np.all(np.isfinite(z), axis=1)
        if bad.any():          # nodes without an image (outside the water): the nearest good value
            good = np.flatnonzero(~bad)
            for i in np.flatnonzero(bad):
                j = good[np.argmin(np.linalg.norm(P[good] - P[i], axis=1))]
                z[i] = z[j]
        pk = params[k]
        f_px = math.sqrt(fd.K[0, 0] * fd.K[1, 1])
        Lh, Lv = blur_lengths(z[:, 0] + pk.get("delta", 0.0), z[:, 1] + pk.get("delta", 0.0), f_px, pk["A"],
                              pk["z_f"])
        Lh, Lv = BOX * Lh, BOX * Lv               # the box lengths of equal second moment
        cam = lpt.loadCameraModel(f)
        al = np.array([axis_angle(lpt, cam, x, axis) if not bad[i] else 0.0 for i, x in enumerate(P)])
        b[k], c[k], alpha[k] = Lh, Lv, al
        sig[k] = pk["sigma0"]
        a[k] = pk["F"] / (profile_integral(Lh, pk["sigma0"]) * profile_integral(Lv, pk["sigma0"]))
    write_otf(path, vol, n, a, b, c, alpha, sig)
    return {"points": P, "a": a, "b": b, "c": c, "alpha": alpha, "sigma": sig}


def self_test(lpt, camfile):
    """A PINHOLE copy of the camera (same K, R, t): both focal depths equal the point's own depth."""
    lines = _lines(camfile)
    R, t, _K = _rt(lines)
    pin = os.path.join(cstb.DATA, "shape_model", "selftest_pinhole.txt")
    keep = lines[:_idx(lines, "# Inverse of Translation Vector") + 1]
    keep[next(k for k, ln in enumerate(keep) if ln.strip() == "PINCYL")] = "PINHOLE"
    os.makedirs(os.path.dirname(pin), exist_ok=True)
    with open(pin, "w", encoding="utf-8") as fh:
        fh.write("\n".join(keep) + "\n")
    fd = FocalDepths(lpt, pin)
    rng = np.random.default_rng(5)
    X = rng.uniform(-10, 10, (50, 3))
    z = fd(X)
    z_true = (X @ R.T + t)[:, 2]
    return float(np.max(np.abs(z - z_true[:, None])))
