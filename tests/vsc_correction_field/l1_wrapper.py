"""L1: the corrected camera, method by method (plan section 4), on build vcf.

    python l1_wrapper.py

Cameras: T1's two PINCYL cameras (cases/cyl_20261007/data/cams_T0), T1's cam1 relabelled PINSPH (written to data/l1/),
a wide-angle PINCYL camera 300 mm from T1's tube (f = 1500 px; T1's long lenses almost never put two images of a
point on the sensor, this one does), the pinhole tests/cylinder_stb/data/cams_T2_pinhole/cam1.txt and a PINPLATE
camera of the plate case.

Points, per camera:
- curved ports: 1,000 in T1's view volume; 1,000 over the tube's whole depth (|z| up to 99 mm, so some lie outside the
  water and some near the far wall); 300 whose image lies within 4 px of the sensor's edge (traced back from such
  pixels); for the sphere also 200 behind its centre on and beside the line from the pinhole through the centre;
- the wide camera: 1,500 over the tube's water, and 300 with two or more images;
- pinhole and plate: 1,500 on the lines of sight of pixels spread over the sensor and 40 px beyond it.
Fields: zero; a smooth analytic field (up to 3 px); for T1's cameras the self-calibration's two-pass field
(t1_vc2_total); a constant 3 px shift, on 200 points per camera whose image lies within 3 px of the sensor's right
edge (the wide camera: its bottom edge, since its right edge looks past the tube).

Checks (the Python reference is modules/vsc/correction_io.Correction, which repeats the library's arithmetic, so most
are exact):
- the field: C++ at / toModel / toImage = the reference, bit for bit; a file read back gives the same doubles;
- project(X) = toImage(inner project(X)) exactly, and project(X) - c(project(X)) = inner project(X) to 1e-9 px;
- projectAll: every image moved so, the rays' other fields and the coverage unchanged;
- projectWithHint along 40 tracks of 40 small steps: pixels as project's rule, the hints the inner model's, bit for
  bit;
- imageStatus (margin, merge in px: (0, 0), (5, 2), and (1e5, 0), (0, 1e5), which take every image as on the sensor
  or as merged, so that points with several images reach the census's other branches) = the rule applied to the
  corrected images (Python's census, the primary the inner model's);
- lineOfSight(d) = inner lineOfSight(d - c(d)) exactly; lineOfSight(project(X)) passes X as closely as the inner
  model's round trip (to 1e-9 mm more);
- the other properties are the inner model's (lineImagesCurve: True for single-valued models); clone keeps the field;
  saveParameters writes the inner model's file byte for byte;
- the zero field changes nothing, bit for bit; the constant 3 px shift moves images over the edge, and imageStatus
  follows;
- config.json's "correction" (BasicSetting.readConfig): a relative path is found and the camera wrapped; "" means
  none; a missing file, a wrong image size, a malformed or a too steep file are errors naming the file;
- the Python handle lpt.Camera: setCorrection gives the wrapper's results; its parameter calls still work;
  clearCorrection and loadParameters drop the correction.
Writes l1_wrapper_out.txt and .json here.
"""
from __future__ import annotations

import filecmp
import os
import shutil

import numpy as np

import vcf_common as vc

OUT = os.path.join(vc.DATA, "l1")
RNG = np.random.default_rng(20261008)


def P3(lpt, x):
    return lpt.Pt3D(float(x[0]), float(x[1]), float(x[2]))


def xy(p):
    return np.array([p[0], p[1]], float)


def line(l):
    return np.array([l.pt[i] for i in range(3)]), np.array([l.unit_vector[i] for i in range(3)])


def dist_to_line(x, l):
    p, d = line(l)
    return float(np.linalg.norm(np.cross(np.asarray(x) - p, d)))


def primary_index(rays, n_col, n_row):
    """refraction_curved::primaryImage: on the sensor first, then the near branch, then the smallest |Lambda|."""
    best, best_on = -1, False
    for k, r in enumerate(rays):
        if r["is_ring"]:
            continue
        u, v = r["pt"][0], r["pt"][1]
        on = (u >= 0.0) and (u < n_col) and (v >= 0.0) and (v < n_row)
        if best < 0 or (on and not best_on):
            best, best_on = k, on
            continue
        if on == best_on:
            b = rays[best]
            pref = (not r["is_far"]) if r["is_far"] != b["is_far"] else abs(r["lambda"]) < abs(b["lambda"])
            if pref:
                best, best_on = k, on
    return best


def census(pix, rings, primary, n_col, n_row, margin, merge):
    """refraction_curved::statusOfImages on pixels pix (n, 2)."""
    if any(rings):
        return "ambiguous", None
    on = [k for k, (u, v) in enumerate(pix)
          if u >= -margin and u < n_col + margin and v >= -margin and v < n_row + margin]
    if not on:
        return "none", None
    if len(on) > 1:
        return "ambiguous", None
    one = on[0]
    for k, (u, v) in enumerate(pix):
        if k == one:
            continue
        du, dv = u - pix[one][0], v - pix[one][1]
        if du * du + dv * dv <= merge * merge:
            return "ambiguous", None
    if primary != one:
        return "ambiguous", None
    return "one", pix[one]


def single_status(p, n_col, n_row, margin):
    """Camera::imageStatus's default: project() and the sensor enlarged by the margin."""
    if p is None:
        return "none", None
    u, v = p
    if u >= -margin and u < n_col + margin and v >= -margin and v < n_row + margin:
        return "one", p
    return "none", None


def proj(cam, lpt, x):
    try:
        return xy(cam.project(P3(lpt, x)))
    except RuntimeError:
        return None


def all_rays(cam, lpt, x):
    try:
        rays, cov = cam.projectAll(P3(lpt, x))
        return rays, cov
    except RuntimeError as e:
        return None, str(e)


# ------------------------------------------------------------------ points
def points_curved(lpt, inner, vol):
    lo = np.array([vol["x"][0], vol["y"][0], vol["z"][0]])
    hi = np.array([vol["x"][1], vol["y"][1], vol["z"][1]])
    A = lo + (hi - lo) * RNG.random((1000, 3))
    B = np.array([-15.0, -12.0, -99.0]) + np.array([40.0, 26.0, 198.0]) * RNG.random((1000, 3))
    C = []
    n_col, n_row = inner.getNCol(), inner.getNRow()
    tries = 0
    while len(C) < 300 and tries < 6000:
        tries += 1
        side = RNG.integers(4)
        a = RNG.uniform(-4, 4)
        if side == 0:
            px = (a, RNG.uniform(0, n_row))
        elif side == 1:
            px = (n_col + a, RNG.uniform(0, n_row))
        elif side == 2:
            px = (RNG.uniform(0, n_col), a)
        else:
            px = (RNG.uniform(0, n_col), n_row + a)
        try:
            los = inner.lineOfSight(lpt.Pt2D(*px))
        except RuntimeError:
            continue
        p, d = line(los)
        x = p + RNG.uniform(5.0, 150.0) * d
        q = proj(inner, lpt, x)
        if q is not None and np.hypot(*(q - px)) < 0.5:
            C.append(x)
    return {"volume": A, "depth": B, "edge": np.array(C)}


def points_ray(lpt, inner, depth_ref, spread):
    """Points on the lines of sight of pixels over the sensor and 40 px beyond, at depths around depth_ref."""
    n_col, n_row = inner.getNCol(), inner.getNRow()
    X = []
    while len(X) < 1500:
        px = (RNG.uniform(-40, n_col + 40), RNG.uniform(-40, n_row + 40))
        try:
            los = inner.lineOfSight(lpt.Pt2D(*px))
        except RuntimeError:
            continue
        p, d = line(los)
        t0 = depth_ref(p, d)
        X.append(p + (t0 + RNG.uniform(-spread, spread)) * d)
    return {"rays": np.array(X)}


def points_axial(lpt, inner, file, n=200):
    """Sphere: points behind the centre, on and next to the line from the pinhole through the centre, where a point's
    image is a ring (on the line) or two images close together (beside it)."""
    with open(file, encoding="utf-8") as f:
        rows = [ln.strip() for ln in f if ln.strip()]
    def after(tag):
        k = next(i for i, r in enumerate(rows) if r.startswith(tag))
        return np.array([float(v) for v in rows[k + 1].split(",")])
    centre = after("# Port Center")
    pinhole = after("# Inverse of Translation Vector")
    a = (centre - pinhole) / np.linalg.norm(centre - pinhole)
    e1 = np.cross(a, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    X = []
    for k in range(n):
        s = RNG.uniform(5.0, 90.0)
        r = 0.0 if k % 4 == 0 else RNG.uniform(0.0, 0.3)
        phi = RNG.uniform(0, 2 * np.pi)
        X.append(centre + s * a + r * (np.cos(phi) * e1 + np.sin(phi) * e2))
    return np.array(X)


def write_wide_camera(path, dist=300.0, f=1500.0):
    """T1's cam1 turned into a wide-angle camera 300 mm from the tube's axis (f = 1500 px, looking at the axis): its
    view reaches the far wall, where far-branch images fold and pairs of images lie on the sensor."""
    with open(vc.CAMS_T1[0], encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    def setrow(tag, rows):
        k = next(i for i, ln in enumerate(lines) if ln.startswith(tag))
        for j, r in enumerate(rows):
            lines[k + 1 + j] = ",".join(f"{v:.16e}" for v in r)
    setrow("# Camera Matrix", [(f, 0, 639.5), (0, f, 399.5), (0, 0, 1)])
    setrow("# Rotation Vector", [(np.pi, 0, 0)])
    setrow("# Rotation Matrix", [(1, 0, 0), (0, -1, 0), (0, 0, -1)])
    setrow("# Inverse of Rotation Matrix", [(1, 0, 0), (0, -1, 0), (0, 0, -1)])
    setrow("# Translation Vector", [(0, 0, dist)])
    setrow("# Inverse of Translation Vector", [(0, 0, dist)])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def points_wide(lpt, inner, n_tube=1500, n_multi=300):
    """The wide camera: points over the tube's water, and points with two or more images (by rejection)."""
    def tube(m):
        X = np.array([-97, -60, -97]) + np.array([194, 120, 194]) * RNG.random((m, 3))
        return X[np.hypot(X[:, 0], X[:, 2]) < 97.0]
    multi = []
    while len(multi) < n_multi:
        for x in tube(4000):
            rays, _cov = all_rays(inner, lpt, x)
            if rays is not None and len(rays) > 1:
                multi.append(x)
    return {"tube": tube(int(n_tube * 1.3))[:n_tube], "several": np.array(multi[:n_multi])}


def tracks_from(X, n=40, steps=40, h=0.05):
    out = []
    for k in range(min(n, len(X))):
        x = X[k].copy()
        tr = [x.copy()]
        for _ in range(steps - 1):
            x = x + h * RNG.normal(size=3)
            tr.append(x.copy())
        out.append(np.array(tr))
    return out


# ------------------------------------------------------------------ the checks of one camera and one field
def check_camera(lpt, log, name, file, corr_file, ref, sets, tracks, zero_field=False):
    inner = lpt.loadCameraModel(file)
    wrap = lpt.loadCameraModel(file, corr_file)
    n_col, n_row = inner.getNCol(), inner.getNRow()
    res = {"camera": name, "correction": os.path.basename(corr_file), "fails": []}

    def fail(msg):
        res["fails"].append(msg)

    # properties
    props = {"type": (wrap.type() == inner.type()), "size": (wrap.getNCol(), wrap.getNRow()) == (n_col, n_row),
             "multiValued": wrap.multiValued() == inner.multiValued(),
             "usesProjectionHint": wrap.usesProjectionHint() == inner.usesProjectionHint(),
             "lineImagesCurve": wrap.lineImagesCurve() == (inner.lineImagesCurve() if inner.multiValued() else True),
             "class": type(wrap).__name__ == "CorrectedCameraModel"}
    for k, ok in props.items():
        if not ok:
            fail(f"property {k}")
    res["properties_ok"] = all(props.values())

    X = np.vstack([v for v in sets.values() if len(v)])
    n_proj = n_fail_both = 0
    exact = 0
    resid = []
    rt_extra = []
    n_status = n_status_ok = 0
    status_counts = {}
    n_rays = n_rays_ok = 0
    n_none_inner = 0
    n_multi = 0
    for x in X:
        q = proj(inner, lpt, x)
        d = proj(wrap, lpt, x)
        if (q is None) != (d is None):
            fail(f"project fails in one model only at {x.tolist()}")
            continue
        # projectAll
        ri, ci = all_rays(inner, lpt, x)
        rw, cw = all_rays(wrap, lpt, x)
        if (ri is None) != (rw is None):
            fail(f"projectAll fails in one model only at {x.tolist()}")
            continue
        if ri is not None:
            n_rays += 1
            n_multi += len(ri) > 1
            ok = len(ri) == len(rw) and ci == cw
            for a, b in zip(ri, rw):
                ok &= all(a[k] == b[k] for k in ("is_far", "lambda", "c_z", "is_ring"))
                if not a["is_ring"]:
                    want = ref.to_image(xy(a["pt"]))[0]
                    ok &= bool(np.array_equal(want, xy(b["pt"])))
            n_rays_ok += bool(ok)
            if not ok:
                fail(f"projectAll differs at {x.tolist()}")
        if q is None:
            n_none_inner += 1
            n_fail_both += 1
        else:
            n_proj += 1
            want = ref.to_image(q)[0]
            if np.array_equal(want, d):
                exact += 1
            resid.append(float(np.hypot(*(d - ref.at(d)[0] - q))))
            # the round trip
            try:
                li = inner.lineOfSight(lpt.Pt2D(*q))
                lw = wrap.lineOfSight(lpt.Pt2D(*d))
                rt_extra.append(dist_to_line(x, lw) - dist_to_line(x, li))
            except RuntimeError:
                fail(f"lineOfSight of a projection fails at {x.tolist()}")
        # imageStatus
        for margin, merge in ((0.0, 0.0), (5.0, 2.0), (1e5, 0.0), (0.0, 1e5)):
            st, pt = wrap.imageStatus(P3(lpt, x), margin, merge)
            if not inner.multiValued() or inner.provablySingleImage(P3(lpt, x)):
                want_st, want_pt = single_status(d, n_col, n_row, margin)
            elif ri is None:
                want_st, want_pt = "none", None
            else:
                prim = primary_index(ri, n_col, n_row)
                pix = [ref.to_image(xy(r["pt"]))[0] if not r["is_ring"] else np.array([np.nan, np.nan])
                       for r in ri]
                want_st, want_pt = census(pix, [r["is_ring"] for r in ri], prim, n_col, n_row, margin, merge)
            n_status += 1
            ok = st == want_st and (want_pt is None or np.array_equal(xy(pt), want_pt))
            n_status_ok += bool(ok)
            status_counts[st] = status_counts.get(st, 0) + 1
            if not ok:
                fail(f"imageStatus {st} != {want_st} at {x.tolist()} (margin {margin})")
        if wrap.provablySingleImage(P3(lpt, x)) != inner.provablySingleImage(P3(lpt, x)):
            fail("provablySingleImage differs")
    res.update({"points": int(len(X)), "projected": n_proj, "no_image": n_none_inner,
                "project_exact": exact, "project_resid_max_px": max(resid) if resid else 0.0,
                "roundtrip_extra_max_mm": max(rt_extra) if rt_extra else 0.0,
                "projectAll_ok": f"{n_rays_ok}/{n_rays}", "imageStatus_ok": f"{n_status_ok}/{n_status}",
                "statuses": status_counts, "points_with_several_images": n_multi})
    if exact != n_proj:
        fail(f"project: {n_proj - exact} of {n_proj} differ from toImage(inner)")
    if resid and max(resid) > 1e-9:
        fail(f"project's fixed point: residual {max(resid):.2e} px")
    if rt_extra and max(rt_extra) > 1e-9:
        fail(f"round trip: {max(rt_extra):.2e} mm worse than the inner model's")

    # lineOfSight at pixels
    pix = np.column_stack([RNG.uniform(-40, n_col + 40, 500), RNG.uniform(-40, n_row + 40, 500)])
    los_ok = 0
    for p in pix:
        try:
            lw = wrap.lineOfSight(lpt.Pt2D(*p))
        except RuntimeError:
            lw = None
        m = ref.to_model(p)[0]
        try:
            li = inner.lineOfSight(lpt.Pt2D(*m))
        except RuntimeError:
            li = None
        if lw is None and li is None:
            los_ok += 1
        elif lw is not None and li is not None:
            a, b = line(lw), line(li)
            los_ok += bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
    res["lineOfSight_ok"] = f"{los_ok}/{len(pix)}"
    if los_ok != len(pix):
        fail("lineOfSight differs from inner(d - c(d))")

    # projectWithHint along tracks: the wrapper's pixel is toImage of the inner's, the hints the inner's
    n_h = n_h_ok = 0
    for tr in tracks:
        hi, hw = lpt.ProjectionHint(), lpt.ProjectionHint()
        for x in tr:
            oki, pi_, hi, _ei = inner.projectWithHint(P3(lpt, x), hi)
            okw, pw, hw, _ew = wrap.projectWithHint(P3(lpt, x), hw)
            n_h += 1
            same_hint = (hi.valid == hw.valid and hi.lam == hw.lam and hi.c_z == hw.c_z and hi.is_far == hw.is_far)
            ok = oki == okw and same_hint
            if oki and okw:
                ok &= bool(np.array_equal(ref.to_image(xy(pi_))[0], xy(pw)))
            n_h_ok += bool(ok)
    res["projectWithHint_ok"] = f"{n_h_ok}/{n_h}"
    if n_h_ok != n_h:
        fail("projectWithHint differs")

    # clone and saveParameters
    cl = wrap.clone()
    cl_ok = type(cl).__name__ == "CorrectedCameraModel" and cl.correction().source == wrap.correction().source
    for x in X[:50]:
        a, b = proj(wrap, lpt, x), proj(cl, lpt, x)
        cl_ok &= (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))
    res["clone_ok"] = bool(cl_ok)
    if not cl_ok:
        fail("clone")
    f1, f2 = os.path.join(OUT, "save_wrapped.txt"), os.path.join(OUT, "save_inner.txt")
    wrap.saveParameters(f1)
    inner.saveParameters(f2)
    res["saveParameters_same"] = filecmp.cmp(f1, f2, shallow=False)
    if not res["saveParameters_same"]:
        fail("saveParameters differs from the inner model's")

    if zero_field:
        # the zero field: every result equal to the inner model's
        same = 0
        for x in X:
            a, b = proj(inner, lpt, x), proj(wrap, lpt, x)
            same += (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))
        res["zero_field_project_same"] = f"{same}/{len(X)}"
        if same != len(X):
            fail("the zero field changes project()")
    status = "PASS" if not res["fails"] else "FAIL"
    log(f"  {status} {name} / {os.path.basename(corr_file)}: {res['points']} points ({res['projected']} with an image, "
        f"{res['no_image']} without); project exact {res['project_exact']}/{res['projected']}, fixed-point residual "
        f"<= {res['project_resid_max_px']:.1e} px; round trip <= inner + {res['roundtrip_extra_max_mm']:.1e} mm; "
        f"projectAll {res['projectAll_ok']} ({res['points_with_several_images']} with several images); imageStatus {res['imageStatus_ok']} {res['statuses']}; lineOfSight "
        f"{res['lineOfSight_ok']}; projectWithHint {res['projectWithHint_ok']}; clone {res['clone_ok']}; "
        f"saveParameters same {res['saveParameters_same']}"
        + (f"; zero field: project same {res['zero_field_project_same']}" if zero_field else ""))
    for m in res["fails"][:8]:
        log(f"      {m}")
    return res


def check_field_binding(lpt, log, ref, path):
    """The C++ field against the reference: values read back, at / toModel / toImage, bit for bit."""
    c = lpt.ImageCorrection.load(path)
    ok_vals = (np.array_equal(np.array(c.du).reshape(ref.nv, ref.nu), ref.du)
               and np.array_equal(np.array(c.dv).reshape(ref.nv, ref.nu), ref.dv)
               and (c.n_col, c.n_row, c.u0, c.v0, c.step, c.nu, c.nv)
               == (ref.n_col, ref.n_row, ref.u0, ref.v0, ref.step, ref.nu, ref.nv))
    pts = np.vstack([np.column_stack([RNG.uniform(-60, ref.n_col + 60, 3000), RNG.uniform(-60, ref.n_row + 60, 3000)]),
                     ref.nodes()[::37], ref.nodes()[::41] + 0.5])
    n_at = n_tm = n_ti = 0
    for p in pts:
        P = lpt.Pt2D(*p)
        n_at += bool(np.array_equal(xy(c.at(P)), ref.at(p)[0]))
        n_tm += bool(np.array_equal(xy(c.toModel(P)), ref.to_model(p)[0]))
        n_ti += bool(np.array_equal(xy(c.toImage(P)), ref.to_image(p)[0]))
    slope_ok = abs(c.slope_bound - ref.slope_bound()) <= 1e-15 * max(1.0, ref.slope_bound())
    ok = ok_vals and n_at == n_ti == n_tm == len(pts) and slope_ok
    log(f"  {'PASS' if ok else 'FAIL'} field {os.path.basename(path)}: values read back {ok_vals}; at {n_at}/{len(pts)}, "
        f"toModel {n_tm}/{len(pts)}, toImage {n_ti}/{len(pts)} bit for bit; slope bound {c.slope_bound:.4f} "
        f"(reference {ref.slope_bound():.4f})")
    return {"file": os.path.basename(path), "values_ok": bool(ok_vals), "at": n_at, "toModel": n_tm,
            "toImage": n_ti, "n": len(pts), "slope_bound": c.slope_bound, "ok": bool(ok)}


def edge_points(lpt, inner, depth, axis, n=200):
    """Points whose image lies within 3 px of the sensor's right edge (axis 0) or bottom edge (axis 1), traced back
    from such pixels at depths depth(p, d) along the line of sight (kept when the projection returns to the pixel
    within 0.5 px)."""
    size = (inner.getNCol(), inner.getNRow())
    X, tries = [], 0
    while len(X) < n and tries < 20 * n:
        tries += 1
        px = [RNG.uniform(0, size[0]), RNG.uniform(0, size[1])]
        px[axis] = RNG.uniform(size[axis] - 3.0, size[axis])
        try:
            los = inner.lineOfSight(lpt.Pt2D(*px))
        except RuntimeError:
            continue
        p, d = line(los)
        x = p + depth(p, d) * d
        q = proj(inner, lpt, x)
        if q is not None and np.hypot(*(q - px)) < 0.5:
            X.append(x)
    return np.array(X)


def check_edge_shift(lpt, log, name, file, edge_pts, axis):
    """A constant shift of +3 px along u (axis 0) or v (axis 1): images within 3 px of that edge leave the sensor."""
    inner = lpt.loadCameraModel(file)
    size = (inner.getNCol(), inner.getNRow())
    shift = vc.Correction.from_function(size[0], size[1], lambda uv: np.column_stack(
        [np.full(len(uv), 3.0 if axis == 0 else 0.0), np.full(len(uv), 3.0 if axis == 1 else 0.0)]))
    path = os.path.join(OUT, f"shift3_{name}.txt")
    shift.write(path, [f"L1: a constant shift of 3 px in {'uv'[axis]}"])
    wrap = lpt.loadCameraModel(file, path)
    moved = left = agree = 0
    for x in edge_pts:
        q = proj(inner, lpt, x)
        if q is None:
            continue
        st_i, _ = inner.imageStatus(P3(lpt, x), 0.0, 0.0)
        st_w, _ = wrap.imageStatus(P3(lpt, x), 0.0, 0.0)
        on = 0 <= q[0] < size[0] and 0 <= q[1] < size[1]
        if on and q[axis] >= size[axis] - 3.0 and st_i == "one":
            moved += 1
            left += st_w == "none"
        d = proj(wrap, lpt, x)
        if d is not None:
            agree += bool(abs(d[axis] - q[axis] - 3.0) < 1e-9 and abs(d[1 - axis] - q[1 - axis]) < 1e-12)
    ok = moved > 0 and left == moved
    edge = ("right", "bottom")[axis]
    log(f"  {'PASS' if ok else 'FAIL'} {name}, constant 3 px shift in {'uv'[axis]}: {moved} images within 3 px of the "
        f"{edge} edge (status one) -> {left} of them none once shifted; projections shifted by exactly 3 px: {agree}")
    return {"camera": name, "edge": edge, "edge_images": moved, "left_sensor": left, "shift_exact": agree,
            "ok": bool(ok)}


def check_config(lpt, log, corr_dir):
    """config.json's "correction", read by BasicSetting.readConfig."""
    base = vc.load_json(os.path.join(vc.CASE_RUNS, "s3_final_m15", "config_final_m15.json"))
    out = {}

    def read(cams, tag):
        cfg = dict(base)
        cfg["cameras"] = cams
        path = os.path.join(OUT, f"config_{tag}.json")
        vc.save_json(path, cfg)
        s = lpt.BasicSetting()
        try:
            s.readConfig(path)
            return s, None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    rel = [os.path.relpath(os.path.join(corr_dir, f"smooth_cam{k + 1}.txt"), OUT) for k in range(2)]
    s, err = read([{"file": vc.CAMS_T1[k], "max_intensity": 255, "correction": rel[k]} for k in range(2)], "rel")
    ok_rel = s is not None and all(type(c).__name__ == "CorrectedCameraModel" for c in s._cam_list)
    if ok_rel:
        for k, c in enumerate(s._cam_list):
            ok_rel &= os.path.normcase(os.path.abspath(c.correction().source)) == \
                os.path.normcase(os.path.abspath(os.path.join(corr_dir, f"smooth_cam{k + 1}.txt")))
            ref = lpt.loadCameraModel(vc.CAMS_T1[k], os.path.join(corr_dir, f"smooth_cam{k + 1}.txt"))
            x = lpt.Pt3D(5.0, 1.0, -3.0)
            ok_rel &= bool(np.array_equal(xy(c.project(x)), xy(ref.project(x))))
    out["relative_path"] = {"ok": bool(ok_rel), "error": err}
    s, err = read([{"file": vc.CAMS_T1[k], "max_intensity": 255, "correction": ""} for k in range(2)], "empty")
    out["empty_means_none"] = {"ok": s is not None and all(type(c).__name__ != "CorrectedCameraModel"
                                                          for c in s._cam_list), "error": err}
    s, err = read([{"file": vc.CAMS_T1[0], "max_intensity": 255}, {"file": vc.CAMS_T1[1], "max_intensity": 255}],
                  "none")
    out["no_key"] = {"ok": s is not None and all(type(c).__name__ != "CorrectedCameraModel" for c in s._cam_list),
                     "error": err}
    # errors
    missing = os.path.join(OUT, "no_such_correction.txt")
    wrong = vc.Correction.zeros(1280, 801)
    wrong_path = os.path.join(OUT, "wrong_size.txt")
    wrong.write(wrong_path)
    bad_path = os.path.join(OUT, "malformed.txt")
    with open(os.path.join(corr_dir, "smooth_cam1.txt"), encoding="utf-8") as f:
        text = f.read()
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write(text[:text.index("DV")])
    steep_path = os.path.join(OUT, "steep.txt")
    with open(steep_path, "w", encoding="utf-8") as f:   # 8 px of change over one 8 px cell: slope 1
        u0, v0, nu, nv = vc.Correction.grid_for(1280, 800)
        du = np.zeros((nv, nu))
        du[:, 50:] = 8.0
        f.write("VERSION 1\nIMAGE_SIZE 1280 800\nGRID 0.0 0.0 8.0 %d %d\nDU\n" % (nu, nv)
                + "\n".join(" ".join(repr(float(x)) for x in row) for row in du) + "\nDV\n"
                + "\n".join(" ".join("0.0" for _ in range(nu)) for _ in range(nv)) + "\n")
    for tag, path, want in (("missing", missing, "no_such_correction"), ("wrong_size", wrong_path, "not the camera's"),
                            ("malformed", bad_path, "malformed"), ("steep", steep_path, "too steep")):
        s, err = read([{"file": vc.CAMS_T1[0], "max_intensity": 255, "correction": path},
                       {"file": vc.CAMS_T1[1], "max_intensity": 255}], tag)
        named = err is not None and (os.path.basename(path) in err)
        has = err is not None and ((want in err) or tag == "malformed")
        out[tag] = {"ok": s is None and named and has, "error": err}
    ok = all(v["ok"] for v in out.values())
    log(f"  {'PASS' if ok else 'FAIL'} config.json \"correction\": " + "; ".join(f"{k} {'ok' if v['ok'] else 'NO'}"
                                                                           for k, v in out.items()))
    for k, v in out.items():
        if v["error"]:
            log(f"      {k}: {v['error'].splitlines()[0][:220]}")
    out["ok"] = bool(ok)
    return out


def check_handle(lpt, log, corr_dir):
    """lpt.Camera with setCorrection."""
    path = os.path.join(corr_dir, "smooth_cam1.txt")
    h = lpt.Camera(vc.CAMS_T1[0])
    w = lpt.loadCameraModel(vc.CAMS_T1[0], path)
    i = lpt.loadCameraModel(vc.CAMS_T1[0])
    h.setCorrection(path)
    xs = [lpt.Pt3D(*map(float, x)) for x in (np.array([5.0, 1.0, -3.0]) + RNG.normal(size=(20, 3)) * 4)]
    same = all(np.array_equal(xy(h.project(x)), xy(w.project(x))) for x in xs)
    pxs = [lpt.Pt2D(*map(float, p)) for p in RNG.uniform([0, 0], [1280, 800], (20, 2))]
    same_los = all(np.array_equal(line(h.lineOfSight(p))[0], line(w.lineOfSight(p))[0]) for p in pxs)
    flags = h.hasCorrection() and os.path.normcase(h.correctionFile()) == os.path.normcase(path) and \
        type(h.cameraModel()).__name__ == "CorrectedCameraModel"
    h.clearCorrection()
    cleared = (not h.hasCorrection()) and all(np.array_equal(xy(h.project(x)), xy(i.project(x))) for x in xs)
    h.setCorrection(path)
    h.loadParameters(vc.CAMS_T1[0])
    dropped = not h.hasCorrection()
    # a pinhole camera keeps its parameter calls with a correction
    hp = lpt.Camera(vc.CAM_PINHOLE)
    pzero = os.path.join(OUT, "zero_pinhole.txt")
    vc.Correction.zeros(hp.getNCol(), hp.getNRow()).write(pzero)
    hp.setCorrection(pzero)
    try:
        par = hp._pinhole_param
        params_ok = par.n_col == hp.getNCol()
    except Exception:  # noqa: BLE001
        params_ok = False
    try:
        hp.setCorrection(os.path.join(OUT, "wrong_size.txt"))
        wrong_raises = False
    except RuntimeError:
        wrong_raises = True
    ok = same and same_los and flags and cleared and dropped and params_ok and wrong_raises
    log(f"  {'PASS' if ok else 'FAIL'} lpt.Camera handle: setCorrection = the wrapper (project {same}, lineOfSight "
        f"{same_los}); flags {flags}; clearCorrection {cleared}; loadParameters drops it {dropped}; pinhole parameter "
        f"call with a correction {params_ok}; a wrong size raises {wrong_raises}")
    return {"ok": bool(ok), "project": same, "lineOfSight": same_los, "flags": flags, "clear": cleared,
            "load_drops": dropped, "params": params_ok, "wrong_size_raises": wrong_raises}


def main():
    lpt = vc.lib("vcf")
    log = vc.Log("l1_wrapper")
    os.makedirs(OUT, exist_ok=True)
    log(f"L1 on {lpt.__file__}")
    vol = vc.load_json(os.path.join(vc.CASE_DATA, "t1_view_volume.json"))

    # the sphere: T1's cam1 relabelled
    sph = os.path.join(OUT, "cam1_PINSPH.txt")
    with open(vc.CAMS_T1[0], encoding="utf-8") as f:
        text = f.read()
    with open(sph, "w", encoding="utf-8") as f:
        f.write(text.replace("\nPINCYL\n", "\nPINSPH\n", 1))
    wide = os.path.join(OUT, "wide_PINCYL.txt")
    write_wide_camera(wide)
    cams = {"T1cam1": vc.CAMS_T1[0], "T1cam2": vc.CAMS_T1[1], "T1cam1_sphere": sph, "wide_cyl": wide,
            "pinhole": vc.CAM_PINHOLE, "pinplate": vc.CAM_PINPLATE}

    # the fields per camera: zero, smooth analytic, (T1) the two-pass field
    corr_dir = os.path.join(OUT, "fields")
    fields = {}
    out = {"fields": [], "cameras": [], "edge": []}
    for name, file in cams.items():
        inner = lpt.loadCameraModel(file)
        n_col, n_row = inner.getNCol(), inner.getNRow()
        zero = vc.Correction.zeros(n_col, n_row)

        def smooth(uv, W=n_col, H=n_row):
            return np.column_stack([2.2 * np.sin(2 * np.pi * 1.3 * uv[:, 0] / W + 0.4) * np.cos(np.pi * uv[:, 1] / H)
                                    + 0.6, -1.7 * np.cos(2 * np.pi * 0.9 * uv[:, 1] / H - 0.2)
                                    + 0.9 * np.sin(np.pi * uv[:, 0] / W)])
        sm = vc.Correction.from_function(n_col, n_row, smooth)
        fl = {"zero": zero, "smooth": sm}
        if name in ("T1cam1", "T1cam2"):
            k = 0 if name == "T1cam1" else 1
            fl["t1_vc2_total"], dev = vc.selfcal_correction("t1_vc2_total", k)
            log(f"  {name}: the two-pass field as a correction (8 px nodes): {dev['field_rms_px']:.2f} px RMS, "
                f"{dev['field_max_px']:.2f} px max; the bilinear read departs from the dense map by "
                f"{dev['rms_dev_px']:.4f} px RMS, {dev['max_dev_px']:.4f} px max; slope bound {dev['slope_bound']:.4f}")
            out.setdefault("t1_field_dev", {})[name] = dev
        fields[name] = {}
        for tag, c in fl.items():
            p = os.path.join(corr_dir, f"{tag}_{name}.txt")
            c.write(p, [f"L1: {tag} field of {name}"])
            fields[name][tag] = (p, vc.Correction.read(p))
        if name in ("T1cam1", "T1cam2"):   # config.json's test reads these
            shutil.copy2(fields[name]["smooth"][0], os.path.join(corr_dir, f"smooth_cam{1 if name == 'T1cam1' else 2}.txt"))

    log("The fields in C++ against the Python reference:")
    for name in cams:
        for tag, (p, ref) in fields[name].items():
            if tag != "zero" or name == "T1cam1":
                out["fields"].append(check_field_binding(lpt, log, ref, p))

    log("The cameras:")
    for name, file in cams.items():
        inner = lpt.loadCameraModel(file)
        if name == "wide_cyl":
            sets = points_wide(lpt, inner)
        elif inner.multiValued():
            sets = points_curved(lpt, inner, vol)
            if name.endswith("sphere"):
                sets["axial"] = points_axial(lpt, inner, file)
        elif name == "pinhole":
            sets = points_ray(lpt, inner, lambda p, d: float(np.dot(-p, d)), 30.0)
        else:
            sets = points_ray(lpt, inner, lambda p, d: 60.0, 55.0)
        log(f" {name} ({inner.type()}): point sets " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
        X0 = sets.get("volume", sets.get("rays", sets.get("tube")))
        tracks = tracks_from(X0)
        for tag, (p, ref) in fields[name].items():
            out["cameras"].append(check_camera(lpt, log, name, file, p, ref, sets, tracks, zero_field=(tag == "zero")))
        if name == "wide_cyl":
            depth = lambda p, d: RNG.uniform(5.0, 120.0)  # noqa: E731
        elif inner.multiValued():
            depth = lambda p, d: RNG.uniform(5.0, 150.0)  # noqa: E731
        elif name == "pinhole":
            depth = lambda p, d: float(np.dot(-p, d)) + RNG.uniform(-30.0, 30.0)  # noqa: E731
        else:
            depth = lambda p, d: RNG.uniform(5.0, 115.0)  # noqa: E731
        axis = 0
        pts = edge_points(lpt, inner, depth, 0)
        if len(pts) < 50:   # the wide camera's right edge looks past the tube: its bottom edge then
            axis = 1
            pts = edge_points(lpt, inner, depth, 1)
        out["edge"].append(check_edge_shift(lpt, log, name, file, pts, axis))

    log("config.json and the Python handle:")
    out["config"] = check_config(lpt, log, corr_dir)
    out["handle"] = check_handle(lpt, log, corr_dir)

    fails = [c for c in out["cameras"] if c["fails"]] + [f for f in out["fields"] if not f["ok"]] + \
        [e for e in out["edge"] if not e["ok"]]
    ok = not fails and out["config"]["ok"] and out["handle"]["ok"]
    out["pass"] = bool(ok)
    log(f"L1: {'PASS' if ok else 'FAIL'} ({len(out['cameras'])} camera/field pairs, {len(out['fields'])} fields, "
        f"{len(out['edge'])} edge tests, config, handle)")
    log.save(out)


if __name__ == "__main__":
    main()
