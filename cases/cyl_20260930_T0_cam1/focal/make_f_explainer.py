"""An explainer page for a labmate: why cam1's camera-model focal length is about 41,600 px with a 180 mm lens on
20 um pixels (asked 2026-10-05).

Every number on the page comes from this run:
  * the 15 images of the board in air (X:\\...\\Raw\\20260930\\T0\\focal, taken 5 Oct with the cylinder removed, at
    random tilts, depths and rotations): dots detected with the case's detector, indexed on the board's fiducial L,
    then OpenCV's calibrateCamera with f the only lens parameter (principal point at the image centre, no
    distortion).  From it: f, the fit error against a forced f, the end points from four starting guesses, f with
    each image left out, and the magnification and distance of the square-on images;
  * the cylinder calibration's result file (X:\\...\\Processed\\20260930\\T0\\cam1\\calibration_result.json): f and its
    sigma, the vertical dot spacing of the 20 planes, the straight line through it, the camera's distance from the
    tube axis, and the photo's distances (lens front and camera body to the tube wall).
The one quoted number is f with a bowed board allowed, about 40,200 px, from the scratch analysis in
2026-10-05-plan.md, section 2.

    python make_f_explainer.py                # writes X:\\...\\Processed\\20260930\\T0\\focal\\focal_length_explained.html
    python make_f_explainer.py --fragment F   # also writes F: the same page without the document skeleton
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case                                   # noqa: E402
import dots as D                              # noqa: E402

RAW = r"X:\Shijie Zhong\Refraction\Raw\20260930\T0\focal"
OUT_DIR = r"X:\Shijie Zhong\Refraction\Processed\20260930\T0\focal"
TUBE_RESULT = r"X:\Shijie Zhong\Refraction\Processed\20260930\T0\cam1\calibration_result.json"
OUT_NAME = "focal_length_explained.html"
N_IMAGES = 15
W, H = 1280, 800
PITCH = 3.0              # mm, dot pitch of the board
PIXEL = 0.020            # mm
F_LENS = 180.0           # mm, the lens's focal length (focused at infinity)
FRONTAL_DEG = 5.0        # images tilted less than this (7 and 14) give the magnification and distance quoted;
                         # tilted ones foreshorten the dot spacing in one direction
BOW_F_QUOTED = 40200     # px, from 2026-10-05-plan.md section 2 (bow fitted); the only number not computed here
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


# ------------------------------------------------------------------ detection and indexing
def normalised_darkness_median(img, det):
    """dots.normalised_darkness with a 151 px median as the first background instead of a closing.  The board in
    these images is grainy; the closing (a local maximum) sits about 12 % above the board and the whole board then
    segments as one blob.  The rest (masking the dots, normalised convolution) is unchanged."""
    im8 = np.clip(img, 0, 255).astype(np.uint8)
    bg0 = cv2.GaussianBlur(cv2.medianBlur(im8, 151).astype(np.float32), (0, 0), 15.0)
    n0 = (bg0 - img) / np.maximum(bg0, 1.0)
    core0 = float(np.median(n0[n0 > 0.25])) if np.any(n0 > 0.25) else 0.4
    mask = (n0 > 0.25 * core0).astype(np.uint8)
    d = int(det["mask_dilate_px"])
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    w = (1 - mask).astype(np.float32)
    s = float(det["background_sigma_px"])
    num = cv2.GaussianBlur(img * w, (0, 0), s)
    den = cv2.GaussianBlur(w, (0, 0), s)
    bg = np.where(den > 0.05, num / np.maximum(den, 1e-6), bg0).astype(np.float32)
    N = (bg - img) / np.maximum(bg, 1.0)
    core = float(np.median(N[N > 0.25])) if np.any(N > 0.25) else core0
    return N.astype(np.float32), bg, float(det["seg_fraction"]) * core


def _homography(ij, P):
    Hm, _ = cv2.findHomography(ij.astype(np.float64), P.astype(np.float64), 0)
    return Hm


def _apply(Hm, ij):
    q = np.column_stack([ij, np.ones(len(ij))]) @ Hm.T
    return q[:, :2] / q[:, 2:3]


def _inverse(Hm, P):
    q = np.column_stack([P, np.ones(len(P))]) @ np.linalg.inv(Hm).T
    return q[:, :2] / q[:, 2:3]


def index_on_fiducial(ds, a_ref):
    """(P, ij) of the usable regular dots, indexed from the fiducial L.

    The L is one large dot (about 2.8 times a regular dot's area) at the corner and two medium ones (about 1.95
    times) at the neighbouring nodes; a medium one cut by the border still shows more area than a regular dot.
    The arm A (i direction) and arm B (jd direction, image-down in image 1) are told apart by the L's handedness,
    cross(A, B) > 0 in image coordinates.  Dots are then assigned by an affine start from the L and homography
    refits until the assignment no longer changes."""
    fid = [d for d in ds if d["area"] > 1.5 * a_ref or (d["kind"] == "border" and d["area"] > 1.08 * a_ref)]
    if len(fid) != 3:
        raise RuntimeError(f"fiducial: {len(fid)} candidate dots, expected 3")
    fid.sort(key=lambda d: -d["area"])
    xy = [np.array([d["u"], d["v"]]) if d["ok"] else np.array([d["u_rough"], d["v_rough"]]) for d in fid]
    C, v1, v2 = xy[0], xy[1] - xy[0], xy[2] - xy[0]
    a, b = (v1, v2) if v1[0] * v2[1] - v1[1] * v2[0] > 0 else (v2, v1)
    ok = [d for d in ds if d["ok"] and d["area"] < 1.5 * a_ref]
    P = np.array([[d["u"], d["v"]] for d in ok], float)
    ij_L = np.array([[0, 0], [1, 0], [0, 1]], float)
    P_L = np.array([C, C + a, C + b])
    ij = np.rint(np.linalg.solve(np.column_stack([a, b]), (P - C).T).T)
    near = np.hypot(*(P - C).T) < 3.5 * np.linalg.norm(a)
    for it in range(10):
        use = near if it == 0 else np.ones(len(P), bool)
        Hm = _homography(np.vstack([ij[use], ij_L]), np.vstack([P[use], P_L]))
        new = np.rint(_inverse(Hm, P))
        if it > 0 and np.array_equal(new, ij):
            break
        ij = new
    Hm = _homography(ij, P)
    res = np.hypot(*(_apply(Hm, ij) - P).T)
    pitch_px = float(np.linalg.norm(a))
    keys = [tuple(v) for v in ij.astype(int)]
    if len(keys) != len(set(keys)):
        raise RuntimeError("fiducial indexing: duplicate indices")
    if res.max() > 0.25 * pitch_px:
        raise RuntimeError(f"fiducial indexing: lattice residual {res.max():.1f} px")
    return P, ij.astype(int), float(res.max())


def collect():
    det = dict(case.load_config()["detect"])
    D.normalised_darkness = normalised_darkness_median          # these images only; dots.py is unchanged
    images, data = [], []
    for k in range(1, N_IMAGES + 1):
        img = case.read_image(os.path.join(RAW, f"{k}.tif"))
        r = D.detect(img, det)
        P, ij, res = index_on_fiducial(r["dots"], r["a_ref"])
        data.append({"k": k, "P": P, "ij": ij, "lattice_res": res})
        images.append(img)
    return images, data


# ------------------------------------------------------------------ the calibration (OpenCV, f only)
FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
         cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_PRINCIPAL_POINT)
CRIT = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 500, 1e-12)


def views(data, skip=None):
    objp = [np.column_stack([PITCH * d["ij"][:, 0], -PITCH * d["ij"][:, 1], np.zeros(len(d["P"]))]).astype(np.float32)
            for d in data if d["k"] != skip]
    imgp = [d["P"].astype(np.float32).reshape(-1, 1, 2) for d in data if d["k"] != skip]
    return objp, imgp


def calibrate(objp, imgp, f0, hold_f=False):
    K = np.array([[f0, 0, (W - 1) / 2], [0, f0, (H - 1) / 2], [0, 0, 1]], float)
    flags = FLAGS | (cv2.CALIB_FIX_FOCAL_LENGTH if hold_f else 0)
    rms, K, _d, rv, tv, sdi, _sde, pve = cv2.calibrateCameraExtended(objp, imgp, (W, H), K, np.zeros(5),
                                                                     flags=flags, criteria=CRIT)
    return {"rms": float(rms), "f": float(K[1, 1]), "sf": float(sdi[1, 0]), "rv": rv, "tv": tv,
            "per_image": np.asarray(pve, float).ravel()}


def air_numbers(data):
    objp, imgp = views(data)
    best = calibrate(objp, imgp, 41500.0)
    starts = {f0: calibrate(objp, imgp, f0)["f"] for f0 in (9000.0, 16000.0, 41500.0, 80000.0)}
    loo = [calibrate(*views(data, skip=d["k"]), best["f"])["f"] for d in data]
    poses = []
    for d, o, r_, t_ in zip(data, objp, best["rv"], best["tv"]):
        R, _ = cv2.Rodrigues(r_)
        Pc = o.astype(float) @ R.T + t_.ravel()
        tilt = math.degrees(math.acos(min(1.0, abs(R[2, 2]))))
        # median nearest-neighbour spacing of the detected dots, px
        dd = np.hypot(*(d["P"][:, None, :] - d["P"][None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(dd, np.inf)
        poses.append({"k": d["k"], "tilt_deg": tilt, "depth_mm": float(np.mean(Pc[:, 2])),
                      "pitch_px": float(np.median(dd.min(axis=1))), "n": len(d["P"])})
    frontal = [p for p in poses if p["tilt_deg"] < FRONTAL_DEG]
    m = float(np.mean([best["f"] * PIXEL / p["depth_mm"] for p in frontal]))
    Z_E = float(np.mean([p["depth_mm"] for p in frontal]))
    return {"best": best, "starts": starts, "loo": loo, "poses": poses, "frontal": frontal, "m": m, "Z_E": Z_E,
            "n_obs": int(sum(len(d["P"]) for d in data)), "n_dots": len({tuple(v) for d in data for v in d["ij"]}),
            "lattice_res_max": max(d["lattice_res"] for d in data)}


def rms_scan(data, f_values):
    """Fit error (RMS per dot, px) with f held.  With the principal point fixed and no distortion the images
    decouple, so each pose is solved alone: IPPE's two planar candidates, each refined by Levenberg-Marquardt, the
    better kept.  calibrateCamera with f fixed can stop in a poorer pose far from the best f (the curve then jumps
    between 7,000 and 9,000 px)."""
    objp, imgp = views(data)
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    out = {}
    for f in f_values:
        K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], float)
        sse, n = 0.0, 0
        for o, i in zip(objp, imgp):
            _ok, rvs, tvs, _e = cv2.solvePnPGeneric(o, i, K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
            best = None
            for rv, tv in zip(rvs, tvs):
                rv, tv = cv2.solvePnPRefineLM(o, i, K, np.zeros(5), rv.copy(), tv.copy(), criteria=crit)
                proj, _j = cv2.projectPoints(o, rv, tv, K, np.zeros(5))
                e = float(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2))
                best = e if best is None else min(best, e)
            sse += best
            n += len(o)
        out[float(f)] = math.sqrt(sse / n)
    return out


# ------------------------------------------------------------------ the cylinder calibration's numbers
def tube_numbers():
    with open(TUBE_RESULT, encoding="utf-8") as fh:
        r = json.load(fh)
    n_air, n_acr, n_w = r["port"]["n"]
    R1, R2 = r["port"]["R"]
    sc = r["spacing_check"]
    z = np.asarray(r["planes_fitted_z_mm"], float)
    s = np.asarray(sc["vertical_spacing_px"], float)
    axis_z = float(r["rig"]["axis_point_at_Y0_mm"][2]) - z[0]
    line = sc["line_through_1_over_vertical_spacing"]
    setup = r["setup"]

    def through_walls(L):
        """Air-equivalent distance from a pinhole L mm from the tube wall to plane 0 (as figures.pinhole_growth)."""
        return L + (R1 - R2) * n_air / n_acr + (R2 + axis_z) * n_air / n_w

    def growth(D0):
        return lambda zz: D0 / (D0 - (np.asarray(zz, float) - z[0]) * n_air / n_w) - 1.0

    curves = {"fit": growth(line["D0_mm"]), "body": growth(through_walls(setup["camera_body_front_to_tube_wall_mm"])),
              "lens": growth(through_walls(setup["lens_front_to_tube_wall_mm"]))}
    # the same growths as the result file's own check
    assert abs(curves["lens"](z[-1:])[0] - sc["vertical_growth_if_pinhole_at_lens_front"]) < 1e-9
    assert abs(curves["body"](z[-1:])[0] - sc["vertical_growth_if_pinhole_at_camera_body_front"]) < 1e-9
    cam_axis = float(r["rig"]["camera_distance_from_axis_mm"])
    return {"f": float(r["intrinsics"]["f_px"]), "sf": float(r["sigma"]["f_px"]), "z": z, "s": s,
            "g_meas": s / s[0] - 1.0, "curves": curves, "line_f": float(line["f_px"]), "D0": float(line["D0_mm"]),
            "growth_meas": float(sc["vertical_growth_measured"]), "traverse_mm": float(z[-1] - z[0]),
            "traverse_air_mm": float((z[-1] - z[0]) * n_air / n_w), "n_w": n_w,
            "cam_axis": cam_axis, "R1": R1,
            "pupil_behind_body": cam_axis - (R1 + setup["camera_body_front_to_tube_wall_mm"]),
            "pupil_behind_lens_front": cam_axis - (R1 + setup["lens_front_to_tube_wall_mm"]),
            "growth_lens": float(sc["vertical_growth_if_pinhole_at_lens_front"]),
            "growth_body": float(sc["vertical_growth_if_pinhole_at_camera_body_front"])}


# ------------------------------------------------------------------ SVG pieces
def t(x, y, s, cls="lbl", anchor="start", size=12, weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" font-size="{size}"{w}>'
            f"{html.escape(s)}</text>")


def chart_tube(T):
    """Growth of the vertical dot spacing over the traverse: measured, the line through it, and a pinhole at the
    camera body or the lens front."""
    Wv, Hv = 720, 352
    x0, x1, y0, y1 = 60, 492, 16, 292
    zmax, gmax = 50.0, 16.0
    X = lambda z: x0 + (x1 - x0) * z / zmax                      # noqa: E731
    Y = lambda g: y1 - (y1 - y0) * g / gmax                      # noqa: E731
    o = [f'<svg viewBox="0 0 {Wv} {Hv}" class="chart" role="img" aria-labelledby="ca-t">',
         '<title id="ca-t">Growth of the dot spacing as the board moves towards the camera, measured and for three '
         'pinhole positions</title>']
    for g in (0, 5, 10, 15):
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="{"axis" if g == 0 else "grid"}"/>')
        o.append(t(x0 - 8, Y(g) + 4, f"{g}", "tick", "end"))
    for z in (0, 10, 20, 30, 40, 50):
        o.append(f'<line x1="{X(z):.1f}" x2="{X(z):.1f}" y1="{y1}" y2="{y1 + 5}" class="axis"/>')
        o.append(t(X(z), y1 + 20, f"{z}", "tick", "middle"))
    o.append(t((x0 + x1) / 2, y1 + 44, "board position z (mm)", "lbl", "middle"))
    o.append(f'<text transform="translate(16 {(y0 + y1) / 2:.1f}) rotate(-90)" class="lbl" text-anchor="middle" '
             f'font-size="12">spacing growth since z = 0 (%)</text>')
    zz = np.linspace(T["z"][0], T["z"][-1], 120)
    styles = {"lens": ("s3", "8 5"), "body": ("s2", "3 4"), "fit": ("s1", None)}
    for key in ("lens", "body", "fit"):
        cls, dash = styles[key]
        g = 100 * T["curves"][key](zz)
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(zz, g))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<polyline points="{pts}" class="line {cls}" fill="none"{da}/>')
    for z, gm, s in zip(T["z"], 100 * T["g_meas"], T["s"]):
        o.append(f'<circle cx="{X(z):.1f}" cy="{Y(gm):.1f}" r="4.5" class="pt"><title>z = {z:.2f} mm: spacing '
                 f'{s:.2f} px, {gm:+.2f} %</title></circle>')
    labels = (("lens", f"{100 * T['growth_lens']:+.1f} %", "pinhole at the lens front"),
              ("body", f"{100 * T['growth_body']:+.1f} %", "pinhole at the camera body"),
              ("fit", f"{100 * T['growth_meas']:+.1f} %", f"measured: pinhole {T['D0'] / 1000:.2f} m away"))
    for key, val, txt in labels:
        cls, dash = styles[key]
        y = Y(100 * T["curves"][key](T["z"][-1:])[0])
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<line x1="{x1 + 8}" x2="{x1 + 30}" y1="{y:.1f}" y2="{y:.1f}" class="line {cls}"{da}/>')
        o.append(t(x1 + 38, y - 2, val, "lblb", "start", 13, 600))
        o.append(t(x1 + 38, y + 14, txt, "lbl", "start", 12))
    o.append("</svg>")
    return "\n".join(o)


def chart_air(scan, marks):
    """Fit error of the 15 air images against a forced f (log axis)."""
    Wv, Hv = 720, 352
    x0, x1, y0, y1 = 60, 684, 16, 292                             # room on the right for the "100,000" tick label
    lo, hi = math.log10(5000), math.log10(100000)
    ymax = 6
    X = lambda f: x0 + (x1 - x0) * (math.log10(f) - lo) / (hi - lo)    # noqa: E731
    Y = lambda r: y1 - (y1 - y0) * r / ymax                            # noqa: E731
    o = [f'<svg viewBox="0 0 {Wv} {Hv}" class="chart" role="img" aria-labelledby="cb-t">',
         '<title id="cb-t">Fit error of the 15 air images when f is held at each value</title>']
    for r in range(0, ymax + 1):
        o.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(r):.1f}" y2="{Y(r):.1f}" class="{"axis" if r == 0 else "grid"}"/>')
        o.append(t(x0 - 8, Y(r) + 4, f"{r}", "tick", "end"))
    for f in (5000, 10000, 20000, 50000, 100000):
        o.append(f'<line x1="{X(f):.1f}" x2="{X(f):.1f}" y1="{y1}" y2="{y1 + 5}" class="axis"/>')
        o.append(t(X(f), y1 + 20, f"{f:,}", "tick", "middle"))
    o.append(t((x0 + x1) / 2, y1 + 44, "f held at (px, log scale)", "lbl", "middle"))
    o.append(f'<text transform="translate(16 {(y0 + y1) / 2:.1f}) rotate(-90)" class="lbl" text-anchor="middle" '
             f'font-size="12">fit error, RMS (px)</text>')
    fs = sorted(scan)
    o.append(f'<clipPath id="cb-clip"><rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0 + 6}"/></clipPath>')
    pts = " ".join(f"{X(f):.1f},{Y(scan[f]):.1f}" for f in fs)
    o.append(f'<polyline points="{pts}" class="line s1" fill="none" clip-path="url(#cb-clip)"/>')
    for f in fs:
        if scan[f] <= ymax:
            o.append(f'<circle cx="{X(f):.1f}" cy="{Y(scan[f]):.1f}" r="7" class="hit"><title>f = {f:,.0f} px: '
                     f'{scan[f]:.2f} px RMS</title></circle>')
    # the cylinder calibration as a reference line; its label and the best fit's sit at its top left, where the
    # curve never goes (the curve's minimum is at the line's foot)
    fb, rb = marks["best"], scan[marks["best"]]
    xt = X(marks["tube"])
    o.append(f'<line x1="{xt:.1f}" x2="{xt:.1f}" y1="{y1}" y2="{Y(ymax) + 4:.1f}" class="ref" stroke-dasharray="4 4"/>')
    # right of the line, in the upper half: the curve stays below 1 px there
    o.append(t(xt + 10, Y(4.6), "cylinder calibration", "lbl", "start", 12))
    o.append(t(xt + 10, Y(4.6) + 17, f"{marks['tube']:,.0f} px", "lblb", "start", 13, 600))
    o.append(t(xt + 10, Y(4.6) + 42, "best fit to the air images", "lbl", "start", 12))
    o.append(t(xt + 10, Y(4.6) + 59, f"{fb:,.0f} px, {rb:.2f} px RMS", "lblb", "start", 13, 600))
    o.append(f'<circle cx="{X(fb):.1f}" cy="{Y(rb):.1f}" r="5" class="mk"/>')
    # 180 mm at infinity: label above-right of its point (the curve falls away below it)
    f = marks["inf"]
    r = scan[f]
    o.append(f'<circle cx="{X(f):.1f}" cy="{Y(r):.1f}" r="5" class="mk"/>')
    o.append(t(X(f) + 10, Y(r) - 22, f"{f:,.0f} px: {r:.2f} px", "lblb", "start", 13, 600))
    o.append(t(X(f) + 10, Y(r) - 6, "the lens's 180 mm at infinity focus", "lbl", "start", 12))
    # the thin lens: label below-left of its point (the curve is above there)
    f = marks["thin"]
    r = scan[f]
    o.append(f'<circle cx="{X(f):.1f}" cy="{Y(r):.1f}" r="5" class="mk"/>')
    o.append(t(X(f) - 12, Y(r) + 30, f"{f:,.0f} px: {r:.2f} px", "lblb", "end", 13, 600))
    o.append(t(X(f) - 12, Y(r) + 46, "thin 180 mm lens, same magnification", "lbl", "end", 12))
    o.append("</svg>")
    return "\n".join(o)


def diagram(m, Z_E, f_thin_mm, f_mm, so_thin, pupil_body, pupil_lens):
    """The equivalent pinhole cameras along the optical axis, to scale in distance (heights exaggerated)."""
    Wv, Hv = 760, 458
    x0, x1 = 40, 740
    Lmax = 2000.0
    X = lambda d: x0 + (x1 - x0) * d / Lmax                      # noqa: E731
    vs = 2.0                                                      # px per mm vertically
    half_board = 0.5 * W * PIXEL / m                              # mm on the board across the sensor width
    half_sensor = 0.5 * W * PIXEL                                 # mm
    o = [f'<svg viewBox="0 0 {Wv} {Hv}" class="chart diagram" role="img" aria-labelledby="dg-t">',
         '<title id="dg-t">The real camera and two equivalent pinhole cameras, drawn along the optical axis</title>']
    rows = [(78, "real"), (216, "thin"), (354, "this")]
    for yc, kind in rows:
        o.append(f'<line x1="{X(0):.1f}" x2="{x1}" y1="{yc}" y2="{yc}" class="axisline"/>')
        o.append(f'<rect x="{X(0) - 3:.1f}" y="{yc - half_board * vs:.1f}" width="6" height="{2 * half_board * vs:.1f}" '
                 f'class="board"/>')
    # row 1: the hardware (positions from the photo, the pupil from the cylinder calibration)
    yc = rows[0][0]
    lens_front, body_front = Z_E - pupil_lens, Z_E - pupil_body
    o.append(t(X(0), yc - 52, "the real camera (positions approximate)", "lblb", "start", 13, 600))
    o.append(f'<rect x="{X(lens_front):.1f}" y="{yc - 16}" width="{X(body_front) - X(lens_front):.1f}" height="32" '
             f'rx="4" class="hw"/>')
    o.append(f'<rect x="{X(body_front):.1f}" y="{yc - 26}" width="{X(body_front + 160) - X(body_front):.1f}" '
             f'height="52" rx="3" class="hw"/>')
    o.append(t((X(lens_front) + X(body_front)) / 2, yc + 4, "lens", "lbl", "middle", 11))
    o.append(t((X(body_front) + X(body_front + 160)) / 2, yc + 4, "camera", "lbl", "middle", 11))
    o.append(f'<circle cx="{X(Z_E):.1f}" cy="{yc}" r="6" class="pupil"/>')
    o.append(t(X(Z_E) + 12, yc - 12, "entrance pupil = the model's pinhole", "lblb", "start", 12, 600))
    o.append(t(X(Z_E) + 12, yc + 4, f"{pupil_body / 1000:.2f} m behind the camera's front,", "lbl", "start", 12))
    o.append(t(X(Z_E) + 12, yc + 20, "a virtual point the rays aim at", "lbl", "start", 12))
    # rows 2 and 3: pinhole cameras with the same magnification
    for (yc, kind), pin, fmm, cls, title, flabel in (
            (rows[1], so_thin, f_thin_mm, "s2", "thin 180 mm lens, same magnification",
             f"f = {f_thin_mm:.0f} mm = {f_thin_mm / PIXEL:,.0f} px"),
            (rows[2], Z_E, f_mm, "s1", "this lens, as calibrated",
             f"f = {f_mm:.0f} mm = {f_mm / PIXEL:,.0f} px")):
        sensor = pin + fmm
        o.append(t(X(0), yc - 52, title, "lblb", "start", 13, 600))
        for sgn in (-1, 1):
            o.append(f'<polyline points="{X(0):.1f},{yc + sgn * half_board * vs:.1f} {X(pin):.1f},{yc} '
                     f'{X(sensor):.1f},{yc - sgn * half_sensor * vs:.1f}" class="ray {cls}" fill="none"/>')
        o.append(f'<rect x="{X(sensor) - 2:.1f}" y="{yc - half_sensor * vs:.1f}" width="5" '
                 f'height="{2 * half_sensor * vs:.1f}" class="sensor"/>')
        o.append(f'<circle cx="{X(pin):.1f}" cy="{yc}" r="4.5" class="pinhole"/>')
        yd = yc + 42                                              # dimension lines, below the ray fan
        for a_, b_ in ((X(0), X(pin) - 3), (X(pin) + 3, X(sensor))):
            o.append(f'<line x1="{a_:.1f}" x2="{b_:.1f}" y1="{yd}" y2="{yd}" class="dim"/>')
            for e in (a_, b_):
                o.append(f'<line x1="{e:.1f}" x2="{e:.1f}" y1="{yd - 4}" y2="{yd + 4}" class="dim"/>')
        o.append(t(X(pin / 2), yd + 16, f"{pin / 1000:.2f} m", "lbl", "middle", 12))
        o.append(t((X(pin) + X(sensor)) / 2, yd + 16, flabel, "lblb", "middle", 12, 600))
        o.append(t(X(sensor) + 8, yc + 4, "sensor", "lbl", "start", 11))
    o.append(f'<line x1="{X(Z_E):.1f}" x2="{X(Z_E):.1f}" y1="{rows[0][0] + 8}" y2="{rows[2][0] - 6}" class="guide" '
             f'stroke-dasharray="2 4"/>')
    ya = Hv - 30
    for dm in (0, 0.5, 1.0, 1.5, 2.0):
        o.append(f'<line x1="{X(1000 * dm):.1f}" x2="{X(1000 * dm):.1f}" y1="{ya - 4}" y2="{ya + 4}" class="axis"/>')
        o.append(t(X(1000 * dm), ya + 20, f"{dm:g} m", "tick", "middle"))
    o.append(f'<line x1="{X(0):.1f}" x2="{x1}" y1="{ya}" y2="{ya}" class="axis"/>')
    o.append(t(X(0) + 8, ya - 8, "board", "lbl", "start", 11))
    o.append("</svg>")
    return "\n".join(o)


def montage(images):
    tiles = []
    for k, im in enumerate(images, start=1):
        lo, hi = np.percentile(im, [0.5, 99.5])
        g = np.clip((im - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
        g = cv2.cvtColor(cv2.resize(g, (256, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(g, (0, 0), (40, 26), (255, 255, 255), -1)
        cv2.putText(g, str(k), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        tiles.append(g)
    gap = np.full((160, 4, 3), 255, np.uint8)
    rows = []
    for r in range(3):
        row = []
        for c in range(5):
            row += [tiles[5 * r + c]] + ([gap] if c < 4 else [])
        rows.append(np.hstack(row))
    hgap = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
    m_img = np.vstack([rows[0], hgap, rows[1], hgap, rows[2]])
    ok, buf = cv2.imencode(".jpg", m_img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"), m_img.shape


# ------------------------------------------------------------------ the page
STYLE = """
/* Layout: one reading column (about 68ch) with figures a little wider; the answer and its formula first. */
:root {
  --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --rule: rgba(11, 11, 11, 0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --wash: rgba(42, 120, 214, 0.07); --hw: #ecebe6;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-body: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --font-ui: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --rule: rgba(255, 255, 255, 0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --wash: rgba(57, 135, 229, 0.12); --hw: #262624;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --rule: rgba(255, 255, 255, 0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --wash: rgba(57, 135, 229, 0.12); --hw: #262624;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 17px;
  line-height: 1.6; }
main { max-width: 52rem; margin: 0 auto; padding-inline: 16px; padding-block: 40px 64px; display: grid; gap: 40px; }
section, header { display: grid; gap: 14px; min-width: 0; }
.prose { max-width: 68ch; }
.eyebrow { font-family: var(--font-ui); font-size: 14px; letter-spacing: 0.02em; color: var(--ink-2); margin: 0; }
h1 { font-family: var(--font-display); font-weight: 600; font-size: clamp(30px, 5vw, 42px); line-height: 1.12;
  margin: 0; text-wrap: balance; }
h2 { font-family: var(--font-display); font-weight: 600; font-size: 25px; line-height: 1.2; margin: 0;
  text-wrap: balance; }
h3 { font-family: var(--font-ui); font-weight: 600; font-size: 17px; margin: 0; }
p { margin: 0; }
.nowrap { white-space: nowrap; }
.lede { font-size: 19px; }
.formula { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 20px 22px;
  display: grid; gap: 14px; }
.formula .eq { font-family: var(--font-mono); font-size: clamp(22px, 4vw, 30px); color: var(--ink); }
.formula .eq var, .defs var { font-family: var(--font-body); font-style: italic; }
.formula .calc { font-family: var(--font-mono); font-size: 16px; color: var(--ink); background: var(--wash);
  padding: 8px 12px; border-radius: 4px; overflow-wrap: anywhere; }
.defs { display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 0; font-size: 16px; }
.defs dt { font-family: var(--font-mono); }
.defs dd { margin: 0; color: var(--ink-2); min-width: 0; }
ol.steps { margin: 0; padding-left: 1.4em; display: grid; gap: 10px; max-width: 68ch; }
ol.steps li::marker { font-family: var(--font-ui); font-weight: 600; color: var(--ink-2); }
figure { margin: 0; display: grid; gap: 10px; min-width: 0; }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 14px 8px; }
.scroll { overflow-x: auto; }
svg.chart { display: block; width: 100%; min-width: 560px; height: auto; font-family: var(--font-ui); }
figcaption { font-family: var(--font-ui); font-size: 14px; color: var(--ink-2); max-width: 72ch; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-family: var(--font-ui); font-size: 13px;
  color: var(--ink-2); margin: 0 0 4px; padding: 0; list-style: none; }
.legend li { display: flex; align-items: center; gap: 6px; }
.legend svg { width: 26px; height: 10px; flex: none; }
.grid { stroke: var(--grid); stroke-width: 1; fill: none; }
.axis { stroke: var(--axis); stroke-width: 1; fill: none; }
.axisline { stroke: var(--grid); stroke-width: 1; fill: none; }
.tick { fill: var(--muted); font-variant-numeric: tabular-nums; }
.lbl { fill: var(--ink-2); }
.lblb { fill: var(--ink); }
.line { stroke-width: 2; fill: none; }
.ray { stroke-width: 1.6; fill: none; }
.s1 { stroke: var(--s1); }
.s2 { stroke: var(--s2); }
.s3 { stroke: var(--s3); }
.pt { fill: var(--ink); stroke: var(--surface); stroke-width: 2; }
.mk { fill: var(--ink); stroke: var(--surface); stroke-width: 2; }
.hit { fill: transparent; stroke: none; }
.hit:hover { fill: var(--wash); }
.ref { stroke: var(--ink-2); stroke-width: 1.5; fill: none; }
.board { fill: var(--ink); }
.sensor { fill: var(--ink-2); }
.pinhole { fill: var(--surface); stroke: var(--ink); stroke-width: 2; }
.pupil { fill: var(--surface); stroke: var(--s1); stroke-width: 2.5; }
.hw { fill: var(--hw); stroke: var(--axis); stroke-width: 1; }
.guide { stroke: var(--s1); stroke-width: 1.2; fill: none; opacity: 0.7; }
.dim { stroke: var(--muted); stroke-width: 1; fill: none; }
table { border-collapse: collapse; font-family: var(--font-ui); font-size: 15px; width: 100%; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--ink-2); font-size: 13px; letter-spacing: 0.02em; }
td.num { text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.key td { font-weight: 600; }
details { font-family: var(--font-ui); font-size: 14px; color: var(--ink-2); }
details summary { cursor: pointer; }
details summary:focus-visible, a:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
details table { margin-top: 8px; font-size: 13px; }
ul.points { margin: 0; padding-left: 1.2em; display: grid; gap: 10px; max-width: 68ch; }
img.montage { width: 100%; height: auto; border-radius: 4px; display: block; }
a { color: var(--s1); }
footer { font-family: var(--font-ui); font-size: 13px; color: var(--muted); border-top: 1px solid var(--rule);
  padding-top: 16px; display: grid; gap: 6px; }
footer code { font-family: var(--font-mono); font-size: 12px; overflow-wrap: anywhere; }
@media (max-width: 480px) { body { font-size: 16px; } .lede { font-size: 17px; } }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
         '&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600'
         '&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">')


def legend_item(cls, dash, text, point=False):
    if point:
        sw = '<svg viewBox="0 0 26 10" aria-hidden="true"><circle cx="13" cy="5" r="4" class="pt"/></svg>'
    else:
        da = f' stroke-dasharray="{dash}"' if dash else ""
        sw = (f'<svg viewBox="0 0 26 10" aria-hidden="true"><line x1="1" x2="25" y1="5" y2="5" class="line {cls}"{da}/>'
              f"</svg>")
    return f"<li>{sw}{html.escape(text)}</li>"


def build_page(A, T, scan, marks, mont, derived):
    fb = A["best"]["f"]
    m, Z_E = A["m"], A["Z_E"]
    thin_px, thin_so = derived["thin_px"], derived["thin_so"]
    pitch_lo = min(p["pitch_px"] for p in A["frontal"])
    pitch_hi = max(p["pitch_px"] for p in A["frontal"])
    loo_lo, loo_hi = min(A["loo"]), max(A["loo"])
    starts = ", ".join(f"{f0:,.0f}" for f0 in A["starts"])
    agree = 100 * (fb / T["f"] - 1)
    rows_tbl = [
        ("Cylinder calibration (30 Sep): full model through the tube", f"{T['f']:,.0f} ± {T['sf']:.0f}", True),
        ("Cylinder images: a straight line through the dot spacing, no cylinder model", f"{T['line_f']:,.0f}", False),
        ("Air images (5 Oct): OpenCV calibrateCamera, f only", f"{fb:,.0f}", True),
        ("Air images, leaving out one image at a time", f"{10 * round(loo_lo / 10):,.0f} – {10 * round(loo_hi / 10):,.0f}",
         False),
        ("Air images, a slightly bowed board allowed in the fit", f"about {BOW_F_QUOTED:,.0f}", False),
        ("The lens's 180 mm, focused at infinity", f"{F_LENS / PIXEL:,.0f}", False),
    ]
    table_html = "".join(f'<tr class="{"key" if key else ""}"><td>{html.escape(a)}</td><td class="num">{b}</td></tr>'
                         for a, b, key in rows_tbl)
    tube_rows = "".join(f'<tr><td class="num">{z:.2f}</td><td class="num">{s:.2f}</td>'
                        f'<td class="num">{100 * g:+.2f}</td></tr>' for z, s, g in zip(T["z"], T["s"], T["g_meas"]))
    scan_rows = "".join(f'<tr><td class="num">{f:,.0f}</td><td class="num">{r:.3f}</td></tr>'
                        for f, r in sorted(scan.items()))
    pose_rows = "".join(f'<tr><td class="num">{p["k"]}</td><td class="num">{p["n"]}</td>'
                        f'<td class="num">{p["tilt_deg"]:.1f}</td><td class="num">{p["depth_mm"]:,.0f}</td></tr>'
                        for p in A["poses"])
    tilt_lo, tilt_hi = min(p["tilt_deg"] for p in A["poses"]), max(p["tilt_deg"] for p in A["poses"])
    dep_lo, dep_hi = min(p["depth_mm"] for p in A["poses"]), max(p["depth_mm"] for p in A["poses"])
    legend_html = "".join([legend_item("", None, "measured, 20 planes", point=True),
                           legend_item("s1", None, "line through the data"),
                           legend_item("s2", "3 4", "pinhole at the camera body"),
                           legend_item("s3", "8 5", "pinhole at the lens front")])
    demo_this = 100 * 10.0 / (Z_E - 10.0)
    demo_thin = 100 * 10.0 / (thin_so - 10.0)
    body = f"""
<main>
<header>
  <p class="eyebrow">cam1 · Phantom VEO710, 20 µm pixels · 180 mm macro lens · run 20260930</p>
  <h1>Why is f about <span class="nowrap">{round(fb, -2):,.0f} px</span> with a <span class="nowrap">180 mm</span>
  lens?</h1>
  <p class="lede prose">A 180 mm lens on 20 µm pixels suggests f = 180 / 0.020 = {F_LENS / PIXEL:,.0f} px. The cylinder
  calibration gave {T['f']:,.0f} px. Both numbers are right, about different things. 180 mm is the lens's focal length
  when it is focused at infinity. The f of a camera model is the distance from the model's pinhole to its sensor.
  Focused this close, the two are far apart.</p>
  <div class="formula">
    <div class="eq"><var>f</var> = <var>m</var> × <var>Z</var><sub>E</sub> / <var>p</var></div>
    <dl class="defs">
      <dt><var>m</var></dt><dd>magnification: {m:.3f}. In the {NUMBER_WORDS.get(len(A['frontal']), str(len(A['frontal'])))} most square-on images the 3 mm dot
      pitch spans {pitch_lo:.0f}–{pitch_hi:.0f} px on the sensor, {pitch_lo * PIXEL:.2f}–{pitch_hi * PIXEL:.2f} mm.</dd>
      <dt><var>Z</var><sub>E</sub></dt><dd>distance from the board to the lens's entrance pupil, which is the pinhole of
      the camera model: {Z_E:,.0f} mm in those images</dd>
      <dt><var>p</var></dt><dd>pixel size: 0.020 mm</dd>
    </dl>
    <div class="calc">{m:.3f} × {Z_E:,.0f} mm / 0.020 mm = {m * Z_E / PIXEL:,.0f} px</div>
  </div>
</header>

<section>
  <h2>Where the factor of {fb / (F_LENS / PIXEL):.1f} comes from</h2>
  <ol class="steps">
    <li><strong>Focused at infinity,</strong> the model's f is the lens's focal length:
    180 mm / 0.020 mm = {F_LENS / PIXEL:,.0f} px.</li>
    <li><strong>Focusing close puts the image farther behind the lens.</strong> At magnification {m:.2f}, a simple thin
    lens sits F(1 + m) = {F_LENS * (1 + m):.0f} mm in front of the sensor, so f = {thin_px:,.0f} px, {1 + m:.2f} times
    more. Its pinhole, the lens itself, would be {thin_so / 1000:.2f} m from the board.</li>
    <li><strong>This lens's pinhole is not at the lens.</strong> It is {Z_E / 1000:.2f} m from the board,
    {Z_E / thin_so:.1f} times farther than a thin lens's. The magnification is the same, so the distance from pinhole
    to sensor grows by the same factor: f = {m:.3f} × {Z_E:,.0f} mm = {m * Z_E:.0f} mm = {m * Z_E / PIXEL:,.0f} px.</li>
  </ol>
  <figure>
    <div class="card scroll">{derived['diagram']}</div>
    <figcaption>Both pinhole cameras image the same {2 * 0.5 * W * PIXEL / m:.0f} mm of board onto the same
    {W * PIXEL:.1f} mm of sensor, because the magnification is the same. What differs is where the pinhole sits, and
    so the distance from pinhole to sensor, which is f. Distances along the axis are to scale; heights are about
    {derived['exaggeration']:.0f} times larger.</figcaption>
  </figure>
</section>

<section>
  <h2>The pinhole is the lens's entrance pupil</h2>
  <div class="prose" style="display:grid;gap:12px">
  <p>A camera model follows the rays that pass through the centre of the lens's aperture. Seen from the object, these
  rays all aim at one point: the image of the aperture formed by the glass in front of it, called the entrance pupil.
  That point is the pinhole of the camera model, wherever it falls in the hardware.</p>
  <p>For most lenses focused far away, the entrance pupil lies inside the lens. This macro lens, focused at
  {m:.2f}×, places it about {T['pupil_behind_body'] / 1000:.2f} m behind the front of the camera body. It is a virtual
  point, which the rays aim at but never reach. The lens is then part way to being telecentric: a lens whose entrance
  pupil is at infinity shows no perspective at all, and its f is infinite.</p>
  <p>So perspective is weak here. Moving an object 10 mm closer enlarges its image by only {demo_this:.2f} %. With
  the pinhole at a thin lens it would be {demo_thin:.1f} %.</p>
  </div>
</section>

<section>
  <h2>Measured twice, two independent ways</h2>
  <figure>
    <h3>Through the cylinder (30 Sep): the dot spacing barely grows as the board approaches</h3>
    <ul class="legend">{legend_html}</ul>
    <div class="card scroll">{derived['chart_tube']}</div>
    <figcaption>The board moved {T['traverse_mm']:.2f} mm towards the camera in twenty 0.1 in steps. Along the tube's
    axis the wall acts like a flat window, so this direction needs no cylinder model, only the water's index. The
    vertical dot spacing grew by {100 * T['growth_meas']:.2f} %. A pinhole at the lens front would have made it grow
    {100 * T['growth_lens']:.1f} %, and one at the camera body {100 * T['growth_body']:.1f} % (distances from the photo
    of the rig). The line through the data puts the pinhole {T['D0']:,.0f} mm from the farthest plane, counted in air,
    and gives f = {T['line_f']:,.0f} px. The full model gives {T['f']:,.0f} px.</figcaption>
    <details><summary>Show the numbers</summary><div class="scroll"><table>
      <thead><tr><th>z (mm)</th><th>spacing (px)</th><th>growth (%)</th></tr></thead>
      <tbody>{tube_rows}</tbody></table></div></details>
  </figure>
  <figure>
    <h3>In air (5 Oct): only f ≈ {round(fb, -2):,.0f} px fits</h3>
    <div class="card scroll">{derived['chart_air']}</div>
    <figcaption>With the cylinder removed, 15 images of the board went through OpenCV's standard calibrateCamera,
    with f the only lens parameter (principal point at the image centre, no distortion). The board was
    {dep_lo:,.0f}–{dep_hi:,.0f} mm from the pinhole and tilted {tilt_lo:.0f}–{tilt_hi:.0f}° from square-on;
    {A['n_obs']:,} dot images of {A['n_dots']} distinct dots. The fit returns f = {fb:,.0f} px with a fit error of
    {A['best']['rms']:.2f} px from every starting guess tried ({starts} px). Holding f at the lens's
    {F_LENS / PIXEL:,.0f} px raises the error to {scan[marks['inf']]:.1f} px; holding it at the thin-lens
    {thin_px:,.0f} px, to {scan[marks['thin']]:.2f} px.</figcaption>
    <details><summary>Show the numbers</summary><div class="scroll"><table>
      <thead><tr><th>f held at (px)</th><th>fit error, RMS (px)</th></tr></thead>
      <tbody>{scan_rows}</tbody></table></div></details>
  </figure>
  <figure>
    <img class="montage" src="{mont}" alt="The 15 images of the dot board in air, at different tilts, distances and
    rotations">
    <figcaption>The 15 air images. The three large dots mark the board's origin, so every dot keeps its own name in
    every image.</figcaption>
    <details><summary>Show the poses</summary><div class="scroll"><table>
      <thead><tr><th>image</th><th>dots</th><th>tilt (°)</th><th>distance (mm)</th></tr></thead>
      <tbody>{pose_rows}</tbody></table></div></details>
  </figure>
</section>

<section>
  <h2>How close are the two values?</h2>
  <div class="scroll"><table>
    <thead><tr><th>measurement</th><th style="text-align:right">f (px)</th></tr></thead>
    <tbody>{table_html}</tbody></table></div>
  <div class="prose" style="display:grid;gap:12px">
  <p>The two best values differ by {abs(agree):.1f} %. Read that as "the same within a few per cent", for two
  reasons.</p>
  <ul class="points">
    <li><strong>The set-ups differ.</strong> The camera may have moved a little between the two days. Moving the camera
    does not change f, but refocusing does, a little.</li>
    <li><strong>Tilted boards at 1 m pin f only to a few per cent.</strong> Leaving out one image at a time moves the
    air value between {10 * round(loo_lo / 10):,.0f} and {10 * round(loo_hi / 10):,.0f} px. A board bowed by 5 µm over
    25 mm would shift it by about 1 %, and allowing such a bow in the fit gives about {BOW_F_QUOTED:,.0f} px.</li>
  </ul>
  <p>Either way f is about {round(fb, -2):,.0f} px, within a few per cent. It is not {F_LENS / PIXEL:,.0f} px,
  and not the {round(thin_px, -3):,.0f} px of a simple lens.</p>
  </div>
</section>

<section>
  <h2>What this means in practice</h2>
  <ul class="points">
    <li><strong>f belongs to this focus setting.</strong> Turning the focus ring moves the entrance pupil and changes
    f, so recalibrate after refocusing.</li>
    <li><strong>Do not take f from the number on the lens for close-up work.</strong> If a calibration asks for a
    starting focal length, use about {round(fb * PIXEL, -1):.0f} mm ({round(fb, -2):,.0f} px × 0.020 mm) for this
    lens at this focus, not 180 mm.</li>
    <li><strong>Give the calibration depth.</strong> With weak perspective, f trades off against the camera's distance.
    A traverse of several centimetres, as in the cylinder runs, separates them. Tilted boards do so less well.</li>
    <li><strong>A two-minute check.</strong> Move a square-on target 10 mm towards the camera and compare the dot
    spacing. Here it grows by about {demo_this:.2f} %; with the pinhole at a thin lens it would grow by about
    {demo_thin:.1f} %.</li>
  </ul>
</section>

<footer>
  <p>Made on {time.strftime('%Y-%m-%d')} by <code>pyOpenLPT/cases/cyl_20260930_T0_cam1/focal/make_f_explainer.py</code>.</p>
  <p>Air images: <code>{html.escape(RAW)}</code>. Cylinder calibration:
  <code>{html.escape(TUBE_RESULT)}</code> (report <code>cases/cyl_20260930_T0_cam1/2026-10-01-report.md</code>).
  The bowed-board value is from <code>focal/2026-10-05-plan.md</code>, section 2.</p>
</footer>
</main>
"""
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", default=None, help="also write the page without the document skeleton here")
    args = ap.parse_args()
    t0 = time.time()
    images, data = collect()
    A = air_numbers(data)
    T = tube_numbers()
    fb, m, Z_E = A["best"]["f"], A["m"], A["Z_E"]
    thin_px = F_LENS * (1 + m) / PIXEL
    thin_so = F_LENS * (1 + 1 / m)
    marks = {"inf": F_LENS / PIXEL, "thin": thin_px, "tube": T["f"], "best": fb}
    grid = list(np.geomspace(5000, 100000, 49)) + [marks["inf"], marks["thin"], marks["best"]]
    scan = rms_scan(data, grid)
    # check: at the best f the per-image poses reproduce the joint fit
    assert abs(scan[marks["best"]] - A["best"]["rms"]) < 1e-4, (scan[marks["best"]], A["best"]["rms"])
    derived = {"thin_px": thin_px, "thin_so": thin_so,
               "diagram": diagram(m, Z_E, F_LENS * (1 + m), m * Z_E, thin_so, T["pupil_behind_body"],
                                  T["pupil_behind_lens_front"]),
               "chart_tube": chart_tube(T), "chart_air": chart_air(scan, marks),
               "exaggeration": 2.0 / (700 / 2000.0)}                 # diagram: 2 px/mm up, 700 px per 2 m along
    mont, mshape = montage(images)
    body = build_page(A, T, scan, marks, mont, derived)
    title = f"<title>Why f is {round(fb, -2):,.0f} px</title>"
    head = f'{title}\n{FONTS}\n<style>{STYLE}</style>'
    page = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n{head}\n'
            f'</head>\n<body>\n{body}\n</body>\n</html>\n')
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, OUT_NAME)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    if args.fragment:
        with open(args.fragment, "w", encoding="utf-8") as fh:
            fh.write(f"{head}\n{body}\n")
    numbers = {"air_f_px": fb, "air_f_formal_sigma_px": A["best"]["sf"], "air_rms_px": A["best"]["rms"],
               "air_starts": {f"{k:.0f}": v for k, v in A["starts"].items()},
               "air_leave_one_out_px": A["loo"], "air_n_obs": A["n_obs"], "air_n_dots": A["n_dots"],
               "air_lattice_residual_max_px": A["lattice_res_max"],
               "air_poses": A["poses"], "m_frontal": m, "Z_E_frontal_mm": Z_E,
               "f_from_m_and_Z_px": m * Z_E / PIXEL, "thin_lens_f_px": thin_px, "thin_lens_object_mm": thin_so,
               "infinity_f_px": F_LENS / PIXEL, "rms_vs_held_f": {f"{k:.0f}": v for k, v in sorted(scan.items())},
               "tube_f_px": T["f"], "tube_f_sigma_px": T["sf"], "tube_line_f_px": T["line_f"], "tube_line_D0_mm": T["D0"],
               "tube_growth_measured": T["growth_meas"], "tube_growth_lens_front": T["growth_lens"],
               "tube_growth_camera_body": T["growth_body"], "pupil_behind_camera_body_mm": T["pupil_behind_body"],
               "pupil_behind_lens_front_mm": T["pupil_behind_lens_front"],
               "air_vs_tube": fb / T["f"] - 1, "growth_per_10mm_this_lens": 10 / (Z_E - 10),
               "growth_per_10mm_thin_lens": 10 / (thin_so - 10), "bow_f_quoted_px": BOW_F_QUOTED}
    with open(os.path.join(OUT_DIR, "focal_length_explained_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, default=float)
    print(f"air: f {fb:,.1f} px (formal sigma {A['best']['sf']:.1f}), rms {A['best']['rms']:.4f} px, "
          f"{A['n_obs']} observations of {A['n_dots']} dots, lattice residual max {A['lattice_res_max']:.2f} px")
    print("     starts -> " + ", ".join(f"{k:,.0f}: {v:,.1f}" for k, v in A["starts"].items()))
    print(f"     leave one out: {min(A['loo']):,.0f} .. {max(A['loo']):,.0f} px")
    print(f"     frontal images {[p['k'] for p in A['frontal']]}: m {m:.4f}, Z_E {Z_E:.1f} mm, pitch "
          f"{np.median([p['pitch_px'] for p in A['frontal']]):.2f} px (m * 150 = {m * PITCH / PIXEL:.2f})")
    print(f"     held f: 9,000 -> {scan[marks['inf']]:.3f} px; thin {thin_px:,.0f} -> {scan[marks['thin']]:.3f} px")
    print(f"tube: f {T['f']:,.1f} +- {T['sf']:.1f}; line {T['line_f']:,.1f}, D0 {T['D0']:.1f} mm; growth "
          f"{100 * T['growth_meas']:.3f} % (lens front {100 * T['growth_lens']:.2f}, body {100 * T['growth_body']:.2f})")
    print(f"      pupil behind the camera body {T['pupil_behind_body']:.1f} mm, behind the lens front "
          f"{T['pupil_behind_lens_front']:.1f} mm")
    print(f"thin lens: f {thin_px:,.1f} px, object distance {thin_so:.1f} mm; air vs tube {100 * (fb / T['f'] - 1):+.2f} %;"
          f" 10 mm demo {100 * 10 / (Z_E - 10):.2f} % vs {100 * 10 / (thin_so - 10):.2f} %")
    print(f"montage {mshape}; page {len(page) / 1e3:.0f} kB -> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
