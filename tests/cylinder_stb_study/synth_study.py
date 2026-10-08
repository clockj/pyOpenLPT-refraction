"""The study's rendered cases (plan: 2026-10-08-plan.md, sections 3, 4 and 7).

    python synth_study.py main                  # Parts S and O, G-cal, G-shape: data/main/   (about 15 min)
    python synth_study.py gall-trial            # G-all's density: frame 0 at three densities, detections per frame
    python synth_study.py gall --density RHO    # G-all: data/gall/
    python synth_study.py gshape0               # a diagnostic of G-shape: T1's model without the scatter (data/main)
    python synth_study.py gshape12              # G-shape at half the scatter (SD 12.5 %, the same draws)
    python synth_study.py gall12                # G-all at half the scatter: data/gall's set Gall12
    python synth_study.py gall12m15             # Gall12 preprocessed with a 15 px median (D1's best): set Gall12_m15
    python synth_study.py gallm-trial [--scatter S] [--trial-rho ...] [--trial-bc ...]
    python synth_study.py gallm --density RHO --bc-max B --scatter S     # G-all matched to T1: data/gallm/

data/main (git-ignored, regenerable):
  - the population, one for every rig and set: particles moving with synth.py's swirl about the tube's axis (scaled
    so that the median image step is 1.5 px, as r5_lines), in three layers of uniform density:
      'in'    r < 30 mm (the central region), |y - T_y| < 30 mm, RHO_IN (chosen so that cam1 and cam2 see about as many
              images as r5_lines' cameras did: 1,696 and 2,154);
      'outA', 'outB'  30 <= r < 97.6 mm (the rest of the water), |y - T_y| < 60 mm, RHO_IN / 2 each;
    each particle kept if a camera of the 4-camera rig sees it on its sensor at the first, middle or last frame
    (T: make_rigs.py's aim point);
  - per camera: the pixel along each track (hinted projections), the focal depths z_tan, z_sag (shape_model's
    parallax of a displaced pinhole, hinted along the track), the image angle of the tube's axis (first frame seen);
  - shapes ("variants"):
      clean  synth_lines.py's model (A 15 mm, sigma0 1 px, flux 2,500 x U(0.6, 1.4) per particle), each camera focused
             at the circle of least confusion of the 'in' particles it sees: 1/z_f = the median of
             (1/z_tan + 1/z_sag)/2 (cam1 and cam2: 12 mm before z_tan, as T1's fitted focus, 11 mm; the elevated
             cameras' astigmatism is larger, 19 mm);
      cal    clean, with each image moved by a smooth error field: T0's residual pattern (a quadratic fit to the
             10 x 6 cell means of cases/cyl_20261007/tests/c1_residuals.json) scaled to 0.7 px RMS over the sensor;
             the tracker keeps the true cameras, so its model is off by the field;
      t1     T1's fitted model (cases/cyl_20261007/data/T1_shape.json: A, z_f, sigma0, the flux law
             flux = e^a (|b| + |c| + 2)^b with b, c the box lengths, held at |b| + |c| = 50 px beyond: T1's largest
             detected objects, L2, are about 60 px long), each length times a factor 1 + 0.25 N(0, 1) (per particle
             and camera, clipped to 0.4-1.6), the flux from the scattered lengths x U(0.6, 1.4);
  - image sets (img/<set>/camK/img%06d.tif, imgLists/<set>/camKImageNames.txt): the sum of the layers' images plus
    a standard-normal field per camera and frame (the same field in every set) times the set's noise, then S1's
    preprocessing as on T1 (cases/cyl_20261007/lines.py: minus the 31 px median, computed at a raw scale where the
    noise is 1.4 counts as on T1, recentred on the noise level of the negative tail, rounded and clipped to 8 bits),
    so that the smooth background of many out-of-focus particles is removed as on T1:
      S_n4, S_n8, S_n16   'in' (clean), noise 4, 8, 16                        Part S, 4 cameras
      O05_n8, O10_n8      'in' + 'outA' (+ 'outB') (clean), noise 8           Part O, 4 cameras
      Gcal_n8             'in' (cal), noise 8                                 G-cal, cam1 and cam2
      Gshape_nT1          'in' (t1), T1's noise (8.06, 5.92)                  G-shape, cam1 and cam2
      Gshape0_nT1         'in' (t1x: t1 without the scatter), T1's noise      gshape0, a diagnostic of G-shape
      Gshape12_nT1        'in' (t1h: t1 at half the scatter), T1's noise      gshape12: T1's depth cue lies between
  - otf_r2.txt, otf_r3.txt, otf_r4.txt: the clean model's OTF tables (model "segment", 9 x 7 x 11 nodes over T1's
    view volume) for the rigs; otf_t1.txt: T1's own table (the case's data/T1_otf_shape.txt), for G-shape;
  - arrays/ (the truth), sets.json, synth.json (parameters and counts).
data/gall: one layer, uniform over the water (r < 97.6 mm, |y - T_y| < 60 mm) at RHO, 80 frames, cam1 and cam2, every
  factor at once: the error field, T1's model with scatter, T1's noise; set 'Gall'; G3's control pairs cam1's frames
  0-39 with cam2's 40-79.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from multiprocessing import Pool

import cv2
import numpy as np

import study
import cstb
import shape_model as sm
from synth import swirl_positions
from synth_lines import render_ellipse

RHO_IN = 0.07            # particles per mm^3 in the central region
FRAMES = 40
STEP_PX = 1.5
CLEAN = {"A": 15.0, "sigma0": 1.0, "F": 2500.0}
FIELD_RMS = 0.7          # px, the error field's RMS over the sensor
SCATTER = 0.25           # the lengths' scatter (SD of the factor)
NOISE_T1 = [8.058751135183766, 5.920146272161428]     # T1's noise per camera (S1's median), 8-bit counts
H_PARALLAX = 0.5         # mm, the displaced pinholes of the focal depths (shape_model.FocalDepths)
FLUX_BC_MAX = 50.0       # px, T1's flux law held beyond this |b| + |c| (its measured range)
HP_SIZE = 31             # px, S1's median window
HP_RAW_NOISE = 1.4       # counts, T1's noise at the raw scale of S1's median (lines.median_lowpass's resolution)
N_PROC = 30

SETS_MAIN = {
    "S_n4": {"cams": [0, 1, 2, 3], "layers": ["in"], "variant": "clean", "noise": [4.0] * 4},
    "S_n8": {"cams": [0, 1, 2, 3], "layers": ["in"], "variant": "clean", "noise": [8.0] * 4},
    "S_n16": {"cams": [0, 1, 2, 3], "layers": ["in"], "variant": "clean", "noise": [16.0] * 4},
    "O05_n8": {"cams": [0, 1, 2, 3], "layers": ["in", "outA"], "variant": "clean", "noise": [8.0] * 4},
    "O10_n8": {"cams": [0, 1, 2, 3], "layers": ["in", "outA", "outB"], "variant": "clean", "noise": [8.0] * 4},
    "Gcal_n8": {"cams": [0, 1], "layers": ["in"], "variant": "cal", "noise": [8.0] * 2},
    "Gshape_nT1": {"cams": [0, 1], "layers": ["in"], "variant": "t1", "noise": NOISE_T1},
}
LAYERS = ["in", "outA", "outB"]


# ------------------------------------------------------------------ worker processes
_CACHE = {}


def _lpt():
    if "lpt" not in _CACHE:
        _CACHE["lpt"] = study.lib()[0]
    return _CACHE["lpt"]


def _one_thread():
    """One thread per worker process (N_PROC of them share the cores): OpenCV's pool."""
    if "cv2_threads" not in _CACHE:
        cv2.setNumThreads(1)
        _CACHE["cv2_threads"] = True


def _cam(path):
    key = ("cam", path)
    if key not in _CACHE:
        _CACHE[key] = _lpt().loadCameraModel(path)
    return _CACHE[key]


def _vis_job(job):
    """On the sensor (bool) for the points X (m, 3) in one camera."""
    path, X = job
    lpt, cam = _lpt(), _cam(path)
    res = cam.projectBatchStatus([lpt.Pt3D(*map(float, x)) for x in X])
    W, H = cam.getNCol(), cam.getNRow()
    return np.array([r[0] and 0 <= r[1][0] < W and 0 <= r[1][1] < H for r in res], bool)


def _track_job(job):
    """Along the tracks Xt (F, m, 3) in one camera: pixels (F, m, 2), focal depths (F, m, 2) where the track is near
    the sensor, the image angle of the tube's axis at the first frame on the sensor (m,)."""
    path, disp, K, Xt, margin = job
    lpt = _lpt()
    cam = _cam(path)
    dc = {k: _cam(p) for k, p in disp.items()}
    W, H = cam.getNCol(), cam.getNRow()
    F, m, _ = Xt.shape
    uv = np.full((F, m, 2), np.nan)
    z = np.full((F, m, 2), np.nan)
    alpha = np.zeros(m)
    for i in range(m):
        h = lpt.ProjectionHint()
        near = False
        for t in range(F):
            ok, p, h2, _e = cam.projectWithHint(lpt.Pt3D(*map(float, Xt[t, i])), h)
            if ok:
                uv[t, i] = (p[0], p[1])
                h = h2
                if -margin <= p[0] < W + margin and -margin <= p[1] < H + margin:
                    near = True
            else:
                h = lpt.ProjectionHint()
        if not near:
            continue
        hints = {k: None for k in dc}
        first = True
        for t in range(F):
            P = lpt.Pt3D(*map(float, Xt[t, i]))
            q = {}
            for k in ("x+", "x-", "y+", "y-"):
                hk = hints[k] if hints[k] is not None else (hints["x+"] if hints["x+"] is not None
                                                            else lpt.ProjectionHint())
                ok, p, h2, _e = dc[k].projectWithHint(P, hk)
                if not ok:
                    hints[k] = None
                    break
                hints[k] = h2
                q[k] = p
            if len(q) == 4:
                du = q["x+"][0] - q["x-"][0]
                dv = q["y+"][1] - q["y-"][1]
                z[t, i] = (-K[0, 0] * 2 * H_PARALLAX / du, -K[1, 1] * 2 * H_PARALLAX / dv)
            if first and np.isfinite(uv[t, i, 0]) and 0 <= uv[t, i, 0] < W and 0 <= uv[t, i, 1] < H:
                try:
                    a = cam.project(P)
                    b = cam.project(lpt.Pt3D(float(Xt[t, i, 0]), float(Xt[t, i, 1]) + 0.05, float(Xt[t, i, 2])))
                except Exception:          # no image of one of the two points: keep the axis vertical
                    first = False
                    continue
                ang = math.atan2(b[1] - a[1], b[0] - a[0]) - math.pi / 2
                while ang <= -math.pi / 2:
                    ang += math.pi
                while ang > math.pi / 2:
                    ang -= math.pi
                alpha[i] = ang
                first = False
    return uv, z, alpha


def _arr(case_dir, name):
    key = ("arr", case_dir, name)
    if key not in _CACHE:
        _CACHE[key] = np.load(os.path.join(case_dir, "arrays", name + ".npy"), mmap_mode="r")
    return _CACHE[key]


def _render_job(job):
    """Every set's image of camera c at frame t."""
    case_dir, c, t, sets, frame_names, noise_seed, H, W = job
    _one_thread()
    layer = np.asarray(_arr(case_dir, "layer"))
    need = sorted({(s["variant"], ly) for s in sets.values() if c in s["cams"] for ly in s["layers"]})
    imgs = {}
    for variant, ly in need:
        uv = _arr(case_dir, f"uv_{variant}_c{c}")[t]
        Lh = _arr(case_dir, f"Lh_{variant}_c{c}")[t]
        Lv = _arr(case_dir, f"Lv_{variant}_c{c}")[t]
        fl = _arr(case_dir, f"flux_{variant}_c{c}")[t]
        al = _arr(case_dir, f"alpha_c{c}")
        s0 = float(np.asarray(_arr(case_dir, f"sigma0_{variant}"))[c])
        img = np.zeros((H, W))
        for i in np.flatnonzero((layer == LAYERS.index(ly)) & np.isfinite(uv[:, 0]) & np.isfinite(Lh)):
            render_ellipse(img, uv[i, 0], uv[i, 1], Lh[i], Lv[i], al[i], fl[i], s0)
        imgs[(variant, ly)] = img
    zf = np.random.default_rng((noise_seed, c, t)).standard_normal((H, W))
    out = []
    for name, s in sets.items():
        if c not in s["cams"]:
            continue
        noise = s["noise"][s["cams"].index(c)]
        img = sum(imgs[(s["variant"], ly)] for ly in s["layers"]) + noise * zf
        path = os.path.join(case_dir, "img", name, f"cam{c + 1}", frame_names[t])
        cv2.imwrite(path, preprocess(img, noise, s.get("median_px", HP_SIZE)))
        out.append(path)
    return c, t, out


def preprocess(img, noise, size=HP_SIZE):
    """S1's preprocessing of a rendered float image (8-bit units): minus its size x size median, computed by
    lines.median_lowpass at the raw scale where the noise is HP_RAW_NOISE counts (T1's), recentred on the noise level
    of the negative tail (lines.noise_level), rounded and clipped to 8 bits."""
    import lines
    g = noise / HP_RAW_NOISE
    x = np.asarray(img, np.float32)
    hp = x - lines.median_lowpass(x / np.float32(g), size) * np.float32(g)
    hp -= np.float32(lines.noise_level(hp))
    return np.clip(np.rint(hp), 0, 255).astype(np.uint8)


def float_image(case_dir, set_name, c, t, noise_seed):
    """The set's float image of camera c at frame t before the preprocessing (for D2's median windows)."""
    sets = study.load_json(os.path.join(case_dir, "sets.json"))
    s = sets[set_name]
    layer = np.asarray(_arr(case_dir, "layer"))
    uv = _arr(case_dir, f"uv_{s['variant']}_c{c}")[t]
    Lh = _arr(case_dir, f"Lh_{s['variant']}_c{c}")[t]
    Lv = _arr(case_dir, f"Lv_{s['variant']}_c{c}")[t]
    fl = _arr(case_dir, f"flux_{s['variant']}_c{c}")[t]
    al = _arr(case_dir, f"alpha_c{c}")
    s0 = float(np.asarray(_arr(case_dir, f"sigma0_{s['variant']}"))[c])
    img = np.zeros((800, 1280))
    sel = np.isin(layer, [LAYERS.index(ly) for ly in s["layers"]]) & np.isfinite(uv[:, 0]) & np.isfinite(Lh)
    for i in np.flatnonzero(sel):
        render_ellipse(img, uv[i, 0], uv[i, 1], Lh[i], Lv[i], al[i], fl[i], s0)
    zf = np.random.default_rng((noise_seed, c, t)).standard_normal((800, 1280))
    return img + s["noise"][s["cams"].index(c)] * zf


# ------------------------------------------------------------------ the population
def sample_annulus(rng, rho, r_lo, r_hi, y_lo, y_hi):
    n = rng.poisson(rho * math.pi * (r_hi ** 2 - r_lo ** 2) * (y_hi - y_lo))
    r = np.sqrt(r_lo ** 2 + rng.random(n) * (r_hi ** 2 - r_lo ** 2))
    ph = rng.uniform(0.0, 2 * math.pi, n)
    return np.column_stack([r * np.sin(ph), rng.uniform(y_lo, y_hi, n), r * np.cos(ph)])


def visible_any(pool, files, Xs, chunk=4000):
    """Seen on the sensor of any camera at any of the positions Xs (k, n, 3)."""
    jobs, keys = [], []
    for f in files:
        for k in range(Xs.shape[0]):
            for a in range(0, Xs.shape[1], chunk):
                jobs.append((f, Xs[k, a:a + chunk]))
                keys.append(a)
    vis = np.zeros(Xs.shape[1], bool)
    for a, res in zip(keys, pool.map(_vis_job, jobs)):
        vis[a:a + len(res)] |= res
    return vis


def displaced(case_dir, files):
    """shape_model's four displaced copies of each camera (pinhole moved by +-H_PARALLAX along its x and y)."""
    out = []
    for c, f in enumerate(files):
        d = {}
        for tag, s in (("x+", (H_PARALLAX, 0, 0)), ("x-", (-H_PARALLAX, 0, 0)), ("y+", (0, H_PARALLAX, 0)),
                       ("y-", (0, -H_PARALLAX, 0))):
            d[tag] = sm.displaced_camfile(f, os.path.join(case_dir, "displaced", f"cam{c + 1}_{tag}.txt"), s)
        out.append(d)
    return out


def tracks(pool, files, disp, Xt, chunk=400, margin=150.0):
    """Per camera: pixels (F, n, 2), focal depths (F, n, 2), axis angles (n,)."""
    out = []
    for f, d in zip(files, disp):
        K = sm.camera_geometry(f)[2]
        jobs = [(f, d, K, Xt[:, a:a + chunk], margin) for a in range(0, Xt.shape[1], chunk)]
        res = pool.map(_track_job, jobs)
        out.append((np.concatenate([r[0] for r in res], axis=1), np.concatenate([r[1] for r in res], axis=1),
                    np.concatenate([r[2] for r in res])))
    return out


# ------------------------------------------------------------------ the error field
def error_fields(n_cam=2, rms=FIELD_RMS):
    """Per camera a function (u, v) -> (du, dv): a quadratic fit to T0's residual cells, scaled to rms."""
    res = study.load_json(os.path.join(study.CASE_T1, "tests", "c1_residuals.json"))
    out, info = [], []
    for c in range(n_cam):
        cells = res["cameras"][f"cam{c + 1}"]["cells"]
        U = np.array([cl["centre"] for cl in cells])
        w = np.array([cl["n"] for cl in cells], float)
        D = np.array([[cl["mean_du"], cl["mean_dv"]] for cl in cells])
        x, y = (U[:, 0] - 639.5) / 640.0, (U[:, 1] - 399.5) / 400.0
        B = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
        coef = np.linalg.solve(B.T @ (w[:, None] * B), B.T @ (w[:, None] * D))
        gu, gv = np.meshgrid(np.linspace(0, 1279, 65), np.linspace(0, 799, 41))
        gx, gy = (gu.ravel() - 639.5) / 640.0, (gv.ravel() - 399.5) / 400.0
        G = np.column_stack([np.ones_like(gx), gx, gy, gx * gx, gx * gy, gy * gy]) @ coef
        rms0 = float(np.sqrt(np.mean((G ** 2).sum(axis=1))))
        cells_rms = float(np.sqrt(np.average((D ** 2).sum(axis=1), weights=w)))
        scale = rms / rms0
        coef_s = coef * scale

        def f(u, v, coef_s=coef_s):
            xx, yy = (np.asarray(u) - 639.5) / 640.0, (np.asarray(v) - 399.5) / 400.0
            Bm = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], axis=-1)
            return Bm @ coef_s

        out.append(f)
        Gs = G * scale
        info.append({"cells": len(cells), "cells_rms_px": cells_rms, "fit_rms_px": rms0, "scale": scale,
                     "rms_px": rms, "max_px": float(np.sqrt((Gs ** 2).sum(axis=1)).max()),
                     "coef_du": coef_s[:, 0], "coef_dv": coef_s[:, 1]})
    return out, info


# ------------------------------------------------------------------ the shapes
def t1_model():
    m = study.load_json(os.path.join(study.CASE_T1, "data", "T1_shape.json"))
    return [{"A": c["A_mm"], "z_f": c["z_f_mm"], "sigma0": c["sigma0_px"], "delta": c["delta_mm"],
             "flux_a": c["flux_log_a"], "flux_b": c["flux_log_b"], "f_px": c["f_px"]} for c in m["cameras"]]


def lengths(z, f_px, A, z_f, delta=0.0):
    return A * f_px * (1.0 / (z[..., 0] + delta) - 1.0 / z_f), A * f_px * (1.0 / (z[..., 1] + delta) - 1.0 / z_f)


def t1_shapes(z, f_px, p, fac, scat, bc_max=FLUX_BC_MAX):
    """T1's lengths with the scatter factors scat (n, 2) and its flux law (held beyond |b| + |c| = bc_max), times the
    per-particle factor fac."""
    Lh, Lv = lengths(z, f_px, p["A"], p["z_f"], p["delta"])
    Lh, Lv = Lh * scat[None, :, 0], Lv * scat[None, :, 1]
    bc = np.minimum(sm.BOX * (np.abs(Lh) + np.abs(Lv)), bc_max)
    flux = math.exp(p["flux_a"]) * (bc + 2.0) ** p["flux_b"] * fac[None, :]
    return Lh, Lv, flux


# ------------------------------------------------------------------ writing a case
def save_arrays(case_dir, arrs):
    d = os.path.join(case_dir, "arrays")
    os.makedirs(d, exist_ok=True)
    for k, v in arrs.items():
        np.save(os.path.join(d, k + ".npy"), np.asarray(v))


def render_sets(pool, case_dir, sets, n_frames, noise_seed, H, W, log):
    names = [f"img{t:06d}.tif" for t in range(n_frames)]
    for name, s in sets.items():
        for c in s["cams"]:
            os.makedirs(os.path.join(case_dir, "img", name, f"cam{c + 1}"), exist_ok=True)
    cams = sorted({c for s in sets.values() for c in s["cams"]})
    jobs = [(case_dir, c, t, sets, names, noise_seed, H, W) for c in cams for t in range(n_frames)]
    t0 = time.time()
    for _r in pool.imap_unordered(_render_job, jobs):
        pass
    for name, s in sets.items():
        os.makedirs(os.path.join(case_dir, "imgLists", name), exist_ok=True)
        for c in s["cams"]:
            with open(os.path.join(case_dir, "imgLists", name, f"cam{c + 1}ImageNames.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write("\n".join(os.path.join(case_dir, "img", name, f"cam{c + 1}", n) for n in names) + "\n")
    log(f"  rendered {len(sets)} sets ({len(jobs)} camera-frames) in {time.time() - t0:.0f} s")


def on_sensor(uv, W, H):
    return np.isfinite(uv[..., 0]) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)


def summary(log, tag, uv, Lh, Lv, flux, s0, noise, layer, W, H):
    on = on_sensor(uv, W, H)
    peak = flux / (sm.profile_integral(sm.BOX * Lh, s0) * sm.profile_integral(sm.BOX * Lv, s0))
    rec = {}
    for li, ly in enumerate(LAYERS):
        m = on & (layer[None, :] == li)
        if not m.any():
            continue
        rec[ly] = {"images_per_frame": float(m.sum(axis=1).mean()),
                   "Lh_px_pcts": np.nanpercentile(np.abs(Lh[m]), [10, 50, 90]).tolist(),
                   "Lv_px_pcts": np.nanpercentile(np.abs(Lv[m]), [10, 50, 90]).tolist(),
                   "peak_pcts": np.nanpercentile(peak[m], [10, 50, 90]).tolist(),
                   "peak_to_noise_median": float(np.nanmedian(peak[m]) / noise)}
        log(f"    {tag} {ly}: {rec[ly]['images_per_frame']:.0f} images per frame; |L_h| 10/50/90 % "
            f"{'/'.join(f'{x:.1f}' for x in rec[ly]['Lh_px_pcts'])} px, |L_v| "
            f"{'/'.join(f'{x:.1f}' for x in rec[ly]['Lv_px_pcts'])} px; peak (model) "
            f"{'/'.join(f'{x:.0f}' for x in rec[ly]['peak_pcts'])}, median peak / noise {noise:g}: "
            f"{rec[ly]['peak_to_noise_median']:.1f}")
    return rec


# ------------------------------------------------------------------ main: Parts S, O, G-cal, G-shape
def make_main(args, pool, log):
    lpt, build = study.lib()
    case_dir = os.path.join(study.DATA, "main")
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(os.path.join(case_dir, "camFile"))
    files = []
    for c, f in enumerate(study.rig_files(4)):
        dst = os.path.join(case_dir, "camFile", f"cam{c + 1}.txt")
        shutil.copy2(f, dst)
        files.append(dst)
    cams = [lpt.loadCameraModel(f) for f in files]
    H, W = cams[0].getNRow(), cams[0].getNCol()
    rig = study.load_json(os.path.join(study.HERE, "make_rigs.json"))
    T = np.array(rig["aim_point_mm"])
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # the layers, kept where a camera sees them at the first, middle or last frame
    X_in = sample_annulus(rng, RHO_IN, 0.0, study.R_IN, T[1] - 30.0, T[1] + 30.0)
    X_oa = sample_annulus(rng, RHO_IN / 2, study.R_IN, study.R_WATER, T[1] - 60.0, T[1] + 60.0)
    X_ob = sample_annulus(rng, RHO_IN / 2, study.R_IN, study.R_WATER, T[1] - 60.0, T[1] + 60.0)
    X0 = np.vstack([X_in, X_oa, X_ob])
    layer = np.concatenate([np.full(len(X_in), 0), np.full(len(X_oa), 1), np.full(len(X_ob), 2)]).astype(np.int8)
    # the swirl's scale from 300 central particles (as synth_lines.py): the median image step in the four cameras
    trial = swirl_positions(X_in[:600], 2, 1e-3, 0.0)
    P = np.array([[_cam_project(lpt, cam, x) for x in trial[k]] for k in range(2) for cam in cams]).reshape(
        2, len(cams), -1, 2)
    d = np.hypot(*(P[1] - P[0]).transpose(2, 0, 1))
    on0 = on_sensor(P[0], W, H)
    omega = 1e-3 * STEP_PX / float(np.nanmedian(d[on0]))
    vy = 0.3 * omega * float(np.median(np.hypot(X_in[:, 0], X_in[:, 2])))
    chk = swirl_positions(X0, FRAMES, omega, vy)[[0, FRAMES // 2, FRAMES - 1]]
    keep = visible_any(pool, files, chk)
    X0, layer = X0[keep], layer[keep]
    log(f"  population: sampled in {len(X_in)}, outA {len(X_oa)}, outB {len(X_ob)}; kept (seen by a camera) "
        + ", ".join(f"{ly} {int((layer == k).sum())}" for k, ly in enumerate(LAYERS))
        + f"; swirl {omega:.3e} rad per frame, axial drift {vy * 1e3:.1f} um per frame; {time.time() - t0:.0f} s")
    Xt = swirl_positions(X0, FRAMES, omega, vy)
    disp = displaced(case_dir, files)
    tr = tracks(pool, files, disp, Xt)
    log(f"  tracks projected in the four cameras: {time.time() - t0:.0f} s")

    # shapes
    fac = rng.uniform(0.6, 1.4, len(X0))
    arrs = {"X": Xt, "layer": layer}
    params, sig = [], {"clean": [], "cal": [], "t1": []}
    fields, finfo = error_fields(2)
    t1 = t1_model()
    meta_shapes = {}
    for c, f in enumerate(files):
        uv, z, alpha = tr[c]
        K = sm.camera_geometry(f)[2]
        f_px = math.sqrt(K[0, 0] * K[1, 1])
        sel = (layer == 0) & on_sensor(uv[0], W, H) & np.isfinite(z[0, :, 0])
        zf = 1.0 / float(np.median(0.5 * (1.0 / z[0, sel, 0] + 1.0 / z[0, sel, 1])))   # least confusion
        params.append({"A": CLEAN["A"], "z_f": zf, "sigma0": CLEAN["sigma0"], "F": CLEAN["F"], "f_px": f_px})
        Lh, Lv = lengths(z, f_px, CLEAN["A"], zf)
        flux = CLEAN["F"] * np.broadcast_to(fac[None, :], Lh.shape)
        arrs.update({f"uv_clean_c{c}": uv, f"Lh_clean_c{c}": Lh, f"Lv_clean_c{c}": Lv, f"flux_clean_c{c}": flux,
                     f"alpha_c{c}": alpha, f"z_c{c}": z})
        sig["clean"].append(CLEAN["sigma0"])
        log(f"  cam{c + 1}: z_f {zf:.1f} mm (least confusion; the central particles' median z_tan "
            f"{np.median(z[0, sel, 0]):.1f}, z_sag {np.median(z[0, sel, 1]):.1f} mm)")
        meta_shapes[f"cam{c + 1}"] = summary(log, "clean, noise 8:", uv, Lh, Lv, flux, CLEAN["sigma0"], 8.0, layer,
                                             W, H)
        if c < 2:
            du = fields[c](uv[..., 0], uv[..., 1])
            arrs.update({f"uv_cal_c{c}": uv + du, f"Lh_cal_c{c}": Lh, f"Lv_cal_c{c}": Lv, f"flux_cal_c{c}": flux})
            sig["cal"].append(CLEAN["sigma0"])
            scat = np.clip(1.0 + SCATTER * rng.standard_normal((len(X0), 2)), 0.4, 1.6)
            Lh1, Lv1, fl1 = t1_shapes(z, f_px, t1[c], fac, scat)
            arrs.update({f"uv_t1_c{c}": uv, f"Lh_t1_c{c}": Lh1, f"Lv_t1_c{c}": Lv1, f"flux_t1_c{c}": fl1,
                         f"scatter_c{c}": scat})
            sig["t1"].append(t1[c]["sigma0"])
            meta_shapes[f"cam{c + 1}_t1"] = summary(log, f"t1, noise {NOISE_T1[c]:.2f}:", uv, Lh1, Lv1, fl1,
                                                    t1[c]["sigma0"], NOISE_T1[c], layer, W, H)
    for v, s in sig.items():
        arrs[f"sigma0_{v}"] = np.array(s)
    save_arrays(case_dir, arrs)

    # the OTF tables
    vol = study.view_volume()
    for n in (2, 3, 4):
        sm.build_otf(lpt, files[:n], params[:n], study.vol_box(vol), (9, 7, 11),
                     os.path.join(case_dir, f"otf_r{n}.txt"))
    shutil.copy2(os.path.join(study.CASE_T1, "data", "T1_otf_shape.txt"), os.path.join(case_dir, "otf_t1.txt"))
    log(f"  OTF tables written: {time.time() - t0:.0f} s")

    render_sets(pool, case_dir, SETS_MAIN, FRAMES, args.seed + 100, H, W, log)
    meta = {"args": vars(args), "build": build, "rho_in_per_mm3": RHO_IN, "frames": FRAMES,
            "noise_seed": args.seed + 100, "preprocess": {"median_px": HP_SIZE, "raw_noise": HP_RAW_NOISE},
            "flux_bc_max_px": FLUX_BC_MAX,
            "omega_rad_per_frame": omega, "vy_mm_per_frame": vy, "aim_point_mm": T, "params_clean": params,
            "params_t1": t1, "error_field": finfo, "scatter_sd": SCATTER, "noise_t1": NOISE_T1,
            "particles": {ly: int((layer == k).sum()) for k, ly in enumerate(LAYERS)}, "shapes": meta_shapes,
            "view_volume": vol, "seconds": time.time() - t0}
    study.save_json(os.path.join(case_dir, "synth.json"), meta)
    study.save_json(os.path.join(case_dir, "sets.json"), SETS_MAIN)
    return meta


def _cam_project(lpt, cam, x):
    ok, p, _h, _e = cam.projectWithHint(lpt.Pt3D(*map(float, x)), lpt.ProjectionHint())
    return (p[0], p[1]) if ok else (np.nan, np.nan)


# ------------------------------------------------------------------ G-all
def gall_population(pool, files, T, rho, frames, omega, vy, rng, check):
    X0 = sample_annulus(rng, rho, 0.0, study.R_WATER, T[1] - 60.0, T[1] + 60.0)
    chk = swirl_positions(X0, frames, omega, vy)[check]
    keep = visible_any(pool, files, chk)
    return X0[keep], int(len(X0))


def make_gall(args, pool, log, trial=False):
    lpt, build = study.lib()
    main_meta = study.load_json(os.path.join(study.DATA, "main", "synth.json"))
    omega, vy, T = main_meta["omega_rad_per_frame"], main_meta["vy_mm_per_frame"], np.array(main_meta["aim_point_mm"])
    name = "gall_trial" if trial else "gall"
    case_dir = os.path.join(study.DATA, name)
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(os.path.join(case_dir, "camFile"))
    files = []
    for c, f in enumerate(study.rig_files(2)):
        dst = os.path.join(case_dir, "camFile", f"cam{c + 1}.txt")
        shutil.copy2(f, dst)
        files.append(dst)
    H, W = 800, 1280
    disp = displaced(case_dir, files)
    fields, finfo = error_fields(2)
    t1 = t1_model()
    rhos = args.trial_rho if trial else [args.density]
    frames = 1 if trial else 2 * FRAMES
    out = {"build": build, "omega": omega, "vy": vy, "runs": []}
    for rho in rhos:
        rng = np.random.default_rng(args.seed + 7)
        t0 = time.time()
        X0, n_s = gall_population(pool, files, T, rho, frames, omega, vy, rng,
                                  [0] if trial else [0, FRAMES // 2, FRAMES, 3 * FRAMES // 2, 2 * FRAMES - 1])
        Xt = swirl_positions(X0, frames, omega, vy)
        layer = (np.hypot(X0[:, 0], X0[:, 2]) >= study.R_IN).astype(np.int8)    # 0 central, 1 outside
        tr = tracks(pool, files, disp, Xt)
        fac = rng.uniform(0.6, 1.4, len(X0))
        arrs = {"X": Xt, "layer": layer}
        for c, f in enumerate(files):
            uv, z, alpha = tr[c]
            K = sm.camera_geometry(f)[2]
            f_px = math.sqrt(K[0, 0] * K[1, 1])
            scat = np.clip(1.0 + SCATTER * rng.standard_normal((len(X0), 2)), 0.4, 1.6)
            Lh, Lv, fl = t1_shapes(z, f_px, t1[c], fac, scat)
            arrs.update({f"uv_all_c{c}": uv + fields[c](uv[..., 0], uv[..., 1]), f"uv_model_c{c}": uv,
                         f"Lh_all_c{c}": Lh, f"Lv_all_c{c}": Lv, f"flux_all_c{c}": fl, f"alpha_c{c}": alpha,
                         f"z_c{c}": z, f"scatter_c{c}": scat})
        arrs["sigma0_all"] = np.array([t1[0]["sigma0"], t1[1]["sigma0"]])
        sub = os.path.join(case_dir, f"rho{rho:g}") if trial else case_dir
        os.makedirs(sub, exist_ok=True)
        save_arrays(sub, arrs)
        sets = {"Gall": {"cams": [0, 1], "layers": ["in", "outA"], "variant": "all", "noise": NOISE_T1}}
        render_sets(pool, sub, sets, frames, args.seed + 200, H, W, log)
        rec = {"rho": rho, "sampled": n_s, "kept": int(len(X0)), "central": int((layer == 0).sum()),
               "seconds": time.time() - t0}
        if trial:
            P = lpt.LineFinderParam()
            dets = []
            for c in range(2):
                img = cv2.imread(os.path.join(sub, "img", "Gall", f"cam{c + 1}", "img000000.tif"),
                                 cv2.IMREAD_UNCHANGED)
                dets.append(len(lpt.detectLines(lpt.Image.from_numpy(img), NOISE_T1[c], P)))
            rec["detections"] = dets
            for c in range(2):
                rec[f"images_cam{c + 1}"] = int(on_sensor(arrs[f"uv_all_c{c}"][0], W, H).sum())
        log(f"  rho {rho:g} per mm^3: sampled {n_s}, kept {len(X0)} ({rec['central']} central)"
            + (f"; images cam1 {rec['images_cam1']}, cam2 {rec['images_cam2']}; detections {rec['detections']}"
               if trial else "") + f"; {rec['seconds']:.0f} s")
        out["runs"].append(rec)
    if not trial:
        study.save_json(os.path.join(case_dir, "sets.json"),
                        {"Gall": {"cams": [0, 1], "layers": ["in", "outA"], "variant": "all", "noise": NOISE_T1}})
        study.save_json(os.path.join(case_dir, "synth.json"),
                        {"args": vars(args), "build": build, "rho_per_mm3": args.density, "frames": 2 * FRAMES,
                         "noise_seed": args.seed + 200, "preprocess": {"median_px": HP_SIZE, "raw_noise": HP_RAW_NOISE},
                         "flux_bc_max_px": FLUX_BC_MAX,
                         "omega_rad_per_frame": omega, "vy_mm_per_frame": vy, "aim_point_mm": T,
                         "params_t1": t1, "error_field": finfo, "scatter_sd": SCATTER, "noise_t1": NOISE_T1,
                         "particles": out["runs"][0], "view_volume": study.view_volume(),
                         "layers": "0: r < 30 mm, 1: r >= 30 mm (one population)"})
        shutil.copy2(os.path.join(study.CASE_T1, "data", "T1_otf_shape.txt"), os.path.join(case_dir, "otf_t1.txt"))
    return out


def make_gshape0(args, pool, log):
    """Adds the variant t1x (T1's model, every scatter factor 1) and the set Gshape0_nT1 to data/main, and renders
    only that set: the same particles, flux factors and noise field as Gshape_nT1."""
    case_dir = os.path.join(study.DATA, "main")
    meta = study.load_json(os.path.join(case_dir, "synth.json"))
    t1 = t1_model()
    a = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"))
    arrs = {}
    for c in range(2):
        z = a(f"z_c{c}")
        fac = a(f"flux_clean_c{c}")[0] / CLEAN["F"]          # the per-particle factors
        K = sm.camera_geometry(os.path.join(case_dir, "camFile", f"cam{c + 1}.txt"))[2]
        f_px = math.sqrt(K[0, 0] * K[1, 1])
        Lh, Lv, fl = t1_shapes(z, f_px, t1[c], fac, np.ones((z.shape[1], 2)))
        arrs.update({f"uv_t1x_c{c}": a(f"uv_t1_c{c}"), f"Lh_t1x_c{c}": Lh, f"Lv_t1x_c{c}": Lv, f"flux_t1x_c{c}": fl})
    arrs["sigma0_t1x"] = np.array([t1[0]["sigma0"], t1[1]["sigma0"]])
    save_arrays(case_dir, arrs)
    sets = study.load_json(os.path.join(case_dir, "sets.json"))
    sets["Gshape0_nT1"] = {"cams": [0, 1], "layers": ["in"], "variant": "t1x", "noise": NOISE_T1}
    study.save_json(os.path.join(case_dir, "sets.json"), sets)
    render_sets(pool, case_dir, {"Gshape0_nT1": sets["Gshape0_nT1"]}, FRAMES, meta["noise_seed"], 800, 1280, log)
    return {"set": "Gshape0_nT1", "variant": "t1x"}


def make_gshape12(args, pool, log):
    """Adds the variant t1h (T1's model, each scatter factor halfway to 1: SD 12.5 % with G-shape's draws) and the
    set Gshape12_nT1 to data/main, and renders only that set."""
    case_dir = os.path.join(study.DATA, "main")
    meta = study.load_json(os.path.join(case_dir, "synth.json"))
    t1 = t1_model()
    a = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"))
    arrs = {}
    for c in range(2):
        z = a(f"z_c{c}")
        fac = a(f"flux_clean_c{c}")[0] / CLEAN["F"]
        scat = np.clip(1.0 + 0.5 * (a(f"scatter_c{c}") - 1.0), 0.4, 1.6)
        K = sm.camera_geometry(os.path.join(case_dir, "camFile", f"cam{c + 1}.txt"))[2]
        f_px = math.sqrt(K[0, 0] * K[1, 1])
        Lh, Lv, fl = t1_shapes(z, f_px, t1[c], fac, scat)
        arrs.update({f"uv_t1h_c{c}": a(f"uv_t1_c{c}"), f"Lh_t1h_c{c}": Lh, f"Lv_t1h_c{c}": Lv, f"flux_t1h_c{c}": fl})
    arrs["sigma0_t1h"] = np.array([t1[0]["sigma0"], t1[1]["sigma0"]])
    save_arrays(case_dir, arrs)
    sets = study.load_json(os.path.join(case_dir, "sets.json"))
    sets["Gshape12_nT1"] = {"cams": [0, 1], "layers": ["in"], "variant": "t1h", "noise": NOISE_T1}
    study.save_json(os.path.join(case_dir, "sets.json"), sets)
    render_sets(pool, case_dir, {"Gshape12_nT1": sets["Gshape12_nT1"]}, FRAMES, meta["noise_seed"], 800, 1280, log)
    return {"set": "Gshape12_nT1", "variant": "t1h"}


def make_gall12(args, pool, log):
    """Adds the variant all12 (G-all with each scatter factor halfway to 1: SD 12.5 % with G-all's draws; the same
    particles, error field and noise) and the set Gall12 to data/gall, and renders only that set (80 frames)."""
    case_dir = os.path.join(study.DATA, "gall")
    meta = study.load_json(os.path.join(case_dir, "synth.json"))
    t1 = t1_model()
    a = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"))
    arrs = {}
    for c in range(2):
        z = a(f"z_c{c}")
        scat0 = a(f"scatter_c{c}")
        Lh0 = a(f"Lh_all_c{c}")
        K = sm.camera_geometry(os.path.join(case_dir, "camFile", f"cam{c + 1}.txt"))[2]
        f_px = math.sqrt(K[0, 0] * K[1, 1])
        # the per-particle flux factors: G-all's flux over its law at its own lengths
        _Lh, _Lv, unit = t1_shapes(z, f_px, t1[c], np.ones(z.shape[1]), scat0)
        with np.errstate(invalid="ignore", divide="ignore"):
            fac = np.nanmedian(a(f"flux_all_c{c}") / unit, axis=0)
        fac = np.where(np.isfinite(fac), fac, 1.0)
        scat = np.clip(1.0 + 0.5 * (scat0 - 1.0), 0.4, 1.6)
        Lh, Lv, fl = t1_shapes(z, f_px, t1[c], fac, scat)
        arrs.update({f"uv_all12_c{c}": a(f"uv_all_c{c}"), f"Lh_all12_c{c}": Lh, f"Lv_all12_c{c}": Lv,
                     f"flux_all12_c{c}": fl})
        del Lh0
    arrs["sigma0_all12"] = a("sigma0_all")
    save_arrays(case_dir, arrs)
    sets = study.load_json(os.path.join(case_dir, "sets.json"))
    sets["Gall12"] = {"cams": [0, 1], "layers": ["in", "outA"], "variant": "all12", "noise": NOISE_T1}
    study.save_json(os.path.join(case_dir, "sets.json"), sets)
    render_sets(pool, case_dir, {"Gall12": sets["Gall12"]}, meta["frames"], meta["noise_seed"], 800, 1280, log)
    return {"set": "Gall12", "variant": "all12"}


def make_gall12m15(args, pool, log):
    """The set Gall12_m15: Gall12's images preprocessed with a 15 px median (D1's best on T1), and its noise as the
    detector should be told it: measured as S1 measures it (lines.noise_left of the float high-pass image) on frame 0
    of each camera."""
    import lines
    case_dir = os.path.join(study.DATA, "gall")
    meta = study.load_json(os.path.join(case_dir, "synth.json"))
    sets = study.load_json(os.path.join(case_dir, "sets.json"))
    spec = dict(sets["Gall12"], median_px=15)
    noise = []
    for c in range(2):
        x = np.asarray(float_image(case_dir, "Gall12", c, 0, meta["noise_seed"]), np.float32)
        g = NOISE_T1[c] / HP_RAW_NOISE
        hp = x - lines.median_lowpass(x / np.float32(g), 15) * np.float32(g)
        hp -= np.float32(lines.noise_level(hp))
        noise.append(float(lines.noise_left(hp)))
    spec["noise_measured"] = noise
    sets["Gall12_m15"] = spec
    study.save_json(os.path.join(case_dir, "sets.json"), sets)
    log(f"  Gall12_m15: noise measured on frame 0 after a 15 px median {np.round(noise, 2).tolist()} "
        f"(rendered at {np.round(NOISE_T1, 2).tolist()})")
    render_sets(pool, case_dir, {"Gall12_m15": spec}, meta["frames"], meta["noise_seed"], 800, 1280, log)
    return {"set": "Gall12_m15", "noise_measured": noise}


def _trial_img(job):
    """A float image (8-bit units) of the given particles: (uv, Lh, Lv, alpha, flux, sigma0, H, W)."""
    uv, Lh, Lv, al, fl, s0, H, W = job
    _one_thread()
    img = np.zeros((H, W))
    for i in np.flatnonzero(np.isfinite(uv[:, 0]) & np.isfinite(Lh)):
        render_ellipse(img, uv[i, 0], uv[i, 1], Lh[i], Lv[i], al[i], fl[i], s0)
    return img


def measured_noise(x, noise, size=HP_SIZE):
    """S1's noise of a rendered float image: lines.noise_left of its high-pass (minus the size px median, recentred)."""
    import lines
    g = noise / HP_RAW_NOISE
    x = np.asarray(x, np.float32)
    hp = x - lines.median_lowpass(x / np.float32(g), size) * np.float32(g)
    hp -= np.float32(lines.noise_level(hp))
    return float(lines.noise_left(hp))


def make_gallm(args, pool, log, trial=False):
    """G-all matched to T1 (data/gallm): uniform over the water at DENSITY, the error field, T1's shape model with
    the scatter SCATTER and T1's flux law held beyond |b| + |c| = BC_MAX (fainter out-of-focus particles than G-all's
    50 px), T1's noise rendered, and the noise the detector is told measured as S1 measures it on T1.  The trial
    renders frame 0 for each density in --trial-rho and each BC_MAX in --trial-bc (one population, thinned) and
    reports the measured noise and the detections (told that noise), against T1's 8.06 / 5.88 and 1,002 / 865."""
    lpt, build = study.lib()
    main_meta = study.load_json(os.path.join(study.DATA, "main", "synth.json"))
    omega, vy, T = main_meta["omega_rad_per_frame"], main_meta["vy_mm_per_frame"], np.array(main_meta["aim_point_mm"])
    case_dir = os.path.join(study.DATA, "gallm_trial" if trial else "gallm")
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(os.path.join(case_dir, "camFile"))
    files = []
    for c, f in enumerate(study.rig_files(2)):
        dst = os.path.join(case_dir, "camFile", f"cam{c + 1}.txt")
        shutil.copy2(f, dst)
        files.append(dst)
    H, W = 800, 1280
    disp = displaced(case_dir, files)
    fields, finfo = error_fields(2)
    t1 = t1_model()
    frames = 1 if trial else 2 * FRAMES
    rho_max = max(args.trial_rho) if trial else args.density
    rng = np.random.default_rng(args.seed + 11)
    X0, n_s = gall_population(pool, files, T, rho_max, frames, omega, vy, rng,
                              [0] if trial else [0, FRAMES // 2, FRAMES, 3 * FRAMES // 2, 2 * FRAMES - 1])
    u_thin = rng.random(len(X0))
    Xt = swirl_positions(X0, frames, omega, vy)
    layer = (np.hypot(X0[:, 0], X0[:, 2]) >= study.R_IN).astype(np.int8)
    tr = tracks(pool, files, disp, Xt)
    fac = rng.uniform(0.6, 1.4, len(X0))
    scat = [np.clip(1.0 + args.scatter * rng.standard_normal((len(X0), 2)), 0.4, 1.6) for _c in range(2)]
    K = [sm.camera_geometry(f)[2] for f in files]
    f_px = [math.sqrt(k[0, 0] * k[1, 1]) for k in K]
    out = {"build": build, "scatter": args.scatter, "trials": []}
    if trial:
        P = lpt.LineFinderParam()
        for rho in args.trial_rho:
            keep = u_thin < rho / rho_max
            for bc in args.trial_bc:
                rec = {"rho": rho, "bc_max": bc, "kept": int(keep.sum()), "noise_measured": [], "detections": [],
                       "images": []}
                for c in range(2):
                    uv, z, al = tr[c]
                    Lh, Lv, fl = t1_shapes(z, f_px[c], t1[c], fac, scat[c], bc)
                    uvr = uv + fields[c](uv[..., 0], uv[..., 1])
                    m = keep
                    img = _trial_img((uvr[0][m], Lh[0][m], Lv[0][m], al[m], fl[0][m], t1[c]["sigma0"], H, W))
                    zf = np.random.default_rng((args.seed + 300, c, 0)).standard_normal((H, W))
                    x = img + NOISE_T1[c] * zf
                    nm = measured_noise(x, NOISE_T1[c])
                    img8 = preprocess(x, NOISE_T1[c])
                    det = lpt.detectLines(lpt.Image.from_numpy(img8), nm, P)
                    rec["noise_measured"].append(round(nm, 2))
                    rec["detections"].append(len(det))
                    rec["images"].append(int(on_sensor(uvr[0][m], W, H).sum()))
                log(f"  rho {rho:g}, flux held beyond |b|+|c| = {bc:g} px: kept {rec['kept']}; images {rec['images']}; "
                    f"noise measured {rec['noise_measured']} (T1 8.06, 5.88); detections {rec['detections']} "
                    f"(T1 1002, 865)")
                out["trials"].append(rec)
        return out
    keep = u_thin < 2.0          # every particle at the chosen density
    arrs = {"X": Xt, "layer": layer}
    for c in range(2):
        uv, z, al = tr[c]
        Lh, Lv, fl = t1_shapes(z, f_px[c], t1[c], fac, scat[c], args.bc_max)
        arrs.update({f"uv_allm_c{c}": uv + fields[c](uv[..., 0], uv[..., 1]), f"uv_model_c{c}": uv,
                     f"Lh_allm_c{c}": Lh, f"Lv_allm_c{c}": Lv, f"flux_allm_c{c}": fl, f"alpha_c{c}": al,
                     f"z_c{c}": z, f"scatter_c{c}": scat[c]})
    arrs["sigma0_allm"] = np.array([t1[0]["sigma0"], t1[1]["sigma0"]])
    save_arrays(case_dir, arrs)
    seed = args.seed + 300
    spec = {"cams": [0, 1], "layers": ["in", "outA"], "variant": "allm", "noise": NOISE_T1}
    nm = [measured_noise(float_image_arrays(case_dir, spec, c, 0, seed), NOISE_T1[c]) for c in range(2)]
    spec["noise_measured"] = nm
    study.save_json(os.path.join(case_dir, "sets.json"), {"Gallm": spec})
    render_sets(pool, case_dir, {"Gallm": spec}, frames, seed, H, W, log)
    shutil.copy2(os.path.join(study.CASE_T1, "data", "T1_otf_shape.txt"), os.path.join(case_dir, "otf_t1.txt"))
    meta = {"args": vars(args), "build": build, "rho_per_mm3": args.density, "bc_max_px": args.bc_max,
            "scatter_sd": args.scatter, "frames": frames, "noise_seed": seed, "omega_rad_per_frame": omega,
            "vy_mm_per_frame": vy, "aim_point_mm": T, "params_t1": t1, "error_field": finfo, "noise_t1": NOISE_T1,
            "noise_measured": nm, "particles": {"sampled": n_s, "kept": int(len(X0)), "central": int((layer == 0).sum())},
            "preprocess": {"median_px": HP_SIZE, "raw_noise": HP_RAW_NOISE}, "view_volume": study.view_volume()}
    study.save_json(os.path.join(case_dir, "synth.json"), meta)
    log(f"  G-all matched to T1: rho {args.density:g}, flux held beyond {args.bc_max:g} px, scatter {args.scatter:g}; "
        f"kept {len(X0)} ({int((layer == 0).sum())} central); noise measured {np.round(nm, 2).tolist()}")
    return meta


def float_image_arrays(case_dir, spec, c, t, noise_seed):
    """float_image for a set spec not yet in sets.json."""
    layer = np.asarray(_arr(case_dir, "layer"))
    v = spec["variant"]
    uv = _arr(case_dir, f"uv_{v}_c{c}")[t]
    Lh = _arr(case_dir, f"Lh_{v}_c{c}")[t]
    Lv = _arr(case_dir, f"Lv_{v}_c{c}")[t]
    fl = _arr(case_dir, f"flux_{v}_c{c}")[t]
    al = _arr(case_dir, f"alpha_c{c}")
    s0 = float(np.asarray(_arr(case_dir, f"sigma0_{v}"))[c])
    img = np.zeros((800, 1280))
    sel = np.isin(layer, [LAYERS.index(ly) for ly in spec["layers"]]) & np.isfinite(uv[:, 0]) & np.isfinite(Lh)
    for i in np.flatnonzero(sel):
        render_ellipse(img, uv[i, 0], uv[i, 1], Lh[i], Lv[i], al[i], fl[i], s0)
    zf = np.random.default_rng((noise_seed, c, t)).standard_normal((800, 1280))
    return img + spec["noise"][spec["cams"].index(c)] * zf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["main", "gall-trial", "gall", "gshape0", "gshape12", "gall12", "gall12m15",
                                     "gallm-trial", "gallm"])
    ap.add_argument("--bc-max", type=float, default=FLUX_BC_MAX)
    ap.add_argument("--scatter", type=float, default=SCATTER)
    ap.add_argument("--trial-bc", type=float, nargs="+", default=[8.0, 15.0, 25.0])
    ap.add_argument("--density", type=float, default=None, help="gall: particles per mm^3")
    ap.add_argument("--trial-rho", type=float, nargs="+", default=[0.01, 0.02, 0.035])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    log = study.Log(f"synth_study_{args.what.replace('-', '_')}")
    log(f"synth_study {args.what}: build {study.BUILD}")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # inherited by the worker processes
        os.environ[k] = "1"
    with Pool(N_PROC) as pool:
        if args.what == "main":
            out = make_main(args, pool, log)
        elif args.what == "gall-trial":
            out = make_gall(args, pool, log, trial=True)
        elif args.what == "gshape0":
            out = make_gshape0(args, pool, log)
        elif args.what == "gshape12":
            out = make_gshape12(args, pool, log)
        elif args.what == "gall12":
            out = make_gall12(args, pool, log)
        elif args.what == "gall12m15":
            out = make_gall12m15(args, pool, log)
        elif args.what == "gallm-trial":
            out = make_gallm(args, pool, log, trial=True)
        elif args.what == "gallm":
            if args.density is None:
                raise SystemExit("gallm needs --density (from gallm-trial)")
            out = make_gallm(args, pool, log)
        else:
            if args.density is None:
                raise SystemExit("gall needs --density (from gall-trial)")
            out = make_gall(args, pool, log)
    log.save(out)


if __name__ == "__main__":
    main()
