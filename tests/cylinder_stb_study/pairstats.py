"""Stereo-pair statistics of two cameras' detections, with and without the truth (plan: Parts D and G).

The method is S2's (cases/cyl_20261007/tests/s2_shape_fit.py, whose functions are imported, not copied):
  - candidate pairs: for every cam1 detection, its curved epipolar line on cam2 (StereoMatch.makeCurve2DFromLOS3D over
    the view volume), every cam2 detection within BAND px of it whose nearest point is inside the curve, the two lines
    of sight passing within 1 mm, the midpoint X inside the view volume; d = the distance from the curve;
  - the signal band d <= 1.5 px holds true and random pairs, the background band 4 <= d <= 8 px random ones only, at
    the same density per px, so (signal) - k (background), k = 1.5 / 4, estimates the true pairs in the signal band
    (no truth needed: the score of Part D and T1's side of the gap table);
  - the depth cue (tracer.shape_check): the measured RMS widths sqrt(L^2/12 + s^2) of both detections against the shape
    model's sqrt(L^2/16 + sigma0^2) at X, within max(0.8 px, 0.2 x the predicted) in both directions and cameras.
With the truth (synthetic sets), each detection is assigned to the nearest unassigned image of the set within
R_ASSIGN px (nearest pairs first), so a pair is true when both detections belong to one particle.
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

import study
import shape_model as sm

sys.path.insert(0, os.path.join(study.CASE_T1, "tests"))
from s2_shape_fit import BG0, SIG, candidate_pairs  # noqa: E402  (S2's own code; imports build p4c, as BUILD)

BAND = 8.0
K_BG = SIG / (BAND - BG0)
TOL = (0.8, 0.2)
R_ASSIGN = 2.5
EPI_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)


def matcher(lpt, camfiles, work_dir):
    """A StereoMatch object of the cameras over T1's view volume (its curved lines), from a minimal config."""
    import s_runs
    vol = study.view_volume()
    lists = []
    os.makedirs(work_dir, exist_ok=True)
    for c in range(len(camfiles)):
        p = os.path.join(work_dir, f"dummy_cam{c + 1}.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(os.path.join(study.CASE_T1, "data", "T1pre_s4", "cam1", "frame000000.tif") + "\n")
        lists.append(p)
    cfg = os.path.join(work_dir, "config_matcher.json")
    # an OTF table, so that the config does not estimate one from images (any table of as many cameras: the curves
    # need only the cameras and the volume)
    otf = os.path.join(study.CASE_T1, "data", "T1_otf_shape.txt") if len(camfiles) == 2 else         os.path.join(study.DATA, "main", f"otf_r{len(camfiles)}.txt")
    s_runs.write_config(cfg, work_dir, "-", len(camfiles), (0, 0), {"intensity_threshold": 30, "radius_px": 2.0,
                                                                      "otf_file": otf},
                        s_runs.ipr_for(len(camfiles), 1.0), image_lists=lists)
    with open(cfg, encoding="utf-8") as fh:
        import json
        c = json.load(fh)
    c["cameras"] = [{"file": f, "max_intensity": 255} for f in camfiles]
    with open(cfg, "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=1)
    bs = lpt.BasicSetting()
    bs.readConfig(cfg)
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    return cams, lpt.StereoMatch(cams, [[] for _ in cams], tc), vol


def detect(lpt, img8, noise, param):
    """LineFinder::detect: rows (u, v, Lx, Ly, sx, sy, A, c, flux, peak, score, chi2, npx, pass)."""
    rows = lpt.detectLines(lpt.Image.from_numpy(np.ascontiguousarray(img8)), float(noise), param)
    return np.array(rows, float).reshape(-1, 14)


def line_param(lpt, **over):
    P = lpt.LineFinderParam()
    for k, v in over.items():
        setattr(P, k, v)
    return P


def widths_meas(det):
    return np.sqrt(det[:, 2] ** 2 / 12 + det[:, 4] ** 2), np.sqrt(det[:, 3] ** 2 / 12 + det[:, 5] ** 2)


class Model:
    """A camera's shape model: widths predicted at points X (S2's widths_pred)."""

    def __init__(self, lpt, camfile, A, z_f, sigma0, delta=0.0):
        self.fd = sm.FocalDepths(lpt, camfile)
        self.f = math.sqrt(self.fd.K[0, 0] * self.fd.K[1, 1])
        self.A, self.z_f, self.s0, self.dl = A, z_f, sigma0, delta

    def widths(self, X):
        z = self.fd(X)
        Lh = self.A * self.f * np.abs(1.0 / (z[:, 0] + self.dl) - 1.0 / self.z_f)
        Lv = self.A * self.f * np.abs(1.0 / (z[:, 1] + self.dl) - 1.0 / self.z_f)
        return np.sqrt(Lh ** 2 / 16 + self.s0 ** 2), np.sqrt(Lv ** 2 / 16 + self.s0 ** 2)


def agree(wm, wp, tol=TOL):
    return np.abs(wm - wp) <= np.maximum(tol[0], tol[1] * wp)


def frame_pairs(cams, sm_, dets, vol, models=None):
    """The candidate pairs of one frame: array rows (i, j, X, Y, Z, d, gap, shape_ok_cam1, shape_ok_cam2)."""
    pairs = candidate_pairs(cams, sm_, dets, vol, BAND)
    if not pairs:
        return np.zeros((0, 9))
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    X = np.array([p[2] for p in pairs])
    d = np.array([p[3] for p in pairs])
    gap = np.array([p[4] for p in pairs])
    ok = np.ones((len(pairs), 2), bool)
    if models is not None:
        for c, (m, idx) in enumerate(zip(models, (i, j))):
            wh, wv = widths_meas(dets[c][idx])
            ph, pv = m.widths(X)
            ok[:, c] = agree(wh, ph) & agree(wv, pv)
    return np.column_stack([i, j, X, d, gap, ok[:, 0], ok[:, 1]])


def band_estimate(P, mask=None):
    """(estimated true pairs in the signal band, its SE, signal-band pairs, background-band pairs)."""
    m = np.ones(len(P), bool) if mask is None else mask
    d = P[:, 5]
    s = int(np.sum(m & (d <= SIG)))
    b = int(np.sum(m & (d >= BG0) & (d <= BAND)))
    return s - K_BG * b, math.sqrt(s + K_BG ** 2 * b), s, b


def distance_profile(P):
    """S2's profile: the share of the true pairs within 3 px that lie within each d of EPI_GRID (the excess over the
    floor of the 4-8 px bins, per 0.25 px)."""
    h, e = np.histogram(P[:, 5], bins=np.arange(0.0, BAND + 1e-9, 0.25))
    floor = h[e[:-1] >= BG0].mean()
    cum = np.cumsum(h - floor)
    tot = cum[e[1:] <= 3.0][-1]
    return {f"{x:.2f}": float(cum[np.searchsorted(e[1:], x)] / tot) for x in EPI_GRID}, float(tot)


def summarize(frames_pairs, n_det, label=""):
    """The truth-free figures of a list of frames' pairs (Part D's score)."""
    P = np.vstack([p for p in frames_pairs if len(p)]) if any(len(p) for p in frames_pairs) else np.zeros((0, 9))
    nf = len(frames_pairs)
    t, se, s, b = band_estimate(P)
    js = (P[:, 7] > 0) & (P[:, 8] > 0)
    t_sc, se_sc, s_sc, b_sc = band_estimate(P, js)
    prof, _tot = distance_profile(P) if len(P) else ({}, 0.0)
    nd = np.array(n_det, float)
    return {"frames": nf, "detections_per_frame": nd.mean(axis=0).tolist(),
            "true_pairs_per_frame": t / nf, "true_pairs_se_per_frame": se / nf, "signal_pairs_per_frame": s / nf,
            "background_pairs_per_frame": b / nf, "true_share_of_signal": t / s if s else None,
            "detections_with_partner": [t / nf / x for x in nd.mean(axis=0)],
            "shape_true_pairs_per_frame": t_sc / nf, "shape_true_pairs_se_per_frame": se_sc / nf,
            "shape_signal_pairs_per_frame": s_sc / nf,
            "shape_true_share_of_signal": t_sc / s_sc if s_sc else None,
            "shape_true_kept": t_sc / t if t > 0 else None,
            "shape_random_kept": (b_sc / b) if b else None,
            "epipolar_true_within": prof}


# ------------------------------------------------------------------ with the truth
def assign(det, uv, ids, r=R_ASSIGN):
    """Each detection's particle id (or -1): nearest pairs first within r px, each image and detection once."""
    out = np.full(len(det), -1)
    if not len(det) or not len(uv):
        return out
    tree = cKDTree(uv)
    lst = tree.query_ball_point(det[:, :2], r)
    cand = []
    for k, js in enumerate(lst):
        for j in js:
            cand.append((float(np.hypot(*(det[k, :2] - uv[j]))), k, j))
    cand.sort()
    used = set()
    for dd, k, j in cand:
        if out[k] >= 0 or j in used:
            continue
        out[k] = ids[j]
        used.add(j)
    return out


def truth_frame(case_dir, set_name, t, cams=(0, 1)):
    """The set's images in frame t: per camera (ids, uv) of the particles on the sensor; X (N, 3); layer."""
    sets = study.load_json(os.path.join(case_dir, "sets.json"))
    s = sets[set_name]
    a = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"), mmap_mode="r")
    layer = np.asarray(a("layer"))
    member = np.isin(layer, [["in", "outA", "outB"].index(ly) for ly in s["layers"]])
    out = []
    for k in cams:
        uv = np.asarray(a(f"uv_{s['variant']}_c{k}")[t])
        on = member & np.isfinite(uv[:, 0]) & (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 800)
        ids = np.flatnonzero(on)
        out.append((ids, uv[ids]))
    return out, np.asarray(a("X")[t]), layer


def truth_figures(dets, imgs, X, layer, P, vol):
    """Per frame, with the truth: detection recall (of the particles inside the view volume seen by both cameras),
    false detections, centre errors, detections with a partner, and the candidate pairs' true / random labels."""
    lab = [assign(dets[c], imgs[c][1], imgs[c][0]) for c in (0, 1)]
    both = np.intersect1d(imgs[0][0], imgs[1][0])
    inv = both[study.in_volume(X[both], vol)]
    rec = {}
    for c in (0, 1):
        found = np.isin(inv, lab[c])
        rec[f"recall_cam{c + 1}"] = (int(found.sum()), int(len(inv)))
        rec[f"false_cam{c + 1}"] = int(np.sum(lab[c] < 0))
        rec[f"det_cam{c + 1}"] = int(len(dets[c]))
        # centre errors of the assigned detections
        pos = {int(i): k for k, i in enumerate(imgs[c][0])}
        e = [np.hypot(*(dets[c][k, :2] - imgs[c][1][pos[int(i)]])) for k, i in enumerate(lab[c]) if i >= 0]
        rec[f"centre_err_cam{c + 1}"] = e
    partner = np.intersect1d(lab[0][lab[0] >= 0], lab[1][lab[1] >= 0])
    rec["paired_particles"] = int(len(partner))
    rec["paired_in_volume"] = int(np.sum(study.in_volume(X[partner], vol))) if len(partner) else 0
    rec["paired_central"] = int(np.sum(layer[partner] == 0)) if len(partner) else 0
    true = np.zeros(len(P), bool)
    if len(P):
        li, lj = lab[0][P[:, 0].astype(int)], lab[1][P[:, 1].astype(int)]
        true = (li >= 0) & (li == lj)
    rec["pair_true"] = true
    return rec


def truth_summary(recs, Ps):
    """Aggregate truth_figures over frames, with the candidate pairs Ps."""
    nf = len(recs)
    P = np.vstack([p for p in Ps if len(p)]) if any(len(p) for p in Ps) else np.zeros((0, 9))
    T = np.concatenate([r["pair_true"] for r in recs]) if recs else np.zeros(0, bool)
    d = P[:, 5] if len(P) else np.zeros(0)
    js = (P[:, 7] > 0) & (P[:, 8] > 0) if len(P) else np.zeros(0, bool)
    sig = d <= SIG
    out = {"frames": nf}
    for c in (1, 2):
        f, n = sum(r[f"recall_cam{c}"][0] for r in recs), sum(r[f"recall_cam{c}"][1] for r in recs)
        e = np.concatenate([np.asarray(r[f"centre_err_cam{c}"]) for r in recs]) if recs else np.zeros(0)
        out[f"cam{c}"] = {"recall_both_in_volume": f / max(1, n), "present_per_frame": n / nf,
                          "detections_per_frame": sum(r[f"det_cam{c}"] for r in recs) / nf,
                          "false_per_frame": sum(r[f"false_cam{c}"] for r in recs) / nf,
                          "centre_err_px_pcts": np.percentile(e, [50, 90, 99]).tolist() if len(e) else None}
    out["paired_particles_per_frame"] = sum(r["paired_particles"] for r in recs) / nf
    out["paired_in_volume_per_frame"] = sum(r["paired_in_volume"] for r in recs) / nf
    out["detections_with_partner"] = [out["paired_particles_per_frame"] / max(1e-9, out[f"cam{c}"]["detections_per_frame"])
                                      for c in (1, 2)]
    n_true_all = int(T.sum())
    out["true_pairs_found_per_frame"] = n_true_all / nf
    out["true_pairs_in_signal_band_per_frame"] = int(np.sum(T & sig)) / nf
    out["true_epipolar_within"] = {f"{x:.2f}": float(np.mean(d[T] <= x)) if n_true_all else None for x in EPI_GRID}
    out["signal_band_true_share"] = float(np.mean(T[sig])) if sig.any() else None
    out["shape_true_kept"] = float(np.mean(js[T & sig])) if np.any(T & sig) else None
    out["shape_random_kept"] = float(np.mean(js[~T & sig])) if np.any(~T & sig) else None
    out["shape_signal_true_share"] = float(np.mean(T[sig & js])) if np.any(sig & js) else None
    return out


def read8(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return img
