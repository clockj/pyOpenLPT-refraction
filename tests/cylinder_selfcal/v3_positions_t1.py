"""V3: how far the correction moved T1's 3-D positions (the gauge, report section 5): the tracks of the two-pass run
against today's at the same frames.

    python v3_positions_t1.py [--run s3_final_m15vc2] [--ref s3_final_m15]

Every long-track point of RUN with a long-track point of REF within 0.1 mm at the same frame (the recurrence test's
radius): the difference RUN - REF per axis, its median (a common shift) and robust spread, and its smooth part (the
medians in 4 x 4 x 4 cells of the view volume), i.e. how far the corrected cameras move the reconstruction where both
runs track the same particle.  Also the particles per frame in long tracks once tracking has built up (frames >= 150),
as T1's README quotes it.  Writes v3_positions_t1_out.txt and .json here.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.spatial import cKDTree

import selfcal as sc


def points(run):
    tr, _last = sc.read_tracks(os.path.join(sc.CASE_RUNS, run))
    return np.vstack([t[:, :4] for t in tr.values()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s3_final_m15vc2")
    ap.add_argument("--ref", default="s3_final_m15")
    a = ap.parse_args()
    log = sc.Log("v3_positions_t1")
    A, B = points(a.run), points(a.ref)
    out = {"run": a.run, "ref": a.ref}
    for nm, P in ((a.run, A), (a.ref, B)):
        fr = P[:, 0].astype(int)
        per = np.bincount(fr)[150:]
        out[f"particles_per_frame_from_150_{nm}"] = float(per.mean())
        log(f"  {nm}: {len(P)} long-track points; {per.mean():.0f} per frame from frame 150")
    d, X = [], []
    for f in np.unique(A[:, 0]).astype(int):
        pa, pb = A[A[:, 0] == f, 1:4], B[B[:, 0] == f, 1:4]
        if not len(pa) or not len(pb):
            continue
        dist, j = cKDTree(pb).query(pa)
        m = dist <= 0.1
        d.append(pa[m] - pb[j[m]])
        X.append(pa[m])
    d, X = np.vstack(d), np.vstack(X)
    rsd = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0)
    log(f"  {len(d)} points of {a.run} within 0.1 mm of {a.ref}'s at the same frame: difference median x/y/z "
        f"{1e3 * np.median(d[:, 0]):+.1f} / {1e3 * np.median(d[:, 1]):+.1f} / {1e3 * np.median(d[:, 2]):+.1f} um, robust SD "
        f"{1e3 * rsd[0]:.1f} / {1e3 * rsd[1]:.1f} / {1e3 * rsd[2]:.1f} um")
    vol = sc.view_volume()
    edges = [np.linspace(vol[k][0], vol[k][1], 5) for k in ("x", "y", "z")]
    idx = [np.clip(np.digitize(X[:, k], edges[k]) - 1, 0, 3) for k in range(3)]
    cell = idx[0] * 16 + idx[1] * 4 + idx[2]
    meds = []
    for c in np.unique(cell):
        m = cell == c
        if m.sum() >= 50:
            meds.append(np.median(d[m], axis=0))
    meds = np.array(meds)
    smooth = 1e3 * np.sqrt(np.mean(meds ** 2, axis=0))
    log(f"  the smooth part (medians in {len(meds)} cells of 4 x 4 x 4 over the view volume, >= 50 points): RMS x/y/z "
        f"{smooth[0]:.1f} / {smooth[1]:.1f} / {smooth[2]:.1f} um; largest |median| {1e3 * np.abs(meds).max():.1f} um")
    out.update({"matched_points": int(len(d)), "median_um": (1e3 * np.median(d, axis=0)).tolist(),
                "robust_sd_um": (1e3 * rsd).tolist(), "cells": int(len(meds)), "smooth_rms_um": smooth.tolist(),
                "smooth_max_um": float(1e3 * np.abs(meds).max())})
    log.save(out)


if __name__ == "__main__":
    main()
