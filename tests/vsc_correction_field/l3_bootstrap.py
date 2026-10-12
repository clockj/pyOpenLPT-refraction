"""L3 (c): is the 3-camera position RMS gap (library vs re-rendered, the same particles) larger than its noise?

    python l3_bootstrap.py [N]

The paired errors of l3_positions.py (particles matched in both runs at the same frame), resampled by particle (a
particle's errors over its frames are correlated, so it is the unit of resampling), 2,000 times: the RMS gap per axis,
library - re-rendered, its bootstrap standard error and 95 % interval. Writes l3_bootstrap_out.txt and .json here.
"""
from __future__ import annotations

import os
import sys

import numpy as np

import vcf_common as vc
from l3_positions import CAL4, matched_errors  # noqa: F401


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    log = vc.Log("l3_bootstrap" if n == 3 else f"l3_bootstrap_r{n}")
    A = matched_errors(os.path.join(vc.RUNS, f"l3_lib_r{n}"), "Gcal4_n8", n)
    B = matched_errors(os.path.join(vc.SELFCAL, "runs", f"Gcal4c{n}_n8_r{n}_tol1.0"), f"Gcal4c{n}_n8", n)
    ka = {(int(t), int(p)): 1e3 * np.array(e) for t, p, *e in A}
    kb = {(int(t), int(p)): 1e3 * np.array(e) for t, p, *e in B}
    common = sorted(set(ka) & set(kb))
    pid = np.array([k[1] for k in common])
    ea = np.array([ka[k] for k in common])
    eb = np.array([kb[k] for k in common])
    ids, inv = np.unique(pid, return_inverse=True)
    # per particle: sums of squares, counts
    sa = np.zeros((len(ids), 3))
    sb = np.zeros((len(ids), 3))
    cnt = np.bincount(inv)
    for ax in range(3):
        sa[:, ax] = np.bincount(inv, ea[:, ax] ** 2)
        sb[:, ax] = np.bincount(inv, eb[:, ax] ** 2)
    rng = np.random.default_rng(1)
    gaps = []
    for _ in range(2000):
        s = rng.integers(0, len(ids), len(ids))
        m = cnt[s].sum()
        gaps.append(np.sqrt(sa[s].sum(axis=0) / m) - np.sqrt(sb[s].sum(axis=0) / m))
    gaps = np.array(gaps)
    gap = np.sqrt(sa.sum(axis=0) / cnt.sum()) - np.sqrt(sb.sum(axis=0) / cnt.sum())
    se = gaps.std(axis=0)
    lo, hi = np.percentile(gaps, [2.5, 97.5], axis=0)
    log(f"{n} cameras: {len(common)} paired points of {len(ids)} particles; RMS gap library - re-rendered x/y/z "
        + "/".join(f"{g:+.2f}" for g in gap) + " um; bootstrap SE " + "/".join(f"{x:.2f}" for x in se)
        + " um; 95 % interval " + ", ".join(f"[{a:+.2f}, {b:+.2f}]" for a, b in zip(lo, hi)) + " um")
    log.save({"cameras": n, "paired_points": len(common), "particles": len(ids), "gap_um": gap.tolist(),
              "se_um": se.tolist(), "ci95_um": [lo.tolist(), hi.tolist()]})


if __name__ == "__main__":
    main()
