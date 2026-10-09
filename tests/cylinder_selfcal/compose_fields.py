"""The second pass of the correction (VSC's iteration): one field of the original pixels from two passes.

    python compose_fields.py OUT.npz FIELD1.npz FIELD2.npz

FIELD1 was fitted on the original images, FIELD2 on the images corrected by FIELD1 (selfcal.compose: c_tot(d) =
c1(d) + c2(d - c1(d))).  Writes OUT.npz and compose_fields_<OUT>_out.txt here (the fields' sizes and the second
pass's share of the total).
"""
from __future__ import annotations

import os
import sys

import numpy as np

import selfcal as sc


def main():
    out_p, p1, p2 = sys.argv[1:4]
    name = os.path.splitext(os.path.basename(out_p))[0]
    log = sc.Log(f"compose_fields_{name}")
    f1, f2 = sc.Field.load(p1), sc.Field.load(p2)
    f = sc.compose(f1, f2)
    f.save(out_p)
    log(f"compose {p1} then {p2} -> {out_p}")
    rec = {}
    for c in range(f.n_cam):
        cov = f.covered[c]
        r1 = [float(np.sqrt(np.mean(f1.dense[c][k][cov] ** 2))) for k in (0, 1)]
        r2 = [float(np.sqrt(np.mean(f2.dense[c][k][cov] ** 2))) for k in (0, 1)]
        rt = [float(np.sqrt(np.mean(f.dense[c][k][cov] ** 2))) for k in (0, 1)]
        rec[f"cam{c + 1}"] = {"pass1_rms": r1, "pass2_rms": r2, "total_rms": rt}
        log(f"  cam{c + 1} over its covered pixels: du RMS {r1[0]:.3f} + {r2[0]:.3f} -> {rt[0]:.3f}, dv RMS {r1[1]:.3f} + "
            f"{r2[1]:.3f} -> {rt[1]:.3f} px (pass 1 + pass 2 -> total)")
    log.save(rec)


if __name__ == "__main__":
    main()
