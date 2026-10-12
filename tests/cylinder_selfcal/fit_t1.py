"""A T1 field from one gathered run, as V0 fits it (the second pass of V3 fits the warped run's tracks).

    python fit_t1.py GATHER OUT.npz

The projected least squares of selfcal (Cauchy weights, the noise model by the residuals with s_f >= 0.15 px, the
bending weight by held-out loss on frames 0, 10, ... against 5, 15, ...), then the fit on every gathered frame, its
dense form with the coverage fill.  Reports the offset across the curves it predicts at the points and how much of
the points' offset it takes away.  Writes OUT.npz and fit_t1_<GATHER>_out.txt, .json here.
"""
from __future__ import annotations

import sys

import numpy as np

import selfcal as sc

LAMS = [1e3, 1e4, 1e5, 1e6, 1e7]


def main():
    gname, out_p = sys.argv[1:3]
    log = sc.Log(f"fit_t1_{gname}")
    recs = sc.load_gather(gname)
    fd = sc.fitdata(recs)
    log(f"fit {gname}: {len(recs)} frames, {len(fd)} points paired in both cameras within {sc.R_PAIR} px")
    m = fd.frame % 10 == 0
    f = sc.Field(2)
    rec, lam, noise = sc.fit_with_noise(f, fd.subset(m), fd.subset(~m), LAMS, log)
    held = sc.heldout_score(f, fd.subset(~m), noise)
    g = sc.Field(2)
    r_all = sc.fit_field(g, fd, lam, noise, log=log)
    g.make_dense([fd.d[fd.mask[:, c], c] for c in (0, 1)])
    g.meta = {"gather": gname, "lam": lam, "noise": list(noise), "fit": r_all}
    g.save(out_p)
    obs0 = sc.observable(fd, fd.r)
    pred = sc.observable(fd, sc.field_at(g, fd))
    rsd = lambda x: float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))  # noqa: E731
    log(f"  held out (frames 5, 15, ...): projected MAD {held['mad_before']:.3f} -> {held['mad_after']:.3f}")
    log(f"  the field's offset across the curves at the points: RMS {np.sqrt(np.nanmean(pred ** 2)):.3f} px; the points' "
        f"projected offset robust SD {rsd(obs0):.3f} -> {rsd(obs0 - pred):.3f} px")
    for c in (0, 1):
        cov = g.covered[c]
        log(f"  cam{c + 1}: dv RMS over its covered pixels {np.sqrt(np.mean(g.dense[c][1][cov] ** 2)):.3f} px, du "
            f"{np.sqrt(np.mean(g.dense[c][0][cov] ** 2)):.3f}")
    log.save({"gather": gname, "out": out_p, "lam": lam, "noise": list(noise), "selection": rec, "heldout": held,
              "offset_rms_px": float(np.sqrt(np.nanmean(pred ** 2))), "points_sd": [rsd(obs0), rsd(obs0 - pred)]})


if __name__ == "__main__":
    main()
