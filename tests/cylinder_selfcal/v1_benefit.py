"""V1 (plan, section 3): the benefit, truth-free.  A field fitted on half of T1's frames, applied to the other half's
detections: are the true pairs sharper?

    python v1_benefit.py [--base t1_m15] [--variants tracks=t1_m15_v1trk tf=t1_m15_v1tf]

The fields are fitted on frames 0, 10, ..., 250 (v0_pattern.py: from the tracks; v0b_truthfree.py: from all candidate
pairs, no tracks).  gather.py --correct applies each to the detections of the held-out frames 5, 15, ..., 245 (every
detection moved by -c(d)) and recomputes S2's candidate pairs with the unchanged cameras.  On those frames, per
variant (no field = the base gather's frames):
  - S2's estimate of the true pairs within a tolerance t (the curve distance d <= t) for t = 0.5, 0.75, 1.0, 1.5 px:
    the pairs within t minus the random ones, whose density per px comes from the band 4 <= d <= 8 px; the random
    pairs within t; the true share;
  - the true pairs' distance profile (S2's: the share of the true pairs within 3 px that lie within each d; T1's was
    58 % within 1 px on the 31 px frames);
  - the depth cue at the tracker's tolerances (0.8 px / 20 %, T1's shape model): the true and random pairs it keeps.
Judged (plan): sharper true pairs, the share within 1 px rising from 58 % towards 80 %; at a tighter tolerance the true
pairs kept and the random ones falling in proportion.
Writes v1_benefit_out.txt, .json and v1_benefit.png (with --name NAME: NAME_out.txt, ...).  V3's warp check
(--name v3a_warpcheck --variants detections=t1_m15_v1all warped=t1_vcwarp): the field fitted on all frames
applied to the detections, against T1's frames warped by it (v3_warp.py): the same statistics if the warp
works as intended (not held out: that field saw these frames).
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import selfcal as sc

TOLS = [0.5, 0.75, 1.0, 1.5]
BG0, BAND = 4.0, 8.0
GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
HELD = list(range(5, 246, 10))


def shape_ok(dets, idx, z, model, tol=(0.8, 0.2)):
    """The depth cue: both RMS widths of the detections agree with the model at the pairs' depths."""
    A, zf, s0, dl, f = model["A_mm"], model["z_f_mm"], model["sigma0_px"], model["delta_mm"], model["f_px"]
    Lh = A * f * np.abs(1.0 / (z[:, 0] + dl) - 1.0 / zf)
    Lv = A * f * np.abs(1.0 / (z[:, 1] + dl) - 1.0 / zf)
    ph, pv = np.sqrt(Lh ** 2 / 16 + s0 ** 2), np.sqrt(Lv ** 2 / 16 + s0 ** 2)
    d = dets[idx]
    wh, wv = np.sqrt(d[:, 2] ** 2 / 12 + d[:, 4] ** 2), np.sqrt(d[:, 3] ** 2 / 12 + d[:, 5] ** 2)
    ag = lambda wm, wp: np.abs(wm - wp) <= np.maximum(tol[0], tol[1] * wp)  # noqa: E731
    return ag(wh, ph) & ag(wv, pv)


def stats(recs, models):
    nf = len(recs)
    d_all, cue = [], []
    n_det = []
    for g in recs:
        C = g["cand"]
        n_det.append([len(g["det0"]), len(g["det1"])])
        if not len(C):
            continue
        d_all.append(C[:, 5])
        ok = np.ones(len(C), bool)
        for c in (0, 1):
            idx = C[:, c].astype(int)
            z = g[f"z_cand{c}"]
            good = np.all(np.isfinite(z), axis=1)
            ok &= good & shape_ok(g[f"det{c}"], idx, np.where(good[:, None], z, 1e3), models[c])
        cue.append(ok)
    d = np.concatenate(d_all)
    cue = np.concatenate(cue)
    bg = (d >= BG0) & (d <= BAND)
    rate = bg.sum() / (BAND - BG0)                       # random pairs per px of d
    out = {"frames": nf, "detections_per_frame": np.mean(n_det, axis=0).tolist(), "tol": {}}
    for t in TOLS:
        s = int(np.sum(d <= t))
        rnd = rate * t
        tr = s - rnd
        se = math.sqrt(s + (t / (BAND - BG0)) ** 2 * bg.sum())
        sc_ = int(np.sum((d <= t) & cue))
        rnd_c = np.sum(bg & cue) / (BAND - BG0) * t
        out["tol"][str(t)] = {"true_per_frame": tr / nf, "true_se_per_frame": se / nf, "random_per_frame": rnd / nf,
                              "share": tr / s if s else None, "cue_true_per_frame": (sc_ - rnd_c) / nf,
                              "cue_random_per_frame": rnd_c / nf}
    h, e = np.histogram(d, bins=np.arange(0.0, BAND + 1e-9, 0.25))
    floor = h[e[:-1] >= BG0].mean()
    cum = np.cumsum(h - floor)
    tot = cum[e[1:] <= 3.0][-1]
    out["profile"] = {f"{x:.2f}": float(cum[np.searchsorted(e[1:], x)] / tot) for x in GRID}
    out["true_within_3px_per_frame"] = float(tot / nf)
    t15 = int(np.sum(d <= 1.5))
    t15c = int(np.sum((d <= 1.5) & cue))
    out["cue_true_kept"] = ((t15c - np.sum(bg & cue) / (BAND - BG0) * 1.5) /
                            max(t15 - rate * 1.5, 1e-9))
    out["cue_random_kept"] = float(np.sum(bg & cue) / max(bg.sum(), 1))
    out["hist"] = h.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="t1_m15")
    ap.add_argument("--variants", nargs="+", default=["tracks=t1_m15_v1trk", "truthfree=t1_m15_v1tf"])
    ap.add_argument("--name", default="v1_benefit", help="the outputs' stem (v3a_warpcheck: the warp check)")
    args = ap.parse_args()
    log = sc.Log(args.name)
    models = sc.load_json(os.path.join(sc.CASE_DATA, "T1_shape.json"))["cameras"]
    variants = [("none", args.base)] + [tuple(v.split("=")) for v in args.variants]
    out = {"held_out_frames": HELD, "variants": {}}
    log(f"V1: the held-out frames {HELD[0]}, {HELD[1]}, ..., {HELD[-1]} ({len(HELD)}); variants "
        + ", ".join(f"{a} ({b})" for a, b in variants))
    for name, gname in variants:
        recs = sc.load_gather(gname, frames=set(HELD))
        if len(recs) != len(HELD):
            raise SystemExit(f"{gname}: {len(recs)} of {len(HELD)} held-out frames gathered")
        s = stats(recs, models)
        out["variants"][name] = {"gather": gname, **s}
        log(f"  {name}: detections per frame {np.round(s['detections_per_frame'], 0).tolist()}; true pairs within 3 px "
            f"{s['true_within_3px_per_frame']:.0f} per frame; profile (share within d) "
            + ", ".join(f"{k} px {100 * v:.0f} %" for k, v in s["profile"].items()))
        for t in TOLS:
            r = s["tol"][str(t)]
            log(f"    t = {t:4.2f} px: true {r['true_per_frame']:6.1f} +- {r['true_se_per_frame']:.1f}, random "
                f"{r['random_per_frame']:6.1f}, true share {100 * r['share']:.1f} %; after the depth cue: true "
                f"{r['cue_true_per_frame']:.1f}, random {r['cue_random_per_frame']:.1f}")
        log(f"    the depth cue (t = 1.5): keeps {100 * s['cue_true_kept']:.0f} % of the true pairs and "
            f"{100 * s['cue_random_kept']:.1f} % of the random ones")
    base = out["variants"]["none"]
    for name, _g in variants[1:]:
        v = out["variants"][name]
        log(f"  {name} against none: within 1 px {100 * base['profile']['1.00']:.0f} % -> {100 * v['profile']['1.00']:.0f} %; "
            + "; ".join(f"t {t}: true {base['tol'][str(t)]['true_per_frame']:.0f} -> {v['tol'][str(t)]['true_per_frame']:.0f}"
                        f" ({100 * (v['tol'][str(t)]['true_per_frame'] / base['tol'][str(t)]['true_per_frame'] - 1):+.1f} %)"
                        for t in TOLS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, _g in variants:
        v = out["variants"][name]
        h = np.array(v["hist"])
        e = np.arange(0.0, BAND + 1e-9, 0.25)
        ax[0].step(0.5 * (e[1:] + e[:-1]), h / len(HELD), where="mid", label=name)
        ax[1].plot(GRID, [v["profile"][f"{x:.2f}"] for x in GRID], marker="o", label=name)
        ax[2].plot(TOLS, [v["tol"][str(t)]["true_per_frame"] for t in TOLS], marker="o", label=f"{name}: true")
        ax[2].plot(TOLS, [v["tol"][str(t)]["random_per_frame"] for t in TOLS], marker="x", ls="--",
                   label=f"{name}: random")
    ax[0].set_xlabel("distance from the curve d [px]")
    ax[0].set_ylabel("candidate pairs per frame per 0.25 px")
    ax[0].legend()
    ax[1].set_xlabel("d [px]")
    ax[1].set_ylabel("share of the true pairs (within 3 px) within d")
    ax[1].legend()
    ax[2].set_xlabel("tolerance t [px]")
    ax[2].set_ylabel("pairs per frame within t")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(sc.HERE, f"{args.name}.png"), dpi=100)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
