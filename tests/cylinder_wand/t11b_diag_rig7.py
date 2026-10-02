"""T11b diagnosis: T5 rig 7 at 5 px, the one start from scratch that missed the global minimum.

Rebuilds the same data set as T11b (T5's seed 5007 for the rig and frames, noise seed 140703), then follows the
bootstrap: its six short-listed starts, each carried through stages 3 and 4 as `calibrate` does, against the fit
started at the truth.  `calibrate` keeps the best two starts (n_final = 2) and picks the one keeping the most
observations, then the lowest RMS.  Reported per start: its bootstrap cost, mirror flag and axis offset, the final
RMS and active count, and its largest parameter difference from the truth-started fit; and what the wrong minimum
is, in physical units.  A diagnosis only: no code is changed.
Outputs: t11b_diag_rig7_out.txt, t11b_diag_rig7.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402

OUT = rigs.HERE / "t11b_diag_rig7"
log = rigs.Tee(str(OUT) + "_out.txt")


def main():
    t0 = time.time()
    tube = rigs.CASE_TUBE
    rng = np.random.default_rng(5007)
    truth, centre, half = rigs.random_rig(rng)
    wd = rigs.wand_frames(truth, tube, 500, tc.L_WAND, rng, half=(half,) * 3, centre=centre, noise_px=0.0)
    ds = tc.DataSet(truth, tube, {"kind": "gauss", "sigma": 5.0}, seed=140000 + 100 * 7 + 3, held=False, wd=wd)
    log(f"rig 7: {len(truth)} cameras at azimuths "
        f"{', '.join(f'{math.degrees(math.atan2(c.C[0], c.C[2])) % 360:.0f}' for c in truth)} deg, distances "
        f"{', '.join(f'{math.hypot(c.C[0], c.C[2]):.0f}' for c in truth)} mm, f "
        f"{', '.join(f'{c.f:.0f}' for c in truth)} px; cloud +-{half:.1f} mm; noise 5 px")
    ref = cw.calibrate(ds.data, tube, start=truth)
    log(f"started at the truth: rms {ref.rms:.4f} px, active {int(ref.obs.active.sum())}/{len(ref.obs)}")
    sol = cw.calibrate(ds.data, tube)
    log(f"from scratch (calibrate): rms {sol.rms:.4f} px, max diff {tc.zmax_between(sol, ref):.2f} sigma "
        f"(T11b found 5.4202 px, 7.7 sigma)")
    obs = ds.obs()
    ranked, binfo = cw.bootstrap(obs, tube, tc.L_WAND, ds.sizes, verbose=True, log=log)
    log(f"bootstrap: affine {binfo['affine_rms_px']:.2f} px over {binfo['affine_frames']} frames, "
        f"{binfo['candidates']} candidates, best mirror {binfo['best']['mirror']} at {binfo['best']['p']}")
    starts = []
    for k, (cost, cams0) in enumerate(ranked):
        o = ds.obs()
        lay = cw.Layout(len(truth))
        cams3, info3 = cw.coarse_fit(cams0, tube, o, tc.L_WAND, lay)
        s4 = cw.refine(cams3, tube, cw.initial_frames(cams3, tube, o, tc.L_WAND), tc.L_WAND, o, lay)
        c4, _f, flipped = cw.canonical(s4.cams, s4.frames)
        s4.cams = c4
        z = tc.zmax_between(s4, ref)
        ph = tc.physical(s4.cams, ref.cams)
        rec = {"rank": k, "bootstrap_cost": cost, "stage3_rms_px": info3["rms_px"], "rms_px": s4.rms,
               "active": int(o.active.sum()), "max_diff_sigma": z, "flipped": bool(flipped),
               "f_diff_pct": [100 * p["f_rel"] for p in ph], "distance_diff_mm": [p["distance_mm"] for p in ph],
               "azimuth_diff_deg": [p["azimuth_deg"] for p in ph]}
        starts.append(rec)
        log(f"  start {k}: bootstrap cost {cost:.4g} -> stage 3 {info3['rms_px']:.3f} px -> stage 4 {s4.rms:.4f} px, "
            f"active {rec['active']} | vs truth-start: {z:.2g} sigma, f diff "
            f"{', '.join(f'{v:+.2f}' for v in rec['f_diff_pct'])} %, distance diff "
            f"{', '.join(f'{v:+.1f}' for v in rec['distance_diff_mm'])} mm"
            + ("  <- kept by calibrate (n_final = 2)" if k < 2 else ""))
    best = min(starts, key=lambda r: (-r["active"], r["rms_px"]))
    out = {"truth_start_rms_px": ref.rms, "scratch_rms_px": sol.rms, "scratch_max_diff_sigma": tc.zmax_between(sol, ref),
           "bootstrap": binfo, "starts": starts, "best_of_all_starts": best["rank"],
           "global_found_by_any": any(r["max_diff_sigma"] < 0.01 for r in starts),
           "global_found_in_top2": any(r["max_diff_sigma"] < 0.01 for r in starts[:2])}
    ph = tc.physical(sol.cams, ref.cams)
    out["wrong_minimum"] = {"f_diff_pct": [100 * p["f_rel"] for p in ph], "distance_diff_mm": [p["distance_mm"] for p in ph],
                            "azimuth_diff_deg": [p["azimuth_deg"] for p in ph], "height_diff_mm": [p["height_mm"] for p in ph],
                            "rms_excess_px": sol.rms - ref.rms}
    log("")
    log(f"the wrong minimum vs the global one: f {', '.join(f'{v:+.2f}' for v in out['wrong_minimum']['f_diff_pct'])} %, "
        f"distance {', '.join(f'{v:+.1f}' for v in out['wrong_minimum']['distance_diff_mm'])} mm, azimuth "
        f"{', '.join(f'{v:+.2f}' for v in out['wrong_minimum']['azimuth_diff_deg'])} deg; rms higher by "
        f"{1e3 * out['wrong_minimum']['rms_excess_px']:.1f} mpx")
    log(f"global minimum reached by any of the six starts: {out['global_found_by_any']} (in the top two: "
        f"{out['global_found_in_top2']}); the start calibrate's rule would pick from all six: {best['rank']}")
    out["seconds"] = time.time() - t0
    log(f"time {out['seconds']:.0f} s")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
