"""T13b: the detectors on rendered wand images through the tube, against the exact image of each ball's centre.  Plan:
2026-10-02-ball-centres-plan.md.

The cam1-like rig (three cam1 replicas at 0/45/90 deg, 1280 x 800 px) sees 400 wand frames (midpoints in +-8 mm,
random directions).  Every camera that sees both balls whole renders each frame:
  - in the four lightings (backlit, even, side, highlight);
  - at two focus settings: "best" (the volume centred between the tube's two foci, 18 mm of the plate case's
    traverse) and "defocused" (centred at 40 mm, where both directions blur);
  - each with noise (2 grey levels) and without (the noise-free image gives the detector's systematic error at
    that geometry).
Every detector measures every image, except pyOpenLPT's C++ "reliable" detector.  It takes about 6 s on a full
frame, so it measures every fourth frame, at best focus only, on a crop around the wand.  The detections, smallest
first, are labelled A and B and compared with the truth (`t13_common.assign`, gate 5 px).
Reported per detector, lighting and focus:
  - the share of camera-frames measured (both balls within 5 px, right labels), swapped, wrong or missing;
  - the bias (mean error) and RMS of the measured centres, with noise;
  - their systematic part (noise-free) and random part (with noise minus without);
  - the largest error;
  - the errors of the nearest detection to each ball whatever its label (how far off the failures are).
Frames where the two balls' images come within 10 px of each other ("crowded") or overlap are reported apart.
Criteria for the recommended detector, lit backlit or even:
  - |bias| <= 0.02 px and RMS <= 0.05 px;
  - A and B told apart in every uncrowded frame;
  - no wrong centre (error > 1 px) returned for overlapping balls.
Outputs: t13b_detectors_out.txt, t13b_detectors.json, t13b_detectors_rows.npz.
"""
import t13_common as T  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402

OUT = rigs.HERE / "t13b_detectors"
log = rigs.Tee(str(OUT) + "_out.txt")
N_FRAMES = 400
CHUNK = 25
FOCI = {"best": T.FOCUS_Z_MM, "defocused": T.DEFOCUS_Z_MM}
SEED = 1302


def nearest(dets, t):
    if not dets:
        return np.array([np.nan, np.nan])
    P = np.array([d[:2] for d in dets])
    e = P - t[None, :]
    k = int(np.argmin(np.hypot(e[:, 0], e[:, 1])))
    return e[k] if np.hypot(*e[k]) < 20.0 else np.array([np.nan, np.nan])


def task(args):
    k, XA, XB, UV = args
    rng = np.random.default_rng(SEED + 1000 * k)
    cams = T.cam1_rig()
    rows = []
    tdet = {n: [0.0, 0] for n in T.DETECTOR_NAMES}
    for f in range(len(XA)):
        for c, cam in enumerate(cams):
            if not np.all(np.isfinite(UV[f, c])):
                continue
            tA, tB = UV[f, c, 0], UV[f, c, 1]
            geo = T.render_geometry(cam, XA[f], XB[f], tA, tB)
            crop = T.crop_around(geo, cam)
            gap = T.gap_px(tA, tB)
            for fi, (fname, fz) in enumerate(FOCI.items()):
                bA, bB = T.ball_blur(cam, XA[f], fz), T.ball_blur(cam, XB[f], fz)
                for li, light in enumerate(T.LIGHTINGS):
                    pol = T.POLARITY[light]
                    noisy = T.compose(cam, geo, light, bA, bB, rng=rng)
                    clean = T.compose(cam, geo, light, bA, bB)
                    for di, name in enumerate(T.DETECTOR_NAMES):
                        if name == "pyopenlpt_reliable" and (fname != "best" or (k * CHUNK + f) % 4):
                            continue
                        t0 = time.time()
                        dn = T.detect(name, noisy, pol, crop=crop)
                        tdet[name][0] += time.time() - t0
                        tdet[name][1] += 1
                        dc = T.detect(name, clean, pol, crop=crop)
                        sn, eAn, eBn = T.assign(dn, tA, tB)
                        sc, eAc, eBc = T.assign(dc, tA, tB)
                        rows.append([k * CHUNK + f, c, fi, li, di, ["ok", "swapped", "wrong", "missing"].index(sn),
                                     ["ok", "swapped", "wrong", "missing"].index(sc), *eAn, *eBn, *eAc, *eBc,
                                     *nearest(dn, tA), *nearest(dn, tB), gap, *tA, *tB, *bA, *bB])
    return {"rows": rows, "tdet": tdet, "line": f"{len(rows)} detections"}


COLS = ["frame", "cam", "focus", "light", "det", "st_noisy", "st_clean", "eAu", "eAv", "eBu", "eBv", "cAu", "cAv",
        "cBu", "cBv", "nAu", "nAv", "nBu", "nBv", "gap", "tAu", "tAv", "tBu", "tBv", "sAu", "sAv", "sBu", "sBv"]


def stats(R, ci, crowd):
    """Statistics of one (detector, lighting, focus) block of rows R (columns index ci)."""
    unc = R[:, ci["gap"]] >= 10.0 if not crowd else np.ones(len(R), bool)
    Ru = R[unc]
    st = Ru[:, ci["st_noisy"]].astype(int)
    n = len(Ru)
    out = {"camera_frames": n, "ok": int(np.sum(st == 0)), "swapped": int(np.sum(st == 1)),
           "wrong": int(np.sum(st == 2)), "missing": int(np.sum(st == 3))}
    okm = (st == 0) & (Ru[:, ci["st_clean"]].astype(int) == 0)
    E = np.concatenate([Ru[okm][:, [ci["eAu"], ci["eAv"]]], Ru[okm][:, [ci["eBu"], ci["eBv"]]]])
    C = np.concatenate([Ru[okm][:, [ci["cAu"], ci["cAv"]]], Ru[okm][:, [ci["cBu"], ci["cBv"]]]])
    if len(E):
        out.update({"bias_px": E.mean(axis=0).tolist(), "rms_px": float(np.sqrt(np.mean(np.sum(E ** 2, axis=1)))),
                    "max_px": float(np.max(np.hypot(E[:, 0], E[:, 1]))),
                    "sys_bias_px": C.mean(axis=0).tolist(),
                    "sys_rms_px": float(np.sqrt(np.mean(np.sum(C ** 2, axis=1)))),
                    "sys_max_px": float(np.max(np.hypot(C[:, 0], C[:, 1]))),
                    "random_rms_px": float(np.sqrt(np.mean(np.sum((E - C) ** 2, axis=1)))),
                    "n_balls": int(len(E))})
        # A and B separately (systematic part)
        for lab, cols in (("A", ("cAu", "cAv")), ("B", ("cBu", "cBv"))):
            Cb = Ru[okm][:, [ci[cols[0]], ci[cols[1]]]]
            out[f"sys_bias_{lab}_px"] = Cb.mean(axis=0).tolist()
    # nearest detection to each ball, whatever its label: how far off a detector is when it fails
    Nn = np.concatenate([Ru[:, [ci["nAu"], ci["nAv"]]], Ru[:, [ci["nBu"], ci["nBv"]]]])
    d = np.hypot(Nn[:, 0], Nn[:, 1])
    fin = np.isfinite(d)
    out["nearest_found"] = int(fin.sum())
    if fin.any():
        out["nearest_median_px"] = float(np.median(d[fin]))
        out["nearest_p90_px"] = float(np.percentile(d[fin], 90))
        out["nearest_max_px"] = float(np.max(d[fin]))
        out["nearest_over_1px"] = int(np.sum(d[fin] > 1.0))
    return out


def main():
    t0 = time.time()
    cams = T.cam1_rig()
    rng = np.random.default_rng(SEED)
    XA, XB, UV = T.wand_frames(cams, N_FRAMES, rng)
    seen = int(np.sum(np.all(np.isfinite(UV[:, :, :, 0]), axis=2)))
    log("T13b: detectors on rendered wand images through the tube (cam1-like rig, 1280 x 800 px)")
    log(f"  {N_FRAMES} frames, {seen} camera-frames with both balls whole; 4 lightings x 2 focus settings; each image "
        f"with and without noise; {len(T.DETECTOR_NAMES)} detectors")
    tasks = [(k, XA[k * CHUNK:(k + 1) * CHUNK], XB[k * CHUNK:(k + 1) * CHUNK], UV[k * CHUNK:(k + 1) * CHUNK])
             for k in range(math.ceil(N_FRAMES / CHUNK))]
    res = T.tc.run_parallel(task, tasks, log, label=lambda t: f"frames {t[0] * CHUNK}-{t[0] * CHUNK + len(t[1]) - 1}")
    bad = [r for r in res if "error" in r]
    if bad:
        log(bad[0]["error"])
        raise SystemExit("a task failed")
    R = np.array([row for r in res for row in r["rows"]], float)
    ci = {c: i for i, c in enumerate(COLS)}
    np.savez_compressed(str(OUT) + "_rows.npz", rows=R, cols=np.array(COLS))
    tdet = {n: sum(r["tdet"][n][0] for r in res) / max(sum(r["tdet"][n][1] for r in res), 1)
            for n in T.DETECTOR_NAMES}
    gaps = R[(R[:, ci["det"]] == 0) & (R[:, ci["light"]] == 0) & (R[:, ci["focus"]] == 0), ci["gap"]]
    log(f"  camera-frames: {len(gaps)}; crowded (gap < 10 px): {int(np.sum(gaps < 10))}, of which overlapping: "
        f"{int(np.sum(gaps < 0))}")
    log("  time per image: " + ", ".join(f"{n} {1e3 * tdet[n]:.0f} ms" for n in T.DETECTOR_NAMES))
    out = {"n_frames": N_FRAMES, "camera_frames": int(len(gaps)), "crowded": int(np.sum(gaps < 10)),
           "overlapping": int(np.sum(gaps < 0)), "time_per_image_s": tdet, "blocks": {}}
    for fi, fname in enumerate(FOCI):
        for li, light in enumerate(T.LIGHTINGS):
            log("")
            log(f"Focus {fname}, lighting {light} (uncrowded camera-frames; errors in px)")
            log(f"  {'detector':<19s} {'measured':>9s} {'swap':>5s} {'wrong':>5s} {'miss':>5s} "
                f"{'bias u, v':>17s} {'RMS':>7s} {'max':>7s} | {'systematic: bias u, v':>22s} {'RMS':>7s} "
                f"{'max':>7s} | {'random':>7s} | {'nearest: median':>15s} {'p90':>6s} {'max':>7s}")
            for di, name in enumerate(T.DETECTOR_NAMES):
                sel = (R[:, ci["focus"]] == fi) & (R[:, ci["light"]] == li) & (R[:, ci["det"]] == di)
                s = stats(R[sel], ci, crowd=False)
                Rc = R[sel & (R[:, ci["gap"]] < 10.0)]
                stc = stats(Rc, ci, crowd=True) if len(Rc) else None
                Ro = R[sel & (R[:, ci["gap"]] < 0.0)]
                viol = 0
                for row in Ro:
                    stn = int(row[ci["st_noisy"]])
                    if stn in (1, 2):
                        viol += 1
                    elif stn == 0 and max(math.hypot(row[ci["eAu"]], row[ci["eAv"]]),
                                          math.hypot(row[ci["eBu"]], row[ci["eBv"]])) > 1.0:
                        viol += 1
                s["crowded"] = stc
                s["overlap_wrong"] = viol
                s["overlap_n"] = int(len(Ro))
                out["blocks"][f"{fname}/{light}/{name}"] = s
                if "bias_px" in s:
                    log(f"  {name:<19s} {s['ok']:>4d}/{s['camera_frames']:<4d} {s['swapped']:>5d} {s['wrong']:>5d} "
                        f"{s['missing']:>5d} {s['bias_px'][0]:>+8.4f},{s['bias_px'][1]:>+8.4f} {s['rms_px']:>7.4f} "
                        f"{s['max_px']:>7.3f} | {s['sys_bias_px'][0]:>+10.4f},{s['sys_bias_px'][1]:>+10.4f} "
                        f"{s['sys_rms_px']:>7.4f} {s['sys_max_px']:>7.3f} | {s['random_rms_px']:>7.4f} | "
                        f"{s.get('nearest_median_px', float('nan')):>15.3f} {s.get('nearest_p90_px', float('nan')):>6.2f} "
                        f"{s.get('nearest_max_px', float('nan')):>7.2f}")
                else:
                    log(f"  {name:<19s} {s['ok']:>4d}/{s['camera_frames']:<4d} {s['swapped']:>5d} {s['wrong']:>5d} "
                        f"{s['missing']:>5d}   (none measured) | nearest: median "
                        f"{s.get('nearest_median_px', float('nan')):.3f}, p90 {s.get('nearest_p90_px', float('nan')):.2f}, "
                        f"max {s.get('nearest_max_px', float('nan')):.2f}")
            log("  crowded camera-frames (gap < 10 px): "
                + "; ".join(f"{n}: {out['blocks'][f'{fname}/{light}/{n}']['crowded']['ok']}/"
                            f"{out['blocks'][f'{fname}/{light}/{n}']['crowded']['camera_frames']} measured, "
                            f"{out['blocks'][f'{fname}/{light}/{n}']['overlap_wrong']}/"
                            f"{out['blocks'][f'{fname}/{light}/{n}']['overlap_n']} overlapping with a wrong centre"
                            for n in T.DETECTOR_NAMES if out['blocks'][f'{fname}/{light}/{n}']['crowded']))
    # verdict for each detector: the criteria with lighting backlit and even, best focus
    log("")
    log("Criteria (best focus, lightings backlit and even): |bias| <= 0.02 px, RMS <= 0.05 px, no swap in an "
        "uncrowded frame, no wrong centre for overlapping balls")
    verdict = {}
    for name in T.DETECTOR_NAMES:
        checks = []
        for light in ("backlit", "even"):
            s = out["blocks"][f"best/{light}/{name}"]
            if "bias_px" not in s:
                checks.append(False)
                continue
            checks += [max(abs(s["bias_px"][0]), abs(s["bias_px"][1])) <= 0.02, s["rms_px"] <= 0.05,
                       s["swapped"] == 0 and s["wrong"] == 0, s["overlap_wrong"] == 0]
        verdict[name] = bool(all(checks))
        log(f"  {name:<19s} {'PASS' if verdict[name] else 'fail'}")
    out["verdict"] = verdict
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log("")
    log(f"done in {time.time() - t0:.0f} s")
    log.close()


if __name__ == "__main__":
    main()
