"""T13b3: the rod-masked ellipse estimator with a symmetric mask, against the one-sided mask.  Plan:
2026-10-02-ball-centres-plan.md (added after T13c; see the report).

T13c found a small f bias (+0.03 to +0.07 %, t up to 3.5) and error bars 1.5x too small with ellipse_rodmask, from its
systematic error of about 0.01 px.  A suspected cause: leaving out only the strip towards the rod makes the edge
points lopsided, so the angle-dependent inward shift of a blurred outline's half-height crossing moves the fitted
centre.  ellipse_sym also leaves out the opposite strip, and adds T13b2's overlap guard; ellipse_sym2 is the same
with 192 rays and fits from 25 % of them (ellipse_sym measured only a quarter of the frames: too few points left).
The first 150 of T13b's frames (seed 1302) are rendered at best focus in the four lightings, with and without noise,
and both detectors measure them, as in T13b.
Outputs: t13b3_symmetric_out.txt, t13b3_symmetric.json.
"""
import t13_common as T  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import t13b_detectors as B  # noqa: E402

OUT = rigs.HERE / "t13b3_symmetric"
log = rigs.Tee(str(OUT) + "_out.txt")
DETS = ("ellipse_rodmask", "ellipse_sym", "ellipse_sym2")
N_FRAMES = 150
CHUNK = 5


def task(args):
    k, XA, XB, UV = args
    rng = np.random.default_rng(1303000 + k)
    cams = T.cam1_rig()
    rows = []
    for f in range(len(XA)):
        for c, cam in enumerate(cams):
            if not np.all(np.isfinite(UV[f, c])):
                continue
            tA, tB = UV[f, c, 0], UV[f, c, 1]
            geo = T.render_geometry(cam, XA[f], XB[f], tA, tB)
            gap = T.gap_px(tA, tB)
            bA, bB = T.ball_blur(cam, XA[f]), T.ball_blur(cam, XB[f])
            for li, light in enumerate(T.LIGHTINGS):
                pol = T.POLARITY[light]
                noisy = T.compose(cam, geo, light, bA, bB, rng=rng)
                clean = T.compose(cam, geo, light, bA, bB)
                for di, name in enumerate(DETS):
                    dn, dc = T.detect(name, noisy, pol), T.detect(name, clean, pol)
                    sn, eAn, eBn = T.assign(dn, tA, tB)
                    sc, eAc, eBc = T.assign(dc, tA, tB)
                    st = ["ok", "swapped", "wrong", "missing"]
                    rows.append([k * CHUNK + f, c, 0, li, di, st.index(sn), st.index(sc), *eAn, *eBn, *eAc, *eBc,
                                 *B.nearest(dn, tA), *B.nearest(dn, tB), gap, *tA, *tB, *bA, *bB])
    return {"rows": rows, "line": f"{len(rows)} detections"}


def main():
    t0 = time.time()
    cams = T.cam1_rig()
    XA, XB, UV = T.wand_frames(cams, 400, np.random.default_rng(1302))      # T13b's frames
    XA, XB, UV = XA[:N_FRAMES], XB[:N_FRAMES], UV[:N_FRAMES]
    log("T13b3: one-sided against symmetric rod mask (ellipse_rodmask vs ellipse_sym, the latter with the overlap "
        "guard); T13b's first 150 frames, best focus")
    tasks = [(k, XA[k * CHUNK:(k + 1) * CHUNK], XB[k * CHUNK:(k + 1) * CHUNK], UV[k * CHUNK:(k + 1) * CHUNK])
             for k in range(math.ceil(N_FRAMES / CHUNK))]
    res = T.tc.run_parallel(task, tasks, log, label=lambda t: f"frames {t[0] * CHUNK}-", workers=30)
    bad = [r for r in res if "error" in r]
    if bad:
        log(bad[0]["error"])
        raise SystemExit("a task failed")
    R = np.array([row for r in res for row in r["rows"]], float)
    ci = {c: i for i, c in enumerate(B.COLS)}
    out = {}
    log("")
    log(f"  {'lighting':<9s} {'detector':<16s} {'measured':>9s} {'bias u, v':>17s} {'RMS':>7s} {'max':>6s} | "
        f"{'systematic RMS':>14s} {'per camera (u, v) mean':>40s} | {'random':>7s}")
    for li, light in enumerate(T.LIGHTINGS):
        for di, name in enumerate(DETS):
            sel = (R[:, ci["light"]] == li) & (R[:, ci["det"]] == di)
            s = B.stats(R[sel], ci, crowd=False)
            percam = []
            for c in range(3):
                m = sel & (R[:, ci["cam"]] == c) & (R[:, ci["st_clean"]] == 0) & (R[:, ci["gap"]] >= 10)
                C = np.concatenate([R[m][:, [ci["cAu"], ci["cAv"]]], R[m][:, [ci["cBu"], ci["cBv"]]]])
                percam.append(C.mean(axis=0).tolist() if len(C) else [float("nan")] * 2)
            s["sys_mean_per_camera_px"] = percam
            out[f"{light}/{name}"] = s
            if "bias_px" in s:
                pc = " ".join(f"({p[0]:+.3f},{p[1]:+.3f})" for p in percam)
                log(f"  {light:<9s} {name:<16s} {s['ok']:>4d}/{s['camera_frames']:<4d} {s['bias_px'][0]:>+8.4f},"
                    f"{s['bias_px'][1]:>+8.4f} {s['rms_px']:>7.4f} {s['max_px']:>6.3f} | {s['sys_rms_px']:>14.4f} "
                    f"{pc:>40s} | {s['random_rms_px']:>7.4f}")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log("")
    log(f"done in {time.time() - t0:.0f} s")
    log.close()


if __name__ == "__main__":
    main()
