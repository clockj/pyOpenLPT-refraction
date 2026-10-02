"""T13b2: a guard against overlapping balls, for the best detector of T13b (ellipse_rodmask).  Plan:
2026-10-02-ball-centres-plan.md (added after T13b; see the report).

T13b's ellipse_rodmask met the bias and RMS criteria, but in front light it returned a centre more than 1 px off for
1-3 of the 17 camera-frames where the two balls overlap.  The guard: when the two fitted ellipses come within GUARD_PX
of each other along the line joining their centres, the camera's pair is dropped (no centres).
The same frames as T13b (seed 1302) are re-rendered at best focus in the four lightings, with new noise:
  - every crowded camera-frame (true gap < 10 px, 22 of them; 17 overlap);
  - 150 uncrowded ones drawn at random, to check that the guard drops nothing it should keep.
Criteria: no centre more than 1 px off returned for overlapping balls; no uncrowded pair dropped.
Outputs: t13b2_overlap_guard_out.txt, t13b2_overlap_guard.json.
"""
import t13_common as T  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import t13_ellipse as te  # noqa: E402

OUT = rigs.HERE / "t13b2_overlap_guard"
log = rigs.Tee(str(OUT) + "_out.txt")
GUARD_PX = 3.0


def ellipse_radius_dir(e, phi):
    """Radius of a fitted ellipse e = (u, v, area, a, b, theta, ...) in the image direction phi."""
    a, b, th = e[3], e[4], e[5]
    return 1.0 / math.sqrt((math.cos(phi - th) / a) ** 2 + (math.sin(phi - th) / b) ** 2)


def ellipse_gap(e1, e2):
    du, dv = e2[0] - e1[0], e2[1] - e1[1]
    phi = math.atan2(dv, du)
    return math.hypot(du, dv) - ellipse_radius_dir(e1, phi) - ellipse_radius_dir(e2, phi + math.pi)


def detect_guarded(img, pol):
    r = te.detect_balls(img, pol, T.OPEN_PX, rod_mask_px=T.ROD_MASK_PX)
    if len(r) >= 2 and ellipse_gap(r[0], r[1]) < GUARD_PX:
        return [], r
    return [(x[0], x[1], x[2]) for x in r], r


def main():
    t0 = time.time()
    cams = T.cam1_rig()
    rng = np.random.default_rng(1302)
    XA, XB, UV = T.wand_frames(cams, 400, rng)            # T13b's frames
    cf = [(f, c) for f in range(len(XA)) for c in range(3) if np.all(np.isfinite(UV[f, c]))]
    gaps = np.array([T.gap_px(UV[f, c, 0], UV[f, c, 1]) for f, c in cf])
    crowded = [cf[i] for i in np.flatnonzero(gaps < 10.0)]
    pick = np.random.default_rng(7).choice(np.flatnonzero(gaps >= 10.0), 150, replace=False)
    normal = [cf[i] for i in pick]
    log("T13b2: a guard against overlapping balls for ellipse_rodmask (pairs closer than "
        f"{GUARD_PX:g} px are dropped)")
    log(f"  T13b's frames: {len(crowded)} crowded camera-frames (gap < 10 px; {int(np.sum(gaps < 0))} overlap) and "
        f"{len(normal)} uncrowded ones; best focus; four lightings; new noise")
    nrng = np.random.default_rng(13021)
    res = {}
    for light in T.LIGHTINGS:
        pol = T.POLARITY[light]
        rec = {"crowded": {"n": 0, "overlap": 0, "wrong_without": 0, "wrong_with": 0, "measured_without": 0,
                           "measured_with": 0},
               "normal": {"n": 0, "dropped": 0, "measured_without": 0, "errors_without": [], "errors_with": []}}
        for group, items in (("crowded", crowded), ("normal", normal)):
            for f, c in items:
                cam = cams[c]
                tA, tB = UV[f, c, 0], UV[f, c, 1]
                geo = T.render_geometry(cam, XA[f], XB[f], tA, tB)
                img = T.compose(cam, geo, light, T.ball_blur(cam, XA[f]), T.ball_blur(cam, XB[f]), rng=nrng)
                with_g, raw = detect_guarded(img, pol)
                without = [(x[0], x[1], x[2]) for x in raw]
                g = T.gap_px(tA, tB)
                r = rec[group]
                r["n"] += 1
                if group == "crowded":
                    over = g < 0.0
                    r["overlap"] += int(over)
                    for key, dets in (("without", without), ("with", with_g)):
                        st, eA, eB = T.assign(dets, tA, tB)
                        r["measured_" + key] += int(st == "ok")
                        bad = st in ("swapped", "wrong") or (st == "ok" and max(np.hypot(*eA), np.hypot(*eB)) > 1.0)
                        if over and bad:
                            r["wrong_" + key] += 1
                else:
                    st0, eA0, eB0 = T.assign(without, tA, tB)
                    st1, eA1, eB1 = T.assign(with_g, tA, tB)
                    r["measured_without"] += int(st0 == "ok")
                    r["dropped"] += int(st0 == "ok" and st1 != "ok")
                    if st0 == "ok":
                        r["errors_without"] += [float(np.hypot(*eA0)), float(np.hypot(*eB0))]
                    if st1 == "ok":
                        r["errors_with"] += [float(np.hypot(*eA1)), float(np.hypot(*eB1))]
        cr, nr = rec["crowded"], rec["normal"]
        log(f"  {light:<9s} crowded {cr['n']} (overlapping {cr['overlap']}): wrong centres for overlapping balls "
            f"{cr['wrong_without']} without the guard, {cr['wrong_with']} with it; measured {cr['measured_without']} "
            f"-> {cr['measured_with']} | uncrowded {nr['n']}: measured {nr['measured_without']}, dropped by the guard "
            f"{nr['dropped']}")
        rec["normal"]["rms_without_px"] = float(np.sqrt(np.mean(np.square(nr["errors_without"])))) \
            if nr["errors_without"] else None
        res[light] = rec
    ok = all(res[li]["crowded"]["wrong_with"] == 0 and res[li]["normal"]["dropped"] == 0 for li in T.LIGHTINGS)
    log(f"  {'PASS' if ok else 'FAIL'}: no wrong centre for overlapping balls and no uncrowded pair dropped, in all "
        "four lightings")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump({"guard_px": GUARD_PX, "results": res, "pass": ok}, fh, indent=1)
    log(f"done in {time.time() - t0:.0f} s")
    log.close()


if __name__ == "__main__":
    main()
