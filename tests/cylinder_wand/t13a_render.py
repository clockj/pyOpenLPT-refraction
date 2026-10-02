"""T13a: the renderer of wand images through the tube, and the detectors' pixel convention.  Plan:
2026-10-02-ball-centres-plan.md.

1. Renderer against the exact silhouette.  A single ball (no rod, no blur, no noise) is rendered at 6 random positions
   in +-8 mm, for each camera of the cam1-like rig and both ball sizes.  The area centroid of its rendered coverage is
   compared with the exact centroid of its silhouette: the polygon of 1440 outline points found by bisection, as in
   t13_pre_sizing.py.  Supersampling SS = 4, 8, 16, 32.  Criterion: within 1e-3 px at SS = 32.  The error at the
   production SS is reported as the renderer's own floor.
2. The detectors' pixel convention, without the tube.  Two discs (radii 20 and 30 px), and then two ellipses
   1.3 : 1 (26 x 20 and 39 x 30 px), are drawn at random sub-pixel positions: supersampled 16 x 16, blurred
   (sigma 1.5 px), no noise, both polarities, 20 images each.  The error of each detector is reported.  On discs every
   detector should be within a few 0.01 px; a constant offset would be a convention difference, not a tube effect.
   2b. How the detectors degrade with the aspect ratio: ellipses a x 20 and 1.5 a x 30 px for a / b = 1.0 to 1.4 (the
   tube gives 1.26-1.34), dark on bright, blur 1.5 px, no noise, 10 images each.  Reported for each detector:
   the images measured (both balls within 5 px, right labels), and the median and largest distance from each ball
   to the nearest detection, whatever its label.
3. Example images: one wand frame, the three cameras, the four lightings, with noise (t13a_examples/*.png; crops at
   2x, plus one full frame).
Outputs: t13a_render_out.txt, t13a_render.json, t13a_examples/.
"""
import t13_common as T  # first: one BLAS thread per process

import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import rigs  # noqa: E402
from rigs import cm  # noqa: E402
import t13_pre_sizing as pre  # noqa: E402

OUT = rigs.HERE / "t13a_render"
EX = rigs.HERE / "t13a_examples"
log = rigs.Tee(str(OUT) + "_out.txt")


def exact_centroid(cam, cc, X, rho):
    p0 = cm.cpp_project(cc, X[None])[0]
    rw1 = cm.rows([cam], [0])
    P = pre.silhouette(cam, rw1, T.TUBE, X, rho, p0, 60.0)
    return p0 + pre.polygon_centroid(P), p0


def part1(cams, rng):
    log("1. Renderer against the exact silhouette (single ball, no rod, no blur, no noise)")
    cpp = [cm.to_cpp(c, T.TUBE) for c in cams]
    pos = rng.uniform(-6.0, 6.0, size=(6, 3))
    rows = []
    for ss in (4, 8, 16, 32):
        errs = []
        for X in pos:
            for c, (cam, cc) in enumerate(zip(cams, cpp)):
                for k, rho in enumerate(T.RHO):
                    ex, p0 = exact_centroid(cam, cc, X, rho)
                    far = X + np.array([0.0, 40.0, 0.0])          # the other ball, far off the window
                    XA, XB = (X, far) if k == 0 else (far, X)
                    geo = T.render_geometry(cam, XA, XB, p0, p0, ss=ss, r_rod=0.0)
                    got = T.coverage_centroid(geo, k + 1)
                    errs.append(got - ex)
        errs = np.array(errs)
        mx = float(np.max(np.hypot(errs[:, 0], errs[:, 1])))
        rms = float(np.sqrt(np.mean(np.sum(errs ** 2, axis=1))))
        rows.append({"ss": ss, "max_px": mx, "rms_px": rms, "mean_px": errs.mean(axis=0).tolist(), "n": len(errs)})
        log(f"  SS {ss:>2d}: rendered centroid - exact: RMS {rms:.5f} px, max {mx:.5f} px, mean "
            f"({errs[:, 0].mean():+.5f}, {errs[:, 1].mean():+.5f}) px over {len(errs)} balls")
    ok = rows[-1]["max_px"] <= 1e-3
    log(f"  {'PASS' if ok else 'FAIL'}: within 1e-3 px at SS 32 (max {rows[-1]['max_px']:.5f} px); the production "
        f"SS {T.SS} leaves {next(r['rms_px'] for r in rows if r['ss'] == T.SS):.4f} px RMS (the renderer's floor)")
    return {"rows": rows, "pass": ok}


def draw_ellipses(W, H, objs, polarity, sigma=1.5, ss=16):
    """objs: [(u, v, a, b), ...] axis-aligned ellipses (a along u).  Supersampled coverage, blurred, 8-bit."""
    cov = np.zeros((H, W))
    off = (np.arange(ss) + 0.5) / ss - 0.5
    for (u, v, a, b) in objs:
        u0, u1 = int(math.floor(u - a - 2)), int(math.ceil(u + a + 2))
        v0, v1 = int(math.floor(v - b - 2)), int(math.ceil(v + b + 2))
        uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        su = uu[..., None, None] + off[None, None, None, :]
        sv = vv[..., None, None] + off[None, None, :, None]
        inside = ((su - u) / a) ** 2 + ((sv - v) / b) ** 2 <= 1.0
        cov[v0:v1 + 1, u0:u1 + 1] += inside.mean(axis=(2, 3))
    bg, ob = (140.0, 30.0) if polarity == "dark" else (20.0, 130.0)
    img = bg + (ob - bg) * cov
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def part2(rng):
    log("")
    log("2. The detectors' pixel convention, without the tube (no noise, blur 1.5 px, 20 images per case)")
    res = {}
    for shape, sizes in (("discs", ((20, 20), (30, 30))), ("ellipses 1.3:1", ((26, 20), (39, 30)))):
        for pol in ("dark", "bright"):
            E = {n: [] for n in T.DETECTOR_NAMES}
            fails = {n: 0 for n in T.DETECTOR_NAMES}
            for _ in range(20):
                cA = np.array([150.0, 150.0]) + rng.uniform(-0.5, 0.5, 2)
                cB = np.array([300.0, 150.0]) + rng.uniform(-0.5, 0.5, 2)
                img = draw_ellipses(450, 300, [(cA[0], cA[1], *sizes[0]), (cB[0], cB[1], *sizes[1])], pol)
                for n in T.DETECTOR_NAMES:
                    st, eA, eB = T.assign(T.detect(n, img, pol), cA, cB)
                    if st == "ok":
                        E[n] += [eA, eB]
                    else:
                        fails[n] += 1
            for n in T.DETECTOR_NAMES:
                e = np.array(E[n]).reshape(-1, 2)
                key = f"{shape} / {pol} / {n}"
                if len(e):
                    rec = {"mean": e.mean(axis=0).tolist(), "rms": float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
                           "max": float(np.max(np.hypot(e[:, 0], e[:, 1]))), "failed_images": fails[n]}
                else:
                    rec = {"mean": [float("nan")] * 2, "rms": float("nan"), "max": float("nan"),
                           "failed_images": fails[n]}
                res[key] = rec
                log(f"  {shape:<15s} {pol:<6s} {n:<19s} mean ({rec['mean'][0]:+.4f}, {rec['mean'][1]:+.4f}) px  "
                    f"RMS {rec['rms']:.4f}  max {rec['max']:.4f}  failed {fails[n]}/20")
    return res


def part2b(rng):
    log("")
    log("2b. The detectors against the aspect ratio, without the tube (ellipses a x 20 and 1.5 a x 30 px, dark on "
        "bright, blur 1.5 px, no noise, 10 images each)")
    log(f"  {'a / b':>5s}  " + "  ".join(f"{n:>26s}" for n in T.DETECTOR_NAMES))
    log(f"  {'':>5s}  " + "  ".join(f"{'measured, nearest med/max':>26s}" for _ in T.DETECTOR_NAMES))
    res = {}
    for asp in (1.0, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4):
        cells = []
        for n in T.DETECTOR_NAMES:
            ok, near = 0, []
            for _ in range(10):
                cA = np.array([150.0, 150.0]) + rng.uniform(-0.5, 0.5, 2)
                cB = np.array([300.0, 150.0]) + rng.uniform(-0.5, 0.5, 2)
                img = draw_ellipses(450, 300, [(cA[0], cA[1], 20 * asp, 20), (cB[0], cB[1], 30 * asp, 30)], "dark")
                dets = T.detect(n, img, "dark")
                st, _eA, _eB = T.assign(dets, cA, cB)
                ok += st == "ok"
                P = np.array([d[:2] for d in dets]).reshape(-1, 2)
                for t in (cA, cB):
                    near.append(float(np.min(np.hypot(*(P - t).T))) if len(P) else float("inf"))
            near = np.array(near)
            res[f"{asp}/{n}"] = {"measured": int(ok), "nearest_median_px": float(np.median(near)),
                                 "nearest_max_px": float(np.max(near))}
            cells.append(f"{ok:>2d}/10, {np.median(near):6.2f} / {np.max(near):6.2f} px")
        log(f"  {asp:>5.2f}  " + "  ".join(f"{c:>26s}" for c in cells))
    return res


def part3(cams, rng):
    log("")
    log("3. Example images: one wand frame, three cameras, four lightings (t13a_examples/)")
    EX.mkdir(exist_ok=True)
    # a frame seen whole by all three cameras with the wand well inclined to every view
    XA, XB, UV = T.wand_frames(cams, 400, rng)
    best, score = None, -1.0
    for f in range(len(XA)):
        if not np.all(np.isfinite(UV[f, :, :, 0])):
            continue
        g = min(T.gap_px(UV[f, c, 0], UV[f, c, 1]) for c in range(3))
        if g > score:
            best, score = f, g
    f = best
    info = {"frame_XA_mm": XA[f].tolist(), "frame_XB_mm": XB[f].tolist(), "images": []}
    for c, cam in enumerate(cams):
        geo = T.render_geometry(cam, XA[f], XB[f], UV[f, c, 0], UV[f, c, 1])
        bA, bB = T.ball_blur(cam, XA[f]), T.ball_blur(cam, XB[f])
        for li in T.LIGHTINGS:
            img = T.compose(cam, geo, li, bA, bB, rng=rng)
            u0, v0, w, h = geo["window"]
            crop = img[max(v0 - 10, 0):v0 + h + 10, max(u0 - 10, 0):u0 + w + 10]
            name = f"cam{c + 1}_{li}.png"
            cv2.imwrite(str(EX / name), cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            if c == 0 and li == "backlit":
                cv2.imwrite(str(EX / "cam1_backlit_full_frame.png"), img)
            info["images"].append({"file": name, "camera": c + 1, "lighting": li,
                                   "true_centres_px": UV[f, c].tolist(), "blur_A_px": bA, "blur_B_px": bB})
        log(f"  camera {c + 1}: ball A at ({UV[f, c, 0, 0]:.1f}, {UV[f, c, 0, 1]:.1f}) px, blur "
            f"({bA[0]:.2f}, {bA[1]:.2f}) px; ball B at ({UV[f, c, 1, 0]:.1f}, {UV[f, c, 1, 1]:.1f}) px, blur "
            f"({bB[0]:.2f}, {bB[1]:.2f}) px")
    log(f"  written: {len(info['images'])} crops (2x, nearest neighbour) and cam1_backlit_full_frame.png")
    return info


def main():
    t0 = time.time()
    rng = np.random.default_rng(1301)
    cams = T.cam1_rig()
    log("T13a: the renderer, and the detectors' pixel convention")
    log(f"  rig: three cam1 replicas at 0/45/90 deg, {T.SENSOR[0]} x {T.SENSOR[1]} px; balls {2 * T.RHO[0]:g} and "
        f"{2 * T.RHO[1]:g} mm, rod {2 * T.R_ROD:g} mm; blur fitted to the plate case's three states, volume centre at "
        f"{T.FOCUS_Z_MM:g} mm of its traverse")
    zs = [T.FOCUS_Z_MM - 8, T.FOCUS_Z_MM, T.FOCUS_Z_MM + 8]
    log("  blur (sigma_u, sigma_v) px at the volume's near / centre / far depth: "
        + ", ".join(f"({T.blur_sigma(z)[0]:.2f}, {T.blur_sigma(z)[1]:.2f})" for z in zs[::-1]))
    log("  check of the blur fit against the plate case's states: "
        + ", ".join(f"z {z:g}: ({T.blur_sigma(z)[0]:.2f}, {T.blur_sigma(z)[1]:.2f}) vs ({su}, {sv})"
                    for z, su, sv in T.BLUR_STATES))
    log("")
    r1 = part1(cams, rng)
    r2 = part2(rng)
    r2b = part2b(rng)
    r3 = part3(cams, rng)
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump({"renderer": r1, "convention": r2, "aspect": r2b, "examples": r3}, fh, indent=1)
    log("")
    log(f"done in {time.time() - t0:.0f} s")
    log.close()


if __name__ == "__main__":
    main()
