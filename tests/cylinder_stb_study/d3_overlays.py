"""D3 (plan: 2026-10-08-plan.md, Part D): overlays of the detector's settings, bright tracers on black.

    python d3_overlays.py [--other NAME]

Crops (300 x 200 px, shown 3x) of T1's frame i = 100 (raw 400) in both cameras, and of G-all's frame 20 in both cameras
(with the truth: a green dot at each image of a particle inside the view volume seen by both cameras), with the
detections of D1's base setting (magenta) and of OTHER (cyan): D1's "best" if it ran, else the setting with the most
estimated true pairs.  Each detection is drawn as its fitted box, (Lx + 2 sx) x (Ly + 2 sy) px.  The display is
clip(2.2 (image - 6)) as in the HTML report.  Writes d3_overlays.png and d3_overlays_out.txt beside itself.
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

import study
import pairstats as ps
import d1_detect_t1 as d1

CROPS = [(500, 300), (200, 450)]       # top-left corners (u, v)
CW, CH, ZOOM = 300, 200, 3


def show(img):
    d = np.clip((img.astype(np.float32) - 6.0) * 2.2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)


def draw(tile, dets, u0, v0, color):
    for r in dets:
        u, v, Lx, Ly, sx, sy = r[:6]
        w, h = Lx + 2 * sx, Ly + 2 * sy
        a = (int(round((u - u0 - w / 2) * ZOOM)), int(round((v - v0 - h / 2) * ZOOM)))
        b = (int(round((u - u0 + w / 2) * ZOOM)), int(round((v - v0 + h / 2) * ZOOM)))
        cv2.rectangle(tile, a, b, color, 1)


def panel(img, dets_a, dets_b, u0, v0, label, truth=None):
    crop = img[v0:v0 + CH, u0:u0 + CW]
    tile = cv2.resize(show(crop), (CW * ZOOM, CH * ZOOM), interpolation=cv2.INTER_NEAREST)
    if truth is not None:
        for (u, v) in truth:
            if u0 <= u < u0 + CW and v0 <= v < v0 + CH:
                cv2.circle(tile, (int((u - u0) * ZOOM), int((v - v0) * ZOOM)), 3, (80, 220, 80), -1)
    draw(tile, dets_a, u0, v0, (255, 80, 255))
    draw(tile, dets_b, u0, v0, (255, 230, 60))
    cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", default=None)
    args = ap.parse_args()
    lpt, build = study.lib()
    log = study.Log("d3_overlays")
    res = study.load_json(os.path.join(study.HERE, "d1_detect_t1.json"))
    other = args.other
    if other is None:
        if "best" in res.get("results", {}):
            other = "best"
        else:
            other = max((k for k in res["results"] if k != "base"),
                        key=lambda k: res["results"][k]["true_pairs_per_frame"])
    over_o, med_o = (res["best"]["param"], res["best"]["median_px"]) if other == "best" else d1.SETTINGS[other]
    log(f"D3: base (magenta) against {other} (cyan: {over_o}, median {med_o} px); build {build['name']}")
    Pb, Po = ps.line_param(lpt), ps.line_param(lpt, **over_o)
    rows, cache = [], {}
    # T1
    base_imgs = d1.frame_images(100, 31, cache)
    oth_imgs = d1.frame_images(100, med_o, cache) if med_o != 31 else base_imgs
    for c in range(2):
        (ib, nb), (io, no) = base_imgs[c], oth_imgs[c]
        da, db = ps.detect(lpt, ib, nb, Pb), ps.detect(lpt, io, no, Po)
        log(f"  T1 cam{c + 1} frame i = 100: base {len(da)}, {other} {len(db)} detections")
        rows.append([panel(ib, da, db, u0, v0, f"T1 cam{c + 1}, raw frame 400, ({u0}, {v0})") for (u0, v0) in CROPS])
    # G-all
    case_dir = os.path.join(study.DATA, "gall")
    if os.path.isdir(case_dir):
        s = study.load_json(os.path.join(case_dir, "sets.json"))["Gall"]
        imgs, X, layer = ps.truth_frame(case_dir, "Gall", 20)
        both = np.intersect1d(imgs[0][0], imgs[1][0])
        inv = set(both[study.in_volume(X[both], study.view_volume())].tolist())
        for c in range(2):
            path = os.path.join(case_dir, "img", "Gall", f"cam{c + 1}", "img000020.tif")
            img = ps.read8(path)
            if med_o != 31:
                import synth_study
                meta = study.load_json(os.path.join(case_dir, "synth.json"))
                img_o = synth_study.preprocess(synth_study.float_image(case_dir, "Gall", c, 20, meta["noise_seed"]),
                                               s["noise"][c], med_o)
            else:
                img_o = img
            da, db = ps.detect(lpt, img, s["noise"][c], Pb), ps.detect(lpt, img_o, s["noise"][c], Po)
            tr = [uv for i, uv in zip(imgs[c][0], imgs[c][1]) if int(i) in inv]
            log(f"  G-all cam{c + 1} frame 20: base {len(da)}, {other} {len(db)} detections; {len(tr)} images of "
                f"particles in the volume seen by both cameras")
            rows.append([panel(img, da, db, u0, v0, f"G-all cam{c + 1}, frame 20, ({u0}, {v0}); green: truth",
                               truth=tr) for (u0, v0) in CROPS])
    sheet = np.vstack([np.hstack(r) for r in rows])
    cv2.imwrite(os.path.join(study.HERE, "d3_overlays.png"), sheet)
    log(f"  wrote d3_overlays.png ({sheet.shape[1]} x {sheet.shape[0]} px)")
    log.save({"other": other, "param": over_o, "median_px": med_o, "build": build})


if __name__ == "__main__":
    main()
