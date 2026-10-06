"""The case's results page: page_template.html filled with the saved outputs of S1, S2, T1-T7 and D1-D8.

    python make_page.py [--fragment PATH]

Reads only saved results (no calibration is run): config.json, the detections and selection (case.load_selection),
<out_dir>/calibration_result.json and calibration_data.json, the variants' calibration_result.json in
<out_dir>/checks/, and tests/*.json and tests/p0_survey_out.txt.  Draws four images from one raw frame pair (the
example frame, chosen as the kept frame whose wand is longest in both images): the two frames with the fitted
ellipses, the wand after the rod opening, and the small ball's edge points kept and dropped by the rod mask.
Writes <out_dir>/wand_calibration_20261006_T0.html (a standalone page) and, with --fragment, the same page without
its document wrapper (for publishing as an artifact).
"""
import argparse
import base64
import json
import math
import os
import re

import cv2
import numpy as np

import case

HERE = case.HERE
TESTS = os.path.join(HERE, "tests")


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def jpeg_uri(img, width=None, quality=85):
    if width and img.shape[1] != width:
        img = cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


YELLOW = (0, 212, 255)          # BGR
GREEN = (60, 200, 60)
RED = (70, 70, 230)


def draw_ellipse(vis, row, col, label, thick=3):
    u, v, _r, _m, a, b, th = row[:7]
    cv2.ellipse(vis, ((float(u), float(v)), (2 * float(a), 2 * float(b)), math.degrees(float(th))), col, thick,
                cv2.LINE_AA)
    cv2.drawMarker(vis, (int(round(u)), int(round(v))), col, cv2.MARKER_CROSS, 22, 2)
    cv2.putText(vis, label, (int(u + 0.75 * a), int(v - 0.75 * b)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3, cv2.LINE_AA)


def example_frame(cfg, kept, frames):
    """The kept frame whose wand image is longest in both cameras, with both balls well inside both images."""
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    best, score = None, -1.0
    for f in frames:
        ok, length = True, 1e9
        for c in case.cam_ids(cfg):
            a, b = kept[(f, c)]
            for r in (a, b):
                m = 1.3 * max(r[4], r[5])
                if not (m < r[0] < W - m and m < r[1] < H - m):
                    ok = False
            length = min(length, math.hypot(b[0] - a[0], b[1] - a[1]))
        if ok and length > score:
            best, score = f, length
    return best


def images(cfg, kept, f):
    be = case.ball_ellipses()
    det = cfg["detect"]
    out = {}
    for c in case.cam_ids(cfg):
        img = case.read_image(case.image_path(cfg, c, f))
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        a, b = kept[(f, c)]
        draw_ellipse(vis, a, YELLOW, "A")
        draw_ellipse(vis, b, YELLOW, "B")
        out[f"cam{c}"] = jpeg_uri(vis, 900)
    # cam1: the opening, and the small ball's edge points
    c = case.cam_ids(cfg)[0]
    img = case.read_image(case.image_path(cfg, c, f))
    dd = be.settings_for(float(det["max_r_px"]))
    N, _bg, seg = be.normalised_darkness(img.astype(np.float32), dd)
    m = (N > seg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    k = int(det["open_px"]) | 1
    mo = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    a, b = kept[(f, c)]
    x0 = int(max(0, min(a[0] - 1.6 * a[4], b[0] - 1.6 * b[4])))
    x1 = int(min(img.shape[1], max(a[0] + 1.6 * a[4], b[0] + 1.6 * b[4])))
    y0 = int(max(0, min(a[1], b[1]) - 230))
    y1 = int(min(img.shape[0], max(a[1], b[1]) + 1.7 * max(a[5], b[5])))
    vis = cv2.cvtColor((0.55 * img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, (200, 200, 200), 2, cv2.LINE_AA)
    cs, _ = cv2.findContours(mo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)[:2]
    cv2.drawContours(vis, cs, -1, YELLOW, 4, cv2.LINE_AA)
    out["opened"] = jpeg_uri(vis[y0:y1, x0:x1], 900)
    # the small ball's edge points, kept and dropped by the rod mask
    Ns = cv2.GaussianBlur(N, (0, 0), float(dd["smooth_sigma_px"]))
    u, v, _r, _m, ax, bx, th = a[:7]
    ct, st = math.cos(th), math.sin(th)
    rx = math.sqrt((ax * ct) ** 2 + (bx * st) ** 2)
    ry = math.sqrt((ax * st) ** 2 + (bx * ct) ** 2)
    pts, _angs, _complete = be.radial_edges(Ns, (u, v), rx, ry, dd)
    q = np.array([u, v])
    e = (np.array(b[:2]) - q) / np.linalg.norm(np.array(b[:2]) - q)
    rel = pts - q[None, :]
    M = float(det["rod_mask_px"])
    drop = (np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0]) < M) & (rel @ e > 0)
    S = 2                                                     # upscale for visibility
    half = int(1.55 * ax)
    cx0, cy0 = int(round(u)) - half, int(round(v)) - half
    crop = np.zeros((2 * half, 2 * half), np.uint8)
    ys, xs = max(cy0, 0), max(cx0, 0)
    ye, xe = min(cy0 + 2 * half, img.shape[0]), min(cx0 + 2 * half, img.shape[1])
    crop[ys - cy0:ye - cy0, xs - cx0:xe - cx0] = img[ys:ye, xs:xe]
    vis = cv2.cvtColor(cv2.resize(crop, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    sh = lambda p: (int(round((p[0] - cx0 + 0.5) * S - 0.5)), int(round((p[1] - cy0 + 0.5) * S - 0.5)))
    nrm = np.array([-e[1], e[0]])
    for s in (-1, 1):                                          # the strip's edges, dashed
        for t in np.arange(0, 2.2 * half, 14):
            p0 = q + s * M * nrm + t * e
            p1 = q + s * M * nrm + (t + 7) * e
            cv2.line(vis, sh(p0), sh(p1), (230, 230, 230), 2, cv2.LINE_AA)
    cv2.ellipse(vis, (sh((u, v)), (2 * ax * S, 2 * bx * S), math.degrees(th)), YELLOW, 2, cv2.LINE_AA)
    for p, dr in zip(pts, drop):
        cv2.circle(vis, sh(p), 5, RED if dr else GREEN, -1, cv2.LINE_AA)
    out["mask"] = jpeg_uri(vis, 700)
    out["mask_counts"] = [int((~drop).sum()), int(drop.sum())]
    return out


def cfg_rays(cfg):
    """The estimator's number of rays per ball (ball_ellipses.DET)."""
    return int(case.ball_ellipses().DET["rays"])


def scan_rows():
    rows = []
    with open(os.path.join(TESTS, "p0_survey_out.txt"), encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"\s*frame\s+(\d+)\s+(cam\d):\s+(.*)$", line)
            if m:
                pairs = [p.split(":") for p in m.group(3).split()]
                rows.append({"frame": int(m.group(1)), "cam": m.group(2), "ks": [int(p[0]) for p in pairs],
                             "counts": [int(p[1]) for p in pairs]})
    return {"ks": rows[0]["ks"], "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", default=None)
    args = ap.parse_args()
    cfg = case.load_config()
    out_dir = cfg["out_dir"]
    dets, kept, reasons, medians, frames = case.load_selection(cfg)
    cams = case.cam_ids(cfg)
    res = jload(os.path.join(out_dir, "calibration_result.json"))
    data = jload(os.path.join(out_dir, "calibration_data.json"))
    cal = jload(os.path.join(HERE, "calibrate.json"))
    t1 = jload(os.path.join(TESTS, "t1_detection.json"))
    t2 = jload(os.path.join(TESTS, "t2_residuals.json"))
    t36 = jload(os.path.join(TESTS, "t3_t6_variants.json"))
    t7 = jload(os.path.join(TESTS, "t7_vs_20260930.json"))
    d1 = jload(os.path.join(TESTS, "d1_sync.json"))
    d2 = jload(os.path.join(TESTS, "d2_shape_prior.json"))
    d5 = jload(os.path.join(TESTS, "d5_jitter_cameras.json"))
    d6 = jload(os.path.join(TESTS, "d6_length_vs_direction.json"))
    d7 = jload(os.path.join(TESTS, "d7_length_autocorr.json"))
    d8 = jload(os.path.join(TESTS, "d8_held_out.json"))
    S2 = cal["summary"]

    # usable frames per bin
    BIN = 80
    nb = int(math.ceil(len(case.all_frames(cfg)) / BIN))
    usable = {"bin": BIN, "both": [0] * nb, "cam1": [0] * nb, "cam2": [0] * nb}
    fset = set(frames)
    for (f, c) in kept:
        usable[f"cam{c}"][f // BIN] += 1
    for f in fset:
        usable["both"][f // BIN] += 1

    # rod settings
    sens = []
    for key, r in t1["b"].items():
        o, m = key.split("_")
        sens.append({"open": int(o), "mask": int(m), "lost": r["lost"], "A_rms": r["A_rms"], "A_max": r["A_max"],
                     "B_rms": r["B_rms"], "B_max": r["B_max"]})
    d = t1["d"]

    # the rig and the volume
    cams_out = [{"name": c["name"], "C": c["centre_mm"], "f": c["f_px"], "D": c["distance_from_axis_mm"],
                 "az": c["azimuth_deg"]} for c in res["cameras"]]
    XA, XB = np.array(data["ends_A_mm"], float), np.array(data["ends_B_mm"], float)
    act = np.array(data["frame_active"], bool) & np.all(np.isfinite(XA), axis=1) & np.all(np.isfinite(XB), axis=1)
    vc = (0.5 * (XA[act] + XB[act])).mean(axis=0)
    sel = np.flatnonzero(act)[::4]
    vol = {"A": np.round(XA[sel][:, [0, 2]], 3).tolist(), "B": np.round(XB[sel][:, [0, 2]], 3).tolist()}
    ends = np.vstack([XA[act], XB[act]])
    vol_x = float(np.ptp(ends[:, 0]))
    vol_z = float(np.ptp(ends[:, 2]))

    # residual cells
    rescells = {}
    for c in cams:
        rescells[f"cam{c}"] = [{"cx": cc["centre"][0], "cy": cc["centre"][1], "du": cc["mean_du"], "dv": cc["mean_dv"],
                                "m": cc["mean_px"], "n": cc["n"], "z": cc["z"]} for cc in t2["cameras"][f"cam{c}"]["cells"]]

    # wand length along the recording
    L = np.array([np.nan if x is None else x for x in data["wand_length_free_mm"]], float)
    ok = np.isfinite(L) & np.array(data["frame_active"], bool)
    lenp = {"frame": np.array(data["frame_ids"])[ok].tolist(), "err": np.round(1e3 * (L[ok] - 10.0), 2).tolist()}

    # the valley
    pts = []
    labels = {"h1": "first half", "h2": "second half", "q1": "1st quarter", "q2": "2nd quarter", "q3": "3rd quarter",
              "q4": "4th quarter", "fit_L": "wand length free", "fit_k1": "k1 free"}

    def add(summary, label, kind, note=""):
        for cs in summary["cameras"]:
            pts.append({"cam": cs["name"], "label": label, "kind": kind, "D": cs["distance_mm"],
                        "sD": cs.get("distance_sigma_mm") or 0.0, "f": cs["f_px"], "sf": cs.get("f_sigma_px") or 0.0,
                        "note": note})
    add(S2, "full fit (delivered)", "full", f"{S2['frames']} frames, RMS {S2['rms_px']:.3f} px")
    for name, lab in labels.items():
        s = t36["variants"][name]["summary"]
        kind = "option" if name in ("fit_L", "fit_k1") else "part"
        note = f"{s['frames']} frames, RMS {s['rms_px']:.3f} px"
        if name == "fit_L":
            note += f"; L {s['wand_length_mm']:.2f} ± {s['L_sigma_mm']:.2f} mm"
        add(s, lab, kind, note)
    add(d2["summary"], "other centre estimator (D2)", "diag", f"RMS {d2['summary']['rms_px']:.3f} px")
    slope = {cs["name"]: cs["f_px"] / cs["distance_mm"] for cs in S2["cameras"]}
    board = {"f": t7["f"]["dot_board"], "sf": t7["f"]["dot_board_sigma"],
             "D": t7["geometry"]["dot_board"]["distance_from_axis_mm"]}

    # held out
    held = []
    for name in ("S2", "h1", "h2", "q1", "q2", "q3", "q4", "fit_k1", "fit_L"):
        r = d8[name]
        held.append({"name": "full fit" if name == "S2" else name.replace("fit_", "") + (" free" if name.startswith("fit_") else ""),
                     "f1": r["f"][0], "own": r["own"]["length_rms_um"], "own_mean": r["own"]["length_mean_um"],
                     "held": r["held_out"]["length_rms_um"], "held_mean": r["held_out"]["length_mean_um"]})

    # ruled out
    off = d1.get("offset_cam1_minus_cam2_frames"), d1.get("offset_se")
    ruled = [
        ["The cameras out of step", "D1", f"Residuals do not follow the balls' velocity: offset {off[0]:+.4f} ± {off[1]:.4f} frames.", "ruled out"],
        ["Motion blur", "camera header", "100 µs exposures at 100 frames/s: the balls move about 0.2 px during an exposure.", "ruled out"],
        ["The wand bouncing on its string", "D3, D5", f"Both balls jump together by about 0.5 px in v, and both cameras see it (correlation {d5['v']['corr']:.3f}).", "real motion, harmless"],
        ["Camera vibration", "D5", f"Each camera's image jitters by 0.2–0.3 px in u on its own (correlation {d5['u']['corr_core']:.2f} between the cameras). The fit absorbs it.", "present, absorbed"],
        ["A stretched calibration (lens, tube)", "D6, T2", f"The wand's 3-D direction explains {100 * d6['direction (stretch matrix)']['r2']:.1f} % of the length error; the residuals have no smooth pattern.", "ruled out"],
        ["Lens distortion", "T5", f"With k1 free the RMS drops only from {S2['rms_px']:.2f} to {t36['variants']['fit_k1']['summary']['rms_px']:.2f} px, and f barely moves.", "not the cause"],
        ["Wrong tube constants or wand length", "T4", f"With the wand length free the fit gives {t36['variants']['fit_L']['summary']['wand_length_mm']:.2f} ± {t36['variants']['fit_L']['summary']['L_sigma_mm']:.2f} mm, but only by moving further along the same valley.", "not decidable here"],
        ["Ball centres biased by the rod", "T1 b, D4, D7", f"The small ball's centre moves 0.6–2 px with the rod mask; the length error changes with the wand's pose within about 5 frames (correlation {d7['length']['5']:.2f} after 5).", "the likely main cause"],
    ]

    f = example_frame(cfg, kept, frames)
    imgs = images(cfg, kept, f)

    parts = {k: v for k, v in t36["variants"].items() if k in ("h1", "h2", "q1", "q2", "q3", "q4")}
    f1s = [v["summary"]["cameras"][0]["f_px"] for v in parts.values()]
    f1min, f1max = min(f1s), max(f1s)
    hm = [abs(d8[k]["held_out"]["length_mean_um"]) / 100.0 for k in ("h1", "h2")]         # um per 10 mm -> %
    hr = [d8[k]["held_out"]["reproj_rms_px"] for k in ("h1", "h2")]
    by_angle = lambda cam: [v for k, v in d[cam].items() if k.startswith("width_median_px_rod_")]
    w1, w2 = by_angle("cam1"), by_angle("cam2")
    fill = {
        "frames_used": f"{len(frames):,}", "frames_total": f"{len(case.all_frames(cfg)):,}",
        "d8_mean_range": f"{min(hm):.2f}–{max(hm):.2f} %",
        "reproj_range": f"{min(hr):.2f}–{max(hr):.2f} px",
        "f1_min": f"{round(f1min, -2):,.0f}", "f1_max": f"{round(f1max, -2):,.0f}",
        "f1_sig_pct": f"{100 * S2['cameras'][0]['f_sigma_px'] / S2['cameras'][0]['f_px']:.1f}",
        "rms_px": f"{S2['rms_px']:.2f}", "det_used": f"{S2['active']:,}", "det_total": f"{S2['observations']:,}",
        "wand_rms_um": f"{S2['wand_free_rms_um']:.0f}", "wand_rms_pct": f"{S2['wand_free_rms_um'] / 100:.2f}",
        "f1_min_k": f"{f1min / 1000:.1f}", "f1_max_k": f"{f1max / 1000:.1f}", "f1_s2": f"{S2['cameras'][0]['f_px']:,.0f}",
        "example_frame": str(f), "bin": str(BIN),
        "open101_A": f"{t1['b']['101_50']['A_rms']:.2f}",
        "rod_w1": f"{min(w1):.0f}–{max(w1):.0f}", "rod_w2": f"{min(w2):.0f}–{max(w2):.0f}",
        "blur": f"{min(d['cam1']['blur_sigma_median_px'], d['cam2']['blur_sigma_median_px']):.0f}–"
                f"{math.ceil(max(d['cam1']['blur_sigma_median_px'], d['cam2']['blur_sigma_median_px'])):.0f}",
        "rule1": f"{d['cam1']['rule_median_px']:.0f}", "rule2": f"{d['cam2']['rule_median_px']:.0f}",
        "rule_p90": f"{min(d['cam1']['rule_p90_px'], d['cam2']['rule_p90_px']):.0f}–"
                    f"{max(d['cam1']['rule_p90_px'], d['cam2']['rule_p90_px']):.0f}",
        "d1": f"{S2['cameras'][0]['distance_mm']:,.0f}", "d2": f"{S2['cameras'][1]['distance_mm']:,.0f}",
        "vol_x": f"{vol_x:.0f}", "vol_z": f"{vol_z:.0f}",
        "mask_kept": str(imgs["mask_counts"][0]), "mask_dropped": str(imgs["mask_counts"][1]),
        "mask_none": str(int(cfg_rays(cfg)) - sum(imgs["mask_counts"])),
    }
    payload = {"images": {k: v for k, v in imgs.items() if k != "mask_counts"}, "scan": scan_rows(), "usable": usable,
               "sens": sens, "rig": {"cams": cams_out, "radii": res["tube"]["radii_mm"], "vol_centre": vc.tolist()},
               "vol": vol, "res": rescells, "len": lenp,
               "valley": {"points": pts, "board": board, "slope": slope}, "held": held, "ruled": ruled}

    with open(os.path.join(HERE, "page_template.html"), encoding="utf-8") as fh:
        page = fh.read()
    for k, v in fill.items():
        page = page.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[a-z0-9_]+\}\}", page)
    if left:
        raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace("__REPORT_DATA__", blob)
    head, body = page.split("<main", 1)
    standalone = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                  "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                  + head + "</head>\n<body>\n<main" + body + "\n</body>\n</html>\n")
    path = os.path.join(out_dir, "wand_calibration_20261006_T0.html")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(standalone)
    os.replace(tmp, path)
    print(f"example frame {f}; mask points kept / dropped {imgs['mask_counts']}; page {len(standalone) / 1e6:.2f} MB")
    print("written", path)
    if args.fragment:
        with open(args.fragment, "w", encoding="utf-8") as fh:
            fh.write(page)
        print("written", args.fragment)


if __name__ == "__main__":
    main()
