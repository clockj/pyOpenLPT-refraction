"""The case's overview page: page_template.html filled with the saved outputs of S1, S2, P0, T1-T3, T7, X1, D1 and D8,
and with T0's saved results beside them.

    python make_page.py [--fragment PATH]

Reads only saved results (no calibration is run): config.json, the detections and selection (case.load_selection),
<out_dir>/calibration_result.json and calibration_data.json, the parts' results in <out_dir>/checks/, tests/*.json;
and T0's: its case's calibrate.json and tests/t3_t6_variants.json, and its calibration_data.json.  Draws the example
frame pair (the kept frame whose wand is longest in both images) with the fitted ellipses.
Writes <out_dir>/wand_calibration_20261006_T2.html (a standalone page) and, with --fragment, the same page without
its document wrapper (for publishing as an artifact).
"""
import argparse
import base64
import json
import math
import os
import re
from collections import Counter

import cv2
import numpy as np

import case

HERE = case.HERE
TESTS = os.path.join(HERE, "tests")
YELLOW = (0, 212, 255)          # BGR


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def jpeg_uri(img, width=None, quality=85):
    if width and img.shape[1] != width:
        img = cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


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
    out = {}
    for c in case.cam_ids(cfg):
        vis = cv2.cvtColor(case.read_image(case.image_path(cfg, c, f)), cv2.COLOR_GRAY2BGR)
        a, b = kept[(f, c)]
        draw_ellipse(vis, a, YELLOW, "A")
        draw_ellipse(vis, b, YELLOW, "B")
        out[f"cam{c}"] = jpeg_uri(vis, 900)
    return out


def volume(data, step):
    XA, XB = np.array(data["ends_A_mm"], float), np.array(data["ends_B_mm"], float)
    act = np.array(data["frame_active"], bool) & np.all(np.isfinite(XA), axis=1) & np.all(np.isfinite(XB), axis=1)
    sel = np.flatnonzero(act)[::step]
    ends = np.vstack([XA[act], XB[act]])
    return ({"A": np.round(XA[sel][:, [0, 2]], 3).tolist(), "B": np.round(XB[sel][:, [0, 2]], 3).tolist()},
            (0.5 * (XA[act] + XB[act])).mean(axis=0), float(np.ptp(ends[:, 0])), float(np.ptp(ends[:, 2])))


def err2d(data, R=2.0, step=0.1):
    """Each camera's reprojection errors (du, dv: the model's position minus the detection) of the detections the
    fit used, counted in step x step px cells over +-R px, with the marginal histograms and the statistics."""
    ob = data["observations"]
    cam = np.array(ob["cam_id"])
    du = np.array([np.nan if x is None else x for x in ob["du"]], float)
    dv = np.array([np.nan if x is None else x for x in ob["dv"]], float)
    act = np.array(ob["active"], bool) & np.isfinite(du) & np.isfinite(dv)
    nb = int(round(2 * R / step))
    out = {"R": R, "step": step, "cams": {}}
    for c in sorted(set(cam.tolist())):
        s = act & (cam == c)
        u, v = du[s], dv[s]
        inside = (np.abs(u) < R) & (np.abs(v) < R)
        H, _, _ = np.histogram2d(u[inside], v[inside], bins=nb, range=[[-R, R], [-R, R]])
        out["cams"][f"cam{c}"] = {
            "cells": [[int(i), int(j), int(H[i, j])] for i, j in zip(*np.nonzero(H))],
            "mu": np.histogram(u[np.abs(u) < R], bins=nb, range=(-R, R))[0].tolist(),
            "mv": np.histogram(v[np.abs(v) < R], bins=nb, range=(-R, R))[0].tolist(),
            "n": int(s.sum()), "outside": int((~inside).sum()), "inactive": int(((cam == c) & ~act).sum()),
            "rms_u": float(np.sqrt(np.mean(u ** 2))), "rms_v": float(np.sqrt(np.mean(v ** 2))),
            "rms": float(np.sqrt(np.mean(u ** 2 + v ** 2))), "corr": float(np.corrcoef(u, v)[0, 1])}
    out["max"] = max(max(q[2] for q in cc["cells"]) for cc in out["cams"].values())
    return out


def err_caption(err):
    c1, c2 = err["cams"]["cam1"], err["cams"]["cam2"]
    shape = lambda c: ("across" if c["rms_u"] > 1.2 * c["rms_v"] else "up and down" if c["rms_v"] > 1.2 * c["rms_u"]
                       else "round")
    ratio = lambda c: c["rms_u"] / c["rms_v"]
    if shape(c1) == shape(c2) == "across":
        spread = (f"Both cameras' errors are {min(ratio(c1), ratio(c2)):.1f}–{max(ratio(c1), ratio(c2)):.1f} times "
                  f"wider across (du) than up and down (dv).")
    else:
        word = {"across": "wider across (du) than up and down (dv)", "up and down": "wider up and down (dv) than "
                "across (du)", "round": "about as wide across as up and down"}
        spread = f"cam1's errors are {word[shape(c1)]}; cam2's are {word[shape(c2)]}."
    why = (" The wand hangs roughly level, so across is along the rod: there the rod hides part of each ball's "
           "outline (T0's T1 b), and each camera's image also jitters sideways on its own (T0's D5)."
           if "across" in (shape(c1), shape(c2)) else "")
    return (f"du, dv: where the calibration puts each ball minus where the detector found it, for the {c1['n']:,} and "
            f"{c2['n']:,} detections the fit used (it set aside {c1['inactive']} and {c2['inactive']} as outliers). "
            f"Dashed: the 2-D RMS, {c1['rms']:.2f} px in cam1 and {c2['rms']:.2f} px in cam2. Bars: du (top) and dv "
            f"(right). Beyond ±{err['R']:.0f} px, not drawn: {c1['outside']} and {c2['outside']}. " + spread + why)


def t5_verdict(t5):
    """T5's verdict by its judging rule (T0's T5): keep k1 only if it clearly lowers the residuals."""
    drop = 1.0 - t5["fit_k1"]["rms_px"] / t5["S2"]["rms_px"]
    if drop < 0.10:
        return (f"little ({100 * drop:.0f} %): k1 soaks up other errors rather than describing the lens, so it is "
                "left off")
    return f"it lowers the RMS by {100 * drop:.0f} % (see the report)"


def signed(v, d):
    return f"{'+' if v >= 0 else '−'}{abs(v):.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", default=None)
    args = ap.parse_args()
    cfg = case.load_config()
    out_dir = cfg["out_dir"]
    t0_case = os.path.normpath(os.path.join(HERE, cfg["reference_T0"]["case_dir"]))
    t0_out = cfg["reference_T0"]["result_dir"]
    dets, kept, reasons, medians, frames = case.load_selection(cfg)
    cams = case.cam_ids(cfg)
    n_total = len(case.all_frames(cfg))
    res = jload(os.path.join(out_dir, "calibration_result.json"))
    data = jload(os.path.join(out_dir, "calibration_data.json"))
    cal = jload(os.path.join(HERE, "calibrate.json"))
    t1 = jload(os.path.join(TESTS, "t1_detection.json"))
    t2 = jload(os.path.join(TESTS, "t2_residuals.json"))
    t3 = jload(os.path.join(TESTS, "t3_variants.json"))
    t7 = jload(os.path.join(TESTS, "t7_references.json"))
    d1 = jload(os.path.join(TESTS, "d1_sync.json"))
    d8 = jload(os.path.join(TESTS, "d8_held_out.json"))
    x1 = jload(os.path.join(TESTS, "x1_cross_runs.json"))
    p0 = jload(os.path.join(TESTS, "p0_survey.json"))
    t0_cal = jload(os.path.join(t0_case, "calibrate.json"))
    t0_t36 = jload(os.path.join(t0_case, "tests", "t3_t6_variants.json"))
    t0_data = jload(os.path.join(t0_out, "calibration_data.json"))
    S2, S0 = cal["summary"], t0_cal["summary"]
    LIGHT = [2000, 3000]

    # usable frames per bin
    BIN = 80
    nb = int(math.ceil(n_total / BIN))
    usable = {"bin": BIN, "both": [0] * nb, "cam1": [0] * nb, "cam2": [0] * nb, "light": LIGHT}
    for (f, c) in kept:
        usable[f"cam{c}"][f // BIN] += 1
    for f in frames:
        usable["both"][f // BIN] += 1

    # which camera-frames were kept
    sel = []
    for c in cams:
        keys = [k for k in dets if k[1] == c]
        why = Counter(reasons[k] for k in keys if k in reasons)
        sel.append({"cam": f"cam{c}", "total": len(keys), "kept": sum(1 for k in keys if k in kept),
                    "one": why.get("one ball", 0), "none": why.get("no ball", 0),
                    "other": sum(v for k, v in why.items() if k not in ("one ball", "no ball")),
                    "rA": medians[c][0], "rB": medians[c][1], "why": dict(why)})

    # the rig and the volumes
    cams_out = [{"name": c["name"], "C": c["centre_mm"], "f": c["f_px"], "D": c["distance_from_axis_mm"],
                 "az": c["azimuth_deg"]} for c in res["cameras"]]
    vol, vc, vol_x, vol_z = volume(data, 4)
    vol0, _vc0, vol0_x, vol0_z = volume(t0_data, 8)

    # residual cells, wand length
    rescells = {f"cam{c}": [{"cx": cc["centre"][0], "cy": cc["centre"][1], "du": cc["mean_du"], "dv": cc["mean_dv"],
                             "m": cc["mean_px"], "n": cc["n"], "z": cc["z"]} for cc in t2["cameras"][f"cam{c}"]["cells"]]
                for c in cams}
    L = np.array([np.nan if x is None else x for x in data["wand_length_free_mm"]], float)
    ok = np.isfinite(L) & np.array(data["frame_active"], bool)
    lenp = {"frame": np.array(data["frame_ids"])[ok].tolist(), "err": np.round(1e3 * (L[ok] - 10.0), 2).tolist(),
            "n": n_total}

    # the cameras' move (X1)
    cfg0 = case.load_config(os.path.join(t0_case, "config.json"))
    off = {"n0": len(case.all_frames(cfg0)), "n2": n_total, "light": LIGHT,
           "series": {k: {"frame": v["frame"], "dv": v["dv_smooth"]} for k, v in x1["series"].items()}}
    cross = []
    for cal_name in ("T0", "T2"):
        for run in ("T0", "T2"):
            p = x1["pairs"][f"{cal_name}_cams_on_{run}"]
            cross.append({"cal": f"{cal_name}'s", "run": run, "frames": p["frames"], "mean": p["length_mean_um"],
                          "se": p["length_se_um"], "rms": p["length_rms_um"], "dv": p["dv_mean_px"]})
    pr = x1["pairs"]
    dv_cross = pr["T0_cams_on_T2"]["dv_mean_px"]
    dv_back = pr["T2_cams_on_T0"]["dv_mean_px"]
    dv_own = pr["T2_cams_on_T2"]["dv_mean_px"]

    # the valley: T2's full fit and parts, T0's full fit and parts, the dot board
    pts = []
    labels = {"h1": "first half", "h2": "second half", "q1": "1st quarter", "q2": "2nd quarter",
              "q3": "3rd quarter", "q4": "4th quarter"}

    def add(summary, run, label, kind, note=""):
        for cs in summary["cameras"]:
            pts.append({"cam": cs["name"], "run": run, "label": label, "kind": kind, "D": cs["distance_mm"],
                        "sD": cs.get("distance_sigma_mm") or 0.0, "f": cs["f_px"], "sf": cs.get("f_sigma_px") or 0.0,
                        "note": note})
    add(S2, "T2", "full fit (delivered)", "full", f"{S2['frames']} frames, RMS {S2['rms_px']:.3f} px")
    for name, lab in labels.items():
        s = t3["variants"][name]["summary"]
        add(s, "T2", lab, "part", f"{s['frames']} frames, RMS {s['rms_px']:.3f} px")
    add(S0, "T0", "full fit", "t0full", f"{S0['frames']} frames, RMS {S0['rms_px']:.3f} px")
    for name, lab in labels.items():
        s = t0_t36["variants"][name]["summary"]
        add(s, "T0", lab, "t0part", f"{s['frames']} frames, RMS {s['rms_px']:.3f} px")
    slope = {cs["name"]: cs["f_px"] / cs["distance_mm"] for cs in S2["cameras"]}
    ref = {cs["name"]: cs for cs in S2["cameras"]}
    for p in pts:                       # a part that ran far along the valley is noted, not drawn to scale
        p["off"] = bool(p["f"] > 1.5 * ref[p["cam"]]["f_px"] or p["D"] > 1.5 * ref[p["cam"]]["distance_mm"])
    tv = t7["vs_20260930"]
    board = {"f": tv["f"]["dot_board"], "sf": tv["f"]["dot_board_sigma"],
             "D": tv["geometry"]["dot_board"]["distance_from_axis_mm"]}

    # held out
    held = []
    for name in ("S2", "h1", "h2", "q1", "q2", "q3", "q4"):
        r = d8[name]
        held.append({"name": "full fit" if name == "S2" else name, "f1": r["f"][0],
                     "own": r["own"]["length_rms_um"], "own_mean": r["own"]["length_mean_um"],
                     "held": r["held_out"]["length_rms_um"], "held_mean": r["held_out"]["length_mean_um"]})

    parts = {k: t3["variants"][k]["summary"] for k in labels}
    f1s = [p["cameras"][0]["f_px"] for p in parts.values()]
    f2s = [p["cameras"][1]["f_px"] for p in parts.values()]
    t0f1 = [t0_t36["variants"][k]["summary"]["cameras"][0]["f_px"] for k in labels]
    zmax = {k: t3["variants"][k]["comparison"]["largest_abs_z"] for k in labels}
    hm = [abs(d8[k]["held_out"]["length_mean_um"]) / 100.0 for k in ("h1", "h2")]         # um per 10 mm -> %
    hr = [d8[k]["held_out"]["reproj_rms_px"] for k in ("h1", "h2")]
    hrms = [d8[k]["held_out"]["length_rms_um"] for k in ("h1", "h2")]
    grey = p0["grey_every_500"]
    gi_a, gi_b = LIGHT[0] // 500 - 1, LIGHT[1] // 500 + 1        # samples just before and after the light's step
    swaps = sum(t1["c"][f"cam{c}"]["swaps"] for c in cams)
    pairs_c = sum(t1["c"][f"cam{c}"]["pairs"] for c in cams)
    off_d1 = d1.get("offset_cam1_minus_cam2_frames"), d1.get("offset_se")
    big = {c: t2["cameras"][f"cam{c}"]["largest_cell"] for c in cams}
    n_above = {c: sum(1 for d in t2["cameras"][f"cam{c}"]["cells"] if d["mean_px"] > 0.1) for c in cams}
    n_cells = {c: len(t2["cameras"][f"cam{c}"]["cells"]) for c in cams}
    wl = t2["wand_length"]

    edge = {}
    for c in cams:
        for tag, sel_f in (("before", lambda f: f < LIGHT[0]), ("after", lambda f: f > LIGHT[1])):
            vals = [case.edge_rms(r) for (f, cc), ab in kept.items() if cc == c and sel_f(f) for r in ab]
            edge[(c, tag)] = float(np.median(vals))
    checks = [
        ["Are the cameras in step?", "D1", f"Residuals do not follow the balls' velocity: offset {signed(off_d1[0], 4)} ± {off_d1[1]:.4f} frames.", "yes"],
        ["Do balls A and B swap?", "T1 c", f"{swaps} swaps in {pairs_c:,} consecutive kept pairs.", "no"],
        ["Does the light's change upset the detection?", "T1 a, X1",
         f"The overlays before and after it look alike. Edge scatter about the fitted ellipses, median before / after: "
         f"cam1 {edge[(1, 'before')]:.2f} / {edge[(1, 'after')]:.2f} px, cam2 {edge[(2, 'before')]:.2f} / "
         f"{edge[(2, 'after')]:.2f} px. The cameras' disagreement is the same on both sides (X1).", "no"],
        ["A pattern over the image?", "T2, T5",
         f"Cell means up to {big[1]['mean_px']:.2f} px (cam1) and {big[2]['mean_px']:.2f} px (cam2); "
         f"{n_above[1]} of {n_cells[1]} and {n_above[2]} of {n_cells[2]} cells above 0.1 px, mostly across the image. "
         f"Freeing k1 does not remove it.",
         "yes, as in T0, a little stronger; not lens distortion"],
        ["Does the wand length drift along the recording?", "T2",
         f"{signed(wl['slopes']['frame']['um_per_1000_frames'], 2)} ± {wl['slopes']['frame']['se']:.2f} µm per 1,000 frames "
         f"({abs(wl['slopes']['frame']['um_per_1000_frames']) * n_total / 1000:.0f} µm over the recording, against "
         f"{wl['sd_um']:.0f} µm of scatter).",
         "no" if abs(wl['slopes']['frame']['um_per_1000_frames']) < 3 * wl['slopes']['frame']['se'] else "slightly"],
        ["Does cam1 match the dot board of 30 Sep?", "T7",
         f"f {tv['f']['wand']:,.0f} against {tv['f']['dot_board']:,.0f} px ({signed(100 * (tv['f']['wand'] / tv['f']['dot_board'] - 1), 1)} %).",
         "not decidable (f is not fixed)"],
    ]
    t5_path = os.path.join(TESTS, "t5_k1.json")
    if os.path.exists(t5_path):
        t5 = jload(t5_path)
        k1c = t5["cameras"]
        cl = {n: (k1c[n]["cells_S2"]["above_0p1"], k1c[n]["cells_k1"]["above_0p1"], k1c[n]["cells_k1"]["cells"])
              for n in ("cam1", "cam2")}
        checks.insert(4, ["Would lens distortion (k1) help?", "T5",
                          f"With k1 free the RMS goes from {t5['S2']['rms_px']:.3f} to {t5['fit_k1']['rms_px']:.3f} px "
                          f"({signed(100 * (t5['fit_k1']['rms_px'] / t5['S2']['rms_px'] - 1), 0)} %). k1 comes out "
                          + " and ".join(f"{signed(k1c[n]['k1'], 1)} ± {k1c[n]['k1_sigma']:.1f} ({n})" for n in ("cam1", "cam2"))
                          + f": {abs(k1c['cam1']['shift_edge_px']):.1f} and {abs(k1c['cam2']['shift_edge_px']):.1f} px at "
                          f"the image's side edge, of opposite signs for two alike lenses. The residual pattern stays "
                          f"(cells above 0.1 px: cam1 {cl['cam1'][0]} → {cl['cam1'][1]} of {cl['cam1'][2]}, cam2 "
                          f"{cl['cam2'][0]} → {cl['cam2'][1]} of {cl['cam2'][2]}).", t5_verdict(t5)])

    f = example_frame(cfg, kept, frames)
    imgs = images(cfg, kept, f)
    S2c, S0c = S2["cameras"], S0["cameras"]
    fill = {
        "frames_used": f"{len(frames):,}", "frames_total": f"{n_total:,}", "t0_frames": f"{S0['frames']:,}",
        "rms_px": f"{S2['rms_px']:.2f}", "det_used": f"{S2['active']:,}", "det_total": f"{S2['observations']:,}",
        "t0_rms_px": f"{S0['rms_px']:.2f}",
        "wand_rms_um": f"{S2['wand_free_rms_um']:.0f}", "wand_rms_pct": f"{S2['wand_free_rms_um'] / 100:.2f}",
        "t0_wand_rms_um": f"{S0['wand_free_rms_um']:.0f}",
        "x1_dv": f"{abs(dv_cross):.1f}", "x1_dv_own": f"{dv_own:+.2f}".replace("-", "−"),
        "example_frame": str(f), "bin": str(BIN), "light_a": f"{LIGHT[0]:,}", "light_b": f"{LIGHT[1]:,}",
        "grey1_a": f"{grey['cam1'][gi_a]:.0f}", "grey1_b": f"{grey['cam1'][gi_b]:.0f}",
        "grey2_a": f"{grey['cam2'][gi_a]:.0f}", "grey2_b": f"{grey['cam2'][gi_b]:.0f}",
        "d1": f"{S2c[0]['distance_mm']:,.0f}", "d2": f"{S2c[1]['distance_mm']:,.0f}",
        "t0_d1": f"{S0c[0]['distance_mm']:,.0f}", "t0_d2": f"{S0c[1]['distance_mm']:,.0f}",
        "vol_x": f"{vol_x:.0f}", "vol_z": f"{vol_z:.0f}", "t0_vol_x": f"{vol0_x:.0f}", "t0_vol_z": f"{vol0_z:.0f}",
    }
    text = page_text(fill, S2, S0, f1s, f2s, t0f1, zmax, hm, hr, hrms, dv_cross, dv_back, dv_own, pr, x1, sel, big,
                     n_above, n_cells, wl, d8, tv, parts)
    fill.update(text)
    err = err2d(data)
    fill["err_caption"] = err_caption(err)
    payload = {"images": imgs, "usable": usable, "sel": sel, "err": err,
               "rig": {"cams": cams_out, "radii": res["tube"]["radii_mm"], "vol_centre": vc.tolist()},
               "vol": vol, "vol0": vol0, "res": rescells, "len": lenp, "off": off, "cross": cross,
               "valley": {"points": pts, "board": board, "slope": slope}, "held": held, "checks": checks}

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
    path = os.path.join(out_dir, "wand_calibration_20261006_T2.html")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(standalone)
    os.replace(tmp, path)
    print(f"example frame {f}; page {len(standalone) / 1e6:.2f} MB")
    print("written", path)
    if args.fragment:
        with open(args.fragment, "w", encoding="utf-8") as fh:
            fh.write(page)
        print("written", args.fragment)


def page_text(fill, S2, S0, f1s, f2s, t0f1, zmax, hm, hr, hrms, dv_cross, dv_back, dv_own, pr, x1, sel, big, n_above,
              n_cells, wl, d8, tv, parts):
    """The page's sentences that quote results (the rest of the text is in page_template.html)."""
    k = lambda v: f"{round(v, -2):,.0f}"
    t = {}
    t["verdict_use"] = (f"Cameras fitted on one half of T2 predict the other half's wand lengths to {min(hm):.2f}–"
                        f"{max(hm):.2f} % on average ({min(hrms):.0f}–{max(hrms):.0f} µm RMS); two-camera reprojection "
                        f"{min(hr):.2f}–{max(hr):.2f} px.")
    t["verdict_moved"] = (f"On T2's frames, T0's cameras disagree by {abs(dv_cross):.2f} px vertically, from T2's first "
                          f"frames on; T2's own cameras by {abs(dv_own):.2f} px. Use T2's camFiles with T2, T0's with "
                          f"T0: a calibration holds only while the cameras stay untouched.")
    t["verdict_f"] = (f"T2's halves and quarters give cam1 f from {k(min(f1s))} to {k(max(f1s))} px (T0: "
                      f"{k(min(t0f1))}–{k(max(t0f1))}), each with its distance in proportion. The fit's error bars "
                      f"(±{100 * S2['cameras'][0]['f_sigma_px'] / S2['cameras'][0]['f_px']:.1f} %) are far too small.")
    t["lead_recording"] = ("The wand swings through the same small views as in T0 (about 26 × 21 mm and 29 × 23 mm at the "
                           "wand), more often end-on to a camera. About 25 s in, the light got brighter and stayed so; "
                           "the detector divides by the background, so it found the balls alike before and after.")
    t["sel_caption"] = ("Camera-frames of the whole recording. <b>One ball</b>: the other is out of view, or hidden "
                        "behind it when the wand points at the camera. <b>Other</b>: two balls but overlapping, a poor "
                        "fit or an odd size. A frame is used when both cameras keep it.")
    t["res_caption"] = (f"Mean residual in each image cell with 20 or more detections, the arrow drawn 500× (0.1 px = "
                        f"50 px): up to {big[1]['mean_px']:.2f} px in cam1 and {big[2]['mean_px']:.2f} px in cam2. In "
                        f"cam1 they point mostly across the image, outward on both sides. Freeing the lens distortion "
                        f"k1 does not remove them (T5), so they are not lens distortion; as in T0, the likely source is "
                        f"the rod's effect on the ball centres, which follows the wand's pose.")
    t["len_caption"] = (f"The wand length from a free two-camera triangulation, minus 10 mm, frame by frame "
                        f"({S2['wand_free_rms_um']:.0f} µm RMS; trend {signed(wl['slopes']['frame']['um_per_1000_frames'], 2)} "
                        f"± {wl['slopes']['frame']['se']:.2f} µm per 1,000 frames). Shaded: the light's change.")
    t["lead_moved"] = ("Two cameras side by side around the tube must agree on the height of every ball. T0's calibration "
                       "still measures T2's wand at the right length, but its two cameras now disagree about the balls' "
                       f"height by {abs(dv_cross):.2f} px, and T2's calibration disagrees on T0's frames by the same "
                       f"amount the other way ({signed(dv_back, 2)} px). One px is a tilt of 26 µrad, a shift of "
                       "26 µm at the wand, or a lens refocused. This test cannot say which camera changed, nor see a "
                       "sideways move.")
    tr0 = pr["T0_cams_on_T2"]["dv_trend_px_per_1000_frames"]
    t["off_caption"] = (f"The cameras' vertical disagreement (cam1 minus cam2), the median of {x1['window']} consecutive "
                        f"kept frames, along each recording, with each calibration. On its own recording each "
                        f"calibration sits on zero; on the other it is off by about 1 px, and the step stays level "
                        f"along all of T2, before and after the light's change (trend of T0's cameras on T2: "
                        f"{signed(tr0[0], 3)} ± {tr0[1]:.3f} px per 1,000 frames). Within each recording the "
                        f"disagreement also wanders by about ±0.3 px. Both calibrations see the same wander, so it is "
                        f"in the data: the cameras wobbling, or errors that follow the wand's position. This test "
                        f"cannot tell which.")
    t["cross_caption"] = ("Free triangulation of every kept frame (no wand length, no fitted poses). The wand length "
                          "barely notices the move; the cameras' disagreement shows it plainly.")
    t["valley_h2"] = "T2 does not fix f either: its parts spread along the same line as T0's"
    t["lead_valley"] = ("Long lenses looking at a 30 mm volume from 1 m see almost no perspective, so a camera farther "
                        "away with a longer lens makes nearly the same image. T2's wand covered about T0's volume, with "
                        "no more depth, so it cannot separate the two any better.")
    q2c2 = next(p for p in parts["q2"]["cameras"] if p["name"] == "cam2")
    t["valley_caption"] = (f"Every calibration of both recordings: the full fits and their halves and quarters. Bars: "
                           f"T2's fits' own ±1σ. All lie along lines through the origin (dashed: T2's full fit's "
                           f"f / distance), so f and the distance trade against each other; the parts differ from the "
                           f"full fit by up to {max(zmax.values()):.1f}σ. T2's 2nd quarter, with only "
                           f"{parts['q2']['frames']} frames, put cam2 at f {q2c2['f_px']:,.0f} px and "
                           f"{q2c2['distance_mm'] / 1000:.1f} m, far along the same line (noted at the top, off the "
                           f"chart). The 30 Sep dot-board calibration of cam1 (star) lies on cam1's line too (f / "
                           f"distance {tv['f']['dot_board'] / tv['geometry']['dot_board']['distance_from_axis_mm']:.1f} "
                           f"against {S2['cameras'][0]['f_px'] / S2['cameras'][0]['distance_mm']:.1f} px/mm). So the "
                           f"wand fixes f / distance, and an f measured with the dot board (if the lens and focus are "
                           f"unchanged), held in the wand calibration, would fix the rest.")
    hmax = max(abs(d8[k]["held_out"]["length_mean_um"]) for k in ("h1", "h2", "q1", "q2", "q3", "q4"))
    hr_all = [d8[k]["held_out"]["length_rms_um"] for k in ("h1", "h2", "q1", "q2", "q3", "q4")]
    t["agree_h2"] = "Inside the wand's volume T2's cameras agree"
    t["lead_agree"] = (f"Each part's cameras triangulate every kept frame of T2, including the frames they were not "
                       f"fitted on. Despite their different f, the wand lengths they measure on the held-out frames "
                       f"agree with 10 mm to within {hmax / 100:.2f} % on average over all six parts "
                       f"({min(hr_all):.0f}–{max(hr_all):.0f} µm RMS). In T0 the halves gave 0.07–0.13 %.")
    t["held_caption"] = ("RMS error of the triangulated wand length on the frames each part was fitted on, and on the "
                         "others (held out). Hover for the mean error.")
    t["next_items"] = "\n".join("      <li>" + s + "</li>" for s in [
        "<b>Use T2's camFiles for T2</b>, and T0's for T0, each only inside the volume its wand covered. Neither "
        "fixes f or the distances, so do not rely on them outside that volume.",
        "<b>Record the wand right before (or after) each experiment, without touching the rig in between.</b> The "
        "cameras moved by about 1 px between T0 and T2.",
        "<b>Check what changed between T0 and T2</b>: a camera, its mount or lens touched, or the light moved. If "
        "nothing was touched, the mounts settle on their own and should be checked.",
        "<b>The focal length needs an independent measurement</b>: the dot board through the tube for each camera, "
        "held fixed in the wand calibration (a small library option, not built yet), or a third camera.",
        "<b>A thinner rod</b> (1 mm or less) or larger balls, and <b>more depth</b>: as for T0.",
    ])
    return t


if __name__ == "__main__":
    main()
