"""Figures of the flat-window wand-calibration tests: gathers the numbers the page draws.

Reads the saved results of W0-W9 (the *.json beside this script), rebuilds the rigs' geometry (prigs.RIGS for A-E,
prigs.random_rig with W5's seeds for the 60 random rigs) and traces each camera's rays through its window with
plate_model, for the drawings.  Nothing is re-fitted, so the page shows exactly the results the report quotes.

Writes figures/figure_data.json and figures/plate_wand_results.html: the page, built from figures/template.html with
the data filled in.  Open it in a browser; it needs nothing else (fonts fall back to the system's offline).  With
--fragment PATH it also writes the page without the document skeleton (for publishing as an artifact).
Run: python tests/plate_wand/make_figures.py [--fragment PATH]   (refract env)
"""
import argparse
import json
import math
import re
import time

import numpy as np

import prigs

pm = prigs.pm
HERE = prigs.HERE
FIG = HERE / "figures"
TEMPLATE = FIG / "template.html"
PAGE = FIG / "plate_wand_results.html"


def load(name):
    with open(HERE / name, encoding="utf-8") as fh:
        return json.load(fh)


def sig(x, n=5):
    """x rounded to n significant digits (lists and dicts recursively), to keep the page small."""
    if isinstance(x, dict):
        return {k: sig(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig(v, n) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if x == 0.0 or not math.isfinite(x):
            return x if math.isfinite(x) else None
        return float(f"{x:.{n}g}")
    return x


# ------------------------------------------------------------------ geometry
def far_depth(win, centre, half):
    """The cloud box's largest depth along the window's normal (n.X)."""
    corners = prigs.box_corners(centre, half)
    return float(np.max(corners @ win.n))


def camera_rays(cam, win, depth, n_rays=5):
    """Polylines of n_rays rays across the sensor's centre row: centre, entry into the window, exit into the water,
    and the point at the cloud's far depth."""
    us = np.linspace(0.0, cam.width - 1.0, n_rays)
    uv = np.column_stack([us, np.full(n_rays, cam.cy)])
    rw = pm.rows([cam], np.zeros(n_rays, int))
    ww = pm.wrows([win], np.zeros(n_rays, int))
    o, d, ok = pm.rays(rw, ww, uv)
    xy = pm.undistort(uv, rw)
    out = []
    for k in range(n_rays):
        if not ok[k]:
            continue
        q = cam.R.T @ np.array([xy[k, 0], xy[k, 1], 1.0])
        q /= np.linalg.norm(q)
        entry = cam.C + ((win.d - float(win.n @ cam.C)) / float(win.n @ q)) * q
        end = o[k] + ((depth - float(win.n @ o[k])) / float(win.n @ d[k])) * d[k]
        out.append([cam.C.tolist(), entry.tolist(), o[k].tolist(), end.tolist()])
    return out


def window_patch(win, pts):
    """Corners of the window's camera-side and water-side faces, a rectangle in its plane covering `pts` (their
    projections onto the plane) with a margin."""
    t1, t2 = pm.tangent_basis(win.n)
    t1, t2 = t1[0], t2[0]
    P = np.asarray(pts, float)
    a, b = P @ t1, P @ t2
    pad_a, pad_b = 0.08 * (a.max() - a.min()) + 15.0, 0.08 * (b.max() - b.min()) + 15.0
    a0, a1, b0, b1 = a.min() - pad_a, a.max() + pad_a, b.min() - pad_b, b.max() + pad_b
    near = [win.d * win.n + aa * t1 + bb * t2 for aa, bb in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))]
    far = [p + win.T * win.n for p in near]
    return [[p.tolist() for p in near], [p.tolist() for p in far]]


def rig_record(name, rig_t, n_rays=5, wands=0, seed=0, labels=None):
    """Everything the drawing of one rig needs (world frame of the test: walls Z = 0 / X = 0, Y up)."""
    cams = []
    rays_by_win = {w: [] for w in range(len(rig_t.wins))}
    for k, cam in enumerate(rig_t.cams):
        win = rig_t.win_of(k)
        depth = far_depth(win, rig_t.centre, rig_t.half) + 0.05 * float(np.max(rig_t.half))
        rays = camera_rays(cam, win, depth, n_rays)
        rays_by_win[rig_t.cam_win[k]] += [r[1] for r in rays]
        inc = math.degrees(math.acos(min(1.0, float(cam.R[2] @ win.n))))
        rec = {"C": cam.C.tolist(), "f": cam.f, "w": cam.width, "hgt": cam.height, "win": rig_t.cam_win[k],
               "inc": inc, "h1": win.distance(cam.C), "rays": rays}
        if labels:
            rec["design"] = labels[k]
        cams.append(rec)
    corners = prigs.box_corners(rig_t.centre, rig_t.half)
    wins = []
    for w, win in enumerate(rig_t.wins):
        pts = rays_by_win[w] + [list(p) for p in corners]
        wins.append({"n": win.n.tolist(), "d": win.d, "t": list(win.t), "name": win.name,
                     "patch": window_patch(win, np.array(pts))})
    out = {"name": name, "cams": cams, "wins": wins, "centre": rig_t.centre.tolist(), "half": rig_t.half.tolist(),
           "L": rig_t.L}
    if wands:
        wd = prigs.make_wand(rig_t, wands, np.random.default_rng(seed), noise_px=0.0)
        out["segments"] = [[a.tolist(), b.tolist()] for a, b in zip(wd.XA, wd.XB)]
    return out


DESIGN = {"A": [(-18, -12), (18, -12), (-18, 12), (18, 12)], "B": [(-15, 0), (15, 0), (0, -15), (0, 15)],
          "C": [(0, 0), (5, 0), (-2.5, 4.33)], "D": [(35, 0), (-40, 0), (0, 45)], "E": [(-15, 0), (15, 0), (0, 15)]}
RIG_TEXT = {"A": "one wall, 4 cameras", "B": "two walls, 2 + 2 cameras", "C": "3 cameras 5° apart, straight through",
            "D": "steep, 3 cameras at 35–45°", "E": "macro, f 40 000 px, 16 mm cloud"}


# ------------------------------------------------------------------ the tests' results
def tests_table(w0c, w0s, w0l, w1, w3n, w3m, w5, w7, w9):
    s01 = w5["summary"]["0.1"]
    s1 = w5["summary"]["1.0"]
    w7rows = [v for k, v in w7.items() if k != "pass"]
    n7 = sum(1 for r in w7rows if r.get("pass"))
    g = w0c["groups"]["no distortion"]
    return [
        {"test": "W0", "what": "derivatives of the projection, checked in SymPy", "rigs": "symbolic",
         "result": "77 proved, 2 skipped (limits, checked numerically in W1)", "status": "pass"},
        {"test": "W0", "what": "the new camera model against pyOpenLPT's C++ camera", "rigs": "2,000 random cameras",
         "result": f"projections within {g['proj_vs_cpp_px']:.1e} px, rays within {g['ray_dir']:.1e}",
         "status": "pass" if w0c["pass"] else "fail"},
        {"test": "W0", "what": "rotation matrix to vector and back", "rigs": "36,000 rotations",
         "result": f"within {w0l['worst']:.1e} rad (OpenCV: {w0l['cv2_worst']:.1e})", "status": "pass"},
        {"test": "W1", "what": "every derivative against finite differences", "rigs": "300 random sets, rigs A, B, D",
         "result": "within 3e-7 relative", "status": "pass" if w1["pass"] else "fail"},
        {"test": "W2", "what": "what the data can determine (Cramér–Rao)", "rigs": "A–E", "result": "section 04",
         "status": "reported"},
        {"test": "W3", "what": "fit from the truth: noise-free, then 40 noisy data sets per rig", "rigs": "A–E",
         "result": "noise-free RMS 1e-12 px; errors 0.97–0.99 of the bound; z SD 0.97–1.00",
         "status": "pass" if (w3n["pass"] and w3m["pass"]) else "fail"},
        {"test": "W4", "what": "how close a start must be", "rigs": "A–E, 1,000 runs", "result": "section 05",
         "status": "reported"},
        {"test": "W5", "what": "calibration from scratch, no hints", "rigs": "60 random rigs",
         "result": f"{s01['same_minimum']}/{s01['runs']} at 0.1 px, {s1['same_minimum']}/{s1['runs']} at 1 px "
                   f"({s1['equivalent']}/{s1['runs']} equivalent); target 49/50 and 10/10", "status": "fail"},
        {"test": "W6", "what": "the GUI's current calibration on the same data", "rigs": "A–D and 10 random rigs",
         "result": "section 03", "status": "reported"},
        {"test": "W7", "what": "bad data", "rigs": "A, B", "result": f"{n7} of {len(w7rows)} scenarios pass; the "
         "other is a marginal statistical miss", "status": "partial"},
        {"test": "W8", "what": "wrong plate thickness, water index, wand length, lens distortion", "rigs": "A, D",
         "result": "section 07", "status": "reported"},
        {"test": "W9", "what": "camera files and the GUI's wand CSV", "rigs": "A–E",
         "result": f"camFiles reload within {w9['camfile']['worst_px']:.1e} px; CSV exact",
         "status": "pass" if w9["pass"] else "fail"},
        {"test": "W10", "what": "real data", "rigs": "none sent", "result": "not run", "status": "na"},
    ]


def w5_records(w5):
    out = []
    for r in w5["runs"]:
        rig_t = prigs.random_rig(np.random.default_rng(r["seed"]))
        geo = rig_record(f"seed {r['seed']}", rig_t, n_rays=1)
        for c in geo["cams"]:
            c.pop("w")
            c.pop("hgt")
        rec = {k: r.get(k) for k in ("seed", "noise_px", "kind", "cams_per_window", "n_cams", "n_windows", "frames",
                                     "incidence_deg", "f_px", "h1_mm", "cloud_mm", "L", "seconds", "rms_px",
                                     "ref_rms_px", "max_abs_z", "worst", "active", "active_ref", "ok", "equivalent")}
        rec["bootstrap_s"] = (r.get("timing") or {}).get("bootstrap", {}).get("total")
        rec["stage4_s"] = (r.get("timing") or {}).get("stage4")
        pt = (r.get("starts") or {}).get("perturbed")
        if pt:                                    # the restarts perturbed about the answer (rigs with several walls)
            rnds = pt.get("rounds") or []
            dep = [x for x in (rnds[0].get("min_depth_mm") or []) if x is not None] if rnds else []
            rec["restarts"] = {"improved": bool(pt.get("improved")), "rms_before": pt.get("rms_before_px"),
                               "seconds": pt.get("seconds"), "pinned": bool(dep and min(dep) < 0.1)}
        rec["geo"] = geo
        out.append(rec)
    return out


def w6_records(w6):
    out = []
    for name, r in w6.items():
        rig, noise = name.split("_")
        rec = {"name": name, "rig": rig, "noise": float(noise.replace("px", "")), "gui": r["gui"], "new": r["new"]}
        for m in ("gui", "new"):
            if r[m].get("f_rel_err"):
                rec[m]["f_err_max"] = max(abs(v) for v in r[m]["f_rel_err"])
            rec[m].pop("f_rel_err", None)
            rec[m].pop("trace", None)
        out.append(rec)
    return out


def w2_rows(w2):
    out = []
    for name in "ABCDE":
        r = w2[name]
        p = r["points"]["grow_1"]
        out.append({"rig": name, "text": RIG_TEXT[name],
                    "f_pct": 100 * max(c["f_rel"] for c in r["cameras"]),
                    "normal_deg": max(w["normal_deg"] for w in r["windows"]),
                    "dist_mm": max(c["distance_mm"] for c in r["cameras"]),
                    "shape_um": 1e3 * p["shape_mm"], "noise3d_um": 1e3 * p["triangulation_noise_mm"],
                    "wand_cal_um": 1e3 * p["wand_length_mm"]["calibration"],
                    "wand_tot_um": 1e3 * p["wand_length_mm"]["total"],
                    "shape_2x_um": 1e3 * r["points"]["grow_2"]["shape_mm"],
                    "retri_px": p["projection"]["retriangulated_rms_px"], "f_px": r["f_px"], "h1_mm": r["h1_mm"]})
    return out


def w3_data(w3):
    rigs = {}
    z_all = []
    for name in "ABCDE":
        r = w3["part1"][name]
        z = np.array(r["z"])
        z_all += z.ravel().tolist()
        rigs[name] = {"eff": [v["eff"] for v in r["per_param"].values()], "names": list(r["per_param"].keys()),
                      "bias": [v["bias"] for v in r["per_param"].values()], "pooled": r["pooled_efficiency"],
                      "z_sd": r["z_sd"], "z_mean": r["z_mean"], "params": r["params"], "sets": r["sets"]}
    sweep = [{"sigma": float(k), "eff": v["pooled_efficiency"], "z_sd": v["z_sd"],
              "f_pct": 100 * v["rms_logf_rel"], "f_crb_pct": 100 * v["crb_logf_rel"]}
             for k, v in sorted(w3["part2"].items(), key=lambda kv: float(kv[0]))]
    return {"rigs": rigs, "z": z_all, "sweep": sweep}


def w4_matrix(w4):
    m = {}
    for r in w4:
        key = f"{r['factor']}|{r['level']}"
        m.setdefault(key, {}).setdefault(r["rig"], [0, 0])
        m[key][r["rig"]][0] += int(bool(r["ok"]))
        m[key][r["rig"]][1] += 1
    order = [("normal", [5, 10, 20, 30, 45, 60]), ("distance", [-0.9, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 0.9]),
             ("f", [0.1, 0.2, 0.4]), ("rotation", [5, 10, 20])]
    rows = []
    for factor, levels in order:
        for lv in levels:
            cells = m[f"{factor}|{lv}"]
            rows.append({"factor": factor, "level": lv, "cells": {k: v for k, v in sorted(cells.items())}})
    return rows


def w7_rows(w7):
    out = []
    for key, r in w7.items():
        if key == "pass":
            continue
        rig, scen = key.split("_", 1)
        out.append({"rig": rig, "scenario": scen, **{k: r.get(k) for k in (
            "rms_px", "rows", "active", "bad_rows", "bad_rejected", "good_rejected", "z_vs_clean_rms",
            "z_vs_clean_max", "z_expected_rms", "same_as_truth_start", "pass", "seconds", "noise")}})
    return out


def w8_rows(w8):
    out = []
    for r in w8:
        if r.get("scratch"):
            continue
        out.append({k: r.get(k) for k in ("rig", "case", "fit_L", "rms_px", "shape_mm", "wand_mean_mm", "wand_rms_mm",
                                          "L_fit", "L_sigma", "L_true", "k1_fit", "sigma_cam_px")})
    return out


def build_pages(text, fragment_path=None):
    """Fill the template's data block; write the standalone page (and the bare fragment if asked)."""
    page = TEMPLATE.read_text(encoding="utf-8")
    pat = re.compile(r'(<script id="fig-data" type="application/json">)(.*?)(</script>)', re.S)
    assert len(pat.findall(page)) == 1, "the template needs exactly one fig-data block"
    page = pat.sub(lambda m: m.group(1) + text.replace("</", "<\\/") + m.group(3), page)
    head, sep, body = page.partition("</style>\n")
    assert sep, "the template's head ends at its </style>"
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           + head + sep + "</head>\n<body>\n" + body + "</body>\n</html>\n")
    PAGE.write_text(doc, encoding="utf-8")
    if fragment_path:
        with open(fragment_path, "w", encoding="utf-8") as fh:
            fh.write(page)


def main(fragment_path=None):
    t_start = time.time()
    rigs = []
    for k, name in enumerate("ABCDE"):
        rig_t = prigs.RIGS[name]()
        rec = rig_record(f"Rig {name}", rig_t, n_rays=5, wands=60, seed=11 + k, labels=DESIGN[name])
        rec["key"] = name
        rec["text"] = RIG_TEXT[name]
        rigs.append(rec)
    w5 = load("w5_from_scratch.json")
    data = {
        "meta": {"generated": time.strftime("%Y-%m-%d %H:%M"), "noise_px": 0.1, "plate_mm": prigs.T_WALL,
                 "n": [prigs.N_AIR, prigs.N_ACRYLIC, prigs.N_WATER], "sensor": list(prigs.SENSOR)},
        "tests": tests_table(load("w0_model_vs_cpp.json"), None, load("w0_logmap.json"), load("w1_jacobians.json"),
                             load("w3_noise_free.json"), load("w3_monte_carlo.json"), w5, load("w7_bad_data.json"),
                             load("w9_files.json")),
        "rigs": rigs,
        "w5": w5_records(w5), "w5_summary": w5["summary"],
        "w6": w6_records(load("w6_vs_gui.json")),
        "w2": w2_rows(load("w2_observability.json")),
        "w3": w3_data(load("w3_monte_carlo.json")),
        "w4": w4_matrix(load("w4_basin.json")),
        "w7": w7_rows(load("w7_bad_data.json")),
        "w8": w8_rows(load("w8_constants.json")),
    }
    data = sig(data)
    FIG.mkdir(exist_ok=True)
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    with open(FIG / "figure_data.json", "w", encoding="utf-8") as fh:
        fh.write(text)
    build_pages(text, fragment_path)
    print(f"wrote {PAGE.name}" + (f" and {fragment_path}" if fragment_path else "")
          + f"; figure_data.json {len(text) / 1024:.0f} kB, {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fragment", help="also write the page without the document skeleton here")
    main(ap.parse_args().fragment)
