"""Figures of the cylinder wand-calibration tests: gathers the numbers the page draws.

Reads the saved results of T0-T9 (the *.json beside this script), rebuilds the rigs' geometry from their seeds
(rigs.reference_rig, rigs.irregular_rig, and rigs.random_rig with T5's seeds 5000 + k), recomputes the T4 z-scores
from t4_fits.json, and traces the reference rig's sensor-edge rays through the tube for the zoomed view.  Nothing
is re-fitted, so the page shows exactly the results the report quotes.

Writes figures/figure_data.json and figures/cylinder_wand_results.html: the page, built from figures/template.html
with the data filled in.  Open it in a browser; it needs nothing else (fonts fall back to the system's offline).
With --fragment PATH it also writes the page without the document skeleton (for publishing as an artifact).
Run: python tests/cylinder_wand/make_figures.py [--fragment PATH]   (refract env)
"""
import argparse
import json
import math
import re
import time

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

HERE = rigs.HERE
FIG = HERE / "figures"
TEMPLATE = FIG / "template.html"
PAGE = FIG / "cylinder_wand_results.html"


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


# ------------------------------------------------------------------ geometry (top view: x right, z up; axis = Y)
def hit_circle(p, d, R, far=False):
    """Parameter t > 0 where p + t d meets the circle |.| = R (2-D), the near or far crossing; None if it misses."""
    b = float(p @ d)
    c = float(p @ p) - R * R
    disc = b * b - c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    ts = [t for t in ((-b - s), (-b + s)) if t > 1e-9]
    if not ts:
        return None
    return max(ts) if far else min(ts)


def refract(d, nrm, n1, n2):
    """Vector Snell: unit d crossing a surface of unit normal nrm (either orientation) from index n1 to n2."""
    if d @ nrm > 0:
        nrm = -nrm
    ci = -float(d @ nrm)
    eta = n1 / n2
    k = 1.0 - eta * eta * (1.0 - ci * ci)
    if k < 0:
        return None
    t = eta * d + (eta * ci - math.sqrt(k)) * nrm
    return t / np.linalg.norm(t)


def camera_record(cam, tube):
    """A camera in the top view: position, height, distance, azimuth, f, and its field of view as two edge rays
    (the extreme sensor directions projected on the XZ plane) up to the outer wall."""
    C = cam.C
    p = np.array([C[0], C[2]])
    corners = [(0, 0), (cam.width - 1, 0), (0, cam.height - 1), (cam.width - 1, cam.height - 1),
               ((cam.width - 1) / 2, 0), ((cam.width - 1) / 2, cam.height - 1),
               (0, (cam.height - 1) / 2), (cam.width - 1, (cam.height - 1) / 2)]
    axis = cam.R[2]
    a0 = np.array([axis[0], axis[2]])
    a0 /= np.linalg.norm(a0)
    angs = []
    for u, v in corners:
        d = cam.R.T @ np.array([(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1.0])
        q = np.array([d[0], d[2]])
        q /= np.linalg.norm(q)
        angs.append(math.atan2(a0[0] * q[1] - a0[1] * q[0], a0 @ q))
    ends = []
    for a in (min(angs), max(angs)):
        c, s = math.cos(a), math.sin(a)
        q = np.array([c * a0[0] - s * a0[1], s * a0[0] + c * a0[1]])
        t = hit_circle(p, q, tube.radii[0])
        t = np.linalg.norm(p) if t is None else t
        ends.append((p + t * q).tolist())
    D = float(np.hypot(C[0], C[2]))
    roll = math.degrees(math.atan2(-cam.R[0, 1], -cam.R[1, 1]))      # rigs.look_at's roll (image y ~ -world Y)
    return {"x": C[0], "z": C[2], "h": C[1], "D": D, "az": math.degrees(math.atan2(C[0], C[2])) % 360.0,
            "f": cam.f, "w": cam.width, "hgt": cam.height, "fov": ends,
            "fov_deg": math.degrees(max(angs) - min(angs)), "roll": roll}


def edge_rays(cam, tube, n_rays=5):
    """Rays of the central sensor row (the plane y = 0 for an unrolled camera at height 0 aimed at the axis),
    traced in 2-D through air, acrylic and water to the far inner wall.  Returns polylines [[x, z], ...]."""
    out = []
    p0 = np.array([cam.C[0], cam.C[2]])
    n_air, n_acr, n_wat = tube.n
    R_o, R_i = tube.radii
    for u in np.linspace(0, cam.width - 1, n_rays):
        d3 = cam.R.T @ np.array([(u - cam.cx) / cam.f, 0.0, 1.0])
        assert abs(d3[1]) < 1e-12, "the central row must lie in the plane y = 0"
        d = np.array([d3[0], d3[2]])
        d /= np.linalg.norm(d)
        t = hit_circle(p0, d, R_o)
        p1 = p0 + t * d
        d = refract(d, p1 / R_o, n_air, n_acr)
        t = hit_circle(p1, d, R_i)
        p2 = p1 + t * d
        d = refract(d, p2 / R_i, n_acr, n_wat)
        t = hit_circle(p2, d, R_i, far=True)
        p3 = p2 + t * d
        out.append([p0.tolist(), p1.tolist(), p2.tolist(), p3.tolist()])
    return out


def rig_record(name, cams, tube, **extra):
    rec = {"name": name, "n": len(cams), "cams": [camera_record(c, tube) for c in cams]}
    rec.update(extra)
    return rec


# ------------------------------------------------------------------ per test
def camera_counts(t0, t1, t2, t5):
    by_n = {}
    for r in t2["rigs"]:
        by_n[r["n_cams"]] = by_n.get(r["n_cams"], 0) + 1
    t5n = {}
    for r in t5["rigs"]:
        t5n[r["n_cams"]] = t5n.get(r["n_cams"], 0) + 1
    media = sorted({r["media"] for r in t0["rigs"]})
    return [
        {"test": "T0", "what": "tube camera model against the C++ camera", "rigs": f"{len(t0['rigs'])} single cameras",
         "cams": "1", "total": len(t0["rigs"]),
         "note": f"random tubes of {min(media)}–{max(media)} media, with and without distortion"},
        {"test": "T1", "what": "derivatives", "rigs": f"{len(t1['rigs'])} single cameras", "cams": "1",
         "total": len(t1["rigs"]), "note": "random cameras, inside and beyond the fold radius"},
        {"test": "T2", "what": "precision by rig design (Cramér–Rao, no fitting)", "rigs": f"{len(t2['rigs'])} designs",
         "cams": "2–4", "total": sum(r["n_cams"] for r in t2["rigs"]),
         "note": ", ".join(f"{v} with {k} cameras" for k, v in sorted(by_n.items()))},
        {"test": "T3", "what": "recovery without noise", "rigs": "reference + irregular", "cams": "3 and 4",
         "total": 7, "note": "irregular: 4 cameras 0.6–1.2 m away, heights −60…+50 mm, rolled"},
        {"test": "T4", "what": "recovery with noise", "rigs": "reference × 20 data sets", "cams": "3", "total": 3,
         "note": "19 free parameters per fit"},
        {"test": "T5", "what": "calibration from scratch, no hints", "rigs": f"{len(t5['rigs'])} random rigs",
         "cams": "2–4", "total": sum(r["n_cams"] for r in t5["rigs"]),
         "note": ", ".join(f"{v} with {k} cameras" for k, v in sorted(t5n.items()))},
        {"test": "T6–T10", "what": "gauge, held-out data, bad data, tube constants, export", "rigs": "reference",
         "cams": "3", "total": 3, "note": "T8 also on a 10 000 px-wide sensor to reach the wall"},
    ]


def t2_rows(t2):
    labels = {
        "reference": ("Reference", "3 cameras 45° apart, 1094 mm away, ±20 mm cloud, 500 frames"),
        "cams=2": ("Cameras", "2 cameras"), "cams=4": ("Cameras", "4 cameras"),
        "spacing=30": ("Spacing", "30° apart"), "spacing=60": ("Spacing", "60° apart"),
        "spacing=90": ("Spacing", "90° apart"), "spacing=120": ("Spacing", "120° apart"),
        "D=500 (f scaled)": ("Distance", "500 mm away (f scaled to keep the view)"),
        "half=10": ("Wand cloud", "±10 mm cloud"), "half=40": ("Wand cloud", "±40 mm cloud"),
        "cam1 sensor-like: half=8": ("Wand cloud", "±8 mm cloud (what cam1's sensor sees)"),
        "frames=200": ("Frames", "200 frames"), "frames=2000": ("Frames", "2000 frames"),
        "4 cams at 90": ("Combined", "4 cameras 90° apart"),
        "3 cams at 90, D=500": ("Combined", "3 cameras 90° apart, 500 mm away"),
        "2 cams at 90": ("Combined", "2 cameras 90° apart"),
        "3 cams at 60, half=10": ("Combined", "3 cameras 60° apart, ±10 mm cloud"),
        "2 cams at 45, half=10": ("Combined", "2 cameras 45° apart, ±10 mm cloud"),
    }
    rows = []
    for r in t2["rigs"]:
        g, lab = labels[r["name"]]
        rows.append({"key": r["name"], "group": g, "label": lab, "n": r["n_cams"], "spacing": r["spacing_deg"],
                     "D": r["D_mm"], "half": r["half_mm"], "frames": r["frames"], "obs": r["observations"],
                     "f_pct": 100 * max(r["sigma_f_rel"]), "dist_mm": max(r["sigma_distance_mm"]),
                     "az_deg": max(r["sigma_azimuth_deg"]), "h_mm": max(r["sigma_height_mm"]),
                     "rot_deg": max(r["sigma_rotation_deg"]), "retri_px": r["unseen_retriangulated_rms_px"],
                     "stab_px": r["unseen_stabiliser_rms_px"]})
    order = ["Reference", "Cameras", "Spacing", "Distance", "Wand cloud", "Frames", "Combined"]
    rows.sort(key=lambda q: (order.index(q["group"]), q["n"] if q["group"] == "Cameras" else 0,
                             q["spacing"] if q["group"] == "Spacing" else 0,
                             q["half"] if q["group"] == "Wand cloud" else 0,
                             q["frames"] if q["group"] == "Frames" else 0))
    return rows


def cam_of(d):
    return cm.Camera(np.array(d["R"]), np.array(d["C"]), d["f"], d["cx"], d["cy"], d["width"], d["height"],
                     np.array(d["dist"]))


def t4_data(t4fits, t4sum):
    truth = [cam_of(d) for d in t4fits["truth"]]
    lay = cw.Layout(3)
    assert lay.names == t4fits["layout_names"]
    zs, f_err = [], [[] for _ in truth]
    for fit in t4fits["fits"]:
        cams = [cam_of(d) for d in fit["fit"]]
        dl = rigs.local_delta(cams, truth)
        delta = np.zeros(lay.P)
        for c in range(3):
            for j in range(len(lay.cols)):
                if lay.pos[c, j] >= 0:
                    delta[lay.pos[c, j]] = dl[c][j]
        cov = np.array(fit["cov_crb"])
        s = np.sqrt(np.diag(cov))
        zs.append(delta / s)
        for c in range(3):
            i = lay.pos[c, lay.cols.index("logf")]
            f_err[c].append({"seed": fit["seed"], "err_pct": 100 * (cams[c].f / truth[c].f - 1),
                             "sig_pct": 100 * s[i], "z": delta[i] / s[i], "rms_px": fit["rms_px"]})
    Z = np.concatenate(zs)
    assert abs(Z.mean() - t4sum["z_mean"]) < 1e-9 and abs(Z.std(ddof=1) - t4sum["z_std"]) < 1e-9
    return {"z": Z.tolist(), "names": lay.names, "z_by_seed": [z.tolist() for z in zs], "f_err": f_err,
            "f_true": [c.f for c in truth], "summary": {k: t4sum[k] for k in
                                                        ("z_mean", "z_std", "white_mean", "white_std", "chi2_mean",
                                                         "chi2_dof", "fit_sigma_over_crb_median", "seeds")}}


def t7_rows(t7):
    u = t7["unseen"]
    rows = [{"label": "Unseen points, re-triangulated", "detail": "7 x 7 x 7 grid over the cloud, pooled over T4's 20 fits",
             "realised": u["retriangulated"]["realised_rms_px"], "predicted": u["retriangulated"]["predicted_rms_px"],
             "unit": "px", "ratio": u["retriangulated"]["ratio"]}]
    for h in t7["heldout_frames"]:
        rows.append({"label": f"Held-out frames, reprojection (set {h['seed'] + 1})", "detail": "5-fold: fit on 4/5 of 500 frames, test on the rest",
                     "realised": h["heldout_reproj_rms_px"], "predicted": h["heldout_reproj_pred_px"], "unit": "px",
                     "ratio": h["reproj_ratio"]})
        rows.append({"label": f"Held-out frames, wand length (set {h['seed'] + 1})",
                     "detail": "triangulated length − 10 mm", "realised": h["length_rms_mm"],
                     "predicted": h["length_pred_mm"], "unit": "mm", "ratio": h["length_ratio"]})
    return rows


def t9_rows(t9):
    out = []
    for c in t9["cases"]:
        out.append({"case": c["case"], "rms_px": c["rms_px"], "L_free_mm": c["L_free_mm"],
                    "L_sigma_mm": c["L_free_sigma_mm"], "L_offset_sigma": c["L_free_offset_sigma"],
                    "retri_px": c["unseen_retriangulated_px"], "stab_px": c["unseen_stabiliser_px"],
                    "shift_sigma": c["max_shift_sigma"], "shift_param": c.get("max_shift_param"),
                    "f_shift_pct": 100 * max((abs(v) for v in c.get("f_shift_rel", [0.0])), default=0.0)})
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
    tube = rigs.CASE_TUBE
    t0, t1, t2, t5 = load("t0_fast_vs_cpp.json"), load("t1_derivatives.json"), load("t2_observability.json"), \
        load("t5_from_scratch.json")

    # rigs: the reference (T3, T4, T6-T10), the irregular one (T3), and T5's 50 from their seeds
    ref = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    rng = np.random.default_rng(31)
    wd = rigs.wand_frames(ref, tube, 100, 10.0, rng, half=(20, 20, 20), noise_px=0.0)
    seg = [[a[0], a[2], b[0], b[2]] for a, b in zip(wd.XA, wd.XB)]
    reference = rig_record("Reference rig", ref, tube, used_in="T3, T4, T6–T10",
                           rays=[edge_rays(c, tube) for c in ref], segments=seg, cloud_half=20.0)
    irregular = rig_record("Irregular rig", rigs.irregular_rig(), tube, used_in="T3")
    t5rigs = []
    for r in t5["rigs"]:
        cams, centre, half = rigs.random_rig(np.random.default_rng(5000 + r["rig"]))
        assert len(cams) == r["n_cams"]
        assert np.allclose([math.hypot(c.C[0], c.C[2]) for c in cams], r["D_mm"], rtol=0, atol=1e-9)
        assert abs(half - r["half_mm"]) < 1e-12
        b = r.get("bootstrap") or {}
        res = {"ok": r["ok"], "rms_px": r["rms_px"], "ref_rms_px": r["ref_rms_px"],
               "max_diff_sigma": r["max_diff_sigma"], "rms_diff_px": r["rms_diff_px"],
               "affine_rms_px": b.get("affine_rms_px"), "affine_frames": b.get("affine_frames"),
               "candidates": b.get("candidates"), "mirror": (b.get("best") or {}).get("mirror"),
               "bootstrap_s": b.get("seconds"), "stage3_rms_px": r["stage3_rms_px"], "active": r["active"],
               "n_obs": r["n_obs"], "seconds": r["seconds"]}
        rec = rig_record(f"Rig {r['rig']}", cams, tube, k=r["rig"], centre=centre.tolist(), half=half, result=res)
        t5rigs.append(rec)

    t3 = load("t3_noiseless_recovery.json")
    t4 = t4_data(load("t4_fits.json"), load("t4_noisy_recovery.json"))
    t7 = load("t7_heldout.json")
    t8 = load("t8_robustness.json")
    t9 = load("t9_tube_constants.json")
    data = {
        "meta": {"generated": time.strftime("%Y-%m-%d %H:%M"), "noise_px": 0.1, "wand_mm": 10.0,
                 "radii": list(tube.radii), "n": list(tube.n), "fold_radius": tube.fold_radius(),
                 "t5_seconds": t5["seconds"], "t5_ok": t5["n_ok"], "t5_n": t5["n"]},
        "counts": camera_counts(t0, t1, t2, t5),
        "reference": reference, "irregular": irregular, "t5": t5rigs,
        "t2": t2_rows(t2), "t2_gate": t2["gate"],
        "t3": [{"rig": r["rig"], "final_rms_px": r["final_rms_px"], "max_rot_deg": r["max_rot_deg"],
                "max_centre_mm": r["max_centre_mm"], "max_f_rel": r["max_f_rel"]} for r in t3["runs"]],
        "t4": t4, "t7": t7_rows(t7), "t8": t8, "t9": t9_rows(t9),
    }
    data = sig(data)
    FIG.mkdir(exist_ok=True)
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    with open(FIG / "figure_data.json", "w", encoding="utf-8") as fh:
        fh.write(text)
    build_pages(text, fragment_path)
    print(f"wrote {PAGE.name}" + (f" and {fragment_path}" if fragment_path else ""))
    counts = {}
    for r in t5rigs:
        counts[r["n"]] = counts.get(r["n"], 0) + 1
    print(f"T5 rigs by camera count: {dict(sorted(counts.items()))}; cameras {sum(r['n'] for r in t5rigs)}")
    print(f"T4 pooled z: mean {np.mean(t4['z']):+.3f}, std {np.std(t4['z'], ddof=1):.3f} over {len(t4['z'])}")
    print(f"figure_data.json: {len(text) / 1024:.0f} kB, {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fragment", help="also write the page without the document skeleton here")
    main(ap.parse_args().fragment)
