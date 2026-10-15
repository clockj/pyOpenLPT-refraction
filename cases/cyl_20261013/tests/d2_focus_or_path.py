"""D2 (your question, 15 Oct: "so the main reason is that this experiment does not focus the wand points well,
right?"): is today's larger error in the detections (focus, light), or in where the wand went?

    python tests/d2_focus_or_path.py        (read-only: both runs' detections and calibration files)

For 8 Oct's run (cases/cyl_20261008, its delivered fit) and today's (C1), per camera, every used detection with:
  - its ball's aspect ratio (the tube's stretch, which depends on where the ball is in the tube);
  - its ball's distance from the tube's axis, r (mm), from the fit's 3-D wand ends;
  - its depth along the camera's view, relative to the camera's median depth (mm): blur grows away from focus;
  - its residual (du, dv).
Then, per camera and per variable:
  - each run's distribution (median, 10-90 %) and the residual RMS per bin;
  - "today's mix at 8 Oct's error": sqrt(sum_bins n_today * rms_8Oct^2 / sum n_today), the RMS today would have if
    each bin were as good as on 8 Oct.  If it reproduces today's RMS, the wand's path explains the difference; if it
    stays near 8 Oct's, today's detections are worse at the same place (focus, light).
Writes d2_focus_or_path_out.txt and d2_focus_or_path.json beside this script.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

BINS = {"aspect": [1.0, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 2.0],
        "r_mm": [0, 10, 20, 30, 40, 50, 60, 80],
        "depth_mm": [-80, -30, -20, -10, 0, 10, 20, 30, 80]}


def load_case(name):
    path = os.path.join(CASES, name, "case.py")
    spec = importlib.util.spec_from_file_location(f"case_{name}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_table(cm):
    """One row per used detection: camera, aspect, r, depth offset, residual^2."""
    cfg = cm.load_config()
    _d, kept, _r, _m, _f = cm.load_selection(cfg)
    folder = cfg["out_dir"]
    with open(os.path.join(folder, "calibration_data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    with open(os.path.join(folder, "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    ob = data["observations"]
    fid, cam, end = np.array(ob["frame_id"]), np.array(ob["cam_id"]), np.array(ob["end"])
    du = np.array([np.nan if x is None else x for x in ob["du"]], float)
    dv = np.array([np.nan if x is None else x for x in ob["dv"]], float)
    act = np.array(ob["active"], bool) & np.isfinite(du)
    idx = {int(f): i for i, f in enumerate(data["frame_ids"])}
    A, B = np.array(data["ends_A_mm"], float), np.array(data["ends_B_mm"], float)
    camd = {}
    for c in res["cameras"]:
        d = np.asarray(c["viewing_direction"], float)
        camd[c["name"]] = (np.asarray(c["centre_mm"], float), d / np.linalg.norm(d))
    rows = {"cam": [], "aspect": [], "r_mm": [], "depth": [], "r2": []}
    for i in np.nonzero(act)[0]:
        k = (int(fid[i]), int(cam[i]))
        if k not in kept or int(fid[i]) not in idx:
            continue
        P = (A, B)[int(end[i])][idx[int(fid[i])]]
        if not np.all(np.isfinite(P)):
            continue
        C, d = camd[f"cam{int(cam[i])}"]
        rows["cam"].append(int(cam[i]))
        rows["aspect"].append(cm.aspect(kept[k][int(end[i])]))
        rows["r_mm"].append(float(np.hypot(P[0], P[2])))           # the axis is Y
        rows["depth"].append(float((P - C) @ d))
        rows["r2"].append(float(du[i] ** 2 + dv[i] ** 2))
    t = {k: np.array(v) for k, v in rows.items()}
    t["depth_mm"] = t["depth"].copy()
    for c in set(t["cam"].tolist()):
        s = t["cam"] == c
        t["depth_mm"][s] = t["depth"][s] - np.median(t["depth"][s])
    return t


def fmt(b):
    return "-" if b["rms_px"] is None else f"{b['rms_px']:.2f}"


def per_bin(x, r2, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x < hi)
        out.append({"lo": lo, "hi": hi, "n": int(s.sum()),
                    "rms_px": float(np.sqrt(np.mean(r2[s]))) if s.sum() >= 20 else None})
    return out


def main():
    log = case.Log(os.path.join(HERE, "d2_focus_or_path"))
    runs = {"8 Oct": run_table(load_case("cyl_20261008")), "13 Oct": run_table(load_case("cyl_20261013"))}
    log("D2: the used detections of both runs' delivered fits, by the ball's stretch (aspect), its distance from the "
        "axis (r) and its depth offset along the camera's view")
    out = {}
    for c in (1, 2, 3):
        o, t = (runs[k] for k in ("8 Oct", "13 Oct"))
        so, st = o["cam"] == c, t["cam"] == c
        rms_o = float(np.sqrt(np.mean(o["r2"][so])))
        rms_t = float(np.sqrt(np.mean(t["r2"][st])))
        rec = {"rms_8Oct": rms_o, "rms_13Oct": rms_t, "vars": {}}
        log(f"  cam{c}: RMS 8 Oct {rms_o:.3f} px (n {int(so.sum())}), 13 Oct {rms_t:.3f} px (n {int(st.sum())})")
        for var, edges in BINS.items():
            xo, xt = o[var][so], t[var][st]
            bo, bt = per_bin(xo, o["r2"][so], edges), per_bin(xt, t["r2"][st], edges)
            # today's mix at 8 Oct's per-bin error (bins 8 Oct lacks keep today's own error)
            num, den, miss = 0.0, 0, 0
            for b8, b13 in zip(bo, bt):
                if b13["n"] == 0:
                    continue
                if b8["rms_px"] is None:
                    miss += b13["n"]
                    continue
                num += b13["n"] * b8["rms_px"] ** 2
                den += b13["n"]
            mix = float(np.sqrt(num / den)) if den else float("nan")
            share = (mix - rms_o) / (rms_t - rms_o) if rms_t != rms_o else float("nan")
            rec["vars"][var] = {"dist_8Oct": [float(np.percentile(xo, q)) for q in (10, 50, 90)],
                                "dist_13Oct": [float(np.percentile(xt, q)) for q in (10, 50, 90)],
                                "bins_8Oct": bo, "bins_13Oct": bt, "todays_mix_at_8Oct_error": mix,
                                "share_of_increase_explained": share, "today_in_bins_8Oct_lacks": miss}
            log(f"    {var}: 8 Oct p10/50/90 {np.round(rec['vars'][var]['dist_8Oct'], 2).tolist()}, 13 Oct "
                f"{np.round(rec['vars'][var]['dist_13Oct'], 2).tolist()}")
            log("      RMS per bin (8 Oct / 13 Oct, px; n): " + "; ".join(
                f"[{b8['lo']:g}, {b8['hi']:g}) {fmt(b8)} / {fmt(b13)} ({b8['n']}, {b13['n']})"
                for b8, b13 in zip(bo, bt) if b8["n"] or b13["n"]))
            log(f"      today's mix at 8 Oct's error: {mix:.3f} px -> explains {100 * share:.0f} % of the rise "
                f"{rms_o:.3f} -> {rms_t:.3f} px" + (f" ({miss} of today's detections in bins 8 Oct lacks)" if miss else ""))
        out[f"cam{c}"] = rec
    log.save(out)


if __name__ == "__main__":
    main()
