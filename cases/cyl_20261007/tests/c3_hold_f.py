"""C3 (plan: 2026-10-07-plan.md, Phase C): T0 refitted with f held where the board puts it.

    python tests/c3_hold_f.py scan                 the full recording refitted with cam2's f held at each value of
                                                   F2_SCAN (cam1's f free), and the board scored on each fit
    python tests/c3_hold_f.py parts F2 [F1]        the halves and quarters refitted with the same f held
    python tests/c3_hold_f.py fit F2 [F1] NAME     one fit of the whole recording, into <out_dir>/checks/NAME

Why: C1-T found f undetermined by the wand (cam1 41,351-53,010 px, cam2 23,639-30,417 px over the halves and
quarters, each with its distance from the axis in proportion: the near-telecentric valley of 6 Oct).  C2b found
that the board, triangulated with the valley's fits, is square (its pitch along the tube's axis equal to its pitch
across it, and its rows and columns at 90 deg) only for one value of cam2's f (c2b_valley_board: both measures move
with cam2's f, by -1.0 % and -0.096 deg per 1,000 px, and hardly with cam1's).  This holds f with the option added
to cylinder_wand for it (calibrate(..., fix_f=...), Job.fix_f; tests/cylinder_wand/t14_hold_f.py).

Each fit starts from C1's cameras (in the calibration's gauge: camera 1 at azimuth 0 and height 0), with each held
camera's f set to its value and its distance from the axis scaled by the same factor (a move along the valley, which
keeps the magnification), and runs cylinder_wand_gui.run with config.calibrate's settings and job.fix_f.  The board
is scored as in c2b_valley_board.py (the pairs and maps of C2a).
Writes c3_hold_f_<mode>_out.txt and .json beside this script; the fits' GUI files in <out_dir>/checks/<name>/.
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import case  # noqa: E402

import numpy as np  # noqa: E402

F2_SCAN = [26000.0, 27000.0, 28000.0, 29000.0, 30000.0]


def start_cams(lib, held):
    """C1's cameras in the calibration's gauge, each camera k in `held` (index -> f) moved along the valley: f set,
    distance from the axis (and height) scaled by the same factor about the axis."""
    cfg = case.load_config()
    with open(os.path.join(cfg["out_dir"], "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    cams = [lib.cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"],
                          c["height"], np.array(c["dist"])) for c in res["cameras"]]
    cams = lib.cw.gauge_fix(cams)
    for k, f in held.items():
        s = float(f) / cams[k].f
        cams[k].f = float(f)
        C = cams[k].C.copy()
        cams[k].C = np.array([C[0] * s, C[1], C[2] * s])
    return lib.cw.gauge_fix(cams)


def run_fit(task):
    """(name, held {camera index: f}, frame range or None) -> (name, summary)."""
    name, held, rng = task
    cfg = case.load_config()
    lib = case.library()
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    cams_ids = case.cam_ids(cfg)
    uv = case.uv_array(kept, frames, cams_ids)
    if rng is not None:
        idx = [i for i, f in enumerate(frames) if rng[0] <= f <= rng[1]]
        frames = [frames[i] for i in idx]
        uv = uv[idx]
    job = case.make_job(lib, cfg, frames, uv)
    job.start = start_cams(lib, {int(k): float(v) for k, v in held.items()})
    job.fix_f = sorted(int(k) for k in held)
    folder = os.path.join(cfg["out_dir"], "checks", name)
    try:
        _sol, result, secs = case.run_calibration(lib, job, folder, log=lambda m: None)
    except Exception as e:
        return name, {"error": f"{type(e).__name__}: {e}"}
    s = case.summary(result)
    s["seconds"] = secs
    s["held"] = {str(k): v for k, v in held.items()}
    s["frame_range"] = rng
    s["folder"] = folder
    return name, s


def board_scores(folders):
    import c2b_valley_board as vb
    cfg = case.load_config()
    lib = case.library()
    data = vb.pairs_of_c1(lib, cfg)
    out = {}
    for name, folder in folders.items():
        _res, cams, tube = vb.load_cams(lib, folder)
        recs, steps, para = vb.score(lib, cams, tube, data)
        pa = float(np.mean([100 * (r["pitch_A"] / 3.0 - 1) for r in recs]))
        pb = float(np.mean([100 * (r["pitch_B"] / 3.0 - 1) for r in recs]))
        out[name] = {"pitch_A_pct": pa, "pitch_B_pct": pb, "B_minus_A_pct": pb - pa,
                     "angle_deg": float(np.mean([r["angle_deg"] for r in recs])),
                     "flat_rms_um": [r["flat_rms_um"] for r in recs], "steps_mm": [s["length_mm"] for s in steps],
                     "reproj_px": [[r["reproj1_px"], r["reproj2_px"]] for r in recs]}
    return out


def fmt_summary(s):
    c1, c2 = s["cameras"]
    return (f"f {c1['f_px']:8.0f} / {c2['f_px']:8.0f} px, D {c1['distance_mm']:7.1f} / {c2['distance_mm']:7.1f} mm, "
            f"f/D {c1['f_px'] / c1['distance_mm']:.3f} / {c2['f_px'] / c2['distance_mm']:.3f}; RMS {s['rms_px']:.4f} px "
            f"({s['active']}/{s['observations']}), wand length RMS {s['wand_free_rms_um']:.1f} um")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, f"c3_hold_f_{mode}"))
    t0 = time.time()
    if mode == "scan":
        tasks = [(f"c3_scan_f2_{int(f)}", {1: f}, None) for f in F2_SCAN]
        log(f"C3 scan: the whole recording refitted with cam2's f held at {F2_SCAN} px (cam1's f free)")
    elif mode == "parts":
        held = {1: float(sys.argv[2])}
        if len(sys.argv) > 3:
            held[0] = float(sys.argv[3])
        first, last = int(cfg["frames"]["first"]), int(cfg["frames"]["last"])
        n = last - first + 1
        tag = "_".join(f"f{k + 1}_{int(v)}" for k, v in sorted(held.items()))
        tasks = []
        for k in range(2):
            tasks.append((f"c3_{tag}_h{k + 1}", held, (first + k * n // 2, first + (k + 1) * n // 2 - 1)))
        for k in range(4):
            tasks.append((f"c3_{tag}_q{k + 1}", held, (first + k * n // 4, first + (k + 1) * n // 4 - 1)))
        log(f"C3 parts: the halves and quarters refitted with f held at {held} (camera index: f)")
    elif mode == "fit":
        held = {1: float(sys.argv[2])}
        if len(sys.argv) > 4:
            held[0] = float(sys.argv[3])
        name = sys.argv[-1]
        tasks = [(name, held, None)]
        log(f"C3 fit: the whole recording with f held at {held} (camera index: f) -> checks/{name}")
    else:
        raise SystemExit(f"unknown mode {mode}")
    res = {}
    with ProcessPoolExecutor(max_workers=min(len(tasks), 6)) as ex:
        for name, s in ex.map(run_fit, tasks):
            res[name] = s
    log(f"  {len(tasks)} fits in {time.time() - t0:.0f} s")
    folders = {}
    for name, s in res.items():
        if "error" in s:
            log(f"  {name}: FAILED: {s['error']}")
            continue
        log(f"  {name}: " + fmt_summary(s) + f" ({s['seconds']:.0f} s)")
        folders[name] = s["folder"]
    scores = board_scores(folders) if folders else {}
    log("\n  The board on each fit (c2b_valley_board's measures; a square board has B - A = 0 and angle 90 deg):")
    for name, b in scores.items():
        log(f"    {name}: pitch A {b['pitch_A_pct']:+.3f} %, B {b['pitch_B_pct']:+.3f} %, B - A {b['B_minus_A_pct']:+.3f} %; "
            f"angle {b['angle_deg']:.3f} deg; flatness RMS " + " ".join(f"{v:.1f}" for v in b["flat_rms_um"])
            + " um; steps " + " ".join(f"{v:.3f}" for v in b["steps_mm"]) + " mm; reprojection "
            + " ".join(f"{a:.2f}/{c:.2f}" for a, c in b["reproj_px"]) + " px")
    out = {"fits": res, "board": scores}
    if mode == "scan" and len(scores) >= 3:
        f2 = np.array([res[n]["cameras"][1]["f_px"] for n in scores])
        f1 = np.array([res[n]["cameras"][0]["f_px"] for n in scores])
        ba = np.array([scores[n]["B_minus_A_pct"] for n in scores])
        an = np.array([scores[n]["angle_deg"] - 90.0 for n in scores])
        pba = np.polyfit(f2, ba, 1)
        pan = np.polyfit(f2, an, 1)
        p1 = np.polyfit(f2, f1, 1)
        z_ba, z_an = -pba[1] / pba[0], -pan[1] / pan[0]
        out["zero"] = {"f2_square_pitch": float(z_ba), "f2_right_angle": float(z_an),
                       "f1_on_line_at_square_pitch": float(np.polyval(p1, z_ba)),
                       "slopes": {"B_minus_A_pct_per_1000px": 1e3 * float(pba[0]), "angle_deg_per_1000px": 1e3 * float(pan[0]),
                                  "f1_per_f2": float(p1[0])}}
        log(f"\n  Along the scan: B - A crosses 0 at cam2 f = {z_ba:,.0f} px (slope {1e3 * pba[0]:+.3f} % per 1,000 px), "
            f"the angle 90 deg at {z_an:,.0f} px (slope {1e3 * pan[0]:+.4f} deg per 1,000 px); cam1's f follows cam2's "
            f"as f1 = {p1[0]:.3f} f2 + {p1[1]:,.0f} px, so at {z_ba:,.0f}: cam1 f = {np.polyval(p1, z_ba):,.0f} px")
    log.save(out)


if __name__ == "__main__":
    main()
