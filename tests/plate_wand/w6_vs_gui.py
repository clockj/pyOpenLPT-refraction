"""W6: the new calibration against the GUI's current refractive wand calibration, on the same data.

Cases: rigs A-D at 0.1 and 0.5 px, and ten W5 random rigs (seeds 1000-1009) at 0.1 px; 1,000 frames each, every frame
seen whole by every camera (the GUI drops a camera missing from any frame).  Each case folder w6_cases/<case>/ holds
the data as the GUI's CSV (wand.csv), case.json and truth.json.  Both calibrations run as separate processes
(w6_run.py), in parallel and single-threaded, with a 2 h limit each; the GUI gets the true focal lengths in its camera
table, the new calibration gets none.
Metrics, computed from each calibration's exported camFiles (loaded through the C++):
  - time, failures;
  - f error per camera;
  - 3-D shape error: 500 unseen points in the cloud, their exact pixels triangulated (maximum likelihood) with the
    calibration, then the best rigid motion onto the truth: RMS 3-D error (mm);
  - held-out wand lengths: 300 new wands, exact pixels, free triangulation: mean and RMS length error (mm);
  - reprojection of the unseen points (the truth moved into the calibration's frame by that rigid motion): RMS px.
Output: w6_vs_gui_out.txt / .json.  Usage: w6_vs_gui.py [prepare|run|evaluate|all] [case names...]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402

pm = prigs.pm
cm = prigs.cm
CASES_DIR = HERE / "w6_cases"
PY = sys.executable
LIMIT_S = 7200


def case_list():
    out = []
    for name in "ABCD":
        for noise in (0.1, 0.5):
            out.append({"name": f"{name}_{noise:g}px", "rig": name, "noise": noise})
    for k in range(10):
        out.append({"name": f"rand{1000 + k}_0.1px", "rig": f"seed{1000 + k}", "noise": 0.1})
    return out


def build_rig(spec):
    if spec["rig"].startswith("seed"):
        rng = np.random.default_rng(int(spec["rig"][4:]))
        return prigs.random_rig(rng)
    return prigs.RIGS[spec["rig"]]()


def cam_json(c):
    return {"R": c.R.tolist(), "C": c.C.tolist(), "f": c.f, "cx": c.cx, "cy": c.cy, "width": c.width,
            "height": c.height, "dist": c.dist.tolist()}


def prepare(specs):
    import plate_wand_io as pio
    CASES_DIR.mkdir(exist_ok=True)
    for spec in specs:
        d = CASES_DIR / spec["name"]
        d.mkdir(exist_ok=True)
        rig_t = build_rig(spec)
        rng = np.random.default_rng(zlib.crc32(spec["name"].encode()))
        wand = prigs.make_wand(rig_t, 1000, rng, noise_px=spec["noise"], all_cams=True)
        pio.write_gui_csv(str(d / "wand.csv"), wand.uv)
        case = {"spec": spec, "sizes": rig_t.sizes, "L": rig_t.L, "f_true": [c.f for c in rig_t.cams],
                "cam_to_window": rig_t.cam_win,
                "windows": [{"thickness_mm": list(w.t), "n": list(w.idx), "name": w.name} for w in rig_t.wins]}
        (d / "case.json").write_text(json.dumps(case, indent=1))
        truth = {"cams": [cam_json(c) for c in rig_t.cams], "cam_win": rig_t.cam_win,
                 "wins": [{"n": w.n.tolist(), "d": w.d, "t": list(w.t), "idx": list(w.idx)} for w in rig_t.wins],
                 "centre": rig_t.centre.tolist(), "half": rig_t.half.tolist(), "L": rig_t.L}
        (d / "truth.json").write_text(json.dumps(truth, indent=1))


def _launch(args):
    d, mode = args
    status = d / f"{mode}_status.json"
    if status.exists():
        return
    with open(d / f"{mode}_process.txt", "w", encoding="utf-8", errors="replace") as fh:
        t0 = time.time()
        try:
            subprocess.run([PY, str(HERE / "w6_run.py"), str(d), mode], stdout=fh, stderr=subprocess.STDOUT,
                           timeout=LIMIT_S, cwd=str(HERE.parents[1]))
        except subprocess.TimeoutExpired:
            status.write_text(json.dumps({"mode": mode, "ok": False, "error": f"time limit {LIMIT_S} s",
                                          "seconds": time.time() - t0}))


def run(specs, workers=30):
    jobs = []
    for spec in specs:
        d = CASES_DIR / spec["name"]
        jobs += [(d, "gui"), (d, "new")]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_launch, jobs))


def load_truth(d):
    t = json.loads((d / "truth.json").read_text())
    cams = [pm.Camera(np.array(c["R"]), np.array(c["C"]), c["f"], c["cx"], c["cy"], c["width"], c["height"],
                      np.array(c["dist"])) for c in t["cams"]]
    wins = [pm.Window(np.array(w["n"]), w["d"], tuple(w["t"]), tuple(w["idx"])) for w in t["wins"]]
    return cams, wins, t


def load_cal(d, mode, n_cams):
    import plate_wand_io as pio
    if mode == "gui":
        paths = [d / "gui_parent" / "camFile" / f"cam{k}.txt" for k in range(n_cams)]
    else:
        paths = [d / "new" / "camFile" / f"cam{k}.txt" for k in range(n_cams)]
    if not all(p.exists() for p in paths):
        return None
    out = [pio.load_camfile(str(p)) for p in paths]
    return [c for c, _w in out], [w for _c, w in out]


def evaluate_case(d):
    import plate_wand as pw
    cams_t, wins_t, t = load_truth(d)
    rig_t = prigs.TrueRig("t", cams_t, wins_t, t["cam_win"], np.array(t["centre"]), np.array(t["half"]), t["L"])
    rng = np.random.default_rng(66)
    P = prigs.cloud_points(rig_t, rng, 500)
    cpp_t = rig_t.cpp()
    U = np.stack([cm.cpp_project(cc, P) for cc in cpp_t], axis=1)          # (P, Nc, 2) exact pixels
    wand = prigs.make_wand(rig_t, 300, rng, noise_px=0.0, all_cams=True)
    res = {}
    for mode in ("gui", "new"):
        st_path = d / f"{mode}_status.json"
        st = json.loads(st_path.read_text()) if st_path.exists() else {"ok": False, "error": "not run"}
        r = {"ok": bool(st.get("ok")), "seconds": st.get("seconds"), "error": st.get("error")}
        cal = load_cal(d, mode, len(cams_t))
        if cal is None:
            r["ok"] = False
            r["error"] = r.get("error") or "no camFiles"
            res[mode] = r
            continue
        cams, wins = cal
        rig = pw.Rig(wins, list(range(len(cams))))           # one window per camera file (same planes per window)
        r["f_rel_err"] = [float(c.f / ct.f - 1.0) for c, ct in zip(cams, cams_t)]
        # unseen points: ML triangulation with the calibration
        n = len(P)
        cam_idx = np.tile(np.arange(len(cams)), n)
        pidx = np.repeat(np.arange(n), len(cams))
        uv = U.reshape(-1, 2)
        sel = np.all(np.isfinite(uv), axis=1)
        XA, XB, cnt = pw.triangulate(cams, rig, pw.Obs(pidx[sel], cam_idx[sel], np.zeros(sel.sum(), int), uv[sel], n,
                                                       len(cams), np.arange(n)))
        X0 = np.where(np.isfinite(XA), XA, 0.0)
        Xc, _r, _cov = pw.triangulate_ml(cams, rig, cam_idx[sel], uv[sel], pidx[sel], X0)
        good = np.all(np.isfinite(Xc), axis=1) & np.all(np.isfinite(XA), axis=1)
        Rr, tt, rms3 = prigs.best_rigid(Xc[good], P[good])
        r["shape_rms_mm"] = rms3
        # reprojection of the truth moved into the calibration's frame
        Pc = (P[good] - tt) @ Rr                                # inverse of X_true = R X_cal + t
        errs = []
        for c in range(len(cams)):
            pr = pm.project(pm.rows([cams[c]], np.zeros(len(Pc), int)), pm.wrows([wins[c]], np.zeros(len(Pc), int)), Pc)
            e = np.linalg.norm(pr.uv - U[good, c], axis=1)
            errs.append(e[np.isfinite(e)])
        allc = np.concatenate(errs)
        r["reproj_rms_px"] = float(np.sqrt(np.mean(allc ** 2))) if allc.size else float("nan")
        # held-out wands
        data_w = pw.WandData(wand.clean, wand.L, rig_t.sizes)
        obs_w = pw.observations(data_w)
        Lm = pw.wand_lengths(cams, rig, obs_w)
        err = Lm[np.isfinite(Lm)] - wand.L
        r["wand_mean_mm"] = float(err.mean())
        r["wand_rms_mm"] = float(np.sqrt(np.mean(err ** 2)))
        res[mode] = r
    return res


def evaluate(specs, log):
    out = {}
    for spec in specs:
        d = CASES_DIR / spec["name"]
        try:
            out[spec["name"]] = evaluate_case(d)
        except Exception as exc:  # noqa: BLE001
            out[spec["name"]] = {"error": repr(exc)}
            log(f"{spec['name']}: evaluation error {exc!r}")
            continue
        r = out[spec["name"]]
        for mode in ("gui", "new"):
            m = r[mode]
            if not m.get("ok") and "shape_rms_mm" not in m:
                log(f"{spec['name']:16s} {mode}: FAILED ({m.get('error')}) after {m.get('seconds') or 0:.0f} s")
                continue
            log(f"{spec['name']:16s} {mode}: {m.get('seconds') or 0:7.0f} s; f err max {100 * max(map(abs, m['f_rel_err'])):.3f} %; "
                f"3-D shape {1e3 * m['shape_rms_mm']:8.2f} um; wand {1e3 * m['wand_mean_mm']:+8.2f} um mean, "
                f"{1e3 * m['wand_rms_mm']:8.2f} um RMS; reprojection {m['reproj_rms_px']:.4f} px"
                + ("" if m.get("ok") else f"  (status: {m.get('error')})"))
    return out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = sys.argv[2:]
    specs = [s for s in case_list() if not names or s["name"] in names]
    log = prigs.Tee(HERE / ("w6_vs_gui_out.txt" if not names else "w6_vs_gui_partial_out.txt"))
    if what in ("prepare", "all"):
        prepare(specs)
        log(f"prepared {len(specs)} cases in {CASES_DIR}")
    if what in ("run", "all"):
        t0 = time.time()
        run(specs)
        log(f"ran {2 * len(specs)} calibrations in {time.time() - t0:.0f} s wall")
    if what in ("evaluate", "all"):
        res = evaluate(specs, log)
        (HERE / ("w6_vs_gui.json" if not names else "w6_vs_gui_partial.json")).write_text(
            json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
