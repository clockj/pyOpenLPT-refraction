"""F3 and F5 (fixes plan 2026-10-06-fixes-plan.md, section 3): the calibrations of runs 20261006/T0 and T2 with the
fixed library.

    python tests/cylinder_wand_gui/f3_real_data.py [RUN:MODE[:noshare][:n2] ...]
        default: T0:camera T0:none T0:camera:noshare T0:camera:n2 T2:camera T2:none T2:camera:noshare T2:camera:n2

Each run's delivered inputs exactly as its case's calibrate.py builds them (the case's detections and selection,
config.json's tube, wand and settings, six starts), with the noise weights MODE ("camera": the GUI's "Per camera",
as delivered; "none": "Equal").  cylinder_wand.calibrate is called as cylinder_wand_gui.run calls it; ":noshare"
turns off F-b (share_equal_starts=False) for F5; ":n2" carries two starts (the GUI's default, thorough search
off) instead of the cases' six.  The outputs go to data/f3_<run>_<mode>[_noshare][_n2]/ (not to X:):
the GUI's files (cylinder_wand_gui.write_outputs) and calibration_log.txt.
Per calibration: the time; per start the stage-4 rounds (from the log, as P1), whether stage 4 was shared (F-b),
flipped (F-a; F-a3, the four-jumps rule of 7 Oct: noted from runs made after that day's rerun of F3, whose own notes
list only F-a2; its "final" record names the rule) or stopped on a repeated active set (F-a2); the result against the
delivered one (the case's calibrate.json) and, for "none", against P5 (the library before the fixes).
F3 passes when no start reaches the round limit (20) and every result is within the delivered one's standard errors
or explained; F5 when the shared and unshared runs keep the same detections and their cameras agree within 0.01 of
their standard errors.
Writes f3_real_data_out.txt and f3_real_data.json beside this script (each calibration adds its entry).
"""
import importlib.util
import json
import os
import re
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CASES = {"T0": os.path.join(REPO, "cases", "cyl_20261006_T0_wand"),
         "T2": os.path.join(REPO, "cases", "cyl_20261006_T2_wand")}
ROUND = re.compile(r"^round (\d+): rms ([\d.]+) px, (\d+) LM iterations, rejected (\d+), re-admitted (\d+)")
START = re.compile(r"^start (\d+): stage 3 rms ([\d.]+) px(?: -> stage 4 rms ([\d.]+) px, active (\d+)/(\d+)"
                   r"|, as start (\d+)'s)")
DEFAULT = ["T0:camera", "T0:none", "T0:camera:noshare", "T0:camera:n2", "T2:camera", "T2:none", "T2:camera:noshare",
           "T2:camera:n2"]
OUT_JSON = os.path.join(HERE, "f3_real_data.json")
OUT_TXT = os.path.join(HERE, "f3_real_data_out.txt")


def load_case(run):
    spec = importlib.util.spec_from_file_location(f"case_{run}", os.path.join(CASES[run], "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def one(run, mode, share, n_final=None):
    case = load_case(run)
    cfg = case.load_config()
    lib = case.library()
    cams = case.cam_ids(cfg)
    _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
    over = {"noise_weights": mode} if n_final is None else {"noise_weights": mode, "n_final": n_final}
    job = case.make_job(lib, cfg, frames, case.uv_array(kept, frames, cams), **over)
    tag = f"f3_{run}_{mode}" + ("" if share else "_noshare") + ("" if n_final is None else f"_n{n_final}")
    folder = os.path.join(HERE, "data", tag)
    os.makedirs(folder, exist_ok=True)
    lines = []
    t0 = time.time()
    with open(os.path.join(folder, "calibration_log.txt"), "w", encoding="utf-8") as fh:
        def log(msg):
            t = time.time() - t0
            fh.write(f"[{t:8.1f} s] {msg}\n")
            lines.append((t, str(msg).strip()))
        job.check()
        data, tube = job.data(), job.tube
        sol = lib.cw.calibrate(data, tube, start=None, fit_k1=bool(job.fit_k1), fit_L=bool(job.fit_L),
                               n_final=int(job.n_final), noise_weights=job.noise_weights, verbose=True, log=log,
                               log_fits=True, share_equal_starts=share)
    secs = time.time() - t0
    result = lib.cg.write_outputs(sol, job, folder)
    summ = case.summary(result)
    # stage 4 per start, from the log (as P1)
    starts, cur, t_first, t_prev = [], [], None, 0.0
    for t, text in lines:
        r = ROUND.match(text)
        if r:
            if t_first is None:
                t_first = t_prev
            cur.append({"rms": float(r.group(2)), "rejected": int(r.group(4)), "readmitted": int(r.group(5))})
        s = START.match(text)
        if s:
            shared_from = s.group(6)
            starts.append({"start": int(s.group(1)), "stage3_rms": float(s.group(2)), "rounds": len(cur),
                           "seconds": 0.0 if shared_from else t - (t_first if t_first is not None else t),
                           "stage4_rms": None if shared_from else float(s.group(3)),
                           "active": None if shared_from else int(s.group(4)),
                           "shared_from": None if shared_from is None else int(shared_from), "notes": []})
            cur, t_first = [], None
        t_prev = t
    # the fixes' own lines, per start (between two start lines)
    k = 0
    for _t, text in lines:
        if START.match(text):
            k += 1
        elif "flips between two states" in text and k < len(starts):
            starts[k]["notes"].append("F-a: " + text)
        elif "does not settle" in text and k < len(starts):       # the four-jumps rule (F7, 7 Oct)
            starts[k]["notes"].append("F-a3: " + text)
        elif "the active set repeats" in text and k < len(starts):
            starts[k]["notes"].append("F-a2: " + text)
    nw = sol.info.get("noise_weights") or {}
    rec = {"run": run, "mode": mode, "share": share, "n_final": int(job.n_final), "seconds": secs, "starts": starts,
           "final": {"rounds": sol.info.get("rounds"), "cycle": sol.info.get("cycle"), "flipped": nw.get("flipped"),
                     "weights_used": nw.get("used"), "stage4_shared": (sol.info.get("starts") or {}).get("stage4_shared"),
                     "why": (result.get("noise") or {}).get("why")},
           "summary": summ}
    return rec


def report(recs):
    with open(os.path.join(CASES["T0"], "calibrate.json"), encoding="utf-8") as fh:
        deliv = {"T0": json.load(fh)["summary"]}
    with open(os.path.join(CASES["T2"], "calibrate.json"), encoding="utf-8") as fh:
        deliv["T2"] = json.load(fh)["summary"]
    p5 = {}
    p5_path = os.path.join(HERE, "p5_equal_weights.json")
    if os.path.exists(p5_path):
        with open(p5_path, encoding="utf-8") as fh:
            p5 = {k: v["summary"] for k, v in json.load(fh).items()}
    out = ["F3 / F5: the calibrations of 20261006/T0 and T2 with the fixed library (six starts)"]
    for key in sorted(recs, key=lambda s: (s.split(":")[0], s.count("noshare"), s)):
        r = recs[key]
        s, d = r["summary"], deliv[r["run"]]
        out.append(f"\n{key}: {r['seconds']:.0f} s; final: {r['final']}")
        out.append("  start  stage-3 rms  stage-4 rounds  seconds  stage-4 rms  active       notes")
        for st in r["starts"]:
            if st["shared_from"] is not None:
                out.append(f"  {st['start']:5d}  {st['stage3_rms']:11.3f}  stage 4 shared with start {st['shared_from']}")
            else:
                out.append(f"  {st['start']:5d}  {st['stage3_rms']:11.3f}  {st['rounds']:14d}  {st['seconds']:7.1f}  "
                           f"{st['stage4_rms']:11.5f}  {st['active']:5d}       " + " | ".join(st["notes"]))
        lim = max([st["rounds"] for st in r["starts"] if st["shared_from"] is None] or [0])
        out.append(f"  most rounds in a start: {lim} (limit 20): {'PASS' if lim < 20 else 'FAIL'}")
        out.append(f"  result: RMS {s['rms_px']:.4f} px, active {s['active']} of {s['observations']}, free wand length "
                   f"RMS {s['wand_free_rms_um']:.1f} um")
        for name, ref in (("delivered", d),) + ((("P5 (before the fixes, equal weights)", p5[r["run"]]),)
                                                 if r["mode"] == "none" and r["run"] in p5 else ()):
            out.append(f"  against {name}: RMS {ref['rms_px']:.4f} px, active {ref['active']}")
            for a, b in zip(s["cameras"], ref["cameras"]):
                sf = b.get("f_sigma_px") or float("nan")
                sd = b.get("distance_sigma_mm") or float("nan")
                out.append(f"    {a['name']}: f {a['f_px']:.0f} px ({b['f_px']:.0f}; {100 * (a['f_px'] / b['f_px'] - 1):+.2f} %, "
                           f"{(a['f_px'] - b['f_px']) / sf:+.2f} sigma), distance {a['distance_mm']:.1f} mm "
                           f"({b['distance_mm']:.1f}; {(a['distance_mm'] - b['distance_mm']) / sd:+.2f} sigma)")
    # F5: shared against not shared
    for run in ("T0", "T2"):
        a, b = recs.get(f"{run}:camera"), recs.get(f"{run}:camera:noshare")
        if not (a and b):
            continue
        sa, sb = a["summary"], b["summary"]
        worst = 0.0
        for ca, cb in zip(sa["cameras"], sb["cameras"]):
            for v, sv in (("f_px", "f_sigma_px"), ("distance_mm", "distance_sigma_mm"),
                          ("height_rel_mm", "height_sigma_mm")):
                sig = cb.get(sv) or 0.0
                if sig > 0:
                    worst = max(worst, abs(ca[v] - cb[v]) / sig)
        same = sa["active"] == sb["active"]
        out.append(f"\nF5 {run}: shared {a['seconds']:.0f} s, not shared {b['seconds']:.0f} s; active {sa['active']} / "
                   f"{sb['active']}; RMS {sa['rms_px']:.6f} / {sb['rms_px']:.6f} px; largest camera difference "
                   f"{worst:.4f} sigma: {'PASS' if same and worst <= 0.01 else 'FAIL'}")
    return out


def main():
    args = sys.argv[1:] or DEFAULT
    recs = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as fh:
            recs = json.load(fh)
    for a in args:
        parts = a.split(":")
        run, mode, flags = parts[0], parts[1], set(parts[2:])
        print(f"F3: {a} ...", flush=True)
        recs[a] = one(run, mode, "noshare" not in flags, 2 if "n2" in flags else None)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(recs, fh, indent=1, default=float)
        print(f"F3: {a} done in {recs[a]['seconds']:.0f} s", flush=True)
    lines = report(recs)
    print("\n".join(lines))
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
