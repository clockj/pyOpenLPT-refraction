"""P5 (diagnosis before the fixes plan, 2026-10-06-fixes-plan.md): does the workaround given to the user, the GUI's
Noise weights "Equal", stop the long stage 4 on the real data of 20261006/T0 and T2?

    python tests/cylinder_wand_gui/p5_equal_weights.py [T0|T2 ...]      (default: both)

The library unchanged.  Each run's delivered inputs exactly as its case's calibrate.py builds them (the case's
detections and selection, config.json's tube, wand and settings, six starts), with one change: noise_weights "none"
(the GUI's "Equal").  cylinder_wand_gui.run as the GUI runs it; its outputs go to data/p5_<run>_equal/ (not to X:).
Per start: the stage-4 rounds, rejections and time (parsed as in P1); then the result against the delivered one
(the case's calibrate.json): RMS, active detections, f, distance from the axis, the free wand length.
Writes p5_equal_weights_out.txt and p5_equal_weights.json beside this script.
"""
import importlib.util
import json
import os
import re
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CASES = {"T0": os.path.join(REPO, "cases", "cyl_20261006_T0_wand"),
         "T2": os.path.join(REPO, "cases", "cyl_20261006_T2_wand")}
ROUND = re.compile(r"round (\d+): rms ([\d.]+) px, (\d+) LM iterations, rejected (\d+), re-admitted (\d+)")
START = re.compile(r"start (\d+): stage 3 rms ([\d.]+) px -> stage 4 rms ([\d.]+) px, active (\d+)/(\d+)")


def load_case(run):
    spec = importlib.util.spec_from_file_location(f"case_{run}", os.path.join(CASES[run], "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    runs = sys.argv[1:] or ["T0", "T2"]
    lines, out = [], {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say("P5: the delivered inputs of T0 and T2 with Noise weights \"Equal\" (noise_weights \"none\"), six starts")
    for run in runs:
        case = load_case(run)
        cfg = case.load_config()
        lib = case.library()
        cams = case.cam_ids(cfg)
        _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
        uv = case.uv_array(kept, frames, cams)
        job = case.make_job(lib, cfg, frames, uv, noise_weights="none")
        folder = os.path.join(HERE, "data", f"p5_{run}_equal")
        os.makedirs(folder, exist_ok=True)
        log_path = os.path.join(folder, "calibration_log.txt")
        t0 = time.time()
        stamps = []
        with open(log_path, "w", encoding="utf-8") as fh:
            def progress(stage, rms, text):
                t = time.time() - t0
                fh.write(f"[{t:8.1f} s] {stage}: {text}\n")
                stamps.append((t, stage, text))
            sol = lib.cg.run(job, progress=progress)
        secs = time.time() - t0
        result = lib.cg.write_outputs(sol, job, folder)
        summ = case.summary(result)
        # stage-4 rounds per start, as P1 parses them
        starts, cur, t_first = [], [], None
        t_prev = 0.0
        for t, _stage, text in stamps:
            r = ROUND.search(text)
            if r and "coarse" not in text:
                if t_first is None:
                    t_first = t_prev
                cur.append({"rms": float(r.group(2)), "rejected": int(r.group(4)), "readmitted": int(r.group(5))})
            s = START.search(text)
            if s:
                starts.append({"start": int(s.group(1)), "rounds": len(cur), "stage4_rms": float(s.group(3)),
                               "active": int(s.group(4)), "n": int(s.group(5)),
                               "seconds": t - (t_first if t_first is not None else t),
                               "last_rounds": cur[-3:]})
                cur, t_first = [], None
            t_prev = t
        with open(os.path.join(CASES[run], "calibrate.json"), encoding="utf-8") as fh:
            dl = json.load(fh)["summary"]
        say(f"\n{run}: {len(frames)} frames; whole calibration {secs:.0f} s (delivered: see P1)")
        say("  start  rounds  seconds  stage-4 rms  active       last rounds: rms (rejected/re-admitted)")
        for st in starts:
            tail = "; ".join(f"{x['rms']:.4f} ({x['rejected']}/{x['readmitted']})" for x in st["last_rounds"])
            say(f"  {st['start']:5d}  {st['rounds']:6d}  {st['seconds']:7.1f}  {st['stage4_rms']:11.5f}  "
                f"{st['active']:5d}/{st['n']:<5d}  {tail}")
        say(f"  result, Equal against the delivered (Per camera):")
        say(f"    RMS {summ['rms_px']:.4f} px (delivered {dl['rms_px']:.4f}); active {summ['active']} of "
            f"{summ['observations']} (delivered {dl['active']}); free wand length RMS {summ['wand_free_rms_um']:.1f} "
            f"um (delivered {dl['wand_free_rms_um']:.1f})")
        for a, b in zip(summ["cameras"], dl["cameras"]):
            say(f"    {a['name']}: f {a['f_px']:.0f} px (delivered {b['f_px']:.0f}, {100 * (a['f_px'] / b['f_px'] - 1):+.2f} %), "
                f"distance {a['distance_mm']:.1f} mm (delivered {b['distance_mm']:.1f})")
        out[run] = {"seconds": secs, "starts": starts, "summary": summ, "delivered": dl}
    with open(os.path.join(HERE, "p5_equal_weights_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "p5_equal_weights.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
