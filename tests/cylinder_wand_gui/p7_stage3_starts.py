"""P7 (diagnosis before the fixes plan, 2026-10-06-fixes-plan.md): how alike are the starts' stage-3 results, which
fix F-b would refine only once?

    python tests/cylinder_wand_gui/p7_stage3_starts.py [T0|T2 ...]      (default: both)

The library unchanged; each run's delivered inputs as in P5 (per-camera weights do not enter stages 1-3).
cylinder_wand.calibrate's stages 1-3 by hand: bootstrap, then coarse_fit from each of the six short-listed starts.
Per start against start 0: the stage-3 RMS, whether the active set is the same, and the largest difference of each
camera's f (relative), position (mm) and orientation (rad), and of the wand length.
Writes p7_stage3_starts_out.txt and p7_stage3_starts.json beside this script.
"""
import importlib.util
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CASES = {"T0": os.path.join(REPO, "cases", "cyl_20261006_T0_wand"),
         "T2": os.path.join(REPO, "cases", "cyl_20261006_T2_wand")}


def load_case(run):
    spec = importlib.util.spec_from_file_location(f"case_{run}", os.path.join(CASES[run], "case.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cam_vec(c):
    """f, the projection centre (mm) and the rotation matrix of one fitted camera (cyl_model.Camera).  The matrices
    are compared entry by entry (about the angle, rad, for small differences): a rotation vector is unreliable near
    the half-turns these cameras sit at."""
    return float(c.f), np.asarray(c.C, float).ravel(), np.asarray(c.R, float).ravel()


def main():
    runs = sys.argv[1:] or ["T0", "T2"]
    lines, out = [], {}

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say("P7: the six starts' stage-3 results, against start 0")
    for run in runs:
        case = load_case(run)
        cfg = case.load_config()
        lib = case.library()
        cw = lib.cw
        cams = case.cam_ids(cfg)
        _dets, kept, _reasons, _medians, frames = case.load_selection(cfg)
        job = case.make_job(lib, cfg, frames, case.uv_array(kept, frames, cams))
        data, tube = job.data(), job.tube
        obs = cw.observations(data)
        layout = cw.Layout(data.n_cams, fit_k1=bool(job.fit_k1), fit_L=bool(job.fit_L))
        t0 = time.time()
        ranked, _binfo = cw.bootstrap(obs, tube, data.L, data.sizes, verbose=False, log=print, log_fits=False)
        res = []
        for k, (_cost, cams0) in enumerate(ranked):
            o = cw.Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids,
                       obs.active.copy())
            cams3, info3 = cw.coarse_fit(cams0, tube, o, data.L, layout, verbose=False, log=print)
            res.append((cams3, info3, o.active.copy()))
        say(f"\n{run}: {len(frames)} frames; stages 1-3 for {len(ranked)} starts in {time.time() - t0:.0f} s")
        c0, i0, a0 = res[0]
        rows = []
        say("  start  stage-3 rms (px)    d rms (rel)  same active set  max |d f|/f   max |d C| (mm)  max |d r| (rad)")
        for k, (c3, i3, a3) in enumerate(res):
            df = max(abs(cam_vec(a)[0] / cam_vec(b)[0] - 1.0) for a, b in zip(c3, c0))
            dC = max(float(np.max(np.abs(cam_vec(a)[1] - cam_vec(b)[1]))) for a, b in zip(c3, c0))
            dR = max(float(np.max(np.abs(cam_vec(a)[2] - cam_vec(b)[2]))) for a, b in zip(c3, c0))
            drms = i3["rms_px"] / i0["rms_px"] - 1.0
            same = bool(np.array_equal(a3, a0))
            say(f"  {k:5d}  {i3['rms_px']:.10f}  {drms:+.1e}      {str(same):5s}          {df:.1e}       {dC:.1e}"
                f"         {dR:.1e}")
            rows.append({"start": k, "rms_px": i3["rms_px"], "rel_rms": drms, "same_active": same,
                         "n_active": int(a3.sum()), "df_rel": df, "dC_mm": dC, "dR_rad": dR})
        out[run] = rows
    with open(os.path.join(HERE, "p7_stage3_starts_out.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "p7_stage3_starts.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
