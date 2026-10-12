"""L2 and G1's end: T1's runs compared end to end, as the self-calibration's v3_compare.py compares them.

    python compare_t1.py [--name compare_t1] LABEL=FINAL[,CONTROL] ... [--ref LABEL]

Each run is a final run (frames 0-250) in the case's data/runs/, with its control (0-100, cam2 100 steps later: every
track a ghost) when there is one. T1 has no truth, so, with the study's and the case's functions (read only):
  - active long tracks per frame (final, control); at frame 100: the counts estimate of the ghost share (the control's
    active long tracks over the final's) and the swirl estimate (the share of long tracks at 100 whose mean |v_r|/|v|
    exceeds 0.3, over the control's); the real long tracks at 100 they leave; the whole run's long tracks and points;
  - recurrence against the reference: a track recurs when at least half its points have one of the other run's
    long-track points within 0.1 mm at the same frame;
  - the residual no 3-D move explains (with two cameras the offset across the epipolar curve), on every 5th frame:
    each long-track point (>= 10 points) projected through the run's own cameras (with their corrections), the line
    detector on the run's own frames (warped ones for a warped run), detections paired within 3 px; median and robust
    SD (modules/vsc/field_vsc.py's collection and its point_offsets).
Writes <name>_out.txt, .json and <name>.png here.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import vcf_common as vc

STUDY = os.path.join(vc.REPO, "tests", "cylinder_stb_study")
for _p in (STUDY, os.path.join(vc.CASE, "tests"), vc.CASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from r1_t1_rerun import cut, frame_log, recurrence  # noqa: E402  (the study's)
from t1e_ghost_control import swirl  # noqa: E402  (the case's)

CUT = 100


def read_tracks(run_dir):
    """The self-calibration's read_tracks (the last ConvergeTrack files: {(kind, id): rows (frame, X, Y, Z, ...)})."""
    import glob
    import re
    files = glob.glob(os.path.join(run_dir, "Results", "ConvergeTrack", "*.csv"))
    last = max(int(re.search(r"_(\d+)\.csv$", f).group(1)) for f in files)
    out = {}
    for kind in ("LongTrackActive", "LongTrackInactive", "ExitTrack"):
        p = os.path.join(run_dir, "Results", "ConvergeTrack", f"{kind}_{last}.csv")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        a = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
        if not len(a):
            continue
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
        ids, start = np.unique(a[:, 0], return_index=True)
        ends = np.r_[start[1:], len(a)]
        for tid, s, e in zip(ids, start, ends):
            out[(kind, int(tid))] = a[s:e, 1:]
    return out, last


def run_config(run_dir):
    import glob
    cfgs = [p for p in glob.glob(os.path.join(run_dir, "*.json")) if not p.endswith("run_info.json")]
    return cfgs[0]


def offsets(run_dir, workers=8):
    """The residual no 3-D move explains, on every 5th frame of the run, through its own cameras and frames."""
    from modules.vsc import field_vsc as fv
    cfg_path = run_config(run_dir)
    cfg = vc.load_json(cfg_path)
    root = os.path.dirname(cfg_path)
    res = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))  # noqa: E731
    tracer = cfg["objects"][0]["tracer"]
    lines = dict(tracer.get("lines", {}))
    noise = lines.pop("noise")
    setup = {"cam_files": [res(c["file"]) for c in cfg["cameras"]],
             "corr_files": [res(c["correction"]) if c.get("correction") else "" for c in cfg["cameras"]],
             "image_lists": [res(p) for p in cfg["image_lists"]], "detector": "lines",
             "noise": noise if isinstance(noise, list) else [noise] * len(cfg["cameras"]), "line_params": lines,
             "max_intensity": [255.0] * len(cfg["cameras"]), "pair_radius": [3.0] * len(cfg["cameras"]),
             "config_path": cfg_path}
    tracks = fv.read_last_tracks(os.path.join(run_dir, "Results", "ConvergeTrack"), 10)
    frames, _k, _per = fv.choose_frames(tracks, 1, stride=5)
    jobs = [(f, fv.points_at(tracks, f)) for f in frames]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=fv._worker_init,
                             initargs=(setup,)) as ex:
        recs = list(ex.map(fv._worker_frame, jobs))
    parts = []
    for g in recs:
        n = len(g["X"])
        if not n:
            continue
        mask = (np.isfinite(g["d"][:, :, 0]) & np.isfinite(g["P"][:, :, 0])
                & np.all(np.isfinite(g["J"].reshape(n, g["J"].shape[1], -1)), axis=2))
        keep = mask.sum(axis=1) >= 2
        parts.append(fv.FitData(mask[keep], g["d"][keep], (g["d"] - g["P"])[keep], g["J"][keep], g["L"][keep],
                                np.full(int(keep.sum()), int(g["frame"]))))
    fd = fv.FitData.concat(parts)
    off = fv.point_offsets(fd, fd.r)
    off = off[np.isfinite(off)]
    rsd = float(1.4826 * np.median(np.abs(off - np.median(off))))
    return {"points": int(len(off)), "frames": len(frames), "median": float(np.median(off)), "robust_sd": rsd,
            "rms": float(np.sqrt(np.mean(off ** 2))), "within_1px": float(np.mean(np.abs(off) <= 1.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--name", default="compare_t1")
    ap.add_argument("--no-offsets", action="store_true")
    a = ap.parse_args()
    vc.lib("vcf")   # the workers' pyopenlpt: build vcf (the corrections)
    log = vc.Log(a.name)
    var = []
    for v in a.variants:
        label, rest = v.split("=")
        parts = rest.split(",")
        var.append((label, parts[0], parts[1] if len(parts) > 1 else None))
    ref = a.ref or var[0][0]
    res, runs = {}, {}
    for label, rf, rc in var:
        df = os.path.join(vc.CASE_RUNS, rf)
        fl = frame_log(os.path.join(df, "log.txt"))
        lf = {f["frame"]: f["long"] for f in fl if f["long"] is not None}
        det = np.array([f["det"] for f in fl if f["det"]], float)
        tf, _last = read_tracks(df)
        v = {"final": rf, "control": rc, "frames": len(fl), "seconds_per_frame": float(np.mean([f["seconds"] for f in fl])),
             "detections_per_frame": det.mean(axis=0).round(1).tolist() if len(det) else None,
             "long_tracks_total": len(tf), "points_total": int(sum(len(t) for t in tf.values())),
             "long_final_series": {str(k): x for k, x in lf.items()}}
        if rc:
            dc = os.path.join(vc.CASE_RUNS, rc)
            cl = frame_log(os.path.join(dc, "log.txt"))
            lc = {f["frame"]: f["long"] for f in cl if f["long"] is not None}
            tc, last_c = read_tracks(dc)
            tf_cut, tc_cut = cut(tf, last_c), cut(tc, last_c)
            wf, _ = swirl(tf, last_c)
            wc, _ = swirl(tc, last_c)
            sh = lambda w: float(np.mean(w[np.isfinite(w)] > 0.3))  # noqa: E731
            g_counts = lc.get(CUT, np.nan) / max(lf.get(CUT, 1), 1)
            g_swirl = sh(wf) / max(sh(wc), 1e-9)
            n100 = len(tf_cut)
            v.update({"active_long_100": [lf.get(CUT), lc.get(CUT)], "ghost_counts": g_counts,
                      "ghost_swirl": g_swirl, "long_at_100": n100, "real_at_100_counts": n100 * (1 - g_counts),
                      "real_at_100_swirl": n100 * (1 - g_swirl),
                      "long_control_series": {str(k): x for k, x in lc.items()}})
        else:
            tf_cut = cut(tf, CUT)
            v.update({"active_long_100": [lf.get(CUT), None], "long_at_100": len(tf_cut)})
        if not a.no_offsets:
            v["offsets"] = offsets(df)
        res[label] = v
        runs[label] = {"cut": tf_cut, "all": tf}
        msg = (f"{label}: {rf}" + (f" / {rc}" if rc else "") + f"; {v['frames']} frames, {v['seconds_per_frame']:.1f} s "
               f"per frame; detections {v['detections_per_frame']}; active long tracks at {CUT}: "
               f"{v['active_long_100'][0]}")
        if rc:
            msg += (f" (control {v['active_long_100'][1]}): ghosts by the counts {100 * v['ghost_counts']:.0f} %, by "
                    f"the swirl {100 * v['ghost_swirl']:.0f} %; long tracks at {CUT} {v['long_at_100']}: real "
                    f"{v['real_at_100_counts']:.0f} (counts), {v['real_at_100_swirl']:.0f} (swirl)")
        else:
            msg += f"; long tracks at {CUT} {v['long_at_100']} (no control)"
        msg += f"; whole run {v['long_tracks_total']} long tracks, {v['points_total']} points"
        log(msg)
        if "offsets" in v:
            o = v["offsets"]
            log(f"  the residual no 3-D move explains ({o['points']} points on {o['frames']} frames): median "
                f"{o['median']:+.3f} px, robust SD {o['robust_sd']:.3f} px, RMS {o['rms']:.3f} px; within 1 px "
                f"{100 * o['within_1px']:.0f} %")
    rec = {}
    for lab in res:
        if lab == ref:
            continue
        for key, which in (("at_100", "cut"), ("whole", "all")):
            r_ab = recurrence(runs[lab][which], runs[ref][which])
            r_ba = recurrence(runs[ref][which], runs[lab][which])
            va, vb = np.array(list(r_ab.values())), np.array(list(r_ba.values()))
            rec[f"{lab}_{key}"] = {f"{lab}_in_{ref}": [int(va.sum()), int(len(va)), float(va.mean())],
                                   f"{ref}_in_{lab}": [int(vb.sum()), int(len(vb)), float(vb.mean())]}
            log(f"  recurrence {'frames 0-100' if key == 'at_100' else 'whole run'}: {lab}'s {len(va)} tracks in {ref}: "
                f"{int(va.sum())} ({100 * va.mean():.0f} %); {ref}'s {len(vb)} in {lab}: {int(vb.sum())} "
                f"({100 * vb.mean():.0f} %)")
    out = {"variants": res, "recurrence": rec, "ref": ref}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for lab, v in res.items():
        f = sorted(int(k) for k in v["long_final_series"])
        ax[0].plot(f, [v["long_final_series"][str(k)] for k in f], label=lab)
        if "long_control_series" in v:
            c = sorted(int(k) for k in v["long_control_series"])
            ax[0].plot(c, [v["long_control_series"][str(k)] for k in c], ls="--", lw=1,
                       color=ax[0].lines[-1].get_color())
    ax[0].set_xlabel("frame")
    ax[0].set_ylabel("active long tracks (dashed: control, all ghosts)")
    ax[0].legend(fontsize=8)
    labs = [l for l in res if "real_at_100_counts" in res[l]]
    x = np.arange(len(labs))
    ax[1].bar(x - 0.27, [res[l]["long_at_100"] for l in labs], 0.27, color="#bbbbbb", label="long tracks at 100")
    ax[1].bar(x, [res[l]["real_at_100_counts"] for l in labs], 0.27, color="#8172b3", label="real, by the counts")
    ax[1].bar(x + 0.27, [res[l]["real_at_100_swirl"] for l in labs], 0.27, color="#64b5cd", label="real, by the swirl")
    ax[1].set_xticks(x, labs)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(vc.HERE, f"{a.name}.png"), dpi=100)
    plt.close(fig)
    log.save(out)


if __name__ == "__main__":
    main()
