"""Phase 5 (plan): end-to-end ablation on a synthetic line-image recording with its truth.

    python r5_ablation.py [--case r5_lines] [--runs R0 R1 R2 R3]

The case: synth_lines.py's (data/CASE: T1's density, the physical line shapes, the truth).  Runs, each in a fresh
process with the Python module (stbruns.run), scored by eval_truth.evaluate:
  - R0: today's code (build head1007) on the case's config (today's tracer finder, the Gaussian OTF estimated from
        the first images, square regions of radius_px, straight image lines);
  - R1: R0 plus Phase 1 (build p1: projection hints);
  - R2: R1 plus Phases 2 and 3 (build P23: curved image lines; the true shape model's OTF table as tracer.otf_file and
        regions from it, tracer.roi_from_otf);
  - R3: R2 plus Phase 4 (build P4: the line detector, tracer.detector "lines", and the shape check of the depth cue).
Judged (plan): R1 equals R0 (the same tracks to the printed resolution); R3 has fewer ghosts and a higher recall
than R0; the numbers are reported for the targets to be set with you.  Writes r5_ablation_out.txt and
r5_ablation.json; the runs in runs/r5_* (git-ignored).
"""
import argparse
import json
import os

import cstb
import eval_truth
import stbruns

BUILDS = {"R0": "head1007", "R1": "p1", "R2": "p123", "R3": "p4c"}


def config_for(case_dir, run, extra_tracer=None):
    with open(os.path.join(case_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    tr = cfg["objects"][0]["tracer"]
    if run in ("R2", "R3"):
        tr["otf_file"] = os.path.join(case_dir, "otf_truth.txt")
        tr["roi_from_otf"] = True
    if run == "R3":
        tr["detector"] = "lines"
        tr["shape_check"] = True
        with open(os.path.join(case_dir, "synth.json"), encoding="utf-8") as fh:
            noise = json.load(fh)["args"]["noise"]
        tr["lines"] = {"noise": noise}              # the images' noise (synth_lines.py's NOISE)
    if extra_tracer:
        tr.update(extra_tracer)
    path = os.path.join(case_dir, f"config_{run}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="r5_lines")
    ap.add_argument("--runs", nargs="+", default=["R0", "R1", "R2"])
    ap.add_argument("--builds", default=None, help="JSON overriding run -> build")
    args = ap.parse_args()
    builds = dict(BUILDS)
    if args.builds:
        builds.update(json.loads(args.builds))
    case_dir = os.path.join(cstb.DATA, args.case)
    log = cstb.Log(os.path.join(cstb.HERE, "r5_ablation"))
    log(f"Phase 5: ablation on data/{args.case}; runs {args.runs}")
    out_path = os.path.join(cstb.HERE, "r5_ablation.json")
    out = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {"runs": {}}
    out["case"] = args.case
    for run in args.runs:
        cfg = config_for(case_dir, run)
        rd = os.path.join(cstb.HERE, "runs", f"r5_{args.case}_{run}")
        info = stbruns.run(cfg, rd, builds[run])
        res = eval_truth.evaluate(rd, case_dir)
        res.pop("per_frame", None)
        out["runs"][run] = {"build": builds[run], "info": info, "score": res}
        log(f"  {run} (build {builds[run]}): exit {info['returncode']}, wall {info['wall_s']:.0f} s, "
            f"{res['frame_time_s_mean'] or float('nan'):.2f} s per frame")
        log(f"    per frame: trackable {res['trackable'] / max(1, res['frames_scored']):.0f}, reconstructed "
            f"{res['reconstructed'] / max(1, res['frames_scored']):.0f}, recall {res['recall']:.3f}, ghosts "
            f"{res['ghosts'] / max(1, res['frames_scored']):.0f} ({100 * res['ghost_share']:.1f} %), duplicates "
            f"{res['duplicates'] / max(1, res['frames_scored']):.0f}; position RMS (x, y, z) "
            + (", ".join(f"{1e3 * v:.1f}" for v in res["pos_rms_mm"]) + " um" if res["pos_rms_mm"] else "-"))
        log(f"    long tracks {res['long_tracks']} (ghost tracks {res['long_tracks_ghost']}), mean length "
            f"{res['long_track_mean_length']:.1f}, purity {res['long_track_purity_mean']:.3f} "
            f"(>= 0.9: {100 * res['long_track_purity_ge_0.9']:.1f} %), track recall {res['track_recall']:.3f}")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    if "R0" in out["runs"] and "R1" in out["runs"]:
        r0 = os.path.join(cstb.HERE, "runs", f"r5_{args.case}_R0")
        r1 = os.path.join(cstb.HERE, "runs", f"r5_{args.case}_R1")
        cmp_ = stbruns.compare(r0, r1)
        out["R0_vs_R1"] = cmp_
        diffs = {f: rec for f, rec in cmp_["files"].items() if rec != "identical"}
        log(f"  R0 vs R1: {cmp_['identical_files']} of {cmp_['compared_files']} files identical"
            + ("" if not diffs else "; largest differences " + "; ".join(
                f"{f}: rows {r.get('rows')}, " + ", ".join(f"{k} {v:.2g}" for k, v in (r.get("max_abs_diff") or {}).items()
                                                         if k.startswith("World") and v > 0)
                for f, r in list(diffs.items())[:4])))
    log.save(out)


if __name__ == "__main__":
    main()
