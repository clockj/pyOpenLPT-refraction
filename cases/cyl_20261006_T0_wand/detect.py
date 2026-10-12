"""S1: the balls in every frame of both cameras (plan: 2026-10-06-plan.md, section 2).

    python detect.py [--step N] [--workers W]

ball_ellipses.detect_wand_balls (the GUI's "ellipse" mode, unchanged) with config.detect, on every N-th frame
(default: all 10,640) of each camera, in W worker processes.  Writes every ball found, with its ellipse, edge RMS
and the overlap guard's verdict, to <out_dir>/detection/detections_all.csv (case.write_detections), and a summary to
detect_out.txt beside this script.  Which camera-frames are kept is decided later (case.classify).
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = case.load_config()
    log = case.Log(os.path.join(case.HERE, "detect"))
    det = cfg["detect"]
    workers = int(args.workers or det.get("workers", 8))
    frames = case.all_frames(cfg)[::args.step]
    cams = case.cam_ids(cfg)
    tasks = [(f, c, case.image_path(cfg, c, f), det) for f in frames for c in cams]
    log(f"S1 detection: {len(frames)} frames x {len(cams)} cameras = {len(tasks)} images, {workers} workers")
    log(f"  settings: {det}")
    t0 = time.time()
    dets = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (f, c, rows) in enumerate(ex.map(case.detect_task, tasks, chunksize=8)):
            dets[(f, c)] = rows
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                log(f"  {i + 1} / {len(tasks)} images, {el:.0f} s ({(i + 1) / el:.1f} images/s)")
    el = time.time() - t0
    path = case.detections_path(cfg)
    case.write_detections(path, dets)
    log(f"  done in {el:.0f} s ({len(tasks) / el:.1f} images/s); written {path}")
    for c in cams:
        keys = [k for k in dets if k[1] == c]
        n = {"unreadable": sum(1 for k in keys if dets[k] is None)}
        for m in (0, 1, 2):
            n[f"{m} balls"] = sum(1 for k in keys if dets[k] is not None and len(dets[k]) == m)
        n["more"] = sum(1 for k in keys if dets[k] is not None and len(dets[k]) > 2)
        n["pairs dropped by the guard"] = sum(1 for k in keys if dets[k] and any(r[7] for r in dets[k]))
        log(f"  cam{c}: " + ", ".join(f"{v} {k}" for k, v in n.items()))
    log.save()


if __name__ == "__main__":
    main()
