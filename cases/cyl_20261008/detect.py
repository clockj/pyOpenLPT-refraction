"""C1, detection: the balls in every frame of the three cameras of T0 (plan: 2026-10-08-plan.md, Phase C; the 7 Oct
case's detect.py, adapted: each camera has its own detector settings and frames).

    python detect.py [--step N] [--workers W]

ball_ellipses.detect_wand_balls (the GUI's "ellipse" mode, unchanged) with case.det_settings (config.detect with
the camera's overrides: cam3's pixels are 1.48x smaller), on every N-th image number (default: all) that the camera
saved (cam3: 2396-13000), in W worker processes.  Writes every ball found, with its ellipse, edge RMS and the overlap
guard's verdict, to <out_dir>/detection/detections_all.csv (case.write_detections; frame = image number), and a
summary to detect_out.txt beside this script.  Which camera-frames are kept is decided later (case.classify).
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
    workers = int(args.workers or cfg["detect"].get("workers", 8))
    frames = case.all_frames(cfg)[::args.step]
    cams = case.cam_ids(cfg)
    tasks = []
    for f in frames:
        for c in cams:
            p = case.image_path(cfg, c, f)
            if p is not None:
                tasks.append((f, c, p, case.det_settings(cfg, c)))
    log(f"C1 detection: {len(frames)} image numbers, {len(tasks)} camera-frames, {workers} workers")
    for c in cams:
        log(f"  cam{c}: first image {case.first_image(cfg, c)}, settings {case.det_settings(cfg, c)}")
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
