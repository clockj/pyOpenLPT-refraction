"""W5 (B5): Pause in the detection dialog pauses the worker processes.  The dialog's Pause calls the GUI's
WandDetectionWorker.pause(), which stops reading the detection generator; before the fix every task was submitted
to the process pool at the start, so the pool went on detecting during the pause (only the progress bar paused).
Now a task is submitted only when a result has been read, at most 2 per worker in flight.

    python w5_pause.py

The images: W1's six T2 frames (data/w1/dark/, cameras 1 and 2, dark balls), cycled to 600 frames per camera
(1,200 tasks), detected in the "ellipse" mode with T2's settings.  Every task's start and end are logged by a
wrapper of run_detection_task that the worker processes load (data/w5/<run>/tasks_<pid>.log).  Runs, through
WandDetectionWorker as the GUI's Process All Frames starts it:
  - "unpaused": the reference;
  - "paused": pause() once 50 frames are detected, resume() 10 s later;
  - "paused, before the fix": the same with git HEAD's generator (before_lib/wand_calibrator.py).
Checks:
  - during the pause at most 2 x workers tasks start (those in flight when it began), none in its last 7 s;
  - the paused run then completes, with the raw detections and the wand points of the unpaused run, to the bit;
  - before the fix, tasks kept starting through the pause (for the record: the bug);
  - the GUI's end of a detection (_on_process_finished) clears the frames deleted before (B4: new points start
    with no deleted frame).
Writes w5_pause_out.txt and .json.
"""
import common as C  # noqa: I001

import importlib.util  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

N_FRAMES = 600
PAUSE_AT_FRAMES, PAUSE_FOR = 50, 10.0
log = C.Log("w5_pause")


def timed_task(args):
    """run_detection_task, its start and end appended to tasks_<pid>.log in $W5_LOG (one file per process)."""
    from modules.camera_calibration.wand_calibration.wand_calibrator import run_detection_task as real
    t0 = time.time()
    out = real(args)
    t1 = time.time()
    with open(Path(os.environ["W5_LOG"]) / f"tasks_{os.getpid()}.log", "a", encoding="utf-8") as fh:
        fh.write(f"{t0:.4f} {t1:.4f} {args[0]} {args[1]}\n")
    return out


def read_tasks(folder):
    rows = []
    for p in Path(folder).glob("tasks_*.log"):
        for line in p.read_text(encoding="utf-8").splitlines():
            a, b, f, c = line.split()
            rows.append((float(a), float(b), int(f), int(c)))
    return np.array(rows) if rows else np.zeros((0, 4))


def run(module, name, paths, pause):
    """One detection through module's WandDetectionWorker; returns the calibrator, the run's times and its log."""
    folder = C.DATA / "w5" / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    os.environ["W5_LOG"] = str(folder)
    module.run_detection_task = timed_task               # the generator submits the name it finds in its module
    cal = module.WandCalibrator()
    opts = {"open_px": 95, "rod_mask_px": 50.0, "guard_px": 3.0, "full": False}
    worker = module.WandDetectionWorker(cal, paths, "dark", 40, 120, 0.85, autosave_path=None, resume=False,
                                        detect_mode="ellipse", detect_opts=opts)
    done, prog = {}, {"cur": 0}
    worker.finished_signal.connect(lambda ok, msg: done.update(ok=ok, msg=msg, t=time.time()))
    worker.progress.connect(lambda cur, tot: prog.update(cur=cur))
    t_start = time.time()
    times = {"start": t_start}
    worker.start()
    if pause:
        # once PAUSE_AT_FRAMES frames are detected (the worker processes start one by one, and slowly)
        C.wait_for(lambda: prog["cur"] >= PAUSE_AT_FRAMES or time.time() - t_start > 60, 90, step_ms=10)
        worker.pause()
        times["pause"] = time.time()
        C.wait_for(lambda: time.time() - times["pause"] >= PAUSE_FOR, 60, step_ms=10)
        worker.resume()
        times["resume"] = time.time()
    C.wait_for(lambda: "ok" in done and worker.isFinished(), 900)
    times["end"] = done.get("t", float("nan"))
    return cal, times, read_tasks(folder), done


def same_nested(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same_nested(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        try:
            x, y = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
            return x.shape == y.shape and all(same_nested(u, v) for u, v in zip(x.ravel(), y.ravel()))
        except Exception:  # noqa: BLE001
            return False
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return a == b


def main():
    app = C.qt_app()  # noqa: F841
    dlg = C.Dialogs()
    import modules.camera_calibration.wand_calibration.wand_calibrator as new
    spec = importlib.util.spec_from_file_location("wand_calibrator_before", C.HERE / "before_lib" / "wand_calibrator.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules["wand_calibrator_before"] = old
    spec.loader.exec_module(old)

    src = {c: sorted(str(p) for p in (C.DATA / "w1" / "dark" / f"cam{c + 1}").glob("*.png")) for c in (0, 1)}
    if any(len(v) != 6 for v in src.values()):
        raise SystemExit("run w1_bright_balls.py first (its images)")
    paths = {c: [v[i % 6] for i in range(N_FRAMES)] for c, v in src.items()}
    workers = max(1, int((os.cpu_count() or 4) * 0.8))
    log(f"W5: {N_FRAMES} frames x 2 cameras ({2 * N_FRAMES} tasks), ellipse mode, {workers} worker processes; "
        f"pause {PAUSE_FOR:.0f} s once {PAUSE_AT_FRAMES} frames are detected")

    out = {}
    for name, module, pause in (("unpaused", new, False), ("paused", new, True), ("paused_before_fix", old, True)):
        cal, times, tasks, done = run(module, name, paths, pause)
        rec = {"seconds": times["end"] - times["start"], "tasks": len(tasks), "finished": done}
        if pause:
            tp, tr = times["pause"], times["resume"]
            st = tasks[:, 0] if len(tasks) else np.zeros(0)
            rec.update(started_in_pause=int(np.sum((st > tp) & (st < tr))),
                       started_in_last_7s=int(np.sum((st > tr - 7.0) & (st < tr))),
                       finished_in_pause=int(np.sum((tasks[:, 1] > tp) & (tasks[:, 1] < tr))) if len(tasks) else 0,
                       done_before_pause=int(np.sum(tasks[:, 1] <= tp)) if len(tasks) else 0,
                       last_start_after_pause_s=float(np.max(st[(st > tp) & (st < tr)]) - tp)
                       if np.any((st > tp) & (st < tr)) else 0.0)
        log(f"  {name}: {rec['seconds']:.1f} s, {rec['tasks']} tasks, {done.get('msg')}"
            + (f"; {rec['done_before_pause']} done before the pause; during it {rec['started_in_pause']} started "
               f"(the last {rec['last_start_after_pause_s']:.2f} s after it began; {rec['started_in_last_7s']} in its "
               f"last 7 s) and {rec['finished_in_pause']} finished" if pause else ""))
        out[name] = (cal, rec)

    ref, rec_u = out["unpaused"]
    cal, rec_p = out["paused"]
    _old_cal, rec_o = out["paused_before_fix"]
    log.check("during the pause at most 2 x workers tasks start (those in flight), none in its last 7 s",
              rec_p["started_in_pause"] <= 2 * workers and rec_p["started_in_last_7s"] == 0,
              f"{rec_p['started_in_pause']} started (limit {2 * workers}), {rec_p['started_in_last_7s']} in the last 7 s")
    log.check("the paused run completes with the unpaused run's raw detections and wand points, to the bit",
              rec_p["finished"].get("ok") and rec_p["tasks"] == rec_u["tasks"] == 2 * N_FRAMES
              and same_nested(cal.wand_data_raw, ref.wand_data_raw) and same_nested(cal.wand_points, ref.wand_points)
              and len(ref.wand_points) > 0,
              f"{len(cal.wand_data_raw)} frames detected, {len(cal.wand_points)} paired (unpaused: "
              f"{len(ref.wand_data_raw)}, {len(ref.wand_points)})")
    log.check("before the fix, tasks kept starting through the pause (the bug, for the record)",
              rec_o["started_in_last_7s"] > 0 or rec_o["done_before_pause"] + rec_o["finished_in_pause"]
              >= 2 * N_FRAMES - 2 * workers,
              f"{rec_o['started_in_pause']} started during the pause, {rec_o['started_in_last_7s']} in its last 7 s")

    # the GUI's end of a detection: frames deleted before are forgotten
    from modules.camera_calibration import CameraCalibrationView
    view = CameraCalibrationView()
    view.wand_num_cams.setValue(2)
    view.wand_images = {c: list(v) for c, v in paths.items()}
    view.wand_calibrator = cal
    view._removed_frames = {1, 2, 3}
    view._manual_del = {1: True, 7: False}
    cal.wand_points_filtered = {k: v for k, v in cal.wand_points.items() if k not in (1, 2, 3)}

    class _Dialog:
        def close(self):
            pass
    view._proc_dialog = _Dialog()
    view._busy_begin('wand_process_all', 'Processing wand frames')
    view._on_process_finished(True, "Detection completed.")
    log.check("the end of a detection clears the frames deleted before (new points start with none deleted)",
              not view._removed_frames and not view._manual_del and cal.wand_points_filtered is None,
              f"deleted {sorted(view._removed_frames)}, marks {view._manual_del}, filtered "
              f"{None if cal.wand_points_filtered is None else len(cal.wand_points_filtered)}; status "
              f"'{view.status_label.text()}'; messages {[m[:2] for m in dlg.messages]}")
    view.close()
    return log.save({k: v[1] for k, v in out.items()})


if __name__ == "__main__":
    ok = main()
    sys.stdout.flush()
    os._exit(0 if ok else 1)
