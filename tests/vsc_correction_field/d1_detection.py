"""D1: every GUI path detects as the tracking run does (9 Oct: "please fix the bugs on line detector, i want the gui
behaves the same as the actual tracking code run without gui").

    python d1_detection.py

The reference is the tracking run itself. Its log prints, every frame, the 2-D detections per camera of its first IPR
pass, which in the initial phase (frames 0-3) runs on the frame as loaded (main.cpp: ImageIO from the config's folder;
IPR: ObjectFinder2D.findObject2D(frame, settings, camera index)). Runs (read only):
- T1, the 15 px run (cases/cyl_20261007/data/runs/s3_final_m15): line detector, a noise per camera, 2 cameras;
- 20261008's final run (cases/cyl_20261008/data/runs/s3_final_c3): line detector, 3 cameras;
- the synthetic 3-camera run (tests/cylinder_selfcal/runs/Gcal4c3_n8_r3_tol1.0): line detector, noise 8 each;
- the 20260727 plate run (X:, stb_new_calib/new_cams, config.txt): the peak finder, 4 PINPLATE cameras (frames 0-1).
  Its log (4 Oct) comes from a library that printed the per-camera counts as its threads finished, not in camera
  order (consecutive lines are permuted differently; IPR.cpp now prints after the loop), so there the counts are
  compared as a set.
Per frame and camera, with the GUI's installed module:
- the run's call, here (OMP threads 12): its counts against the log;
- Validate before the fix: the same call without the camera index;
- the field VSC now: its _detect in a worker process set up as the VSC sets it up (one OMP thread), against the run's
  call bit for bit (u, v and the line's box lengths);
- the field VSC before the fix (OpenCV's frame; detectLines with the configured noise and lines settings, or the peak
  finder without the index), against the run's call;
- the plate/pinhole VSC's crops (T1, frame 0, 40 crops per camera): findObject2D with and without the camera index,
  against the line detector on the crop with the configured noise; on the plate run, with and without the index.
- Validate through the settings page (projects/d1_*), its printed counts against the log's at the start frame.
Writes d1_detection_out.txt and .json here.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np  # noqa: E402

import vcf_common as vc  # noqa: E402

PLATE_RUN = r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results\10Hz_r_b_1\stb_new_calib\new_cams"
RUNS = [
    ("T1, 2 cameras", os.path.join(vc.CASE_RUNS, "s3_final_m15"), "config_final_m15.json", [0, 1, 2, 3]),
    ("20261008, 3 cameras", os.path.join(vc.REPO, "cases", "cyl_20261008", "data", "runs", "s3_final_c3"),
     "config_final_c3.json", [0, 1, 2, 3]),
    ("synthetic, 3 cameras", os.path.join(vc.SELFCAL, "runs", "Gcal4c3_n8_r3_tol1.0"), "Gcal4c3_n8_r3_tol1.0.json",
     [0, 1, 2, 3]),
    ("plate 20260727, 4 cameras", PLATE_RUN, "config.txt", [0, 1]),
]


def log_counts(log_path, frames):
    """The first '2D detections per active camera' line after 'Initial phase at frame N', per frame N."""
    out, cur = {}, None
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = re.match(r"\s*Initial phase at frame (\d+)", ln)
            if m:
                cur = int(m.group(1))
                continue
            m = re.search(r"2D detections per active camera:\s*([\d\s]+)\(", ln)
            if m and cur is not None and cur not in out:
                out[cur] = [int(x) for x in m.group(1).split()]
    return {f: out.get(f) for f in frames}


def as_rows(objs):
    a = np.zeros((len(objs), 4))
    for i, o in enumerate(objs):
        a[i, 0], a[i, 1] = o._pt_center[0], o._pt_center[1]
        if getattr(o, "_has_shape", False):
            a[i, 2], a[i, 3] = o._len_h_px, o._len_v_px
    return a


def run_setup(lpt, config_path):
    """The run's setup: settings, the object config, an ImageIO per camera (main.cpp)."""
    settings = lpt.BasicSetting()
    with contextlib.redirect_stdout(io.StringIO()):
        settings.readConfig(config_path)
    cfg = lpt.TracerConfig() if settings._object_types[0] == "Tracer" else lpt.BubbleConfig()
    cfg.readConfig(settings._object_config_paths[0], settings)
    root = os.path.dirname(os.path.abspath(config_path)).replace("\\", "/") + "/"
    ios = []
    for p in settings._image_file_paths:
        im = lpt.ImageIO()
        im.loadImgPath(root, p)
        ios.append(im)
    return settings, cfg, ios


def old_field_detect(lpt, config_path, c, frame):
    """The field VSC's detection before the fix (copied from its _detect and _image_list of 8 Oct)."""
    import cv2
    from modules import stb_config
    cfg = vc.load_json(config_path)
    tracer = cfg["objects"][0].get("tracer", {})
    detector = str(tracer.get("detector", "spots")).lower()
    lines = dict(tracer.get("lines", {}))
    noise = lines.pop("noise", None)
    n_cam = len(cfg["cameras"])
    if noise is None:
        noise_l = [0.0] * n_cam
    elif isinstance(noise, (int, float)):
        noise_l = [float(noise)] * n_cam
    else:
        noise_l = [float(x) for x in noise][:n_cam]
    max_int = [float(cc.get("max_intensity", 255)) for cc in cfg["cameras"]]
    lst = stb_config.resolve(config_path, cfg["image_lists"][c])
    with open(lst, encoding="utf-8") as fh:
        names = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    base, root = os.path.dirname(os.path.abspath(lst)), os.path.dirname(os.path.abspath(config_path))
    n = names[frame]
    if not os.path.isabs(n):
        in_root = os.path.normpath(os.path.join(root, n))
        n = in_root if os.path.exists(in_root) else os.path.normpath(os.path.join(base, n))
    img = cv2.imread(n, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if detector == "lines":
        if img.dtype != np.uint8:
            img = np.clip(img.astype(np.float64) * 255.0 / max(max_int[c], 1.0), 0, 255).astype(np.uint8)
        P = lpt.LineFinderParam()
        for k, v in lines.items():
            if hasattr(P, k):
                setattr(P, k, v)
        rows = lpt.detectLines(lpt.Image.from_numpy(np.ascontiguousarray(img)), float(noise_l[c]), P)
        return np.array(rows, float).reshape(-1, 14)[:, :4]
    settings, tcfg, _ios = run_setup(lpt, config_path)
    objs = lpt.ObjectFinder2D().findObject2D(lpt.Image.from_numpy(np.ascontiguousarray(img.astype(np.float64))), tcfg)
    return as_rows(objs)


def compare(a, b):
    """Identical arrays, else (count a, count b, the largest |difference| of the pairs matched within 0.5 px)."""
    if a.shape == b.shape and np.array_equal(a, b):
        return {"identical": True}
    if not len(a) or not len(b):
        return {"identical": False, "counts": [len(a), len(b)]}
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    j = d.argmin(axis=1)
    ok = d[np.arange(len(a)), j] <= 0.5
    return {"identical": False, "counts": [len(a), len(b)], "matched": int(ok.sum()),
            "max_dev_px": float(d[np.arange(len(a)), j][ok].max()) if ok.any() else None}


def _child_detect(jobs):
    """In a worker set up by field_vsc._worker_init: the field VSC's _detect for (camera, frame) jobs."""
    from modules.vsc import field_vsc as fv
    return [fv._detect(c, f) for c, f in jobs]


def crops(lpt, cfg, img, rows, c, noise_c, line_params, n=40, half=25):
    """The plate/pinhole VSC's crops (vsc_service: an lpt.Image of the box, pixel by pixel) around n detections."""
    H, W = img.getDimRow(), img.getDimCol()
    pick = rows[np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)]
    finder = lpt.ObjectFinder2D()
    res = {"crops": 0, "with_index_eq_configured_noise": 0, "without_index_eq_configured_noise": 0,
           "with_eq_without": 0}
    for u, v, _lx, _ly in pick:
        x0, x1 = max(0, int(u) - half), min(W, int(u) + half)
        y0, y1 = max(0, int(v) - half), min(H, int(v) + half)
        roi = lpt.Image(y1 - y0, x1 - x0, 0.0)
        for i in range(y1 - y0):
            for j in range(x1 - x0):
                roi[i, j] = img[y0 + i, x0 + j]
        a = as_rows(finder.findObject2D(roi, cfg, c))
        b = as_rows(finder.findObject2D(roi, cfg))
        res["crops"] += 1
        res["with_eq_without"] += int(compare(a, b)["identical"])
        if noise_c is not None:
            P = lpt.LineFinderParam()
            for k, val in line_params.items():
                if hasattr(P, k):
                    setattr(P, k, val)
            ref = np.array(lpt.detectLines(roi, float(noise_c), P), float).reshape(-1, 14)[:, :4]
            res["with_index_eq_configured_noise"] += int(compare(a, ref)["identical"])
            res["without_index_eq_configured_noise"] += int(compare(b, ref)["identical"])
    return res


def page_validate(lpt, label, config_path):
    """Validate on the settings page (the change), in a project folder holding a copy of the run's config.json."""
    from h1_vsc_thread import Immediate
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import gui.views.tracking_settings_view as ts
    proj = os.path.join(vc.HERE, "projects", "d1_" + re.sub(r"\W+", "_", label.split(",")[0].lower()).strip("_"))
    if os.path.exists(proj):
        shutil.rmtree(proj)
    os.makedirs(os.path.join(proj, "Results"))
    cfg = vc.load_json(config_path)
    cfg["output_dir"] = "Results"
    cfg["_comment"] = f"D1 (tests/vsc_correction_field): {config_path} as a GUI project, for Validate"
    vc.save_json(os.path.join(proj, "config.json"), cfg)
    Immediate.shown = []
    ts.QMessageBox = Immediate
    view = ts.TrackingSettingsView()
    view.project_path.setText(proj)
    app.processEvents()
    view._update_paths()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        view._validate_settings()
        app.processEvents()
    counts = [int(m.group(2)) for m in re.finditer(r"Camera (\d+): found (\d+) 2D objects", buf.getvalue())]
    view.deleteLater()
    app.processEvents()
    return counts, [m["title"] for m in Immediate.shown]


def main():
    lpt = vc.lib("gui")
    log = vc.Log("d1_detection")
    sig = lpt.ObjectFinder2D.findObject2D.__doc__.strip().splitlines()[0]
    log(f"D1 with the installed module {lpt.__file__}; findObject2D: {sig}; OMP threads here "
        f"{os.environ.get('OMP_NUM_THREADS')}")
    from modules.vsc import field_vsc as fv
    out = {"runs": []}
    ok_all = True
    for label, run_dir, cfg_name, frames in RUNS:
        config_path = os.path.join(run_dir, cfg_name)
        if not os.path.exists(config_path):
            log(f"{label}: {config_path} not found, skipped")
            continue
        logc = log_counts(os.path.join(run_dir, "log.txt"), frames)
        settings, cfg, ios = run_setup(lpt, config_path)
        n_cam = len(ios)
        is_json = config_path.endswith(".json")
        rec = {"run": label, "config": config_path, "frames": frames, "log_counts": logc, "per_frame": []}
        # the field VSC now, in a worker set up as the VSC sets it up
        field_new = {}
        if is_json:
            setup = {"config_path": config_path, "obj_type": settings._object_types[0]}
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=fv._worker_init,
                                     initargs=(setup,)) as ex:
                jobs = [(c, f) for f in frames for c in range(n_cam)]
                for (c, f), a in zip(jobs, ex.submit(_child_detect, jobs).result()):
                    field_new[(c, f)] = a
        for f in frames:
            row = {"frame": f, "log": logc[f], "run_call": [], "validate_before": [], "field_now": [],
                   "field_before": []}
            for c in range(n_cam):
                img = ios[c].loadImg(f)
                ref = as_rows(lpt.ObjectFinder2D().findObject2D(img, cfg, c))
                before = as_rows(lpt.ObjectFinder2D().findObject2D(img, cfg))
                row["run_call"].append(len(ref))
                row["validate_before"].append(len(before))
                if is_json:
                    row["field_now"].append(compare(field_new[(c, f)], ref))
                    row["field_before"].append(compare(old_field_detect(lpt, config_path, c, f), ref))
                if label.startswith("T1") and f == frames[0]:
                    tr = vc.load_json(config_path)["objects"][0]["tracer"]
                    lines = dict(tr.get("lines", {}))
                    noise = lines.pop("noise", None)
                    row.setdefault("crops", []).append(crops(lpt, cfg, img, ref, c, noise[c], lines))
                if label.startswith("plate") and f == frames[0]:
                    row.setdefault("crops", []).append(crops(lpt, cfg, img, ref, c, None, {}, n=40))
            rec["per_frame"].append(row)
            same = (sorted(row["log"] or []) == sorted(row["run_call"])) if label.startswith("plate") else \
                row["log"] == row["run_call"]
            okf = same and all(x["identical"] for x in row["field_now"])
            ok_all &= okf
            fmt = lambda xs: "/".join(str(x) for x in xs)  # noqa: E731
            msg = (f"{label}, frame {f}: the run's log {fmt(row['log'] or [])}; the run's call here "
                   f"{fmt(row['run_call'])}; Validate before the fix {fmt(row['validate_before'])}")
            if is_json:
                msg += ("; the field VSC now: " + ("identical to the run's call" if all(x["identical"] for x in
                                                                                        row["field_now"])
                                                   else f"DIFFERENT {row['field_now']}")
                        + "; the field VSC before: " + ", ".join("identical" if x["identical"] else
                                                                 f"{x.get('counts')} max dev {x.get('max_dev_px')}"
                                                                 for x in row["field_before"]))
            log(msg + f" -> {'PASS' if okf else 'FAIL'}")
            for k, cr in enumerate(row.get("crops", [])):
                log(f"  crops, camera {k}: {cr}")
                if label.startswith("T1"):
                    ok_all &= cr["with_index_eq_configured_noise"] == cr["crops"]
                else:
                    ok_all &= cr["with_eq_without"] == cr["crops"]
        if is_json:
            counts, boxes = page_validate(lpt, label, config_path)
            rec["page_validate"] = {"counts": counts, "boxes": boxes}
            okp = counts == logc[frames[0]]
            ok_all &= okp
            log(f"  Validate on the settings page (frame {frames[0]}): {fmt(counts)} against the log's "
                f"{fmt(logc[frames[0]] or [])} -> {'PASS' if okp else 'FAIL'}")
        out["runs"].append(rec)
    out["pass"] = bool(ok_all)
    log(f"D1: {'PASS' if ok_all else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
