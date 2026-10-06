"""Tests A5-A7 (plan A4b): the GUI's config writer and every reader of a project's tracking config, offscreen.

    python a4b_gui_config.py

A5, the writer: the tracking-settings form (TrackingSettingsView, offscreen Qt) is filled with this case's settings
and with 30 random forms (fixed seed, Tracer and Bubble), and saved by HEAD's writer (config.txt + tracerConfig.txt
or bubbleConfig.txt) and by the new one (config.json) into twin project folders.  Both are read by the C++ code
(BasicSetting and TracerConfig/BubbleConfig, through build-py/Release's module, in a fresh process) and by
modules/stb_config.py: the settings must be identical (the object file's path aside), and the form's save interval
must reach the C++ settings.

A6, the readers: HEAD's code on the text project, and the new code on the text project and on its JSON twin, must
give the same values: TrackLoaderWorker (the tracks loaded), _check_project_files, _discover_cameras,
_read_tracer_radius, the output folder that _run_vsc and _load_track_statistics use, the post-processing loader, and
VSC's loaders (cameras, object type and radius, tracks, image loaders, detection config).  Also: a project without
a config falls back as before; a broken config gives a message, not a crash; a relative output folder other than
Results is found from any working directory (HEAD resolved it against the working directory).

A7, VSC's writers: _update_object_config and _update_config on the text project (HEAD's and the new code must write
identical files) and on the JSON project (the cameras and tolerances change as intended, config_backup.json holds
the original and is written once, and the C++ code reads the same settings as from the updated text project).

Writes results/a4b_gui_config.txt and .json; exit code 1 if anything fails.
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import os
import random
import shutil
import subprocess
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"
BUILD = os.path.join(REPO, "build-py", "Release")
TRACKS_SRC = os.path.join(RUNS, "s50_final3", "Results", "ConvergeTrack")
for p in (os.path.join(REPO, "gui"), REPO):
    sys.path.insert(0, p)

import openlpt  # noqa: E402,F401  (the GUI's entry: puts build-py/Release first on the path)
import pyopenlpt as lpt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from modules import stb_config  # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv)
LOG, CHECKS = [], []
MESSAGES = []


def say(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})
    say(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def _fake_box(kind):
    def box(*args, **kwargs):
        MESSAGES.append((kind, str(args[1]) if len(args) > 1 else "", str(args[2]) if len(args) > 2 else ""))
        return QMessageBox.StandardButton.Ok
    return box


# message boxes are modal: offscreen they would wait forever, so they are recorded instead
QMessageBox.information = _fake_box("information")
QMessageBox.warning = _fake_box("warning")
QMessageBox.critical = _fake_box("critical")


def load_head(rel, modname):
    """HEAD's version of a repository file, imported as modname (its package gives the relative imports)."""
    src = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout
    folder = os.path.join(RUNS, "a4b_head_src")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, rel.replace("/", "__"))
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


import modules.vsc  # noqa: E402,F401
import modules.post_processing  # noqa: E402,F401
from views import tracking_settings_view as new_settings  # noqa: E402
from views import tracking_view as new_tracking  # noqa: E402
from modules.vsc import vsc_service as new_vsc  # noqa: E402
from modules.post_processing import processor as new_proc  # noqa: E402

head_settings = load_head("gui/views/tracking_settings_view.py", "views._head_tracking_settings_view")
head_tracking = load_head("gui/views/tracking_view.py", "views._head_tracking_view")
head_vsc = load_head("modules/vsc/vsc_service.py", "modules.vsc._head_vsc_service")
head_proc = load_head("modules/post_processing/processor.py", "modules.post_processing._head_processor")


# ------------------------------------------------------------------------------------------------ projects and forms
def make_project(name, tracks=True, output="Results"):
    """runs/NAME: camFile/cam0-3.txt, imgFile/cam0-3ImageNames.txt (the local data, whose image paths are relative to
    a folder two levels below this one) and OUTPUT/ConvergeTrack with the first 3000 rows of each CSV of s50_final3."""
    proj = os.path.join(RUNS, name)
    if os.path.exists(proj):
        shutil.rmtree(proj)
    shutil.copytree(os.path.join(DATA, "camFile"), os.path.join(proj, "camFile"))
    os.makedirs(os.path.join(proj, "imgFile"))
    for i in range(4):
        shutil.copy(os.path.join(DATA, "imgLists", f"cam{i}ImageNames.txt"), os.path.join(proj, "imgFile"))
    if tracks:
        dst = os.path.join(proj, output, "ConvergeTrack")
        os.makedirs(dst)
        for fn in sorted(os.listdir(TRACKS_SRC)):
            with open(os.path.join(TRACKS_SRC, fn)) as f:
                rows = list(itertools.islice(f, 3001))
            with open(os.path.join(dst, fn), "w", newline="") as f:
                f.writelines(rows)
    return proj


CASE = {"type": "Tracer", "frames": (0, 300), "fps": 5000, "threads": 26, "voxel": 0.129,
        "volume": (-81.0, 54.0, -42.0, 59.0, -36.0, 52.5), "ipr_reduce": 1, "ipr_loops": 2, "ipr_reduced": 2,
        "tol2d": 2.0, "tol3d": 1.0, "stb_radius": 10.0, "stb_frames": 4, "stb_spacing": 30.0,
        "pf_grid": (51, 51, 51), "pf_radius": 25.0, "shake_width": 0.25, "shake_loops": 4, "shake_ghost": 0.01,
        "tracer_thresh": 20, "tracer_radius": 2.0, "bubble": (5.0, 50.0, 0.8), "resume": False, "resume_frame": 0,
        "save_interval": 500, "output": "Results"}


def random_form(rng):
    v = copy.deepcopy(CASE)
    v["type"] = "Bubble" if rng.random() < 0.3 else "Tracer"
    f0 = rng.randint(0, 300)  # TracerConfig estimates the OTF from frames f0..f0+4: 311 local images
    v["frames"] = (f0, f0 + rng.randint(1, 5000))
    v["fps"] = rng.randint(1, 10000)
    v["threads"] = rng.randint(1, 64)
    v["voxel"] = rng.uniform(1e-4, 1.0)
    v["volume"] = (rng.uniform(-500, -1), rng.uniform(1, 500), rng.uniform(-500, -1), rng.uniform(1, 500),
                   rng.uniform(-500, -1), rng.uniform(1, 500))
    v["ipr_reduce"], v["ipr_loops"], v["ipr_reduced"] = rng.randint(0, 2), rng.randint(1, 10), rng.randint(1, 10)
    v["tol2d"], v["tol3d"] = rng.uniform(0.1, 5), rng.uniform(0.1, 5)
    v["stb_radius"], v["stb_frames"], v["stb_spacing"] = rng.uniform(1, 50), rng.randint(1, 10), rng.uniform(1, 99)
    v["pf_grid"] = (rng.randint(2, 80), rng.randint(2, 80), rng.randint(2, 80))
    v["pf_radius"] = rng.uniform(1, 99)
    v["shake_width"], v["shake_loops"], v["shake_ghost"] = rng.uniform(0.01, 2), rng.randint(1, 10), rng.uniform(0.001, 0.5)
    v["tracer_thresh"], v["tracer_radius"] = rng.randint(0, 255), rng.uniform(0.5, 10)
    v["bubble"] = (rng.uniform(1, 20), rng.uniform(20, 200), rng.uniform(0.1, 1.0))
    v["resume"] = rng.random() < 0.3
    v["resume_frame"] = rng.randint(v["frames"][0], v["frames"][1])
    v["output"] = "Results" if rng.random() < 0.7 else "out_custom"
    return v


def fill_form(view, proj, v, cam_dir=None, img_dir=None):
    """Set the form as a user would; the camera folder (default proj/camFile) is set first, because choosing it
    overwrites the IPR tolerances and the view volume with estimates from the camera files."""
    view.project_path.setText(proj.replace("\\", "/"))
    view.image_path_display.setText((img_dir or os.path.join(proj, "imgFile")).replace("\\", "/"))
    view.camera_path_display.setText((cam_dir or os.path.join(proj, "camFile")).replace("\\", "/"))  # finds cam0-3.txt
    view.tri_err_3sigma_mm = None
    view.output_path.setText(os.path.join(proj, v["output"]).replace("\\", "/"))
    view.n_cam_spin.setValue(4)
    view.voxel_spin.setValue(v["voxel"])
    view.frame_start_spin.setValue(v["frames"][0])
    view.frame_end_spin.setValue(v["frames"][1])
    view.fps_spin.setValue(v["fps"])
    view.n_threads_spin.setValue(v["threads"])
    for w, x in zip((view.vol_x_min, view.vol_x_max, view.vol_y_min, view.vol_y_max, view.vol_z_min, view.vol_z_max),
                    v["volume"]):
        w.setValue(x)
    view.obj_type_combo.setCurrentText(v["type"])
    view.ipr_reduce_spin.setValue(v["ipr_reduce"])
    view.ipr_loop_spin.setValue(v["ipr_loops"])
    view.ipr_reduced_spin.setValue(v["ipr_reduced"])
    view.ipr_2d_tol.setValue(v["tol2d"])
    view.ipr_3d_tol.setValue(v["tol3d"])
    view.stb_initial_radius.setValue(v["stb_radius"])
    view.stb_initial_frames.setValue(v["stb_frames"])
    view.stb_avg_spacing.setValue(v["stb_spacing"])
    view.pred_grid_x.setValue(v["pf_grid"][0])
    view.pred_grid_y.setValue(v["pf_grid"][1])
    view.pred_grid_z.setValue(v["pf_grid"][2])
    view.pred_search_radius.setValue(v["pf_radius"])
    view.shake_width.setValue(v["shake_width"])
    view.shake_loops.setValue(v["shake_loops"])
    view.shake_ghost.setValue(v["shake_ghost"])
    view.tracer_int_thresh.setValue(v["tracer_thresh"])
    view.tracer_radius.setValue(v["tracer_radius"])
    view.bubble_min_rad.setValue(v["bubble"][0])
    view.bubble_max_rad.setValue(v["bubble"][1])
    view.bubble_sens.setValue(v["bubble"][2])
    view.resume_check.setChecked(v["resume"])
    view.resume_frame_spin.setValue(v["resume_frame"])
    if hasattr(view, "save_interval_spin"):
        view.save_interval_spin.setValue(v["save_interval"])


# ------------------------------------------------------------------------------------------------ C++ settings dump
DUMP_CODE = r'''
import sys, json
sys.path.insert(0, sys.argv[1])
import pyopenlpt as lpt
def lim(l):
    return [getattr(l, a) for a in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]]
for cfg_path in sys.argv[2:]:
    d = {"config": cfg_path}
    try:
        bs = lpt.BasicSetting()
        d["ok"] = bs.readConfig(cfg_path)
        for k in ["_n_cam", "_frame_start", "_frame_end", "_fps", "_n_thread", "_voxel_to_mm", "_output_path",
                  "_image_file_paths", "_object_types", "_object_config_paths", "_load_track", "_load_track_frame",
                  "_load_track_path", "_save_interval", "_profile"]:
            d[k] = getattr(bs, k)
        d["_axis_limit"] = lim(bs._axis_limit)
        d["cam_max_intensity"] = [c.max_intensity for c in bs._cam_list]
        t = bs._object_types[0]
        oc = lpt.TracerConfig() if t == "Tracer" else lpt.BubbleConfig()
        oc.readConfig(bs._object_config_paths[0], bs)
        s, sh, pf, ipr, sm = oc._stb_param, oc._shake_param, oc._pf_param, oc._ipr_param, oc._sm_param
        d["stb"] = [s._radius_search_obj, s._n_initial_frames, s._radius_search_track]
        d["shake"] = [sh._shake_width, sh._n_shake_loop, sh._thred_ghost, sh._shakewidth_min, sh._ratio_augimg]
        d["pf"] = [pf.nx, pf.ny, pf.nz, pf.r, pf.nBin_x, pf.nBin_y, pf.nBin_z, pf.is_smooth, pf.sigma_x,
                   pf.sigma_y, pf.sigma_z, lim(pf.limit)]
        d["ipr"] = [ipr.n_cam_reduced, ipr.n_loop_ipr, ipr.n_loop_ipr_reduced, ipr.n_obj2d_process_max]
        d["sm"] = [sm.match_cam_count, sm.idmap_cell_px, sm.tol_2d_px, sm.tol_3d_mm]
        if t == "Tracer":
            d["tracer"] = [oc._min_obj_int, oc._radius_obj]
            o = oc._otf._param
            d["otf_grid"] = [o.n_cam, o.nx, o.ny, o.nz, o.n_grid, o.dx, o.dy, o.dz]
        else:
            d["bubble"] = [oc._radius_min, oc._radius_max, oc._sense]
    except Exception as e:
        d["error"] = str(e)
    print("\n@@" + json.dumps(d), flush=True)  # on a line of its own, whatever C++ printed before
'''


def cpp_dump(paths):
    """Every bound setting the C++ code reads from each config, in one fresh process."""
    r = subprocess.run([PY, "-c", DUMP_CODE, BUILD] + list(paths), capture_output=True, text=True)
    out = [json.loads(ln[2:]) for ln in r.stdout.splitlines() if ln.startswith("@@")]
    if len(out) != len(paths):
        raise RuntimeError(f"dump failed: {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return out


def norm_paths(obj, projs):
    """Paths with the project folders replaced by <proj>, so twin projects compare equal."""
    if isinstance(obj, dict):
        return {k: norm_paths(x, projs) for k, x in obj.items()}
    if isinstance(obj, list):
        return [norm_paths(x, projs) for x in obj]
    if isinstance(obj, str):
        s = obj.replace("\\", "/")
        for p in projs:
            s = s.replace(p.replace("\\", "/"), "<proj>")
        return s
    return obj


def cpp_settings_equal(d_txt, d_json, projs):
    a = {k: v for k, v in d_txt.items() if k not in ("config", "_object_config_paths", "_save_interval")}
    b = {k: v for k, v in d_json.items() if k not in ("config", "_object_config_paths", "_save_interval")}
    a, b = norm_paths(a, projs), norm_paths(b, projs)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return not diff, diff


def module_settings(path):
    """stb_config's view of a config, in a form where the two formats compare: no bookkeeping keys, the default
    save interval written out, and load_tracks reduced to what is read when it is off."""
    c = stb_config.to_json_dict(stb_config.read(path))
    c.setdefault("save_interval", 500)
    c.setdefault("threads", 0)
    if not c["load_tracks"].get("enabled"):
        c["load_tracks"] = {"enabled": False}
    return c


# ------------------------------------------------------------------------------------------------ A5
def test_writer():
    say("\nA5 the writer: HEAD's config.txt + object file against the new config.json, from the same form")
    rng = random.Random(20261005)
    forms = [("case", CASE)] + [(f"random{k:02d}", random_form(rng)) for k in range(30)]
    head_view = head_settings.TrackingSettingsView()
    new_view = new_settings.TrackingSettingsView()
    pairs = []
    for name, v in forms:
        p_txt = make_project(f"a5_{name}_txt", tracks=False)
        p_json = make_project(f"a5_{name}_json", tracks=False)
        fill_form(head_view, p_txt, v)
        fill_form(new_view, p_json, v)
        n_msg = len(MESSAGES)
        head_view._save_configuration()
        new_view._save_configuration()
        boxes = MESSAGES[n_msg:]
        if any(k != "information" for k, _, _ in boxes):
            check(f"{name}: both saves succeed", False, str(boxes))
        txt, js = os.path.join(p_txt, "config.txt"), os.path.join(p_json, "config.json")
        pairs.append((name, v, p_txt, p_json, txt, js))
        if os.path.exists(os.path.join(p_json, "config.txt")) or os.path.exists(os.path.join(p_json, "tracerConfig.txt")):
            check(f"{name}: the new writer writes only config.json", False)
    check("the new writer writes only config.json (no config.txt or object file), 31 forms",
          all(sorted(os.listdir(p[3])) == ["camFile", "config.json", "imgFile"] for p in pairs))

    dumps = cpp_dump([p[4] for p in pairs] + [p[5] for p in pairs])
    d_txt, d_json = dumps[:len(pairs)], dumps[len(pairs):]
    n_same, n_types = 0, {"Tracer": 0, "Bubble": 0}
    for (name, v, p_txt, p_json, txt, js), a, b in zip(pairs, d_txt, d_json):
        if "error" in a or "error" in b:
            check(f"{name}: C++ reads both", False, f"{a.get('error')} | {b.get('error')}")
            continue
        same, diff = cpp_settings_equal(a, b, [p_txt, p_json])
        if not same:
            check(f"{name}: identical C++ settings", False, f"differ: {diff}")
            continue
        if b["_save_interval"] != v["save_interval"] or a["_save_interval"] != 500:
            check(f"{name}: save interval", False, f"{a['_save_interval']} / {b['_save_interval']}")
            continue
        n_same += 1
        n_types[v["type"]] += 1
    check(f"identical C++ settings from both writers ({n_types['Tracer']} Tracer and {n_types['Bubble']} Bubble "
          f"forms, every bound setting)", n_same == len(pairs), f"{n_same}/{len(pairs)}")
    case_json = d_json[0]
    say(f"    the case form, read by C++ from config.json: frames {case_json['_frame_start']}-"
        f"{case_json['_frame_end']}, voxel {case_json['_voxel_to_mm']}, tol {case_json['sm'][2]} px / "
        f"{case_json['sm'][3]} mm, save interval {case_json['_save_interval']}")

    n_mod = 0
    for name, v, p_txt, p_json, txt, js in pairs:
        a = norm_paths(module_settings(txt), [p_txt, p_json])
        b = norm_paths(module_settings(js), [p_txt, p_json])
        if a == b:
            n_mod += 1
        else:
            check(f"{name}: identical stb_config settings", False,
                  str(sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))))
    check("identical settings through modules/stb_config.py, 31 forms", n_mod == len(pairs), f"{n_mod}/{len(pairs)}")

    # the save interval field reaches the C++ settings
    p = make_project("a5_save_interval", tracks=False)
    v = dict(CASE, save_interval=200)
    fill_form(new_view, p, v)
    new_view._save_configuration()
    d = cpp_dump([os.path.join(p, "config.json")])[0]
    check("the form's save interval 200 reaches the C++ settings", d.get("_save_interval") == 200,
          str(d.get("_save_interval")))
    # the save message mentions an old config.txt left in the folder
    shutil.copy(pairs[0][4], os.path.join(p, "config.txt"))
    n_msg = len(MESSAGES)
    new_view._save_configuration()
    msg = MESSAGES[n_msg:]
    check("saving beside an old config.txt says that config.json is used",
          len(msg) == 1 and "no longer used" in msg[0][2], msg[0][2].replace("\n", " ") if msg else "no message")
    with open(pairs[0][5]) as f:
        say("    config.json of the case form:\n" + "".join("      " + ln for ln in f))
    return pairs


# ------------------------------------------------------------------------------------------------ A6
def run_track_loader(worker_cls, proj):
    out = {}
    w = worker_cls(proj)
    w.finished.connect(lambda *a: out.update(result=a))
    w.error.connect(lambda e: out.update(error=e))
    w.run()
    return out


def head_output_dir(proj):
    """HEAD's _run_vsc/_load_track_statistics: the line after "Output Folder Path" in config.txt."""
    config_path = os.path.join(proj, "config.txt")
    output_dir = os.path.join(proj, "Results")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "Output Folder Path" in line:
                    if i + 1 < len(lines):
                        path_line = lines[i + 1].strip()
                        if path_line and not path_line.startswith('#'):
                            if os.path.isabs(path_line):
                                output_dir = path_line
                            else:
                                output_dir = os.path.normpath(os.path.join(proj, path_line))
                            break
    return output_dir


def vsc_view(svc_cls, proj):
    """What VSC's loaders read from a project."""
    logs = []
    s = svc_cls(proj, log_callback=logs.append)
    r = {"cameras_ok": s._load_cameras()[0]}
    r["camera_paths"] = {k: os.path.normcase(os.path.normpath(p)) for k, p in s.camera_paths.items()}
    r["camera_params"] = {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in c.items()
                              if kk != "file_path"} for k, c in s.cameras.items()}
    s._load_object_config()
    r["obj_type"], r["obj_radius"] = s.obj_type, s.obj_radius
    tracks = s._load_tracks()
    r["n_tracks"] = len(tracks)
    r["tracks_hash"] = hash(json.dumps({str(k): [list(map(str, p)) for p in t] for k, t in sorted(tracks.items())}))
    r["image_loaders"] = sorted(s._load_image_loaders(lpt).keys())
    oc = s._load_obj_config_for_detection()
    r["detect_cfg"] = None if oc is None else ([oc._min_obj_int, oc._radius_obj] if s.obj_type == "Tracer"
                                              else [oc._radius_min, oc._radius_max, oc._sense])
    return r, logs


def view_readers(tracking_mod, settings_view, proj):
    tv = tracking_mod.TrackingView(settings_view=settings_view)
    tv.proj_path_edit.setText(proj)
    r = {"check": tv._check_project_files(proj)}
    tv._discover_cameras()
    r["num_cameras"] = tv.num_cameras
    r["cam_image_paths"] = {k: [os.path.normcase(x) for x in v] for k, v in tv.cam_image_paths.items()}
    tv.cached_tracer_radius = -1.0
    tv._read_tracer_radius(proj)
    r["tracer_radius"] = tv.cached_tracer_radius
    tv.deleteLater()
    return r


def test_readers(pairs):
    say("\nA6 the readers: HEAD's code on the text project, the new code on the text and the JSON project")
    v = dict(CASE, tracer_radius=2.5)  # a radius other than the readers' defaults
    p_txt, p_json = make_project("a6_txt"), make_project("a6_json")
    hv, nv = head_settings.TrackingSettingsView(), new_settings.TrackingSettingsView()
    fill_form(hv, p_txt, v)
    hv._save_configuration()
    fill_form(nv, p_json, v)
    nv._save_configuration()
    os.chdir(REPO)  # not the project folder: relative paths must be resolved against the project

    # TrackLoaderWorker
    h = run_track_loader(head_tracking.TrackLoaderWorker, p_txt)["result"]
    n1 = run_track_loader(new_tracking.TrackLoaderWorker, p_txt)["result"]
    n2 = run_track_loader(new_tracking.TrackLoaderWorker, p_json)["result"]
    check(f"TrackLoaderWorker: same tracks, object type and 2-D views ({len(h[0])} tracks, {h[2]})",
          len(h[0]) > 0 and h == n1 == n2)

    # TrackingView readers
    rh = view_readers(head_tracking, hv, p_txt)
    rn1 = view_readers(new_tracking, nv, p_txt)
    rn2 = view_readers(new_tracking, nv, p_json)
    check(f"_check_project_files: passes on both projects", rh["check"] == rn1["check"] == rn2["check"] == (True, ""),
          f"{rh['check']} {rn1['check']} {rn2['check']}")
    check(f"_discover_cameras: the same {rh['num_cameras']} cameras and image paths "
          f"({len(rh['cam_image_paths']['cam 0'])} per camera)",
          rh["num_cameras"] == 4 and rh["num_cameras"] == rn1["num_cameras"] == rn2["num_cameras"]
          and rh["cam_image_paths"] == rn1["cam_image_paths"] == rn2["cam_image_paths"]
          and len(rh["cam_image_paths"]["cam 0"]) > 0)
    check("_read_tracer_radius: 2.5 px from all three", rh["tracer_radius"] == rn1["tracer_radius"] ==
          rn2["tracer_radius"] == 2.5, f"{rh['tracer_radius']} {rn1['tracer_radius']} {rn2['tracer_radius']}")
    o_h = head_output_dir(p_txt)
    o_1 = new_tracking._output_dir(p_txt, *new_tracking._project_config(p_txt))
    o_2 = new_tracking._output_dir(p_json, *new_tracking._project_config(p_json))
    check("the output folder of _run_vsc / _load_track_statistics", os.path.normcase(o_h) == os.path.normcase(o_1)
          and os.path.normcase(o_2) == os.path.normcase(os.path.join(p_json, "Results")), f"{o_h} | {o_1} | {o_2}")

    # post-processing loader
    ph = run_track_loader(head_proc.DataLoaderWorker, p_txt)["result"]
    pn1 = run_track_loader(new_proc.DataLoaderWorker, p_txt)["result"]
    pn2 = run_track_loader(new_proc.DataLoaderWorker, p_json)["result"]
    same = repr(ph) == repr(pn1) == repr(pn2)
    check(f"post-processing loader: the same tracks and metadata ({len(ph[0])} tracks)", len(ph[0]) > 0 and same)

    # VSC loaders
    vh, _ = vsc_view(head_vsc.VSCService, p_txt)
    vn1, _ = vsc_view(new_vsc.VSCService, p_txt)
    vn2, _ = vsc_view(new_vsc.VSCService, p_json)
    keys = ["cameras_ok", "camera_paths", "camera_params", "obj_type", "obj_radius", "n_tracks", "tracks_hash",
            "image_loaders", "detect_cfg"]
    for k in keys:
        a, b, c = vh[k], vn1[k], vn2[k]
        if k == "camera_paths":  # twin projects: compare the paths inside the project
            a, b, c = ({i: os.path.relpath(x, p).lower() for i, x in d.items()}
                       for d, p in ((a, p_txt), (b, p_txt), (c, p_json)))
        check(f"VSC {k}: {str(a)[:70]}", a == b == c, "" if a == b == c else f"{a} | {b} | {c}")

    # without a config, HEAD and the new code fall back alike
    p_none = make_project("a6_noconfig")
    h0 = run_track_loader(head_tracking.TrackLoaderWorker, p_none)["result"]
    n0 = run_track_loader(new_tracking.TrackLoaderWorker, p_none)["result"]
    vh0, _ = vsc_view(head_vsc.VSCService, p_none)
    vn0, _ = vsc_view(new_vsc.VSCService, p_none)
    tv = new_tracking.TrackingView(settings_view=nv)
    ok_n0, msg_n0 = tv._check_project_files(p_none)
    tvh = head_tracking.TrackingView(settings_view=hv)
    ok_h0, _ = tvh._check_project_files(p_none)
    check("no config: the same tracks from Results/, the same VSC fallbacks (camFile/, defaults), and the run "
          "check stops", h0 == n0 and len(h0[0]) > 0 and all(vh0[k] == vn0[k] for k in keys if k != "camera_paths")
          and not ok_n0 and not ok_h0, msg_n0.replace("\n", " "))

    # a broken config.json: messages, no crash
    p_bad = make_project("a6_broken")
    with open(os.path.join(p_bad, "config.json"), "w") as f:
        f.write('{"frames": [0, 10], "fps": 5000,, }')
    ok_b, msg_b = tv._check_project_files(p_bad)
    nb = run_track_loader(new_tracking.TrackLoaderWorker, p_bad)
    logs_b = []
    svc = new_vsc.VSCService(p_bad, log_callback=logs_b.append)
    ok_v, msg_v, _ = svc.run()  # as the GUI's VSC worker runs it
    check("a broken config.json: the run check names it, the track loader falls back to Results/, and VSC stops "
          "at step 1 with the parser's message",
          not ok_b and "config.json cannot be read" in msg_b and "result" in nb and len(nb["result"][0]) > 0
          and not ok_v and msg_v.startswith("Cannot read config.json") and "invalid JSON" in msg_v
          and not any("[Step 2]" in m for m in logs_b),
          msg_b.replace("\n", " ")[:100] + " | VSC: " + msg_v[:90])

    # a relative output folder other than Results, read from another working directory
    p_ct = make_project("a6_custom_txt", output="out_custom")
    p_cj = make_project("a6_custom_json", output="out_custom")
    vc = dict(v, output="out_custom")
    fill_form(hv, p_ct, vc)
    hv._save_configuration()
    fill_form(nv, p_cj, vc)
    nv._save_configuration()
    hc = run_track_loader(head_tracking.TrackLoaderWorker, p_ct)
    nct = run_track_loader(new_tracking.TrackLoaderWorker, p_ct)
    ncj = run_track_loader(new_tracking.TrackLoaderWorker, p_cj)
    vct, _ = vsc_view(head_vsc.VSCService, p_ct)
    vnj, _ = vsc_view(new_vsc.VSCService, p_cj)
    check("output folder 'out_custom': found by the new loaders (HEAD's TrackLoaderWorker and VSC looked for it "
          "relative to the working directory and found no tracks)",
          len(nct["result"][0]) > 0 and nct["result"] == ncj["result"] and vnj["n_tracks"] > 0
          and len(hc.get("result", ({},))[0]) == 0 and vct["n_tracks"] == 0,
          f"HEAD {len(hc.get('result', ({},))[0])} tracks, VSC {vct['n_tracks']}; new {len(nct['result'][0])}, "
          f"VSC {vnj['n_tracks']}")
    tv.deleteLater()
    tvh.deleteLater()
    return p_txt, p_json


# ------------------------------------------------------------------------------------------------ A7
def files_of(proj, names):
    out = {}
    for n in names:
        p = os.path.join(proj, n)
        out[n] = open(p, "rb").read() if os.path.exists(p) else None
    return out


def vsc_update(svc_cls, proj, stats):
    s = svc_cls(proj, log_callback=lambda m: None)
    s.cameras = {i: {} for i in range(4)}
    s._update_object_config(stats)
    s._update_config()


def test_vsc_writers():
    say("\nA7 VSC's writers: the new tolerances and the optimised camera files")
    v = dict(CASE)
    projs = {}
    for name in ("a7_txt_head", "a7_txt_new", "a7_json"):
        p = make_project(name, tracks=False)
        shutil.copytree(os.path.join(p, "camFile"), os.path.join(p, "camFile_VSC"))
        for i in range(4):
            os.replace(os.path.join(p, "camFile_VSC", f"cam{i}.txt"), os.path.join(p, "camFile_VSC", f"vsc_cam{i}.txt"))
        projs[name] = p
    hv, nv = head_settings.TrackingSettingsView(), new_settings.TrackingSettingsView()
    for name in ("a7_txt_head", "a7_txt_new"):
        fill_form(hv, projs[name], v)
        hv._save_configuration()
    fill_form(nv, projs["a7_json"], v)
    nv._save_configuration()
    json_before = open(os.path.join(projs["a7_json"], "config.json"), "rb").read()

    stats = {"proj_tol": 0.734567, "triang_tol": 0.045678}
    vsc_update(head_vsc.VSCService, projs["a7_txt_head"], stats)
    vsc_update(new_vsc.VSCService, projs["a7_txt_new"], stats)
    vsc_update(new_vsc.VSCService, projs["a7_json"], stats)
    names = ["config.txt", "config_backup.txt", "tracerConfig.txt", "tracerConfig_backup.txt"]
    fh, fn = files_of(projs["a7_txt_head"], names), files_of(projs["a7_txt_new"], names)
    check("text project: HEAD and the new code write byte-identical config.txt, tracerConfig.txt and backups",
          fh == fn and all(fh[n] is not None for n in names))

    cfg = stb_config.read_json_raw(os.path.join(projs["a7_json"], "config.json"))
    exp_2d, exp_3d = round(stats["proj_tol"], 4), round(stats["triang_tol"] / v["voxel"], 4)
    ipr = cfg["objects"][0]["ipr"]
    check(f"config.json: tolerances {exp_2d} px and {exp_3d} vox", ipr["tol_2d_px"] == exp_2d and
          ipr["tol_3d_vox"] == exp_3d, f"{ipr['tol_2d_px']} {ipr['tol_3d_vox']}")
    check("config.json: the cameras are camFile_VSC/vsc_camX.txt",
          [c["file"] for c in cfg["cameras"]] == [f"camFile_VSC/vsc_cam{i}.txt" for i in range(4)],
          str([c["file"] for c in cfg["cameras"]]))
    backup = os.path.join(projs["a7_json"], "config_backup.json")
    check("config_backup.json holds the original config.json",
          os.path.exists(backup) and open(backup, "rb").read() == json_before)
    d_txt, d_json = cpp_dump([os.path.join(projs["a7_txt_new"], "config.txt"),
                              os.path.join(projs["a7_json"], "config.json")])
    same, diff = cpp_settings_equal(d_txt, d_json, [projs["a7_txt_new"], projs["a7_json"]])
    check(f"the C++ code reads the same settings from both updated projects (tol {d_json.get('sm', ['?'] * 4)[2]} px, "
          f"{d_json.get('sm', ['?'] * 4)[3]} mm)", same and "error" not in d_json, str(diff or d_json.get("error", "")))
    a = stb_config.read(os.path.join(projs["a7_txt_new"], "config.txt"))["cameras"]
    b = stb_config.read(os.path.join(projs["a7_json"], "config.json"))["cameras"]
    check("both updated projects name the same camera files", a == b, f"{a} | {b}")

    vsc_update(new_vsc.VSCService, projs["a7_json"], {"proj_tol": 1.111111, "triang_tol": 0.2})
    cfg2 = stb_config.read_json_raw(os.path.join(projs["a7_json"], "config.json"))
    check("a second VSC run updates the tolerances again and keeps the first backup",
          cfg2["objects"][0]["ipr"]["tol_2d_px"] == 1.1111 and open(backup, "rb").read() == json_before)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    say("A4b GUI config tests (A5 writer, A6 readers, A7 VSC writers), offscreen Qt")
    say(f"module: {lpt.__file__}")
    say(f"HEAD: {subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}")
    pairs = test_writer()
    test_readers(pairs)
    test_vsc_writers()
    n_fail = sum(not c["ok"] for c in CHECKS)
    say(f"\n{len(CHECKS) - n_fail} of {len(CHECKS)} checks passed" + ("" if not n_fail else f"; {n_fail} FAILED"))
    with open(os.path.join(RESULTS, "a4b_gui_config.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(LOG) + "\n")
    with open(os.path.join(RESULTS, "a4b_gui_config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"checks": CHECKS, "messages": MESSAGES}, f, indent=1)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
