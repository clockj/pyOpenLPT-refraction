"""pytest subset of the stb_improve campaign (tests/stb_improve/): the config formats and the new settings.

The campaign's full tests (Matrix storage in C++, identical tracking outputs, benchmarks, epipolar lines) are in
tests/stb_improve/; these are the fast ones:
  - modules/stb_config.py: text <-> JSON conversion and round trips, the optional settings, the GUI's track-path
    lines, comments in JSON, bad values;
  - the C++ reader through the Python module (skipped on a build without the new settings): config.txt and its
    converted config.json give the same basic settings; save_interval and profile are read; bad files fail with
    messages that name the key;
  - the GUI (offscreen Qt) and VSC on config.json (A4b): the settings form writes config.json only, with its save
    interval; the tracking view's readers accept both formats and report a missing or broken config; VSC writes
    its tolerances and optimised camera files into config.json and keeps the original as config_backup.json.
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import types

import pytest

from modules import stb_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = [os.path.join(REPO, "test", "inputs", "test_Camera", f"cam{i}.txt") for i in (1, 2, 3, 4)]

TRACER = textwrap.dedent("""\
    ############################
    ######### Tracking #########
    10.0 # Search radius for connecting tracks to objects
    4 # Number of frames for initial phase
    30.0 # Avg Interparticle spacing. (vox) to identify neighbour tracks
    0.25 # shake width 0.25
    51 # xgrid
    51 # ygrid
    51 # zgrid
    25.0 # searchRadius [voxel]
    2   # No. of IPR loop
    4   # No. of Shake loop
    0.01 # ghost threshold
    2.0   # 2D tolerance [px]
    1.0  # 3D tolerance [voxel]
    1 # number of reduced camera
    2 # no. of ipr loops for each reduced camera combination
    20 # 2D particle finder threshold
    2.0 # Particle radius [px], for calculating residue image and shaking
    """)


def write_case(folder, extra_lines="", load="0", gui_tail=False):
    """A text config (config.txt + tracerConfig.txt) in folder, with the repository's test cameras."""
    cams = "\n".join(f"{c.replace(os.sep, '/')},255" for c in CAMS)
    txt = textwrap.dedent("""\
        # Frame Range: [startID,endID]
        0,100
        # Frame Rate: [Hz]
        5000
        # Number of Threads: (0: use as many as possible)
        8
        # Number of Cameras:
        4
        # Camera File Path, Max Intensity
        {cams}
        # Image File Path
        imgFile/cam0ImageNames.txt
        imgFile/cam1ImageNames.txt
        imgFile/cam2ImageNames.txt
        imgFile/cam3ImageNames.txt
        # View Volume: (xmin,xmax,ymin,ymax,zmin,zmax)
        -81.0,54.0,-42.0,59.0,-36.0,52.5
        # Voxel to MM: e.g. use 1000^3 voxel, (xmax-xmin)/1000
        0.129
        # Output Folder Path:
        Results
        # Object Types:
        Tracer
        # STB Config Files:
        tracerConfig.txt
        # Flag to load previous track files, previous frameID
        {load}
        """).format(cams=cams, load=load)
    if gui_tail:
        txt += ("# Path to active long track files\nResults/ConvergeTrack/\n"
                "# Path to active short track files\nResults/ConvergeTrack/\n")
    txt += extra_lines
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "config.txt"), "w", newline="\n") as f:
        f.write(txt)
    with open(os.path.join(folder, "tracerConfig.txt"), "w", newline="\n") as f:
        f.write(TRACER)
    return os.path.join(folder, "config.txt")


# ---------------------------------------------------------------- the Python module

def test_text_and_converted_json_agree(tmp_path):
    txt = write_case(str(tmp_path), extra_lines="save_interval = 200\nprofile = 1\n")
    js = stb_config.convert(txt)
    a = stb_config._public(stb_config.read_text(txt))
    b = stb_config._public(stb_config.read_json(js))
    assert a == b
    assert a["save_interval"] == 200 and a["profile"] is True
    assert a["objects"][0]["ipr"]["tol_2d_px"] == 2.0
    assert a["objects"][0]["predictive_field"]["grid"] == [51, 51, 51]
    assert a["view_volume_mm"] == {"x": [-81.0, 54.0], "y": [-42.0, 59.0], "z": [-36.0, 52.5]}


def test_round_trips_are_exact(tmp_path):
    txt = write_case(str(tmp_path / "a"), extra_lines="save_interval = 100\n")
    cfg = stb_config.read_text(txt)
    out = str(tmp_path / "b")
    os.makedirs(out)
    stb_config.write_text(cfg, os.path.join(out, "config.txt"))
    assert stb_config._public(stb_config.read_text(os.path.join(out, "config.txt"))) == stb_config._public(cfg)
    stb_config.write_json(cfg, os.path.join(out, "config.json"))
    assert stb_config._public(stb_config.read_json(os.path.join(out, "config.json"))) == stb_config._public(cfg)


def test_gui_track_path_lines(tmp_path):
    # loading off: the GUI's two track-path lines are ignored, as by the C++ reader
    cfg = stb_config.read_text(write_case(str(tmp_path / "off"), gui_tail=True))
    assert cfg["load_tracks"] == {"enabled": False, "frame": 0}
    assert "save_interval" not in cfg and "_unknown_settings" not in cfg
    # loading on: the first path line is the folder
    cfg = stb_config.read_text(write_case(str(tmp_path / "on"), load="1,200", gui_tail=True))
    assert cfg["load_tracks"] == {"enabled": True, "frame": 200, "folder": "Results/ConvergeTrack/"}


@pytest.mark.parametrize("line", ["save_interval = 0", "save_interval = -5", "save_interval = abc",
                                  "save_interval = 20x", "profile = 2"])
def test_bad_optional_values(tmp_path, line):
    txt = write_case(str(tmp_path), extra_lines=line + "\n")
    with pytest.raises(stb_config.ConfigError):
        stb_config.read_text(txt)


def test_unknown_setting_is_recorded(tmp_path):
    cfg = stb_config.read_text(write_case(str(tmp_path), extra_lines="frobnicate = 3\n"))
    assert cfg["_unknown_settings"] == ["frobnicate"]


def test_json_comments_and_object_file(tmp_path):
    txt = write_case(str(tmp_path))
    d = stb_config.to_json_dict(stb_config.read_text(txt))
    d["objects"] = [{"type": "Tracer", "config_file": "tracerConfig.txt"}]   # the object settings in the text file
    text = "// a comment\n" + json.dumps(d, indent=1).replace('"fps"', '/* frame rate */ "fps"')
    p = os.path.join(str(tmp_path), "config.json")
    with open(p, "w") as f:
        f.write(text)
    cfg = stb_config.read_json(p)
    assert cfg["fps"] == 5000
    assert cfg["objects"][0]["_config_file"] == "tracerConfig.txt"
    assert cfg["objects"][0]["tracer"]["intensity_threshold"] == 20


def test_find_config_prefers_json(tmp_path):
    assert stb_config.find_config(str(tmp_path)) is None
    txt = write_case(str(tmp_path))
    assert stb_config.find_config(str(tmp_path)) == txt
    js = stb_config.convert(txt)
    assert stb_config.find_config(str(tmp_path)) == js


def test_read_project_and_raw_edit(tmp_path):
    assert stb_config.read_project(str(tmp_path)) == (None, None)
    txt = write_case(str(tmp_path))
    os.remove(os.path.join(str(tmp_path), "tracerConfig.txt"))
    # without inline_objects the object file is not needed
    path, cfg = stb_config.read_project(str(tmp_path), inline_objects=False)
    assert path == txt and cfg["objects"] == [{"type": "Tracer", "config_file": "tracerConfig.txt"}]
    with pytest.raises(OSError):
        stb_config.read_project(str(tmp_path))
    # read_json_raw / write_json_raw keep every key, also the comment keys that write_json drops
    p = os.path.join(str(tmp_path), "config.json")
    with open(p, "w") as f:
        f.write('{"_note": "keep me", "fps": 5000} // trailing comment\n')
    d = stb_config.read_json_raw(p)
    d["fps"] = 100
    stb_config.write_json_raw(d, p)
    assert stb_config.read_json_raw(p) == {"_note": "keep me", "fps": 100}
    with open(p, "w") as f:
        f.write('{"fps": 5000,, }')
    with pytest.raises(stb_config.ConfigError, match="invalid JSON"):
        stb_config.read_json_raw(p)


# ---------------------------------------------------------------- the GUI and VSC (A4b), offscreen

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # an earlier test may have left MagicMock stand-ins for PySide6 in sys.modules (test_precalibrate_3d_view.py
    # does); drop them so that the real bindings load
    for name in [n for n, m in sys.modules.items()
                 if (n == "PySide6" or n.startswith("PySide6.")) and not isinstance(m, types.ModuleType)]:
        del sys.modules[name]
    widgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("qtawesome")
    pytest.importorskip("cv2")
    gui = os.path.join(REPO, "gui")
    if gui not in sys.path:
        sys.path.insert(0, gui)
    return widgets.QApplication.instance() or widgets.QApplication([])


def gui_project(folder):
    """A project folder as the GUI expects it: camFile/cam0-3.txt and imgFile/cam0-3ImageNames.txt."""
    import shutil
    os.makedirs(os.path.join(folder, "camFile"))
    os.makedirs(os.path.join(folder, "imgFile"))
    for i, c in enumerate(CAMS):
        shutil.copy(c, os.path.join(folder, "camFile", f"cam{i}.txt"))
        with open(os.path.join(folder, "imgFile", f"cam{i}ImageNames.txt"), "w") as f:
            f.write(f"img/cam{i}/img000000.tif\n")
    return folder


def test_gui_writes_config_json(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from views.tracking_settings_view import TrackingSettingsView
    boxes = []
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, kind=kind, **k: boxes.append((kind, a[2] if len(a) > 2 else "")))
    proj = gui_project(str(tmp_path / "proj"))
    v = TrackingSettingsView()
    v.project_path.setText(proj)
    v.image_path_display.setText(os.path.join(proj, "imgFile"))
    v.camera_path_display.setText(os.path.join(proj, "camFile"))
    v.output_path.setText(os.path.join(proj, "Results"))
    v.n_cam_spin.setValue(4)
    v.save_interval_spin.setValue(200)
    v.tracer_radius.setValue(2.5)
    v._save_configuration()
    assert [k for k, _ in boxes] == ["information"]
    assert sorted(os.listdir(proj)) == ["camFile", "config.json", "imgFile"]
    cfg = stb_config.read(os.path.join(proj, "config.json"))
    assert cfg["save_interval"] == 200 and cfg["output_dir"] == "Results"
    assert [c["file"] for c in cfg["cameras"]] == [f"camFile/cam{i}.txt" for i in range(4)]
    assert cfg["image_lists"] == [f"imgFile/cam{i}ImageNames.txt" for i in range(4)]
    assert cfg["objects"][0]["type"] == "Tracer" and cfg["objects"][0]["tracer"]["radius_px"] == 2.5
    try:
        import pyopenlpt
    except ImportError:
        return
    if hasattr(pyopenlpt.BasicSetting(), "_save_interval"):
        bs = pyopenlpt.BasicSetting()
        assert bs.readConfig(os.path.join(proj, "config.json"))
        assert bs._save_interval == 200 and bs._object_config_paths[0].endswith("config.json#0")


def test_gui_readers_accept_both_formats(qapp, tmp_path):
    from views import tracking_view
    p_txt = gui_project(str(tmp_path / "txt"))
    write_case(p_txt)
    p_json = gui_project(str(tmp_path / "json"))
    stb_config.convert(write_case(p_json))
    os.remove(os.path.join(p_json, "config.txt"))
    os.remove(os.path.join(p_json, "tracerConfig.txt"))
    view = tracking_view.TrackingView(settings_view=None)
    for proj, name in ((p_txt, "config.txt"), (p_json, "config.json")):
        path, cfg = tracking_view._project_config(proj)
        assert os.path.basename(path) == name
        assert tracking_view._output_dir(proj, path, cfg) == os.path.join(proj, "Results")
        assert tracking_view._object_type(cfg) == "Tracer"
        assert view._check_project_files(proj) == (True, "")
    # no config, and a broken one
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    assert tracking_view._project_config(empty) == (None, None)
    assert tracking_view._output_dir(empty, None, None) == os.path.join(empty, "Results")
    ok, msg = view._check_project_files(empty)
    assert not ok and "missing" in msg
    with open(os.path.join(p_json, "config.json"), "w") as f:
        f.write("{")
    ok, msg = view._check_project_files(p_json)
    assert not ok and "config.json cannot be read" in msg


def test_vsc_updates_config_json(tmp_path):
    pytest.importorskip("scipy")
    from modules.vsc.vsc_service import VSCService
    proj = gui_project(str(tmp_path / "proj"))
    stb_config.convert(write_case(proj))
    os.remove(os.path.join(proj, "config.txt"))
    os.remove(os.path.join(proj, "tracerConfig.txt"))
    before = open(os.path.join(proj, "config.json"), "rb").read()
    svc = VSCService(proj, log_callback=lambda m: None)
    svc.cameras = {0: {}, 1: {}, 2: {}, 3: {}}
    svc._update_object_config({"proj_tol": 0.734567, "triang_tol": 0.045678})
    svc._update_config()
    cfg = stb_config.read_json_raw(os.path.join(proj, "config.json"))
    assert cfg["objects"][0]["ipr"]["tol_2d_px"] == 0.7346
    assert cfg["objects"][0]["ipr"]["tol_3d_vox"] == round(0.045678 / 0.129, 4)
    # the test cameras are cam1-4.txt: cam1-3 match VSC's cameras 1-3, cam4 is not one of them
    assert [c["file"].split("/")[-1] for c in cfg["cameras"]] == ["vsc_cam1.txt", "vsc_cam2.txt", "vsc_cam3.txt",
                                                                    "cam4.txt"]
    assert open(os.path.join(proj, "config_backup.json"), "rb").read() == before
    svc._update_object_config({"proj_tol": 1.0, "triang_tol": 0.1})
    assert open(os.path.join(proj, "config_backup.json"), "rb").read() == before


# ---------------------------------------------------------------- the C++ reader

@pytest.fixture(scope="module")
def lpt():
    lpt = pytest.importorskip("pyopenlpt")
    if not hasattr(lpt.BasicSetting(), "_save_interval"):
        pytest.skip("this pyopenlpt build predates the JSON config and save_interval")
    return lpt


BASIC_FIELDS = ["_n_cam", "_frame_start", "_frame_end", "_fps", "_n_thread", "_voxel_to_mm", "_image_file_paths",
                "_object_types", "_load_track", "_load_track_frame", "_save_interval", "_profile"]


def basic(lpt, path):
    bs = lpt.BasicSetting()
    assert bs.readConfig(path)
    d = {k: getattr(bs, k) for k in BASIC_FIELDS}
    d["limit"] = [getattr(bs._axis_limit, a) for a in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")]
    d["output"] = bs._output_path
    return d, bs


def test_cpp_text_and_json_agree(lpt, tmp_path):
    txt = write_case(str(tmp_path), extra_lines="save_interval = 200\nprofile = 1\n")
    js = stb_config.convert(txt)
    a, bs_t = basic(lpt, txt)
    b, bs_j = basic(lpt, js)
    assert a == b
    assert a["_save_interval"] == 200 and a["_profile"] is True
    assert bs_t._object_config_paths[0].endswith("tracerConfig.txt")
    assert bs_j._object_config_paths[0].endswith("config.json#0")


def test_cpp_defaults_and_gui_lines(lpt, tmp_path):
    a, _ = basic(lpt, write_case(str(tmp_path), gui_tail=True))
    assert a["_save_interval"] == 500 and a["_profile"] is False and a["_load_track"] is False


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d.pop("voxel_mm"), '"voxel_mm" is missing'),
    (lambda d: d.update(fps="5000"), '"fps" must be an integer'),
    (lambda d: d.update(save_interval=0), '"save_interval"'),
    (lambda d: d.update(frames=[10, 5]), '"frames"'),
    (lambda d: d.update(image_lists=d["image_lists"][:3]), '"image_lists" must be an array of 4'),
])
def test_cpp_bad_json(lpt, tmp_path, mutate, needle):
    d = stb_config.to_json_dict(stb_config.read_text(write_case(str(tmp_path))))
    mutate(d)
    p = os.path.join(str(tmp_path), "bad.json")
    with open(p, "w") as f:
        json.dump(d, f)
    with pytest.raises(Exception) as e:
        lpt.BasicSetting().readConfig(p)
    assert needle in str(e.value)


def test_cpp_bad_text_setting(lpt, tmp_path):
    with pytest.raises(Exception) as e:
        lpt.BasicSetting().readConfig(write_case(str(tmp_path), extra_lines="save_interval = 0\n"))
    assert "save_interval must be a positive integer" in str(e.value)
