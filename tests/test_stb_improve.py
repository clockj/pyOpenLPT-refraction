"""pytest subset of the stb_improve campaign (tests/stb_improve/): the config formats and the new settings.

The campaign's full tests (Matrix storage in C++, identical tracking outputs, benchmarks, epipolar lines) are in
tests/stb_improve/; these are the fast ones:
  - modules/stb_config.py: text <-> JSON conversion and round trips, the optional settings, the GUI's track-path
    lines, comments in JSON, bad values;
  - the C++ reader through the Python module (skipped on a build without the new settings): config.txt and its
    converted config.json give the same basic settings; save_interval and profile are read; bad files fail with
    messages that name the key.
"""
from __future__ import annotations

import json
import os
import textwrap

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
