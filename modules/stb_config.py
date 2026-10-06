"""Read, write and convert OpenLPT tracking (STB) configuration files.

Two formats hold the same settings:
  - text: config.txt (the basic settings, one value per line in a fixed order, '#' comments) plus one file per object
    type (tracerConfig.txt or bubbleConfig.txt), as the GUI writes them;
  - JSON: one config.json with the basic settings and every object's settings inline.  The C++ code reads both
    (src/srcSTB/Config.cpp picks the format by the file extension).

read(path) returns the same dict for either format, in the JSON layout below, with paths as written in the file
(relative paths are relative to the config file's folder; resolve() makes them absolute).  write_json() and
write_text() write such a dict, and convert() turns a text config into config.json beside it.  A project folder's
config is config.json if it has one, else config.txt (find_config, read_project); the GUI writes config.json.
read_json_raw() / write_json_raw() edit a config.json in place, keeping every key it holds.

    python -m modules.stb_config convert config.txt [-o config.json]
    python -m modules.stb_config show CONFIG

The JSON layout (lengths in voxels where the key ends in _vox, in mm where it ends in _mm, pixels for _px):

    {
      "frames": [0, 500], "fps": 5000, "threads": 26,
      "cameras": [{"file": "camFile/cam0.txt", "max_intensity": 255}, ...],
      "image_lists": ["imgFile/cam0ImageNames.txt", ...],
      "view_volume_mm": {"x": [-81.0, 54.0], "y": [-42.0, 59.0], "z": [-36.0, 52.5]},
      "voxel_mm": 0.129,
      "output_dir": "Results",
      "save_interval": 500,
      "load_tracks": {"enabled": false, "frame": 0, "folder": "Results/ConvergeTrack/"},
      "objects": [{
        "type": "Tracer",
        "initial_phase": {"search_radius_vox": 10.0, "frames": 4},
        "convergence_phase": {"avg_spacing_vox": 30.0},
        "shake": {"width_vox": 0.25, "loops": 4, "ghost_threshold": 0.01},
        "predictive_field": {"grid": [51, 51, 51], "search_radius_vox": 25.0},
        "ipr": {"loops": 2, "reduced_cameras": 1, "reduced_loops": 2, "tol_2d_px": 2.0, "tol_3d_vox": 1.0},
        "tracer": {"intensity_threshold": 20, "radius_px": 2.0}
      }]
    }

A Bubble object has "bubble": {"radius_min_px": .., "radius_max_px": .., "sense": ..} instead of "tracer".  An object
may instead name a text object file: {"type": "Tracer", "config_file": "tracerConfig.txt"}.  "threads",
"save_interval", "profile" (true: print stereo-matching counters per IPR round) and "load_tracks" are optional; keys starting with "_" are comments.  The text reader mirrors the C++
one (std::stoi/std::stod read the leading number of a line), so both give the same values.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

_INT = re.compile(r"^\s*([+-]?\d+)")
_FLOAT = re.compile(r"^\s*([+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|inf(?:inity)?|nan))", re.I)
_KEY_VALUE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S(?:.*\S)?)$")


class ConfigError(ValueError):
    """A configuration file that cannot be read."""


def _stoi(text: str, what: str) -> int:
    """The leading integer of text, as std::stoi reads it."""
    m = _INT.match(text)
    if not m:
        raise ConfigError(f"{what}: expected an integer, got {text!r}")
    return int(m.group(1))


def _stod(text: str, what: str) -> float:
    """The leading number of text, as std::stod reads it."""
    m = _FLOAT.match(text)
    if not m:
        raise ConfigError(f"{what}: expected a number, got {text!r}")
    return float(m.group(1))


def _clean_lines(path: str) -> List[str]:
    """The setting lines of a text config, as the C++ reader sees them: trimmed, without empty lines, comment lines
    and trailing '#' comments."""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read().splitlines()
    out = []
    for line in raw:
        line = line.strip(" \t\r\n")
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line[: line.index("#")].strip(" \t\r\n")
            if not line:
                continue
        out.append(line)
    return out


def _split(line: str) -> List[str]:
    """Comma-separated fields, as std::getline(parser, field, ',') gives them (no trailing empty field)."""
    parts = line.split(",")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return parts


def read_text_object(path: str, obj_type: str) -> Dict:
    """One object's settings from a text object file (tracerConfig.txt / bubbleConfig.txt)."""
    lines = _clean_lines(path)
    n_need = 17 if obj_type == "Tracer" else 18
    if len(lines) < n_need:
        raise ConfigError(f"{path}: {len(lines)} setting lines, a {obj_type} config needs {n_need}")
    it = iter(lines)
    obj: Dict = {"type": obj_type}
    obj["initial_phase"] = {"search_radius_vox": _stod(next(it), "initial search radius"),
                            "frames": _stoi(next(it), "initial frames")}
    obj["convergence_phase"] = {"avg_spacing_vox": _stod(next(it), "avg interparticle spacing")}
    shake_width = _stod(next(it), "shake width")
    grid = [_stoi(next(it), "predictive field grid") for _ in range(3)]
    pf_radius = _stod(next(it), "predictive field radius")
    ipr_loops = _stoi(next(it), "IPR loops")
    shake_loops = _stoi(next(it), "shake loops")
    ghost = _stod(next(it), "ghost threshold")
    tol2d = _stod(next(it), "2D tolerance")
    tol3d = _stod(next(it), "3D tolerance")
    reduced_cams = _stoi(next(it), "reduced cameras")
    reduced_loops = _stoi(next(it), "reduced IPR loops")
    obj["shake"] = {"width_vox": shake_width, "loops": shake_loops, "ghost_threshold": ghost}
    obj["predictive_field"] = {"grid": grid, "search_radius_vox": pf_radius}
    obj["ipr"] = {"loops": ipr_loops, "reduced_cameras": reduced_cams, "reduced_loops": reduced_loops,
                  "tol_2d_px": tol2d, "tol_3d_vox": tol3d}
    if obj_type == "Tracer":
        obj["tracer"] = {"intensity_threshold": _stoi(next(it), "particle intensity threshold"),
                         "radius_px": _stod(next(it), "particle radius")}
    elif obj_type == "Bubble":
        obj["bubble"] = {"radius_min_px": _stoi(next(it), "bubble radius min"),
                         "radius_max_px": _stoi(next(it), "bubble radius max"),
                         "sense": _stod(next(it), "bubble sense")}
    else:
        raise ConfigError(f"{path}: unknown object type {obj_type!r}")
    return obj


def read_text(path: str, inline_objects: bool = True) -> Dict:
    """A text config.txt.  With inline_objects the object files are read into "objects" (their file names kept
    under "_config_file"); otherwise each object is {"type", "config_file"}."""
    lines = _clean_lines(path)
    if len(lines) < 11:
        raise ConfigError(f"{path}: too few lines")
    i = 0
    cfg: Dict = {}
    fr = _split(lines[i]); i += 1
    if len(fr) < 2:
        raise ConfigError(f"{path}: frame range must be start,end")
    cfg["frames"] = [_stoi(fr[0], "frame start"), _stoi(fr[1], "frame end")]
    cfg["fps"] = _stoi(lines[i], "frame rate"); i += 1
    cfg["threads"] = _stoi(lines[i], "threads"); i += 1
    n_cam = _stoi(lines[i], "number of cameras"); i += 1
    cams = []
    for _ in range(n_cam):
        parts = _split(lines[i]); i += 1
        if len(parts) < 2:
            raise ConfigError(f"{path}: camera line must be path,max_intensity")
        cams.append({"file": parts[0], "max_intensity": _stoi(parts[1], "max intensity")})
    cfg["cameras"] = cams
    cfg["image_lists"] = [lines[i + k] for k in range(n_cam)]
    i += n_cam
    vv = _split(lines[i]); i += 1
    if len(vv) < 6:
        raise ConfigError(f"{path}: view volume needs 6 numbers")
    v = [_stod(x, "view volume") for x in vv[:6]]
    cfg["view_volume_mm"] = {"x": v[0:2], "y": v[2:4], "z": v[4:6]}
    cfg["voxel_mm"] = _stod(lines[i], "voxel to mm"); i += 1
    cfg["output_dir"] = lines[i]; i += 1
    types = [t for t in _split(lines[i]) if t]; i += 1
    files = [lines[i + k] for k in range(len(types))]
    i += len(types)
    load = _split(lines[i]); i += 1
    enabled = _stoi(load[0], "load flag") != 0
    lt: Dict = {"enabled": enabled, "frame": cfg["frames"][0]}
    if enabled:
        if len(load) > 1 and load[1] != "":
            lt["frame"] = _stoi(load[1], "load frame")
        if i < len(lines):
            lt["folder"] = lines[i]; i += 1
    cfg["load_tracks"] = lt
    for line in lines[i:]:  # optional "key = value" settings; other lines are ignored, as in C++
        m = _KEY_VALUE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        if key == "save_interval":
            n = _stoi(value, "save_interval")
            if str(n) != value.strip() or n <= 0:
                raise ConfigError(f"{path}: save_interval must be a positive integer, got {value!r}")
            cfg["save_interval"] = n
        elif key == "profile":
            if value not in ("0", "1"):
                raise ConfigError(f"{path}: profile must be 0 or 1, got {value!r}")
            cfg["profile"] = value == "1"
        else:
            cfg.setdefault("_unknown_settings", []).append(key)
    root = os.path.dirname(os.path.abspath(path))
    objs = []
    for t, fname in zip(types, files):
        if inline_objects:
            obj = read_text_object(fname if os.path.isabs(fname) else os.path.join(root, fname), t)
            obj["_config_file"] = fname
        else:
            obj = {"type": t, "config_file": fname}
        objs.append(obj)
    cfg["objects"] = objs
    return cfg


_TOP_KEYS = ["frames", "fps", "threads", "cameras", "image_lists", "view_volume_mm", "voxel_mm", "output_dir",
             "objects", "load_tracks", "save_interval", "profile"]


def read_json_raw(path: str) -> Dict:
    """config.json exactly as written (comments removed), for editing in place: write_json_raw() writes it back with
    every key it holds."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        cfg = json.loads(_strip_json_comments(text))
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON: {e}") from None
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: must hold an object {{...}}")
    return cfg


def write_json_raw(cfg: Dict, path: str) -> None:
    """A dict from read_json_raw(), written back (comments in the original file are not kept)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def read_json(path: str, inline_objects: bool = True) -> Dict:
    """config.json, with objects that name a text file ("config_file") read into place (the name kept under
    "_config_file"); with inline_objects=False such objects stay {"type", "config_file"}."""
    cfg = read_json_raw(path)
    for key in ("frames", "fps", "cameras", "image_lists", "view_volume_mm", "voxel_mm", "output_dir", "objects"):
        if key not in cfg:
            raise ConfigError(f"{path}: \"{key}\" is missing")
    if not isinstance(cfg["objects"], list) or not all(isinstance(o, dict) and "type" in o for o in cfg["objects"]):
        raise ConfigError(f"{path}: \"objects\" must be a list of objects with a \"type\"")
    if not isinstance(cfg["cameras"], list) or not all(isinstance(c, dict) and "file" in c for c in cfg["cameras"]):
        raise ConfigError(f"{path}: \"cameras\" must be a list of objects with a \"file\"")
    root = os.path.dirname(os.path.abspath(path))
    objs = []
    for k, obj in enumerate(cfg["objects"]):
        if "config_file" in obj and inline_objects:
            fname = obj["config_file"]
            full = read_text_object(fname if os.path.isabs(fname) else os.path.join(root, fname), obj["type"])
            full["_config_file"] = fname
            objs.append(full)
        else:
            objs.append(obj)
    cfg["objects"] = objs
    cfg.setdefault("threads", 0)
    lt = cfg.setdefault("load_tracks", {"enabled": False})
    lt.setdefault("frame", cfg["frames"][0])
    return cfg


def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments outside strings (the C++ reader allows them)."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise ConfigError("unterminated /* comment")
            i = j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def read(path: str, inline_objects: bool = True) -> Dict:
    """Either format, by the file extension (.json or text).  With inline_objects=False the object files are not
    read: an object that lives in a text file stays {"type", "config_file"}."""
    if path.lower().endswith(".json"):
        return read_json(path, inline_objects)
    return read_text(path, inline_objects)


def find_config(project_dir: str) -> Optional[str]:
    """A project's config: config.json if present, else config.txt; None if neither exists."""
    for name in ("config.json", "config.txt"):
        p = os.path.join(project_dir, name)
        if os.path.isfile(p):
            return p
    return None


def read_project(project_dir: str, inline_objects: bool = True):
    """(path, settings) of a project's config (find_config), or (None, None) if the folder has none.  A config that
    cannot be read raises ConfigError or OSError."""
    path = find_config(project_dir)
    if path is None:
        return None, None
    return path, read(path, inline_objects)


def resolve(config_path: str, path: str) -> str:
    """A path from a config, made absolute against the config file's folder."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(config_path)), path))


def _public(d):
    """A copy without the "_..." bookkeeping keys."""
    if isinstance(d, dict):
        return {k: _public(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [_public(v) for v in d]
    return d


def to_json_dict(cfg: Dict) -> Dict:
    """The dict as config.json holds it: keys in the documented order, objects inline."""
    c = _public(cfg)
    out = {k: c[k] for k in _TOP_KEYS if k in c}
    lt = out.get("load_tracks")
    if lt is not None and not lt.get("enabled") and "folder" not in lt:
        out["load_tracks"] = {"enabled": False}
    return out


def write_json(cfg: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(to_json_dict(cfg), f, indent=2)
        f.write("\n")


def _num(x) -> str:
    """A number as the GUI writes it (repr keeps every digit, so a round trip is exact)."""
    return repr(float(x)) if isinstance(x, float) else str(x)


def write_text(cfg: Dict, path: str, object_files: Optional[List[str]] = None) -> List[str]:
    """config.txt at path and one object file per object (default names tracerConfig.txt / bubbleConfig.txt beside
    it, or object_files, relative to path's folder).  Returns the object file names written."""
    c = _public(cfg)
    root = os.path.dirname(os.path.abspath(path))
    names = object_files or [f"{o['type'].lower()}Config.txt" for o in c["objects"]]
    vv = c["view_volume_mm"]
    lt = c.get("load_tracks", {"enabled": False})
    lines = ["# Frame Range: [startID,endID]", f"{c['frames'][0]},{c['frames'][1]}",
             "# Frame Rate: [Hz]", str(c["fps"]),
             "# Number of Threads: (0: use as many as possible)", str(c.get("threads", 0)),
             "# Number of Cameras:", str(len(c["cameras"])),
             "# Camera File Path, Max Intensity"]
    lines += [f"{cam['file']},{cam.get('max_intensity', 255)}" for cam in c["cameras"]]
    lines += ["# Image File Path"] + list(c["image_lists"])
    lines += ["# View Volume: (xmin,xmax,ymin,ymax,zmin,zmax)",
              ",".join(_num(x) for x in (*vv["x"], *vv["y"], *vv["z"])),
              "# Voxel to MM: e.g. use 1000^3 voxel, (xmax-xmin)/1000", _num(c["voxel_mm"]),
              "# Output Folder Path:", c["output_dir"],
              "# Object Types:", ",".join(o["type"] for o in c["objects"]),
              "# STB Config Files:"] + names
    lines += ["# Flag to load previous track files, previous frameID",
              f"{1 if lt.get('enabled') else 0},{lt.get('frame', c['frames'][0])}"]
    if lt.get("enabled"):
        lines += ["# Path to active track files", lt["folder"]]
    optional = []
    if "save_interval" in c:
        optional.append(f"save_interval = {c['save_interval']}")
    if c.get("profile"):
        optional.append("profile = 1")
    if optional:
        lines += ["# Optional settings"] + optional
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    for o, name in zip(c["objects"], names):
        _write_text_object(o, name if os.path.isabs(name) else os.path.join(root, name))
    return names


def _write_text_object(o: Dict, path: str) -> None:
    ip, cp, sh, pf, ipr = o["initial_phase"], o["convergence_phase"], o["shake"], o["predictive_field"], o["ipr"]
    lines = ["######### Initial Phase #########",
             f"{_num(ip['search_radius_vox'])} # Search radius for connecting tracks to objects",
             f"{ip['frames']} # Number of frames for initial phase",
             "######### Convergence Phase #########",
             f"{_num(cp['avg_spacing_vox'])} # Avg Interparticle spacing. (vox) to identify neighbour tracks",
             "######### Shake #########",
             f"{_num(sh['width_vox'])} # shake width",
             "######### Predict Field #########",
             f"{pf['grid'][0]} # xgrid", f"{pf['grid'][1]} # ygrid", f"{pf['grid'][2]} # zgrid",
             f"{_num(pf['search_radius_vox'])} # searchRadius [voxel]",
             "######### IPR #########",
             f"{ipr['loops']} # No. of IPR loop", f"{sh['loops']} # No. of Shake loop",
             f"{_num(sh['ghost_threshold'])} # ghost threshold",
             f"{_num(ipr['tol_2d_px'])} # 2D tolerance [px]", f"{_num(ipr['tol_3d_vox'])} # 3D tolerance [voxel]",
             f"{ipr['reduced_cameras']} # number of reduced camera",
             f"{ipr['reduced_loops']} # no. of ipr loops for each reduced camera combination",
             "######### Object Info #########"]
    if o["type"] == "Tracer":
        t = o["tracer"]
        lines += [f"{t['intensity_threshold']} # 2D particle finder threshold",
                  f"{_num(t['radius_px'])} # Particle radius [px], for calculating residue image and shaking"]
    else:
        b = o["bubble"]
        lines += [f"{b['radius_min_px']} # minimum bubble radius [px]", f"{b['radius_max_px']} # maximum bubble radius [px]",
                  f"{_num(b['sense'])} # sensitivity for bubble detection"]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def convert(text_path: str, json_path: Optional[str] = None) -> str:
    """Write config.json (default: beside text_path) with the settings of a text config and its object files."""
    cfg = read_text(text_path)
    out = json_path or os.path.join(os.path.dirname(os.path.abspath(text_path)), "config.json")
    if os.path.dirname(os.path.abspath(out)) != os.path.dirname(os.path.abspath(text_path)):
        raise ConfigError("config.json must go in the same folder as config.txt, so its relative paths stay valid")
    write_json(cfg, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m modules.stb_config", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="write config.json from config.txt and its object files")
    c.add_argument("config_txt")
    c.add_argument("-o", "--output", default=None)
    s = sub.add_parser("show", help="print the settings of either format as JSON")
    s.add_argument("config")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "convert":
            print(convert(args.config_txt, args.output))
        else:
            print(json.dumps(to_json_dict(read(args.config)), indent=2))
    except (ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
