"""Test C1 (plan A4): the JSON config gives the same settings as the text config, and bad files give clear errors.

    python c1_config.py [--build dev]

1. Settings: this case's config.txt + tracerConfig.txt (the benchmark setup on the local data, frames 0-20,
   save_interval 10) and the config.json that modules/stb_config.py converts them to are both read by the C++ code
   (BasicSetting.readConfig, TracerConfig.readConfig, through the build's Python module); every bound setting is
   dumped and compared exactly.
2. The Python module: read_text and read_json of the two files give the same dict; write_text -> read_text and
   write_json -> read_json round trips are exact.
3. Bad files, each read by the C++ code in a fresh process (its warnings go to stderr) and by the Python module:
   missing and mistyped keys, an unknown key, a JSON syntax error, bad save_interval values, a GUI-written
   config.txt with its track-path lines.
Writes results/c1_config.txt and .json; exit code 1 if anything fails.  (The end-to-end check, a run from each file
with identical outputs, is done with stbrun.py --json and compare_runs.py identical.)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(HERE, "runs")
RESULTS = os.path.join(HERE, "results")
PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from modules import stb_config  # noqa: E402
import stbrun  # noqa: E402

LOG = []


def say(msg=""):
    print(msg)
    LOG.append(msg)


def dump(lpt_build, cfg_path):
    """Every bound setting the C++ code reads from a config, in a fresh process (so OTF estimation and warnings are
    those of a normal start)."""
    code = f"""
import sys, json
sys.path.insert(0, {lpt_build!r})
import pyopenlpt as lpt
bs = lpt.BasicSetting()
ok = bs.readConfig({cfg_path!r})
d = {{"ok": ok}}
for k in ["_n_cam", "_frame_start", "_frame_end", "_fps", "_n_thread", "_voxel_to_mm", "_output_path",
          "_image_file_paths", "_object_types", "_object_config_paths", "_load_track", "_load_track_frame",
          "_load_track_path", "_save_interval"]:
    d[k] = getattr(bs, k)
d["_axis_limit"] = [getattr(bs._axis_limit, a) for a in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]]
d["cam_max_intensity"] = [c.max_intensity for c in bs._cam_list] if hasattr(bs._cam_list[0], "max_intensity") else []
tc = lpt.TracerConfig()
tc.readConfig(bs._object_config_paths[0], bs)
s, sh, pf, ipr, sm = tc._stb_param, tc._shake_param, tc._pf_param, tc._ipr_param, tc._sm_param
d["stb"] = [s._radius_search_obj, s._n_initial_frames, s._radius_search_track]
d["shake"] = [sh._shake_width, sh._n_shake_loop, sh._thred_ghost, sh._shakewidth_min, sh._ratio_augimg]
d["pf"] = [pf.nx, pf.ny, pf.nz, pf.r, pf.nBin_x, pf.nBin_y, pf.nBin_z, pf.is_smooth, pf.sigma_x, pf.sigma_y,
           pf.sigma_z, [getattr(pf.limit, a) for a in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]]]
d["ipr"] = [ipr.n_cam_reduced, ipr.n_loop_ipr, ipr.n_loop_ipr_reduced, ipr.n_obj2d_process_max]
d["sm"] = [sm.match_cam_count, sm.idmap_cell_px, sm.tol_2d_px, sm.tol_3d_mm]
d["tracer"] = [tc._min_obj_int, tc._radius_obj]
o = tc._otf._param
d["otf_grid"] = [o.n_cam, o.nx, o.ny, o.nz, o.n_grid, o.dx, o.dy, o.dz]
print("@@" + json.dumps(d))
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    out = [ln for ln in r.stdout.splitlines() if ln.startswith("@@")]
    return (json.loads(out[-1][2:]) if out else None), r.stdout, r.stderr, r.returncode


def try_read_cpp(lpt_build, cfg_path):
    """Read a config with the C++ code in a fresh process: (ok, error message, stderr)."""
    code = f"""
import sys
sys.path.insert(0, {lpt_build!r})
import pyopenlpt as lpt
bs = lpt.BasicSetting()
try:
    ok = bs.readConfig({cfg_path!r})
    print("@@OK" if ok else "@@FALSE")
except Exception as e:
    print("@@ERR " + str(e).replace(chr(10), " | "))
"""
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    res = [ln for ln in r.stdout.splitlines() if ln.startswith("@@")]
    return (res[-1] if res else "@@CRASH"), r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="dev")
    args = ap.parse_args()
    bdir = stbrun.build_dir(args.build)
    failures = []

    # ---- the two configs ----
    work = os.path.join(RUNS, "c1_config")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    ns = argparse.Namespace(frames=[0, 20], threads=26, load=None, load_from=None,
                            config_extra=["save_interval = 10"], tracer_extra=[])
    stbrun.write_inputs(work, ns)
    txt = os.path.join(work, "config.txt")
    js = stb_config.convert(txt)
    say(f"C1 config test, build {args.build} ({bdir})")
    say(f"text: {txt}\njson: {js}")
    with open(js) as f:
        say("config.json:\n" + f.read())

    # ---- 1. C++ settings ----
    dt, out_t, err_t, rc_t = dump(bdir, txt)
    dj, out_j, err_j, rc_j = dump(bdir, js)
    if dt is None or dj is None:
        failures.append("C++ read failed")
        say(f"C++ read failed: text rc {rc_t} {err_t[-500:]}, json rc {rc_j} {err_j[-500:]}")
    else:
        diff = {k: (dt[k], dj[k]) for k in dt if dt[k] != dj[k]}
        expected = {"_object_config_paths"}  # the object settings' address differs by design
        say(f"1. C++ settings: {len(dt)} groups compared; differing: {sorted(diff)}")
        say(f"   object config address: text {dt['_object_config_paths']}, json {dj['_object_config_paths']}")
        for k in sorted(set(diff) - expected):
            failures.append(f"setting {k} differs: {diff[k]}")
            say(f"   FAIL {k}: text {diff[k][0]} json {diff[k][1]}")
        say(f"   save_interval read: text {dt['_save_interval']}, json {dj['_save_interval']}")
        if err_j.strip():
            say(f"   stderr of the JSON read (expect none):\n{err_j.strip()}")
            if "Warning" in err_j:
                failures.append("warnings when reading the converted JSON")

    # ---- 2. Python module ----
    pt = stb_config._public(stb_config.read_text(txt))
    pj = stb_config._public(stb_config.read_json(js))
    same = pt == pj
    say(f"2. Python read_text == read_json: {same}")
    if not same:
        failures.append("python text and json dicts differ")
    rt_dir = os.path.join(work, "roundtrip")
    os.makedirs(rt_dir)
    stb_config.write_text(stb_config.read_text(txt), os.path.join(rt_dir, "config.txt"))
    rt = stb_config._public(stb_config.read_text(os.path.join(rt_dir, "config.txt")))
    stb_config.write_json(stb_config.read_json(js), os.path.join(rt_dir, "config.json"))
    rj = stb_config._public(stb_config.read_json(os.path.join(rt_dir, "config.json")))
    say(f"   round trips exact: text {rt == pt}, json {rj == pj}")
    if rt != pt or rj != pj:
        failures.append("round trip differs")

    # ---- 3. bad files ----
    base = json.load(open(js))

    def variant(name, mutate, text=None):
        p = os.path.join(work, name)
        if text is not None:
            with open(p, "w", newline="\n") as f:
                f.write(text)
        else:
            d = copy.deepcopy(base)
            mutate(d)
            with open(p, "w", newline="\n") as f:
                json.dump(d, f, indent=1)
        return p

    def drop(path):
        def m(d):
            node = d
            for k in path[:-1]:
                node = node[k]
            del node[path[-1]]
        return m

    def setv(path, value):
        def m(d):
            node = d
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
        return m

    with open(txt) as f:
        txt_lines = f.read()
    gui_style = txt_lines.replace("save_interval = 10\n", "") + \
        "# Path to active long track files\nResults/ConvergeTrack/\n# Path to active short track files\n" \
        "Results/ConvergeTrack/\n"
    cases = [
        # name, file, expect ("ok" / "error" / "warning"), text that must appear
        ("missing voxel_mm", variant("bad_missing.json", drop(["voxel_mm"])), "error", '"voxel_mm" is missing'),
        ("fps as a string", variant("bad_type.json", setv(["fps"], "5000")), "error", '"fps" must be an integer'),
        ("missing ipr.tol_2d_px (read with the object)", variant("bad_obj.json", drop(["objects", 0, "ipr", "tol_2d_px"])),
         "object_error", '"objects[0].ipr.tol_2d_px" is missing'),
        ("unknown key", variant("warn_unknown.json", setv(["voxel_size"], 0.1)), "warning", 'unknown setting "voxel_size"'),
        ("syntax error", variant("bad_syntax.json", None, text=json.dumps(base)[:-1] + ",}"), "error", "invalid JSON"),
        ("save_interval 0", variant("bad_si0.json", setv(["save_interval"], 0)), "error", "save_interval"),
        ("frames reversed", variant("bad_frames.json", setv(["frames"], [10, 5])), "error", '"frames"'),
        ("3 image lists for 4 cameras", variant("bad_imgs.json", setv(["image_lists"], base["image_lists"][:3])),
         "error", '"image_lists" must be an array of 4'),
        ("comments allowed", variant("ok_comments.json", None,
                                     text="// a comment\n" + json.dumps(base, indent=1).replace(
                                         '"fps"', '/* frame rate */ "fps"')), "ok", ""),
        ("text: save_interval = 0", variant("bad_si0.txt", None, text=txt_lines.replace("save_interval = 10", "save_interval = 0")),
         "error", "save_interval must be a positive integer"),
        ("text: save_interval = abc", variant("bad_si_abc.txt", None, text=txt_lines.replace("save_interval = 10", "save_interval = abc")),
         "error", "save_interval must be a positive integer"),
        ("text: unknown key", variant("warn_unknown.txt", None, text=txt_lines + "frobnicate = 3\n"), "warning",
         "unknown setting"),
        ("text: GUI-written (track-path lines, no save_interval)", variant("gui_style.txt", None, text=gui_style), "ok", ""),
    ]
    say("3. bad and edge-case files (C++ in a fresh process; Python module):")
    for name, path, expect, needle in cases:
        res, err = try_read_cpp(bdir, path)
        msg = res + (" | stderr: " + err.replace("\n", " | ") if err else "")
        if expect == "object_error":
            # the basic settings read fine; the object's settings fail with the named key
            code = f"""
import sys
sys.path.insert(0, {bdir!r})
import pyopenlpt as lpt
bs = lpt.BasicSetting(); bs.readConfig({path!r})
tc = lpt.TracerConfig()
try:
    tc.readConfig(bs._object_config_paths[0], bs); print("@@OK")
except Exception as e:
    print("@@ERR " + str(e).replace(chr(10), " | "))
"""
            r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("@@")]
            res = lines[-1] if lines else "@@CRASH"
            msg = res
            good = res.startswith("@@ERR") and needle in res
        elif expect == "error":
            good = res.startswith("@@ERR") and needle in res
        elif expect == "warning":
            good = res == "@@OK" and needle in err
        else:
            good = res == "@@OK" and "Warning" not in err
        # the Python module on the same file
        try:
            stb_config.read(path)
            py_res = "read"
        except stb_config.ConfigError as e:
            py_res = f"ConfigError: {e}"
        except Exception as e:  # noqa: BLE001 (reported)
            py_res = f"{type(e).__name__}: {e}"
        say(f"   {'ok  ' if good else 'FAIL'} {name}: C++ {msg[:220]}")
        say(f"        Python: {py_res[:200]}")
        if not good:
            failures.append(f"case {name}: {msg}")

    say("")
    say("C1 PASSED" if not failures else f"C1 FAILED ({len(failures)}): " + "; ".join(failures))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "c1_config.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(LOG) + "\n")
    with open(os.path.join(RESULTS, "c1_config.json"), "w") as f:
        json.dump({"failures": failures, "text_settings": dt, "json_settings": dj}, f, indent=1)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
