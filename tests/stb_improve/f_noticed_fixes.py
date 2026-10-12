"""Tests F1-F3: the fixes of the three issues noticed during A4b (report section 9), offscreen.

    python f_noticed_fixes.py

F1, the settings form loads the project's saved config (TrackingSettingsView._load_config, called when the project
folder is chosen), and Save keeps what the form does not show:
  a. round trip: the 31 forms of A5 saved as config.json, loaded by a fresh view, saved again: byte-identical files;
  b. text to JSON: the same 31 forms saved by the writer before A4b (config.txt + object file), and the benchmark's
     own config.txt (stbrun.py's, with the case's commented tracerConfig.txt), loaded and saved: the C++ code reads
     the same settings from the new config.json as from the text config (every bound setting);
  c. edits: loading, changing two fields and saving changes those two settings and nothing else;
  d. values the old form could not hold and keys it has no field for: threads 0 ("All"), fps 20000, frame 150000,
     voxel 0.1234567, a volume of -81.125 mm, "profile", a comment key, an unknown key and a second object come back
     unchanged;
  e. no config: the form keeps its values, nothing is reported; a broken or incomplete config: one warning, no crash,
     and the form is not marked as loaded;
  f. showing the view again (a tab switch) does not reload the config over unsaved edits.
F2, the GUI's log.txt: TrackingView._process_text turns the process's '\r\n' into '\n', also across a chunk boundary
between '\r' and '\n'; a GUI run of frames 0-5 (g_gui_e2e.py e2e) writes log.txt with plain CRLF line ends and no
'\r' in the log window, and the log still parses (6 frames).
F3, test isolation: tests/test_precalibrate_3d_view.py, before (commit acd78d4) and after the fix, followed by a check
that sys.modules holds the real PySide6: fails before, passes after.

Writes results/f_noticed_fixes.txt and .json; exit code 1 if anything fails.
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
sys.path.insert(0, HERE)
import a4b_gui_config as a4  # noqa: E402  (offscreen Qt, message boxes recorded, the code before A4b, helpers)
import stbrun  # noqa: E402
from a4b_gui_config import (CASE, MESSAGES, REPO, RESULTS, RUNS, check, cpp_dump, cpp_settings_equal,  # noqa: E402
                            fill_form, make_project, random_form, say)
from modules import stb_config  # noqa: E402
from views import tracking_view  # noqa: E402
from views.tracking_settings_view import TrackingSettingsView  # noqa: E402

PY = a4.PY


def load_into_fresh_view(proj):
    """A new settings view that opens the project folder as the GUI does (_update_paths after choosing it)."""
    v = TrackingSettingsView()
    v.project_path.setText(proj.replace("\\", "/"))
    v._update_paths()
    return v


def forms():
    import random
    rng = random.Random(20261005)  # A5's forms
    return [("case", CASE)] + [(f"random{k:02d}", random_form(rng)) for k in range(30)]


def test_round_trip():
    say("\nF1a round trip: config.json saved by the GUI, opened by a fresh view, saved again")
    writer = TrackingSettingsView()
    n_same, diffs = 0, []
    for name, v in forms():
        p = make_project(f"f1a_{name}", tracks=False)
        fill_form(writer, p, v)
        writer._save_configuration()
        path = os.path.join(p, "config.json")
        before = open(path, "rb").read()
        view = load_into_fresh_view(p)
        loaded = view._loaded_config is not None
        view._save_configuration()
        after = open(path, "rb").read()
        if loaded and before == after:
            n_same += 1
        else:
            diffs.append(name)
        view.deleteLater()
    check(f"31 forms (Tracer, Bubble, resume on/off, output folders): byte-identical after load and save",
          n_same == 31, f"{n_same}/31" + (f", differ: {diffs}" if diffs else ""))


def test_text_to_json():
    say("\nF1b text to JSON: a config.txt opened by the form and saved as config.json gives the same C++ settings")
    head_writer = a4.head_settings.TrackingSettingsView()
    pairs = []
    for name, v in forms():
        p = make_project(f"f1b_{name}", tracks=False)
        fill_form(head_writer, p, v)
        head_writer._save_configuration()  # config.txt + tracerConfig.txt / bubbleConfig.txt
        txt = os.path.join(p, "config.txt")
        txt_copy = os.path.join(p, "config_before.txt")
        shutil.copy(txt, txt_copy)
        view = load_into_fresh_view(p)
        ok_loaded = view._loaded_config is not None and view._loaded_config[1].endswith("config.txt")
        view._save_configuration()
        pairs.append((name, p, txt, os.path.join(p, "config.json"), ok_loaded))
        view.deleteLater()
    # the benchmark's own config.txt (stbrun.py), with the case's commented tracerConfig.txt
    p = os.path.join(RUNS, "f1b_benchmark")
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)
    ns = argparse.Namespace(frames=[0, 300], threads=26, load=None, load_from=None,
                            config_extra=["save_interval = 200", "profile = 1"], tracer_extra=[])
    stbrun.write_inputs(p, ns)
    view = load_into_fresh_view(p)
    ok_b = view._loaded_config is not None
    view._save_configuration()
    pairs.append(("benchmark", p, os.path.join(p, "config.txt"), os.path.join(p, "config.json"), ok_b))
    view.deleteLater()

    dumps = cpp_dump([q[2] for q in pairs] + [q[3] for q in pairs])
    d_txt, d_json = dumps[:len(pairs)], dumps[len(pairs):]
    n_ok, bad = 0, []
    for (name, p, txt, js, ok_loaded), a, b in zip(pairs, d_txt, d_json):
        if not ok_loaded or "error" in a or "error" in b:
            bad.append(f"{name}: loaded {ok_loaded}, {a.get('error')} | {b.get('error')}")
            continue
        same, diff = cpp_settings_equal(a, b, [p])
        if same and a["_save_interval"] == b["_save_interval"]:
            n_ok += 1
        else:
            bad.append(f"{name}: {diff} save_interval {a['_save_interval']} / {b['_save_interval']}")
    check(f"32 text configs (31 forms written before A4b and the benchmark's): the same C++ settings from the "
          f"config.json the form writes, save interval and profile included", n_ok == len(pairs),
          f"{n_ok}/{len(pairs)}" + (f"; {bad[:3]}" if bad else ""))
    bench = d_json[-1]
    check("the benchmark's config: save_interval 200 and profile on survive the conversion",
          bench.get("_save_interval") == 200 and bench.get("_profile") is True,
          f"{bench.get('_save_interval')} {bench.get('_profile')}")


def test_edits():
    say("\nF1c edits: two fields changed after loading; nothing else changes")
    p = make_project("f1c_edit", tracks=False)
    w = TrackingSettingsView()
    fill_form(w, p, CASE)
    w._save_configuration()
    before = stb_config.read_json_raw(os.path.join(p, "config.json"))
    view = load_into_fresh_view(p)
    view.ipr_2d_tol.setValue(1.5)
    view.n_threads_spin.setValue(0)
    view._save_configuration()
    after = stb_config.read_json_raw(os.path.join(p, "config.json"))
    expected = copy.deepcopy(before)
    expected["objects"][0]["ipr"]["tol_2d_px"] = 1.5
    expected["threads"] = 0
    check("2D tolerance 1.5 and threads 'All' (0) saved; every other setting as loaded", after == expected,
          "" if after == expected else json.dumps(after)[:300])
    check("the threads field shows 'All' for 0", view.n_threads_spin.text() == "All", view.n_threads_spin.text())
    view.deleteLater()


def test_special_values():
    say("\nF1d values the old form could not hold, and keys it has no field for")
    p = make_project("f1d_special", tracks=False)
    w = TrackingSettingsView()
    fill_form(w, p, CASE)
    w._save_configuration()
    path = os.path.join(p, "config.json")
    cfg = stb_config.read_json_raw(path)
    cfg.update(threads=0, fps=20000, frames=[0, 150000], voxel_mm=0.1234567, profile=True,
               _note="kept by the GUI", frobnicate={"a": 1})
    cfg["view_volume_mm"]["x"] = [-81.125, 54.0]
    bubble = {"type": "Bubble", "initial_phase": {"search_radius_vox": 8.0, "frames": 4},
              "convergence_phase": {"avg_spacing_vox": 20.0},
              "shake": {"width_vox": 0.5, "loops": 3, "ghost_threshold": 0.02},
              "predictive_field": {"grid": [31, 31, 31], "search_radius_vox": 20.0},
              "ipr": {"loops": 2, "reduced_cameras": 1, "reduced_loops": 2, "tol_2d_px": 1.0, "tol_3d_vox": 1.0},
              "bubble": {"radius_min_px": 4, "radius_max_px": 40, "sense": 0.85}}
    cfg["objects"].append(bubble)
    stb_config.write_json_raw(cfg, path)
    original = stb_config.read_json_raw(path)
    view = load_into_fresh_view(p)
    shown = (view.n_threads_spin.value(), view.fps_spin.value(), view.frame_end_spin.value(), view.voxel_spin.value(),
             view.vol_x_min.value())
    check("the form shows threads 0 ('All'), fps 20000, frame 150000, voxel 0.1234567, x min -81.125",
          shown == (0, 20000, 150000, 0.1234567, -81.125), str(shown))
    view._save_configuration()
    after = stb_config.read_json_raw(path)
    check("saved unchanged: those values, profile, the comment key, the unknown key and the second object",
          after == original, "" if after == original else
          str(sorted(k for k in set(after) | set(original) if after.get(k) != original.get(k))))
    view.deleteLater()


def test_missing_and_broken():
    say("\nF1e no config, a broken config, an incomplete config")
    p = make_project("f1e_none", tracks=False)
    n = len(MESSAGES)
    view = load_into_fresh_view(p)
    check("no config: nothing reported, the form keeps its values (not marked as loaded)",
          view._loaded_config is None and len(MESSAGES) == n and view.n_cam_spin.value() == 4)
    view.deleteLater()

    for name, text in (("broken", '{"frames": [0, 10],, }'), ("incomplete", None)):
        p = make_project(f"f1e_{name}", tracks=False)
        if text is None:  # valid JSON, but the object lacks its IPR settings
            w = TrackingSettingsView()
            fill_form(w, p, CASE)
            w._save_configuration()
            cfg = stb_config.read_json_raw(os.path.join(p, "config.json"))
            del cfg["objects"][0]["ipr"]
            stb_config.write_json_raw(cfg, os.path.join(p, "config.json"))
        else:
            with open(os.path.join(p, "config.json"), "w") as f:
                f.write(text)
        n = len(MESSAGES)
        view = load_into_fresh_view(p)
        new = MESSAGES[n:]
        check(f"{name} config.json: one warning that names it, no crash, the form not marked as loaded",
              view._loaded_config is None and len(new) == 1 and new[0][0] == "warning"
              and "config.json in this project cannot be read" in new[0][2], new[0][2][:110].replace("\n", " ")
              if new else "no message")
        view.deleteLater()


def test_no_reload_on_show():
    say("\nF1f showing the view again does not reload over unsaved edits")
    from PySide6.QtGui import QShowEvent
    p = make_project("f1f_show", tracks=False)
    w = TrackingSettingsView()
    fill_form(w, p, CASE)
    w._save_configuration()
    view = load_into_fresh_view(p)
    view.ipr_2d_tol.setValue(3.25)  # an unsaved edit
    view.showEvent(QShowEvent())    # switching back to the tab
    check("the edit survives a show event", view.ipr_2d_tol.value() == 3.25, str(view.ipr_2d_tol.value()))
    view.deleteLater()


def test_log_line_ends():
    say("\nF2 the GUI's log.txt line ends")
    tv = tracking_view.TrackingView(settings_view=None)
    tv._pending_cr = {}
    chunks = [b"a\r\nb\r", b"\nc\r\n", b"d\r\n"]
    out = "".join(tv._process_text(c, "out") for c in chunks)
    err = tv._process_text(b"e\r", "err") + tv._process_text(b"\nf\r\n", "err")
    check("'\\r\\n' becomes '\\n', also split across two chunks; stdout and stderr kept apart",
          out == "a\nb\nc\nd\n" and err == "e\nf\n", repr(out) + " " + repr(err))
    tv.deleteLater()

    name = "f2_log_smoke"
    r = subprocess.run([PY, os.path.join(HERE, "g_gui_e2e.py"), "e2e", "--frames", "0", "5", "--name", name],
                       capture_output=True, text=True)
    run_dir = os.path.join(RUNS, name)
    raw = open(os.path.join(run_dir, "log.txt"), "rb").read()
    info = json.load(open(os.path.join(run_dir, "run_info.json")))
    n_lf, n_crlf, n_crcrlf = raw.count(b"\n"), raw.count(b"\r\n"), raw.count(b"\r\r\n")
    check(f"a GUI run of frames 0-5: log.txt has {n_lf} line ends, all CRLF, none CRCRLF; {info['frames_done']} frames "
          f"parsed", r.returncode == 0 and n_lf == n_crlf and n_crcrlf == 0 and info["frames_done"] == 6,
          f"exit {r.returncode}; LF {n_lf}, CRLF {n_crlf}, CRCRLF {n_crcrlf}")
    check("the log window holds no '\\r'", info.get("log_window_has_cr") is False, str(info.get("log_window_has_cr")))


def test_isolation():
    say("\nF3 test isolation: does tests/test_precalibrate_3d_view.py leave mocks in sys.modules?")
    # named check_*.py, not test_*.py: pytest runs them when named on its command line, but a plain
    # "pytest tests" does not collect them
    folder = os.path.join(RUNS, "f3_isolation")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    with open(os.path.join(folder, "check_zz_qt_is_real.py"), "w") as f:
        f.write("import sys, types\n\n\ndef test_pyside6_is_the_real_module():\n"
                "    assert isinstance(sys.modules['PySide6'], types.ModuleType)\n"
                "    assert isinstance(sys.modules['PySide6.QtWidgets'], types.ModuleType)\n")
    old = subprocess.run(["git", "-C", REPO, "show", "acd78d4:tests/test_precalibrate_3d_view.py"],
                         capture_output=True, check=True).stdout
    with open(os.path.join(folder, "check_precalibrate_before.py"), "wb") as f:
        f.write(old)
    shutil.copy(os.path.join(REPO, "tests", "test_precalibrate_3d_view.py"),
                os.path.join(folder, "check_precalibrate_after.py"))
    res = {}
    for which in ("before", "after"):
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", REPO,
                            os.path.join(folder, f"check_precalibrate_{which}.py"),
                            os.path.join(folder, "check_zz_qt_is_real.py")],
                           capture_output=True, text=True, cwd=REPO)
        res[which] = (r.returncode, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])
    check("before the fix the check fails; after it, all pass",
          res["before"][0] != 0 and "1 failed" in res["before"][1] and res["after"][0] == 0,
          f"before: {res['before'][1]} | after: {res['after'][1]}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.chdir(REPO)
    say("F1-F3: the fixes of the issues noticed during A4b, offscreen Qt")
    test_round_trip()
    test_text_to_json()
    test_edits()
    test_special_values()
    test_missing_and_broken()
    test_no_reload_on_show()
    test_log_line_ends()
    test_isolation()
    n_fail = sum(not c["ok"] for c in a4.CHECKS)
    say(f"\n{len(a4.CHECKS) - n_fail} of {len(a4.CHECKS)} checks passed" + ("" if not n_fail else f"; {n_fail} FAILED"))
    with open(os.path.join(RESULTS, "f_noticed_fixes.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(a4.LOG) + "\n")
    with open(os.path.join(RESULTS, "f_noticed_fixes.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"checks": a4.CHECKS, "messages": MESSAGES}, f, indent=1)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
