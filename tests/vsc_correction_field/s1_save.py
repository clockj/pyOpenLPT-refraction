"""S1: the tracking settings page's Save on tube, sphere, plate and pinhole projects (decision 2, 2026-10-09-plan.md).

    python s1_save.py

Every case runs the same flow with today's page (gui/views/tracking_settings_view.py as commit cffed7d has it) and with
the change (the working tree's), each on its own fresh copy of the project: the project folder set, the paths and the
saved config loaded as the page does when it is shown (_update_paths), the form edited, Save pressed (message boxes
recorded). Qt offscreen, build vcf's pyopenlpt. The projects (projects/s1_*, git-ignored) are built here from read-only
sources, each with its cameras in camFile/:
- tube: T1 after a field VSC pass (projects/t1_selfcalproto: the line detector, the OTF table, the depth cue, the
  cameras' corrections), T0's PINCYL cameras;
- sphere: the same with the cameras made PINSPH (the model line changed);
- plate: the 20260727 plate run's project (X:, read only: config.txt, tracerConfig.txt, camFile/cam0..3.txt,
  PINPLATE); and the same saved once to config.json with settings the form does not show added by hand;
- pinhole: the tube project with T0's cameras cut to pinholes (the model line PINHOLE, the port lines dropped).
Pass:
- tube and sphere: the change's file = today's file + exactly the object's settings the form does not show, with
  their saved values (the tracer's eight), the form's edits applied in both; a change of the object's type gives
  today's file; a second load and Save of the change's file changes nothing;
- plate and pinhole: the change's file is byte-identical to today's.
Writes s1_save_out.txt and .json here.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import vcf_common as vc  # noqa: E402
from h1_vsc_thread import Immediate  # noqa: E402

PROJ = os.path.join(vc.HERE, "projects")
T1_SRC = os.path.join(PROJ, "t1_selfcalproto")
PLATE_X = r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results\10Hz_r_b_1\stb_new_calib\new_cams"
TUBE_EXTRAS = ["detector", "lines", "otf_file", "roi_from_otf", "roi_sigma", "shape_check", "shape_tol_px",
               "shape_tol_rel"]


# ---------------------------------------------------------------------------------------------- the projects
def _cam_files(kind):
    """T0's two camera files as text, as `kind` (PINCYL as they are, PINSPH, PINHOLE)."""
    out = []
    for p in vc.CAMS_T1:
        with open(p, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        assert lines[1].strip() == "PINCYL", p
        if kind == "PINSPH":
            lines[1] = "PINSPH"
        elif kind == "PINHOLE":
            lines[1] = "PINHOLE"
            cut = next(i for i, ln in enumerate(lines) if ln.startswith("# Port Center"))
            lines = lines[:cut]
        out.append("\n".join(lines) + "\n")
    return out


def make_t1(name, kind):
    """T1 after a VSC pass as a self-contained project, its cameras of `kind`."""
    proj = os.path.join(PROJ, name)
    if os.path.exists(proj):
        shutil.rmtree(proj)
    os.makedirs(os.path.join(proj, "camFile"))
    shutil.copytree(os.path.join(T1_SRC, "camFile_VSC"), os.path.join(proj, "camFile_VSC"))
    os.makedirs(os.path.join(proj, "Results"))
    for k, text in enumerate(_cam_files(kind)):
        with open(os.path.join(proj, "camFile", f"cam{k + 1}.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    cfg = vc.load_json(os.path.join(T1_SRC, "config.json"))
    for k, cam in enumerate(cfg["cameras"]):
        cam["file"] = f"camFile/cam{k + 1}.txt"
    cfg["_comment"] = f"S1 (tests/vsc_correction_field): T1 after a VSC pass, its cameras {kind}"
    vc.save_json(os.path.join(proj, "config.json"), cfg)
    return proj


def make_plate(name):
    proj = os.path.join(PROJ, name)
    if os.path.exists(proj):
        shutil.rmtree(proj)
    os.makedirs(proj)
    for f in ("config.txt", "tracerConfig.txt"):
        shutil.copy2(os.path.join(PLATE_X, f), proj)
    shutil.copytree(os.path.join(PLATE_X, "camFile"), os.path.join(proj, "camFile"))
    os.makedirs(os.path.join(proj, "Results"))
    return proj


def clone(src, name):
    dst = os.path.join(PROJ, name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


# ---------------------------------------------------------------------------------------------- the flow
def save_flow(mod, app, proj, edits):
    """The page of module `mod` on `proj`: shown (paths and the saved config), edited, saved. Returns (seconds of the
    load, the boxes shown, the saved file's bytes)."""
    Immediate.shown = []
    mod.QMessageBox = Immediate
    view = mod.TrackingSettingsView()
    view.project_path.setText(proj)
    app.processEvents()
    t0 = time.time()
    view._update_paths()
    app.processEvents()
    t_load = time.time() - t0
    for name, value in edits:
        w = getattr(view, name)
        if name == "obj_type_combo":
            w.setCurrentIndex(w.findText(value))
        else:
            w.setValue(value)
    app.processEvents()
    view._save_configuration()
    app.processEvents()
    with open(os.path.join(proj, "config.json"), "rb") as fh:
        data = fh.read()
    boxes = [m["title"] for m in Immediate.shown]
    view.deleteLater()
    app.processEvents()
    return t_load, boxes, data


def json_diff(a, b, path=""):
    """Where JSON values a and b differ: (path, a's value or '<absent>', b's value or '<absent>')."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            p = f"{path}.{k}" if path else k
            if k not in a:
                out.append((p, "<absent>", b[k]))
            elif k not in b:
                out.append((p, a[k], "<absent>"))
            else:
                out += json_diff(a[k], b[k], p)
    elif a != b:
        out.append((path, a, b))
    return out


def main():
    lpt = vc.lib("vcf")
    log = vc.Log("s1_save")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    today = vc.committed_module("gui/views/tracking_settings_view.py", "tracking_settings_today")
    import gui.views.tracking_settings_view as change
    log(f"S1: today's page {today.__file__}, the change {change.__file__}; pyopenlpt {lpt.__file__}")

    tube = make_t1("s1_tube", "PINCYL")
    sphere = make_t1("s1_sphere", "PINSPH")
    pinhole = make_t1("s1_pinhole", "PINHOLE")
    plate = make_plate("s1_plate")
    # the plate project as config.json (saved once by today's page), then settings the form does not show added
    plate_json = clone(plate, "s1_plate_json")
    save_flow(today, app, plate_json, [])
    os.remove(os.path.join(plate_json, "config.txt"))
    pj = vc.load_json(os.path.join(plate_json, "config.json"))
    pj["objects"][0]["tracer"].update({"detector": "peaks", "otf_file": "otf_by_hand.txt", "shape_check": False})
    pj["objects"][0]["ipr"]["by_hand"] = 1
    pj["objects"][0]["by_hand"] = {"note": "a setting the form does not show"}
    vc.save_json(os.path.join(plate_json, "config.json"), pj)

    tol = [("ipr_2d_tol", 1.0)]
    cases = [
        ("tube", tube, tol, "curved"),
        ("tube, the tracer's threshold and radius edited", tube,
         tol + [("tracer_int_thresh", 30), ("tracer_radius", 2.5)], "curved"),
        ("tube, the object's type changed to Bubble", tube, tol + [("obj_type_combo", "Bubble")], "same"),
        ("sphere", sphere, tol, "curved"),
        ("plate (config.txt)", plate, tol, "same"),
        ("plate (config.json with settings the form does not show)", plate_json, tol, "same"),
        ("pinhole", pinhole, tol, "same"),
    ]
    out = {"cases": []}
    ok_all = True
    for label, src, edits, expect in cases:
        a = clone(src, "s1_run_today")
        b = clone(src, "s1_run_change")
        before_path = os.path.join(src, "config.json")
        before = vc.load_json(before_path) if os.path.exists(before_path) else None
        ta, boxes_a, da = save_flow(today, app, a, edits)
        tb, boxes_b, db = save_flow(change, app, b, edits)
        ja, jb = json.loads(da), json.loads(db)
        diff = json_diff(ja, jb)
        rec = {"case": label, "expect": expect, "edits": edits, "load_s_today": round(ta, 2),
               "load_s_change": round(tb, 2), "boxes_today": boxes_a, "boxes_change": boxes_b,
               "identical_bytes": da == db, "diff_today_to_change": diff}
        if expect == "same":
            ok = da == db
        else:
            tr0 = before["objects"][0]["tracer"]
            want = sorted(f"objects.0.tracer.{k}" for k in TUBE_EXTRAS)
            ok = False
            # json_diff walks dicts only: compare objects[0] and the rest of the file apart
            if len(ja["objects"]) == len(jb["objects"]):
                d0 = json_diff(ja["objects"][0], jb["objects"][0])
                got = sorted(f"objects.0.{p}" for p, _x, _y in d0)
                others = json_diff({k: v for k, v in ja.items() if k != "objects"},
                                   {k: v for k, v in jb.items() if k != "objects"})
                values_ok = all(y == tr0[p.split(".")[-1]] and x == "<absent>" for p, x, y in d0)
                edits_ok = all(
                    (jb["objects"][0]["ipr"]["tol_2d_px"] == v) if n == "ipr_2d_tol" else
                    (jb["objects"][0]["tracer"]["intensity_threshold"] == v) if n == "tracer_int_thresh" else
                    (jb["objects"][0]["tracer"]["radius_px"] == v) if n == "tracer_radius" else True
                    for n, v in edits)
                corr_ok = [c.get("correction") for c in jb["cameras"]] == [c.get("correction")
                                                                           for c in before["cameras"]]
                ok = got == want and not others and values_ok and edits_ok and corr_ok
                rec.update({"kept": got, "values_as_saved": values_ok, "edits_applied": edits_ok,
                            "corrections_kept": corr_ok, "other_differences": others})
            # a second load and Save of the change's file changes nothing
            c2 = clone(b, "s1_run_change2")
            _t, _bx, d2 = save_flow(change, app, c2, [])
            rec["second_save_identical"] = d2 == db
            ok = ok and d2 == db
        rec["pass"] = bool(ok)
        ok_all &= ok
        out["cases"].append(rec)
        msg = (f"{label}: " + ("byte-identical" if da == db else f"{len(diff)} difference(s) today -> change")
               + f"; load {ta:.1f} / {tb:.1f} s (today / change); boxes {boxes_b}")
        if expect == "curved":
            msg += (f"; kept {len(rec.get('kept', []))} of {len(TUBE_EXTRAS)} ({', '.join(p.split('.')[-1] for p in rec.get('kept', []))}), "
                    f"values as saved {rec.get('values_as_saved')}, edits applied {rec.get('edits_applied')}, "
                    f"corrections kept {rec.get('corrections_kept')}, other differences {rec.get('other_differences')}, "
                    f"second Save identical {rec.get('second_save_identical')}")
        elif da != db:
            msg += f"; differences: {diff[:6]}"
        log(msg + f" -> {'PASS' if ok else 'FAIL'}")
    out["pass"] = bool(ok_all)
    log(f"S1: {'PASS' if ok_all else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
