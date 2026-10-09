"""S2: the tracking settings page's view-volume estimate on tube, plate and pinhole rigs (decision 2, 2026-10-09-plan.md).

    python s2_volume.py

The page's camera step (the camera folder set: its files parsed, the 2-D and 3-D tolerances and the view volume
estimated) with today's page (commit cffed7d's file) and with the change, on copies of the camera files
(projects/s2_*/camFile, git-ignored; the page may patch a file's rotation vector). Qt offscreen, build vcf's pyopenlpt.
- Tube rigs (every camera PINCYL):
  - T1 (two cameras, T0's calibration);
  - 20261008 (three cameras, the cropped third camera of its final run);
  - the synthetic sets with 2, 3 and 4 cameras (tests/cylinder_selfcal/data/cal4).
  Each is scored against two things:
  - a reference: the points every camera sees, by the library's projection (on the sensor, as the page counts) on a
    grid of spacing h over the estimate widened by 8 mm. The reference is complete when no such point lies on that
    grid's outer faces;
  - the particles: T1's and 20261008's tracked points (their final runs), and the synthetic sets' true particles seen
    by every camera.
- Plate (the 20260727 run's four PINPLATE cameras) and pinhole (T0's cameras cut to pinholes): today's and the
  change's estimates are equal.
Pass:
- tube: the change's box (before the page's rounding to 5 mm) holds the reference box within h, and lies within 5 mm
  of it on every side; every particle lies in the page's rounded box;
- plate and pinhole: every value equal.
Writes s2_volume_out.txt and .json here.
"""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import vcf_common as vc  # noqa: E402
from h1_vsc_thread import Immediate  # noqa: E402
from s1_save import PLATE_X, _cam_files  # noqa: E402

STUDY = os.path.join(vc.REPO, "tests", "cylinder_stb_study")
sys.path.insert(0, STUDY)
import eval_study  # noqa: E402
from eval_truth import read_tracks  # noqa: E402

PROJ = os.path.join(vc.HERE, "projects")
C08 = os.path.join(vc.REPO, "cases", "cyl_20261008", "data")
CAL4 = os.path.join(vc.SELFCAL, "data", "cal4")
FIELDS = ("vol_x_min", "vol_x_max", "vol_y_min", "vol_y_max", "vol_z_min", "vol_z_max", "voxel_spin", "ipr_2d_tol",
          "ipr_3d_tol")


def camfolder(name, files=None, texts=None):
    d = os.path.join(PROJ, name, "camFile")
    if os.path.exists(os.path.dirname(d)):
        shutil.rmtree(os.path.dirname(d))
    os.makedirs(d)
    if files:
        for f in files:
            shutil.copy2(f, d)
    for k, text in enumerate(texts or []):
        with open(os.path.join(d, f"cam{k + 1}.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return d


def camera_step(mod, app, cam_dir):
    """The page of module `mod`, its camera folder set: (values of FIELDS before, after, seconds, boxes)."""
    Immediate.shown = []
    mod.QMessageBox = Immediate
    view = mod.TrackingSettingsView()
    app.processEvents()
    before = {f: getattr(view, f).value() for f in FIELDS}
    t0 = time.time()
    view.camera_path_display.setText(cam_dir)   # textChanged -> _on_cam_path_changed
    app.processEvents()
    dt = time.time() - t0
    after = {f: getattr(view, f).value() for f in FIELDS}
    view.deleteLater()
    app.processEvents()
    return before, after, dt, [m["title"] for m in Immediate.shown]


def box_of(vals):
    return (np.array([vals["vol_x_min"], vals["vol_y_min"], vals["vol_z_min"]]),
            np.array([vals["vol_x_max"], vals["vol_y_max"], vals["vol_z_max"]]))


def seen_by_all(lpt, cam_files, lo, hi, h):
    """Grid points (spacing h, over [lo, hi]) every camera sees on its sensor, by the library's projection; and
    whether any of them lies on the grid's outer faces."""
    axes = [np.arange(lo[k], hi[k] + 0.5 * h, h) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    keep = np.ones(len(G), dtype=bool)
    for f in cam_files:
        cam = lpt.Camera(f)
        rows, cols = image_size(f)
        idx = np.flatnonzero(keep)
        st = cam.projectBatchStatus([lpt.Pt3D(*map(float, p)) for p in G[idx]], False)
        ok = np.array([bool(s[0]) and 0 <= s[1][0] < cols and 0 <= s[1][1] < rows for s in st])
        keep[idx[~ok]] = False
    P = G[keep]
    face = False
    if len(P):
        n = [len(a) for a in axes]
        ijk = np.stack(np.unravel_index(np.flatnonzero(keep), n), axis=1)
        face = bool(((ijk == 0) | (ijk == np.array(n) - 1)).any())
    return P, face, len(G)


def image_size(path):
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh]
    i = next(i for i, ln in enumerate(lines) if ln.startswith("# Image Size"))
    r, c = [int(float(x)) for x in lines[i + 1].replace(",", " ").split()[:2]]
    return r, c


def track_points(run_dir):
    tracks, _kinds = read_tracks(run_dir)
    return np.array([[x, y, z] for pts in tracks.values() for (_t, x, y, z) in pts], float).reshape(-1, 3)


def main():
    lpt = vc.lib("vcf")
    log = vc.Log("s2_volume")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    today = vc.committed_module("gui/views/tracking_settings_view.py", "tracking_settings_today")
    import gui.views.tracking_settings_view as change
    log(f"S2: today's page {today.__file__}, the change {change.__file__}; pyopenlpt {lpt.__file__}")
    c3 = vc.load_json(os.path.join(C08, "runs", "s3_final_c3", "config_final_c3.json"))
    sets = vc.load_json(os.path.join(CAL4, "sets.json"))

    tube = [("T1, 2 cameras", vc.CAMS_T1, 1.0, ("tracks", os.path.join(vc.CASE_RUNS, "s3_final_m15"))),
            ("20261008, 3 cameras", [c["file"] for c in c3["cameras"]], 2.0,
             ("tracks", os.path.join(C08, "runs", "s3_final_c3")))]
    for n in (2, 3, 4):
        tube.append((f"synthetic, {n} cameras", [os.path.join(CAL4, "camFile", f"cam{k + 1}.txt") for k in range(n)],
                     1.0, ("truth", n)))
    out = {"tube": [], "same": []}
    ok_all = True
    for label, files, h, parts in tube:
        d = camfolder("s2_" + label.split(",")[0].lower().replace(" ", "") + f"_{len(files)}", files=files)
        b0, ta, dta, _bx = camera_step(today, app, d)
        _b1, tb, dtb, _bx = camera_step(change, app, d)
        cams = sorted(os.path.join(d, f) for f in os.listdir(d))
        # the change's box before rounding: the page's own method on the copied files
        view = change.TrackingSettingsView()
        data = []
        for f in cams:
            p = view._parse_cam_file(f)
            p["cam_file_path"] = f
            data.append(p)
        raw = view._curved_port_common_box(data)
        view.deleteLater()
        rec = {"rig": label, "cameras": cams, "today": ta, "change": tb, "defaults": b0,
               "today_changed_volume": any(ta[f] != b0[f] for f in FIELDS[:6]),
               "seconds_today": round(dta, 2), "seconds_change": round(dtb, 2), "h_mm": h}
        if raw is None:
            rec["pass"] = False
            log(f"{label}: the change found no box -> FAIL")
            out["tube"].append(rec)
            ok_all = False
            continue
        lo, hi = raw
        P, face, n_grid = seen_by_all(lpt, cams, lo - 8.0, hi + 8.0, h)
        rlo, rhi = P.min(axis=0), P.max(axis=0)
        holds = bool(np.all(lo <= rlo + h) and np.all(hi >= rhi - h))
        excess = np.concatenate([rlo - lo, hi - rhi])
        tight = bool(np.all(excess <= 5.0))
        plo, phi = box_of(tb)
        if parts[0] == "tracks":
            X = track_points(parts[1])
            what = f"tracked points of {os.path.basename(parts[1])}"
        else:
            set_name = "Gcal4_n8"
            Xt, _layer, n_seen = eval_study.truth(CAL4, set_name, list(range(parts[1])))
            X = Xt[n_seen >= parts[1]]
            what = f"true particles seen by all {parts[1]} cameras ({set_name}, every frame)"
        inside = np.all((X >= plo) & (X <= phi), axis=1)
        inside_today = np.all((X >= box_of(ta)[0]) & (X <= box_of(ta)[1]), axis=1)
        ok = holds and tight and not face and bool(inside.all())
        ok_all &= ok
        rec.update({"raw_box": [lo.tolist(), hi.tolist()], "reference_box": [rlo.tolist(), rhi.tolist()],
                    "reference_points": int(len(P)), "reference_grid": int(n_grid), "reference_complete": not face,
                    "holds_reference": holds, "excess_mm": excess.tolist(), "within_5mm": tight,
                    "particles": what, "n_particles": int(len(X)), "inside_change": int(inside.sum()),
                    "inside_today": int(inside_today.sum()), "pass": ok})
        out["tube"].append(rec)
        fmt = lambda a, b: ", ".join(f"{ax} [{p:.1f}, {q:.1f}]" for ax, p, q in zip("xyz", a, b))  # noqa: E731
        log(f"{label}: today's page {'set ' + fmt(*box_of(ta)) if rec['today_changed_volume'] else 'found no volume'} "
            f"({dta:.1f} s); the change {fmt(plo, phi)}, voxel {tb['voxel_spin']:.4f} mm ({dtb:.1f} s)")
        log(f"  the change's box before rounding {fmt(lo, hi)}; the reference (projection, h = {h} mm, "
            f"{len(P)} of {n_grid} points, complete {not face}) {fmt(rlo, rhi)}; holds it {holds}, wider by "
            f"{excess.min():.2f} to {excess.max():.2f} mm")
        log(f"  {what}: {int(inside.sum())} of {len(X)} inside the change's box"
            + (f", {int(inside_today.sum())} inside today's" if rec['today_changed_volume'] else "")
            + f"; tolerances 2-D {tb['ipr_2d_tol']} px, 3-D {tb['ipr_3d_tol']} vox (today {ta['ipr_2d_tol']}, "
              f"{ta['ipr_3d_tol']}) -> {'PASS' if ok else 'FAIL'}")

    plate = camfolder("s2_plate", files=[os.path.join(PLATE_X, "camFile", f"cam{k}.txt") for k in range(4)])
    pinhole = camfolder("s2_pinhole", texts=_cam_files("PINHOLE"))
    for label, d in (("plate, 4 cameras (PINPLATE)", plate), ("pinhole, 2 cameras", pinhole)):
        b0, ta, dta, _ = camera_step(today, app, d)
        _b, tb, dtb, _ = camera_step(change, app, d)
        same = ta == tb
        ok_all &= same
        out["same"].append({"rig": label, "today": ta, "change": tb, "defaults": b0, "equal": same,
                            "seconds_today": round(dta, 2), "seconds_change": round(dtb, 2)})
        moved = any(ta[f] != b0[f] for f in FIELDS[:6])
        log(f"{label}: today {'estimated ' + str([round(ta[f], 3) for f in FIELDS[:7]]) if moved else 'found no volume'}"
            f" ({dta:.1f} s), the change {'equal' if same else 'DIFFERENT: ' + str(tb)} ({dtb:.1f} s) -> "
            f"{'PASS' if same else 'FAIL'}")
    out["pass"] = bool(ok_all)
    log(f"S2: {'PASS' if ok_all else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
