"""X2 (B11): the Pinhole model calibrates cameras of different image sizes; X3 (B12): cameras numbered with a gap.
Through the GUI, offscreen: the camera table, Run Calibration with the model "Pinhole" (distortion None), then Save
into data/x2_x3_run/<case>/ (the result box's "Save" is clicked by the test).

    python x2_x3_pinhole_sizes_ids.py

The rig is W10's (w9_w10_pinhole.py: OpenCV cameras at -30, 0 and 30 deg, 600 mm from a 100 mm volume, f 2,000 px,
the principal point at each image's centre, 300 wand frames of 10 mm, 0.1 px noise; `seed` draws the frames and the
noise); a camera of 1024 x 768 px among cameras of 1280 x 800 px:
  ref     three cameras of 1280 x 800, ids 0, 1, 2, seed 1314 (W10's run)
  refS    three cameras of 1280 x 800, seed 1315: X2a's and X2b's noise
  ref2S   its first two cameras: X2c's noise
  X2a     seed 1315, camera 1 of 1024 x 768
  X2b     seed 1315, camera 0 of 1024 x 768
  X2c     seed 1315, two cameras, camera 1 of 1024 x 768 (the two-camera path)
  X3a     ref's points with the ids 0, 2, 3 (a camera table of four rows; camera 1 has no points)
  X3b     X2a's points with the ids 0, 2, 3 (camera 2 of 1024 x 768): both fixes together
A camera of the small size, with its principal point at its centre, sees the same image as a large one shifted by the
difference of the centres; with each camera bounded around its own centre the calibration is the same problem, so
X2a, X2b and X2c must give the answer of the run of large cameras with the same noise (refS, ref2S), up to the
optimiser's tolerance (the best pair it starts from may differ: it is chosen on raw pixel disparities).
Checks:
  - X2: the calibration succeeds; against its reference of large cameras, in the quantities that do not depend on the
    world frame, f within 1e-3 (relative), the principal point's offset from the image centre within 0.5 px, the
    distances between the cameras within 1e-3 (relative), the rotations between the cameras within 1e-4 rad, the RMS
    within 0.01 px; each camFile and the results page with the camera's own size; every frame of the error table
    with its projections;
  - X3: the calibration succeeds and every camera's parameters equal those of the run numbered 0, 1, 2 (ref, X2a) to
    the bit; the camFiles cam0, cam2, cam3 equal its cam0, cam1, cam2 byte for byte; the error table's columns
    Cam 1, Cam 3, Cam 4; the results page's cameras cam0, cam2, cam3.
The accuracy against the truth is printed for each run.  (The first version compared X2c with a run of other noise
and failed on its baseline: x2_x3_pinhole_sizes_ids_first_try_out.txt.)
Writes x2_x3_pinhole_sizes_ids_out.txt and .json.
"""
import common as C  # noqa: I001

import io  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from w9_w10_pinhole import gui_points, image_size_line, rig  # noqa: E402

log = C.Log("x2_x3_pinhole_sizes_ids")
BIG, SMALL = (1280, 800), (1024, 768)


class Tee(io.TextIOBase):
    """The console output, kept as well (the calibrator prints the pair it starts from)."""

    def __init__(self, stream):
        self.stream, self.parts = stream, []

    def write(self, s):
        self.parts.append(s)
        return self.stream.write(s)

    def flush(self):
        self.stream.flush()


def angle(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def main():
    run_dir = C.DATA / "x2_x3_run"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    app = C.qt_app()  # noqa: F841
    dlg = C.Dialogs(folder=str(run_dir))
    from PySide6.QtWidgets import QMessageBox

    def click_save(box):
        for b in box.buttons():
            if b.text() == "Save":
                b.click()
                break
        return 0
    QMessageBox.exec = click_save
    from modules.camera_calibration import CameraCalibrationView
    from modules.camera_calibration.results_page import page as rp

    def calibrate(name, sizes, ids, seed, cams=(0, 1, 2)):
        """sizes: per camera of the run; ids: the camera ids they get in the points and the table."""
        true, uv, _A, _B, on = rig(np.random.default_rng(seed), list(sizes) + [BIG] * (3 - len(sizes)))
        uv = uv[:, list(cams)]
        folder = run_dir / name
        folder.mkdir()
        dlg.folder = str(folder)
        view = CameraCalibrationView()
        view.wand_num_cams.setValue(max(ids) + 1)
        size_of = {cid: sizes[k] for k, cid in enumerate(ids)}
        for r in range(max(ids) + 1):
            view.wand_cam_table.cellWidget(r, 2).setValue(2000)
            view.wand_cam_table.cellWidget(r, 3).setValue(size_of.get(r, BIG)[0])
            view.wand_cam_table.cellWidget(r, 4).setValue(size_of.get(r, BIG)[1])
        wp = {f: {ids[c]: pts for c, pts in fr.items()} for f, fr in gui_points(uv).items()}
        view.wand_calibrator.wand_points = wp
        view.wand_calibrator.wand_points_filtered = None
        view.wand_model_combo.setCurrentText("Pinhole")
        view.wand_len_spin.setValue(10.0)
        view.dist_model_combo.setCurrentIndex(0)
        n = len(dlg.messages)
        tee = Tee(sys.stdout)
        sys.stdout = tee
        t0 = time.time()
        try:
            view._run_wand_calibration()
            w = view._calib_worker
            C.wait_for(lambda: w.isFinished() and any(m[1] in ("Saved", "Calibration Failed")
                                                      for m in dlg.messages[n:]), 900)
        finally:
            sys.stdout = tee.stream
        secs = time.time() - t0
        pair = re.findall(r"Auto-selected best pair: Cam (\d+) <-> Cam (\d+)", "".join(tee.parts))
        msgs = dlg.messages[n:]
        wc = view.wand_calibrator
        ok = any(m[1] == "Saved" for m in msgs) and bool(wc.final_params)
        rec = {"name": name, "ok": ok, "seconds": secs, "pair": [int(v) for v in pair[-1]] if pair else None,
               "messages": [m[:2] for m in msgs], "fail": next((m[2] for m in msgs if m[1] == "Calibration Failed"), None),
               "seed": seed, "sizes": {cid: list(size_of[cid]) for cid in ids}}
        if not ok:
            view.close()
            return rec
        fp = wc.final_params
        # per camera, in the run's order (k), frame-free quantities, and against the truth
        rec["f"] = [float(fp[ids[k]]["K"][0, 0]) for k in range(len(cams))]
        rec["pp_off"] = [[float(fp[ids[k]]["K"][0, 2] - size_of[ids[k]][0] / 2.0),
                          float(fp[ids[k]]["K"][1, 2] - size_of[ids[k]][1] / 2.0)] for k in range(len(cams))]
        rec["R"] = [np.asarray(fp[ids[k]]["R"], float) for k in range(len(cams))]
        cen = [(-fp[ids[k]]["R"].T @ fp[ids[k]]["T"]).ravel() for k in range(len(cams))]
        cen_true = [(-true[c]["R"].T @ true[c]["T"]).ravel() for c in cams]
        pairs = [(a, b) for a in range(len(cams)) for b in range(a + 1, len(cams))]
        rec["baselines"] = [float(np.linalg.norm(cen[a] - cen[b])) for a, b in pairs]
        rec["f_err_rel"] = [rec["f"][k] / float(true[c]["K"][0, 0]) - 1 for k, c in enumerate(cams)]
        rec["pp_err_px"] = [float(np.hypot(*rec["pp_off"][k])) for k in range(len(cams))]   # the truth: the centre
        rec["baseline_err_rel"] = [rec["baselines"][j] / float(np.linalg.norm(cen_true[a] - cen_true[b])) - 1
                                   for j, (a, b) in enumerate(pairs)]
        d, missing = [], 0
        for fid, e in wc.per_frame_errors.items():
            for c, pts in wp[fid].items():
                pp = (e.get("proj_pts") or {}).get(c)
                if pp is None:
                    missing += 1
                    continue
                for k in (0, 1):
                    d.append(np.linalg.norm(np.asarray(pp[k], float) - np.asarray(pts[k][:2], float)))
        rec["rms_px"] = float(np.sqrt(np.mean(np.square(d)))) if d else float("nan")
        rec["proj_missing"] = missing
        rec["frames"] = len(wc.per_frame_errors)
        rec["camfile_sizes"] = {cid: image_size_line(folder / "camFile" / f"cam{cid}.txt")
                                for cid in sorted(fp) if (folder / "camFile" / f"cam{cid}.txt").exists()}
        rec["camfiles"] = {cid: (folder / "camFile" / f"cam{cid}.txt").read_bytes()
                           for cid in sorted(fp) if (folder / "camFile" / f"cam{cid}.txt").exists()}
        rec["want_sizes"] = {cid: f"{size_of[cid][1]},{size_of[cid][0]}" for cid in ids}
        rec["headers"] = [view.error_table.horizontalHeaderItem(i).text() for i in range(view.error_table.columnCount())]
        report = rp.report_from_pinhole(fp, wp, 10.0)
        rec["page_cams"] = [(c["name"], c["width"], c["height"]) for c in report["cameras"]]
        rec["final_params"] = {cid: {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                                     for k, v in p.items()} for cid, p in fp.items()}
        view.close()
        return rec

    def show(r):
        if not r["ok"]:
            log(f"  {r['name']}: FAILED ({r['fail']}); pair {r['pair']}; {r['seconds']:.0f} s")
            return
        log(f"  {r['name']} (seed {r['seed']}, sizes {r['sizes']}): pair {r['pair']}, {r['seconds']:.0f} s; RMS "
            f"{r['rms_px']:.4f} px; against the truth: f " + ", ".join(f"{100 * v:+.2f} %" for v in r["f_err_rel"])
            + "; principal point " + ", ".join(f"{v:.1f} px" for v in r["pp_err_px"])
            + "; baselines " + ", ".join(f"{100 * v:+.2f} %" for v in r["baseline_err_rel"]))
        log(f"      camFiles {r['camfile_sizes']}; page {r['page_cams']}; table {r['headers']}")

    runs = {}
    for name, sizes, ids, seed, cams in (("ref", [BIG, BIG, BIG], (0, 1, 2), 1314, (0, 1, 2)),
                                         ("refS", [BIG, BIG, BIG], (0, 1, 2), 1315, (0, 1, 2)),
                                         ("ref2S", [BIG, BIG], (0, 1), 1315, (0, 1)),
                                         ("X2a", [BIG, SMALL, BIG], (0, 1, 2), 1315, (0, 1, 2)),
                                         ("X2b", [SMALL, BIG, BIG], (0, 1, 2), 1315, (0, 1, 2)),
                                         ("X2c", [BIG, SMALL], (0, 1), 1315, (0, 1)),
                                         ("X3a", [BIG, BIG, BIG], (0, 2, 3), 1314, (0, 1, 2)),
                                         ("X3b", [BIG, SMALL, BIG], (0, 2, 3), 1315, (0, 1, 2))):
        runs[name] = calibrate(name, sizes, ids, seed, cams)
        show(runs[name])

    diffs = {}
    for name, ref in (("X2a", "refS"), ("X2b", "refS"), ("X2c", "ref2S")):
        r, q = runs[name], runs[ref]
        ok, detail = r["ok"] and q["ok"], f"failed: {r['fail'] or q['fail']}"
        if ok:
            n = len(r["f"])
            df = max(abs(r["f"][k] / q["f"][k] - 1) for k in range(n))
            dpp = max(float(np.hypot(r["pp_off"][k][0] - q["pp_off"][k][0], r["pp_off"][k][1] - q["pp_off"][k][1]))
                      for k in range(n))
            db = max(abs(a / b - 1) for a, b in zip(r["baselines"], q["baselines"]))
            drot = max(abs(angle(r["R"][a] @ r["R"][b].T, q["R"][a] @ q["R"][b].T))
                       for a in range(n) for b in range(a + 1, n))
            drms = abs(r["rms_px"] - q["rms_px"])
            diffs[name] = {"f_rel": df, "pp_px": dpp, "baseline_rel": db, "rotation_rad": drot, "rms_px": drms}
            own = (r["camfile_sizes"] == r["want_sizes"] and r["proj_missing"] == 0 and r["frames"] == 300
                   and all((w, h) == tuple(int(v) for v in reversed(r["want_sizes"][int(nm[3:])].split(",")))
                           for nm, w, h in r["page_cams"]))
            ok = df <= 1e-3 and dpp <= 0.5 and db <= 1e-3 and drot <= 1e-4 and drms <= 0.01 and own
            detail = (f"pairs {r['pair']} / {q['pair']}; against {ref}: f {df:.1e}, principal point {dpp:.3f} px, "
                      f"baselines {db:.1e}, rotations {drot:.1e} rad, RMS {drms:.1e} px; own sizes, projections {own}")
        log.check(f"{name}: calibrates, with the answer of the large cameras with the same noise ({ref}); own sizes "
                  f"in the camFiles and on the page; projections in the table", ok, detail)

    for name, ref, mapping in (("X3a", "ref", {0: 0, 2: 1, 3: 2}), ("X3b", "X2a", {0: 0, 2: 1, 3: 2})):
        r, q = runs[name], runs[ref]
        ok = r["ok"] and q["ok"]
        detail = f"failed: {r['fail'] or q['fail']}"
        if ok:
            same = all(set(r["final_params"][a]) == set(q["final_params"][b]) and all(
                np.asarray(r["final_params"][a][k]).tobytes() == np.asarray(q["final_params"][b][k]).tobytes()
                for k in r["final_params"][a]) for a, b in mapping.items())
            files = all(r["camfiles"].get(a) == q["camfiles"].get(b) for a, b in mapping.items())
            heads = r["headers"] == ["Frame", "Cam 1", "Cam 3", "Cam 4", "Len Err (mm)"]
            page = [n for n, _w, _h in r["page_cams"]] == ["cam0", "cam2", "cam3"]
            sizes = r["camfile_sizes"] == r["want_sizes"]
            ok = same and files and heads and page and sizes
            detail = (f"parameters identical {same}; camFiles identical {files}; table {r['headers']}; page "
                      f"{[n for n, _w, _h in r['page_cams']]}; sizes {r['camfile_sizes']}")
        log.check(f"{name}: ids 0, 2, 3 give {ref}'s answer (ids 0, 1, 2) to the bit", ok, detail)

    keep = {k: {kk: (vv if kk != "R" else [m.tolist() for m in vv]) for kk, vv in v.items()
                if kk not in ("camfiles", "final_params")} for k, v in runs.items()}
    return log.save({"runs": keep, "x2_differences": diffs})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
