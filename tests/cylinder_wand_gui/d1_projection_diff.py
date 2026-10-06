"""D1 (diagnosis after G3): where do G3's 3.7e-3 px between the camFiles' C++ projections and the overlay's crosses
come from, and why were frames 47, 136 and 170 not kept?

    python d1_projection_diff.py

Part 1, the projections, using G3's results (data/g3_run/results):
  (a) the Python model (cyl_model.project) with the cameras of calibration_result.json, from the exact ends
      (cylinder_wand.relabel of a refit... not available) -> from the rounded ends of calibration_data.json;
  (b) the C++ camera loaded from each camFile, same ends;
  (c) the same with the ends un-rounded: the ends are rebuilt from the cameras by re-running the GUI's job from the
      saved detections is not possible either, so instead the cameras of the JSON are turned into C++ cameras with
      cyl_model.to_cpp (no file) and compared with the camFile's C++ camera on the same points.
  Then the size of the rounding of the ends (5 decimals) is measured by perturbing them by 5e-6 mm.
Part 2, the three frames: what each camera's "ellipse" detection returns (full rows: dropped by the overlap guard,
  border, a third blob) for frames 47, 136 and 170.
Writes d1_projection_diff_out.txt and d1_projection_diff.json.
"""
import common as C  # noqa: I001

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("d1_projection_diff")


def main():
    C.qt_app()
    import pyopenlpt as lpt
    from modules.camera_calibration.curved_calibration import cyl_model as cm
    from modules.camera_calibration.wand_calibration import wand_calibrator as wc
    res_dir = C.DATA / "g3_run" / "results"
    res = json.load(open(res_dir / "calibration_result.json", encoding="utf-8"))
    data = json.load(open(res_dir / "calibration_data.json", encoding="utf-8"))
    tube = cm.Tube(res["tube"]["radii_mm"], res["tube"]["n"])
    cams = [cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"],
                      c["height"], np.array(c["dist"])) for c in res["cameras"]]
    XA = np.array(data["ends_A_mm"])
    XB = np.array(data["ends_B_mm"])
    X = np.vstack([XA[:40], XB[:40]])
    out = {}
    for k, cam in enumerate(cams):
        file_cam = lpt.Camera(str(res_dir / "camFile" / f"cam{k}.txt"))
        mem_cam = cm.to_cpp(cam, tube)
        q_file = cm.cpp_project(file_cam, X)
        q_mem = cm.cpp_project(mem_cam, X)
        rw = cm.rows([cam], np.zeros(len(X), int))
        pr = cm.project(rw, X, cm.undistort(np.where(np.isfinite(q_mem), q_mem, [cam.cx, cam.cy]), rw), tube)
        d_file_mem = np.nanmax(np.hypot(*(q_file - q_mem).T))
        d_py_mem = np.nanmax(np.hypot(*(pr.uv - q_mem).T))
        # the rounding of the ends: 5 decimals, so up to 5e-6 mm per coordinate
        Xp = X + np.random.default_rng(k).uniform(-5e-6, 5e-6, X.shape)
        d_round = np.nanmax(np.hypot(*(cm.cpp_project(mem_cam, Xp) - q_mem).T))
        log(f"cam{k}: C++ from the camFile vs C++ from the JSON's camera: {d_file_mem:.2e} px; Python vs C++: "
            f"{d_py_mem:.2e} px; ends moved by up to 5e-6 mm: {d_round:.2e} px")
        out[f"cam{k}"] = {"file_vs_json_cpp_px": float(d_file_mem), "python_vs_cpp_px": float(d_py_mem),
                          "rounding_px": float(d_round)}
        with open(res_dir / "camFile" / f"cam{k}.txt", encoding="utf-8") as fh:
            txt = fh.read()
        if k == 0:
            values = [ln for ln in txt.split("\n") if ln and not ln.startswith("#")]
            log("  camFile cam0, its last two values (tolerance, iterations): " + " | ".join(values[-2:]))
    # the GUI's overlay crosses are uv + r of the fit in memory; calibration_data.json has u, v, du, dv rounded
    obs = data["observations"]
    fid0 = data["frame_ids"][0]
    rows = [i for i, f in enumerate(obs["frame_id"]) if f == fid0]
    log(f"frame {fid0}: rows {rows}; u, v, du, dv as saved: "
        + "; ".join(f"cam{obs['cam_id'][i]} end {obs['end'][i]}: ({obs['u'][i]}, {obs['v'][i]}) + ({obs['du'][i]}, "
                    f"{obs['dv'][i]})" for i in rows))

    # part 2: the three frames
    folder = C.DATA / "g3_backlit"
    lists = C.image_lists(folder)
    truth = C.load_truth(folder)
    opts = {"open_px": 25, "rod_mask_px": 18.0, "guard_px": 3.0, "full": True}
    for f in (47, 136, 170):
        for c in range(3):
            _a, _b, rows_full = wc.run_detection_task((f, c, lists[c][f], "dark", 10.0, 60.0, 0.85, "ellipse", opts))
            tA, tB = truth["UV"][f, c]
            desc = "none" if rows_full is None else "; ".join(
                f"({r[0]:.2f}, {r[1]:.2f}) r {r[2]:.1f} dropped {int(r[7])}" for r in rows_full)
            log(f"frame {f} cam{c}: truth A ({tA[0]:.2f}, {tA[1]:.2f}) B ({tB[0]:.2f}, {tB[1]:.2f}); detected: {desc}")
    return log.save({"projections": out})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
