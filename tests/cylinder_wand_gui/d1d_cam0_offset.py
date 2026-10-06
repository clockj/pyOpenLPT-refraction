"""D1d: G3's cam0 projects 2.75e-3 px off in u in C++ (D1b), constant over all points.  Which camera parameter
differs between the C++ camera and the Python model?  Element by element: the camFile's Rotation Matrix and Rotation
Vector against the fitted R; then the Python projection with R taken from each of them, against the C++ projection.
(D1c's angles came from acos(trace) and cannot resolve below about 1e-7 rad; here the angles are |log(Ra^T Rb)|.)

    python d1d_cam0_offset.py
"""
import common as C  # noqa: I001

import json  # noqa: E402
import sys  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

log = C.Log("d1d_cam0_offset")


def section(txt, head, rows):
    lines = txt.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# " + head))
    return np.array([[float(v) for v in lines[i + 1 + r].split(",")] for r in range(rows)])


def main():
    C.qt_app()
    import pyopenlpt as lpt  # noqa: F401
    from modules.camera_calibration.curved_calibration import cyl_model as cm
    res_dir = C.DATA / "g3_run" / "results"
    res = json.load(open(res_dir / "calibration_result.json", encoding="utf-8"))
    data = json.load(open(res_dir / "calibration_data.json", encoding="utf-8"))
    tube = cm.Tube(res["tube"]["radii_mm"], res["tube"]["n"])
    X = np.vstack([np.array(data["ends_A_mm"])[:60], np.array(data["ends_B_mm"])[:60]])
    out = {}
    for k, c in enumerate(res["cameras"]):
        R = np.array(c["R"])
        cam = cm.Camera(R, np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"], c["height"],
                        np.array(c["dist"]))
        txt = open(res_dir / "camFile" / f"cam{k}.txt", encoding="utf-8").read()
        R_file = section(txt, "Rotation Matrix", 3)
        w_file = section(txt, "Rotation Vector", 1)[0]
        K_file = section(txt, "Camera Matrix", 3)
        t_file = section(txt, "Translation Vector", 1)[0]
        q_cpp = cm.cpp_project(lpt.Camera(str(res_dir / "camFile" / f"cam{k}.txt")), X)
        rows = []
        for label, Rv in (("fitted R", R), ("camFile matrix", R_file), ("exp(file rvec)", cm.rot_of(w_file)),
                          ("cv2(file rvec)", cv2.Rodrigues(w_file.reshape(3, 1))[0])):
            k2 = cam.copy()
            k2.R = Rv
            rw = cm.rows([k2], np.zeros(len(X), int))
            pr = cm.project(rw, X, cm.undistort(np.where(np.isfinite(q_cpp), q_cpp, [cam.cx, cam.cy]), rw), tube)
            d = pr.uv - q_cpp
            ang = float(np.linalg.norm(cm.log_so3(R.T @ Rv)))
            rows.append((label, ang, float(np.nanmax(np.abs(d[:, 0]))), float(np.nanmax(np.abs(d[:, 1])))))
        log(f"{c['name']}: |R_file - R| {np.abs(R_file - R).max():.1e}; |t_file - (-R C)| "
            f"{np.abs(t_file + R @ cam.C).max():.1e} mm; K_file f {K_file[0, 0]:.9f} vs {cam.f:.9f}, cx {K_file[0, 2]:.6f} "
            f"vs {cam.cx:.6f}; |file rvec - rvec_of(R)| {np.linalg.norm(w_file - cm.rvec_of(R)):.1e}")
        for label, ang, du, dv in rows:
            log(f"    Python with {label:15s} ({ang:.1e} rad from the fitted R): |du| {du:.2e}, |dv| {dv:.2e} px "
                "against C++")
        out[c["name"]] = {"rows": rows}
    return log.save({"cameras": out})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
