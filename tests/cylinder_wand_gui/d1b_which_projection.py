"""D1b: for G3's cam0, where the C++ camera and the Python model disagree by up to 2.75e-3 px (D1), which one is
right?  For every point: back-project each answer's pixel (cyl_model.trace, the exact closed form of the rays) and
measure how far the ray passes from the point (mm), and the C++ answer's own status; also the Python solver's
residual history for the worst point.

    python d1b_which_projection.py
"""
import common as C  # noqa: I001

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

log = C.Log("d1b_which_projection")


def ray_miss(cm, cam, tube, uv, X):
    """Distance (mm) from X to the ray of pixel uv in the water."""
    rw = cm.rows([cam], np.zeros(len(uv), int))
    o, d, ok, _a, _b = cm.trace(rw, cm.undistort(uv, rw), tube)
    v = X - o
    return np.where(ok, np.linalg.norm(v - np.sum(v * d, axis=1)[:, None] * d, axis=1), np.nan)


def main():
    C.qt_app()
    from modules.camera_calibration.curved_calibration import cyl_model as cm
    res_dir = C.DATA / "g3_run" / "results"
    res = json.load(open(res_dir / "calibration_result.json", encoding="utf-8"))
    data = json.load(open(res_dir / "calibration_data.json", encoding="utf-8"))
    tube = cm.Tube(res["tube"]["radii_mm"], res["tube"]["n"])
    X = np.vstack([np.array(data["ends_A_mm"]), np.array(data["ends_B_mm"])])
    out = {}
    for k, c in enumerate(res["cameras"]):
        cam = cm.Camera(np.array(c["R"]), np.array(c["centre_mm"]), c["f_px"], c["cx"], c["cy"], c["width"],
                        c["height"], np.array(c["dist"]))
        cc = cm.to_cpp(cam, tube)
        q = cm.cpp_project(cc, X)
        rw = cm.rows([cam], np.zeros(len(X), int))
        start = np.where(np.isfinite(q), q, [cam.cx, cam.cy])
        pr = cm.project(rw, X, cm.undistort(start, rw), tube)
        # Python from a neutral start too (the principal point), as the calibration starts from the detection
        pr0 = cm.project(rw, X, cm.undistort(np.tile([cam.cx, cam.cy], (len(X), 1)), rw), tube)
        diff = np.hypot(*(pr.uv - q).T)
        diff0 = np.hypot(*(pr0.uv - q).T)
        miss_cpp = ray_miss(cm, cam, tube, q, X)
        miss_py = ray_miss(cm, cam, tube, pr.uv, X)
        bad = np.flatnonzero(diff > 1e-6)
        log(f"cam{k}: {len(X)} points; Python vs C++ above 1e-6 px at {len(bad)} points (largest {np.nanmax(diff):.2e}, "
            f"from the principal point {np.nanmax(diff0):.2e}); ray misses: C++ median {np.nanmedian(miss_cpp):.1e} "
            f"max {np.nanmax(miss_cpp):.1e} mm, Python median {np.nanmedian(miss_py):.1e} max {np.nanmax(miss_py):.1e} mm")
        if len(bad):
            for i in bad[np.argsort(-diff[bad])][:5]:
                rho = float(np.hypot(X[i, 0], X[i, 2]))
                imgs = cm.cpp_images(cc, X[i:i + 1])[0]
                log(f"    point {i}: X ({X[i, 0]:.3f}, {X[i, 1]:.3f}, {X[i, 2]:.3f}) mm, {rho:.2f} mm from the axis; "
                    f"C++ ({q[i, 0]:.5f}, {q[i, 1]:.5f}) misses by {miss_cpp[i]:.2e} mm; Python ({pr.uv[i, 0]:.5f}, "
                    f"{pr.uv[i, 1]:.5f}) misses by {miss_py[i]:.2e} mm; ok {bool(pr.ok[i])}; all C++ images "
                    f"{np.round(imgs, 4).tolist()}")
        out[c["name"]] = {"n_diff": int(len(bad)), "max_diff_px": float(np.nanmax(diff)),
                          "cpp_miss_max_mm": float(np.nanmax(miss_cpp)), "py_miss_max_mm": float(np.nanmax(miss_py))}
    return log.save({"cameras": out})


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
