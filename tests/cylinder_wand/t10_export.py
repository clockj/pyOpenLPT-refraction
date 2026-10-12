"""T10: export.  A calibration from scratch (reference rig, 300 frames, 0.1 px) is written by
`cylinder_wand_io.write_result` (camFiles through the C++ writer, result JSON, text report) into t10_export/; every
camFile is loaded back with `pyopenlpt.Camera(path)` and must project a grid of points over the volume exactly as
the calibrated camera does (the in-memory C++ model of the written parameters): at most 1e-9 px (plan section 7).
The fast model is compared too (1e-6 px), and the lines of sight of the reloaded files against the in-memory ones.
Outputs: t10_export_out.txt, t10_export/ (the written files).
"""
import json
import os

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw
import cylinder_wand_io as cio

OUT = rigs.HERE / "t10_export"
log = rigs.Tee(str(OUT) + "_out.txt")


def main():
    lpt = cm.import_lpt()
    tube = rigs.CASE_TUBE
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    rng = np.random.default_rng(1010)
    wd = rigs.wand_frames(truth, tube, 300, 10.0, rng, half=(20, 20, 20), noise_px=0.1)
    data = cw.WandData(wd.uv, 10.0, [(c.width, c.height) for c in truth])
    sol = cw.calibrate(data, tube)
    folder = str(OUT)
    res = cio.write_result(sol, folder, names=["cam1", "cam2", "cam3"], pixel_mm=[0.02] * 3)
    cams_out, _fr, y0 = cw.relabel(sol.cams, sol.frames)
    g = np.linspace(-20, 20, 9)
    P = np.array([[x, y - y0, z] for x in g for y in g for z in g])      # the volume, in the written frame
    worst_file, worst_fast, worst_ray = 0.0, 0.0, 0.0
    for c, cam in enumerate(cams_out):
        path = os.path.join(folder, res["cameras"][c]["camfile"])
        loaded = lpt.Camera(path)
        mem = cm.to_cpp(cam, tube)
        a = cm.cpp_project(loaded, P)
        b = cm.cpp_project(mem, P)
        ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
        d_file = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else float("nan")
        rw = cm.rows([cam], np.zeros(len(P), int))
        pr = cm.project(rw, P, cm.undistort(b, rw), tube)
        okf = ok & pr.ok
        d_fast = float(np.max(np.abs(pr.uv[okf] - b[okf])))
        o1, d1, k1 = cm.cpp_rays(loaded, b[ok])
        o2, d2, k2 = cm.cpp_rays(mem, b[ok])
        d_ray = float(np.max(np.linalg.norm(o1 - o2, axis=1) + 100 * np.linalg.norm(d1 - d2, axis=1)))
        worst_file, worst_fast, worst_ray = max(worst_file, d_file), max(worst_fast, d_fast), max(worst_ray, d_ray)
        with open(path, "r", encoding="utf-8") as fh:
            head = fh.read().split("\n")[:7]
        log(f"{res['cameras'][c]['name']}: {ok.sum()} points | reloaded file vs calibrated camera {d_file:.1e} px, "
            f"fast model {d_fast:.1e} px, lines of sight {d_ray:.1e} mm | header: {' | '.join(head[1:6])}")
    log("")
    log(open(os.path.join(folder, "calibration_report.txt"), encoding="utf-8").read())
    ok_all = worst_file <= 1e-9 and worst_fast <= 1e-6 and worst_ray <= 1e-9
    log(f"  {'PASS' if ok_all else 'FAIL'}  camFiles reload exactly: {worst_file:.1e} px (limit 1e-9), fast model "
        f"{worst_fast:.1e} px (limit 1e-6), rays {worst_ray:.1e} mm")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump({"reload_px": worst_file, "fast_px": worst_fast, "ray_mm": worst_ray, "pass": bool(ok_all)}, fh,
                  indent=1)
    log.close()


if __name__ == "__main__":
    main()
