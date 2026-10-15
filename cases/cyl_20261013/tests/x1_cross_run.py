"""X1 (cross-run test, as 6 Oct T2's X1): did the cameras move between 8 Oct and 13 Oct?

    python tests/x1_cross_run.py        (after calibrate.py and c4_track_camfiles.py)

The plan (section 1) read the balls' 5-8 % larger images as a camera move, but C1's poses and f / distance came out
close to 8 Oct's.  This puts both calibrations to today's frames:
  - today's cameras (C1, <out_dir>/camFile_track) and 8 Oct's (X:\\...\\Processed\\20261008\\T0\\camFile_track);
  - the frames all three cameras keep under 8 Oct's selection rule (1.45, as V1);
  - per camera: the residual (projection of the ball triangulated from all three cameras - detection): RMS and
    mean (du, dv).  A camera that moved since 8 Oct shows as a mean offset with 8 Oct's cameras;
  - each camera predicted from the other two (V1's held_out).
Library build p4c, as C1-H.  Writes x1_cross_run_out.txt and x1_cross_run.json beside this script.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import case  # noqa: E402
import v1_variants as v1  # noqa: E402

import numpy as np  # noqa: E402

OCT8 = r"X:\Shijie Zhong\Refraction\Processed\20261008\T0\camFile_track"


def all_three(lpt, files, kept, frames, ids):
    """Each ball triangulated from all three cameras' lines of sight; per camera the residual projection -
    detection: RMS, mean (du, dv) with its standard error."""
    cams = {c: lpt.loadCameraModel(files[c]) for c in ids}
    rays = {c: cams[c].lineOfSightBatchStatus([lpt.Pt2D(float(kept[(f, c)][e][0]), float(kept[(f, c)][e][1]))
                                                for f in frames for e in (0, 1)]) for c in ids}
    X, ok, tri = [], [], []
    for i in range(2 * len(frames)):
        rs = [rays[c][i] for c in ids]
        if not all(r[0] for r in rs):
            X.append(lpt.Pt3D(0.0, 0.0, 0.0))
            ok.append(False)
            tri.append(np.nan)
            continue
        p, e = lpt.myMATH.triangulation([r[1] for r in rs])
        X.append(p)
        ok.append(True)
        tri.append(e)
    out = {"triangulation_error_um_median": float(1e3 * np.nanmedian(tri))}
    for c in ids:
        proj = cams[c].projectBatchStatus(X)
        d = []
        for i, (f, e) in enumerate((f, e) for f in frames for e in (0, 1)):
            if ok[i] and proj[i][0]:
                q = proj[i][1]
                d.append((q[0] - kept[(f, c)][e][0], q[1] - kept[(f, c)][e][1]))
        d = np.array(d)
        out[f"cam{c}"] = {"n": int(len(d)), "rms_px": float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))),
                          "mean_du": float(d[:, 0].mean()), "mean_dv": float(d[:, 1].mean()),
                          "se": float(np.sqrt((d[:, 0].var() + d[:, 1].var()) / 2 / len(d)))}
    return out


def main():
    cfg = case.load_config()
    log = case.Log(os.path.join(HERE, "x1_cross_run"))
    sys.path.insert(0, v1.BUILD)
    import pyopenlpt as lpt  # noqa: E402
    ids = case.cam_ids(cfg)
    _d, kept, _r, _m, _f = case.load_selection(v1.variant_cfg("a145"))
    frames = case.frames_in_all(kept, ids)
    sets = {"13 Oct (C1)": os.path.join(cfg["out_dir"], cfg["tracking_camfiles"]["folder"]), "8 Oct": OCT8}
    log(f"X1: today's {len(frames)} frames kept by all three cameras (1.45 rule) through both calibrations; library "
        f"{lpt.__file__}")
    out = {"frames": len(frames)}
    for name, folder in sets.items():
        files = {c: os.path.join(folder, f"cam{c}.txt") for c in ids}
        a3 = all_three(lpt, files, kept, frames, ids)
        ho = v1.held_out(lpt, files, kept, frames, ids)
        out[name] = {"folder": folder, "all_three": a3, "held_out": ho}
        log(f"  {name} ({folder}): triangulation error median {a3['triangulation_error_um_median']:.1f} um")
        for c in ids:
            r, h = a3[f"cam{c}"], ho[f"cam{c}"]
            log(f"    cam{c}: from all three: RMS {r['rms_px']:.3f} px, mean (du, dv) ({r['mean_du']:+.3f}, "
                f"{r['mean_dv']:+.3f}) px (SE {r['se']:.3f}); predicted from the other two: RMS {h['rms_px']:.3f} px, "
                f"median {h['median_px']:.3f}, 95 % {h['p95_px']:.3f}, failed {h['failed']}")
    log.save(out)


if __name__ == "__main__":
    main()
