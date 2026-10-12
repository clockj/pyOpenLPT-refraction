"""D7 (diagnosis): how far fits that explain the dots (nearly) as well disagree on points the calibration did not see.

    python tests/d7_prediction_spread.py

The delivered camera (fit_state.pkl in the processed folder) is compared with refits, all with the planes at their
stage positions unless stated:
  near-equal fits (the realistic spread):
  * n_water = 1.3376, the index the data prefer (T7), instead of 1.333;
  * the stage steps scaled by a free factor (the data prefer about 1.011, T5);
  * M1b (no plane shifts) and M2b (every plane's position free, the first run's model);
  sensitivity probes (fits that are clearly worse or not comparable):
  * the axis held 5 mm to either side (X_a -+ 5 mm): costs 0.04-0.07 px of fit;
  * the delivered model + the diagnostic wall field, its PINCYL part alone and with the field.  The field's spline
    (knots every 80 px) is not constrained between the dot columns (about 150 px apart), so its predictions there
    are not a measure of accuracy;
  * the traverse tilt: the data cannot tell a board moving along its normal from one moving about 6 deg off it.
    That fit puts the world frame elsewhere (the dots of plane k are shifted sideways by tau z_k), so its pixel
    differences at the same (X, Y, z) measure the frame, not the camera; the angle between the rays is reported.
Each projects two 3-D grids, and the pixel differences from the delivered camera are reported, with the angle
between their rays in the water through the image centre:
  * the calibrated volume: X 0..24 mm, Y -9..6 mm, z 0..48.26 mm;
  * the depth of the tube in view: the same X, Y, and z from -80 to +100 mm (plane 0 is about 13 mm behind the axis,
    the inner wall 97.6 mm from it), points without an image left out.
Writes d7_out.txt beside this script.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402
import run_calibration as RC                  # noqa: E402

LOG = []
NW_PREFERRED = 1.3376


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def _job(args):
    """('fit', rig, obs, blocks) refits; ('axis', rig, obs, blocks, xa) refits with the axis's lateral position held
    at xa (its depth, its direction and the rest free)."""
    kind, rig, obs, blocks = args[:4]
    if kind == "fit":
        return CF.fit(CF.restrict(rig, blocks), obs, blocks)
    xa = args[4]
    r = rig.copy()
    a0 = r.axis_at_y0()
    dx = xa - a0[0]
    r.axis_pt = a0 + np.array([dx, 0.0, 0.0])
    r.C = r.C + np.array([dx, 0.0, 0.0])
    b = tuple(x if x != "axpt" else "axptz" for x in blocks)
    fb = CF.fit(r, obs, b, residual="backproject", max_outer=4, max_nfev=400)
    return CF.fit(fb.rig, obs, b)


def spread(ref, rig, G):
    uv_ref = CF.project(ref.camera(), G)
    uv = CF.project_fast(rig, G, uv_ref) if rig.has_wall() else CF.project(rig.camera(), G)
    m = np.all(np.isfinite(uv_ref), axis=1) & np.all(np.isfinite(uv), axis=1)
    d = np.linalg.norm(uv[m] - uv_ref[m], axis=1)
    return float(np.sqrt(np.mean(d ** 2))), float(d.max()), int(m.sum())


def main():
    cfg = case.load_config()
    st = RC.load_state(os.path.join(cfg["out_dir"], "fit_state.pkl"))
    ref, obs, blocks, model = st["rig"], st["obs"], tuple(st["blocks"]), st["info"]["model"]
    xa0 = float(ref.axis_at_y0()[0])
    other = "M1b" if model == "M1bx" else "M1bx"
    nw = ref.copy()
    nw.n = [ref.n[0], ref.n[1], NW_PREFERRED]
    jobs = [("fit", nw, obs, blocks),
            ("fit", ref, obs, blocks + ("zs",)),
            ("fit", ref, obs, CF.MODELS[other]),
            ("fit", ref, obs, CF.MODELS["M2b"]),
            ("axis", ref, obs, blocks, xa0 - 5.0),
            ("axis", ref, obs, blocks, xa0 + 5.0),
            ("fit", ref, obs, blocks + ("wall",)),
            ("fit", ref, obs, blocks + ("tau",))]
    fits = RC.parallel_map(_job, jobs)
    pin = fits[6].rig.copy()
    pin.wall = None
    rows = [("near-equal", f"n_water = {NW_PREFERRED}", fits[0].rig, fits[0].rms),
            ("near-equal", f"stage scale free ({fits[1].rig.zscale:.4f})", fits[1].rig, fits[1].rms),
            ("near-equal", other, fits[2].rig, fits[2].rms),
            ("near-equal", "M2b (planes free)", fits[3].rig, fits[3].rms),
            ("probe", f"axis held at X_a = {xa0 - 5:.1f} mm", fits[4].rig, fits[4].rms),
            ("probe", f"axis held at X_a = {xa0 + 5:.1f} mm", fits[5].rig, fits[5].rms),
            ("probe", "+ wall field, PINCYL part only", pin, float("nan")),
            ("probe", "+ wall field, with the field", fits[6].rig, fits[6].rms),
            ("frame", "+ traverse tilt", fits[7].rig, fits[7].rms)]

    xs = np.arange(0.0, 24.01, 2.0)
    ys = np.arange(-9.0, 6.01, 1.5)
    G1 = np.array([(x, y, z) for z in np.linspace(0.0, 48.26, 13) for x in xs for y in ys], float)
    G2 = np.array([(x, y, z) for z in np.arange(-80.0, 100.01, 10.0) for x in xs for y in ys], float)
    centre = np.array([[ref.cx, ref.cy]])
    _o0, d0, _ = CF.trace(ref, centre)
    sref = ref.summary()
    say(f"D7 spread of the predictions; reference: the delivered camera ({model}, rms {st['rms']:.4f} px), camera "
        f"{np.round(sref['camera_centre_mm'], 1)} mm, f {ref.f:.0f} px, axis X,Z "
        f"{np.round(np.array(sref['axis_point_at_Y0_mm'])[[0, 2]], 2)} mm, view {sref['view_tilt_from_minusZ_deg']:.2f} "
        f"deg from -Z, lean {sref['axis_lean_towards_camera_deg']:.2f} deg")
    say("  grids: calibrated volume (X 0..24, Y -9..6, z 0..48.26 mm) and the tube depth in view (same X, Y; "
        "z -80..100 mm); differences from the reference in px")
    for kind, label, rig, rms in rows:
        a = spread(ref, rig, G1)
        b = spread(ref, rig, G2)
        _o1, d1, _ = CF.trace(rig, centre)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(d0[0] @ d1[0])))))
        s = rig.summary()
        say(f"  [{kind:10s}] {label:32s} fit rms {rms:.4f} | calibrated volume: rms {a[0]:.3f}, max {a[1]:.3f} "
            f"({a[2]} pts) | tube depth: rms {b[0]:.3f}, max {b[1]:.3f} ({b[2]} pts) | ray at the image centre "
            f"turned {ang:.3f} deg; f {rig.f:.0f} px, camera {np.round(s['camera_centre_mm'], 1)} mm, axis X "
            f"{s['axis_point_at_Y0_mm'][0]:.2f} mm" + (f", tau {np.round(rig.tau, 4)}" if np.any(rig.tau != 0) else ""))
    with open(os.path.join(HERE, "d7_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
