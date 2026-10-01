"""D6 (diagnosis): is the tube axis's lateral position determined by the data?

    python tests/d6_lateral_profile.py STATE.pkl MODEL [tag]

Profile of the fit over the axis's lateral position X_a (where the axis crosses Y = 0): for each X_a on a grid the
model is refitted with X_a held (everything else free, the axis depth included) and the RMS is recorded.  A sharp
minimum means the data fix X_a; a flat profile means X_a trades with the camera's lateral position and the
traverse direction (to first order the contraction about the axis and a drift of the board relative to the
camera-axis plane look alike).  Also records, for each fit, the camera centre, the viewing direction, the axis
lean and the fitted travel.  Writes d6_<tag>_out.txt beside this script.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import cylfit as CF                           # noqa: E402
import run_calibration as RC                  # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def _job(args):
    rig, obs, blocks, xa = args
    r = rig.copy()
    # move the axis laterally to X_a (at Y = 0), and the camera by the same amount to start from a similar view
    a0 = r.axis_at_y0()
    dx = xa - a0[0]
    r.axis_pt = a0 + np.array([dx, 0.0, 0.0])
    r.C = r.C + np.array([dx, 0.0, 0.0])
    # the back-projection residual is defined everywhere, so it brings a distant start in; forward then refines
    fb = CF.fit(r, obs, blocks, residual="backproject", max_outer=4, max_nfev=400)
    fr = CF.fit(fb.rig, obs, blocks)
    return xa, fr


def main():
    st = RC.load_state(sys.argv[1])
    model = sys.argv[2]
    tag = sys.argv[3] if len(sys.argv) > 3 else model
    rig, obs = st["rig"], st["obs"]
    blocks = tuple(b if b != "axpt" else "axptz" for b in CF.MODELS[model])
    a0 = rig.axis_at_y0()
    grid = list(np.round(np.arange(-10.0, 40.1, 5.0), 3)) + [round(float(a0[0]), 3)]
    t0 = time.time()
    res = RC.parallel_map(_job, [(rig, obs, blocks, xa) for xa in grid])
    res.sort(key=lambda t: t[0])
    say(f"D6 profile over the axis's lateral position, model {model} (from {sys.argv[1]}; fitted X_a = "
        f"{a0[0]:.3f} mm); {time.time() - t0:.0f} s")
    say("   X_a (mm)   rms (px)   camera centre (mm)            view from -Z (deg)  axis lean (deg)  travel (mm)  f (px)")
    for xa, fr in res:
        s = fr.rig.summary()
        z = fr.rig.plane_z(obs.z_nom)
        say(f"   {xa:7.2f}   {fr.rms:8.4f}   {np.round(s['camera_centre_mm'], 1)!s:30s}  "
            f"{s['view_tilt_from_minusZ_deg']:8.3f}           {s['axis_lean_towards_camera_deg']:7.3f}        "
            f"{z[-1] - z[0]:7.2f}    {s['f_px']:8.0f}")
    with open(os.path.join(HERE, f"d6_{tag}_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
