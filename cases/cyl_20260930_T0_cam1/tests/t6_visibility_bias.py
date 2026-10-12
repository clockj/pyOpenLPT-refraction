"""T6: visibility, save/reload, and the dot-centre bias of the final model.

  * projectAll on every calibration point: exactly one image each (no shadow, no double images);
  * the saved camFile/cam1.txt is reloaded with pyOpenLPT's loader and must reproduce the projections (1e-6 px);
  * centre bias: each dot's circular outline (diameter d) is projected through the model, an ellipse is fitted to
    it as the detector does, and its centre is compared with the projection of the dot's centre.  d is the board's
    nominal 1.5 mm (user, 2026-10-01); the diameter of the detected half-height outlines is also measured against
    the model (horizontal and vertical separately, which also checks the model's anisotropic magnification), and
    the bias is reported for both;
  * passes if the bias is below 0.02 px, or it is reported for correction.
Writes t6_out.txt beside this script.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402
import dots as D                              # noqa: E402
import run_calibration as RC                  # noqa: E402

LOG = []
lpt = CF.lpt


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def outline_points(P, radius, n=96):
    """Points of a circle of `radius` around P in the board plane (X, Y at fixed Z)."""
    t = np.arange(n) * 2 * np.pi / n
    return np.column_stack([P[0] + radius * np.cos(t), P[1] + radius * np.sin(t), np.full(n, P[2])])


def main():
    cfg = case.load_config()
    out = cfg["out_dir"]
    state = RC.load_state(os.path.join(out, "fit_state.pkl"))
    rig, obs = state["rig"], state["obs"]
    cam = rig.camera()
    P = rig.world_points(obs)
    say(f"T6 visibility, reload and centre bias; model {state['info']['model']}, {len(P)} calibration points")

    # 1. every image of every point
    allimg = CF.project_all(cam, P)
    counts = np.array([len(a) for a in allimg])
    say(f"  images per point: " + ", ".join(f"{c}: {int(np.sum(counts == c))}" for c in sorted(set(counts))))
    uv1 = CF.project(cam, P)
    on = np.array([np.all((a[:, 0] >= 0) & (a[:, 0] < rig.width) & (a[:, 1] >= 0) & (a[:, 1] < rig.height))
                   if len(a) else False for a in allimg])
    vis_ok = bool(np.all(counts == 1))
    # 2. save / reload
    cam_file = os.path.join(out, "camFile", f"cam{cfg['camera_id']}.txt")
    cam2 = lpt.loadCameraModel(cam_file)
    uv2 = CF.project(cam2, P)
    d_reload = float(np.nanmax(np.abs(uv2 - uv1)))
    say(f"  reload of {cam_file}: type {cam2.type()}, max |projection difference| {d_reload:.2e} px")
    reload_ok = d_reload < 1e-6
    # 3. dot diameter and centre bias
    planes = RC.collect(cfg)
    keymap = {(k, int(planes[k]['ij'][n][0]), int(planes[k]['ij'][n][1])): planes[k]["dots"][n]
              for k in range(len(planes)) for n in range(len(planes[k]["dots"]))}
    a_det, b_det, a_mod, b_mod = [], [], [], []
    r_try = 0.7
    for n in range(len(P)):
        d = keymap.get((int(obs.plane_ids[obs.plane[n]]) if obs.plane_ids is not None else int(obs.plane[n]),
                        int(obs.i[n]), int(obs.j[n])))
        if d is None or not np.isfinite(d["a"]):
            continue
        pts = CF.project(cam, outline_points(P[n], r_try))
        ell = D.fit_ellipse(pts[:, 0], pts[:, 1])
        if ell is None:
            continue
        # compare axis-aligned extents (the dots' ellipses are near axis-aligned)
        ct, st = np.cos(ell[4]), np.sin(ell[4])
        a_mod.append(np.hypot(ell[2] * ct, ell[3] * st))
        b_mod.append(np.hypot(ell[2] * st, ell[3] * ct))
        ct, st = np.cos(d["theta"]), np.sin(d["theta"])
        a_det.append(np.hypot(d["a"] * ct, d["b"] * st))
        b_det.append(np.hypot(d["a"] * st, d["b"] * ct))
    a_det, b_det, a_mod, b_mod = map(np.array, (a_det, b_det, a_mod, b_mod))
    d_h = 2 * r_try * np.median(a_det / a_mod)
    d_v = 2 * r_try * np.median(b_det / b_mod)
    d_nom = float(cfg["target"]["dot_diameter_mm"])
    say(f"  dot diameter from the detected half-height outlines: horizontal {d_h:.3f} mm, vertical {d_v:.3f} mm "
        f"(nominal {d_nom} mm); per-plane ratio spread (std of det/model) h {np.std(a_det / a_mod):.4f}, "
        f"v {np.std(b_det / b_mod):.4f}")
    bmax = 0.0
    for d_try in (d_nom, 0.5 * (d_h + d_v)):
        radius = 0.5 * d_try
        bias = []
        for n in range(len(P)):
            pts = CF.project(cam, outline_points(P[n], radius))
            ell = D.fit_ellipse(pts[:, 0], pts[:, 1])
            if ell is None or not np.all(np.isfinite(uv1[n])):
                continue
            bias.append((ell[0] - uv1[n][0], ell[1] - uv1[n][1]))
        bias = np.array(bias)
        bm = float(np.max(np.linalg.norm(bias, axis=1)))
        bmax = max(bmax, bm)
        say(f"  centre bias (outline ellipse centre minus projected centre), d = {d_try:.3f} mm: mean "
            f"({bias[:, 0].mean():+.4f}, {bias[:, 1].mean():+.4f}) px, rms "
            f"{np.sqrt(np.mean(np.sum(bias ** 2, 1))):.4f} px, max {bm:.4f} px")
    bias_ok = bmax < 0.02
    say("RESULT: " + ("PASS" if vis_ok and reload_ok and bias_ok else
                      f"FAIL (visibility {'ok' if vis_ok else 'fail'}, reload {'ok' if reload_ok else 'fail'}, "
                      f"bias {'ok' if bias_ok else 'over 0.02 px'})"))
    with open(os.path.join(HERE, "t6_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
