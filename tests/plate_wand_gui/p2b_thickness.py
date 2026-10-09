"""P2 (b): what a window's thickness changes in the fit, after P2's case A failed its second criterion (a wrong
thickness did not fit worse: the same RMS and f).

    python p2b_thickness.py

P2's rig B' (window 1 truly 10 mm) and wand (200 frames, seed 2610), set in the view as P2 sets them, with window 1 at
10 mm (right) and at 12.7 mm (2.7 mm too thick). For each: the RMS, the largest f error, and per camera behind
window 1 the fitted distance from its centre to the window's camera-side surface and to its water-side surface, against
the truth (frame-free quantities). The question: does a flat plate's position absorb a thickness error (same RMS),
how far does the window move, and do the wand ends in the water move (the output frame)?
Writes p2b_thickness_out.txt and .json here.
"""
import pgcommon as P  # noqa: I001

import numpy as np  # noqa: E402

log = P.Log("p2b_thickness")


def main():
    P.qt_app()
    P.Dialogs(folder=str(P.DATA / "p2b_unused"))
    import prigs
    from modules.camera_calibration.curved_calibration import plate_wand_gui as pg
    rig = prigs.rig_B()
    rig.wins[1] = prigs.wall([1, 0, 0], 0.0, t=(10.0,), name="wall X")
    w = prigs.make_wand(rig, 200, np.random.default_rng(2610), noise_px=0.1)
    wp = {}
    for i in range(w.uv.shape[0]):
        fr = {c: [[*w.uv[i, c, 0], 20.0, 0.9, "Filtered_Small", 0], [*w.uv[i, c, 1], 30.0, 0.9, "Filtered_Large", 1]]
              for c in range(rig.n_cams) if np.all(np.isfinite(w.uv[i, c]))}
        if len(fr) >= 2:
            wp[i] = fr
    truth = {c: (rig.wins[rig.cam_win[c]].distance(rig.cams[c].C),
                 rig.wins[rig.cam_win[c]].distance(rig.cams[c].C) + rig.wins[rig.cam_win[c]].T) for c in (2, 3)}
    log(f"truth, cameras 2 and 3 behind window 1 (10 mm): camera side / water side "
        + "; ".join(f"cam{c} {a:.3f} / {b:.3f} mm" for c, (a, b) in truth.items()))
    out = {"truth": truth}
    for t in (10.0, 12.7):
        v = P.make_view(rig, wp, [0, 0, 1, 1], thickness=12.7)
        v.refr_window_table.item(1, 1).setText(f"{t:g}")
        job = P.catch_job(v)[1]
        sol = pg.run(job)
        fe = max(abs(c.f / tc.f - 1) for c, tc in zip(sol.cams, rig.cams))
        dist = {c: (sol.rig.windows[sol.rig.cam_win[c]].distance(sol.cams[c].C),
                    sol.rig.windows[sol.rig.cam_win[c]].distance(sol.cams[c].C) + sol.rig.windows[sol.rig.cam_win[c]].T)
                for c in (2, 3)}
        err = {c: (dist[c][0] - truth[c][0], dist[c][1] - truth[c][1]) for c in dist}
        XA, XB = sol.frames.ends(sol.L)
        out[f"{t:g}"] = {"rms": sol.rms, "f_err": fe, "dist": dist, "err": err, "ends": np.vstack([XA, XB])}
        log(f"window 1 at {t:g} mm: RMS {sol.rms:.5f} px, f error {100 * fe:.3f} %; distance error camera side / water "
            "side " + "; ".join(f"cam{c} {a:+.3f} / {b:+.3f} mm" for c, (a, b) in err.items()))
    right, wrong = out["10"], out["12.7"]
    dX = np.linalg.norm(right.pop("ends") - wrong.pop("ends"), axis=1)
    log(f"the wand ends in the output frame (Z along window 1's normal, origin at the wand's mean), 10 against 12.7 mm: "
        f"they differ by {1e3 * np.median(dX):.1f} um median, {1e3 * dX.max():.1f} um at most")
    out["ends_diff_um"] = {"median": 1e3 * float(np.median(dX)), "max": 1e3 * float(dX.max())}
    cam_side = max(abs(right["err"][c][0]) for c in (2, 3))
    moved = np.mean([wrong["err"][c][0] - right["err"][c][0] for c in (2, 3)])
    log.check("the right thickness puts window 1 where it is (camera-side distance within 0.5 mm of the truth)",
              cam_side < 0.5, f"largest error {cam_side:.3f} mm")
    log.check("2.7 mm too thick: the same fit (RMS within 1 %), the window moved toward the cameras instead",
              abs(wrong["rms"] / right["rms"] - 1) < 0.01 and moved < -0.5,
              f"RMS {right['rms']:.5f} / {wrong['rms']:.5f} px; the camera-side surface moved {moved:+.3f} mm")
    log.save(out)


if __name__ == "__main__":
    main()
