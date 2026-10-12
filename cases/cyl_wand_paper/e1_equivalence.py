"""E1: the paper's residual against the library's, on the campaign's data.

evaluate_by_projection (Newton started at the image of the C++ projection) and cylinder_wand.evaluate (Newton started
at the detection) are compared at two parameter points:
  - the truth: the true cameras and wand frames;
  - the campaign's start: each camera 2 deg / 20 mm / 5 % in f off (rigs.perturb_rig, as T4 draws it), with the wand
    frames triangulated from it (cylinder_wand.initial_frames).
Data: the reference rig (T4 seed 0: rng 1000, 500 frames, 0.1 px, and T4's own start) and T5 rigs 0-9
(rng 5000 + k; start from rng 6000 + k).
Compared: the ok flags, the residuals (px) and the Jacobians (camera and frame columns, each relative to its largest
entry), over the rows that are ok in both.
Criteria: at the truth, residuals within 1e-8 px and Jacobians within 1e-6 relative, with the same ok flags.  At the
start every difference is reported, with the rows' image counts.
Outputs: e1_equivalence_out.txt, e1_equivalence.json.
"""
import numpy as np

import projection_fit as pf
from projection_fit import cw, rigs

L = 10.0
TOL_R, TOL_J = 1e-8, 1e-6


def compare(cams, tube, frames, obs, layout):
    pf.reset_stats()
    a = cw.evaluate(cams, tube, frames, L, obs, layout)
    b = pf.evaluate_by_projection(cams, tube, frames, L, obs, layout)
    both = a.ok & b.ok
    rel = lambda x, y: float(np.max(np.abs(x - y)) / max(np.max(np.abs(x)), 1e-300)) if both.any() else 0.0
    return {"rows": int(len(a.ok)), "ok_library": int(a.ok.sum()), "ok_projection": int(b.ok.sum()),
            "ok_differ": int(np.sum(a.ok != b.ok)),
            "max_dr_px": float(np.max(np.abs(a.r[both] - b.r[both]))) if both.any() else 0.0,
            "max_dJ_cam_rel": rel(a.Jg[both], b.Jg[both]), "max_dJ_frame_rel": rel(a.Jf[both], b.Jf[both]),
            "rms_px": float(np.sqrt(np.mean(np.sum(a.r[both] ** 2, axis=1)))) if both.any() else float("nan"),
            "cpp_images": {k: pf.STATS[k] for k in ("no_image", "one_image", "more_images")}}


def data_sets():
    tube = rigs.CASE_TUBE
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    rng = np.random.default_rng(1000)
    wd = rigs.wand_frames(truth, tube, 500, L, rng, half=(20, 20, 20), noise_px=0.1)
    yield "reference rig (T4 seed 0)", truth, wd, rigs.perturb_rig(truth, rng, rot_deg=2.0, trans_mm=20.0, f_rel=0.05)
    for k in range(10):
        rng = np.random.default_rng(5000 + k)
        cams, centre, half = rigs.random_rig(rng)
        wd = rigs.wand_frames(cams, tube, 500, L, rng, half=(half, half, half), centre=centre, noise_px=0.1)
        start = rigs.perturb_rig(cams, np.random.default_rng(6000 + k), rot_deg=2.0, trans_mm=20.0, f_rel=0.05)
        yield f"T5 rig {k}", cams, wd, start


def main():
    run = pf.Run("e1_equivalence")
    tube = rigs.CASE_TUBE
    out, passed = [], True
    for name, truth, wd, start in data_sets():
        data = cw.WandData(wd.uv, L, [(c.width, c.height) for c in truth])
        obs = cw.observations(data)
        layout = cw.Layout(len(truth))
        fr_true = cw.frames_from_ends(wd.XA[obs.frame_ids], wd.XB[obs.frame_ids])
        at_truth = compare(truth, tube, fr_true, obs, layout)
        at_start = compare(start, tube, cw.initial_frames(start, tube, obs, L), obs, layout)
        ok = (at_truth["ok_differ"] == 0 and at_truth["max_dr_px"] <= TOL_R
              and max(at_truth["max_dJ_cam_rel"], at_truth["max_dJ_frame_rel"]) <= TOL_J)
        passed &= ok
        out.append({"data": name, "n_cams": len(truth), "truth": at_truth, "start": at_start, "pass": ok})
        for label, c in (("truth", at_truth), ("start", at_start)):
            run.log(f"{name:26s} {label:5s}: {c['rows']} rows, ok {c['ok_library']}/{c['ok_projection']} "
                    f"(library/projection, {c['ok_differ']} differ) | max |dr| {c['max_dr_px']:.1e} px | max dJ "
                    f"cam {c['max_dJ_cam_rel']:.1e}, frame {c['max_dJ_frame_rel']:.1e} | RMS {c['rms_px']:.4g} px | "
                    f"C++ images {c['cpp_images']}")
    run.log(f"E1: {'PASS' if passed else 'FAIL'} at the truth (residuals within {TOL_R:g} px, Jacobians within "
            f"{TOL_J:g} relative, the same ok flags); the start's differences are listed above")
    run.finish({"criteria": {"residual_px": TOL_R, "jacobian_rel": TOL_J}, "cases": out, "pass": passed})


if __name__ == "__main__":
    main()
