"""T5a: what each piece of the start without input (stages 1-2) recovers, on known rigs (diagnostic for T5).

For the reference rig and the irregular 4-camera rig (0.1 px noise, 500 frames):
  stage 1  affine factorisation: rank-3 residual, metric-upgrade residual, frames used;
  stage 2a orientation (both mirror images): each camera's rotation against the truth (tube frame, gauge);
  stage 2b at the TRUE axis position: the distance from the magnification ratio and f from the scale, against
           the truth;
  search   where the best candidates sit, and their cost; then stages 3-4 from the best two, against a fit
           started at the truth.
Outputs: t5a_bootstrap_diagnostics_out.txt.
"""
import math

import numpy as np

import rigs
from rigs import cm
import cylinder_wand as cw

log = rigs.Tee(str(rigs.HERE / "t5a_bootstrap_diagnostics") + "_out.txt")


def run(name, truth, half, seed, centre=(0.0, 0.0, 0.0)):
    tube = rigs.CASE_TUBE
    rng = np.random.default_rng(seed)
    wd = rigs.wand_frames(truth, tube, 500, 10.0, rng, half=(half, half, half), centre=centre, noise_px=0.1)
    sizes = [(c.width, c.height) for c in truth]
    data = cw.WandData(wd.uv, 10.0, sizes)
    obs = cw.observations(data)
    log(f"== {name}: {obs.n_frames} frames, {len(obs)} observations")
    aff = cw.affine_factorization(obs, 10.0)
    log(f"stage 1: affine rank-3 residual {aff.rms_px:.2f} px RMS per image, metric upgrade |d|^2/L^2 - 1 RMS "
        f"{aff.length_rms_rel:.4f}, frames used {len(aff.frames)}")
    cxcy = [(c.cx, c.cy) for c in truth]
    # the truth's cloud centroid, in the gauge frame, for comparison
    XA, XB = wd.XA[obs.frame_ids[aff.frames]], wd.XB[obs.frame_ids[aff.frames]]
    cent_true = np.vstack([XA, XB]).mean(axis=0)
    for mirror in (False, True):
        ori = cw.orient(aff, cxcy, mirror)
        rot = [cm.rotation_angle_deg(ori.R[c], truth[c].R) for c in range(len(truth))]
        log(f"stage 2a, mirror {mirror}: rotation vs truth {', '.join(f'{r:.2f}' for r in rot)} deg; |axis . view| "
            f"{', '.join(f'{v:.3f}' for v in ori.axis_view_cos)}; ratios {', '.join(f'{v:.3f}' for v in ori.ratio)}")
    # the mirror image whose rotations match the truth
    errs = {m: max(cm.rotation_angle_deg(cw.orient(aff, cxcy, m).R[c], truth[c].R) for c in range(len(truth)))
            for m in (False, True)}
    ori = cw.orient(aff, cxcy, min(errs, key=errs.get))
    log(f"stage 2b uses mirror {ori.mirror}")
    # at the true axis position relative to the centroid: p = axis - centroid = -centroid (xz)
    p_true = (-cent_true[0], -cent_true[2])
    pl = cw.place(ori, p_true, tube, sizes, cxcy)
    if pl is not None:
        cams, _ = pl
        for c, (k, t) in enumerate(zip(cams, truth)):
            Dk, Dt = math.hypot(k.C[0], k.C[2]), math.hypot(t.C[0], t.C[2])
            log(f"stage 2b at the true axis: camera {c + 1}: distance {Dk:7.1f} mm (true {Dt:7.1f}), f {k.f:8.0f} px "
                f"(true {t.f:8.0f}, ratio {k.f / t.f:.3f}), rotation error {cm.rotation_angle_deg(k.R, t.R):.2f} deg")
    ranked, info = cw.bootstrap(obs, tube, 10.0, sizes, verbose=True, log=log)
    log(f"search: {info['candidates']} candidates, best mirror {info['best']['mirror']} at p "
        f"({info['best']['p'][0]:+.1f}, {info['best']['p'][1]:+.1f}) mm (true ({p_true[0]:+.1f}, {p_true[1]:+.1f})), "
        f"{info['seconds']:.1f} s")
    sol = cw.calibrate(data, tube, verbose=True, log=log)
    obs_t = cw.observations(data)
    ref = cw.calibrate(data, tube, start=truth)
    cmpr = rigs.compare(sol.cams, ref.cams)
    log(f"from scratch: rms {sol.rms:.6f} px vs from the truth {ref.rms:.6f} px; differences "
        f"{max(e['rot_deg'] for e in cmpr):.1e} deg, {max(e['centre_mm'] for e in cmpr):.1e} mm, f "
        f"{max(abs(e['f_rel']) for e in cmpr):.1e}; total {sol.info['total_seconds']:.0f} s")


if __name__ == "__main__":
    run("reference rig", rigs.reference_rig(3, 45.0, sensor=(3200, 2400)), 20.0, 51)
    run("irregular 4 cams, cloud off-axis", rigs.irregular_rig(), 15.0, 52, centre=(8.0, 5.0, -6.0))
    log.close()
