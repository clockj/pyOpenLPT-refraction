"""E0: does every wand end of the campaign's data lie in the single-image volume of every camera?

For the reference rig (three copies of cam1 at 0, 45 and 90 deg; tests T4, T7, T9, T11) and for each of the 50
random rigs of T5 and T12d, the C++ projectAll (the paper's projection) counts the images, on the sensor or not, of:
  (a) every wand end of the data sets behind the paper's numbers, regenerated with the campaign's generator and
      seeds: T4's 20 seeds (rng 1000-1019, 500 frames) and T9's (rng 909, 2000 frames) on the reference rig, and
      T5's 50 rigs (rng 5000 + k, 500 frames);
  (b) a dense random sample of the volume the generator draws ends from: the cloud's box widened by L/2 on every side
      and clipped to r < 0.98 r_c(0), as rigs.wand_frames clips the ends.  If every point of it has one image in
      every camera, so does every data set the generator makes on that rig, whatever its seed or noise.
Reported: the image counts, the largest wand-end radius (distance from the axis) and the largest |C_z| of the rays.
Criterion: every count is 1.
Outputs: e0_census_out.txt, e0_census.json.
"""
import numpy as np

import projection_fit as pf
from projection_fit import cm, rigs

L = 10.0
N_BOX_REF, N_BOX_T5 = 20000, 4000


def census(cams, tube, P):
    """Image counts (n, n_cams) and the largest |C_z| over the rays, by the C++ projectAll."""
    lpt = cm.import_lpt()
    counts = np.zeros((len(P), len(cams)), int)
    cz = 0.0
    for c, cam in enumerate(cams):
        cc = cm.to_cpp(cam, tube)
        for i, p in enumerate(P):
            rays, _cov = cc.projectAll(lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])))
            rays = [r for r in rays if not r["is_ring"]]
            counts[i, c] = len(rays)
            if rays:
                cz = max(cz, max(abs(float(r["c_z"])) for r in rays))
    return counts, cz


def box_sample(rng, centre, half, n, r_max):
    """n points uniform in the box centre +- (half + L/2), kept where r < r_max (r: distance from the axis Y)."""
    pts = []
    while sum(len(p) for p in pts) < n:
        q = np.asarray(centre) + rng.uniform(-1, 1, size=(2 * n, 3)) * (np.asarray(half) + 0.5 * L)
        pts.append(q[np.hypot(q[:, 0], q[:, 2]) < r_max])
    return np.concatenate(pts)[:n]


def summary(counts):
    vals, num = np.unique(counts, return_counts=True)
    return {int(v): int(k) for v, k in zip(vals, num)}


def main():
    run = pf.Run("e0_census")
    tube = rigs.CASE_TUBE
    r_max = tube.fold_radius() * 0.98
    run.log(f"tube {tube.radii} mm, n {tube.n}; r_c(0) = {tube.fold_radius():.4f} mm; the generator keeps ends within "
            f"r_max = 0.98 r_c(0) = {r_max:.4f} mm")
    out = {"r_c0_mm": tube.fold_radius(), "r_max_mm": r_max, "reference": {}, "t5": []}
    all_one = True

    # ---- the reference rig: T4's and T9's data, and its whole volume
    truth = rigs.reference_rig(3, 45.0, sensor=(3200, 2400))
    ends, n_sets = [], 0
    for seed in range(20):
        wd = rigs.wand_frames(truth, tube, 500, L, np.random.default_rng(1000 + seed), half=(20, 20, 20),
                              noise_px=0.1)
        ends += [wd.XA, wd.XB]
        n_sets += 1
    wd = rigs.wand_frames(truth, tube, 2000, L, np.random.default_rng(909), half=(20, 20, 20), noise_px=0.1)
    ends += [wd.XA, wd.XB]
    E = np.concatenate(ends)
    cnt, cz = census(truth, tube, E)
    rE = np.hypot(E[:, 0], E[:, 2])
    box = box_sample(np.random.default_rng(42), (0, 0, 0), (20, 20, 20), N_BOX_REF, r_max)
    cntb, czb = census(truth, tube, box)
    ok = bool(np.all(cnt == 1) and np.all(cntb == 1))
    all_one &= ok
    out["reference"] = {"data_sets": "T4 seeds 0-19 (500 frames) and T9 (2000 frames)", "ends": int(len(E)),
                        "counts_data": summary(cnt), "max_radius_mm": float(rE.max()), "max_abs_cz_data": cz,
                        "box_points": int(len(box)), "counts_box": summary(cntb),
                        "box_max_radius_mm": float(np.hypot(box[:, 0], box[:, 2]).max()), "max_abs_cz_box": czb,
                        "all_one": ok}
    run.log(f"reference rig: {len(E)} wand ends of T4 (20 seeds) and T9: image counts per camera {summary(cnt)}; "
            f"largest radius {rE.max():.2f} mm, largest |C_z| {cz:.4f}")
    run.log(f"  its volume (+-{20 + L / 2:.0f} mm box): {len(box)} points, counts {summary(cntb)}, largest |C_z| "
            f"{czb:.4f} -> {'every count is 1' if ok else 'NOT ALL 1'}")

    # ---- the 50 random rigs of T5 (and T12d, which reuses them)
    worst_r, worst_cz = 0.0, 0.0
    for k in range(50):
        rng = np.random.default_rng(5000 + k)
        cams, centre, half = rigs.random_rig(rng)
        wd = rigs.wand_frames(cams, tube, 500, L, rng, half=(half, half, half), centre=centre, noise_px=0.1)
        E = np.concatenate([wd.XA, wd.XB])
        cnt, cz = census(cams, tube, E)
        box = box_sample(np.random.default_rng(7000 + k), centre, (half, half, half), N_BOX_T5, r_max)
        cntb, czb = census(cams, tube, box)
        rE = float(np.hypot(E[:, 0], E[:, 2]).max())
        ok = bool(np.all(cnt == 1) and np.all(cntb == 1))
        all_one &= ok
        worst_r, worst_cz = max(worst_r, rE), max(worst_cz, cz, czb)
        out["t5"].append({"rig": k, "n_cams": len(cams), "half_mm": float(half), "centre_mm": centre.tolist(),
                          "ends": int(len(E)), "counts_data": summary(cnt), "max_radius_mm": rE,
                          "max_abs_cz_data": cz, "box_points": int(len(box)), "counts_box": summary(cntb),
                          "max_abs_cz_box": czb, "all_one": ok})
        run.log(f"T5 rig {k:2d}: {len(cams)} cams, centre ({centre[0]:+5.1f}, {centre[1]:+5.1f}, {centre[2]:+5.1f}) "
                f"half {half:4.1f} | data {len(E)} ends {summary(cnt)}, r <= {rE:5.2f} mm, |C_z| <= {cz:.3f} | box "
                f"{summary(cntb)}, |C_z| <= {czb:.3f} | {'ok' if ok else 'NOT ALL 1'}")
    out["t5_max_radius_mm"], out["t5_max_abs_cz"] = worst_r, worst_cz
    out["all_one"] = all_one
    run.log(f"T5 rigs: largest wand-end radius {worst_r:.2f} mm, largest |C_z| {worst_cz:.4f}")
    run.log(f"E0: {'PASS: every wand end and every sampled point has exactly one image in every camera' if all_one else 'FAIL: some count is not 1'}")
    run.finish(out)


if __name__ == "__main__":
    main()
