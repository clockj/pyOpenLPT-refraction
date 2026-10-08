"""E1 and E2 (plan, Phase 3): the curved image lines against the straight ones.

    python e1_coverage.py [--build p3] [--case f5_spots] [--n 50000]

Cameras and view volume: the synthetic case data/CASE (synth.py: 6 Oct T2's PINCYL cameras at tolerance 1e-9, its
view volume).  N random points of the view volume that both cameras see on their sensors; their exact pixels.  For
each point and each ordered camera pair (i -> j): camera i's line of sight through the point's pixel, drawn on
camera j by StereoMatch (debug binding) as
  - the polyline of makeCurve2DFromLOS3D (Phase 3), and
  - the straight line of makeLine2DFromLOS3D (today's: through the images of the view volume's entry and exit);
and the distance from the point's true pixel on camera j to each.
E1 judged by (plan): >= 99.9 % of the points within 0.25 px of the polyline.  The straight line is shown beside it.
E2: the candidate load at T1's density (about 1,500 detections per image, P0 / recon_t1): for each tolerance the
expected number of other detections inside a curve's strip = density x 2 tol x the curve's length on the sensor,
for the tolerance each curve needs to keep 99.9 % of the true matches (the 99.9th percentile of E1's distances, at
least 0.25 px).  Writes e1_coverage_out.txt and e1_coverage.json.
"""
import argparse
import itertools
import os

import numpy as np

import cstb

T1_DENSITY = 1500.0 / (1280 * 800)    # detections per px^2 (P0: 1,179 and 1,608 objects in frame 0)


def length_on_sensor(poly, W, H):
    """Length of the polyline's parts inside [0, W) x [0, H) (segments clipped by sampling)."""
    tot = 0.0
    for a, b in zip(poly[:-1], poly[1:]):
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        seg = np.linspace(0, 1, 65)
        pts = a[None, :] + seg[:, None] * (b - a)[None, :]
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
        tot += np.linalg.norm(b - a) * inside[:-1].mean()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p3")
    ap.add_argument("--case", default="f5_spots")
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    log = cstb.Log(os.path.join(cstb.HERE, "e1_coverage"))
    rng = np.random.default_rng(args.seed)
    cfg = os.path.join(cstb.DATA, args.case, "config.json")
    bs = lpt.BasicSetting()
    bs.readConfig(cfg)
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)
    cams = list(bs._cam_list)
    sm = lpt.StereoMatch(cams, [[] for _ in cams], tc)
    al = bs._axis_limit
    box = ((al.x_min, al.y_min, al.z_min), (al.x_max, al.y_max, al.z_max))
    log(f"E1/E2: curved against straight image lines; build {build['name']} ({build['git_head']} + diff "
        f"{(build['git_diff_sha256'] or '')[:12]}); case {args.case}, view volume {box}")
    pts, pix = cstb.sample_shared(lpt, cams, args.n, box, rng)
    W, H = cams[0].getNCol(), cams[0].getNRow()
    out = {"build": build, "n": len(pts), "box": box, "pairs": {}}
    for i, j in itertools.permutations(range(len(cams)), 2):
        d_poly, d_line, n_vert, lens_poly, lens_line, fails = [], [], [], [], [], 0
        for k in range(len(pts)):
            los = cams[i].lineOfSight(lpt.Pt2D(float(pix[k, i, 0]), float(pix[k, i, 1])))
            ok_c, _chord, poly = sm.makeCurve2DFromLOS3D(j, los)
            ok_l, line = sm.makeLine2DFromLOS3D(j, los)
            q = lpt.Pt2D(float(pix[k, j, 0]), float(pix[k, j, 1]))
            if not (ok_c and ok_l):
                fails += 1
                continue
            d_poly.append(float(np.sqrt(lpt.dist2ToPolyline(q, poly))))
            ux, uy = line.unit_vector[0], line.unit_vector[1]
            d_line.append(abs((q[1] - line.pt[1]) * ux - (q[0] - line.pt[0]) * uy))
            P = np.array([[p[0], p[1]] for p in poly])
            n_vert.append(len(P))
            if k < 2000:
                lens_poly.append(length_on_sensor(P, W, H))
        d_poly, d_line = np.array(d_poly), np.array(d_line)
        rec = {"points": len(pts), "failed": fails,
               "poly_px": {"median": float(np.median(d_poly)), "p99": float(np.percentile(d_poly, 99)),
                           "p999": float(np.percentile(d_poly, 99.9)), "max": float(d_poly.max()),
                           "share_le_0.25": float(np.mean(d_poly <= 0.25)), "share_le_0.1": float(np.mean(d_poly <= 0.1))},
               "line_px": {"median": float(np.median(d_line)), "p99": float(np.percentile(d_line, 99)),
                           "p999": float(np.percentile(d_line, 99.9)), "max": float(d_line.max()),
                           "share_le_0.25": float(np.mean(d_line <= 0.25)), "share_le_1": float(np.mean(d_line <= 1.0))},
               "vertices": {"median": float(np.median(n_vert)), "max": int(np.max(n_vert))}}
        # E2: tolerance for 99.9 % of the true matches, and the expected other detections in the strip
        L = float(np.median(lens_poly))
        for name, d in (("poly", d_poly), ("line", d_line)):
            tol = max(0.25, float(np.percentile(d, 99.9)))
            rec[f"e2_{name}"] = {"tol_for_99.9_px": tol, "length_on_sensor_px": L,
                                 "expected_other_candidates": T1_DENSITY * 2 * tol * L}
        rec["pass"] = bool(rec["poly_px"]["share_le_0.25"] >= 0.999)
        out["pairs"][f"cam{i + 1}->cam{j + 1}"] = rec
        log(f"\n  cam{i + 1} -> cam{j + 1}: {len(d_poly)} points ({fails} without a line)")
        log(f"    polyline: median {rec['poly_px']['median']:.4f}, p99 {rec['poly_px']['p99']:.4f}, "
            f"p99.9 {rec['poly_px']['p999']:.4f}, max {rec['poly_px']['max']:.4f} px; within 0.25 px "
            f"{100 * rec['poly_px']['share_le_0.25']:.3f} %; vertices median {rec['vertices']['median']:.0f}, "
            f"max {rec['vertices']['max']}")
        log(f"    straight: median {rec['line_px']['median']:.4f}, p99 {rec['line_px']['p99']:.4f}, "
            f"p99.9 {rec['line_px']['p999']:.4f}, max {rec['line_px']['max']:.4f} px; within 0.25 px "
            f"{100 * rec['line_px']['share_le_0.25']:.2f} %, within 1 px {100 * rec['line_px']['share_le_1']:.2f} %")
        log(f"    E2 at T1's density ({T1_DENSITY * 1e6:.0f} per Mpx), curve length on the sensor {L:.0f} px: "
            f"polyline tol {rec['e2_poly']['tol_for_99.9_px']:.2f} px -> "
            f"{rec['e2_poly']['expected_other_candidates']:.2f} other candidates; straight tol "
            f"{rec['e2_line']['tol_for_99.9_px']:.2f} px -> {rec['e2_line']['expected_other_candidates']:.2f}")
        log(f"    E1 judged (>= 99.9 % within 0.25 px): {'PASS' if rec['pass'] else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
