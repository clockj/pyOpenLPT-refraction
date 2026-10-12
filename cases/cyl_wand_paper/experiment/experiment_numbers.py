"""Every number that the paper's Section 9.3 (the experiment) quotes, and the data of its two figures, read from the
results of run 20261007, each printed with its source.

The run is processed by the case pyOpenLPT/cases/cyl_20261007 (plan, reports and checks there); its deliverables are on
X:\\Shijie Zhong\\Refraction\\Processed\\20261007 (T0: the wand calibration; Planes: the dot board; T1: the tracks of
the 15 px median rerun of 8 Oct). Nothing is fitted or tracked afresh. Three things are computed here:
  - the tracks smoothed with the group's Gaussian filter (pyOpenLPT modules/post_processing/processor.py,
    compute_kinematics_and_filter: position kernel exp(-k^2/w^2), velocity kernel k exp(-k^2/w^2), both over
    |k| <= ceil(3 w), each normalised as there; the ceil(3 w) points at each end of a track are dropped), with w = 3
    steps (the author, 2026-10-08), and their speeds and swirl profile;
  - the line detections of one sample frame, with the library build the tracking used (build-stb/p4c), its default
    LineFinder settings and the noise the run gave it, as the case's tests/t1_checks.py does;
  - which crop of that frame and which wand frame the figures show (rules below).
Coordinates: the run's frame has the tube's axis along Y through the origin and camera 1 on +Z. The paper's frame has
the axis along z and camera 1 on +x, so (x, y, z)_paper = (Z, X, Y)_run (a cyclic permutation, right-handed).

    python experiment_numbers.py      (refract environment; CYL_EXPERIMENT_PYD = a folder with a copy of p4c's .pyd)

Outputs beside this script: experiment_numbers_out.txt, experiment_numbers.json, and experiment_data.npz, which
code/make_paper_experiment_figs.py --refresh copies into the paper's figures/data/.
"""
import csv
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                            # pyOpenLPT
CASE = REPO / "cases" / "cyl_20261007"
PROC = Path(r"X:\Shijie Zhong\Refraction\Processed\20261007")
RAW = Path(r"X:\Shijie Zhong\Refraction\Raw\20261007")
STUDY = REPO / "tests" / "cylinder_stb_study"

W_FILTER = 3.0            # the Gaussian filter's width, in steps (the author)
STEP_S = 4 / 4000.0       # one processed step: every 4th raw frame at 4,000 frames/s
T1_FRAME = 200            # the sample frame (processed index; raw frame 800): tracking has built up from frame 150
CROP = (200, 125)         # the image crops of the tracer figure, px (the sensor's 1.6 aspect)
TRAIL = 15                # steps of image trail drawn behind each tracked particle
LONG = 100                # "long" tracks for the figure: at least 100 positions (0.1 s)

LINES, OUT = [], {}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def say(key, value, source, fmt="{}"):
    """One number: printed with its source, and kept under `key` in the JSON."""
    OUT[key] = value
    text = fmt.format(value) if not isinstance(value, (list, tuple)) else ", ".join(fmt.format(v) for v in value)
    LINES.append(f"{key:46s} {text:46s} <- {source}")


def read_image(path):
    with open(path, "rb") as fh:
        img = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def gaussian_kernels(w):
    """processor.py's kernels (get_pv_kernels): position and first derivative, half-width ceil(3 w)."""
    fw = int(np.ceil(3 * w))
    k = np.arange(-fw, fw + 1)
    kp = np.arange(1, fw + 1)
    p = np.exp(-k ** 2 / w ** 2) / (2 * np.sum(np.exp(-kp ** 2 / w ** 2)) + 1)
    v = k * np.exp(-k ** 2 / w ** 2) / (2 * np.sum(kp ** 2 * np.exp(-kp ** 2 / w ** 2)))
    return fw, p, v


def smooth(X, w, dt):
    """Smoothed positions and velocities of one track (rows: steps), the ends dropped as processor.py does."""
    fw, p, v = gaussian_kernels(w)
    xs = np.column_stack([np.convolve(X[:, d], p, mode="same") for d in range(3)])
    vs = np.column_stack([-np.convolve(X[:, d], v, mode="same") / dt for d in range(3)])
    return xs[fw:len(X) - fw], vs[fw:len(X) - fw]


def read_tracks(folder):
    """Long tracks (active, inactive, exited) at the last frame: {(kind, id): rows [frame, X, Y, Z, u1, v1, u2, v2]}."""
    files = glob.glob(str(folder / "*.csv"))
    last = max(int(Path(f).stem.rsplit("_", 1)[1]) for f in files)
    tracks = {}
    for kind in ("LongTrackActive", "LongTrackInactive", "ExitTrack"):
        with open(folder / f"{kind}_{last}.csv", newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd)
            for row in rd:
                if row:
                    tracks.setdefault((kind, int(row[0])), []).append([float(x) for x in row[1:9]])
    return {k: np.array(sorted(v)) for k, v in tracks.items()}, last


def trace(P, d, radii, n):
    """A ray from the pinhole P along d (the run's frame: the axis along Y through the origin), refracted by vector
    Snell at each interface: [P, the crossing of each interface, the far side of the innermost one]."""
    p, dd, pts = np.array(P, float), np.array(d, float) / np.linalg.norm(d), [np.array(P, float)]
    for k, Rk in enumerate(radii):
        a, b = dd[0] ** 2 + dd[2] ** 2, 2 * (p[0] * dd[0] + p[2] * dd[2])
        disc = b * b - 4 * a * (p[0] ** 2 + p[2] ** 2 - Rk ** 2)
        assert disc > 0, "the ray misses an interface"
        p = p + (-b - math.sqrt(disc)) / (2 * a) * dd
        pts.append(p.copy())
        nrm = np.array([p[0], 0.0, p[2]]) / math.hypot(p[0], p[2])  # outward: against the incoming ray
        eta, cosi = n[k] / n[k + 1], -float(dd @ nrm)
        dd = eta * dd + (eta * cosi - math.sqrt(1 - eta ** 2 * (1 - cosi ** 2))) * nrm
        dd /= np.linalg.norm(dd)
    a, b = dd[0] ** 2 + dd[2] ** 2, 2 * (p[0] * dd[0] + p[2] * dd[2])
    disc = b * b - 4 * a * (p[0] ** 2 + p[2] ** 2 - radii[-1] ** 2)
    pts.append(p + (-b + math.sqrt(disc)) / (2 * a) * dd)
    return np.array(pts)


def to_paper(A):
    """(X, Y, Z) of the run's frame -> (x, y, z) of the paper's: x = Z (towards camera 1), y = X, z = Y (the axis)."""
    A = np.asarray(A, float)
    return np.stack([A[..., 2], A[..., 0], A[..., 1]], axis=-1)


def main():
    cfg = load(CASE / "config.json")
    p0a = load(CASE / "tests" / "p0a_files.json")
    data = {}

    # ---- the rig
    say("rig.camera", cfg["image"]["camera"], "cases/cyl_20261007/config.json image.camera")
    say("rig.sensor_px", [cfg["image"]["width"], cfg["image"]["height"]], "config.json image.width/height")
    say("rig.pixel_um", 1000 * cfg["image"]["pixel_mm"], "config.json image.pixel_mm", "{:.0f}")
    say("rig.tube_od_mm", 25.4 * cfg["tube"]["outer_diameter_in"], "config.json tube.outer_diameter_in x 25.4", "{:.4f}")
    say("rig.tube_wall_mm", 25.4 * cfg["tube"]["wall_in"], "config.json tube.wall_in x 25.4", "{:.3f}")
    say("rig.radii_mm", cfg["tube"]["radii_mm"], "config.json tube.radii_mm", "{:.5f}")
    say("rig.n_air_acrylic_water", cfg["tube"]["n"], "config.json tube.n", "{:.3f}")
    say("rig.wand_length_mm", cfg["wand"]["length_mm"], "config.json wand.length_mm", "{:.1f}")
    say("rig.wand_balls_mm", cfg["wand"]["ball_diameters_mm"], "config.json wand.ball_diameters_mm", "{:.0f}")
    say("rig.tracer", f'{cfg["tracers"]["particle_diameter_um"]} um {cfg["tracers"]["particle_material"]}',
        "config.json tracers")
    say("rig.t1_fps_exposure_us", [cfg["tracers"]["fps"], cfg["tracers"]["exposure_us"]],
        "config.json tracers.fps/exposure_us (cam2's .chd; the user: both cameras alike, one trigger)")
    say("rig.t0_frames_fps_exposure_us",
        [p0a["T0/cam1"]["frames"], p0a["T0/cam1"]["header"]["fps"], p0a["T0/cam1"]["header"]["exposure_us"]],
        "cases/cyl_20261007/tests/p0a_files.json T0/cam1 (cam2 the same)")

    # ---- the calibration (T0)
    res = load(PROC / "T0" / "calibration_result.json")
    say("cal.model", res["model"], "T0/calibration_result.json model")
    say("cal.frames", res["frames"], "T0/calibration_result.json frames")
    say("cal.detections_used_of", [res["active"], res["observations"]], "calibration_result.json active/observations")
    say("cal.multi_image_or_beyond_fold_observations",
        [res["info"]["multi_image_observations"], res["info"]["beyond_fold_observations"]],
        "calibration_result.json info.multi_image_observations, beyond_fold_observations")
    say("cal.rms_px", res["rms_px"], "calibration_result.json rms_px", "{:.3f}")
    say("cal.rms_px_per_camera", [c["residuals"]["rms_px"] for c in res["cameras"]],
        "calibration_result.json cameras[].residuals.rms_px", "{:.3f}")
    say("cal.distance_from_axis_mm", [c["distance_from_axis_mm"] for c in res["cameras"]],
        "calibration_result.json cameras[].distance_from_axis_mm", "{:.1f}")
    say("cal.azimuth_deg", [c["azimuth_deg"] for c in res["cameras"]], "calibration_result.json cameras[].azimuth_deg",
        "{:.1f}")
    say("cal.f_px_(not_quoted)", [c["f_px"] for c in res["cameras"]],
        "calibration_result.json cameras[].f_px; f and distance trade off (case Phase C report, C1-T)", "{:.0f}")
    say("cal.wand_length_fixed_mm", [res["wand_length_mm"], res["fit_L"]], "calibration_result.json wand_length_mm, fit_L")
    wl = res["wand_length_free_triangulation"]
    say("cal.free_wand_length_frames", wl["frames"], "calibration_result.json wand_length_free_triangulation.frames")
    say("cal.free_wand_length_rms_um", 1000 * wl["rms_mm"], "... .rms_mm (the mean is held by the fit)", "{:.1f}")
    say("cal.free_wand_length_rms_pct", 100 * wl["rms_mm"] / res["wand_length_mm"], "... .rms_mm / wand_length_mm",
        "{:.2f}")
    held = load(CASE / "tests" / "c1t_held_out.json")
    parts = [k for k in held if k != "C1"]
    hr = [held[k]["held_out"]["length_rms_um"] for k in parts]
    hm = [held[k]["held_out"]["length_mean_um"] for k in parts]
    say("cal.held_out_parts", parts, "cases/cyl_20261007/tests/c1t_held_out.json (halves h1-h2, quarters q1-q4)")
    say("cal.held_out_length_rms_um_range", [min(hr), max(hr)], "c1t_held_out.json <part>.held_out.length_rms_um",
        "{:.1f}")
    say("cal.held_out_length_mean_um_range", [min(hm), max(hm)], "c1t_held_out.json <part>.held_out.length_mean_um",
        "{:+.1f}")
    say("cal.held_out_length_mean_pct_max", 100 * max(abs(x) for x in hm) / 1000 / res["wand_length_mm"],
        "max |held_out.length_mean_um| / 10 mm", "{:.2f}")
    for c in (0, 1):
        fp = [held[k]["f"][c] for k in parts]
        f_all = res["cameras"][c]["f_px"]
        say(f"cal.parts_f_px_range_cam{c + 1}", [min(fp), max(fp)], "c1t_held_out.json <part>.f", "{:.0f}")
        say(f"cal.parts_f_rel_pct_cam{c + 1}", [100 * (min(fp) / f_all - 1), 100 * (max(fp) / f_all - 1)],
            "the parts' f against the whole recording's", "{:+.1f}")
    c1r = load(CASE / "tests" / "c1_residuals.json")
    say("cal.residual_cell_max_px_(not_quoted)",
        [c1r["cameras"][c]["largest_cell"]["mean_px"] for c in ("cam1", "cam2")],
        "cases/cyl_20261007/tests/c1_residuals.json cameras.<c>.largest_cell.mean_px (10 x 6 cells)", "{:.2f}")

    # the board (Planes), triangulated with these cameras
    bc = load(CASE / "tests" / "c2a_board_check.json")
    pos = sorted(bc["positions"])
    flat = [bc["positions"][p]["plane"]["rms_um"] for p in pos]
    ang = [bc["positions"][p]["lattice"]["angle_deg"] for p in pos]
    pa = [bc["positions"][p]["lattice"]["pitch_A_mm"] for p in pos]
    pb = [bc["positions"][p]["lattice"]["pitch_B_mm"] for p in pos]
    say("board.dots_paired", [bc["positions"][p]["pairs"] for p in pos], "c2a_board_check.json positions.<p>.pairs")
    say("board.flatness_rms_um", flat, "c2a_board_check.json positions.<p>.plane.rms_um", "{:.1f}")
    say("board.angle_deg", ang, "c2a_board_check.json positions.<p>.lattice.angle_deg", "{:.3f}")
    say("board.pitch_across_axis_mm", pa, "c2a_board_check.json lattice.pitch_A_mm (i: across the axis)", "{:.4f}")
    say("board.pitch_along_axis_mm", pb, "c2a_board_check.json lattice.pitch_B_mm (j: along the axis)", "{:.4f}")
    say("board.pitch_along_over_across_pct", [100 * (b / a - 1) for a, b in zip(pa, pb)], "pitch_B / pitch_A - 1",
        "{:+.2f}")
    say("board.pitch_vs_3mm_pct", [100 * (min(pa) / 3 - 1), 100 * (max(pb) / 3 - 1)],
        "min pitch_A, max pitch_B against the nominal 3 mm", "{:+.2f}")
    vb = load(CASE / "tests" / "c2b_valley_board.json")["fits"]
    pk = [k for k in vb if k != "C1"]
    pB = [100 * (q["pitch_B"] / 3 - 1) for k in pk for q in vb[k]["positions"]]
    pA = [100 * (q["pitch_A"] / 3 - 1) for k in pk for q in vb[k]["positions"]]
    an = [q["angle_deg"] for k in pk for q in vb[k]["positions"]]
    say("board.parts_fits", pk, "cases/cyl_20261007/tests/c2b_valley_board.json fits (the halves and quarters of C1-T)")
    say("board.parts_pitch_along_vs_3mm_pct", [min(pB), max(pB)], "c2b_valley_board.json <part>.positions[].pitch_B",
        "{:+.1f}")
    say("board.parts_pitch_across_vs_3mm_pct", [min(pA), max(pA)], "c2b_valley_board.json <part>.positions[].pitch_A",
        "{:+.2f}")
    say("board.parts_angle_deg", [min(an), max(an)], "c2b_valley_board.json <part>.positions[].angle_deg", "{:.2f}")
    say("board.reprojection_rms_px", [min(min(bc["positions"][p]["reproj_rms_px"].values()) for p in pos),
                                      max(max(bc["positions"][p]["reproj_rms_px"].values()) for p in pos)],
        "c2a_board_check.json positions.<p>.reproj_rms_px, range over cameras and positions", "{:.2f}")

    # the wand images: the detections of the frames in the fit
    cd = load(PROC / "T0" / "calibration_data.json")
    active = {f for f, a in zip(cd["frame_ids"], cd["frame_active"]) if a}
    det = {}
    with open(PROC / "T0" / "detection" / "detections_all.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            det.setdefault((int(r["frame"]), int(r["camera"])), []).append(r)
    ratio = {1: [], 2: []}
    for f in active:
        for c in (1, 2):
            for r in det.get((f, c), []):
                ratio[c].append(float(r["a"]) / float(r["b"]))
    say("cal.ball_image_aspect_median", [float(np.median(ratio[c])) for c in (1, 2)],
        "T0/detection/detections_all.csv a / b, the balls of the frames in the fit, per camera", "{:.3f}")
    # sample wand frame: in the fit, both balls of both cameras at least 20 px inside the image, chosen among the 10
    # frames whose wand midpoint image is nearest the image centre in both cameras, as the one whose image is longest
    # in camera 1 (the whole wand in view)
    cands = []
    for f in sorted(active):
        ok, score = True, 0.0
        for c in (1, 2):
            rows = det.get((f, c), [])
            if len(rows) != 2:
                ok = False
                break
            us, vs, a = ([float(r[k]) for r in rows] for k in ("u", "v", "a"))
            if min(us) - max(a) < 20 or max(us) + max(a) > 1260 or min(vs) - max(a) < 20 or max(vs) + max(a) > 780:
                ok = False
                break
            score += math.hypot(np.mean(us) - 639.5, np.mean(vs) - 399.5)
        if ok:
            cands.append((score, f))
    cands.sort()
    top = [f for _s, f in cands[:10]]
    f0 = max(top, key=lambda f: abs(float(det[(f, 1)][0]["u"]) - float(det[(f, 1)][1]["u"])))
    say("fig.wand_frame", f0, "the rule above, over T0's detections_all.csv and calibration_data.json")
    data["t0_frame"] = np.array(f0)
    for c in (1, 2):
        data[f"t0_img{c}"] = read_image(RAW / "T0" / f"cam{c}" / f"frame{f0:06d}.tif")
        data[f"t0_ell{c}"] = np.array([[float(r[k]) for k in ("u", "v", "a", "b", "theta_rad")] for r in det[(f0, c)]])
    ob = cd["observations"]
    cam = np.array(ob["cam_id"])
    act = np.array(ob["active"], bool)
    uv = np.column_stack([ob["u"], ob["v"]]).astype(float)
    ids = sorted(set(cam.tolist()))
    for c, cid in zip((1, 2), ids):
        data[f"t0_cov{c}"] = uv[(cam == cid) & act].astype(np.float32)
    say("fig.wand_detections_used_per_camera", [int(len(data["t0_cov1"])), int(len(data["t0_cov2"]))],
        "calibration_data.json observations, active, per camera")
    wlf = np.array(cd["wand_length_free_mm"], float)
    fa = np.array(cd["frame_active"], bool)
    wdir = (np.array(cd["ends_B_mm"], float) - np.array(cd["ends_A_mm"], float))[fa]
    wdir /= np.linalg.norm(wdir, axis=1)[:, None]
    tilt = np.degrees(np.arcsin(np.abs(wdir[:, 1])))          # run frame: the axis is Y
    say("cal.wand_tilt_from_cross_section_deg_50_90_max", [float(np.median(tilt)), float(np.percentile(tilt, 90)),
                                                           float(tilt.max())],
        "calibration_data.json ends_A_mm / ends_B_mm, active frames: angle between the wand and the plane across the "
        "axis", "{:.1f}")
    data["t0_len_um"] = (1000 * (wlf[fa & np.isfinite(wlf)] - res["wand_length_mm"])).astype(np.float32)
    say("fig.wand_length_values", int(len(data["t0_len_um"])), "calibration_data.json wand_length_free_mm, active frames")
    say("fig.wand_length_rms_um_check", float(np.sqrt(np.mean(data["t0_len_um"].astype(float) ** 2))),
        "the same values (cf. cal.free_wand_length_rms_um)", "{:.1f}")

    # the calibrated rig (Fig. 16(a)): each camera's lines of sight through the left and right edges and the centre of
    # its image's centre row, traced through the walls; a point a third of the way across the water on each must
    # project back to its pixel with the library's camera (camFile_track)
    sys.path.insert(0, os.environ.get("CYL_EXPERIMENT_PYD", str(REPO / "build-stb" / "p4c" / "out")))
    import pyopenlpt as lpt  # noqa: E402
    radii, nidx = res["tube"]["radii_mm"], res["tube"]["n"]
    worst = 0.0
    for c, cam in enumerate(res["cameras"], start=1):
        R, t = np.array(cam["R"], float), np.array(cam["tvec"], float)
        C = -R.T @ t
        assert np.allclose(C, cam["centre_mm"], atol=1e-6), "camera centre"
        cpp = lpt.loadCameraModel(str(PROC / "T0" / "camFile_track" / f"cam{c}.txt"))
        rays = []
        for u in (0.0, cam["width"] - 1.0, cam["cx"]):
            pts = trace(C, R.T @ np.array([(u - cam["cx"]) / cam["f_px"], 0.0, 1.0]), radii, nidx)
            X = pts[-2] + (pts[-1] - pts[-2]) / 3.0
            r = cpp.projectBatchStatus([lpt.Pt3D(*map(float, X))])[0]
            ok, uv = r[0], r[1]
            assert ok, "the traced point has no image"
            worst = max(worst, math.hypot(uv[0] - u, uv[1] - cam["cy"]))
            rays.append(to_paper(pts))
        data[f"rig_rays{c}"] = np.array(rays)                     # (left, right, centre) x (P, outer, inner, far) x 3
    say("rig.traced_rays_reprojected_px_max", worst,
        "the traced lines of sight, projected back with camFile_track (pyopenlpt p4c)", "{:.2e}")
    say("rig.camera_heights_mm", [to_paper(np.array(c["centre_mm"]))[2] for c in res["cameras"]],
        "calibration_result.json cameras[].centre_mm, along the axis from the wand's mean height", "{:.1f}")
    data["rig_radii"] = np.array(radii)
    data["t0_ends"] = to_paper(np.stack([np.array(cd["ends_A_mm"], float)[fa], np.array(cd["ends_B_mm"], float)[fa]],
                                        axis=1)).astype(np.float32)

    # ---- the tracking (T1)
    trk_dir = PROC / "T1" / "Results" / "ConvergeTrack"
    local = CASE / "data" / "runs" / "s3_final_m15" / "Results" / "ConvergeTrack"
    same = all(sha256(f) == sha256(local / f.name) for f in sorted(trk_dir.glob("*.csv")))
    say("trk.deliverable_equals_local_run", same, "sha256 of T1/Results/ConvergeTrack/*.csv against data/runs/s3_final_m15")
    tracks, last = read_tracks(trk_dir)
    info = load(PROC / "T1" / "run_info.json")
    say("trk.build", info["build"], "T1/run_info.json build")
    say("trk.frames_processed", last + 1, "T1/Results/ConvergeTrack/*_<last>.csv: frames 0..last")
    say("trk.step_ms", 1000 * STEP_S, "every 4th raw frame at 4,000 frames/s", "{:.1f}")
    lens = np.array([len(t) for t in tracks.values()])
    say("trk.long_tracks", int(len(tracks)), "long tracks: active + inactive + exited at the last frame")
    say("trk.positions", int(lens.sum()), "their rows")
    say("trk.length_pcts_10_50_90", np.percentile(lens, [10, 50, 90]).tolist(), "positions per long track", "{:.0f}")
    say("trk.tracks_ge_100_200", [int((lens >= 100).sum()), int((lens >= 200).sum())], "long tracks with >= 100, 200 positions")
    per_frame = np.zeros(last + 1, int)
    for t in tracks.values():
        per_frame[t[:, 0].astype(int)] += 1
    say("trk.particles_per_frame_150_250", float(per_frame[150:251].mean()),
        "long-track positions per frame, mean over frames 150-250", "{:.0f}")
    say("trk.particles_at_sample_frame", int(per_frame[T1_FRAME]), f"long-track positions at frame {T1_FRAME}")
    chk = load(PROC / "T1" / "checks" / "t1_checks_s3_final_m15.json")
    rms = [chk[f"T1a_cam{c}"][f"rms_{a}"] for c in (1, 2) for a in ("du", "dv")]
    say("trk.image_residual_rms_px_range", [min(rms), max(rms)],
        "T1/checks/t1_checks_s3_final_m15.json T1a_cam<c>.rms_du/rms_dv (detection - the track's image)", "{:.2f}")
    say("trk.image_residual_points", chk["T1a_cam1"]["n"], "t1_checks_s3_final_m15.json T1a_cam1.n (6 sample frames)")
    say("trk.epipolar_mean_px", [chk["T1a_epipolar"]["mean"], chk["T1a_epipolar"]["sem"]],
        "t1_checks_s3_final_m15.json T1a_epipolar.mean, sem (the cameras did not move after T0)", "{:+.3f}")
    rr = load(STUDY / "r1_t1_rerun.json")["variants"]["m15"]
    assert rr["final"] == "s3_final_m15" and rr["long_tracks_total"] == len(tracks), "the rerun's record"
    say("trk.detections_per_frame", rr["detections_per_frame"],
        "tests/cylinder_stb_study/r1_t1_rerun.json variants.m15.detections_per_frame (cam1, cam2)", "{:.0f}")
    ghost = load(PROC / "T1" / "checks" / "t1e_ghost_control_s3_final_m15.json")
    say("ghost.counts_estimate_frame_100", ghost["ghost_share_upper_bound_last"],
        "T1/checks/t1e_ghost_control_s3_final_m15.json ghost_share_upper_bound_last (control / run, active long "
        "tracks at frame 100; cam2's images 100 steps late in the control)", "{:.3f}")
    say("ghost.swirl_estimate_frame_100", ghost["swirl"]["ghost_share_estimate"],
        "t1e_ghost_control_s3_final_m15.json swirl.ghost_share_estimate", "{:.3f}")
    gm = load(STUDY / "g3_control_Gallm.json")
    say("ghost.matched_synthetic_counts_vs_truth", [gm["counts_estimate_last"], gm["truth_ghost_share_tracks_last"]],
        "tests/cylinder_stb_study/g3_control_Gallm.json counts_estimate_last, truth_ghost_share_tracks_last "
        "(the synthetic case matched to T1)", "{:.3f}")

    # smoothed tracks (paper's frame), speeds, the swirl profile about the axis
    fw, pk, vk = gaussian_kernels(W_FILTER)
    tline = np.arange(40)[:, None] * np.array([[0.05, -0.02, 0.01]])
    _xs, vs_line = smooth(tline, W_FILTER, STEP_S)
    assert np.allclose(vs_line, np.array([0.05, -0.02, 0.01]) / STEP_S), "velocity kernel check"
    assert abs(pk.sum() - 1) < 1e-12, "position kernel check"
    say("smooth.filter_width_steps", W_FILTER, "the author (2026-10-08); processor.py's filter_width", "{:.0f}")
    say("smooth.half_width_steps", fw, "ceil(3 w): kernel support and the points dropped at each end")
    ids_s, fr_s, X_s, V_s, L_s = [], [], [], [], []
    for n, ((kind, tid), t) in enumerate(sorted(tracks.items())):
        if len(t) < 2 * fw + 1:
            continue
        xs, vs = smooth(t[:, 1:4], W_FILTER, STEP_S)
        ids_s.append(np.full(len(xs), n, np.int32))
        fr_s.append(t[fw:len(t) - fw, 0].astype(np.int16))
        X_s.append(to_paper(xs))
        V_s.append(to_paper(vs))
        L_s.append(np.full(len(xs), len(t), np.int32))
    ids_s, fr_s, X_s, V_s, L_s = (np.concatenate(a) for a in (ids_s, fr_s, X_s, V_s, L_s))
    say("smooth.tracks_kept", int(len(np.unique(ids_s))), f"long tracks with >= {2 * fw + 1} positions")
    sp = np.linalg.norm(V_s, axis=1)
    say("smooth.speed_pcts_10_50_90_mm_s", np.percentile(sp, [10, 50, 90]).tolist(), "smoothed speed, all points", "{:.0f}")
    r = np.hypot(X_s[:, 0], X_s[:, 1])
    th = np.arctan2(X_s[:, 1], X_s[:, 0])
    vr = V_s[:, 0] * np.cos(th) + V_s[:, 1] * np.sin(th)
    vt = -V_s[:, 0] * np.sin(th) + V_s[:, 1] * np.cos(th)
    sense = float(np.sign(np.median(vt)))
    vt = sense * vt
    say("swirl.sense_seen_from_plus_z", "counter-clockwise" if sense > 0 else "clockwise",
        "sign of the median azimuthal velocity in the paper's frame")
    say("swirl.radius_pcts_5_50_95_mm", np.percentile(r, [5, 50, 95]).tolist(), "distance from the axis, all points",
        "{:.1f}")
    edges = np.arange(0.0, 31.0, 3.0)
    q = {"vt": [], "vr": [], "vz": [], "n": [], "rc": []}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() < 200:
            continue
        q["rc"].append(float(np.median(r[m])))
        q["n"].append(int(m.sum()))
        for k, val in (("vt", vt), ("vr", vr), ("vz", V_s[:, 2])):
            q[k].append(np.percentile(val[m], [25, 50, 75]).tolist())
    say("swirl.bins_rmedian_mm", q["rc"], "3 mm bins of r with >= 200 points: median r", "{:.1f}")
    say("swirl.vtheta_median_mm_s", [x[1] for x in q["vt"]], "median azimuthal velocity per bin", "{:.1f}")
    say("swirl.vr_median_mm_s", [x[1] for x in q["vr"]], "median radial velocity per bin", "{:.1f}")
    say("swirl.vz_median_mm_s", [x[1] for x in q["vz"]], "median axial velocity per bin", "{:.1f}")
    omega = [x[1] / rc for x, rc in zip(q["vt"], q["rc"])]
    say("swirl.vtheta_over_r_per_s", omega, "median v_theta / median r per bin", "{:.2f}")
    say("swirl.vr_abs_median_over_vtheta_median_r_ge_6", max(abs(a[1]) / b[1] for a, b, rc in
                                                              zip(q["vr"], q["vt"], q["rc"]) if rc >= 6),
        "largest |median v_r| / median v_theta over the bins beyond 6 mm", "{:.2f}")
    say("swirl.vz_abs_median_over_vtheta_median_r_ge_6", max(abs(a[1]) / b[1] for a, b, rc in
                                                              zip(q["vz"], q["vt"], q["rc"]) if rc >= 6),
        "largest |median v_z| / median v_theta over the bins beyond 6 mm", "{:.2f}")
    data.update(trk_id=ids_s, trk_frame=fr_s, trk_xyz=X_s.astype(np.float32), trk_v=V_s.astype(np.float32),
                trk_len=L_s, prof_r=np.array(q["rc"]), prof_vt=np.array(q["vt"]), prof_vr=np.array(q["vr"]),
                prof_vz=np.array(q["vz"]), prof_n=np.array(q["n"]), prof_edges=edges)
    nl = len(np.unique(ids_s[L_s >= LONG]))
    say("fig.tracks_long", nl, f"smoothed tracks of >= {LONG} positions (before smoothing)")
    # the neighbour test (ghost_filter.py): the tracks it drops are left out of Fig. 17(c), not of anything else
    import ghost_filter as gf  # noqa: E402
    t1s = gf.smoothed(gf.read_long_tracks(trk_dir))
    drop = gf.dropped(gf.judge(t1s))
    keep_track = np.array([k not in drop for k in sorted(tracks)])
    data["trk_keep"] = keep_track[ids_s]
    long_ids = np.unique(ids_s[L_s >= LONG])
    n_drop_long = int(np.sum(~keep_track[long_ids]))
    say("ghost.neighbour_test_setting_rho_nmin_rth_eps", [gf.RHO, gf.N_MIN, gf.R_TH, gf.EPS],
        "ghost_filter.py (rho mm, neighbours, threshold, noise floor mm/s)", "{:g}")
    say("ghost.neighbour_test_dropped_of_smoothed", [len(drop), len(t1s)], "ghost_filter.judge / dropped on T1")
    say("ghost.neighbour_test_dropped_of_long", [n_drop_long, len(long_ids)], f"the same, tracks of >= {LONG} positions")
    say("fig.tracks_kept_neighbour", len(long_ids) - n_drop_long, f"tracks of >= {LONG} positions kept by the neighbour test")
    # the swirl test (display only: it assumes the flow): Fig. 17(c) draws the long tracks that pass both tests
    off = gf.off_swirl(t1s)
    swirl_track = np.array([k not in off for k in sorted(tracks)])
    data["trk_swirl"] = swirl_track[ids_s]
    both = keep_track[long_ids] & swirl_track[long_ids]
    say("ghost.swirl_test_setting_rmin_max_corermin", [gf.SWIRL_R_MIN, gf.SWIRL_MAX, gf.CORE_R_MIN],
        "ghost_filter.py SWIRL_R_MIN (mm), SWIRL_MAX, CORE_R_MIN (mm): T1-e's swirl test on the smoothed tracks",
        "{:g}")
    cc = gf.core_centre(t1s)
    say("ghost.core_centre_paper_xy_mm", [float(cc[1]), float(cc[0])],
        "ghost_filter.core_centre on T1 ((X, Z) of the run's frame = (y, x) of the paper's)", "{:.2f}")
    say("ghost.swirl_test_dropped_of_long_after_neighbour", [int(np.sum(keep_track[long_ids] & ~swirl_track[long_ids])),
                                                             len(long_ids) - n_drop_long],
        f"tracks of >= {LONG} positions that pass the neighbour test but not the swirl test")
    say("fig.tracks_drawn", int(both.sum()), f"tracks of >= {LONG} positions that pass both tests (Fig. 17(c))")
    for stem, run in (("ghost_filter_check", "Gallm_r2"), ("ghost_filter_check_Gall12m15_r2", "Gall12m15_r2"),
                      ("ghost_filter_check_Gall_r2", "Gall_r2")):
        row = load(HERE / f"{stem}.json")["swirl"]["neighbour test, then the swirl test"]
        say(f"ghost.both_tests_synthetic_{run}",
            [row["ghosts_dropped"] / row["ghosts"], row["real_dropped"] / row["real"], row["ghost_share_after"]],
            f"{stem}.json swirl: ghosts dropped, real tracks dropped, ghost share after", "{:.3f}")
    for stem, run in (("ghost_filter_check", "Gallm_r2"), ("ghost_filter_check_Gall12m15_r2", "Gall12m15_r2"),
                      ("ghost_filter_check_Gall_r2", "Gall_r2")):
        rows = load(HERE / f"{stem}.json")["grid"]
        row = next(r for r in rows if (r["rho"], r["r_th"], r["eps"]) == (gf.RHO, gf.R_TH, gf.EPS))
        say(f"ghost.neighbour_test_synthetic_{run}",
            [row["ghosts_dropped"] / row["ghosts"], row["real_dropped"] / row["real"], row["ghost_share_before"],
             row["ghost_share_after"]],
            f"{stem}.json at the setting: ghosts dropped, real tracks dropped, ghost share before, after", "{:.3f}")
    cams_xyz = [to_paper(np.array(c["centre_mm"])) for c in res["cameras"]]
    data["cam_xyz"] = np.array(cams_xyz)

    # the sample frame: raw and preprocessed crops, the line detections, the tracked particles and their trails
    sys.path.insert(0, os.environ.get("CYL_EXPERIMENT_PYD", str(REPO / "build-stb" / "p4c" / "out")))
    import pyopenlpt as lpt  # noqa: E402
    man = load(PROC / "T1" / "images" / "manifest.json")
    noise = [man["cameras"][f"cam{c}"]["noise_8bit_median"] for c in (1, 2)]
    cfg_run = load(PROC / "T1" / "config.json")
    say("trk.detector_noise", noise, "T1/images/manifest.json noise_8bit_median (as in T1/config.json)", "{:.3f}")
    assert np.allclose(noise, cfg_run["objects"][0]["tracer"]["lines"]["noise"]), "the run's detector noise"
    pts = {1: {}, 2: {}}
    for (kind, tid), t in tracks.items():
        for row in t:
            f = int(row[0])
            if T1_FRAME - TRAIL <= f <= T1_FRAME:
                for c in (1, 2):
                    pts[c].setdefault((kind, tid), []).append([f, row[4 + 2 * (c - 1)], row[5 + 2 * (c - 1)]])
    P = lpt.LineFinderParam()
    W, H = CROP
    for c in (1, 2):
        pre = read_image(PROC / "T1" / "images" / f"cam{c}" / f"frame{4 * T1_FRAME:06d}.tif")
        raw = read_image(RAW / "T1" / f"cam{c}" / f"frame{4 * T1_FRAME:06d}.tif")
        d = np.array(lpt.detectLines(lpt.Image.from_numpy(pre), float(noise[c - 1]), P))
        say(f"fig.t1_detections_cam{c}", int(len(d)), f"detectLines on T1/images/cam{c} frame {4 * T1_FRAME}")
        now = np.array([[v[-1][1], v[-1][2]] for v in pts[c].values() if v[-1][0] == T1_FRAME])
        # crop: the window with the most tracked particles (8 px inside its edges); ties: nearest the centre
        best = None
        for v0 in range(0, 800 - H + 1, 5):
            for u0 in range(0, 1280 - W + 1, 5):
                n_in = int(np.sum((now[:, 0] >= u0 + 8) & (now[:, 0] <= u0 + W - 8) &
                                  (now[:, 1] >= v0 + 8) & (now[:, 1] <= v0 + H - 8)))
                key = (n_in, -math.hypot(u0 + W / 2 - 640, v0 + H / 2 - 400))
                if best is None or key > best[0]:
                    best = (key, u0, v0)
        _k, u0, v0 = best
        say(f"fig.t1_crop_cam{c}_u0_v0_w_h", [u0, v0, W, H], "the crop rule above")
        say(f"fig.t1_tracked_in_crop_cam{c}", best[0][0], "tracked particles inside the crop")
        data[f"t1_crop{c}"] = np.array([u0, v0, W, H])
        data[f"t1_raw{c}"] = raw[v0:v0 + H, u0:u0 + W]
        data[f"t1_pre{c}"] = pre[v0:v0 + H, u0:u0 + W]
        sel = (d[:, 0] > u0 - 20) & (d[:, 0] < u0 + W + 20) & (d[:, 1] > v0 - 20) & (d[:, 1] < v0 + H + 20)
        data[f"t1_det{c}"] = d[sel][:, :6]
        tr_rows = []
        for n, v in enumerate(pts[c].values()):
            v = np.array(v)
            if v[-1, 0] != T1_FRAME:
                continue
            if not (u0 - 20 < v[-1, 1] < u0 + W + 20 and v0 - 20 < v[-1, 2] < v0 + H + 20):
                continue
            tr_rows.append(np.column_stack([np.full(len(v), n), v]))
        data[f"t1_trail{c}"] = np.concatenate(tr_rows) if tr_rows else np.zeros((0, 4))
    data["t1_frame"] = np.array(T1_FRAME)

    np.savez_compressed(HERE / "experiment_data.npz", **data)
    with open(HERE / "experiment_numbers.json", "w", encoding="utf-8") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    text = "\n".join(LINES)
    with open(HERE / "experiment_numbers_out.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print("wrote", HERE / "experiment_data.npz")


if __name__ == "__main__":
    main()
