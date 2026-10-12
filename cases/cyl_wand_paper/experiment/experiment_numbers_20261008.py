"""Every number that the paper's Section 9.3 (the experiment) and Supplement S8 quote, and the data of its two figures,
read from the results of run 20261008 (three cameras), each printed with its source.

The run is processed by the case pyOpenLPT/cases/cyl_20261008 (plan, report and checks there); its deliverables are on
X:\\Shijie Zhong\\Refraction\\Processed\\20261008 (T0: the wand calibration of the three cameras; T1: the tracks of the
first 250 tracer frames, every particle matched in all three cameras). Nothing is fitted or tracked afresh. Computed
here:
  - the tracks smoothed with the group's Gaussian filter (pyOpenLPT modules/post_processing/processor.py,
    compute_kinematics_and_filter: position kernel exp(-k^2/w^2), velocity kernel k exp(-k^2/w^2), both over
    |k| <= ceil(3 w), each normalised as there; the ceil(3 w) points at each end of a track are dropped), with w = 3
    steps (the author, 2026-10-08), and their speeds and swirl profile;
  - the line detections of one sample frame, with the library build the tracking used (build-stb/p4c), its default
    LineFinder settings and the noise the run gave it, as the case's tests/t1_checks.py does;
  - the lines of sight drawn in the rig figure, traced through the walls and checked against the library's cameras;
  - which crops and which wand frame the figures show (rules below).
Coordinates: the run's frame has the tube's axis along Y through the origin (at the wand's mean height, +Y up) and
camera 1 on +Z. The paper's frame has the axis along z and camera 1 on +x: (x, y, z)_paper = (Z, X, Y)_run (a cyclic
permutation, right-handed).

    python experiment_numbers_20261008.py   (refract environment; CYL_EXPERIMENT_PYD = a folder with p4c's .pyd)

Outputs beside this script: experiment_numbers_20261008_out.txt, experiment_numbers_20261008.json, and
experiment_data_20261008.npz, which code/make_paper_experiment_figs.py --refresh copies into the paper's figures/data/.
(experiment_numbers.py is the same for run 20261007, two cameras, which the paper quoted before 9 Oct.)
"""
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                            # pyOpenLPT
CASE = REPO / "cases" / "cyl_20261008"
PROC = Path(r"X:\Shijie Zhong\Refraction\Processed\20261008")
RAW = Path(r"X:\Shijie Zhong\Refraction\Raw\20261008")
RUN = CASE / "data" / "runs" / "s3_final_c3"                      # the final run (its Results = T1\Results on X:)
TAG = "s3_final_c3"

CAMS = (1, 2, 3)
W_FILTER = 3.0            # the Gaussian filter's width, in steps (the author)
STEP_S = 1 / 4000.0       # one step: every raw frame at 4,000 frames/s
T1_FRAME = 200            # the sample frame of the tracer figure (raw frame 200): 335 active long tracks by then
CROP = (200, 125)         # the image crops of the tracer figure, px (cams 1-2's sensor aspect 1.6)
LONG = 100                # "long" tracks of Table 3: at least 100 positions (25 ms)
DET_STEP = 25             # particles detected per frame: the detector on every 25th frame the run read
EDGE_IN = 20              # the sample wand frame: every ball at least this far inside its image, px

LINES, OUT = [], {}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def say(key, value, source, fmt="{}"):
    """One number: printed with its source, and kept under `key` in the JSON."""
    OUT[key] = value
    text = fmt.format(value) if not isinstance(value, (list, tuple)) else ", ".join(fmt.format(v) for v in value)
    LINES.append(f"{key:50s} {text:46s} <- {source}")


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
    """Long tracks (active, inactive, exited) at the last frame: {(kind, id): rows [frame, X, Y, Z, u1, v1, ..., u3, v3]}
    (camera 3's v in its cropped frame, rows 360-1575)."""
    files = glob.glob(str(folder / "*.csv"))
    last = max(int(Path(f).stem.rsplit("_", 1)[1]) for f in files)
    tracks = {}
    for kind in ("LongTrackActive", "LongTrackInactive", "ExitTrack"):
        with open(folder / f"{kind}_{last}.csv", newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd)
            for row in rd:
                if row:
                    tracks.setdefault((kind, int(row[0])), []).append([float(x) for x in row[1:11]])
    return {k: np.array(sorted(v)) for k, v in tracks.items()}, last


def first_loop_detections(log_path):
    """{frame: (cam1, cam2, cam3)}: the 2-D detections of each frame's first IPR loop, from the run's log."""
    out, frame = {}, None
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = re.search(r"phase at frame (\d+)", ln)
            if m:
                frame = int(m.group(1))
                continue
            m = re.search(r"2D detections per active camera: ([0-9 \t]+)", ln)
            if m and frame is not None and frame not in out:
                out[frame] = tuple(int(v) for v in m.group(1).split())
    return out


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
    cams_cfg = {c["id"]: c for c in cfg["cameras"]}
    data = {}

    # ---- the rig
    say("rig.cameras", [f'{cams_cfg[c]["model"]} {cams_cfg[c]["width"]}x{cams_cfg[c]["height"]} px of '
                        f'{1000 * cams_cfg[c]["pixel_mm"]:g} um' for c in CAMS],
        "cases/cyl_20261008/config.json cameras[] (the .chd headers; pixel sizes from the Phantom data sheets)")
    say("rig.lenses", cfg["_cameras"].split("The user (8 Oct): ")[1], "config.json _cameras (the user, 8 Oct)")
    say("rig.tube_od_mm", 25.4 * cfg["tube"]["outer_diameter_in"], "config.json tube.outer_diameter_in x 25.4", "{:.4f}")
    say("rig.tube_wall_mm", 25.4 * cfg["tube"]["wall_in"], "config.json tube.wall_in x 25.4", "{:.3f}")
    say("rig.radii_mm", cfg["tube"]["radii_mm"], "config.json tube.radii_mm", "{:.5f}")
    say("rig.n_air_acrylic_water", cfg["tube"]["n"], "config.json tube.n", "{:.3f}")
    say("rig.wand_length_mm", cfg["wand"]["length_mm"], "config.json wand.length_mm (the user: the wand of 6-7 Oct)",
        "{:.1f}")
    say("rig.wand_balls_mm", cfg["wand"]["ball_diameters_mm"], "config.json wand.ball_diameters_mm", "{:.0f}")
    h0 = p0a["T0/cam1"]
    say("rig.t0_frames_fps_exposure_us", [h0["frames"], h0["header"]["fps"], h0["header"]["exposure_us"]],
        "cases/cyl_20261008/tests/p0a_files.json T0/cam1 (cam2 the same)")
    say("rig.t0_seconds", h0["frames"] / h0["header"]["fps"], "frames / fps", "{:.1f}")
    say("rig.t0_first_image_per_camera", [p0a[f"T0/cam{c}"]["header"]["first_image_no"] for c in CAMS],
        "p0a_files.json T0/cam<c>.header.first_image_no (cam3 saved images 2396-13000)")
    h1 = p0a["T1/cam1"]
    say("rig.t1_frames_fps_exposure_us", [h1["frames"], h1["header"]["fps"], h1["header"]["exposure_us"]],
        "p0a_files.json T1/cam1 (all three alike, one trigger)")
    say("rig.t1_tracked_raw_frames", cfg["tracers"]["track_raw_frames"], "config.json tracers.track_raw_frames "
        "(the user, 8 Oct: the first 250 frames)")

    # ---- the calibration (T0)
    res = load(PROC / "T0" / "calibration_result.json")
    say("cal.model", res["model"], "T0/calibration_result.json model")
    say("cal.frames", res["frames"], "T0/calibration_result.json frames (kept by at least two cameras)")
    sel = load(PROC / "T0" / "detection" / "selection.json")
    say("cal.frames_kept_in_2_3_cameras", [sel["frames_kept_in_min_cameras"], sel["frames_kept_in_all_three"]],
        "T0/detection/selection.json frames_kept_in_min_cameras, frames_kept_in_all_three")
    say("cal.camera_frames_kept", [sel["cameras"][f"cam{c}"]["kept"] for c in CAMS],
        "selection.json cameras.<c>.kept (two clean ball fits)")
    say("cal.camera_frames_recorded", [sel["cameras"][f"cam{c}"]["camera_frames"] for c in CAMS],
        "selection.json cameras.<c>.camera_frames")
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
    say("cal.height_above_wand_mean_mm", [c["height_mm"] for c in res["cameras"]],
        "calibration_result.json cameras[].height_mm (the origin: the wand's mean height)", "{:.1f}")
    say("cal.height_above_camera1_mm", [c["height_mm"] - res["cameras"][0]["height_mm"] for c in res["cameras"]],
        "the same, against camera 1's", "{:.1f}")
    say("cal.looks_down_deg", [math.degrees(math.asin(-c["viewing_direction"][1])) for c in res["cameras"]],
        "calibration_result.json cameras[].viewing_direction: its angle below the plane across the axis", "{:.1f}")
    say("cal.f_px_(not_quoted)", [c["f_px"] for c in res["cameras"]],
        "calibration_result.json cameras[].f_px; f and distance trade off (case report, C1-T)", "{:.0f}")
    say("cal.f_mm_(not_quoted)", [c["f_mm"] for c in res["cameras"]], "calibration_result.json cameras[].f_mm", "{:.1f}")
    say("cal.wand_length_fixed_mm", [res["wand_length_mm"], res["fit_L"]], "calibration_result.json wand_length_mm, fit_L")
    wl = res["wand_length_free_triangulation"]
    say("cal.free_wand_length_frames", wl["frames"], "calibration_result.json wand_length_free_triangulation.frames")
    say("cal.free_wand_length_rms_um", 1000 * wl["rms_mm"], "... .rms_mm (the mean is held by the fit)", "{:.1f}")
    say("cal.free_wand_length_mean_um", 1000 * wl["mean_mm"], "... .mean_mm", "{:+.1f}")
    say("cal.free_wand_length_rms_pct", 100 * wl["rms_mm"] / res["wand_length_mm"], "... .rms_mm / wand_length_mm",
        "{:.2f}")

    # the checks of the case (C1-H, residual maps, C1-T)
    c1h = load(CASE / "tests" / "c1h_third_camera.json")
    say("cal.held_out_camera_frames", c1h["frames"], "cases/cyl_20261008/tests/c1h_third_camera.json frames "
        "(kept by all three cameras)")
    say("cal.held_out_camera_rms_px", [c1h["held_out"][f"cam{c}"]["rms_px"] for c in CAMS],
        "c1h_third_camera.json held_out.<c>.rms_px (each camera's balls from the other two's triangulation)", "{:.2f}")
    say("cal.held_out_camera_median_px", [c1h["held_out"][f"cam{c}"]["median_px"] for c in CAMS],
        "c1h_third_camera.json held_out.<c>.median_px", "{:.2f}")
    c1r = load(CASE / "tests" / "c1_residuals.json")
    say("cal.residual_cell_max_px", [c1r["cameras"][f"cam{c}"]["largest_cell"]["mean_px"] for c in CAMS],
        "c1_residuals.json cameras.<c>.largest_cell.mean_px (10 x 6 cells)", "{:.2f}")
    say("cal.residual_cell_median_px", [float(np.median([q["mean_px"] for q in c1r["cameras"][f"cam{c}"]["cells"]]))
                                        for c in CAMS],
        "c1_residuals.json cameras.<c>.cells[].mean_px, median over the cells", "{:.2f}")
    sl = c1r["wand_length"]["slopes"]["X"]
    say("cal.wand_length_slope_across_um_per_mm_z", [sl["um_per_mm"], sl["z"]],
        "c1_residuals.json wand_length.slopes.X (run X = paper y: across the axis, across camera 1's view)", "{:.1f}")
    c1t = load(CASE / "tests" / "c1t_variants.json")
    whole = c1t["S2"]["cameras"]
    assert abs(whole[0]["f_px"] - res["cameras"][0]["f_px"]) < 1e-6, "C1-T's reference is the delivered fit"
    parts = [k for k in ("h1", "h2", "q1", "q2", "q3", "q4") if k in c1t["variants"]]
    say("cal.parts", parts, "cases/cyl_20261008/tests/c1t_variants.json variants (halves and quarters of images "
        "2396-13000)")
    for c in CAMS:
        fs = [c1t["variants"][k]["summary"]["cameras"][c - 1]["f_px"] for k in parts]
        fd = [c1t["variants"][k]["summary"]["cameras"][c - 1]["f_px"] /
              c1t["variants"][k]["summary"]["cameras"][c - 1]["distance_mm"] for k in parts]
        f0, fd0 = whole[c - 1]["f_px"], whole[c - 1]["f_px"] / whole[c - 1]["distance_mm"]
        say(f"cal.parts_f_rel_pct_cam{c}", [100 * (min(fs) / f0 - 1), 100 * (max(fs) / f0 - 1)],
            "c1t_variants.json <part>.summary.cameras[].f_px against the whole recording's", "{:+.1f}")
        say(f"cal.parts_f_over_distance_rel_pct_cam{c}", [100 * (min(fd) / fd0 - 1), 100 * (max(fd) / fd0 - 1)],
            "the same for f_px / distance_mm", "{:+.2f}")

    # the wand images: the detections of the frames in the fit
    cd = load(PROC / "T0" / "calibration_data.json")
    active = {f for f, a in zip(cd["frame_ids"], cd["frame_active"]) if a}
    det = {}
    with open(PROC / "T0" / "detection" / "detections_all.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["dropped"] == "0":                                   # a ball kept by the guard ("": no ball found)
                det.setdefault((int(r["frame"]), int(r["camera"])), []).append(r)
    ratio = {c: [] for c in CAMS}
    for f in active:
        for c in CAMS:
            rows = det.get((f, c), [])
            if len(rows) == 2:
                ratio[c].extend(float(r["a"]) / float(r["b"]) for r in rows)
    say("cal.ball_image_aspect_median", [float(np.median(ratio[c])) for c in CAMS],
        "T0/detection/detections_all.csv a / b, the balls of the frames in the fit, per camera", "{:.3f}")
    size = {c: (cams_cfg[c]["width"], cams_cfg[c]["height"]) for c in CAMS}
    # sample wand frame: in the fit and kept by all three cameras, every ball at least EDGE_IN px inside its image;
    # among the 10 frames whose wand midpoint image is nearest the image centre (in units of the image's half-diagonal,
    # summed over the cameras), the one whose image is longest in camera 1 (the whole wand in view)
    cands = []
    for f in sorted(active):
        ok, score = True, 0.0
        for c in CAMS:
            rows = det.get((f, c), [])
            W, H = size[c]
            if len(rows) != 2:
                ok = False
                break
            us, vs, a = ([float(r[k]) for r in rows] for k in ("u", "v", "a"))
            if (min(us) - max(a) < EDGE_IN or max(us) + max(a) > W - EDGE_IN or min(vs) - max(a) < EDGE_IN
                    or max(vs) + max(a) > H - EDGE_IN):
                ok = False
                break
            score += math.hypot(np.mean(us) - (W - 1) / 2, np.mean(vs) - (H - 1) / 2) / math.hypot(W / 2, H / 2)
        if ok:
            cands.append((score, f))
    cands.sort()
    top = [f for _s, f in cands[:10]]
    f0 = max(top, key=lambda f: math.hypot(float(det[(f, 1)][0]["u"]) - float(det[(f, 1)][1]["u"]),
                                           float(det[(f, 1)][0]["v"]) - float(det[(f, 1)][1]["v"])))
    say("fig.wand_frame_image_number", f0, "the rule above, over T0's detections_all.csv and calibration_data.json")
    say("fig.wand_frame_candidates", len(cands), "frames in the fit that all three cameras keep with the balls inside")
    data["t0_frame"] = np.array(f0)
    for c in CAMS:
        k = f0 - cams_cfg[c]["first_image"]["T0"]
        img = read_image(RAW / "T0" / f"cam{c}" / f"frame{k:06d}.tif")
        assert img.shape == (size[c][1], size[c][0]), "image size"
        data[f"t0_img{c}"] = img
        data[f"t0_ell{c}"] = np.array([[float(r[q]) for q in ("u", "v", "a", "b", "theta_rad")] for r in det[(f0, c)]])
    ob = cd["observations"]
    cam = np.array(ob["cam_id"])
    act = np.array(ob["active"], bool)
    uv = np.column_stack([ob["u"], ob["v"]]).astype(float)
    ids = sorted(set(cam.tolist()))
    assert len(ids) == 3, "three cameras"
    for c, cid in zip(CAMS, ids):
        data[f"t0_cov{c}"] = uv[(cam == cid) & act].astype(np.float32)
    say("fig.wand_detections_used_per_camera", [int(len(data[f"t0_cov{c}"])) for c in CAMS],
        "calibration_data.json observations, active, per camera")
    assert [int(len(data[f"t0_cov{c}"])) for c in CAMS] == [c["residuals"]["n_active"] for c in res["cameras"]]
    wlf = np.array(cd["wand_length_free_mm"], float)
    fa = np.array(cd["frame_active"], bool)
    wdir = (np.array(cd["ends_B_mm"], float) - np.array(cd["ends_A_mm"], float))[fa]
    wdir /= np.linalg.norm(wdir, axis=1)[:, None]
    tilt = np.degrees(np.arcsin(np.abs(wdir[:, 1])))          # run frame: the axis is Y
    say("cal.wand_tilt_from_cross_section_deg_50_90_max", [float(np.median(tilt)), float(np.percentile(tilt, 90)),
                                                           float(tilt.max())],
        "calibration_data.json ends_A_mm / ends_B_mm, active frames: angle between the wand and the plane across the "
        "axis", "{:.1f}")
    ends = np.stack([np.array(cd["ends_A_mm"], float)[fa], np.array(cd["ends_B_mm"], float)[fa]], axis=1)
    rr = np.hypot(ends[..., 0], ends[..., 2]).ravel()
    say("cal.wand_ends_r_mm_95_max", [float(np.percentile(rr, 95)), float(rr.max())],
        "the same ends: distance from the axis", "{:.1f}")
    say("cal.wand_ends_z_mm_5_95", np.percentile(ends[..., 1].ravel(), [5, 95]).tolist(),
        "the same ends: along the axis from their mean height", "{:+.1f}")
    data["t0_len_um"] = (1000 * (wlf[fa & np.isfinite(wlf)] - res["wand_length_mm"])).astype(np.float32)
    say("fig.wand_length_values", int(len(data["t0_len_um"])), "calibration_data.json wand_length_free_mm, active frames")
    e = data["t0_len_um"].astype(float)
    say("fig.wand_length_rms_um_check", float(np.sqrt(np.mean(e ** 2))),
        "the same values (cf. cal.free_wand_length_rms_um)", "{:.1f}")
    say("fig.wand_length_sd_um_and_excess_kurtosis",
        [float(e.std()), float(np.mean((e - e.mean()) ** 4) / e.var() ** 2 - 3)],
        "the same values: standard deviation, excess kurtosis (0 for a Gaussian)", "{:.2f}")
    say("fig.wand_length_share_within_1_2_3_sd", [float(np.mean(np.abs(e - e.mean()) <= k * e.std())) for k in (1, 2, 3)],
        "the same values (a Gaussian: 0.683, 0.954, 0.997)", "{:.3f}")
    data["t0_ends"] = to_paper(ends).astype(np.float32)

    # the calibrated rig (Fig. 16(a)): each camera's lines of sight through the left and right edges and the centre of
    # its image's centre row, and through its image's centre and four corners, traced through the walls; a point a third
    # of the way across the water on each must project back to its pixel with the library's camera (camFile_track)
    sys.path.insert(0, os.environ.get("CYL_EXPERIMENT_PYD", str(REPO / "build-stb" / "p4c" / "out")))
    import pyopenlpt as lpt  # noqa: E402
    radii, nidx = res["tube"]["radii_mm"], res["tube"]["n"]
    worst = 0.0
    for c, cam in zip(CAMS, res["cameras"]):
        R, t = np.array(cam["R"], float), np.array(cam["tvec"], float)
        C = -R.T @ t
        assert np.allclose(C, cam["centre_mm"], atol=1e-6), "camera centre"
        cpp = lpt.loadCameraModel(str(PROC / "T0" / "camFile_track" / f"cam{c}.txt"))
        W, H = cam["width"], cam["height"]
        pix = [(0.0, cam["cy"]), (W - 1.0, cam["cy"]), (cam["cx"], cam["cy"]),
               (0.0, 0.0), (W - 1.0, 0.0), (W - 1.0, H - 1.0), (0.0, H - 1.0)]
        rays = []
        for u, v in pix:
            pts = trace(C, R.T @ np.array([(u - cam["cx"]) / cam["f_px"], (v - cam["cy"]) / cam["f_px"], 1.0]), radii,
                        nidx)
            X = pts[-2] + (pts[-1] - pts[-2]) / 3.0
            r = cpp.projectBatchStatus([lpt.Pt3D(*map(float, X))])[0]
            ok, uvp = r[0], r[1]
            assert ok, "the traced point has no image"
            worst = max(worst, math.hypot(uvp[0] - u, uvp[1] - v))
            rays.append(to_paper(pts))
        # rows: the centre row's left, right and centre, then the image's corners (top left, clockwise); each ray:
        # the pinhole, the outer wall, the inner wall, the far side of the water
        data[f"rig_rays{c}"] = np.array(rays)
        data[f"cam_axes{c}"] = to_paper(R)                         # rows: image x, image y, optical axis
    say("rig.traced_rays_reprojected_px_max", worst,
        "the traced lines of sight, projected back with camFile_track (pyopenlpt p4c)", "{:.2e}")
    data["rig_radii"] = np.array(radii)
    data["cam_xyz"] = np.array([to_paper(np.array(c["centre_mm"])) for c in res["cameras"]])
    data["cam_wh_f"] = np.array([[c["width"], c["height"], c["f_px"]] for c in res["cameras"]])

    # ---- the tracking (T1)
    trk_dir = PROC / "T1" / "Results" / "ConvergeTrack"
    local = RUN / "Results" / "ConvergeTrack"
    same = all(sha256(f) == sha256(local / f.name) for f in sorted(trk_dir.glob("*.csv")))
    say("trk.deliverable_equals_local_run", same, f"sha256 of T1/Results/ConvergeTrack/*.csv against data/runs/{TAG}")
    tracks, last = read_tracks(local)
    info = load(PROC / "T1" / "run_info.json")
    say("trk.build", info["build"], "T1/run_info.json build")
    say("trk.frames_processed", last + 1, "Results/ConvergeTrack/*_<last>.csv: frames 0..last")
    say("trk.step_ms", 1000 * STEP_S, "every raw frame at 4,000 frames/s", "{:.2f}")
    s3 = load(CASE / "tests" / "s3_track_final_c3.json")
    say("trk.ipr_all_three_cameras_tol_px", [s3["ipr"]["reduced_cameras"], s3["ipr"]["tol_2d_px"], s3["ipr"]["loops"]],
        "tests/s3_track_final_c3.json ipr.reduced_cameras (0: every match in all three), tol_2d_px, loops")
    say("trk.depth_cue_tol_px_rel", [s3["tracer"]["shape_tol_px"], s3["tracer"]["shape_tol_rel"]],
        "s3_track_final_c3.json tracer.shape_tol_px, shape_tol_rel (the three image widths must agree)")
    log_x = PROC / "T1" / "log.txt"
    say("trk.log_deliverable_equals_local_run", sha256(log_x) == sha256(RUN / "log.txt"),
        f"sha256 of T1/log.txt against data/runs/{TAG}/log.txt")
    fl = first_loop_detections(RUN / "log.txt")
    assert sorted(fl) == list(range(250)), "a detection line for every frame"
    det1 = np.array([fl[f] for f in range(250)], float)
    say("trk.first_ipr_loop_detections_mean_(not_quoted)", det1.mean(axis=0).tolist(),
        f"data/runs/{TAG}/log.txt: each frame's first IPR loop, mean over frames 0-249; from frame 4 on, IPR detects "
        "on the residual image (the tracked particles' images removed), so these undercount", "{:.0f}")
    lens = np.array([len(t) for t in tracks.values()])
    say("trk.long_tracks", int(len(tracks)), "long tracks: active + inactive + exited at the last frame")
    say("trk.positions", int(lens.sum()), "their rows")
    say("trk.length_pcts_10_50_90", np.percentile(lens, [10, 50, 90]).tolist(), "positions per long track", "{:.0f}")
    say("trk.length_min_max", [int(lens.min()), int(lens.max())], "positions per long track")
    say("trk.tracks_ge_100_200", [int((lens >= LONG).sum()), int((lens >= 200).sum())],
        "long tracks with >= 100, 200 positions")
    say("trk.tracks_full_length", int((lens == last + 1).sum()), "long tracks spanning every frame")
    per_frame = np.zeros(last + 1, int)
    for t in tracks.values():
        per_frame[t[:, 0].astype(int)] += 1
    say("trk.particles_per_frame_150_249", float(per_frame[150:last + 1].mean()),
        "long-track positions per frame, mean over frames 150-249", "{:.0f}")
    say("trk.particles_per_frame_at_50_100_149_200_249", [int(per_frame[f]) for f in (50, 100, 149, 200, 249)],
        "long-track positions at those frames")
    chk = load(PROC / "T1" / "checks" / f"t1_checks_{TAG}.json")
    rms = [chk[f"T1a_cam{c}"][f"rms_{a}"] for c in CAMS for a in ("du", "dv")]
    say("trk.image_residual_rms_px", rms, f"T1/checks/t1_checks_{TAG}.json T1a_cam<c>.rms_du/rms_dv "
        "(detection - the track's image; cam1 du, dv, cam2, cam3)", "{:.2f}")
    say("trk.image_residual_rms_px_range", [min(rms), max(rms)], "the same: range", "{:.2f}")
    say("trk.image_residual_points", [chk[f"T1a_cam{c}"]["n"] for c in CAMS], f"t1_checks_{TAG}.json T1a_cam<c>.n "
        "(the sample frames' long-track points with a detection within 2 px)")
    say("trk.speed_stride_mm_s_10_50_90", chk["T1d"]["speed_mm_s_pcts"], "t1_checks T1d.speed_mm_s_pcts (over 20 "
        "frames, unsmoothed)", "{:.0f}")
    cmp_ = load(PROC / "T1" / "checks" / "t1_compare.json")["today"]
    gf, gp = cmp_["ghost_estimate_final"], cmp_["ghost_estimate_pair12"]
    say("ghost.control_vs_final_active_long_at_50_100_149", [[gf[f]["control"], gf[f]["final"]] for f in gf],
        "T1/checks/t1_compare.json today.ghost_estimate_final (control: cam2 50 and cam3 100 frames late)")
    say("ghost.bound_three_cameras_at_50_100_149", [gf[f]["ghost_share_upper"] for f in gf],
        "the same: control / final", "{:.3f}")
    say("ghost.bound_cams12_only_at_50_100_149", [gp[f]["ghost_share_upper"] for f in gp],
        "t1_compare.json today.ghost_estimate_pair12 (the same tracking with cams 1-2 alone, its own control)",
        "{:.3f}")
    say("ghost.cams12_only_vs_control_active_long_at_50_100_149", [[gp[f]["control"], gp[f]["final"]] for f in gp],
        "the same: control, run")
    t1e = load(PROC / "T1" / "checks" / f"t1e_ghost_control_{TAG}.json")
    say("ghost.swirl_estimate_(rough)", t1e["swirl"]["ghost_share_estimate"],
        f"t1e_ghost_control_{TAG}.json swirl.ghost_share_estimate (16 control tracks)", "{:.3f}")
    say("trk.position_noise_um_median", t1e["stats_final"]["jitter_um_pcts"][1],
        f"t1e_ghost_control_{TAG}.json stats_final.jitter_um_pcts (median)", "{:.1f}")
    man = load(PROC / "T1" / "images" / "manifest.json")
    say("trk.preprocess_median_px", [man["median_px"][f"cam{c}"] for c in CAMS], "T1/images/manifest.json median_px")
    say("trk.preprocess_method", man["method"], "T1/images/manifest.json method")
    say("trk.cam3_rows", man["cam3_crop_rows"], "T1/images/manifest.json cam3_crop_rows (end exclusive)")

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
        assert np.all(np.diff(t[:, 0]) == 1), "a track with a gap"
        xs, vs = smooth(t[:, 1:4], W_FILTER, STEP_S)
        ids_s.append(np.full(len(xs), n, np.int32))
        fr_s.append(t[fw:len(t) - fw, 0].astype(np.int16))
        X_s.append(to_paper(xs))
        V_s.append(to_paper(vs))
        L_s.append(np.full(len(xs), len(t), np.int32))
    ids_s, fr_s, X_s, V_s, L_s = (np.concatenate(a) for a in (ids_s, fr_s, X_s, V_s, L_s))
    say("smooth.tracks_kept", int(len(np.unique(ids_s))), f"long tracks with >= {2 * fw + 1} positions (all drawn in "
        "Fig. 17)")
    say("smooth.positions_kept", int(len(ids_s)), "their smoothed positions")
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
    say("swirl.z_pcts_5_95_mm", np.percentile(X_s[:, 2], [5, 95]).tolist(), "along the axis, all points", "{:+.1f}")
    edges = np.arange(0.0, 34.0, 3.0)
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
    say("swirl.bins_points", q["n"], "points per bin")
    say("swirl.vtheta_median_mm_s", [x[1] for x in q["vt"]], "median azimuthal velocity per bin", "{:.1f}")
    say("swirl.vr_median_mm_s", [x[1] for x in q["vr"]], "median radial velocity per bin", "{:.1f}")
    say("swirl.vz_median_mm_s", [x[1] for x in q["vz"]], "median axial velocity per bin", "{:.1f}")
    say("swirl.vtheta_over_r_per_s", [x[1] / rc for x, rc in zip(q["vt"], q["rc"])], "median v_theta / median r per bin",
        "{:.1f}")
    say("swirl.vr_abs_median_over_vtheta_median", [abs(a[1]) / b[1] for a, b in zip(q["vr"], q["vt"])],
        "|median v_r| / median v_theta per bin", "{:.2f}")
    say("swirl.vz_abs_median_over_vtheta_median", [abs(a[1]) / b[1] for a, b in zip(q["vz"], q["vt"])],
        "|median v_z| / median v_theta per bin", "{:.2f}")
    # the vortex core's centre (not quoted unless needed): the point to which the velocities of the points within 8 mm of
    # the axis are most nearly perpendicular, by least squares on (p - c) . v_hat = 0 in the cross-section
    m = r < 8.0
    vh = V_s[m, :2] / np.linalg.norm(V_s[m, :2], axis=1)[:, None]
    cc = np.linalg.lstsq(vh, np.sum(vh * X_s[m, :2], axis=1), rcond=None)[0]
    say("swirl.core_centre_xy_mm_(not_quoted)", cc.tolist(), "least squares, points within 8 mm of the axis", "{:+.2f}")
    data.update(trk_id=ids_s, trk_frame=fr_s, trk_xyz=X_s.astype(np.float32), trk_v=V_s.astype(np.float32),
                trk_len=L_s, prof_r=np.array(q["rc"]), prof_vt=np.array(q["vt"]), prof_vr=np.array(q["vr"]),
                prof_vz=np.array(q["vz"]), prof_n=np.array(q["n"]), prof_edges=edges)

    # the sample frame: raw crops and the line detections
    run_cfg = load(RUN / f"config_final_c3.json")
    noise = run_cfg["objects"][0]["tracer"]["lines"]["noise"]
    say("trk.detector_noise", noise, f"data/runs/{TAG}/config_final_c3.json objects[0].tracer.lines.noise", "{:.3f}")
    lists = []
    for lst in run_cfg["image_lists"]:
        with open(lst, encoding="utf-8") as fh:
            lists.append([ln for ln in fh.read().splitlines() if ln.strip()])
    P = lpt.LineFinderParam()
    say("trk.detector_k", P.k_detect, "LineFinderParam() default k_detect (the run's)", "{:g}")
    row0 = {1: 0, 2: 0, 3: int(man["cam3_crop_rows"][0])}
    # particles detected per frame: the detector on the whole frames the run read, every DET_STEP-th frame (in the
    # initial phase, frame 0, IPR detects on the whole frame too, so its count must equal the log's)
    counts = []
    for f in range(0, 250, DET_STEP):
        counts.append([len(lpt.detectLines(lpt.Image.from_numpy(read_image(lists[c - 1][f])), float(noise[c - 1]), P))
                       for c in CAMS])
    assert tuple(counts[0]) == fl[0], "the detector reproduces the run's at frame 0"
    say("trk.detections_per_frame", np.mean(counts, axis=0).tolist(),
        f"detectLines on the run's frames 0, {DET_STEP}, ..., {250 - DET_STEP} (whole frames; cam3: rows 360-1575), "
        "mean", "{:.0f}")
    say("trk.detections_per_frame_range", [[min(x[c] for x in counts), max(x[c] for x in counts)] for c in range(3)],
        "the same frames: range per camera")
    W, H = CROP
    for c in CAMS:
        pre = read_image(lists[c - 1][T1_FRAME])
        raw = read_image(CASE / "data" / "T1raw" / f"cam{c}" / f"frame{T1_FRAME:06d}.tif")
        raw_x = RAW / "T1" / f"cam{c}" / f"frame{T1_FRAME:06d}.tif"
        assert sha256(raw_x) == sha256(CASE / "data" / "T1raw" / f"cam{c}" / f"frame{T1_FRAME:06d}.tif"), "raw copy"
        d = np.array(lpt.detectLines(lpt.Image.from_numpy(pre), float(noise[c - 1]), P))
        say(f"fig.t1_detections_cam{c}", [int(len(d)), int(fl[T1_FRAME][c - 1])],
            f"detectLines on the run's frame {T1_FRAME} against the run's first IPR loop")
        d[:, 1] += row0[c]                                            # cam3: the full frame's rows
        now = np.array([[t[t[:, 0] == T1_FRAME][0, 2 + 2 * c], t[t[:, 0] == T1_FRAME][0, 3 + 2 * c] + row0[c]]
                        for t in tracks.values() if np.any(t[:, 0] == T1_FRAME)])
        # crop: the window with the most tracked particles (8 px inside its edges) inside the rows the run read;
        # ties: nearest the image centre
        Wc, Hc = cams_cfg[c]["width"], cams_cfg[c]["height"]
        v_lo, v_hi = (row0[c], int(man["cam3_crop_rows"][1])) if c == 3 else (0, Hc)
        best = None
        for v0 in range(v_lo, v_hi - H + 1, 5):
            for u0 in range(0, Wc - W + 1, 5):
                n_in = int(np.sum((now[:, 0] >= u0 + 8) & (now[:, 0] <= u0 + W - 8) &
                                  (now[:, 1] >= v0 + 8) & (now[:, 1] <= v0 + H - 8)))
                key = (n_in, -math.hypot(u0 + W / 2 - Wc / 2, v0 + H / 2 - Hc / 2))
                if best is None or key > best[0]:
                    best = (key, u0, v0)
        _k, u0, v0 = best
        say(f"fig.t1_crop_cam{c}_u0_v0_w_h", [u0, v0, W, H], "the crop rule above (full-frame pixels)")
        say(f"fig.t1_tracked_in_crop_cam{c}", best[0][0], "tracked particles inside the crop")
        data[f"t1_crop{c}"] = np.array([u0, v0, W, H])
        data[f"t1_raw{c}"] = raw[v0:v0 + H, u0:u0 + W]
        sel_d = (d[:, 0] > u0 - 20) & (d[:, 0] < u0 + W + 20) & (d[:, 1] > v0 - 20) & (d[:, 1] < v0 + H + 20)
        data[f"t1_det{c}"] = d[sel_d][:, :6]
        say(f"fig.t1_detections_in_crop_cam{c}", int(np.sum((d[:, 0] >= u0) & (d[:, 0] < u0 + W) & (d[:, 1] >= v0) &
                                                            (d[:, 1] < v0 + H))), "detections centred in the crop")
    data["t1_frame"] = np.array(T1_FRAME)

    np.savez_compressed(HERE / "experiment_data_20261008.npz", **data)
    with open(HERE / "experiment_numbers_20261008.json", "w", encoding="utf-8") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    text = "\n".join(LINES)
    with open(HERE / "experiment_numbers_20261008_out.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print("wrote", HERE / "experiment_data_20261008.npz")


if __name__ == "__main__":
    main()
