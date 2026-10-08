"""C4 (plan: 2026-10-07-plan.md, Phase C): the camFiles for tracking.

    python tests/c4_track_camfiles.py [CAMFILE_FOLDER]      (default: <out_dir>/camFile, the delivered calibration)

The wand calibration writes its camFiles with `Projection Tolerance` 1e-12 (cyl_model.to_cpp's default).  At that
tolerance a point in a camera's near/far junction plane (the plane through the axis and the camera's centre, where a
ray grazes the point's own radius) can get no image: rounding limits the residual there to about 1e-12-4e-11, a
rounding-level effect that the user decided not to fix (memory: rigor-level-decision).  On 7 Oct 3 of 40,000 central
projections of 6 Oct T2's cameras failed that way; at the C++ default, 1e-9, none did, with the same pixels.  A
tracking camera that silently loses a particle loses a 2-camera track, so the tracking copies say 1e-9.

Writes <out_dir>/camFile_track/cam1.txt and cam2.txt: the delivered files with the value line after
"# Projection Tolerance:" replaced by config.tracking_camfiles.proj_tol, every other byte unchanged.  Checks, on
points of the volume the wand filled (the bounding box of the wand ends of calibration_data.json, widened by 5 mm,
inside the water) and on points in each camera's junction plane:
  - every other line of each file is unchanged;
  - both files load in pyOpenLPT and report PINCYL;
  - where both versions give an image, the pixels agree (largest difference);
  - how many points get an image from one version only.
Writes c4_track_camfiles_out.txt and c4_track_camfiles.json beside this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

import numpy as np  # noqa: E402

N_BOX = 20000
N_PLANE = 4000
SEED = 20261007


def retolerance(src: str, dst: str, tol: float) -> dict:
    with open(src, "rb") as fh:
        raw = fh.read()
    eol = b"\r\n" if b"\r\n" in raw else b"\n"
    lines = raw.split(eol)
    hits = [k for k, ln in enumerate(lines) if ln.strip().startswith(b"# Projection Tolerance")]
    if len(hits) != 1 or hits[0] + 1 >= len(lines):
        raise RuntimeError(f"{src}: expected one '# Projection Tolerance' line, found {len(hits)}")
    k = hits[0] + 1
    old = lines[k].decode("ascii").strip()
    lines[k] = f"{tol:.16e}".encode("ascii")
    out = eol.join(lines)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(out)
    with open(dst, "rb") as fh:
        back = fh.read().split(eol)
    changed = [i for i, (a, b) in enumerate(zip(raw.split(eol), back)) if a != b]
    return {"old_value": old, "new_value": lines[k].decode("ascii"), "changed_lines": changed,
            "n_lines": len(back), "same_length": len(back) == len(raw.split(eol))}


def project(lpt, cam, X):
    pts = [lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])) for p in X]
    res = cam.projectBatchStatus(pts, False)
    uv = np.full((len(X), 2), np.nan)
    for i, (ok, pt, _msg) in enumerate(res):
        if ok:
            uv[i] = (pt[0], pt[1])
    return uv


def main():
    cfg = case.load_config()
    src_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(cfg["out_dir"], "camFile")
    tc = cfg["tracking_camfiles"]
    dst_dir = os.path.join(cfg["out_dir"], tc["folder"])
    tol = float(tc["proj_tol"])
    log = case.Log(os.path.join(HERE, "c4_track_camfiles"))
    lib = case.library()
    lpt = lib.lpt
    log(f"C4: {src_dir} -> {dst_dir}, projection tolerance {tol:g}")
    out = {"source": src_dir, "target": dst_dir, "tol": tol, "files": {}, "checks": {}}
    names = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".txt"))
    cams = {}
    for name in names:
        info = retolerance(os.path.join(src_dir, name), os.path.join(dst_dir, name), tol)
        out["files"][name] = info
        log(f"  {name}: tolerance {info['old_value']} -> {info['new_value']}; lines changed {info['changed_lines']} "
            f"of {info['n_lines']}")
        a = lpt.loadCameraModel(os.path.join(src_dir, name))
        b = lpt.loadCameraModel(os.path.join(dst_dir, name))
        cams[name] = (a, b)
        log(f"    both load: {a.type()} / {b.type()}")

    with open(os.path.join(cfg["out_dir"], "calibration_data.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    ends = np.vstack([np.array(data["ends_A_mm"], float), np.array(data["ends_B_mm"], float)])
    ends = ends[np.all(np.isfinite(ends), axis=1)]
    lo, hi = ends.min(axis=0) - 5.0, ends.max(axis=0) + 5.0
    rng = np.random.default_rng(SEED)
    X = lo + (hi - lo) * rng.random((N_BOX, 3))
    r_in = float(cfg["tube"]["radii_mm"][-1])
    X = X[np.hypot(X[:, 0], X[:, 2]) < r_in - 1e-6]          # inside the water (the axis is Y in this frame)
    log(f"  box points: {len(X)} inside the water, box {np.round(lo, 1).tolist()} - {np.round(hi, 1).tolist()} mm")
    sets = {"box": X}
    with open(os.path.join(cfg["out_dir"], "calibration_result.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    for c in res["cameras"]:
        # the plane through the axis (Y) and the camera's centre: points X = s * e_r + y * e_y
        C = np.asarray(c["centre_mm"], float)
        er = np.array([C[0], 0.0, C[2]]) / np.hypot(C[0], C[2])
        s = rng.uniform(-r_in + 1.0, r_in - 1.0, N_PLANE)
        y = rng.uniform(lo[1], hi[1], N_PLANE)
        P = s[:, None] * er[None, :] + y[:, None] * np.array([0.0, 1.0, 0.0])[None, :]
        sets[f"junction plane of {c['name']}"] = P
    for set_name, P in sets.items():
        rec = {"points": int(len(P))}
        for name, (a, b) in cams.items():
            ua, ub = project(lpt, a, P), project(lpt, b, P)
            oka, okb = np.all(np.isfinite(ua), axis=1), np.all(np.isfinite(ub), axis=1)
            both = oka & okb
            d = np.hypot(*(ua[both] - ub[both]).T) if both.any() else np.array([0.0])
            on = both & (ub[:, 0] >= 0) & (ub[:, 0] < 1280) & (ub[:, 1] >= 0) & (ub[:, 1] < 800)
            rec[name] = {"image_1e-12": int(oka.sum()), "image_1e-9": int(okb.sum()), "both": int(both.sum()),
                         "only_1e-12": int((oka & ~okb).sum()), "only_1e-9": int((okb & ~oka).sum()),
                         "on_sensor_both": int(on.sum()), "max_px_diff": float(d.max())}
            r = rec[name]
            log(f"  {set_name}, {name}: {rec['points']} points; image at 1e-12 {r['image_1e-12']}, at 1e-9 "
                f"{r['image_1e-9']}; only at 1e-12 {r['only_1e-12']}, only at 1e-9 {r['only_1e-9']}; on the sensor "
                f"(both) {r['on_sensor_both']}; largest pixel difference {r['max_px_diff']:.2e} px")
        out["checks"][set_name] = rec
    ok_lines = all(info["changed_lines"] == [info["changed_lines"][0]] and len(info["changed_lines"]) == 1
                   for info in out["files"].values())
    ok_px = all(v["max_px_diff"] <= 1e-6 for s in out["checks"].values() for k, v in s.items() if k != "points")
    ok_lost = all(v["only_1e-12"] == 0 for s in out["checks"].values() for k, v in s.items() if k != "points")
    out["verdict"] = {"one_line_changed": ok_lines, "same_pixels_1e-6": ok_px, "nothing_lost_at_1e-9": ok_lost}
    log(f"  verdict: one line changed per file {ok_lines}; same pixels (<= 1e-6 px) where both give one {ok_px}; "
        f"no point loses its image at 1e-9 {ok_lost}")
    log.save(out)


if __name__ == "__main__":
    main()
