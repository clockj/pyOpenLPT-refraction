"""Which rigs does the calibration get wrong?  A study of the stored results (no calibration is run).

For every rig of W5 (60 cases) and R4 (30 rigs at 0.5 and 1 px): per window, its cameras, their focal lengths and
sensors, and the angle the wand cloud spans as seen from each camera (the largest angle between the directions to the
cloud box's corners, in air: a proxy for how much of the window's refraction the camera sees).  Joined with the stored
verdicts (w5_from_scratch.json; r4_group_rigs.json, the runs with the restarts; both from the current code without
hints), it asks which simple rule on the rig separates the wrong answers from the right ones:
  - a window seen by one or two cameras only;
  - ... whose cameras all have f above a threshold;
  - ... whose cameras all see the cloud under less than a threshold angle;
and counts, per rule, the wrong answers flagged and the right answers flagged by mistake.
Output: p_weak_windows_out.txt / .json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def cloud_angle_deg(prigs, rig, cam):
    """The largest angle between the directions from the camera to the cloud box's corners."""
    P = prigs.box_corners(rig.centre, rig.half)
    D = P - cam.C
    D /= np.linalg.norm(D, axis=1)[:, None]
    G = np.clip(D @ D.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(G.min())))


def image_angles_deg(seed, noise, rig):
    """Per camera, the angle under which it sees the wand in the case's own detections, as the report's warning
    measures it (plate_wand_io.view_angles_deg: the 1st-99th percentile box of the detections, 2 atan(diagonal / 2f)),
    with the true f.  The detections are the case's (W5's and R4's generator, restarted from the seed)."""
    import prigs
    rng = np.random.default_rng(seed)
    prigs.random_rig(rng)
    n_frames = int(rng.integers(500, 1501))
    wand = prigs.make_wand(rig, n_frames, rng, noise_px=noise)
    out = []
    for c, cam in enumerate(rig.cams):
        uv = wand.uv[:, c].reshape(-1, 2)
        uv = uv[np.all(np.isfinite(uv), axis=1)]
        lo, hi = np.percentile(uv, [1.0, 99.0], axis=0)
        out.append(math.degrees(2.0 * math.atan(math.hypot(*(hi - lo)) / (2.0 * cam.f))))
    return out


def describe(prigs, seed, noise):
    rig = prigs.random_rig(np.random.default_rng(seed))
    img = image_angles_deg(seed, noise, rig)
    wins = []
    for w in range(len(rig.wins)):
        cs = [c for c, k in enumerate(rig.cam_win) if k == w]
        dirs = [rig.cams[c].R[2] for c in cs]
        spread = max((math.degrees(math.acos(min(1.0, abs(float(a @ b))))) for i, a in enumerate(dirs)
                      for b in dirs[i + 1:]), default=0.0)
        wins.append({"cams": cs, "f_px": [rig.cams[c].f for c in cs],
                     "sensor": [(rig.cams[c].width, rig.cams[c].height) for c in cs],
                     "cloud_deg": [cloud_angle_deg(prigs, rig, rig.cams[c]) for c in cs],
                     "image_deg": [img[c] for c in cs],
                     "axes_spread_deg": spread})
    return {"n_cams": len(rig.cams), "windows": wins}


def verdict(r):
    return "same" if r["ok"] else "equivalent" if r["equivalent"] else "DIFFERENT"


def main():
    import prigs
    log = prigs.Tee(HERE / "p_weak_windows_out.txt")
    runs = []
    for r in json.loads((HERE / "w5_from_scratch.json").read_text(encoding="utf-8"))["runs"]:
        runs.append(("W5", r["seed"], r["noise_px"], verdict(r)))
    for r in json.loads((HERE / "r4_group_rigs.json").read_text(encoding="utf-8"))["runs"]:
        if r.get("mode") == "with":
            runs.append(("R4", r["seed"], r["noise_px"], verdict(r)))
    rigs = {}
    for st, seed, noise, v in runs:
        if seed not in rigs:
            rigs[seed] = describe(prigs, seed, noise)
    log(f"W5 and R4: {len(runs)} cases on {len(rigs)} rigs (the stored answers without hints)")
    log("")
    log("The wrong answers (DIFFERENT) and their windows:")
    for st, seed, noise, v in runs:
        if v != "DIFFERENT":
            continue
        d = rigs[seed]
        log(f"  {st} seed {seed} ({noise:g} px): " + "; ".join(
            f"window {k + 1}: {len(w['cams'])} cam(s), f {', '.join(f'{f:.0f}' for f in w['f_px'])} px, cloud "
            f"{', '.join(f'{a:.1f}' for a in w['cloud_deg'])} deg, image {', '.join(f'{a:.1f}' for a in w['image_deg'])}"
            f" deg" for k, w in enumerate(d["windows"])))

    def weak(w, max_cams, f_min=None, ang_max=None, img_max=None):
        if len(w["cams"]) > max_cams:
            return False
        if f_min is not None and min(w["f_px"]) < f_min:
            return False
        if ang_max is not None and max(w["cloud_deg"]) >= ang_max:
            return False
        if img_max is not None and max(w["image_deg"]) >= img_max:
            return False
        return True

    rules = []
    for max_cams in (1, 2):
        rules.append((f"a window with <= {max_cams} cameras", dict(max_cams=max_cams)))
        for f_min in (8000, 10000, 12000, 13000):
            rules.append((f"a window with <= {max_cams} cameras, all f >= {f_min}", dict(max_cams=max_cams,
                                                                                         f_min=f_min)))
        for ang in (4.0, 5.0, 6.0, 8.0, 10.0):
            rules.append((f"a window with <= {max_cams} cameras, all seeing the cloud under < {ang:g} deg",
                          dict(max_cams=max_cams, ang_max=ang)))
        for ang in (3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
            rules.append((f"a window with <= {max_cams} cameras, all seeing the wand under < {ang:g} deg (image)",
                          dict(max_cams=max_cams, img_max=ang)))
    log("")
    log("Rules on the rig (flagged: wrong answers flagged of all wrong; right answers flagged of all right):")
    out_rules = []
    n_bad = sum(1 for x in runs if x[3] == "DIFFERENT")
    n_good = len(runs) - n_bad
    for name, kw in rules:
        hit = [x for x in runs if x[3] == "DIFFERENT" and any(weak(w, **kw) for w in rigs[x[1]]["windows"])]
        fa = [x for x in runs if x[3] != "DIFFERENT" and any(weak(w, **kw) for w in rigs[x[1]]["windows"])]
        out_rules.append({"rule": name, "wrong_flagged": len(hit), "wrong": n_bad, "right_flagged": len(fa),
                          "right": n_good, "right_flagged_cases": [f"{x[0]} {x[1]} {x[2]:g}" for x in fa]})
        log(f"  {name}: wrong {len(hit)}/{n_bad}, right {len(fa)}/{n_good}"
            + (f" ({', '.join(f'{x[0]} {x[1]} {x[2]:g} px {x[3]}' for x in fa)})" if 0 < len(fa) <= 8 else ""))
    log("")
    log("Every window with one or two cameras, by its widest view of the wand in the image (verdict of its case):")
    rows = []
    for st, seed, noise, v in runs:
        for k, w in enumerate(rigs[seed]["windows"]):
            if len(w["cams"]) <= 2:
                rows.append((max(w["image_deg"]), max(w["cloud_deg"]), min(w["f_px"]), st, seed, noise, k,
                             len(w["cams"]), v, w["axes_spread_deg"]))
    for ai, a, f, st, seed, noise, k, nc, v, sp in sorted(rows):
        log(f"  image {ai:5.1f} deg (cloud {a:5.1f}), f >= {f:6.0f} px, {nc} cam(s), axes {sp:4.1f} deg apart: {st} "
            f"seed {seed} ({noise:g} px) window {k + 1}: {v}")
    (HERE / "p_weak_windows.json").write_text(json.dumps({"runs": runs, "rigs": {str(k): v for k, v in rigs.items()},
                                                          "rules": out_rules}, indent=1, default=float),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
