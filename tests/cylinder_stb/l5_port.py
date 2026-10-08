"""L5 (plan, Phase 4): the C++ line detector (LineFinder::detect) against its Python prototype (lines.detect).

    python l5_port.py [--build p4c] [--t1-frames 0 500]

Images, each run through both with the same settings (lines.DET, copied field by field into LineFinderParam) and
the same noise:
  - synthetic: frame 0 of both cameras of data/r5_lines (synth_lines.py; noise 7 grey levels);
  - T1: the given frames of both cameras, preprocessed as the case's L1 does (lines.preprocess with its background
    and gain; noise = gain x noise_left(hp)), from the case's local copy.
Detections are paired one to one by nearest centre within 0.5 px.  Judged (plan): the same detections (every one
paired) and centres within 1e-3 px; reported besides: the largest differences of the lengths, blurs and amplitudes,
the unpaired detections with their scores, the noise estimates (C++ estimateNoiseClipped and lines.
estimate_noise_clipped) and both run times.  Writes l5_port_out.txt and l5_port.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np

import cstb

CASE = os.path.join(cstb.REPO, "cases", "cyl_20261007")


def load_case_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CASE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def cpp_param(lpt, P):
    q = lpt.LineFinderParam()
    for k in ("lengths_x", "lengths_y", "sigma0", "bg_guard", "noise_sub", "k_detect", "core_px",
              "fit_margin_sigma", "fit_max_half", "fit_iters", "deblend_passes", "second_pass", "validate", "L_min",
              "L_max", "sigma_min", "sigma_max", "centre_box_frac", "dynamic_bounds", "length_growth",
              "length_margin", "nms_symmetric", "c_max_sigma", "clipped"):
        setattr(q, k, type(getattr(q, k))(P[k]) if not isinstance(P[k], list) else [float(x) for x in P[k]])
    return q


def pair(a, b, r=0.5):
    """Greedy one-to-one pairing of rows a (u, v, ...) and b by nearest centre within r."""
    from scipy.spatial import cKDTree
    if len(a) == 0 or len(b) == 0:
        return []
    tb = cKDTree(b[:, :2])
    cand = []
    for i, x in enumerate(a[:, :2]):
        for j in tb.query_ball_point(x, r):
            cand.append((float(np.hypot(*(a[i, :2] - b[j, :2]))), i, j))
    cand.sort()
    ua, ub, out = set(), set(), []
    for d, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        out.append((i, j))
    return out


def compare(lpt, lines, img8, noise, P, label, log):
    t0 = time.perf_counter()
    d = lines.detect(img8, noise, P)
    t_py = time.perf_counter() - t0
    py = np.column_stack([d["u"], d["v"], d["Lx"], d["Ly"], d["sx"], d["sy"], d["A"], d["c"], d["score"]]) \
        if len(d) else np.zeros((0, 9))
    t0 = time.perf_counter()
    rows = lpt.detectLines(lpt.Image.from_numpy(img8), float(noise), cpp_param(lpt, P))
    t_cpp = time.perf_counter() - t0
    cpp = np.array([[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[10]] for r in rows]) if rows else np.zeros((0, 9))
    pr = pair(py, cpp)
    ip = np.array([i for i, _ in pr], int)
    jc = np.array([j for _, j in pr], int)
    rec = {"label": label, "noise": float(noise), "python": len(py), "cpp": len(cpp), "paired": len(pr),
           "python_s": t_py, "cpp_s": t_cpp}
    if len(pr):
        dd = np.abs(py[ip] - cpp[jc])
        names = ["u", "v", "Lx", "Ly", "sx", "sy", "A", "c", "score"]
        rec["max_abs_diff"] = {n: float(dd[:, k].max()) for k, n in enumerate(names)}
        rec["p99_abs_diff"] = {n: float(np.percentile(dd[:, k], 99)) for k, n in enumerate(names)}
        rec["centre_le_1e-3"] = float(np.mean(np.hypot(dd[:, 0], dd[:, 1]) <= 1e-3))
    un_py = sorted(set(range(len(py))) - set(ip.tolist()))
    un_cpp = sorted(set(range(len(cpp))) - set(jc.tolist()))
    rec["unpaired_python_scores"] = [float(py[i, 8]) for i in un_py][:20]
    rec["unpaired_cpp_scores"] = [float(cpp[j, 8]) for j in un_cpp][:20]
    rec["unpaired"] = [len(un_py), len(un_cpp)]
    rec["pass"] = bool(not un_py and not un_cpp and rec.get("centre_le_1e-3", 0.0) == 1.0)
    log(f"  {label}: noise {noise:.2f}; Python {len(py)} objects in {t_py:.1f} s, C++ {len(cpp)} in {t_cpp:.2f} s; "
        f"paired {len(pr)}, unpaired {len(un_py)} / {len(un_cpp)}")
    if len(pr):
        md = rec["max_abs_diff"]
        log(f"    largest differences: centre u {md['u']:.2e}, v {md['v']:.2e} px; Lx {md['Lx']:.2e}, Ly {md['Ly']:.2e}; "
            f"sx {md['sx']:.2e}, sy {md['sy']:.2e}; A {md['A']:.2e}; score {md['score']:.2e}; centres within 1e-3 px "
            f"{100 * rec['centre_le_1e-3']:.2f} %")
    if un_py or un_cpp:
        log(f"    unpaired scores: Python {rec['unpaired_python_scores'][:8]}, C++ {rec['unpaired_cpp_scores'][:8]}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p4c")
    ap.add_argument("--t1-frames", type=int, nargs="+", default=[0, 500])
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    lines = load_case_module("lines")
    t1 = load_case_module("t1")
    P = lines.det_params()
    log = cstb.Log(os.path.join(cstb.HERE, "l5_port"))
    log(f"L5: the C++ line detector against lines.detect; build {build['name']} ({build['git_head']} + diff "
        f"{(build['git_diff_sha256'] or '')[:12]}); settings lines.DET")
    out = {"build": build, "params": P, "images": []}
    for c in (1, 2):
        img8 = cv2.imread(os.path.join(cstb.DATA, "r5_lines", "img", f"cam{c}", "img000000.tif"), cv2.IMREAD_UNCHANGED)
        rec = compare(lpt, lines, img8, 7.0, P, f"synthetic r5_lines cam{c} frame 0", log)
        rec["noise_cpp_estimate"] = lpt.estimateNoiseClipped(lpt.Image.from_numpy(img8))
        rec["noise_py_estimate"] = lines.estimate_noise_clipped(img8)
        log(f"    noise estimate of the clipped image: C++ {rec['noise_cpp_estimate']:.4f}, Python "
            f"{rec['noise_py_estimate']:.4f} (true 7)")
        out["images"].append(rec)
    with open(os.path.join(CASE, "tests", "l1_preprocess.json"), encoding="utf-8") as fh:
        l1 = json.load(fh)["T1"]
    for cam in t1.CAMS:
        gain = float(l1[cam]["gain"])
        bg = np.load(os.path.join(t1.DATA, "T1bg", f"{cam}_p90.npy"))
        for k in args.t1_frames:
            hp, img8 = lines.preprocess(t1.read(cam, k), bg, gain)
            noise = gain * lines.noise_left(hp)
            rec = compare(lpt, lines, img8, noise, P, f"T1 {cam} frame {k}", log)
            out["images"].append(rec)
    out["pass"] = bool(all(r["pass"] for r in out["images"]))
    log(f"  judged (every detection paired, centres within 1e-3 px): {'PASS' if out['pass'] else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
