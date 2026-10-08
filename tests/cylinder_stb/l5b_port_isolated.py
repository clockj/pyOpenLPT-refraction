"""L5b (diagnosis of L5): is the C++ fit the Python fit, where the objects are alone?

    python l5b_port_isolated.py [--build p4c]

L5 found 2-4 % of the paired objects' fits more than 1e-3 px apart (up to 0.4 px), on images of T1's density.  Here
every step is separated:
  1. the profile P(t; L, s) and the matched-filter templates: C++ lineProfile against lines.profile on a grid;
  2. isolated objects: a synthetic image of objects (lines, ellipses, round; flux 300-3000), each >= 60 px from
     the others (as many as random packing fits, about 75), so that no fit sees a neighbour; run with deblend_passes 1, no second pass and no validation, then
     with the full settings;
  3. on L5's T1 frame (cam1 frame 0): for each pair more than 1e-3 px apart, both solutions' costs on the Python's
     own window and model (lines.fit_objects machinery: chi2 of each parameter set with the neighbours of the Python
     run held fixed), to tell a bug (a worse optimum) from the flat valleys of short, faint or blended objects (equal
     costs).
Writes l5b_port_isolated_out.txt and l5b_port_isolated.json.
"""
import argparse
import os

import numpy as np

import cstb
from l5_port import cpp_param, load_case_module, pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="p4c")
    args = ap.parse_args()
    lpt, build = cstb.use_build(args.build)
    lines = load_case_module("lines")
    t1 = load_case_module("t1")
    log = cstb.Log(os.path.join(cstb.HERE, "l5b_port_isolated"))
    out = {"build": build}
    log(f"L5b: the C++ fit against the Python fit, step by step; build {build['name']}")

    # 1. the profile
    rng = np.random.default_rng(3)
    t = rng.uniform(-30, 30, 4000)
    L = np.where(rng.random(4000) < 0.2, 0.0, rng.uniform(0.05, 40, 4000))
    s = rng.uniform(0.5, 4.0, 4000)
    py = lines.profile(t, L, s)
    cpp = np.array([lpt.lineProfile(float(a), float(b), float(c)) for a, b, c in zip(t, L, s)])
    out["profile_max_rel"] = float(np.max(np.abs(cpp - py) / np.maximum(np.abs(py), 1e-300)))
    out["profile_max_abs"] = float(np.max(np.abs(cpp - py)))
    log(f"  1. profile on 4,000 points: max |C++ - Python| {out['profile_max_abs']:.2e} "
        f"(relative {out['profile_max_rel']:.2e})")

    # 2. isolated objects
    H, W = 800, 1280
    objs = []
    for _try in range(20000):          # random sequential packing: about 75 fit at 60 px apart
        if len(objs) >= 200:
            break
        u, v = rng.uniform(40, W - 40), rng.uniform(40, H - 40)
        if all(np.hypot(u - o[0], v - o[1]) > 60 for o in objs):
            kind = rng.integers(3)
            a = rng.uniform(3, 15) if kind != 1 else 0.0          # semi-axes of the uniform ellipse
            b = rng.uniform(3, 10) if kind != 0 else 0.0
            if kind == 2:
                a, b = rng.uniform(0, 2), rng.uniform(0, 2)
            objs.append((u, v, a, b, 1.0, rng.uniform(300, 3000)))
    img, _peaks = lines.render((H, W), np.array(objs), sigma_n=7.0, rng=rng)
    img8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
    for label, over in (("deblend 1, no second pass, no validation",
                         {"deblend_passes": 1, "second_pass": False, "validate": False}),
                        ("full settings", {})):
        P = lines.det_params(**over)
        d = lines.detect(img8, 7.0, P)
        rows = lpt.detectLines(lpt.Image.from_numpy(img8), 7.0, cpp_param(lpt, P))
        a = np.column_stack([d["u"], d["v"], d["Lx"], d["Ly"], d["sx"], d["sy"], d["A"]])
        b = np.array([[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
        pr = pair(a, b)
        dd = np.array([np.abs(a[i] - b[j]) for i, j in pr])
        rec = {"python": len(a), "cpp": len(b), "paired": len(pr),
               "max_centre": float(np.max(np.hypot(dd[:, 0], dd[:, 1]))),
               "share_centre_le_1e-6": float(np.mean(np.hypot(dd[:, 0], dd[:, 1]) <= 1e-6)),
               "share_centre_le_1e-3": float(np.mean(np.hypot(dd[:, 0], dd[:, 1]) <= 1e-3)),
               "max_L": float(dd[:, 2:4].max()), "max_s": float(dd[:, 4:6].max())}
        out[f"isolated: {label}"] = rec
        log(f"  2. isolated objects, {label}: Python {len(a)}, C++ {len(b)}, paired {len(pr)}; centres within "
            f"1e-6 px {100 * rec['share_centre_le_1e-6']:.1f} %, within 1e-3 px {100 * rec['share_centre_le_1e-3']:.1f} %, "
            f"max {rec['max_centre']:.2e} px; lengths max {rec['max_L']:.2e}, blurs max {rec['max_s']:.2e}")

    # 3. T1 frame 0 of cam1: costs of both solutions where they differ
    import json
    l1 = json.load(open(os.path.join(cstb.REPO, "cases", "cyl_20261007", "tests", "l1_preprocess.json"),
                        encoding="utf-8"))["T1"]
    gain = float(l1["cam1"]["gain"])
    bg = np.load(os.path.join(t1.DATA, "T1bg", "cam1_p90.npy"))
    hp, img8 = lines.preprocess(t1.read("cam1", 0), bg, gain)
    noise = gain * lines.noise_left(hp)
    P = lines.det_params()
    d = lines.detect(img8, noise, P)
    rows = lpt.detectLines(lpt.Image.from_numpy(img8), float(noise), cpp_param(lpt, P))
    par_py = np.column_stack([d["A"], d["u"], d["v"], d["Lx"], d["Ly"], d["sx"], d["sy"], d["c"]])
    par_cpp = np.array([[r[6], r[0], r[1], r[2], r[3], r[4], r[5], r[7]] for r in rows])
    pr = pair(np.column_stack([d["u"], d["v"]]), np.array([[r[0], r[1]] for r in rows]))
    I = img8.astype(np.float64)
    allm = lines.render_models(I.shape, par_py, P)
    ratios, cases = [], []
    for i, j in pr:
        if np.hypot(*(par_py[i, 1:3] - par_cpp[j, 1:3])) <= 1e-3:
            continue
        # each solution's cost on the same window, the Python run's neighbours fixed (fit_objects with 0 iterations)
        P0 = dict(P, fit_iters=0)
        _p, c_py, n_py, _ = lines.fit_objects(I, par_py[i:i + 1].copy(), P0, noise,
                                              offsets_img=allm, bounds=None)
        own_cpp = par_cpp[j:j + 1].copy()
        # keep the window of the Python solution so both costs are on the same pixels
        own_cpp_w = own_cpp.copy()
        own_cpp_w[:, 3:7] = np.maximum(own_cpp_w[:, 3:7], 0)
        allm_c = allm - lines.render_models(I.shape, par_py[i:i + 1], P) + lines.render_models(I.shape, own_cpp, P)
        _p, c_cpp, n_cpp, _ = lines.fit_objects(I, own_cpp, P0, noise, offsets_img=allm_c, bounds=None)
        ratios.append(float(c_cpp[0] / c_py[0]))
        cases.append({"py": par_py[i].tolist(), "cpp": par_cpp[j].tolist(), "chi2_py": float(c_py[0]),
                      "chi2_cpp": float(c_cpp[0]), "npx": [int(n_py[0]), int(n_cpp[0])]})
    ratios = np.array(ratios)
    out["t1_differing"] = {"n": len(ratios), "chi2_ratio_cpp_over_py_pcts":
                           np.percentile(ratios, [0, 10, 50, 90, 100]).tolist() if len(ratios) else None,
                           "examples": cases[:10]}
    # 4. the prototype's own sensitivity: the same frame plus 1e-9 grey levels of noise, through lines.detect again
    rng4 = np.random.default_rng(9)
    d2 = lines.detect(img8.astype(np.float64) + 1e-9 * rng4.standard_normal(img8.shape), noise, P)
    a4 = np.column_stack([d["u"], d["v"]])
    b4 = np.column_stack([d2["u"], d2["v"]])
    pr4 = pair(a4, b4)
    dd4 = np.array([np.hypot(*(a4[i] - b4[j])) for i, j in pr4])
    out["python_perturbed"] = {"python": len(d), "perturbed": len(d2), "paired": len(pr4),
                               "share_centre_le_1e-3": float(np.mean(dd4 <= 1e-3)), "max_centre": float(dd4.max())}
    log(f"  4. lines.detect on the same frame + 1e-9 noise: {len(d)} and {len(d2)} objects, paired {len(pr4)}; centres "
        f"within 1e-3 px {100 * np.mean(dd4 <= 1e-3):.2f} %, max {dd4.max():.3g} px (the prototype's own sensitivity)")
    if len(ratios):
        log(f"  3. T1 cam1 frame 0: {len(ratios)} pairs more than 1e-3 px apart; chi2(C++) / chi2(Python) on the same "
            f"window: min {ratios.min():.4f}, median {np.median(ratios):.4f}, max {ratios.max():.4f} (< 1: the C++ "
            f"solution fits better)")
        for cse in cases[:4]:
            log(f"     Python {np.round(cse['py'], 3).tolist()} chi2 {cse['chi2_py']:.1f}; C++ "
                f"{np.round(cse['cpp'], 3).tolist()} chi2 {cse['chi2_cpp']:.1f}")
    log.save(out)


if __name__ == "__main__":
    main()
