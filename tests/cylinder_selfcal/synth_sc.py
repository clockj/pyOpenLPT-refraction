"""The synthetic sets of V2 (plan: section 3, V2; and, on your request, the synthetic test with 2, 3 and 4 cameras).

    python synth_sc.py cal4                 # data/cal4: the study's 4-camera population with an error field in every camera
    python synth_sc.py gallm                # data/gallm: the study's G-all matched to T1 (Gallm), and Gallm0 (no error)
    python synth_sc.py corrected CASE BASE NEW FIELD.npz      # BASE's images moved by -c(uv) of a fitted field (re-rendered)
    python synth_sc.py warped CASE BASE NEW FIELD.npz         # BASE's preprocessed float images warped by the field

Every set is rendered as the study renders (tests/cylinder_stb_study/synth_study.py: its ellipse renderer, its noise
fields seeded per camera and frame, S1's high-pass with a 31 px median), from the study's arrays (copied here, read
only there):
  - cal4 (data/cal4, 40 frames, the central layer, the clean shape model, noise 8): the study's data/main population
    seen by its 4-camera rig, each image moved by an error field of 0.7 px RMS.  cam1 and cam2 get the study's fields
    (T0's wand-residual pattern, as its G-cal); cam3 and cam4 get cam2's and cam1's patterns (their lenses' twins)
    turned by 180 deg on the sensor, e(u, v) = -e_twin(1279 - u, 799 - v), so that no two cameras share a field.
    Set Gcal4_n8 (cameras 1-4).  Its cam1 and cam2 images equal the study's Gcal_n8 pixel for pixel (checked), so the
    study's 2-camera runs of Gcal_n8 are this set's 2-camera runs;
  - gallm (data/gallm, 80 frames): the study's Gallm (everything as on T1: particles throughout the tube, T1's shape
    model with scatter, the error field, T1's noise) and Gallm0, the same rendered at the model's positions (no error
    field: the most a correction can give);
  - corrected: NEW = BASE with every image moved by -c(uv) (the fitted field at the rendered position), re-rendered:
    what a correction inside the library would give, with no interpolation;
  - warped: NEW = BASE's float image after the high-pass, warped by the field (W(q) = I(q + c), cubic spline), then
    recentred, rounded and clipped as before; the noise as S1 measures it, per camera (the median over the frames),
    in sets.json's noise_measured: the method V3 uses on T1.
Writes data/<case>/ (arrays, img/<set>/camK/, imgLists/<set>/, sets.json, synth.json) and synth_sc_<what>_out.txt,
.json here.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np

import selfcal as sc

import synth_study as ss  # noqa: E402  (the study's renderer: preprocess, error_fields, render_ellipse, layers)

STUDY_DATA = os.path.join(sc.STUDY, "data")
H, W = 800, 1280
N_PROC = 10
_C = {}


def _arr(case_dir, name):
    key = (case_dir, name)
    if key not in _C:
        _C[key] = np.load(os.path.join(case_dir, "arrays", name + ".npy"), mmap_mode="r")
    return _C[key]


def _init():
    sc.one_thread_env()


def float_image(case_dir, spec, c, t, seed):
    """The set's float image of camera c at frame t (8-bit units) before the preprocessing: its layers' ellipses at
    the uv variant's positions with the shape variant's lengths and flux, plus the seeded noise field."""
    layer = np.asarray(_arr(case_dir, "layer"))
    uvv, shp = spec["variant"], spec.get("shape", spec["variant"])
    uv = _arr(case_dir, f"uv_{uvv}_c{c}")[t]
    Lh = _arr(case_dir, f"Lh_{shp}_c{c}")[t]
    Lv = _arr(case_dir, f"Lv_{shp}_c{c}")[t]
    fl = _arr(case_dir, f"flux_{shp}_c{c}")[t]
    al = _arr(case_dir, f"alpha_c{c}")
    s0 = float(np.asarray(_arr(case_dir, f"sigma0_{shp}"))[c])
    imgs = []
    for ly in spec["layers"]:            # one image per layer, summed, as synth_study._render_job does
        img = np.zeros((H, W))
        for i in np.flatnonzero((layer == ss.LAYERS.index(ly)) & np.isfinite(uv[:, 0]) & np.isfinite(Lh)):
            ss.render_ellipse(img, uv[i, 0], uv[i, 1], Lh[i], Lv[i], al[i], fl[i], s0)
        imgs.append(img)
    zf = np.random.default_rng((seed, c, t)).standard_normal((H, W))
    noise = spec["noise"][spec["cams"].index(c)]
    return sum(imgs) + noise * zf


def highpass_float(x, noise, size):
    """synth_study.preprocess without the rounding: the float high-pass, recentred."""
    import lines
    g = noise / ss.HP_RAW_NOISE
    x = np.asarray(x, np.float32)
    hp = x - lines.median_lowpass(x / np.float32(g), size) * np.float32(g)
    hp -= np.float32(lines.noise_level(hp))
    return hp


def _render(job):
    case_dir, name, spec, c, t, seed = job
    import cv2
    x = float_image(case_dir, spec, c, t, seed)
    noise = spec["noise"][spec["cams"].index(c)]
    path = os.path.join(case_dir, "img", name, f"cam{c + 1}", f"img{t:06d}.tif")
    if spec.get("warp"):
        import lines
        hp = highpass_float(x, noise, spec.get("median_px", ss.HP_SIZE))
        maps = np.load(spec["warp"][c], mmap_mode="r")
        hw = sc.warp(hp, (np.asarray(maps[0]), np.asarray(maps[1])))
        hw -= np.float32(lines.noise_level(hw))
        cv2.imwrite(path, np.clip(np.rint(hw), 0, 255).astype(np.uint8))
        return c, t, float(lines.noise_left(hw)), float(lines.noise_left(hp))
    cv2.imwrite(path, ss.preprocess(x, noise, spec.get("median_px", ss.HP_SIZE)))
    return c, t, None, None


def render(case_dir, name, spec, frames, seed, log):
    for c in spec["cams"]:
        os.makedirs(os.path.join(case_dir, "img", name, f"cam{c + 1}"), exist_ok=True)
    jobs = [(case_dir, name, spec, c, t, seed) for c in spec["cams"] for t in range(frames)]
    t0 = time.time()
    noise_w, noise_0 = {c: [] for c in spec["cams"]}, {c: [] for c in spec["cams"]}
    with Pool(N_PROC, initializer=_init) as pool:
        for c, t, nw, n0 in pool.imap_unordered(_render, jobs):
            if nw is not None:
                noise_w[c].append(nw)
                noise_0[c].append(n0)
    os.makedirs(os.path.join(case_dir, "imgLists", name), exist_ok=True)
    for c in spec["cams"]:
        with open(os.path.join(case_dir, "imgLists", name, f"cam{c + 1}ImageNames.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(os.path.join(case_dir, "img", name, f"cam{c + 1}", f"img{t:06d}.tif")
                               for t in range(frames)) + "\n")
    log(f"  rendered {name}: cameras {[c + 1 for c in spec['cams']]}, {frames} frames ({len(jobs)} images) in "
        f"{time.time() - t0:.0f} s")
    if spec.get("warp"):
        nm = [float(np.median(noise_w[c])) for c in spec["cams"]]
        n0 = [float(np.median(noise_0[c])) for c in spec["cams"]]
        log(f"  {name}: noise measured after the warp {np.round(nm, 3).tolist()} (before {np.round(n0, 3).tolist()})")
        return {"noise_measured": nm, "noise_measured_before_warp": n0}
    return {}


def copy_arrays(src_case, dst_case, names):
    os.makedirs(os.path.join(dst_case, "arrays"), exist_ok=True)
    for n in names:
        shutil.copy2(os.path.join(src_case, "arrays", n + ".npy"), os.path.join(dst_case, "arrays", n + ".npy"))


def copy_files(src_case, dst_case, names):
    for n in names:
        s, d = os.path.join(src_case, n), os.path.join(dst_case, n)
        if os.path.isdir(s):
            if os.path.isdir(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


def sets_update(case_dir, name, spec):
    p = os.path.join(case_dir, "sets.json")
    sets = sc.load_json(p) if os.path.exists(p) else {}
    sets[name] = spec
    sc.save_json(p, sets)
    return sets


# ------------------------------------------------------------------ cal4
def rot180(f):
    return lambda u, v: -f(1279.0 - np.asarray(u, float), 799.0 - np.asarray(v, float))


def cal4_fields():
    fields, info = ss.error_fields(2)
    return [fields[0], fields[1], rot180(fields[1]), rot180(fields[0])], info


def make_cal4(log):
    src = os.path.join(STUDY_DATA, "main")
    dst = os.path.join(sc.DATA, "cal4")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    meta_src = sc.load_json(os.path.join(src, "synth.json"))
    names = ["X", "layer", "sigma0_clean"] + [f"{k}_c{c}" for c in range(4)
                                               for k in ("uv_clean", "Lh_clean", "Lv_clean", "flux_clean", "alpha")]
    copy_arrays(src, dst, names)
    copy_files(src, dst, ["camFile", "otf_r2.txt", "otf_r3.txt", "otf_r4.txt"])
    fields, info = cal4_fields()
    rms = []
    gu, gv = np.meshgrid(np.linspace(0, 1279, 65), np.linspace(0, 799, 41))
    for c in range(4):
        uv = np.load(os.path.join(dst, "arrays", f"uv_clean_c{c}.npy"))
        e = fields[c](uv[..., 0], uv[..., 1])
        np.save(os.path.join(dst, "arrays", f"uv_cal4_c{c}.npy"), uv + e)
        np.save(os.path.join(dst, "arrays", f"err_cal4_c{c}.npy"), e)
        G = fields[c](gu.ravel(), gv.ravel())
        rms.append(float(np.sqrt(np.mean((G ** 2).sum(axis=1)))))
    # the study's G-cal positions for cam1 and cam2: the same
    same = [bool(np.array_equal(np.load(os.path.join(dst, "arrays", f"uv_cal4_c{c}.npy")),
                                np.load(os.path.join(src, "arrays", f"uv_cal_c{c}.npy")), equal_nan=True))
            for c in (0, 1)]
    log(f"  error fields: RMS over the sensor {np.round(rms, 3).tolist()} px; cam1, cam2 positions equal the study's "
        f"G-cal: {same}")
    spec = {"cams": [0, 1, 2, 3], "layers": ["in"], "variant": "cal4", "shape": "clean", "noise": [8.0] * 4}
    sets_update(dst, "Gcal4_n8", spec)
    seed = int(meta_src["noise_seed"])
    render(dst, "Gcal4_n8", spec, int(meta_src["frames"]), seed, log)
    # pixel-for-pixel against the study's Gcal_n8 (cam1, cam2), every 8th frame
    import cv2
    diff = []
    for c in (0, 1):
        for t in range(0, int(meta_src["frames"]), 8):
            a = cv2.imread(os.path.join(dst, "img", "Gcal4_n8", f"cam{c + 1}", f"img{t:06d}.tif"), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(os.path.join(src, "img", "Gcal_n8", f"cam{c + 1}", f"img{t:06d}.tif"), cv2.IMREAD_UNCHANGED)
            diff.append(int(np.abs(a.astype(int) - b.astype(int)).max()))
    log(f"  Gcal4_n8's cam1 and cam2 against the study's Gcal_n8 (frames 0, 8, ..., 32): max pixel difference "
        f"{max(diff)}")
    meta = {"source": src, "frames": meta_src["frames"], "noise_seed": seed, "params_clean": meta_src["params_clean"],
            "error_fields": {"cam1_cam2": info, "cam3": "cam2's turned by 180 deg", "cam4": "cam1's turned by 180 deg",
                             "rms_px": rms},
            "equal_to_study_gcal": {"positions": same, "max_pixel_difference": max(diff)},
            "view_volume": sc.view_volume()}
    sc.save_json(os.path.join(dst, "synth.json"), meta)
    return meta


# ------------------------------------------------------------------ gallm
def make_gallm(log):
    src = os.path.join(STUDY_DATA, "gallm")
    dst = os.path.join(sc.DATA, "gallm")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    meta_src = sc.load_json(os.path.join(src, "synth.json"))
    names = ["X", "layer", "sigma0_allm"] + [f"{k}_c{c}" for c in range(2)
                                              for k in ("uv_allm", "uv_model", "Lh_allm", "Lv_allm", "flux_allm",
                                                        "alpha", "z")]
    copy_arrays(src, dst, names)
    copy_files(src, dst, ["camFile", "otf_t1.txt"])
    fields, _info = ss.error_fields(2)
    for c in range(2):
        uv0 = np.load(os.path.join(dst, "arrays", f"uv_model_c{c}.npy"))
        uv1 = np.load(os.path.join(dst, "arrays", f"uv_allm_c{c}.npy"))
        e = fields[c](uv0[..., 0], uv0[..., 1])
        log(f"  cam{c + 1}: uv_allm - uv_model = the study's field at uv_model: max difference "
            f"{float(np.nanmax(np.abs(uv1 - uv0 - e))):.2e} px")
        np.save(os.path.join(dst, "arrays", f"err_allm_c{c}.npy"), uv1 - uv0)
    s0 = sc.load_json(os.path.join(src, "sets.json"))["Gallm"]
    # Gallm: the study's images, listed from there (not re-rendered)
    sets_update(dst, "Gallm", {**s0, "images": "the study's data/gallm/img/Gallm"})
    os.makedirs(os.path.join(dst, "imgLists", "Gallm"), exist_ok=True)
    for c in range(2):
        shutil.copy2(os.path.join(src, "imgLists", "Gallm", f"cam{c + 1}ImageNames.txt"),
                     os.path.join(dst, "imgLists", "Gallm", f"cam{c + 1}ImageNames.txt"))
    spec0 = {"cams": [0, 1], "layers": ["in", "outA"], "variant": "model", "shape": "allm", "noise": s0["noise"],
             "noise_measured": s0["noise_measured"]}
    sets_update(dst, "Gallm0", spec0)
    render(dst, "Gallm0", spec0, int(meta_src["frames"]), int(meta_src["noise_seed"]), log)
    meta = {"source": src, "frames": meta_src["frames"], "noise_seed": meta_src["noise_seed"],
            "params_t1": meta_src["params_t1"], "noise_measured": s0["noise_measured"], "view_volume": sc.view_volume()}
    sc.save_json(os.path.join(dst, "synth.json"), meta)
    return meta


# ------------------------------------------------------------------ corrected and warped sets
def make_corrected(case, base, new, field_path, log):
    case_dir = os.path.join(sc.DATA, case)
    meta = sc.load_json(os.path.join(case_dir, "synth.json"))
    sets = sc.load_json(os.path.join(case_dir, "sets.json"))
    b = sets[base]
    f = sc.Field.load(field_path)
    var = f"{b['variant']}_{new}"
    shp = b.get("shape", b["variant"])
    stats = []
    for k, c in enumerate(b["cams"][:f.n_cam]):
        uv = np.load(os.path.join(case_dir, "arrays", f"uv_{b['variant']}_c{c}.npy"))
        F, N, _ = uv.shape
        flat = uv.reshape(-1, 2)
        ok = np.isfinite(flat[:, 0])
        corr = np.zeros_like(flat)
        if k < f.n_cam:
            corr[ok] = f.eval_dense(k, flat[ok])
        out = flat - corr
        np.save(os.path.join(case_dir, "arrays", f"uv_{var}_c{c}.npy"), out.reshape(F, N, 2))
        err_p = os.path.join(case_dir, "arrays", f"err_{b['variant']}_c{c}.npy")
        if os.path.exists(err_p):          # the error left after the correction, for a second pass's judgement
            np.save(os.path.join(case_dir, "arrays", f"err_{var}_c{c}.npy"),
                    np.load(err_p) - corr.reshape(F, N, 2))
        stats.append(float(np.sqrt(np.mean((corr[ok] ** 2).sum(axis=1)))))
    spec = {k: v for k, v in b.items() if k not in ("images", "warp")}
    spec.update({"variant": var, "shape": shp, "corrected_by": field_path, "base": base,
                 "cams": b["cams"][:f.n_cam], "noise": b["noise"][:f.n_cam]})
    if "noise_measured" in b:
        spec["noise_measured"] = b["noise_measured"][:f.n_cam]
    sets_update(case_dir, new, spec)
    log(f"  {new}: {base}'s images moved by -c(uv) of {field_path}: correction RMS per camera {np.round(stats, 3).tolist()} px")
    render(case_dir, new, spec, int(meta["frames"]), int(meta["noise_seed"]), log)
    return {"set": new, "correction_rms_px": stats}


def effective_arrays(case_dir, b, new, f):
    """A warped set's effective positions and the error left in them (uv - c(uv), err - c(uv)), for a second pass's
    judgement (v2_fit.py); the set itself renders BASE's positions and warps them.  Returns the variant's name."""
    eff = f"{b['variant']}_{new}"
    for k, c in enumerate(b["cams"][:f.n_cam]):
        uv = np.load(os.path.join(case_dir, "arrays", f"uv_{b['variant']}_c{c}.npy"))
        F, N, _ = uv.shape
        flat = uv.reshape(-1, 2)
        ok = np.isfinite(flat[:, 0])
        corr = np.zeros_like(flat)
        corr[ok] = f.eval_dense(k, flat[ok])
        np.save(os.path.join(case_dir, "arrays", f"uv_{eff}_c{c}.npy"), (flat - corr).reshape(F, N, 2))
        err_p = os.path.join(case_dir, "arrays", f"err_{b['variant']}_c{c}.npy")
        if os.path.exists(err_p):
            np.save(os.path.join(case_dir, "arrays", f"err_{eff}_c{c}.npy"), np.load(err_p) - corr.reshape(F, N, 2))
    return eff


def add_effective(case, name, log):
    """effective_arrays for a warped set made before they existed (Gallmw); for a corrected set, the error left in
    it, err_<variant> = err_<base> - (uv_<base> - uv_<variant>)."""
    case_dir = os.path.join(sc.DATA, case)
    sets = sc.load_json(os.path.join(case_dir, "sets.json"))
    s = sets[name]
    if "corrected_by" in s:
        b = sets[s["base"]]
        for c in s["cams"]:
            a = lambda n: np.load(os.path.join(case_dir, "arrays", n + ".npy"))  # noqa: E731
            e = a(f"err_{b['variant']}_c{c}") - (a(f"uv_{b['variant']}_c{c}") - a(f"uv_{s['variant']}_c{c}"))
            np.save(os.path.join(case_dir, "arrays", f"err_{s['variant']}_c{c}.npy"), e)
        log(f"  {name}: remaining errors written as err_{s['variant']}")
        return {"set": name, "err": f"err_{s['variant']}"}
    eff = effective_arrays(case_dir, sets[s["base"]], name, sc.Field.load(s["warped_by"]))
    s["variant_effective"] = eff
    sets_update(case_dir, name, s)
    log(f"  {name}: effective positions and remaining errors written as variant {eff}")
    return {"set": name, "variant_effective": eff}


def make_warped(case, base, new, field_path, log):
    case_dir = os.path.join(sc.DATA, case)
    meta = sc.load_json(os.path.join(case_dir, "synth.json"))
    sets = sc.load_json(os.path.join(case_dir, "sets.json"))
    b = sets[base]
    f = sc.Field.load(field_path)
    mdir = os.path.join(case_dir, "warp_maps", new)
    os.makedirs(mdir, exist_ok=True)
    paths = []
    for k, c in enumerate(b["cams"]):
        mx, my = sc.warp_maps(f, k)
        p = os.path.join(mdir, f"cam{c + 1}.npy")
        np.save(p, np.stack([mx, my]))
        paths.append(p)
    spec = {k: v for k, v in b.items() if k not in ("images",)}
    spec.update({"warp": paths, "warped_by": field_path, "base": base})
    spec["variant_effective"] = effective_arrays(case_dir, b, new, f)
    res = render(case_dir, new, spec, int(meta["frames"]), int(meta["noise_seed"]), log)
    spec.update(res)
    sets_update(case_dir, new, spec)
    return {"set": new, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cal4", "gallm", "corrected", "warped", "effective"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.what == "effective":
        case, name = a.args
        log = sc.Log(f"synth_sc_effective_{name}")
        log.save(add_effective(case, name, log))
        return
    tag = a.what if a.what in ("cal4", "gallm") else f"{a.what}_{a.args[2]}"
    log = sc.Log(f"synth_sc_{tag}")
    log(f"synth_sc {a.what} {' '.join(a.args)}: build {sc.BUILD}, {N_PROC} processes")
    if a.what == "cal4":
        out = make_cal4(log)
    elif a.what == "gallm":
        out = make_gallm(log)
    elif a.what == "corrected":
        case, base, new, fp = a.args
        out = make_corrected(case, base, new, fp, log)
    else:
        case, base, new, fp = a.args
        out = make_warped(case, base, new, fp, log)
    log.save(out)


if __name__ == "__main__":
    main()
