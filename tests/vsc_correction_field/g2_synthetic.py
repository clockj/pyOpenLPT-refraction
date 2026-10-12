"""G2: the cylinder VSC on the synthetic sets with 2, 3 and 4 cameras (truth known).

    python g2_synthetic.py prepare N NAME         # projects/NAME from the run V2 fitted for N cameras
    python g2_synthetic.py vsc NAME --via direct --compare FIELD     # V2's protocol, against V2's field
    python g2_synthetic.py vsc NAME --via service                    # the GUI's defaults
    python g2_synthetic.py track NAME TAG          # track the project's config at 1.0 px (runs/TAG), tracks back in
    python g2_synthetic.py score                   # the two-pass runs against the truth

- prepare: V2's runs (read only): 2 cameras the study's Gcal_n8_r2_tol1.5 (its images are Gcal4_n8's cameras 1-2),
  3 and 4 cameras the self-calibration's Gcal4_n8_r3/r4_tol1.5; the run's config as projects/NAME/config.json and its
  Results/ copied;
- vsc --via direct: modules/vsc/field_vsc.FieldVSC with V2's data (every frame 0-39, tracks of >= 10 points) and
  bending grid (1e3 ... 1e8); --compare FIELD: the correction against tests/cylinder_selfcal/data/fields/FIELD.npz over
  the pixels it covers. Same inputs, same estimator: the field must be V2's (to the 8 px grid), so V2's truth scores
  (one pass recovers 0.69 / 0.80 / 0.89 of the observable error with 2 / 3 / 4 cameras) are the module's;
- vsc --via service: VSCService with the tracking view's defaults (tracks of >= 15 points, 20,000 points);
- track: the project's config (its cameras' corrections) at 1.0 px with build vcf into runs/TAG, and its tracks copied
  into the project (as a GUI run would leave them), for the next pass;
- score: the 2-camera GUI path over two passes (runs g2_lp1_r2, g2_lp2_r2) against the truth (the study's
  eval_study.evaluate), beside V2's re-rendered one- and two-pass runs (Gcal4c2_n8_r2_tol1.0, Gcal4c2b_n8_r2_tol1.0)
  and the uncorrected run at 1.0 px (the study's Gcal_n8_r2_tol1.0).
Writes g2_synthetic_<part>_<NAME>_out.txt and .json here.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

import vcf_common as vc

import stbruns  # noqa: E402  (tests/cylinder_stb)

STUDY = os.path.join(vc.REPO, "tests", "cylinder_stb_study")
CAL4 = os.path.join(vc.SELFCAL, "data", "cal4")
V2_RUNS = {2: (os.path.join(STUDY, "runs", "Gcal_n8_r2_tol1.5"), "Gcal_n8_r2_tol1.5.json"),
           3: (os.path.join(vc.SELFCAL, "runs", "Gcal4_n8_r3_tol1.5"), "Gcal4_n8_r3_tol1.5.json"),
           4: (os.path.join(vc.SELFCAL, "runs", "Gcal4_n8_r4_tol1.5"), "Gcal4_n8_r4_tol1.5.json")}
V2_LAMS = (1e3, 1e4, 1e5, 1e6, 1e7, 1e8)


def prepare(n: int, name: str):
    rd, cfg_name = V2_RUNS[n]
    proj = os.path.join(vc.HERE, "projects", name)
    if os.path.exists(proj):
        raise SystemExit(f"{proj} exists")
    os.makedirs(proj)
    cfg = vc.load_json(os.path.join(rd, cfg_name))
    cfg["output_dir"] = "Results"
    cfg["_comment"] = f"G2 (tests/vsc_correction_field): V2's {n}-camera run {rd} as a GUI project"
    vc.save_json(os.path.join(proj, "config.json"), cfg)
    shutil.copytree(os.path.join(rd, "Results"), os.path.join(proj, "Results"))
    log = vc.Log(f"g2_synthetic_prepare_{name}")
    log(f"{proj}: {n} cameras from {rd}; cameras {[c['file'] for c in cfg['cameras']]}; tol_2d_px "
        f"{cfg['objects'][0]['ipr']['tol_2d_px']}; noise {cfg['objects'][0]['tracer'].get('lines')}")
    log.save({"project": proj, "run": rd})


def compare(cfg_path: str, field: str, log) -> dict:
    cfg = vc.load_json(cfg_path)
    z = np.load(os.path.join(vc.SELFCAL, "data", "fields", field + ".npz"), allow_pickle=False)
    dense = np.asarray(z["dense"], float)
    covered = np.asarray(z["covered"], bool)
    out, ok = {}, True
    for c, cam in enumerate(cfg["cameras"]):
        corr = vc.Correction.read(os.path.join(os.path.dirname(cfg_path), cam["correction"]))
        H, W = dense.shape[2:]
        yy, xx = np.mgrid[0:H, 0:W]
        uv = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
        diff = np.hypot(*(corr.at(uv) - dense[c].reshape(2, -1).T).T).reshape(H, W)[covered[c]]
        mag = np.hypot(dense[c][0], dense[c][1])[covered[c]]
        rms, frms = float(np.sqrt(np.mean(diff ** 2))), float(np.sqrt(np.mean(mag ** 2)))
        good = rms <= 0.02 * max(frms, 0.1) + 0.01
        ok &= good
        out[f"cam{c + 1}"] = {"rms_diff_px": rms, "max_diff_px": float(diff.max()), "field_rms_px": frms,
                              "pass": bool(good)}
        log(f"  cam{c + 1}: against {field} over its covered pixels {rms:.4f} px RMS ({diff.max():.3f} max); the field "
            f"there {frms:.3f} px RMS -> {'PASS' if good else 'FAIL'}")
    out["pass"] = bool(ok)
    return out


def vsc(name: str, via: str, field: str | None):
    import glob
    lpt = vc.lib("vcf")
    proj = os.path.join(vc.HERE, "projects", name)
    stem = f"g2_synthetic_vsc_{name}_{via}"
    k = 1 + len(glob.glob(os.path.join(vc.HERE, f"{stem}*_out.txt")))
    log = vc.Log(stem if k == 1 else f"{stem}_{k}")
    log(f"G2 {name} via {via} ({lpt.__file__})")
    if via == "direct":
        from modules.vsc.field_vsc import FieldVSC
        f = FieldVSC(proj, log, min_track_len=10, frame_stride=1, lams=V2_LAMS)
        ok, msg, data = f.run()
    else:
        from modules.vsc.vsc_service import VSCService
        svc = VSCService(proj, log_callback=log)
        svc.set_params(min_track_len=15, sample_points=20000, min_valid_points=2000, tolerance_mode="default",
                       tolerance_value=5.0)
        ok, msg, data = svc.run()
    log(f"  -> {'ok' if ok else 'FAILED'}: {msg}")
    out = {"project": proj, "via": via, "ok": bool(ok), "message": msg}
    if ok:
        fv = data.get("field_vsc", {})
        out["field_vsc"] = {k: v for k, v in fv.items() if k != "fit"}
        out["fit"] = {k: fv["fit"][k] for k in ("lam", "noise", "heldout") if k in fv.get("fit", {})}
        if field:
            out["compare"] = compare(os.path.join(proj, "config.json"), field, log)
    out["pass"] = bool(ok) and (out.get("compare", {"pass": True})["pass"])
    log(f"G2 {name} via {via}: {'PASS' if out['pass'] else 'FAIL'}")
    log.save(out)


def track(name: str, tag: str):
    proj = os.path.join(vc.HERE, "projects", name)
    c = vc.load_json(os.path.join(proj, "config.json"))
    for cam in c["cameras"]:
        if cam.get("correction") and not os.path.isabs(cam["correction"]):
            cam["correction"] = os.path.normpath(os.path.join(proj, cam["correction"]))
    c["objects"][0]["ipr"]["tol_2d_px"] = 1.0
    c["output_dir"] = "Results"
    cfg = os.path.join(vc.RUNS, "configs", f"{tag}.json")
    vc.save_json(cfg, c)
    rd = os.path.join(vc.RUNS, tag)
    info = stbruns.run(cfg, rd, "vcf", env={"OMP_NUM_THREADS": str(vc.THREADS_MAX), "PYTHONDONTWRITEBYTECODE": "1"})
    shutil.rmtree(os.path.join(proj, "Results"))
    shutil.copytree(os.path.join(rd, "Results"), os.path.join(proj, "Results"))
    log = vc.Log(f"g2_synthetic_track_{tag}")
    log(f"{tag}: {name}'s config ({[cam.get('correction') for cam in c['cameras']]}) at 1.0 px -> {rd}: exit "
        f"{info['returncode']}, {info['wall_s']:.0f} s; its tracks copied into {proj}/Results")
    log.save(info)


def score():
    sys.path.insert(0, STUDY)
    import eval_study
    log = vc.Log("g2_synthetic_score")
    runs = {"uncorrected_1.0": (os.path.join(STUDY, "runs", "Gcal_n8_r2_tol1.0"), os.path.join(STUDY, "data", "main"),
                                "Gcal_n8"),
            "gui_pass1": (os.path.join(vc.RUNS, "g2_lp1_r2"), CAL4, "Gcal4_n8"),
            "gui_pass2": (os.path.join(vc.RUNS, "g2_lp2_r2"), CAL4, "Gcal4_n8"),
            "v2_rerendered_pass1": (os.path.join(vc.SELFCAL, "runs", "Gcal4c2_n8_r2_tol1.0"), CAL4, "Gcal4c2_n8"),
            "v2_rerendered_pass2": (os.path.join(vc.SELFCAL, "runs", "Gcal4c2b_n8_r2_tol1.0"), CAL4, "Gcal4c2b_n8")}
    res = {}
    for name, (rd, case_dir, set_name) in runs.items():
        if not os.path.isdir(os.path.join(rd, "Results")):
            log(f"{name}: no run at {rd}")
            continue
        r = eval_study.evaluate(rd, case_dir, set_name, [0, 1])
        r.pop("per_frame", None)
        res[name] = r
        log(f"  {eval_study.line(name, r)}")
    out = {"scores": res}
    if "gui_pass2" in res and "v2_rerendered_pass2" in res:
        a, b = res["gui_pass2"], res["v2_rerendered_pass2"]
        ok = (abs(a["recall"]["all_in"] - b["recall"]["all_in"]) <= 0.03 * b["recall"]["all_in"]
              and abs(a["ghost_share"] - b["ghost_share"]) <= 0.03)
        out["pass"] = bool(ok)
        log(f"2 cameras, two passes through the GUI's VSC: recall {a['recall']['all_in']:.3f}, ghosts "
            f"{100 * a['ghost_share']:.1f} %; V2's two passes on re-rendered images {b['recall']['all_in']:.3f}, "
            f"{100 * b['ghost_share']:.1f} % -> {'PASS' if ok else 'FAIL'}")
    log.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prepare", "vsc", "track", "score"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--via", default="service")
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    if a.what == "prepare":
        prepare(int(a.args[0]), a.args[1])
    elif a.what == "vsc":
        vsc(a.args[0], a.via, a.compare)
    elif a.what == "track":
        track(a.args[0], a.args[1])
    else:
        score()


if __name__ == "__main__":
    main()
