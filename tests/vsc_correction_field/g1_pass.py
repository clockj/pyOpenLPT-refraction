"""G1: one pass of the cylinder VSC on a test project (projects/NAME), and its correction against the self-calibration's.

    python g1_pass.py NAME --via service [--compare FIELD] [--expect-refusal]
    python g1_pass.py NAME --via direct --min-len 10 --stride 5 [--compare FIELD]

- via service: modules/vsc/vsc_service.VSCService with the tracking view's defaults (min track length 15, 20,000
  points, 2,000 valid points, tolerance mode "default"): the code path of the GUI's Run VSC;
- via direct: modules/vsc/field_vsc.FieldVSC with the self-calibration's data (tracks of >= 10 points, every 5th
  frame), so that its field can be set against that campaign's;
- --compare FIELD: the correction now in config.json against tests/cylinder_selfcal/data/fields/FIELD.npz (its dense
  map), per camera, over the pixels that campaign's field covers; the yardstick is the difference between the
  campaign's two half-data fields (t1_m15_h1, t1_m15_h2) over the same pixels. Pass: no larger;
- --expect-refusal: the run must refuse (the guard: tracks older than the correction).
Also the project's VSC_log.txt, camFile_VSC/ and config.json (correction keys, one backup). Writes
g1_pass_<NAME>_<n>_out.txt and .json here (n: the run's number on that project).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import vcf_common as vc


def split_half_rms(cam: int, covered: np.ndarray) -> float:
    a, b = vc.selfcal_dense("t1_m15_h1")[cam], vc.selfcal_dense("t1_m15_h2")[cam]
    d = np.hypot(a[0] - b[0], a[1] - b[1])[covered]
    return float(np.sqrt(np.mean(d ** 2)))


def compare(cfg_path: str, field: str, log) -> dict:
    cfg = vc.load_json(cfg_path)
    z = np.load(os.path.join(vc.SELFCAL, "data", "fields", field + ".npz"), allow_pickle=False)
    dense = np.asarray(z["dense"], float)
    covered = np.asarray(z["covered"], bool)
    out = {}
    ok = True
    for c, cam in enumerate(cfg["cameras"]):
        path = os.path.join(os.path.dirname(cfg_path), cam["correction"])
        corr = vc.Correction.read(path)
        H, W = dense.shape[2:]
        yy, xx = np.mgrid[0:H, 0:W]
        uv = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
        diff = np.hypot(*(corr.at(uv) - dense[c].reshape(2, -1).T).T).reshape(H, W)[covered[c]]
        mag = np.hypot(dense[c][0], dense[c][1])[covered[c]]
        ref = split_half_rms(c, covered[c])
        rms = float(np.sqrt(np.mean(diff ** 2)))
        good = rms <= ref
        ok &= good
        out[f"cam{c + 1}"] = {"rms_diff_px": rms, "max_diff_px": float(diff.max()), "split_half_rms_px": ref,
                              "field_rms_px": float(np.sqrt(np.mean(mag ** 2))), "pass": bool(good)}
        log(f"  cam{c + 1}: {os.path.basename(path)} against {field} over its covered pixels: {rms:.3f} px RMS "
            f"({diff.max():.3f} max); the campaign's half-data fields differ by {ref:.3f} px RMS; its field there "
            f"{np.sqrt(np.mean(mag ** 2)):.3f} px RMS -> {'PASS' if good else 'FAIL'}")
    out["pass"] = bool(ok)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--via", choices=["service", "direct"], required=True)
    ap.add_argument("--min-len", type=int, default=15)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--expect-refusal", action="store_true")
    a = ap.parse_args()
    lpt = vc.lib("vcf")
    proj = os.path.join(vc.HERE, "projects", a.name)
    k = 1 + len(glob.glob(os.path.join(vc.HERE, f"g1_pass_{a.name}_*_out.txt")))
    log = vc.Log(f"g1_pass_{a.name}_{k}")
    log(f"G1 on {proj} via {a.via} (pyopenlpt {lpt.__file__})")
    cfg_path = os.path.join(proj, "config.json")
    before = vc.load_json(cfg_path)
    if a.via == "service":
        from modules.vsc.vsc_service import VSCService
        svc = VSCService(proj, log_callback=log)
        svc.set_params(min_track_len=15, sample_points=20000, min_valid_points=2000, tolerance_mode="default",
                       tolerance_value=5.0)
        ok, msg, data = svc.run()
    else:
        from modules.vsc.field_vsc import FieldVSC
        f = FieldVSC(proj, log, min_track_len=a.min_len, frame_stride=a.stride)
        ok, msg, data = f.run()
    log(f"  -> {'ok' if ok else 'refused/failed'}: {msg}")
    out = {"project": proj, "via": a.via, "ok": bool(ok), "message": msg}
    if a.expect_refusal:
        out["pass"] = (not ok) and ("rerun tracking" in msg)
        after = vc.load_json(cfg_path)
        out["config_unchanged"] = after == before
        log(f"G1 guard: {'PASS' if out['pass'] and out['config_unchanged'] else 'FAIL'} (refused: {not ok}; "
            f"config.json unchanged: {out['config_unchanged']})")
        out["pass"] = out["pass"] and out["config_unchanged"]
        log.save(out)
        return
    if not ok:
        log("G1: FAIL (the run did not finish)")
        out["pass"] = False
        log.save(out)
        return
    fv = data.get("field_vsc", {})
    out["field_vsc"] = {k2: v for k2, v in fv.items() if k2 != "fit"}
    out["fit"] = {k2: fv["fit"][k2] for k2 in ("lam", "noise", "heldout") if k2 in fv.get("fit", {})}
    after = vc.load_json(cfg_path)
    keys_ok = all(c.get("correction", "").startswith("camFile_VSC/") for c in after["cameras"])
    files_ok = all(os.path.exists(os.path.join(proj, c["correction"])) for c in after["cameras"])
    backup_ok = os.path.exists(os.path.join(proj, "config_backup.json"))
    only_corr = all({k2: v for k2, v in c1.items() if k2 != "correction"} == {k2: v for k2, v in c0.items()
                                                                             if k2 != "correction"}
                    for c0, c1 in zip(before["cameras"], after["cameras"]))
    rest_same = {k2: v for k2, v in after.items() if k2 != "cameras"} == \
        {k2: v for k2, v in before.items() if k2 != "cameras"}
    log(f"  config.json: correction keys {[c['correction'] for c in after['cameras']]} ({'ok' if keys_ok else 'NO'}), "
        f"files exist {files_ok}, backup {backup_ok}; nothing else changed: cameras {only_corr}, the rest {rest_same}")
    out.update({"keys_ok": keys_ok, "files_ok": files_ok, "backup": backup_ok, "only_correction_changed": only_corr,
                "rest_unchanged": rest_same, "valid_points": len(data.get("valid_points", [])),
                "tracks_for_gui": len(data.get("tracks", {}) or {})})
    ok_all = keys_ok and files_ok and backup_ok and only_corr and rest_same
    if a.compare:
        out["compare"] = compare(cfg_path, a.compare, log)
        ok_all &= out["compare"]["pass"]
    out["pass"] = bool(ok_all)
    log(f"G1 {a.name} #{k}: {'PASS' if ok_all else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
