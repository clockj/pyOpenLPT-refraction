"""G1 (d): which VSC a project gets (a light check).

    python g1_dispatch.py

modules/vsc/field_vsc.is_field_project on the repository's projects: the tube runs (cases/cyl_20261007, cyl_20261008:
PINCYL, config.json) take the field VSC; the plate case (cases/stb_20260727_10Hz_r_b_1: PINPLATE) and a pinhole set
(tests/cylinder_stb/data/reg_pinhole: PINHOLE, config.txt) keep today's VSC. A tube project with only config.txt
(T1's config written as text into data/dispatch/) takes the field VSC, which refuses it with the conversion to do,
before it reads anything else. Nothing is written outside data/. Writes g1_dispatch_out.txt and .json here.
"""
from __future__ import annotations

import os
import shutil

import vcf_common as vc
from modules import stb_config
from modules.vsc.field_vsc import is_field_project


def camera_types(proj):
    from modules.vsc.field_vsc import _model_token
    p = stb_config.find_config(proj)
    cfg = stb_config.read(p, inline_objects=False)
    return os.path.basename(p), [_model_token(stb_config.resolve(p, c["file"])) for c in cfg["cameras"]]


def main():
    vc.lib("vcf")
    log = vc.Log("g1_dispatch")
    # tracking projects: T1's config as a GUI project's config.json, the plate set's config.txt as config.json
    t1 = os.path.join(vc.DATA, "dispatch", "t1_json")
    os.makedirs(t1, exist_ok=True)
    vc.save_json(os.path.join(t1, "config.json"),
                 vc.load_json(os.path.join(vc.CASE_RUNS, "s3_final_m15", "config_final_m15.json")))
    plate_src = os.path.join(vc.CSTB, "data", "reg_plate")
    plate = os.path.join(vc.DATA, "dispatch", "plate_json")
    os.makedirs(plate, exist_ok=True)
    pcfg = stb_config.read_text(os.path.join(plate_src, "config.txt"))
    for c in pcfg["cameras"]:
        c["file"] = stb_config.resolve(os.path.join(plate_src, "config.txt"), c["file"])
    pcfg["image_lists"] = [stb_config.resolve(os.path.join(plate_src, "config.txt"), p) for p in pcfg["image_lists"]]
    stb_config.write_json(pcfg, os.path.join(plate, "config.json"))
    cases = {"T1 (its tracking config as config.json)": (t1, True),
             "three cameras (cases/cyl_20261008/data/s2)": (os.path.join(vc.REPO, "cases", "cyl_20261008", "data",
                                                                          "s2"), True),
             "synthetic tube (tests/cylinder_stb/data/v2)": (os.path.join(vc.CSTB, "data", "v2"), True),
             "plate (tests/cylinder_stb/data/reg_plate, config.txt)": (plate_src, False),
             "plate, its config as config.json": (plate, False),
             "pinhole (tests/cylinder_stb/data/reg_pinhole, config.txt)": (os.path.join(vc.CSTB, "data", "reg_pinhole"),
                                                                           False)}
    out = {}
    ok = True
    for name, (proj, want) in cases.items():
        got = is_field_project(proj)
        cfg_name, types = camera_types(proj)
        good = got == want
        ok &= good
        out[name] = {"project": proj, "config": cfg_name, "cameras": types, "field_vsc": got, "expected": want,
                     "pass": good}
        log(f"  {name}: {cfg_name}, cameras {types} -> {'field VSC' if got else 'the parameter VSC'} "
            f"({'as expected' if good else 'NOT as expected'})")
    # a tube project with only config.txt
    d = os.path.join(vc.DATA, "dispatch", "t1_text")
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    cfg = stb_config.read_json(os.path.join(vc.CASE_RUNS, "s3_final_m15", "config_final_m15.json"))
    stb_config.write_text(cfg, os.path.join(d, "config.txt"))
    got = is_field_project(d)
    from modules.vsc.vsc_service import VSCService
    svc = VSCService(d, log_callback=log)
    res_ok, msg, _data = svc.run()
    refused = (not res_ok) and "config.txt" in msg and "convert" in msg
    good = got and refused and not os.path.exists(os.path.join(d, "camFile_VSC"))
    ok &= good
    out["tube project with config.txt"] = {"field_vsc": got, "message": msg, "pass": good}
    log(f"  a tube project with only config.txt: {'field VSC' if got else 'the parameter VSC'}; it says: {msg} -> "
        f"{'PASS' if good else 'FAIL'}")
    out["pass"] = bool(ok)
    log(f"G1 dispatch: {'PASS' if ok else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
