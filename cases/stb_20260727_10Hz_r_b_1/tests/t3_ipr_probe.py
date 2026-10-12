"""T3: one IPR, without tracking, on single frames with each camFile set (2026-10-04-plan.md, section 4).

Runs the configs ipr_probe/<set>_f<frame>/config.txt written by make_configs.py (frame range [frame, frame]: one
initial-phase IPR on the raw images) one after another with run_stb.py, then parses their logs: detections per camera,
objects matched per camera combination and 2-D tolerance, matching and shaking times, objects found.  A finished run
(run_info.json with status ok) is not run again.

Writes t3_ipr_probe.json and t3_ipr_probe_out.txt beside this script; the runs' logs stay on X: in ipr_probe/.
    python tests/t3_ipr_probe.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402
import parse_log  # noqa: E402

OUT_JSON = os.path.join(case.TESTS, "t3_ipr_probe.json")
OUT_TXT = os.path.join(case.TESTS, "t3_ipr_probe_out.txt")
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def run(cfg, cfg_path):
    info = os.path.join(os.path.dirname(cfg_path), "run_info.json")
    if os.path.exists(info):
        with open(info, encoding="utf-8") as f:
            if json.load(f).get("status") == "ok":
                return
    subprocess.run([cfg["python"], os.path.join(case.HERE, "run_stb.py"), cfg_path, "--quiet"], check=True)


def main():
    cfg = case.load_config()
    res = {"frames": cfg["ipr_probe_frames"], "runs": {}}
    for fr in cfg["ipr_probe_frames"]:
        for which in ("new", "old"):
            d = os.path.join(cfg["out_dir"], "ipr_probe", f"{which}_f{fr:04d}")
            run(cfg, os.path.join(d, "config.txt"))
            recs = parse_log.parse(os.path.join(d, "log.txt"))
            assert len(recs) == 1, (d, len(recs))
            r = recs[0]
            with open(os.path.join(d, "run_info.json"), encoding="utf-8") as f:
                wall = json.load(f)["wall_s"]
            otf = [ln.strip() for ln in parse_log.read_lines(os.path.join(d, "log.txt")) if "(a,b,c,alpha)" in ln]
            res["runs"][f"{which}_f{fr:04d}"] = {"set": which, "frame": fr, "record": r, "flat": parse_log.flatten(r),
                                                 "wall_s": wall, "otf": otf}
    # the table
    log("T3: one IPR on raw images (initial phase, no tracking), per frame and camFile set")
    log("  t: the frame's total time; match: matching time of all rounds; found: objects IPR kept")
    combos = ["0123", "012", "013", "023", "123"]
    for fr in cfg["ipr_probe_frames"]:
        for which in ("old", "new"):
            run_ = res["runs"][f"{which}_f{fr:04d}"]
            r, fl = run_["record"], run_["flat"]
            det0 = r["rounds"][0]["det"]
            parts = []
            for c in combos:
                rounds = [x for x in r["rounds"] if x["cams"] == c]
                if rounds:
                    parts.append(f"{c}: " + "+".join(str(x.get("matched", 0)) for x in rounds)
                                 + f" ({sum(x.get('t_match', 0) for x in rounds):.2f}s)")
            log(f"  f{fr:4d} {which}: t {r['t_total']:6.2f}s match {fl['t_match_all']:6.2f}s found {r['ipr_found']:5d}"
                f" | det {det0} | " + "  ".join(parts))
    case.save_json(OUT_JSON, res)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwrote {OUT_JSON}\n      {OUT_TXT}")


if __name__ == "__main__":
    main()
