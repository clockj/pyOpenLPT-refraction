"""Does T1\\after_vsc\\ on X: rerun as it is?  Its config.json (T0's cameras, the frames and the OTF there) run for the
first 4 frames into runs/xcheck/ here (nothing is written to X:), and the objects IPR finds per frame compared with
the original run's (s3_final_m15vc2).

    python v3_xcheck.py

Writes v3_xcheck_out.txt and .json here.
"""
from __future__ import annotations

import json
import os
import re

import selfcal as sc

import stbruns  # noqa: E402

SRC = r"X:\Shijie Zhong\Refraction\Processed\20261007\T1\after_vsc"


def ipr(path, n=4):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [int(x) for x in re.findall(r"IPR FINISH! FOUND (\d+) OBJECTS", fh.read())][:n]


def main():
    log = sc.Log("v3_xcheck")
    c = sc.load_json(os.path.join(SRC, "config.json"))
    c["frames"] = [0, 3]
    c["objects"][0]["tracer"]["otf_file"] = os.path.join(SRC, "otf_shape.txt")   # the run dir here is not X:'s
    d = os.path.join(sc.RUNS, "xcheck")
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, "config.json")
    with open(cfg, "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=1)
    log(f"X:'s after_vsc config, frames 0-3: cameras {[x['file'] for x in c['cameras']]}; lists {c['image_lists']}")
    info = stbruns.run(cfg, os.path.join(d, "run"), sc.BUILD, env={"OMP_NUM_THREADS": "8"})
    a = ipr(os.path.join(d, "run", "log.txt"))
    b = ipr(os.path.join(sc.CASE_RUNS, "s3_final_m15vc2", "log.txt"))
    log(f"  exit {info['returncode']}; objects IPR found, frames 0-3: the rerun from X: {a}, the original run {b}: "
        f"{'equal' if a == b else 'DIFFERENT'}")
    log.save({"rerun": a, "original": b, "equal": a == b, "info": info})


if __name__ == "__main__":
    main()
