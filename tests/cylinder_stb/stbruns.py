"""Run OpenLPT's STB on a case's config.json in a fresh process, and compare runs (cylinder STB tests).

    run(config_json, run_dir, build, exe=False, env=None) -> info dict (also run_dir/run_info.json)
    compare(run_a, run_b) -> dict: per output file identical bytes, or the largest numeric difference per column

A build is a name under build-stb/ (build_variant.py's out/ folder).  The config is copied into run_dir and run there
(its relative output_dir lands in run_dir); stdout and stderr go to run_dir/log.txt.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import subprocess
import time

import cstb

PY = r"C:\Users\zcloc\.conda\envs\refract\python.exe"


def run(config_json: str, run_dir: str, build: str, exe: bool = False, env: dict | None = None,
        timeout: float | None = None) -> dict:
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    cfg_path = os.path.join(run_dir, os.path.basename(config_json))   # config.json or config.txt
    shutil.copy2(config_json, cfg_path)
    bdir = os.path.join(cstb.REPO, "build-stb", build, "out")
    full_env = dict(os.environ, PYTHONUNBUFFERED="1")
    if env:
        full_env.update(env)
    if exe:
        cmd = [os.path.join(bdir, "OpenLPT.exe"), cfg_path]
    else:
        code = ("import sys; sys.path.insert(0, {b!r}); import pyopenlpt as lpt; "
                "print('[stbruns] pyopenlpt', lpt.__file__, flush=True); lpt.run({c!r})").format(b=bdir, c=cfg_path)
        cmd = [PY, "-u", "-c", code]
    t0 = time.perf_counter()
    with open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8", newline="\n") as log:
        p = subprocess.run(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, env=full_env, timeout=timeout)
    info = {"config": config_json, "build": build, "exe": exe, "env": env or {}, "returncode": p.returncode,
            "wall_s": round(time.perf_counter() - t0, 2), "end": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(cstb.REPO, "build-stb", build, "out", "build_info.json"), encoding="utf-8") as fh:
        bi = json.load(fh)
    info["build_info"] = {k: bi.get(k) for k in ("name", "git_head", "git_diff_sha256", "start")}
    with open(os.path.join(run_dir, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    return info


def _files(root: str) -> list[str]:
    out = []
    for d, _subs, names in os.walk(root):
        for n in names:
            out.append(os.path.relpath(os.path.join(d, n), root))
    return sorted(out)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.reader(fh))


def compare(run_a: str, run_b: str, sub: str = "Results") -> dict:
    """Every file under run_a/sub against run_b/sub: identical bytes, or for CSVs the row count and the largest
    absolute difference of each numeric column (rows matched by position)."""
    ra, rb = os.path.join(run_a, sub), os.path.join(run_b, sub)
    fa, fb = _files(ra), _files(rb)
    out = {"only_a": sorted(set(fa) - set(fb)), "only_b": sorted(set(fb) - set(fa)), "files": {}}
    n_ident = 0
    for f in sorted(set(fa) & set(fb)):
        with open(os.path.join(ra, f), "rb") as fh:
            ba = fh.read()
        with open(os.path.join(rb, f), "rb") as fh:
            bb = fh.read()
        if ba == bb:
            n_ident += 1
            out["files"][f] = "identical"
            continue
        rec = {"identical": False}
        if f.lower().endswith(".csv"):
            A, B = _read_csv(os.path.join(ra, f)), _read_csv(os.path.join(rb, f))
            rec["rows"] = [len(A), len(B)]
            if A and B:
                head = A[0]
                maxd = {}
                for row_a, row_b in zip(A[1:], B[1:]):
                    for k, (x, y) in enumerate(zip(row_a, row_b)):
                        try:
                            d = abs(float(x) - float(y))
                        except ValueError:
                            continue
                        name = head[k] if k < len(head) else str(k)
                        maxd[name] = max(maxd.get(name, 0.0), d)
                rec["max_abs_diff"] = maxd
        out["files"][f] = rec
    out["identical_files"] = n_ident
    out["compared_files"] = len(set(fa) & set(fb))
    out["all_identical"] = (n_ident == out["compared_files"] and not out["only_a"] and not out["only_b"])
    return out
