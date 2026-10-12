"""L0: without a correction nothing changes (plan section 4). Build vcf (this change) against p4c (today's library).

    python l0_off.py run NAME          # one run: t1_p4c_a, t1_p4c_b, t1_vcf, syn_p4c, syn_vcf
    python l0_off.py pytest BUILD      # the repository's pytest suite with BUILD's module (p4c or vcf)
    python l0_off.py compare           # the verdict

- T1: the m15 run's config (cases/cyl_20261007/data/runs/s3_final_m15/config_final_m15.json, no correction) on frames
  0-60, run twice with p4c (does a run repeat exactly?) and once with vcf. Pass: every Results file byte-identical.
- synthetic: the self-calibration's 2-camera config tests/cylinder_selfcal/runs/configs/Gcal4c2_n8_r2_tol1.0.json
  (40 frames), with p4c and vcf. Pass: byte-identical.
- pytest from the repository's root, the build's module imported first (conftest.py would put build-py/ first on the
  path), without the two files that cannot be imported (pyOpenLPT's CLAUDE.md: they need packages not in the
  repository): the same tests pass and fail with both.
Runs go to runs/l0_<NAME>/ (git-ignored), with OMP_NUM_THREADS = 12. Writes l0_off_<part>_out.txt and .json here.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import vcf_common as vc

import stbruns  # noqa: E402  (tests/cylinder_stb)

T1_CFG = os.path.join(vc.CASE_RUNS, "s3_final_m15", "config_final_m15.json")
SYN_CFG = os.path.join(vc.SELFCAL, "runs", "configs", "Gcal4c2_n8_r2_tol1.0.json")
RUNS = {"t1_p4c_a": ("t1", "p4c"), "t1_p4c_b": ("t1", "p4c"), "t1_vcf": ("t1", "vcf"),
        "syn_p4c": ("syn", "p4c"), "syn_vcf": ("syn", "vcf")}


def t1_config():
    c = vc.load_json(T1_CFG)
    c["frames"] = [0, 60]
    c["output_dir"] = "Results"
    c["_comment"] = "L0 (tests/vsc_correction_field): T1's m15 config, frames 0-60, no correction"
    path = os.path.join(vc.RUNS, "configs", "l0_t1.json")
    vc.save_json(path, c)
    return path


def run(name):
    kind, build = RUNS[name]
    cfg = t1_config() if kind == "t1" else SYN_CFG
    rd = os.path.join(vc.RUNS, f"l0_{name}")
    info = stbruns.run(cfg, rd, build, env={"OMP_NUM_THREADS": str(vc.THREADS_MAX), "PYTHONDONTWRITEBYTECODE": "1"})
    log = vc.Log(f"l0_off_run_{name}")
    log(f"{name}: {cfg} with build {build} -> {rd}: exit {info['returncode']}, {info['wall_s']:.0f} s")
    log.save(info)


def pytest(build):
    out = os.path.join(vc.REPO, "build-stb", build, "out")
    code = ("import sys; sys.path.insert(0, {o!r}); import pyopenlpt; print('[l0] pyopenlpt', pyopenlpt.__file__, "
            "flush=True); import pytest; sys.exit(pytest.main(['-q', '-rfE', '-p', 'no:cacheprovider', "
            "'--basetemp', {t!r}, '--ignore=tests/test_p0_bootstrap_failure.py', '--ignore=tests/test_plane_pipeline.py', 'tests']))").format(o=out, t=os.path.join(vc.RUNS, f"l0_pytest_tmp_{build}"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(vc.THREADS_MAX))
    p = subprocess.run([sys.executable, "-c", code], cwd=vc.REPO, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    text = p.stdout + "\n" + p.stderr
    with open(os.path.join(vc.HERE, f"l0_off_pytest_{build}_console.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)))
    summary = [ln for ln in text.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    log = vc.Log(f"l0_off_pytest_{build}")
    log(f"pytest with {build}'s module: exit {p.returncode}; {summary[-1] if summary else '(no summary)'}")
    for f in failed:
        log(f"  {f}")
    log.save({"build": build, "returncode": p.returncode, "summary": summary[-1] if summary else "",
              "failed": failed, "module_line": next((ln for ln in text.splitlines() if ln.startswith("[l0]")), "")})


def compare():
    log = vc.Log("l0_off_compare")
    out = {}
    ok_all = True
    for a, b, what in (("t1_p4c_a", "t1_p4c_b", "T1, p4c twice (does a run repeat?)"),
                       ("t1_p4c_a", "t1_vcf", "T1, p4c against vcf"),
                       ("syn_p4c", "syn_vcf", "synthetic, p4c against vcf")):
        ra, rb = os.path.join(vc.RUNS, f"l0_{a}"), os.path.join(vc.RUNS, f"l0_{b}")
        c = stbruns.compare(ra, rb)
        diff = [f for f, r in c["files"].items() if r != "identical"]
        log(f"{what}: {c['identical_files']}/{c['compared_files']} files identical; only in {a}: {len(c['only_a'])}, "
            f"only in {b}: {len(c['only_b'])} -> {'IDENTICAL' if c['all_identical'] else 'DIFFERENT'}")
        for f in diff[:10]:
            log(f"  differs: {f} {c['files'][f]}")
        out[f"{a}_vs_{b}"] = c
        ok_all &= c["all_identical"]
    pt = {}
    for b in ("p4c", "vcf"):
        p = os.path.join(vc.HERE, f"l0_off_pytest_{b}.json")
        if os.path.exists(p):
            pt[b] = vc.load_json(p)
    if len(pt) == 2:
        same = pt["p4c"]["failed"] == pt["vcf"]["failed"]
        log(f"pytest: p4c '{pt['p4c']['summary']}', vcf '{pt['vcf']['summary']}'; the same failures: {same}")
        out["pytest_same_failures"] = same
        ok_all &= same
    log(f"L0: {'PASS' if ok_all else 'FAIL'}")
    out["pass"] = bool(ok_all)
    log.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "pytest", "compare"])
    ap.add_argument("arg", nargs="?")
    a = ap.parse_args()
    if a.what == "run":
        run(a.arg)
    elif a.what == "pytest":
        pytest(a.arg)
    else:
        compare()


if __name__ == "__main__":
    main()
