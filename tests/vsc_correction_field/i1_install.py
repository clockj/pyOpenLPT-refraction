"""I1: the install (decision 1, 2026-10-09-plan.md): build_local.py rebuilds the GUI's module and exe.

    python i1_install.py before        # the GUI's build before: hashes and times
    (from the repository's root, refract's python:  python build_local.py  > i1_build_console.txt)
    python i1_install.py after         # after: hashes, times, the build's warnings and errors, the module's new API
    python i1_install.py run NAME      # one tracking run: t1_gui (installed exe, no correction),
                                       #   t1c_gui / t1c_vcf (installed / vcf exe, T1 frames 0-60 with a correction)
    python i1_install.py compare       # the runs against each other and against L0's vcf run: byte for byte
    python i1_install.py pytest [NAME] # the repository's pytest with the installed module (conftest.py: build-py);
                                       # NAME: the output's stem (default i1_install_pytest)

The runs use L0's T1 config (the m15 config on frames 0-60, runs/configs/l0_t1.json); the corrected one adds G1's
pass-2 corrections (projects/t1_gui/camFile_VSC). Runs go to runs/i1_<NAME>/ (git-ignored), OMP_NUM_THREADS = 12,
one at a time. Writes i1_install_<part>_out.txt and .json here.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

import vcf_common as vc

import stbruns  # noqa: E402  (tests/cylinder_stb)

GUI_EXE = os.path.join(vc.REPO, "build", "Release", "OpenLPT.exe")
GUI_MOD_DIR = os.path.join(vc.REPO, "build-py", "Release")
VCF_EXE = os.path.join(vc.REPO, "build-stb", "vcf", "out", "OpenLPT.exe")
L0_CFG = os.path.join(vc.RUNS, "configs", "l0_t1.json")
CORR_DIR = os.path.join(vc.HERE, "projects", "t1_gui", "camFile_VSC")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def products():
    out = {}
    for p in [GUI_EXE] + [os.path.join(GUI_MOD_DIR, f) for f in sorted(os.listdir(GUI_MOD_DIR))
                          if f.endswith(".pyd")]:
        st = os.stat(p)
        out[os.path.relpath(p, vc.REPO)] = {"sha256": sha(p), "bytes": st.st_size,
                                            "modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                                                      time.localtime(st.st_mtime))}
    return out


def before():
    log = vc.Log("i1_install_before")
    prod = products()
    for k, v in prod.items():
        log(f"{k}: {v['sha256'][:16]}..., {v['bytes']} bytes, {v['modified']}")
    log.save(prod)


def after():
    log = vc.Log("i1_install_after")
    prev = vc.load_json(os.path.join(vc.HERE, "i1_install_before.json"))
    prod = products()
    for k, v in prod.items():
        old = prev.get(k, {})
        log(f"{k}: {v['sha256'][:16]}..., {v['bytes']} bytes, {v['modified']} "
            f"({'changed' if v['sha256'] != old.get('sha256') else 'UNCHANGED'}; before {old.get('modified')})")
    console = os.path.join(vc.HERE, "i1_build_console.txt")
    text = open(console, encoding="utf-8", errors="replace").read() if os.path.exists(console) else ""
    import re
    m = re.findall(r"(\d+) Warning\(s\)\s+(\d+) Error\(s\)", text)
    log(f"build console: {'Build done' in text and 'finished' or 'NOT finished'}; MSBuild summary (warnings, "
        f"errors): {m[-1] if m else 'none found'}")
    # the module, in a fresh process: the correction's API
    code = ("import sys; sys.path.insert(0, {d!r}); import pyopenlpt as lpt; "
            "names = ['ImageCorrection', 'CorrectedCameraModel', 'loadCameraModel', 'detectLines']; "
            "print('module', lpt.__file__); print('api', [n for n in names if hasattr(lpt, n)]); "
            "c = lpt.Camera(); print('camera', [n for n in ('setCorrection', 'clearCorrection', 'hasCorrection', "
            "'correctionFile', 'cameraModel') if hasattr(c, n)])").format(d=GUI_MOD_DIR)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=vc.HERE)
    for ln in (p.stdout + p.stderr).strip().splitlines():
        log(f"  {ln}")
    log.save({"products": prod, "before": prev, "msbuild": m[-1] if m else None, "module_check": p.stdout,
              "module_check_err": p.stderr, "module_exit": p.returncode})


def corrected_config():
    c = vc.load_json(L0_CFG)
    for k, cam in enumerate(c["cameras"]):
        cam["correction"] = os.path.join(CORR_DIR, f"vsc_cam{k + 1}_corr_p2.txt")
    c["_comment"] = "I1 (tests/vsc_correction_field): L0's T1 config (frames 0-60) with G1's pass-2 corrections"
    path = os.path.join(vc.RUNS, "configs", "i1_t1c.json")
    vc.save_json(path, c)
    return path


def run(name):
    cfg = L0_CFG if name == "t1_gui" else corrected_config()
    exe = GUI_EXE if name.endswith("_gui") else VCF_EXE
    rd = os.path.join(vc.RUNS, f"i1_{name}")
    os.makedirs(rd, exist_ok=True)
    cfg_copy = os.path.join(rd, os.path.basename(cfg))
    with open(cfg, encoding="utf-8") as src, open(cfg_copy, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    env = dict(os.environ, OMP_NUM_THREADS=str(vc.THREADS_MAX), PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    with open(os.path.join(rd, "console.txt"), "w", encoding="utf-8") as fh:
        p = subprocess.run([exe, cfg_copy], cwd=rd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    info = {"name": name, "exe": exe, "exe_sha256": sha(exe), "config": cfg_copy, "returncode": p.returncode,
            "wall_s": time.time() - t0}
    vc.save_json(os.path.join(rd, "run_info.json"), info)
    log = vc.Log(f"i1_install_run_{name}")
    log(f"{name}: {exe} on {cfg_copy}: exit {p.returncode}, {info['wall_s']:.0f} s")
    log.save(info)


def results(rd):
    """Every file under the run's Results folder: relative path -> sha256."""
    res = os.path.join(rd, "Results")
    out = {}
    for root, _dirs, files in os.walk(res):
        for f in files:
            p = os.path.join(root, f)
            out[os.path.relpath(p, res).replace(os.sep, "/")] = sha(p)
    return out


def compare():
    log = vc.Log("i1_install_compare")
    pairs = [("t1_gui", os.path.join(vc.RUNS, "i1_t1_gui"), "L0's vcf run", os.path.join(vc.RUNS, "l0_t1_vcf")),
             ("t1c_gui", os.path.join(vc.RUNS, "i1_t1c_gui"), "t1c_vcf", os.path.join(vc.RUNS, "i1_t1c_vcf"))]
    out = {}
    for a, ra, b, rb in pairs:
        A, B = results(ra), results(rb)
        same = sorted(k for k in set(A) & set(B) if A[k] == B[k])
        diff = sorted(k for k in set(A) & set(B) if A[k] != B[k])
        only = sorted(set(A) ^ set(B))
        ok = bool(A) and not diff and not only
        out[a] = {"against": b, "files": len(A), "identical": len(same), "different": diff, "unpaired": only,
                  "pass": ok}
        log(f"{a} against {b}: {len(A)} result files, {len(same)} identical, {len(diff)} different, "
            f"{len(only)} unpaired -> {'PASS' if ok else 'FAIL'}" + (f"; different: {diff[:8]}" if diff else ""))
    cons = os.path.join(vc.RUNS, "i1_t1c_gui", "console.txt")
    if os.path.exists(cons):
        lines = [ln.strip() for ln in open(cons, encoding="utf-8", errors="replace") if "image correction" in ln]
        log(f"the installed exe's log on the corrected config: {lines}")
        out["correction_lines"] = lines
    log.save(out)


def pytest(name="i1_install_pytest"):
    """As L0's pytest, but with the module conftest.py picks (build-py/Release, the installed one)."""
    import re
    code = ("import sys; sys.path.insert(0, {d!r}); import pyopenlpt; print('[i1] pyopenlpt', pyopenlpt.__file__, "
            "flush=True); import pytest; sys.exit(pytest.main(['-q', '-rfE', '-p', 'no:cacheprovider', "
            "'--basetemp', {t!r}, '--ignore=tests/test_p0_bootstrap_failure.py', "
            "'--ignore=tests/test_plane_pipeline.py', 'tests']))").format(
        d=GUI_MOD_DIR, t=os.path.join(vc.RUNS, "i1_pytest_tmp"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(vc.THREADS_MAX))
    p = subprocess.run([sys.executable, "-c", code], cwd=vc.REPO, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    text = p.stdout + "\n" + p.stderr
    with open(os.path.join(vc.HERE, f"{name}_console.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M)))
    summary = [ln for ln in text.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    log = vc.Log(name)
    log(f"pytest with the installed module: exit {p.returncode}; {summary[-1] if summary else '(no summary)'}")
    log(next((ln for ln in text.splitlines() if ln.startswith("[i1]")), "(module line missing)"))
    for f in failed:
        log(f"  {f}")
    log.save({"returncode": p.returncode, "summary": summary[-1] if summary else "", "failed": failed})


def main():
    part = sys.argv[1]
    if part == "before":
        before()
    elif part == "after":
        after()
    elif part == "run":
        run(sys.argv[2])
    elif part == "compare":
        compare()
    elif part == "pytest":
        pytest(*sys.argv[2:3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
