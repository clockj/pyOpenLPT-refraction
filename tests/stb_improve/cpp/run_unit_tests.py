"""Build and run the campaign's C++ unit tests in every variant (test U1).

    python run_unit_tests.py [VARIANT ...]

Variants (default: all):
  head          HEAD's Matrix (heap storage), from the worktree build-stb/src-base;
  head-asan     the same with AddressSanitizer;
  inline9       the working tree's Matrix with OPENLPT_MATRIX_INLINE=9;
  inline3       the same with OPENLPT_MATRIX_INLINE=3;
  inline9-check, inline3-check   with the index checks (OPENLPT_MATRIX_CHECKS);
  inline9-asan, inline3-asan     with AddressSanitizer;
  default       the working tree's default (3 since 5 Oct evening).
Each builds in build-stb/cpp-VARIANT and writes its output to ../results/u1_VARIANT.txt; a summary line per variant
is printed.  The value tests are the same in every variant, so head and inline* must all pass.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
ASAN_BIN = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64"
BASE_CXX = "/DWIN32 /D_WINDOWS /EHsc"
HEAD_SRC = os.path.join(REPO, "build-stb", "src-base")

VARIANTS = {
    "head": (HEAD_SRC, [], False),
    "head-asan": (HEAD_SRC, [], True),
    "inline9": (REPO, ["OPENLPT_MATRIX_INLINE=9"], False),
    "inline3": (REPO, ["OPENLPT_MATRIX_INLINE=3"], False),
    "inline9-check": (REPO, ["OPENLPT_MATRIX_INLINE=9", "OPENLPT_MATRIX_CHECKS"], False),
    "inline3-check": (REPO, ["OPENLPT_MATRIX_INLINE=3", "OPENLPT_MATRIX_CHECKS"], False),
    "inline9-asan": (REPO, ["OPENLPT_MATRIX_INLINE=9"], True),
    "inline3-asan": (REPO, ["OPENLPT_MATRIX_INLINE=3"], True),
    "default": (REPO, [], False),
}


def main():
    names = sys.argv[1:] or list(VARIANTS)
    os.makedirs(RESULTS, exist_ok=True)
    total_fail = 0
    for name in names:
        src, defs, asan = VARIANTS[name]
        bdir = os.path.join(REPO, "build-stb", f"cpp-{name}")
        config = "RelWithDebInfo" if asan else "Release"
        flags = BASE_CXX + "".join(f" /D{d}" for d in defs) + (" /fsanitize=address" if asan else "")
        cfg = ["cmake", "-S", HERE, "-B", bdir, "-G", "Visual Studio 17 2022", "-A", "x64",
               f"-DOPENLPT_SRC={src}", f"-DCMAKE_CXX_FLAGS={flags}"]
        if asan:
            cfg.append("-DCMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO=/debug /INCREMENTAL:NO")
        out_path = os.path.join(RESULTS, f"u1_{name}.txt")
        with open(out_path, "w", encoding="utf-8", newline="\n") as out:
            out.write(f"variant {name}: src {src}, flags {flags}, config {config}\n")
            out.flush()
            for cmd in (cfg, ["cmake", "--build", bdir, "--config", config, "--", "/m", "/v:minimal"]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    out.write(r.stdout[-5000:] + r.stderr[-5000:])
                    print(f"{name}: BUILD FAILED (see {out_path})")
                    total_fail += 1
                    break
            else:
                env = dict(os.environ)
                if asan:
                    env["PATH"] = ASAN_BIN + os.pathsep + env["PATH"]
                exe = os.path.join(bdir, config, "test_matrix_storage.exe")
                r = subprocess.run([exe], capture_output=True, text=True, env=env)
                out.write(r.stdout + r.stderr + f"exit code {r.returncode}\n")
                last = [ln for ln in r.stdout.splitlines() if "checks," in ln]
                first = r.stdout.splitlines()[0] if r.stdout else ""
                asan_err = "AddressSanitizer" in (r.stdout + r.stderr)
                print(f"{name}: {first} | {last[-1] if last else 'no summary'} | exit {r.returncode}"
                      + (" | ASAN ERROR" if asan_err else ""))
                total_fail += (r.returncode != 0) + asan_err
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
