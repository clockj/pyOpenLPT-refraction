"""Run the whole campaign on the final code, in order, and collect each test's verdict lines.

Each script writes its own *_out.txt and *.json beside it; this runner records which ran, how long they took and
the PASS/FAIL/GATE lines into run_all_out.txt.  T5 (50 random rigs, ~25 min) runs only with --with-t5.
Usage: python run_all.py [--with-t5]
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["p0_sympy_derivations.py", "t0_fast_vs_cpp.py", "t0b_cpp_direction_roundoff.py", "t1_derivatives.py",
           "t1b_ba_jacobian.py", "t2_observability.py", "t2b_error_modes.py", "t3_noiseless_recovery.py",
           "t4_noisy_recovery.py", "t5a_bootstrap_diagnostics.py", "t6_gauge_frame.py", "t7_heldout.py",
           "t8_robustness.py", "t9_tube_constants.py", "t10_export.py"]


def main():
    scripts = list(SCRIPTS)
    if "--with-t5" in sys.argv:
        scripts.insert(scripts.index("t5a_bootstrap_diagnostics.py"), "t5_from_scratch.py")
    lines = []
    for s in scripts:
        t0 = time.time()
        # the SymPy check runs under plain python (it needs only sympy); the rest need the refract env
        exe = sys.executable
        p = subprocess.run([exe, "-u", s], cwd=HERE, capture_output=True, text=True)
        dt = time.time() - t0
        if s == "p0_sympy_derivations.py":           # the only script that does not write its own log
            (HERE / "p0_sympy_derivations_out.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
        verdicts = [ln.strip() for ln in p.stdout.splitlines()
                    if ln.strip().startswith(("PASS", "FAIL", "GATE", "f target", "literal")) or "checks:" in ln]
        head = f"{s}: exit {p.returncode}, {dt:.0f} s"
        print(head, flush=True)
        lines.append(head)
        for v in verdicts:
            print("    " + v, flush=True)
            lines.append("    " + v)
        if p.returncode != 0:
            tail = p.stderr.strip().splitlines()[-5:]
            lines += ["    stderr: " + t for t in tail]
    (HERE / "run_all_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
