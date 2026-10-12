"""F4 (fixes plan 2026-10-06-fixes-plan.md, section 3): tests/cylinder_wand's noise tests T11a, T11c-T11e and the
weight tests T12a-T12e, run once on the library before the fixes and once after ("flip4", 7 Oct: once more after the
four-round flip fix, F7, in its own folder run_flip4/; the library as it is now, like "after").

    python f4_synthetic.py before|after|flip4 [TEST ...]      (default: all nine, in order)

Each version runs in its own folder, run_before/ or run_after/ beside this script, so the campaign's saved outputs
in tests/cylinder_wand are not touched:
  - the test scripts and their shared modules are copied there unchanged, except two lines added to the copies
    (asserted below):
      rigs.py        with F4_CYLINDER_WAND_DIR set, that folder goes first on sys.path: the "before" run imports
                     before_lib/cylinder_wand.py, the library file as committed (git HEAD 8f46a01) before the fixes;
      t11_common.py  and t12_common.py: each fit's record also keeps which of the fixes acted (sol.info "cycle",
                     noise_weights "flipped"; None before the fixes);
  - the saved outputs two of them read are copied too: t5_from_scratch.json (T12d) and t11b_noise_scratch.json (T12e).
    T12a and T12b read T11c's and T11a's outputs, which the same run writes first.
Every test's output (_out.txt, .json) lands in the run folder, with each script's console in <test>_console.txt and
the times in run_<version>/f4_times.json.  f4_compare.py compares the two runs.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1] / "cylinder_wand"
PY = sys.executable
TESTS = ["t11a_noise_sweep", "t11c_unequal_noise", "t11d_heavy_tails", "t11e_systematic", "t12a_weights_unequal",
         "t12b_weights_equal", "t12c_two_cameras", "t12d_scratch_t5", "t12e_scratch_5px"]
SUPPORT = ["rigs.py", "t11_common.py", "t12_common.py"]
REFS = ["t5_from_scratch.json", "t11b_noise_scratch.json"]
PATCHES = {
    "rigs.py": [(
        "REPO = HERE.parents[1]\n",
        "REPO = HERE.parents[3]                              # F4 copy: run_<version> -> pyOpenLPT\n"), (
        "import cyl_model as cm  # noqa: E402\n",
        "if os.environ.get(\"F4_CYLINDER_WAND_DIR\"):          # F4 copy: the library before the fixes\n"
        "    sys.path.insert(0, os.environ[\"F4_CYLINDER_WAND_DIR\"])\n"
        "import cyl_model as cm  # noqa: E402\n")],
    "t11_common.py": [(
        "\"lm_iterations\": len(sol.history), \"rounds\": sol.info.get(\"rounds\"),",
        "\"lm_iterations\": len(sol.history), \"rounds\": sol.info.get(\"rounds\"), \"cycle\": sol.info.get(\"cycle\"),"
        " \"flipped\": (sol.info.get(\"noise_weights\") or {}).get(\"flipped\"),")],
    "t12_common.py": [(
        "\"rounds\": sol.info.get(\"rounds\")}",
        "\"rounds\": sol.info.get(\"rounds\"), \"cycle\": sol.info.get(\"cycle\"), "
        "\"flipped\": nw.get(\"flipped\")}")],
}


def setup(folder: Path) -> None:
    folder.mkdir(exist_ok=True)
    for name in SUPPORT + [t + ".py" for t in TESTS] + REFS:
        shutil.copy2(CAMPAIGN / name, folder / name)
    for name, subs in PATCHES.items():
        p = folder / name
        text = p.read_bytes().decode("utf-8")
        nl = "\r\n" if "\r\n" in text else "\n"          # keep the file's own line ends
        for old, new in subs:
            old, new = old.replace("\n", nl), new.replace("\n", nl)
            assert text.count(old) == 1, f"{name}: the patch's anchor is not unique: {old!r}"
            text = text.replace(old, new)
        p.write_bytes(text.encode("utf-8"))


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else ""
    if version not in ("before", "after", "flip4"):
        raise SystemExit("usage: python f4_synthetic.py before|after|flip4 [TEST ...]")
    tests = sys.argv[2:] or TESTS
    folder = HERE / f"run_{version}"
    setup(folder)
    env = dict(os.environ)
    env.pop("F4_CYLINDER_WAND_DIR", None)
    if version == "before":
        env["F4_CYLINDER_WAND_DIR"] = str(HERE / "before_lib")
    times_path = folder / "f4_times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else {}
    for t in tests:
        t0 = time.time()
        with open(folder / f"{t}_console.txt", "w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, str(folder / f"{t}.py")], cwd=str(folder), env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        times[t] = {"seconds": time.time() - t0, "returncode": rc}
        times_path.write_text(json.dumps(times, indent=1))
        print(f"{version} {t}: exit {rc}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
