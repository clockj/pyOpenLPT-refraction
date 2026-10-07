"""F7 (fix of the four-round flip, 2026-10-07): keep, before anything is rerun, every output the reruns overwrite.

    python tests/cylinder_wand_gui/f7_keep_before.py

Copies (never moves; an existing copy is not overwritten):
  - F3's outputs (f3_real_data.json, _out.txt, _console.txt; data/f3_<run>/) to the same names with "_before_flip4";
  - every calibration log of the variants of runs 20261006/T0 and T2 (X: ...\\Results\\checks\\<name>\\) and of the
    delivered calibrations (...\\Results\\calibration_log.txt, T0's first delivery in before_stage4_fix\\) to
    data/f7_logs/<run>_<name>_calibration_log.txt, the input of f7_flip_rule.py;
  - the variants' result folders on X: to ...\\Results\\before_flip4\\checks\\<name>\\;
  - the cases' variant and held-out tests' outputs (T0: t3_t6_variants, d2_shape_prior, d8_held_out; T2: t3_variants,
    t5_k1, d8_held_out) to the case's before_flip4\\tests\\.
Prints what it copied.
"""
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = {"T0": r"X:\Shijie Zhong\Refraction\Processed\20261006\T0\Results",
           "T2": r"X:\Shijie Zhong\Refraction\Processed\20261006\T2\Results"}
CASES = {"T0": os.path.join(REPO, "cases", "cyl_20261006_T0_wand"),
         "T2": os.path.join(REPO, "cases", "cyl_20261006_T2_wand")}
CASE_TESTS = {"T0": ["t3_t6_variants", "d2_shape_prior", "d8_held_out"],
              "T2": ["t3_variants", "t5_k1", "d8_held_out"]}
F3 = ["f3_T0_camera", "f3_T0_camera_n2", "f3_T0_camera_noshare", "f3_T0_none", "f3_T2_camera", "f3_T2_camera_n2",
      "f3_T2_camera_noshare", "f3_T2_none"]


def copy(src, dst):
    if not os.path.exists(src):
        print(f"  missing: {src}")
        return
    if os.path.exists(dst):
        print(f"  kept (already there): {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"  {src} -> {dst}")


def main():
    print("F3's outputs")
    for name in ("f3_real_data.json", "f3_real_data_out.txt", "f3_real_data_console.txt"):
        copy(os.path.join(HERE, name), os.path.join(HERE, name.replace("f3_real_data", "f3_real_data_before_flip4")))
    for tag in F3:
        copy(os.path.join(HERE, "data", tag), os.path.join(HERE, "data", tag + "_before_flip4"))
    print("the calibration logs (F7's input)")
    logs = os.path.join(HERE, "data", "f7_logs")
    for run, res in RESULTS.items():
        copy(os.path.join(res, "calibration_log.txt"), os.path.join(logs, f"{run}_delivered_calibration_log.txt"))
        checks = os.path.join(res, "checks")
        for name in sorted(os.listdir(checks)):
            if os.path.isfile(os.path.join(checks, name, "calibration_log.txt")):
                copy(os.path.join(checks, name, "calibration_log.txt"),
                     os.path.join(logs, f"{run}_{name}_calibration_log.txt"))
    copy(os.path.join(RESULTS["T0"], "before_stage4_fix", "calibration_log.txt"),
         os.path.join(logs, "T0_first_delivery_calibration_log.txt"))
    print("the variants' results on X:")
    for run, res in RESULTS.items():
        checks = os.path.join(res, "checks")
        for name in sorted(os.listdir(checks)):
            if os.path.isdir(os.path.join(checks, name)):
                copy(os.path.join(checks, name), os.path.join(res, "before_flip4", "checks", name))
    print("the cases' test outputs")
    for run, names in CASE_TESTS.items():
        tests = os.path.join(CASES[run], "tests")
        for name in names:
            for suffix in (".json", "_out.txt", "_console.txt"):
                copy(os.path.join(tests, name + suffix), os.path.join(CASES[run], "before_flip4", "tests", name + suffix))


if __name__ == "__main__":
    main()
