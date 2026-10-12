"""B1: the batch's end-of-run steps on a finished smoke run, before the batch relies on them.

  - run_all.check_finished on S1 (frames 0-50): no problems, 51 frames;
  - run_all.deliver to a local test folder instead of X: (case.output_dir patched): robocopy, the extra files, the
    size check and DONE.json;
  - the delivered folder holds every output file of S1, byte for byte.
The smoke folder itself is copied first, so S1 stays as it was.

    python tests/batch_steps.py
Writes tests/batch_steps.txt.
"""
from __future__ import annotations

import filecmp
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import run_all  # noqa: E402


def main():
    cfg = case.load_config()
    smoke = os.path.join(cfg["staging_root"], "smoke")
    src = os.path.join(smoke, "s1_10Hz_r_b_1_f0-50")
    work = os.path.join(smoke, "b1_batch_steps")
    if os.path.exists(work):
        shutil.rmtree(work)
    sd = os.path.join(work, "stage")
    shutil.copytree(src, sd)
    dest = os.path.join(work, "delivered", "Results_tracer")
    os.makedirs(os.path.join(dest, "camFile"))          # as make_configs.py --case leaves it
    with open(os.path.join(dest, "camFile", "cam0.txt"), "w") as f:
        f.write("placeholder\n")
    lines = ["B1 the batch's end-of-run steps on S1 (a copy), delivered to a local folder", ""]
    problems, info = run_all.check_finished(cfg, "10Hz_r_b_1", sd)
    lines.append(f"check_finished: problems {problems}, {info['n_frames']} frames, frames {info['frames']}")
    ok = not problems and info["n_frames"] == 51

    case.output_dir = lambda c, r: dest  # deliver to the test folder
    batch = run_all.Batch(dict(cfg, staging_root=work))
    delivered = run_all.deliver(batch, "10Hz_r_b_1", sd)
    lines.append(f"deliver: returned {delivered}; DONE.json {os.path.exists(os.path.join(sd, 'DONE.json'))}")
    ok &= delivered and os.path.exists(os.path.join(sd, "DONE.json"))
    mism, n = [], 0
    for dp, _dn, fn in os.walk(os.path.join(src, cfg["output_folder"])):
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, os.path.join(src, cfg["output_folder"]))
            n += 1
            if not filecmp.cmp(p, os.path.join(dest, rel), shallow=False):
                mism.append(rel)
    extra = ["log.txt", "run_info.json", "resources.csv", "config_staging.json", "camFile/cam0.txt"]
    have = [e for e in extra if os.path.exists(os.path.join(dest, e))]
    lines.append(f"delivered output files byte-identical to S1's: {n - len(mism)} of {n} {mism[:3]}")
    lines.append(f"extra files present: {have}")
    ok &= not mism and have == extra
    lines += ["", "PASS" if ok else "FAIL"]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "batch_steps.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


if __name__ == "__main__":
    main()
