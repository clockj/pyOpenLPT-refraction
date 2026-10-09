"""T-copy (plan: Phase T): one local copy of the tracked frames of T1 (three cameras, frames 0-249: config
tracers.track_raw_frames) in data/T1raw/<cam>/ (git-ignored), checked against the share by sha256 on three
frames per camera.

    python tests/t_copy_t1.py

Writes t_copy_t1_out.txt beside itself and data/T1raw/manifest.json.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import t1  # noqa: E402

lines = []


def log(msg=""):
    print(msg, flush=True)
    lines.append(str(msg))


t0 = time.time()
log(f"T-copy: local copy of {t1.load_config()['tracers']['raw_dir']} -> {t1.LOCAL_RAW}")
r = t1.copy_local(log)
log(f"done in {time.time() - t0:.0f} s; manifest {os.path.join(t1.LOCAL_RAW, 'manifest.json')}")
with open(os.path.join(HERE, "t_copy_t1_out.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
