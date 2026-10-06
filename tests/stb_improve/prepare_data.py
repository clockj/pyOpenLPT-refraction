"""Copy the benchmark inputs of the stb_improve campaign to the local disk (tests/stb_improve/data/, git-ignored).

    python prepare_data.py [--frames N]

The benchmarks use the tracking case cases/stb_20260727_10Hz_r_b_1, setup new_cams: the new flat-window camFiles,
the T0 view volume and the case's tracerConfig.  Reading images and writing tracks over the network share (X:) adds
minutes per save and noise per frame, so the inputs are copied once:
  - data/imgFile_mask/cam0..cam3/img000000.tif ... (frames 0..N-1, default 311), each copy checked by SHA-256 against
    the bytes read from the share;
  - data/camFile/cam0..cam3.txt and data/tracerConfig.txt, byte-exact copies of the ones new_cams ran with;
  - data/imgLists/cam0..cam3ImageNames.txt, listing the copies relative to a run folder runs/<run>/ (OpenLPT resolves
    image paths from the config's folder).
data/manifest.json records sources, sizes and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CASE = r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results\10Hz_r_b_1"
RUN = os.path.join(CASE, "stb_new_calib", "new_cams")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_checked(src, dst):
    """Copy src to dst; return the hash, after checking the written file hashes the same as the bytes read."""
    with open(src, "rb") as f:
        data = f.read()
    h = hashlib.sha256(data).hexdigest()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    assert sha256(dst) == h, f"copy mismatch: {dst}"
    return h, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=311)
    args = ap.parse_args()
    t0 = time.time()
    manifest = {"case": CASE, "run": RUN, "frames": args.frames, "files": {}}

    for name in ["camFile/cam0.txt", "camFile/cam1.txt", "camFile/cam2.txt", "camFile/cam3.txt", "tracerConfig.txt"]:
        h, n = copy_checked(os.path.join(RUN, name), os.path.join(DATA, name))
        manifest["files"][name] = {"sha256": h, "bytes": n}

    for cam in range(4):
        lines = []
        for fr in range(args.frames):
            rel = f"imgFile_mask/cam{cam}/img{fr:06d}.tif"
            dst = os.path.join(DATA, rel)
            src = os.path.join(CASE, rel)
            if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                h, n = sha256(dst), os.path.getsize(dst)
            else:
                h, n = copy_checked(src, dst)
            manifest["files"][rel] = {"sha256": h, "bytes": n}
            lines.append(f"../../data/{rel}")
        os.makedirs(os.path.join(DATA, "imgLists"), exist_ok=True)
        with open(os.path.join(DATA, "imgLists", f"cam{cam}ImageNames.txt"), "w", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"cam{cam}: {args.frames} images ({time.time() - t0:.0f} s)", flush=True)

    with open(os.path.join(DATA, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
