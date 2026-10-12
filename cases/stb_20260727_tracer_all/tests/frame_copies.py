"""D2: are there other copied frames besides 256?  Checks every frame where the conversion's chunks begin (read only).

imgFile (the videos minus the background) was written in chunks of 256 frames: in imgFile/cam0 the file time jumps
at every multiple of 256 (256, 512, 768, ...), and frame 256 is a byte copy of frame 255 in all 48 sequences (D1).  This
compares the file at every multiple of 256 with the file before it, byte for byte, in one camera of imgFile_mask (what
STB reads; the copy at 256 is in every camera and in imgFile too), and confirms any copy found in all four cameras of
imgFile and imgFile_mask.  It also reports the time jump at each chunk start.

    python tests/frame_copies.py RUN [RUN ...] [--cam 0]
Writes tests/frame_copies.txt and frame_copies.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

CHUNK = 256
FOLDERS = ["imgFile", "imgFile_mask"]


def mtimes(folder):
    out = {}
    with os.scandir(folder) as it:
        for e in it:
            m = re.fullmatch(r"img(\d+)\.tif", e.name)
            if m:
                out[int(m.group(1))] = e.stat().st_mtime
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def is_copy(cfg, run, folder, cam, k):
    d = os.path.join(case.case_dir(cfg, run), folder, f"cam{cam}")
    return md5(os.path.join(d, cfg["image_name"].format(k - 1))) == md5(os.path.join(d, cfg["image_name"].format(k)))


def check(cfg, run, cam):
    t = mtimes(os.path.join(case.case_dir(cfg, run), "imgFile", "cam0"))
    frames = sorted(t)
    starts = list(range(CHUNK, frames[-1] + 1, CHUNK))
    rows = []
    for k in starts:
        row = {"frame": k, "jump_s": round(t[k] - t[k - 1], 1), "copy_cam": is_copy(cfg, run, "imgFile_mask", cam, k)}
        if row["copy_cam"]:
            row["copies"] = {f"{folder}/cam{c}": is_copy(cfg, run, folder, c, k) for folder in FOLDERS for c in range(4)}
        rows.append(row)
    return {"run": run, "cam": cam, "n_files": len(frames), "first": frames[0], "last": frames[-1], "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    cfg = case.load_config()
    out, lines = [], [f"D2 every chunk start (multiples of {CHUNK}): is the frame a byte copy of the frame before it?", ""]
    for run in args.runs:
        r = check(cfg, run, args.cam)
        out.append(r)
        copies = [x for x in r["rows"] if x["copy_cam"]]
        lines.append(f"{run}: {r['n_files']} files ({r['first']}-{r['last']}); {len(r['rows'])} chunk starts checked in "
                     f"imgFile_mask/cam{args.cam}; copies at {[x['frame'] for x in copies]}")
        for x in copies:
            n = sum(x["copies"].values())
            lines.append(f"  frame {x['frame']}: copy in {n}/8 sequences (imgFile and imgFile_mask, cameras 0-3)")
        jumps = [x["jump_s"] for x in r["rows"]]
        lines.append(f"  file-time jump at the chunk starts: {min(jumps):.0f} to {max(jumps):.0f} s")
        print("\n".join(lines[-len(copies) - 2:]), flush=True)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "frame_copies.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(os.path.join(HERE, "frame_copies.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
