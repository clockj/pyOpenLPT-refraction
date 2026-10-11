"""D3: which recorded frame of the raw video is in each imgFile file (read only).

imgFile was made from the raw videos (Raw_Data\\20260727\\<run>\\CamC1..4.cine) by pyOpenLPT's image preprocessing
(GUI batch: modules/image_preprocessing/view.py).  That code lists each video's frames by the camera's own numbering,
FirstImageNo ... FirstImageNo + ImageCount - 1, and reads them in chunks of 256 with pycine's
read_frames(cine, start_frame=<that number>, count=...).  pycine's start_frame is a 1-based position in the file
(frame_reader reads pImage[start_frame - 1]), and pycine takes start_frame = 0 as "not given", i.e. 1.  If the
numbering starts at 0, the first chunk (0 -> 1) reads stored frames 0-255 correctly, and every later chunk starts one
stored frame early: file 256 repeats stored frame 255, file j holds stored frame j - 1 from 256 on, and the last stored
frame is never written.

This check:
  1. reads each video's header (FirstImageNo, ImageCount, image format, frame rate; pycine.file.read_header, which
     only reads);
  2. shows pycine's quirk: read_frames(start_frame=0) and (start_frame=1) return the same stored frame;
  3. reads stored frames by position (start_frame = position + 1) and correlates each with the imgFile files around
     it; each file's best-matching stored frame (the largest |correlation|: imgFile is the raw frame inverted, so
     particles dark in the video are bright on black, and the match is negative) is the one it was made from;
  4. compares that with what the GUI's chunked reading predicts.

    python tests/cine_frames.py RUN [--cams 0 1]
Writes tests/cine_frames_<RUN>.txt and .json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
from pycine.file import read_header
from pycine.raw import read_frames

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402

RAW_ROOT = r"X:\Shijie Zhong\Bubble Shear Project\Raw_Data\20260727"
CHUNK = 256  # view.py: chunk_size = 256
# files checked, each against the stored frames from 2 before to 2 after it
FILES = [0, 1, 2, 254, 255, 256, 257, 258, 511, 512, 513]


def cine_path(run, cam):
    return os.path.join(RAW_ROOT, run, f"CamC{cam + 1}.cine")


def stored(path, pos):
    """Stored frame at 0-based position pos (pycine's start_frame is 1-based)."""
    imgs = list(read_frames(path, start_frame=pos + 1, count=1)[0])
    return np.asarray(imgs[0])


def predicted(j, first_no, chunk=CHUNK):
    """Stored position the GUI's chunked reading puts in output file j (frames listed from first_no)."""
    c = j // chunk
    s = first_no + c * chunk          # the chunk's start_frame handed to pycine
    head = s if s != 0 else 1         # pycine: start_frame == 0 counts as not given -> 1
    return head - 1 + (j - c * chunk)  # frame_reader reads pImage[head - 1 + i]


def corr(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def digest(arr):
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def check(cfg, run, cam, files):
    path = cine_path(run, cam)
    h = read_header(path)
    cfh, bih, setup = h["cinefileheader"], h["bitmapinfoheader"], h["setup"]
    n = int(cfh.ImageCount)
    info = {"run": run, "cam": cam, "cine": path, "FirstImageNo": int(cfh.FirstImageNo), "ImageCount": n,
            "TotalImageCount": int(cfh.TotalImageCount), "FirstMovieImage": int(cfh.FirstMovieImage),
            "width": int(bih.biWidth), "height": int(bih.biHeight), "biCompression": int(bih.biCompression),
            "RealBPP": int(setup.RealBPP), "FrameRate": int(setup.FrameRate)}
    # pycine's quirk
    a0 = np.asarray(list(read_frames(path, start_frame=0, count=1)[0])[0])
    a1 = stored(path, 0)
    a2 = stored(path, 1)
    info["quirk_start0_equals_start1"] = digest(a0) == digest(a1)
    info["quirk_start1_differs_from_start2"] = digest(a1) != digest(a2)
    last_file = n - 1
    files = sorted(set(files) | {last_file - 1, last_file})
    cache = {}

    def raw(pos):
        if pos not in cache:
            cache[pos] = stored(path, pos)
        return cache[pos]

    rows = []
    for j in files:
        img = cv2.imread(case.image_path(cfg, run, cam, j).replace(cfg["image_folder"], "imgFile"), cv2.IMREAD_UNCHANGED)
        cands = [k for k in range(j - 2, j + 3) if 0 <= k < n]
        cs = {k: corr(img, raw(k)) for k in cands}
        best = max(cs, key=lambda k: abs(cs[k]))  # imgFile is the raw frame inverted, so the match is negative
        rows.append({"file": j, "best_stored": best, "corr": cs, "predicted": predicted(j, info["FirstImageNo"])})
    info["rows"] = rows
    # is the last stored frame in any file?  It would have to be the last file
    img_last = cv2.imread(case.image_path(cfg, run, cam, last_file).replace(cfg["image_folder"], "imgFile"),
                          cv2.IMREAD_UNCHANGED)
    info["last_stored_vs_last_file"] = corr(img_last, raw(n - 1))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cams", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    cfg = case.load_config()
    lines = [f"D3 {args.run}: which stored frame of the raw video is in each imgFile file (best: the largest "
             "|correlation|; imgFile is the raw frame inverted, so the match is negative; predicted: the GUI's "
             "chunked reading)", ""]
    out = []
    for cam in args.cams:
        r = check(cfg, args.run, cam, FILES)
        out.append(r)
        lines += [f"camera {cam}: {r['cine']}",
                  f"  header: FirstImageNo {r['FirstImageNo']}, ImageCount {r['ImageCount']}, TotalImageCount "
                  f"{r['TotalImageCount']}, FirstMovieImage {r['FirstMovieImage']}, {r['width']} x {r['height']}, "
                  f"biCompression {r['biCompression']}, RealBPP {r['RealBPP']}, {r['FrameRate']} fps",
                  f"  pycine: start_frame=0 gives the same frame as start_frame=1: {r['quirk_start0_equals_start1']}; "
                  f"start_frame=1 and 2 differ: {r['quirk_start1_differs_from_start2']}",
                  f"  {'file':>6s} {'best stored':>12s} {'predicted':>10s}   correlation with stored frames file-2 .. file+2"]
        for x in r["rows"]:
            cs = "  ".join(f"{k}:{v:.3f}" for k, v in x["corr"].items())
            ok = "ok" if x["best_stored"] == x["predicted"] else "DIFFERS"
            lines.append(f"  {x['file']:6d} {x['best_stored']:12d} {x['predicted']:10d} {ok:>7s}   {cs}")
        lines.append(f"  last stored frame ({r['ImageCount'] - 1}) against the last file: "
                     f"{r['last_stored_vs_last_file']:.3f}")
        lines.append("")
        print("\n".join(lines[-(len(r["rows"]) + 6):]), flush=True)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, f"cine_frames_{args.run}.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(os.path.join(HERE, f"cine_frames_{args.run}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
