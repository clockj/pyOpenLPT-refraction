"""Copy a run's masked images to its staging folder on D: before the batch tracks it, so STB reads them locally
(reading them from X: costs about 1 s per frame, test R1).

    python prefetch_images.py RUN [--switch] [--unswitch] [--remove]

  - copy (default): robocopy imgFile_mask\\cam0..3 to <stage>\\<run>\\images\\cam0..3 (/MT:16; a rerun copies only what
    is missing), then check that every image of the image list is there with its size on X:;
  - --switch: after a complete check, point the staging image lists (imgLists\\cam<k>ImageNames.txt) at the copies;
    the lists naming X: are kept as imgLists_X\\.  The images are byte copies, so the results are the same;
  - --unswitch: put the X: lists back;
  - --remove: delete the copies (only after the run is delivered; the lists must name X: again first).
Logs to <stage>\\<run>\\prefetch.log.
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess

import case


def say(path, msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def x_sizes(cfg, run, cam):
    folder = os.path.join(case.case_dir(cfg, run), cfg["image_folder"], f"cam{cam}")
    with os.scandir(folder) as it:
        return {e.name: e.stat().st_size for e in it if e.is_file()}


def list_names(sd, cam, sub="imgLists"):
    with open(os.path.join(sd, sub, f"cam{cam}ImageNames.txt"), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def check(cfg, run, sd):
    """Every image the X: lists name is copied with its X: size; returns (ok, message)."""
    bad = []
    n = 0
    for cam in range(4):
        sizes = x_sizes(cfg, run, cam)
        sub = "imgLists_X" if os.path.exists(os.path.join(sd, "imgLists_X")) else "imgLists"
        for p in list_names(sd, cam, sub):
            name = os.path.basename(p)
            q = os.path.join(sd, "images", f"cam{cam}", name)
            n += 1
            if not os.path.exists(q) or os.path.getsize(q) != sizes.get(name, -1):
                bad.append(q)
    return not bad, f"{n} images checked, {len(bad)} missing or of another size {bad[:3]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--switch", action="store_true")
    ap.add_argument("--unswitch", action="store_true")
    ap.add_argument("--remove", action="store_true")
    args = ap.parse_args()
    cfg = case.load_config()
    sd = case.stage_dir(cfg, args.run)
    log = os.path.join(sd, "prefetch.log")
    lists, lists_x = os.path.join(sd, "imgLists"), os.path.join(sd, "imgLists_X")
    if args.unswitch or args.remove:
        if os.path.exists(lists_x):
            shutil.rmtree(lists)
            os.replace(lists_x, lists)
            say(log, "image lists name X: again")
        if args.remove:
            assert not os.path.exists(lists_x)
            shutil.rmtree(os.path.join(sd, "images"), ignore_errors=True)
            say(log, "local image copies removed")
        return
    if not args.switch:
        for cam in range(4):
            src = os.path.join(case.case_dir(cfg, args.run), cfg["image_folder"], f"cam{cam}")
            dst = os.path.join(sd, "images", f"cam{cam}")
            say(log, f"copy {src} -> {dst}")
            rc = subprocess.call(["robocopy", src, dst, "*.tif", "/MT:16", "/R:5", "/W:30", "/NP", "/NFL",
                                  "/NDL", f"/LOG+:{os.path.join(sd, 'prefetch_robocopy.log')}"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            say(log, f"cam{cam}: robocopy exit {rc}")
    ok, msg = check(cfg, args.run, sd)
    say(log, f"check: {'OK' if ok else 'INCOMPLETE'}: {msg}")
    if args.switch:
        if not ok:
            raise SystemExit("not switched: the copies are incomplete")
        if os.path.exists(lists_x):
            say(log, "already switched")
            return
        shutil.copytree(lists, lists_x)
        for cam in range(4):
            names = [os.path.basename(p) for p in list_names(sd, cam, "imgLists_X")]
            text = "".join(os.path.join(sd, "images", f"cam{cam}", nm) + "\n" for nm in names)
            tmp = os.path.join(lists, f"cam{cam}ImageNames.txt.tmp")
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            os.replace(tmp, os.path.join(lists, f"cam{cam}ImageNames.txt"))
        say(log, "image lists now name the local copies (X: lists kept in imgLists_X)")


if __name__ == "__main__":
    main()
