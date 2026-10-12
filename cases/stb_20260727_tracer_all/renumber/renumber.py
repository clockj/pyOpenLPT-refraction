"""B2: renumber the 20260727 images on X: so that each file holds its own frame (plan:
pyOpenLPT/tests/cine_frame_order/2026-10-10-plan.md, Part B; your choice, 10 Oct).

pyOpenLPT's image preprocessing wrote file 256 as a copy of 255 and every later file one frame late; the video's last
frame (12,604) was never written (D3; fixed in pyOpenLPT, test F3: a reconversion equals the old file k + 1 at k).
For each run, folder (imgFile, imgFile_bb, imgFile_mask) and camera, this
  1. checks the folder: 12,605 files img000000-img012604, and 256 a byte copy of 255;
  2. moves img000256.tif to <run>\\removed_copy256\\<folder>\\camN\\;
  3. renames img000257 -> img000256, img000258 -> img000257, ... in increasing order (each name was just freed), one
     log line per rename (renumber/logs/<run>_<folder>_camN.log, on D:);
  4. checks: 0-12,603 present, new 256 = old 257 and new 12,603 = old 12,604 (MD5 taken before), new 256 != new 255;
  5. imgFile only: writes img012604.tif, the video's last frame, converted with the fixed reader and the run's cached
     background (imgFile\\background_cache.mat) after checking that those settings reproduce img000250.tif exactly.
     The masks' last frame needs the masking tool and is left missing.
An interrupted folder has one gap and resumes from it.  --check only reports each folder's state; --undo reverses a
folder from its log.  16 folders run at a time (a rename on X: takes 48 ms, 8.5 ms per file with 16 folders at once).

    python renumber/renumber.py RUN ... [--check] [--undo] [--threads 16] [--folders imgFile imgFile_bb imgFile_mask]
Progress and results: renumber/renumber_log.txt and renumber/summary_<time>.json.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402  (puts the pyOpenLPT repository on sys.path)
from modules.image_preprocessing import io as pio  # noqa: E402
from modules.image_preprocessing.core import apply_processing_pipeline_with_settings  # noqa: E402

RAW_ROOT = r"X:\Shijie Zhong\Bubble Shear Project\Raw_Data\20260727"
FOLDERS = ["imgFile", "imgFile_bb", "imgFile_mask"]
N_FILES = 12605          # files img000000-img012604 as the old conversion wrote them
COPY = 256               # the copy of 255
LAST = N_FILES - 1       # 12,604: the video's last frame, written anew in imgFile
CHECK_FRAME = 250        # an untouched old frame, to check the conversion settings
NAME = "img{:06d}.tif"
LOGS = os.path.join(HERE, "logs")
LOCK = threading.Lock()


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def say(msg):
    line = f"{now()} {msg}"
    with LOCK:
        print(line, flush=True)
        with open(os.path.join(HERE, "renumber_log.txt"), "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def present(d):
    out, other = set(), []
    with os.scandir(d) as it:
        for e in it:
            m = re.fullmatch(r"img(\d{6})\.tif", e.name)
            if m:
                out.add(int(m.group(1)))
            else:
                other.append(e.name)
    return out, other


def paths(cfg, run, folder, cam):
    d = os.path.join(case.case_dir(cfg, run), folder, f"cam{cam}")
    aside = os.path.join(case.case_dir(cfg, run), "removed_copy256", folder, f"cam{cam}")
    log = os.path.join(LOGS, f"{run}_{folder}_cam{cam}.log")
    return d, aside, log


def read_log(log):
    if not os.path.exists(log):
        return {}
    info = {}
    with open(log, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition(" ")
                info[k] = v
    return info


def state(cfg, run, folder, cam):
    """original / partial (gap at g) / renumbered / unexpected, without changing anything."""
    d, aside, log = paths(cfg, run, folder, cam)
    have, other = present(d)
    copy_aside = os.path.exists(os.path.join(aside, NAME.format(COPY)))
    full = set(range(N_FILES))
    if have == full and not copy_aside:
        return {"state": "original", "other": other}
    if copy_aside and have and max(have) <= LAST:
        missing = sorted(full - have)
        if missing == [LAST] or (folder == "imgFile" and not missing and read_log(log).get("RENAMED")):
            return {"state": "renumbered", "other": other, "last_written": LAST in have}
        if len(missing) == 1 and COPY <= missing[0] < LAST:
            return {"state": "partial", "gap": missing[0], "other": other}
    return {"state": "unexpected", "n_files": len(have), "copy_aside": copy_aside,
            "missing_first": sorted(full - have)[:5], "extra": sorted(have - full)[:5], "other": other}


def load_cache(cfg, run):
    d = loadmat(os.path.join(case.case_dir(cfg, run), "imgFile", "background_cache.mat"))
    cams = [int(c) for c in d["cam_ids"].ravel()]
    return {"shifts": {c: int(s) for c, s in zip(cams, d["cine_shifts"].ravel())},
            "invert": bool(int(d["invert_enabled"].ravel()[0])),
            "bg": {c: np.asarray(d[f"bg_cam{c}"], np.float32) for c in cams}}


def write_last_frame(cfg, run, cam, cache):
    """imgFile's img012604.tif from the video, after checking the settings on an untouched old frame."""
    d = os.path.join(case.case_dir(cfg, run), "imgFile", f"cam{cam}")
    cine = os.path.join(RAW_ROOT, run, f"CamC{cam + 1}.cine")
    settings = {"bg_enabled": True, "invert": cache["invert"], "cine_shifts": dict(cache["shifts"]),
                "low_in": 0, "high_in": 255, "denoise": False}
    header = pio._import_pycine()[1](cine)
    first_no = int(header["cinefileheader"].FirstImageNo)
    raw = pio.read_cine_frames(cine, first_no + CHECK_FRAME, 1, header=header)[0]
    new = apply_processing_pipeline_with_settings(raw, cache["bg"][cam], cam, settings)
    old = cv2.imread(os.path.join(d, NAME.format(CHECK_FRAME)), cv2.IMREAD_UNCHANGED)
    if old is None or not np.array_equal(new, old):
        raise RuntimeError(f"settings do not reproduce {NAME.format(CHECK_FRAME)}; last frame not written")
    raw = pio.read_cine_frames(cine, first_no + LAST, 1, header=header)[0]
    img = apply_processing_pipeline_with_settings(raw, cache["bg"][cam], cam, settings)
    out = os.path.join(d, NAME.format(LAST))
    if os.path.exists(out):
        raise RuntimeError(f"{out} exists")
    if not cv2.imwrite(out, img):
        raise RuntimeError(f"could not write {out}")
    back = cv2.imread(out, cv2.IMREAD_UNCHANGED)
    if back is None or not np.array_equal(back, img):
        raise RuntimeError(f"{out} reads back different")
    prev = cv2.imread(os.path.join(d, NAME.format(LAST - 1)), cv2.IMREAD_UNCHANGED)
    return {"written": out, "differs_from_12603": bool(not np.array_equal(prev, img))}


def renumber(cfg, run, folder, cam, caches):
    d, aside, log = paths(cfg, run, folder, cam)
    t0 = time.time()
    st = state(cfg, run, folder, cam)
    unit = f"{run} {folder} cam{cam}"
    if st["state"] == "unexpected":
        say(f"{unit}: UNEXPECTED state, not touched: {st}")
        return {"unit": unit, **st}
    if st["state"] == "original":
        a, b = os.path.join(d, NAME.format(COPY - 1)), os.path.join(d, NAME.format(COPY))
        if md5(a) != md5(b):
            say(f"{unit}: file 256 is not a copy of 255 here; not touched")
            return {"unit": unit, "state": "unexpected", "why": "256 is not a copy of 255"}
        sums = {"OLD257": md5(os.path.join(d, NAME.format(COPY + 1))), "OLD12604": md5(os.path.join(d, NAME.format(LAST)))}
        os.makedirs(aside, exist_ok=True)
        os.makedirs(LOGS, exist_ok=True)
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"#START {now()}\n#FOLDER {d}\n#ASIDE {aside}\n#OLD257 {sums['OLD257']}\n#OLD12604 {sums['OLD12604']}\n")
        os.rename(b, os.path.join(aside, NAME.format(COPY)))
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write(f"#MOVED_COPY {now()}\n")
        start = COPY + 1
    elif st["state"] == "partial":
        start = st["gap"] + 1
        say(f"{unit}: resuming at the gap {st['gap']}")
    else:
        start = N_FILES  # renumbered: nothing to rename
    if start < N_FILES:
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            for k in range(start, N_FILES):
                os.rename(os.path.join(d, NAME.format(k)), os.path.join(d, NAME.format(k - 1)))
                f.write(f"{k},{k - 1}\n")
                f.flush()  # every rename is in the log before the next, so --undo is exact after a kill
            f.write(f"#RENAMED {now()}\n")
    # checks
    info = read_log(log)
    have, _ = present(d)
    problems = []
    if not set(range(LAST)) <= have:
        problems.append(f"missing {sorted(set(range(LAST)) - have)[:5]}")
    if md5(os.path.join(d, NAME.format(COPY))) != info.get("OLD257"):
        problems.append("new 256 is not old 257")
    if md5(os.path.join(d, NAME.format(LAST - 1))) != info.get("OLD12604"):
        problems.append("new 12603 is not old 12604")
    if md5(os.path.join(d, NAME.format(COPY))) == md5(os.path.join(d, NAME.format(COPY - 1))):
        problems.append("new 256 still equals 255")
    out = {"unit": unit, "state": "renumbered" if not problems else "CHECK FAILED", "problems": problems,
           "renames": max(0, N_FILES - start), "s": round(time.time() - t0, 1)}
    if folder == "imgFile" and not problems and LAST not in have:
        try:
            out["last_frame"] = write_last_frame(cfg, run, cam, caches[run])
            with open(log, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"#LAST_WRITTEN {now()}\n")
        except Exception as e:  # noqa: BLE001 (reported; the renumbering itself stands)
            out["last_frame"] = f"NOT WRITTEN: {e}"
    say(f"{unit}: {out['state']} ({out['renames']} renames, {out['s']} s)"
        + (f" problems {problems}" if problems else "")
        + (f"; last frame: {out['last_frame']}" if "last_frame" in out else ""))
    return out


def renumber_safe(cfg, run, folder, cam, caches):
    """One folder; an error stops only that folder (it keeps one gap and resumes from it on the next run)."""
    try:
        return renumber(cfg, run, folder, cam, caches)
    except Exception as e:  # noqa: BLE001
        say(f"{run} {folder} cam{cam}: ERROR {type(e).__name__}: {e}")
        return {"unit": f"{run} {folder} cam{cam}", "state": "ERROR", "error": f"{type(e).__name__}: {e}"}


def undo(cfg, run, folder, cam):
    """Reverse a folder from its log: renames back in decreasing order, the copy back to 256."""
    d, aside, log = paths(cfg, run, folder, cam)
    unit = f"{run} {folder} cam{cam}"
    pairs = []
    with open(log, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                k, j = map(int, line.split(","))
                pairs.append((k, j))
    last = os.path.join(d, NAME.format(LAST))
    if folder == "imgFile" and "LAST_WRITTEN" in read_log(log) and os.path.exists(last):
        os.rename(last, os.path.join(aside, "img012604_written_by_renumber.tif"))
    for k, j in reversed(pairs):
        os.rename(os.path.join(d, NAME.format(j)), os.path.join(d, NAME.format(k)))
    os.rename(os.path.join(aside, NAME.format(COPY)), os.path.join(d, NAME.format(COPY)))
    os.rename(log, log + f".undone_{datetime.datetime.now():%Y%m%d-%H%M%S}")
    say(f"{unit}: undone ({len(pairs)} renames back)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--check", action="store_true", help="only report each folder's state")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--folders", nargs="+", default=FOLDERS)
    args = ap.parse_args()
    cfg = case.load_config()
    units = [(r, f, c) for r in args.runs for f in args.folders for c in range(4)]
    if args.check:
        with ThreadPoolExecutor(args.threads) as ex:
            res = list(ex.map(lambda u: (u, state(cfg, *u)), units))
        for (r, f, c), st in res:
            print(f"{r:11s} {f:13s} cam{c}: {st}")
        return
    if args.undo:
        for u in units:
            undo(cfg, *u)
        return
    caches = {r: load_cache(cfg, r) for r in args.runs if "imgFile" in args.folders}
    say(f"renumbering starts: {len(units)} folders of {args.runs}, {args.threads} at a time")
    t0 = time.time()
    with ThreadPoolExecutor(args.threads) as ex:
        res = list(ex.map(lambda u: renumber_safe(cfg, *u, caches), units))
    bad = [r for r in res if r.get("state") != "renumbered" or str(r.get("last_frame", "")).startswith("NOT")]
    say(f"renumbering ends: {len(res) - len(bad)} of {len(res)} folders done in {(time.time() - t0) / 60:.1f} min"
        + (f"; NEEDS A LOOK: {[r['unit'] for r in bad]}" if bad else ""))
    with open(os.path.join(HERE, f"summary_{datetime.datetime.now():%Y%m%d-%H%M%S}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
