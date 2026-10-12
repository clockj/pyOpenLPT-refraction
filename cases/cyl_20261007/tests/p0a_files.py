"""P0a (plan: Phase 0): the files of run 20261007, read-only.

    python tests/p0a_files.py

For T0, the Planes and T1, per camera folder: the number of frames, the first and last frame names, the files'
modification times (first and last frame), the image size and type of frame 0, and the .chd header if there is one
(camera serial, frame rate, exposure: the uint16 pair 1280, 800, the serial 6 bytes after it, frame rate and
exposure 31 bytes after it, as read for the 6 Oct runs).  Writes p0a_files_out.txt and p0a_files.json beside itself.
"""
import datetime
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402

import numpy as np  # noqa: E402


def header(path):
    with open(path, "rb") as fh:
        b = fh.read()
    i = b.find(struct.pack("<HH", 1280, 800))
    if i < 0:
        return None
    serial, = struct.unpack_from("<I", b, i + 6)
    fps, shutter = struct.unpack_from("<II", b, i + 31)
    return {"serial": serial, "fps": fps, "exposure_us": shutter, "bytes": len(b)}


def stamp(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def survey_folder(folder, log):
    names = sorted(e.name for e in os.scandir(folder))
    tifs = [n for n in names if n.lower().endswith(".tif")]
    others = [n for n in names if not n.lower().endswith(".tif")]
    out = {"folder": folder, "frames": len(tifs), "first": tifs[0] if tifs else None, "last": tifs[-1] if tifs else None,
           "other_files": others}
    if tifs:
        out["time_first"] = stamp(os.path.join(folder, tifs[0]))
        out["time_last"] = stamp(os.path.join(folder, tifs[-1]))
        img = case.read_image(os.path.join(folder, tifs[0]))
        out["size"] = [int(img.shape[1]), int(img.shape[0])]
        out["dtype"] = str(img.dtype)
        out["median_grey_frame0"] = float(np.median(img))
    hdrs = [n for n in others if n.lower().endswith(".chd")]
    out["header"] = header(os.path.join(folder, hdrs[0])) if hdrs else None
    h = out["header"]
    log(f"  {os.path.relpath(folder, CFG['raw_root'])}: {out['frames']} frames ({out['first']} ... {out['last']}), "
        f"written {out.get('time_first')} ... {out.get('time_last')}, {out.get('size')} {out.get('dtype')}, "
        f"median grey (frame 0) {out.get('median_grey_frame0')}; "
        + (f"header: serial {h['serial']}, {h['fps']} frames/s, exposure {h['exposure_us']} us" if h else "no .chd header")
        + (f"; other files: {others}" if others and not hdrs else ""))
    return out


def main():
    log = case.Log(os.path.join(HERE, "p0a_files"))
    log(f"P0a: the files of {CFG['raw_root']} (read-only)")
    out = {}
    for run in ("T0", "Planes", "T1"):
        log(f"\n{run}")
        root = os.path.join(CFG["raw_root"], run)
        subs = sorted(e.name for e in os.scandir(root) if e.is_dir())
        extra = sorted(e.name for e in os.scandir(root) if e.is_file())
        if extra:
            log(f"  files in {run}: {extra}")
            for n in extra:
                if n.lower().endswith(".txt"):
                    with open(os.path.join(root, n), encoding="utf-8", errors="replace") as fh:
                        log(f"    {n}: {fh.read().strip()!r}")
        if run == "Planes":
            for p in subs:
                for cam in sorted(e.name for e in os.scandir(os.path.join(root, p)) if e.is_dir()):
                    out[f"Planes/{p}/{cam}"] = survey_folder(os.path.join(root, p, cam), log)
        else:
            for cam in subs:
                out[f"{run}/{cam}"] = survey_folder(os.path.join(root, cam), log)
    log.save(out)


CFG = case.load_config()

if __name__ == "__main__":
    main()
