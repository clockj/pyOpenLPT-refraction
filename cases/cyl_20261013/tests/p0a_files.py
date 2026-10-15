"""P0a (Phase 0): the files of run 20261013's wand recording T0, read-only (cases/cyl_20261008's p0a, T0 only).

    python tests/p0a_files.py

For T0, per camera folder: the number of frames, the first and last names, the files' modification times,
the image size and type of frame 0 and its median grey, and the .chd header (the Cine file header: FirstMovieImage,
TotalImageCount, FirstImageNo, ImageCount; the bitmap header's size and bit depth; the camera serial, frame rate and
exposure read after the (width, height) pair, as for the 6-7 Oct runs).  FirstImageNo says which image (counted from
the common trigger) a camera's frame000000.tif holds: config.cameras[].first_image must match it.
Writes p0a_files_out.txt and p0a_files.json beside itself.
"""
import datetime
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CASE_DIR)
import case  # noqa: E402

import numpy as np  # noqa: E402


def header(path, size):
    with open(path, "rb") as fh:
        b = fh.read()
    out = {"bytes": len(b), "type": b[0:2].decode("latin-1")}
    (out["first_movie_image"], out["total_image_count"], out["first_image_no"], out["image_count"], off_ih, _os,
     _oo) = struct.unpack_from("<iIiIIII", b, 8)
    bi = struct.unpack_from("<IiiHHIIiiII", b, off_ih)
    out["bitmap_size"] = [int(bi[1]), int(bi[2])]
    out["bits"] = int(bi[4])
    i = b.find(struct.pack("<HH", *size))
    if i >= 0:
        out["serial"], = struct.unpack_from("<I", b, i + 6)
        out["fps"], out["exposure_us"] = struct.unpack_from("<II", b, i + 31)
    return out


def stamp(path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")


def survey_folder(folder, cam, run, log):
    names = sorted(e.name for e in os.scandir(folder))
    tifs = [n for n in names if n.lower().endswith(".tif")]
    others = [n for n in names if not n.lower().endswith(".tif")]
    out = {"folder": folder, "frames": len(tifs), "first": tifs[0] if tifs else None, "last": tifs[-1] if tifs else None,
           "other_files": others}
    W, H = case.size(CFG, cam)
    if tifs:
        out["time_first"] = stamp(os.path.join(folder, tifs[0]))
        out["time_last"] = stamp(os.path.join(folder, tifs[-1]))
        img = case.read_image(os.path.join(folder, tifs[0]))
        out["size"] = [int(img.shape[1]), int(img.shape[0])]
        out["dtype"] = str(img.dtype)
        out["median_grey_frame0"] = float(np.median(img))
    hdrs = [n for n in others if n.lower().endswith(".chd")]
    out["header"] = header(os.path.join(folder, hdrs[0]), (W, H)) if hdrs else None
    h = out["header"]
    fi = case.first_image(CFG, cam, run)
    out["config_first_image"] = fi
    out["config_size_ok"] = out.get("size") == [W, H]
    out["config_first_image_ok"] = bool(h and h["first_image_no"] == fi and h["image_count"] == len(tifs))
    log(f"  {run}\\cam{cam}: {out['frames']} frames ({out['first']} ... {out['last']}), written {out.get('time_first')}"
        f" ... {out.get('time_last')}, {out.get('size')} {out.get('dtype')}, median grey (frame 0) "
        f"{out.get('median_grey_frame0')}")
    if h:
        log(f"      header: serial {h.get('serial')}, {h.get('fps')} frames/s, exposure {h.get('exposure_us')} us, "
            f"bitmap {h['bitmap_size']} x {h['bits']} bit; FirstMovieImage {h['first_movie_image']}, TotalImageCount "
            f"{h['total_image_count']}, FirstImageNo {h['first_image_no']}, ImageCount {h['image_count']}")
    log(f"      config: size {'matches' if out['config_size_ok'] else 'DIFFERS'}, first_image {fi} "
        f"{'matches the header' if out['config_first_image_ok'] else 'DIFFERS from the header'}; image numbers "
        f"{fi} ... {fi + len(tifs) - 1}")
    return out


def main():
    log = case.Log(os.path.join(HERE, "p0a_files"))
    log(f"P0a: the files of {CFG['raw_root']} (read-only)")
    out = {}
    for run in ("T0",):
        log(f"\n{run}")
        root = os.path.join(CFG["raw_root"], run)
        extra = sorted(e.name for e in os.scandir(root) if e.is_file())
        if extra:
            log(f"  files in {run}: {extra}")
        for c in CFG["cameras"]:
            out[f"{run}/{c['folder']}"] = survey_folder(os.path.join(root, c["folder"]), int(c["id"]), run, log)
    t0 = [out[f"T0/{c['folder']}"] for c in CFG["cameras"]]
    lo = max(o["config_first_image"] for o in t0)
    hi = min(o["config_first_image"] + o["frames"] - 1 for o in t0)
    log(f"\nT0: image numbers saved by every camera: {lo} ... {hi} ({hi - lo + 1} frames); by cams 1-2 only before {lo}")
    out["T0_common_images"] = [lo, hi]
    log.save(out)


CFG = case.load_config()

if __name__ == "__main__":
    main()
