"""Inventory of the runs to track (I1): images, existing image lists and outputs, and a look at two frames per camera.

For every run in config.json and camera 0-3:
  - lists <run>\\imgFile_mask\\cam<k>: names img000000.tif ... (contiguous from 0?), other files, total size;
  - the same number of frames in all four cameras;
  - an existing image list (imgFile_mask\\cam<k>ImageNames.txt): does it name exactly these files, in order?
  - frames 0 and 6000: dtype, size, mean, 99th percentile, maximum, the fraction of pixels above the tracer finder's
    threshold and the number of 3x3 local maxima above it (a rough particle count): particles must be bright on black
    (OpenLPT's finder) and look alike in every run;
  - what the run folder already holds (config.json, config.txt, Results_tracer).
Read only.  Writes tests/inventory.json and tests/inventory_out.txt.

    python tests/inventory.py
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import case  # noqa: E402

NAME = re.compile(r"^img(\d{6})\.tif$")
LOOK_FRAMES = (0, 6000)


def list_camera(cfg, run, cam):
    folder = os.path.join(case.case_dir(cfg, run), cfg["image_folder"], f"cam{cam}")
    ids, other, size = [], [], 0
    with os.scandir(folder) as it:
        for e in it:
            m = NAME.match(e.name)
            if m and e.is_file():
                ids.append(int(m.group(1)))
                size += e.stat().st_size
            else:
                other.append(e.name)
    ids.sort()
    n = len(ids)
    return {"run": run, "cam": cam, "n": n, "contiguous": ids == list(range(n)), "first": ids[0] if ids else None,
            "last": ids[-1] if ids else None, "other": sorted(other)[:10], "n_other": len(other),
            "GB": size / 1e9}


def check_list(cfg, run, cam, n):
    path = os.path.join(case.case_dir(cfg, run), cfg["image_folder"], f"cam{cam}ImageNames.txt")
    if not os.path.exists(path):
        return {"exists": False}
    with open(path, encoding="utf-8") as f:
        entries = [ln.strip() for ln in f if ln.strip()]
    want = [f"{cfg['image_folder']}/cam{cam}/" + cfg["image_name"].format(i) for i in range(n)]
    return {"exists": True, "n": len(entries), "matches_files": entries == want, "first": entries[:1],
            "last": entries[-1:]}


def look(cfg, run, cam, frame, threshold):
    img = cv2.imread(case.image_path(cfg, run, cam, frame), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"frame": frame, "error": "unreadable"}
    a = img.astype(np.float64)
    peaks = (img >= cv2.dilate(img, np.ones((3, 3), np.uint8))) & (img > threshold)
    return {"frame": frame, "dtype": str(img.dtype), "shape": list(img.shape), "mean": float(a.mean()),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()),
            "frac_above": float((img > threshold).mean()), "peaks": int(peaks.sum())}


def main():
    cfg = case.load_config()
    runs = cfg["cases"]
    threshold = 20  # the tracer finder's threshold in tracerConfig.txt
    jobs = [(r, c) for r in runs for c in range(4)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        cams = list(ex.map(lambda rc: list_camera(cfg, *rc), jobs))
        looks = list(ex.map(lambda rcf: look(cfg, *rcf, threshold),
                            [(r, c, f) for r in runs for c in range(4) for f in LOOK_FRAMES]))
    out = {"threshold": threshold, "runs": {}}
    lines = ["I1 inventory of the runs (read only)", ""]
    li = iter(looks)
    ok_all = True
    for r in runs:
        rc = [x for x in cams if x["run"] == r]
        ns = {x["n"] for x in rc}
        n = min(ns)
        lists = [check_list(cfg, r, c, n) for c in range(4)]
        lk = [[next(li) for _ in LOOK_FRAMES] for _ in range(4)]
        d = case.case_dir(cfg, r)
        holds = {k: os.path.exists(os.path.join(d, k)) for k in ("config.json", "config.txt", "tracerConfig.txt",
                                                                 cfg["output_folder"])}
        ok = len(ns) == 1 and all(x["contiguous"] and x["n_other"] == 0 for x in rc) and \
            all((not L["exists"]) or L["matches_files"] for L in lists)
        ok_all &= ok
        out["runs"][r] = {"cameras": rc, "lists": lists, "look": lk, "holds": holds, "frames": n, "ok": ok}
        lines.append(f"{r}: {'OK' if ok else 'PROBLEM'}  frames {sorted(ns)}  "
                     f"{sum(x['GB'] for x in rc):.0f} GB of images  lists: "
                     f"{'existing, ' + ('match' if all(L.get('matches_files') for L in lists) else 'DIFFER') if all(L['exists'] for L in lists) else ('none' if not any(L['exists'] for L in lists) else 'SOME')}"
                     f"  holds: {', '.join(k for k, v in holds.items() if v) or '-'}")
        for c in range(4):
            x = rc[c]
            extra = f"  other files: {x['n_other']} {x['other'][:3]}" if x["n_other"] else ""
            row = "  ".join(f"f{v['frame']}: mean {v['mean']:5.2f} p99 {v['p99']:5.0f} max {v['max']:4.0f} "
                            f">{threshold} {100 * v['frac_above']:5.2f}% peaks {v['peaks']:5d}"
                            if "error" not in v else f"f{v['frame']}: {v['error']}" for v in lk[c])
            dt = lk[c][0].get("dtype", "?")
            shp = "x".join(map(str, lk[c][0].get("shape", [])))
            lines.append(f"   cam{c}: {x['n']:6d} files {x['first']}-{x['last']} contiguous {x['contiguous']}  "
                         f"{dt} {shp}  {row}{extra}")
    lines += ["", "ALL OK" if ok_all else "PROBLEMS FOUND"]
    out["ok"] = ok_all
    case.save_json(os.path.join(case.TESTS, "inventory.json"), out)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(case.TESTS, "inventory_out.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
