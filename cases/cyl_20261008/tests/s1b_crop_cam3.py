"""S1b (plan: 2026-10-08-plan.md, Phase T; added while building, section 6): cam3's frames cropped to the rows that see
the volume cams 1 and 2 see, for speed.

    python tests/s1b_crop_cam3.py [--rows 360 1576]

Why: the line detector's time grows with the image's area (it filters every pixel): on S3's first frame it took
15-31 s per IPR loop, most of it cam3's 2048 x 1952 frame (27 s alone, against 9 s for a 1280 x 800 one), and STB
detects once per loop and frame.  The points all three cameras see project into cam3's rows 366-1561 (99.6 % of
them within 382-1551), every column; rows 360-1575 keep them all.
What it writes (git-ignored):
  - data/T1pre_c3crop/: cam1 and cam2's image lists as S1's (the same files), and cam3's frames cropped to the rows
    [r0, r1) with its own list; S1's manifest with the crop noted;
  - data/cams_T0_c3crop/: cams 1-2's tracking camFiles unchanged, cam3's with the image size (r1 - r0) x 2048 and
    the principal point's row moved by -r0 (every other byte unchanged).  The pixels then map to the same rays.
Check: 2,000 points of the view volume, projected by the full and the cropped cam3: the same pixel minus r0 in v (to
1e-9 px) wherever both give one, and the cropped one inside its rows.
Writes s1b_crop_cam3_out.txt beside itself.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case  # noqa: E402
import t1_setup as ts  # noqa: E402


def crop_camfile(src, dst, r0, r1):
    with open(src, "rb") as fh:
        raw = fh.read()
    eol = b"\r\n" if b"\r\n" in raw else b"\n"
    lines = raw.split(eol)
    k = [i for i, ln in enumerate(lines) if ln.strip().startswith(b"# Image Size")][0] + 1
    n_row, n_col = (int(v) for v in lines[k].split(b","))
    lines[k] = f"{r1 - r0},{n_col}".encode("ascii")
    km = [i for i, ln in enumerate(lines) if ln.strip().startswith(b"# Camera Matrix")][0] + 2   # the K row of cy
    vals = [float(v) for v in lines[km].split(b",")]
    vals[2] -= r0
    lines[km] = ",".join(f"{v:.16e}" for v in vals).encode("ascii")
    with open(dst, "wb") as fh:
        fh.write(eol.join(lines))
    return n_row, n_col, vals[2] + r0, vals[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs=2, default=[360, 1576])
    args = ap.parse_args()
    r0, r1 = args.rows
    log = case.Log(os.path.join(HERE, "s1b_crop_cam3"))
    src_pre = os.path.join(ts.DATA, "T1pre")
    dst_pre = os.path.join(ts.DATA, "T1pre_c3crop")
    os.makedirs(os.path.join(dst_pre, "cam3"), exist_ok=True)
    for c in (1, 2):
        with open(os.path.join(src_pre, f"cam{c}ImageNames.txt"), encoding="utf-8") as fh:
            text = fh.read()
        with open(os.path.join(dst_pre, f"cam{c}ImageNames.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
    with open(os.path.join(src_pre, "cam3ImageNames.txt"), encoding="utf-8") as fh:
        names = [ln for ln in fh.read().splitlines() if ln.strip()]
    out = []
    for p in names:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        q = os.path.join(dst_pre, "cam3", os.path.basename(p))
        cv2.imwrite(q, np.ascontiguousarray(img[r0:r1]))
        out.append(q)
    with open(os.path.join(dst_pre, "cam3ImageNames.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    with open(os.path.join(src_pre, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    man["cam3_crop_rows"] = [r0, r1]
    man["method"] += f"; cam3 cropped to rows {r0}-{r1 - 1} (s1b_crop_cam3.py)"
    with open(os.path.join(dst_pre, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    log(f"S1b: cam3's {len(out)} frames cropped to rows {r0}-{r1 - 1} ({r1 - r0} x 2048) into {dst_pre}")

    src_cam = os.path.join(ts.DATA, "cams_T0")
    dst_cam = os.path.join(ts.DATA, "cams_T0_c3crop")
    os.makedirs(dst_cam, exist_ok=True)
    for c in (1, 2):
        with open(os.path.join(src_cam, f"cam{c}.txt"), "rb") as fh:
            b = fh.read()
        with open(os.path.join(dst_cam, f"cam{c}.txt"), "wb") as fh:
            fh.write(b)
    n_row, n_col, cy0, cy1 = crop_camfile(os.path.join(src_cam, "cam3.txt"), os.path.join(dst_cam, "cam3.txt"), r0, r1)
    log(f"  cam3.txt: image {n_row} x {n_col} -> {r1 - r0} x {n_col}; principal point row {cy0} -> {cy1}")

    lpt = ts.lib()
    full = lpt.loadCameraModel(os.path.join(src_cam, "cam3.txt"))
    crop = lpt.loadCameraModel(os.path.join(dst_cam, "cam3.txt"))
    vol = ts.view_volume(lpt)
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.uniform(*vol["x"], 2000), rng.uniform(*vol["y"], 2000), rng.uniform(*vol["z"], 2000)])
    X = X[np.hypot(X[:, 0], X[:, 2]) < 97.0]
    P = [lpt.Pt3D(*map(float, x)) for x in X]
    a, b = full.projectBatchStatus(P), crop.projectBatchStatus(P)
    both = [(p, q) for p, q in zip(a, b) if p[0] and q[0]]
    diff = max(max(abs(p[1][0] - q[1][0]), abs(p[1][1] - r0 - q[1][1])) for p, q in both)
    inside = sum(1 for p, q in both if r0 <= p[1][1] < r1)
    log(f"  check: {len(both)} of {len(X)} points projected by both; largest |pixel(full) - (pixel(crop) + r0)| "
        f"{diff:.2e} px; {inside} of them on the cropped rows")
    log(f"  {'PASS' if diff < 1e-9 and crop.getNRow() == r1 - r0 else 'FAIL'}: the cropped camera maps every point to the "
        f"full camera's pixel shifted by {r0} rows; reports {crop.getNRow()} x {crop.getNCol()}")
    log.save({"rows": [r0, r1], "max_diff_px": diff, "points": len(both)})


if __name__ == "__main__":
    main()
