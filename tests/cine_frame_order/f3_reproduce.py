"""F3: the fixed conversion reproduces the old 20260727 files, one frame later from 256 on (read only; plan:
2026-10-10-plan.md, Part A).

The old imgFile was made by the GUI batch with each run's cached background (imgFile\\background_cache.mat:
backgrounds, bit shifts, invert) and a grey-level range and denoise setting that were not saved.  Those two are
recovered from an old frame that is right (frame 250: files 0-255 hold the right stored frames), by trying the
defaults (0-255, denoise off, then on) and else the range that the joint histogram of old and recomputed values
points to.  With them, stored frames are converted with the processing function the GUI and the command line share
(core.apply_processing_pipeline_with_settings) and compared pixel for pixel with the old files:
  - files 250-255: new file i = old file i;
  - files 256-262 and 12,603: new file i = old file i + 1 (old file 256 is the copy of 255);
  - file 12,604, the video's last frame, has no old file: it must differ from file 12,603.
If they all hold, renumbering the old files (B2) gives the same pixels as reconverting.

    python tests/cine_frame_order/f3_reproduce.py [RUN:CAM ...]   (default 35Hz_r_b_1:0 35Hz_r_b_1:1 10Hz_r_b_1:0)
Writes f3_reproduce.txt, f3_reproduce.json and f3_settings.json (the recovered settings per run and camera) here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from modules.image_preprocessing import io as pio  # noqa: E402
from modules.image_preprocessing.core import apply_processing_pipeline_with_settings  # noqa: E402

RAW_ROOT = Path(r"X:\Shijie Zhong\Bubble Shear Project\Raw_Data\20260727")
PROC_ROOT = Path(r"X:\Shijie Zhong\Bubble Shear Project\Processed\20260727\Results")
DEFAULT = ["35Hz_r_b_1:0", "35Hz_r_b_1:1", "10Hz_r_b_1:0"]
INFER_FRAME = 250
FRAMES = list(range(250, 263)) + [12603, 12604]


def old_file(run: str, cam: int, i: int) -> np.ndarray | None:
    return cv2.imread(str(PROC_ROOT / run / "imgFile" / f"cam{cam}" / f"img{i:06d}.tif"), cv2.IMREAD_UNCHANGED)


def load_cache(run: str) -> dict:
    d = loadmat(str(PROC_ROOT / run / "imgFile" / "background_cache.mat"))
    cams = [int(c) for c in d["cam_ids"].ravel()]
    shifts = {c: int(s) for c, s in zip(cams, d["cine_shifts"].ravel())}
    return {"cams": cams, "shifts": shifts, "invert": bool(int(d["invert_enabled"].ravel()[0])),
            "bg": {c: np.asarray(d[f"bg_cam{c}"], np.float32) for c in cams},
            "skip_frames": int(d["skip_frames"].ravel()[0]), "avg_count": int(d["avg_count"].ravel()[0])}


def settings_for(cache: dict, low: int, high: int, denoise: bool) -> dict:
    return {"bg_enabled": True, "invert": cache["invert"], "cine_shifts": dict(cache["shifts"]),
            "low_in": low, "high_in": high, "denoise": denoise}


def recover(raw: np.ndarray, old: np.ndarray, cache: dict, cam: int) -> tuple[dict | None, list]:
    """Settings that turn the stored frame into the old file exactly, and what was tried."""
    tried = []

    def test(low, high, denoise):
        s = settings_for(cache, low, high, denoise)
        new = apply_processing_pipeline_with_settings(raw, cache["bg"][cam], cam, s)
        eq = float(np.mean(new == old))
        tried.append({"low_in": low, "high_in": high, "denoise": denoise, "equal_fraction": eq})
        return s if eq == 1.0 else None

    for denoise in (False, True):
        s = test(0, 255, denoise)
        if s:
            return s, tried
    # the range the joint histogram points to (denoise off): pre-range values that give 0 and 255
    pre = apply_processing_pipeline_with_settings(raw, cache["bg"][cam], cam, settings_for(cache, 0, 255, False))
    zero = pre[old == 0]
    full = pre[old == 255]
    low0 = int(zero.max()) if zero.size else 0
    high0 = int(full.min()) if full.size else 255
    for low in range(max(0, low0 - 2), min(254, low0 + 2) + 1):
        for high in range(max(low + 1, high0 - 2), min(255, high0 + 2) + 1):
            s = test(low, high, False)
            if s:
                return s, tried
    return None, tried


def check(run: str, cam: int, cache: dict) -> dict:
    path = RAW_ROOT / run / f"CamC{cam + 1}.cine"
    raw_infer = pio.read_cine_frames(path, INFER_FRAME, 1)[0]
    settings, tried = recover(raw_infer, old_file(run, cam, INFER_FRAME), cache, cam)
    out = {"run": run, "cam": cam, "settings": settings, "tried": tried[-3:], "n_tried": len(tried)}
    if settings is None:
        out["ok"] = False
        return out
    new = {}
    for f in FRAMES:
        raw = pio.read_cine_frames(path, f, 1)[0]
        new[f] = apply_processing_pipeline_with_settings(raw, cache["bg"][cam], cam, settings)
    rows = []
    for f in FRAMES:
        if f == 12604:
            rows.append({"new": f, "old": None, "equal": None,
                         "differs_from_new_12603": bool(not np.array_equal(new[12604], new[12603]))})
            continue
        j = f if f <= 255 else f + 1
        o = old_file(run, cam, j)
        eq = bool(o is not None and np.array_equal(new[f], o))
        rows.append({"new": f, "old": j, "equal": eq,
                     "max_abs_diff": None if o is None else int(np.max(np.abs(new[f].astype(int) - o.astype(int))))})
    out["rows"] = rows
    out["ok"] = all(r["equal"] for r in rows if r["old"] is not None) and rows[-1]["differs_from_new_12603"]
    return out


def main() -> int:
    specs = sys.argv[1:] or DEFAULT
    caches, out = {}, []
    lines = ["F3 the fixed conversion (cached background, recovered range/denoise) against the old imgFile files", ""]
    for spec in specs:
        run, cam = spec.split(":")
        cam = int(cam)
        if run not in caches:
            caches[run] = load_cache(run)
        r = check(run, cam, caches[run])
        out.append(r)
        c = caches[run]
        lines.append(f"{'PASS' if r['ok'] else 'FAIL'}  {run} camera {cam}: cache shift {c['shifts'][cam]}, invert "
                     f"{c['invert']}, background of every {c['skip_frames']}th frame (up to {c['avg_count']})")
        if r["settings"] is None:
            lines.append(f"      settings not recovered; last tried: {r['tried']}")
            continue
        s = r["settings"]
        lines.append(f"      recovered: range {s['low_in']}-{s['high_in']}, denoise {s['denoise']} "
                     f"(after {r['n_tried']} tries)")
        lines.append("      " + ", ".join(
            f"new {x['new']} = old {x['old']}" if x["equal"] else
            (f"new {x['new']} vs old {x['old']}: DIFFERS (max {x['max_abs_diff']})" if x["old"] is not None else
             f"new {x['new']} (no old file) differs from new 12603: {x['differs_from_new_12603']}")
            for x in r["rows"]))
    n_fail = sum(1 for r in out if not r["ok"])
    lines += ["", "PASS" if not n_fail else f"{n_fail} FAIL"]
    text = "\n".join(lines) + "\n"
    print(text)
    (HERE / "f3_reproduce.txt").write_text(text, encoding="utf-8")
    (HERE / "f3_reproduce.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (HERE / "f3_settings.json").write_text(json.dumps(
        {f"{r['run']}:{r['cam']}": r["settings"] for r in out}, indent=1), encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
