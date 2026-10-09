"""Files of the tracer recording T1 of run 20261008 (plan: 2026-10-08-plan.md, Phase T; the 7 Oct case's t1.py with
a third camera).

The raw frames live on the slow network share (config.json: tracers.raw_dir).  `copy_local()` makes one local copy
in data/T1raw/<cam>/ (git-ignored) with a manifest, and everything else reads that copy.

    python t1.py copy      # the local copy (about 6 GB), skipped for files already copied with the same size
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
DATA = os.path.join(HERE, "data")
LOCAL_RAW = os.path.join(DATA, "T1raw")
LOCAL_PRE = os.path.join(DATA, "T1pre")
CAMS = ("cam1", "cam2", "cam3")

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # the Phantom TIFF tag warning


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def frames(cfg: dict | None = None) -> list:
    cfg = cfg or load_config()
    fr = cfg["tracers"]["frames"]
    return list(range(int(fr["first"]), int(fr["last"]) + 1))


def name(k: int, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    return cfg["tracers"]["frames"]["pattern"].format(frame=int(k))


def raw_path(cam: str, k: int, local: bool = True, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    root = LOCAL_RAW if local else cfg["tracers"]["raw_dir"]
    return os.path.join(root, cam, name(k, cfg))


def read_file(path: str) -> np.ndarray:
    """An 8-bit grey image, read in one go and decoded in memory."""
    with open(path, "rb") as fh:
        buf = np.frombuffer(fh.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def read(cam: str, k: int, local: bool = True) -> np.ndarray:
    return read_file(raw_path(cam, k, local))


def read_stack(cam: str, ks, local: bool = True, dtype=np.float32) -> np.ndarray:
    return np.stack([read(cam, k, local).astype(dtype) for k in ks])


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def track_frames(cfg: dict | None = None) -> list:
    """The raw frames that are tracked (config tracers.track_raw_frames: the user, 8 Oct: the first 250)."""
    cfg = cfg or load_config()
    a, b = cfg["tracers"].get("track_raw_frames", [cfg["tracers"]["frames"]["first"], cfg["tracers"]["frames"]["last"]])
    return list(range(int(a), int(b) + 1))


def copy_local(log=print) -> dict:
    """Copy the tracked T1 frames (track_frames) of every camera from the share to data/T1raw/<cam>/; a file already
    there with the same size is kept.  Writes data/T1raw/manifest.json: per camera the count, total bytes, and the
    sha256 of the first, middle and last frame copied (checked against the share's copies)."""
    cfg = load_config()
    ks = track_frames(cfg)
    out = {"source": cfg["tracers"]["raw_dir"], "frames": [ks[0], ks[-1]], "cameras": {}}
    t0 = time.time()
    for cam in CAMS:
        os.makedirs(os.path.join(LOCAL_RAW, cam), exist_ok=True)
        n_copied = n_kept = total = 0
        for k in ks:
            src, dst = raw_path(cam, k, False, cfg), raw_path(cam, k, True, cfg)
            size = os.path.getsize(src)
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                n_kept += 1
            else:
                with open(src, "rb") as fh:
                    b = fh.read()
                with open(dst + ".tmp", "wb") as fh:
                    fh.write(b)
                os.replace(dst + ".tmp", dst)
                n_copied += 1
            total += size
        sample = {}
        for k in (ks[0], ks[len(ks) // 2], ks[-1]):
            a, b = _sha256(raw_path(cam, k, False, cfg)), _sha256(raw_path(cam, k, True, cfg))
            sample[name(k, cfg)] = {"sha256": b, "matches_share": a == b}
        out["cameras"][cam] = {"frames": len(ks), "copied": n_copied, "already_there": n_kept,
                               "bytes": total, "sample_sha256": sample}
        log(f"  {cam}: {len(ks)} frames ({n_copied} copied, {n_kept} already there), {total / 1e6:.0f} MB; "
            f"sample sha256 match the share: {all(v['matches_share'] for v in sample.values())}")
    out["seconds"] = time.time() - t0
    with open(os.path.join(LOCAL_RAW, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "copy":
        r = copy_local()
        print(f"done in {r['seconds']:.0f} s")
    else:
        print(__doc__)
