"""F1: the .cine frame reading on synthetic videos, through the real pycine (plan: 2026-10-10-plan.md, Part A).

Each synthetic video holds N = 600 frames of 16 x 8 pixels (16-bit, uncompressed).  Stored frame k carries k twice:
in binary in columns 0-11 (4095 for a set bit, 0 otherwise, so it survives any processing to 8 bits) and as the raw
value of columns 12-15 (for averages).  FirstImageNo is 0, 1, 5 or -300, so the camera's frame numbers run from
FirstImageNo to FirstImageNo + 599 and stored frame k is frame FirstImageNo + k.

Checks:
  1. pycine's quirk: read_frames(start_frame=0) and (start_frame=1) both return stored frame 0.
  2. the old reading (the camera's frame number handed to pycine, chunks of 256 as the GUI batch did) reproduces the
     20260727 pattern: file 256 repeats stored frame 255, file j holds j - 1 after it, the last frame is lost.
  3. io.read_cine_frames: every frame alone and in chunks of 256, 128 and 100 is the right stored frame, for every
     FirstImageNo; frames outside the video raise ValueError.
  4. the GUI batch worker (PreprocessWorker.run, no window): file i holds stored frame i, all N files written.
  5. the GUI background average (BackgroundWorker._compute_one_camera), every 20th frame: the mean of the right frames.
  6. io.compute_mean_background_from_cine (command line), stride 1 (chunks of 128) and 20: the mean of the right frames
     (from frame max(0, FirstImageNo): its `start` is a frame number, default 0).
  7. the command-line conversion (plan_cine_tasks + run_batch_processing, 1 and 4 workers): file i holds stored
     frame i, all N files written.
  8. the GUI preview (ImagePreprocessingView._read_image): the right stored frame.

    python tests/cine_frame_order/f1_synthetic.py [--code DIR --label NAME]
Writes f1_synthetic.txt and f1_synthetic.json here (f1_synthetic_NAME.* with --label); the videos and converted files
go to f1_data/ here.  --code DIR runs the checks on another copy of `modules` (e.g. the code before the fix, exported
with git archive); check 3 is then skipped if that code has no read_cine_frames.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--code", default=str(REPO), help="folder holding the `modules` package to test")
_ap.add_argument("--label", default="")
ARGS = _ap.parse_args()
sys.path.insert(0, ARGS.code)

from pycine import cine as pycine_cine  # noqa: E402
from pycine.raw import read_frames  # noqa: E402

from modules.image_preprocessing import io as pio  # noqa: E402
from modules.image_preprocessing import runner  # noqa: E402
from modules.image_preprocessing import view  # noqa: E402

N = 600
W, H = 16, 8
FIRST_NOS = [0, 1, 5, -300]
SUFFIX = f"_{ARGS.label}" if ARGS.label else ""
DATA = HERE / f"f1_data{SUFFIX}"
SETTINGS = {"bg_enabled": False, "invert": False, "low_in": 0, "high_in": 255, "denoise": False,
            "cine_shifts": {0: 4}}


def frame_image(k: int) -> np.ndarray:
    img = np.zeros((H, W), np.uint16)
    for c in range(12):
        if (k >> c) & 1:
            img[:, c] = 4095
    img[:, 12:] = k
    return img


def decode(img: np.ndarray) -> int:
    """Stored position from the binary columns of a raw (12-bit) or converted (8-bit) image."""
    img = np.asarray(img)
    half = 2047 if img.max() > 255 else 127
    bits = np.median(img[:, :12], axis=0) > half
    return int(sum(1 << c for c in range(12) if bits[c]))


def write_cine(path: Path, first_no: int) -> None:
    fh, bh, su = pycine_cine.CINEFILEHEADER(), pycine_cine.BITMAPINFOHEADER(), pycine_cine.SETUP()
    fh.Type = 0x4943  # "CI"
    fh.Headersize = ctypes.sizeof(fh)
    fh.Version = 1
    fh.FirstMovieImage = first_no
    fh.TotalImageCount = N
    fh.FirstImageNo = first_no
    fh.ImageCount = N
    fh.OffImageHeader = ctypes.sizeof(fh)  # pycine reads the bitmap header right after the file header
    fh.OffSetup = fh.OffImageHeader + ctypes.sizeof(bh)
    fh.OffImageOffsets = fh.OffSetup + ctypes.sizeof(su)
    bh.biSize = ctypes.sizeof(bh)
    bh.biWidth, bh.biHeight, bh.biPlanes, bh.biBitCount = W, H, 1, 16
    bh.biCompression, bh.biSizeImage = 0, W * H * 2
    su.RealBPP, su.BlackLevel, su.WhiteLevel = 12, 0, 4095  # pycine maps [0, 4095] onto itself
    block = 8 + W * H * 2  # annotation size (8: no annotation), image size, pixels
    first_off = fh.OffImageOffsets + 8 * N
    with open(path, "wb") as f:
        f.write(bytes(fh))
        f.write(bytes(bh))
        f.write(bytes(su))
        f.write(struct.pack(f"<{N}q", *[first_off + k * block for k in range(N)]))
        for k in range(N):
            f.write(struct.pack("<II", 8, W * H * 2))
            f.write(np.flipud(frame_image(k)).astype("<u2").tobytes())  # pycine flips uncompressed frames back


def old_gui_files(path: Path, first_no: int, chunk: int = 256) -> list[int]:
    """The pre-fix GUI batch: chunks of the camera's frame numbers handed to pycine as they are."""
    frames = list(range(first_no, first_no + N))
    out = []
    for p in range(0, N, chunk):
        start = frames[p]
        cnt = min(chunk, N - p)
        imgs = list(read_frames(str(path), start_frame=start, count=cnt)[0])
        out += [decode(im) for im in imgs]
    return out


def converted(folder: Path, name: str = "img{:06d}.tif") -> list[int | None]:
    """Stored position in each converted file (the GUI names them img<i>.tif, the command line frame_<i>.tif)."""
    out = []
    for i in range(N):
        img = cv2.imread(str(folder / name.format(i)), cv2.IMREAD_UNCHANGED)
        out.append(None if img is None else decode(img))
    return out


def summary(got: list, want: list) -> dict:
    bad = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
    return {"ok": not bad and len(got) == len(want), "n": len(got), "n_wrong": len(bad), "first_wrong": bad[:5],
            "got_at_first_wrong": [got[i] for i in bad[:5]]}


def main() -> int:
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir()
    res, lines = {}, ["F1 .cine frame reading on synthetic videos (600 frames each), through the real pycine",
                      f"code tested: {Path(pio.__file__).parent}", ""]

    def record(name, ok, detail):
        res[name] = {"ok": bool(ok), **detail}
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}: {json.dumps(detail, default=str)}")

    videos = {}
    for first_no in FIRST_NOS:
        p = DATA / f"video_first{first_no}.cine"
        write_cine(p, first_no)
        videos[first_no] = p

    # 1. pycine's quirk
    p0 = videos[0]
    k0 = decode(list(read_frames(str(p0), start_frame=0, count=1)[0])[0])
    k1 = decode(list(read_frames(str(p0), start_frame=1, count=1)[0])[0])
    k2 = decode(list(read_frames(str(p0), start_frame=2, count=1)[0])[0])
    record("1 pycine start_frame 0, 1, 2 give stored frames", (k0, k1, k2) == (0, 0, 1),
           {"stored": [k0, k1, k2], "expected": [0, 0, 1]})

    # 2. the old reading reproduces the 20260727 pattern
    old = old_gui_files(p0, 0)
    want_old = list(range(256)) + list(range(255, N - 1))
    record("2 old GUI reading, FirstImageNo 0: file 256 = stored 255, file j = j - 1 after, last frame lost",
           old == want_old, {"files_255_258": old[255:259], "last_file": old[-1], "stored_599_in_files": N - 1 in old})
    old_neg = old_gui_files(videos[-300], -300)
    record("2b old GUI reading, FirstImageNo -300 (shown, not required)", True,
           {"files_0_3": old_neg[:4], "files_298_303": old_neg[298:304]})

    # 3. read_cine_frames
    if not hasattr(pio, "read_cine_frames"):
        lines.append("SKIP  3 read_cine_frames: not in this code")
    for first_no, p in (videos.items() if hasattr(pio, "read_cine_frames") else []):
        single = [decode(pio.read_cine_frames(p, first_no + k, 1)[0]) for k in range(N)]
        record(f"3 read_cine_frames one at a time, FirstImageNo {first_no}", single == list(range(N)),
               summary(single, list(range(N))))
        for chunk in (256, 128, 100):
            got = []
            for s in range(0, N, chunk):
                got += [decode(im) for im in pio.read_cine_frames(p, first_no + s, min(chunk, N - s))]
            record(f"3 read_cine_frames chunks of {chunk}, FirstImageNo {first_no}", got == list(range(N)),
                   summary(got, list(range(N))))
        errors = []
        for frame, count in ((first_no - 1, 1), (first_no + N, 1), (first_no + N - 2, 3)):
            try:
                pio.read_cine_frames(p, frame, count)
                errors.append(f"{frame}+{count}: no error")
            except ValueError:
                pass
        record(f"3 read_cine_frames out of range raises, FirstImageNo {first_no}", not errors, {"problems": errors})

    # 4. the GUI batch worker
    for first_no in (0, -300):
        p = videos[first_no]
        out = DATA / f"gui_batch_first{first_no}" / "cam0"
        out.mkdir(parents=True)
        tasks = [{"src": f"{p}#{j}", "dst": str(out / f"img{i:06d}.tif"), "cam_idx": 0, "frame_idx": i,
                  "is_cine": True, "cine_file": str(p), "cine_frame": j}
                 for i, j in enumerate(range(first_no, first_no + N))]
        settings = dict(SETTINGS, camera_backgrounds={})
        finished = {}
        worker = view.PreprocessWorker(tasks, settings, process_workers=4, write_workers=2)
        worker.finished.connect(lambda d: finished.update(d))
        worker.run()
        got = converted(out)
        record(f"4 GUI batch worker, FirstImageNo {first_no}", got == list(range(N)) and finished.get("failed") == 0,
               {**summary(got, list(range(N))), "worker": {k: finished.get(k) for k in ("total", "processed", "failed")}})

    # 5. the GUI background average (every 20th frame, chunks of 100)
    bgw = view.BackgroundWorker({}, {}, invert_enabled=False)
    for first_no in (0, 5):
        p = videos[first_no]
        images = [f"{p}#{j}" for j in range(first_no, first_no + N)]
        sel = list(range(0, N, 20))
        r = bgw._compute_one_camera(0, images, sel)
        mean = float(np.median(r["bg"][:, 12:]))
        want = float(np.mean(sel))
        record(f"5 GUI background average, every 20th frame, FirstImageNo {first_no}",
               abs(mean - want) < 1e-3 and r["count"] == len(sel),
               {"mean_of_stored": mean, "expected": want, "count": r["count"]})

    # 6. the command-line background average
    for first_no in (0, -300):
        p = videos[first_no]
        for stride in (1, 20):
            bg = pio.compute_mean_background_from_cine(cine_path=p, stride=stride)
            mean = float(np.median(bg[:, 12:]))
            # its `start` (default 0) is a frame number in the camera's numbering, so it begins at frame
            # max(0, FirstImageNo): for FirstImageNo -300 that is stored frame 300
            frames = range(max(0, first_no), first_no + N, stride)
            want = float(np.mean([f - first_no for f in frames]))
            record(f"6 command-line background, stride {stride}, FirstImageNo {first_no}", abs(mean - want) < 1e-3,
                   {"mean_of_stored": mean, "expected": want, "frames_averaged": len(frames)})

    # 7. the command-line conversion
    for first_no in (0, -300):
        p = videos[first_no]
        for workers in (1, 4):
            out = DATA / f"cli_first{first_no}_w{workers}"
            tasks = pio.plan_cine_tasks(cam_idx=0, cine_path=p, output_dir=out)
            result = runner.run_batch_processing(tasks, output_dir=out, settings=SETTINGS, workers=workers)
            got = converted(out / "cam0", "frame_{:06d}.tif")
            record(f"7 command-line conversion, {workers} worker(s), FirstImageNo {first_no}",
                   got == list(range(N)) and not result.failures,
                   {**summary(got, list(range(N))), "failures": list(result.failures)[:3]})

    # 8. the GUI preview
    for first_no in (0, -300):
        p = videos[first_no]
        ks = [0, 1, 255, 256, 257, N - 1]
        got = [decode(view.ImagePreprocessingView._read_image(None, f"{p}#{first_no + k}")) for k in ks]
        record(f"8 GUI preview, FirstImageNo {first_no}", got == ks, {"frames": ks, "stored": got})

    n_fail = sum(1 for r in res.values() if not r["ok"])
    lines += ["", f"{len(res) - n_fail} of {len(res)} checks pass" + ("" if n_fail else ": PASS")]
    text = "\n".join(lines) + "\n"
    print(text)
    (HERE / f"f1_synthetic{SUFFIX}.txt").write_text(text, encoding="utf-8")
    (HERE / f"f1_synthetic{SUFFIX}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
