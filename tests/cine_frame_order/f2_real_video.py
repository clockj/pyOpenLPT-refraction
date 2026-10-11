"""F2: the fixed reading on the real 20260727 videos (read only; plan: 2026-10-10-plan.md, Part A).

For each video, frames 0, 1, 255, 256, 257 and the last are read with io.read_cine_frames, alone and inside chunks of
256 (the GUI batch's size), and compared with the stored frames read straight from their file offsets (the header's
pImage table, decoded with pycine's create_raw_array, so pycine's start_frame handling plays no part).  Also shown:
what the old call (pycine's read_frames with the frame number) returns for frames 1 and 256.

    python tests/cine_frame_order/f2_real_video.py [RUN:CAM ...]     (default 35Hz_r_b_1:0 35Hz_r_b_1:1)
Writes f2_real_video.txt and f2_real_video.json here.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from pycine.file import read_header  # noqa: E402
from pycine.raw import create_raw_array, read_frames  # noqa: E402

from modules.image_preprocessing import io as pio  # noqa: E402

RAW_ROOT = Path(r"X:\Shijie Zhong\Bubble Shear Project\Raw_Data\20260727")
DEFAULT = ["35Hz_r_b_1:0", "35Hz_r_b_1:1"]


def stored_direct(path: Path, header: dict, pos: int) -> np.ndarray:
    """Stored frame at 0-based position pos, read at its file offset."""
    with open(path, "rb") as f:
        f.seek(header["pImage"][pos])
        annotation_size = struct.unpack("<I", f.read(4))[0]
        f.read(annotation_size - 8)
        image_size = struct.unpack("<I", f.read(4))[0]
        data = f.read(image_size)
    return create_raw_array(data, header)


def check(run: str, cam: int) -> dict:
    path = RAW_ROOT / run / f"CamC{cam + 1}.cine"
    header = read_header(str(path))
    first_no = int(header["cinefileheader"].FirstImageNo)
    n = int(header["cinefileheader"].ImageCount)
    last = first_no + n - 1
    frames = [first_no, first_no + 1, first_no + 255, first_no + 256, first_no + 257, last]
    direct = {f: stored_direct(path, header, f - first_no) for f in frames}
    rows = []
    for f in frames:
        alone = pio.read_cine_frames(path, f, 1, header=header)[0]
        rows.append({"frame": f, "alone_equal": bool(np.array_equal(alone, direct[f]))})
    # inside chunks of 256: the chunk starting at 256 and the last chunk
    chunk_start = first_no + 256
    chunk = pio.read_cine_frames(path, chunk_start, 256, header=header)
    in_chunk = {chunk_start: np.array_equal(chunk[0], direct[first_no + 256]),
                chunk_start + 1: np.array_equal(chunk[1], direct[first_no + 257])}
    last_start = first_no + 256 * ((n - 1) // 256)
    tail = pio.read_cine_frames(path, last_start, last - last_start + 1, header=header)
    in_chunk[last] = np.array_equal(tail[-1], direct[last])
    # the old call: the camera's frame number handed to pycine
    old1 = np.asarray(list(read_frames(str(path), start_frame=first_no + 1, count=1)[0])[0])
    old256 = np.asarray(list(read_frames(str(path), start_frame=first_no + 256, count=1)[0])[0])
    old = {"frame_1_is_stored_0": bool(np.array_equal(old1, direct[first_no])),
           "frame_256_is_stored_255": bool(np.array_equal(old256, direct[first_no + 255]))}
    ok = all(r["alone_equal"] for r in rows) and all(in_chunk.values())
    return {"run": run, "cam": cam, "cine": str(path), "FirstImageNo": first_no, "ImageCount": n, "rows": rows,
            "in_chunks_equal": {str(k): bool(v) for k, v in in_chunk.items()}, "old_call": old, "ok": bool(ok)}


def main() -> int:
    specs = sys.argv[1:] or DEFAULT
    lines = ["F2 io.read_cine_frames on the real videos against the stored frames at their file offsets", ""]
    out = []
    for spec in specs:
        run, cam = spec.split(":")
        r = check(run, int(cam))
        out.append(r)
        lines.append(f"{'PASS' if r['ok'] else 'FAIL'}  {run} camera {cam} (FirstImageNo {r['FirstImageNo']}, "
                     f"{r['ImageCount']} frames)")
        lines.append("      alone: " + ", ".join(f"{x['frame']} {'=' if x['alone_equal'] else 'DIFFERS'}"
                                             for x in r["rows"]))
        lines.append("      in chunks of 256: " + ", ".join(f"{k} {'=' if v else 'DIFFERS'}"
                                                        for k, v in r["in_chunks_equal"].items()))
        lines.append(f"      old call: frame 1 gives stored frame 0: {r['old_call']['frame_1_is_stored_0']}; "
                     f"frame 256 gives stored frame 255: {r['old_call']['frame_256_is_stored_255']}")
    n_fail = sum(1 for r in out if not r["ok"])
    lines += ["", "PASS" if not n_fail else f"{n_fail} FAIL"]
    text = "\n".join(lines) + "\n"
    print(text)
    (HERE / "f2_real_video.txt").write_text(text, encoding="utf-8")
    (HERE / "f2_real_video.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
