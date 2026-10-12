"""G3: the cameras' image lists aligned by their Phantom headers (plan: 2026-10-08-plan.md, test G3).  No Qt.

    python g3_alignment.py

  1. phantom_image_numbers on run 20261008's folders: T0 cam1, cam2 (0, 13001), cam3 (2396, 10605); T1's three
     (0, 1001); a folder without a .chd (this test's own folder) gives None;
  2. align_by_image_numbers on T0's real file lists: cams 1-2 start at frame002396.tif, every list has 10,605 files,
     and the i-th files are the same image number (the names give it); cam3 is unchanged;
  3. no change when the first numbers are equal (T1), when a camera has no number, or with one camera;
  4. a synthetic pair that also ends apart (0-99 and 10-119): cut to 10-99 in both.
Writes g3_alignment_out.txt and .json beside itself.
"""
import common as C  # noqa: I001

import os  # noqa: E402
import re  # noqa: E402

log = C.Log("g3_alignment")
T1_RAW = C.T0_RAW.parent / "T1"


def tif_list(folder):
    return sorted(str(folder / n) for n in os.listdir(folder) if n.lower().endswith(".tif"))


def number(path, first):
    return int(re.search(r"frame(\d+)\.tif$", path).group(1)) + first


def main():
    from modules.camera_calibration.wand_calibration import wand_calibrator as wc
    out = {}

    log("1. phantom_image_numbers")
    hdr = {}
    for run, root in (("T0", C.T0_RAW), ("T1", T1_RAW)):
        for c in (1, 2, 3):
            hdr[(run, c)] = wc.phantom_image_numbers(str(root / f"cam{c}"))
            log(f"  {run} cam{c}: {hdr[(run, c)]}")
    log.check("T0: cams 1-2 (0, 13001), cam3 (2396, 10605)", hdr[("T0", 1)] == (0, 13001) and hdr[("T0", 2)] == (0, 13001)
              and hdr[("T0", 3)] == (2396, 10605))
    log.check("T1: every camera (0, 1001)", all(hdr[("T1", c)] == (0, 1001) for c in (1, 2, 3)))
    log.check("a folder without a .chd: None", wc.phantom_image_numbers(str(C.HERE)) is None)
    out["headers"] = {f"{r}/cam{c}": v for (r, c), v in hdr.items()}

    log("\n2. T0's file lists aligned")
    lists = {k: tif_list(C.T0_RAW / f"cam{k + 1}") for k in range(3)}
    firsts = {k: hdr[("T0", k + 1)][0] for k in range(3)}
    al, info = wc.align_by_image_numbers(lists, firsts)
    log(f"  info: {info}")
    log(f"  first files: " + ", ".join(f"cam{k + 1} {os.path.basename(al[k][0])}" for k in range(3))
        + "; last: " + ", ".join(f"cam{k + 1} {os.path.basename(al[k][-1])}" for k in range(3)))
    n = {k: len(al[k]) for k in range(3)}
    same = all(number(al[0][i], firsts[0]) == number(al[1][i], firsts[1]) == number(al[2][i], firsts[2])
               for i in range(min(n.values())))
    log.check("every list has 10,605 files; cams 1-2 start at frame002396.tif, cam3 at frame000000.tif",
              set(n.values()) == {10605} and os.path.basename(al[0][0]) == "frame002396.tif"
              and os.path.basename(al[1][0]) == "frame002396.tif" and os.path.basename(al[2][0]) == "frame000000.tif",
              f"{n}")
    log.check("the i-th files are the same image number in every camera", same)
    log.check("the dropped counts: cams 1-2 2,396 at the start, none at the end; cam3 none",
              info and info["dropped"] == {0: (2396, 0), 1: (2396, 0), 2: (0, 0)} and (info["start"], info["end"]) == (2396, 13000))
    log.check("cam3's list is unchanged; the input lists are not modified", al[2] == lists[2] and len(lists[0]) == 13001)

    log("\n3. no alignment when it is not needed or not possible")
    t1 = {k: tif_list(T1_RAW / f"cam{k + 1}") for k in range(3)}
    a1, i1 = wc.align_by_image_numbers(t1, {k: hdr[("T1", k + 1)][0] for k in range(3)})
    a2, i2 = wc.align_by_image_numbers(lists, {0: 0, 1: 0, 2: None})
    a3, i3 = wc.align_by_image_numbers({0: lists[2]}, {0: 2396})
    log.check("equal first numbers (T1): lists unchanged, no info", i1 is None and all(a1[k] == t1[k] for k in range(3)))
    log.check("a camera without a header: lists unchanged, no info", i2 is None and all(a2[k] == lists[k] for k in range(3)))
    log.check("one camera: unchanged", i3 is None and a3[0] == lists[2])

    log("\n4. a synthetic pair that also ends apart")
    s = {0: [f"a{i}" for i in range(100)], 1: [f"b{i}" for i in range(110)]}
    a4, i4 = wc.align_by_image_numbers(s, {0: 0, 1: 10})
    log(f"  info: {i4}; first {a4[0][0]}, {a4[1][0]}; last {a4[0][-1]}, {a4[1][-1]}")
    log.check("cut to images 10-99 in both", i4 == {"start": 10, "end": 99, "dropped": {0: (10, 0), 1: (0, 20)}}
              and a4[0][0] == "a10" and a4[1][0] == "b0" and a4[0][-1] == "a99" and a4[1][-1] == "b89")
    log.save(out)


if __name__ == "__main__":
    main()
