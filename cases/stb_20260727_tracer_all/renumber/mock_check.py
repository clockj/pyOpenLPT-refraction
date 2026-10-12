"""R0: renumber.py on a local mock run before it touches X: (rename, interrupt, resume, checks, undo).

The mock run (renumber/mock/<run>/imgFile_bb/cam0) holds 12,605 small files whose content is their old index, with
256 a copy of 255.  renumber() is interrupted after 3,000 renames (an os.rename that fails once), resumed, checked
(file k holds old k + 1 from 256 on, the copy aside, 0-12,603 present), then undone and checked against the start.

    python renumber/mock_check.py
Writes renumber/mock_check.txt; removes the mock folder at the end.
"""
from __future__ import annotations

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import renumber as rn  # noqa: E402

ROOT = os.path.join(HERE, "mock")
RUN = "mockrun"
FOLDER = "imgFile_bb"


def content(d, k):
    with open(os.path.join(d, rn.NAME.format(k)), "rb") as f:
        return f.read()


def main():
    lines = ["R0 renumber.py on a local mock run (12,605 files, 256 a copy of 255)", ""]
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    d = os.path.join(ROOT, RUN, FOLDER, "cam0")
    os.makedirs(d)
    for k in range(rn.N_FILES):
        with open(os.path.join(d, rn.NAME.format(k)), "wb") as f:
            f.write(f"old {k if k != rn.COPY else rn.COPY - 1}".encode())
    rn.case.case_dir = lambda cfg, run: os.path.join(ROOT, run)  # the mock instead of X:
    rn.LOGS = os.path.join(ROOT, "logs")
    rn.HERE = ROOT  # renumber_log.txt into the mock folder
    cfg = rn.case.load_config()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        lines.append(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")

    check("state before", rn.state(cfg, RUN, FOLDER, 0)["state"] == "original")
    real_rename, calls = os.rename, {"n": 0}

    def failing_rename(a, b):
        calls["n"] += 1
        if calls["n"] == 3002:  # the copy aside, then 3,000 renames, then a failure
            raise OSError("simulated interruption")
        real_rename(a, b)

    rn.os.rename = failing_rename
    try:
        rn.renumber(cfg, RUN, FOLDER, 0, {})
        check("interruption happened", False)
    except OSError:
        pass
    rn.os.rename = real_rename
    st = rn.state(cfg, RUN, FOLDER, 0)
    check("state after the interruption is partial with one gap", st["state"] == "partial", str(st))
    out = rn.renumber(cfg, RUN, FOLDER, 0, {})
    check("resumed and checked", out["state"] == "renumbered" and not out["problems"], str(out))
    want = [f"old {k}".encode() for k in range(rn.COPY)] + [f"old {k + 1}".encode() for k in range(rn.COPY, rn.LAST)]
    have = [content(d, k) for k in range(rn.LAST)]
    check("file k holds old k (k < 256) and old k + 1 (256 <= k <= 12603)", have == want)
    check("12604 is gone", not os.path.exists(os.path.join(d, rn.NAME.format(rn.LAST))))
    aside = os.path.join(ROOT, RUN, "removed_copy256", FOLDER, "cam0", rn.NAME.format(rn.COPY))
    check("the copy is aside", os.path.exists(aside) and open(aside, "rb").read() == b"old 255")
    check("state renumbered", rn.state(cfg, RUN, FOLDER, 0)["state"] == "renumbered")
    out2 = rn.renumber(cfg, RUN, FOLDER, 0, {})
    check("a second run renames nothing", out2["renames"] == 0 and out2["state"] == "renumbered", str(out2))
    rn.undo(cfg, RUN, FOLDER, 0)
    back = [content(d, k) for k in range(rn.N_FILES)]
    orig = [f"old {k if k != rn.COPY else rn.COPY - 1}".encode() for k in range(rn.N_FILES)]
    check("undo restores every file", back == orig)
    check("state after undo is original", rn.state(cfg, RUN, FOLDER, 0)["state"] == "original")
    lines += ["", "PASS" if ok else "FAIL"]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "mock_check.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    shutil.rmtree(ROOT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
