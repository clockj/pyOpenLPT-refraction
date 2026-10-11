"""The batch's configs before and after the tracking changes of 11 Oct (report section 12): for each run, every setting
in which the config on disk (staging on D:, and the run's config.json on X:) differs from what make_configs.py now
writes.  Read only; run it before rewriting the configs, and again after (then nothing may differ).

    python tests/config_change_check.py RUN ... [--out NAME]     appends to tests/config_change_check.txt
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402
import make_configs  # noqa: E402


def diff(a, b, path=""):
    """Paths at which two JSON values differ: [(path, a, b)]."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}" if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [(path, a, b)]


def short(v):
    t = json.dumps(v)
    return t if len(t) <= 70 else t[:67] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    cfg = case.load_config()
    lines = [f"{datetime.datetime.now().isoformat(timespec='seconds')} {args.label}".rstrip()]
    for run in args.runs:
        n = make_configs.frames_of(run)
        for where, path, new in (
                ("staging", os.path.join(case.stage_dir(cfg, run), "config.json"),
                 make_configs.stage_config(cfg, n, run=run)),
                ("X:", os.path.join(case.case_dir(cfg, run), "config.json"), make_configs.case_config(cfg, n, run))):
            new = json.loads(make_configs.json_text(new))
            if not os.path.exists(path):
                lines.append(f"  {run} {where}: no config yet")
                continue
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            d = [(p, a, b) for p, a, b in diff(old, new) if p != "_comment"]
            comment = " (and its _comment)" if old.get("_comment") != new.get("_comment") else ""
            lines.append(f"  {run} {where}: {len(d)} setting(s) differ{comment}" + ("" if d else ": none"))
            lines += [f"      {p}: {short(a)} -> {short(b)}" for p, a, b in d]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "config_change_check.txt"), "a", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
