"""Parse an OpenLPT STB log into one record per frame.

    python parse_log.py LOG [--csv OUT.csv] [--jsonl OUT.jsonl]

Each record holds the frame, its phase (init / conv), the total time, the step times of the convergence phase
(prediction, shaking the predictions, removing overlaps, linking, pruning), every IPR round (camera combination,
2-D tolerance, detections per camera, matched objects, matching time, objects after shaking, shaking time), the
objects IPR found, and the track counters printed at the end of a convergence frame.  The CSV flattens the rounds into
per-combination sums.  Works on the old logs too (CR/CRLF line ends).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from typing import Dict, List

RE_FRAME = re.compile(r"^(Initial|Convergence) phase at frame (\d+)")
RE_TOTAL = re.compile(r"^Total time for frame (\d+): ([\d.]+)")
RE_COMBO = re.compile(r"^Combination \d+ cams=\[([\d, ]+)\]")
RE_DET = re.compile(r"2D detections per active camera:\s+([\d\s]+?)\s+\(([\d.]+) s\)")
RE_MATCH = re.compile(r"Matched (\d+) objects\. \( 2D tol: ([\d.]+), t:([\d.]+) s\)")
RE_SHAKE = re.compile(r"After shaking: (\d+) objects remain\. \(([\d.]+) s\)")
RE_FOUND = re.compile(r"IPR FINISH! FOUND (\d+) OBJECTS")
RE_STEP = {
    "t_pred": re.compile(r"^\s*Prediction: ([\d.]+) s"),
    "t_shake_pred": re.compile(r"^\s*Shake prediction: ([\d.]+) s"),
    "t_overlap": re.compile(r"Remove overlap: ([\d.]+) s"),
    "t_link": re.compile(r"^\s*Linking: ([\d.]+) s"),
    "t_prune": re.compile(r"Pruning time: ([\d.]+) s"),
}
RE_CNT = {
    "sa": re.compile(r"No\. of active short tracks: (\d+) \+ (\d+) - (\d+) = (\d+)"),
    "la": re.compile(r"No\. of active long tracks: (\d+) \+ (\d+) - (\d+) = (\d+)"),
    "ex": re.compile(r"No\. of exited long tracks: (\d+) \+ (\d+) = (\d+)"),
    "li": re.compile(r"No\. of inactive Long tracks: (\d+) \+ (\d+) = (\d+)"),
}
RE_ONE = {
    "ex_del": re.compile(r"No\. of exited short tracks\(deleted\): (\d+)"),
    "fail_shake": re.compile(r"No\. of fail shaking intensity/repeated: (\d+)"),
    "fail_lf": re.compile(r"No\. of fail linear fit: (\d+)"),
}
RE_INIT_DONE = re.compile(r"ACTIVE SHORT TRACKS: (\d+); ACTIVE LONG TRACKS: (\d+)")


def read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    return txt.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def parse(path: str) -> List[Dict]:
    """One dict per frame block that has a 'Total time' line (an unfinished last frame is dropped)."""
    out, cur, combo = [], None, None
    for ln in read_lines(path):
        m = RE_FRAME.match(ln)
        if m:
            cur = {"frame": int(m.group(2)), "phase": "init" if m.group(1) == "Initial" else "conv", "rounds": [],
                   "saved": False}
            combo = None
            continue
        if cur is None:
            continue
        m = RE_COMBO.match(ln)
        if m:
            combo = m.group(1).replace(" ", "").replace(",", "")
            continue
        m = RE_DET.search(ln)
        if m:
            cur["rounds"].append({"cams": combo, "det": [int(x) for x in m.group(1).split()]})
        m = RE_MATCH.search(ln)
        if m and cur["rounds"]:
            r = cur["rounds"][-1]
            r["matched"], r["tol"], r["t_match"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
        m = RE_SHAKE.search(ln)
        if m and cur["rounds"]:
            cur["rounds"][-1]["after_shake"], cur["rounds"][-1]["t_shake"] = int(m.group(1)), float(m.group(2))
        m = RE_FOUND.search(ln)
        if m:
            cur["ipr_found"] = int(m.group(1))
        for k, rx in RE_STEP.items():
            m = rx.search(ln)
            if m:
                cur[k] = float(m.group(1))
        for k, rx in RE_CNT.items():
            m = rx.search(ln)
            if m:
                vals = [int(x) for x in m.groups()]
                if k in ("sa", "la"):
                    cur[k + "_n"], cur[k + "_add"], cur[k + "_rm"], cur[k + "_new"] = vals
                else:
                    cur[k + "_n"], cur[k + "_add"], cur[k + "_new"] = vals
        for k, rx in RE_ONE.items():
            m = rx.search(ln)
            if m:
                cur[k] = int(m.group(1))
        m = RE_INIT_DONE.search(ln)
        if m:
            cur["sa_new"], cur["la_new"] = int(m.group(1)), int(m.group(2))
        if "Saved all tracks" in ln:
            cur["saved"] = True
        m = RE_TOTAL.match(ln)
        if m and int(m.group(1)) == cur["frame"]:
            cur["t_total"] = float(m.group(2))
            out.append(cur)
            cur = None
    return out


def flatten(rec: Dict) -> Dict:
    """A CSV row: scalars plus per-combination sums of matching time, matched objects and shaking time."""
    row = {k: v for k, v in rec.items() if k != "rounds"}
    row["t_match_all"] = sum(r.get("t_match", 0.0) for r in rec["rounds"])
    row["t_shake_ipr"] = sum(r.get("t_shake", 0.0) for r in rec["rounds"])
    for r in rec["rounds"]:
        key = r["cams"]
        row[f"t_match_{key}"] = row.get(f"t_match_{key}", 0.0) + r.get("t_match", 0.0)
        row[f"matched_{key}"] = row.get(f"matched_{key}", 0) + r.get("matched", 0)
        tk = f"matched_{key}_tol{r.get('tol', 0):g}"
        row[tk] = row.get(tk, 0) + r.get("matched", 0)
    if rec["rounds"]:
        row["det_first"] = sum(rec["rounds"][0]["det"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--csv")
    ap.add_argument("--jsonl")
    args = ap.parse_args()
    recs = parse(args.log)
    print(f"{len(recs)} frames ({recs[0]['frame'] if recs else '-'}..{recs[-1]['frame'] if recs else '-'})")
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    if args.csv:
        rows = [flatten(r) for r in recs]
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
