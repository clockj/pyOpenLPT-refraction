"""The report on when the flat-window calibration cannot work (2026-10-03-when-calibration-fails.md), as a page: fills
figures/fails_template.html with the data of its two figures, drawn from saved results (nothing is re-fitted), and
writes 2026-10-03-when-calibration-fails.html beside this script.
  Figure 1: every W5/R4 case with a window seen by only one or two cameras, at the widest view among that window's
            cameras (the case's narrowest such window; p_weak_windows.json, the true f), with its verdict before the
            scan (the stored answers), after it (h5a_warnings.json) and with hints 0-10 deg off (h1b_hints_regression.json).
  Figure 2: the five hard cases with hints 0-30 deg off, run by run (h4b_scan.json).
The page opens in a browser as it is.  --fragment PATH also writes it without its document skeleton, for publishing as
an artifact (the publisher adds its own).
Usage: make_fails_page.py [--fragment PATH]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "figures" / "fails_template.html"
OUT = HERE / "2026-10-03-when-calibration-fails.html"
HARD = [("R4", 3110, 1.0), ("W5", 1031, 0.1), ("W5", 1034, 0.1), ("W5", 1001, 0.1), ("W5", 1045, 0.1)]


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def verdict(r):
    return "same" if r["ok"] else "equivalent" if r["equivalent"] else "wrong"


def weak_cases():
    pw = load("p_weak_windows.json")
    after = {(r["set"], r["seed"], r["noise_px"]): r for r in load("h5a_warnings.json")["runs"] if "error" not in r}
    hinted = {(r["set"], r["seed"], r["noise_px"]): r for r in load("h1b_hints_regression.json")["runs"]
              if "error" not in r}
    cases, others = [], []
    for st, seed, noise, v_before in pw["runs"]:
        rig = pw["rigs"][str(seed)]
        key = (st, seed, noise)
        row = {"set": st, "seed": seed, "noise": noise, "per_window": [len(w["cams"]) for w in rig["windows"]],
               "before": "wrong" if v_before == "DIFFERENT" else v_before, "after": verdict(after[key]),
               "hinted": verdict(hinted[key])}
        wins = [(k, w) for k, w in enumerate(rig["windows"]) if len(w["cams"]) <= 2]
        if not wins:
            others.append(row)
            continue
        k, w = min(wins, key=lambda kw: max(kw[1]["image_deg"]))
        row.update(window=k + 1, cams=len(w["cams"]), f=[round(x) for x in w["f_px"]],
                   views=[round(a, 2) for a in w["image_deg"]], angle=round(max(w["image_deg"]), 3))
        row["cat"] = "wrong" if row["after"] == "wrong" else ("fixed" if row["before"] == "wrong" else "right")
        cases.append(row)
    return cases, others


def hint_grid():
    runs = load("h4b_scan.json")["runs"]
    rows = []
    for st, seed, noise in HARD:
        rr = [r for r in runs if r["seed"] == seed and r["noise_px"] == noise]
        none = next(r for r in rr if r["delta_deg"] is None)
        cells = []
        for r in sorted(rr, key=lambda r: (-1.0 if r["delta_deg"] is None else r["delta_deg"], r["draw"])):
            cells.append({"delta": r["delta_deg"], "draw": r["draw"], "v": verdict(r), "rms": r["rms_px"],
                          "ref": r["ref_rms_px"], "z": r["max_abs_z"],
                          "normals": [round(x, 2) for x in r["normal_err_deg"]],
                          "path": r.get("chosen") or "default",
                          "scan": bool((r.get("scan") or {}).get("improved"))})
        rows.append({"set": st, "seed": seed, "noise": noise, "per_window": none["cams_per_window"],
                     "f": [round(x) for x in none["f_px"]], "cells": cells})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", help="also write the page without its document skeleton to this path")
    args = ap.parse_args()
    cases, others = weak_cases()
    data = {"weak": cases, "others": len(others), "others_wrong": sum(1 for o in others if o["after"] == "wrong"),
            "hints": hint_grid()}
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count("__FIG_DATA__") == 1 and html.count("<!--BODY-->") == 1
    html = html.replace("__FIG_DATA__", json.dumps(data, separators=(",", ":")))
    head, body = html.split("<!--BODY-->")
    OUT.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   + head + "</head>\n<body>\n" + body + "</body>\n</html>\n", encoding="utf-8")
    if args.fragment:
        Path(args.fragment).write_text(head + body, encoding="utf-8")
    print(f"wrote {OUT.name}: figure 1 {len(cases)} cases (+{len(others)} with three or more cameras on every "
          f"window), figure 2 {sum(len(r['cells']) for r in data['hints'])} runs"
          + (f"; fragment {args.fragment}" if args.fragment else ""))


if __name__ == "__main__":
    main()
