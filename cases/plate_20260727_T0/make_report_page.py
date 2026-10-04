"""The case report (2026-10-03-report.md) as a page: fills page/report_template.html with the data of its four
figures and its tables, drawn from saved results (nothing is re-fitted), and writes 2026-10-03-report.html beside
this script.
  Figure 1 and the camera table: the delivered fit and calib_3 placed against the two walls
            (tests/fits_summary.json "plan" and "fits"; tests/t7_vs_gui.json "rigs").
  Figure 2 and the volume table: the rig in 3-D (three.js), the cameras' refracted fields of view, the wand's ends
            and its volume (tests/rig_3d.json).
  Figure 3: every fit of the case by its noise and its nearest camera's distance to its wall (tests/fits_summary.json).
  Figure 4 and the rig table: the calibrations against the GUI's (tests/t7_vs_gui.json).
  The sigmas of the delivered f come from its calibration_result.json in the output folder (left out if unreadable).
The page opens in a browser as it is.  --fragment PATH also writes it without its document skeleton, for publishing as
an artifact (the publisher adds its own).
Usage: make_report_page.py [--fragment PATH]
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "page" / "report_template.html"
OUT = HERE / "2026-10-03-report.html"
ROWS = [("GUI calib", "calib", False), ("GUI calib_2", "calib_2", False), ("GUI calib_3", "calib_3", False),
        ("plate_wand (T2, every frame)", "this calibration", True),
        ("plate_wand on calib_3's frames", "fitted on calib_3's frames", True)]
SETS = {"all": "every frame of the table", "c3": "calib_3's 5,563 frames"}


SPECIAL = {"t2_all": "T2, delivered: every frame, k1 fitted, started from the first half's fit",
           "t2_scratch": "T2 from scratch: every frame, k1 fitted",
           "probe20": "every 20th frame"}


def load(rel):
    return json.loads((HERE / rel).read_text(encoding="utf-8"))


def desc(f) -> str:
    """A fit in words: its frames and settings."""
    if f["name"] in SPECIAL:
        return SPECIAL[f["name"]]
    if f["detections"] == "gui_5563":
        parts = ["calib_3's frames"]
    elif f["first"]:
        parts = ["first half"]
    elif f["last"]:
        parts = ["second half"]
    elif f["every"] and f["every"] > 1:
        parts = [f"subset s{f['offset'] or 0} (every {f['every']}th frame)"]
    else:
        parts = ["every frame"]
    parts.append("k1 fitted" if f["fit_k1"] else "k1 held at 0")
    if f["hints"]:
        parts.append("photo hints" if f["hints"] == "photo" else f"hints from {f['hints'][4:]}")
    if f["edge_margin"] is not None:
        parts.append("border-cut balls dropped")
    if f["n_water"]:
        parts.append(f"n_water {f['n_water']}")
    if f["thickness"]:
        parts.append(f"walls {f['thickness']} mm")
    if f["start"]:
        parts.append("started from the first half's fit" if f["start"] == "t3k_h1" else f"started from {f['start']}")
    return ", ".join(parts)


def sigma_f():
    """The delivered fit's sigma of f per camera (px), from the output folder's calibration_result.json."""
    cfg = load("config.json")
    try:
        res = json.loads(Path(cfg["out_dir"], "calibration_result.json").read_text(encoding="utf-8"))
        return [c["sigma"]["f_px"] for c in res["cameras"]]
    except (OSError, KeyError, TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", help="also write the page without its document skeleton to this path")
    args = ap.parse_args()
    fs = load("tests/fits_summary.json")
    for f in fs["fits"]:
        f["desc"] = desc(f)
    t7 = load("tests/t7_vs_gui.json")
    ev = {k: t7["evaluations"][v] for k, v in SETS.items()}
    rows = []
    for key, label, ours in ROWS:
        r = {"label": label, "ours": ours}
        for s in SETS:
            e = ev[s].get(key)
            r["repro_" + s] = e["euclidean_px"]["median"] if e else None
            r["wand_" + s] = e["wand_error_mm"]["abs"]["median"] if e else None
        rows.append(r)
    data = {"fits": fs["fits"], "plan": fs["plan"], "gui_rows": rows, "rig3d": load("tests/rig_3d.json"),
            "gui_rig": t7["rigs"]["GUI calib_3"], "our_rig": t7["rigs"]["plate_wand (T2, every frame)"],
            "t2_sigma_f": sigma_f()}
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count("__FIG_DATA__") == 1 and html.count("<!--BODY-->") == 1
    html = html.replace("__FIG_DATA__", json.dumps(data, separators=(",", ":")))
    head, body = html.split("<!--BODY-->")
    OUT.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   + head + "</head>\n<body>\n" + body + "</body>\n</html>\n", encoding="utf-8")
    if args.fragment:
        Path(args.fragment).write_text(head + body, encoding="utf-8")
    print(f"wrote {OUT.name}: {len(fs['fits'])} fits, {len(rows)} calibrations compared"
          + ("" if data["t2_sigma_f"] else " (no sigma of f: the output folder was not readable)")
          + (f"; fragment {args.fragment}" if args.fragment else ""))


if __name__ == "__main__":
    main()
