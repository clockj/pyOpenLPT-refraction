"""Option (b) of 2026-10-07-plan.md: the results pages of the real tube runs 20261006/T0 and T2 on X:, rendered again
with the error charts.  Only calibration_report.html changes; the camFiles and results stay as they are.

    python rerender_x_pages.py            the comparison only
    python rerender_x_pages.py --write    the comparison, then each page written if it holds

Before a page is written, the report built now from its calibration_result.json and calibration_data.json must
equal the report embedded in the page on X: in every key but the new error_pdf, so that the page shows what it
showed, plus the new section.  Otherwise nothing is written for that run.  After writing, the page is read back.
Appends to rerender_x_pages_out.txt.
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from modules.camera_calibration.results_page import page as rp  # noqa: E402

RUNS = {"T0": Path(r"X:\Shijie Zhong\Refraction\Processed\20261006\T0\Results"),
        "T2": Path(r"X:\Shijie Zhong\Refraction\Processed\20261006\T2\Results")}
LINES = []


def say(msg=""):
    print(msg, flush=True)
    LINES.append(str(msg))


def embedded(path):
    txt = path.read_text(encoding="utf-8")
    m = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', txt, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def differences(a, b, at="", out=None):
    """The places where two JSON values differ (at most 10)."""
    out = [] if out is None else out
    if len(out) >= 10:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{at}/{k}: only in the {'new' if k in a else 'old'} report")
            else:
                differences(a[k], b[k], f"{at}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            differences(x, y, f"{at}[{i}]", out)
    elif a != b:
        out.append(f"{at}: {str(b)[:80]} -> {str(a)[:80]}")
    return out


def main():
    write = "--write" in sys.argv[1:]
    say(f"\n== {time.strftime('%Y-%m-%d %H:%M')}: rerender_x_pages.py{' --write' if write else ''}")
    ok_all = True
    for name, folder in RUNS.items():
        page = folder / "calibration_report.html"
        old = embedded(page)
        res = json.load(open(folder / "calibration_result.json", encoding="utf-8"))
        data = json.load(open(folder / "calibration_data.json", encoding="utf-8"))
        report = rp.report_from_cylinder(res, data, old.get("folder"))
        new = json.loads(json.dumps(rp._clean(report), allow_nan=False))
        diffs = differences({k: v for k, v in new.items() if k != "error_pdf"}, old)
        say(f"{name}: {page}")
        say(f"  the page now: {page.stat().st_size:,} bytes, written {time.strftime('%Y-%m-%d %H:%M', time.localtime(page.stat().st_mtime))}; "
            f"its report has error_pdf: {'error_pdf' in old}")
        say(f"  the report built now equals the page's in every key but error_pdf: {not diffs}"
            + ("" if not diffs else "; differences: " + "; ".join(diffs)))
        if diffs:
            ok_all = False
            say("  not written")
            continue
        if write:
            rp.write_page(report, str(page))
            back = embedded(page)
            same = back == new
            ok_all &= same
            say(f"  written: {page.stat().st_size:,} bytes; read back, its report is the one built: {same}")
    with open(HERE / "rerender_x_pages_out.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
