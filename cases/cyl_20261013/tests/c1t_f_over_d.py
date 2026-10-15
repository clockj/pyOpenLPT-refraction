"""C1-T, read-out: f, the distance from the axis, and their ratio f / distance for C1 and each part of
c1t_variants.json (halves, quarters, camera pairs).  The wand fixes the ratio (each camera's magnification) but not f
itself (6-8 Oct); this prints both spreads.

    python tests/c1t_f_over_d.py        (after c1t_variants.py)

Writes c1t_f_over_d_out.txt and c1t_f_over_d.json beside this script.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402


def main():
    log = case.Log(os.path.join(HERE, "c1t_f_over_d"))
    with open(os.path.join(HERE, "c1t_variants.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    ref = {c["name"]: c for c in data["S2"]["cameras"]}
    out = {"parts": {}, "spread": {}}
    log("C1-T: f, distance and f / distance per camera (C1 first)")
    rows = {"C1": ref}
    for name, rec in data["variants"].items():
        if "summary" in rec:
            rows[name] = {c["name"]: c for c in rec["summary"]["cameras"]}
    per_cam = {}
    for name, cams in rows.items():
        out["parts"][name] = {}
        parts = []
        for cn, c in sorted(cams.items()):
            ratio = c["f_px"] / c["distance_mm"]
            dev = 100.0 * (ratio / (ref[cn]["f_px"] / ref[cn]["distance_mm"]) - 1.0)
            fdev = 100.0 * (c["f_px"] / ref[cn]["f_px"] - 1.0)
            out["parts"][name][cn] = {"f_px": c["f_px"], "distance_mm": c["distance_mm"], "f_over_d_px_per_mm": ratio,
                                      "f_vs_C1_pct": fdev, "f_over_d_vs_C1_pct": dev}
            per_cam.setdefault(cn, []).append((name, fdev, dev))
            parts.append(f"{cn} f {c['f_px']:.0f} px ({fdev:+.1f} %), {c['distance_mm']:.0f} mm, f/d {ratio:.3f} px/mm "
                         f"({dev:+.2f} %)")
        log(f"  {name}: " + "; ".join(parts))
    for cn, L in sorted(per_cam.items()):
        f = [x[1] for x in L if x[0] != "C1"]
        d = [x[2] for x in L if x[0] != "C1"]
        out["spread"][cn] = {"f_pct": [min(f), max(f)], "f_over_d_pct": [min(d), max(d)]}
        log(f"  {cn}: over the parts f {min(f):+.1f} to {max(f):+.1f} % of C1's; f / distance {min(d):+.2f} to "
            f"{max(d):+.2f} %")
    log.save(out)


if __name__ == "__main__":
    main()
