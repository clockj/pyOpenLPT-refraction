"""F4: compare the synthetic tests' outputs before and after the fixes (run_before/ and run_after/, f4_synthetic.py).

    python f4_compare.py [A B]      (default: before after; F7, 7 Oct: "before flip4" and "after flip4" too)

A and B name the run folders run_A/ and run_B/; the output names carry them unless they are the default pair
(f4_compare_<A>_<B>_out.txt and .json).  "F-a" counts the two-state flip rule, "F-a3" the four-jumps rule (F7).

Every fit a test records (a dict with "rounds") is paired with the same fit of the other run, by its place in the
test's JSON.  Per test:
  - fits that converged before (fewer than 20 rounds): identical after (every number the test records, times
    excepted, the same to 1e-12 relative), or else listed with their largest difference;
  - fits that ran to the round limit before: their rounds after, whether F-a (weights flip) or F-a2 (repeated active
    set) stopped them, the observations rejected, and how far their z-scores (error over the bound, where the test
    records them) moved;
  - the test's own verdicts ("pass" and "criteria" entries), before and after.
Writes f4_compare_out.txt and f4_compare.json beside this script.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = ["t11a_noise_sweep", "t11c_unequal_noise", "t11d_heavy_tails", "t11e_systematic", "t12a_weights_unequal",
         "t12b_weights_equal", "t12c_two_cameras", "t12d_scratch_t5", "t12e_scratch_5px"]
SKIP = {"seconds", "time", "t", "cycle", "flipped", "elapsed"}
LIMIT = 20


def fits(o, path=""):
    """(path, record) of every dict with an integer "rounds"."""
    out = []
    if isinstance(o, dict):
        if isinstance(o.get("rounds"), int):
            out.append((path, o))
        for k, v in o.items():
            out += fits(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += fits(v, f"{path}[{i}]")
    return out


def numbers(o, path=""):
    """(path, value) of every number below o, times and the fixes' own fields left out."""
    out = []
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return out
    if isinstance(o, (int, float)):
        return [(path, float(o))]
    if isinstance(o, dict):
        for k, v in o.items():
            if k not in SKIP:
                out += numbers(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += numbers(v, f"{path}[{i}]")
    return out


def verdicts(o, path=""):
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("pass", "criteria") or k.startswith("pass_"):
                out.append((f"{path}/{k}", v))
            else:
                out += verdicts(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out += verdicts(v, f"{path}[{i}]")
    return out


def zs(rec):
    """The record's z-scores against the Cramer-Rao bound (T11's measure), if any."""
    z = rec.get("z_crb")
    return [float(x) for x in z] if isinstance(z, list) else None


def main():
    lines, summary = [], {}
    a_name, b_name = (sys.argv[1:3] if len(sys.argv) >= 3 else ["before", "after"])
    stem = "f4_compare" if (a_name, b_name) == ("before", "after") else f"f4_compare_{a_name}_{b_name}"

    def say(s=""):
        print(s, flush=True)
        lines.append(s)
    say(f"F4: the synthetic noise and weight tests, run_{a_name} against run_{b_name}"
        + (" (before: git HEAD before the fixes)" if a_name == "before" else ""))
    for t in TESTS:
        pb, pa = HERE / f"run_{a_name}" / f"{t}.json", HERE / f"run_{b_name}" / f"{t}.json"
        if not (pb.exists() and pa.exists()):
            say(f"\n{t}: missing ({'before' if not pb.exists() else 'after'})")
            continue
        B, A = json.loads(pb.read_text(encoding="utf-8")), json.loads(pa.read_text(encoding="utf-8"))
        fb, fa = dict(fits(B)), dict(fits(A))
        conv_same, conv_diff, looped = 0, [], []
        fixes = {"F-a": 0, "F-a2": 0, "F-a3": 0}
        for path, rb in fb.items():
            ra = fa.get(path)
            if ra is None:
                conv_diff.append((path, "missing after"))
                continue
            if ra.get("flipped"):
                fixes["F-a" if ra["flipped"].get("kind", "alternates") == "alternates" else "F-a3"] += 1
            if ra.get("cycle"):
                fixes["F-a2"] += 1
            if rb["rounds"] < LIMIT:
                nb, na = dict(numbers(rb)), dict(numbers(ra))
                worst, where = 0.0, ""
                for k in set(nb) | set(na):
                    x, y = nb.get(k), na.get(k)
                    if x is None or y is None:
                        worst, where = math.inf, k
                        break
                    if x != y and not (math.isnan(x) and math.isnan(y)):
                        d = abs(x - y) / max(abs(x), abs(y), 1e-300)
                        if d > worst:
                            worst, where = d, k
                if worst <= 1e-12:
                    conv_same += 1
                else:
                    conv_diff.append((path, f"{where}: {worst:.1e} relative; rounds {rb['rounds']} -> {ra['rounds']}"))
            else:
                zb, za = zs(rb), zs(ra)
                dz = (math.sqrt(sum((x - y) ** 2 for x, y in zip(zb, za)) / len(zb)) if zb and za else None)
                looped.append({"path": path, "rounds_after": ra["rounds"], "rejected": [rb.get("rejected"), ra.get("rejected")],
                               "F-a": (None if not ra.get("flipped") else
                                       "F-a" if ra["flipped"].get("kind", "alternates") == "alternates" else "F-a3"),
                               "F-a2": ra.get("cycle"),
                               "z_rms": [math.sqrt(sum(x * x for x in zb) / len(zb)) if zb else None,
                                         math.sqrt(sum(x * x for x in za) / len(za)) if za else None],
                               "dz_rms": dz})
        vb, va = verdicts(B), verdicts(A)
        say(f"\n{t}: {len(fb)} fits; converged before: {conv_same} identical after, {len(conv_diff)} different; "
            f"at the round limit before: {len(looped)}; the fixes acted after: F-a {fixes['F-a']}, F-a2 {fixes['F-a2']}"
            f", F-a3 {fixes['F-a3']}")
        for p, why in conv_diff[:12]:
            say(f"    differs: {p}: {why}")
        if len(conv_diff) > 12:
            say(f"    ... and {len(conv_diff) - 12} more")
        for r in looped:
            say(f"    looped before: {r['path']}: now {r['rounds_after']} rounds"
                + (f" ({r['F-a']})" if r["F-a"] else "") + (f" (F-a2: {r['F-a2']})" if r["F-a2"] else "")
                + f"; rejected {r['rejected'][0]} -> {r['rejected'][1]}"
                + (f"; z RMS {r['z_rms'][0]:.3f} -> {r['z_rms'][1]:.3f}, moved {r['dz_rms']:.3f} RMS"
                   if r["dz_rms"] is not None else ""))
        if looped:
            zb = [r["z_rms"][0] for r in looped if r["z_rms"][0] is not None]
            za = [r["z_rms"][1] for r in looped if r["z_rms"][1] is not None]
            say(f"    the {len(looped)} that looped: rounds now {min(r['rounds_after'] for r in looped)}-"
                f"{max(r['rounds_after'] for r in looped)} (were 20)"
                + (f"; mean z RMS {sum(zb) / len(zb):.3f} before, {sum(za) / len(za):.3f} after" if zb and za else ""))
        # the test's own summaries (every number outside the fit records): the largest changes
        inside = tuple(fb)
        sb = {k: v for k, v in numbers(B) if not k.startswith(inside)}
        sa = {k: v for k, v in numbers(A) if not k.startswith(inside)}
        moved = []
        for k in sorted(set(sb) & set(sa)):
            x, y = sb[k], sa[k]
            if x != y and not (math.isnan(x) and math.isnan(y)):
                moved.append((abs(x - y) / max(abs(x), abs(y), 1e-300), k, x, y))
        moved.sort(reverse=True)
        say(f"    the test's summary numbers: {len(sb)}; changed {len(moved)}"
            + ("" if not moved else "; largest: " + "; ".join(f"{k} {x:.4g} -> {y:.4g}" for _d, k, x, y in moved[:4])))
        same_v = sum(1 for (p, x), (q, y) in zip(vb, va) if p == q and x == y)
        say(f"    the test's verdicts: {same_v} of {len(vb)} the same after"
            + ("" if len(vb) == len(va) else f" (after has {len(va)})"))
        for (p, x), (_q, y) in zip(vb, va):
            if x != y:
                say(f"    verdict changed {p}: before {json.dumps(x)}  after {json.dumps(y)}")
        summary[t] = {"fits": len(fb), "converged_identical": conv_same, "converged_different": conv_diff,
                      "looped_before": looped, "fixes_after": fixes,
                      "verdicts_before": vb, "verdicts_after": va}
    (HERE / f"{stem}_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / f"{stem}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
