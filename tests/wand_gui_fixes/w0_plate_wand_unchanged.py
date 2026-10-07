"""W0: plate_wand's answers are unchanged by B3's move: calibrate's nested make_starts (stages 1-3) became the
module-level start_candidates, which the GUI's flat-window precheck calls.

    python w0_plate_wand_unchanged.py

1. The source: the old nested function's statements (with the four lines before it that it used: multi, used, few,
   searches) equal the new function's, compared as syntax trees after renaming the old local names binfo_ and
   starts_ (the new function returns them under the plain names); calibrate keeps no other use of the four names.
2. The answers: plate_wand.calibrate from git HEAD (before_lib/plate_wand.py) and from the working tree on the same
   detections: rig A of the plate campaign (tests/plate_wand/prigs.py; one window, four cameras) and rig B (two
   windows, two cameras behind each), 300 frames at 0.1 px; rig B again with hints on the window normals (5 deg off),
   which runs the hint-held and hint-searched start paths (make_starts(hold=True) and (widen=True)).  Every camera
   and window parameter, the wand length, the frames, the RMS, the kept observations, the weights, the covariance
   and the info (without its timings) must be identical to the last bit.
Writes w0_plate_wand_unchanged_out.txt and .json beside this script.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import ast  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CURVED = REPO / "modules" / "camera_calibration" / "curved_calibration"
NEW = CURVED / "plate_wand.py"
OLD = HERE / "before_lib" / "plate_wand.py"
for _p in (str(REPO / "tests" / "plate_wand"), str(CURVED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load(which):
    """The working tree's plate_wand, or git HEAD's (before_lib) under another name; both import cyl_model,
    plate_model and cylinder_wand from the curved_calibration folder (their fallback when not in a package)."""
    if which == "new":
        import plate_wand
        return plate_wand
    spec = importlib.util.spec_from_file_location("plate_wand_before", OLD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plate_wand_before"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 1. the source
class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        if node.id in self.names:
            node.id = self.names[node.id]
        return node


def source_check(log, check):
    old_tree = ast.parse(OLD.read_text(encoding="utf-8"))
    new_tree = ast.parse(NEW.read_text(encoding="utf-8"))
    cal_old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "calibrate")
    cal_new = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "calibrate")
    sc_new = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "start_candidates")
    ms_old = next(n for n in cal_old.body if isinstance(n, ast.FunctionDef) and n.name == "make_starts")
    moved = ("multi", "used", "few", "searches")
    pre_old = [s for s in cal_old.body if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
               and s.targets[0].id in moved]
    body_old = pre_old + [Rename({"binfo_": "binfo", "starts_": "starts"}).visit(s) for s in ms_old.body[1:]]
    body_new = sc_new.body[2:]                       # without the docstring and `hints = hints or {}`
    dump = lambda stmts: [ast.dump(s) for s in stmts]  # noqa: E731
    same = dump(body_old) == dump(body_new)
    check("the new start_candidates is the old nested make_starts, moved (syntax trees)", same,
          f"{len(body_old)} statements against {len(body_new)}")
    if not same:
        for i, (a, b) in enumerate(zip(dump(body_old), dump(body_new))):
            if a != b:
                log(f"    first difference at statement {i}:\n      old {a[:300]}\n      new {b[:300]}")
                break
    first = ast.dump(sc_new.body[1])
    check("start_candidates begins with `hints = hints or {}` (as calibrate does before)",
          first == ast.dump(ast.parse("hints = hints or {}").body[0]))
    # calibrate keeps no use of the moved names, and its make_starts now calls start_candidates
    ms_new = next(n for n in cal_new.body if isinstance(n, ast.FunctionDef) and n.name == "make_starts")
    inside = {id(n) for n in ast.walk(ms_new)}
    left = sorted({n.id for n in ast.walk(cal_new) if isinstance(n, ast.Name) and n.id in moved
                   and id(n) not in inside})
    calls = [n for n in ast.walk(ms_new) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "start_candidates"]
    check("calibrate keeps no use of multi/used/few/searches; its make_starts calls start_candidates",
          not left and len(calls) == 1, f"left {left}, calls {len(calls)}")
    # the rest of calibrate is the same statement for statement (make_starts and the four lines aside)
    rest_old = [s for s in cal_old.body if s not in pre_old and s is not ms_old]
    rest_new = [s for s in cal_new.body if s is not ms_new]
    i_old = next(i for i, s in enumerate(cal_old.body) if s is ms_old)
    check("the rest of calibrate is unchanged (syntax trees, docstring included)",
          dump(rest_old) == dump(rest_new), f"{len(rest_old)} statements against {len(rest_new)}; "
                                              f"make_starts was statement {i_old}")


# ------------------------------------------------------------------ 2. the answers
SKIP = ("second", "timing")


def plain(x):
    """The solution as plain data, timings left out (keys containing 'second' or 'timing')."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items() if not any(s in str(k) for s in SKIP)}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if hasattr(x, "__dataclass_fields__") or hasattr(x, "__dict__") and not isinstance(x, type):
        d = {k: plain(v) for k, v in vars(x).items() if not any(s in k for s in SKIP)}
        d["__class__"] = type(x).__name__
        return d
    return x


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in b else 'old'}")
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} against {len(b)}")
        for i, (u, v) in enumerate(zip(a, b)):
            diff(u, v, f"{path}[{i}]", out)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_, b_ = np.asarray(a), np.asarray(b)
        if a_.shape != b_.shape or a_.dtype != b_.dtype:
            out.append(f"{path}: array {a_.shape} {a_.dtype} against {b_.shape} {b_.dtype}")
        elif a_.dtype.kind in "fc":
            if a_.tobytes() != b_.tobytes():
                k = np.flatnonzero(a_.ravel() != b_.ravel())
                out.append(f"{path}: {k.size} of {a_.size} values differ, largest "
                           f"{np.nanmax(np.abs(a_.ravel()[k] - b_.ravel()[k])) if k.size else 'NaN pattern'}")
        elif not np.array_equal(a_, b_):
            out.append(f"{path}: arrays differ")
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (math.isnan(a) and math.isnan(b))) or (a == b and math.copysign(1, a) != math.copysign(1, b)):
            out.append(f"{path}: {a!r} against {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:80]!r} against {str(b)[:80]!r}")
    return out


def run(task):
    which, case, uv, L, sizes, wins, cam_win, hints = task
    pw = load(which)
    import plate_model as pm
    rig = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, t, idx, name) for (t, idx, name) in wins], cam_win)
    data = pw.WandData(np.asarray(uv), float(L), [tuple(s) for s in sizes])
    t0 = time.time()
    sol = pw.calibrate(data, rig, hints=hints, verbose=False, log=lambda *a: None)
    sec = time.time() - t0
    rec = {"cams": [plain(c) for c in sol.cams], "windows": [plain(w) for w in sol.rig.windows],
           "cam_win": list(sol.rig.cam_win), "L": sol.L, "rms": sol.rms, "sigma_px": sol.sigma_px,
           "frames": plain(sol.frames), "obs": plain(sol.obs), "weights": sol.weights, "cov": sol.cov,
           "layout": plain(sol.layout), "history": plain(sol.history), "info": plain(sol.info)}
    return which, case, rec, sec


def main():
    lines, checks = [], []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(str(msg))

    def check(label, ok, detail=""):
        checks.append({"check": label, "ok": bool(ok), "detail": detail})
        log(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))

    t_start = time.time()
    log("W0: plate_wand before (git HEAD, before_lib/plate_wand.py) and after B3's move")
    log("1. The source")
    source_check(log, check)

    log("2. The answers")
    import prigs
    cases = {}
    for name, rig_t, seed, hinted in (("rig A", prigs.rig_A(), 7, False), ("rig B", prigs.rig_B(), 1308, False),
                                      ("rig B, hints 5 deg", prigs.rig_B(), 1308, True)):
        w = prigs.make_wand(rig_t, 300, np.random.default_rng(seed), noise_px=0.1)
        hints = None
        if hinted:
            rng = np.random.default_rng(11)
            normals = []
            for c, cam in enumerate(rig_t.cams):
                v = cam.R @ rig_t.win_of(c).n
                e = np.cross(v, rng.normal(size=3))
                e /= np.linalg.norm(e)
                normals.append((math.cos(math.radians(5.0)) * v + math.sin(math.radians(5.0)) * e).tolist())
            hints = {"normals": normals}
        wins = [(w_.t, w_.idx, w_.name) for w_ in rig_t.wins]
        cases[name] = (w.uv, w.L, [tuple(s) for s in rig_t.sizes], wins, list(rig_t.cam_win), hints)
        log(f"  {name}: {rig_t.n_cams} cameras, {len(rig_t.wins)} window(s), {w.uv.shape[0]} frames, seed {seed}"
            + (", hints on the normals" if hinted else ""))
    tasks = [(which, name, *cases[name]) for name in cases for which in ("old", "new")]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        res = pool.map(run, tasks, chunksize=1)
    out = {(which, case): (rec, sec) for which, case, rec, sec in res}
    summary = {}
    for name in cases:
        (a, sa), (b, sb) = out[("old", name)], out[("new", name)]
        d = diff(a, b)
        f_new = [c["f"] for c in b["cams"]]
        check(f"{name}: identical answers (cameras, windows, L, frames, RMS, observations, weights, covariance, "
              f"info)", not d, f"RMS {b['rms']:.6f} px; f {', '.join(f'{v:,.2f}' for v in f_new)}; "
                               f"{sa:.0f} s old, {sb:.0f} s new" + (f"; {len(d)} differences" if d else ""))
        for line in d[:15]:
            log(f"      {line}")
        hint_paths = (b["info"].get("hints") or {}).get("paths") if isinstance(b["info"].get("hints"), dict) else None
        summary[name] = {"differences": d, "rms_px": b["rms"], "f_px": f_new, "seconds_old": sa, "seconds_new": sb,
                         "info_keys": sorted(b["info"]), "hint_paths": plain(hint_paths)}
        if name.endswith("hints 5 deg"):
            hinfo = b["info"].get("hints")
            log(f"      info['hints'] keys: {sorted(hinfo) if isinstance(hinfo, dict) else hinfo}")

    n_fail = sum(not c["ok"] for c in checks)
    log(f"\n{len(checks) - n_fail} of {len(checks)} checks passed; {time.time() - t_start:.0f} s")
    (HERE / "w0_plate_wand_unchanged_out.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "w0_plate_wand_unchanged.json").write_text(
        json.dumps({"checks": checks, "cases": summary}, indent=1, default=str), encoding="utf-8")
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
