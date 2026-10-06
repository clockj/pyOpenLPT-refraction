"""E1c (Phase E): how accurate must a prediction be?  The shake capture test, with the C++ shake on real images.

  - The reference: B300's active long tracks at frame 300 (runs/b300_final3), loaded by the C++ (STB.loadTracksAll),
    predicted with STB.predictNext, cut as runConvPhase cuts them (view volume, at least two cameras), and shaken
    together on frame 301's images (Shake.runShake), with the module the GUI runs (build-py/Release, read only).
    This is STB's own frame 301: checked against runs/r301_e1c, a restart that loads the same tracks and runs frame
    301 (stbrun.py r301_e1c --build user --frames 0 301 --load 300 --load-from b300_final3).
  - Moved starts: every prediction moved by delta in a random direction, all shaken together again; per delta the
    share of the reference's particles that end within 0.005 mm of where they ended before (captured), that are
    flagged ghost or repeated (lost), or that end elsewhere, and how far the captured ones shift.  Controls: delta 0
    (the repeat must give the reference to the last bit), and half of the starts moved (how many of the unmoved half
    change through the shared residual images).
  - Other predictors' predictions as starts (E1b's choices on B300): the same measures.

Writes results/e1c_shake_capture.txt, .json and results/e1c_capture.png.
    C:\\Users\\zcloc\\.conda\\envs\\refract\\python.exe tests/stb_improve/e1c_shake_capture.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(REPO, "build-py", "Release")
sys.path.insert(0, HERE)
import predictors as pr  # noqa: E402

RUN = os.path.join(HERE, "runs", "r301_e1c")
TRACKS = os.path.join(HERE, "runs", "b300_final3", "Results", "ConvergeTrack")
LOAD_FRAME, FRAME = 300, 301
DELTAS = (0.0, 1e-6, 1e-5, 1e-4, 0.001, 0.002, 0.005, 0.01, 0.02, 0.032, 0.05, 0.075, 0.1, 0.15, 0.2)
CAPTURE = 0.005
SEED = 20261006
_lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def centers(objs):
    return np.array([[o._pt_center[0], o._pt_center[1], o._pt_center[2]] for o in objs])


def main():
    t0 = time.time()
    sys.path.insert(0, BUILD)
    import pyopenlpt as lpt
    assert os.path.normcase(os.path.dirname(lpt.__file__)) == os.path.normcase(BUILD), lpt.__file__
    log(f"E1c: module {lpt.__file__}")
    bs = lpt.BasicSetting()
    assert bs.readConfig(os.path.join(RUN, "config.txt"))
    cams = bs._cam_list
    stb = lpt.STB(bs, "Tracer", bs._object_config_paths[0])
    stb.loadTracksAll(TRACKS.replace(os.sep, "/") + "/", LOAD_FRAME)
    tracks = stb._long_track_active
    log(f"loaded {len(tracks)} active long tracks at frame {LOAD_FRAME}")

    # predictions, cut as runConvPhase cuts them
    pred_all = [stb.predictNext(tr) for tr in tracks]
    ax = bs._axis_limit
    keep = []
    for k, o in enumerate(pred_all):
        c = o._pt_center
        if ax.check(c[0], c[1], c[2]) and o.isReconstructable(cams):
            keep.append(k)
    keep = np.array(keep)
    objs = [pred_all[k] for k in keep]
    Pstart = centers(objs)
    r2d = np.array([o._r2d_px for o in objs])
    log(f"predictions in range and seen by >= 2 cameras: {len(keep)} (left the volume: {len(tracks) - len(keep)})")

    imgs = []
    for path in bs._image_file_paths:
        io = lpt.ImageIO()
        io.loadImgPath(RUN.replace(os.sep, "/") + "/", path)
        imgs.append(io.loadImg(FRAME))
    tc = lpt.TracerConfig()
    tc.readConfig(bs._object_config_paths[0], bs)

    def shake(P):
        os_ = [lpt.Tracer3D(lpt.Pt3D(float(p[0]), float(p[1]), float(p[2])), float(r)) for p, r in zip(P, r2d)]
        s = lpt.Shake(cams, tc)
        flags, out = s.runShake(os_, imgs)
        X = np.array([[o._pt_center[0], o._pt_center[1], o._pt_center[2]] if o is not None else [np.nan] * 3
                      for o in out])
        ok = np.array([int(f) == 0 for f in flags])
        return X, ok

    ts = time.time()
    X0, ok0 = shake(Pstart)
    log(f"reference shake: {ok0.sum()} kept, {np.sum(~ok0)} flagged ghost or repeated  [{time.time() - ts:.1f} s]")

    # --- validation against the restart run's frame 301
    rest = {}
    for kind in ("LongTrackActive", "LongTrackInactive"):
        tid, frame, xyz = pr._read_csv(os.path.join(RUN, "Results", "ConvergeTrack", f"{kind}_{FRAME}.csv"))
        brk = np.nonzero(np.diff(tid) != 0)[0] + 1
        s = np.concatenate([[0], brk])
        e = np.concatenate([brk, [len(tid)]])
        for a, b in zip(s, e):
            if b - a >= 2 and frame[b - 1] == FRAME and frame[b - 2] == LOAD_FRAME:
                rest[tuple(np.round(xyz[b - 2], 6))] = xyz[b - 1]
    last300 = np.array([[tracks[k]._obj3d_list[-1]._pt_center[i] for i in range(3)] for k in keep])
    keys = [tuple(np.round(p, 6)) for p in last300]
    matched = np.array([kk in rest for kk in keys])
    d = np.array([np.linalg.norm(X0[i] - rest[kk]) if m else np.nan for i, (kk, m) in enumerate(zip(keys, matched))])
    n_ok_match = int(np.sum(matched & ok0))
    flagged_in_restart = int(np.sum(matched & ~ok0))
    maxd = float(np.nanmax(d[matched & ok0])) if n_ok_match else float("nan")
    valid = n_ok_match > 0.95 * ok0.sum() and maxd < 2e-6 and flagged_in_restart == 0
    log(f"validation against r301_e1c: {n_ok_match} of {ok0.sum()} kept particles found in the restart's tracks "
        f"(the rest belonged to tracks of < 7 points that failed the linear-fit test and were not saved); max "
        f"|difference| {maxd:.2g} mm (saved to 1e-6 mm); particles flagged here but kept by the restart: "
        f"{flagged_in_restart} -> {'PASS' if valid else 'FAIL'}")
    res = {"n_tracks": len(tracks), "n_pred": int(len(keep)), "n_kept_ref": int(ok0.sum()),
           "validation": {"n_matched_kept": n_ok_match, "max_diff_mm": maxd, "flagged_but_kept": flagged_in_restart,
                          "pass": bool(valid)}}

    # --- how far the shake moved each kept particle from its start (W's prediction)
    mv = np.linalg.norm(X0[ok0] - Pstart[ok0], axis=1)
    res["shake_move_from_W"] = {q: float(np.percentile(mv, p)) for q, p in
                                (("median", 50), ("p90", 90), ("p99", 99), ("max", 100))}
    log(f"shake's move from W's prediction (kept particles): median {np.median(mv):.4f}, p90 "
        f"{np.percentile(mv, 90):.4f}, p99 {np.percentile(mv, 99):.4f} mm")

    # --- moved starts
    rng = np.random.default_rng(SEED)
    u = rng.normal(size=Pstart.shape)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    res["moved"] = {}
    log(f"\nmoved starts (all {len(keep)} predictions moved by delta in random directions, shaken together; "
        f"delta 0 repeats the reference):")
    log(f"  {'delta mm':>8s} {'captured':>9s} {'flagged':>8s} {'elsewhere':>9s} {'elsewhere: median':>18s} "
        f"{'p90 mm':>7s} {'end - start: median':>20s} {'shift (captured): median':>25s} {'rescued':>8s}")
    for dlt in DELTAS:
        S = Pstart + dlt * u
        X, ok = shake(S)
        dd = np.linalg.norm(X - X0, axis=1)
        cap = ok0 & ok & (dd < CAPTURE)
        fl = ok0 & ~ok
        el = ok0 & ok & (dd >= CAPTURE)
        es = np.linalg.norm(X - S, axis=1)
        r = {"captured": float(cap.sum() / ok0.sum()), "flagged": float(fl.sum() / ok0.sum()),
             "elsewhere": float(el.sum() / ok0.sum()),
             "elsewhere_median_mm": float(np.median(dd[el])) if el.any() else float("nan"),
             "elsewhere_p90_mm": float(np.percentile(dd[el], 90)) if el.any() else float("nan"),
             "elsewhere_end_to_start_median_mm": float(np.median(es[el])) if el.any() else float("nan"),
             "captured_shift_median_mm": float(np.median(dd[cap])) if cap.any() else float("nan"),
             "max_diff_all_kept_mm": float(np.max(dd[ok0 & ok])), "rescued": int(np.sum(~ok0 & ok))}
        res["moved"][f"{dlt:g}"] = r
        log(f"  {dlt:8.6f} {100 * r['captured']:8.2f}% {100 * r['flagged']:7.2f}% {100 * r['elsewhere']:8.2f}% "
            f"{r['elsewhere_median_mm']:18.4f} {r['elsewhere_p90_mm']:7.4f} "
            f"{r['elsewhere_end_to_start_median_mm']:20.4f} {r['captured_shift_median_mm']:25.6f} {r['rescued']:8d}")
    det = res["moved"]["0"]
    res["deterministic"] = det["max_diff_all_kept_mm"] == 0.0 and det["flagged"] == 0.0
    log(f"  the repeat (delta 0) gives the reference to the last bit: {res['deterministic']}")

    # control: move only half of the starts; how many of the unmoved half end elsewhere (through the shared
    # residual images)
    half = rng.random(len(keep)) < 0.5
    res["half_moved"] = {}
    for dlt in (0.01, 0.032, 0.1):
        S = Pstart + dlt * u * half[:, None]
        X, ok = shake(S)
        dd = np.linalg.norm(X - X0, axis=1)
        r = {}
        for nm, grp in (("moved", half), ("unmoved", ~half)):
            g_ = ok0 & grp
            r[nm] = {"elsewhere": float(np.sum(g_ & ok & (dd >= CAPTURE)) / g_.sum()),
                     "flagged": float(np.sum(g_ & ~ok) / g_.sum())}
        res["half_moved"][f"{dlt:g}"] = r
        log(f"  half the starts moved by {dlt:g} mm: elsewhere {100 * r['moved']['elsewhere']:.2f} % of the moved, "
            f"{100 * r['unmoved']['elsewhere']:.2f} % of the unmoved")

    # --- other predictors' predictions as starts
    e1b = json.load(open(os.path.join(HERE, "results", "e1b_replay.json")))["b300"]["chosen"]
    tid, frame, xyz = pr._read_csv(os.path.join(TRACKS, f"LongTrackActive_{LOAD_FRAME}.csv"))
    brk = np.nonzero(np.diff(tid) != 0)[0] + 1
    st = np.concatenate([[0], brk])
    ln = np.diff(np.concatenate([st, [len(tid)]]))
    assert len(st) == len(tracks)
    # a prediction for the point after each track's last point: append a dummy row per track
    P = np.insert(xyz, st[1:], np.nan, axis=0)
    P = np.vstack([P, np.full((1, 3), np.nan)])
    st2 = st + np.arange(len(st))
    g = st2 + ln                      # the dummy row of each track
    Tr = pr.Tracks(P, st2, ln + 1, np.zeros(len(st), np.int8), tid[st], frame[st])
    wpy = pr.wiener(P, g, ln)
    wcpp = centers(pred_all)
    res["w_python_vs_cpp_max_mm"] = float(np.max(np.abs(wpy - wcpp)))
    log(f"\nW recomputed in Python from the CSV against the C++ predictions: max |diff| "
        f"{res['w_python_vs_cpp_max_mm']:.2g} mm")
    alts = {"MD": pr.mean_step(P, g, ln),
            f"LIN-{e1b['LIN']}": pr.poly(P, g, ln, 1, e1b["LIN"]),
            f"QUAD-{e1b['QUAD']}": pr.poly(P, g, ln, 2, e1b["QUAD"])}
    for kind in ("cv", "ca"):
        lam = e1b[f"KF-{kind.upper()}"]
        pk, _ = pr.Kalman(kind, lam).run(Tr)
        alts[f"KF-{kind.upper()}"] = pk[g]
    res["alternatives"] = {}
    log(f"other predictors as starts (the same {len(keep)} particles):")
    log(f"  {'start':>8s} {'|start - W|: median':>20s} {'p99':>7s} {'captured':>9s} {'flagged':>8s} "
        f"{'elsewhere':>9s} {'rescued':>8s}")
    for lab, pa in alts.items():
        S = pa[keep]
        sep = np.linalg.norm(S - Pstart, axis=1)
        X, ok = shake(S)
        dd = np.linalg.norm(X - X0, axis=1)
        cap = ok0 & ok & (dd < CAPTURE)
        r = {"sep_median_mm": float(np.median(sep)), "sep_p99_mm": float(np.percentile(sep, 99)),
             "captured": float(cap.sum() / ok0.sum()), "flagged": float(np.sum(ok0 & ~ok) / ok0.sum()),
             "elsewhere": float(np.sum(ok0 & ok & (dd >= CAPTURE)) / ok0.sum()), "rescued": int(np.sum(~ok0 & ok))}
        res["alternatives"][lab] = r
        log(f"  {lab:>8s} {r['sep_median_mm']:20.4f} {r['sep_p99_mm']:7.4f} {100 * r['captured']:8.2f}% "
            f"{100 * r['flagged']:7.2f}% {100 * r['elsewhere']:8.2f}% {r['rescued']:8d}")

    # --- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dl = np.array(DELTAS)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.semilogx(dl, [100 * res["moved"][f"{x:g}"]["captured"] for x in dl], "o-", color="#0072B2", lw=1.5,
                label="same end point (within 0.005 mm)")
    ax.semilogx(dl, [100 * res["moved"][f"{x:g}"]["flagged"] for x in dl], "s-", color="#D55E00", lw=1.5,
                label="flagged ghost / repeated")
    ax.semilogx(dl, [100 * res["moved"][f"{x:g}"]["elsewhere"] for x in dl], "^-", color="#009E73", lw=1.5,
                label="kept, elsewhere")
    try:
        rep = json.load(open(os.path.join(HERE, "results", "e1b_replay.json")))["b300"]["test"]
        for lab, ls in (("W", "--"),):
            ax.axvline(rep[lab]["median"], color="#000000", lw=0.8, ls=":")
            ax.axvline(rep[lab]["p99"], color="#000000", lw=0.8, ls=ls)
        ax.text(rep["W"]["p99"], 15, " W's p99\n one-step\n error", fontsize=7, va="center")
        ax.text(rep["W"]["median"], 15, " W's\n median", fontsize=7, va="center")
    except Exception:
        pass
    ax.set_xlabel("start moved from the prediction by (mm)")
    ax.set_ylabel("share of the reference's particles (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, color="#eeeeee", lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.set_title(f"Shake capture, frame {FRAME} ({ok0.sum()} particles)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "results", "e1c_capture.png"), dpi=130)
    plt.close(fig)

    log(f"(total {time.time() - t0:.0f} s)")
    with open(os.path.join(HERE, "results", "e1c_shake_capture.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(_lines) + "\n")
    with open(os.path.join(HERE, "results", "e1c_shake_capture.json"), "w") as f:
        json.dump(res, f, indent=1)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
