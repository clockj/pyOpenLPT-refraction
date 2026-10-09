"""Gather a run's track points with their detections, frame by frame (V0, V1, V2, M0): the data every fit and statistic
of this plan reads.

    python gather.py NAME --run RUN_DIR --lists L1 L2 [...] --noise N1 N2 [...] --cams C1 C2 [...]
                     [--frames F ...] [--every 5] [--pairs] [--depths] [--gate 6] [--procs 8] [--min-len 10]
                     [--correct FIELD.npz]

Per frame (in parallel, one process per frame):
  - the line detector (LineFinder::detect, build p4c, defaults) on every camera's image (line t of each list), told the
    camera's noise; with --correct, the detections are then moved by -c(d) of the field (the correction applied to the
    detections, as a library correction would apply it to the projections);
  - the long tracks (>= MIN_LEN points: the long and exit files) at that frame: the stored 3-D point and projections,
    the projections and Jacobians d(u, v)/dX recomputed by the cameras (central differences, 0.01 mm);
  - per camera, the detection paired with each track point's projection: one to one, nearest pairs first, within
    GATE px (wide on purpose: the analyses apply their own gates);
  - two cameras: the signed offset across the curved epipolar line of the paired detections (cam2's detection's v
    minus cam1's curve at its u; and cam1's against cam2's curve), as T1-a and t1_checks measure it;
  - with --pairs (two cameras): S2's candidate pairs of all detections (curve distance <= 8 px, inside the curve, the
    lines of sight within 1 mm, the midpoint in the view volume), with their signed offsets, and every cam2 detection
    within +-15 px of each cam1 curve (T1-a's histogram);
  - with --depths: the focal depths z_tan, z_sag of every camera at the track points and at the candidate pairs'
    midpoints (shape_model.FocalDepths, displaced camFiles in data/work/focal).
Writes data/gather/NAME/frame_<t>.npz and NAME.json (the settings), and gather_NAME_out.txt here.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

import selfcal as sc

OPTS = {}


def _init(opts):
    sc.one_thread_env()
    OPTS.update(opts)


def _image_path(list_file, t):
    key = ("list", list_file)
    if key not in OPTS:
        with open(list_file, encoding="utf-8") as fh:
            OPTS[key] = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    return OPTS[key][t]


def _focal(c):
    key = ("focal", c)
    if key not in OPTS:
        import shape_model as sm
        OPTS[key] = sm.FocalDepths(sc.lib(), OPTS["cams"][c], work=os.path.join(sc.WORK, "focal"))
    return OPTS[key]


def _tracks():
    if "tracks" not in OPTS:
        OPTS["tracks"] = sc.read_tracks(OPTS["run"])[0]
    return OPTS["tracks"]


def _field():
    if "field_obj" not in OPTS:
        OPTS["field_obj"] = sc.Field.load(OPTS["correct"]) if OPTS.get("correct") else None
    return OPTS["field_obj"]


def work(t):
    t0 = time.time()
    o = OPTS
    n_cam = len(o["cams"])
    P = sc.line_param()
    dets = []
    for c in range(n_cam):
        img = sc.read8(_image_path(o["lists"][c], t))
        d = sc.detect(img, o["noise"][c], P)
        dets.append(d)
    raw_uv = [d[:, :2].copy() for d in dets]
    f = _field()
    if f is not None:
        for c in range(n_cam):
            if len(dets[c]):
                dets[c][:, :2] = dets[c][:, :2] - f.eval_dense(c, dets[c][:, :2])
    t_det = time.time() - t0
    tracks = _tracks()
    ti, ff, X, UV, keys = sc.track_points(tracks, [t], min_len=o["min_len"])
    out = {"t": t, "n_cam": n_cam, "track": ti, "track_kind": np.array([keys[i][0] for i in ti], dtype="U20"),
           "track_id": np.array([keys[i][1] for i in ti], int), "X": X, "uv_stored": UV,
           "track_len": np.array([len(tracks[keys[i]]) for i in ti], int)}
    for c in range(n_cam):
        out[f"det{c}"] = dets[c]
        out[f"det_raw_uv{c}"] = raw_uv[c]
    Pc = np.full((len(X), n_cam, 2), np.nan)
    Jc = np.full((len(X), n_cam, 2, 3), np.nan)
    pair = np.full((len(X), n_cam), -1)
    pdist = np.full((len(X), n_cam), np.nan)
    for c in range(n_cam):
        if len(X):
            Pc[:, c], Jc[:, c] = sc.project_jacobian(o["cams"][c], X)
            pr = sc.pair_one_to_one(dets[c][:, :2], UV[:, c], o["gate"])
            pair[:, c] = pr
            ok = pr >= 0
            pdist[ok, c] = np.hypot(*(dets[c][pr[ok], :2] - UV[ok, c]).T)
    out.update({"P": Pc, "J": Jc, "pair": pair, "pair_dist": pdist})
    if n_cam == 2 and len(X):
        cams, sm_ = sc.load_matcher(o["matcher"])
        e12 = np.full(len(X), np.nan)
        e21 = np.full(len(X), np.nan)
        for k in np.flatnonzero((pair[:, 0] >= 0) & (pair[:, 1] >= 0)):
            a, b = dets[0][pair[k, 0], :2], dets[1][pair[k, 1], :2]
            C, _l = sc.curve(cams, sm_, 0, 1, a)
            if C is not None:
                e12[k] = b[1] - sc.curve_v(C, np.array([b[0]]))[0]
            C, _l = sc.curve(cams, sm_, 1, 0, b)
            if C is not None:
                e21[k] = a[1] - sc.curve_v(C, np.array([a[0]]))[0]
        out["epi12"], out["epi21"] = e12, e21
    t_tr = time.time() - t0 - t_det
    if o["pairs"] and n_cam == 2:
        cams, sm_ = sc.load_matcher(o["matcher"])
        C, Wn = sc.candidate_pairs(cams, sm_, dets[0], dets[1], o["vol"], 8.0, window=15.0)
        out["cand"], out["window"] = C, Wn
    t_pairs = time.time() - t0 - t_det - t_tr
    if o["depths"]:
        for c in range(n_cam):
            fdp = _focal(c)
            out[f"z_track{c}"] = fdp(X) if len(X) else np.zeros((0, 2))
            if "cand" in out:
                out[f"z_cand{c}"] = fdp(out["cand"][:, 2:5]) if len(out["cand"]) else np.zeros((0, 2))
    out["seconds"] = np.array([t_det, t_tr, t_pairs, time.time() - t0])
    np.savez_compressed(os.path.join(o["out_dir"], f"frame_{t:04d}.npz"), **out)
    return t, len(X), [len(d) for d in dets], int(np.sum(np.all(pair >= 0, axis=1))) if len(X) else 0, \
        float(time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--run", required=True)
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--noise", type=float, nargs="+", required=True)
    ap.add_argument("--cams", nargs="+", required=True)
    ap.add_argument("--otf", default=os.path.join(sc.CASE_DATA, "T1_otf_shape.txt"),
                    help="any OTF table of as many cameras (the matcher config needs one)")
    ap.add_argument("--frames", type=int, nargs="+", default=None)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--depths", action="store_true")
    ap.add_argument("--gate", type=float, default=6.0)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--correct", default=None, help="a Field (.npz) whose dense correction moves the detections")
    args = ap.parse_args()
    if not (len(args.lists) == len(args.noise) == len(args.cams)):
        raise SystemExit("--lists, --noise and --cams need one entry per camera")
    if args.procs > sc.THREADS_MAX:
        raise SystemExit(f"--procs {args.procs}: at most {sc.THREADS_MAX}")
    log = sc.Log(f"gather_{args.name}")
    out_dir = os.path.join(sc.DATA, "gather", args.name)
    os.makedirs(out_dir, exist_ok=True)
    _tr, last = sc.read_tracks(args.run)
    frames = args.frames or list(range(0, (args.last if args.last is not None else last) + 1, args.every))
    matcher = sc.write_matcher_config([os.path.abspath(p) for p in args.cams],
                                      os.path.join(sc.WORK, f"matcher_{args.name}"), os.path.abspath(args.otf))
    opts = {"run": os.path.abspath(args.run), "lists": [os.path.abspath(p) for p in args.lists],
            "noise": args.noise, "cams": [os.path.abspath(p) for p in args.cams], "matcher": matcher,
            "pairs": args.pairs, "depths": args.depths, "gate": args.gate, "min_len": args.min_len,
            "out_dir": out_dir, "vol": sc.view_volume(), "correct": os.path.abspath(args.correct) if args.correct
            else None}
    sc.save_json(os.path.join(out_dir, f"{args.name}.json"), {**opts, "frames": frames, "build": sc.build_record()})
    log(f"gather {args.name}: run {args.run} (last frame {last}), {len(frames)} frames {frames[0]}..{frames[-1]}, "
        f"{len(args.cams)} cameras, noise {args.noise}, gate {args.gate} px, min length {args.min_len}, pairs "
        f"{args.pairs}, depths {args.depths}, correction {args.correct}; {args.procs} processes")
    t0 = time.time()
    with Pool(args.procs, initializer=_init, initargs=(opts,)) as pool:
        for t, n, nd, nb, s in pool.imap_unordered(work, frames):
            log(f"  frame {t:4d}: {n} long-track points, detections {nd}, paired in every camera {nb} ({s:.0f} s)")
    log(f"  done in {time.time() - t0:.0f} s")
    log.save({"name": args.name, "frames": frames, "seconds": time.time() - t0})


if __name__ == "__main__":
    main()
