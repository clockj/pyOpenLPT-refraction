"""L3 (b): where the 3-camera library run's positions differ from the re-rendered set's.

    python l3_positions.py [N]          # N cameras (default 3)

The library run (runs/l3_lib_r<N>) moves the model by V2's field as read from the correction files (8 px nodes,
bilinear); the re-rendered set's images were moved by the field's dense map itself. The two agree where the dense map
is smooth, and differ where its fill beyond the data (the nearest covered pixel's value) has steps, which the 8 px
nodes smooth (up to 0.71 px for cal4_r3's camera 3, convert_fields_out.txt). Here every reconstructed point is matched
to the truth as the study's eval_study.evaluate matches it (0.2 mm, frames >= 4, particles seen by >= 2 cameras), and
the position errors are split by how far the correction file departs from the dense map at the particle's images: in
every camera within 0.05 px (where the two runs move the images alike), or more in some camera. Writes
l3_positions_out.txt and .json here.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

import vcf_common as vc

STUDY = os.path.join(vc.REPO, "tests", "cylinder_stb_study")
sys.path.insert(0, STUDY)
import eval_study  # noqa: E402
from eval_truth import read_tracks  # noqa: E402

CAL4 = os.path.join(vc.SELFCAL, "data", "cal4")


def matched_errors(run_dir, set_name, n, r_match=0.2, first=4):
    X, _layer, n_seen = eval_study.truth(CAL4, set_name, list(range(n)))
    tracks, _kinds = read_tracks(run_dir)
    by_frame = {}
    for key, pts in tracks.items():
        for (t, x, y, z) in pts:
            by_frame.setdefault(t, []).append(np.array([x, y, z]))
    rows = []
    for t in sorted(by_frame):
        if t < first or t >= X.shape[0]:
            continue
        ids = np.flatnonzero(n_seen[t] >= 2)
        if not len(ids):
            continue
        tree = cKDTree(X[t, ids])
        cand = []
        for k, Xr in enumerate(by_frame[t]):
            d, j = tree.query(Xr)
            if d <= r_match:
                cand.append((d, k, j))
        cand.sort()
        used = set()
        for d, k, j in cand:
            if j in used:
                continue
            used.add(j)
            e = by_frame[t][k] - X[t, ids[j]]
            rows.append((t, int(ids[j]), e[0], e[1], e[2]))
    return np.array(rows)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    log = vc.Log("l3_positions" if n == 3 else f"l3_positions_r{n}")
    sets = vc.load_json(os.path.join(CAL4, "sets.json"))
    variant = sets["Gcal4_n8"]["variant"]
    dense = vc.selfcal_dense(f"cal4_r{n}")
    uv = [np.load(os.path.join(CAL4, "arrays", f"uv_{variant}_c{k}.npy"), mmap_mode="r") for k in range(n)]
    corr = [vc.Correction.read(os.path.join(vc.DATA, "fields", f"cal4_r{n}_cam{k + 1}.txt")) for k in range(n)]
    out = {}
    rows = {}
    runs = {"library": (os.path.join(vc.RUNS, f"l3_lib_r{n}"), "Gcal4_n8"),
            "re-rendered": (os.path.join(vc.SELFCAL, "runs", f"Gcal4c{n}_n8_r{n}_tol1.0"), f"Gcal4c{n}_n8")}
    for name, (rd, set_name) in runs.items():
        R = matched_errors(rd, set_name, n)
        rows[name] = R
        t, pid = R[:, 0].astype(int), R[:, 1].astype(int)
        dev = np.zeros(len(R))
        for k in range(n):
            p = np.asarray(uv[k][t, pid], float)
            ok = np.isfinite(p).all(axis=1)
            d = np.zeros(len(R))
            if ok.any():
                d[ok] = np.hypot(*(corr[k].at(p[ok]) - vc.dense_at(dense[k], p[ok])).T)
            dev = np.maximum(dev, d)
        e = 1e3 * R[:, 2:5]
        res = {}
        for lab, m in (("all", np.ones(len(R), bool)), ("file = dense map (<= 0.05 px)", dev <= 0.05),
                       ("file != dense map (> 0.05 px)", dev > 0.05)):
            rms = np.sqrt(np.mean(e[m] ** 2, axis=0)) if m.any() else np.full(3, np.nan)
            res[lab] = {"points": int(m.sum()), "rms_um": rms.tolist()}
        out[name] = res
        log(f"{name} ({rd}): " + "; ".join(f"{lab}: {v['points']} points, RMS x/y/z "
                                           + "/".join(f"{x:.1f}" for x in v["rms_um"]) + " um"
                                           for lab, v in res.items()))
    # paired: the same particle at the same frame, matched in both runs
    ka = {(int(t), int(p)): 1e3 * np.array(e) for t, p, *e in rows["library"]}
    kb = {(int(t), int(p)): 1e3 * np.array(e) for t, p, *e in rows["re-rendered"]}
    common = sorted(set(ka) & set(kb))
    ea = np.array([ka[k] for k in common])
    eb = np.array([kb[k] for k in common])
    only_a = np.array([ka[k] for k in set(ka) - set(kb)]).reshape(-1, 3)
    only_b = np.array([kb[k] for k in set(kb) - set(ka)]).reshape(-1, 3)
    rms = lambda e: np.sqrt(np.mean(e ** 2, axis=0)) if len(e) else np.full(3, np.nan)  # noqa: E731
    d = ea - eb
    out["paired"] = {"common": len(common), "library_rms_um": rms(ea).tolist(), "rerendered_rms_um": rms(eb).tolist(),
                     "mean_diff_um": d.mean(axis=0).tolist(), "rms_diff_um": rms(d).tolist(),
                     "only_library": {"points": len(only_a), "rms_um": rms(only_a).tolist()},
                     "only_rerendered": {"points": len(only_b), "rms_um": rms(only_b).tolist()}}
    p = out["paired"]
    log(f"paired (the same particle and frame in both): {p['common']} points; RMS x/y/z library "
        + "/".join(f"{x:.1f}" for x in p["library_rms_um"]) + ", re-rendered "
        + "/".join(f"{x:.1f}" for x in p["rerendered_rms_um"]) + " um; the difference library - re-rendered: mean "
        + "/".join(f"{x:+.2f}" for x in p["mean_diff_um"]) + ", RMS " + "/".join(f"{x:.1f}" for x in p["rms_diff_um"])
        + " um")
    log(f"  only in the library run: {p['only_library']['points']} points, RMS "
        + "/".join(f"{x:.1f}" for x in p["only_library"]["rms_um"]) + " um; only in the re-rendered run: "
        f"{p['only_rerendered']['points']} points, RMS " + "/".join(f"{x:.1f}" for x in p["only_rerendered"]["rms_um"])
        + " um")
    log.save(out)


if __name__ == "__main__":
    main()
