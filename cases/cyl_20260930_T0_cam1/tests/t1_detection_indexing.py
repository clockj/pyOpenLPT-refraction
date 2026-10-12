"""T1: detection and indexing on the 20 real images.

Checks (plan, section 5):
  * every complete dot is found once, and nothing that is not a dot is accepted;
  * the indices are consistent across all 20 planes: the tracking cross-check, no duplicates, small lattice
    residuals;
  * the stationary column is i = 0 in every plane, and the extra column on the right appears with the next index;
  * the centre estimators agree: RMS difference < 0.1 px.

Writes t1_out.txt, t1_dots.json and t1_overlay_zXX.png beside this script.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import dots as D                              # noqa: E402
import grid_index as G                        # noqa: E402
import viz                                    # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def main():
    cfg = case.load_config()
    det = cfg["detect"]
    W, H = cfg["image"]["width"], cfg["image"]["height"]
    per_plane = []
    for (z, path), lab in zip(case.plane_files(cfg), case.plane_labels(cfg)):
        img = case.read_image(path)
        assert img.shape == (H, W), (path, img.shape)
        r = D.detect(img, det)
        dots = r["dots"]
        P = np.array([[d["u"], d["v"]] if d["ok"] else [d["u_rough"], d["v_rough"]] for d in dots], float)
        ok = np.array([d["ok"] for d in dots], bool)
        per_plane.append({"z": z, "label": lab, "path": path, "img": img, "dots": dots, "P": P, "ok": ok,
                          "seg": r["seg_level"], "a_ref": r["a_ref"]})

    idx = G.index_planes(per_plane, cfg["frame"]["origin_row_v"], cfg["index"]["track_gate_px"],
                         cfg["index"]["lattice_tol_frac"])
    ch = idx["checks"]

    say("T1 detection and indexing, 20 real images")
    say(f"{'z':>6} {'ok':>3} {'bord':>4} {'oth':>3} {'i range':>8} {'j range':>8} {'miss':>4} {'dup':>3} "
        f"{'trk':>7} {'latres':>6} {'u(i=0)':>7} {'d_iso rms':>9} {'d_sat rms':>9} {'d_iso mean':>15} "
        f"{'a,b px':>11} {'edge rms':>8}")
    all_fail = []
    grad_stats = []
    col0_u, col8_first = [], None
    out_json = []
    for k, pl in enumerate(per_plane):
        dots, ok, ij = pl["dots"], pl["ok"], idx["ij"][k]
        n_ok = int(ok.sum())
        n_border = sum(d["kind"] == "border" for d in dots)
        n_other = len(dots) - n_ok - n_border
        ij_ok = ij[ok]
        irng = (int(ij_ok[:, 0].min()), int(ij_ok[:, 0].max()))
        jrng = (int(ij_ok[:, 1].min()), int(ij_ok[:, 1].max()))
        # expected complete dots from the lattice and the median dot size
        a_med = float(np.nanmedian([d["a"] for d in dots if d["ok"]]))
        b_med = float(np.nanmedian([d["b"] for d in dots if d["ok"]]))
        p0, A = idx["lattice"][k]
        exp = set(G.expected_complete(p0, A, ((irng[0] - 2, irng[1] + 2), (-jrng[1] - 2, -jrng[0] + 2)),
                                      W, H, a_med + 6.0, b_med + 6.0, 0.0))
        got = set(tuple(v) for v in ij_ok)
        missing = sorted(exp - got)
        extra = sorted(got - exp)
        # estimator differences (ok dots)
        du_iso = np.array([d["u"] - d["u_iso"] for d in dots if d["ok"]])
        dv_iso = np.array([d["v"] - d["v_iso"] for d in dots if d["ok"]])
        du_sat = np.array([d["u"] - d["u_sat"] for d in dots if d["ok"]])
        dv_sat = np.array([d["v"] - d["v_sat"] for d in dots if d["ok"]])
        du_grd = np.array([d["u"] - d["u_grad"] for d in dots if d["ok"]])
        dv_grd = np.array([d["v"] - d["v_grad"] for d in dots if d["ok"]])
        r_iso = float(np.sqrt(np.nanmean(du_iso ** 2 + dv_iso ** 2)))
        r_sat = float(np.sqrt(np.nanmean(du_sat ** 2 + dv_sat ** 2)))
        r_grd = float(np.sqrt(np.nanmean(du_grd ** 2 + dv_grd ** 2)))
        grad_stats.append((pl["z"], r_grd, float(np.nanmean(du_grd)), float(np.nanmean(dv_grd)),
                           int(np.sum(~np.isfinite(du_grd)))))
        e_rms = float(np.nanmedian([d["edge_rms"] for d in dots if d["ok"]]))
        u0 = float(np.mean(pl["P"][ok][ij_ok[:, 0] == 0, 0])) if np.any(ij_ok[:, 0] == 0) else float("nan")
        col0_u.append(u0)
        if col8_first is None and np.any(ij_ok[:, 0] == 8):
            col8_first = pl["z"]
        trk = f"{ch['track_mismatch'][k]}/{ch['track_pairs'][k]}" if k else "-"
        say(f"{pl['z']:6.2f} {n_ok:3d} {n_border:4d} {n_other:3d} {str(irng):>8} {str(jrng):>8} {len(missing):4d} "
            f"{ch['duplicates'][k]:3d} {trk:>7} {ch['lattice_max_res_px'][k]:6.2f} {u0:7.2f} {r_iso:9.3f} "
            f"{r_sat:9.3f} ({np.nanmean(du_iso):+.2f},{np.nanmean(dv_iso):+.2f}) "
            f"{a_med:5.1f},{b_med:5.1f} {e_rms:8.3f}")
        if missing:
            say(f"      missing (expected complete, not found ok): {missing}")
        if extra:
            say(f"      found ok beyond the conservative expectation: {extra}")
        others = [(d["kind"], round(d["u_rough"]), round(d["v_rough"]), d["area"]) for d in dots
                  if d["kind"] not in ("dot", "border")]
        if others:
            say(f"      rejected blobs: {others}")
        fails = []
        if missing:
            fails.append("missing")
        if ch["duplicates"][k]:
            fails.append("duplicates")
        if k and ch["track_mismatch"][k]:
            fails.append("tracking")
        if ch["lattice_bad"][k]:
            fails.append("lattice")
        if not r_grd < 0.1:
            fails.append("estimators")
        all_fail.append(fails)
        vis = viz.draw_dots(pl["img"], dots, [tuple(v) for v in ij],
                            title=f"{os.path.basename(pl['path'])}: z = {pl['z']:.2f} mm")
        cv2.imwrite(os.path.join(HERE, f"t1_overlay_z{int(pl['label']):02d}.png"), vis)
        out_json.append({"z": pl["z"], "label": pl["label"], "image": pl["path"],
                         "dots": [{**{k2: (v2 if not isinstance(v2, (np.floating, np.integer)) else v2.item())
                                      for k2, v2 in d.items()}, "i": int(ij[n][0]), "j": int(ij[n][1])}
                                  for n, d in enumerate(dots)]})

    say("")
    say("primary (half-height edges) minus check (steepest-descent edges), per plane: z, rms, mean du, mean dv, "
        "failed fits")
    for z, rg, mu, mv, nf in grad_stats:
        say(f"  z={z:6.2f}  rms {rg:.3f}  mean ({mu:+.3f}, {mv:+.3f})  failed {nf}")
    say(f"stationary column i = 0: u from {np.nanmin(col0_u):.2f} to {np.nanmax(col0_u):.2f} px over the 20 planes")
    say(f"column i = 8 first complete at z = {col8_first} mm")
    say(f"origin dot (plane z = 0): u, v = {per_plane[0]['P'][idx['seed'][0]]}")
    n_total = sum(int(p["ok"].sum()) for p in per_plane)
    say(f"usable dots in total: {n_total}")
    bad = {per_plane[k]["z"]: f for k, f in enumerate(all_fail) if f}
    say("RESULT: " + ("PASS" if not bad else f"FAIL {bad}"))
    with open(os.path.join(HERE, "t1_dots.json"), "w") as f:
        json.dump(out_json, f, indent=1, default=float)
    with open(os.path.join(HERE, "t1_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
