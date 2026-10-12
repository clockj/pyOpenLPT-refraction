"""P2: each new option of the GUI's flat-window calibration on a synthetic rig (plan P2).

    python p2_options.py

Rig B' (tests/plate_wand prigs.rig_B with its second window 10 mm thick instead of 12.7; 200 frames, 0.1 px). Each case
sets the option in the view (offscreen), takes the Job the view builds, runs it as the GUI's process does
(plate_wand_gui.run) and, where the record matters, writes the outputs (plate_wand_gui.write_outputs, data/p2_*):
- A  per-window thickness (D2): window 1 at 10 mm in the Windows table: f within 0.5 % of the truth; and A' with both
     windows at 12.7 mm (one value for every window, as before) fits worse (a larger RMS);
- B  start from a saved fit (F2): A's folder (calibration_result.json) and its camFiles alone give A's answer; the
     panel's thickness replaces the saved one (D3);
- C  constraints (F3): a channel wide enough to hold every true end and an aim at the true depth: the answer as A's,
     the constraints met, recorded in calibration_result.json and on the page;
- D  lens terms (F4): "Radial k1+k2", the principal point and tangential: fit_lens recorded, f within 1 %;
- E  hints (F5): "Windows face the cameras", and from A's camFiles: recorded, f within 0.5 %;
- F  refusals: a constraint on a window or camera that is not there, a saved fit grouping the cameras otherwise.
Writes p2_options_out.txt and .json here.
"""
import pgcommon as P  # noqa: I001

import json  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402

log = P.Log("p2_options")


def f_err(sol, rig):
    return max(abs(c.f / t.f - 1) for c, t in zip(sol.cams, rig.cams))


def main():
    P.qt_app()
    for d in ("p2_A", "p2_A_camfiles", "p2_C", "p2_D", "p2_E"):
        shutil.rmtree(P.DATA / d, ignore_errors=True)
    dlg = P.Dialogs(folder=str(P.DATA / "p2_unused"))
    import prigs
    from PySide6.QtWidgets import QTableWidgetItem
    from modules.camera_calibration.curved_calibration import plate_wand as pw
    from modules.camera_calibration.curved_calibration import plate_wand_gui as pg
    from modules.camera_calibration.curved_calibration import plate_wand_io as pio
    rig = prigs.rig_B()
    rig.wins[1] = prigs.wall([1, 0, 0], 0.0, t=(10.0,), name="wall X")
    w = prigs.make_wand(rig, 200, np.random.default_rng(2610), noise_px=0.1)
    wp = {}
    for i in range(w.uv.shape[0]):
        fr = {c: [[*w.uv[i, c, 0], 20.0, 0.9, "Filtered_Small", 0], [*w.uv[i, c, 1], 30.0, 0.9, "Filtered_Large", 1]]
              for c in range(rig.n_cams) if np.all(np.isfinite(w.uv[i, c]))}
        if len(fr) >= 2:
            wp[i] = fr
    log(f"P2: rig B' (window 1 10 mm), {len(wp)} frames")
    out = {}

    def view_for(thick1=10.0):
        v = P.make_view(rig, wp, [0, 0, 1, 1], thickness=12.7)
        v.refr_window_table.item(1, 1).setText(f"{thick1:g}")
        return v

    def run(view, folder=None):
        got = P.catch_job(view)
        if got is None:
            raise RuntimeError(f"the view refused: {dlg.messages[-1] if dlg.messages else ''}")
        job = got[1]
        sol = pg.run(job)
        res = pg.write_outputs(sol, job, str(P.DATA / folder)) if folder else None
        return job, sol, res

    # A: per-window thickness
    jA, sA, rA = run(view_for(10.0), "p2_A")
    jW, sW, _ = run(view_for(12.7))
    log.check("A: window 1 at 10 mm in the Windows table: f within 0.5 % of the truth; both at 12.7 mm fits worse",
              jA.windows[1]["thickness_mm"] == [10.0] and f_err(sA, rig) < 0.005 and sW.rms > sA.rms,
              f"f error {100 * f_err(sA, rig):.3f} % (12.7 for both: {100 * f_err(sW, rig):.3f} %); RMS {sA.rms:.4f} px "
              f"against {sW.rms:.4f} px; active {int(sA.obs.active.sum())} of {len(sA.obs)}")
    out["A"] = {"rms": sA.rms, "f_err": f_err(sA, rig), "rms_wrong": sW.rms, "f_err_wrong": f_err(sW, rig)}

    # B: start from the saved fit (its JSON; its camFiles alone), and D3
    shutil.copytree(P.DATA / "p2_A" / "camFile", P.DATA / "p2_A_camfiles" / "camFile")
    res_b = {}
    for label, folder in (("calibration_result.json", "p2_A"), ("camFiles alone", "p2_A_camfiles")):
        v = view_for(10.0)
        v.refr_start_combo.setCurrentIndex(2)
        v.refr_start_folder.setText(str(P.DATA / folder))
        jB, sB, _ = run(v)
        dfr = max(abs(c.f / a.f - 1) for c, a in zip(sB.cams, sA.cams))
        res_b[label] = {"rms_diff": abs(sB.rms - sA.rms), "f_rel": dfr, "start": jB.start_source}
    log.check("B: started from A's folder (its JSON, its camFiles alone), the answer is A's",
              all(r["rms_diff"] < 1e-6 and r["f_rel"] < 1e-6 for r in res_b.values()), str(res_b))
    v = view_for(11.0)
    v.refr_start_combo.setCurrentIndex(2)
    v.refr_start_folder.setText(str(P.DATA / "p2_A"))
    jD3 = P.catch_job(v)[1]
    _cams, rig3 = pg.start_with_windows(jD3.start, jD3.rig())
    log.check("D3: a start keeps its planes and takes the panel's thickness (11 mm, saved 10 mm)",
              rig3.windows[1].t == (11.0,) and np.allclose(rig3.windows[1].n, jD3.start[1].windows[1].n),
              f"run's window 1 t {rig3.windows[1].t}, saved {jD3.start[1].windows[1].t}")
    out["B"] = res_b

    # C: constraints
    XA, XB = w.XA, w.XB
    widths = [float(max(rig.wins[k].depth(XA).max(), rig.wins[k].depth(XB).max())) + 5.0 for k in range(2)]
    aim_true = float(pw.aim_depth(rig.cams, pw.Rig(rig.wins, rig.cam_win), [0, 1], 1))
    v = view_for(10.0)
    v.refr_window_table.item(0, 3).setText(f"{widths[0]:.1f}")
    v.refr_window_table.item(1, 3).setText(f"{widths[1]:.1f}")
    v.refr_cons_group.setChecked(True)
    v.refr_channel_check.setChecked(True)
    v.refr_channel_tol.setValue(1.0)
    v.refr_aim_table.insertRow(0)
    for j, t in enumerate(["0,1", "1", f"{aim_true:.3f}", ""]):
        v.refr_aim_table.setItem(0, j, QTableWidgetItem(t))
    jC, sC, rC = run(v, "p2_C")
    rep = sC.info.get("constraints") or {}
    page = (P.DATA / "p2_C" / "calibration_report.html").read_text(encoding="utf-8")
    aim_fit = float(pw.aim_depth(sC.cams, sC.rig, [0, 1], 1))
    log.check("C: the channel and the aim at the truth: A's answer, the constraints met, recorded and on the page",
              f_err(sC, rig) < 0.005 and (rep.get("channel") or {}).get("max_breach_mm", 1.0) < 1e-6
              and abs(aim_fit - aim_true) < 0.05 and rC["gui"]["settings"]["constraints"] == jC.constraints
              and "channel" in page and "aim of camera" in page,
              f"widths {np.round(widths, 1).tolist()} mm; aim {aim_true:.3f} -> {aim_fit:.3f} mm; f error "
              f"{100 * f_err(sC, rig):.3f} %; breach {(rep.get('channel') or {}).get('max_breach_mm')}; lines "
              f"{pg.constraint_lines(rC)}")
    out["C"] = {"aim_true": aim_true, "aim_fit": aim_fit, "f_err": f_err(sC, rig)}

    # D: lens terms
    v = view_for(10.0)
    v.dist_model_combo.setCurrentIndex(2)
    v.dist_model_combo.activated.emit(2)
    v.refr_pp_check.setChecked(True)
    v.refr_tangential_check.setChecked(True)
    jDl, sDl, rDl = run(v, "p2_D")
    pp = max(max(abs(c.cx - (c.width - 1) / 2), abs(c.cy - (c.height - 1) / 2)) for c in sDl.cams)
    log.check("D: Radial k1+k2, the principal point and tangential reach the fit and the record; f within 1 %",
              rDl["fit_lens"] == ["pp", "k2", "tangential"] and rDl["fit_k1"] and f_err(sDl, rig) < 0.01,
              f"fit_lens {rDl['fit_lens']}; f error {100 * f_err(sDl, rig):.3f} %; principal point at most {pp:.1f} px "
              f"from the centre; RMS {sDl.rms:.4f} px")
    out["D"] = {"f_err": f_err(sDl, rig), "pp_off_px": pp}

    # E: hints
    res_e = {}
    for k, folder in ((1, None), (2, str(P.DATA / "p2_A_camfiles"))):
        v = view_for(10.0)
        v.refr_hints_combo.setCurrentIndex(k)
        if folder:
            v.refr_hints_folder.setText(folder)
        jE, sE, rE = run(v, f"p2_E{k}")
        res_e[jE.hints_source] = {"f_err": f_err(sE, rig), "recorded": (rE["gui"]["settings"].get("hints") or {}).get("source"),
                                  "normals": np.round(np.asarray(jE.hints["normals"]), 3).tolist()}
    log.check("E: hints (windows face the cameras; from A's camFiles) reach the search and the record; f within 0.5 %",
              len(res_e) == 2 and all(r["f_err"] < 0.005 and r["recorded"] == s for s, r in res_e.items()), str(res_e))
    out["E"] = res_e

    # F: refusals
    msgs = []
    for cons in ({"aim": [{"cams": [0, 5], "wall": 1, "mm": 50.0}]}, {"aim": [{"cams": [0], "wall": 3, "mm": 50.0}]},
                 {"channel": {"width_mm": [None, None], "tolerance_mm": 1.0}},
                 {"distance": [{"cam": 0, "mm": -1.0}]}):
        try:
            pg.check_constraints(cons, 4, 2)
            msgs.append(None)
        except ValueError as e:
            msgs.append(str(e))
    try:
        pio.load_start(str(P.DATA / "p2_A"), [0, 1, 2, 3], [0, 1, 0, 1])
        msgs.append(None)
    except ValueError as e:
        msgs.append(str(e))
    log.check("F: bad constraints and a saved fit with another camera grouping are refused with a message",
              all(msgs), "; ".join(str(m)[:70] for m in msgs))
    log.save(out)


if __name__ == "__main__":
    main()
