"""Every number that the paper's Section 9.2.2 and Supplement 1 quote about the synthetic wand calibration, read from
the saved outputs of the campaign tests/cylinder_wand/ and of this case's E0-E2, each printed with its source.

Nothing is computed afresh: the campaign's results stand for the paper's calibration because E0-E2 show that, on
the campaign's data, the paper's residual equals the library's (2026-10-06-report.md).  Where the campaign stored
no prediction, none is printed.
Outputs: paper_numbers_out.txt; paper_numbers.json, which also holds the series of the paper's figure
(code/make_paper_wand_figs.py --refresh copies them into the paper's figures/data/).
Runs with plain Python and NumPy; no extension is imported.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parents[1] / "tests" / "cylinder_wand"
LINES, OUT = [], {}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def say(key, value, source, fmt="{}"):
    """One number: printed with its source, and kept under `key` in the JSON."""
    OUT[key] = value
    text = fmt.format(value) if not isinstance(value, (list, tuple)) else ", ".join(fmt.format(v) for v in value)
    LINES.append(f"{key:44s} {text:44s} <- {source}")


def main():
    # ---- the setup
    t4f = load(T / "t4_fits.json")
    tr = t4f["truth"]
    say("rig.D_mm", float(np.hypot(tr[0]["C"][0], tr[0]["C"][2])), "t4_fits.json truth[0].C", "{:.2f}")
    say("rig.azimuth_deg", [float(np.degrees(np.arctan2(c["C"][0], c["C"][2]))) for c in tr], "t4_fits.json truth[c].C",
        "{:.1f}")
    say("rig.f_px", tr[0]["f"], "t4_fits.json truth[0].f", "{:.1f}")
    say("rig.sensor_px", [tr[0]["width"], tr[0]["height"]], "t4_fits.json truth[0].width/height")
    e0 = load(HERE / "e0_census.json")
    say("tube.r_c0_mm", e0["r_c0_mm"], "e0_census.json r_c0_mm", "{:.4f}")
    say("tube.generator_r_max_mm", e0["r_max_mm"], "e0_census.json r_max_mm", "{:.4f}")
    say("e0.reference_max_radius_mm", e0["reference"]["max_radius_mm"], "e0_census.json reference.max_radius_mm",
        "{:.2f}")
    say("e0.reference_max_abs_cz", max(e0["reference"]["max_abs_cz_data"], e0["reference"]["max_abs_cz_box"]),
        "e0_census.json reference.max_abs_cz_*", "{:.4f}")
    say("e0.t5_max_radius_mm", e0["t5_max_radius_mm"], "e0_census.json t5_max_radius_mm", "{:.2f}")
    say("e0.t5_max_abs_cz", e0["t5_max_abs_cz"], "e0_census.json t5_max_abs_cz", "{:.4f}")
    counts = {}
    for blk in [e0["reference"]] + e0["t5"]:
        for part in ("counts_data", "counts_box"):
            for k, v in blk[part].items():
                counts[int(k)] = counts.get(int(k), 0) + v
    say("e0.pairs_total", sum(counts.values()), "e0_census.json counts_data + counts_box, summed")
    say("e0.pairs_by_image_count", {str(k): v for k, v in sorted(counts.items())}, "e0_census.json, summed")
    e0b = load(HERE / "e0b_zero_images.json")["pairs"]
    say("e0b.zero_pairs", len(e0b), "e0b_zero_images.json pairs")
    say("e0b.python_finds_one", sum(p["python_images"] == 1 for p in e0b), "e0b_zero_images.json python_images")
    say("e0b.max_one_minus_lam_ratio", max(max(p["python_one_minus_lam_ratio"]) for p in e0b),
        "e0b_zero_images.json python_one_minus_lam_ratio", "{:.1e}")
    say("e0b.abs_dphi_deg_range", [min(abs(p["dphi_deg"]) for p in e0b), max(abs(p["dphi_deg"]) for p in e0b)],
        "e0b_zero_images.json dphi_deg", "{:.2f}")

    # ---- the paper's residual against the library's (E1, E2)
    e1 = load(HERE / "e1_equivalence.json")
    say("e1.max_dr_px_truth", max(c["truth"]["max_dr_px"] for c in e1["cases"]), "e1_equivalence.json cases[].truth",
        "{:.1e}")
    say("e1.max_dr_px_start", max(c["start"]["max_dr_px"] for c in e1["cases"]), "e1_equivalence.json cases[].start",
        "{:.1e}")
    say("e1.max_dJ_rel", max(max(c[p]["max_dJ_cam_rel"], c[p]["max_dJ_frame_rel"]) for c in e1["cases"]
                             for p in ("truth", "start")), "e1_equivalence.json cases[].*.max_dJ_*", "{:.1e}")
    say("e1.data_sets", len(e1["cases"]), "e1_equivalence.json cases")
    e2 = load(HERE / "e2_same_fit.json")
    fits = e2["t4"] + e2["t5"] + [e2["t9"]]
    say("e2.fits", len(fits), "e2_same_fit.json t4 + t5 + t9")
    say("e2.max_shift_sigma", max(f["max_shift_sigma"] for f in fits), "e2_same_fit.json *.max_shift_sigma", "{:.1e}")
    say("e2.max_rms_rel_diff", max(f["rms_rel_diff"] for f in fits), "e2_same_fit.json *.rms_rel_diff", "{:.1e}")
    say("e2.active_differ_total", sum(f["active_differ"] for f in fits), "e2_same_fit.json *.active_differ")
    say("e2b.max_shift_crb_sigma", max(r["max_shift_crb_sigma"] for r in e2["e2b"]),
        "e2_same_fit.json e2b[].max_shift_crb_sigma", "{:.3f}")

    # ---- recovery from scratch (T5, T12d, T11b, T12e)
    t5 = load(T / "t5_from_scratch.json")
    say("t5.ok_of_n", [t5["n_ok"], t5["n"]], "t5_from_scratch.json n_ok, n")
    ncams = [r["n_cams"] for r in t5["rigs"]]
    say("t5.rigs_with_2_3_4_cams", [ncams.count(2), ncams.count(3), ncams.count(4)], "t5_from_scratch.json rigs[].n_cams")
    D = [d for r in t5["rigs"] for d in r["D_mm"]]
    say("t5.D_mm_range", [min(D), max(D)], "t5_from_scratch.json rigs[].D_mm", "{:.0f}")
    say("t5.max_diff_sigma", max(r["max_diff_sigma"] for r in t5["rigs"]), "t5_from_scratch.json rigs[].max_diff_sigma",
        "{:.1e}")
    first = load(T / "t5_first_pass.json")
    say("t5.first_pass_ok_of_n", [first["n_ok"], first["n"]], "t5_first_pass.json n_ok, n (superseded run)")
    t12d = load(T / "t12d_scratch_t5.json")
    say("t12d.ok_of_n", [t12d["n_ok"], t12d["n"]], "t12d_scratch_t5.json n_ok, n")
    t11b = load(T / "t11b_noise_scratch.json")
    say("t11b.reference_ok", [[lv["sigma"], lv["ok"], lv["n"]] for lv in t11b["ref"]], "t11b_noise_scratch.json ref[]")
    say("t11b.random_rigs_ok", [[lv["sigma"], lv["ok"], lv["n"]] for lv in t11b["rigs"]],
        "t11b_noise_scratch.json rigs[]")
    t12e = load(T / "t12e_scratch_5px.json")
    say("t12e.ok_of_n_5px", [t12e["n_ok"], t12e["n"]], "t12e_scratch_5px.json n_ok, n")

    # ---- precision on the reference rig at 0.1 px (T11a level 0.1 px, T7, T4)
    t11a = load(T / "t11a_noise_sweep.json")
    lv = next(v for v in t11a["levels"] if abs(v["sigma"] - 0.1) < 1e-12)
    f = lv["phys"]["f_rel"]
    say("t11a.n_sets_per_level", lv["n"], "t11a_noise_sweep.json levels[0.1].n")
    say("t11a.f_rel_rms_worst_pct", 100 * max(f["rms_per_cam"]), "t11a levels[0.1].phys.f_rel.rms_per_cam (max)",
        "{:.4f}")
    say("t11a.f_rel_pred_worst_pct", 100 * max(f["pred_per_cam"]), "t11a levels[0.1].phys.f_rel.pred_per_cam (max)",
        "{:.4f}")
    say("t11a.f_ratio_pooled", f["ratio"], "t11a levels[0.1].phys.f_rel.ratio", "{:.3f}")
    say("t11a.f_rel_rms_cams_pct", 100 * float(np.sqrt(np.mean(np.square(f["rms_per_cam"])))),
        "t11a levels[0.1].phys.f_rel.rms_per_cam, RMS over cameras", "{:.4f}")
    say("t11a.f_rel_pred_cams_pct", 100 * float(np.sqrt(np.mean(np.square(f["pred_per_cam"])))),
        "t11a levels[0.1].phys.f_rel.pred_per_cam, RMS over cameras", "{:.4f}")
    say("t11a.rigid_3d_um", lv["unseen_rigid_um"], "t11a levels[0.1].unseen_rigid_um", "{:.3f}")
    say("t11a.position_vs_tube_um", lv["unseen_stab_um"], "t11a levels[0.1].unseen_stab_um", "{:.2f}")
    say("t11a.heldout_wand_um", lv["held_noisy_rms_um"], "t11a levels[0.1].held_noisy_rms_um", "{:.3f}")
    say("t11a.heldout_wand_calibration_share_um", lv["held_exact_rms_um"], "t11a levels[0.1].held_exact_rms_um",
        "{:.3f}")
    say("t11a.z_crb_sd_range", [min(v["z_crb"]["std"] for v in t11a["levels"]),
                                max(v["z_crb"]["std"] for v in t11a["levels"])], "t11a levels[].z_crb.std", "{:.3f}")
    say("t11a.z_fit_sd_range", [min(v["z_fit"]["std"] for v in t11a["levels"]),
                                max(v["z_fit"]["std"] for v in t11a["levels"])], "t11a levels[].z_fit.std", "{:.3f}")
    t7 = load(T / "t7_heldout.json")
    say("t7.wand_rms_um", [1e3 * h["length_rms_mm"] for h in t7["heldout_frames"]],
        "t7_heldout.json heldout_frames[].length_rms_mm", "{:.3f}")
    say("t7.wand_pred_um", [1e3 * h["length_pred_mm"] for h in t7["heldout_frames"]],
        "t7_heldout.json heldout_frames[].length_pred_mm", "{:.3f}")
    say("t7.wand_rms_um_pooled", 1e3 * float(np.sqrt(np.mean([h["length_rms_mm"] ** 2 for h in t7["heldout_frames"]]))),
        "t7 heldout_frames[].length_rms_mm, RMS over the two data sets", "{:.3f}")
    say("t7.wand_pred_um_pooled", 1e3 * float(np.sqrt(np.mean([h["length_pred_mm"] ** 2 for h in t7["heldout_frames"]]))),
        "t7 heldout_frames[].length_pred_mm, RMS over the two data sets", "{:.3f}")
    say("t7.unseen_px", t7["unseen"]["retriangulated"]["realised_rms_px"],
        "t7_heldout.json unseen.retriangulated.realised_rms_px", "{:.5f}")
    say("t7.unseen_pred_px", t7["unseen"]["retriangulated"]["predicted_rms_px"],
        "t7_heldout.json unseen.retriangulated.predicted_rms_px", "{:.5f}")
    say("t7.gated_ratios", t7["gated_ratios"], "t7_heldout.json gated_ratios", "{:.3f}")
    t4 = load(T / "t4_noisy_recovery.json")
    say("t4.z_mean_sd", [t4["z_mean"], t4["z_std"]], "t4_noisy_recovery.json z_mean, z_std", "{:.3f}")
    say("t4.chi2_mean_dof", [t4["chi2_mean"], t4["chi2_dof"]], "t4_noisy_recovery.json chi2_mean, chi2_dof", "{:.2f}")
    say("t4.fit_sigma_over_crb", t4["fit_sigma_over_crb_median"], "t4_noisy_recovery.json fit_sigma_over_crb_median",
        "{:.3f}")
    t12b = load(T / "t12b_weights_equal.json")
    say("t12b.loss_0.1_1px", [v["loss"] for v in t12b["levels"]], "t12b_weights_equal.json levels[].loss", "{:.4f}")
    say("t12b.moved_sigma_0.1_1px", [v["moved_sigma"] for v in t12b["levels"]], "t12b_weights_equal.json levels[].moved_sigma",
        "{:.3f}")

    # ---- the noise sweep (figure, panel a)
    series = []
    for v in t11a["levels"]:
        fr = v["phys"]["f_rel"]
        series.append({"sigma_px": v["sigma"], "n": v["n"], "f_rel_rms_per_cam": fr["rms_per_cam"],
                       "f_rel_pred_per_cam": fr["pred_per_cam"], "f_ratio": fr["ratio"],
                       "rigid_3d_um": v["unseen_rigid_um"], "position_vs_tube_um": v["unseen_stab_um"],
                       "heldout_wand_um": v["held_noisy_rms_um"], "z_crb_sd": v["z_crb"]["std"]})
    OUT["figure.noise_sweep"] = series
    say("t11a.f_ratio_range", [min(s["f_ratio"] for s in series), max(s["f_ratio"] for s in series)],
        "t11a levels[].phys.f_rel.ratio", "{:.3f}")
    at1 = next(s for s in series if abs(s["sigma_px"] - 1.0) < 1e-12)
    say("t11a.per_px.f_pct_3d_um_wand_um", [100 * max(at1["f_rel_rms_per_cam"]), at1["rigid_3d_um"],
                                            at1["heldout_wand_um"]], "t11a levels[1 px]", "{:.3g}")

    # ---- the tube constants (figure, panel b)
    t9 = load(T / "t9_tube_constants.json")
    cases = [{"case": c["case"], "L_free_mm": c["L_free_mm"], "L_free_sigma_mm": c["L_free_sigma_mm"],
              "offset_sigma": c["L_free_offset_sigma"], "rms_px": c["rms_px"], "max_shift_sigma": c["max_shift_sigma"]}
             for c in t9["cases"]]
    OUT["figure.tube_constants"] = cases
    base = cases[0]
    say("t9.L_free_true_mm", [base["L_free_mm"], base["L_free_sigma_mm"]], "t9_tube_constants.json cases[0]", "{:.6f}")
    say("t9.scale_sigma_rel_pct", 100 * base["L_free_sigma_mm"] / 10.0, "t9 cases[0].L_free_sigma_mm / 10 mm", "{:.4f}")
    for c in cases[1:]:
        say(f"t9.offset_sigma[{c['case']}]", c["offset_sigma"], "t9_tube_constants.json cases[].L_free_offset_sigma",
            "{:+.1f}")
        say(f"t9.rms_change_pct[{c['case']}]", 100 * (c["rms_px"] / base["rms_px"] - 1),
            "t9 cases[].rms_px / cases[0].rms_px - 1", "{:+.2f}")

    # ---- the limitation: a pattern fixed to each image (T11e)
    t11e = load(T / "t11e_systematic.json")
    for v in t11e["variants"]:
        nm = v["name"]
        say(f"t11e[{nm}].pattern_at_detections_px", v["pattern_rms_px"], "t11e_systematic.json variants[].pattern_rms_px",
            "{:.3f}")
        say(f"t11e[{nm}].absorbed_fraction", v["absorbed_fraction_mean"], "t11e variants[].absorbed_fraction_mean",
            "{:.2f}")
        say(f"t11e[{nm}].f_bias_pct_rms", v["f_bias_pct_rms"], "t11e variants[].f_bias_pct_rms", "{:.2f}")
        say(f"t11e[{nm}].f_error_over_noise_only_crb", v["summary"]["phys"]["f_rel"]["ratio"],
            "t11e variants[].summary.phys.f_rel.ratio", "{:.2f}")
        say(f"t11e[{nm}].z_fit_rms", v["summary"]["z_fit"]["rms"], "t11e variants[].summary.z_fit.rms", "{:.2f}")
        say(f"t11e[{nm}].fit_rms_px", v["summary"]["rms_px"], "t11e variants[].summary.rms_px", "{:.4f}")
        say(f"t11e[{nm}].L_flagged_of_20", v["L_flagged_3sigma"], "t11e variants[].L_flagged_3sigma")
    say("t11a.rms_px_noise_only_0.1", lv["rms_px"], "t11a levels[0.1].rms_px", "{:.4f}")

    text = "\n".join(LINES)
    print(text)
    (HERE / "paper_numbers_out.txt").write_text(text + "\n", encoding="utf-8")
    with open(HERE / "paper_numbers.json", "w", encoding="utf-8") as fh:
        json.dump(OUT, fh, indent=1)


if __name__ == "__main__":
    main()
