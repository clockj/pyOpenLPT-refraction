# Cylinder STB study (8 Oct 2026)

A follow-up to the tracking through the tube (`tests/cylinder_stb/`, `cases/cyl_20261007/`). It asks:
- can the line detector do better on T1?
- how does synthetic tracking do with 2, 3 and 4 cameras, at three noise levels?
- do particles outside the central region cause trouble?
- why does T1 track worse than the synthetic case?

| file | what |
|---|---|
| `2026-10-08-plan.md` | the plan, approved 8 Oct; its section 7 holds the decisions and the changes made while building |
| `2026-10-08-report.md` | the report |
| `study.py` | shared paths and helpers (build p4c, T1's view volume, the rig's camFiles) |
| `make_rigs.py` | the 3rd and 4th cameras (`data/rig/`), and checks of the rig |
| `synth_study.py` | the rendered cases: `data/main` (Parts S and O, G-cal, G-shape), `data/gall` (G-all) |
| `eval_study.py` | scores a run against the truth, with n cameras, layers and the view volume |
| `pairstats.py` | S2's pair statistics, with and without the truth (Parts D and G) |
| `s_runs.py` | Parts S and O: the runs on 2, 3 and 4 cameras, and today's code |
| `d1_detect_t1.py`, `d2_detect_synth.py`, `d3_overlays.py` | Part D: the detector's settings on T1, on synthetic sets, overlays |
| `g1_gap.py`, `g2_realism.py`, `g3_control.py` | Part G: the gap table, the realism ablation, the ghost estimate checked |
| `r1_t1_rerun.py` | T1 rerun with a 15 px median (decision D3, on your request): the three runs compared (report section 7) |
| `summary_tables.py` | every table of the report, from the saved results (`summary_tables_out.txt`) |
| `before_hp/` | the first render's logs, made before the high-pass was added (see the report) |

Each script writes `<name>_out.txt` and `<name>.json` beside itself (and its console as `<name>_console.txt` when
run in the background). Rendered images and runs go to `data/` and `runs/`, git-ignored and regenerable.

## Running it

Everything runs with the `refract` interpreter on build `build-stb/p4c` (and `head1007` for today's code):

```
python make_rigs.py                                   # 3 min
python synth_study.py main                            # 10 min: data/main (Parts S, O; G-cal, G-shape)
python synth_study.py gshape0                         # G-shape's diagnostics: T1's model exact,
python synth_study.py gshape12                        #   and at half the error
python synth_study.py gall-trial                      # G-all's density from one frame
python synth_study.py gall --density 0.035            # data/gall
python synth_study.py gall12                          # G-all at half the error
python synth_study.py gall12m15                       # the same with a 15 px median (D1's best)
python synth_study.py gallm-trial --scatter 0.15 --trial-rho 0.035 0.05 0.07 0.1 --trial-bc 8 15 25
python synth_study.py gallm --density 0.06 --bc-max 8 --scatter 0.19      # data/gallm: matched to T1
python s_runs.py --stream work --threads 12 --tag a   # Parts S and O; more streams (--tag b, c, ...) share the runs
python s_runs.py --stream today --threads 16          # today's code
python g2_realism.py --threads 16                     # G2's runs (--only NAME ... for some)
python g3_control.py [--set Gall12 | --set Gallm --case gallm]
python d1_detect_t1.py                                # about 1.5 h (--only med15_k5 --no-best adds a setting)
python d2_detect_synth.py --sets S_n8 --only base k5 med15 med15_k5 --tag S_n8   # and --sets Gall ... --tag Gall
python d3_overlays.py
python g1_gap.py                                      # (--only SET ... adds sets)
# T1's rerun (report section 7): the case's runs first (commands in cases/cyl_20261007/README.md), then
python g2_realism.py --only Gall12mn_r2_control Gall12m15_r2_control Gall12m15k5_r2_control --threads 12
python g3_control.py --set Gall12 --run Gall12mn_r2   # and --set Gall12_m15 --run Gall12m15_r2 (or Gall12m15k5_r2)
python r1_t1_rerun.py
python summary_tables.py                              # every table of the report
```

## Results in brief (8 Oct; the report has the tables)

- **Detection on T1.** A 15 px median in S1's preprocessing (31 px now) gives +48 % true stereo pairs per frame at a
  higher share; with k_detect 5 as well, +136 %. On the synthetic case most like T1 it raises tracking recall from
  0.14 to 0.22–0.24 with fewer ghosts. The detector's own settings are near their best.
- **More cameras.** Noise 8, recall of the particles all cameras see / ghosts: 2 cameras 0.71 / 11 %, 3 cameras
  0.88 / 5.5 %, 4 cameras 0.77 / 1.4 %. The 4-camera rig needs three cameras per match. The elevated cameras' images
  are larger and fainter (more astigmatism; cam3 detects a third of its images), so a real third camera wants a
  smaller aperture.
- **Noise.** 4 and 8 make no difference; 16 costs about a tenth of the recall.
- **Particles outside the central region** cost recall but add no ghosts: 2 cameras 0.71 → 0.64 → 0.55 at 0, 0.5 and
  1× the central density.
- **Today's code** stays at 78–92 % ghosts with 2, 3 or 4 cameras.
- **Why T1 tracks worse.** In order of effect:
  - the shape model's error (about ±19 % on T1; the depth cue then keeps half the true pairs);
  - the calibration error (0.7 px: recall 0.71 → 0.57);
  - the particles outside the central region;
  - T1's larger, fainter images.

  A synthetic case matched to T1 reproduces its truth-free statistics, except that T1's detections pair half as
  often.
- **T1's ghost share.** The two estimators read 2–3.5 times the truth on synthetic cases, so T1's 44–51 % is more
  likely 15–25 %.
- **T1 rerun with a 15 px median (8 Oct evening).** About twice the real tracks at the same ghost share: 296 against
  152 at frame 100 by the counts estimate, 1,228 long tracks in the whole run against 701. Residuals, shapes and
  lengths are unchanged, and 73 % of the old run's tracks recur in it. With k 5 as well: 418 at frame 100 and 1,818
  in the whole run, slightly rougher. On G-all the counts estimate follows the true gain; the swirl estimate
  understates it. Since 8 Oct the 15 px run is T1's result, on X: and in the HTML report; the first run is in
  `T1\run_31px\`.
