# Self-calibration of T1 from its own tracks (8 Oct 2026)

A follow-up to the study (`tests/cylinder_stb_study/`): would a VSC-style self-calibration (reliable long tracks used
to correct the cameras) help T1's tracking through the tube, and would refitting the shape model from the tracked
particles help the depth cue?  You asked for the synthetic test with different numbers of cameras as well.

| file | what |
|---|---|
| `2026-10-08-plan.md` | the plan, approved 8 Oct with D1-D4 as recommended |
| `2026-10-08-report.md` | the report |
| `selfcal.py` | shared code: paths, tracks, detections and pairing, Jacobians, the correction field and its projected least-squares fit (2, 3, 4 cameras), the noise model, the truth-free estimators, the warp, field composition |
| `t0_selftest.py` | self-tests of `selfcal.py` |
| `gather.py` | per frame: a run's track points with their detections, Jacobians, epipolar offsets; S2's candidate pairs; focal depths |
| `v0_pattern.py`, `v0b_truthfree.py` | V0: the pattern on T1; the truth-free estimates |
| `v1_benefit.py` | V1: the benefit on held-out T1 frames, truth-free |
| `synth_sc.py` | V2's synthetic sets: the 4-camera set with an error field in every camera (`data/cal4`), Gallm and Gallm0 (`data/gallm`), corrected (re-rendered) and warped sets |
| `sruns.py` | V2's tracking runs (2, 3, 4 cameras) and their scores against the truth |
| `v2_fit.py`, `v2_diag.py` | V2: the field fitted from a run's tracks against the truth; the estimator check (noise floor, robust loss, oracle) |
| `v2_positions.py`, `v2_gauge.py` | V2: what the correction does to positions (no detection noise); the gauge (per-point and affine penalties) |
| `compose_fields.py` | the second pass's total field |
| `v3_warp.py`, `t1_runs.py`, `fit_t1.py`, `v3_compare.py` | V3: T1's frames warped by the field (`--check`: without a warp it reproduces the 15 px frames), its tagged runs, the second pass's fit, the runs compared |
| `v3_t1checks.py`, `v3_positions_t1.py` | T1's checks T1-a to T1-e on a run (the case's, ported: outputs here; reproduces the case's on the 15 px run); positions against today's run |
| `v3_to_x.py`, `v3_xcheck.py` | the two-pass run and its checks into `T1\after_vsc\` on X: (8 Oct evening, on request); its config rerun from X: for 4 frames |
| `m01_shape.py` | M0 and M1: the shape model from the tracks' widths (with the ROC and the image-position term M1b) |
| `summary_sc.py` | every table of the report, from the saved results; `summary_sc.png` |
| `queue_*.sh`, `prep_vc2.sh`, `final_v3.sh` | the run streams as they were run (two streams of 12 threads), T1's second-pass preparation, V3's last step |
| `before_cauchy/` | the first fits' outputs (Huber weights, no noise floor), before the estimator check changed them |

Each script writes `<name>_out.txt` and `<name>.json` beside itself (and its console as `<name>_console.txt` when run in
the background).  Rendered sets, gathers and fields go to `data/`, synthetic runs to `runs/` (git-ignored,
regenerable).  T1's tagged runs go to `cases/cyl_20261007/data/runs/s3_{final,control}_<tag>/`:
`m15cue` (the depth cue loosened, M0), `m15vc`, `m15vc10` (V3: the warped frames at 1.5 and 1.0 px), `m15t10` (the
15 px frames uncorrected at 1.0 px) and `m15vc2` (the second pass, 1.0 px).

## Running it

Everything runs with the `refract` interpreter on build `build-stb/p4c` (read only), `OMP_NUM_THREADS` at most 12:

```
python t0_selftest.py
P=../../cases/cyl_20261007/data
python gather.py t1_m15 --run $P/runs/s3_final_m15 --lists $P/T1pre_s4_m15/cam1ImageNames.txt $P/T1pre_s4_m15/cam2ImageNames.txt \
       --noise 7.514877384110587 5.502475966380505 --cams $P/cams_T0/cam1.txt $P/cams_T0/cam2.txt --every 5 --pairs --depths
python v0_pattern.py                       # V0; writes data/fields/t1_m15_{fit,all,h1,h2}.npz
python v0b_truthfree.py                    # V0's truth-free estimates (about 1 h); data/fields/t1_m15_tf_*.npz
python gather.py t1_m15_v1trk ... --frames 5 15 ... 245 --pairs --depths --correct data/fields/t1_m15_fit.npz   # (and _v1tf)
python v1_benefit.py
python t1_runs.py final --tag m15cue --pre $P/T1pre_s4_m15 --tol2d 1.5 --shape-tol 1.5 0.4
python gather.py t1_m15cue --run $P/runs/s3_final_m15cue ... --every 5 --depths
python m01_shape.py                        # M0, M1
python synth_sc.py cal4                    # the 4-camera set (its cam1, cam2 equal the study's Gcal_n8)
python synth_sc.py gallm                   # Gallm, Gallm0
python sruns.py Gcal4_n8_r3_tol1.5 Gcal4_n8_r4_tol1.5         # before (2 cameras: the study's Gcal_n8 runs)
python gather.py cal4_r<n> ...; python v2_fit.py cal4_r<n> --gather cal4_r<n> --case cal4 --set Gcal4_n8
python synth_sc.py corrected cal4 Gcal4_n8 Gcal4c<n>_n8 data/fields/cal4_r<n>.npz
python sruns.py Gcal4c2_n8_r2_tol1.0 Gcal4c3_n8_r3_tol1.0 Gcal4c4_n8_r4_tol1.0
python v2_diag.py cal4_r4 cal4 Gcal4_n8    # the estimator check (its result is selfcal's default: Cauchy, s_f >= 0.15)
python v2_positions.py; python v2_gauge.py # positions and the gauge (no tracking)
python gather.py gallm_r2 ...; python v2_fit.py gallm_r2 --gather gallm_r2 --case gallm --set Gallm --tf
python synth_sc.py corrected gallm Gallm Gallmc data/fields/gallm_r2.npz; python synth_sc.py warped gallm Gallm Gallmw data/fields/gallm_r2.npz
python sruns.py Gallm0_r2_tol1.5 Gallmw_r2_tol1.5 Gallmc_r2_tol1.5 Gallm_r2_tol1.0 Gallm0_r2_tol1.0 Gallmc_r2_tol1.0 Gallmw_r2_tol1.0
bash queue_c.sh                            # the second pass on the 2-camera set and on Gallm (gather, fit, compose, apply, track)
python v3_warp.py --check; python v3_warp.py vc data/fields/t1_m15_all.npz
python gather.py t1_m15_v1all ... --correct data/fields/t1_m15_all.npz; python gather.py t1_vcwarp ...   # the warp check
python v1_benefit.py --name v3a_warpcheck --variants detections=t1_m15_v1all warped=t1_vcwarp
python t1_runs.py final --tag m15vc --pre data/T1pre_s4_m15vc --tol2d 1.5      # and control; m15vc10 at 1.0 px
python t1_runs.py final --tag m15t10 --pre $P/T1pre_s4_m15 --tol2d 1.0         # and control: the tolerance alone
bash prep_vc2.sh                           # T1's second pass: gather the m15vc run, fit_t1.py, compose_fields.py, v3_warp.py vc2
python t1_runs.py final --tag m15vc2 --pre data/T1pre_s4_m15vc2 --tol2d 1.0     # and control
bash final_v3.sh                           # gather the second pass, v3_compare.py (five runs), summary_sc.py
```

The run streams as they were run are `queue_a.sh`, `queue_a2.sh`, `queue_b.sh`, `queue_c.sh` and `queue_e.sh`.
