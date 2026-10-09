# The library's image correction and the GUI's VSC for tube / sphere cameras (8 Oct 2026; follow-up 9 Oct)

You asked, after the self-calibration study (`tests/cylinder_selfcal/`):

> ok, you can add that library. also you should update the gui so that the vsc part can work for cylinder case.

| file | what |
|---|---|
| `2026-10-08-plan.md` | the plan, approved 8 Oct with D1 (a) and D2 (a) |
| `2026-10-08-report.md` | the report |
| `build_vcf.py` | builds the library with the change (`build-stb/vcf`) from a copy of the source, so the GUI's `build/Release/OpenLPT.exe` is never touched |
| `vcf_common.py` | shared helpers: paths, the build's module, the log, the self-calibration's fields as correction files |
| `convert_fields.py` | the self-calibration's fields as correction files (`data/fields/`) |
| `l0_off.py`, `queue_l0.sh` | L0: without a correction nothing changes (T1, a synthetic set, pytest; vcf against p4c) |
| `l1_wrapper.py` | L1: the corrected camera, method by method, against the Python reference |
| `t1_lib_runs.py` | T1's tagged runs on build vcf (L2, G1), into the case's `data/runs/s3_<final|control>_<tag>/` |
| `l3_synthetic.py` | L3: the library's correction against re-rendered images, 2 / 3 / 4 cameras, scored against the truth |
| `queue_l2_l3.sh` | stream 1 as it was run: L2's runs, then L3 |
| `g0_port.py` | G0: the GUI's estimator (`modules/vsc/field_vsc.py`) against the self-calibration's, on the same data |
| `g1_project.py`, `g1_pass.py` | G1: test projects (T1 as the GUI sees it, `projects/`), a pass of the cylinder VSC and its correction against the self-calibration's |
| `g1_dispatch.py` | G1: which VSC a project gets (tube projects the field VSC; plate and pinhole projects today's; a tube project with only config.txt refused with the conversion to do) |
| `queue_g1.sh` | stream 2 as it was run: G1's two passes end to end (tracking, VSC, the guard, tracking) |
| `g2_synthetic.py` | G2: the cylinder VSC on the synthetic sets (2 / 3 / 4 cameras) and the GUI path over two passes |
| `g3_gui.py`, `g3_settings.py` | G3: the tracking view's Run VSC on a tube project (offscreen, screenshots `g3_*.png`); what the tracking settings page does to its config.json |
| `g3_debug.py` | G3's debugging aid: the service's result handed to the view's finish handler, traced (it found that the offscreen harness must keep the message box's event loop) |
| `compare_t1.py` | T1's runs compared end to end (ghost estimates, real tracks, recurrence, the residual across the epipolar curves) |
| `2026-10-09-plan.md`, `2026-10-09-report.md` | the follow-up (your decisions 1–4 of 9 Oct): the install, the tracking settings page for tube projects, the VSC thread, the vendored oracle |
| `s0_noise.py` | S0: whether the settings page's Validate detects as the tracking run does (the line detector's per-camera noise) |
| `s1_save.py` | S1: the settings page's Save on tube, sphere, plate and pinhole projects, today's page (commit cffed7d's, by `git show`) against the change |
| `s2_volume.py` | S2: the settings page's view-volume estimate on tube rigs (against a dense projection reference and the particles) and on plate and pinhole rigs (equal to today's) |
| `s3_validate.py` | S3: the settings page's Validate on a T1 project, today's page against the change |
| `h1_vsc_thread.py` | H1: the tracking view's VSC finish handler with message boxes that return at once (today's handler aborts; the change must not) |
| `i1_install.py`, `queue_i1.sh` | I1: the install (`build_local.py`): hashes, the module's API, the installed exe against build vcf on T1 (byte for byte), pytest with the installed module |
| `d1_detection.py`, `queue_d1.sh` | D1 (follow-up 2, 9 Oct afternoon): every GUI detection path (Validate, the field VSC, the plate/pinhole VSC's crops) against the tracking run's own log (T1, 20261008, the synthetic 3-camera set, the 20260727 plate run on X:); the queue then reruns G3 and pytest on the rebuilt module |

Each script writes `<name>_out.txt` and `<name>.json` beside itself. Correction files, run outputs and test projects go
to `data/`, `runs/` and `projects/` (git-ignored, regenerable). T1's runs go to the case's
`data/runs/s3_{final,control}_<tag>/`: `lc2` (L2: the library correction set to the self-calibration's two-pass field),
`lp1`, `lp2` (G1: the GUI's passes 1 and 2).

`vcf_common.lib("gui")` loads the GUI's installed module (`build-py/Release`), and `vcf_common.committed_module()`
imports a file as commit cffed7d has it ("today's code" of the follow-up's before/after tests; the copy goes to
`data/ref/`, git-ignored). `g3_gui.py NAME gui` reruns G3 with the installed module (outputs `g3_gui_gui_*`).

## Running it

Everything runs with the `refract` interpreter; the library is build `build-stb/vcf` (`python build_vcf.py`), the
baseline `build-stb/p4c`; `OMP_NUM_THREADS` at most 12.

```
python build_vcf.py                  # the library with the change, from a copy of the source
python convert_fields.py             # the self-calibration's fields as correction files
python l1_wrapper.py                 # L1
bash queue_l0.sh                     # L0 (then: python l0_off.py pytest p4c|vcf; python l0_off.py compare)
python g0_port.py                    # G0
python g1_project.py t1_selfcalproto; python g1_pass.py t1_selfcalproto --via direct --min-len 10 --stride 5 --compare t1_m15_all
python g1_project.py t1_gui;          python g1_pass.py t1_gui --via service --compare t1_m15_all
bash queue_l2_l3.sh                  # stream 1: L2, L3
bash queue_g1.sh                     # stream 2: G1's passes end to end
for n in 2 3 4; do python g2_synthetic.py prepare $n syn_r$n; python g2_synthetic.py vsc syn_r$n --via direct --compare cal4_r$n; done
python g1_dispatch.py
python g1_project.py t1_g3; python g3_gui.py t1_g3; python g3_settings.py
bash queue_g2.sh                     # G2's 2-camera GUI path over two passes, after stream 1
python compare_t1.py --name compare_t1 --ref warped_2pass today_1.5=s3_final_m15,s3_control_m15 \
    uncorrected_1.0=s3_final_m15t10,s3_control_m15t10 warped_2pass=s3_final_m15vc2,s3_control_m15vc2 \
    library_2pass=s3_final_lc2,s3_control_lc2 gui_pass1=s3_final_lp1 gui_pass2=s3_final_lp2,s3_control_lp2
# the follow-up (9 Oct)
python s0_noise.py; python s1_save.py; python s2_volume.py; python s3_validate.py
python g1_project.py t1_h1; python h1_vsc_thread.py
python i1_install.py before   # then, from the repository's root: python build_local.py
python i1_install.py after; bash queue_i1.sh
python g1_project.py t1_g3gui; python g3_gui.py t1_g3gui gui
# follow-up 2 (the detection): rebuild (build_local.py, build_vcf.py), then
python s3_validate.py; python g1_project.py t1_g3gui2; bash queue_d1.sh
```
