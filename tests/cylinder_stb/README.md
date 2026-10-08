# Cylinder STB tests

Tests of the work that lets pyOpenLPT's STB track tracers through refractive cylinder windows (PINCYL cameras).
Plan: `cases/cyl_20261007/2026-10-07-plan.md`, Phases 1–5 (the case's own phases, C and 6, are in
`cases/cyl_20261007/`). Report: `2026-10-07-report.md` here.

Every script writes `<name>_out.txt` (and `.json`, figures) beside itself. Runs go to `runs/` and test data to
`data/`, both git-ignored (the synthetic cases are rebuilt by `synth.py` / `synth_lines.py`).

## Running

Use the `refract` interpreter (`C:\Users\zcloc\.conda\envs\refract\python.exe`) from this folder. The tests load a
named build of the module (`cstb.use_build(NAME)`: `build-stb/NAME/out/`, made by
`python tests/stb_improve/build_variant.py NAME` from the working tree). The builds named in the outputs:

| build | what |
|---|---|
| `head1007` | HEAD without this work (the baseline: "today's code") |
| `p1` | Phase 1 (projection hints) |
| `p123`, `p3` | Phases 1–3 (visibility, curved image lines) |
| `p4b`, `p4dev` | Phase 4 during development (the line detector, before the width-based depth cue) |
| `p4c` | **the final build**: Phases 1–4, the depth cue on RMS widths |

    python run_final.py --build p4c          # every test below on one build, with its verdict lines
    python r_regression.py p4c               # existing cameras: bit-identical tracking (about 20 min)
    python r5_ablation.py --runs R0 R1 R2 R3 # the synthetic end-to-end ablation (about 25 min)
    python synth.py f5_spots ...             # the synthetic cases (see each script's docstring)
    python synth_lines.py r5_lines
    python synth_lines.py r5_lines --otf-only   # rebuild only its true OTF table

and from the repository root, the fast pytest subset: `CONFIG=../build-stb/p4c/out pytest tests/test_cylinder_stb.py`.

## The tests

| id | script | what | judged by (plan) |
|---|---|---|---|
| F1 | `f1_agreement.py` | the hinted projection against `project()` on 500,000 points | pixels within 1e-6 px; no new failure |
| F1b | `f1b_residuals.py` | diagnosis of F1–F3's largest differences | they are the projection's own tolerance |
| F2 | `f2_los_hints.py` | hints from lines of sight (as StereoMatch gives them) | within 1e-6 px |
| F3 | `f3_junction.py` | particles crossing a camera's near/far junction, chained hints | within 1e-6 px; no new failure |
| F3b | `f3b_junction_diag.py` | diagnosis of F3 path B | at the rounding scale of the grazing ray |
| F4 | `f4_speed.py` | cost of a projection with and without a hint | hinted ≤ 3 µs |
| F5 | `f5_stb_hints.py` | STB through the tube with the hints on and off | off: bit-identical to HEAD; on: the same tracks |
| E1, E2 | `e1_coverage.py` | curved against straight image lines | ≥ 99.9 % of true images within 0.25 px |
| E3 | `e3_large_box.py` | a view volume past the tube's wall | every line built |
| V1 | `v1_statuses.py` | per-camera image statuses on a rig that sees the whole tube | statuses agree with the full search |
| V2 | `v2_shadow_edge.py` | a shake next to one camera's shadow | no lower cost in the shadow |
| V3 | `v3_line_roi.py` | line images removed from the residual with the OTF's regions | ≥ 98 % removed |
| L5 | `l5_port.py` | the C++ line detector against its Python prototype (`cases/cyl_20261007/lines.py`) | same detections, centres within 1e-3 px |
| L5b | `l5b_port_isolated.py` | diagnosis of L5: isolated objects, and the prototype against itself | — |
| L6 | `sympy_checks.py` | SymPy checks of the shape model | PROVED where SymPy can express the step |
| R5 | `r5_ablation.py` | synthetic end-to-end: R0 today, R1 + hints, R2 + Phases 2–3, R3 + Phase 4 | R1 = R0; R3 fewer ghosts and higher recall than R0 |
| reg | `r_regression.py` | the stb_improve benchmark (PINPLATE, and PINHOLE copies), module and exe | bit-identical to HEAD |

Results on the final build: see the report.

## Helpers

| file | does |
|---|---|
| `cstb.py` | paths, builds, logging |
| `stbruns.py` | runs STB on a config in a fresh process; compares two runs' results |
| `synth.py`, `synth_lines.py` | synthetic recordings through the tube with their truth (spots; line images) |
| `shape_model.py` | the astigmatic shape model (focal depths by parallax, blur lengths, the OTF table) |
| `eval_truth.py` | scores a run against the truth: recall, ghosts, duplicates, position error, track purity |

## Earlier outputs kept

| folder | holds |
|---|---|
| `before_stol13/` | F1–F3 before the hinted Newton's stop tolerance was tightened to 1e-13 |
| `before_otf/` | V2 before it wrote and used an OTF file |
| `before_p4b/` | L5 and L5b on `p4dev`, before the prototype's last changes were ported |
