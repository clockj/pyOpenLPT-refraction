# How the intermediate builds' sources were made

The Phase B changes were applied to the working tree by these scripts, in this order; each benchmark build of
`build-stb/` was built from the tree after one of them (see its `out/build_info.json`, which lists the changed files
and line counts):

| build | source | 
|---|---|
| `base` | HEAD 14ad4b6 (worktree `build-stb/src-base`, plus the `cmake/openLPT.cmake` export fix for the C++ tests) |
| `a1-9`, `a1-3`, `a1-9-check`, `asan-a1` | HEAD + A1 + the index-check option (`out/source.diff`) |
| `b0` | HEAD + A1 + A3 + A4 + `1_b0_profile.py` (`out/source.diff`) |
| `b1` | `b0` + `2_b1_tables.py` + `3_b1_bindings.py` |
| `b14` | `b1` + `4_b4_kdtree.py` |
| `b134` | `b14` + `5_b3_scratch.py` |
| `b2` | `b134` + `6_b2_volume_lines.py` |
| `final`, `final-check`, `asan-final` | `b2` + IPR's log line printed after its parallel loop |

The scripts replace exact text and assert each match, so they apply only to the tree state they were written for
(absolute paths to `D:\Refraction\pyOpenLPT`). The working tree holds the final state; `git diff` shows it whole.
Until 5 Oct evening `build_variant.py` read `git diff` in the Windows code page, which cannot decode the UTF-8
comments in `StereoMatch.cpp`, so the builds from `b1` on have no `source.diff` of their own; this table and the
scripts take its place.
