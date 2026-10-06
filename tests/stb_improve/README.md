# STB speed and robustness improvements: summary and progress

Last updated: 2026-10-05, 21:15 (Phases A and B done and tested).

Start here to catch up. The plan is `2026-10-05-plan.md` (approved for Phases A and B on 5 Oct); the final report is
`2026-10-05-report.md`. Every number below comes from a file in `results/`, written by the scripts named.

**In short.** Tracking this case now takes a flat **2.5 s per frame** over 300 frames, where HEAD went from 6.4 to
16.9 s: 6.8× faster by frame 300, 3.9× less frame time, at HEAD's memory, while finding more particles (B2: active
long tracks +15 %). The slowdown was the matching search's allocations in an aging process; B3 removed it. Without
B2 the outputs are identical to HEAD's. Every check passed (section 3). Nothing is committed, and your own build is
untouched until you run `python build_local.py`.

## 1. Progress log (newest first)

- **5 Oct, night: done.** The final code (inline size 3, chosen over 9: same speed, 0.9 GB less memory) runs flat
  at 2.49 → 2.48 s per frame; a restarted process is no faster (1.01, criterion < 1.2); index checks, AddressSanitizer
  and the unit tests pass on it; `save_interval = 200` leaves tracking unchanged over 300 frames and saves 0.4 GB;
  the whole pytest suite shows no regression (242 → 250 passed, the same 4 old failures). Report written.
- **5 Oct, evening: the slowdown is found and removed** (`analyze.py p1`, `analyze.py b`, `results/p1_*`, `b_*`).
  - **Where it was:** all the growth of the time per frame was IPR's stereo matching, and almost all of it one
    round: the reduced camera set [0,2,3] (the heaviest, because camera 1, left out, has by far the fewest
    detections). Its matching time grew 10× over 300 frames while its detections and its search work fell.
  - **What it was:** the same work getting slower in an aging process. A1 (no heap for small matrices) made every
    frame 1.65× faster but kept the growth (HEAD 6.4 → 16.9 s per frame over 300 frames, A1 3.8 → 10.3 s); the
    search's own allocations (five vectors copied per step, a per-call `seen` array) kept it. With B1 and B3 the
    search no longer allocates, and the time per frame stays flat: **2.65 → 2.58 s** (6.5× faster than HEAD at
    frames 281–300, 3.8× less frame time over the 300 frames), with identical outputs.
  - **Which change did it:** B3, the search without allocations. B1 alone (caching the lines of sight) starts
    faster (3.20 s per frame) but grows like A1 (to 11.95 s at frames 281–300); with the same search work, its
    search-and-check time in round [0,2,3] grows 0.34 → 4.28 s per round, B3's falls 0.112 → 0.074 s.
  - **Restart tests:** a process restarted from frame 200 runs 1.9× (HEAD), 2.0× (A1, 9) and 2.2× (A1, 3) faster
    than the aged one on frames 201–300, then slows down again; HEAD 5.9 → 11.5 s per frame within 100 frames.
  - **B2** keeps the speed (2.57 → 2.56 s) and finds more: IPR objects +24 %, 4-camera matches at 2 px +98 %,
    active long tracks +15 % (frames 281–300).
  - S1 now also over 300 frames: HEAD, A1 and A1+B0/B1/B3/B4 identical in every frame and in all output files.
- **5 Oct, Phases A and B coded and checked; benchmarks running.**
  - Code: A1 (small matrices inside the object), A3 (`save_interval`), A4 (`config.json`), B0 (profile counters),
    B1 (tables of lines of sight and image lines), B3 (no allocations in the search), B4 (k-d tree in
    `checkRepeat`), B2 (image lines anchored in the view volume), and a fix of IPR's log line.
  - Identical results (S1) for A1, A3, A4, B0, B1, B3 and B4; B2 changes results on purpose (more matches).
  - B5 (pruning) is not needed: it takes 2.5 % of matching.

## 2. What changed in the code

| item | files | what it does | results |
|---|---|---|---|
| A1 | `inc/libMath/Matrix.h`, `src/srcMath/Matrix.hpp` | a `Matrix` with at most 3 elements (`OPENLPT_MATRIX_INLINE`, chosen over 9 on the final code: same speed, 0.9 GB less memory) keeps them inside the object: points and lines never touch the heap. `OPENLPT_MATRIX_CHECKS` (build option) checks every index. | identical |
| A3 | `inc/libSTB/Config.h`, `src/srcSTB/Config.cpp`, `src/srcSTB/STB.cpp` | `save_interval = N` (config.txt, an optional line after the fixed ones; config.json key): frames between track saves, default 500. Each save frees the lost and exited tracks. | identical (default) |
| A4 | `src/srcSTB/Config.cpp`, `inc/nlohmann/json.hpp`, `modules/stb_config.py` | `config.json`: one file with the basic and the object settings; errors name the key; unknown keys warn. The Python module reads, writes and converts both formats (`python -m modules.stb_config convert config.txt`). | identical |
| B0 | `src/srcSTB/StereoMatch.cpp` | `profile = 1` prints one `[profile]` line per IPR round: lines of sight, projections, triangulations, candidates, and the time of each matching step. | identical |
| B1 | `StereoMatch.h/.cpp` | each round first computes, in parallel, the line of sight of every 2-D observation and its image line on every other camera; the search reads these tables instead of recomputing them per candidate. | identical |
| B3 | `StereoMatch.h/.cpp` | the depth-first search and the candidate enumeration reuse per-thread buffers (no `Frame` copies, no per-call `seen` array). | identical |
| B4 | `src/srcSTB/STB.cpp` | `checkRepeat` finds the nearby track ends with a k-d tree instead of comparing every pair. | identical |
| B2 | `src/srcSTB/StereoMatch.cpp` | an image line passes through the images of the points where the line of sight enters and leaves the view volume, not 1 and 2 mm past the window. | more matches |
| log | `src/srcSTB/IPR.cpp` | the detections per camera are printed after IPR's parallel loop, in camera order (they were printed from inside it, in the threads' order and sometimes with interleaved digits). | log text only |
| build | `cmake/openLPT.cmake` | `RefractCurved` added to the install export, so the non-Python configuration (C++ tests) configures again. | — |

## 3. Results

**Speed** (`analyze.py p1` → `results/p1_benchmarks.txt`, `p1_time_per_frame.png`; frames 0–300, 26 threads):

| build | s per frame, frames 4–23 | frames 281–300 | frame time | peak memory | restarted at 200 vs aged |
|---|---|---|---|---|---|
| HEAD | 6.39 | 16.85 | 49.2 min | 2.25 GB | 1.86× faster |
| A1 (inline 9 / 3) | 3.83 / 3.91 | 10.33 / 11.01 | 29.4 / 32.3 min | 2.63 / 2.18 GB | 2.0 / 2.2× faster |
| A1 + B1 (caching only) | 3.20 | 11.95 | 31.8 min | 2.79 GB | — |
| A1 + B0, B1, B3, B4 (identical outputs) | 2.65 | 2.58 | 13.1 min | 2.67 GB | — |
| **final (inline 3)** | **2.49** | **2.48** | **12.5 min** | **2.28 GB** | **1.01 (no aging)** |

**Correctness (all passed).**
- **U1, `Matrix` storage** (`cpp/run_unit_tests.py` → `results/u1_*.txt`): 189 checks in each of 7 variants (inline
  9 and 3, plain, index-checked and under AddressSanitizer, and the default); HEAD's heap storage passes the same
  value checks. Every storage change (small ↔ large, `*=` across 9 elements, vector growth, moves of an
  `Object3D`-like holder), and 16 threads at once.
- **The repo's C++ tests:** only `test_Matrix`, `test_myMATH`, `test_ImageIO` still compile; the same result on HEAD
  and on every new build, under AddressSanitizer too (`test_myMATH` stops at its test 12 everywhere).
- **Index checks** on tracking, frames 0–10: no out-of-range index on HEAD, A1 and the final code (inline 9 and 3).
- **AddressSanitizer** on tracking, frames 0–5: no report on HEAD, A1 and the final code (inline 9 and 3).
- **S1, identical outputs** (`compare_runs.py identical` → `results/s1_*.txt`): HEAD run twice; HEAD against A1
  (9 and 3), against A1 + B1, and against A1 + B0/B1/B3/B4, over frames 0–50 and 0–300: every logged counter of every
  frame and all output files byte-identical. The final build against `b2` (the IPR log fix changes nothing else), and
  the final code at inline 9 against 3, over 300 frames.
- **C1, config** (`c1_config.py` → `results/c1_config.txt`): identical settings from `config.txt` and its converted
  `config.json` (24 groups); 13 bad or edge-case files give clear errors or warnings; a run from each file gives
  identical outputs.
- **P2, save interval** (`results/p2_save_interval.txt`; report 4.1): saves where expected, each freeing the lost
  tracks; over 300 frames at 200 the tracking is unchanged (active tracks byte-identical, the lost tracks of the two
  saves add up exactly) and peak memory is 0.4 GB lower.
- **pytest:** the subset `tests/test_stb_improve.py` 19 passed; the whole suite (`results/pytest_full_*.txt`) shows no
  regression (HEAD's build 242 passed, the final 250; the same 4 failures from before on both).

**B2, the image lines** (M1, `m1_epipolar.py` → `results/m1_epipolar.txt`): with exact pixels, the share of
4-camera particles whose lines all lie within 2 px rises from 35.9 % (HEAD) to 91.8 %, within 3 px from 54.4 % to
100 %. In tracking (300 frames): IPR objects +24 %, 4-camera matches at 2 px +98 %, active long tracks +15 %, at the
same speed.

**Matching** (`analyze.py b` → `results/b_matching.txt`): matching time per frame 7.04 s (HEAD) → 0.64 s (final, with
B2); `checkRepeat` 0.56 → 0.002 s. B1 cut the lines of sight per IPR round from 678,000 to 60,000 and the projections
from 3.2 million to 275,000.

## 4. How to rerun

```sh
python tests/stb_improve/prepare_data.py                 # local copies of frames 0-310 (once)
python tests/stb_improve/build_variant.py NAME [--def OPENLPT_MATRIX_INLINE=3] [--asan] [--mode tests]
python tests/stb_improve/cpp/run_unit_tests.py           # U1
python tests/stb_improve/stbrun.py RUN --build NAME --frames 0 300 [--config-extra "profile = 1"]
python tests/stb_improve/compare_runs.py identical RUN_A RUN_B
python tests/stb_improve/c1_config.py --build NAME       # C1
python tests/stb_improve/m1_epipolar.py --builds base b2 # M1
python tests/stb_improve/analyze.py s1|p2|p1|b
bash tests/stb_improve/queue1.sh                         # the runs of Phases A and B, in order
bash tests/stb_improve/queue2.sh                         # the final code: B1 alone, inline 9 vs 3, save interval, restart, ASan
bash tests/stb_improve/queue3.sh                         # the shipped configuration (inline 3): checks, ASan, unit tests
```
The python is the `refract` env's (`C:\Users\zcloc\.conda\envs\refract\python.exe`).
