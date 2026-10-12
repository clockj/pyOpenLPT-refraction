# Perturbed restarts of the final fit on multi-window rigs: plan (R1–R6)

Date: 2026-10-02. Status: **approved 2026-10-03 ("go ahead and implement the plan") and carried out; R4 failed its
criterion.** A change to how a restart is accepted was then approved ("ok, go ahead and compare"), implemented and
compared (R7). Results, deviations and the comparison: `2026-10-03-perturbed-starts-report.md`. It follows the diagnosis of W5 seed 2000
(`d4_seed2000.py`; README progress log, 2026-10-02) and your request: "yes, write the plan for the perturbed starts".

Scope:
- Only `modules/camera_calibration/curved_calibration/plate_wand.py` changes, plus one report line in
  `plate_wand_io.py`. The tests go in this folder.
- Nothing in `D:\Refraction\paper`, the C++ library, the GUI, the case folder or `cylinder_wand*.py`.

## 1. The finding

- **W5 seed 2000:** two windows, two cameras each, 1 px noise. The answer from scratch is 0.3 % above the global
  minimum in RMS (50 px² more in squared residuals). Cameras 3 and 4's f are off by +2.0 % and −2.1 % (3.8σ, 2.8σ).
- **The cause is the start** (`d4_seed2000_out.txt`):
  - each window's pair is calibrated on its own, then the pairs are joined;
  - alone, a pair is 1.2–2.5× less precise than the full rig, so the joined start is 2–3.4 full-rig σ off;
  - the final fit then settles in a narrow wrong basin.
- **That basin is narrow.** The final fit, restarted a quarter of the way (or more) from the wrong answer towards the
  global minimum, reaches the global minimum.
- **How common it is, is unknown.**
  - At 0.1 px, all 8 W5 rigs with two or more cameras on every window reached the global minimum.
  - At 1 px, 2 of 4 did, and one more gave an equivalent answer.
  - R4 measures it on 30 such rigs.

## 2. The change: a few perturbed restarts around the answer

After the final fit, try a few restarts perturbed around its answer, and keep a better minimum if one turns up.
1. **Draw K perturbations** of the answer's cameras and windows from the answer's own covariance, scaled by s:
   - Δ = s · L z, where Σ = L Lᵀ is the answer's covariance (camera 1's pose held) and z ~ N(0, I);
   - a draw that puts a camera on the water side of its window is redrawn, up to 10 times.
2. **Screen each draw.** Re-triangulate the frames, then run one Levenberg–Marquardt fit with the answer's active
   observations and noise weights held fixed. With the noise model fixed, the costs compare exactly, as likelihoods.
3. **Fully fit the best draw,** if it lowers the cost by more than 1e-6 relative. That is the full final fit, with
   rejection and noise weights re-estimated. Its result joins the other starts' results, and the existing rule chooses
   among them.
4. **Repeat once** around the new answer if it improved: at most two rounds.

**Where it applies:**
- Rigs that use more than one window: the window-group and placement starts, and their whole-rig fallback.
- One-window rigs are unchanged (37/37 in W5). `perturb_all=True` turns it on for them too.

**Why around the answer, not the joined start (as I first suggested):**
- The answer's covariance gives the perturbation its size and shape directly; the joined start has no covariance of
  its own.
- Seed 2000's answer is about 4σ from the global minimum, and the global basin begins about a quarter of the way.
- R1 compares the two centres anyway.

**Parameters:** `perturb_starts=K` (0 turns it off) and `perturb_scale=s`, with defaults set from R1:
- s: the scale with the best per-draw success p on seed 2000;
- K: the smallest number with at least a 95 % chance of success on seed 2000, 1 − (1 − p)^K ≥ 0.95, and at most 8.

**Further details:**
- **Deterministic:** the draws' seed comes from the data (CRC32 of the detections), so a rerun gives the same answer.
- **Reported:** `info["starts"]["perturbed"]` holds the draws, the scale, the screened costs, whether the answer
  improved, the RMS before and after, and the seconds. `calibration_report.txt` gets one line.
- **Cost:**
  - a screened draw costs roughly a third of a final fit;
  - W5's final fits on multi-window rigs take a median 6.9 s (max 52 s, machine fully loaded);
  - K = 6 would add about 15 s at the median (+12 % of 121 s), and about 2 min in the worst case.

## 3. Tests (R1–R6), with criteria

| test | what | criterion |
|---|---|---|
| R1 | sizing on seed 2000, before any code change: from its wrong answer and from the joined start, 20 draws each at s = 1, 2, 3, 4, each through the full final fit | per-draw success p(s) reported; sets s and K. If p < 0.2 at every scale: stop and report, since the remedy would need rethinking |
| R2 | the draws alone (no fitting): 2,000 draws around rig B's answer | sample covariance matches s²Σ (each σ within 5 %); camera 1 untouched; the same data give the same draws |
| R3 | W5 again on the new code (60 rigs) | seed 2000 reaches the global minimum; no rig ends worse than before; one-window rigs identical (RMS to 1e-12); median time increase on multi-window rigs ≤ 25 % |
| R4 | 30 new random rigs with two or more cameras on every window (2–3 windows), at 0.5 and 1 px, without and with the restarts | misses before and after, per rig; no rig worse; worth keeping if at least half of the misses before are fixed |
| R5 | W7 again (rigs A and B, bad data, from scratch) | every answer still at its data's minimum; its distance from the clean answer unchanged or smaller |
| R6 | pytest: `tests/test_plate_wand.py`, plus a fast test of the draws and a slow one (seed 2000 from scratch reaches the global minimum); the full suite | all pass; no new failures |

"Worse" means a different minimum with a higher RMS. A restart that finds a minimum below the truth-started fit's
counts as fine, as in W5's "equivalent".

**Risks:**
- **The narrow basin may be hit rarely.** R1 decides before any code changes.
- **The selection rule prefers the answer that keeps more observations** (within 1.5× of the best RMS). A restart only
  enters the pool after lowering the cost on the same observations, and R3 and R4 check that no rig ends worse.
- **Time on large rigs.** K is capped at 8, and only one draw per round gets the full fit.

## 4. Deliverables

- The code change (`plate_wand.py`, and the report line in `plate_wand_io.py`).
- `r1_sizing.py`, `r2_draws.py` and `r4_group_rigs.py`, with their outputs.
- W5 and W7 rerun; the previous runs kept in `runs_v4/`.
- `2026-10-03-perturbed-starts-report.md`.
- The README updated (progress log, conclusions, open items), and the figures page rebuilt from the new W5 data and
  republished.

Cost: about 30–45 min of computing in parallel. R4 is 120 calibrations and R3 is W5's 60.

**Stop points:** approval covers R1–R6 and the code change. I stop and ask if R1 rules the remedy out, or if a
criterion fails in a way that needs a design change.
