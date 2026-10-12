"""Run several fits of the case in parallel, each a run_calibration.py call in its own process (one BLAS thread
each), its state saved to tests/states/<job>.pkl and its console to tests/logs/<job>.txt.

    python tests/run_batch.py t3            # T3: 5 interleaved subsets and the two halves, k1 held at 0
    python tests/run_batch.py t3k           # the same with k1 fitted
    python tests/run_batch.py t5 t6         # T5 (hints) and T6 (water index, wall thickness) on subset 0
Writes tests/run_batch_<groups>.txt: each job's command, exit code and time.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
PY = sys.executable

SUBSETS = {f"s{k}": ["--every", "5", "--offset", str(k)] for k in range(5)}
SUBSETS["h1"] = ["--first", "0.5"]
SUBSETS["h2"] = ["--last", "0.5"]
GROUPS = {
    "t3": {f"t3_{s}": a for s, a in SUBSETS.items()},
    "t3k": {f"t3k_{s}": a + ["--fit-k1"] for s, a in SUBSETS.items()},
    "t5": {"t5_hints_s0": SUBSETS["s0"] + ["--hints", "photo"]},
    "t6": {"t6_nw1.33_s0": SUBSETS["s0"] + ["--n-water", "1.33"],
           "t6_t12.4_s0": SUBSETS["s0"] + ["--thickness", "12.4"],
           "t6_t13.0_s0": SUBSETS["s0"] + ["--thickness", "13.0"]},
    # T6 again with k1 fitted, on subset s2 (the k1-held T6 runs on s0 all ended in t3_s0's wrong minimum)
    "t6k": {"t6k_nw1.33_s2": SUBSETS["s2"] + ["--fit-k1", "--n-water", "1.33"],
            "t6k_t12.4_s2": SUBSETS["s2"] + ["--fit-k1", "--thickness", "12.4"],
            "t6k_t13.0_s2": SUBSETS["s2"] + ["--fit-k1", "--thickness", "13.0"],
            "t5k_hints_s2": SUBSETS["s2"] + ["--fit-k1", "--hints", "photo"]},
    # T5 on the second half with k1 fitted (where the plain fit t3k_h2 ended in a wrong minimum): the photo hints, and
    # hints from the GUI's calib_3
    "t5h2": {"t5k_photo_h2": SUBSETS["h2"] + ["--fit-k1", "--hints", "photo"],
             "t5k_gui3_h2": SUBSETS["h2"] + ["--fit-k1", "--hints", "gui:calib_3"]},
    # D3: the detections whose ball crosses the image border dropped (k1 fitted)
    "d3": {"d3_edge_s4k": SUBSETS["s4"] + ["--fit-k1", "--edge-margin", "2"],
           "d3_edge_s0k": SUBSETS["s0"] + ["--fit-k1", "--edge-margin", "2"]},
}


def run(job: str, args) -> tuple:
    os.makedirs(os.path.join(HERE, "states"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    cmd = [PY, os.path.join(CASE, "run_calibration.py"), *args, "--out", "none",
           "--state", os.path.join(HERE, "states", f"{job}.pkl"), "--log", os.path.join(HERE, "logs", f"{job}.txt")]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=CASE, env=env, capture_output=True, text=True)
    if p.returncode:
        with open(os.path.join(HERE, "logs", f"{job}_stderr.txt"), "w", encoding="utf-8") as fh:
            fh.write(p.stderr)
    return job, " ".join(args), p.returncode, time.time() - t0


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:                     # --only job1,job2: just those jobs of the named groups
        k = args.index("--only")
        only = set(args[k + 1].split(","))
        args = args[:k] + args[k + 2:]
    groups = args or ["t3"]
    jobs = {}
    for g in groups:
        jobs.update(GROUPS[g])
    if only is not None:
        jobs = {j: a for j, a in jobs.items() if j in only}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        results = list(ex.map(lambda kv: run(*kv), jobs.items()))
    lines = [f"run_batch {' '.join(groups)}: {len(jobs)} jobs in parallel, {time.time() - t0:.0f} s"]
    for job, args, rc, secs in results:
        lines.append(f"  {job}: exit {rc}, {secs:.0f} s  ({args})")
    tag = "_".join(groups) + ("_only" if only is not None else "")
    with open(os.path.join(HERE, f"run_batch_{tag}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
