"""Sizing for a scan of a weak window's normal (D8, H4): how long one held fit takes, on all frames and on a subset.

On W5 seeds 1001, 1034 and 1045 (0.1 px): from the truth-started fit, the weak window's normal turned by 3 deg and
held, everything else fitted (plate_wand.solve with held_normals, equal weights) -- on all frames, and on a 300-frame
subset (cylinder_wand's _frame_subset, as the pinhole start uses).  Reports the seconds and the held fit's RMS.
Output: p_held_fit_timing_out.txt.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CASES = [(1001, 0.1, 1), (1034, 0.1, 2), (1045, 0.1, 0)]


def main():
    import prigs
    import plate_wand as pw
    from prigs import pm
    from h1_hints import case
    from d8_profile import held_fit
    log = prigs.Tee(HERE / "p_held_fit_timing_out.txt")
    log("One held fit (the weak window's normal turned 3 deg from the truth and held): seconds, all frames and 300")
    for seed, noise, w in CASES:
        rig_t, data = case(seed, noise)
        truth = pw.Rig([x.copy() for x in rig_t.wins], rig_t.cam_win)
        t0 = time.time()
        ref = pw.solve(data, [c.copy() for c in rig_t.cams], truth, noise_weights="none")
        t_ref = time.time() - t0
        n_new = pm.retract_normal(rig_t.wins[w].n, (math.radians(3.0), 0.0))
        t0 = time.time()
        s_all = held_fit(pw, pm, data, ref.cams, ref.rig, w, n_new)
        t_all = time.time() - t0
        obs = pw.observations(data)
        sub = pw._frame_subset(obs, 300)
        keep = np.isin(np.arange(data.uv.shape[0]), sub.frame_ids)
        d_sub = pw.WandData(data.uv[keep], data.L, data.sizes)
        t0 = time.time()
        s_sub = held_fit(pw, pm, d_sub, ref.cams, ref.rig, w, n_new)
        t_sub = time.time() - t0
        log(f"seed {seed}: {int(obs.n_frames)} frames, {len(obs)} observations; truth-started fit {t_ref:.1f} s; held fit "
            f"all frames {t_all:.1f} s (rms {s_all.rms:.6f}, {len(s_all.history)} iterations), 300 frames {t_sub:.1f} s "
            f"(rms {s_sub.rms:.6f})")


if __name__ == "__main__":
    main()
