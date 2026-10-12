"""D3 (diagnosis): how the dot-fixed residual pattern changes through the traverse.

    python tests/d3_pattern_vs_z.py STATE.pkl [tag]

The pooled pattern P(dot) is each dot's mean residual over all planes (D1).  For every plane k the residual is
projected on it, r_k(dot) ~ alpha_k P(dot) (u and v separately), to see whether the pattern keeps its shape and how
its strength follows z: a board error in mm would keep alpha ~ 1 (within the ~10 % change of magnification), a
detection bias tied to the blur would follow the blur (largest at the ends of the traverse, where the dots are
most out of focus), a model error would change smoothly in another way.  The measured edge blur of each plane
(10-90 % widths of the dot outlines, from the detector's darkness profiles) is printed beside it.
Writes d3_<tag>_out.txt beside this script.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import run_calibration as RC                  # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def main():
    st = RC.load_state(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "fit"
    obs, r = st["obs"], st["resid"]
    keys = list(zip(obs.i.astype(int), obs.j.astype(int)))
    P = {}
    for key in set(keys):
        m = np.array([k == key for k in keys])
        P[key] = r[m].mean(axis=0)
    Pr = np.array([P[k] for k in keys])
    say(f"D3 dot pattern against z ({sys.argv[1]}, model {st['info']['model']}); pattern rms "
        f"({np.sqrt(np.mean(Pr[:, 0] ** 2)):.3f}, {np.sqrt(np.mean(Pr[:, 1] ** 2)):.3f}) px")
    say("   z    alpha_u  alpha_v   rms     rms - P   rms - alpha P   corr_u  corr_v")
    for k in range(obs.n_planes):
        m = obs.plane == k
        a_u = float(Pr[m, 0] @ r[m, 0] / (Pr[m, 0] @ Pr[m, 0]))
        a_v = float(Pr[m, 1] @ r[m, 1] / (Pr[m, 1] @ Pr[m, 1]))
        rm = lambda x: float(np.sqrt(np.mean(np.sum(x ** 2, axis=1))))      # noqa: E731
        fit_a = np.column_stack([a_u * Pr[m, 0], a_v * Pr[m, 1]])
        cu = np.corrcoef(Pr[m, 0], r[m, 0])[0, 1]
        cv = np.corrcoef(Pr[m, 1], r[m, 1])[0, 1]
        say(f"  {obs.z_nom[k]:4.0f}   {a_u:6.3f}   {a_v:6.3f}   {rm(r[m]):.3f}   {rm(r[m] - Pr[m]):.3f}     "
            f"{rm(r[m] - fit_a):.3f}        {cu:+.2f}   {cv:+.2f}")
    with open(os.path.join(HERE, f"d3_{tag}_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
