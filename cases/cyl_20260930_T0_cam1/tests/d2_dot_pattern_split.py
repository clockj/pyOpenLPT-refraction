"""D2 (diagnosis): is the dot-fixed residual pattern of D1 real?

    python tests/d2_dot_pattern_split.py STATE.pkl [tag]

Computes each dot's mean residual separately from the even planes and from the odd planes of a fit, and compares
them.  A board error (dot positions off the nominal grid, or a per-dot centre bias) is the same in both halves; a
coincidence of noise is not.  Also splits by the near and far half of the traverse.
Writes d2_<tag>_out.txt beside this script.
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


def dot_means(obs, r, mask):
    out = {}
    for key in set(zip(obs.i[mask].astype(int), obs.j[mask].astype(int))):
        m = mask & (obs.i == key[0]) & (obs.j == key[1])
        out[key] = r[m].mean(axis=0)
    return out


def compare(a, b, label):
    keys = sorted(set(a) & set(b))
    A = np.array([a[k] for k in keys])
    B = np.array([b[k] for k in keys])
    cu = np.corrcoef(A[:, 0], B[:, 0])[0, 1]
    cv = np.corrcoef(A[:, 1], B[:, 1])[0, 1]
    d = A - B
    say(f"  {label}: {len(keys)} dots; correlation du {cu:+.3f}, dv {cv:+.3f}; rms of the pattern "
        f"({np.sqrt(np.mean(A[:, 0] ** 2)):.3f}, {np.sqrt(np.mean(A[:, 1] ** 2)):.3f}) and "
        f"({np.sqrt(np.mean(B[:, 0] ** 2)):.3f}, {np.sqrt(np.mean(B[:, 1] ** 2)):.3f}) px; rms of their "
        f"difference ({np.sqrt(np.mean(d[:, 0] ** 2)):.3f}, {np.sqrt(np.mean(d[:, 1] ** 2)):.3f}) px")


def main():
    st = RC.load_state(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "fit"
    obs, r = st["obs"], st["resid"]
    say(f"D2 dot-fixed pattern, split test ({sys.argv[1]}, model {st['info']['model']})")
    even = obs.plane % 2 == 0
    compare(dot_means(obs, r, even), dot_means(obs, r, ~even), "even vs odd planes")
    near = obs.z_nom[obs.plane] < 19
    compare(dot_means(obs, r, near), dot_means(obs, r, ~near), "z < 19 vs z > 19 (far vs near half)")
    with open(os.path.join(HERE, f"d2_{tag}_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
