"""T0b: checks of the fitting machinery added for the board, wall, plane-shift and stage-scale blocks.

  * the B-spline basis of the wall field is a partition of unity over the image width and matches the piecewise
    definition;
  * the shift basis is orthonormal and orthogonal to a common shift and to a shift linear in z;
  * the grouped Jacobian (plane offsets perturbed together; plane shifts and board corrections perturbed per
    coordinate on the rig) equals the brute-force one (every parameter perturbed on its own) at a fitted rig with
    plane shifts, board corrections, a wall field and the stage scale free.
Passes if the basis sums to 1 within 1e-12, the shift basis is exact to 1e-12, and the Jacobians agree within 1e-6
relative to the largest entry.
Writes t0b_out.txt beside this script.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402
import run_calibration as RC                  # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def piecewise(x, lo, hi, step):
    knots = np.arange(lo - 3 * step, hi + 3.5 * step, step)
    n = len(knots) - 4
    B = np.zeros((len(x), n))
    for q in range(n):
        t = (x - knots[q]) / step
        b = np.zeros_like(t)
        m = (t >= 0) & (t < 1)
        b[m] = t[m] ** 3 / 6
        m = (t >= 1) & (t < 2)
        b[m] = (-3 * t[m] ** 3 + 12 * t[m] ** 2 - 12 * t[m] + 4) / 6
        m = (t >= 2) & (t < 3)
        b[m] = (3 * t[m] ** 3 - 24 * t[m] ** 2 + 60 * t[m] - 44) / 6
        m = (t >= 3) & (t < 4)
        b[m] = (4 - t[m]) ** 3 / 6
        B[:, q] = b
    return B


def main():
    x = np.linspace(0.0, 1279.99, 5001)
    B = CF.bspline_basis(x, *CF.WALL_SPEC)
    P = piecewise(x, *CF.WALL_SPEC)
    pu = float(np.max(np.abs(B.sum(axis=1) - 1)))
    dp = float(np.max(np.abs(B - P)))
    say(f"T0b: B-spline basis: max |sum - 1| = {pu:.1e}; max |vectorised - piecewise| = {dp:.1e}")

    cfg = case.load_config()
    st = RC.load_state(sys.argv[1] if len(sys.argv) > 1 else os.path.join(cfg["out_dir"], "fit_state.pkl"))
    obs = st["obs"]
    Ns = CF.shift_basis(obs.z_nom)
    A = np.column_stack([np.ones(obs.n_planes), obs.z_nom])
    e_ns = max(float(np.max(np.abs(Ns.T @ Ns - np.eye(Ns.shape[1])))), float(np.max(np.abs(A.T @ Ns))))
    say(f"  shift basis {Ns.shape}: max |N'N - I|, |[1 z]' N| = {e_ns:.1e}")
    blocks = CF.MODELS["M2bw"] + ("zs", "shift")
    rig = CF.prepare(st["rig"], obs, blocks)
    rng = np.random.default_rng(3)
    rig.board = rig.board + rng.normal(0, 0.01, rig.board.shape)
    rig.wall = rig.wall + CF.wall_basis_null(rig.wall_spec) @ rng.normal(0, 2e-4, CF.wall_basis_null(
        rig.wall_spec).shape[1])
    rig.shift = rig.shift + Ns @ rng.normal(0, 0.01, (Ns.shape[1], 2))
    rig.zscale = 1.003
    layout = CF.Layout(blocks, obs.n_planes, rig, obs.pitch, z=obs.z_nom)

    def fun(xx):
        return CF.forward_residual(layout.apply(xx, rig), obs)
    x0 = np.zeros(layout.size)
    Jg = CF._jacobian(fun, x0, layout, obs, rig, lambda rg: CF.forward_residual(rg, obs))
    Jb = CF._jacobian(fun, x0, layout, obs)               # no rig_fun: board columns one by one
    # the dz block is grouped in both; check it separately one column at a time
    h = layout.steps()
    dz = layout.slices["dz"]
    Jd = np.zeros((Jg.shape[0], dz.stop - dz.start))
    for q in range(dz.start, dz.stop):
        xp, xm = x0.copy(), x0.copy()
        xp[q] += h[q]
        xm[q] -= h[q]
        Jd[:, q - dz.start] = (fun(xp) - fun(xm)) / (2 * h[q])
    scale = float(np.max(np.abs(Jb)))
    e_all = float(np.max(np.abs(Jg - Jb))) / scale
    e_dz = float(np.max(np.abs(Jg[:, dz] - Jd))) / scale
    bd = layout.slices["board"]
    e_bd = float(np.max(np.abs(Jg[:, bd] - Jb[:, bd]))) / scale
    sh = layout.slices["shift"]
    e_sh = float(np.max(np.abs(Jg[:, sh] - Jb[:, sh]))) / scale
    say(f"  Jacobian {Jg.shape} (blocks {'+'.join(layout.blocks)}): grouped vs brute force, max difference / max "
        f"entry: all {e_all:.1e}, board block {e_bd:.1e}, plane shifts {e_sh:.1e}, plane offsets {e_dz:.1e}")
    ok = pu < 1e-12 and dp < 1e-12 and e_ns < 1e-12 and max(e_all, e_dz) < 1e-6
    say("RESULT: " + ("PASS" if ok else "FAIL"))
    with open(os.path.join(HERE, "t0b_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
