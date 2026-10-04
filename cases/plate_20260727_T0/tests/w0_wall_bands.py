"""W0 (2026-10-03-walls-plan.md, sections 1-3): does every saved fit put the wand outside the channel?

For each saved fit (tests/states/*.pkl): the depth of its wand ends (the frames it kept) below each wall's water-side
surface, and the band of shifts of each wall along its normal (positive: away from its cameras, towards the wand) that
would put every end inside a channel of the given inside widths: 100 mm across wall 1 (C1, C2: the long side) and
150 mm across wall 2 (C3, C4: the short side).  An end must not cross a wall (shift <= the smallest depth) nor the far
wall (shift >= the largest depth - width).  An empty band means the wand's spread is wider than the channel.

    python tests/w0_wall_bands.py [width1 width2]
Writes w0_wall_bands_out.txt beside this script.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import case  # noqa: E402


def main():
    widths = [float(v) for v in sys.argv[1:3]] if len(sys.argv) >= 3 else [100.0, 150.0]
    log = case.Tee(os.path.join(HERE, "w0_wall_bands_out.txt"))
    log(f"W0: the wand's depth below each wall's water side in every saved fit, and the shift of each wall that puts "
        f"the wand inside {widths[1]:.0f} x {widths[0]:.0f} mm (inside widths across wall 1 and wall 2: "
        f"{widths[0]:.0f}, {widths[1]:.0f} mm)")
    log("  fit                     k1   RMS px   depth below wall 1   depth below wall 2   wall 1 shift band   "
        "wall 2 shift band (centred)")
    for p in sorted(glob.glob(os.path.join(HERE, "states", "*.pkl"))):
        name = os.path.splitext(os.path.basename(p))[0]
        st = case.load_state(p)
        fr = case.pw.Frames(np.asarray(st["frames_m"]), np.asarray(st["frames_w"]))
        XA, XB = fr.ends(st["L"])
        act = np.zeros(len(XA), bool)
        act[np.asarray(st["obs_frame"])[np.asarray(st["active"], bool)]] = True
        P = np.vstack([XA[act], XB[act]])
        cols = []
        for w, width in zip(st["rig"].windows, widths):
            d = P @ w.n - w.far_d
            cols.append((d.min(), d.max(), d.max() - width, d.min(), (d.min() + d.max() - width) / 2))
        k1 = "fit " if st["layout"]["fit_k1"] else "held"
        (a1, b1, lo1, hi1, _), (a2, b2, lo2, hi2, c2) = cols
        band1 = f"{lo1:+6.1f} .. {hi1:+6.1f}" if lo1 <= hi1 else "  none (too wide) "
        band2 = f"{lo2:+6.1f} .. {hi2:+6.1f} ({c2:+6.1f})" if lo2 <= hi2 else "  none (too wide)        "
        log(f"  {name:22s} {k1} {st['rms']:7.4f}   {a1:7.1f} .. {b1:6.1f}     {a2:7.1f} .. {b2:6.1f}     {band1}   {band2}")
    log.close()


if __name__ == "__main__":
    main()
