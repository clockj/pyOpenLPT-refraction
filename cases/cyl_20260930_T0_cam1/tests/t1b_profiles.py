"""T1b (diagnostic for T1): edge profiles of the same dots at three planes, to see how the blur and the rim change
with z.  Prints the darkness N along the vertical and horizontal lines through the centre of a few dots, measures
the 10-90 % edge widths on each side, and saves zoomed crops beside this script (t1b_crop_zXX_dotI_J.png).
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
import case                                   # noqa: E402
import dots as D                              # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def edge_width(prof, r, outward=True):
    """10-90 % width of the outer edge of a profile sampled at radii r (darkness falling outward)."""
    k_pk = int(np.argmax(prof))
    pk = prof[k_pk]
    tail = prof[k_pk:]
    rr = r[k_pk:]

    def cross(level):
        j = np.where(tail < level)[0]
        if len(j) == 0:
            return np.nan
        j = j[0]
        p0, p1 = tail[j - 1], tail[j]
        return rr[j - 1] + (p0 - level) / (p0 - p1) * (rr[j] - rr[j - 1])
    return cross(0.1 * pk) - cross(0.9 * pk), pk


def main():
    cfg = case.load_config()
    det = cfg["detect"]
    t1 = json.load(open(os.path.join(HERE, "t1_dots.json")))
    want = [(3, 0), (6, 1), (0, 0)]
    for lab in (0, 18, 38):                      # file labels z_0mm, z_18mm, z_38mm
        pl = next(p for p in t1 if p.get("label", p["z"]) == lab)
        img = case.read_image(pl["image"])
        N, bg, seg = D.normalised_darkness(img, det)
        say(f"\n=== {os.path.basename(pl['image'])}: z = {pl['z']:.2f} mm ===")
        for (i, j) in want:
            d = next(dd for dd in pl["dots"] if dd["i"] == i and dd["j"] == j and dd["ok"])
            u, v = d["u"], d["v"]
            r = np.arange(-45, 45.01, 0.5)
            vert = cv2.remap(N, np.full((1, len(r)), u, np.float32), (v + r).astype(np.float32)[None, :],
                             cv2.INTER_LINEAR)[0]
            hori = cv2.remap(N, (u + r).astype(np.float32)[None, :], np.full((1, len(r)), v, np.float32),
                             cv2.INTER_LINEAR)[0]
            # outward profiles: top (v decreasing), bottom, left, right
            top, bot = vert[r <= 0][::-1], vert[r >= 0]
            lef, rig = hori[r <= 0][::-1], hori[r >= 0]
            rp = np.arange(0, 45.01, 0.5)
            wt, pt = edge_width(top, rp)
            wb, pb = edge_width(bot, rp)
            wl, pl_ = edge_width(lef, rp)
            wr, pr = edge_width(rig, rp)
            say(f"dot ({i},{j}) at ({u:.1f},{v:.1f}): a,b = {d['a']:.1f},{d['b']:.1f}; "
                f"core {d['core']:.3f}; peaks top {pt:.3f} bottom {pb:.3f} left {pl_:.3f} right {pr:.3f}; "
                f"10-90% widths top {wt:.1f} bottom {wb:.1f} left {wl:.1f} right {wr:.1f} px; "
                f"half-height minus gradient centre ({d['u'] - d['u_grad']:+.2f}, {d['v'] - d['v_grad']:+.2f}), "
                f"minus iso ({d['u'] - d['u_iso']:+.2f}, {d['v'] - d['v_iso']:+.2f})")
            say("   vertical   (v-45..v+45 step 2):", " ".join(f"{x:.2f}" for x in vert[::4]))
            say("   horizontal (u-45..u+45 step 2):", " ".join(f"{x:.2f}" for x in hori[::4]))
            x0, y0 = int(u) - 50, int(v) - 40
            crop = img[max(y0, 0):y0 + 80, max(x0, 0):x0 + 100]
            lo, hi = np.percentile(crop, [1, 99.5])
            b8 = np.clip((crop - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(HERE, f"t1b_crop_z{z:02d}_dot{i}_{j}.png"),
                        cv2.resize(b8, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST))
    with open(os.path.join(HERE, "t1b_out.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
