"""README figures for this branch's calibrations through refractive windows, drawn from two real calibrations.

cylinder_calibration.png - run 2026-10-08: three cameras through a water-filled acrylic tube (camera model PINCYL).
    The setup; the calibrated rig to scale (the tube, each camera's lines of sight bent at the walls, the wand's
    ends); the swirling flow tracked through the wall with this calibration, coloured by speed; and the wand's length
    from the calibrated cameras, frame by frame.
plate_calibration.png - run 2026-07-27: four cameras through two perpendicular flat walls of a water tank (PINPLATE).
    The calibrated rig to scale, and the channel's corner seen from the water: the walls, each camera's field of view
    refracted into the water, and the wand's ends.

Data, all in this repository:
    cases/cyl_wand_paper/experiment/experiment_data_20261008.npz and experiment_numbers_20261008.json, written by
        experiment_numbers_20261008.py from the case cases/cyl_20261008 (calibration and tracking);
    cases/plate_20260727_T0/tests/rig_3d.json (rig_3d.py) and w4c_delivery.json (w4_delivery.py), for the chosen
        calibration calib_plate_wand_ref_centre;
    assets/readme/setup_cylinder.jpg, the authors' photograph, cropped and shrunk once by --photo.
The script prints every number README.md quotes.

    python assets/readme/make_readme_figures.py                  # the refract env (numpy, matplotlib, Pillow)
    python assets/readme/make_readme_figures.py --photo SRC.jpg  # first, from the original photograph
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.cm import ScalarMappable                            # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize    # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from PIL import Image                                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CYL_NPZ = os.path.join(ROOT, "cases", "cyl_wand_paper", "experiment", "experiment_data_20261008.npz")
CYL_JSON = os.path.join(ROOT, "cases", "cyl_wand_paper", "experiment", "experiment_numbers_20261008.json")
PLATE_RIG = os.path.join(ROOT, "cases", "plate_20260727_T0", "tests", "rig_3d.json")
PLATE_NUM = os.path.join(ROOT, "cases", "plate_20260727_T0", "tests", "w4c_delivery.json")
PHOTO = os.path.join(HERE, "setup_cylinder.jpg")
PHOTO_CROP = (0, 130, 4032, 2818)       # the paper's crop of the photograph (3:2)
PHOTO_WIDTH = 1500

BG = "#0d1117"                           # GitHub's dark page
INK = "#e6edf3"
MUTED = "#8b949e"
CAM = ["#58a6ff", "#ff8b4c", "#3fd68f", "#e3b341"]   # cameras 1-4
GLASS = "#9ecbff"
SPEED = LinearSegmentedColormap.from_list("speed", ["#4263eb", "#22b8cf", "#94d82d", "#fcc419", "#ff6b6b"])
DPI = 150

# the views (elevation, azimuth in degrees) and the boxes drawn (mm)
CYL_VIEW, CYL_BOX, CYL_TUBE_Z = (28, 18), ((-130, -130, -240), (900, 660, 440)), (-230.0, 430.0)
TRK_VIEW, TRK_BOX = (52, -40), ((-36, -32, -24), (30, 30, 22))
RIG_VIEW, RIG_BOX = (38, -48), ((-960, -1080, -300), (240, 160, 300))
CORNER_VIEW, CORNER_BOX = (28, 42), ((-20, -20, -95), (190, 135, 75))


def nested(path):
    """experiment_numbers_*.json keys are "group.name": group them."""
    out = {}
    for k, v in json.load(open(path)).items():
        g, _, name = k.partition(".")
        out.setdefault(g, {})[name] = v
    return out


def axes3d(fig, rect, view, box, zoom=1.0):
    ax = fig.add_axes(rect, projection="3d")
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.computed_zorder = False
    lo, hi = (np.asarray(b, float) for b in box)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=zoom)
    ax.view_init(elev=view[0], azim=view[1])
    return ax


def label(fig, x, y, text, sub=None):
    fig.text(x, y, text, color=INK, fontsize=13, fontweight="bold", va="top")
    if sub:
        fig.text(x, y - 0.034, sub, color=MUTED, fontsize=9.5, va="top")


def tube(ax, R, z0, z1, alpha, zorder, n=120):
    t = np.linspace(0, 2 * np.pi, n)
    T, Z = np.meshgrid(t, [z0, z1])
    ax.plot_surface(R * np.cos(T), R * np.sin(T), Z, color=GLASS, alpha=alpha, linewidth=0, shade=False,
                    zorder=zorder)
    for z in (z0, z1):
        ax.plot(R * np.cos(t), R * np.sin(t), np.full(n, z), color=GLASS, lw=0.9, alpha=0.6, zorder=zorder + 1)


def camera(ax, C, dirs, size, color, name, lift):
    """A camera as a pyramid: its pinhole C and its image's four corners' directions; and its name above it."""
    base = [C + size * u for u in dirs]
    faces = [[C, base[i], base[(i + 1) % 4]] for i in range(4)] + [base]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor=color, alpha=0.85, lw=0.6, zorder=20))
    ax.text(*(C + np.array([0.0, 0.0, lift])), name, color=color, fontsize=10.5, fontweight="bold", ha="center",
            zorder=21)


def ring_order(C, pts, ax3):
    """The four corner points in order around the camera's optical axis (rows of ax3: x, y, z axes)."""
    u = np.asarray(pts, float) - C
    return list(np.argsort(np.arctan2(u @ ax3[1], u @ ax3[0])))


# ---------------------------------------------------------------------------------------------------------- cylinder
def cylinder(out):
    d = np.load(CYL_NPZ)
    num = nested(CYL_JSON)
    cal, trk = num["cal"], num["trk"]
    R1, R2 = (float(r) for r in d["rig_radii"])
    fig = plt.figure(figsize=(12.0, 8.6), dpi=DPI, facecolor=BG)

    # the photograph
    axp = fig.add_axes([0.015, 0.515, 0.47, 0.47 * 12.0 / 8.6 / 1.5])
    axp.imshow(Image.open(PHOTO))
    axp.set_axis_off()
    axp.text(0.025, 0.955, "Real setup", transform=axp.transAxes, color="white", fontsize=13, fontweight="bold",
             va="top")
    axp.text(0.025, 0.885, "three high-speed cameras, a water-filled acrylic tube", transform=axp.transAxes,
             color="#f2f2f2", fontsize=9.5, va="top")

    # the calibrated rig, to scale
    ax = axes3d(fig, [0.49, 0.45, 0.51, 0.49], CYL_VIEW, CYL_BOX, zoom=1.25)
    tube(ax, R1, *CYL_TUBE_Z, 0.10, 1)
    ends = d["t0_ends"].reshape(-1, 3).astype(float)
    pick = np.random.default_rng(0).choice(len(ends), size=4000, replace=False)
    ax.scatter(*ends[pick].T, s=0.4, c=INK, alpha=0.5, depthshade=False, lw=0, zorder=4)
    for i, c in enumerate((1, 2, 3)):
        rays = np.asarray(d[f"rig_rays{c}"], float)            # (7 rays, [pinhole, outer, inner, far wall], xyz)
        C = np.asarray(d["cam_xyz"][i], float)
        corners = rays[3:7]
        for r in corners:
            ax.plot(*r[:2].T, color=CAM[i], lw=0.8, alpha=0.8, zorder=5)    # in the air
            ax.plot(*r[1:].T, color=CAM[i], lw=1.1, alpha=1.0, zorder=6)    # the wall and the water
        o = ring_order(C, corners[:, 1], np.asarray(d[f"cam_axes{c}"], float))
        faces = [[corners[o[k], 2], corners[o[k], 3], corners[o[(k + 1) % 4], 3], corners[o[(k + 1) % 4], 2]]
                 for k in range(4)]
        ax.add_collection3d(Poly3DCollection(faces, facecolor=CAM[i], edgecolor="none", alpha=0.12, zorder=3))
        dirs = [(r[1] - r[0]) / np.linalg.norm(r[1] - r[0]) for r in corners[o]]
        camera(ax, C, dirs, 130.0, CAM[i], f"cam {c}", 85.0)
    label(fig, 0.515, 0.985, "Calibrated rig, to scale",
          "each camera's lines of sight bend at the tube's walls; dots: the wand's ends")

    # the tracks
    ax = axes3d(fig, [0.0, 0.0, 0.58, 0.47], TRK_VIEW, TRK_BOX, zoom=1.2)
    ids, fr = d["trk_id"], d["trk_frame"]
    xyz = d["trk_xyz"].astype(float)
    sp = np.linalg.norm(d["trk_v"].astype(float), axis=1)
    order = np.lexsort((fr, ids))
    ids, fr, xyz, sp = ids[order], fr[order], xyz[order], sp[order]
    same = (ids[1:] == ids[:-1]) & (fr[1:] - fr[:-1] == 1)
    segs = np.stack([xyz[:-1][same], xyz[1:][same]], axis=1)
    norm = Normalize(vmin=float(np.percentile(sp, 2)), vmax=float(np.percentile(sp, 98)))
    ax.add_collection3d(Line3DCollection(segs, colors=SPEED(norm(0.5 * (sp[:-1][same] + sp[1:][same]))),
                                         linewidths=0.75, alpha=0.95, zorder=5))
    n_tracks = len(np.unique(ids))
    label(fig, 0.025, 0.465, "Particles tracked through the wall",
          f"{n_tracks} tracks of a swirling flow, {trk['frames_processed']} frames at "
          f"{num['rig']['t1_frames_fps_exposure_us'][1]} fps")
    cax = fig.add_axes([0.55, 0.07, 0.011, 0.25])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SPEED), cax=cax)
    cb.set_label("speed (mm/s)", color=INK, fontsize=9)
    cb.ax.tick_params(colors=INK, labelsize=8)
    cb.outline.set_edgecolor(MUTED)

    # the wand's length
    ax = fig.add_axes([0.665, 0.085, 0.31, 0.28])
    ax.set_facecolor(BG)
    L = d["t0_len_um"].astype(float)
    rms = cal["free_wand_length_rms_um"]
    ax.hist(L, bins=np.linspace(-250, 250, 81), color=CAM[0], alpha=0.95)
    ax.axvline(0, color=MUTED, lw=0.8)
    for s in (-rms, rms):
        ax.axvline(s, color=CAM[1], lw=1.0, ls="--")
    ax.text(rms + 8, ax.get_ylim()[1] * 0.92, "±RMS", color=CAM[1], fontsize=8.5, va="top")
    ax.set_xlabel("error of the 10 mm wand's length (µm)", color=INK, fontsize=9)
    ax.set_ylabel("frames", color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    for s in ax.spines.values():
        s.set_color(MUTED)
    label(fig, 0.64, 0.465, "Calibration accuracy",
          f"wand length RMS {rms:.0f} µm ({cal['free_wand_length_rms_pct']:.2f} %), {len(L)} frames")
    fig.text(0.64, 0.397, f"reprojection RMS {cal['rms_px']:.2f} px over {cal['frames']} calibration frames",
             color=MUTED, fontsize=9.5, va="top")
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)

    print("cylinder (run 2026-10-08):")
    print("  model %s, %d cameras, %d calibration frames" % (cal["model"], len(cal["rms_px_per_camera"]),
                                                             cal["frames"]))
    print("  reprojection RMS %.2f px (per camera %s)" % (cal["rms_px"],
                                                         ", ".join("%.2f" % v for v in cal["rms_px_per_camera"])))
    print("  free wand length: RMS %.1f um = %.2f %% of 10 mm, %d frames" % (rms, cal["free_wand_length_rms_pct"],
                                                                         len(L)))
    print("  tube OD %.1f mm, wall %.3f mm" % (num["rig"]["tube_od_mm"], num["rig"]["tube_wall_mm"]))
    print("  tracks drawn %d (%d positions), %d frames at %s fps; median speed %.0f mm/s"
          % (n_tracks, len(xyz), trk["frames_processed"], num["rig"]["t1_frames_fps_exposure_us"][1],
             float(np.median(sp))))
    print("  ->", out)


# ------------------------------------------------------------------------------------------------------------ plate
def plate(out):
    rig = json.load(open(PLATE_RIG))
    f = rig["fits"]["ref"]
    ev = json.load(open(PLATE_NUM))["every_frame"]["w4c_ref"]
    up = np.array([1.0, 1.0, f["up"]])                  # the frame's z may point down: draw height up
    t1, t2 = (w["t"] for w in f["walls"])
    n1, n2 = (np.asarray(w["n"], float) for w in f["walls"])
    w_y, w_x = rig["channel"]["width_mm"]               # inside widths across wall 1 (along y) and wall 2 (along x)
    zlo, zhi = CORNER_BOX[0][2], CORNER_BOX[1][2]

    def slab(n, s0, s1, a_lo, a_hi, axis):
        """The six faces of the wall s0 <= n.p <= s1, from a_lo to a_hi along x (axis 0) or y (axis 1), zlo to zhi."""
        q = {}
        for i, s in enumerate((s0, s1)):
            for j, a in enumerate((a_lo, a_hi)):
                for k, z in enumerate((zlo, zhi)):
                    q[i, j, k] = (np.array([a, (s - n[0] * a) / n[1], z]) if axis == 0 else
                                  np.array([(s - n[1] * a) / n[0], a, z]))
        return [[q[0, 0, 0], q[0, 1, 0], q[0, 1, 1], q[0, 0, 1]], [q[1, 0, 0], q[1, 1, 0], q[1, 1, 1], q[1, 0, 1]],
                [q[0, 0, 0], q[1, 0, 0], q[1, 0, 1], q[0, 0, 1]], [q[0, 1, 0], q[1, 1, 0], q[1, 1, 1], q[0, 1, 1]],
                [q[0, 0, 0], q[0, 1, 0], q[1, 1, 0], q[1, 0, 0]], [q[0, 0, 1], q[0, 1, 1], q[1, 1, 1], q[1, 0, 1]]]

    def walls(ax, close):
        """The two walls the cameras look through; from afar also the channel's far walls, close up only the
        outline of the channel's inside."""
        span_x = (-t2, t2 + w_x + t2)                   # wall 1 runs along x, over the corner and the channel
        span_y = (0.0, t1 + w_y + t1)
        slabs = [(slab(n1, 0.0, t1, *span_x, axis=0), 0.07 if close else 0.30),
                 (slab(n2, 0.0, t2, *span_y, axis=1), 0.07 if close else 0.30)]
        if not close:
            slabs += [(slab(n1, t1 + w_y, 2 * t1 + w_y, *span_x, axis=0), 0.12),
                      (slab(n2, t2 + w_x, 2 * t2 + w_x, *span_y, axis=1), 0.12)]
        for faces, fa in slabs:
            ax.add_collection3d(Poly3DCollection([[p * up for p in fc] for fc in faces], facecolor=GLASS,
                                                 edgecolor=(0.62, 0.80, 1.0, 0.75), lw=0.6, alpha=fa, zorder=2))
        if close:
            x0, x1, y0, y1 = t2, t2 + w_x, t1, t1 + w_y
            for z in (zlo, zhi):
                rect = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]])
                ax.plot(*(rect * up).T, color=MUTED, lw=0.7, alpha=0.8, zorder=3)
            for x, y in ((x1, y0), (x1, y1), (x0, y1)):
                ax.plot(*(np.array([[x, y, zlo], [x, y, zhi]]) * up).T, color=MUTED, lw=0.7, alpha=0.8, zorder=3)

    def far(cam, o, dvec):
        """Where a ray in the water reaches the wall opposite the camera's."""
        n, s = (n1, t1 + w_y) if cam["wall"] == 1 else (n2, t2 + w_x)
        return o + (s - n @ o) / (n @ dvec) * dvec

    pts = np.asarray(f["points"], float) * up
    fig = plt.figure(figsize=(12.0, 5.6), dpi=DPI, facecolor=BG)
    for p, (rect, view, box, zoom) in enumerate((([0.0, 0.0, 0.5, 0.86], RIG_VIEW, RIG_BOX, 1.05),
                                                 ([0.5, 0.0, 0.5, 0.86], CORNER_VIEW, CORNER_BOX, 1.1))):
        ax = axes3d(fig, rect, view, box, zoom)
        walls(ax, close=bool(p))
        ax.scatter(*pts.T, s=2.2 if p else 0.5, c=INK, alpha=0.8 if p else 0.6, depthshade=False, lw=0, zorder=6)
        for i, cam in enumerate(f["cameras"]):
            C = np.asarray(cam["C"], float)
            ax3 = np.vstack([cam["x_axis"], cam["y_axis"], cam["z_axis"]])
            rays = cam["rays"][:4]                      # the image's corners (the fifth is its centre)
            o = [np.asarray(r["o"], float) for r in rays]
            e = [far(cam, oo, np.asarray(r["d"], float)) for oo, r in zip(o, rays)]
            k = ring_order(C, o, ax3)
            for j in range(4):
                if p == 0:                              # the air: pinhole to the wall
                    ax.plot(*(np.vstack([C, o[j]]) * up).T, color=CAM[i], lw=0.7, alpha=0.8, zorder=4)
                ax.plot(*(np.vstack([o[j], e[j]]) * up).T, color=CAM[i], lw=1.3 if p else 0.8, alpha=0.95, zorder=7)
            if p:                                       # where the field of view meets the far wall
                ax.plot(*(np.array([e[j] for j in k + k[:1]]) * up).T, color=CAM[i], lw=1.0, alpha=0.9, zorder=7)
            faces = [[o[k[j]], e[k[j]], e[k[(j + 1) % 4]], o[k[(j + 1) % 4]]] for j in range(4)]
            ax.add_collection3d(Poly3DCollection([[q * up for q in fc] for fc in faces], facecolor=CAM[i],
                                                 edgecolor="none", alpha=0.07 if p else 0.06, zorder=5))
            if p == 0:
                dirs = [((o[j] - C) / np.linalg.norm(o[j] - C)) * up for j in k]
                camera(ax, C * up, dirs, 140.0, CAM[i], cam["name"].replace("C", "cam "), 100.0)
    label(fig, 0.02, 0.975, "Calibrated rig, to scale",
          "four cameras through two perpendicular walls of a water tank")
    label(fig, 0.52, 0.975, "The channel's corner, seen from the water",
          "each camera's field of view refracted into the water; dots: the wand's ends")
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)

    v = f["volume"]
    print("plate (run 2026-07-27, calib_plate_wand_ref_centre):")
    print("  %d cameras, %d frames; walls %.2f deg apart, %.1f mm thick; cameras %s mm from the volume's centre"
          % (len(f["cameras"]), ev["frames"], v["walls_angle_deg"], t1,
             ", ".join("%.0f" % c for c in v["camera_to_centre_mm"])))
    print("  reprojection: median %.3f px, RMS of inliers %.3f px" % (ev["median_px"], ev["rms_inliers_px"]))
    print("  wand length (10 mm): median |error| %.1f um, SD %.1f um (%.2f %%)"
          % (1e3 * ev["wand_abs_median_mm"], 1e3 * ev["wand_clean_sd_mm"], 10 * ev["wand_clean_sd_mm"]))
    print("  channel %g x %g mm inside" % (w_x, w_y))
    print("  ->", out)


def make_photo(src):
    im = Image.open(src).crop(PHOTO_CROP)
    im = im.resize((PHOTO_WIDTH, round(PHOTO_WIDTH * im.height / im.width)), Image.LANCZOS)
    im.save(PHOTO, quality=85, optimize=True)
    print("photo ->", PHOTO, im.size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--photo", help="the original photograph, to crop and shrink into setup_cylinder.jpg first")
    a = ap.parse_args()
    if a.photo:
        make_photo(a.photo)
    for fn, name in ((cylinder, "cylinder_calibration.png"), (plate, "plate_calibration.png")):
        path = os.path.join(HERE, name)
        fn(path)
        print("  %s: %.0f kB" % (name, os.path.getsize(path) / 1024))
