"""Diagnostic figures of the calibration (static PNG, matplotlib).

Colours follow the dataviz reference palette: slot 1 blue for u (or a single series), slot 2 orange for v, text in
neutral ink, a recessive grid.  The pair was run through the palette validator (all checks pass, light mode).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 5,
})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def residuals_per_plane(per_plane: list, path: str, title: str):
    z = np.array([p["z_nominal"] for p in per_plane])
    ru = np.array([p["rms_u"] for p in per_plane], float)
    rv = np.array([p["rms_v"] for p in per_plane], float)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(z, ru, "-o", color=BLUE, label="u (horizontal)")
    ax.plot(z, rv, "-s", color=ORANGE, label="v (vertical)")
    ax.text(z[-1] + 0.8, ru[-1], "u", color=INK2, va="center")
    ax.text(z[-1] + 0.8, rv[-1], "v", color=INK2, va="center")
    ax.set_xlabel("plane position z (mm)")
    ax.set_ylabel("RMS residual (px)")
    ax.set_ylim(0, max(np.nanmax(ru), np.nanmax(rv)) * 1.15)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, path)


def plane_offsets(z_nom, dz, sdz, path: str, title: str):
    """(a) fitted plane position minus the stage position; (b) the fitted length of each stage step.  Drawn only
    for models that fit every plane's position (M2*)."""
    z_nom, dz, sdz = map(np.asarray, (z_nom, dz, sdz))
    step0 = float(np.mean(np.diff(z_nom)))
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.3))
    a = axs[0]
    a.axhline(0, color=INK2, lw=0.8)
    a.errorbar(z_nom, dz, yerr=sdz, fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.0, capsize=2.5)
    a.set_xlabel("plane, stage position z (mm)")
    a.set_ylabel("fitted z minus stage z (mm)")
    a.set_title("(a) plane positions", loc="left")
    b = axs[1]
    steps = np.diff(z_nom + dz)
    sst = np.sqrt(sdz[1:] ** 2 + sdz[:-1] ** 2)
    mid = 0.5 * (z_nom[1:] + z_nom[:-1])
    b.axhline(step0, color=INK2, lw=0.8)
    b.text(mid[0], step0 + 0.03, f"stage: {step0:.2f} mm", color=INK2, fontsize=8, va="bottom")
    b.axhline(float(np.mean(steps)), color=BLUE, lw=0.8, ls="--")
    b.text(mid[-1], float(np.mean(steps)) + 0.03, f"mean {np.mean(steps):.3f} mm", color=INK2, fontsize=8,
           ha="right", va="bottom")
    b.errorbar(mid, steps, yerr=sst, fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.0, capsize=2.5)
    b.set_xlabel("step, between stage readings (mm)")
    b.set_ylabel("fitted step length (mm)")
    b.set_title("(b) each step", loc="left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def spacing_per_plane(obs, uv, u_band=(440.0, 840.0), v_band=(300.0, 500.0)):
    """Mean image spacing of neighbouring dots in each plane (px): vertical (j to j+1) over the central columns and
    horizontal (i to i+1) over the central rows.  The pairs are the same in every plane (dots seen in all planes,
    chosen by their position in the middle plane), so the board's own dot errors cancel from plane to plane.
    Returns (vertical, horizontal, number of vertical pairs, number of horizontal pairs)."""
    K = obs.n_planes
    pos = [dict() for _ in range(K)]
    for n in range(len(obs.plane)):
        pos[int(obs.plane[n])][(int(obs.i[n]), int(obs.j[n]))] = np.asarray(uv[n], float)
    common = set.intersection(*(set(p) for p in pos))
    mid = pos[K // 2]
    vp = [(a, (a[0], a[1] + 1)) for a in common
          if (a[0], a[1] + 1) in common and u_band[0] < mid[a][0] < u_band[1]]
    hp = [(a, (a[0] + 1, a[1])) for a in common
          if (a[0] + 1, a[1]) in common and v_band[0] < mid[a][1] < v_band[1]
          and u_band[0] < 0.5 * (mid[a][0] + mid[(a[0] + 1, a[1])][0]) < u_band[1]]
    sv = np.array([np.mean([np.hypot(*(p[b] - p[a])) for a, b in vp]) for p in pos])
    sh = np.array([np.mean([np.hypot(*(p[b] - p[a])) for a, b in hp]) for p in pos])
    return sv, sh, len(vp), len(hp)


def spacing_line(z, sv, cfg) -> dict:
    """Straight line through 1 / (vertical spacing) against the stage position.  Along the tube axis the wall acts
    as flat plates, so paraxially 1/s = (D0 - z n_air/n_w) / (f p): the slope gives f (px) and the intercept the
    air-equivalent distance D0 (mm) from the pinhole to plane 0, without the cylinder model."""
    n_air, _n_acr, n_w = cfg["port"]["n"]
    p = float(cfg["target"]["pitch_mm"][1])
    c = np.polyfit(np.asarray(z, float), 1.0 / np.asarray(sv, float), 1)
    f = -n_air / (n_w * p * c[0])
    fit = 1.0 / np.polyval(c, z)
    return {"f_px": float(f), "D0_mm": float(c[1] * f * p), "rms_px": float(np.sqrt(np.mean((fit - sv) ** 2)))}


def pinhole_growth(z, wall_to_pinhole_mm: float, cfg, axis_z: float) -> np.ndarray:
    """Growth of the vertical spacing (fraction) if the pinhole were `wall_to_pinhole_mm` from the tube wall: the
    air-equivalent distance to plane 0 is that distance + wall / n_acrylic + (inner radius + axis depth) / n_water."""
    n_air, n_acr, n_w = cfg["port"]["n"]
    R1, R2 = cfg["port"]["R"]
    D0 = wall_to_pinhole_mm + (R1 - R2) * n_air / n_acr + (R2 + axis_z) * n_air / n_w
    return D0 / (D0 - np.asarray(z, float) * n_air / n_w) - 1.0


def magnification(obs, resid, cfg, rig, path: str, title: str) -> dict:
    """(a) growth of the vertical dot spacing over the traverse: measured, the calibrated model, and what a pinhole
    at the lens front or at the camera body would give (the photo's distances); (b) the change of the horizontal
    spacing, which the tube's curvature sets, measured and model."""
    z = np.asarray(obs.z_nom, float) * rig.zscale + rig.dz
    sv, sh, nv, nh = spacing_per_plane(obs, obs.uv)
    pv, ph, _, _ = spacing_per_plane(obs, obs.uv - resid)
    line = spacing_line(z, sv, cfg)
    za = float(rig.axis_at_y0()[2])
    setup = cfg.get("setup", {})
    refs = [("pinhole at the lens front", setup.get("lens_front_to_tube_wall_mm"), "--"),
            ("pinhole at the camera body", setup.get("camera_body_front_to_tube_wall_mm"), ":")]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    a = axs[0]
    zz = np.linspace(z[0], z[-1], 100)
    out = {"vertical_growth_measured": float(sv[-1] / sv[0] - 1), "vertical_growth_model": float(pv[-1] / pv[0] - 1),
           "horizontal_change_measured": float(sh[-1] / sh[0] - 1),
           "horizontal_change_model": float(ph[-1] / ph[0] - 1), "pairs": [nv, nh], "line": line}
    def end_label(ax, y, s):
        ax.text(zz[-1] + 1.0, y, s, color=INK2, fontsize=8, va="center")

    for name, L, ls in refs:
        if L is None:
            continue
        g = pinhole_growth(zz - z[0], L, cfg, za - z[0])
        a.plot(zz, 100 * g, color=INK2, lw=1.0, ls=ls, label=f"{name} (photo)")
        end_label(a, 100 * g[-1], f"{100 * g[-1]:+.1f} %")
        out[f"growth_{name.replace(' ', '_')}"] = float(g[-1])
    gm = 100 * (sv / sv[0] - 1)
    a.plot(z, 100 * (pv / pv[0] - 1), color=ORANGE, lw=1.5, label=f"calibrated model, f = {rig.f:,.0f} px")
    a.plot(z, gm, "o", color=ORANGE, ms=4.5, mec=SURFACE, mew=0.8, label="measured")
    end_label(a, gm[-1], f"{gm[-1]:+.1f} %")
    a.set_xlabel("plane position z (mm)")
    a.set_ylabel("vertical dot spacing, change since z = 0 (%)")
    a.set_title(f"(a) along the tube axis ({nv} dot pairs per plane)", loc="left")
    a.set_xlim(z[0] - 1, z[-1] + 9)
    a.set_ylim(-0.5, None)
    a.legend(loc="upper left", fontsize=8)
    b = axs[1]
    b.axhline(0, color=INK2, lw=0.8)
    hm, hp = 100 * (sh / sh[0] - 1), 100 * (ph / ph[0] - 1)
    b.plot(z, hp, color=BLUE, lw=1.5, label="calibrated model")
    b.plot(z, hm, "o", color=BLUE, ms=4.5, mec=SURFACE, mew=0.8, label="measured")
    end_label(b, hm[-1] + 0.35, f"{hm[-1]:+.1f} %")
    end_label(b, hp[-1] - 0.35, f"{hp[-1]:+.1f} %")
    b.set_xlabel("plane position z (mm)")
    b.set_ylabel("horizontal dot spacing, change since z = 0 (%)")
    b.set_title(f"(b) across the tube axis ({nh} dot pairs per plane)", loc="left")
    b.set_xlim(z[0] - 1, z[-1] + 9)
    b.legend(loc="lower left", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)
    return out


def residual_vectors(planes_xy: list, path: str, title: str, width: int, height: int, scale_px: float = 0.5,
                     gain: float = 120.0):
    """Small multiples: each plane's dots at their image position, residual (observed - model) x gain."""
    n = len(planes_xy)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 1.85), sharex=True, sharey=True)
    for a in axs.ravel():
        a.set_visible(False)
    for a, (z, uv, r) in zip(axs.ravel(), planes_xy):
        a.set_visible(True)
        a.grid(False)
        a.set_xlim(0, width)
        a.set_ylim(height, 0)
        a.set_aspect("equal")
        a.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color=GRID, lw=0.8)
        a.scatter(uv[:, 0], uv[:, 1], s=2, color=INK2, lw=0)
        a.quiver(uv[:, 0], uv[:, 1], r[:, 0] * gain, r[:, 1] * gain, angles="xy", scale_units="xy", scale=1,
                 color=BLUE, width=0.006, headwidth=3, headlength=3.5)
        a.set_title(f"z = {z:.2f} mm", fontsize=8, loc="left")
        a.tick_params(labelsize=7)
    ax0 = axs.ravel()[0]
    ax0.quiver([60], [760], [scale_px * gain], [0], angles="xy", scale_units="xy", scale=1, color=INK, width=0.006,
               headwidth=3, headlength=3.5)
    ax0.text(60, 735, f"{scale_px} px", fontsize=7, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def residual_histograms(resid: np.ndarray, path: str, title: str):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    lim = float(np.percentile(np.abs(resid), 99.8)) * 1.1
    bins = np.linspace(-lim, lim, 41)
    for a, k, name, col in ((axs[0], 0, "u (horizontal)", BLUE), (axs[1], 1, "v (vertical)", ORANGE)):
        a.hist(resid[:, k], bins=bins, color=col, edgecolor=SURFACE, linewidth=0.6)
        s = float(np.sqrt(np.mean(resid[:, k] ** 2)))
        a.set_title(f"{name}: rms {s:.3f} px", loc="left", fontsize=9)
        a.set_xlabel("residual (px)")
    axs[0].set_ylabel("dots")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def board_errors(rig, pitch, path: str, title: str, gain: float = 150.0, scale_mm: float = 0.02):
    """The fitted per-dot board corrections as arrows on the board grid (x gain)."""
    X = pitch[0] * rig.board_ij[:, 0].astype(float)
    Y = pitch[1] * rig.board_ij[:, 1].astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.grid(False)
    ax.scatter(X, Y, s=8, color=INK2, lw=0, zorder=2)
    ax.quiver(X, Y, rig.board[:, 0] * gain, rig.board[:, 1] * gain, angles="xy", scale_units="xy", scale=1,
              color=BLUE, width=0.005, headwidth=3, headlength=3.5, zorder=3)
    x0, y0 = X.min(), Y.min() - 2.2
    ax.quiver([x0], [y0], [scale_mm * gain], [0], angles="xy", scale_units="xy", scale=1, color=INK, width=0.005,
              headwidth=3, headlength=3.5)
    ax.text(x0, y0 - 0.9, f"{scale_mm * 1000:.0f} um", fontsize=8, color=INK)
    ax.set_aspect("equal")
    ax.set_xlim(X.min() - 2, X.max() + 2)
    ax.set_ylim(Y.min() - 4, Y.max() + 2)
    ax.set_xlabel("board X (mm)")
    ax.set_ylabel("board Y (mm)")
    ax.set_title(title, loc="left")
    _save(fig, path)


def make_all(cfg: dict, planes: list, res: dict, result: dict):
    out = os.path.join(cfg["out_dir"], "figures")
    os.makedirs(out, exist_ok=True)
    fr, obs = res["fit"], res["obs"]
    per = result["residuals"]["per_plane"]
    residuals_per_plane(per, os.path.join(out, "residuals_per_plane.png"),
                        f"Reprojection residual per plane, model {res['info']['model']} "
                        f"(all planes: {fr.rms:.3f} px rms)")
    if "dz_mm" in result["sigma"]:
        plane_offsets(obs.z_nom, fr.rig.dz, np.array(result["sigma"]["dz_mm"]),
                      os.path.join(out, "plane_offsets.png"),
                      "Fitted plane positions minus the stage positions (1 sigma bars)")
    magnification(obs, fr.resid, cfg, fr.rig, os.path.join(out, "dot_spacing.png"),
                  "How the dot spacing changes as the board moves towards the camera")
    pxy = []
    for k in range(obs.n_planes):
        m = obs.plane == k
        pxy.append((obs.z_nom[k], obs.uv[m], -fr.resid[m]))      # detected minus model (resid is model - detected)
    residual_vectors(pxy, os.path.join(out, "residual_vectors.png"),
                     "Residuals (detected minus model) at each dot, x120", cfg["image"]["width"],
                     cfg["image"]["height"])
    residual_histograms(-fr.resid, os.path.join(out, "residual_histograms.png"),
                        "Distribution of the reprojection residuals (detected minus model)")
    if fr.rig.board is not None:
        board_errors(fr.rig, obs.pitch, os.path.join(out, "board_corrections.png"),
                     "Fitted dot positions minus the nominal 3 mm grid (x150)")
    try:                                         # the interactive 3-D view of the rig (viz3d.py)
        import viz3d
        viz3d.build(cfg, fr.rig, obs, result, os.path.join(out, "rig_3d.html"))
    except Exception as exc:                     # optional: never stop the outputs for it
        print(f"  rig_3d.html not written: {exc}")
