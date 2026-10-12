"""Interactive 3-D view of the calibrated rig: a standalone HTML page (plotly.js 2.27.0 from cdnjs).

    python viz3d.py        # writes figures/rig_3d.html in the output folder, from out_dir/fit_state.pkl

It shows, in the calibration's board frame (mm):
  * the tube's outer and inner walls and its axis, with the axis's angles;
  * the camera's pinhole (the lens's entrance pupil) with its axes;
  * the chief rays of the image centre and corners, traced through the walls with the fitted model.  Behind the lens
    front a chief ray is virtual: it is the straight extension towards the entrance pupil, drawn dashed;
  * the 20 board planes with the calibration dots (board corrections and plane shifts included);
  * the lens front and the camera body front at the distances measured on the photo of the rig.
The numbers come from the fitted rig and calibration_result.json.  A self-check compares the drawn rays with the
model's own back-projection (cylfit.trace) and with the calibration dots.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402

PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&"
         "family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap")
# trace colours: chosen to read on the light and the dark scene ground alike
C_OUTER, C_INNER, C_AXIS, C_RAY = "#7f9fbd", "#3b8fd6", "#d6336c", "#e8890c"
C_PIN, C_PHOTO, C_PLANE = "#7b5cd6", "#8a96a3", "#8a96a3"
C_X, C_Y, C_Z = "#e5484d", "#2f9e66", "#3e63dd"
DOT_SCALE = [[0.0, "#a9c9f2"], [1.0, "#1c4f9c"]]


# ------------------------------------------------------------------ geometry
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def cylinder_hits(o, d, a, e, radius) -> list:
    """Ray parameters t (ascending) where o + t d meets the cylinder of `radius` about the line a + s e."""
    w = o - a
    wp = w - (w @ e) * e
    dp = d - (d @ e) * e
    A, B, C = dp @ dp, 2.0 * (wp @ dp), wp @ wp - radius * radius
    disc = B * B - 4.0 * A * C
    if A <= 0 or disc < 0:
        return []
    sq = math.sqrt(disc)
    return sorted([(-B - sq) / (2.0 * A), (-B + sq) / (2.0 * A)])


def refract(d, N, n1, n2):
    """Vector Snell's law; N is the unit normal facing the incident side."""
    r = n1 / n2
    cosi = -float(N @ d)
    k = 1.0 - r * r * (1.0 - cosi * cosi)
    if k < 0:
        raise ValueError("total internal reflection")
    t = r * d + (r * cosi - math.sqrt(k)) * N
    return unit(t)


def chief_ray(rig, uv):
    """The chief ray of pixel uv: points [pinhole, outer wall, inner wall, far inner wall] and its direction in water.
    Same construction as cylfit.trace (no distortion in this camera), keeping every interface point."""
    x = (uv[0] - rig.cx) / rig.f
    y = (uv[1] - rig.cy) / rig.f
    d = unit(np.array([x, y, 1.0]) @ rig.R)
    e = unit(rig.axis_dir)
    a = np.asarray(rig.axis_pt, float)
    o = np.asarray(rig.C, float)
    pts = [o]
    for k, radius in enumerate(rig.radii):
        t = [s for s in cylinder_hits(o, d, a, e, radius) if s > 1e-9][0]
        h = o + t * d
        hp = (h - a) - ((h - a) @ e) * e
        d = refract(d, unit(hp), rig.n[k], rig.n[k + 1])
        pts.append(h)
        o = h
    t = [s for s in cylinder_hits(o, d, a, e, rig.radii[-1]) if s > 1e-6][-1]
    pts.append(o + t * d)
    return pts, d


def tube_surface(a, e, radius, s0, s1, ns=24, nt=73):
    """x, y, z grids of the cylinder of `radius` about a + s e, s in [s0, s1]."""
    e1 = unit(np.cross(e, [0.0, 0.0, 1.0]))
    e2 = np.cross(e, e1)
    s = np.linspace(s0, s1, ns)[:, None, None]
    th = np.linspace(0.0, 2.0 * np.pi, nt)[None, :, None]
    P = a + s * e + radius * (np.cos(th) * e1 + np.sin(th) * e2)
    return P[..., 0], P[..., 1], P[..., 2]


def ring(a, e, radius, s, n=145):
    e1 = unit(np.cross(e, [0.0, 0.0, 1.0]))
    e2 = np.cross(e, e1)
    th = np.linspace(0.0, 2.0 * np.pi, n)[:, None]
    return a + s * e + radius * (np.cos(th) * e1 + np.sin(th) * e2)


# ------------------------------------------------------------------ numbers for the page
def rig_numbers(rig, cfg) -> dict:
    """Angles and distances of the fitted rig (degrees, mm)."""
    d = unit(rig.axis_dir)
    if d[1] < 0:
        d = -d
    a = rig.axis_at_y0()
    xc, yc, zc = rig.R[0], rig.R[1], rig.R[2]
    C = np.asarray(rig.C, float)
    n = np.cross(zc, d)
    miss = float(abs((C - a) @ n) / np.linalg.norm(n))
    t_wall = [s for s in cylinder_hits(C, zc, a, d, rig.radii[0]) if s > 0][0]
    W = C + t_wall * zc
    setup = cfg.get("setup", {})
    L = W - setup.get("lens_front_to_tube_wall_mm", float("nan")) * zc
    B = W - setup.get("camera_body_front_to_tube_wall_mm", float("nan")) * zc
    # the tube axis in the image: two of its points inside the calibrated volume, projected with the model
    cam = rig.camera()
    uv_ax = CF.project(cam, np.array([a + (-8.0) * d, a + 6.0 * d]))
    du, dv = uv_ax[1] - uv_ax[0]
    s = rig.summary()
    return {
        "d": d, "a": a, "xc": xc, "yc": yc, "zc": zc, "C": C, "W": W, "L": L, "B": B, "t_wall": float(t_wall),
        "tilt": s["axis_tilt_in_board_plane_deg"], "lean": s["axis_lean_towards_camera_deg"],
        "axis_from_Y": math.degrees(math.acos(min(1.0, float(d[1])))),
        "axis_in_image_deg": math.degrees(math.atan2(du, -dv)),
        "axis_u_at": (float(uv_ax[0][0] + (399.5 - uv_ax[0][1]) * du / dv)) if dv else float("nan"),
        "yaw": math.degrees(math.atan2(zc[0], -zc[2])), "pitch": math.degrees(math.asin(zc[1])),
        "view_from_minusZ": s["view_tilt_from_minusZ_deg"],
        "optical_vs_axis": math.degrees(math.acos(abs(float(zc @ d)))),
        "miss": miss, "dist_axis": s["camera_distance_from_axis_mm"],
    }


def heldout_line() -> str:
    """The leave-one-out line of tests/t4_out.txt, if present: (RMS, max) px."""
    p = os.path.join(HERE, "tests", "t4_out.txt")
    try:
        txt = open(p, encoding="utf-8").read()
    except OSError:
        return ""
    m = re.search(r"\(b\) each plane left out, stage z.*?RMS ([0-9.]+).*?max ([0-9.]+)", txt)
    return f"{float(m.group(1)):.2f} px RMS, max {float(m.group(2)):.2f} px" if m else ""


# ------------------------------------------------------------------ traces
def r3(a):
    return np.round(np.asarray(a, float), 3).tolist()


def polyline(segments) -> tuple:
    """x, y, z lists of several polylines separated by None (one plotly trace)."""
    xs, ys, zs = [], [], []
    for seg in segments:
        seg = np.asarray(seg, float)
        xs += r3(seg[:, 0]) + [None]
        ys += r3(seg[:, 1]) + [None]
        zs += r3(seg[:, 2]) + [None]
    return xs, ys, zs


def line_trace(segments, name, color, width=3, dash="solid", group=None, show=True, hover=None):
    x, y, z = polyline(segments)
    t = {"type": "scatter3d", "mode": "lines", "x": x, "y": y, "z": z, "name": name,
         "line": {"color": color, "width": width, "dash": dash}, "showlegend": show,
         "hoverinfo": "text" if hover else "skip"}
    if hover:
        t["text"] = hover
        t["hoverinfo"] = "text"
    if group:
        t["legendgroup"] = group
    return t


def surface_trace(grids, name, color, opacity):
    x, y, z = grids
    return {"type": "surface", "x": r3(x), "y": r3(y), "z": r3(z), "name": name, "showscale": False,
            "showlegend": True, "opacity": opacity, "colorscale": [[0, color], [1, color]],
            "surfacecolor": np.zeros_like(x).tolist(), "hoverinfo": "name", "lighting": {"ambient": 0.9},
            "contours": {"x": {"highlight": False}, "y": {"highlight": False}, "z": {"highlight": False}}}


def label_trace(points, texts, name="labels", group=None):
    P = np.asarray(points, float)
    return {"type": "scatter3d", "mode": "text", "x": r3(P[:, 0]), "y": r3(P[:, 1]), "z": r3(P[:, 2]),
            "text": texts, "textposition": "middle right", "showlegend": False, "hoverinfo": "skip",
            "name": name, "meta": "label", **({"legendgroup": group} if group else {})}


def frame_traces(origin, axes, length, names, group, legend_name, width=5):
    out = []
    for k, (vec, col, nm) in enumerate(zip(axes, (C_X, C_Y, C_Z), names)):
        tip = origin + length * np.asarray(vec, float)
        out.append(line_trace([np.array([origin, tip])], legend_name if k == 0 else nm, col, width,
                              group=group, show=(k == 0)))
    tips = [origin + 1.08 * length * np.asarray(v, float) for v in axes]
    out.append(label_trace(tips, names, group=group))
    return out


def build(cfg: dict, rig: CF.Rig, obs: CF.Obs, result: dict, path: str) -> dict:
    nb = rig_numbers(rig, cfg)
    d, a, C = nb["d"], nb["a"], nb["C"]
    R_out, R_in = rig.radii
    z_planes = rig.plane_z(obs.z_nom)
    P = rig.world_points(obs)

    # chief rays: the image centre and the four corners
    W_, H_ = rig.width, rig.height
    pix = [("centre", (rig.cx, rig.cy)), ("top left", (0.0, 0.0)), ("top right", (W_ - 1.0, 0.0)),
           ("bottom right", (W_ - 1.0, H_ - 1.0)), ("bottom left", (0.0, H_ - 1.0))]
    rays = []
    for name, uv in pix:
        pts, dw = chief_ray(rig, uv)
        # where the ray crosses the plane of the lens front (perpendicular to the optical axis)
        dir0 = unit(pts[1] - pts[0])
        tL = float((nb["L"] - C) @ nb["zc"]) / float(dir0 @ nb["zc"])
        rays.append({"name": name, "uv": uv, "pts": pts, "lens": C + tL * dir0, "dw": dw})

    # self-check 1: the drawn water rays against cylfit.trace
    uvs = np.array([r["uv"] for r in rays], float)
    o_m, d_m, ok = CF.trace(rig, uvs)
    chk_o = float(np.max(np.abs(np.array([r["pts"][2] for r in rays]) - o_m)))
    chk_d = float(np.max(np.abs(np.array([r["dw"] for r in rays]) - d_m)))
    # self-check 2: the model's rays through the predicted pixels of the calibration dots pass through the dots
    cam = rig.camera()
    uv_p = CF.project(cam, P)
    o2, d2, _ = CF.trace(rig, uv_p)
    w = P - o2
    miss = np.linalg.norm(w - (np.einsum("ij,ij->i", w, d2))[:, None] * d2, axis=1)
    checks = {"ray_origin_vs_trace_mm": chk_o, "ray_direction_vs_trace": chk_d, "trace_ok": bool(np.all(ok)),
              "dot_ray_miss_max_mm": float(np.nanmax(miss))}

    # ---------------------------------------------------------- shared pieces
    X, Y = obs.pitch[0] * obs.i, obs.pitch[1] * obs.j
    x0, x1, y0, y1 = X.min() - 2.0, X.max() + 2.0, Y.min() - 2.0, Y.max() + 2.0
    shifts = rig.shift if rig.shift is not None else np.zeros((obs.n_planes, 2))
    plane_segs, plane_hover = [], []
    for k, z in enumerate(z_planes):
        sx, sy = shifts[k]
        plane_segs.append(np.array([[x0 + sx, y0 + sy, z], [x1 + sx, y0 + sy, z], [x1 + sx, y1 + sy, z],
                                    [x0 + sx, y1 + sy, z], [x0 + sx, y0 + sy, z]]))
        h = (f"board plane {k} (file z_{int(cfg['planes']['labels'][k]):02d}mm.tif)<br>z = {z:.2f} mm, "
             f"{z - nb['a'][2]:+.2f} mm from the axis<br>stage play ({sx * 1000:+.1f}, {sy * 1000:+.1f}) um")
        plane_hover += [h] * 5 + [None]
    zc_dots = z_planes[obs.plane]
    dots = {"type": "scatter3d", "mode": "markers", "x": r3(P[:, 0]), "y": r3(P[:, 1]), "z": r3(P[:, 2]),
            "name": "calibration dots (1027)", "marker": {"size": 2.6, "color": r3(zc_dots), "colorscale": DOT_SCALE,
                                                         "cmin": float(z_planes.min()), "cmax": float(z_planes.max()),
                                                         "showscale": False},
            "text": [f"dot ({int(i)}, {int(j)}), plane z = {z:.2f} mm" for i, j, z in zip(obs.i, obs.j, zc_dots)],
            "hoverinfo": "text"}
    axis_hover = (f"tube axis<br>through ({a[0]:.2f}, 0, {a[2]:.2f}) mm at Y = 0<br>tilt in the board plane "
                  f"{nb['tilt']:+.3f} deg, lean towards the camera {nb['lean']:+.3f} deg<br>{nb['axis_from_Y']:.3f} deg "
                  f"from the board's Y axis")
    pin_hover = (f"pinhole = entrance pupil of the lens<br>C = ({C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}) mm<br>"
                 f"f = {rig.f:,.0f} px ({rig.f * cfg['image']['pixel_mm']:.1f} mm)<br>{nb['dist_axis']:.1f} mm from "
                 f"the tube axis, {nb['t_wall']:.1f} mm from the tube wall")

    def ray_traces(with_air: bool):
        real, virt, hov = [], [], []
        for r in rays:
            p = r["pts"]
            seg = ([r["lens"], p[1]] if with_air else []) + [p[1], p[2], p[3]]
            real.append(np.array(seg))
            if with_air:
                virt.append(np.array([p[0], r["lens"]]))
            hov += [f"chief ray of the {r['name']} pixel ({r['uv'][0]:.0f}, {r['uv'][1]:.0f})"] * len(seg) + [None]
        out = [line_trace(real, "chief rays (centre and corners)", C_RAY, 3, hover=hov)]
        if with_air:
            out.append(line_trace(virt, "chief rays behind the lens front (virtual)", C_RAY, 2, dash="dash"))
        return out

    # ---------------------------------------------------------- scene 1: the whole rig
    s1 = [surface_trace(tube_surface(a, d, R_out, -110.0, 110.0), "tube, outer wall (acrylic)", C_OUTER, 0.16),
          surface_trace(tube_surface(a, d, R_in, -110.0, 110.0), "tube, inner wall (water)", C_INNER, 0.14),
          line_trace([np.array([a - 140.0 * d, a + 140.0 * d])], "tube axis", C_AXIS, 6,
                     hover=[axis_hover] * 2 + [None])]
    s1 += ray_traces(True)
    s1.append({"type": "scatter3d", "mode": "markers", "x": [round(C[0], 3)], "y": [round(C[1], 3)],
               "z": [round(C[2], 3)], "name": "pinhole (entrance pupil)",
               "marker": {"size": 6, "color": C_PIN, "symbol": "diamond"}, "text": [pin_hover], "hoverinfo": "text"})
    s1 += frame_traces(C, [nb["xc"], nb["yc"], nb["zc"]], 140.0, ["x (image right)", "y (image down)", "z (view)"],
                       "camaxes", "camera axes x, y, z")
    photo = np.array([nb["L"], nb["B"]])
    s1.append(line_trace([photo], "lens front and camera body (photo)", C_PHOTO, 9,
                         hover=["lens front: 200 mm from the tube wall (photo)",
                                "camera body front: 365 mm from the tube wall (photo)", None]))
    s1.append(label_trace([nb["L"] + np.array([0, 22, 0]), nb["B"] + np.array([0, -30, 0])],
                          ["lens front (photo)", "camera body (photo)"]))
    s1.append(line_trace(plane_segs, "board planes (20)", C_PLANE, 1.5, hover=plane_hover))
    s1.append(dots)
    s1 += frame_traces(np.zeros(3), np.eye(3), 70.0, ["X", "Y", "Z"], "world", "board frame X, Y, Z", 4)
    s1.append(label_trace([C + np.array([0, -40, 0]), a + 130.0 * d], ["pinhole (entrance pupil)", "tube axis"]))

    # ---------------------------------------------------------- scene 2: inside the tube
    s2 = [surface_trace(tube_surface(a, d, R_out, -22.0, 18.0, 8), "tube, outer wall (acrylic)", C_OUTER, 0.12),
          surface_trace(tube_surface(a, d, R_in, -22.0, 18.0, 8), "tube, inner wall (water)", C_INNER, 0.10),
          line_trace([ring(a, d, R_out, -a[1]), ring(a, d, R_in, -a[1])], "wall cross-section at Y = 0", C_OUTER, 2),
          line_trace([np.array([a - 24.0 * d, a + 20.0 * d])], "tube axis", C_AXIS, 7,
                     hover=[axis_hover] * 2 + [None])]
    s2 += ray_traces(False)
    s2.append(line_trace(plane_segs, "board planes (20)", C_PLANE, 1.5, hover=plane_hover))
    s2.append(dots)
    s2 += frame_traces(np.zeros(3), np.eye(3), 18.0, ["X", "Y", "Z"], "world", "board frame X, Y, Z", 4)
    s2.append(label_trace([a + 21.0 * d, np.array([x1 + 1.0, y1, z_planes[0]]), np.array([x1 + 1.0, y1, z_planes[-1]])],
                          [f"tube axis: tilt {nb['tilt']:+.3f} deg, lean {nb['lean']:+.3f} deg",
                           f"plane z = {z_planes[0]:.2f} mm", f"plane z = {z_planes[-1]:.2f} mm"]))

    def box(pts_list, pad, zoom):
        """Axis ranges, and the aspect ratio of true proportions times `zoom` (an orthographic scene's scale)."""
        Q = np.vstack(pts_list)
        lo, hi = Q.min(axis=0) - pad, Q.max(axis=0) + pad
        span = hi - lo
        return {"x": r3([lo[0], hi[0]]), "y": r3([lo[1], hi[1]]), "z": r3([lo[2], hi[2]]),
                "aspect": {"x": zoom * float(span[0] / span.max()), "y": zoom * float(span[1] / span.max()),
                           "z": zoom * float(span[2] / span.max())}}

    tube_pts = np.vstack([ring(a, d, R_out, -110.0), ring(a, d, R_out, 110.0)])
    b1 = box([tube_pts, C[None, :], (C + 150 * nb["xc"])[None, :]], 15.0, 2.0)
    b2 = box([ring(a, d, R_out, -22.0), ring(a, d, R_out, 18.0)], 4.0, 1.6)
    data = {
        "s1": {"traces": s1, "range": b1, "camera": {"up": {"x": 0, "y": 1, "z": 0}, "eye": {"x": 0.95, "y": 0.42,
                                                                                            "z": 0.22},
                                                    "center": {"x": 0, "y": 0, "z": 0},
                                                    "projection": {"type": "orthographic"}}},
        "s2": {"traces": s2, "range": b2, "camera": {"up": {"x": 0, "y": 1, "z": 0}, "eye": {"x": 0.44, "y": 0.62,
                                                                                            "z": 0.62},
                                                    "center": {"x": 0, "y": 0, "z": 0},
                                                    "projection": {"type": "orthographic"}}},
    }
    page = render_page(cfg, rig, obs, result, nb, data, checks)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return checks


# ------------------------------------------------------------------ the page
def fmt(v, nd=2):
    return f"{v:,.{nd}f}".replace("-", "−")


def render_page(cfg, rig, obs, result, nb, data, checks) -> str:
    sig = result.get("sigma", {})
    sc = sig.get("camera_centre_mm", [float("nan")] * 3)
    sa = sig.get("axis_position_X_Z_mm", [float("nan")] * 2)
    st = sig.get("axis_tilt_lean_deg", [float("nan")] * 2)
    res = result.get("residuals", {})
    z_planes = rig.plane_z(obs.z_nom)
    C, a, d = nb["C"], nb["a"], nb["d"]
    shifts = rig.shift if rig.shift is not None else np.zeros((obs.n_planes, 2))
    srms = np.sqrt(np.mean(shifts ** 2, axis=0)) * 1000
    held = heldout_line()
    pix = cfg["image"]["pixel_mm"]
    e = html.escape

    def row(k, v, note=""):
        return (f"<div class=\"row\"><dt>{e(k)}</dt><dd>{v}</dd></div>" if not note else
                f"<div class=\"row\"><dt>{e(k)}</dt><dd>{v}<span class=\"note\">{e(note)}</span></dd></div>")

    sheet = f"""
<section class="block">
  <h3>Camera <span>pinhole = the lens's entrance pupil</span></h3>
  <dl>
    {row("Position X, Y, Z", f"{fmt(C[0])} ± {sc[0]:.2f}<br>{fmt(C[1])} ± {sc[1]:.2f}<br>{fmt(C[2])} ± {sc[2]:.2f} mm")}
    {row("Distance to the tube axis", f"{fmt(nb['dist_axis'], 1)} mm")}
    {row("Pinhole to the tube wall", f"{fmt(nb['t_wall'], 1)} mm", "along the optical axis")}
    {row("Focal length f", f"{fmt(rig.f, 0)} ± {sig.get('f_px', float('nan')):.0f} px<br>{fmt(rig.f * pix, 1)} mm at 20 µm")}
    {row("Principal point", f"({rig.cx:.1f}, {rig.cy:.1f}) px", "fixed at the sensor centre")}
    {row("Viewing direction", f"{fmt(nb['view_from_minusZ'])}° from −Z<br>yaw {fmt(nb['yaw'])}°, pitch {fmt(nb['pitch'])}°")}
    {row("Optical axis to tube axis", f"{fmt(nb['optical_vs_axis'])}°<br>passes it at {fmt(nb['miss'], 1)} mm")}
  </dl>
</section>
<section class="block">
  <h3>Tube axis</h3>
  <dl>
    {row("Point at Y = 0", f"X {fmt(a[0])} ± {sa[0]:.2f}<br>Z {fmt(a[2])} ± {sa[1]:.2f} mm")}
    {row("Direction", f"({d[0]:+.5f}, {d[1]:.5f}, {d[2]:+.5f})".replace("-", "−"))}
    {row("Tilt in the board plane", f"{fmt(nb['tilt'], 3)}° ± {st[0]:.3f}°", "top towards +X is positive")}
    {row("Lean towards the camera", f"{fmt(nb['lean'], 3)}° ± {st[1]:.3f}°", "top towards +Z is positive")}
    {row("Angle from the board's Y axis", f"{fmt(nb['axis_from_Y'], 3)}°")}
    {row("In the image", f"{fmt(nb['axis_in_image_deg'])}° from vertical<br>column u ≈ {fmt(nb['axis_u_at'], 0)} px at row 400")}
  </dl>
</section>
<section class="block">
  <h3>Tube</h3>
  <dl>
    {row("Outer / inner radius", f"{rig.radii[0]:.2f} / {rig.radii[1]:.2f} mm", "OD 7.9375 in, wall 0.125 in")}
    {row("Refractive index", f"air {rig.n[0]:.3f} · acrylic {rig.n[1]:.2f} · water {rig.n[2]:.3f}", "nominal, not fitted")}
  </dl>
</section>
<section class="block">
  <h3>Board planes</h3>
  <dl>
    {row("Positions", f"{len(z_planes)} planes, z = {fmt(z_planes[0])} … {fmt(z_planes[-1])} mm", "2.54 mm (0.1 in) stage steps")}
    {row("Relative to the axis", f"{z_planes[0] - a[2]:+.2f} … {z_planes[-1] - a[2]:+.2f} mm".replace("-", "−"), "along Z; negative is behind the axis")}
    {row("Dots", f"1.5 mm at 3 mm pitch; {len(obs.plane)} used")}
    {row("Stage play", f"{srms[0]:.1f} / {srms[1]:.1f} µm RMS", "sideways shift of each plane, X / Y")}
  </dl>
</section>
<section class="block">
  <h3>Fit</h3>
  <dl>
    {row("Residual", f"{res.get('rms_px', float('nan')):.3f} px RMS, max {res.get('max_px', float('nan')):.2f} px")}
    {row("Planes left out", e(held) if held else "see tests/t4_out.txt", "each plane predicted at its stage position")}
  </dl>
</section>
<section class="block">
  <h3>From the photo <span>approximate</span></h3>
  <dl>
    {row("Lens front", "≈ 200 mm from the tube wall")}
    {row("Camera body front", "≈ 365 mm from the tube wall")}
    {row("Pinhole", f"{fmt(nb['t_wall'] - 365.0, 0)} mm behind the body front", "a virtual point: the entrance pupil")}
  </dl>
</section>"""

    title = "cam1 Calibration Geometry"
    data_json = json.dumps(data, separators=(",", ":"))
    return TEMPLATE.replace("{{TITLE}}", e(title)).replace("{{FONTS}}", FONTS).replace("{{PLOTLY}}", PLOTLY) \
        .replace("{{SHEET}}", sheet).replace("{{DATA}}", data_json) \
        .replace("{{RUN}}", e(f"Run 20260930 · T0 · cam{cfg['camera_id']}")) \
        .replace("{{MODEL}}", e(str(result.get("model", "")))) \
        .replace("{{CHECK}}", e(f"Drawn rays agree with the model's back-projection to "
                                f"{max(checks['ray_origin_vs_trace_mm'], 1e-15):.0e} mm; the model's rays through "
                                f"the {len(obs.plane)} dots pass within {checks['dot_ray_miss_max_mm']:.0e} mm of them."))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{{TITLE}}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{{FONTS}}">
<style>
/* Layout: an instrument viewer. Two to-scale 3-D scenes on the left, the fitted numbers as a datasheet on the
   right; one column on narrow screens. */
:root {
  --bg: #f3f5f7;
  --panel: #ffffff;
  --scene: #edf1f4;
  --ink: #142029;
  --muted: #56636f;
  --rule: #d6dde3;
  --grid: #cfd8df;
  --accent: #0e6b8a;
  --font-sans: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
  --step--1: 0.8125rem;
  --step-0: 0.9375rem;
  --step-1: 1.125rem;
  --step-3: 1.75rem;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1418; --panel: #161d23; --scene: #1b232a; --ink: #e3e9ee; --muted: #9aa8b4; --rule: #2a343d;
    --grid: #33404a; --accent: #5cb8d6; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1418; --panel: #161d23; --scene: #1b232a; --ink: #e3e9ee; --muted: #9aa8b4; --rule: #2a343d;
  --grid: #33404a; --accent: #5cb8d6; color-scheme: dark;
}
* { box-sizing: border-box; }
html, body { margin: 0; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-sans); font-size: var(--step-0);
  line-height: 1.5; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
.wrap { max-width: 1480px; margin: 0 auto; padding-inline: max(16px, 2.5vw); padding-block: 28px 40px;
  display: grid; gap: 22px; }
header { display: grid; gap: 6px; max-width: 78ch; }
.eyebrow { margin: 0; font-family: var(--font-mono); font-size: var(--step--1); letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--accent); }
h1 { margin: 0; font-size: var(--step-3); font-weight: 600; line-height: 1.15; text-wrap: balance; }
.lede { margin: 0; color: var(--muted); }
.layout { display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(0, 1fr); gap: 22px; align-items: start; }
@media (max-width: 980px) { .layout { grid-template-columns: minmax(0, 1fr); } }
.figs { display: grid; gap: 22px; min-width: 0; }
figure { margin: 0; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; min-width: 0;
  overflow: hidden; }
figcaption { display: grid; gap: 2px; padding: 14px 16px 6px; }
figcaption h2 { margin: 0; font-size: var(--step-1); font-weight: 600; }
figcaption p { margin: 0; color: var(--muted); font-size: var(--step--1); max-width: 90ch; }
.plot { width: 100%; height: 600px; }
@media (max-width: 600px) { .plot { height: 460px; } }
.sheet { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 6px 16px 14px;
  display: grid; gap: 4px; min-width: 0; }
.block { display: grid; gap: 6px; padding-block: 10px; border-bottom: 1px solid var(--rule); }
.block:last-child { border-bottom: 0; }
.block h3 { margin: 0; font-size: var(--step--1); font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
  color: var(--accent); display: flex; gap: 8px; flex-wrap: wrap; align-items: baseline; }
.block h3 span { text-transform: none; letter-spacing: 0; font-weight: 400; color: var(--muted); }
dl { margin: 0; display: grid; gap: 7px; }
.row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); gap: 10px; align-items: baseline; }
dt { color: var(--muted); font-size: var(--step--1); }
dd { margin: 0; font-family: var(--font-mono); font-size: var(--step--1); font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere; }
dd .note { display: block; font-family: var(--font-sans); color: var(--muted); font-size: 0.75rem; }
.reading { display: grid; gap: 8px; padding: 14px 16px; background: var(--panel); border: 1px solid var(--rule);
  border-radius: 6px; }
.reading h2 { margin: 0; font-size: var(--step-1); font-weight: 600; }
.reading ul { margin: 0; padding-left: 1.1rem; display: grid; gap: 4px; max-width: 95ch; }
.reading p { margin: 0; color: var(--muted); font-size: var(--step--1); max-width: 95ch; }
.swatch { display: inline-block; width: 0.9em; height: 0.35em; border-radius: 1px; vertical-align: middle;
  margin-right: 6px; }
.failed { padding: 16px; color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <p class="eyebrow">{{RUN}} · model {{MODEL}}</p>
    <h1>cam1 calibration geometry</h1>
    <p class="lede">The fitted camera, the water-filled tube and the calibration board, drawn to scale in the board's
      frame. Units are mm. Drag to rotate, scroll to zoom, and click a legend entry to hide it.</p>
  </header>
  <div class="layout">
    <div class="figs">
      <figure>
        <figcaption>
          <h2>Whole rig</h2>
          <p>The pinhole of the camera model is the lens's entrance pupil, about 1.1 m from the board and well behind the
            real camera. Between it and the lens front, the chief rays are virtual (dashed).</p>
        </figcaption>
        <div id="p1" class="plot" role="img" aria-label="3-D view of the camera pinhole, the lens and camera positions from the photo, the chief rays and the tube"></div>
      </figure>
      <figure>
        <figcaption>
          <h2>Inside the tube</h2>
          <p>The rays refract at the outer and inner walls and cross the 20 board planes. The dots are coloured by plane,
            from z = 0 (light, farthest from the camera) to z = 48.26 mm (dark).</p>
        </figcaption>
        <div id="p2" class="plot" role="img" aria-label="3-D view of the tube walls, its axis, the board planes with the calibration dots and the refracted chief rays"></div>
      </figure>
    </div>
    <aside class="sheet" aria-label="Fitted geometry">{{SHEET}}</aside>
  </div>
  <section class="reading">
    <h2>Reading the views</h2>
    <ul>
      <li><span class="swatch" style="background:#d6336c"></span><b>Tube axis</b>: tilt is the top of the axis moving
        towards +X within the board plane; lean is the top moving towards the camera (+Z).</li>
      <li><span class="swatch" style="background:#e8890c"></span><b>Chief rays</b> of the image centre and its four
        corners, traced with the fitted model. They bend at the outer and inner walls.</li>
      <li><span class="swatch" style="background:#3e63dd"></span><b>Axes.</b> Red, green and blue are x, y and z: for
        the camera, image right, image down and the viewing direction; for the board frame, X along the dot rows, Y
        up the columns, Z towards the camera.</li>
      <li><span class="swatch" style="background:#8a96a3"></span><b>Lens front and camera body</b> sit on the optical
        axis at the distances measured on the photo of the rig. They are approximate.</li>
    </ul>
    <p>{{CHECK}}</p>
  </section>
</div>
<script src="{{PLOTLY}}"></script>
<script>
const D = {{DATA}};
function tok(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function rgba(hex, a) {
  const h = hex.replace('#', '');
  const v = h.length === 3 ? h.split('').map(c => c + c).join('') : h;
  const n = parseInt(v, 16);
  return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
}
function axis(title, range, t) {
  return { title: { text: title, font: { size: 12, color: t.muted } }, range: range, color: t.muted,
    gridcolor: t.grid, zerolinecolor: t.grid, backgroundcolor: t.scene, showbackground: true,
    tickfont: { size: 10, color: t.muted }, showspikes: false };
}
function layoutFor(s, t, cam, narrow) {
  // wide: the legend has its own column on the left; narrow: a strip along the top
  const legend = narrow
    ? { orientation: 'h', x: 0, y: 1, xanchor: 'left', yanchor: 'top', font: { size: 10, color: t.ink } }
    : { x: 0, y: 1, xanchor: 'left', yanchor: 'top', font: { size: 11, color: t.ink } };
  return {
    paper_bgcolor: t.panel, plot_bgcolor: t.panel,
    font: { family: 'IBM Plex Sans, Segoe UI, system-ui, sans-serif', size: 12, color: t.ink },
    margin: { l: 8, r: 0, t: 0, b: 0 },
    hoverlabel: { font: { family: 'IBM Plex Mono, Consolas, monospace', size: 11 } },
    legend: Object.assign(legend, { bgcolor: rgba(t.panel, 0.9) }),
    scene: {
      domain: narrow ? { x: [0, 1], y: [0, 0.74] } : { x: [0.25, 1], y: [0, 1] },
      aspectmode: 'manual', aspectratio: s.range.aspect,
      xaxis: axis('X (mm)', s.range.x, t), yaxis: axis('Y (mm)', s.range.y, t), zaxis: axis('Z (mm)', s.range.z, t),
      camera: cam || s.camera, dragmode: 'orbit'
    }
  };
}
function themed(traces, t) {
  return traces.map(tr => tr.meta === 'label' ? Object.assign({}, tr, { textfont: { size: 11, color: t.ink,
    family: 'IBM Plex Sans, Segoe UI, sans-serif' } }) : tr);
}
function draw() {
  if (typeof Plotly === 'undefined') {
    for (const id of ['p1', 'p2']) {
      document.getElementById(id).innerHTML = '<p class="failed">The 3-D views need the plotting library from cdnjs.cloudflare.com. Check the internet connection and reload the page.</p>';
    }
    return;
  }
  const t = { panel: tok('--panel'), scene: tok('--scene'), ink: tok('--ink'), muted: tok('--muted'),
    rule: tok('--rule'), grid: tok('--grid') };
  for (const [id, key] of [['p1', 's1'], ['p2', 's2']]) {
    const el = document.getElementById(id);
    const cam = el._fullLayout && el._fullLayout.scene ? el._fullLayout.scene.camera : null;
    const narrow = el.clientWidth < 720;
    el.dataset.narrow = narrow ? '1' : '0';
    Plotly.react(el, themed(D[key].traces, t), layoutFor(D[key], t, cam, narrow),
      { responsive: true, displaylogo: false });
  }
}
draw();
try {
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
} catch (err) { /* older browsers: the theme is fixed at load */ }
let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const changed = ['p1', 'p2'].some(id => {
      const el = document.getElementById(id);
      return el.dataset.narrow !== (el.clientWidth < 720 ? '1' : '0');
    });
    if (changed) draw();
  }, 200);
});
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
</script>
</body>
</html>
"""


def main():
    cfg = case.load_config()
    import run_calibration as RC
    st = RC.load_state(os.path.join(cfg["out_dir"], "fit_state.pkl"))
    with open(os.path.join(cfg["out_dir"], "calibration_result.json"), encoding="utf-8") as f:
        result = json.load(f)
    path = os.path.join(cfg["out_dir"], "figures", "rig_3d.html")
    checks = build(cfg, st["rig"], st["obs"], result, path)
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} kB)")
    print("self-check:", checks)


if __name__ == "__main__":
    main()
