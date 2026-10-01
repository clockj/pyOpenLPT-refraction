"""Overlays for checking detection, indexing and the fit by eye."""
from __future__ import annotations

import math

import cv2
import numpy as np

GREEN, RED, ORANGE, CYAN, MAGENTA, YELLOW = (0, 200, 0), (0, 0, 255), (0, 140, 255), (255, 200, 0), (255, 0, 255), (0, 230, 230)


def to_bgr(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.8])
    g = np.clip((a - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def draw_dots(img: np.ndarray, dots: list, ij=None, title: str = "", pred=None, resid_scale: float = 100.0,
              flags=None) -> np.ndarray:
    """Detections (ellipse + centre), their (i, j), and optionally predictions with residual vectors x resid_scale.

    pred: (N,2) predicted pixels aligned with `dots` (NaN where none).  flags: optional per-dot string drawn in
    magenta (e.g. "out" for an outlier).
    """
    vis = to_bgr(img)
    for k, d in enumerate(dots):
        if d["kind"] == "dot" and d.get("ok"):
            col = GREEN
            u, v = d["u"], d["v"]
            if np.isfinite(d["a"]):
                cv2.ellipse(vis, (int(round(u * 16)), int(round(v * 16))),
                            (int(round(d["a"] * 16)), int(round(d["b"] * 16))), math.degrees(d["theta"]), 0, 360,
                            col, 1, cv2.LINE_AA, 4)
        else:
            col = RED if d["kind"] == "border" else ORANGE
            u, v = d["u_rough"], d["v_rough"]
        cv2.drawMarker(vis, (int(round(u)), int(round(v))), col, cv2.MARKER_CROSS, 8, 1)
        if ij is not None:
            txt = f"{ij[k][0]},{ij[k][1]}"
            cv2.putText(vis, txt, (int(u) - 14, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        if flags is not None and flags[k]:
            cv2.putText(vis, flags[k], (int(u) - 14, int(v) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MAGENTA, 1,
                        cv2.LINE_AA)
        if pred is not None and np.all(np.isfinite(pred[k])):
            pu, pv = pred[k]
            cv2.circle(vis, (int(round(pu)), int(round(pv))), 3, CYAN, 1, cv2.LINE_AA)
            if d.get("ok"):
                tip = (int(round(u + resid_scale * (u - pu))), int(round(v + resid_scale * (v - pv))))
                cv2.arrowedLine(vis, (int(round(u)), int(round(v))), tip, YELLOW, 1, cv2.LINE_AA, tipLength=0.15)
    if title:
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (0, 0), (tw + 16, th + 12), (30, 30, 30), -1)
        cv2.putText(vis, title, (8, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
