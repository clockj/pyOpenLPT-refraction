"""S0: does the settings page's Validate detect as the tracking run does on a tube project with the line detector?

    python s0_noise.py

The run calls the 2-D finder with the camera's index, so the line detector takes the camera's noise from
tracer.lines.noise; the page's Validate calls the Python binding findObject2D(image, obj_cfg), which has no camera
argument, so with one noise per camera the detector estimates the noise from the image instead. Here, on T1's start
frame (G1's T1 project, config.json read only): per camera the configured noise and the image's estimate, and the
detections of the three paths: findObject2D as Validate calls it, detectLines with the image's estimate, detectLines
with the configured noise (the run's call). Writes s0_noise_out.txt and .json here.
"""
from __future__ import annotations

import os

import numpy as np

import vcf_common as vc

PROJ = os.path.join(vc.HERE, "projects", "t1_selfcalproto")


def main():
    lpt = vc.lib("vcf")
    log = vc.Log("s0_noise")
    cfg_path = os.path.join(PROJ, "config.json")
    cfg = vc.load_json(cfg_path)
    tr = cfg["objects"][0]["tracer"]
    noise_cfg = [float(x) for x in tr["lines"]["noise"]]
    frame = int(cfg["frames"][0])
    settings = lpt.BasicSetting()
    settings.readConfig(cfg_path)
    obj_cfg = lpt.TracerConfig()
    obj_cfg.readConfig(settings._object_config_paths[0], settings)
    P = lpt.LineFinderParam()
    for k, v in tr["lines"].items():
        if k != "noise" and hasattr(P, k):
            setattr(P, k, v)
    finder = lpt.ObjectFinder2D()
    out = {"project": PROJ, "frame": frame, "cameras": []}
    log(f"T1 project {PROJ}, frame {frame}, detector {obj_cfg._detector}")
    for c, lst in enumerate(cfg["image_lists"]):
        io = lpt.ImageIO()
        io.loadImgPath(PROJ.replace("\\", "/") + "/", lst)
        img = io.loadImg(frame)
        est = float(lpt.estimateNoiseClipped(img))
        n_find = len(finder.findObject2D(img, obj_cfg))
        a_est = np.array(lpt.detectLines(img, est, P), float).reshape(-1, 14)
        a_cfg = np.array(lpt.detectLines(img, noise_cfg[c], P), float).reshape(-1, 14)
        # how many of the run's detections the Validate path also has (within 0.5 px)
        if len(a_cfg) and len(a_est):
            d = np.hypot(a_cfg[:, None, 0] - a_est[None, :, 0], a_cfg[:, None, 1] - a_est[None, :, 1]).min(axis=1)
            shared = int((d <= 0.5).sum())
        else:
            shared = 0
        rec = {"camera": c, "noise_config": noise_cfg[c], "noise_image": est, "findObject2D": n_find,
               "detectLines_image_noise": len(a_est), "detectLines_config_noise": len(a_cfg),
               "run_detections_also_found": shared}
        out["cameras"].append(rec)
        log(f"  camera {c}: noise configured {noise_cfg[c]:.3f}, estimated from the frame {est:.3f} "
            f"({100 * (est / noise_cfg[c] - 1):+.1f} %); detections: Validate's call {n_find}, detectLines with the "
            f"frame's estimate {len(a_est)}, with the configured noise (the run) {len(a_cfg)} "
            f"({100 * (len(a_est) / max(len(a_cfg), 1) - 1):+.1f} %); of the run's, {shared} found by both")
    log.save(out)


if __name__ == "__main__":
    main()
