"""The self-calibration campaign's fields as library correction files (data/fields/<name>_cam<k>.txt).

    python convert_fields.py [NAME ...]        # default: t1_vc2_total t1_m15_all cal4_r2 cal4_r3 cal4_r4

Each field's dense map (tests/cylinder_selfcal/data/fields/<name>.npz, what its warp and re-rendering read) sampled at
nodes every 8 px, read bilinearly in between. Logs how far that read departs from the dense map at every pixel.
Writes convert_fields_out.txt and .json here.
"""
from __future__ import annotations

import os
import sys

import numpy as np

import vcf_common as vc

DEFAULT = ["t1_vc2_total", "t1_m15_all", "cal4_r2", "cal4_r3", "cal4_r4"]


def main():
    names = sys.argv[1:] or DEFAULT
    log = vc.Log("convert_fields")
    out = {}
    for name in names:
        n_cam = int(np.load(os.path.join(vc.SELFCAL, "data", "fields", name + ".npz"))["n_cam"])
        for k in range(n_cam):
            corr, dev = vc.selfcal_correction(name, k)
            path = os.path.join(vc.DATA, "fields", f"{name}_cam{k + 1}.txt")
            corr.write(path, [f"from tests/cylinder_selfcal/data/fields/{name}.npz (camera {k + 1}): its dense map at "
                              f"nodes every {corr.step:g} px"])
            out[f"{name}_cam{k + 1}"] = {"file": path, **dev}
            log(f"{name} cam{k + 1}: {dev['field_rms_px']:.3f} px RMS, {dev['field_max_px']:.3f} px max; the 8 px "
                f"nodes read bilinearly depart from the dense map by {dev['rms_dev_px']:.4f} px RMS, "
                f"{dev['max_dev_px']:.4f} px max; slope bound {dev['slope_bound']:.4f} -> {os.path.relpath(path, vc.HERE)}")
    log.save(out)


if __name__ == "__main__":
    main()
