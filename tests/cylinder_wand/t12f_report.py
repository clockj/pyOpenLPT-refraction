"""T12f: what the calibration report says about the noise weights, in five typical cases.

Each case is calibrated from scratch with calibrate's defaults and written with cylinder_wand_io.write_result to
t12f_report/<case>/ (camFile/, calibration_result.json, calibration_report.txt), so the reports can be read as a
user would read them:
  1  three cameras 45 deg apart, cam2 at 0.5 px, the others at 0.1 px      (T11c c1, data set 0)
  2  three cameras, all at 0.1 px                                         (T11a, 0.1 px, data set 0)
  3  two cameras 90 deg apart, cam2 at 0.5 px                             (T12c, data set 1)
  4  two cameras 45 deg apart, cam2 at 0.5 px, only 150 frames            (fewer frames: may not separate)
  5  three cameras, wand end A at 0.2 px and B at 0.1 px, noise_weights "camera_end"   (T11c c2, data set 0)
Checks: the report has the noise section; each group's detections add up; where the weights are applied, every
estimated sigma lies within 4 of its own standard errors of the truth; where they are not applied, the report says
why.  Outputs: t12f_report_out.txt, t12f_report.json and the folders t12f_report/<case>/.
"""
import t12_common as t12  # first: one BLAS thread per process
from t12_common import tc  # noqa: E402

import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
import cylinder_wand as cw  # noqa: E402
import cylinder_wand_io as cio  # noqa: E402

OUT = rigs.HERE / "t12f_report"
log = rigs.Tee(str(OUT) + "_out.txt")
CASES = [
    ("1_three_cams_cam2_noisy", 3, 45.0, {"kind": "per_cam", "sigma": [0.1, 0.5, 0.1]}, 150000, 500, "camera",
     [0.1, 0.5, 0.1]),
    ("2_three_cams_equal", 3, 45.0, {"kind": "gauss", "sigma": 0.1}, 112000, 500, "camera", [0.1, 0.1, 0.1]),
    ("3_two_cams_90deg_cam2_noisy", 2, 90.0, {"kind": "per_cam", "sigma": [0.1, 0.5]}, 212001, 500, "camera",
     [0.1, 0.5]),
    ("4_two_cams_45deg_cam2_noisy_150_frames", 2, 45.0, {"kind": "per_cam", "sigma": [0.1, 0.5]}, 230000, 150,
     "camera", [0.1, 0.5]),
    ("5_three_cams_end_A_noisy", 3, 45.0, {"kind": "per_end", "sigma": [0.2, 0.1]}, 151000, 500, "camera_end",
     [0.2, 0.1, 0.2, 0.1, 0.2, 0.1]),
]


def task(i):
    name, nc, spacing, spec, seed, n_frames, mode, true_sig = CASES[i]
    t0 = time.time()
    truth = rigs.reference_rig(nc, spacing, sensor=tc.SENSOR)
    ds = tc.DataSet(truth, rigs.CASE_TUBE, spec, seed=seed, n_frames=n_frames, held=False)
    sol = cw.calibrate(ds.data, ds.tube, noise_weights=mode)
    folder = OUT / name
    if folder.exists():
        shutil.rmtree(folder)
    res = cio.write_result(sol, str(folder))
    report = (folder / "calibration_report.txt").read_text(encoding="utf-8")
    nz = res["noise"]
    checks = {"section": "Noise levels and weights" in report,
              "counts": all(g["n_used"] + g["n_not_used"] == g["n_total"] for g in nz["groups"])}
    if nz["weights_used"]:
        z = [abs(g["sigma_px"] / t - 1.0) / g["rel_se"] for g, t in zip(nz["groups"], true_sig)]
        checks["sigma_within_4_se"] = bool(max(z) <= 4.0)
    else:
        checks["says_why"] = bool(nz["why"])
    return {"case": name, "noise": nz, "checks": checks, "pass": all(checks.values()), "report": report,
            "seconds": time.time() - t0,
            "line": f"weights {'applied' if nz['weights_used'] else 'NOT applied'}; sigma "
                    f"{[None if g['sigma_px'] is None else round(g['sigma_px'], 4) for g in nz['groups']]} | {checks}"}


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    log("T12f: the calibration report's noise section, five cases from scratch")
    res = tc.run_parallel(task, list(range(len(CASES))), log, label=lambda i: CASES[i][0])
    for r in res:
        log("")
        log(f"===== {r['case']}  ({'PASS' if r['pass'] else 'FAIL'}; report in t12f_report/{r['case']}/) =====")
        for ln in r["report"].rstrip("\n").split("\n"):
            log(ln)
    ok = all(r["pass"] for r in res)
    log("")
    log(f"  {'PASS' if ok else 'FAIL'}  {sum(r['pass'] for r in res)}/{len(res)} reports as expected ({time.time() - t0:.0f} s)")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump({"cases": [{k: v for k, v in r.items() if k not in ("line", "report")} for r in res], "pass": ok},
                  fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
