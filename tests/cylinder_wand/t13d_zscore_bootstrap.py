"""T13d supplement: bootstrap standard errors of the z RMS over T13c's and T13d's 8 data sets.  Plan:
2026-10-02-ball-centres-plan.md.

A calibration's parameters move together, so the RMS of z over 8 data sets is close to 8 draws of a few modes; its
standard error comes from resampling whole data sets (2000 resamples), as T12's windows did.
Output: t13d_zscore_bootstrap_out.txt.
"""
import json
import math

import numpy as np

import rigs

OUT = rigs.HERE / "t13d_zscore_bootstrap_out.txt"


def main():
    c = json.load(open(str(rigs.HERE / "t13c_calibration.json"), encoding="utf-8"))
    d = json.load(open(str(rigs.HERE / "t13d_correction.json"), encoding="utf-8"))
    rng = np.random.default_rng(7)

    def boot(Z, n_boot=2000):
        n = len(Z)
        vals = [math.sqrt(float(np.mean(np.concatenate([Z[i] for i in rng.integers(0, n, n)]) ** 2)))
                for _ in range(n_boot)]
        return float(np.std(vals, ddof=1))

    lines = ["T13d supplement: z RMS over the 8 data sets with its bootstrap standard error (data sets resampled whole)"]
    rows = [("ideal (T13c)", c["configs"]["ideal"]["runs"])]
    for li in ("backlit", "even"):
        rows.append((f"ellipse_rodmask|{li} (T13c)", c["configs"][f"ellipse_rodmask|{li}"]["runs"]))
        rows.append((f"ellipse_rodmask|{li}|corrected (T13d)",
                     d["configs"][f"ellipse_rodmask|{li}|corrected"]["runs"]))
    for name, runs in rows:
        Z = [np.asarray(r["z"]) for r in runs if r.get("ok")]
        zr = math.sqrt(float(np.mean(np.concatenate(Z) ** 2)))
        per = [math.sqrt(float(np.mean(z ** 2))) for z in Z]
        lines.append(f"  {name:<40s} z RMS {zr:.2f} +- {boot(Z):.2f} (bootstrap SE); per data set "
                     + ", ".join(f"{v:.2f}" for v in per))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
