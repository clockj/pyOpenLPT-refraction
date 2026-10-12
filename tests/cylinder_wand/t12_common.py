"""T12 (noise weights): helpers shared by the T12 scripts.  Plan: 2026-10-02-weights-plan.md.

Ratio windows: a camera's free parameters move together (T11a's recheck: the heights of two cameras correlate at
0.9), so 40 data sets are close to 40 draws of a few modes and an RMS over them has a standard error of up to ~11 %.
A ratio passes when it lies within 0.8-1.2 or within 3 bootstrap standard errors of 1, whichever is wider.
"""
import t11_common as tc  # first: one BLAS thread per process

import math  # noqa: E402

import numpy as np  # noqa: E402

N_BOOT = 1000


def boot_rms_se(Z, rng=None):
    """Standard error of sqrt(mean(Z^2)) over the rows (data sets) of Z, by bootstrap over the rows."""
    rng = np.random.default_rng(7) if rng is None else rng
    n = len(Z)
    vals = [math.sqrt(float(np.mean(Z[rng.integers(0, n, n)] ** 2))) for _ in range(N_BOOT)]
    return float(np.std(vals, ddof=1))


def window_ok(ratio, se):
    return bool(abs(ratio - 1.0) <= max(0.2, 3.0 * se))


def per_camera(runs, key, lay):
    """Per camera: the RMS of its parameters' z-scores (key: z_crb, z_fit or z_sand) over the data sets, and its
    bootstrap standard error."""
    Z = np.array([r[key] for r in runs])
    out = []
    for c in range(lay.n_cams):
        idx = [int(g) for g in lay.pos[c] if g >= 0]
        sub = Z[:, idx]
        out.append((float(np.sqrt(np.mean(sub ** 2))), boot_rms_se(sub)))
    return out


def sigma_errors(runs, true_sigma):
    """RMS over the data sets of each noise group's estimated sigma over the true one, minus 1; and the mean."""
    S = np.array([r["sigma_group_px"] for r in runs if r.get("sigma_group_px")])
    if not len(S):
        return None, None
    rel = S / np.asarray(true_sigma, float)[None, :] - 1.0
    return np.sqrt(np.mean(rel ** 2, axis=0)).tolist(), np.mean(rel, axis=0).tolist()


def weights_record(sol):
    nw = sol.info.get("noise_weights") or {}
    return {"used": nw.get("used"), "separable": nw.get("separable"), "sigma_group_px": nw.get("sigma_group_px"),
            "rel_se": nw.get("rel_se"), "variance_factor": sol.info.get("variance_factor"),
            "rounds": sol.info.get("rounds")}


__all__ = ["tc", "boot_rms_se", "window_ok", "per_camera", "sigma_errors", "weights_record"]
