"""E1d (Phase E): the predictors on synthetic tracks with known truth (no shaking, no selection by W).

  1. The Kalman code: tracks generated from each Kalman model (cv, ca) with B300's chosen q/r (E2) and r = (0.0055
     mm)^2.  Checks: the filter of the generating model with the true q/r has the lowest mean squared one-step error
     of all predictors and q/r values; its squared innovations divided by its own predicted variance average 1; the
     likelihood recovers r; the q/r chosen by the mean squared error is the true one (to one grid step).
  2. W's origin dependence: straight tracks with B300's velocities and position noise, placed over the view volume;
     W's mean error away from 0, per axis, by the distance from 0, against the real tracks (E1b) and against MD.
  3. The ranking without shaking: tracks with B300's speeds, an Ornstein-Uhlenbeck acceleration with the strength and
     correlation time measured on B300's tracks (15-point quadratic fits), and Gaussian noise of B300's size.  Every
     predictor of E1b, with k and q/r chosen on half of these tracks, scored on the other half against the true
     positions and against the noisy next point.

Writes results/e1d_synthetic.txt and .json.  Needs results/e1b_replay.json (run e1b_replay.py first).
    C:\\Users\\zcloc\\.conda\\envs\\refract\\python.exe tests/stb_improve/e1d_synthetic.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import predictors as pr  # noqa: E402
from e1b_replay import LAM_GRID, LIN_K, QUAD_K, BANDS, metrics  # noqa: E402

SEED = 20261006
VOLUME = np.array([[-81.0, 54.0], [-42.0, 59.0], [-36.0, 52.5]])     # B300's view volume, mm
SIGMA = 0.0055
_lines = []
results = {"checks": {}}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def check(name, ok, detail):
    results["checks"][name] = {"ok": bool(ok), "detail": detail}
    log(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def flat(Ps):
    lens = np.array([len(p) for p in Ps])
    st = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return pr.Tracks(np.concatenate(Ps), st, lens, np.zeros(len(Ps), np.int8), np.arange(len(Ps)),
                     np.zeros(len(Ps), int))


def simulate_kf(kind, lam, r, n_tracks, length, rng):
    """Tracks drawn from the Kalman model itself (per axis independent), with r in mm^2 and q = lam r."""
    kf = pr.Kalman(kind, lam)
    d, F, G = kf.d, kf.F, kf.G
    q = lam * r
    X = np.zeros((n_tracks, 3, d))
    X[:, :, 0] = rng.uniform(VOLUME[:, 0], VOLUME[:, 1], size=(n_tracks, 3))
    X[:, :, 1] = rng.normal(0, 0.06, size=(n_tracks, 3))
    truth = np.zeros((n_tracks, length, 3))
    for t in range(length):
        truth[:, t] = X[:, :, 0]
        X = X @ F.T + np.sqrt(q) * rng.normal(size=(n_tracks, 3))[:, :, None] * G[None, None, :]
    meas = truth + np.sqrt(r) * rng.normal(size=truth.shape)
    return truth, meas


def all_predictions(tr, U, n, lam_cv, lam_ca, extra_lams=()):
    """Every predictor of E1b at the points U; returns {label: (len(U), 3)}."""
    out = {"W": pr.wiener(tr.P, U, n), "MD": pr.mean_step(tr.P, U, n)}
    for k in LIN_K:
        out[f"LIN-{k}"] = pr.poly(tr.P, U, n, 1, k)
    for k in QUAD_K:
        out[f"QUAD-{k}"] = pr.poly(tr.P, U, n, 2, k)
    for kind, lam in (("cv", lam_cv), ("ca", lam_ca)):
        p, _ = pr.Kalman(kind, lam).run(tr)
        out[f"KF-{kind.upper()}"] = p[U]
    for kind, lam in extra_lams:
        p, _ = pr.Kalman(kind, lam).run(tr)
        out[f"KF-{kind.upper()}({lam:.3g})"] = p[U]
    return out


def part1(e1b):
    log("\n=== 1. the Kalman code on tracks drawn from its own model")
    rng = np.random.default_rng(SEED)
    r = SIGMA ** 2
    res = {}
    for kind in ("cv", "ca"):
        lam_true = e1b["b300"]["chosen"][f"KF-{kind.upper()}"]
        grid = np.array(sorted(set(np.round(LAM_GRID, 12)) | {lam_true}))
        truth, meas = simulate_kf(kind, lam_true, r, 3000, 150, rng)
        tr = flat(list(meas))
        ii = tr.index_in_track
        U = np.nonzero(ii >= 4)[0]
        y = tr.P[U]
        mse = {}
        nis = None
        for lam in grid:
            p, s = pr.Kalman(kind, lam).run(tr)
            e = p[U] - y
            mse[float(lam)] = float(np.mean(np.sum(e ** 2, axis=1)))
            if lam == lam_true:
                nis = float(np.mean(e ** 2 / (r * s[U, None])))
                r_ml = float(np.mean(e ** 2 / s[U, None]))
        other = all_predictions(tr, U, ii[U], 1.0, 1.0)
        other = {k: float(np.mean(np.sum((v - y) ** 2, axis=1))) for k, v in other.items()
                 if not k.startswith(f"KF-{kind.upper()}")}
        okind = "ca" if kind == "cv" else "cv"
        for lam in grid:
            p, _ = pr.Kalman(okind, lam).run(tr)
            other[f"KF-{okind.upper()}({lam:.3g})"] = float(np.mean(np.sum((p[U] - y) ** 2, axis=1)))
        best_lam = min(mse, key=mse.get)
        i_true = int(np.nonzero(grid == lam_true)[0][0])
        i_best = int(np.nonzero(grid == best_lam)[0][0])
        best_other = min(other, key=other.get)
        res[kind] = {"lam_true": lam_true, "lam_by_mse": best_lam, "mse_true": mse[float(lam_true)],
                     "best_other": best_other, "mse_best_other": other[best_other], "nis_mean": nis,
                     "r_ml_over_r": r_ml / r}
        check(f"1 {kind}: the true model with the true q/r has the lowest MSE",
              mse[float(lam_true)] <= other[best_other] and abs(i_best - i_true) <= 1,
              f"true q/r {lam_true:.3g}: MSE {mse[float(lam_true)]:.4g} mm^2; chosen by MSE {best_lam:.3g}; "
              f"best other predictor {best_other} {other[best_other]:.4g} mm^2")
        check(f"1 {kind}: squared innovations / predicted variance average 1", abs(nis - 1) < 0.02,
              f"{nis:.4f}")
        check(f"1 {kind}: the likelihood recovers r", abs(r_ml / r - 1) < 0.02, f"r_ML / r = {r_ml / r:.4f}")
    results["part1"] = res


def velocities_and_acceleration(tr):
    """B300's per-track mean velocity vectors, and the OU parameters of the acceleration per axis."""
    ii = tr.index_in_track
    v = (tr.P[tr.starts + tr.lengths - 1] - tr.P[tr.starts]) / np.maximum(tr.lengths - 1, 1)[:, None]
    v = v[tr.lengths >= 20]
    # acceleration from 15-point quadratic fits centred every 5 frames, on tracks of >= 60 points
    w = np.linalg.pinv(np.vander(np.arange(-7, 8, dtype=float), 3, increasing=True))[2] * 2   # acc weights
    rows = []
    for s, L in zip(tr.starts[tr.lengths >= 60], tr.lengths[tr.lengths >= 60]):
        cs = np.arange(7, L - 7, 5)
        idx = s + cs[:, None] + np.arange(-7, 8)[None, :]
        rows.append(np.einsum("j,ijk->ik", w, tr.P[idx]))
    lags = (1, 2, 4, 8)     # in steps of 5 frames
    acc = np.concatenate(rows)
    sd = acc.std(axis=0)
    ac = {}
    for lag in lags:
        a, b = [], []
        for A in rows:
            if len(A) > lag:
                a.append(A[:-lag])
                b.append(A[lag:])
        a, b = np.concatenate(a), np.concatenate(b)
        ac[5 * lag] = [float(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in range(3)]
    # correlation time from the 5- and 10-frame lags (exponential): tau = -lag / ln(rho)
    tau = []
    for k in range(3):
        rr = [ac[L][k] for L in (5, 10, 20) if ac[L][k] > 0.05]
        Ls = [L for L in (5, 10, 20) if ac[L][k] > 0.05]
        tau.append(float(np.mean([-L / np.log(rv) for L, rv in zip(Ls, rr)])) if rr else 5.0)
    return v, sd, ac, np.array(tau), len(acc)


def simulate_ou(vel, sd_a, tau, sig, n_tracks, length, rng, near_planes=False):
    """Tracks with B300's velocities, an OU acceleration (per axis: sd_a, tau frames) and Gaussian noise sig."""
    pos = rng.uniform(VOLUME[:, 0], VOLUME[:, 1], size=(n_tracks, 3))
    if near_planes:
        pos = rng.uniform(-12, 12, size=(n_tracks, 3))
    v = vel[rng.integers(0, len(vel), n_tracks)].copy()
    rho = np.exp(-1.0 / tau)
    a = rng.normal(size=(n_tracks, 3)) * sd_a
    truth = np.zeros((n_tracks, length, 3))
    for t in range(length):
        truth[:, t] = pos
        pos = pos + v + 0.5 * a
        v = v + a
        a = rho * a + np.sqrt(1 - rho ** 2) * sd_a * rng.normal(size=(n_tracks, 3))
    meas = truth + rng.normal(size=truth.shape) * sig
    return truth, meas


def part2(e1b, vel, sig_axes):
    log("\n=== 2. W's origin dependence on straight synthetic tracks (B300's velocities and noise)")
    rng = np.random.default_rng(SEED + 1)
    truth, meas = simulate_ou(vel, np.zeros(3), np.ones(3), sig_axes, 20000, 40, rng, near_planes=True)
    tr = flat(list(meas))
    ii = tr.index_in_track
    U = np.nonzero(ii >= 4)[0]
    T = truth.reshape(-1, 3)[U]
    last = tr.P[U - 1]
    preds = {"W": pr.wiener(tr.P, U, ii[U]), "MD": pr.mean_step(tr.P, U, ii[U])}
    res = {}
    log(f"  {'band':>8s} {'W synthetic':>12s} {'MD synthetic':>13s} {'W real (B300)':>14s}   mean error away from 0, mm")
    for lo, hi in BANDS:
        key = f"{lo:g}-{hi:g}" if hi < 1e9 else f">={lo:g}"
        row = {}
        for lab, p in preds.items():
            e, sg = [], []
            for ax in range(3):
                c = np.abs(last[:, ax])
                sel = (c >= lo) & (c < hi)
                e.append((p - T)[sel, ax])
                sg.append(np.sign(last[sel, ax]))
            e, sg = np.concatenate(e), np.concatenate(sg)
            row[lab] = float(np.mean(e * sg)) if len(e) else float("nan")
        real = e1b["b300"]["by_band"][key]["W"]["mean_outward"]
        row["W_real"] = real
        res[key] = row
        log(f"  {key:>8s} {row['W']:+12.4f} {row['MD']:+13.4f} {real:+14.4f}")
    w13 = res["1-3"]
    check("2 W pushes predictions away from 0 at 1-3 mm, as on the real tracks; MD does not",
          w13["W"] > 0.003 and abs(w13["MD"]) < 0.0005 and abs(w13["W"] - w13["W_real"]) < 0.5 * w13["W_real"],
          f"1-3 mm: W {w13['W']:+.4f}, MD {w13['MD']:+.4f}, W on B300 {w13['W_real']:+.4f} mm")
    results["part2"] = res


def part3(e1b, vel, sd_a, tau, sig_axes):
    log("\n=== 3. the ranking without shaking (OU acceleration, Gaussian noise)")
    rng = np.random.default_rng(SEED + 2)
    truth, meas = simulate_ou(vel, sd_a, tau, sig_axes, 8000, 150, rng)
    tr = flat(list(meas))
    ii = tr.index_in_track
    U = np.nonzero(ii >= 4)[0]
    T = truth.reshape(-1, 3)[U]
    Y = tr.P[U]
    half = (np.arange(len(tr.starts)) % 2 == 0)[tr.track_of_point[U]]
    # choose k and q/r on half A by the MSE against the noisy next point (as E2 does on real tracks)
    chosen = {}
    for fam, ks, deg in (("LIN", LIN_K, 1), ("QUAD", QUAD_K, 2)):
        mse = {k: float(np.mean(np.sum((pr.poly(tr.P, U, ii[U], deg, k) - Y)[half] ** 2, axis=1))) for k in ks}
        chosen[fam] = min(mse, key=mse.get)
    for kind in ("cv", "ca"):
        mse = {}
        for lam in LAM_GRID:
            p, _ = pr.Kalman(kind, lam).run(tr)
            mse[float(lam)] = float(np.mean(np.sum((p[U] - Y)[half] ** 2, axis=1)))
        chosen[f"KF-{kind.upper()}"] = min(mse, key=mse.get)
    preds = {"W": pr.wiener(tr.P, U, ii[U]), "MD": pr.mean_step(tr.P, U, ii[U]),
             f"LIN-{chosen['LIN']}": pr.poly(tr.P, U, ii[U], 1, chosen["LIN"]),
             f"QUAD-{chosen['QUAD']}": pr.poly(tr.P, U, ii[U], 2, chosen["QUAD"])}
    for kind in ("cv", "ca"):
        p, _ = pr.Kalman(kind, chosen[f"KF-{kind.upper()}"]).run(tr)
        preds[f"KF-{kind.upper()}"] = p[U]
    res = {"chosen": chosen, "truth": {}, "next_point": {}}
    log(f"  chosen on half A: LIN-{chosen['LIN']}, QUAD-{chosen['QUAD']}, KF-CV q/r {chosen['KF-CV']:.3g}, "
        f"KF-CA q/r {chosen['KF-CA']:.3g} (real B300: LIN-{e1b['b300']['chosen']['LIN']}, "
        f"QUAD-{e1b['b300']['chosen']['QUAD']}, KF-CV {e1b['b300']['chosen']['KF-CV']:.3g}, "
        f"KF-CA {e1b['b300']['chosen']['KF-CA']:.3g})")
    log(f"  {'predictor':10s} {'vs truth: median':>17s} {'p99':>8s}   {'vs next point: median':>22s} {'p99':>8s}"
        f"   {'real B300: median':>18s} {'p99':>8s}")
    for lab, p in preds.items():
        mt = metrics(np.linalg.norm(p - T, axis=1)[~half])
        mn = metrics(np.linalg.norm(p - Y, axis=1)[~half])
        res["truth"][lab], res["next_point"][lab] = mt, mn
        fam = lab if lab in ("W", "MD", "KF-CV", "KF-CA") else None
        real = e1b["b300"]["test"].get(lab) or (e1b["b300"]["test"].get(fam) if fam else None)
        rs = f"{real['median']:18.4f} {real['p99']:8.4f}" if real else ""
        log(f"  {lab:10s} {mt['median']:17.4f} {mt['p99']:8.4f}   {mn['median']:22.4f} {mn['p99']:8.4f}   {rs}")
    bt = min(res["truth"], key=lambda k: res["truth"][k]["p99"])
    gain = 1 - res["truth"][bt]["p99"] / res["truth"]["W"]["p99"]
    res["best_vs_truth"] = bt
    res["p99_gain_vs_truth"] = gain
    gn = 1 - min(v["p99"] for v in res["next_point"].values()) / res["next_point"]["W"]["p99"]
    res["p99_gain_vs_next_point"] = gn
    p99_ratio_real = e1b["b300"]["test"]["W"]["p99"] / e1b["b300"]["test"]["W"]["median"]
    p99_ratio_syn = res["next_point"]["W"]["p99"] / res["next_point"]["W"]["median"]
    res["W_p99_over_median"] = {"synthetic": p99_ratio_syn, "real": p99_ratio_real}
    log(f"  best against the truth: {bt} ({100 * gain:+.1f} % lower p99 than W); against the next point: "
        f"{100 * gn:+.1f} %")
    log(f"  W's p99 / median: synthetic {p99_ratio_syn:.2f}, real B300 {p99_ratio_real:.2f} (Gaussian errors give "
        f"about 2.2)")
    results["part3"] = res


def main():
    t0 = time.time()
    e1b = json.load(open(os.path.join(HERE, "results", "e1b_replay.json")))
    tr = pr.load_run("b300")
    vel, sd_a, ac, tau, n_acc = velocities_and_acceleration(tr)
    sig_axes = np.array(e1b["b300"]["noise_sigma_mm"])
    speed = np.linalg.norm(vel, axis=1)
    results["b300_motion"] = {"speed_p10_p50_p90": np.percentile(speed, [10, 50, 90]).tolist(),
                              "acc_sd_per_axis": sd_a.tolist(), "acc_autocorr": ac, "acc_tau_frames": tau.tolist(),
                              "n_acc_samples": n_acc, "noise_sigma_mm": sig_axes.tolist()}
    log(f"B300 motion: speed p10/p50/p90 {np.percentile(speed, 10):.3f} / {np.median(speed):.3f} / "
        f"{np.percentile(speed, 90):.3f} mm/frame; acceleration (15-point fits, {n_acc} samples) sd per axis "
        f"{sd_a[0]:.4f} / {sd_a[1]:.4f} / {sd_a[2]:.4f} mm/frame^2, autocorrelation at 5/10/20/40 frames "
        + " ".join(f"{L}:{np.mean(v):.2f}" for L, v in ac.items())
        + f", correlation time {tau[0]:.1f} / {tau[1]:.1f} / {tau[2]:.1f} frames; noise {sig_axes.round(4).tolist()} mm")
    part1(e1b)
    part2(e1b, vel, sig_axes)
    part3(e1b, vel, sd_a, tau, sig_axes)
    n_fail = sum(not c["ok"] for c in results["checks"].values())
    log(f"\nE1d: {len(results['checks']) - n_fail} of {len(results['checks'])} checks passed "
        f"({time.time() - t0:.0f} s)")
    with open(os.path.join(HERE, "results", "e1d_synthetic.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(_lines) + "\n")
    with open(os.path.join(HERE, "results", "e1d_synthetic.json"), "w") as f:
        json.dump(results, f, indent=1)
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
