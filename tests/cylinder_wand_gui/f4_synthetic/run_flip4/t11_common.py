"""T11 (detection noise): the machinery the T11 scripts share.  Plan: 2026-10-01-noise-plan.md.

Each run takes the C++ camera's exact images of synthetic wand frames (rigs.wand_frames with no noise), adds noise
from one of the models of `NoiseModel`, calibrates, and measures the result against the truth:
  the free camera parameters (local coordinates, as T4), as z-scores against three covariances:
    crb   the Cramer-Rao bound at the truth for this data set, with each detection's own sigma (the correctly
          weighted bound; for uniform Gaussian noise the usual one),
    fit   the covariance the fit reports (one sigma, estimated from its residuals),
    sand  the covariance of the unweighted estimator the calibration uses, under the true noise (sandwich
          S^-1 M S^-1 over all observations; equal to crb for uniform noise);
  per camera: f, distance from the axis, azimuth, height, centre, rotation;
  the fit's sigma, RMS, and the observations it rejected (per camera, per end, among wide-noise detections);
  unseen points (7x7x7 grid over the cloud), re-triangulated (maximum likelihood) from their exact images with the
    fitted cameras: 3-D error in um, raw (camera-1 gauge), after removing the best common spin about the axis and
    slide along it (unobservable: the error relative to the tube), and after the best rigid motion (the shape and
    scale error, what distances and velocities feel); and the re-triangulated residual in px (as T7);
  100 held-out frames, triangulated without the length constraint: wand-length error in um, from their noisy
    detections (what one sees in practice) and from their exact ones (the calibration's share).
The scripts run their cases in parallel processes (one BLAS thread each).
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
from rigs import cm  # noqa: E402
import cylinder_wand as cw  # noqa: E402

L_WAND = 10.0
N_FRAMES = 500
N_HELD = 100
HALF = 20.0
SENSOR = (3200, 2400)
WORKERS = 16
_g = np.linspace(-HALF, HALF, 7)
GRID = np.array([[x, y, z] for x in _g for y in _g for z in _g])
EY = np.array([0.0, 1.0, 0.0])


def reference():
    """The reference rig of T3, T4, T6-T10: three cam1 replicas at azimuths 0, 45, 90 deg, 1094 mm from the axis."""
    return rigs.reference_rig(3, 45.0, sensor=SENSOR)


# ------------------------------------------------------------------ noise
def _legendre(n, x):
    return (np.ones_like(x), x, 0.5 * (3 * x ** 2 - 1), 0.5 * (5 * x ** 3 - 3 * x))[n]


class NoiseModel:
    """Detection noise, per coordinate unless stated.  `apply(uv, rng)` returns (noisy uv, each detection's sigma
    (F, Nc, 2), wide flags (F, Nc, 2)).  The sigma is what a correctly weighted fit would use: the Gaussian sigma, or
    for Student-t the RMS.  Kinds:
      gauss    {"sigma": s}
      per_cam  {"sigma": [s per camera]}
      per_end  {"sigma": [s_A, s_B]}
      mixture  {"sigma": s, "wide": s_w, "p": p}: a detection is wide (both coordinates) with probability p
      student  {"scale": s, "nu": nu}: bivariate t (one chi-square per detection), RMS s sqrt(nu / (nu - 2))
      field    {"rms": a, "sigma": s}: Gaussian s plus, per camera, an image-fixed smooth pattern (Legendre products
               of degree 2 and 3 in u and v over the box the wand covers on that sensor), a px RMS per coordinate
               over the box.  The pattern is drawn when the model is built (`uv0`: the exact training images) and is
               the same for every detection of that camera, held-out frames included."""

    def __init__(self, spec, cams=None, uv0=None, rng=None):
        self.spec = dict(spec)
        self.kind = spec["kind"]
        self.field = []
        if self.kind == "field":
            terms = [(p, k - p) for k in (2, 3) for p in range(k + 1)]
            g = np.stack(np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21)), -1).reshape(-1, 2)
            for c in range(len(cams)):
                q = uv0[:, c].reshape(-1, 2)
                q = q[np.all(np.isfinite(q), axis=1)]
                lo, hi = q.min(axis=0), q.max(axis=0)
                coef = rng.normal(size=(len(terms), 2))
                val = self._basis(g, terms) @ coef
                coef *= spec["rms"] / math.sqrt(float(np.mean(val ** 2)))
                self.field.append((lo, hi, terms, coef))

    @staticmethod
    def _basis(xn, terms):
        return np.stack([_legendre(p, xn[:, 0]) * _legendre(q, xn[:, 1]) for p, q in terms], axis=1)

    def pattern(self, c, uv):
        """Camera c's image-fixed pattern at pixels uv (n, 2)."""
        lo, hi, terms, coef = self.field[c]
        return self._basis(2.0 * (uv - lo) / (hi - lo) - 1.0, terms) @ coef

    @property
    def rms(self):
        """RMS per coordinate of the noise (the Gaussian-equivalent sigma)."""
        s = self.spec
        if self.kind == "gauss":
            return s["sigma"]
        if self.kind == "mixture":
            return math.sqrt((1 - s["p"]) * s["sigma"] ** 2 + s["p"] * s["wide"] ** 2)
        if self.kind == "student":
            return s["scale"] * math.sqrt(s["nu"] / (s["nu"] - 2))
        if self.kind == "field":
            return s["sigma"]
        return float("nan")

    def apply(self, uv, rng):
        s, k = self.spec, self.kind
        F, Nc = uv.shape[:2]
        wide = np.zeros((F, Nc, 2), bool)
        if k == "gauss":
            sig = np.full((F, Nc, 2), float(s["sigma"]))
        elif k == "per_cam":
            sig = np.broadcast_to(np.asarray(s["sigma"], float)[None, :, None], (F, Nc, 2)).copy()
        elif k == "per_end":
            sig = np.broadcast_to(np.asarray(s["sigma"], float)[None, None, :], (F, Nc, 2)).copy()
        elif k == "mixture":
            wide = rng.random((F, Nc, 2)) < s["p"]
            sig = np.where(wide, s["wide"], s["sigma"])
        elif k == "student":
            sig = np.full((F, Nc, 2), self.rms)
        elif k == "field":
            sig = np.full((F, Nc, 2), float(s["sigma"]))
        else:
            raise ValueError(k)
        if k == "student":
            chi = rng.chisquare(s["nu"], size=(F, Nc, 2)) / s["nu"]
            e = rng.normal(size=uv.shape) * (s["scale"] / np.sqrt(chi))[..., None]
        else:
            e = rng.normal(size=uv.shape) * sig[..., None]
        if k == "field":
            for c in range(Nc):
                q = uv[:, c].reshape(-1, 2)
                d = np.full_like(q, np.nan)
                fin = np.all(np.isfinite(q), axis=1)
                d[fin] = self.pattern(c, q[fin])
                e[:, c] += d.reshape(F, 2, 2)
        return uv + e, sig, wide


# ------------------------------------------------------------------ data
class DataSet:
    """One synthetic data set: training frames (noisy), held-out frames (exact and noisy), and per observation of
    the training data its noise sigma, wide flag and true noise."""

    def __init__(self, truth, tube, spec, seed, n_frames=N_FRAMES, half=HALF, centre=(0.0, 0.0, 0.0), held=True,
                 wd=None):
        rng = np.random.default_rng(seed)
        self.truth, self.tube, self.spec = truth, tube, spec
        self.sizes = [(c.width, c.height) for c in truth]
        self.wd = wd if wd is not None else rigs.wand_frames(truth, tube, n_frames, L_WAND, rng, half=(half,) * 3,
                                                             centre=centre, noise_px=0.0)
        self.noise = NoiseModel(spec, truth, self.wd.uv, rng)
        uv, sig, wide = self.noise.apply(self.wd.uv, rng)
        self.uv_exact = self.wd.uv
        self.data = cw.WandData(uv, L_WAND, self.sizes)
        self.sig, self.wide = sig, wide
        obs = cw.observations(self.data)
        f0 = obs.frame_ids[obs.frame]
        self.sig_obs = sig[f0, obs.cam, obs.end]
        self.wide_obs = wide[f0, obs.cam, obs.end]
        self.noise_obs = obs.uv - self.uv_exact[f0, obs.cam, obs.end]
        self.held = None
        if held:
            wh = rigs.wand_frames(truth, tube, N_HELD, L_WAND, rng, half=(half,) * 3, centre=centre, noise_px=0.0)
            uvh, _s, _w = self.noise.apply(wh.uv, rng)
            self.held = (wh, uvh)

    def obs(self):
        return cw.observations(self.data)


# ------------------------------------------------------------------ fits
# T11 characterised the calibration as it was before the noise weights (T12): equal weights, and only the best two
# bootstrap starts.  These defaults keep a rerun of T11 meaning the same; T12 passes the new behaviour explicitly.
T11_REFINE = {"noise_weights": "none"}
T11_CALIBRATE = {"noise_weights": "none", "more_starts_px": float("inf")}


def local_fit(ds: DataSet, rng, fit_L=False, **refine_kw):
    """Stages 3-4 from a start 2 deg / 20 mm / 5 % in f off the truth (as T4).  refine_kw: passed to `refine`
    (default: T11's equal weights)."""
    refine_kw = refine_kw or T11_REFINE
    obs = ds.obs()
    layout = cw.Layout(len(ds.truth), fit_L=fit_L)
    start = rigs.perturb_rig(ds.truth, rng, rot_deg=2.0, trans_mm=20.0, f_rel=0.05)
    cams3, info3 = cw.coarse_fit(start, ds.tube, obs, L_WAND, layout)
    sol = cw.refine(cams3, ds.tube, cw.initial_frames(cams3, ds.tube, obs, L_WAND), L_WAND, obs, layout, **refine_kw)
    sol.info["stage3"] = info3
    return sol


def zmax_between(sol, ref):
    lay = ref.layout
    dl = rigs.local_delta(sol.cams, ref.cams)
    d = np.array([dl[c][j] for c in range(len(ref.cams)) for j in range(len(lay.cols)) if lay.pos[c, j] >= 0])
    return float(np.max(np.abs(d) / np.sqrt(np.diag(ref.cov))))


def scratch_pair(ds: DataSet, calib=None):
    """The calibration from nothing against the same calibration started at the truth (the global minimum).
    Success as T5: RMS within 1e-6 px and every parameter within 0.01 sigma, on the same active set.  calib: keyword
    arguments of `calibrate` (default: T11's behaviour; {} for the current defaults)."""
    calib = T11_CALIBRATE if calib is None else calib
    t0 = time.time()
    ref = cw.calibrate(ds.data, ds.tube, start=ds.truth, **calib)
    t_ref = time.time() - t0
    rec = {"ref_rms_px": ref.rms, "ref_active": int(ref.obs.active.sum()), "n_obs": int(len(ref.obs)),
           "ref_seconds": t_ref}
    t1 = time.time()
    try:
        sol = cw.calibrate(ds.data, ds.tube, **calib)
    except Exception as exc:  # noqa: BLE001 -- a failed start is a result
        rec.update({"ok": False, "error": repr(exc), "seconds": time.time() - t1})
        return rec
    z = zmax_between(sol, ref)
    drms = abs(sol.rms - ref.rms)
    same = bool(np.array_equal(sol.obs.active, ref.obs.active))
    b = sol.info.get("bootstrap") or {}
    rec.update({"ok": bool(drms < 1e-6 and z < 0.01 and same), "rms_px": sol.rms, "max_diff_sigma": z,
                "rms_diff_px": drms, "same_active": same, "active": int(sol.obs.active.sum()),
                "affine_rms_px": b.get("affine_rms_px"), "candidates": b.get("candidates"),
                "bootstrap_s": b.get("seconds"), "stage3_rms_px": sol.info["stage3"]["rms_px"],
                "starts_tried": sol.info.get("starts_tried"), "seconds": time.time() - t1})
    return rec


# ------------------------------------------------------------------ bounds at the truth
def bounds(ds: DataSet, layout):
    """(crb, sand, S): the correctly weighted Cramer-Rao covariance, the unweighted estimator's sandwich
    covariance under the true per-detection sigmas, and the unweighted reduced information, all over every
    observation at the truth (frames marginalised)."""
    obs = ds.obs()
    fr = cw.frames_from_ends(ds.wd.XA[obs.frame_ids], ds.wd.XB[obs.frame_ids])
    ev = cw.evaluate(ds.truth, ds.tube, fr, L_WAND, obs, layout)
    ok = ev.ok
    w = np.where(ok, 1.0 / ds.sig_obs ** 2, 0.0)
    V, gV, U, gU, W = cw.normal_blocks(ev, obs.frame, obs.n_frames, weights=w)
    Sw, *_ = cw.reduced_system(V, gV, U, gU, W, 0.0)
    crb = np.linalg.inv(Sw)
    Jg = np.where(ok[:, None, None], ev.Jg, 0.0)
    Jf = np.where(ok[:, None, None], ev.Jf, 0.0)
    Uf = np.zeros((obs.n_frames, 5, 5))
    np.add.at(Uf, obs.frame, np.einsum("nai,naj->nij", Jf, Jf))
    Wf = np.zeros((obs.n_frames, Jg.shape[2], 5))
    np.add.at(Wf, obs.frame, np.einsum("nap,naj->npj", Jg, Jf))
    K = np.einsum("fij,fpj->fip", np.linalg.inv(Uf), Wf)           # U^-1 W^T per frame
    Jh = Jg - np.einsum("naj,njp->nap", Jf, K[obs.frame])           # camera columns, frames projected out
    S = np.einsum("nap,naq->pq", Jh, Jh)
    M = np.einsum("n,nap,naq->pq", np.where(ok, ds.sig_obs ** 2, 0.0), Jh, Jh)
    Si = np.linalg.inv(S)
    return crb, Si @ M @ Si, S


# ------------------------------------------------------------------ measurements
def delta_vec(cams, truth, layout):
    dl = rigs.local_delta(cams, truth)
    d = np.zeros(layout.P)
    for c in range(len(truth)):
        for j in range(len(layout.cols)):
            if layout.pos[c, j] >= 0:
                d[layout.pos[c, j]] = dl[c][j]
    return d


def physical(cams, truth):
    """Per camera: f (relative), distance from the axis (mm), azimuth (deg), height (mm), centre (mm), rotation
    (deg) errors."""
    out = []
    for e, t in zip(cams, truth):
        az_e, az_t = math.degrees(math.atan2(e.C[0], e.C[2])), math.degrees(math.atan2(t.C[0], t.C[2]))
        out.append({"f_rel": e.f / t.f - 1.0, "distance_mm": math.hypot(e.C[0], e.C[2]) - math.hypot(t.C[0], t.C[2]),
                    "azimuth_deg": (az_e - az_t + 180.0) % 360.0 - 180.0, "height_mm": float(e.C[1] - t.C[1]),
                    "centre_mm": float(np.linalg.norm(e.C - t.C)), "rotation_deg": cm.rotation_angle_deg(e.R, t.R)})
    return out


def unseen(fit_cams, truth, tube, P=GRID):
    """Unseen points re-triangulated from their exact images with the fitted cameras (see the module docstring)."""
    nc = len(truth)
    ok = np.ones(len(P), bool)
    uv_t = []
    for ct in truth:
        rw = cm.rows([ct], np.zeros(len(P), int))
        pr = cm.project(rw, P, cm.pinhole_start(rw, P), tube)
        ok &= pr.ok & (pr.uv[:, 0] >= 0) & (pr.uv[:, 0] < ct.width) & (pr.uv[:, 1] >= 0) & (pr.uv[:, 1] < ct.height)
        uv_t.append(pr.uv)
    idx = np.flatnonzero(ok)
    m = len(idx)
    cam_idx = np.tile(np.arange(nc), m)
    pt_idx = np.repeat(np.arange(m), nc)
    uvs = np.stack([uv_t[c][idx] for c in range(nc)], axis=1).reshape(-1, 2)
    X, r, _cov = cw.triangulate_ml(fit_cams, tube, cam_idx, uvs, pt_idx, P[idx])
    E = X - P[idx]
    G = np.zeros((3 * m, 2))
    G[:, 0] = np.cross(EY, P[idx]).reshape(-1)
    G[:, 1] = np.tile(EY, m)
    coef = np.linalg.lstsq(G, E.reshape(-1), rcond=None)[0]
    Es = (E.reshape(-1) - G @ coef).reshape(-1, 3)
    # after the best rigid motion (Kabsch): the shape and scale errors, what distances and velocities feel
    A, B = X - X.mean(axis=0), P[idx] - P[idx].mean(axis=0)
    Uu, _s, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ Uu.T)))])
    Er = A @ (Vt.T @ D @ Uu.T).T - B
    return {"n_points": m, "raw_um": 1e3 * math.sqrt(float(np.mean(np.sum(E ** 2, axis=1)))),
            "stab_um": 1e3 * math.sqrt(float(np.mean(np.sum(Es ** 2, axis=1)))),
            "rigid_um": 1e3 * math.sqrt(float(np.mean(np.sum(Er ** 2, axis=1)))),
            "spin_urad": 1e6 * float(coef[0]), "slide_um": 1e3 * float(coef[1]),
            "retri_px": math.sqrt(float(np.nanmean(np.sum(r ** 2, axis=1))))}


def held_out_lengths(fit_cams, ds: DataSet):
    """Held-out frames triangulated by maximum likelihood without the length constraint: length - L (mm), from
    the noisy detections and from the exact ones."""
    wh, uvh = ds.held
    out = {}
    for name, uv in (("noisy", uvh), ("exact", wh.uv)):
        F = uv.shape[0]
        rows = [(f, c, e) for f in range(F) for e in range(2) for c in range(uv.shape[1])
                if np.all(np.isfinite(uv[f, c, e]))]
        fr_ = np.array([q[0] for q in rows])
        ca_ = np.array([q[1] for q in rows])
        en_ = np.array([q[2] for q in rows])
        X0 = np.zeros((2 * F, 3))
        X0[0::2], X0[1::2] = wh.XA, wh.XB
        X, _r, _cov = cw.triangulate_ml(fit_cams, ds.tube, ca_, uv[fr_, ca_, en_], fr_ * 2 + en_, X0)
        d = np.linalg.norm(X[1::2] - X[0::2], axis=1) - L_WAND
        out[name] = {"rms_um": 1e3 * math.sqrt(float(np.nanmean(d ** 2))), "mean_um": 1e3 * float(np.nanmean(d))}
    return out


def measure(sol, ds: DataSet, crb, sand, extra_unseen=True):
    """Everything T11 records about one fit against the truth."""
    lay = sol.layout
    d = delta_vec(sol.cams, ds.truth, lay)
    if lay.fit_L:
        d[lay.L_index] = sol.L - L_WAND
    obs = sol.obs
    rej = ~obs.active
    rec = {"rms_px": sol.rms, "sigma_fit_px": sol.sigma_px, "robust_sigma_px": sol.info.get("robust_sigma_px"),
           "noise_rms_px": math.sqrt(float(np.mean(ds.noise_obs ** 2))),
           "noise_rms_active_px": math.sqrt(float(np.mean(ds.noise_obs[obs.active] ** 2))),
           "n_obs": int(len(obs)), "rejected": int(rej.sum()),
           "rejected_per_cam": [int(np.sum(rej & (obs.cam == c))) for c in range(obs.n_cams)],
           "obs_per_cam": [int(np.sum(obs.cam == c)) for c in range(obs.n_cams)],
           "rejected_per_end": [int(np.sum(rej & (obs.end == e))) for e in (0, 1)],
           "obs_per_end": [int(np.sum(obs.end == e)) for e in (0, 1)],
           "wide": int(ds.wide_obs.sum()), "wide_rejected": int(np.sum(rej & ds.wide_obs)),
           "lm_iterations": len(sol.history), "rounds": sol.info.get("rounds"), "cycle": sol.info.get("cycle"), "flipped": (sol.info.get("noise_weights") or {}).get("flipped"),
           "delta": d.tolist(),
           "z_crb": (d[:crb.shape[0]] / np.sqrt(np.diag(crb))).tolist(),
           "z_fit": (d / np.sqrt(np.diag(sol.cov))).tolist(),
           "z_sand": (d[:sand.shape[0]] / np.sqrt(np.diag(sand))).tolist(),
           "phys": physical(sol.cams, ds.truth), "pred": predicted_physical(ds.truth, lay, crb),
           "pred_sand": predicted_physical(ds.truth, lay, sand)}
    if lay.fit_L:
        rec["L_mm"] = sol.L
        rec["L_sigma_mm"] = math.sqrt(sol.cov[lay.L_index, lay.L_index])
    if extra_unseen:
        rec["unseen"] = unseen(sol.cams, ds.truth, ds.tube)
        if ds.held is not None:
            rec["held"] = held_out_lengths(sol.cams, ds)
    return rec


def predicted_physical(truth, layout, cov):
    """Per camera sigma of f (relative), distance (mm), azimuth (deg), height (mm) from a covariance."""
    return [{k: s[k] for k in ("f_rel", "distance_mm", "azimuth_deg", "height_mm")}
            for s in cw.camera_sigmas(truth, layout, cov)]


# ------------------------------------------------------------------ aggregation
PHYS_KEYS = ("f_rel", "distance_mm", "azimuth_deg", "height_mm")


def summarise(runs, names):
    """Pooled statistics of a list of `measure` records (one per data set) with their predicted sigmas."""
    out = {"n": len(runs)}
    for key in ("z_crb", "z_fit", "z_sand"):
        Z = np.array([r[key] for r in runs])
        n = len(Z)
        t = Z.mean(axis=0) / (Z.std(axis=0, ddof=1) / math.sqrt(n))
        i = int(np.argmax(np.abs(t)))
        out[key] = {"mean": float(Z.mean()), "std": float(Z.std(ddof=1)), "rms": float(np.sqrt(np.mean(Z ** 2))),
                    "max_abs_t": float(abs(t[i])), "max_t_param": names[i] if i < len(names) else "L",
                    "rms_by_param": np.sqrt(np.mean(Z ** 2, axis=0)).tolist(),
                    "mean_by_param": Z.mean(axis=0).tolist()}
    nc = len(runs[0]["phys"])
    phys = {}
    for k in PHYS_KEYS + ("centre_mm", "rotation_deg"):
        E = np.array([[r["phys"][c][k] for c in range(nc)] for r in runs])
        rms = np.sqrt(np.mean(E ** 2, axis=0))
        phys[k] = {"rms_per_cam": rms.tolist(), "mean_per_cam": E.mean(axis=0).tolist()}
        if k in PHYS_KEYS:
            for src, name in (("pred", "pred_per_cam"), ("pred_sand", "pred_sand_per_cam")):
                P = np.array([[r[src][c][k] for c in range(nc)] for r in runs])
                phys[k][name] = np.sqrt(np.mean(P ** 2, axis=0)).tolist()
            pr = np.array(phys[k]["pred_per_cam"])
            sel = pr > 0          # camera 1's azimuth is fixed by the gauge
            phys[k]["ratio"] = float(np.sqrt(np.mean((rms[sel] / pr[sel]) ** 2)))
    out["phys"] = phys
    for k in ("rms_px", "sigma_fit_px", "robust_sigma_px", "noise_rms_px", "noise_rms_active_px", "rejected",
              "wide", "wide_rejected", "lm_iterations", "seconds"):
        v = [r[k] for r in runs if r.get(k) is not None]
        if v:
            out[k] = float(np.mean(v))
    out["rejected_per_cam"] = np.sum([r["rejected_per_cam"] for r in runs], axis=0).tolist()
    out["obs_per_cam"] = np.sum([r["obs_per_cam"] for r in runs], axis=0).tolist()
    out["rejected_per_end"] = np.sum([r["rejected_per_end"] for r in runs], axis=0).tolist()
    out["obs_per_end"] = np.sum([r["obs_per_end"] for r in runs], axis=0).tolist()
    if "unseen" in runs[0]:
        for k in ("raw_um", "stab_um", "rigid_um", "retri_px"):
            out["unseen_" + k] = float(np.sqrt(np.mean([r["unseen"][k] ** 2 for r in runs])))
    if "held" in runs[0]:
        for k in ("noisy", "exact"):
            out["held_" + k + "_rms_um"] = float(np.sqrt(np.mean([r["held"][k]["rms_um"] ** 2 for r in runs])))
            out["held_" + k + "_mean_um"] = float(np.mean([r["held"][k]["mean_um"] for r in runs]))
    return out


# ------------------------------------------------------------------ parallel runner
def run_parallel(fn, tasks, log, label=lambda t: str(t), workers=WORKERS):
    """fn(task) -> dict in worker processes; results in task order.  An exception in a task is a result
    ({"error": ...}), logged, not a crash."""
    results = [None] * len(tasks)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=ctx) as ex:
        futs = {ex.submit(_guarded, fn, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            r = results[i]
            msg = r.get("line") or ("ERROR " + r["error"].splitlines()[-1] if "error" in r else "")
            log(f"  [{done:3d}/{len(tasks)} {time.time() - t0:5.0f} s] {label(tasks[i])}: {msg}")
    return results


def _guarded(fn, task):
    try:
        return fn(task)
    except Exception:  # noqa: BLE001
        return {"error": traceback.format_exc()}
