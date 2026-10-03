"""R4b (diagnosis, 2026-10-03, after R4): the restart that the selection rule blocked on R4 seed 3111 at 1 px.

R4: the answer from scratch is pinned against window 2 (a wand end on its water side, rms 1.245158 px; the fit from the
truth 1.231979 px).  The restarts' cheapest draw was refitted to 1.231995 px, but calibrate's rule kept the pinned
answer: among results within 1.5x of the best RMS it prefers the one that keeps the most observations.  Here:
  1. the pinned answer (calibrate without restarts) and one round of restarts about it (plate_wand.perturbed_restart,
     the defaults), as calibrate runs them;
  2. both results' active observations, RMS and the rule's verdict;
  3. the proposed rule: both results' cost on the answer's own active observations and noise weights (the screen's
     terms, each with its own fitted frames, no iteration), and the restart against the fit from the truth.
Output: r4b_blocked_restart_out.txt
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prigs  # noqa: E402
import r4_group_rigs as r4  # noqa: E402

pm = prigs.pm
import plate_wand as pw  # noqa: E402

SEED, NOISE = 3111, 1.0
log = prigs.Tee(HERE / "r4b_blocked_restart_out.txt")
t_all = time.time()
rig_t, data = r4.case(SEED, NOISE)
placeholder = pw.Rig([pm.Window([0.0, 0.0, 1.0], 0.0, w.t, w.idx, w.name) for w in rig_t.wins], rig_t.cam_win)
ans = pw.calibrate(data, placeholder, output_frame=False, perturb_starts=0)
ref = pw.solve(data, [c.copy() for c in rig_t.cams], pw.Rig([w.copy() for w in rig_t.wins], rig_t.cam_win))
log(f"R4b: seed {SEED} at {NOISE:g} px, {rig_t.n_cams} cameras on windows {rig_t.cam_win}; the answer without restarts: "
    f"rms {ans.rms:.6f} px, active {int(ans.obs.active.sum())}, smallest wand-end depth per window "
    + ", ".join(f"{v:.3g}" for v in pw.window_depths(ans.cams, ans.rig, ans.frames, ans.L, ans.obs))
    + f" mm; the fit from the truth: rms {ref.rms:.6f} px, active {int(ref.obs.active.sum())}")
obs = pw.observations(data)
solve_kw = dict(fit_k1=False, fit_L=False, k_sigma=5.0, noise_weights="camera", sigma_cam_px=None, priors=[],
                verbose=False, log=print)
rng = np.random.default_rng(pw._perturb_seed(data))
t0 = time.time()
new, info = pw.perturbed_restart(ans, data, obs, obs.active.copy(), 4, 2.0, rng, solve_kw)
log(f"  one round of restarts ({time.time() - t0:.0f} s): screened costs relative to the answer's "
    + ", ".join("-" if v is None else f"{v:+.2e}" for v in info["cost_rel"])
    + (f"; the cheapest refitted: rms {new.rms:.6f} px, active {int(new.obs.active.sum())}" if new is not None else
       "; none lower"))
if new is not None:
    chosen = pw._choose([ans, new], "camera")
    log(f"  calibrate's rule (within 1.5x of the best RMS, then the most observations kept) keeps "
        f"{'the restart' if chosen is new else 'the pinned answer'}")
    o_a = pw.Obs(obs.frame, obs.cam, obs.end, obs.uv, obs.n_frames, obs.n_cams, obs.frame_ids, ans.obs.active.copy())
    c_ans = pw.bundle_adjust(ans.cams, ans.rig, ans.frames, ans.L, o_a, ans.layout, max_iter=0, weights=ans.weights)
    c_new = pw.bundle_adjust(new.cams, new.rig, new.frames, new.L, o_a, ans.layout, max_iter=0, weights=ans.weights)
    lost = ans.obs.active & ~new.obs.active
    ev = pw.evaluate(new.cams, new.rig, new.frames, new.L, o_a, rows=np.flatnonzero(lost))
    rn = np.linalg.norm(ev.r, axis=1)
    log(f"  on the answer's own {int(o_a.active.sum())} observations and weights: the answer's cost "
        f"{c_ans.info['weighted_cost']:.2f}, the restart's {c_new.info['weighted_cost']:.2f} "
        f"({c_new.info['weighted_cost'] / c_ans.info['weighted_cost'] - 1.0:+.2e}; n_fail {c_new.info['n_fail']}); "
        f"the {int(lost.sum())} observations the restart rejected have residuals "
        + (f"{rn.min():.2f}-{rn.max():.2f} px there" if lost.any() else "-"))
    # the proposed rule: every usable observation, with the answer's noise weights, its weighted squared residual capped
    # at the rejection limit, (k_sigma = 5)^2 times the answer's variance per coordinate; one that does not project
    # counts the cap
    s2 = ans.sigma_px ** 2 if ans.weights is None else float(ans.info.get("variance_factor", 1.0))
    cap = 25.0 * s2
    w_all = np.ones(len(obs)) if ans.weights is None else np.asarray(ans.weights, float)

    def capped(sol_):
        ev_ = pw.evaluate(sol_.cams, sol_.rig, sol_.frames, sol_.L, obs)
        e_ = np.where(ev_.ok, w_all * np.sum(np.where(ev_.ok[:, None], ev_.r, 0.0) ** 2, axis=1), np.inf)
        return float(np.sum(np.minimum(e_, cap))), int(np.sum(e_ >= cap))

    ca, na = capped(ans)
    cn, nn = capped(new)
    log(f"  the proposed rule, on all {len(obs)} usable observations with the answer's weights, each capped at "
        f"25 x {s2:.4f}: the answer {ca:.2f} ({na} at the cap), the restart {cn:.2f} ({nn} at the cap): "
        f"{cn / ca - 1.0:+.2e}, the {'restart' if cn < (1.0 - 1e-6) * ca else 'answer'} would be kept")
    m = prigs.same_minimum(new, ref)
    log(f"  the restart against the fit from the truth: rms {new.rms:.6f} vs {ref.rms:.6f}, active {m['active']} vs "
        f"{m['active_ref']}, max |z| {m['max_abs_z']:.2f} ({m['worst']}): "
        f"{'the same minimum' if m['same'] else 'not the same minimum'}")
log(f"done, {time.time() - t_all:.0f} s")
