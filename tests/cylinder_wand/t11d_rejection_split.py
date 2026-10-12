"""T11d check: why are 1.7 % of the narrow (good) detections rejected under contaminated noise (d1)?

Reruns the first 10 of T11d's d1 data sets (the same seeds, the same fits) and splits the rejected narrow
detections into those dropped with a whole frame (the thin-frame rule: once a wide detection goes, an end seen by
fewer than two cameras takes its frame with it) and those rejected one by one in a frame that stays.
Outputs: t11d_rejection_split_out.txt, t11d_rejection_split.json.
"""
import t11_common as tc  # first: one BLAS thread per process

import json  # noqa: E402

import numpy as np  # noqa: E402

import rigs  # noqa: E402
from t11d_heavy_tails import VARIANTS  # noqa: E402

OUT = rigs.HERE / "t11d_rejection_split"
log = rigs.Tee(str(OUT) + "_out.txt")


def one(k):
    name, spec = VARIANTS[0]
    ds = tc.DataSet(tc.reference(), rigs.CASE_TUBE, spec, seed=170000 + k, held=True)
    sol = tc.local_fit(ds, np.random.default_rng(180000 + k))
    obs = sol.obs
    rej = ~obs.active
    frame_active = np.zeros(obs.n_frames, bool)
    frame_active[obs.frame[obs.active]] = True
    whole = rej & ~frame_active[obs.frame]
    narrow = ~ds.wide_obs
    return {"set": k, "narrow_rejected": int(np.sum(rej & narrow)), "narrow_with_whole_frame": int(np.sum(whole & narrow)),
            "narrow_singly": int(np.sum(rej & narrow & ~whole)), "wide_rejected": int(np.sum(rej & ds.wide_obs)),
            "frames_dropped": int(np.sum(~frame_active)), "n_obs": int(len(obs))}


def main():
    res = tc.run_parallel(one, list(range(10)), log, label=lambda k: f"d1 set {k}")
    tot = {key: sum(r[key] for r in res) for key in res[0] if key != "set"}
    log("")
    log(f"narrow detections rejected: {tot['narrow_rejected']} over 10 data sets; with a whole frame "
        f"{tot['narrow_with_whole_frame']} ({100 * tot['narrow_with_whole_frame'] / max(tot['narrow_rejected'], 1):.0f} %), "
        f"one by one {tot['narrow_singly']}; frames dropped {tot['frames_dropped']}; wide rejected {tot['wide_rejected']}")
    with open(str(OUT) + ".json", "w", encoding="utf-8") as fh:
        json.dump({"runs": res, "totals": tot}, fh, indent=1)
    log.close()


if __name__ == "__main__":
    main()
