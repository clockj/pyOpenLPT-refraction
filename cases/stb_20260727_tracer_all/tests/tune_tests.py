"""U0-U6: tuning the tracking for the dense runs (plan: 2026-10-11-plan-tuning.md, approved 11 Oct).

    python tests/tune_tests.py run [--threads 12]     the queue: U0, U1-U3, then U4-U6 as the plan's rule decides
                                                     (tests already run are skipped)
    python tests/tune_tests.py summary                tests/u_summary.txt/.json, tests/u_quality.txt/.json,
                                                     tests/u_tracks.png

Each test tracks frames 0-250 (U0: 0-30) of one run with the batch's settings, the run's finder threshold and the
tested changes, into <staging_root>\\smoke\\<test>_<run> (saves at frame 200 and at the last frame).  Before a test
starts, the C++ reader reads its config back and must find the changed settings.
Baselines:
  - T3's runs t3_35Hz_r_b_1_th30 and t3_35Hz_r_b_2_th80: their frames 0-255 used images the renumbering did not
    change, and U0 checks that the code repeats them;
  - the batch's run of 10Hz_r_b_1 on the renumbered images (U6).
Measures (plan, section 3): long tracks at frames 50-250; per-frame means over frames 151-250 of the log's counters
(IPR particles, short tracks, promotions, line-test failures, established long tracks lost or exited; lasting
promotions = net gain + losses; loss rate; level-off = lasting / loss rate); seconds per frame and hours per run;
quality (track_quality.py) of the tracks alive at the end of frame 200 and at the last frame.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import case  # noqa: E402
import make_configs  # noqa: E402
import parse_log  # noqa: E402
import smoke_test  # noqa: E402
import threshold_tests as t1  # noqa: E402
import track_quality as tq  # noqa: E402

FRAMES = (0, 250)
U0_FRAMES = (0, 30)
WINDOW = (151, 250)          # rates: means per frame over these frames
Q_FRAME = 200                # quality: the tracks alive at the end of this frame (a save frame in every run)
MARKS = [50, 100, 150, 200, 250]
SPACING = "convergence_phase.avg_spacing_vox"
SHAKE = "shake.width_vox"
DEFAULTS = {SPACING: 30.0, SHAKE: 0.25}
THRESHOLD = {"35Hz_r_b_1": 30, "35Hz_r_b_2": 80, "10Hz_r_b_1": 20}
FIRST = [("U1", {SPACING: 10.0}), ("U2", {SPACING: 5.0}), ("U3", {SHAKE: 0.5})]
RETRIES = 2                 # a test whose STB process crashes is run again, at most this many times
BATCH_MAX_SAVE = 1600        # the batch's tracks alive at frame 200 have all ended by then (loss 0.57 % per frame)
QUEUE_LOG = os.path.join(HERE, "u_queue.log")
DECISION = os.path.join(HERE, "u_decision.json")


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(QUEUE_LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def name_of(test, run):
    return f"{test.lower()}_{run}"


def test_dir(cfg, test, run):
    return smoke_test.smoke_dir(cfg, name_of(test, run))


def baseline(cfg, run):
    """(label, folder holding log.txt and Results_tracer, max save to read for the quality at Q_FRAME)."""
    if run == "10Hz_r_b_1":  # the batch's run with spacing 30, set aside on 11 Oct when the run was queued again
        return "batch 10Hz_r_b_1", case.stage_dir(cfg, run) + "_spacing30", BATCH_MAX_SAVE
    return f"T3 {run} at {THRESHOLD[run]}", smoke_test.smoke_dir(cfg, f"t3_{run}_th{THRESHOLD[run]}"), None


# ----------------------------------------------------------------------------------------------- running a test

def write_config(cfg, test, run, frames, threads, changes):
    name = name_of(test, run)
    path = smoke_test.write_config(cfg, name, run, *frames, threads=threads, threshold=THRESHOLD[run])
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    obj = c["objects"][0]
    for key, val in changes.items():
        sect, field = key.split(".")
        assert field in obj[sect], key
        obj[sect][field] = val
    c["_tune"] = {"test": test, "run": run, "changes": changes}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(make_configs.json_text(c))
    return path


def cpp_check(path, run, frames, changes):
    """The C++ reader must read the frames, the threshold and the changed settings back."""
    lpt = case.import_lpt()
    s = lpt.BasicSetting()
    s.readConfig(path)
    tc = lpt.TracerConfig()
    tc.readConfig(path + "#0", s)
    vox = s._voxel_to_mm
    got = {"frames": [s._frame_start, s._frame_end], "threshold": tc._min_obj_int,
           SPACING: tc._stb_param._radius_search_track / vox, SHAKE: tc._shake_param._shake_width / vox}
    want = {"frames": list(frames), "threshold": THRESHOLD[run], **DEFAULTS, **changes}
    bad = [k for k in want if (abs(got[k] - want[k]) > 1e-9 if isinstance(want[k], float) else got[k] != want[k])]
    return got, bad


def run_test(cfg, test, run, frames, changes, threads):
    """Run one test; a crashed STB process (no run_info.json, not stopped) is kept aside and run again."""
    d = test_dir(cfg, test, run)
    name = name_of(test, run)
    if os.path.exists(os.path.join(d, "run_info.json")) or os.path.exists(os.path.join(d, "stopped.json")):
        say(f"{test} {name}: already run, skipped")
        return
    for attempt in range(RETRIES + 1):
        if os.path.exists(d):  # a crashed attempt: keep it aside
            aside = f"{d}_crash{attempt}_{time.strftime('%Y%m%d-%H%M%S')}"
            os.rename(d, aside)
            say(f"{test} {name}: the crashed attempt is kept in {aside}")
        rc = run_attempt(cfg, test, run, frames, changes, threads, d, name)
        if rc == 0 or os.path.exists(os.path.join(d, "run_info.json")) or os.path.exists(os.path.join(d, "stopped.json")):
            return
        say(f"{test} {name}: STB CRASHED (exit code 0x{rc & 0xFFFFFFFF:08X}, no run_info.json), attempt {attempt + 1} "
            f"of {RETRIES + 1}")
    raise SystemExit(f"{test}: STB crashed {RETRIES + 1} times; the queue stops")


def run_attempt(cfg, test, run, frames, changes, threads, d, name):
    path = write_config(cfg, test, run, frames, threads, changes)
    got, bad = cpp_check(path, run, frames, changes)
    if bad:
        raise SystemExit(f"{test}: the C++ reader reads {got}, not the intended {bad}")
    say(f"{test} {name}: start ({threads} threads, frames {frames[0]}-{frames[1]}, threshold {THRESHOLD[run]}, "
        f"changes {changes or 'none'}; C++ reads {got})")
    proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(HERE), "run_stb.py"), path, "--quiet"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stopped = None
    while proc.poll() is None:
        time.sleep(15)
        t = t1.frame_times(os.path.join(d, "log.txt"))
        if len(t) >= t1.SLOW_AFTER and statistics.median(t[-t1.SLOW_AFTER:]) > t1.SLOW_S:
            proc.kill()
            proc.wait()
            stopped = {"after_frames": len(t), "median_s": statistics.median(t[-t1.SLOW_AFTER:])}
            case.save_json(os.path.join(d, "stopped.json"), stopped)
    t = t1.frame_times(os.path.join(d, "log.txt"))
    la = log_curve(os.path.join(d, "log.txt")).get(frames[1])
    say(f"{test} {name}: {'STOPPED (too slow) ' if stopped else ''}exit {proc.returncode}, {len(t)} frames, "
        f"median {statistics.median(t) if t else float('nan'):.1f} s per frame, long tracks at the end {la}")
    return proc.returncode


# ----------------------------------------------------------------------------------------------- measures

TIMING = ("t_", "time")


def counters(rec):
    """A log record without timings, at every level (each IPR round carries its own t_match and t_shake)."""
    if isinstance(rec, dict):
        return {k: counters(v) for k, v in rec.items() if not any(k.startswith(t) for t in TIMING)}
    if isinstance(rec, list):
        return [counters(v) for v in rec]
    return rec


def check_u0(cfg):
    d = test_dir(cfg, "U0", "35Hz_r_b_1")
    new = {r["frame"]: counters(r) for r in parse_log.parse(os.path.join(d, "log.txt"))}
    old = {r["frame"]: counters(r) for r in parse_log.parse(os.path.join(baseline(cfg, "35Hz_r_b_1")[1], "log.txt"))
           if r["frame"] <= U0_FRAMES[1]}
    frames = list(range(U0_FRAMES[0], U0_FRAMES[1] + 1))
    diff = [f for f in frames if new.get(f) != old.get(f)]
    ok = not diff and all(f in new for f in frames)
    msg = (f"U0: frames {frames[0]}-{frames[-1]} of the new run against T3's log, every counter of every frame "
           f"(timings left out): {len(frames) - len(diff)} of {len(frames)} identical -> {'PASS' if ok else 'FAIL'}"
           + (f"; first differing frame {diff[0]}: new {new.get(diff[0])} old {old.get(diff[0])}" if diff else ""))
    return ok, msg


def log_curve(log):
    if not os.path.exists(log):
        return {}
    return {r["frame"]: r.get("la_new") for r in parse_log.parse(log) if r["phase"] == "conv"}


def log_metrics(log, last):
    recs = [r for r in parse_log.parse(log) if r["phase"] == "conv" and r["frame"] <= last]
    by = {r["frame"]: r for r in recs}
    win = [by[f] for f in range(WINDOW[0], WINDOW[1] + 1) if f in by]
    if len(win) < WINDOW[1] - WINDOW[0] + 1:
        return None
    mean = lambda k: statistics.mean(r[k] for r in win)  # noqa: E731
    net = statistics.mean(r["la_new"] - r["la_n"] for r in win)
    loss = statistics.mean(r["li_add"] + r["ex_add"] for r in win)
    rate = loss / mean("la_n")
    lasting = net + loss
    return {
        "la": {m: by[m]["la_new"] for m in MARKS if m in by},
        "curve": [(r["frame"], r["la_new"]) for r in recs],
        "ipr": mean("ipr_found"), "short": mean("sa_new"), "promoted": mean("la_add"), "fail_lf": mean("fail_lf"),
        "lost": mean("li_add"), "exited": mean("ex_add"), "net": net, "lasting": lasting, "loss_rate": rate,
        "level_off": lasting / rate if rate > 0 else math.inf,
        "t_conv": statistics.mean(r["t_total"] for r in recs if not r["saved"]),
    }


def t1a_t_conv(cfg):
    return smoke_test.run_summary(cfg, t1.name_of("10Hz_r_b_1", 20))["t_conv"]


def hours_per_run(t_conv, t1a):
    return (t_conv / t1a * t1.ALONE_S + t1.LOAD_S) * t1.N_FRAMES / 3600


def evaluate(cfg, folder, last, max_save=None, last_quality=True):
    """Every measure of one run folder (a test or a baseline)."""
    m = log_metrics(os.path.join(folder, "log.txt"), last)
    if m is None:
        return None
    out = os.path.join(folder, cfg["output_folder"])
    m["q200"] = tq.quality_at(out, Q_FRAME, max_save=max_save)
    m["q_last"] = tq.quality_at(out, last, use_active=True) if last_quality else None
    return m


def passes(m, b):
    """The plan's rule (section 4): (passed, reasons it failed)."""
    why = []
    if not m["la"][FRAMES[1]] >= 1.05 * b["la"][FRAMES[1]]:
        why.append(f"long tracks at {FRAMES[1]} {m['la'][FRAMES[1]]} < 1.05 x {b['la'][FRAMES[1]]}")
    if not m["level_off"] > b["level_off"]:
        why.append(f"level-off {m['level_off']:.0f} not above {b['level_off']:.0f}")
    qa, qb = m["q200"], b["q200"]
    if abs(qa["step_med"] / qb["step_med"] - 1) > 0.03:
        why.append(f"step median {qa['step_med']:.4f} vs {qb['step_med']:.4f} (> 3 %)")
    if qa["lf_med"] > 1.10 * qb["lf_med"]:
        why.append(f"line-fit residual median {qa['lf_med']:.4f} > 1.1 x {qb['lf_med']:.4f}")
    if qa["nb_med"] > 1.10 * qb["nb_med"]:
        why.append(f"neighbour check median {qa['nb_med']:.3f} > 1.1 x {qb['nb_med']:.3f}")
    return not why, why


def decide(cfg):
    """After U1-U3 (and U4): which tests pass, whether U4 runs, and the change(s) U5 and U6 take."""
    _label, bdir, _ms = baseline(cfg, "35Hz_r_b_1")
    b = evaluate(cfg, bdir, FRAMES[1], last_quality=False)
    changes = dict(FIRST)
    res = {}
    for test in ("U1", "U2", "U3", "U4"):
        d = test_dir(cfg, test, "35Hz_r_b_1")
        if not os.path.exists(os.path.join(d, "run_info.json")):
            continue
        with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
            changes[test] = json.load(f)["_tune"]["changes"]
        m = evaluate(cfg, d, FRAMES[1], last_quality=False)
        ok, why = passes(m, b) if m else (False, ["not finished"])
        res[test] = {"passed": ok, "why": why, "long_end": m["la"][FRAMES[1]] if m else None,
                     "changes": changes[test]}
    spacing = [t for t in ("U1", "U2") if res.get(t, {}).get("passed")]
    best_spacing = max(spacing, key=lambda t: res[t]["long_end"]) if spacing else None
    u4 = ({**changes[best_spacing], **changes["U3"]}
          if best_spacing and res.get("U3", {}).get("passed") else None)
    cands = [t for t in res if res[t]["passed"]]
    chosen = max(cands, key=lambda t: res[t]["long_end"]) if cands else None
    dec = {"baseline_long_end": b["la"][FRAMES[1]], "tests": res, "u4": u4, "chosen": chosen,
           "chosen_changes": changes.get(chosen) if chosen else None}
    case.save_json(DECISION, dec)
    return dec


# ----------------------------------------------------------------------------------------------- the queue

def cmd_run(cfg, threads):
    say(f"queue starts (pid {os.getpid()}, {threads} threads)")
    run_test(cfg, "U0", "35Hz_r_b_1", U0_FRAMES, {}, threads)
    ok, msg = check_u0(cfg)
    say(msg)
    if not ok:
        say("queue stops: U0 does not repeat T3, so T3 cannot be the baseline")
        return
    for test, changes in FIRST:
        run_test(cfg, test, "35Hz_r_b_1", FRAMES, changes, threads)
    dec = decide(cfg)
    say(f"after U1-U3: {json.dumps({t: (r['passed'], r['long_end'], r['why']) for t, r in dec['tests'].items()})}; "
        f"baseline {dec['baseline_long_end']}; U4: {dec['u4'] or 'not run'}")
    if dec["u4"]:
        run_test(cfg, "U4", "35Hz_r_b_1", FRAMES, dec["u4"], threads)
        dec = decide(cfg)
        r = dec["tests"].get("U4", {})
        say(f"U4: passed {r.get('passed')}, long tracks at the end {r.get('long_end')}, {r.get('why')}")
    if dec["chosen"] is None:
        say("queue stops: no change passed, so U5 and U6 are not run")
        return
    say(f"chosen: {dec['chosen']} {dec['chosen_changes']}")
    run_test(cfg, "U5", "35Hz_r_b_2", FRAMES, dec["chosen_changes"], threads)
    run_test(cfg, "U6", "10Hz_r_b_1", FRAMES, dec["chosen_changes"], threads)
    say("queue ends")


# ----------------------------------------------------------------------------------------------- the summary

def fmt(x, spec):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def cmd_summary(cfg):
    t1a = t1a_t_conv(cfg)
    rows = []
    for run in ("35Hz_r_b_1", "35Hz_r_b_2", "10Hz_r_b_1"):
        label, bdir, ms = baseline(cfg, run)
        b = evaluate(cfg, bdir, FRAMES[1], max_save=ms, last_quality=False)
        if b is None:
            continue
        b.update(name=label, run=run, test="base", changes={},
                 h_run=None if run == "10Hz_r_b_1" else hours_per_run(b["t_conv"], t1a))
        rows.append(b)
        for test in ("U1", "U2", "U3", "U4", "U5", "U6"):
            d = test_dir(cfg, test, run)
            if not os.path.exists(os.path.join(d, "run_info.json")):
                continue
            m = evaluate(cfg, d, FRAMES[1])
            if m is None:
                continue
            with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
                ch = json.load(f)["_tune"]["changes"]
            ok, why = passes(m, b)
            m.update(name=name_of(test, run), run=run, test=test, changes=ch, passed=ok, why=why,
                     h_run=hours_per_run(m["t_conv"], t1a), base=label)
            rows.append(m)
    ch_txt = lambda c: ", ".join(f"{k.split('.')[1]} {v:g}" for k, v in c.items()) or "-"  # noqa: E731
    lines = ["U0-U6: tuning the tracking for the dense runs (plan 2026-10-11-plan-tuning.md); 12 threads beside the "
             "10 Hz batch; baselines T3 (frames 0-255 unchanged by the renumbering) and the batch's 10Hz_r_b_1", ""]
    if os.path.exists(QUEUE_LOG):
        with open(QUEUE_LOG, encoding="utf-8") as f:
            u0 = [ln.strip() for ln in f if ln[20:].startswith("U0: frames")]
        lines += [u0[-1][20:] if u0 else "U0: not checked yet", ""]
    lines.append(f"{'test':22s} {'change':24s} " + " ".join(f"{'long@' + str(m):>9s}" for m in MARKS) +
                 f" {'vs base':>8s} {'IPR':>7s} {'short':>7s} {'promoted':>9s} {'fail lf':>8s} {'lost':>6s} "
                 f"{'exited':>6s} {'lasting':>8s} {'loss %':>7s} {'level-off':>10s} {'s/frame':>8s} {'h/run':>6s} "
                 f"{'rule':>5s}")
    base_la = {}
    for r in rows:
        if r["test"] == "base":
            base_la[r["run"]] = r["la"][FRAMES[1]]
        rel = r["la"][FRAMES[1]] / base_la[r["run"]] - 1 if r["test"] != "base" else None
        lines.append(
            f"{r['name']:22s} {ch_txt(r['changes']):24s} " + " ".join(f"{r['la'].get(m, 0):9,d}" for m in MARKS) +
            f" {fmt(None if rel is None else 100 * rel, '+7.1f') + ('%' if rel is not None else ''):>8s}"
            f" {r['ipr']:7.0f} {r['short']:7.0f} {r['promoted']:9.0f} {r['fail_lf']:8.0f} {r['lost']:6.1f} "
            f"{r['exited']:6.1f} {r['lasting']:8.1f} {100 * r['loss_rate']:7.2f} {r['level_off']:10,.0f} "
            f"{r['t_conv']:8.2f} {fmt(r['h_run'], '6.0f')} "
            f"{('-' if r['test'] == 'base' else ('PASS' if r['passed'] else 'fail')):>5s}")
    lines += ["", "long@N: active long tracks at the end of frame N; vs base: at frame 250 against the run's baseline; "
                  "IPR ... exited: means per frame over frames 151-250 (exited/lost: established tracks of 7+ points "
                  "that left the volume / were dropped); lasting: promotions that last = net gain + lost + exited; "
                  "loss %: (lost + exited) / long tracks; level-off: lasting / loss rate; s/frame: mean convergence "
                  f"frame without saves on 12 threads beside the batch (the batch's own: 26 threads); h/run: as T3 "
                  f"(ratio to T1a's {t1a:.2f} s x {t1.ALONE_S} s + {t1.LOAD_S} s loading) x {t1.N_FRAMES} frames; "
                  "rule: the plan's decision rule against the baseline", ""]
    fails = [r for r in rows if r["test"] != "base" and not r["passed"]]
    for r in fails:
        lines.append(f"  {r['name']}: fails the rule: {'; '.join(r['why'])}")
    if os.path.exists(DECISION):
        with open(DECISION, encoding="utf-8") as f:
            dec = json.load(f)
        lines += ["", f"Decision after U1-U3/U4: U4 {dec['u4'] or 'not run'}; chosen for U5 and U6: "
                      f"{dec['chosen'] or 'none'} {dec['chosen_changes'] or ''}"]
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(HERE, "u_summary.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    case.save_json(os.path.join(HERE, "u_summary.json"), rows)

    # quality
    ql = ["U0-U6: quality of the long tracks (track_quality.py): tracks alive at the end of frame 200, rebuilt from "
          "the saved files, and at the last frame (250, as saved)", "",
          f"{'test':22s} {'frame':>5s} {'tracks':>7s} {'step med':>9s} {'p90':>6s} {'LF med':>7s} {'p90':>7s} "
          f"{'nb cover':>8s} {'nb med':>7s} {'p90':>6s} {'> 0.5':>6s}"]
    for r in rows:
        for q in (r["q200"], r.get("q_last")):
            if not q:
                continue
            ql.append(f"{r['name']:22s} {q['frame']:5d} {q['tracks']:7d} {q['step_med']:9.4f} {q['step_p90']:6.3f} "
                      f"{q['lf_med']:7.4f} {q['lf_p90']:7.4f} {q['nb_coverage']:8.3f} {q['nb_med']:7.3f} "
                      f"{q['nb_p90']:6.3f} {100 * q['nb_over_half']:5.1f}%")
    ql += ["", "step: |p(k+1) - p(k)| in mm per frame; LF: STB's linear-fit residual of the last four points (limit "
               f"0.05 mm); nb: velocity against the median of the other tracks within {tq.RADIUS_MM:g} mm (tracks with "
               f"{tq.MIN_NB}+ such neighbours; cover = their share), relative difference"]
    qtext = "\n".join(ql) + "\n"
    print(qtext)
    with open(os.path.join(HERE, "u_quality.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(qtext)
    case.save_json(os.path.join(HERE, "u_quality.json"),
                   [{"name": r["name"], "q200": r["q200"], "q_last": r.get("q_last")} for r in rows])
    draw(rows)


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [run for run in ("35Hz_r_b_1", "35Hz_r_b_2", "10Hz_r_b_1") if any(r["run"] == run for r in rows)]
    colours = {"U1": "#2171b5", "U2": "#08306b", "U3": "#e6550d", "U4": "#31a354", "U5": "#31a354", "U6": "#31a354"}
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2), squeeze=False)
    for ax, run in zip(axes[0], runs):
        for r in [r for r in rows if r["run"] == run]:
            fr = [f for f, _ in r["curve"]]
            la = [n for _, n in r["curve"]]
            if r["test"] == "base":
                ax.plot(fr, la, color="#555555", ls="--", lw=2, label=r["name"])
            else:
                lab = f"{r['test']}: " + ", ".join(f"{k.split('.')[1]} {v:g}" for k, v in r["changes"].items())
                ax.plot(fr, la, color=colours[r["test"]], lw=1.8, label=lab)
        ax.set_title(run, fontsize=10)
        ax.set_xlabel("frame")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0][0].set_ylabel("active long tracks at the end of the frame")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "u_tracks.png"), dpi=90)
    print(os.path.join(HERE, "u_tracks.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    cfg = case.load_config()
    if args.cmd == "run":
        cmd_run(cfg, args.threads)
    else:
        cmd_summary(cfg)


if __name__ == "__main__":
    main()
