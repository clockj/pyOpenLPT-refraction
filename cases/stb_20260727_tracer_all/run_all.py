"""Track every run of the case, one after another (I5), and deliver each to <run>\\Results_tracer on X:.

    python run_all.py [RUN ...] [--jobs K] [--threads T]
    python run_all.py --status

For each run (default: all of config.json, in order):
  1. skip it if it is delivered already (<stage>\\<run>\\DONE.json);
  2. an unfinished earlier attempt (a log.txt without DONE.json) is moved aside, not deleted, to
     <stage>\\<run>\\attempt_<time>\\;
  3. STB runs from the staging config (<stage>\\<run>\\config.json, written by make_configs.py --stage) through
     run_stb.py: log.txt, run_info.json and resources.csv beside the config, the output in <stage>\\<run>\\Results_tracer;
  4. the run must have ended normally: status ok, every frame done, the last frame's tracks saved;
  5. delivery, in a background thread while the next run starts: robocopy copies Results_tracer to X: (next to the
     camFile copies make_configs.py --case put there), plus the log, run info, resource samples and the staging config
     (as config_staging.json); every file must arrive with its size; then DONE.json.
Each step is logged with its time in <stage>\\batch_log.txt; <stage>\\batch_status.json holds the state of every run.
--status prints each run's state, frames done, recent time per frame and an estimate of the time left.
--jobs K runs K runs at a time; --threads T rewrites their staging configs to T threads (default: as written).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time

import case
import parse_log

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class Batch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg["staging_root"]
        self.log_path = os.path.join(self.root, "batch_log.txt")
        self.status_path = os.path.join(self.root, "batch_status.json")
        self.status = {}
        if os.path.exists(self.status_path):
            with open(self.status_path, encoding="utf-8") as f:
                self.status = json.load(f)

    def say(self, msg):
        line = f"{now()} {msg}"
        with LOCK:
            print(line, flush=True)
            with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")

    def set(self, run, **kw):
        with LOCK:
            self.status.setdefault(run, {}).update(kw)
            tmp = self.status_path + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(self.status, f, indent=2)
            os.replace(tmp, self.status_path)


def frames_done(log_path):
    """(last frame with a 'Total time' line, recent seconds per frame) of a running or finished log, read from its
    end only."""
    if not os.path.exists(log_path):
        return None, None
    size = os.path.getsize(log_path)
    with open(log_path, "rb") as f:
        f.seek(max(0, size - 400_000))
        tail = f.read().decode("utf-8", errors="replace").splitlines()
    times = []
    for ln in tail:
        m = parse_log.RE_TOTAL.match(ln)
        if m:
            times.append((int(m.group(1)), float(m.group(2))))
    if not times:
        return None, None
    recent = [t for _, t in times[-25:] if t < 300]
    return times[-1][0], (statistics.median(recent) if recent else None)


def last_save(log_path):
    """The last frame whose tracks were all saved to ConvergeTrack (the log's "Saved all tracks" line comes after the
    four files are closed), or None."""
    frame, last = None, None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = parse_log.RE_FRAME.match(ln)
            if m:
                frame = int(m.group(2))
            elif "Saved all tracks to folder" in ln and "ConvergeTrack" in ln and frame is not None:
                last = frame
    return last


def parts(sd):
    """The log parts of a resumed run, oldest first: (log path, the last frame it contributes or None for all)."""
    out = []
    k = 1
    while os.path.exists(os.path.join(sd, f"log_part{k}.txt")):
        with open(os.path.join(sd, f"resume_part{k}.json"), encoding="utf-8") as f:
            out.append((os.path.join(sd, f"log_part{k}.txt"), json.load(f)["resumed_after"]))
        k += 1
    return out


def check_finished(cfg, run, sd):
    """Whether the STB run in sd ended normally: status ok, every frame done (over all parts of a resumed run), the
    last frame's tracks saved."""
    with open(os.path.join(sd, "config.json"), encoding="utf-8") as f:
        f0, f1 = json.load(f)["frames"]
    with open(os.path.join(sd, "run_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    problems = []
    if info.get("status") != "ok":
        problems.append(f"status {info.get('status')}")
    # part k ran from the save part k-1 stopped at (exclusive) to its own last save (inclusive), where part k+1 resumed
    recs, prev = [], None
    for path, upto in parts(sd):
        recs += [r for r in parse_log.parse(path) if (prev is None or r["frame"] > prev) and r["frame"] <= upto]
        prev = upto
    recs += [r for r in parse_log.parse(os.path.join(sd, "log.txt")) if prev is None or r["frame"] > prev]
    frames = [r["frame"] for r in recs]
    if frames != list(range(f0, f1 + 1)):
        problems.append(f"frames {frames[:1]}..{frames[-1:]} ({len(frames)}), expected {f0}..{f1}")
    ct = os.path.join(sd, cfg["output_folder"], "ConvergeTrack")
    for p in ("LongTrackActive", "ShortTrackActive", "ExitTrack", "LongTrackInactive"):
        if not os.path.exists(os.path.join(ct, f"{p}_{f1}.csv")):
            problems.append(f"no {p}_{f1}.csv")
    return problems, {"frames": [f0, f1], "n_frames": len(frames), "wall_s": info.get("wall_s"),
                      "sum_frame_s": sum(r["t_total"] for r in recs),
                      "la_end": recs[-1].get("la_n") if recs else None}


def robocopy(src, dst, log):
    """Copy the folder src into dst (robocopy /E, restartable mode off, 8 retries); robocopy's exit codes below 8 mean
    success."""
    cmd = ["robocopy", src, dst, "/E", "/COPY:DAT", "/R:8", "/W:30", "/NP", "/NFL", "/NDL", f"/LOG+:{log}"]
    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc


def deliver(batch, run, sd):
    """Copy a finished run to X: and check that every file arrived with its size."""
    cfg = batch.cfg
    out = case.output_dir(cfg, run)
    batch.set(run, state="copying", copy_start=now())
    batch.say(f"{run}: copying {sd}\\{cfg['output_folder']} to {out}")
    t0 = time.time()
    log = os.path.join(sd, "robocopy.log")
    rc = robocopy(os.path.join(sd, cfg["output_folder"]), out, log)
    extra = {"log.txt": "log.txt", "run_info.json": "run_info.json", "resources.csv": "resources.csv",
             "config.json": "config_staging.json"}
    for name in sorted(os.listdir(sd)):  # the earlier parts of a resumed run
        if name.startswith(("log_part", "run_info_part", "resources_part", "resume_part", "config_resume_")):
            extra[name] = name
    for src, dst in extra.items():
        shutil.copyfile(os.path.join(sd, src), os.path.join(out, dst))
    missing, n, nbytes = [], 0, 0
    for dp, _dn, fn in os.walk(os.path.join(sd, cfg["output_folder"])):
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, os.path.join(sd, cfg["output_folder"]))
            q = os.path.join(out, rel)
            n += 1
            nbytes += os.path.getsize(p)
            if not os.path.exists(q) or os.path.getsize(q) != os.path.getsize(p):
                missing.append(rel)
    for src, dst in extra.items():
        if os.path.getsize(os.path.join(out, dst)) != os.path.getsize(os.path.join(sd, src)):
            missing.append(dst)
    minutes = (time.time() - t0) / 60
    if rc >= 8 or missing:
        batch.set(run, state="copy failed", robocopy_rc=rc, missing=missing[:20])
        batch.say(f"{run}: COPY FAILED (robocopy exit {rc}, {len(missing)} files missing or short: {missing[:5]})")
        return False
    done = {"run": run, "delivered": now(), "to": out, "files": n, "GB": nbytes / 1e9, "copy_min": minutes,
            "robocopy_rc": rc}
    case.save_json(os.path.join(sd, "DONE.json"), done)
    batch.set(run, state="done", copy_end=now(), files=n, GB=round(nbytes / 1e9, 2), copy_min=round(minutes, 1))
    batch.say(f"{run}: delivered, {n} files, {nbytes / 1e9:.1f} GB in {minutes:.0f} min")
    return True


def set_threads(sd, threads):
    p = os.path.join(sd, "config.json")
    with open(p, encoding="utf-8") as f:
        c = json.load(f)
    if c["threads"] != threads:
        c["threads"] = threads
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(c, f, indent=2)
            f.write("\n")


def prepare_resume(batch, run, sd):
    """An unfinished earlier attempt: resume it from its last complete save (its log becomes log_part<k>.txt and
    STB starts from config_resume_<frame>.json, which loads the active tracks of that frame); without a usable save,
    move it aside to attempt_<time>\\ and start again.  Returns the config to run."""
    cfg = batch.cfg
    out = cfg["output_folder"]
    f_save = last_save(os.path.join(sd, "log.txt"))
    ct = os.path.join(sd, out, "ConvergeTrack")
    if f_save is not None and all(os.path.exists(os.path.join(ct, f"{p}_{f_save}.csv"))
                                  for p in ("LongTrackActive", "ShortTrackActive")):
        k = len(parts(sd)) + 1
        for name, new in (("log.txt", f"log_part{k}.txt"), ("run_info.json", f"run_info_part{k}.json"),
                          ("resources.csv", f"resources_part{k}.csv")):
            if os.path.exists(os.path.join(sd, name)):
                os.replace(os.path.join(sd, name), os.path.join(sd, new))
        case.save_json(os.path.join(sd, f"resume_part{k}.json"), {"resumed_after": f_save, "at": now()})
        with open(os.path.join(sd, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
        c["load_tracks"] = {"enabled": True, "frame": f_save, "folder": f"{out}/ConvergeTrack/"}
        path = os.path.join(sd, f"config_resume_{f_save}.json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(c, f, indent=2)
            f.write("\n")
        batch.say(f"{run}: resuming an unfinished attempt after its last save, frame {f_save} (part {k + 1})")
        return path
    aside = os.path.join(sd, "attempt_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(aside)
    for name in os.listdir(sd):
        if name in ("log.txt", "run_info.json", "resources.csv", out, "robocopy.log") or \
                name.startswith(("log_part", "run_info_part", "resources_part", "resume_part", "config_resume_")):
            shutil.move(os.path.join(sd, name), os.path.join(aside, name))
    batch.say(f"{run}: an unfinished earlier attempt without a usable save moved to {aside}; starting again")
    return os.path.join(sd, "config.json")


def track(batch, run, threads, retries=2):
    """Run STB on one run (resuming an unfinished attempt; after a failure, up to `retries` more tries from the last
    save, 10 min apart); True if it ended normally."""
    cfg = batch.cfg
    sd = case.stage_dir(cfg, run)
    if threads:
        set_threads(sd, threads)
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(600)
        config = prepare_resume(batch, run, sd) if os.path.exists(os.path.join(sd, "log.txt")) else \
            os.path.join(sd, "config.json")
        batch.set(run, state="tracking", start=now())
        batch.say(f"{run}: tracking starts ({config})")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "run_stb.py"), config, "--quiet",
                              "--log", os.path.join(sd, "log.txt")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        problems, info = check_finished(cfg, run, sd) if os.path.exists(os.path.join(sd, "run_info.json")) else \
            ([f"no run_info.json (exit {rc})"], {})
        if rc == 0 and not problems:
            break
        batch.set(run, state="tracking failed", end=now(), exit=rc, problems=problems)
        batch.say(f"{run}: TRACKING FAILED (exit {rc}): {problems}" +
                  (" - retrying in 10 min from the last save" if attempt < retries else ""))
    else:
        return False
    batch.set(run, state="tracked", end=now(), **{k: info[k] for k in ("n_frames", "wall_s", "la_end")})
    batch.say(f"{run}: tracked {info['n_frames']} frames in {info['wall_s'] / 3600:.1f} h "
              f"({info['sum_frame_s'] / max(1, info['n_frames']):.2f} s per frame)")
    return True


def worker(batch, queue, threads, copies):
    while True:
        with LOCK:
            if not queue:
                return
            run = queue.pop(0)
        if track(batch, run, threads):
            t = threading.Thread(target=deliver, args=(batch, run, case.stage_dir(batch.cfg, run)))
            t.start()
            copies.append(t)


def cmd_status(cfg):
    batch = Batch(cfg)
    for run in cfg["cases"]:
        sd = case.stage_dir(cfg, run)
        st = batch.status.get(run, {})
        state = "done" if os.path.exists(os.path.join(sd, "DONE.json")) else st.get("state", "not started")
        last, spf = frames_done(os.path.join(sd, "log.txt"))
        line = f"{run:11s} {state:16s}"
        if last is not None:
            with open(os.path.join(sd, "config.json"), encoding="utf-8") as f:
                f1 = json.load(f)["frames"][1]
            line += f" frame {last:5d}/{f1}  {spf:6.2f} s/frame of tracking" if spf else f" frame {last}"
            if state == "tracking" and st.get("start"):
                # wall time per frame since this attempt started: tracking, image loading and saves
                elapsed = (datetime.datetime.now() - datetime.datetime.fromisoformat(st["start"])).total_seconds()
                resumes = sorted(n for n in os.listdir(sd) if n.startswith("resume_part"))
                after = -1
                if resumes:
                    with open(os.path.join(sd, resumes[-1]), encoding="utf-8") as f:
                        after = json.load(f)["resumed_after"]
                done = last - after  # frames this attempt has finished
                if done > 0:
                    wall = elapsed / done
                    line += f", {wall:.2f} s/frame in all, ~{(f1 - last) * wall / 3600:.1f} h left"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    cfg = case.load_config()
    if args.status:
        cmd_status(cfg)
        return
    batch = Batch(cfg)
    runs = args.runs or cfg["cases"]
    queue = [r for r in runs if not os.path.exists(os.path.join(case.stage_dir(cfg, r), "DONE.json"))]
    batch.say(f"batch starts: {len(queue)} runs to track {queue} (jobs {args.jobs}, threads "
              f"{args.threads or 'as configured'}; pid {os.getpid()})")
    copies = []
    workers = [threading.Thread(target=worker, args=(batch, queue, args.threads, copies)) for _ in range(args.jobs)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    for t in copies:
        t.join()
    states = {r: batch.status.get(r, {}).get("state") for r in runs}
    batch.say(f"batch ends: {states}")


if __name__ == "__main__":
    main()
