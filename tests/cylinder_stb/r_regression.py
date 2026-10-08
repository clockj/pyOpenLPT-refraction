"""Regression (plan: F6, E4, V4): existing cameras give bit-identical tracking with this work's builds.

    python r_regression.py BUILD [--base head1007] [--frames 0 50] [--sets plate pinhole] [--modes module exe]

The stb_improve campaign's benchmark case (tests/stb_improve/data: run 20260727/10Hz_r_b_1, four PINPLATE cameras,
frames copied locally, its tracerConfig; the view volume and voxel of its new_cams setup) is tracked over FRAMES:
  - "plate": with its PINPLATE camFiles;
  - "pinhole": with PINHOLE copies of them (the same K, R, t; the window dropped), so that the pinhole code paths run
    too (no pinhole case exists here; the tracking is then not physical, which does not matter for identity).
Each set runs with BUILD and BASE (HEAD without this work), with the Python module and/or OpenLPT.exe, in fresh
processes.  Judged: every file under Results/ byte-identical, and the same log counters (stbruns.compare).
Writes r_regression_<BUILD>_out.txt and .json; runs in runs/reg_* (git-ignored).
"""
import argparse
import os
import shutil

import cstb
import stbruns

SRC = os.path.join(cstb.REPO, "tests", "stb_improve", "data")
CONFIG = """# Frame Range: [startID,endID]
{f0},{f1}
# Frame Rate: [Hz]
5000
# Number of Threads: (0: use as many as possible)
26
# Number of Cameras:
4
# Camera File Path, Max Intensity
{cams}
# Image File Path
{lists}
# View Volume: (xmin,xmax,ymin,ymax,zmin,zmax)
-81.0,54.0,-42.0,59.0,-36.0,52.5
# Voxel to MM: e.g. use 1000^3 voxel, (xmax-xmin)/1000
0.129
# Output Folder Path:
Results
# Object Types:
Tracer
# STB Config Files:
{tracer}
# Flag to load previous track files, previous frameID
0
"""


def pinhole_copy(src, dst):
    with open(src, encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("# Inverse of Translation Vector"))
    lines = lines[:i + 2]
    j = next(k for k, ln in enumerate(lines) if ln.strip() == "PINPLATE")
    lines[j] = "PINHOLE"
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def prepare(set_name, frames):
    d = os.path.join(cstb.DATA, f"reg_{set_name}")
    os.makedirs(os.path.join(d, "camFile"), exist_ok=True)
    os.makedirs(os.path.join(d, "imgLists"), exist_ok=True)
    cams = []
    for c in range(4):
        src = os.path.join(SRC, "camFile", f"cam{c}.txt")
        dst = os.path.join(d, "camFile", f"cam{c}.txt")
        if set_name == "plate":
            shutil.copy2(src, dst)
        else:
            pinhole_copy(src, dst)
        cams.append(dst + ",255")
    lists = []
    for c in range(4):
        names = [os.path.join(SRC, "imgFile_mask", f"cam{c}", f"img{k:06d}.tif") for k in range(311)]
        p = os.path.join(d, "imgLists", f"cam{c}ImageNames.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(names) + "\n")
        lists.append(p)
    shutil.copy2(os.path.join(SRC, "tracerConfig.txt"), os.path.join(d, "tracerConfig.txt"))
    cfg = os.path.join(d, "config.txt")
    with open(cfg, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(CONFIG.format(f0=frames[0], f1=frames[1], cams="\n".join(cams), lists="\n".join(lists),
                               tracer=os.path.join(d, "tracerConfig.txt")))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--base", default="head1007")
    ap.add_argument("--frames", type=int, nargs=2, default=[0, 50])
    ap.add_argument("--sets", nargs="+", default=["plate", "pinhole"])
    ap.add_argument("--modes", nargs="+", default=["module", "exe"])
    args = ap.parse_args()
    log = cstb.Log(os.path.join(cstb.HERE, f"r_regression_{args.build}"))
    log(f"Regression: build {args.build} against {args.base}, frames {args.frames[0]}-{args.frames[1]}, "
        f"sets {args.sets}, modes {args.modes}")
    out = {"build": args.build, "base": args.base, "frames": args.frames, "results": {}}
    all_ok = True
    for s in args.sets:
        cfg = prepare(s, args.frames)
        for mode in args.modes:
            runs = {}
            for b in (args.base, args.build):
                rd = os.path.join(cstb.HERE, "runs", f"reg_{s}_{mode}_{b}")
                info = stbruns.run(cfg, rd, b, exe=(mode == "exe"))
                runs[b] = (rd, info)
                log(f"  {s} {mode} {b}: exit {info['returncode']}, wall {info['wall_s']:.0f} s")
            cmp_ = stbruns.compare(runs[args.base][0], runs[args.build][0])
            ok = cmp_["all_identical"] and all(r[1]["returncode"] == 0 for r in runs.values())
            all_ok &= ok
            out["results"][f"{s}_{mode}"] = {"compare": cmp_, "ok": ok,
                                             "wall_s": {b: r[1]["wall_s"] for b, r in runs.items()}}
            log(f"    {s} {mode}: {cmp_['identical_files']} of {cmp_['compared_files']} files identical"
                f"{'' if not (cmp_['only_a'] or cmp_['only_b']) else ', only in one: ' + str(cmp_['only_a'] + cmp_['only_b'])}"
                f": {'IDENTICAL' if ok else 'DIFFERENT'}")
    out["pass"] = bool(all_ok)
    log(f"  judged: {'PASS (bit-identical)' if all_ok else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
