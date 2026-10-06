"""Every number the paper's Section 9.2.1 and Supplement S8 quote from the C++ comparison (run_comparison.py, C5) and
the near-grazing test (c6_near_grazing.py, C6), each printed with its key, and S8's table as LaTeX.

    python cpp_numbers.py [results/comparison_data.npz]  ->  cpp_numbers_out.txt, cpp_numbers.json

Reads only the saved results; runs nothing.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ALL, SOME, NONE_VIS, NONE_INV, FALSE, UNMATCHED = 0, 1, 2, 3, 4, 5
METHODS = ("ours", "afrt", "gn", "lm", "shoot")
BASELINES = METHODS[1:]
TOL = 1e-9
TEX = {"ours": "this work", "afrt": r"AFRT~\cite{belden2013calibration}",
       "gn": r"dewarping, GN~\cite{paolillo2022perspective}", "lm": r"dewarping, LM~\cite{paolillo2022perspective}",
       "shoot": "shooting, LM"}
GROUPS = {"ref": (("uniform",), ("edge", "band", "steep", "axis")), "pa": (("uniform", "steep", "axis"),),
          "sph": (("uniform",), ("edge", "band", "axis"))}

path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "comparison_data.npz"
z = np.load(path)
out = open(HERE / "cpp_numbers_out.txt", "w", encoding="utf-8")
num = {}


def say(msg=""):
    print(msg)
    out.write(msg + "\n")


def put(key, value, note=""):
    num[key] = value
    say(f"{key} = {value}" + (f"   ({note})" if note else ""))


def have(key, kind):
    return f"{key}/{kind}/n_true" in z.files


def cat(key, kinds, field):
    return np.concatenate([z[f"{key}/{k}/{field}"] for k in kinds if have(key, k)])


def stats(key, kinds, name):
    nt = cat(key, kinds, "n_true")
    vis = nt >= 1
    c = cat(key, kinds, f"{name}/code")
    t = cat(key, kinds, f"{name}/t") * 1e6                       # us
    found = cat(key, kinds, f"{name}/found")
    ev = cat(key, kinds, f"{name}/ev").astype(float)
    return dict(N=int(nt.size), vis=int(vis.sum()), inv=int((~vis).sum()),
                i1=int(np.sum(vis & np.isin(c, (ALL, SOME)))), i2=int(np.sum(c == ALL)), i3=int(np.sum(c == FALSE)),
                i3i=int(np.sum(~vis & (c == FALSE))), i4v=int(np.sum(c == NONE_VIS)), i4i=int(np.sum(c == NONE_INV)),
                um=int(np.sum(c == UNMATCHED)), t_med=float(np.median(t)), t_p95=float(np.percentile(t, 95)),
                t_img=float(t.sum() / found.sum()) if found.sum() else float("nan"), ev_med=float(np.median(ev)))


def pc(k, m):
    return round(100.0 * k / m, 1) if m else None


say(f"source: {path}")
# ------------------------------------------------------------------------------- outcomes, single start
for key, groups in GROUPS.items():
    for kinds in groups:
        if not any(have(key, k) for k in kinds):
            continue
        g = f"{key}.{'+'.join(kinds)}"
        for name in METHODS:
            s = stats(key, kinds, name)
            put(f"{g}.{name}.every_image_pct_of_visible", pc(s["i2"], s["vis"]), f"of {s['vis']} visible points")
            put(f"{g}.{name}.false_image_pct_of_all", pc(s["i3"], s["N"]), f"of {s['N']} points")
            put(f"{g}.{name}.no_answer_pct_of_visible", pc(s["i4v"], s["vis"]))
            put(f"{g}.{name}.false_at_unseen", [s["i3i"], s["inv"]])
            put(f"{g}.{name}.t_median_us", round(s["t_med"], 2))
for key in ("ref", "sph"):
    if have(key, "band"):
        put(f"{key}.band.points", int(z[f"{key}/band/n_true"].size))
        put(f"{key}.band.both_images_found_single_start",
            {n: int(np.sum(z[f"{key}/band/{n}/code"] == ALL)) for n in BASELINES})

# ------------------------------------------------------------------------------- accuracy: where the rays land
ref_sets = [k for k in ("uniform", "edge", "band", "steep", "axis") if have("ref", k)]
m = cat("ref", ref_sets, "ours/miss")
put("ref.ours.rays", int(m.size))
put("ref.ours.miss_median_m", float(f"{np.median(m):.2g}"))
put("ref.ours.miss_max_m", float(f"{m.max():.2g}"))
put("ref.ours.miss_p99_m", float(f"{np.percentile(m, 99):.2g}"))
put("ref.ours.rays_missing_by_more_than_1e-11_m", int(np.sum(m > 1e-11)))
# how close to grazing the rays are that miss by more than 1e-11 m (recomputed with the module: same rays, C2)
import math                                                     # noqa: E402
from common import SCENES, args, cpp, e1                        # noqa: E402
sc = SCENES["ref"]
worst = []
for k in ref_sets:
    mk, ip, Xk = z[f"ref/{k}/ours/miss"], z[f"ref/{k}/ours/miss_point"], z[f"ref/{k}/X"]
    for j in np.flatnonzero(mk > 1e-11):
        x = Xk[ip[j]]
        rX = math.hypot(x[0], x[1])
        for b, L, C, d in cpp.ours(*args(sc), np.array([x]))[0]:
            if abs(e1.landing(sc, tuple(x), d)[0] - mk[j]) <= 1e-3 * mk[j]:
                worst.append((float(mk[j]), 1.0 - abs(L) / e1.pc.Lmax_kstar(sc.n, sc.R, rX, abs(C))[0], k))
worst.sort(reverse=True)
put("ref.ours.worst_rays_miss_delta_set", [[float(f"{a:.2g}"), float(f"{b:.2g}"), c] for a, b, c in worst],
    "miss (m), 1 - Lambda/Lmax, set")
for key in ("pa", "sph"):
    ks = [k for k in ("uniform", "edge", "band", "steep", "axis") if have(key, k)]
    if ks:
        mm = cat(key, ks, "ours/miss")
        put(f"{key}.ours.miss_max_m", float(f"{mm.max():.2g}"), f"{mm.size} rays")
for name in BASELINES:
    mm = cat("ref", ref_sets, f"{name}/miss")
    mm = mm[~np.isnan(mm)]
    good, bad = mm[mm <= TOL], mm[mm > TOL]
    put(f"ref.{name}.correct_miss_median_m", float(f"{np.median(good):.2g}"), f"{good.size} correct answers")
    put(f"ref.{name}.correct_miss_max_m", float(f"{good.max():.2g}"))
    fin = bad[np.isfinite(bad)]
    put(f"ref.{name}.false_answers", int(bad.size), f"{int(np.sum(~np.isfinite(bad)))} never reach the water")
    if fin.size:
        put(f"ref.{name}.false_miss_mm", [float(f"{1e3 * fin.min():.2g}"), float(f"{1e3 * fin.max():.2g}"),
                                          float(f"{1e3 * np.median(fin):.2g}")], "min, max, median")

# ------------------------------------------------------------------------------- our time by branch
for key in ("ref",):
    for k in ref_sets[:1]:
        t = z[f"{key}/{k}/ours/t"] * 1e6
        found, far = z[f"{key}/{k}/ours/found"], z[f"{key}/{k}/ours/far"]
        near = (found - far) > 0
        put(f"{key}.{k}.ours.t_median_us_with_near_image", round(float(np.median(t[near])), 1), f"{int(near.sum())} points")
        put(f"{key}.{k}.ours.t_median_us_without", round(float(np.median(t[~near])), 1), f"{int((~near).sum())} points")
        put(f"{key}.{k}.ours.t_mean_us", round(float(t.mean()), 1))

# ------------------------------------------------------------------------------- multi-start
if "ms_ks" in z.files:
    ks = z["ms_ks"]
    for key, groups in (("ref", GROUPS["ref"]), ("sph", GROUPS["sph"])):
        for kinds in groups:
            kinds = [k for k in kinds if f"{key}/{k}/afrt/ms_found" in z.files]
            if not kinds:
                continue
            g = f"{key}.{'+'.join(kinds)}"
            nt = cat(key, kinds, "n_true")
            vis = nt >= 1
            so = stats(key, kinds, "ours")
            target = so["i2"] / so["vis"]
            for name in BASELINES:
                found = cat(key, kinds, f"{name}/ms_found")
                kept = cat(key, kinds, f"{name}/ms_kept")
                t = cat(key, kinds, f"{name}/ms_t") * 1e3      # us
                every = np.mean(found[vis] == nt[vis, None], axis=0)
                silent = np.mean(kept[vis] == 0, axis=0)
                put(f"{g}.ms.{name}.every_image_pct", {int(k): round(100 * float(v), 1) for k, v in zip(ks, every)})
                put(f"{g}.ms.{name}.silent_pct_of_visible_at_Kmax", round(100 * float(silent[-1]), 1))
                put(f"{g}.ms.{name}.t_median_us", {int(k): round(float(np.median(t[:, j])), 1)
                                                    for j, k in enumerate(ks)})
                first = cat(key, kinds, f"{name}/ms_first_all")
                Kmax = int(ks[-1])
                reach = [K for K in range(1, Kmax + 1) if np.mean((first[vis] >= 1) & (first[vis] <= K)) >= target - 1e-12]
                if reach:
                    K = reach[0]
                    put(f"{g}.ms.{name}.first_K_reaching_ours", K, f"ours {100 * target:.1f} %; time at the nearest "
                        f"saved K >= K: see t_median_us")
                else:
                    put(f"{g}.ms.{name}.first_K_reaching_ours", None, f"not within K = {Kmax}")
            put(f"{g}.ours.every_image_pct", pc(so["i2"], so["vis"]))
            put(f"{g}.ours.t_median_us", round(so["t_med"], 1))

# ------------------------------------------------------------------------------- near grazing (C6)
c6 = HERE / "c6_near_grazing.json"
if c6.exists():
    d = json.loads(c6.read_text(encoding="utf-8"))
    put("c6.cpp", d["cpp"])
    put("c6.python", d["python"])
    lost = [r[2] for r in d["cpp_lost"]]
    put("c6.cpp_lost_max_delta", float(f"{max(lost):.2g}") if lost else None, "1 - Lambda/Lmax")

# ------------------------------------------------------------------------------- S8's table (LaTeX)
rows = []
for key, kinds, title in (("ref", ("uniform",), "reference window, uniform"),
                          ("ref", ("edge", "band", "steep", "axis"), "reference window, difficult sets pooled"),
                          ("pa", ("uniform", "steep", "axis"),
                           r"cylinder of Ref.~\cite{paolillo2022perspective}, Table~1, all sets"),
                          ("sph", ("uniform",), "sphere, uniform"),
                          ("sph", ("edge", "band", "axis"), "sphere, difficult sets pooled")):
    if not all(have(key, k) for k in kinds):
        continue
    nt = cat(key, kinds, "n_true")
    hist = [int(np.sum(np.minimum(nt, 2) == k)) for k in range(3)]
    rows.append("\\midrule\n\\multicolumn{9}{@{}l}{\\emph{%s}: %d points; %d / %d / %d with 0 / 1 / 2+ images} \\\\"
                % (title, nt.size, *hist))
    for name in METHODS:
        s = stats(key, kinds, name)
        f = lambda k, m_: "--" if not m_ else "%.1f" % (100.0 * k / m_)
        t = lambda v: "%.3g" % v if v < 1000 else "%.0f" % v
        rows.append("%s & %s & %s & %s & %s & %s & %s & %s (%s) & %s \\\\" % (
            TEX[name], f(s["i1"], s["vis"]), f(s["i2"], s["vis"]), f(s["i4v"], s["vis"]), f(s["i3"], s["N"]),
            f(s["i4i"], s["inv"]) + ("$^{a}$" if name == "ours" and s["inv"] else ""),
            "--" if name == "ours" else "%.0f" % s["ev_med"], t(s["t_med"]), t(s["t_p95"]), t(s["t_img"])))
say()
say("% Supplement 1, Section S8: the single-start table (C++)")
say(r"\begin{table}[htbp]")
say(r"\caption{Our projection against iterative ray-tracing projection, all in C++}")
say(r"\label{s:tab:baselines}")
say("\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}")
say(r"\begin{tabular}{@{}lrrrrrrrr@{}}")
say(r"\toprule")
say(r" & \multicolumn{3}{c}{points with an image, \%} & \multicolumn{2}{c}{all points, \%} & ray & "
    r"\multicolumn{2}{c}{time, $\mu$s} \\")
say(r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){8-9}")
say(r"method & (i) & (ii) & (iv) & (iii) & (iv)$'$ & paths & per point & per image \\")
for r in rows:
    say(r)
say("\\bottomrule\n\\end{tabular}")
say(r"\\[1mm]\footnotesize (i) at least one correct image; (ii) every image; (iii) a false image; (iv) no answer; "
    r"(iv)$'$ no answer at a point that has no image, as a percentage of those points. Ray paths: median ray-path "
    r"evaluations per point. Time: median (95th percentile) per point, and per image found; C++, one thread, "
    r"Intel Core i9-13900K, each call's fastest of five passes. $^{a}$The empty list; tracing confirms that these "
    r"points have no image.")
say(r"\end{table}")
out.close()
(HERE / "cpp_numbers.json").write_text(json.dumps(num, indent=1) + "\n", encoding="utf-8")
