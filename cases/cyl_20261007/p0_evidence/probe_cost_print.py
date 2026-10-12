import runpy, sys, io, json, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path("probe_cost.py")
res = g["res"]
it = res.pop("warm_newton_iters_hist")
import numpy as np
iters = np.repeat(np.arange(len(it)), it)
res["warm_newton_iters"] = {"median": float(np.median(iters)), "p99": float(np.percentile(iters, 99)), "max": int(iters.max()), "n": int(len(iters))}
warm = g["warm"]; ok = g["ok"]
bad = [k for k, w in enumerate(warm) if not w[0]]
res["warm_newton_failed"] = len(bad)
print(json.dumps(res))
red2 = g["red2"]; sol = g["sol"]; rc = g["rc"]; n = g["n"]; nR = g["nR"]; r_p = g["r_p"]; u = g["u"]
for k in bad[:5]:
    i = ok[k]
    print("fail", k, "red2", red2[i], "seed", sol[i][1:3], "->", warm[k])
    print("   classify_near on moved point:", rc.classify_near(n, nR, r_p/u, *red2[i]))
