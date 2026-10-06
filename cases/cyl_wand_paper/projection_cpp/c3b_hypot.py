"""C3b (diagnosis of C3): the cylinder's differences come from hypot.

C3 found the sphere's runs identical to the last bit and the cylinder's not. The cylinder alone calls hypot (its
normals, and the xtol tests), and Python's math.hypot and math.dist do not call the C runtime: since Python 3.10 they
compute a correctly rounded norm (vector_norm in Modules/mathmodule.c: power-of-two scaling, Veltkamp-Dekker
splitting, compensated summation, a differential correction of the square root). This test re-implements that
algorithm in plain floats, as the C++ port will, and checks it against math.hypot and math.dist on random inputs,
and against the C runtime's hypot (ctypes), which the port called so far.
"""
import ctypes
import math

import numpy as np

from common import Log

T27 = 134217729.0  # 2**27 + 1, Veltkamp's splitting constant


def vector_norm(vec):
    """Python 3.11's vector_norm for finite, nonzero inputs (math.hypot, math.dist)."""
    vec = [abs(x) for x in vec]
    mx = max(vec)
    if mx == 0.0 or len(vec) <= 1:
        return mx
    _, max_e = math.frexp(mx)
    scale = math.ldexp(1.0, -max_e)
    csum, frac1, frac2, frac3 = 1.0, 0.0, 0.0, 0.0
    for x in vec:
        x *= scale
        t = x * T27
        hi = t - (t - x)
        lo = x - hi
        x = hi * hi
        oldcsum = csum
        csum += x
        frac1 += (oldcsum - csum) + x
        x = 2.0 * hi * lo
        oldcsum = csum
        csum += x
        frac2 += (oldcsum - csum) + x
        frac3 += lo * lo
    h = math.sqrt(csum - 1.0 + (frac1 + frac2 + frac3))
    x = h
    t = x * T27
    hi = t - (t - x)
    lo = x - hi
    x = -hi * hi
    oldcsum = csum
    csum += x
    frac1 += (oldcsum - csum) + x
    x = -2.0 * hi * lo
    oldcsum = csum
    csum += x
    frac2 += (oldcsum - csum) + x
    x = -lo * lo
    oldcsum = csum
    csum += x
    frac3 += (oldcsum - csum) + x
    x = csum - 1.0 + (frac1 + frac2 + frac3)
    return (h + x / (2.0 * h)) / scale


log = Log("c3b_hypot")
crt = ctypes.CDLL("ucrtbase")
crt._hypot.restype = ctypes.c_double
crt._hypot.argtypes = [ctypes.c_double, ctypes.c_double]
rng = np.random.default_rng(20261008)
N = 200000
same2 = same3 = crt_same = 0
for _ in range(N):
    e = rng.uniform(-30, 5, 3)
    v = rng.uniform(-1, 1, 3) * 10.0 ** e
    a, b = float(v[0]), float(v[1])
    same2 += vector_norm([a, b]) == math.hypot(a, b)
    crt_same += crt._hypot(a, b) == math.hypot(a, b)
    p, q = tuple(float(x) for x in v), tuple(float(x) for x in rng.uniform(-1, 1, 3) * 10.0 ** e)
    same3 += vector_norm([pi - qi for pi, qi in zip(p, q)]) == math.dist(p, q)
log(f"{N} random pairs: the re-implementation equals math.hypot at {same2}; the C runtime's _hypot at {crt_same}")
log(f"{N} random pairs of 3-D points: the re-implementation equals math.dist at {same3}")
ok = same2 == N and same3 == N
log(f"C3b {'PASS' if ok else 'FAIL'}: {'the re-implementation is bitwise math.hypot and math.dist' if ok else 'mismatch'}"
    f"; the C runtime's hypot differs from Python's at {N - crt_same} of {N} pairs")
log.close(passed=ok, n=N, same_hypot=same2, same_dist=same3, crt_same=crt_same)
