"""E4c control (TEST BUILD ONLY, never for the working tree): today's STB::predictNext plus an offset as large as the
origin-free change |W - MD|, in a pseudo-random direction (from a hash of the track's last point and length), so the
tracking is perturbed by as much as E4's change but in no particular direction.

    python e4c_wjit_control.py SRC_ROOT        (a worktree at HEAD c3dd9e5, whose predictNext is today's)
"""
import os
import sys

root = sys.argv[1]
p = os.path.join(root, "src", "srcSTB", "STB.cpp")
raw = open(p, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


rep('''#include "STB.h"
#include <cmath>
''', '''#include "STB.h"
#include <cmath>
#include <cstdint>
#include <cstring>
''')

rep('''    if (shifted)
      prediction -= kShift;
    obj3d->_pt_center[axis] = prediction; // base-class field
  }

  return obj3d;
}''', '''    if (shifted)
      prediction -= kShift;
    obj3d->_pt_center[axis] = prediction; // base-class field
  }

  // TEST BUILD ONLY (tests/stb_improve E4c): move the prediction by |W - MD| in
  // a pseudo-random direction.
  {
    const size_t mm = (n < 6) ? n - 1 : 5;
    const Pt3D &l = list[n - 1]->_pt_center;
    const Pt3D &f = list[n - 1 - mm]->_pt_center;
    double d2 = 0.0;
    for (int ax = 0; ax < 3; ++ax) {
      const double md = l[ax] + (l[ax] - f[ax]) / static_cast<double>(mm);
      const double dd = obj3d->_pt_center[ax] - md;
      d2 += dd * dd;
    }
    uint64_t h = 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(n);
    for (int ax = 0; ax < 3; ++ax) {
      const double v = l[ax];
      uint64_t bits;
      std::memcpy(&bits, &v, sizeof bits);
      h ^= bits + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    }
    auto next = [&h]() { // splitmix64
      uint64_t z = (h += 0x9E3779B97F4A7C15ull);
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      return z ^ (z >> 31);
    };
    const double u1 = static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    const double u2 = static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    const double z = 2.0 * u1 - 1.0;
    const double phi = 6.283185307179586 * u2;
    const double sxy = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double d = std::sqrt(d2);
    obj3d->_pt_center[0] += d * sxy * std::cos(phi);
    obj3d->_pt_center[1] += d * sxy * std::sin(phi);
    obj3d->_pt_center[2] += d * z;
  }

  return obj3d;
}''')

out = s.replace("\n", "\r\n") if crlf else s
open(p, "wb").write(out.encode("utf-8"))
print("patched", p, "(CRLF)" if crlf else "(LF)")
