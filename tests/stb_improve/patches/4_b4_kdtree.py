"""B4: STB::checkRepeat finds its candidates with a k-d tree (identical flags)."""
p = r"D:\Refraction\pyOpenLPT\src\srcSTB\STB.cpp"
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

namespace fs = std::filesystem;
''', '''#include "STB.h"
#include <cmath>

namespace fs = std::filesystem;

namespace {
// nanoflann adaptor over a vector of points (STB::checkRepeat)
struct PointCloud3D {
  const std::vector<Pt3D> &pts;
  size_t kdtree_get_point_count() const { return pts.size(); }
  double kdtree_get_pt(const size_t idx, int dim) const { return pts[idx][dim]; }
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};
using KDTreePoints = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud3D>, PointCloud3D, 3>;
} // namespace
''')

rep('''  const double tol = _obj_config->_sm_param.tol_3d_mm;

  const int ni_obj3d = static_cast<int>(n_obj3d);

#pragma omp parallel for if (!omp_in_parallel())
  for (int i = 0; i < ni_obj3d; ++i) {
    const Pt3D &p = objs[i]->_pt_center;

    // for bubbles, tol needs to be added with radius
    double r_obj = 0.0;
    if (const auto *bubble = dynamic_cast<const Bubble3D *>(objs[i].get())) {
      r_obj = bubble->_r3d;
    }

    for (const Pt3D &q : last_centers) {
      Pt3D d = p - q;
      double dist = d.norm();
      if (dist <= tol + r_obj) {
        flags[i] = ObjFlag::Repeated;
        break;
      }
    }
  }

  return flags;''', '''  const double tol = _obj_config->_sm_param.tol_3d_mm;

  const int ni_obj3d = static_cast<int>(n_obj3d);

  // A k-d tree over the last points finds the candidates near each object; the
  // same distance test as comparing every pair then decides, so the flags are
  // the same.
  PointCloud3D cloud{last_centers};
  KDTreePoints tree(3, cloud, {10 /* max leaf */});
  tree.buildIndex();

#pragma omp parallel for if (!omp_in_parallel())
  for (int i = 0; i < ni_obj3d; ++i) {
    const Pt3D &p = objs[i]->_pt_center;

    // for bubbles, tol needs to be added with radius
    double r_obj = 0.0;
    if (const auto *bubble = dynamic_cast<const Bubble3D *>(objs[i].get())) {
      r_obj = bubble->_r3d;
    }
    const double r = tol + r_obj;

    // query a slightly larger radius (nanoflann compares squared distances),
    // so rounding cannot drop a point the exact test accepts
    const double r_query = r * (1.0 + 1e-9) + 1e-12;
    std::vector<nanoflann::ResultItem<size_t, double>> hits;
    nanoflann::RadiusResultSet<double, size_t> result_set(r_query * r_query,
                                                          hits);
    tree.findNeighbors(result_set, p.data(), nanoflann::SearchParameters());

    for (const auto &h : hits) {
      Pt3D d = p - last_centers[h.first];
      double dist = d.norm();
      if (dist <= r) {
        flags[i] = ObjFlag::Repeated;
        break;
      }
    }
  }

  return flags;''')

out = s.replace("\n", "\r\n") if crlf else s
open(p, "wb").write(out.encode("utf-8"))
print("patched STB.cpp, crlf =", crlf)
