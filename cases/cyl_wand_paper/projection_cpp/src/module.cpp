// projbench: the paper's projection comparison (Section 9.2.1) in C++.
//
// Our search is pyOpenLPT's C++ core (src/srcMath/RefractCurved.cpp, compiled read-only into this module, as
// projectAll() calls it: tolerance 1e-9, default iteration caps, with the coverage record); the four iterative methods
// are the ports of baselines.cpp. Every call is timed here, in C++, with std::chrono::steady_clock, so Python's
// overhead is not timed. Loops run with the GIL released; inputs and outputs are NumPy arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "RefractCurved.h"
#include "baselines.hpp"

namespace py = pybind11;
namespace rc = refract_curved;
using clk = std::chrono::steady_clock;

namespace {

constexpr int NM = 4;  // baselines: AFRT, GN, LM, shooting

struct RayOut {
    int branch;  // 0 near, 1 far, 2 ring (sphere only)
    double lambda, c_z;
    pb::V3 d;
};

pb::Scene scene(const std::vector<double>& n, const std::vector<double>& R, double rP, bool sphere) {
    pb::Scene s;
    s.n = n;
    s.R = R;
    s.rP = rP;
    s.sphere = sphere;
    return s;
}

std::vector<pb::V3> points(const py::array_t<double, py::array::c_style | py::array::forcecast>& a, const char* what) {
    if (a.ndim() != 2 || a.shape(1) != 3) throw std::invalid_argument(std::string(what) + " must be (m, 3)");
    std::vector<pb::V3> out(static_cast<std::size_t>(a.shape(0)));
    auto r = a.unchecked<2>();
    for (py::ssize_t i = 0; i < a.shape(0); ++i) out[i] = {r(i, 0), r(i, 1), r(i, 2)};
    return out;
}

// One call of our search, as pyOpenLPT's camera makes it (Camera.cpp, projectAll).
std::vector<RayOut> ours_one(const pb::Scene& sc, const rc::Stack& st, const pb::V3& X) {
    const double P0[3] = {sc.rP, 0.0, 0.0};
    const double x[3] = {X[0], X[1], X[2]};
    std::vector<rc::Ray> rays;
    if (sc.sphere) {
        const double c[3] = {0.0, 0.0, 0.0};
        rays = rc::projectSphere(st, c, P0, x, 1e-9, -1);
    } else {
        rc::CylinderCoverage cov;
        rays = rc::projectCylinder(st, P0, x, &cov, 1e-9, -1);
    }
    std::vector<RayOut> out;
    out.reserve(rays.size());
    for (const auto& r : rays)
        out.push_back({r.is_ring ? 2 : (r.branch == rc::Branch::Far ? 1 : 0), r.lambda, r.c_z,
                       {r.dir[0], r.dir[1], r.dir[2]}});
    return out;
}

template <class F>
inline std::int64_t timed(F&& f) {
    const auto t0 = clk::now();
    f();
    const auto t1 = clk::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

py::list rays_to_py(const std::vector<std::vector<RayOut>>& all) {
    static const char* names[3] = {"near", "far", "ring"};
    py::list out;
    for (const auto& rays : all) {
        py::list one;
        for (const auto& r : rays)
            one.append(py::make_tuple(names[r.branch], r.lambda, r.c_z, py::make_tuple(r.d[0], r.d[1], r.d[2])));
        out.append(one);
    }
    return out;
}

bool same_rays(const std::vector<RayOut>& a, const std::vector<RayOut>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].branch != b[i].branch || a[i].lambda != b[i].lambda || a[i].c_z != b[i].c_z || a[i].d != b[i].d)
            return false;
    return true;
}

bool same_result(const pb::Result& a, const pb::Result& b) {
    return a.status == b.status && a.ev == b.ev && a.it == b.it && a.evj == b.evj && a.has_d == b.has_d && a.d == b.d;
}

template <class T>
py::array_t<T> to_array(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
    py::array_t<T> a(shape);
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

}  // namespace

PYBIND11_MODULE(projbench, m) {
    m.doc() = "The paper's projection comparison in C++: our search (pyOpenLPT's RefractCurved core) and the C++ "
              "ports of the iterative projections of code/paper_baselines.py, timed in C++.";

    m.def("info", []() {
        py::dict d;
#ifdef _MSC_FULL_VER
        d["compiler"] = "MSVC " + std::to_string(_MSC_FULL_VER);
#else
        d["compiler"] = "unknown";
#endif
#ifdef _MSVC_LANG
        d["cxx_standard"] = static_cast<long>(_MSVC_LANG);
#else
        d["cxx_standard"] = static_cast<long>(__cplusplus);
#endif
#ifdef NDEBUG
        d["ndebug"] = true;
#else
        d["ndebug"] = false;
#endif
#ifdef PB_RELEASE_FLAGS
        d["release_flags"] = PB_RELEASE_FLAGS;
#endif
        d["clock"] = "std::chrono::steady_clock";
        d["clock_period_ns"] = 1e9 * static_cast<double>(clk::period::num) / static_cast<double>(clk::period::den);
        return d;
    });

    m.def(
        "ours",
        [](const std::vector<double>& n, const std::vector<double>& R, double rP, bool sphere,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& X) {
            const pb::Scene sc = scene(n, R, rP, sphere);
            const rc::Stack st{n, R};
            const auto pts = points(X, "X");
            std::vector<std::vector<RayOut>> all(pts.size());
            {
                py::gil_scoped_release nogil;
                for (std::size_t i = 0; i < pts.size(); ++i) all[i] = ours_one(sc, st, pts[i]);
            }
            return rays_to_py(all);
        },
        py::arg("n"), py::arg("R"), py::arg("rP"), py::arg("sphere"), py::arg("X"),
        "Our search on every point of X (m, 3): per point a list of (branch, Lambda, C_z, direction).");

    m.def(
        "baseline",
        [](const std::vector<double>& n, const std::vector<double>& R, double rP, bool sphere, int method,
           const std::vector<double>& X, std::optional<std::vector<double>> d0) {
            const pb::Scene sc = scene(n, R, rP, sphere);
            const pb::V3 x{X.at(0), X.at(1), X.at(2)};
            pb::V3 s{};
            if (d0) s = {d0->at(0), d0->at(1), d0->at(2)};
            const pb::Result r = pb::call(sc, method, x, d0 ? &s : nullptr);
            py::dict d;
            d["d"] = py::make_tuple(r.d[0], r.d[1], r.d[2]);
            d["has_d"] = r.has_d;
            d["status"] = r.status;
            d["ev"] = r.ev;
            d["it"] = r.it;
            d["evj"] = r.evj;
            d["e"] = r.e;
            return d;
        },
        py::arg("n"), py::arg("R"), py::arg("rP"), py::arg("sphere"), py::arg("method"), py::arg("X"),
        py::arg("d0") = py::none(), "One baseline (0 AFRT, 1 GN, 2 LM, 3 shooting) on X from d0 (None: its paper's).");

    m.def(
        "landing",
        [](const std::vector<double>& n, const std::vector<double>& R, double rP, bool sphere,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& X,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& D) {
            const pb::Scene sc = scene(n, R, rP, sphere);
            const auto xs = points(X, "X"), ds = points(D, "D");
            if (xs.size() != ds.size()) throw std::invalid_argument("X and D differ in length");
            std::vector<double> out(xs.size());
            for (std::size_t i = 0; i < xs.size(); ++i) out[i] = pb::miss(sc, xs[i], ds[i]);
            return to_array(out, {static_cast<py::ssize_t>(xs.size())});
        },
        py::arg("n"), py::arg("R"), py::arg("rP"), py::arg("sphere"), py::arg("X"), py::arg("D"),
        "The landing miss (_miss) of the ray from the pinhole along each row of D at the point of the row of X.");

    m.def(
        "single_start",
        [](const std::vector<double>& n, const std::vector<double>& R, double rP, bool sphere,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& X, int passes) {
            const pb::Scene sc = scene(n, R, rP, sphere);
            const rc::Stack st{n, R};
            const auto pts = points(X, "X");
            const std::size_t mm = pts.size();
            if (passes < 1) throw std::invalid_argument("passes must be at least 1");
            std::vector<std::vector<RayOut>> rays(mm);
            std::vector<pb::Result> res(NM * mm);
            std::vector<std::int64_t> t_ours(passes * mm), t(passes * NM * mm);
            std::int64_t differ = 0;
            {
                py::gil_scoped_release nogil;
                for (int p = 0; p < passes; ++p) {
                    for (std::size_t i = 0; i < mm; ++i) {  // per point, every method in turn: load hits all alike
                        std::vector<RayOut> got;
                        t_ours[p * mm + i] = timed([&] { got = ours_one(sc, st, pts[i]); });
                        if (p == 0)
                            rays[i] = std::move(got);
                        else if (!same_rays(rays[i], got))
                            ++differ;
                        for (int k = 0; k < NM; ++k) {
                            pb::Result r;
                            t[(p * NM + k) * mm + i] = timed([&] { r = pb::call(sc, k, pts[i], nullptr); });
                            if (p == 0)
                                res[k * mm + i] = r;
                            else if (!same_result(res[k * mm + i], r))
                                ++differ;
                        }
                    }
                }
            }
            std::vector<double> d(NM * mm * 3), e(NM * mm);
            std::vector<std::int8_t> has_d(NM * mm), status(NM * mm);
            std::vector<std::int32_t> ev(NM * mm), it(NM * mm), evj(NM * mm);
            for (std::size_t j = 0; j < NM * mm; ++j) {
                for (int c = 0; c < 3; ++c) d[3 * j + c] = res[j].d[c];
                has_d[j] = res[j].has_d;
                status[j] = static_cast<std::int8_t>(res[j].status);
                ev[j] = res[j].ev;
                it[j] = res[j].it;
                evj[j] = res[j].evj;
                e[j] = res[j].e;
            }
            const auto M = static_cast<py::ssize_t>(mm);
            py::dict out;
            out["ours"] = rays_to_py(rays);
            out["ours_t_ns"] = to_array(t_ours, {passes, M});
            out["d"] = to_array(d, {NM, M, 3});
            out["has_d"] = to_array(has_d, {NM, M});
            out["status"] = to_array(status, {NM, M});
            out["ev"] = to_array(ev, {NM, M});
            out["it"] = to_array(it, {NM, M});
            out["evj"] = to_array(evj, {NM, M});
            out["e"] = to_array(e, {NM, M});
            out["t_ns"] = to_array(t, {passes, NM, M});
            out["passes_differ"] = differ;
            return out;
        },
        py::arg("n"), py::arg("R"), py::arg("rP"), py::arg("sphere"), py::arg("X"), py::arg("passes") = 3,
        "Every method once per point from its paper's start, `passes` times (results from the first pass; "
        "passes_differ counts results that changed between passes), each call timed in ns.");

    m.def(
        "multistart",
        [](const std::vector<double>& n, const std::vector<double>& R, double rP, bool sphere,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& X,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& starts) {
            const pb::Scene sc = scene(n, R, rP, sphere);
            const auto pts = points(X, "X");
            const std::size_t mm = pts.size();
            if (starts.ndim() != 3 || static_cast<std::size_t>(starts.shape(0)) != mm || starts.shape(2) != 3)
                throw std::invalid_argument("starts must be (m, K - 1, 3)");
            const std::size_t K = static_cast<std::size_t>(starts.shape(1)) + 1;
            std::vector<pb::V3> st(mm * (K - 1));
            {
                auto r = starts.unchecked<3>();
                for (std::size_t i = 0; i < mm; ++i)
                    for (std::size_t j = 0; j + 1 < K; ++j) st[i * (K - 1) + j] = {r(i, j, 0), r(i, j, 1), r(i, j, 2)};
            }
            const std::size_t total = NM * mm * K;
            std::vector<std::int8_t> status(total), traced(total), kept(total);
            std::vector<std::int32_t> ev(total), it(total);
            std::vector<std::int64_t> t(total);
            std::vector<double> d(3 * total);
            {
                py::gil_scoped_release nogil;
                for (std::size_t i = 0; i < mm; ++i) {
                    for (int k = 0; k < NM; ++k) {
                        for (std::size_t j = 0; j < K; ++j) {
                            const pb::V3* d0 = j == 0 ? nullptr : &st[i * (K - 1) + j - 1];
                            pb::Result r;
                            bool tr = false, kp = false;
                            const std::size_t q = (k * mm + i) * K + j;
                            // the solve and the re-trace that keeps its answer, timed together (multistart())
                            t[q] = timed([&] {
                                r = pb::call(sc, k, pts[i], d0);
                                tr = (r.status == pb::ZERO || r.status == pb::STATIONARY) && r.has_d;
                                kp = tr && pb::miss(sc, pts[i], r.d) <= pb::TOL;
                            });
                            status[q] = static_cast<std::int8_t>(r.status);
                            ev[q] = r.ev;
                            it[q] = r.it;
                            traced[q] = tr;
                            kept[q] = kp;
                            for (int c = 0; c < 3; ++c) d[3 * q + c] = r.d[c];
                        }
                    }
                }
            }
            const auto M = static_cast<py::ssize_t>(mm), KK = static_cast<py::ssize_t>(K);
            py::dict out;
            out["status"] = to_array(status, {NM, M, KK});
            out["ev"] = to_array(ev, {NM, M, KK});
            out["it"] = to_array(it, {NM, M, KK});
            out["traced"] = to_array(traced, {NM, M, KK});
            out["kept"] = to_array(kept, {NM, M, KK});
            out["d"] = to_array(d, {NM, M, KK, 3});
            out["t_ns"] = to_array(t, {NM, M, KK});
            return out;
        },
        py::arg("n"), py::arg("R"), py::arg("rP"), py::arg("sphere"), py::arg("X"), py::arg("starts"),
        "Every baseline on each point from K starts: its paper's, then the rays of starts (m, K - 1, 3). Each start "
        "is timed with the re-trace of its answer, which is kept if it lands within TOL.");
}
