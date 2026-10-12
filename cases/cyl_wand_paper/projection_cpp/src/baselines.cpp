#include "baselines.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <optional>
#include <tuple>
#include <utility>

// Each function ports the paper_baselines.py function named in its comment, in the same order of operations, so that
// the two agree to rounding (test C3). Where the Python would raise on an input it never met, the port returns a
// breakdown instead; the comment says so.

// Python's math module calls the C runtime's functions; /O2 lets MSVC substitute its intrinsic forms, which can differ
// in the last bit. Force the runtime's, so that the port computes what the Python computes (test C3).
#ifdef _MSC_VER
#pragma function(pow, sin, cos, atan2, asin)
#endif

namespace pb {
namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
using X2 = std::array<double, 2>;

inline double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline V3 unit(const V3& a) {  // _unit
    const double s = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    return {a[0] / s, a[1] / s, a[2] / s};
}

inline V3 sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline V3 neg(const V3& a) { return {-a[0], -a[1], -a[2]}; }
inline V3 along(const V3& p, double t, const V3& d) { return {p[0] + t * d[0], p[1] + t * d[1], p[2] + t * d[2]}; }

// Python's math.hypot and math.dist (Python 3.10+, Modules/mathmodule.c, vector_norm): a correctly rounded norm by
// power-of-two scaling, Veltkamp-Dekker splitting, compensated summation and a differential correction of the square
// root. The C runtime's hypot differs from it in the last bit at 7 % of random pairs (test C3b), which moved the
// cylinder's runs; this re-implementation equals math.hypot and math.dist bitwise (C3b, 200 000 cases each).
double vector_norm(double* vec, int n) {
    constexpr double T27 = 134217729.0;  // 2^27 + 1
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        vec[i] = std::fabs(vec[i]);
        if (vec[i] > mx) mx = vec[i];
    }
    if (std::isinf(mx)) return mx;
    for (int i = 0; i < n; ++i)
        if (std::isnan(vec[i])) return std::numeric_limits<double>::quiet_NaN();
    if (mx == 0.0 || n <= 1) return mx;
    int max_e;
    std::frexp(mx, &max_e);
    if (max_e < -1023) {  // subnormal: rescale first, as the Python does
        for (int i = 0; i < n; ++i) vec[i] /= DBL_MIN;
        return DBL_MIN * vector_norm(vec, n);
    }
    const double scale = std::ldexp(1.0, -max_e);
    double csum = 1.0, frac1 = 0.0, frac2 = 0.0, frac3 = 0.0, oldcsum, x, t, hi, lo;
    for (int i = 0; i < n; ++i) {
        x = vec[i] * scale;
        t = x * T27;
        hi = t - (t - x);
        lo = x - hi;
        x = hi * hi;
        oldcsum = csum;
        csum += x;
        frac1 += (oldcsum - csum) + x;
        x = 2.0 * hi * lo;
        oldcsum = csum;
        csum += x;
        frac2 += (oldcsum - csum) + x;
        frac3 += lo * lo;
    }
    const double h = std::sqrt(csum - 1.0 + (frac1 + frac2 + frac3));
    x = h;
    t = x * T27;
    hi = t - (t - x);
    lo = x - hi;
    x = -hi * hi;
    oldcsum = csum;
    csum += x;
    frac1 += (oldcsum - csum) + x;
    x = -2.0 * hi * lo;
    oldcsum = csum;
    csum += x;
    frac2 += (oldcsum - csum) + x;
    x = -lo * lo;
    oldcsum = csum;
    csum += x;
    frac3 += (oldcsum - csum) + x;
    x = csum - 1.0 + (frac1 + frac2 + frac3);
    return (h + x / (2.0 * h)) / scale;
}

inline double py_hypot(double a, double b) {  // math.hypot(a, b)
    double v[2] = {a, b};
    return vector_norm(v, 2);
}

// math.dist
inline double dist(const V3& a, const V3& b) {
    double v[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    return vector_norm(v, 3);
}

// _roots: both parameters t, in increasing order, where p + t d meets the cylinder r = R (axis z) or the sphere |x| = R
std::optional<std::pair<double, double>> roots(const V3& p, const V3& d, double R, bool cyl) {
    double a, b, c;
    if (cyl) {
        a = d[0] * d[0] + d[1] * d[1];
        b = p[0] * d[0] + p[1] * d[1];
        c = p[0] * p[0] + p[1] * p[1] - R * R;
    } else {
        a = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        b = p[0] * d[0] + p[1] * d[1] + p[2] * d[2];
        c = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - R * R;
    }
    const double disc = b * b - a * c;
    if (a <= 0.0 || disc < 0.0) return std::nullopt;
    const double q = -(b + std::copysign(std::sqrt(disc), b));  // the roots without cancellation
    if (q == 0.0) return std::make_pair(0.0, 0.0);
    const double t1 = q / a, t2 = c / q;
    return t1 <= t2 ? std::make_pair(t1, t2) : std::make_pair(t2, t1);
}

// _normal: outward unit normal of the interface through h
V3 normal(const V3& h, bool cyl) {
    if (cyl) {
        const double s = py_hypot(h[0], h[1]);
        return {h[0] / s, h[1] / s, 0.0};
    }
    return unit(h);
}

// _snell: vector Snell's law (Belden's Eq. 11); nullopt on total internal reflection
std::optional<V3> snell(const V3& d, V3 nrm, double eta) {
    double c = -dot(nrm, d);
    if (c < 0.0) {
        nrm = neg(nrm);
        c = -c;
    }
    const double k = 1.0 - eta * eta * (1.0 - c * c);
    if (k < 0.0) return std::nullopt;
    const double f = eta * c - std::sqrt(k);
    return V3{eta * d[0] + f * nrm[0], eta * d[1] + f * nrm[1], eta * d[2] + f * nrm[2]};
}

// _to_inner: trace from p (outside R_1) into the innermost medium; its entry point there and its direction
std::optional<std::pair<V3, V3>> to_inner(const Scene& sc, V3 p, V3 d) {
    const bool cyl = sc.cyl();
    for (std::size_t k = 0; k < sc.R.size(); ++k) {
        const auto ts = roots(p, d, sc.R[k], cyl);
        if (!ts || ts->first <= 0.0) return std::nullopt;
        p = along(p, ts->first, d);
        const auto dn = snell(d, normal(p, cyl), sc.n[k] / sc.n[k + 1]);
        if (!dn) return std::nullopt;
        d = *dn;
    }
    return std::make_pair(p, d);
}

Result exit_with(int status, int ev, int it, const V3* d = nullptr) {
    Result r;
    r.status = status;
    r.ev = ev;
    r.it = it;
    if (d) {
        r.d = *d;
        r.has_d = true;
    }
    return r;
}

// ------------------------------------------------------------------ least squares: Gauss-Newton, Levenberg-Marquardt

struct LSQ {
    X2 x{0.0, 0.0};
    int status = BREAKDOWN;
    int ev = 0;
    int it = 0;
    int evj = 0;
    double e = INF;
};

// _fdjac: forward-difference Jacobian (3 x 2), a backward difference where the forward point has no value
template <class F>
bool fdjac(F& res, const X2& x, const V3& E, const X2& hs, std::array<V3, 2>& cols, int& ev, int& evj) {
    int e = 0;
    for (int j = 0; j < 2; ++j) {
        double h = hs[j];
        X2 xp = x;
        xp[j] += h;
        std::optional<V3> Ep = res(xp);
        e += 1;
        if (!Ep) {
            xp[j] = x[j] - h;
            Ep = res(xp);
            e += 1;
            if (!Ep) {
                evj += e;
                ev += e;
                return false;
            }
            h = -h;
        }
        cols[j] = {((*Ep)[0] - E[0]) / h, ((*Ep)[1] - E[1]) / h, ((*Ep)[2] - E[2]) / h};
    }
    evj += e;
    ev += e;
    return true;
}

struct NormalEq {
    double a11, a12, a22, g1, g2;
};

NormalEq normal_eq(const std::array<V3, 2>& cols, const V3& E) {  // _normal_eq
    return {dot(cols[0], cols[0]), dot(cols[0], cols[1]), dot(cols[1], cols[1]), dot(cols[0], E), dot(cols[1], E)};
}

bool stationary(const V3& E, const NormalEq& q) {  // _stationary: MINPACK's gtol test
    const double e = std::sqrt(dot(E, E));
    if (e == 0.0) return true;
    const double c1 = q.a11 > 0 ? std::abs(q.g1) / (std::sqrt(q.a11) * e) : 0.0;
    const double c2 = q.a22 > 0 ? std::abs(q.g2) / (std::sqrt(q.a22) * e) : 0.0;
    return std::max(c1, c2) <= GTOL;
}

// gauss_newton: undamped Gauss-Newton on the residual res (3 components, 2 unknowns)
template <class F>
LSQ gauss_newton(F& res, const X2& x0, const X2& hs, double tol, int kmax) {
    LSQ o;
    X2 x = x0;
    std::optional<V3> E = res(x);
    int ev = 1;
    auto done = [&](int status, int it, double e) {
        o.x = x;
        o.status = status;
        o.ev = ev;
        o.it = it;
        o.e = e;
        return o;
    };
    if (!E) return done(BREAKDOWN, 0, INF);
    std::array<V3, 2> cols;
    for (int k = 1; k <= kmax; ++k) {
        if (std::sqrt(dot(*E, *E)) <= tol) return done(ZERO, k - 1, std::sqrt(dot(*E, *E)));
        if (!fdjac(res, x, *E, hs, cols, ev, o.evj)) return done(BREAKDOWN, k, std::sqrt(dot(*E, *E)));
        const NormalEq q = normal_eq(cols, *E);
        if (stationary(*E, q)) return done(STATIONARY, k, std::sqrt(dot(*E, *E)));
        const double det = q.a11 * q.a22 - q.a12 * q.a12;
        if (!(det > 0.0)) return done(BREAKDOWN, k, std::sqrt(dot(*E, *E)));
        const X2 h{(-q.a22 * q.g1 + q.a12 * q.g2) / det, (q.a12 * q.g1 - q.a11 * q.g2) / det};
        if (py_hypot(h[0], h[1]) <= XTOL * (py_hypot(x[0], x[1]) + XTOL))
            return done(STATIONARY, k, std::sqrt(dot(*E, *E)));
        x = {x[0] + h[0], x[1] + h[1]};
        E = res(x);
        ev += 1;
        if (!E) return done(BREAKDOWN, k, INF);
    }
    const double e = std::sqrt(dot(*E, *E));
    return done(e <= tol ? ZERO : CAP, kmax, e);
}

// levenberg_marquardt: Nielsen's damping update, the residual test first, then MINPACK's gtol and xtol tests; a trial
// point without a value is a rejected step
template <class F>
LSQ levenberg_marquardt(F& res, const X2& x0, const X2& hs, double tol, int kmax, double tau) {
    LSQ o;
    X2 x = x0;
    std::optional<V3> E = res(x);
    int ev = 1;
    auto done = [&](int status, int it, double e) {
        o.x = x;
        o.status = status;
        o.ev = ev;
        o.it = it;
        o.e = e;
        return o;
    };
    if (!E) return done(BREAKDOWN, 0, INF);
    if (std::sqrt(dot(*E, *E)) <= tol) return done(ZERO, 0, std::sqrt(dot(*E, *E)));
    std::array<V3, 2> cols;
    if (!fdjac(res, x, *E, hs, cols, ev, o.evj)) return done(BREAKDOWN, 0, std::sqrt(dot(*E, *E)));
    NormalEq q = normal_eq(cols, *E);
    double mu = tau * std::max(q.a11, q.a22), nu = 2.0;
    for (int k = 1; k <= kmax; ++k) {
        if (stationary(*E, q)) return done(STATIONARY, k - 1, std::sqrt(dot(*E, *E)));
        const double b11 = q.a11 + mu, b22 = q.a22 + mu;
        const double det = b11 * b22 - q.a12 * q.a12;
        const X2 h{(-b22 * q.g1 + q.a12 * q.g2) / det, (q.a12 * q.g1 - b11 * q.g2) / det};
        if (py_hypot(h[0], h[1]) <= XTOL * (py_hypot(x[0], x[1]) + XTOL))
            return done(STATIONARY, k - 1, std::sqrt(dot(*E, *E)));
        const X2 xn{x[0] + h[0], x[1] + h[1]};
        const std::optional<V3> En = res(xn);
        ev += 1;
        double rho = -1.0;
        if (En) {
            const double gain = h[0] * (mu * h[0] - q.g1) + h[1] * (mu * h[1] - q.g2);  // 2 (L(0) - L(h))
            rho = gain > 0.0 ? (dot(*E, *E) - dot(*En, *En)) / gain : -1.0;
        }
        if (rho > 0.0) {
            x = xn;
            E = En;
            if (std::sqrt(dot(*E, *E)) <= tol) return done(ZERO, k, std::sqrt(dot(*E, *E)));
            if (!fdjac(res, x, *E, hs, cols, ev, o.evj)) return done(BREAKDOWN, k, std::sqrt(dot(*E, *E)));
            q = normal_eq(cols, *E);
            mu *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
            nu = 2.0;
        } else {
            mu *= nu;
            nu *= 2.0;
        }
    }
    return done(CAP, kmax, std::sqrt(dot(*E, *E)));
}

// ------------------------------------------------------------------ (c) dewarping, Paolillo & Astarita 2022

struct Frame {
    V3 zeta, eu, ev;
};

// _dewarp_frame: zeta from the axis (the centre) to the focus; the plane of distortion is through X normal to zeta
Frame dewarp_frame(const Scene& sc) {
    const V3 P0 = sc.P0();
    if (sc.cyl()) {
        const double s = py_hypot(P0[0], P0[1]);
        const V3 zeta{P0[0] / s, P0[1] / s, 0.0};
        return {zeta, {-zeta[1], zeta[0], 0.0}, {0.0, 0.0, 1.0}};
    }
    const V3 zeta = unit(P0);
    const V3 a = std::abs(zeta[2]) < 0.9 ? V3{0.0, 0.0, 1.0} : V3{1.0, 0.0, 0.0};
    const V3 eu = unit({a[1] * zeta[2] - a[2] * zeta[1], a[2] * zeta[0] - a[0] * zeta[2],
                        a[0] * zeta[1] - a[1] * zeta[0]});
    const V3 ev{zeta[1] * eu[2] - zeta[2] * eu[1], zeta[2] * eu[0] - zeta[0] * eu[2],
                zeta[0] * eu[1] - zeta[1] * eu[0]};
    return {zeta, eu, ev};
}

// _los_body: the nearest lambda > 0 at which xi + lambda r meets an interface, and which one (-1: none); xi lies on
// the interface `last` (-1: none), whose root there is dropped
std::pair<double, int> los_body(const Scene& sc, const V3& xi, const V3& r, int last) {
    double best = 0.0;
    int which = -1;
    for (int j = 0; j < static_cast<int>(sc.R.size()); ++j) {
        const auto ts = roots(xi, r, sc.R[j], sc.cyl());
        if (!ts) continue;
        double cand[2];
        int nc;
        if (j == last) {
            cand[0] = std::abs(ts->first) <= std::abs(ts->second) ? ts->second : ts->first;
            nc = 1;
        } else {
            cand[0] = ts->first;
            cand[1] = ts->second;
            nc = 2;
        }
        for (int i = 0; i < nc; ++i) {
            const double t = cand[i];
            if (t > 1e-12 && (which < 0 || t < best)) {
                best = t;
                which = j;
            }
        }
    }
    return {best, which};
}

// pa_distort: their Algorithm 1, the point where the line of sight from the focus through the guess A, refracted at
// every interface it meets, reaches the plane through A normal to zeta; the point of total internal reflection if one
// occurs. One ray-path evaluation.
V3 pa_distort(const Scene& sc, const V3& A, const V3& zeta) {
    const auto& n = sc.n;
    const bool cyl = sc.cyl();
    const int N = static_cast<int>(n.size());
    V3 xi = sc.P0();
    V3 r = unit(sub(A, xi));
    int med = 0, last = -1;
    auto [lam, surf] = los_body(sc, xi, r, last);
    double lamA = dot(sub(A, xi), zeta) / dot(r, zeta);
    while (lamA > lam && lam > 0.0) {
        xi = along(xi, lam, r);
        V3 nrm = normal(xi, cyl);
        double c = dot(nrm, r);
        if (c > 0.0) {
            nrm = neg(nrm);
            c = -c;
        }
        int nxt = med == surf ? med + 1 : med - 1;  // interface j separates media j and j + 1
        if (nxt < 0) nxt += N;                       // Python's n[-1]; unreachable for a ray from outside
        const double rho = n[med] / n[nxt];
        const double d = 1.0 - rho * rho * (1.0 - c * c);
        if (d <= 0.0) return xi;
        const double f = rho * c + std::sqrt(d);
        r = {rho * r[0] - f * nrm[0], rho * r[1] - f * nrm[1], rho * r[2] - f * nrm[2]};
        med = nxt;
        last = surf;
        std::tie(lam, surf) = los_body(sc, xi, r, last);
        lamA = dot(sub(A, xi), zeta) / dot(r, zeta);
    }
    return along(xi, lamA, r);
}

// ------------------------------------------------------------------ (d) shooting on the direction angles

struct ShootFrame {
    V3 ein, et, ez;
};

ShootFrame shoot_frame(const Scene& sc) {  // _shoot_frame
    const V3 P0 = sc.P0();
    const double s = py_hypot(P0[0], P0[1]);
    const V3 ein{-P0[0] / s, -P0[1] / s, 0.0};
    return {ein, {-ein[1], ein[0], 0.0}, {0.0, 0.0, 1.0}};
}

V3 shoot_dir(const X2& ang, const ShootFrame& fr) {  // _dir
    const double ca = std::cos(ang[0]), sa = std::sin(ang[0]), cb = std::cos(ang[1]), sb = std::sin(ang[1]);
    return {cb * (ca * fr.ein[0] + sa * fr.et[0]) + sb * fr.ez[0], cb * (ca * fr.ein[1] + sa * fr.et[1]) + sb * fr.ez[1],
            cb * (ca * fr.ein[2] + sa * fr.et[2]) + sb * fr.ez[2]};
}

}  // namespace

// ------------------------------------------------------------------ (b) AFRT, Belden 2013, Fig. 5

Result afrt(const Scene& sc, const V3& X, const V3* d0, double tol, int kmax) {
    const auto& n = sc.n;
    const double R1 = sc.R[0], R2 = sc.R[1];
    const bool cyl = sc.cyl();
    const V3 XC = sc.P0();
    const double n12 = n[0] / n[1], n23 = n[1] / n[2], n32 = n[2] / n[1];
    int ev = 0;
    V3 r1 = d0 ? *d0 : unit(sub(X, XC));  // k = 1: the line to P
    const auto t = roots(XC, r1, R1, cyl);
    if (!t || t->first <= 0.0) return exit_with(BREAKDOWN, ev, 0);  // a start ray that misses S1
    V3 Xi1 = along(XC, t->first, r1);
    V3 Xi1p{0.0, 0.0, 0.0};
    for (int k = 1; k <= kmax; ++k) {
        if (k > 1) {
            // the midpoint of the forward and backward hits on S1, projected onto S1 (radially)
            const V3 M{0.5 * (Xi1[0] + Xi1p[0]), 0.5 * (Xi1[1] + Xi1p[1]), 0.5 * (Xi1[2] + Xi1p[2])};
            if (cyl) {
                const double s = R1 / py_hypot(M[0], M[1]);
                Xi1 = {M[0] * s, M[1] * s, M[2]};
            } else {
                const double s = R1 / std::sqrt(dot(M, M));
                Xi1 = {M[0] * s, M[1] * s, M[2] * s};
            }
            r1 = unit(sub(Xi1, XC));
        }
        // forward path: refract at S1, meet S2, refract into the liquid
        ev += 1;
        const auto r2 = snell(r1, normal(Xi1, cyl), n12);
        if (!r2) return exit_with(BREAKDOWN, ev, k);
        auto ts = roots(Xi1, *r2, R2, cyl);
        if (!ts || ts->first <= 0.0) return exit_with(BREAKDOWN, ev, k);  // the ray passes through the wall only
        const V3 Xi2 = along(Xi1, ts->first, *r2);
        const auto r3 = snell(*r2, normal(Xi2, cyl), n23);
        if (!r3) return exit_with(BREAKDOWN, ev, k);
        // backward path: from P along -r3 out to S2, refract into the wall, meet S1
        ev += 1;
        const V3 m3 = neg(*r3);
        ts = roots(X, m3, R2, cyl);
        if (!ts) return exit_with(BREAKDOWN, ev, k);  // the Python would raise here: never met (X lies inside S2)
        const V3 Xi2p = along(X, ts->second, m3);
        const auto r4 = snell(m3, normal(Xi2p, cyl), n32);
        if (!r4) return exit_with(BREAKDOWN, ev, k);
        ts = roots(Xi2p, *r4, R1, cyl);
        if (!ts) return exit_with(BREAKDOWN, ev, k);
        Xi1p = along(Xi2p, ts->second, *r4);
        const double dX1 = dist(Xi1, Xi1p);
        const double dX2 = dist(Xi2, Xi2p);
        if (dX1 <= tol && dX2 <= tol) return exit_with(ZERO, ev, k, &r1);
    }
    return exit_with(CAP, ev, kmax, &r1);
}

// ------------------------------------------------------------------ dewarp: their Eq. (15) for A = X + u eu + v ev

Result dewarp(const Scene& sc, const V3& X, bool lm, const V3* d0, double tol) {
    const Frame fr = dewarp_frame(sc);
    const V3& zeta = fr.zeta;
    const V3& eu = fr.eu;
    const V3& ev_ = fr.ev;
    auto res = [&](const X2& x) -> std::optional<V3> {
        const V3 A{X[0] + x[0] * eu[0] + x[1] * ev_[0], X[1] + x[0] * eu[1] + x[1] * ev_[1],
                   X[2] + x[0] * eu[2] + x[1] * ev_[2]};
        const V3 F = pa_distort(sc, A, zeta);
        return V3{F[0] - X[0], F[1] - X[1], F[2] - X[2]};
    };
    X2 x0{0.0, 0.0};
    const V3 P0 = sc.P0();
    if (d0) {
        const V3& d = *d0;
        const double s = dot(sub(X, P0), zeta) / dot(d, zeta);
        const V3 w{P0[0] + s * d[0] - X[0], P0[1] + s * d[1] - X[1], P0[2] + s * d[2] - X[2]};
        x0 = {dot(w, eu), dot(w, ev_)};
    }
    const double h = 1e-8 * sc.R[0];  // forward-difference step, m
    const LSQ o = lm ? levenberg_marquardt(res, x0, {h, h}, tol, LM_KMAX, LM_TAU)
                     : gauss_newton(res, x0, {h, h}, tol, GN_KMAX);
    const X2& x = o.x;
    const V3 A{X[0] + x[0] * eu[0] + x[1] * ev_[0], X[1] + x[0] * eu[1] + x[1] * ev_[1],
               X[2] + x[0] * eu[2] + x[1] * ev_[2]};
    Result r;
    r.d = unit(sub(A, P0));
    r.has_d = true;
    r.status = o.status;
    r.ev = o.ev;
    r.it = o.it;
    r.evj = o.evj;
    r.e = o.e;
    return r;
}

// ------------------------------------------------------------------ shooting: LM on (alpha, beta) at the pinhole

Result shooting(const Scene& sc, const V3& X, const V3* d0, double tol) {
    const ShootFrame fr = shoot_frame(sc);
    const V3 P0 = sc.P0();
    auto res = [&](const X2& ang) -> std::optional<V3> {
        const auto tr = to_inner(sc, P0, shoot_dir(ang, fr));
        if (!tr) return std::nullopt;
        const V3& Q = tr->first;
        const V3& d = tr->second;
        const V3 v = sub(X, Q);
        const double t = dot(v, d);
        return V3{v[0] - t * d[0], v[1] - t * d[1], v[2] - t * d[2]};
    };
    const V3 dd = d0 ? *d0 : unit(sub(X, P0));
    const X2 a0{std::atan2(dot(dd, fr.et), dot(dd, fr.ein)), std::asin(std::max(-1.0, std::min(1.0, dot(dd, fr.ez))))};
    const LSQ o = levenberg_marquardt(res, a0, {1e-8, 1e-8}, tol, LM_KMAX, LM_TAU);
    Result r;
    r.d = shoot_dir(o.x, fr);
    r.has_d = true;
    r.status = o.status;
    r.ev = o.ev;
    r.it = o.it;
    r.evj = o.evj;
    r.e = o.e;
    return r;
}

Result call(const Scene& sc, int method, const V3& X, const V3* d0) {
    switch (method) {
        case AFRT: return afrt(sc, X, d0);
        case GN: return dewarp(sc, X, false, d0);
        case LM: return dewarp(sc, X, true, d0);
        default: return shooting(sc, X, d0);
    }
}

double miss(const Scene& sc, const V3& X, const V3& d) {
    const auto tr = to_inner(sc, sc.P0(), d);
    if (!tr) return INF;
    const V3& Q = tr->first;
    const V3& e = tr->second;
    const V3 v = sub(X, Q);
    const double t = dot(v, e);
    if (t <= 0.0) return INF;
    const V3 w{v[0] - t * e[0], v[1] - t * e[1], v[2] - t * e[2]};
    return std::sqrt(dot(w, w));
}

}  // namespace pb
