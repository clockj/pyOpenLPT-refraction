#pragma once

// C++ ports of the iterative projections of the paper's evidence E1, D:/Refraction/code/paper_baselines.py, line for
// line: the same algorithms, constants, stopping tests, exit statuses and count of ray-path evaluations. Plain doubles
// and std::array, like pyOpenLPT's RefractCurved, so the two are compiled and timed alike.
//
//   (b) AFRT, the alternating forward ray tracing of Belden, Exp. Fluids 54:1463 (2013), his Fig. 5;
//   (c) dewarping after Paolillo & Astarita, IEEE TPAMI 44:3185 (2022), their Algorithm 1 and Eq. (15), solved by
//       Gauss-Newton or by Levenberg-Marquardt;
//   (d) shooting: Levenberg-Marquardt on the two direction angles at the pinhole.
//
// Each returns at most one ray. The docstrings of paper_baselines.py describe each step; the comments here name the
// Python function each piece ports.

#include <array>
#include <limits>
#include <vector>

namespace pb {

using V3 = std::array<double, 3>;

// Exit statuses, numbered as STATUS in paper_baselines.py ("list" is our search's).
enum Status : int { LIST = 0, ZERO = 1, STATIONARY = 2, CAP = 3, BREAKDOWN = 4 };

// The methods, in the order of BASELINES in paper_baselines.py.
enum Method : int { AFRT = 0, GN = 1, LM = 2, SHOOT = 3 };

constexpr double TOL = 1e-9;     // m: the landing tolerance of every method, the stopping tolerance of every baseline
constexpr int AFRT_KMAX = 2000;  // iterations (Belden's k_max; he states no value)
constexpr int GN_KMAX = 100;     // Gauss-Newton iterations
constexpr int LM_KMAX = 300;     // Levenberg-Marquardt trial steps
constexpr double LM_TAU = 1e-6;  // initial damping mu_0 = LM_TAU * max diag(J^T J)
constexpr double GTOL = 1e-10;   // stationary: MINPACK's gtol test
constexpr double XTOL = 1e-10;   // stationary: MINPACK's xtol test

/** n from the camera inward, interface radii R from outside in (m), pinhole P0 = (rP, 0, 0); the axis is z for a
    cylinder, the centre is the origin for a sphere (class Scene). */
struct Scene {
    std::vector<double> n;
    std::vector<double> R;
    double rP = 0.0;
    bool sphere = false;
    V3 P0() const { return {rP, 0.0, 0.0}; }
    bool cyl() const { return !sphere; }
};

/** One run of a baseline: what _call() returns, plus the Jacobian evaluations (_JAC) and |residual|. */
struct Result {
    V3 d{0.0, 0.0, 0.0};  // the direction at the pinhole: the answer, or the last iterate
    bool has_d = false;   // false where the Python returns None
    int status = BREAKDOWN;
    int ev = 0;           // ray-path evaluations
    int it = 0;           // iterations
    int evj = 0;          // of which spent on finite-difference Jacobians
    double e = std::numeric_limits<double>::infinity();  // |residual| (Gauss-Newton, Levenberg-Marquardt)
};

Result afrt(const Scene& sc, const V3& X, const V3* d0 = nullptr, double tol = TOL, int kmax = AFRT_KMAX);
Result dewarp(const Scene& sc, const V3& X, bool lm, const V3* d0 = nullptr, double tol = TOL);
Result shooting(const Scene& sc, const V3& X, const V3* d0 = nullptr, double tol = TOL);

/** _call: one baseline on X from the start ray d0 (nullptr: the start of its paper). */
Result call(const Scene& sc, int method, const V3& X, const V3* d0 = nullptr);

/** _miss: the landing miss of the ray P0 + t d, traced like the baselines; inf if it does not reach the innermost
    medium or X lies behind its entry. One ray-path evaluation. */
double miss(const Scene& sc, const V3& X, const V3& d);

}  // namespace pb
