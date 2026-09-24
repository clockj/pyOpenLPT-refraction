// Bindings for the curved-port refraction core.
//
// These expose refract_curved:: directly, one function per closed form, so the
// C++ can be compared term by term with the reference implementations in
// test/refs/.  The camera classes are bound in pyCamera.cpp; nothing here is
// meant for end users, it is the seam the validation suite tests through.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

#include "RefractCurved.h"

namespace py = pybind11;
using namespace refract_curved;

namespace {

Stack makeStack(const std::vector<double>& n, const std::vector<double>& R) {
    Stack s;
    s.n = n;
    s.R = R;
    if (!s.isValid()) {
        throw std::invalid_argument(
            "invalid stack: need len(R) == len(n)-1, positive indices and "
            "strictly decreasing radii");
    }
    return s;
}

std::array<double, 3> toArray3(const std::vector<double>& v, const char* what) {
    if (v.size() != 3) {
        throw std::invalid_argument(std::string(what) + " must have 3 components");
    }
    return {v[0], v[1], v[2]};
}

Branch toBranch(bool far) { return far ? Branch::Far : Branch::Near; }

} // namespace

void bind_RefractCurved(py::module_& m) {
    py::module_ sub = m.def_submodule(
        "refract_curved",
        "Closed forms and projection for cylindrical and spherical refractive "
        "ports (doc/refract_curved.tex). Index order: medium 1 holds the "
        "pinhole, medium N the object point.");

    sub.def(
        "nus",
        [](const std::vector<double>& n, double c_z) { return nus(n, c_z); },
        py::arg("n"), py::arg("c_z"),
        "nu_k = sqrt(n_k^2 - C_z^2) for each medium.");

    sub.def(
        "lmax_kstar",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_x,
           double c_z) {
            const LmaxKstar v = lmaxKstar(makeStack(n, R), r_x, c_z);
            return py::make_tuple(v.lmax, v.kstar);
        },
        py::arg("n"), py::arg("R"), py::arg("r_x"), py::arg("c_z"),
        "Eq. (adm): (Lmax, k*) at this tilt.");

    sub.def(
        "phi_z",
        [](double lam, double c_z, const std::vector<double>& n,
           const std::vector<double>& R, double r_p, double r_x,
           bool far) -> py::object {
            const auto v = phiZ(lam, c_z, makeStack(n, R), r_p, r_x, toBranch(far));
            if (!v) {
                return py::none();
            }
            return py::make_tuple(v->phi, v->z);
        },
        py::arg("lam"), py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"),
        py::arg("r_x"), py::arg("far") = false,
        "Eqs. (Phinear)/(Phifar)/(Z); None when the ray never reaches r_X.");

    sub.def(
        "radial_action",
        [](double lam, double c_z, const std::vector<double>& n,
           const std::vector<double>& R, double r_p, double r_x,
           bool far) -> py::object {
            const auto v =
                radialAction(lam, c_z, makeStack(n, R), r_p, r_x, toBranch(far));
            return v ? py::cast(*v) : py::none();
        },
        py::arg("lam"), py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"),
        py::arg("r_x"), py::arg("far") = false, "Eq. (W), the radial action.");

    sub.def(
        "dphi_dlam",
        [](double lam, double c_z, const std::vector<double>& n,
           const std::vector<double>& R, double r_p, double r_x, bool far) {
            return dPhiDLam(lam, c_z, makeStack(n, R), r_p, r_x, toBranch(far));
        },
        py::arg("lam"), py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"),
        py::arg("r_x"), py::arg("far") = false, "Eq. (dPhi).");

    sub.def(
        "dphi_dcz",
        [](double lam, double c_z, const std::vector<double>& n,
           const std::vector<double>& R, double r_p, double r_x, bool far) {
            return dPhiDCz(lam, c_z, makeStack(n, R), r_p, r_x, toBranch(far));
        },
        py::arg("lam"), py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"),
        py::arg("r_x"), py::arg("far") = false,
        "dPhi/dC_z, which equals dZ/dLambda (Property symmetry).");

    sub.def(
        "dz_dcz",
        [](double lam, double c_z, const std::vector<double>& n,
           const std::vector<double>& R, double r_p, double r_x, bool far,
           double h) -> py::object {
            const auto v =
                dZDCz(lam, c_z, makeStack(n, R), r_p, r_x, toBranch(far), h);
            return v ? py::cast(*v) : py::none();
        },
        py::arg("lam"), py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"),
        py::arg("r_x"), py::arg("far") = false, py::arg("h") = 1e-7,
        "dZ/dC_z by central differences, as the reference implementation does.");

    sub.def(
        "z0",
        [](double c_z, const std::vector<double>& n, const std::vector<double>& R,
           double r_p, double r_x) { return z0(c_z, makeStack(n, R), r_p, r_x); },
        py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"),
        "Eq. (z0): the axial offset of the radial ray (Property zbound).");

    sub.def(
        "in_domain",
        [](double lam, double c_z, const std::vector<double>& n,
           const std::vector<double>& R, double r_x) {
            return inDomain(lam, c_z, makeStack(n, R), r_x);
        },
        py::arg("lam"), py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_x"));

    sub.def(
        "lam_at",
        [](double c_z, const std::vector<double>& n, const std::vector<double>& R,
           double r_p, double r_x, double target, int iters) -> py::object {
            const auto v = lamAt(c_z, makeStack(n, R), r_p, r_x, target, iters);
            return v ? py::cast(*v) : py::none();
        },
        py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"),
        py::arg("target"), py::arg("iters") = 90,
        "Near-branch root of Phi = target by bisection; None if out of reach.");

    sub.def(
        "polish",
        [](double lam, double c_z, const std::vector<double>& n,
           const std::vector<double>& R, double r_p, double r_x, double target,
           bool far) {
            return polish(lam, c_z, makeStack(n, R), r_p, r_x, target,
                          toBranch(far));
        },
        py::arg("lam"), py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"),
        py::arg("r_x"), py::arg("target"), py::arg("far") = false,
        "Clamped Newton on Phi(Lambda) = target at fixed tilt.");

    sub.def(
        "far_turns",
        [](double c_z, const std::vector<double>& n, const std::vector<double>& R,
           double r_p, double r_x, double tol) {
            return farTurns(c_z, makeStack(n, R), r_p, r_x, tol);
        },
        py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"),
        py::arg("tol") = 1e-14,
        "Every turning point of Phi_far on (0, Lmax); nothing is sampled.");

    sub.def(
        "far_roots",
        [](double c_z, const std::vector<double>& n, const std::vector<double>& R,
           double r_p, double r_x, double target) {
            py::list out;
            for (const FarRoot& fr : farRoots(c_z, makeStack(n, R), r_p, r_x, target)) {
                out.append(py::make_tuple(fr.lambda, fr.slope, fr.z));
            }
            return out;
        },
        py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"),
        py::arg("target"), "Roots of Phi_far = target as (Lambda, slope, Z).");

    sub.def(
        "chord_bound",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_p) {
            return chordBound(makeStack(n, R), r_p);
        },
        py::arg("n"), py::arg("R"), py::arg("r_p"), "Property chord.");

    sub.def(
        "sweeps",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_p,
           double r_x) {
            const Sweeps v = sweeps(makeStack(n, R), r_p, r_x);
            return py::make_tuple(v.near_reach, v.m0);
        },
        py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"),
        "(Phi_near(Lmax), M(0)): the near reach and the largest sweep.");

    sub.def(
        "ray_plane",
        [](const std::vector<double>& centre, const std::vector<double>& pinhole,
           const std::vector<double>& point) -> py::object {
            const auto c = toArray3(centre, "centre");
            const auto p = toArray3(pinhole, "pinhole");
            const auto x = toArray3(point, "point");
            const auto rp = rayPlane(c.data(), p.data(), x.data());
            if (!rp) {
                return py::none();
            }
            return py::make_tuple(
                rp->r_p, rp->r_x, rp->dphi,
                std::vector<double>(rp->e1, rp->e1 + 3),
                std::vector<double>(rp->e2, rp->e2 + 3), rp->axial);
        },
        py::arg("centre"), py::arg("pinhole"), py::arg("point"),
        "Section rayplane: (r_p, r_x, dphi, e1, e2, axial).");

    sub.def(
        "project_sphere",
        [](const std::vector<double>& n, const std::vector<double>& R,
           const std::vector<double>& centre, const std::vector<double>& pinhole,
           const std::vector<double>& point, double tol, int itmax) {
            const auto c = toArray3(centre, "centre");
            const auto p = toArray3(pinhole, "pinhole");
            const auto x = toArray3(point, "point");
            py::list out;
            for (const Ray& r : projectSphere(makeStack(n, R), c.data(), p.data(),
                                              x.data(), tol, itmax)) {
                out.append(py::make_tuple(
                    r.is_ring ? "ring" : (r.branch == Branch::Far ? "far" : "near"),
                    r.lambda, std::vector<double>(r.dir, r.dir + 3)));
            }
            return out;
        },
        py::arg("n"), py::arg("R"), py::arg("centre"), py::arg("pinhole"),
        py::arg("point"), py::arg("tol") = 1e-9, py::arg("itmax") = -1,
        "Section spherealg: every image, as (branch, Lambda, direction). Lengths "
        "are scaled by 2^floor(log2 r_P) inside, so tol is dimensionless; itmax caps the Newton "
        "polish (-1: 30). No ray when the model does not apply.");

    sub.def(
        "solve_cylinder",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_p,
           double r_x, double dphi, double dz, double lam0, double cz0,
           bool far) {
            const SolveResult v = solveCylinder(makeStack(n, R), r_p, r_x, dphi, dz,
                                                lam0, cz0, toBranch(far));
            return py::make_tuple(v.ok, v.lambda, v.c_z, v.residual, v.iters);
        },
        py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"), py::arg("dphi"),
        py::arg("dz"), py::arg("lam0"), py::arg("cz0"), py::arg("far") = false,
        "Damped Newton on eq. (system): (ok, Lambda, C_z, residual, iters).");

    sub.def(
        "switch_tilts",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_x,
           double chi) { return switchTilts(makeStack(n, R), r_x, chi); },
        py::arg("n"), py::arg("R"), py::arg("r_x"), py::arg("chi"),
        "Tilts where k* can switch.");

    sub.def(
        "reach_terms",
        [](double c_z, const std::vector<double>& n, const std::vector<double>& R,
           double r_p, double r_x) {
            return reachTerms(c_z, makeStack(n, R), r_p, r_x);
        },
        py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"),
        "The near-branch reach, medium by medium.");

    sub.def(
        "components",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_p,
           double r_x, double dphi, double chi, double tol) {
            py::list out;
            for (const Component& c :
                 components(makeStack(n, R), r_p, r_x, dphi, chi, tol)) {
                out.append(py::make_tuple(c.lo, c.hi));
            }
            return out;
        },
        py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"), py::arg("dphi"),
        py::arg("chi") = -1.0, py::arg("tol") = 1e-13,
        "Property reachpieces: the tilt components that can reach this sweep.");

    sub.def(
        "zeta_at",
        [](double c_z, const std::vector<double>& n, const std::vector<double>& R,
           double r_p, double r_x, double dphi) -> py::object {
            const auto v = zetaAt(c_z, makeStack(n, R), r_p, r_x, dphi);
            return v ? py::cast(*v) : py::none();
        },
        py::arg("c_z"), py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"),
        py::arg("dphi"), "zeta(C_z) on the near branch.");

    sub.def(
        "classify_near",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_p,
           double r_x, double dphi, double dz) -> py::object {
            const auto v = classifyNear(makeStack(n, R), r_p, r_x, dphi, dz);
            if (!v) {
                return py::none();
            }
            return py::make_tuple(v->lambda, v->c_z);
        },
        py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"), py::arg("dphi"),
        py::arg("dz"), "Step (3), near branch: a seed (Lambda, C_z) or None.");

    sub.def(
        "classify_far",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_p,
           double r_x, double dphi, double dz, double tol, int maxboxes) {
            FarSearchStats stats;
            py::list seeds;
            for (const Seed& sd : classifyFar(makeStack(n, R), r_p, r_x, dphi, dz,
                                              tol, &stats, maxboxes)) {
                seeds.append(py::make_tuple(sd.lambda, sd.c_z));
            }
            py::dict st;
            st["boxes"] = stats.boxes;
            st["precise"] = stats.precise;
            st["undecided"] = stats.undecided;
            st["from_undecided"] = stats.from_undecided;
            st["capped"] = stats.capped;
            return py::make_tuple(seeds, st);
        },
        py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"), py::arg("dphi"),
        py::arg("dz"), py::arg("tol") = 1e-13, py::arg("maxboxes") = 200000,
        "Steps (3)-(4), far branch: ((Lambda, C_z) seeds, stats). Certified by "
        "Krawczyk's test; nothing is sampled. stats['capped'] is True when the "
        "search reached maxboxes, and the result is then not certified.");

    sub.def(
        "solve_far",
        [](const std::vector<double>& n, const std::vector<double>& R, double r_p,
           double r_x, double dphi, double dz, double lam0, double cz0) {
            const SolveResult v =
                solveFar(makeStack(n, R), r_p, r_x, dphi, dz, lam0, cz0);
            return py::make_tuple(v.ok, v.lambda, v.c_z, v.residual, v.iters);
        },
        py::arg("n"), py::arg("R"), py::arg("r_p"), py::arg("r_x"), py::arg("dphi"),
        py::arg("dz"), py::arg("lam0"), py::arg("cz0"),
        "Step (4), far branch: (ok, Lambda, C_z, residual, iters).");

    sub.def(
        "project_cylinder",
        [](const std::vector<double>& n, const std::vector<double>& R,
           const std::vector<double>& pinhole, const std::vector<double>& point,
           double tol, int itmax) {
            const auto p = toArray3(pinhole, "pinhole");
            const auto x = toArray3(point, "point");
            CylinderCoverage cov;
            py::list rays;
            for (const Ray& r : projectCylinder(makeStack(n, R), p.data(), x.data(),
                                                &cov, tol, itmax)) {
                rays.append(py::make_tuple(r.branch == Branch::Far ? "far" : "near",
                                           r.lambda, r.c_z,
                                           std::vector<double>(r.dir, r.dir + 3)));
            }
            py::dict coverage;
            coverage["near_branch"] = cov.near_branch;
            coverage["far_branch"] = cov.far_branch;
            return py::make_tuple(rays, coverage);
        },
        py::arg("n"), py::arg("R"), py::arg("pinhole"), py::arg("point"),
        py::arg("tol") = 1e-9, py::arg("itmax") = -1,
        "Section cylalg, both branches and the wrapped targets. Returns "
        "(rays, coverage). Lengths are scaled by 2^floor(log2 r_P) inside, so tol is "
        "dimensionless; itmax caps the step-(4) solvers (-1: 80 near, 40 far). "
        "Coverage is false where the model does not apply or the box search "
        "reached its cap.");

    sub.def(
        "direction_from_invariants",
        [](const std::vector<double>& pinhole, double lam, double c_z, double n1,
           double nu1) {
            const auto p = toArray3(pinhole, "pinhole");
            double out[3];
            directionFromInvariants(p.data(), lam, c_z, n1, nu1, out);
            return std::vector<double>(out, out + 3);
        },
        py::arg("pinhole"), py::arg("lam"), py::arg("c_z"), py::arg("n1"),
        py::arg("nu1"), "Step (5): the ray direction at the pinhole.");
}
