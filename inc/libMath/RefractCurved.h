#pragma once

// Refraction through coaxial cylindrical or concentric spherical interfaces.
//
// Implements the closed forms and the projection algorithm of
// doc/refract_curved.tex (Sections cylalg and spherealg) in the invariant
// variables of that document.  Deliberately free of the project's Matrix/Pt3D
// types: everything here is plain doubles and std::vector, so it can be unit
// tested on its own and compared term by term with the Python reference
// implementations in test/refs/.  Camera.cpp adapts it to world coordinates.
//
// Index order follows refract_curved.tex, NOT refract_plate.tex: medium 1 holds
// the PINHOLE and medium N holds the OBJECT POINT.  (The plate document numbers
// them the other way round.)  Interfaces satisfy
//
//     R_0 = r_P  >  R_1 > R_2 > ... > R_{N-1}  >  R_N = r_X
//
// with medium k spanning R_k <= r <= R_{k-1}, so `n` has N entries and `R` has
// N-1.  r_P and r_X are radii from the axis (cylinder) or the centre (sphere).
//
// A ray is fixed by at most two numbers, conserved across every interface
// because the normals are radial (Section invariants):
//
//     C_z    = n d_z                  the axial momentum   (cylinder only)
//     Lambda = n r d_phi              the azimuthal momentum
//
// For a sphere the ray lies in the plane through the centre, the pinhole and
// the point, and C_z = 0 there, so Lambda alone fixes it (Section rayplane).

#include <cstddef>
#include <optional>
#include <vector>

namespace refract_curved {

/** @brief Which way the ray reaches r_X: before its closest approach, or after. */
enum class Branch { Near, Far };

/**
  * @brief A stack of coaxial cylinders or concentric spheres.
  *
  * `n[k]` is the index of medium k+1 (0-based), spanning `R[k] <= r <= R[k-1]`.
  * `R` is strictly decreasing and holds N-1 entries for N media.
  */
struct Stack {
    std::vector<double> n; ///< Refractive indices, outermost (pinhole side) first.
    std::vector<double> R; ///< Interface radii, strictly decreasing. Size n.size()-1.

    std::size_t nMedia() const { return n.size(); }
    /** @brief True when sizes agree, indices are positive and radii strictly decrease. */
    bool isValid() const;
};

/** @brief Azimuthal sweep and axial offset accumulated along a ray. */
struct PhiZ {
    double phi = 0.0; ///< Total azimuthal sweep, radians.
    double z = 0.0;   ///< Total axial offset (always 0 when C_z = 0).
};

/** @brief The edge of the admissible set, eq. (adm): Lmax = min_k nu_k R_k. */
struct LmaxKstar {
    double lmax = 0.0; ///< Largest admissible |Lambda| at this tilt.
    int kstar = 0;     ///< 1-based index of the medium attaining it.
};

/** @brief One image of a point: the ray that makes it. */
struct Ray {
    Branch branch = Branch::Near;
    double lambda = 0.0; ///< Signed; the sign is the sense of rotation.
    double c_z = 0.0;    ///< Always 0 for a sphere.
    double dir[3] = {0.0, 0.0, 0.0}; ///< Unit direction at the pinhole, world frame.
    bool is_ring = false; ///< Sphere only: a whole circle of images (Section ring).
};

// --------------------------------------------------------------- closed forms

/** @brief nu_k = sqrt(n_k^2 - C_z^2), the in-plane momentum scale of medium k. */
std::vector<double> nus(const std::vector<double>& n, double c_z);

/** @brief Eq. (adm). `r_x` is R_N. Returns {0,0} if the tilt exceeds every index. */
LmaxKstar lmaxKstar(const Stack& s, double r_x, double c_z);

/**
  * @brief Sweep and axial offset of eqs. (Phinear), (Phifar) and (Z).
  * @return Nullopt when the ray never reaches r_X at this (Lambda, C_z).
  */
std::optional<PhiZ> phiZ(double lam, double c_z, const Stack& s, double r_p,
                         double r_x, Branch b);

/**
  * @brief The radial action W of eq. (W), in the elementary form
  *        int sqrt(nu^2 - L^2/r^2) dr = nu [ sqrt(r^2-b^2) - b acos(b/r) ].
  * @return Nullopt when the ray never reaches r_X.
  */
std::optional<double> radialAction(double lam, double c_z, const Stack& s,
                                   double r_p, double r_x, Branch b);

/** @brief dPhi/dLambda of eq. (dPhi). Positive on the near branch (Property convexnear). */
double dPhiDLam(double lam, double c_z, const Stack& s, double r_p, double r_x,
                Branch b);

/** @brief dPhi/dC_z. Equals dZ/dLambda by Property symmetry, so J is symmetric. */
double dPhiDCz(double lam, double c_z, const Stack& s, double r_p, double r_x,
               Branch b);

/**
  * @brief dZ/dC_z by central differences, matching the reference implementation.
  * @return Nullopt when either offset sample leaves the domain.
  */
std::optional<double> dZDCz(double lam, double c_z, const Stack& s, double r_p,
                            double r_x, Branch b, double h = 1e-7);

/** @brief Eq. (z0): no ray of this tilt covers less axial distance (Property zbound). */
double z0(double c_z, const Stack& s, double r_p, double r_x);

/** @brief True when 0 <= Lambda < Lmax(C_z) and the tilt is below every index. */
bool inDomain(double lam, double c_z, const Stack& s, double r_x);

// ------------------------------------------------------------- the near branch

/**
  * @brief Solve Phi_near(Lambda; C_z) = target by bisection on the monotone arc.
  *
  * Phi_near increases strictly in Lambda (Property convexnear), so the root is
  * unique when the target is within reach.
  * @return Nullopt when the target exceeds the reach at this tilt.
  */
std::optional<double> lamAt(double c_z, const Stack& s, double r_p, double r_x,
                            double target, int iters = 90);

/** @brief Newton on Phi(Lambda) = target at fixed tilt, clamped into the domain. */
double polish(double lam, double c_z, const Stack& s, double r_p, double r_x,
              double target, Branch b, int itmax = 30, double tol = 1e-14);

// -------------------------------------------------------------- the far branch

/**
  * @brief Every turning point of Phi_far on (0, Lmax), none missed.
  *
  * With t = Lambda^2, Phi_far' is a sum of terms each increasing in t
  * (eq. intervalbound), so a cell whose end values exclude zero holds no turning
  * point and subdivision brackets every sign change. Nothing is sampled.
  */
std::vector<double> farTurns(double c_z, const Stack& s, double r_p, double r_x,
                             double tol = 1e-14);

/** @brief A root of Phi_far = target, with the slope sign of the arc holding it. */
struct FarRoot {
    double lambda = 0.0;
    int slope = 0; ///< +1 where Phi_far increases through the target, -1 where it falls.
    double z = 0.0;
};

/**
  * @brief Roots of Phi_far(Lambda; C_z) = target, by increasing Lambda.
  *
  * [0, Lmax) is cut at the turning points, and an arc whose end values bracket
  * the target holds exactly one root (Theorem count).
  */
std::vector<FarRoot> farRoots(double c_z, const Stack& s, double r_p, double r_x,
                              double target);

// -------------------------------------------------------------------- spheres

/** @brief The ray plane of Section rayplane, built from centre, pinhole and point. */
struct RayPlane {
    double r_p = 0.0;
    double r_x = 0.0;
    double dphi = 0.0;     ///< In [0, pi].
    double e1[3] = {0.0, 0.0, 0.0}; ///< Centre towards pinhole, unit.
    double e2[3] = {0.0, 0.0, 0.0}; ///< In plane, towards the point, unit.
    bool axial = false;    ///< Centre, pinhole and point collinear: e2 undetermined.
};

/** @brief Section rayplane. `centre`, `pinhole`, `point` are world 3-vectors. */
std::optional<RayPlane> rayPlane(const double centre[3], const double pinhole[3],
                                 const double point[3]);

/** @brief (Phi_near(Lmax), M(0)) of step (2): the near reach and the largest sweep. */
struct Sweeps {
    double near_reach = 0.0;
    double m0 = 0.0;
};
Sweeps sweeps(const Stack& s, double r_p, double r_x);

/** @brief Property chord: no ray sweeps more than sum arccos(R_k/R_{k-1}) + pi. */
double chordBound(const Stack& s, double r_p);

/**
  * @brief Every image of `point` through concentric spheres (Section spherealg).
  *
  * Returns the near-branch ray when the target is within reach, every
  * far-branch root on its own monotone arc, and the same again for each wrapped
  * target that M(0) allows. On the axial line the answer is Section ring's: the
  * radial or central ray, plus a ring when eq. (ring) holds.
  */
std::vector<Ray> projectSphere(const Stack& s, const double centre[3],
                               const double pinhole[3], const double point[3],
                               double tol = 1e-9);

// ------------------------------------------------------------------ cylinders

/** @brief Tilts in (0, chi) where two caps nu_k R_k of eq. (adm) cross, so k* can switch. */
std::vector<double> switchTilts(const Stack& s, double r_x, double chi);

/** @brief The reach Phi_near(Lmax(C_z); C_z) medium by medium, each term monotone while k* holds. */
std::vector<double> reachTerms(double c_z, const Stack& s, double r_p, double r_x);

/** @brief One connected piece of the set of tilts that can reach a given sweep. */
struct Component {
    double lo = 0.0;
    double hi = 0.0;
};

/**
  * @brief Property reachpieces: the components of {C_z : reach(C_z) >= dphi}, found exactly.
  *
  * The range is cut where k* can switch; on each piece every term of the reach is
  * monotone in C_z, so end values bound it and subdivision decides each interval.
  * Nothing is sampled, so no component is lost.
  */
std::vector<Component> components(const Stack& s, double r_p, double r_x,
                                  double dphi, double chi = -1.0,
                                  double tol = 1e-13);

/** @brief zeta(C_z) = Z(Lambda(C_z), C_z) on the near branch; nullopt out of reach. */
std::optional<double> zetaAt(double c_z, const Stack& s, double r_p, double r_x,
                             double dphi);

/**
  * @brief Step (3), near branch: a seed (Lambda, C_z) for (dphi, dz), or nullopt.
  *
  * zeta is strictly increasing on each component (Property zeta), so its values
  * just inside the two ends decide whether the target is attained there, and one
  * bisection locates it.
  */
struct Seed {
    double lambda = 0.0;
    double c_z = 0.0;
};
std::optional<Seed> classifyNear(const Stack& s, double r_p, double r_x,
                                 double dphi, double dz, int iters = 60);

/** @brief What the far-branch box search did, for diagnostics and tests. */
struct FarSearchStats {
    int boxes = 0;          ///< Boxes examined.
    int undecided = 0;      ///< Boxes still undecided at the width limit.
    int from_undecided = 0; ///< Rays recovered by Newton from those.
};

/**
  * @brief Steps (3) and (4), far branch: every (Lambda, C_z) reaching the target.
  *
  * Boxes in (theta*, C_z), with Lambda = Lmax(C_z) sin(theta*), cover
  * [0, pi/2] x [0, C_max] between the cuts of eq. (cuts); no steeper ray reaches
  * the point (Property zbound). A box is dropped only when the bounds of
  * eq. (farbox) exclude the target, or when Krawczyk's test on the box grown by
  * 10% shows it holds no ray; it yields a ray when the test shows it holds
  * exactly one. Otherwise it is halved, and a box still undecided at `tol` is
  * kept and polished by Newton. **Nothing is sampled**, so no ray is lost.
  *
  * Deviation from the reference implementation: where the reference recomputes
  * the residual in 34-digit arithmetic to sharpen Krawczyk's test next to a
  * caustic, this uses double precision throughout. That only ever makes the test
  * more conservative -- a box that would have been certified becomes undecided
  * and is resolved by Newton instead -- so it cannot drop a box that holds a ray.
  */
std::vector<Seed> classifyFar(const Stack& s, double r_p, double r_x, double dphi,
                              double dz, double tol = 1e-13,
                              FarSearchStats* stats = nullptr,
                              int maxboxes = 200000);

/**
  * @brief Step (4), far branch: Newton on eq. (system) from a step-(3) seed.
  *
  * There is no convex program on this branch, so the step is backtracked on the
  * residual itself while staying inside the domain.
  */
struct SolveResult;
SolveResult solveFar(const Stack& s, double r_p, double r_x, double dphi, double dz,
                     double lam0, double cz0, int itmax = 40, double tol = 1e-10);

/** @brief Result of the two-dimensional Newton solve on eq. (system). */
struct SolveResult {
    bool ok = false;
    double lambda = 0.0;
    double c_z = 0.0;
    double residual = 0.0;
    int iters = 0;
};

/**
  * @brief Damped Newton on G = -W - Lambda dphi - C_z dz, whose gradient is
  *        (Phi - dphi, Z - dz) (Property consequences(d)).
  *
  * On the near branch G is strictly convex, so this is a convex program and the
  * solution is unique. Steps are backtracked to stay inside the domain.
  */
SolveResult solveCylinder(const Stack& s, double r_p, double r_x, double dphi,
                          double dz, double lam0, double cz0, Branch b,
                          int itmax = 80, double tol = 1e-10);

/** @brief Rebuild the ray direction at the pinhole from its invariants (step 5). */
void directionFromInvariants(const double pinhole[3], double lam, double c_z,
                             double n1, double nu1, double out[3]);

/** @brief What `projectCylinder` searched, so a caller can tell a real
  *        "no image" from "not looked for". Both are true now.
  */
struct CylinderCoverage {
    bool near_branch = false;
    bool far_branch = false;
};

/**
  * @brief Every image of `point` through coaxial cylinders: steps (1)-(5) of
  *        Section cylalg, both branches, including rays the long way round.
  *
  * Near branch: its components come from Property reachpieces and zeta is
  * strictly increasing on each, so one bisection per component finds the ray.
  * Far branch: the certified box search of `classifyFar`. Beside Delta phi the
  * targets 2 pi - Delta phi, Delta phi + 2 pi, ... are solved while they stay
  * below the chord bound of Property chord.
  *
  * A point opposite the pinhole, Delta phi = pi, is reached by the radial ray of
  * the far branch, which sweeps pi at every tilt; its root sits on the edge
  * Lambda = 0 for both senses at once, so it is recorded once here and every
  * other ray is returned in both senses.
  *
  * The axis is aligned with z and passes through the origin; callers transform
  * into that frame first.
  */
std::vector<Ray> projectCylinder(const Stack& s, const double pinhole[3],
                                 const double point[3],
                                 CylinderCoverage* coverage = nullptr,
                                 double tol = 1e-9);

} // namespace refract_curved
