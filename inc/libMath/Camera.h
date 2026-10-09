#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "../error.hpp"
#include "Matrix.h"
#include "STBCommons.h"

/**
  * @brief Parameters for a standard pinhole camera model.
  */
struct PinholeParam {
    int n_row = 0; // Image height in pixels.
    int n_col = 0; // Image width in pixels.
    Matrix<double> cam_mtx; // Intrinsic matrix K.
    bool is_distorted = false; // True if distortion is enabled.
    int n_dist_coeff = 0; // Number of distortion coefficients (4/5/8/12).
    std::vector<double> dist_coeff; // OpenCV-compatible distortion coefficients.
    Matrix<double> r_mtx; // Rotation matrix: world -> camera.
    Pt3D t_vec; // Translation vector: world -> camera.
    Matrix<double> r_mtx_inv; // Inverse rotation: camera -> world.
    Pt3D t_vec_inv; // Camera center in world coordinates.
};

enum RefPlane {
    REF_X = 1,
    REF_Y,
    REF_Z
};

/**
  * @brief Parameters for the polynomial camera model.
  */
struct PolyParam {
    int n_row = 0; // Image height in pixels.
    int n_col = 0; // Image width in pixels.
    RefPlane ref_plane = REF_Z; // Two depth planes are defined along this axis.
    std::vector<double> plane = {0, 0}; // Two reference plane coordinates.
    int n_coeff = 0; // Number of rows in polynomial coefficient matrices.
    Matrix<double> u_coeffs; // u projection polynomial coefficients.
    Matrix<double> du_coeffs; // Partial derivatives of u wrt the two solved axes.
    Matrix<double> v_coeffs; // v projection polynomial coefficients.
    Matrix<double> dv_coeffs; // Partial derivatives of v wrt the two solved axes.
};

struct PinPlateParam : public PinholeParam {
    Plane3D plane; // Farthest refractive interface plane.

    // Refraction ordering is always FROM FARTHEST to NEAREST to the camera.
    // - refract_array size = n_plate + 2 (media interfaces)
    // - w_array       size = n_plate     (plate thicknesses)
    std::vector<double> refract_array;
    std::vector<double> w_array;
    int n_plate = 0; // Number of physical plates.
    double proj_tol = 0.0; // Projection solver tolerance.
    int proj_nmax = 0; // Projection solver max iterations.
    double lr = 0.0; // Projection solver update scale.
    double refract_ratio_max = 0.0; // Max refractive ratio n0/ni for radius checks.

    // Refractive interfaces ordered FROM FARTHEST to NEAREST.
    std::vector<Plane3D> plane_array;

    // 2D basis on the farthest reference plane.
    Pt3D u_axis, v_axis;
};

/**
  * @brief Parameters shared by the curved refractive ports.
  *
  * Index order follows doc/refract_curved.tex: `refract_array[0]` is the medium
  * holding the PINHOLE and the last entry the medium holding the object point,
  * which is the opposite of PinPlateParam's farthest-to-nearest ordering.
  * `radius_array` holds the N-1 interface radii, strictly decreasing.
  */
struct PinCurvedParam : public PinholeParam {
    Pt3D center; // Point on the axis (cylinder) or the centre (sphere).
    Pt3D axis;   // Unit axis direction; ignored by the sphere.
    std::vector<double> refract_array; // n_1..n_N, pinhole side first.
    std::vector<double> radius_array;  // R_1..R_{N-1}, strictly decreasing.
    // Tolerance on each ray's residual hypot(Phi - dphi, (Z - dz) / u), with u
    // the power of two in (r_P/2, r_P] and r_P the pinhole's distance from the
    // axis or centre: dimensionless, so the same in any length unit. The solvers
    // converge to min(1e-10, proj_tol).
    double proj_tol = 1e-9;
    int proj_nmax = 80; // Iteration cap of each Newton solve of the projection.
};

struct PinCylinderParam : public PinCurvedParam {};
struct PinSphereParam : public PinCurvedParam {};

enum class CameraType {
    Pinhole,
    Polynomial,
    RefractionPinhole,
    RefractionCylinder,
    RefractionSphere,
};

/**
  * @brief One image of a world point, for models whose projection is multi-valued.
  *
  * A curved port can send several rays from the pinhole to the same point, so
  * `projectAll()` returns a list of these rather than a single Pt2D. `branch`
  * and `sense` identify which ray made the image, and `is_ring` marks the
  * spherical degeneracy where the image is a whole circle, not a point.
  */
struct ImageRay {
    Pt2D pt_img_dist;      // Distorted image point in pixels; NaN for a ring.
    bool is_far = false;   // True on the far branch (past the closest approach).
    double lambda = 0.0;   // Azimuthal invariant; its sign is the sense of rotation.
    double c_z = 0.0;      // Axial invariant; always 0 for a sphere.
    bool is_ring = false;  // Sphere only: a circle of images (Section ring).
};

/**
  * @brief What a multi-valued projection actually searched.
  *
  * Without this an empty image list is ambiguous between "the point is invisible"
  * and "that branch was not looked for".
  */
struct ProjectionCoverage {
    bool near_branch = true;
    bool far_branch = true;
};

/**
  * @brief How a point appears in one camera (the rule of the cylinder STB plan,
  *        Phase 2): no image on the sensor, exactly one, or more than one (or
  *        one with another image close enough to merge with it).
  */
enum class ImageStatus { None, One, Ambiguous };

/**
  * @brief The ray of a point's last image, to start the projection of the
  *        point after a small move (tracking).
  *
  * Used only by models whose projection is solved iteratively (the cylindrical
  * port, see Camera::usesProjectionHint); the others ignore it.
  */
struct ProjectionHint {
    double lambda = 0.0;  // The ray's azimuthal invariant, signed, as ImageRay::lambda.
    double c_z = 0.0;     // Its axial invariant.
    bool is_far = false;  // Its branch.
    bool valid = false;   // False: no ray yet; the projection runs the full search.
};

class Camera {
public:
    virtual ~Camera() = default;

    /** @brief Return camera model type. */
    virtual CameraType type() const = 0;
    /**
      * @brief Project a world point to image space.
      * @param pt_world World point in physical units.
      * @param is_print_detail Enable debug print for iterative models.
      * @return Distorted image point on success.
      */
    virtual StatusOr<Pt2D> project(const Pt3D& pt_world,
                                                                  bool is_print_detail = false) const = 0;
    /**
      * @brief Compute line-of-sight in world coordinates from a distorted image point.
      * @param pt_img_dist Distorted image point in pixels.
      * @return World-space ray as a line.
      */
    virtual StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const = 0;

    /**
      * @brief Every image of a world point, for models whose projection is
      *        multi-valued.
      *
      * Single-valued models inherit this default, which wraps `project()` in a
      * one-element list, so callers can use it uniformly. Curved refractive
      * ports override it; `project()` then returns only the primary image.
      * @param pt_world World point in physical units.
      * @param coverage Optional: filled with which branches were searched.
      * @return All images on success; an empty list means no image exists
      *         within whatever `coverage` reports as searched.
      */
    virtual StatusOr<std::vector<ImageRay>> projectAll(
        const Pt3D& pt_world, ProjectionCoverage* coverage = nullptr) const;

    /**
      * @brief project(), started from the ray of a nearby point's image.
      *
      * Returns the image project() returns. Models that solve no equations
      * ignore the hint and call project(). The cylindrical port solves from the
      * hint when it can prove the point has a single image, and otherwise runs
      * the full search. Either way `hint` is left holding the ray of the image
      * returned, for the next nearby point, or is invalidated when there is
      * none (models without hints leave it untouched).
      */
    virtual StatusOr<Pt2D> projectWithHint(const Pt3D& pt_world,
                                           ProjectionHint& hint) const {
        (void)hint;
        return project(pt_world);
    }
    /** @brief True when projectWithHint() uses the hint, so that callers keep
      *        one only for such cameras. */
    virtual bool usesProjectionHint() const { return false; }
    /**
      * @brief A hint for `pt_world` from a line of sight of this camera that
      *        passes near it, e.g. that of the detection the point was
      *        triangulated from. False, with `hint` untouched, for models that
      *        take no hint.
      */
    virtual bool hintFromLineOfSight(const Line3D& los, const Pt3D& pt_world,
                                     ProjectionHint& hint) const {
        (void)los;
        (void)pt_world;
        (void)hint;
        return false;
    }
    /**
      * @brief True when the image of a straight line in object space is
      *        curved enough to be drawn as a polyline (stereo matching's
      *        image lines). False for the pinhole and the flat window, whose
      *        straight lines are kept.
      */
    virtual bool lineImagesCurve() const { return false; }
    /**
      * @brief True for models whose projection can give a point no image or
      *        several (the curved ports). The per-camera image statuses of
      *        tracking apply to these only; the others behave as before.
      */
    virtual bool multiValued() const { return false; }
    /** @brief True when the model proves `pt_world` has exactly one image
      *        (single-valued models: always). */
    virtual bool provablySingleImage(const Pt3D& pt_world) const {
        (void)pt_world;
        return true;
    }
    /**
      * @brief The point's status: None when no image lies on the sensor
      *        enlarged by `margin_px` on every side; One when exactly one does
      *        and no other image lies within `merge_px` of it; Ambiguous
      *        otherwise. `pt`, when given, receives the one image. This default
      *        (single-valued models) is project() and the enlarged sensor.
      */
    virtual ImageStatus imageStatus(const Pt3D& pt_world, double margin_px,
                                    double merge_px, Pt2D* pt = nullptr) const;
    /**
      * @brief Restricts [t_in, t_out] along `los` (los.pt + t los.unit_vector,
      *        another camera's line of sight) to where this model's projection
      *        applies. False when nothing is left. Models that apply
      *        everywhere only check t_in <= t_out.
      */
    virtual bool clipToModel(const Line3D& los, double& t_in, double& t_out) const {
        (void)los;
        return t_in <= t_out;
    }

    /** @brief Image height in pixels. */
    virtual int getNRow() const = 0;
    /** @brief Image width in pixels. */
    virtual int getNCol() const = 0;

    /**
      * @brief Save camera parameters in OpenLPT text format.
      * @param file_name Output file path.
      */
    virtual Status saveParameters(const std::string& file_name) const = 0;
    /** @brief Deep clone camera model object. */
    virtual std::shared_ptr<Camera> clone() const = 0;

    bool is_active = true;
    double max_intensity = 255.0;

protected:
    /**
      * @brief Project world point to normalized pinhole image coordinates.
      * @param pt_world World point.
      * @param param Pinhole-compatible camera parameters.
      * @return Undistorted normalized image point.
      */
    static Pt2D worldToUndistImg(const Pt3D& pt_world, const PinholeParam& param);
    /**
      * @brief Apply distortion and intrinsics to normalized image point.
      * @param pt_img_undist Undistorted normalized image point.
      * @param param Pinhole-compatible camera parameters.
      * @return Distorted image point in pixel coordinates.
      */
    static Pt2D distort(const Pt2D& pt_img_undist, const PinholeParam& param);
    /**
      * @brief Remove distortion from pixel coordinate to normalized image point.
      * @param pt_img_dist Distorted image point in pixel coordinates.
      * @param param Pinhole-compatible camera parameters.
      * @return Undistorted normalized image point.
      */
    static Pt2D undistort(const Pt2D& pt_img_dist, const PinholeParam& param);
};

class CameraFactory {
public:
    /**
      * @brief Load one camera model from a parameter text file.
      * @param file_name Input file path.
      * @return Concrete camera model on success.
      */
    static StatusOr<std::shared_ptr<Camera>> loadFromFile(const std::string& file_name);
};


class PinholeCamera final : public Camera {
public:
    PinholeCamera() = default;

    /** @brief Return pinhole camera model type. */
    CameraType type() const override { return CameraType::Pinhole; }
    /** @brief Project one world point to distorted image coordinate. */
    StatusOr<Pt2D> project(const Pt3D& pt_world,
                           bool is_print_detail = false) const override;
    /** @brief Back-project one distorted image point to world-space line. */
    StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const override;
    /** @brief Image height in pixels. */
    int getNRow() const override;
    /** @brief Image width in pixels. */
    int getNCol() const override;
    /** @brief Save pinhole parameters to text file. */
    Status saveParameters(const std::string& file_name) const override;
    /** @brief Deep clone pinhole camera object. */
    std::shared_ptr<Camera> clone() const override;

    /**
      * @brief Set image size in pixels.
      * @param n_row Image height.
      * @param n_col Image width.
      * @return OK on success.
      */
    Status setImageSize(int n_row, int n_col);
    /**
      * @brief Set pinhole intrinsics and distortion coefficients.
      * @param fx,fy Focal lengths in pixels.
      * @param cx,cy Principal point in pixels.
      * @param dist_coeff Distortion coefficient array.
      * @return OK on success.
      */
    Status setIntrinsics(double fx, double fy, double cx, double cy,
                                              const std::vector<double>& dist_coeff);
    /**
      * @brief Set extrinsics from Rodrigues rvec and translation tvec.
      * @param rvec Rodrigues rotation vector.
      * @param tvec Translation vector.
      * @return OK on success.
      */
    Status setExtrinsics(const Pt3D& rvec, const Pt3D& tvec);
    /** @brief Finalize derived fields after setters. @return OK on success. */
    Status commitUpdate();

    const PinholeParam& param() const { return _param; }
    PinholeParam& param() { return _param; }

private:
    PinholeParam _param;
};

class RefractionPinholeCamera final : public Camera {
public:
    RefractionPinholeCamera() = default;

    /** @brief Return refraction-pinhole camera model type. */
    CameraType type() const override { return CameraType::RefractionPinhole; }
    /** @brief Project one world point with refractive correction to image coordinate. */
    StatusOr<Pt2D> project(const Pt3D& pt_world,
                           bool is_print_detail = false) const override;
    /** @brief Back-project one distorted image point through refractive stack. */
    StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const override;
    /** @brief Image height in pixels. */
    int getNRow() const override;
    /** @brief Image width in pixels. */
    int getNCol() const override;
    /** @brief Save pinplate parameters to text file. */
    Status saveParameters(const std::string& file_name) const override;
    /** @brief Deep clone refraction-pinhole camera object. */
    std::shared_ptr<Camera> clone() const override;

    /**
      * @brief Set image size in pixels.
      * @param n_row Image height.
      * @param n_col Image width.
      * @return OK on success.
      */
    Status setImageSize(int n_row, int n_col);
    /**
      * @brief Set pinhole intrinsics and distortion coefficients.
      * @param fx,fy Focal lengths in pixels.
      * @param cx,cy Principal point in pixels.
      * @param dist_coeff Distortion coefficient array.
      * @return OK on success.
      */
    Status setIntrinsics(double fx, double fy, double cx, double cy,
                                              const std::vector<double>& dist_coeff);
    /**
      * @brief Set extrinsics from Rodrigues rvec and translation tvec.
      * @param rvec Rodrigues rotation vector.
      * @param tvec Translation vector.
      * @return OK on success.
      */
    Status setExtrinsics(const Pt3D& rvec, const Pt3D& tvec);
    /**
      * @brief Configure refractive stack.
      * @param plane_pt Point on the farthest refractive interface.
      * @param plane_n Interface normal (pointing away from camera).
      * @param refract_array Refractive indices from farthest medium to nearest.
      * @param w_array Plate widths from farthest plate to nearest.
      */
    Status setRefraction(const Pt3D& plane_pt, const Pt3D& plane_n,
                                              const std::vector<double>& refract_array,
                                              const std::vector<double>& w_array);
    /**
      * @brief Configure projection solver options.
      * @param proj_tol Residual tolerance.
      * @param proj_nmax Maximum solver iterations.
      * @param lr Solver update scale.
      * @return OK on success.
      */
    Status setSolverOptions(double proj_tol, int proj_nmax, double lr);
    /** @brief Finalize derived refractive geometry buffers. @return OK on success. */
    Status commitUpdate();

    const PinPlateParam& param() const { return _param; }
    PinPlateParam& param() { return _param; }

private:
    /**
      * @brief Solve refractive projection with LM + line-search.
      * @param pt_world Input world point.
      * @param pin Refractive parameter container.
      * @return (failed, refracted_point, residual, iterations).
      */
    static std::tuple<bool, Pt3D, double, int>
    solveProjectionByRefractionLM(const Pt3D& pt_world, const PinPlateParam& pin);
    /**
      * @brief Solve refractive projection using 1D Snell-length optimization.
      * @param pt_world Input world point.
      * @param pin Refractive parameter container.
      * @return (failed, refracted_point, residual, iterations).
      */
    static std::tuple<bool, Pt3D, double, int>
    solveProjectionBySnell1D(const Pt3D& pt_world, const PinPlateParam& pin);

    PinPlateParam _param;
};

/**
  * @brief Shared behaviour of the cylindrical and spherical refractive ports.
  *
  * Both reduce to the same closed forms in the invariant variables of
  * doc/refract_curved.tex (see RefractCurved.h); they differ only in how world
  * coordinates reduce to the port frame and in which projector is called. The
  * projection is multi-valued, so `projectAll()` is the real interface and
  * `project()` returns the primary image.
  */
class RefractionCurvedCamera : public Camera {
public:
    /**
      * @brief The primary image of a point.
      *
      * Among the images on the sensor (0 <= u < n_col, 0 <= v < n_row), the
      * near-branch one, else the one of smallest |Lambda|; when none is on the
      * sensor, the same rule over every image, so the pixel may lie off the
      * sensor. Rings carry no pixel and are skipped. A GeometryFailure error when
      * the point has no image, or only a ring.
      */
    StatusOr<Pt2D> project(const Pt3D& pt_world,
                           bool is_print_detail = false) const override;
    /**
      * @brief Every image of a point.
      *
      * An InvalidArgument error when the pinhole is not outside the outermost
      * interface. No image, with coverage false, when the point is not strictly
      * inside the innermost one (outside the model), or when the cylinder's box
      * search reached its cap. Rays leaving the pinhole away from the viewing
      * direction form no image and are dropped.
      */
    StatusOr<std::vector<ImageRay>> projectAll(
        const Pt3D& pt_world, ProjectionCoverage* coverage = nullptr) const override;
    /** @brief Back-project by tracing the pixel ray through the quadrics; single-valued.
      *        An InvalidArgument error when the pinhole is not outside the outermost interface. */
    StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const override;
    /**
      * @brief project() for a point that moved little since its last image.
      *
      * Cylinder: refract_curved::projectCylinderFrom() from the hint's ray,
      * when the hint is valid and the point provably has one image; otherwise
      * projectAll() and project()'s rule. The hint is then set to the ray of
      * the image returned, or invalidated when there is none. Sphere: project(),
      * the hint untouched.
      */
    StatusOr<Pt2D> projectWithHint(const Pt3D& pt_world,
                                   ProjectionHint& hint) const override;
    /** @brief True for the cylinder, unless the environment variable
      *        OPENLPT_PROJECTION_HINTS is "0" (to compare a run with the full
      *        search alone). */
    bool usesProjectionHint() const override;
    /** @brief The invariants of the line of sight (a segment in the innermost
      *        medium, as lineOfSight() returns), and the branch on which
      *        `pt_world`'s foot on it lies: far beyond the ray's closest
      *        approach to the axis. Cylinder only. */
    bool hintFromLineOfSight(const Line3D& los, const Pt3D& pt_world,
                             ProjectionHint& hint) const override;
    /** @brief True, unless the environment variable OPENLPT_CURVED_LINES is
      *        "0" (straight lines, as before, for comparison). */
    bool lineImagesCurve() const override;
    /** @brief Inside the innermost interface: the object medium, where every
      *        projected point must lie (with a relative margin of 1e-9). */
    bool clipToModel(const Line3D& los, double& t_in, double& t_out) const override;
    bool multiValued() const override { return true; }
    /** @brief Cylinder: refract_curved::singleImage(); sphere: false. */
    bool provablySingleImage(const Pt3D& pt_world) const override;
    /** @brief project() when provablySingleImage(); otherwise the census of
      *        projectAll(). A ring, or an image that project() would not
      *        return, makes the status Ambiguous. */
    ImageStatus imageStatus(const Pt3D& pt_world, double margin_px,
                            double merge_px, Pt2D* pt = nullptr) const override;
    int getNRow() const override;
    int getNCol() const override;

    Status setImageSize(int n_row, int n_col);
    Status setIntrinsics(double fx, double fy, double cx, double cy,
                         const std::vector<double>& dist_coeff);
    Status setExtrinsics(const Pt3D& rvec, const Pt3D& tvec);
    /**
      * @brief Configure the port.
      * @param center Point on the axis (cylinder) or the centre (sphere).
      * @param axis Axis direction; ignored by the sphere, normalised here.
      * @param refract_array n_1..n_N, the pinhole's medium first.
      * @param radius_array R_1..R_{N-1}, strictly decreasing.
      */
    Status setRefraction(const Pt3D& center, const Pt3D& axis,
                         const std::vector<double>& refract_array,
                         const std::vector<double>& radius_array);
    Status setSolverOptions(double proj_tol, int proj_nmax);
    Status commitUpdate();

    const PinCurvedParam& param() const { return _param; }
    PinCurvedParam& param() { return _param; }

protected:
    /** @brief True when this port is a sphere, so the axis is unused. */
    virtual bool isSphere() const = 0;
    /** @brief Token written by saveParameters and matched by the loader. */
    virtual const char* typeToken() const = 0;
    /** @brief Common body of saveParameters, given the type token. */
    Status saveCurvedParameters(const std::string& file_name) const;

    PinCurvedParam _param;
};

class RefractionCylinderCamera final : public RefractionCurvedCamera {
public:
    RefractionCylinderCamera() = default;
    CameraType type() const override { return CameraType::RefractionCylinder; }
    Status saveParameters(const std::string& file_name) const override;
    std::shared_ptr<Camera> clone() const override;

protected:
    bool isSphere() const override { return false; }
    const char* typeToken() const override { return "PINCYL"; }
};

class RefractionSphereCamera final : public RefractionCurvedCamera {
public:
    RefractionSphereCamera() = default;
    CameraType type() const override { return CameraType::RefractionSphere; }
    Status saveParameters(const std::string& file_name) const override;
    std::shared_ptr<Camera> clone() const override;

protected:
    bool isSphere() const override { return true; }
    const char* typeToken() const override { return "PINSPH"; }
};

class PolynomialCamera final : public Camera {
public:
    PolynomialCamera() = default;

    /** @brief Return polynomial camera model type. */
    CameraType type() const override { return CameraType::Polynomial; }
    /** @brief Project one world point to image coordinate using polynomial model. */
    StatusOr<Pt2D> project(const Pt3D& pt_world,
                                                  bool is_print_detail = false) const override;
    /** @brief Back-project one image point to world-space line using two reference planes. */
    StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const override;
    /** @brief Image height in pixels. */
    int getNRow() const override;
    /** @brief Image width in pixels. */
    int getNCol() const override;
    /** @brief Save polynomial parameters to text file. */
    Status saveParameters(const std::string& file_name) const override;
    /** @brief Deep clone polynomial camera object. */
    std::shared_ptr<Camera> clone() const override;

    /**
      * @brief Set image size in pixels.
      * @param n_row Image height.
      * @param n_col Image width.
      * @return OK on success.
      */
    Status setImageSize(int n_row, int n_col);
    /**
      * @brief Set two reference planes used for polynomial inversion.
      * @param ref_plane Axis along which two planes are specified.
      * @param p0,p1 Coordinates of the two reference planes.
      * @return OK on success.
      */
    Status setReferencePlane(RefPlane ref_plane, double p0, double p1);
    /**
      * @brief Set polynomial coefficients for u and v projections.
      * @param u_coeffs U polynomial table.
      * @param v_coeffs V polynomial table.
      * @return OK on success.
      */
    Status setCoefficients(const Matrix<double>& u_coeffs,
                                                  const Matrix<double>& v_coeffs);
    /** @brief Finalize derivative coefficient buffers. @return OK on success. */
    Status commitUpdate();

    const PolyParam& param() const { return _param; }
    PolyParam& param() { return _param; }

private:
    /**
      * @brief Build derivative coefficient buffers for polynomial inversion.
      * @param poly Polynomial parameter container.
      */
    static void buildPolyDerivatives(PolyParam& poly);
    /**
      * @brief Solve one world point on selected reference plane from image point.
      * @param pt_img_dist Input image coordinate.
      * @param plane_world Target plane coordinate.
      * @param poly Polynomial parameter container.
      * @return Reconstructed world point on requested plane.
      */
    static Pt3D solveWorldOnRefPlane(const Pt2D& pt_img_dist, double plane_world,
                                     const PolyParam& poly);

    PolyParam _param;
};

/**
  * @brief A smooth correction of a camera's image positions, in pixels (VSC's
  *        correction field, fitted from tracks between runs).
  *
  * A detection at pixel d has the image point d - c(d) in the camera model the
  * correction belongs to: c says where the camera's images sit relative to
  * where the model puts them. c = (du, dv) is held at the nodes of a regular
  * grid, read bilinearly between them and as the nearest edge value beyond
  * the grid. Read-only once built, so clones of a camera share one.
  *
  * The file is text; '#' starts a comment:
  *   VERSION 1
  *   IMAGE_SIZE n_col n_row     the camera's image size, checked when wrapped
  *   GRID u0 v0 step nu nv      node (i, j) at pixel (u0 + i step, v0 + j step)
  *   DU                         then nv rows of nu values (v outer)
  *   DV                         the same for dv
  */
class ImageCorrection {
public:
    /** @brief Read a correction file. An IOfailure error when it cannot be
      *        opened; InvalidArgument when its layout or values are wrong. */
    static StatusOr<std::shared_ptr<const ImageCorrection>> loadFromFile(
        const std::string& file_name);
    /**
      * @brief The same from values in memory, du and dv row by row (v outer).
      *        Checks: positive sizes, at least 2 x 2 finite nodes, a positive
      *        step, and a slope bound below 1/2, which makes toImage() a
      *        contraction. `source` names the correction in messages.
      */
    static StatusOr<std::shared_ptr<const ImageCorrection>> fromGrid(
        int n_col, int n_row, double grid_u0, double grid_v0, double grid_step,
        int grid_nu, int grid_nv,
        const std::vector<double>& du, const std::vector<double>& dv,
        const std::string& source = "");

    /** @brief c at pixel `pt`; NaN for a non-finite pixel. */
    Pt2D at(const Pt2D& pt) const;
    /** @brief The model's image point of a detection at `pt_img`: pt_img - c(pt_img).
      *        A non-finite pixel is returned unchanged. */
    Pt2D toModel(const Pt2D& pt_img) const;
    /** @brief The pixel d whose model point is `pt_model` (d - c(d) = pt_model),
      *        by the fixed point d <- pt_model + c(d) to 1e-11 px. A non-finite
      *        point is returned unchanged. */
    Pt2D toImage(const Pt2D& pt_model) const;

    int nCol() const { return _n_col; }
    int nRow() const { return _n_row; }
    double u0() const { return _u0; }
    double v0() const { return _v0; }
    double step() const { return _step; }
    int nu() const { return _nu; }
    int nv() const { return _nv; }
    const std::vector<double>& du() const { return _du; }
    const std::vector<double>& dv() const { return _dv; }
    /** @brief A bound on the norm of c's derivative: the fixed point of
      *        toImage() shrinks its error at least this much per step. */
    double slopeBound() const { return _slope; }
    /** @brief The file it was read from, or fromGrid()'s `source`. */
    const std::string& source() const { return _source; }

private:
    ImageCorrection() = default;
    void eval(double u, double v, double& cu, double& cv) const;

    int _n_col = 0, _n_row = 0;
    double _u0 = 0.0, _v0 = 0.0, _step = 1.0;
    int _nu = 0, _nv = 0;
    std::vector<double> _du, _dv;
    double _slope = 0.0;
    std::string _source;
};

/**
  * @brief A camera model with its image positions corrected by an
  *        ImageCorrection.
  *
  * A projection moves the model's image point q to the pixel d with
  * d - c(d) = q, and a pixel's line of sight is the model's line of sight of
  * d - c(d). Built only when a camera of config.json has a "correction"; other
  * cameras are not wrapped, so they behave exactly as before.
  *
  * Which image of a multi-valued model project() returns is the inner model's
  * choice, made on its own (uncorrected) positions, and hints are the inner
  * model's rays; the correction then moves every image. type() and the other
  * properties are the inner model's. Code that reads a concrete model's
  * parameters goes through baseCamera().
  */
class CorrectedCamera final : public Camera {
public:
    /** @brief Wrap `inner`. An InvalidArgument error when either is missing or
      *        the correction's image size is not the camera's. */
    static StatusOr<std::shared_ptr<CorrectedCamera>> create(
        std::shared_ptr<Camera> inner, std::shared_ptr<const ImageCorrection> correction);

    const std::shared_ptr<Camera>& inner() const { return _inner; }
    const std::shared_ptr<const ImageCorrection>& correction() const { return _correction; }

    CameraType type() const override { return _inner->type(); }
    StatusOr<Pt2D> project(const Pt3D& pt_world,
                           bool is_print_detail = false) const override;
    StatusOr<Line3D> lineOfSight(const Pt2D& pt_img_dist) const override;
    StatusOr<std::vector<ImageRay>> projectAll(
        const Pt3D& pt_world, ProjectionCoverage* coverage = nullptr) const override;
    StatusOr<Pt2D> projectWithHint(const Pt3D& pt_world,
                                   ProjectionHint& hint) const override;
    bool usesProjectionHint() const override { return _inner->usesProjectionHint(); }
    bool hintFromLineOfSight(const Line3D& los, const Pt3D& pt_world,
                             ProjectionHint& hint) const override {
        return _inner->hintFromLineOfSight(los, pt_world, hint);
    }
    /** @brief The curved ports' own setting; true for the single-valued models,
      *        whose straight line images a varying shift bends. */
    bool lineImagesCurve() const override;
    bool multiValued() const override { return _inner->multiValued(); }
    bool provablySingleImage(const Pt3D& pt_world) const override {
        return _inner->provablySingleImage(pt_world);
    }
    /** @brief The inner model's rule on the corrected images: project() and
      *        the sensor when the point provably has one image, otherwise the
      *        census of the moved images, the primary being the inner model's. */
    ImageStatus imageStatus(const Pt3D& pt_world, double margin_px,
                            double merge_px, Pt2D* pt = nullptr) const override;
    bool clipToModel(const Line3D& los, double& t_in, double& t_out) const override {
        return _inner->clipToModel(los, t_in, t_out);
    }
    int getNRow() const override { return _inner->getNRow(); }
    int getNCol() const override { return _inner->getNCol(); }
    /** @brief The inner model's file; the correction stays in its own file. */
    Status saveParameters(const std::string& file_name) const override;
    /** @brief Clones the inner model and shares the (read-only) correction. */
    std::shared_ptr<Camera> clone() const override;

private:
    CorrectedCamera(std::shared_ptr<Camera> inner,
                    std::shared_ptr<const ImageCorrection> correction);

    std::shared_ptr<Camera> _inner;
    std::shared_ptr<const ImageCorrection> _correction;
};

/** @brief The model a CorrectedCamera wraps (through any number of wrappers),
  *        or `cam` itself: for code that reads a concrete model's parameters. */
const Camera& baseCamera(const Camera& cam);
