#include "Camera.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "myMATH.h"
#include "RefractCurved.h"

// Convert Rodrigues rotation vector to rotation matrix.
// Input: rotation vector (axis * angle).
// Output: 3x3 rotation matrix.
Matrix<double> Rodrigues(const Pt3D& r_vec) {
    const double theta = std::sqrt(r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] +
                                                                  r_vec[2] * r_vec[2]);
    Matrix<double> R(3, 3, 0.0);
    if (theta < 1e-8) {
        R(0, 0) = 1.0;
        R(1, 1) = 1.0;
        R(2, 2) = 1.0;
        return R;
    }

    const double x = r_vec[0] / theta;
    const double y = r_vec[1] / theta;
    const double z = r_vec[2] / theta;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double t = 1.0 - c;

    R(0, 0) = t * x * x + c;
    R(0, 1) = t * x * y - s * z;
    R(0, 2) = t * x * z + s * y;
    R(1, 0) = t * x * y + s * z;
    R(1, 1) = t * y * y + c;
    R(1, 2) = t * y * z - s * x;
    R(2, 0) = t * x * z - s * y;
    R(2, 1) = t * y * z + s * x;
    R(2, 2) = t * z * z + c;
    return R;
}

// Convert rotation matrix to Rodrigues vector.
// Input: 3x3 rotation matrix.
// Output: Rodrigues rotation vector.
Pt3D rmtxTorvec(Matrix<double> const& r_mtx) {
    Pt3D r_vec;
    double tr = (myMATH::trace<double>(r_mtx) - 1) / 2;
    tr = tr > 1 ? 1 : tr < -1 ? -1 : tr;
    double theta = std::acos(tr);
    double s = std::sin(theta);

    if (s > SMALLNUMBER) {
        double ratio = theta / (2 * s);
        r_vec[0] = (r_mtx(2, 1) - r_mtx(1, 2)) * ratio;
        r_vec[1] = (r_mtx(0, 2) - r_mtx(2, 0)) * ratio;
        r_vec[2] = (r_mtx(1, 0) - r_mtx(0, 1)) * ratio;
    } else if (tr > 0) {
        r_vec[0] = 0;
        r_vec[1] = 0;
        r_vec[2] = 0;
    } else {
        r_vec[0] = theta * std::sqrt((r_mtx(0, 0) + 1) / 2);
        r_vec[1] =
                theta * std::sqrt((r_mtx(1, 1) + 1) / 2) * (r_mtx(0, 1) > 0 ? 1 : -1);
        r_vec[2] =
                theta * std::sqrt((r_mtx(2, 2) + 1) / 2) * (r_mtx(0, 2) > 0 ? 1 : -1);
    }
    return r_vec;
}

StatusOr<Pt2D> normalizeProjectResult(const Pt2D& p) {
    const double cut = -1e300;
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || p[0] <= cut ||
            p[1] <= cut) {
        return STATUS_OR_ERR(Pt2D, ErrorCode::GeometryFailure,
                                                  "project failed: invalid projection result");
    }
    return p;
}

Status mapExceptionToStatus(const std::exception& e, const char* where) {
    const std::string msg = e.what();
    if (msg.find("total internal reflection") != std::string::npos) {
        return STATUS_ERR_CTX(ErrorCode::TotalInternalReflection, where, msg);
    }
    if (msg.find("parallel") != std::string::npos) {
        return STATUS_ERR_CTX(ErrorCode::ParallelGeometry, where, msg);
    }
    return STATUS_ERR_CTX(ErrorCode::GeometryFailure, where, msg);
}

void logRefractionProjectFailure(const char* reason,
                                 const Pt3D& pt_world,
                                 double residual = -1.0,
                                 int iter = -1) {
    // Print first failures densely, then sample periodically to avoid flooding.
    static int fail_count = 0;
    fail_count++;
    if (fail_count > 100 && (fail_count % 200) != 0) {
        return;
    }

    std::cerr << "[REFR-PROJ-FAIL] reason=" << reason
              << " pt=(" << pt_world[0] << "," << pt_world[1] << "," << pt_world[2] << ")";
    if (residual >= 0.0) {
        std::cerr << " residual=" << residual;
    }
    if (iter >= 0) {
        std::cerr << " iter=" << iter;
    }
    std::cerr << std::endl;
}

namespace {
namespace refraction_snell1d {

// Compute projected in-plane length at layer k from first-segment length L1.
// Input: L1/H1 at first segment, target segment thickness Hk, refractive indices n1/nk.
// Output: Lk projected length in target segment; false when geometry is not physically valid.
bool calProjLen(double L1, double H1, double Hk,
                double n1, double nk, double& Lk) {
    const double ratio = n1 / nk;
    const double den = H1 * H1 + (1.0 - ratio * ratio) * L1 * L1;
    if (den <= 0.0) {
        return false;
    }
    Lk = L1 / std::sqrt(den) * ratio * Hk;
    return std::isfinite(Lk);
}

// Compute d(Lk)/d(L1) for one segment under parallel-plate Snell geometry.
// Input: same parameters as calProjLen.
// Output: derivative value dLk_dL1; false when denominator is invalid.
bool calProjLenGrad(double L1, double H1, double Hk,
                    double n1, double nk,
                    double& dLk_dL1) {
    const double ratio = n1 / nk;
    const double H1_2 = H1 * H1;
    const double den = H1_2 + (1.0 - ratio * ratio) * L1 * L1;
    if (den <= 0.0) {
        return false;
    }
    dLk_dL1 = ratio * Hk * H1_2 / (std::sqrt(den) * den);
    return std::isfinite(dLk_dL1);
}

// Accumulate total projected length across all segments given L1.
// Input: segment thickness list H_list and refractive-index list n_list.
// Output: L_tot; false if any segment becomes invalid.
bool calLtot(double L1,
             const std::vector<double>& H_list,
             const std::vector<double>& n_list,
             double& L_tot) {
    L_tot = L1;
    for (size_t i = 0; i + 1 < H_list.size(); ++i) {
        double Lk = 0.0;
        if (!calProjLen(L1, H_list[0], H_list[i + 1],
                        n_list[0], n_list[i + 1], Lk)) {
            return false;
        }
        L_tot += Lk;
    }
    return std::isfinite(L_tot);
}

// Accumulate gradient d(L_tot)/d(L1) across all segments.
// Input: same H_list/n_list as total length function.
// Output: dL_tot_dL1; false if any segment gradient is invalid.
bool calLtotGrad(double L1,
                 const std::vector<double>& H_list,
                 const std::vector<double>& n_list,
                 double& dL_tot_dL1) {
    dL_tot_dL1 = 1.0;
    for (size_t i = 0; i + 1 < H_list.size(); ++i) {
        double dLk = 0.0;
        if (!calProjLenGrad(L1, H_list[0], H_list[i + 1],
                            n_list[0], n_list[i + 1],
                            dLk)) {
            return false;
        }
        dL_tot_dL1 += dLk;
    }
    return std::isfinite(dL_tot_dL1);
}

} // namespace refraction_snell1d
} // namespace

// Shared pinhole projection helper.
// Input: world point + pinhole parameters.
// Output: normalized undistorted image coordinates.
Pt2D Camera::worldToUndistImg(Pt3D const& pt_world,
                              PinholeParam const& param) {
    Pt3D temp = param.r_mtx * pt_world + param.t_vec;
    Pt2D pt_img_mm;
    pt_img_mm[0] = temp[2] ? (temp[0] / temp[2]) : temp[0];
    pt_img_mm[1] = temp[2] ? (temp[1] / temp[2]) : temp[1];
    return pt_img_mm;
}

// Shared distortion helper.
// Input: normalized undistorted image coordinate.
// Output: distorted image coordinate in pixels.
Pt2D Camera::distort(Pt2D const& pt_img_undist,
                     PinholeParam const& param) {
    double x = pt_img_undist[0];
    double y = pt_img_undist[1];
    double xd, yd;

    if (param.is_distorted) {
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;
        double a1 = 2 * x * y;
        double a2 = r2 + 2 * x * x;
        double a3 = r2 + 2 * y * y;

        double cdist = 1 + param.dist_coeff[0] * r2 + param.dist_coeff[1] * r4;
        if (param.n_dist_coeff > 4) {
            cdist += param.dist_coeff[4] * r6;
        }

        double icdist2 = 1.0;
        if (param.n_dist_coeff > 5) {
            icdist2 = 1.0 / (1 + param.dist_coeff[5] * r2 + param.dist_coeff[6] * r4 +
                                              param.dist_coeff[7] * r6);
        }

        xd = x * cdist * icdist2 + param.dist_coeff[2] * a1 +
                  param.dist_coeff[3] * a2;
        yd = y * cdist * icdist2 + param.dist_coeff[2] * a3 +
                  param.dist_coeff[3] * a1;
        if (param.n_dist_coeff > 8) {
            xd += param.dist_coeff[8] * r2 + param.dist_coeff[9] * r4;
            yd += param.dist_coeff[10] * r2 + param.dist_coeff[11] * r4;
        }
    } else {
        xd = x;
        yd = y;
    }

    return Pt2D(xd * param.cam_mtx(0, 0) + param.cam_mtx(0, 2),
                            yd * param.cam_mtx(1, 1) + param.cam_mtx(1, 2));
}

// Shared undistortion helper.
// Input: distorted image coordinate in pixels.
// Output: normalized undistorted image coordinate.
Pt2D Camera::undistort(Pt2D const& pt_img_dist,
                       PinholeParam const& param) {
    double fx = param.cam_mtx(0, 0);
    double fy = param.cam_mtx(1, 1);
    double cx = param.cam_mtx(0, 2);
    double cy = param.cam_mtx(1, 2);

    double u = pt_img_dist[0];
    double v = pt_img_dist[1];
    double x = (u - cx) / fx;
    double y = (v - cy) / fy;
    Pt2D img_undist(x, y);
    Pt2D img_dist(0, 0);

    if (param.is_distorted) {
        double x0 = x;
        double y0 = y;
        double error = std::numeric_limits<double>::max();
        int iter = 0;

        while (error > UNDISTORT_EPS && iter < UNDISTORT_MAX_ITER) {
            double r2 = x * x + y * y;
            double a1 = 2 * x * y;
            double a2 = r2 + 2 * x * x;
            double a3 = r2 + 2 * y * y;

            double icdist;
            if (param.n_dist_coeff == 4) {
                icdist = 1 / (1 + (param.dist_coeff[1] * r2 + param.dist_coeff[0]) * r2);
            } else if (param.n_dist_coeff == 5) {
                icdist =
                        1 / (1 + ((param.dist_coeff[4] * r2 + param.dist_coeff[1]) * r2 +
                                            param.dist_coeff[0]) *
                                                  r2);
            } else {
                icdist = (1 + ((param.dist_coeff[7] * r2 + param.dist_coeff[6]) * r2 +
                                              param.dist_coeff[5]) *
                                                    r2) /
                                  (1 + ((param.dist_coeff[4] * r2 + param.dist_coeff[1]) * r2 +
                                              param.dist_coeff[0]) *
                                                    r2);
            }

            if (icdist < 0) {
                img_undist[0] = (u - cx) / fx;
                img_undist[1] = (v - cy) / fy;
                break;
            }

            double deltaX = param.dist_coeff[2] * a1 + param.dist_coeff[3] * a2;
            double deltaY = param.dist_coeff[2] * a3 + param.dist_coeff[3] * a1;
            if (param.n_dist_coeff > 8) {
                deltaX += param.dist_coeff[8] * r2 + param.dist_coeff[9] * r2 * r2;
                deltaY += param.dist_coeff[10] * r2 + param.dist_coeff[11] * r2 * r2;
            }
            x = (x0 - deltaX) * icdist;
            y = (y0 - deltaY) * icdist;

            img_undist[0] = x;
            img_undist[1] = y;
            img_dist = distort(img_undist, param);
            error = std::sqrt(std::pow(img_dist[0] - u, 2) +
                                                std::pow(img_dist[1] - v, 2));
            iter++;
        }
    }

    return img_undist;
}

// Read text file and strip inline comments, preserving legacy parsing behavior.
void loadFileToStream(const std::string& file_name, std::stringstream& out) {
    std::ifstream infile(file_name.c_str(), std::ios::in);
    if (!infile) {
        throw std::runtime_error("cannot open file: " + file_name);
    }
    std::string line;
    while (std::getline(infile, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos > 0) {
            if (comment_pos < std::string::npos) {
                line.erase(comment_pos);
            }
        } else if (comment_pos == 0) {
            continue;
        }
        out << line << '\t';
    }
}

// distort() reads 4, 5, 8 or 12 coefficients, so a vector of another length up to
// 12 is padded with zeros to the next of those; without this it read past the end.
// A longer vector (OpenCV's 14) is kept: distort() reads its first 12.
std::vector<double> paddedDistCoeff(const std::vector<double>& dist_coeff) {
    std::vector<double> out = dist_coeff;
    if (out.empty()) {
        return out;
    }
    for (std::size_t len : {std::size_t(4), std::size_t(5), std::size_t(8), std::size_t(12)}) {
        if (out.size() <= len) {
            out.resize(len, 0.0);
            break;
        }
    }
    return out;
}

// Parse distortion coefficients from one comma-separated field.
void loadDistCoeff(const std::string& dist_coeff_str, PinholeParam& p) {
    p.dist_coeff.clear();
    p.is_distorted = false;
    p.n_dist_coeff = 0;

    std::stringstream dist_coeff_stream(dist_coeff_str);
    double dist_coeff;
    int id = 0;
    while (dist_coeff_stream >> dist_coeff) {
        p.dist_coeff.push_back(dist_coeff);
        id++;
        // A negative coefficient distorts too (k1 < 0 is barrel distortion).
        if (std::abs(dist_coeff) > SMALLNUMBER) {
            p.is_distorted = true;
        }
        if (dist_coeff_stream.peek() == ',') {
            dist_coeff_stream.ignore();
        }
    }

    if (p.is_distorted) {
        p.n_dist_coeff = id;
        if (id != 4 && id != 5 && id != 8 && id != 12) {
            throw std::runtime_error("number of distortion coefficients is wrong");
        }
    }
}

// Camera files carry every double exactly: max_digits10 significant digits,
// i.e. max_digits10 - 1 after the point in scientific notation. At
// SAVEPRECISION (8) a save and load moved a projection by up to 1.7e-6 px.
constexpr int kSaveDigits = std::numeric_limits<double>::max_digits10 - 1;

void setSavePrecision(std::ofstream& outfile) {
    outfile.setf(std::ios_base::scientific);
    outfile.precision(kSaveDigits);
}

template <typename Param>
void saveDistCoeffLine(std::ofstream& outfile, const Param& p) {
    int size = static_cast<int>(p.dist_coeff.size());
    if (size <= 0) {
        outfile << std::endl;
        return;
    }
    for (int i = 0; i < size - 1; i++) {
        outfile << p.dist_coeff[i] << ",";
    }
    outfile << p.dist_coeff[size - 1] << std::endl;
}

// Project world point using pinhole model.
// Input: world point in physical coordinates.
// Output: distorted image coordinate in pixels.
StatusOr<Pt2D> PinholeCamera::project(const Pt3D& pt_world, bool) const {
    try {
        // Main flow:
        // 1) Transform world point to normalized pinhole image coordinates.
        // 2) Apply lens distortion and camera intrinsics.
        Pt3D pt_cam = _param.r_mtx * pt_world + _param.t_vec;
        Pt2D pt_img_undist;
        pt_img_undist[0] = pt_cam[2] ? (pt_cam[0] / pt_cam[2]) : pt_cam[0];
        pt_img_undist[1] = pt_cam[2] ? (pt_cam[1] / pt_cam[2]) : pt_cam[1];
        Pt2D pt_img_dist = PinholeCamera::distort(pt_img_undist, _param);
        return normalizeProjectResult(pt_img_dist);
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "PinholeCamera::project");
        return STATUS_OR_ERR_CTX(Pt2D, st.err.code, st.err.message, st.err.context);
    }
}

// Build line-of-sight from one image point using pinhole model.
// Input: distorted image coordinate in pixels.
// Output: line in world coordinates.
StatusOr<Line3D> PinholeCamera::lineOfSight(const Pt2D& pt_img_dist) const {
    try {
        // Main flow:
        // 1) Remove image distortion to get normalized image coordinates.
        // 2) Back-project through inverse extrinsics to construct world-space ray.
        Pt2D pt_img_undist = PinholeCamera::undistort(pt_img_dist, _param);
        Pt3D pt_world(pt_img_undist[0], pt_img_undist[1], 1.0);
        pt_world = _param.r_mtx_inv * pt_world + _param.t_vec_inv;

        Line3D line;
        line.pt = _param.t_vec_inv;
        line.unit_vector = myMATH::createUnitVector(_param.t_vec_inv, pt_world);
        return line;
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "PinholeCamera::lineOfSight");
        return STATUS_OR_ERR_CTX(Line3D, st.err.code, st.err.message, st.err.context);
    }
}

int PinholeCamera::getNRow() const { return _param.n_row; }
int PinholeCamera::getNCol() const { return _param.n_col; }

// Save pinhole parameters to text file.
// Input: output file path.
// Output: file with exactly the legacy pinhole format.
Status PinholeCamera::saveParameters(const std::string& file_name) const {
    try {
        std::ofstream outfile(file_name.c_str(), std::ios::out);
        setSavePrecision(outfile);
        outfile << "# Camera Model: (PINHOLE/POLYNOMIAL)" << std::endl;
        outfile << "PINHOLE" << std::endl;
        outfile << "# Camera Calibration Error: \nNone\n# Pose Calibration Error: \nNone"
                        << std::endl;
        outfile << "# Image Size: (n_row,n_col)" << std::endl;
        outfile << _param.n_row << "," << _param.n_col << std::endl;
        outfile << "# Camera Matrix: " << std::endl;
        auto cam_mtx = _param.cam_mtx;
        cam_mtx.write(outfile, kSaveDigits);
        outfile << "# Distortion Coefficients: " << std::endl;
        saveDistCoeffLine(outfile, _param);
        outfile << "# Rotation Vector: " << std::endl;
        auto r_vec = rmtxTorvec(_param.r_mtx);
        auto r_vec_t = r_vec.transpose();
        r_vec_t.write(outfile, kSaveDigits);
        outfile << "# Rotation Matrix: " << std::endl;
        auto r_mtx = _param.r_mtx;
        r_mtx.write(outfile, kSaveDigits);
        outfile << "# Inverse of Rotation Matrix: " << std::endl;
        auto r_mtx_inv = _param.r_mtx_inv;
        r_mtx_inv.write(outfile, kSaveDigits);
        outfile << "# Translation Vector: " << std::endl;
        auto t_vec = _param.t_vec;
        auto t_vec_t = t_vec.transpose();
        t_vec_t.write(outfile, kSaveDigits);
        outfile << "# Inverse of Translation Vector: " << std::endl;
        auto t_vec_inv = _param.t_vec_inv;
        auto t_vec_inv_t = t_vec_inv.transpose();
        t_vec_inv_t.write(outfile, kSaveDigits);
        return Status::OK();
    } catch (const std::exception& e) {
        return STATUS_ERR_CTX(ErrorCode::IOfailure, "PinholeCamera::saveParameters", e.what());
    }
}

std::shared_ptr<Camera> PinholeCamera::clone() const {
    return std::make_shared<PinholeCamera>(*this);
}

Status PinholeCamera::setImageSize(int n_row, int n_col) {
    REQUIRE(n_row > 0 && n_col > 0, ErrorCode::InvalidArgument,
                    "PinholeCamera::setImageSize invalid image size");
    _param.n_row = n_row;
    _param.n_col = n_col;
    return Status::OK();
}

Status PinholeCamera::setIntrinsics(double fx, double fy, double cx, double cy,
                                                                        const std::vector<double>& dist_coeff) {
    // Main flow:
    // 1) Write intrinsic matrix.
    // 2) Store distortion coefficients.
    // 3) Infer distortion enable flag.
    REQUIRE(fx > 0 && fy > 0, ErrorCode::InvalidArgument,
                    "PinholeCamera::setIntrinsics invalid focal length");
    _param.cam_mtx = Matrix<double>(3, 3, 0.0);
    _param.cam_mtx(0, 0) = fx;
    _param.cam_mtx(1, 1) = fy;
    _param.cam_mtx(0, 2) = cx;
    _param.cam_mtx(1, 2) = cy;
    _param.cam_mtx(2, 2) = 1.0;
    _param.dist_coeff = paddedDistCoeff(dist_coeff);
    _param.n_dist_coeff = static_cast<int>(_param.dist_coeff.size());
    _param.is_distorted = false;
    for (double v : dist_coeff) {
        if (std::abs(v) > SMALLNUMBER) {
            _param.is_distorted = true;
            break;
        }
    }
    return Status::OK();
}

Status PinholeCamera::setExtrinsics(const Pt3D& rvec, const Pt3D& tvec) {
    // Main flow:
    // 1) Convert Rodrigues vector to rotation matrix.
    // 2) Cache inverse rotation and camera center.
    _param.r_mtx = Rodrigues(rvec);
    _param.t_vec = tvec;
    _param.r_mtx_inv = myMATH::inverse(_param.r_mtx);
    _param.t_vec_inv = (_param.r_mtx_inv * _param.t_vec) * -1.0;
    return Status::OK();
}

Status PinholeCamera::commitUpdate() {
    return Status::OK();
}

namespace {

namespace refraction_common {

// Build orthonormal in-plane basis from plane normal.
// Input: plane normal and point.
// Output: u_axis and v_axis lying on plane.
void buildPlaneOrthonormalBasis(Pt3D& u_axis,
                                Pt3D& v_axis,
                                const Plane3D& plane) {
    if (std::abs(plane.norm_vector[2]) < 0.9) {
        u_axis = myMATH::cross(plane.norm_vector, {0, 0, 1});
    } else {
        u_axis = myMATH::cross(plane.norm_vector, {1, 0, 0});
    }
    u_axis /= u_axis.norm();
    v_axis = myMATH::cross(plane.norm_vector, u_axis);
    v_axis /= v_axis.norm();
}

// Build all refractive interface planes and in-plane basis.
// Input: farthest refractive plane, widths and plate count.
// Output: plane_array/u_axis/v_axis ready for tracing.
void buildRefractionPlaneStackAndBasis(PinPlateParam& pin) {
    // Main flow:
    // 1) Build refractive interface planes from farthest to nearest.
    // 2) Build a 2D coordinate basis on the farthest reference plane.
    pin.plane_array = std::vector<Plane3D>(pin.n_plate + 1);
    pin.plane_array[0] = pin.plane;

    for (int i = 1; i <= pin.n_plate; i++) {
        Plane3D& prev_plane = pin.plane_array[i - 1];
        Plane3D& curr_plane = pin.plane_array[i];
        curr_plane.pt = prev_plane.pt - prev_plane.norm_vector * pin.w_array[i - 1];
        curr_plane.norm_vector = prev_plane.norm_vector;
    }

    buildPlaneOrthonormalBasis(pin.u_axis, pin.v_axis, pin.plane);
}

// Refract a direction vector across one interface.
// Input: current direction, interface normal, n_in/n_out ratio.
// Output: updated refracted direction, false if total internal reflection.
bool refractDirection(Pt3D& dir_refract,
                      const Pt3D& normal,
                      double refract_ratio,
                      bool is_forward) {
    double cos_1 = std::clamp(myMATH::dot(dir_refract, normal), -1.0, 1.0);
    double cos_2 = 1.0 - refract_ratio * refract_ratio * (1.0 - cos_1 * cos_1);
    if (cos_2 <= 0) {
        return false;
    }

    if (std::abs(cos_1) < SMALLNUMBER) {
        cos_1 = 0.0;
        double tan_2 = std::sqrt((1.0 - cos_2) / cos_2);
        double factor = -dir_refract.norm() / tan_2 * (is_forward ? 1.0 : -1.0);
        for (int i = 0; i < 3; i++) {
            dir_refract[i] = dir_refract[i] + normal[i] * factor;
        }
    } else {
        double factor = -refract_ratio * cos_1 -
                                        std::sqrt(cos_2) * (is_forward ? 1.0 : -1.0);
        for (int i = 0; i < 3; i++) {
            dir_refract[i] = dir_refract[i] * refract_ratio + normal[i] * factor;
        }
    }

    dir_refract /= dir_refract.norm();
    return true;
}

} // namespace refraction_common

namespace refraction_lm {

// Project a 3D point to 2D coordinates on a plane basis.
// Input: point and plane basis.
// Output: 2D coordinates in (u,v) basis.
void projectToPlaneBasis(std::vector<double>& coords,
                         const Pt3D& pt,
                         const Plane3D& plane,
                         const Pt3D& u_axis,
                         const Pt3D& v_axis) {
    Pt3D diff = {pt[0] - plane.pt[0], pt[1] - plane.pt[1], pt[2] - plane.pt[2]};
    coords[0] = myMATH::dot(diff, u_axis);
    coords[1] = myMATH::dot(diff, v_axis);
}

// Reconstruct a 3D point from 2D plane basis coordinates.
// Input: basis origin/axes and 2D coordinates.
// Output: 3D point on that plane.
void reconstructFromPlaneBasis(
    Pt3D& result, const Pt3D& origin, const Pt3D& u_axis, const Pt3D& v_axis,
    const std::vector<double>& coords) {
    for (int i = 0; i < 3; i++) {
        result[i] = origin[i] + coords[0] * u_axis[i] + coords[1] * v_axis[i];
    }
}

// Forward trace from world point to camera side through plate stack.
// Input: world point + entry point at farthest interface.
// Output: exit point/direction at nearest interface and final medium.
// Return false on TIR / parallel crossing / invalid interface intersection.
bool traceRayToCam(Pt3D& pt_exit,
                   Pt3D& exit_direction,
                   const Pt3D& pt_world,
                   const Pt3D& pt_entry,
                   const PinPlateParam& pin,
                   const char** out_fail_code = nullptr) {
    auto setFailCode = [&](const char* code) {
        if (out_fail_code) {
            *out_fail_code = code;
        }
    };

    Line3D line;
    line.pt = pt_world;
    line.unit_vector = myMATH::createUnitVector(pt_world, pt_entry);
    Pt3D pt_cross = pt_entry;

    for (int plane_id = 1; plane_id <= pin.n_plate; plane_id++) {
        const Plane3D& plane_curr = pin.plane_array[plane_id - 1];
        const Plane3D& plane_next = pin.plane_array[plane_id];
        double refract_ratio = pin.refract_array[plane_id - 1] / pin.refract_array[plane_id];

        if (!refraction_common::refractDirection(line.unit_vector, plane_curr.norm_vector,
                                                 refract_ratio, true)) {
            setFailCode("sample_total_internal_reflection");
            return false;
        }
        line.pt = pt_cross;
        if (myMATH::crossPoint(pt_cross, line, plane_next)) {
            setFailCode("sample_parallel_to_interface");
            return false;
        }
    }

    pt_exit = pt_cross;
    exit_direction = line.unit_vector;
    double refract_ratio = pin.refract_array[pin.n_plate] / pin.refract_array[pin.n_plate + 1];
    if (!refraction_common::refractDirection(
            exit_direction, pin.plane_array[pin.n_plate].norm_vector,
            refract_ratio, true)) {
        setFailCode("sample_total_internal_reflection");
        return false;
    }

    if (out_fail_code) {
        *out_fail_code = nullptr;
    }
    return true;
}

// Refractive projection solver (LM + line search).
// Input: world point and full pinplate parameters.
// Output: (failure_flag, refracted_point_on_farthest_plane, residual, iterations).
std::tuple<bool, Pt3D, double, int>
solveProjectionByRefractionLMImpl(Pt3D const& pt_world,
                                  const PinPlateParam& pin) {
    std::tuple<bool, Pt3D, double, int> result = {false, {0, 0, 0}, -1, 0};
    const int proj_nmax = pin.proj_nmax;
    const double proj_tol = pin.proj_tol;
    const double eps = 1e-5;

    auto logLmFail = [&](const char* code, int iter, double residual = std::numeric_limits<double>::quiet_NaN()) {
        static int fail_count = 0;
        fail_count++;
        if (fail_count > 100 && (fail_count % 200) != 0) {
            return;
        }

        std::ostringstream oss;
        oss << "\t\t[REFR-LM-FAIL] "
            << std::left << std::setw(58) << (std::string("code=") + code)
            << " iter=" << std::right << std::setw(4) << iter
            << " pt=(" << std::fixed << std::setprecision(4)
            << pt_world[0] << "," << pt_world[1] << "," << pt_world[2] << ")";
        if (std::isfinite(residual)) {
            oss << " res=" << std::scientific << std::setprecision(3) << residual;
        }
        std::cerr << oss.str() << std::endl;
    };

    // Radius guard is enabled when total internal reflection can occur.
    bool is_check_radius = false;
    double radius_max2 = 0;
    if (pin.refract_ratio_max > 1.0) {
        is_check_radius = true;
        double sin_angle = 1.0 / pin.refract_ratio_max;
        radius_max2 = myMATH::dist2(pt_world, pin.plane) / (1.0 - sin_angle * sin_angle);
    }

    // Build initial ray from object point to camera center.
    Line3D line;
    line.pt = pt_world;
    line.unit_vector = myMATH::createUnitVector(pt_world, pin.t_vec_inv);

    Pt3D pt_init;
    if (myMATH::crossPoint(pt_init, line, pin.plane)) {
        std::get<0>(result) = true;
        logLmFail("entry_plane_intersection_failed", 0);
        return result;
    }

    if (is_check_radius) {
        // Clamp the initial entry point inside valid radius before LM starts.
        double radius2 = myMATH::dist2(pt_init, pt_world);
        if (radius2 >= radius_max2) {
            double scale = std::sqrt(radius_max2 * 0.95 / radius2);
            for (int i = 0; i < 3; i++) {
                pt_init[i] = pt_world[i] + (pt_init[i] - pt_world[i]) * scale;
            }
            Pt3D pt_exit_test, dir_test;
            const char* trace_fail_code = nullptr;
            if (!traceRayToCam(pt_exit_test, dir_test, pt_world, pt_init, pin, &trace_fail_code)) {
                std::get<0>(result) = true;
                logLmFail(trace_fail_code ? trace_fail_code : "initial_ray_trace_failed", 0);
                return result;
            }
        }
    }

    // Convert the initial 3D entry point to 2D plane-basis coordinates.
    std::vector<double> x0(2);
    projectToPlaneBasis(x0, pt_init, pin.plane, pin.u_axis, pin.v_axis);

    // Estimate scaling between 2D basis updates and world-space displacement.
    const double L_min = 0.1;
    const double mu_min = 1e-4;
    const double mu_scale = 0.1;
    double L = L_min;
    {
        Pt3D pt_exit_scale, exit_dir_scale;
        if (traceRayToCam(pt_exit_scale, exit_dir_scale, pt_world, pt_init, pin)) {
            Pt3D diff = pt_exit_scale - pin.plane.pt;
            double dist = myMATH::dot(diff, pin.plane.norm_vector);
            Pt3D pt_exit_proj = pt_exit_scale - pin.plane.norm_vector * dist;
            double line_dist = myMATH::dist(pt_exit_proj, line);
            L = std::max(line_dist, L_min);
        }
    }

    Pt3D pt_current, pt_exit, exit_direction;
    Pt3D vec_to_pinhole, residual_vec;
    std::vector<double> delta_y(2, 0.0);
    std::vector<double> delta_x(2);
    std::vector<double> x(2);
    std::vector<double> delta_y_perturb(2);
    std::vector<double> x_perturb(2);
    double J[2][2];
    double F[2];
    double JTJ[2][2];

    double lambda = 0.01;
    const double lambda_min = 1e-10;
    const double lambda_max = 1e10;
    const double mu_max = 0.1 * L;

    for (int iter = 0; iter < proj_nmax; iter++) {
        // Reconstruct entry point on farthest interface from current 2D parameters.
        delta_x[0] = L * delta_y[0];
        delta_x[1] = L * delta_y[1];
        x[0] = x0[0] + delta_x[0];
        x[1] = x0[1] + delta_x[1];
        reconstructFromPlaneBasis(pt_current, pin.plane.pt, pin.u_axis, pin.v_axis, x);

        double radius2_current = 0.0;
        if (is_check_radius) {
            radius2_current = myMATH::dist2(pt_current, pt_world);
            if (radius2_current >= radius_max2) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                logLmFail("entry_point_outside_valid_radius", iter + 1);
                return result;
            }
        }

        // Forward trace through interfaces ordered from farthest to nearest.
        const char* trace_fail_code = nullptr;
        if (!traceRayToCam(pt_exit, exit_direction, pt_world, pt_current, pin, &trace_fail_code)) {
            std::get<0>(result) = true;
            std::get<3>(result) = iter + 1;
            logLmFail(trace_fail_code ? trace_fail_code : "ray_trace_failed_at_current_step", iter + 1);
            return result;
        }

        vec_to_pinhole = pin.t_vec_inv - pt_exit;
        double proj_along_ray = myMATH::dot(vec_to_pinhole, exit_direction);
        residual_vec = vec_to_pinhole - exit_direction * proj_along_ray;

        double residual = residual_vec.norm();
        double barrier_current = 0.0;
        if (is_check_radius) {
            double mu = std::max(mu_min, std::min(mu_scale * residual, mu_max));
            double d = radius_max2 - radius2_current;
            barrier_current = -mu * std::log(d);
        }

        // Stop when ray-to-camera-center distance is under tolerance.
        if (residual < proj_tol) {
            std::get<1>(result) = pt_exit;
            std::get<2>(result) = residual;
            std::get<3>(result) = iter + 1;
            return result;
        }

        F[0] = myMATH::dot(residual_vec, pin.u_axis);
        F[1] = myMATH::dot(residual_vec, pin.v_axis);

        // Build numerical Jacobian in the 2D basis parameter space.
        for (int j = 0; j < 2; j++) {
            delta_y_perturb = delta_y;
            delta_y_perturb[j] += eps;
            delta_x[0] = L * delta_y_perturb[0];
            delta_x[1] = L * delta_y_perturb[1];
            x_perturb[0] = x0[0] + delta_x[0];
            x_perturb[1] = x0[1] + delta_x[1];
            reconstructFromPlaneBasis(pt_current, pin.plane.pt, pin.u_axis, pin.v_axis, x_perturb);

            if (is_check_radius) {
                double radius2 = myMATH::dist2(pt_current, pt_world);
                if (radius2 >= radius_max2) {
                    std::get<0>(result) = true;
                    std::get<3>(result) = iter + 1;
                    logLmFail("jacobian_sample_outside_valid_radius", iter + 1);
                    return result;
                }
            }

            trace_fail_code = nullptr;
            if (!traceRayToCam(pt_exit, exit_direction, pt_world, pt_current, pin, &trace_fail_code)) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                logLmFail(trace_fail_code ? trace_fail_code : "ray_trace_failed_at_jacobian_sample", iter + 1);
                return result;
            }

            vec_to_pinhole = pin.t_vec_inv - pt_exit;
            proj_along_ray = myMATH::dot(vec_to_pinhole, exit_direction);
            residual_vec = vec_to_pinhole - exit_direction * proj_along_ray;
            double F_perturb[2];
            F_perturb[0] = myMATH::dot(residual_vec, pin.u_axis);
            F_perturb[1] = myMATH::dot(residual_vec, pin.v_axis);
            J[0][j] = (F_perturb[0] - F[0]) / (delta_y_perturb[j] - delta_y[j]);
            J[1][j] = (F_perturb[1] - F[1]) / (delta_y_perturb[j] - delta_y[j]);
        }

        JTJ[0][0] = J[0][0] * J[0][0] + J[1][0] * J[1][0] + lambda;
        JTJ[0][1] = J[0][0] * J[0][1] + J[1][0] * J[1][1];
        JTJ[1][0] = JTJ[0][1];
        JTJ[1][1] = J[0][1] * J[0][1] + J[1][1] * J[1][1] + lambda;

        double JTF[2];
        JTF[0] = -(J[0][0] * F[0] + J[1][0] * F[1]);
        JTF[1] = -(J[0][1] * F[0] + J[1][1] * F[1]);

        // Solve damped normal equation and reject ill-conditioned updates.
        double det = JTJ[0][0] * JTJ[1][1] - JTJ[0][1] * JTJ[1][0];
        double trace = JTJ[0][0] + JTJ[1][1];
        double condition_estimate = trace / std::abs(det);
        if (std::abs(det) < 1e-20 || condition_estimate > 1e12) {
            lambda = std::min(lambda * 10.0, lambda_max);
            if (lambda >= lambda_max) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                logLmFail("lm_damping_upper_limit_reached@linear_system_ill_conditioned",
                          iter + 1, residual);
                return result;
            }
            continue;
        }

        double dx[2];
        dx[0] = (JTF[0] * JTJ[1][1] - JTF[1] * JTJ[0][1]) / det;
        dx[1] = (JTF[1] * JTJ[0][0] - JTF[0] * JTJ[1][0]) / det;

        // Line-search the LM step in case of radius or residual barrier violations.
        double alpha = 1.0;
        bool step_accepted = false;
        bool line_search_trace_failed = false;
        const int max_line_search = 10;
        for (int ls = 0; ls < max_line_search; ls++) {
            delta_y_perturb[0] = delta_y[0] + alpha * dx[0];
            delta_y_perturb[1] = delta_y[1] + alpha * dx[1];
            delta_x[0] = L * delta_y_perturb[0];
            delta_x[1] = L * delta_y_perturb[1];
            x_perturb[0] = x0[0] + delta_x[0];
            x_perturb[1] = x0[1] + delta_x[1];
            reconstructFromPlaneBasis(pt_current, pin.plane.pt, pin.u_axis, pin.v_axis, x_perturb);

            if (is_check_radius) {
                double radius2 = myMATH::dist2(pt_current, pt_world);
                if (radius2 >= radius_max2) {
                    alpha *= 0.5;
                    continue;
                }
            }

            const char* ls_trace_fail_code = nullptr;
            if (traceRayToCam(pt_exit, exit_direction, pt_world, pt_current, pin,
                              &ls_trace_fail_code)) {
                vec_to_pinhole = pin.t_vec_inv - pt_exit;
                double proj_along_ray_ls = myMATH::dot(vec_to_pinhole, exit_direction);
                residual_vec = vec_to_pinhole - exit_direction * proj_along_ray_ls;
                double residual_new = residual_vec.norm();

                double barrier_new = 0.0;
                if (is_check_radius) {
                    double mu = std::max(mu_min, std::min(mu_scale * residual_new, mu_max));
                    double d = radius_max2 - myMATH::dist2(pt_current, pt_world);
                    barrier_new = -mu * std::log(d);
                }

                if (residual_new + barrier_new < residual + barrier_current) {
                    projectToPlaneBasis(x, pt_current, pin.plane, pin.u_axis, pin.v_axis);
                    delta_y[0] = (x[0] - x0[0]) / L;
                    delta_y[1] = (x[1] - x0[1]) / L;
                    step_accepted = true;
                    lambda = std::max(lambda * 0.1, lambda_min);
                    break;
                }
            } else {
                line_search_trace_failed = true;
            }
            alpha *= 0.5;
        }

        // Increase damping when no acceptable step is found.
        if (!step_accepted) {
            lambda = std::min(lambda * 10.0, lambda_max);
            if (lambda >= lambda_max) {
                std::get<0>(result) = true;
                std::get<3>(result) = iter + 1;
                if (line_search_trace_failed) {
                    logLmFail("lm_damping_upper_limit_reached@line_search_sample_trace_failed",
                              iter + 1, residual);
                } else {
                    logLmFail("lm_damping_upper_limit_reached@line_search_no_descent_step",
                              iter + 1, residual);
                }
                return result;
            }
        }
    }

    std::get<1>(result) = pt_exit;
    std::get<2>(result) = residual_vec.norm();
    std::get<3>(result) = proj_nmax;
    std::get<0>(result) = true;
    logLmFail("lm_max_iterations_reached", proj_nmax, std::get<2>(result));
    return result;
}

} // namespace refraction_lm

namespace refraction_snell1d {

// Refractive projection solver for parallel-plate stacks using 1D Snell
// path-length optimization.
// Input: world point and full pinplate parameters.
// Output: (failure_flag, refracted_point_on_farthest_plane, residual, iterations).
std::tuple<bool, Pt3D, double, int>
solveProjectionBySnell1DImpl(const Pt3D& pt_world,
                             const PinPlateParam& pin) {
    std::tuple<bool, Pt3D, double, int> result = {false, {0, 0, 0}, -1, 0};

    // Centralized failure logger for this solver. Keep formatting consistent
    // with LM diagnostics and throttle repeated messages in long runs.
    auto logSnellFail = [&](const char* code, int iter,
                            double residual = std::numeric_limits<double>::quiet_NaN()) {
        static int fail_count = 0;
        fail_count++;
        if (fail_count > 100 && (fail_count % 200) != 0) {
            return;
        }

        std::ostringstream oss;
        oss << "\t\t[REFR-SNELL1D-FAIL] "
            << std::left << std::setw(52) << (std::string("code=") + code)
            << " iter=" << std::right << std::setw(4) << iter
            << " pt=(" << std::fixed << std::setprecision(4)
            << pt_world[0] << "," << pt_world[1] << "," << pt_world[2] << ")";
        if (std::isfinite(residual)) {
            oss << " res=" << std::scientific << std::setprecision(3) << residual;
        }
        std::cerr << oss.str() << std::endl;
    };

    // Step 0: validate refractive stack layout.
    // For n plates, refract_array must contain [n1, n_plate..., n_last],
    // so size must be n_plate + 2 and at least 3 (two outer media + one layer).
    if (pin.refract_array.size() != pin.w_array.size() + 2 || pin.refract_array.size() < 3) {
        std::get<0>(result) = true;
        logSnellFail("snell1d_invalid_geometry", 0);
        return result;
    }

    // Step 1: reduce 3D geometry to a 1D in-plane length problem.
    // We project world point and pinhole center to the farthest interface plane,
    // then solve only the in-plane path length that satisfies Snell across layers.
    const Plane3D& plane_far = pin.plane;
    const Pt3D pt_world_proj = myMATH::projectPointToPlaneAlongNormal(pt_world, plane_far);
    const Pt3D pt_pinhole_proj = myMATH::projectPointToPlaneAlongNormal(pin.t_vec_inv, plane_far);

    const double H1 = myMATH::dist(pt_world, pt_world_proj);
    const double thickness_tot = std::accumulate(pin.w_array.begin(), pin.w_array.end(), 0.0);
    const double Hn = myMATH::dist(pin.t_vec_inv, pt_pinhole_proj) - thickness_tot;
    const double L_tot = myMATH::dist(pt_world_proj, pt_pinhole_proj);

    // Degenerate geometry guard.
    if (!std::isfinite(H1) || !std::isfinite(Hn) || H1 <= SMALLNUMBER ||
        Hn <= SMALLNUMBER || !std::isfinite(L_tot)) {
        std::get<0>(result) = true;
        logSnellFail("snell1d_invalid_geometry", 0);
        return result;
    }

    std::vector<double> H_list(pin.n_plate + 2, 0.0);
    H_list[0] = H1;
    H_list[pin.n_plate + 1] = Hn;
    for (int i = 1; i <= pin.n_plate; ++i) {
        H_list[i] = pin.w_array[i - 1];
    }

    // Step 2: compute feasible open interval for first-segment projected length L1.
    // Critical bounds come from all layers where n0/nk > 1, i.e. TIR can occur.
    // Use an open upper bound (< critical) so solver never evaluates singular points.
    const double H_tot = std::accumulate(H_list.begin(), H_list.end(), 0.0);
    constexpr double kOpenBoundEps = 1e-8;
    double L1_crit = std::numeric_limits<double>::infinity();
    for (size_t i = 1; i < pin.refract_array.size(); ++i) {
        const double ratio = pin.refract_array.front() / pin.refract_array[i];
        if (ratio > 1.0) {
            const double tmp = ratio * ratio - 1.0;
            if (tmp <= 0.0) {
                std::get<0>(result) = true;
                logSnellFail("snell1d_invalid_geometry", 0);
                return result;
            }
            L1_crit = std::min(L1_crit, H1 / std::sqrt(tmp));
        }
    }
    double L1_max = L_tot;
    if (std::isfinite(L1_crit)) {
        L1_max = std::min(L1_max, L1_crit * (1.0 - kOpenBoundEps));
    }
    if (!std::isfinite(L1_max) || L1_max <= 0.0) {
        std::get<0>(result) = true;
        logSnellFail("snell1d_invalid_geometry", 0);
        return result;
    }

    // Evaluate total projected length. At boundary singularities return +inf as a
    // feasibility signal for bracket construction (not as a converged solution).
    auto evalTotalLength = [&](double L1, double& Lcur) {
        if (!calLtot(L1, H_list, pin.refract_array, Lcur)) {
            Lcur = std::numeric_limits<double>::infinity();
            return false;
        }
        return true;
    };

    // Early feasibility check: if max reachable projected length is still smaller
    // than required L_tot, no physical solution exists for this point.
    double L_max = 0.0;
    evalTotalLength(L1_max, L_max);
    if (!std::isfinite(L_max)) {
        L_max = std::numeric_limits<double>::infinity();
    }
    if (L_max + pin.proj_tol < L_tot) {
        std::get<0>(result) = true;
        logSnellFail("snell1d_unreachable_total_length", 0);
        return result;
    }

    const int proj_nmax = pin.proj_nmax;
    const double proj_tol = pin.proj_tol;

    // Step 3: solve scalar equation f(L1) = L_tot - L_tot_cur(L1) = 0.
    // Use safeguarded Newton updates inside a bracket [L_lo, L_hi], and fall
    // back to bisection when Newton leaves feasible domain or hits singularities.
    double L_lo = 0.0;
    double L_hi = L1_max;
    double L1 = (H_tot > SMALLNUMBER) ? (L_hi * H1 / H_tot) : 0.0;
    L1 = std::clamp(L1, L_lo, L_hi);

    double residue = std::numeric_limits<double>::infinity();
    int iter = 0;
    bool converged = false;
    for (; iter < proj_nmax; ++iter) {
        double L_tot_cur = 0.0;
        const bool eval_ok = evalTotalLength(L1, L_tot_cur);
        if (!eval_ok || !std::isfinite(L_tot_cur)) {
            L_hi = std::max(L_lo, L1);
            L1 = 0.5 * (L_lo + L_hi);
            continue;
        }

        // Residual is positive when current path is too short.
        residue = L_tot - L_tot_cur;
        if (std::abs(residue) < proj_tol) {
            converged = true;
            break;
        }

        if (residue > 0.0) {
            L_lo = std::max(L_lo, L1);
        } else {
            L_hi = std::min(L_hi, L1);
        }
        if (L_hi - L_lo <= proj_tol * 1e-6) {
            L1 = 0.5 * (L_lo + L_hi);
            converged = true;
            break;
        }

        double dL_tot_dL1 = 0.0;
        bool grad_ok = calLtotGrad(
            L1, H_list, pin.refract_array, dL_tot_dL1);
        double L1_new = std::numeric_limits<double>::quiet_NaN();
        if (grad_ok && std::isfinite(dL_tot_dL1) && std::abs(dL_tot_dL1) > proj_tol * 1e-5) {
            L1_new = L1 + residue / dL_tot_dL1;
        }
        if (!std::isfinite(L1_new) || L1_new <= L_lo || L1_new >= L_hi) {
            L1_new = 0.5 * (L_lo + L_hi);
        }
        L1 = L1_new;
    }

    if (!converged) {
        std::get<0>(result) = true;
        std::get<2>(result) = residue;
        std::get<3>(result) = proj_nmax;
        logSnellFail("snell1d_max_iterations_reached", proj_nmax, residue);
        return result;
    }

    // Refresh residual at final iterate for reporting consistency.
    {
        double L_tot_final = 0.0;
        if (calLtot(L1, H_list, pin.refract_array, L_tot_final)
            && std::isfinite(L_tot_final)) {
            residue = L_tot - L_tot_final;
        }
    }

    // Step 4: lift the solved 1D length back to 3D exit point on farthest plate.
    double L_shift = 0.0;
    if (!calProjLen(L1, H1, Hn, pin.refract_array.front(),
                    pin.refract_array.back(), L_shift)) {
        std::get<0>(result) = true;
        std::get<3>(result) = iter + 1;
        logSnellFail("snell1d_total_internal_reflection", iter + 1, residue);
        return result;
    }

    Pt3D pt_exit = pt_pinhole_proj;
    if (L_tot > SMALLNUMBER) {
        const Pt3D in_plane_dir = myMATH::createUnitVector(pt_world_proj, pt_pinhole_proj);
        pt_exit -= in_plane_dir * L_shift;
    }
    pt_exit -= plane_far.norm_vector * thickness_tot;

    std::get<0>(result) = false;
    std::get<1>(result) = pt_exit;
    std::get<2>(result) = std::abs(residue);
    std::get<3>(result) = iter + 1;

    // Success: caller will continue with pinhole projection/distortion.
    return result;
}

} // namespace refraction_snell1d

} // namespace

std::tuple<bool, Pt3D, double, int>
RefractionPinholeCamera::solveProjectionByRefractionLM(
    const Pt3D& pt_world, const PinPlateParam& pin) {
    return refraction_lm::solveProjectionByRefractionLMImpl(pt_world, pin);
}

std::tuple<bool, Pt3D, double, int>
RefractionPinholeCamera::solveProjectionBySnell1D(
    const Pt3D& pt_world, const PinPlateParam& pin) {
    return refraction_snell1d::solveProjectionBySnell1DImpl(
        pt_world, pin);
}


// Project world point using pinplate refraction model.
// Input: world point in physical coordinates.
// Output: distorted image coordinate in pixels after refraction solve.
StatusOr<Pt2D> RefractionPinholeCamera::project(const Pt3D& pt_world,
                                                bool is_print_detail) const {
    try {
        // Step 1: Prefer the dedicated 1D Snell solver for parallel-plate stacks.
        std::tuple<bool, Pt3D, double, int> result =
            solveProjectionBySnell1D(pt_world, _param);
        const bool snell_failed = std::get<0>(result);
        if (snell_failed) {
            // Step 2: Fallback to LM solver for robustness.
            result = solveProjectionByRefractionLM(pt_world, _param);
        }

        if (is_print_detail) {
            std::cout << "RefractionPinholeCamera::project result (snell_failed,is_failed,error): "
                      << snell_failed << "," << std::get<0>(result) << ","
                      << std::get<2>(result) << std::endl;
        }

        // Step 3: Return sentinel for invalid geometry branch.
        if (std::get<0>(result)) {
          // Refraction solve failed.
          // Return a sentinel image point (very large negative values) so
          // normalizeProjectResult() reports a projection failure.
            double value = std::numeric_limits<double>::lowest();
            return normalizeProjectResult(Pt2D(value, value));
        }

        // Step 4: Apply pinhole projection and distortion from refracted point.
        return normalizeProjectResult(RefractionPinholeCamera::distort(
            RefractionPinholeCamera::worldToUndistImg(std::get<1>(result),
                                                      _param),
            _param));
    } catch (const std::exception& e) {
        logRefractionProjectFailure(e.what(), pt_world);
        Status st = mapExceptionToStatus(e, "RefractionPinholeCamera::project");
        return STATUS_OR_ERR_CTX(Pt2D, st.err.code, st.err.message, st.err.context);
    }
}

// Build line-of-sight from one image point through refractive plates.
// Input: distorted image coordinate in pixels.
// Output: line in world coordinates after backward refraction.
StatusOr<Line3D> RefractionPinholeCamera::lineOfSight(const Pt2D& pt_img_dist) const {
    try {
        // Step 1: Undistort image point and build initial pinhole ray.
        Pt2D pt_img_undist = RefractionPinholeCamera::undistort(pt_img_dist, _param);
        Pt3D pt_world(pt_img_undist[0], pt_img_undist[1], 1.0);
        pt_world = _param.r_mtx_inv * pt_world + _param.t_vec_inv;

        Line3D line;
        line.pt = _param.t_vec_inv;
        line.unit_vector = myMATH::createUnitVector(_param.t_vec_inv, pt_world);

        // Step 2: Backward-trace through refractive interfaces.
        // Ordering note: backward tracing always visits nearest -> farthest.
        for (int plane_id = _param.n_plate; plane_id >= 0; plane_id--) {
            const Plane3D& plane = _param.plane_array[plane_id];
            Pt3D pt_cross;
            bool is_parallel = myMATH::crossPoint(pt_cross, line, plane);
            if (is_parallel) {
                throw std::runtime_error("line is parallel to the plane");
            }

            double refract_ratio =
                    _param.refract_array[plane_id + 1] / _param.refract_array[plane_id];
            bool success = refraction_common::refractDirection(
                line.unit_vector, plane.norm_vector, refract_ratio, false);
            if (!success) {
                throw std::runtime_error("total internal reflection");
            }
            line.pt = pt_cross;
        }

        return line;
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "RefractionPinholeCamera::lineOfSight");
        return STATUS_OR_ERR_CTX(Line3D, st.err.code, st.err.message, st.err.context);
    }
}

int RefractionPinholeCamera::getNRow() const { return _param.n_row; }
int RefractionPinholeCamera::getNCol() const { return _param.n_col; }

// Save pinplate parameters to text file.
// Input: output file path.
// Output: file with exactly the legacy pinplate format.
Status RefractionPinholeCamera::saveParameters(const std::string& file_name) const {
    try {
        std::ofstream outfile(file_name.c_str(), std::ios::out);
        setSavePrecision(outfile);
        outfile << "# Camera Model: (PINHOLE/POLYNOMIAL/PINPLATE)" << std::endl;
        outfile << "PINPLATE" << std::endl;
        outfile << "# Camera Calibration Error: \nNone\n# Pose Calibration Error: \nNone"
                        << std::endl;
        outfile << "# Image Size: (n_row,n_col)" << std::endl;
        outfile << _param.n_row << "," << _param.n_col << std::endl;
        outfile << "# Camera Matrix: " << std::endl;
        auto cam_mtx = _param.cam_mtx;
        cam_mtx.write(outfile, kSaveDigits);
        outfile << "# Distortion Coefficients: " << std::endl;
        saveDistCoeffLine(outfile, _param);
        outfile << "# Rotation Vector: " << std::endl;
        auto r_vec = rmtxTorvec(_param.r_mtx);
        auto r_vec_t = r_vec.transpose();
        r_vec_t.write(outfile, kSaveDigits);
        outfile << "# Rotation Matrix: " << std::endl;
        auto r_mtx = _param.r_mtx;
        r_mtx.write(outfile, kSaveDigits);
        outfile << "# Inverse of Rotation Matrix: " << std::endl;
        auto r_mtx_inv = _param.r_mtx_inv;
        r_mtx_inv.write(outfile, kSaveDigits);
        outfile << "# Translation Vector: " << std::endl;
        auto t_vec = _param.t_vec;
        auto t_vec_t = t_vec.transpose();
        t_vec_t.write(outfile, kSaveDigits);
        outfile << "# Inverse of Translation Vector: " << std::endl;
        auto t_vec_inv = _param.t_vec_inv;
        auto t_vec_inv_t = t_vec_inv.transpose();
        t_vec_inv_t.write(outfile, kSaveDigits);
        outfile << "# Reference Point of Refractive Plate: " << std::endl;
        auto plane_pt = _param.plane.pt;
        auto plane_pt_t = plane_pt.transpose();
        plane_pt_t.write(outfile, kSaveDigits);
        outfile << "# Normal Vector of Refractive Plate: " << std::endl;
        auto plane_norm = _param.plane.norm_vector;
        auto plane_norm_t = plane_norm.transpose();
        plane_norm_t.write(outfile, kSaveDigits);
        outfile << "# Refractive Index Array: " << std::endl;
        for (int i = 0; i < _param.n_plate + 1; i++) {
            outfile << _param.refract_array[i] << ",";
        }
        outfile << _param.refract_array[_param.n_plate + 1] << std::endl;
        outfile << "# Width of the Refractive Plate: (same as other physical units)"
                        << std::endl;
        for (int i = 0; i < _param.n_plate - 1; i++) {
            outfile << _param.w_array[i] << ",";
        }
        outfile << _param.w_array[_param.n_plate - 1] << std::endl;
        outfile << "# Projection Tolerance: " << std::endl;
        outfile << _param.proj_tol << std::endl;
        outfile << "# Maximum Number of Iterations for Projection: " << std::endl;
        outfile << _param.proj_nmax << std::endl;
        outfile << "# Optimization Rate: " << std::endl;
        outfile << _param.lr << std::endl;
        return Status::OK();
    } catch (const std::exception& e) {
        return STATUS_ERR_CTX(ErrorCode::IOfailure,
                                                    "RefractionPinholeCamera::saveParameters", e.what());
    }
}

std::shared_ptr<Camera> RefractionPinholeCamera::clone() const {
    return std::make_shared<RefractionPinholeCamera>(*this);
}

Status RefractionPinholeCamera::setImageSize(int n_row, int n_col) {
    REQUIRE(n_row > 0 && n_col > 0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setImageSize invalid image size");
    _param.n_row = n_row;
    _param.n_col = n_col;
    return Status::OK();
}

Status RefractionPinholeCamera::setIntrinsics(double fx, double fy, double cx,
                                                                                            double cy,
                                                                                            const std::vector<double>& dist_coeff) {
    // Main flow:
    // 1) Write intrinsic matrix.
    // 2) Store distortion coefficients.
    // 3) Infer distortion enable flag.
    REQUIRE(fx > 0 && fy > 0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setIntrinsics invalid focal length");
    _param.cam_mtx = Matrix<double>(3, 3, 0.0);
    _param.cam_mtx(0, 0) = fx;
    _param.cam_mtx(1, 1) = fy;
    _param.cam_mtx(0, 2) = cx;
    _param.cam_mtx(1, 2) = cy;
    _param.cam_mtx(2, 2) = 1.0;
    _param.dist_coeff = paddedDistCoeff(dist_coeff);
    _param.n_dist_coeff = static_cast<int>(_param.dist_coeff.size());
    _param.is_distorted = false;
    for (double v : dist_coeff) {
        if (std::abs(v) > SMALLNUMBER) {
            _param.is_distorted = true;
            break;
        }
    }
    return Status::OK();
}

Status RefractionPinholeCamera::setExtrinsics(const Pt3D& rvec, const Pt3D& tvec) {
    // Main flow:
    // 1) Convert Rodrigues vector to rotation matrix.
    // 2) Cache inverse rotation and camera center.
    _param.r_mtx = Rodrigues(rvec);
    _param.t_vec = tvec;
    _param.r_mtx_inv = myMATH::inverse(_param.r_mtx);
    _param.t_vec_inv = (_param.r_mtx_inv * _param.t_vec) * -1.0;
    return Status::OK();
}

// Set refraction model parameters.
// Input:
// - plane_pt / plane_n: farthest refractive interface.
// - refract_array: refractive indices from farthest medium to nearest medium.
// - w_array: plate thicknesses from farthest plate to nearest plate.
// Output: updates refraction buffers and derived ratio bound.
Status RefractionPinholeCamera::setRefraction(const Pt3D& plane_pt,
                                                                                            const Pt3D& plane_n,
                                                                                            const std::vector<double>& refract_array,
                                                                                            const std::vector<double>& w_array) {
    // Main flow:
    // 1) Validate array lengths: refract_array size must be n_plate + 2.
    // 2) Store farthest interface plane and normalize its normal vector.
    // 3) Store refractive stack ordered from farthest to nearest.
    // 4) Precompute refract_ratio_max for radius safety checks.
    REQUIRE(!refract_array.empty(), ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setRefraction empty refract array");
    REQUIRE(refract_array.size() == w_array.size() + 2, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setRefraction size mismatch");

    _param.plane.pt = plane_pt;
    Pt3D plane_n_local = plane_n;
    const double nrm = plane_n_local.norm();
    REQUIRE(nrm > SMALLNUMBER, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setRefraction invalid plane normal");
    _param.plane.norm_vector = plane_n_local / nrm;
    _param.refract_array = refract_array;
    _param.w_array = w_array;
    _param.n_plate = static_cast<int>(w_array.size());
    _param.refract_ratio_max = 0.0;
    for (double n_i : _param.refract_array) {
        _param.refract_ratio_max = std::max(_param.refract_ratio_max,
                                                                                _param.refract_array[0] / n_i);
    }
    return Status::OK();
}

Status RefractionPinholeCamera::setSolverOptions(double proj_tol, int proj_nmax,
                                                                                                  double lr) {
    // Main flow: validate and store projection solver parameters.
    REQUIRE(proj_tol > 0.0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setSolverOptions invalid proj_tol");
    REQUIRE(proj_nmax > 0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setSolverOptions invalid proj_nmax");
    REQUIRE(lr > 0.0, ErrorCode::InvalidArgument,
                    "RefractionPinholeCamera::setSolverOptions invalid lr");
    _param.proj_tol = proj_tol;
    _param.proj_nmax = proj_nmax;
    _param.lr = lr;
    return Status::OK();
}

Status RefractionPinholeCamera::commitUpdate() {
    refraction_common::buildRefractionPlaneStackAndBasis(_param);
    return Status::OK();
}

// Build polynomial derivative coefficient buffers.
// Input: polynomial projection coefficients.
// Output: du/dv coefficient tables for inverse solver Jacobian.
void PolynomialCamera::buildPolyDerivatives(PolyParam& poly) {
    int derivative_id[2] = {1, 2};
    if (poly.ref_plane == REF_X) {
        derivative_id[0] = 2;
        derivative_id[1] = 3;
    } else if (poly.ref_plane == REF_Y) {
        derivative_id[0] = 1;
        derivative_id[1] = 3;
    } else if (poly.ref_plane == REF_Z) {
        derivative_id[0] = 1;
        derivative_id[1] = 2;
    } else {
        throw std::runtime_error("reference plane is wrong");
    }

    poly.du_coeffs = Matrix<double>(poly.n_coeff * 2, 4, 0);
    poly.dv_coeffs = Matrix<double>(poly.n_coeff * 2, 4, 0);
    for (int i = 0; i < poly.n_coeff; i++) {
        poly.du_coeffs(i, 0) = poly.u_coeffs(i, 0) * poly.u_coeffs(i, derivative_id[0]);
        poly.du_coeffs(i, poly.ref_plane) = poly.u_coeffs(i, poly.ref_plane);
        poly.du_coeffs(i, derivative_id[0]) =
                std::max(poly.u_coeffs(i, derivative_id[0]) - 1, 0.0);
        poly.du_coeffs(i, derivative_id[1]) = poly.u_coeffs(i, derivative_id[1]);

        int j = i + poly.n_coeff;
        poly.du_coeffs(j, 0) = poly.u_coeffs(i, 0) * poly.u_coeffs(i, derivative_id[1]);
        poly.du_coeffs(j, poly.ref_plane) = poly.u_coeffs(i, poly.ref_plane);
        poly.du_coeffs(j, derivative_id[0]) = poly.u_coeffs(i, derivative_id[0]);
        poly.du_coeffs(j, derivative_id[1]) =
                std::max(poly.u_coeffs(i, derivative_id[1]) - 1, 0.0);

        poly.dv_coeffs(i, 0) = poly.v_coeffs(i, 0) * poly.v_coeffs(i, derivative_id[0]);
        poly.dv_coeffs(i, poly.ref_plane) = poly.v_coeffs(i, poly.ref_plane);
        poly.dv_coeffs(i, derivative_id[0]) =
                std::max(poly.v_coeffs(i, derivative_id[0]) - 1, 0.0);
        poly.dv_coeffs(i, derivative_id[1]) = poly.v_coeffs(i, derivative_id[1]);

        poly.dv_coeffs(j, 0) = poly.v_coeffs(i, 0) * poly.v_coeffs(i, derivative_id[1]);
        poly.dv_coeffs(j, poly.ref_plane) = poly.v_coeffs(i, poly.ref_plane);
        poly.dv_coeffs(j, derivative_id[0]) = poly.v_coeffs(i, derivative_id[0]);
        poly.dv_coeffs(j, derivative_id[1]) =
                std::max(poly.v_coeffs(i, derivative_id[1]) - 1, 0.0);
    }
}

// Solve inverse polynomial mapping on one fixed reference plane.
// Input: image coordinate + one plane coordinate + polynomial parameters.
// Output: world point on the requested reference plane.
Pt3D PolynomialCamera::solveWorldOnRefPlane(Pt2D const& pt_img_dist,
                                            double plane_world,
                                            PolyParam const& poly) {
    // Initialize one seed point and clamp the reference-axis coordinate.
    Pt3D pt_world((double)rand() / RAND_MAX, (double)rand() / RAND_MAX,
                                (double)rand() / RAND_MAX);

    switch (poly.ref_plane) {
    case REF_X:
        pt_world[0] = plane_world;
        break;
    case REF_Y:
        pt_world[1] = plane_world;
        break;
    case REF_Z:
        pt_world[2] = plane_world;
        break;
    default:
        throw std::runtime_error("unknown reference plane");
    }

    Matrix<double> jacobian(2, 2, 0);
    Matrix<double> jacobian_inv(2, 2, 0);
    Pt2D pt_img_temp;
    double du, dv, dx, dy;
    double err = std::numeric_limits<double>::max();
    int iter = 0;

    while (err > UNDISTORT_EPS && iter < UNDISTORT_MAX_ITER) {
        // Build local Jacobian from derivative coefficient buffers.
        jacobian *= 0;
        jacobian_inv *= 0;

        for (int i = 0; i < poly.n_coeff; i++) {
            double dudx = poly.du_coeffs(i, 0);
            double dudy = poly.du_coeffs(i + poly.n_coeff, 0);
            double dvdx = poly.dv_coeffs(i, 0);
            double dvdy = poly.dv_coeffs(i + poly.n_coeff, 0);

            for (int j = 1; j < 4; j++) {
                dudx *= std::pow(pt_world[j - 1], poly.du_coeffs(i, j));
                dudy *= std::pow(pt_world[j - 1], poly.du_coeffs(i + poly.n_coeff, j));
                dvdx *= std::pow(pt_world[j - 1], poly.dv_coeffs(i, j));
                dvdy *= std::pow(pt_world[j - 1], poly.dv_coeffs(i + poly.n_coeff, j));
            }

            jacobian(0, 0) += dudx;
            jacobian(0, 1) += dudy;
            jacobian(1, 0) += dvdx;
            jacobian(1, 1) += dvdy;
        }

        // Evaluate forward polynomial projection at current estimate.
        jacobian_inv = myMATH::inverse(jacobian, "det");
        pt_img_temp = Pt2D(0.0, 0.0);
        for (int i = 0; i < poly.u_coeffs.getDimRow(); i++) {
            double u_val = poly.u_coeffs(i, 0);
            double v_val = poly.v_coeffs(i, 0);
            for (int j = 1; j < 4; j++) {
                u_val *= std::pow(pt_world[j - 1], poly.u_coeffs(i, j));
                v_val *= std::pow(pt_world[j - 1], poly.v_coeffs(i, j));
            }
            pt_img_temp[0] += u_val;
            pt_img_temp[1] += v_val;
        }
        du = pt_img_dist[0] - pt_img_temp[0];
        dv = pt_img_dist[1] - pt_img_temp[1];
        dx = jacobian_inv(0, 0) * du + jacobian_inv(0, 1) * dv;
        dy = jacobian_inv(1, 0) * du + jacobian_inv(1, 1) * dv;

        // Update the two solved axes while keeping the fixed reference axis.
        switch (poly.ref_plane) {
        case REF_X:
            pt_world[1] += dx;
            pt_world[2] += dy;
            break;
        case REF_Y:
            pt_world[0] += dx;
            pt_world[2] += dy;
            break;
        case REF_Z:
            pt_world[0] += dx;
            pt_world[1] += dy;
            break;
        default:
            throw std::runtime_error("unknown reference plane");
        }

        err = std::sqrt(du * du + dv * dv);
        iter++;
    }

    return pt_world;
}


// Project world point using polynomial model.
// Input: world point in physical coordinates.
// Output: image coordinate in pixels.
StatusOr<Pt2D> PolynomialCamera::project(const Pt3D& pt_world, bool) const {
    try {
        // Main flow:
        // 1) Evaluate polynomial projection for u and v.
        // 2) Return projected image coordinate.
        Pt2D pt_img(0.0, 0.0);
        for (int i = 0; i < _param.u_coeffs.getDimRow(); i++) {
            double u_val = _param.u_coeffs(i, 0);
            double v_val = _param.v_coeffs(i, 0);
            for (int j = 1; j < 4; j++) {
                u_val *= std::pow(pt_world[j - 1], _param.u_coeffs(i, j));
                v_val *= std::pow(pt_world[j - 1], _param.v_coeffs(i, j));
            }
            pt_img[0] += u_val;
            pt_img[1] += v_val;
        }
        return normalizeProjectResult(pt_img);
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "PolynomialCamera::project");
        return STATUS_OR_ERR_CTX(Pt2D, st.err.code, st.err.message, st.err.context);
    }
}

// Build line-of-sight from one image point using polynomial model.
// Input: image coordinate in pixels.
// Output: line in world coordinates.
StatusOr<Line3D> PolynomialCamera::lineOfSight(const Pt2D& pt_img_dist) const {
    try {
        // Step 1: Solve inverse projection on two reference planes.
        Pt3D pt_world_1 = solveWorldOnRefPlane(pt_img_dist, _param.plane[0], _param);
        Pt3D pt_world_2 = solveWorldOnRefPlane(pt_img_dist, _param.plane[1], _param);
        // Step 2: Build line passing through the two recovered world points.
        Line3D line = {pt_world_1, myMATH::createUnitVector(pt_world_1, pt_world_2)};
        return line;
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "PolynomialCamera::lineOfSight");
        return STATUS_OR_ERR_CTX(Line3D, st.err.code, st.err.message, st.err.context);
    }
}

int PolynomialCamera::getNRow() const { return _param.n_row; }
int PolynomialCamera::getNCol() const { return _param.n_col; }

// Save polynomial parameters to text file.
// Input: output file path.
// Output: file with exactly the legacy polynomial format.
Status PolynomialCamera::saveParameters(const std::string& file_name) const {
    try {
        std::ofstream outfile(file_name.c_str(), std::ios::out);
        setSavePrecision(outfile);
        outfile << "# Camera Model: (PINHOLE/POLYNOMIAL)" << std::endl;
        outfile << "POLYNOMIAL" << std::endl;
        outfile << "# Camera Calibration Error: \nNone" << std::endl;
        outfile << "# Image Size: (n_row,n_col)" << std::endl;
        outfile << _param.n_row << "," << _param.n_col << std::endl;
        outfile << "# Reference Plane: (REF_X/REF_Y/REF_Z,coordinate,coordinate)"
                        << std::endl;
        if (_param.ref_plane == REF_X) {
            outfile << "REF_X,";
        } else if (_param.ref_plane == REF_Y) {
            outfile << "REF_Y,";
        } else {
            outfile << "REF_Z,";
        }
        outfile << _param.plane[0] << "," << _param.plane[1] << std::endl;
        outfile << "# Number of Coefficients: " << std::endl;
        outfile << _param.n_coeff << std::endl;
        outfile << "# U_Coeff,X_Power,Y_Power,Z_Power" << std::endl;
        auto u_coeffs = _param.u_coeffs;
        u_coeffs.write(outfile, kSaveDigits);
        outfile << "# V_Coeff,X_Power,Y_Power,Z_Power" << std::endl;
        auto v_coeffs = _param.v_coeffs;
        v_coeffs.write(outfile, kSaveDigits);
        return Status::OK();
    } catch (const std::exception& e) {
        return STATUS_ERR_CTX(ErrorCode::IOfailure,
                                                    "PolynomialCamera::saveParameters", e.what());
    }
}

std::shared_ptr<Camera> PolynomialCamera::clone() const {
    return std::make_shared<PolynomialCamera>(*this);
}

Status PolynomialCamera::setImageSize(int n_row, int n_col) {
    REQUIRE(n_row > 0 && n_col > 0, ErrorCode::InvalidArgument,
                    "PolynomialCamera::setImageSize invalid image size");
    _param.n_row = n_row;
    _param.n_col = n_col;
    return Status::OK();
}

Status PolynomialCamera::setReferencePlane(RefPlane ref_plane, double p0, double p1) {
    _param.ref_plane = ref_plane;
    _param.plane[0] = p0;
    _param.plane[1] = p1;
    return Status::OK();
}

Status PolynomialCamera::setCoefficients(const Matrix<double>& u_coeffs,
                                                                                  const Matrix<double>& v_coeffs) {
    // Main flow:
    // 1) Validate coefficient matrix dimensions.
    // 2) Store u/v coefficients and coefficient count.
    REQUIRE(u_coeffs.getDimRow() == v_coeffs.getDimRow(), ErrorCode::InvalidArgument,
                    "PolynomialCamera::setCoefficients row mismatch");
    REQUIRE(u_coeffs.getDimCol() == 4 && v_coeffs.getDimCol() == 4,
                    ErrorCode::InvalidArgument,
                    "PolynomialCamera::setCoefficients coeff matrix must have 4 cols");
    _param.n_coeff = u_coeffs.getDimRow();
    _param.u_coeffs = u_coeffs;
    _param.v_coeffs = v_coeffs;
    return Status::OK();
}

Status PolynomialCamera::commitUpdate() {
    buildPolyDerivatives(_param);
    return Status::OK();
}

// Load camera model from legacy text format.
// Input: camera parameter file path.
// Output: typed camera model with parameters populated.
// ===========================================================================
// Curved refractive ports (cylinder, sphere)
// ===========================================================================

ImageStatus Camera::imageStatus(const Pt3D& pt_world, double margin_px,
                                double merge_px, Pt2D* pt) const {
    (void)merge_px;
    auto st = project(pt_world);
    if (!st) {
        return ImageStatus::None;
    }
    const Pt2D p = st.value();
    const bool on = p[0] >= -margin_px && p[0] < getNCol() + margin_px &&
                    p[1] >= -margin_px && p[1] < getNRow() + margin_px;
    if (!on) {
        return ImageStatus::None;
    }
    if (pt) {
        *pt = p;
    }
    return ImageStatus::One;
}

// Single-valued models inherit this, so callers can use projectAll() uniformly.
StatusOr<std::vector<ImageRay>> Camera::projectAll(
    const Pt3D& pt_world, ProjectionCoverage* coverage) const {
    if (coverage) {
        coverage->near_branch = true;
        coverage->far_branch = true;
    }
    auto pt = project(pt_world);
    if (!pt) {
        return STATUS_OR_ERR(std::vector<ImageRay>, pt.status().err.code,
                             pt.status().err.message);
    }
    ImageRay ray;
    ray.pt_img_dist = pt.value();
    return std::vector<ImageRay>{ray};
}

namespace {
namespace refraction_curved {

// The core works in a frame whose axis is z through the origin; this is the
// change of basis into it. For a sphere the rotation is the identity, because
// projectSphere() builds its own ray plane.
struct PortFrame {
    Pt3D origin;
    double u[3] = {1.0, 0.0, 0.0};
    double v[3] = {0.0, 1.0, 0.0};
    double w[3] = {0.0, 0.0, 1.0};
};

PortFrame buildFrame(const PinCurvedParam& p, bool is_sphere) {
    PortFrame f;
    f.origin = p.center;
    if (is_sphere) {
        return f;
    }
    Pt3D axis_copy = p.axis; // Matrix::norm() is non-const
    const double len = axis_copy.norm();
    if (len < SMALLNUMBER) {
        return f; // caller validates; fall back to the world axes
    }
    for (int i = 0; i < 3; ++i) {
        f.w[i] = p.axis[i] / len;
    }
    // Any vector not parallel to w gives a usable first tangent.
    const double trial[3] = {std::abs(f.w[0]) < 0.9 ? 1.0 : 0.0,
                             std::abs(f.w[0]) < 0.9 ? 0.0 : 1.0, 0.0};
    const double dotw = trial[0] * f.w[0] + trial[1] * f.w[1] + trial[2] * f.w[2];
    double u[3];
    for (int i = 0; i < 3; ++i) {
        u[i] = trial[i] - dotw * f.w[i];
    }
    const double nu = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (int i = 0; i < 3; ++i) {
        f.u[i] = u[i] / nu;
    }
    f.v[0] = f.w[1] * f.u[2] - f.w[2] * f.u[1];
    f.v[1] = f.w[2] * f.u[0] - f.w[0] * f.u[2];
    f.v[2] = f.w[0] * f.u[1] - f.w[1] * f.u[0];
    return f;
}

void toLocal(const PortFrame& f, const Pt3D& world, double out[3]) {
    const double d[3] = {world[0] - f.origin[0], world[1] - f.origin[1],
                         world[2] - f.origin[2]};
    out[0] = d[0] * f.u[0] + d[1] * f.u[1] + d[2] * f.u[2];
    out[1] = d[0] * f.v[0] + d[1] * f.v[1] + d[2] * f.v[2];
    out[2] = d[0] * f.w[0] + d[1] * f.w[1] + d[2] * f.w[2];
}

void dirToLocal(const PortFrame& f, const Pt3D& world, double out[3]) {
    out[0] = world[0] * f.u[0] + world[1] * f.u[1] + world[2] * f.u[2];
    out[1] = world[0] * f.v[0] + world[1] * f.v[1] + world[2] * f.v[2];
    out[2] = world[0] * f.w[0] + world[1] * f.w[1] + world[2] * f.w[2];
}

Pt3D dirToWorld(const PortFrame& f, const double local[3]) {
    return Pt3D(local[0] * f.u[0] + local[1] * f.v[0] + local[2] * f.w[0],
                local[0] * f.u[1] + local[1] * f.v[1] + local[2] * f.w[1],
                local[0] * f.u[2] + local[1] * f.v[2] + local[2] * f.w[2]);
}

Pt3D pointToWorld(const PortFrame& f, const double local[3]) {
    const Pt3D d = dirToWorld(f, local);
    return Pt3D(d[0] + f.origin[0], d[1] + f.origin[1], d[2] + f.origin[2]);
}

/** @brief Build the RefractCurved stack from the stored parameters. */
::refract_curved::Stack toStack(const PinCurvedParam& p) {
    ::refract_curved::Stack s;
    s.n = p.refract_array;
    s.R = p.radius_array;
    return s;
}

/** @brief Distance of a port-frame point from the axis (cylinder) or the centre (sphere). */
double portRadius(const double p[3], bool is_sphere) {
    return is_sphere ? std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                     : std::hypot(p[0], p[1]);
}

/**
  * @brief Nearest positive intersection of a ray with the quadric of radius R.
  *
  * The ray travels inward, so the smaller positive root is the entry point.
  */
bool hitRadius(const double p[3], const double d[3], double R, bool is_sphere,
               double& t_out) {
    const double a = is_sphere ? (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                               : (d[0] * d[0] + d[1] * d[1]);
    if (a <= 0.0) {
        return false;
    }
    const double b = is_sphere ? 2.0 * (p[0] * d[0] + p[1] * d[1] + p[2] * d[2])
                               : 2.0 * (p[0] * d[0] + p[1] * d[1]);
    const double c = (is_sphere ? (p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                                : (p[0] * p[0] + p[1] * p[1])) -
                     R * R;
    const double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return false;
    }
    const double sq = std::sqrt(disc);
    const double t1 = (-b - sq) / (2.0 * a);
    const double t2 = (-b + sq) / (2.0 * a);
    const double eps = 1e-12;
    if (t1 > eps) {
        t_out = t1;
    } else if (t2 > eps) {
        t_out = t2;
    } else {
        return false;
    }
    return true;
}

/** @brief Vector Snell's law at an outward normal; false on total internal reflection. */
bool refractAt(double d[3], const double nrm[3], double n_from, double n_to) {
    const double eta = n_from / n_to;
    const double cosi = -(nrm[0] * d[0] + nrm[1] * d[1] + nrm[2] * d[2]);
    const double k = 1.0 - eta * eta * (1.0 - cosi * cosi);
    if (k < 0.0) {
        return false;
    }
    const double f = eta * cosi - std::sqrt(k);
    for (int i = 0; i < 3; ++i) {
        d[i] = eta * d[i] + f * nrm[i];
    }
    const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (!(len > 0.0)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        d[i] /= len;
    }
    return true;
}

/**
  * @brief project()'s choice among the images: on the sensor first, then the
  *        near branch, then the smallest |Lambda|; rings skipped (`ring` is set
  *        when there was one). Null when no image carries a pixel.
  */
const ImageRay* primaryImage(const std::vector<ImageRay>& rays, int n_col, int n_row,
                             bool& ring) {
    // The primary image must be picked by a property of the ray, not by the
    // order the search happened to find it in. A point can have two images on
    // the SAME branch; taking "the first" then makes the answer depend on the
    // enumeration order of the targets, so two nearly identical cameras can
    // return the two images in opposite order and appear to disagree by hundreds
    // of pixels while actually agreeing. Prefer the near branch, then the
    // smallest |Lambda| -- the most direct ray, which is geometric and stable.
    //
    // An image on the sensor comes first: next to a caustic the preferred image
    // can fall off the sensor while the other is on it, and a caller testing the
    // pixel would then take a visible point for an unseen one (audit C3). The
    // answer changes only there.
    auto on_sensor = [&](const ImageRay& r) {
        const double u = r.pt_img_dist[0], v = r.pt_img_dist[1];
        return u >= 0.0 && u < n_col && v >= 0.0 && v < n_row;
    };
    auto preferred = [](const ImageRay& a, const ImageRay& b) {
        if (a.is_far != b.is_far) {
            return !a.is_far; // a near-branch image always wins
        }
        return std::abs(a.lambda) < std::abs(b.lambda);
    };
    const ImageRay* best = nullptr;
    bool best_on = false;
    ring = false;
    for (const ImageRay& r : rays) {
        if (r.is_ring) {
            ring = true;
            continue;
        }
        const bool on = on_sensor(r);
        if (best == nullptr || (on && !best_on) ||
            (on == best_on && preferred(r, *best))) {
            best = &r;
            best_on = on;
        }
    }
    return best;
}

/**
  * @brief imageStatus()'s census of a multi-valued model's images `rays`, at
  *        the pixels the camera reports, where `primary` indexes the image
  *        project() returns (-1 for none): None when no image lies on the
  *        sensor enlarged by `margin_px`; One when exactly one does, no other
  *        lies within `merge_px` of it and it is the primary; Ambiguous
  *        otherwise, and for any ring.
  */
ImageStatus statusOfImages(const std::vector<ImageRay>& rays, std::ptrdiff_t primary,
                           int n_col, int n_row, double margin_px, double merge_px,
                           Pt2D* pt) {
    std::ptrdiff_t one = -1;
    int n_on = 0;
    for (std::size_t i = 0; i < rays.size(); ++i) {
        const ImageRay& r = rays[i];
        if (r.is_ring) {
            return ImageStatus::Ambiguous; // a circle of images
        }
        const double u = r.pt_img_dist[0], v = r.pt_img_dist[1];
        if (u >= -margin_px && u < n_col + margin_px && v >= -margin_px &&
            v < n_row + margin_px) {
            ++n_on;
            one = static_cast<std::ptrdiff_t>(i);
        }
    }
    if (n_on == 0) {
        return ImageStatus::None;
    }
    if (n_on > 1) {
        return ImageStatus::Ambiguous;
    }
    const ImageRay& the_one = rays[static_cast<std::size_t>(one)];
    for (std::size_t i = 0; i < rays.size(); ++i) {
        if (static_cast<std::ptrdiff_t>(i) == one) {
            continue;
        }
        const double du = rays[i].pt_img_dist[0] - the_one.pt_img_dist[0];
        const double dv = rays[i].pt_img_dist[1] - the_one.pt_img_dist[1];
        if (du * du + dv * dv <= merge_px * merge_px) {
            return ImageStatus::Ambiguous; // a merged pair next to a fold
        }
    }
    // Tracking draws the point where project() puts it: the one image must be
    // that one (it is, unless it lies in the margin and project() prefers
    // another image off the sensor).
    if (primary != one) {
        return ImageStatus::Ambiguous;
    }
    if (pt) {
        *pt = the_one.pt_img_dist;
    }
    return ImageStatus::One;
}

const char* noImageMessage(bool ring) {
    return ring ? "project failed: the point's only image is a ring (the axial line "
                  "of a sphere); use projectAll()"
                : "project failed: the point has no image (no ray reaches it, or "
                  "every ray leaves behind the camera)";
}

// OPENLPT_PROJECTION_HINTS=0 turns the projection hints off, so that a run can
// be compared with one on the full search alone.
bool projectionHintsEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("OPENLPT_PROJECTION_HINTS");
        return !(v != nullptr && v[0] == '0' && v[1] == '\0');
    }();
    return on;
}

// OPENLPT_CURVED_LINES=0 draws a curved port's image lines straight, as before.
bool curvedLinesEnabled() {
    static const bool on = [] {
        const char* v = std::getenv("OPENLPT_CURVED_LINES");
        return !(v != nullptr && v[0] == '0' && v[1] == '\0');
    }();
    return on;
}

} // namespace refraction_curved
} // namespace

int RefractionCurvedCamera::getNRow() const { return _param.n_row; }
int RefractionCurvedCamera::getNCol() const { return _param.n_col; }

StatusOr<std::vector<ImageRay>> RefractionCurvedCamera::projectAll(
    const Pt3D& pt_world, ProjectionCoverage* coverage) const {
    using namespace refraction_curved;
    try {
        const bool sphere = isSphere();
        const PortFrame frame = buildFrame(_param, sphere);
        const ::refract_curved::Stack stack = toStack(_param);

        double pinhole_local[3], point_local[3];
        toLocal(frame, _param.t_vec_inv, pinhole_local);
        toLocal(frame, pt_world, point_local);

        // The model holds only for a pinhole outside the outermost interface; a
        // camera inside the port is a configuration error, not a missing image.
        if (!stack.R.empty() && !(portRadius(pinhole_local, sphere) > stack.R.front())) {
            return STATUS_OR_ERR(std::vector<ImageRay>, ErrorCode::InvalidArgument,
                                 "RefractionCurvedCamera::projectAll: the pinhole is "
                                 "not outside the outermost interface");
        }

        // Both searches take proj_tol as a dimensionless tolerance (lengths in
        // units of 2^floor(log2 r_P)) and proj_nmax as their iteration cap.
        std::vector<::refract_curved::Ray> rays;
        ::refract_curved::CylinderCoverage cov;
        if (sphere) {
            const double centre_local[3] = {0.0, 0.0, 0.0};
            rays = ::refract_curved::projectSphere(stack, centre_local, pinhole_local,
                                                   point_local, _param.proj_tol,
                                                   _param.proj_nmax);
            // A point not strictly inside the innermost sphere is outside the
            // model: not searched, rather than certified invisible.
            const bool inside = stack.isValid() &&
                                (stack.R.empty() ||
                                 portRadius(point_local, true) < stack.R.back());
            cov.near_branch = inside;
            cov.far_branch = inside;
        } else {
            rays = ::refract_curved::projectCylinder(stack, pinhole_local,
                                                     point_local, &cov,
                                                     _param.proj_tol,
                                                     _param.proj_nmax);
        }
        if (coverage) {
            coverage->near_branch = cov.near_branch;
            coverage->far_branch = cov.far_branch;
        }

        std::vector<ImageRay> out;
        out.reserve(rays.size());
        for (const ::refract_curved::Ray& r : rays) {
            ImageRay img;
            img.is_far = (r.branch == ::refract_curved::Branch::Far);
            img.lambda = r.lambda;
            img.c_z = r.c_z;
            img.is_ring = r.is_ring;
            if (r.is_ring) {
                // A ring is a set of rays, not one; it carries no single pixel.
                // NaN, not (0, 0), which is a real pixel on the sensor's corner.
                const double nan = std::numeric_limits<double>::quiet_NaN();
                img.pt_img_dist = Pt2D(nan, nan);
                out.push_back(img);
                continue;
            }
            // Projection is central at the pinhole, so a point one unit along the
            // ray lands on the same pixel as the ray itself.
            const Pt3D dir_world = dirToWorld(frame, r.dir);
            // A ray that leaves the pinhole away from the viewing direction
            // (depth <= 0 in the camera frame) forms no image.
            const double depth = _param.r_mtx(2, 0) * dir_world[0] +
                                 _param.r_mtx(2, 1) * dir_world[1] +
                                 _param.r_mtx(2, 2) * dir_world[2];
            if (!(depth > 0.0)) {
                continue;
            }
            const Pt3D along(_param.t_vec_inv[0] + dir_world[0],
                             _param.t_vec_inv[1] + dir_world[1],
                             _param.t_vec_inv[2] + dir_world[2]);
            img.pt_img_dist = RefractionCurvedCamera::distort(
                RefractionCurvedCamera::worldToUndistImg(along, _param), _param);
            out.push_back(img);
        }
        return out;
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "RefractionCurvedCamera::projectAll");
        return STATUS_OR_ERR_CTX(std::vector<ImageRay>, st.err.code, st.err.message,
                                 st.err.context);
    }
}

// The primary image: among the images on the sensor, the near-branch one, else
// the smallest |Lambda|; with none on the sensor, the same rule over them all.
// Callers that need the whole census use projectAll().
StatusOr<Pt2D> RefractionCurvedCamera::project(const Pt3D& pt_world,
                                               bool is_print_detail) const {
    auto all = projectAll(pt_world, nullptr);
    if (!all) {
        return STATUS_OR_ERR(Pt2D, all.status().err.code, all.status().err.message);
    }
    const std::vector<ImageRay>& rays = all.value();
    if (is_print_detail) {
        std::cout << "RefractionCurvedCamera::project found " << rays.size()
                  << " image(s)" << std::endl;
    }
    bool ring = false;
    const ImageRay* best =
        refraction_curved::primaryImage(rays, _param.n_col, _param.n_row, ring);
    if (best == nullptr) {
        return STATUS_OR_ERR(Pt2D, ErrorCode::GeometryFailure,
                             refraction_curved::noImageMessage(ring));
    }
    return normalizeProjectResult(best->pt_img_dist);
}

bool RefractionCurvedCamera::usesProjectionHint() const {
    return !isSphere() && refraction_curved::projectionHintsEnabled();
}

StatusOr<Pt2D> RefractionCurvedCamera::projectWithHint(const Pt3D& pt_world,
                                                       ProjectionHint& hint) const {
    using namespace refraction_curved;
    if (isSphere()) {
        return project(pt_world);
    }
    if (hint.valid) {
        try {
            const PortFrame frame = buildFrame(_param, false);
            const ::refract_curved::Stack stack = toStack(_param);
            double pinhole_local[3], point_local[3];
            toLocal(frame, _param.t_vec_inv, pinhole_local);
            toLocal(frame, pt_world, point_local);
            const auto ray = ::refract_curved::projectCylinderFrom(
                stack, pinhole_local, point_local, hint.lambda, hint.c_z, hint.is_far,
                _param.proj_tol, _param.proj_nmax);
            if (ray) {
                // As projectAll(): the pixel of a point one unit along the ray,
                // when the ray leaves the pinhole towards the viewing direction.
                const Pt3D dir_world = dirToWorld(frame, ray->dir);
                const double depth = _param.r_mtx(2, 0) * dir_world[0] +
                                     _param.r_mtx(2, 1) * dir_world[1] +
                                     _param.r_mtx(2, 2) * dir_world[2];
                if (depth > 0.0) {
                    const Pt3D along(_param.t_vec_inv[0] + dir_world[0],
                                     _param.t_vec_inv[1] + dir_world[1],
                                     _param.t_vec_inv[2] + dir_world[2]);
                    StatusOr<Pt2D> px = normalizeProjectResult(
                        RefractionCurvedCamera::distort(
                            RefractionCurvedCamera::worldToUndistImg(along, _param),
                            _param));
                    if (px) {
                        hint.lambda = ray->lambda;
                        hint.c_z = ray->c_z;
                        hint.is_far = (ray->branch == ::refract_curved::Branch::Far);
                        return px;
                    }
                }
            }
        } catch (const std::exception&) {
            // the full search below decides
        }
    }

    auto all = projectAll(pt_world, nullptr);
    if (!all) {
        hint.valid = false;
        return STATUS_OR_ERR(Pt2D, all.status().err.code, all.status().err.message);
    }
    bool ring = false;
    const ImageRay* best = primaryImage(all.value(), _param.n_col, _param.n_row, ring);
    if (best == nullptr) {
        hint.valid = false;
        return STATUS_OR_ERR(Pt2D, ErrorCode::GeometryFailure, noImageMessage(ring));
    }
    hint.lambda = best->lambda;
    hint.c_z = best->c_z;
    hint.is_far = best->is_far;
    hint.valid = true;
    return normalizeProjectResult(best->pt_img_dist);
}

bool RefractionCurvedCamera::provablySingleImage(const Pt3D& pt_world) const {
    using namespace refraction_curved;
    if (isSphere()) {
        return false;
    }
    const PortFrame frame = buildFrame(_param, false);
    const ::refract_curved::Stack stack = toStack(_param);
    double p[3], x[3];
    toLocal(frame, _param.t_vec_inv, p);
    toLocal(frame, pt_world, x);
    return ::refract_curved::singleImage(stack, std::hypot(p[0], p[1]),
                                         std::hypot(x[0], x[1]), x[2] - p[2]);
}

ImageStatus RefractionCurvedCamera::imageStatus(const Pt3D& pt_world, double margin_px,
                                                double merge_px, Pt2D* pt) const {
    if (provablySingleImage(pt_world)) {
        return Camera::imageStatus(pt_world, margin_px, merge_px, pt);
    }
    auto all = projectAll(pt_world, nullptr);
    if (!all) {
        return ImageStatus::None;
    }
    const std::vector<ImageRay>& rays = all.value();
    bool ring = false;
    const ImageRay* primary =
        refraction_curved::primaryImage(rays, _param.n_col, _param.n_row, ring);
    return refraction_curved::statusOfImages(rays, primary ? primary - rays.data() : -1,
                                             _param.n_col, _param.n_row, margin_px,
                                             merge_px, pt);
}

bool RefractionCurvedCamera::lineImagesCurve() const {
    return refraction_curved::curvedLinesEnabled();
}

bool RefractionCurvedCamera::clipToModel(const Line3D& los, double& t_in,
                                         double& t_out) const {
    using namespace refraction_curved;
    if (!(t_in <= t_out)) {
        return false;
    }
    if (_param.radius_array.empty()) {
        return true; // one medium: no interface
    }
    const bool sphere = isSphere();
    const PortFrame frame = buildFrame(_param, sphere);
    double p[3], d[3];
    toLocal(frame, los.pt, p);
    dirToLocal(frame, los.unit_vector, d);
    const double R = _param.radius_array.back() * (1.0 - 1e-9);
    // |p + t d|^2 < R^2 (over x, y for the cylinder): a t^2 + 2 b t + c < 0
    const int dims = sphere ? 3 : 2;
    double a = 0.0, b = 0.0, c = -R * R;
    for (int i = 0; i < dims; ++i) {
        a += d[i] * d[i];
        b += p[i] * d[i];
        c += p[i] * p[i];
    }
    if (!(a > 0.0)) {
        return c < 0.0; // along the axis: inside everywhere or nowhere
    }
    const double disc = b * b - a * c;
    if (!(disc > 0.0)) {
        return false;
    }
    const double sq = std::sqrt(disc);
    // the stable pair of roots
    const double q = -(b + std::copysign(sq, b));
    double t1 = q / a, t2 = (q != 0.0) ? c / q : -b / a;
    if (t1 > t2) {
        std::swap(t1, t2);
    }
    t_in = std::max(t_in, t1);
    t_out = std::min(t_out, t2);
    return t_in <= t_out;
}

bool RefractionCurvedCamera::hintFromLineOfSight(const Line3D& los,
                                                 const Pt3D& pt_world,
                                                 ProjectionHint& hint) const {
    using namespace refraction_curved;
    if (isSphere() || _param.refract_array.empty()) {
        return false;
    }
    const PortFrame frame = buildFrame(_param, false);
    double p[3], d[3], x[3];
    toLocal(frame, los.pt, p);
    dirToLocal(frame, los.unit_vector, d);
    toLocal(frame, pt_world, x);
    const double dd = d[0] * d[0] + d[1] * d[1];
    const double len = std::sqrt(dd + d[2] * d[2]);
    if (!(dd > 0.0) || !(len > 0.0)) {
        return false; // along the axis: no azimuth
    }
    // Both invariants are the same in every medium, so the line of sight's
    // segment gives the ray's: C_z = n d_z, Lambda = n (x d_y - y d_x).
    const double n_in = _param.refract_array.back();
    const double lam = n_in * (p[0] * d[1] - p[1] * d[0]) / len;
    const double c_z = n_in * d[2] / len;
    if (!std::isfinite(lam) || !std::isfinite(c_z)) {
        return false;
    }
    // Along p + t d: the closest approach to the axis, and the point's foot.
    const double t_close = -(p[0] * d[0] + p[1] * d[1]) / dd;
    const double t_pt = ((x[0] - p[0]) * d[0] + (x[1] - p[1]) * d[1] +
                         (x[2] - p[2]) * d[2]) / (len * len);
    hint.lambda = lam;
    hint.c_z = c_z;
    hint.is_far = t_pt > t_close;
    hint.valid = true;
    return true;
}

StatusOr<Line3D> RefractionCurvedCamera::lineOfSight(const Pt2D& pt_img_dist) const {
    using namespace refraction_curved;
    try {
        // Step 1: the straight ray leaving the pinhole, in world coordinates.
        Pt2D pt_img_undist =
            RefractionCurvedCamera::undistort(pt_img_dist, _param);
        Pt3D pt_world(pt_img_undist[0], pt_img_undist[1], 1.0);
        pt_world = _param.r_mtx_inv * pt_world + _param.t_vec_inv;

        const bool sphere = isSphere();
        const PortFrame frame = buildFrame(_param, sphere);

        double p[3], d[3];
        toLocal(frame, _param.t_vec_inv, p);
        // As in projectAll(): the interfaces are crossed from outside in, which
        // needs the pinhole outside the outermost one.
        if (!_param.radius_array.empty() &&
            !(portRadius(p, sphere) > _param.radius_array.front())) {
            return STATUS_OR_ERR(Line3D, ErrorCode::InvalidArgument,
                                 "RefractionCurvedCamera::lineOfSight: the pinhole is "
                                 "not outside the outermost interface");
        }
        const Pt3D dir_world = myMATH::createUnitVector(_param.t_vec_inv, pt_world);
        dirToLocal(frame, dir_world, d);

        // Step 2: cross each interface in turn, outermost first. Unlike the
        // forward direction this is single-valued and needs no solve.
        for (std::size_t k = 0; k < _param.radius_array.size(); ++k) {
            double t = 0.0;
            if (!hitRadius(p, d, _param.radius_array[k], sphere, t)) {
                throw std::runtime_error("ray misses a refractive interface");
            }
            for (int i = 0; i < 3; ++i) {
                p[i] += t * d[i];
            }
            const double r = sphere ? std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                                    : std::hypot(p[0], p[1]);
            if (!(r > 0.0)) {
                throw std::runtime_error("degenerate interface crossing");
            }
            const double nrm[3] = {p[0] / r, p[1] / r, sphere ? p[2] / r : 0.0};
            if (!refractAt(d, nrm, _param.refract_array[k],
                           _param.refract_array[k + 1])) {
                throw std::runtime_error("total internal reflection");
            }
        }

        // Step 3: inside the innermost medium there is no further interface, so
        // the ray is straight and this segment is the whole line of sight.
        Line3D line;
        line.pt = pointToWorld(frame, p);
        line.unit_vector = dirToWorld(frame, d);
        return line;
    } catch (const std::exception& e) {
        Status st = mapExceptionToStatus(e, "RefractionCurvedCamera::lineOfSight");
        return STATUS_OR_ERR_CTX(Line3D, st.err.code, st.err.message, st.err.context);
    }
}

Status RefractionCurvedCamera::setImageSize(int n_row, int n_col) {
    REQUIRE(n_row > 0 && n_col > 0, ErrorCode::InvalidArgument,
            "RefractionCurvedCamera::setImageSize invalid image size");
    _param.n_row = n_row;
    _param.n_col = n_col;
    return Status::OK();
}

Status RefractionCurvedCamera::setIntrinsics(double fx, double fy, double cx,
                                             double cy,
                                             const std::vector<double>& dist_coeff) {
    REQUIRE(fx > 0 && fy > 0, ErrorCode::InvalidArgument,
            "RefractionCurvedCamera::setIntrinsics invalid focal length");
    _param.cam_mtx = Matrix<double>(3, 3, 0.0);
    _param.cam_mtx(0, 0) = fx;
    _param.cam_mtx(1, 1) = fy;
    _param.cam_mtx(0, 2) = cx;
    _param.cam_mtx(1, 2) = cy;
    _param.cam_mtx(2, 2) = 1.0;
    _param.dist_coeff = paddedDistCoeff(dist_coeff);
    _param.n_dist_coeff = static_cast<int>(_param.dist_coeff.size());
    _param.is_distorted = false;
    for (double v : dist_coeff) {
        if (std::abs(v) > SMALLNUMBER) {
            _param.is_distorted = true;
            break;
        }
    }
    return Status::OK();
}

Status RefractionCurvedCamera::setExtrinsics(const Pt3D& rvec, const Pt3D& tvec) {
    _param.r_mtx = Rodrigues(rvec);
    _param.t_vec = tvec;
    _param.r_mtx_inv = myMATH::inverse(_param.r_mtx);
    _param.t_vec_inv = (_param.r_mtx_inv * _param.t_vec) * -1.0;
    return Status::OK();
}

Status RefractionCurvedCamera::setRefraction(const Pt3D& center, const Pt3D& axis,
                                             const std::vector<double>& refract_array,
                                             const std::vector<double>& radius_array) {
    REQUIRE(refract_array.size() >= 2, ErrorCode::InvalidArgument,
            "RefractionCurvedCamera::setRefraction needs at least two media");
    REQUIRE(radius_array.size() + 1 == refract_array.size(),
            ErrorCode::InvalidArgument,
            "RefractionCurvedCamera::setRefraction needs one radius fewer than media");
    for (double n : refract_array) {
        REQUIRE(n > 0.0, ErrorCode::InvalidArgument,
                "RefractionCurvedCamera::setRefraction non-positive refractive index");
    }
    for (std::size_t k = 0; k < radius_array.size(); ++k) {
        REQUIRE(radius_array[k] > 0.0, ErrorCode::InvalidArgument,
                "RefractionCurvedCamera::setRefraction non-positive radius");
        REQUIRE(k == 0 || radius_array[k - 1] > radius_array[k],
                ErrorCode::InvalidArgument,
                "RefractionCurvedCamera::setRefraction radii must strictly decrease");
    }
    Pt3D axis_copy = axis; // Matrix::norm() is non-const
    const double axis_len = axis_copy.norm();
    if (!isSphere()) {
        REQUIRE(axis_len > SMALLNUMBER, ErrorCode::InvalidArgument,
                "RefractionCurvedCamera::setRefraction axis is near zero");
    }
    _param.center = center;
    _param.axis = axis;
    if (axis_len > SMALLNUMBER) {
        _param.axis /= axis_len;
    }
    _param.refract_array = refract_array;
    _param.radius_array = radius_array;
    return Status::OK();
}

Status RefractionCurvedCamera::setSolverOptions(double proj_tol, int proj_nmax) {
    REQUIRE(proj_tol > 0, ErrorCode::InvalidArgument,
            "RefractionCurvedCamera::setSolverOptions invalid proj_tol");
    REQUIRE(proj_nmax > 0, ErrorCode::InvalidArgument,
            "RefractionCurvedCamera::setSolverOptions invalid proj_nmax");
    _param.proj_tol = proj_tol;
    _param.proj_nmax = proj_nmax;
    return Status::OK();
}

Status RefractionCurvedCamera::commitUpdate() {
    _param.r_mtx_inv = myMATH::inverse(_param.r_mtx);
    _param.t_vec_inv = (_param.r_mtx_inv * _param.t_vec) * -1.0;
    const double axis_len = _param.axis.norm();
    if (axis_len > SMALLNUMBER) {
        _param.axis /= axis_len;
    }
    return Status::OK();
}

Status RefractionCurvedCamera::saveCurvedParameters(
    const std::string& file_name) const {
    try {
        std::ofstream outfile(file_name.c_str(), std::ios::out);
        setSavePrecision(outfile);
        outfile << "# Camera Model: (PINHOLE/POLYNOMIAL/PINPLATE/PINCYL/PINSPH)"
                << std::endl;
        outfile << typeToken() << std::endl;
        outfile << "# Camera Calibration Error: \nNone\n# Pose Calibration Error: \nNone"
                << std::endl;
        outfile << "# Image Size: (n_row,n_col)" << std::endl;
        outfile << _param.n_row << "," << _param.n_col << std::endl;
        outfile << "# Camera Matrix: " << std::endl;
        auto cam_mtx = _param.cam_mtx;
        cam_mtx.write(outfile, kSaveDigits);
        outfile << "# Distortion Coefficients: " << std::endl;
        saveDistCoeffLine(outfile, _param);
        outfile << "# Rotation Vector: " << std::endl;
        auto r_vec_t = rmtxTorvec(_param.r_mtx).transpose();
        r_vec_t.write(outfile, kSaveDigits);
        outfile << "# Rotation Matrix: " << std::endl;
        auto r_mtx = _param.r_mtx;
        r_mtx.write(outfile, kSaveDigits);
        outfile << "# Inverse of Rotation Matrix: " << std::endl;
        auto r_mtx_inv = _param.r_mtx_inv;
        r_mtx_inv.write(outfile, kSaveDigits);
        // Matrix::transpose() is non-const, so each vector is copied first.
        outfile << "# Translation Vector: " << std::endl;
        Pt3D t_vec_copy = _param.t_vec;
        auto t_vec_t = t_vec_copy.transpose();
        t_vec_t.write(outfile, kSaveDigits);
        outfile << "# Inverse of Translation Vector: " << std::endl;
        Pt3D t_vec_inv_copy = _param.t_vec_inv;
        auto t_vec_inv_t = t_vec_inv_copy.transpose();
        t_vec_inv_t.write(outfile, kSaveDigits);
        outfile << "# Port Center (axis point for a cylinder, centre for a sphere): "
                << std::endl;
        Pt3D center_copy = _param.center;
        auto center_t = center_copy.transpose();
        center_t.write(outfile, kSaveDigits);
        outfile << "# Port Axis (unit; unused by a sphere): " << std::endl;
        Pt3D axis_copy = _param.axis;
        auto axis_t = axis_copy.transpose();
        axis_t.write(outfile, kSaveDigits);
        outfile << "# Refractive Index Array: (pinhole medium first)" << std::endl;
        for (std::size_t i = 0; i + 1 < _param.refract_array.size(); ++i) {
            outfile << _param.refract_array[i] << ",";
        }
        outfile << _param.refract_array.back() << std::endl;
        outfile << "# Interface Radii: (strictly decreasing)" << std::endl;
        for (std::size_t i = 0; i + 1 < _param.radius_array.size(); ++i) {
            outfile << _param.radius_array[i] << ",";
        }
        outfile << _param.radius_array.back() << std::endl;
        outfile << "# Projection Tolerance: " << std::endl;
        outfile << _param.proj_tol << std::endl;
        outfile << "# Maximum Number of Iterations for Projection: " << std::endl;
        outfile << _param.proj_nmax << std::endl;
        return Status::OK();
    } catch (const std::exception& e) {
        return STATUS_ERR_CTX(ErrorCode::IOfailure,
                              "RefractionCurvedCamera::saveParameters", e.what());
    }
}

Status RefractionCylinderCamera::saveParameters(const std::string& file_name) const {
    return saveCurvedParameters(file_name);
}

std::shared_ptr<Camera> RefractionCylinderCamera::clone() const {
    return std::make_shared<RefractionCylinderCamera>(*this);
}

Status RefractionSphereCamera::saveParameters(const std::string& file_name) const {
    return saveCurvedParameters(file_name);
}

std::shared_ptr<Camera> RefractionSphereCamera::clone() const {
    return std::make_shared<RefractionSphereCamera>(*this);
}

StatusOr<std::shared_ptr<Camera>> CameraFactory::loadFromFile(
        const std::string& file_name) {
    try {
        std::stringstream file_content;
        loadFileToStream(file_name, file_content);

        std::string type_name;
        file_content >> type_name;

        if (type_name == "PINHOLE") {
            auto cam = std::make_shared<PinholeCamera>();
            std::string useless;
            file_content >> useless;
            file_content >> useless;

            std::string img_size_str;
            file_content >> img_size_str;
            std::stringstream img_size_stream(img_size_str);
            std::string temp;
            std::getline(img_size_stream, temp, ',');
            cam->param().n_row = std::stoi(temp);
            std::getline(img_size_stream, temp, ',');
            cam->param().n_col = std::stoi(temp);

            cam->param().cam_mtx = Matrix<double>(3, 3, file_content);

            std::string dist_coeff_str;
            file_content >> dist_coeff_str;
            loadDistCoeff(dist_coeff_str, cam->param());

            file_content >> useless; // rvec, ignored for compatibility
            cam->param().r_mtx = Matrix<double>(3, 3, file_content);
            cam->param().r_mtx_inv = Matrix<double>(3, 3, file_content);
            cam->param().t_vec = Pt3D(file_content);
            cam->param().t_vec_inv = Pt3D(file_content);
            return std::static_pointer_cast<Camera>(cam);
        }

        if (type_name == "POLYNOMIAL") {
            auto cam = std::make_shared<PolynomialCamera>();
            std::string useless;
            file_content >> useless;

            std::string img_size_str;
            file_content >> img_size_str;
            std::stringstream img_size_stream(img_size_str);
            std::string temp;
            std::getline(img_size_stream, temp, ',');
            cam->param().n_row = std::stoi(temp);
            std::getline(img_size_stream, temp, ',');
            cam->param().n_col = std::stoi(temp);

            std::string ref_plane_str;
            file_content >> ref_plane_str;
            std::stringstream plane_stream(ref_plane_str);
            std::getline(plane_stream, temp, ',');
            if (temp == "REF_X") {
                cam->param().ref_plane = REF_X;
            } else if (temp == "REF_Y") {
                cam->param().ref_plane = REF_Y;
            } else if (temp == "REF_Z") {
                cam->param().ref_plane = REF_Z;
            } else {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::UnsupportedType,
                                                          "reference plane is wrong");
            }
            std::getline(plane_stream, temp, ',');
            cam->param().plane[0] = std::stod(temp);
            std::getline(plane_stream, temp, ',');
            cam->param().plane[1] = std::stod(temp);

            file_content >> cam->param().n_coeff;
            cam->param().u_coeffs = Matrix<double>(cam->param().n_coeff, 4, file_content);
            cam->param().v_coeffs = Matrix<double>(cam->param().n_coeff, 4, file_content);
            auto st = cam->commitUpdate();
            if (!st) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, st.err.code,
                                     st.err.message);
            }
            return std::static_pointer_cast<Camera>(cam);
        }

        if (type_name == "PINPLATE") {
            auto cam = std::make_shared<RefractionPinholeCamera>();
            // Read and ignore calibration-error tokens to keep parsing aligned with
            // PINHOLE-style header:
            //   # Camera Calibration Error:
            //   <mean,std>
            //   # Pose Calibration Error:
            //   <mean,std>
            // Legacy pinplate files that used placeholder numeric tokens remain
            // compatible because they still provide two tokens here.
            std::string calib_err_token;
            std::string pose_err_token;
            file_content >> calib_err_token;
            file_content >> pose_err_token;

            std::string img_size_str;
            file_content >> img_size_str;
            std::stringstream img_size_stream(img_size_str);
            std::string temp;
            std::getline(img_size_stream, temp, ',');
            cam->param().n_row = std::stoi(temp);
            std::getline(img_size_stream, temp, ',');
            cam->param().n_col = std::stoi(temp);

            cam->param().cam_mtx = Matrix<double>(3, 3, file_content);
            std::string dist_coeff_str;
            file_content >> dist_coeff_str;
            loadDistCoeff(dist_coeff_str, cam->param());

            std::string rvec_token;
            file_content >> rvec_token; // rvec, ignored for compatibility
            cam->param().r_mtx = Matrix<double>(3, 3, file_content);
            cam->param().r_mtx_inv = Matrix<double>(3, 3, file_content);
            cam->param().t_vec = Pt3D(file_content);
            cam->param().t_vec_inv = Pt3D(file_content);

            cam->param().plane.pt = Pt3D(file_content);
            cam->param().plane.norm_vector = Pt3D(file_content);
            {
                double nrm = cam->param().plane.norm_vector.norm();
                if (nrm < 1e-12) {
                    return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::InvalidArgument,
                                                              "plane normal is near zero");
                }
                cam->param().plane.norm_vector /= nrm;
            }

            std::string refract_str;
            file_content >> refract_str;
            std::stringstream refract_stream(refract_str);
            double refract;
            while (refract_stream >> refract) {
                cam->param().refract_array.push_back(refract);
                if (refract_stream.peek() == ',') {
                    refract_stream.ignore();
                }
            }

            cam->param().refract_ratio_max = 0.0;
            for (int i = 0; i < static_cast<int>(cam->param().refract_array.size()); i++) {
                cam->param().refract_ratio_max =
                        std::max(cam->param().refract_ratio_max,
                                          cam->param().refract_array[0] / cam->param().refract_array[i]);
            }

            std::string w_str;
            file_content >> w_str;
            std::stringstream w_stream(w_str);
            double w;
            while (w_stream >> w) {
                cam->param().w_array.push_back(w);
                if (w_stream.peek() == ',') {
                    w_stream.ignore();
                }
            }

            if (cam->param().refract_array.size() != cam->param().w_array.size() + 2 ||
                    cam->param().refract_array.size() < 3) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::InvalidArgument,
                                                          "number of refractive index and width is not consistent");
            }
            file_content >> cam->param().proj_tol;
            file_content >> cam->param().proj_nmax;
            if (!(file_content >> cam->param().lr)) {
                cam->param().lr = 0.1;
            }

            cam->param().n_plate = static_cast<int>(cam->param().w_array.size());
            auto st = cam->commitUpdate();
            if (!st) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, st.err.code,
                                     st.err.message);
            }

            return std::static_pointer_cast<Camera>(cam);
        }

        if (type_name == "PINCYL" || type_name == "PINSPH") {
            std::shared_ptr<RefractionCurvedCamera> cam;
            if (type_name == "PINCYL") {
                cam = std::make_shared<RefractionCylinderCamera>();
            } else {
                cam = std::make_shared<RefractionSphereCamera>();
            }
            // Same header layout as PINPLATE: two calibration-error tokens.
            std::string useless;
            file_content >> useless;
            file_content >> useless;

            std::string img_size_str;
            file_content >> img_size_str;
            std::stringstream img_size_stream(img_size_str);
            std::string temp;
            std::getline(img_size_stream, temp, ',');
            cam->param().n_row = std::stoi(temp);
            std::getline(img_size_stream, temp, ',');
            cam->param().n_col = std::stoi(temp);

            cam->param().cam_mtx = Matrix<double>(3, 3, file_content);
            std::string dist_coeff_str;
            file_content >> dist_coeff_str;
            loadDistCoeff(dist_coeff_str, cam->param());

            file_content >> useless; // rvec, ignored for compatibility
            cam->param().r_mtx = Matrix<double>(3, 3, file_content);
            cam->param().r_mtx_inv = Matrix<double>(3, 3, file_content);
            cam->param().t_vec = Pt3D(file_content);
            cam->param().t_vec_inv = Pt3D(file_content);

            cam->param().center = Pt3D(file_content);
            cam->param().axis = Pt3D(file_content);

            auto read_csv = [&file_content](std::vector<double>& out) {
                std::string line;
                file_content >> line;
                std::stringstream ss(line);
                double v;
                while (ss >> v) {
                    out.push_back(v);
                    if (ss.peek() == ',') {
                        ss.ignore();
                    }
                }
            };
            read_csv(cam->param().refract_array);
            read_csv(cam->param().radius_array);

            if (cam->param().refract_array.size() < 2 ||
                cam->param().radius_array.size() + 1 !=
                    cam->param().refract_array.size()) {
                return STATUS_OR_ERR(
                    std::shared_ptr<Camera>, ErrorCode::InvalidArgument,
                    "number of refractive indices and radii is not consistent");
            }
            // The same checks as setRefraction().
            for (double n : cam->param().refract_array) {
                if (!(n > 0.0) || !std::isfinite(n)) {
                    return STATUS_OR_ERR(std::shared_ptr<Camera>,
                                         ErrorCode::InvalidArgument,
                                         "refractive indices must be positive and finite");
                }
            }
            for (double R : cam->param().radius_array) {
                if (!(R > 0.0) || !std::isfinite(R)) {
                    return STATUS_OR_ERR(std::shared_ptr<Camera>,
                                         ErrorCode::InvalidArgument,
                                         "interface radii must be positive and finite");
                }
            }
            for (std::size_t k = 1; k < cam->param().radius_array.size(); ++k) {
                if (!(cam->param().radius_array[k - 1] > cam->param().radius_array[k])) {
                    return STATUS_OR_ERR(std::shared_ptr<Camera>,
                                         ErrorCode::InvalidArgument,
                                         "interface radii must strictly decrease");
                }
            }
            if (type_name == "PINCYL" && cam->param().axis.norm() < SMALLNUMBER) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>,
                                     ErrorCode::InvalidArgument,
                                     "cylinder axis is near zero");
            }

            file_content >> cam->param().proj_tol;
            if (!(file_content >> cam->param().proj_nmax)) {
                cam->param().proj_nmax = 80;
            }
            // The same checks as setSolverOptions().
            if (!(cam->param().proj_tol > 0.0) || cam->param().proj_nmax <= 0) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::InvalidArgument,
                                     "invalid proj_tol or proj_nmax");
            }

            auto st = cam->commitUpdate();
            if (!st) {
                return STATUS_OR_ERR(std::shared_ptr<Camera>, st.err.code,
                                     st.err.message);
            }
            return std::static_pointer_cast<Camera>(cam);
        }

        return STATUS_OR_ERR(std::shared_ptr<Camera>, ErrorCode::UnsupportedType,
                                                  "CameraFactory::loadFromFile unknown camera type");
    } catch (const std::exception& e) {
        return STATUS_OR_ERR_CTX(std::shared_ptr<Camera>, ErrorCode::IOfailure,
                                                          "CameraFactory::loadFromFile failed", e.what());
    }
}

// ===========================================================================
// Image corrections (VSC's field) and the corrected camera
// ===========================================================================

StatusOr<std::shared_ptr<const ImageCorrection>> ImageCorrection::loadFromFile(
    const std::string& file_name) {
    using Ptr = std::shared_ptr<const ImageCorrection>;
    std::ifstream in(file_name);
    if (!in) {
        return STATUS_OR_ERR_CTX(Ptr, ErrorCode::IOfailure,
                                 "cannot open the image correction file", file_name);
    }
    // Comments out, then a stream of keywords and numbers.
    std::stringstream body;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line.erase(hash);
        }
        body << line << '\n';
    }
    auto fail = [&](const std::string& what) {
        return STATUS_OR_ERR_CTX(Ptr, ErrorCode::InvalidArgument,
                                 "invalid image correction file: " + what, file_name);
    };
    std::string key;
    int version = 0;
    if (!(body >> key) || key != "VERSION" || !(body >> version) || version != 1) {
        return fail("expected VERSION 1");
    }
    int n_col = 0, n_row = 0;
    if (!(body >> key) || key != "IMAGE_SIZE" || !(body >> n_col >> n_row)) {
        return fail("expected IMAGE_SIZE n_col n_row");
    }
    double grid_u0 = 0.0, grid_v0 = 0.0, grid_step = 0.0;
    int grid_nu = 0, grid_nv = 0;
    if (!(body >> key) || key != "GRID" || !(body >> grid_u0 >> grid_v0 >> grid_step >> grid_nu >> grid_nv)) {
        return fail("expected GRID u0 v0 step nu nv");
    }
    if (grid_nu < 2 || grid_nv < 2 || grid_nu > 100000 || grid_nv > 100000) {
        return fail("the grid needs 2 to 100000 nodes each way");
    }
    const std::size_t n = static_cast<std::size_t>(grid_nu) * static_cast<std::size_t>(grid_nv);
    std::vector<double> du(n), dv(n);
    if (!(body >> key) || key != "DU") {
        return fail("expected DU");
    }
    for (double& x : du) {
        if (!(body >> x)) {
            return fail("DU needs nu * nv numbers");
        }
    }
    if (!(body >> key) || key != "DV") {
        return fail("expected DV after nu * nv numbers of DU");
    }
    for (double& x : dv) {
        if (!(body >> x)) {
            return fail("DV needs nu * nv numbers");
        }
    }
    if (body >> key) {
        return fail("unexpected text after DV: " + key);
    }
    return fromGrid(n_col, n_row, grid_u0, grid_v0, grid_step, grid_nu, grid_nv, du, dv, file_name);
}

StatusOr<std::shared_ptr<const ImageCorrection>> ImageCorrection::fromGrid(
    int n_col, int n_row, double grid_u0, double grid_v0, double grid_step, int grid_nu, int grid_nv,
    const std::vector<double>& du, const std::vector<double>& dv,
    const std::string& source) {
    using Ptr = std::shared_ptr<const ImageCorrection>;
    auto fail = [&](const std::string& what) {
        return STATUS_OR_ERR_CTX(Ptr, ErrorCode::InvalidArgument,
                                 "invalid image correction: " + what, source);
    };
    if (n_col <= 0 || n_row <= 0) {
        return fail("the image size must be positive");
    }
    if (!std::isfinite(grid_u0) || !std::isfinite(grid_v0) || !std::isfinite(grid_step) || !(grid_step > 0.0)) {
        return fail("the grid's origin and step must be finite, and the step positive");
    }
    if (grid_nu < 2 || grid_nv < 2 || grid_nu > 100000 || grid_nv > 100000) {
        return fail("the grid needs 2 to 100000 nodes each way");
    }
    const std::size_t n = static_cast<std::size_t>(grid_nu) * static_cast<std::size_t>(grid_nv);
    if (du.size() != n || dv.size() != n) {
        return fail("du and dv need nu * nv values each");
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (!std::isfinite(du[k]) || !std::isfinite(dv[k])) {
            return fail("a value is not finite (node " + std::to_string(k) + ")");
        }
    }
    // In a cell, the derivative of the bilinear field along u lies between its
    // values on the cell's two edges, and likewise along v; the Frobenius norm
    // of the largest entries bounds the derivative's norm. Beyond the grid the
    // field is its edge value, which only drops derivatives.
    double slope = 0.0;
    for (int j = 0; j + 1 < grid_nv; ++j) {
        for (int i = 0; i + 1 < grid_nu; ++i) {
            const std::size_t k00 = static_cast<std::size_t>(j) * grid_nu + i;
            const std::size_t k10 = k00 + 1, k01 = k00 + grid_nu, k11 = k01 + 1;
            double g = 0.0;
            for (const std::vector<double>* a : {&du, &dv}) {
                const std::vector<double>& c = *a;
                const double gu =
                    std::max(std::abs(c[k10] - c[k00]), std::abs(c[k11] - c[k01])) / grid_step;
                const double gv =
                    std::max(std::abs(c[k01] - c[k00]), std::abs(c[k11] - c[k10])) / grid_step;
                g += gu * gu + gv * gv;
            }
            slope = std::max(slope, std::sqrt(g));
        }
    }
    if (!(slope < 0.5)) {
        return fail("too steep: its slope bound is " + std::to_string(slope) +
                    "; it must be below 0.5");
    }
    std::shared_ptr<ImageCorrection> c(new ImageCorrection());
    c->_n_col = n_col;
    c->_n_row = n_row;
    c->_u0 = grid_u0;
    c->_v0 = grid_v0;
    c->_step = grid_step;
    c->_nu = grid_nu;
    c->_nv = grid_nv;
    c->_du = du;
    c->_dv = dv;
    c->_slope = slope;
    c->_source = source;
    return Ptr(c);
}

void ImageCorrection::eval(double u, double v, double& cu, double& cv) const {
    // Grid coordinates, held to the grid: beyond it, the edge value.
    double s = (u - _u0) / _step;
    double t = (v - _v0) / _step;
    s = s < 0.0 ? 0.0 : (s > _nu - 1 ? static_cast<double>(_nu - 1) : s);
    t = t < 0.0 ? 0.0 : (t > _nv - 1 ? static_cast<double>(_nv - 1) : t);
    const int i = std::min(static_cast<int>(s), _nu - 2);
    const int j = std::min(static_cast<int>(t), _nv - 2);
    const double fs = s - i, ft = t - j;
    const std::size_t k00 = static_cast<std::size_t>(j) * _nu + i;
    const std::size_t k10 = k00 + 1, k01 = k00 + _nu, k11 = k01 + 1;
    const double w00 = (1.0 - fs) * (1.0 - ft), w10 = fs * (1.0 - ft);
    const double w01 = (1.0 - fs) * ft, w11 = fs * ft;
    cu = w00 * _du[k00] + w10 * _du[k10] + w01 * _du[k01] + w11 * _du[k11];
    cv = w00 * _dv[k00] + w10 * _dv[k10] + w01 * _dv[k01] + w11 * _dv[k11];
}

Pt2D ImageCorrection::at(const Pt2D& pt) const {
    const double u = pt[0], v = pt[1];
    if (!std::isfinite(u) || !std::isfinite(v)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return Pt2D(nan, nan);
    }
    double cu = 0.0, cv = 0.0;
    eval(u, v, cu, cv);
    return Pt2D(cu, cv);
}

Pt2D ImageCorrection::toModel(const Pt2D& pt_img) const {
    const double u = pt_img[0], v = pt_img[1];
    if (!std::isfinite(u) || !std::isfinite(v)) {
        return pt_img;
    }
    double cu = 0.0, cv = 0.0;
    eval(u, v, cu, cv);
    return Pt2D(u - cu, v - cv);
}

Pt2D ImageCorrection::toImage(const Pt2D& pt_model) const {
    const double qu = pt_model[0], qv = pt_model[1];
    if (!std::isfinite(qu) || !std::isfinite(qv)) {
        return pt_model;
    }
    double cu = 0.0, cv = 0.0;
    eval(qu, qv, cu, cv);
    double du = qu + cu, dv = qv + cv;
    // d <- q + c(d) shrinks its error by slopeBound() < 1/2 per step, so 60
    // steps take any start of the field's size well below the tolerance.
    for (int k = 0; k < 60; ++k) {
        eval(du, dv, cu, cv);
        const double next_u = qu + cu, next_v = qv + cv;
        const double change = std::abs(next_u - du) + std::abs(next_v - dv);
        du = next_u;
        dv = next_v;
        if (change <= 1e-11) {
            break;
        }
    }
    return Pt2D(du, dv);
}

CorrectedCamera::CorrectedCamera(std::shared_ptr<Camera> inner,
                                 std::shared_ptr<const ImageCorrection> correction)
    : _inner(std::move(inner)), _correction(std::move(correction)) {
    is_active = _inner->is_active;
    max_intensity = _inner->max_intensity;
}

StatusOr<std::shared_ptr<CorrectedCamera>> CorrectedCamera::create(
    std::shared_ptr<Camera> inner, std::shared_ptr<const ImageCorrection> correction) {
    using Ptr = std::shared_ptr<CorrectedCamera>;
    if (!inner || !correction) {
        return STATUS_OR_ERR(Ptr, ErrorCode::InvalidArgument,
                             "CorrectedCamera needs a camera and a correction");
    }
    if (correction->nCol() != inner->getNCol() || correction->nRow() != inner->getNRow()) {
        return STATUS_OR_ERR_CTX(
            Ptr, ErrorCode::InvalidArgument,
            "the correction's image size (" + std::to_string(correction->nCol()) + " x " +
                std::to_string(correction->nRow()) + ") is not the camera's (" +
                std::to_string(inner->getNCol()) + " x " + std::to_string(inner->getNRow()) +
                ")",
            correction->source());
    }
    return Ptr(new CorrectedCamera(std::move(inner), std::move(correction)));
}

StatusOr<Pt2D> CorrectedCamera::project(const Pt3D& pt_world, bool is_print_detail) const {
    auto q = _inner->project(pt_world, is_print_detail);
    if (!q) {
        return q;
    }
    return normalizeProjectResult(_correction->toImage(q.value()));
}

StatusOr<Line3D> CorrectedCamera::lineOfSight(const Pt2D& pt_img_dist) const {
    return _inner->lineOfSight(_correction->toModel(pt_img_dist));
}

StatusOr<std::vector<ImageRay>> CorrectedCamera::projectAll(
    const Pt3D& pt_world, ProjectionCoverage* coverage) const {
    auto all = _inner->projectAll(pt_world, coverage);
    if (!all) {
        return all;
    }
    for (ImageRay& r : all.value()) {
        if (!r.is_ring) {
            r.pt_img_dist = _correction->toImage(r.pt_img_dist);
        }
    }
    return all;
}

StatusOr<Pt2D> CorrectedCamera::projectWithHint(const Pt3D& pt_world,
                                                ProjectionHint& hint) const {
    auto q = _inner->projectWithHint(pt_world, hint);
    if (!q) {
        return q;
    }
    return normalizeProjectResult(_correction->toImage(q.value()));
}

bool CorrectedCamera::lineImagesCurve() const {
    return _inner->multiValued() ? _inner->lineImagesCurve() : true;
}

ImageStatus CorrectedCamera::imageStatus(const Pt3D& pt_world, double margin_px,
                                         double merge_px, Pt2D* pt) const {
    if (!_inner->multiValued() || _inner->provablySingleImage(pt_world)) {
        // project() and the sensor, project() being this camera's
        return Camera::imageStatus(pt_world, margin_px, merge_px, pt);
    }
    auto all = _inner->projectAll(pt_world, nullptr);
    if (!all) {
        return ImageStatus::None;
    }
    std::vector<ImageRay> rays = all.value();
    // The primary is the inner model's choice, as project() returns it.
    bool ring = false;
    const ImageRay* primary = refraction_curved::primaryImage(
        rays, _inner->getNCol(), _inner->getNRow(), ring);
    const std::ptrdiff_t k = primary ? primary - rays.data() : -1;
    for (ImageRay& r : rays) {
        if (!r.is_ring) {
            r.pt_img_dist = _correction->toImage(r.pt_img_dist);
        }
    }
    return refraction_curved::statusOfImages(rays, k, getNCol(), getNRow(), margin_px,
                                             merge_px, pt);
}

Status CorrectedCamera::saveParameters(const std::string& file_name) const {
    return _inner->saveParameters(file_name);
}

std::shared_ptr<Camera> CorrectedCamera::clone() const {
    std::shared_ptr<CorrectedCamera> c(new CorrectedCamera(_inner->clone(), _correction));
    c->is_active = is_active;
    c->max_intensity = max_intensity;
    return c;
}

const Camera& baseCamera(const Camera& cam) {
    const Camera* c = &cam;
    while (const auto* w = dynamic_cast<const CorrectedCamera*>(c)) {
        c = w->inner().get();
    }
    return *c;
}
