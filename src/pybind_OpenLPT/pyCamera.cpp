#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "Camera.h"
#include "STBCommons.h"
#include "py_camera_handle.h"

namespace py = pybind11;

namespace {

[[noreturn]] void throwLegacyFieldWrite(const char* field_name,
                                        const char* replacement) {
  throw std::runtime_error(std::string("Camera.") + field_name +
                           " is read-only in hard migration mode; use " +
                           replacement + " instead.");
}

} // namespace

void bind_Camera(py::module_& m) {
  py::class_<ProjectionHint>(m, "ProjectionHint",
                             "The ray of a point's last image (Camera.projectWithHint).")
      .def(py::init<>())
      .def_readwrite("lam", &ProjectionHint::lambda)
      .def_readwrite("c_z", &ProjectionHint::c_z)
      .def_readwrite("is_far", &ProjectionHint::is_far)
      .def_readwrite("valid", &ProjectionHint::valid)
      .def("__repr__", [](const ProjectionHint& h) {
        return "ProjectionHint(lam=" + std::to_string(h.lambda) +
               ", c_z=" + std::to_string(h.c_z) +
               ", is_far=" + (h.is_far ? std::string("True") : std::string("False")) +
               ", valid=" + (h.valid ? std::string("True") : std::string("False")) + ")";
      });

  py::class_<Camera, std::shared_ptr<Camera>>(m, "CameraModel")
      .def_property(
          "is_active", [](const Camera& c) { return c.is_active; },
          [](Camera& c, bool v) { c.is_active = v; })
      .def_property(
          "max_intensity", [](const Camera& c) { return c.max_intensity; },
          [](Camera& c, double v) { c.max_intensity = v; })
      .def("getNRow", &Camera::getNRow)
      .def("getNCol", &Camera::getNCol)
      // Status is not a registered pybind type, so raise instead of returning it.
      .def(
          "saveParameters",
          [](const Camera& c, const std::string& file_name) {
            auto st = c.saveParameters(file_name);
            if (!st) throw std::runtime_error(st.err.toString());
          },
          py::arg("file_name"))
      .def("type", &Camera::type)
      .def("clone", &Camera::clone, "A deep copy of the camera model.")
      .def(
          "project",
          [](const Camera& c, const Pt3D& pt_world, bool is_print_detail) {
            auto st = c.project(pt_world, is_print_detail);
            if (!st)
              throw std::runtime_error(st.status().err.toString());
            return st.value();
          },
          py::arg("pt_world"), py::arg("is_print_detail") = false)
      .def(
          "projectAll",
          [](const Camera& c, const Pt3D& pt_world) {
            // Curved ports are multi-valued: a point can have no image, one or
            // several. `coverage` distinguishes "no image" from "not searched".
            ProjectionCoverage cov;
            auto st = c.projectAll(pt_world, &cov);
            if (!st)
              throw std::runtime_error(st.status().err.toString());
            py::list rays;
            for (const ImageRay& r : st.value()) {
              py::dict d;
              d["pt"] = r.pt_img_dist;
              d["is_far"] = r.is_far;
              d["lambda"] = r.lambda;
              d["c_z"] = r.c_z;
              d["is_ring"] = r.is_ring;
              rays.append(d);
            }
            py::dict coverage;
            coverage["near_branch"] = cov.near_branch;
            coverage["far_branch"] = cov.far_branch;
            return py::make_tuple(rays, coverage);
          },
          py::arg("pt_world"),
          "Every image of a world point, as (rays, coverage).")
      .def(
          "lineOfSight",
          [](const Camera& c, const Pt2D& pt_img_dist) {
            auto st = c.lineOfSight(pt_img_dist);
            if (!st)
              throw std::runtime_error(st.status().err.toString());
            return st.value();
          },
          py::arg("pt_img_dist"))
      .def("usesProjectionHint", &Camera::usesProjectionHint)
      .def("lineImagesCurve", &Camera::lineImagesCurve)
      .def("multiValued", &Camera::multiValued)
      .def("provablySingleImage", &Camera::provablySingleImage, py::arg("pt_world"))
      .def(
          "imageStatus",
          [](const Camera& c, const Pt3D& pt_world, double margin_px, double merge_px) {
            Pt2D pt;
            const ImageStatus st = c.imageStatus(pt_world, margin_px, merge_px, &pt);
            const char* name = st == ImageStatus::None
                                   ? "none"
                                   : (st == ImageStatus::One ? "one" : "ambiguous");
            return py::make_tuple(std::string(name), pt);
          },
          py::arg("pt_world"), py::arg("margin_px") = 0.0, py::arg("merge_px") = 0.0,
          "Phase 2's status: ('none' | 'one' | 'ambiguous', the one image).")
      .def(
          "projectWithHint",
          [](const Camera& c, const Pt3D& pt_world, ProjectionHint hint) -> py::tuple {
            auto st = c.projectWithHint(pt_world, hint);
            if (!st)
              return py::make_tuple(false, Pt2D(), hint, st.status().err.toString());
            return py::make_tuple(true, st.value(), hint, std::string());
          },
          py::arg("pt_world"), py::arg("hint"),
          "project() started from a hint: (ok, pt, updated hint, error).")
      .def(
          "hintFromLineOfSight",
          [](const Camera& c, const Line3D& los, const Pt3D& pt_world) {
            ProjectionHint h;
            const bool ok = c.hintFromLineOfSight(los, pt_world, h);
            return py::make_tuple(ok, h);
          },
          py::arg("los"), py::arg("pt_world"),
          "A hint from a line of sight passing near the point: (ok, hint).")
      .def(
          "projectBatchStatus",
          [](const Camera& c, const std::vector<Pt3D>& pts_world,
             bool is_print_detail) {
            std::vector<std::tuple<bool, Pt2D, std::string>> out;
            out.reserve(pts_world.size());
            for (const auto& pt_world : pts_world) {
              auto st = c.project(pt_world, is_print_detail);
              if (st) {
                out.emplace_back(true, st.value(), std::string(""));
              } else {
                out.emplace_back(false, Pt2D(), st.status().err.toString());
              }
            }
            return out;
          },
          py::arg("pts_world"), py::arg("is_print_detail") = false)
      .def(
          "lineOfSightBatchStatus",
          [](const Camera& c, const std::vector<Pt2D>& pts_img_dist) {
            std::vector<std::tuple<bool, Line3D, std::string>> out;
            out.reserve(pts_img_dist.size());
            for (const auto& pt_img_dist : pts_img_dist) {
              auto st = c.lineOfSight(pt_img_dist);
              if (st) {
                out.emplace_back(true, st.value(), std::string(""));
              } else {
                out.emplace_back(false, Line3D(), st.status().err.toString());
              }
            }
            return out;
          },
          py::arg("pts_img_dist"));

  py::class_<PinholeCamera, Camera, std::shared_ptr<PinholeCamera>>(
      m, "PinholeCameraModel")
      .def(py::init<>());
  py::class_<PolynomialCamera, Camera,
             std::shared_ptr<PolynomialCamera>>(m, "PolynomialCameraModel")
      .def(py::init<>());
  py::class_<RefractionPinholeCamera, Camera,
             std::shared_ptr<RefractionPinholeCamera>>(
      m, "RefractionPinholeCameraModel")
      .def(py::init<>());

  // Curved refractive ports. The setters are shared, so they are registered on
  // the common base and inherited by both concrete models.
  py::class_<RefractionCurvedCamera, Camera,
             std::shared_ptr<RefractionCurvedCamera>>(
      m, "RefractionCurvedCameraModel")
      .def(
          "setImageSize",
          [](RefractionCurvedCamera& c, int n_row, int n_col) {
            auto st = c.setImageSize(n_row, n_col);
            if (!st) throw std::runtime_error(st.err.toString());
          },
          py::arg("n_row"), py::arg("n_col"))
      .def(
          "setIntrinsics",
          [](RefractionCurvedCamera& c, double fx, double fy, double cx, double cy,
             const std::vector<double>& dist_coeff) {
            auto st = c.setIntrinsics(fx, fy, cx, cy, dist_coeff);
            if (!st) throw std::runtime_error(st.err.toString());
          },
          py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
          py::arg("dist_coeff"))
      .def(
          "setExtrinsics",
          [](RefractionCurvedCamera& c, const Pt3D& rvec, const Pt3D& tvec) {
            auto st = c.setExtrinsics(rvec, tvec);
            if (!st) throw std::runtime_error(st.err.toString());
          },
          py::arg("rvec"), py::arg("tvec"))
      .def(
          "setRefraction",
          [](RefractionCurvedCamera& c, const Pt3D& center, const Pt3D& axis,
             const std::vector<double>& refract_array,
             const std::vector<double>& radius_array) {
            auto st = c.setRefraction(center, axis, refract_array, radius_array);
            if (!st) throw std::runtime_error(st.err.toString());
          },
          py::arg("center"), py::arg("axis"), py::arg("refract_array"),
          py::arg("radius_array"),
          "refract_array is n_1..n_N with the pinhole's medium first; "
          "radius_array is R_1..R_{N-1}, strictly decreasing.")
      .def(
          "setSolverOptions",
          [](RefractionCurvedCamera& c, double proj_tol, int proj_nmax) {
            auto st = c.setSolverOptions(proj_tol, proj_nmax);
            if (!st) throw std::runtime_error(st.err.toString());
          },
          py::arg("proj_tol"), py::arg("proj_nmax"))
      .def("commitUpdate", [](RefractionCurvedCamera& c) {
        auto st = c.commitUpdate();
        if (!st) throw std::runtime_error(st.err.toString());
      });

  py::class_<RefractionCylinderCamera, RefractionCurvedCamera,
             std::shared_ptr<RefractionCylinderCamera>>(
      m, "RefractionCylinderCameraModel")
      .def(py::init<>());
  py::class_<RefractionSphereCamera, RefractionCurvedCamera,
             std::shared_ptr<RefractionSphereCamera>>(
      m, "RefractionSphereCameraModel")
      .def(py::init<>());

  // A camera's image correction (VSC's field) and the camera it corrects. The
  // correction is read-only in C++; pybind holds it without const.
  py::class_<ImageCorrection, std::shared_ptr<ImageCorrection>>(
      m, "ImageCorrection",
      "A smooth correction of a camera's image positions: a detection at pixel d has "
      "the model's image point d - c(d). Built by load() or fromGrid().")
      .def_static(
          "load",
          [](const std::string& file_name) {
            auto st = ImageCorrection::loadFromFile(file_name);
            if (!st) throw std::runtime_error(st.status().err.toString());
            return std::const_pointer_cast<ImageCorrection>(st.value());
          },
          py::arg("file_name"))
      .def_static(
          "fromGrid",
          [](int n_col, int n_row, double u0, double v0, double step, int nu, int nv,
             const std::vector<double>& du, const std::vector<double>& dv,
             const std::string& source) {
            auto st = ImageCorrection::fromGrid(n_col, n_row, u0, v0, step, nu, nv, du,
                                                dv, source);
            if (!st) throw std::runtime_error(st.status().err.toString());
            return std::const_pointer_cast<ImageCorrection>(st.value());
          },
          py::arg("n_col"), py::arg("n_row"), py::arg("u0"), py::arg("v0"),
          py::arg("step"), py::arg("nu"), py::arg("nv"), py::arg("du"), py::arg("dv"),
          py::arg("source") = "",
          "du and dv row by row (v outer), nu * nv values each.")
      .def("at", &ImageCorrection::at, py::arg("pt"), "c at a pixel.")
      .def("toModel", &ImageCorrection::toModel, py::arg("pt_img"),
           "The model's image point of a detection: pt_img - c(pt_img).")
      .def("toImage", &ImageCorrection::toImage, py::arg("pt_model"),
           "The pixel d with d - c(d) = pt_model.")
      .def_property_readonly("n_col", &ImageCorrection::nCol)
      .def_property_readonly("n_row", &ImageCorrection::nRow)
      .def_property_readonly("u0", &ImageCorrection::u0)
      .def_property_readonly("v0", &ImageCorrection::v0)
      .def_property_readonly("step", &ImageCorrection::step)
      .def_property_readonly("nu", &ImageCorrection::nu)
      .def_property_readonly("nv", &ImageCorrection::nv)
      .def_property_readonly("du", &ImageCorrection::du)
      .def_property_readonly("dv", &ImageCorrection::dv)
      .def_property_readonly("slope_bound", &ImageCorrection::slopeBound)
      .def_property_readonly("source", &ImageCorrection::source);

  py::class_<CorrectedCamera, Camera, std::shared_ptr<CorrectedCamera>>(
      m, "CorrectedCameraModel",
      "A camera model with its image positions corrected by an ImageCorrection.")
      .def_static(
          "create",
          [](std::shared_ptr<Camera> inner, std::shared_ptr<ImageCorrection> correction) {
            auto st = CorrectedCamera::create(std::move(inner), std::move(correction));
            if (!st) throw std::runtime_error(st.status().err.toString());
            return st.value();
          },
          py::arg("inner"), py::arg("correction"))
      .def("inner", &CorrectedCamera::inner, "The camera model it corrects.")
      .def(
          "correction",
          [](const CorrectedCamera& c) {
            return std::const_pointer_cast<ImageCorrection>(c.correction());
          },
          "Its ImageCorrection.");

  m.def(
      "loadCameraModel",
      [](const std::string& file_name, const std::string& correction) {
        auto st = CameraFactory::loadFromFile(file_name);
        if (!st) {
          throw std::runtime_error(st.status().err.toString());
        }
        if (correction.empty()) {
          return st.value();
        }
        // As config.json's "correction": the model wrapped in its image correction.
        auto corr = ImageCorrection::loadFromFile(correction);
        if (!corr) {
          throw std::runtime_error(corr.status().err.toString());
        }
        auto wrapped = CorrectedCamera::create(st.value(), corr.value());
        if (!wrapped) {
          throw std::runtime_error(wrapped.status().err.toString());
        }
        return std::static_pointer_cast<Camera>(wrapped.value());
      },
      py::arg("file_name"), py::arg("correction") = "",
      "A camera model from its file; with `correction`, wrapped in that image correction.");

  py::enum_<CameraType>(m, "CameraType")
      .value("PINHOLE", CameraType::Pinhole)
      .value("POLYNOMIAL", CameraType::Polynomial)
      .value("PINPLATE", CameraType::RefractionPinhole)
      .value("PINCYL", CameraType::RefractionCylinder)
      .value("PINSPH", CameraType::RefractionSphere)
      .export_values();

  py::enum_<RefPlane>(m, "RefPlane")
      .value("REF_X", RefPlane::REF_X)
      .value("REF_Y", RefPlane::REF_Y)
      .value("REF_Z", RefPlane::REF_Z)
      .export_values();

  py::class_<PinholeParam>(m, "PinholeParam")
      .def(py::init<>())
      .def_readwrite("n_row", &PinholeParam::n_row)
      .def_readwrite("n_col", &PinholeParam::n_col)
      .def_readwrite("cam_mtx", &PinholeParam::cam_mtx)
      .def_readwrite("is_distorted", &PinholeParam::is_distorted)
      .def_readwrite("n_dist_coeff", &PinholeParam::n_dist_coeff)
      .def_readwrite("dist_coeff", &PinholeParam::dist_coeff)
      .def_readwrite("r_mtx", &PinholeParam::r_mtx)
      .def_readwrite("t_vec", &PinholeParam::t_vec)
      .def_readwrite("r_mtx_inv", &PinholeParam::r_mtx_inv)
      .def_readwrite("t_vec_inv", &PinholeParam::t_vec_inv);

  py::class_<PolyParam>(m, "PolyParam")
      .def(py::init<>())
      .def_readwrite("n_row", &PolyParam::n_row)
      .def_readwrite("n_col", &PolyParam::n_col)
      .def_readwrite("ref_plane", &PolyParam::ref_plane)
      .def_readwrite("plane", &PolyParam::plane)
      .def_readwrite("n_coeff", &PolyParam::n_coeff)
      .def_readwrite("u_coeffs", &PolyParam::u_coeffs)
      .def_readwrite("du_coeffs", &PolyParam::du_coeffs)
      .def_readwrite("v_coeffs", &PolyParam::v_coeffs)
      .def_readwrite("dv_coeffs", &PolyParam::dv_coeffs);

  py::class_<PinPlateParam, PinholeParam>(m, "PinPlateParam")
      .def(py::init<>())
      .def_readwrite("plane", &PinPlateParam::plane)
      .def_readwrite("plane_array", &PinPlateParam::plane_array)
      .def_readwrite("u_axis", &PinPlateParam::u_axis)
      .def_readwrite("v_axis", &PinPlateParam::v_axis)
      .def_readwrite("refract_array", &PinPlateParam::refract_array)
      .def_readwrite("w_array", &PinPlateParam::w_array)
      .def_readwrite("n_plate", &PinPlateParam::n_plate)
      .def_readwrite("proj_tol", &PinPlateParam::proj_tol)
      .def_readwrite("proj_nmax", &PinPlateParam::proj_nmax)
      .def_readwrite("lr", &PinPlateParam::lr)
      .def_readwrite("refract_ratio_max", &PinPlateParam::refract_ratio_max);

  py::class_<PyCameraHandle>(m, "Camera")
      .def(py::init<>())
      .def(py::init<const std::string&>(), py::arg("file_name"))
      .def(py::init<const PyCameraHandle&>(), py::arg("other"))
      .def("loadParameters", &PyCameraHandle::loadParameters,
           py::arg("file_name"))
      .def("setCorrection", &PyCameraHandle::setCorrection, py::arg("file_name"),
           "Correct the image positions by an image correction file (config.json's "
           "\"correction\"); project() and lineOfSight() then go through it.")
      .def(
          "setCorrectionObject",
          [](PyCameraHandle& c, std::shared_ptr<ImageCorrection> correction) {
            c.setCorrectionObject(std::move(correction));
          },
          py::arg("correction"))
      .def("clearCorrection", &PyCameraHandle::clearCorrection)
      .def("hasCorrection", &PyCameraHandle::hasCorrection)
      .def("correctionFile", &PyCameraHandle::correctionFile,
           "The correction's file, or '' without one.")
      .def(
          "cameraModel", [](const PyCameraHandle& c) { return c.model(); },
          "The camera as tracking uses it (a CameraModel; corrected when a correction is set).")
      .def("saveParameters", &PyCameraHandle::saveParameters,
           py::arg("file_name"))
      .def_property("_type", &PyCameraHandle::legacyType,
                    &PyCameraHandle::setLegacyType)
      .def_property("_pinhole_param", &PyCameraHandle::pinholeParam,
                    [](PyCameraHandle&, const PinholeParam&) {
                      throwLegacyFieldWrite(
                          "_pinhole_param",
                          "setPinplateIntrinsics/setPinplateExtrinsics");
                    })
      .def_property("_poly_param", &PyCameraHandle::polyParam,
                    [](PyCameraHandle&, const PolyParam&) {
                      throwLegacyFieldWrite("_poly_param",
                                            "PolynomialCameraModel setters");
                    })
      .def_property("_pinplate_param", &PyCameraHandle::pinplateParam,
                    [](PyCameraHandle&, const PinPlateParam&) {
                      throwLegacyFieldWrite(
                          "_pinplate_param",
                          "setPinplate* methods + commitPinplateUpdate");
                    })
      .def_property("_max_intensity", &PyCameraHandle::maxIntensity,
                    &PyCameraHandle::setMaxIntensity)
      .def_property("_is_active", &PyCameraHandle::isActive,
                    &PyCameraHandle::setIsActive)
      .def("rmtxTorvec", &PyCameraHandle::rmtxTorvec, py::arg("r_mtx"))
      .def("setPinplateImageSize", &PyCameraHandle::setPinplateImageSize,
           py::arg("n_row"), py::arg("n_col"))
      .def("setPinplateIntrinsics", &PyCameraHandle::setPinplateIntrinsics,
           py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
           py::arg("dist_coeff"))
      .def("setPinplateExtrinsics", &PyCameraHandle::setPinplateExtrinsics,
           py::arg("rvec"), py::arg("tvec"))
      .def("setPinplateRefraction", &PyCameraHandle::setPinplateRefraction,
           py::arg("plane_pt"), py::arg("plane_n"), py::arg("refract_array"),
           py::arg("w_array"), py::arg("proj_tol"), py::arg("proj_nmax"),
           py::arg("lr"))
      .def("commitPinplateUpdate", &PyCameraHandle::commitPinplateUpdate,
           py::arg("is_active"), py::arg("max_intensity"))
      .def("getNRow", &PyCameraHandle::getNRow)
      .def("getNCol", &PyCameraHandle::getNCol)
      .def("project", &PyCameraHandle::project, py::arg("pt_world"),
           py::arg("is_print_detail") = false)
      .def(
          "projectStatus",
          [](const PyCameraHandle& c, const Pt3D& pt_world,
             bool is_print_detail) {
            try {
              return py::make_tuple(true, c.project(pt_world, is_print_detail),
                                    std::string(""));
            } catch (const std::exception& e) {
              return py::make_tuple(false, Pt2D(), std::string(e.what()));
            }
          },
          py::arg("pt_world"), py::arg("is_print_detail") = false)
      .def("projectBatchStatus", &PyCameraHandle::projectBatchStatus,
           py::arg("pts_world"), py::arg("is_print_detail") = false)
      .def("lineOfSight", &PyCameraHandle::lineOfSight,
           py::arg("pt_img_dist"))
      .def(
          "lineOfSightStatus",
          [](const PyCameraHandle& c, const Pt2D& pt_img_dist) {
            try {
              return py::make_tuple(true, c.lineOfSight(pt_img_dist),
                                    std::string(""));
            } catch (const std::exception& e) {
              return py::make_tuple(false, Line3D(), std::string(e.what()));
            }
          },
          py::arg("pt_img_dist"))
      .def("lineOfSightBatchStatus", &PyCameraHandle::lineOfSightBatchStatus,
           py::arg("pts_img_dist"));
}
