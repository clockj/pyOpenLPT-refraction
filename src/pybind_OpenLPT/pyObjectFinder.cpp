// pyObjectFinder.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ObjectFinder.h"
#include "LineFinder.h"
#include "ObjectInfo.h"
#include "Config.h"
#include "Matrix.h"

namespace py = pybind11;

void bind_ObjectFinder(py::module_& m) {
    py::class_<ObjectFinder2D>(m, "ObjectFinder2D")
        .def(py::init<>())

        // 直接返回 std::vector<std::unique_ptr<Object2D>>
        .def("findObject2D",
             [](ObjectFinder2D& self, const Image& image, const ObjectConfig& obj_cfg) {
                 return self.findObject2D(image, obj_cfg);
             },
             py::arg("image"), py::arg("obj_cfg"));

    // the line detector (LineFinder.h), for tests against its Python prototype
    py::class_<LineFinderParam>(m, "LineFinderParam")
        .def(py::init<>())
        .def_readwrite("lengths_x", &LineFinderParam::lengths_x)
        .def_readwrite("lengths_y", &LineFinderParam::lengths_y)
        .def_readwrite("sigma0", &LineFinderParam::sigma0)
        .def_readwrite("bg_guard", &LineFinderParam::bg_guard)
        .def_readwrite("noise_sub", &LineFinderParam::noise_sub)
        .def_readwrite("k_detect", &LineFinderParam::k_detect)
        .def_readwrite("core_px", &LineFinderParam::core_px)
        .def_readwrite("fit_margin_sigma", &LineFinderParam::fit_margin_sigma)
        .def_readwrite("fit_max_half", &LineFinderParam::fit_max_half)
        .def_readwrite("fit_iters", &LineFinderParam::fit_iters)
        .def_readwrite("deblend_passes", &LineFinderParam::deblend_passes)
        .def_readwrite("second_pass", &LineFinderParam::second_pass)
        .def_readwrite("validate", &LineFinderParam::validate)
        .def_readwrite("L_min", &LineFinderParam::L_min)
        .def_readwrite("L_max", &LineFinderParam::L_max)
        .def_readwrite("sigma_min", &LineFinderParam::sigma_min)
        .def_readwrite("sigma_max", &LineFinderParam::sigma_max)
        .def_readwrite("centre_box_frac", &LineFinderParam::centre_box_frac)
        .def_readwrite("length_growth", &LineFinderParam::length_growth)
        .def_readwrite("dynamic_bounds", &LineFinderParam::dynamic_bounds)
        .def_readwrite("length_margin", &LineFinderParam::length_margin)
        .def_readwrite("nms_symmetric", &LineFinderParam::nms_symmetric)
        .def_readwrite("c_max_sigma", &LineFinderParam::c_max_sigma)
        .def_readwrite("clipped", &LineFinderParam::clipped);
    m.def("detectLines",
          [](const Image& img, double noise, const LineFinderParam& p) {
              std::vector<std::vector<double>> out;
              for (const LineObject& o : LineFinder::detect(img, noise, p))
                  out.push_back({o.u, o.v, o.Lx, o.Ly, o.sx, o.sy, o.A, o.c, o.flux, o.peak,
                                 o.score, o.chi2, static_cast<double>(o.npx),
                                 static_cast<double>(o.pass)});
              return out;
          },
          py::arg("image"), py::arg("noise"), py::arg("param"),
          "LineFinder::detect: rows (u, v, Lx, Ly, sx, sy, A, c, flux, peak, score, chi2, npx, pass).");
    m.def("estimateNoiseClipped", &LineFinder::estimateNoiseClipped, py::arg("image"),
          py::arg("max_k") = 1.0);
    m.def("lineProfile", [](double t, double L, double s) { return LineFinder::profile(t, L, s); },
          py::arg("t"), py::arg("L"), py::arg("s"));
}
