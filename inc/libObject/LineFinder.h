#ifndef LINEFINDER_H
#define LINEFINDER_H

// The line detector for tracer images that the tube's astigmatism stretches
// into short horizontal or vertical lines (the cylinder STB plan, Phase 4, L5:
// cases/cyl_20261007/2026-10-07-plan.md). A C++ port of the Python prototype
// cases/cyl_20261007/lines.py (detect), step by step:
//
// One object is an axis-aligned box of lengths (Lx, Ly) blurred by a Gaussian
// (sx, sy) and integrated over each pixel,
//     m(x, y) = A * P(x - u; Lx, sx) * P(y - v; Ly, sy) + c,
// P(t; L, s) = the pixel integral of S(tau) = (1/L) int_{-L/2}^{L/2}
// exp(-(tau - q)^2 / (2 s^2)) dq (S(t; 0, s) = exp(-t^2 / (2 s^2))).
//   1. matched filters: every template (Lx, Ly) of the bank lengths_x x
//      lengths_y, minus the mean of a box around it, each response divided by
//      its own robust noise level over the frame;
//   2. candidates: 3 x 3 local maxima of the best normalised response >= k;
//   3. non-maximum suppression by boxes, by decreasing score;
//   4. Levenberg-Marquardt of m's 8 parameters on each object's window, the
//      other objects' models held fixed (deblending passes), on the
//      expectation of the clipped image when `clipped`;
//   5. a second pass on the residual image;
//   6. validation: an object stays only if its own score, on the image minus
//      every other object's model, still reaches k.
// The input is a preprocessed shadowgraph (particles bright on a background
// about 0, clipped to 8 bits); `noise` is its noise sigma before clipping
// (estimateNoiseClipped() when unknown).

#include <vector>

#include "Matrix.h"

struct LineFinderParam {
  std::vector<double> lengths_x = {0.0, 3.0, 5.0, 8.0, 12.0, 17.0, 24.0, 34.0};
  std::vector<double> lengths_y = {0.0, 3.0, 5.0, 8.0, 12.0, 17.0, 24.0};
  double sigma0 = 1.0;          // the templates' Gaussian blur [px]
  int bg_guard = 4;             // local-mean box = template support + this [px]
  int noise_sub = 4;            // every n-th row and column for the noise levels
  double k_detect = 6.0;        // threshold on the normalised score
  double core_px = 2.0;         // suppression box half-size beyond L/2 [px]
  double fit_margin_sigma = 3.0;// fit window: L/2 + this * sigma + 2 px
  int fit_max_half = 28;        // cap on a fit window's half-size [px]
  int fit_iters = 40;           // Levenberg-Marquardt iterations per pass
  int deblend_passes = 2;
  bool second_pass = true;
  bool validate = true;
  double L_min = 0.05, L_max = 60.0, sigma_min = 0.5, sigma_max = 4.0;
  double centre_box_frac = 0.25;// centre within +-(this * length + 2.5 px) of the candidate's pixel
  bool dynamic_bounds = true;   // that length: the object's current fitted one, the lengths only below L_max;
                                // else the template's, the lengths below length_growth * it + length_margin
  double length_growth = 1.5;
  double length_margin = 8.0;
  bool nms_symmetric = true;    // step 3 also drops a candidate whose box holds an accepted object's centre
  double c_max_sigma = 1.5;     // |c| <= this * noise
  bool clipped = true;          // the image is clipped at 0
};

struct LineObject {
  double u = 0, v = 0, Lx = 0, Ly = 0, sx = 0, sy = 0, A = 0, c = 0;
  double flux = 0, peak = 0, score = 0, chi2 = 0;
  int npx = 0;
  int pass = 0; // 0: the first pass, 1: found on the residual
};

namespace LineFinder {

// P(t; L, s) and, with deriv, dP/dt, dP/dL, dP/ds (lines.py: profile)
double profile(double t, double L, double s);
void profileDeriv(double t, double L, double s, double &P, double &dP_dt,
                  double &dP_dL, double &dP_ds);

// the detector (lines.py: detect)
std::vector<LineObject> detect(const Image &img, double noise,
                               const LineFinderParam &P);

// noise sigma of an image clipped at 0 (lines.py: estimate_noise_clipped)
double estimateNoiseClipped(const Image &img, double max_k = 1.0);

} // namespace LineFinder

#endif
