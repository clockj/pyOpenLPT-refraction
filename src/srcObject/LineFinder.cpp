// The line detector: a C++ port of cases/cyl_20261007/lines.py (detect); see
// LineFinder.h. Each function names the Python function it follows.

#include "LineFinder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include <omp.h>

namespace {

const double kSq2Pi = std::sqrt(2.0 * 3.141592653589793);
constexpr int kNPar = 8; // A, u, v, Lx, Ly, sx, sy, c

// scipy.special.ndtr (Cephes): the standard normal CDF
inline double ndtr(double a) {
  const double x = a * 0.70710678118654752440;
  const double z = std::fabs(x);
  if (z < 0.70710678118654752440)
    return 0.5 + 0.5 * std::erf(x);
  const double y = 0.5 * std::erfc(z);
  return x > 0.0 ? 1.0 - y : y;
}

inline double psi(double z) { return z * ndtr(z) + std::exp(-0.5 * z * z) / kSq2Pi; }

// numpy.round / Python round: halves to even
inline int roundHalfEven(double x) { return static_cast<int>(std::nearbyint(x)); }

// Python's float floor division
inline int floorDiv(double a, double b) { return static_cast<int>(std::floor(a / b)); }

// the image as rows of floats (the prototype works on float32 copies)
struct FImg {
  int H = 0, W = 0;
  std::vector<float> d;
  FImg() = default;
  FImg(int h, int w, float v = 0.0f) : H(h), W(w), d(static_cast<size_t>(h) * w, v) {}
  float &at(int r, int c) { return d[static_cast<size_t>(r) * W + c]; }
  float at(int r, int c) const { return d[static_cast<size_t>(r) * W + c]; }
};

struct DImg {
  int H = 0, W = 0;
  std::vector<double> d;
  DImg() = default;
  DImg(int h, int w, double v = 0.0) : H(h), W(w), d(static_cast<size_t>(h) * w, v) {}
  double &at(int r, int c) { return d[static_cast<size_t>(r) * W + c]; }
  double at(int r, int c) const { return d[static_cast<size_t>(r) * W + c]; }
};

// scipy.ndimage's "reflect" (= OpenCV BORDER_REFLECT): d c b a | a b c d | d c b a
inline int reflect(int i, int n) {
  if (n == 1)
    return 0;
  const int period = 2 * n;
  i %= period;
  if (i < 0)
    i += period;
  return i < n ? i : period - 1 - i;
}

// kernel1d(L, s): P(n; L, s) on n = -half..half, half = ceil(L/2 + 3 s)
std::vector<double> kernel1d(double L, double s) {
  const int half = static_cast<int>(std::ceil(L / 2 + 3 * s));
  std::vector<double> k(2 * half + 1);
  for (int n = -half; n <= half; ++n)
    k[n + half] = LineFinder::profile(static_cast<double>(n), L, s);
  return k;
}

// scipy.ndimage.correlate1d of a float32 image with a symmetric kernel (the
// kernel rounded to float32, as the prototype passes it), mode reflect; the
// sums in double, the result rounded to float32 (scipy's symmetric branch:
// w0 x0 + sum_k w_k (x_-k + x_k)).
void correlate1d(const FImg &in, const std::vector<double> &k64, int axis, FImg &out) {
  const int half = static_cast<int>(k64.size() / 2);
  std::vector<double> w(k64.size());
  for (size_t i = 0; i < k64.size(); ++i)
    w[i] = static_cast<double>(static_cast<float>(k64[i]));
  out = FImg(in.H, in.W);
  const int H = in.H, W = in.W;
  if (axis == 1) {
#pragma omp parallel for if (!omp_in_parallel())
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        double t = static_cast<double>(in.at(r, c)) * w[half];
        for (int j = -half; j < 0; ++j)
          t += (static_cast<double>(in.at(r, reflect(c + j, W))) +
                static_cast<double>(in.at(r, reflect(c - j, W)))) * w[j + half];
        out.at(r, c) = static_cast<float>(t);
      }
    }
  } else {
#pragma omp parallel for if (!omp_in_parallel())
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        double t = static_cast<double>(in.at(r, c)) * w[half];
        for (int j = -half; j < 0; ++j)
          t += (static_cast<double>(in.at(reflect(r + j, H), c)) +
                static_cast<double>(in.at(reflect(r - j, H), c))) * w[j + half];
        out.at(r, c) = static_cast<float>(t);
      }
    }
  }
}

// cv2.blur(img, (2 bx + 1, 2 by + 1), BORDER_REFLECT): the box mean, by an
// integral image of the reflected image (double sums, float result)
struct BoxMean {
  int H = 0, W = 0, pad = 0;
  std::vector<double> S; // (H + 2 pad + 1) x (W + 2 pad + 1)
  void build(const FImg &img, int max_half) {
    H = img.H;
    W = img.W;
    pad = max_half + 1;
    const int HH = H + 2 * pad, WW = W + 2 * pad;
    S.assign(static_cast<size_t>(HH + 1) * (WW + 1), 0.0);
    for (int r = 0; r < HH; ++r) {
      double row = 0.0;
      const int sr = reflect(r - pad, H);
      for (int c = 0; c < WW; ++c) {
        row += img.at(sr, reflect(c - pad, W));
        S[static_cast<size_t>(r + 1) * (WW + 1) + (c + 1)] =
            S[static_cast<size_t>(r) * (WW + 1) + (c + 1)] + row;
      }
    }
  }
  float mean(int r, int c, int bx, int by) const {
    const int WW = W + 2 * pad;
    const int r0 = r - by + pad, r1 = r + by + pad + 1, c0 = c - bx + pad, c1 = c + bx + pad + 1;
    auto at = [&](int rr, int cc) { return S[static_cast<size_t>(rr) * (WW + 1) + cc]; };
    const double s = at(r1, c1) - at(r0, c1) - at(r1, c0) + at(r0, c0);
    return static_cast<float>(s / ((2.0 * bx + 1) * (2.0 * by + 1)));
  }
};

double median(std::vector<double> v) {
  const size_t n = v.size();
  if (n == 0)
    return 0.0;
  const size_t mid = n / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double m = v[mid];
  if ((n & 1) == 0) {
    const double lo = *std::max_element(v.begin(), v.begin() + mid);
    m = 0.5 * (m + lo);
  }
  return m;
}

double robustSigma(const std::vector<double> &x) {
  const double med = median(x);
  std::vector<double> dev(x.size());
  for (size_t i = 0; i < x.size(); ++i)
    dev[i] = std::fabs(x[i] - med);
  return 1.4826 * median(dev);
}

struct Template {
  double Lx, Ly;
  std::vector<double> kx, ky;
  double ssum;
  double sig;
};

// matched_filter(img, P, noise_ref): the best normalised response, its
// template index, and the templates (with their noise levels)
void matchedFilter(const FImg &img, const FImg *ref, const LineFinderParam &P, FImg &best,
                   std::vector<int16_t> &idx, std::vector<Template> &templates) {
  const int H = img.H, W = img.W;
  const double s0 = P.sigma0;
  const int g = P.bg_guard, sub = P.noise_sub;
  best = FImg(H, W, -std::numeric_limits<float>::infinity());
  idx.assign(static_cast<size_t>(H) * W, 0);
  templates.clear();
  int max_half = 0;
  for (double L : P.lengths_x)
    max_half = std::max(max_half, static_cast<int>(std::ceil(L / 2 + 3 * s0)) + g);
  for (double L : P.lengths_y)
    max_half = std::max(max_half, static_cast<int>(std::ceil(L / 2 + 3 * s0)) + g);
  BoxMean box_img, box_ref;
  box_img.build(img, max_half);
  if (ref)
    box_ref.build(*ref, max_half);
  FImg cx, cxr, resp, rr;
  for (double Lx : P.lengths_x) {
    const std::vector<double> kx = kernel1d(Lx, s0);
    correlate1d(img, kx, 1, cx);
    if (ref)
      correlate1d(*ref, kx, 1, cxr);
    const int hx = static_cast<int>(kx.size() / 2);
    for (double Ly : P.lengths_y) {
      const std::vector<double> ky = kernel1d(Ly, s0);
      const int hy = static_cast<int>(ky.size() / 2);
      correlate1d(cx, ky, 0, resp);
      const double ssum = std::accumulate(kx.begin(), kx.end(), 0.0) *
                          std::accumulate(ky.begin(), ky.end(), 0.0);
      const float ssum_f = static_cast<float>(ssum);
      const int bx = hx + g, by = hy + g;
#pragma omp parallel for if (!omp_in_parallel())
      for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
          resp.at(r, c) = resp.at(r, c) - ssum_f * box_img.mean(r, c, bx, by);
      std::vector<double> rs;
      rs.reserve(static_cast<size_t>((H + sub - 1) / sub) * ((W + sub - 1) / sub));
      if (!ref) {
        for (int r = 0; r < H; r += sub)
          for (int c = 0; c < W; c += sub)
            rs.push_back(resp.at(r, c));
      } else {
        correlate1d(cxr, ky, 0, rr);
        for (int r = 0; r < H; r += sub)
          for (int c = 0; c < W; c += sub)
            rs.push_back(static_cast<float>(rr.at(r, c) - ssum_f * box_ref.mean(r, c, bx, by)));
      }
      const double sig = std::max(robustSigma(rs), 1e-6);
      const float sig_f = static_cast<float>(sig);
      const int16_t tid = static_cast<int16_t>(templates.size());
#pragma omp parallel for if (!omp_in_parallel())
      for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
          const float nr = resp.at(r, c) / sig_f;
          if (nr > best.at(r, c)) {
            best.at(r, c) = nr;
            idx[static_cast<size_t>(r) * W + c] = tid;
          }
        }
      templates.push_back(Template{Lx, Ly, kx, ky, ssum, sig});
    }
  }
}

// local_maxima(score, k): 3 x 3 maxima >= k (mode nearest), plateaus deduplicated
void localMaxima(const FImg &score, double k, std::vector<int> &rows, std::vector<int> &cols) {
  const int H = score.H, W = score.W;
  rows.clear();
  cols.clear();
  std::vector<char> taken(static_cast<size_t>(H) * W, 0);
  const float kf = static_cast<float>(k);
  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      const float s = score.at(r, c);
      if (!(s >= kf))
        continue;
      float mx = s;
      for (int dr = -1; dr <= 1; ++dr)
        for (int dc = -1; dc <= 1; ++dc) {
          const int rr = std::clamp(r + dr, 0, H - 1), cc = std::clamp(c + dc, 0, W - 1);
          mx = std::max(mx, score.at(rr, cc));
        }
      if (!(s >= mx))
        continue;
      bool dup = false;
      const int nb[4][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};
      for (const auto &d : nb) {
        const int rr = r + d[0], cc = c + d[1];
        if (rr < 0 || cc < 0 || cc >= W)
          continue;
        if (taken[static_cast<size_t>(rr) * W + cc] && score.at(rr, cc) == s) {
          dup = true;
          break;
        }
      }
      if (dup)
        continue;
      taken[static_cast<size_t>(r) * W + c] = 1;
      rows.push_back(r);
      cols.push_back(c);
    }
  }
}

struct Accepted {
  double u, v, Lx, Ly;
};

// suppress(rows, cols, scores, Ls, P, existing, symmetric, core): the indices kept
std::vector<int> suppress(const std::vector<int> &rows, const std::vector<int> &cols,
                          const std::vector<float> &scores,
                          const std::vector<std::pair<double, double>> &Ls,
                          const std::vector<Accepted> &existing, bool symmetric, double core) {
  const double cell = 16.0;
  std::map<std::pair<int, int>, std::vector<int>> boxes, centres;
  std::vector<Accepted> acc;
  auto add = [&](double u, double v, double Lx, double Ly) {
    const int k = static_cast<int>(acc.size());
    acc.push_back(Accepted{u, v, Lx, Ly});
    const double rx = Lx / 2 + core, ry = Ly / 2 + core;
    for (int gx = floorDiv(u - rx, cell); gx <= floorDiv(u + rx, cell); ++gx)
      for (int gy = floorDiv(v - ry, cell); gy <= floorDiv(v + ry, cell); ++gy)
        boxes[{gx, gy}].push_back(k);
    centres[{floorDiv(u, cell), floorDiv(v, cell)}].push_back(k);
  };
  for (const Accepted &a : existing)
    add(a.u, a.v, a.Lx, a.Ly);
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return -scores[a] < -scores[b]; });
  std::vector<int> kept;
  for (int i : order) {
    const double u = cols[i], v = rows[i];
    const double Lx = Ls[i].first, Ly = Ls[i].second;
    bool hit = false;
    auto itb = boxes.find({floorDiv(u, cell), floorDiv(v, cell)});
    if (itb != boxes.end()) {
      for (int k : itb->second) {
        const Accepted &a = acc[k];
        if (std::fabs(u - a.u) <= a.Lx / 2 + core && std::fabs(v - a.v) <= a.Ly / 2 + core) {
          hit = true;
          break;
        }
      }
    }
    if (!hit && symmetric) {
      const double rx = Lx / 2 + core, ry = Ly / 2 + core;
      for (int gx = floorDiv(u - rx, cell); gx <= floorDiv(u + rx, cell) && !hit; ++gx)
        for (int gy = floorDiv(v - ry, cell); gy <= floorDiv(v + ry, cell) && !hit; ++gy) {
          auto itc = centres.find({gx, gy});
          if (itc == centres.end())
            continue;
          for (int k : itc->second) {
            if (std::fabs(u - acc[k].u) <= rx && std::fabs(v - acc[k].v) <= ry) {
              hit = true;
              break;
            }
          }
        }
    }
    if (hit)
      continue;
    kept.push_back(i);
    add(u, v, Lx, Ly);
  }
  return kept;
}

using Par = std::array<double, kNPar>;
struct Bound {
  double u_lo, u_hi, v_lo, v_hi, Lx_hi, Ly_hi;
};

// _initial: the amplitude by least squares on the template, and the bounds
void initial(const DImg &img, const std::vector<int> &rows, const std::vector<int> &cols,
             const std::vector<int> &tids, const std::vector<Template> &templates,
             const LineFinderParam &P, std::vector<Par> &p, std::vector<Bound> &b) {
  const int H = img.H, W = img.W;
  const double fr = P.centre_box_frac, gr = P.length_growth;
  p.resize(rows.size());
  b.resize(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    const int r = rows[i], c = cols[i];
    const Template &t = templates[tids[i]];
    const int hx = static_cast<int>(t.kx.size() / 2), hy = static_cast<int>(t.ky.size() / 2);
    const int r0 = std::max(r - hy, 0), r1 = std::min(r + hy, H - 1);
    const int c0 = std::max(c - hx, 0), c1 = std::min(c + hx, W - 1);
    double tp = 0.0, tt = 0.0;
    for (int rr = r0; rr <= r1; ++rr)
      for (int cc = c0; cc <= c1; ++cc) {
        const double T = t.ky[rr - r + hy] * t.kx[cc - c + hx];
        tp += T * img.at(rr, cc);
        tt += T * T;
      }
    const double A = tp / std::max(tt, 1e-12);
    p[i] = Par{std::max(A, 1e-3), static_cast<double>(c), static_cast<double>(r),
               std::max(t.Lx, 0.5), std::max(t.Ly, 0.5), 1.0, 1.0, 0.0};
    const double bx = fr * t.Lx + 2.5, by = fr * t.Ly + 2.5;
    b[i] = Bound{c - bx, c + bx, r - by, r + by, gr * t.Lx + P.length_margin,
                 gr * t.Ly + P.length_margin};
  }
}

void windowOf(const Par &p, const LineFinderParam &P, int &hx, int &hy) {
  const double m = P.fit_margin_sigma;
  const int cap = P.fit_max_half;
  hx = std::max(std::min(static_cast<int>(std::ceil(p[3] / 2 + m * p[5] + 2)), cap), 3);
  hy = std::max(std::min(static_cast<int>(std::ceil(p[4] / 2 + m * p[6] + 2)), cap), 3);
}

void clampPar(Par &p, const LineFinderParam &P, const Bound *bd, double cmax) {
  p[0] = std::max(p[0], 0.0);
  p[7] = std::clamp(p[7], -cmax, cmax);
  p[3] = std::clamp(p[3], P.L_min, P.L_max);
  p[4] = std::clamp(p[4], P.L_min, P.L_max);
  p[5] = std::clamp(p[5], P.sigma_min, P.sigma_max);
  p[6] = std::clamp(p[6], P.sigma_min, P.sigma_max);
  if (bd) {
    if (P.dynamic_bounds) {
      // the centre within the current lengths' box around the candidate's pixel
      const double u0 = 0.5 * (bd->u_lo + bd->u_hi), v0 = 0.5 * (bd->v_lo + bd->v_hi);
      const double hu = P.centre_box_frac * p[3] + 2.5, hv = P.centre_box_frac * p[4] + 2.5;
      p[1] = std::clamp(p[1], u0 - hu, u0 + hu);
      p[2] = std::clamp(p[2], v0 - hv, v0 + hv);
    } else {
      p[1] = std::clamp(p[1], bd->u_lo, bd->u_hi);
      p[2] = std::clamp(p[2], bd->v_lo, bd->v_hi);
      p[3] = std::min(p[3], bd->Lx_hi);
      p[4] = std::min(p[4], bd->Ly_hi);
    }
  }
}

// render_models: every object's model without its constant, over the image
void renderModels(int H, int W, const std::vector<Par> &params, const LineFinderParam &P,
                  DImg &out) {
  out = DImg(H, W, 0.0);
  std::vector<double> px, py;
  for (const Par &p : params) {
    int hx, hy;
    windowOf(p, P, hx, hy);
    const int u0 = roundHalfEven(p[1]), v0 = roundHalfEven(p[2]);
    const double Lx = std::max(p[3], 1e-6), Ly = std::max(p[4], 1e-6);
    px.resize(2 * hx + 1);
    py.resize(2 * hy + 1);
    for (int j = -hx; j <= hx; ++j)
      px[j + hx] = LineFinder::profile(u0 + j - p[1], Lx, p[5]);
    for (int i = -hy; i <= hy; ++i)
      py[i + hy] = LineFinder::profile(v0 + i - p[2], Ly, p[6]);
    for (int i = -hy; i <= hy; ++i) {
      const int r = v0 + i;
      if (r < 0 || r >= H)
        continue;
      for (int j = -hx; j <= hx; ++j) {
        const int c = u0 + j;
        if (c < 0 || c >= W)
          continue;
        out.at(r, c) += p[0] * py[i + hy] * px[j + hx];
      }
    }
  }
}

// clip_expect(m, sn): E[max(0, m + noise)] and its derivative Phi(m / sn)
inline void clipExpect(double m, double sn, double &e, double &d) {
  const double z = m / sn;
  const double Phi = ndtr(z);
  e = m * Phi + sn * std::exp(-0.5 * z * z) / kSq2Pi;
  d = Phi;
}

// Gaussian elimination with partial pivoting (numpy.linalg.solve's method); false if singular
bool solve8(double A[kNPar][kNPar], double b[kNPar], double x[kNPar]) {
  int perm[kNPar];
  for (int i = 0; i < kNPar; ++i)
    perm[i] = i;
  double M[kNPar][kNPar + 1];
  for (int i = 0; i < kNPar; ++i) {
    for (int j = 0; j < kNPar; ++j)
      M[i][j] = A[i][j];
    M[i][kNPar] = b[i];
  }
  for (int col = 0; col < kNPar; ++col) {
    int piv = col;
    double best = std::fabs(M[col][col]);
    for (int r = col + 1; r < kNPar; ++r)
      if (std::fabs(M[r][col]) > best) {
        best = std::fabs(M[r][col]);
        piv = r;
      }
    if (!(best > 0.0))
      return false;
    if (piv != col)
      for (int j = 0; j <= kNPar; ++j)
        std::swap(M[col][j], M[piv][j]);
    for (int r = col + 1; r < kNPar; ++r) {
      const double f = M[r][col] / M[col][col];
      if (f == 0.0)
        continue;
      for (int j = col; j <= kNPar; ++j)
        M[r][j] -= f * M[col][j];
    }
  }
  for (int i = kNPar - 1; i >= 0; --i) {
    double s = M[i][kNPar];
    for (int j = i + 1; j < kNPar; ++j)
      s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  (void)perm;
  return std::all_of(x, x + kNPar, [](double v) { return std::isfinite(v); });
}

// fit_objects for one object: Levenberg-Marquardt on its window, the other
// objects' models (offsets, without its own model) held fixed
struct FitOut {
  double chi2 = 0.0;
  int npx = 0;
  bool conv = false;
};

FitOut fitOne(const DImg &img, const DImg *offsets, Par &p, const Bound *bd,
              const LineFinderParam &P, double sn, double cmax) {
  const int H = img.H, W = img.W;
  int hx, hy;
  windowOf(p, P, hx, hy);
  const int u0 = roundHalfEven(p[1]), v0 = roundHalfEven(p[2]);
  const int c_lo = std::max(u0 - hx, 0), c_hi = std::min(u0 + hx, W - 1);
  const int r_lo = std::max(v0 - hy, 0), r_hi = std::min(v0 + hy, H - 1);
  const int nx = std::max(c_hi - c_lo + 1, 0), ny = std::max(r_hi - r_lo + 1, 0);
  FitOut out;
  out.npx = nx * ny;
  if (out.npx == 0)
    return out;
  const bool clipped = P.clipped;
  std::vector<double> D(static_cast<size_t>(nx) * ny), O(static_cast<size_t>(nx) * ny, 0.0);
  std::vector<double> px(nx), py(ny), dpx_t(nx), dpx_L(nx), dpx_s(nx), dpy_t(ny), dpy_L(ny),
      dpy_s(ny);
  auto parts = [&](const Par &pp, bool deriv) {
    const double Lx = std::max(pp[3], 1e-6), Ly = std::max(pp[4], 1e-6);
    for (int j = 0; j < nx; ++j) {
      const double t = (c_lo + j) - pp[1];
      if (deriv)
        LineFinder::profileDeriv(t, Lx, pp[5], px[j], dpx_t[j], dpx_L[j], dpx_s[j]);
      else
        px[j] = LineFinder::profile(t, Lx, pp[5]);
    }
    for (int i = 0; i < ny; ++i) {
      const double t = (r_lo + i) - pp[2];
      if (deriv)
        LineFinder::profileDeriv(t, Ly, pp[6], py[i], dpy_t[i], dpy_L[i], dpy_s[i]);
      else
        py[i] = LineFinder::profile(t, Ly, pp[6]);
    }
  };
  for (int i = 0; i < ny; ++i)
    for (int j = 0; j < nx; ++j)
      D[static_cast<size_t>(i) * nx + j] = img.at(r_lo + i, c_lo + j);
  if (offsets) {
    parts(p, false);
    for (int i = 0; i < ny; ++i)
      for (int j = 0; j < nx; ++j)
        O[static_cast<size_t>(i) * nx + j] =
            offsets->at(r_lo + i, c_lo + j) - p[0] * py[i] * px[j];
  }
  auto cost_of = [&](const Par &pp) {
    parts(pp, false);
    double cost = 0.0;
    for (int i = 0; i < ny; ++i)
      for (int j = 0; j < nx; ++j) {
        const size_t q = static_cast<size_t>(i) * nx + j;
        const double mlin = pp[0] * py[i] * px[j] + O[q] + pp[7];
        double m = mlin, dfac;
        if (clipped)
          clipExpect(mlin, sn, m, dfac);
        const double r = D[q] - m;
        cost += r * r;
      }
    return cost;
  };
  double lam = 1e-2;
  bool done = false;
  for (int it = 0; it < P.fit_iters; ++it) {
    // the cost, the residual and the Jacobian at p
    parts(p, true);
    double JTJ[kNPar][kNPar] = {};
    double JTr[kNPar] = {};
    double cost = 0.0;
    for (int i = 0; i < ny; ++i)
      for (int j = 0; j < nx; ++j) {
        const size_t q = static_cast<size_t>(i) * nx + j;
        const double base = py[i] * px[j];
        const double mlin = p[0] * base + O[q] + p[7];
        double m = mlin, fac = 1.0;
        if (clipped)
          clipExpect(mlin, sn, m, fac);
        const double r = D[q] - m;
        cost += r * r;
        double J[kNPar];
        J[0] = base;
        J[1] = -p[0] * py[i] * dpx_t[j];
        J[2] = -p[0] * dpy_t[i] * px[j];
        J[3] = p[0] * py[i] * dpx_L[j];
        J[4] = p[0] * dpy_L[i] * px[j];
        J[5] = p[0] * py[i] * dpx_s[j];
        J[6] = p[0] * dpy_s[i] * px[j];
        J[7] = 1.0;
        for (int a = 0; a < kNPar; ++a)
          J[a] *= fac;
        for (int a = 0; a < kNPar; ++a) {
          JTr[a] += J[a] * r;
          for (int b2 = 0; b2 < kNPar; ++b2)
            JTJ[a][b2] += J[a] * J[b2];
        }
      }
    double Areg[kNPar][kNPar];
    for (int a = 0; a < kNPar; ++a)
      for (int b2 = 0; b2 < kNPar; ++b2)
        Areg[a][b2] = JTJ[a][b2];
    for (int a = 0; a < kNPar; ++a)
      Areg[a][a] += lam * std::max(JTJ[a][a], 1e-12);
    double step[kNPar] = {};
    if (!solve8(Areg, JTr, step))
      std::fill(step, step + kNPar, 0.0);
    Par pn = p;
    for (int a = 0; a < kNPar; ++a)
      pn[a] = p[a] + step[a];
    clampPar(pn, P, bd, cmax);
    const double cn = cost_of(pn);
    const bool better = cn < cost;
    const double rel = better ? (cost - cn) / std::max(cost, 1e-12) : 0.0;
    const double pa0 = p[0];
    if (better)
      p = pn;
    lam = better ? std::max(lam / 3.0, 1e-7) : std::min(lam * 4.0, 1e7);
    bool small_step = std::fabs(step[0]) < 1e-6 * std::max(pa0, 1e-9);
    for (int a = 1; a < 7; ++a)
      small_step = small_step && std::fabs(step[a]) < 1e-5;
    if ((better && rel < 1e-10 && small_step) || lam >= 1e7) {
      done = true;
      break;
    }
  }
  out.chi2 = cost_of(p) / (sn * sn);
  out.conv = done || lam < 1e7;
  return out;
}

// fit_objects(img, params, P, noise, offsets_img, fit_mask, bounds)
void fitObjects(const DImg &img, std::vector<Par> &params, const LineFinderParam &P, double sn,
                const DImg *offsets, const std::vector<char> *fit_mask,
                const std::vector<Bound> &bounds, std::vector<FitOut> *outs) {
  const double cmax = P.c_max_sigma * sn;
  const int n = static_cast<int>(params.size());
  for (int i = 0; i < n; ++i)
    clampPar(params[i], P, bounds.empty() ? nullptr : &bounds[i], cmax);
  if (outs)
    outs->assign(n, FitOut{});
#pragma omp parallel for schedule(dynamic, 8) if (!omp_in_parallel())
  for (int i = 0; i < n; ++i) {
    if (fit_mask && !(*fit_mask)[i])
      continue;
    const FitOut o = fitOne(img, offsets, params[i], bounds.empty() ? nullptr : &bounds[i], P, sn,
                            cmax);
    if (outs)
      (*outs)[i] = o;
  }
}

// _own_scores: each object's own score on resid + its own model
std::vector<double> ownScores(const DImg &resid, const std::vector<Par> &params,
                              const LineFinderParam &P, const std::vector<Template> &templates) {
  const int H = resid.H, W = resid.W;
  const int g = P.bg_guard;
  std::vector<double> out(params.size(), 0.0);
  auto nearest = [](const std::vector<double> &Ls, double x) {
    double best = Ls[0];
    for (double L : Ls)
      if (std::fabs(L - x) < std::fabs(best - x))
        best = L;
    return best;
  };
  for (size_t i = 0; i < params.size(); ++i) {
    const Par &pr = params[i];
    const double Lx = nearest(P.lengths_x, pr[3]), Ly = nearest(P.lengths_y, pr[4]);
    const Template *t = nullptr;
    for (const Template &tt : templates)
      if (tt.Lx == Lx && tt.Ly == Ly) {
        t = &tt;
        break;
      }
    if (!t)
      continue;
    const int hx = static_cast<int>(t->kx.size() / 2), hy = static_cast<int>(t->ky.size() / 2);
    const int bx = hx + g, by = hy + g;
    const int r = roundHalfEven(pr[2]), c = roundHalfEven(pr[1]);
    if (!(0 <= r && r < H && 0 <= c && c < W))
      continue;
    const int r0 = std::max(r - by, 0), r1 = std::min(r + by, H - 1);
    const int c0 = std::max(c - bx, 0), c1 = std::min(c + bx, W - 1);
    const int nx = c1 - c0 + 1, ny = r1 - r0 + 1;
    std::vector<double> img(static_cast<size_t>(nx) * ny);
    std::vector<double> px(nx), py(ny);
    for (int j = 0; j < nx; ++j)
      px[j] = LineFinder::profile((c0 + j) - pr[1], std::max(pr[3], 1e-6), pr[5]);
    for (int k = 0; k < ny; ++k)
      py[k] = LineFinder::profile((r0 + k) - pr[2], std::max(pr[4], 1e-6), pr[6]);
    double sum = 0.0;
    for (int k = 0; k < ny; ++k)
      for (int j = 0; j < nx; ++j) {
        const double v = resid.at(r0 + k, c0 + j) + pr[0] * py[k] * px[j];
        img[static_cast<size_t>(k) * nx + j] = v;
        sum += v;
      }
    const double mean = sum / (static_cast<double>(nx) * ny);
    const int rr0 = std::max(r - hy, 0), rr1 = std::min(r + hy, H - 1);
    const int cc0 = std::max(c - hx, 0), cc1 = std::min(c + hx, W - 1);
    double s = 0.0;
    for (int rr = rr0; rr <= rr1; ++rr)
      for (int cc = cc0; cc <= cc1; ++cc) {
        const double T = t->ky[rr - r + hy] * t->kx[cc - c + hx];
        s += T * (img[static_cast<size_t>(rr - r0) * nx + (cc - c0)] - mean);
      }
    out[i] = s / t->sig;
  }
  return out;
}

} // namespace

namespace LineFinder {

double profile(double t, double L, double s) {
  if (L < 1e-6) {
    const double za = (t + 0.5) / s, zb = (t - 0.5) / s;
    return s * kSq2Pi * (ndtr(za) - ndtr(zb));
  }
  const double z1 = (t + 0.5 + L / 2) / s, z2 = (t + 0.5 - L / 2) / s;
  const double z3 = (t - 0.5 + L / 2) / s, z4 = (t - 0.5 - L / 2) / s;
  const double K = s * s * kSq2Pi / L;
  return K * (psi(z1) - psi(z2) - psi(z3) + psi(z4));
}

void profileDeriv(double t, double L, double s, double &P, double &dP_dt, double &dP_dL,
                  double &dP_ds) {
  if (L < 1e-6) {
    const double za = (t + 0.5) / s, zb = (t - 0.5) / s;
    P = s * kSq2Pi * (ndtr(za) - ndtr(zb));
    const double ga = std::exp(-0.5 * za * za), gb = std::exp(-0.5 * zb * zb);
    dP_dt = ga - gb;
    dP_dL = 0.0;
    dP_ds = P / s - (za * ga - zb * gb);
    return;
  }
  const double z1 = (t + 0.5 + L / 2) / s, z2 = (t + 0.5 - L / 2) / s;
  const double z3 = (t - 0.5 + L / 2) / s, z4 = (t - 0.5 - L / 2) / s;
  const double K = s * s * kSq2Pi / L;
  P = K * (psi(z1) - psi(z2) - psi(z3) + psi(z4));
  const double F1 = ndtr(z1), F2 = ndtr(z2), F3 = ndtr(z3), F4 = ndtr(z4);
  dP_dt = K * (F1 - F2 - F3 + F4) / s;
  dP_dL = -P / L + K * (F1 + F2 - F3 - F4) / (2.0 * s);
  dP_ds = 2.0 * P / s - K * (z1 * F1 - z2 * F2 - z3 * F3 + z4 * F4) / s;
}

std::vector<LineObject> detect(const Image &image, double noise, const LineFinderParam &P) {
  const int H = image.getDimRow(), W = image.getDimCol();
  std::vector<LineObject> result;
  if (H <= 0 || W <= 0 || P.lengths_x.empty() || P.lengths_y.empty())
    return result;
  FImg I32(H, W);
  DImg I(H, W);
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      I32.at(r, c) = static_cast<float>(image(r, c));
      I.at(r, c) = static_cast<double>(I32.at(r, c));
    }
  const double sn = noise;
  // 1-3: matched filters, candidates, suppression
  FImg score;
  std::vector<int16_t> tidx;
  std::vector<Template> templates;
  matchedFilter(I32, nullptr, P, score, tidx, templates);
  std::vector<int> rows, cols;
  localMaxima(score, P.k_detect, rows, cols);
  std::vector<float> sc0(rows.size());
  std::vector<std::pair<double, double>> Ls(rows.size());
  std::vector<int> tids(rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    sc0[i] = score.at(rows[i], cols[i]);
    tids[i] = tidx[static_cast<size_t>(rows[i]) * W + cols[i]];
    Ls[i] = {templates[tids[i]].Lx, templates[tids[i]].Ly};
  }
  std::vector<int> keep = suppress(rows, cols, sc0, Ls, {}, P.nms_symmetric, P.core_px);
  std::vector<int> kr, kc, kt;
  std::vector<double> sc;
  for (int i : keep) {
    kr.push_back(rows[i]);
    kc.push_back(cols[i]);
    kt.push_back(tids[i]);
    sc.push_back(sc0[i]);
  }
  std::vector<Par> params;
  std::vector<Bound> bounds;
  initial(I, kr, kc, kt, templates, P, params, bounds);
  std::vector<int> passes(params.size(), 0);
  // 4: fit, deblending
  DImg allm;
  std::vector<FitOut> fo;
  for (int k = 0; k < P.deblend_passes; ++k) {
    renderModels(H, W, params, P, allm);
    fitObjects(I, params, P, sn, &allm, nullptr, bounds, &fo);
  }
  // 5: the second pass on the residual
  if (P.second_pass && !params.empty()) {
    renderModels(H, W, params, P, allm);
    FImg R32(H, W);
    DImg resid(H, W);
    for (int r = 0; r < H; ++r)
      for (int c = 0; c < W; ++c) {
        double m = allm.at(r, c), d;
        if (P.clipped)
          clipExpect(allm.at(r, c), sn, m, d);
        resid.at(r, c) = I.at(r, c) - m;
        R32.at(r, c) = static_cast<float>(resid.at(r, c));
      }
    FImg score2;
    std::vector<int16_t> tidx2;
    std::vector<Template> templates2;
    matchedFilter(R32, &I32, P, score2, tidx2, templates2);
    std::vector<int> r2, c2;
    localMaxima(score2, P.k_detect, r2, c2);
    std::vector<float> s2(r2.size());
    std::vector<std::pair<double, double>> Ls2(r2.size());
    std::vector<int> t2(r2.size());
    for (size_t i = 0; i < r2.size(); ++i) {
      s2[i] = score2.at(r2[i], c2[i]);
      t2[i] = tidx2[static_cast<size_t>(r2[i]) * W + c2[i]];
      Ls2[i] = {templates2[t2[i]].Lx, templates2[t2[i]].Ly};
    }
    std::vector<Accepted> existing;
    for (const Par &p : params)
      existing.push_back(Accepted{p[1], p[2], p[3], p[4]});
    std::vector<int> keep2 = suppress(r2, c2, s2, Ls2, existing, false, P.core_px + 2.0);
    if (!keep2.empty()) {
      std::vector<int> nr, nc, nt;
      for (int i : keep2) {
        nr.push_back(r2[i]);
        nc.push_back(c2[i]);
        nt.push_back(t2[i]);
        sc.push_back(s2[i]);
      }
      std::vector<Par> p2;
      std::vector<Bound> b2;
      initial(resid, nr, nc, nt, templates2, P, p2, b2);
      const size_t n_old = params.size();
      params.insert(params.end(), p2.begin(), p2.end());
      bounds.insert(bounds.end(), b2.begin(), b2.end());
      passes.insert(passes.end(), p2.size(), 1);
      // refit the new objects and the old ones whose windows reach them
      const size_t n = params.size();
      std::vector<char> fm(n, 0);
      std::vector<int> hx(n), hy(n);
      for (size_t i = 0; i < n; ++i)
        windowOf(params[i], P, hx[i], hy[i]);
      const double reach = 2.0 * P.fit_max_half;
      for (size_t i = 0; i < n; ++i) {
        if (i >= n_old) {
          fm[i] = 1;
          continue;
        }
        for (size_t q = n_old; q < n; ++q) {
          const double du = params[i][1] - params[q][1], dv = params[i][2] - params[q][2];
          if (std::sqrt(du * du + dv * dv) > reach)
            continue;
          if (std::fabs(du) <= hx[i] + hx[q] && std::fabs(dv) <= hy[i] + hy[q]) {
            fm[i] = 1;
            break;
          }
        }
      }
      for (int k = 0; k < P.deblend_passes; ++k) {
        renderModels(H, W, params, P, allm);
        fitObjects(I, params, P, sn, &allm, &fm, bounds, &fo);
      }
    }
  }
  // 6: validation
  if (P.validate && !params.empty()) {
    for (int round = 0; round < 2; ++round) {
      renderModels(H, W, params, P, allm);
      DImg resid(H, W);
      for (size_t q = 0; q < resid.d.size(); ++q)
        resid.d[q] = I.d[q] - allm.d[q];
      const std::vector<double> own = ownScores(resid, params, P, templates);
      std::vector<size_t> kept;
      for (size_t i = 0; i < params.size(); ++i)
        if (own[i] >= P.k_detect)
          kept.push_back(i);
      if (kept.size() == params.size())
        break;
      std::vector<Par> p3;
      std::vector<Bound> b3;
      std::vector<double> s3;
      std::vector<int> ps3;
      for (size_t i : kept) {
        p3.push_back(params[i]);
        b3.push_back(bounds[i]);
        s3.push_back(sc[i]);
        ps3.push_back(passes[i]);
      }
      params.swap(p3);
      bounds.swap(b3);
      sc.swap(s3);
      passes.swap(ps3);
      renderModels(H, W, params, P, allm);
      fitObjects(I, params, P, sn, &allm, nullptr, bounds, &fo);
    }
  }
  // the final fit
  fo.assign(params.size(), FitOut{});
  if (!params.empty()) {
    renderModels(H, W, params, P, allm);
    fitObjects(I, params, P, sn, &allm, nullptr, bounds, &fo);
  }
  result.resize(params.size());
  for (size_t i = 0; i < params.size(); ++i) {
    const Par &p = params[i];
    LineObject &o = result[i];
    o.A = p[0];
    o.u = p[1];
    o.v = p[2];
    o.Lx = p[3];
    o.Ly = p[4];
    o.sx = p[5];
    o.sy = p[6];
    o.c = p[7];
    o.flux = 2.0 * 3.141592653589793 * p[0] * p[5] * p[6];
    o.peak = p[0] * profile(0.0, std::max(p[3], 1e-6), p[5]) * profile(0.0, std::max(p[4], 1e-6), p[6]);
    o.score = sc[i];
    o.chi2 = fo[i].chi2;
    o.npx = fo[i].npx;
    o.pass = passes[i];
  }
  return result;
}

double estimateNoiseClipped(const Image &img, double max_k) {
  const int H = img.getDimRow(), W = img.getDimCol();
  std::vector<double> h(256, 0.0);
  std::vector<double> pos;
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < W; ++c) {
      const int v = static_cast<int>(std::lround(img(r, c)));
      if (v >= 0 && v < 256)
        h[v] += 1.0;
      if (v > 0)
        pos.push_back(v);
    }
  double sig = 1.0;
  if (!pos.empty()) {
    // numpy.percentile(v[v > 0], 40), linear interpolation
    std::sort(pos.begin(), pos.end());
    const double rank = 0.40 * (pos.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = std::min(lo + 1, pos.size() - 1);
    const double q = pos[lo] + (rank - lo) * (pos[hi] - pos[lo]);
    sig = std::max(1.0, q / 0.5244);
  }
  for (int it = 0; it < 6; ++it) {
    const int top = std::max(4, roundHalfEven(max_k * sig));
    // weighted least squares of log(y) on (x^2, x, 1), weights sqrt(y)
    double M[3][3] = {}, b[3] = {};
    int n_ok = 0;
    for (int x = 1; x <= top && x < 256; ++x) {
      const double y = h[x];
      if (!(y > 0.0))
        continue;
      ++n_ok;
      const double w2 = y; // (sqrt(y))^2
      const double v[3] = {static_cast<double>(x) * x, static_cast<double>(x), 1.0};
      const double ly = std::log(y);
      for (int a = 0; a < 3; ++a) {
        b[a] += w2 * v[a] * ly;
        for (int c2 = 0; c2 < 3; ++c2)
          M[a][c2] += w2 * v[a] * v[c2];
      }
    }
    if (n_ok < 3)
      break;
    // solve 3 x 3 by Cramer's rule
    auto det3 = [](double A[3][3]) {
      return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
             A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
             A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    };
    const double d = det3(M);
    if (!(std::fabs(d) > 0.0))
      break;
    double M0[3][3];
    for (int a = 0; a < 3; ++a)
      for (int c2 = 0; c2 < 3; ++c2)
        M0[a][c2] = (c2 == 0) ? b[a] : M[a][c2];
    const double c0 = det3(M0) / d;
    if (c0 >= 0.0)
      break;
    const double nsig = std::sqrt(-1.0 / (2.0 * c0));
    if (std::fabs(nsig - sig) < 1e-3 * sig) {
      sig = nsig;
      break;
    }
    sig = nsig;
  }
  return sig;
}

} // namespace LineFinder
