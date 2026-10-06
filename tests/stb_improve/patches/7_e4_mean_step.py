"""E4: STB::predictNext predicts with the mean step over the last m frames (origin-free), replacing the LMS fit."""
p = r"D:\Refraction\pyOpenLPT\src\srcSTB\STB.cpp"
raw = open(p, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


rep('''  // Polynomial order for LMS/Wiener-like predictor (cap at 5)
  const int order = (n < 6) ? static_cast<int>(n) - 1 : 5;
  const size_t start = n - 1 - static_cast<size_t>(order);

  // Create a same-kind object seeded from the last one (copy type-specific
  // params)
  CreateArgs a;
  a._proto = list.back().get(); // center will be overwritten below
  auto obj3d = _obj_config->creatObject3D(std::move(a));
  if (!obj3d)
    return nullptr;

  std::vector<double> series(static_cast<size_t>(order) + 1);
  std::vector<double> filter(static_cast<size_t>(order), 0.0);

  for (int axis = 0; axis < 3; ++axis) {
    // Collect the last (order+1) samples along this axis
    for (int j = 0; j <= order; ++j) {
      series[static_cast<size_t>(j)] =
          list[start + static_cast<size_t>(j)]->_pt_center[axis];
    }

    // Numerical guard: shift if the target is too close to zero
    bool shifted = false;
    constexpr double kShift = 10.0;
    if (std::fabs(series[static_cast<size_t>(order)]) < 1.0) {
      shifted = true;
      for (double &v : series)
        v += kShift;
    }

    // LMS step size ≈ 1 / ||regressors||^2 (guard against zero)
    double denom = 0.0;
    for (int j = 0; j < order; ++j)
      denom += series[static_cast<size_t>(j)] * series[static_cast<size_t>(j)];
    const double step = (denom > 1e-12) ? (1.0 / denom) : 0.0;

    std::fill(filter.begin(), filter.end(), 0.0);

    // Fit coefficients to predict y_t from [y_{t-1}, ..., y_{t-order}]
    double prediction = 0.0;
    double error = series[static_cast<size_t>(order)] - prediction;

    int iter = 0;
    while (std::fabs(error) > SMALLNUMBER && iter < WIENER_MAX_ITER &&
           step > 0.0) {
      prediction = 0.0;
      for (int j = 0; j < order; ++j) {
        const size_t jj = static_cast<size_t>(j);
        filter[jj] += step * series[jj] * error;
        prediction += filter[jj] * series[jj];
      }
      error = series[static_cast<size_t>(order)] - prediction;
      ++iter;
    }

    // One-step-ahead prediction
    if (order >= 1 && step == 0.0) {
      // Fallback: simple linear extrapolation
      prediction = series[static_cast<size_t>(order)] +
                   (series[static_cast<size_t>(order)] -
                    series[static_cast<size_t>(order - 1)]);
    } else {
      prediction = 0.0;
      for (int j = 0; j < order; ++j)
        prediction +=
            filter[static_cast<size_t>(j)] * series[static_cast<size_t>(j + 1)];
    }

    if (shifted)
      prediction -= kShift;
    obj3d->_pt_center[axis] = prediction; // base-class field
  }

  return obj3d;
}''', '''  // Constant velocity from the mean step over the last m frames (m = 5, fewer
  // for short tracks): p = y_last + (y_last - y_(last-m)) / m on each axis.
  // It does not depend on where the world origin lies. The LMS fit it replaces
  // tends to it far from the origin, but pushed predictions away from 0 near
  // the coordinate planes (tests/stb_improve, report section 10).
  const size_t m = (n < 6) ? n - 1 : 5;

  // Create a same-kind object seeded from the last one (copy type-specific
  // params)
  CreateArgs a;
  a._proto = list.back().get(); // center will be overwritten below
  auto obj3d = _obj_config->creatObject3D(std::move(a));
  if (!obj3d)
    return nullptr;

  const Pt3D &last = list[n - 1]->_pt_center;
  const Pt3D &first = list[n - 1 - m]->_pt_center;
  for (int axis = 0; axis < 3; ++axis)
    obj3d->_pt_center[axis] =
        last[axis] + (last[axis] - first[axis]) / static_cast<double>(m);

  return obj3d;
}''')

out = s.replace("\n", "\r\n") if crlf else s
open(p, "wb").write(out.encode("utf-8"))
print("patched", p, "(CRLF)" if crlf else "(LF)")
