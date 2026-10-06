// Unit tests of Matrix storage for the stb_improve campaign (test U1, plan A1).
//
// Built against the working tree's Matrix (inline storage, OPENLPT_MATRIX_INLINE defined by Matrix.h) and against
// HEAD's Matrix (heap storage only).  Every value test runs on both, so the two must agree; the storage checks
// (where data() points) run only when the inline storage is present.
//
// Each check prints FAIL with its line on failure; the program prints a summary and returns the number of failures.

#include "Matrix.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

static int g_checks = 0;
static int g_fail = 0;
#define CHECK(cond)                                                                                     \
  do {                                                                                                  \
    ++g_checks;                                                                                         \
    if (!(cond)) {                                                                                      \
      ++g_fail;                                                                                         \
      std::cout << "FAIL line " << __LINE__ << ": " #cond << std::endl;                                 \
    }                                                                                                   \
  } while (0)

#ifdef OPENLPT_MATRIX_INLINE
static constexpr bool kHasInline = true;
static constexpr int kInline = OPENLPT_MATRIX_INLINE;
#else
static constexpr bool kHasInline = false;
static constexpr int kInline = 0;
#endif

// true when the elements live inside the object itself
template <class M> bool isInline(const M &m) {
  const char *p = reinterpret_cast<const char *>(m.data());
  const char *o = reinterpret_cast<const char *>(&m);
  return p >= o && p < o + sizeof(M);
}

// storage expected for a matrix with n elements
template <class M> void checkStorage(const M &m, int n, int line) {
  if (!kHasInline)
    return;
  ++g_checks;
  const bool want = n <= kInline;
  if (isInline(m) != want) {
    ++g_fail;
    std::cout << "FAIL line " << line << ": storage of a " << n << "-element matrix is "
              << (isInline(m) ? "inline" : "heap") << std::endl;
  }
}
#define CHECK_STORAGE(m, n) checkStorage((m), (n), __LINE__)

// a matrix with distinct elements v0 + i
Matrix<double> filled(int r, int c, double v0) {
  Matrix<double> m(r, c, 0.0);
  for (int i = 0; i < r * c; ++i)
    m[i] = v0 + i;
  return m;
}

bool sameValues(const Matrix<double> &a, const Matrix<double> &b) {
  if (a.getDimRow() != b.getDimRow() || a.getDimCol() != b.getDimCol())
    return false;
  for (int i = 0; i < a.getDimRow() * a.getDimCol(); ++i)
    if (a[i] != b[i]) // exact: the same operations in the same order
      return false;
  return true;
}

void testCreate() {
  const int shapes[][2] = {{0, 0}, {1, 1}, {2, 1}, {1, 2}, {3, 1}, {1, 3}, {2, 2}, {4, 2},
                           {3, 3}, {1, 9}, {9, 1}, {2, 5}, {5, 2}, {4, 4}, {16, 1}, {37, 23}};
  for (auto &s : shapes) {
    Matrix<double> m(s[0], s[1], 1.5);
    CHECK(m.getDimRow() == s[0] && m.getDimCol() == s[1]);
    CHECK_STORAGE(m, s[0] * s[1]);
    bool all = true;
    for (int i = 0; i < s[0] * s[1]; ++i)
      all = all && m[i] == 1.5;
    CHECK(all);
  }
  Matrix<int> big(1024, 1024, 255);
  CHECK_STORAGE(big, 1024 * 1024);
  CHECK(big(1023, 1023) == 255 && big(0, 0) == 255);

  Matrix<double> lst({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  CHECK_STORAGE(lst, 9);
  CHECK(lst(2, 1) == 8 && lst(0, 2) == 3);

  std::istringstream is("1,2,3,4,5,6");
  Matrix<double> fromStream(2, 3, is);
  CHECK(fromStream(1, 2) == 6 && fromStream(0, 0) == 1);
  CHECK_STORAGE(fromStream, 6);

  Pt3D p(1, 2, 3), p0;
  Pt2D q(4, 5), q0;
  CHECK(p[0] == 1 && p[1] == 2 && p[2] == 3 && p0[2] == 0);
  CHECK(q[0] == 4 && q[1] == 5 && q0[1] == 0);
  CHECK_STORAGE(p, 3);
  CHECK_STORAGE(q, 2);
  Pt3D pm(filled(3, 1, 7)); // from a Matrix
  CHECK(pm[0] == 7 && pm[2] == 9);
  Image img(480, 640, 3.0);
  CHECK_STORAGE(img, 480 * 640);
  CHECK(img(479, 639) == 3.0);
  Image img1; // 1x1
  CHECK_STORAGE(img1, 1);

  // a negative size fails as new T[n] did
#if defined(__SANITIZE_ADDRESS__) && !defined(OPENLPT_MATRIX_INLINE)
  // (skipped: HEAD's create() calls new T[-3], which AddressSanitizer aborts on instead of throwing)
#else
  bool threw = false;
  try {
    Matrix<double> bad(-1, 3, 0.0);
  } catch (const std::bad_array_new_length &) {
    threw = true;
  }
  CHECK(threw);
#endif
}

void testCopy() {
  for (int n : {1, 2, 3, 6, 9, 10, 12, 100}) {
    auto src = std::make_unique<Matrix<double>>(filled(n, 1, 10.0 * n));
    Matrix<double> cp(*src);
    CHECK(sameValues(cp, *src));
    CHECK(cp.data() != src->data()); // own storage, never the source's
    CHECK_STORAGE(cp, n);
    src.reset(); // the copy must not depend on the source
    bool ok = true;
    for (int i = 0; i < n; ++i)
      ok = ok && cp[i] == 10.0 * n + i;
    CHECK(ok);
  }
  Matrix<double> empty;
  Matrix<double> cpEmpty(empty);
  CHECK(cpEmpty.getDimRow() == 0 && cpEmpty.getDimCol() == 0);
}

void testAssign() {
  // every storage change of operator=, the source left intact
  const int cases[][4] = {
      {3, 1, 3, 1},   // small -> small, same shape
      {3, 1, 2, 1},   // small -> small, other shape
      {2, 2, 5, 5},   // small -> large
      {6, 6, 3, 1},   // large -> small
      {6, 6, 6, 6},   // large -> large, same shape (the heap block is reused)
      {6, 6, 10, 3},  // large -> large, other shape
      {3, 3, 1, 9},   // small -> small, same size, other shape
  };
  for (auto &c : cases) {
    Matrix<double> dst = filled(c[0], c[1], 100.0);
    const double *before = dst.data();
    Matrix<double> src = filled(c[2], c[3], -50.0);
    dst = src;
    CHECK(sameValues(dst, src));
    CHECK(dst.data() != src.data());
    CHECK_STORAGE(dst, c[2] * c[3]);
    if (c[0] == c[2] && c[1] == c[3])
      CHECK(dst.data() == before); // same shape: elements copied in place
    CHECK(sameValues(src, filled(c[2], c[3], -50.0)));
  }
  // self-assignment, small and large
  for (int n : {3, 50}) {
    Matrix<double> m = filled(n, 1, 1.0);
    Matrix<double> &alias = m;
    m = alias;
    CHECK(sameValues(m, filled(n, 1, 1.0)));
    CHECK_STORAGE(m, n);
  }
  // assignment to and from an empty matrix
  Matrix<double> e1, e2;
  e1 = e2;
  CHECK(e1.getDimRow() == 0);
  Matrix<double> e3;
  e3 = filled(4, 4, 2.0);
  CHECK(sameValues(e3, filled(4, 4, 2.0)));
  // the derived types
  Pt3D a(1, 2, 3), b(4, 5, 6);
  a = b;
  CHECK(a[0] == 4 && a[2] == 6 && a.data() != b.data());
  Pt2D c(1, 2), d(3, 4);
  c = d;
  CHECK(c[1] == 4);
  Image i1(10, 10, 1.0), i2(20, 30, 2.0);
  i1 = i2;
  CHECK(i1.getDimRow() == 20 && i1(19, 29) == 2.0);
  i2 = Image(); // large -> 1x1
  CHECK(i2.getDimRow() == 1);
  CHECK_STORAGE(i2, 1);
}

// reference product with plain loops, in the same order as Matrix::operator*
Matrix<double> refProduct(const Matrix<double> &a, const Matrix<double> &b) {
  Matrix<double> r(a.getDimRow(), b.getDimCol(), 0.0);
  for (int i = 0; i < a.getDimRow(); ++i)
    for (int j = 0; j < b.getDimCol(); ++j)
      for (int k = 0; k < a.getDimCol(); ++k)
        r(i, j) += a(i, k) * b(k, j);
  return r;
}

void testTimesAssign() {
  // operator*= is the one function that changes a matrix's shape in place
  const int shapes[][3] = {
      {3, 3, 4}, // 9 -> 12: inline -> heap
      {2, 4, 1}, // 8 -> 2: inline -> inline
      {4, 4, 2}, // 16 -> 8: heap -> inline
      {3, 3, 3}, // 9 -> 9: same shape
      {5, 5, 4}, // 25 -> 20: heap -> heap
      {1, 3, 1}, // 3 -> 1
  };
  for (auto &s : shapes) {
    Matrix<double> a = filled(s[0], s[1], 0.5);
    Matrix<double> b = filled(s[1], s[2], -1.25);
    Matrix<double> want = refProduct(a, b);
    a *= b;
    CHECK(sameValues(a, want));
    CHECK_STORAGE(a, s[0] * s[2]);
    Matrix<double> prod = filled(s[0], s[1], 0.5) * b;
    CHECK(sameValues(prod, want));
  }
}

void testArithmetic() {
  for (int n : {2, 3, 9, 12, 64}) {
    Matrix<double> a = filled(n, 1, 1.0), b = filled(n, 1, 0.25);
    Matrix<double> s = a + b, d = a - b, m = a * 2.0, q = a / 4.0, ps = a + 1.0, ms = a - 1.0;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
      ok = ok && s[i] == a[i] + b[i] && d[i] == a[i] - b[i] && m[i] == a[i] * 2.0 && q[i] == a[i] / 4.0;
      ok = ok && ps[i] == a[i] + 1.0 && ms[i] == a[i] - 1.0;
    }
    CHECK(ok);
    Matrix<double> c = a;
    c += b;
    c -= b;
    c *= 3.0;
    c /= 3.0;
    c += 2.0;
    c -= 2.0;
    double norm2 = 0;
    for (int i = 0; i < n; ++i)
      norm2 += a[i] * a[i];
    CHECK(std::fabs(c.norm() - std::sqrt(norm2)) <= 1e-12 * std::sqrt(norm2));
    CHECK(a == a && !(a != a) && a != b);
    Matrix<double> t = a.transpose();
    CHECK(t.getDimRow() == 1 && t.getDimCol() == n && t(0, n - 1) == a[n - 1]);
  }
  Pt3D x(1, 2, 3), y(4, 6, 8);
  Pt3D z = y - x; // a Matrix converted to Pt3D
  CHECK(z[0] == 3 && z[1] == 4 && z[2] == 5);
  CHECK(x * y == 1 * 4 + 2 * 6 + 3 * 8);
  Pt3D w = x * 2.0;
  CHECK(w[2] == 6);
  // (typeToInt/typeToDouble compile only from Matrix<double>, which reads the other type's private storage)
  Matrix<double> di = filled(2, 2, 1.0);
  Matrix<double> dd = di.typeToDouble();
  CHECK(dd(1, 0) == 3.0 && sameValues(dd, di));
  Matrix<double> sd(2, 2, 0.0);
  const double raw[4] = {9, 8, 7, 6};
  sd.setData(raw, 4);
  CHECK(sd(1, 1) == 6);
}

struct Holder2D {
  Pt2D p;
};
struct Holder { // like Object3D: a Pt3D, a vector of unique_ptr, move only
  Pt3D p;
  std::vector<std::unique_ptr<Holder2D>> v;
  Holder() = default;
  NONCOPYABLE_MOVABLE(Holder)
};

void testContainers() {
  // copies made when a vector grows: each element must own its storage
  std::vector<Pt3D> pts;
  for (int i = 0; i < 1000; ++i)
    pts.push_back(Pt3D(i, 2.0 * i, 3.0 * i));
  bool ok = true, own = true;
  for (int i = 0; i < 1000; ++i) {
    ok = ok && pts[i][0] == i && pts[i][1] == 2.0 * i && pts[i][2] == 3.0 * i;
    own = own && (!kHasInline || isInline(pts[i]));
  }
  CHECK(ok);
  CHECK(own);
  pts.erase(pts.begin(), pts.begin() + 500); // shifts by assignment
  CHECK(pts.front()[0] == 500 && pts.back()[0] == 999);

  std::vector<Line3D> lines;
  for (int i = 0; i < 300; ++i)
    lines.push_back(Line3D{Pt3D(i, 0, 0), Pt3D(0, i, 0)});
  ok = true;
  for (int i = 0; i < 300; ++i)
    ok = ok && lines[i].pt[0] == i && lines[i].unit_vector[1] == i;
  CHECK(ok);

  std::vector<Matrix<double>> mixed;
  for (int i = 1; i <= 40; ++i)
    mixed.push_back(filled(i, 1, i)); // inline and heap sizes mixed
  ok = true;
  for (int i = 1; i <= 40; ++i)
    ok = ok && sameValues(mixed[i - 1], filled(i, 1, i));
  CHECK(ok);
  std::swap(mixed[0], mixed[30]); // copies through a temporary
  CHECK(sameValues(mixed[0], filled(31, 1, 31)) && sameValues(mixed[30], filled(1, 1, 1)));

  std::vector<Holder> hs;
  for (int i = 0; i < 200; ++i) {
    Holder h;
    h.p = Pt3D(i, i, i);
    h.v.push_back(std::make_unique<Holder2D>(Holder2D{Pt2D(i, -i)}));
    hs.push_back(std::move(h));
  }
  ok = true;
  for (int i = 0; i < 200; ++i)
    ok = ok && hs[i].p[1] == i && hs[i].v[0]->p[1] == -i && (!kHasInline || isInline(hs[i].p));
  CHECK(ok);
  Holder moved = std::move(hs[5]);
  CHECK(moved.p[0] == 5 && moved.v.size() == 1);
}

// Threads that create, copy, assign and multiply matrices of every size at once; each must get the same checksum as
// the same work done on one thread.
double work(int seed) {
  double acc = 0;
  for (int it = 0; it < 2000; ++it) {
    const int n = 1 + (seed + it) % 20;
    Matrix<double> a = filled(n, 1, seed + it);
    Matrix<double> b(a);
    Pt3D p(it, seed, n), q = p;
    Matrix<double> c = filled(n, n, 0.001 * it);
    c *= filled(n, 1, 1.0);
    b = c;
    a = b + c;
    acc += a.norm() + q[2];
  }
  return acc;
}

void testThreads() {
  const int nt = 16;
  std::vector<double> serial(nt), parallel(nt);
  for (int t = 0; t < nt; ++t)
    serial[t] = work(t);
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t)
    th.emplace_back([t, &parallel] { parallel[t] = work(t); });
  for (auto &x : th)
    x.join();
  bool ok = true;
  for (int t = 0; t < nt; ++t)
    ok = ok && serial[t] == parallel[t];
  CHECK(ok);
}

int main() {
  std::cout << "Matrix storage: " << (kHasInline ? "inline up to " + std::to_string(kInline) : std::string("heap"))
            << ", sizeof(Matrix<double>) = " << sizeof(Matrix<double>) << ", sizeof(Pt3D) = " << sizeof(Pt3D)
            << std::endl;
  testCreate();
  testCopy();
  testAssign();
  testTimesAssign();
  testArithmetic();
  testContainers();
  testThreads();
  std::cout << g_checks << " checks, " << g_fail << " failed" << std::endl;
  return g_fail;
}
