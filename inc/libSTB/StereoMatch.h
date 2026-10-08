
/**
 * StereoMatch
 * Pipeline: build (first m active cams) -> check (remaining active cams)
 *           -> prune (global disjoint selection) -> triangulate (final
 * Object3D).
 *
 * Notes:
 * - A 2D observation may be reused across different matches until the prune
 * step.
 * - Per-call inputs are bound at the start of match() via internal pointers.
 *   Do not call match() concurrently on the same instance.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex> // for std::mutex
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "Camera.h" // Camera model interface
#include "Config.h"
#include "Matrix.h"
#include "ObjectInfo.h"
#include "STBCommons.h"
#include "myMATH.h" // Pt2D, Pt3D, Line2D, Line3D

/**
 * @brief Sparse grid (cell buckets) over an image for fast 2D candidate lookup.
 *        Cells are axis-aligned squares of size _cell_px (in pixels).
 *        Each cell stores the indices (pids) of 2D observations.
 */
// IDMap.hpp

/**
 * @brief An observation's image curve in another camera: the straight line
 *        `line`, or, for cameras whose line images curve (curved ports), the
 *        polyline `poly` (vertices in order along the line of sight; a NaN
 *        vertex breaks it), with `line` then its chord. Views, not owners.
 */
struct ImageCurve {
  const Line2D *line = nullptr;
  const std::vector<Pt2D> *poly = nullptr; // null or empty: the straight line
};

/** @brief Squared distance from q to a polyline (NaN vertices break it);
 *         infinity when it has no segment. */
double dist2ToPolyline(const Pt2D &q, const std::vector<Pt2D> &poly);

class IDMap {
public:
  struct RowSpan {
    int x_min, x_max;
  }; // invalid if x_min > x_max

  IDMap(int img_rows_px, int img_cols_px, int cell_px);

  // Rebuild the buckets for this camera using a contiguous array of 2D objects.
  // 'objs' must outlive the IDMap while you iterate.
  void rebuild(const std::vector<const Object2D *> &objs);

  // Compute per-row intersection of K LOS strips; result in CELL indices.
  // 'spans' is resized to rowsCell(), each entry holds [cx_min, cx_max]
  // (inclusive).
  void computeStripIntersection(const std::vector<Line2D> &lines_px,
                                double tol_px,
                                std::vector<RowSpan> &spans) const;

  // Enumerate points ONLY inside the final per-row spans, with precise
  // point-to-line checks. Geometry check: for each point q, require
  // distance_to_every_line(q) <= tol_px. Each point is in exactly one cell
  // (rebuild) and each cell is visited at most once, so none is reported twice.
  void visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                             const std::vector<Line2D> &lines_px, double tol_px,
                             std::vector<int> &out_indices) const;
  // The same, with a caller's buffer for the lines' unit directions (no
  // allocation once the buffer has grown).
  void visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                             const std::vector<Line2D> &lines_px, double tol_px,
                             std::vector<int> &out_indices,
                             std::vector<Pt2D> &dirs) const;

  // The same two for image curves: a straight curve as above, a polyline by
  // the union of its segments' strips (each limited to the segment's extent)
  // and the distance to its nearest segment. Used when any curve is a
  // polyline; with straight lines only the functions above are called, so
  // their results do not change.
  void computeStripIntersection(const std::vector<ImageCurve> &curves,
                                double tol_px,
                                std::vector<RowSpan> &spans) const;
  void visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                             const std::vector<ImageCurve> &curves,
                             double tol_px,
                             std::vector<int> &out_indices) const;

  // ---- small helpers (inline) ----
  inline int rowsCell() const { return _rows_cell; }
  inline int colsCell() const { return _cols_cell; }
  inline int cellSizePx() const { return _cell_px; }

private:
  // Row-major bucket index
  inline int idx(int cy, int cx) const { return cy * _cols_cell + cx; }

  // Normalize a 2D vector defensively (used inside .cpp)
  static Pt2D normalized(const Pt2D &v);

private:
  int _img_rows_px = 0;
  int _img_cols_px = 0;
  int _cell_px = 1;
  int _rows_cell = 0;
  int _cols_cell = 0;

  // Buckets: one vector<int> per cell (row-major)
  std::vector<std::vector<int>> _buckets;

  // Pointer to external storage of Object2D* for coordinate access during
  // enumeration
  const std::vector<const Object2D *> *_objs = nullptr;
};

class StereoMatch {
public:
  // Constructor
  explicit StereoMatch(
      const std::vector<std::shared_ptr<Camera>> &camera_models,
      const std::vector<std::vector<std::unique_ptr<Object2D>>> &obj2d_list,
      const ObjectConfig
          &obj_cfg); // All reference variable should be initialized first

  // Main entry
  std::vector<std::unique_ptr<Object3D>> match() const;

private:
  // ---- bound per-call inputs (read-only during a match) ----
  const std::vector<std::shared_ptr<Camera>> &_cam_list;
  const ObjectConfig &_obj_cfg;
  std::vector<std::vector<const Object2D *>> _obj2d_list;

  // ---- shared accelerators usable by build/check during a match ----
  std::vector<std::unique_ptr<IDMap>>
      _idmaps; // one per camera (nullptr if inactive)
  // _idmaps must be built in the contructor, because match() const cannot
  // change any member variable due to "const".

  // ---- profiling (setting "profile"): work counters of one match() ----
  // One entry per thread, each on its own cache line; summed and printed at the
  // end of match().
  struct alignas(64) ProfileCounters {
    long long los = 0;          // lines of sight computed
    long long proj = 0;         // camera projections
    long long tri = 0;          // triangulations
    long long enumerations = 0; // enumerateCandidatesOnCam calls
    long long strip_points = 0; // points inside all strips (before back-check)
    long long candidates = 0;   // points kept after the back-projection check
    long long dfs_steps = 0;    // candidates tried by the depth-first search
    long long built = 0;        // candidates built (before the check cameras)
  };
  bool _profile = false;
  mutable std::vector<ProfileCounters> _prof;

  // ---- per-match tables: the line of sight of every 2D observation of the
  // active cameras, and its image line on every other active camera, computed
  // once at the start of match() (in parallel); the search reads them instead
  // of recomputing them for every candidate. Same functions, same inputs, so
  // the same values as computing them on demand.
  struct LosEntry {
    Line3D los;
    bool ok = false; // false: lineOfSight failed
  };
  struct LineEntry {
    Line2D line;
    bool ok = false; // false: the projection failed
    // Cameras whose line images curve (Camera::lineImagesCurve): the image as
    // a polyline (makeCurve2DFromLOS3D), `line` its chord. Empty otherwise.
    std::vector<Pt2D> poly;
  };
  mutable std::vector<std::vector<LosEntry>> _los_tab; // [cam][pid]
  mutable std::vector<std::vector<std::vector<LineEntry>>>
      _line_tab; // [cam][target cam][pid]: image line on the target camera
  void buildTables() const;

  // ---- per-thread buffers of the search: reused from one candidate to the
  // next, so the search does not allocate once they have grown ----
  struct Scratch {
    // depth-first search: the path, and per depth the remaining cameras and
    // the candidates
    std::vector<int> path_cams, path_ids;
    std::vector<Pt2D> path_pts;
    std::vector<Line3D> path_los;
    std::vector<std::vector<int>> rem, cand;
    // enumerateCandidatesOnCam
    std::vector<Line2D> lines;
    std::vector<ImageCurve> curves;
    std::vector<IDMap::RowSpan> spans;
    std::vector<Pt2D> dirs;
    // checkMatch
    std::vector<Line3D> los3d, los_all;
    std::vector<int> chosen_cams, chosen_ids, inliers;
    std::vector<Pt2D> chosen_pts;
    // selectNextCameraByMaxPairAngle
    std::vector<Line2D> sel_lines;
  };
  void buildMatch(std::vector<int> &match_candidate_id,
                  std::vector<std::vector<int>> &build_candidates,
                  Scratch &s) const;
  void expandMatch(int depth, int target_cam, int build_cam_count,
                   const std::vector<int> &match_candidate_id,
                   std::vector<std::vector<int>> &build_candidates,
                   Scratch &s) const;
  void enumerateCandidatesOnCam(int target_cam,
                                const std::vector<int> &chosen_cams,
                                const std::vector<int> &chosen_ids,
                                const std::vector<Pt2D> &chosen_pts,
                                std::vector<int> &out_candidates,
                                Scratch &s) const;
  bool checkMatch(const std::vector<int> &candidate_ids, double &out_e_check,
                  Scratch &s) const;
  int selectNextCameraByMaxPairAngle(const std::vector<int> &chosen_cams,
                                     const std::vector<int> &chosen_ids,
                                     const std::vector<int> &remaining_cams,
                                     Scratch &s) const;

  // ---- main pipeline pieces ----
  void buildMatch(std::vector<int> &match_candidate_id,
                  std::vector<std::vector<int>> &build_candidates) const;

  // Enumerate candidates on a specific camera: the points within tol_2d_px of
  // the image lines of all chosen observations (cams, ids), that pass the
  // back-projection check against the chosen points.
  void enumerateCandidatesOnCam(int target_cam,
                                const std::vector<int> &chosen_cams,
                                const std::vector<int> &chosen_ids,
                                const std::vector<Pt2D> &chosen_pts,
                                std::vector<int> &out_candidates) const;

  bool checkMatch(const std::vector<int> &candidate_ids,
                  double &out_e_check)
      const; // out_e_check: the triangulation error obtained during checking

  std::vector<std::vector<int>>
  pruneMatch(const std::vector<std::vector<int>> &match_candidates,
             const std::vector<double> &e_checks)
      const; // e_checks is used to prune matches

  std::vector<std::unique_ptr<Object3D>>
  triangulateMatch(const std::vector<std::vector<int>> &selected_matches) const;

  // ---- camera selection helpers ----
  int selectSecondCameraByLineLength(
      int ref_cam, int ref_id, const std::vector<int> &remaining_cams) const;

  int selectNextCameraByMaxPairAngle(
      const std::vector<int> &chosen_cams, const std::vector<int> &chosen_ids,
      const std::vector<int> &remaining_cams) const;

  // ---- LOS→image line helper (fills the tables) ----
  bool makeLine2DFromLOS3D(int cam_id, const Line3D &los, Line2D &out_line) const;
  // The same for a camera whose line images curve: the line of sight clipped
  // to the view volume and to where the camera's model applies, projected at
  // points subdivided until every chord stays within kCurveTolPx of the curve.
  bool makeCurve2DFromLOS3D(int cam_id, const Line3D &los, Line2D &out_chord,
                            std::vector<Pt2D> &out_poly) const;

  // ---- early checks & tolerances (decl only; you已有实现或后续实现) ----
  double computeMinParallaxDeg(const std::vector<Line3D> &los3d) const;
  double calTriangulateTol(double final_tol_3d_mm, int k_selected, int k_target,
                           double min_parallax_deg) const;
  bool TriangulationCheckWithTol(const std::vector<Line3D> &los3d,
                                 double tol_3d_mm) const;
  double triangulationVariance(const std::vector<Line3D> &los) const;

  // project the line of sight of point pid on target_cam back to all chosen
  // cams, check the distance to chosen_pts
  bool checkBackProjection(int target_cam, int pid,
                           const std::vector<int> &chosen_cams,
                           const std::vector<Pt2D> &chosen_pts) const;

  bool objectEarlyCheck(const std::vector<int> &cams_used,
                        const std::vector<int> &ids_on_used) const;
  bool tracerEarlyCheck(const std::vector<int> &cams_in_path,
                        const std::vector<int> &ids_in_path) const;
  bool bubbleEarlyCheck(const std::vector<int> &cams_in_path,
                        const std::vector<int> &ids_in_path) const;

  FRIEND_DEBUG(StereoMatch); // for debugging private members
};


