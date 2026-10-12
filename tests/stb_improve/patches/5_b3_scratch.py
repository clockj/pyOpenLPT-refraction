"""B3: allocation-free stereo-matching search (per-thread buffers, same traversal, identical results).

Rewrites in StereoMatch.cpp: the parallel loop of match(), buildMatch (recursive DFS on per-thread buffers),
enumerateCandidatesOnCam, checkMatch, selectSecondCameraByLineLength (fixed arrays), selectNextCameraByMaxPairAngle
and IDMap::visitPointsInRowSpans (no per-call `seen` array: each observation is in exactly one cell).  The old
signatures stay as wrappers (used by the debug bindings).  StereoMatch.h: IDMap moves above StereoMatch, and
StereoMatch gets a Scratch struct and the buffer overloads.
"""
import re

H = r"D:\Refraction\pyOpenLPT\inc\libSTB\StereoMatch.h"
C = r"D:\Refraction\pyOpenLPT\src\srcSTB\StereoMatch.cpp"


def load(p):
    raw = open(p, "rb").read().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def save(p, s, crlf):
    open(p, "wb").write((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))


def func_span(s, signature_start):
    """(start, end) of the function whose definition begins with signature_start (end: after its closing brace)."""
    assert s.count(signature_start) == 1, (s.count(signature_start), signature_start)
    i = s.index(signature_start)
    j = s.index("{", i)
    depth, k, n = 0, j, len(s)
    while k < n:
        c = s[k]
        if s.startswith("//", k):
            k = s.index("\n", k)
            continue
        if s.startswith("/*", k):
            k = s.index("*/", k) + 2
            continue
        if c == '"' or c == "'":
            q = c
            k += 1
            while s[k] != q:
                k += 2 if s[k] == "\\" else 1
            k += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i, k + 1
        k += 1
    raise AssertionError("no closing brace")


def replace_func(s, signature_start, new_text):
    a, b = func_span(s, signature_start)
    return s[:a] + new_text.rstrip("\n") + s[b:]


# ============================== header ==============================
h, h_crlf = load(H)
# move IDMap's declaration (from its comment to the end of the class) above StereoMatch
idmap_start = h.index("/**\n * @brief Sparse grid (cell buckets)")
_cls = h.index("class IDMap {")
_open = h.index("{", _cls)
_d, _k = 0, _open
while True:
    if h[_k] == "{":
        _d += 1
    elif h[_k] == "}":
        _d -= 1
        if _d == 0:
            break
    _k += 1
assert h[_k:_k + 2] == "};"
idmap_end = _k + 2
idmap_block = h[idmap_start:idmap_end] + "\n\n"
h = h[:idmap_start] + h[idmap_end:]
anchor = "class StereoMatch {\n"
assert h.count(anchor) == 1
h = h.replace(anchor, idmap_block + anchor)
h = h.replace("// Forward declarations of types used here\nclass IDMap;\n\n", "")

# IDMap: the buffer overload of visitPointsInRowSpans
old = '''  void visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                             const std::vector<Line2D> &lines_px, double tol_px,
                             std::vector<int> &out_indices) const;
'''
assert h.count(old) == 1
h = h.replace(old, old + '''  // The same, with a caller's buffer for the lines' unit directions (no
  // allocation once the buffer has grown).
  void visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                             const std::vector<Line2D> &lines_px, double tol_px,
                             std::vector<int> &out_indices,
                             std::vector<Pt2D> &dirs) const;
''')

# StereoMatch: Scratch + overloads
old = '''  void buildTables() const;
'''
assert h.count(old) == 1
h = h.replace(old, old + '''
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
''')
save(H, h, h_crlf)

# ============================== source ==============================
s, c_crlf = load(C)

# ---- match(): per-thread buffers ----
old = '''#pragma omp parallel num_threads(T)
  {
    const int tid = omp_get_thread_num();
    auto &bucket = thread_buckets[tid]; // thread-local bucket (by index)

#pragma omp for schedule(dynamic, 32)
    for (int id_ref = 0; id_ref < static_cast<int>(ref_obs.size()); ++id_ref) {
      // One candidate initialized with -1 everywhere and ref_cam filled.
      std::vector<int> match_candidate_id(n_cams, -1);
      match_candidate_id[ref_cam] = id_ref;

      // Build across the first m active cams (DFS inside; uses new selection
      // rules).
      std::vector<std::vector<int>> build_candidates;
      buildMatch(match_candidate_id, build_candidates); // must be thread-safe
      SM_PROFILE_ADD(built, static_cast<long long>(build_candidates.size()));

      // Check on remaining active cams (existence / object-specific light
      // checks).
      std::vector<MatchWithScore> selected; // per-iter local
      selected.reserve(build_candidates.size());
      for (auto &cand : build_candidates) {
        double e_check = std::numeric_limits<double>::quiet_NaN();
        if (!checkMatch(static_cast<const std::vector<int> &>(cand), e_check))
          continue;
        REQUIRE(!std::isnan(e_check), ErrorCode::InvalidArgument,
                "StereoMatch::match: checkMatch returned NaN e_check.");
        selected.emplace_back(MatchWithScore{std::move(cand), e_check});
      }

      // Move this thread's selected candidates into its own bucket (no lock)
      if (!selected.empty()) {
        bucket.insert(bucket.end(), std::make_move_iterator(selected.begin()),
                      std::make_move_iterator(selected.end()));
      }
    }
  } // parallel region ends'''
assert s.count(old) == 1
s = s.replace(old, '''#pragma omp parallel num_threads(T)
  {
    const int tid = omp_get_thread_num();
    auto &bucket = thread_buckets[tid]; // thread-local bucket (by index)
    Scratch scratch;                    // this thread's search buffers
    std::vector<int> match_candidate_id;
    std::vector<std::vector<int>> build_candidates;

#pragma omp for schedule(dynamic, 32)
    for (int id_ref = 0; id_ref < static_cast<int>(ref_obs.size()); ++id_ref) {
      // One candidate initialized with -1 everywhere and ref_cam filled.
      match_candidate_id.assign(n_cams, -1);
      match_candidate_id[ref_cam] = id_ref;

      // Build across the first m active cams (DFS inside; uses new selection
      // rules).
      buildMatch(match_candidate_id, build_candidates, scratch);
      SM_PROFILE_ADD(built, static_cast<long long>(build_candidates.size()));

      // Check on remaining active cams (existence / object-specific light
      // checks); the passing candidates go to this thread's bucket, in order.
      for (auto &cand : build_candidates) {
        double e_check = std::numeric_limits<double>::quiet_NaN();
        if (!checkMatch(static_cast<const std::vector<int> &>(cand), e_check,
                        scratch))
          continue;
        REQUIRE(!std::isnan(e_check), ErrorCode::InvalidArgument,
                "StereoMatch::match: checkMatch returned NaN e_check.");
        bucket.push_back(MatchWithScore{std::move(cand), e_check});
      }
    }
  } // parallel region ends''')

# ---- buildMatch: wrapper + recursive DFS on buffers ----
s = replace_func(s, "void StereoMatch::buildMatch(\n    std::vector<int> &match_candidate_id,\n    std::vector<std::vector<int>> &build_candidates) const {", '''void StereoMatch::buildMatch(
    std::vector<int> &match_candidate_id,
    std::vector<std::vector<int>> &build_candidates) const {
  Scratch s;
  buildMatch(match_candidate_id, build_candidates, s);
}

// Build across the first m active cameras: a depth-first search over the
// candidates of one camera after another (expandMatch), on the thread's
// buffers. The order is that of the former stack version: candidates in list
// order, each one's subtree finished before the next.
void StereoMatch::buildMatch(std::vector<int> &match_candidate_id,
                             std::vector<std::vector<int>> &build_candidates,
                             Scratch &s) const {
  build_candidates.clear();

  const int n_cams = static_cast<int>(_cam_list.size());

  // ---- Extract (ref_cam, ref_id) from the incoming pattern (match() already
  // guarantees it exists) ----
  int ref_cam = -1, ref_id = -1;
  for (int c = 0; c < n_cams; ++c) {
    if (!_cam_list[c] || !_cam_list[c]->is_active)
      continue;
    if (match_candidate_id[c] >= 0) {
      ref_cam = c;
      ref_id = match_candidate_id[c];
      break;
    }
  }

  // one remaining-camera list and one candidate list per depth
  if (s.rem.size() < static_cast<size_t>(n_cams) + 1)
    s.rem.resize(static_cast<size_t>(n_cams) + 1);
  if (s.cand.size() < static_cast<size_t>(n_cams) + 1)
    s.cand.resize(static_cast<size_t>(n_cams) + 1);

  // ---- Active cams & pool (active, excluding ref) ----
  std::vector<int> &pool = s.rem[0];
  pool.clear();
  int n_active = 0;
  for (int c = 0; c < n_cams; ++c) {
    if (!_cam_list[c] || !_cam_list[c]->is_active)
      continue;
    ++n_active;
    if (c != ref_cam)
      pool.push_back(c);
  }

  const int build_cam_count =
      std::min(std::max(2, _obj_cfg._sm_param.match_cam_count), n_active);

  // ---- The path starts at (ref_cam, ref_id) ----
  const LosEntry &le = _los_tab[ref_cam][ref_id];
  if (!le.ok)
    return;
  s.path_cams.assign(1, ref_cam);
  s.path_ids.assign(1, ref_id);
  s.path_los.clear();
  s.path_los.push_back(le.los);
  s.path_pts.clear();
  s.path_pts.push_back(_obj2d_list[ref_cam][ref_id]->_pt_center);

  // Select the 2nd build camera by "shortest projected LOS segment", then
  // enumerate its candidates.
  if (pool.empty())
    return;
  const int target_cam = selectSecondCameraByLineLength(ref_cam, ref_id, pool);
  if (target_cam < 0)
    return;
  enumerateCandidatesOnCam(target_cam, s.path_cams, s.path_ids, s.path_pts,
                           s.cand[0], s);
  if (s.cand[0].empty())
    return; // cannot grow from this ref

  expandMatch(0, target_cam, build_cam_count, match_candidate_id,
              build_candidates, s);
}

// Try each candidate on target_cam after the current path (depth: the path has
// depth + 1 cameras): a quick triangulation check and the object check, then
// either a complete candidate (build_cam_count cameras) or the next camera
// (max pairwise 2D angle) and its candidates, one level deeper.
void StereoMatch::expandMatch(int depth, int target_cam, int build_cam_count,
                              const std::vector<int> &match_candidate_id,
                              std::vector<std::vector<int>> &build_candidates,
                              Scratch &s) const {
  const std::vector<int> &cands = s.cand[depth];
  const std::vector<int> &rem = s.rem[depth];
  for (size_t ci = 0; ci < cands.size(); ++ci) {
    const int pid = cands[ci];
    SM_PROFILE_ADD(dfs_steps, 1);
    const LosEntry &le = _los_tab[target_cam][pid];
    if (!le.ok)
      continue;

    // Extend path with (target_cam, pid)
    s.path_cams.push_back(target_cam);
    s.path_ids.push_back(pid);
    s.path_los.push_back(le.los);
    s.path_pts.push_back(_obj2d_list[target_cam][pid]->_pt_center);

    // ---- Early pruning: quick triangulation check with unified tolerance
    // (>=2 LOS here), then the object-specific cheap check ----
    const int k = static_cast<int>(s.path_los.size());
    const double th_min_deg = computeMinParallaxDeg(s.path_los);
    const double tol_quickmm = calTriangulateTol(
        _obj_cfg._sm_param.tol_3d_mm, k, build_cam_count, th_min_deg);
    if (TriangulationCheckWithTol(s.path_los, tol_quickmm) &&
        objectEarlyCheck(s.path_cams, s.path_ids)) {
      if (static_cast<int>(s.path_cams.size()) == build_cam_count) {
        // Complete: a cam-aligned result from the incoming pattern
        std::vector<int> ids_norm = match_candidate_id;
        for (size_t i = 0; i < s.path_cams.size(); ++i)
          ids_norm[s.path_cams[i]] = s.path_ids[i];
        build_candidates.push_back(std::move(ids_norm));
      } else {
        // The remaining cameras without target_cam (order kept), the next
        // build camera, and its candidates
        std::vector<int> &rem_next = s.rem[depth + 1];
        rem_next.clear();
        for (int c : rem)
          if (c != target_cam)
            rem_next.push_back(c);
        const int next_cam = selectNextCameraByMaxPairAngle(
            s.path_cams, s.path_ids, rem_next, s);
        if (next_cam >= 0) {
          enumerateCandidatesOnCam(next_cam, s.path_cams, s.path_ids,
                                   s.path_pts, s.cand[depth + 1], s);
          if (!s.cand[depth + 1].empty())
            expandMatch(depth + 1, next_cam, build_cam_count,
                        match_candidate_id, build_candidates, s);
        }
      }
    }

    // Undo the extension; try the next candidate
    s.path_cams.pop_back();
    s.path_ids.pop_back();
    s.path_los.pop_back();
    s.path_pts.pop_back();
  }
}''')

# ---- enumerateCandidatesOnCam: wrapper + buffers ----
s = replace_func(s, "void StereoMatch::enumerateCandidatesOnCam(\n    int target_cam, const std::vector<int> &chosen_cams,", '''void StereoMatch::enumerateCandidatesOnCam(
    int target_cam, const std::vector<int> &chosen_cams,
    const std::vector<int> &chosen_ids, const std::vector<Pt2D> &chosen_pts,
    std::vector<int> &out_candidates) const {
  Scratch s;
  enumerateCandidatesOnCam(target_cam, chosen_cams, chosen_ids, chosen_pts,
                           out_candidates, s);
}

void StereoMatch::enumerateCandidatesOnCam(
    int target_cam, const std::vector<int> &chosen_cams,
    const std::vector<int> &chosen_ids, const std::vector<Pt2D> &chosen_pts,
    std::vector<int> &out_candidates, Scratch &s) const {
  out_candidates.clear();
  SM_PROFILE_ADD(enumerations, 1);
  if (target_cam < 0 || target_cam >= static_cast<int>(_idmaps.size()))
    return;
  IDMap *idm = _idmaps[target_cam].get();
  if (!idm || chosen_cams.empty())
    return;

  // 1) The chosen observations' image lines on this camera (from the table)
  std::vector<Line2D> &lines_px = s.lines;
  lines_px.clear();
  for (size_t i = 0; i < chosen_cams.size(); ++i) {
    const LineEntry &le = _line_tab[chosen_cams[i]][target_cam][chosen_ids[i]];
    if (!le.ok)
      return;
    lines_px.push_back(le.line);
  }

  // 2) Compute per-row strip intersection in CELL indices
  std::vector<IDMap::RowSpan> &spans = s.spans;
  idm->computeStripIntersection(lines_px, _obj_cfg._sm_param.tol_2d_px, spans);

  bool any_valid = false;
  for (const auto &sp : spans) {
    if (sp.x_min <= sp.x_max) {
      any_valid = true;
      break;
    }
  }
  if (!any_valid)
    return;

  // 3) Visit-and-collect with precise distance test
  idm->visitPointsInRowSpans(spans, lines_px, _obj_cfg._sm_param.tol_2d_px,
                             out_candidates, s.dirs);
  SM_PROFILE_ADD(strip_points, static_cast<long long>(out_candidates.size()));
  if (out_candidates.empty())
    return;

  // 4) check back=projection to chosen cams & points
  int write = 0;
  for (int pid : out_candidates) {
    const Object2D *o = _obj2d_list[target_cam][pid];
    if (!o)
      continue;
    if (checkBackProjection(target_cam, pid, chosen_cams, chosen_pts)) {
      out_candidates[write++] = pid;
    }
  }
  out_candidates.resize(write);
  SM_PROFILE_ADD(candidates, write);
}''')

# ---- checkMatch: wrapper + buffers ----
s = replace_func(s, "bool StereoMatch::checkMatch(const std::vector<int> &candidate_ids,\n                             double &out_e_check) const {", '''bool StereoMatch::checkMatch(const std::vector<int> &candidate_ids,
                             double &out_e_check) const {
  Scratch s;
  return checkMatch(candidate_ids, out_e_check, s);
}

bool StereoMatch::checkMatch(const std::vector<int> &candidate_ids,
                             double &out_e_check, Scratch &s) const {
  const int n_cams = static_cast<int>(_cam_list.size());

  // ---- 1) Gather LOS from chosen (build) cameras ----
  std::vector<Line3D> &los3d = s.los3d;
  los3d.clear();
  for (int cam = 0; cam < n_cams; ++cam) {
    const int pid = candidate_ids[cam];
    if (pid < 0)
      continue; // not chosen on this cam
    const LosEntry &le = _los_tab[cam][pid];
    if (!le.ok)
      return false;
    los3d.push_back(le.los);
  }
  if (los3d.size() < 2) {
    // Should not happen if build stage produced a valid candidate,
    // but guard against malformed input.
    return false;
  }

  int n_active_cam = 0;
  for (int c = 0; c < n_cams; ++c)
    if (_cam_list[c] && _cam_list[c]->is_active)
      ++n_active_cam;
  if (los3d.size() ==
      n_active_cam) { // all active cams are used in build → nothing to check
    Pt3D Xw{};
    myMATH::triangulation(Xw, out_e_check, los3d);
    SM_PROFILE_ADD(tri, 1);
    if (out_e_check > _obj_cfg._sm_param.tol_3d_mm)
      return false;
    if (!(_obj_cfg._sm_param.limit.check(Xw[0], Xw[1], Xw[2])))
      return false; // if out of bounds
    return true;
  }

  // Prepare chosen cams, ids & points for back-projection checks
  s.chosen_cams.clear();
  s.chosen_ids.clear();
  s.chosen_pts.clear();
  for (int c = 0; c < n_cams; ++c) {
    const int pid = candidate_ids[c];
    if (pid >= 0) {
      s.chosen_cams.push_back(c);
      s.chosen_ids.push_back(pid);
      s.chosen_pts.push_back(_obj2d_list[c][pid]->_pt_center);
    }
  }

  const double tol_3d_mm = _obj_cfg._sm_param.tol_3d_mm;

  // ---- 2) For every active camera that is NOT chosen yet (check cameras) ----
  for (int cam = 0; cam < n_cams; ++cam) {
    if (!_cam_list[cam] || !_cam_list[cam]->is_active)
      continue; // inactive -> ignore
    if (candidate_ids[cam] >= 0)
      continue; // already used in build -> skip

    // obtain points in the intersection of LOS from chosen cameras ----
    std::vector<int> &inliers = s.inliers;
    enumerateCandidatesOnCam(/*target_cam=*/cam, s.chosen_cams, s.chosen_ids,
                             s.chosen_pts, inliers, s);

    if (inliers.empty())
      return false; // this check cam has no supporting observation

    // get the best triangulation error among all inliers on this check cam
    // if the best is larger than tol_3d_mm, reject this candidate
    double best_err_c = std::numeric_limits<double>::infinity();

    for (int pid : inliers) {
      const LosEntry &lq = _los_tab[cam][pid];
      if (!lq.ok)
        continue;

      std::vector<Line3D> &los_all = s.los_all;
      los_all.assign(los3d.begin(), los3d.end());
      los_all.push_back(lq.los);

      Pt3D Xw{};
      double err = 0.0;
      myMATH::triangulation(Xw, err, los_all);
      SM_PROFILE_ADD(tri, 1);

      if (!std::isfinite(err))
        continue;
      if (!(_obj_cfg._sm_param.limit.check(Xw[0], Xw[1], Xw[2])))
        continue; // if out of bounds

      if (err < best_err_c)
        best_err_c = err;
    }

    if (!std::isfinite(best_err_c) || best_err_c > tol_3d_mm)
      return false; // if the best is larger than tol_3d_mm, reject this
                    // candidate
    if (!std::isfinite(out_e_check) || best_err_c > out_e_check)
      out_e_check = best_err_c; // E_check = max(best_err_c) over check cameras
  }

  // ---- 6) All check cameras have ≥1 supporting point ----
  return true;
}''')

# ---- selectSecondCameraByLineLength: fixed arrays instead of vectors ----
old = '''    auto in = [](double v, double lo, double hi) { return v >= lo && v <= hi; };
    auto push_if = [&](double x, double y, std::vector<Pt2D> &v) {
      if (in(x, 0.0, W) && in(y, 0.0, H))
        v.emplace_back(x, y);
    };

    std::vector<Pt2D> pts;
    pts.reserve(4);
    if (std::fabs(ux) > 1e-12) {
      double t = (0.0 - px) / ux;
      push_if(0.0, py + t * uy, pts);
      t = (W - 1 - px) / ux;
      push_if(W - 1, py + t * uy, pts);
    }
    if (std::fabs(uy) > 1e-12) {
      double t = (0.0 - py) / uy;
      push_if(px + t * ux, 0.0, pts);
      t = (H - 1 - py) / uy;
      push_if(px + t * ux, H - 1, pts);
    }
    // de-dup corners
    const double eps = 1e-9;
    std::vector<Pt2D> uniq;
    uniq.reserve(pts.size());
    for (auto &q : pts) {
      bool dup = false;
      for (auto &p : uniq)
        if (std::fabs(p[0] - q[0]) < eps && std::fabs(p[1] - q[1]) < eps) {
          dup = true;
          break;
        }
      if (!dup)
        uniq.push_back(q);
    }
    if (uniq.size() < 2)
      return 0.0;
    const double dx = uniq[0][0] - uniq[1][0], dy = uniq[0][1] - uniq[1][1];
    return std::sqrt(dx * dx + dy * dy);'''
assert s.count(old) == 1
s = s.replace(old, '''    auto in = [](double v, double lo, double hi) { return v >= lo && v <= hi; };
    // the line's crossings of the image border (at most 4), in fixed arrays
    double pts_x[4], pts_y[4];
    int n_pts = 0;
    auto push_if = [&](double x, double y) {
      if (in(x, 0.0, W) && in(y, 0.0, H)) {
        pts_x[n_pts] = x;
        pts_y[n_pts] = y;
        ++n_pts;
      }
    };

    if (std::fabs(ux) > 1e-12) {
      double t = (0.0 - px) / ux;
      push_if(0.0, py + t * uy);
      t = (W - 1 - px) / ux;
      push_if(W - 1, py + t * uy);
    }
    if (std::fabs(uy) > 1e-12) {
      double t = (0.0 - py) / uy;
      push_if(px + t * ux, 0.0);
      t = (H - 1 - py) / uy;
      push_if(px + t * ux, H - 1);
    }
    // de-dup corners
    const double eps = 1e-9;
    double uniq_x[4], uniq_y[4];
    int n_uniq = 0;
    for (int a = 0; a < n_pts; ++a) {
      bool dup = false;
      for (int b = 0; b < n_uniq; ++b)
        if (std::fabs(uniq_x[b] - pts_x[a]) < eps &&
            std::fabs(uniq_y[b] - pts_y[a]) < eps) {
          dup = true;
          break;
        }
      if (!dup) {
        uniq_x[n_uniq] = pts_x[a];
        uniq_y[n_uniq] = pts_y[a];
        ++n_uniq;
      }
    }
    if (n_uniq < 2)
      return 0.0;
    const double dx = uniq_x[0] - uniq_x[1], dy = uniq_y[0] - uniq_y[1];
    return std::sqrt(dx * dx + dy * dy);''')

# ---- selectNextCameraByMaxPairAngle: wrapper + buffer ----
old = '''int StereoMatch::selectNextCameraByMaxPairAngle(
    const std::vector<int> &chosen_cams, const std::vector<int> &chosen_ids,
    const std::vector<int> &remaining_cams) const {
  if (remaining_cams.empty())
    return -1;

  int best_cam = -1;
  double best_score = -1.0;

  std::vector<Line2D> lines_px; // reused buffer
'''
assert s.count(old) == 1
s = s.replace(old, '''int StereoMatch::selectNextCameraByMaxPairAngle(
    const std::vector<int> &chosen_cams, const std::vector<int> &chosen_ids,
    const std::vector<int> &remaining_cams) const {
  Scratch s;
  return selectNextCameraByMaxPairAngle(chosen_cams, chosen_ids, remaining_cams,
                                        s);
}

int StereoMatch::selectNextCameraByMaxPairAngle(
    const std::vector<int> &chosen_cams, const std::vector<int> &chosen_ids,
    const std::vector<int> &remaining_cams, Scratch &s) const {
  if (remaining_cams.empty())
    return -1;

  int best_cam = -1;
  double best_score = -1.0;

  std::vector<Line2D> &lines_px = s.sel_lines; // the thread's buffer
''')

# ---- IDMap::visitPointsInRowSpans: the buffer overload, without the seen array ----
s = replace_func(s, "void IDMap::visitPointsInRowSpans(const std::vector<RowSpan> &spans,", '''void IDMap::visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                                  const std::vector<Line2D> &lines_px,
                                  double tol_px,
                                  std::vector<int> &out_indices) const {
  std::vector<Pt2D> dirs;
  visitPointsInRowSpans(spans, lines_px, tol_px, out_indices, dirs);
}

// Each observation is in exactly one cell (rebuild), and each cell is visited
// at most once (one span per row), so no point can be met twice.
void IDMap::visitPointsInRowSpans(const std::vector<RowSpan> &spans,
                                  const std::vector<Line2D> &lines_px,
                                  double tol_px, std::vector<int> &out_indices,
                                  std::vector<Pt2D> &dirs) const {
  out_indices.clear();

  if (!_objs || lines_px.empty() || spans.empty())
    return;

  // Precompute unit directions (required by cross-product distance)
  dirs.resize(lines_px.size());
  for (size_t i = 0; i < lines_px.size(); ++i)
    dirs[i] = normalized(lines_px[i].unit_vector);

  const double tol2 = tol_px * tol_px;
  const auto &objs = *_objs;

  // Iterate every row of cells
  const int rowsC = _rows_cell;
  for (int cy = 0; cy < rowsC; ++cy) {
    const RowSpan &row = spans[cy];
    if (row.x_min > row.x_max)
      continue; // invalid span -> nothing on this row

    // Iterate each cell in the inclusive [x_min, x_max] range
    for (int cx = row.x_min; cx <= row.x_max; ++cx) {
      const auto &cell = _buckets[idx(cy, cx)];
      for (int pid : cell) {
        if (pid < 0 || static_cast<size_t>(pid) >= objs.size())
          continue;

        const Pt2D &q = objs[pid]->_pt_center;

        // Precise distance check w.r.t. all lines: |(q - p) x u| <= tol
        for (size_t i = 0; i < lines_px.size(); ++i) {
          const Pt2D &Lp = lines_px[i].pt;
          const Pt2D &u = dirs[i];
          const double dx = q[0] - Lp[0];
          const double dy = q[1] - Lp[1];
          const double z =
              dy * u[0] -
              dx * u[1]; // signed 2D cross (perp distance if u is unit)
          if (z * z > tol2)
            goto reject_point;
        }

        // All distances within tol -> accept
        out_indices.push_back(pid);

      reject_point:;
      }
    }
  }
}''')

save(C, s, c_crlf)
print("B3 patched; crlf", h_crlf, c_crlf)
