"""B0: profiling counters in StereoMatch.cpp (work counts per thread, phase times, one line per match)."""
p = r"D:\Refraction\pyOpenLPT\src\srcSTB\StereoMatch.cpp"
raw = open(p, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:80])
    s = s.replace(old, new)


# includes + counting macro
rep('''#include "StereoMatch.h"
#include <array>
#include <iterator>
#include <omp.h>
''', '''#include "StereoMatch.h"
#include <array>
#include <chrono>
#include <iterator>
#include <omp.h>

// Profiling (setting "profile"): add n to this thread's counter `field`; a
// no-op unless profiling is on.
#define SM_PROFILE_ADD(field, n)                                               \\
  do {                                                                         \\
    if (_profile) {                                                            \\
      const size_t t_ = static_cast<size_t>(omp_get_thread_num());             \\
      if (t_ < _prof.size())                                                   \\
        _prof[t_].field += (n);                                                \\
    }                                                                          \\
  } while (0)
''')

# constructor: read the flag
rep('''    // build IDMap buckets
    _idmaps[c]->rebuild(_obj2d_list[c]);
  }
} // All reference variable should be initialized first''', '''    // build IDMap buckets
    _idmaps[c]->rebuild(_obj2d_list[c]);
  }
  _profile = _obj_cfg._sm_param.profile;
} // All reference variable should be initialized first''')

# match(): reset counters, time the phases, print
rep('''  std::vector<std::vector<MatchWithScore>> thread_buckets(T);

#pragma omp parallel num_threads(T)''', '''  std::vector<std::vector<MatchWithScore>> thread_buckets(T);

  using clk = std::chrono::steady_clock;
  const auto t_start = clk::now();
  if (_profile)
    _prof.assign(static_cast<size_t>(T), ProfileCounters{});

#pragma omp parallel num_threads(T)''')
rep('''  } // parallel region ends

  // Flatten buckets sequentially (single-thread).''', '''  } // parallel region ends
  const auto t_built = clk::now();

  // Flatten buckets sequentially (single-thread).''')
rep('''  // ---- Global pruning (disjoint usage of 2D points, pick best by
  // triangulation error, etc.). ----
  std::vector<std::vector<int>> selected =
      pruneMatch(match_candidates, e_checks);

  // ---- Final triangulation and Object3D construction (all active cams
  // participate). ----
  std::vector<std::unique_ptr<Object3D>> out = triangulateMatch(selected);

  return out;
}''', '''  const auto t_sorted = clk::now();

  // ---- Global pruning (disjoint usage of 2D points, pick best by
  // triangulation error, etc.). ----
  std::vector<std::vector<int>> selected =
      pruneMatch(match_candidates, e_checks);
  const auto t_pruned = clk::now();

  // ---- Final triangulation and Object3D construction (all active cams
  // participate). ----
  std::vector<std::unique_ptr<Object3D>> out = triangulateMatch(selected);

  if (_profile) {
    const auto t_end = clk::now();
    auto sec = [](clk::duration d) {
      return std::chrono::duration<double>(d).count();
    };
    ProfileCounters sum;
    for (const auto &c : _prof) {
      sum.los += c.los;
      sum.proj += c.proj;
      sum.tri += c.tri;
      sum.enumerations += c.enumerations;
      sum.strip_points += c.strip_points;
      sum.candidates += c.candidates;
      sum.dfs_steps += c.dfs_steps;
      sum.built += c.built;
    }
    std::cout << "\\t\\t[profile] refs " << ref_obs.size() << ", built "
              << sum.built << ", checked " << match_candidates.size()
              << ", selected " << selected.size() << ", objects " << out.size()
              << " | los " << sum.los
              << ", proj " << sum.proj << ", tri " << sum.tri << ", enum "
              << sum.enumerations << ", strip pts " << sum.strip_points
              << ", cands " << sum.candidates << ", dfs " << sum.dfs_steps
              << " | t build+check " << sec(t_built - t_start) << " s, sort "
              << sec(t_sorted - t_built) << " s, prune "
              << sec(t_pruned - t_sorted) << " s, tri "
              << sec(t_end - t_pruned) << " s" << std::endl;
  }

  return out;
}''')

# buildMatch root LOS
rep('''    const Pt2D &p = _obj2d_list[ref_cam][ref_id]->_pt_center;
    auto los_status = _cam_list[ref_cam]->lineOfSight(p);
    if (!los_status)
      return;''', '''    const Pt2D &p = _obj2d_list[ref_cam][ref_id]->_pt_center;
    auto los_status = _cam_list[ref_cam]->lineOfSight(p);
    SM_PROFILE_ADD(los, 1);
    if (!los_status)
      return;''')
# candidates built per reference observation
rep('''      buildMatch(match_candidate_id, build_candidates); // must be thread-safe
''', '''      buildMatch(match_candidate_id, build_candidates); // must be thread-safe
      SM_PROFILE_ADD(built, static_cast<long long>(build_candidates.size()));
''')
# DFS step + candidate LOS
rep('''    const int pid = fr.candidates[fr.cand_idx++]; // advance for the next loop
    const int cam = fr.target_cam;
''', '''    const int pid = fr.candidates[fr.cand_idx++]; // advance for the next loop
    const int cam = fr.target_cam;
    SM_PROFILE_ADD(dfs_steps, 1);
''')
rep('''      const Pt2D &p = _obj2d_list[cam][pid]->_pt_center;
      auto los_status = _cam_list[cam]->lineOfSight(p);
      if (!los_status)
        continue;
      nxt.los3d.push_back(los_status.value());''', '''      const Pt2D &p = _obj2d_list[cam][pid]->_pt_center;
      auto los_status = _cam_list[cam]->lineOfSight(p);
      SM_PROFILE_ADD(los, 1);
      if (!los_status)
        continue;
      nxt.los3d.push_back(los_status.value());''')

# enumerateCandidatesOnCam
rep('''  out_candidates.clear();
  if (target_cam < 0 || target_cam >= static_cast<int>(_idmaps.size()))
    return;
  IDMap *idm = _idmaps[target_cam].get();''', '''  out_candidates.clear();
  SM_PROFILE_ADD(enumerations, 1);
  if (target_cam < 0 || target_cam >= static_cast<int>(_idmaps.size()))
    return;
  IDMap *idm = _idmaps[target_cam].get();''')
rep('''  idm->visitPointsInRowSpans(spans, lines_px, _obj_cfg._sm_param.tol_2d_px,
                             out_candidates);
  if (out_candidates.empty())
    return;
''', '''  idm->visitPointsInRowSpans(spans, lines_px, _obj_cfg._sm_param.tol_2d_px,
                             out_candidates);
  SM_PROFILE_ADD(strip_points, static_cast<long long>(out_candidates.size()));
  if (out_candidates.empty())
    return;
''')
rep('''      out_candidates[write++] = pid;
    }
  }
  out_candidates.resize(write);
}''', '''      out_candidates[write++] = pid;
    }
  }
  out_candidates.resize(write);
  SM_PROFILE_ADD(candidates, write);
}''')

# checkMatch: gather LOS, full-cam triangulation, per-inlier LOS + triangulation
rep('''    const Pt2D &q = _obj2d_list[cam][pid]->_pt_center;
    auto los_status = _cam_list[cam]->lineOfSight(q);
    if (!los_status)
      return false;
    los3d.push_back(los_status.value());''', '''    const Pt2D &q = _obj2d_list[cam][pid]->_pt_center;
    auto los_status = _cam_list[cam]->lineOfSight(q);
    SM_PROFILE_ADD(los, 1);
    if (!los_status)
      return false;
    los3d.push_back(los_status.value());''')
rep('''    Pt3D Xw{};
    myMATH::triangulation(Xw, out_e_check, los3d);
    if (out_e_check > _obj_cfg._sm_param.tol_3d_mm)''', '''    Pt3D Xw{};
    myMATH::triangulation(Xw, out_e_check, los3d);
    SM_PROFILE_ADD(tri, 1);
    if (out_e_check > _obj_cfg._sm_param.tol_3d_mm)''')
rep('''      const Pt2D &q = _obj2d_list[cam][pid]->_pt_center;
      auto los_status = _cam_list[cam]->lineOfSight(q);
      if (!los_status)
        continue;
      Line3D los_q = los_status.value();''', '''      const Pt2D &q = _obj2d_list[cam][pid]->_pt_center;
      auto los_status = _cam_list[cam]->lineOfSight(q);
      SM_PROFILE_ADD(los, 1);
      if (!los_status)
        continue;
      Line3D los_q = los_status.value();''')
rep('''      Pt3D Xw{};
      double err = 0.0;
      myMATH::triangulation(Xw, err, los_all);
''', '''      Pt3D Xw{};
      double err = 0.0;
      myMATH::triangulation(Xw, err, los_all);
      SM_PROFILE_ADD(tri, 1);
''')

# triangulateMatch
rep('''      const Pt2D &q = obs2d[c][pid]->_pt_center;
      auto los_status = _cam_list[c]->lineOfSight(q);
      if (!los_status) {''', '''      const Pt2D &q = obs2d[c][pid]->_pt_center;
      auto los_status = _cam_list[c]->lineOfSight(q);
      SM_PROFILE_ADD(los, 1);
      if (!los_status) {''')
rep('''    try {
      myMATH::triangulation(pt_world, err, los);
    } catch (...) {
      continue;
    }''', '''    try {
      myMATH::triangulation(pt_world, err, los);
      SM_PROFILE_ADD(tri, 1);
    } catch (...) {
      continue;
    }''')

# makeLine2DFromLOS3D: projections
rep('''  auto a_status = _cam_list[cam_id]->project(p_a);
  if (!a_status)
    return false;
  auto b_status = _cam_list[cam_id]->project(p_b);
  if (!b_status)
    return false;''', '''  auto a_status = _cam_list[cam_id]->project(p_a);
  SM_PROFILE_ADD(proj, 1);
  if (!a_status)
    return false;
  auto b_status = _cam_list[cam_id]->project(p_b);
  SM_PROFILE_ADD(proj, 1);
  if (!b_status)
    return false;''')

# TriangulationCheckWithTol
rep('''  try {
    myMATH::triangulation(pt, err, los3d);
  } catch (...) {
    return false;
  }
  return err <= tol_3d_mm;''', '''  try {
    myMATH::triangulation(pt, err, los3d);
    SM_PROFILE_ADD(tri, 1);
  } catch (...) {
    return false;
  }
  return err <= tol_3d_mm;''')

# checkBackProjection
rep('''  auto los_t_status = _cam_list[target_cam]->lineOfSight(q_t);
  if (!los_t_status)
    return false;''', '''  auto los_t_status = _cam_list[target_cam]->lineOfSight(q_t);
  SM_PROFILE_ADD(los, 1);
  if (!los_t_status)
    return false;''')

# bubbleEarlyCheck
rep('''    const Pt2D &q = base->_pt_center; // same domain as calibration
    auto los_status = _cam_list[cam]->lineOfSight(q);
    if (!los_status)
      return true;''', '''    const Pt2D &q = base->_pt_center; // same domain as calibration
    auto los_status = _cam_list[cam]->lineOfSight(q);
    SM_PROFILE_ADD(los, 1);
    if (!los_status)
      return true;''')

out = s.replace("\n", "\r\n") if crlf else s
open(p, "wb").write(out.encode("utf-8"))
print("patched, crlf =", crlf)
