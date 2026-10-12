"""B1: per-match tables of lines of sight and image lines in StereoMatch.cpp (identical values, computed once)."""
p = r"D:\Refraction\pyOpenLPT\src\srcSTB\StereoMatch.cpp"
raw = open(p, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


# ---- match(): build the tables first ----
rep('''  using clk = std::chrono::steady_clock;
  const auto t_start = clk::now();
  if (_profile)
    _prof.assign(static_cast<size_t>(T), ProfileCounters{});
''', '''  using clk = std::chrono::steady_clock;
  const auto t_start = clk::now();
  if (_profile)
    _prof.assign(static_cast<size_t>(T), ProfileCounters{});

  // lines of sight and image lines of all observations, once
  buildTables();
  const auto t_tables = clk::now();
''')
rep('''              << " | t build+check " << sec(t_built - t_start) << " s, sort "''',
    '''              << " | t tables " << sec(t_tables - t_start)
              << " s, build+check " << sec(t_built - t_tables) << " s, sort "''')

# ---- buildTables() definition, before buildMatch ----
rep('''// Build across the first m active cameras (iterative DFS).''', '''// The tables of lines of sight and image lines (see StereoMatch.h): one work
// item per observation of an active camera, in parallel.
void StereoMatch::buildTables() const {
  const int n_cams = static_cast<int>(_cam_list.size());
  auto active = [&](int c) { return _cam_list[c] && _cam_list[c]->is_active; };

  _los_tab.assign(static_cast<size_t>(n_cams), {});
  _line_tab.assign(static_cast<size_t>(n_cams),
                   std::vector<std::vector<LineEntry>>(static_cast<size_t>(n_cams)));
  std::vector<std::pair<int, int>> items; // (cam, pid)
  for (int c = 0; c < n_cams; ++c) {
    if (!active(c))
      continue;
    const size_t n = _obj2d_list[c].size();
    _los_tab[c].resize(n);
    for (int t = 0; t < n_cams; ++t)
      if (t != c && active(t))
        _line_tab[c][t].resize(n);
    for (size_t pid = 0; pid < n; ++pid)
      items.emplace_back(c, static_cast<int>(pid));
  }

#pragma omp parallel for schedule(dynamic, 64)
  for (int k = 0; k < static_cast<int>(items.size()); ++k) {
    const int c = items[k].first, pid = items[k].second;
    const Object2D *o = _obj2d_list[c][pid];
    if (!o)
      continue;
    auto los_status = _cam_list[c]->lineOfSight(o->_pt_center);
    SM_PROFILE_ADD(los, 1);
    if (!los_status)
      continue;
    LosEntry &e = _los_tab[c][pid];
    e.los = los_status.value();
    e.ok = true;
    for (int t = 0; t < n_cams; ++t) {
      if (t == c || !active(t))
        continue;
      LineEntry &le = _line_tab[c][t][pid];
      le.ok = makeLine2DFromLOS3D(t, e.los, le.line);
    }
  }
}

// Build across the first m active cameras (iterative DFS).''')

# ---- buildMatch: root LOS from the table, second camera by ids ----
rep('''    const Pt2D &p = _obj2d_list[ref_cam][ref_id]->_pt_center;
    auto los_status = _cam_list[ref_cam]->lineOfSight(p);
    SM_PROFILE_ADD(los, 1);
    if (!los_status)
      return;
    root.los3d.push_back(los_status.value());
    root.chosen_pts = {p};''', '''    const Pt2D &p = _obj2d_list[ref_cam][ref_id]->_pt_center;
    const LosEntry &le = _los_tab[ref_cam][ref_id];
    if (!le.ok)
      return;
    root.los3d.push_back(le.los);
    root.chosen_pts = {p};''')
rep('''    root.target_cam =
        selectSecondCameraByLineLength(root.los3d.front(), root.rem_cams);
    if (root.target_cam >= 0) {
      enumerateCandidatesOnCam(root.los3d, root.target_cam, root.chosen_cams,
                               root.chosen_pts, root.candidates);''', '''    root.target_cam =
        selectSecondCameraByLineLength(ref_cam, ref_id, root.rem_cams);
    if (root.target_cam >= 0) {
      enumerateCandidatesOnCam(root.target_cam, root.chosen_cams,
                               root.chosen_ids, root.chosen_pts,
                               root.candidates);''')
# DFS step: candidate LOS from the table
rep('''      const Pt2D &p = _obj2d_list[cam][pid]->_pt_center;
      auto los_status = _cam_list[cam]->lineOfSight(p);
      SM_PROFILE_ADD(los, 1);
      if (!los_status)
        continue;
      nxt.los3d.push_back(los_status.value());
      nxt.chosen_pts.push_back(p);''', '''      const Pt2D &p = _obj2d_list[cam][pid]->_pt_center;
      const LosEntry &le = _los_tab[cam][pid];
      if (!le.ok)
        continue;
      nxt.los3d.push_back(le.los);
      nxt.chosen_pts.push_back(p);''')
rep('''    nxt.target_cam = selectNextCameraByMaxPairAngle(nxt.los3d, nxt.rem_cams);''',
    '''    nxt.target_cam = selectNextCameraByMaxPairAngle(
        nxt.chosen_cams, nxt.chosen_ids, nxt.rem_cams);''')
rep('''    enumerateCandidatesOnCam(nxt.los3d, nxt.target_cam, nxt.chosen_cams,
                             nxt.chosen_pts, nxt.candidates);''', '''    enumerateCandidatesOnCam(nxt.target_cam, nxt.chosen_cams, nxt.chosen_ids,
                             nxt.chosen_pts, nxt.candidates);''')

# ---- enumerateCandidatesOnCam: lines from the table ----
rep('''void StereoMatch::enumerateCandidatesOnCam(
    const std::vector<Line3D> &los3d, int target_cam,
    const std::vector<int> &chosen_cams, const std::vector<Pt2D> &chosen_pts,
    std::vector<int> &out_candidates) const {
  out_candidates.clear();
  SM_PROFILE_ADD(enumerations, 1);
  if (target_cam < 0 || target_cam >= static_cast<int>(_idmaps.size()))
    return;
  IDMap *idm = _idmaps[target_cam].get();
  if (!idm || los3d.empty())
    return;

  // 1) Project LOS -> 2D image lines on this camera
  std::vector<Line2D> lines_px;
  if (!buildLinesOnCam(los3d, target_cam, lines_px) || lines_px.empty())
    return;
''', '''void StereoMatch::enumerateCandidatesOnCam(
    int target_cam, const std::vector<int> &chosen_cams,
    const std::vector<int> &chosen_ids, const std::vector<Pt2D> &chosen_pts,
    std::vector<int> &out_candidates) const {
  out_candidates.clear();
  SM_PROFILE_ADD(enumerations, 1);
  if (target_cam < 0 || target_cam >= static_cast<int>(_idmaps.size()))
    return;
  IDMap *idm = _idmaps[target_cam].get();
  if (!idm || chosen_cams.empty())
    return;

  // 1) The chosen observations' image lines on this camera (from the table)
  std::vector<Line2D> lines_px;
  lines_px.reserve(chosen_cams.size());
  for (size_t i = 0; i < chosen_cams.size(); ++i) {
    const LineEntry &le = _line_tab[chosen_cams[i]][target_cam][chosen_ids[i]];
    if (!le.ok)
      return;
    lines_px.push_back(le.line);
  }
''')
rep('''    const Pt2D &q_t = o->_pt_center;
    if (checkBackProjection(target_cam, q_t, chosen_cams, chosen_pts)) {
      out_candidates[write++] = pid;
    }''', '''    if (checkBackProjection(target_cam, pid, chosen_cams, chosen_pts)) {
      out_candidates[write++] = pid;
    }''')

# ---- checkMatch ----
rep('''    const Pt2D &q = _obj2d_list[cam][pid]->_pt_center;
    auto los_status = _cam_list[cam]->lineOfSight(q);
    SM_PROFILE_ADD(los, 1);
    if (!los_status)
      return false;
    los3d.push_back(los_status.value());''', '''    const LosEntry &le = _los_tab[cam][pid];
    if (!le.ok)
      return false;
    los3d.push_back(le.los);''')
rep('''  // Prepare chosen cams & points for back-projection checks
  std::vector<int> chosen_cams;
  std::vector<Pt2D> chosen_pts;
  chosen_cams.reserve(los3d.size());
  chosen_pts.reserve(los3d.size());
  for (int c = 0; c < n_cams; ++c) {
    const int pid = candidate_ids[c];
    if (pid >= 0) {
      chosen_cams.push_back(c);
      chosen_pts.push_back(_obj2d_list[c][pid]->_pt_center);
    }
  }''', '''  // Prepare chosen cams, ids & points for back-projection checks
  std::vector<int> chosen_cams;
  std::vector<int> chosen_ids;
  std::vector<Pt2D> chosen_pts;
  chosen_cams.reserve(los3d.size());
  chosen_ids.reserve(los3d.size());
  chosen_pts.reserve(los3d.size());
  for (int c = 0; c < n_cams; ++c) {
    const int pid = candidate_ids[c];
    if (pid >= 0) {
      chosen_cams.push_back(c);
      chosen_ids.push_back(pid);
      chosen_pts.push_back(_obj2d_list[c][pid]->_pt_center);
    }
  }''')
rep('''    enumerateCandidatesOnCam(los3d, /*target_cam=*/cam, chosen_cams, chosen_pts,
                             inliers);''', '''    enumerateCandidatesOnCam(/*target_cam=*/cam, chosen_cams, chosen_ids,
                             chosen_pts, inliers);''')
rep('''      const Pt2D &q = _obj2d_list[cam][pid]->_pt_center;
      auto los_status = _cam_list[cam]->lineOfSight(q);
      SM_PROFILE_ADD(los, 1);
      if (!los_status)
        continue;
      Line3D los_q = los_status.value();''', '''      const LosEntry &lq = _los_tab[cam][pid];
      if (!lq.ok)
        continue;
      const Line3D &los_q = lq.los;''')

# ---- triangulateMatch ----
rep('''      const Pt2D &q = obs2d[c][pid]->_pt_center;
      auto los_status = _cam_list[c]->lineOfSight(q);
      SM_PROFILE_ADD(los, 1);
      if (!los_status) {
        los.clear();
        break;
      }
      los.push_back(los_status.value());''', '''      const LosEntry &le = _los_tab[c][pid];
      if (!le.ok) {
        los.clear();
        break;
      }
      los.push_back(le.los);''')

# ---- remove buildLinesOnCam (replaced by the table) ----
rep('''// ---- helper: project a set of LOS to 2D lines on a camera (re-uses output
// buffer) ----
inline bool StereoMatch::buildLinesOnCam(const std::vector<Line3D> &los3d,
                                         int cam_id,
                                         std::vector<Line2D> &out_lines) const {
  out_lines.clear();
  out_lines.reserve(los3d.size());
  for (const auto &L3 : los3d) {
    Line2D line;
    if (!makeLine2DFromLOS3D(cam_id, L3, line))
      return false;
    out_lines.push_back(line);
  }
  return true;
}

''', '')
# makeLine2DFromLOS3D is now called from buildTables (another function): not inline-only
rep('''inline bool StereoMatch::makeLine2DFromLOS3D(int cam_id, const Line3D &los,''',
    '''bool StereoMatch::makeLine2DFromLOS3D(int cam_id, const Line3D &los,''')

# ---- checkBackProjection ----
rep('''bool StereoMatch::checkBackProjection(
    int target_cam, const Pt2D &q_t, const std::vector<int> &chosen_cams,
    const std::vector<Pt2D> &chosen_pts) const {
  if (chosen_cams.empty())
    return true;

  auto los_t_status = _cam_list[target_cam]->lineOfSight(q_t);
  SM_PROFILE_ADD(los, 1);
  if (!los_t_status)
    return false;
  const Line3D los_t = los_t_status.value();

  const double tol2 =
      _obj_cfg._sm_param.tol_2d_px * _obj_cfg._sm_param.tol_2d_px;
  const size_t K = chosen_cams.size();

  for (size_t i = 0; i < K; ++i) {
    const int ci = chosen_cams[i];
    const Pt2D &qi = chosen_pts[i];

    Line2D L;
    if (!makeLine2DFromLOS3D(ci, los_t, L))
      return false;
    if (myMATH::dist2(qi, L) > tol2)
      return false;
  }
  return true;
}''', '''bool StereoMatch::checkBackProjection(
    int target_cam, int pid, const std::vector<int> &chosen_cams,
    const std::vector<Pt2D> &chosen_pts) const {
  if (chosen_cams.empty())
    return true;

  if (!_los_tab[target_cam][pid].ok)
    return false;

  const double tol2 =
      _obj_cfg._sm_param.tol_2d_px * _obj_cfg._sm_param.tol_2d_px;
  const size_t K = chosen_cams.size();

  for (size_t i = 0; i < K; ++i) {
    const int ci = chosen_cams[i];
    const Pt2D &qi = chosen_pts[i];

    const LineEntry &le = _line_tab[target_cam][ci][pid];
    if (!le.ok)
      return false;
    if (myMATH::dist2(qi, le.line) > tol2)
      return false;
  }
  return true;
}''')

# ---- bubbleEarlyCheck ----
rep('''    const Pt2D &q = base->_pt_center; // same domain as calibration
    auto los_status = _cam_list[cam]->lineOfSight(q);
    SM_PROFILE_ADD(los, 1);
    if (!los_status)
      return true;
    los.push_back(los_status.value());''', '''    const LosEntry &le = _los_tab[cam][pid];
    if (!le.ok)
      return true;
    los.push_back(le.los);''')

# ---- selectSecondCameraByLineLength ----
rep('''int StereoMatch::selectSecondCameraByLineLength(
    const Line3D &los_ref, const std::vector<int> &remaining_cams) const {
  if (remaining_cams.empty())
    return -1;

  auto segLenOnCam = [&](int cam) -> double {
    Line2D L;
    if (!makeLine2DFromLOS3D(cam, los_ref, L))
      return std::numeric_limits<double>::infinity();''', '''int StereoMatch::selectSecondCameraByLineLength(
    int ref_cam, int ref_id, const std::vector<int> &remaining_cams) const {
  if (remaining_cams.empty())
    return -1;

  auto segLenOnCam = [&](int cam) -> double {
    const LineEntry &le = _line_tab[ref_cam][cam][ref_id];
    if (!le.ok)
      return std::numeric_limits<double>::infinity();
    const Line2D &L = le.line;''')

# ---- selectNextCameraByMaxPairAngle ----
rep('''int StereoMatch::selectNextCameraByMaxPairAngle(
    const std::vector<Line3D> &los3d,
    const std::vector<int> &remaining_cams) const {
  if (remaining_cams.empty())
    return -1;

  int best_cam = -1;
  double best_score = -1.0;

  std::vector<Line2D> lines_px; // reused buffer

  for (int cam : remaining_cams) {
    if (!buildLinesOnCam(los3d, cam, lines_px))
      continue;
''', '''int StereoMatch::selectNextCameraByMaxPairAngle(
    const std::vector<int> &chosen_cams, const std::vector<int> &chosen_ids,
    const std::vector<int> &remaining_cams) const {
  if (remaining_cams.empty())
    return -1;

  int best_cam = -1;
  double best_score = -1.0;

  std::vector<Line2D> lines_px; // reused buffer

  for (int cam : remaining_cams) {
    // the chosen observations' image lines on this camera (from the table)
    lines_px.clear();
    bool all_ok = true;
    for (size_t i = 0; i < chosen_cams.size(); ++i) {
      const LineEntry &le = _line_tab[chosen_cams[i]][cam][chosen_ids[i]];
      if (!le.ok) {
        all_ok = false;
        break;
      }
      lines_px.push_back(le.line);
    }
    if (!all_ok)
      continue;
''')

out = s.replace("\n", "\r\n") if crlf else s
open(p, "wb").write(out.encode("utf-8"))
print("patched, crlf =", crlf)
