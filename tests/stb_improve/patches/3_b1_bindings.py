"""B1: the StereoMatch debug bindings follow the new private signatures; table readers build the tables first."""
p = r"D:\Refraction\pyOpenLPT\src\pybind_OpenLPT\pyStereoMatch.cpp"
raw = open(p, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:90])
    s = s.replace(old, new)


rep('''struct DebugAccess_StereoMatch {
  // main pipeline
  static void buildMatch(const StereoMatch &m,
                         std::vector<int> &match_candidate_id,
                         std::vector<std::vector<int>> &build_candidates) {
    m.buildMatch(match_candidate_id, build_candidates);
  }
  static bool checkMatch(const StereoMatch &m,
                         const std::vector<int> &candidate_ids,
                         double &out_e_check) {
    return m.checkMatch(candidate_ids, out_e_check);
  }''', '''struct DebugAccess_StereoMatch {
  // The private steps read the tables of lines of sight and image lines that
  // match() builds; build them here when match() has not run yet.
  static void ensureTables(const StereoMatch &m) {
    if (m._los_tab.empty())
      m.buildTables();
  }

  // main pipeline
  static void buildMatch(const StereoMatch &m,
                         std::vector<int> &match_candidate_id,
                         std::vector<std::vector<int>> &build_candidates) {
    ensureTables(m);
    m.buildMatch(match_candidate_id, build_candidates);
  }
  static bool checkMatch(const StereoMatch &m,
                         const std::vector<int> &candidate_ids,
                         double &out_e_check) {
    ensureTables(m);
    return m.checkMatch(candidate_ids, out_e_check);
  }''')
rep('''  triangulateMatch(const StereoMatch &m,
                   const std::vector<std::vector<int>> &selected_matches) {
    return m.triangulateMatch(selected_matches);
  }''', '''  triangulateMatch(const StereoMatch &m,
                   const std::vector<std::vector<int>> &selected_matches) {
    ensureTables(m);
    return m.triangulateMatch(selected_matches);
  }''')
rep('''  static void enumerateCandidatesOnCam(const StereoMatch &m,
                                       const std::vector<Line3D> &los3d,
                                       int target_cam,
                                       const std::vector<int> &chosen_cams,
                                       const std::vector<Pt2D> &chosen_pts,
                                       std::vector<int> &out) {
    m.enumerateCandidatesOnCam(los3d, target_cam, chosen_cams, chosen_pts, out);
  }''', '''  static void enumerateCandidatesOnCam(const StereoMatch &m, int target_cam,
                                       const std::vector<int> &chosen_cams,
                                       const std::vector<int> &chosen_ids,
                                       const std::vector<Pt2D> &chosen_pts,
                                       std::vector<int> &out) {
    ensureTables(m);
    m.enumerateCandidatesOnCam(target_cam, chosen_cams, chosen_ids, chosen_pts,
                               out);
  }''')
rep('''  static bool buildLinesOnCam(const StereoMatch &m,
                              const std::vector<Line3D> &los3d, int cam_id,
                              std::vector<Line2D> &out) {
    return m.buildLinesOnCam(los3d, cam_id, out);
  }
  static int
  selectSecondCameraByLineLength(const StereoMatch &m, const Line3D &los_ref,
                                 const std::vector<int> &remaining_cams) {
    return m.selectSecondCameraByLineLength(los_ref, remaining_cams);
  }
  static int
  selectNextCameraByMaxPairAngle(const StereoMatch &m,
                                 const std::vector<Line3D> &los3d,
                                 const std::vector<int> &remaining_cams) {
    return m.selectNextCameraByMaxPairAngle(los3d, remaining_cams);
  }''', '''  // the image lines of a set of lines of sight on one camera (the search
  // itself reads them from the tables)
  static bool buildLinesOnCam(const StereoMatch &m,
                              const std::vector<Line3D> &los3d, int cam_id,
                              std::vector<Line2D> &out) {
    out.clear();
    for (const auto &L3 : los3d) {
      Line2D line;
      if (!m.makeLine2DFromLOS3D(cam_id, L3, line))
        return false;
      out.push_back(line);
    }
    return true;
  }
  static int
  selectSecondCameraByLineLength(const StereoMatch &m, int ref_cam, int ref_id,
                                 const std::vector<int> &remaining_cams) {
    ensureTables(m);
    return m.selectSecondCameraByLineLength(ref_cam, ref_id, remaining_cams);
  }
  static int
  selectNextCameraByMaxPairAngle(const StereoMatch &m,
                                 const std::vector<int> &chosen_cams,
                                 const std::vector<int> &chosen_ids,
                                 const std::vector<int> &remaining_cams) {
    ensureTables(m);
    return m.selectNextCameraByMaxPairAngle(chosen_cams, chosen_ids,
                                            remaining_cams);
  }''')
rep('''  static bool checkBackProjection(const StereoMatch &m, int target_cam,
                                  const Pt2D &q_t,
                                  const std::vector<int> &chosen_cams,
                                  const std::vector<Pt2D> &chosen_pts) {
    return m.checkBackProjection(target_cam, q_t, chosen_cams, chosen_pts);
  }''', '''  static bool checkBackProjection(const StereoMatch &m, int target_cam,
                                  int pid, const std::vector<int> &chosen_cams,
                                  const std::vector<Pt2D> &chosen_pts) {
    ensureTables(m);
    return m.checkBackProjection(target_cam, pid, chosen_cams, chosen_pts);
  }''')
rep('''  static bool objectEarlyCheck(const StereoMatch &m,
                               const std::vector<int> &cams_used,
                               const std::vector<int> &ids_on_used) {
    return m.objectEarlyCheck(cams_used, ids_on_used);
  }''', '''  static bool objectEarlyCheck(const StereoMatch &m,
                               const std::vector<int> &cams_used,
                               const std::vector<int> &ids_on_used) {
    ensureTables(m);
    return m.objectEarlyCheck(cams_used, ids_on_used);
  }''')
rep('''  static bool bubbleEarlyCheck(const StereoMatch &m,
                               const std::vector<int> &cams_in_path,
                               const std::vector<int> &ids_in_path) {
    return m.bubbleEarlyCheck(cams_in_path, ids_in_path);
  }''', '''  static bool bubbleEarlyCheck(const StereoMatch &m,
                               const std::vector<int> &cams_in_path,
                               const std::vector<int> &ids_in_path) {
    ensureTables(m);
    return m.bubbleEarlyCheck(cams_in_path, ids_in_path);
  }''')

# Python-facing defs
rep('''      .def(
          "enumerateCandidatesOnCam",
          [](const StereoMatch &self, const std::vector<Line3D> &los3d,
             int target_cam, const std::vector<int> &chosen_cams,
             const std::vector<Pt2D> &chosen_pts) {
            std::vector<int> out;
            DebugAccess_StereoMatch::enumerateCandidatesOnCam(
                self, los3d, target_cam, chosen_cams, chosen_pts, out);
            return out;
          },
          py::arg("los3d"), py::arg("target_cam"),
          py::arg("chosen_cams") = std::vector<int>{},
          py::arg("chosen_pts") = std::vector<Pt2D>{})''', '''      .def(
          "enumerateCandidatesOnCam",
          [](const StereoMatch &self, int target_cam,
             const std::vector<int> &chosen_cams,
             const std::vector<int> &chosen_ids,
             const std::vector<Pt2D> &chosen_pts) {
            std::vector<int> out;
            DebugAccess_StereoMatch::enumerateCandidatesOnCam(
                self, target_cam, chosen_cams, chosen_ids, chosen_pts, out);
            return out;
          },
          py::arg("target_cam"), py::arg("chosen_cams"), py::arg("chosen_ids"),
          py::arg("chosen_pts"))''')
rep('''      .def(
          "selectSecondCameraByLineLength",
          [](const StereoMatch &self, const Line3D &los_ref,
             const std::vector<int> &remaining_cams) {
            return DebugAccess_StereoMatch::selectSecondCameraByLineLength(
                self, los_ref, remaining_cams);
          },
          py::arg("los_ref"), py::arg("remaining_cams"))
      .def(
          "selectNextCameraByMaxPairAngle",
          [](const StereoMatch &self, const std::vector<Line3D> &los3d,
             const std::vector<int> &remaining_cams) {
            return DebugAccess_StereoMatch::selectNextCameraByMaxPairAngle(
                self, los3d, remaining_cams);
          },
          py::arg("los3d"), py::arg("remaining_cams"))''', '''      .def(
          "selectSecondCameraByLineLength",
          [](const StereoMatch &self, int ref_cam, int ref_id,
             const std::vector<int> &remaining_cams) {
            return DebugAccess_StereoMatch::selectSecondCameraByLineLength(
                self, ref_cam, ref_id, remaining_cams);
          },
          py::arg("ref_cam"), py::arg("ref_id"), py::arg("remaining_cams"))
      .def(
          "selectNextCameraByMaxPairAngle",
          [](const StereoMatch &self, const std::vector<int> &chosen_cams,
             const std::vector<int> &chosen_ids,
             const std::vector<int> &remaining_cams) {
            return DebugAccess_StereoMatch::selectNextCameraByMaxPairAngle(
                self, chosen_cams, chosen_ids, remaining_cams);
          },
          py::arg("chosen_cams"), py::arg("chosen_ids"),
          py::arg("remaining_cams"))''')
rep('''      .def(
          "checkBackProjection",
          [](const StereoMatch &self, int target_cam, const Pt2D &q_t,
             const std::vector<int> &chosen_cams,
             const std::vector<Pt2D> &chosen_pts) {
            return DebugAccess_StereoMatch::checkBackProjection(
                self, target_cam, q_t, chosen_cams, chosen_pts);
          },
          py::arg("target_cam"), py::arg("q_t"), py::arg("chosen_cams"),
          py::arg("chosen_pts"))''', '''      .def(
          "checkBackProjection",
          [](const StereoMatch &self, int target_cam, int pid,
             const std::vector<int> &chosen_cams,
             const std::vector<Pt2D> &chosen_pts) {
            return DebugAccess_StereoMatch::checkBackProjection(
                self, target_cam, pid, chosen_cams, chosen_pts);
          },
          py::arg("target_cam"), py::arg("pid"), py::arg("chosen_cams"),
          py::arg("chosen_pts"))''')

out = s.replace("\n", "\r\n") if crlf else s
open(p, "wb").write(out.encode("utf-8"))
print("patched bindings, crlf =", crlf)
