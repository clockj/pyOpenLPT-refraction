#include "Config.h"
#include "ImageIO.h" // ensure correct case

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <system_error>
#include <tuple>

#include "nlohmann/json.hpp"

namespace {
namespace fs = std::filesystem;

fs::path absolute_lexical(const fs::path &path) {
  std::error_code ec;
  const fs::path abs_path = fs::absolute(path, ec);
  if (ec) {
    return path.lexically_normal();
  }
  return abs_path.lexically_normal();
}

std::string resolve_path_from_config_root(const fs::path &config_root,
                                          const std::string &raw_path) {
  if (raw_path.empty()) {
    return raw_path;
  }
  const fs::path input(raw_path);
  const fs::path resolved =
      input.is_absolute() ? input : (config_root / input);
  return resolved.lexically_normal().generic_string();
}

// An optional setting line "key = value"; the key is a name (letters, digits,
// '_', not starting with a digit). Any other line is not a setting.
bool parse_key_value(const std::string &line, std::string &key,
                     std::string &value) {
  const size_t eq = line.find('=');
  if (eq == std::string::npos)
    return false;
  auto trim = [](std::string s) {
    s.erase(0, s.find_first_not_of(" \t"));
    s.erase(s.find_last_not_of(" \t") + 1);
    return s;
  };
  key = trim(line.substr(0, eq));
  value = trim(line.substr(eq + 1));
  if (key.empty() || value.empty())
    return false;
  if (!std::isalpha(static_cast<unsigned char>(key[0])) && key[0] != '_')
    return false;
  for (char ch : key)
    if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_')
      return false;
  return true;
}

int parse_positive_int(const std::string &key, const std::string &value,
                       const std::string &config_path) {
  size_t pos = 0;
  int v = 0;
  try {
    v = std::stoi(value, &pos);
  } catch (...) {
    pos = 0;
  }
  if (pos != value.size() || v <= 0) {
    THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                    "Config setting " + key +
                        " must be a positive integer, got: " + value,
                    config_path + "\n");
  }
  return v;
}

// ---------------------------------------------------------------------------
// JSON configuration (config.json): one file with the basic settings and the
// object settings that config.txt and tracerConfig.txt / bubbleConfig.txt hold
// in the text format. Both formats fill the same structs.
// ---------------------------------------------------------------------------
using json = nlohmann::json;

bool has_json_extension(const std::string &path) {
  if (path.size() < 5)
    return false;
  std::string ext = path.substr(path.size() - 5);
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return ext == ".json";
}

// An object's settings inside config.json are addressed as "config.json#i"
// (the i-th entry of "objects"); split such a path into file and index.
bool split_json_object_path(const std::string &path, std::string &file,
                            int &index) {
  const size_t hash = path.rfind('#');
  if (hash == std::string::npos || hash + 1 >= path.size())
    return false;
  const std::string tail = path.substr(hash + 1);
  if (!std::all_of(tail.begin(), tail.end(),
                   [](unsigned char ch) { return std::isdigit(ch); }))
    return false;
  file = path.substr(0, hash);
  if (!has_json_extension(file))
    return false;
  index = std::stoi(tail);
  return true;
}

json load_json(const std::string &file) {
  std::ifstream in(file);
  REQUIRE_CTX(in.is_open(), ErrorCode::IOfailure,
              "Config Error: Cannot open file ", file + "\n");
  try {
    // comments (// and /* */) are allowed
    return json::parse(in, nullptr, true, true);
  } catch (const json::parse_error &e) {
    THROW_FATAL_CTX(ErrorCode::IOfailure,
                    std::string("Config Error: invalid JSON: ") + e.what(),
                    file + "\n");
  }
}

// Typed access to one JSON object, with errors that name the file and the key.
class JsonReader {
public:
  JsonReader(const json &j, std::string where, std::string file)
      : _j(j), _where(std::move(where)), _file(std::move(file)) {
    if (!_j.is_object())
      fail(_where.empty() ? "the file" : _where, "must be an object {...}");
  }

  bool has(const std::string &key) const { return _j.contains(key); }

  const json &req(const std::string &key) const {
    if (!_j.contains(key))
      fail(path(key), "is missing");
    return _j.at(key);
  }

  double num(const std::string &key) const {
    const json &v = req(key);
    if (!v.is_number())
      fail(path(key), "must be a number");
    return v.get<double>();
  }

  int integer(const std::string &key) const { return asInt(req(key), path(key)); }

  bool boolean(const std::string &key) const {
    const json &v = req(key);
    if (!v.is_boolean())
      fail(path(key), "must be true or false");
    return v.get<bool>();
  }

  std::string str(const std::string &key) const {
    const json &v = req(key);
    if (!v.is_string())
      fail(path(key), "must be a string");
    return v.get<std::string>();
  }

  JsonReader child(const std::string &key) const {
    return JsonReader(req(key), path(key), _file);
  }

  // an array of n numbers (n < 0: any length)
  std::vector<double> numbers(const std::string &key, int n) const {
    const json &v = req(key);
    if (!v.is_array() || (n >= 0 && static_cast<int>(v.size()) != n))
      fail(path(key), n >= 0 ? "must be an array of " + std::to_string(n) +
                                   " numbers"
                             : std::string("must be an array of numbers"));
    std::vector<double> out;
    for (size_t i = 0; i < v.size(); ++i) {
      if (!v[i].is_number())
        fail(path(key) + "[" + std::to_string(i) + "]", "must be a number");
      out.push_back(v[i].get<double>());
    }
    return out;
  }

  std::vector<int> integers(const std::string &key, int n) const {
    const json &v = req(key);
    if (!v.is_array() || static_cast<int>(v.size()) != n)
      fail(path(key), "must be an array of " + std::to_string(n) + " integers");
    std::vector<int> out;
    for (size_t i = 0; i < v.size(); ++i)
      out.push_back(asInt(v[i], path(key) + "[" + std::to_string(i) + "]"));
    return out;
  }

  // a warning for every key not in `allowed` (keys starting with '_' are
  // comments and always allowed)
  void warnUnknown(std::initializer_list<const char *> allowed) const {
    for (auto it = _j.begin(); it != _j.end(); ++it) {
      const std::string &k = it.key();
      if (!k.empty() && k[0] == '_')
        continue;
      bool ok = false;
      for (const char *a : allowed)
        ok = ok || k == a;
      if (!ok)
        std::cerr << "Warning: " << _file << ": unknown setting \"" << path(k)
                  << "\" is ignored" << std::endl;
    }
  }

  std::string path(const std::string &key) const {
    return _where.empty() ? key : _where + "." + key;
  }

  [[noreturn]] void fail(const std::string &what,
                         const std::string &problem) const {
    THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                    "Config Error: \"" + what + "\" " + problem, _file + "\n");
  }

private:
  int asInt(const json &v, const std::string &what) const {
    if (v.is_number_integer())
      return v.get<int>();
    if (v.is_number_float()) { // 4.0 is accepted as 4
      const double d = v.get<double>();
      if (d == std::floor(d) && std::fabs(d) < 2e9)
        return static_cast<int>(d);
    }
    fail(what, "must be an integer");
  }

  const json &_j;
  std::string _where;
  std::string _file;
};

// the basic settings from config.json
bool read_basic_json(BasicSetting &s, const fs::path &config_abs,
                     const fs::path &config_root) {
  const std::string file = config_abs.generic_string();
  const json doc = load_json(file);
  JsonReader r(doc, "", file);
  r.warnUnknown({"frames", "fps", "threads", "cameras", "image_lists",
                 "view_volume_mm", "voxel_mm", "output_dir", "objects",
                 "load_tracks", "save_interval", "profile"});

  const std::vector<int> frames = r.integers("frames", 2);
  s._frame_start = frames[0];
  s._frame_end = frames[1];
  s._fps = r.integer("fps");
  s._n_thread = r.has("threads") ? r.integer("threads") : 0;

  const json &cams = r.req("cameras");
  if (!cams.is_array() || cams.empty())
    r.fail("cameras", "must be a non-empty array");
  s._n_cam = static_cast<int>(cams.size());
  s._cam_list.clear();
  s._cam_list.reserve(cams.size());
  for (size_t i = 0; i < cams.size(); ++i) {
    JsonReader c(cams[i], "cameras[" + std::to_string(i) + "]", file);
    c.warnUnknown({"file", "max_intensity", "correction"});
    const std::string cam_path =
        resolve_path_from_config_root(config_root, c.str("file"));
    const int max_intensity =
        c.has("max_intensity") ? c.integer("max_intensity") : 255;
    auto cam_model_st = CameraFactory::loadFromFile(cam_path);
    if (!cam_model_st) {
      THROW_FATAL_CTX(ErrorCode::InvalidCameraState,
                      "Failed to load camera model from:", cam_path);
    }
    auto cam_model = cam_model_st.value();
    // Optional: the camera's image correction (VSC's field, fitted between
    // runs). Without it the camera is used as loaded.
    if (c.has("correction") && !c.str("correction").empty()) {
      const std::string corr_path =
          resolve_path_from_config_root(config_root, c.str("correction"));
      auto corr_st = ImageCorrection::loadFromFile(corr_path);
      if (!corr_st) {
        THROW_FATAL_CTX(ErrorCode::InvalidCameraState,
                        "Failed to load the image correction of cameras[" +
                            std::to_string(i) + "]:",
                        corr_st.status().err.toString());
      }
      auto wrapped_st = CorrectedCamera::create(cam_model, corr_st.value());
      if (!wrapped_st) {
        THROW_FATAL_CTX(ErrorCode::InvalidCameraState,
                        "Cannot apply the image correction of cameras[" +
                            std::to_string(i) + "]:",
                        wrapped_st.status().err.toString());
      }
      cam_model = wrapped_st.value();
      std::cout << "Camera " << i << ": image correction " << corr_path
                << std::endl;
    }
    cam_model->max_intensity = max_intensity;
    cam_model->is_active = true;
    s._cam_list.emplace_back(std::move(cam_model));
  }

  const json &imgs = r.req("image_lists");
  if (!imgs.is_array() || static_cast<int>(imgs.size()) != s._n_cam)
    r.fail("image_lists", "must be an array of " + std::to_string(s._n_cam) +
                              " paths, one per camera");
  s._image_file_paths.clear();
  for (size_t i = 0; i < imgs.size(); ++i) {
    if (!imgs[i].is_string())
      r.fail("image_lists[" + std::to_string(i) + "]", "must be a string");
    s._image_file_paths.push_back(resolve_path_from_config_root(
        config_root, imgs[i].get<std::string>()));
  }

  JsonReader vv = r.child("view_volume_mm");
  vv.warnUnknown({"x", "y", "z"});
  const auto x = vv.numbers("x", 2), y = vv.numbers("y", 2),
             z = vv.numbers("z", 2);
  s._axis_limit.x_min = x[0];
  s._axis_limit.x_max = x[1];
  s._axis_limit.y_min = y[0];
  s._axis_limit.y_max = y[1];
  s._axis_limit.z_min = z[0];
  s._axis_limit.z_max = z[1];

  s._voxel_to_mm = r.num("voxel_mm");
  s._output_path =
      resolve_path_from_config_root(config_root, r.str("output_dir"));
  if (!s._output_path.empty() && s._output_path.back() != '/' &&
      s._output_path.back() != '\\')
    s._output_path.push_back('/');

  const json &objs = r.req("objects");
  if (!objs.is_array() || objs.empty())
    r.fail("objects", "must be a non-empty array");
  s._object_types.clear();
  s._object_config_paths.clear();
  for (size_t i = 0; i < objs.size(); ++i) {
    JsonReader o(objs[i], "objects[" + std::to_string(i) + "]", file);
    s._object_types.push_back(o.str("type"));
    // the object's settings are inline, or in a text file ("config_file")
    s._object_config_paths.push_back(
        o.has("config_file")
            ? resolve_path_from_config_root(config_root, o.str("config_file"))
            : file + "#" + std::to_string(i));
  }

  s._load_track = false;
  s._load_track_frame = s._frame_start;
  if (r.has("load_tracks")) {
    JsonReader l = r.child("load_tracks");
    l.warnUnknown({"enabled", "frame", "folder"});
    s._load_track = l.boolean("enabled");
    if (s._load_track) {
      if (l.has("frame"))
        s._load_track_frame = l.integer("frame");
      s._load_track_path =
          resolve_path_from_config_root(config_root, l.str("folder"));
      if (!s._load_track_path.empty() && s._load_track_path.back() != '/')
        s._load_track_path.push_back('/');
    }
  }

  if (r.has("save_interval")) {
    s._save_interval = r.integer("save_interval");
    if (s._save_interval <= 0)
      r.fail("save_interval", "must be a positive number of frames");
  }
  if (r.has("profile"))
    s._profile = r.boolean("profile");

  // sanity checks (as for config.txt)
  if (s._frame_end < s._frame_start)
    r.fail("frames", "must be [start, end] with end >= start");
  if (s._n_thread < 0)
    s._n_thread = 0;
  return true;
}

// entry i of "objects" in config.json
const json &object_entry(const json &doc, int index, const std::string &file) {
  JsonReader top(doc, "", file);
  const json &objs = top.req("objects");
  if (!objs.is_array() || index < 0 || index >= static_cast<int>(objs.size()))
    top.fail("objects", "has no entry " + std::to_string(index));
  return objs[static_cast<size_t>(index)];
}

// the settings every object type has, from its entry in config.json
void read_common_json(ObjectConfig &c, const JsonReader &o,
                      const BasicSetting &settings) {
  const double vox = settings._voxel_to_mm;

  JsonReader ip = o.child("initial_phase");
  ip.warnUnknown({"search_radius_vox", "frames"});
  c._stb_param._radius_search_obj = ip.num("search_radius_vox") * vox;
  c._stb_param._n_initial_frames = ip.integer("frames");
  if (c._stb_param._n_initial_frames < 4)
    c._stb_param._n_initial_frames = 4; // n_initial_frames must be >= 4.

  JsonReader cp = o.child("convergence_phase");
  cp.warnUnknown({"avg_spacing_vox"});
  c._stb_param._radius_search_track = cp.num("avg_spacing_vox") * vox;

  JsonReader sh = o.child("shake");
  sh.warnUnknown({"width_vox", "loops", "ghost_threshold"});
  c._shake_param._shake_width = sh.num("width_vox") * vox;
  c._shake_param._shakewidth_min = c._shake_param._shake_width / 20;
  c._shake_param._n_shake_loop = sh.integer("loops");
  c._shake_param._thred_ghost = sh.num("ghost_threshold");

  JsonReader pf = o.child("predictive_field");
  pf.warnUnknown({"grid", "search_radius_vox"});
  c._pf_param.limit = settings._axis_limit;
  const std::vector<int> grid = pf.integers("grid", 3);
  c._pf_param.nx = grid[0];
  c._pf_param.ny = grid[1];
  c._pf_param.nz = grid[2];
  c._pf_param.r = pf.num("search_radius_vox") * vox;

  JsonReader ipr = o.child("ipr");
  ipr.warnUnknown({"loops", "reduced_cameras", "reduced_loops", "tol_2d_px",
                   "tol_3d_vox"});
  c._ipr_param.n_loop_ipr = ipr.integer("loops");
  c._sm_param.tol_2d_px = ipr.num("tol_2d_px");
  c._sm_param.tol_3d_mm = ipr.num("tol_3d_vox") * vox; // in mm
  c._sm_param.match_cam_count =
      std::min(c._sm_param.match_cam_count,
               settings._n_cam); // cannot be larger than n_cam
  c._sm_param.limit = settings._axis_limit;
  c._sm_param.profile = settings._profile;
  c._ipr_param.n_cam_reduced = ipr.integer("reduced_cameras");
  c._ipr_param.n_loop_ipr_reduced = ipr.integer("reduced_loops");

  c._vsc_param._output_path = settings._output_path;
}
} // namespace

// --------------------------- BasicSetting ---------------------------
bool BasicSetting::readConfig(const std::string &config_path) {
  const fs::path config_abs = absolute_lexical(fs::path(config_path));
  const fs::path config_root = config_abs.parent_path();
  _config_root = config_root.generic_string();
  if (!_config_root.empty()) {
    char back = _config_root.back();
    if (back != '/' && back != '\\') {
      _config_root.push_back('/');
    }
  }

  if (has_json_extension(config_abs.string()))
    return read_basic_json(*this, config_abs, config_root);

  std::ifstream file(config_abs.string());
  if (!file.is_open()) {
    std::cerr << "Failed to open config file: " << config_abs.string()
              << std::endl;
    return false;
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    // trim spaces
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    // remove comments
    size_t comment_pos = line.find('#');
    if (comment_pos == 0)
      continue;
    if (comment_pos != std::string::npos)
      line.erase(comment_pos);

    // trim again
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    lines.push_back(line);
  }
  file.close();

  if (lines.size() < 11) {
    std::cerr << "Invalid config: too few lines." << std::endl;
    return false;
  }

  std::stringstream parser;
  int line_id = 0;

  // frame range
  parser.str(lines[line_id++]);
  std::getline(parser, line, ',');
  _frame_start = std::stoi(line);
  std::getline(parser, line, ',');
  _frame_end = std::stoi(line);
  parser.clear();

  // fps and threads
  _fps = std::stoi(lines[line_id++]);
  _n_thread = std::stoi(lines[line_id++]);

  // number of cameras
  _n_cam = std::stoi(lines[line_id++]);
  _cam_list.clear();
  _cam_list.reserve(static_cast<size_t>(_n_cam));
  for (int i = 0; i < _n_cam; ++i) {
    parser.str(lines[line_id++]);
    std::string cam_path;
    std::getline(parser, cam_path, ',');
    std::getline(parser, line, ',');
    int max_intensity = std::stoi(line);

    cam_path = resolve_path_from_config_root(config_root, cam_path);

    auto cam_model_st = CameraFactory::loadFromFile(cam_path);
    if (!cam_model_st) {
      THROW_FATAL_CTX(ErrorCode::InvalidCameraState,
                      "Failed to load camera model from:", cam_path);
    }
    auto cam_model = cam_model_st.value();
    cam_model->max_intensity = max_intensity;
    cam_model->is_active = true;
    _cam_list.emplace_back(std::move(cam_model));
    parser.clear();
  }

  // image file paths
  _image_file_paths.reserve(static_cast<size_t>(_n_cam));
  for (int i = 0; i < _n_cam; ++i) {
    _image_file_paths.push_back(
        resolve_path_from_config_root(config_root, lines[line_id++]));
  }

  // axis limits
  parser.str(lines[line_id++]);
  std::getline(parser, line, ',');
  _axis_limit.x_min = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.x_max = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.y_min = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.y_max = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.z_min = std::stod(line);
  std::getline(parser, line, ',');
  _axis_limit.z_max = std::stod(line);
  parser.clear();

  // unit conversion and output path
  _voxel_to_mm = std::stod(lines[line_id++]);
  _output_path = resolve_path_from_config_root(config_root, lines[line_id++]);
  if (!_output_path.empty()) {
    char back = _output_path.back();
    if (back != '/' && back != '\\')
      _output_path.push_back('/');
  }

  // object types
  parser.str(lines[line_id++]);
  while (std::getline(parser, line, ',')) {
    if (!line.empty())
      _object_types.push_back(line);
  }
  parser.clear();

  // object config file paths
  _object_config_paths.reserve(_object_types.size());
  for (size_t i = 0; i < _object_types.size(); ++i) {
    _object_config_paths.push_back(
        resolve_path_from_config_root(config_root, lines[line_id++]));
  }

  // track loading
  parser.str(lines[line_id++]);
  std::getline(parser, line, ',');
  _load_track = std::stoi(line) != 0;
  _load_track_frame = _frame_start;
  if (_load_track) {
    std::getline(parser, line, ',');
    if (!line.empty())
      _load_track_frame = std::stoi(line);
    if (line_id < static_cast<int>(lines.size()))
      _load_track_path =
          resolve_path_from_config_root(config_root, lines[line_id++]);
  }
  parser.clear();

  // Optional settings: "key = value" lines after the fixed ones. Other lines
  // there (such as the GUI's track-path lines when no tracks are loaded) are
  // ignored, as before.
  for (; line_id < static_cast<int>(lines.size()); ++line_id) {
    std::string key, value;
    if (!parse_key_value(lines[line_id], key, value))
      continue;
    if (key == "save_interval") {
      _save_interval = parse_positive_int(key, value, config_path);
    } else if (key == "profile") {
      if (value != "0" && value != "1")
        THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                        "Config setting profile must be 0 or 1, got: " + value,
                        config_path + "\n");
      _profile = value == "1";
    } else {
      std::cerr << "Warning: unknown setting in " << config_path << ": "
                << key << std::endl;
    }
  }

  // sanity checks
  if (_n_cam <= 0) {
    THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                    "Invalid n_cam in config: ", config_path + "\n");
  }
  if (_frame_end < _frame_start) {
    THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                    "Invalid frame range in config: ", config_path + "\n");
  }
  if ((int)_image_file_paths.size() != _n_cam) {
    THROW_FATAL_CTX(
        ErrorCode::InvalidArgument,
        "Image path count != n_cam in config: ", config_path + "\n");
  }
  if (_object_types.size() != _object_config_paths.size()) {
    THROW_FATAL_CTX(ErrorCode::InvalidArgument,
                    "Object type count != config path count in config: ",
                    config_path + "\n");
  }
  if (_n_thread < 0)
    _n_thread = 0;

  return true;
}

// --------------------------- ObjectConfig (Common) ---------------------------
std::pair<std::vector<std::string>, int>
ObjectConfig::readCommonConfig(const std::string &filepath,
                               BasicSetting &settings) {
  std::ifstream file(filepath);
  REQUIRE_CTX(file.is_open(), ErrorCode::IOfailure,
              "Config Error: Cannot open file ", filepath + "\n");

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    // trim
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    // remove comments
    size_t commentpos = line.find('#');
    if (commentpos == 0)
      continue;
    if (commentpos != std::string::npos)
      line.erase(commentpos);

    // trim again
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty())
      continue;

    lines.push_back(line);
  }
  file.close();

  int id = 0;
  try {
    // STB params
    _stb_param._radius_search_obj =
        std::stod(lines[id++]) * settings._voxel_to_mm;
    _stb_param._n_initial_frames = std::stoi(lines[id++]);
    if (_stb_param._n_initial_frames < 4)
      _stb_param._n_initial_frames = 4; // n_initial_frames must be >= 4.
    _stb_param._radius_search_track =
        std::stod(lines[id++]) * settings._voxel_to_mm;

    // Shake params
    _shake_param._shake_width = std::stod(lines[id++]) * settings._voxel_to_mm;
    _shake_param._shakewidth_min = _shake_param._shake_width / 20;

    // Predictive field params
    _pf_param.limit = settings._axis_limit;
    _pf_param.nx = std::stoi(lines[id++]);
    _pf_param.ny = std::stoi(lines[id++]);
    _pf_param.nz = std::stoi(lines[id++]);
    _pf_param.r = std::stod(lines[id++]) * settings._voxel_to_mm;

    // IPR & shake loop
    _ipr_param.n_loop_ipr = std::stoi(lines[id++]);
    _shake_param._n_shake_loop = std::stoi(lines[id++]);
    _shake_param._thred_ghost = std::stod(lines[id++]);

    // StereoMatch params
    _sm_param.tol_2d_px = std::stod(lines[id++]);
    _sm_param.tol_3d_mm =
        std::stod(lines[id++]) * settings._voxel_to_mm; // in mm
    _sm_param.match_cam_count =
        std::min(_sm_param.match_cam_count,
                 settings._n_cam); // cannot be larger than n_cam
    _sm_param.limit = settings._axis_limit;
    _sm_param.profile = settings._profile;

    _ipr_param.n_cam_reduced = std::stoi(lines[id++]);
    _ipr_param.n_loop_ipr_reduced = std::stoi(lines[id++]);

    _vsc_param._output_path = settings._output_path;

  } catch (...) {
    THROW_FATAL_CTX(ErrorCode::IOfailure,
                    "Config Error: Failed to parse common config in ",
                    filepath + "\n");
  }

  return {lines, id};
}

// --------------------------- TracerConfig ---------------------------
bool TracerConfig::readConfig(const std::string &filepath,
                              BasicSetting &settings) {
  std::string json_file;
  int json_index = -1;
  const bool is_json = split_json_object_path(filepath, json_file, json_index);
  std::vector<std::string> lines;
  int id = 0;
  if (is_json) {
    // an entry of "objects" in config.json
    const json doc = load_json(json_file);
    const std::string where = "objects[" + std::to_string(json_index) + "]";
    JsonReader o(object_entry(doc, json_index, json_file), where, json_file);
    o.warnUnknown({"type", "initial_phase", "convergence_phase", "shake",
                   "predictive_field", "ipr", "tracer"});
    if (o.str("type") != "Tracer")
      o.fail(where + ".type", "is not \"Tracer\"");
    read_common_json(*this, o, settings);
    JsonReader t = o.child("tracer");
    t.warnUnknown({"intensity_threshold", "radius_px", "roi_from_otf", "roi_sigma",
                   "otf_file", "detector", "shape_check", "shape_tol_px",
                   "shape_tol_rel", "lines"});
    _min_obj_int = t.integer("intensity_threshold");
    _radius_obj = t.num("radius_px");
    if (t.has("roi_from_otf"))
      _roi_from_otf = t.boolean("roi_from_otf");
    if (t.has("roi_sigma"))
      _roi_sigma = t.num("roi_sigma");
    if (t.has("detector")) {
      _detector = t.str("detector");
      if (_detector != "peaks" && _detector != "lines")
        t.fail("objects[" + std::to_string(json_index) + "].tracer.detector",
               "must be \"peaks\" or \"lines\"");
    }
    if (t.has("shape_check"))
      _shape_check = t.boolean("shape_check");
    if (t.has("shape_tol_px"))
      _shape_tol_px = t.num("shape_tol_px");
    if (t.has("shape_tol_rel"))
      _shape_tol_rel = t.num("shape_tol_rel");
    if (t.has("lines")) {
      // every setting optional; the names of lines.py's DET (LineFinder.h)
      JsonReader l = t.child("lines");
      l.warnUnknown({"lengths_x", "lengths_y", "sigma0", "bg_guard", "noise_sub",
                     "k_detect", "core_px", "fit_margin_sigma", "fit_max_half",
                     "fit_iters", "deblend_passes", "second_pass", "validate",
                     "L_min", "L_max", "sigma_min", "sigma_max", "centre_box_frac",
                     "dynamic_bounds", "length_growth", "length_margin", "nms_symmetric",
                     "c_max_sigma", "clipped", "noise"});
      LineFinderParam &q = _lines;
      if (l.has("lengths_x")) q.lengths_x = l.numbers("lengths_x", -1);
      if (l.has("lengths_y")) q.lengths_y = l.numbers("lengths_y", -1);
      if (l.has("sigma0")) q.sigma0 = l.num("sigma0");
      if (l.has("bg_guard")) q.bg_guard = l.integer("bg_guard");
      if (l.has("noise_sub")) q.noise_sub = l.integer("noise_sub");
      if (l.has("k_detect")) q.k_detect = l.num("k_detect");
      if (l.has("core_px")) q.core_px = l.num("core_px");
      if (l.has("fit_margin_sigma")) q.fit_margin_sigma = l.num("fit_margin_sigma");
      if (l.has("fit_max_half")) q.fit_max_half = l.integer("fit_max_half");
      if (l.has("fit_iters")) q.fit_iters = l.integer("fit_iters");
      if (l.has("deblend_passes")) q.deblend_passes = l.integer("deblend_passes");
      if (l.has("second_pass")) q.second_pass = l.boolean("second_pass");
      if (l.has("validate")) q.validate = l.boolean("validate");
      if (l.has("L_min")) q.L_min = l.num("L_min");
      if (l.has("L_max")) q.L_max = l.num("L_max");
      if (l.has("sigma_min")) q.sigma_min = l.num("sigma_min");
      if (l.has("sigma_max")) q.sigma_max = l.num("sigma_max");
      if (l.has("centre_box_frac")) q.centre_box_frac = l.num("centre_box_frac");
      if (l.has("dynamic_bounds")) q.dynamic_bounds = l.boolean("dynamic_bounds");
      if (l.has("length_growth")) q.length_growth = l.num("length_growth");
      if (l.has("length_margin")) q.length_margin = l.num("length_margin");
      if (l.has("nms_symmetric")) q.nms_symmetric = l.boolean("nms_symmetric");
      if (l.has("c_max_sigma")) q.c_max_sigma = l.num("c_max_sigma");
      if (l.has("clipped")) q.clipped = l.boolean("clipped");
      if (l.has("noise")) {
        // a number for every camera, or one per camera
        if (l.req("noise").is_number())
          _lines_noise = {l.num("noise")};
        else
          _lines_noise = l.numbers("noise", settings._n_cam);
      }
    }
    if (t.has("otf_file")) {
      // relative to the config file's folder
      _otf_file = t.str("otf_file");
      if (!_otf_file.empty() && !std::filesystem::path(_otf_file).is_absolute())
        _otf_file = (std::filesystem::path(json_file).parent_path() / _otf_file).string();
    }
  } else {
    std::tie(lines, id) = readCommonConfig(filepath, settings);
  }

  try {
    if (!is_json) {
      _min_obj_int = std::stoi(lines[id++]);
      _radius_obj = std::stod(lines[id++]);
    }

    // Initialize OTF: from tracer.otf_file when given (config.json only)
    if (!_otf_file.empty()) {
      _otf.loadParam(_otf_file);
      _otf._output_path = settings._output_path;
      REQUIRE_CTX(_otf._param.n_cam == settings._n_cam, ErrorCode::InvalidArgument,
                  "The OTF file's camera count differs from the config's: ",
                  _otf_file);
      std::cout << "Loaded the OTF from " << _otf_file << " (model "
                << (_otf._param.model == 1 ? "segment" : "gauss") << ")\n";
      return true;
    }
    _otf.loadParam(settings._n_cam, 5, 5, 5, settings._axis_limit,
                   settings._cam_list);
    _otf._output_path = settings._output_path;

    // Estimate OTF using first few frames
    std::vector<ImageIO> imgio_list;
    imgio_list.reserve(settings._n_cam);
    for (const auto &path : settings._image_file_paths) {
      ImageIO io;
      io.loadImgPath(settings._config_root, path);
      imgio_list.push_back(io);
    }

    int n_img_otf = 5;
    std::vector<Image> img_list(n_img_otf);
    std::cout << "Estimating OTF...\n";
    for (int i = 0; i < settings._n_cam; i++) {
      for (int j = 0; j < n_img_otf; j++) {
        img_list[j] = imgio_list[i].loadImg(settings._frame_start + j);
      }

      _otf.estimateUniformOTFFromImage(i, *this, img_list);
    }
  } catch (...) {
    THROW_FATAL_CTX(ErrorCode::IOfailure,
                    "Config Error: Failed to parse common config in ",
                    filepath + "\n");
  }

  return true;
}

// --------------------------- BubbleConfig ---------------------------
bool BubbleConfig::readConfig(const std::string &filepath,
                              BasicSetting &settings) {
  std::string json_file;
  int json_index = -1;
  if (split_json_object_path(filepath, json_file, json_index)) {
    // an entry of "objects" in config.json
    const json doc = load_json(json_file);
    const std::string where = "objects[" + std::to_string(json_index) + "]";
    JsonReader o(object_entry(doc, json_index, json_file), where, json_file);
    o.warnUnknown({"type", "initial_phase", "convergence_phase", "shake",
                   "predictive_field", "ipr", "bubble"});
    if (o.str("type") != "Bubble")
      o.fail(where + ".type", "is not \"Bubble\"");
    read_common_json(*this, o, settings);
    JsonReader b = o.child("bubble");
    b.warnUnknown({"radius_min_px", "radius_max_px", "sense"});
    _radius_min = b.integer("radius_min_px");
    _radius_max = b.integer("radius_max_px");
    _sense = b.num("sense");
  } else {
    auto [lines, id] = readCommonConfig(filepath, settings);

    try {
      _radius_min = std::stoi(lines[id++]);
      _radius_max = std::stoi(lines[id++]);
      _sense = std::stod(lines[id++]);

    } catch (...) {
      THROW_FATAL_CTX(ErrorCode::IOfailure,
                      "Config Error: Failed to parse common config in ",
                      filepath + "\n");
    }
  }

  _output_path = settings._output_path;

  if (settings._load_track) {
    if (!_bb_ref_img.loadRefImg(_output_path, settings._n_cam)) {
      std::cerr << "Warning: Failed to load bubble reference images from "
                << _output_path << std::endl;
    } else {
      std::cout << "Loaded bubble reference images from " << _output_path
                << std::endl;
    }
  }

  return true;
}

std::unique_ptr<Object3D> TracerConfig::creatObject3D(CreateArgs args) const {
  auto up = std::make_unique<Tracer3D>();

  if (args._proto)
    up->copyCommonFrom(*args._proto);
  if (args._pt_center)
    up->_pt_center = *args._pt_center;
  if (!args._obj2d_ready.empty())
    up->_obj2d_list = std::move(args._obj2d_ready);

  // Tracer radius selection
  if (args._r2d_px_tracer) {
    up->_r2d_px = *args._r2d_px_tracer;
  } else if (args._proto) {
    if (auto *tr = dynamic_cast<const Tracer3D *>(args._proto))
      up->_r2d_px = tr->_r2d_px;
    else
      up->_r2d_px = _radius_obj;
  } else {
    up->_r2d_px = _radius_obj;
  }

  up->_is_tracked = false;
  return up;
}

// --------------------------- BubbleConfig ---------------------------
std::unique_ptr<Object3D> BubbleConfig::creatObject3D(CreateArgs args) const {
  auto up = std::make_unique<Bubble3D>();

  if (args._proto)
    up->copyCommonFrom(*args._proto);
  if (args._pt_center)
    up->_pt_center = *args._pt_center;
  if (!args._obj2d_ready.empty())
    up->_obj2d_list = std::move(args._obj2d_ready);

  // Bubble radius selection
  if (args._r3d_bubble) {
    up->_r3d = *args._r3d_bubble;
  } else if (args._proto) {
    if (auto *bb = dynamic_cast<const Bubble3D *>(args._proto))
      up->_r3d = bb->_r3d;
  }
  if (up->_r3d <= 0.0 && args._compute_bubble_radius &&
      !up->_obj2d_list.empty()) {
    if (args._cam_list) {
      up->_r3d = Bubble::calRadiusFromCams(*args._cam_list, up->_pt_center,
                                           up->_obj2d_list);
    }
  }
  if (up->_r3d <= 0.0)
    up->_r3d = -1.0;

  up->_is_tracked = false;
  return up;
}
