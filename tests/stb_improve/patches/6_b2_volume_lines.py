"""B2: anchor the image line of a line of sight at the points where it enters and leaves the view volume."""
p = r"D:\Refraction\pyOpenLPT\src\srcSTB\StereoMatch.cpp"
raw = open(p, "rb").read().decode("utf-8")
crlf = "\r\n" in raw
s = raw.replace("\r\n", "\n")

old = '''bool StereoMatch::makeLine2DFromLOS3D(int cam_id, const Line3D &los,
                                              Line2D &out_line) const {
  // TODO(refraction): This workaround avoids projecting los.pt (interface point)
  // to reduce boundary degeneracy in refractive projection. It does NOT address
  // physical invalid branches (e.g., TIR/parallel geometry) and should be
  // replaced by a robust failure-aware LOS-to-2D construction.
  Pt3D p_a{los.pt[0] + los.unit_vector[0], los.pt[1] + los.unit_vector[1],
           los.pt[2] + los.unit_vector[2]};
  Pt3D p_b{los.pt[0] + 2.0 * los.unit_vector[0],
           los.pt[1] + 2.0 * los.unit_vector[1],
           los.pt[2] + 2.0 * los.unit_vector[2]};
'''
assert s.count(old) == 1, s.count(old)
new = '''bool StereoMatch::makeLine2DFromLOS3D(int cam_id, const Line3D &los,
                                              Line2D &out_line) const {
  // Seen through a refractive window, the image of a line of sight is a curve,
  // and a straight line through the images of two of its points follows it
  // only near them. The points are where the line of sight enters and leaves
  // the view volume, so the line follows the curve where the matches can be
  // (a point outside the volume is never matched). A line of sight that misses
  // the volume gets no image line.
  const AxisLimit &lim = _obj_cfg._sm_param.limit;
  const double lo[3] = {lim.x_min, lim.y_min, lim.z_min};
  const double hi[3] = {lim.x_max, lim.y_max, lim.z_max};
  double t_in = 0.0, t_out = std::numeric_limits<double>::infinity();
  for (int d = 0; d < 3; ++d) { // the slab method, for los.pt + t * u, t >= 0
    const double o = los.pt[d], u = los.unit_vector[d];
    if (std::fabs(u) < 1e-15) {
      if (o < lo[d] || o > hi[d])
        return false;
      continue;
    }
    double ta = (lo[d] - o) / u, tb = (hi[d] - o) / u;
    if (ta > tb)
      std::swap(ta, tb);
    t_in = std::max(t_in, ta);
    t_out = std::min(t_out, tb);
    if (t_in > t_out)
      return false;
  }
  if (t_out - t_in < 1e-6) // grazing a corner: any second point on the ray
    t_out = t_in + 1.0;
  Pt3D p_a{los.pt[0] + t_in * los.unit_vector[0],
           los.pt[1] + t_in * los.unit_vector[1],
           los.pt[2] + t_in * los.unit_vector[2]};
  Pt3D p_b{los.pt[0] + t_out * los.unit_vector[0],
           los.pt[1] + t_out * los.unit_vector[1],
           los.pt[2] + t_out * los.unit_vector[2]};
'''
s = s.replace(old, new)
out = s.replace("\n", "\r\n") if crlf else s
open(p, "wb").write(out.encode("utf-8"))
print("B2 patched; crlf", crlf)
