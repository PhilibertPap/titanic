import gmsh


filename = "mesh/coque"
SHELL_CELL_TAG = 1
RIVET_CELL_TAG = 2  # kept for compatibility with the solver configuration


# Geometry scales for a local shell patch
L = 100.0   # longitudinal length [m] -> x
# Titanic draft is commonly cited around 10.5 m; we span approximately the immersed height.
DRAFT = 10.5
Z_BOTTOM = -DRAFT
Z_TOP = 0.0

# Shell mainly in OXZ plane: y is the local outward offset of the shell patch.
# Shape cues (from Titanic hull lines) used here:
# - rounded bilge / stronger flare toward the upper side shell
# - long fuller mid-body
# - narrowing toward bow and stern
Y_KEEL = 0.08
Y_BILGE_BASE = 0.75
Y_WATERLINE_BASE = 2.40
Y_MIDSHIP_FULLNESS = 0.55
Y_LONGITUDINAL_WARP = 0.06

# Longitudinal curvature in z (very mild):
# - near-waterline sheer-like variation stronger than near keel
# - weak global sag/hog over the modeled segment
Z_SHEER_AMPLITUDE = 0.28
Z_KEEL_SAG_AMPLITUDE = 0.10
Z_END_RISE_AMPLITUDE = 0.22

# Longitudinal planform narrowing (half-breadth effect): near-constant mid-body, tapered ends
MIDBODY_U0 = 0.18
MIDBODY_U1 = 0.82
END_TAPER_MIN = 0.65

# Loft sections (couples) along x
N_SECTIONS_X = 11
N_SECTION_PTS = 13

# B-spline control net (geometry smoothness, not mesh density)
NU_CTRL = 24
NV_CTRL = 10

# Iceberg trajectory (used only to define a mesh-refinement band)
ICEBERG_CENTER_Y = 1.10     # m, aligned with stronger hull-side offset near impact depth
ICEBERG_CENTER_Z = -7.5     # m, below waterline (z=0)

# Mesh size field (refine around the iceberg trajectory band)
SIZE_MIN = 0.30
SIZE_MAX = 2.20
DIST_MIN = 1.0
DIST_MAX = 5.0


def _smoothstep(a: float, b: float, x: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    t = (x - a) / (b - a)
    return t * t * (3.0 - 2.0 * t)


def _midbody_fullness_factor(u: float) -> float:
    # 1.0 in the parallel middle body, tapered near both ends.
    left = _smoothstep(0.0, MIDBODY_U0, u)
    right = 1.0 - _smoothstep(MIDBODY_U1, 1.0, u)
    plateau = min(left, right)
    return END_TAPER_MIN + (1.0 - END_TAPER_MIN) * plateau


def hull_xyz(u: float, v: float) -> tuple[float, float, float]:
    """Reference hull-side geometry used for diagnostics and control-point generation."""
    x = L * u
    xu = 2.0 * u - 1.0
    s = 0.5 * (v + 1.0)  # 0 = lower z, 1 = upper z
    fullness_x = _midbody_fullness_factor(u)
    endness = 1.0 - fullness_x

    # z = vertical coordinate, with mild longitudinal curvature (sag + sheer-like effect)
    z = Z_BOTTOM + (Z_TOP - Z_BOTTOM) * s
    z += Z_KEEL_SAG_AMPLITUDE * (xu * xu)
    z += Z_SHEER_AMPLITUDE * (s ** 2.2) * (xu * xu)
    # Extra end rise/sheer near bow/stern, stronger toward the upper hull
    z += Z_END_RISE_AMPLITUDE * (s ** 2.6) * (endness ** 1.4)

    # Section profile y(z): bilge + flare, with mild longitudinal fullness.
    # Fuller near midship, slimmer near the ends.
    fullness = 1.0 + Y_MIDSHIP_FULLNESS * (1.0 - xu * xu)
    y_bilge = Y_BILGE_BASE * fullness * fullness_x
    y_waterline = Y_WATERLINE_BASE * fullness * fullness_x
    if s <= 0.50:
        t = s / 0.50
        y_section = Y_KEEL + (y_bilge - Y_KEEL) * (t ** 1.9)
    else:
        t = (s - 0.50) / 0.50
        y_section = y_bilge + (y_waterline - y_bilge) * (t ** 1.15)
    y = y_section + Y_LONGITUDINAL_WARP * xu * (2.0 * s - 1.0)
    return x, y, z


def _build_smooth_hull_surface(occ) -> tuple[int, dict[str, list[int]]]:
    # Build a smooth hull patch by lofting several smooth transverse sections (couples).
    # This gives a more ship-like geometry than a single control-net B-spline surface.
    section_wires: list[int] = []
    for i_sec in range(N_SECTIONS_X):
        u = i_sec / (N_SECTIONS_X - 1)
        pts: list[int] = []
        for j in range(N_SECTION_PTS):
            v = -1.0 + 2.0 * j / (N_SECTION_PTS - 1)
            x, y, z = hull_xyz(u, v)
            pts.append(occ.addPoint(x, y, z))
        spline = occ.addSpline(pts)
        wire = occ.addWire([spline], checkClosed=False)
        section_wires.append(wire)

    loft_out = occ.addThruSections(
        section_wires,
        makeSolid=False,
        makeRuled=False,
        maxDegree=5,
    )
    surf_tags = [tag for dim, tag in loft_out if dim == 2]
    if len(surf_tags) != 1:
        raise RuntimeError(f"Expected one lofted surface, got: {loft_out}")
    surf = surf_tags[0]
    occ.synchronize()

    # Identify boundary curves geometrically to preserve BC tags used by the solver.
    boundary = gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False)
    edge_map = {"left": [], "right": [], "bottom": [], "top": []}
    remaining: list[tuple[int, float]] = []
    for dim, tag in boundary:
        if dim != 1:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
        xmid = 0.5 * (xmin + xmax)
        zmid = 0.5 * (zmin + zmax)
        if abs(xmid - 0.0) < 1e-3 * L:
            edge_map["left"].append(tag)
        elif abs(xmid - L) < 1e-3 * L:
            edge_map["right"].append(tag)
        else:
            remaining.append((tag, zmid))

    if len(remaining) == 2:
        remaining.sort(key=lambda item: item[1])
        edge_map["bottom"].append(remaining[0][0])  # low z
        edge_map["top"].append(remaining[1][0])     # high z
    elif len(remaining) != 0:
        raise RuntimeError(f"Unexpected number of non-x boundary curves: {len(remaining)}")

    return surf, edge_map


def _add_mesh_size_field(occ) -> None:
    # Refinement trajectory follows the shell more closely than a straight line:
    # sample a 3D path at the impact depth across x (x+ -> x-).
    s_impact = max(0.0, min(1.0, (ICEBERG_CENTER_Z - Z_BOTTOM) / (Z_TOP - Z_BOTTOM)))
    v_impact = 2.0 * s_impact - 1.0
    path_points = []
    for k in range(41):
        u = 1.0 - k / 40.0
        x, y_shell, z = hull_xyz(u, v_impact)
        # Shift slightly outward/inward so the distance field stays centered on the target band
        y = y_shell + (ICEBERG_CENTER_Y - hull_xyz(0.5, v_impact)[1])
        path_points.append(occ.addPoint(x, y, z))
    traj = occ.addSpline(path_points)
    occ.synchronize()

    field = gmsh.model.mesh.field
    f_dist = field.add("Distance")
    field.setNumbers(f_dist, "CurvesList", [traj])
    field.setNumber(f_dist, "Sampling", 200)

    f_th = field.add("Threshold")
    field.setNumber(f_th, "InField", f_dist)
    field.setNumber(f_th, "SizeMin", SIZE_MIN)
    field.setNumber(f_th, "SizeMax", SIZE_MAX)
    field.setNumber(f_th, "DistMin", DIST_MIN)
    field.setNumber(f_th, "DistMax", DIST_MAX)

    field.setAsBackgroundMesh(f_th)

    # Let the background field drive the mesh size instead of point-wise defaults.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


def main():
    gmsh.initialize()
    gmsh.model.add("titanic_hull_segment")
    occ = gmsh.model.occ

    surf, edges = _build_smooth_hull_surface(occ)

    gmsh.model.addPhysicalGroup(2, [surf], SHELL_CELL_TAG)
    gmsh.model.setPhysicalName(2, SHELL_CELL_TAG, "HullPlate")

    # Boundary tags expected by the solver
    if edges["left"]:
        gmsh.model.addPhysicalGroup(1, edges["left"], 1)
        gmsh.model.setPhysicalName(1, 1, "Left")
    if edges["right"]:
        gmsh.model.addPhysicalGroup(1, edges["right"], 2)
        gmsh.model.setPhysicalName(1, 2, "Right")
    if edges["bottom"]:
        gmsh.model.addPhysicalGroup(1, edges["bottom"], 3)
        gmsh.model.setPhysicalName(1, 3, "Bottom")
    if edges["top"]:
        gmsh.model.addPhysicalGroup(1, edges["top"], 4)
        gmsh.model.setPhysicalName(1, 4, "Top")

    _add_mesh_size_field(occ)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename + ".msh")
    gmsh.finalize()


if __name__ == "__main__":
    main()
