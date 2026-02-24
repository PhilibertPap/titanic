import gmsh


filename = "mesh/coque"
SHELL_CELL_TAG = 1
RIVET_CELL_TAG = 2  # kept for compatibility with the solver configuration


# Geometry scales for a Titanic-like half-hull shell surface
# (order of magnitude anchors: length ~269 m, beam ~28.2 m, draft ~10.5 m)
L = 269.0   # ship length [m] -> x
B_HALF = 14.1  # half-beam [m] -> y
DRAFT = 10.5   # draft below waterline [m]
FREEBOARD_TO_DECK = 7.8  # approx keel->deck total height ~18.3 m
Z_BOTTOM = -DRAFT
Z_TOP = FREEBOARD_TO_DECK

# Half-hull surface: y is the half-breadth (outboard side).
# Shape cues (from Titanic hull lines) approximated here:
# - rounded bilge
# - broad parallel mid-body
# - tapered bow/stern
# - slight tumblehome toward upper deck (very mild)
Y_KEEL = 0.08
Y_BILGE_BASE = 8.6
Y_WATERLINE_BASE = 13.9
Y_DECK_BASE = 13.5
Y_MIDSHIP_FULLNESS = 0.10
Y_LONGITUDINAL_WARP = 0.12

# Longitudinal curvature in z (very mild):
# - near-waterline sheer-like variation stronger than near keel
# - weak global sag/hog over the modeled segment
Z_SHEER_AMPLITUDE = 0.45
Z_KEEL_SAG_AMPLITUDE = 0.10
Z_END_RISE_AMPLITUDE = 0.18

# Longitudinal planform narrowing (half-breadth effect): long parallel mid-body and smoother ends
MIDBODY_U0 = 0.14
MIDBODY_U1 = 0.86
END_TAPER_MIN = 0.22

# Loft sections (couples) along x
N_SECTIONS_X = 11
N_SECTION_PTS = 13

# B-spline control net (geometry smoothness, not mesh density)
NU_CTRL = 24
NV_CTRL = 10

# Iceberg trajectory (used only to define a mesh-refinement band)
ICEBERG_CENTER_Y = -10.8    # m, starboard side (sign flipped)
ICEBERG_CENTER_Z = -7.5     # m, below waterline (z=0)

# Mesh size field (refine around the iceberg trajectory band)
SIZE_MIN = 0.25
SIZE_MAX = 2.20
DIST_MIN = 1.0
DIST_MAX = 5.5

# Extra refinement in a horizontal band (all x, limited z-thickness) around
# the strongest section-curvature zone (roughly mid-height between keel and deck).
MIDHEIGHT_SIZE_MIN = 0.45
MIDHEIGHT_SIZE_MAX = 2.20
MIDHEIGHT_BAND_HALF_THICKNESS_Z = 2.0


def _smoothstep(a: float, b: float, x: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    t = (x - a) / (b - a)
    return t * t * (3.0 - 2.0 * t)


def _midbody_fullness_factor(u: float) -> float:
    # 1.0 in the parallel middle body, smoothly tapered near both ends.
    left = _smoothstep(0.0, MIDBODY_U0, u)
    right = 1.0 - _smoothstep(MIDBODY_U1, 1.0, u)
    plateau = min(left, right)
    return END_TAPER_MIN + (1.0 - END_TAPER_MIN) * plateau


def hull_xyz(u: float, v: float) -> tuple[float, float, float]:
    """Reference hull-side geometry used for diagnostics and control-point generation."""
    x = L * u
    xu = 2.0 * u - 1.0
    s = 0.5 * (v + 1.0)  # 0 = keel, 1 = upper deck line
    fullness_x = _midbody_fullness_factor(u)
    endness = 1.0 - fullness_x
    s_water = (0.0 - Z_BOTTOM) / (Z_TOP - Z_BOTTOM)

    # z = vertical coordinate, with mild longitudinal curvature (sag + sheer-like effect)
    z = Z_BOTTOM + (Z_TOP - Z_BOTTOM) * s
    z += Z_KEEL_SAG_AMPLITUDE * (xu * xu)
    z += Z_SHEER_AMPLITUDE * (max(s - s_water, 0.0) / max(1.0 - s_water, 1e-9)) ** 1.8 * (xu * xu)
    # Extra end rise/sheer near bow/stern, stronger toward the upper hull
    z += Z_END_RISE_AMPLITUDE * (s ** 2.2) * (endness ** 1.6)

    # Section profile y(z): bilge + flare, with mild longitudinal fullness.
    # Fuller near midship, slimmer near the ends.
    fullness = 1.0 + Y_MIDSHIP_FULLNESS * (1.0 - xu * xu)
    y_bilge = Y_BILGE_BASE * fullness * fullness_x
    y_waterline = Y_WATERLINE_BASE * fullness * fullness_x
    y_deck = Y_DECK_BASE * fullness * fullness_x

    # Piecewise smooth half-breadth profile from keel to deck
    s_bilge = 0.38
    if s <= s_bilge:
        t = s / s_bilge
        y_section = Y_KEEL + (y_bilge - Y_KEEL) * (t ** 1.75)
    elif s <= s_water:
        t = (s - s_bilge) / max(s_water - s_bilge, 1e-9)
        y_section = y_bilge + (y_waterline - y_bilge) * (t ** 0.95)
    else:
        t = (s - s_water) / max(1.0 - s_water, 1e-9)
        # very mild tumblehome toward the deck
        y_section = y_waterline + (y_deck - y_waterline) * (t ** 1.15)
    y = -(y_section + Y_LONGITUDINAL_WARP * xu * (2.0 * s - 1.0))
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

    # 1) Refinement along the iceberg trajectory
    f_dist_traj = field.add("Distance")
    field.setNumbers(f_dist_traj, "CurvesList", [traj])
    field.setNumber(f_dist_traj, "Sampling", 200)

    f_th_traj = field.add("Threshold")
    field.setNumber(f_th_traj, "InField", f_dist_traj)
    field.setNumber(f_th_traj, "SizeMin", SIZE_MIN)
    field.setNumber(f_th_traj, "SizeMax", SIZE_MAX)
    field.setNumber(f_th_traj, "DistMin", DIST_MIN)
    field.setNumber(f_th_traj, "DistMax", DIST_MAX)

    # 2) Extra refinement along x (not along z): a horizontal band centered at mid-height
    # to better resolve section curvature transitions.
    z_mid = 0.5 * (Z_BOTTOM + Z_TOP)
    y_extent = 1.2 * max(
        abs(Y_WATERLINE_BASE * (1.0 + Y_MIDSHIP_FULLNESS)),
        abs(Y_DECK_BASE * (1.0 + Y_MIDSHIP_FULLNESS)),
        abs(ICEBERG_CENTER_Y),
    ) + 1.0

    f_box_midheight = field.add("Box")
    field.setNumber(f_box_midheight, "VIn", MIDHEIGHT_SIZE_MIN)
    field.setNumber(f_box_midheight, "VOut", MIDHEIGHT_SIZE_MAX)
    field.setNumber(f_box_midheight, "XMin", -1.0)
    field.setNumber(f_box_midheight, "XMax", L + 1.0)
    field.setNumber(f_box_midheight, "YMin", -y_extent)
    field.setNumber(f_box_midheight, "YMax", y_extent)
    field.setNumber(
        f_box_midheight, "ZMin", z_mid - MIDHEIGHT_BAND_HALF_THICKNESS_Z
    )
    field.setNumber(
        f_box_midheight, "ZMax", z_mid + MIDHEIGHT_BAND_HALF_THICKNESS_Z
    )

    # 3) Combine both criteria
    f_min = field.add("Min")
    field.setNumbers(f_min, "FieldsList", [f_th_traj, f_box_midheight])

    field.setAsBackgroundMesh(f_min)

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
