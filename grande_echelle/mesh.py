import gmsh

filename = "mesh/coque"
SHELL_CELL_TAG = 1
RIVET_CELL_TAG = 2

gmsh.initialize()
geom = gmsh.model.geo

# Segment coque "Titanic impact" (ordre de grandeur)
L = 100.0   # longueur de segment [m]
B = 14.1    # demi-largeur [m]
T = 10.5    # tirant d'eau [m]
a = 0.25    # courbure longitudinale
c = 0.08    # courbure verticale (reduite)
H = 0.02

# Raffine en u pour eviter des bandes rivets trop larges.
Nu = 320
Nv = 48
rivet_u_centers = [0.2, 0.4, 0.6, 0.8]
# Rivet Titanic: diametre nominal ~0.875 in (22.2 mm), pas longitudinal ~3 in (76.2 mm).
# La bande materiau est homogenisee. Avec Nu=320, une colonne vaut L/Nu=0.3125 m.
rivet_band_half_width_u = 0.0018

def hull_xyz(u, v):
    x = L * u
    z_keel = -T + a * (u - 0.5)**2
    y = B * v
    # Partie immergée + franc-bord
    z = z_keel + c * v**2 + H * v**4
    return x, y, z

lcar = 0.15
pts = [[0]*(Nv+1) for _ in range(Nu+1)]

for i in range(Nu+1):
    u = i / Nu
    for j in range(Nv+1):
        v = -1 + 2 * j / Nv
        x, y, z = hull_xyz(u, v)
        pts[i][j] = geom.addPoint(x, y, z, lcar)

# Lignes longitudinales (u = const)
long_lines = []
for i in range(Nu+1):
    row = []
    for j in range(Nv):
        l = geom.addLine(pts[i][j], pts[i][j+1])
        row.append(l)
    long_lines.append(row)

# Lignes de niveau (v = const)
vert_lines = []
for j in range(Nv+1):
    col = []
    for i in range(Nu):
        l = geom.addLine(pts[i][j], pts[i+1][j])
        col.append(l)
    vert_lines.append(col)

# Surfaces quadrilatères
shell_surfaces = []
rivet_surfaces = []
for i in range(Nu):
    for j in range(Nv):
        # les 4 lignes autour de la "cellule" (i,j)
        l1 = long_lines[i][j]       # u = i,   v: j -> j+1
        l3 = long_lines[i+1][j]     # u = i+1, v: j -> j+1
        l2 = vert_lines[j+1][i]     # v = j+1, u: i -> i+1
        l4 = vert_lines[j][i]       # v = j,   u: i -> i+1

        cl = geom.addCurveLoop([l1, l2, -l3, -l4])
        s  = geom.addSurfaceFilling([cl])
        u_center = (i + 0.5) / Nu
        is_rivet = any(abs(u_center - u0) <= rivet_band_half_width_u for u0 in rivet_u_centers)
        if is_rivet:
            rivet_surfaces.append(s)
        else:
            shell_surfaces.append(s)

geom.synchronize()

# --- Définition des lignes de bord pour les conditions limites ---

# Bord gauche (u = 0) : toutes les lignes long_lines[0][j] pour j=0..Nv-1
left_lines = [long_lines[0][j] for j in range(Nv)]

# Bord droit (u = 1) : toutes les lignes long_lines[Nu][j]
right_lines = [long_lines[Nu][j] for j in range(Nv)]

# (optionnel) Bord bas (v = 0) et haut (v = Nv) si tu en as besoin
bottom_lines = [vert_lines[0][i]  for i in range(Nu)]
top_lines    = [vert_lines[Nv][i] for i in range(Nu)]

# Coque: matériau standard et zones rivetées
gmsh.model.addPhysicalGroup(2, shell_surfaces, SHELL_CELL_TAG)
gmsh.model.setPhysicalName(2, SHELL_CELL_TAG, "HullPlate")
if rivet_surfaces:
    gmsh.model.addPhysicalGroup(2, rivet_surfaces, RIVET_CELL_TAG)
    gmsh.model.setPhysicalName(2, RIVET_CELL_TAG, "HullRivets")

# Bords pour BC :
gmsh.model.addPhysicalGroup(1, left_lines, 1)
gmsh.model.setPhysicalName(1, 1, "Left")
gmsh.model.addPhysicalGroup(1, right_lines, 2)
gmsh.model.setPhysicalName(1, 2, "Right")
gmsh.model.addPhysicalGroup(1, bottom_lines, 3)
gmsh.model.setPhysicalName(1, 3, "Bottom")
gmsh.model.addPhysicalGroup(1, top_lines, 4)
gmsh.model.setPhysicalName(1, 4, "Top")

for i in range(Nu+1):
    for j in range(Nv):
        geom.mesh.setTransfiniteCurve(long_lines[i][j], 2)
for j in range(Nv+1):
    for i in range(Nu):
        geom.mesh.setTransfiniteCurve(vert_lines[j][i], 2)

gmsh.model.mesh.generate(dim=2)
gmsh.write(filename + ".msh")
gmsh.finalize()
