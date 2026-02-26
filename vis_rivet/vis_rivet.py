import gmsh
import dolfinx.io.gmsh
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import dolfinx.nls.petsc
from dolfinx.io import VTXWriter
from mpi4py import MPI
import ufl
import numpy as np
import petsc4py.PETSc as PETSc
import os

# =============================================================================
# 1. GÉOMÉTRIE ET MAILLAGE DU RIVET (Gmsh)
# =============================================================================
def create_rivet_mesh():
    gmsh.initialize()
    model = gmsh.model()
    model.add("Rivet_Titanic")
    factory = model.occ

    r_shank = 0.0165 
    L_shank = 0.05   
    r_head = 0.025   

    cylinder = factory.addCylinder(0, 0, 0, 0, 0, L_shank, r_shank)
    sphere = factory.addSphere(0, 0, L_shank, r_head)
    
    # Boîte pour trancher le bas de la sphère
    cut_box = factory.addBox(-0.05, -0.05, -0.05, 0.1, 0.1, L_shank + 0.05)
    head, _ = factory.cut([(3, sphere)], [(3, cut_box)])

    rivet, _ = factory.fuse([(3, cylinder)], head)
    factory.synchronize()

    volumes = model.getEntities(3)
    model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1, name="Volume_Rivet")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.001)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.004)

    model.mesh.generate(3)
    mdata = dolfinx.io.gmsh.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return mdata

# =============================================================================
# 2. PARAMÈTRES PHYSIQUES (Fer puddlé)
# =============================================================================
E = 190e9   
nu = 0.3    
Gc = 1.0e3  
l0 = 0.005  

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# =============================================================================
# 3. ESPACES DE FONCTIONS
# =============================================================================
mdata = create_rivet_mesh()
domain = mdata.mesh

# V = Déplacement (Continu), D = Endommagement (Continu)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
D = fem.functionspace(domain, ("Lagrange", 1))

# L'énergie historique H DOIT être constante par élément (DG 0) pour être exacte !
W = fem.functionspace(domain, ("DG", 0))

u = fem.Function(V, name="Deplacement")
d = fem.Function(D, name="Endommagement")
d_old = fem.Function(D)
H = fem.Function(W, name="Historique")

dx = ufl.Measure("dx", domain=domain)

# =============================================================================
# 4. CONDITIONS AUX LIMITES ET MARQUEURS
# =============================================================================
fdim = domain.topology.dim - 1

def bottom_face(x):
    return x[2] < 0.001

def underside_face(x):
    # Tolérance élargie pour être sûr de capter toute la surface sous la tête
    return np.logical_and(x[2] > 0.049, x[2] < 0.051)

bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom_face)
under_facets = mesh.locate_entities_boundary(domain, fdim, underside_face)

facets = np.concatenate([bottom_facets, under_facets])
values = np.concatenate([np.full_like(bottom_facets, 1, dtype=np.int32), 
                         np.full_like(under_facets, 2, dtype=np.int32)])
sort_idx = np.argsort(facets)
facet_tags = mesh.meshtags(domain, fdim, facets[sort_idx], values[sort_idx])

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

dofs_bottom = fem.locate_dofs_topological(V, fdim, bottom_facets)
bcs_u = [fem.dirichletbc(np.array([0., 0., 0.], dtype=default_scalar_type), dofs_bottom, V)]

# =============================================================================
# 5. FORMULATION VARIATIONNELLE (AT1)
# =============================================================================
k_ell = 1e-6 
def g(d):
    return (1 - d)**2 + k_ell

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma_undamaged(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

psi_elastic = 0.5 * ufl.inner(sigma_undamaged(u), epsilon(u))

# --- A. Solver Déplacement ---
u_trial = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

T_mag = fem.Constant(domain, default_scalar_type(0.0))
traction_vector = ufl.as_vector([0.0, 0.0, T_mag])

a_u = g(d) * ufl.inner(sigma_undamaged(u_trial), epsilon(v)) * dx
L_u = ufl.inner(traction_vector, v) * ds(2) 

problem_u = LinearProblem(a_u, L_u, bcs=bcs_u, 
                          petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                          petsc_options_prefix="elasticity_")

# --- B. Solver Phase Field (AT1) ---
d_trial = ufl.TrialFunction(D)
w = ufl.TestFunction(D)

# TRUC MATHÉMATIQUE MAJEUR : On coupe les sources négatives pour empêcher le solveur de fausser d
source_term = ufl.max_value(2.0 * H - (3.0 * Gc / (8.0 * l0)), 0.0)

# On ajoute + 1e-8 * d_trial * w pour garantir que la matrice ne soit jamais singulière au Step 0
a_d = ( (3.0 * Gc * l0 / 4.0) * ufl.inner(ufl.grad(d_trial), ufl.grad(w)) 
      + (2.0 * H + 1e-8) * d_trial * w ) * dx

L_d = source_term * w * dx

problem_d = LinearProblem(a_d, L_d, bcs=[], 
                          petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                          petsc_options_prefix="phasefield_")

# L'énergie est calculée proprement au centre de chaque élément (DG0)
psi_expr = fem.Expression(psi_elastic, W.element.interpolation_points)
psi_func = fem.Function(W, name="Energie_Elastique")

# =============================================================================
# 6. BOUCLE TEMPORELLE (Force Imposée)
# =============================================================================
# On récupère le dossier exact où se trouve ce script Python
script_dir = os.path.dirname(os.path.abspath(__file__))
# On crée le chemin complet vers le dossier .bp dans ce même répertoire
export_path = os.path.join(script_dir, "rivet_titanic_AT1.bp")

vtx = VTXWriter(domain.comm, export_path, [u, d], engine="BP4")

print(f"Démarrage de la simulation dans {export_path}...")
vtx.write(0.0)

num_steps = 500
max_traction = 300e6 
tractions = np.linspace(0, max_traction, num_steps)

for step, t_val in enumerate(tractions[1:], 1):
    T_mag.value = t_val
    
    max_iter = 10
    for i in range(max_iter):
        d_prev_iter = d.x.array.copy()

        # 1. Calcul du déplacement (u)
        u_new = problem_u.solve()
        u.x.array[:] = u_new.x.array[:]
        
        # 2. Mise à jour de l'historique sur l'espace DG0 !
        psi_func.interpolate(psi_expr)
        H.x.array[:] = np.maximum(H.x.array, psi_func.x.array)
        
        # 3. Calcul de l'endommagement (d)
        d_new = problem_d.solve()
        d.x.array[:] = d_new.x.array[:]
        
        # 4. Irréversibilité et bornes vitales
        d.x.array[:] = np.maximum(d.x.array, d_old.x.array)
        d.x.array[:] = np.clip(d.x.array, 0.0, 1.0)
        
        error_d = np.linalg.norm(d.x.array - d_prev_iter)
        if error_d < 1e-4:
            break

    d_old.x.array[:] = d.x.array
    vtx.write(float(step))
    
    max_d = np.max(d.x.array)
    max_u = np.max(np.abs(u.x.array))
    
    # On affiche l'étirement (Max U) et l'endommagement (Max Damage) !
    print(f"Step {step}/{num_steps}: Force = {t_val/1e6:.1f} MPa | Max U = {max_u*1000:.3f} mm | Max Damage = {max_d:.4f} | Iters = {i+1}")
    
    if max_d > 0.99:
        print(f" Rupture totale détectée sous une contrainte de {t_val/1e6:.1f} MPa !")
        break

vtx.close()
print("Terminé. Ouvrez le dossier .bp dans ParaView.")