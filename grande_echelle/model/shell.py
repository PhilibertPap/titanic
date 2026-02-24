from dataclasses import dataclass

import basix
import ufl
from dolfinx import fem

try:
    from ..materials import build_material_fields
except ImportError:  # pragma: no cover - script execution fallback
    from materials import build_material_fields


def _normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))


def _local_frame(domain):
    t = ufl.Jacobian(domain)
    if domain.geometry.dim == 2:
        t1 = ufl.as_vector([t[0, 0], t[1, 0], 0])
        t2 = ufl.as_vector([t[0, 1], t[1, 1], 0])
    else:
        t1 = ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]])
        t2 = ufl.as_vector([t[0, 1], t[1, 1], t[2, 1]])

    e3 = _normalize(ufl.cross(t1, t2))
    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1 = ufl.cross(ey, e3)
    norm_e1 = ufl.sqrt(ufl.dot(e1, e1))
    e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, _normalize(e1))
    e2 = _normalize(ufl.cross(e3, e1))
    return e1, e2, e3


def _vstack(vectors):
    return ufl.as_matrix([[v[i] for i in range(len(v))] for v in vectors])


def _hstack(vectors):
    return _vstack(vectors).T


@dataclass
class ShellModel:
    domain: any
    facets: any
    gdim: int
    tdim: int
    V: any
    Vu: any
    Vtheta: any
    v: any
    u_test: any
    a: any
    bcs: list
    e1: any
    e2: any
    e3: any
    E_field: any
    nu_field: any
    thick_field: any


def build_shell_model(domain, cell_tags, facets, cfg) -> ShellModel:
    gdim = domain.geometry.dim
    tdim = domain.topology.dim

    E, nu, thick = build_material_fields(domain, cell_tags, cfg)
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)

    VT = fem.functionspace(domain, ("DG", 0, (gdim,)))
    V0, _ = VT.sub(0).collapse()
    frame = _local_frame(domain)
    basis_vectors = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(gdim)]
    e1, e2, e3 = basis_vectors
    for i in range(gdim):
        e_exp = fem.Expression(frame[i], V0.element.interpolation_points)
        basis_vectors[i].interpolate(e_exp)

    Ue = basix.ufl.element("P", domain.basix_cell(), 2, shape=(gdim,))
    Te = basix.ufl.element("CR", domain.basix_cell(), 1, shape=(gdim,))
    V = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))

    v = fem.Function(V)
    u, theta = ufl.split(v)
    v_test = ufl.TestFunction(V)
    u_test, theta_test = ufl.split(v_test)
    dv = ufl.TrialFunction(V)

    P_plane = _hstack([e1, e2])

    def t_grad(field):
        grad_field = ufl.grad(field)
        return ufl.dot(grad_field, P_plane)

    t_gu = ufl.dot(P_plane.T, t_grad(u))
    eps = ufl.sym(t_gu)
    beta = ufl.cross(e3, theta)
    kappa = ufl.sym(ufl.dot(P_plane.T, t_grad(beta)))
    gamma = t_grad(ufl.dot(u, e3)) - ufl.dot(P_plane.T, beta)

    eps_test = ufl.derivative(eps, v, v_test)
    kappa_test = ufl.derivative(kappa, v, v_test)
    gamma_test = ufl.derivative(gamma, v, v_test)

    def plane_stress_elasticity(strain):
        return lmbda_ps * ufl.tr(strain) * ufl.Identity(tdim) + 2 * mu * strain

    N = thick * plane_stress_elasticity(eps)
    M = thick**3 / 12 * plane_stress_elasticity(kappa)
    Q = mu * thick * gamma

    drilling_strain = (t_gu[0, 1] - t_gu[1, 0]) / 2 + ufl.dot(theta, e3)
    drilling_strain_test = ufl.replace(drilling_strain, {v: v_test})
    h_mesh = ufl.CellDiameter(domain)
    drilling_stiffness = E * thick**3 / h_mesh**2
    drilling_stress = drilling_stiffness * drilling_strain

    Vu, _ = V.sub(0).collapse()
    Vtheta, _ = V.sub(1).collapse()

    edge_tags = [cfg.left_facet_tag, cfg.right_facet_tag]
    if cfg.clamp_all_edges:
        edge_tags.extend([cfg.bottom_facet_tag, cfg.top_facet_tag])
    edge_tags = list(dict.fromkeys(edge_tags))

    uD = fem.Function(Vu)
    thetaD = fem.Function(Vtheta)
    bcs = []
    for facet_tag in edge_tags:
        disp_dofs = fem.locate_dofs_topological((V.sub(0), Vu), 1, facets.find(facet_tag))
        bcs.append(fem.dirichletbc(uD, disp_dofs, V.sub(0)))
        if cfg.clamp_rotations:
            rot_dofs = fem.locate_dofs_topological((V.sub(1), Vtheta), 1, facets.find(facet_tag))
            bcs.append(fem.dirichletbc(thetaD, rot_dofs, V.sub(1)))

    Wdef = (
        ufl.inner(N, eps_test)
        + ufl.inner(M, kappa_test)
        + ufl.dot(Q, gamma_test)
        + drilling_stress * drilling_strain_test
    ) * ufl.dx
    a = ufl.derivative(Wdef, v, dv)

    return ShellModel(
        domain=domain,
        facets=facets,
        gdim=gdim,
        tdim=tdim,
        V=V,
        Vu=Vu,
        Vtheta=Vtheta,
        v=v,
        u_test=u_test,
        a=a,
        bcs=bcs,
        e1=e1,
        e2=e2,
        e3=e3,
        E_field=E,
        nu_field=nu,
        thick_field=thick,
    )
