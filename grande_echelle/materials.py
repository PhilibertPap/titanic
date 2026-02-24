from dolfinx import fem


def _make_dg0_scalar_function(domain, name: str, value: float):
    V0 = fem.functionspace(domain, ("DG", 0))
    field = fem.Function(V0, name=name)
    field.x.array[:] = value
    return field


def _set_cells_value(field, cells, value: float):
    if cells is None or len(cells) == 0:
        return
    field.x.array[cells] = value


def build_material_fields(domain, cell_tags, cfg):
    """
    Build piecewise material fields from cell tags.
    - cfg.shell_cell_tag: base hull material
    - cfg.rivet_cell_tag: rivet-line material
    """
    E = _make_dg0_scalar_function(domain, "YoungModulus", cfg.shell_young_modulus)
    nu = _make_dg0_scalar_function(domain, "PoissonRatio", cfg.shell_poisson_ratio)
    thick = _make_dg0_scalar_function(domain, "Thickness", cfg.shell_thickness)

    if cell_tags is not None:
        rivet_cells = cell_tags.find(cfg.rivet_cell_tag)
        _set_cells_value(E, rivet_cells, cfg.rivet_young_modulus)
        _set_cells_value(nu, rivet_cells, cfg.rivet_poisson_ratio)
        _set_cells_value(thick, rivet_cells, cfg.rivet_thickness)

    return E, nu, thick