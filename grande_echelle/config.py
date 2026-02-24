from dataclasses import dataclass, field


@dataclass
class SimulationConfig:
    mesh_stem: str = "mesh/coque"
    results_root: str = "results"
    case_name: str = "titanic_1912_impact_segment"
    shell_cell_tag: int = 1
    rivet_cell_tag: int = 2
    shell_thickness: float = 2.5e-2
    shell_young_modulus: float = 210e9
    shell_poisson_ratio: float = 0.3
    rivet_thickness: float = 2.5e-2
    rivet_young_modulus: float = 190e9
    rivet_poisson_ratio: float = 0.3
    shell_yield_strength_pa: float = 270e6
    shell_ultimate_strength_pa: float = 431e6
    steel_sulfur_mass_fraction: float = 0.00065
    sigma: float = 3.0
    pressure_peak: float = 5e6
    iceberg_loading: str = "dirichlet_displacement"
    iceberg_disp_peak: float = 5e-2
    iceberg_disp_sign: float = 1.0
    iceberg_patch_radius_factor: float = 3.0
    t_final: float = 1.0
    num_steps: int = 20
    left_facet_tag: int = 1
    right_facet_tag: int = 2
    bottom_facet_tag: int = 3
    top_facet_tag: int = 4
    clamp_all_edges: bool = True
    clamp_rotations: bool = True
    y_mid_factor: float = 0.9
    z_mid_factor: float = 0.2
    phase_field_preset_file: str = "coupling/phase_field_selected_preset.json"
    enable_global_phase_field: bool = True
    phase_field_use_selected_preset: bool = True
    phase_field_gc_j_m2: float = 7000.0
    phase_field_l0_m: float = 0.4
    phase_field_residual_stiffness: float = 1e-6
    petsc_options: dict[str, str] = field(
        default_factory=lambda: {"ksp_type": "preonly", "pc_type": "lu"}
    )


DEFAULT_CONFIG = SimulationConfig()
