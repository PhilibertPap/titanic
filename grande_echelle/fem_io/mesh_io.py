from pathlib import Path
import json

from mpi4py import MPI
from dolfinx import io, fem


def read_mesh(mesh_stem: str, comm=MPI.COMM_WORLD):
    candidate = Path(mesh_stem)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate.with_suffix(".msh"))
    else:
        candidates.append((Path.cwd() / candidate).with_suffix(".msh"))
        project_mesh = Path(__file__).resolve().parents[1] / candidate
        candidates.append(project_mesh.with_suffix(".msh"))
    mesh_path = next((p for p in candidates if p.exists()), None)
    if mesh_path is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Mesh file not found. Tried: {tried}")
    return io.gmsh.read_from_msh(str(mesh_path), comm)


def prepare_results_layout(results_root: str | Path, case_name: str) -> dict[str, Path]:
    base_dir = Path(results_root) / case_name
    local_frame_dir = base_dir / "local_frame"
    quasi_static_dir = base_dir / "quasi_static"

    local_frame_dir.mkdir(exist_ok=True, parents=True)
    quasi_static_dir.mkdir(exist_ok=True, parents=True)

    return {
        "base_dir": base_dir,
        "local_frame_dir": local_frame_dir,
        "quasi_static_dir": quasi_static_dir,
        "local_frame_file": local_frame_dir / "local_basis_vectors.pvd",
        "displacement_file": quasi_static_dir / "displacement.pvd",
        "rotation_file": quasi_static_dir / "rotation.pvd",
        "damage_file": quasi_static_dir / "damage.pvd",
        "monitor_file": quasi_static_dir / "monitor.csv",
        "metadata_file": base_dir / "run_metadata.json",
    }


def write_local_frame(local_frame_file: Path, e1, e2, e3) -> None:
    with io.VTKFile(MPI.COMM_WORLD, local_frame_file, "w") as vtk:
        vtk.write_function(e1, 0.0)
        vtk.write_function(e2, 0.0)
        vtk.write_function(e3, 0.0)


def write_material_regions(material_file: Path, domain, cell_tags, default_tag: int) -> None:
    V0 = fem.functionspace(domain, ("DG", 0))
    material_regions = fem.Function(V0, name="MaterialRegion")
    material_regions.x.array[:] = float(default_tag)
    if cell_tags is not None and len(cell_tags.indices) > 0:
        material_regions.x.array[cell_tags.indices] = cell_tags.values.astype(float)
    with io.VTKFile(MPI.COMM_WORLD, material_file, "w") as vtk:
        vtk.write_function(material_regions, 0.0)


def write_run_metadata(metadata_file: Path, cfg, phase_field_preset: dict | None = None) -> None:
    if MPI.COMM_WORLD.rank != 0:
        return
    metadata = {
        "mesh_stem": cfg.mesh_stem,
        "case_name": cfg.case_name,
        "num_steps": cfg.num_steps,
        "t_final": cfg.t_final,
        "sigma": cfg.sigma,
        "pressure_peak": cfg.pressure_peak,
        "iceberg_loading": cfg.iceberg_loading,
        "iceberg_disp_peak": cfg.iceberg_disp_peak,
        "iceberg_disp_sign": cfg.iceberg_disp_sign,
        "iceberg_center_y": getattr(cfg, "iceberg_center_y", None),
        "waterline_z": getattr(cfg, "waterline_z", None),
        "iceberg_depth_below_waterline": getattr(cfg, "iceberg_depth_below_waterline", None),
        "shell_cell_tag": cfg.shell_cell_tag,
        "rivet_cell_tag": cfg.rivet_cell_tag,
        "clamp_all_edges": cfg.clamp_all_edges,
        "clamp_rotations": cfg.clamp_rotations,
        "shell_thickness": cfg.shell_thickness,
        "rivet_thickness": cfg.rivet_thickness,
        "shell_young_modulus": cfg.shell_young_modulus,
        "rivet_young_modulus": cfg.rivet_young_modulus,
        "shell_yield_strength_pa": cfg.shell_yield_strength_pa,
        "shell_ultimate_strength_pa": cfg.shell_ultimate_strength_pa,
        "steel_sulfur_mass_fraction": cfg.steel_sulfur_mass_fraction,
        "phase_field_preset_file": cfg.phase_field_preset_file,
        "enable_global_phase_field": cfg.enable_global_phase_field,
        "phase_field_use_selected_preset": cfg.phase_field_use_selected_preset,
        "phase_field_gc_j_m2": cfg.phase_field_gc_j_m2,
        "phase_field_l0_m": cfg.phase_field_l0_m,
        "phase_field_residual_stiffness": cfg.phase_field_residual_stiffness,
        "phase_field_preset": phase_field_preset,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
