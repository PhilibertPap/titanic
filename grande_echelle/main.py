import numpy as np
import json
from pathlib import Path

try:
    from .config import DEFAULT_CONFIG
    from .fem_io.mesh_io import (
        prepare_results_layout,
        read_mesh,
        write_local_frame,
        write_material_regions,
        write_run_metadata,
    )
    from .model.shell import build_shell_model
    from .solvers.quasi_static import run_quasi_static
except ImportError:  # pragma: no cover - script execution fallback
    from config import DEFAULT_CONFIG
    from fem_io.mesh_io import (
        prepare_results_layout,
        read_mesh,
        write_local_frame,
        write_material_regions,
        write_run_metadata,
    )
    from model.shell import build_shell_model
    from solvers.quasi_static import run_quasi_static


def run(cfg=None):
    if cfg is None:
        cfg = DEFAULT_CONFIG
    meshdata = read_mesh(cfg.mesh_stem)
    domain = meshdata.mesh
    cell_tags = meshdata.cell_tags
    facets = meshdata.facet_tags

    print("cell_tags is None:", cell_tags is None)
    if cell_tags is not None:
        print("Cell tag values:", np.unique(cell_tags.values))

    print("facet_tags is None:", facets is None)
    if facets is not None:
        print("Facet tag values:", np.unique(facets.values))

    print(f"Geometrical dimension = {domain.geometry.dim}")
    print(f"Topological dimension = {domain.topology.dim}")

    model = build_shell_model(domain, cell_tags, facets, cfg)
    output_layout = prepare_results_layout(cfg.results_root, cfg.case_name)
    phase_field_preset = None
    preset_candidates = []
    preset_cfg_path = Path(cfg.phase_field_preset_file)
    if preset_cfg_path.is_absolute():
        preset_candidates.append(preset_cfg_path)
    else:
        preset_candidates.append(Path.cwd() / preset_cfg_path)
        preset_candidates.append(Path(__file__).resolve().parent / preset_cfg_path)
    preset_path = next((p for p in preset_candidates if p.exists()), None)
    if preset_path is not None:
        phase_field_preset = json.loads(preset_path.read_text(encoding="utf-8"))
        print(f"Using phase-field preset: {preset_path}")
    else:
        print(f"Phase-field preset not found. Tried: {preset_candidates}")

    write_local_frame(output_layout["local_frame_file"], model.e1, model.e2, model.e3)
    write_material_regions(
        output_layout["local_frame_dir"] / "material_regions.pvd",
        domain,
        cell_tags,
        cfg.shell_cell_tag,
    )
    write_run_metadata(output_layout["metadata_file"], cfg, phase_field_preset)
    run_quasi_static(model, cfg, output_layout, phase_field_preset=phase_field_preset)


if __name__ == "__main__":
    run()
