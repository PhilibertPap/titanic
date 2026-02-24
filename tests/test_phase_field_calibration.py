from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from petite_echelle.phase_field_config import PhaseFieldCalibrationConfig


def _load_calibration_script_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "petite_echelle"
        / "scripts"
        / "calibrate_phase_field.py"
    )
    spec = importlib.util.spec_from_file_location("calibrate_phase_field_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_candidates_grid_shape_and_ranges():
    module = _load_calibration_script_module()
    cfg = PhaseFieldCalibrationConfig(
        num_kic_samples=3,
        num_l0_samples=4,
        element_size_m=0.2,
        l0_over_h_min=4.0,
        l0_over_h_max=7.0,
    )

    candidates = module.generate_candidates(cfg)

    assert len(candidates) == 12
    assert min(row["l0_over_h"] for row in candidates) == 4.0
    assert max(row["l0_over_h"] for row in candidates) == 7.0
    assert all(row["Gc_J_m2"] > 0.0 for row in candidates)
    assert all(row["l0_m"] > 0.0 for row in candidates)


def test_write_outputs_creates_metadata_candidates_and_baseline(tmp_path, monkeypatch):
    module = _load_calibration_script_module()
    cfg = PhaseFieldCalibrationConfig(
        case_name="unit_test_calibration",
        num_kic_samples=2,
        num_l0_samples=2,
        element_size_m=0.25,
    )
    candidates = module.generate_candidates(cfg)

    monkeypatch.chdir(tmp_path)
    module.write_outputs(candidates, cfg)

    out_dir = tmp_path / "petite_echelle" / "results" / cfg.case_name
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "candidates.csv").exists()
    assert (out_dir / "baseline.json").exists()

    baseline = json.loads((out_dir / "baseline.json").read_text(encoding="utf-8"))
    assert baseline in candidates
    assert {"Gc_J_m2", "l0_m", "l0_over_h", "recommended_gc_band"} <= baseline.keys()
