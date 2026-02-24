import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petite_echelle.phase_field_config import DEFAULT_PHASE_FIELD_CONFIG


def _effective_modulus(young_modulus_pa: float, poisson_ratio: float) -> float:
    # Plane strain assumption for brittle-fracture estimate
    return young_modulus_pa / (1.0 - poisson_ratio**2)


def _gc_from_kic(kic_mpa_sqrt_m: float, effective_modulus_pa: float) -> float:
    kic_pa_sqrt_m = kic_mpa_sqrt_m * 1e6
    return (kic_pa_sqrt_m**2) / effective_modulus_pa


def generate_candidates(config=DEFAULT_PHASE_FIELD_CONFIG) -> list[dict]:
    e_prime = _effective_modulus(config.young_modulus_pa, config.poisson_ratio)
    kic_values = np.linspace(
        config.kic_min_mpa_sqrt_m,
        config.kic_max_mpa_sqrt_m,
        config.num_kic_samples,
    )
    gc_values = [_gc_from_kic(kic, e_prime) for kic in kic_values]
    l0_over_h_values = np.linspace(
        config.l0_over_h_min, config.l0_over_h_max, config.num_l0_samples
    )

    candidates = []
    for gc in gc_values:
        for ratio in l0_over_h_values:
            l0 = ratio * config.element_size_m
            in_recommended_gc = (
                config.gc_recommended_min_j_m2 <= gc <= config.gc_recommended_max_j_m2
            )
            candidates.append(
                {
                    "Gc_J_m2": float(gc),
                    "l0_m": float(l0),
                    "l0_over_h": float(ratio),
                    "recommended_gc_band": bool(in_recommended_gc),
                }
            )
    return candidates


def write_outputs(candidates: list[dict], config=DEFAULT_PHASE_FIELD_CONFIG) -> None:
    out_dir = Path("petite_echelle/results") / config.case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": config.to_dict(),
        "notes": {
            "gc_formula": "Gc = KIc^2 / E' with E' = E/(1-nu^2)",
            "modeling_note": "l0 should typically satisfy l0/h in [4,8]",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with (out_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Gc_J_m2", "l0_m", "l0_over_h", "recommended_gc_band"]
        )
        writer.writeheader()
        writer.writerows(candidates)

    # Suggested baseline pair for first nonlinear run
    baseline = min(
        candidates,
        key=lambda row: abs(row["Gc_J_m2"] - 7000.0) + abs(row["l0_over_h"] - 6.0),
    )
    (out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def main():
    candidates = generate_candidates(DEFAULT_PHASE_FIELD_CONFIG)
    write_outputs(candidates, DEFAULT_PHASE_FIELD_CONFIG)
    print(f"Generated {len(candidates)} phase-field candidates.")


if __name__ == "__main__":
    main()
