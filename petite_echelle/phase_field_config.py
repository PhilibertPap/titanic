from dataclasses import dataclass, asdict


@dataclass
class PhaseFieldCalibrationConfig:
    case_name: str = "titanic_local_calibration"
    # Material anchors (from Titanic metallurgy studies)
    young_modulus_pa: float = 210e9
    poisson_ratio: float = 0.30
    yield_strength_pa: float = 270e6
    ultimate_strength_pa: float = 431e6
    # Fracture calibration ranges
    kic_min_mpa_sqrt_m: float = 30.0
    kic_max_mpa_sqrt_m: float = 50.0
    num_kic_samples: int = 9
    # Mesh and regularization calibration
    element_size_m: float = 0.25
    l0_over_h_min: float = 4.0
    l0_over_h_max: float = 8.0
    num_l0_samples: int = 9
    # Optional narrowed "recommended" interval
    gc_recommended_min_j_m2: float = 6000.0
    gc_recommended_max_j_m2: float = 8000.0

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PHASE_FIELD_CONFIG = PhaseFieldCalibrationConfig()

