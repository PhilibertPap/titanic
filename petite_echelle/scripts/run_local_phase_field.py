from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petite_echelle.local_phase_field import LocalPhaseFieldRunConfig, run_local_phase_field


def main():
    config = LocalPhaseFieldRunConfig()
    out_dir = run_local_phase_field(config)
    print(f"Phase-field results written to: {out_dir}")


if __name__ == "__main__":
    main()

