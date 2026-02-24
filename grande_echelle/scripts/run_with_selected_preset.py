from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grande_echelle.config import DEFAULT_CONFIG
from grande_echelle.main import run


def main():
    cfg = DEFAULT_CONFIG
    preset = Path(cfg.phase_field_preset_file)
    if not preset.is_absolute():
        candidates = [Path.cwd() / preset, ROOT / "grande_echelle" / preset]
    else:
        candidates = [preset]
    selected = next((path for path in candidates if path.exists()), None)
    if selected is None:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Selected phase-field preset is missing. "
            f"Tried: {tried}. Run petite_echelle/scripts/select_phase_field_preset.py first."
        )
    print(f"Using selected preset: {selected}")
    run(cfg)


if __name__ == "__main__":
    main()
