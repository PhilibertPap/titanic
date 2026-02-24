from pathlib import Path
import csv
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_float(row: dict, key: str) -> float:
    return float(row[key])


def main():
    summary_file = Path("petite_echelle/results/local_phase_field_sweep/summary.csv")
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Missing sweep summary: {summary_file}. Run sweep_local_phase_field.py first."
        )

    with summary_file.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Selection policy: target moderate-high damage extent without taking the most extreme case.
    target_frac = 0.035
    selected = min(
        rows,
        key=lambda r: (
            abs(_to_float(r, "final_frac_damage_ge_095") - target_frac),
            -_to_float(r, "Gc_J_m2"),
            abs(_to_float(r, "l0_over_h") - 6.0),
        ),
    )

    preset = {
        "selection_policy": {
            "target_frac_damage_ge_095": target_frac,
            "tie_break": ["higher_Gc", "l0_over_h_closest_to_6"],
        },
        "selected_case": selected["case_name"],
        "Gc_J_m2": _to_float(selected, "Gc_J_m2"),
        "l0_m": _to_float(selected, "l0_m"),
        "l0_over_h": _to_float(selected, "l0_over_h"),
        "final_max_damage": _to_float(selected, "final_max_damage"),
        "final_mean_damage": _to_float(selected, "final_mean_damage"),
        "final_frac_damage_ge_095": _to_float(selected, "final_frac_damage_ge_095"),
        "source_summary": str(summary_file),
    }

    out_local = summary_file.parent / "selected_preset.json"
    out_local.write_text(json.dumps(preset, indent=2), encoding="utf-8")

    out_global = Path("grande_echelle/coupling/phase_field_selected_preset.json")
    out_global.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_local, out_global)

    print(f"Selected preset: {preset['selected_case']}")
    print(f"Wrote: {out_local}")
    print(f"Linked to grande_echelle: {out_global}")


if __name__ == "__main__":
    main()

