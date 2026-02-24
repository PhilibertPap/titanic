from pathlib import Path
import json
import csv
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from petite_echelle.local_phase_field import LocalPhaseFieldRunConfig, run_local_phase_field


def _load_baseline(path: Path) -> tuple[float, float]:
    if not path.exists():
        return 7000.0, 1.5
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["Gc_J_m2"]), float(data["l0_m"])


def _last_monitor_row(monitor_path: Path) -> dict:
    with monitor_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def main():
    baseline_file = Path("petite_echelle/results/titanic_local_calibration/baseline.json")
    gc0, l00 = _load_baseline(baseline_file)
    h = 8.0 / 120.0

    gc_factors = [0.8, 1.0, 1.2]
    l0_over_h_values = [5.0, 6.0, 7.0]

    out_root = Path("petite_echelle/results/local_phase_field_sweep")
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for gf in gc_factors:
        for ratio in l0_over_h_values:
            gc = gc0 * gf
            l0 = ratio * h
            case_name = f"sweep_gc{int(round(gc))}_l0h{ratio:.1f}".replace(".", "p")
            cfg = LocalPhaseFieldRunConfig(
                case_name=case_name,
                baseline_file=str(baseline_file),
                uy_max_m=0.03,
            )

            # Override baseline through temporary json
            tmp_baseline = out_root / f"{case_name}_baseline.json"
            tmp_baseline.write_text(
                json.dumps({"Gc_J_m2": gc, "l0_m": l0}, indent=2), encoding="utf-8"
            )
            cfg.baseline_file = str(tmp_baseline)

            print(f"Running {case_name} (Gc={gc:.1f}, l0={l0:.4f}, l0/h={ratio:.1f})")
            result_dir = run_local_phase_field(cfg)
            last = _last_monitor_row(result_dir / "monitor.csv")
            summary_rows.append(
                {
                    "case_name": case_name,
                    "Gc_J_m2": gc,
                    "l0_m": l0,
                    "l0_over_h": ratio,
                    "final_max_damage": float(last["max_damage"]),
                    "final_mean_damage": float(last["mean_damage"]),
                    "final_frac_damage_ge_095": float(last["frac_damage_ge_095"]),
                }
            )

    summary_file = out_root / "summary.csv"
    with summary_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "Gc_J_m2",
                "l0_m",
                "l0_over_h",
                "final_max_damage",
                "final_mean_damage",
                "final_frac_damage_ge_095",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Sweep complete. Summary: {summary_file}")


if __name__ == "__main__":
    main()

