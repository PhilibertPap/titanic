from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SuiviRun:
    source: Path
    step: np.ndarray
    time: np.ndarray
    max_u: np.ndarray
    max_damage: np.ndarray
    mean_damage: np.ndarray
    frac_d95: np.ndarray
    temps_pas: np.ndarray
    temps_meca: np.ndarray
    temps_pf: np.ndarray


def charger_monitor_csv(path: Path) -> SuiviRun:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"Fichier vide: {path}")

    def col(name: str, default: float = 0.0) -> np.ndarray:
        out = []
        for r in rows:
            raw = r.get(name, "")
            out.append(float(raw) if raw != "" else default)
        return np.asarray(out, dtype=float)

    return SuiviRun(
        source=path,
        step=col("step"),
        time=col("time"),
        max_u=col("max_u_inf"),
        max_damage=col("max_damage"),
        mean_damage=col("mean_damage"),
        frac_d95=col("frac_damage_ge_095"),
        temps_pas=col("temps_pas_s"),
        temps_meca=col("temps_meca_s"),
        temps_pf=col("temps_phase_field_s"),
    )


def _sauver(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tracer_evolution_dommage(run: SuiviRun, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(run.time, run.max_damage, label="max_damage", linewidth=2.2)
    ax.plot(run.time, run.mean_damage, label="mean_damage", linewidth=2.0)
    ax.plot(run.time, run.frac_d95, label="frac_damage_ge_095", linewidth=2.0)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Dommage")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Evolution de l'endommagement")
    _sauver(fig, outdir / "evolution_endommagement.png")


def tracer_deplacement_vs_dommage(run: SuiviRun, outdir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(run.time, run.max_u, color="tab:blue", linewidth=2.2, label="max_u_inf")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel("Deplacement max [m]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(run.time, run.max_damage, color="tab:red", linewidth=2.2, label="max_damage")
    ax2.set_ylabel("Dommage max", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(-0.02, 1.02)
    ax1.set_title("Couplage deformation / endommagement")
    _sauver(fig, outdir / "deplacement_vs_dommage.png")


def tracer_cout_calcul(run: SuiviRun, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(run.step, run.temps_pas, label="temps_pas_s", linewidth=2.2)
    ax.plot(run.step, run.temps_meca, label="temps_meca_s", linewidth=2.0)
    ax.plot(run.step, run.temps_pf, label="temps_phase_field_s", linewidth=2.0)
    ax.set_xlabel("Pas")
    ax.set_ylabel("Temps [s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Cout de calcul par pas")
    _sauver(fig, outdir / "cout_calcul_par_pas.png")


def tracer_portrait_avarie(run: SuiviRun, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(run.max_u, run.max_damage, c=run.time, cmap="viridis", s=24)
    ax.set_xlabel("max_u_inf [m]")
    ax.set_ylabel("max_damage")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title("Portrait avarie: deformation vs dommage")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Temps")
    _sauver(fig, outdir / "portrait_avarie.png")


def tracer_comparaison(ref: SuiviRun, cible: SuiviRun, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(ref.time, ref.max_damage, linewidth=2.1, label=f"ref: {ref.source.parent.parent.name}")
    axes[0].plot(cible.time, cible.max_damage, linewidth=2.1, label=f"cible: {cible.source.parent.parent.name}")
    axes[0].set_title("Comparaison max_damage")
    axes[0].set_xlabel("Temps")
    axes[0].set_ylabel("max_damage")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ref.time, ref.max_u, linewidth=2.1, label=f"ref: {ref.source.parent.parent.name}")
    axes[1].plot(cible.time, cible.max_u, linewidth=2.1, label=f"cible: {cible.source.parent.parent.name}")
    axes[1].set_title("Comparaison max_u_inf")
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel("max_u_inf [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    _sauver(fig, outdir / "comparaison_cas.png")


def ecrire_resume(run: SuiviRun, outdir: Path, ref: SuiviRun | None = None) -> None:
    lines = []
    lines.append(f"source={run.source}")
    lines.append(f"n_steps={len(run.step)}")
    lines.append(f"max_damage_final={run.max_damage[-1]:.6f}")
    lines.append(f"mean_damage_final={run.mean_damage[-1]:.6f}")
    lines.append(f"frac_d95_final={run.frac_d95[-1]:.6f}")
    lines.append(f"max_u_final={run.max_u[-1]:.6e} m")
    lines.append(f"temps_total_pas={run.temps_pas.sum():.3f} s")
    lines.append(f"temps_total_meca={run.temps_meca.sum():.3f} s")
    lines.append(f"temps_total_phase_field={run.temps_pf.sum():.3f} s")

    if ref is not None:
        lines.append("")
        lines.append(f"source_ref={ref.source}")
        lines.append(f"delta_max_damage_final={run.max_damage[-1] - ref.max_damage[-1]:+.6f}")
        lines.append(f"delta_max_u_final={run.max_u[-1] - ref.max_u[-1]:+.6e} m")
        lines.append(f"delta_temps_total_pas={run.temps_pas.sum() - ref.temps_pas.sum():+.3f} s")

    (outdir / "resume_avarie.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trouver_monitor_le_plus_recent(results_root: Path) -> Path:
    candidats = list(results_root.glob("*/quasi_static/monitor.csv"))
    if not candidats:
        raise FileNotFoundError(f"Aucun monitor.csv trouve dans: {results_root}")
    return max(candidats, key=lambda p: p.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser(description="Genere des graphes pour analyser l'avarie a partir de monitor.csv")
    parser.add_argument("--monitor", type=str, default=None, help="Chemin vers monitor.csv du cas cible")
    parser.add_argument("--monitor-ref", type=str, default=None, help="Chemin vers monitor.csv de reference")
    parser.add_argument("--results-root", type=str, default="results", help="Racine des resultats (utilisee si --monitor absent)")
    parser.add_argument("--outdir", type=str, default=None, help="Dossier de sortie des graphes")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.monitor is None:
        monitor = _trouver_monitor_le_plus_recent(Path(args.results_root))
    else:
        monitor = Path(args.monitor)
    if not monitor.exists():
        raise FileNotFoundError(f"Fichier non trouve: {monitor}")

    run = charger_monitor_csv(monitor)
    run_ref = None
    if args.monitor_ref:
        ref_path = Path(args.monitor_ref)
        if not ref_path.exists():
            raise FileNotFoundError(f"Fichier de reference non trouve: {ref_path}")
        run_ref = charger_monitor_csv(ref_path)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = monitor.parent.parent / "analyse_avarie"
    outdir.mkdir(parents=True, exist_ok=True)

    tracer_evolution_dommage(run, outdir)
    tracer_deplacement_vs_dommage(run, outdir)
    tracer_cout_calcul(run, outdir)
    tracer_portrait_avarie(run, outdir)
    if run_ref is not None:
        tracer_comparaison(run_ref, run, outdir)

    ecrire_resume(run, outdir, ref=run_ref)
    print(f"Graphes ecrits dans: {outdir}")


if __name__ == "__main__":
    main()
