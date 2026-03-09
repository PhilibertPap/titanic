from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SuiviRivet:
    name: str
    source: Path
    time: np.ndarray
    max_u: np.ndarray
    max_damage: np.ndarray
    mean_damage: np.ndarray
    frac_d95: np.ndarray
    config: dict[str, Any]


def _lire_monitor(path: Path) -> SuiviRivet:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"monitor.csv vide: {path}")

    def col(name: str) -> np.ndarray:
        return np.asarray([float(r.get(name, 0.0) or 0.0) for r in rows], dtype=float)

    config = {}
    metadata_path = path.parent.parent / "run_metadata.json"
    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            config = data.get("config", {})
        except Exception:
            config = {}

    return SuiviRivet(
        name=path.parent.parent.name,
        source=path,
        time=col("time"),
        max_u=col("max_u_inf"),
        max_damage=col("max_damage"),
        mean_damage=col("mean_damage"),
        frac_d95=col("frac_damage_ge_095"),
        config=config,
    )


def _avec_rivets(config: dict[str, Any], case_name: str) -> bool:
    if config.get("utiliser_bandes_rivets_z"):
        return True
    lower = case_name.lower()
    return "with_rivets" in lower or "avec_rivets" in lower


def _collecter_cas(results_root: Path, monitor_explicites: list[Path] | None = None) -> list[SuiviRivet]:
    if monitor_explicites:
        paths = [Path(p) for p in monitor_explicites]
    else:
        paths = []
        for d in sorted(results_root.iterdir()):
            if not d.is_dir():
                continue
            p = d / "quasi_static" / "monitor.csv"
            if p.exists():
                paths.append(p)

    cas = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"monitor introuvable: {p}")
        suivi = _lire_monitor(p)
        if _avec_rivets(suivi.config, suivi.name):
            cas.append(suivi)

    if not cas:
        raise RuntimeError(
            "Aucun cas avec rivets trouve. Utilisez --monitor avec un fichier monitor.csv de cas avec rivets."
        )
    cas.sort(key=lambda c: c.name)
    return cas


def _sauver(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _nettoyer_graphes(outdir: Path, *, include_max: bool = False, include_frac: bool = False, include_u: bool = False) -> None:
    legacy = [
        "evolution_dommage_with_rivets.png",
        "u_vs_damage_with_rivets.png",
    ]
    for nom in legacy:
        p = outdir / nom
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    if not include_max:
        for p in outdir.glob("evolution_max_damage_*.png"):
            try:
                p.unlink()
            except OSError:
                pass
    if not include_frac:
        for p in outdir.glob("evolution_frac_damage_*.png"):
            try:
                p.unlink()
            except OSError:
                pass
    if not include_u:
        for p in outdir.glob("u_vs_damage_*.png"):
            try:
                p.unlink()
            except OSError:
                pass


def _param_val_from_case(cas: SuiviRivet, param: str) -> float | None:
    if param == "max_u_final":
        return float(cas.max_u[-1]) if len(cas.max_u) else None
    if param == "mean_damage_final":
        return float(cas.mean_damage[-1]) if len(cas.mean_damage) else None
    if param == "max_damage_final":
        return float(cas.max_damage[-1]) if len(cas.max_damage) else None
    if param == "n_steps":
        return float(len(cas.time))
    valeur = cas.config.get(param, None)
    if isinstance(valeur, bool):
        return float(valeur)
    if isinstance(valeur, int):
        return float(valeur)
    if isinstance(valeur, float):
        return valeur
    if isinstance(valeur, str):
        try:
            return float(valeur)
        except ValueError:
            return None
    return None


def _param_peut_t_etre_trace(cases: list[SuiviRivet], param: str) -> bool:
    vals = [_param_val_from_case(c, param) for c in cases]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return False
    uniques = sorted({round(v, 12) for v in vals})
    return len(uniques) > 1


def _params_a_tracer_auto(cases: list[SuiviRivet], max_params: int = 4) -> list[str]:
    preferes = [
        "phase_field_gc_j_m2",
        "deplacement_pic_iceberg",
        "phase_field_l0_m",
        "iceberg_dx_max_par_pas_m",
        "temps_final",
        "nombre_pas",
        "deplacement_pic_iceberg",
        "max_u_final",
    ]

    trouves: list[str] = []
    for p in preferes:
        if _param_peut_t_etre_trace(cases, p):
            trouves.append(p)
        if len(trouves) >= max_params:
            break

    if not trouves and len(cases) > 1 and len({len(c.time) for c in cases}) > 1:
        trouves.append("n_steps")
    return trouves


def tracer_evolution(cases: list[SuiviRivet], outdir: Path, *, include_max: bool = False, include_frac: bool = False) -> None:
    for c in cases:
        label = c.name
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)
        safe_label = safe_label.strip("_").lower()

        fig, ax_mean = plt.subplots(figsize=(8.5, 5.1))
        ax_mean.plot(c.max_u, c.mean_damage, linewidth=2.1, label="mean_damage")
        ax_mean.set_title(f"Dommage moyen vs deplacement impose ({label})")
        ax_mean.set_xlabel("Deplacement impose [m]")
        ax_mean.set_ylabel("Mean damage")
        ax_mean.set_ylim(-0.02, 1.02)
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend(fontsize=9)
        _sauver(fig, outdir, f"evolution_mean_damage_{safe_label}.png")

        if include_max:
            fig, ax_max = plt.subplots(figsize=(8.5, 5.1))
            ax_max.plot(c.time, c.max_damage, linewidth=2.1, label="max_damage")
            ax_max.set_title(f"Dommage max vs temps ({label})")
            ax_max.set_xlabel("Temps")
            ax_max.set_ylabel("Max damage")
            ax_max.set_ylim(-0.02, 1.02)
            ax_max.grid(True, alpha=0.3)
            ax_max.legend(fontsize=9)
            _sauver(fig, outdir, f"evolution_max_damage_{safe_label}.png")

        if include_frac:
            fig, ax_frac = plt.subplots(figsize=(8.5, 5.1))
            ax_frac.plot(c.time, c.frac_d95, linewidth=2.1, linestyle="--", label="frac_damage_ge_095")
            ax_frac.set_title(f"Zone fortement endommagee vs temps ({label})")
            ax_frac.set_xlabel("Temps")
            ax_frac.set_ylabel("frac_damage_ge_095")
            ax_frac.set_ylim(-0.02, 1.02)
            ax_frac.grid(True, alpha=0.3)
            ax_frac.legend(fontsize=9)
            _sauver(fig, outdir, f"evolution_frac_damage_{safe_label}.png")


def tracer_courbe_u_contre_dommage(cases: list[SuiviRivet], outdir: Path) -> None:
    for c in cases:
        label = c.name
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)
        safe_label = safe_label.strip("_").lower()
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(c.max_u, c.max_damage, linewidth=2.0, label="max_damage")
        ax.set_title(f"Trajectoire max_u_inf -> max_damage ({label})")
        ax.set_xlabel("max_u_inf [m]")
        ax.set_ylabel("max_damage")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        _sauver(fig, outdir, f"u_vs_damage_{safe_label}.png")


def tracer_param_vs_dommage(cases: list[SuiviRivet], params: list[str], outdir: Path) -> None:
    for param in params:
        pvs = []
        for c in cases:
            v = _param_val_from_case(c, param)
            if v is None:
                continue
            pvs.append((v, c))
        if len(pvs) < 2:
            continue
        pvs.sort(key=lambda t: t[0])
        p = np.array([float(v) for v, _ in pvs], dtype=float)
        max_d = np.array([c.max_damage[-1] for _, c in pvs], dtype=float)
        mean_d = np.array([c.mean_damage[-1] for _, c in pvs], dtype=float)

        fig, ax1 = plt.subplots(figsize=(8.3, 4.8))
        ax1.plot(p, max_d, "o-", linewidth=2.0, label="max_damage final")
        ax1.set_xlabel(param)
        ax1.set_ylabel("max_damage final", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(p, mean_d, "s--", color="tab:red", linewidth=2.0, label="mean_damage final")
        ax2.set_ylabel("mean_damage final", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim(0.0, 1.0)

        ax1.set_title(f"Parametre '{param}' vs dommage final (cas rivets)")
        _sauver(fig, outdir, f"param_{param.replace('/', '_')}_vs_dommage.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graphes d'endommagement pour les cas AVEC rivets uniquement."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Dossier racine des resultats.",
    )
    parser.add_argument(
        "--monitor",
        action="append",
        default=[],
        help="Chemin monitor.csv explicite (peut etre repete).",
    )
    parser.add_argument(
        "--param",
        nargs="*",
        default=None,
        help=(
            "Parametres de config a utiliser pour les graphes dommage final vs parametre. "
            "Ex: deplacement_pic_iceberg, phase_field_gc_j_m2, phase_field_l0_m, n_steps."
        ),
    )
    parser.add_argument(
        "--include-max",
        action="store_true",
        help="Ajouter le graphe de max_damage versus temps (un fichier par cas).",
    )
    parser.add_argument(
        "--include-frac",
        action="store_true",
        help="Ajouter le graphe frac_damage_ge_095 versus temps (un fichier par cas).",
    )
    parser.add_argument(
        "--include-u-vs-d",
        action="store_true",
        help="Ajouter les courbes max_u_inf vs max_damage (un fichier par cas).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Dossier de sortie des graphes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    monitors = [Path(m) for m in args.monitor] if args.monitor else None
    cas = _collecter_cas(results_root, monitors)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = cas[0].source.parent.parent / "analyse_only_with_rivets"
    outdir.mkdir(parents=True, exist_ok=True)

    params = args.param or []
    if not params:
        params = _params_a_tracer_auto(cas)

    _nettoyer_graphes(
        outdir,
        include_max=args.include_max,
        include_frac=args.include_frac,
        include_u=args.include_u_vs_d,
    )
    tracer_evolution(cas, outdir, include_max=args.include_max, include_frac=args.include_frac)
    if args.include_u_vs_d:
        tracer_courbe_u_contre_dommage(cas, outdir)
    if params:
        tracer_param_vs_dommage(cas, params, outdir)
    print(f"Graphes ecrits dans: {outdir}")


if __name__ == "__main__":
    main()
