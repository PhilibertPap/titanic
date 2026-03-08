from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

LABEL_FRAC_D95 = "part zone tres endommagee (d >= 0.95)"


@dataclass
class Suivi:
    source: Path
    label: str
    time: np.ndarray
    max_u: np.ndarray
    max_damage: np.ndarray
    mean_damage: np.ndarray
    frac_d95: np.ndarray
    temps_pas: np.ndarray
    temps_meca: np.ndarray
    temps_pf: np.ndarray


def _charger_monitor(path: Path, label: str) -> Suivi:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"monitor.csv vide: {path}")

    def col(name: str) -> np.ndarray:
        return np.asarray([float(r.get(name, 0.0) or 0.0) for r in rows], dtype=float)

    return Suivi(
        source=path,
        label=label,
        time=col("time"),
        max_u=col("max_u_inf"),
        max_damage=col("max_damage"),
        mean_damage=col("mean_damage"),
        frac_d95=col("frac_damage_ge_095"),
        temps_pas=col("temps_pas_s"),
        temps_meca=col("temps_meca_s"),
        temps_pf=col("temps_phase_field_s"),
    )


def _find_case_monitor(results_root: Path, pattern: str) -> Path | None:
    candidats = sorted(results_root.glob(pattern))
    if not candidats:
        return None
    return max(candidats, key=lambda p: p.stat().st_mtime)


def _save(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _grille_temps_commune(a: Suivi, b: Suivi, n: int = 600) -> np.ndarray:
    tmax = min(float(a.time.max()), float(b.time.max()))
    return np.linspace(0.0, tmax, n)


def _interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp)


def tracer_dommages(avec: Suivi, sans: Suivi, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(avec.time, avec.max_damage, linewidth=2.2, label=avec.label)
    axes[0].plot(sans.time, sans.max_damage, linewidth=2.2, label=sans.label)
    axes[0].set_title("max_damage")
    axes[0].set_xlabel("Temps")
    axes[0].set_ylabel("Dommage")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(avec.time, avec.mean_damage, linewidth=2.2, label=f"{avec.label} dommage moyen")
    axes[1].plot(sans.time, sans.mean_damage, linewidth=2.2, label=f"{sans.label} dommage moyen")
    axes[1].plot(avec.time, avec.frac_d95, "--", linewidth=2.0, label=f"{avec.label} {LABEL_FRAC_D95}")
    axes[1].plot(sans.time, sans.frac_d95, "--", linewidth=2.0, label=f"{sans.label} {LABEL_FRAC_D95}")
    axes[1].set_title("Dommage moyen et zone tres endommagee")
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel("Niveau dommage")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    _save(fig, outdir / "comparaison_dommages.png")


def tracer_ecarts_dommages(avec: Suivi, sans: Suivi, outdir: Path):
    grille = _grille_temps_commune(avec, sans)
    max_avec = _interp(grille, avec.time, avec.max_damage)
    max_sans = _interp(grille, sans.time, sans.max_damage)
    mean_avec = _interp(grille, avec.time, avec.mean_damage)
    mean_sans = _interp(grille, sans.time, sans.mean_damage)
    frac_avec = _interp(grille, avec.time, avec.frac_d95)
    frac_sans = _interp(grille, sans.time, sans.frac_d95)

    def _delta_rel_percent(val_avec: np.ndarray, val_sans: np.ndarray) -> np.ndarray:
        delta = val_avec - val_sans
        denom = np.maximum(np.abs(val_sans), 1e-8)
        out = 100.0 * delta / denom
        out[np.abs(val_sans) < 1e-6] = 0.0
        return out

    delta_mean = _delta_rel_percent(mean_avec, mean_sans)
    delta_frac = _delta_rel_percent(frac_avec, frac_sans)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True)
    axes[0].plot(grille, delta_mean, linewidth=2.0)
    axes[0].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    axes[0].set_ylabel("Delta dommage moyen [%]")
    axes[0].set_title("Ecarts temporels relatifs (avec - sans) / sans")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(grille, delta_frac, linewidth=2.0)
    axes[1].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    axes[1].set_ylabel(f"Delta {LABEL_FRAC_D95} [%]")
    axes[1].set_xlabel("Temps")
    axes[1].grid(True, alpha=0.3)

    _save(fig, outdir / "ecarts_temporels_dommages.png")


def _extract_step_from_damage_name(path: Path) -> int:
    m = re.search(r"damage(\d+)\.pvtu$", path.name)
    return int(m.group(1)) if m else -1


def _mask_points_from_bandes(points: np.ndarray, bandes: list[dict]) -> np.ndarray:
    if len(points.shape) != 2 or points.shape[1] < 3 or not bandes:
        return np.zeros(points.shape[0], dtype=bool)
    x = points[:, 0]
    z = points[:, 2]
    mask = np.zeros(points.shape[0], dtype=bool)
    for b in bandes:
        xc = float(b.get("x_centre_m", 0.0))
        w = float(b.get("largeur_x_m", 0.3))
        xmin = xc - 0.5 * w
        xmax = xc + 0.5 * w
        zmin = float(b.get("z_min_m", -np.inf))
        zmax = float(b.get("z_max_m", np.inf))
        if zmax < zmin:
            zmin, zmax = zmax, zmin
        mask |= (x >= xmin) & (x <= xmax) & (z >= zmin) & (z <= zmax)
    return mask


def _lire_bandes_avec_depuis_metadata(avec: Suivi) -> list[dict]:
    run_dir = avec.source.parent.parent
    meta_path = run_dir / "run_metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    bandes = data.get("config", {}).get("bandes_rivets_z", [])
    return list(bandes) if isinstance(bandes, list) else []


def _serie_dommage_bandes(quasi_static_dir: Path, bandes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(quasi_static_dir.glob("damage*.pvtu"), key=_extract_step_from_damage_name)
    if not files:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    steps = []
    means = []
    for p in files:
        step = _extract_step_from_damage_name(p)
        if step < 0:
            continue
        mesh = pv.read(str(p))
        if "Damage" not in mesh.array_names:
            continue
        mask = _mask_points_from_bandes(mesh.points, bandes)
        values = np.asarray(mesh["Damage"], dtype=float)
        mean_band = float(np.mean(values[mask])) if np.any(mask) else 0.0
        steps.append(step)
        means.append(mean_band)
    return np.asarray(steps, dtype=int), np.asarray(means, dtype=float)


def tracer_dommage_moyen_bandes(avec: Suivi, sans: Suivi, outdir: Path):
    bandes = _lire_bandes_avec_depuis_metadata(avec)
    if not bandes:
        return

    dir_avec = avec.source.parent
    dir_sans = sans.source.parent
    step_avec, mean_band_avec = _serie_dommage_bandes(dir_avec, bandes)
    step_sans, mean_band_sans = _serie_dommage_bandes(dir_sans, bandes)
    if step_avec.size == 0 or step_sans.size == 0:
        return

    idx_t_avec = np.clip(step_avec, 0, len(avec.time) - 1)
    idx_t_sans = np.clip(step_sans, 0, len(sans.time) - 1)
    t_avec = avec.time[idx_t_avec]
    t_sans = sans.time[idx_t_sans]

    t_common = _grille_temps_commune(
        Suivi(avec.source, avec.label, t_avec, avec.max_u[: len(t_avec)], avec.max_damage[: len(t_avec)],
              avec.mean_damage[: len(t_avec)], avec.frac_d95[: len(t_avec)],
              avec.temps_pas[: len(t_avec)], avec.temps_meca[: len(t_avec)], avec.temps_pf[: len(t_avec)]),
        Suivi(sans.source, sans.label, t_sans, sans.max_u[: len(t_sans)], sans.max_damage[: len(t_sans)],
              sans.mean_damage[: len(t_sans)], sans.frac_d95[: len(t_sans)],
              sans.temps_pas[: len(t_sans)], sans.temps_meca[: len(t_sans)], sans.temps_pf[: len(t_sans)]),
        n=500,
    )
    band_avec_i = np.interp(t_common, t_avec, mean_band_avec)
    band_sans_i = np.interp(t_common, t_sans, mean_band_sans)
    delta_rel = 100.0 * (band_avec_i - band_sans_i) / np.maximum(np.abs(band_sans_i), 1e-8)
    delta_rel[np.abs(band_sans_i) < 1e-6] = 0.0

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    axes[0].plot(t_avec, mean_band_avec, linewidth=2.2, label=f"{avec.label} (bandes)")
    axes[0].plot(t_sans, mean_band_sans, linewidth=2.2, label=f"{sans.label} (meme zone)")
    axes[0].set_ylabel("Dommage moyen")
    axes[0].set_title("Dommage moyen dans les bandes rivets (meme zone comparee)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(t_common, delta_rel, linewidth=2.2, color="tab:red")
    axes[1].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel("Delta relatif [%]")
    axes[1].grid(True, alpha=0.3)

    _save(fig, outdir / "comparaison_dommage_moyen_bandes.png")


def tracer_deplacement(avec: Suivi, sans: Suivi, outdir: Path):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(avec.time, avec.max_u, linewidth=2.2, label=avec.label)
    ax.plot(sans.time, sans.max_u, linewidth=2.2, label=sans.label)
    ax.set_title("max_u_inf (deplacement max)")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Deplacement [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir / "comparaison_deplacement.png")


def tracer_trajectoire_u_d(avec: Suivi, sans: Suivi, outdir: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(avec.max_u, avec.max_damage, linewidth=2.2, label=avec.label)
    ax.plot(sans.max_u, sans.max_damage, linewidth=2.2, label=sans.label)
    ax.set_title("Trajectoire max_u vs max_damage")
    ax.set_xlabel("max_u_inf [m]")
    ax.set_ylabel("max_damage")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir / "trajectoire_u_vs_damage.png")


def tracer_couts(avec: Suivi, sans: Suivi, outdir: Path):
    labels = ["temps_pas", "temps_meca", "temps_phase_field"]
    val_avec = [avec.temps_pas.sum(), avec.temps_meca.sum(), avec.temps_pf.sum()]
    val_sans = [sans.temps_pas.sum(), sans.temps_meca.sum(), sans.temps_pf.sum()]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width / 2, val_avec, width, label=avec.label)
    ax.bar(x + width / 2, val_sans, width, label=sans.label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Temps cumule [s]")
    ax.set_title("Comparaison des couts de calcul")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, outdir / "comparaison_couts.png")


def ecrire_resume(avec: Suivi, sans: Suivi, outdir: Path):
    grille = _grille_temps_commune(avec, sans)
    dmean_avec = _interp(grille, avec.time, avec.mean_damage)
    dmean_sans = _interp(grille, sans.time, sans.mean_damage)
    auc_avec = float(np.trapezoid(dmean_avec, grille))
    auc_sans = float(np.trapezoid(dmean_sans, grille))

    lignes = []
    lignes.append(f"avec={avec.source}")
    lignes.append(f"sans={sans.source}")
    lignes.append("")
    lignes.append(f"n_steps_avec={len(avec.time)-1}")
    lignes.append(f"n_steps_sans={len(sans.time)-1}")
    lignes.append("")
    lignes.append(f"max_damage_final_avec={avec.max_damage[-1]:.6f}")
    lignes.append(f"max_damage_final_sans={sans.max_damage[-1]:.6f}")
    lignes.append(f"delta_max_damage_final={(avec.max_damage[-1]-sans.max_damage[-1]):+.6f}")
    lignes.append("")
    lignes.append(f"mean_damage_final_avec={avec.mean_damage[-1]:.6f}")
    lignes.append(f"mean_damage_final_sans={sans.mean_damage[-1]:.6f}")
    lignes.append(f"delta_mean_damage_final={(avec.mean_damage[-1]-sans.mean_damage[-1]):+.6f}")
    lignes.append("")
    lignes.append(f"frac_d95_final_avec={avec.frac_d95[-1]:.6f}")
    lignes.append(f"frac_d95_final_sans={sans.frac_d95[-1]:.6f}")
    lignes.append(f"delta_frac_d95_final={(avec.frac_d95[-1]-sans.frac_d95[-1]):+.6f}")
    lignes.append("")
    lignes.append(f"max_u_final_avec={avec.max_u[-1]:.6e}")
    lignes.append(f"max_u_final_sans={sans.max_u[-1]:.6e}")
    lignes.append(f"delta_max_u_final={(avec.max_u[-1]-sans.max_u[-1]):+.6e}")
    lignes.append("")
    lignes.append(f"temps_total_pas_avec={avec.temps_pas.sum():.3f}")
    lignes.append(f"temps_total_pas_sans={sans.temps_pas.sum():.3f}")
    lignes.append(f"delta_temps_total_pas={(avec.temps_pas.sum()-sans.temps_pas.sum()):+.3f}")
    lignes.append("")
    lignes.append(f"auc_mean_damage_avec={auc_avec:.6f}")
    lignes.append(f"auc_mean_damage_sans={auc_sans:.6f}")
    lignes.append(f"delta_auc_mean_damage={(auc_avec-auc_sans):+.6f}")

    (outdir / "resume_comparaison_rivets.txt").write_text("\n".join(lignes) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Graphes de comparaison avec vs sans rivets")
    parser.add_argument("--monitor-avec", type=str, default=None, help="monitor.csv du cas avec rivets")
    parser.add_argument("--monitor-sans", type=str, default=None, help="monitor.csv du cas sans rivets")
    parser.add_argument("--results-root", type=str, default="results", help="dossier racine des resultats")
    parser.add_argument("--outdir", type=str, default=None, help="dossier de sortie des graphes")
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)

    if args.monitor_avec:
        p_avec = Path(args.monitor_avec)
    else:
        p_avec = _find_case_monitor(results_root, "*with_rivets/quasi_static/monitor.csv")
    if args.monitor_sans:
        p_sans = Path(args.monitor_sans)
    else:
        p_sans = _find_case_monitor(results_root, "*without_rivets/quasi_static/monitor.csv")

    if p_avec is None or not p_avec.exists():
        raise FileNotFoundError("Impossible de trouver le monitor du cas AVEC rivets")
    if p_sans is None or not p_sans.exists():
        raise FileNotFoundError("Impossible de trouver le monitor du cas SANS rivets")

    avec = _charger_monitor(p_avec, "avec rivets")
    sans = _charger_monitor(p_sans, "sans rivets")

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = p_avec.parent.parent / "analyse_comparaison_rivets"
    outdir.mkdir(parents=True, exist_ok=True)

    tracer_dommages(avec, sans, outdir)
    tracer_ecarts_dommages(avec, sans, outdir)
    tracer_dommage_moyen_bandes(avec, sans, outdir)
    tracer_deplacement(avec, sans, outdir)
    tracer_trajectoire_u_d(avec, sans, outdir)
    tracer_couts(avec, sans, outdir)
    ecrire_resume(avec, sans, outdir)

    print(f"Graphes de comparaison ecrits dans: {outdir}")


if __name__ == "__main__":
    main()
