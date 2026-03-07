from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from grande_echelle.main import (
        config_etude_rivets_production,
        config_etude_rivets_rapide,
        config_etude_rivets_screening,
        lancer_calcul,
    )
except ModuleNotFoundError:  # pragma: no cover - execution directe du script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from main import (  # type: ignore
        config_etude_rivets_production,
        config_etude_rivets_rapide,
        config_etude_rivets_screening,
        lancer_calcul,
    )


def _charger_monitor(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"monitor vide: {path}")

    def col(name: str) -> np.ndarray:
        return np.asarray([float(r.get(name, 0.0) or 0.0) for r in rows], dtype=float)

    return {
        "time": col("time"),
        "mean_damage": col("mean_damage"),
        "frac_d95": col("frac_damage_ge_095"),
        "max_damage": col("max_damage"),
    }


def _grille_commune(a_t: np.ndarray, b_t: np.ndarray, n: int = 600) -> np.ndarray:
    t_max = min(float(np.max(a_t)), float(np.max(b_t)))
    return np.linspace(0.0, t_max, n)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _evaluer_score(ref: dict[str, np.ndarray], sim: dict[str, np.ndarray]) -> dict[str, float]:
    t = _grille_commune(ref["time"], sim["time"])
    ref_mean = np.interp(t, ref["time"], ref["mean_damage"])
    sim_mean = np.interp(t, sim["time"], sim["mean_damage"])
    ref_frac = np.interp(t, ref["time"], ref["frac_d95"])
    sim_frac = np.interp(t, sim["time"], sim["frac_d95"])

    rmse_mean = _rmse(sim_mean, ref_mean)
    rmse_frac = _rmse(sim_frac, ref_frac)
    score = 0.7 * rmse_mean + 0.3 * rmse_frac
    return {"rmse_mean_damage": rmse_mean, "rmse_frac_d95": rmse_frac, "score_total": score}


def _build_config(mode: str):
    if mode == "screening":
        return config_etude_rivets_screening(with_rivets=True)
    if mode == "rapide":
        return config_etude_rivets_rapide(with_rivets=True)
    if mode == "production":
        return config_etude_rivets_production(with_rivets=True)
    raise ValueError(f"mode inconnu: {mode}")


def _charger_bandes(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    bandes = data.get("bandes_rivets_z")
    if not isinstance(bandes, list) or not bandes:
        raise ValueError(f"Format invalide pour bandes rivets dans {path}")
    return bandes


def _creer_preset_uniforme_gc(bandes_src: list[dict], facteur_gc: float, outpath: Path) -> Path:
    bandes = []
    for b in bandes_src:
        b2 = dict(b)
        b2["facteur_Gc"] = float(facteur_gc)
        bandes.append(b2)
    payload = {
        "format": "titanic-rivet-bands/v1",
        "origine": "calibration_inverse_gc",
        "bandes_rivets_z": bandes,
    }
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return outpath


def _parse_values(raw: str) -> list[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("Aucune valeur facteur_Gc fournie")
    return vals


def _ecrire_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    champs = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=champs)
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibration inverse du facteur_Gc uniforme sur les bandes rivets"
    )
    p.add_argument(
        "--monitor-cible",
        required=True,
        type=str,
        help="monitor.csv de reference (mesure ou cas cible)",
    )
    p.add_argument(
        "--preset-bandes",
        default="rivet/bandes_rivets_grande_echelle_calibre.json",
        type=str,
        help="preset de bandes rivets de base",
    )
    p.add_argument(
        "--mode",
        default="screening",
        choices=["screening", "rapide", "production"],
        help="niveau de cout des simulations",
    )
    p.add_argument(
        "--valeurs-gc",
        default="0.40,0.50,0.60,0.70",
        type=str,
        help="liste CSV des facteurs_Gc a tester",
    )
    p.add_argument(
        "--deplacement-pic",
        default=0.025,
        type=float,
        help="deplacement impose iceberg [m] pour tous les essais",
    )
    p.add_argument(
        "--outdir",
        default="results/calibration_inverse_gc",
        type=str,
        help="dossier des sorties calibration",
    )
    return p.parse_args()


def main():
    args = parse_args()
    monitor_cible = Path(args.monitor_cible)
    if not monitor_cible.exists():
        raise FileNotFoundError(f"monitor cible introuvable: {monitor_cible}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dossiers_presets = outdir / "presets_temp"
    dossiers_presets.mkdir(parents=True, exist_ok=True)

    ref = _charger_monitor(monitor_cible)
    bandes_src = _charger_bandes(Path(args.preset_bandes))
    facteurs_gc = _parse_values(args.valeurs_gc)

    lignes = []
    meilleur = None
    for i, facteur_gc in enumerate(facteurs_gc, start=1):
        cfg = _build_config(args.mode)
        cfg.nom_cas = f"calib_inverse_gc_{args.mode}_{facteur_gc:.3f}".replace(".", "p")
        cfg.deplacement_pic_iceberg = float(args.deplacement_pic)
        cfg.uniformiser_facteur_gc_bandes = False
        cfg.utiliser_bandes_rivets_z = True

        preset_name = f"bandes_gc_{facteur_gc:.3f}".replace(".", "p") + ".json"
        preset_run = dossiers_presets / preset_name
        _creer_preset_uniforme_gc(bandes_src, facteur_gc, preset_run)
        cfg.fichier_preset_bandes_rivets = str(preset_run)

        print(f"[{i}/{len(facteurs_gc)}] run facteur_Gc={facteur_gc:.4f} -> {cfg.nom_cas}")
        lancer_calcul(cfg)

        monitor_run = Path(cfg.dossier_resultats) / cfg.nom_cas / "quasi_static" / "monitor.csv"
        sim = _charger_monitor(monitor_run)
        score = _evaluer_score(ref, sim)
        row = {
            "facteur_Gc": facteur_gc,
            "score_total": score["score_total"],
            "rmse_mean_damage": score["rmse_mean_damage"],
            "rmse_frac_d95": score["rmse_frac_d95"],
            "monitor_run": str(monitor_run),
        }
        lignes.append(row)
        if meilleur is None or row["score_total"] < meilleur["score_total"]:
            meilleur = row

    lignes = sorted(lignes, key=lambda r: r["score_total"])
    _ecrire_csv(outdir / "resultats_calibration_inverse_gc.csv", lignes)
    if meilleur is not None:
        (outdir / "meilleur_facteur_gc.json").write_text(
            json.dumps(meilleur, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Meilleur facteur_Gc={meilleur['facteur_Gc']:.4f}, "
            f"score={meilleur['score_total']:.6e}"
        )
    print(f"Sorties calibration: {outdir}")


if __name__ == "__main__":
    main()
