# Petite echelle

Sous-modele local pour calibrer la rupture phase-field avant integration
dans le modele `grande_echelle`.

## Objectif

- Calibrer des couples `(Gc, l0)` coherents avec les ordres de grandeur Titanic.
- Fournir un point de depart robuste pour les futurs runs fissuration.

## Fichiers

- `phase_field_config.py`: configuration de calibration.
- `scripts/calibrate_phase_field.py`: generation des candidats `Gc/l0`.
- `docs/phase_field_notes.md`: hypotheses et formules.
- `results/<case>/`: sorties de calibration (`metadata.json`, `candidates.csv`, `baseline.json`).

## Lancer la calibration

```bash
python petite_echelle/scripts/calibrate_phase_field.py
```

## Lancer un premier cas phase-field local

```bash
python petite_echelle/scripts/run_local_phase_field.py
```

Sorties:

- `results/local_phase_field_baseline/displacement.pvd`
- `results/local_phase_field_baseline/damage.pvd`
- `results/local_phase_field_baseline/monitor.csv`

## Sweep autour du baseline

```bash
python petite_echelle/scripts/sweep_local_phase_field.py
```

Le sweep explore:

- `Gc = baseline * {0.8, 1.0, 1.2}`
- `l0/h = {5, 6, 7}`

Synthese:

- `results/local_phase_field_sweep/summary.csv`

## Selection automatique d'un preset

```bash
python petite_echelle/scripts/select_phase_field_preset.py
```

Sorties:

- `petite_echelle/results/local_phase_field_sweep/selected_preset.json`
- `grande_echelle/coupling/phase_field_selected_preset.json` (copie de reference pour le modele global)

## Suite recommandee

1. Prendre `baseline.json` pour un premier solveur phase-field local.
2. Faire un sweep cible autour du baseline (ex: `Gc +/- 20%`, `l0/h` de 5 a 7).
3. Confronter la longueur/forme de dommage a l'objectif physique.
