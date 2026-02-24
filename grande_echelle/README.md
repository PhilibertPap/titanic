# Grande echelle

Contient le modèle coque principal.

## Fichiers

- `main.py`: orchestration du calcul
- `config.py`: paramètres de simulation
- `mesh.py`: générateur Gmsh de coque
- `mesh_ex.py`: exemple alternatif de géométrie
- `loads.py`: chargements (gravité, gaussiennes)
- `materials.py`: champs matériau hétérogènes (zones rivet / non-rivet)
- `model/shell.py`: construction du modèle coque (espaces + formes)
- `solvers/quasi_static.py`: résolution quasi-statique
- `fem_io/mesh_io.py`: lecture maillage et exports VTK
- `cli/run.py`: point d’entrée CLI
- `docs/titanic_reference_values.md`: valeurs de reference + sources
- `docs/titanic_rivet_dimensions.md`: cotes rivets Titanic + mapping modele
- `coupling/phase_field_selected_preset.json`: preset local phase-field retenu pour couplage
- `scripts/run_with_selected_preset.py`: lancement global avec verification preset
- `mesh/`: fichiers maillage `.msh`
- `results/<case_name>/`: résultats de simulation organisés par cas

## Points techniques

- Les tags physiques Gmsh `1` et `2` sont utilisés pour les bords gauche/droit (conditions limites).
- Le couplage principal se fait via `SimulationConfig`.
- Deux matériaux de coque sont gérés via tags de cellules:
  - `shell_cell_tag=1` (plaque standard)
  - `rivet_cell_tag=2` (bandes de rivets)
- Les propriétés associées se règlent dans `config.py` (`shell_*` et `rivet_*`).
- L’encastrement peut être appliqué sur 2 ou 4 bords via:
  - `clamp_all_edges` (ajoute `bottom/top`)
  - `clamp_rotations` (blocage des rotations en plus des déplacements)
- L’iceberg peut être modélisé de 2 façons:
  - `iceberg_loading="neumann_pressure"` (pression surfacique)
  - `iceberg_loading="dirichlet_displacement"` (déplacement imposé sur zone d’impact)
  - Le sens du déplacement imposé se règle avec `iceberg_disp_sign` (`+1` ou `-1`)
- Les sorties quasi-statiques sont séparées en:
  - `local_frame/local_basis_vectors.pvd`
  - `local_frame/material_regions.pvd` (tags matière pour visualiser les rivets)
  - `quasi_static/displacement.pvd`
  - `quasi_static/rotation.pvd`
  - `quasi_static/damage.pvd` (phase-field global)
  - `quasi_static/monitor.csv`
  - `run_metadata.json` (inclut notamment `sigma`)
- Si `phase_field_preset_file` existe, son contenu est injecté dans `run_metadata.json` et peut piloter `Gc/l0`.

## Lancement conseille (preset local -> global)

```bash
python petite_echelle/scripts/select_phase_field_preset.py
python grande_echelle/scripts/run_with_selected_preset.py
```
