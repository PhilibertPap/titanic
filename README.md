# Titanic FEM Sandbox

Projet de simulation éléments finis (coques), centré sur un modèle coque.

- `grande_echelle/`: génération de maillage, solveur et post-traitement
- `petite_echelle/`: calibration locale phase-field (fissuration)

## Structure actuelle

- `grande_echelle/main.py`: orchestration du calcul
- `grande_echelle/config.py`: paramètres de simulation
- `grande_echelle/model/shell.py`: modèle coque (cinématique + formes faibles)
- `grande_echelle/solvers/quasi_static.py`: solveur quasi-statique
- `grande_echelle/fem_io/mesh_io.py`: lecture/écriture I/O FEM
- `grande_echelle/mesh.py`: génération Gmsh de la coque
- `grande_echelle/mesh_ex.py`: générateur alternatif (profil I en arche)
- `grande_echelle/loads.py`: fonctions de chargement
- `grande_echelle/mesh/*.msh`: maillages
- `grande_echelle/results/`: sorties VTK (ignorées par Git)
- `petite_echelle/scripts/calibrate_phase_field.py`: génération de candidats `Gc/l0`

## Exécution type

1. Générer le maillage:

```bash
python grande_echelle/mesh.py
```

2. Lancer la simulation:

```bash
python grande_echelle/main.py
```

## Prochaine phase recommandée

- Isoler les paramètres de simulation dans un module de configuration
- Ajouter des tests minimaux sur `loads.py` et les routines géométriques
- Ajouter un `pyproject.toml` pour exécution standardisée et dépendances explicites
