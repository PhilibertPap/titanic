# Projet Titanic - Analyse d'avarie (FEM)

## But du projet

Ce projet sert a etudier, avec un modele elements finis simplifie, si la zone des rivets peut modifier
la reponse mecanique de la coque lors d'un impact (deformation / endommagement).

L'idee est de travailler a deux echelles :

- `grande_echelle/` : modele de coque (impact localise sur un segment de coque)
- `petite_echelle/` : modele local de fissuration (phase-field) pour choisir des parametres `Gc` et `l0`

## Fichiers principaux (version simplifiee)

- `grande_echelle/mesh.py` : generation du maillage
- `grande_echelle/main.py` : configuration + lecture maillage + lancement du calcul global
- `grande_echelle/shell.py` : modele de coque (cinematique + materiaux)
- `grande_echelle/quasi_static.py` : solveur quasi-statique + dommage
- `petite_echelle/local_phase_field.py` : calcul local phase-field
- `petite_echelle/phase_field_config.py` : parametres de calibration locale

## Commandes utiles

1. Generer le maillage

```bash
python grande_echelle/mesh.py
```

2. Lancer le calcul global standard

```bash
python grande_echelle/main.py
```

3. Lancer le calcul local phase-field (utile pour la calibration)

```bash
python petite_echelle/local_phase_field.py
```

4. Comparaison avec / sans rivets : modifier dans `grande_echelle/main.py` la config de base

- soit en gardant les proprietes rivets differentes (cas "avec effet rivets")
- soit en mettant les memes proprietes que la coque (`rivet_* = shell_*`) pour un cas "sans effet rivets"
- ou utiliser les presets deja prets dans `grande_echelle/main.py` :
  - `config_etude_rivets_rapide(with_rivets=True/False)`
  - `config_etude_rivets_production(with_rivets=True/False)`

Exemple (dans un petit script Python ou un terminal interactif) :

```python
from grande_echelle.main import lancer_calcul, config_etude_rivets_rapide

cfg = config_etude_rivets_rapide(with_rivets=True)
lancer_calcul(cfg)
```

Ou plus simplement :

```bash
python run_rivets_ab.py
```

Alternative sans script (meme logique) :

```python
from grande_echelle.main import lancer_comparaison_rivets_rapide

lancer_comparaison_rivets_rapide()
```

## Preset rivets (optionnel)

Le sous-modele `rivet/` peut ecrire un preset JSON simple pour alimenter
les bandes homogenisees du modele `grande_echelle`.

```python
from rivet import creer_preset_bandes_grande_echelle

creer_preset_bandes_grande_echelle()
```

Puis dans `grande_echelle` :

```python
from grande_echelle.main import creer_config, lancer_calcul

cfg = creer_config(rivet_bandes_preset_file="rivet/bandes_rivets_grande_echelle.json")
lancer_calcul(cfg)
```

## Sorties importantes

- `run_metadata.json` : parametres du run
- `quasi_static/monitor.csv` : evolution des indicateurs + temps de calcul par pas
- `*.pvd` : champs a visualiser dans ParaView
