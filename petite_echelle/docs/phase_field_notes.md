# Notes phase-field (calibration locale)

## Formule de depart pour `Gc`

On utilise l'analogie LEFM:

- `Gc = KIc^2 / E'`
- `E' = E / (1 - nu^2)` en deformation plane

Hypotheses Titanic utilisees:

- `E = 210 GPa`
- `nu = 0.30`
- `KIc` explore entre `30` et `50 MPa*sqrt(m)` (plage acier doux fragile/froid)

Cette plage donne un ordre de grandeur de `Gc` de quelques kJ/m2 a une dizaine de kJ/m2.

## Choix de `l0`

En pratique numerique phase-field:

- prendre `l0 / h` entre `4` et `8` pour eviter un dommage trop diffus ou mal resolu.

Avec `h = 0.25 m`, cela donne:

- `l0` entre `1.0 m` et `2.0 m`.

## Baseline propose

Pour demarrer:

- `Gc = 7000 J/m2`
- `l0/h = 6` (donc `l0 = 1.5 m` si `h = 0.25 m`)