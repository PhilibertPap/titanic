# Cotes Rivets Titanic (Modele Global)

## Valeurs historiques retenues

- Diametre de rivet de coque: **7/8 in** (soit **22.2 mm**).
- Pas longitudinal entre rivets: **3 in** (soit **76.2 mm**).

Ces valeurs proviennent de la deposition de E.J. Wilding (Harland & Wolff)
au British Wreck Commissioner's Inquiry (1912), sur la pratique de rivetage
de coque.

## Mapping dans `grande_echelle/mesh.py`

- Segment de coque modele: `L=100.0 m`.
- Maillage longitudinal: `Nu=320`, soit `L/Nu = 0.3125 m` par colonne.
- Bandes rivets: `rivet_u_centers = [0.2, 0.4, 0.6, 0.8]`.
- `rivet_band_half_width_u = 0.0018`.

Avec ce maillage, une ligne de rivets est representee par environ
**une colonne d'elements** (largeur effective ~0.31 m). C'est plus large
que la cote physique (~0.02-0.08 m selon interpretation mono/multi-rangs),
mais coherent avec une approche homogenisee a grande echelle.

## Consequence pratique

- Si tu veux une largeur de bande plus proche de la cote physique, il faut
  augmenter `Nu` (raffinement en `u`) ou passer a un marquage par lignes
  (1D) + loi interface, au lieu de bandes 2D de cellules.

## Sources

1. British Wreck Commissioner's Inquiry (1912), testimony E.J. Wilding
   (diametre et pas des rivets):  
   https://www.titanicinquiry.org/BOTInq/BOTReport/botRepRivets.php
2. UK Inquiry transcripts (archive):  
   https://www.gutenberg.org/ebooks/39415.html.images
