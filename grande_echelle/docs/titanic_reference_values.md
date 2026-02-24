# Titanic Impact Reference Values

Ce document trace les valeurs "reference" utilisees pour le cas
`titanic_1912_impact_segment`.

## Geometrie de reference (historique)

- Longueur hors tout navire: **269.1 m**
- Largeur max navire: **28.2 m**
- Tirant d'eau (charge): **~10.5 m**
- Longueur de la zone d'impact/de dommage: **ordre 80-100 m**
- Compartiments touches: **6** (forepeak, holds 1-3, boiler rooms 6-5)
- Surface totale des ouvertures equivalentes: **~1.171 m2**

## Materiau acier (ordres de grandeur)

- Limite d'elasticite: **~262-280 MPa** (modele: 270 MPa)
- Resistance ultime: **~430-432 MPa** (modele: 431 MPa)
- Module d'Young elastique (modele FEM): **210 GPa**
- Coefficient de Poisson (modele FEM): **0.3**
- Soufre (essais NIST): **~0.061-0.069 wt%**

## Mapping dans ce projet

- `mesh.py`
  - `L=100.0` m (portion impact)
  - `B=14.1` m (demi-largeur, largeur totale 28.2 m)
  - `T=10.5` m
- `config.py`
  - `case_name="titanic_1912_impact_segment"`
  - `shell_thickness=0.025` m (plaque effective)
  - `rivet_thickness=0.025` m (zone seam effective)
  - `shell_yield_strength_pa=270e6`
  - `shell_ultimate_strength_pa=431e6`

Pour le detail des cotes de rivets et leur translation dans le maillage:
voir `docs/titanic_rivet_dimensions.md`.

## Sources

1. NIST, metallurgie Titanic (chimie et proprietes):  
   https://www.nist.gov/publications/metallurgy-rms-titanic
2. NISTIR 6118 (valeurs d'essais, details):  
   https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=852863
3. JOM/TMS, donnees dommage et dimensions navire:  
   https://www.tms.org/pubs/journals/jom/9801/felkins-9801.html
4. UK Inquiry (dommage de l'ordre de 300 ft):  
   https://www.gutenberg.org/ebooks/39415.html.images
