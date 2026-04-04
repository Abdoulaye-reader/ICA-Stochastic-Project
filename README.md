# ICA Stochastic Project

Ce projet presente une comparaison experimentale entre methodes ICA classiques et stochastiques, avec une progression en trois volets:
1. scalabilite,
2. convergence,
3. robustesse au bruit,
4. impact de la non-gaussianite des sources.

Le travail est accompagne d'un rapport.

## Objectif 

On considere le modele ICA lineaire:

$$x = A s$$

ou $s$ est le vecteur des sources independantes et $A$ la matrice de melange inconnue.
L'objectif est d'estimer une matrice de demelange $V$ telle que $y = Vx$ recupere les sources a permutation et echelle pres.

Deux familles d'indicateurs sont utilisees:
- qualite de separation: indice d'Amari,
- dynamique d'optimisation: evolution de la fonction objective Infomax.

## Etat actuel du projet

Le projet est maintenant structure autour de:
- `scalability.py`: comparaison FastICA / SGD-ICA / Adam-ICA en scalabilite,
- `convergence_analysis.py`: analyse de convergence des algorithmes stochastiques via la fonction objective,
- `noise_robustness.py`: comparaison des 3 algorithmes sous bruit gaussien,
- `gaussian_problem.ipynb`: etude de l'effet de la forme des sources sur l'identifiabilite ICA.

## Arborescence

```text
ICA-Stochastic-Project/
  src/
    algorithms.py
    utils.py
  experiments/
    scalability.py
    convergence_analysis.py
    noise_robustness.py
    results/
      amari_vs_d.pdf
      amari_vs_n.pdf
      time_vs_d.pdf
      time_vs_n.pdf
  notebooks/
    brouillon.ipynb
    gaussian_problem.ipynb
  results/
    convergence_plot.png
    noise_robustness_plot.png
  report/
    rapport.tex
    references.bib
  requirements.txt
```

## Algorithmes implementes

### FastICA
Implementation de reference via `sklearn.decomposition.FastICA` (mode parallel, non-linearite `logcosh`), avec whitening effectue en amont.

### SGD-ICA
Version stochastique mini-batch du gradient Infomax, avec re-orthogonalisation QR a chaque iteration.

### Adam-ICA
Meme gradient que SGD-ICA, mais avec adaptation des pas par moments d'Adam.

## Installation

Depuis la racine du depot:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer les experiences

### 1) Scalabilite

```bash
python experiments/scalability.py
```

Ce script produit des tableaux/figures de qualite et de temps selon la dimension et le nombre d'echantillons. Les sorties sont ecrites dans `experiments/results/`.

### 2) Convergence stochastique

```bash
python experiments/convergence_analysis.py --d 10 --n 5000 --n-iter 2000
```

Sorties generees:
- `results/convergence_raw.csv`
- `results/convergence_summary.csv`
- `results/convergence_plot.png`

### 3) Robustesse au bruit

```bash
python experiments/noise_robustness.py --d 8 --n 4000 --n-iter 1500
```

Sorties generees:
- `results/noise_robustness_raw.csv`
- `results/noise_robustness_summary.csv`
- `results/noise_robustness_plot.png`

## Rapport

Le rapport final est disponible dans:
- `report/rapport.tex`

Compilation locale:

```bash
cd report
pdflatex rapport.tex
bibtex rapport
pdflatex rapport.tex
pdflatex rapport.tex
```

Le PDF produit est `report/rapport.pdf`.

## Notes notebook

Les notebooks sont fournis pour l'exploration et la validation progressive. Ils jouent un role de support a la version finale du projet, avec un passage progressif du brouillon vers des resultats stabilises.

Le notebook `gaussian_problem.ipynb` documente l'impact de la non-gaussianite sur les performances ICA. Il met en evidence la degradation autour du cas quasi-gaussien et sert de validation theoriquement interpretable pour la partie finale du rapport.

Le notebook `brouillon.ipynb` reste un espace de travail plus exploratoire pour les tests de grande dimension et les analyses complementaires.

## Notes de lecture des resultats

- Un indice d'Amari proche de 0 signifie une meilleure separation.
- L'objectif Infomax est maximise: une augmentation de la courbe objective indique une convergence numerique coherente.
- Avec les reglages actuels, FastICA reste la baseline la plus performante en precision sur les cas tests principaux.
- Les methodes stochastiques sont surtout utiles pour l'analyse de convergence, la robustesse et certains regimes de grande dimension.
- La partie gaussienne montre que la non-gaussianite des sources est un facteur determinant, avec une zone critique proche du cas gaussien.


