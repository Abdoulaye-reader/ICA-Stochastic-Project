# ICA Stochastic Project

Projet de comparaison de methodes ICA classiques et stochastiques, avec un focus sur:
- la qualite de separation (indice d'Amari),
- le passage a l'echelle (N et D),
- l'impact de la gaussianite des sources sur l'identifiabilite.

Ce depot sert a la fois de base experimentale (scripts + notebooks) et de support de presentation academique.

## Idee generale

On part du modele ICA lineaire:

$$x = A s$$

avec des sources independantes $s$ et une matrice de melange $A$ inconnue.
L'objectif est d'estimer une matrice de demelange $V$ telle que $y = Vx$ recupere les sources (a permutation/scale pres).

La metrique principale est l'indice d'Amari, calcule sur:

$$C = \hat{V}A$$

Plus l'indice est proche de 0, meilleure est la separation.

## Structure du projet

```text
ICA-Stochastic-Project/
	src/
		algorithms.py          # FastICA, SGD-ICA, Adam-ICA, amari_index
		utils.py               # whitening, generation synthetique, metriques
	experiments/
		exp1_scalability.py    # experience N x D + generation de figures
		results/               # CSV + PDF produits par les experiences
	notebooks/
		brouillon.ipynb        # notebook de validation rapide
		gaussian_problem.ipynb # etude de la gaussianite (en francais)
		figs/                  # figures exportees depuis notebooks
	report/
		rapport.tex
		references.bib
	requirements.txt
```

## Algorithmes disponibles

Les algorithmes sont dans src/algorithms.py.

1. FastICA (baseline)
- via sklearn.decomposition.FastICA
- mode parallel, non-linearite logcosh
- utilise les donnees deja whitened (whiten=False)

2. SGD-ICA
- optimisation stochastique mini-batch
- orthogonalisation de $W$ par QR a chaque iteration
- non-linearite logcosh (tanh)

3. Adam-ICA
- meme objectif que SGD-ICA mais avec moments Adam
- correction de biais + stabilisation numerique
- orthogonalisation QR egalement

Fonctions utiles:
- amari_index(C)
- fastica_components(Xw, ...)
- fastica_unmixing_matrix(X, whitening_fn, ...)
- sgd_ica(Xw, ...)
- adam_ica(Xw, ...)

## Utilitaires disponibles

Dans src/utils.py:
- center_whiten(X): centrage + whitening PCA
- generate_synthetic_data(...): generation de sources synthetiques
- compute_performance_metrics(V_est, A, ...): metriques (Amari, source_mse)
- unmixing_from_whitened(W, W_white): conversion vers l'espace original

## Installation

Depuis la racine du projet:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependances principales:
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas

## Utilisation rapide

### 1) Test rapide en Python

```python
import numpy as np
from src.utils import generate_synthetic_data, center_whiten
from src.algorithms import sgd_ica, adam_ica, amari_index, fastica_components

# 1. Donnees synthetiques
X, S, A = generate_synthetic_data(D=3, N=5000, source_types=['laplace', 'uniform', 'student'], seed=42)

# 2. Pretraitement
Xw, mu, W_white = center_whiten(X)

# 3. FastICA
B_fast = fastica_components(Xw, n_components=3, max_iter=1000, tol=1e-6, seed=42)
V_fast = B_fast @ W_white
amari_fast = amari_index(V_fast @ A)

# 4. SGD-ICA
W_sgd, hist_sgd = sgd_ica(Xw, n_iter=500, lr=0.01, batch_size=64, seed=42)
V_sgd = W_sgd @ W_white
amari_sgd = amari_index(V_sgd @ A)

# 5. Adam-ICA
W_adam, hist_adam = adam_ica(Xw, n_iter=500, lr=0.001, batch_size=64, seed=42)
V_adam = W_adam @ W_white
amari_adam = amari_index(V_adam @ A)

print('Amari FastICA:', amari_fast)
print('Amari SGD-ICA:', amari_sgd)
print('Amari Adam-ICA:', amari_adam)
```

### 2) Lancer l'experience de scalabilite

```bash
python experiments/exp1_scalability.py \
	--n_samples 1000 5000 10000 50000 \
	--n_dims 2 3 5 \
	--n_runs 2 \
	--output experiments/results
```

Sorties generees:
- experiments/results/scalability_results.csv
- experiments/results/amari_vs_n.pdf
- experiments/results/time_vs_n.pdf
- experiments/results/amari_vs_d.pdf

### 3) Notebook gaussian_problem

Ouvrir notebooks/gaussian_problem.ipynb pour l'experience theorique sur la gaussianite:
- variation du parametre $\alpha$ d'une gaussienne generalisee,
- observation de la degradation ICA autour du cas gaussien,
- interpretation en lien avec le theoreme de Comon.

## Remarques pratiques

- Les algos stochastiques sont sensibles aux hyperparametres (lr, batch_size, n_iter).
- FastICA sert de baseline stable pour verifier les pipelines.
- Un warning sklearn peut apparaitre sur n_components avec whiten=False; c'est attendu ici car le whitening est fait en amont.

## Objectif academique du projet

Le projet ne se limite pas a "faire tourner des algos".
L'objectif est aussi de montrer une lecture mathematique correcte:
- quand ICA reussit,
- quand ICA se degrade,
- et surtout pourquoi (non-gaussianite et identifiabilite).

En pratique, cela permet de presenter un travail coherent entre theorie, implementation et validation experimentale.
