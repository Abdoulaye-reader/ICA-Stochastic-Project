# ICA-Stochastic-Project

Projet M2 Data Science : **ICA et Optimisation Stochastique**

This project implements three Independent Component Analysis (ICA) algorithms
from scratch in Python and evaluates them through original experiments.

---

## Algorithms implemented

| Algorithm | Module | Reference |
|-----------|--------|-----------|
| **FastICA** (deflation & symmetric) | `ica/fastica.py` | Hyvärinen & Oja (2000) |
| **Infomax** (standard & extended) | `ica/infomax.py` | Bell & Sejnowski (1995) |
| **JADE** | `ica/jade.py` | Cardoso & Souloumiac (1993) |

All algorithms share a consistent scikit-learn-style API:

```python
from ica import FastICA, InfomaxICA, JADE

model = FastICA(algorithm="deflation", g="logcosh")
S_estimated = model.fit_transform(X)   # X: (n_channels, n_samples)
```

---

## Project structure

```
ICA-Stochastic-Project/
├── ica/                   # Core ICA package
│   ├── __init__.py
│   ├── fastica.py         # FastICA algorithm
│   ├── infomax.py         # Infomax / Extended Infomax
│   ├── jade.py            # JADE algorithm
│   └── utils.py           # Signal generation, whitening, metrics
├── experiments/
│   ├── experiment_bss.py          # Blind Source Separation benchmark
│   └── experiment_sensitivity.py  # Sample size & SNR sensitivity
├── tests/
│   ├── test_fastica.py
│   ├── test_infomax.py
│   ├── test_jade.py
│   └── test_utils.py
├── requirements.txt
└── README.md
```

---

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the tests

```bash
pytest tests/ -v
```

### Run the experiments

```bash
# Blind Source Separation benchmark (compares all 5 algorithm variants)
python experiments/experiment_bss.py

# Sensitivity to number of samples and SNR
python experiments/experiment_sensitivity.py
```

Results (figures) are saved to `experiments/results/`.

---

## Utility functions

| Function | Description |
|----------|-------------|
| `generate_sources(n, T)` | Generate *n* independent non-Gaussian sources of length *T* |
| `mix_sources(S)` | Apply a random invertible mixing matrix |
| `whiten(X)` | PCA whitening (identity covariance output) |
| `amari_error(W, A)` | Amari performance index (0 = perfect separation) |
| `signal_to_interference_ratio(S_est, S_true)` | Per-source SIR in dB |

---

## Experiment results summary

### Experiment 1 — Blind Source Separation

Generates 4 independent sources (sub-/super-Gaussian mix), mixes them
with a random matrix, then recovers them with each algorithm.
Metrics: Amari error and mean SIR.

### Experiment 2 — Sensitivity analysis

* **Sub-experiment A**: Amari error vs number of samples T
  (T ∈ {100, 250, 500, 1000, 2000, 5000}) — shows sample complexity.
* **Sub-experiment B**: Amari error vs SNR (5–40 dB) — shows noise robustness.

**Key findings:**
- FastICA (symmetric, logcosh) achieves the best speed/accuracy trade-off.
- JADE is competitive and parameter-free.
- Infomax was designed for super-Gaussian sources; it performs best when
  all sources have positive kurtosis.

---

## References

1. Hyvärinen A. & Oja E. (2000). Independent component analysis: algorithms
   and applications. *Neural Networks*, 13(4–5), 411–430.
2. Bell A. J. & Sejnowski T. J. (1995). An information-maximization approach to
   blind separation and blind deconvolution. *Neural Computation*, 7(6),
   1129–1159.
3. Cardoso J.-F. & Souloumiac A. (1993). Blind beamforming for non-Gaussian
   signals. *IEE Proceedings F*, 140(6), 362–370.
4. Lee T.-W., Girolami M. & Sejnowski T. J. (1999). Independent component
   analysis using an extended infomax algorithm for mixed subgaussian and
   supergaussian sources. *Neural Computation*, 11(2), 417–441.
