# ICA Stochastic Optimization Project

Projet M2 Data Science: ICA et Optimisation Stochastique

This project implements various Independent Component Analysis (ICA) algorithms using stochastic optimization methods.

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── algorithms.py      # ICA algorithm implementations
│   ├── utils.py          # Utility functions (Amari index, data generation, etc.)
├── experiments/
│   ├── basic_comparison.py    # Basic experiment comparing algorithms
├── notebooks/            # Jupyter notebooks for exploratory analysis
├── data/                # Synthetic and real datasets
├── report/              # Final report (LaTeX/Quarto)
└── README.md
```

## Implemented Algorithms

### Baseline Algorithms
1. **Infomax Batch** (Bell & Sejnowski, 1995) - Batch gradient descent on mutual information
2. **FastICA** (Hyvärinen, 1999) - Fast fixed-point algorithm

### Stochastic Variants
3. **SGD-ICA** - Stochastic Gradient Descent variant of Infomax
4. **Adam-ICA** - Infomax with Adam optimizer

### Advanced (Optional)
5. **Natural Gradient SGD** (Amari, 1998) - Natural gradient with efficient Riemannian geometry

## Key Features

- **Modular design**: Separate utilities, algorithms, and experiments
- **Amari Index**: Standard metric for evaluating ICA separation quality
- **Data generation**: Synthetic ICA data with known sources and mixing matrices
- **Whitening/Centering**: Standard preprocessing for ICA
- **Reproducible**: All algorithms use fixed random seeds

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Basic Experiment

```bash
cd /path/to/ICA-Stochastic-Project
python experiments/basic_comparison.py
```

## Evaluation Metrics

The **Amari Index** measures how well the estimated unmixing matrix V recovers the true mixing matrix W:
- Value 0: Perfect separation (ideal)
- Value 1: Worst case (complete failure)
- See `src/utils.py` for implementation details

## Project Timeline

1. ✅ Setup project structure and core modules
2. ⏳ Complete algorithm implementations
3. ⏳ Design and run comprehensive experiments
4. ⏳ Implement creative contribution
5. ⏳ Write final report
6. ⏳ Prepare presentation

## References

- Hyvärinen, A., Karhunen, J., & Oja, E. (2001). *Independent Component Analysis*. Wiley.
- Bell, A.J., & Sejnowski, T.J. (1995). Neural Computation, 7(6), 1129-1159.
- Amari, S. (1998). Neural Computation, 10(2), 251-276.
- Kingma, D.P., & Ba, J. (2015). ICLR.

## Author

Abdoulaye Diallo
