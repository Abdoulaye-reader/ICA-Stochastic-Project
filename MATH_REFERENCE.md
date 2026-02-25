# ICA Mathematical Reference Card

## Core ICA Model
```
x = W*z          (mixing model, x observed, z sources, W mixing matrix)
y = V*x          (unmixing, V = W^{-1} estimated, y estimated sources)
```

## Log-Likelihood Objective (Infomax)
```
ℓ(V) = log|det(V)| + E[Σⱼ log pⱼ(yⱼ)]
     = log|det(V)| + Σⱼ E[log pⱼ(V·x)]
```

## Gradient of Log-Likelihood (Key!)
```
∂ℓ/∂V = -V^{-T} + E[g(y)·x^T]

where:
- g(y) = d/dy log p(y)  (score function)
- V^{-T} = (V^T)^{-1}
```

## Common Score Functions g(y)
| Function | Formula | Use Case |
|----------|---------|----------|
| **logcosh** | tanh(αy), α≈1 | General (robust) |
| **exp** | y·exp(-y²/2) | Super-Gaussian sources |
| **cube** | y³ | Super-Gaussian sources |
| **Gaussian** | y | Not recommended for ICA |

## Preprocessing Steps

### 1. Centering
```
x_centered = x - mean(x)
```

### 2. Whitening (ZCA)
```
Cov(x) = E[x·x^T]
Λ, U = eig(Cov(x))
Q = U·Λ^{-1/2}·U^T
x_white = x·Q^T
```

## Algorithm Update Rules

### Infomax Batch Gradient
```
V_new = V + η·∂ℓ/∂V
      = V + η·(-V^{-T} + (1/N)·Σ g(yᵢ)·xᵢ^T)
```

### SGD-ICA (Mini-batch)
```
Sample batch of size m from N samples
V_new = V + η·(-V^{-T} + (1/m)·Σᵢ∈batch g(yᵢ)·xᵢ^T)
```

### Adam-ICA
```
gₜ = ∂ℓ/∂V |ₜ (gradient at iteration t)
mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ
vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²
m̂ₜ = mₜ/(1-β₁^t)  (bias correction)
v̂ₜ = vₜ/(1-β₂^t)  (bias correction)
V_new = V - α·m̂ₜ/(√v̂ₜ + ε)

Default: β₁=0.9, β₂=0.999, ε=1e-8, α=0.001
```

### Natural Gradient (Amari 1998)
```
V_new = V + η·(I + g(y)·y^T)·V

Why it works: Exploits Fisher information geometry
for faster convergence
```

### FastICA (Fixed-Point)
```
For each component wⱼ (row of W):
1. wⱼ ← E[x·g(wⱼ^T·x)] - E[g'(wⱼ^T·x)]·wⱼ
2. wⱼ ← wⱼ / ||wⱼ||  (normalize)
3. Orthogonalize: wⱼ ← wⱼ - Σ(wⱼ^T·wₖ)·wₖ for k<j
4. Repeat until convergence
```

## Performance Matrix and Amari Index

### Performance Matrix
```
C = V·W
(Should be permutation-scaling matrix if perfect)
```

### Amari Index Formula
```
r(C) = (1/(2D(D-1))) · [
  Σᵢ(Σⱼ|cᵢⱼ|/max_k|cᵢₖ| - 1) +
  Σⱼ(Σᵢ|cᵢⱼ|/max_k|cₖⱼ| - 1)
]

Perfect: r = 0
Worst: r ≈ 1
```

### Interpretation
```
r < 0.1   → Excellent separation
0.1 ≤ r < 0.3  → Good separation
0.3 ≤ r < 0.5  → Fair separation
r ≥ 0.5   → Poor separation
```

## Hyperparameters to Tune

| Param | Default | Range | Notes |
|-------|---------|-------|-------|
| **Learning Rate (η)** | 0.01 | [0.001, 0.1] | Smaller for SGD, larger for batch |
| **Batch Size** | 32 | [16, 256] | Larger = more stable but slower |
| **Max Iterations** | 500 | [100, 2000] | Stop early if converged |
| **Nonlinearity** | tanh | logcosh, exp, cube | Try multiple! |
| **Initialization** | Random | QR decomp | Important for FastICA |

## Convergence Criteria

### Absolute Tolerance
```
||V_new - V_old||_F < tol  (Frobenius norm)
Typical: tol = 1e-5
```

### Relative Tolerance
```
||V_new - V_old||_F / ||V||_F < rel_tol
Typical: rel_tol = 1e-5
```

### Loss-based
```
|Loss_new - Loss_old| < tol
Typical: tol = 1e-6
```

## Numerical Stability Tips

1. **Always whiten first** - Normalizes scale, improves conditioning
2. **Check determinant** - |det(V)| shouldn't be too large/small
3. **Monitor for NaN** - If NaN appears, learning rate too high
4. **Use stable nonlinearity** - tanh(y) more stable than y³
5. **Batch normalize** - Divide gradients by batch size

## Debugging Checklist

- [ ] Data is whitened (zero mean, unit variance)
- [ ] Shapes match: V (n_comp × n_feat), x (n_samples × n_feat)
- [ ] Gradient ∂ℓ/∂V has same shape as V
- [ ] Learning rate is reasonable (not NaN/Inf after update)
- [ ] Amari index calculated from C = V·W
- [ ] Multiple runs with different seeds
- [ ] Compare against sklearn.decomposition.FastICA

## Key References

1. **Infomax gradient**: Bell & Sejnowski (1995)
2. **FastICA**: Hyvärinen (1999)
3. **Natural Gradient**: Amari (1998)
4. **Adam Optimizer**: Kingma & Ba (2015)
5. **Amari Index**: Amari et al. (1996)

---

**Tip**: Print this card and keep it nearby while implementing!
