# GEK & RVO for VMC
**Gradient-Enhanced Kriging (GEK)** + **Restricted Variance Optimization (RVO)** with known/estimable noise on function values *and* gradients.

This repo contains:
- An implementation of a gradient-aware GP surrogate (`GEK.py`)
- A sequential restricted-variance optimizer (RVO-GEK)
- Minimal experiment drivers

> Parts of the system/ansatz/MC scaffolding were inspired by and adapted from  
> https://github.com/rjwebber/rgn_optimization/tree/main (for an older JAX version).

## Installation

```bash
# Python 3.10–3.12 should work
conda create -n GEK python=3.11 -y
conda activate GEK

pip install numpy scipy jax tqdm matplotlib
```
---

## Quickstart

### A) Surrogate only

Use the gradient-enhanced GP as a drop-in regressor on values + gradients.  
(This mirrors `Fig22.py`.)

```python
import numpy as np
from GEK import GradientGPSurrogate

# toy 1D function and gradient
def f(x):      return np.sin(1.2*x) + 0.15*x
def fprime(x): return 1.2*np.cos(1.2*x) + 0.15

X  = np.array([[-3.5],[0.0],[3.5]])     # (n, d)
y  = f(X).ravel()                        # (n,)
dy = fprime(X).reshape(-1, 1)            # (n, d)

gp = GradientGPSurrogate(length_scale=1.5, sigma=1.0)
gp.fit(X, y, dy)                         # optionally pass sigma_f, sigma_g

x_test = np.linspace(-5, 5, 400).reshape(-1, 1)
mu, var = gp.predict(x_test)
mu, var, dmu, dvar = gp.predict_with_grad(x_test)  # includes ∇μ and ∇Var
```

**Shapes**
- `X`: `(n_samples, n_params)`
- `y`: `(n_samples,)`
- `dy`: `(n_samples, n_params)`
- Optional noise arrays: `sigma_f -> (n,)`, `sigma_g -> (n*n_params,)` (flattened)

---

### B) Sequential RVO-GEK (outer loop)

Provide an objective that returns energy and gradient (and optionally their noise estimates).  
The runner alternates between sampling your oracle and minimizing the surrogate under a **variance constraint**.

```python
import numpy as np
from GEK import GEKRunner

# Minimal signature: return energy, grad
def my_objective(x):
    e = float(x @ x) + np.random.normal(0, 0.1)
    g = 2.0*x + np.random.normal(0, 0.01, size=x.shape)
    return e, g
    # Then, try:
    # e_var = 0.1
    # g_var = np.full_like(x, 1e-2)
    # return np.array(e), g, e_var, g_var

d  = 5
x0 = np.zeros(d)

runner = GEKRunner(length_scale=1.2, sigma=1.0)
x_opt, path, surrogate, E_path, V_path = runner.GEK_optimize(
    my_objective, x0,
    var_threshold=1.0,      # keep inner steps where GP Var(x) ≤ threshold
    method='NLC',           # 'GD' | 'BFGS' | 'NLC' (trust-constr with Var constraint)
    outer_tol=1e-7,         # stop outer loop when ||∇f|| small
    inner_tol=1e-2,         # tol for inner GD if method='GD'
    alpha=0.1,              # step size for inner GD if method='GD'
    max_iter=50,
    internal_max_iter=100,
    return_path=True,
    return_surrogate=True,
    return_energy=True,
    return_variance=True,
)
print("x* ≈", x_opt)
```

---

## API

### `GradientGPSurrogate(length_scale=1.0, sigma=1.0, sigma_f=0.0, sigma_g=0.0)`
- `.fit(X, Y, dY, sigma_f=None, sigma_g=None)`
- `.predict(X_test) -> (mu, var)`
- `.predict_with_grad(X_test) -> (mu, var, dmu, dvar)`
- `.check_kernel(X) -> np.ndarray` (augmented K block for debugging)

### `GEKRunner(length_scale=1.0, sigma=1.0, sigma_f=0.0, sigma_g=0.0)`
- `.add_data(X, Y, dY, sigma_f=None, sigma_g=None)`
- `.optimize_surrogate(x0, var_threshold, tol, alpha, max_iter)` (GD + variance gate)
- `.optimize_surrogate_BFGS(...)` (stops if variance exceeds threshold)
- `.optimize_surrogate_NonLinearConstraint(...)` (trust-constr with `Var(x) ≤ threshold`)
- `.GEK_optimize(func, x0, ..., method='GD'|'BFGS'|'NLC', ...)`

**Expected `func(x)` signatures**
- Minimal: `return energy: float, grad: np.ndarray(d,)`
- Noise-aware: `return energy, grad, energy_var: float, grad_var: np.ndarray(d,)`

---

## Reproducing Figures

- **Fig. 2.1 — Kernel fan & draws**: `Fig21.py`  
  (Brownian/RBF/Matérn samples + GP conditioning demo.)
- **Fig. 2.2 — Value-only GP vs GEK**: `Fig22.py`  
  (Small 1D example with analytic gradients.)
- **Fig. 2.3 — RVO-GEK acquisition**: `Fig23.py`  
  (Feasibility mask `σ(x) ≤ ε`, next point, and updated surrogate.)
- **Fig. 3.3 — Method comparison (NLC / GD / SR / LM / RGN)**: `Fig33.ipynb`

---

## Deterministic Experiments (Slurm helper)

`main_deterministic.py` emulates noisy VMC energy optimization (10 runs of GEK vs. noisy GD), saves arrays/plots, and tags outputs with all hyperparameters.

**Local run**
```bash
python main_deterministic.py   --d 8 --h 1.5 --alpha 5 --seed 1274   --var_threshold 1.0 --length_scale 1.2 --sigma 1.0   --sigma_f 1.0 --sigma_g 1.0 --inner_lr 0.1 --method NLC
```

**Cluster run (Slurm)**
```bash
# submit.sh
sbatch submit.sh 8 1.5 5 1274 1.0 1.2 1.0 1.0 1.0 0.1 NLC
#                d h  α seed vt   ℓ   σ  σ_f σ_g  lr  method
```

Outputs:
- `results/` — compressed arrays: paths, means/stds, and all tags
- `results_sweep_2/` — PDFs of averaged trajectories (and sweep plots used by `Fig3132.ipynb`)

---

## Monte-Carlo (VMC) Experiments (Slurm helper)

`main_MC.py` runs a true VMC loop (parallel JAX, batched sampling) and optimizes with RVO-GEK.

**Cluster run**
```bash
# run_gek_MC.sh
sbatch run_gek_MC.sh 100 1.5 5 1234 0.1 1.2 1.0 0.1 NLC 100 2000 50
#                 d   h  α seed vt  ℓ   σ  lr  meth par  T    max_iter
```

Saves tagged results (paths, energies, variances) and a trajectory plot under `results_cluster_MC_new/`.

---
