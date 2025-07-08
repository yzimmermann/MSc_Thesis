"""
Emulated VMC energy optimization using RVO-GEK and baseline GD.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
import jax.numpy as jnp
import jax.random as random
import scipy.sparse
from jax import grad, hessian, jit, vmap
from GEK import GEKRunner

jax.config.update("jax_enable_x64", True)

plt.style.use('thesis.mplstyle')

# ===============
# Argument parser
# ===============
parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=8)
parser.add_argument("--h", type=float, default=1.5)
parser.add_argument("--alpha", type=int, default=5)
parser.add_argument("--seed", type=int, default=1274)
parser.add_argument("--var_threshold", type=float, default=1.0)
parser.add_argument("--length_scale", type=float, default=1.2)
parser.add_argument("--sigma", type=float, default=1.0)
parser.add_argument("--sigma_f", type=float, default=1.0)
parser.add_argument("--sigma_g", type=float, default=1.0)
parser.add_argument("--inner_lr", type=float, default=0.1)
parser.add_argument("--method", type=str, default="NLC")
args = parser.parse_args()

# ============
# System setup
# ============
np.random.seed(args.seed)
d, h, alpha = args.d, args.h, args.alpha
key = random.PRNGKey(args.seed)
key, key1, key2 = random.split(key, num=3)
weights_save = 0.001 * random.normal(key1, shape=(alpha * (d + 1),)) + \
               0.001j * random.normal(key2, shape=(alpha * (d + 1),))

configs = jnp.arange(2 ** d)[:, None] >> jnp.arange(d)[::-1] & 1
configs = configs.astype(jnp.bool_)
diffs = configs ^ configs[:, (jnp.arange(d) + 1) % d]
spin_z = 2 * jnp.sum(diffs, axis=1) - d

spin_x = jnp.zeros(configs.shape, dtype=jnp.int_)
for i in range(d):
    new = configs.at[:, i].set(~configs[:, i])
    new = jnp.dot(new, 2 ** jnp.arange(d)[::-1])
    spin_x = spin_x.at[:, i].set(new)
i_vals = jnp.repeat(jnp.arange(2 ** d), d)
j_vals = jnp.ravel(spin_x)
data = jnp.append(spin_z, jnp.repeat(-h, i_vals.size))
i_vals = jnp.append(jnp.arange(2 ** d), i_vals)
j_vals = jnp.append(jnp.arange(2 ** d), j_vals)
matrix = scipy.sparse.coo_matrix((data, (i_vals, j_vals))).toarray()

subset1 = (2 * jnp.sum(configs, axis=1) + configs[:, 0] <= d)
subset1 = jnp.arange(2 ** d)[subset1]
subset2 = jnp.dot(configs[subset1, :], 2 ** jnp.arange(d)[::-1])
subset2 = 2 ** d - 1 - subset2
matrix = matrix[subset1[:, None], subset1] + matrix[subset1[:, None], subset2]
configs = configs[subset1, :]

q = np.arange(1 / d, 1, 2 / d)
soln = -np.mean(np.sqrt(1 + h ** 2 + 2 * h * np.cos(np.pi * q)))

# =====================
# Ansatz and gradient
# =====================
@jit
def ansatz(state, features2, bias):
    state2 = jnp.fft.fft(state)
    angles = jnp.fft.ifft(features2 * jnp.conj(state2)) + bias
    return jnp.sum(jnp.log(jnp.cosh(angles)))

ansatz1 = vmap(ansatz, (0, None, None), 0)
jansatz = jit(ansatz1)

@jit
def gradient(state, features2, bias):
    state2 = jnp.fft.fft(state)
    angles = jnp.fft.ifft(features2 * jnp.conj(state2)) + bias
    y = jnp.tanh(angles)
    grad_bias = jnp.sum(y, axis=-1)
    y2 = jnp.fft.fft(y)
    grad_features = jnp.fft.ifft(y2 * state2)
    return grad_features, grad_bias

gradient1 = vmap(gradient, (0, None, None), (0, 0))
jgradient = jit(gradient1)

@jit
def rayleigh(weights):
    bias = jnp.reshape(weights[-alpha:], (alpha, 1))
    features = jnp.reshape(weights[:-alpha], (alpha, d))
    features2 = jnp.fft.fft(features)
    y = jnp.exp(jansatz(configs, features2, bias))
    return jnp.real(jnp.vdot(y, jnp.dot(matrix, y)) / jnp.vdot(y, y))

@jit
def objective(weights):
    weights = weights[:alpha * (d + 1)] + 1j * weights[alpha * (d + 1):]
    bias = jnp.reshape(weights[-alpha:], (alpha, 1))
    features = jnp.reshape(weights[:-alpha], (alpha, d))
    features2 = jnp.fft.fft(features)
    configs2 = jnp.fft.fft(configs, axis=1)
    angles = features2[jnp.newaxis, :, :] * jnp.conj(configs2)[:, jnp.newaxis, :]
    angles = jnp.fft.ifft(angles, axis=-1) + bias[jnp.newaxis, :, :]
    y = jnp.exp(jnp.sum(jnp.log(jnp.cosh(angles)), axis=(1, 2)))
    return jnp.real(jnp.vdot(y, jnp.dot(matrix, y)) / jnp.vdot(y, y))

hess = jit(hessian(objective))
grad_energy = jit(grad(objective))

w0_real = np.concatenate([np.real(weights_save), np.imag(weights_save)])

def energy_and_grad(w):
    return objective(w), grad_energy(w)

def noisy_energy_and_grad(w):
    return objective(w) + np.random.normal(scale=args.sigma_f), grad_energy(w) + np.random.normal(scale=args.sigma_g, size=w.shape)

# ==============
# Run optimizer
# ==============
NLC_paths = []
GD_paths = []

for _ in range(10):
    runner = GEKRunner(length_scale=args.length_scale, sigma=args.sigma, sigma_f=args.sigma_f, sigma_g=args.sigma_g)
    x_opt, path, surrogate = runner.GEK_optimize(
        noisy_energy_and_grad, w0_real,
        var_threshold=args.var_threshold,
        outer_tol=1e-7,
        inner_tol=1e-2,
        alpha=args.inner_lr,
        max_iter=50,
        internal_max_iter=100,
        method=args.method,
        return_path=True,
        return_surrogate=True
    )
    NLC_paths.append(path)

for _ in range(10):
    iterations = 50
    gd_path = [w0_real]
    for i in range(iterations):
        _, grad = noisy_energy_and_grad(gd_path[i])
        gd_path.append(gd_path[i] - grad * 0.03)
    GD_paths.append(gd_path)

energy_paths_NLC = [[objective(point) for point in path] for path in NLC_paths]
energy_paths_GD = [[objective(point) for point in path] for path in GD_paths]

# ==============
# Save results
# ==============
def stack_paths(paths):
    min_len = min(len(p) for p in paths)
    return np.array([p[:min_len] for p in paths])

NLC_arr = stack_paths(energy_paths_NLC)
GD_arr = stack_paths(energy_paths_GD)

NLC_mu, NLC_sigma = NLC_arr.mean(0), NLC_arr.std(0)
GD_mu, GD_sigma = GD_arr.mean(0), GD_arr.std(0)

fig, ax = plt.subplots(figsize=(3.4, 2.2))
colors = mpl.colormaps['tab10'].colors
ax.plot(NLC_mu, color=colors[0], label=f'{args.method} (mean)')
ax.fill_between(range(len(NLC_mu)), NLC_mu - NLC_sigma, NLC_mu + NLC_sigma, color=colors[0], alpha=0.2)
ax.plot(GD_mu, color=colors[1], label=f'GD (mean)')
ax.fill_between(range(len(GD_mu)), GD_mu - GD_sigma, GD_mu + GD_sigma, color=colors[1], alpha=0.2)
ax.axhline(soln*d, linestyle='--', color='red', label='Exact')
ax.set_xlabel("Iteration")
ax.set_ylabel("Energy")
ax.legend()
fig.tight_layout()

os.makedirs("results_sweep_2", exist_ok=True)
tag = f"d{d}_h{h}_alpha{alpha}_vt{args.var_threshold}_ls{args.length_scale}_sig{args.sigma}_sf{args.sigma_f}_sg{args.sigma_g}_meth{args.method}_seed{args.seed}"
plt.savefig(f"results_sweep_2/average_trajectories_{tag}.pdf", dpi=600)

np.savez_compressed(
    f"results/run_data_{tag}.npz",
    NLC_paths=NLC_paths,
    GD_paths=GD_paths,
    energy_paths_NLC=energy_paths_NLC,
    energy_paths_GD=energy_paths_GD,
    NLC_mean=NLC_mu,
    NLC_std=NLC_sigma,
    GD_mean=GD_mu,
    GD_std=GD_sigma,
    d=d, h=h, alpha=alpha, seed=args.seed,
    var_threshold=args.var_threshold,
    length_scale=args.length_scale,
    sigma=args.sigma,
    sigma_f=args.sigma_f,
    sigma_g=args.sigma_g,
    method=args.method
)
print(f"Done. Saved results under : {tag}")
