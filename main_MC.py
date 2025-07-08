"""
Actual VMC optimization with RVO-GEK. (Not used in the end)
"""

import os

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=16 "
    "--xla_cpu_multi_thread_eigen=true "
    "intra_op_parallelism_threads=1 "
    "inter_op_parallelism_threads=1"
)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap, pmap
from GEK import GEKRunner

# Enable 64-bit
jax.config.update("jax_enable_x64", True)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, default=100)
parser.add_argument("--h", type=float, default=1.5)
parser.add_argument("--alpha", type=int, default=5)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--var_threshold", type=float, default=0.1)
parser.add_argument("--length_scale", type=float, default=1.2)
parser.add_argument("--sigma", type=float, default=1.0)
parser.add_argument("--inner_lr", type=float, default=0.1)
parser.add_argument("--method", type=str, default="NLC")
parser.add_argument("--max_iter", type=int, default=50)
parser.add_argument("--internal_max_iter", type=int, default=100)
parser.add_argument("--parallel", type=int, default=100)
parser.add_argument("--T", type=int, default=2000)
args = parser.parse_args()

print("Using arguments...")
for k, v in vars(args).items():
    print(f"  {k}: {v}")
    
print("JAX Devices:", jax.devices())

# =================
# Ansatz + gradient
# =================

@jit
def ansatz(state, features2, bias):
    state2 = jnp.fft.fft(state)
    angles = jnp.fft.ifft(features2 * jnp.conj(state2)) + bias
    log_wave = jnp.sum(jnp.log(jnp.cosh(angles)))
    return (log_wave)


ansatz1 = vmap(ansatz, (0, None, None), 0)
jansatz = jit(ansatz1)


@jit
def gradient(state, features2, bias):
    # derivative of log-wavefunction
    state2 = jnp.fft.fft(state)
    angles = jnp.fft.ifft(features2 * jnp.conj(state2)) + bias
    y = jnp.tanh(angles)
    grad_bias = jnp.sum(y, axis=-1)
    y2 = jnp.fft.fft(y)
    grad_features = jnp.fft.ifft(y2 * state2)
    return (grad_features, grad_bias)


gradient1 = vmap(gradient, (0, None, None), (0, 0))
jgradient = jit(gradient1)

# ==============
# Local energies
# ==============

@jit
def e_locs(states, features2, bias):
    # gradients
    g1, g2 = jgradient(states, features2, bias)
    grads = jnp.column_stack((jnp.reshape(g1, (args.parallel, -1)), g2))
    # spin-z
    diffs = states ^ states[:, (jnp.arange(args.d) + 1) % args.d]
    energies = 2 * jnp.sum(diffs, axis = -1) - args.d
    # neighbor states
    states2 = jnp.expand_dims(states, axis=-2)
    states2 = jnp.repeat(states2, args.d + 1, axis=-2)
    i0 = jnp.arange(args.d)
    states2 = states2.at[:, i0, i0].set(~states2[:, i0, i0])

    # flip neighbors
    flips = (2 * jnp.sum(states2, axis=-1) + states2[..., 0] > args.d)
    states2 = states2 ^ jnp.expand_dims(flips, -1)
    # fast evaluation
    locs = jloc(states2, energies, features2, bias)
    return (grads, locs)

@jit
def loc(states, energy, features2, bias):
    log_waves = jansatz(states, features2, bias)
    wave_ratios = jnp.exp(log_waves[:-1] - log_waves[-1])
    local = energy - args.h * jnp.sum(wave_ratios)
    return (local)


loc1 = vmap(loc, (0, 0, None, None), 0)

jloc = jit(loc1)


@jit
def e_locs_more(states, features2, bias):
    # spin-z
    diffs = states ^ states[:, (jnp.arange(args.d) + 1) % args.d]
    energies = 2 * jnp.sum(diffs, axis = -1) - args.d
    # neighbor states
    states2 = jnp.expand_dims(states, axis=-2)
    states2 = jnp.repeat(states2, args.d + 1, axis=-2)
    i0 = jnp.arange(args.d)
    states2 = states2.at[:, i0, i0].set(~states2[:, i0, i0])

    # flip neighbors
    flips = (2 * jnp.sum(states2, axis=-1) + states2[..., 0] > args.d)
    states2 = states2 ^ jnp.expand_dims(flips, -1)
    # fast evaluation
    g1, g2, locs, h1, h2 = jmore(states2, energies, features2, bias)
    grads = jnp.column_stack((jnp.reshape(g1, (args.parallel, -1)), g2))
    hams = jnp.column_stack((jnp.reshape(h1, (args.parallel, -1)), h2))
    return (grads, locs, hams)

@jit
def more(states, energy, features2, bias):
    # local energy
    log_waves = jansatz(states, features2, bias)
    wave_ratios = jnp.exp(log_waves[:-1] - log_waves[-1])
    local = energy - args.h * jnp.sum(wave_ratios)
    # gradients
    g1, g2 = jgradient(states, features2, bias)
    # local gradient energy
    loc1 = jnp.tensordot(wave_ratios, g1[:-1, ...], axes=(0, 0))
    loc2 = jnp.tensordot(wave_ratios, g2[:-1, ...], axes=(0, 0))
    loc1 = energy * g1[-1, ...] - args.h * loc1
    loc2 = energy * g2[-1, ...] - args.h * loc2
    return (g1[-1, ...], g2[-1, ...], local, loc1, loc2)


more1 = vmap(more, (0, 0, None, None), (0, 0, 0, 0, 0))
jmore = jit(more1)


# ========
# Sampling
# ========

@jit
def get_data(states, key, weights):
    # initialize
    bias = jnp.reshape(weights[-args.alpha:], (args.alpha, 1))
    features = jnp.reshape(weights[:-args.alpha], (args.alpha, args.d))
    features2 = jnp.fft.fft(features)
    currents = jansatz(states, features2, bias)
    # generate data
    (states, currents, key, _, _), (store_grad, store_energy) = \
        jax.lax.scan(sample_less, (states, currents, key, features2, bias), None, args.T)
    return (states, key, store_grad, store_energy)


# Parallelizes the get_data function across multiple devices
parallel_data = pmap(get_data, in_axes=(0, 0, None), out_axes=(0, 0, 0, 0))


@jit
def get_more_data(states, key, weights):
    # initialize
    bias = jnp.reshape(weights[-args.alpha:], (args.alpha, 1))
    features = jnp.reshape(weights[:-args.alpha], (args.alpha, args.d))
    features2 = jnp.fft.fft(features)
    currents = jansatz(states, features2, bias)
    # generate data
    (states, currents, key, _, _), (store_grad, store_energy, store_ham) = \
        jax.lax.scan(sample_more, (states, currents, key, features2, bias), None, args.T)
    return (states, key, store_grad, store_energy, store_ham)


parallel_more_data = pmap(get_more_data, in_axes=(0, 0, None), out_axes=(0, 0, 0, 0, 0))


@jit
def sample_more(inputs, i):
    (states, currents, key, features2, bias), _ = \
        jax.lax.scan(update, inputs, None, args.d)
    grads, energies, hams = e_locs_more(states, features2, bias)
    return (states, currents, key, features2, bias), \
        (grads, energies, hams)


@jit
def sample_less(inputs, i):
    (states, currents, key, features2, bias), _ = \
        jax.lax.scan(update, inputs, None, args.d)
    grads, locs = e_locs(states, features2, bias)
    return (states, currents, key, features2, bias), \
        (grads, locs)

@jit
def update(inputs, i):
    (states, currents, key, features2, bias) = inputs
    key, key1, key2, key3, key4 = random.split(key, num=5)

    # update
    i0 = jnp.arange(args.parallel)
    i1 = random.choice(key1, args.d, shape = (args.parallel,))
    perturbs = states.at[i0, i1].set(~states[i0, i1])

    # flip spins
    flips = (2 * jnp.sum(perturbs, axis=-1) + perturbs[..., 0] > args.d)
    perturbs = perturbs ^ jnp.expand_dims(flips, -1)
    # accept or reject moves
    futures = jnp.real(jansatz(perturbs, features2, bias))
    accept = random.exponential(key2, shape=(args.parallel,))
    accept = (futures - currents) > -.5 * accept
    accept2 = jnp.broadcast_to(accept[:, jnp.newaxis], (args.parallel, args.d))
    # update information
    currents = jnp.where(accept, futures, currents)
    states = jnp.where(accept2, perturbs, states)

    return (states, currents, key, features2, bias), None

def actually_noisy(states, key, weights):
    (states, key, store_grad, store_energy) = parallel_data(states, key, weights)
    store_grad   = jnp.reshape(store_grad,  (-1, args.alpha * (args.d + 1)))
    store_energy = jnp.reshape(store_energy, (-1,))

    energy       = jnp.mean(store_energy)
    e_centered   = store_energy - energy
    energy_var   = jnp.mean(jnp.abs(e_centered) ** 2) / store_energy.size

    g_centered   = store_grad - jnp.mean(store_grad, axis=0)
    grad         = jnp.dot(g_centered.conj().T, e_centered) / e_centered.size

    per_sample_g = g_centered * e_centered[:, None]
    grad_var     = jnp.mean(jnp.abs(per_sample_g - grad) ** 2, axis=0) / store_energy.size

    return (states, key), (energy, grad, energy_var, grad_var)

def real2complex(w_real):
    half = w_real.size // 2
    return w_real[:half] + 1j * w_real[half:]

def complex2real(w_complex):
    return jnp.concatenate([jnp.real(w_complex), jnp.imag(w_complex)])

plt.style.use('thesis.mplstyle')

class NoisyObjective:
    """
    Helper class to not have to pass states and key every time and to do real and complex handling.
    """
    def __init__(self, states0, key0):
        self._carry = (states0, key0)

    def __call__(self, weights_real):
        weights_complex = real2complex(weights_real)
        self._carry, (energy, grad_complex, energy_var, grad_var) = actually_noisy(*self._carry, weights_complex)
        # safety
        energy = jnp.real(energy)
        energy_var = jnp.real(energy_var)
        # conversion
        grad_real = complex2real(grad_complex)
        grad_var_real = jnp.concatenate([jnp.real(grad_var), jnp.real(grad_var)])
        return energy, jax.device_get(grad_real), energy_var, jax.device_get(grad_var_real) # To fix weird error with JAX

def main():
    # Setup
    np.random.seed(args.seed)
    key = random.PRNGKey(args.seed)
    key, key1, key2, key3 = random.split(key, 4)
    key3 = random.split(key3, args.parallel * jax.local_device_count())
    key_save = random.split(key, jax.local_device_count())
    
    weights_save = 0.001 * random.normal(key1, shape=(args.alpha * (args.d + 1),)) \
                 + 0.001j * random.normal(key2, shape=(args.alpha * (args.d + 1),))
                 
    w0_real = complex2real(weights_save)

    states_old = jnp.tile(jnp.array([True, False]), args.d // 2)
    states_save = jax.vmap(random.permutation, in_axes=(0, None))(key3, states_old)
    states_save = states_save.at[:, ::2].set(~states_save[:, ::2])
    flips = (2 * jnp.sum(states_save, axis=-1) + states_save[..., 0] > args.d)
    states_save = states_save ^ jnp.expand_dims(flips, -1)
    states_save = jnp.reshape(states_save, (jax.local_device_count(), args.parallel, -1))

    # Optimize
    key = jnp.array(key_save)
    states = jnp.array(states_save)
    func = NoisyObjective(states, key)
    runner = GEKRunner(length_scale=args.length_scale, sigma=args.sigma)
    print("Starting Optimization...")
    x_opt, path, surrogate, energy_path, variance_path = runner.GEK_optimize(
        func, w0_real,
        var_threshold=args.var_threshold,
        outer_tol=1e-7,
        inner_tol=1e-2,
        alpha=args.inner_lr,
        max_iter=args.max_iter,
        internal_max_iter=args.internal_max_iter,
        method=args.method,
        return_path=True,
        return_surrogate=True,
        return_energy=True,
        return_variance=True
    )

    # Save results
    os.makedirs("results_cluster_MC_new", exist_ok=True)
    tag = f"d{args.d}_h{args.h}_a{args.alpha}_vt{args.var_threshold}_ls{args.length_scale}_sig{args.sigma}_m{args.method}_s{args.seed}"
    np.savez_compressed(f"results_cluster_MC_new/gek_opt_{tag}.npz",
                        x_opt=x_opt, path=path,
                        energy_path=energy_path, variance_path=variance_path)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 3))
    colors = colormaps['tab10'].colors
    ax.plot(energy_path, label="GEK Energy", color=colors[0])
    
    """
    ax.fill_between(range(len(energy_path)),
                    np.array(energy_path) - np.array(variance_path),
                    np.array(energy_path) + np.array(variance_path),
                    alpha=0.2, color=colors[0])
    """
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"results_cluster_MC_new/energy_trajectory_{tag}.pdf", dpi=300)
    print(f"Finished. Results and plot saved under tag: {tag}")

if __name__ == "__main__":
    main()
