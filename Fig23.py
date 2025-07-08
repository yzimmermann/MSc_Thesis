"""
Example of RVO-GEK. Figure 2.3.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from GEK import GradientGPSurrogate

plt.style.use('thesis.mplstyle')

def true_energy(x):
    return 0.4 * np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x) + 0.25 * (x - 0.5) ** 2

def d_true_energy(x):
    return (0.4 * 2 * np.pi * np.cos(2 * np.pi * x)
            - 0.3 * 4 * np.pi * np.sin(4 * np.pi * x)
            + 0.5 * (x - 0.5))


# Initial noisy samples
np.random.seed(111)
x_train  = np.array([0.05, 0.2, 0.6])
y_train  = true_energy(x_train)  + np.random.normal(0, 0.1, len(x_train))
dy_train = d_true_energy(x_train) + np.random.normal(0, 0.1, len(x_train))

gek = GradientGPSurrogate(length_scale=0.15, sigma=1.0,
                          sigma_f=np.sqrt(0.1), sigma_g=np.sqrt(0.1))
gek.fit(x_train[:, None], y_train, dy_train[:, None])

x_grid   = np.linspace(0, 1, 500)[:, None]
x_plot = x_grid[:, 0]
mu0, sigma0   = gek.predict(x_grid)

epsilon          = 0.3
feasible   = sigma0 <= epsilon
idx_next   = np.argmin(mu0 + (~feasible) * 1e9) # ~ because feasible is False where sigma > epsilon
theta_next     = x_grid[idx_next]
y_next   = true_energy(theta_next)  + np.random.normal(0, 0.04)
dy_next  = d_true_energy(theta_next) + np.random.normal(0, 0.04)

x_train2  = np.append(x_train, theta_next)
y_train2  = np.append(y_train, y_next)
dy_train2 = np.append(dy_train, dy_next)
gek.fit(x_train2[:, None], y_train2, dy_train2[:, None])

mu1, sigma1 = gek.predict(x_grid)

# Nice colors thanks to https://siegal.bio.nyu.edu/color-palette/
COL_TRUE, COL_SAMP, COL_GP0, COL_GP1, COL_NEXT, COL_INF = (
    "black", "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E6E6E6"
)

fig = plt.figure(figsize=(6, 4))
gs  = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.5)

# A) initial samples
axA = fig.add_subplot(gs[0, 0])
axA.plot(x_plot, true_energy(x_plot), color=COL_TRUE, label=r"True $E(\theta)$")
axA.scatter(x_train, y_train, s=30, c=COL_SAMP, edgecolor="k", zorder=3, label="QMC samples")
axA.set_title("(A) Previous QMC evaluations")
axA.set_xlabel(r"$\theta$")
axA.set_ylabel(r"$E(\theta)$")
axA.set_xlim(0, 1)
axA.set_ylim(-1.1, 1.01)
axA.legend(frameon=False, loc="lower left")

# B) surrogate after first fit
axB = fig.add_subplot(gs[0, 1])
axB.plot(x_plot, true_energy(x_plot), color=COL_TRUE)
axB.plot(x_plot, mu0, color=COL_GP0, label="GP mean")
axB.fill_between(x_plot, mu0 - sigma0, mu0 + sigma0, color=COL_GP0, alpha=0.25, label=r"$\pm\sigma$")
axB.scatter(x_train, y_train, s=30, c=COL_SAMP, edgecolor="k", zorder=3)
axB.set_title("(B) Build surrogate")
axB.set_xlabel(r"$\theta$")
axB.set_ylabel(r"$E(\theta)$")
axB.set_xlim(0, 1)
axB.set_ylim(-1.1, 1.01)
axB.legend(frameon=False, loc="lower left")

# C) acquisition
axC = fig.add_subplot(gs[1, 0])
axC.plot(x_plot, true_energy(x_plot), color=COL_TRUE)
axC.plot(x_plot, mu0, color=COL_GP0)
axC.fill_between(x_plot, mu0 - sigma0, mu0 + sigma0, color=COL_GP0, alpha=0.25)
axC.fill_between(
    x_plot, mu0 - sigma0, mu0 + sigma0,
    where=~feasible, color=COL_INF, alpha=0.9, label=r"$\sigma(\theta)>\epsilon$"
)
axC.scatter(theta_next, mu0[idx_next], marker="*", s=120, c=COL_NEXT, edgecolor="k",
            label=r"Next $\theta_{t+1}$", zorder=100)
axC.set_title("(C) Minimize surrogate with $\\sigma^2\\leq\\epsilon$")
axC.set_xlabel(r"$\theta$")
axC.set_ylabel(r"$E(\theta)$")
axC.set_xlim(0, 1)
axC.set_ylim(-1.1, 1.01)
axC.legend(frameon=False, loc="lower left", fontsize=7)

# D) surrogate after update
axD = fig.add_subplot(gs[1, 1])
axD.plot(x_plot, true_energy(x_plot), color=COL_TRUE)
axD.plot(x_plot, mu1, color=COL_GP1, label="GP mean")
axD.fill_between(x_plot, mu1 - sigma1, mu1 + sigma1, color=COL_GP1, alpha=0.25, label=r"$\pm\sigma$")
axD.scatter(x_train2, y_train2, s=30, c=COL_SAMP, edgecolor="k", zorder=3)
axD.set_title("(D) Surrogate after update")
axD.set_xlabel(r"$\theta$")
axD.set_ylabel(r"$E(\theta)$")
axD.set_xlim(0, 1)
axD.set_ylim(-1.1, 1.01)
axD.legend(frameon=False, loc="lower left")

plt.tight_layout()
plt.savefig("", bbox_inches="tight", dpi=600)
plt.show()
