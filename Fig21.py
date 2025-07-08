"""
Kernel fan and examples of draws from different kernels. Figure 2.1.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

plt.style.use('thesis.mplstyle')

# Faste to just define the kernels here
def rbf_kernel(x, y, lengthscale=0.1, sigma2=1.0):
    r2 = (x[:, None] - y[None, :])**2
    return sigma2 * np.exp(-0.5 * r2 / lengthscale**2)

def brownian_kernel(x, y):
    return np.minimum.outer(x, y)

def matern32_kernel(x, y, lengthscale=0.2, sigma2=1.0):
    r = np.abs(x[:, None] - y[None, :])
    f = np.sqrt(3) * r / lengthscale
    return sigma2 * (1 + f) * np.exp(-f)

# for conditioning
def f_true(x):
    return 0.4*np.sin(2*np.pi*x) + 0.3*np.cos(4*np.pi*x) + 0.25*(x-0.5)**2

np.random.seed(4)
x_full  = np.linspace(0, 1, 500)
x_obs   = x_full[x_full <= 0.4]
y_obs   = f_true(x_obs)
x_pred  = x_full[x_full > 0.4]

K_oo = rbf_kernel(x_obs,  x_obs) + 1e-10*np.eye(len(x_obs))
K_po = rbf_kernel(x_pred, x_obs)
K_pp = rbf_kernel(x_pred, x_pred)

L        = la.cholesky(K_oo, lower=True)
alpha    = la.cho_solve((L, True), y_obs)
mu_pred  = K_po @ alpha
cov_pred = K_pp - K_po @ la.cho_solve((L, True), K_po.T)
std_pred = np.sqrt(np.clip(np.diag(cov_pred), 0, None))

samples = np.random.multivariate_normal(mu_pred, cov_pred, 10)

kernels = {
    r"\textit{Brownian Motion}": brownian_kernel,
    r"\textit{RBF}":      rbf_kernel,
    r"\textit{Matérn} $\nu=3/2$": matern32_kernel,
}
draws = {}
for name, kfun in kernels.items():
    K = kfun(x_full, x_full) + 1e-10*np.eye(len(x_full))
    draws[name] = np.random.multivariate_normal(np.zeros(len(x_full)), K)

fig = plt.figure(figsize=(7.5, 4))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])

axA = fig.add_subplot(gs[0, 0])
x0, y0, w, h = axA.get_position().bounds
axA.set_position([x0, y0 + 0.10*h,
                  w,  0.80*h]) # manual fix
axA.plot(x_full[x_full <= 0.4], f_true(x_full[x_full <= 0.4]), color="k", lw=1.2, label="Observed $f(x)$")
axA.plot(x_obs, y_obs, color="k", lw=2.2)

mu_full = np.concatenate([y_obs, mu_pred])
axA.plot(x_full[x_full > 0.4], mu_full[x_full > 0.4], color="#0072B2", lw=1.8,
         label=r"GP mean")

axA.fill_between(x_pred, mu_pred-std_pred, mu_pred+std_pred,
                 color="#0072B2", alpha=0.20, label=r"$\pm \sigma$")
for s in samples:
    axA.plot(x_pred, s, color="#0072B2", alpha=0.30, lw=0.9)

axA.axvline(0.4, color="k", ls="--", lw=0.8)
axA.text(-0.1, 1.15, r"(A)", transform=axA.transAxes,
         va="top", ha="left", fontsize=14)
axA.text(0.09, 2.1, r"Past", fontsize=14)
axA.text(0.6, 2.1, r"Future", fontsize=14)
axA.set_xlabel(r"$x$")
axA.set_ylabel(r"$y$")
axA.legend(frameon=False, loc="lower left")

gsR = gs[0, 1].subgridspec(3, 1, hspace=0.40)
for i, (name, y_draw) in enumerate(draws.items()):
    ax = fig.add_subplot(gsR[i, 0])
    ax.plot(x_full, y_draw, color="#D55E00", lw=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylabel(r"$y$")
    if i == 2:
        ax.set_xlabel(r"$x$")
    ax.set_title(name, fontsize=11)
    panel_letter = chr(66 + i)    # it's 66 not 65
    ax.text(-0.05, 1.2, rf"({panel_letter})",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=14)
    ax.tick_params(axis="x", labelbottom=(i == 2))

plt.savefig("", bbox_inches="tight", dpi=600)
plt.show()
