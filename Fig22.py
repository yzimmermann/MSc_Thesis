"""
Example of using the GradientGPSurrogate class from the GEK module for Fig. 2.2.
"""
import numpy as np
import matplotlib.pyplot as plt

from GEK import GradientGPSurrogate

plt.style.use('thesis.mplstyle')

def f(x):
    return np.sin(1.2 * x) + 0.15 * x

def fprime(x):
    return 1.2 * np.cos(1.2 * x) + 0.15


X_train = np.array([-3.5, 0.0, 3.5]).reshape(-1, 1)
y_train = f(X_train).ravel()
dy_train = fprime(X_train).reshape(-1, 1)


common_kwargs = dict(length_scale=1.5, sigma=1.0)

# GP on values only, big gradient noise do dY is ignored
gp_val = GradientGPSurrogate(**common_kwargs, sigma_g=1e6)
gp_val.fit(X_train, y_train, np.zeros_like(dy_train))

# Gradient-Enhanced GP
gp_gek = GradientGPSurrogate(**common_kwargs)
gp_gek.fit(X_train, y_train, dy_train)


x_test = np.linspace(-5, 5, 500).reshape(-1, 1)

mu_val,  var_val  = gp_val.predict(x_test)
mu_gek,  var_gek  = gp_gek.predict(x_test)

sigma_val = np.sqrt(np.maximum(var_val, 0))
sigma_gek = np.sqrt(np.maximum(var_gek, 0))

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

ax.plot(x_test, f(x_test), "-", label=r"True $f(x)$")

ax.scatter(X_train, y_train, marker="o", label="Training points", zorder=10)

ax.plot(x_test, mu_val, "--", label="GP")
ax.fill_between(
    x_test.ravel(),
    mu_val -sigma_val,
    mu_val + sigma_val,
    alpha=0.25,
    #label=r"$\pm \sigma$",
)

ax.plot(x_test, mu_gek, "--", label="Gradient-enhanced GP")
ax.fill_between(
    x_test.ravel(),
    mu_gek - sigma_gek,
    mu_gek + sigma_gek,
    alpha=0.25,
    #label=r"$\pm \sigma$",
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 1.5)
# ax.legend(loc="lower right")
fig.tight_layout()

#fig.savefig("", dpi=600)
plt.show()
