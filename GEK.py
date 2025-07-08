"""
Restricted Variance Optimization (RVO) and Gradient-Enhanced Kriging (GEK) with known noise levels.

Implements the RVO with gradient-enhanced GP surrogate for efficient stochastic optimization.
The surrogate supports heteroscedastic noise in both function and gradient observations.

Main Classes:
- GradientGPSurrogate: Gradient-aware GP regression using an RBF kernel.
- GEKRunner: Sequential optimizer that fits the surrogate and performs GEK-based optimization.

Dependencies: numpy, scipy, jax, tqdm (optional)
"""

import numpy as np
import scipy
from tqdm import tqdm
from jax import numpy as jnp
from jax import scipy as jsp
from jax import grad as jaxgrad
from jax import jit
from functools import partial
from scipy.optimize import NonlinearConstraint, minimize


class GradientGPSurrogate:
    """
    A gradient-enhanced Gaussian Process surrogate model using the RBF kernel.

    The model uses both function values and gradients, enabling more accurate
    interpolation and extrapolation. One can also set noise levels.
    """

    def __init__(self, length_scale=1.0, sigma=1.0, sigma_f=0.0, sigma_g=0.0):
        """
        Initialize the surrogate model with kernel parameters.

        Parameters:
            length_scale (float): Characteristic length scale of the RBF kernel.
            sigma (float): Signal variance of the RBF kernel.
            sigma_f (float or ndarray): Noise level for function observations.
            sigma_g (float or ndarray): Noise level for gradient observations.
        """
        self.l = length_scale
        self.sigma = sigma
        self.sigma_f0 = sigma_f
        self.sigma_g0 = sigma_g
        
    @partial(jit, static_argnums=0) # Do not just put @jit because it leads to error with self, so need to declare static arg
    def _rbf(self, X, Y):
        """
        Radial basis function (RBF) kernel.
        Should work for X, Y with any shape (..., d).  Returns array with shape (...).
        """
        diff2 = jnp.sum((X - Y) ** 2, axis=-1)
        return self.sigma**2 * jnp.exp(-0.5 * diff2 / self.l ** 2)

    @partial(jit, static_argnums=0)
    def _drbf(self, X, Y):
        """
        Gradient w.r.t. X of k(X,Y).
        For X, Y of shape (..., d) retruns (..., d).
        """
        k_xy = self._rbf(X, Y)[..., None]  # broadcastng
        return -(X - Y) / self.l ** 2 * k_xy

    @partial(jit, static_argnums=0)
    def _d2rbf(self, X, Y):
        """
        Cross-Hessian  del2_k/delXdelY  (shape (..., d, d)).
        """
        diff = X - Y
        outer = diff[..., :, None] * diff[..., None, :]
        k_xy = self._rbf(X, Y)[..., None, None]
        return (jnp.eye(diff.shape[-1]) / self.l**2 - outer / self.l**4) * k_xy

    def _solve_Kinv_k(self, k_star):
        return jsp.linalg.cho_solve((self.L, True), k_star)

    def _gp_mean_scalar(self, x):
        k_f = self._rbf(x, self.X_train)
        k_g = self._drbf(self.X_train, x).reshape(-1)
        k_star = jnp.concatenate([k_f, k_g])
        return k_star @ self.alpha

    def make_grad_mu(self):
        return jaxgrad(self._gp_mean_scalar)

    def check_kernel(self, X):
        n, d = X.shape
        K_ff = self._rbf(X[:, None, :], X[None, :, :])  # (n, n)

        K_gf = self._drbf(X[:, None, :], X[None, :, :])  # (n, n, d)
        K_gf = K_gf.transpose(0, 2, 1)  # FIX  (i,k,j)
        K_gf = K_gf.reshape(n * d, n)  # (n*d, n)

        K_fg = K_gf.T

        K_gg = self._d2rbf(X[:, None, :], X[None, :, :])  # (n, n, d, d)
        K_gg = K_gg.transpose(0, 2, 1, 3)  # FIX  (i,k,j,l)
        K_gg = K_gg.reshape(n * d, n * d)  # (n*d, n*d)

        K_aug = np.block([[K_ff, K_fg],
                          [K_gf, K_gg]])

        return K_aug

    def fit(self, X, Y, dY, sigma_f=None, sigma_g=None):
        """
        Fit the surrogate model using training data.

        Parameters:
            X (ndarray): Shape (n_samples, n_params). Input parameter values.
            Y (ndarray): Shape (n_samples,). Function values.
            dY (ndarray): Shape (n_samples, n_params). Gradient values.
            sigma_f (ndarray): Shape (n_samples,). Noise levels for function values.
            sigma_g (ndarray): Shape (n_samples, n_params). Noise levels for gradient values.
        """
        self.X_train = X  # (n_samples, n_params)
        dY = dY.reshape(-1)  # (n_samples * n_params,)
        n, d = X.shape

        K_ff = self._rbf(X[:, None, :], X[None, :, :])  # (n, n)

        K_gf = self._drbf(X[:, None, :], X[None, :, :])  # (n, n, d)
        K_gf = K_gf.transpose(0, 2, 1)  # FIX  (i,k,j)
        K_gf = K_gf.reshape(n * d, n)  # (n*d, n)

        K_fg = K_gf.T

        K_gg = self._d2rbf(X[:, None, :], X[None, :, :])  # (n, n, d, d)
        K_gg = K_gg.transpose(0, 2, 1, 3)  # FIX  (i,k,j,l)
        K_gg = K_gg.reshape(n * d, n * d)  # (n*d, n*d)

        sigma_f = (np.full(n, self.sigma_f0) if sigma_f is None
                   else np.asarray(sigma_f).reshape(-1))
        sigma_g = (np.full(n * d, self.sigma_g0) if sigma_g is None
                   else np.asarray(sigma_g).reshape(-1))

        if len(sigma_f) != n:
            raise ValueError("Length of sigma_f must match number of training points.")
        if len(sigma_g) != n * d:
            raise ValueError("Length of sigma_g must match number of training points times number of parameters.")

        K_ff += np.diag(sigma_f ** 2)
        K_gg += np.diag(sigma_g ** 2)

        K_aug = np.block([[K_ff, K_fg],
                          [K_gf, K_gg]])
        K_aug += 1e-8 * np.eye(n + n * d) # * np.mean(np.diag(K_aug))

        y_aug = np.concatenate([Y, dY])

        L = np.linalg.cholesky(K_aug)
        self.alpha = jsp.linalg.cho_solve((L, True), y_aug)
        self.L = L
        self.Kinv = scipy.linalg.cho_solve((L, True), np.eye(L.shape[0]))

    def predict(self, X_test):
        """
        Predict function values and variances at new input points.

        Parameters:
            X_test (ndarray): Shape (n_test, n_params). Points to predict.

        Returns:
            mu (ndarray): Predicted mean values.
            sigma2 (ndarray): Predicted variances.
        """
        mu, sigma2 = [], []

        for x in X_test:
            k_f = self._rbf (x, self.X_train)
            k_g = self._drbf(self.X_train, x).reshape(-1)
            k_star = np.concatenate([k_f, k_g])

            v    = self.Kinv @ k_star
            mean = k_star @ self.alpha
            var  = self._rbf(x, x) - k_star @ v

            mu.append(mean)
            sigma2.append(var)

        return np.array(mu), np.array(sigma2)

    def predict_with_grad(self, X_test):
        """
        Predict function values, variances, and gradients at new input points.

        Parameters:
            X_test (ndarray): Shape (n_test, n_params). Points to predict.

        Returns:
            mu (ndarray): Predicted mean values.
            sigma2 (ndarray): Predicted variances.
            dmu (ndarray): Predicted gradients of the mean.
        """
        n, d = self.X_train.shape
        mu, sigma2, dmu, dsigma2 = [], [], [], []
        # grad_mu = self.make_grad_mu()

        for x in X_test:
            # k_*
            k_f = self._rbf (x, self.X_train)
            k_g = self._drbf(self.X_train, x).reshape(-1)
            k_star = np.concatenate([k_f, k_g])

            # d/dx k(x, xi)  for each xi  -->  shape (n, d)
            dk_f = self._drbf(x, self.X_train)

            # d/dx (grad_xi k(xi, x))  =  H(xi, x)  -->  shape (n, d, d)
            dk_g_blocks = self._d2rbf(self.X_train, x)
            dk_g = dk_g_blocks.reshape(n*d, d)

            # shape (d, n+n*d)
            dk_star = np.hstack([dk_f.T, dk_g.T])

            # posterior mean / grad / var
            v    = self.Kinv @ k_star
            mean = k_star @ self.alpha
            grad = dk_star @ self.alpha
            var  = self._rbf(x, x) - k_star @ v
            dvar = -2.0 * dk_star @ v

            # print(grad)
            # print(grad_mu(x))

            mu.append(mean)
            dmu.append(grad)
            sigma2.append(var)
            dsigma2.append(dvar)

        return np.array(mu), np.array(sigma2), np.array(dmu), np.array(dsigma2)

class GEKRunner:
    """
    A class to perform restricted variance optimization (RVO) with gradient-enhanced Kriging (GEK) with known noise levels on
    function values and gradients.

    This class stores all past optimization steps and allows for the prediction of function values, variances, and gradients at new points.
    """
    def __init__(self, length_scale=1.0, sigma=1.0, sigma_f=0.0, sigma_g=0.0):
        """
        Initialize the GEKRunner with kernel parameters.

        Parameters:
            length_scale (float): Characteristic length scale of the RBF kernel.
            sigma_f (float): Noise level for function observations.
            sigma_g (float): Noise level for gradient observations.
        """
        self.surrogate = GradientGPSurrogate(length_scale, sigma, sigma_f, sigma_g)
        self.X_train = []
        self.Y_train = []
        self.dY_train = []
        self.sigma_f_arr = np.empty(0)
        self.sigma_g_arr = np.empty((0,))

    def add_data(self, X, Y, dY, sigma_f=None, sigma_g=None):
        """
        Add one or more new training points with or without known noise levels.
        Parameters:
            X (ndarray): Shape (n_samples, n_params). New input parameter values.
            Y (ndarray): Shape (n_samples,). New function values.
            dY (ndarray): Shape (n_samples, n_params). New gradient values.
            sigma_f (ndarray): Shape (n_samples,). Noise levels for function values.
            sigma_g (ndarray): Shape (n_samples, n_params). Noise levels for gradient values.
        """
        if Y.ndim == 0:
            Y = np.array([Y])
        # Sanity check
        if X.ndim != 2 or Y.ndim !=1 or dY.ndim != 2:
            raise ValueError("X must be 2D, Y must be 1D, and dY must be 2D.")
        if X.shape[0] != Y.shape[0] or X.shape[0] != dY.shape[0]:
            raise ValueError("X, Y, and dY must have the same number of samples.")
        if X.shape[1] != dY.shape[1]:
            raise ValueError("X and dY must have the same number of parameters.")

        sigma_f = (np.full(X.shape[0], self.surrogate.sigma_f0)
                   if sigma_f is None else np.asarray(sigma_f).reshape(-1))
        sigma_g = (np.full(X.size, self.surrogate.sigma_g0)
                   if sigma_g is None else np.asarray(sigma_g).reshape(-1))

        self.sigma_f_arr = np.concatenate([self.sigma_f_arr, sigma_f])
        self.sigma_g_arr = np.concatenate([self.sigma_g_arr, sigma_g])

        self.X_train.append(X)
        self.Y_train.append(Y)
        self.dY_train.append(dY)

        X_all = np.vstack(self.X_train)
        Y_all = np.concatenate(self.Y_train)
        dY_all = np.vstack(self.dY_train)

        self.surrogate.fit(X_all, Y_all, dY_all,
                       sigma_f=self.sigma_f_arr,
                       sigma_g=self.sigma_g_arr)

    def optimize_surrogate(self, x0, var_threshold=10, tol=1e-9, alpha=0.01, max_iter=100):
        """
        Restricted-variance-ptimize the surrogate model using gradient descent.
        Parameters:
            x0 (ndarray): Initial point for optimization.
            var_threshold (float): Variance threshold.
            tol (float): Tolerance for convergence.
            alpha (float): Step size for gradient descent.
            max_iter (int): Maximum number of iterations.
        Returns:
            x_opt (ndarray): Optimized point.
        """
        x_current = x0.copy()
        steps = [x_current.copy()]
        for _ in range(max_iter):
            mean, sigma, grad, trash = self.surrogate.predict_with_grad(x_current.reshape(1, -1))
            grad = grad.flatten()
            x_new = x_current - alpha * grad
            mean, var = self.surrogate.predict(x_new.reshape(1, -1))
            #if _ % 1 == 0:
            #    print(_, np.linalg.norm(grad), mean, var)
            if len(steps) > 1:
                if var[0] > var_threshold:
                    x_current = steps[-1].copy()
                    print(f"Converged at internal step {_} because of variance: {var[0]}")
                    break
                if np.linalg.norm(grad) < tol:
                    print(f"Converged at internal step {_} because of gradient norm: {np.linalg.norm(grad)}")
                    break
            x_current = x_new
            steps.append(x_current.copy())
        return x_current

    def optimize_surrogate_BFGS(self, x0, *, var_threshold=10.0, tol=1e-9, max_iter=100):
        """
        BFGS RVO optimisation of the surrogate.
        """
        # cache dict keys: "x", "mu", "var", "grad"
        cache = {}

        def eval_gp(x):
            """
            Query GP only when x differs from cache.
            """
            if cache.get("x") is None or not np.array_equal(x, cache["x"]):
                mu, var, grad, _ = self.surrogate.predict_with_grad(x.reshape(1, -1))
                cache.update(x=x.copy(), mu=float(mu[0]), var=float(var[0]), grad=grad.flatten())
            return cache["mu"], cache["var"], cache["grad"]

        # objective, gradient, callback
        def fun(x):
            mu, _, _ = eval_gp(x)
            return mu

        def jac(x):
            _, _, grad = eval_gp(x)
            return grad

        safe_x = x0.copy()
        iter_count = 0

        def cb(xk):
            nonlocal iter_count, safe_x
            iter_count += 1
            mu, var, grad = eval_gp(xk)  # from caache

            g_norm = np.linalg.norm(grad)

            print(iter_count - 1, g_norm, mu, [var])

            if var > var_threshold:
                print(f"Converged at internal step {iter_count - 1} "
                      f"because variance exceeded threshold: {var:.3g}")
                raise StopIteration
            safe_x = xk.copy()

        # run BFGS                                                           #
        try:
            res = minimize(fun,
                           x0,
                           jac=jac,
                           method="BFGS",
                           tol=tol,
                           options={"maxiter": max_iter, "disp": False},
                           callback=cb)
            return res.x
        except StopIteration:
            return safe_x

    def optimize_surrogate_NonLinearConstraint(self, x0, *, var_threshold=10.0, tol=1e-9, max_iter=100):
        """
        Variance-constrained optimisation of the surrogate with NLC.
        """

        cache = {}

        def eval_gp(x: np.ndarray):
            """
            Query GP only when x differs from what is in the cache.
            Returns (mu, var, grad_mu, grad_var)
            """
            if cache.get("x") is None or not np.array_equal(x, cache["x"]):
                mu, var, grad_mu, grad_var = self.surrogate.predict_with_grad(
                    x.reshape(1, -1)
                )
                cache.update(
                    x=x.copy(),
                    mu=float(mu[0]),
                    var=float(var[0]),
                    grad=grad_mu.flatten(),
                    dvar=grad_var
                )
            return cache["mu"], cache["var"], cache["grad"], cache["dvar"]

        def fun(x):
            mu, _, _, _ = eval_gp(x)
            return mu

        def jac(x):
            _, _, grad_mu, _ = eval_gp(x)
            return grad_mu

        def var_fun(x):
            _, var, _, _ = eval_gp(x)
            return var  # constraint:  var(x)  <=  var_threshold

        def var_jac(x):
            _, _, _, dvar = eval_gp(x)
            return dvar

        nlc = NonlinearConstraint(
            fun=var_fun,
            lb=-np.inf,
            ub=var_threshold,
            jac=var_jac,
        )

        res = minimize(
            fun,
            x0,
            method="trust-constr",
            jac=jac,
            constraints=[nlc],
            tol=tol,
            options={
                "maxiter": max_iter,
                "verbose": 2,
            },
        )

        return res.x

    def GEK_optimize(self, func, x0, var_threshold=10, outer_tol=1e-9, inner_tol=1e-9, alpha=0.01, max_iter=100, internal_max_iter=1000, method='GD', return_path=False, return_surrogate=False, return_energy=False, return_variance=False, verbose=False):
        """
        Optimize the objective function by iteratively updating the surrogate model and optimizing the objective function.
        Parameters:
            func (function): Function to be optimized.
            x0 (ndarray): Initial point for optimization.
            var_threshold (float): Variance threshold for stopping criteria.
            inner_tol (float): Tolerance for convergence in outer loop.
            outer_tol (float): Tolerance for convergence in inner loop.
            alpha (float): Step size for gradient descent.
            max_iter (int): Maximum number of iterations.
            method (str): Optimization method. Currently, one of {'GD', 'BFGS', 'NLC'}.
            return_path (bool): If True, return the optimization path.
            return_surrogate (bool): If True, return the surrogate model.
            return_energy (bool): If True, return the energy path.
            return_variance (bool): If True, return the variance path.
            verbose (bool): If True, print current energy and variance after each iteration.
        Returns:
            x_opt (ndarray): Optimized point.
            path (ndarray): Optimization path (if return_path is True).
            surrogate (GradientGPSurrogate): Surrogate model (if return_surrogate is True).
            energy_path (ndarray): Energy path (if return_energy is True).
            variance_path (ndarray): Variance path (if return_variance is True).
        """
        x_current = x0.copy()
        path = [x_current.copy()]
        # get initial function value and gradient
        y_current, grad_current, *rest = func(x_current)
        energy_path = [y_current]

        if len(rest) > 0:
            y_var = rest[0]
            grad_var = rest[-1]
            y_sigma = np.sqrt(y_var)
            grad_sigma = np.sqrt(grad_var)
        else:
            y_var = None
            grad_var = None
            y_sigma = None
            grad_sigma = None

        variance_path = [y_var]

        self.add_data(x_current.reshape(1, -1), y_current, grad_current.reshape(1, -1), y_sigma, grad_sigma)
        for _ in tqdm(range(max_iter)):

            if method == 'GD':
                x_current = self.optimize_surrogate(x_current, var_threshold=var_threshold, tol=inner_tol, alpha=alpha, max_iter=internal_max_iter)
            elif method == 'BFGS':
                x_current = self.optimize_surrogate_BFGS(x_current, var_threshold=var_threshold, tol=outer_tol, max_iter=internal_max_iter)
            elif method == 'NLC':
                x_current = self.optimize_surrogate_NonLinearConstraint(x_current, var_threshold=var_threshold, tol=outer_tol, max_iter=internal_max_iter)
            else:
                raise ValueError(f"Unknown method: {method}")

            y_current, grad_current, *rest = func(x_current)
            if verbose == True:
                print("Energy: ", y_current)

            if len(rest) > 0:
                y_var = rest[0]
                grad_var = rest[-1]
                y_sigma = np.sqrt(y_var)
                grad_sigma = np.sqrt(grad_var)
                if verbose == True:
                    print("Energy Variance: ", y_var)
            else:
                y_var = None
                grad_var = None
                y_sigma = None
                grad_sigma = None

            self.add_data(x_current.reshape(1, -1), y_current, grad_current.reshape(1, -1), y_sigma, grad_sigma)
            path.append(x_current.copy())
            energy_path.append(y_current)
            variance_path.append(y_var)
            if np.linalg.norm(grad_current) < outer_tol:
                print("Converged early at iteration:", _)
                break

        # Conditional return of path and surrogate and rest
        if return_path:
            path = np.array(path)
        else:
            path = None
        if return_surrogate:
            surrogate = self.surrogate
        else:
            surrogate = None
        if return_energy:
            energy_path = np.array(energy_path)
        else:
            energy_path = None
        if return_variance:
            variance_path = np.array(variance_path)
        else:
            variance_path = None

        return x_current, path, surrogate, energy_path, variance_path
