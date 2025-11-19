"""
Sparse Gradient-Enhanced Kriging (Sparse GEK) with inducing points.

This module implements a sparse approximation to GEK using inducing points,
which reduces computational complexity from O(n³) to O(m²n) where m << n.

The inducing points are intelligently selected from low-variance RVO iterations,
as these represent well-explored regions of the parameter space.

Main Classes:
- SparseGradientGPSurrogate: Sparse GP with inducing points
- SparseGEKRunner: RVO optimizer using sparse GP surrogate
"""

import numpy as np
import scipy
from jax import numpy as jnp
from jax import scipy as jsp
from jax import grad as jaxgrad
from jax import jit
from functools import partial
from scipy.optimize import NonlinearConstraint, minimize
from tqdm import tqdm


class SparseGradientGPSurrogate:
    """
    A sparse gradient-enhanced Gaussian Process using inducing points.
    
    Uses FITC (Fully Independent Training Conditional) approximation to
    reduce complexity from O(n³) to O(m²n) where m is number of inducing points.
    """
    
    def __init__(self, length_scale=1.0, sigma=1.0, sigma_f=0.0, sigma_g=0.0, 
                 n_inducing=10, inducing_strategy='auto'):
        """
        Initialize the sparse surrogate model.
        
        Parameters:
            length_scale (float): Characteristic length scale of the RBF kernel.
            sigma (float): Signal variance of the RBF kernel.
            sigma_f (float or ndarray): Noise level for function observations.
            sigma_g (float or ndarray): Noise level for gradient observations.
            n_inducing (int): Maximum number of inducing points.
            inducing_strategy (str): Strategy for selecting inducing points:
                - 'auto': Automatically select from low-variance points
                - 'random': Random subset
                - 'kmeans': K-means clustering
        """
        self.l = length_scale
        self.sigma = sigma
        self.sigma_f0 = sigma_f
        self.sigma_g0 = sigma_g
        self.n_inducing = n_inducing
        self.inducing_strategy = inducing_strategy
        
        # Will be set during fit
        self.X_inducing = None
        self.inducing_indices = None
        
    @partial(jit, static_argnums=0)
    def _rbf(self, X, Y):
        """Radial basis function (RBF) kernel."""
        diff2 = jnp.sum((X - Y) ** 2, axis=-1)
        return self.sigma**2 * jnp.exp(-0.5 * diff2 / self.l ** 2)
    
    @partial(jit, static_argnums=0)
    def _drbf(self, X, Y):
        """Gradient w.r.t. X of k(X,Y)."""
        k_xy = self._rbf(X, Y)[..., None]
        return -(X - Y) / self.l ** 2 * k_xy
    
    @partial(jit, static_argnums=0)
    def _d2rbf(self, X, Y):
        """Cross-Hessian del2_k/delXdelY."""
        diff = X - Y
        outer = diff[..., :, None] * diff[..., None, :]
        k_xy = self._rbf(X, Y)[..., None, None]
        return (jnp.eye(diff.shape[-1]) / self.l**2 - outer / self.l**4) * k_xy
    
    def _select_inducing_points(self, X_train, variance_history=None):
        """
        Select inducing points from training data.
        
        Parameters:
            X_train: All training points (n, d)
            variance_history: Optional variance values for each training point
            
        Returns:
            indices: Indices of selected inducing points
        """
        n = X_train.shape[0]
        m = min(self.n_inducing, n)
        
        if self.inducing_strategy == 'auto' and variance_history is not None:
            # Select points with lowest variance (most confident predictions)
            # These are typically from well-explored regions in RVO
            indices = np.argsort(variance_history)[:m]
        elif self.inducing_strategy == 'kmeans':
            # Use k-means clustering (not implemented yet, fall back to random)
            indices = np.random.choice(n, m, replace=False)
        else:
            # Random or fallback
            # Ensure we have diverse coverage
            step = max(1, n // m)
            indices = np.arange(0, n, step)[:m]
            
        return indices
    
    def fit(self, X_train, Y_train, dY_train, sigma_f=None, sigma_g=None, 
            variance_history=None):
        """
        Fit the sparse GP model using inducing points.
        
        Parameters:
            X_train: Training inputs (n, d)
            Y_train: Training outputs (n,)
            dY_train: Training gradients (n, d)
            sigma_f: Noise levels for function values
            sigma_g: Noise levels for gradients
            variance_history: Optional variance values to guide inducing point selection
        """
        n, d = X_train.shape
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        self.dY_train = np.array(dY_train)
        
        # Handle noise levels
        if sigma_f is None:
            sigma_f = np.full(n, self.sigma_f0)
        if sigma_g is None:
            sigma_g = np.full(n * d, self.sigma_g0)
        
        sigma_f = np.asarray(sigma_f).reshape(-1)
        sigma_g = np.asarray(sigma_g).reshape(-1)
        
        # Ensure correct sizes
        if len(sigma_f) != n:
            sigma_f = np.full(n, self.sigma_f0 if np.isscalar(self.sigma_f0) else np.mean(sigma_f))
        if len(sigma_g) != n * d:
            sigma_g = np.full(n * d, self.sigma_g0 if np.isscalar(self.sigma_g0) else np.mean(sigma_g))
        
        # Select inducing points
        self.inducing_indices = self._select_inducing_points(X_train, variance_history)
        self.X_inducing = X_train[self.inducing_indices]
        m = len(self.inducing_indices)
        
        # Build kernel matrices for inducing points
        # K_mm: kernel between inducing points (including gradients)
        K_ff_m = self._rbf(self.X_inducing[:, None, :], self.X_inducing[None, :, :])
        
        K_gf_m = self._drbf(self.X_inducing[:, None, :], self.X_inducing[None, :, :])
        K_gf_m = K_gf_m.transpose(0, 2, 1)
        K_gf_m = K_gf_m.reshape(m * d, m)
        
        K_fg_m = -K_gf_m.T
        
        K_gg_m = np.zeros((m * d, m * d))
        for i in range(m):
            for j in range(m):
                H_ij = self._d2rbf(self.X_inducing[i], self.X_inducing[j])
                K_gg_m[i*d:(i+1)*d, j*d:(j+1)*d] = H_ij
        
        K_mm = np.block([[K_ff_m, K_fg_m],
                         [K_gf_m, K_gg_m]])
        
        # K_nm: kernel between all points and inducing points
        K_ff_nm = self._rbf(X_train[:, None, :], self.X_inducing[None, :, :])
        
        # K_gf_nm: gradients of all train points w.r.t inducing points
        # Shape: (n, m, d) -> (n*d, m)
        K_gf_nm_temp = self._drbf(X_train[:, None, :], self.X_inducing[None, :, :])
        K_gf_nm = K_gf_nm_temp.reshape(n, m, d).transpose(0, 2, 1).reshape(n * d, m)
        
        K_fg_nm = self._drbf(X_train[:, None, :], self.X_inducing[None, :, :])
        K_fg_nm = K_fg_nm.reshape(n, m * d)
        
        K_gg_nm = np.zeros((n * d, m * d))
        for i in range(n):
            for j in range(m):
                H_ij = self._d2rbf(X_train[i], self.X_inducing[j])
                K_gg_nm[i*d:(i+1)*d, j*d:(j+1)*d] = H_ij
        
        K_nm = np.block([[K_ff_nm, K_fg_nm],
                         [K_gf_nm, K_gg_nm]])
        
        # Diagonal of full kernel (for FITC approximation)
        K_diag_f = self.sigma**2 * np.ones(n)
        K_diag_g = self.sigma**2 * np.ones(n * d)
        K_diag = np.concatenate([K_diag_f, K_diag_g])
        
        # FITC approximation: Q_nn = K_nm @ K_mm^{-1} @ K_mn
        # Predictive variance correction: Lambda = diag(K_nn - Q_nn) + noise
        
        # Add jitter for numerical stability
        K_mm += 1e-6 * np.eye(K_mm.shape[0])
        
        # Cholesky decomposition of K_mm
        L_mm = np.linalg.cholesky(K_mm)
        self.L_mm = L_mm
        
        # Compute K_mm^{-1} @ K_mn
        K_mm_inv_K_mn = scipy.linalg.cho_solve((L_mm, True), K_nm.T)
        
        # Compute diagonal correction: diag(K_nn - K_nm @ K_mm^{-1} @ K_mn)
        Q_diag = np.sum(K_nm * K_mm_inv_K_mn.T, axis=1)
        Lambda_diag = K_diag - Q_diag
        
        # Add noise - ensure proper shapes
        noise_f = sigma_f**2
        noise_g = sigma_g**2
        noise_vec = np.concatenate([noise_f, noise_g])
        
        # Double-check sizes match
        if len(noise_vec) != len(Lambda_diag):
            raise ValueError(f"Noise vector size {len(noise_vec)} doesn't match Lambda_diag size {len(Lambda_diag)}. "
                           f"sigma_f: {len(sigma_f)}, sigma_g: {len(sigma_g)}, n: {n}, d: {d}")
        
        Lambda_diag = Lambda_diag + noise_vec
        
        # Ensure positive
        Lambda_diag = np.maximum(Lambda_diag, 1e-8)
        
        self.Lambda_diag = Lambda_diag
        
        # Compute B = K_mm + K_mn @ Lambda^{-1} @ K_nm
        K_mn_Lambda_inv = K_nm.T / Lambda_diag[None, :]
        B = K_mm + K_mn_Lambda_inv @ K_nm
        
        # Add jitter
        B += 1e-6 * np.eye(B.shape[0])
        
        # Cholesky of B
        L_B = np.linalg.cholesky(B)
        self.L_B = L_B
        
        # Compute alpha: alpha = K_mm^{-1} @ K_mn @ Lambda^{-1} @ y
        y_aug = np.concatenate([Y_train, dY_train.ravel()])
        Lambda_inv_y = y_aug / Lambda_diag
        
        # alpha = B^{-1} @ K_mn @ Lambda^{-1} @ y
        self.alpha = scipy.linalg.cho_solve((L_B, True), K_mn_Lambda_inv @ y_aug)
        
        # Store for predictions
        self.K_nm = K_nm
        self.K_mm_inv_K_mn = K_mm_inv_K_mn
        
    def predict(self, X_test):
        """
        Predict function values and variances using sparse GP.
        
        Parameters:
            X_test: Test inputs (n_test, d)
            
        Returns:
            mu: Predicted means (n_test,)
            var: Predicted variances (n_test,)
        """
        n_test = X_test.shape[0]
        m, d = self.X_inducing.shape
        
        # Compute k*m: kernel between test points and inducing points
        K_ff_test = self._rbf(X_test[:, None, :], self.X_inducing[None, :, :])
        
        K_fg_test = self._drbf(self.X_inducing[None, :, :], X_test[:, None, :])
        K_fg_test = K_fg_test.transpose(1, 0, 2)
        K_fg_test = K_fg_test.reshape(n_test, m * d)
        
        k_star_m = np.hstack([K_ff_test, K_fg_test])
        
        # Mean: mu = k*m @ alpha
        mu = k_star_m @ self.alpha
        
        # Variance: var = k** - k*m @ (K_mm - B)^{-1} @ km*
        k_star_star = self.sigma**2 * np.ones(n_test)
        
        # v1 = L_mm^{-1} @ km*
        v1 = scipy.linalg.solve_triangular(self.L_mm, k_star_m.T, lower=True)
        
        # v2 = L_B^{-1} @ km*
        v2 = scipy.linalg.solve_triangular(self.L_B, k_star_m.T, lower=True)
        
        var = k_star_star - np.sum(v1**2, axis=0) + np.sum(v2**2, axis=0)
        
        # Ensure non-negative
        var = np.maximum(var, 0)
        
        return mu, var
    
    def predict_with_grad(self, X_test):
        """
        Predict function values, variances, and gradients using sparse GP.
        
        Parameters:
            X_test: Test inputs (n_test, d)
            
        Returns:
            mu: Predicted means (n_test,)
            var: Predicted variances (n_test,)
            dmu: Predicted gradients (n_test, d)
            dvar: Predicted variance gradients (n_test, d)
        """
        n_test = X_test.shape[0]
        m, d = self.X_inducing.shape
        
        # Compute kernels
        K_ff_test = self._rbf(X_test[:, None, :], self.X_inducing[None, :, :])
        # K_fg_test: gradients of X_test w.r.t inducing points
        # _drbf gives shape (..., d), with X_inducing[None,:,:] and X_test[:, None,:] gives (m, n_test, d)
        K_fg_test_temp = self._drbf(X_test[:, None, :], self.X_inducing[None, :, :])
        # Shape is (n_test, m, d), reshape to (n_test, m*d)
        K_fg_test = K_fg_test_temp.reshape(n_test, m * d)
        
        k_star_m = np.hstack([K_ff_test, K_fg_test])
        
        # For gradients, we need dk/dx
        # dK_ff_test has shape (n_test, m, d)
        dK_ff_test = self._drbf(X_test[:, None, :], self.X_inducing[None, :, :])
        
        dK_fg_test = np.zeros((n_test, m * d, d))
        for i in range(n_test):
            for j in range(m):
                H_ij = self._d2rbf(X_test[i], self.X_inducing[j])
                dK_fg_test[i, j*d:(j+1)*d, :] = H_ij
        
        # dk_star_m should have shape (n_test, d, m + m*d)
        # dK_ff_test: (n_test, m, d) -> (n_test, d, m)
        # dK_fg_test: (n_test, m*d, d) -> (n_test, d, m*d)
        dk_star_m = np.concatenate([dK_ff_test.transpose(0, 2, 1), 
                                     dK_fg_test.transpose(0, 2, 1)], axis=2)
        
        # Mean
        mu = k_star_m @ self.alpha
        
        # Gradient of mean
        dmu = np.einsum('ijk,k->ij', dk_star_m, self.alpha)
        
        # Variance
        k_star_star = self.sigma**2 * np.ones(n_test)
        v1 = scipy.linalg.solve_triangular(self.L_mm, k_star_m.T, lower=True)
        v2 = scipy.linalg.solve_triangular(self.L_B, k_star_m.T, lower=True)
        var = k_star_star - np.sum(v1**2, axis=0) + np.sum(v2**2, axis=0)
        var = np.maximum(var, 0)
        
        # Gradient of variance (simplified approximation)
        # For sparse GP, variance gradient computation is complex
        # Use a simpler approximation: numerical gradient or set to zero
        # In practice, variance gradients are less critical for optimization
        dvar = np.zeros((n_test, d))
        
        return mu, var, dmu, dvar


class SparseGEKRunner:
    """
    RVO optimizer using sparse GEK surrogate with intelligent inducing point selection.
    
    Inducing points are selected from low-variance RVO iterations, representing
    well-explored regions of the parameter space.
    """
    
    def __init__(self, length_scale=1.0, sigma=1.0, sigma_f=0.0, sigma_g=0.0,
                 n_inducing=10, inducing_strategy='auto'):
        """
        Initialize the sparse GEK runner.
        
        Parameters:
            length_scale (float): Characteristic length scale of the RBF kernel.
            sigma (float): Signal variance.
            sigma_f (float): Noise level for function observations.
            sigma_g (float): Noise level for gradient observations.
            n_inducing (int): Maximum number of inducing points.
            inducing_strategy (str): Strategy for selecting inducing points.
        """
        self.surrogate = SparseGradientGPSurrogate(
            length_scale, sigma, sigma_f, sigma_g, n_inducing, inducing_strategy
        )
        self.X_train = []
        self.Y_train = []
        self.dY_train = []
        self.variance_history = []
        self.sigma_f_arr = np.empty(0)
        self.sigma_g_arr = np.empty((0,))
        
    def add_data(self, X, Y, dY, sigma_f=None, sigma_g=None, variance=None):
        """
        Add training data with optional variance information.
        
        Parameters:
            X: Input points (n_samples, n_params)
            Y: Function values (n_samples,)
            dY: Gradients (n_samples, n_params)
            sigma_f: Noise for function values
            sigma_g: Noise for gradients
            variance: Optional variance values (for inducing point selection)
        """
        if Y.ndim == 0:
            Y = np.array([Y])
        
        if X.ndim != 2 or Y.ndim != 1 or dY.ndim != 2:
            raise ValueError("X must be 2D, Y must be 1D, and dY must be 2D.")
        
        sigma_f = (np.full(X.shape[0], self.surrogate.sigma_f0)
                   if sigma_f is None else np.asarray(sigma_f).reshape(-1))
        sigma_g = (np.full(X.size, self.surrogate.sigma_g0)
                   if sigma_g is None else np.asarray(sigma_g).reshape(-1))
        
        self.sigma_f_arr = np.concatenate([self.sigma_f_arr, sigma_f])
        self.sigma_g_arr = np.concatenate([self.sigma_g_arr, sigma_g])
        
        self.X_train.append(X)
        self.Y_train.append(Y)
        self.dY_train.append(dY)
        
        # Track variance for inducing point selection
        if variance is not None:
            if isinstance(variance, (int, float)):
                variance = np.array([variance])
            self.variance_history.extend(variance)
        else:
            self.variance_history.extend([0.0] * X.shape[0])
        
        X_all = np.vstack(self.X_train)
        Y_all = np.concatenate(self.Y_train)
        dY_all = np.vstack(self.dY_train)
        
        # Fit with variance history for intelligent inducing point selection
        self.surrogate.fit(
            X_all, Y_all, dY_all,
            sigma_f=self.sigma_f_arr,
            sigma_g=self.sigma_g_arr,
            variance_history=np.array(self.variance_history)
        )
    
    def optimize_surrogate(self, x0, var_threshold=10, tol=1e-9, alpha=0.01, max_iter=100):
        """
        Optimize the surrogate using gradient descent with variance constraint.
        
        Parameters:
            x0: Initial point
            var_threshold: Maximum allowed variance
            tol: Gradient tolerance
            alpha: Step size
            max_iter: Maximum iterations
            
        Returns:
            x_opt: Optimized point
        """
        x_current = x0.copy()
        steps = [x_current.copy()]
        
        for _ in range(max_iter):
            # Get gradient at current point
            mean, var_current, grad, _ = self.surrogate.predict_with_grad(x_current.reshape(1, -1))
            grad = grad.flatten()
            grad_norm = np.linalg.norm(grad)
            
            # Take GD step
            x_new = x_current - alpha * grad
            
            # Get variance at new point (only if needed for check)
            if len(steps) > 1:
                _, var_new = self.surrogate.predict(x_new.reshape(1, -1))
                if var_new[0] > var_threshold:
                    x_current = steps[-1].copy()
                    print(f"Converged at internal step {_} because of variance: {var_new[0]}")
                    break
                if grad_norm < tol:
                    print(f"Converged at internal step {_} because of gradient norm: {grad_norm}")
                    break
            
            x_current = x_new
            steps.append(x_current.copy())
            
        return x_current
    
    def optimize_surrogate_BFGS(self, x0, *, var_threshold=10.0, tol=1e-9, max_iter=100):
        """
        BFGS optimization of the surrogate with variance constraint.
        """
        cache = {}
        
        def eval_gp(x):
            """Query GP with caching."""
            if cache.get("x") is None or not np.array_equal(x, cache["x"]):
                mu, var, grad, _ = self.surrogate.predict_with_grad(x.reshape(1, -1))
                cache.update(x=x.copy(), mu=float(mu[0]), var=float(var[0]), grad=grad.flatten())
            return cache["mu"], cache["var"], cache["grad"]
        
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
            mu, var, grad = eval_gp(xk)
            g_norm = np.linalg.norm(grad)
            
            print(iter_count - 1, g_norm, mu, [var])
            
            if var > var_threshold:
                print(f"Converged at internal step {iter_count - 1} "
                      f"because variance exceeded threshold: {var:.3g}")
                raise StopIteration
            safe_x = xk.copy()
        
        try:
            res = minimize(
                fun, x0, method='BFGS', jac=jac,
                callback=cb,
                options={'maxiter': max_iter, 'gtol': tol}
            )
            return safe_x if res.nit > 0 else x0
        except StopIteration:
            return safe_x
    
    def GEK_optimize(self, objective, x0, var_threshold=1.0, outer_tol=1e-6,
                     inner_tol=1e-3, alpha=0.01, max_iter=50, internal_max_iter=100,
                     method='GD', return_path=False, return_surrogate=False,
                     return_energy=False, return_variance=False):
        """
        Perform GEK optimization with sparse GP.
        
        Parameters similar to original GEKRunner.GEK_optimize
        """
        x_current = x0.copy()
        path = [x_current.copy()] if return_path else None
        E_path = [] if return_energy else None
        V_path = [] if return_variance else None
        
        for outer_iter in tqdm(range(max_iter)):
            # Evaluate objective
            y_current, grad_current = objective(x_current)
            
            if return_energy:
                E_path.append(float(y_current))
            
            # Get variance if available
            if len(self.X_train) > 0:
                _, var_current = self.surrogate.predict(x_current.reshape(1, -1))
                if return_variance:
                    V_path.append(float(var_current[0]))
            else:
                var_current = np.array([0.0])
            
            # Add data
            y_sigma = np.sqrt(getattr(y_current, 'item', lambda: y_current)() 
                             if hasattr(y_current, 'item') else float(y_current)) * 0.01
            grad_sigma = np.linalg.norm(grad_current) * 0.01
            
            self.add_data(
                x_current.reshape(1, -1),
                y_current,
                grad_current.reshape(1, -1),
                y_sigma,
                grad_sigma,
                variance=var_current[0] if len(self.X_train) > 0 else None
            )
            
            # Optimize surrogate
            if method == 'GD':
                x_new = self.optimize_surrogate(
                    x_current, var_threshold, inner_tol, alpha, internal_max_iter
                )
            elif method == 'BFGS' or method == 'NLC':
                x_new = self.optimize_surrogate_BFGS(
                    x_current, var_threshold=var_threshold,
                    tol=inner_tol, max_iter=internal_max_iter
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Check convergence
            if np.linalg.norm(x_new - x_current) < outer_tol:
                x_current = x_new
                if return_path:
                    path.append(x_current.copy())
                break
            
            x_current = x_new
            if return_path:
                path.append(x_current.copy())
        
        # Build return tuple
        result = [x_current]
        if return_path:
            result.append(path)
        if return_surrogate:
            result.append(self.surrogate)
        if return_energy:
            result.append(E_path)
        if return_variance:
            result.append(V_path)
        
        return tuple(result) if len(result) > 1 else result[0]
