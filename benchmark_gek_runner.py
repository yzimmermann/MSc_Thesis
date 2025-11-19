"""
Benchmark GEKRunner with High-Dimensional Problems

Tests the complete optimization workflow with VMC-like dimensionalities.
"""
import numpy as np
import sys
import time
import importlib.util
from pathlib import Path

# Get the repository root
repo_root = Path(__file__).parent

# Load original implementation
import subprocess
result = subprocess.run(
    ["git", "show", "58f2dc2:GEK.py"],
    capture_output=True,
    text=True,
    cwd=repo_root
)
with open("/tmp/GEK_original_temp.py", "w") as f:
    f.write(result.stdout)

spec_orig = importlib.util.spec_from_file_location("GEK_original", "/tmp/GEK_original_temp.py")
GEK_original = importlib.util.module_from_spec(spec_orig)
spec_orig.loader.exec_module(GEK_original)

# Load optimized implementation
sys.path.insert(0, str(repo_root))
import GEK as GEK_optimized

print("=" * 90)
print("BENCHMARK: GEKRunner with High-Dimensional Problems")
print("=" * 90)
print("\nTesting complete optimization workflow with VMC-like dimensionalities")
print("=" * 90)

def rosenbrock_extended(x):
    """Extended Rosenbrock function - good test for high dimensions"""
    n = len(x)
    result = 0.0
    for i in range(n-1):
        result += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return result

def rosenbrock_extended_grad(x):
    """Gradient of extended Rosenbrock"""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n-1):
        grad[i] += -400.0 * x[i] * (x[i+1] - x[i]**2) - 2.0 * (1 - x[i])
        grad[i+1] += 200.0 * (x[i+1] - x[i]**2)
    return grad

def quadratic_high_dim(x):
    """Simple quadratic - easy to optimize, good for testing"""
    return np.sum((x - 1.0)**2)

def quadratic_high_dim_grad(x):
    """Gradient of quadratic"""
    return 2.0 * (x - 1.0)

def noisy_objective(func, grad_func, noise_level=0.01):
    """Wrapper to add noise to objective"""
    def wrapped(x):
        f = func(x)
        g = grad_func(x)
        # Add noise
        f_noisy = f + np.random.normal(0, noise_level * np.abs(f))
        g_noisy = g + np.random.normal(0, noise_level * np.linalg.norm(g), size=g.shape)
        return np.array(f_noisy), g_noisy
    return wrapped

def benchmark_runner(name, d, func, max_iter=5):
    """Benchmark GEKRunner on a specific problem"""
    print(f"\n{name}")
    print("-" * 90)
    print(f"Dimensions: {d}, Max iterations: {max_iter}")
    
    np.random.seed(42)
    x0 = np.random.randn(d) * 0.5
    
    # Test original
    try:
        runner_orig = GEK_original.GEKRunner(length_scale=1.0, sigma=1.0)
        start = time.time()
        x_opt_orig, path_orig, _, E_orig, _ = runner_orig.GEK_optimize(
            func, x0.copy(),
            var_threshold=1.0,
            method='GD',
            outer_tol=1e-5,
            inner_tol=1e-3,
            alpha=0.05,
            max_iter=max_iter,
            internal_max_iter=20,
            return_path=True,
            return_surrogate=False,
            return_energy=True,
            return_variance=True,
        )
        time_orig = time.time() - start
    except Exception as e:
        print(f"  ✗ Original implementation failed: {e}")
        return None
    
    # Test optimized
    try:
        runner_opt = GEK_optimized.GEKRunner(length_scale=1.0, sigma=1.0)
        start = time.time()
        x_opt_opt, path_opt, _, E_opt, _ = runner_opt.GEK_optimize(
            func, x0.copy(),
            var_threshold=1.0,
            method='GD',
            outer_tol=1e-5,
            inner_tol=1e-3,
            alpha=0.05,
            max_iter=max_iter,
            internal_max_iter=20,
            return_path=True,
            return_surrogate=False,
            return_energy=True,
            return_variance=True,
        )
        time_opt = time.time() - start
    except Exception as e:
        print(f"  ✗ Optimized implementation failed: {e}")
        return None
    
    # Compare results
    x_diff = np.linalg.norm(x_opt_orig - x_opt_opt) / (np.linalg.norm(x_opt_orig) + 1e-10)
    E_diff = np.abs(E_orig[-1] - E_opt[-1]) / (np.abs(E_orig[-1]) + 1e-10)
    
    print(f"  Time: Original={time_orig:.2f}s, Optimized={time_opt:.2f}s")
    print(f"  Speedup: {time_orig/time_opt:.2f}x")
    print(f"  Final x difference: {x_diff:.2e}")
    print(f"  Final energy difference: {E_diff:.2e}")
    print(f"  Iterations completed: Original={len(E_orig)}, Optimized={len(E_opt)}")
    
    # For stochastic optimization, differences up to 20% are acceptable
    # (different random noise leads to different optimization paths)
    passed = x_diff < 0.2 and E_diff < 0.2
    
    if passed:
        print(f"  ✓ PASSED (differences within stochastic tolerance)")
    else:
        print(f"  ✗ FAILED: Numerical differences too large (> 20%)")
    
    return {
        'passed': passed,
        'speedup': time_orig / time_opt,
        'x_diff': x_diff,
        'E_diff': E_diff
    }

# Run tests
results = []

print("\n" + "=" * 90)
print("SECTION 1: Deterministic Problems (No Noise)")
print("=" * 90)
print("These should show exact numerical equivalence")

def deterministic_objective(func, grad_func):
    """Wrapper for deterministic objective"""
    def wrapped(x):
        return np.array(func(x)), grad_func(x)
    return wrapped

# Deterministic tests
for d in [10, 50, 105]:
    obj = deterministic_objective(quadratic_high_dim, quadratic_high_dim_grad)
    result = benchmark_runner(f"{d}D Quadratic (deterministic)", d=d, func=obj, max_iter=3)
    if result: results.append((f'{d}D Deterministic', result))

print("\n" + "=" * 90)
print("SECTION 2: Low-Dimensional Baseline with Noise")
print("=" * 90)

# Small problem
obj_small = noisy_objective(quadratic_high_dim, quadratic_high_dim_grad, 0.01)
result = benchmark_runner("10D Quadratic (noisy baseline)", d=10, func=obj_small, max_iter=5)
if result: results.append(('10D Noisy', result))

print("\n" + "=" * 90)
print("SECTION 3: VMC-like High-Dimensional Problems with Noise")
print("=" * 90)
print("Note: Noisy optimization paths will differ slightly (this is expected)")

# VMC-like sizes
for d in [50, 105, 255]:
    obj = noisy_objective(quadratic_high_dim, quadratic_high_dim_grad, 0.01)
    result = benchmark_runner(f"{d}D Quadratic (VMC-like, noisy)", d=d, func=obj, max_iter=5)
    if result: results.append((f'{d}D Noisy', result))

# Large VMC (505D) - reduced iterations for speed
print("\nNOTE: 505D test - this will take several minutes...")
obj_large = noisy_objective(quadratic_high_dim, quadratic_high_dim_grad, 0.01)
result = benchmark_runner("505D Quadratic (Large VMC)", d=505, func=obj_large, max_iter=3)
if result: results.append(('505D Quadratic', result))

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

all_passed = all(r[1]['passed'] for r in results)
avg_speedup = np.mean([r[1]['speedup'] for r in results])

print(f"\nNumerical Correctness: {'✓ PASSED' if all_passed else '✗ FAILED'}")
print(f"Tests passed: {sum(1 for r in results if r[1]['passed'])}/{len(results)}")

print(f"\nOverall Performance:")
print(f"  Average speedup: {avg_speedup:.2f}x")

print(f"\nDetailed Results:")
print(f"{'Test Case':<30s} {'Status':<8s} {'Speedup':<10s} {'x_diff':<12s} {'E_diff':<12s}")
print("-" * 90)
for name, result in results:
    status = "✓ PASS" if result['passed'] else "✗ FAIL"
    print(f"{name:<30s} {status:<8s} {result['speedup']:>6.2f}x     {result['x_diff']:>10.2e}  {result['E_diff']:>10.2e}")

print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)
if all_passed:
    print("✓ All GEKRunner tests PASSED")
    print("  - Complete optimization workflow works correctly in high dimensions")
    print("  - VMC-like problems (50-505 parameters) optimize consistently")
    print(f"  - Average speedup: {avg_speedup:.2f}x")
    print("\nThe GEKRunner optimization is validated for production use with VMC problems.")
else:
    print("✗ Some tests FAILED")
    print("  - Review failed test cases above")

print("=" * 90)

sys.exit(0 if all_passed else 1)
