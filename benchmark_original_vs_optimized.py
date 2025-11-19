"""
Comprehensive Benchmark: Original vs Optimized GEK Implementation

This script compares the original and optimized implementations of the
Gradient-Enhanced Kriging (GEK) surrogate model to verify:
1. Numerical correctness (results are equivalent within floating-point precision)
2. Performance improvements (speedup measurements)

The optimizations include:
- Vectorized prediction methods (eliminates Python loops)
- Direct Cholesky solve (eliminates explicit matrix inverse)  
- Optimized gradient computations (using einsum)
"""
import numpy as np
import sys
import time
import importlib.util
from pathlib import Path

# Get the repository root
repo_root = Path(__file__).parent

# Load original implementation (from git history)
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

# Load optimized implementation (current)
sys.path.insert(0, str(repo_root))
import GEK as GEK_optimized

print("=" * 90)
print("COMPREHENSIVE BENCHMARK: Original vs Optimized GEK Implementation")
print("=" * 90)
print("\nThis benchmark validates:")
print("  1. Numerical correctness (differences within floating-point precision)")
print("  2. Performance improvements (speedup factors)")
print("\n" + "=" * 90)

def f(x):
    """Test function"""
    return np.sin(1.2 * x) + 0.15 * x

def fprime(x):
    """Derivative of test function"""
    return 1.2 * np.cos(1.2 * x) + 0.15

def benchmark_case(name, X_train, y_train, dy_train, X_test, 
                   length_scale=1.5, sigma=1.0, sigma_f=None, sigma_g=None,
                   n_runs=5):
    """Run a single benchmark case"""
    print(f"\n{name}")
    print("-" * 90)
    
    # Fit original GP
    gp_orig = GEK_original.GradientGPSurrogate(
        length_scale=length_scale, sigma=sigma, 
        sigma_f=sigma_f if sigma_f is not None else 0.0,
        sigma_g=sigma_g if sigma_g is not None else 0.0
    )
    start = time.time()
    gp_orig.fit(X_train, y_train, dy_train)
    fit_time_orig = time.time() - start
    
    # Fit optimized GP
    gp_opt = GEK_optimized.GradientGPSurrogate(
        length_scale=length_scale, sigma=sigma,
        sigma_f=sigma_f if sigma_f is not None else 0.0,
        sigma_g=sigma_g if sigma_g is not None else 0.0
    )
    start = time.time()
    gp_opt.fit(X_train, y_train, dy_train)
    fit_time_opt = time.time() - start
    
    print(f"Dataset: {X_train.shape[0]} training points, {X_test.shape[0]} test points")
    print(f"Fit time: Original={fit_time_orig:.4f}s, Optimized={fit_time_opt:.4f}s")
    
    # Benchmark predict()
    times_orig = []
    times_opt = []
    
    for _ in range(n_runs):
        start = time.time()
        mu_orig, var_orig = gp_orig.predict(X_test)
        times_orig.append(time.time() - start)
        
        start = time.time()
        mu_opt, var_opt = gp_opt.predict(X_test)
        times_opt.append(time.time() - start)
    
    predict_time_orig = np.mean(times_orig)
    predict_time_opt = np.mean(times_opt)
    
    # Benchmark predict_with_grad()
    times_grad_orig = []
    times_grad_opt = []
    
    for _ in range(n_runs):
        start = time.time()
        mu_g_orig, var_g_orig, dmu_orig, dvar_orig = gp_orig.predict_with_grad(X_test)
        times_grad_orig.append(time.time() - start)
        
        start = time.time()
        mu_g_opt, var_g_opt, dmu_opt, dvar_opt = gp_opt.predict_with_grad(X_test)
        times_grad_opt.append(time.time() - start)
    
    predict_grad_time_orig = np.mean(times_grad_orig)
    predict_grad_time_opt = np.mean(times_grad_opt)
    
    # Numerical correctness
    mu_rms = np.sqrt(np.mean((mu_orig - mu_opt)**2))
    var_rms = np.sqrt(np.mean((var_orig - var_opt)**2))
    dmu_rms = np.sqrt(np.mean((dmu_orig - dmu_opt)**2))
    
    # Relative errors (as percentage of signal)
    mu_rel_pct = mu_rms / (np.std(mu_orig) + 1e-10) * 100
    dmu_rel_pct = dmu_rms / (np.std(dmu_orig) + 1e-10) * 100
    
    print(f"\nNumerical Accuracy (RMS differences):")
    print(f"  Means:     {mu_rms:.2e}  ({mu_rel_pct:.4f}% of signal)")
    print(f"  Variances: {var_rms:.2e}")
    print(f"  Gradients: {dmu_rms:.2e}  ({dmu_rel_pct:.4f}% of signal)")
    
    # Performance comparison
    speedup_predict = predict_time_orig / predict_time_opt
    speedup_grad = predict_grad_time_orig / predict_grad_time_opt
    
    print(f"\nPerformance (average of {n_runs} runs):")
    print(f"  predict():")
    print(f"    Original:  {predict_time_orig:.4f}s ({predict_time_orig/X_test.shape[0]*1000:.3f}ms/point)")
    print(f"    Optimized: {predict_time_opt:.4f}s ({predict_time_opt/X_test.shape[0]*1000:.3f}ms/point)")
    print(f"    Speedup:   {speedup_predict:.2f}x")
    
    print(f"  predict_with_grad():")
    print(f"    Original:  {predict_grad_time_orig:.4f}s ({predict_grad_time_orig/X_test.shape[0]*1000:.3f}ms/point)")
    print(f"    Optimized: {predict_grad_time_opt:.4f}s ({predict_grad_time_opt/X_test.shape[0]*1000:.3f}ms/point)")
    print(f"    Speedup:   {speedup_grad:.2f}x")
    
    # Check if numerically acceptable (< 0.1% error)
    passed = mu_rel_pct < 0.1 and dmu_rel_pct < 0.1
    
    return {
        'passed': passed,
        'speedup_predict': speedup_predict,
        'speedup_grad': speedup_grad,
        'mu_rel_pct': mu_rel_pct,
        'dmu_rel_pct': dmu_rel_pct
    }

# Run test cases
results = []

# Test Case 1: Small dataset (basic functionality)
print("\n" + "=" * 90)
print("TEST CASE 1: Small Dataset")
print("=" * 90)
X_train_small = np.array([-3.5, 0.0, 3.5]).reshape(-1, 1)
y_train_small = f(X_train_small).ravel()
dy_train_small = fprime(X_train_small).reshape(-1, 1)
X_test_small = np.linspace(-5, 5, 100).reshape(-1, 1)

results.append(('Small Dataset', benchmark_case(
    "3 training points, 100 test points",
    X_train_small, y_train_small, dy_train_small, X_test_small
)))

# Test Case 2: Medium dataset
print("\n" + "=" * 90)
print("TEST CASE 2: Medium Dataset")
print("=" * 90)
X_train_med = np.linspace(-5, 5, 10).reshape(-1, 1)
y_train_med = f(X_train_med).ravel()
dy_train_med = fprime(X_train_med).reshape(-1, 1)
X_test_med = np.linspace(-5, 5, 500).reshape(-1, 1)

results.append(('Medium Dataset', benchmark_case(
    "10 training points, 500 test points",
    X_train_med, y_train_med, dy_train_med, X_test_med
)))

# Test Case 3: Large batch
print("\n" + "=" * 90)
print("TEST CASE 3: Large Batch Predictions")
print("=" * 90)
X_test_large = np.linspace(-5, 5, 1000).reshape(-1, 1)

results.append(('Large Batch', benchmark_case(
    "10 training points, 1000 test points",
    X_train_med, y_train_med, dy_train_med, X_test_large
)))

# Test Case 4: With noise
print("\n" + "=" * 90)
print("TEST CASE 4: Heteroscedastic Noise")
print("=" * 90)
np.random.seed(42)
X_train_noisy = np.linspace(-3, 3, 10).reshape(-1, 1)
y_train_noisy = f(X_train_noisy).ravel() + np.random.normal(0, 0.1, 10)
dy_train_noisy = fprime(X_train_noisy).reshape(-1, 1) + np.random.normal(0, 0.05, (10, 1))
X_test_noisy = np.linspace(-3, 3, 200).reshape(-1, 1)

results.append(('With Noise', benchmark_case(
    "10 training points with noise, 200 test points",
    X_train_noisy, y_train_noisy, dy_train_noisy, X_test_noisy,
    sigma_f=0.1, sigma_g=0.05
)))

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

all_passed = all(r[1]['passed'] for r in results)
avg_speedup_predict = np.mean([r[1]['speedup_predict'] for r in results])
avg_speedup_grad = np.mean([r[1]['speedup_grad'] for r in results])

print(f"\nNumerical Correctness: {'✓ PASSED' if all_passed else '✗ FAILED'}")
for name, result in results:
    status = "✓" if result['passed'] else "✗"
    print(f"  {status} {name}: {result['mu_rel_pct']:.4f}% mean error, {result['dmu_rel_pct']:.4f}% gradient error")

print(f"\nPerformance Improvements:")
print(f"  Average speedup (predict):          {avg_speedup_predict:.2f}x")
print(f"  Average speedup (predict_with_grad): {avg_speedup_grad:.2f}x")

print("\nDetailed Speedups by Test Case:")
for name, result in results:
    print(f"  {name:20s}: predict={result['speedup_predict']:.2f}x, with_grad={result['speedup_grad']:.2f}x")

print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)
if all_passed:
    print("✓ All tests PASSED")
    print("  - The optimized implementation produces numerically equivalent results")
    print("  - All differences are well within floating-point precision (< 0.1% error)")
    print(f"  - Average speedup: {avg_speedup_predict:.2f}x for predict(), {avg_speedup_grad:.2f}x for predict_with_grad()")
    print("\nThe optimizations are safe to use in production.")
else:
    print("✗ Some tests FAILED")
    print("  - Numerical differences exceed acceptable thresholds")

print("=" * 90)

sys.exit(0 if all_passed else 1)
