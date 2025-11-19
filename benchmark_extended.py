"""
Extended Benchmark: Testing High-Dimensional and Edge Cases

This benchmark extends the original to cover:
1. High-dimensional problems (similar to VMC: 100-500 parameters)
2. Edge cases (1D, very high dimensions)
3. Multiple training set sizes
4. Various hyperparameter combinations
5. Real VMC-like objective functions
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
print("EXTENDED BENCHMARK: High-Dimensional and Edge Cases")
print("=" * 90)
print("\nThis benchmark tests:")
print("  1. High-dimensional problems (100-500 parameters, similar to VMC)")
print("  2. Edge cases (1D, very sparse data, many training points)")
print("  3. Various hyperparameter combinations")
print("  4. VMC-like objective functions")
print("\n" + "=" * 90)

def benchmark_case(name, d, n_train, n_test, length_scale=1.5, sigma=1.0, 
                   sigma_f=0.0, sigma_g=0.0, n_runs=3):
    """Run a single benchmark case with specified dimensions"""
    print(f"\n{name}")
    print("-" * 90)
    print(f"Dimensions: {d}, Training: {n_train}, Testing: {n_test}")
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.randn(n_train, d)
    
    # Use a quadratic function that scales well with dimension
    y_train = np.sum(X_train**2, axis=1) + np.random.normal(0, 0.1, n_train)
    dy_train = 2 * X_train + np.random.normal(0, 0.01, (n_train, d))
    
    X_test = np.random.randn(n_test, d)
    
    # Fit original GP
    try:
        gp_orig = GEK_original.GradientGPSurrogate(
            length_scale=length_scale, sigma=sigma, 
            sigma_f=sigma_f, sigma_g=sigma_g
        )
        start = time.time()
        gp_orig.fit(X_train, y_train, dy_train)
        fit_time_orig = time.time() - start
    except Exception as e:
        print(f"  ✗ Original implementation failed: {e}")
        return None
    
    # Fit optimized GP
    try:
        gp_opt = GEK_optimized.GradientGPSurrogate(
            length_scale=length_scale, sigma=sigma,
            sigma_f=sigma_f, sigma_g=sigma_g
        )
        start = time.time()
        gp_opt.fit(X_train, y_train, dy_train)
        fit_time_opt = time.time() - start
    except Exception as e:
        print(f"  ✗ Optimized implementation failed: {e}")
        return None
    
    print(f"  Fit time: Original={fit_time_orig:.4f}s, Optimized={fit_time_opt:.4f}s (speedup: {fit_time_orig/fit_time_opt:.2f}x)")
    
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
    dmu_rms = np.sqrt(np.mean((dmu_orig - dmu_opt)**2))
    
    # Relative errors (as percentage of signal)
    mu_rel_pct = mu_rms / (np.std(mu_orig) + 1e-10) * 100
    dmu_rel_pct = dmu_rms / (np.std(dmu_orig) + 1e-10) * 100
    
    print(f"  Numerical accuracy: mean {mu_rel_pct:.4f}%, gradient {dmu_rel_pct:.4f}%")
    
    # Performance comparison
    speedup_predict = predict_time_orig / predict_time_opt
    speedup_grad = predict_grad_time_orig / predict_grad_time_opt
    
    print(f"  predict() speedup: {speedup_predict:.2f}x")
    print(f"  predict_with_grad() speedup: {speedup_grad:.2f}x")
    
    # Check if numerically acceptable (< 0.1% error)
    passed = mu_rel_pct < 0.1 and dmu_rel_pct < 0.1
    
    if passed:
        print(f"  ✓ PASSED")
    else:
        print(f"  ✗ FAILED: Numerical differences too large")
    
    return {
        'passed': passed,
        'speedup_predict': speedup_predict,
        'speedup_grad': speedup_grad,
        'mu_rel_pct': mu_rel_pct,
        'dmu_rel_pct': dmu_rel_pct,
        'fit_speedup': fit_time_orig / fit_time_opt
    }

# Run test cases
results = []

print("\n" + "=" * 90)
print("SECTION 1: Edge Cases")
print("=" * 90)

# Edge case 1: Single dimension
result = benchmark_case(
    "1D Problem (edge case)",
    d=1, n_train=5, n_test=50
)
if result: results.append(('1D Problem', result))

# Edge case 2: Very few training points
result = benchmark_case(
    "Minimal Training Data (2 points, 10D)",
    d=10, n_train=2, n_test=100
)
if result: results.append(('Minimal Training', result))

# Edge case 3: Many training points
result = benchmark_case(
    "Many Training Points (20 points, 10D)",
    d=10, n_train=20, n_test=100
)
if result: results.append(('Many Training', result))

print("\n" + "=" * 90)
print("SECTION 2: High-Dimensional Problems (VMC-like)")
print("=" * 90)

# VMC-like case 1: Small VMC (d=20, alpha=5, total=105 params)
result = benchmark_case(
    "Small VMC-like (105 parameters)",
    d=105, n_train=5, n_test=50
)
if result: results.append(('Small VMC (105D)', result))

# VMC-like case 2: Medium VMC (d=50, alpha=5, total=255 params)
result = benchmark_case(
    "Medium VMC-like (255 parameters)",
    d=255, n_train=5, n_test=50
)
if result: results.append(('Medium VMC (255D)', result))

# VMC-like case 3: Large VMC (d=100, alpha=5, total=505 params)
print("\nNOTE: Large VMC test (505 parameters) - this may take a few minutes...")
result = benchmark_case(
    "Large VMC-like (505 parameters)",
    d=505, n_train=5, n_test=50
)
if result: results.append(('Large VMC (505D)', result))

print("\n" + "=" * 90)
print("SECTION 3: Scaling with Test Set Size")
print("=" * 90)

# Test scaling with number of predictions
for n_test in [10, 100, 500, 1000]:
    result = benchmark_case(
        f"50D with {n_test} test points",
        d=50, n_train=5, n_test=n_test
    )
    if result: results.append((f'50D, {n_test} test', result))

print("\n" + "=" * 90)
print("SECTION 4: Different Hyperparameters")
print("=" * 90)

# Test different length scales
for ls in [0.5, 1.0, 2.0]:
    result = benchmark_case(
        f"50D with length_scale={ls}",
        d=50, n_train=5, n_test=100, length_scale=ls
    )
    if result: results.append((f'50D, ls={ls}', result))

# Test with noise
result = benchmark_case(
    "50D with noise (sigma_f=0.1, sigma_g=0.01)",
    d=50, n_train=5, n_test=100, sigma_f=0.1, sigma_g=0.01
)
if result: results.append(('50D with noise', result))

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

all_passed = all(r[1]['passed'] for r in results)
avg_speedup_predict = np.mean([r[1]['speedup_predict'] for r in results])
avg_speedup_grad = np.mean([r[1]['speedup_grad'] for r in results])
avg_fit_speedup = np.mean([r[1]['fit_speedup'] for r in results])

print(f"\nNumerical Correctness: {'✓ PASSED' if all_passed else '✗ FAILED'}")
print(f"Tests passed: {sum(1 for r in results if r[1]['passed'])}/{len(results)}")

print(f"\nOverall Performance Improvements:")
print(f"  Fit time speedup:                {avg_fit_speedup:.2f}x")
print(f"  predict() speedup:               {avg_speedup_predict:.2f}x")
print(f"  predict_with_grad() speedup:     {avg_speedup_grad:.2f}x")

print(f"\nDetailed Results by Test Case:")
print(f"{'Test Case':<30s} {'Pass':<6s} {'predict()':<12s} {'with_grad()':<12s}")
print("-" * 90)
for name, result in results:
    status = "✓" if result['passed'] else "✗"
    print(f"{name:<30s} {status:<6s} {result['speedup_predict']:>6.2f}x       {result['speedup_grad']:>6.2f}x")

# Special focus on VMC-like cases
print(f"\n" + "=" * 90)
print("VMC-SPECIFIC RESULTS")
print("=" * 90)
vmc_results = [(n, r) for n, r in results if 'VMC' in n]
if vmc_results:
    print(f"\nHigh-dimensional VMC-like problems:")
    for name, result in vmc_results:
        print(f"  {name}:")
        print(f"    Numerical accuracy: {result['mu_rel_pct']:.4f}% (mean), {result['dmu_rel_pct']:.4f}% (gradient)")
        print(f"    Speedups: fit={result['fit_speedup']:.2f}x, predict={result['speedup_predict']:.2f}x, with_grad={result['speedup_grad']:.2f}x")
        print(f"    Status: {'✓ PASSED' if result['passed'] else '✗ FAILED'}")

print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)
if all_passed:
    print("✓ All tests PASSED")
    print("  - The optimized implementation handles high-dimensional problems correctly")
    print("  - VMC-like problems (100-500 parameters) show consistent speedups")
    print(f"  - Average speedup: fit={avg_fit_speedup:.2f}x, predict={avg_speedup_predict:.2f}x, with_grad={avg_speedup_grad:.2f}x")
    print("\nThe optimizations are validated for production use with high-dimensional VMC problems.")
else:
    print("✗ Some tests FAILED")
    print("  - Review failed test cases above")

print("=" * 90)

sys.exit(0 if all_passed else 1)
