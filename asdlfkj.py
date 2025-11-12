# Choose sensible lengthscale from previous experiment
optimal_lengthscale = 0.4
theta_fixed = np.array([1.0, optimal_lengthscale])

# Experiment parameters
n_init_values = [1, 2, 3, 5, 7, 10]
n_iter = 25
n_runs = 20

# Run experiments
all_regret_curves = {}

for n_init in n_init_values:
    print(f"Testing n_init = {n_init} with {n_runs} runs...")
    regret_runs = []
    
    for run in range(n_runs):
        regret = compute_regret_curve_2d(
            kernel=matern52_kernel,
            theta=theta_fixed,
            acquisition_function=expected_improvement,
            X_grid=X_grid,
            x1_vals=x1_vals,
            x2_vals=x2_vals,
            f=rosenbrock_2d,
            n_init=n_init,
            n_iter=n_iter,
            seed=42 + run
        )
        regret_runs.append(regret)
    
    all_regret_curves[n_init] = np.array(regret_runs)
    print(f"  Completed {n_runs} runs\n")

# ============================================================================
# MAIN COMPARISON PLOT WITH PERCENTILE BANDS (better for log scale!)
# ============================================================================
fig, ax = plt.subplots(figsize=(15, 8))

colors = plt.cm.viridis(np.linspace(0, 0.9, len(n_init_values)))

for idx, n_init in enumerate(n_init_values):
    regret_array = all_regret_curves[n_init]
    n_evals = np.arange(1, regret_array.shape[1] + 1)
    
    # Compute statistics using PERCENTILES (scale-invariant!)
    median_regret = np.median(regret_array, axis=0)
    p25 = np.percentile(regret_array, 25, axis=0)
    p75 = np.percentile(regret_array, 75, axis=0)
    p025 = np.percentile(regret_array, 2.5, axis=0)
    p975 = np.percentile(regret_array, 97.5, axis=0)
    
    # Plot median line (bold)
    ax.plot(n_evals, median_regret, linewidth=3, color=colors[idx], 
            label=f'n_init={n_init}', marker='o', markersize=4, markevery=4, zorder=10)
    
    # Plot IQR (25th-75th percentile) - darker shade
    ax.fill_between(n_evals, p25, p75,
                     alpha=0.25, color=colors[idx], zorder=6)
    
    # Plot 95% confidence interval (2.5th-97.5th percentile) - lighter shade
    ax.fill_between(n_evals, p025, p975,
                     alpha=0.12, color=colors[idx], zorder=5)

ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal', zorder=1)
ax.set_xlabel('Total Function Evaluations', fontsize=14, fontweight='bold')
ax.set_ylabel('Regret (f_best - f_optimal)', fontsize=14, fontweight='bold')
ax.set_title(f'Effect of Initial Sample Size on BO Performance\n(ℓ={optimal_lengthscale}, {n_runs} runs, median ± IQR/95% CI)', 
             fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--', which='both')
ax.set_yscale('log')
plt.tight_layout()
plt.show()

# ============================================================================
# DIAGNOSTIC PLOT: Show individual runs with percentiles
# ============================================================================
diagnostic_n_init = 3
fig, ax = plt.subplots(figsize=(15, 7))

regret_array = all_regret_curves[diagnostic_n_init]
n_evals = np.arange(1, regret_array.shape[1] + 1)

# Plot ALL individual runs (semi-transparent)
for run_idx in range(regret_array.shape[0]):
    ax.plot(n_evals, regret_array[run_idx], alpha=0.25, color='gray', linewidth=1.5)

# Compute percentiles
median_regret = np.median(regret_array, axis=0)
p25 = np.percentile(regret_array, 25, axis=0)
p75 = np.percentile(regret_array, 75, axis=0)
p025 = np.percentile(regret_array, 2.5, axis=0)
p975 = np.percentile(regret_array, 97.5, axis=0)
best_regret = np.min(regret_array, axis=0)
worst_regret = np.max(regret_array, axis=0)

# Plot statistics
ax.plot(n_evals, median_regret, linewidth=4, color='blue', label='Median', zorder=10)
ax.plot(n_evals, p25, linewidth=2, color='green', linestyle='--', label='25th percentile', zorder=9)
ax.plot(n_evals, p75, linewidth=2, color='orange', linestyle='--', label='75th percentile', zorder=9)

# Shaded regions
ax.fill_between(n_evals, p25, p75, alpha=0.3, color='blue', label='IQR (25-75%)', zorder=6)
ax.fill_between(n_evals, p025, p975, alpha=0.15, color='blue', label='95% CI (2.5-97.5%)', zorder=5)

# Best/worst envelope
ax.plot(n_evals, best_regret, linewidth=2, color='darkgreen', linestyle=':', label='Best run', zorder=8)
ax.plot(n_evals, worst_regret, linewidth=2, color='darkred', linestyle=':', label='Worst run', zorder=8)

ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Optimal')
ax.set_xlabel('Total Function Evaluations', fontsize=14, fontweight='bold')
ax.set_ylabel('Regret (log scale)', fontsize=14, fontweight='bold')
ax.set_title(f'Diagnostic: All {n_runs} runs for n_init={diagnostic_n_init}\n(Gray = individual runs, dark shade = IQR, light shade = 95% CI)', 
             fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, ncol=2)
ax.grid(alpha=0.3, which='both')
ax.set_yscale('log')
plt.tight_layout()
plt.show()

# ============================================================================
# SUMMARY STATISTICS TABLE (with percentiles!)
# ============================================================================
print("\n" + "="*110)
print(f"{'n_init':<8} {'Final Median':<14} {'Final IQR':<14} {'Best Run':<12} {'Worst Run':<12} {'95% CI Width':<12}")
print("="*110)
for n_init in n_init_values:
    regret_array = all_regret_curves[n_init]
    final_regrets = regret_array[:, -1]
    
    final_median = np.median(final_regrets)
    final_p25 = np.percentile(final_regrets, 25)
    final_p75 = np.percentile(final_regrets, 75)
    final_iqr = final_p75 - final_p25
    final_p025 = np.percentile(final_regrets, 2.5)
    final_p975 = np.percentile(final_regrets, 97.5)
    ci_width = final_p975 - final_p025
    best_run = np.min(final_regrets)
    worst_run = np.max(final_regrets)
    
    print(f"{n_init:<8} {final_median:<14.4e} {final_iqr:<14.4e} {best_run:<12.4e} {worst_run:<12.4e} {ci_width:<12.4e}")
print("="*110)

# ============================================================================
# BONUS: Side-by-side linear and log comparison
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

for ax, scale_name in zip([ax1, ax2], ['linear', 'log']):
    for idx, n_init in enumerate(n_init_values):
        regret_array = all_regret_curves[n_init]
        n_evals = np.arange(1, regret_array.shape[1] + 1)
        
        median_regret = np.median(regret_array, axis=0)
        p25 = np.percentile(regret_array, 25, axis=0)
        p75 = np.percentile(regret_array, 75, axis=0)
        
        ax.plot(n_evals, median_regret, linewidth=2.5, color=colors[idx], 
                label=f'n_init={n_init}', marker='o', markersize=3, markevery=5)
        ax.fill_between(n_evals, p25, p75, alpha=0.2, color=colors[idx])
    
    ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Optimal')
    ax.set_xlabel('Total Function Evaluations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regret', fontsize=12, fontweight='bold')
    ax.set_title(f'{scale_name.capitalize()} Scale', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    
    if scale_name == 'log':
        ax.set_yscale('log')

plt.suptitle(f'Comparison: Linear vs Log Scale (median ± IQR)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()