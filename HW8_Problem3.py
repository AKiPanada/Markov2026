import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize


np.random.seed(42)

print("--- Problem 3 ---")

# 3(c)
lam_max = 20.0
T_DAYS = 30
T_TOTAL = T_DAYS * 24.0

def lam(t):
    return 12.0 + 8.0 * np.cos(np.pi * (t - 16.0) / 12.0)

n_prop = np.random.poisson(lam_max * T_TOTAL)
t_prop = np.sort(np.random.uniform(0.0, T_TOTAL, n_prop))
kept_mask = np.random.uniform(0.0, 1.0, n_prop) < (lam(t_prop) / lam_max)
t_kept = t_prop[kept_mask]
n_kept = len(t_kept)

frac_kept_sim = n_kept / n_prop

clock_hours = np.floor(t_kept % 24.0).astype(int)
hourly_counts = np.bincount(clock_hours, minlength=24) / T_DAYS

day_indices = np.floor(t_kept / 24.0).astype(int)
counts_3_4_per_day = np.bincount(day_indices[clock_hours == 3], minlength=T_DAYS)

mean_3_4_sim = np.mean(counts_3_4_per_day)
var_3_4_sim = np.var(counts_3_4_per_day, ddof=1)

def neg_log_lik(params):
    a, b, phi = params
    rates = a + b * np.cos(np.pi * (t_kept - phi) / 12.0)
    if np.any(rates <= 0):
        return 1e12
    return -(np.sum(np.log(rates)) - a * T_TOTAL)

res = minimize(neg_log_lik, [10.0, 5.0, 14.0], method='Nelder-Mead')
a_hat, b_hat, phi_hat = res.x
max_ll = -res.fun

const_rate = n_kept / T_TOTAL
const_ll = n_kept * np.log(const_rate) - n_kept

print(f"3(c): Fraction kept = {frac_kept_sim:.4f}")
print(f"3(c): 3-4am mean = {mean_3_4_sim:.4f}, var = {var_3_4_sim:.4f}")
print(f"3(c): MLE (a, b, phi) = ({a_hat:.4f}, {b_hat:.4f}, {phi_hat:.4f})")
print(f"3(c): Log-lik = {max_ll:.4f}, Constant log-lik = {const_ll:.4f}\n")

hours_grid = np.arange(24)
t_fine = np.linspace(0, 24, 500)

plt.figure(figsize=(10, 6))
plt.bar(hours_grid + 0.5, hourly_counts, width=0.8, alpha=0.5, color='blue',
        label='Simulated Mean Arrivals (per hour)')
plt.plot(t_fine, lam(t_fine), 'r-', linewidth=2, label='True Rate $\\lambda(t)$')
plt.title("Emergency Room Hourly Arrival Rate (30-Day Thinning Simulation)")
plt.xlabel("Hour of Day (t)")
plt.ylabel("Arrivals per Hour")
plt.xticks(np.arange(0, 25, 2))
plt.xlim(0, 24)
plt.legend()
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW8-Problem-3-Figure.png", dpi=300, bbox_inches="tight")
plt.show()
