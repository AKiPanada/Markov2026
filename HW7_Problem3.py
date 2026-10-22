import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import poisson, gamma, binom

lam = 4
N_sim = 200000
rng = np.random.default_rng(4560)

# Method 1: Exponential Gaps
gaps = rng.exponential(1 / lam, size=(N_sim, 40))
T_exp = np.cumsum(gaps, axis=1)
N1_exp = (T_exp <= 1).sum(axis=1)

# Method 2: Poisson count + Uniform points
N1_pois = rng.poisson(lam, N_sim)

print("Exponential Gaps N(1) - Mean:", N1_exp.mean(), "Var:", N1_exp.var())
print("Poisson + Uniform N(1) - Mean:", N1_pois.mean(), "Var:", N1_pois.var())

k_vals = np.arange(0, 16)
poisson_pmf = poisson.pmf(k_vals, lam)

plt.figure(figsize=(10, 5))
plt.hist(N1_exp, bins=np.arange(-0.5, 16), density=True, alpha=0.5, label='Simulated N(1)')
plt.plot(k_vals, poisson_pmf, 'ko', label='Poisson(4) pmf')
plt.title("Histogram of N(1) vs Poisson(4) pmf")
plt.xlabel("N(1) count")
plt.ylabel("Probability")
plt.legend()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW7-Problem-3d1-Figure.png", dpi=300, bbox_inches="tight")
plt.show()

# Rasters
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.eventplot([T_exp[i][T_exp[i] <= 1] for i in range(10)], color='b')
plt.title("Raster: Exponential Gaps")
plt.xlim(0, 1)

plt.subplot(1, 2, 2)
plt.eventplot([np.sort(rng.random(n)) for n in N1_pois[:10]], color='r')
plt.title("Raster: Poisson + Uniform")
plt.xlim(0, 1)
plt.savefig("figures/HW7-Problem-3d2-Figure.png", dpi=300, bbox_inches="tight")
plt.show()

# T3 Histogram
t_vals = np.linspace(0, 3, 300)
gamma_pdf = gamma.pdf(t_vals, a=3, scale=1/lam)

plt.figure(figsize=(8, 5))
plt.hist(T_exp[:, 2], bins=100, density=True, alpha=0.6, label='Simulated T_3')
plt.plot(t_vals, gamma_pdf, 'k-', lw=2, label='Gamma(3, 4) pdf')
plt.title("Time of 3rd Meteor (T_3)")
plt.xlabel("Time (hours)")
plt.legend()
plt.savefig("figures/HW7-Problem-3d3-Figure.png", dpi=300, bbox_inches="tight")
plt.show()

# N(1) = 3 Subpopulation Stats
idx_3 = (N1_exp == 3)
T_sub = T_exp[idx_3]
N_half = (T_sub <= 0.5).sum(axis=1)

print(f"Given N(1)=3: P(N(1/2)=0) = {(N_half == 0).mean():.4f}")
print(f"Given N(1)=3: E[T1] = {T_sub[:, 0].mean():.4f}")
print(f"Given N(1)=3: Correlation = {np.corrcoef(N_half, 3 - N_half)[0, 1]:.4f}")

# Total Variation Distance
dt_vals = 1 / np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
k_range = np.arange(100)
tv_dist = [0.5 * np.abs(binom.pmf(k_range, round(1 / dt), lam * dt) - poisson.pmf(k_range, lam)).sum() for dt in dt_vals]

plt.figure(figsize=(8, 5))
plt.loglog(dt_vals, tv_dist, 'bo-', label='TV Distance')
slope, intercept = np.polyfit(np.log(dt_vals), np.log(tv_dist), 1)
plt.loglog(dt_vals, np.exp(intercept) * dt_vals**slope, 'r--', label=f'Fit slope: {slope:.2f}')
plt.title("TV Distance: Binomial vs Poisson")
plt.xlabel("Delta t")
plt.ylabel("Total Variation Distance")
plt.legend()
plt.savefig("figures/HW7-Problem-3d4-Figure.png", dpi=300, bbox_inches="tight")
plt.show()
