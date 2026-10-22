import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import iqr

flashes = np.array([-1.77, 2.47, 1.34, 2.18, 13.05, 1.60, 1.50, -1.97, 0.79, 1.50, 2.00, 2.63, 1.19, -2.31, 0.87])
n = len(flashes)

def log_g(alpha, b):
    return n * b - np.sum(np.log(np.exp(2 * b) + (flashes - alpha)**2))

def iat(y):
    y = y - y.mean()
    tau, k = 1, 1
    while True:
        r = np.dot(y[:-k], y[k:]) / np.dot(y, y)
        if r < 0: return tau
        tau += 2 * r
        k += 1

rng = np.random.default_rng(4560)
T_steps, burn = 50000, 5000
step_sizes = [0.02, 0.5, 5]
chains = {}

plt.figure(figsize=(10, 6))
for idx, s in enumerate(step_sizes):
    z = np.zeros((T_steps + 1, 2))
    acc = 0
    for t in range(T_steps):
        y = z[t] + s * rng.standard_normal(2)
        if np.log(rng.random()) < log_g(*y) - log_g(*z[t]):
            z[t + 1] = y
            acc += 1
        else:
            z[t + 1] = z[t]
    chains[s] = z
    tau = iat(z[burn:, 0])
    print(f"s = {s}: Acceptance Rate = {acc / T_steps:.4f}, Autocorr Time (alpha) = {tau:.1f}")

    plt.subplot(1, 3, idx + 1)
    plt.plot(z[:400, 0], label=f's={s}')
    plt.title(f"First 400 steps (s={s})")
    plt.xlabel("Step")
    plt.ylabel("Alpha")

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW7-Problem-1c1-Figure.png", dpi=300, bbox_inches="tight")
plt.show()

# Best s is 0.5
a, b = chains[0.5][burn:, 0], chains[0.5][burn:, 1]

# Grid for exact posterior contours
A, B = np.meshgrid(np.linspace(-2, 5, 400), np.linspace(-2.5, 2, 400))
G = np.exp(np.vectorize(log_g)(A, B))
G /= G.sum()
pa, pb = G.sum(axis=0), G.sum(axis=1)

plt.figure(figsize=(8, 6))
plt.contour(A, B, G, levels=8, colors='k')
plt.plot(a[::10], b[::10], '.', ms=2, alpha=0.1, color='red')
plt.xlim(0, 3)
plt.ylim(-2, 1.5)
plt.title("MCMC Samples vs Exact Posterior Contours")
plt.xlabel("alpha")
plt.ylabel("log beta (b)")
plt.savefig("figures/HW7-Problem-1c2-Figure.png", dpi=300, bbox_inches="tight")
plt.show()

# Posterior means and 90% intervals (Chain vs Grid)
print(f"Chain Alpha - Mean: {a.mean():.4f}, 90% CI: {np.percentile(a, [5, 95])}")
print(f"Grid Alpha  - Mean: {(A[0] * pa).sum():.4f}, 90% CI: {np.interp([0.05, 0.95], np.cumsum(pa), A[0])}")
print(f"Chain Beta  - Mean: {np.exp(b).mean():.4f}, 90% CI: {np.percentile(np.exp(b), [5, 95])}")
print(f"Grid Beta   - Mean: {(np.exp(B[:, 0]) * pb).sum():.4f}, 90% CI: {np.exp(np.interp([0.05, 0.95], np.cumsum(pb), B[:, 0]))}")
print(f"Sample Mean: {np.mean(flashes):.4f}, Sample Median: {np.median(flashes):.4f}")

# Problem 1(d)
for m in (10, 100, 1000):
    data = 1.5 + 0.8 * np.tan(np.pi * (rng.random((2000, m)) - 0.5))
    scatter_mean = iqr(data.mean(axis=1)) / 1.349
    scatter_median = iqr(np.median(data, axis=1)) / 1.349
    print(f"n = {m:4d} | Scatter Mean: {scatter_mean:.4f} | Scatter Median: {scatter_median:.4f}")
