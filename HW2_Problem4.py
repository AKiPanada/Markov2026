import numpy as np
import matplotlib.pyplot as plt
import os

N = 100000
a = 0.9
lambda_f = 1000.0
lambda_s = 10.0

# Bernoulli component selection
U1 = np.random.rand(N)
# If U1 <= a, use lambda_f, else use lambda_s
lambda_array = np.where(U1 <= a, lambda_f, lambda_s)

# 2. Inverse transform for exponential given selected rate
U2 = np.random.rand(N)
dwell_times = - (1.0 / lambda_array) * np.log(U2)

empirical_mean = np.mean(dwell_times)
empirical_prob_gt_50ms = np.mean(dwell_times > 0.05)
theoretical_mean = 0.0109
theoretical_prob = 0.06065 # Calculated derived above

print(f"Empirical Mean: {empirical_mean:.5f} (Theory: {theoretical_mean:.5f})")
print(f"Empirical P(T > 50ms): {empirical_prob_gt_50ms:.5f} (Theory: {theoretical_prob:.5f})")

# Histogram on semilog vertical axes (linear x, log y)
plt.figure(figsize=(10,6))
bins = np.linspace(0, max(dwell_times), 100)
plt.hist(dwell_times, bins=bins, density=True, color='mediumpurple', alpha=0.7, label='Empirical PDF')

# Overlay theory
t_vals = np.linspace(0, max(dwell_times), 500)
f_vals = a * lambda_f * np.exp(-lambda_f * t_vals) + (1 - a) * lambda_s * np.exp(-lambda_s * t_vals)
plt.plot(t_vals, f_vals, color='black', lw=2, label='Theory f(t)')

# Apply semilog vertical axes
plt.yscale('log')
plt.title('Ion Channel Dwell Times Mixture (Semilog-Y Axes)')
plt.xlabel('Dwell Time T (seconds)')
plt.ylabel('Density (log scale)')
plt.legend()

# Save figure to file system
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW2-Problem-4-Figure.png", dpi=300, bbox_inches="tight")

# Display figure
plt.show()