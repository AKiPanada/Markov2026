import numpy as np
import matplotlib.pyplot as plt
import os

# --- Part (b): Inverse Transform Test ---
U = np.random.rand(100000)
z_inversion = 1 - np.sqrt(1 - U)
print(f"Part (b) Empirical Mean (Inversion): {np.mean(z_inversion):.5f} (Theory: 0.33333)\n")

# --- Part (c): Vectorized Simulation ---
N_max = 10000
R = 1000

# Start network at N=3 with C(3)=1
C = np.ones(R)

for N in range(3, N_max):
    U = np.random.rand(R)
    # Probability to add a core node is C(N)/N (since mu = 1)
    prob_add = C / N
    C += (U < prob_add)

# Target variable z = C/N at N_max
z_sim = C / N_max

emp_mean = np.mean(z_sim)
emp_std = np.std(z_sim)
emp_ratio = emp_std / emp_mean
theory_ratio = 1 / np.sqrt(2) # ~0.7071

print(f"Part (c) Simulation Results:")
print(f"Empirical Mean: {emp_mean:.5f} (Theory: 0.33333)")
print(f"Emp. Std/Mean Ratio: {emp_ratio:.5f} (Theory: {theory_ratio:.5f})")
print(f"Smallest Core C seen: {int(np.min(C))} nodes")
print(f"Largest Core C seen:  {int(np.max(C))} nodes")

plt.figure(figsize=(10,6))
plt.hist(z_sim, bins=50, density=True, color='lightgreen', edgecolor='black', alpha=0.7, label='Empirical PDF z=C/N')

# Overlay theory
z_vals = np.linspace(0, 1, 200)
h_vals = 2 * (1 - z_vals)
plt.plot(z_vals, h_vals, color='darkgreen', lw=2, label='Theory h(z) = 2(1-z)')

plt.title('Growth by Redirection Core Fraction Simulation (R=10^3 realizations)')
plt.xlabel('Fraction z = C/N')
plt.ylabel('Density')
plt.legend()

# Save figure to file system
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW2-Problem-5-Figure.png", dpi=300, bbox_inches="tight")

# Display figure
plt.show()