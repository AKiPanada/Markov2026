import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import poisson


np.random.seed(42)

print("--- Problem 2 ---")

# 2(c)
lam = 12.0
mu = 3.0
rho = lam / mu
T = 500.0

t = 0.0
n = 0
time_in_state = {}

while t < T:
    rate = lam + n * mu
    dt = np.random.exponential(1.0 / rate)
    if t + dt > T:
        time_in_state[n] = time_in_state.get(n, 0.0) + (T - t)
        t = T
    else:
        time_in_state[n] = time_in_state.get(n, 0.0) + dt
        t += dt
        if np.random.rand() < lam / rate:
            n += 1
        else:
            n -= 1

max_state = max(time_in_state.keys())
states = np.arange(max_state + 1)
emp_probs = np.array([time_in_state.get(k, 0.0) / T for k in states])
theo_probs = poisson.pmf(states, rho)

max_diff = np.max(np.abs(emp_probs - theo_probs))
time_avg_X = np.sum(states * emp_probs)

c = 0
while 1.0 - poisson.cdf(c, rho) > 0.01:
    c += 1

print(f"2(c): Largest difference = {max_diff:.4f}")
print(f"2(c): Time average of X_t = {time_avg_X:.4f} (rho = {rho:.4f})")
print(f"2(c): Smallest fleet size c = {c} (P(X > {c}) = {1.0 - poisson.cdf(c, rho):.5f})\\n")

plt.figure(figsize=(10, 6))
width = 0.4
plt.bar(states - width / 2, emp_probs, width=width, alpha=0.6, color="blue", label="Simulated Fraction of Time")
plt.bar(states + width / 2, theo_probs, width=width, alpha=0.6, color="red", label="Poisson(4) PMF")
plt.title("M/M/inf Bike Share Simulation vs. Stationary Poisson(4) Distribution (T = 500 h)")
plt.xlabel("Number of Bikes Out (n)")
plt.ylabel("Fraction of Time / Probability")
plt.xticks(states)
plt.legend()
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW10-Problem-2-Figure.png", dpi=300, bbox_inches="tight")
plt.savefig("HW10-Problem-2-Figure.png", dpi=300, bbox_inches="tight")
plt.show()
