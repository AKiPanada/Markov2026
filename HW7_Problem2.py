import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from itertools import combinations
import os

rates = (3, 2, 1.5, 1.2, 1, 0.75)
num_students = len(rates)

@lru_cache(None)
def EM(S, lam):
    if not S: return 0.0
    tot = sum(lam[i] for i in S)
    return 1 / tot + sum(lam[i] / tot * EM(S - frozenset([i]), lam) for i in S if len(S) > 1)

@lru_cache(None)
def prob_last(S, j, lam):
    if len(S) == 1: return 1.0 if j in S else 0.0
    tot = sum(lam[i] for i in S)
    return sum(lam[i] / tot * prob_last(S - frozenset([i]), j, lam) for i in S if i != j)

everyone = frozenset(range(num_students))
print(f"Exact E[M] via DP: {EM(everyone, rates):.4f}")

E_M_alt = sum((-1)**(r + 1) * sum(1 / sum(g) for g in combinations(rates, r)) for r in range(1, num_students + 1))
print(f"Exact E[M] via Alternating Formula: {E_M_alt:.4f}")

p_last_exact = np.array([prob_last(everyone, j, rates) for j in range(num_students)])
print("Exact Prob Last per student:", p_last_exact)

rng = np.random.default_rng(4560)
N_sim = 200000
exams = rng.exponential(1 / np.array(rates), size=(N_sim, num_students))
M_sim = exams.max(axis=1)

print(f"Simulated E[M]: {M_sim.mean():.4f}")
print("Simulated Prob Last:", np.bincount(exams.argmax(axis=1)) / N_sim)

# First student finish check
first_times = exams.min(axis=1)
first_idx = exams.argmin(axis=1)
print("Mean first finish time by who was first:",
      np.array([first_times[first_idx == i].mean() for i in range(num_students)]))
print(f"1/Lambda = {1 / sum(rates):.4f}")

t_vals = np.linspace(0, max(M_sim), 500)
density = np.gradient(np.prod(1 - np.exp(-np.outer(t_vals, rates)), axis=1), t_vals)

plt.figure(figsize=(10, 5))
plt.hist(M_sim, bins=100, density=True, alpha=0.6, label='Simulated Histogram')
plt.plot(t_vals, density, 'k-', lw=2, label='Exact Density')
plt.title("Maximum Finish Time M")
plt.xlabel("Time (hours)")
plt.ylabel("Density")
plt.legend()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW7-Problem-2d1-Figure.png", dpi=300, bbox_inches="tight")
plt.show()

ks = range(1, 13)
EMk = [EM(frozenset(range(k)), tuple(range(1, k + 1))) for k in ks]

plt.figure(figsize=(8, 5))
plt.plot(ks, EMk, 'bo-', label='E[M]')
plt.axhline(2 * np.log(2), color='r', linestyle='--', label='2 ln(2)')
plt.title("E[M] vs number of students k")
plt.xlabel("k students")
plt.ylabel("E[M]")
plt.legend()
plt.savefig("figures/HW7-Problem-2d2-Figure.png", dpi=300, bbox_inches="tight")
plt.show()
