# HW6_Problem3.py
import numpy as np
import scipy.sparse.linalg as sla
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

# --- Problem 3(c) ---
def compute_h1(N, K, R0_array):
    h1_vals = []
    for R0 in R0_array:
        rho = [1.0]
        for j in range(1, K):
            rho.append(rho[-1] * (N / (R0 * (N - j))))
        h1_vals.append(1.0 / np.sum(rho))
    return np.array(h1_vals)

R0_vals = np.linspace(0.5, 3, 100)
h1_100 = compute_h1(100, 10, R0_vals)
h1_1000 = compute_h1(1000, 100, R0_vals)
limit = np.maximum(0, 1 - 1/R0_vals)

plt.figure(figsize=(10, 6))
plt.plot(R0_vals, h1_100, label='N=100')
plt.plot(R0_vals, h1_1000, label='N=1000')
plt.plot(R0_vals, limit, 'k--', label='1 - 1/R0')
plt.xlabel('R0')
plt.ylabel('h1')
plt.legend()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW6-Problem-3-Figure.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Problem 3(d) ---
N = 100
K = 10
gamma = 1.0
dt = 0.005

def solve_e1(R0):
    beta = R0 * gamma / N
    size = K - 1
    main_diag = np.zeros(size)
    upper_diag = np.zeros(size - 1)
    lower_diag = np.zeros(size - 1)
    
    for n in range(1, K):
        pn = beta * n * (N - n) * dt
        qn = gamma * n * dt
        idx = n - 1
        main_diag[idx] = pn + qn
        if n < K - 1:
            upper_diag[idx] = -pn
        if n > 1:
            lower_diag[idx - 1] = -qn
            
    A = sp.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
    b = np.ones(size)
    e = sla.spsolve(A, b)
    return e[0] * dt

print("Analytic e1*dt:")
for R0 in [0.5, 0.8, 1.5, 3.0]:
    print(f"R0 = {R0}: {solve_e1(R0):.4f}")

R0 = 1.5
beta = R0 * gamma / N
np.random.seed(42)
N_sim = 10000
total_steps = 0
for _ in range(N_sim):
    n = 1
    steps = 0
    while 0 < n < K:
        pn = beta * n * (N - n) * dt
        qn = gamma * n * dt
        r = np.random.rand()
        if r < pn:
            n += 1
        elif r < pn + qn:
            n -= 1
        steps += 1
    total_steps += steps
print(f"\nSimulated e1*dt for R0=1.5: {total_steps / N_sim * dt:.4f}")
