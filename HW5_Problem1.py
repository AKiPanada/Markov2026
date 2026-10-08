import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Problem 1(b)
p = np.zeros((5, 5))
for k in range(5):
    if k < 4:
        p[k, k+1] = (4 - k) / 4
    if k > 0:
        p[k, k-1] = k / 4

q = np.zeros((61, 5))
q[0, 0] = 1.0

for n in range(60):
    q[n+1] = q[n] @ p

print(f"q_50: {q[50]}")
print(f"q_51: {q[51]}")

running_avg = np.cumsum(q, axis=0) / np.arange(1, 62)[:, None]

plt.figure(figsize=(10,6))

plt.plot(q[:, 2], label="q_n(2)", marker='o', markersize=3)
plt.plot(q[:, 4], label="q_n(4)", marker='o', markersize=3)
plt.plot(running_avg[:, 2], label="Running Avg (2)", linestyle='--')
plt.plot(running_avg[:, 4], label="Running Avg (4)", linestyle='--')
plt.xlabel("Step n")
plt.ylabel("Probability")
plt.legend()
plt.title("Probabilities over time (a=1, b=1)")

# Save figure to file system
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW5-Problem-1-Figure.png", dpi=300, bbox_inches="tight")

# Display figure
plt.show()

# Problem 1(c)
a, b = 0.3, 0.1
p = np.zeros((5, 5))
for k in range(5):
    if k < 4:
        p[k, k+1] = a * (4 - k) / 4
    if k > 0:
        p[k, k-1] = b * k / 4
    p[k, k] = 1 - (p[k, k+1] if k < 4 else 0) - (p[k, k-1] if k > 0 else 0)

A = (p - np.eye(5)).T
A[-1, :] = 1
b_vec = np.zeros(5)
b_vec[-1] = 1
pi_solve = np.linalg.solve(A, b_vec)

theta = a / (a + b)
pi_formula = np.array([math.comb(4, k) * (theta**k) * ((1-theta)**(4-k)) for k in range(5)])
print(f"pi (linear solve): {pi_solve}")
print(f"pi (formula): {pi_formula}")

q = np.zeros(5)
q[0] = 1.0
n = 0
while True:
    if np.max(np.abs(q - pi_solve)) < 1e-6:
        break
    q = q @ p
    n += 1
print(f"Smallest n for convergence: {n}")

eigenvalues = np.sort(np.abs(np.linalg.eigvals(p)))
print(f"Second largest eigenvalue modulus: {eigenvalues[-2]:.6f}")
