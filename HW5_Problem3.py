import matplotlib.pyplot as plt
import numpy as np
import os

# Problem 3(a)
p = np.zeros((8, 8))
edges = {
    0: [1, 2],
    1: [2, 3],
    2: [0, 4],
    3: [0, 2, 4],
    4: [1, 5, 6],
    5: [5],
    6: [7],
    7: [6]}
for i, outs in edges.items():
    for j in outs:
        p[i, j] = 1.0 / len(outs)

q = np.ones(8) / 8
for n in range(101):
    if n == 100:
        q_100 = q.copy()
    q = q @ p
q_101 = q

print(f"q_100: {q_100}")
print(f"q_101: {q_101}")

# Problem 3(b)
d = 0.85
E = np.ones((8, 8)) / 8
G = d * p + (1 - d) * E

q = np.ones(8) / 8
iterations = 0
while True:
    q_next = q @ G
    iterations += 1
    if np.linalg.norm(q_next - q, 1) < 1e-10:
        break
    q = q_next
pi_power = q_next

A = np.eye(8) - d * p
b_vec = np.ones(8) * (1 - d) / 8
pi_solve = np.linalg.solve(A.T, b_vec)

print(f"Iterations: {iterations}")
print(f"pi (power): {pi_power}")
print(f"pi (solve): {pi_solve}")
print(f"Ranking: {np.argsort(-pi_solve) + 1}")

# Problem 3(c)
R = 100
T_max = 10**5
surfers = np.zeros(R, dtype=int)
counts = np.zeros((R, 8))

G_cumsum = np.cumsum(G, axis=1)

errors = []
T_vals = np.logspace(2, 5, num=20, dtype=int)
T_set = set(T_vals)

for t in range(1, T_max + 1):
    rand_vals = np.random.rand(R, 1)
    surfers = (rand_vals > G_cumsum[surfers]).sum(axis=1)

    for i in range(R):
        counts[i, surfers[i]] += 1

    if t in T_set:
        pi_hat = counts / t
        rms_error = np.sqrt(np.mean(np.max(np.abs(pi_hat - pi_solve), axis=1)**2))
        errors.append((t, rms_error))

T_plot = np.array([e[0] for e in errors])
rms_plot = np.array([e[1] for e in errors])

pi_hat_single = counts[0] / T_max

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, 9), pi_hat_single, alpha=0.7, label='Estimate')
plt.plot(range(1, 9), pi_solve, 'ro', label='True pi')
plt.title("One Surfer's Estimate vs True pi")
plt.xlabel("Page")
plt.ylabel("Stationary probability")
plt.legend()

plt.subplot(1, 2, 2)
plt.loglog(T_plot, rms_plot, 'bo-', label='RMS error')
slope, intercept = np.polyfit(np.log(T_plot), np.log(rms_plot), 1)
plt.loglog(T_plot, np.exp(intercept) * T_plot**slope, 'r--', label=f'Fit slope: {slope:.2f}')
plt.xlabel("Steps T")
plt.ylabel("RMS Error")
plt.legend()
plt.tight_layout()

# Save figure to file system
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW5-Problem-3-Figure.png", dpi=300, bbox_inches="tight")

# Display figure
plt.show()
