import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import brentq


print("--- Problem 3 ---")

# 3(b)
Q = np.array([
    [-2.0,  2.0,  0.0,  0.0],
    [ 1.0, -3.0,  2.0,  0.0],
    [ 0.0,  2.0, -4.0,  2.0],
    [ 0.0,  0.0,  3.0, -3.0]
])

eigvals = np.sort(np.real(np.linalg.eigvals(Q)))[::-1]
lam2 = eigvals[1]

print(f"3(b): Eigenvalues = {np.round(eigvals, 4)}")
print(f"3(b): Sum of eigenvalues = {np.sum(eigvals):.4f}, tr(Q) = {np.trace(Q):.4f}\n")

# 3(c)
x0 = np.array([1.0, 0.0, 0.0, 0.0])
pi = np.array([3.0, 6.0, 6.0, 4.0]) / 19.0

def get_x(t):
    return x0 @ expm(t * Q)

def get_tv(t):
    return 0.5 * np.sum(np.abs(get_x(t) - pi))

t_vals = np.linspace(0.0, 5.0, 500)
x_vals = np.array([get_x(t) for t in t_vals])
tv_vals = np.array([get_tv(t) for t in t_vals])

t_001 = brentq(lambda t: get_tv(t) - 0.01, 0.0, 5.0)
t_est = np.log(100.0) / abs(lam2)

print(f"3(c): Time for TV = 0.01 is {t_001:.4f} hr")
print(f"3(c): ln(100)/|lam2| = {t_est:.4f} hr\n")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
for i in range(4):
    plt.plot(t_vals, x_vals[:, i], label=f"$x_{i}(t)$")
plt.title("State Probabilities $x_i(t)$")
plt.xlabel("Time $t$ (hours)")
plt.ylabel("Probability")
plt.xlim(0, 5)
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(t_vals, tv_vals, 'b-', linewidth=2, label="$TV(t)$")
plt.semilogy(t_vals, np.exp(lam2 * t_vals), 'r--', linewidth=2, label="$e^{\\lambda_2 t}$ (slope $\\lambda_2$)")
plt.axhline(0.01, color='gray', linestyle=':', label="$TV = 0.01$")
plt.title("Total Variation Distance $TV(t)$ (Log Scale)")
plt.xlabel("Time $t$ (hours)")
plt.ylabel("Total Variation Distance")
plt.xlim(0, 5)
plt.legend()

plt.tight_layout()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW9-Problem-3-Figure.png", dpi=300, bbox_inches="tight")
plt.show()
