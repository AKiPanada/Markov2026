import numpy as np


# 3(b): Convergence step for 10^-6 variation
# 1/3 * (1/4)^n < 10^-6
n = np.ceil(np.log(3e-6) / np.log(0.25))
print(f"3(b): Smallest n for convergence = {int(n)}\n")
