import numpy as np

p_1 = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.75, 0.25],
    [0.5, 0.0, 0.5]
])
p_50 = np.linalg.matrix_power(p_1, 50)
print("Problem 1(c): p^50 matrix:")
print(np.round(p_50, 4))
print("-" * 30)