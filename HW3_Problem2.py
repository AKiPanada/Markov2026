import numpy as np

p_2 = np.array([
    [1/2, 1/2, 0,   0,   0,   0],
    [1/3, 0,   1/3, 0,   1/3, 0],
    [0,   0,   1/4, 3/4, 0,   0],
    [0,   0,   1,   0,   0,   0],
    [0,   0,   0,   0,   0,   1],
    [0,   0,   0,   0,   1,   0]
])

p_20 = np.linalg.matrix_power(p_2, 20)
p_21 = np.linalg.matrix_power(p_2, 21)
print("Problem 2(d): p^20 matrix:")
print(np.round(p_20, 3))
print("Problem 2(d): p^21 matrix:")
print(np.round(p_21, 3))
print("-" * 30)