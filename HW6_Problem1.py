# HW6_Problem1.py
import numpy as np

# --- Problem 1(c) ---
res = []
for m in range(3, 13):
    p = np.zeros((m, m))
    for i in range(m-1):
        p[i, i+1] = 1
    p[m-1, 0] = 0.5
    p[m-1, 1] = 0.5
    
    B = p > 0
    M = 1
    while not np.all(B):
        B = B @ (p > 0)
        M += 1
        
    w, _ = np.linalg.eig(p)
    w_sorted = np.sort(np.abs(w))
    bound = (m-1)**2 + 1
    res.append((m, M, bound, w_sorted[-2]))

print("m | M  | Bound | lambda_2")
for row in res:
    print(f"{row[0]:2d}| {row[1]:3d}| {row[2]:3d}   | {row[3]:.6f}")
