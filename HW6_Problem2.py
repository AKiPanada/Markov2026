# HW6_Problem2.py
import pandas as pd
import numpy as np

# --- Problem 2(c) ---
def h(x):
    return 0.5 + 0.5 * np.sign(x - 0.5) * np.sqrt(np.abs(2*x - 1))

df = pd.read_csv('Homework6-data/big12.csv')
teams = sorted(list(set(df['home']).union(set(df['away']))))
m = len(teams)
team2id = {t: i for i, t in enumerate(teams)}

df_2025 = df[(df['season'] == 2025) & (df['completed'] == 1)]
h_2025 = np.zeros((m, m))
counts_2025 = np.zeros((m, m))
wins_2025 = np.zeros(m)

for _, row in df_2025.iterrows():
    i = team2id[row['home']]
    j = team2id[row['away']]
    S = row['home_pts']
    T = row['away_pts']
    
    hij = h((S + 1) / (S + T + 2))
    hji = h((T + 1) / (S + T + 2))
    
    h_2025[i, j] += hij
    h_2025[j, i] += hji
    counts_2025[i, j] += 1
    counts_2025[j, i] += 1
    if S > T:
        wins_2025[i] += 1
    elif S < T:
        wins_2025[j] += 1

Pi_last_season = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        if i != j:
            if counts_2025[i, j] > 0:
                Pi_last_season[i, j] = h_2025[i, j] / counts_2025[i, j]
            else:
                Pi_last_season[i, j] = 0.5

Pi_flat = np.ones((m, m)) * 0.5
np.fill_diagonal(Pi_flat, 0)

strengths = wins_2025 + 1
Pi_own = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        if i != j:
            Pi_own[i, j] = strengths[i] / (strengths[i] + strengths[j])

df_2026 = df[(df['season'] == 2026) & (df['completed'] == 1)]
h_2026 = np.zeros((m, m))
n_games = np.zeros(m)

for _, row in df_2026.iterrows():
    i = team2id[row['home']]
    j = team2id[row['away']]
    S = row['home_pts']
    T = row['away_pts']
    
    hij = h((S + 1) / (S + T + 2))
    hji = h((T + 1) / (S + T + 2))
    
    h_2026[i, j] += hij
    h_2026[j, i] += hji
    n_games[i] += 1
    n_games[j] += 1

def build_A(Pi, w=1):
    A = np.zeros((m, m))
    for i in range(m):
        denom = n_games[i] + w * (m - 1)
        for j in range(m):
            if i != j:
                A[i, j] = (h_2026[i, j] + w * Pi[i, j]) / denom
    return A

def compute_ranking(A):
    w, v = np.linalg.eig(A)
    idx = np.argsort(np.abs(w))[::-1]
    w = w[idx]
    v = v[:, idx]
    r = np.real(v[:, 0])
    if np.sum(r) < 0:
        r = -r
    return np.real(w[0]), w[1], r / np.sum(r)

res_flat = compute_ranking(build_A(Pi_flat))
res_last = compute_ranking(build_A(Pi_last_season))
res_own = compute_ranking(build_A(Pi_own))

print(f"Flat: lambda* = {res_flat[0]:.4f}, |lambda_2/lambda*| = {np.abs(res_flat[1]/res_flat[0]):.4f}")
print(f"Last: lambda* = {res_last[0]:.4f}, |lambda_2/lambda*| = {np.abs(res_last[1]/res_last[0]):.4f}")
print(f"Own:  lambda* = {res_own[0]:.4f}, |lambda_2/lambda*| = {np.abs(res_own[1]/res_own[0]):.4f}\n")

df_unplayed = df[(df['season'] == 2026) & (df['completed'] == 0) & (df['week'] == 6)]
r_last = res_last[2]

print("Predictions for Week 6:")
for _, row in df_unplayed.iterrows():
    h_idx = team2id[row['home']]
    a_idx = team2id[row['away']]
    r_h = r_last[h_idx]
    r_a = r_last[a_idx]
    winner = row['home'] if r_h > r_a else row['away']
    print(f"{row['home']} ({r_h:.4f}) vs {row['away']} ({r_a:.4f}) -> {winner} wins")
