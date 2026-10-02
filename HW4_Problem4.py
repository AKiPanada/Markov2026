import numpy as np
import os
import matplotlib.pyplot as plt


np.random.seed(42)
N_SIM = 10000

print("--- Problem 4 ---")

# States: 0=U, 1=I, 2=M, 3=F(Folded), 4=A(Aggregated)
# Transition matrix for simulation
p = np.array([
    [0.00, 0.50, 0.50, 0.00, 0.00],
    [0.25, 0.00, 0.00, 0.50, 0.25],
    [0.75, 0.00, 0.00, 0.00, 0.25],
    [0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 1.00]
])

results = {'U': [], 'I': [], 'M': []}
states_map = {0: 'U', 1: 'I', 2: 'M'}

# Simulate chains
for start_state in [0, 1, 2]:
    for _ in range(N_SIM):
        state = start_state
        steps = 0
        while state < 3:
            # Inverse transform sampling
            state = np.random.choice(5, p=p[state])
            steps += 1
        fate = 'F' if state == 3 else 'A'
        results[states_map[start_state]].append({'fate': fate, 'time': steps})

# Calculate empirical hat statistics
stats = {}
for start, data in results.items():
    folds = [d['time'] for d in data if d['fate'] == 'F']
    aggs = [d['time'] for d in data if d['fate'] == 'A']
    stats[start] = {
        'h_hat': len(folds) / N_SIM,
        'g_hat': np.mean([d['time'] for d in data]),
        'tau_F_hat': np.mean(folds),
        'tau_A_hat': np.mean(aggs)
    }

print("Simulated Hat Statistics (10^4 chains per state):")
for start, st in stats.items():
    print(f"Start {start}: h_hat={st['h_hat']:.3f}, g_hat={st['g_hat']:.3f}, "
            f"tau_F_hat={st['tau_F_hat']:.3f}, tau_A_hat={st['tau_A_hat']:.3f}")

# Exact PMF Calculation
Q = p[0:3, 0:3]
R_F = p[0:3, 3]
R_A = p[0:3, 4]

# Exact h values from 1(c) for normalization
h = np.array([0.5, 0.625, 0.375])

max_steps = 25
steps_range = np.arange(1, max_steps + 1)

pmf_F = []
pmf_A = []
Q_n = np.eye(3)

# Compute Exact PMFs
for n in steps_range:
    # (Q^(n-1) R)_x / P(fate)
    p_F_n = Q_n @ R_F
    p_A_n = Q_n @ R_A

    # We want start state I (index 1)
    pmf_F.append(p_F_n[1] / h[1])
    pmf_A.append(p_A_n[1] / (1 - h[1]))

    Q_n = Q_n @ Q

# Plotting for start state I
times_F = [d['time'] for d in results['I'] if d['fate'] == 'F']
times_A = [d['time'] for d in results['I'] if d['fate'] == 'A']

plt.figure(figsize=(10, 6))

# Folded Histogram
plt.subplot(1, 2, 1)
plt.hist(times_F, bins=np.arange(0.5, max_steps+1.5, 1), density=True,
            alpha=0.5, color='blue', label='Simulated (Folded)')
plt.plot(steps_range, pmf_F, 'bo-', label='Exact PMF')
plt.title("Conditional PMF of Time T (Start=I, Fate=Folded)")
plt.xlabel("Steps (n)")
plt.ylabel("Probability")
plt.xticks(steps_range[::2])
plt.legend()

# Aggregated Histogram
plt.subplot(1, 2, 2)
plt.hist(times_A, bins=np.arange(0.5, max_steps+1.5, 1), density=True,
            alpha=0.5, color='red', label='Simulated (Aggregated)')
plt.plot(steps_range, pmf_A, 'ro-', label='Exact PMF')
plt.title("Conditional PMF of Time T (Start=I, Fate=Aggregated)")
plt.xlabel("Steps (n)")
plt.ylabel("Probability")
plt.xticks(steps_range[::2])
plt.legend()

plt.tight_layout()

# Save figure to file system
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW4-Problem-4-Figure.png", dpi=300, bbox_inches="tight")

# Display figure
plt.show()

