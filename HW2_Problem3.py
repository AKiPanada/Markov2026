import numpy as np
import matplotlib.pyplot as plt
import time
import os

def run_rejection_sampling(lam, target_samples=10000):
    c = 1 / (lam * (1 - lam) * np.e)
    
    samples = np.zeros(target_samples)
    accepted = 0
    proposals = 0
    
    start_time = time.time()
    
    while accepted < target_samples:
        # Sample from proposal g(x) = lambda * e^(-lambda * x) using inversion
        U1 = np.random.rand()
        x = - (1 / lam) * np.log(U1)
        
        # Acceptance condition U < f(x)/(c*g(x))
        U2 = np.random.rand()
        f_x = x * np.exp(-x)
        g_x = lam * np.exp(-lam * x)
        
        if U2 < (f_x / (c * g_x)):
            samples[accepted] = x
            accepted += 1
        proposals += 1
        
    end_time = time.time()
    
    empirical_acceptance_rate = target_samples / proposals
    mean_time_per_sample = (end_time - start_time) / target_samples
    
    return samples, c, empirical_acceptance_rate, mean_time_per_sample

# Lambda = 0.5
samples_05, c_05, acc_rate_05, time_05 = run_rejection_sampling(0.5)
print("Lambda = 0.5")
print(f"1/c (Theoretical Acceptance Rate): {1/c_05:.4f}")
print(f"Empirical Acceptance Rate:         {acc_rate_05:.4f}")
print(f"Mean Time per accepted sample:     {time_05:.7f} sec\n")

# Lambda = 0.2
samples_02, c_02, acc_rate_02, time_02 = run_rejection_sampling(0.2)
print("Lambda = 0.2")
print(f"1/c (Theoretical Acceptance Rate): {1/c_02:.4f}")
print(f"Empirical Acceptance Rate:         {acc_rate_02:.4f}")
print(f"Mean Time per accepted sample:     {time_02:.7f} sec")

# Plotting for Lambda = 0.5
plt.figure(figsize=(10,6))
plt.hist(samples_05, bins=100, density=True, color='lightskyblue', edgecolor='black', alpha=0.7, label='Empirical PDF (Bin width ~0.1)')
x_vals = np.linspace(0, max(samples_05), 500)
f_vals = x_vals * np.exp(-x_vals)
plt.plot(x_vals, f_vals, color='red', lw=2, label='Theory: f(x) = x * exp(-x)')
plt.title('Acceptance-Rejection Sampling with Lambda=0.5')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()

# Save figure to file system
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW2-Problem-3-Figure.png", dpi=300, bbox_inches="tight")

# Show figure
plt.show()
