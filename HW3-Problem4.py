import numpy as np
import matplotlib.pyplot as plt
import os


def custom_erf(z):
    z = np.asarray(z)
    res = np.zeros_like(z, dtype=float)
    
    for i, val in enumerate(z.flat):
        u = np.linspace(0, val, 200)
        res.flat[i] = (2.0 / np.sqrt(np.pi)) * np.trapezoid(np.exp(-u**2), u)
    return res

# Simulation Parameters
R = 20000      # 2 x 10^4 independent simulations
T = 10000      # 10^4 time steps
d0 = 10        # Initial lion position

print("Running simulation...")


lamb_steps = np.random.choice([-1, 1], size=(R, T))
lion1_steps = np.random.choice([-1, 1], size=(R, T))
lion2_steps = np.random.choice([-1, 1], size=(R, T))


lamb_pos = np.cumsum(lamb_steps, axis=1)
lion1_pos = d0 + np.cumsum(lion1_steps, axis=1)
lion2_pos = d0 + np.cumsum(lion2_steps, axis=1)


cap1 = (lamb_pos == lion1_pos)
cap2 = (lamb_pos == lion2_pos)
cap_both = cap1 | cap2

time1 = np.argmax(cap1, axis=1)
time1[(time1 == 0) & (~cap1[:, 0])] = T + 1

time2 = np.argmax(cap_both, axis=1)
time2[(time2 == 0) & (~cap_both[:, 0])] = T + 1

t_vals = np.arange(1, T + 1)
S1 = np.array([np.sum(time1 > t) for t in t_vals]) / R
S2 = np.array([np.sum(time2 > t) for t in t_vals]) / R

S1_continuum = custom_erf(d0 / (2.0 * np.sqrt(t_vals)))

fit_mask = (t_vals >= 100) & (t_vals <= 10000)
log_t = np.log10(t_vals[fit_mask])


beta1 = -np.polyfit(log_t, np.log10(S1[fit_mask]), 1)[0]
beta2 = -np.polyfit(log_t, np.log10(S2[fit_mask]), 1)[0]

print(f"Fitted Exponent beta_1: {beta1:.4f}")
print(f"Fitted Exponent beta_2: {beta2:.4f}")


print("\n--- Tabulated Values ---")
print(f"{'t':<8}{'S2(t)':<12}{'S1(t)^2':<12}{'Gap (S2 - S1^2)':<15}")
for t_check in [100, 1000, 10000]:
    idx = t_check - 1
    val_s2 = S2[idx]
    val_s1_sq = S1[idx]**2
    print(f"{t_check:<8}{val_s2:<12.5f}{val_s1_sq:<12.5f}{val_s2 - val_s1_sq:<15.5f}")


plt.figure(figsize=(9, 6))

plt.loglog(t_vals, S1, label=f'$S_1(t)$ Simulated (fit $\\beta_1 = {beta1:.3f}$)', color='blue', lw=1.5)
plt.loglog(t_vals, S1_continuum, 'k--', label='$S_1(t)$ Continuum Prediction', lw=1.5)
plt.loglog(t_vals, S2, label=f'$S_2(t)$ Simulated (fit $\\beta_2 = {beta2:.3f}$)', color='green', lw=1.5)
plt.loglog(t_vals, S1**2, 'r:', label='$S_1(t)^2$ (Independent baseline)', lw=1.5)

plt.title(f'Capture of the Lamb: Fit Window $[10^2, 10^4]$, $\\beta_1={beta1:.3f}$, $\\beta_2={beta2:.3f}$')
plt.xlabel('Time step $t$')
plt.ylabel('Survival Probability $S(t)$')
plt.xlim([100, 10000])
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()

# Save figure to file system
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/HW3-Problem-4-Figure.png", dpi=300, bbox_inches="tight")

# Display figure
plt.show()