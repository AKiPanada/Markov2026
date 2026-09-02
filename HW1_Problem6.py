import os
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_probability(N_values):
    estimates = []

    for N in N_values:
        # Generate independent Uniform[0,1] samples
        X = np.random.uniform(0, 1, N)
        Y = np.random.uniform(0, 1, N)
        Z = np.random.uniform(0, 1, N)

        # Check inequalities
        condition1 = (X**2 + Y**2) < Z
        condition2 = (Z**2) > (X * Y)

        # Compute fraction of samples satisfying both
        prob_estimate = np.sum(condition1 & condition2) / N
        estimates.append(prob_estimate)

    return estimates

if __name__ == "__main__":
    # Define sample sizes on a logarithmic scale
    N_values = np.logspace(2, 7, num=15, dtype=int)

    # Get Monte Carlo estimates
    estimates = monte_carlo_probability(N_values)

    # Analytical value calculated in 6a
    analytical_val = (23 * np.pi) / 192

    # Show info in terminal for reference
    print("Analytic value: ", analytical_val, "\nFinal error:", abs(estimates[-1] - analytical_val))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogx(N_values, estimates, marker='o', linestyle='-', label='Monte Carlo Estimate')

    # Draw analytic value as a horizontal line
    plt.axhline(analytical_val, color='red', linestyle='--', label=f'Analytical Value ({analytical_val:.4f})')

    # Formatting
    plt.xlabel('Sample Size N (Logarithmic Scale)')
    plt.ylabel('Probability Estimate')
    plt.title('Monte Carlo Estimation of P(X^2 + Y^2 < Z and Z^2 > XY)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Save figure to file system
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/HW1-Problem-6-Figure.png", dpi=300, bbox_inches="tight")

    # Display Plot
    plt.show()