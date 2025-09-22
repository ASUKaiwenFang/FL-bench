#!/usr/bin/env python3
"""
Differential Privacy Parameter Adjustment Example
Demonstrates how to adjust federated learning configuration based on epsilon bounds
"""

import numpy as np
from scipy.special import gammaln, logsumexp
import math

def logcomb(n, k):
    """Returns the logarithm of comb(n,k)"""
    return (gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1))

def RDP_epsilon_bound_gaussian(alpha, sigma_actual):
    """Returns the epsilon RDP bound for Gaussian mechanism with std parameter sigma_actual"""
    return 0.5 * alpha / (sigma_actual ** 2)

def cgf_subsampling_for_int_alpha(alpha: int, eps_func, sub_ratio):
    """CGF bound for subsampled mechanism"""
    alpha = int(alpha)
    log_moment_two = 2 * np.log(sub_ratio) + logcomb(alpha, 2) + np.minimum(
        np.log(4) + eps_func(2.) + np.log(1 - np.exp(-eps_func(2.))), eps_func(2.) + np.log(2))
    log_moment_j = lambda j: np.log(2) + (j - 1) * eps_func(j) + j * np.log(sub_ratio) + logcomb(alpha, j)
    all_log_moments_j = [log_moment_j(j) for j in range(3, alpha + 1, 1)]
    return logsumexp([0, log_moment_two] + all_log_moments_j)

def calculate_epsilon_bound(T, K, M, R, join_ratio, sigma_config, training_ratio=0.8):
    """
    Calculate epsilon bound for given parameters

    Args:
        T: Number of communication rounds (global_epoch)
        K: Number of local updates (local_epoch)
        M: Total number of users
        R: Number of data points per user
        join_ratio: Participation ratio per round
        sigma_config: Sigma parameter in configuration file
        training_ratio: Training data ratio
    """

    # Calculate actual parameters
    delta = 1 / (M * R * training_ratio)
    l = join_ratio  # User sampling ratio
    s = 0.2  # Assume 20% data sampling ratio

    # Convert sigma from config file to actual noise standard deviation
    # This conversion may need adjustment based on specific implementation
    sigma_gaussian_actual = sigma_config * np.sqrt(l * M)

    print(f"=== Parameter Settings ===")
    print(f"Communication rounds (T): {T}")
    print(f"Local updates (K): {K}")
    print(f"Total users (M): {M}")
    print(f"Data points per user (R): {R}")
    print(f"User participation ratio (join_ratio): {join_ratio}")
    print(f"Config file sigma: {sigma_config}")
    print(f"Actual noise standard deviation: {sigma_gaussian_actual:.2f}")
    print(f"Delta: {delta:.2e}")
    print()

    # Define RDP functions
    def eps_func_intermediate(alpha):
        def eps_func_basic(a):
            return RDP_epsilon_bound_gaussian(a, sigma_gaussian_actual)
        return K * cgf_subsampling_for_int_alpha(int(alpha), eps_func_basic, s) / (alpha - 1)

    def eps_func_final(alpha):
        return T * cgf_subsampling_for_int_alpha(int(alpha), eps_func_intermediate, l) / (alpha - 1)

    def epsilon_dp_bound(alpha):
        return eps_func_final(alpha) + np.log(1 / delta) / (alpha - 1)

    # Find optimal alpha
    alpha_range = np.arange(2, 101, 1)
    epsilon_values = [epsilon_dp_bound(alpha) for alpha in alpha_range]
    min_idx = np.argmin(epsilon_values)
    best_alpha = alpha_range[min_idx]
    best_epsilon = epsilon_values[min_idx]

    print(f"=== Results ===")
    print(f"Optimal alpha: {best_alpha}")
    print(f"Final epsilon bound: {best_epsilon:.4f}")

    return best_epsilon, best_alpha

# Example 1: Match current configuration file
print("Example 1: Current configuration file parameters")
epsilon1, alpha1 = calculate_epsilon_bound(
    T=100,           # global_epoch
    K=5,             # local_epoch
    M=100,           # Assume 100 users
    R=5000,          # Assume 5000 data points per user
    join_ratio=0.1,  # join_ratio
    sigma_config=1   # sigma in configuration file
)

print("\n" + "="*50 + "\n")

# Example 2: If you want stronger privacy protection (smaller epsilon)
print("Example 2: Stronger privacy protection (increased noise)")
epsilon2, alpha2 = calculate_epsilon_bound(
    T=100,
    K=5,
    M=100,
    R=5000,
    join_ratio=0.1,
    sigma_config=5   # Increase noise
)

print("\n" + "="*50 + "\n")

# Example 3: If you want better performance (reduced privacy protection)
print("Example 3: Better performance (reduced noise)")
epsilon3, alpha3 = calculate_epsilon_bound(
    T=100,
    K=5,
    M=100,
    R=5000,
    join_ratio=0.1,
    sigma_config=0.5  # Reduce noise
)

print("\n" + "="*50 + "\n")

# Recommendation section
print("=== Adjustment Recommendations ===")
print(f"1. Current config (sigma=1): ε = {epsilon1:.4f}")
print(f"2. Strong privacy config (sigma=5): ε = {epsilon2:.4f} - Better privacy protection")
print(f"3. Weak privacy config (sigma=0.5): ε = {epsilon3:.4f} - Better model performance")
print()
print("Choose based on your privacy requirements:")
print("- If you need ε < 1: Use sigma >= 5")
print("- If you need ε < 5: Use sigma >= 1")
print("- If you can accept ε > 10: You can use sigma = 0.5")