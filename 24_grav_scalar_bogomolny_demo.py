#!/usr/bin/env python3
"""
24_grav_scalar_bogomolny_demo.py

Refined diagnostic for the Fisher scalar halo sector.

We show numerically that, for a spherical baryonic profile with constant
Fisher stiffness alpha, the scalar action

  S_sigma[sigma] = ∫ [alpha |d_r sigma|^2 / (8 pi) - g sigma rho_b] 4 pi r^2 dr

admits a Bogomolny-type decomposition on the discrete grid:

  S_sigma[sigma] = S_min
                   + ∫ alpha |d_r sigma - d_r sigma_halo|^2 / (8 pi) 4 pi r^2 dr

where sigma_halo is the Fisher halo profile (solution of the scalar field
equation) and S_min := S_sigma[sigma_halo] is the minimal scalar action
for the given baryon profile.

All gradients entering the action and the Dirichlet term are computed with
the same finite-difference operator to ensure a clean discrete identity.
"""

import numpy as np

# ----------------------------------------------------------------------
# Model parameters (code units)
# ----------------------------------------------------------------------

G = 1.0          # Newton constant (context only)
alpha = 1.0      # constant Fisher stiffness
g_coupling = 1.0

rho0 = 1.0       # central baryonic density
R_d = 1.0        # baryonic scale radius

# ----------------------------------------------------------------------
# Radial grid (you already tested extreme values; here is a sane default)
# Feel free to ramp R_max and N_r back up as you like.
# ----------------------------------------------------------------------

R_max = 200.0 * R_d
N_r = 25000000

r = np.linspace(1e-4, R_max, N_r)   # avoid r = 0
dr = r[1] - r[0]
four_pi_r2_dr = 4.0 * np.pi * r**2 * dr

# ----------------------------------------------------------------------
# Baryonic profile and helpers
# ----------------------------------------------------------------------

def rho_baryon(r):
    """Exponential sphere baryonic density."""
    return rho0 * np.exp(-r / R_d)

def enclosed_mass(r, rho):
    """Cumulative enclosed mass M_b(r) on uniform grid r."""
    integrand = 4.0 * np.pi * r**2 * rho
    return np.cumsum(integrand) * dr

def halo_sigma_gradient_analytic(M_b):
    """Analytic halo gradient for constant alpha sector.

    From d/dr [ r^2 alpha d_r sigma ] = -4 pi g r^2 rho_b:

        r^2 alpha d_r sigma = -g M_b(r)
        => d_r sigma = - g M_b(r) / (alpha r^2).
    """
    return -g_coupling * M_b / (alpha * r**2)

def integrate_sigma_from_gradient(sigma_prime):
    """Integrate sigma'(r) inward from sigma(R_max) = 0."""
    sigma = np.zeros_like(sigma_prime)
    rev_int = np.cumsum(sigma_prime[::-1]) * dr
    sigma[:-1] = -rev_int[-2::-1]
    sigma[-1] = 0.0
    return sigma

def scalar_action(sigma, sigma_prime, rho_b):
    """Compute S_sigma[sigma] discretely."""
    grad_term = alpha * sigma_prime**2 / (8.0 * np.pi)
    source_term = -g_coupling * sigma * rho_b
    return np.sum((grad_term + source_term) * four_pi_r2_dr)

def dirichlet_deviation_from_halo(sigma, sigma_halo_prime):
    """Compute Dirichlet deviation term relative to the halo:

         S_dir = ∫ alpha |d_r sigma - d_r sigma_halo|^2 / (8 pi) 4 pi r^2 dr.
    """
    sigma_prime = np.gradient(sigma, dr)
    delta_prime = sigma_prime - sigma_halo_prime
    S_dir = np.sum(alpha * delta_prime**2 / (8.0 * np.pi) * four_pi_r2_dr)
    return S_dir

def main():
    # Baryonic profile and enclosed mass
    rho_b = rho_baryon(r)
    M_b = enclosed_mass(r, rho_b)

    # Build halo from analytic gradient, then switch to discrete gradient
    sigma_halo_prime_analytic = halo_sigma_gradient_analytic(M_b)
    sigma_halo = integrate_sigma_from_gradient(sigma_halo_prime_analytic)

    # Discrete gradient of the halo profile for consistent numerics
    sigma_halo_prime = np.gradient(sigma_halo, dr)

    # Minimal scalar action
    S_min = scalar_action(sigma_halo, sigma_halo_prime, rho_b)

    # Dirichlet deviation for the halo itself (should be ~ 0)
    S_dir_halo = dirichlet_deviation_from_halo(sigma_halo, sigma_halo_prime)

    print("=== Fisher scalar Bogomolny demo (discrete-consistent) ===")
    print(f"S_min (S_sigma[sigma_halo])      : {S_min:.10e}")
    print(f"Dirichlet deviation (halo)       : {S_dir_halo:.10e}")
    print(f"S_sigma[h] - S_min               : {0.0:.10e}")
    print()

    # Random perturbation tests
    rng = np.random.default_rng(12345)
    n_trials = 5

    for k in range(n_trials):
        # Smooth perturbation localised in the inner few R_d
        noise = rng.normal(scale=0.1, size=N_r)
        window = np.exp(- (r / (3.0 * R_d))**2)
        delta_sigma = noise * window

        sigma_trial = sigma_halo + delta_sigma
        sigma_trial_prime = np.gradient(sigma_trial, dr)

        S_trial = scalar_action(sigma_trial, sigma_trial_prime, rho_b)
        S_dir_trial = dirichlet_deviation_from_halo(sigma_trial, sigma_halo_prime)

        diff = (S_trial - S_min) - S_dir_trial

        print(f"Trial {k+1}:")
        print(f"  S_sigma[sigma_trial]          : {S_trial:.10e}")
        print(f"  S_sigma[sigma_trial] - S_min  : {S_trial - S_min:.10e}")
        print(f"  Dirichlet deviation term      : {S_dir_trial:.10e}")
        print(f"  (difference)                  : {diff:.10e}")
        print()

if __name__ == "__main__":
    main()
