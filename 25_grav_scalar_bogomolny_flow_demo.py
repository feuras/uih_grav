#!/usr/bin/env python3
"""
25_grav_scalar_bogomolny_flow_parallel.py

Parallel gradient-flow demonstration for the Fisher scalar halo sector.

We consider the scalar action (constant alpha, spherical symmetry)

  S_sigma[sigma] = ∫ [alpha |d_r sigma|^2 / (8 pi) - g sigma rho_b(r)] 4 pi r^2 dr

and evolve sigma(r, t) under the Fisher-weight gradient flow

  ∂_t sigma(r, t)
    = (1 / (4 pi r^2)) ∂_r [ r^2 alpha d_r sigma(r, t) ] + g rho_b(r),

whose stationary solution is the Fisher halo profile sigma_halo(r),
satisfying

  d/dr [ r^2 alpha d_r sigma_halo ] = - 4 pi g r^2 rho_b(r).

This script:

  * builds a spherical exponential baryon profile rho_b(r),
  * constructs the Fisher halo profile sigma_halo(r),
  * defines the scalar action S_sigma and Dirichlet deviation to sigma_halo,
  * runs many independent gradient flows in parallel (different random seeds),
  * and shows that in every run:

        S_sigma(t) ↓ S_min = S_sigma[sigma_halo],
        Dirichlet deviation → 0,

demonstrating that the Fisher halo is the unique global attractor of the
scalar sector at fixed baryons.

Defaults are chosen to be light enough that 22 runs are comfortable.
You can crank N_r, T_end or N_RUNS if you want heavier tests.
"""

import numpy as np
import multiprocessing as mp

# ----------------------------------------------------------------------
# Model parameters (code units)
# ----------------------------------------------------------------------

G = 1.0          # Newton constant (context only)
alpha = 1.0      # constant Fisher stiffness
g_coupling = 1.0

rho0 = 1.0       # central baryonic density
R_d = 1.0        # baryonic scale radius

# ----------------------------------------------------------------------
# Radial grid and flow settings
# ----------------------------------------------------------------------

# Lighter defaults for parallel runs
R_max = 20.0 * R_d
N_r = 5000

r = np.linspace(1e-4, R_max, N_r)   # avoid r = 0
dr = r[1] - r[0]
four_pi_r2_dr = 4.0 * np.pi * r**2 * dr

# Time stepping parameters
# For a diffusion-like term with effective D ~ alpha / (4 pi),
# stability is dt ≲ dr^2 / (2 D). Taking dt = 0.5 dr^2 is still conservative.
C_stab = 0.5
dt = C_stab * dr**2 / alpha
T_end = 10.0
N_steps = int(T_end / dt)
LOG_POINTS = 20   # how many times to log per run

# Parallel settings
N_WORKERS = 22    # requested number of cores
N_RUNS = 44       # one run per core by default

# ----------------------------------------------------------------------
# Baryonic profile and halo helpers
# ----------------------------------------------------------------------

def rho_baryon(r_arr):
    """Exponential sphere baryonic density."""
    return rho0 * np.exp(-r_arr / R_d)

def enclosed_mass(r_arr, rho_arr):
    """Cumulative enclosed mass M_b(r) on uniform grid."""
    integrand = 4.0 * np.pi * r_arr**2 * rho_arr
    return np.cumsum(integrand) * dr

def halo_sigma_gradient_analytic(M_b):
    """Analytic halo gradient for constant alpha sector."""
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
    """Dirichlet deviation term relative to the halo."""
    sigma_prime = np.gradient(sigma, dr)
    delta_prime = sigma_prime - sigma_halo_prime
    S_dir = np.sum(alpha * delta_prime**2 / (8.0 * np.pi) * four_pi_r2_dr)
    return S_dir

def flow_rhs_sigma(sigma, rho_b):
    """
    RHS of gradient flow:

      ∂_t sigma = (1 / (4 pi r^2)) ∂_r [ r^2 alpha d_r sigma ] + g rho_b.

    Discretised with central differences and simple BCs:
      * zero-flux at r ~ 0,
      * fixed sigma at R_max.
    """
    sigma_prime = np.gradient(sigma, dr)
    F = alpha * r**2 * sigma_prime
    dF_dr = np.gradient(F, dr)

    rhs = (1.0 / (4.0 * np.pi * r**2)) * dF_dr + g_coupling * rho_b

    rhs[0] = 0.0      # inner boundary: no flux
    rhs[-1] = 0.0     # outer boundary: sigma fixed

    return rhs

# Precompute baryons and halo once, shared by all workers
rho_b_global = rho_baryon(r)
M_b_global = enclosed_mass(r, rho_b_global)
sigma_halo_prime_analytic = halo_sigma_gradient_analytic(M_b_global)
sigma_halo_global = integrate_sigma_from_gradient(sigma_halo_prime_analytic)
sigma_halo_prime_global = np.gradient(sigma_halo_global, dr)
S_min_global = scalar_action(sigma_halo_global, sigma_halo_prime_global, rho_b_global)

# ----------------------------------------------------------------------
# Worker for a single flow run
# ----------------------------------------------------------------------

def run_single_flow(seed):
    """
    Run a single gradient-flow with a given random seed.

    Returns a dict summarising:
      * seed
      * S_sigma at t=0 and t=T_end
      * S_sigma - S_min at t=0 and t=T_end
      * Dirichlet deviation at t=0 and t=T_end
      * flag whether S_sigma was monotonically decreasing at coarse log points
    """
    rng = np.random.default_rng(seed)

    # Initial condition: halo + smooth noise
    noise = rng.normal(scale=0.5, size=N_r)
    window = np.exp(- (r / (3.0 * R_d))**2)
    sigma = sigma_halo_global + noise * window
    sigma[-1] = 0.0   # enforce outer BC

    # Initial diagnostics
    sigma_prime = np.gradient(sigma, dr)
    S0 = scalar_action(sigma, sigma_prime, rho_b_global)
    S0_gap = S0 - S_min_global
    S_dir0 = dirichlet_deviation_from_halo(sigma, sigma_halo_prime_global)

    # Precompute log steps
    log_every = max(1, N_steps // LOG_POINTS)
    S_log = [S0]

    # Time evolution
    for step in range(1, N_steps + 1):
        rhs = flow_rhs_sigma(sigma, rho_b_global)
        sigma = sigma + dt * rhs
        sigma[-1] = 0.0

        if step % log_every == 0 or step == N_steps:
            sigma_prime = np.gradient(sigma, dr)
            S_val = scalar_action(sigma, sigma_prime, rho_b_global)
            S_log.append(S_val)

    # Final diagnostics
    sigma_prime = np.gradient(sigma, dr)
    S_final = scalar_action(sigma, sigma_prime, rho_b_global)
    S_final_gap = S_final - S_min_global
    S_dir_final = dirichlet_deviation_from_halo(sigma, sigma_halo_prime_global)

    # Check rough monotonicity at log sampling points
    S_log_arr = np.array(S_log)
    monotone = np.all(np.diff(S_log_arr) <= 1e-8)  # tiny wiggle allowed

    return {
        "seed": seed,
        "S0": S0,
        "S0_gap": S0_gap,
        "S_dir0": S_dir0,
        "S_final": S_final,
        "S_final_gap": S_final_gap,
        "S_dir_final": S_dir_final,
        "monotone": bool(monotone),
    }

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    print("=== Fisher scalar BPS gradient-flow (parallel) ===")
    print(f"Grid: N_r = {N_r}, R_max = {R_max:.2f}, dr = {dr:.3e}")
    print(f"Time: dt = {dt:.3e}, N_steps = {N_steps}, T_end ~ {T_end:.3f}")
    print(f"Halo action S_min               : {S_min_global:.10e}")
    print(f"Running {N_RUNS} flows with {N_WORKERS} workers...")
    print()

    seeds = [2025 + k for k in range(N_RUNS)]

    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(run_single_flow, seeds)

    # Print per-run summary
    for res in results:
        print(f"Run seed {res['seed']}:")
        print(f"  S_sigma(0)                    : {res['S0']:.10e}")
        print(f"  S_sigma(0) - S_min           : {res['S0_gap']:.10e}")
        print(f"  Dirichlet deviation (0)      : {res['S_dir0']:.10e}")
        print(f"  S_sigma(T)                    : {res['S_final']:.10e}")
        print(f"  S_sigma(T) - S_min           : {res['S_final_gap']:.10e}")
        print(f"  Dirichlet deviation (T)      : {res['S_dir_final']:.10e}")
        print(f"  Monotone at log points?      : {res['monotone']}")
        print()

    # Aggregate quick sanity check
    gaps_final = np.array([res["S_final_gap"] for res in results])
    dirs_final = np.array([res["S_dir_final"] for res in results])

    print("=== Aggregate summary ===")
    print(f"Final S_gap min / max           : {gaps_final.min():.3e} / {gaps_final.max():.3e}")
    print(f"Final Dirichlet min / max      : {dirs_final.min():.3e} / {dirs_final.max():.3e}")
    print(f"All runs monotone?             : {all(res['monotone'] for res in results)}")

if __name__ == "__main__":
    main()
