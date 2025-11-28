#!/usr/bin/env python3
"""
27_grav_scalar_bogomolny_running_alpha_and_Ksplit_demo.py

Fisher scalar Bogomolny test with running stiffness alpha(r),
together with a finite-dimensional K = G + iJ consistency check.

Part A: 1D spherical Fisher scalar with radial grid and non-constant alpha(r).
        We construct the exact halo solution sigma_halo(r) for a fixed baryon
        profile rho_b(r), using the general Euler Lagrange equation

            1/r^2 d/dr [ r^2 alpha(r) d sigma/dr ] = -4 pi g rho_b(r),

        and then verify numerically the Bogomolny identity

            S_sigma[sigma] - S_sigma[sigma_halo]
                = (1/8 pi) ∫ alpha(r) |∇(sigma - sigma_halo)|^2 d^3x

        for a collection of random trial fields sigma_trial(r) that respect
        the same boundary conditions. This generalises the earlier constant
        alpha tests and shows that the BPS structure is robust under smooth
        running Fisher stiffness.

Part B: Finite-dimensional K-splitting toy. We build an arbitrary real symmetric
        positive matrix G (playing the role of a Dirichlet form generator) and
        a random real skew-symmetric matrix J, form K = G + i J, and verify
        numerically that the symmetric part of K in the standard inner product
        recovers exactly G:

            G_from_K = (K + K^†)/2  =  G.

        This illustrates at the algebraic level that adding a metric skew J
        sector to K does not change the static Fisher energy functional, which
        depends only on the symmetric part G. In the continuum Dunkley Fisher
        gravity setting, this is the reason why the Bogomolny structure of the
        scalar halo sector is insensitive to the detailed J sector dynamics.
"""

import numpy as np


# ----------------------------------------------------------------------
# Baryon profile and running alpha
# ----------------------------------------------------------------------

def baryon_profile(r, R_d=1.0):
    """
    Exponential spherical baryon profile with unit total mass:

        rho_b(r) = (1 / (8 pi R_d^3)) * exp(-r / R_d),

    so that 4 pi ∫_0^∞ r^2 rho_b(r) dr = 1.
    """
    rho0 = 1.0 / (8.0 * np.pi * R_d**3)
    return rho0 * np.exp(-r / R_d)


def alpha_running(r, R_d=1.0, alpha0=1.0, alpha_outer=5.0):
    """
    Example of a smooth running Fisher stiffness alpha(r).

    Inside the disc (r ≲ R_d) we keep alpha ≃ alpha0. Outside we let alpha
    grow towards alpha_outer with a simple saturating profile:

        alpha(r) = alpha0 * [ 1 + (alpha_outer/alpha0 - 1) * f(r/R_d) ],

    with f(x) = x^2 / (1 + x^2), so that f(0)=0 and f(x)→1 as x→∞.
    """
    x = r / R_d
    f = x * x / (1.0 + x * x)
    return alpha0 * (1.0 + (alpha_outer / alpha0 - 1.0) * f)


# ----------------------------------------------------------------------
# Radial grid and mass
# ----------------------------------------------------------------------

def build_radial_grid(R_max, N_r):
    """
    Uniform radial grid on (0, R_max) using cell midpoints.

    r_j = (j + 0.5) * dr,  j = 0,...,N_r-1,  dr = R_max / N_r.
    """
    dr = R_max / float(N_r)
    j = np.arange(N_r, dtype=float)
    r = (j + 0.5) * dr
    return r, dr


def cumulative_mass(r, rho, dr):
    """
    Compute cumulative baryonic mass M_b(r_j) = 4 pi ∫_0^{r_j} s^2 rho(s) ds
    by simple Riemann sum on the midpoint grid.
    """
    shell_vol = 4.0 * np.pi * r**2 * dr
    dM = shell_vol * rho
    M = np.cumsum(dM)
    return M


# ----------------------------------------------------------------------
# Halo construction and functionals
# ----------------------------------------------------------------------

def construct_sigma_halo(r, dr, rho_b, alpha_r, g=1.0):
    """
    Construct the exact halo solution sigma_halo(r) for given rho_b(r)
    and running alpha(r), assuming spherical symmetry and a boundary
    condition sigma(R_max) = 0.

    From the Euler Lagrange equation:

        d/dr [ r^2 alpha(r) d sigma/dr ] = -4 pi g r^2 rho_b(r),

    we can write in terms of the enclosed mass M_b(r) = 4 pi ∫_0^r s^2 rho_b(s) ds:

        d sigma/dr = - g M_b(r) / [ r^2 alpha(r) ].

    We evaluate d sigma/dr on the midpoint grid using M_b(r) from cumulative_mass,
    and then integrate inward from R_max to 0 with the boundary condition
    sigma(R_max) = 0.
    """
    # cumulative baryonic mass
    M_b = cumulative_mass(r, rho_b, dr)  # shape (N_r,)

    # avoid division by zero at r~0: M_b(r) ~ O(r^3), so the ratio is regular.
    # To be safe, set the first point by copying the second one.
    denom = r**2 * alpha_r
    dsigma_dr = -g * M_b / denom
    if dsigma_dr.size > 1:
        dsigma_dr[0] = dsigma_dr[1]

    # integrate inward from R_max with sigma(R_max) = 0
    N_r = r.size
    sigma = np.zeros_like(r)
    sigma[-1] = 0.0
    for j in range(N_r - 2, -1, -1):
        sigma[j] = sigma[j + 1] - dsigma_dr[j] * dr

    return sigma, dsigma_dr


def finite_difference_gradient(f, dr):
    """
    Simple centred finite difference for df/dr on the midpoint grid, with
    Neumann type extrapolation at the boundaries.
    """
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * dr)
    # one sided at boundaries
    df[0] = (f[1] - f[0]) / dr
    df[-1] = (f[-1] - f[-2]) / dr
    return df


def fisher_scalar_action(r, dr, sigma, rho_b, alpha_r, g=1.0):
    """
    Discretised Fisher scalar action

        S_sigma[sigma]
            = ∫ [ alpha(r)/(8 pi) |∇sigma|^2  -  g sigma rho_b ] d^3x
            = 4 pi ∫_0^{R_max} [ alpha(r)/(8 pi) (d sigma/dr)^2
                                 - g sigma(r) rho_b(r) ] r^2 dr.

    We use a simple midpoint Riemann sum on the radial grid.
    """
    dsigma_dr = finite_difference_gradient(sigma, dr)
    grad_term = alpha_r * dsigma_dr**2 / (8.0 * np.pi)
    pot_term = -g * sigma * rho_b
    integrand = grad_term + pot_term
    shell_vol = 4.0 * np.pi * r**2 * dr
    return np.sum(integrand * shell_vol)


def dirichlet_deviation(r, dr, sigma, sigma_ref, alpha_r):
    """
    Weighted Dirichlet deviation

        D[sigma | sigma_ref]
            = (1 / 8 pi) ∫ alpha(r) |∇(sigma - sigma_ref)|^2 d^3x.

    Again evaluated by midpoint Riemann sum.
    """
    delta = sigma - sigma_ref
    ddelta_dr = finite_difference_gradient(delta, dr)
    integrand = alpha_r * ddelta_dr**2 / (8.0 * np.pi)
    shell_vol = 4.0 * np.pi * r**2 * dr
    return np.sum(integrand * shell_vol)


# ----------------------------------------------------------------------
# Part A: running alpha Bogomolny test
# ----------------------------------------------------------------------

def part_A_running_alpha_bps_demo():
    """
    Part A: verify Fisher scalar Bogomolny identity with running alpha(r).
    """
    # Radial grid and parameters
    R_d = 1.0
    R_max = 200.0 * R_d
    N_r = 250000  # high resolution for good accuracy; adjust if needed

    g = 1.0

    r, dr = build_radial_grid(R_max, N_r)
    rho_b = baryon_profile(r, R_d=R_d)
    alpha_r = alpha_running(r, R_d=R_d, alpha0=1.0, alpha_outer=5.0)

    # Construct halo solution
    sigma_halo, dsigma_halo_dr = construct_sigma_halo(r, dr, rho_b, alpha_r, g=g)

    # Action at the halo (should be minimal)
    S_min = fisher_scalar_action(r, dr, sigma_halo, rho_b, alpha_r, g=g)

    # Dirichlet deviation of halo relative to itself (should be ~0)
    D_halo = dirichlet_deviation(r, dr, sigma_halo, sigma_halo, alpha_r)

    print("=== Fisher scalar Bogomolny demo with running alpha(r) ===")
    print(f"R_max = {R_max:.2f}, N_r = {N_r}, dr = {dr:.6e}")
    print(f"S_min (S_sigma[sigma_halo])      : {S_min: .10e}")
    print(f"Dirichlet deviation (halo)       : {D_halo: .10e}")
    print()

    rng = np.random.default_rng(seed=2025)
    n_trials = 5

    for k in range(1, n_trials + 1):
        # Random perturbation that vanishes at r=0 and r=R_max
        noise = rng.standard_normal(N_r)

        # Smooth polynomial bump: zero at boundaries, O(1) in the interior
        window = (r / R_max) * (1.0 - r / R_max)

        delta_sigma = 0.1 * window * noise  # small perturbation

        sigma_trial = sigma_halo + delta_sigma

        S_trial = fisher_scalar_action(r, dr, sigma_trial, rho_b, alpha_r, g=g)
        D_trial = dirichlet_deviation(r, dr, sigma_trial, sigma_halo, alpha_r)

        gap = S_trial - S_min
        diff = gap - D_trial

        print(f"Trial {k}:")
        print(f"  S_sigma[sigma_trial]          : {S_trial: .10e}")
        print(f"  S_sigma[sigma_trial] - S_min  : {gap: .10e}")
        print(f"  Dirichlet deviation term      : {D_trial: .10e}")
        print(f"  (difference)                  : {diff: .10e}")
        print()

    print("Part A complete.\n")


# ----------------------------------------------------------------------
# Part B: finite-dimensional K = G + iJ split test
# ----------------------------------------------------------------------

def part_B_K_split_demo():
    """
    Part B: finite-dimensional K = G + iJ consistency check.

    We generate a random symmetric positive-definite matrix G and a random
    real skew-symmetric matrix J, form K = G + iJ, and verify numerically that

        (K + K^†) / 2  =  G

    to machine precision. We also check that the associated quadratic form
    v^T G v is identical whether we compute it from G directly or from the
    symmetric part of K.
    """
    rng = np.random.default_rng(seed=314159)

    N = 128  # modest dimension; large enough to be non-trivial
    A = rng.standard_normal((N, N))
    G = A.T @ A  # symmetric positive-definite

    R = rng.standard_normal((N, N))
    J = 0.5 * (R - R.T)  # real skew-symmetric

    K = G + 1j * J

    # Symmetric (Hermitian) part of K
    G_from_K = 0.5 * (K + K.conj().T)

    # Diagnostics
    imag_norm = np.linalg.norm(G_from_K.imag)
    real_diff_norm = np.linalg.norm(G_from_K.real - G)

    # Quadratic-form check on a random real vector
    v = rng.standard_normal(N)
    q_direct = float(v.T @ (G @ v))
    q_fromK = float(v.T @ (G_from_K.real @ v))

    print("=== Finite-dimensional K = G + iJ split demo ===")
    print(f"Matrix dimension N               : {N}")
    print(f"||Im(G_from_K)||_F               : {imag_norm: .3e}")
    print(f"||Re(G_from_K) - G||_F           : {real_diff_norm: .3e}")
    print(f"Quadratic form v^T G v (direct)  : {q_direct: .10e}")
    print(f"Quadratic form v^T G v (from K)  : {q_fromK: .10e}")
    print(f"Relative difference              : {(q_fromK - q_direct)/q_direct: .3e}")
    print("Part B complete.\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    part_A_running_alpha_bps_demo()
    part_B_K_split_demo()


if __name__ == "__main__":
    main()
