#!/usr/bin/env python3
"""
26_grav_scalar_bogomolny_3d_fft_demo.py

3D Bogomolny demonstration for the Fisher scalar halo sector on a periodic box.

We consider the scalar action

  S_sigma[sigma] = ∫ [ alpha |∇sigma(x)|^2 / (8 pi) - g sigma(x) rho_b(x) ] d^3x

with constant alpha and an arbitrary 3D baryon density rho_b(x).

The Fisher halo profile sigma_halo(x) is defined as the unique periodic
solution of the elliptic equation

  ∇·(alpha ∇sigma_halo) = -4 pi g rho_b(x)

with zero mean, obtained here by a spectral (FFT-based) Poisson solver.

In the continuum, the scalar sector admits a Bogomolny-type identity

  S_sigma[sigma] = S_sigma[sigma_halo]
                   + (alpha / 8 pi) ∫ |∇sigma - ∇sigma_halo|^2 d^3x

for all sigma(x) with the same boundary conditions. This script demonstrates
this numerically on a 3D periodic cube for a non-spherical baryon profile.

Procedure:

  * define a 3D grid on a cube of side L with periodic boundary conditions,
  * build rho_b(x) as a sum of Gaussian clumps,
  * solve ∇^2 sigma_halo = -(4 pi g / alpha) rho_b via FFT,
  * compute the scalar action S_sigma[sigma_halo],
  * generate several random trial fields sigma_trial(x),
  * for each trial, compute:
        S_sigma[sigma_trial] - S_sigma[sigma_halo]
        (alpha / 8 pi) ∫ |∇sigma_trial - ∇sigma_halo|^2 d^3x,
    and print the difference.

Referees get a direct 3D check that the Fisher scalar sector is genuinely
Bogomolny beyond spherical symmetry.
"""

import numpy as np

# ----------------------------------------------------------------------
# Model parameters (code units)
# ----------------------------------------------------------------------

alpha = 1.0          # constant Fisher stiffness
g_coupling = 1.0     # scalar-baryon coupling
L = 10.0             # box size (side length in code units)

# Grid resolution (tune as needed)
N = 128               # N^3 grid; increase to 96 or 128 for heavier runs

# Number of random Bogomolny tests
N_TRIALS = 8

# ----------------------------------------------------------------------
# 3D grid and Fourier wavenumbers
# ----------------------------------------------------------------------

# Real-space grid
x = np.linspace(0.0, L, N, endpoint=False)
y = np.linspace(0.0, L, N, endpoint=False)
z = np.linspace(0.0, L, N, endpoint=False)
dx = x[1] - x[0]
volume = L**3

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# Fourier-space wavenumbers (periodic)
kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
kz = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
K2 = KX**2 + KY**2 + KZ**2

# Mask to avoid division by zero at k = 0
nonzero_mask = (K2 != 0.0)

# ----------------------------------------------------------------------
# Baryon density: sum of Gaussian clumps
# ----------------------------------------------------------------------

def gaussian_clump(X, Y, Z, x0, y0, z0, sigma):
    """3D Gaussian clump centred at (x0, y0, z0) with width sigma."""
    dx2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
    return np.exp(-dx2 / (2.0 * sigma**2))

def build_rho_b():
    """
    Build a non-spherical baryon density as a superposition of clumps.
    Normalise so that total baryonic mass M_b = ∫ rho_b d^3x is of order 1.
    """
    # Example: three clumps at different positions, different widths
    clumps = []
    clumps.append(gaussian_clump(X, Y, Z, 0.3 * L, 0.4 * L, 0.5 * L, sigma=0.4))
    clumps.append(gaussian_clump(X, Y, Z, 0.7 * L, 0.6 * L, 0.2 * L, sigma=0.6))
    clumps.append(gaussian_clump(X, Y, Z, 0.5 * L, 0.2 * L, 0.8 * L, sigma=0.5))

    rho_raw = clumps[0]
    for c in clumps[1:]:
        rho_raw = rho_raw + c

    # Normalise to a desired total mass M_b (here arbitrary, say 1.0)
    M_raw = rho_raw.mean() * volume
    M_target = 1.0
    rho_b = rho_raw * (M_target / M_raw)
    return rho_b

# ----------------------------------------------------------------------
# Scalar action and gradients
# ----------------------------------------------------------------------

def grad_sigma_fft(sigma):
    """
    Compute ∇sigma via spectral differentiation:

      grad sigma = i k sigma_k (in Fourier space).

    Returns (grad_x, grad_y, grad_z) on the real-space grid.
    """
    sigma_k = np.fft.fftn(sigma)
    grad_x_k = 1j * KX * sigma_k
    grad_y_k = 1j * KY * sigma_k
    grad_z_k = 1j * KZ * sigma_k

    grad_x = np.fft.ifftn(grad_x_k).real
    grad_y = np.fft.ifftn(grad_y_k).real
    grad_z = np.fft.ifftn(grad_z_k).real
    return grad_x, grad_y, grad_z

def scalar_action(sigma, rho_b):
    """
    Compute S_sigma[sigma] discretely:

      S_sigma = ∫ [ alpha |∇sigma|^2 / (8 pi) - g sigma rho_b ] d^3x

    using spectral gradients and simple Riemann sum.
    """
    grad_x, grad_y, grad_z = grad_sigma_fft(sigma)
    grad_sq = grad_x**2 + grad_y**2 + grad_z**2

    grad_term = alpha * grad_sq / (8.0 * np.pi)
    source_term = -g_coupling * sigma * rho_b

    return (grad_term + source_term).mean() * volume

def dirichlet_deviation_from_halo(sigma, sigma_halo):
    """
    Compute the Fisher-Dirichlet deviation:

      S_dir = (alpha / 8 pi) ∫ |∇sigma - ∇sigma_halo|^2 d^3x.
    """
    gx, gy, gz = grad_sigma_fft(sigma)
    ghx, ghy, ghz = grad_sigma_fft(sigma_halo)

    dx = gx - ghx
    dy = gy - ghy
    dz = gz - ghz
    delta_grad_sq = dx**2 + dy**2 + dz**2

    S_dir = (alpha / (8.0 * np.pi)) * delta_grad_sq.mean() * volume
    return S_dir

# ----------------------------------------------------------------------
# Solve for sigma_halo via FFT Poisson solver
# ----------------------------------------------------------------------

def solve_sigma_halo(rho_b):
    """
    Solve alpha ∇^2 sigma_halo = -4 pi g rho_b(x) on a periodic box.

    In Fourier space:

      -alpha K2 sigma_k = -4 pi g rho_k
      => sigma_k = (4 pi g / alpha) rho_k / K2, for K2 != 0
         sigma_k(0) = 0 (fix zero mode)

    Returns sigma_halo with zero mean.
    """
    rho_k = np.fft.fftn(rho_b)
    sigma_k = np.zeros_like(rho_k, dtype=complex)

    factor = (4.0 * np.pi * g_coupling) / alpha
    sigma_k[nonzero_mask] = factor * rho_k[nonzero_mask] / K2[nonzero_mask]
    sigma_k[~nonzero_mask] = 0.0  # zero mode fixed to 0

    sigma_halo = np.fft.ifftn(sigma_k).real

    # Ensure numerical zero mean
    sigma_halo = sigma_halo - sigma_halo.mean()
    return sigma_halo

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    print("=== 3D Fisher scalar Bogomolny demo (FFT, periodic box) ===")
    print(f"Grid: N = {N} (N^3 = {N**3}), L = {L:.2f}, dx = {dx:.3e}")
    print()

    # Build baryon density and solve for halo
    rho_b = build_rho_b()
    M_b = rho_b.mean() * volume

    print(f"Total baryonic mass M_b         : {M_b:.6e}")
    print("Solving alpha ∇^2 sigma_halo = -4 pi g rho_b ...")

    sigma_halo = solve_sigma_halo(rho_b)
    S_min = scalar_action(sigma_halo, rho_b)

    print(f"S_sigma[sigma_halo] (S_min)     : {S_min:.10e}")
    print()

    # Bogomolny tests with random trial fields
    rng = np.random.default_rng(12345)

    for k in range(N_TRIALS):
        # Random sigma_trial: smooth noise built from low-k modes
        # to avoid aliasing and to keep it reasonably regular
        noise = rng.normal(scale=1.0, size=(N, N, N))
        noise_k = np.fft.fftn(noise)

        # Apply a low-pass filter in k-space
        k_cut = N // 4
        mask_lp = (np.abs(KX * L / (2.0 * np.pi)) < k_cut) & \
                  (np.abs(KY * L / (2.0 * np.pi)) < k_cut) & \
                  (np.abs(KZ * L / (2.0 * np.pi)) < k_cut)
        noise_k[~mask_lp] = 0.0
        sigma_trial = np.fft.ifftn(noise_k).real

        # Remove mean to keep same gauge as sigma_halo
        sigma_trial = sigma_trial - sigma_trial.mean()

        # Compute action and Dirichlet deviation
        S_trial = scalar_action(sigma_trial, rho_b)
        S_gap = S_trial - S_min
        S_dir = dirichlet_deviation_from_halo(sigma_trial, sigma_halo)
        diff = S_gap - S_dir

        print(f"Trial {k+1}:")
        print(f"  S_sigma[sigma_trial]          : {S_trial:.10e}")
        print(f"  S_sigma[sigma_trial] - S_min  : {S_gap:.10e}")
        print(f"  Dirichlet deviation term      : {S_dir:.10e}")
        print(f"  (difference)                  : {diff:.10e}")
        print()

if __name__ == "__main__":
    main()
