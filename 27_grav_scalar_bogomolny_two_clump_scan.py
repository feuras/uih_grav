#!/usr/bin/env python3
"""
27_grav_scalar_bogomolny_two_clump_scan.py

Two-clump Fisher–Bogomolny interaction scan in 3D (periodic box, FFT solver).

Scalar action:

  S_sigma[sigma; rho_b] = ∫ [ alpha |∇sigma(x)|^2 / (8 pi) - g sigma(x) rho_b(x) ] d^3x

with constant alpha and 3D baryon density rho_b(x).

Setup:

  1) Build a single Gaussian clump at the box centre, normalised to mass M_target.
     Solve alpha ∇^2 sigma_single = - 4 pi g rho_single, fix k = 0 mode to zero,
     and compute

        S_single = S_sigma[sigma_single; rho_single].

  2) For a range of separations d, build a two-clump baryon density

        rho_two(x; d) = rho_single(x - x1(d)) + rho_single(x - x2(d)),

     where the shifts are implemented by exact spectral translations on the
     periodic cube. Each clump carries mass M_target, so total mass is
     exactly 2 M_target independent of d.

  3) For each d, solve for the halo sigma_two(d) of rho_two(d) and compute

        S_two(d) = S_sigma[sigma_two(d); rho_two(d)],
        S_int(d) = S_two(d) - 2 S_single.

     This is the Fisher interaction energy between two BPS halos at separation d.

  4) Construct the naive superposition halo

        sigma_sup(d, x) = sigma_single(x - x1) + sigma_single(x - x2),

     again via spectral translations, and compute the Fisher–Dirichlet deviation

        S_dir_sup(d) = (alpha / 8 pi) ∫ |∇sigma_two(d) - ∇sigma_sup(d)|^2 d^3x.

     For a strict linear Poisson sector we expect sigma_two ≈ sigma_sup and
     S_dir_sup(d) ≈ 0 up to numerical error; S_int(d) then measures the
     usual quadratic interaction term in the action functional.

All separations are processed in parallel with up to N_WORKERS cores.
"""

import numpy as np
import multiprocessing as mp

# ----------------------------------------------------------------------
# Model parameters
# ----------------------------------------------------------------------

alpha = 1.0          # Fisher stiffness
g_coupling = 1.0     # scalar–baryon coupling
L = 10.0             # box size (side length)
N = 128               # grid resolution (N^3)

# Gaussian clump parameters
sigma_clump = 0.6    # width of each baryon lump
M_target = 0.5       # mass of a single clump; total mass for two clumps ~ 1

# Parallel settings
N_WORKERS = 22

# Separation scan (code units)
d_min = 1.0
d_max = 6.0
N_d = 16
d_values = np.linspace(d_min, d_max, N_d)

# ----------------------------------------------------------------------
# 3D grid and Fourier wavenumbers
# ----------------------------------------------------------------------

x = np.linspace(0.0, L, N, endpoint=False)
y = np.linspace(0.0, L, N, endpoint=False)
z = np.linspace(0.0, L, N, endpoint=False)
dx = x[1] - x[0]
volume = L**3

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
kz = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
K2 = KX**2 + KY**2 + KZ**2
nonzero_mask = (K2 != 0.0)

# ----------------------------------------------------------------------
# Basic building blocks
# ----------------------------------------------------------------------

def gaussian_clump(X, Y, Z, x0, y0, z0, sigma):
    """Gaussian baryon clump centred at (x0, y0, z0) with width sigma."""
    dx2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
    return np.exp(-dx2 / (2.0 * sigma**2))

def normalise_to_mass(rho_raw, M_desired):
    """Rescale rho_raw so that ∫ rho d^3x = M_desired."""
    M_raw = rho_raw.mean() * volume
    return rho_raw * (M_desired / M_raw)

def spectral_shift(field, dx_shift, dy_shift=0.0, dz_shift=0.0):
    """
    Periodic translation of a real field by (dx_shift, dy_shift, dz_shift)
    using spectral (FFT) phase factors:

      f(x + delta)  <->  f_k exp(i k · delta).

    We implement a shift of +delta in real space by multiplying by exp(i k · delta).
    """
    field_k = np.fft.fftn(field)
    phase = np.exp(1j * (KX * dx_shift + KY * dy_shift + KZ * dz_shift))
    shifted_k = field_k * phase
    shifted = np.fft.ifftn(shifted_k).real
    return shifted

def grad_sigma_fft(sigma):
    """Compute ∇sigma via spectral differentiation."""
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
    Compute S_sigma[sigma; rho_b] = ∫ [ alpha |∇sigma|^2 / (8 pi) - g sigma rho_b ] d^3x.
    """
    grad_x, grad_y, grad_z = grad_sigma_fft(sigma)
    grad_sq = grad_x**2 + grad_y**2 + grad_z**2

    grad_term = alpha * grad_sq / (8.0 * np.pi)
    source_term = -g_coupling * sigma * rho_b

    return (grad_term + source_term).mean() * volume

def dirichlet_between(sigma_a, sigma_b):
    """
    Fisher–Dirichlet deviation between two profiles:

      S_dir = (alpha / 8 pi) ∫ |∇sigma_a - ∇sigma_b|^2 d^3x.
    """
    ga_x, ga_y, ga_z = grad_sigma_fft(sigma_a)
    gb_x, gb_y, gb_z = grad_sigma_fft(sigma_b)

    dx_ = ga_x - gb_x
    dy_ = ga_y - gb_y
    dz_ = ga_z - gb_z

    delta_grad_sq = dx_**2 + dy_**2 + dz_**2
    S_dir = (alpha / (8.0 * np.pi)) * delta_grad_sq.mean() * volume
    return S_dir

def solve_sigma_halo(rho_b):
    """
    Solve alpha ∇^2 sigma_halo = -4 pi g rho_b in Fourier space:

      -alpha K^2 sigma_k = -4 pi g rho_k
      => sigma_k = (4 pi g / alpha) rho_k / K^2, for K^2 != 0,
         sigma_k(0) = 0.
    """
    rho_k = np.fft.fftn(rho_b)
    sigma_k = np.zeros_like(rho_k, dtype=complex)

    factor = (4.0 * np.pi * g_coupling) / alpha
    sigma_k[nonzero_mask] = factor * rho_k[nonzero_mask] / K2[nonzero_mask]
    sigma_k[~nonzero_mask] = 0.0

    sigma = np.fft.ifftn(sigma_k).real
    sigma = sigma - sigma.mean()
    return sigma

# ----------------------------------------------------------------------
# Precompute single-clump reference
# ----------------------------------------------------------------------

def build_single_clump():
    """Build a single Gaussian clump at the box centre and normalise to M_target."""
    x0 = 0.5 * L
    y0 = 0.5 * L
    z0 = 0.5 * L
    rho_raw = gaussian_clump(X, Y, Z, x0, y0, z0, sigma_clump)
    rho_single = normalise_to_mass(rho_raw, M_target)
    return rho_single

rho_single_global = build_single_clump()
sigma_single_global = solve_sigma_halo(rho_single_global)
S_single_global = scalar_action(sigma_single_global, rho_single_global)

# Precompute Fourier images for translation
rho_single_k_global = np.fft.fftn(rho_single_global)
sigma_single_k_global = np.fft.fftn(sigma_single_global)

# ----------------------------------------------------------------------
# Worker for a single separation d
# ----------------------------------------------------------------------

def process_separation(d):
    """
    For a given separation d, build a two-clump baryon density by spectral
    translations of rho_single, solve for the Fisher halo, compute S_two(d),
    S_int(d), and a Dirichlet distance to the superposition of translated
    single-clump halos.
    """
    # Positions of the two clumps along x-axis
    x1 = 0.5 * L - 0.5 * d
    x2 = 0.5 * L + 0.5 * d
    y0 = 0.5 * L
    z0 = 0.5 * L

    # Translation vectors (only x shifts are nonzero)
    dx1 = x1 - 0.5 * L
    dx2 = x2 - 0.5 * L

    # Spectral translations for rho
    phase1 = np.exp(1j * (KX * dx1))
    phase2 = np.exp(1j * (KX * dx2))

    rho1_k = rho_single_k_global * phase1
    rho2_k = rho_single_k_global * phase2

    rho1 = np.fft.ifftn(rho1_k).real
    rho2 = np.fft.ifftn(rho2_k).real
    rho_two = rho1 + rho2

    # Check total mass (should be ~ 2 M_target)
    M_two = rho_two.mean() * volume

    # Solve for two-clump halo
    sigma_two = solve_sigma_halo(rho_two)
    S_two = scalar_action(sigma_two, rho_two)

    # Interaction energy relative to two isolated clumps
    S_int = S_two - 2.0 * S_single_global

    # Superposition halo built from translated single-clump halos
    sigma1_k = sigma_single_k_global * phase1
    sigma2_k = sigma_single_k_global * phase2

    sigma1 = np.fft.ifftn(sigma1_k).real
    sigma2 = np.fft.ifftn(sigma2_k).real
    sigma_sup = sigma1 + sigma2
    sigma_sup = sigma_sup - sigma_sup.mean()

    # Dirichlet deviation between true two-clump halo and superposed single halos
    S_dir_sup = dirichlet_between(sigma_two, sigma_sup)

    return {
        "d": float(d),
        "M_two": float(M_two),
        "S_two": float(S_two),
        "S_int": float(S_int),
        "S_dir_sup": float(S_dir_sup),
    }

# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    print("=== Two-clump Fisher Bogomolny scan (3D FFT, periodic) ===")
    print(f"Grid: N = {N}, N^3 = {N**3}, L = {L:.2f}, dx = {dx:.3e}")
    print(f"Sigma_clump = {sigma_clump:.3f}, single-clump mass M_target = {M_target:.3f}")
    print()
    print(f"S_single (one clump)            : {S_single_global:.10e}")
    print(f"Separation range d              : [{d_min:.2f}, {d_max:.2f}] with {N_d} samples")
    print(f"Using up to {N_WORKERS} worker processes")
    print()

    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(process_separation, d_values)

    # Sort results by separation
    results_sorted = sorted(results, key=lambda r: r["d"])

    d_arr = np.array([r["d"] for r in results_sorted])
    M_two_arr = np.array([r["M_two"] for r in results_sorted])
    S_two_arr = np.array([r["S_two"] for r in results_sorted])
    S_int_arr = np.array([r["S_int"] for r in results_sorted])
    S_dir_sup_arr = np.array([r["S_dir_sup"] for r in results_sorted])

    print("Results (d, M_two, S_int, S_dir_sup):")
    for d, M_two, Sint, Sdir in zip(d_arr, M_two_arr, S_int_arr, S_dir_sup_arr):
        print(f"  d = {d:6.3f} :  M_two = {M_two:.6e}  S_int = {Sint:.6e}  S_dir_sup = {Sdir:.6e}")
    print()

    out_file = "grav_bps_two_clump_scan.npz"
    np.savez(
        out_file,
        d=d_arr,
        M_two=M_two_arr,
        S_two=S_two_arr,
        S_int=S_int_arr,
        S_dir_sup=S_dir_sup_arr,
        S_single=S_single_global,
        L=L,
        N=N,
        sigma_clump=sigma_clump,
        M_target=M_target,
        alpha=alpha,
        g_coupling=g_coupling,
    )

    print(f"Saved scan results to {out_file}")

if __name__ == "__main__":
    main()
