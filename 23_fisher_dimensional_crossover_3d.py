#!/usr/bin/env python3
"""
51_fisher_dimensional_crossover_3d.py

Dimensional crossover and three dimensional Fisher geometry.

This script numerically probes how the Fisher "curvature" of a constant-mode
collective coordinate scales with system size in 2D discs and 3D balls for a
Gaussian field with local precision operator

    G = m^2 I - Δ

with Dirichlet boundary conditions on a rectangular / cuboidal lattice.

For each radius R we:

  * Build a binary mask chi_R that is 1 inside a disc (2D) or ball (3D) of
    radius R centred in the domain and 0 outside.

  * Compute the discrete Fisher quadratic form

        I_sigma(R) = chi_R^T G chi_R

    which plays the role of the Fisher "curvature" in the constant-mode
    direction.

  * Scan I_sigma(R) as a function of R and estimate an effective scaling
    exponent d_eff from a least-squares fit of log I_sigma(R) vs log R.

The 2D results illustrate the area law and perimeter correction discussed in
the paper. The 3D results provide a benchmark for how the same local Fisher
geometry behaves in fully three dimensional settings and help to identify the
crossover between effectively two dimensional and three dimensional regimes.

Outputs:
    - results/fisher_dim_crossover_3d/fisher_dim_crossover_3d.npz
        Arrays:
            radii_2d, I_sigma_2d, radii_3d, I_sigma_3d
            fit_2d_slope, fit_2d_intercept
            fit_3d_slope, fit_3d_intercept

    - Optional PNG plots (if make_plots=True):
        results/fisher_dim_crossover_3d/fisher_dim_crossover_3d.png
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Lattice and operator construction
# ----------------------------------------------------------------------

def laplacian_1d(n: int, dx: float) -> sp.csr_matrix:
    """
    Construct the 1D Dirichlet Laplacian on an n-point grid with spacing dx.

    The operator acts on interior points with implicit Dirichlet boundaries
    at the ends; we represent it as an n x n matrix with the usual
    second-difference stencil.
    """
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    L = sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    L /= dx ** 2
    return L


def precision_2d(nx: int, ny: int, dx: float, dy: float, m2: float) -> sp.csr_matrix:
    """
    Build G_2D = m^2 I - Δ_2D on an (nx x ny) grid with spacings dx, dy.

    Dirichlet boundary conditions are implemented via the 1D Laplacians.
    """
    Lx = laplacian_1d(nx, dx)
    Ly = laplacian_1d(ny, dy)
    Ix = sp.eye(nx, format="csr")
    Iy = sp.eye(ny, format="csr")

    # 2D Laplacian via Kronecker sum
    L2 = sp.kron(Iy, Lx, format="csr") + sp.kron(Ly, Ix, format="csr")
    G2 = m2 * sp.eye(nx * ny, format="csr") - L2
    return G2


def precision_3d(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float, m2: float) -> sp.csr_matrix:
    """
    Build G_3D = m^2 I - Δ_3D on an (nx x ny x nz) grid with spacings dx,dy,dz.

    Dirichlet boundary conditions are again implemented via the 1D Laplacians.
    """
    Lx = laplacian_1d(nx, dx)
    Ly = laplacian_1d(ny, dy)
    Lz = laplacian_1d(nz, dz)
    Ix = sp.eye(nx, format="csr")
    Iy = sp.eye(ny, format="csr")
    Iz = sp.eye(nz, format="csr")

    # 3D Laplacian via Kronecker sums
    L3 = (
        sp.kron(sp.kron(Iz, Iy), Lx, format="csr")
        + sp.kron(sp.kron(Iz, Ly), Ix, format="csr")
        + sp.kron(sp.kron(Lz, Iy), Ix, format="csr")
    )
    G3 = m2 * sp.eye(nx * ny * nz, format="csr") - L3
    return G3


# ----------------------------------------------------------------------
# Masks for discs and balls
# ----------------------------------------------------------------------

def disc_mask_2d(nx: int, ny: int, dx: float, dy: float, radius: float) -> np.ndarray:
    """
    Build a flat (nx*ny,) mask equal to 1 inside a disc of given radius centred
    in the domain, and 0 outside.
    """
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    mask = (R <= radius).astype(float)
    return mask.ravel()


def ball_mask_3d(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float, radius: float) -> np.ndarray:
    """
    Build a flat (nx*ny*nz,) mask equal to 1 inside a ball of given radius
    centred in the domain, and 0 outside.
    """
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    z = (np.arange(nz) - (nz - 1) / 2.0) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    mask = (R <= radius).astype(float)
    return mask.ravel()


# ----------------------------------------------------------------------
# Scaling analysis helpers
# ----------------------------------------------------------------------

@dataclass
class ScalingFit:
    slope: float
    intercept: float


def fit_loglog(x: np.ndarray, y: np.ndarray, x_min: float | None = None, x_max: float | None = None) -> ScalingFit:
    """
    Fit log y = slope * log x + intercept over a specified x-range.

    If x_min/x_max are None, use the full range.
    """
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max

    x_fit = x[mask]
    y_fit = y[mask]

    logx = np.log(x_fit)
    logy = np.log(y_fit)

    A = np.vstack([logx, np.ones_like(logx)]).T
    slope, intercept = np.linalg.lstsq(A, logy, rcond=None)[0]
    return ScalingFit(slope=slope, intercept=intercept)


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def main() -> None:
    # Grid parameters: chosen to keep the 3D operator manageable but still give
    # a useful dynamic range in R.
    nx_2d, ny_2d = 128, 128
    nx_3d, ny_3d, nz_3d = 48, 48, 48

    dx = dy = dz = 1.0
    m2 = 1  # small mass term to regularise long modes

    # Radii to probe (in grid units). We keep these conservative to avoid
    # significant boundary contamination.
    max_R_2d = min(nx_2d, ny_2d) * 0.4
    max_R_3d = min(nx_3d, ny_3d, nz_3d) * 0.4

    radii_2d = np.linspace(4.0, max_R_2d, 16)
    radii_3d = np.linspace(4.0, max_R_3d, 12)

    # Build precision operators
    print("[DimCrossover] Building 2D precision operator...")
    G2 = precision_2d(nx_2d, ny_2d, dx, dy, m2)

    print("[DimCrossover] Building 3D precision operator...")
    G3 = precision_3d(nx_3d, ny_3d, nz_3d, dx, dy, dz, m2)

    # Compute Fisher curvature of constant-mode masks
    I_sigma_2d = []
    I_sigma_3d = []

    print("[DimCrossover] Scanning radii in 2D...")
    for R in radii_2d:
        chi = disc_mask_2d(nx_2d, ny_2d, dx, dy, radius=R)
        # I_sigma(R) = chi^T G chi
        I = chi @ (G2 @ chi)
        I_sigma_2d.append(I)
        print(f"  R_2D = {R:5.1f}, I_sigma_2D = {I:.6e}")

    print("[DimCrossover] Scanning radii in 3D...")
    for R in radii_3d:
        chi = ball_mask_3d(nx_3d, ny_3d, nz_3d, dx, dy, dz, radius=R)
        I = chi @ (G3 @ chi)
        I_sigma_3d.append(I)
        print(f"  R_3D = {R:5.1f}, I_sigma_3D = {I:.6e}")

    I_sigma_2d = np.array(I_sigma_2d)
    I_sigma_3d = np.array(I_sigma_3d)

    # Fit effective exponents over the inner range where boundary effects are mild.
    # These ranges can be tuned; here we simply avoid the smallest and largest radii.
    fit_2d = fit_loglog(radii_2d, I_sigma_2d, x_min=radii_2d[2], x_max=radii_2d[-3])
    fit_3d = fit_loglog(radii_3d, I_sigma_3d, x_min=radii_3d[2], x_max=radii_3d[-3])

    print("\n[DimCrossover] Effective scaling exponents (log-log fits):")
    print(f"  2D: I_sigma ~ R^{fit_2d.slope:.3f}")
    print(f"  3D: I_sigma ~ R^{fit_3d.slope:.3f}")

    # Save data
    out_dir = Path("results") / "fisher_dim_crossover_3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fisher_dim_crossover_3d.npz"

    np.savez(
        out_path,
        radii_2d=radii_2d,
        I_sigma_2d=I_sigma_2d,
        radii_3d=radii_3d,
        I_sigma_3d=I_sigma_3d,
        fit_2d_slope=fit_2d.slope,
        fit_2d_intercept=fit_2d.intercept,
        fit_3d_slope=fit_3d.slope,
        fit_3d_intercept=fit_3d.intercept,
        nx_2d=nx_2d,
        ny_2d=ny_2d,
        nx_3d=nx_3d,
        ny_3d=ny_3d,
        nz_3d=nz_3d,
        dx=dx,
        dy=dy,
        dz=dz,
        m2=m2,
    )

    print(f"\n[DimCrossover] Saved data to: {out_path}")

    # Optional plotting
    make_plots = True
    if make_plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(radii_2d, I_sigma_2d, "o-", label="2D disc")
        ax.loglog(radii_3d, I_sigma_3d, "s-", label="3D ball")

        # Reference slopes for comparison
        R_ref = np.array([radii_2d[0], radii_2d[-1]])
        ax.loglog(R_ref, np.exp(fit_2d.intercept) * R_ref ** fit_2d.slope, "k--", label=f"2D fit: R^{fit_2d.slope:.2f}")
        R_ref3 = np.array([radii_3d[0], radii_3d[-1]])
        ax.loglog(R_ref3, np.exp(fit_3d.intercept) * R_ref3 ** fit_3d.slope, "k-.", label=f"3D fit: R^{fit_3d.slope:.2f}")

        ax.set_xlabel(r"$R$ (lattice units)")
        ax.set_ylabel(r"$I_\sigma(R)$")
        ax.set_title("Fisher curvature of constant mode: 2D disc vs 3D ball")
        ax.legend(loc="best")
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig_path = out_dir / "fisher_dim_crossover_3d.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        print(f"[DimCrossover] Saved plot to: {fig_path}")


if __name__ == "__main__":
    main()
