#!/usr/bin/env python3
"""
21_bullet_cluster_toy_nonlocal.py

UIH Bullet test with a non-local halo kernel and plotting.

We set up a 2D "plane of the sky" toy model with:
  - two compact galaxy clumps (collisionless),
  - two broader gas clumps (collisional),
and construct:
  - Sigma_gal(x, y): stellar surface density,
  - Sigma_gas(x, y): gas surface density,
  - Sigma_b(x, y)   = Sigma_gal + Sigma_gas.

The UIH halo surface density is modelled as a non-local quadratic functional
of the baryons:

    Sigma_h(x)  ∝  (K * [Sigma_b^2])(x),

where K(R) is an isotropic softened 1/R kernel on the periodic grid and *
denotes convolution. This mimics the Fisher-type ρ_b^2 sourcing with a
finite-range interaction.

The effective surface density for lensing is then

    Sigma_eff(x, y) = Sigma_b(x, y) + Sigma_h(x, y)

(up to an overall normalisation which we can absorb).

For a small grid of gas mass fractions q = M_gas / M_gal and size ratios
L = sigma_gas / sigma_gal we:

  1. Compute Sigma_gal, Sigma_gas, Sigma_b, Sigma_h, Sigma_eff.
  2. Extract a 1D cut along the merger axis y = 0.
  3. Locate the peak positions of Sigma_eff on the left-hand side
     (galaxy region) and near the gas.
  4. Measure the ratio Sigma_eff(galaxy peak) / Sigma_eff(gas peak).

We also produce plots for one fiducial configuration that a referee can
eyeball:

  - 2x2 map panel: Sigma_gal, Sigma_gas, Sigma_b, Sigma_eff.
  - 1D cut plot along y = 0, showing the relative peaks.

This script is self-contained and uses only NumPy and Matplotlib.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Grid and basic primitives
# ----------------------------

def make_grid(nx=512, ny=512, Lbox=2.0):
    """
    Construct a uniform 2D grid on [-Lbox, Lbox] x [-Lbox, Lbox].

    Returns:
        X, Y : 2D arrays with shape (ny, nx)
        x, y : 1D coordinate arrays
    """
    x = np.linspace(-Lbox, Lbox, nx)
    y = np.linspace(-Lbox, Lbox, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y, x, y


def gaussian2d(X, Y, x0, y0, sigma_x, sigma_y=None, mass=1.0):
    """
    Normalised 2D Gaussian of total mass "mass" centred at (x0, y0).

    Sigma(x, y) = mass / (2 pi sigma_x sigma_y) *
                  exp(-((x-x0)^2 / (2 sigma_x^2) + (y-y0)^2 / (2 sigma_y^2)))
    """
    if sigma_y is None:
        sigma_y = sigma_x
    norm = mass / (2.0 * np.pi * sigma_x * sigma_y)
    return norm * np.exp(
        -(((X - x0) ** 2) / (2.0 * sigma_x**2)
          + ((Y - y0) ** 2) / (2.0 * sigma_y**2))
    )


# ----------------------------
# Non-local kernel construction
# ----------------------------

def build_softened_newton_kernel(nx, ny, Lbox=2.0, r_soft=0.05, r_cut=None):
    """
    Build an isotropic softened 1/R kernel on the same periodic grid.

        K(R) = 1 / sqrt(R^2 + r_soft^2) for R > 0,
        K(0) is set to K(r_soft) to avoid divergence.

    If r_cut is not None, we multiply by exp(-(R / r_cut)^2) to suppress
    the largest scales. The kernel is then normalised so that sum(K) = 1.

    Returns:
        K      : 2D kernel (ny, nx)
        K_fft  : 2D FFT of the kernel (ny, nx), ready for convolution.
    """
    x = np.linspace(-Lbox, Lbox, nx, endpoint=False)
    y = np.linspace(-Lbox, Lbox, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    K = 1.0 / np.sqrt(R**2 + r_soft**2)

    if r_cut is not None and r_cut > 0.0:
        K *= np.exp(-(R / r_cut) ** 2)

    # Avoid artefacts at the origin
    K[0, 0] = 1.0 / np.sqrt(r_soft**2 + r_soft**2)

    # Normalise for convenience (overall factor is irrelevant for ratios)
    K /= K.sum()

    K_fft = np.fft.fft2(K)
    return K, K_fft


def convolve_periodic(field, K_fft):
    """
    Periodic convolution using FFTs:

        result = ifft2( fft2(field) * K_fft )

    Assumes "field" and K_fft have the same shape.
    """
    F_fft = np.fft.fft2(field)
    conv_fft = F_fft * K_fft
    conv = np.fft.ifft2(conv_fft).real
    return conv


# ----------------------------
# Bullet-like configuration
# ----------------------------

def build_bullet_config(X, Y, q=5.0, L_ratio=10.0):
    """
    Construct Sigma_gal, Sigma_gas, Sigma_b for a Bullet-like configuration.

    Parameters:
        q       : total gas mass fraction M_gas / M_gal
        L_ratio : size ratio sigma_gas / sigma_gal

    Geometry is hard-coded to something Bullet-like but simple:
      - two compact galaxy Gaussians at x = +-d/2, y = 0,
      - two broad gas Gaussians centred closer to the middle.
    """
    # Characteristic scales (dimensionless)
    d = 0.8           # separation of galaxy clumps along x
    sigma_gal = 0.08  # compact galaxy dispersion
    sigma_gas = L_ratio * sigma_gal

    # Total masses (arbitrary units, relative only)
    M_gal_total = 1.0
    M_gas_total = q * M_gal_total

    # Galaxies: two equal clumps
    gal1 = gaussian2d(X, Y, -d / 2.0, 0.0, sigma_gal, mass=M_gal_total / 2.0)
    gal2 = gaussian2d(X, Y,  d / 2.0, 0.0, sigma_gal, mass=M_gal_total / 2.0)
    Sigma_gal = gal1 + gal2

    # Gas: broader, located nearer the middle
    gas1 = gaussian2d(X, Y, -0.2, 0.0, sigma_gas, mass=M_gas_total / 2.0)
    gas2 = gaussian2d(X, Y,  0.2, 0.0, sigma_gas, mass=M_gas_total / 2.0)
    Sigma_gas = gas1 + gas2

    Sigma_b = Sigma_gal + Sigma_gas
    return Sigma_gal, Sigma_gas, Sigma_b


def analyse_merger_axis(x, y, Sigma_gal, Sigma_gas, Sigma_b, Sigma_eff):
    """
    Take a cut along the merger axis y = 0 and extract peak positions and
    amplitudes for baryons and effective lensing density.

    Returns a dict with positions, amplitudes and ratios.
    """
    iy_mid = np.argmin(np.abs(y))
    x_line = x
    gal_line = Sigma_gal[iy_mid, :]
    gas_line = Sigma_gas[iy_mid, :]
    b_line = Sigma_b[iy_mid, :]
    eff_line = Sigma_eff[iy_mid, :]

    def peak_pos(arr, side="left"):
        if side == "left":
            mask = x_line < 0.0
        else:
            mask = x_line > 0.0
        xs = x_line[mask]
        ys = arr[mask]
        return xs[np.argmax(ys)]

    def value_at(arr, x0):
        i = np.argmin(np.abs(x_line - x0))
        return float(arr[i])

    # Left-hand peaks
    x_gal = float(peak_pos(gal_line, "left"))
    x_gas = float(peak_pos(gas_line, "left"))
    x_b   = float(peak_pos(b_line,   "left"))
    x_eff = float(peak_pos(eff_line, "left"))

    b_at_gal   = value_at(b_line,   x_gal)
    b_at_gas   = value_at(b_line,   x_gas)
    eff_at_gal = value_at(eff_line, x_gal)
    eff_at_gas = value_at(eff_line, x_gas)

    return {
        "x_gal": x_gal,
        "x_gas": x_gas,
        "x_b_peak": x_b,
        "x_eff_peak": x_eff,
        "b_at_gal": b_at_gal,
        "b_at_gas": b_at_gas,
        "eff_at_gal": eff_at_gal,
        "eff_at_gas": eff_at_gas,
        "b_ratio_gal_over_gas": b_at_gal / b_at_gas,
        "eff_ratio_gal_over_gas": eff_at_gal / eff_at_gas,
        "x_line": x_line,
        "gal_line": gal_line,
        "gas_line": gas_line,
        "b_line": b_line,
        "eff_line": eff_line,
    }


# ----------------------------
# Plotting helpers
# ----------------------------

def plot_maps(X, Y, Sigma_gal, Sigma_gas, Sigma_b, Sigma_eff, out_path):
    """
    2x2 panel: Sigma_gal, Sigma_gas, Sigma_b, Sigma_eff.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    extent = [X.min(), X.max(), Y.min(), Y.max()]

    im0 = axes[0, 0].imshow(Sigma_gal, origin="lower", extent=extent)
    axes[0, 0].set_title("Galaxy surface density")
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(Sigma_gas, origin="lower", extent=extent)
    axes[0, 1].set_title("Gas surface density")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[1, 0].imshow(Sigma_b, origin="lower", extent=extent)
    axes[1, 0].set_title("Total baryon surface density")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.8)

    im3 = axes[1, 1].imshow(Sigma_eff, origin="lower", extent=extent)
    axes[1, 1].set_title("Effective surface density (baryons + UIH halo)")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)

    for ax in axes.ravel():
        ax.set_xlabel("x (arbitrary units)")
        ax.set_ylabel("y (arbitrary units)")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_merger_cut(analysis, out_path, q, L_ratio):
    """
    1D cut along y = 0, showing Sigma_gal, Sigma_gas, Sigma_b, Sigma_eff.
    """
    x = analysis["x_line"]
    gal_line = analysis["gal_line"]
    gas_line = analysis["gas_line"]
    b_line = analysis["b_line"]
    eff_line = analysis["eff_line"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, gal_line, label="galaxies")
    ax.plot(x, gas_line, label="gas")
    ax.plot(x, b_line, label="baryons (gal + gas)")
    ax.plot(x, eff_line, label="effective (baryons + halo)")

    ax.set_xlabel("x along merger axis (y = 0)")
    ax.set_ylabel("surface density (arb. units)")
    ax.set_title(f"Bullet toy cut, q = {q}, L = {L_ratio}")
    ax.legend(loc="upper right")

    # Mark the left galaxy and gas peaks
    ax.axvline(analysis["x_gal"], linestyle="--", linewidth=0.8)
    ax.axvline(analysis["x_gas"], linestyle=":", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main driver
# ----------------------------

def main():
    out_dir = Path("results") / "bullet_toy_nonlocal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bullet_toy_nonlocal_scan.csv"

    # Grid and kernel shared across configurations
    nx = ny = 512
    Lbox = 2.0

    X, Y, x, y = make_grid(nx=nx, ny=ny, Lbox=Lbox)
    # Softened Newtonian kernel: small softening, cut on box scale
    K, K_fft = build_softened_newton_kernel(
        nx=nx,
        ny=ny,
        Lbox=Lbox,
        r_soft=0.05,
        r_cut=1.0,
    )

    q_values = [1.0, 3.0, 5.0, 10.0]
    L_values = [5.0, 10.0, 20.0]

    rows = []

    # Choose one fiducial config for full plots
    fiducial_q = 5.0
    fiducial_L = 10.0
    fiducial_analysis = None

    for q in q_values:
        for L_ratio in L_values:
            print(f"[BulletToyNL] Running config q={q}, L={L_ratio} ...")

            Sigma_gal, Sigma_gas, Sigma_b = build_bullet_config(
                X, Y, q=q, L_ratio=L_ratio
            )

            # Non-local UIH halo: convolution of Sigma_b^2 with the kernel
            Sigma_h = convolve_periodic(Sigma_b**2, K_fft)

            # Normalise halo to have similar scale to baryons (arbitrary)
            # so that Sigma_eff is not dominated by trivial scaling.
            # Using median of Sigma_b on non-zero pixels as reference.
            mask_nonzero = Sigma_b > (1e-6 * Sigma_b.max())
            scale = np.median(Sigma_b[mask_nonzero]) / np.median(
                Sigma_h[mask_nonzero]
            )
            Sigma_h_scaled = scale * Sigma_h

            Sigma_eff = Sigma_b + Sigma_h_scaled

            analysis = analyse_merger_axis(
                x, y, Sigma_gal, Sigma_gas, Sigma_b, Sigma_eff
            )

            res = {
                "q": q,
                "L": L_ratio,
                "x_gal": analysis["x_gal"],
                "x_gas": analysis["x_gas"],
                "x_b_peak": analysis["x_b_peak"],
                "x_eff_peak": analysis["x_eff_peak"],
                "b_at_gal": analysis["b_at_gal"],
                "b_at_gas": analysis["b_at_gas"],
                "eff_at_gal": analysis["eff_at_gal"],
                "eff_at_gas": analysis["eff_at_gas"],
                "b_ratio_gal_over_gas": analysis["b_ratio_gal_over_gas"],
                "eff_ratio_gal_over_gas": analysis["eff_ratio_gal_over_gas"],
            }
            rows.append(res)

            print(
                f"  Peaks: x_gal={res['x_gal']:.3f}, "
                f"x_gas={res['x_gas']:.3f}, "
                f"x_eff_peak={res['x_eff_peak']:.3f}"
            )
            print(
                f"  Sigma_eff(gal)/Sigma_eff(gas) "
                f"= {res['eff_ratio_gal_over_gas']:.2f}"
            )

            if q == fiducial_q and L_ratio == fiducial_L:
                fiducial_analysis = analysis
                # Save maps and cut plots for this configuration
                map_path = out_dir / f"bullet_maps_q{q:g}_L{L_ratio:g}.png"
                cut_path = out_dir / f"bullet_cut_q{q:g}_L{L_ratio:g}.png"
                plot_maps(X, Y, Sigma_gal, Sigma_gas, Sigma_b, Sigma_eff, map_path)
                plot_merger_cut(fiducial_analysis, cut_path, q, L_ratio)

    # Save summary table
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[BulletToyNL] Saved scan table to {out_csv}")
    else:
        print("[BulletToyNL] No rows to save?")


if __name__ == "__main__":
    main()
