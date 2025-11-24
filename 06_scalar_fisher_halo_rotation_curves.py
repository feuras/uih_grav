#!/usr/bin/env python3
"""
06_scalar_fisher_halo_rotation_curves.py

Compute rotation curves for a spherical baryonic profile plus a Fisher scalar halo
with alpha(r) ∝ r^2, using disc-Fisher-motivated scaling.

Units: G = c = 1.

Model:
  - Baryons: spherical exponential density
        rho_b(r) = rho0 * exp(-r / R_d)
    with rho0 chosen so that M_b(R_max) = M_b_total.

  - Fisher scalar:
        sigma'(r) ≈ -2 M_b(r) / r^2
    and
        rho_grad(r) = alpha(r) * (sigma'(r))^2 / (8*pi)
                     = alpha0 * (r/R0)^2 * M_b(r)^2 / (2*pi*r^4)

    Enclosed halo mass:
        M_grad(r) = 4*pi ∫_0^r s^2 rho_grad(s) ds
                  = (2*alpha0/R0^2) ∫_0^r M_b(s)^2 ds.

  - Total mass:
        M_tot(r) = M_b(r) + M_grad(r)

  - Circular speed:
        v_c(r) = sqrt(M_tot(r) / r)
"""

import numpy as np


def build_radial_grid(R_max, N_r):
    """Radial grid from a small epsilon to R_max."""
    r_min = R_max / (N_r * 10.0)
    return np.linspace(r_min, R_max, N_r)


def build_baryon_profile(r, R_d, M_b_total):
    """
    Build exponential baryon density and enclosed mass M_b(r)
    such that M_b(R_max) = M_b_total.

    rho_b(r) = rho0 * exp(-r / R_d)

    We determine rho0 by normalising at the largest radius in r.
    """
    # Unnormalised density with rho0 = 1
    rho_unnorm = np.exp(-r / R_d)

    # Compute unnormalised enclosed mass
    dr = np.diff(r)
    r_mid = 0.5 * (r[:-1] + r[1:])
    integrand = 4.0 * np.pi * (r_mid**2) * rho_unnorm[:-1]
    M_b_unnorm = np.zeros_like(r)
    M_b_unnorm[1:] = np.cumsum(integrand * dr)

    # Normalise
    rho0 = M_b_total / M_b_unnorm[-1]
    rho_b = rho0 * rho_unnorm
    M_b = rho0 * M_b_unnorm

    return rho_b, M_b


def compute_fisher_halo(r, M_b, alpha0, R0):
    """
    Given radial grid r, enclosed baryon mass M_b(r),
    and Fisher scaling alpha(r) = alpha0 * (r/R0)^2,
    compute sigma'(r), rho_grad(r), M_grad(r).
    """
    # Avoid division by zero at the origin by starting at r[0] > 0
    sigma_prime = -2.0 * M_b / (r**2)  # G = c = 1
    alpha = alpha0 * (r / R0) ** 2

    # rho_grad(r) = alpha(r) * (sigma')^2 / (8*pi)
    # using the simplified expression:
    # rho_grad = alpha * M_b^2 / (2*pi * r^4)
    rho_grad = alpha * (M_b**2) / (2.0 * np.pi * r**4)

    # Enclosed halo mass:
    # M_grad(r) = 4*pi ∫_0^r s^2 rho_grad(s) ds
    dr = np.diff(r)
    r_mid = 0.5 * (r[:-1] + r[1:])
    integrand = 4.0 * np.pi * (r_mid**2) * rho_grad[:-1]
    M_grad = np.zeros_like(r)
    M_grad[1:] = np.cumsum(integrand * dr)

    return sigma_prime, rho_grad, M_grad


def calibrate_alpha0(r, M_b, R0, R_target, f_target):
    """
    Given r, M_b(r), and R0, find alpha0 such that at radius R_target
    the halo fraction M_grad/M_b is approximately f_target in the
    asymptotic Fisher halo regime.

    Using the analytic relation:
        M_grad(R) ≈ (2*alpha0/R0^2) ∫_0^R M_b(s)^2 ds
    so
        f(R) = M_grad(R) / M_b(R)
             ≈ (2*alpha0/R0^2) * I(R) / M_b(R),

    hence
        alpha0 = f_target * R0^2 * M_b(R_target) / (2 * I(R_target)).
    """
    # Find nearest grid index for R_target
    idx_t = np.argmin(np.abs(r - R_target))
    R_t = r[idx_t]
    M_b_t = M_b[idx_t]

    # Segment up to R_t
    r_seg = r[:idx_t + 1]
    M_b_seg = M_b[:idx_t + 1]

    # Integral I = ∫_0^{R_t} M_b(s)^2 ds
    I_Mb2 = np.trapz(M_b_seg**2, r_seg)

    # Avoid division by zero
    if I_Mb2 <= 0.0 or M_b_t <= 0.0:
        raise RuntimeError("Calibration failed: non positive integral or M_b at R_target.")

    alpha0 = f_target * (R0**2) * M_b_t / (2.0 * I_Mb2)

    return alpha0, R_t, M_b_t, I_Mb2


def main():
    # Core numerical parameters
    R_max = 100000.0
    N_r = 40000000

    # Baryon profile parameters
    R_d = 3.0           # baryon exponential scale length
    M_b_total = 1.0     # total baryonic mass at R_max in code units

    # Fisher scaling parameters
    R0 = 5.0            # reference radius for alpha(r)

    # Calibration settings
    use_calibration = True
    R_target = 50.0     # radius at which to fix halo fraction
    f_target = 5.0      # desired M_grad / M_b at R_target if calibrating

    print("[FisherHaloRC] Building radial grid...")
    r = build_radial_grid(R_max, N_r)

    print("[FisherHaloRC] Building baryonic profile...")
    rho_b, M_b = build_baryon_profile(r, R_d, M_b_total)

    if use_calibration:
        print("[FisherHaloRC] Calibrating alpha0...")
        alpha0, R_t, M_b_t, I_Mb2 = calibrate_alpha0(r, M_b, R0, R_target, f_target)
        print(
            f"[FisherHaloRC] Calibration at R ≈ {R_t:.3f}: "
            f"M_b={M_b_t:.4e}, ∫ M_b^2 ds={I_Mb2:.4e}, "
            f"alpha0 needed for M_grad/M_b ≈ {f_target} is ≈ {alpha0:.4g}"
        )
    else:
        # Manual alpha0 if you want to bypass calibration
        alpha0 = 1.0
        print(f"[FisherHaloRC] Using manual alpha0 = {alpha0:.4g}")

    print("[FisherHaloRC] Computing Fisher halo...")
    sigma_prime, rho_grad, M_grad = compute_fisher_halo(r, M_b, alpha0, R0)

    M_tot = M_b + M_grad
    v_c = np.sqrt(M_tot / r)  # G = 1

    # Diagnostics at a few radii
    sample_rs = [R_d, 2 * R_d, 4 * R_d, R_max * 0.5, R_max * 0.8]
    print()
    print("[FisherHaloRC] Sample radii diagnostics:")
    for R_s in sample_rs:
        idx = np.argmin(np.abs(r - R_s))
        print(
            f"  r ≈ {r[idx]:6.2f} : "
            f"M_b={M_b[idx]:.4e}, M_grad={M_grad[idx]:.4e}, "
            f"M_grad/M_b={M_grad[idx]/M_b[idx]:.3g}  "
            f"v_c={v_c[idx]:.4e}"
        )

    # Global diagnostics
    print()
    print("[FisherHaloRC] Global diagnostics:")
    print(f"  Total baryonic mass M_b(R_max)  = {M_b[-1]:.6e}")
    print(f"  Total halo mass M_grad(R_max)   = {M_grad[-1]:.6e}")
    print(f"  Halo fraction M_grad/M_b        = {M_grad[-1]/M_b[-1]:.3g}")

    # Save profiles to a text file for plotting or further analysis
    out = np.column_stack([r, rho_b, M_b, rho_grad, M_grad, v_c])
    np.savetxt(
        "06_fisher_halo_rotation_curve_profile.txt",
        out,
        header="r  rho_b  M_b  rho_grad  M_grad  v_c"
    )
    print()
    print("[FisherHaloRC] Saved radial profiles to "
          "06_fisher_halo_rotation_curve_profile.txt")

    # Optional quick plots (requires matplotlib)
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(r, v_c, label="v_c total")
    plt.loglog(r, np.sqrt(M_b / r), "--", label="v_c baryons only")
    plt.xlabel("r")
    plt.ylabel("v_c")
    plt.legend()
    plt.title("Rotation curve with Fisher halo")
    plt.tight_layout()

    plt.figure()
    plt.loglog(r, rho_b, label="rho_b")
    plt.loglog(r, rho_grad, label="rho_grad (halo)")
    plt.xlabel("r")
    plt.ylabel("density")
    plt.legend()
    plt.title("Baryon and Fisher halo densities")
    plt.tight_layout()

    plt.show()
    """


if __name__ == "__main__":
    main()
