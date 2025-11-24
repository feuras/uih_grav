#!/usr/bin/env python3
"""
03_scalar_gravity_compactness_scan.py

Scan scalar Fisher halos across many amplitudes and scale radii
in the weak-field regime, and verify that

    M_grad / M  ~  const * (G M) / (R_core c^2)

for a simple spherical exponential profile.

We work in code units G = c = 1 and choose a small gamma to mimic
the physical weak-field regime. The script uses spherical symmetry
and integrates the Poisson equation for u(r) analytically via the
enclosed mass M(r).
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def make_r_grid(R_max, N_r):
    """Uniform radial grid from 0 to R_max."""
    return np.linspace(0.0, R_max, N_r)


def rho_m_exponential(r, rho_c, r_scale):
    """Spherical exponential mass density profile."""
    return rho_c * np.exp(-r / r_scale)


def compute_mass_profile(r, rho_m):
    """
    Compute enclosed mass M(r) = 4 pi ∫_0^r s^2 rho_m(s) ds
    using a simple trapezoidal rule on a uniform grid.
    """
    dr = r[1] - r[0]
    shell = 4.0 * np.pi * r**2 * rho_m
    # Trapezoidal cumulative integral: cumtrapz with uniform spacing
    # M[i] ≈ ∑_{j=0}^{i-1} 0.5*(shell[j] + shell[j+1]) * dr
    shell_mid = 0.5 * (shell[:-1] + shell[1:])
    M = np.empty_like(r)
    M[0] = 0.0
    M[1:] = np.cumsum(shell_mid) * dr
    return M


def compute_u_from_mass(r, M):
    """
    Solve Δu = -rho_m in spherical symmetry with u(R_max) = 0.

    Using (r^2 u')' = -r^2 rho_m and M(r) = 4 pi ∫_0^r s^2 rho_m(s) ds,
    one has u'(r) = -M(r) / (4 pi r^2). Then

        u(r) = ∫_r^{R_max} u'(s) ds

    which we compute with a reversed cumulative trapezoid.
    """
    dr = r[1] - r[0]
    up = np.zeros_like(r)
    # Avoid division by zero at r = 0 by copying the value from the first shell
    up[1:] = -M[1:] / (4.0 * np.pi * r[1:]**2)
    up[0] = up[1]

    # a[i] = 0.5 * (up[i] + up[i+1]) for trapezoid weights
    a = 0.5 * (up[:-1] + up[1:])
    # s[i] = sum_{k=i}^{N-2} a[k]
    s = np.cumsum(a[::-1])[::-1]

    u = np.zeros_like(r)
    # u[N-1] = 0 by boundary condition
    u[:-1] = dr * s
    u[-1] = 0.0
    return u, up


def compute_sigma_and_grad(r, u, gamma):
    """
    Compute sigma(r) = log(1 + gamma u) and its radial derivative
    with a central-difference scheme.
    """
    dr = r[1] - r[0]
    sigma = np.log1p(gamma * u)

    sigma_prime = np.zeros_like(r)
    sigma_prime[1:-1] = (sigma[2:] - sigma[:-2]) / (2.0 * dr)
    sigma_prime[0] = sigma_prime[1]
    sigma_prime[-1] = sigma_prime[-2]
    return sigma, sigma_prime


def integrate_mass(r, rho):
    """Total mass 4 pi ∫ r^2 rho(r) dr."""
    return np.trapz(4.0 * np.pi * r**2 * rho, r)


def run_single_case(case_idx,
                    rho_c,
                    r_scale,
                    R_max,
                    N_r,
                    gamma,
                    alpha,
                    G_code,
                    c_code):
    """
    Compute M and M_grad for one (rho_c, r_scale) case.
    """
    r = make_r_grid(R_max, N_r)
    rho_m = rho_m_exponential(r, rho_c, r_scale)

    # Enclosed mass and u
    M_r = compute_mass_profile(r, rho_m)
    u, u_prime = compute_u_from_mass(r, M_r)

    # Fisher potential and gradient
    sigma, sigma_prime = compute_sigma_and_grad(r, u, gamma)

    # Fisher halo density: rho_grad = alpha * c^2 / (8 pi G) * |sigma'|^2
    rho_grad = alpha * (c_code**2 / (8.0 * np.pi * G_code)) * sigma_prime**2

    # Integrate baryonic and halo masses
    M_tot = integrate_mass(r, rho_m)
    M_grad = integrate_mass(r, rho_grad)

    # Define core radius as r_scale for this test
    R_core = r_scale
    compactness = G_code * M_tot / (R_core * c_code**2)

    result = {
        "idx": case_idx,
        "rho_c": rho_c,
        "r_scale": r_scale,
        "M_tot": M_tot,
        "M_grad": M_grad,
        "R_core": R_core,
        "compactness": compactness,
        "ratio": M_grad / M_tot if M_tot > 0 else np.nan,
    }
    return result


def main():
    # Code units
    G_code = 1.0
    c_code = 1.0

    # Weak-field gamma = 8 pi G / c^2 multiplied by a small factor if desired.
    # Here we pick a small effective gamma to keep |gamma u| << 1 across the grid.
    gamma = 1.0e-5

    # UIH information coupling in code units
    alpha = 1.0

    # Radial domain and resolution
    r_scales = [0.5, 1.0, 2.0, 4.0]
    rho_cs = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

    R_max = 20.0 * max(r_scales)
    N_r = 200000

    cases = []
    idx = 0
    for r_scale in r_scales:
        for rho_c in rho_cs:
            cases.append((idx, rho_c, r_scale))
            idx += 1

    print("[CompactnessScan] Number of parameter cases:", len(cases))
    print("[CompactnessScan] R_MAX = {}, N_R = {}, gamma = {}, alpha = {}".format(
        R_max, N_r, gamma, alpha
    ))

    max_workers = 21

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_case = {}
        for idx_case, rho_c, r_scale in cases:
            fut = ex.submit(
                run_single_case,
                idx_case,
                rho_c,
                r_scale,
                R_max,
                N_r,
                gamma,
                alpha,
                G_code,
                c_code,
            )
            future_to_case[fut] = (idx_case, rho_c, r_scale)

        for fut in as_completed(future_to_case):
            res = fut.result()
            results.append(res)
            print(
                "  Case {:03d} rho_c={:8.3g} r_scale={:5.2f} "
                "M={:9.3g} C=GM/(Rc^2)={:9.3g} Mgrad/M={:9.3g}".format(
                    res["idx"],
                    res["rho_c"],
                    res["r_scale"],
                    res["M_tot"],
                    res["compactness"],
                    res["ratio"],
                )
            )

    # Collect arrays for a simple linear fit M_grad/M = A * compactness
    compactness_vals = []
    ratio_vals = []
    for res in results:
        if np.isfinite(res["ratio"]):
            compactness_vals.append(res["compactness"])
            ratio_vals.append(res["ratio"])

    compactness_vals = np.array(compactness_vals)
    ratio_vals = np.array(ratio_vals)

    # Least-squares slope A = sum(C * R) / sum(C^2)
    num = np.sum(compactness_vals * ratio_vals)
    den = np.sum(compactness_vals**2)
    if den > 0:
        A_fit = num / den
        # Compute a simple R^2-like measure
        pred = A_fit * compactness_vals
        ss_tot = np.sum((ratio_vals - np.mean(ratio_vals))**2)
        ss_res = np.sum((ratio_vals - pred)**2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print()
        print("[CompactnessScan] Best-fit M_grad/M ≈ A * (GM/(R_core c^2))")
        print("[CompactnessScan] A_fit ≈ {:.3g}, R2 ≈ {:.4f}".format(A_fit, R2))
    else:
        print("[CompactnessScan] Could not perform linear fit (denominator zero).")


if __name__ == "__main__":
    main()
