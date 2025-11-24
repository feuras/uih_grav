#!/usr/bin/env python3
"""
Fisher scalar gravity scan in spherical symmetry.

For a grid of toy matter density profiles rho_m(r), this script:
  - Solves the scalar Fisher equation in the Newtonian regime:
        Δ rho(r) = -kappa * rho_m(r)
    using the radial integral formula.
  - Constructs sigma(r) = log(rho(r)/rho0).
  - Computes the Fisher gradient density:
        rho_grad(r) = (c^2 / (8*pi*G)) * |d sigma / dr|^2
  - Forms an effective density:
        rho_eff(r) = rho_m(r) + rho_grad(r)
  - Computes enclosed mass and circular velocities for rho_m and rho_eff.

Each parameter set is processed in parallel (up to 21 cores), and the
results for each case are written to an .npz file:

    fisher_scalar_case_<case_id>.npz

containing:
    r, rho_m, rho, sigma, rho_grad, rho_eff,
    M_m, M_eff, v_c_m, v_c_eff, residual

The console output prints a summary of how large the Fisher gradient
contribution becomes in each case.

This script deliberately keeps ρ and the mapping to geometry abstract:
it only assumes the scalar Fisher equation and Newtonian matching at
the level of the potential. You can rescale units and ranges at the top.
"""

import os
import math
import json
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Callable

import numpy as np
import multiprocessing as mp

# =========================
# Global configuration
# =========================

# Target number of worker processes; will be capped by os.cpu_count()
TARGET_PROCESSES = 21

# Radial grid settings
R_MAX = 50.0          # outer radius (in arbitrary length units; rescale as you like)
N_R = 4000            # number of radial points

# Physics constants (you can switch to SI if you like)
# For structural scans it's convenient to use "natural" units.
USE_SI_UNITS = False

if USE_SI_UNITS:
    # SI units
    G = 6.67430e-11          # m^3 kg^-1 s^-2
    c = 2.99792458e8         # m s^-1
    rho0_default = 1.0       # kg m^-3 (placeholder, adjust as desired)
else:
    # Dimensionless "natural" units for structure exploration
    G = 1.0
    c = 1.0
    rho0_default = 1.0

# kappa from Newtonian matching: ΔΦ_eff ≈ 4πG ρ_m when ρ ≈ rho0
# Recall kappa = (8πG / c^2) * rho0
def kappa_from_rho0(rho0: float) -> float:
    return (8.0 * math.pi * G / (c ** 2)) * rho0


# =========================
# Density profile models
# =========================

@dataclass
class ProfileParams:
    profile_type: str      # 'gaussian', 'exponential', 'nfw'
    rho0: float            # reference Fisher density
    rho_c: float           # characteristic matter density scale
    r_scale: float         # scale radius
    case_id: int           # integer id for file naming
    extra: Dict = None     # optional extra config

def rho_m_gaussian(r: np.ndarray, rho_c: float, r_scale: float) -> np.ndarray:
    """
    Spherically symmetric Gaussian core:
        rho_m(r) = rho_c * exp(-r^2 / (2 r_scale^2))
    """
    return rho_c * np.exp(-0.5 * (r / r_scale) ** 2)

def rho_m_exponential(r: np.ndarray, rho_c: float, r_scale: float) -> np.ndarray:
    """
    Spherically symmetric exponential:
        rho_m(r) = rho_c * exp(-r / r_scale)
    """
    return rho_c * np.exp(-r / r_scale)

def rho_m_nfw_like(r: np.ndarray, rho_c: float, r_scale: float, eps: float = 1e-6) -> np.ndarray:
    """
    NFW-like cusp:
        rho_m(r) = rho_c / ((r/r_scale) * (1 + r/r_scale)^2 + eps)
    eps regularises the r->0 behavior slightly.
    """
    x = r / r_scale
    return rho_c / (x * (1.0 + x) ** 2 + eps)

PROFILE_FUNCS: Dict[str, Callable[..., np.ndarray]] = {
    "gaussian": rho_m_gaussian,
    "exponential": rho_m_exponential,
    "nfw": rho_m_nfw_like,
}


# =========================
# Numerical helpers
# =========================

def make_radial_grid(r_max: float, n_r: int) -> np.ndarray:
    """
    Build a 1D radial grid in [0, r_max]. We avoid r=0 exactly to
    keep formulas involving 1/r^2 stable; r[0] is a small positive value.
    """
    # start slightly away from zero
    return np.linspace(r_max / n_r, r_max, n_r)


def cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Simple cumulative trapezoidal integral:
        F[i] = ∫_{x[0]}^{x[i]} y(s) ds
    with F[0] = 0.
    """
    n = len(x)
    F = np.zeros_like(y)
    if n < 2:
        return F
    dx = np.diff(x)
    # trapezoid between points i and i+1
    integrand = 0.5 * (y[:-1] + y[1:]) * dx
    F[1:] = np.cumsum(integrand)
    return F


def compute_rho_from_rho_m(
    r: np.ndarray,
    rho_m: np.ndarray,
    kappa: float,
    rho0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve Δ rho(r) = -kappa * rho_m(r) in spherical symmetry
    using the exact radial integral relations:

        d/dr (r^2 rho') = -kappa r^2 rho_m(r)
        => rho'(r) = -kappa / r^2 ∫_0^r s^2 rho_m(s) ds

        rho(r) = rho(R_max) + kappa ∫_r^{R_max} [1/s^2 ∫_0^s u^2 rho_m(u) du] ds
               ≈ rho0 + kappa * J(r)

    We approximate the integrals with cumulative trapezoids.
    Returns:
        rho(r), I(r) where I(r) = ∫_0^r s^2 rho_m(s) ds
    """
    # First integral: I(r) = ∫_0^r s^2 rho_m(s) ds
    integrand1 = (r ** 2) * rho_m
    I = cumulative_trapz(integrand1, r)

    # Compute f2(s) = I(s) / s^2 for s>0
    # (safe because r>0 by construction)
    f2 = I / (r ** 2)

    # Second integral: J(r) = ∫_r^{R_max} f2(s) ds
    # We do this via F(s) = ∫_0^s f2(u) du, J(r) = F(R_max) - F(r).
    F_f2 = cumulative_trapz(f2, r)
    F_total = F_f2[-1]
    J = F_total - F_f2

    rho = rho0 + kappa * J
    return rho, I


def finite_difference_derivative(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Simple 1D derivative df/dr using central differences in the interior
    and one-sided differences at boundaries.
    """
    df = np.zeros_like(f)
    dr = np.diff(r)

    # interior: central
    df[1:-1] = (f[2:] - f[:-2]) / (r[2:] - r[:-2])

    # boundaries: one-sided
    df[0] = (f[1] - f[0]) / (r[1] - r[0])
    df[-1] = (f[-1] - f[-2]) / (r[-1] - r[-2])

    return df


def compute_laplacian_rho(
    r: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """
    Approximate Δ rho(r) in spherical symmetry via:

        Δ rho = (1/r^2) d/dr (r^2 d rho/dr).

    Uses finite differences for derivatives.
    """
    drho_dr = finite_difference_derivative(r, rho)
    # d/dr (r^2 drho/dr)
    y = r ** 2 * drho_dr
    dy_dr = finite_difference_derivative(r, y)
    lap = dy_dr / (r ** 2)
    return lap


def enclosed_mass(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute M(r) = 4π ∫_0^r s^2 rho(s) ds via cumulative trapz.
    """
    integrand = 4.0 * math.pi * (r ** 2) * rho
    return cumulative_trapz(integrand, r)


def circular_velocity(r: np.ndarray, M_r: np.ndarray, G: float) -> np.ndarray:
    """
    Circular velocity v_c(r) = sqrt(G M(r) / r).
    Returns zero at r=0.
    """
    v2 = np.zeros_like(r)
    # Avoid division by zero at r=0; start from index 1
    v2[1:] = G * M_r[1:] / r[1:]
    # negative due to numerical rounding could appear; clip
    v2 = np.maximum(v2, 0.0)
    return np.sqrt(v2)


# =========================
# Worker: one case
# =========================

def run_single_case(params: ProfileParams, r: np.ndarray) -> Dict:
    """
    Run the full pipeline for a single profile:
      1. Build rho_m(r) for the chosen model.
      2. Solve Δ rho = -kappa rho_m to get rho(r).
      3. Compute sigma = log(rho / rho0) and its radial derivative.
      4. Compute Fisher gradient density rho_grad.
      5. Build effective density rho_eff = rho_m + rho_grad.
      6. Compute M_m, M_eff, v_c_m, v_c_eff.
      7. Compute residual of the Fisher equation for diagnostics.
      8. Save all arrays to NPZ and return summary stats.
    """
    try:
        profile_type = params.profile_type
        rho0 = params.rho0
        rho_c = params.rho_c
        r_scale = params.r_scale

        kappa = kappa_from_rho0(rho0)

        if profile_type not in PROFILE_FUNCS:
            raise ValueError(f"Unknown profile_type '{profile_type}'")

        # Build rho_m(r)
        if profile_type == "nfw":
            rho_m = PROFILE_FUNCS[profile_type](r, rho_c, r_scale)
        else:
            rho_m = PROFILE_FUNCS[profile_type](r, rho_c, r_scale)

        # Solve for rho(r)
        rho, I = compute_rho_from_rho_m(r, rho_m, kappa, rho0)

        # Discard points where rho <= 0 (log undefined)
        # In practice, if this happens a lot, the parameter choice is not viable.
        mask_positive = rho > 0
        if not np.all(mask_positive):
            # Keep only region where rho>0 for sigma etc.
            # But still save full arrays for inspection.
            sigma = np.full_like(rho, np.nan)
            sigma[mask_positive] = np.log(rho[mask_positive] / rho0)
        else:
            sigma = np.log(rho / rho0)

        # d sigma / dr
        dsigma_dr = finite_difference_derivative(r, sigma)

        # Fisher gradient density (uses c,G)
        rho_grad = (c ** 2 / (8.0 * math.pi * G)) * (dsigma_dr ** 2)

        # Effective density
        rho_eff = rho_m + rho_grad

        # Mass profiles and circular velocities
        M_m = enclosed_mass(r, rho_m)
        M_eff = enclosed_mass(r, rho_eff)
        v_c_m = circular_velocity(r, M_m, G)
        v_c_eff = circular_velocity(r, M_eff, G)

        # Check residual of Δ rho + kappa rho_m ≈ 0
        lap_rho = compute_laplacian_rho(r, rho)
        residual = lap_rho + kappa * rho_m

        # Diagnostics
        tiny = 1e-12
        max_rel_grad = float(np.nanmax(rho_grad / (rho_m + tiny)))
        max_abs_residual = float(np.nanmax(np.abs(residual)))

        # Save NPZ
        out_fname = f"fisher_scalar_case_{params.case_id:04d}.npz"
        np.savez_compressed(
            out_fname,
            r=r,
            rho_m=rho_m,
            rho=rho,
            sigma=sigma,
            dsigma_dr=dsigma_dr,
            rho_grad=rho_grad,
            rho_eff=rho_eff,
            M_m=M_m,
            M_eff=M_eff,
            v_c_m=v_c_m,
            v_c_eff=v_c_eff,
            residual=residual,
            meta=json.dumps(asdict(params)),
            G=G,
            c=c,
            rho0=rho0,
            kappa=kappa,
        )

        return {
            "case_id": params.case_id,
            "profile_type": profile_type,
            "rho0": rho0,
            "rho_c": rho_c,
            "r_scale": r_scale,
            "max_rel_grad": max_rel_grad,
            "max_abs_residual": max_abs_residual,
            "status": "ok",
        }

    except Exception as e:
        return {
            "case_id": params.case_id,
            "profile_type": params.profile_type,
            "rho0": params.rho0,
            "rho_c": params.rho_c,
            "r_scale": params.r_scale,
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "status": "error",
        }


# =========================
# Parameter grid
# =========================

def build_default_parameter_grid() -> List[ProfileParams]:
    """
    Build a modest scan over profile types, central densities, and
    scale radii. All units are arbitrary unless USE_SI_UNITS=True.

    You can expand or change these ranges freely.
    """
    profiles = ["gaussian", "exponential", "nfw"]

    # Characteristic central densities and scale radii in code units.
    # In SI units, interpret these as kg/m^3 and meters respectively.
    rho_c_values = [0.1 * rho0_default, 1.0 * rho0_default, 10.0 * rho0_default]
    r_scale_values = [0.5 * R_MAX, 0.2 * R_MAX, 0.1 * R_MAX]

    params_list: List[ProfileParams] = []
    case_id = 0
    for p in profiles:
        for rho_c in rho_c_values:
            for r_s in r_scale_values:
                params_list.append(
                    ProfileParams(
                        profile_type=p,
                        rho0=rho0_default,
                        rho_c=rho_c,
                        r_scale=r_s,
                        case_id=case_id,
                        extra={},
                    )
                )
                case_id += 1
    return params_list


# =========================
# Main
# =========================

def main():
    # Build radial grid
    r = make_radial_grid(R_MAX, N_R)

    # Build parameter grid
    params_list = build_default_parameter_grid()
    n_cases = len(params_list)
    print(f"[FisherScalar] Number of parameter cases: {n_cases}")

    # Decide number of processes
    cpu_count = os.cpu_count() or 1
    n_procs = min(TARGET_PROCESSES, cpu_count)
    if n_procs < 1:
        n_procs = 1

    print(f"[FisherScalar] Detected {cpu_count} CPUs, using up to {n_procs} worker processes.")
    print(f"[FisherScalar] R_MAX = {R_MAX}, N_R = {N_R}, G = {G}, c = {c}, rho0_default = {rho0_default}")

    # Prepare arguments
    # We partially apply r to avoid pickling large arrays repeatedly by
    # passing r as a global in the worker; but to keep it simple and robust,
    # we pass r as an argument here. For large runs, you can optimise this.
    args = [(p, r) for p in params_list]

    results: List[Dict] = []

    if n_procs == 1:
        print("[FisherScalar] Running sequentially.")
        for p in params_list:
            res = run_single_case(p, r)
            results.append(res)
    else:
        print("[FisherScalar] Running in parallel.")
        # Use multiprocessing Pool
        with mp.Pool(processes=n_procs) as pool:
            for res in pool.starmap(run_single_case, args):
                results.append(res)

    # Summarise
    print("\n[FisherScalar] Summary of results:")
    for res in results:
        cid = res.get("case_id")
        if res.get("status") != "ok":
            print(f"  Case {cid:04d} ERROR: {res.get('error')}")
            continue
        print(
            f"  Case {cid:04d} "
            f"{res['profile_type']:10s} "
            f"rho_c/rho0={res['rho_c']/res['rho0']:.3g} "
            f"r_scale={res['r_scale']:.3g} "
            f"max_rel_grad={res['max_rel_grad']:.3g} "
            f"max_abs_residual={res['max_abs_residual']:.3g}"
        )

    # Optionally, write a JSON summary
    summary_fname = "fisher_scalar_summary.json"
    with open(summary_fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[FisherScalar] Wrote JSON summary to {summary_fname}")
    print("[FisherScalar] Each case data saved as fisher_scalar_case_XXXX.npz")


if __name__ == "__main__":
    main()
