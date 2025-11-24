#!/usr/bin/env python3
"""
02_scalar_gravity_scan_spherical_deep.py

Deeper scan of the scalar Fisher gravity toy model in spherical symmetry.

Compared to 01:
  - Larger parameter grid (more values of rho_c and r_scale).
  - More diagnostics per case:
        * radius where rho_grad/rho_m is maximal,
        * first radius where rho_grad ~ rho_m (if any),
        * total mass ratio M_grad(R_max)/M_m(R_max),
        * max v_c,eff / v_c,m.
  - Still uses multiprocessing up to 21 processes.

Units: by default uses dimensionless "code units" G = c = rho0 = 1.
You can switch to SI by setting USE_SI_UNITS = True, and then interpret
r, rho_m, etc physically.
"""

import os
import math
import json
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Callable, Optional

import numpy as np
import multiprocessing as mp

# =========================
# Global configuration
# =========================

TARGET_PROCESSES = 21

R_MAX = 1000.0          # Outer radius
N_R = 1000000            # Finer radial grid

USE_SI_UNITS = False

if USE_SI_UNITS:
    G = 6.67430e-11
    c = 2.99792458e8
    rho0_default = 1.0   # kg/m^3 placeholder, change as needed
else:
    G = 1.0
    c = 1.0
    rho0_default = 1.0

def kappa_from_rho0(rho0: float) -> float:
    return (8.0 * math.pi * G / (c ** 2)) * rho0


# =========================
# Density profiles
# =========================

@dataclass
class ProfileParams:
    profile_type: str
    rho0: float
    rho_c: float
    r_scale: float
    case_id: int
    extra: Dict = None

def rho_m_gaussian(r: np.ndarray, rho_c: float, r_scale: float) -> np.ndarray:
    return rho_c * np.exp(-0.5 * (r / r_scale) ** 2)

def rho_m_exponential(r: np.ndarray, rho_c: float, r_scale: float) -> np.ndarray:
    return rho_c * np.exp(-r / r_scale)

def rho_m_nfw_like(r: np.ndarray, rho_c: float, r_scale: float, eps: float = 1e-6) -> np.ndarray:
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
    return np.linspace(r_max / n_r, r_max, n_r)

def cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(x)
    F = np.zeros_like(y)
    if n < 2:
        return F
    dx = np.diff(x)
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
    Same integral formula as in 01:
        I(r) = ∫_0^r s^2 rho_m(s) ds
        f2(s) = I(s)/s^2
        J(r) = ∫_r^{R_max} f2(s) ds
        rho(r) = rho0 + kappa * J(r)
    """
    integrand1 = (r ** 2) * rho_m
    I = cumulative_trapz(integrand1, r)
    f2 = I / (r ** 2)
    F_f2 = cumulative_trapz(f2, r)
    F_total = F_f2[-1]
    J = F_total - F_f2
    rho = rho0 + kappa * J
    return rho, I

def finite_difference_derivative(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (r[2:] - r[:-2])
    df[0] = (f[1] - f[0]) / (r[1] - r[0])
    df[-1] = (f[-1] - f[-2]) / (r[-1] - r[-2])
    return df

def compute_laplacian_rho(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    drho_dr = finite_difference_derivative(r, rho)
    y = r ** 2 * drho_dr
    dy_dr = finite_difference_derivative(r, y)
    return dy_dr / (r ** 2)

def enclosed_mass(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    integrand = 4.0 * math.pi * (r ** 2) * rho
    return cumulative_trapz(integrand, r)

def circular_velocity(r: np.ndarray, M_r: np.ndarray, G: float) -> np.ndarray:
    v2 = np.zeros_like(r)
    v2[1:] = G * M_r[1:] / r[1:]
    v2 = np.maximum(v2, 0.0)
    return np.sqrt(v2)


# =========================
# Diagnostics
# =========================

def find_r_eq(
    r: np.ndarray,
    rho_m: np.ndarray,
    rho_grad: np.ndarray,
    tol: float = 0.1,
) -> Optional[float]:
    """
    Find the first radius where rho_grad ~ rho_m, in the sense that
        |rho_grad - rho_m| / (rho_m + tiny) < tol
    Returns None if no such radius exists.
    """
    tiny = 1e-14
    denom = rho_m + tiny
    rel_diff = np.abs(rho_grad - rho_m) / denom
    idx = np.where(rel_diff < tol)[0]
    if idx.size == 0:
        return None
    return float(r[idx[0]])

def max_vc_ratio(
    v_c_eff: np.ndarray,
    v_c_m: np.ndarray,
) -> float:
    tiny = 1e-14
    ratio = v_c_eff / (v_c_m + tiny)
    return float(np.nanmax(ratio))


# =========================
# Worker
# =========================

def run_single_case(params: ProfileParams, r: np.ndarray) -> Dict:
    try:
        profile_type = params.profile_type
        rho0 = params.rho0
        rho_c = params.rho_c
        r_scale = params.r_scale

        kappa = kappa_from_rho0(rho0)

        if profile_type not in PROFILE_FUNCS:
            raise ValueError(f"Unknown profile_type '{profile_type}'")

        if profile_type == "nfw":
            rho_m = PROFILE_FUNCS[profile_type](r, rho_c, r_scale)
        else:
            rho_m = PROFILE_FUNCS[profile_type](r, rho_c, r_scale)

        rho, I = compute_rho_from_rho_m(r, rho_m, kappa, rho0)

        mask_positive = rho > 0
        sigma = np.full_like(rho, np.nan)
        sigma[mask_positive] = np.log(rho[mask_positive] / rho0)

        dsigma_dr = finite_difference_derivative(r, sigma)
        rho_grad = (c ** 2 / (8.0 * math.pi * G)) * (dsigma_dr ** 2)

        rho_eff = rho_m + rho_grad

        M_m = enclosed_mass(r, rho_m)
        M_eff = enclosed_mass(r, rho_eff)

        v_c_m = circular_velocity(r, M_m, G)
        v_c_eff = circular_velocity(r, M_eff, G)

        lap_rho = compute_laplacian_rho(r, rho)
        residual = lap_rho + kappa * rho_m

        tiny = 1e-12
        ratio_grad = rho_grad / (rho_m + tiny)
        max_rel_grad = float(np.nanmax(ratio_grad))
        idx_max = int(np.nanargmax(ratio_grad))
        r_max_rel = float(r[idx_max])

        max_abs_residual = float(np.nanmax(np.abs(residual)))

        r_eq = find_r_eq(r, rho_m, rho_grad, tol=0.1)
        M_m_tot = float(M_m[-1])
        M_grad_tot = float(enclosed_mass(r, rho_grad)[-1])
        mass_ratio = M_grad_tot / (M_m_tot + tiny)

        v_ratio = max_vc_ratio(v_c_eff, v_c_m)

        out_fname = f"fisher_scalar_deep_case_{params.case_id:04d}.npz"
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
            "r_max_rel": r_max_rel,
            "max_abs_residual": max_abs_residual,
            "r_eq": r_eq,
            "mass_ratio_grad_to_m": mass_ratio,
            "max_vc_ratio": v_ratio,
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

def build_parameter_grid() -> List[ProfileParams]:
    """
    Expanded grid. Adjust as you like.

    rho_c/rho0 in {0.01, 0.1, 1, 10, 100}
    r_scale as fractions of R_MAX
    """
    profiles = ["gaussian", "exponential", "nfw"]

    rho_c_factors = [0.01, 0.1, 1.0, 10.0, 100.0]
    r_scale_factors = [0.05, 0.1, 0.2, 0.5]

    params_list: List[ProfileParams] = []
    case_id = 0
    for p in profiles:
        for fc in rho_c_factors:
            for fs in r_scale_factors:
                params_list.append(
                    ProfileParams(
                        profile_type=p,
                        rho0=rho0_default,
                        rho_c=fc * rho0_default,
                        r_scale=fs * R_MAX,
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
    r = make_radial_grid(R_MAX, N_R)
    params_list = build_parameter_grid()
    n_cases = len(params_list)
    print(f"[FisherScalarDeep] Number of parameter cases: {n_cases}")

    cpu_count = os.cpu_count() or 1
    n_procs = min(TARGET_PROCESSES, cpu_count)
    if n_procs < 1:
        n_procs = 1
    print(f"[FisherScalarDeep] Detected {cpu_count} CPUs, using up to {n_procs} worker processes.")
    print(f"[FisherScalarDeep] R_MAX = {R_MAX}, N_R = {N_R}, G = {G}, c = {c}, rho0_default = {rho0_default}")

    args = [(p, r) for p in params_list]
    results: List[Dict] = []

    if n_procs == 1:
        print("[FisherScalarDeep] Running sequentially.")
        for p in params_list:
            res = run_single_case(p, r)
            results.append(res)
    else:
        print("[FisherScalarDeep] Running in parallel.")
        with mp.Pool(processes=n_procs) as pool:
            for res in pool.starmap(run_single_case, args):
                results.append(res)

    print("\n[FisherScalarDeep] Summary of results:")
    for res in results:
        cid = res.get("case_id")
        if res.get("status") != "ok":
            print(f"  Case {cid:04d} ERROR: {res.get('error')}")
            continue
        r_eq_str = "None" if res["r_eq"] is None else f"{res['r_eq']:.3g}"
        print(
            f"  Case {cid:04d} {res['profile_type']:11s} "
            f"rho_c/rho0={res['rho_c']/res['rho0']:.3g} "
            f"r_scale={res['r_scale']:.3g} "
            f"max_rel_grad={res['max_rel_grad']:.3g} "
            f"r_max_rel={res['r_max_rel']:.3g} "
            f"Mgrad/Mm={res['mass_ratio_grad_to_m']:.3g} "
            f"r_eq={r_eq_str} "
            f"max_vc_ratio={res['max_vc_ratio']:.3g} "
            f"max_abs_residual={res['max_abs_residual']:.3g}"
        )

    summary_fname = "fisher_scalar_deep_summary.json"
    with open(summary_fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[FisherScalarDeep] Wrote JSON summary to {summary_fname}")
    print("[FisherScalarDeep] Case data saved as fisher_scalar_deep_case_XXXX.npz")


if __name__ == "__main__":
    main()
