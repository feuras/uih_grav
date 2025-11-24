#!/usr/bin/env python3
"""
05_grav_disc_precision_rg_scaling.py

UIH-inspired Newtonian disc Fisher RG toy.

We consider a 2D lattice disc of radius R (in lattice units). On each site i,
we introduce a "gravitational channel" h_i with a quadratic energy

    Q[h] = 0.5 * h^T Q h,

where the precision matrix is

    Q = -Delta_disc + m^2 I,

with Delta_disc the standard 5-point Laplacian on the 2D lattice restricted
to the disc, and m^2 > 0 a small "mass" term.

We then define a family of Gaussian measures

    p(h | sigma) ∝ exp( -0.5 (h - a*sigma)^T Q (h - a*sigma) )

with sensitivity vector a_i. The Fisher information for sigma is

    I_sigma = a^T Q a.

We compute I_sigma for discs of growing radius R and fit I_sigma(R) ∝ R^Delta.
For a uniform a_i and short-range Q, we expect Delta ≈ 2 (area scaling).
"""

import numpy as np
from math import pi
from concurrent.futures import ProcessPoolExecutor, as_completed


def build_disc_coords(R):
    """Return lattice coordinates (x,y) inside a disc of radius R."""
    coords = []
    R2 = R * R
    for x in range(-R, R + 1):
        for y in range(-R, R + 1):
            if x * x + y * y <= R2:
                coords.append((x, y))
    return coords


def build_laplacian_precision(coords, m2=1e-2):
    """
    Build precision matrix Q = -Delta_disc + m^2 I on the given disc.

    We use a standard 5-point Laplacian on the square lattice. Sites outside
    the disc are treated as boundary with Dirichlet BC (h=0), so they do not
    appear explicitly; their effect enters through fewer neighbours at the edge.
    """
    N = len(coords)
    Q = np.zeros((N, N), dtype=float)
    index_of = {coord: idx for idx, coord in enumerate(coords)}

    for idx, (x, y) in enumerate(coords):
        # Diagonal term: 4 from Laplacian plus mass term
        diag = 0.0
        # Check four neighbours
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nbr = (x + dx, y + dy)
            j = index_of.get(nbr)
            if j is not None:
                # Off-diagonal Laplacian coupling
                Q[idx, j] -= 1.0
                diag += 1.0
        Q[idx, idx] += diag + m2

    return Q


def fisher_for_sigma_precision(coords, profile="uniform", m2=1e-2):
    """
    Compute I_sigma = a^T Q a for a disc with given coords and precision Q.

    profile:
        "uniform": a_i = 1 for all sites
        "radial":  a_i ∝ exp( -r_i^2 / (2 R^2) ) as a simple disc-like profile
    """
    N = len(coords)
    if N == 0:
        return 0.0, 0

    Q = build_laplacian_precision(coords, m2=m2)

    # Build sensitivity vector a
    a_vec = np.empty(N, dtype=float)
    if profile == "uniform":
        a_vec.fill(1.0)
    elif profile == "radial":
        # Simple Gaussian profile peaked at centre, width ~ R
        # Convert coords to radii
        # Estimate disc radius from max |coord|
        R_est = max((x * x + y * y) for (x, y) in coords) ** 0.5
        for k, (x, y) in enumerate(coords):
            r = (x * x + y * y) ** 0.5
            a_vec[k] = np.exp(-0.5 * (r / R_est) ** 2)
    else:
        raise ValueError(f"Unknown profile: {profile}")

    # I_sigma = a^T Q a
    Qa = Q @ a_vec
    I_sigma = float(a_vec @ Qa)
    return I_sigma, N


def run_single_R(R, profile="uniform", m2=1e-2):
    coords = build_disc_coords(R)
    I_sigma, N = fisher_for_sigma_precision(coords, profile=profile, m2=m2)
    area_pi = pi * R * R
    return {
        "R": R,
        "N": N,
        "area_pi": area_pi,
        "I_sigma": I_sigma,
    }


def fit_power_law(xs, ys):
    """Fit ys ~ A * xs^Delta in log-log space."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = (xs > 0) & (ys > 0)
    xs = xs[mask]
    ys = ys[mask]
    logx = np.log(xs)
    logy = np.log(ys)
    A = np.vstack([logx, np.ones_like(logx)]).T
    sol, *_ = np.linalg.lstsq(A, logy, rcond=None)
    Delta = sol[0]
    logA = sol[1]
    return float(np.exp(logA)), float(Delta)


def main():
    R_values = list(range(3, 31, 2))  # 3,5,...,29
    profile = "uniform"  # or "radial"
    m2 = 1e-2

    print("[GravDiscRG] Radii:", R_values)
    print("[GravDiscRG] Profile:", profile)
    print("[GravDiscRG] m^2:", m2)

    max_workers = 4
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_single_R, R, profile, m2) for R in R_values]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(
                f"  R={res['R']:2d} N={res['N']:5d} "
                f"I_sigma={res['I_sigma']:.4g} "
                f"I_sigma/N={res['I_sigma']/res['N']:.4g}"
            )

    results.sort(key=lambda d: d["R"])
    Rs = [r["R"] for r in results]
    Is = [r["I_sigma"] for r in results]
    Ns = [r["N"] for r in results]

    A_R, Delta_R = fit_power_law(Rs, Is)
    print()
    print("[GravDiscRG] Fit I_sigma(R) ≈ A * R^Delta")
    print(f"[GravDiscRG] A ≈ {A_R:.4g}, Delta ≈ {Delta_R:.4f}")

    A_N, Delta_N = fit_power_law(Ns, Is)
    print()
    print("[GravDiscRG] Fit I_sigma(N) ≈ A * N^Delta")
    print(f"[GravDiscRG] A ≈ {A_N:.4g}, Delta ≈ {Delta_N:.4f}")


if __name__ == "__main__":
    main()
