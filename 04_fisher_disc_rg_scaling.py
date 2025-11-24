#!/usr/bin/env python3
"""
04_fisher_disc_rg_scaling.py

UIH Fisher RG toy on a 2D disc.

We build a family of 2D lattice discs of increasing radius R, with lattice spacing a = 1.
On each disc, we define a Gaussian distribution

    p(h | sigma) ∝ exp( -0.5 (h - a*sigma)^T C^{-1} (h - a*sigma) )

where:
  - h is an N-site vector of "metric channels" h_i,
  - a is an N-site vector of sensitivities (here taken uniform),
  - C is a covariance matrix (here: either diagonal or with simple NN correlations).

The Fisher information for sigma is

    I(sigma) = a^T C^{-1} a

which we compute exactly. We then measure how I scales with the disc radius R:

    I(R) ∝ R^{Delta}

and fit Delta. For a disc with short-range correlations, we expect Delta ≈ 2.
"""

import numpy as np
from math import sqrt
from concurrent.futures import ProcessPoolExecutor, as_completed


def build_disc_indices(R_max):
    """
    Return the coordinates (i,j) of lattice sites lying in a disc of radius R_max
    on a square grid with spacing 1, centered at (0,0).

    We include integer points (x,y) with x^2 + y^2 <= R_max^2.
    """
    coords = []
    R2 = R_max * R_max
    # We scan a bounding box [-R_max, R_max] in each direction
    for x in range(-R_max, R_max + 1):
        for y in range(-R_max, R_max + 1):
            if x * x + y * y <= R2:
                coords.append((x, y))
    return coords


def build_covariance(coords, mode="diagonal", sigma2=1.0, rho=0.1):
    """
    Build covariance matrix C for the disc.

    modes:
        "diagonal": independent sites, C = sigma2 * I
        "nn": nearest-neighbour correlated, C_ij = sigma2 if i==j,
              and rho * sigma2 if coords i,j are nearest neighbours.

    Returns:
        C : (N,N) array
    """
    N = len(coords)
    C = np.zeros((N, N), dtype=float)

    if mode == "diagonal":
        np.fill_diagonal(C, sigma2)
        return C

    if mode == "nn":
        # Diagonal
        np.fill_diagonal(C, sigma2)
        # Build a map from coord -> index
        index_of = {coord: idx for idx, coord in enumerate(coords)}
        # Add nearest-neighbour correlations
        for idx, (x, y) in enumerate(coords):
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nbr = (x + dx, y + dy)
                j = index_of.get(nbr)
                if j is not None:
                    C[idx, j] = rho * sigma2
                    C[j, idx] = rho * sigma2
        return C

    raise ValueError(f"Unknown covariance mode: {mode}")


def fisher_for_sigma(coords, cov_mode="diagonal", sigma2=1.0, rho=0.1):
    """
    Compute Fisher information I(sigma) = a^T C^{-1} a for a disc with given coords
    and covariance mode. We take a_i = 1 for all sites.

    Returns:
        I_sigma, N_sites
    """
    N = len(coords)
    if N == 0:
        return 0.0, 0

    C = build_covariance(coords, mode=cov_mode, sigma2=sigma2, rho=rho)
    # Invert covariance (cost ~ N^3, so we should not make N too huge)
    C_inv = np.linalg.inv(C)
    a_vec = np.ones(N, dtype=float)
    I_sigma = float(a_vec @ (C_inv @ a_vec))
    return I_sigma, N


def run_single_R(R, cov_mode="diagonal", sigma2=1.0, rho=0.1):
    """
    Build disc of radius R, compute Fisher information, and return summary.
    """
    coords = build_disc_indices(R)
    I_sigma, N = fisher_for_sigma(coords, cov_mode=cov_mode, sigma2=sigma2, rho=rho)
    area_est = np.pi * R * R
    return {
        "R": R,
        "N": N,
        "area_est": area_est,
        "I_sigma": I_sigma,
    }


def fit_power_law(xs, ys):
    """
    Fit ys ~ A * xs^Delta by least squares in log-log space.
    Returns (A, Delta).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = (xs > 0) & (ys > 0)
    xs = xs[mask]
    ys = ys[mask]
    logx = np.log(xs)
    logy = np.log(ys)
    A = np.vstack([logx, np.ones_like(logx)]).T
    # Solve [logy] ~ [logx,1] [Delta, logA]^T
    sol, *_ = np.linalg.lstsq(A, logy, rcond=None)
    Delta = sol[0]
    logA = sol[1]
    return float(np.exp(logA)), float(Delta)


def main():
    # Radii to test
    R_values = list(range(3, 31, 2))  # 3,5,7,...,29
    cov_mode = "nn"  # "diagonal" or "nn"
    sigma2 = 1.0
    rho = 0.1

    print("[FisherDiscRG] Radii:", R_values)
    print("[FisherDiscRG] Covariance mode:", cov_mode)

    max_workers = 8  # per-R runs; each run does its own dense linear algebra

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for R in R_values:
            futs.append(ex.submit(run_single_R, R, cov_mode, sigma2, rho))
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            print(
                f"  R={res['R']:2d} N={res['N']:5d} "
                f"I_sigma={res['I_sigma']:.4g} "
                f"I_sigma/N={res['I_sigma']/res['N']:.4g}"
            )

    # Sort by R
    results.sort(key=lambda d: d["R"])
    Rs = [r["R"] for r in results]
    Is = [r["I_sigma"] for r in results]
    Ns = [r["N"] for r in results]

    # Fit I_sigma ~ A * R^Delta
    A_fit, Delta_fit = fit_power_law(Rs, Is)
    print()
    print("[FisherDiscRG] Power-law fit I_sigma(R) ≈ A * R^Delta")
    print(f"[FisherDiscRG] A ≈ {A_fit:.4g}, Delta ≈ {Delta_fit:.4f}")

    # Also check I_sigma vs N (should be linear)
    A_N, Delta_N = fit_power_law(Ns, Is)
    print()
    print("[FisherDiscRG] Power-law fit I_sigma(N) ≈ A * N^Delta")
    print(f"[FisherDiscRG] A ≈ {A_N:.4g}, Delta ≈ {Delta_N:.4f}")


if __name__ == "__main__":
    main()
