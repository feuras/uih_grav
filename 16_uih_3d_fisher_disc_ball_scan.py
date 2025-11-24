#!/usr/bin/env python3
"""
16_uih_3d_fisher_disc_ball_scan.py

UIH gravity project: 3D Fisher curvature scan for disc and ball modes.

Purpose
-------
This script constructs a simple 3D lattice model with a symmetric operator

    Q = m2 * I + gamma * L

where L is the standard 3D discrete Laplacian with Dirichlet zero boundary
conditions. The operator Q plays the role of a Fisher precision / Dirichlet
operator in the density sector of a UIH K operator.

For each radius R (in lattice units) the script defines two regions:

  1. A thin disc: points with radius in the x y plane less than or equal to R,
     and z within a fixed number of planes around the mid plane.

  2. A ball: points with full 3D radius less than or equal to R.

For each region S it defines an indicator vector a_S whose entries are 1 on S
and 0 outside, then computes the Fisher curvature

    I_sigma(S) = a_S^T Q a_S

for both disc and ball. This is exactly the quadratic form that appears as the
Fisher curvature of a scalar parameter sigma that shifts the mean field
uniformly inside S for a Gaussian with precision Q.

The script then reports

  - N_disc(R), I_disc(R), I_disc(R) / N_disc(R)
  - N_ball(R), I_ball(R), I_ball(R) / N_ball(R)

and performs log log fits

  - I_disc(R) ~ c_disc * R^p_disc
  - I_ball(R) ~ c_ball * R^p_ball

to estimate effective scaling exponents p_disc and p_ball.

A minimal RG crossover scale R_cut is also estimated as

    R_cut ~ (c_disc / c_ball) ** (1.0 / (p_ball - p_disc))

which is the radius where the two asymptotic power laws would be equal.

All results are saved to a single npz file so a referee can reproduce any
plots or fits without rerunning the script.

Usage
-----
    python 51_uih_3d_fisher_disc_ball_scan.py

Optional arguments
------------------
    --nx, --ny, --nz        Grid dimensions (default 21 21 21)
    --m2                    Mass term m^2 (default 10.0)
    --gamma                 Laplacian prefactor (default 1.0)
    --disc_thickness        Disc thickness in lattice planes (default 3)
    --r_min, --r_max        Minimum and maximum integer radius to scan
                             (defaults chosen from grid size)
    --output                Output npz path
                             (default: uih_3d_fisher_disc_ball_scan.npz)

This script is deterministic and does not use any random numbers.
"""

import argparse
import math
import os
import sys
from typing import Dict, Tuple

import numpy as np


def apply_laplacian_3d(field: np.ndarray) -> np.ndarray:
    """
    Apply the standard 3D 6 neighbour discrete Laplacian with Dirichlet
    zero boundary conditions.

    L f(i,j,k) = 6 f(i,j,k) - sum over 6 nearest neighbours f

    Boundary condition: field is treated as zero outside the grid,
    implemented by padding with a layer of zeros.
    """
    if field.ndim != 3:
        raise ValueError("apply_laplacian_3d expects a 3D array")

    padded = np.pad(field, 1, mode="constant", constant_values=0.0)

    center = padded[1:-1, 1:-1, 1:-1]
    lap = (
        6.0 * center
        - padded[2:, 1:-1, 1:-1]
        - padded[:-2, 1:-1, 1:-1]
        - padded[1:-1, 2:, 1:-1]
        - padded[1:-1, :-2, 1:-1]
        - padded[1:-1, 1:-1, 2:]
        - padded[1:-1, 1:-1, :-2]
    )
    return lap


def apply_Q(field: np.ndarray, m2: float, gamma: float) -> np.ndarray:
    """
    Apply Q = m2 * I + gamma * L to a 3D field.
    """
    lap = apply_laplacian_3d(field)
    return m2 * field + gamma * lap


def build_disc_mask(
    nx: int, ny: int, nz: int, R: int, thickness: int
) -> np.ndarray:
    """
    Build a boolean mask for a thin disc centered in the grid.

    The disc is defined by
        radius in x y plane <= R
        and z between z0 - thickness_half and z0 + thickness_half inclusive
    """
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    cz = 0.5 * (nz - 1)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r2_xy = (X - cx) ** 2 + (Y - cy) ** 2
    disc_xy = r2_xy <= (float(R) ** 2)

    half_thick = max(0, thickness // 2)
    z_min = int(max(0, math.floor(cz - half_thick)))
    z_max = int(min(nz - 1, math.ceil(cz + half_thick)))

    z_mask = (Z >= z_min) & (Z <= z_max)

    disc_mask = disc_xy & z_mask
    return disc_mask


def build_ball_mask(nx: int, ny: int, nz: int, R: int) -> np.ndarray:
    """
    Build a boolean mask for a 3D ball centered in the grid.

    The ball is defined by
        radius in x y z <= R
    """
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    cz = 0.5 * (nz - 1)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    r2_xyz = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    ball_mask = r2_xyz <= (float(R) ** 2)
    return ball_mask


def compute_curvature_for_mask(
    mask: np.ndarray, m2: float, gamma: float
) -> Tuple[int, float, float]:
    """
    Given a boolean mask and parameters m2, gamma, build the indicator field
    a, apply Q, and compute

        I_sigma = a^T Q a

    Returns (N_sites, I_sigma, I_sigma_per_site).
    """
    a = mask.astype(float)
    n_sites = int(mask.sum())
    if n_sites == 0:
        return 0, 0.0, 0.0

    Qa = apply_Q(a, m2=m2, gamma=gamma)
    I_sigma = float(np.sum(a * Qa))
    I_per_site = I_sigma / float(n_sites)
    return n_sites, I_sigma, I_per_site


def fit_loglog_scaling(
    R_vals: np.ndarray,
    I_vals: np.ndarray,
    label: str,
) -> Tuple[float, float]:
    """
    Fit I_vals ~ c * R_vals^p in log log space.

    Returns (p, c) where p is the exponent and c the prefactor.
    """
    mask = (R_vals > 0) & (I_vals > 0)
    R = R_vals[mask].astype(float)
    I = I_vals[mask].astype(float)

    if R.size < 2:
        print(
            f"[UIH-3D-Fisher] Not enough points to fit scaling for {label}.",
            file=sys.stderr,
        )
        return float("nan"), float("nan")

    logR = np.log(R)
    logI = np.log(I)

    slope, intercept = np.polyfit(logR, logI, 1)
    p = float(slope)
    c = float(np.exp(intercept))

    print(
        f"[UIH-3D-Fisher] Scaling fit for {label}: I ~ {c:.6e} * R^{p:.3f}"
    )
    return p, c


def estimate_crossover_radius(
    p_disc: float,
    c_disc: float,
    p_ball: float,
    c_ball: float,
) -> float:
    """
    Solve c_disc * R^{p_disc} = c_ball * R^{p_ball} for R.

    Returns a rough crossover radius R_cut where disc like and ball like
    asymptotic power laws would be equal. This is only meaningful if
    p_ball > p_disc and both exponents and prefactors are finite.
    """
    if (
        not np.isfinite(p_disc)
        or not np.isfinite(p_ball)
        or not np.isfinite(c_disc)
        or not np.isfinite(c_ball)
    ):
        return float("nan")

    dp = p_ball - p_disc
    if dp <= 0:
        return float("nan")

    if c_ball <= 0 or c_disc <= 0:
        return float("nan")

    R_cut = (c_disc / c_ball) ** (1.0 / dp)
    return float(R_cut)


def run_scan(params: Dict) -> Dict:
    """
    Run the full disc and ball Fisher curvature scan for the given parameters.

    params is a dict with keys:
        nx, ny, nz, m2, gamma, disc_thickness, R_values
    """
    nx = int(params["nx"])
    ny = int(params["ny"])
    nz = int(params["nz"])
    m2 = float(params["m2"])
    gamma = float(params["gamma"])
    disc_thickness = int(params["disc_thickness"])
    R_values = np.array(params["R_values"], dtype=int)

    print("[UIH-3D-Fisher] Starting 3D Fisher disc ball scan")
    print(
        f"[UIH-3D-Fisher] Grid: {nx} x {ny} x {nz}, "
        f"m2 = {m2:.6g}, gamma = {gamma:.6g}, "
        f"disc_thickness = {disc_thickness}"
    )
    print(
        f"[UIH-3D-Fisher] Radii to scan: {R_values.tolist()}"
    )

    disc_counts = []
    disc_I = []
    disc_I_per_site = []

    ball_counts = []
    ball_I = []
    ball_I_per_site = []

    for R in R_values:
        print(
            f"[UIH-3D-Fisher] R = {R}: building masks and computing curvatures"
        )

        disc_mask = build_disc_mask(
            nx=nx, ny=ny, nz=nz, R=R, thickness=disc_thickness
        )
        n_disc, I_disc, I_disc_site = compute_curvature_for_mask(
            disc_mask, m2=m2, gamma=gamma
        )
        print(
            f"  Disc: N = {n_disc}, I = {I_disc:.6e}, "
            f"I_per_site = {I_disc_site:.6e}"
        )

        ball_mask = build_ball_mask(nx=nx, ny=ny, nz=nz, R=R)
        n_ball, I_ball, I_ball_site = compute_curvature_for_mask(
            ball_mask, m2=m2, gamma=gamma
        )
        print(
            f"  Ball: N = {n_ball}, I = {I_ball:.6e}, "
            f"I_per_site = {I_ball_site:.6e}"
        )

        disc_counts.append(n_disc)
        disc_I.append(I_disc)
        disc_I_per_site.append(I_disc_site)

        ball_counts.append(n_ball)
        ball_I.append(I_ball)
        ball_I_per_site.append(I_ball_site)

    disc_counts = np.array(disc_counts, dtype=int)
    disc_I = np.array(disc_I, dtype=float)
    disc_I_per_site = np.array(disc_I_per_site, dtype=float)

    ball_counts = np.array(ball_counts, dtype=int)
    ball_I = np.array(ball_I, dtype=float)
    ball_I_per_site = np.array(ball_I_per_site, dtype=float)

    print("[UIH-3D-Fisher] Performing log log scaling fits")
    p_disc, c_disc = fit_loglog_scaling(
        R_values.astype(float), disc_I, label="disc"
    )
    p_ball, c_ball = fit_loglog_scaling(
        R_values.astype(float), ball_I, label="ball"
    )

    R_cut = estimate_crossover_radius(
        p_disc=p_disc, c_disc=c_disc, p_ball=p_ball, c_ball=c_ball
    )
    if np.isfinite(R_cut):
        print(
            f"[UIH-3D-Fisher] Estimated crossover radius R_cut "
            f"from asymptotic fits: R_cut ≈ {R_cut:.3f}"
        )
    else:
        print(
            "[UIH-3D-Fisher] Could not estimate a meaningful crossover radius "
            "from the asymptotic fits."
        )

    results = dict(
        nx=nx,
        ny=ny,
        nz=nz,
        m2=m2,
        gamma=gamma,
        disc_thickness=disc_thickness,
        R_values=R_values,
        disc_counts=disc_counts,
        disc_I=disc_I,
        disc_I_per_site=disc_I_per_site,
        ball_counts=ball_counts,
        ball_I=ball_I,
        ball_I_per_site=ball_I_per_site,
        p_disc=p_disc,
        c_disc=c_disc,
        p_ball=p_ball,
        c_ball=c_ball,
        R_cut_estimate=R_cut,
    )
    return results


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D Fisher curvature scan for disc and ball modes."
    )
    parser.add_argument(
        "--nx", type=int, default=21, help="Grid size in x (default 21)"
    )
    parser.add_argument(
        "--ny", type=int, default=21, help="Grid size in y (default 21)"
    )
    parser.add_argument(
        "--nz", type=int, default=21, help="Grid size in z (default 21)"
    )
    parser.add_argument(
        "--m2", type=float, default=10.0, help="Mass term m^2 (default 10.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Laplacian prefactor gamma (default 1.0)",
    )
    parser.add_argument(
        "--disc_thickness",
        type=int,
        default=3,
        help="Disc thickness in lattice planes (default 3)",
    )
    parser.add_argument(
        "--r_min",
        type=int,
        default=None,
        help="Minimum integer radius to scan (default: chosen from grid size)",
    )
    parser.add_argument(
        "--r_max",
        type=int,
        default=None,
        help="Maximum integer radius to scan (default: chosen from grid size)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="uih_3d_fisher_disc_ball_scan.npz",
        help="Output npz file (default uih_3d_fisher_disc_ball_scan.npz)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    nx = args.nx
    ny = args.ny
    nz = args.nz

    # Choose default R range if not provided: stay away from boundaries.
    if args.r_min is None or args.r_max is None:
        # Centered disc and ball, leave at least 2 sites margin
        R_max_allowed = min(nx, ny, nz) // 2 - 2
        if R_max_allowed < 2:
            raise ValueError(
                "Grid too small to define a reasonable radius range."
            )
        r_min = 2
        r_max = R_max_allowed
    else:
        r_min = args.r_min
        r_max = args.r_max

    if r_min < 1 or r_max <= r_min:
        raise ValueError("Require 1 <= r_min < r_max.")

    R_values = np.arange(r_min, r_max + 1, dtype=int)

    params = dict(
        nx=nx,
        ny=ny,
        nz=nz,
        m2=args.m2,
        gamma=args.gamma,
        disc_thickness=args.disc_thickness,
        R_values=R_values,
    )

    results = run_scan(params)

    output_path = os.path.abspath(args.output)
    np.savez_compressed(output_path, **results)
    print(f"[UIH-3D-Fisher] Saved results to {output_path}")


if __name__ == "__main__":
    main()
