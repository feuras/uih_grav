#!/usr/bin/env python3
"""
10_rg_grav_3d_disc_ball_rIndex_bridge.py

RG-to-gravity bridge: hypocoercive r index for the three dimensional Fisher
precision operator Q = m2 * I + gamma * L used in the gravity paper.

For each radius R, we build disc and ball regions inside a cubic lattice,
restrict Q to those sites to obtain an SPD Fisher matrix A_R, then promote
A_R to a full finite dimensional UIH generator K_R = -A_R + J_R by drawing
random A_R-skew Hamiltonian parts J_R.

For each geometry and radius, we estimate:

  - the Fisher gap lambda_F(R) of A_R, from its smallest eigenvalue,
  - the hypocoercive index r_star(R) = lambda_hyp(R) / lambda_F(R), where
    lambda_hyp(R) is the smallest positive decay rate of K_R (largest
    negative real part of its spectrum).

Results are written to an .npz file with arrays of R, lambda_F, r_mean,
r_std and region sizes for discs and balls.
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


CONFIG = {
    "nx": 48,
    "ny": 48,
    "nz": 48,
    "m2": 10.0,
    "gamma": 1.0,
    "R_min": 2,
    "R_max": 7,
    "disc_half_thickness": 1,
    "n_J_samples": 4,
    "B_scale": 1.0,
    "seed": 12345,
    "output": "10_rg_grav_3d_disc_ball_rIndex_bridge.npz",
}


def build_laplacian_3d(nx: int, ny: int, nz: int) -> sp.csr_matrix:
    """
    Build the standard 6 neighbour discrete Laplacian L on a 3D grid with
    Dirichlet boundaries.

    L acts on a vector f of length nx * ny * nz and implements:

        (L f)(i,j,k) = sum of neighbour values minus count * f(i,j,k)

    so that L is negative semidefinite. We will use Q = m2 * I - gamma * L,
    which is SPD for m2 > 0.
    """
    N = nx * ny * nz
    rows = []
    cols = []
    data = []

    def idx(i: int, j: int, k: int) -> int:
        return i + nx * (j + ny * k)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                center = idx(i, j, k)
                diag = 0.0

                # left neighbour
                if i > 0:
                    rows.append(center)
                    cols.append(idx(i - 1, j, k))
                    data.append(1.0)
                    diag -= 1.0
                # right neighbour
                if i < nx - 1:
                    rows.append(center)
                    cols.append(idx(i + 1, j, k))
                    data.append(1.0)
                    diag -= 1.0
                # down neighbour
                if j > 0:
                    rows.append(center)
                    cols.append(idx(i, j - 1, k))
                    data.append(1.0)
                    diag -= 1.0
                # up neighbour
                if j < ny - 1:
                    rows.append(center)
                    cols.append(idx(i, j + 1, k))
                    data.append(1.0)
                    diag -= 1.0
                # back neighbour
                if k > 0:
                    rows.append(center)
                    cols.append(idx(i, j, k - 1))
                    data.append(1.0)
                    diag -= 1.0
                # front neighbour
                if k < nz - 1:
                    rows.append(center)
                    cols.append(idx(i, j, k + 1))
                    data.append(1.0)
                    diag -= 1.0

                # diagonal entry
                rows.append(center)
                cols.append(center)
                data.append(diag)

    L = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return L


def build_masks(
    nx: int,
    ny: int,
    nz: int,
    R_values: np.ndarray,
    disc_half_thickness: int,
):
    """
    Build boolean masks for discs and balls inside a 3D lattice.

    Coordinates are centred so that the origin sits at the middle of the
    domain in lattice units.
    """
    N = nx * ny * nz
    coords = np.zeros((N, 3), dtype=float)

    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    cz = 0.5 * (nz - 1)

    def idx(i: int, j: int, k: int) -> int:
        return i + nx * (j + ny * k)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                p = idx(i, j, k)
                coords[p, 0] = i - cx
                coords[p, 1] = j - cy
                coords[p, 2] = k - cz

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    masks_disc = {}
    masks_ball = {}

    for R in R_values:
        R2 = float(R) ** 2
        disc_mask = (x * x + y * y <= R2) & (np.abs(z) <= float(disc_half_thickness))
        ball_mask = x * x + y * y + z * z <= R2
        masks_disc[int(R)] = disc_mask
        masks_ball[int(R)] = ball_mask

    return masks_disc, masks_ball


def restrict_precision(Q: sp.csr_matrix, mask: np.ndarray) -> np.ndarray:
    """
    Restrict the global precision matrix Q to the subset of sites given by mask.

    Returns a dense SPD matrix A of shape (M, M) where M is the number of
    True entries in mask.
    """
    if Q.shape[0] != mask.size:
        raise ValueError("Mask length does not match matrix size")

    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise ValueError("Empty mask")

    Q_sub = Q[idx][:, idx]
    A = Q_sub.toarray()
    return A


def compute_r_index_for_A(
    A: np.ndarray,
    rng: np.random.Generator,
    n_J_samples: int,
    B_scale: float,
):
    """
    Given an SPD Fisher matrix A, estimate lambda_F and the r index by drawing
    random A-skew Hamiltonian parts J.

    We define G = -A and K = G + J = -A + J with J = A^{-1} B, where B is
    a random antisymmetric matrix. This ensures J^T A + A J = 0.

    For each random draw, we compute:

      - lambda_F as the smallest eigenvalue of A,
      - lambda_hyp as the smallest positive decay rate magnitude for K,
        obtained from the eigenvalues of K,
      - r = lambda_hyp / lambda_F.

    Returns lambda_F, r_mean, r_std.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    # Fisher gap from A
    evals_A = np.linalg.eigvalsh(A)
    lambda_F = float(np.min(evals_A).real)
    if lambda_F <= 0.0:
        raise RuntimeError(f"Non positive Fisher gap {lambda_F:.3e}")

    r_values = []

    for _ in range(n_J_samples):
        X = rng.standard_normal(size=(n, n))
        B = X - X.T
        if B_scale is not None:
            # Normalise Frobenius norm and rescale
            frob = np.linalg.norm(B, ord="fro")
            if frob > 0.0:
                B *= (B_scale / frob)

        # Solve A J = B for J, which gives J^T A + A J = 0
        J = np.linalg.solve(A, B)

        K = -A + J

        # Full spectrum of K
        evals_K = np.linalg.eigvals(K)
        real_parts = np.real(evals_K)

        # We expect a zero or near zero mode for conservation.
        # Take the largest negative real part as -lambda_hyp.
        mask_neg = real_parts < -1e-10
        if not np.any(mask_neg):
            # No clear negative mode; skip this sample
            continue

        max_neg = np.max(real_parts[mask_neg])
        lambda_hyp = float(-max_neg)

        r_values.append(lambda_hyp / lambda_F)

    if len(r_values) == 0:
        r_mean = np.nan
        r_std = np.nan
    else:
        r_values = np.array(r_values)
        r_mean = float(np.mean(r_values))
        r_std = float(np.std(r_values))

    return lambda_F, r_mean, r_std


def parse_args():
    parser = argparse.ArgumentParser(
        description="RG-to-gravity bridge: r index for 3D disc and ball Fisher precision."
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=CONFIG["nx"],
        help="Number of grid points in x.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=CONFIG["ny"],
        help="Number of grid points in y.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=CONFIG["nz"],
        help="Number of grid points in z.",
    )
    parser.add_argument(
        "--m2",
        type=float,
        default=CONFIG["m2"],
        help="Mass squared term in Q = m2 * I - gamma * L.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=CONFIG["gamma"],
        help="Laplacian coefficient in Q = m2 * I - gamma * L.",
    )
    parser.add_argument(
        "--R-min",
        type=int,
        default=CONFIG["R_min"],
        help="Minimum radius (lattice units).",
    )
    parser.add_argument(
        "--R-max",
        type=int,
        default=CONFIG["R_max"],
        help="Maximum radius (lattice units).",
    )
    parser.add_argument(
        "--disc-half-thickness",
        type=int,
        default=CONFIG["disc_half_thickness"],
        help="Half thickness of the disc in lattice layers.",
    )
    parser.add_argument(
        "--n-J-samples",
        type=int,
        default=CONFIG["n_J_samples"],
        help="Number of random J draws per region.",
    )
    parser.add_argument(
        "--B-scale",
        type=float,
        default=CONFIG["B_scale"],
        help="Scale factor for antisymmetric B used to construct J.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG["seed"],
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=CONFIG["output"],
        help="Output .npz file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    nx = args.nx
    ny = args.ny
    nz = args.nz

    R_values = np.arange(args.R_min, args.R_max + 1, dtype=int)

    print("=== RG to gravity bridge: r index on 3D disc and ball geometries ===")
    print(f"Grid: {nx} x {ny} x {nz}, R in [{R_values[0]}, {R_values[-1]}]")
    print(f"m2 = {args.m2:.3f}, gamma = {args.gamma:.3f}")
    print(f"Disc half thickness = {args.disc_half_thickness}")
    print(f"J samples per region = {args.n_J_samples}")
    print(f"B scale = {args.B_scale:.3f}")
    print(f"Random seed = {args.seed}")
    print(f"Output file: {args.output}")
    print("Building 3D Laplacian...")

    L = build_laplacian_3d(nx, ny, nz)
    N = nx * ny * nz

    print("Assembling precision Q = m2 * I - gamma * L...")
    m2 = float(args.m2)
    gamma = float(args.gamma)
    Q = m2 * sp.eye(N, format="csr") - gamma * L

    print("Building masks for discs and balls...")
    masks_disc, masks_ball = build_masks(
        nx,
        ny,
        nz,
        R_values,
        args.disc_half_thickness,
    )

    rng = np.random.default_rng(args.seed)

    results = {
        "config": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "m2": m2,
            "gamma": gamma,
            "R_min": int(R_values[0]),
            "R_max": int(R_values[-1]),
            "disc_half_thickness": int(args.disc_half_thickness),
            "n_J_samples": int(args.n_J_samples),
            "B_scale": float(args.B_scale),
            "seed": int(args.seed),
        }
    }

    R_disc_list = []
    N_disc_list = []
    lambdaF_disc_list = []
    rmean_disc_list = []
    rstd_disc_list = []

    R_ball_list = []
    N_ball_list = []
    lambdaF_ball_list = []
    rmean_ball_list = []
    rstd_ball_list = []

    t0 = time.time()

    for R in R_values:
        print(f"\nRadius R = {int(R)}")

        # Disc region
        mask_disc = masks_disc[int(R)]
        n_disc = int(np.count_nonzero(mask_disc))
        if n_disc > 0:
            print(f"  Disc region: {n_disc} sites")
            A_disc = restrict_precision(Q, mask_disc)
            lambdaF_disc, rmean_disc, rstd_disc = compute_r_index_for_A(
                A_disc,
                rng,
                args.n_J_samples,
                args.B_scale,
            )
            print(
                f"    Disc lambda_F = {lambdaF_disc:.4e}, "
                f"r_mean = {rmean_disc:.3f}, r_std = {rstd_disc:.3f}"
            )
            R_disc_list.append(int(R))
            N_disc_list.append(n_disc)
            lambdaF_disc_list.append(lambdaF_disc)
            rmean_disc_list.append(rmean_disc)
            rstd_disc_list.append(rstd_disc)
        else:
            print("  Disc region: empty, skipping")

        # Ball region
        mask_ball = masks_ball[int(R)]
        n_ball = int(np.count_nonzero(mask_ball))
        if n_ball > 0:
            print(f"  Ball region: {n_ball} sites")
            A_ball = restrict_precision(Q, mask_ball)
            lambdaF_ball, rmean_ball, rstd_ball = compute_r_index_for_A(
                A_ball,
                rng,
                args.n_J_samples,
                args.B_scale,
            )
            print(
                f"    Ball lambda_F = {lambdaF_ball:.4e}, "
                f"r_mean = {rmean_ball:.3f}, r_std = {rstd_ball:.3f}"
            )
            R_ball_list.append(int(R))
            N_ball_list.append(n_ball)
            lambdaF_ball_list.append(lambdaF_ball)
            rmean_ball_list.append(rmean_ball)
            rstd_ball_list.append(rstd_ball)
        else:
            print("  Ball region: empty, skipping")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.2f} s")

    results["R_disc"] = np.array(R_disc_list, dtype=int)
    results["N_disc"] = np.array(N_disc_list, dtype=int)
    results["lambdaF_disc"] = np.array(lambdaF_disc_list, dtype=float)
    results["rmean_disc"] = np.array(rmean_disc_list, dtype=float)
    results["rstd_disc"] = np.array(rstd_disc_list, dtype=float)

    results["R_ball"] = np.array(R_ball_list, dtype=int)
    results["N_ball"] = np.array(N_ball_list, dtype=int)
    results["lambdaF_ball"] = np.array(lambdaF_ball_list, dtype=float)
    results["rmean_ball"] = np.array(rmean_ball_list, dtype=float)
    results["rstd_ball"] = np.array(rstd_ball_list, dtype=float)

    # Save as npz. Store the config JSON inside as a byte string.
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.savez(
        out_path,
        config_json=json.dumps(results["config"]),
        R_disc=results["R_disc"],
        N_disc=results["N_disc"],
        lambdaF_disc=results["lambdaF_disc"],
        rmean_disc=results["rmean_disc"],
        rstd_disc=results["rstd_disc"],
        R_ball=results["R_ball"],
        N_ball=results["N_ball"],
        lambdaF_ball=results["lambdaF_ball"],
        rmean_ball=results["rmean_ball"],
        rstd_ball=results["rstd_ball"],
    )

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
