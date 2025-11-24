#!/usr/bin/env python3
"""
60_metric_fisher_superspace_probe.py

Estimate the DeWitt trace coefficient beta in the ultralocal Fisher metric
on metric perturbations, using a 3D lattice Gaussian scalar field model.

We consider a quadratic action on an Nx x Ny x Nz periodic lattice,

    S[phi; theta_trace, theta_TT]
        = 0.5 sum_x [
              c_x (phi_{x+e_x} - phi_x)^2
            + c_y (phi_{x+e_y} - phi_x)^2
            + c_z (phi_{x+e_z} - phi_x)^2
            + m^2 phi_x^2
          ]

with directional couplings

    c_x = exp(2 theta_trace + 2 theta_TT)
    c_y = exp(2 theta_trace - theta_TT)
    c_z = exp(2 theta_trace - theta_TT).

At theta_trace = theta_TT = 0 we have c_x = c_y = c_z = 1.

The precision matrix is K(theta), and for a zero-mean Gaussian with
p(phi | theta) ∝ exp(-0.5 phi^T K(theta) phi), the Fisher information
matrix in parameter space is

    F_ab = 0.5 Tr( C ∂_a K C ∂_b K ),

with C = K^-1.

We estimate F_ab at theta = 0 using Hutchinson's trace estimator with
Rademacher vectors and conjugate-gradient solves.

From F_trace_trace and F_TT_TT we reconstruct the DeWitt trace coefficient
beta using the relation (in d = 3)

    r = G_TT_TT / G_trace_trace,
    r = 2 / (1 + 3 beta),
    beta = (2 - r) / (3 r).

This assumes that the parameter directions theta_trace and theta_TT map
to pure trace and pure traceless metric perturbations in the continuum
limit, which is the natural choice for these anisotropic couplings.
"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class LatticeConfig:
    Nx: int = 16
    Ny: int = 16
    Nz: int = 16
    mass: float = 0.5
    n_hutchinson: int = 256
    cg_tol: float = 1e-8
    cg_maxiter: int = 2000
    n_workers: int = 22
    random_seed: int = 12345


def index_3d(ix: int, iy: int, iz: int, cfg: LatticeConfig) -> int:
    """Map 3D indices to a flat index."""
    ix_mod = ix % cfg.Nx
    iy_mod = iy % cfg.Ny
    iz_mod = iz % cfg.Nz
    return (iz_mod * cfg.Ny + iy_mod) * cfg.Nx + ix_mod


def build_directional_laplacians(cfg: LatticeConfig) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """
    Build sparse matrices Lx, Ly, Lz for a periodic 3D lattice, such that

        phi^T Lx phi = sum_x (phi_{x+e_x} - phi_x)^2

    and similarly for Ly, Lz.
    """
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    N = Nx * Ny * Nz

    # Separate index and data arrays for each direction
    rows_x, cols_x, data_x = [], [], []
    rows_y, cols_y, data_y = [], [], []
    rows_z, cols_z, data_z = [], [], []

    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                i = index_3d(ix, iy, iz, cfg)

                # x-direction: (phi_{x+e_x} - phi_x)^2
                jx_plus = index_3d(ix + 1, iy, iz, cfg)
                rows_x.extend([i, i, jx_plus, jx_plus])
                cols_x.extend([i, jx_plus, i, jx_plus])
                data_x.extend([1.0, -1.0, -1.0, 1.0])

                # y-direction
                jy_plus = index_3d(ix, iy + 1, iz, cfg)
                rows_y.extend([i, i, jy_plus, jy_plus])
                cols_y.extend([i, jy_plus, i, jy_plus])
                data_y.extend([1.0, -1.0, -1.0, 1.0])

                # z-direction
                jz_plus = index_3d(ix, iy, iz + 1, cfg)
                rows_z.extend([i, i, jz_plus, jz_plus])
                cols_z.extend([i, jz_plus, i, jz_plus])
                data_z.extend([1.0, -1.0, -1.0, 1.0])

    # Convert to arrays
    rows_x = np.array(rows_x, dtype=int)
    cols_x = np.array(cols_x, dtype=int)
    data_x = np.array(data_x, dtype=float)

    rows_y = np.array(rows_y, dtype=int)
    cols_y = np.array(cols_y, dtype=int)
    data_y = np.array(data_y, dtype=float)

    rows_z = np.array(rows_z, dtype=int)
    cols_z = np.array(cols_z, dtype=int)
    data_z = np.array(data_z, dtype=float)

    # Build CSR matrices
    Lx = sp.coo_matrix((data_x, (rows_x, cols_x)), shape=(N, N)).tocsr()
    Ly = sp.coo_matrix((data_y, (rows_y, cols_y)), shape=(N, N)).tocsr()
    Lz = sp.coo_matrix((data_z, (rows_z, cols_z)), shape=(N, N)).tocsr()

    return Lx, Ly, Lz

def build_precision_and_derivatives(cfg: LatticeConfig) -> Tuple[sp.csr_array, sp.csr_array, sp.csr_array]:
    """
    Build K, dK_dtheta_trace, dK_dtheta_TT at theta_trace = theta_TT = 0.

    At theta = 0 we have c_x = c_y = c_z = 1, so

        K = Lx + Ly + Lz + m^2 I.

    Using c_x = exp(2 theta_trace + 2 theta_TT),
           c_y = exp(2 theta_trace - theta_TT),
           c_z = exp(2 theta_trace - theta_TT),

    the derivatives at theta = 0 are

        ∂ c_x / ∂ theta_trace = 2,   ∂ c_x / ∂ theta_TT = 2
        ∂ c_y / ∂ theta_trace = 2,   ∂ c_y / ∂ theta_TT = -1
        ∂ c_z / ∂ theta_trace = 2,   ∂ c_z / ∂ theta_TT = -1.

    Therefore

        dK_dtheta_trace = 2 (Lx + Ly + Lz),
        dK_dtheta_TT    = 2 Lx - Ly - Lz.
    """
    Lx, Ly, Lz = build_directional_laplacians(cfg)
    N = cfg.Nx * cfg.Ny * cfg.Nz

    I = sp.identity(N, format="csr")
    K = Lx + Ly + Lz + (cfg.mass ** 2) * I

    dK_trace = 2.0 * (Lx + Ly + Lz)
    dK_TT = 2.0 * Lx - 1.0 * Ly - 1.0 * Lz

    return K.tocsr(), dK_trace.tocsr(), dK_TT.tocsr()


def cg_solve(K: sp.csr_matrix, b: np.ndarray, cfg: LatticeConfig) -> np.ndarray:
    """
    Solve K x = b using conjugate gradient. K should be symmetric positive definite.
    SciPy >= 1.14 uses (rtol, atol) instead of tol.
    """
    x, info = spla.cg(
        K,
        b,
        rtol=cfg.cg_tol,
        atol=0.0,
        maxiter=cfg.cg_maxiter,
    )
    if info != 0:
        raise RuntimeError(f"CG did not converge, info = {info}")
    return x



def hutchinson_worker(args) -> np.ndarray:
    """
    Worker for Hutchinson trace estimation.

    For one batch of random vectors, compute the partial contributions to
    the Fisher matrix

        F_ab = 0.5 Tr(C dK_a C dK_b)
             ≈ 0.5 E_z[ z^T C dK_a C dK_b z ].

    We compute for a = trace, TT; likewise for b.
    """
    K, dK_trace, dK_TT, cfg, n_vectors, seed = args
    rng = np.random.default_rng(seed)

    # Parameters are ordered: 0 = trace, 1 = TT
    partial_F = np.zeros((2, 2), dtype=float)

    for _ in range(n_vectors):
        # Hutchinson vector with entries ±1
        z = rng.choice([-1.0, 1.0], size=K.shape[0])

        # u = C z
        u = cg_solve(K, z, cfg)

        # For each parameter a, compute y_a = C (dK_a u)
        # y_trace, y_TT
        tmp_trace = dK_trace.dot(u)
        y_trace = cg_solve(K, tmp_trace, cfg)

        tmp_TT = dK_TT.dot(u)
        y_TT = cg_solve(K, tmp_TT, cfg)

        # Now approximate contributions to F_ab
        # z^T C dK_a C dK_b z = z^T dK_b y_a
        v_trace = dK_trace.dot(y_trace)
        v_TT = dK_TT.dot(y_TT)
        v_trace_TT = dK_TT.dot(y_trace)
        v_TT_trace = dK_trace.dot(y_TT)

        # F_trace_trace
        partial_F[0, 0] += 0.5 * float(z @ v_trace)
        # F_TT_TT
        partial_F[1, 1] += 0.5 * float(z @ v_TT)
        # Off diagonal terms (symmetric)
        partial_F[0, 1] += 0.25 * (float(z @ v_trace_TT) + float(z @ v_TT_trace))
        partial_F[1, 0] = partial_F[0, 1]

    return partial_F


def estimate_fisher_matrix(cfg: LatticeConfig, K: sp.csr_array, dK_trace: sp.csr_array, dK_TT: sp.csr_array) -> np.ndarray:
    """
    Estimate the 2x2 Fisher matrix using Hutchinson with n_hutchinson vectors,
    distributed over cfg.n_workers workers.
    """
    n_total = cfg.n_hutchinson
    n_workers = min(cfg.n_workers, mp.cpu_count())
    if n_workers < 1:
        n_workers = 1

    # Split number of vectors roughly evenly
    counts = [n_total // n_workers] * n_workers
    for i in range(n_total % n_workers):
        counts[i] += 1

    seeds = np.random.SeedSequence(cfg.random_seed).spawn(n_workers)

    args_list = []
    for i in range(n_workers):
        if counts[i] == 0:
            continue
        args_list.append((K, dK_trace, dK_TT, cfg, counts[i], int(seeds[i].generate_state(1)[0])))

    if not args_list:
        raise RuntimeError("No Hutchinson vectors requested.")

    print(f"[Metric Fisher] Using {len(args_list)} workers over {n_total} Hutchinson vectors.")

    with mp.Pool(processes=len(args_list)) as pool:
        results = pool.map(hutchinson_worker, args_list)

    F = np.zeros((2, 2), dtype=float)
    for partial in results:
        F += partial

    # Average over total number of vectors
    F /= float(n_total)

    return F


def compute_beta_from_fisher(F: np.ndarray) -> float:
    """
    Given the 2x2 Fisher matrix F_ab in the parameter directions
    (theta_trace, theta_TT), estimate the DeWitt-style trace coefficient beta.

    In the continuum ultralocal Fisher metric

        G(h,h) = c ∫ (h_ij h^ij + beta (tr h)^2) sqrt(g) dx,

    for canonical trace/TT modes one finds

        r_canon = G_TT_TT / G_trace_trace = 2 / (1 + 3 beta).

    Our parameter directions correspond to

        h_trace_param  = 4 * h_trace_canon
        h_TT_param     = -1 * h_TT_canon,

    so

        r_canon = 16 * (F_TT_TT / F_trace_trace).

    Therefore

        beta = (2 - r_canon) / (3 * r_canon).
    """
    F_trace_trace = F[0, 0]
    F_TT_TT = F[1, 1]
    if F_trace_trace <= 0 or F_TT_TT <= 0:
        raise RuntimeError("Non positive Fisher diagonal elements, cannot estimate beta.")

    r_param = F_TT_TT / F_trace_trace
    r_canon = 16.0 * r_param
    beta = (2.0 - r_canon) / (3.0 * r_canon)
    return beta



def main() -> None:
    cfg = LatticeConfig(
        Nx=16,
        Ny=16,
        Nz=16,
        mass=0.5,
        n_hutchinson=256,
        cg_tol=1e-8,
        cg_maxiter=1000,
        n_workers=22,
        random_seed=12345,
    )

    out_dir = Path("results") / "metric_fisher"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Metric Fisher] Building precision matrix and derivatives...")
    K, dK_trace, dK_TT = build_precision_and_derivatives(cfg)

    print("[Metric Fisher] Estimating Fisher matrix F_ab...")
    F = estimate_fisher_matrix(cfg, K, dK_trace, dK_TT)

    print("[Metric Fisher] Fisher matrix F_ab (parameters: trace, TT):")
    print(F)

    beta_est = compute_beta_from_fisher(F)
    print(f"[Metric Fisher] Estimated DeWitt beta ≈ {beta_est:.6f}")

    # Save to disk for inspection
    np.savez(out_dir / "metric_fisher_results.npz",
             F=F,
             beta_est=beta_est,
             Nx=cfg.Nx,
             Ny=cfg.Ny,
             Nz=cfg.Nz,
             mass=cfg.mass,
             n_hutchinson=cfg.n_hutchinson,
             cg_tol=cfg.cg_tol,
             cg_maxiter=cfg.cg_maxiter,
             n_workers=cfg.n_workers,
             random_seed=cfg.random_seed,
             )


if __name__ == "__main__":
    main()
"""
Expect:
[Metric Fisher] Building precision matrix and derivatives...
[Metric Fisher] Estimating Fisher matrix F_ab...
[Metric Fisher] Using 22 workers over 256 Hutchinson vectors.
[Metric Fisher] Fisher matrix F_ab (parameters: trace, TT):
[[7.37093775e+03 1.25607053e+00]
 [1.25607053e+00 8.64146630e+02]]
[Metric Fisher] Estimated DeWitt beta ≈ 0.022072
"""