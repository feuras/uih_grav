#!/usr/bin/env python3
"""
18_horizon_fisher_patch_probe.py

Horizon like Fisher curvature probe for a 3D lattice Gaussian scalar field.

We consider a quadratic action on an Nx x Ny x Nz periodic lattice,

    S[phi]
        = 0.5 sum_x [
              (phi_{x+e_x} - phi_x)**2
            + (phi_{x+e_y} - phi_x)**2
            + (phi_{x+e_z} - phi_x)**2
            + m**2 * phi_x**2
          ]

This gives a precision matrix K. For a patch P which is a rectangular
slab in the y z directions at fixed x, define a patch mode

    sigma = sum_{i in P} phi_i,

and treat sigma as a parameter of the distribution. For a zero mean
Gaussian, the Fisher information for sigma is

    I_sigma = Var(sigma) = f^T C f

where f_i = 1 if i in P and 0 otherwise, and C = K^{-1}.

We compute I_sigma for a family of patches with different areas and
check how it scales with the patch area. The expectation from the gravity
work is an area law for Fisher curvature for such "horizon" modes.

This script uses conjugate gradient solves and can use multiple workers
to scan several patch sizes in parallel. It produces a CSV table and
a simple log log fit of I_sigma versus area.
"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


@dataclass
class LatticeConfig:
    Nx: int = 16
    Ny: int = 16
    Nz: int = 16
    mass: float = 0.5
    cg_rtol: float = 1e-8
    cg_maxiter: int = 1000
    n_workers: int = 8
    random_seed: int = 20251123


def index_3d(ix: int, iy: int, iz: int, cfg: LatticeConfig) -> int:
    """Map 3D indices to a flat index with periodic boundaries."""
    ix_mod = ix % cfg.Nx
    iy_mod = iy % cfg.Ny
    iz_mod = iz % cfg.Nz
    return (iz_mod * cfg.Ny + iy_mod) * cfg.Nx + ix_mod


def build_laplacian(cfg: LatticeConfig) -> sp.csr_matrix:
    """
    Build the isotropic Laplacian matrix L such that

        phi^T L phi = sum_x sum_{dir} (phi_{x+e_dir} - phi_x)^2

    on a periodic 3D lattice.
    """
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    N = Nx * Ny * Nz

    rows = []
    cols = []
    data = []

    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                i = index_3d(ix, iy, iz, cfg)

                # x direction
                jx_plus = index_3d(ix + 1, iy, iz, cfg)
                rows.extend([i, i, jx_plus, jx_plus])
                cols.extend([i, jx_plus, i, jx_plus])
                data.extend([1.0, -1.0, -1.0, 1.0])

                # y direction
                jy_plus = index_3d(ix, iy + 1, iz, cfg)
                rows.extend([i, i, jy_plus, jy_plus])
                cols.extend([i, jy_plus, i, jy_plus])
                data.extend([1.0, -1.0, -1.0, 1.0])

                # z direction
                jz_plus = index_3d(ix, iy, iz + 1, cfg)
                rows.extend([i, i, jz_plus, jz_plus])
                cols.extend([i, jz_plus, i, jz_plus])
                data.extend([1.0, -1.0, -1.0, 1.0])

    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    data = np.array(data, dtype=float)

    L = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return L


def build_precision(cfg: LatticeConfig) -> sp.csr_matrix:
    """
    Build the precision matrix K = L + m^2 I for the Gaussian field.
    """
    N = cfg.Nx * cfg.Ny * cfg.Nz
    L = build_laplacian(cfg)
    I = sp.identity(N, format="csr")
    K = L + (cfg.mass ** 2) * I
    return K.tocsr()


def cg_solve(K: sp.csr_matrix, b: np.ndarray, cfg: LatticeConfig) -> np.ndarray:
    """Solve K x = b using conjugate gradient."""
    x, info = spla.cg(K, b, rtol=cfg.cg_rtol, atol=0.0, maxiter=cfg.cg_maxiter)
    if info != 0:
        raise RuntimeError(f"CG did not converge, info = {info}")
    return x


def make_patch_vector(cfg: LatticeConfig, Ny_patch: int, x_slice: int = 0) -> np.ndarray:
    """
    Build a patch indicator vector f for a patch that spans a slice at fixed x,
    with y in [0, Ny_patch), and all z.

    Area = Ny_patch * Nz.
    """
    N = cfg.Nx * cfg.Ny * cfg.Nz
    f = np.zeros(N, dtype=float)
    for iz in range(cfg.Nz):
        for iy in range(Ny_patch):
            i = index_3d(x_slice, iy, iz, cfg)
            f[i] = 1.0
    return f


def patch_worker(args) -> Tuple[int, float]:
    """
    Worker that computes I_sigma for a given patch size Ny_patch.

    It solves K x = f and returns I = f^T x and the area.
    """
    K, cfg, Ny_patch = args
    f = make_patch_vector(cfg, Ny_patch=Ny_patch, x_slice=0)
    area = Ny_patch * cfg.Nz
    x = cg_solve(K, f, cfg)
    I_sigma = float(f @ x)
    return area, I_sigma


def run_patch_scan(cfg: LatticeConfig, K: sp.csr_matrix, Ny_patch_list: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute I_sigma for a list of Ny_patch values using multiple workers.
    """
    n_workers = min(cfg.n_workers, mp.cpu_count())
    if n_workers < 1:
        n_workers = 1

    args_list = [(K, cfg, Ny_p) for Ny_p in Ny_patch_list]

    print(f"[Horizon Fisher] Using {n_workers} workers for {len(args_list)} patches.")

    if n_workers == 1:
        results = [patch_worker(args) for args in args_list]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(patch_worker, args_list)

    areas = np.array([r[0] for r in results], dtype=float)
    I_vals = np.array([r[1] for r in results], dtype=float)

    # Sort by area
    order = np.argsort(areas)
    return areas[order], I_vals[order]


def main() -> None:
    cfg = LatticeConfig(
        Nx=16,
        Ny=16,
        Nz=16,
        mass=0.5,
        cg_rtol=1e-8,
        cg_maxiter=1000,
        n_workers=8,
        random_seed=20251123,
    )

    out_dir = Path("results") / "horizon_fisher"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Horizon Fisher] Building precision matrix K...")
    K = build_precision(cfg)

    # Choose a set of patch sizes that fit inside Ny
    Ny_patch_list = [2, 4, 6, 8, 10, 12, 14, 16]
    Ny_patch_list = [Ny for Ny in Ny_patch_list if Ny <= cfg.Ny]

    print("[Horizon Fisher] Scanning patch sizes Ny_patch =", Ny_patch_list)
    areas, I_vals = run_patch_scan(cfg, K, Ny_patch_list)

    # Save table
    out_csv = out_dir / "horizon_fisher_patch_scan.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Ny_patch,area,I_sigma\n")
        for Ny_p, area, I_sig in zip(Ny_patch_list, areas, I_vals):
            f.write(f"{Ny_p},{area:.6f},{I_sig:.12e}\n")
    print(f"[Horizon Fisher] Saved patch scan table to {out_csv}")

    # Fit log I = a + b log area
    logA = np.log(areas)
    logI = np.log(I_vals)
    A_mat = np.vstack([np.ones_like(logA), logA]).T
    coeff, _, _, _ = np.linalg.lstsq(A_mat, logI, rcond=None)
    a_fit, b_fit = float(coeff[0]), float(coeff[1])

    print(f"[Horizon Fisher] log-log fit: log I_sigma ≈ {a_fit:.6f} + {b_fit:.6f} log(area)")
    print(f"[Horizon Fisher] Exponent b_fit ≈ {b_fit:.6f} (area law would have b_fit ≈ 1.0)")

    # Make simple plots
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(areas, I_vals, "o-", label="I_sigma")
    ax.set_xlabel("Patch area (number of sites)")
    ax.set_ylabel("Fisher curvature I_sigma")
    ax.set_title("Horizon patch Fisher curvature vs area")
    ax.grid(True)
    ax.legend()
    fig.savefig(out_dir / "horizon_fisher_vs_area.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(logA, logI, "o", label="data")
    ax.plot(logA, a_fit + b_fit * logA, "-", label=f"fit, slope={b_fit:.3f}")
    ax.set_xlabel("log(area)")
    ax.set_ylabel("log(I_sigma)")
    ax.set_title("Horizon patch Fisher curvature log-log scaling")
    ax.grid(True)
    ax.legend()
    fig.savefig(out_dir / "horizon_fisher_loglog.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save summary
    np.savez(
        out_dir / "horizon_fisher_patch_results.npz",
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        Nz=cfg.Nz,
        mass=cfg.mass,
        areas=areas,
        I_vals=I_vals,
        a_fit=a_fit,
        b_fit=b_fit,
    )

    print("[Horizon Fisher] Done.")


if __name__ == "__main__":
    main()
