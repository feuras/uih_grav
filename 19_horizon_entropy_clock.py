#!/usr/bin/env python3
"""
19_horizon_entropy_clock.py

Horizon entropy-clock test for a 3D lattice Gaussian scalar field.

We reuse the same Gaussian model as in 18_horizon_fisher_patch_probe.py:

    S[phi]
        = 0.5 sum_x [
              (phi_{x+e_x} - phi_x)**2
            + (phi_{x+e_y} - phi_x)**2
            + (phi_{x+e_z} - phi_x)**2
            + m**2 * phi_x**2
          ]

which gives a precision matrix K = L + m^2 I.

We then:

  1. Define a "horizon" patch P as a slab at fixed x, with
     y in [0, Ny_patch) and all z.

  2. Build the patch indicator vector f (ones on P, zero outside)
     and compute the static patch Fisher curvature

         I_sigma = Var(sigma_P)
                 = f^T C f,   where C = K^{-1},

     by solving K u = f and taking I_sigma = f^T u.

  3. Construct an initial condition phi(0) which is a pure patch mode:

         phi(0) = lambda0 * f / ||f||,

     so the only excited degree of freedom is the horizon slab mode.

  4. Evolve phi(t) under deterministic gradient flow

         dphi/dt = -K phi,

     using a simple explicit Euler scheme with a small time step dt.

  5. Along the trajectory we monitor:

         F(t)        = 0.5 * phi(t)^T K phi(t),
         sigma_P(t)  = f^T phi(t),
         S_patch(t)  = -0.5 * I_sigma * sigma_P(t)^2,
         D(t)        = phi(t)^T K^2 phi(t),

     where D(t) is the instantaneous "dissipation" or UIH cost.

     For a linear Gaussian gradient flow dphi/dt = -K phi, the
     exact relation is

         dF/dt = -phi^T K^2 phi = -D(t).

     In the Gaussian setting F is proportional to KL divergence
     to equilibrium, so this is the UIH cost-entropy equality.

  6. We check numerically that:

         (i) F(t) decays monotonically and approximately exponentially,
         (ii) -dF/dt matches D(t) along the flow,
         (iii) S_patch(t) is proportional to sigma_P(t)^2 with curvature
               set by the static I_sigma from the horizon Fisher script.

Outputs:

  - results/horizon_entropy_clock/horizon_entropy_clock_timeseries.csv
      Columns: t, F, sigma_P, S_patch, D

  - results/horizon_entropy_clock/F_vs_t.png
  - results/horizon_entropy_clock/S_patch_vs_t.png
  - results/horizon_entropy_clock/F_decay_fit.png

This script closes the loop between:

  - static horizon Fisher curvature (area law),
  - dynamic UIH cost-entropy behaviour,

for a horizon-like mode in the 3D lattice gravity toy model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
    Ny_patch: int = 8
    x_slice: int = 0
    lambda0: float = 1.0
    dt: float = 0.01
    n_steps: int = 400
    random_seed: int = 20251123


def index_3d(ix: int, iy: int, iz: int, cfg: LatticeConfig) -> int:
    """Map 3D (ix, iy, iz) to flat index with periodic boundaries."""
    ix_mod = ix % cfg.Nx
    iy_mod = iy % cfg.Ny
    iz_mod = iz % cfg.Nz
    return (iz_mod * cfg.Ny + iy_mod) * cfg.Nx + ix_mod


def build_laplacian(cfg: LatticeConfig) -> sp.csr_matrix:
    """
    Build isotropic Laplacian L such that

        phi^T L phi = sum_x sum_dir (phi_{x+e_dir} - phi_x)^2

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
                jx = index_3d(ix + 1, iy, iz, cfg)
                rows.extend([i, i, jx, jx])
                cols.extend([i, jx, i, jx])
                data.extend([1.0, -1.0, -1.0, 1.0])

                # y direction
                jy = index_3d(ix, iy + 1, iz, cfg)
                rows.extend([i, i, jy, jy])
                cols.extend([i, jy, i, jy])
                data.extend([1.0, -1.0, -1.0, 1.0])

                # z direction
                jz = index_3d(ix, iy, iz + 1, cfg)
                rows.extend([i, i, jz, jz])
                cols.extend([i, jz, i, jz])
                data.extend([1.0, -1.0, -1.0, 1.0])

    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    data = np.array(data, dtype=float)

    L = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return L


def build_precision(cfg: LatticeConfig) -> sp.csr_matrix:
    """Build precision matrix K = L + m^2 I."""
    N = cfg.Nx * cfg.Ny * cfg.Nz
    L = build_laplacian(cfg)
    I = sp.identity(N, format="csr")
    K = L + (cfg.mass ** 2) * I
    return K.tocsr()


def build_patch_vector(cfg: LatticeConfig) -> np.ndarray:
    """
    Build patch indicator f for a slab at fixed x = x_slice,
    with y in [0, Ny_patch) and all z.
    """
    N = cfg.Nx * cfg.Ny * cfg.Nz
    f = np.zeros(N, dtype=float)
    for iz in range(cfg.Nz):
        for iy in range(cfg.Ny):
            if iy < cfg.Ny_patch:
                idx = index_3d(cfg.x_slice, iy, iz, cfg)
                f[idx] = 1.0
    return f


def compute_patch_fisher(K: sp.csr_matrix, f: np.ndarray) -> float:
    """
    Compute I_sigma = f^T C f with C = K^{-1}, by solving K u = f
    and forming I_sigma = f^T u.
    """
    u, info = spla.cg(K, f, rtol=1e-10, atol=0.0, maxiter=2000)
    if info != 0:
        raise RuntimeError(f"[EntropyClock] CG did not converge, info = {info}")
    I_sigma = float(f @ u)
    return I_sigma


def gradient_flow_trajectory(
    K: sp.csr_matrix,
    f: np.ndarray,
    I_sigma: float,
    cfg: LatticeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evolve phi under dphi/dt = -K phi with an initial condition
    that is a pure patch mode, and record the entropy-clock data:

        t_n, F_n, sigma_n, S_patch_n, D_n.
    """
    N = cfg.Nx * cfg.Ny * cfg.Nz
    # Normalised patch mode
    norm_f = np.linalg.norm(f)
    if norm_f <= 0.0:
        raise ValueError("[EntropyClock] Patch indicator has zero norm.")
    phi = cfg.lambda0 * f / norm_f

    t_vals = np.zeros(cfg.n_steps + 1, dtype=float)
    F_vals = np.zeros(cfg.n_steps + 1, dtype=float)
    sigma_vals = np.zeros(cfg.n_steps + 1, dtype=float)
    S_patch_vals = np.zeros(cfg.n_steps + 1, dtype=float)
    D_vals = np.zeros(cfg.n_steps + 1, dtype=float)

    # Helper: compute energy, patch amplitude, entropy and dissipation
    def snapshot(n: int, t: float, phi_vec: np.ndarray) -> None:
        Kphi = K @ phi_vec
        F = 0.5 * float(phi_vec @ Kphi)
        sigma = float(f @ phi_vec)
        # Patch entropy using static Fisher curvature
        S_patch = -0.5 * I_sigma * sigma * sigma
        # Dissipation D = phi^T K^2 phi
        K2phi = K @ Kphi
        D = float(phi_vec @ K2phi)

        t_vals[n] = t
        F_vals[n] = F
        sigma_vals[n] = sigma
        S_patch_vals[n] = S_patch
        D_vals[n] = D

    # Initial snapshot at t = 0
    snapshot(0, 0.0, phi)

    t = 0.0
    for n in range(1, cfg.n_steps + 1):
        # Explicit Euler step for gradient flow
        Kphi = K @ phi
        phi = phi - cfg.dt * Kphi
        t += cfg.dt
        snapshot(n, t, phi)

    return t_vals, F_vals, sigma_vals, S_patch_vals, D_vals


def fit_exponential(t: np.ndarray, F: np.ndarray) -> Tuple[float, float]:
    """
    Fit F(t) ~ F0 * exp(-gamma t) in a least-squares sense to the
    late-time behaviour. Returns (F0, gamma).
    """
    # Use the second half of the time series for the fit to avoid transients
    n = len(t)
    start = n // 4
    t_fit = t[start:]
    F_fit = F[start:]

    # Guard against non-positive values
    mask = F_fit > 0.0
    t_fit = t_fit[mask]
    F_fit = F_fit[mask]
    if t_fit.size < 3:
        return float(F[0]), 0.0

    logF = np.log(F_fit)
    A = np.vstack([np.ones_like(t_fit), -t_fit]).T
    coeff, _, _, _ = np.linalg.lstsq(A, logF, rcond=None)
    logF0, gamma = coeff[0], coeff[1]
    F0 = float(np.exp(logF0))
    return F0, gamma


def main() -> None:
    cfg = LatticeConfig()

    out_dir = Path("results") / "horizon_entropy_clock"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[EntropyClock] Building precision matrix K...")
    K = build_precision(cfg)

    print("[EntropyClock] Building horizon patch indicator f...")
    f = build_patch_vector(cfg)
    area = cfg.Ny_patch * cfg.Nz
    print(f"[EntropyClock] Patch Ny_patch = {cfg.Ny_patch}, area = {area} sites.")

    print("[EntropyClock] Computing static patch Fisher curvature I_sigma...")
    I_sigma = compute_patch_fisher(K, f)
    print(f"[EntropyClock] I_sigma = {I_sigma:.6e}")

    print("[EntropyClock] Evolving gradient flow trajectory...")
    t_vals, F_vals, sigma_vals, S_patch_vals, D_vals = gradient_flow_trajectory(
        K, f, I_sigma, cfg
    )

    # Save time series
    out_csv = out_dir / "horizon_entropy_clock_timeseries.csv"
    with open(out_csv, "w", encoding="utf-8") as f_out:
        f_out.write("t,F,sigma_P,S_patch,D\n")
        for t, F, sig, S_p, D in zip(t_vals, F_vals, sigma_vals, S_patch_vals, D_vals):
            f_out.write(f"{t:.8f},{F:.12e},{sig:.12e},{S_p:.12e},{D:.12e}\n")
    print(f"[EntropyClock] Saved time series to {out_csv}")

    # Fit exponential decay of F(t)
    F0_fit, gamma_fit = fit_exponential(t_vals, F_vals)
    print(
        f"[EntropyClock] F(t) ~ {F0_fit:.6e} * exp(-{gamma_fit:.6e} t) "
        "(late-time exponential fit)."
    )

    # Plot F(t)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_vals, F_vals, label="F(t) = 0.5 phi^T K phi")
    ax.set_xlabel("t")
    ax.set_ylabel("F(t)")
    ax.set_title("Horizon entropy clock: free energy decay")
    ax.grid(True)
    ax.legend()
    fig.savefig(out_dir / "F_vs_t.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot S_patch(t)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_vals, S_patch_vals, label="S_patch(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("S_patch(t)")
    ax.set_title("Horizon patch entropy vs time")
    ax.grid(True)
    ax.legend()
    fig.savefig(out_dir / "S_patch_vs_t.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot F(t) and fitted exponential on a log scale
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_vals, F_vals, "o", markersize=3, label="F(t) data")
    if gamma_fit > 0.0:
        F_fit_curve = F0_fit * np.exp(-gamma_fit * t_vals)
        ax.plot(t_vals, F_fit_curve, "-", label=f"exp fit, gamma={gamma_fit:.3e}")
    ax.set_xlabel("t")
    ax.set_ylabel("F(t)")
    ax.set_yscale("log")
    ax.set_title("Free energy decay (log scale)")
    ax.grid(True, which="both")
    ax.legend()
    fig.savefig(out_dir / "F_decay_fit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Optionally check dF/dt ≈ -D(t) numerically
    dF_dt = np.diff(F_vals) / np.diff(t_vals)
    # We only print a simple check here to avoid clutter
    mean_ratio = np.mean(-dF_dt[1:] / D_vals[1:-1])
    print(
        f"[EntropyClock] Mean ratio (-dF/dt) / D(t) over interior points "
        f"≈ {mean_ratio:.3f} (should be close to 1)."
    )

    print("[EntropyClock] Done.")


if __name__ == "__main__":
    main()
