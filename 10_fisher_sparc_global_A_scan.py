#!/usr/bin/env python3
"""
10_fisher_sparc_global_A_scan.py

Scan a global Fisher halo amplitude A_global across SPARC, allowing only
a per galaxy R_cut. For each A, we:

  - min over R_cut per galaxy
  - compute chi2_red per galaxy
  - count improvement vs baryon only

Outputs:
  - results/fisher_sparc/global_A_scan_summary.csv
"""

import csv
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

import numpy as np

DATA_DIR = Path("data") / "sparc_npz"
OUT_DIR = Path("results") / "fisher_sparc"


# Reuse the same UIH halo helpers as in 09_fish_halo_sparc.py

def compute_baryonic_mass_profile(R_kpc: np.ndarray, v_baryon_kms: np.ndarray) -> np.ndarray:
    return v_baryon_kms**2 * R_kpc


def compute_g_baryon(R_kpc: np.ndarray, v_baryon_kms: np.ndarray) -> np.ndarray:
    R_safe = np.maximum(R_kpc, 1.0e-6)
    return v_baryon_kms**2 / R_safe


def compute_g_base_halo(
    R_kpc: np.ndarray,
    M_b: np.ndarray,
    R_cut_kpc: float,
) -> np.ndarray:
    R_cut = max(float(R_cut_kpc), 1.0e-6)
    R = R_kpc

    x = R / R_cut
    alpha_shape = np.where(R <= R_cut, x**2, 1.0)

    R_sq_safe = np.maximum(R**2, 1.0e-10)
    f = alpha_shape * (M_b**2) / R_sq_safe

    I = np.zeros_like(R)
    for i in range(1, len(R)):
        dR = R[i] - R[i - 1]
        I[i] = I[i - 1] + 0.5 * (f[i] + f[i - 1]) * dR

    g_base = I / R_sq_safe
    return g_base


def chi2_from_g(
    A: float,
    R_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
    err_v_kms: np.ndarray,
    g_b: np.ndarray,
    g_base: np.ndarray,
) -> float:
    g_tot = g_b + A * g_base
    v_model = np.sqrt(np.maximum(g_tot * R_kpc, 0.0))
    sigma2 = np.maximum(err_v_kms, 1.0e-3) ** 2
    return float(np.sum((v_obs_kms - v_model) ** 2 / sigma2))


# Global A grid
LOGA_GRID = np.linspace(-6.0, -1.0, 26)
A_GRID = 10.0 ** LOGA_GRID


def analyse_galaxy_for_A_grid(gal_path: Path) -> Dict:
    try:
        data = np.load(gal_path)
    except Exception as e:
        return {"name": gal_path.stem, "status": f"error_loading_npz:{e}"}

    required_keys = ["R", "v_obs", "err_v_obs", "v_baryon"]
    if any(k not in data for k in required_keys):
        missing = [k for k in required_keys if k not in data]
        return {"name": gal_path.stem, "status": f"missing_keys:{','.join(missing)}"}

    R = np.asarray(data["R"], dtype=float)
    v_obs = np.asarray(data["v_obs"], dtype=float)
    err_v = np.asarray(data["err_v_obs"], dtype=float)
    v_baryon = np.asarray(data["v_baryon"], dtype=float)

    # Basic cleaning and sort
    mask = (
        np.isfinite(R)
        & np.isfinite(v_obs)
        & np.isfinite(err_v)
        & np.isfinite(v_baryon)
        & (R > 0.05)
    )
    R = R[mask]
    v_obs = v_obs[mask]
    err_v = err_v[mask]
    v_baryon = v_baryon[mask]

    if len(R) < 4:
        return {"name": gal_path.stem, "status": "too_few_points_after_cleaning"}

    idx = np.argsort(R)
    R = R[idx]
    v_obs = v_obs[idx]
    err_v = err_v[idx]
    v_baryon = v_baryon[idx]

    # Baryon chi2
    sigma2 = np.maximum(err_v, 1.0e-3) ** 2
    chi2_baryon = float(np.sum((v_obs - v_baryon) ** 2 / sigma2))

    # Precompute baryonic profiles
    g_b = compute_g_baryon(R, v_baryon)
    M_b = compute_baryonic_mass_profile(R, v_baryon)

    # R_cut grid per galaxy
    R_min = float(np.min(R))
    R_max = float(np.max(R))
    R_cut_grid = np.linspace(0.5 * R_min, 2.0 * R_max, 25)

    # Precompute g_base for each R_cut
    g_base_grid = []
    for R_cut in R_cut_grid:
        g_base_grid.append(compute_g_base_halo(R, M_b, R_cut))
    g_base_grid = np.stack(g_base_grid, axis=0)  # shape (nRcut, nR)

    # Vectorised chi2 over R_cut and A
    n_Rcut = g_base_grid.shape[0]
    n_A = A_GRID.size
    chi2_min_perA = np.empty(n_A, dtype=float)

    for j, A in enumerate(A_GRID):
        # g_tot for all R_cut at once: shape (nRcut, nR)
        g_tot = g_b[None, :] + A * g_base_grid
        v_model = np.sqrt(np.maximum(g_tot * R[None, :], 0.0))
        residual = v_obs[None, :] - v_model
        chi2_all = np.sum(residual**2 / sigma2[None, :], axis=1)  # per R_cut
        chi2_min_perA[j] = float(np.min(chi2_all))

    return {
        "name": gal_path.stem,
        "status": "ok",
        "chi2_baryon": chi2_baryon,
        "N_points": int(len(R)),
        "chi2_min_perA": chi2_min_perA,
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gal_files: List[Path] = sorted(DATA_DIR.glob("galaxy_*.npz"))
    if not gal_files:
        print(f"[ERROR] No galaxy_*.npz files found in {DATA_DIR}")
        raise SystemExit(1)

    print(f"[INFO] Found {len(gal_files)} galaxies for global A scan")

    cpu_count = mp.cpu_count()
    n_procs = min(22, cpu_count, len(gal_files))
    if n_procs < 1:
        n_procs = 1

    print(f"[INFO] Using {n_procs} process(es) (cpu_count={cpu_count})")

    results: List[Dict] = []
    if n_procs == 1:
        for gf in gal_files:
            r = analyse_galaxy_for_A_grid(gf)
            results.append(r)
            print(f"[GAL] {r.get('name')}: {r.get('status')}")
    else:
        with mp.Pool(processes=n_procs) as pool:
            for r in pool.imap_unordered(analyse_galaxy_for_A_grid, gal_files, chunksize=1):
                results.append(r)
                print(f"[GAL] {r.get('name')}: {r.get('status')}")

    # Filter ok galaxies
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        print("[ERROR] No ok galaxies in global A scan results")
        raise SystemExit(1)

    n_gal = len(ok_results)
    print(f"[INFO] Global A scan uses {n_gal} galaxies")

    # Build arrays
    chi2_baryon = np.array([r["chi2_baryon"] for r in ok_results], dtype=float)
    Npts = np.array([r["N_points"] for r in ok_results], dtype=float)
    chi2_min_perA_all = np.stack([r["chi2_min_perA"] for r in ok_results], axis=0)  # shape (n_gal, n_A)

    dof_baryon = np.maximum(Npts, 1.0)
    chi2_red_baryon = chi2_baryon / dof_baryon

    # Per A statistics
    out_rows = []
    for j, logA in enumerate(LOGA_GRID):
        A = A_GRID[j]
        chi2_A = chi2_min_perA_all[:, j]
        dof_fisher = np.maximum(Npts - 1.0, 1.0)  # one extra parameter per galaxy (R_cut)
        chi2_red_f = chi2_A / dof_fisher

        # Improvement factors
        improvement_ratio = chi2_red_baryon / chi2_red_f
        n_improve_10 = int(np.sum(improvement_ratio > 1.10))
        n_degrade_10 = int(np.sum(improvement_ratio < 0.90))

        # Robust stats
        def stats(x):
            x = x[np.isfinite(x)]
            if x.size == 0:
                return np.nan, np.nan, np.nan
            return float(np.min(x)), float(np.median(x)), float(np.max(x))

        c2f_min, c2f_med, c2f_max = stats(chi2_red_f)
        c2b_min, c2b_med, c2b_max = stats(chi2_red_baryon)

        print(
            f"[A_SCAN] log10A={logA:+5.2f} A={A:.3e} "
            f"chi2_red_F median={c2f_med:.3f} "
            f"improve>10percent={n_improve_10} degrade>10percent={n_degrade_10}"
        )

        out_rows.append({
            "log10A": logA,
            "A": A,
            "chi2_red_F_min": c2f_min,
            "chi2_red_F_median": c2f_med,
            "chi2_red_F_max": c2f_max,
            "chi2_red_B_min": c2b_min,
            "chi2_red_B_median": c2b_med,
            "chi2_red_B_max": c2b_max,
            "N_improve_10percent": n_improve_10,
            "N_degrade_10percent": n_degrade_10,
        })

    # Save summary CSV
    out_csv = OUT_DIR / "global_A_scan_summary.csv"
    with open(out_csv, "w", newline="") as f:
        fieldnames = list(out_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[INFO] Saved global A scan summary to {out_csv}")
