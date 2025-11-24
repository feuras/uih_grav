#!/usr/bin/env python3
"""
14_fisher_sparc_residual_diagnostics.py

Residual diagnostics for the UIH Fisher halo fits.

For each galaxy with status ok in fisher_sparc_summary.csv we:

  - Reconstruct the model velocities using A and R_cut.
  - Compute residuals (v_obs - v_model) / err_v.
  - Collect residuals over all galaxies.
  - Bin residuals by radius fraction R / R_max_kpc.

Prints global mean and variance of residuals and radial trends.
"""

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path("data") / "sparc_npz"
SUM_CSV = Path("results") / "fisher_sparc" / "fisher_sparc_summary.csv"


def load_summary(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


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


if __name__ == "__main__":
    if not SUM_CSV.exists():
        print(f"[ERROR] fisher_sparc_summary.csv not found at {SUM_CSV}")
        raise SystemExit(1)

    summary_rows = load_summary(SUM_CSV)
    residuals_all = []
    radii_frac_all = []

    n_ok = 0

    for row in summary_rows:
        if row["status"] != "ok":
            continue

        name = row["name"]
        A = float(row["A"]) if row["A"] not in ("", "None") else np.nan
        R_cut = float(row["R_cut_kpc"]) if row["R_cut_kpc"] not in ("", "None") else np.nan

        if not np.isfinite(A) or not np.isfinite(R_cut):
            continue

        gal_path = DATA_DIR / f"galaxy_{name}.npz"
        if not gal_path.exists():
            print(f"[WARN] Missing npz for galaxy {name} at {gal_path}")
            continue

        data = np.load(gal_path)
        R = np.asarray(data["R"], dtype=float)
        v_obs = np.asarray(data["v_obs"], dtype=float)
        err_v = np.asarray(data["err_v_obs"], dtype=float)
        v_bar = np.asarray(data["v_baryon"], dtype=float)

        mask = (
            np.isfinite(R)
            & np.isfinite(v_obs)
            & np.isfinite(err_v)
            & np.isfinite(v_bar)
            & (R > 0.05)
        )
        R = R[mask]
        v_obs = v_obs[mask]
        err_v = err_v[mask]
        v_bar = v_bar[mask]

        if R.size < 4:
            continue

        idx = np.argsort(R)
        R = R[idx]
        v_obs = v_obs[idx]
        err_v = err_v[idx]
        v_bar = v_bar[idx]

        # Rebuild Fisher model
        g_b = compute_g_baryon(R, v_bar)
        M_b = compute_baryonic_mass_profile(R, v_bar)
        g_base = compute_g_base_halo(R, M_b, R_cut)
        g_tot = g_b + A * g_base
        v_model = np.sqrt(np.maximum(g_tot * R, 0.0))

        res = (v_obs - v_model) / np.maximum(err_v, 1.0e-3)
        residuals_all.append(res)

        R_frac = R / float(np.max(R))
        radii_frac_all.append(R_frac)

        n_ok += 1

    if not residuals_all:
        print("[ERROR] No residuals collected")
        raise SystemExit(1)

    residuals_all = np.concatenate(residuals_all)
    radii_frac_all = np.concatenate(radii_frac_all)

    print(f"[INFO] Collected {residuals_all.size} residuals from {n_ok} galaxies")

    # Global stats
    mean_res = float(np.mean(residuals_all))
    std_res = float(np.std(residuals_all))
    print(f"[RESID] Global residual mean = {mean_res:.3f}, std = {std_res:.3f}")

    # Simple histogram summary
    for thr in [1.0, 2.0, 3.0]:
        frac = float(np.mean(np.abs(residuals_all) <= thr))
        print(f"[RESID] Fraction of points with |residual| <= {thr:.1f}: {frac:.3f}")

    # Radial trend: bin in R/R_max
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    print("[RESID] Radial residual stats (R / R_max bins):")
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        mask_bin = (radii_frac_all >= lo) & (radii_frac_all < hi)
        if np.sum(mask_bin) < 10:
            print(f"  bin {lo:.1f}-{hi:.1f}: N < 10, skipping")
            continue
        res_bin = residuals_all[mask_bin]
        mean_bin = float(np.mean(res_bin))
        std_bin = float(np.std(res_bin))
        print(
            f"  bin {lo:.1f}-{hi:.1f}: "
            f"N={res_bin.size}, mean={mean_bin:.3f}, std={std_bin:.3f}"
        )
