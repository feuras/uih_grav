#!/usr/bin/env python3
"""
16_fisher_sparc_btfr.py

Build a BTFR footprint for the UIH Fisher halo + SPARC using code-unit
baryonic masses:

  - For each galaxy with an ok Fisher fit:
      * load R, v_obs, v_baryon from galaxy_*.npz
      * reconstruct v_model(R) from best fit (A, R_cut)
      * extract v_flat_obs and v_flat_mod
      * define total baryonic "mass" M_b,tot = M_b(R_max) with
        M_b(R) = v_b(R)**2 * R in the same units as in the grav paper

Outputs:
  - results/fisher_sparc/btfr_fisher_sparc.csv
  - results/fisher_sparc/btfr_fisher_sparc.png
  - results/fisher_sparc/btfr_residuals.png
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("data") / "sparc_npz"
SUMMARY_CSV = Path("results") / "fisher_sparc" / "fisher_sparc_summary.csv"
OUT_DIR = Path("results") / "fisher_sparc"


def compute_baryonic_mass_profile(R_kpc: np.ndarray, v_baryon_kms: np.ndarray) -> np.ndarray:
    """
    Code unit baryonic mass profile M_b(R) from the baryonic rotation curve.

        v_b(R)**2 = M_b(R) / R   (G = 1 units)

    so M_b(R) = v_b(R)**2 * R.
    """
    return v_baryon_kms**2 * R_kpc


def compute_g_baryon(R_kpc: np.ndarray, v_baryon_kms: np.ndarray) -> np.ndarray:
    """Code unit baryonic acceleration g_b(R) = v_b(R)**2 / R."""
    R_safe = np.maximum(R_kpc, 1.0e-6)
    return v_baryon_kms**2 / R_safe


def compute_g_base_halo(R_kpc: np.ndarray, M_b: np.ndarray, R_cut_kpc: float) -> np.ndarray:
    """
    Base halo acceleration profile g_base(R) for A = 1, consistent with
    the scalar Fisher halo construction in 06/07/09 scripts.

        M_grad(R) = ∫_0^R alpha(r) M_b(r)**2 / r**2 dr
        g_h(R)    = M_grad(R) / R**2

    with alpha(r) ∝ (r/R_cut)**2 inside R_cut and alpha(r) = const outside.
    """
    R_cut = max(float(R_cut_kpc), 1.0e-6)
    R = np.asarray(R_kpc, dtype=float)

    # Shape factor alpha(r; R_cut)
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


def v_flat_from_curve(R: np.ndarray, v: np.ndarray, R_d: float = None) -> float:
    """
    Extract a robust v_flat from a rotation curve using a log-slope cut
    in the outer disc, with a simple outer-radius fallback.
    """
    R = np.asarray(R, dtype=float)
    v = np.asarray(v, dtype=float)

    # Sort radii just in case
    idx = np.argsort(R)
    R = R[idx]
    v = v[idx]

    if R.size < 4:
        return float(np.median(v))

    logR = np.log(R)
    logv = np.log(np.maximum(v, 1.0e-6))
    s = np.diff(logv) / np.diff(logR)

    Rmax = R[-1]
    if R_d is not None and np.isfinite(R_d) and R_d > 0.0:
        rad_mask = (R >= 0.5 * Rmax) | (R >= 2.0 * R_d)
    else:
        rad_mask = (R >= 0.5 * Rmax)

    slope_mask = np.ones_like(R, dtype=bool)
    slope_mask[1:] &= np.abs(s) <= 0.1
    slope_mask[:-1] &= np.abs(s) <= 0.1

    mask = rad_mask & slope_mask
    if np.count_nonzero(mask) >= 3:
        return float(np.median(v[mask]))
    else:
        outer_mask = (R >= Rmax * (2.0 / 3.0))
        if np.count_nonzero(outer_mask) >= 2:
            return float(np.median(v[outer_mask]))
        return float(np.median(v))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_CSV.exists():
        raise SystemExit(f"[ERROR] Summary CSV not found at {SUMMARY_CSV}")
    if not DATA_DIR.exists():
        raise SystemExit(f"[ERROR] SPARC npz directory not found at {DATA_DIR}")

    # Load Fisher fit summary
    rows = []
    with open(SUMMARY_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    names = []
    labels = []
    log10Mb_list = []
    vflat_obs_list = []
    vflat_mod_list = []

    for r in rows:
        if r.get("status", "ok") != "ok":
            continue

        # e.g. "CamB", "D512-2", ...
        name = r["name"]

        try:
            A = float(r["A"])
            R_cut = float(r["R_cut_kpc"])
        except Exception:
            print(f"[BTFR] Skipping {name}, could not parse A or R_cut_kpc")
            continue

        # Try "galaxy_<name>.npz" first (earlier convention), then "<name>.npz"
        npz_path = DATA_DIR / f"galaxy_{name}.npz"
        if not npz_path.exists():
            npz_path = DATA_DIR / f"{name}.npz"
        if not npz_path.exists():
            print(f"[BTFR] Skipping {name}, npz not found at {npz_path}")
            continue

        data = np.load(npz_path)
        R = np.asarray(data["R"], dtype=float)
        v_obs = np.asarray(data["v_obs"], dtype=float)
        err_v = np.asarray(data["err_v_obs"], dtype=float)
        v_baryon = np.asarray(data["v_baryon"], dtype=float)

        if "galaxy_name" in data.files:
            gal_label = str(data["galaxy_name"])
        else:
            gal_label = name

        # Cleaning
        mask = (
            np.isfinite(R)
            & np.isfinite(v_obs)
            & np.isfinite(err_v)
            & np.isfinite(v_baryon)
            & (R > 0.05)
        )
        R = R[mask]
        v_obs = v_obs[mask]
        v_baryon = v_baryon[mask]

        if R.size < 4:
            print(f"[BTFR] Skipping {name}, too few points after cleaning")
            continue

        # Ensure monotone radii for integration and v_flat
        idx = np.argsort(R)
        R = R[idx]
        v_obs = v_obs[idx]
        v_baryon = v_baryon[idx]

        # Baryonic mass and accelerations in code units
        M_b_profile = compute_baryonic_mass_profile(R, v_baryon)
        M_b_tot = float(M_b_profile[-1])
        if not np.isfinite(M_b_tot) or M_b_tot <= 0.0:
            print(f"[BTFR] Skipping {name}, nonpositive M_b_tot")
            continue

        g_b = compute_g_baryon(R, v_baryon)
        g_base = compute_g_base_halo(R, M_b_profile, R_cut)
        g_tot_mod = g_b + A * g_base
        v_mod = np.sqrt(np.maximum(g_tot_mod * R, 0.0))

        # v_flat from observed and model curves
        v_flat_obs = v_flat_from_curve(R, v_obs)
        v_flat_mod = v_flat_from_curve(R, v_mod)

        # Code-unit baryonic mass: log10 M_b_tot
        log10Mb = np.log10(M_b_tot)

        names.append(name)
        labels.append(gal_label)
        log10Mb_list.append(log10Mb)
        vflat_obs_list.append(v_flat_obs)
        vflat_mod_list.append(v_flat_mod)

    if not log10Mb_list:
        raise SystemExit("[BTFR] No galaxies with usable Fisher fit + M_b_tot")

    log10Mb_arr = np.array(log10Mb_list, dtype=float)
    log10v_obs = np.log10(np.array(vflat_obs_list, dtype=float))
    log10v_mod = np.log10(np.array(vflat_mod_list, dtype=float))

    # Save BTFR data table
    out_csv = OUT_DIR / "btfr_fisher_sparc.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "label", "log10M_b_code", "v_flat_obs", "v_flat_mod"])
        for n, lab, m, vo, vm in zip(names, labels, log10Mb_arr, vflat_obs_list, vflat_mod_list):
            writer.writerow([n, lab, f"{m:.5f}", f"{vo:.5f}", f"{vm:.5f}"])
    print(f"[BTFR] Saved BTFR table to {out_csv}")
    print(f"[BTFR] Galaxies used: {len(log10Mb_arr)}")

    # Simple linear fits in log space: log10 v_flat = a + b log10 M_b_code
    A_obs_mat = np.vstack([np.ones_like(log10Mb_arr), log10Mb_arr]).T
    beta_obs, _, _, _ = np.linalg.lstsq(A_obs_mat, log10v_obs, rcond=None)
    a_obs, b_obs = float(beta_obs[0]), float(beta_obs[1])

    A_mod_mat = np.vstack([np.ones_like(log10Mb_arr), log10Mb_arr]).T
    beta_mod, _, _, _ = np.linalg.lstsq(A_mod_mat, log10v_mod, rcond=None)
    a_mod, b_mod = float(beta_mod[0]), float(beta_mod[1])

    print(f"[BTFR] Observed fit (code units): log10 v_flat = {a_obs:.3f} + {b_obs:.3f} log10 M_b")
    print(f"[BTFR] Fisher   fit (code units): log10 v_flat = {a_mod:.3f} + {b_mod:.3f} log10 M_b")

    # BTFR plot in code units
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(log10Mb_arr, log10v_obs, s=12, alpha=0.4, label="SPARC observed")
    ax.scatter(log10Mb_arr, log10v_mod, s=12, alpha=0.4, label="UIH Fisher model")

    x_grid = np.linspace(np.min(log10Mb_arr) - 0.2, np.max(log10Mb_arr) + 0.2, 200)
    y_obs_fit = a_obs + b_obs * x_grid
    y_mod_fit = a_mod + b_mod * x_grid
    ax.plot(x_grid, y_obs_fit, linestyle="--", linewidth=1.0, label="Obs BTFR fit")
    ax.plot(x_grid, y_mod_fit, linestyle="-.", linewidth=1.0, label="Fisher BTFR fit")

    ax.set_xlabel(r"$\log_{10} M_b$ (code units)")
    ax.set_ylabel(r"$\log_{10} v_{\mathrm{flat}}\ [\mathrm{km\,s^{-1}}]$")
    ax.set_title("BTFR (code units): SPARC vs UIH Fisher halo")
    ax.legend(fontsize=8)

    out_png = OUT_DIR / "btfr_fisher_sparc.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[BTFR] Saved BTFR plot to {out_png}")

    # Residuals: log10 v_flat_mod - log10 v_flat_obs vs log10 M_b_code
    fig, ax = plt.subplots(figsize=(6, 4))
    dv = log10v_mod - log10v_obs
    ax.axhline(0.0, linewidth=1.0)
    ax.scatter(log10Mb_arr, dv, s=12, alpha=0.5)
    ax.set_xlabel(r"$\log_{10} M_b$ (code units)")
    ax.set_ylabel(r"$\Delta \log_{10} v_{\mathrm{flat}}$ (Fisher - obs)")
    ax.set_title("BTFR residuals (code units): Fisher vs SPARC")

    out_resid = OUT_DIR / "btfr_residuals.png"
    fig.savefig(out_resid, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[BTFR] Saved BTFR residuals to {out_resid}")
