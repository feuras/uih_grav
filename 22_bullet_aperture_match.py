#!/usr/bin/env python3
"""
22_bullet_aperture_match.py

Compare UIH Bullet toy nonlocal ratios to the observed Bullet Cluster
aperture masses and mean kappa from Clowe et al. (2006).

Inputs:
  - results/bullet_toy_nonlocal/bullet_toy_nonlocal_scan.csv

Outputs:
  - prints observed baryon and kappa ratios
  - prints model ratios for each (q, L)
  - saves a simple comparison plot to results/bullet_toy_nonlocal/bullet_ratio_comparison.png
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCAN_CSV = Path("results") / "bullet_toy_nonlocal" / "bullet_toy_nonlocal_scan.csv"
OUT_PNG = Path("results") / "bullet_toy_nonlocal" / "bullet_ratio_comparison.png"

def bullet_observed_ratios():
    # Table 2 numbers from Clowe et al. 2006, in 1e12 M_sun and mean kappa
    # Main cluster
    Mx_main_BCG, Ms_main_BCG, k_main_BCG = 5.5, 0.54, 0.36
    Mx_main_gas, Ms_main_gas, k_main_gas = 6.6, 0.23, 0.05

    # Subcluster
    Mx_sub_BCG, Ms_sub_BCG, k_sub_BCG = 2.7, 0.58, 0.20
    Mx_sub_gas, Ms_sub_gas, k_sub_gas = 5.8, 0.12, 0.02

    Mb_main_BCG = Mx_main_BCG + Ms_main_BCG
    Mb_main_gas = Mx_main_gas + Ms_main_gas
    Mb_sub_BCG = Mx_sub_BCG + Ms_sub_BCG
    Mb_sub_gas = Mx_sub_gas + Ms_sub_gas

    # baryon ratios (plasma / BCG)
    Rb_main = Mb_main_gas / Mb_main_BCG
    Rb_sub = Mb_sub_gas / Mb_sub_BCG

    # kappa ratios (BCG / plasma)
    Rk_main = k_main_BCG / k_main_gas
    Rk_sub = k_sub_BCG / k_sub_gas

    # simple "target" Bullet ratio, geometric mean of the two kappa ratios
    Rk_target = np.sqrt(Rk_main * Rk_sub)

    return {
        "Mb_main_BCG": Mb_main_BCG,
        "Mb_main_gas": Mb_main_gas,
        "Mb_sub_BCG": Mb_sub_BCG,
        "Mb_sub_gas": Mb_sub_gas,
        "Rb_main": Rb_main,
        "Rb_sub": Rb_sub,
        "Rk_main": Rk_main,
        "Rk_sub": Rk_sub,
        "Rk_target": Rk_target,
    }

def load_model_scan():
    if not SCAN_CSV.exists():
        raise SystemExit(f"[BulletMatch] Scan CSV not found at {SCAN_CSV}")

    out = []
    with open(SCAN_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = float(row["q"])
            L = float(row["L"])
            eff_ratio = float(row["eff_ratio_gal_over_gas"])
            b_ratio = float(row["b_ratio_gal_over_gas"])
            out.append((q, L, b_ratio, eff_ratio))
    return out

def main():
    obs = bullet_observed_ratios()
    print("[BulletMatch] Observed Bullet aperture masses and ratios (100 kpc):")
    print(f"  Main: Mb_gas / Mb_BCG ≈ {obs['Rb_main']:.2f}, kappa_BCG / kappa_gas ≈ {obs['Rk_main']:.1f}")
    print(f"  Sub : Mb_gas / Mb_BCG ≈ {obs['Rb_sub']:.2f}, kappa_BCG / kappa_gas ≈ {obs['Rk_sub']:.1f}")
    print(f"  Target lensing ratio (geometric mean) Rk_target ≈ {obs['Rk_target']:.1f}")
    print()

    scan = load_model_scan()
    print("[BulletMatch] UIH nonlocal Bullet toy scan (from 21_bullet_cluster_toy_nonlocal.py):")
    print("  q, L,  b_ratio_gal/gas,  eff_ratio_gal/gas,  eff_ratio / Rk_target")
    model_ratios = []
    for q, L, b_ratio, eff_ratio in scan:
        rel = eff_ratio / obs["Rk_target"]
        print(f"  {q:4.1f}, {L:4.1f}, {b_ratio:8.3f}, {eff_ratio:8.3f}, {rel:7.3f}")
        model_ratios.append((q, L, eff_ratio))

    # simple plot: model eff_ratio vs observed target
    eff_vals = np.array([m[2] for m in model_ratios])
    q_vals = np.array([m[0] for m in model_ratios])
    L_vals = np.array([m[1] for m in model_ratios])
    Rk_target = obs["Rk_target"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(q_vals, eff_vals, c=L_vals, s=50)
    ax.axhline(Rk_target, linestyle="--", linewidth=1.0, label="Bullet target Rk")
    ax.set_xlabel("q (gas to galaxy amplitude ratio)")
    ax.set_ylabel(r"$\Sigma_{\rm eff,gal} / \Sigma_{\rm eff,gas}$")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("L (kernel range)")
    ax.set_title("UIH Bullet toy vs observed Bullet Cluster ratio")
    ax.legend(fontsize=8)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[BulletMatch] Saved comparison plot to {OUT_PNG}")

if __name__ == "__main__":
    main()
