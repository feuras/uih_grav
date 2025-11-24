#!/usr/bin/env python3
"""
12_plot_fisher_sparc_rar.py

Plot the SPARC radial acceleration relation:

  - Observed: g_tot_obs vs g_b
  - UIH Fisher model: g_tot_mod vs g_b

using rar_all.npy produced by 09_fish_halo_sparc.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("results") / "fisher_sparc"
RAR_PATH = OUT_DIR / "rar_all.npy"


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAR_PATH.exists():
        print(f"[ERROR] rar_all.npy not found at {RAR_PATH}")
        raise SystemExit(1)

    rar_all = np.load(RAR_PATH)  # shape (N_total_points, 3)
    if rar_all.shape[1] != 3:
        print(f"[ERROR] rar_all.npy has unexpected shape {rar_all.shape}, expected (N, 3)")
        raise SystemExit(1)

    g_b = rar_all[:, 0]
    g_tot_obs = rar_all[:, 1]
    g_tot_mod = rar_all[:, 2]

    # Basic sanity filter
    mask = (
        np.isfinite(g_b)
        & np.isfinite(g_tot_obs)
        & np.isfinite(g_tot_mod)
        & (g_b > 0.0)
        & (g_tot_obs > 0.0)
        & (g_tot_mod > 0.0)
    )
    g_b = g_b[mask]
    g_tot_obs = g_tot_obs[mask]
    g_tot_mod = g_tot_mod[mask]

    print(f"[INFO] RAR points after filtering: {g_b.size}")

    # Main RAR scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(g_b, g_tot_obs, s=4, alpha=0.25, label="SPARC observed")
    ax.scatter(g_b, g_tot_mod, s=4, alpha=0.25, label="UIH Fisher model")

    # 1-1 line for reference
    g_min = np.min(g_b)
    g_max = np.max(g_b)
    g_line = np.logspace(np.log10(g_min), np.log10(g_max), 200)
    ax.plot(g_line, g_line, linestyle="--", linewidth=1.0, label="g_tot = g_b")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$g_b$ [m s$^{-2}$]")
    ax.set_ylabel(r"$g_{\mathrm{tot}}$ [m s$^{-2}$]")
    ax.set_title("SPARC RAR: observed vs UIH Fisher halo")
    ax.legend(markerscale=2, fontsize=8)

    out_png = OUT_DIR / "rar_observed_vs_fisher.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved RAR plot to {out_png}")
