#!/usr/bin/env python3
"""
15_fisher_sparc_rcut_scaling.py

Simple regression of log10 R_cut vs log10 R_max using fisher_sparc_summary.csv.

Outputs regression coefficients and scatter, to characterise how tightly
R_cut is tied to disc size.
"""

import csv
import math
from pathlib import Path

import numpy as np

SUM_CSV = Path("results") / "fisher_sparc" / "fisher_sparc_summary.csv"


def load_summary(path: Path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


if __name__ == "__main__":
    if not SUM_CSV.exists():
        print(f"[ERROR] Summary CSV not found at {SUM_CSV}")
        raise SystemExit(1)

    rows = load_summary(SUM_CSV)

    logRcut = []
    logRmax = []
    names = []

    for r in rows:
        if r["status"] != "ok":
            continue
        try:
            Rcut = float(r["R_cut_kpc"])
            Rmax = float(r["R_max_kpc"])
        except Exception:
            continue
        if Rcut <= 0.0 or Rmax <= 0.0:
            continue
        names.append(r["name"])
        logRcut.append(math.log10(Rcut))
        logRmax.append(math.log10(Rmax))

    logRcut = np.array(logRcut)
    logRmax = np.array(logRmax)

    if logRcut.size < 10:
        print("[ERROR] Not enough galaxies for regression")
        raise SystemExit(1)

    # Ordinary least squares: logRcut = alpha + beta logRmax
    X = np.vstack([np.ones_like(logRmax), logRmax]).T
    y = logRcut
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])

    y_pred = alpha + beta * logRmax
    residual = y - y_pred
    std_res = float(np.std(residual))

    print(f"[RCUT] Regression log10 R_cut = alpha + beta log10 R_max")
    print(f"[RCUT] alpha = {alpha:.3f}, beta = {beta:.3f}")
    print(f"[RCUT] Scatter std(residual) = {std_res:.3f} dex")
