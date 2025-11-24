#!/usr/bin/env python3
"""
10_analyse_fisher_sparc_summary.py

Quick diagnostics on fisher_sparc_summary.csv:
  - Distributions of A, log10A, R_cut, chi2_red
  - Correlations between log10A and R_max, distance
  - Correlations between R_cut and R_max
"""

import csv
from pathlib import Path

import numpy as np

SUMMARY_CSV = Path("results") / "fisher_sparc" / "fisher_sparc_summary.csv"


def load_summary(path: Path):
    names = []
    status = []
    A = []
    logA = []
    Rcut = []
    chi2 = []
    chi2_b = []
    Npts = []
    Rmin = []
    Rmax = []
    dist = []
    Abound = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"])
            status.append(row["status"])
            A.append(float(row["A"]) if row["A"] not in ("", "None") else np.nan)
            logA.append(float(row["log10A"]) if row.get("log10A", "") not in ("", "None") else np.nan)
            Rcut.append(float(row["R_cut_kpc"]) if row["R_cut_kpc"] not in ("", "None") else np.nan)
            chi2.append(float(row["chi2"]) if row["chi2"] not in ("", "None") else np.nan)
            chi2_b.append(float(row["chi2_baryon"]) if row["chi2_baryon"] not in ("", "None") else np.nan)
            Npts.append(int(row["N_points"]) if row["N_points"] not in ("", "None") else 0)
            Rmin.append(float(row["R_min_kpc"]) if row["R_min_kpc"] not in ("", "None") else np.nan)
            Rmax.append(float(row["R_max_kpc"]) if row["R_max_kpc"] not in ("", "None") else np.nan)
            dist.append(float(row["distance_Mpc"]) if row["distance_Mpc"] not in ("", "None") else np.nan)
            Abound.append(row.get("A_boundary", "none") or "none")

    return {
        "name": np.array(names, dtype=object),
        "status": np.array(status, dtype=object),
        "A": np.array(A),
        "logA": np.array(logA),
        "Rcut": np.array(Rcut),
        "chi2": np.array(chi2),
        "chi2_b": np.array(chi2_b),
        "N": np.array(Npts, dtype=float),
        "Rmin": np.array(Rmin),
        "Rmax": np.array(Rmax),
        "dist": np.array(dist),
        "Abound": np.array(Abound, dtype=object),
    }


def robust_stats(x):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.min(x)), float(np.median(x)), float(np.max(x))


if __name__ == "__main__":
    if not SUMMARY_CSV.exists():
        print(f"[ERROR] Summary CSV not found at {SUMMARY_CSV}")
        raise SystemExit(1)

    data = load_summary(SUMMARY_CSV)

    ok_mask = data["status"] == "ok"
    for key in ("A", "logA", "Rcut", "chi2", "chi2_b", "N", "Rmin", "Rmax", "dist"):
        data[key] = data[key][ok_mask]
    names = data["name"][ok_mask]
    Abound = data["Abound"][ok_mask]

    N_eff = data["N"]
    dof_fisher = np.maximum(N_eff - 2.0, 1.0)
    dof_baryon = np.maximum(N_eff, 1.0)
    chi2_red_f = data["chi2"] / dof_fisher
    chi2_red_b = data["chi2_b"] / dof_baryon

    print(f"[INFO] Analysing {len(names)} ok galaxies")

    A_min, A_med, A_max = robust_stats(data["A"])
    logA_min, logA_med, logA_max = robust_stats(data["logA"])
    Rc_min, Rc_med, Rc_max = robust_stats(data["Rcut"])
    c2f_min, c2f_med, c2f_max = robust_stats(chi2_red_f)
    c2b_min, c2b_med, c2b_max = robust_stats(chi2_red_b)

    print(f"[STATS] A:      min={A_min:.3e}, median={A_med:.3e}, max={A_max:.3e}")
    print(f"[STATS] log10A: min={logA_min:.2f}, median={logA_med:.2f}, max={logA_max:.2f}")
    print(f"[STATS] R_cut:  min={Rc_min:.3f} kpc, median={Rc_med:.3f} kpc, max={Rc_max:.3f} kpc")
    print(f"[STATS] chi2_red Fisher:  min={c2f_min:.3f}, median={c2f_med:.3f}, max={c2f_max:.3f}")
    print(f"[STATS] chi2_red baryon:  min={c2b_min:.3f}, median={c2b_med:.3f}, max={c2b_max:.3f}")

    # Improvement factors per galaxy
    improvement = chi2_red_b / chi2_red_f
    imp_min, imp_med, imp_max = robust_stats(improvement)
    print(f"[STATS] chi2_red_b / chi2_red_f: min={imp_min:.3f}, median={imp_med:.3f}, max={imp_max:.3f}")

    # Boundary counts
    n_lower = int(np.sum(Abound == "lower"))
    n_upper = int(np.sum(Abound == "upper"))
    print(f"[STATS] A boundary hits: lower={n_lower}, upper={n_upper}")

    # Correlations
    def corr(x, y, label):
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3:
            print(f"[CORR] {label}: insufficient data")
            return
        xm = x[mask] - np.mean(x[mask])
        ym = y[mask] - np.mean(y[mask])
        num = float(np.sum(xm * ym))
        den = float(np.sqrt(np.sum(xm**2) * np.sum(ym**2)))
        r = num / den if den > 0 else np.nan
        print(f"[CORR] {label}: r = {r:.3f}, N = {np.sum(mask)}")

    corr(data["logA"], np.log10(np.maximum(data["Rmax"], 1e-3)), "log10A vs log10 R_max")
    corr(data["logA"], np.log10(np.maximum(data["dist"], 1e-3)), "log10A vs log10 distance")
    corr(np.log10(np.maximum(data["Rcut"], 1e-6)), np.log10(np.maximum(data["Rmax"], 1e-3)), "log10 R_cut vs log10 R_max")
