#!/usr/bin/env python3
"""
13_simple_halo_sparc_fit.py

Control fit for SPARC rotation curves using a generic 2-parameter dark halo:

    v_DM^2(R) = v0^2 * R^2 / (R^2 + Rc^2)
    v_tot^2(R) = v_baryon^2(R) + v_DM^2(R).

This uses the same SPARC npz files as the UIH Fisher halo scripts and
returns per galaxy best fit (v0, Rc) and chi2.

Outputs:
  - results/simple_halo_sparc/simple_halo_summary.json
  - results/simple_halo_sparc/simple_halo_summary.csv
  - results/simple_halo_sparc/rar_simple_all.npy
"""

import csv
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

import numpy as np

DATA_DIR = Path("data") / "sparc_npz"
OUT_DIR = Path("results") / "simple_halo_sparc"


def load_galaxy_npz(path: Path):
    data = np.load(path)
    required = ["R", "v_obs", "err_v_obs", "v_baryon"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"missing keys: {','.join(missing)}")

    R = np.asarray(data["R"], dtype=float)
    v_obs = np.asarray(data["v_obs"], dtype=float)
    err_v = np.asarray(data["err_v_obs"], dtype=float)
    v_baryon = np.asarray(data["v_baryon"], dtype=float)

    distance_Mpc = float(data["distance_Mpc"]) if "distance_Mpc" in data else float("nan")
    galaxy_name = str(data["galaxy_name"]) if "galaxy_name" in data else path.stem

    # Cleaning and sorting
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

    if R.size < 4:
        raise ValueError("too few points after cleaning")

    idx = np.argsort(R)
    return {
        "name": galaxy_name,
        "R": R[idx],
        "v_obs": v_obs[idx],
        "err_v": err_v[idx],
        "v_bar": v_baryon[idx],
        "distance_Mpc": distance_Mpc,
    }


def chi2_from_halo(
    v0: float,
    Rc: float,
    R: np.ndarray,
    v_obs: np.ndarray,
    err_v: np.ndarray,
    v_baryon: np.ndarray,
) -> float:
    R2 = R**2
    Rc2 = Rc**2
    v_dm2 = v0**2 * R2 / (R2 + Rc2)
    v_model = np.sqrt(v_baryon**2 + v_dm2)
    sigma2 = np.maximum(err_v, 1.0e-3) ** 2
    return float(np.sum((v_obs - v_model) ** 2 / sigma2))


def fit_simple_halo_for_galaxy(gal_path: Path) -> Dict:
    try:
        gal = load_galaxy_npz(gal_path)
    except Exception as e:
        return {"name": gal_path.stem, "status": f"load_error:{e}"}

    name = gal["name"]
    R = gal["R"]
    v_obs = gal["v_obs"]
    err_v = gal["err_v"]
    v_bar = gal["v_bar"]
    distance_Mpc = gal["distance_Mpc"]

    # Baryon chi2 and dof
    sigma2 = np.maximum(err_v, 1.0e-3) ** 2
    chi2_baryon = float(np.sum((v_obs - v_bar) ** 2 / sigma2))

    N = R.size
    R_min = float(np.min(R))
    R_max = float(np.max(R))

    # Parameter ranges
    v_obs_max = float(np.max(v_obs))
    v0_min = 1.0
    v0_max = max(v_obs_max * 3.0, 20.0)

    Rc_min = max(0.1 * R_min, 0.05)
    Rc_max = 2.0 * R_max

    # Log grids
    Nv = 40
    Nr = 40
    log_v0_grid = np.linspace(math.log10(v0_min), math.log10(v0_max), Nv)
    log_Rc_grid = np.linspace(math.log10(Rc_min), math.log10(Rc_max), Nr)

    v0_grid = 10.0**log_v0_grid
    Rc_grid = 10.0**log_Rc_grid

    # Vectorised chi2 evaluation
    V0, RC = np.meshgrid(v0_grid, Rc_grid, indexing="ij")  # shape (Nv, Nr)
    R_arr = R[None, None, :]  # shape (1, 1, N)
    err2 = sigma2[None, None, :]
    vbar2 = v_bar[None, None, :] ** 2

    R2 = R_arr**2
    Rc2 = RC[..., None] ** 2
    v_dm2 = V0[..., None] ** 2 * R2 / (R2 + Rc2)
    v_model = np.sqrt(vbar2 + v_dm2)
    residual = v_obs[None, None, :] - v_model
    chi2_grid = np.sum(residual**2 / err2, axis=2)  # shape (Nv, Nr)

    idx_min = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    best_v0 = float(v0_grid[idx_min[0]])
    best_Rc = float(Rc_grid[idx_min[1]])
    best_chi2 = float(chi2_grid[idx_min])

    # Boundary diagnostics
    v0_idx, Rc_idx = idx_min
    at_v0_lower = v0_idx == 0
    at_v0_upper = v0_idx == Nv - 1
    at_Rc_lower = Rc_idx == 0
    at_Rc_upper = Rc_idx == Nr - 1
    boundary_flags = []
    if at_v0_lower:
        boundary_flags.append("v0_lower")
    if at_v0_upper:
        boundary_flags.append("v0_upper")
    if at_Rc_lower:
        boundary_flags.append("Rc_lower")
    if at_Rc_upper:
        boundary_flags.append("Rc_upper")
    boundary_flag = ",".join(boundary_flags) if boundary_flags else "none"

    # Build RAR triple for this model
    kpc_to_m = 3.0856775814913673e19
    km_to_m = 1.0e3
    R_m = R * kpc_to_m
    v_obs_mps = v_obs * km_to_m
    vbar_mps = v_bar * km_to_m

    # Reconstruct best model velocities
    R2_1d = R**2
    v_dm2_best = best_v0**2 * R2_1d / (R2_1d + best_Rc**2)
    v_model_best = np.sqrt(v_bar**2 + v_dm2_best)
    v_model_best_mps = v_model_best * km_to_m

    g_b = vbar_mps**2 / np.maximum(R_m, 1.0e-3 * kpc_to_m)
    g_tot_obs = v_obs_mps**2 / np.maximum(R_m, 1.0e-3 * kpc_to_m)
    g_tot_mod = v_model_best_mps**2 / np.maximum(R_m, 1.0e-3 * kpc_to_m)

    rar = np.stack([g_b, g_tot_obs, g_tot_mod], axis=1)
    rar_path = OUT_DIR / f"rar_simple_{name}.npy"
    np.save(rar_path, rar)

    return {
        "name": name,
        "status": "ok",
        "v0": best_v0,
        "Rc_kpc": best_Rc,
        "chi2": best_chi2,
        "chi2_baryon": chi2_baryon,
        "boundary": boundary_flag,
        "N_points": int(N),
        "R_min_kpc": R_min,
        "R_max_kpc": R_max,
        "distance_Mpc": distance_Mpc,
    }


def _worker(path: Path) -> Dict:
    try:
        return fit_simple_halo_for_galaxy(path)
    except Exception as e:
        return {"name": path.stem, "status": f"worker_error:{e}"}


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gal_files: List[Path] = sorted(DATA_DIR.glob("galaxy_*.npz"))
    if not gal_files:
        print(f"[ERROR] No galaxy_*.npz files in {DATA_DIR}")
        raise SystemExit(1)

    print(f"[INFO] Found {len(gal_files)} galaxies in {DATA_DIR}")

    cpu_count = mp.cpu_count()
    n_procs = min(22, cpu_count, len(gal_files))
    if n_procs < 1:
        n_procs = 1
    print(f"[INFO] Using {n_procs} process(es) (cpu_count={cpu_count})")

    results: List[Dict] = []
    if n_procs == 1:
        for gf in gal_files:
            r = _worker(gf)
            results.append(r)
            print(f"[GAL] {r.get('name')}: {r.get('status')}")
    else:
        with mp.Pool(processes=n_procs) as pool:
            for r in pool.imap_unordered(_worker, gal_files, chunksize=1):
                results.append(r)
                print(f"[GAL] {r.get('name')}: {r.get('status')}")

    # Save JSON and CSV
    json_path = OUT_DIR / "simple_halo_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved JSON summary to {json_path}")

    csv_path = OUT_DIR / "simple_halo_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "name",
            "status",
            "v0_kms",
            "Rc_kpc",
            "chi2",
            "chi2_baryon",
            "boundary",
            "N_points",
            "R_min_kpc",
            "R_max_kpc",
            "distance_Mpc",
        ]
        writer.writerow(header)
        for r in results:
            writer.writerow([
                r.get("name"),
                r.get("status"),
                r.get("v0"),
                r.get("Rc_kpc"),
                r.get("chi2"),
                r.get("chi2_baryon"),
                r.get("boundary"),
                r.get("N_points"),
                r.get("R_min_kpc"),
                r.get("R_max_kpc"),
                r.get("distance_Mpc"),
            ])
    print(f"[INFO] Saved CSV summary to {csv_path}")

    # Aggregate simple halo RAR
    rar_all = []
    for r in results:
        if r.get("status") != "ok":
            continue
        name = r.get("name")
        rar_path = OUT_DIR / f"rar_simple_{name}.npy"
        if rar_path.exists():
            rar_all.append(np.load(rar_path))

    if rar_all:
        rar_all = np.concatenate(rar_all, axis=0)
        rar_all_path = OUT_DIR / "rar_simple_all.npy"
        np.save(rar_all_path, rar_all)
        print(f"[INFO] Saved aggregated simple halo RAR to {rar_all_path}")
    else:
        print("[WARN] No ok galaxies for simple halo RAR aggregation")

    # High level chi2 stats
    ok_res = [r for r in results if r.get("status") == "ok"]
    n_total = len(results)
    n_ok = len(ok_res)
    print(f"[SUMMARY] Galaxies total: {n_total}, ok: {n_ok}, failed: {n_total - n_ok}")

    if ok_res:
        chi2 = np.array([r["chi2"] for r in ok_res], dtype=float)
        chi2_b = np.array([r["chi2_baryon"] for r in ok_res], dtype=float)
        Npts = np.array([r["N_points"] for r in ok_res], dtype=float)

        dof_halo = np.maximum(Npts - 2.0, 1.0)
        dof_bar = np.maximum(Npts, 1.0)
        chi2_red_halo = chi2 / dof_halo
        chi2_red_bar = chi2_b / dof_bar

        def stats(x):
            x = x[np.isfinite(x)]
            return float(np.min(x)), float(np.median(x)), float(np.max(x))

        h_min, h_med, h_max = stats(chi2_red_halo)
        b_min, b_med, b_max = stats(chi2_red_bar)
        print(f"[SUMMARY] chi2_red simple halo: min={h_min:.3f}, median={h_med:.3f}, max={h_max:.3f}")
        print(f"[SUMMARY] chi2_red baryon:      min={b_min:.3f}, median={b_med:.3f}, max={b_max:.3f}")

        ratio = chi2_red_bar / chi2_red_halo
        n_improve_10 = int(np.sum(ratio > 1.10))
        n_degrade_10 = int(np.sum(ratio < 0.90))
        print(f"[SUMMARY] Galaxies with >10 percent improvement: {n_improve_10}")
        print(f"[SUMMARY] Galaxies with >10 percent worse:       {n_degrade_10}")
