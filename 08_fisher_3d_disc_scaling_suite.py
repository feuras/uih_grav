import numpy as np
import multiprocessing as mp
from pathlib import Path
import json
import csv
from typing import Dict, Tuple, List

DATA_DIR = Path("data") / "sparc_npz"
OUT_DIR = Path("results") / "fisher_sparc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def fisher_halo_accel(R: np.ndarray, g_b: np.ndarray, A: float, R_cut: float) -> np.ndarray:
    """
    Fisher halo acceleration model.

    This is a template consistent with:
      - g_halo ∝ sqrt(g_b) in the disc
      - saturation beyond R_cut

    Replace this with your exact model from the paper when ready.
    """
    g_b_clipped = np.maximum(g_b, 1e-15)
    # Core relation g_halo ~ sqrt(g_b)
    g_core = np.sqrt(g_b_clipped)

    # Smooth saturation factor in radius
    x = np.maximum(R, 1e-6) / max(R_cut, 1e-6)
    f_sat = x / (1.0 + x)  # rises ∝ x for small R, tends to 1 for large R

    return A * g_core * f_sat

def model_velocity(R: np.ndarray,
                   v_baryon: np.ndarray,
                   A: float,
                   R_cut: float) -> np.ndarray:
    # baryonic acceleration
    g_b = v_baryon**2 / np.maximum(R, 1e-6)
    g_h = fisher_halo_accel(R, g_b, A, R_cut)
    g_tot = g_b + g_h
    return np.sqrt(np.maximum(g_tot * R, 0.0))

def chi2_for_params(R: np.ndarray,
                    v_baryon: np.ndarray,
                    v_obs: np.ndarray,
                    err_v: np.ndarray,
                    A: float,
                    R_cut: float) -> float:
    v_mod = model_velocity(R, v_baryon, A, R_cut)
    w = 1.0 / np.maximum(err_v, 1e-3)**2
    return float(np.sum(w * (v_obs - v_mod)**2))

def fit_single_galaxy(gal_path: Path) -> Dict:
    data = np.load(gal_path)
    R = data["R"].astype(float)
    v_obs = data["v_obs"].astype(float)
    err_v = data["err_v_obs"].astype(float)
    v_baryon = data["v_baryon"].astype(float)

    # Restrict to radii where data is usable
    mask = (R > 0.1) & np.isfinite(v_obs) & np.isfinite(err_v) & np.isfinite(v_baryon)
    R = R[mask]
    v_obs = v_obs[mask]
    err_v = err_v[mask]
    v_baryon = v_baryon[mask]

    if len(R) < 4:
        return {
            "name": gal_path.stem,
            "status": "too_few_points",
        }

    # simple grid search over R_cut
    R_min, R_max = float(np.min(R)), float(np.max(R))
    R_cut_grid = np.linspace(0.5 * R_min, 2.0 * R_max, 25)

    best = {
        "chi2": np.inf,
        "A": None,
        "R_cut": None,
    }

    for R_cut in R_cut_grid:
        # one dimensional line search for A on [0, A_max]
        A_lo, A_hi = 0.0, 20.0  # adjust if needed

        for _ in range(50):
            A1 = A_lo + (A_hi - A_lo) / 3.0
            A2 = A_hi - (A_hi - A_lo) / 3.0
            chi1 = chi2_for_params(R, v_baryon, v_obs, err_v, A1, R_cut)
            chi2 = chi2_for_params(R, v_baryon, v_obs, err_v, A2, R_cut)
            if chi1 < chi2:
                A_hi = A2
            else:
                A_lo = A1

        A_best = 0.5 * (A_lo + A_hi)
        chi = chi2_for_params(R, v_baryon, v_obs, err_v, A_best, R_cut)

        if chi < best["chi2"]:
            best["chi2"] = chi
            best["A"] = float(A_best)
            best["R_cut"] = float(R_cut)

    # Build RAR points for this galaxy with best fit
    g_b = v_baryon**2 / np.maximum(R, 1e-6)
    v_mod = model_velocity(R, v_baryon, best["A"], best["R_cut"])
    g_tot_mod = v_mod**2 / np.maximum(R, 1e-6)
    g_tot_obs = v_obs**2 / np.maximum(R, 1e-6)

    rar_data = np.stack([g_b, g_tot_obs, g_tot_mod], axis=1)

    rar_path = OUT_DIR / f"rar_{gal_path.stem}.npy"
    np.save(rar_path, rar_data)

    return {
        "name": gal_path.stem,
        "status": "ok",
        "A": best["A"],
        "R_cut": best["R_cut"],
        "chi2": best["chi2"],
        "N_points": int(len(R)),
        "R_min": float(R_min),
        "R_max": float(R_max),
    }

def _worker(gal_path: Path) -> Dict:
    try:
        return fit_single_galaxy(gal_path)
    except Exception as e:
        return {"name": gal_path.stem, "status": f"error: {e}"}

if __name__ == "__main__":
    gal_files: List[Path] = sorted(DATA_DIR.glob("galaxy_*.npz"))

    with mp.Pool(processes=22) as pool:
        results: List[Dict] = list(pool.map(_worker, gal_files))

    # Save summary as JSON and CSV
    json_path = OUT_DIR / "fisher_sparc_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = OUT_DIR / "fisher_sparc_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["name", "status", "A", "R_cut", "chi2", "N_points", "R_min", "R_max"]
        writer.writerow(header)
        for r in results:
            writer.writerow([
                r.get("name"),
                r.get("status"),
                r.get("A"),
                r.get("R_cut"),
                r.get("chi2"),
                r.get("N_points"),
                r.get("R_min"),
                r.get("R_max"),
            ])

    # Aggregate RAR into one file for plotting later
    rar_all: List[np.ndarray] = []
    for gal_path in gal_files:
        rar_file = OUT_DIR / f"rar_{gal_path.stem}.npy"
        if rar_file.exists():
            rar_all.append(np.load(rar_file))
    if rar_all:
        rar_all_arr = np.concatenate(rar_all, axis=0)
        np.save(OUT_DIR / "rar_all.npy", rar_all_arr)
