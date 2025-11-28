#!/usr/bin/env python3
"""
29_grav_sparc_two_knob_universality_scan.py

Fisher scalar halo universality tests on SPARC rotation curves.

Models:

(1) One-knob BPS Fisher halo:
    For each galaxy we build a Fisher halo integral

        I_tilde_i(R) = I_i(R) / R_disc,i^2,

    with
        M_b(R) = R v_b(R)^2,
        I_i(R) = ∫_0^R M_b(s)^2 ds.

    A single Fisher amplitude kappa gives halo enclosed mass

        M_grad(R) = kappa * I_tilde_i(R),

    and halo circular velocity

        v_halo(R)^2 = M_grad(R) / R.

    Total velocity:

        v_model(R)^2 = v_b(R)^2 + v_halo(R)^2.

    We compute:
        - per-galaxy best-fit kappa_i,
        - global kappa_global (total chi^2),
        - global kappa_med_star (median reduced chi^2 scan).

(2) Two-knob compactness-dressed Fisher halo:
    For each galaxy we define a compactness proxy

        C_raw,i = M_b,max,i / R_disc,i^2,

    where M_b,max,i is M_b(R) at the outermost radius.
    Normalise:

        C_hat,i = C_raw,i / median_j C_raw,j.

    Then we define a dressed amplitude

        kappa_eff,i = K0 * C_hat,i^beta.

    There are now two global parameters (K0, beta). We scan a grid
    in (log10 K0, beta) and find the pair that minimises the median
    reduced chi^2 across galaxies.

    At the best point we compute:
        - per-galaxy kappa_eff,i,
        - chi^2 and reduced chi^2,
        - fraction of galaxies with >10 percent improvement over baryons.

Additional diagnostics:

    - Log-normal statistics of individual kappa_i.
    - Log-log regression of log10 kappa_i against log10 C_hat,i to
      get a data-driven (beta_reg, K0_reg) independent of the grid,
      including Pearson r and R^2.
    - Grids of median and mean reduced chi^2 for the one-knob and
      two-knob scans, saved for plotting.

Input NPZ format (per galaxy):

    R            [N]   radii
    v_obs        [N]   observed circular velocity
    err_v_obs    [N]   observational uncertainty on v_obs
    v_baryon     [N]   baryonic circular velocity (stars+gas)
    galaxy_name  []    string name

All NPZ files are assumed to live under:  ./data/sparc_npz

Outputs:

    - Console summary of key statistics.
    - NPZ: "sparc_kappa_universality_results.npz"
    - JSON: "sparc_kappa_universality_summary.json"
"""

import os
import glob
import json
import numpy as np

try:
    from scipy.optimize import minimize_scalar
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

SPARC_DIR = os.path.join("data", "sparc_npz")

MIN_POINTS = 6             # minimal number of usable data points per galaxy

# One-knob kappa grid for median reduced chi^2 scan
KAPPA_GRID_MIN_EXP = -6.0
KAPPA_GRID_MAX_EXP = 0.0
KAPPA_GRID_N = 121         # 0.05 dex spacing

# Two-knob grid for (K0, beta):
K0_GRID_MIN_EXP = -5.5     # log10 K0 range
K0_GRID_MAX_EXP = -2.0
K0_GRID_N = 71             # 0.05 dex spacing in log10 K0

BETA_GRID_MIN = -2.0       # beta range
BETA_GRID_MAX = 2.0
BETA_GRID_N = 81           # step 0.05 in beta

# Bounds for individual kappa_i fits
KAPPA_MAX = 10.0

OUT_JSON = "sparc_kappa_universality_summary.json"
OUT_NPZ = "sparc_kappa_universality_results.npz"


# ----------------------------------------------------------------------
# Helpers: loading and preprocessing
# ----------------------------------------------------------------------

def baryon_scale_radius(R, v_b):
    """
    Baryon-weighted mean radius:

        R_disc = sum R_i w_i / sum w_i,  w_i = R_i v_b(R_i)^2.
    """
    weights = (v_b ** 2) * R
    mask = np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return np.nan
    w = weights[mask]
    r = R[mask]
    return float(np.sum(r * w) / np.sum(w))


def load_sparc_galaxy(path, min_points=MIN_POINTS):
    """
    Load and clean one SPARC NPZ file.
    Returns a dict or None if the galaxy is unsuitable.
    """
    dat = np.load(path)

    R = dat["R"].astype(float)
    v_obs = dat["v_obs"].astype(float)
    err_v = dat["err_v_obs"].astype(float)
    v_b = dat["v_baryon"].astype(float)

    name = dat["galaxy_name"]
    if isinstance(name, np.ndarray):
        name = name.item()
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")
    name = str(name)

    mask = (
        np.isfinite(R)
        & np.isfinite(v_obs)
        & np.isfinite(err_v)
        & np.isfinite(v_b)
        & (err_v >= 0.0)
    )

    R = R[mask]
    v_obs = v_obs[mask]
    err_v = err_v[mask]
    v_b = v_b[mask]

    idx = np.argsort(R)
    R = R[idx]
    v_obs = v_obs[idx]
    err_v = err_v[idx]
    v_b = v_b[idx]

    if R.size < min_points:
        return None

    if np.max(np.abs(v_b)) < 1e-3:
        return None

    return {
        "name": name,
        "R": R,
        "v_obs": v_obs,
        "err_v": err_v,
        "v_b": v_b,
    }


def preprocess_galaxy(gal):
    """
    From raw galaxy dict to precomputed Fisher objects:

      R, v_obs, err_v, v_b,
      R_disc, I_tilde(R), C_raw (compactness proxy).
    """
    R = gal["R"]
    v_b = gal["v_b"]

    R_disc = baryon_scale_radius(R, v_b)
    if not np.isfinite(R_disc) or R_disc <= 0.0:
        return None

    # Pseudo enclosed mass M_b(R) = R v_b^2 (G absorbed)
    M_b = R * (v_b ** 2)

    # Fisher integral I(R) = ∫_0^R M_b(s)^2 ds
    integrand = M_b ** 2
    dR = np.diff(R)
    avg = 0.5 * (integrand[1:] + integrand[:-1])
    I = np.concatenate(([0.0], np.cumsum(avg * dR)))

    I_tilde = I / (R_disc ** 2)

    # Simple compactness proxy: outermost pseudo mass over R_disc^2
    M_b_max = M_b[-1]
    C_raw = M_b_max / (R_disc ** 2) if R_disc > 0.0 else np.nan

    out = dict(gal)
    out["R_disc"] = float(R_disc)
    out["I_tilde"] = I_tilde.astype(float)
    out["C_raw"] = float(C_raw)
    return out


# ----------------------------------------------------------------------
# Chi^2 machinery
# ----------------------------------------------------------------------

def galaxy_chi2_kappa(pg, kappa):
    """
    Chi^2 misfit for one galaxy and a given Fisher amplitude kappa.
    """
    R = pg["R"]
    v_obs = pg["v_obs"]
    err_v = pg["err_v"]
    v_b = pg["v_b"]
    I_tilde = pg["I_tilde"]

    M_grad = kappa * I_tilde
    with np.errstate(divide="ignore", invalid="ignore"):
        v_halo2 = np.where(R > 0.0, M_grad / R, 0.0)

    v_model2 = v_b ** 2 + v_halo2
    v_model2 = np.maximum(v_model2, 0.0)
    v_model = np.sqrt(v_model2)

    if np.any(err_v <= 0.0):
        positive = err_v[err_v > 0.0]
        fallback = np.median(positive) if positive.size else 1.0
        sigma = np.where(err_v > 0.0, err_v, fallback)
    else:
        sigma = err_v

    resid = (v_obs - v_model) / sigma
    return float(np.sum(resid ** 2))


def fit_kappa_for_galaxy(pg):
    """
    One-dimensional fit of kappa for a single galaxy.

    Returns (kappa_star, chi2_star, chi2_baryons).
    """
    if not HAVE_SCIPY:
        grid = np.linspace(0.0, KAPPA_MAX, 201)
        chi_vals = [galaxy_chi2_kappa(pg, k) for k in grid]
        idx = int(np.argmin(chi_vals))
        k_star = float(grid[idx])
        chi_star = float(chi_vals[idx])
    else:
        res = minimize_scalar(
            lambda k: galaxy_chi2_kappa(pg, k),
            bounds=(0.0, KAPPA_MAX),
            method="bounded",
        )
        k_star = float(res.x)
        chi_star = float(res.fun)

    chi_bary = float(galaxy_chi2_kappa(pg, 0.0))
    return k_star, chi_star, chi_bary


def total_chi2_kappa(kappa, galaxies):
    """
    Total chi^2 across all galaxies for a given kappa.
    """
    return float(sum(galaxy_chi2_kappa(pg, kappa) for pg in galaxies))


def fit_global_kappa_total_chi2(galaxies):
    """
    Find kappa_global that minimises total chi^2 across all galaxies.
    """
    if not HAVE_SCIPY:
        grid = np.linspace(0.0, KAPPA_MAX, 401)
        chi_vals = [total_chi2_kappa(k, galaxies) for k in grid]
        idx = int(np.argmin(chi_vals))
        return float(grid[idx]), float(chi_vals[idx])

    res = minimize_scalar(
        lambda k: total_chi2_kappa(k, galaxies),
        bounds=(0.0, KAPPA_MAX),
        method="bounded",
    )
    return float(res.x), float(res.fun)


def build_kappa_grid():
    """
    Log10 grid of kappa values for the one-knob median chi^2 scan.
    """
    exps = np.linspace(KAPPA_GRID_MIN_EXP, KAPPA_GRID_MAX_EXP, KAPPA_GRID_N)
    return exps, 10.0 ** exps


def scan_median_chi2_grid_one_knob(galaxies, dof_i):
    """
    Scan log10 kappa grid, computing at each point:
        - total chi^2,
        - mean reduced chi^2,
        - median reduced chi^2.
    """
    log10_kappa_grid, kappa_grid = build_kappa_grid()
    n_gal = len(galaxies)

    chi2_total_grid = np.empty_like(kappa_grid)
    mean_chi2_red_grid = np.empty_like(kappa_grid)
    median_chi2_red_grid = np.empty_like(kappa_grid)

    for idx, kappa in enumerate(kappa_grid):
        chi2_i = np.array(
            [galaxy_chi2_kappa(pg, kappa) for pg in galaxies],
            dtype=float,
        )
        chi2_red_i = chi2_i / dof_i

        chi2_total_grid[idx] = np.sum(chi2_i)
        mean_chi2_red_grid[idx] = np.mean(chi2_red_i)
        median_chi2_red_grid[idx] = np.median(chi2_red_i)

    return (
        log10_kappa_grid,
        kappa_grid,
        chi2_total_grid,
        mean_chi2_red_grid,
        median_chi2_red_grid,
    )


def build_two_knob_grid():
    """
    Build 2D grid in (log10 K0, beta) for the dressed model.
    """
    log10_K0_grid = np.linspace(K0_GRID_MIN_EXP, K0_GRID_MAX_EXP, K0_GRID_N)
    K0_grid = 10.0 ** log10_K0_grid
    beta_grid = np.linspace(BETA_GRID_MIN, BETA_GRID_MAX, BETA_GRID_N)
    return log10_K0_grid, K0_grid, beta_grid


def scan_two_knob_grid(galaxies, dof_i, C_hat):
    """
    Scan a 2D grid in (K0, beta) for the dressed amplitude model

        kappa_eff,i = K0 * C_hat_i^beta.

    Returns:
        log10_K0_grid, K0_grid, beta_grid,
        chi2_total_two_grid, mean_chi2_red_two_grid,
        median_chi2_red_two_grid.
    """
    log10_K0_grid, K0_grid, beta_grid = build_two_knob_grid()
    n_beta = beta_grid.size
    n_K0 = K0_grid.size
    n_gal = len(galaxies)

    chi2_total_two = np.empty((n_beta, n_K0))
    mean_chi2_red_two = np.empty((n_beta, n_K0))
    median_chi2_red_two = np.empty((n_beta, n_K0))

    for ib, beta in enumerate(beta_grid):
        C_factor = C_hat ** beta
        for ik, K0 in enumerate(K0_grid):
            chi2_i = np.empty(n_gal, dtype=float)
            for g, pg in enumerate(galaxies):
                kappa_eff = K0 * C_factor[g]
                chi2_i[g] = galaxy_chi2_kappa(pg, kappa_eff)
            chi2_red_i = chi2_i / dof_i

            chi2_total_two[ib, ik] = np.sum(chi2_i)
            mean_chi2_red_two[ib, ik] = np.mean(chi2_red_i)
            median_chi2_red_two[ib, ik] = np.median(chi2_red_i)

    return (
        log10_K0_grid,
        K0_grid,
        beta_grid,
        chi2_total_two,
        mean_chi2_red_two,
        median_chi2_red_two,
    )


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------

def main():
    if not os.path.isdir(SPARC_DIR):
        raise SystemExit(
            f"SPARC directory not found: {SPARC_DIR}. "
            f"Place NPZ files under this path."
        )

    paths = sorted(glob.glob(os.path.join(SPARC_DIR, "*.npz")))
    if not paths:
        raise SystemExit(f"No NPZ files found under {SPARC_DIR}")

    raw_gals = []
    for path in paths:
        gal = load_sparc_galaxy(path)
        if gal is not None:
            raw_gals.append(gal)

    if not raw_gals:
        raise SystemExit("No suitable SPARC galaxies after cleaning")

    galaxies = []
    for gal in raw_gals:
        pg = preprocess_galaxy(gal)
        if pg is not None:
            galaxies.append(pg)

    if not galaxies:
        raise SystemExit("All galaxies failed preprocessing")

    print(f"Loaded {len(galaxies)} SPARC galaxies from {SPARC_DIR}")

    # ------------------------------------------------------------------
    # Per-galaxy one-knob fits
    # ------------------------------------------------------------------
    names = []
    n_points = []
    R_disc_list = []
    C_raw_list = []
    kappa_individual = []
    chi2_individual = []
    chi2_baryons = []

    for pg in galaxies:
        name = pg["name"]
        k_star, chi_star, chi_bary = fit_kappa_for_galaxy(pg)

        names.append(name)
        n_points.append(pg["R"].size)
        R_disc_list.append(pg["R_disc"])
        C_raw_list.append(pg["C_raw"])
        kappa_individual.append(k_star)
        chi2_individual.append(chi_star)
        chi2_baryons.append(chi_bary)

        print(
            f"{name:20s}  N={pg['R'].size:3d}  "
            f"R_disc={pg['R_disc']:.3f}  "
            f"kappa_i={k_star:.5f}  "
            f"chi2_i={chi_star:.2f}  "
            f"chi2_bary={chi_bary:.2f}"
        )

    names = np.array(names)
    n_points = np.array(n_points, dtype=int)
    R_disc_arr = np.array(R_disc_list, dtype=float)
    C_raw_arr = np.array(C_raw_list, dtype=float)
    kappa_individual = np.array(kappa_individual, dtype=float)
    chi2_individual = np.array(chi2_individual, dtype=float)
    chi2_baryons = np.array(chi2_baryons, dtype=float)

    # Degrees of freedom per galaxy
    dof_i = np.maximum(1, n_points - 1).astype(float)

    # ------------------------------------------------------------------
    # Global one-knob fit by total chi^2
    # ------------------------------------------------------------------
    kappa_global, chi2_global_total = fit_global_kappa_total_chi2(galaxies)
    chi2_bary_total = float(np.sum(chi2_baryons))
    dof_total = int(np.sum(dof_i))

    chi2_global_per_gal = np.array(
        [galaxy_chi2_kappa(pg, kappa_global) for pg in galaxies],
        dtype=float,
    )
    chi2_red_global_per_gal = chi2_global_per_gal / dof_i
    chi2_red_baryons = chi2_baryons / dof_i

    print("")
    print("Global Fisher parameter fit by total chi^2 (one knob):")
    print(f"  kappa_global           = {kappa_global:.6e}")
    print(f"  chi2_total(kappa*)     = {chi2_global_total:.2f}")
    print(f"  chi2_total(baryon)     = {chi2_bary_total:.2f}")
    print(f"  dof_total              = {dof_total:d}")
    print(f"  chi2_red_total(k*)     = {chi2_global_total / dof_total:.3f}")
    print(f"  chi2_red_total(bary)   = {chi2_bary_total / dof_total:.3f}")

    # Distribution of individual kappa_i
    finite_kappa = np.isfinite(kappa_individual) & (kappa_individual > 0.0)
    if np.any(finite_kappa):
        kappa_pos = kappa_individual[finite_kappa]
        logk = np.log10(kappa_pos)
        mean_logk = float(np.mean(logk))
        std_logk = float(np.std(logk))
        median_logk = float(np.median(logk))
        print("")
        print(
            "Individual kappa_i distribution "
            f"(log10 mean={mean_logk:.3f}, median={median_logk:.3f}, "
            f"std={std_logk:.3f})"
        )
    else:
        mean_logk = np.nan
        std_logk = np.nan
        median_logk = np.nan

    # ------------------------------------------------------------------
    # One-knob median reduced chi^2 scan
    # ------------------------------------------------------------------
    (
        log10_kappa_grid,
        kappa_grid,
        chi2_total_grid,
        mean_chi2_red_grid,
        median_chi2_red_grid,
    ) = scan_median_chi2_grid_one_knob(galaxies, dof_i)

    idx_star_1 = int(np.argmin(median_chi2_red_grid))
    kappa_med_star = float(kappa_grid[idx_star_1])
    median_chi2_red_star = float(median_chi2_red_grid[idx_star_1])
    mean_chi2_red_star = float(mean_chi2_red_grid[idx_star_1])

    chi2_med_per_gal = np.array(
        [galaxy_chi2_kappa(pg, kappa_med_star) for pg in galaxies],
        dtype=float,
    )
    chi2_red_med = chi2_med_per_gal / dof_i

    improvement_med = (chi2_red_baryons - chi2_red_med) / chi2_red_baryons
    frac_improved_10_med = float(np.mean(improvement_med > 0.10))

    improvement_global = (
        chi2_red_baryons - chi2_red_global_per_gal
    ) / chi2_red_baryons
    frac_improved_10_global = float(np.mean(improvement_global > 0.10))

    print("")
    print("Global Fisher parameter by median reduced chi^2 (one knob):")
    print(f"  kappa_med_star           = {kappa_med_star:.6e}")
    print(f"  median chi2_red(kappa*)  = {median_chi2_red_star:.3f}")
    print(f"  mean chi2_red(kappa*)    = {mean_chi2_red_star:.3f}")
    print(f"  median chi2_red(baryons) = {np.median(chi2_red_baryons):.3f}")
    print(f"  mean chi2_red(baryons)   = {np.mean(chi2_red_baryons):.3f}")
    print(
        f"  frac galaxies improved by >10 percent "
        f"(median kappa*)  = {frac_improved_10_med:.3f}"
    )
    print(
        f"  frac galaxies improved by >10 percent "
        f"(total-chi^2 kappa*) = {frac_improved_10_global:.3f}"
    )

    # ------------------------------------------------------------------
    # Compactness proxy and two-knob model
    # ------------------------------------------------------------------
    # Median R_disc for reference
    R_disc_median = float(np.median(R_disc_arr))
    print("")
    print(f"Median R_disc (baryon weighted)       = {R_disc_median:.3f}")

    # Normalise compactness
    mask_C = np.isfinite(C_raw_arr) & (C_raw_arr > 0.0)
    if np.any(mask_C):
        C_median = float(np.median(C_raw_arr[mask_C]))
        print(f"Median C_raw (M_b,max / R_disc^2)     = {C_median:.3e}")
    else:
        C_median = 1.0
    if C_median <= 0.0:
        C_median = 1.0

    C_hat = C_raw_arr / C_median
    print(f"Using C_hat = C_raw / {C_median:.3e} as dimensionless compactness")

    # Regression: log10 kappa_i vs log10 C_hat
    beta_reg = np.nan
    log10_K0_reg = np.nan
    r_reg = np.nan
    r2_reg = np.nan
    if np.any(finite_kappa & mask_C):
        reg_mask = finite_kappa & mask_C
        x = np.log10(C_hat[reg_mask])
        y = np.log10(kappa_individual[reg_mask])
        if x.size >= 3:
            slope, intercept = np.polyfit(x, y, 1)
            beta_reg = float(slope)
            log10_K0_reg = float(intercept)
            r_matrix = np.corrcoef(x, y)
            r_reg = float(r_matrix[0, 1])
            r2_reg = float(r_reg * r_reg)
            print("")
            print("Log-log regression of kappa_i vs compactness C_hat:")
            print(f"  beta_reg                         = {beta_reg:.3f}")
            print(f"  log10 K0_reg                     = {log10_K0_reg:.3f}")
            print(f"  K0_reg                           = {10.0 ** log10_K0_reg:.3e}")
            print(
                f"  Pearson r(log10 kappa, log10 C_hat) = {r_reg:.3f}"
            )
            print(f"  R^2                              = {r2_reg:.3f}")

    # Two-knob grid scan
    (
        log10_K0_grid,
        K0_grid,
        beta_grid,
        chi2_total_two_grid,
        mean_chi2_red_two_grid,
        median_chi2_red_two_grid,
    ) = scan_two_knob_grid(galaxies, dof_i, C_hat)

    idx_two = np.unravel_index(
        int(np.argmin(median_chi2_red_two_grid)),
        median_chi2_red_two_grid.shape,
    )
    ib_star, ik_star = idx_two
    beta_star = float(beta_grid[ib_star])
    K0_star = float(K0_grid[ik_star])
    log10_K0_star = float(log10_K0_grid[ik_star])

    median_chi2_red_two_star = float(median_chi2_red_two_grid[ib_star, ik_star])
    mean_chi2_red_two_star = float(mean_chi2_red_two_grid[ib_star, ik_star])

    # Per-galaxy at best two-knob point
    C_factor_star = C_hat ** beta_star
    kappa_two_star_per_gal = K0_star * C_factor_star

    chi2_two_star_per_gal = np.array(
        [
            galaxy_chi2_kappa(pg, kappa_two_star_per_gal[g])
            for g, pg in enumerate(galaxies)
        ],
        dtype=float,
    )
    chi2_red_two_star = chi2_two_star_per_gal / dof_i

    improvement_two_star = (
        chi2_red_baryons - chi2_red_two_star
    ) / chi2_red_baryons
    frac_improved_10_two = float(np.mean(improvement_two_star > 0.10))

    print("")
    print("Two-knob compactness-dressed Fisher model (median chi^2 minimum):")
    print(f"  beta_star                = {beta_star:.3f}")
    print(f"  log10 K0_star            = {log10_K0_star:.3f}")
    print(f"  K0_star                  = {K0_star:.3e}")
    print(f"  median chi2_red(two*)    = {median_chi2_red_two_star:.3f}")
    print(f"  mean chi2_red(two*)      = {mean_chi2_red_two_star:.3f}")
    print(f"  median chi2_red(baryons) = {np.median(chi2_red_baryons):.3f}")
    print(f"  mean chi2_red(baryons)   = {np.mean(chi2_red_baryons):.3f}")
    print(
        f"  frac galaxies improved by >10 percent "
        f"(two-knob best) = {frac_improved_10_two:.3f}"
    )

    # ------------------------------------------------------------------
    # Save detailed results
    # ------------------------------------------------------------------
    np.savez(
        OUT_NPZ,
        galaxy_name=names,
        n_points=n_points,
        R_disc=R_disc_arr,
        C_raw=C_raw_arr,
        C_hat=C_hat,
        R_disc_median=np.array([R_disc_median]),
        C_median=np.array([C_median]),
        kappa_individual=kappa_individual,
        chi2_individual=chi2_individual,
        chi2_baryons=chi2_baryons,
        chi2_global_per_gal=chi2_global_per_gal,
        chi2_red_baryons=chi2_red_baryons,
        chi2_red_global_per_gal=chi2_red_global_per_gal,
        chi2_med_per_gal=chi2_med_per_gal,
        chi2_red_med=chi2_red_med,
        kappa_global=np.array([kappa_global]),
        chi2_global_total=np.array([chi2_global_total]),
        chi2_bary_total=np.array([chi2_bary_total]),
        dof_total=np.array([dof_total]),
        kappa_med_star=np.array([kappa_med_star]),
        median_chi2_red_star=np.array([median_chi2_red_star]),
        mean_chi2_red_star=np.array([mean_chi2_red_star]),
        frac_improved_10pct_med=np.array([frac_improved_10_med]),
        frac_improved_10pct_global=np.array([frac_improved_10_global]),
        log10_kappa_grid=log10_kappa_grid,
        kappa_grid=kappa_grid,
        chi2_total_grid=chi2_total_grid,
        mean_chi2_red_grid=mean_chi2_red_grid,
        median_chi2_red_grid=median_chi2_red_grid,
        # Two-knob grids
        log10_K0_grid=log10_K0_grid,
        K0_grid=K0_grid,
        beta_grid=beta_grid,
        chi2_total_two_grid=chi2_total_two_grid,
        mean_chi2_red_two_grid=mean_chi2_red_two_grid,
        median_chi2_red_two_grid=median_chi2_red_two_grid,
        # Two-knob best point
        beta_star=np.array([beta_star]),
        log10_K0_star=np.array([log10_K0_star]),
        K0_star=np.array([K0_star]),
        kappa_two_star_per_gal=kappa_two_star_per_gal,
        chi2_two_star_per_gal=chi2_two_star_per_gal,
        chi2_red_two_star=chi2_red_two_star,
        frac_improved_10pct_two=np.array([frac_improved_10_two]),
        # Regression diagnostics
        beta_reg=np.array([beta_reg]),
        log10_K0_reg=np.array([log10_K0_reg]),
        r_reg=np.array([r_reg]),
        r2_reg=np.array([r2_reg]),
    )

    summary = {
        "n_galaxies": int(len(galaxies)),
        # one-knob totals
        "kappa_global": float(kappa_global),
        "chi2_global_total": float(chi2_global_total),
        "chi2_bary_total": float(chi2_bary_total),
        "dof_total": int(dof_total),
        "chi2_red_total_kappa": float(chi2_global_total / dof_total),
        "chi2_red_total_bary": float(chi2_bary_total / dof_total),
        "log10_kappa_mean": float(mean_logk),
        "log10_kappa_median": float(median_logk),
        "log10_kappa_std": float(std_logk),
        "kappa_med_star": float(kappa_med_star),
        "median_chi2_red_star": float(median_chi2_red_star),
        "mean_chi2_red_star": float(mean_chi2_red_star),
        "median_chi2_red_bary": float(np.median(chi2_red_baryons)),
        "mean_chi2_red_bary": float(np.mean(chi2_red_baryons)),
        "frac_improved_10pct_med": float(frac_improved_10_med),
        "frac_improved_10pct_global": float(frac_improved_10_global),
        # two-knob best
        "beta_star": float(beta_star),
        "log10_K0_star": float(log10_K0_star),
        "K0_star": float(K0_star),
        "median_chi2_red_two_star": float(median_chi2_red_two_star),
        "mean_chi2_red_two_star": float(mean_chi2_red_two_star),
        "frac_improved_10pct_two": float(frac_improved_10_two),
        # compactness and regression
        "R_disc_median": float(R_disc_median),
        "C_median": float(C_median),
        "beta_reg": float(beta_reg),
        "log10_K0_reg": float(log10_K0_reg),
        "r_reg": float(r_reg),
        "r2_reg": float(r2_reg),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("")
    print(f"Wrote summary JSON to {OUT_JSON}")
    print(f"Wrote detailed NPZ to {OUT_NPZ}")


if __name__ == "__main__":
    main()
