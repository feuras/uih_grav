#!/usr/bin/env python3
"""
30_grav_sparc_bps_saturation_scan.py

BPS saturation diagnostics across the SPARC compactness scan.

This script takes the SPARC baryonic NPZ files and the output of
29_grav_sparc_two_knob_universality_scan.py and computes, for each
galaxy, the "BPS amplitude"

    kappa_BPS,i(R_ref) = M_b(R_ref) / I_tilde(R_ref),

at three reference radii:

    R_ref = R_disc          (baryon weighted disc radius)
    R_ref = 2.2 * R_disc    (approximate disc peak)
    R_ref = R_max           (last measured radius)

Here

    M_b(R)        = R * v_b(R)^2
    I_tilde(R)    = I(R) / R_disc^2
    I(R)          = integral_0^R M_b(s)^2 ds

as in the universality scan.

For each reference radius and each galaxy, the script then forms three
dimensionless BPS ratios:

    q_ind  = kappa_individual / kappa_BPS
    q_two  = kappa_two_star_per_gal / kappa_BPS
    q_glob = kappa_global / kappa_BPS

where

    kappa_individual        is the per galaxy best fit from the one knob fits,
    kappa_two_star_per_gal  is the compactness dressed amplitude at the
                             best two knob point (K0_star, beta_star),
    kappa_global            is the global single amplitude.

It reports summary statistics (mean, median, standard deviation of
log10 q, fractions within a factor 3 and 10 of unity), their correlation
with compactness C_hat, and writes an NPZ and JSON for further analysis.

Inputs:

    - SPARC NPZ files under ./data/sparc_npz
    - sparc_kappa_universality_results.npz
    - sparc_kappa_universality_summary.json

Outputs:

    - NPZ:   sparc_bps_saturation_results.npz
    - JSON:  sparc_bps_saturation_summary.json

"""

import os
import glob
import json
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

SPARC_DIR = os.path.join("data", "sparc_npz")

RESULTS_NPZ = "sparc_kappa_universality_results.npz"
RESULTS_JSON = "sparc_kappa_universality_summary.json"

OUT_NPZ = "sparc_bps_saturation_results.npz"
OUT_JSON = "sparc_bps_saturation_summary.json"

MIN_POINTS = 6


# ----------------------------------------------------------------------
# Helpers: loading and preprocessing (copied from 29_ script for consistency)
# ----------------------------------------------------------------------

def baryon_scale_radius(R, v_b):
    """
    Baryon weighted mean radius:

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
      R_disc, I_tilde(R), M_b(R), R_max.
    """
    R = gal["R"]
    v_b = gal["v_b"]

    R_disc = baryon_scale_radius(R, v_b)
    if not np.isfinite(R_disc) or R_disc <= 0.0:
        return None

    # Pseudo enclosed mass M_b(R) = R v_b^2
    M_b = R * (v_b ** 2)

    # Fisher integral I(R) = integral_0^R M_b(s)^2 ds
    integrand = M_b ** 2
    dR = np.diff(R)
    avg = 0.5 * (integrand[1:] + integrand[:-1])
    I = np.concatenate(([0.0], np.cumsum(avg * dR)))

    I_tilde = I / (R_disc ** 2)

    R_max = float(R[-1])

    out = dict(gal)
    out["R_disc"] = float(R_disc)
    out["M_b"] = M_b.astype(float)
    out["I_tilde"] = I_tilde.astype(float)
    out["R_max"] = R_max
    return out


def interpolate_at_radius(R, y, R_ref):
    """
    Simple linear interpolation y(R_ref) given arrays R and y.

    If R_ref is outside the data range, we clip to the boundary.
    """
    if not np.isfinite(R_ref):
        return np.nan
    if R_ref <= R[0]:
        return float(y[0])
    if R_ref >= R[-1]:
        return float(y[-1])
    return float(np.interp(R_ref, R, y))


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------

def main():
    # Check input files
    if not os.path.isdir(SPARC_DIR):
        raise SystemExit(
            f"SPARC directory not found: {SPARC_DIR}. "
            f"Place NPZ files under this path."
        )

    if not os.path.isfile(RESULTS_NPZ):
        raise SystemExit(
            f"Results NPZ not found: {RESULTS_NPZ}. "
            f"Run 29_grav_sparc_two_knob_universality_scan.py first."
        )

    if not os.path.isfile(RESULTS_JSON):
        raise SystemExit(
            f"Results JSON not found: {RESULTS_JSON}. "
            f"Run 29_grav_sparc_two_knob_universality_scan.py first."
        )

    # Load the universality results
    res = np.load(RESULTS_NPZ, allow_pickle=True)
    names_res = res["galaxy_name"]
    kappa_individual = res["kappa_individual"]
    kappa_two_star_per_gal = res["kappa_two_star_per_gal"]
    kappa_global_arr = res["kappa_global"]
    C_hat = res["C_hat"]

    if kappa_global_arr.size != 1:
        raise SystemExit("Expected single kappa_global in results NPZ.")
    kappa_global = float(kappa_global_arr[0])

    # Load JSON summary for K0_star, beta_star (not strictly needed,
    # but useful to keep around)
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    beta_star = float(summary["beta_star"])
    K0_star = float(summary["K0_star"])
    log10_K0_star = float(summary["log10_K0_star"])

    print("Loaded universality results:")
    print(f"  n_galaxies           = {int(summary['n_galaxies'])}")
    print(f"  kappa_global         = {kappa_global:.6e}")
    print(f"  beta_star            = {beta_star:.3f}")
    print(f"  K0_star              = {K0_star:.3e} (log10 K0_star = {log10_K0_star:.3f})")

    # Load SPARC galaxies in the same order as the universality script:
    # sorted by file name, then cleaned and preprocessed.
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

    if len(galaxies) != names_res.size:
        print(
            f"Warning: preprocessed galaxy count {len(galaxies)} "
            f"does not match results count {names_res.size}."
        )

    # We assume the ordering is consistent: universality script also
    # used sorted(glob) and then kept galaxies that passed cleaning.
    # For safety, we check names.
    names_local = [pg["name"] for pg in galaxies]
    mismatch = [
        (i, names_local[i], str(names_res[i]))
        for i in range(min(len(names_local), names_res.size))
        if names_local[i] != str(names_res[i])
    ]
    if mismatch:
        print("Warning: galaxy name mismatch at indices:")
        for idx, n_loc, n_res in mismatch[:10]:
            print(f"  idx {idx}: local='{n_loc}', results='{n_res}'")
        print("Proceeding, but inspect ordering if results look odd.")

    n_gal = len(galaxies)
    print(f"Using {n_gal} SPARC galaxies for BPS saturation analysis.")

    # Reference radius choices for each galaxy
    R_disc_arr = np.array([pg["R_disc"] for pg in galaxies], dtype=float)
    R_max_arr = np.array([pg["R_max"] for pg in galaxies], dtype=float)
    R_2p2_arr = 2.2 * R_disc_arr

    # Pack into a dict of reference radius arrays
    R_ref_kinds = {
        "Rdisc": R_disc_arr,
        "2p2Rdisc": R_2p2_arr,
        "Rmax": R_max_arr,
    }

    # Storage for kappa_BPS and q ratios
    kappa_BPS = {key: np.full(n_gal, np.nan, dtype=float)
                 for key in R_ref_kinds.keys()}
    q_ind = {key: np.full(n_gal, np.nan, dtype=float)
             for key in R_ref_kinds.keys()}
    q_two = {key: np.full(n_gal, np.nan, dtype=float)
             for key in R_ref_kinds.keys()}
    q_glob = {key: np.full(n_gal, np.nan, dtype=float)
              for key in R_ref_kinds.keys()}

    # Compute kappa_BPS and q ratios
    for i, pg in enumerate(galaxies):
        R = pg["R"]
        v_b = pg["v_b"]
        M_b = pg["M_b"]
        I_tilde = pg["I_tilde"]

        # Cheap guard
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(I_tilde)):
            continue

        for key, R_ref_arr in R_ref_kinds.items():
            R_ref = float(R_ref_arr[i])
            if not np.isfinite(R_ref) or R_ref <= 0.0:
                continue

            # Interpolate M_b and I_tilde at R_ref
            M_b_ref = interpolate_at_radius(R, M_b, R_ref)
            I_tilde_ref = interpolate_at_radius(R, I_tilde, R_ref)

            if not np.isfinite(M_b_ref) or not np.isfinite(I_tilde_ref):
                continue
            if I_tilde_ref <= 0.0:
                continue

            k_BPS = M_b_ref / I_tilde_ref
            kappa_BPS[key][i] = k_BPS

            if k_BPS <= 0.0 or not np.isfinite(k_BPS):
                continue

            k_ind = float(kappa_individual[i])
            k_two = float(kappa_two_star_per_gal[i])

            q_ind[key][i] = k_ind / k_BPS
            q_two[key][i] = k_two / k_BPS
            q_glob[key][i] = kappa_global / k_BPS

    # Helper to compute summary stats for q arrays
    def summarise_q(q_arr, C_hat_arr, label, R_kind):
        mask = np.isfinite(q_arr) & (q_arr > 0.0)
        if not np.any(mask):
            return None

        q_good = q_arr[mask]
        C_good = C_hat_arr[mask]

        logq = np.log10(q_good)
        mean_logq = float(np.mean(logq))
        median_logq = float(np.median(logq))
        std_logq = float(np.std(logq))

        frac_within_3 = float(np.mean((q_good > 1.0 / 3.0) & (q_good < 3.0)))
        frac_within_10 = float(np.mean((q_good > 0.1) & (q_good < 10.0)))

        # Correlation of log10 q with log10 C_hat
        mask_corr = np.isfinite(C_good) & (C_good > 0.0)
        if np.any(mask_corr):
            x = np.log10(C_good[mask_corr])
            y = logq[mask_corr]
            if x.size >= 3:
                r_mat = np.corrcoef(x, y)
                r = float(r_mat[0, 1])
            else:
                r = np.nan
        else:
            r = np.nan

        print("")
        print(f"BPS ratios for {label} at {R_kind}:")
        print(f"  N_used                 = {q_good.size}")
        print(f"  mean log10 q           = {mean_logq:.3f}")
        print(f"  median log10 q         = {median_logq:.3f}")
        print(f"  std log10 q            = {std_logq:.3f}")
        print(f"  frac within factor 3   = {frac_within_3:.3f}")
        print(f"  frac within factor 10  = {frac_within_10:.3f}")
        print(f"  corr(log10 q, log10 C) = {r:.3f}")

        return {
            "N_used": int(q_good.size),
            "mean_log10_q": mean_logq,
            "median_log10_q": median_logq,
            "std_log10_q": std_logq,
            "frac_within_3": frac_within_3,
            "frac_within_10": frac_within_10,
            "r_logq_logC": r,
        }

    # Build JSON summary
    summary_out = {
        "description": "BPS saturation diagnostics for SPARC Fisher scalar halos.",
        "R_ref_kinds": list(R_ref_kinds.keys()),
        "beta_star": beta_star,
        "K0_star": K0_star,
        "log10_K0_star": log10_K0_star,
    }

    for R_kind in R_ref_kinds.keys():
        key_block = {}
        key_block["kappa_BPS_valid"] = int(np.sum(np.isfinite(kappa_BPS[R_kind]) & (kappa_BPS[R_kind] > 0.0)))

        stats_ind = summarise_q(q_ind[R_kind], C_hat, "kappa_individual", R_kind)
        stats_two = summarise_q(q_two[R_kind], C_hat, "kappa_two_star", R_kind)
        stats_glob = summarise_q(q_glob[R_kind], C_hat, "kappa_global", R_kind)

        key_block["q_individual"] = stats_ind
        key_block["q_two_star"] = stats_two
        key_block["q_global"] = stats_glob

        summary_out[R_kind] = key_block

    # Save NPZ with detailed arrays
    np.savez(
        OUT_NPZ,
        galaxy_name=names_res,
        R_disc=R_disc_arr,
        R_max=R_max_arr,
        R_2p2=R_2p2_arr,
        C_hat=C_hat,
        kappa_individual=kappa_individual,
        kappa_two_star_per_gal=kappa_two_star_per_gal,
        kappa_global=np.array([kappa_global]),
        kappa_BPS_Rdisc=kappa_BPS["Rdisc"],
        kappa_BPS_2p2Rdisc=kappa_BPS["2p2Rdisc"],
        kappa_BPS_Rmax=kappa_BPS["Rmax"],
        q_ind_Rdisc=q_ind["Rdisc"],
        q_ind_2p2Rdisc=q_ind["2p2Rdisc"],
        q_ind_Rmax=q_ind["Rmax"],
        q_two_Rdisc=q_two["Rdisc"],
        q_two_2p2Rdisc=q_two["2p2Rdisc"],
        q_two_Rmax=q_two["Rmax"],
        q_glob_Rdisc=q_glob["Rdisc"],
        q_glob_2p2Rdisc=q_glob["2p2Rdisc"],
        q_glob_Rmax=q_glob["Rmax"],
    )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    print("")
    print(f"Wrote BPS saturation summary JSON to {OUT_JSON}")
    print(f"Wrote detailed NPZ to {OUT_NPZ}")

    # Optional quick diagnostic plot if matplotlib is available
    if HAVE_MPL:
        # Example: scatter of log10 q_two vs log10 C_hat at Rdisc
        mask = (
            np.isfinite(q_two["Rdisc"])
            & (q_two["Rdisc"] > 0.0)
            & np.isfinite(C_hat)
            & (C_hat > 0.0)
        )
        if np.any(mask):
            x = np.log10(C_hat[mask])
            y = np.log10(q_two["Rdisc"][mask])

            plt.figure()
            plt.scatter(x, y, s=10, alpha=0.6)
            plt.axhline(0.0, ls="--")
            plt.xlabel("log10 C_hat")
            plt.ylabel("log10 q_two (R_disc)")
            plt.title("BPS ratio q_two at R_disc vs compactness")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
