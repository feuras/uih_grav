#!/usr/bin/env python3
"""
20_bullet_cluster_toy_model.py

UIH Bullet test: 2D toy model with two compact galaxy clumps and
two extended gas clumps. Baryon surface density is
    Sigma_b = Sigma_gal + Sigma_gas
and the UIH halo proxy is
    Sigma_h = Sigma_b**2
so the effective lensing surface density is
    Sigma_eff = Sigma_b + Sigma_h.

We compare where the peaks of Sigma_eff lie relative to the galaxies
and the gas, and how large the effective surface density is at those
locations, as a function of gas mass fraction q and size ratio L.
"""

from pathlib import Path
import csv
import numpy as np


def make_grid(nx=512, ny=512, L=2.0):
    x = np.linspace(-L, L, nx)
    y = np.linspace(-L, L, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y, x, y


def gaussian2d(X, Y, x0, y0, sigma_x, sigma_y=None, mass=1.0):
    if sigma_y is None:
        sigma_y = sigma_x
    norm = mass / (2.0 * np.pi * sigma_x * sigma_y)
    return norm * np.exp(
        -(((X - x0) ** 2) / (2.0 * sigma_x**2)
          + ((Y - y0) ** 2) / (2.0 * sigma_y**2))
    )


def one_config(q=5.0, L=10.0, nx=512, ny=512):
    """
    q = M_gas / M_gal (total gas mass relative to total galaxy mass)
    L = sigma_gas / sigma_gal (size ratio)
    """
    X, Y, x_grid, y_grid = make_grid(nx=nx, ny=ny, L=2.0)

    # Geometry and scales (dimensionless)
    d = 0.8            # separation of galaxy clumps
    sigma_gal = 0.08   # compact size
    sigma_gas = L * sigma_gal

    M_gal = 1.0
    M_gas = q * M_gal

    # Two compact galaxy clumps at +/- d/2
    gal1 = gaussian2d(X, Y, -d / 2.0, 0.0, sigma_gal, mass=M_gal)
    gal2 = gaussian2d(X, Y,  d / 2.0, 0.0, sigma_gal, mass=M_gal)

    # Two gas blobs, broader and closer to the centre
    gas1 = gaussian2d(X, Y, -0.2, 0.0, sigma_gas, mass=M_gas / 2.0)
    gas2 = gaussian2d(X, Y,  0.2, 0.0, sigma_gas, mass=M_gas / 2.0)

    Sigma_gal = gal1 + gal2
    Sigma_gas = gas1 + gas2
    Sigma_b = Sigma_gal + Sigma_gas

    # UIH halo proxy: quadratic in baryon surface density
    Sigma_h = Sigma_b**2
    Sigma_eff = Sigma_b + Sigma_h

    # Examine a cut along the merger axis y = 0
    iy_mid = np.argmin(np.abs(y_grid))
    x_line = x_grid
    gal_line = Sigma_gal[iy_mid, :]
    gas_line = Sigma_gas[iy_mid, :]
    b_line = Sigma_b[iy_mid, :]
    eff_line = Sigma_eff[iy_mid, :]

    def peak_pos(arr, side="left"):
        if side == "left":
            mask = x_line < 0.0
        else:
            mask = x_line > 0.0
        xs = x_line[mask]
        ys = arr[mask]
        return xs[np.argmax(ys)]

    def value_at(arr, x0):
        i = np.argmin(np.abs(x_line - x0))
        return arr[i]

    # Left-hand side (the right is symmetric)
    x_gal = peak_pos(gal_line, "left")
    x_gas = peak_pos(gas_line, "left")
    x_b   = peak_pos(b_line, "left")
    x_eff = peak_pos(eff_line, "left")

    # Amplitudes at those locations
    b_at_gal   = value_at(b_line,   x_gal)
    b_at_gas   = value_at(b_line,   x_gas)
    eff_at_gal = value_at(eff_line, x_gal)
    eff_at_gas = value_at(eff_line, x_gas)

    return {
        "q": q,
        "L": L,
        "x_gal": float(x_gal),
        "x_gas": float(x_gas),
        "x_b_peak": float(x_b),
        "x_eff_peak": float(x_eff),
        "b_at_gal": float(b_at_gal),
        "b_at_gas": float(b_at_gas),
        "eff_at_gal": float(eff_at_gal),
        "eff_at_gas": float(eff_at_gas),
        "b_ratio_gal_over_gas": float(b_at_gal / b_at_gas),
        "eff_ratio_gal_over_gas": float(eff_at_gal / eff_at_gas),
    }


def main():
    out_dir = Path("results") / "bullet_toy"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bullet_toy_scan.csv"

    q_values = [1.0, 3.0, 5.0, 10.0]
    L_values = [5.0, 10.0, 20.0]

    rows = []
    for q in q_values:
        for L in L_values:
            print(f"[BulletToy] Running config q={q}, L={L} ...")
            res = one_config(q=q, L=L)
            rows.append(res)
            print(
                f"  Peaks: x_gal={res['x_gal']:.3f}, "
                f"x_gas={res['x_gas']:.3f}, "
                f"x_eff_peak={res['x_eff_peak']:.3f}"
            )
            print(
                f"  Sigma_eff(gal)/Sigma_eff(gas) "
                f"= {res['eff_ratio_gal_over_gas']:.2f}"
            )

    # Save summary table
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[BulletToy] Saved scan table to {out_csv}")


if __name__ == "__main__":
    main()
