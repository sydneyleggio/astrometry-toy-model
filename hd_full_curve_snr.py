"""
hd_full_curve_snr.py
====================
Plots the full-curve HD SNR vs r = P_gw(f_l)/P_n.

Uses the pair-by-pair diagonal approximation of C^{-1}, which is:
  - Smooth across all regimes (no matrix inversion required)
  - Equivalent to treating each pair estimator independently
  - Consistent with Romano's approach for figure 3

For each pair (a,b), the contribution to rho^2 is:

    rho^2_HD = sum_{a<b}  P_gw^2 / [ P_gw^2 * gamma_ab^2
                                      + 2 * P_gw * gamma_ab * sigma^2 / (192pi^3)
                                      + sigma^4 / (192pi^3)^2 ]

where gamma_ab = gamma_parallel(theta_ab) is the (unnormalised) overlap function
and r = P_gw / P_n is the x-axis variable.

Limits:
  Weak signal (r -> 0):   rho ~ sqrt(2 * sum chi_ab^2) * r
  Intermediate (r -> inf): rho plateaus at sqrt(sum 1/chi_ab^2)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    build_star_positions,
    pairwise_theta,
    compute_ell_limits,
    gamma_parallel,
    rho_hd_full,
    STAR_COORDS_DEG,
    N_STARS,
    FIELD_SIZE_DEG,
    RANDOM_SEED,
    P_n,
    sigma_bar_sq,
)

EPS          = 1e-14
ASTRO_FACTOR = 192.0 * np.pi**3


# ============================================================
#  ANALYTIC ASYMPTOTES
# ============================================================

def rho_hd_weak(gammas, r_values):
    """
    Weak-signal limit: rho^2 = 2 * sum_ab gamma_ab^2 * r^2
    Matches Romano eq. 40 with gamma_ab -> gamma_parallel values.
    """
    sum_gamma2 = float(np.sum(gammas**2))
    return np.sqrt(2.0 * sum_gamma2) * r_values


def rho_hd_intermediate(gammas):
    """
    Intermediate-signal plateau: rho^2 = sum_ab 1/gamma_ab^2
    Matches Romano eq. 41.
    """
    return float(np.sqrt(np.sum(1.0 / gammas**2)))


# ============================================================
#  SWEEP AND PLOT
# ============================================================

def plot_full_curve(gamma_matrix, n_r=400, save_path=None):
    """
    Sweep r = P_gw/P_n and plot rho_HD with asymptotes.

    Uses rho_hd_full from main.py, which sums pair contributions directly.
    """
    # Extract valid off-diagonal gamma values
    vals   = gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    gammas = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
    Np     = len(gammas)
    print(f"N_pairs used: {Np}")

    r_values = np.logspace(-6, 2, n_r)

    print(f"Computing rho_HD for {n_r} r values ...")
    rho_full = rho_hd_full(r_values, gamma_matrix)
    print("Done.")

    rho_weak_vals = rho_hd_weak(gammas, r_values)
    rho_int_val   = rho_hd_intermediate(gammas)
    rho_int_line  = np.full_like(r_values, rho_int_val)

    sum_gamma2 = float(np.sum(gammas**2))
    print(f"\nsum gamma_ab^2                  = {sum_gamma2:.4f}")
    print(f"Weak-signal prefactor sqrt(2*sum) = {np.sqrt(2*sum_gamma2):.4f}")
    print(f"Intermediate plateau rho_int  = {rho_int_val:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(r_values, rho_full,
              color='steelblue', lw=2.5, label=r'$\rho_{\rm HD}$ (full curve)')
    ax.loglog(r_values, rho_int_line,
              color='coral', lw=1.5, ls=':',
              label=rf'intermediate plateau')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho_{\rm HD}$',                fontsize=13)
    ax.set_title('HD SNR Full Curve',              fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return r_values, rho_full


# ============================================================
#                        ENTRY POINT
# ============================================================

if __name__ == '__main__':

    stars_deg        = build_star_positions(
        STAR_COORDS_DEG, N_STARS, FIELD_SIZE_DEG, RANDOM_SEED)
    theta_mat        = pairwise_theta(stars_deg)
    ell_min, ell_max = compute_ell_limits(theta_mat, FIELD_SIZE_DEG)

    print(f"ell_min={ell_min}, ell_max={ell_max}, N_stars={N_STARS}")

    gamma = gamma_parallel(theta_mat, ell_min, ell_max)

    r_vals, rho_vals = plot_full_curve(gamma, n_r=400,
                                       save_path="hd_full_curve_snr.png")

    print("\nSelected output (r, rho_HD):")
    for rv, rhov in zip(r_vals[::80], rho_vals[::80]):
        print(f"  r = {rv:.2e}   rho_HD = {rhov:.4f}")