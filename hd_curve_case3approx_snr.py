"""
hd_curve_case3approx_snr.py
====================
Plots the full-curve HD SNR with Case 3 approximation vs r = P_gw(f_l)/P_n.

Uses the pair-by-pair diagonal approximation of C^{-1}.

For each pair (a,b), the contribution to rho^2 is (from Romano Case 3 inversion):

    rho^2_HD = sum_{a<b}  2 * Pgw^2 * (F*gamma_ab)^2
                          / [ (Pgw * F*gamma_ab)^2 + (Pgw + sigma^2)^2 ]

where F = 192*pi^3, gamma_ab = gamma_parallel(theta_ab) (raw, without F),
and r = P_gw / P_n is the x-axis variable.

Correct asymptotes:
  Weak signal (r -> 0):
    rho^2 -> 2 * F^2 * sum(gamma_ab^2) * r^2
    rho   -> sqrt(2) * F * sqrt(sum(gamma_ab^2)) * r

  Intermediate (r -> inf):
    rho^2 -> sum_{a<b}  2*(F*gamma_ab)^2 / ((F*gamma_ab)^2 + 1)
    rho   -> sqrt(sum_{a<b} 2*(F*gamma_ab)^2 / ((F*gamma_ab)^2 + 1))
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
    PHYSICAL_RATIO,
)

EPS          = 1e-14
ASTRO_FACTOR = 192.0 * np.pi**3


# ============================================================
#  ANALYTIC ASYMPTOTES
# ============================================================

def rho_hd_weak(gammas, r_values):
    """
    Weak-signal asymptote: rho^2 = 2 * F^2 * sum(gamma_ab^2) * r^2

    Derived from the correct HD formula in the limit Pgw << sigma^2:
      numer -> 2 * Pgw^2 * (F*g)^2
      denom -> sigma^4
      rho^2 -> 2 * (F*g*r)^2  (summed over pairs, using Pgw = r*P_n = r*sigma^2)
    """
    F = ASTRO_FACTOR
    sum_gamma2 = float(np.sum(gammas**2))
    return np.sqrt(2.0) * F * np.sqrt(sum_gamma2) * np.asarray(r_values, dtype=float)


def rho_hd_intermediate(gammas):
    """
    Intermediate-signal plateau: rho^2 = sum_{a<b} 2*(F*g)^2 / ((F*g)^2 + 1)
    """
    F  = ASTRO_FACTOR
    Fg = F * gammas
    rho_sq = float(np.sum(2.0 * Fg**2 / (Fg**2 + 1.0)))
    return np.sqrt(max(rho_sq, 0.0))


# ============================================================
#  SWEEP AND PLOT
# ============================================================

def plot_full_curve(gamma_matrix, n_r=400, save_path=None):
    """
    Sweep r = P_gw/P_n over a fixed range and plot rho_HD with asymptotes.

    The sweep [1e-13, 1e2] is fixed.
    """
    vals   = gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    gammas = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
    Np     = len(gammas)
    print(f"N_pairs used: {Np}")

    # Fixed sweep — physical ratio and transitions arise naturally
    r_values = np.logspace(-13, 2, n_r)

    print(f"Computing rho_HD for {n_r} r values ...")
    rho_full = rho_hd_full(r_values, gamma_matrix)
    print("Done.")

    rho_weak_vals = rho_hd_weak(gammas, r_values)
    rho_int_val   = rho_hd_intermediate(gammas)
    rho_int_line  = np.full_like(r_values, rho_int_val)

    F = ASTRO_FACTOR
    sum_gamma2 = float(np.sum(gammas**2))
    print(f"\nsum gamma_ab^2                            = {sum_gamma2:.4e}")
    print(f"F^2 * sum gamma_ab^2                       = {F**2 * sum_gamma2:.4e}")
    print(f"Weak-signal prefactor sqrt(2)*F*sqrt(sum)  = {np.sqrt(2.0)*F*np.sqrt(sum_gamma2):.4f}")
    print(f"Intermediate plateau rho_int               = {rho_int_val:.4f}")
    print(f"Physical r = P_gw/P_n                      = {PHYSICAL_RATIO:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(r_values, rho_full,
              color='steelblue', lw=2.5, label=r'$\rho_{\rm HD}$ (full curve)')
    ax.loglog(r_values, rho_weak_vals,
              color='grey', lw=1.5, ls='--',
              label=r'weak-signal asymptote')
    ax.loglog(r_values, rho_int_line,
              color='coral', lw=1.5, ls=':',
              label=f'intermediate plateau ({rho_int_val:.2f})')
    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho_{\rm HD}$',                 fontsize=13)
    ax.set_title('HD SNR Full Curve',                  fontsize=13)
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
                                       save_path="hd_curve_case3approx_snr.png")

    print("\nSelected output (r, rho_HD):")
    for rv, rhov in zip(r_vals[::80], rho_vals[::80]):
        print(f"  r = {rv:.2e}   rho_HD = {rhov:.4f}")