"""
cp_full_curve_snr.py
====================
Plots the full-curve CP SNR vs r = P_gw(f_l)/P_n.

Uses the Sherman-Morrison inversion of the N x N auto-correlation covariance.

The full CP SNR formula is:

    rho^2_CP = N*(F*r)^2 / [1 + 2*F*gamma0*r + N*(F*gamma0*r)^2]

where F = 192*pi^3, gamma0 = Gamma_o(0) (single-star self-overlap),
and r = P_gw/P_n is the x-axis variable.
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
    cp_single_star_gamma,
    rho_cp_full,
    rho_cp_weak,
    rho_cp_intermediate,
    STAR_COORDS_DEG,
    N_STARS,
    FIELD_SIZE_DEG,
    RANDOM_SEED,
    PHYSICAL_RATIO,
)

EPS = 1e-14


# ============================================================
#  SWEEP AND PLOT
# ============================================================

def plot_full_curve(ell_min, ell_max, n_r=400, save_path=None):
    """
    Sweep r = P_gw/P_n over a fixed range and plot rho_CP with asymptotes.

    The x range [1e-13, 1e2] is fixed.
    """
    gamma0   = cp_single_star_gamma(ell_min, ell_max)
    rho_plat = rho_cp_intermediate(gamma0)

    print(f"gamma0                = {gamma0:.6f}")
    print(f"CP intermediate plateau = {rho_plat:.4f}  (= 1/gamma0, independent of N)")
    print(f"CP weak-signal slope    = sqrt(N)*F = {np.sqrt(N_STARS) * 192.0*np.pi**3:.4e}")
    print(f"Physical r = P_gw/P_n = {PHYSICAL_RATIO:.3e}")

    r_values = np.logspace(-13, 2, n_r)

    print(f"\nComputing rho_CP for {n_r} r values ...")
    rho_full      = rho_cp_full(r_values, ell_min, ell_max)
    rho_weak_vals = rho_cp_weak(r_values)
    rho_int_line  = np.full_like(r_values, rho_plat)
    print("Done.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(r_values, rho_full,
              color='C0', lw=2.5, label=r'$\rho_{\rm CP}$ (full curve)')
    ax.loglog(r_values, rho_weak_vals,
              color='grey', lw=1.5, ls='--',
              label=r'weak-signal asymptote')
    ax.loglog(r_values, rho_int_line,
              color='coral', lw=1.5, ls=':',
              label=f'intermediate plateau ({rho_plat:.1f})')
    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho_{\rm CP}$',                 fontsize=13)
    ax.set_title('CP SNR Full Curve',                  fontsize=13)
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

    r_vals, rho_vals = plot_full_curve(ell_min, ell_max, n_r=400,
                                       save_path="cp_full_curve_snr.png")

    print("\nSelected output (r, rho_CP):")
    for rv, rhov in zip(r_vals[::80], rho_vals[::80]):
        print(f"  r = {rv:.2e}   rho_CP = {rhov:.4f}")
