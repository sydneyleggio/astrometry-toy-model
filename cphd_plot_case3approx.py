"""
cphd_plot_case3approx.py
=============
Plots CP and HD SNR curves on the same axes vs P_gw(f_l)/P_n.

Uses the formulas already in main.py:
  - rho_cp_full: full CP curve via Sherman-Morrison (smooth, exact)
  - rho_hd_full: full HD Case 3 approximation curve summing pair-by-pair (smooth, diagonal approx of C^{-1})

Both use the same x-axis: r = P_gw(f_l) / P_n.
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
    rho_cp_full,
    rho_hd_full,
    STAR_COORDS_DEG,
    N_STARS,
    FIELD_SIZE_DEG,
    RANDOM_SEED,
    P_n,
    sigma_bar_sq,
)
import astropy.units as u

# ── Build star field ─────────────────────────────────────────────────────────
stars_deg        = build_star_positions(STAR_COORDS_DEG, N_STARS, FIELD_SIZE_DEG, RANDOM_SEED)
theta_mat        = pairwise_theta(stars_deg)
ell_min, ell_max = compute_ell_limits(theta_mat, FIELD_SIZE_DEG)

print(f"ell_min={ell_min}, ell_max={ell_max}, N_stars={N_STARS}")

gamma = gamma_parallel(theta_mat, ell_min, ell_max)
print("Gamma matrix computed.")

# ── Sweep r = P_gw / P_n ─────────────────────────────────────────────────────
r_values = np.logspace(-6, 2, 400)   # wide range to capture both regimes
P_gw_arr = r_values * P_n            # physical P_gw values

print("Computing CP full curve...")
rho_cp = rho_cp_full(r_values, ell_min, ell_max)

print("Computing HD full curve...")
rho_hd = rho_hd_full(r_values, gamma)

# ── Weak-signal asymptotes (analytic) ────────────────────────────────────────
ASTRO  = 192.0 * np.pi**3
factor = ASTRO

# CP weak: rho^2 = N * (192pi^3)^2 * P_gw^2 / sigma^4
rho_cp_weak = np.sqrt(N_STARS) * factor * r_values   # linear in r

# HD weak: rho^2 = 2 * sum_ab chi_ab^2 * P_gw^2 / sigma^4
vals = gamma[np.triu_indices_from(gamma, k=1)]
gammas = vals[np.isfinite(vals) & (np.abs(vals) > 1e-14)]
sum_chi2 = float(np.sum(gammas**2))
rho_hd_weak = np.sqrt(2.0 * sum_chi2) * r_values

print(f"\nsum gamma_ab^2 = {sum_chi2:.4f}")
print(f"HD weak prefactor sqrt(2*sum) = {np.sqrt(2*sum_chi2):.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(r_values, rho_cp,      color='C0', lw=2.5, label=r'$\rho_{\rm CP}$ (full curve)')
ax.loglog(r_values, rho_hd,      color='C1', lw=2.5, label=r'$\rho_{\rm HD}$ (full curve)')

ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
ax.set_ylabel(r'$\rho$',                      fontsize=13)
ax.set_title('CP and HD SNR Full Curves',   fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('cp_hd_full_curves.png', dpi=150)
print("\nFigure saved to cp_hd_full_curves.png")