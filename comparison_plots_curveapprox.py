"""
comparison_plots.py
===================
Produces two sets of comparison plots for advisor presentation:

  Plot set 1 — N_stars sweep (fixed 10-degree field):
    (a) rho_CP for N = 50, 100, 250, 500, 1000   [fast: no gamma recomputation]
    (b) rho_HD for N = 50, 100, 250, 500          [slower: gamma per N, skip 1000]

  Plot set 2 — Field size sweep (fixed N=100):
    (a) rho_CP for fields = 2, 5, 10, 40 degrees  [one gamma per field]
    (b) rho_HD for fields = 2, 5, 10, 40 degrees  [same gamma reused for HD]

Runtime notes:
  - CP N-sweep: rho_cp_full only needs (ell_min, ell_max, n_stars).
    gamma_parallel is computed ONCE for a representative field at each N, but
    only ell_min/ell_max are extracted from it for CP. Very fast.
  - HD N-sweep: rho_hd_full needs the full gamma_matrix. One gamma_parallel
    call per N value. N=1000 is skipped by default (can be enabled).
  - Field size sweep: one gamma_parallel call per field size, reused for both
    CP and HD.

All plots use the fixed sweep [1e-13, 1e2] and mark the physical ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    build_star_positions,
    pairwise_theta,
    compute_ell_limits,
    gamma_parallel_matrix,
    cp_single_star_gamma,
    rho_cp_full,
    rho_hd_full,
    FIELD_SIZE_DEG,
    RANDOM_SEED,
    PHYSICAL_RATIO,
)

# ── Shared sweep ──────────────────────────────────────────────────────────────
R_VALUES = np.logspace(-13, 2, 400)


# ============================================================
#  HELPER: compute geometry for a given (N, field_size_deg)
# ============================================================

def compute_geometry(n_stars, field_size_deg, seed=RANDOM_SEED, cp_only=False):
    """
    Build star positions, compute pairwise separations, ell limits,
    and gamma_matrix. Returns (ell_min, ell_max, gamma_matrix).

    cp_only=True: skips the gamma_matrix computation entirely and returns
    None for gamma_matrix. Use this when you only need ell_min/ell_max
    for rho_cp_full, saving significant time and memory for large N.

    For HD, gamma_parallel_matrix is used instead of gamma_parallel to
    avoid the O(ell_max * N^2) memory allocation that kills the process
    for N >= 500. It processes pairs in batches of 5000, keeping peak
    memory well under 200 MB regardless of N.
    """
    stars_deg        = build_star_positions(n_stars=n_stars,
                                            field_size_deg=field_size_deg,
                                            seed=seed)
    theta_mat        = pairwise_theta(stars_deg)
    ell_min, ell_max = compute_ell_limits(theta_mat, field_size_deg)

    if cp_only:
        return ell_min, ell_max, None

    gamma = gamma_parallel_matrix(theta_mat, ell_min, ell_max)
    return ell_min, ell_max, gamma


# ============================================================
#  PLOT 1a — rho_CP, N sweep
# ============================================================

def plot_cp_N_sweep(N_values, field_size_deg=FIELD_SIZE_DEG,
                   save_path=None):
    """
    rho_CP for multiple N values on the same axes.

    CP only depends on (ell_min, ell_max, n_stars), so gamma_parallel is
    computed once per N to extract ell limits, but the gamma_matrix itself
    is not used. This is fast even for large N.
    """
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_values)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for N, color in zip(N_values, colors):
        print(f"  CP N={N}...", end=" ", flush=True)
        t0 = time.time()
        ell_min, ell_max, _ = compute_geometry(N, field_size_deg, cp_only=True)
        rho = rho_cp_full(R_VALUES, ell_min, ell_max, n_stars=N)
        print(f"{time.time()-t0:.1f}s")
        ax.loglog(R_VALUES, rho, color=color, lw=2.0,
                  label=f'N = {N}')

    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho_{\rm CP}$',                 fontsize=13)
    ax.set_title(f'CP SNR — N sweep ({field_size_deg:.0f}° field)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")
    else:
        plt.show()


# ============================================================
#  PLOT 1b — rho_HD, N sweep
# ============================================================

def plot_hd_N_sweep(N_values, field_size_deg=FIELD_SIZE_DEG,
                   save_path=None):
    """
    rho_HD for multiple N values on the same axes.

    gamma_matrix must be recomputed for each N. N=1000 takes significantly
    longer and is noted in the docstring of compute_geometry.
    """
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(N_values)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for N, color in zip(N_values, colors):
        print(f"  HD N={N}...", end=" ", flush=True)
        t0 = time.time()
        _, _, gamma = compute_geometry(N, field_size_deg)
        rho = rho_hd_full(R_VALUES, gamma)
        print(f"{time.time()-t0:.1f}s")
        ax.loglog(R_VALUES, rho, color=color, lw=2.0,
                  label=f'N = {N}')

    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho_{\rm HD}$',                 fontsize=13)
    ax.set_title(f'HD SNR (diagonal approx.) — N sweep ({field_size_deg:.0f}° field)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")
    else:
        plt.show()


# ============================================================
#  PLOT 2a — rho_CP, field size sweep
# ============================================================

def plot_cp_field_sweep(field_sizes_deg, n_stars=100, save_path=None):
    """
    rho_CP for multiple field sizes on the same axes.

    Each field size requires one gamma_parallel call to extract ell limits.
    CP plateau (1/gamma0) changes with field size because ell_max changes.
    """
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(field_sizes_deg)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for field_deg, color in zip(field_sizes_deg, colors):
        print(f"  CP field={field_deg}°...", end=" ", flush=True)
        t0 = time.time()
        ell_min, ell_max, _ = compute_geometry(n_stars, field_deg, cp_only=True)
        rho = rho_cp_full(R_VALUES, ell_min, ell_max, n_stars=n_stars)
        gamma0 = cp_single_star_gamma(ell_min, ell_max)
        print(f"{time.time()-t0:.1f}s  (gamma0={gamma0:.5f})")
        ax.loglog(R_VALUES, rho, color=color, lw=2.0,
                  label=f'{field_deg}° field')

    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho_{\rm CP}$',                 fontsize=13)
    ax.set_title(f'CP SNR — field size sweep (N={n_stars})', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")
    else:
        plt.show()


# ============================================================
#  PLOT 2b — rho_HD, field size sweep
# ============================================================

def plot_hd_field_sweep(field_sizes_deg, n_stars=100, save_path=None):
    """
    rho_HD for multiple field sizes on the same axes.

    Each field size requires one gamma_parallel call. The gamma_matrix is
    reused for both CP and HD if you call both sweep functions with the
    same field sizes — see the combined runner below.
    """
    colors = plt.cm.autumn(np.linspace(0.1, 0.9, len(field_sizes_deg)))

    fig, ax = plt.subplots(figsize=(9, 5))

    for field_deg, color in zip(field_sizes_deg, colors):
        print(f"  HD field={field_deg}°...", end=" ", flush=True)
        t0 = time.time()
        _, _, gamma = compute_geometry(n_stars, field_deg)
        rho = rho_hd_full(R_VALUES, gamma)
        print(f"{time.time()-t0:.1f}s")
        ax.loglog(R_VALUES, rho, color=color, lw=2.0,
                  label=f'{field_deg}° field')

    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho_{\rm HD}$',                 fontsize=13)
    ax.set_title(f'HD SNR (diagonal approx.) — field size sweep (N={n_stars})',
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")
    else:
        plt.show()


# ============================================================
#  COMBINED: CP and HD on same axes (optional, for each sweep)
# ============================================================

def plot_combined_N_sweep(N_values, field_size_deg=FIELD_SIZE_DEG,
                          save_path=None):
    """
    CP and HD together for each N, on the same axes.
    CP is solid, HD is dashed, same color per N.
    Efficient: gamma computed once per N, used for both.
    """
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_values)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for N, color in zip(N_values, colors):
        print(f"  N={N}...", end=" ", flush=True)
        t0 = time.time()
        # cp_only=False: gamma_matrix needed for HD; uses batched computation
        ell_min, ell_max, gamma = compute_geometry(N, field_size_deg)
        rho_cp = rho_cp_full(R_VALUES, ell_min, ell_max, n_stars=N)
        rho_hd = rho_hd_full(R_VALUES, gamma)
        print(f"{time.time()-t0:.1f}s")
        ax.loglog(R_VALUES, rho_cp, color=color, lw=2.0, ls='-',
                  label=f'CP  N={N}')
        ax.loglog(R_VALUES, rho_hd, color=color, lw=2.0, ls='--',
                  label=f'HD  N={N}')

    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls=':',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho$',                           fontsize=13)
    ax.set_title(f'CP (solid) and HD (dashed) — N sweep ({field_size_deg:.0f}° field)',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")
    else:
        plt.show()


def plot_combined_field_sweep(field_sizes_deg, n_stars=100, save_path=None):
    """
    CP and HD together for each field size, on the same axes.
    Efficient: gamma computed once per field size, used for both.
    """
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(field_sizes_deg)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for field_deg, color in zip(field_sizes_deg, colors):
        print(f"  field={field_deg}°...", end=" ", flush=True)
        t0 = time.time()
        ell_min, ell_max, gamma = compute_geometry(n_stars, field_deg)
        rho_cp = rho_cp_full(R_VALUES, ell_min, ell_max, n_stars=n_stars)
        rho_hd = rho_hd_full(R_VALUES, gamma)
        print(f"{time.time()-t0:.1f}s")
        ax.loglog(R_VALUES, rho_cp, color=color, lw=2.0, ls='-',
                  label=f'CP  {field_deg}°')
        ax.loglog(R_VALUES, rho_hd, color=color, lw=2.0, ls='--',
                  label=f'HD  {field_deg}°')

    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls=':',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho$',                           fontsize=13)
    ax.set_title(f'CP (solid) and HD (dashed) — field sweep (N={n_stars})',
                 fontsize=13)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved to {save_path}")
    else:
        plt.show()


# ============================================================
#                        ENTRY POINT
# ============================================================

if __name__ == '__main__':

    # ── N sweep settings ─────────────────────────────────────
    # N=1000 is commented out — enable if you have time (~10+ min for HD)
    N_VALUES = [50, 100, 250, 500]
    # N_VALUES = [50, 100, 250, 500, 1000]  # uncomment to include N=1000

    # ── Field size sweep settings ─────────────────────────────
    FIELD_SIZES_DEG = [2, 5, 10, 40]
    N_FIXED         = 100

    print("=" * 50)
    print("PLOT 1: N sweep — CP only")
    print("=" * 50)
    plot_cp_N_sweep(N_VALUES, save_path="compare_cp_N_sweep.png")

    print()
    print("=" * 50)
    print("PLOT 2: N sweep — HD only")
    print("=" * 50)
    plot_hd_N_sweep(N_VALUES, save_path="compare_hd_N_sweep.png")

    print()
    print("=" * 50)
    print("PLOT 3: N sweep — CP and HD combined")
    print("=" * 50)
    plot_combined_N_sweep(N_VALUES, save_path="compare_combined_N_sweep.png")

    print()
    print("=" * 50)
    print("PLOT 4: Field size sweep — CP only")
    print("=" * 50)
    plot_cp_field_sweep(FIELD_SIZES_DEG, n_stars=N_FIXED,
                        save_path="compare_cp_field_sweep.png")

    print()
    print("=" * 50)
    print("PLOT 5: Field size sweep — HD only")
    print("=" * 50)
    plot_hd_field_sweep(FIELD_SIZES_DEG, n_stars=N_FIXED,
                        save_path="compare_hd_field_sweep.png")

    print()
    print("=" * 50)
    print("PLOT 6: Field size sweep — CP and HD combined")
    print("=" * 50)
    plot_combined_field_sweep(FIELD_SIZES_DEG, n_stars=N_FIXED,
                              save_path="compare_combined_field_sweep.png")