"""
gamma_tilde_ab_sum.py
==================
Computes the sum  Σ_{a<b} [Γ̃''_{ab}]²  over all distinct star pairs,
where Γ̃''_{ab} = Γ₀''(Θ_{ab}) is the normalised astrometric overlap function
for GR tensor modes (parallel component).

The sum is truncated at ℓ_min = 2 and ℓ_max = floor(2π / Θ_min),
where Θ_min is the MINIMUM pairwise angular separation in the catalogue
(sets the resolution limit of the survey).

Note: if two stars happen to be extremely close (Θ_min ≪ 1°),
ℓ_max can become very large and computation slow. Use the theta_min_floor_deg
parameter to cap this (e.g. set it to your instrument's resolution limit).

Usage
-----
1. Random catalogue (demo):
       python gamma_tilde_sum.py

2. From a sky-position file:
       python gamma_tilde_sum.py my_stars.txt
   The file should have two columns: RA (deg)  Dec (deg), one star per line.

Output
------
  - Printed summary (N stars, ℓ range, sum, mean, plot path)
  - gamma_tilde_results.png  — Γ₀''(Θ) curve + pair histogram
"""

import sys
import numpy as np
from scipy.special import lpmv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Kernel functions (vectorised over cos_theta array)
# ─────────────────────────────────────────────────────────────────────────────

def gamma0_parallel_vectorised(cos_theta_arr, ell_min, ell_max_int):
    """
    Unnormalised Γ₀''(Θ) for GR tensor modes, vectorised over cos_theta.
    w_ℓ = 1/[ℓ(ℓ+1)(ℓ-1)(ℓ+2)]  proportional to C_ℓ^{EE} = C_ℓ^{BB}.
    """
    ct = np.clip(np.asarray(cos_theta_arr, dtype=float), -1 + 1e-10, 1 - 1e-10)
    sin_t = np.sqrt(1.0 - ct**2)
    sin_t = np.where(sin_t < 1e-10, 1e-10, sin_t)
    result = np.zeros_like(ct)
    for ell in range(ell_min, ell_max_int + 1):
        w    = 1.0 / (ell * (ell + 1) * (ell - 1) * (ell + 2))
        pref = (2 * ell + 1) / (4 * np.pi) * w
        P2   = lpmv(2, ell, ct)
        P0   = lpmv(0, ell, ct)
        P1   = lpmv(1, ell, ct)
        g1   = -0.5 * (P2 / (ell * (ell + 1)) - P0)
        g2   = -P1 / (ell * (ell + 1) * sin_t)
        result += pref * (g1 + g2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sky positions → unit vectors
# ─────────────────────────────────────────────────────────────────────────────

def radec_to_uvec(ra_deg, dec_deg):
    """Convert (RA, Dec) in degrees to unit 3-vector."""
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_gamma_sum(ra_deg, dec_deg, ell_min=2,
                      theta_min_floor_rad=None, verbose=True):
    """
    Compute Σ_{a<b} [Γ₀''(Θ_{ab})]² for a catalogue of stars.

    Parameters
    ----------
    ra_deg, dec_deg : array-like, shape (N,)
        Star positions in degrees.
    ell_min : int
        Minimum multipole (default 2).
    theta_min_floor_rad : float or None
        If set, enforces a minimum angular separation (in radians) for the
        ℓ_max calculation, capping ℓ_max when two stars are very close.
        Set this to your instrument's angular resolution in radians.
    verbose : bool
        Print progress and results.

    Returns
    -------
    dict with keys:
        'sum_gamma2'    : float  — Σ [Γ₀'']²
        'mean_gamma2'   : float  — mean per pair
        'ell_max'       : int    — ℓ_max used
        'theta_min_rad' : float  — min pairwise separation in radians (sets ℓ_max)
        'theta_max_rad' : float  — max pairwise separation in radians
        'pair_thetas'   : array  — all pairwise separations in radians
        'pair_gamma'    : array  — Γ₀''(Θ) for each pair (normalised to 1 at Θ=0)
        'norm'          : float  — divisor used for normalisation
    """
    ra_deg  = np.asarray(ra_deg,  dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    N = len(ra_deg)
    assert N == len(dec_deg), "ra and dec must have the same length"

    # Unit vectors and pairwise dot products
    uvec = radec_to_uvec(ra_deg, dec_deg)          # (N, 3)
    dots = np.clip(uvec @ uvec.T, -1.0, 1.0)       # (N, N)

    # Minimum separation = largest off-diagonal dot product
    d_off = dots.copy()
    np.fill_diagonal(d_off, -2.0)                  # exclude self (dot=1)
    theta_min_rad_actual = np.arccos(np.clip(np.max(d_off), -1, 1))

    # Maximum separation = smallest off-diagonal dot product
    np.fill_diagonal(d_off, 2.0)
    theta_max_rad = np.arccos(np.clip(np.min(d_off), -1, 1))

    # Apply floor if requested
    if theta_min_floor_rad is not None and theta_min_rad_actual < theta_min_floor_rad:
        theta_min_rad = theta_min_floor_rad
        floored = True
    else:
        theta_min_rad = theta_min_rad_actual
        floored = False

    # ℓ_max from MINIMUM separation
    ell_max_float = 2 * np.pi / theta_min_rad
    ell_max_int   = int(np.floor(ell_max_float))

    if verbose:
        print(f"\n{'─'*62}")
        print(f"  N stars          : {N}")
        print(f"  N pairs          : {N*(N-1)//2}")
        print(f"  Θ_min (actual)   : {theta_min_rad_actual:.6f} rad"
              + (f"  → floored to {theta_min_floor_rad:.6f} rad" if floored else ""))
        print(f"  Θ_max            : {theta_max_rad:.6f} rad")
        print(f"  ℓ_min            : {ell_min}")
        print(f"  ℓ_max            : {ell_max_float:.2f}  →  integer {ell_max_int}")

    if ell_max_int < ell_min:
        raise ValueError(
            f"ℓ_max ({ell_max_int}) < ℓ_min ({ell_min}). "
            f"Θ_min is too large. Check your catalogue or ℓ definition."
        )

    # Normalization: Γ₀''(Θ → 0) = 1  (astrometric convention — no pulsar/source term)
    norm = gamma0_parallel_vectorised(np.array([1 - 1e-9]), ell_min, ell_max_int)[0]
    if verbose:
        print(f"  Normalization    : {norm:.6e}  (raw value at Θ→0, maps to 1)")

    # All pair cos_thetas in one vectorised call
    idx_a, idx_b = np.triu_indices(N, k=1)
    ct_arr    = dots[idx_a, idx_b]
    gamma_arr = gamma0_parallel_vectorised(ct_arr, ell_min, ell_max_int) / norm
    sum_gamma2  = float(np.sum(gamma_arr**2))
    mean_gamma2 = sum_gamma2 / len(gamma_arr)
    pair_thetas = np.arccos(np.clip(ct_arr, -1, 1))  

    if verbose:
        print(f"\n  Σ_{{ab}} (Γ̃''_ab)²  : {sum_gamma2:.4f}")
        print(f"  Mean (Γ̃''_ab)²    : {mean_gamma2:.6f}")
        print(f"{'─'*58}\n")

    return {
        'sum_gamma2'    : sum_gamma2,
        'mean_gamma2'   : mean_gamma2,
        'ell_max'       : ell_max_int,
        'ell_max_float' : ell_max_float,
        'theta_min_rad' : theta_min_rad,
        'theta_max_rad' : theta_max_rad,
        'pair_thetas'   : pair_thetas,
        'pair_gamma'    : gamma_arr,
        'norm'          : norm,
        'N'             : N,
        'ell_min'       : ell_min,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(res, outpath='gamma_tilde_results.png'):
    """
    Two-panel plot:
      Left  — Γ₀''(Θ) curve (normalised to 0.5 at Θ=0) with pair values overlaid
      Right — histogram of (Γ̃''_{ab})² values across pairs
    """
    ell_min     = res['ell_min']
    ell_max_int = res['ell_max']
    theta_min   = res['theta_min_rad']
    theta_max   = res['theta_max_rad']
    norm        = res['norm']

    # Smooth curve in radians
    theta_arr   = np.linspace(1e-4, theta_max, 400)
    ct_arr      = np.cos(theta_arr)
    gamma_curve = gamma0_parallel_vectorised(ct_arr, ell_min, ell_max_int) / norm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left panel: Γ₀''(Θ) ──
    ax1.plot(theta_arr, gamma_curve, 'navy', lw=2.5,
             label=r"$\Gamma_0^{\prime\prime}(\Theta)$")
    ax1.scatter(res['pair_thetas'], res['pair_gamma'],
                s=4, alpha=0.25, color='steelblue', zorder=2,
                label=f"Pairs ({res['N']*(res['N']-1)//2})")
    ax1.axhline(0, color='k', lw=0.7, ls='--')
    ax1.axvline(theta_min, color='firebrick', lw=1.5, ls=':',
                label=fr"$\Theta_{{\rm min}}={theta_min:.4f}$ rad")
    ax1.set_xlabel(r"$\Theta$ (rad)", fontsize=13)
    ax1.set_ylabel(r"$\Gamma_0^{\prime\prime}(\Theta)$", fontsize=13)
    ax1.set_title(
        r"$\ell_{\rm min}=$" + f"{ell_min},  "
        r"$\ell_{\rm max}=$" + f"{ell_max_int}  "
        r"($\Theta_{\rm min}=$" + f"{theta_min:.4f} rad)",
        fontsize=11
    )
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, theta_max * 1.05)

    # ── Right panel: histogram of (Γ̃'')² ──
    ax2.hist(res['pair_gamma']**2, bins=60, color='steelblue',
             edgecolor='white', linewidth=0.4)
    ax2.axvline(res['mean_gamma2'], color='firebrick', lw=2,
                ls='--', label=f"Mean = {res['mean_gamma2']:.4f}")
    ax2.set_xlabel(r"$(\tilde{\Gamma}_{ab}^{\prime\prime})^2$", fontsize=13)
    ax2.set_ylabel("Number of pairs", fontsize=13)
    ax2.set_title(
        r"$\sum_{ab}(\tilde{\Gamma}_{ab}^{\prime\prime})^2 = $"
        + f"{res['sum_gamma2']:.2f}",
        fontsize=12
    )
    ax2.legend(fontsize=10)

    plt.suptitle(
        r"$\ell_{\rm max} = 2\pi\,/\,\Theta_{\rm min}$"
        r",   $\Gamma_0^{\prime\prime}(0) = 1$",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    if len(sys.argv) > 1:
        # Load from file
        fname = sys.argv[1]
        data  = np.loadtxt(fname, comments='#')
        ra_deg, dec_deg = data[:, 0], data[:, 1]
        print(f"Loaded {len(ra_deg)} stars from '{fname}'")
    else:
        # Demo: 100 stars randomly distributed on the full sphere
        print("No file provided — using 100 random stars on the full sphere.")
        rng = np.random.default_rng(seed=42)
        ra_deg  = rng.uniform(0, 360, 100)
        # Uniform on sphere: cos(dec) uniform in [-1,1]
        dec_deg = np.degrees(np.arcsin(rng.uniform(-1, 1, 100)))

    # Optional: set theta_min_floor_rad to your instrument resolution (radians)
    # to prevent ℓ_max blowing up if two stars happen to be very close.
    # e.g. theta_min_floor_rad=np.radians(0.1) for ~6 arcmin resolution.
    res = compute_gamma_sum(ra_deg, dec_deg, theta_min_floor_rad=None)
    plot_results(res, outpath='gamma_tilde_results.png')