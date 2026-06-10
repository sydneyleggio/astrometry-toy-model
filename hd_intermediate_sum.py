"""
hd_intermediate_snr.py
======================
Computes the intermediate-regime HD SNR using Romano eq. 20:

    rho^2_HD = (A_bar^2_gw)^2 * sum_{ab,cd} (C^{-1})_{ab,cd}
             = 2 * 1^T M^{-1} 1

where M = C / [(1/2)(A_bar^2_gw)^2].

All formulas use F_ab = 192*pi^3 * tilde_Gamma''_0(Theta_ab), which is
the quantity tilde_F''_ab defined in the derivation notes (page 1).
F_ab is the astrometric analog of Romano's chi_ab, with 192*pi^3 absorbed.

P_a(f) = P_na + P_gw = sigma_bar^2 + P_gw

In the intermediate regime P_gw >> P_na, so P_a/P_gw -> 1, and the
three M matrix cases (from the full derivation, pages 2-7) become:

  Case 1 (no shared stars):
      M = (F_ac*F_bd + F_ad*F_bc) / (F_ab * F_cd)

  Case 2 (one shared star, e.g. c=a):
      M = 1 + (P_a/P_gw) * F_bd / (F_ab * F_ad)
        -> intermediate limit (P_a/P_gw -> 1):
      M = 1 + F_bd / (F_ab * F_ad)

  Case 3 (same pair, c=a, d=b):
      M = 1 + (P_a*P_b/P_gw^2) * 1/F_ab^2
        -> intermediate limit (P_a P_b / P_gw^2 -> 1):
      M = 1 + 1 / F_ab^2

The key point: F_ab = 192*pi^3 * G_ab where G_ab is the normalised
tilde_Gamma''(0)=1 overlap. Using F (not G) is essential because the
1/F_ab^2 in Case 3 is physically 1/(192*pi^3)^2 / G_ab^2, which is
very small — stabilising the matrix and giving the correct plateau value.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    build_star_positions,
    pairwise_theta,
    compute_ell_limits,
    gamma_parallel,
    STAR_COORDS_DEG,
    N_STARS,
    FIELD_SIZE_DEG,
    RANDOM_SEED,
)

EPS = 1e-14
ASTRO_FACTOR = 192.0 * np.pi**3   # = 192*pi^3, absorbed into F_ab


def get_valid_pairs(F, f_threshold=None):
    """
    Return indices of pairs (a,b) where |F_ab| > threshold.
    Pairs near zero are excluded: the estimator A^2_ab = P_ab/F_ab is
    undefined there, and their 1/F_ab^2 terms would make M singular.
    threshold defaults to 1% of the max |F_ab| off-diagonal value.
    """
    N = F.shape[0]
    a_all, b_all = np.triu_indices(N, k=1)
    F_vals = np.abs(F[a_all, b_all])
    if f_threshold is None:
        f_threshold = 0.01 * F_vals.max()
    valid = F_vals > f_threshold
    return a_all[valid], b_all[valid]


def build_F_matrix(stars_deg, ell_min, ell_max):
    """
    Build the N x N matrix F_ab = 192*pi^3 * tilde_Gamma''_0(Theta_ab).
    This is tilde_F''_ab from the derivation — the astrometric analog of
    Romano's chi_ab.  F(0) = 192*pi^3 (diagonal).
    """
    theta      = pairwise_theta(stars_deg)
    theta_safe = np.where(np.isnan(theta), 0.0, theta)
    gamma_raw  = gamma_parallel(theta_safe, ell_min, ell_max)

    # F_ab = 192*pi^3 * tilde_Gamma''(Theta_ab), NOT normalised to 1
    F = ASTRO_FACTOR * gamma_raw

    # Diagonal: F_aa = 192*pi^3 * Gamma''(0) = 192*pi^3 * g0
    g0 = gamma_parallel(np.array([0.0]), ell_min, ell_max)[0]
    np.fill_diagonal(F, ASTRO_FACTOR * g0)
    return F


def build_shape_matrix_intermediate(F, a_idx, b_idx):
    """
    N_pairs x N_pairs shape matrix M in the intermediate regime,
    for valid pairs only (a_idx, b_idx from get_valid_pairs).

    Case 1: M = (F_ac*F_bd + F_ad*F_bc) / (F_ab * F_cd)
    Case 2: M = 1 + F_bd / (F_ab * F_ad)
    Case 3: M = 1 + 1 / F_ab^2
    """
    Np = len(a_idx)
    M  = np.zeros((Np, Np))

    for I in range(Np):
        a, b = a_idx[I], b_idx[I]
        F_ab = F[a, b]

        for J in range(I, Np):
            c, d     = a_idx[J], b_idx[J]
            shared   = {a, b} & {c, d}
            n_shared = len(shared)

            if n_shared == 2:
                val = 1.0 + 1.0 / F_ab**2

            elif n_shared == 1:
                shared_star = next(iter(shared))
                outer_ab    = b if shared_star == a else a
                outer_cd    = d if shared_star == c else c
                F_cd        = F[c, d]
                F_outer     = F[outer_ab, outer_cd]
                if abs(F_cd) < EPS:
                    continue
                val = 1.0 + F_outer / (F_ab * F_cd)

            else:
                F_cd = F[c, d]
                if abs(F_cd) < EPS:
                    continue
                F_ac = F[a, c];  F_bd = F[b, d]
                F_ad = F[a, d];  F_bc = F[b, c]
                val  = (F_ac * F_bd + F_ad * F_bc) / (F_ab * F_cd)

            M[I, J] = val
            M[J, I] = val

    return M


def compute_rho_hd_intermediate(F, verbose=True):
    """
    rho^2 = 2 * 1^T M^{-1} 1,  using valid pairs only.
    """
    a_idx, b_idx = get_valid_pairs(F)
    Np = len(a_idx)
    N  = F.shape[0]

    if verbose:
        print(f"\n{'─'*58}")
        print(f"  N stars        : {N}")
        print(f"  Total pairs    : {N*(N-1)//2}")
        print(f"  Valid pairs    : {Np}  (|F_ab| > 1% of max)")
        print(f"  F diagonal     : {F[0,0]:.4e}")
        print(f"  Building M matrix (intermediate regime) ...")

    M    = build_shape_matrix_intermediate(F, a_idx, b_idx)
    ones = np.ones(Np)

    # Diagonal preconditioning for numerical stability
    diag = np.diag(M)
    diag = np.where(diag > 0, diag, 1.0)
    D    = 1.0 / np.sqrt(diag)
    M_pre = M * D[:, None] * D[None, :]
    rhs   = D * ones

    w_pre, _, _, _ = np.linalg.lstsq(M_pre, rhs, rcond=1e-12)
    rho_sq = 2.0 * float(ones @ (D * w_pre))
    rho    = np.sqrt(max(rho_sq, 0.0))

    if verbose:
        print(f"\n  rho^2_HD (intermediate) : {rho_sq:.6f}")
        print(f"  rho_HD  (intermediate)  : {rho:.6f}")
        print(f"  (Derivation benchmark   : rho^2~38.79, rho~6.23)")
        print(f"{'─'*58}\n")

    return rho, rho_sq


if __name__ == '__main__':
    stars_deg        = build_star_positions(
        STAR_COORDS_DEG, N_STARS, FIELD_SIZE_DEG, RANDOM_SEED)
    theta_mat        = pairwise_theta(stars_deg)
    ell_min, ell_max = compute_ell_limits(theta_mat, FIELD_SIZE_DEG)

    print(f"ell_min = {ell_min},  ell_max = {ell_max}")

    F = build_F_matrix(stars_deg, ell_min, ell_max)
    print(f"F(0) diagonal = {F[0,0]:.4e}  (should be 192*pi^3 * g0)")

    rho, rho_sq = compute_rho_hd_intermediate(F)
    print(f"rho^2_HD = {rho_sq:.4f},  rho_HD = {rho:.4f}")