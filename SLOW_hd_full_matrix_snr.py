"""
hd_full_matrix_snr.py
=====================
Computes the full HD SNR curve by explicitly building and inverting the
N_pairs x N_pairs estimator covariance matrix, including all three Cases:

  Case 1 (no shared stars):
    C_{ab,cd} = (1/2)(A_bar^2)^2 * (Fg_ac*Fg_bd + Fg_ad*Fg_bc) / (Fg_ab*Fg_cd)

  Case 2 (one shared star, e.g. c=a):
    C_{ab,ad} = (1/2)(A_bar^2)^2 * [1 + (Pa/Pgw) * Fg_bd / (Fg_ab*Fg_ad)]

  Case 3 (same pair, c=a and d=b):
    C_{ab,ab} = (1/2)(A_bar^2)^2 * [1 + Pa*Pb/Pgw^2 * 1/Fg_ab^2]

where Fg_ab = 192*pi^3 * Gamma_o(Theta_ab)  (tilde-gamma convention).

The matrix M(r) = A + B/r + D/r^2 is built once from the geometry,
then for each r = P_gw/sigma_bar^2 the SNR is:

    rho^2_HD = 2 * sum_{ab,cd} (M^+)_{ab,cd}
             = 2 * sum_k (1/lambda_k) * (sum_i V_{ik})^2

where lambda_k, V are the eigendecomposition of M, and the pseudo-inverse
handles the negative eigenvalues that arise in the intermediate signal regime.

WARNING: For N=100 stars (N_pairs=4950), each eigendecomposition takes ~30s.
Use a coarse r grid (n_r=30-50) and save results to disk.
Total runtime: approximately 15-25 minutes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    build_star_positions,
    pairwise_theta,
    compute_ell_limits,
    gamma_parallel,
    rho_cp_full,
    STAR_COORDS_DEG,
    N_STARS,
    FIELD_SIZE_DEG,
    RANDOM_SEED,
    P_n,
    sigma_bar_sq,
    PHYSICAL_RATIO,
)

EPS    = 1e-14
F_PHYS = 192.0 * np.pi**3


# ============================================================
#            BUILD R-INDEPENDENT COVARIANCE MATRICES
# ============================================================

def build_HD_matrices(gamma_matrix):
    """
    Decompose the HD estimator covariance into r-independent matrices A, B, D:

        C_{ab,cd} = (P_gw^2 / 2) * M_{ab,cd}
        M(r)      = A + B/r + D/r^2      where r = P_gw / sigma_bar^2

    A: Case 1 geometry ratios + the leading '1' from Cases 2 and 3
    B: Case 2 geometry terms (Pa/Pgw ~ sigma^2/Pgw = 1/r)
    D: Case 3 diagonal terms ((Pa*Pb/Pgw^2) * 1/Fg_ab^2 ~ (1/r^2)/Fg_ab^2)

    Returns: pairs array (N_pairs, 2), A, B, D matrices (N_pairs x N_pairs)
    """
    N     = gamma_matrix.shape[0]
    pairs = np.array([(a, b) for a in range(N) for b in range(a+1, N)])
    Np    = len(pairs)

    Fg_pair = F_PHYS * gamma_matrix[pairs[:, 0], pairs[:, 1]]
    Fg_mat  = F_PHYS * gamma_matrix

    a_idx = pairs[:, 0];  b_idx = pairs[:, 1]
    a_i = a_idx[:, None]; b_i = b_idx[:, None]
    c_j = a_idx[None, :]; d_j = b_idx[None, :]

    ac = (a_i == c_j);  bc = (b_i == c_j)
    ad = (a_i == d_j);  bd = (b_i == d_j)

    case3    = ac & bd
    case1    = ~ac & ~bc & ~ad & ~bd
    case2_ac = ac & ~bd & ~bc & ~ad
    case2_bc = bc & ~ac & ~bd & ~ad
    case2_ad = ad & ~ac & ~bd & ~bc
    case2_bd = bd & ~ac & ~ad & ~bc

    Fg_ab = Fg_pair[:, None];  Fg_cd = Fg_pair[None, :]
    Fg_ac = Fg_mat[a_i, c_j];  Fg_bd = Fg_mat[b_i, d_j]
    Fg_ad = Fg_mat[a_i, d_j];  Fg_bc = Fg_mat[b_i, c_j]

    A = np.ones((Np, Np))
    with np.errstate(divide='ignore', invalid='ignore'):
        A_c1 = (Fg_ac * Fg_bd + Fg_ad * Fg_bc) / (Fg_ab * Fg_cd)
    A[case1] = A_c1[case1]

    B = np.zeros((Np, Np))
    with np.errstate(divide='ignore', invalid='ignore'):
        B[case2_ac] = (Fg_mat[b_i, d_j] / (Fg_ab * Fg_mat[a_i, d_j]))[case2_ac]
        B[case2_bc] = (Fg_mat[a_i, d_j] / (Fg_ab * Fg_mat[b_i, d_j]))[case2_bc]
        B[case2_ad] = (Fg_mat[b_i, c_j] / (Fg_ab * Fg_mat[a_i, c_j]))[case2_ad]
        B[case2_bd] = (Fg_mat[a_i, c_j] / (Fg_ab * Fg_mat[b_i, c_j]))[case2_bd]

    D = np.zeros((Np, Np))
    np.fill_diagonal(D, 1.0 / Fg_pair**2)

    return pairs, A, B, D


# ============================================================
#           FULL HD SNR VIA MATRIX PSEUDO-INVERSE
# ============================================================

def rho_hd_full_matrix(x_arr, gamma_matrix, svd_rcond=1e-10, verbose=True):
    """
    Full HD SNR curve including all three covariance Cases.

    Builds M(r) = A + B/r + D/r^2 for each r, computes the symmetric
    eigendecomposition, and evaluates:

        rho^2_HD = 2 * sum_{ab,cd} (M^+)_{ab,cd}
                 = 2 * sum_k (1/lambda_k) * (sum_i V_{ik})^2

    x_arr    : r = P_gw / sigma_bar^2 (same x-axis as rho_cp_full)
    svd_rcond: threshold for pseudo-inverse eigenvalue truncation
    verbose  : print progress (each r-value takes ~30s for N=100)
    """
    if verbose:
        print("Building HD covariance matrices A, B, C, D...", flush=True)
    t0 = time.time()
    _, A, B, D = build_HD_matrices(gamma_matrix)
    Np = A.shape[0]
    if verbose:
        print(f"  Done ({time.time()-t0:.1f}s). Matrix: {Np}x{Np}, "
              f"estimated total: {len(x_arr)*31/60:.0f} min", flush=True)

    x_arr    = np.asarray(x_arr, dtype=float)
    rho_vals = np.zeros(len(x_arr))

    for k, r in enumerate(x_arr):
        if verbose:
            print(f"  r[{k+1}/{len(x_arr)}] = {r:.3e}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

        M = A + B / r + D / r**2

        eigvals, eigvecs = np.linalg.eigh(M)
        thresh   = svd_rcond * np.max(np.abs(eigvals))
        inv_eigs = np.where(np.abs(eigvals) > thresh, 1.0 / eigvals, 0.0)

        row_sums = eigvecs.sum(axis=0)
        rho_sq   = 2.0 * float(np.dot(inv_eigs, row_sums**2))
        rho_vals[k] = np.sqrt(max(rho_sq, 0.0))

    return rho_vals


# ============================================================
#                          PLOT
# ============================================================

def plot_full_comparison(gamma_matrix, ell_min, ell_max, n_r=30,
                         save_path=None):
    """
    Plot CP (full curve) and HD (full matrix, all Cases) SNR on the same axes.

    Uses a coarse fixed sweep [1e-13, 1e2] appropriate for the ~30s/point
    runtime. The physical operating point is marked automatically.
    """
    # Fixed sweep — physical ratio and transitions arise naturally
    r_values = np.logspace(-13, 2, n_r)

    print("Computing CP full curve...", flush=True)
    rho_cp = rho_cp_full(r_values, ell_min, ell_max)

    print("\nComputing HD full matrix curve...", flush=True)
    rho_hd = rho_hd_full_matrix(r_values, gamma_matrix, verbose=True)

    print(f"\nPhysical r = P_gw/P_n = {PHYSICAL_RATIO:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(r_values, rho_cp, color='C0', lw=2.5,
              label=r'$\rho_{\rm CP}$')
    ax.loglog(r_values, rho_hd, color='C1', lw=2.5,
              label=r'$\rho_{\rm HD}$ (full matrix, all Cases)')
    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho$',                           fontsize=13)
    ax.set_title('CP and HD SNR — Full Covariance Matrix', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nFigure saved to {save_path}")
    else:
        plt.show()

    return r_values, rho_cp, rho_hd


# ============================================================
#                        ENTRY POINT
# ============================================================

if __name__ == '__main__':
    stars_deg        = build_star_positions(STAR_COORDS_DEG, N_STARS,
                                            FIELD_SIZE_DEG, RANDOM_SEED)
    theta_mat        = pairwise_theta(stars_deg)
    ell_min, ell_max = compute_ell_limits(theta_mat, FIELD_SIZE_DEG)

    print(f"ell_min={ell_min}, ell_max={ell_max}, N_stars={N_STARS}")
    print(f"N_pairs = {N_STARS*(N_STARS-1)//2}")

    gamma = gamma_parallel(theta_mat, ell_min, ell_max)

    r_vals, rho_cp, rho_hd = plot_full_comparison(
        gamma, ell_min, ell_max,
        n_r=30,
        save_path="hd_full_matrix_snr.png"
    )

    print("\nSelected output:")
    print(f"{'r':>10}  {'rho_CP':>10}  {'rho_HD':>10}")
    for rv, rcp, rhd in zip(r_vals[::5], rho_cp[::5], rho_hd[::5]):
        print(f"  {rv:.2e}   {rcp:.4f}   {rhd:.4f}")