"""
hd_full_matrix_snr.py
====================
Plots the full-curve HD SNR vs r = P_gw(f_l)/P_n using an exact matrix-free solver for the full covariance.

"""
#to prevent errors from using Python 3.8 or 3.9 (or anything before 3.10) on the cluster
from __future__ import annotations

import inspect
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, minres, gmres
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    build_star_positions,
    pairwise_theta,
    compute_ell_limits,
    gamma_parallel,
    gamma_parallel_matrix,
    gamma_scale_factor,
    cp_single_star_gamma,
    NORMALIZED_GAMMA,
    POL_FACTOR,
    rho_cp_full,
    STAR_COORDS_DEG,
    N_STARS,
    FIELD_SIZE_DEG,
    RANDOM_SEED,
    P_n,
    sigma_bar_sq,
    PHYSICAL_RATIO,
)

EPS = 1e-14
# Raw physical prefactor, kept importable for backward compatibility (some
# notebooks compute F_PHYS * gamma_field directly for plotting/diagnostics).
#
# HD and CP share one definition of the power spectra (c = POL_FACTOR,
# F = gamma_scale_factor() = 192 pi^3). The factor c sits on P_ab ONLY:
#     P_ab = c * F * P_gw * Gamma_ab                (a != b, cross-power)
#     P_a  = P_n + c * F * P_gw * Gamma(0)          (a = b, auto-power)
# The pair estimator is normalized by F_ab = F * Gamma_ab (no c), and the
# SNR prefactor is the amplitude P_gw^2 (no c). In units of P_gw:
#     P~_ab = c F_ab,    P~_a = 1/r + c F Gamma(0)


def hd_auto_power(gamma0: float) -> float:
    """GW part of the auto-power, P_a^gw / P_gw = c * F * Gamma(0)."""
    return POL_FACTOR * gamma_scale_factor() * float(gamma0)


F_PHYS = 192.0 * np.pi**3

#keeps the code compatible with SciPy 1.10 and 1.11, which changed the iterative solver keyword from tol -> rtol
def _iterative_tol_kwargs(func, tol: float):
    """Return version-compatible tolerance keyword arguments for SciPy solvers."""
    params = inspect.signature(func).parameters
    if "rtol" in params:
        return {"rtol": tol}
    if "tol" in params:
        return {"tol": tol}
    return {}


@dataclass(frozen=True)
class HDPairData:
    """Compact pair geometry for the matrix-free solver."""

    a_idx: np.ndarray
    b_idx: np.ndarray
    Fab: np.ndarray       # estimator normalization F * Gamma_ab (pairs), no c
    F: np.ndarray         # N x N matrix F * Gamma_ab, zero diagonal, no c
    Paa_gw: float         # c * F * Gamma(0)

    @property
    def n_pairs(self) -> int:
        return int(self.a_idx.size)


# ============================================================
#                 PAIR GEOMETRY / HELPERS
# ============================================================

def build_hd_pair_data(gamma_matrix: np.ndarray, gamma0: float) -> HDPairData:
    """Build the pair index arrays, the N x N F*Gamma matrix, and the
    GW auto-power once. POL_FACTOR is NOT applied here; it enters only
    when P~ is formed in make_hd_matvec / build_HD_matrices."""
    n_star = int(gamma_matrix.shape[0])
    #only upper-triangle when a < b, since the pair matrix is symmetric and we only need one copy of each pair
    a_idx, b_idx = np.triu_indices(n_star, k=1)
    #F is gamma tilde in the written math
    F = gamma_scale_factor() * np.array(gamma_matrix, dtype=float, copy=True)
    np.fill_diagonal(F, 0.0)
    Fab = F[a_idx, b_idx]

    if np.any(np.abs(Fab) < EPS):
        raise ValueError(
            "Some pairwise F_g values are too close to zero for the current "
            "matrix-free formulation. Check the geometry / gamma_parallel output."
        )

    return HDPairData(a_idx=a_idx, b_idx=b_idx, Fab=Fab, F=F, Paa_gw=hd_auto_power(gamma0))


# ============================================================
#                EXACT MATVEC FOR M(r)
# ============================================================

def make_hd_matvec(data: HDPairData, r: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return an exact matrix-vector product for M(r) = C(r) / P_gw^2.

    The pair-pair covariance of the estimators x_ab = d_a d_b / F_ab is

        M_ab,cd = (P~_ac P~_bd + P~_ad P~_bc) / (F_ab F_cd)

    with P~ the N x N power matrix in units of P_gw:
        off-diagonal  P~_ab = c F_ab               (factor of 2 on P_ab)
        diagonal      P~_aa = 1/r + c F Gamma(0)   (noise + GW auto-power)

    This single expression covers Case 1 (no shared star), Case 2 (one
    shared star) and Case 3 (same pair). Summing over c<d is the same as
    summing over all ordered c != d, so with Z_cd = x_cd / F_cd (symmetric,
    zero diagonal):

        y_ab = (P~ Z P~)_ab / F_ab

    Two N x N matrix products per call; the N_pairs x N_pairs matrix is
    never formed.
    """
    a_idx = data.a_idx
    b_idx = data.b_idx
    Fab = data.Fab
    n_pairs = data.n_pairs

    P = POL_FACTOR * data.F                              # cross-power P~_ab
    np.fill_diagonal(P, 1.0 / float(r) + data.Paa_gw)    # auto-power P~_aa

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size != n_pairs:
            raise ValueError(f"Expected vector of length {n_pairs}, got {x.shape}")

        # Lift the pair vector into a symmetric N x N matrix, Z_ab = x_ab / F_ab.
        Z = np.zeros_like(P)
        z = x / Fab
        Z[a_idx, b_idx] = z
        Z[b_idx, a_idx] = z

        # (P~ Z P~)_ab = sum_{c != d} P~_ac Z_cd P~_db does the whole
        # pairs-of-pairs sum, including shared-star and same-pair terms.
        W = P @ Z @ P
        return np.asarray(W[a_idx, b_idx] / Fab, dtype=float)

    return matvec


# ============================================================
#                 DENSE FALLBACK FOR SMALL N
# ============================================================

def build_HD_matrices(gamma_matrix: np.ndarray, gamma0: float):
    """Dense decomposition M(r) = A + B/r + D/r^2 (small problems only).

    M_ab,cd = (P~_ac P~_bd + P~_ad P~_bc) / (F_ab F_cd), F_ab = F Gamma_ab,
    P~_ab = c F_ab (a != b), P~_aa = 1/r + p0 with p0 = c F Gamma(0).
    The factor c = POL_FACTOR appears only through P~:

      Case 1 (no shared star):  c^2 (F_ac F_bd + F_ad F_bc) / (F_ab F_cd)
      Case 2 (a = c, b != d):   c^2 + c (1/r + p0) F_bd / (F_ab F_ad)
      Case 3 (same pair):       c^2 + (1/r + p0)^2 / F_ab^2
    """
    n_star = gamma_matrix.shape[0]
    pairs = np.array([(a, b) for a in range(n_star) for b in range(a + 1, n_star)])
    n_pairs = len(pairs)

    c = POL_FACTOR
    p0 = hd_auto_power(gamma0)
    Fg_mat = gamma_scale_factor() * np.array(gamma_matrix, dtype=float)
    Fg_pair = Fg_mat[pairs[:, 0], pairs[:, 1]]

    a_idx = pairs[:, 0]
    b_idx = pairs[:, 1]
    a_i = a_idx[:, None]
    b_i = b_idx[:, None]
    c_j = a_idx[None, :]
    d_j = b_idx[None, :]

    ac = a_i == c_j
    bc = b_i == c_j
    ad = a_i == d_j
    bd = b_i == d_j

    case1 = ~ac & ~bc & ~ad & ~bd
    case2_ac = ac & ~bd & ~bc & ~ad
    case2_bc = bc & ~ac & ~bd & ~ad
    case2_ad = ad & ~ac & ~bd & ~bc
    case2_bd = bd & ~ac & ~ad & ~bc

    Fg_ab = Fg_pair[:, None]
    Fg_cd = Fg_pair[None, :]
    Fg_ac = Fg_mat[a_i, c_j]
    Fg_bd = Fg_mat[b_i, d_j]
    Fg_ad = Fg_mat[a_i, d_j]
    Fg_bc = Fg_mat[b_i, c_j]

    # r^0 part: c^2 everywhere (the P~_ab P~_cd-type product), Case 1 geometry
    A = np.full((n_pairs, n_pairs), c * c, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        A_c1 = c * c * (Fg_ac * Fg_bd + Fg_ad * Fg_bc) / (Fg_ab * Fg_cd)
    A[case1] = A_c1[case1]

    # Case 2 geometry factor F_(other,other) / (F_ab F_(shared,other))
    B2 = np.zeros((n_pairs, n_pairs), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        B2[case2_ac] = (Fg_bd / (Fg_ab * Fg_ad))[case2_ac]
        B2[case2_bc] = (Fg_ad / (Fg_ab * Fg_bd))[case2_bc]
        B2[case2_ad] = (Fg_bc / (Fg_ab * Fg_ac))[case2_ad]
        B2[case2_bd] = (Fg_ac / (Fg_ab * Fg_bc))[case2_bd]

    # Case 2: c (1/r + p0) B2
    A += c * p0 * B2
    B = c * B2

    # Case 3: (1/r + p0)^2 / F_ab^2
    diag_idx = np.diag_indices(n_pairs)
    A[diag_idx] += p0**2 / Fg_pair**2
    B[diag_idx] += 2.0 * p0 / Fg_pair**2
    D = np.zeros((n_pairs, n_pairs), dtype=float)
    np.fill_diagonal(D, 1.0 / Fg_pair**2)

    return pairs, A, B, D


def rho_hd_full_matrix_dense(x_arr, gamma_matrix, gamma0, svd_rcond=1e-10, verbose=True):
    """Original dense eigendecomposition path."""
    if verbose:
        print("Building HD covariance matrices A, B, D...", flush=True)
    t0 = time.time()
    _, A, B, D = build_HD_matrices(gamma_matrix, gamma0)
    n_pairs = A.shape[0]
    if verbose:
        print(
            f"  Done ({time.time()-t0:.1f}s). Matrix: {n_pairs}x{n_pairs}",
            flush=True,
        )

    x_arr = np.asarray(x_arr, dtype=float)
    rho_vals = np.zeros(len(x_arr), dtype=float)

    for k, r in enumerate(x_arr):
        if verbose:
            print(
                f"  r[{k+1}/{len(x_arr)}] = {r:.3e}  ({time.time()-t0:.0f}s elapsed)",
                flush=True,
            )

        M = A + B / r + D / r**2
        eigvals, eigvecs = np.linalg.eigh(M)
        thresh = svd_rcond * np.max(np.abs(eigvals))
        inv_eigs = np.where(np.abs(eigvals) > thresh, 1.0 / eigvals, 0.0)
        row_sums = eigvecs.sum(axis=0)
        rho_sq = 2.0 * float(np.dot(inv_eigs, row_sums**2))
        rho_vals[k] = np.sqrt(max(rho_sq, 0.0))

    return rho_vals


# ============================================================
#            FULL HD SNR VIA MATRIX-FREE SOLVER
# ============================================================

def rho_hd_full_matrix(
    x_arr,
    gamma_matrix,
    gamma0,
    svd_rcond=1e-10,
    verbose=True,
    dense_cutover_pairs: int = 3000,
    maxiter: int | None = None,
):
    """Full HD SNR curve including all three covariance cases.

    For small pair counts, this uses the original dense eigendecomposition.
    For larger problems, it solves M(r) x = 1 with MINRES and a Jacobi
    preconditioner, using the exact matrix-vector product above.

    Parameters
    ----------
    x_arr : array_like
        r = P_gw / sigma_bar^2 values.
    gamma_matrix : ndarray
        Geometry-dependent gamma matrix.
    gamma0 : float
        Gamma(0), sets the per-star GW auto-power c*F*Gamma(0).
    svd_rcond : float
        Used as the relative tolerance for the iterative solver in the large-N
        path, and as the eigenvalue cutoff in the dense fallback.
    dense_cutover_pairs : int
        Use the dense path only when N_pairs <= this threshold.
    maxiter : int or None
        Maximum MINRES iterations per r-value. None lets SciPy choose.
    """
    x_arr = np.asarray(x_arr, dtype=float)
    data = build_hd_pair_data(gamma_matrix, gamma0)

    if data.n_pairs <= dense_cutover_pairs:
        if verbose:
            print(
                f"Using dense fallback path (N_pairs={data.n_pairs} <= {dense_cutover_pairs})",
                flush=True,
            )
        return rho_hd_full_matrix_dense(x_arr, gamma_matrix, gamma0, svd_rcond=svd_rcond, verbose=verbose)

    if verbose:
        print("Building matrix-free HD operator...", flush=True)
        print(
            f"  N_stars={gamma_matrix.shape[0]}, N_pairs={data.n_pairs}",
            flush=True,
        )

    rho_vals = np.zeros(len(x_arr), dtype=float)
    t0 = time.time()

    ones_rhs = np.ones(data.n_pairs, dtype=float)
    x0 = None

    for k, r in enumerate(x_arr):
        if verbose:
            print(
                f"  r[{k+1}/{len(x_arr)}] = {r:.3e}  ({time.time()-t0:.0f}s elapsed)",
                flush=True,
            )

        matvec = make_hd_matvec(data, r)
        Aop = LinearOperator((data.n_pairs, data.n_pairs), matvec=matvec, dtype=float)

        # Diagonal of M(r): c^2 + (1/r + p0)^2 / F_ab^2
        diag = POL_FACTOR**2 + (1.0 / r + data.Paa_gw) ** 2 / (data.Fab * data.Fab)
        inv_diag = 1.0 / diag
        Mop = LinearOperator(
            (data.n_pairs, data.n_pairs),
            matvec=lambda v, inv_diag=inv_diag: inv_diag * np.asarray(v, dtype=float),
            dtype=float,
        )

        base_maxiter = 2000 if maxiter is None else int(maxiter)
        minres_trials = [
            (svd_rcond, base_maxiter),
            (max(svd_rcond * 10.0, 1e-8), max(base_maxiter * 2, 4000)),
            (max(svd_rcond * 100.0, 1e-7), max(base_maxiter * 5, 10000)),
        ]

        sol = None
        info = None
        last_tol = None
        last_maxiter = None

        for tol, trial_maxiter in minres_trials:
            last_tol = tol
            last_maxiter = trial_maxiter
            kwargs = _iterative_tol_kwargs(minres, tol)
            minres_kwargs = dict(M=Mop, maxiter=trial_maxiter, **kwargs)
            if x0 is not None:
                minres_kwargs["x0"] = x0
            sol, info = minres(Aop, ones_rhs, **minres_kwargs)
            if info == 0:
                break

        if info != 0:
            # GMRES is less memory-frugal than MINRES, but it is a useful
            # fallback when the symmetric iteration struggles at a few r values.
            gmres_restart = min(200, data.n_pairs)
            gmres_maxiter = max(100, base_maxiter)
            gmres_tol = last_tol if last_tol is not None else svd_rcond
            gmres_kwargs = _iterative_tol_kwargs(gmres, gmres_tol)
            gmres_call = dict(M=Mop, restart=gmres_restart, maxiter=gmres_maxiter, **gmres_kwargs)
            if x0 is not None:
                gmres_call["x0"] = x0
            sol, info = gmres(Aop, ones_rhs, **gmres_call)

        if info != 0:
            raise RuntimeError(
                f"Iterative solver did not converge for r={r:.3e} (info={info}). "
                f"Last MINRES tol={last_tol:.1e}, maxiter={last_maxiter}."
            )

        x0 = sol
        rho_sq = 2.0 * float(np.sum(sol))
        rho_vals[k] = np.sqrt(max(rho_sq, 0.0))

    return rho_vals

def hd_strong_signal_plateau(gamma_matrix):
    """
    Closed-form approximation for the strong-signal (r -> infinity) HD plateau,
    using the full N_pairs x N_pairs covariance matrix (not just the diagonal
    Case-3-only approximation).

    rho^2_HD,intermediate ~= F0^2 * N(N-1) / [F0^2*(N^2-3N+3) + 2*F0*(N-2) + 1]

    LEGACY: derived with per-star GW auto-power = P_gw and no POL_FACTOR.
    It does not include the c*F*Gamma(0) auto-power or the factor of 2 on
    P_ab now used in the covariance, so it no longer predicts the numeric
    plateau. Kept for reference only.

    IMPORTANT: this approximation degrades for wide or full-sky fields, where
    gamma_ab is no longer close to uniform across pairs. Treat this
    as a narrow-field-only diagnostic, and rely on the numeric
    plateau (max(rho_hd)) instead for wide or full-sky fields.
    """
    n_star = gamma_matrix.shape[0]
    Fg = gamma_scale_factor() * gamma_matrix
    vals = Fg[np.triu_indices_from(Fg, k=1)]
    vals = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
    if vals.size == 0:
        return 0.0
    F0 = float(np.mean(vals))
    N = n_star
    rho_sq = (F0**2 * N * (N - 1)) / (F0**2 * (N**2 - 3*N + 3) + 2*F0*(N - 2) + 1)
    return float(np.sqrt(max(rho_sq, 0.0)))


def print_snr_diagnostics(r_values, rho_cp, rho_hd, ell_min, ell_max, gamma_matrix, n_stars=N_STARS):
    """
    Print weak/strong-signal slopes and plateau values for CP and HD curves.

    Slopes computed via log-log linear regression over designated windows.
    CP plateau is the exact r -> inf limit of the full covariance. HD plateau now has a closed-form
    approximation too (see hd_strong_signal_plateau), valid when F_ab is
    close to uniform across pairs -- which is the geometrically-uniform,
    narrow-field regime this paper's results are computed in.
    """
    from main import cp_single_star_gamma

    gamma0   = cp_single_star_gamma(ell_min, ell_max)
    n_pairs  = n_stars * (n_stars - 1) // 2

    log_r      = np.log10(r_values)
    log_rho_cp = np.log10(np.maximum(rho_cp, 1e-300))
    log_rho_hd = np.log10(np.maximum(rho_hd, 1e-300))

    def slope_in_window(log_x, log_y, x_lo, x_hi, label):
        mask = (10**log_x >= x_lo) & (10**log_x <= x_hi)
        if mask.sum() < 2:
            print(f'  WARNING: fewer than 2 points in {label} window [{x_lo:.0e}, {x_hi:.0e}] — '
                  f'try increasing n_r in plot_full_comparison.')
            return float('nan')
        return float(np.polyfit(log_x[mask], log_y[mask], 1)[0])

    # Weak-signal window: well below the physical ratio (~6e-11)
    slope_cp_weak = slope_in_window(log_r, log_rho_cp, 1e-13, 1e-11, 'CP weak')
    slope_hd_weak = slope_in_window(log_r, log_rho_hd, 1e-13, 1e-11, 'HD weak')

    # Strong-signal window: deep in saturation
    slope_cp_strong = slope_in_window(log_r, log_rho_cp, 1e-2, 1e1, 'CP strong')
    slope_hd_strong = slope_in_window(log_r, log_rho_hd, 1e-2, 1e1, 'HD strong')

    # CP plateau: exact r -> inf limit of the full covariance, sqrt(1^T G^+ 1)
    from main import rho_cp_strong_plateau
    cp_plateau_anal = rho_cp_strong_plateau(gamma_matrix, ell_min, ell_max)
    cp_plateau_num  = float(np.max(rho_cp))

    # HD plateau: closed-form approximation (uniform-F_ab limit) vs. numeric.
    hd_plateau_anal = hd_strong_signal_plateau(gamma_matrix)
    hd_plateau_num  = float(np.max(rho_hd))

    print('\n' + '='*60)
    print('              SNR CURVE DIAGNOSTICS (full matrix)')
    print(f'  ell_min={ell_min}, ell_max={ell_max}, N_stars={n_stars}, N_pairs={n_pairs}')
    print('='*60)

    print('\n── Common Process (CP) ──')
    print(f'  Weak-signal slope   (r ~ 1e-13 to 1e-11):  {slope_cp_weak:+.3f}')
    print(f'  Strong-signal slope (r ~ 1e-2  to 1e+1 ):  {slope_cp_strong:+.3f}')
    print(f'  Plateau [exact   ]  = sqrt(1^T G^+ 1)      = {cp_plateau_anal:.4f}')
    print(f'  Plateau [numeric ]  = max(rho_CP)          = {cp_plateau_num:.4f}')

    print('\n── Hellings-Downs (HD) — full covariance matrix ──')
    print(f'  Weak-signal slope   (r ~ 1e-13 to 1e-11):  {slope_hd_weak:+.3f}')
    print(f'  Strong-signal slope (r ~ 1e-2  to 1e+1 ):  {slope_hd_strong:+.3f}')
    print(f'  Plateau [legacy  ]  = F_aa=1 uniform approx = {hd_plateau_anal:.4f}')
    print(f'  Plateau [numeric ]  = max(rho_HD)          = {hd_plateau_num:.4f}')
    print('='*60 + '\n')


# ============================================================
#                          PLOT
# ============================================================

def plot_full_comparison(gamma_matrix, ell_min, ell_max, n_r=150, save_path=None):
    """Plot CP and HD SNR on the same axes."""
    r_values = np.logspace(-13, 2, n_r)

    print("Computing CP full curve...", flush=True)
    rho_cp = rho_cp_full(r_values, gamma_matrix, ell_min, ell_max)

    print("\nComputing HD full curve...", flush=True)
    rho_hd = rho_hd_full_matrix(r_values, gamma_matrix, cp_single_star_gamma(ell_min, ell_max), verbose=True)

    print(f"\nPhysical r = P_gw/P_n = {PHYSICAL_RATIO:.3e}")
    print_snr_diagnostics(r_values, rho_cp, rho_hd, ell_min, ell_max, gamma_matrix, n_stars=gamma_matrix.shape[0])


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(r_values, rho_cp, color="C0", lw=2.5, label=r"$\rho_{\rm CP}$")
    ax.loglog(
        r_values,
        rho_hd,
        color="C1",
        lw=2.5,
        label=r"$\rho_{\rm HD}$",
    )
    ax.axvline(PHYSICAL_RATIO, color="k", lw=1.2, ls="--", label=rf"physical $r = {PHYSICAL_RATIO:.1e}$")

    ax.set_xlabel(r"$P_{\rm gw}(f_l)\,/\,P_n(f_l)$", fontsize=13)
    ax.set_ylabel(r"$\rho$", fontsize=13)
    ax.set_title("CP and HD Full SNR", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
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

if __name__ == "__main__":
    stars_deg = build_star_positions(STAR_COORDS_DEG, N_STARS, FIELD_SIZE_DEG, RANDOM_SEED)
    theta_mat = pairwise_theta(stars_deg)
    ell_min, ell_max = compute_ell_limits(theta_mat, FIELD_SIZE_DEG)

    print(f"ell_min={ell_min}, ell_max={ell_max}, N_stars={N_STARS}")
    print(f"N_pairs = {N_STARS * (N_STARS - 1) // 2}")
    print(f"NORMALIZED_GAMMA = {NORMALIZED_GAMMA}  (gamma_scale_factor = {gamma_scale_factor():.4f})")
    print(f"POL_FACTOR       = {POL_FACTOR:g}  (multiplies P_ab)")

    gamma = gamma_parallel_matrix(theta_mat, ell_min, ell_max)

    # Tag the output filename with N/FoV (and normalization state) so
    # multiple Slurm array tasks / comparison runs don't overwrite each
    # other's plot.
    norm_tag = ("_normGamma" if NORMALIZED_GAMMA else "") + (f"_pol{POL_FACTOR:g}" if POL_FACTOR != 1 else "")
    out_name = f"hd_full_matrix_snr_N{N_STARS}_FoV{FIELD_SIZE_DEG:g}{norm_tag}.png"

    r_vals, rho_cp, rho_hd = plot_full_comparison(
        gamma,
        ell_min,
        ell_max,
        n_r=150,
        save_path=out_name,
    )

    # Save the underlying arrays alongside the plot, tagged with the same
    # N/FoV convention as the PNG filename, so future runs can be compared
    # and overlaid with compare_snr_runs_fullmatrix.py.
    data_name = f"hd_full_matrix_snr_N{N_STARS}_FoV{FIELD_SIZE_DEG:g}{norm_tag}.npz"
    np.savez(
        data_name,
        r_vals=r_vals,
        rho_cp=rho_cp,
        rho_hd=rho_hd,
        N_STARS=N_STARS,
        FIELD_SIZE_DEG=FIELD_SIZE_DEG,
        ell_min=ell_min,
        ell_max=ell_max,
        PHYSICAL_RATIO=PHYSICAL_RATIO,
        NORMALIZED_GAMMA=NORMALIZED_GAMMA,
        POL_FACTOR=POL_FACTOR,
    )
    print(f"Data saved to {data_name}")

    print("\nSelected output:")
    print(f"{'r':>10}  {'rho_CP':>10}  {'rho_HD':>10}")
    for rv, rcp, rhd in zip(r_vals[::5], rho_cp[::5], rho_hd[::5]):
        print(f"  {rv:.2e}   {rcp:.4f}   {rhd:.4f}")