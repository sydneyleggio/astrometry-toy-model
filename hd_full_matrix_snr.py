"""
hd_full_matrix_snr.py
====================
Plots the full-curve HD SNR vs r = P_gw(f_l)/P_n using an exact matrix-free solver for the full covariance.

"""

from __future__ import annotations

import inspect
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, minres, gmres

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (  # noqa: E402
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

EPS = 1e-14
F_PHYS = 192.0 * np.pi**3


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
    Fab: np.ndarray
    F: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.a_idx.size)


# ============================================================
#                 PAIR GEOMETRY / HELPERS
# ============================================================

def build_hd_pair_data(gamma_matrix: np.ndarray) -> HDPairData:
    """Build the pair index arrays and the N x N F_g matrix once."""
    n_star = int(gamma_matrix.shape[0])
    a_idx, b_idx = np.triu_indices(n_star, k=1)
    F = F_PHYS * np.array(gamma_matrix, dtype=float, copy=True)
    np.fill_diagonal(F, 0.0)
    Fab = F[a_idx, b_idx]

    if np.any(np.abs(Fab) < EPS):
        raise ValueError(
            "Some pairwise F_g values are too close to zero for the current "
            "matrix-free formulation. Check the geometry / gamma_parallel output."
        )

    return HDPairData(a_idx=a_idx, b_idx=b_idx, Fab=Fab, F=F)


# ============================================================
#                EXACT MATVEC FOR M(r) x
# ============================================================

def make_hd_matvec(data: HDPairData, r: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return an exact matrix-vector product for M(r).

    The pair-space operator is never formed explicitly. Instead we use the
    identity

        y_ab = S_ab / F_ab - U_aa - U_bb + row_sum[a] + row_sum[b]
               + (U_ab + U_ba)/(r F_ab) + x_ab/(r^2 F_ab^2)

    where
        X_ab = x_ab / F_ab for off-diagonal pair entries,
        U    = F X,
        S    = F X F,
        row_sum[i] = sum of pair amplitudes incident to star i.

    This reproduces the original A + B/r + D/r^2 algebra exactly.
    """
    a_idx = data.a_idx
    b_idx = data.b_idx
    Fab = data.Fab
    F = data.F
    n_pairs = data.n_pairs
    inv_r = 1.0 / float(r)
    inv_r2 = inv_r * inv_r

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size != n_pairs:
            raise ValueError(f"Expected vector of length {n_pairs}, got {x.shape}")

        # Symmetric pair matrix X with X_ab = x_ab / F_ab for a != b.
        X = np.zeros_like(F)
        x_scaled = x / Fab
        X[a_idx, b_idx] = x_scaled
        X[b_idx, a_idx] = x_scaled

        # U = F X, S = F X F.
        U = F @ X
        S = U @ F

        diag_U = np.diag(U)
        row_sum = np.bincount(a_idx, weights=x, minlength=F.shape[0])
        row_sum += np.bincount(b_idx, weights=x, minlength=F.shape[0])

        y = (
            S[a_idx, b_idx] / Fab
            - diag_U[a_idx]
            - diag_U[b_idx]
            + row_sum[a_idx]
            + row_sum[b_idx]
            + (U[a_idx, b_idx] + U[b_idx, a_idx]) * (inv_r / Fab)
            + x * (inv_r2 / (Fab * Fab))
        )
        return np.asarray(y, dtype=float)

    return matvec


# ============================================================
#                 DENSE FALLBACK FOR SMALL N
# ============================================================

def build_HD_matrices(gamma_matrix: np.ndarray):
    """Original dense decomposition (kept for small problems only)."""
    n_star = gamma_matrix.shape[0]
    pairs = np.array([(a, b) for a in range(n_star) for b in range(a + 1, n_star)])
    n_pairs = len(pairs)

    Fg_pair = F_PHYS * gamma_matrix[pairs[:, 0], pairs[:, 1]]
    Fg_mat = F_PHYS * gamma_matrix

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

    case3 = ac & bd
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

    A = np.ones((n_pairs, n_pairs), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        A_c1 = (Fg_ac * Fg_bd + Fg_ad * Fg_bc) / (Fg_ab * Fg_cd)
    A[case1] = A_c1[case1]

    B = np.zeros((n_pairs, n_pairs), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        B[case2_ac] = (Fg_mat[b_i, d_j] / (Fg_ab * Fg_mat[a_i, d_j]))[case2_ac]
        B[case2_bc] = (Fg_mat[a_i, d_j] / (Fg_ab * Fg_mat[b_i, d_j]))[case2_bc]
        B[case2_ad] = (Fg_mat[b_i, c_j] / (Fg_ab * Fg_mat[a_i, c_j]))[case2_ad]
        B[case2_bd] = (Fg_mat[a_i, c_j] / (Fg_ab * Fg_mat[b_i, c_j]))[case2_bd]

    D = np.zeros((n_pairs, n_pairs), dtype=float)
    np.fill_diagonal(D, 1.0 / Fg_pair**2)

    return pairs, A, B, D


def rho_hd_full_matrix_dense(x_arr, gamma_matrix, svd_rcond=1e-10, verbose=True):
    """Original dense eigendecomposition path."""
    if verbose:
        print("Building HD covariance matrices A, B, C, D...", flush=True)
    t0 = time.time()
    _, A, B, D = build_HD_matrices(gamma_matrix)
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
    svd_rcond : float
        Used as the relative tolerance for the iterative solver in the large-N
        path, and as the eigenvalue cutoff in the dense fallback.
    dense_cutover_pairs : int
        Use the dense path only when N_pairs <= this threshold.
    maxiter : int or None
        Maximum MINRES iterations per r-value. None lets SciPy choose.
    """
    x_arr = np.asarray(x_arr, dtype=float)
    data = build_hd_pair_data(gamma_matrix)

    if data.n_pairs <= dense_cutover_pairs:
        if verbose:
            print(
                f"Using dense fallback path (N_pairs={data.n_pairs} <= {dense_cutover_pairs})",
                flush=True,
            )
        return rho_hd_full_matrix_dense(x_arr, gamma_matrix, svd_rcond=svd_rcond, verbose=verbose)

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

        # Diagonal of M(r): 1 + 1/(r^2 F_ab^2)
        diag = 1.0 + 1.0 / (r * r * data.Fab * data.Fab)
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

def print_snr_diagnostics(r_values, rho_cp, rho_hd, ell_min, ell_max, n_stars=N_STARS):
    """
    Print weak/strong-signal slopes and plateau values for CP and HD curves.

    Slopes computed via log-log linear regression over designated windows.
    CP plateau is analytic (1/gamma0); HD plateau is numeric only, since the
    full-matrix formulation has no simple closed form like the diagonal approx.
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

    # CP plateau: analytic = 1/gamma0
    cp_plateau_anal = 1.0 / max(abs(gamma0), EPS)
    cp_plateau_num  = float(np.max(rho_cp))

    # HD plateau: no simple closed form for the full-matrix case.
    # The diagonal-approx analytic value sqrt(N*(N-1)) is printed as a reference,
    # but the full-matrix result will generally differ.
    hd_plateau_anal_diag = np.sqrt(n_stars * (n_stars - 1))
    hd_plateau_num       = float(np.max(rho_hd))

    print('\n' + '='*60)
    print('              SNR CURVE DIAGNOSTICS (full matrix)')
    print('='*60)

    print('\n── Common Process (CP) ──')
    print(f'  Weak-signal slope   (r ~ 1e-13 to 1e-11):  {slope_cp_weak:+.3f}  (expect +1.0)')
    print(f'  Strong-signal slope (r ~ 1e-2  to 1e+1 ):  {slope_cp_strong:+.3f}  (expect ~0)')
    print(f'  Plateau [analytic]  = 1/gamma0             = {cp_plateau_anal:.4f}')
    print(f'  Plateau [numeric ]  = max(rho_CP)          = {cp_plateau_num:.4f}')

    print('\n── Hellings-Downs (HD) — full covariance matrix ──')
    print(f'  Weak-signal slope   (r ~ 1e-13 to 1e-11):  {slope_hd_weak:+.3f}  (expect +1.0)')
    print(f'  Strong-signal slope (r ~ 1e-2  to 1e+1 ):  {slope_hd_strong:+.3f}  (expect ~0)')
    print(f'  Plateau [numeric ]  = max(rho_HD)          = {hd_plateau_num:.4f}')
    print('='*60 + '\n')


# ============================================================
#                          PLOT
# ============================================================

def plot_full_comparison(gamma_matrix, ell_min, ell_max, n_r=60, save_path=None):
    """Plot CP and HD SNR on the same axes."""
    r_values = np.logspace(-13, 2, n_r)

    print("Computing CP full curve...", flush=True)
    rho_cp = rho_cp_full(r_values, ell_min, ell_max)

    print("\nComputing HD full curve...", flush=True)
    rho_hd = rho_hd_full_matrix(r_values, gamma_matrix, verbose=True)

    print(f"\nPhysical r = P_gw/P_n = {PHYSICAL_RATIO:.3e}")
    print_snr_diagnostics(r_values, rho_cp, rho_hd, ell_min, ell_max, n_stars=gamma_matrix.shape[0])


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

    gamma = gamma_parallel(theta_mat, ell_min, ell_max)

    # Tag the output filename with N/FoV so multiple Slurm array tasks
    # (different N_STARS / FIELD_SIZE_DEG) don't overwrite each other's plot.
    out_name = f"hd_full_matrix_snr_N{N_STARS}_FoV{FIELD_SIZE_DEG:g}.png"

    r_vals, rho_cp, rho_hd = plot_full_comparison(
        gamma,
        ell_min,
        ell_max,
        n_r=30,
        save_path=out_name,
    )

    # Save the underlying arrays alongside the plot, tagged with the same
    # N/FoV convention as the PNG filename, so future runs can be compared
    # and overlaid using the real data rather than extracting curves from
    # the rendered image pixels.
    data_name = f"hd_full_matrix_snr_N{N_STARS}_FoV{FIELD_SIZE_DEG:g}.npz"
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
    )
    print(f"Data saved to {data_name}")

    print("\nSelected output:")
    print(f"{'r':>10}  {'rho_CP':>10}  {'rho_HD':>10}")
    for rv, rcp, rhd in zip(r_vals[::5], rho_cp[::5], rho_hd[::5]):
        print(f"  {rv:.2e}   {rcp:.4f}   {rhd:.4f}")