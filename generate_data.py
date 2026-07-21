"""
generate_data.py
================
Standalone, checkpointed data generator for the FoV-flip and constant-density
diagnostics. Run this once (in the background, however long it takes); the
analysis notebook only ever reads the .npz files this script produces, so
plotting/diagnosis stays fast and separate from the (variable-runtime) SNR
computation.

Checkpoints after every FoV / every combo, so an interruption only costs the
work since the last save -- same pattern as hd_spike_sweep_results.npz in
spike_diagnostics.ipynb.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from main import (
    build_star_positions, pairwise_theta, compute_ell_limits, gamma_parallel,
    cp_single_star_gamma, rho_cp_full, RANDOM_SEED,
)
from hd_full_matrix_snr import rho_hd_full_matrix, hd_strong_signal_plateau, F_PHYS

LOG = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_field(n_stars, field_size_deg, seed=RANDOM_SEED):
    stars = build_star_positions(None, n_stars=n_stars, field_size_deg=field_size_deg, seed=seed)
    theta = pairwise_theta(stars)
    ell_min, ell_max = compute_ell_limits(theta, field_size_deg)
    gamma = gamma_parallel(theta, ell_min, ell_max)
    return gamma, ell_min, ell_max


def weak_signal_sum(gamma_matrix):
    vals = gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    vals = vals[np.isfinite(vals)]
    return float(np.sum(vals**2))


def pair_diversity_stats(gamma_matrix):
    vals = F_PHYS * gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    vals = vals[np.isfinite(vals)]
    F0 = float(np.mean(vals))
    cv = float(np.std(vals) / F0) if F0 != 0 else np.nan
    return F0, cv


# ============================================================
#                SECTION A: FoV sweep, N fixed
# ============================================================

def run_fov_sweep(path="fov_sweep_results.npz", N_fixed=100,
                   FoV_values=(10, 20, 40, 60, 80, 100, 120, 140, 160),
                   n_r=30):
    r_grid = np.logspace(-13, 2, n_r)

    if os.path.exists(path):
        d = dict(np.load(path, allow_pickle=True))
        done = set(d["fov_done"].tolist()) if "fov_done" in d else set()
        LOG(f"[FoV sweep] resuming, {len(done)} FoV values already done")
    else:
        d = dict(
            r_grid=r_grid, N_fixed=np.array(N_fixed),
            fov_done=np.array([], dtype=int),
            fov_arr=np.array([], dtype=int),
            S_weak=np.array([]), F0=np.array([]), cv=np.array([]),
            plateau_numeric=np.array([]), plateau_meanfield=np.array([]),
            rho_hd_curves=np.zeros((0, n_r)), rho_cp_curves=np.zeros((0, n_r)),
        )
        done = set()

    for fov in FoV_values:
        if fov in done:
            continue
        t0 = time.time()
        gamma, ell_min, ell_max = build_field(N_fixed, fov)
        gamma0 = cp_single_star_gamma(ell_min, ell_max)
        rho_cp = rho_cp_full(r_grid, ell_min, ell_max, n_stars=N_fixed)
        rho_hd = rho_hd_full_matrix(r_grid, gamma, verbose=False)
        S_weak = weak_signal_sum(gamma)
        F0, cv = pair_diversity_stats(gamma)
        plateau_meanfield = hd_strong_signal_plateau(gamma)
        plateau_numeric = float(np.max(rho_hd))

        d["fov_done"] = np.append(d["fov_done"], fov)
        d["fov_arr"] = np.append(d["fov_arr"], fov)
        d["S_weak"] = np.append(d["S_weak"], S_weak)
        d["F0"] = np.append(d["F0"], F0)
        d["cv"] = np.append(d["cv"], cv)
        d["plateau_numeric"] = np.append(d["plateau_numeric"], plateau_numeric)
        d["plateau_meanfield"] = np.append(d["plateau_meanfield"], plateau_meanfield)
        d["rho_hd_curves"] = np.vstack([d["rho_hd_curves"], rho_hd[None, :]])
        d["rho_cp_curves"] = np.vstack([d["rho_cp_curves"], rho_cp[None, :]])

        np.savez(path, **d)
        LOG(f"[FoV sweep] FoV={fov:3d} done in {time.time()-t0:.1f}s "
            f"(ell_max={ell_max}, S_weak={S_weak:.3e}, plateau_num={plateau_numeric:.3f})")

    LOG("[FoV sweep] COMPLETE")


# ============================================================
#          SECTION B: joint N/FoV (constant density) sweep
# ============================================================

def run_joint_sweep(path="joint_sweep_results.npz",
                     k_values=(1, 2, 3, 4, 5, 6, 7, 8),
                     k_full_curve_max=8,
                     seeds=(RANDOM_SEED, RANDOM_SEED + 1),
                     n_r=30):
    r_grid = np.logspace(-13, 2, n_r)
    combos = [(10 * k * k, 10 * k) for k in k_values]

    if os.path.exists(path):
        d = dict(np.load(path, allow_pickle=True))
        done = set(d["k_done"].tolist()) if "k_done" in d else set()
        LOG(f"[Joint sweep] resuming, k already done: {sorted(done)}")
    else:
        d = dict(
            r_grid=r_grid, seeds=np.array(seeds),
            k_done=np.array([], dtype=int),
            k_arr=np.array([], dtype=int), N_arr=np.array([], dtype=int),
            FoV_arr=np.array([], dtype=int), n_pairs_arr=np.array([], dtype=int),
            S_weak_seeds=np.zeros((0, len(seeds))),
            F0_seeds=np.zeros((0, len(seeds))), cv_seeds=np.zeros((0, len(seeds))),
            has_full_curve=np.array([], dtype=bool),
            rho_hd_curves=np.zeros((0, n_r)), rho_cp_curves=np.zeros((0, n_r)),
        )
        done = set()

    for k, (N, fov) in zip(k_values, combos):
        if k in done:
            continue
        t0 = time.time()
        seed_S, seed_F0, seed_cv = [], [], []
        gamma_ref = ell_min_ref = ell_max_ref = None
        for seed in seeds:
            gamma, ell_min, ell_max = build_field(N, fov, seed=seed)
            seed_S.append(weak_signal_sum(gamma))
            F0, cv = pair_diversity_stats(gamma)
            seed_F0.append(F0); seed_cv.append(cv)
            if seed == seeds[0]:
                gamma_ref, ell_min_ref, ell_max_ref = gamma, ell_min, ell_max

        has_curve = k <= k_full_curve_max
        if has_curve:
            rho_cp = rho_cp_full(r_grid, ell_min_ref, ell_max_ref, n_stars=N)
            rho_hd = rho_hd_full_matrix(r_grid, gamma_ref, verbose=False)
        else:
            rho_cp = np.full(n_r, np.nan)
            rho_hd = np.full(n_r, np.nan)

        d["k_done"] = np.append(d["k_done"], k)
        d["k_arr"] = np.append(d["k_arr"], k)
        d["N_arr"] = np.append(d["N_arr"], N)
        d["FoV_arr"] = np.append(d["FoV_arr"], fov)
        d["n_pairs_arr"] = np.append(d["n_pairs_arr"], N * (N - 1) // 2)
        d["S_weak_seeds"] = np.vstack([d["S_weak_seeds"], np.array(seed_S)[None, :]])
        d["F0_seeds"] = np.vstack([d["F0_seeds"], np.array(seed_F0)[None, :]])
        d["cv_seeds"] = np.vstack([d["cv_seeds"], np.array(seed_cv)[None, :]])
        d["has_full_curve"] = np.append(d["has_full_curve"], has_curve)
        d["rho_hd_curves"] = np.vstack([d["rho_hd_curves"], rho_hd[None, :]])
        d["rho_cp_curves"] = np.vstack([d["rho_cp_curves"], rho_cp[None, :]])

        np.savez(path, **d)
        LOG(f"[Joint sweep] k={k} (N={N}, FoV={fov}) done in {time.time()-t0:.1f}s "
            f"(full_curve={has_curve}, S_weak(seed0)={seed_S[0]:.3e})")

    LOG("[Joint sweep] COMPLETE")


# ============================================================
#   SECTION C: full eigenspectrum of M(r) at large r vs FoV
#   -- quantifies where the uniform-Gamma_ab (Johnson-scheme)
#      ansatz stops describing the real covariance.
# ============================================================

def run_eigenspectrum_diagnostic(path="eigenspectrum_results.npz", N_fixed=100,
                                  FoV_values=(10, 20, 40, 60, 80, 100, 120, 140, 160),
                                  r_probe=100.0):
    from hd_full_matrix_snr import build_HD_matrices

    if os.path.exists(path):
        d = dict(np.load(path, allow_pickle=True))
        done = set(d["fov_done"].tolist()) if "fov_done" in d else set()
        LOG(f"[Eigenspectrum] resuming, {len(done)} FoV values already done")
    else:
        d = dict(
            N_fixed=np.array(N_fixed), r_probe=np.array(r_probe),
            fov_done=np.array([], dtype=int), fov_arr=np.array([], dtype=int),
            F0=np.array([]), n_neg=np.array([], dtype=int),
            eig_min=np.array([]), eig_max=np.array([]),
            mu1_pred=np.array([]), mu2_pred=np.array([]),
        )
        done = set()

    for fov in FoV_values:
        if fov in done:
            continue
        t0 = time.time()
        gamma, ell_min, ell_max = build_field(N_fixed, fov)
        F0, cv = pair_diversity_stats(gamma)

        _, A, B, D = build_HD_matrices(gamma)
        M = A + B / r_probe + D / r_probe**2
        w = np.linalg.eigvalsh(M)

        x_inf = 1.0 / F0
        mu1_pred = (x_inf - 1) * (x_inf + N_fixed - 3)
        mu2_pred = (x_inf - 1) ** 2

        d["fov_done"] = np.append(d["fov_done"], fov)
        d["fov_arr"] = np.append(d["fov_arr"], fov)
        d["F0"] = np.append(d["F0"], F0)
        d["n_neg"] = np.append(d["n_neg"], int(np.sum(w < 0)))
        d["eig_min"] = np.append(d["eig_min"], float(w.min()))
        d["eig_max"] = np.append(d["eig_max"], float(w.max()))
        d["mu1_pred"] = np.append(d["mu1_pred"], mu1_pred)
        d["mu2_pred"] = np.append(d["mu2_pred"], mu2_pred)

        np.savez(path, **d)
        LOG(f"[Eigenspectrum] FoV={fov:3d} done in {time.time()-t0:.1f}s "
            f"(F0={F0:.3f}, n_neg={int(np.sum(w<0))}, eig_min={w.min():.2e}, eig_max={w.max():.2e})")

    LOG("[Eigenspectrum] COMPLETE")


# ============================================================
#   SECTION D: multi-seed strong-signal plateau check for the
#   joint N/FoV combos (tests whether the plateau ordering is
#   a robust geometric effect or single-draw noise)
# ============================================================

def run_joint_plateau_seed_check(path="joint_plateau_seeds.npz",
                                  k_values=(1, 2, 3, 4, 5, 6, 7, 8),
                                  seeds=(RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2),
                                  n_r=30):
    # IMPORTANT: rho_hd_full_matrix warm-starts its iterative (MINRES) solve at each r
    # from the solution at the previous (smaller) r in the array. A cold, single-point
    # call at large r (no warm-start chain) can converge to a substantially different,
    # less accurate answer for an ill-conditioned M(r) -- confirmed directly: for
    # (N=100, FoV=80, seed=1234), a single-point r=[100] call gives 3.63, while the
    # correct, full-sequential-r_grid call (matching the production sweep) gives 6.17.
    # So this function must always solve the FULL r_grid per seed and read off the last
    # value, exactly mirroring how the production sweep computes its curves.
    r_grid = np.logspace(-13, 2, n_r)
    combos = [(10 * k * k, 10 * k) for k in k_values]

    if os.path.exists(path):
        d = dict(np.load(path, allow_pickle=True))
        done = set(tuple(row) for row in d["done_pairs"]) if "done_pairs" in d else set()
        LOG(f"[Joint plateau seeds] resuming, {len(done)} (k,seed) pairs already done")
    else:
        d = dict(
            r_grid=r_grid,
            done_pairs=np.zeros((0, 2), dtype=int),
            k_arr=np.array([], dtype=int), seed_arr=np.array([], dtype=int),
            N_arr=np.array([], dtype=int), FoV_arr=np.array([], dtype=int),
            F0_arr=np.array([]), plateau_arr=np.array([]),
            meanfield_arr=np.array([]),
        )
        done = set()

    for k, (N, fov) in zip(k_values, combos):
        for seed in seeds:
            if (k, seed) in done:
                continue
            t0 = time.time()
            gamma, ell_min, ell_max = build_field(N, fov, seed=seed)
            F0, cv = pair_diversity_stats(gamma)
            rho_hd_curve = rho_hd_full_matrix(r_grid, gamma, verbose=False)
            plateau = float(rho_hd_curve[-1])
            meanfield = hd_strong_signal_plateau(gamma)

            d["done_pairs"] = np.vstack([d["done_pairs"], [[k, seed]]])
            d["k_arr"] = np.append(d["k_arr"], k)
            d["seed_arr"] = np.append(d["seed_arr"], seed)
            d["N_arr"] = np.append(d["N_arr"], N)
            d["FoV_arr"] = np.append(d["FoV_arr"], fov)
            d["F0_arr"] = np.append(d["F0_arr"], F0)
            d["plateau_arr"] = np.append(d["plateau_arr"], plateau)
            d["meanfield_arr"] = np.append(d["meanfield_arr"], meanfield)

            np.savez(path, **d)
            LOG(f"[Joint plateau seeds] k={k} seed={seed} done in {time.time()-t0:.1f}s "
                f"(plateau={plateau:.4f}, meanfield={meanfield:.4f})")

    LOG("[Joint plateau seeds] COMPLETE")


# ============================================================
#   SECTION E: multi-seed plateau check for the FoV sweep
#   (tests whether the FoV=100 / FoV=140 "dips" are real or
#   single-realization noise)
# ============================================================

def run_fov_plateau_seed_check(path="fov_plateau_seeds.npz",
                                N_fixed=100,
                                FoV_values=(60, 80, 100, 120, 140, 160),
                                seeds=(RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2),
                                n_r=30):
    # Same fix as run_joint_plateau_seed_check: must solve the full, sequential,
    # warm-started r_grid per seed (matching the production sweep) rather than a
    # cold single-point call, or the plateau value will be unreliable.
    r_grid = np.logspace(-13, 2, n_r)

    if os.path.exists(path):
        d = dict(np.load(path, allow_pickle=True))
        done = set(tuple(row) for row in d["done_pairs"]) if "done_pairs" in d else set()
        LOG(f"[FoV plateau seeds] resuming, {len(done)} (FoV,seed) pairs already done")
    else:
        d = dict(
            r_grid=r_grid, N_fixed=np.array(N_fixed),
            done_pairs=np.zeros((0, 2), dtype=int),
            fov_arr=np.array([], dtype=int), seed_arr=np.array([], dtype=int),
            F0_arr=np.array([]), plateau_arr=np.array([]),
        )
        done = set()

    for fov in FoV_values:
        for seed in seeds:
            if (fov, seed) in done:
                continue
            t0 = time.time()
            gamma, ell_min, ell_max = build_field(N_fixed, fov, seed=seed)
            F0, cv = pair_diversity_stats(gamma)
            rho_hd_curve = rho_hd_full_matrix(r_grid, gamma, verbose=False)
            plateau = float(rho_hd_curve[-1])

            d["done_pairs"] = np.vstack([d["done_pairs"], [[fov, seed]]])
            d["fov_arr"] = np.append(d["fov_arr"], fov)
            d["seed_arr"] = np.append(d["seed_arr"], seed)
            d["F0_arr"] = np.append(d["F0_arr"], F0)
            d["plateau_arr"] = np.append(d["plateau_arr"], plateau)

            np.savez(path, **d)
            LOG(f"[FoV plateau seeds] FoV={fov} seed={seed} done in {time.time()-t0:.1f}s "
                f"(plateau={plateau:.4f})")

    LOG("[FoV plateau seeds] COMPLETE")


if __name__ == "__main__":
    LOG("Starting FoV sweep...")
    run_fov_sweep()
    LOG("Starting joint sweep...")
    run_joint_sweep()
    LOG("Starting eigenspectrum diagnostic...")
    run_eigenspectrum_diagnostic()
    LOG("Starting joint plateau seed check...")
    run_joint_plateau_seed_check()
    LOG("Starting FoV plateau seed check...")
    run_fov_plateau_seed_check()
    LOG("ALL DONE")
    with open("GENERATION_COMPLETE.flag", "w") as f:
        f.write("done\n")
