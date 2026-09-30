import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend needed for Slurm jobs
import matplotlib.pyplot as plt
import astropy.units as u


# ============================================================
#                        CONSTANTS
# ============================================================

# 1 mas astrometric noise, converted to radians
sigma_rad = (1 * u.mas).to(u.rad).value
# 30-minute Kepler cadence, converted to seconds
dt_seconds = (30 * u.min).to(u.s).value
# 3.5-year observation window, converted to seconds
T_obs_seconds = (3.5 * u.yr).to(u.s).value

# Low-frequency cutoff = 1/T_obs, Nyquist = 1/(2*dt)
f_l = 1.0 / T_obs_seconds
f_h = 1.0 / (2.0 * dt_seconds)

# Noise PSD: P_n = 2 * sigma_rad^2 * dt
P_n = 2.0 * sigma_rad**2 * dt_seconds

# Reference frequency: 1/year in Hz
f_yr = (1 / u.yr).to(u.Hz).value

# GW amplitude (NANOGrav value)
A_gw = 1e-15

# sigma_bar^2 = P_n — all stars have identical noise
sigma_bar_sq = P_n

# Field parameters
FIELD_SIZE_DEG = 10
N_STARS        = 900
STAR_COORDS_DEG = None
RANDOM_SEED     = 1234

# Batch size for gamma_parallel_matrix's Legendre-recurrence chunking.
# This only affects memory/runtime, never the computed gamma values
GAMMA_BATCH_SIZE = 500

#For the overwrites in the slurm file
import os as _os
if _os.environ.get('SLURM_N_STARS'):
    N_STARS = int(_os.environ['SLURM_N_STARS'])
if _os.environ.get('SLURM_FIELD_SIZE_DEG'):
    FIELD_SIZE_DEG = float(_os.environ['SLURM_FIELD_SIZE_DEG'])
if _os.environ.get('SLURM_GAMMA_BATCH_SIZE'):
    GAMMA_BATCH_SIZE = int(_os.environ['SLURM_GAMMA_BATCH_SIZE'])
# ------------------------------------------------------------

# ------------------------------------------------------------
#              GAMMA NORMALIZATION TOGGLE
# ------------------------------------------------------------
# False (default): gamma_parallel returns the raw multipole-sum overlap
#   function exactly as originally implemented (Gamma(theta=0) ~ 0.0066).
# True: gamma_parallel self-normalizes so Gamma(theta=0) = 1 exactly
#   (Gamma_raw(theta) / Gamma_raw(0)). This is the same status 
#   chi(0)=0.5 has for Hellings & Downs.

NORMALIZED_GAMMA = True
if _os.environ.get('SLURM_NORMALIZED_GAMMA'):
    NORMALIZED_GAMMA = _os.environ['SLURM_NORMALIZED_GAMMA'].strip().lower() in ('1', 'true', 'yes')

F_PHYS = 192.0 * np.pi**3


def gamma_scale_factor():
    return F_PHYS
# ------------------------------------------------------------

EPS = 1e-14

# ------------------------------------------------------------
#          PARALLEL + PERPENDICULAR POLARIZATION FACTOR
# ------------------------------------------------------------
# Each star measures both the parallel and perpendicular deflection, and
# Gamma_perp = Gamma_par, so the GW cross-power picks up a factor of 2:
#     P_ab = POL_FACTOR * F * P_gw * Gamma_par(Theta_ab)
# POL_FACTOR multiplies P_ab ONLY. It is not applied to Gamma, to F, to the
# amplitude-estimator normalization, or to (A_bar^2)^2.
# Set to 1.0 to recover the parallel-only result.
POL_FACTOR = 2.0
if _os.environ.get('SLURM_POL_FACTOR'):
    POL_FACTOR = float(_os.environ['SLURM_POL_FACTOR'])

# Physical GW power spectrum at f_l
# P_gw(f) = A_gw^2 / (12*pi^2) * (f/f_yr)^(-4/3) * f^(-1)
P_gw_fl = (A_gw**2 / (12.0 * np.pi**2)) * (f_l / f_yr)**(-4.0/3.0) / f_l

# Physical operating point on the x-axis used for vertical marker on plots
# This is the actual value of P_gw/P_n with all constants plugged in
PHYSICAL_RATIO = P_gw_fl / P_n


# ============================================================
#                      STAR POSITIONS
# ============================================================

def build_star_positions(star_coords_deg=None, n_stars=N_STARS,
                         field_size_deg=FIELD_SIZE_DEG, seed=RANDOM_SEED,
                         center_ra_deg=0.0, center_dec_deg=0.0):
    """
    Sample n_stars uniformly within an angular-diameter patch of
    field_size_deg, centered at (center_ra_deg, center_dec_deg), using
    true spherical (great-circle) geometry.

    field_size_deg=360 (or >=180) samples the FULL SKY uniformly.
    Smaller field_size_deg samples a circular patch of that angular
    diameter.

    Returns an (n_stars, 2) array of [RA_deg, Dec_deg].
    """
    if star_coords_deg is not None:
        stars = np.asarray(star_coords_deg, dtype=float)
        if stars.ndim != 2 or stars.shape[1] != 2:
            raise ValueError('star_coords_deg must have shape (N, 2).')
        return stars

    rng = np.random.default_rng(seed)
    half_rad = np.deg2rad(min(field_size_deg, 360.0) / 2.0)
    half_rad = min(half_rad, np.pi)  # cap at full sky

    # Uniform sampling within angular radius 'half_rad' of the north pole:
    # cos(theta) uniform in [cos(half_rad), 1] gives uniform AREA density
    # on the spherical cap (not uniform theta, which would oversample the
    # center).
    cos_theta = rng.uniform(np.cos(half_rad), 1.0, size=n_stars)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0, 2 * np.pi, size=n_stars)

    # Cartesian coords of the cap, centered on the north pole (0,0,1).
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Rotate the cap from the north pole down to (center_ra_deg, center_dec_deg):
    # first tilt about the y-axis by (90deg - dec0), then rotate about the
    # z-axis by ra0.
    dec0 = np.deg2rad(center_dec_deg)
    ra0 = np.deg2rad(center_ra_deg)

    tilt = (np.pi / 2.0) - dec0
    x1 = x * np.cos(tilt) + z * np.sin(tilt)
    y1 = y
    z1 = -x * np.sin(tilt) + z * np.cos(tilt)

    x2 = x1 * np.cos(ra0) - y1 * np.sin(ra0)
    y2 = x1 * np.sin(ra0) + y1 * np.cos(ra0)
    z2 = z1

    dec = np.degrees(np.arcsin(np.clip(z2, -1.0, 1.0)))
    ra = np.degrees(np.arctan2(y2, x2)) % 360.0

    return np.column_stack([ra, dec])


def pairwise_theta(stars_deg):
    """
    True pairwise angular separation (radians) between every pair of
    stars on the sphere, via the haversine formula. 

    stars_deg: (N, 2) array of [RA_deg, Dec_deg].
    """
    ra = np.deg2rad(stars_deg[:, 0])
    dec = np.deg2rad(stars_deg[:, 1])

    dra = ra[:, None] - ra[None, :]
    ddec = dec[:, None] - dec[None, :]

    a = (
        np.sin(ddec / 2.0) ** 2
        + np.cos(dec[:, None]) * np.cos(dec[None, :]) * np.sin(dra / 2.0) ** 2
    )
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    np.fill_diagonal(theta, np.nan)
    return theta


def compute_ell_limits(theta_matrix, field_size_deg):
    """
    ell_min = 2 (per Kris suggestion)
    ell_max = 2*pi / min_angular_separation (in radians)
    """
    ell_min = 2
    finite_seps = theta_matrix[np.isfinite(theta_matrix) & (theta_matrix > 0)]
    min_sep_rad  = np.min(finite_seps)
    ell_max = int(np.floor(2.0 * np.pi / min_sep_rad))
    return ell_min, ell_max


# ============================================================
#               VECTORIZED LEGENDRE RECURRENCE
# ============================================================

def compute_legendre_recurrence(mu, ell_max):
    """
    Compute P_l(mu), P_l^1(mu), P_l^2(mu) for all l up to ell_max
    using the standard 3-term recurrence relation.
    Returns arrays of shape (ell_max+1, *mu.shape).
    """
    shape = mu.shape
    P0 = np.zeros((ell_max + 1,) + shape)
    P1 = np.zeros((ell_max + 1,) + shape)
    P2 = np.zeros((ell_max + 1,) + shape)

    sin_t = np.sqrt(np.maximum(1.0 - mu**2, 0.0))

    P0[0] = 1.0
    P0[1] = mu
    P1[1] = -sin_t
    P2[2] = 3.0 * sin_t**2

    for l in range(1, ell_max):
        P0[l + 1] = ((2*l + 1) * mu * P0[l] - l * P0[l - 1]) / (l + 1)
        if l >= 1:
            P1[l + 1] = (
                ((2*l + 1) * mu * P1[l] - (l + 1) * P1[l - 1]) / l
                if l >= 2 else (2*l + 1) * mu * P1[l]
            )
        if l >= 2:
            P2[l + 1] = (
                ((2*l + 1) * mu * P2[l] - (l + 2) * P2[l - 1]) / (l - 1)
                if l >= 3 else (2*l + 1) * mu * P2[l]
            )

    return P0, P1, P2


# ============================================================
#                         G KERNELS
# ============================================================

def G1(ell, P0_ell, P2_ell):
    """G_l^(1)(Theta) = -1/2 * [P_l^2(cos Theta)/(l(l+1)) - P_l(cos Theta)]"""
    ll1 = ell * (ell + 1.0)
    return -0.5 * (P2_ell / ll1 - P0_ell)


def G2(ell, P1_ell, theta):
    """
    G_l^(2)(Theta) = -1/(l(l+1)) * P_l^1(cos Theta)/sin(Theta)
    Singularities: theta->0 gives +0.5, theta->pi gives 0.
    """
    ll1   = ell * (ell + 1.0)
    sin_t = np.sin(theta)

    mask_zero = np.abs(theta) < 1e-12
    mask_pi   = np.abs(theta - np.pi) < 1e-12
    mask_reg  = ~mask_zero & ~mask_pi

    g2 = np.zeros_like(theta)
    g2[mask_reg]  = -P1_ell[mask_reg] / (ll1 * sin_t[mask_reg])
    g2[mask_zero] = 0.5
    g2[mask_pi]   = 0.0
    return g2


# ============================================================
#                  MODE COUPLING COEFFICIENT
# ============================================================

def F_sq(ell):
    """
    |F_l^E|^2 = |F_l^B|^2 = 1 / (N_l^2 * l(l+1))
    N_l^2 = (l+2)(l+1)l(l-1) / 2
    """
    N_sq = ((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0)) / 2.0
    return 1.0 / (N_sq * ell * (ell + 1.0))


# ============================================================
#                  GAMMA OVERLAP FUNCTION
# ============================================================

def _gamma_parallel_raw(theta, ell_min, ell_max):
    """
    Gamma_o^parallel(Theta) = sum_{l=ell_min}^{ell_max}
        (2l+1)/(4pi) * F_sq(l) * (G1_l(Theta) + G2_l(Theta))

    This is the RAW overlap function with no normalization applied.
    Gamma_raw(0) ~ 0.0066, not 1 (see gamma_parallel() below for the
    self-normalized version). Do not call this directly outside of
    gamma_parallel(); it exists only so gamma_parallel() can evaluate the
    raw sum at theta=0 without recursing into its own normalization step.
    """
    theta = np.clip(np.asarray(theta, dtype=float), 0, np.pi)
    mu    = np.cos(theta)
    sin_t = np.sqrt(np.maximum(1.0 - mu**2, 0.0))

    total = np.zeros_like(theta)

    # l=0 values (same initialization as compute_legendre_recurrence: P0[0]=1, P1[0]=0, P2[0]=0)
    P0_l, P1_l, P2_l = np.ones_like(mu), np.zeros_like(mu), np.zeros_like(mu)

    # l=1 values (same initialization: P0[1]=mu, P1[1]=-sin_t, P2[1]=0)
    P0_lp1, P1_lp1, P2_lp1 = mu.copy(), -sin_t.copy(), np.zeros_like(mu)

    for ell in (0, 1):
        if ell_min <= ell <= ell_max:
            P0_ell, P1_ell, P2_ell = (P0_l, P1_l, P2_l) if ell == 0 else (P0_lp1, P1_lp1, P2_lp1)
            g1     = G1(ell, P0_ell, P2_ell)
            g2     = G2(ell, P1_ell, theta)
            weight = (2.0 * ell + 1.0) / (4.0 * np.pi) * F_sq(ell)
            total += weight * (g1 + g2)

    # Advance the 3-term recurrence one l at a time, accumulating the
    # weighted sum immediately instead of storing every l.
    for l in range(1, ell_max):
        P0_next = ((2*l + 1) * mu * P0_lp1 - l * P0_l) / (l + 1)

        if l == 1:
            P1_next = (2*l + 1) * mu * P1_lp1
        else:
            P1_next = ((2*l + 1) * mu * P1_lp1 - (l + 1) * P1_l) / l

        if l + 1 == 2:
            P2_next = 3.0 * sin_t**2          # matches original P2[2] initialization (set before loop, l=1 never touches it)
        elif l >= 3:
            P2_next = ((2*l + 1) * mu * P2_lp1 - (l + 2) * P2_l) / (l - 1)
        elif l == 2:
            P2_next = (2*l + 1) * mu * P2_lp1  # matches original's "if l>=2: ... else: (2l+1)*mu*P2[l]" branch
        else:
            P2_next = np.zeros_like(mu)       # l+1 < 2: P2 not yet defined, matches original zeros

        ell = l + 1
        if ell_min <= ell <= ell_max:
            g1     = G1(ell, P0_next, P2_next)
            g2     = G2(ell, P1_next, theta)
            weight = (2.0 * ell + 1.0) / (4.0 * np.pi) * F_sq(ell)
            total += weight * (g1 + g2)

        P0_l, P0_lp1 = P0_lp1, P0_next
        P1_l, P1_lp1 = P1_lp1, P1_next
        P2_l, P2_lp1 = P2_lp1, P2_next

    return total


def gamma_parallel(theta, ell_min, ell_max):
    """
    Public overlap-function entry point. Every other function in this
    project (gamma_parallel_matrix, cp_single_star_gamma, and everything
    downstream in hd_full_matrix_snr.py) calls this, not
    _gamma_parallel_raw, so the NORMALIZED_GAMMA toggle applies everywhere
    consistently.

    NORMALIZED_GAMMA = False (default): returns _gamma_parallel_raw(theta)
        unchanged, Gamma(0) ~ 0.0066.
    NORMALIZED_GAMMA = True: returns _gamma_parallel_raw(theta) /
        _gamma_parallel_raw(0), so Gamma(0) = 1 exactly.
    """
    raw = _gamma_parallel_raw(theta, ell_min, ell_max)
    if not NORMALIZED_GAMMA:
        return raw
    gamma0_raw = _gamma_parallel_raw(np.array([0.0]), ell_min, ell_max)[0]
    return raw / gamma0_raw


def cp_single_star_gamma(ell_min, ell_max):
    """
    Single-star CP overlap at zero separation: Gamma_o(0).
    Used in the Sherman-Morrison reduction of the N x N CP covariance.
    Independent of N and of the star field layout.
    """
    return float(gamma_parallel(np.array([0.0]), ell_min, ell_max)[0])


def gamma_parallel_matrix(theta_matrix, ell_min, ell_max, batch_size=None):
    """
    Compute the full N x N gamma_parallel matrix memory-safely by processing
    unique pairs in batches rather than passing the full N x N array to
    compute_legendre_recurrence at once.

    Returns the full symmetric N x N gamma matrix (NaN on diagonal,
    consistent with the output of gamma_parallel called on the full matrix).

    batch_size : int or None
        Number of pairs processed per Legendre-recurrence call. Smaller
        values use less peak memory per call (more, smaller calls);
        larger values are faster but use more memory per call. This is
        purely a memory/runtime tradeoff — verified to produce identical
        output regardless of batch_size. Defaults to GAMMA_BATCH_SIZE.
    """
    if batch_size is None:
        batch_size = GAMMA_BATCH_SIZE

    N = theta_matrix.shape[0]

    # Extract unique upper-triangle pairs
    rows, cols   = np.triu_indices(N, k=1)
    theta_pairs  = theta_matrix[rows, cols]          # shape (N_pairs,)
    gamma_pairs  = np.zeros(len(theta_pairs))

    n_pairs = len(theta_pairs)
    for start in range(0, n_pairs, batch_size):
        end    = min(start + batch_size, n_pairs)
        g_batch = gamma_parallel(theta_pairs[start:end], ell_min, ell_max)
        gamma_pairs[start:end] = g_batch

    # Reconstruct symmetric N x N matrix
    gamma_mat = np.full((N, N), np.nan)
    gamma_mat[rows, cols] = gamma_pairs
    gamma_mat[cols, rows] = gamma_pairs   # symmetry

    return gamma_mat


# ============================================================
#     SNR CALCULATIONS
#
#  Common Process (CP):
#    - Estimator: one per STAR (auto-correlation); covariance is N x N.
#    - Full N x N covariance inverted exactly (rho_cp_full); the
#      Sherman-Morrison closed form is kept only for comparison.
#
#  Hellings-and-Downs (HD):
#    - Estimator: one per PAIR (a<b) (cross-correlation); covariance is
#      N_pairs x N_pairs.
# ============================================================

def rho_cp_weak(x, n_stars=N_STARS):
    """
    CP weak-signal asymptote: rho ~ sqrt(N)*F*r.
    x = r = P_gw/P_n (dimensionless). 
    """
    factor = gamma_scale_factor()
    return np.sqrt(n_stars) * factor * np.asarray(x, dtype=float)


def rho_cp_intermediate(gamma0):
    """
    CP plateau IN THE UNIFORM-GAMMA LIMIT: rho -> 1/(POL_FACTOR*gamma0).
    Only exact when Gamma_ab = gamma0 for every pair (the Sherman-Morrison
    assumption). For the real field use rho_cp_strong_plateau(), which
    depends on N and on the star layout.
    """
    return 1.0 / max(abs(POL_FACTOR * gamma0), EPS)


# ------------------------------------------------------------
#   CP: exact inversion of the full N x N covariance
# ------------------------------------------------------------
#
# From my notes, with the factor of 2 (POL_FACTOR = c) on P_ab ONLY:
#   P_ab    = 2 * F * P_gw * Gamma_ab                     (F = 192 pi^3)
#   C~_ab   = P_ab + sigma^2 delta_ab
#   C_ab    = f_l^{14/3} |C~_ab|^2 / F^2                  (Gaussian 4th moment)
#   rho^2   = (A_bar^2)^2 sum_ab (C^-1)_ab,  (A_bar^2)^2 = P_gw^2 f_l^{14/3}
# Gamma, F, the 1/F^2 estimator normalization and (A_bar^2)^2 carry no c.
# ------------------------------------------------------------

def cp_gamma_matrix(gamma_matrix, gamma0):
    """Copy of the pairwise Gamma matrix with the NaN diagonal set to
    gamma0 = Gamma(0), i.e. the auto-correlation each star has with itself."""
    Gam = np.array(gamma_matrix, dtype=float, copy=True)
    np.fill_diagonal(Gam, gamma0)
    if not np.all(np.isfinite(Gam)):
        raise ValueError('Gamma matrix has non-finite off-diagonal entries.')
    return Gam


def cp_covariance_eigensystem(gamma_matrix, gamma0, verbose=True):
    """
    Eigendecomposition of G = Gamma o Gamma (element-wise square).

    Returns (g, s2): eigenvalues g_k of G and s2_k = (1^T v_k)^2.

    G is the signal-dominated shape of the CP covariance, so it must be
    positive semidefinite for the covariance to be physical. A significantly
    negative eigenvalue is reported rather than silently clipped.
    """
    Gam = cp_gamma_matrix(gamma_matrix, gamma0)
    G = Gam * Gam                                   # element-wise |.|^2
    g, V = np.linalg.eigh(G)
    s2 = V.sum(axis=0) ** 2

    g_max = np.max(np.abs(g))
    g_min = float(np.min(g))
    if verbose:
        print(f'  CP covariance G = Gamma^2: eig range [{g_min:.3e}, {g_max:.3e}], '
              f'cond ~ {g_max / max(abs(g_min), EPS * g_max):.2e}')
    if g_min < -1e-8 * g_max:
        print(f'  WARNING: G has a negative eigenvalue ({g_min:.3e}). The CP '
              f'covariance is not positive semidefinite at large r; rho_CP '
              f'values near r ~ 1/(F*|g_min|) are not trustworthy.')
    return g, s2


def rho_cp_full(x_arr, gamma_matrix, ell_min, ell_max, verbose=True):
    """
    CP full SNR curve from the EXACT inverse of the full N x N covariance
    (no uniform-Gamma assumption). See the block comment above.

    x_arr        : r = P_gw / P_n values
    gamma_matrix : N x N pairwise Gamma_par matrix from gamma_parallel_matrix
                   (diagonal may be NaN; it is replaced by Gamma(0))
    """
    F      = gamma_scale_factor()
    c      = POL_FACTOR
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    x_arr  = np.asarray(x_arr, dtype=float)

    g, s2 = cp_covariance_eigensystem(gamma_matrix, gamma0, verbose=verbose)

    p   = c * F * x_arr                              # P_ab / (P_n Gamma_ab)
    lam = 2.0 * gamma0 / p + 1.0 / p**2              # shape (n_r,)

    denom  = g[None, :] + lam[:, None]               # shape (n_r, N)
    rho_sq = np.sum(s2[None, :] / denom, axis=1) / c**2   # (F r / p)^2 = 1/c^2
    return np.sqrt(np.maximum(rho_sq, 0.0))


def rho_cp_full_direct(x_arr, gamma_matrix, ell_min, ell_max):
    """
    Reference implementation: literally builds C_ab (up to the constant
    f_l^{14/3} P_n^2 / F^2, which cancels against (A_bar^2)^2) at each r
    and solves C y = 1. O(N^3) per r, so use it only to cross-check
    rho_cp_full at small N.
    """
    F      = gamma_scale_factor()
    c      = POL_FACTOR
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    Gam = cp_gamma_matrix(gamma_matrix, gamma0)
    N = Gam.shape[0]
    ones = np.ones(N)

    out = np.zeros(len(np.atleast_1d(x_arr)))
    for k, r in enumerate(np.atleast_1d(x_arr)):
        P_ab_over_Pn = c * F * r * Gam               # factor of 2 on P_ab only
        Ctilde = P_ab_over_Pn + np.eye(N)            # C~ / P_n
        C = Ctilde * Ctilde                          # element-wise |C~|^2
        y = np.linalg.solve(C, ones)
        out[k] = np.sqrt(max((F * r)**2 * float(ones @ y), 0.0))   # (A_bar^2)^2, no c
    return out


def rho_cp_full_sherman_morrison(x_arr, ell_min, ell_max, n_stars=N_STARS):
    """
    OLD closed form, kept only for comparison. Assumes Gamma_ab = gamma0
    for every pair, so C = alpha J + beta I and Sherman-Morrison applies:

    rho^2_CP = N*(F*r)^2 / [1 + 2*c*F*gamma0*r + N*(c*F*gamma0*r)^2]

    with c = POL_FACTOR on P_ab only. NOT correct for a real star field,
    where Gamma_ab varies with separation. Use rho_cp_full instead.
    """
    F      = gamma_scale_factor()
    c      = POL_FACTOR
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    x_arr  = np.asarray(x_arr, dtype=float)

    Fr  = F * x_arr
    pgr = c * F * gamma0 * x_arr                     # P_aa / P_n

    numer = n_stars * Fr**2
    denom = 1.0 + 2.0 * pgr + n_stars * pgr**2

    return np.sqrt(np.maximum(numer / denom, 0.0))


def rho_cp_signal_dominated(gamma_matrix, ell_min, ell_max):
    """
    The signal-dominated CP SNR, inverting exactly

        C_ab   = f_l^{14/3} (c P_gw)^2 Gamma(Theta_ab)^2      (c = POL_FACTOR on P_ab)
        rho^2  = (A_bar^2)^2 sum_ab (C^-1)_ab

    Since (A_bar^2)^2 = P_gw^2 f_l^{14/3}, the constants cancel and
    rho^2 = (1/c^2) sum_ab [(Gamma o Gamma)^-1]_ab. Solved with a linear
    solve (no pseudo-inverse cutoff); compare with rho_cp_strong_plateau,
    which drops near-null eigenmodes, to judge how much conditioning matters.
    """
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    Gam = cp_gamma_matrix(gamma_matrix, gamma0)
    G = Gam * Gam
    y = np.linalg.solve(G, np.ones(G.shape[0]))
    return float(np.sqrt(max(np.sum(y), 0.0)) / POL_FACTOR)


def rho_cp_strong_plateau(gamma_matrix, ell_min, ell_max, rcond=1e-10):
    """
    r -> infinity CP plateau with a pseudo-inverse: rho^2 = (1/c^2) 1^T G^+ 1,
    G = Gamma o Gamma. Reduces to 1/(c gamma0) only for uniform Gamma.
    Eigenvalues below rcond * max|g| are dropped.
    """
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    g, s2 = cp_covariance_eigensystem(gamma_matrix, gamma0, verbose=False)
    keep = g > rcond * np.max(np.abs(g))
    return float(np.sqrt(np.sum(s2[keep] / g[keep])) / POL_FACTOR)


def rho_hd_full(x_arr, gamma_matrix, ell_min, ell_max):
    """
    HD SNR via the diagonal (Case 3 only) approximation of the
    N_pairs x N_pairs covariance. For the full covariance use
    hd_full_matrix_snr.rho_hd_full_matrix.

    Pair estimator normalized by F*Gamma_ab (no c), so for pair (a,b):
        C_ab,ab / P_gw^2 = [P_a P_b + P_ab^2] / (P_gw F Gamma_ab)^2
        P_ab = c F P_gw Gamma_ab,   P_a = P_n + c F P_gw Gamma(0)
        rho^2_HD = sum_{a<b} 2 (P_gw F Gamma_ab)^2 / [P_ab^2 + P_a P_b]

    Asymptotes:
      Weak  : rho^2 -> 2 F^2 sum(Gamma_ab^2) r^2
      Strong: rho^2 -> (2/c^2) sum Gamma_ab^2 / (Gamma_ab^2 + gamma0^2)
    """
    F      = gamma_scale_factor()
    c      = POL_FACTOR
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    vals   = gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    gammas = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
    if gammas.size == 0:
        return np.zeros_like(np.asarray(x_arr, dtype=float))

    x_arr    = np.asarray(x_arr, dtype=float)
    P_gw_arr = x_arr[:, None] * P_n       # shape (n_r, 1)
    g        = gammas[None, :]            # shape (1, n_pairs)

    P_ab = c * F * P_gw_arr * g                       # cross-power, factor on P_ab
    P_a  = sigma_bar_sq + c * F * P_gw_arr * gamma0   # auto-power (a = b entry of P)
    norm = F * P_gw_arr * g                           # estimator normalization, no c

    rho_sq = np.sum(2.0 * norm**2 / (P_ab**2 + P_a**2), axis=1)
    return np.sqrt(np.maximum(rho_sq, 0.0))


def hd_diag_strong_plateau(gamma_matrix, ell_min, ell_max):
    """r -> infinity limit of rho_hd_full (diagonal approximation)."""
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    vals = gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    vals = vals[np.isfinite(vals)]
    return float(np.sqrt(2.0 * np.sum(vals**2 / (vals**2 + gamma0**2))) / POL_FACTOR)


def print_snr_diagnostics(x_arr, rho_cp, rho_hd, ell_min, ell_max, gamma_matrix, n_stars=N_STARS):
    """
    Print weak-signal slopes and plateau values for both CP and HD SNR curves.

    Slope is computed in log-log space via finite differences over a
    designated 'weak signal' window well below the physical operating point.
    Plateau is read both analytically and numerically (max of each curve).
    """
    gamma0    = cp_single_star_gamma(ell_min, ell_max)
    log_x     = np.log10(x_arr)
    log_rho_cp = np.log10(np.maximum(rho_cp, 1e-300))
    log_rho_hd = np.log10(np.maximum(rho_hd, 1e-300))

    # ── Weak-signal slope window: pick indices in x ~ [1e-12, 1e-10] ──
    # This sits well below the physical ratio (~6e-11) and below any plateau.
    weak_mask = (x_arr >= 1e-12) & (x_arr <= 1e-10)
    if weak_mask.sum() >= 2:
        slope_cp_weak = np.polyfit(log_x[weak_mask], log_rho_cp[weak_mask], 1)[0]
        slope_hd_weak = np.polyfit(log_x[weak_mask], log_rho_hd[weak_mask], 1)[0]
    else:
        slope_cp_weak = slope_hd_weak = float('nan')

    # ── Strong-signal slope window: pick indices in x ~ [1e-2, 1e1] ──
    # Both curves should be saturating here; slope -> 0 at a true plateau.
    strong_mask = (x_arr >= 1e-2) & (x_arr <= 1e1)
    if strong_mask.sum() >= 2:
        slope_cp_strong = np.polyfit(log_x[strong_mask], log_rho_cp[strong_mask], 1)[0]
        slope_hd_strong = np.polyfit(log_x[strong_mask], log_rho_hd[strong_mask], 1)[0]
    else:
        slope_cp_strong = slope_hd_strong = float('nan')

    # ── Analytical plateaus ──
    n_pairs          = n_stars * (n_stars - 1) // 2
    cp_plateau_anal  = rho_cp_strong_plateau(gamma_matrix, ell_min, ell_max)  # sqrt(1^T G^+ 1)
    cp_plateau_unif  = rho_cp_intermediate(gamma0)            # 1/(c*gamma0), uniform-Gamma limit
    hd_plateau_anal  = hd_diag_strong_plateau(gamma_matrix, ell_min, ell_max)

    # ── Numerical plateaus: max of each curve ──
    cp_plateau_num = float(np.max(rho_cp))
    hd_plateau_num = float(np.max(rho_hd))

    print('\n' + '='*55)
    print('          SNR CURVE DIAGNOSTICS')
    print('='*55)

    print('\n── Common Process (CP) ──')
    print(f'  Weak-signal slope   (x ~ 1e-12 to 1e-10):  {slope_cp_weak:+.3f}  (expect +1.0)')
    print(f'  Strong-signal slope (x ~ 1e-2  to 1e+1 ):  {slope_cp_strong:+.3f}  (expect ~0)')
    print(f'  Plateau  [pinv    ] = sqrt(1^T G^+ 1)/c   = {cp_plateau_anal:.4f}')
    print(f'  Plateau  [solve   ] = sqrt(1^T(GoG)^-1 1)/c = {rho_cp_signal_dominated(gamma_matrix, ell_min, ell_max):.4f}')
    print(f'  Plateau  [uniform ] = 1/(c*gamma0) (SM)   = {cp_plateau_unif:.4f}')
    print(f'  Plateau  [numeric ] = max(rho_CP)         = {cp_plateau_num:.4f}')

    print('\n── Hellings-Downs (HD) ──')
    print(f'  Weak-signal slope   (x ~ 1e-12 to 1e-10):  {slope_hd_weak:+.3f}  (expect +1.0)')
    print(f'  Strong-signal slope (x ~ 1e-2  to 1e+1 ):  {slope_hd_strong:+.3f}  (expect ~0)')
    print(f'  Plateau  [analytic] = diag-approx r->inf  = {hd_plateau_anal:.4f}  ({n_pairs} pairs)')
    print(f'  Plateau  [numeric ] = max(rho_HD)         = {hd_plateau_num:.4f}')
    print('='*55 + '\n')


# ============================================================
#                        MAIN
# ============================================================

def main():
    stars_deg        = build_star_positions(STAR_COORDS_DEG)
    theta            = pairwise_theta(stars_deg)
    ell_min, ell_max = compute_ell_limits(theta, FIELD_SIZE_DEG)
    print(f'ell_min = {ell_min},  ell_max = {ell_max}')

    gamma  = gamma_parallel_matrix(theta, ell_min, ell_max)
    gamma0 = cp_single_star_gamma(ell_min, ell_max)

    # Diagnostic quantities — printed for reference, NOT used to set x range
    factor     = gamma_scale_factor()
    rho_plat   = rho_cp_intermediate(gamma0)
    transition = 1.0 / (np.sqrt(N_STARS) * POL_FACTOR * factor * gamma0)  # sqrt(N) F r = 1/(c gamma0)

    print(f'\nNORMALIZED_GAMMA = {NORMALIZED_GAMMA}  (gamma_scale_factor = {factor:.4f})')
    print(f'POL_FACTOR       = {POL_FACTOR:g}  (multiplies P_ab)')
    print(f'\nINPUT PARAMETERS:')
    print(f'  sigma_rad        = {sigma_rad:.4e} rad')
    print(f'  sigma_bar^2      = {sigma_bar_sq:.4e}')
    print(f'  dt               = {dt_seconds:.1f} s')
    print(f'  T_obs            = {T_obs_seconds:.3e} s')
    print(f'  f_low            = {f_l:.3e} Hz')
    print(f'  P_n              = {P_n:.3e} rad^2/Hz')
    print(f'  P_gw(f_l)        = {P_gw_fl:.3e}')
    print(f'  P_gw(f_l) / P_n  = {PHYSICAL_RATIO:.3e}  <-- actual operating point')
    print(f'  N_stars          = {len(stars_deg)}')
    print(f'  gamma0           = {gamma0:.6f}')
    print(f'  CP plateau rho   = {rho_plat:.4f}  (= 1/(c*gamma0), uniform-Gamma limit only)')
    print(f'  CP transition r* = {transition:.4e}  (diagnostic only)')
    print(f'  HD plateau rho   ~ {hd_diag_strong_plateau(gamma, ell_min, ell_max):.2f}  (diagonal approx, r -> inf)')

    # ── Fixed sweep: covers the physical operating point AND both plateaus ──
    # PHYSICAL_RATIO ~ 6e-11 sits deep in the weak-signal regime.
    x_arr = np.logspace(-13, 2, 400)

    print('\nComputing full curves...')
    rho_cp    = rho_cp_full(x_arr, gamma, ell_min, ell_max)
    rho_cp_sm = rho_cp_full_sherman_morrison(x_arr, ell_min, ell_max, n_stars=len(stars_deg))
    rho_hd    = rho_hd_full(x_arr, gamma, ell_min, ell_max)
    print('Done.')

    print_snr_diagnostics(x_arr, rho_cp, rho_hd, ell_min, ell_max, gamma, n_stars=len(stars_deg))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.loglog(x_arr, rho_cp, color='C0', lw=2.5, label=r'$\rho_{\rm CP}$ (full inverse)')
    ax.loglog(x_arr, rho_cp_sm, color='C0', lw=1.2, ls=':', label=r'$\rho_{\rm CP}$ (uniform $\Gamma$, old)')
    ax.loglog(x_arr, rho_hd, color='C1', lw=2.5, label=r'$\rho_{\rm HD}$')

    # Mark the physical operating point — where the real signal sits on the curve
    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'$P_{{\rm gw}}(f_l)/P_n = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho$',                           fontsize=13)
    ax.set_ylim(1e-11, 1e3)
    ax.set_title('CP and HD SNR Full Curves',          fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    # Saved (rather than plt.show()) so this works on a Slurm
    # batch job with no display, and tagged with N/FoV so different runs
    # don't overwrite each other's plot. The same convention is used in
    # hd_full_matrix_snr.py's output naming.
    norm_tag = ("_normGamma" if NORMALIZED_GAMMA else "") + (f"_pol{POL_FACTOR:g}" if POL_FACTOR != 1 else "")
    out_name = f"main_cp_hd_case3_N{N_STARS}_FoV{FIELD_SIZE_DEG:g}{norm_tag}.png"
    plt.savefig(out_name, dpi=150)
    print(f"Saved plot to {out_name}")


# ============================================================
#                   STANDALONE CP PLOT (called by other scripts)
# ============================================================

def plot_full_snr(gamma_matrix, ell_min, ell_max):
    """
    CP full curve with asymptotic guide lines.
    Called by external scripts. Uses the same fixed sweep as main().
    """
    x_arr  = np.logspace(-13, 2, 600)
    gamma0 = cp_single_star_gamma(ell_min, ell_max)

    rho_full_cp  = rho_cp_full(x_arr, gamma_matrix, ell_min, ell_max)
    rho_plat     = rho_cp_strong_plateau(gamma_matrix, ell_min, ell_max)
    rho_weak_arr = rho_cp_weak(x_arr, n_stars=gamma_matrix.shape[0])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(x_arr, rho_weak_arr, lw=1.5, ls='--', color='#919191',
              alpha=0.7, label='Weak-signal asymptote')
    ax.axhline(rho_plat, lw=1.5, ls=':', color='#494949',
               alpha=0.7, label=f'Strong-signal plateau ({rho_plat:.1f})')
    ax.loglog(x_arr, rho_full_cp, lw=2.5, color='C0',
              label=r'$\rho_{\rm CP}$ full')
    ax.axvline(PHYSICAL_RATIO, color='k', lw=1.2, ls='--',
               label=rf'Physical $r = {PHYSICAL_RATIO:.1e}$')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n(f_l)$', fontsize=13)
    ax.set_ylabel(r'$\rho$',                           fontsize=13)
    ax.set_ylim(1e-2, 1e3)
    ax.set_title('Full SNR Curve: Common Process',     fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()