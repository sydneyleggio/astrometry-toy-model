import numpy as np
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

# Noise PSD from its physical equation: P_n = 2 * sigma_rad^2 * dt
P_n = 2.0 * sigma_rad**2 * dt_seconds

# Reference frequency: 1/year in Hz
f_yr = (1 / u.yr).to(u.Hz).value

# GW amplitude (NANOGrav value)
A_gw = 1e-15

# sigma_bar^2 = P_n, same as Romano — all stars have identical noise
sigma_bar_sq = P_n

# Field parameters
FIELD_SIZE_DEG = 10.0
N_STARS = 100
STAR_COORDS_DEG = None   # Set to an (N,2) array in degrees to use fixed positions
RANDOM_SEED = 1234

EPS = 1e-14


# ============================================================
#                      STAR POSITIONS
# ============================================================

def build_star_positions(star_coords_deg=None, n_stars=N_STARS,
                         field_size_deg=FIELD_SIZE_DEG, seed=RANDOM_SEED):
    if star_coords_deg is not None:
        stars = np.asarray(star_coords_deg, dtype=float)
        if stars.ndim != 2 or stars.shape[1] != 2:
            raise ValueError('star_coords_deg must have shape (N, 2).')
        return stars
    rng = np.random.default_rng(seed)
    half = field_size_deg / 2.0
    return rng.uniform(-half, half, size=(n_stars, 2))


def pairwise_theta(stars_deg):
    """Flat-sky pairwise angular separations in radians."""
    stars_rad = np.deg2rad(stars_deg)
    dx = stars_rad[:, 0][:, None] - stars_rad[:, 0][None, :]
    dy = stars_rad[:, 1][:, None] - stars_rad[:, 1][None, :]
    theta = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(theta, np.nan)
    return theta


def compute_ell_limits(theta_matrix, field_size_deg):
    """
    ell_min = 2 (per Kris suggestion)
    ell_max = 2*pi / min_angular_separation  (in radians)
    """
    ell_min = 2

    finite_seps = theta_matrix[np.isfinite(theta_matrix) & (theta_matrix > 0)]
    min_sep_rad = np.min(finite_seps)
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
            P1[l + 1] = ((2*l + 1) * mu * P1[l] - (l + 1) * P1[l - 1]) / l if l >= 2 else (2*l + 1) * mu * P1[l]
        if l >= 2:
            P2[l + 1] = ((2*l + 1) * mu * P2[l] - (l + 2) * P2[l - 1]) / (l - 1) if l >= 3 else (2*l + 1) * mu * P2[l]

    return P0, P1, P2


# ============================================================
#                         G KERNELS
# ============================================================

def G1(ell, P0_ell, P2_ell):
    """
    G_l^(1)(Theta) = -1/2 * [ P_l^2(cos Theta) / (l(l+1)) - P_l(cos Theta) ]
    """
    ll1 = ell * (ell + 1.0)
    return -0.5 * (P2_ell / ll1 - P0_ell)


def G2(ell, P1_ell, theta):
    """
    G_l^(2)(Theta) = -1 / (l(l+1)) * P_l^1(cos Theta) / sin(Theta)

    Singularities handled explicitly:
      theta -> 0:   limit = +0.5
      theta -> pi:  limit = 0
    """
    ll1 = ell * (ell + 1.0)
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

def gamma_parallel(theta, ell_min, ell_max):
    """
    Gamma_o^parallel(Theta) = sum_{l=ell_min}^{ell_max}
        (2l+1)/(4pi) * F_sq(l) * ( G1_l(Theta) + G2_l(Theta) )
    """
    theta = np.clip(np.asarray(theta, dtype=float), 0, np.pi)
    mu = np.cos(theta)

    P0, P1, P2 = compute_legendre_recurrence(mu, ell_max)

    total = np.zeros_like(theta)
    for ell in range(ell_min, ell_max + 1):
        g1 = G1(ell, P0[ell], P2[ell])
        g2 = G2(ell, P1[ell], theta)
        weight = (2.0 * ell + 1.0) / (4.0 * np.pi) * F_sq(ell)
        total += weight * (g1 + g2)

    return total


def sum_inverse_gamma_sq(gamma_matrix):
    """
    Sum of 1/Gamma^2 over unique star pairs only (i < j).

    HD-specific: the HD covariance matrix is N_pairs x N_pairs, one estimator
    per cross-correlated pair. This helper sums the diagonal inverse in the
    intermediate-signal limit and must NOT be used for CP.
    """
    upper = gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    vals = upper[np.isfinite(upper) & (np.abs(upper) > EPS)]
    return np.sum(1.0 / vals**2)


def cp_single_star_gamma(ell_min, ell_max):
    """
    Single-star CP overlap evaluated at zero separation (theta = 0).

    The CP estimator is one per star (auto-correlation), forming an N x N
    covariance matrix. The relevant overlap for the Sherman-Morrison reduction
    is Gamma_o(0): the overlap of a star with itself.
    """
    return float(gamma_parallel(np.array([0.0]), ell_min, ell_max)[0])


# GW power spectrum evaluated at f_l
P_gw_fl = (A_gw**2 / (12.0 * np.pi**2)) * (f_l / f_yr)**(-4.0/3.0) / f_l


# ============================================================
#     SNR CALCULATIONS
#
#  CONCEPTUAL STRUCTURE (Romano et al. 2020):
#
#  Common Process (CP):
#    - Estimator: A_hat^2_a, one per STAR (auto-correlation of star a's data)
#    - Covariance matrix: C_ab is N x N, indices over individual stars
#    - SNR: rho^2_CP = sum_{a,b} (C^{-1})_ab
#    - Sherman-Morrison inversion uses gamma0 = Gamma_o(0), the self-overlap
#
#  Hellings-and-Downs (HD):
#    - Estimator: A_hat^2_ab, one per PAIR (cross-correlation of stars a != b)
#    - Covariance matrix: C_{ab,cd} is N_pairs x N_pairs, indices over pairs
#    - Three cases depending on shared stars between pairs (ab) and (cd)
#    - Diagonal approximation: only Case 3 (same pair) terms retained
#    - SNR: rho^2_HD = sum_{a<b} 1/C_{ab,ab}
#    - Uses gamma_ab = Gamma_o(Theta_ab), the pairwise overlap
#
#  All gamma values are raw Gamma_o output (without 192*pi^3).
#  F = 192*pi^3 is applied explicitly in the SNR formulas.
#
#  Key algebraic note for CP: since sigma_bar^2 = P_n, dividing numerator
#  and denominator of Eq. 22 by (sigma_bar^2/F)^2 = (P_n/F)^2 yields a
#  formula that depends only on the dimensionless ratio r = P_gw/P_n:
#
#    rho^2_CP = N*(F*r)^2 / [1 + 2*F*gamma0*r + N*(F*gamma0*r)^2]
#
#  No explicit P_n factors remain. The x-axis variable r is used directly.
# ============================================================

def rho_cp_weak(x):
    """
    CP weak-signal regime SNR.

    In the weak limit (F*gamma0*r << 1), the denominator -> 1:
        rho^2_CP ~ N * (F*r)^2
        rho_CP   ~ sqrt(N) * F * r

    x = r = P_gw/P_n (dimensionless).
    """
    factor = 192.0 * np.pi**3
    rho_sq = N_STARS * (factor * x)**2
    return np.sqrt(np.maximum(rho_sq, 0.0))


def rho_cp_intermediate(gamma0):
    """
    CP intermediate-signal plateau.

    In the signal-dominated limit (N*(F*gamma0*r)^2 >> 1):
        rho^2_CP -> N*(F*r)^2 / (N*(F*gamma0*r)^2) = 1/gamma0^2
        rho_CP   -> 1/gamma0
    """
    return 1.0 / max(abs(gamma0), EPS)


# ============================================================
#                        MAIN
# ============================================================

def main():
    stars_deg = build_star_positions(STAR_COORDS_DEG)
    theta = pairwise_theta(stars_deg)

    ell_min, ell_max = compute_ell_limits(theta, FIELD_SIZE_DEG)
    print(f'ell_min = {ell_min},  ell_max = {ell_max}')

    gamma = gamma_parallel(theta, ell_min, ell_max)
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    sum_inv_sq = sum_inverse_gamma_sq(gamma)

    rho_plat = rho_cp_intermediate(gamma0)
    print(f'Plateau SNR (CP intermediate-signal limit) = {rho_plat:.4f}')   

    # Transition: rho_weak(r*) = rho_plat
    # sqrt(N)*F*r* = 1/gamma0  =>  r* = 1 / (sqrt(N)*F*gamma0)
    factor = 192.0 * np.pi**3
    transition = 1.0 / (np.sqrt(N_STARS) * factor * gamma0)
    x_min = max(transition / 1e3, 1e-8)
    x_max = max(transition * 1e3, 1e-4)
    x_weak = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    rho_weak_line = rho_cp_weak(x_weak)

    x_int = x_weak
    rho_int_line = np.full_like(x_int, rho_plat)

    physical_ratio = P_gw_fl / P_n

    print(f'\nINPUT PARAMETERS:')
    print(f'  sigma_rad        = {sigma_rad:.4e} rad')
    print(f'  sigma_bar^2      = {sigma_bar_sq:.4e}')
    print(f'  dt               = {dt_seconds:.1f} s')
    print(f'  T_obs            = {T_obs_seconds:.3e} s')
    print(f'  f_low            = {f_l:.3e} Hz')
    print(f'  P_n              = {P_n:.3e} rad^2/Hz')
    print(f'  P_gw(f_l)        = {P_gw_fl:.3e}')
    print(f'  P_gw(f_l) / P_n  = {physical_ratio:.3e}')
    print(f'  N_stars          = {len(stars_deg)}')
    print(f'  Sum(1/Gamma^2)   = {sum_inv_sq:.4e}')
    print(f'  Plateau rho      = {rho_plat:.4e}')
    print(f'  Transition r*    = {transition:.4e}')
    print(f'F^2: {F_sq(ell_max)}')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x_weak, rho_weak_line, linewidth=2,
              color='C0', label='Weak-signal regime')
    ax.loglog(x_int, rho_int_line, linewidth=2,
              color='C1', label='Intermediate regime')
    ax.set_xlabel(r'$P_{\rm gw}/P_n$')
    ax.set_ylabel(r'$\rho_{cp}$')
    ax.set_title('Low-Frequency Common Process SNR')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plot_full_snr(gamma, ell_min, ell_max, rho_plat, x_weak, rho_weak_line)


# ============================================================
#                   FULL SNR CURVES
# ============================================================

def rho_cp_full(x_arr, ell_min, ell_max, n_stars=N_STARS):
    """
    CP full SNR curve via Sherman-Morrison inversion of the N x N covariance.

    The CP estimators are auto-correlations: one per star, N total. The N x N
    data covariance (tilde) is:
        C_tilde_ab = F*P_gw*Gamma_o(Theta_ab) + sigma_bar^2 * delta_ab

    Sherman-Morrison inversion of the resulting estimator covariance yields
    (after normalizing by sigma_bar^4/F^2 and using sigma_bar^2 = P_n):

        rho^2_CP = N*(F*r)^2 / [1 + 2*F*gamma0*r + N*(F*gamma0*r)^2]

    where r = P_gw/P_n (the x-axis), F = 192*pi^3, gamma0 = Gamma_o(0).

    Denominator structure:
      - '1'               : noise-squared term (sigma^4/F^2), normalised away
      - '2*F*gamma0*r'    : signal-noise cross term; NO factor of N — this is a
                            single rank-1 correction in the Sherman-Morrison step
      - 'N*(F*gamma0*r)^2': signal-squared term; N appears because all N
                            auto-correlations contribute coherently
    """
    factor = 192.0 * np.pi**3
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    x_arr  = np.asarray(x_arr, dtype=float)   # r = P_gw/P_n, dimensionless

    Fr  = factor * x_arr                       # F * r
    Fgr = factor * gamma0 * x_arr              # F * gamma0 * r

    numer = n_stars * Fr**2
    denom = 1.0 + 2.0 * Fgr + n_stars * Fgr**2
    return np.sqrt(np.maximum(numer / denom, 0.0))


def rho_hd_full(x_arr, gamma_matrix):
    """
    HD full SNR curve using the diagonal (pair-by-pair) approximation of C^{-1}.

    The HD estimators are cross-correlations: one per unique pair (a,b) with
    a < b, giving N_pairs = N(N-1)/2 estimators. The full N_pairs x N_pairs
    covariance matrix is numerically unstable to invert (genuine negative
    eigenvalues), so only the diagonal Case 3 entries are used.

    For each pair, the SNR contribution is:
        rho^2_ab = P_gw^2 / [P_gw^2*gamma_ab^2
                              + 2*P_gw*gamma_ab*sigma_bar^2/F
                              + sigma_bar^4/F^2]

    where gamma_ab = Gamma_o(Theta_ab) (raw, without F).

    Unlike CP, this formula retains explicit P_gw = r * P_n factors because
    sigma_bar^2/F is not dimensionless relative to P_gw when gamma_ab varies
    across pairs. The algebra does not simplify to a pure function of r alone
    in the same way as CP.
    """
    factor   = 192.0 * np.pi**3
    vals     = gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    gammas   = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
    if gammas.size == 0:
        return np.zeros_like(np.asarray(x_arr, dtype=float))

    x_arr    = np.asarray(x_arr, dtype=float)
    P_gw_arr = x_arr[:, None] * P_n           # shape (n_r, 1)
    g        = gammas[None, :]                 # shape (1, n_pairs)
    Fg       = factor * g                      # tilde-Gamma_ab = F * gamma_ab

    numer = 2.0 * P_gw_arr**2 * Fg**2
    denom = P_gw_arr**2 * Fg**2 + (P_gw_arr + sigma_bar_sq)**2

    rho_sq = np.sum(numer / denom, axis=1)
    return np.sqrt(np.maximum(rho_sq, 0.0))


def plot_full_snr(gamma_matrix, ell_min, ell_max, rho_plat, x_weak, rho_weak_line):
    """CP full curve and asymptotic lines."""
    x_full = np.logspace(np.log10(max(x_weak.min() / 10.0, 1e-10)),
                         np.log10(x_weak.max() * 10.0), 600)

    print("Computing CP full curve...")
    rho_full_cp = rho_cp_full(x_full, ell_min, ell_max)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(x_weak, rho_weak_line, lw=1.5, ls='--', color="#919191",
              alpha=0.7, label='Weak signal regime (CP)')
    ax.axhline(rho_plat, lw=1.5, ls='--', color="#494949",
               alpha=0.7, label='Intermediate regime (CP)')
    ax.loglog(x_full, rho_full_cp, lw=2.5, color='C0', label=r'$\rho_{CP}$ full')

    ax.set_xlabel(r'$P_{\rm gw}(f_l)\,/\,P_n$')
    ax.set_ylabel(r'$\rho$')
    ax.set_title('CP SNR Full Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()