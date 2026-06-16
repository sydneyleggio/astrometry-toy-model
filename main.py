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

# Noise PSD: P_n = 2 * sigma_rad^2 * dt
P_n = 2.0 * sigma_rad**2 * dt_seconds

# Reference frequency: 1/year in Hz
f_yr = (1 / u.yr).to(u.Hz).value

# GW amplitude (NANOGrav value)
A_gw = 1e-15

# sigma_bar^2 = P_n — all stars have identical noise
sigma_bar_sq = P_n

# Field parameters
FIELD_SIZE_DEG = 100
N_STARS        = 100
STAR_COORDS_DEG = None
RANDOM_SEED     = 1234

EPS = 1e-14

# Physical GW power spectrum at f_l
# P_gw(f) = A_gw^2 / (12*pi^2) * (f/f_yr)^(-4/3) * f^(-1),  alpha = -2/3
P_gw_fl = (A_gw**2 / (12.0 * np.pi**2)) * (f_l / f_yr)**(-4.0/3.0) / f_l

# Physical operating point on the x-axis — used for vertical marker on plots
# This is the actual value of P_gw/P_n with all constants plugged in
PHYSICAL_RATIO = P_gw_fl / P_n


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

def gamma_parallel(theta, ell_min, ell_max):
    """
    Gamma_o^parallel(Theta) = sum_{l=ell_min}^{ell_max}
        (2l+1)/(4pi) * F_sq(l) * (G1_l(Theta) + G2_l(Theta))
    """
    theta = np.clip(np.asarray(theta, dtype=float), 0, np.pi)
    mu    = np.cos(theta)

    P0, P1, P2 = compute_legendre_recurrence(mu, ell_max)

    total = np.zeros_like(theta)
    for ell in range(ell_min, ell_max + 1):
        g1     = G1(ell, P0[ell], P2[ell])
        g2     = G2(ell, P1[ell], theta)
        weight = (2.0 * ell + 1.0) / (4.0 * np.pi) * F_sq(ell)
        total += weight * (g1 + g2)

    return total


def cp_single_star_gamma(ell_min, ell_max):
    """
    Single-star CP overlap at zero separation: Gamma_o(0).
    Used in the Sherman-Morrison reduction of the N x N CP covariance.
    Independent of N and of the star field layout.
    """
    return float(gamma_parallel(np.array([0.0]), ell_min, ell_max)[0])


def gamma_parallel_matrix(theta_matrix, ell_min, ell_max, batch_size=5000):
    """
    Compute the full N x N gamma_parallel matrix memory-safely by processing
    unique pairs in batches rather than passing the full N x N array to
    compute_legendre_recurrence at once.

    Returns the full symmetric N x N gamma matrix (NaN on diagonal,
    consistent with the output of gamma_parallel called on the full matrix).
    """
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
#    - Sherman-Morrison inversion uses gamma0 = Gamma_o(0).
#
#  Hellings-and-Downs (HD):
#    - Estimator: one per PAIR (a<b) (cross-correlation); covariance is
#      N_pairs x N_pairs.
# ============================================================

def rho_cp_weak(x, n_stars=N_STARS):
    """
    CP weak-signal asymptote: rho ~ sqrt(N)*F*r.
    x = r = P_gw/P_n (dimensionless).
    Takes n_stars as argument so calling scripts can pass N explicitly.
    """
    factor = 192.0 * np.pi**3
    return np.sqrt(n_stars) * factor * np.asarray(x, dtype=float)


def rho_cp_intermediate(gamma0):
    """
    CP intermediate-signal plateau: rho -> 1/gamma0.
    Independent of N — gamma0 = Gamma_o(0) is a pure geometric quantity.
    """
    return 1.0 / max(abs(gamma0), EPS)


def rho_cp_full(x_arr, ell_min, ell_max, n_stars=N_STARS):
    """
    CP full SNR curve via Sherman-Morrison inversion of the N x N covariance.

    rho^2_CP = N*(F*r)^2 / [1 + 2*F*gamma0*r + N*(F*gamma0*r)^2]

    where r = P_gw/P_n, F = 192*pi^3, gamma0 = Gamma_o(0).

    Denominator:
      '1'               : noise-squared, normalised away
      '2*F*gamma0*r'    : signal-noise cross term, NO factor of N
      'N*(F*gamma0*r)^2': signal-squared, N from coherent sum of auto-correlations
    """
    factor = 192.0 * np.pi**3
    gamma0 = cp_single_star_gamma(ell_min, ell_max)
    x_arr  = np.asarray(x_arr, dtype=float)

    Fr  = factor * x_arr
    Fgr = factor * gamma0 * x_arr

    numer = n_stars * Fr**2
    denom = 1.0 + 2.0 * Fgr + n_stars * Fgr**2

    return np.sqrt(np.maximum(numer / denom, 0.0))


def rho_hd_full(x_arr, gamma_matrix):
    """
    HD full SNR via diagonal (Case 3) approximation of the N_pairs x N_pairs
    covariance matrix.

    From Romano eq 37, inverting C_{ab,ab} and summing:
        rho^2_HD = sum_{a<b} 2*Pgw^2*(F*g_ab)^2
                              / [(Pgw*F*g_ab)^2 + (Pgw + sigma^2)^2]

    where g_ab = Gamma_o(Theta_ab) (raw, without F), F = 192*pi^3,
    and Pgw = r * P_n with r = P_gw/P_n the x-axis variable.

    Asymptotes:
      Weak  (Pgw << sigma^2): rho^2 -> 2*F^2*sum(g^2)*r^2
      Strong (Pgw >> sigma^2, F*g >> 1): rho^2 -> 2*N_pairs = N*(N-1)
    """
    factor = 192.0 * np.pi**3
    vals   = gamma_matrix[np.triu_indices_from(gamma_matrix, k=1)]
    gammas = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
    if gammas.size == 0:
        return np.zeros_like(np.asarray(x_arr, dtype=float))

    x_arr    = np.asarray(x_arr, dtype=float)
    P_gw_arr = x_arr[:, None] * P_n      # shape (n_r, 1)
    g        = gammas[None, :]            # shape (1, n_pairs)
    Fg       = factor * g                 # tilde-Gamma_ab = F * gamma_ab

    numer = 2.0 * P_gw_arr**2 * Fg**2
    denom = (P_gw_arr * Fg)**2 + (P_gw_arr + sigma_bar_sq)**2

    rho_sq = np.sum(numer / denom, axis=1)
    return np.sqrt(np.maximum(rho_sq, 0.0))

def print_snr_diagnostics(x_arr, rho_cp, rho_hd, ell_min, ell_max, n_stars=N_STARS):
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
    cp_plateau_anal  = 1.0 / max(abs(gamma0), EPS)           # = 1/gamma0
    hd_plateau_anal  = np.sqrt(n_stars * (n_stars - 1))      # = sqrt(N*(N-1))

    # ── Numerical plateaus: max of each curve ──
    cp_plateau_num = float(np.max(rho_cp))
    hd_plateau_num = float(np.max(rho_hd))

    print('\n' + '='*55)
    print('          SNR CURVE DIAGNOSTICS')
    print('='*55)

    print('\n── Common Process (CP) ──')
    print(f'  Weak-signal slope   (x ~ 1e-12 to 1e-10):  {slope_cp_weak:+.3f}  (expect +1.0)')
    print(f'  Strong-signal slope (x ~ 1e-2  to 1e+1 ):  {slope_cp_strong:+.3f}  (expect ~0)')
    print(f'  Plateau  [analytic] = 1/gamma0            = {cp_plateau_anal:.4f}')
    print(f'  Plateau  [numeric ] = max(rho_CP)         = {cp_plateau_num:.4f}')

    print('\n── Hellings-Downs (HD) ──')
    print(f'  Weak-signal slope   (x ~ 1e-12 to 1e-10):  {slope_hd_weak:+.3f}  (expect +1.0)')
    print(f'  Strong-signal slope (x ~ 1e-2  to 1e+1 ):  {slope_hd_strong:+.3f}  (expect ~0)')
    print(f'  Plateau  [analytic] = sqrt(N*(N-1))       = {hd_plateau_anal:.4f}  ({n_pairs} pairs)')
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

    gamma  = gamma_parallel(theta, ell_min, ell_max)
    gamma0 = cp_single_star_gamma(ell_min, ell_max)

    # Diagnostic quantities — printed for reference, NOT used to set x range
    factor     = 192.0 * np.pi**3
    rho_plat   = rho_cp_intermediate(gamma0)
    transition = 1.0 / (np.sqrt(N_STARS) * factor * gamma0)

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
    print(f'  CP plateau rho   = {rho_plat:.4f}  (= 1/gamma0, independent of N)')
    print(f'  CP transition r* = {transition:.4e}  (diagnostic only)')
    print(f'  HD plateau rho   ~ {np.sqrt(N_STARS*(N_STARS-1)):.2f}  (= sqrt(N*(N-1)), diagonal approx)')

    # ── Fixed sweep: covers the physical operating point AND both plateaus ──
    # PHYSICAL_RATIO ~ 6e-11 sits deep in the weak-signal regime.
    x_arr = np.logspace(-13, 2, 400)

    print('\nComputing full curves...')
    rho_cp = rho_cp_full(x_arr, ell_min, ell_max)
    rho_hd = rho_hd_full(x_arr, gamma)
    print('Done.')

    rho_cp = rho_cp_full(x_arr, ell_min, ell_max)
    rho_hd = rho_hd_full(x_arr, gamma)
    print_snr_diagnostics(x_arr, rho_cp, rho_hd, ell_min, ell_max, n_stars=len(stars_deg))  # add this
    print('Done.')

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.loglog(x_arr, rho_cp, color='C0', lw=2.5, label=r'$\rho_{\rm CP}$')
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
    plt.show()


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

    rho_full_cp  = rho_cp_full(x_arr, ell_min, ell_max)
    rho_plat     = rho_cp_intermediate(gamma0)
    rho_weak_arr = rho_cp_weak(x_arr)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(x_arr, rho_weak_arr, lw=1.5, ls='--', color='#919191',
              alpha=0.7, label='Weak-signal asymptote')
    ax.axhline(rho_plat, lw=1.5, ls=':', color='#494949',
               alpha=0.7, label=f'Intermediate plateau ({rho_plat:.1f})')
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