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
    ell_min = 1 / field_of_view  (in radians)
    ell_max = 2*pi / min_angular_separation  (in radians)
    Both rounded to nearest integer, with ell_min >= 2.
    """
    #field_rad = np.deg2rad(field_size_deg)
    #ell_min = max(2, int(np.ceil(1.0 / field_rad)))
    ell_min = 2

    finite_seps = theta_matrix[np.isfinite(theta_matrix) & (theta_matrix > 0)]
    min_sep_rad = np.min(finite_seps)
    ell_max = int(np.floor(2.0 * np.pi / min_sep_rad))

    return ell_min, ell_max


# ============================================================
#               VECTORIZED LEGENDRE RECURRENCE
# ============================================================

#need to look into this section more
def compute_legendre_recurrence(mu, ell_max):
    """
    Compute P_l(mu), P_l^1(mu), P_l^2(mu) for all l up to ell_max
    using the standard 3-term recurrence relation — avoids repeated
    lpmv calls and is orders of magnitude faster for large ell_max.

    Returns arrays of shape (ell_max+1, *mu.shape).
    """
    shape = mu.shape
    P0 = np.zeros((ell_max + 1,) + shape)   # P_l^0
    P1 = np.zeros((ell_max + 1,) + shape)   # P_l^1
    P2 = np.zeros((ell_max + 1,) + shape)   # P_l^2

    sin_t = np.sqrt(np.maximum(1.0 - mu**2, 0.0))

    # Seed values
    P0[0] = 1.0
    P0[1] = mu
    P1[1] = -sin_t
    P2[2] = 3.0 * sin_t**2   # P_2^2 = 3 sin^2(theta)

    # Recurrence: (l-m+1) P_{l+1}^m = (2l+1) mu P_l^m - (l+m) P_{l-1}^m
    for l in range(1, ell_max):
        # m=0
        P0[l + 1] = ((2*l + 1) * mu * P0[l] - l * P0[l - 1]) / (l + 1)
        # m=1
        if l >= 1:
            P1[l + 1] = ((2*l + 1) * mu * P1[l] - (l + 1) * P1[l - 1]) / l if l >= 2 else (2*l + 1) * mu * P1[l]
        # m=2
        if l >= 2:
            P2[l + 1] = ((2*l + 1) * mu * P2[l] - (l + 2) * P2[l - 1]) / (l - 1) if l >= 3 else (2*l + 1) * mu * P2[l]

    return P0, P1, P2

# ============================================================
#                         G KERNELS
# ============================================================
def G1(ell, P0_ell, P2_ell):
    """
    G_l^(1)(Theta) = -1/2 * [ P_l^2(cos Theta) / (l(l+1)) - P_l(cos Theta) ]
 
    Takes precomputed Legendre arrays P0[ell] and P2[ell] for efficiency.
    """
    ll1 = ell * (ell + 1.0)
    return -0.5 * (P2_ell / ll1 - P0_ell)
 
 
def G2(ell, P1_ell, theta):
    """
    G_l^(2)(Theta) = -1 / (l(l+1)) * P_l^1(cos Theta) / sin(Theta)

    Singularities handled explicitly:
      theta -> 0:   limit = +0.5  for all ell
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
# MODE COUPLING COEFFICIENT
# ============================================================

def F_sq(ell):
    """
    |F_l^E|^2 = |F_l^B|^2 = 1 / (N_l^2 * l(l+1))
    N_l^2 = (l+2)(l+1)l(l-1) / 2
    Cross terms are zero.
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
 
    G1 and G2 are computed explicitly from the precomputed Legendre arrays.
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
 

# def sum_inverse_gamma_sq(gamma_matrix):
#     """
#     Sum of 1/Gamma^2 over all unique star pairs i < j.
#     From the estimator covariance derivation: C_ab ~ Gamma^2 in the
#     intermediate regime, so C^{-1} sums 1/Gamma^2.
#     """
#     iu = np.triu_indices_from(gamma_matrix, k=1)
#     vals = gamma_matrix[iu]
#     vals = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
#     return np.sum(1.0 / vals**2)

def sum_inverse_gamma_sq(gamma_matrix):
    """
    Sum of 1/Gamma^2 over all star pairs i, j (including i == j and both i<j and i>j).
    From the estimator covariance derivation: C_ab ~ Gamma^2 in the
    intermediate regime, so C^{-1} sums 1/Gamma^2.
    """
    vals = gamma_matrix.flatten()
    vals = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]
    return np.sum(1.0 / vals**2)


# GW power spectrum evaluated at f_l — physical equation, fixed value
# P_gw(f) = A_gw^2 / (12 pi^2) * (f/f_yr)^(2*alpha) * f^(-1),  alpha = -2/3
P_gw_fl = (A_gw**2 / (12.0 * np.pi**2)) * (f_l / f_yr)**(-4.0/3.0) / f_l


# ============================================================
#     SNR CALCULATIONS — from verified estimator covariance derivation
#
#     Estimator: A_hat^2 = [P_hat(f_l) - sigma_bar^2] * f_l^(7/3) / (192 pi^3)
#     Estimator covariance: C_ab = f_l^(14/3) * |C_tilde_ab|^2 / (192 pi^3)^2
#
#     Weak regime (f_l^(14/3) cancels):
#       rho^2 = N * (192 pi^3)^2 * P_gw^2(f_l) / (sigma_bar^2)^2
#
#     Intermediate regime (f_l^(14/3) and P_gw^2 both cancel):
#       rho^2 = sum_{a!=b} 1 / Gamma^2(Theta_ab)
# ============================================================

def rho_cp_weak(x):
    """
    Weak-signal regime SNR from the estimator covariance derivation:
        rho^2 = N * (192 pi^3)^2 * P_gw^2 / (sigma_bar^2)^2

    x = P_gw/P_n is the x-axis variable, so P_gw = x * P_n.
    P_n and sigma_bar^2 are independent physical constants.
    """
    rho_sq = N_STARS * (192.0 * np.pi**3)**2 * (x * P_n)**2 / sigma_bar_sq**2
    return np.sqrt(np.maximum(rho_sq, 0.0))


def rho_cp_intermediate(sum_inv_gamma_sq_val):
    """
    Intermediate-signal regime SNR from the estimator covariance derivation (Section 5.2):
        rho^2 = sum_{a!=b} 1 / Gamma^2(Theta_ab)

    This is a fixed number determined entirely by the star geometry
    via the gamma overlap function. It does not depend on P_gw or f.
    """
    return np.sqrt(max(sum_inv_gamma_sq_val, 0.0))


# ============================================================
#                        MAIN
# ============================================================

def main():
    stars_deg = build_star_positions(STAR_COORDS_DEG)
    theta = pairwise_theta(stars_deg)

    ell_min, ell_max = compute_ell_limits(theta, FIELD_SIZE_DEG)
    print(f'ell_min = {ell_min},  ell_max = {ell_max}')

    gamma = gamma_parallel(theta, ell_min, ell_max)
    sum_inv_sq = sum_inverse_gamma_sq(gamma)

    # Two analytic results — one per regime
    rho_plat = rho_cp_intermediate(sum_inv_sq)

    # Transition point: solve rho_weak(x*) = rho_plat for x*
    # sqrt(N) * (192pi^3) * x* = rho_plat  =>  x* = rho_plat / (sqrt(N) * 192pi^3)
    # (since sigma_bar^2 = P_n, the P_n cancels)
    #transition = rho_plat / (np.sqrt(N_STARS) * (192.0 * np.pi**3))
    transition = 1.e+5 
    # Weak branch: x below the transition, rho rises toward the plateau
    x_weak = np.logspace(-4, np.log10(transition), 300)
    rho_weak_line = rho_cp_weak(x_weak)

    #x_int = np.logspace(np.log10(transition), np.log10(transition) + 2, 300)
    x_int = x_weak
    rho_int_line = np.full_like(x_int, rho_plat)

    # Physical ratio from actual equations
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
    #print(f'  Transition ratio = {transition:.4e}')

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
#                   FULL SNR CURVE
# ============================================================

def rho_cp_full(x_arr, gamma_matrix, ell_min, ell_max):
    factor = 192.0 * np.pi**3

    # Off-diagonal pairs
    vals = gamma_matrix.flatten()
    gammas = vals[np.isfinite(vals) & (np.abs(vals) > EPS)]

    P_gw_arr = x_arr[:, None] * P_n
    g = gammas[None, :]

    numer = P_gw_arr**2
    denom = (P_gw_arr**2 * g**2
             + 2.0 * P_gw_arr * g * sigma_bar_sq / factor
             + sigma_bar_sq**2 / factor**2)

    rho_sq = np.sum(numer / denom, axis=1)

    # Diagonal contribution: theta=0, delta_ab=1
    gamma_diag = gamma_parallel(np.array([0.0]), ell_min, ell_max)[0]
    g_d = gamma_diag

    numer_diag = (P_gw_arr**2).squeeze()
    denom_diag = (P_gw_arr.squeeze()**2 * g_d**2
                  + 2.0 * P_gw_arr.squeeze() * g_d * sigma_bar_sq / factor
                  + sigma_bar_sq**2 / factor**2)

    rho_sq += N_STARS * numer_diag / denom_diag

    return np.sqrt(np.maximum(rho_sq, 0.0))


def plot_full_snr(gamma_matrix, ell_min, ell_max, rho_plat, x_weak, rho_weak_line):
    """
    Plot rho_cp full curve together with the two
    approximate regime lines already computed in main().
    """
    x_full = np.logspace(-4, 8, 600)
    rho_full = rho_cp_full(x_full, gamma_matrix, ell_min, ell_max)


    fig, ax = plt.subplots(figsize=(8, 5))

    # Approximate regime lines (from main) for comparison
    ax.loglog(x_weak, rho_weak_line, linewidth=1.5, linestyle='--',
              color="#919191", alpha=0.7, label='Weak signal regime')
    ax.axhline(rho_plat, linewidth=1.5, linestyle='--',
               color="#494949", alpha=0.7, label='Intermediate signal regime')

    # Full curve
    ax.loglog(x_full, rho_full, linewidth=2.5,
              color='C2', label='Full curve')

    ax.set_xlabel(r'$P_{\rm gw}/P_n$')
    ax.set_ylabel(r'$\rho_{cp}$')
    ax.set_title('Full SNR Curve with Regime Approximations')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()